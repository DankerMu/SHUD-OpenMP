# SHUD 单线程预优化：为高效率、高精度并行改造奠定基础

> 目标：在正式 OpenMP / CVODE 并行改造前，先把 SHUD 的单线程版本整理成一个**可验证、可拆分、可复现、可并行调度**的计算后端。  
> 核心结论：**应该先做单线程预优化**。这不是“多跑快一点”的小修小补，而是并行前的架构整备。没有这一步，后续并行很容易出现三类问题：速度收益不稳定、结果不可信、误差来源不可定位。

---

## 1. 基本判断

可以，而且应该。  
并行前先优化单线程 SHUD，主要有两个目的：

1. **为并行效率打基础**  
   并行效率不只取决于线程数，还取决于数据是否连续、任务是否可拆、热点是否明确、共享写入是否被消除、I/O 是否已经不是瓶颈。否则并行以后，线程可能大部分时间在等待、抢锁、重复计算或读文件。

2. **为并行精度打基础**  
   并行会改变执行顺序，特别是浮点加法顺序。为了判断并行结果是否“准确”，必须先把单线程路径整理成稳定的 reference。否则结果差异到底来自并行 bug、旧 OpenMP 路径与串行路径不一致、浮点顺序变化、I/O 变化还是物理过程变化，将无法区分。

因此，建议在加速路线中新增一个明确阶段：

```text
单线程预优化 / parallel-ready refactor
```

它位于：

```text
现有单线程 base
    ↓
单线程预优化：对齐、拆分、去副作用、缓存、诊断、回归
    ↓
严格等价 RHS 并行
    ↓
确定性容差生产并行
```

---

## 2. 为什么 SHUD 必须先做这一步

### 2.1 当前 OpenMP 路径不是“单线程路径的简单并行版”

当前 `f.cpp` 中，在 `_OPENMP_ON` 下调用的是：

```cpp
MD->f_update_omp(Y, DY, t);
MD->f_loop_omp(Y, DY, t);
MD->f_applyDY_omp(DY, t);
```

而非 OpenMP 下调用的是：

```cpp
MD->f_update(Y, DY, t);
MD->f_loop(t);
MD->f_applyDY(DY, t);
```

也就是说，当前串行与 OpenMP 是两套 RHS 路径，而不是“同一套 kernel 的单线程 / 多线程执行模式”。

这会导致一个根本问题：  
**只要两条路径的方程、状态更新或通量汇总稍有差别，就不能把并行结果差异归因于浮点顺序或线程调度。**

### 2.2 当前串行路径与 OpenMP 路径存在需要先对齐的差异

以目前公开源码看，串行 `f_loop()` 中包含 lake element 分支、`f_etFlux()`、lake vertical / horizontal 过程和 lake evaporation / precipitation 汇总；而 `f_loop_omp()` 的结构主要是 element infiltration / recharge、element lateral flow、segment river exchange、river downflow，并未保持完全同构。

`f_applyDY()` 与 `f_applyDY_omp()` 的 river DY 计算也不同：串行路径中 river DY 先按 river length 处理，再限制负向面积变化，随后通过 `fun_dAtodY()` 从面积变化转换为水深变化；OpenMP 路径中则直接除以 `Riv[i].u_TopArea`。

这说明并行前的第一类工作不是“加线程”，而是：

```text
先让单线程和将来的并行线程调用同一套计算 kernel
```

否则并行验证没有稳定基准。

### 2.3 当前部分函数同时“计算 flux”和“累加 flux”，不适合并行

例如 `fun_Seg_surface()` 和 `fun_Seg_sub()` 当前不仅计算：

```cpp
QsegSurf[i]
QsegSub[i]
```

还直接累加：

```cpp
QrivSurf[iRiv] += QsegSurf[i];
Qe2r_Surf[iEle] += -QsegSurf[i];

QrivSub[iRiv] += QsegSub[i];
Qe2r_Sub[iEle] += -QsegSub[i];
```

这类写法在单线程中没问题，但在并行中会带来两个问题：

1. 多个线程可能同时更新同一个 river 或 element accumulator；
2. 即使使用 `atomic` 避免 data race，浮点加法顺序仍会随线程到达顺序变化，结果不一定等于单线程 base。

因此，单线程阶段就应该把这类函数改成：

```text
只计算 flux，不做共享累加
```

汇总统一交给后续 deterministic gather。

### 2.4 当前 forcing 读取方式会成为并行效率瓶颈

`TimeSeriesData::read_csv()` 当前每次 refill 会重新打开文件，并跳过 `MAXQUE * nQue + 2` 行，再读下一段 queue；`getX()` 返回当前行的值。这种方式在单线程下还能接受，但并行后如果 RHS 调用次数很高，I/O 与缓存局部性会更容易成为瓶颈。

并行前应该先把 forcing 读取改成：

```text
常驻文件流 / 大块缓存 / 全量预加载 / 时间索引缓存
```

但第一阶段必须保持同样的取值语义，例如仍然保留当前 zero-order hold 行为；不要同时引入插值，否则结果变化会和并行重构混在一起。

### 2.5 当前求解控制信息还不足以支撑并行验证

当前 `Control_Data` 中有标量 `abstol`、`reltol`、`InitStep`、`MaxStep`、`SolverStep` 等控制项，但并行验证还需要更完整的诊断信息：

- RHS 调用次数；
- CVODE 内部步数；
- 误差测试失败次数；
- 非线性迭代次数；
- 线性求解迭代次数；
- 每个 RHS 子阶段耗时；
- I/O 耗时；
- gather 耗时；
- 输出耗时。

如果没有这些诊断，并行后只能看到“总时间变了、结果变了”，无法定位瓶颈和误差来源。

---

## 3. 单线程预优化的定位：不是改物理，而是重建计算 contract

这一阶段不能随意改变物理过程，也不能改变数值求解策略。它的定位应是：

```text
保持当前模型语义不变，
把当前单线程实现改造成并行友好的 reference implementation。
```

### 3.1 这一阶段应该做什么

应该做：

- 路径对齐；
- 函数拆分；
- 去共享副作用；
- 固定汇总顺序；
- 预构建拓扑邻接表；
- forcing 缓存；
- 单线程性能画像；
- 回归测试；
- RHS 级别 bitwise 对比；
- 完整 run 级别 bitwise 对比；
- 编译选项固化；
- 输出诊断增强。

### 3.2 这一阶段不应该做什么

暂时不应该做：

- 改变物理过程；
- 改变雨雪分相、融雪、ET、入渗公式；
- 改变 forcing 插值方式；
- 改变求和算法，例如直接切换 Kahan 或 pairwise summation；
- 改变 CVODE 容差策略；
- 启用 `-ffast-math`；
- 允许编译器重排浮点表达式；
- 把 CVODE `N_Vector` 同时换成 OpenMP 版本；
- 引入普通 OpenMP floating reduction 作为 strict 路径。

这些可以作为后续“新参考版本”或“production deterministic mode”评估，但不应混入单线程预优化的第一轮。

---

## 4. 建议实施路线

### 阶段 P0：锁定单线程 base 与性能画像

**目标**  
建立当前 SHUD 单线程版本的速度、精度和结果基准。

**主要工作**

1. 固定编译器、依赖库、编译选项、CPU 环境；
2. 选定 2–3 个代表性算例：
   - 小流域快速回归算例；
   - 中等流域性能算例；
   - 含 river / lake / cryosphere 开关的复杂算例；
3. 记录完整输出文件；
4. 记录 RHS 中间量；
5. 记录性能指标：
   - wall-clock；
   - `nFCall`；
   - CVODE 内部步数；
   - 每个 RHS 子阶段耗时；
   - I/O 耗时；
   - 输出耗时；
   - peak memory。

**为什么先做**  
后续所有重构都需要回答一个问题：  
**是否仍然等价于当前单线程 base？**  
如果 base 没有锁住，任何优化都不可验证。

**适合 SHUD 吗**  
非常适合。SHUD 是 CVODE 驱动的 RHS 密集型模型，RHS 调用次数和每次 RHS 计算成本都会直接影响整体运行时间。

**风险**

- 算例过少导致基线代表性不足；
- 只看总时间，不看 RHS / I/O / CVODE 分项，会误判瓶颈。

**验收标准**

- 当前单线程输出归档；
- 有至少一个小算例可用于每日回归；
- 有中等算例用于性能回归；
- 有 RHS 级别中间量 dump；
- 性能报告可重复。

---

### 阶段 P1：统一 RHS 路径，消除串行 / OpenMP 双实现漂移

**目标**  
把当前串行和 OpenMP 的两套 RHS 逻辑收敛为同一套 kernel。

**主要工作**

1. 将 `f_update()` / `f_update_omp()` 拆成共享 kernel；
2. 将 `f_loop()` / `f_loop_omp()` 拆成共享 kernel；
3. 将 `f_applyDY()` / `f_applyDY_omp()` 拆成共享 kernel；
4. 先保留单线程执行；
5. 对 lake、ET、river DY、boundary/source/sink、state update 等路径逐项对齐；
6. 用编译选项或运行参数控制是否启用并行，而不是复制一套方程。

建议结构：

```cpp
int f(...) {
    read_cvode_vector(...);
    rhs_update(...);
    rhs_compute_flux(...);
    rhs_gather(...);
    rhs_apply_dy(...);
}
```

其中每个阶段先都是单线程。

**为什么这样做**  
并行的第一前提是：  
**单线程和并行线程执行的是同一套模型。**  
否则加速比和结果差异都没有解释力。

**适合 SHUD 吗**  
非常适合。SHUD 的 RHS 结构天然可以拆成：

```text
state update
    ↓
local vertical process
    ↓
lateral flux
    ↓
river/lake flux
    ↓
gather
    ↓
DY assembly
```

这正是并行调度所需的结构。

**风险**

- 重构过程中可能改变执行顺序；
- lake / river / element 索引宏可能隐藏副作用；
- 原有 `_omp` 路径中的旧差异可能被误认为优化。

**验收标准**

- `serial_legacy` 与 `serial_unified` 的 RHS 输出 bitwise identical；
- 完整 run 输出 bitwise identical；
- `nFCall`、CVODE 内部步数、输出时刻完全一致；
- OpenMP 版本暂时不追求速度，只验证路径一致。

---

### 阶段 P2：拆分“flux 计算”和“flux 汇总”

**目标**  
消除 hot functions 中的共享累加副作用，为后续 owner-computes 并行做准备。

**主要工作**

把这类函数：

```cpp
fun_Seg_surface(...)
fun_Seg_sub(...)
```

从：

```text
计算 segment flux + 直接累加 river/element
```

改成：

```text
只计算 segment flux
```

例如：

```cpp
QsegSurf[i] = compute_seg_surface(...);
QsegSub[i]  = compute_seg_sub(...);
```

然后单独执行：

```cpp
gather_segment_to_river_and_element();
```

`PassValue()` 也要重构为显式 gather 阶段。

**为什么这样做**  
并行中最危险的不是 local 计算，而是多个贡献项累加到同一个 accumulator。提前在单线程中拆开，可以保证后续并行时不需要 `atomic +=`。

**适合 SHUD 吗**  
非常适合。SHUD 的 segment → river / element、upstream river → downstream river、lake element → lake 本质上都是图上的 gather 操作，天然适合 owner-local gather。

**风险**

- 如果 gather 的累加顺序和旧单线程不同，可能产生 bitwise 差异；
- 如果拆分时重复或遗漏某个 flux，会破坏水量守恒；
- gather 初始化顺序必须严格定义。

**验收标准**

- `QsegSurf/QsegSub` 与旧版本一致；
- `QrivSurf/QrivSub/Qe2r_Surf/Qe2r_Sub/QrivUp` 与旧版本一致；
- RHS `DY` bitwise identical；
- 完整 run bitwise identical；
- 水量平衡误差不变。

---

### 阶段 P3：构建固定拓扑与 owner-local gather 列表

**目标**  
把所有“谁给谁贡献 flux”的关系预计算成固定顺序的邻接表。

**主要工作**

构建并固定排序以下列表：

1. `river -> segments`  
   每个 river 拥有哪些 river segment，按原始 `segment id` 升序。

2. `element -> segments`  
   每个 element 与哪些 river segment 相连，按原始 `segment id` 升序。

3. `downstream river -> upstream rivers`  
   每个下游 river 的上游贡献 river 列表，按原始 `river id` 升序。

4. `lake -> lake elements`  
   每个 lake 包含哪些 element，按原始 `element id` 升序。

5. `element -> neighbor edges`  
   element 与相邻 element 的通量边，按固定 `j = 0..2` 或 edge id 顺序。

**为什么这样做**  
并行要准确，必须满足：

```text
每个 accumulator 只有一个 owner 写入；
每个 owner 内部按固定顺序累加。
```

拓扑列表是这个模式的基础。

**适合 SHUD 吗**  
非常适合。SHUD 是非结构三角网 + river network + lake coupling，本质上是稀疏图计算。预构建拓扑比在 RHS 中反复查找和临时判断更高效。

**风险**

- 拓扑列表构建错误会造成隐性水量偏差；
- 排序规则如果和旧单线程贡献顺序不同，会破坏 bitwise；
- lake / river 边界条件需要单独处理。

**验收标准**

- 使用拓扑列表后的 gather 与旧 `PassValue()` bitwise identical；
- 每个 flux contribution 可追踪；
- 所有 accumulator 有唯一 owner；
- 能打印拓扑 checksum，用于回归。

---

### 阶段 P4：整理内存布局与 hot data access

**目标**  
减少单线程 cache miss、为多线程降低 false sharing 与内存带宽瓶颈。

**主要工作**

1. 把 RHS 热点数组集中管理；
2. 对 element / river / segment 常用字段建立连续数组视图；
3. 减少在 inner loop 中访问复杂对象成员；
4. 将只读参数与可变状态分离；
5. 将 per-RHS 临时变量改成局部变量或 scratch buffer；
6. 对后续多线程写入数组做 cache-line 友好安排；
7. 避免多个线程后续写入相邻但无关的高频变量导致 false sharing。

**为什么这样做**  
很多并行效率问题不是线程不够，而是内存布局不适合并行。  
如果每个线程访问的是分散对象字段、共享 cache line 或重复计算属性，多线程可能放大内存瓶颈。

**适合 SHUD 吗**  
适合，但要分阶段做。SHUD 当前包含大量 `Element`、`River`、`Lake` 对象。直接大规模 SoA 改造风险较高，建议先做“热字段数组化”，而不是一次性完全重写数据结构。

**风险**

- 改内存布局容易引入索引错误；
- 字段同步问题可能导致旧对象与新数组不一致；
- 编译器可能因为表达式调整引入微小数值差异。

**验收标准**

- 不改变浮点表达式顺序的改造必须 bitwise identical；
- 如果局部重排不可避免，需单独标记为 `new_reference_candidate`，不能混入 strict base；
- hot loop cache miss、RHS 耗时有可测下降；
- 小算例完整输出一致。

---

### 阶段 P5：forcing 与时间序列读取优化

**目标**  
消除 RHS 调用中的 I/O 与时间序列访问瓶颈，同时保持原取值语义。

**主要工作**

1. 把 repeated open / skip / read queue 改为：
   - 持久化 file stream；
   - 大块 buffer；
   - 或全量预加载；
2. 建立时间索引；
3. `getX(t, col)` 第一阶段仍返回与旧版本相同的值；
4. forcing calibration 保持同样顺序；
5. forcing cache 读写增加 checksum；
6. 记录 forcing cache hit/miss 和 refill 时间。

**为什么这样做**  
并行后计算部分更快，I/O 的占比会提高。如果 forcing 读取仍是瓶颈，多线程加速可能被 I/O 吞掉。

**适合 SHUD 吗**  
非常适合。SHUD 的 forcing 是 RHS 高频访问对象，且 `TimeSeriesData` 已经有 ring queue 逻辑，说明作者已经试图避免一次性全读；下一步可以在不改变取值语义的前提下提升缓存策略。

**风险**

- 全量预加载会增加内存占用；
- 时间索引错误会造成 forcing 错位；
- 如果同时引入插值，会改变数值结果。

**验收标准**

- zero-order forcing 取值与旧版本 bitwise identical；
- 完整 run bitwise identical；
- I/O 时间显著下降；
- forcing checksum 一致。

---

### 阶段 P6：增加 solver 与 RHS 诊断，不急于改变 solver

**目标**  
让单线程版本能为后续 CVODE / N_Vector / 线性求解器并行提供诊断依据。

**主要工作**

1. 输出 CVODE 统计：
   - internal steps；
   - RHS evaluations；
   - error test failures；
   - nonlinear iterations；
   - linear iterations；
   - last step size；
   - current order；
2. 输出 RHS 子阶段耗时：
   - update；
   - ET / vertical process；
   - element lateral flux；
   - segment flux；
   - river downflow；
   - gather；
   - applyDY；
   - forcing；
3. 输出关键数组 checksum；
4. 输出水量守恒分解；
5. 保留当前 scalar tolerance，不在这个阶段直接改成 vector tolerance。

**为什么这样做**  
后续一旦引入 OpenMP `N_Vector`、Krylov solver、KLU 或预条件器，完整 run 可能不再 bitwise identical。那时必须依靠 solver diagnostics 判断差异是否合理。

**适合 SHUD 吗**  
非常适合。SHUD 当前由 CVODE 驱动，CVODE 层的内部步数和 RHS 调用次数直接决定速度；没有这些指标，无法判断优化作用点。

**风险**

- 诊断输出过多会影响性能；
- checksum 如果放在热循环中可能改变耗时判断；
- 统计接口要注意 SUNDIALS 版本差异。

**验收标准**

- 诊断开关默认可关闭；
- 开启诊断时，结果不变；
- 小算例诊断可用于定位每次 RHS 的耗时；
- 能比较两个版本的 CVODE 统计差异。

---

### 阶段 P7：建立 strict regression 与 new reference 双轨

**目标**  
把“保持当前 base 不变”和“提高数值精度”分成两条轨道。

**主要工作**

建立两种模式：

#### 1. strict mode

用于并行开发和回归测试。

要求：

```text
RHS bitwise identical
完整 run bitwise identical
CVODE 统计一致
输出文件一致
```

禁止：

```text
fast-math
普通 OpenMP reduction
atomic floating add
改变 forcing 插值
改变容差
改变求和算法
```

#### 2. new reference mode

用于后续提高数值精度。

可以评估：

```text
vector absolute tolerance
forcing interpolation
pairwise summation
compensated summation
rootfinding for threshold processes
更合理的 state scaling
```

但必须重新建立参考结果，并通过水量平衡、多变量验证和跨流域验证。

**为什么这样做**  
当前单线程 base 不一定是数学上最准确的结果，但它是并行开发的唯一可靠 reference。  
如果想提高数值精度，应该单独建立新 reference，而不是和并行重构混在一起。

**适合 SHUD 吗**  
非常适合。SHUD 同时追求速度和精度，必须避免“并行优化”和“物理/数值改进”互相污染。

**风险**

- 团队可能混淆 strict mode 和 new reference mode；
- new reference mode 的结果更好，但无法直接与旧 base bitwise 对比；
- 需要更多验证数据。

**验收标准**

- 两种模式在配置、输出目录和报告中明确区分；
- strict mode 保护并行正确性；
- new reference mode 单独评审和验证。

---

## 5. 单线程预优化与后续并行阶段的对应关系

| 单线程预优化阶段 | 后续并行能改造哪些计算 | 并行方式 | 准确性目标 |
|---|---|---|---|
| P0 基线与画像 | 暂不并行 | 无 | 锁定当前 base |
| P1 RHS 路径统一 | 所有 RHS kernel 后续共享 | 单线程共享 kernel | bitwise identical |
| P2 flux / gather 拆分 | segment flux、river flux、lake flux | flux 计算并行，汇总不抢写 | RHS bitwise identical |
| P3 固定拓扑列表 | river / element / lake owner gather | owner-local deterministic gather | strict mode 下 bitwise identical |
| P4 内存布局整理 | element-local map、segment-local map、river-local map | `parallel for schedule(static)` | strict mode 下 bitwise identical |
| P5 forcing 缓存 | forcing query 不再 I/O 阻塞 | 多线程只读缓存 | forcing 取值 bitwise identical |
| P6 solver 诊断 | OpenMP N_Vector、Krylov、KLU、preconditioner | 后期 solver-level parallel | deterministic tolerance |
| P7 双轨 reference | strict parallel / production parallel 分离 | 两种运行模式 | strict: bitwise；production: deterministic tolerance |

---

## 6. 哪些计算适合在单线程预优化后优先并行

### 6.1 第一批：element-local map

适合内容：

```text
state clipping
BC/SS local update
ET local flux
infiltration
recharge
exfiltration
element-local hydraulic property update
DY element assembly
```

前提：

```text
每个 element 只写自己的数组位置
不写邻居
不写 river
不写 lake accumulator
```

准确性目标：

```text
strict mode 下 RHS bitwise identical
```

### 6.2 第二批：segment-local flux

适合内容：

```text
river segment surface exchange
river segment subsurface exchange
```

前提：

```text
segment flux 只写 QsegSurf[i] / QsegSub[i]
不直接写 QrivSurf/QrivSub/Qe2r_Surf/Qe2r_Sub
```

准确性目标：

```text
Qseg arrays bitwise identical
gather 后 RHS bitwise identical
```

### 6.3 第三批：river-local flux

适合内容：

```text
river downstream flux
river local hydraulic geometry update
```

前提：

```text
每个 river 只写 QrivDown[i] 或自己的 local fields
不直接写 downstream river accumulator
```

准确性目标：

```text
QrivDown bitwise identical
```

### 6.4 第四批：owner-local gather

适合内容：

```text
segments -> river
segments -> element
upstream river -> downstream river
lake elements -> lake
```

前提：

```text
每个 owner 由唯一线程负责
owner 内部按固定 ID 顺序累加
```

准确性目标：

```text
strict mode 下 bitwise identical
production mode 下 deterministic tolerance
```

### 6.5 第五批：CVODE vector / linear solver 层

适合内容：

```text
N_Vector OpenMP
Krylov solver
sparse direct solver
preconditioner
```

前提：

```text
RHS strict parallel 已经通过
诊断体系已经完整
水量守恒与状态误差评价已经固定
```

准确性目标：

```text
通常不再要求完整 run bitwise identical
要求 deterministic within tolerance
```

---

## 7. 精度与一致性标准

### 7.1 strict single-thread refactor 阶段

适用阶段：

```text
P1, P2, P3, P5
```

要求：

```text
RHS 中间数组 bitwise identical
DY bitwise identical
完整输出 bitwise identical
CVODE 统计一致
```

需要比较的关键数组：

```text
uYsf, uYus, uYgw, uYriv, yLakeStg
qElePrep, qEleNetPrep
qEleEvapo, qEleInfil, qEleRecharge, qEleExfil
qEs, qEu, qEg, qTu, qTg
QeleSurf, QeleSub
QsegSurf, QsegSub
QrivSurf, QrivSub, QrivUp, QrivDown
Qe2r_Surf, Qe2r_Sub
qLakePrcp, qLakeEvap, QLakeSurf, QLakeSub, QLakeRivIn, QLakeRivOut
DY
```

### 7.2 strict RHS parallel 阶段

适用阶段：

```text
element-local parallel
segment-local parallel
river-local parallel
owner-local deterministic gather
```

要求：

```text
同一输入 Y 和 t：
DY_parallel == DY_serial_refactored bitwise
```

这一步优先级高于完整 run。  
如果 RHS 都不一致，不应直接进入完整 CVODE run 对比。

### 7.3 complete run strict 阶段

要求：

```text
完整 run 输出 bitwise identical
CVODE internal steps 一致
RHS call count 一致
error test failure count 一致
```

这个阶段用于证明：

```text
应用层 RHS 并行没有改变模型
```

### 7.4 production deterministic 阶段

当引入以下内容时：

```text
OpenMP N_Vector
parallel reduction
Krylov solver
different linear solver
pairwise summation
vector tolerance
forcing interpolation
```

可以放宽 bitwise，但必须满足：

```text
同样配置多次运行结果完全一致
不同线程数结果在容差内
水量守恒不恶化
关键水文指标不恶化
差异可被诊断解释
```

建议阈值初稿：

| 对象 | 建议阈值 |
|---|---|
| 状态变量最大绝对差 | 由变量量纲分组设定，不使用单一阈值 |
| river stage | 以水深尺度设置，例如 1e-6–1e-4 m 级别试验 |
| element surface water | 以 depression / surface depth 尺度设置 |
| unsaturated / groundwater | 以含水深度尺度设置 |
| 累积水量平衡误差 | 不大于原单线程 base |
| outlet flow NSE/KGE | 与 base 相比变化应接近 0，至少不能影响模型判读 |
| peak flow error | 不应因并行策略系统性恶化 |
| 多次运行一致性 | 同线程数应完全可复现 |

这些阈值不应一开始写死，需要用代表性流域做量级校准。

---

## 8. 单线程预优化中的“精度提升”应该怎么处理

需要分清两类精度。

### 8.1 实现精度：必须现在做

实现精度指的是：

```text
不丢 flux
不重复 flux
不改变状态更新顺序
不引入 race condition
不引入非确定性
不改变 forcing 时间定位
不破坏水量守恒
```

这类必须在单线程预优化阶段完成。

### 8.2 数值精度：可以规划，但不要混入 strict refactor

数值精度指的是：

```text
vector absolute tolerance
forcing interpolation
Kahan / pairwise summation
threshold rootfinding
state scaling
更合理的 wet-dry handling
```

这些可能让模型更准确，但它们会改变当前 base。  
因此，建议把它们作为 `new reference mode`，在 strict 并行通过后单独评估。

---

## 9. 编译与运行规则

为了保持 strict mode 可复现，建议：

### 9.1 禁止项

```text
禁止 -ffast-math
禁止不受控的 reassociation
禁止不受控的 FMA contraction
禁止普通 OpenMP reduction 用于 strict floating sum
禁止 atomic floating add 用于 strict accumulator
禁止 dynamic/guided schedule 用于 strict mode
```

### 9.2 推荐项

```text
使用固定编译器和版本
使用固定 SUNDIALS 版本
使用 schedule(static)
使用 owner-local gather
使用固定排序 adjacency list
使用单线程 CVODE N_Vector 作为第一阶段 reference
```

### 9.3 分支策略

建议建立四个层次：

```text
baseline/current
serial-parallel-ready
parallel-strict
parallel-production
```

各层含义：

| 分支 / 模式 | 作用 |
|---|---|
| `baseline/current` | 当前单线程参考 |
| `serial-parallel-ready` | 单线程预优化后版本，仍保持 bitwise |
| `parallel-strict` | RHS 并行，但仍追求 bitwise |
| `parallel-production` | 允许 deterministic tolerance，追求更高性能 |

---

## 10. 推荐总体路线图

```mermaid
flowchart LR
    A[当前单线程 SHUD base] --> B[P0 锁定基线与性能画像]
    B --> C[P1 统一 RHS 路径]
    C --> D[P2 拆分 flux 计算与汇总]
    D --> E[P3 固定拓扑与 owner gather]
    E --> F[P4 内存与 hot data 整理]
    F --> G[P5 forcing 缓存与时间索引]
    G --> H[P6 solver/RHS 诊断增强]
    H --> I[P7 strict/new reference 双轨]
    I --> J[严格等价 RHS 并行]
    J --> K[确定性容差生产并行]
```

---

## 11. 最终建议

为了保证更高的并行效率和精度，**应先对单线程 SHUD 做 parallel-ready 预优化**。这一步的成功标准不是单纯让单线程快多少，而是让 SHUD 具备以下能力：

1. **同一套 RHS kernel 同时服务单线程和并行版本；**
2. **flux 计算与 flux 汇总分离；**
3. **所有汇总都有唯一 owner 和固定顺序；**
4. **forcing 读取不再成为并行后的 I/O 瓶颈；**
5. **所有单线程重构都能通过 bitwise 回归；**
6. **后续并行能先做到 RHS bitwise identical；**
7. **生产并行放宽到 deterministic tolerance 时，有完整诊断和误差解释能力。**

一句话总结：

> **先优化单线程，不是为了替代并行，而是为了让并行变得可验证、可复现、可扩展。**  
> 对 SHUD 来说，最合理的路线是：  
> **当前单线程 base → 单线程 parallel-ready refactor → strict RHS parallel → production deterministic parallel。**

---

## 12. 参考依据

1. SHUD `f.cpp` 中串行与 `_OPENMP_ON` RHS 调用路径不同：  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/Model/f.cpp>

2. SHUD `MD_f.cpp` 中串行 `f_loop()`、`f_applyDY()`、`PassValue()` 的计算与汇总逻辑：  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp>

3. SHUD `MD_f_omp.cpp` 中 OpenMP 版本 `f_loop_omp()`、`f_applyDY_omp()` 逻辑：  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp>

4. SHUD `MD_RiverFlux.cpp` 中 `fun_Seg_surface()` / `fun_Seg_sub()` 同时计算 segment flux 并累加到 river / element accumulator：  
   <https://github.com/SHUD-System/SHUD/blob/master/src/ModelData/MD_RiverFlux.cpp>

5. SHUD `TimeSeriesData.cpp` 中 `read_csv()` 的 repeated open / skip / queue 读取逻辑，以及 `getX()` 的当前取值语义：  
   <https://github.com/SHUD-System/SHUD/blob/master/src/classes/TimeSeriesData.cpp>

6. SHUD `Model_Control.hpp` 中当前标量 `abstol`、`reltol` 与 solver 控制项：  
   <https://github.com/SHUD-System/SHUD/blob/master/src/classes/Model_Control.hpp>

7. OpenMP reduction 规范说明 reduction 用于并行 recurrence，但 strict bitwise reference 不应依赖普通 floating reduction：  
   <https://www.openmp.org/spec-html/5.0/openmpsu107.html>

8. SUNDIALS NVECTOR 文档说明 serial、OpenMP、Pthreads、CUDA、Kokkos 等不同向量实现，这意味着 RHS 并行与 CVODE vector-level 并行应分阶段验证：  
   <https://sundials.readthedocs.io/en/latest/nvectors/NVector_links.html>

9. SUNDIALS CVODE usage 文档说明 CVODE 提供 optional inputs / outputs，可用于 solver 诊断和后续并行验证：  
   <https://sundials.readthedocs.io/en/latest/cvode/Usage/index.html>
