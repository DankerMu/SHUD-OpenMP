# SHUD 并行前对齐工作、分阶段并行改造与精度验收标准

> 文档目的：在以当前单线程 SHUD 计算结果作为 base 的前提下，定义并行前必须完成的对齐工作、每个并行阶段应该改造哪些计算，以及每一阶段需要达到什么精度/一致性标准。  
> 建议定位：作为 SHUD 求解器加速路线中的“并行正确性与验收标准”专项文档。

---

## 0. 核心结论

SHUD 的并行加速不能一开始就以“结果差不多”为标准。  
以当前单线程结果为 base 时，建议采用两级并行准确性路线：

1. **严格等价并行（strict equivalent parallelism）**  
   目标是让 RHS、通量数组、状态导数和完整 run 尽可能与单线程 **bitwise identical**。  
   这是早期开发、调试和回归测试的标准。

2. **确定性容差并行（deterministic tolerance parallelism）**  
   目标是让并行结果每次运行一致，同时允许与单线程 base 有极小、可解释、可量化的浮点差异。  
   这是后期引入 OpenMP `N_Vector`、并行线性求解器、确定性 tree reduction 等更激进优化后的生产标准。

最关键的原则是：

> **先对齐方程路径与副作用，再并行。先实现 RHS bitwise identical，再讨论 CVODE 内部并行。**

当前 SHUD 中，单线程路径和 `_OPENMP_ON` 路径并不是完全相同的“同一套方程的不同执行方式”。例如 `f.cpp` 在 `_OPENMP_ON` 下调用 `f_update_omp() / f_loop_omp() / f_applyDY_omp()`，在串行下调用 `f_update() / f_loop() / f_applyDY()`；这意味着并行前必须先把这几条路径对齐，而不是直接比较 OpenMP 输出和单线程输出。

---

## 1. 为什么必须先做对齐

### 1.1 浮点加法顺序不是小问题

浮点加法不满足结合律。  
也就是说：

```text
((a + b) + c)  不一定等于  (a + (b + c))
```

在并行环境中，只要多个线程同时向同一个浮点变量做 `+=`，即使使用 `atomic` 避免 data race，也不能保证进入加法的顺序与单线程一致。OpenMP reduction 也不能作为严格 bitwise 基准，因为 OpenMP 规范明确说明 reduction 的组合位置与组合顺序是 unspecified，因此不能保证顺序运行与并行运行、甚至两次并行运行之间逐 bit 一致。

因此，SHUD 并行不能把“线程安全”误认为“数值等价”。  
对 SHUD 来说，准确并行应优先满足：

```text
每个输出变量有唯一 owner 写入；
所有多源通量汇总按固定顺序进行；
strict 模式禁止 floating atomic += 和普通 OpenMP floating reduction。
```

### 1.2 当前差异不只是浮点顺序

并行前需要特别注意：当前单线程和 OpenMP 路径中已经存在一些结构性差异。它们会和浮点顺序差异混在一起，使结果差异无法解释。

当前可见的典型差异包括：

| 类型 | 当前现象 | 并行前风险 |
|---|---|---|
| RHS 入口差异 | `_OPENMP_ON` 与串行调用不同的 update/loop/applyDY 函数 | 两边可能不是同一套方程 |
| 湖泊过程差异 | 串行 `f_loop()` 包含 lake element 分支；`f_loop_omp()` 未完整体现同等 lake 分支 | 有湖泊场景下并行结果不可作为单线程等价结果 |
| ET 路径差异 | 串行 `f_loop()` 普通 element 分支调用 `f_etFlux()`；`f_loop_omp()` 中未看到等价调用 | ET、AET、入渗/补给响应可能不同 |
| River DY 差异 | 串行 `f_applyDY()` 对 river 先按 length 得到断面面积变化，再限制负面积并通过 `fun_dAtodY()` 转为水深变化；`f_applyDY_omp()` 直接除以 `u_TopArea` | 河道状态导数公式不同 |
| side-effect 差异 | `fun_Seg_surface()`、`fun_Seg_sub()` 在计算 `Qseg*` 的同时还直接更新 `Qriv*` 与 `Qe2r_*` | 并行 segment loop 中存在共享浮点累加风险 |
| lake / river 汇总差异 | `Flux_RiverDown()` 中存在 `QLakeRivIn[...] += QrivDown[i]` 这类汇总副作用 | 多条河入湖时并行会改变累加顺序或产生竞争 |

所以并行前的第一目标不是“跑得快”，而是：

> **把串行路径、OpenMP 路径、lake 路径、uncouple 路径中的物理方程、状态更新、通量定义和汇总语义统一。**

---

## 2. 并行前必须完成的对齐工作

### 2.1 对齐 1：固定单线程 base 与编译环境

并行前必须先确定一个“可复现的单线程 base”。

建议要求：

| 项目 | 要求 |
|---|---|
| 编译器 | 固定 compiler、版本和标准库 |
| 优化选项 | 禁止 `-ffast-math` 及任何会破坏 IEEE 语义的 aggressive FP 选项 |
| FMA | 明确是否允许 FMA contraction；strict 模式建议关闭或固定 |
| OpenMP | 单线程 base 不启用 OpenMP，或启用 OpenMP 但 `num_threads=1` 作为单独对照，不与纯串行混用 |
| 输出格式 | 用二进制 snapshot 或 `%.17g` 文本输出保存关键数组，避免普通文本格式掩盖差异 |
| 随机性 | 若未来引入随机初始化或采样，必须固定 seed；当前 SHUD 主流程不应依赖随机性 |

验收标准：

```text
同一 executable、同一输入、同一线程数、重复运行 3 次：
- 所有 RHS probe 输出 bitwise identical
- 所有完整 run 输出 bitwise identical
- nFCall、CVODE 内部步数、误差测试失败次数一致
```

若单线程 base 自身不可复现，不应进入并行开发。

---

### 2.2 对齐 2：统一 RHS 主入口

当前 `f.cpp` 下的全耦合 RHS 入口在串行和 OpenMP 下分叉：

```text
_OPENMP_ON:
  f_update_omp()
  f_loop_omp()
  f_applyDY_omp()

serial:
  f_update()
  f_loop()
  f_applyDY()
```

建议把它们统一成一套逻辑：

```text
f_update_core()
f_loop_core()
f_applyDY_core()
```

然后在 core 内部只对安全 loop 增加并行策略，而不是维护两套物理逻辑。

建议目标：

| 对齐项 | 要求 |
|---|---|
| 状态读取 | `Y -> uYsf/uYus/uYgw/uYriv/uYlake` 规则一致 |
| 负值处理 | 是否 clamp 到 0 必须统一；不能串行允许负状态、OpenMP 强行 max(0,Y) |
| BC/SS | 边界条件和源汇项处理位置一致 |
| 临时数组归零 | `Qele*`、`Qriv*`、`QLake*`、`qEle*`、`DY` 的清零时机一致 |
| lake 更新 | `yLakeStg`、`lake[i].update()`、`y2LakeArea`、`QLake*` 初始化一致 |
| forcing 读取 | forcing pointer 移动与 element forcing 读取顺序一致 |
| ET 调用 | `f_etFlux()` 调用位置一致 |
| 通量计算 | element、segment、river、lake 通量定义一致 |
| DY 计算 | surface/unsat/gw/river/lake 的导数公式一致 |

验收标准：

```text
重构后，即使完全不启用并行：
- 新 core 结果必须与原始单线程 base bitwise identical
- 若不能 bitwise identical，必须逐数组定位差异来源
```

---

### 2.3 对齐 3：统一 `f_update()` 与 `f_update_omp()`

`f_update()` 是每次 RHS 评估开始时的状态同步与临时变量初始化阶段。  
在并行前，它必须成为无歧义的状态准备函数。

应对齐内容：

| 类别 | 需要对齐的对象 |
|---|---|
| Element 状态 | `uYsf[i]`、`uYus[i]`、`uYgw[i]` |
| River 状态 | `uYriv[i]`、`Riv[i].updateRiver()` |
| Lake 状态 | `uYlake[i]`、`yLakeStg[i]`、`lake[i].update()`、`y2LakeArea[i]` |
| Element 通量数组 | `QeleSurf[i][j]`、`QeleSub[i][j]`、`QeleSurfTot[i]`、`QeleSubTot[i]` |
| River 通量数组 | `QrivSurf[i]`、`QrivSub[i]`、`QrivUp[i]`、`QrivDown[i]` |
| Lake 通量数组 | `QLakeSub[i]`、`QLakeSurf[i]`、`QLakeRivIn[i]`、`QLakeRivOut[i]`、`qLakeEvap[i]`、`qLakePrcp[i]` |
| ET / vertical arrays | `qEleExfil[i]`、`qEleInfil[i]`、`qEs/qEu/qEg/qTu/qTg` 等 |
| DY | `DY[0:NumY] = 0` |

特别注意：

- 如果串行使用 `uYsf[i] = Y[iSF]`，OpenMP 不能使用 `max(0, Y[iSF])`，除非先把串行也同步修改并重新定义 base。
- `QrivDown[i]` 是否在 `f_update()` 阶段清零必须统一。若不清零，后续 river routing 可能依赖上一次 RHS 的残留状态。
- lake 相关数组在 OpenMP 路径中必须完整初始化，否则有湖泊的流域不能纳入 strict parallel 验证。

验收标准：

```text
给定相同 Y 和 t：
- f_update_core() 后所有 state mirror 数组与原串行 f_update() 完全一致
- 所有临时通量数组清零状态完全一致
- BC/SS 生成的 yBC/QBC/qBC 完全一致
```

---

### 2.4 对齐 4：统一 `f_loop()` 与 `f_loop_omp()`

`f_loop()` 是 RHS 中最核心的物理通量计算阶段。  
并行前必须先让串行和 OpenMP 物理路径等价。

应对齐内容：

| 计算块 | 当前应关注的问题 | 对齐目标 |
|---|---|---|
| ET / AET | 串行普通 element 分支调用 `f_etFlux()`；OpenMP 路径中未完整等价 | 所有非湖泊 element 都按同一位置调用 `f_etFlux()` |
| Lake element vertical | 串行 lake element 调用 `updateLakeElement()`、`fun_Ele_lakeVertical()` | OpenMP 路径必须包含同等处理 |
| Lake evaporation/precipitation | 串行对 `qLakeEvap/qLakePrcp` 做 lake-level 汇总 | 并行中应拆成 per-element contribution，再 deterministic gather |
| Element infiltration/recharge | 可并行，但必须先保证调用顺序和状态依赖一致 | 对每个 element 的内部计算顺序不变 |
| Element surface/sub flow | 当前写 `QeleSurf[i][j]`、`QeleSub[i][j]`，但 lake 邻居时也更新 `QLakeSurf/Sub` | lake 汇总副作用应移出，改为 owner gather |
| Segment surface/sub flow | 当前 `fun_Seg_*` 同时写 `Qseg*` 和累加 `Qriv*`、`Qe2r_*` | 拆成“只计算 Qseg”的纯函数 + 后续 gather |
| River downflow | `Flux_RiverDown()` 写 `QrivDown[i]`，入湖时还累加 `QLakeRivIn` | 拆成“只计算 QrivDown”的纯函数 + 后续 lake gather |
| `PassValue()` | 串行按 segment id 与 river id 汇总 | 改成 deterministic gather 的唯一实现 |

验收标准：

```text
给定相同 update 后状态：
- f_loop_core() 后所有 qEle*、Qele*、Qseg*、Qriv*、QLake* 数组与串行 base bitwise identical
- lakeon = 0 和 lakeon = 1 两类算例都必须通过
```

---

### 2.5 对齐 5：统一 `f_applyDY()` 与 `f_applyDY_omp()`

`f_applyDY()` 是把通量转为状态导数的最后一步。  
这个阶段必须优先对齐，因为 CVODE 对 `DY` 极其敏感。

应对齐内容：

| 状态 | 对齐要求 |
|---|---|
| Surface | `DY[iSF] = qEleNetPrep - qEleInfil + qEleExfil - QeleSurfTot/area - qEs` |
| Unsat | `DY[iUS] = qEleInfil - qEleRecharge - qEu - qTu`，再除以 `Sy` |
| GW | `DY[iGW] = qEleRecharge - qEleExfil - QeleSubTot/area - qEg - qTg`，再除以 `Sy` |
| BC/SS | BC/SS 对 surface/GW 的修正顺序一致 |
| Lake element | lake element 对 element 三层 `DY` 的置零规则一致 |
| River | 必须统一为串行基准中的 river DY 公式，包含 length 归一、负面积限制和 `fun_dAtodY()` 转换 |
| Lake | `DY[iLAKE]` 必须在 OpenMP 路径中完整实现 |
| NaN/negative checks | Debug 检查不应改变结果；错误处理一致 |

验收标准：

```text
给定完全相同的通量数组：
- DY 全量 memcmp 一致
- max_ulp(DY) = 0
- 若只允许文本比较，必须以 %.17g 输出并逐项解析比较
```

---

### 2.6 对齐 6：拆除共享浮点累加副作用

并行前必须识别所有类似下面的写法：

```cpp
shared_array[k] += value;
```

在 SHUD 当前路径中，重点关注：

| 共享累加 | 典型来源 | 并行风险 |
|---|---|---|
| `QrivSurf[ir] += QsegSurf[i]` | segment → river | 多 segment 同属 river，线程竞争 |
| `QrivSub[ir] += QsegSub[i]` | segment → river | 同上 |
| `Qe2r_Surf[ie] += -QsegSurf[i]` | segment → element | 多 segment 同属 element，线程竞争 |
| `Qe2r_Sub[ie] += -QsegSub[i]` | segment → element | 同上 |
| `QrivUp[iDownStrm] += -QrivDown[i]` | upstream river → downstream river | 多上游河段汇入同一 downstream |
| `QLakeSurf[ilake] += Q` | element bank → lake | 多岸边 element 汇入同一 lake |
| `QLakeSub[ilake] += Q` | element bank → lake | 同上 |
| `QLakeRivIn[lake] += QrivDown[i]` | river → lake | 多 river 入湖 |
| `qLakeEvap[lake] += ...` | lake element evaporation | 多 lake element 汇总 |

建议改法：

```text
flux 计算阶段：只写唯一位置，例如 QsegSurf[iseg]、edgeLakeSurf[iedge]、riverToLake[iriv]
汇总阶段：每个 owner 独占写入，例如每个 river/lake/element 由一个线程负责
owner 内部按固定 contributor id 升序累加
```

这就是：

> **owner-computes deterministic gather**

验收标准：

```text
strict 模式下：
- 不允许 floating atomic +=
- 不允许普通 OpenMP floating reduction
- 所有 shared accumulation 都由 owner-local 固定顺序 gather 替代
```

---

### 2.7 对齐 7：建立 RHS probe 和数组级回归测试

不能只比较最终出流过程线。  
并行开发必须先能比较单次 RHS 评估。

建议新增一个测试入口：

```text
shud --rhs-probe <project> --time <t> --state <Y_file> --dump <out_dir>
```

输出至少包括：

```text
Y
uYsf/uYus/uYgw/uYriv/uYlake
qEleInfil/qEleRecharge/qEleExfil
qEs/qEu/qEg/qTu/qTg/qEleETA
QeleSurf/QeleSub/QeleSurfTot/QeleSubTot
QsegSurf/QsegSub
Qe2r_Surf/Qe2r_Sub
QrivSurf/QrivSub/QrivUp/QrivDown
QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut/qLakeEvap/qLakePrcp
DY
```

验收标准：

```text
RHS probe 单点通过后，才能进入完整 CVODE run 对比。
```

否则完整时间积分中任何微小差异都可能被自适应步长放大，难以定位。

---

## 3. 分阶段并行改造路线

下面路线以“先 strict，再 production”为原则。

---

## 阶段 0：单线程基线与性能/精度画像

### 目标

建立单线程 base，并确认当前 base 自身可复现。

### 改造对象

本阶段不做并行改造，只增加观测能力：

| 对象 | 工作 |
|---|---|
| RHS 调用 | 记录 `nFCall`、每次 RHS 时间 |
| CVODE | 记录内部步数、error test failure、linear solver 统计 |
| I/O | 记录 forcing 读取、输出写入耗时 |
| 数组输出 | 增加 RHS probe / debug snapshot |
| 完整 run | 保存输出文件 checksum 和关键 hydrograph 指标 |

### 精度要求

| 项目 | 标准 |
|---|---|
| 重复运行 | bitwise identical |
| RHS probe | `max_ulp(DY)=0` |
| 完整输出 | 文件 checksum 一致，或解析为 double 后所有值一致 |
| CVODE 统计 | `nFCall`、内部步数、失败次数一致 |

### 风险

如果基线只选一个小流域或无湖泊/无边界/无雪工况，后续并行可能只对简单场景有效。  
建议至少准备：

```text
小型单元测试流域；
中等规模真实流域；
含湖泊算例；
含河网汇流算例；
含 cryosphere/snow 算例；
强降水/洪峰算例。
```

---

## 阶段 1：串行、OpenMP、lake 路径完全对齐

### 目标

消除“不是同一套模型”的差异。

### 需要改造的计算

| 模块 | 改造内容 |
|---|---|
| `f.cpp` | 不再让 `_OPENMP_ON` 选择另一套物理逻辑，而是选择同一套 core 的 parallel policy |
| `f_update()` / `f_update_omp()` | 合并状态准备逻辑 |
| `f_loop()` / `f_loop_omp()` | 合并 ET、lake、element、segment、river 逻辑 |
| `f_applyDY()` / `f_applyDY_omp()` | 合并 DY 公式，尤其是 river 和 lake DY |
| lake path | lake element、lake-bank、river-to-lake 全部纳入同一 RHS 逻辑 |
| uncouple path | 暂时不作为并行主线；只作为后续独立分支验证 |

### 精度要求

本阶段即使不开并行，也必须满足：

```text
新统一 core vs 原始单线程 base：
- RHS probe: all arrays bitwise identical
- complete run: bitwise identical
- CVODE stats: identical
```

如果出现差异，应按以下顺序定位：

```text
Y/uY mirror
forcing/tReadForcing
ET/AET
infiltration/recharge
element lateral flux
segment flux
river downflow
gather arrays
DY
```

### 风险

这一步不是性能优化，但它是所有性能优化的前提。  
如果跳过此阶段，后面所有加速比和精度差异都不可信。

---

## 阶段 2：element-local / river-local / lake-local map 并行

### 目标

先并行“每个 owner 只写自己”的计算，不碰共享浮点累加。

### 可以并行的计算

| 计算 | 是否适合 strict 并行 | 条件 |
|---|---:|---|
| element 状态同步 | 是 | 每个线程只写 `uYsf[i] / uYus[i] / uYgw[i]` |
| river 状态同步 | 是 | 每个线程只写 `uYriv[i]` 和 `Riv[i]` 自身属性 |
| lake 状态同步 | 是 | 每个线程只写 `lake[i]` 与 lake-owned 数组 |
| element forcing 读取 | 是 | `TimeSeriesData::movePointer()` 已在单线程或 forcing-owner 阶段完成，`getX()` 只读 |
| ET / snow / interception | 是 | 每个线程只更新 element-owned arrays；`AccT_surf[i]` / `AccT_sub[i]` 只由 element i owner 更新 |
| infiltration / recharge | 是 | 每个线程只写 `qEleInfil[i] / qEleRecharge[i] / qEleExfil[i]` |
| element-to-element surface/sub flux | 基本是 | 只写 `QeleSurf[i][j] / QeleSub[i][j]`，不写 neighbor，不写 lake accumulator |
| river downflow `QrivDown[i]` | 是 | 只写 `QrivDown[i]`，不直接写 `QLakeRivIn` |
| lake element vertical | 是 | 只写 element-owned ET/vertical arrays，不直接累加到 lake |

### 必须禁止的写法

在本阶段不允许：

```cpp
QrivSurf[ir] += ...
Qe2r_Surf[ie] += ...
QLakeSurf[ilake] += ...
QLakeRivIn[ilake] += ...
qLakeEvap[ilake] += ...
```

### 推荐并行策略

```text
#pragma omp parallel for schedule(static)
for owner in owners:
    compute owner-local quantities only
```

要求：

- 不使用 `schedule(dynamic)` 或 `schedule(guided)`；
- 不在 loop 内进行共享浮点 `+=`；
- 不改变每个 element 内部 `j = 0..2` 的计算顺序；
- 不把 `QeleSurf[i][j]` 和 `QeleSurf[neighbor][neighbor_j]` 强行同步写入。

### 精度要求

| 对比对象 | 标准 |
|---|---|
| owner-local arrays | bitwise identical |
| `QeleSurf[i][j] / QeleSub[i][j]` | bitwise identical |
| `Qseg*` 尚未并行时 | 不要求 |
| 完整 RHS | 暂不要求整体通过，除非 gather 仍使用串行基准 |

本阶段结束时，至少应证明：

```text
所有 local map 并行块单独打开时，不改变对应数组任何 bit。
```

### 风险

- `TimeSeriesData::movePointer()` 会改变内部 ring buffer 指针，不应在多个线程对同一 forcing 对象同时调用。
- `AccTemperature` 类对象若按 element 独占更新是安全的；若多个线程更新同一个 element 的累计温度，则不安全。
- element-to-lake flux 当前有 lake accumulator 副作用，必须先拆出临时 per-edge/per-bank contribution。

---

## 阶段 3：segment / edge flux 计算并行

### 目标

把 segment flux 和 edge flux 改成“纯计算、唯一写入”，为 deterministic gather 做准备。

### 需要改造的计算

| 当前函数 | 改造方向 |
|---|---|
| `fun_Seg_surface(iEle, iRiv, iseg)` | 只计算并写 `QsegSurf[iseg]`，不更新 `QrivSurf`、`Qe2r_Surf` |
| `fun_Seg_sub(iEle, iRiv, iseg)` | 只计算并写 `QsegSub[iseg]`，不更新 `QrivSub`、`Qe2r_Sub` |
| `Flux_RiverDown(t, iriv)` | 只计算并写 `QrivDown[iriv]`，不更新 `QLakeRivIn` |
| `fun_Ele_surface()` 中 lake neighbor 分支 | 不直接 `QLakeSurf[ilake] += Q`，改为写 edge/lake-bank contribution |
| `fun_Ele_sub()` 中 lake neighbor 分支 | 不直接 `QLakeSub[ilake] += Q`，改为写 edge/lake-bank contribution |

建议新增逻辑结构：

```text
compute_element_edge_fluxes()
compute_segment_fluxes()
compute_river_down_fluxes()
compute_lake_bank_fluxes()
```

每个函数只做：

```text
output[index] = value
```

不做：

```text
accumulator[owner] += value
```

### 精度要求

| 对比对象 | 标准 |
|---|---|
| `QsegSurf/QsegSub` | bitwise identical |
| `QrivDown` | bitwise identical |
| lake-bank contribution arrays | 与原 lake 汇总分解后一致 |
| 无 gather 的中间数组 | `max_ulp=0` |

### 风险

这一步容易改变原来的 side-effect 顺序。  
所以不能同时改变汇总算法。建议先在串行模式下实现：

```text
旧函数 side-effect 汇总
vs
新函数 pure flux + 串行顺序 gather
```

二者必须 bitwise identical 后，才能打开 flux loop 并行。

---

## 阶段 4：deterministic gather 并行

### 目标

把所有多源汇总改成 owner-local 固定顺序 gather。

### 需要改造的汇总

| 汇总 | owner | contributor | 顺序要求 |
|---|---|---|---|
| segment → river surface | river | segments belonging to river | 按原始 segment id 升序 |
| segment → river subsurface | river | segments belonging to river | 同上 |
| segment → element surface | element | river segments adjacent to element | 按原始 segment id 升序 |
| segment → element subsurface | element | 同上 | 同上 |
| upstream river → downstream river | downstream river | upstream rivers | 按 upstream river id 升序，或严格复现原 `iriv` loop 顺序 |
| river → lake | lake | rivers flowing into lake | 按 river id 升序 |
| bank element → lake surface | lake | lake-bank edge contributions | 按 element id、local edge id 固定排序 |
| bank element → lake subsurface | lake | 同上 | 同上 |
| lake element evap/precip | lake | lake elements | 按 element id 升序 |
| element neighbor flux total | element | 3 local edges + e2r | 先 `Qe2r_*`，再 `j=0,1,2`，保持串行顺序 |

### 推荐数据结构

并行前构建并固定以下 adjacency lists：

```text
seg_by_river[iriv]
seg_by_element[iele]
upstream_by_downstream[iriv]
riv_in_by_lake[ilake]
lake_bank_edge_by_lake[ilake]
lake_element_by_lake[ilake]
```

所有 list 必须满足：

```text
构建一次；
排序一次；
后续只读；
strict 模式下不允许运行时重排。
```

### 推荐 gather 形态

```cpp
#pragma omp parallel for schedule(static)
for (int owner = 0; owner < NumOwner; ++owner) {
    double sum = 0.0;
    for (int p = 0; p < list[owner].size(); ++p) {
        int contributor = list[owner][p];
        sum += contribution[contributor];
    }
    owner_array[owner] = sum;
}
```

注意：

- 每个 owner 只由一个线程写；
- owner 内部求和顺序固定；
- 不需要 `atomic`；
- 不需要 OpenMP reduction；
- 如果 list 顺序与原串行顺序一致，可以实现 bitwise identical。

### 精度要求

本阶段应达到 RHS 级严格等价：

```text
RHS probe:
- all flux arrays bitwise identical
- all gather arrays bitwise identical
- DY bitwise identical
- max_ulp(DY) = 0
```

如果 `DY` 不一致，不能进入完整 CVODE 并行 run。

### 风险

- 若 owner 内部 list 顺序与原串行 `for i=0..N` 的贡献顺序不同，结果可能不是 bitwise identical。
- 若为了速度使用 pairwise/tree reduction，可能数值上更好，但会改变 base；应推迟到 production mode。
- lake-bank edge 的 contributor identity 必须明确，否则很难重现原始累加顺序。

---

## 阶段 5：RHS 全流程 OpenMP，并保持 CVODE serial vector

### 目标

先只并行 SHUD 自己的 RHS 计算，不并行 CVODE 内部向量操作和线性求解器。

这是最重要的验证阶段。

### 需要改造的计算

前面阶段已经完成 local map、flux、gather。  
本阶段要做的是把它们串成完整 RHS：

```text
f_update_core()
f_loop_core()
f_applyDY_core()
```

并且要求：

```text
CVODE 使用 serial N_Vector；
RHS 内部使用 OpenMP parallel for；
不启用 OpenMP N_Vector；
不改变 CVODE linear solver。
```

原因是：如果同时启用 OpenMP `N_Vector`，CVODE 内部 norm、dot product、error test 也会发生 reduction 顺序变化。那样即使 RHS 没问题，完整 run 也可能因为 CVODE 自适应步长差异而不再 bitwise identical。

### 精度要求

| 层级 | 标准 |
|---|---|
| 单次 RHS probe | `DY_parallel == DY_serial`，bitwise identical |
| 完整 CVODE run | 所有输出 bitwise identical |
| CVODE 统计 | 内部步数、RHS 调用次数、error test failure 完全一致 |
| 多线程重复性 | 同一线程数重复运行 bitwise identical |
| 不同线程数 | strict 模式下也应 bitwise identical，前提是 gather 顺序不依赖线程数 |

### 风险

- parallel region 的创建/销毁开销可能降低小流域速度收益。
- 若某些函数内部仍有隐藏 shared side effect，RHS probe 会暴露差异。
- 如果 compiler 自动向量化改变浮点表达式顺序，可能导致 bitwise 不一致；strict 模式应限制相关编译选项。

---

## 阶段 6：CVODE OpenMP `N_Vector` 与线性求解层并行

### 目标

在 RHS 已经证明严格等价后，再进入 CVODE 内部并行和线性求解器优化。

SUNDIALS/CVODE 支持多种 `N_Vector` 实现，包括 serial、MPI-parallel、OpenMP、Pthreads、Hypre、PETSc 和 GPU-enabled 实现。引入 OpenMP `N_Vector` 后，CVODE 内部 vector operations 将由 OpenMP 并行执行，这会改变 norm/dot/reduction 的组合顺序。因此本阶段不应再要求与单线程完整 run bitwise identical。

### 需要改造/评估的计算

| 对象 | 工作 |
|---|---|
| `N_Vector` | 比较 Serial vs OpenMP |
| CVODE tolerance | 考虑从标量 abstol 走向 vector abstol |
| linear solver | 比较 Dense / KLU / GMRES / FGMRES |
| preconditioner | 根据 SHUD 水文耦合结构设计物理分块预条件器 |
| CVODE stats | 暴露并记录 internal step、RHS call、linear iteration、error test failure、nonlinear iteration |

### 精度要求

本阶段进入 **deterministic tolerance**：

| 对比 | 标准 |
|---|---|
| 同 executable、同线程数重复运行 | bitwise identical 或至少全部关键输出完全一致 |
| 不同线程数重复运行 | 允许微小差异，但必须 deterministic |
| 与 serial base | 在容差内，且水量守恒不恶化 |
| CVODE 统计 | 允许变化，但必须记录并解释 |
| 水文指标 | NSE/KGE、洪峰、总水量误差变化不得超过阈值 |

建议初始容差：

| 指标 | 建议阈值 |
|---|---|
| state max abs error | `<= max(1e-10 m, 1e-12 * state_scale)` |
| flux relative error | `<= 1e-10`；极小 flux 使用 absolute floor |
| water balance delta | `<= 1e-9` of total precipitation/input volume，或等效水深 `<= 1e-10 m` |
| hydrograph NSE/KGE change | `<= 1e-4` |
| peak flow relative change | `<= 1e-4` |
| total runoff volume relative change | `<= 1e-5` |
| ULP distribution | 记录 `p50/p95/p99/max`，不作为唯一判据 |

这些阈值是开发初始建议，后续应根据代表性流域规模、时间步、单位和输出精度校准。

### 风险

- CVODE 自适应步长会放大极小浮点差异。
- 线性求解器变化可能改变收敛路径，进而改变 RHS 调用次数。
- 预条件器可显著加速，但也会引入调参成本和收敛失败风险。
- 如果没有 CVODE stats，无法判断加速来自真正的求解效率提升，还是来自步长路径改变。

---

## 阶段 7：生产模式的确定性高性能 reduction

### 目标

在 strict RHS 并行已经通过后，可以单独评估更高性能或更高数值稳定性的求和算法。

可选方法包括：

```text
deterministic pairwise summation
fixed tree reduction
Kahan / Neumaier compensated summation
reproducible accumulator / binned summation
```

注意：这些方法可能比当前串行左折叠更准确，但它们不等于当前单线程 base。  
因此必须作为独立数值算法升级，而不是和并行 bug 修复混在一起。

### 适用计算

| 对象 | 可选方法 |
|---|---|
| large owner gather | fixed pairwise / tree |
| 全域水量 balance summary | compensated summation |
| 大量 cell/edge 汇总 | reproducible accumulator |
| 输出统计 | compensated / pairwise |

### 精度要求

本阶段不再追求 serial base bitwise identical，而追求：

```text
同线程数重复运行一致；
不同线程数运行一致，或差异受控；
与 strict mode 的差异可解释；
水量守恒不恶化，最好改善。
```

### 风险

- 改变求和算法会改变历史 base。
- 如果没有独立记录，会被误认为并行误差。
- 某些 compensated/reproducible 方法会增加计算成本。

建议只有在阶段 5 已经通过后才进入。

---

## 4. 精度与一致性等级定义

建议在 SHUD 加速路线中正式引入以下等级。

| 等级 | 名称 | 定义 | 适用阶段 |
|---|---|---|---|
| A0 | baseline repeatability | 单线程 base 多次运行 bitwise identical | 阶段 0 |
| A1 | refactor equivalence | 重构后不开并行，与原单线程 bitwise identical | 阶段 1 |
| A2 | RHS bitwise equivalence | 单次 RHS 评估中所有关键数组和 `DY` bitwise identical | 阶段 2–5 |
| A3 | full-run bitwise equivalence | 完整 CVODE run 输出与单线程 bitwise identical，CVODE stats 一致 | 阶段 5 |
| A4 | deterministic tolerance | 并行结果重复运行一致，与单线程差异在阈值内 | 阶段 6 |
| A5 | physical acceptance | 水文指标、水量守恒和跨流域表现可接受 | 阶段 7 及生产评估 |

### 4.1 A2：RHS bitwise equivalence

必须比较：

```text
DY
qEleInfil/qEleRecharge/qEleExfil
qEs/qEu/qEg/qTu/qTg
QeleSurf/QeleSub/QeleSurfTot/QeleSubTot
QsegSurf/QsegSub
Qe2r_Surf/Qe2r_Sub
QrivSurf/QrivSub/QrivUp/QrivDown
QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut
qLakeEvap/qLakePrcp
```

通过标准：

```text
max_abs_diff = 0
max_ulp_diff = 0
NaN/Inf pattern identical
array length/order identical
```

### 4.2 A3：full-run bitwise equivalence

必须比较：

```text
所有输出时间点的状态；
所有输出 flux；
最终 restart/init update；
CVODE 内部统计；
nFCall；
输出文件 checksum。
```

通过标准：

```text
全部一致。
```

如果文本输出格式不稳定，应增加二进制 debug dump 或 `%.17g` 输出。

### 4.3 A4：deterministic tolerance

A4 不要求与单线程逐 bit 一致，但要求：

```text
同一 executable + 同一线程数 + 同一输入，重复运行一致；
不同线程数之间差异受控；
与 serial base 差异可解释；
水量守恒和水文指标不恶化。
```

建议记录：

```text
max_abs_error
max_rel_error
ULP p50/p95/p99/max
water balance error
NSE/KGE difference
peak flow difference
runoff volume difference
CVODE stats difference
```

---

## 5. 每个阶段的推荐验收门槛

| 阶段 | 主要改造 | 精度门槛 | 是否允许进入下一阶段 |
|---|---|---|---|
| 0 | 建立基线 | A0 | 单线程不可复现则不能进入 |
| 1 | 路径对齐 | A1 | 不通过则不能并行 |
| 2 | local map 并行 | 对应 local arrays A2 | 可逐块进入阶段 3 |
| 3 | segment/edge flux 并行 | flux arrays A2 | 不通过不能 gather |
| 4 | deterministic gather | full RHS A2 | 不通过不能跑完整 CVODE |
| 5 | RHS OpenMP + serial CVODE | A3 | 不通过不能启用 OpenMP N_Vector |
| 6 | CVODE vector/linear solver 并行 | A4 | 可进入生产评估 |
| 7 | 高性能 deterministic reduction | A4/A5 | 作为可选优化 |

---

## 6. 推荐的并行安全矩阵

| 计算对象 | 并行安全性 | strict 模式策略 | production 模式策略 |
|---|---:|---|---|
| `f_update` element loop | 高 | parallel for static，owner-only write | 同 strict |
| `f_update` river loop | 高 | parallel for static，owner-only write | 同 strict |
| forcing `movePointer` | 中 | 单线程或 forcing-owner 固定顺序 | 可并行，但每个 forcing object 单 owner |
| `tReadForcing` per element | 高 | parallel for static，只读 TSD | 同 strict |
| `ET` per element | 高 | parallel for static，element owner 更新 | 同 strict |
| infiltration/recharge | 高 | parallel for static | 同 strict |
| element lateral flux | 中 | 只写 `Qele[i][j]`，lake 副作用移出 | 可 vectorize |
| segment flux | 中 | 只写 `Qseg[iseg]` | 同 strict |
| river downflow | 中 | 只写 `QrivDown[iriv]` | 同 strict |
| segment → river/element | 高风险 | owner-local fixed-order gather | fixed tree / pairwise 可选 |
| river → downstream | 高风险 | owner-local fixed-order gather | fixed tree 可选 |
| element/lake → lake | 高风险 | owner-local fixed-order gather | fixed tree / compensated 可选 |
| global water balance | 高风险 | 固定顺序串行或 owner gather | compensated/reproducible sum |
| CVODE `N_Vector` ops | 中高 | 先保持 serial | OpenMP N_Vector，A4 验收 |
| linear solver | 高 | 不变 | KLU/GMRES/FGMRES + stats |

---

## 7. 不建议采用的做法

### 7.1 不建议用 `atomic +=` 解决浮点汇总

```cpp
#pragma omp atomic
QrivSurf[ir] += QsegSurf[i];
```

这只能避免 data race，不能保证加法顺序与单线程一致。  
在 strict 模式下不应使用。

### 7.2 不建议用普通 OpenMP reduction 作为 strict 基准

```cpp
#pragma omp parallel for reduction(+:sum)
```

OpenMP reduction 合法，但组合顺序不保证等于串行左折叠。  
它可以进入 production mode，但不应用于 A2/A3 的严格验收。

### 7.3 不建议一开始就启用 OpenMP `N_Vector`

CVODE 内部 vector operations 可能包含 norm/dot/reduction。  
若一开始就启用 OpenMP `N_Vector`，RHS 差异和 CVODE 内部 reduction 差异会混在一起。

### 7.4 不建议同时改并行和求和算法

例如在并行时同步引入 Kahan summation。  
这可能让结果更稳定，但会改变 base，使问题定位困难。

推荐顺序：

```text
先复现当前单线程；
再证明并行不改模型；
最后单独评估求和算法升级。
```

---

## 8. 对 SHUD 的具体建议优先级

### P0：必须先做

```text
1. 固定单线程 base 与编译选项；
2. 建立 RHS probe；
3. 合并 f_update/f_update_omp；
4. 合并 f_loop/f_loop_omp；
5. 合并 f_applyDY/f_applyDY_omp；
6. 拆除 fun_Seg_* 和 lake/river 函数中的共享累加副作用；
7. 建立 deterministic gather。
```

### P1：随后做

```text
1. element-local map 并行；
2. segment flux 并行；
3. river downflow 并行；
4. lake-bank contribution 并行；
5. full RHS OpenMP with serial N_Vector。
```

### P2：生产加速阶段

```text
1. OpenMP N_Vector；
2. KLU / GMRES / FGMRES；
3. 物理分块预条件器；
4. deterministic tree reduction；
5. compensated water-balance summary。
```

---

## 9. 最终推荐实施节奏

```text
单线程基线
  ↓
路径对齐：serial / OpenMP / lake / DY
  ↓
side-effect 拆分：flux compute 与 gather 分离
  ↓
local map 并行
  ↓
segment / edge / river flux 并行
  ↓
deterministic gather
  ↓
RHS bitwise identical
  ↓
完整 CVODE run bitwise identical
  ↓
CVODE OpenMP N_Vector / 线性求解器并行
  ↓
deterministic tolerance 生产模式
```

---

## 10. 参考依据

1. SHUD `f.cpp`：全耦合 RHS 在 `_OPENMP_ON` 和串行下分别调用不同 update/loop/applyDY 路径。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/Model/f.cpp>

2. SHUD `MD_f.cpp`：串行 `f_loop()`、`f_applyDY()`、`PassValue()` 的主路径，包括 lake 分支、river DY 和 segment/river/element 汇总。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp>

3. SHUD `MD_f_omp.cpp`：当前 OpenMP 路径中的 `f_update_omp()`、`f_loop_omp()`、`f_applyDY_omp()`。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp>

4. SHUD `MD_ElementFlux.cpp`：element surface/sub flux 中存在 lake accumulator side effects。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_ElementFlux.cpp>

5. SHUD `MD_RiverFlux.cpp`：segment flux 函数中同时计算 `Qseg*` 并累加 `Qriv*`、`Qe2r_*`；river downflow 中存在 river-to-lake 汇总副作用。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_RiverFlux.cpp>

6. SHUD `MD_ET.cpp`：forcing 读取、ET、snow/interception、`f_etFlux()` 的当前实现。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_ET.cpp>

7. SHUD `TimeSeriesData.cpp`：time-series queue、`read_csv()`、`movePointer()`、`getX()` 的当前行为。  
   <https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/TimeSeriesData.cpp>

8. OpenMP 5.0 reduction 规范：reduction 的组合位置和组合顺序 unspecified，不能保证 bitwise-identical results。  
   <https://www.openmp.org/spec-html/5.0/openmpsu107.html>

9. SIAM News on IEEE 754：浮点加法不满足结合律，并行/向量环境不能假设固定求和顺序。  
   <https://www.siam.org/publications/siam-news/articles/a-new-ieee-754-standard-for-floating-point-arithmetic-in-an-ever-changing-world/>

10. SUNDIALS/CVODE 文档：CVODE 的 `N_Vector` 抽象支持 serial、OpenMP、MPI、Pthreads、Hypre、PETSc、GPU 等实现；向量操作模块与积分算法分离。  
    <https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html>
