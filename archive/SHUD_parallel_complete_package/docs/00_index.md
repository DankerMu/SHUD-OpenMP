# 00 总览：SHUD 并行前对齐与完整并行改造路线

## 1. 这次修订的核心变化

前一版文档主要回答“并行前要对齐什么”，但没有把后续并行阶段系统打包。新版的逻辑改为：

```text
先整理单线程 reference implementation
再打开 strict OpenMP 并行
最后进入 production deterministic 并行
```

也就是说，**对齐工作不是并行阶段的前置说明，而是整个并行工程的第一半部分**。如果 SHUD 当前单线程路径本身不具备并行改造基础，后面任何 OpenMP 改造都会把三类问题混在一起：

1. 物理方程路径差异；
2. 状态更新和共享副作用差异；
3. 浮点加法顺序差异。

这三类问题必须先拆开处理。

## 2. 为什么不能直接继续维护当前 `_omp` 路径

当前 `f.cpp` 在 `_OPENMP_ON` 和非 OpenMP 下调用的是两套不同 RHS 路径：

```text
OpenMP:     f_update_omp() → f_loop_omp() → f_applyDY_omp()
Serial:     f_update()     → f_loop()     → f_applyDY()
```

这意味着当前 OpenMP 路径不是串行路径的执行策略切换版，而是另一套实现。这样做的结果是：当并行结果和单线程结果不一致时，很难判断不一致来自并行、来自物理路径差异，还是来自浮点求和顺序。源码依据见 SHUD `f.cpp`：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/Model/f.cpp

因此，本路线要求：

> 后续不再以“维护 serial / omp 两套 RHS 逻辑”为目标，而是建立唯一 RHS core；OpenMP 只作为 loop policy / execution policy。

## 3. 三个基线定义

### B0：historical serial base

B0 是当前单线程 SHUD 的历史参考结果。用途是：

- 锁定当前模型行为；
- 识别单线程预优化是否改变结果；
- 为后续 intentional change 提供说明依据。

B0 不代表“数学上最准确”，只代表“当前实现的参考答案”。

### B1：parallel-ready serial reference

B1 是完成单线程 parallel-ready 预优化后的新参考实现。用途是：

- 作为 strict 并行阶段的唯一对照；
- 作为后续 production 并行容差比较对象；
- 作为长期回归测试标准。

B1 可能与 B0 完全 bitwise identical；如果在对齐阶段修复了明确的路径不一致或 bug，也可能与 B0 有差异。只要有差异，必须在变更记录中说明：

```text
差异来源：路径修复 / bug 修复 / I/O 缓存但保持语义 / 物理过程修复 / 数值算法变化
影响范围：RHS 局部数组 / 完整 run / 输出文件 / 水量平衡 / hydrograph 指标
是否接受：接受原因 + 验收指标
```

### P-strict：strict parallel result

P-strict 是 strict OpenMP 阶段的并行结果。目标是：

```text
P-strict 与 B1 bitwise identical
```

这个阶段不追求最大速度，而追求证明：**并行没有改掉模型**。

### P-prod：production deterministic result

P-prod 是 production 并行阶段的结果。可允许与 B1 有微小差异，但必须满足：

```text
同一配置多次运行可复现；
不同线程数差异可解释；
水量守恒和关键水文指标不恶化；
CVODE 统计变化可监控。
```

## 4. 总体阶段路线

```text
S0 锁定 B0 historical base
  ↓
S1 建立唯一 RHS core
  ↓
S2 对齐 serial / omp 已存在的语义差异
  ↓
S3 拆分 compute flux 与 deterministic gather
  ↓
S4 固定拓扑顺序、邻接表和 owner 映射
  ↓
S5 整理 scratch arrays、forcing cache 和诊断接口
  ↓
S6 锁定 B1 parallel-ready serial reference
  ↓
P1 并行 local state update / initialization
  ↓
P2 并行 element vertical processes
  ↓
P3 并行 element edge flux compute
  ↓
P4 并行 segment-river pure flux compute
  ↓
P5 并行 owner-local deterministic gather
  ↓
P6 并行 applyDY element / river / lake
  ↓
P7 完整 RHS OpenMP + serial CVODE
  ↓
P8 CVODE OpenMP N_Vector / KLU / Krylov / preconditioner
  ↓
P9 production deterministic reduction / compensated summation
```

## 5. 核心原则

### 原则 1：并行前先形成唯一 RHS core

不要让 `f_loop()` 和 `f_loop_omp()` 继续各自演化。正确方向是：

```cpp
rhs_core(t, Y, DY, execution_policy)
```

其中 execution policy 可以是 serial、strict_omp、production_omp；但物理过程和状态更新顺序由同一个核心实现描述。

### 原则 2：计算通量可以并行，汇总通量必须固定顺序

SHUD 中最危险的并行对象不是每个单元的局部计算，而是多个贡献项汇总到同一个 element、river 或 lake 的浮点加法。strict 模式中不应使用：

```cpp
#pragma omp atomic
sum += x;

#pragma omp parallel for reduction(+:sum)
```

OpenMP 规范明确指出，reduction 值的组合位置和组合顺序是 unspecified，不能保证 sequential 和 parallel bitwise identical。见 OpenMP reduction 规范：https://www.openmp.org/spec-html/5.0/openmpsu107.html

### 原则 3：strict 阶段只验证执行策略，不改物理过程

strict 阶段禁止把以下工作夹带进去：

- 更换 forcing 插值算法；
- 引入 Kahan / pairwise / reproducible summation；
- 调整 CVODE 容差；
- 改物理公式；
- 换线性求解器；
- 修改输出频率或 restart 语义。

这些都属于 B1 之后的独立变更，不能和 strict 并行混在一起。

### 原则 4：CVODE 内部并行晚于 RHS 并行

SUNDIALS/CVODE 支持 serial、MPI、OpenMP、Pthreads 等多种 `N_Vector` 实现，也支持多种 Krylov 线性迭代方法。见 CVODE 文档：https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html

但一旦打开 OpenMP `N_Vector`，norm、dot product、error test 等内部 reduction 顺序都可能变化，自适应步长路径也可能变化。因此必须先做到：

```text
RHS 并行 bitwise identical
完整 run + serial CVODE bitwise identical
```

然后才能进入 CVODE vector/linear solver 并行。

## 6. 推荐交付物

最终建议至少形成以下交付物：

| 交付物 | 内容 |
|---|---|
| B0 benchmark set | 小/中/大流域，含 lake/non-lake、dry/wet、边界条件和源汇项 |
| RHS snapshot harness | 固定 t、Y，导出所有关键 flux 和 DY 数组 |
| B1 reference result | 单线程 parallel-ready 实现的锁定输出 |
| deterministic topology manifest | segment/riv/ele/lake/upstream adjacency list 及排序规则 |
| strict OpenMP report | 每个阶段与 B1 的 bitwise 对比报告 |
| production tolerance report | P-prod 与 B1 的容差、守恒和性能报告 |
| risk register | 每阶段风险、触发条件和回滚方案 |
