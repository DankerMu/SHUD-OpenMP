# 05 精度、一致性与回归验收标准

本节回答：**每个阶段的精度要达到什么程度？**

核心思想：

```text
B0/B1/strict 阶段：以 bitwise identical 为主；
production 阶段：以 deterministic tolerance 为主；
物理精度改进：必须单独立项，不混入并行验收。
```

## 1. 为什么不能只用 NSE/KGE 判断并行是否正确

NSE、KGE、峰值误差、水量平衡等指标适合判断水文模拟是否可接受，但不适合判断并行实现是否正确。原因是：

1. 很多并行 bug 只影响局部数组，短期 hydrograph 未必明显；
2. CVODE 自适应步长会放大小差异，使定位困难；
3. 水文指标可能掩盖 lake、river、groundwater 某个分量的错误；
4. 并行实现应先证明“同一 RHS”，再讨论“水文上是否等价”。

因此验收应分层。

## 2. 验收层级

### L0：自复现

同一版本、同一二进制、同一输入，多次运行结果一致。

适用：B0、B1、P-strict、P-prod。

### L1：RHS bitwise identical

固定 `t` 和 `Y`，直接比较 RHS 输出和所有关键中间数组。

适用：S1–S8、P1–P7。

### L2：完整 run bitwise identical

完整 CVODE run 的输出、CVODE stats、RHS call count 完全一致。

适用：B1 与 B0 的纯重构比较，P1–P7 与 B1 比较。

### L3：deterministic numerical tolerance

不再要求 bitwise identical，但要求同一配置多次运行 deterministic，与 B1 的差异在容差内。

适用：P8、P9。

### L4：hydrological acceptance

比较 NSE、KGE、水量平衡、峰值、峰现时间、状态变量误差等。

适用：P8、P9，以及后续物理精度路线。

## 3. B0 阶段验收

| 项 | 标准 |
|---|---|
| 单线程重复运行 | bitwise identical |
| 输出文件 | 内容一致，忽略时间戳/路径 metadata |
| CVODE stats | identical |
| RHS snapshot | 自身比较 identical |
| 性能记录 | wall-clock、RHS 次数、I/O 时间、CVODE stats 可复现到报告层面 |

如果 B0 不能自复现，不能进入后续改造。

## 4. 单线程预优化 S1–S8 验收

### 4.1 纯重构

包括：唯一 RHS core、函数拆分、接口调整、diagnostics 重构。

标准：

```text
RHS arrays bitwise identical with B0；
full run bitwise identical with B0；
CVODE stats identical；
performance 可改善但不是验收重点。
```

### 4.2 deterministic gather 拆分

如果 gather 顺序与 B0 完全一致：

```text
bitwise identical。
```

如果顺序改变：

```text
先建立 B1；
记录 B0→B1 ULP / abs / rel 差异；
证明水量平衡不恶化；
后续 strict 并行只对 B1 验收。
```

### 4.3 forcing cache

标准：

```text
old_getX(t, col) == cached_getX(t, col) bitwise。
```

并在完整 run 中确认输出与 B0/B1 bitwise identical。

### 4.4 intentional bug fix

如果修复 serial 代码中明确 bug，必须走变更门槛：

| 检查 | 标准 |
|---|---|
| 局部 RHS 差异 | 必须定位到具体数组和公式 |
| 完整 run 差异 | 必须可解释 |
| 水量平衡 | 不得恶化；若改变，应解释为物理修正 |
| 输出指标 | NSE/KGE/peak/baseflow 等不应异常退化 |
| 记录 | `B1_CHANGELOG.md` 必须说明 |

## 5. strict 并行 P1–P7 验收

P1–P7 的目标是证明“并行没有改变 B1”。因此验收标准应非常严格。

### 5.1 RHS snapshot 标准

对以下数组要求 bitwise identical：

```text
DY
uYsf/uYus/uYgw/uYriv/uYlake
qElePrep/qEleNetPrep/qEleInfil/qEleExfil/qEleRecharge
qEleETP/qEleETA/qEs/qEu/qEg/qTu/qTg
QeleSurf/QeleSub/QeleSurfTot/QeleSubTot
QsegSurf/QsegSub
Qe2r_Surf/Qe2r_Sub
QrivSurf/QrivSub/QrivUp/QrivDown
QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut/qLakeEvap/qLakePrcp
```

标准：

```text
max_abs_error = 0
max_rel_error = 0
max_ulp = 0
first_mismatch = none
```

### 5.2 完整 run 标准

| 项 | 标准 |
|---|---|
| 输出文件内容 | bitwise identical |
| CVODE internal steps | identical |
| RHS call count | identical |
| error test failures | identical |
| linear solver setup/iteration stats | identical，如果该阶段未换 solver |
| water balance | identical |

### 5.3 编译和运行环境要求

strict 阶段建议：

```text
schedule(static)
固定线程数
禁用 dynamic/guided schedule
禁用 fast-math
禁用非受控 FMA contraction
固定 SUNDIALS 版本
固定输出频率
固定随机/无随机
```

### 5.4 不接受的情况

strict 阶段不接受：

- “只差 1 ULP”；
- “hydrograph 看起来一样”；
- “NSE/KGE 没变”；
- “多跑几次大概一样”；
- “atomic 解决了 data race 所以正确”。

这些都只能作为 production 阶段讨论，不能通过 strict 验收。

## 6. production 并行 P8–P9 验收

P8/P9 允许引入 CVODE OpenMP N_Vector、Krylov/KLU、预条件器、pairwise/Kahan/reproducible summation 等。这些改变可能使结果不再 bitwise identical。

### 6.1 deterministic 要求

同一配置必须可复现：

```text
same binary + same input + same thread count + same schedule
→ repeated runs deterministic
```

理想情况下 P-prod 同配置 bitwise identical；如果无法完全 bitwise，也必须给出稳定误差范围和原因。

### 6.2 与 B1 的状态误差标准

建议用 CVODE 容差尺度定义状态误差门槛：

```text
|Y_prod_i - Y_B1_i| <= C * (abstol_i + reltol_i * |Y_B1_i|)
```

其中：

- C 初始可取 10–50，用于 production solver 改造早期；
- 如果 SHUD 仍使用 scalar `abstol/reltol`，先按当前 `Control_Data` 中的 scalar 容差计算；
- 更推荐后续改为 vector absolute tolerance，按 surface/unsat/gw/river/lake 不同量纲分别设定。

当前 `Control_Data` 中有 scalar `abstol` 和 `reltol` 控制项，源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/Model_Control.hpp

### 6.3 水量平衡标准

建议同时看两类指标：

```text
production 水量平衡 residual 不得显著大于 B1；
production 与 B1 的 cumulative water balance 差异应小于设定阈值。
```

初始门槛建议：

```text
Δ water balance residual <= max(10% * B1 residual, 0.01% of cumulative precipitation/input)
```

实际项目可根据流域尺度、输出频率和观测误差调整。

### 6.4 hydrograph 指标标准

对纯数值/并行优化，建议初始门槛：

| 指标 | 建议门槛 |
|---|---|
| ΔNSE | <= 0.001–0.005 |
| ΔKGE | <= 0.001–0.005 |
| peak flow relative difference | <= 0.1%–0.5% |
| peak timing difference | <= 1 output interval |
| baseflow relative difference | <= 0.1%–0.5% |
| runoff volume difference | <= 0.1% |

这些不是物理率定标准，而是“数值改造不应明显改变水文行为”的工程门槛。

### 6.5 CVODE stats 标准

P8/P9 应记录：

```text
number of internal steps
number of RHS evaluations
number of nonlinear iterations
number of error test failures
number of linear solver setups
number of linear iterations
number of convergence failures
```

如果 production solver 比 B1 快但 error test failures、nonlinear failures 明显上升，应视为风险，不应直接接受。

## 7. 各阶段精度目标总表

| 阶段 | 对照 | 目标精度 | 是否允许 bit 差异 |
|---|---|---|---|
| B0 | 自身 | 重复运行 bitwise identical | 否 |
| S1 RHS core | B0 | bitwise identical | 否 |
| S2 语义对齐 | B0/B1 | 纯继承 serial 则 bitwise；bug fix 需记录 | 原则上否 |
| S3 flux/gather 拆分 | B0/B1 | fixed order bitwise | 原则上否 |
| S4 topology | B0/B1 | bitwise | 否 |
| S5 scratch/ownership | B0/B1 | bitwise | 否 |
| S6 forcing cache | B0/B1 | getX + full run bitwise | 否 |
| S8 B1 | 自身 | 重复运行 bitwise identical | 否 |
| P1 update 并行 | B1 | RHS/full run bitwise | 否 |
| P2 vertical 并行 | B1 | RHS/full run bitwise | 否 |
| P3 edge flux 并行 | B1 | RHS/full run bitwise | 否 |
| P4 segment flux 并行 | B1 | RHS/full run bitwise | 否 |
| P5 gather 并行 | B1 | RHS/full run bitwise | 否 |
| P6 applyDY 并行 | B1 | RHS/full run bitwise | 否 |
| P7 full RHS OpenMP | B1 | full run bitwise | 否 |
| P8 CVODE/solver 并行 | B1 | deterministic tolerance | 是 |
| P9 production summation | B1 或 new numerical reference | deterministic tolerance | 是 |

## 8. 最终建议

对 SHUD 而言，最稳妥的验收顺序是：

```text
先 RHS bitwise
再 full run bitwise
再 production tolerance
最后 hydrological acceptance
```

不要一开始就用水文指标替代 RHS 级别的精确比较。否则并行 bug、浮点重排、CVODE 自适应差异和物理过程修复会混在一起，后续很难定位。
