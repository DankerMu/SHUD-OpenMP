# Glossary

SHUD 全耦合水文模型的 OpenMP 并行加速工程。本表统一改造过程中**基线、精度等级、阶段门控、数值求解器、代码结构**的术语；阶段路线与实施细节以 `SHUD_openMP_master_plan.md` 为唯一权威，本表只定义"是什么"。

## Language

### 基线 (Baselines)

**B0**:
当前 SHUD 原样编译的单线程结果，锁定编译环境后直接跑；所有后续改动的对照起点。

**B1a**:
S0–S4 纯结构重构后的单线程基线（统一 RHS core、拆 side-effect、固定拓扑，不修 bug）。必须与 B0 **bitwise identical**，否则即重构 bug。

**B1b**:
在 B1a 上完成 bug fix（S5–S6）后的 parallel-ready 单线程基线；并行阶段的唯一长期对照。

**P-strict**:
strict OpenMP 并行结果（P1–P7：RHS 内并行、CVODE 仍用 serial N_Vector）；目标与 B1b bitwise identical。

**P-prod**:
production 并行结果（P8–P9：放开 CVODE 内部并行）；允许与 B1b 有可解释的容差内差异，须 deterministic 可复现。

**B0_vs_B1b_report / B1a_vs_B1b_report**:
基线之间差异的逐项归因报告（每个差异对应哪个 bug fix、影响范围、验收指标）。

**B1b_CHANGELOG.md**:
B1b 相对 B1a 所含 bug fix 的清单。

### 精度等级 (Accuracy Levels)

**A0–A5**:
六级（含 A1b/A3a/A3b/A3c）精度验收阶梯，从单线程可重复（A0）到物理可接受（A5），逐阶段绑定。

**A3a**:
同线程数下完整 CVODE run 与 B1b **bitwise identical**、CVODE stats 一致；P7 强制门控。

**A3b**:
跨线程数 `max_ulp(DY) ≤ 4` 且 `max_abs_diff(state) < 1e-12`、内部步数差 ≤ 0.1%；P7 强制，给跨线程差异的工程上界。

**A3c**:
跨线程数完整 run bitwise identical；P7 仅加分项，不进 go/no-go。

**A4**:
deterministic 容差等级；阈值在 P7 通过后基于 B1b 实测标定，**不预设**。

**bitwise identical**:
逐位相同的浮点输出，本工程对"等价"的最高标准。
_Avoid_: 一致、相同、接近、几乎相等

**ULP**:
units in the last place，浮点最小可分辨差；跨线程差异以 ULP 为单位设上界。

### 阶段与门控 (Stages & Gates)

**预并行 (S0–S6)**:
整理单线程代码、产出 B1a/B1b 的阶段；只改结构与修 bug，不引入并行。

**strict 并行 (P1–P7)**:
逐步开 RHS 内 OpenMP、要求对 B1b bitwise 的阶段。

**production 并行 (P8–P9)**:
放开 CVODE 内部并行、追求最大性能的阶段。

**门控 (go/no-go)**:
每阶段的验收 checklist；不通过不进入下一阶段。

**S0.12**:
RHS 占总 wall-clock 比例 `f` 的强制 profile gate；`f` 决定 P1–P9 优先级与 Amdahl 加速比上限。

**OMP_CUTOFF**:
小流域阈值（编译期/运行期可配，默认 1024）；`NumEle < OMP_CUTOFF` 时 RHS 强制走 serial。

**目标部署平台**:
加速比 go/no-go 的唯一验收平台（单插槽 8 物理核 x86_64 Linux + `-O2 -ffp-contract=off -fopenmp` + 绑核选项）。
_Avoid_: 把本地 Apple Silicon 跑的数字当验收依据

### 数值与求解器 (Numerics & Solver)

**RHS core**:
唯一的右端项计算核（原则 C1）；serial 与 OpenMP 只是它的 execution policy。
_Avoid_: f_loop / f_loop_omp 双路径并存

**CVODE / SUNDIALS**:
驱动 SHUD 时间积分的刚性 ODE 求解器及其库（本工程 pin v6.0.0）。

**N_Vector**:
SUNDIALS 的向量抽象；本工程有 serial 与 OpenMP 两种后端，strict 阶段须用 serial。

**SPGMR**:
基线线性求解器（matrix-free GMRES，无预条件器，Krylov 维度 maxl=5）。
_Avoid_: 误称 dense solver

**matrix-free / Jacobian DQ**:
用差分近似 `Jv ≈ (f(y+σv)−f(y))/σ` 而不显式构造 Jacobian；故每次 Krylov 迭代调一次完整 RHS。

**nfe / nfeLS**:
RHS 调用计数。nfe = 积分器自身调用数；nfeLS = Krylov 线性求解引发的 RHS 调用数。详见下条 §CVODE canonical 15-key set。

**CVODE canonical 15-key set**:
B0 / B1a / P-strict bitwise neutrality + CVODE stats invariance gate 用的 **15 个 canonical stat key**（全部来自 SUNDIALS 6.0.0 CVODE / CVSpils API 的 `PrintFinalStats` 输出，写入 `benchmarks/<case>/B0_output/cvode_stats.txt` 归档；任一键漂移即视为门控 FAIL，即使 `.dat` 输出 SHA256 全等）。15 个键完整集合：

| Key | SUNDIALS API | 语义 |
|---|---|---|
| `nfe` | `CVodeGetNumRhsEvals` | 积分器自身 RHS 调用数 |
| `nfeLS` | `CVSpilsGetNumRhsEvals` | Krylov 线性求解引发的 RHS 调用数 |
| `nni` | `CVodeGetNumNonlinSolvIters` | 非线性求解迭代数 |
| `nli` | `CVSpilsGetNumLinIters` | 线性 (Krylov) 求解迭代数 |
| `nsetups` | `CVodeGetNumLinSolvSetups` | 线性求解器 setup 次数 |
| `netf` | `CVodeGetNumErrTestFails` | 误差控制 step rejection 次数 |
| `nst` | `CVodeGetNumSteps` | 总积分步数 |
| `npe` | `CVSpilsGetNumPrecEvals` | preconditioner evaluation 次数 |
| `nps` | `CVSpilsGetNumPrecSolves` | preconditioner solve 次数 |
| `ncfn` | `CVodeGetNumNonlinSolvConvFails` | 非线性求解 convergence failure 次数 |
| `ncfl` | `CVSpilsGetNumConvFails` | 线性求解 convergence failure 次数 |
| `lenrw` | `CVodeGetWorkSpace` | 实数 workspace 长度 |
| `leniw` | `CVodeGetWorkSpace` | 整数 workspace 长度 |
| `lenrwLS` | `CVSpilsGetWorkSpace` | 线性求解器实数 workspace 长度 |
| `leniwLS` | `CVSpilsGetWorkSpace` | 线性求解器整数 workspace 长度 |

**为什么 `nFCall` 不在 canonical 15-key 内**：`nFCall` 是 SHUD 侧在 `f.cpp::f()` 入口处单独 instrumented 的 RHS 调用 counter（不来自 SUNDIALS API），未写入 `cvode_stats.txt` 归档，由独立 capability 跟踪（F19 修订）。canonical 15-key 与 SUNDIALS API 一一对应、稳定可移植，nFCall 作为 SHUD 内部诊断口径不进入 cross-version invariance gate。

**SUNDIALS 6.0.0 API 名注**：上表中 6 个 `CVSpils*` 名是 SUNDIALS legacy alias（兼容保留），SHUD 6.0.0 实际调用的是 unified `CVodeGet*Lin*` API（见 `SHUD/src/Equations/cvode_config.cpp::PrintFinalStats`：`CVodeGetLinWorkSpace` / `CVodeGetNumLinIters` / `CVodeGetNumPrecEvals` / `CVodeGetNumPrecSolves` / `CVodeGetNumLinConvFails` / `CVodeGetNumLinRhsEvals`）。两套 API 返回同一 counter，bitwise neutrality 不受影响；grep SHUD 源码用 unified 名。

**唯一 enforcement point**：`tools/cvode_stats_diff/cvode_stats_diff.sh` exit code 0 = 15 键全等，任一键缺失 / 数值不等 → exit 非零 + `MISMATCH key=<k> expected=<gold> got=<new>` 报告。SUNDIALS 升级时只改本条目 + diff tool key list，不改 spec。
_Avoid_: 在 spec 内枚举 15 个 key 名（重复 10 处即 entropy；引用本条目即可）

**预条件器 (preconditioner)**:
降低 Krylov 迭代数（nfeLS）的算子；P8-precond 为 P8 第一优先。

**KLU**:
稀疏直接求解器候选；P8-KLU 需构造 sparsity + colored FD Jacobian，再与 preconditioned SPGMR 做 A/B。

**tree reduction**:
确定性的并行归约顺序；production 阶段用以保证可复现的求和。

### 代码结构与数据布局 (Code Structure & Data Layout)

**ExecPolicy**:
把"算什么"与"串行/并行怎么算"解耦的执行策略接口；S1d 引入。

**_OPENMP_ON**:
现有单一宏，耦合了 N_Vector 数据访问、RHS 执行路径、N_Vector 后端三个关注点；待解耦。

**三正交宏**:
替代 _OPENMP_ON 的 `SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS`，各控一个关注点。

**compute / gather 分离 (C2)**:
通量计算只写唯一 slot，汇总由固定顺序的 gather 完成。

**owner-local gather**:
用预构建 adjacency list、按 owner 固定顺序做的确定性汇总；`PassValue()` 的并行安全重构目标。

**PassValue()**:
当前 gather 实现（在 f_loop 末尾清零并重累加 river/ele 通量）。

**死 `+=`**:
被 `PassValue()` 覆盖、实际不影响结果的冗余共享写（如 fun_Seg_surface/sub）；S3 删除。

**SoA / AoS**:
数据布局。热字段从 fat-AoS `_Element` 抽成 SoA 以降 cache miss、提升 prefetch 命中。

**ElementHotData**:
RHS 热路径只读字段的 SoA 容器（S5d）。

**first-touch / NUMA**:
Linux 内存页归属策略；主线程串行初始化会让数据页落单 node，多插槽时跨 NUMA 访问变慢；S5d 做 parallel first-touch。

**OMP_PROC_BIND=close / OMP_PLACES=cores**:
线程绑核的运行时设置；目标部署平台的最低要求，禁止线程迁移。

### 验收工具与产物 (Tooling & Artifacts)

**RHS snapshot**:
在指定 `t_values` 处把 DY/flux 数组 dump 到二进制的工具（`tools/rhs_snapshot/`）。

**compare_snapshot**:
二进制 bitwise diff + 人类可读 ULP report、有差异时返回非零 exit code 的比对工具（`tools/compare_snapshot/`）。

**benchmark 算例**:
S0 注册的 ≥5 类标准算例（各带 `manifest.yaml`）；所有基线与精度对比的固定输入。
