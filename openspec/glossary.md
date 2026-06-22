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
小流域阈值（编译期/运行期可配，默认 1024）；`NumEle < OMP_CUTOFF` 时 RHS 强制走 serial（P1+ RHS 内并行启用后）。理由：小流域线程启动开销 > 并行收益；keliya (NumEle=484) / xinanjiang_upstream (NumEle=801) 等小 case 不进 RHS parallel for。落地于 P1+ ExecPolicy 调度；B1b 阶段（serial-only）不触发。
_Avoid_: 把 OMP_CUTOFF 当作绝对阈值（与硬件 cache 大小 + thread spawn latency 相关，可通过 `-DSHUD_OMP_CUTOFF=N` 编译期 或 `SHUD_OMP_CUTOFF=N` 运行期环境变量 覆盖 — master plan L1933；尚未在 SHUD/src 落地，P1+ ExecPolicy 引入时实施）

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
RHS 调用计数。nfe = 积分器自身调用数；nfeLS = Krylov 线性求解引发的 RHS 调用数。

**nFCall**:
SHUD 内部 RHS kernel 入口自增计数（`Model_Data::nFCall`，`Model_Data.hpp` L58；S5c-C #175 落地）。语义与 CVODE `CVodeGetNumRhsEvals` 的 `nfe` **严格分离**（design D10）：`nfe` = CVODE 主动调用 RHS 次数（进 15-key invariance gate）；`nFCall` = SHUD f.cpp:61 `MD->nFCall++` kernel 入口计数（不进 15-key gate，仅作 §C8 RHS 调用归一指标独立追踪）。`nFCall != nfe` 时 `B1b_CHANGELOG.md` 必须含一行解释（CVODE 内部 Jacobian DQ / step retry 触发额外 internal RHS 调用），**无数值上限阈值**（free-running counter，差异本身不阻 CI）。
_Avoid_: 把 nFCall 和 nfe 当同一指标；把 nFCall 写入 15-key snapshot

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
RHS 热路径只读字段的 SoA 容器，S5d.1 引入，位于 `SHUD/src/ModelData/MD_layout.hpp`。包含 32 个 hot 字段（geometry 6 [Triangle] + topology 7 [AttriuteIndex] + _Element direct 14 + Geol_Layer 1 [Sy] + Landcover 4），覆盖 `MD_ElementFlux.cpp` / `MD_f.cpp` / `MD_ET.cpp` 三个 RHS TU 全部 `Ele[<expr>].<field>` 访问点。字段集 source-of-truth = `docs/s5d_hot_fields.yaml`；与 `_Element` AoS 双轨保留（详 ADR-0001）；`Model_Data::sync_hot_dynamic(i)` inline 同步 4 个动态字段。
_Avoid_: 直接 `Ele[i].nabr` 访问；hot path 必须 `hot.nabr_flat[3*i+j]`

**RiverHotData**:
若 river / segment 热字段进 SoA 抽取，则使用此名命名 SoA 容器。**当前状态**：S5d.2-5b (#180) audit 结论 = 不存在 `double[N]` / `double*` member field（`_River` / `RiverSegement` 全是 plain scalar），SoA fold-in trigger 未触发，**不实现** RiverHotData 容器。未来若 `Riv[i]` / `RivSeg[i]` 内部增加数组字段且被 RHS hot path 高频访问，需重新评估并在新 ADR 中记录。详 `docs/topology_manifest.yaml` `s5d2_riv_audit` 节 + `SHUD/B1b_CHANGELOG.md` S5d.2-5b 段。

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
