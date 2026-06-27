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
RHS 热路径只读字段的 SoA 容器，S5d.1 引入，位于 `SHUD/src/ModelData/MD_layout.hpp`。包含 32 个 hot 字段（geometry 6 [Triangle] + topology 7 [AttriuteIndex] + _Element direct 14 + Geol_Layer 1 [Sy] + Landcover 4），覆盖 `MD_ElementFlux.cpp` / `MD_f.cpp` / `MD_ET.cpp` 三个 RHS TU 全部 `Ele[<expr>].<field>` 访问点。字段集 source-of-truth = `docs/b1b/s5d_hot_fields.yaml`；与 `_Element` AoS 双轨保留（详 ADR-0001）；`Model_Data::sync_hot_dynamic(i)` inline 同步 4 个动态字段。
_Avoid_: 直接 `Ele[i].nabr` 访问；hot path 必须 `hot.nabr_flat[3*i+j]`

**RiverHotData**:
若 river / segment 热字段进 SoA 抽取，则使用此名命名 SoA 容器。**当前状态**：S5d.2-5b (#180) audit 结论 = 不存在 `double[N]` / `double*` member field（`_River` / `RiverSegement` 全是 plain scalar），SoA fold-in trigger 未触发，**不实现** RiverHotData 容器。未来若 `Riv[i]` / `RivSeg[i]` 内部增加数组字段且被 RHS hot path 高频访问，需重新评估并在新 ADR 中记录。详 `docs/topology_manifest.yaml` `s5d2_riv_audit` 节 + `SHUD/B1b_CHANGELOG.md` S5d.2-5b 段。

**first-touch / NUMA**:
Linux 内存页归属策略；主线程串行初始化会让数据页落单 node，多插槽时跨 NUMA 访问变慢；S5d 做 parallel first-touch。**M10 修订（P1d epic close, 2026-06-24）补充**：本术语描述 allocation-time first-touch（一次性 page-fault 初始化模式），P1d 保留有效。P1d 期间 PR-C/D/E 引入的 steady-state first-touch（每步 RHS warm-up loops）则因 owner-compute 未实现（StrictOMP `std::abort` 桩 + Serial RHS path）标 DEPRECATED；详 `**steady-state first-touch (P1d)**` 新术语条目。

**OMP_PROC_BIND=close / OMP_PLACES=cores**:
线程绑核的运行时设置；目标部署平台的最低要求，禁止线程迁移。

### 验收工具与产物 (Tooling & Artifacts)

**RHS snapshot**:
在指定 `t_values` 处把 DY/flux 数组 dump 到二进制的工具（`tools/rhs_snapshot/`）。

**compare_snapshot**:
二进制 bitwise diff + 人类可读 ULP report、有差异时返回非零 exit code 的比对工具（`tools/compare_snapshot/`）。

**benchmark 算例**:
S0 注册的 ≥5 类标准算例（各带 `manifest.yaml`）；所有基线与精度对比的固定输入。

### P1 first-parallel-candidate baseline 集合

P1 epic (#211) 完成后由 PR-N #226 PROMOTE 入册的术语集合（A3a / A3b / A3c 已在 §"精度等级" 节定义，不重复）。

**P1-update-omp-tag**:
P1 epic capstone tag。annotated tag object SHA `ff21c75c8e968d5e47ca53b015425360be9ac879` deref commit `003f58dc079116ef2161d2f96006228ef0e013d0`（≡ PR-K2 #223 capstone log append + main HEAD 时刻）；SHUD submodule pin `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（`openmp-baseline` 分支 HEAD on `SHUD-System/SHUD`，PR-F #218 lake pragma capstone）。D11 immutable：一次锁死禁止 force-update（与 B1a-tag force-update 历史**不同**，与 B1b-tag 一致）；任何后续 retroactive 更新走 forward-compat `P1c-tag stacking` / `P2-* stacking` 路径（master plan C8）。
_Avoid_: 把 P1-update-omp-tag 理解成 lightweight tag 或允许 force-update

**baseline/P1**:
P1 epic 完成后从 `main` 分出的 frozen baseline 分支，HEAD ≡ P1-update-omp-tag deref commit `003f58d`。D11 protection rule enforced：`lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`（与 baseline/B1b 一致）。后续 P2+ 工作从 `main` 分新分支，不再打 `baseline/P1`。仅作历史比对参照（vs B1b / vs P-strict / vs P-prod）。
_Avoid_: 把 baseline/P1 当作活动开发线 / 误以为可 push 新 commit

**3-pragma stack**:
`SHUD/src/ModelData/MD_update.cpp` `f_update()` 三 owner loop 上的 `#pragma omp parallel for schedule(static) default(none) shared(<显式列出>) private(i)` 三个并列 pragma 的总称；落地于 PR-D #216 (element loop L64-L105) + PR-E #217 (river loop L107-L125) + PR-F #218 (lake loop L136-L147)，SHUD pin trail `017c629 → 6a9e684 → 08898a3 → 07c677f`。**禁止** `schedule(dynamic|guided)` / `#pragma omp atomic` / floating `+=` / `reduction(+:sum)`（§8.1 strict 禁止项）。3 pragma 是 P1 候选 commit 的全部 SHUD 数值代码改动。
_Avoid_: 误以为 reset loops (L127-L135 NumRiv+NumEle 闲置) 或 DY zero loop (L148-L150 NumY-bounded) 也并行（本 change 不并行）；把 `f_updatei` case 1-5 五 loop 当作 3-pragma stack 的一部分（NG7：`f_updatei` 留 P2a reviewer 参考）

**owner-local update**:
P1 三 pragma 的安全前提：每 elem/river/lake by-owner write 无 floating reduction（无 `+=` / `reduction(+:sum)` / `atomic`），写目标按 `i` 索引（disjoint slots：`QeleSubAt(i,j)` = `flat[3i+j]` / `Ele[i].QBC` distinct &Ele[i] / `Riv[i]` distinct / `lake[i]` distinct），无 cross-iter 依赖（无 `Ele[i+k]` / `Riv[neighbor]` / `lake[j!=i]` 读）。是 PR-C #215 P1.0 pre-audit 5 函数 (`_Element::updateElement` / `_River::updateRiver` / `_Lake::update` + `f_updatei` case 1-5) 全部 (a) safe verdict 的根据，也是 owner-local + schedule(static) 在 N=2 vs N=1 同 binary A3a bitwise PASS 的根因（PR-K2 #223 实证）。
_Avoid_: 把 owner-local update 与 owner-local gather 混淆（前者写 RHS 内部 state 数组，后者汇总跨 element 的 flux 到 owner slot；详 `compute / gather 分离 (C2)` 与 `owner-local gather`）

**CVODE nst bifurcation**:
CVODE adaptive stepper 在不同 OMP 调度下出现的内部步数分歧现象。实证案例：PR-K2 #223 server heihe N=1/2/4/8 NUM_OPENMP scaling 测试，CVODE `nst` 实测 = 6773/6773/6585/6684（N=1 与 N=2 同步数；N≥4 出现分歧）；伴随 4-cell A3a + A3b dual-FAIL（max_abs_diff 4-20e5 / max_ulp ~9e18 / n_diff 98.4-98.7%）。表征 N>2 reduction-tree depth transition 触发的浮点重排，**不是** PR-D/E/F 三 pragma owner-local 设计缺陷（同一 binary N=2 vs N=1 是 A3a bitwise PASS 的）。Reviewer 根因假设排序：(1) B1b S2 P3-P5 owner-local gather tree-reduction；(2) GCC 13.3.0 auto-vectorization chunk-dependent FMA；(3) CVODE SPGMR norm OMP residual。
_Avoid_: 把 CVODE nst bifurcation 归咎给 PR-D/E/F 三 pragma；把它当作 P1 lock blocker（per design D5 NG3 + spec L184-L209 dual-FAIL Scenario，不阻 P1 lock）

**P7 final-fusion deterministic-reduction**:
master plan §6 P7 stage 目标：fork-join 最小化 + chunk-fixed `schedule(static)` 消除 N-dependent reduction tree depth transition，使跨 N 跨线程数 A3a bitwise / A3b ULP≤4 全部 PASS。是 P1 阶段 CVODE nst bifurcation + N≥4 dual-FAIL 的 forward debt。P1 阶段不展开实施；spec p1-state-update-parallel A3a + A3b dual-FAIL Scenario 显式 cross-ref 此 work scope。
_Avoid_: 把 P7 final-fusion 与 P1 stage 强行并轨 (NG3 per design D5)；P1 阶段 strict A3a-cross-thread bitwise 不强制

**§1.1.1 WARNING**:
P1 epic capstone verdict 状态。其 carve-out 含义：(a) wall/speedup Amdahl-bound for IO-dominant case (heihe sp@8 实测 1.08× ~ 1.13× IO Amdahl 上限) + below P7 strict target for first OMP candidate (heihe_x4 sp@8 实测 1.14× < P7 strict M=1.8×/T=2.2×)；(b) N≥4 cross-thread A3a + A3b dual-FAIL (P7 final-fusion deterministic-reduction debt)。**non-blocking** per design D5 NG3 + master plan §6 P7 final-fusion scope + spec p1-state-update-parallel L184-L209 dual-FAIL Scenario；P1 lock 已通过 PR-H/I/J 3 anchor (NUM_OPENMP=1 vs B1b bitwise 24/24 PASS) 完成 forced gate。
_Avoid_: 把 §1.1.1 WARNING 当作 P1 epic 失败标记；把 P1 stage carve-out 解读为永久豁免（P7 阶段 SHALL 解决）

### P1c deterministic-reduction baseline 集合

P1c epic (#243) 完成后由 PR-M (TBD) PROMOTE 入册的术语集合。

**P1c-tag**:
P1c epic capstone tag。annotated tag object SHA `1da5eb9734680fc61e68f6091964c38fc5f67c6f` deref commit `4b8c60af261e0d1517f52702e4827a4e2d67dd41`（≡ PR-L #268 capstone log append + baseline/P1c HEAD post-PR-L 时刻）；SHUD submodule pin `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4`（`openmp-baseline` 分支 HEAD on `SHUD-System/SHUD`，PR-I #265 Kahan injection on top of post-PR-E helper-wrap）。D11 immutable：一次锁死禁止 force-update（与 P1-update-omp-tag 一致）；任何后续 retroactive 更新走 forward-compat `P2-* stacking` 路径（master plan C8）。
_Avoid_: 把 P1c-tag 理解成 lightweight tag 或允许 force-update；把 P1c-tag 误以为 supersede P1-update-omp-tag（两 tag 并存 D11 immutable）

**baseline/P1c**:
P1c epic 活动开发线分支（PR-A..PR-M base）。Lock prep PR-L #268 + 实际 lock 在 PR-M post-merge 执行。lock 后 HEAD frozen at PR-M 合并 SHA；SHUD pin `3a0004c`（Kahan-injected）。后续 P2+ 工作从 `main` 分新分支，不再打 `baseline/P1c`。仅作历史比对参照（vs P1 / vs P-strict）。**与 baseline/P1 关系**：两 baseline 并存 D11 immutable；P1c **不取代** P1，而是 P1 stage 的 sub-epic capstone（per master plan §6 P1c 子节）。
_Avoid_: 把 baseline/P1c 当作活动开发线（lock 后不允许 push）；把 baseline/P1c 当 supersede baseline/P1 的 successor branch

**4 helpers (fixed-shape canonical reduction)**:
`SHUD/src/Model/MD_rhs_core.cpp` 中 4 个 `static inline` helper functions（`fixed_pairwise_sum_range` / `fixed_pairwise_sum_indexed` / `fixed_leftfold_sum_indexed` / `fixed_leftfold_sum_pair_indexed`）的总称，由 PR-B/C/D/E #258/#259/#260/#261 落地，cover 10 line anchors → 8 logical sites（spec p1c-deterministic-reduction §"全 RHS reduction 站点 grep 清单完整覆盖" + §"MD_rhs_core.cpp 8 reduction 站点 fixed-shape pairwise 改造" Requirements）。helper-wrap **bitwise-equivalent at NUM_OPENMP=1**（server PR-J #266 §2 实证：P1 era N=1 SHA `7f22bd6f...` ≡ P1c pre-Kahan N=1 SHA `7f22bd6f...` heihe / `55403bef...` heihe_x4）。**Kahan-aware variant** 通过 Neumaier compensation 引入（per `Kahan held-in-reserve patch`）。
_Avoid_: 把 helper-wrap 与 OpenMP `reduction(+:sum)` 混淆（helpers 内部仍 serial，OMP 写在调用端）；把 helper-wrap 当作完整 A3a closure（需 Kahan + NUMA 双重补偿）

**Kahan held-in-reserve patch**:
`docs/p1c_kahan_patch.diff` (PR-G #263) — git-apply 风格 patch file，记录 Neumaier 1974 (Kahan-Babuška variant) compensation 在 4 helpers 中的注入。**Held-in-reserve** = patch 作 documentation artifact 存在但 conditional apply：spec p1c-deterministic-reduction Requirement "Kahan 补偿求和兜底 (条件触发, server PR-K2 首跑 FAIL)" 触发时才应用（per `docs/p1c_a3a_root_cause.md` §"Kahan 候选路径" §(c) trigger conditions）。P1c epic 中 PR-H #264 §4.7 trigger fired → PR-I #265 应用 patch → SHUD pin de9545d → 3a0004c。算法：Neumaier 改进版（每 += 注入 floor-comparison + 3 算术 op，保 backward-stable）。Wall-clock 影响实测改善而非 R2 估算 +1-3%（heihe_x4 N=8 -22.9%，per `docs/p1c_perf_baseline.md` §4）— **R2 估算 REFUTED**。
_Avoid_: 把 Kahan held-in-reserve 与 classic Kahan summation 混淆（Neumaier 显式处理 sign-mixing 序列，对站点 5/6/7 sigma 含负数 robust）；把 patch apply 后 SHUD pin 3a0004c 当作 baseline/P1c 的永久 pin（P9 NUMA 治理后可 revert 回 de9545d 等价 pre-Kahan helper-wrap，per `docs/p1c_pr_j_reverse_compat.md` §4 rollback option）

**P1d carve-out (writer noise governance)**:
P1c epic capstone verdict `PARTIAL CLOSURE` 的 forward debt（原命名 `P9 carve-out`，per master plan §6 P1c.7 2026-06-23 修订重命名为 P1d；术语 key 与 status 述语统一为 P1d；narrative 注明源于 P1c epic forward debt）。Per `docs/p1c_summary.md` §5.2 + `docs/p1c_pr_i_kahan_injection.md` §8 + design D9 decision branch 2 CONFIRMED：drift origin 不在 P1c 8-site reduction 内部 (Kahan 完全注入仍残 heihe |Δ_nst|=84)，而是上游 parallel writer first-touch / NUMA-affinity 异序写入 ULP 噪声 → gather 站点忠实累加放大。P1d stage SHALL：(a) MD_update.cpp 3 #pragma omp parallel for region (hot.soa / QeleSurf_flat / Ele_AoS) 加 `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `numactl --interleave=all`; (b) 验证 NUMA 治理后 heihe / heihe_x4 4 N SHA 全等 (A3a closure); (c) nst Δ=0 cross-N closure; (d) NUM_OPENMP=1 reverse-compat 恢复 (revert Kahan)。**non-blocking P1c** per master plan §3 fallback option 2 + spec L100-L103 carve-out Scenario。
**Status (P1d epic close, 2026-06-24)**: PARTIAL CLOSURE via E′ containment path. 原 hypothesis (NUMA writer noise 是 P1c §4.4 A3a + §4.5 nst Δ 漂移根因) 在 P1d PR-H 实测 + 两轮 GPT Pro 复查 + codebase 5/5 事实核查后被颠覆：真正根因是 SUNDIALS 6.0.0 NVECTOR_OPENMP `N_VDotProd_OpenMP` `reduction(+:sum) schedule(static)` reduction tree 顺序不固定。详 master plan v1.5 / M10 §6 P1d + `docs/p1d/p1d_numa_root_cause.md`. **CLOSED via E′ closure (master plan v1.5 / M10 §6 P1d.4): P1d-tag a82bf336 push 完成 + 4-mode spec rewrite + first-touch DEPRECATED note + Kahan revert 保留 + forward handoff to P1e (F 路) per ADR-0002 Path 1 SELECTED.**
**Status (P1e epic close)**: CLOSED via P1e epic 2026-06-25，参见 docs/p1e/p1e_summary.md §"P1d carve-out closure"
_Avoid_: 把 P1d carve-out 与 P7 final-fusion deterministic-reduction 混淆（两 forward debt 独立：P7 = fork-join + chunk-fixed schedule；P1d = NUMA + first-touch governance）；把 P1d work scope 误以为可 P2a 阶段解决（master plan §6 显式 P1d 范围）

**P1d-tag**:
P1d epic capstone annotated git tag (created post-PR-L merge 2026-06-24). Tag-object SHA `a82bf336...`, deref commit SHA `f88f2dc2...`. Pins outer `baseline/P1d` HEAD at PR-L commit + SHUD submodule at `210ac19...`. D11 6-tag chain extension (5 historical immutable + P1d-tag NEW): B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag. Tag message documents E′ containment closure narrative (master plan v1.5 / M10 §6 P1d.4). _Avoid_: 把 P1d-tag 当作 strict mode 验收 PASS（仅 PARTIAL CLOSURE via E′；strict 待 P1e）；改写 D11 5 historical SHA（永不变 per design D11）。

**baseline/P1d**:
P1d epic 主分支，从 `baseline/P1c` HEAD 分出，2026-06-24 lock_branch=true + enforce_admins=true post-PR-M。HEAD = P1d-tag^{}. PR base 用 baseline/P1d 直到 P1d-tag lock；之后改 main 或 baseline/P1e。包含 13 PR merge (PR-C0/C/D/E/F/G/H/I + M10 + K + ADR-0002 + L + M; PR-J 在 E′ closure 收口时被 PR-I + 仿 PR-G Mac 9-SHA 取代)。_Avoid_: 在 baseline/P1d lock 后 force-push；从 baseline/P1d 直接分新 P1e epic（应从 main fast-forward 后的 main 分）。

**steady-state first-touch (P1d)**:
PR-C/D/E 在 `SHUD/src/ModelData/MD_update.cpp` 3 处 `#pragma omp parallel for` 区前置插入的 page warm-up loops，原意为 owner-compute (parallel RHS consumer) 做页面预放置。**M10 修订（E′ closure）标 DEPRECATED**——`f.cpp:54` 始终调 `ExecPolicy::Serial` + `MD_rhs_core.cpp:802-811` StrictOMP/ProductionOMP 是 `std::abort()` 桩 → consumer 是单线程，根本无视 NUMA locality → 无效优化 + bandwidth 浪费。P1e (F 路) 删除 steady-state first-touch + 重新设计为真正 owner-compute (replacing abort stubs) 配套。**与 allocation-time first-touch (`Model_Data.cpp::malloc_EleRiv` L251-L346) 区分**：后者保留有效（page-fault 一次性初始化模式）。_Avoid_: 把 steady-state first-touch 与 allocation-time first-touch 当成同一概念；删 allocation-time first-touch（不是 deprecated 的）。

**P1d NUMA env**:
P1d 引入的标准化 NUMA environment 配置，sbatch template 强制：`OMP_PROC_BIND=close` + `OMP_PLACES=cores`；server PR-F 实证后 drop `numactl --interleave=all`（导致 cross-NUMA cache-line ping-pong）。Mac local 同步 `OMP_PROC_BIND=close`。Per `docs/p1d/p1d_numa_env_runbook.md`. _Avoid_: 加 `numactl --interleave=all`（PR-F 已实证有害）；用 `OMP_PROC_BIND=spread`（不同 NUMA node 之间的 ping-pong 更严重）。

### P1e strict-omp F-path baseline 集合

P1e epic (#308) 完成后由 PR-M (#321) PROMOTE 入册的术语集合。

**P1e-tag**:
P1e epic capstone annotated git tag (created post-PR-L #335 merge, 2026-06-25). Tag-object SHA 与 deref commit SHA 见 `docs/p1e/p1e_summary.md` §"验证 P1e-tag"（PR-M post-PR-L-merge amend 填实）。Pins 外层 `baseline/P1e` HEAD 至 PR-L 合并 SHA + SHUD submodule pin `3341368`（PR-H StrictOMP + first-touch removal capstone）。D11 7-tag chain extension (6 historical immutable + P1e-tag NEW): B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag / P1e-tag. Tag message 记录 P1e SHIP-LOCKED narrative + §4.6.2 partial-closure carve-out (heihe small-case 1.066× < 1.3× threshold per OMP_CUTOFF overhead floor)。
_Avoid_: 把 P1e-tag 当作 lightweight tag 或允许 force-update；改写 D11 6 historical SHA（永不变 per design D11）。

**baseline/P1e**:
P1e epic 主分支，从 `baseline/P1d` HEAD 分出 (PR-A 2026-06-24)，PR-M post-merge 由 orchestrator 重新 lock (`lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`)。HEAD = P1e-tag deref commit。包含 13 PR merge (PR-A..PR-M) cover 2×2 build matrix Phase 1/2 + StrictOMP RHS 实施 + thread split + first-touch removal + 3 SHALL gate verification + capstone docs + PROMOTE。后续 P2a 工作从 `main` fast-forward 后的 `main` 分新分支，不再打 `baseline/P1e`。仅作历史比对参照（vs P1c / vs P1d / vs P-strict）。
_Avoid_: 在 baseline/P1e lock 后 force-push；从 baseline/P1e 直接分新 P2a epic（应从 main 分）。

**strict-omp mode**:
master plan §8.1 4-mode 表中的 `strict-omp` 行 = P1e production candidate mode。规范：N_Vector 用 `N_VNew_Serial`（serial N_Vector），RHS 用 `ExecPolicy::StrictOMP`（单 parallel region + 3 phase + implicit barrier + `default(none)` + `schedule(static)`），bitwise gate = server SHALL 跨 N∈{1,2,4,8} bitwise + nst Δ=0 + N=1 reverse-compat strict（Mac SHOULD advisory per libomp 弱保证），Build = `make shud SHUD_ENABLE_OPENMP_RHS=1` (PR-G post-merge: 自动附加 `-fopenmp` + libgomp/libomp)。**与其它 3 mode 区分**：`serial` (B0/B1b baseline，无 OpenMP) / `det-omp` (P1c helper-wrap + Kahan，NVECTOR_OPENMP) / `fast-omp` (NVECTOR_OPENMP + std::abort 桩 RHS，P1d era 现状)。详 `openspec/specs/p1e-strict-omp-rhs/spec.md` Conventions 节。
_Avoid_: 把 strict-omp mode 与 P1d era 4-mode 表中的其它 3 mode 混淆；误以为 `strict-omp` 走 `N_VNew_OpenMP`（实际走 Serial NVec + RHS 内 OpenMP）。

**2×2 build matrix**:
P1e.2 因果实验矩阵 = 4 build × 4 N (∈{1,2,4,8}) × 3 repeats × 4 case = 192 cell（Mac 96 + server 96）。4 build = {A: serial / B: NVECTOR_OPENMP + RHS serial / C: NVECTOR_SERIAL + RHS OpenMP (= production candidate) / D: NVECTOR_OPENMP + RHS OpenMP}。Phase 1 (PR-C/D, mode A/B) confirm NVECTOR_OPENMP 是 cross-N 散度主因；Phase 2 (PR-G post-merge, mode C/D) verify mode C bitwise + 加速比。Cell 数据归档 `docs/p1e/p1e_2x2_experiment.md` + `docs/p1e/p1e_2x2_verdict.md`（含 D12.1/.2/.3/.4 routing 实际触发分支判定）。
_Avoid_: 把 2×2 build matrix 与一般 OpenMP build flag matrix 混淆（本术语专指 P1e.2 因果实验 192 cell scope）；只跑 Phase 1 不跑 Phase 2（PR-G post-merge 必跑 mode C/D 验证 production candidate）。

### p8pre-spike (SPGMR preconditioner identity stub spike) 集合

p8pre-spike epic (SHUD-OpenMP#338) 完成后由 PR-archive #361 (issue #349) PROMOTE 入册的术语集合。Epic 在 P1e ship 后立项，目的是回答 ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) 的三阶段触发条件 (ROI 比 ≥1.5 + 5+1 gate PASS + ADR-0003 GO)。Step 1 (PR-A/B/C) 落地 Mode C profile recheck 量化 ROI；Step 2 (PR-D/E/F) 落地 identity precond API wire-up + 4 hard + 2 soft gate 实测；Step 3 archive (本 PR) PROMOTE specs + 推 forward debt。结论 ADR-0003 NO-GO option (b)。

**p8pre-spike**:
SPGMR preconditioner identity-stub spike 的 epic 名 (SHUD-OpenMP#338)。覆盖 N=8 Mode C profile recheck (Step 1) + identity precond API wire-up + 4-hard-gate + 2-soft-gate verdict (Step 2)。**结论 NO-GO** per ADR-0003 option (b)：H2 hard FAIL (`ncfn` 跨 18 cell deterministic floor 6/47 完全不变, identity P⁻¹=I 对 SHUD stiff Jacobian Newton 收敛失败 zero ROI) + S5 soft FAIL (PREC_LEFT 状态机 inherent cost 触发 ≈9×10¹⁵ ULP drift 远超 A4 阈值 1024)。H1/H3/H4/S6 PASS 仅证 plumbing 已接通, 不能挽救。**不开正式 P8-precond epic**, 推 design D8 future cleanup PR 还原 PREC_NONE。详 Epic #338, ADR-0003, `docs/p8pre/capstone.md`, `docs/p8pre_summary.md`, master plan §P8-precond.0。
_Avoid_: 把 p8pre-spike 等同于 P8-precond epic（spike 是 ROI 探针, 不是 production epic 启动）；把 NO-GO 解读为 SPGMR + 其它 preconditioner candidate (Diagonal/Jacobi/ILU(0)) 永远不行（Step 1 canonical `r_min=1.819 ≥ 1.5` 说明 ROI window 存在, 只是 identity 不能 unlock）

**identity preconditioner**:
`P⁻¹ = I` SUNDIALS PSetup/PSolve stub (`SHUD/src/Equations/MD_precond_identity.{h,cpp}` 40+61 lines, PR-D #357)。`PSetupIdentity` 返回 0 + 设 `*jcurPtr = SUNFALSE`；`PSolveIdentity` 返回 0 + `N_VScale(1.0, r, z)` 把 r copy 到 z。**数学行为等价 PREC_NONE** (identity 不旋转 residual), 但 SUNDIALS state-machine 走 PREC_LEFT 路径触发额外 `N_VLinearSum` / `N_VScale` op。提供 plumbing-but-no-convergence-help 的 spike — 验证 future preconditioner candidate (Diagonal/Jacobi/ILU(0)/block-Jacobi) 启动时可直接复用 jok-mirror PSetup + memcpy PSolve 骨架 (per SUNDIALS canonical `cvDiurnal_kry.c` L716/L760), 不必重新发现 jok 旗语 + ier 三档 return code contract。**NO-GO 结局** per ADR-0003 + design D8 fall-back: future cleanup PR 还原 `SUN_PREC_NONE` + 删 `MD_precond_identity.{h,cpp}`。
_Avoid_: 把 identity preconditioner 当作 production-ready preconditioner（identity 提供 zero convergence 加速 ROI）；删 `MD_precond_identity.{h,cpp}` 不走 design D8 path（必须经 future cleanup PR + ADR cross-ref, 保留 jok-mirror canonical 接口骨架作 future candidate 复用）

**PREC_LEFT**:
SUNDIALS `SUN_PREC_LEFT` 枚举值；启用 left-preconditioned SPGMR (在 `SUNLinSol_SPGMR(udata, SUN_PREC_LEFT, 0, sunctx)` 调用位 = `SHUD/src/Equations/cvode_config.cpp:259`)。p8pre-spike Step 2 实测：bare-PREC_LEFT (即使 P⁻¹=I) 触发 `cvLsSolve` (`sundials/src/sundials/cvode_spils.c`) 内部额外 `N_VLinearSum` + `N_VScale` op, 在 fp64 累计意义上扰动 `rivqdown.dat` 5,155/214,252 = 2.4% 位置 nonzero-zero rotation, 累 90 天积分 state-machine drift ≈9×10¹⁵ ULP, 远超 A4 阈值 1024 (per spec p8precond-zero-identity-spike L106 fall-back)。这是 SUNDIALS 内部 inherent cost — future preconditioner candidate evaluate 时必须接受此 baseline drift, 或重新固化 mode C-precond reference SHA。`SUN_PREC_NONE` (= 0, 数值字面量) 是 PREC_LEFT 的反面：完全跳过 cvLsSolve preconditioner 状态机, 无 drift。
_Avoid_: 把 PREC_LEFT 当作 "需要 preconditioner 才有效"（identity spike 已实证 wire 通即触发 drift, 与 P⁻¹ 实际形式无关）；把 ≈9×10¹⁵ ULP drift 归咎 identity stub (drift 源自 SUNDIALS state-machine 不是 identity 数学行为)

**nfeLS/nfe ratio**:
SUNDIALS-CVODE Krylov amplification 比率 = `nfeLS` (Krylov 线性求解器 ƒ-evaluation 次数, via matrix-free DQ Jacobian-vector product) ÷ `nfe` (Newton 非线性 ƒ-evaluation 次数, via CVODE integrator 主动调用 RHS)。p8pre-spike Step 1 PREC_NONE canonical 实测：heihe N=8 `r = 12632/6943 = 1.819` (=`r_min`), heihe_x4 N=8 `r = 30509/6741 = 4.526` (=`r_max`; authoritative source per `docs/p8pre/n8_profile_verdict.md` §3.1), 均超 ADR-0002 Path 3 trigger threshold `r ≥ 1.5` → branch a (PROCEED Step 2)。**ADR-0002 Path 3 trigger criterion**：`r ≥ 1.5` 是 SPGMR Krylov 工作量足以摊销 preconditioner setup overhead 的 ROI 阈值, 低于此值 preconditioner 投入 negative ROI。详 `docs/p8pre/n8_profile_verdict.md` §5。
_Avoid_: 把 `nfeLS/nfe` 与 `nfe/nst` 混淆（前者衡量 Krylov 内部工作量 vs Newton 工作量, 后者衡量 Newton 失败重试率）；把 `r ≥ 1.5` 解读为 "已证 preconditioner 必有效"（ROI window 存在 ≠ identity preconditioner 必 unlock, 见 p8pre-spike NO-GO）

**CVodeSetLSetupFrequency**:
SUNDIALS-CVODE 6.0.0 API knob (`SHUD/InstallSundials/include/cvode/cvode.h:132` 声明), 控制 preconditioner setup 调用 cadence。函数签名：`int CVodeSetLSetupFrequency(void *cvode_mem, long int msbp)`. p8pre-spike Step 2 PR-D #357 在 `SHUD/src/Equations/cvode_config.cpp:263` 设 `msbp = 50` (= SUNDIALS 默认 per design D6), 即每 50 Newton 迭代调用一次 `PSetupIdentity`。**与 `CVodeSetJacEvalFrequency` 区分**：后者 (`cvode_ls.h:91`) 是 matrix-based 求解器 Jacobian-eval 频率, 不适用于 SPGMR matrix-free + preconditioned 路径。实测 S6 soft gate PASS：`t_precond_setup / t_wall_total ≤ 1.01×10⁻⁷` (6 数量级 below 5% 阈值, 因 identity setup 工作量为 0 + msbp=50 进一步稀释)。
_Avoid_: 把 `CVodeSetLSetupFrequency` 与 `CVodeSetMaxConvFails` (`cvode.h:133`) 混淆（后者控制 max nonlinear-iteration convergence failures per step 默认 10, 与 setup 频率无关）；改 `msbp` 不更新 spec + design + ADR 引用

**.p1e-i-runs/**:
P1e PR-I 18-cell strict-omp verification scratch dir 的 canonical 命名 (`/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/`, gitignored 通过 `.<stage>-runs/` glob)。**取代 typo `.pr-i-runs/`** (修复于 Step 0 PR-archive #350 / issue #339)。引用位 = `docs/p1e/p1e_perf_baseline.md` §3.5/§7.1 + `docs/p1e/p1e_pr_i_strict_omp_verification.md` + `docs/p8pre/step1_prep.md`. Step 0 anchor doc inline-quoting 把 P1e PR-I 24-cell median wall + nst 值落地到 `docs/p8pre/step1_prep.md`, 下游 (Step 1 aggregator + Step 2 PR-F gate-4 + ADR-0003) 不再穿透回 `.p1e-i-runs/` 路径串。
_Avoid_: 写作 `.pr-i-runs/` (typo, Step 0 已修正)；把 `.p1e-i-runs/` 当 outer git tree commit-able 路径（gitignored via `.gitignore` L34 邻近的 `.<stage>-runs/` glob 模式, scratch-only）

**ncfl**:
SUNDIALS-CVODE canonical Krylov solver convergence-failure counter (在 `cvode_stats.txt` + `tools/cvode_stats_diff/canonical_15_keys.yaml` 全 15-key 集合 `nfe nfeLS nni nli nsetups netf nst npe nps ncfn ncfl lenrw leniw lenrwLS leniwLS`)。**取代 typo `nlcf`** (修复于 Step 0 / `tools/cvode_stats_diff/canonical_15_keys.yaml` 同步; PR-B Step 1 aggregator REJECT 5 typo keys 全集 `{nlcf, nfevals, hcur, qcur, hin}` 中 `nlcf` 是 `ncfl` 的字符 transpose typo)。语义：SPGMR 线性求解收敛失败次数, 触发 Krylov restart。p8pre-spike Step 2 实测 heihe `ncfl=121` (与 `ncfn=6` 区分：`ncfn` = Newton 非线性失败, `ncfl` = Krylov 线性失败), 是 Step 2 spike "future P8-tune.C SPGMR maxl sweep 候选" forward debt 的 anchor 数值 (per spec p8precond-zero-identity-spike §9.2 Prerequisite 4)。
_Avoid_: 写作 `nlcf` (typo, Step 0 已修正)；把 `ncfl` 与 `ncfn` 混淆（前者 Krylov 线性失败计数, 后者 Newton 非线性失败计数, 双 floor 都是 future preconditioner candidate 必须压制的 ROI baseline）

**ADR-0003 (precond-spike-decision)**:
`docs/adr/0003-precond-spike-decision.md` (PR-G #348 / PR #360 merged 2026-06-27, status Accepted), p8pre-spike epic Step 2 close 的 formal decision 记录。**Decision = NO-GO option (b)**: identity precond 不进入 production, p8pre-spike epic 关闭, design D8 future cleanup PR 还原 PREC_NONE。**Related ADR**: 触发条件来自 ADR-0002 Path 3 (deferred → spike trigger), 关闭后 Path 3 trigger 第 (iii) 条件解为 NO-GO。**8-section 结构**：Status / Date / Deciders / Owner / Tags / Context / Decision / Consequences / References (per `docs/adr/0002-solver-path.md` mother template)。Consequences §4 显式列 Spec L26 wording correction (jok-mirror canonical cite) 推 #349 archive — 即本 PR scope。
_Avoid_: 把 ADR-0003 NO-GO 误读为 "SPGMR + preconditioner 永久淘汰"（NO-GO 是 identity stub 一项, future Diagonal/Jacobi/ILU(0)/block-Jacobi candidate 仍可基于 Step 1 `r_min=1.819` ROI ground + Step 2 jok-mirror canonical 骨架启动）；改 ADR-0003 Decision 不同步 master plan §P8-precond.0 + `docs/p8pre/capstone.md` + `docs/p8pre_summary.md` 三 doc

### p8tune-spgmr-maxl (SPGMR maxl tune) 集合

p8tune-spgmr-maxl epic (openspec change `p8tune-spgmr-maxl`) 在 p8pre-spike NO-GO + design D8 cleanup completed 后立项。4 capabilities (`p8pre-doc-state-correction` / `clean-prec-none-baseline` / `spgmr-maxl-env-hook` / `maxl-sweep-verdict`) × 6 PR (PR-0 → PR-A → PR-B → PR-C → PR-D → PR-E + conditional PR-F)，决策入口 ADR-0004 (TBD)。

**mode-C-tune**:
Reference-set design pattern for SHUD CVODE solver-tune campaigns where Mode C strict-omp A3a SHA bitwise gate does not apply because the tune parameter (e.g., SPGMR `maxl`, `MaxNonlinIters`) directly affects reduction order。**Per (case, maxl) tuple**, `rivqdown.dat` SHA12 + `cvode_stats.txt` 15-key set (per `tools/cvode_stats_diff/canonical_15_keys.yaml`) form a separate reference anchor。A3a strict SHA verification does NOT apply across different maxl values。**A4 max_ulp tolerance** (≤1024 default; sweep-specific tolerance per maxl) applies BOTH between mode-C-tune(maxl=N) references AND between cross-N (N=1/4/8) within the same (case, maxl) tuple。Hydrology validation uses water-balance + hydrology-set indicator tolerance per G7 (currently A4 max_ulp fallback only; water-balance threshold TBD)。Precedent: capability [`maxl-sweep-verdict`](../openspec/changes/p8tune-spgmr-maxl/specs/maxl-sweep-verdict/spec.md) (this change); future use: P8-tune.A/B/D + P8-precond reference design。see also: p8tune-spgmr-maxl change; ADR-0004 (TBD)。
_Avoid_: 把 mode-C-tune 当作 "放弃 bitwise reproducibility"（A3a 不跨 maxl 适用 ≠ 同 (case, maxl) 内放弃 strict SHA，cross-N 内仍走 A3a strict; 仅 cross-maxl 走 A4）；遗漏 hydrology-set G7 fallback（A4 fallback 仅 deterministic 数值层，水文物理 indicator 必须独立 PASS 才允许 GO）。

**SHUD_SPGMR_MAXL**:
Runtime env var introduced by capability [`spgmr-maxl-env-hook`](../openspec/changes/p8tune-spgmr-maxl/specs/spgmr-maxl-env-hook/spec.md) controlling SPGMR Krylov subspace dimension at CVODE init time (`SHUD/src/Equations/cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, PREC_NONE, maxl, sunctx)` 调用 maxl 形参)。**Values**: `{unset, "", "0", "5", "10", "15", "20", "30"}`。unset / empty / explicit `"0"` 三种状态均触发 SUNDIALS 默认 `maxl=5`，bit-identical to current SHUD `37be0fe` cleaned-PREC_NONE baseline (per spgmr-maxl-env-hook G3 4-way default-compat gate)。其它值 (`"5"`/`"10"`/`"15"`/`"20"`/`"30"`) 进入 sweep 矩阵。**Safety constraints**: (i) **NEVER opens `PREC_LEFT`** — SUNLinSol_SPGMR 第二参数恒为 `PREC_NONE`; (ii) **NEVER registers preconditioner** (`CVodeSetPreconditioner` 调用不被本 env var 触发); (iii) **default-unset bit-identical** to current SHUD `37be0fe` cleaned-PREC_NONE baseline; (iv) **invalid 值** (e.g., `"7"`, `"50"`, `"foo"`, `"-1"`) 在 cvode init 时 abort via `myexit(ERRCVODE)`。详 capability `spgmr-maxl-env-hook` + ADR-0004 (TBD)。
_Avoid_: 把 SHUD_SPGMR_MAXL 与 `PREC_LEFT` 等同（env var 仅 tune maxl 参数, 不打开任何 preconditioner 状态机）；改 default value (从 5 改其它) without ADR-0004 GO+default-bump branch 形式批准。
