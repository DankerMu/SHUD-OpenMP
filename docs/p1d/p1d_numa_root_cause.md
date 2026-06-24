# P1d — NUMA / N≥4 散度根因技术分析

P1d epic 最重要的一份 doc。记录 PR-H 3 SHALL gate FAIL 后，通过两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查得到的**正确根因**，并把 PR-H 初版 4 个错误诊断逐项纠正。本 doc 是 ADR-0001 (solver-path) + P1e (F 路) openspec change + master plan §6 P1d / §6 P1e 修订的技术依据。

## §1 Original P1d hypothesis（pre-PR-H, master plan v1.4 era）

P1c PARTIAL CLOSURE 后 P1c.7 spec `p1c-deterministic-reduction` L100-L103 把 §4.4 A3a bitwise cross-N + §4.5 nst Δ=0 cross-N 标 "carve-out 推 P9 行"（per master plan §3 fallback option 2）。P1c.7 时刻的 hypothesis：

1. **drift origin OUTSIDE 8 sites reduction**（design D9 decision branch 2 CONFIRMED per `docs/p1c_a3a_root_cause.md`）
2. **drift origin = upstream parallel writer first-touch / NUMA-affinity cross-socket cache-line ping-pong**
3. P1d 走 NUMA env + steady-state first-touch + Kahan revert 三 phase stack 即可关 A3a + nst Δ=0 双门
4. 之后 P2a 启动前置 = `P1c-tag` lock + P1c.3 A3a 全通过

P1d epic 13-PR burst（PR-A → PR-H）依此 hypothesis 实施。

## §2 PR-H 实测：hypothesis 被 falsified

PR-H (post-PR-G Kahan revert + first-touch + NUMA env 全 stack) server 8-cell 实测：

| 指标 | 期望 (按 hypothesis) | PR-H 实测 |
|---|---|---|
| heihe N=4 SHA == N=1 | byte-identical | **不等** (3 distinct SHA per case) |
| heihe nst Δ | 0 strict | **80@N=4 / 152@N=8** |
| heihe_x4 nst \|Δ\| | ≤2 | **11@N=4** |
| rivqdown.dat mean_rel | ≤ULP | **10-25%** |

**Hypothesis falsified**：即使在 P1d 全 stack（PR-C/D/E first-touch + PR-G Kahan revert + PR-B NUMA env standardization + PR-F finding 后 drop `--interleave=all`）后，3 SHALL gate (L130 A3a + L139 nst + L145 reverse-compat) 仍 FAIL；散度量级 (mean_rel 10-25%) 与 P1c era（Kahan IN）相同 → P1d 全部干预都没动摇真正根因。

需要新的根因分析。

## §3 两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查

PR-H FAIL 后两轮独立 GPT Pro 复查指出问题不在 NUMA layer，是 build / execution stack 内部假设错误。Codebase 5/5 事实核查全部支持：

| # | 断言 | 核查 | 结论 |
|---|---|---|---|
| 1 | `f()` 始终调 Serial RHS | `SHUD/src/Model/f.cpp:54` `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` | ✅ |
| 2 | `StrictOMP/ProductionOMP` 是 abort 桩 | `SHUD/src/Model/MD_rhs_core.cpp:802-811` 三 case 全 `std::abort()` | ✅ |
| 3 | `shud_omp` build 硬编码 NVECTOR_OPENMP | `SHUD/Makefile` shud_omp target: `-DSHUD_USE_OPENMP_NVECTOR=1 -lsundials_nvecopenmp` | ✅ |
| 4 | `SHUD_USE_OPENMP_NVECTOR` 与 `SHUD_ENABLE_OPENMP_RHS` 正交，后者默认 0 | `SHUD/Makefile:140` `SHUD_ENABLE_OPENMP_RHS ?= 0` | ✅ |
| 5 | SPGMR 没注册 preconditioner | `SHUD/src/Equations/cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` 后无 `CVodeSetPreconditioner` 调用 | ✅ |

**含义**：

- **fact #1 + #2 一起**：`shud_omp` 入口 CVODE 调 `f()`，`f()` 始终把 `ExecPolicy::Serial` 传给 `rhs_core`，意味着 hydrology RHS 完全单线程；StrictOMP / ProductionOMP 不会被执行（abort 桩），那是个**编译进 binary 但 dead 的代码路径**
- **fact #3 + #4 一起**：`shud_omp` build 链接 NVECTOR_OPENMP backend（`-lsundials_nvecopenmp`），但 RHS 还是 serial → 这是个 **"Serial 水文 RHS + OpenMP N_Vector backend"** 的奇怪组合，不是 "完整 hydrology RHS OpenMP" 也不是 "Serial CVODE / Serial N_Vector"
- **fact #5**：原诊断说 "drift 在 SPGMR multi-threaded preconditioner" 不成立——SPGMR 创建后没有任何 `CVodeSetPreconditioner` 调用，所以**根本没有 preconditioner**

## §4 真正根因：SUNDIALS 6.0.0 NVECTOR_OPENMP 内部 reduction 顺序

`shud_omp` 实际跑 "Serial RHS + OpenMP N_Vector backend"，所以 cross-N 散度来源**只可能**在 N_Vector ops 内部。审 SUNDIALS 6.0.0 source `src/nvector/openmp/nvector_openmp.c`：

### §4.1 N_VDotProd_OpenMP

CVODE/SPGMR 的 Krylov 内积调用：

```c
realtype N_VDotProd_OpenMP(N_Vector x, N_Vector y)
{
  ...
  #pragma omp parallel for default(none) private(i) \
    reduction(+:sum) schedule(static)              \
    shared(N, xd, yd)
  for (i = 0; i < N; i++) sum += xd[i] * yd[i];
  ...
}
```

**关键问题**：`reduction(+:sum) schedule(static)` 的 reduction tree shape **由 runtime 决定，不固定**。OpenMP 规范不保证 reduction tree 跨 N 一致——thread count 变了 reduction 顺序就变，浮点累加非 associative → 不同 N 同输入 → 不同 `sum` 输出（哪怕只差 ULP）。

### §4.2 N_VWSqrSumLocal_OpenMP

WRMS norm 内部调用（用于 CVODE adaptive step-size controller）：

```c
realtype N_VWSqrSumLocal_OpenMP(N_Vector x, N_Vector w)
{
  ...
  #pragma omp parallel for default(none) private(i, prodi) \
    reduction(+:sum) schedule(static)                       \
    shared(N, xd, wd)
  for (i = 0; i < N; i++) {
    prodi = xd[i] * wd[i];
    sum += prodi * prodi;
  }
  ...
}
```

同 `N_VDotProd_OpenMP`，`reduction(+:sum)` 顺序不固定。

### §4.3 调用链路（N_Vector ops → CVODE → trajectory）

```
NVECTOR_OPENMP reduction tree shape (跨 N 不固定)
    ↓
N_VDotProd_OpenMP / N_VWSqrSumLocal_OpenMP 同输入跨 N 不同 sum (差 ULP)
    ↓
CVODE WRMS norm 跨 N 不同 → adaptive step-size controller 收到不同 error estimate
    ↓
CVODE 选不同 step h → 不同 nst / nfe trajectory
    ↓
SPGMR Krylov inner product 同样过 N_VDotProd_OpenMP → 同样跨 N 漂
    ↓
solve trajectory 完全分叉
    ↓
output rivqdown.dat byte-different + mean_rel 10-25% (不是 ULP, 是另一条轨迹)
```

**Reference**: SUNDIALS 6.0.0 source `src/nvector/openmp/nvector_openmp.c` 全部 reduction ops 都用 `reduction(+:sum) schedule(static)` pattern；这是 SUNDIALS 的 fast/non-strict-deterministic backend, 与之对应的 strict-deterministic backend 是 SUNDIALS post-6.x 加的 NVECTOR_OPENMP_REPRO 或 user-implemented serial-order N_Vector wrapper（详 §11 ADR-0001 路径）。

## §5 Why first-touch did NOT help

PR-C/D/E 添加的 steady-state first-touch loops（`MD_rhs_core.cpp::rhs_update` element + `rhs_flux` river + `rhs_update` lake 3 #pragma 区）按设计是为**并行 owner-compute** 做页面预放置——每个 thread 在 RHS 入口先 zero-write 自己 owner-的字段子集，使 page-fault 在 owner thread 触发 → 写时落到 owner 的 NUMA node。

**但 fact-check #1 + #2 揭示**：consumer (CVODE 调 `f()` 调 `rhs_core(Y, DY, t, ExecPolicy::Serial)`) **始终单线程**，没有并行 owner-compute 阶段。所以 first-touch loops：

1. **不能加速 hydrology RHS**：consumer 是 serial，NUMA locality 无意义
2. **不能减少 cross-N 散度**：散度来源在 N_Vector ops 内（§4），不在 hydrology field layout
3. **反而消耗带宽**：每次 RHS 调用都跑一次 first-touch zero-write loop（per CVODE step），对 large case 是 memory bandwidth 浪费
4. **可能 paged in 到错误 NUMA node**：first-touch loop 用 `schedule(static)`，N=8 时把字段写到 8 个 thread 各自的 NUMA node；但接下来 serial RHS 单线程读所有字段，需要跨 NUMA fetch → 反向 cache-line ping-pong

**结论**：PR-C/D/E steady-state first-touch loops 在当前 build（Serial RHS）下是 **无效优化 + 带宽浪费**。M10 修订 marks 它们 **DEPRECATED**；P1e (F 路) 实现 StrictOMP RHS 时再决定 first-touch 怎么重新设计。Allocation-time first-touch（`Model_Data.cpp::malloc_EleRiv` L251-L346 的 first-touch 模式）保留——page fault 一次性触发是合理的, 它影响 allocator-level placement 不影响 hot-path bandwidth。

## §6 Why Kahan did NOT help (full picture)

P1c §4.7 conditional Kahan/Neumaier compensation 注入了 SHUD `MD_rhs_core.cpp` 内 3 reduction helpers（`fixed_pairwise_sum_indexed` / `fixed_leftfold_sum_indexed` / `fixed_leftfold_sum_pair_indexed`）。P1c PR-I 二跑实测 Kahan 减 `|Δ_nst|` 63%（heihe 225 → 84），但 rivqdown.dat A3a cross-N FAIL 模式（3 distinct SHA）**保留**。

PR-H 实测 Kahan revert (post-PR-G) → heihe |Δ_nst| 反而上 84 → 152（恶化 81%），mean_rel 10% 不变。Kahan 在 |Δ_nst| 上有部分压制效果，在 rivqdown.dat 上零效果。

**为什么？** Kahan 注入的位置是 SHUD `MD_rhs_core.cpp` 内的 owner-gather reduction（element-to-river / element-to-lake 累加），属 **hydrology RHS 内部** 的求和；但 CVODE adaptive 路径上的 reduction（WRMS norm + SPGMR Krylov inner product）走的是 **N_Vector layer**（SUNDIALS NVECTOR_OPENMP）—— Kahan 注入完全没碰到。

```
                ┌───────────────────────────┐
                │  Kahan injection scope    │
                │  = SHUD MD_rhs_core.cpp   │
                │    fixed_*_sum helpers     │
                │  (hydrology RHS owner-     │
                │   gather)                  │
                └─────────────┬─────────────┘
                              │
                              ↓ feeds CVODE Y/DY state
                              ↓
                ┌─────────────────────────────┐
                │  N_Vector reduction layer    │
                │  = SUNDIALS NVECTOR_OPENMP   │
                │    N_VDotProd_OpenMP /        │
                │    N_VWSqrSumLocal_OpenMP    │
                │  (CVODE WRMS + SPGMR Krylov)  │
                │                              │
                │  KAHAN 未触及！              │
                │  这层用 reduction(+:sum)      │
                │  schedule(static), 跨 N 漂。 │
                └─────────────────────────────┘
```

**结论**：Kahan 改善 |Δ_nst|（小幅）= 因为 hydrology RHS reduction 不漂了 → 减少了少量 RHS-induced trajectory branching；但 N_Vector reduction 仍漂 → CVODE 主轨迹仍跨 N 分叉 → rivqdown.dat 仍跨 N 不同。**Kahan 与 N_Vector reduction 是 orthogonal axes**。

PR-G Kahan revert 保留 hydrology RHS 内部 acc order 简洁（pre-K2 canonical），为 P1e (F 路) StrictOMP RHS 提供干净的 baseline；revert 本身没影响 cross-N 散度（fact-check：revert 前后 rivqdown.dat mean_rel 都是 10-25%）。

## §7 PR-H 初版 4 个错误诊断的纠正

PR-H 初版 verdict 把 FAIL 解读为 "nst Δ 超 spec ladder + 根因在 SPGMR multi-threaded preconditioner + KLU 唯一根治"。**两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查全部推翻**：

| # | 初版（错） | 修订（对） | 依据 |
|---|---|---|---|
| 1 | "drift origin 在 SPGMR multi-threaded preconditioner" | SPGMR **没有** preconditioner；drift 是 `N_VDotProd_OpenMP` / `N_VWSqrSumLocal_OpenMP` 的 `reduction(+:sum) schedule(static)` 跨 N reduction tree 顺序不固定 | fact-check #5 + SUNDIALS 6.0.0 `nvector_openmp.c` source review |
| 2 | "Amdahl serial fraction ~72%" | 由 wall 反推 `f = (1/S - 1/N)/(1 - 1/N)`：heihe ~**87%**, heihe_x4 ~**76%**（差 15pp）；且不能全部归因 CVODE——含 serial RHS + N_Vector fork/join 开销 + memory bandwidth 饱和 | `docs/p1d/p1d_perf_baseline.md` §4 重新计算 |
| 3 | "reltol=1e-4 但实测 10% → solver 承诺被打穿四个数量级" | CVODE reltol 控制的是**每步状态 WRMS local error**，**不是** 90 天派生流量轨迹的全局相对误差上界。两个 reduction tree shapes 各自 satisfy WRMS local error tolerance，但走出两条 trajectory，输出 90-day 后 derived flow 差 10%；这不是 CVODE 实现错误的证据 | CVODE 6.0 manual §6.2 reltol/abstol definition + step-size controller spec |
| 4 | "KLU 是唯一根治路径" | KLU 不能单独 fix。CVODE WRMS norm 还是过 N_Vector reduction，换 KLU 不换 N_Vector 仍然漂。Determinism 与 solver 选择**正交**。KLU 推到 ADR-0001 作 4 路对比之一 | fact-check + §10 remediation 路径决策表 |

## §8 K=200 vs K=0 在流量层面的真实含义

ladder K (= nst Δ allowed) 的设计意图是 "允许少量 step-count 差异（CVODE 在不同 NVector 顺序下选不同 step）但要求流量收敛仍在 tolerance"。但 PR-H 实测：

| Mode | nst Δ | rivqdown.dat mean_rel | 含义 |
|---|---|---|---|
| K=0 (true strict) | 0 | ≤ULP | 真 reproducible |
| K=200 (PR-H 实测) | 80-152 (heihe) | mean 10-25% rel error | **另一条 CVODE 轨迹**，非 ULP 近似 |

含义：K=200 不是 "noise band", 是 **轨迹分叉**。CVODE adaptive controller 接到不同 WRMS norm（来自 N_Vector reduction order shift）后，在不同 t 选择不同 step h → 累加 90 天得到完全不同 state vector → rivqdown.dat 完全不同。这不是 "small ULP differences accumulating"，是 "two different valid solutions of the same ODE with different step sequences"。

工程对照：vs CMFD forcing 不确定性 (10-20%) → 同量级；vs gauge 精度 (5-10%) → 超过测量精度；vs Manning's n 标定带宽 (~50%) → 量级以下。10-25% mean_rel **不能** claim "in tolerance with CVODE rtol"；它是另一条轨迹，不是噪声。

这是 4-mode 重写把 `fast-omp` mode 标 non-production 的工程依据。

## §9 PR-G Kahan revert 的真正价值（fixed in narrative）

事实核查后，PR-G Kahan revert 的价值不是 "把 P1c PARTIAL CLOSURE 直接转 PASS"（原 hypothesis），而是：

1. **证明 revert 干净**：Mac 9-SHA matrix 显示 post-PR-G == pre-K2 (de9545d) byte-identical（详 `docs/p1d/p1d_kahan_revert.md` §"SHA matrix"）；且 post-PR-G == P1-update-omp-tag (PR-I anchor) byte-identical（详 `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` keliya 表）
2. **为 P1e (F 路) baseline 做准备**：P1e 实现 `ExecPolicy::StrictOMP` 真正并行 hydrology RHS 时，需要 acc order 干净的 reduction helpers 作为 owner-fold 的基础；Kahan injected 版本会引入跨 thread 的 compensation 计算路径差异，干扰 F 路 cross-N bitwise 验证。pre-K2 简洁累加更适合作 F 路起点
3. **PR-H heihe N=1 SHA == spec L123 预写 canonical**：证明 P1d 在 serial path 上完整恢复了 `P1-update-omp-tag` era 的 trajectory（pre-Kahan 等价），这是 4-mode `serial` mode 的 SHALL 验证基础

PR-G revert 保留是 E′ 第 5 项动作的依据。

## §10 Remediation 路径决策（实测后修订）

| 路径 | 描述 | 评估 | 落点 |
|---|---|---|---|
| A | 放宽 ladder K=200 | ❌ false advertising "strict" | 拒 |
| B | SPGMR → KLU 重构 | P2/P3 ADR 评估，**不是 P1d 必选**——KLU 单独不能 fix N_Vector reduction (fact-check)；除非配合 §4.1 NVECTOR_REPRO_OMP，否则换 KLU 也跨 N 漂 | ADR-0001 路径之一 |
| C | SUNDIALS 内部 deterministic patch (用 NVECTOR_OPENMP_REPRO 或 fork) | ❌ 验收周期长（vendor change），未必彻底 fix | ADR-0001 fallback |
| D | N=1/2 bitwise + N≥4 ladder 双 mode | 配合 E′ 4-mode spec（serial mode + strict-omp mode 划分） | E′ 4-mode spec 内 |
| **E′** | **P1d containment closure** | **当前 P1d 收尾路线**（不是简单 E，含 4-mode spec + 错误叙事更正 + first-touch deprecation note） | **P1d capstone (本 epic)** |
| **F** | **Serial N_Vector + StrictOMP RHS** | **下个 epic (P1e) 技术主线**——真正完成 P1d 原计划应该完成的事 | **P1e (next epic)** |

E′ 闭住 P1d 当前 7 项动作 (per `docs/p1d/p1d_summary.md` §5)；F 闭住 P1e 全部技术债 (per master plan §6 P1e)。

## §11 F 路（P1e）详细 plan

### §11.1 技术主线

**Serial N_Vector + StrictOMP RHS**：

- **N_Vector 保持 Serial**（`N_VNew_Serial`）→ CVODE/SPGMR 的 Krylov 点积 + WRMS norm 顺序 deterministic → cross-N reduction order 不再变 → bitwise 跨 N 自然成立
- **Hydrology RHS 真正并行**（`ExecPolicy::StrictOMP` 替换 abort 桩）→ 真正去吃 RHS 66.55% wall → 理想上界 2.39× 加速可达

这是 master plan v1.4 原 P7 计划的"完整 RHS OpenMP + serial CVODE"思路，只是 P1c → P1d 实测路径绕弯后重新走正路。

### §11.2 2×2 build matrix 因果实验（启动前必跑）

P1e 任何代码改造前，必须先用 4 build × N∈{1,2,4,8} × 3 repeats 验证 hypothesis：

| Build | N_Vector | RHS | 目的 |
|---|---|---|---|
| A | `N_VNew_Serial` | Serial | canonical reference |
| B | `N_VNew_OpenMP` | Serial | =当前 `shud_omp`，复现 PR-H 10-25% 散度作 control |
| C | **`N_VNew_Serial`** | **StrictOMP** | **P1e production 候选** |
| D | `N_VNew_OpenMP` | StrictOMP | research only |

实测条件：heihe + heihe_x4 (server) + keliya + qhh (Mac) 4 case + 90-day cap + 同硬件 + 同 SUNDIALS 6.0.0 build + hash `CV_Y` state vector + `rivqdown.dat` + capture `nst/nfe/nli/nni/netf/ncfn` 15-key set。判据：

- **A 同 build 同 N 重复 3 次 bitwise** → solver 本身 deterministic 的前提
- **B 跨 N 不同 而 A 跨 N 相同** → 确认 NVECTOR_OPENMP reduction 是主因（不是 RHS race）
- **C 跨 N bitwise + nst Δ=0 + 加速** → F 路成立，进入 P1e.3 正式实施
- **C 跨 N 也分叉** → 查 RHS race / 共享状态 / phase dependency；可能更细 owner-compute 分解
- **C 加速 < 1.5×** → 进 ADR-0001 评估 (block-Jacobi precond / NVECTOR_REPRO_OMP / KLU)

### §11.3 实施要点

1. **`ExecPolicy::StrictOMP` 路径替换 `std::abort()` 桩**（`MD_rhs_core.cpp:802` 当前 case）
2. **单 parallel region + phase-based for + `default(none)`** + 隐式 barrier：每次 RHS 只创建一个 parallel region，phase 间用 OpenMP 隐式 barrier 同步；不为每个小循环单独 fork-join
3. **复用现有 `rhs_deterministic_gather()` 基础设施**：并行 owner 外层 (每个 element/river/lake 一个 thread) + canonical fold 内层 (owner 的 fixed B0 顺序 left-fold 不变 — 这正是当前 spec 已经设计好的 deterministic gather)
4. **配置项拆**：`NUM_RHS_THREADS` (RHS 并行度) + `NUM_NVECTOR_THREADS` (默认 1 = Serial NVector)；`omp_set_num_threads` 调用从 `SHUD_USE_OPENMP_NVECTOR` 条件内移出
5. **删 PR-C/D/E steady-state first-touch loops**（M10 deprecated）；保留 allocation-time first-touch (`Model_Data.cpp::malloc_EleRiv` L251-L346 的 first-touch 模式仍正确 — page fault 一次性触发是合理的)
6. **`rivqdown.dat` 输出缓存 audit**：确认输出代码是从 `tout` 状态重算 flux，而**不是** 直接写 solver 内部最后一次 RHS 留下的 `FluxRiv` 缓存 (per Pro2 警示——CVODE `CV_NORMAL` 模式下 internal step 可能超过 output time)

## §12 Long-term ADR roadmap

`docs/adr/0001-solver-path.md`（Phase 2(e) 并行 agent 创建）将做 4 路 solver 对比，不阻塞 F 路：

| 路径 | 优势 | 劣势 |
|---|---|---|
| Serial N_Vector + StrictOMP RHS (F 路, P1e 首选) | 用现有 SUNDIALS 不改 vendor；可立即用 deterministic gather；hydrology RHS 真并行 | 若 mode C 跨 N 不可达 strict bitwise 需 fallback |
| Deterministic NVECTOR_REPRO_OMP (P1e fallback) | hydrology RHS 同 F 路并行；NVector layer 也并行（吃 wall 更多）+ deterministic | 需手写或 backport SUNDIALS deterministic backend |
| SPGMR + block-Jacobi physics-based precond | 加速 SPGMR Krylov 收敛 → 减 iter count → 减 N_Vector ops → 间接减 cross-N 散度 source | precond setup 不通用，element/river/lake 3 块独立；compute 量增 |
| KLU sparse direct | 完全消除 SPGMR + Krylov layer → 单步 solve 确定性 + 不依赖 N_Vector reduction | full Jacobian factorize cost (fill ratio, memory peak)；P2/P3 评估 |

P1e 首选 Serial N_Vector + StrictOMP RHS（F 路）。若 2×2 因果实验 mode C 失败，进 ADR-0001 评估二选项 NVECTOR_REPRO_OMP。KLU 仅作 P2/P3 单独决策（量化 fill ratio + memory peak + factor wall 后比较）。

## §13 References

| 文档 | 用途 |
|---|---|
| 本文件 (`docs/p1d/p1d_numa_root_cause.md`) | 技术 autopsy doc |
| `docs/p1d/p1d_summary.md` | P1d capstone narrative (§7 错误诊断对照) |
| `docs/p1d/p1d_pr_h_final_run.md` §"Post-verdict 修订" | PR-H 5/5 fact-check + remediation 决策表 |
| `docs/p1d/p1d_first_touch_design.md` | PR-C/D/E first-touch 设计（M10 后 steady-state 标 DEPRECATED） |
| `docs/p1d/p1d_kahan_revert.md` | PR-G Kahan revert + Mac 9-SHA 证 revert 干净 |
| `SHUD/src/Model/f.cpp:54` | fact-check #1 evidence |
| `SHUD/src/Model/MD_rhs_core.cpp:802-811` | fact-check #2 evidence |
| `SHUD/Makefile` shud_omp target + `SHUD/Makefile:140` | fact-check #3 + #4 evidence |
| `SHUD/src/Equations/cvode_config.cpp:259` | fact-check #5 evidence |
| SUNDIALS 6.0.0 `src/nvector/openmp/nvector_openmp.c` | NVECTOR_OPENMP reduction source |
| CVODE 6.0 manual §6.2 reltol/abstol | reltol semantics (本 doc §7 #3 依据) |
| `SHUD_openMP_master_plan.md` v1.5 / M10 §6 P1d + §6 P1e | master plan 修订主体 |
| `docs/adr/0001-solver-path.md` (forthcoming) | 4 路 solver 对比 (Phase 2(e) 并行 agent owns) |
| `openspec/changes/p1e-strict-omp-rhs/` (forthcoming) | P1e openspec change (Phase 2(e) 并行 agent owns) |
