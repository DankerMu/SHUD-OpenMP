# ADR-0002: solver-path — selecting parallel determinism + speedup recipe for SHUD-OpenMP

- **Status**: Implemented (P1e epic close, 2026-06-25)
- **Date**: 2026-06-24
- **Deciders**: P1d epic team + 2 轮独立 GPT Pro 复查 (per `docs/p1d/p1d_pr_h_final_run.md` § "Post-verdict 修订")
- **Owner**: SHUD OpenMP 改造工程 / P1d capstone → P1e epic intake
- **Tags**: solver / parallel / determinism / NVECTOR / SPGMR / KLU / Amdahl
- **Supersedes**: none (precedent solver-path ADR)
- **Superseded by**: none
- **Related**: master plan v1.5 / M10 §4.13 + §4.17 + §6 P1d + §6 P1e + §7.2 RISK-NEW1/2 + §7.3 + §8.1；openspec changes `p1d-numa-governance` + `p1e-strict-omp-rhs`；`docs/p1d/p1d_pr_h_final_run.md`；`docs/p1d/p1d_numa_root_cause.md` (PR-K forthcoming)

---

## Context

P1c epic (PARTIAL CLOSURE) + P1d epic (PARTIAL CLOSURE via E′ containment) 两次 strict bitwise / nst Δ=0 跨 N 验收失败之后，团队在 PR-H final verdict 阶段做了 2 轮独立 GPT Pro 复查 + codebase 事实核查，结果**全部颠覆**初版 P1d 根因诊断。

**Fact-check 5/5 全部支持错误诊断纠正**：

| # | 断言 | 核查证据 | 结论 |
|---|---|---|---|
| 1 | `f()` 始终调 Serial RHS | `SHUD/src/Model/f.cpp:54` `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` | ✅ |
| 2 | `StrictOMP/ProductionOMP` 是 abort 桩 | `SHUD/src/Model/MD_rhs_core.cpp:802-811` 三 case 全 `std::abort()` | ✅ |
| 3 | `shud_omp` build 硬编码 NVECTOR_OPENMP | `SHUD/Makefile` shud_omp target: `-DSHUD_USE_OPENMP_NVECTOR=1 -lsundials_nvecopenmp` | ✅ |
| 4 | `SHUD_USE_OPENMP_NVECTOR` 与 `SHUD_ENABLE_OPENMP_RHS` 正交，后者默认 0 | `SHUD/Makefile:140` `SHUD_ENABLE_OPENMP_RHS ?= 0` | ✅ |
| 5 | SPGMR 没注册 preconditioner | `SHUD/src/Equations/cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` 后无 `CVodeSetPreconditioner` 调用 | ✅ |

**含义**：当前 `shud_omp` 实际跑的是 **Serial 水文 RHS + OpenMP N_Vector backend**，**不是** 真正的 hydrology RHS OpenMP 并行。PR-C/D/E 添加的 first-touch loops 是为完全没发生的 parallel RHS owner-compute 做的页面预放置——consumer 是单线程，根本无视 NUMA locality。N≥4 跨 N rivqdown.dat 散度（heihe 10%、heihe_x4 25.1%）的真正根因是 SUNDIALS 6.0.0 `NVECTOR_OPENMP` 内 `N_VDotProd_OpenMP` / `N_VWSqrSumLocal_OpenMP` (WRMS norm 底层) 用 `reduction(+:sum) schedule(static)` 跨 N reduction tree 顺序不固定。

**P1d 初版误诊 4 项更正**：

| 初版（错） | 修订（对） |
|---|---|
| "drift origin 在 SPGMR multi-threaded preconditioner" | SPGMR 没有 preconditioner；drift 是 `N_VDotProd_OpenMP` 的 `reduction(+:sum) schedule(static)` 跨 N reduction tree 顺序不固定 |
| "Amdahl serial fraction ~72%" | 由 wall 反推 `f = (1/S - 1/N)/(1 - 1/N)`：heihe ~87%，heihe_x4 ~76%；含 serial RHS + N_Vector fork/join + memory bandwidth 饱和 |
| "reltol=1e-4 但实测 10% → CVODE 承诺被打穿四个数量级" | CVODE reltol 控制的是**每步状态 WRMS local error**，不是 90 天派生流量轨迹全局相对误差上界 |
| "KLU 是唯一根治路径" | KLU 不能单独 fix——CVODE WRMS norm 还是过 N_Vector reduction，换 KLU 不换 N_Vector 仍然漂；determinism 与 solver 选择**正交** |

由于初版误诊四项均被推翻，本 ADR 必须以**正交化的解空间**为基础重新组织，避免后续 P1e / P2 / P3+ epic 再次陷入"先猜根因再修代码"的循环。本 ADR 把 solver-path 解空间显式列为 4 条，给每条一个明确的 status + trigger condition，配合 master plan §6 P1e + §7.2 RISK-NEW1/2 形成闭环。

---

## Problem Statement

需要同时满足两个条件：

1. **Strict bitwise reproducibility across N∈{1,2,4,8} threads**（学术发表必需 + forensic debugging 必需 + master plan §1.1.2 strict 模式承诺）
2. **≥1.5× speedup at N=8**（工程 ROI 阈值 per §1.1.1 量化目标 Medium-非 IO 主导列 M=1.5×）

二者**同时**满足。当前 `shud_omp` 实测 N=8 加速比仅 heihe 1.13× / heihe_x4 1.27×，且跨 N 散度 10-25%，两项均不达标，根本原因不是单一缺陷，而是 NVECTOR 并行 + Serial RHS 这对架构错配。

---

## Decision Drivers

| 编号 | 维度 | 权重 |
|---|---|---|
| D1 | Reproducibility (master plan A3a strict + R3 风险登记) | 高 (硬门控) |
| D2 | Speedup at N=8 (per §1.1.1 quantitative target) | 高 (硬门控) |
| D3 | Implementation cost (epic-weeks) | 中 |
| D4 | Risk (memory blowup, build complexity, SUNDIALS API surface) | 高 |
| D5 | Reversibility (能否在 epic 周期内 walk back) | 中 |
| D6 | Future SUNDIALS upgrade compatibility | 中 |

---

## Candidate Paths

### Path 1: Serial N_Vector + StrictOMP RHS (= F 路 / P1e epic 候选)

**Mechanism**：
- 把 CVODE 用的 N_Vector backend 从 `N_VNew_OpenMP` 切回 `N_VNew_Serial` → CVODE 的 Krylov 点积 (`N_VDotProd`) + WRMS norm (`N_VWSqrSumLocal`) reduction 顺序变 deterministic（serial fold-left）→ cross-N reduction tree 不再随 N 变 → **bitwise 跨 N 自然成立**
- `ExecPolicy::StrictOMP` 路径（当前 `MD_rhs_core.cpp:802-811` 是 `std::abort()` 桩）替换为真正实现：单 `#pragma omp parallel` 外层 + phase-based `#pragma omp for nowait/barrier` 内层 + `default(none) shared(...) private(...)` 严格 + 隐式 barrier 同步
- 复用 P1c 期已落地的 `rhs_deterministic_gather()`：并行 owner 外层（每 element/river/lake 一个 thread）+ canonical fold 内层（owner 的 fixed B0 顺序 left-fold，已 deterministic）
- 把 `NUM_RHS_THREADS` 与 `NUM_NVECTOR_THREADS` 拆开：RHS 多线程并行；NVECTOR 强制单线程（即使保留 OpenMP build）

**Cost**：2-3 epic-weeks（替换 abort 桩 + 配置项拆分 + steady-state first-touch 重设计 + 2×2 因果实验）

**Reproducibility**：强（Serial NVec 按定义 byte-identical；RHS 用 canonical left-fold 已 deterministic）

**Speedup**：目标 1.5-2.4× at N=8。早期 profile 显示 heihe_x4 RHS 占 wall 66.55%，理想 8 核 Amdahl 上界 2.39×；P1e 把真正的 RHS 并行起来后，应能逼近上界。

**Risk**：低
- 若 2×2 mode C (Serial NVec + StrictOMP RHS) 跨 N FAIL → fallback Path 2
- 若 mode C 加速 < 1.5× → 进 Path 3 评估

**Reversibility**：高（单 epic revert，无 SUNDIALS API 边界改动）

**Decision**：**SELECTED — P1e epic 实施**

---

### Path 2: Deterministic NVECTOR_REPRO_OMP custom backend

**Mechanism**：
- 实现自定义 `NVECTOR_REPRO_OMP` backend，封装 SUNDIALS 6.0.0 N_Vector ops 接口
- 关键 reduction ops (`N_VDotProd`, `N_VWSqrSumLocal`, `N_VMin`, `N_VL1Norm`, `N_VInvTest` 等) 内部实现 canonical left-fold（serial-order）+ 可选 atomic ordering
- 通过 SUNDIALS context API (`SUNContext_PushErrHandler` + custom ops table) 注册
- 保留 `ExecPolicy::StrictOMP` RHS（沿用 Path 1 的 RHS 实现）

**Cost**：3-5 epic-weeks
- 实现 18+ N_Vector standard ops（per SUNDIALS 6.0.0 `nvector.h`）的 deterministic variant
- SUNDIALS context 注册 + ops table plumbing
- 自有 unit tests（与 SUNDIALS 内部测试不兼容）

**Reproducibility**：强（reduction 顺序受我们控制）

**Speedup**：中等
- NVector ops 仍可并行，但用 canonical fold + atomic ordering 后 throughput 大约是 stock NVECTOR_OPENMP 的 0.7-1.0×
- 总加速比与 Path 1 接近，但实现复杂度高

**Risk**：中
- SUNDIALS API surface 大，custom backend 需要长期维护
- 未来 SUNDIALS major upgrade（7.x）可能破坏 ABI 兼容性
- 自有 backend = 自有 bug 池

**Reversibility**：中（custom backend 一旦写好不容易丢弃，但可被 Path 1 取代）

**Decision**：**FALLBACK** — 仅在 Path 1 的 2×2 mode C 跨 N strict bitwise FAIL 时启用（per `openspec/changes/p1e-strict-omp-rhs/design.md` D2）

---

### Path 3: SPGMR + block-Jacobi physics-based preconditioner

**Mechanism**：
- 给现有 SPGMR (per `SHUD/src/Equations/cvode_config.cpp:259`) 注册 physics-based preconditioner：
  - element 块：3×3 (surface / unsat / GW) 小块独立 setup + solve
  - river 块：1×1 标量
  - lake 块：1×1 标量
- 通过 `CVodeSetPreconditioner(cvode_mem, p_setup, p_solve)` 注册（**当前缺失**，per fact-check #5）
- 显著降低 Krylov iteration 数 (`nli`)，从而压缩 RHS 调用次数 (`nfeLS`)

**Cost**：2-3 epic-weeks
- precond setup + solve 实现（per master plan §6 P8-precond v1.1 设计）
- 集成测试 + 收敛性验证

**Reproducibility**：**weak alone** — 不能单独解决 cross-N drift
- precond 本身可设计为 deterministic，但 SPGMR Krylov 仍然过 N_Vector 的 `N_VDotProd` 等 reduction
- 即使加 precond，跨 N reduction 顺序不固定的问题依然存在

**Speedup**：potentially high
- 物理 block-Jacobi precond 通常能把 SHUD 这种刚性多尺度系统的 `nli` 降到 1/3 - 1/5
- 但需配合 Path 1 或 Path 2 才能解 reproducibility

**Risk**：中-高
- precond 设计本身有 research 成分（不是纯工程）
- precond quality 取决于物理直觉 + 数值实验

**Reversibility**：高（precond 可 disable）

**Decision**：**DEFERRED — P2 优化阶段** — 与 Path 1 配对使用；如 Path 1 通过 strict-omp mode 但加速比 < 1.5×，则进 P1e.8 作 fallback；正常路径推到 P2 stage。

---

### Path 4: SPGMR → KLU sparse direct solver

**Mechanism**：
- 替换 `SUNLinSol_SPGMR` 为 `SUNLinSol_KLU`（SUNDIALS 6.0.0 提供）
- 构造稀疏 Jacobian（当前 matrix-free SPGMR 不需要 explicit J）：
  - 物理拓扑给定 sparsity pattern（element-element neighbor + element-river coupling + river-lake coupling）
  - colored finite-difference (per master plan §6 P8-KLU) 估计 J 元素
- 每 Newton iteration 做一次 sparse 直接因子分解 + 三角回代

**Cost**：4-6 epic-weeks
- sparse Jacobian 构造（pattern + coloring + FD evaluation）
- KLU 集成 + 内存 profile + perf 调优
- benchmark 验证（fill ratio + factor wall + memory peak）

**Reproducibility**：**weak alone** — KLU 自身 deterministic，但 CVODE WRMS norm 还是过 N_Vector reduction；KLU 不解 cross-N drift

**Speedup**：unknown — 强依赖 sparsity 与 fill ratio
- 文献 (e.g. Davis 2006 *Direct Methods for Sparse Linear Systems*) 显示水文 grid 类问题 fill ratio 常在 10-100×
- 大 case (heihe_x4, heihe_x16) 内存峰值可能爆掉（~10-100 GB 量级）
- factor wall 可能是 SPGMR 单步 wall 的 5-50×

**Risk**：**高**
- 内存爆掉 → heihe_x16 跑不起来（master plan §1.1.1 XLarge 档目标失效）
- factor wall → 大 case 单步时间反而变慢
- sparsity pattern 错算 → silent numerical errors

**Reversibility**：低（架构改动大，KLU 与 SPGMR 在 CVODE 内不能同时挂）

**Decision**：**DEFERRED — ADR-0002 pattern-only spike 先做**
- ADR-0002 spike 量化：heihe + heihe_x4 + heihe_x16 的 (a) fill ratio (b) memory peak (c) factor wall
- spike 数据回来再决定是否进入 KLU 主线
- **KLU 不是 P1e 必选 / 也不是 P2 必选**

---

## Decision Matrix

| Path | Reproducibility | Speedup at N=8 | Cost (epic-weeks) | Risk | Status |
|---|---|---|---|---|---|
| 1 — Serial NVec + StrictOMP RHS | strong | 1.5-2.4× (target) | 2-3 | low | **SELECTED (P1e)** |
| 2 — NVECTOR_REPRO_OMP custom backend | strong | 0.7-1.0× of stock NVECTOR_OPENMP | 3-5 | medium | fallback if Path 1 mode C FAIL |
| 3 — SPGMR + block-Jacobi precond | weak alone | potentially high (paired) | 2-3 | medium | P2 optimization (paired with Path 1) |
| 4 — SPGMR → KLU | weak alone | unknown | 4-6 | high (memory) | ADR-0002 pattern spike first |

---

## Decision

**采纳 Path 1 (Serial N_Vector + StrictOMP RHS) 作为 P1e epic 的实施技术主线**。理由：

1. **Reproducibility 直接闭环**：Serial NVec 按定义 deterministic；RHS 改用 owner-local canonical fold（已有 P1c `rhs_deterministic_gather()` 基础设施）也 deterministic。两者结合后跨 N bitwise 不再依赖任何额外保证。
2. **Speedup 路径清晰**：当前未并行的 RHS 占 wall 66.55%（heihe_x4 profile），理想 8 核 Amdahl 上界 2.39×；Path 1 把这个未被吃的资源吃掉。
3. **Cost 最低 + Risk 最低 + Reversibility 最高**：2-3 epic-weeks 完成；单 epic 内 revert 不留长尾。
4. **保留 fallback 选项**：Path 2/3/4 都没被否决，只是 sequencing 推后；P1e epic 内若 2×2 因果实验 mode C FAIL 或加速比不足，可按 Decision branches 顺序触发。

---

## Consequences

### Positive

- **F 路具体可达**：master plan v1.4 原 P7 阶段计划的"完整 RHS OpenMP + serial CVODE"，由 P1e epic 在 P1d closure 后立即接续完成
- **保留 option value**：Path 2/3/4 在 ADR-0002 + master plan §6 P1e.7 / §6 P1e.8 / ADR-0002 三个 cradle 内待命
- **strict-omp mode SHALL gate 有真实通过路径**：master plan §8.1 4-mode 表中 strict-omp 列 "production candidate (P1e 验收后)" 由本 ADR 提供具体实施路径
- **P2a entry condition 解锁**：原 "P1c-tag lock" 不足；新 "P1e-tag lock + strict-omp 3 SHALL gate PASS + ≥1.5× 加速" 提供 hard gate

### Negative

- **P2a 启动推迟**：从 P1c → P2a 直链变为 P1c → P1d → P1e → P2a（多一个 2-3 周 epic）
- **strict-omp mode 实施前 production 默认 NUM_OPENMP=1**：serial path 比 N=8 慢 11%，是 P1d E′ closure 的 trade-off

### Neutral

- **KLU 仍被社区/外审追问**：本 ADR 已 documented 不选 KLU 的 4 个原因（不能解 determinism / 内存爆 / factor wall / 架构 reversibility 低），交 ADR-0002 spike 量化后再回答；不属于 P1e blocker

---

## Validation Plan (= P1e.2 2×2 build matrix 因果实验)

2×2 build matrix mandatory **before** any P1e code change：

| Build | N_Vector | RHS | 目的 |
|---|---|---|---|
| A | `N_VNew_Serial` | Serial | canonical reference baseline |
| B | `N_VNew_OpenMP` | Serial | = 当前 `shud_omp`，复现 PR-H 10-25% 散度作 control |
| C | `N_VNew_Serial` | StrictOMP | **P1e production 候选** |
| D | `N_VNew_OpenMP` | StrictOMP | research 边界 |

**实测条件**：heihe + heihe_x4 (server) + keliya + qhh (Mac) 4 case × N∈{1,2,4,8} × 3 repeats × 90 天 cfg cap × 同 SUNDIALS 6.0.0 build × hash `CV_Y` state vector + `rivqdown.dat` + capture `nst/nfe/nli/nni/netf/ncfn` 等 15-key set。

**Acceptance criteria**：

| 检查 | 期望 | 结论 |
|---|---|---|
| A 同 build 同 N × 3 reps bitwise | YES | solver 本身 deterministic，是必要前提；若 A 自身跨 reps 不 bitwise → 整个 P1e 暂停查 toolchain |
| A 跨 N bitwise 且 B 跨 N 不 bitwise | YES | 确认 NVECTOR_OPENMP reduction 是主因，**不是** RHS race |
| C 跨 N bitwise + nst Δ=0 + 加速 ≥ 1.5× | YES | **F 路成立** → P1e proceed |
| C 跨 N 也分叉 | NO | RHS race / 共享 state / phase deps → 查更细 owner-compute 分解；可能需要 fallback Path 2 |
| C 加速 < 1.5× | NO | 进 Path 3 评估 (P1e.8 block-Jacobi precond) |
| D 跨 N 散度模式 ≈ B | YES | 进一步交叉验证 NVECTOR_OPENMP 是主因（与 RHS 实现无关） |

---

## References

### master plan

- `SHUD_openMP_master_plan.md` v1.5 / M10 顶部 quote block
- §1.1.2 strict 模式精度等级 P1d/P1e 行 (L123-L124)
- §4.13 OpenMP N_Vector + M10 修订段 (L468)
- §4.17 matrix-free SPGMR + M10 修订段 (L501)
- §6 P1d 全章 (L1760-L1831)
- §6 P1e 全章 (L1833-L1902)
- §7.2 RISK-NEW1 / RISK-NEW2 (L2561-L2562)
- §7.3 → P1e row / → P2a row (L2598-L2599)
- §8.1 4-mode 表 (L2614-L2623)

### openspec

- `openspec/changes/p1d-numa-governance/` (P1d epic capstone)
- `openspec/changes/p1e-strict-omp-rhs/` (P1e epic intake, 本 ADR 配套)
- `openspec/glossary.md` P1d / P1e baseline 集合

### Source code anchors

- `SHUD/src/Model/f.cpp:54` — `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)`
- `SHUD/src/Model/MD_rhs_core.cpp:802-811` — `StrictOMP/ProductionOMP` `std::abort()` 桩
- `SHUD/Makefile` shud_omp target — `-DSHUD_USE_OPENMP_NVECTOR=1 -lsundials_nvecopenmp`
- `SHUD/Makefile:140` — `SHUD_ENABLE_OPENMP_RHS ?= 0`
- `SHUD/src/Equations/cvode_config.cpp:259` — `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` (no preconditioner)

### SUNDIALS upstream

- SUNDIALS 6.0.0 source: `src/nvector/openmp/nvector_openmp.c`
  - `N_VDotProd_OpenMP` (`reduction(+:sum) schedule(static)`)
  - `N_VWSqrSumLocal_OpenMP` (WRMS norm 底层)

### docs/p1d/

- `docs/p1d/p1d_pr_h_final_run.md` — § "Post-verdict 修订" + "F 路（P1e 新 epic）顶层 plan"
- `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` — Mac reference data
- `docs/p1d/p1d_numa_root_cause.md` (PR-K forthcoming)
- `docs/p1d/p1d_first_touch_design.md` (PR-K forthcoming，标 deprecated)

### Forthcoming ADRs

- ADR-0003 (KLU spike, forthcoming): KLU pattern-only spike — quantify fill ratio + memory peak + factor wall for heihe + heihe_x4 + heihe_x16 (renamed from "ADR-0002 (forthcoming)" by PR-K per spec p1e-capstone Scenario "ADR-0002 与 P1e 实施一致性" L154 to eliminate ADR self-reference)

---

## Implementation closure (P1e epic close, 2026-06-25)

per tasks `p1e-strict-omp-rhs` §7.10.1: PR-K capstone 内 close out ADR-0002 (不延迟到 PR-M)，让 PR-M task 6.7 能 verify ADR 已 closed。

### Path 1 SELECTED 实施成果

Path 1 (Serial NVec + StrictOMP RHS) 由 P1e epic 13 sub-PR 完整实施 + 验收：

- **PR-F (#315)**: `ExecPolicy::StrictOMP` impl in `SHUD/src/Model/MD_rhs_core.cpp` per design D2 (单 `#pragma omp parallel` region + 4 RHS method 调用 + omp single scaffolding)
- **PR-G (#315)**: `SHUD/Makefile` `SHUD_ENABLE_OPENMP_RHS=1` 自动 wire `-fopenmp` (Linux) / `-Xpreprocessor -fopenmp -lomp` (Darwin) + `shud.cpp` startup 单点 read `SHUD_RHS_THREADS` env (拆为两段守门 form per tasks §3.5.2 允许)
- **PR-H (#316)**: `MD_rhs_core.cpp` L62-95 / L169-203 / L324-354 三处 inner `#pragma omp parallel for` first-touch loops 删除 (per design D4); PR-F omp single → `omp for schedule(static)` worksharing 改造 (per design D2) + TSan-confirmed nowait 规则
- **PR-I (#317)**: server SHALL closure — 3 SHALL gate verdict (AC-S1 跨 N bitwise PASS + AC-S2 mode C SHA == mode A reference SHA PASS + AC-S3 PARTIAL per-case threshold)
- **PR-J (#318/#333)**: Mac N=1 reverse-compat closure — 4 case × N=1 mode C SHA == 各自 mode A reference SHA PASS (6-case roll-up 含 server PR-I 2 case)

### 4 路 routing 实际触发分支

per Validation Plan §"Decision branches" + `docs/p1e/p1e_2x2_verdict.md` §6.2 + `docs/p1e/p1e_pr_i_strict_omp_verification.md` §8.2 + `docs/p1e/p1e_2x2_experiment.md` §6：

| Branch | 触发条件 | P1e 实测 eval | 触发? |
|---|---|---|:---:|
| **D12.1 (happy path)** | mode C cross-N bitwise PASS + nst Δ=0 + per-case speedup SHALL PASS (BOTH cases meet threshold) | AC-S1+S2+nst PASS, but heihe sp@8 1.066× < 1.3× → per-case SHALL not 全 met | **NOT triggered** |
| **D12.2 (Path 2 fallback = NVECTOR_REPRO_OMP)** | mode C cross-N FAIL → 切自研 deterministic NVECTOR backend | AC-S1 PASS on both cases (cross-N bitwise) → 不触发 | **NOT triggered** |
| **D12.3 (Path 3 fallback = SPGMR + block-Jacobi precond, PR-N)** | cross-N PASS + **BOTH cases** < own threshold (AND-gate per tasks §4.6) | heihe FAIL + heihe_x4 PASS → AND-gate **不满足** → 不触发 | **NOT triggered** (placeholder doc `docs/p1e/p1e_pr_n_block_jacobi.md` 已 PR-K 写出) |
| **D12.4 (Path 4 deferred = ADR-0003 KLU spike)** | D12.3 触发但 fallback 失败 → KLU 深度 refactor | D12.3 未触发, 无递进条件 → 不触发 | **NOT triggered** |

**实际触发**：**§4.6.2 partial-closure → SHIP** (per tasks §4.6.2 + user 决策, 详 `docs/p1e/p1e_2x2_verdict.md` §6.3-§6.4 SHIP rationale + small-case carve-out)。Path 1 实施成功；Path 2/3/4 全部留作 future epic option（per ADR-0002 §"Consequences" Positive 第 2 项 "保留 option value"）。

### Decision Matrix 状态 update

| Path | Status (P1e epic close) |
|---|---|
| 1 — Serial NVec + StrictOMP RHS | **SELECTED + Implemented (P1e epic close, 2026-06-25)** — SHIP via §4.6.2 partial-closure |
| 2 — NVECTOR_REPRO_OMP custom backend | fallback option preserved (not triggered in P1e; future epic if needed) |
| 3 — SPGMR + block-Jacobi precond | P2 optimization option preserved (not triggered in P1e per D12.3 AND-gate 不满足; placeholder `docs/p1e/p1e_pr_n_block_jacobi.md` documented) |
| 4 — SPGMR → KLU | deferred to ADR-0003 spike forthcoming (not in P1e scope) |

### Capstone references

- `docs/p1e/p1e_summary.md` (P1e epic capstone summary, §7 verdict + §6 P1d carve-out closure)
- `docs/p1e/p1e_2x2_verdict.md` (3 SHALL gate verdict + D12 routing decision)
- `docs/p1e/p1e_2x2_experiment.md` (Phase 1 + Phase 2 综合 + D12 4 branch 实测 eval 表)
- `docs/p1e/p1e_perf_baseline.md` (server PR-I 24-cell + Mac PR-J 4-cell perf data)
- `docs/p1e/p1e_strict_omp_design.md` (ExecPolicy::StrictOMP 实现细节 + 单 parallel region rationale)
- `docs/p1e/p1e_thread_split.md` (SHUD_RHS_THREADS vs OMP_NUM_THREADS runbook)
- `docs/p1e/p1e_first_touch_removal.md` (PR-H 3 处删除记录 + allocation/load-time first-touch 保留 verify)
- `docs/p1e/p1e_report.md` (P1e epic executive report)
