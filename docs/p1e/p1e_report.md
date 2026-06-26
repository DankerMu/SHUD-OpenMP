# P1e — epic executive report

executive 层 report，面向项目所有者 + 跨 epic 复盘。详细数据 / 实测表 / 错误纠正逐项分析见 `docs/p1e/p1e_summary.md` + `docs/p1e/p1e_perf_baseline.md` + `docs/p1e/p1e_2x2_experiment.md` + `docs/p1e/p1e_strict_omp_design.md` + `docs/p1e/p1e_pr_i_strict_omp_verification.md` + `docs/p1e/p1e_mac_reverse_compat.md` + `docs/p1e/p1e_2x2_verdict.md`。

## §1 Epic ID + 时间线

| Field | Value |
|---|---|
| Epic | #283 |
| 起 | 2026-06-24 (P1d epic capstone 关闭后 P1e intake) |
| 止 | 2026-06-25 (PR-K capstone + 全 SHALL gate verdict 关闭) |
| 跨度 | 2 天 |
| PR 总数 | **14 PR total** (11 base sub-PR `A,B,C,D,E,F,G,H,I,J` + PR-B0 audit-required + PR-K capstone + PR-L tag + PR-M PROMOTE; PR-J Phase 6 修订 #333 计入 PR-J 不另算) |
| 平均 wall（含 server Slurm + Mac local + 2×2 实验 4h10m + capstone 文档化） | ~10-12 hours engineer time |

## §2 Status

**SHIP via §4.6.2 partial-closure**.

不是简单全 SHALL gate PASS (heihe AC-S3 1.066× < 1.3× threshold)，是 **§4.6.2 partial-closure**: AC-S1 + AC-S2 全 PASS, AC-S3 PARTIAL (heihe FAIL but heihe_x4 ≥ 1.5×) → AND-gate (BOTH FAIL 才触发 D12.3) 不满足 → user 决策 SHIP per `docs/p1e/p1e_2x2_verdict.md` §6.4 + tasks §4.6.2。

详 `docs/p1e/p1e_summary.md` §7 verdict + §6 P1d carve-out closure。

## §3 What was attempted

P1d epic 关闭后 carve-out = "真正应并行的 RHS 还没并行"（master plan v1.5 / M10 §6 P1d.7）。P1e 按 ADR-0002 Path 1 (Serial NVec + StrictOMP RHS)：

1. **2×2 build matrix 因果实验**（PR-A/B/B0/C/D/E）：4 mode × 4 N × 3 reps × 4 case = 192 cell capstone spec 上界；本 epic 实跑 180 cell（mode D Phase 2 96 cell deferred）。Phase 1 mode A/B 96 cell × 2 platform = 192 cell；Phase 2 mode C 24 cell server + 12 cell Mac N=1 = 36 cell。
2. **rivqdown.dat tout-boundary recompute**（PR-B0）：审计发现 output cache 是 internal cache → 新增 `Model_Data::recompute_for_output(N_Vector, double)` helper 在 MainLoop 内 summary 与 ExportResults 之间调 rhs_update + rhs_flux + rhs_apply 全链。perf cost +7.7% (post-Phase6 fix per outer #323 Phase 4.5 verifier)。
3. **`ExecPolicy::StrictOMP` 实施**（PR-F）：`SHUD/src/Model/MD_rhs_core.cpp` 新增 ExecPolicy::StrictOMP case，4 RHS method 调用包在单 `#pragma omp parallel` region 内。
4. **-fopenmp wire + SHUD_RHS_THREADS env split**（PR-G）：`SHUD/Makefile` `SHUD_ENABLE_OPENMP_RHS=1` 自动 wire `-fopenmp` (Linux) / `-Xpreprocessor -fopenmp` (Darwin)；`shud.cpp` startup 单点 read `SHUD_RHS_THREADS` env。守门形式选 "拆为两段" (per tasks §3.5.2 允许)，spec L192 由 PR-K amend to allow both forms。
5. **steady-state first-touch removal + omp single → omp for**（PR-H）：删 `MD_rhs_core.cpp` L62-95 / L169-203 / L324-354 三处 inner `#pragma omp parallel for` first-touch loops；PR-F 初版的 `#pragma omp single` scaffolding 改造为 `#pragma omp for schedule(static)` work-sharing + TSan-confirmed `nowait` 规则（同 bucket disjoint slot 可 nowait，phase 边界末 loop SHALL NOT nowait）。
6. **server SHALL gate 验收**（PR-I）：server Slurm cn14 + cn15 跑 24 cell (heihe + heihe_x4 × 4 N × 3 reps)。3 SHALL gate verdict + binary symbol verification + runtime diagnostic + 15-key cvode_stats 完整 archived。
7. **Mac N=1 reverse-compat 验收**（PR-J）：Mac local 跑 4 case × N=1 mode C；4 case SHA == 各自 mode A reference SHA → cross-platform determinism chain 闭合（Apple Clang + libomp / GCC + libgomp 同 binary-identical 输出 N=1 边界）。

## §4 What was NOT attempted

| 未做 | 原因 | 状态 |
|---|---|---|
| D12.2 NVECTOR_REPRO_OMP custom backend | AC-S1 跨 N bitwise PASS → 后端 fallback 不需要 | NOT triggered |
| D12.3 block-Jacobi precond (PR-N) | AND-gate (BOTH FAIL) 不满足 → fallback 不触发 | NOT triggered (placeholder doc `p1e_pr_n_block_jacobi.md` 已 PR-K 写出) |
| D12.4 KLU spike (ADR-0003 forthcoming) | D12.3 未触发，无递进条件 | NOT triggered |
| mode D Phase 2 96 cell | research 边界，不是本 epic verdict input (per tasks §2.5.1) | deferred to future epic |
| Mac advisory cross-N | 本 epic SHALL scope 仅 Mac N=1 reverse-compat | deferred to future epic |
| heihe small-case ROI 优化 | 1.066× 是 OMP overhead floor 设计预期 → user 决策 SHIP per §4.6.2 carve-out | accepted as carve-out |
| toolchain investigation (tasks §11.A) | Phase 1 mode A 全 24-cell × 3-rep bitwise PASS → 不触发 | NOT triggered |

## §5 What changed in baseline

| 维度 | P1d-tag (`a82bf336`) | P1e-tag |
|---|---|---|
| SHUD pin | `210ac191...` (Kahan revert + first-touch loops, 全 mode Serial RHS) | `3341368...` (ExecPolicy::StrictOMP + -fopenmp wire + SHUD_RHS_THREADS env + 3 first-touch loops 删 + omp single→omp for) |
| build matrix | 2 (`shud` + `shud_omp`) | 4 (`shud` mode A + `shud_omp` mode B + `shud SHUD_ENABLE_OPENMP_RHS=1` mode C + `shud_omp SHUD_ENABLE_OPENMP_RHS=1` mode D) |
| production default | `OMP_NUM_THREADS=1` (serial fallback, P1d E′ closure) | `SHUD_RHS_THREADS` per case (heihe `=1` carve-out / heihe_x4 `=4` 推荐) |
| 4-mode strict-omp 列状态 | "production candidate (P1e 验收前)" | "SHIP via §4.6.2 partial-closure (P1e 验收后)" |
| rivqdown.dat 输出语义 | internal cache | tout-boundary recompute (rhs_update + rhs_flux + rhs_apply 全链) |
| nst Δ 跨 N | mode B 不闭合 (heihe \|Δ\|≤84 / heihe_x4 \|Δ\|>200) | mode C 闭合 (0 / 0) |
| ADR-0002 Status | `Accepted (2026-06-24)` | `Implemented (P1e epic close, 2026-06-25)` |

## §6 Lessons learned

### 6.1 mode 矩阵 verification before code change works

ADR-0002 强制要求 2×2 build matrix 因果实验 **before** any P1e code change。本 epic 严格遵守：PR-C/D 跑 Phase 1 mode A/B 实测后 PR-F 才开始改 RHS。这避免了 P1c/d era "先猜根因再修代码" 的循环（P1c hypothesis SPGMR precond drift / P1d hypothesis NUMA first-touch 全部被 fact-check 推翻）。

### 6.2 AND-gate 设计的 partial-closure 价值

D12.3 设计 AND-gate (BOTH FAIL 才触发 fallback)，而不是 OR-gate (任一 FAIL)。本 epic 实测 heihe FAIL but heihe_x4 PASS，AND-gate 不触发 → 进 §4.6.2 partial-closure → user 决策 SHIP。若是 OR-gate，会触发 D12.3 PR-N block-Jacobi 实施，但实际上 heihe_x4 已 ≥1.5× 不需要 fallback → AND-gate 设计避免了 over-engineering。

### 6.3 small-case carve-out 是设计预期

heihe (6335 cells) sp@8 1.066× 不达 1.3× threshold 不是 implementation bug，是 fork-join overhead vs per-thread workload 比的物理 limit。详 `docs/p1e/p1e_perf_baseline.md` §6 small-case carve-out 三因素分析。production-target mesh (heihe_x4 ~25k cells) 1.729× ≥ 1.5× threshold 是真正的 ROI 收益 site。

### 6.4 单 parallel region rule + nowait 规则

PR-H 实施时的 TSan race fix 表明：单 parallel region 设计 + omp for nowait 规则在大 codebase 内仍然需要 careful field aliasing 分析。ET bucket → lateral bucket 的 `hot.u_effKH[inabr]` 跨 thread read 是典型 race trap，PR-H 用 "末 loop SHALL NOT nowait" 规则规避。future epic 若引入新 cross-thread field read 需重新评估。

### 6.5 first-touch loops 在 Serial path 上是 dead code

PR-H 删除 3 处 first-touch loop 后 mode A path 仍 bitwise reproduce → 这些 loop 在 Serial path 上从来未改变过 SHA → P1c/d era "writer first-touch" hypothesis 在 ADR-0002 Fact #1 (`f.cpp::54` 始终 ExecPolicy::Serial RHS) 确立后即被推翻；P1e PR-H 是把这点 documented 化 + 实证化。

## §7 Cross-epic decision chain (P1c → P1d → P1e)

| Epic | hypothesis | verdict | forward |
|---|---|---|---|
| P1c | 8-site canonical-reduction Kahan injection 可闭合跨 N drift | PARTIAL CLOSURE (heihe \|Δ_nst\|=84 残留 → carve-out 推 P1d) | NUMA writer noise hypothesis → P1d |
| P1d | NUMA first-touch + numactl --interleave + Kahan revert 可闭合 drift | PARTIAL CLOSURE via E′ (fact-check 5/5 推翻 hypothesis：drift 来自 NVECTOR_OPENMP reduction, not NUMA) | F 路 (Serial NVec + StrictOMP RHS) per ADR-0002 → P1e |
| P1e | ExecPolicy::StrictOMP + Serial NVec 可达 bitwise + ≥1.5× speedup | **SHIP via §4.6.2 partial-closure** (mode C cross-N bitwise PASS + heihe_x4 1.729× ≥ 1.5× PASS, heihe small-case 1.066× carve-out 接受) | P2a entry condition 解锁 |

3 epic 共同 lesson: **architectural correctness > hypothesis-driven fix**. P1c/d 两 epic 都尝试在错误 architecture 上加补丁；P1e 直接换 architecture (mode C = Serial NVec + StrictOMP RHS) 一举闭合两 epic 的 forward debt。

## §8 Risk register update

per master plan v1.5 / M10 §7：

| Risk | P1d era 状态 | P1e era 状态 |
|---|---|---|
| RISK-NEW1 (NVECTOR_OPENMP reduction tree 不固定) | identified, not mitigated | **mitigated via mode C (Serial NVec)** |
| RISK-NEW2 (SPGMR 无 preconditioner) | identified, deferred to P1e | deferred to P2/P3 epic (per ADR-0002 Path 3 not triggered) |
| RISK-NEW3 (heihe small-case OMP overhead floor) | not yet observed | **identified via PR-I + carve-out accepted** |

## §9 D11 7-tag chain final state

| tag | SHA | epic | status |
|---|---|---|---|
| B1-tag | `<immutable>` | B1 | immutable |
| B1a-tag | `f7f992c` | B1a | immutable (lock_branch=true) |
| B1b-tag | `<immutable>` | B1b | immutable |
| P1-update-omp-tag | `ff21c75c` | P1 | immutable (lock_branch=true) |
| P1c-tag | `<immutable>` | P1c | immutable |
| P1d-tag | `a82bf336` | P1d | immutable |
| **P1e-tag** | **`<TBD by PR-L>`** | P1e | **new (PR-L pending)** |

D11 由 6-tag chain 升至 **7-tag chain**。前 6 tag SHA 不变（per master plan §6 D11 immutability rule）。

## §10 Forward handoff to P2a

P2a 启动前置（per master plan §6 P2a forthcoming + ADR-0002 §"Consequences" Positive 第 4 项）：

- `P1e-tag` push 完成
- `baseline/P1e` lock_branch=true 完成
- 3 SHALL gate (AC-S1 + AC-S2 PASS + AC-S3 partial-closure SHIP) 已 verdict
- heihe_x4 sp@8 ≥ 1.5× 闭合 (1.729×)
- ADR-0002 Status = Implemented + 4 路 routing 实际触发分支 documented
- OpenSpec p1e-strict-omp-rhs + p1e-capstone 2 spec PROMOTE
- glossary 4 new terms (P1e-tag / baseline/P1e / strict-omp mode / 2×2 build matrix) 入册

P2a additional forward considerations（forthcoming, 非本 epic scope）：

- `OMP_SCHEDULE` tuning per case best speedup
- cache-line padding for owner-local SoA fields (per `docs/adr/0001-soa-hot-fields.md` deferred items)
- NUMA cross-socket migration cost quant (dual-socket server entry condition)
- mode D (OpenMP NVec + StrictOMP RHS) 96-cell verification 是否仍 deferred 或 P2a 内启动
