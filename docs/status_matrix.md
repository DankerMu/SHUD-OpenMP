# SHUD-OpenMP — 阶段 × Benchmark 状态矩阵

阶段 Go/No-Go 决策的权威状态来源，对应 `openspec/changes/s0-baseline-lock/specs/status-matrix/spec.md`。行为 master plan §3 的阶段，列为 `benchmarks/INDEX.md` 里登记的 7 个 benchmark + `aggregate` 汇总列。单元格取值：

- **PASS** — 该 case 通过本阶段验收（附证据链接）
- **FAIL** — 已验证失败（污染 aggregate）
- **BLOCKED** — 受上游/外部阻塞无法评估（如数据缺失）
- **PENDING** — 尚未尝试；属于未来阶段
- **N/A** — 该 case 在结构上被排除在本阶段之外

更新走 PR；CI 通过 PR 评论形式提议变更（`status-matrix` spec L19，不自动 push）。本矩阵文件是**唯一权威**，各阶段文档和 PR 摘要引用它，不反向。

> _最近一次更新：2026-06-23（P1c epic capstone：PR-A #257 / PR-B #258 / PR-C #259 / PR-D #260 / PR-E #261 / PR-F #262 / PR-G #263 / PR-H #264 / PR-I #265 / PR-J #266 / PR-K #267 本 PR / PR-L 下一 PR / PR-M 待 PROMOTE 全 merged；SHUD pin de9545d → 3a0004c (Kahan-injected, openmp-baseline pushed, master untouched)；8-site canonical-reduction Requirement CLOSED via helper-wrap (PR-B/C/D/E)；§4.4 A3a + §4.5 nst Δ=0 PARTIAL CLOSURE + carve-out 推 P1d (per master plan §3 fallback option 2 + spec L100-L103); design D9 decision branch 2 CONFIRMED; NUM_OPENMP=1 binary reverse-compat 在 Kahan-injected baseline/P1c 上 FAIL (trade-off; D11 tag-immutability 保留); 详 [`docs/p1c/p1c_summary.md`](p1c/p1c_summary.md) + [`docs/p1c/p1c_a3a_root_cause.md`](p1c/p1c_a3a_root_cause.md) + [`docs/p1c/p1c_perf_baseline.md`](p1c/p1c_perf_baseline.md) + [`docs/p1c/p1c_pr_h_server_first_run.md`](p1c/p1c_pr_h_server_first_run.md) + [`docs/p1c/p1c_pr_i_kahan_injection.md`](p1c/p1c_pr_i_kahan_injection.md) + [`docs/p1c/p1c_pr_j_reverse_compat.md`](p1c/p1c_pr_j_reverse_compat.md)。**P1 epic 上一次更新 (2026-06-22 earlier)**：P1-update-omp-tag `ff21c75c` deref `003f58d` / SHUD pin `07c677f`；`baseline/P1` lock_branch=true 已锁；详 [`docs/p1_summary.md`](p1_summary.md) + [`docs/p1_rhs_snapshot_bitwise.md`](p1_rhs_snapshot_bitwise.md) + [`docs/p1_fullrun_bitwise.md`](p1_fullrun_bitwise.md) + [`docs/p1_perf_baseline.md`](p1_perf_baseline.md)。_

## 矩阵

| 阶段       | keliya | xinanjiang_upstream | qinyijiang | kashigeer            | qhh     | heihe         | heihe_x4      | aggregate |
|-----------|--------|---------------------|------------|----------------------|---------|---------------|---------------|-----------|
| **B0**    | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
| **B1a**   | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
| **B1b**   | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS (UNCONDITIONAL ship; `B1-tag` 已发布 aliasing main HEAD) |
| **Opt-IO**| PENDING (可选) | PENDING (可选) | PENDING (可选) | PENDING (可选) | PENDING (可选) | PENDING (可选; M7 trim 后 heihe 退回可选 per PR-G #214; 1.90% << 50%) | PENDING (可选; M7 trim 后 heihe_x4 0.19% << 50% per PR-G #214) | PENDING (可选) |
| **P1**    | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS (§1.1.1 WARNING per design D5 NG3; P1 lock NOT BLOCKED; P7 final-fusion debt logged) |
| **P1c**   | PARTIAL @ Mac (pre-Kahan only) | PARTIAL @ Mac (pre-Kahan only) | PARTIAL @ Mac (pre-Kahan only) | N/A (deferred-upstream) | PARTIAL @ Mac (pre-Kahan only) | PARTIAL @ server | PARTIAL @ server | **PARTIAL CLOSURE** (8-site canonical-reduction Requirement CLOSED via PR-B/C/D/E helper-wrap + Kahan held-in-reserve PR-G applied at PR-I; bit-level A3a + nst Δ=0 cross-N carved-out 推 P1d (2026-06-23 修订, 原 "推 P9" 与 master plan §6 P9 production scope 命名冲突); NUM_OPENMP=1 reverse-compat 在 Kahan-injected HEAD FAIL/D11 tag-immutability 保留; Mac 4-case 16-cell scan documented PR-F 但 P1-update-omp-tag Mac canonical rivqdown.dat SHA reference NOT directly available in P1 era docs → spec L154-157 Scenario DEFERRED to P1d + retroactive reference capture — per `docs/p1c/p1c_summary.md` §1 完成定义 + §5.2 P1d hand-off + §6.8 Mac SHALL Scenario deferred note + §8 反向兼容判定) |
| **P1d**   | PENDING (2026-06-23 新增 carve-out 阶段) | PENDING | PENDING | N/A (deferred-upstream) | PENDING | PENDING | PENDING | **PENDING** (master plan §6 P1d.1-6: NUMA env + source first-touch + Kahan revert + Mac SHALL closure; 严格 hard gate per §7.3 → P2a 修订; baseline/P1d 待 P1d epic 完成后建立) |
| **P2a**   | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING (P1d 闭环后启动, per §7.3 修订 2026-06-23)   |
| **P2b**   | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P3**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P4**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P5**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P6**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P7**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P8**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P9**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |

### B0 行证据

| Case                | 单元格        | 证据                                                                 |
|---------------------|---------------|----------------------------------------------------------------------|
| keliya              | PASS          | `benchmarks/keliya/B0_output/` 3 次跑 bitwise 一致（#11 PR #26）+ snapshot_t*.bin × 3（#9 PR #24） |
| xinanjiang_upstream | PASS          | `benchmarks/xinanjiang_upstream/B0_output/` 3 次跑 PASS（#11 PR #26）+ snapshot × 3（#9） |
| qinyijiang          | PASS          | `benchmarks/qinyijiang/B0_output/` 3 次跑 PASS（#11）+ snapshot × 3（#9） |
| kashigeer           | **N/A (deferred-upstream)** | `benchmarks/kashigeer/B0_output/DEFERRED.txt` — 上游 X76 forcing 段在本地 + 服务器两端都缺（#11 PR #26 + #12 PR #29 已交叉核对）。**S0-13 spec 修订**：`benchmarks/INDEX.md` 把 kashigeer endpoint 改为 `deferred-upstream`；`status-matrix` + `rhs-profile-gate` spec 同步把 deferred-upstream 单元格从 A0 bitwise / cvode_stats / snapshot 场景中排除。 |
| qhh                 | PASS          | `benchmarks/qhh/B0_output/` 3 次跑 PASS（4 个 .dat，含 3 个 lake）（#11）+ snapshot × 3（#9） |
| heihe               | PASS @ server | `benchmarks/heihe/B0_output/` 3 次跑 PASS（服务器 cn08，走 Slurm）（#12 PR #29） |
| heihe_x4            | PASS @ server | `benchmarks/heihe_x4/B0_output/` 3 次跑 bitwise PASS（服务器 cn21，Slurm 8256，2026-06-17）。wall_times：1216/1211/1214s（90 天窗口）；共享 SHA：`3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06`；binary SHA：`5b95f617580a41d900961d79102382d027cba32bafdda25d900eda7aea237a2e`（PROFILE=0 / DUMP=0，SHUD `78c37a1`）。4 个 missing_manifest_files（NWM 上游缺口同 #11）。 |

### Aggregate B0 = PASS

按修订后的 `status-matrix` spec（"aggregate = PASS iff 所有非 N/A 单元格都 PASS"）：
- 6 个单元格 PASS（keliya / xinanjiang_upstream / qinyijiang / qhh / heihe / heihe_x4）
- 1 个单元格 N/A（kashigeer，deferred-upstream，按 spec 排除）

aggregate 列 = **PASS**（2026-06-17）。

## B1a 行证据

> **2026-06-21 capstone（PR-12 #156）**：B1a 整体 = master plan §3 "S0–S4 完成"。S0 + S1 + S2 + S3 + S4 全部完成（PR-1 #144 through PR-11 #155 已 merge 进 `baseline/B1a`；详见 [`docs/b1a_summary.md`](b1a_summary.md) 时间线）。下表 evidence 来自 PR-12 capstone 时刻（SHUD `0b3998d`），4-case Mac 本地 + qhh 3 lake outputs bitwise vs B0-tag 全 PASS；7 grep gate 0 hits 全部 enforce；heihe / heihe_x4 通过免密 SSH 直接服务器 Slurm 8537/8538 (cn08) bitwise PASS @ server；aggregate = PASS；follow-up issue #171 已同步关闭（2026-06-21 closedAt）。

| Case | 单元格 | 证据（S0–S4 complete） |
|---|---|---|
| keliya | PASS | PR-12 本地 4-case 复跑 bitwise vs B0-tag PASS（SHA `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc`）。S2.1-S2.17 + S3a/S3b/S3c + S4.1-S4.7 全部已 merged（PR-1..PR-11） |
| xinanjiang_upstream | PASS | 同上 4-case capstone 复跑 PASS（SHA `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020`） |
| qinyijiang | PASS | 同上 4-case capstone 复跑 PASS（SHA `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57`） |
| kashigeer | N/A (deferred-upstream) | 同 B0：上游 X76 forcing 段缺失，CI matrix 排除 (spec b0-tag-ci-integration L24-28 + INDEX 已标 deferred-upstream)，B1a 阶段沿用 N/A |
| qhh | PASS | 4-case capstone 复跑 rivqdown PASS（SHA `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce`）+ 3 个 lake outputs（lakystage / lakqrivin / lakqrivout）bitwise PASS |
| heihe | PASS @ server | PR-12 直接在 `210.77.77.22:32099` Slurm `cn08` 跑 90 天截断（cfg.para START=14245 END=14335 / NUM_OPENMP=1 / SHUD `0b3998d`），JobId 8537，wall 500s，SHA256 `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f` bitwise vs B0-tag golden |
| heihe_x4 | PASS @ server | PR-12 同 SSH/Slurm 通道，JobId 8538 (cn08)，wall 1290s，SHA256 `f90601ef5738b972d688016ba1ee74f92ecb54faddaf46e4e2232f9d46567524` bitwise vs B0-tag golden |

Aggregate gate（PR-12 capstone 验收）：
- 4-case (keliya / xinanjiang_upstream / qinyijiang / qhh) Mac 本地 bitwise vs B0-tag PASS + qhh 3 lake outputs bitwise PASS
- 7 grep gate enforce 0 hits（capstone 静态校验）：
  - `MD_f_omp.cpp` 文件不存在（PR-8 删 TU）
  - `PassValue\b` tree-wide 0 hits（PR-11 退役）
  - `SHUD_LEGACY_OMP_RHS` tree-wide 0 hits（PR-8 退役）
  - `LEGACY_RHS` tree-wide 0 hits（PR-8 退役）
  - `_OPENMP_ON` tree-wide 0 hits（S1d.2 #48 + #50 follow-up）
  - `USE_RHS_CORE` tree-wide 0 hits（S1d.1 #47 退役）
  - `N_VDestroy_Serial` tree-wide 0 hits（S1d.2 #48 退役）
  - 加 bonus：`f_update_omp|f_loop_omp|f_applyDY_omp` 三个 _omp receiver 函数名 tree-wide 0 hits（PR-8 capstone）
- SHUD 在 `openmp-baseline` 分支 commit `0b3998d`（PR-1..PR-11 5-step push workflow 严格遵守）
- CI workflow `serial-baseline.yml` 已升级到 B1a 定版 final state：1-axis matrix (case only) + S2 capstone grep gate + PassValue 0-hit gate + topology_manifest 模式校验 + adjacency fallback 单测 + snapshot 90d HARD-fail gate + CVODE 15-key SHA256 diff
- Server heihe / heihe_x4 90 天截断 Slurm bitwise validation：PR-12 直接通过免密 SSH 在 `210.77.77.22:32099 cn08` 上 Slurm 8537 (heihe) + 8538 (heihe_x4) 跑出 bitwise PASS（详见 B1a 行 evidence 段；issue #171 同步关闭）

## B1b 行证据

> **2026-06-22 capstone（PR-16 #207 #188 + #189 + 本 #190 PR）**：B1b 整体 = master plan §3 "S5 + S6b + S6c 全部完成"。S5a/S5b/S5c/S5d/S6b 全部完成（PR-1 #191 through PR-16 #207 已 merge 进 `baseline/B1b`；#189 创建 `B1b-tag = 18a0c908…` annotated tag + push origin + branch protection `lock_branch=true`；详见 [`docs/b1b_summary.md`](b1b_summary.md) 时间线）。下表 evidence 来自 PR-16 #207 capstone 时刻（SHUD `71b3a1ae` 在 `openmp-baseline`），4-case Mac 本地 + 2-case 服务器 Slurm 8662-8667 (cn03) 全部 bitwise vs B0-tag PASS；4 Mac canonical summary SHA ≡ B0 `repeatability.txt sha256_run1`；aggregate = **PASS (CONDITIONAL ship)** —— #185 S2.17 PI sign-off OPEN / #205 SoA/AoS sync drift P-strict pre-req OPEN / D9 fast-path BLOCKED / C8 forward-compat 允许 B1c-tag stacking。

| Case | 单元格 | 证据（S5 + S6b + S6c-12a complete） |
|---|---|---|
| keliya | PASS | T1 Mac 4-case 3-run canonical summary SHA `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` ≡ `benchmarks/keliya/B0_output/repeatability.txt sha256_run1` |
| xinanjiang_upstream | PASS | T1 Mac canonical SHA `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` ≡ B0 |
| qinyijiang | PASS | T1 Mac canonical SHA `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` ≡ B0 |
| kashigeer | N/A (deferred-upstream) | 同 B0/B1a：上游 X76 forcing 段缺失，CI matrix 排除，B1b 沿用 N/A |
| qhh | PASS | T1 Mac canonical SHA `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e`（覆盖 rivqdown + lakystage + lakqrivin + lakqrivout + cvode_stats）≡ B0 |
| heihe | PASS @ server | T2 cn03 Slurm 8662/8663/8664（PR-16 capstone）wall 480/479/480s，summary SHA `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` 3-run identical；rivqdown SHA ≡ PR-12 B1a-tag golden `55abad28…`（亦在 #177 server re-run 复证：8678 wall 479s 同 SHA） |
| heihe_x4 | PASS @ server | T2 cn03 Slurm 8665/8666/8667 wall 1196/1192/1191s，summary SHA `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` 3-run identical；rivqdown SHA ≡ B0-tag/B1a-tag golden `f90601ef…`（亦在 #177 8679 wall 1192s 复证） |

Aggregate gate（PR-16 #207 capstone + #189 tag + 本 PROMOTE PR 验收）：

- 4-case Mac canonical summary SHA + 2-case Server canonical summary SHA 全部 byte-identical 3-run；4 Mac SHA ≡ B0 `repeatability.txt sha256_run1`；2 Server rivqdown.dat ≡ B0/B1a-tag golden。
- 水量平衡（`docs/B0_vs_B1b_water_balance_report.md`）closure-error delta = 0 bit-by-bit on 6/6 cases（远低于 0.1% 相对容差）。
- SHUD 在 `openmp-baseline` 分支 commit `71b3a1ae`（S5a → S6b 全 5-step push workflow 严格遵守，`SHUD/B1b_CHANGELOG.md` 12 sections evidence trail 完整）。
- `B1b-tag` annotated tag = `96e224da…`（指向 commit `18a0c908…`，含 SHUD pin `71b3a1ae`）；`baseline/B1b` 分支 protection `lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false`（D11 一次锁死 enforced）。
- ship caveats — 全部 RESOLVED 通过 PR-19 #210 升级为 **UNCONDITIONAL ship**（详 `docs/b1b_summary.md` §"B1b ship status" + `SHUD/B1b_CHANGELOG.md` S6b.2/S6b.4/S6b.5 + `docs/s217_lake_formula_audit.md` §E）：
  - #185 S2.17 lake formula PI 审查 — **RESOLVED (E2 signed)** via PR-19 #210（DankerMu 作为 SHUD-System/SHUD upstream-org owner 签 E2 "formula correct, no change"；design.md Open Q1 同时关闭：PI delegate = upstream-org owner / three-surface sign-off pattern）
  - #186 S6b.2 — **CLOSED-via-PI-E2**：原 SKIP path 在 #185 E2 sign-off 后从 "FORECAST per C8" 升级为 "consistent with signed PI E2 directive"
  - #205 `rhs_flux` lake pass-1 SoA/AoS sync drift — **RESOLVED (post-B1b cleanup before P1)** via SHUD `de75743`/`9a376f7` + PR-18 #209；4-case Mac 2-run canonical SHA bitwise vs B1b-tag baseline；NOT retroactively part of B1b per D11；P-strict pre-req gap 已闭；同时 strengthens #185 E2 verdict（audit §A.4/§B.4 strict-reading concern 由 SoA-sync 修复而消解）
  - D9 fast-path trigger #2 — **TRIGGERED in PR-19 #210**：`B1-tag` annotated tag 创建 aliasing main HEAD（含 #205 cleanup + PI E2 sign-off）；`B1a-tag` (`f7f992c…`) + `B1b-tag` (`18a0c908…`) 保留 immutable per D11 history；下游 P1+ SHOULD use `B1-tag`
  - C8 forward-compat — **UNUSED for this ship**（PI 签 E2 不签 E1，B1c-tag stacking 不触发）；仍是 codebase convention 留给未来 P-strict 阶段可能的 overrule。

## P1 行证据

> **2026-06-22 capstone（PR-L #224 P1-update-omp-tag + baseline/P1 lock + 本 PR-M #225 docs PROMOTE）**：P1 整体 = master plan §3 "B1b + MD_update.cpp 三 owner loop OpenMP 并行后的 first parallel candidate baseline"。Phase A (M7 trim) + Phase B (Opt-IO 决策) + Phase C.audit (5 函数 (a) safe) + Phase C.implement (PR-D element + PR-E river + PR-F lake 3-pragma stack) + Phase C.verify (PR-H snapshot + PR-I/J fullrun + PR-K1/K2 scaling) + Phase D.tag (PR-L) 全部完成（13 PR merged 进 `main`；#226 PR-N PROMOTE 是 next 收尾，不影响 P1-update-omp-tag immutable 内容）；详见 [`docs/p1_summary.md`](p1_summary.md) 时间线。下表 evidence 来自 PR-K2 #223 capstone 时刻（SHUD `07c677f` 在 `openmp-baseline`），3 anchor 在 NUM_OPENMP=1 下全 PASS（Mac 4-case RHS snapshot 12/12 + Mac 4-case fullrun 8/8 + server 2-case fullrun 4/4 = 24/24）；§1.1.1 acceptance = **WARNING（P1 不阻塞 per design D5 NG3）**。

| Case | 单元格 | 证据（P1 = NUM_OPENMP=1 vs B1b/B1-tag bitwise + scaling A3a where applicable） |
|---|---|---|
| keliya | PASS | PR-H Mac RHS snapshot 3 t-anchor bitwise PASS + PR-I Mac fullrun canonical SHA `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K1 Mac × N {1,2,4,8} 4/4 A3a PASS bitwise |
| xinanjiang_upstream | PASS | PR-H 3 t-anchor bitwise PASS + PR-I canonical SHA `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K1 4/4 A3a PASS bitwise |
| qinyijiang | PASS | PR-H 3 t-anchor bitwise PASS + PR-I canonical SHA `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K1 4/4 A3a PASS bitwise |
| kashigeer | N/A (deferred-upstream) | 同 B0/B1a/B1b：上游 X76 forcing 段缺失，CI matrix 排除，P1 沿用 N/A；manifest `trimmed_path: null`（PR-A schema 升级标 deferred-upstream） |
| qhh | PASS | PR-H 3 t-anchor bitwise PASS（含 lake 路径）+ PR-I canonical SHA `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K1 4/4 A3a PASS bitwise（lake loop pragma 验证）|
| heihe | PASS @ server | PR-J cn03 Slurm jobid `8794` Elapsed 00:27:18，serial canonical SHA `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K2 jobid `8796` 8-cell wall sp@8=**1.08×** + A3a/A3b：N=2 PASS bitwise（强证）/ N=4 N=8 dual-FAIL（trajectory bifurcation，§1.1.1 WARNING per design D5 NG3）|
| heihe_x4 | PASS @ server | PR-J cn03 Slurm jobid `8795` Elapsed 01:08:45，serial canonical SHA `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` ≡ B1b/B1-tag golden + CVODE 15-key identical + PR-K2 jobid `8797` 8-cell wall sp@8=**1.14×** + A3a/A3b：N=2 PASS bitwise / N=4 N=8 dual-FAIL（同 heihe，P7 final-fusion debt）|

Aggregate gate（PR-L #224 capstone + 本 PR-M #225 验收）：

- 3 anchor at NUM_OPENMP=1 全 PASS = 24/24（Mac RHS snapshot 12/12 + Mac fullrun 8/8 + server fullrun 4/4）+ Mac 16-cell scaling 16/16 A3a PASS（NG1 dev-only）= **40/40 PASS at NUM_OPENMP=1 / Mac all-N anchor**。
- SHUD 在 `openmp-baseline` 分支 commit `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（PR-D/E/F 5-step push workflow 严格遵守，3-pragma stack 完整）。
- `P1-update-omp-tag` annotated tag = `ff21c75c8e968d5e47ca53b015425360be9ac879`（指向 commit `003f58dc079116ef2161d2f96006228ef0e013d0`，含 SHUD pin `07c677f`）；`baseline/P1` 分支 protection `lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false`（D11 一次锁死 enforced）。
- M7 forcing trim：6-case trim PASS（Mac 4-case PR-A + server heihe/heihe_x4 PR-B）；kashigeer N/A deferred-upstream；Opt-IO 决策 = (a) 退回可选（PR-G #214，trimmed `t_forcing_io/t_total` heihe 1.90% / heihe_x4 0.19% << 5% 严格门）；7-case manifest `forcing_dir` schema 升级为 `{original_path, trimmed_path}` mapping + `check_manifest.py` union 校验。
- §1.1.1 verdict = **WARNING（P1 不阻塞）** per design D5 NG3 + master plan §6 P7 final-fusion debt：
  - wall/sp：heihe sp@8=1.08×（Amdahl-bound ~1.13× ✓ + "不独立验收" carve-out per master plan §1.1.1 + §5 Opt-IO 退回可选）；heihe_x4 sp@8=1.14×（P7 strict M=1.8× 退出门，作为 P1 起点报告）
  - A3a/A3b strict at N≥4：4/6 cells dual-FAIL with CVODE nst bifurcation（heihe nst 6773→6585/6684）；root cause hypothesis = B1b S3c owner-local gather tree-reduction-depth N>2 transition；P7 final-fusion deterministic-reduction debt logged（master plan §6 P7 scope；spec wording 升级走 PR-N 6.8b）

## Opt-IO 行证据

> 2026-06-22 由 PR-G #214（`openspec/changes/p1-update-omp` task 2.4）按 `profile-retest-m7` spec L40-L50 同步：trim 后 heihe `t_forcing_io / t_total = 1.90%`、heihe_x4 `= 0.19%`，均 << 50% 触发门、亦 << 5% 严格门。Opt-IO 从 master plan §5 L1533 "heihe 硬性前置" 改判 (a) **退回可选**；详 [`docs/profile_decision.md`](profile_decision.md) §"Opt-IO 硬性前置判断（M7 trim 后重测）"。

| Case | 单元格 | 证据 |
|---|---|---|
| heihe | PENDING (可选) | server cn03 Slurm jobid `8742`（Elapsed 00:07:29，NUM_OPENMP=1，binary `396ad9fb…`），3-run rivqdown SHA byte-identical `55abad28…` ≡ B0/B1a/B1b-tag golden；`benchmarks/heihe/profile_B0.target.trimmed.yaml` `t_forcing_io_pct_of_total = 1.90%`；判 (a) 退回可选 per PR-G #214 |
| heihe_x4 | PENDING (可选) | server cn03 Slurm jobid `8743`（Elapsed 01:09:06，NUM_OPENMP=1，binary `396ad9fb…`），3-run rivqdown SHA byte-identical `f90601ef…` ≡ B0/B1a/B1b-tag golden；`benchmarks/heihe_x4/profile_B0.target.trimmed.yaml` `t_forcing_io_pct_of_total = 0.193%`；判 (a) 退回可选 per PR-G #214 |
| keliya / xinanjiang_upstream / qinyijiang / qhh | PENDING (可选) | 非 IO 主导 case，未重测，沿 master plan §5 原 "B1b 锁定后任意时间执行" 可选定位；与本判 (a) 一致 |
| kashigeer | PENDING (可选) | deferred-upstream（forcing 缺口）；Opt-IO 评估与 forcing 数据补齐绑定，不在本 PR-G 范围 |

Aggregate Opt-IO 列 = **PENDING (可选)**（无 case 阻塞 P1+；本 change `p1-update-omp` 不消费 Opt-IO；P-strict 全部完成后再评估）。

## A0 验收 checklist

对应 `status-matrix` spec L47 + master plan §S0 A0 验收门。各项映射到 S0 PR 的交付物，当前状态：

| # | 项目                                                  | 状态   | 证据                                                                                              |
|---|-------------------------------------------------------|----------|-------------------------------------------------------------------------------------------------------|
| 1 | 7 manifest 完整（registry + INDEX）                   | PASS     | `benchmarks/INDEX.md` + 7 × `benchmarks/<case>/manifest.yaml`（#6 PR #22 + #28 改名 PR）；kashigeer 按修订后 spec 保留为 placeholder + DEFERRED.txt |
| 2 | 各非 deferred-upstream case 3 次 bitwise              | PASS     | 6 个 case PASS（keliya / xinanjiang_upstream / qinyijiang / qhh 本地 + heihe / heihe_x4 服务器 Slurm 8256）；kashigeer 按 S0-13 spec 修订排除 |
| 3 | 同上 case 的 cvode_stats 三次一致                     | PASS     | 同 6 个 case PASS（各自 B0_output 下 cvode_stats.txt）；kashigeer 排除                                                                                  |
| 4 | snapshot probe 三次一致                               | PASS     | 4 个 case × 3 = 12 个 snapshot_t*.bin 入库（`keliya` / `xinanjiang_upstream` / `qinyijiang` / `qhh`，#9 PR #24）；heihe + heihe_x4 仅服务器跑（按修订后 spec 不在 snapshot 范围内）；kashigeer 作为 deferred-upstream 排除 |
| 5 | tools/rhs_snapshot + tools/compare_snapshot 可独立调用 | PASS     | `tools/rhs_snapshot/` + `tools/compare_snapshot/` 干净编译 + 在 #9（PR #24 + CI #13）中被调用                                                       |
| 6 | CI 自动 pass/fail                                     | PASS     | `.github/workflows/serial-baseline.yml`（S0-9 / #13 PR #30）在 push + PR 上绿；skip-label 受尊重                                                       |
| 7 | profile_B0.yaml × 4 real + 3 deferred（local）+ .target.yaml × 6 real + 1 deferred | PASS     | 修订后 spec：4 个 local real（keliya/xinanjiang_upstream/qinyijiang/qhh）+ 3 个 deferred（heihe/heihe_x4/kashigeer）；6 个 target real + 1 个 deferred（kashigeer）（#14 PR #31 + #15 PR #32 + S0-13 修订） |
| 8 | docs/profile_platform.md 声明                         | PASS     | `docs/profile_platform.md`（#15 PR #32）— local + target + decision_consistency 三段齐全                                                              |
| 9 | docs/profile_decision.md 已签署                       | PASS     | `docs/profile_decision.md`（#15 PR #32 + S0-13 #17 签字）— DankerMu 已对外层 `a860eae5` + SHUD `78c37a1` 签字，授权日期 2026-06-17 |

**B0-tag-applied**: `true`
**B0-tag-date**: `2026-06-17`

### B0-tag 已打（2026-06-17）

9 项 A0 checklist 全部 PASS。`B0-tag` 已上 origin：

- Tag object SHA：`95ddc375ffa58115fd5c0a808dde80e9713b4c93`（annotated）
- 指向 commit：`884cfb13ba08ebae02dd64e371c4a19a536b4e26`（PR #35 squash-merge 到 `baseline/current`）
- SHUD submodule pin：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`
- 验证命令：`git rev-parse B0-tag` 返回 tag object SHA；`git rev-parse B0-tag^{}` 返回 commit；`git ls-remote --tags origin | grep B0-tag` 在 origin 上能看到。

## 阶段状态说明

- **B0** 行在 B0-tag 时刻冻结，成为 B1a 回归比对的参照。
- **B1a** 必须与 B0 同机同 case bitwise 一致。S0–S4 全部完成（PR-12 #156 capstone 2026-06-21）；本矩阵 B1a 行 Mac 本地 4-case + qhh 3 lake outputs 已 PASS；heihe / heihe_x4 直接服务器 Slurm 8537 / 8538 (cn08) bitwise PASS；aggregate = PASS。kashigeer 沿 B0 deferred-upstream 排除。
- **B1b** 必须与 B0/B1a 同机同 case bitwise 一致。S5 + S6b + S6c 全部完成（PR-16 #207 capstone + #189 B1b-tag + #190 PR PROMOTE 2026-06-22 + PR-18 #209 #205 post-B1b cleanup + PR-19 #210 #185 PI E2 sign-off + D9 fast-path triggered → `B1-tag`）；本矩阵 B1b 行 Mac 本地 4-case + 服务器 heihe / heihe_x4 (cn03) 全部 bitwise PASS；aggregate = PASS **UNCONDITIONAL ship**（#185 RESOLVED via E2 / #205 RESOLVED post-tag / #186 retroactively consistent / D9 TRIGGERED `B1-tag` 已发布 / C8 forward-compat UNUSED for this ship）。kashigeer 沿用 N/A。下游 P1+ 优先 use `B1-tag`；`B1a-tag` + `B1b-tag` 保留 immutable per D11 historical reference。
- **Opt-IO** 是 master plan §3.5 的 forcing I/O 并行化。落地时机由 `docs/profile_decision.md:bring-forward-IO` 评估，可能早于或晚于首个 OpenMP P1。M7 trim 后 (PR-G #214) heihe 1.90% / heihe_x4 0.19% << 5% 严格门，决策 (a) 退回可选；不阻塞 P1+。
- **P1** = master plan §3 "MD_update.cpp 三 owner loop 并行后的 first parallel candidate baseline"。Phase A/B/C/D 全部完成（PR-A #212 through PR-L #224 capstone + 本 PR-M #225 docs PROMOTE 2026-06-22；PR-N #226 openspec PROMOTE next）；本矩阵 P1 行 Mac 本地 4-case + 服务器 heihe / heihe_x4 (cn03) 全部 NUM_OPENMP=1 vs B1b/B1-tag bitwise PASS；aggregate = PASS（§1.1.1 = WARNING 不阻塞 per design D5 NG3 + master plan §6 P7 final-fusion debt logged）。kashigeer 沿用 N/A deferred-upstream。下游 P2 阶段从 `main` 分新分支；`baseline/P1` 仅作 frozen baseline 历史比对。
- **P2-P9** 各行按 master plan §3 各阶段填。

## 更新规则

- **CI 不自动 push**：serial-baseline.yml 的 `propose-matrix-update` 步骤会在 merge 的 PR 上评论一个 diff 建议，由 maintainer 在下一个常规 PR 中应用或直接 merge suggestion。
- **每个 PR 边界一行**：阶段 PR 落地时，其摘要引用所更新的矩阵行。跨阶段编辑罕见，需明确标注。
- **Aggregate 列**派生计算：`aggregate = PASS iff 所有 per-case 单元格 PASS-or-N/A`。CI proposer 自动填。
