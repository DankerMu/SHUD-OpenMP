# SHUD-OpenMP — 阶段 × Benchmark 状态矩阵

阶段 Go/No-Go 决策的权威状态来源，对应 `openspec/changes/s0-baseline-lock/specs/status-matrix/spec.md`。行为 master plan §3 的阶段，列为 `benchmarks/INDEX.md` 里登记的 7 个 benchmark + `aggregate` 汇总列。单元格取值：

- **PASS** — 该 case 通过本阶段验收（附证据链接）
- **FAIL** — 已验证失败（污染 aggregate）
- **BLOCKED** — 受上游/外部阻塞无法评估（如数据缺失）
- **PENDING** — 尚未尝试；属于未来阶段
- **N/A** — 该 case 在结构上被排除在本阶段之外

更新走 PR；CI 通过 PR 评论形式提议变更（`status-matrix` spec L19，不自动 push）。本矩阵文件是**唯一权威**，各阶段文档和 PR 摘要引用它，不反向。

> _最近一次更新：2026-06-21（B1a capstone / PR-12 #156：S0–S4 全部完成；4-case + qhh 3 lake outputs Mac 本地 bitwise vs B0-tag PASS；7 grep gate 0 hits；heihe / heihe_x4 直接服务器 Slurm 8537 / 8538 bitwise PASS @ cn08；aggregate = PASS；follow-up issue #171 已 close）_

## 矩阵

| 阶段       | keliya | xinanjiang_upstream | qinyijiang | kashigeer            | qhh     | heihe         | heihe_x4      | aggregate |
|-----------|--------|---------------------|------------|----------------------|---------|---------------|---------------|-----------|
| **B0**    | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
| **B1a**   | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
| **B1b**   | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **Opt-IO**| PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P1**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P2a**   | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
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

> **2026-06-20 capstone（PR-12 #156）**：B1a 整体 = master plan §3 "S0–S4 完成"。S0 + S1 + S2 + S3 + S4 全部完成（PR-1 #144 through PR-11 #155 已 merge 进 `baseline/B1a`；详见 [`docs/b1a_summary.md`](b1a_summary.md) 时间线）。下表 evidence 来自 PR-12 capstone 时刻（SHUD `0b3998d`），4-case Mac 本地 + qhh 3 lake outputs bitwise vs B0-tag 全 PASS；7 grep gate 0 hits 全部 enforce；heihe / heihe_x4 服务器 Slurm 验证 = **PENDING**，跟进 issue 捕获 Slurm SHA evidence 后单元格 flip 到 `PASS @ server`，aggregate 同步 flip 到 `PASS`（PR-12 本身在 Mac 端无 Slurm 调用通道）。

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
- **Opt-IO** 是 master plan §3.5 的 forcing I/O 并行化。落地时机由 `docs/profile_decision.md:bring-forward-IO` 评估，可能早于或晚于首个 OpenMP P1。
- **P1-P9** 各行按 master plan §3 各阶段填。

## 更新规则

- **CI 不自动 push**：serial-baseline.yml 的 `propose-matrix-update` 步骤会在 merge 的 PR 上评论一个 diff 建议，由 maintainer 在下一个常规 PR 中应用或直接 merge suggestion。
- **每个 PR 边界一行**：阶段 PR 落地时，其摘要引用所更新的矩阵行。跨阶段编辑罕见，需明确标注。
- **Aggregate 列**派生计算：`aggregate = PASS iff 所有 per-case 单元格 PASS-or-N/A`。CI proposer 自动填。
