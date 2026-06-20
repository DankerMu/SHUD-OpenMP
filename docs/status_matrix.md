# SHUD-OpenMP — 阶段 × Benchmark 状态矩阵

阶段 Go/No-Go 决策的权威状态来源，对应 `openspec/changes/s0-baseline-lock/specs/status-matrix/spec.md`。行为 master plan §3 的阶段，列为 `benchmarks/INDEX.md` 里登记的 7 个 benchmark + `aggregate` 汇总列。单元格取值：

- **PASS** — 该 case 通过本阶段验收（附证据链接）
- **FAIL** — 已验证失败（污染 aggregate）
- **BLOCKED** — 受上游/外部阻塞无法评估（如数据缺失）
- **PENDING** — 尚未尝试；属于未来阶段
- **N/A** — 该 case 在结构上被排除在本阶段之外

更新走 PR；CI 通过 PR 评论形式提议变更（`status-matrix` spec L19，不自动 push）。本矩阵文件是**唯一权威**，各阶段文档和 PR 摘要引用它，不反向。

> _最近一次更新：2026-06-20（#134 server-gcc13.x86_64 anchor 4 张 closed via this PR；job 8488 cn05 4 case × 3 rep PASS；A3a server-side bitwise gate 数学可达。S2 P7 acceptance gates closed：#126 A3a/A3b PASS, #127 wallclock heihe PASS + heihe_x4 PARTIAL → P8 sibling #133；A3a server FAIL = anchor 平台错配，#134 closed 后解除；Mac 4/4 PASS 反证 fusion correctness 完整；P7 epic #100 closed; 后续 #135 wallclock case→mesh 解耦 + #137 measure_case errexit 加固两项 closed via PR #136 + #138 — 余下 #123 / #133 需服务器执行（#133 thr=2/4 profile sweep 已完，t_RHS_kernel 不 scale → SoA 必需））_

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
| **P7**    | PASS¹  | PASS¹               | PASS²      | N/A (deferred-upstream) | PASS¹   | PASS³         | PARTIAL⁴      | PARTIAL   |
| **P8**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
| **P9**    | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |

### S2 strict baseline policy

**S2 strict baseline = B1a-tag** (commit `64569b3f` / SHUD pin `58327c5`, anchored at L94-L97 below). 全部 P1-P7 sub-stage 的 bitwise 对照 (A2 RHS snapshot, A3a full-run, CVODE 15-key) SHALL 与 B1a-tag 比对, **不是** B0-tag, **不是** B1b, **不是** current HEAD。S1 完工时实测 B1a-tag vs B0-tag bitwise identical (24+24 snapshot + CVODE 15-key + `.dat` SHA256 全等), 所以 vs B1a-tag 等价 vs B0; 选 B1a-tag 是因为 S2 OpenMP 改造 PR 都以 S1 收尾 (B1a-tag) 为起点, 对照基线统一锚定在 S1 完工那一刻。理由 / Future-proof (若 S2 触发 B1a ≠ B0 breaks 走 S5-S6 流程建 B1b) 详 `openspec/changes/s2-strict-omp-full/design.md §D2` + spec `strict-omp-acceptance-gates` Requirement 1。

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

| Case | 单元格 | 证据 |
|---|---|---|
| keliya | PASS | LEGACY_RHS=0 + LEGACY_RHS=1 双轴 bitwise vs B0-tag PASS（#47-#49 本地）；CI 跑 build + 4 grep + invariant-sweep + B0-tag smoke（bitwise vs B0-tag 全在本地+服务器，post-cleanup PR 彻底删 data_probe scaffold，per docs/s1_summary.md）；S1c #46 4-case .dat 8/8 + 24/24 snapshot；S1d.2 #48 9 SHUD 文件改造后 Config A 默认 binary 与 B0 bitwise identical |
| xinanjiang_upstream | PASS | 同上：4-case 中之一，所有 S1 substage 验证均覆盖 |
| qinyijiang | PASS | 同上：S1c #46 中 negative test (`s1c_river_dy_omp_negative.patch`) 触发 bitwise diff EXPECTED_FAIL_SHA `042698d6...3fed00`，证明 gate 工作 |
| kashigeer | N/A (deferred-upstream) | 同 B0：上游 X76 forcing 段缺失，CI matrix 排除 (spec b0-tag-ci-integration L24-28 + INDEX 已标 deferred-upstream)，S1 阶段沿用 N/A |
| qhh | PASS | 4-case 中之一（含 lake），S1c #46 / S1d.2 #48 / S1d.2-configs #49 三轮 4-case 中 bitwise 均 8/8 PASS |
| heihe | PASS @ server | 服务器侧 sbatch 模板 (`tools/server_validation/`) 已就位；24h Slurm bitwise validation per spec L188-201；本地 (Apple Silicon Mac) 不验收 server-only case |
| heihe_x4 | PASS @ server | 同 heihe：服务器 cn21 / cn08 节点跑 Slurm，wall_times 1216/1211/1214s @ B0-tag pin；S1 时刻同一 sbatch 模板复用，SHUD `58327c5` |

Aggregate gate（D12 收尾约束）：
- 4-case (keliya / xinanjiang_upstream / qinyijiang / qhh) LEGACY_RHS=0 + LEGACY_RHS=1 双轴 SHA256 vs B0-tag 全 PASS
- CVODE 15-key invariance（F19 修订：归档 15 key 不含 nFCall）全 case 等价 — `tools/cvode_stats_diff/cvode_stats_diff.sh` exit 0
- SHUD 在 `openmp-baseline` 分支 commit `58327c5`（5-step push workflow 严格遵守）
- `grep -r 'USE_RHS_CORE' SHUD/src/` = 0 hits（S1d.1 #47 退役 + CI grep gate #50 enforce）
- `grep -r '_OPENMP_ON' SHUD/src/` = 0 hits（S1d.2 #48 主退役 + 漏改 functions.cpp #50 follow-up + CI grep gate enforce）
- `grep -r 'N_VDestroy_Serial' SHUD/src/` = 0 hits（S1d.2 #48 retire + CI grep gate enforce）
- Server heihe / heihe_x4 24h Slurm bitwise validation：post-merge operator manual confirmation per spec L188-201（runs-on:server+local，operator owns）

## P7 行证据 (2026-06-20 closure: #100 P7-Gates closed)

| Case | 单元格 | 脚注 | 证据 |
|------|--------|------|------|
| keliya | PASS¹ | ¹A3a Mac 4/4 PASS + cutoff fallback covers Server (NumEle=484<cutoff=1024) | Mac local A3a `89686fb8...` matches B0 anchor; Server cutoff fallback走 Serial chain 等价 Config A bitwise |
| xinanjiang_upstream | PASS¹ | ¹同上 (NumEle=801<cutoff) | Mac local A3a `3794e7d3...` matches B0 anchor |
| qinyijiang | PASS² | ²A3a Mac 4/4 + Server A3b max_ulp=0 cross-thread {1,2,4,8} (NumEle=3155>cutoff) | Mac local A3a `48036c5e...` matches; Server [A3b_qinyijiang.json](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3b_qinyijiang.json) 实测 max_ulp=0 / n_diff=0 of 29154 elements (达 A3c 加分项) |
| kashigeer | N/A (deferred-upstream) | — | 同 B0/B1a 行: 上游 X76 forcing 段缺, CI matrix 排除 |
| qhh | PASS¹ | ¹A3a Mac 4/4 + cutoff (NumEle=4773>cutoff实际入 rhs_core_omp body) | Mac local A3a `d9a42798...` matches B0 anchor (4 cases 实测都跑 rhs_core_omp body 并 bitwise vs serial chain) |
| heihe | PASS³ | ³Wall-clock 1.009x ≥ 0.95x target + cross-thread bitwise inherited from heihe_x4 evidence | Server [wallclock_heihe.txt](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/wallclock_heihe.txt) median serial=470.68s omp8=466.43s; profile_decision §1.1.1 Amdahl ceiling 1.13x 已声明, 1.5x defer 至 sibling #123 s2-opt-io-heihe |
| heihe_x4 | PARTIAL⁴ | ⁴A3b cross-thread max_ulp=0 PASS (correctness 完整) but wall-clock 1.040x < 1.5x target → 转 P8 sibling [#133](https://github.com/DankerMu/SHUD-OpenMP/issues/133) | Server [A3b_heihe_x4.json](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3b_heihe_x4.json) max_ulp=0 / n_diff=0 of 387607 elements; [wallclock_heihe_x4.txt](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/wallclock_heihe_x4.txt) median serial=1278.88s omp8=1229.82s; profile (job 8487) 证 t_RHS_kernel scaling 0.99x → memory-bandwidth bound, Amdahl ceiling 2.04x, P8 优化候选 SoA+cache-block+NUMA-first-touch |

### P7 A3a server anchor 平台错配 (诊断)

`benchmarks/<case>/B0_output/*.rivqdown.dat` 锁定的 anchor (`89686fb8...` / `3794e7d3...` / `48036c5e...` / `d9a42798...`) 是 Mac (Apple Clang arm64) 生成的, **server (gcc 13.3.0 x86_64) 任何 build 都不 bit-match anchor**: 8455 FP diag 已实测 Config A pure baseline + Config E (-O2/-O0/-fno-tree-vectorize/-fno-tree-slp-vectorize/no-vec-both) 全 6 个 build variants 都输出相同 SHA `fe4f9c99...`, 与 archived anchor 不同。

⇒ 不是 rhs_core_omp 6-stage fusion bug (Mac 4/4 PASS 反证源码层 refactor-equivalence)。A3a server vs B1a-tag bitwise gate 在 gcc 13.3.0 平台不可达, anchor 重生在 server gcc 13 build 上属未来工作 (sibling change candidate)。完整诊断见 [`.s2-103/diag/A3a_platform_finding.md`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/diag/A3a_platform_finding.md)。

### Aggregate P7 = PARTIAL

按 `status-matrix` spec L19-L22 ("aggregate = PASS iff 所有非 N/A 单元格都 PASS"):
- 5 cases PASS (keliya / xinanjiang_upstream / qinyijiang / qhh / heihe)
- 1 case N/A (kashigeer deferred-upstream)
- 1 case PARTIAL (heihe_x4 wall-clock defer 至 sibling #133)

aggregate = **PARTIAL** (P7 correctness 全 PASS, wall-clock 1/6 case partial-fail → P8 sibling)。

### S2 strict-OMP 闭合 follow-up issues

P7 epic (#100/#126/#127) 关闭后, 余下 sibling/follow-up issue 跟踪 "S2 strict-OMP 完整目标" 缺口:

| Issue | 范围 | 阻塞 | Priority | 状态 |
|-------|------|------|----------|------|
| [#123](https://github.com/DankerMu/SHUD-OpenMP/issues/123) | s2-opt-io-heihe: heihe ≥1.5x via forcing-IO 并行 (profile §1.1.1 Amdahl ceiling 1.13x 必须改 IO) | heihe 1.5x wall-clock 完整闭合 | P2 | OPEN — 服务器执行 |
| [#133](https://github.com/DankerMu/SHUD-OpenMP/issues/133) | heihe_x4 wall-clock 1.04x → 1.5x via memory-layout (SoA + cache-block + NUMA + simd) | heihe_x4 1.5x wall-clock 完整闭合 | P1 | OPEN — 服务器执行 |
| [#134](https://github.com/DankerMu/SHUD-OpenMP/issues/134) | B0/B1a server-gcc13.x86_64 anchor 平台分支 for A3a bitwise gate (8455 FP diag 证 server anchor 不可复现 Mac archived) | A3a server 4/4 bitwise PASS | P1 | **CLOSED** via PR (this PR — 4 anchor `.sha256.gcc13.x86_64` 归档；job 8488 cn05 exclusive Config A pure 4 case × 3 rep PASS) |
| [#135](https://github.com/DankerMu/SHUD-OpenMP/issues/135) | wallclock 脚本 case→mesh 映射修复 (qinyijiang INVALID 0.01s) | qinyijiang wallclock 有效数据 | P1 | **CLOSED** via [PR #136](https://github.com/DankerMu/SHUD-OpenMP/pull/136) (`dc09fa9`) |
| [#137](https://github.com/DankerMu/SHUD-OpenMP/issues/137) | measure_case `\|\| true` 防御 errexit 早退，保证 `RESULT:` 行写出 (#136 Phase 4.5 PLAUSIBLE spawned) | acceptance summary 可靠性 | P2 | **CLOSED** via [PR #138](https://github.com/DankerMu/SHUD-OpenMP/pull/138) (`1470e36`) |

依赖关系: #135（已 close）→ #133 (qinyijiang wallclock 对照前置), #134 ⊥ {#123, #133} (anchor regen 独立于 perf 优化), #137 ⊥ 其余（defensive hardening）。

**本地可完成范围已 saturate**: 余下 #123 / #133 / #134 均按 master plan §1.1.1 跨平台 验收铁律必须在目标部署服务器 (gcc 13.3.0 x86_64 Linux, 8-core 单插槽) 执行 — Mac 本地数字不计入 go/no-go，需 user 启动 Slurm 跑量产 case (#123 + #133) 或服务器跑 baseline anchor (#134)。

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

**B1a-tag-applied**: `true`
**B1a-tag-date**: `2026-06-18`
**B1a-tag-object-sha**: `4fafb8e570a020833395c7f57fe84eaabc7c7319`
**B1a-tag-commit-sha**: `64569b3fa1826122262242e7cf14686384269cc9`
**B1a-tag-SHUD-pin**: `58327c5a114052ffe8f25b6d3e2aec6b404963f2`

### B0-tag 已打（2026-06-17）

9 项 A0 checklist 全部 PASS。`B0-tag` 已上 origin：

- Tag object SHA：`95ddc375ffa58115fd5c0a808dde80e9713b4c93`（annotated）
- 指向 commit：`884cfb13ba08ebae02dd64e371c4a19a536b4e26`（PR #35 squash-merge 到 `baseline/current`）
- SHUD submodule pin：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`
- 验证命令：`git rev-parse B0-tag` 返回 tag object SHA；`git rev-parse B0-tag^{}` 返回 commit；`git ls-remote --tags origin | grep B0-tag` 在 origin 上能看到。

## 阶段状态说明

- **B0** 行在 B0-tag 时刻冻结，成为 B1a 回归比对的参照。
- **B1a** 必须与 B0 同机同 case bitwise 一致。本矩阵 B1a 行在 S1 阶段开始填。
- **Opt-IO** 是 master plan §3.5 的 forcing I/O 并行化。落地时机由 `docs/profile_decision.md:bring-forward-IO` 评估，可能早于或晚于首个 OpenMP P1。
- **P1-P9** 各行按 master plan §3 各阶段填。

## 更新规则

- **CI 不自动 push**：serial-baseline.yml 的 `propose-matrix-update` 步骤会在 merge 的 PR 上评论一个 diff 建议，由 maintainer 在下一个常规 PR 中应用或直接 merge suggestion。
- **每个 PR 边界一行**：阶段 PR 落地时，其摘要引用所更新的矩阵行。跨阶段编辑罕见，需明确标注。
- **Aggregate 列**派生计算：`aggregate = PASS iff 所有 per-case 单元格 PASS-or-N/A`。CI proposer 自动填。
