# SHUD-OpenMP — Stage × Benchmark Status Matrix

Authoritative state source for stage Go/No-Go decisions per
`openspec/changes/s0-baseline-lock/specs/status-matrix/spec.md`. Rows are
stages from master plan §3; columns are the 7 registered benchmarks
(`benchmarks/INDEX.md`) plus an `aggregate` column. Cell values:

- **PASS** — stage criteria verified for this case (link to evidence)
- **FAIL** — verified failure (blocks aggregate)
- **BLOCKED** — cannot evaluate due to upstream/external blocker (e.g. missing data)
- **PENDING** — not yet attempted; future stage
- **N/A** — case is structurally excluded from this stage

Updates land via PR; CI proposes diffs via PR comments (per
`status-matrix` spec L19, no auto-push). The matrix file is the **single
source of truth**; per-stage docs and PR summaries reference it, not the
other way around.

> _Last touched: 2026-06-17 (S0-13 / #17: kashigeer reclassified deferred-upstream via spec amendment + signature signed; heihe_x4 archive in-progress on server Slurm job 8256)_

## Matrix

| Stage     | keliya | xinanjiang_upstream | qinyijiang | kashigeer            | qhh     | heihe         | heihe_x4         | aggregate |
|-----------|--------|---------------------|------------|----------------------|---------|---------------|------------------|-----------|
| **B0**    | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PENDING (archiving) | PENDING   |
| **B1a**   | PENDING| PENDING             | PENDING    | PENDING   | PENDING | PENDING       | PENDING       | PENDING   |
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

### B0 row evidence

| Case                | Cell          | Evidence                                                             |
|---------------------|---------------|----------------------------------------------------------------------|
| keliya              | PASS          | `benchmarks/keliya/B0_output/` 3-run repeatability PASS (#11 PR #26) + snapshot_t*.bin × 3 (#9 PR #24) |
| xinanjiang_upstream | PASS          | `benchmarks/xinanjiang_upstream/B0_output/` 3-run PASS (#11 PR #26) + snapshot × 3 (#9) |
| qinyijiang          | PASS          | `benchmarks/qinyijiang/B0_output/` 3-run PASS (#11) + snapshot × 3 (#9) |
| kashigeer           | **N/A (deferred-upstream)** | `benchmarks/kashigeer/B0_output/DEFERRED.txt` — upstream X76 forcing band missing on BOTH endpoints (#11 PR #26 + #12 PR #29 cross-checked). **S0-13 spec amendment** reclassifies kashigeer endpoint as `deferred-upstream` in `benchmarks/INDEX.md`; `status-matrix` + `rhs-profile-gate` specs amended to exclude deferred-upstream cells from A0 bitwise/cvode_stats/snapshot scenarios. |
| qhh                 | PASS          | `benchmarks/qhh/B0_output/` 3-run PASS (4 .dat incl. 3 lake) (#11) + snapshot × 3 (#9) |
| heihe               | PASS @ server | `benchmarks/heihe/B0_output/` 3-run PASS (server cn08 via Slurm) (#12 PR #29) |
| heihe_x4            | **PENDING (archiving)** | Slurm job 8256 on server cn21 in progress: `tools/archive_b0_output.sh heihe_x4 3` (90-day window, ~1.2h wall × 3 runs + archive). On completion: cell → PASS @ server; aggregate → PASS. |

### Aggregate B0 = PENDING (will → PASS on heihe_x4 archive completion)

After the S0-13 spec amendment, only one case is non-PASS:
- `kashigeer` = N/A (deferred-upstream, no longer blocking per amended `status-matrix` spec Scenario "Deferred-upstream cell is N/A, not blocking")
- `heihe_x4` = PENDING (archiving on Slurm 8256; expected → PASS @ server)

Per amended `status-matrix` spec ("aggregate = PASS iff all non-N/A cells are PASS"), the aggregate flips to PASS the moment heihe_x4 archive lands.

## A0 Acceptance Checklist

Per `status-matrix` spec L47 + master plan §S0 A0 acceptance gate. Each item maps to one or more deliverables in the S0 PRs; current state:

| # | Item                                                  | Status   | Evidence                                                                                              |
|---|-------------------------------------------------------|----------|-------------------------------------------------------------------------------------------------------|
| 1 | 7 manifest 完整 (registry + INDEX)                    | PASS     | `benchmarks/INDEX.md` + 7 × `benchmarks/<case>/manifest.yaml` (#6 PR #22 + #28 PR rename); kashigeer kept as placeholder + DEFERRED.txt per amended spec |
| 2 | 各非 deferred-upstream case 3 次 bitwise              | PARTIAL  | 5 cases PASS (keliya / xinanjiang_upstream / qinyijiang / qhh local + heihe server); heihe_x4 PENDING (Slurm 8256 in progress); kashigeer excluded per S0-13 spec amendment |
| 3 | 同上 case 的 cvode_stats 三次一致                     | PARTIAL  | Same 5 cases PASS (cvode_stats.txt in each B0_output); heihe_x4 PENDING; kashigeer excluded                                                              |
| 4 | snapshot probe 三次一致                               | PASS     | 5 cases × 3 snapshot_t*.bin in repo (#9 PR #24); heihe* + kashigeer are server-only / deferred-upstream and out of snapshot scope                       |
| 5 | tools/rhs_snapshot + tools/compare_snapshot 可独立调用 | PASS     | `tools/rhs_snapshot/` + `tools/compare_snapshot/` build cleanly + invoked in #9 (PR #24 + CI #13)                                                       |
| 6 | CI 自动 pass/fail                                     | PASS     | `.github/workflows/serial-baseline.yml` (S0-9 / #13 PR #30) green on push + PR; skip-label honored                                                       |
| 7 | profile_B0.yaml × 4 real + 3 deferred (local) + .target.yaml × 6 real + 1 deferred | PASS     | Amended spec: 4 local real (keliya/xinanjiang_upstream/qinyijiang/qhh) + 3 deferred (heihe/heihe_x4/kashigeer); 6 target real + 1 deferred (kashigeer) (#14 PR #31 + #15 PR #32 + S0-13 amendment) |
| 8 | docs/profile_platform.md 声明                         | PASS     | `docs/profile_platform.md` (#15 PR #32) — local + target + decision_consistency 三段齐                                                                  |
| 9 | docs/profile_decision.md 已签署                       | PASS     | `docs/profile_decision.md` (#15 PR #32 + S0-13 #17 signature) — DankerMu signed against outer `a860eae5` + SHUD `78c37a1` via delegated grant 2026-06-17 |

**B0-tag-applied**: `false` _(awaits heihe_x4 archive completion + final tag push)_
**B0-tag-date**: `(pending)`

### Remaining blocker for B0-tag

After S0-13 spec amendment + signature, only one A0 item is non-PASS:

1. **heihe_x4 server B0 archive** (items 2, 3) — Slurm job 8256 running on server cn21, `tools/archive_b0_output.sh heihe_x4 3` (90-day window). On completion: B0_output/ rsync back, commit, A0 items 2 + 3 flip to PASS, aggregate B0 → PASS, then `git tag -a B0-tag` push on `baseline/current` HEAD.

## Stage status guidance

- **B0** rows freeze at B0-tag and become reference for B1a regression checks.
- **B1a** SHALL be bitwise to B0 (single-machine, same case). The matrix will populate during S1 stage.
- **Opt-IO** is the master plan §3.5 forcing I/O parallelization. May land before or after first OpenMP P1, depending on what `docs/profile_decision.md:bring-forward-IO` evaluates to.
- **P1-P9** populate per master plan §3 stages.

## Update protocol

- **No CI auto-push**: serial-baseline.yml has a `propose-matrix-update` step that comments on the merging PR with a diff suggestion. Maintainer applies via the next regular PR or merges the suggestion verbatim.
- **One row per PR boundary**: when a stage PR lands, its summary references the matrix row it updates. Cross-stage edits are rare and should be flagged.
- **Aggregate column** is derived: `aggregate = PASS iff all per-case cells are PASS-or-N/A`. CI proposer fills it automatically.
