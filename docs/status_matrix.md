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

> _Last touched: 2026-06-17 (S0-12 / #16 initial scaffold)_

## Matrix

| Stage     | keliya | xinanjiang_upstream | qinyijiang | kashigeer | qhh     | heihe         | heihe_x4      | aggregate |
|-----------|--------|---------------------|------------|-----------|---------|---------------|---------------|-----------|
| **B0**    | PASS   | PASS                | PASS       | BLOCKED   | PASS    | PASS @ server | PENDING       | BLOCKED   |
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
| kashigeer           | **BLOCKED**   | `benchmarks/kashigeer/B0_output/DEFERRED.txt` — upstream X76 forcing band missing on BOTH endpoints (#11 PR #26 + #12 PR #29 cross-checked) |
| qhh                 | PASS          | `benchmarks/qhh/B0_output/` 3-run PASS (4 .dat incl. 3 lake) (#11) + snapshot × 3 (#9) |
| heihe               | PASS @ server | `benchmarks/heihe/B0_output/` 3-run PASS (server cn08 via Slurm) (#12 PR #29) |
| heihe_x4            | **PENDING**   | No B0 archive yet. profile_B0.target.yaml exists (#15 PR #32) but B0 output set not archived. To bring to PASS: run `tools/archive_b0_output.sh heihe_x4 3` on server (estimated ~30-40 min × 3 + archive). Tracking: this row gates B0-tag (#17). |

### Aggregate B0 = BLOCKED

The aggregate B0 cell is BLOCKED rather than PASS because:
- 1 case is BLOCKED upstream (kashigeer — forcing data gap not solvable in this project)
- 1 case is PENDING (heihe_x4 — solvable but not yet executed)

Per `status-matrix` spec L42-43 ("Single case FAIL blocks aggregate"), BLOCKED is morally equivalent for the go/no-go check: aggregate cannot honestly be PASS while any cell is non-PASS/non-N/A. The aggregate moves to PASS when (a) heihe_x4 server B0 archive completes (→ PASS @ server) and (b) kashigeer is either reclassified as `N/A (server-only data deferred)` via a benchmark-registry spec amendment, or its upstream data gap is resolved.

## A0 Acceptance Checklist

Per `status-matrix` spec L47 + master plan §S0 A0 acceptance gate. Each item maps to one or more deliverables in the S0 PRs; current state:

| # | Item                                                  | Status   | Evidence                                                                                              |
|---|-------------------------------------------------------|----------|-------------------------------------------------------------------------------------------------------|
| 1 | 7 manifest 完整 (registry + INDEX)                    | PASS     | `benchmarks/INDEX.md` + 7 × `benchmarks/<case>/manifest.yaml` (#6 PR #22 + #28 PR rename)              |
| 2 | 各 case 3 次 bitwise                                  | PARTIAL  | 5 cases PASS (keliya / xinanjiang_upstream / qinyijiang / qhh local + heihe server); kashigeer BLOCKED; heihe_x4 PENDING |
| 3 | cvode_stats 三次一致                                  | PARTIAL  | Same 5 cases PASS (cvode_stats.txt in each B0_output); kashigeer BLOCKED; heihe_x4 PENDING            |
| 4 | snapshot probe 三次一致                               | PASS     | 5 cases × 3 snapshot_t*.bin in repo (#9 PR #24)                                                       |
| 5 | tools/rhs_snapshot + tools/compare_snapshot 可独立调用 | PASS     | `tools/rhs_snapshot/` + `tools/compare_snapshot/` build cleanly + invoked in #9 (PR #24 + CI #13)     |
| 6 | CI 自动 pass/fail                                     | PASS     | `.github/workflows/serial-baseline.yml` (S0-9 / #13 PR #30) green on push + PR; skip-label honored    |
| 7 | profile_B0.yaml × 5 实跑 + .target.yaml × 7           | PARTIAL  | 4 local real + 3 deferred (#14 PR #31) ≠ spec literal "5 + 2"; 6 target real + 1 deferred (#15 PR #32). Documented spec drift for kashigeer. |
| 8 | docs/profile_platform.md 声明                         | PASS     | `docs/profile_platform.md` (#15 PR #32) — local + target + decision_consistency 三段齐                |
| 9 | docs/profile_decision.md 已签署                       | PENDING  | `docs/profile_decision.md` (#15 PR #32) — 决策 + Amdahl + P8-precond 全填，signer = `<PLACEHOLDER>`，待 B0-tag (#17) 时签 |

**B0-tag-applied**: `false`
**B0-tag-date**: `(pending)`

### Blockers for B0-tag

Per spec L57 ("Any A0 item FAIL blocks tag"), A0 cannot be all-PASS until:

1. **heihe_x4 server B0 archive** (items 2, 3) — run `tools/archive_b0_output.sh heihe_x4 3` on server Slurm; commit B0 archive files + bump aggregate cell. Estimated ~2h wall time. Responsible PR: TBD (likely a sub-PR of #17 or its own follow-up).
2. **kashigeer reclassification** (items 2, 3, 7) — either:
   - Resolve upstream forcing gap (out of project scope; involves NWM dataset re-curation), OR
   - Reclassify in `benchmarks/INDEX.md` as `deferred-upstream` endpoint and update spec literal in `status-matrix` and `rhs-profile-gate` spec to count 6-case A0 rather than 7. Responsible PR: TBD (spec amendment, possibly bundled with #17).
3. **profile_decision.md signature** (item 9) — project owner signs the doc (replacing `<PLACEHOLDER>` with name/handle + commit hash). Per spec L52, the signed doc is required for B0-tag. Responsible: project owner manual action at #17 merge time.

## Stage status guidance

- **B0** rows freeze at B0-tag and become reference for B1a regression checks.
- **B1a** SHALL be bitwise to B0 (single-machine, same case). The matrix will populate during S1 stage.
- **Opt-IO** is the master plan §3.5 forcing I/O parallelization. May land before or after first OpenMP P1, depending on what `docs/profile_decision.md:bring-forward-IO` evaluates to.
- **P1-P9** populate per master plan §3 stages.

## Update protocol

- **No CI auto-push**: serial-baseline.yml has a `propose-matrix-update` step that comments on the merging PR with a diff suggestion. Maintainer applies via the next regular PR or merges the suggestion verbatim.
- **One row per PR boundary**: when a stage PR lands, its summary references the matrix row it updates. Cross-stage edits are rare and should be flagged.
- **Aggregate column** is derived: `aggregate = PASS iff all per-case cells are PASS-or-N/A`. CI proposer fills it automatically.
