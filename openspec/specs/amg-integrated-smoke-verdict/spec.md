# amg-integrated-smoke-verdict Specification

## Purpose
TBD - created by archiving change p8tune-g0-instrumented-amg-smoke. Update Purpose after archive.
## Requirements
### Requirement: G0 verdict evaluates six PASS/FAIL gates

The G0 verdict aggregator MUST evaluate exactly six gates, in this order, and emit one PASS/FAIL flag per gate plus a final `g0_verdict_branch ∈ {GO-G0, NO-GO-G0}` decision. Naming and semantics MUST match SHUD_openMP_master_plan.md §P8-tune.G0 + ADR-0007 §Forward action Amendment 2026-06-29. The six gates are mapped 1:1 onto design.md Goals 1-6.

The six gates are:
1. `G0-1 default-compat`: PASS iff `SHUD_LINSOL` unset and `SHUD_LINSOL=spgmr` both produce bit-identical output bytes vs the pre-G0 SPGMR **same-platform** baseline reference run on the `keliya` 90-day SHORT cell. Per-platform anchors split (bit-identical SHA256 is not guaranteed across macOS clang/libm vs Linux gcc/glibc): the Mac-platform baseline archive is produced by PR-0 task 4.0 at `.review-evidence/g0-spgmr-baseline-90day-keliya/mac/` (SHA256 manifest + `Output/*.dat`); the server-platform baseline archive is produced by PR-A task 6.7 at `.review-evidence/g0-spgmr-baseline-90day-keliya/server/` (same schema). G0-1 evaluation compares the current run's `Output/*.dat` SHA256 set against the **same-platform** archive (mac local run ↔ `mac/`, server Slurm cn-node run ↔ `server/`). Cross-platform bit-identity is intentionally NOT a G0-1 requirement; cross-platform equivalence at G0 is asserted via the separate REQ "G0 verdict byte-identical anchor contract" on the verdict-block level.
2. `G0-2 build`: PASS iff `make shud` and `make shud_omp` both succeed on macOS (brew Hypre 3.1.x), Ubuntu CI runner (apt-installed Hypre, version pinned by PR-0 task 1.6), and server (`/scratch/frd_muziyao/local/hypre-3.1.0/`). Server Hypre version is verified uniformly as 3.1.0 across all CPU partition cn-nodes; legacy `HYPRE_RELEASE_NUMBER < 30000` branch in wrapper Initialize remains as defensive code for the CI runner case only.
3. `G0-3 telemetry-real`: PASS iff at least one cell's stdout contains `MARKER:AMG_TELEMETRY_REAL`, AND the cell_summary `cycle_complexity` value originates from a Hypre native API call (Hypre 3.1.0 substitute: `HYPRE_BoomerAMGGetCumNnzAP(amg, &nnz_AP)` divided by the IJMatrix nnz_A measurement; this is a STATIC hierarchy-size ratio, not a per-cycle work ratio as in the original spec draft) — NOT from the `2 × operator_complexity` hardcoded estimate used in P8-tune.F. The `hypre_iters` ring-buffer field comes from `HYPRE_BoomerAMGGetNumIterations` (Hypre 3.1.0 substitute for the unavailable `GetCycleNumIterations`). `operator_complexity` itself MUST come from `HYPRE_BoomerAMGGetOperatorComplexity(amg, &op_complexity)` (Hypre native), not a derived estimate. Downstream PR-B aggregator MUST treat the renamed `cycle_complexity` metric as a hierarchy-size attestation; the G0 amendment of this metric's semantics is documented here and propagated to ADR-0007 §Forward action.
4. `G0-4 integrated-completes`: PASS iff all four of `{keliya, xinanjiang_upstream, heihe_x4, heihe_x16}` SHORT 90-day cells under `SHUD_LINSOL=amg` complete without crash (exit code 0, no `MARKER:AMG_SETUP_DIVERGE_DETECTED` / `MARKER:AMG_SOLVE_DIVERGE_DETECTED` / `MARKER:AMG_OOM_DETECTED` / `MARKER:AMG_WALL_OVERFLOW_DETECTED` in stdout).
5. `G0-5 wall-signal`: PASS iff at least one of `{heihe_x4, heihe_x16}` produces `amg_wall_per_step[case] < spgmr_wall_per_step[case]` where `spgmr_wall_per_step[heihe_x4] = SPGMR_PER_STEP_HEIHE_X4_S` and `spgmr_wall_per_step[heihe_x16] = SPGMR_PER_STEP_HEIHE_X16_S` are case-specific baselines measured and pinned by PR-0 in `tools/p8tune.G0/spgmr_baseline_walls_g0.h` (NOT the 60-cell `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` constant, which is preserved as historical context only). Smaller cells `{keliya, xinanjiang_upstream}` are exempt from G0-5 (the gate evaluates only the two production-target large cells).
6. `G0-6 solver-stats-documented`: PASS iff for every cell in `{keliya, xinanjiang_upstream, heihe_x4, heihe_x16}` whose `verdict_class=AMG_OK`, all six CVODE solver-stat fields (`cvode_nfe`, `cvode_nli`, `cvode_nfeLS`, `cvode_ncfn`, `cvode_ncfl`, `cvode_netf`) in that cell's cell_summary KV block are non-NA. Cells with non-AMG_OK verdict are exempt (NA values are acceptable when SIGTERM fires before measurement). This gate ensures the BDF/Newton/SPGMR interaction with AMG-as-inner-solver is documented per cell.

`g0_verdict_branch = GO-G0` iff all six gates PASS; otherwise `g0_verdict_branch = NO-GO-G0`.

#### Scenario: All six gates PASS produces GO-G0

- **WHEN** aggregator runs against evidence in which all of G0-1 through G0-6 evaluate PASS
- **THEN** aggregator emits `g0_verdict_branch=GO-G0` AND a line `MARKER:G0_GO_DETECTED` to stdout

#### Scenario: Any gate FAIL produces NO-GO-G0

- **WHEN** aggregator runs against evidence in which any one of G0-1 through G0-6 evaluates FAIL
- **THEN** aggregator emits `g0_verdict_branch=NO-GO-G0`, AND `MARKER:G0_NO_GO_DETECTED gate=<gate_id>` for each failing gate

#### Scenario: Aggregator refuses to declare G0-3 PASS without MARKER:AMG_TELEMETRY_REAL

- **WHEN** any cell's evidence claims to satisfy G0-3 telemetry-real but no `MARKER:AMG_TELEMETRY_REAL` line is present in the corresponding stdout
- **THEN** aggregator MUST evaluate G0-3 as FAIL with reason `AMG_TELEMETRY_MARKER_ABSENT`, even if the `cycle_complexity` numeric value looks plausible

#### Scenario: G0-6 solver-stats FAIL on any AMG_OK cell with NA counter

- **WHEN** any cell in `{keliya, xinanjiang_upstream, heihe_x4, heihe_x16}` reports `verdict_class=AMG_OK` AND any of its six CVODE solver-stat fields is the literal token `NA`
- **THEN** aggregator MUST evaluate G0-6 as FAIL with reason `SOLVER_STATS_INCOMPLETE_CELL=<case>_FIELD=<missing_field>`

### Requirement: G0 evidence schema is per-cell cell_summary KV plus cross-cell aggregate

For each of `{keliya, xinanjiang_upstream, heihe_x4, heihe_x16}` SHORT 90-day runs under `SHUD_LINSOL=amg`, the smoke harness MUST emit a single `cell_summary` KV block per cell. The block has **27 fields across 8 KV content lines**, plus `CELL_SUMMARY_BEGIN` / `CELL_SUMMARY_END` delimiter lines (10 lines total per block). The aggregator MUST consume all per-cell blocks and emit a cross-cell `aggregate.tsv` table.

Per-cell cell_summary KV block (G0 schema — multi-call semantics, NOT P8-tune.F single-cycle semantics):
```
CELL_SUMMARY_BEGIN
case=<name> interp_type=6 coarsen_type=8 NumY=<int> nnz_A=<int>
setup_wall_sec=<float> apply_wall_sec=<float> peak_rss_bytes=<int>
cycle_complexity=<float> operator_complexity=<float>
verdict_class=<AMG_OK|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_OOM|AMG_WALL_OVERFLOW>
hypre_version=<X.Y.Z> colpack_version=<X.Y.Z|absent> shud_pin=<short_sha>
cvode_nfe=<int> cvode_nli=<int> cvode_nfeLS=<int> cvode_ncfn=<int> cvode_ncfl=<int> cvode_netf=<int>
amg_telemetry_mean_iters=<float> amg_telemetry_mean_op_count=<float> amg_telemetry_setup_calls=<int> amg_total_setup_wall_sec=<float> amg_total_solve_wall_sec=<float>
n_cvode_steps=<int> amg_wall_per_step_sec=<float>
CELL_SUMMARY_END
```

Field semantics for multi-call context (clarifications vs P8-tune.F single-call semantics):
- `setup_wall_sec` = arithmetic mean of `setup_wall_sec` across all Setup callback invocations (per-call mean). Cumulative wall is `amg_total_setup_wall_sec`.
- `apply_wall_sec` = arithmetic mean of `solve_wall_sec` across all Solve callback invocations (per-call mean). Cumulative wall is `amg_total_solve_wall_sec`.
- `cycle_complexity` (Hypre 3.1.0 substitute semantics — G0 amendment): `HYPRE_BoomerAMGGetCumNnzAP(amg, &nnz_AP)` divided by IJMatrix `nnz_A` at end-of-run. This is dimensionally a STATIC hierarchy-size ratio, NOT the per-cycle work ratio of the original spec draft. Naming retained for compatibility with downstream aggregator code; semantics renamed in the ADR-0007 §Forward action amendment.
- `operator_complexity` = `HYPRE_BoomerAMGGetOperatorComplexity(amg, &op_complexity)` at end-of-run (Hypre native, not a derived estimate).
- `amg_telemetry_mean_iters` = arithmetic mean of `hypre_iters` (per-Solve `HYPRE_BoomerAMGGetNumIterations` — Hypre 3.1.0 substitute) across all Solve calls.
- `amg_telemetry_mean_op_count` = arithmetic mean of `hypre_op_count` (per-Solve `HYPRE_BoomerAMGGetCumNnzAP` — Hypre 3.1.0 substitute; STATIC hierarchy-size count) across all Solve calls. Note: because `CumNnzAP` is cumulative across Setup calls on the same AMG handle, the per-Solve series may be monotone non-decreasing rather than per-cycle varying; this is documented in the ADR-0007 amendment.
- `residual_reduction_v1` is **omitted** from G0 (P8-tune.F field had per-cycle semantics that do not extend cleanly to multi-cycle multi-Solve runs).
- `colpack_version` = `absent` literal token if PR-0 task 1.7 concludes ColPack is not required for the integrated AMG path.

#### Scenario: Cell with `verdict_class=AMG_OK` populates all 27 fields

- **WHEN** a cell completes successfully under `SHUD_LINSOL=amg`
- **THEN** its `cell_summary` block MUST contain non-NA values for all 27 fields (`case` through `amg_wall_per_step_sec`) AND `verdict_class=AMG_OK`; the aggregator MUST reject any cell_summary block with field count ≠ 27 by emitting `MARKER:CELL_SUMMARY_MALFORMED case=<name> got_fields=<N>` to stderr and evaluating that cell's contribution to G0-4 / G0-5 / G0-6 as FAIL

#### Scenario: Cell with verdict_class != AMG_OK fills NA sentinels

- **WHEN** a cell fails with one of the divergence/OOM/overflow verdicts
- **THEN** unmeasured fields (e.g., wall numbers if SIGTERM trap fires before measurement) MUST be filled with the literal token `NA`; the block MUST still emit all 27 fields and the BEGIN/END delimiters; the aggregator counts the cell against G0-4 (FAIL contribution) but exempts it from G0-6 (NA counters are acceptable on non-AMG_OK cells)

### Requirement: verdict_class enum semantics — G0 introduces AMG_OK as the success sentinel

The G0 `verdict_class` enum MUST be `{AMG_OK, AMG_SETUP_DIVERGE, AMG_SOLVE_DIVERGE, AMG_OOM, AMG_WALL_OVERFLOW}`. The success sentinel MUST be `AMG_OK` (NOT `PASS` as in archived P8-tune.F spec `amg-pattern-spike-verdict` REQ-2). The G0 aggregator parser MUST be a new G0-specific binary at `tools/p8tune.G0/aggregate_g0_smoke.sh` and MUST NOT reuse the P8-tune.F aggregator. The rename is intentional (AMG_OK is a stronger signal than PASS — it asserts both integrated convergence AND wall-signal validity at the cell level), and MUST be recorded in the ADR-0007 §Discussion or §Forward action amendment under the G0 verdict block.

#### Scenario: G0 aggregator rejects archived P8-tune.F PASS sentinel as malformed

- **WHEN** the G0 aggregator parser encounters a cell_summary block with `verdict_class=PASS` (P8-tune.F sentinel) in any G0 evidence file
- **THEN** the parser MUST emit `MARKER:VERDICT_CLASS_MALFORMED expected=AMG_OK got=PASS case=<name>` to stderr and evaluate that cell's contribution to G0-4 as FAIL

#### Scenario: ADR-0007 amendment records the AMG_OK rename rationale

- **WHEN** PR-C lands the G0 verdict
- **THEN** the ADR-0007 §Discussion or §Forward action Amendment block MUST contain a sub-paragraph titled "G0 verdict_class enum amendment" that names the renamed sentinel (`AMG_OK` replacing `PASS`) and the rationale (G0 evaluates integrated convergence + wall-signal jointly per cell, distinct from P8-tune.F's pattern-only PASS)

### Requirement: G0 verdict byte-identical anchor contract

The G0 verdict markdown document (under `docs/p8tune/amg_g0_verdict.md`) MUST contain a verdict-line block that is byte-identical to the line block emitted by the aggregator binary to stdout. This mirrors the P8-tune.F amg-pattern-spike-verdict spec REQ-6 contract.

#### Scenario: Aggregator stdout block matches doc block byte-for-byte

- **WHEN** aggregator emits its verdict block to stdout AND the verdict doc is regenerated from that aggregator run
- **THEN** the lines from `MARKER:G0_VERDICT_BEGIN` through `MARKER:G0_VERDICT_END` in the doc MUST be byte-identical to the corresponding lines in the aggregator stdout (verifiable via `diff <(extract aggregator stdout block) <(extract doc block)` returning zero output)

### Requirement: G0 verdict updates ADR-0007 via append-only Amendment block

The G0 verdict MUST append a new dated `## Amendment <YYYY-MM-DD> (G0 verdict)` block to `docs/adr/0007-amg-spike-decision.md` §Forward action section. The pre-existing ADR-0007 §Status metadata bullet (line 3: `- **Status**: Accepted ...`) and `## Decision` L2-header section MUST be preserved unchanged.

#### Scenario: ADR-0007 receives new Amendment block

- **WHEN** PR-C lands the G0 verdict doc
- **THEN** `docs/adr/0007-amg-spike-decision.md` contains a new `## Amendment <date> (G0 verdict)` section appended at the end of §Forward action AND the Status metadata bullet at line 3 reads exactly `- **Status**: Accepted ...` (unchanged from P8-tune.F merge) AND the `## Decision` section text is byte-identical to its pre-G0 form

#### Scenario: G0 verdict does NOT modify §Status bullet or §Decision section

- **WHEN** PR-C's diff is inspected
- **THEN** neither the Status metadata bullet nor the `## Decision` L2 header section is touched (verifiable via `git diff main -- docs/adr/0007-amg-spike-decision.md | grep -E '^[-+](- \*\*Status\*\*|## Decision)' | wc -l` returning 0; this regex catches the on-disk bullet form of Status — there is no `## Status` L2 header in ADR-0007 — plus the L2 header form of Decision)

### Requirement: Master plan §P8-tune.G0 anchor closes on both PASS and NO-GO outcomes

Regardless of `g0_verdict_branch`, PR-C MUST update `SHUD_openMP_master_plan.md` §P8-tune.G0 header from `[OPEN, HIGH]` to `[CLOSED]` (the gate is decided once G0 evaluation completes, PASS or FAIL). On `g0_verdict_branch=GO-G0`, the body adds a one-line summary referencing the verdict doc and ADR-0007 amendment date AND §P8-tune.G1 / §G2 anchors remain `[OPEN, HIGH]`. On `g0_verdict_branch=NO-GO-G0`, the body adds a NO-GO summary line AND §P8-tune.G1 / §G2 anchors update to `[CLOSED-DEFERRED, pending P8-tune.H GPU sparse fallback evaluation]`.

#### Scenario: PASS path closes G0 anchor; G1/G2 stay open

- **WHEN** PR-C lands with `g0_verdict_branch=GO-G0`
- **THEN** `SHUD_openMP_master_plan.md` §P8-tune.G0 header reads `## §P8-tune.G0 — Axis-4 telemetry + integrated AMG smoke [CLOSED]` AND the section body contains a `[verdict: GO-G0; see docs/p8tune/amg_g0_verdict.md; ADR-0007 amendment <date>]` line AND §P8-tune.G1 and §P8-tune.G2 headers remain `[OPEN, HIGH]`

#### Scenario: NO-GO path also closes G0 anchor; G1/G2 mark deferred

- **WHEN** PR-C lands with `g0_verdict_branch=NO-GO-G0`
- **THEN** `SHUD_openMP_master_plan.md` §P8-tune.G0 header reads `## §P8-tune.G0 — Axis-4 telemetry + integrated AMG smoke [CLOSED]` (the gate is decided, even on FAIL) AND the section body contains a `[NO-GO-G0 verdict; AMG production path closed for G1/G2; see docs/p8tune/amg_g0_verdict.md; ADR-0007 amendment <date>; P8-tune.H GPU sparse fallback evaluation may be invoked]` line AND §P8-tune.G1 and §P8-tune.G2 headers update to `[CLOSED-DEFERRED, pending P8-tune.H GPU sparse fallback evaluation]`

### Requirement: Wall-signal gate evaluation uses case-specific SPGMR baselines

The G0-5 wall-signal gate evaluation MUST compare AMG per-step wall against case-specific SPGMR baselines for `{heihe_x4, heihe_x16}`. PR-0 MUST measure `SPGMR_PER_STEP_HEIHE_X4_S` and `SPGMR_PER_STEP_HEIHE_X16_S` once each (running `SHUD_LINSOL=spgmr` SHORT 90-day on the respective cells) and pin the values into `tools/p8tune.G0/spgmr_baseline_walls_g0.h`. The 60-cell anchor `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` (in `tools/p8tune.D/spgmr_baseline_walls.h`) is retained as historical context only and is NOT used as a G0-5 threshold for either heihe_x4 or heihe_x16. PR-0 task 1.2 determines the setup-inclusion convention from the existing 60-cell baseline docstring AND applies the matching convention to BOTH the new case-specific baselines AND the AMG per-step measurements.

#### Scenario: Both numbers include setup

- **WHEN** PR-0 spike concludes that SPGMR baseline measurements include setup overhead
- **THEN** `amg_wall_per_step_sec = (amg_total_setup_wall_sec + amg_total_solve_wall_sec) / n_cvode_steps`; case-specific `SPGMR_PER_STEP_HEIHE_X4_S` and `SPGMR_PER_STEP_HEIHE_X16_S` likewise include the case's SPGMR setup overhead

#### Scenario: Both numbers exclude setup

- **WHEN** PR-0 spike concludes that SPGMR baseline measurements exclude setup
- **THEN** `amg_wall_per_step_sec = amg_total_solve_wall_sec / n_cvode_steps`; aggregator emits a second line `amg_wall_per_step_with_setup_sec = (amg_total_setup_wall_sec + amg_total_solve_wall_sec) / n_cvode_steps` for diagnostic alongside; case-specific baselines are similarly setup-excluded

#### Scenario: Aggregator emits explicit convention marker

- **WHEN** aggregator emits the wall-signal gate evaluation
- **THEN** the cell_summary aggregate row MUST contain a column `wall_convention ∈ {setup_included, setup_excluded}` reflecting the resolved PR-0 spike outcome; both heihe_x4 and heihe_x16 rows MUST carry the same value (PR-0 enforces consistency)

#### Scenario: Missing case-specific baseline yields wall_signal_unknown

- **WHEN** PR-0 task to measure `SPGMR_PER_STEP_HEIHE_X16_S` fails (e.g., heihe_x16 SPGMR run crashes) AND the case-specific constant is absent from `tools/p8tune.G0/spgmr_baseline_walls_g0.h`
- **THEN** G0-5 for heihe_x16 emits `wall_signal_unknown case=heihe_x16` and is excluded from the OR-aggregate; G0-5 PASS is still possible if `amg_improves[heihe_x4] == true` (the OR-aggregate falls through to the available case)

