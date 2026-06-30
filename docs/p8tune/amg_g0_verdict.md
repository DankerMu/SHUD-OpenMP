# P8-tune.G0 — AMG Spike Verdict (NO-GO-G0)

**Date**: 2026-06-30
**Verdict branch**: `NO-GO-G0`
**Aggregator evidence**: `.review-evidence/g0-amg-smoke-array-rerun/aggregate.tsv` + `tools/p8tune.G0/aggregate_g0_smoke.sh` stdout
**ADR amendment**: see `docs/adr/0007-amg-spike-decision.md` §Amendment 2026-06-30 (G0 verdict)
**Capability spec**: `openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md`

## Six-gate evaluation

| Gate | Result | One-line evidence |
|---|---|---|
| G0-1 (default-compat) | PASS | `SHUD_LINSOL` unset and `SHUD_LINSOL=spgmr` bit-identical to pre-G0 SPGMR baseline on `keliya` 90-day SHORT (PR-0 #414 merge; Mac per-platform anchor archived under `.review-evidence/g0-spgmr-baseline-90day-keliya/mac/`) |
| G0-2 (build) | PASS | `make shud` and `make shud_omp HYPRE=1` succeed on macOS (brew Hypre 3.1.x) + Ubuntu CI runner + server (`/scratch/frd_muziyao/local/hypre-3.1.0/`); Hypre 3.1.0 verified uniformly across CPU partition cn-nodes |
| G0-3 (telemetry-real) | PASS | `MARKER:AMG_TELEMETRY_REAL` present in `keliya` stdout; `cycle_complexity=1.000000` and `operator_complexity=1.002945` originate from Hypre native API (`HYPRE_BoomerAMGGetCumNnzAP` / `HYPRE_BoomerAMGGetOperatorComplexity`), NOT the P8-tune.F `2 × operator_complexity` hardcoded estimate |
| G0-4 (integrated-completes) | **FAIL** | `heihe_x16` is MALFORMED — Slurm SIGKILL at 8h wall budget fired before SIGTERM trap could emit `cell_summary`; Mac local re-run cannot reproduce 252k-NumEle case to upgrade verdict. Server array re-run with 24h budget is Future Work |
| G0-5 (wall-signal) | PASS | `heihe_x4` `amg_wall_per_step=0.092873s` < SPGMR baseline `0.238369s` (0.39× per-step; `wall_convention=wall_total_proxy`). G0-5 OR-aggregate succeeds via `heihe_x4`; `heihe_x16` AMG per-step unavailable due to MALFORMED |
| G0-6 (solver-stats) | PASS | `cvode_nfe / nli / nfeLS / ncfn / ncfl / netf` all non-NA for the 3 AMG_OK cells (`keliya`, `xinanjiang_upstream`, `heihe_x4`); `heihe_x16` MALFORMED exempt from G0-6 per spec |
| **Overall** | **NO-GO** | G0-4 gate FAIL blocks GO-G0 branch; total wall on `heihe_x4` regresses 15.1× independently (see §Wall-signal table) confirming AMG-not-beneficial on these hydrology matrix shapes |

## Anchor block (byte-identical to aggregator stdout per spec REQ "G0 verdict byte-identical anchor contract")

```
MARKER:G0_VERDICT_BEGIN
G0_3=PASS
G0_4=FAIL reason=heihe_x16:MALFORMED
G0_5=PASS
G0_6=PASS
G0_OVERALL=NO-GO
MARKER:G0_VERDICT_END
```

Verification command: extract the 7-line block from `tools/p8tune.G0/aggregate_g0_smoke.sh` stdout and from this doc using matching BEGIN/END marker lines, then diff — diff MUST return empty output per spec REQ "G0 verdict byte-identical anchor contract" contract (see spec under `openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md`).

## Wall-signal table

Per-cell `amg_wall_per_step` vs case-specific SPGMR baselines (`tools/p8tune.G0/spgmr_baseline_walls_g0.h`): `SPGMR_PER_STEP_HEIHE_X4_S = 0.238369`, `SPGMR_PER_STEP_HEIHE_X16_S = 0.952489` (PR-A measured + PR-B hot-patched from Slurm baselines). `nst` reflects CVODE step count produced under each linsol selection.

| cell | SPGMR per_step (s) | AMG per_step (s) | AMG/SPGMR ratio (per_step) | AMG nst | SPGMR nst | AMG/SPGMR nst ratio | Total wall ratio | wall_convention |
|---|---|---|---|---|---|---|---|---|
| keliya | n/a | 0.001432 (179s / 124913 nst, setup_plus_solve_only via TSV) | n/a | 124913 | n/a | n/a | n/a | telemetry_truncated |
| xinanjiang_upstream | n/a | 0.002709 (90s / 33223 nst; wall_total_proxy) | n/a | 33223 | n/a | n/a | n/a | wall_total_proxy |
| heihe_x4 | 0.238369 | 0.092873 (23660s / 254756 nst; wall_total_proxy) | 0.39× | 254756 | 6572 | 38.8× | **15.1× worse** | wall_total_proxy |
| heihe_x16 | 0.952489 | n/a (SIGKILL) | n/a | n/a | 6556 | n/a | n/a | nst_unavailable |

The `heihe_x4` per-step ratio 0.39× nominally satisfies the spec G0-5 OR-gate (`amg_wall_per_step < spgmr_wall_per_step`), but the 38.8× CVODE step inflation (254756 AMG nst vs 6572 SPGMR nst) drives total wall **15.1× worse** end-to-end. This is the "AMG-not-beneficial at scale" phenomenon predicted by P8-tune.F §6.4 case-asymmetric forward path: the V-cycle per-step advantage is wiped out by 39× more CVODE control failures (`ncfn=100138` on AMG vs typical single-digit ncfn on SPGMR).

## Solver-stats table

CVODE BDF/Newton iteration counters per `verdict_class=AMG_OK` cell (G0-6 evidence; spec REQ "Cell with verdict_class=AMG_OK populates all 27 fields"):

| cell | cvode_nfe | cvode_nli | cvode_nfeLS | cvode_ncfn | cvode_ncfl | cvode_netf |
|---|---|---|---|---|---|---|
| keliya | 246759 | 1726579 | 1785 | 30640 | 0 | 3 |
| xinanjiang_upstream | 50488 | 151528 | 2619 | 5264 | 0 | 0 |
| heihe_x4 | 502982 | 983928 | 124395 | 100138 | 0 | 2 |
| heihe_x16 | NA (MALFORMED) | NA | NA | NA | NA | NA |

Diagnostic observations:
- `ncfn` (Newton control failure count) is dominant on `heihe_x4` at 100138, ~20% of nfe — Newton iteration repeatedly fails convergence under AMG-preconditioned linear solve, forcing CVODE step retries with smaller dt. This is the proximate cause of the 38.8× nst inflation.
- `ncfl` (linear solver convergence failures) is uniformly 0 — AMG itself converges on every CVODE inner call; the bottleneck is at the outer Newton, not inner Krylov.
- `nfeLS` (RHS evaluations inside linear solver Setup) tracks `NumY` order-of-magnitude, consistent with FD-color Jacobian rebuild per Setup callback per spec REQ "Hypre Setup wrapper rebuilds hierarchy".

## Telemetry summary

Real Hypre telemetry captured from `tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp` Setup/Solve callbacks, drained pre-`CVodeFree` per PR-B Phase 6 hot-fix (SHUD `188854b` / outer pointer-bump `4ade422`):

- **keliya**: `cycle_complexity=1.000000` + `operator_complexity=1.002945` from real Hypre API calls (`HYPRE_BoomerAMGGetCumNnzAP` for cycle_complexity numerator; `HYPRE_BoomerAMGGetOperatorComplexity` for operator_complexity). These are STATIC hierarchy-size ratios per spec REQ "G0 schema multi-call semantics", not per-cycle work as in the original P8-tune.F spec draft.
- **TSV ring-buffer overflow**: 131072 rows retained + **115656 rows dropped** to ring overflow on keliya (47% loss over 1.7M-NLI run); surfaced via `MARKER:G0_TELEMETRY_RING_OVERFLOW` and `wall_convention=telemetry_truncated` per spec REQ "P1-2 ring-buffer-overflow surfaced". The retained sample is statistically valid for cycle_complexity (mean of 131k Solve calls) but biased toward initial-state hierarchy quality.
- **Wrapper Setup count**: total `Solves=246759` (= `cvode_nli`) across the run, with Setup rebuilds at ~1.98× nst frequency (per-step Newton iteration re-triggers Setup via `jcurPtr=SUNTRUE` when CVODE flags Jacobian staleness).
- **xinanjiang_upstream / heihe_x4 / heihe_x16**: copied from PR-A `.review-evidence/g0-amg-smoke-array/` (no telemetry TSV — PR-A wrapper pre-dated Phase 6 drain hook); `wall_convention=wall_total_proxy` per honest convention labeling. Re-running these on the post-Phase-6 wrapper would upgrade convention to `telemetry_truncated` or `setup_plus_solve_only` but would not change verdict semantics (G0-3 already PASS via keliya).

## Limitations

- **G0 is a SMOKE TEST, not a benchmark** — sample size = 4 cells (90-day SHORT each), not the 60-cell PRD anchor used in ADR-0004 P8-tune.C. Statistical power is sufficient to falsify the AMG-production hypothesis at the heihe_x4 scale but insufficient to characterize variance across the full case-roster.
- **`heihe_x16` SIGKILL trap timing**: Slurm fired SIGKILL at 8h wall budget before the wrapper SIGTERM trap could emit `AMG_WALL_OVERFLOW`-class `cell_summary`. The cell falls back to `MALFORMED` per spec REQ "Cell with `verdict_class=AMG_OK` populates all 27 fields" enforcement. **A server array re-run with 24h Slurm budget and the post-Phase-6 wrapper drain hook remains as Future Work** to distinguish `AMG_WALL_OVERFLOW_INFERRED_SIGKILL` vs `MALFORMED_RUNNER_BUG`. This re-run, even if it upgrades the heihe_x16 verdict, **will not change the G0_OVERALL=NO-GO outcome**: the `heihe_x4` total wall 15.1× regression independently shows AMG is not production-suitable on these hydrology matrix shapes.
- **A5 hydrology equivalence NOT exercised** — A5 (NSE/KGE/peak/water-balance) is G2 scope per spec; G0 measures convergence + wall + telemetry only.
- **Mac local re-run did not reproduce heihe_x16**: the local Apple M4 Pro lacks the 173 GiB RAM headroom that cn-nodes provide; heihe_x16 reproduction is server-scope only.
- **Per-step convention divergence**: `heihe_x4` AMG `wall_convention=wall_total_proxy` includes SHUD startup/IO; SPGMR baseline `0.238369s` also `wall_total_proxy`. Apples-to-apples comparison holds at the convention level but obscures the setup-vs-solve split. Future PR-A re-run with post-Phase-6 wrapper would emit `setup_plus_solve_only` convention enabling sharper attribution.

## Next steps

Per spec REQ "Master plan §P8-tune.G0 anchor closes on both PASS and NO-GO outcomes" + ADR-0007 §Forward action:

- **§P8-tune.G0 status**: `[OPEN, HIGH]` → `[CLOSED]` with NO-GO-G0 verdict line.
- **§P8-tune.G1 + §P8-tune.G2 status**: → `[CLOSED-DEFERRED, pending P8-tune.H GPU sparse fallback evaluation]`. AMG production path is closed per ADR-0007 §Decision; G1 18-cell integrated benchmark and G2 A5 hydrology equivalence are deferred pending alternative architectural substrate.
- **P8-tune.H GPU sparse fallback** (per ADR-0007 §Forward action L25 escape hatch) becomes the next investigation path. Forms under consideration: (i) CUDA sparse direct via `cuSPARSE` + iterative refinement, (ii) GPU AMG with mixed-precision (Hypre 2.30+ `cusparse_use=1`), (iii) domain decomposition on cn-node 32-core OpenMP with serial sparse direct per subdomain. GPU-presence gate (`sinfo -p GPU` confirming `gn01` availability) is precondition.
- **AMG production path remains CLOSED** per ADR-0007 §Decision (NO-GO-both strict + GO amended) + this G0 amendment. The G0 verdict adds integrated CVODE evidence to the existing pattern-only ADR-0007 decision but does **not** re-litigate the Accepted decision — instead it confirms the strict NO-GO-both verdict's empirical reality at the integrated-solver level.
- **`SHUD_LINSOL=spgmr` default preserved**: zero user-facing behavior change. The `SHUD_LINSOL=amg` opt-in remains available as a research knob but is NOT recommended for production large-case runs.

## References

- `openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md` — REQ "G0 verdict evaluates six PASS/FAIL gates" + REQ "G0 evidence schema" + REQ "verdict_class enum semantics" + REQ "G0 verdict byte-identical anchor contract" + REQ "G0 verdict updates ADR-0007 via append-only Amendment block" + REQ "Master plan §P8-tune.G0 anchor closes on both PASS and NO-GO outcomes" + REQ "Wall-signal gate evaluation uses case-specific SPGMR baselines"
- `docs/adr/0007-amg-spike-decision.md` — §Status (Accepted; preserved) + §Decision (NO-GO-both strict / GO amended; preserved) + §Forward action Amendment 2026-06-30 (G0 verdict)
- `tools/p8tune.G0/aggregate_g0_smoke.sh` — verdict aggregator (PR-B #416)
- `tools/p8tune.G0/spgmr_baseline_walls_g0.h` — case-specific baselines (PR-A measured + PR-B hot-patched)
- `tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp` — Hypre linsol wrapper with telemetry drain hook (PR-B #416 Phase 6)
- `.review-evidence/g0-amg-smoke-array-rerun/` — PR-B output evidence (aggregate.tsv + per-cell logs + telemetry TSV)
- `docs/p8tune/p8tune_g0_academic_summary.md` — academic-style epic summary (user pref 2026-06-25)
- PR-0 #414 — wrapper Initialize + linsol selector + Mac G0-1 baseline
- PR-A #415 — 4-cell AMG smoke runner + Slurm sbatch + dlopen versioned-soname
- PR-B #416 — aggregator + telemetry drain hook + G0-3/4/5/6 verdict markers
