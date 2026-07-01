# ADR-0010: CPU Acceleration Program — Status Consolidation + P10 Decision Point

## Status: Accepted (2026-07-01)

- **Date**: 2026-07-01
- **Deciders**: DankerMu + Claude orchestrator (per user directive at post-PR-Z2 planning turn; consolidates ADR-0007 / ADR-0008 / ADR-0009 into a single onboarding + handoff reference)
- **Owner**: SHUD-OpenMP 改造工程 / CPU acceleration program status consolidation
- **Tags**: consolidation / cpu-acceleration / p8-closed / p9-closed / p10-deferred / gpu-not-pursued / production-baseline / a5-infra / onboarding
- **Supersedes**: none (aggregation ADR — does NOT supersede ADR-0007/0008/0009; those remain the authoritative decision records for their respective lines)
- **Superseded by**: none
- **Related**: ADR-0002 (P1e Path 1 solver-path anchor) + ADR-0003 (PREC_NONE production default) + ADR-0004 (SPGMR maxl Optional opt-in; `SHUD_SPGMR_MAXL=30` Performance-tier) + ADR-0005 (KLU NO-GO chain trigger) + ADR-0007 (AMG NO-GO strict + G0-RCA amendment) + ADR-0008 (P8 solver-substitution line CLOSED-FINAL) + ADR-0009 (P9 CVODE outer-policy CLOSED) + master plan §P8-tune.{B,C,D,E,F,G0,G1,G2,H} / §A5-infra / §P9 / §P10 / §P1e / `docs/p8tune/p8_retrospective_academic_summary.md` + `docs/p9/p9_academic_summary.md` + `docs/p1e/p1e_academic_summary.md` + CLAUDE.md project brief (gitignored; mirrors this state)

---

## Context

This ADR consolidates the multi-epic exploration of CPU-side acceleration for the SHUD-OpenMP hydrologic model over the 2025-2026 program cycle. It is **not a new decision** — it aggregates decisions already made in ADR-0007 (AMG NO-GO strict + G0-RCA amendment), ADR-0008 (P8 solver-substitution line closure), and ADR-0009 (P9 outer-policy closure) into a single source of truth for:

- What was tried and closed
- What is retained as the production baseline
- What is explicitly out of scope going forward
- What single remaining decision point (P10 CPU domain decomposition) is deferred pending a dedicated planning turn

The purpose is onboarding aid for future contributors + explicit handoff point for whoever picks up P10 evaluation. It prevents re-litigating decided closures and provides a single-page map of the current program state.

---

## Program summary

The SHUD-OpenMP acceleration program set out to deliver production-acceptable CPU speedup for large NWM cases (heihe_x4 40k-element ≈ 124k NumY, heihe_x16 160k-element ≈ 485k NumY) while preserving hydrology-acceptable behavior. Over ~15 months and ~30 tracked PRs, the program converged on:

- **P1e StrictOMP RHS** (production, deterministic, 1.729× heihe_x4 / 1.066× heihe wall improvement vs B1b baseline at N=8)
- **`SHUD_SPGMR_MAXL` small-case opt-in** (env-var knob; heihe-like Performance-tier; not production-recommended for large cases)
- **A5 hydrology-acceptance validation pipeline** (retained infra, PR-Y1 + PR-Y2 fix)
- All other explored acceleration paths CLOSED

The program's negative-result corpus (6+ closed sub-epic ADRs + 3 academic-style retrospectives) documents systematically why CPU-side solver substitution and CVODE outer-policy tuning did not deliver meaningful acceleration on production-target meshes.

---

## Closed research lines

### CLOSED — P8 solver-substitution family

Reference: ADR-0008 §Decision + §Outcome Table + §Consequences.

- **P8pre (P8-tune.B)** — physical block-diagonal / identity preconditioner over SPGMR: CLOSED. Block-Jacobi `nfeLS / nfe` floor did not improve over PREC_NONE baseline (per ADR-0003).
- **P8-tune.C — SPGMR `maxl` sweep**: PARTIALLY RETAINED as `SHUD_SPGMR_MAXL` small-case opt-in (heihe N=1 +12% Performance-tier per ADR-0004). Large-case path CLOSED — heihe_x4 all maxl ≥ 10 REGRESS; Krylov subspace saturates at NumY ≥ 100K.
- **P8-tune.D — KLU direct solver pattern-only spike**: CLOSED per ADR-0005 Case-aware verdict. Pattern-feasible on keliya/heihe; heihe_x4 wall margin 1.87×; heihe_x16 wall margin 17.9× (AMD-reordered LU O(N^1.5) on 2D-mesh PDE saturates at production scale). The `SHUD_LINSOL=klu` small-case env-var hook remains OPEN (ADR-0005 §Forward action; not gating this ADR).
- **P8-tune.F — BoomerAMG (Hypre) pattern-only spike**: CLOSED via ADR-0007 (strict NO-GO-both / amended GO for Axes 1/2/3/5). 16-cell pattern sweep achieved 16/16 PASS on hierarchy build + V-cycle apply, but Axis 4 was demoted to hierarchy-quality diagnostic per Saad 2003 §13 (V-cycle work ≈ 2× operator_complexity is expected for pure V-cycle, not a rescue-blocker).
- **P8-tune.G0 — Integrated CVODE + SUNLinSol_Hypre BoomerAMG smoke**: CLOSED via G0 NO-GO verdict (per-step 0.39× faster, but total wall 15.1× WORSE due to `ncfn=100138` Newton control failures inflating nst 38.8×).
- **P8-tune.G0-RCA (PR-X1 #420)** — AMG_TOL × EpsLin sanity matrix: CLOSED via G0-RCA amendment (ADR-0007 Amendment 2026-06-30). Cross-scale evidence, 8-cell heihe_x4 sweep across `SHUD_AMG_TOL ∈ {1e-7, 1e-9, 1e-11, 1e-13}` × `SHUD_CVODE_EPSLIN ∈ {0.05, 0.005}`: `ncfn` stays in `[98,286, 104,795]` window across 4 orders-of-magnitude × 10× tolerance change. Interpretation: inner linear-solve tolerance has zero leverage on outer Newton control-failure explosion when AMG replaces SPGMR at fixed CVODE outer policy. The AMG rescue hypothesis is exhausted.
- **P8-tune.G1 (18-cell integrated benchmark) / G2 (A5 hydrology equivalence) / H (GPU fallback)**: CLOSED-FINAL per ADR-0008, downstream of G0-RCA refutation. Section text preserved for historical context.

### CLOSED — P9 CVODE outer-policy tuning

Reference: ADR-0009 §Decision + §Outcome Table + §Consequences.

- **Scope**: bounded sweep of CVODE outer-loop policy parameters (`reltol`, `MaxStep`, `MaxNonlinIters`, `NonlinConvCoef`, vector `abstol`) on the existing SPGMR PREC_NONE maxl=5 baseline — NOT solver substitution.
- **Gate** (per ADR-0008 §Forward action step 2): ≥ 1.5× heihe_x4 wall improvement under A5 PASS; fallback `optional_p9` at 1.2×.
- **PR-Z1 #423 outcome** — 2-cell heihe_x4 90-day spot check on Slurm job 10465 (`cell-0-baseline` at cfg.para default `reltol=1e-4` vs `cell-1-p9` at `SHUD_CVODE_RELTOL=1e-3`, one order-of-magnitude looser):
  - `baseline_wall_sec=1658` / `p9_wall_sec=1618` → **`wall_speedup=1.0247`** (2.5%, below 1.2× fallback threshold and 47.5 percentage points below the 1.5× GO gate)
  - `baseline_nst=6572` / `p9_nst=6515` → `nst_ratio=0.9913` (0.87% step reduction under 10× reltol relaxation — indicates the CVODE step controller is at the stiffness-imposed floor, not the tolerance-imposed floor)
  - `baseline_ncfn=49` / `p9_ncfn=12` (Newton control failures drop 4×, but the baseline was already low)
  - `baseline_nli=30476` / `p9_nli=29381` (Krylov iterations drop 3.6%)
  - A5 near-identity on all six real streamflow metrics: NSE=1.0000, KGE=0.9999, peak_magnitude_ratio=0.9998, peak_timing_offset=0, runoff_volume_ratio=1.0000, monthly_bias_mae=5.52e-05. The p9 trajectory is hydrologically equivalent to baseline at machine precision.
  - Formal `a5_verdict=FAIL` was driven **entirely** by the `water_balance_residual = 7.52e+13` metric bug in `tools/a5/metrics.py` — fixed in PR-Y2 #425 (water_balance now unit-consistent + safe NaN fallback when mesh-area metadata absent; 61/61 pytest coverage retained).
- **Interpretation**: heihe_x4 default `reltol=1e-4` is already moderately loose (most stiff-BDF hydrology settings run 1e-6 to 1e-8). The step controller is at the stiffness-imposed floor. Further reltol relaxation cannot recover meaningful wall time. The remaining four axes (`MaxStep`, `MaxNonlinIters`, `NonlinConvCoef`, vector `abstol`) have theoretically narrower operating ranges than reltol, so the theoretical ceiling of the full P9 family is bounded and does not justify further sweep budget.

---

## Retained deliverables

### Production baseline: P1e StrictOMP RHS

- **1.729× heihe_x4 / 1.066× heihe wall improvement** vs B1b baseline at N=8 (per ADR-0002 P1e capstone AC-S3 D7)
- **Deterministic**: owner-local gather + hot-field SoA + reordering (per ADR-0001 soa-hot-fields + master plan §P1e capstone)
- **A0..A5 correctness**: full A3a bitwise vs B1b @ 4 threads passed; A5 hydrology-acceptance PASS on the 6 streamflow metrics (post-PR-Y2 water balance fix)
- **Merged and stable** since P1e capstone (main commits per submodule pointer)
- **Recommended for all production runs**. `SHUD_LINSOL=spgmr` (default) + `SHUD_RHS_THREADS=<N>` env hooks unchanged from P1e.

### Small-case opt-in: `SHUD_SPGMR_MAXL`

- **Env variable** (per ADR-0004 §Decision); default unset = current SPGMR maxl=5 (bit-identical to P1e baseline)
- **Setting `SHUD_SPGMR_MAXL=30`** gave measurable improvement on `keliya` (484 elements, N=1 Performance-tier) and similar small cases (heihe N=1 +12%)
- **NOT recommended as a production default** because effect on large cases (heihe_x4 all maxl ≥ 10 REGRESS 6.86-24.82%; heihe_x16 Krylov working set exceeds L3) is null-to-negative
- **Kept as a knob** for future small-case investigation without needing recompile. Correctness envelope is A5-informational only, not A5-gated (per ADR-0004 §Consequences).

### Validation infrastructure: A5 pipeline (`tools/a5/`)

- **7 hydrology-acceptance metrics**: NSE, KGE, peak magnitude ratio, peak timing offset, runoff volume ratio, monthly bias MAE, water balance residual
- **YAML-driven thresholds** (`config/a5_thresholds.default.yaml`)
- **Machine-readable output** — `MARKER:A5_VERDICT` block for aggregator consumption (byte-identical contract per PR-Y1)
- **61/61 pytest coverage** after PR-Y2 #425 water_balance unit-consistency fix
- **Water balance metric** is currently informational (falls back to NaN when mesh-area metadata is absent per PR-Y2 fix). A Tier-2 area-weighted parser is deferred to a future PR-Y3 if a use-case emerges; the six real streamflow metrics (NSE, KGE, peak-magnitude, peak-timing, runoff-volume, monthly-bias) are the operational A5 gate.
- **Retained** as the standing validation gate for any future solver-substitution or policy-tuning work. Decoupled from any specific solver substrate (per ADR-0008 §A5-infra anchor + this ADR §Retained).

---

## Explicit NOT-doing

The following are ruled out as directions for the CPU-acceleration program:

- **GPU acceleration**: per user directive (PR-X1 closure preserved by ADR-0008 §Forward action step 4; §P8-tune.H CLOSED-FINAL). CPU-only is the strategic direction. **No GPU epic will be opened without an explicit reversal of this decision** (new ADR-NNNN authoring + fresh GPU hardware audit + cost-benefit analysis).
- **Further global linear-solver substitution on large cases**: the P8 line covered SPGMR-preconditioned / KLU / AMG. Cross-scale evidence (PR-X1 8-cell RCA sweep + G0 integrated smoke) shows the **CVODE outer policy is the bottleneck, not the inner solver**, when substituting a non-SPGMR substrate. Newton control-failure mode is intrinsic to the outer BDF/Newton loop and generalizes from heihe_x4 to heihe_x16 by algorithmic argument. **No new inner-linear-solver epic without new external evidence** (e.g. upstream Hypre or SUNDIALS work outside this project's scope demonstrating a fundamentally different outer-policy interaction).
- **Further P9 reltol-family sweeps on heihe_x4 at current cfg.para reltol=1e-4**: bounded at ~2.5% wall-leverage-per-order-of-magnitude by PR-Z1 #423 evidence. Reopening requires a fundamentally different acceptance framing (e.g. multi-case A5, or reltol vs a different baseline, or a joint sweep with a different axis) rather than an extension of the same axis.
- **Continued exploration of `SHUD_LINSOL=amg` / `SHUD_AMG_TOL` / `SHUD_CVODE_EPSLIN` env hooks** as production tuning axes: these env hooks remain in the codebase as **research knobs** (per ADR-0007 Amendment 2026-06-30 + ADR-0008 §Consequences §Neutral item 3), not as production tuning surfaces. No production documentation, no user-facing recommendation, no CI gate.

---

## Deferred — P10 CPU domain decomposition

**Status: design-only, no implementation commitment. Requires a dedicated planning turn.**

P10 is the only remaining CPU-side acceleration path with a plausible ceiling above the P1e production baseline. Prior analysis (2026-06-30 in ADR-0008 §Forward action step 3) + Linus-role review at the PR-Z2 planning turn both concluded: **P10 is a 6-12 month engineering commitment with high uncertainty**, not a "6 PRs and ship" epic.

Concrete risks (per prior analysis + PR-Z2 review synthesis):

1. **Delaunay mesh partition cuts element-pair flux** — not a simple ghost-cell BC. The SHUD element pair (triangular Delaunay) exchanges lateral GW + unsat + surface flux at every shared edge; partitioning cuts a subset of these edges and requires interface flux substitution or ghost-element replication.
2. **River routing is a directed graph** — cross-partition river segments introduce Schwarz iteration on hyperbolic dynamics. Hyperbolic Schwarz convergence is classically slow (transport-CFL-limited outer iterations) and may not compose cleanly with the BDF/Newton outer loop that is the current SHUD architecture.
3. **Lake mass balance cannot be partitioned** — internal fully-coupled constraint. Lake compartments (e.g. Juyan Lake in heihe) exchange with multiple upstream/downstream river segments and adjacent groundwater elements; splitting the lake state across subdomains violates mass balance.
4. **Outer Newton failure may just move from global to interface Schwarz outer loop** — the G0-RCA evidence (PR-X1 #420) suggests that ncfn-driven step inflation is intrinsic to CVODE outer BDF/Newton control. A domain-decomposed formulation replaces the global Newton with a per-subdomain Newton coupled by outer Schwarz iteration; there is no a-priori guarantee that ncfn goes away — it may reappear as interface Schwarz non-convergence.
5. **A5 acceptance under DDM is non-trivial** — interface flux correction preserving mass balance is a separate contract from the per-subdomain solve. Even if per-subdomain SPGMR PREC_NONE runs cleanly and produces A5-passing single-subdomain metrics, the global A5 (mass balance, monthly bias, runoff volume ratio) requires demonstrating that the interface coupling does not degrade global water balance beyond the ADR-0008 §P8-tune.G2 indicator thresholds.

**Gate to open P10 implementation** (ALL of the following must hold; no automatic downstream consequence of P9 closure):

1. **A dedicated P10-0 design-only feasibility spike lands first**, targeting:
   - Partition feasibility catalog (Delaunay partition strategy candidates + river/lake handling proposals)
   - Coupling risk catalog (per-subdomain Newton + Schwarz outer loop + BDF interaction)
   - Theoretical speedup ceiling (given the coupling constraints, what's the best plausible wall improvement on heihe_x4)
   - A5 risk estimate (which of the 6 streamflow metrics is most at risk under DDM formulation)
   - **The P10-0 spike ships analysis + a partition prototype (no CVODE coupling), NOT an implementation branch.**
2. **P10-0 concludes ceiling ≥ 1.5× wall improvement is plausible** on heihe_x4 given the coupling constraints, **with A5 acceptance credible** (not just "not obviously broken" — credible under the operational A5 thresholds).
3. **Engineering commitment is explicitly re-approved** as a separate planning turn (this ADR-0010 does NOT authorize P10-0; a future ADR-NNNN + planning turn does).

Absent (1) + (2) + (3), **P10 remains deferred indefinitely**. The absence of P10 does **NOT** invalidate the P1e production baseline — that ships regardless.

---

## Decision (this ADR)

The following five decisions are formalized by ADR-0010:

1. **Consolidate P8 / P9 closure** into a single reference document (this ADR). Prevents re-litigating decided closures.
2. **Formalize the production baseline** as P1e StrictOMP RHS + `SHUD_SPGMR_MAXL` small-case opt-in. Zero user-visible behavior change from the P1e capstone default. `SHUD_LINSOL=spgmr` (default) preserved.
3. **Preserve A5 pipeline** as standing validation infrastructure (`tools/a5/`). Decoupled from any specific solver substrate. Ready to gate any future acceleration claim.
4. **Rule out** GPU acceleration, further global linear-solver substitution on large cases, further P9 reltol-family sweeps on the current baseline, and continued exploration of `SHUD_LINSOL=amg` / `SHUD_AMG_TOL` / `SHUD_CVODE_EPSLIN` as production tuning axes.
5. **Mark P10** as design-only-deferred, gated by an explicit future planning turn (P10-0 feasibility spike → ceiling re-check → engineering commitment re-approval). This ADR-0010 does NOT authorize any P10 work.

---

## Amendment policy

§Status bullet and §Decision section are **byte-identical invariants** once this ADR is Accepted (2026-07-01) — mirror ADR-0007 / ADR-0008 / ADR-0009 amendment pattern. New closures, reopenings, or P10-0 spike outcomes do NOT modify §Status / §Decision. Instead, a new dated `## Amendment YYYY-MM-DD — <topic>` section MUST be appended at the end of §Deferred (P10 section).

Verification command (post-merge of any future amendment):

```bash
# Confirm §Status + §Decision byte-identical to pre-amendment baseline
git diff <pre-amendment-sha> HEAD -- docs/adr/0010-cpu-acceleration-status-and-p10-decision.md | \
  awk '/^\+\+\+|^---/ {next} /^[+-]## Status:/||/^[+-]## Decision/ {found=1; print} END {exit found}'
# Expected: empty output + exit 0
```

---

## References

### Internal ADRs (本仓库)

- `docs/adr/0001-soa-hot-fields.md` — Hot-field SoA layout (P1e deterministic gather anchor)
- `docs/adr/0002-solver-path.md` — P1e Path 1 (Serial NVec + StrictOMP RHS) production baseline
- `docs/adr/0003-precond-spike-decision.md` — PREC_NONE production default
- `docs/adr/0004-maxl-sweep-decision.md` — SPGMR maxl Optional opt-in; `SHUD_SPGMR_MAXL=30` Performance-tier
- `docs/adr/0005-klu-spike-decision.md` — KLU NO-GO chain trigger (Case-aware verdict)
- `docs/adr/0007-amg-spike-decision.md` — AMG NO-GO strict / GO amended + G0-RCA amendment (2026-06-30)
- `docs/adr/0008-p8-solver-substitution-closure.md` — P8 solver-substitution line CLOSED-FINAL (2026-06-30)
- `docs/adr/0009-p9-cvode-outer-policy-closure.md` — P9 CVODE outer-policy CLOSED (2026-07-01)

### Academic retrospectives (本仓库)

- `docs/p1e/p1e_academic_summary.md` — P1e baseline (production anchor; academic-summary template母本)
- `docs/p8tune/p8_retrospective_academic_summary.md` — Full P8 retrospective (H1-H4 evidence chain)
- `docs/p8tune/p8tune_d_academic_summary.md` — P8-tune.D KLU spike retrospective
- `docs/p8tune/p8tune_f_academic_summary.md` — P8-tune.F BoomerAMG pattern-only retrospective
- `docs/p8tune/p8tune_g0_academic_summary.md` — P8-tune.G0 integrated AMG smoke retrospective
- `docs/p9/p9_academic_summary.md` — P9 CVODE outer-policy retrospective (H1 falsified evidence)

### Master plan sections

- `SHUD_openMP_master_plan.md` §P1e (CLOSED-FINAL production baseline)
- `SHUD_openMP_master_plan.md` §P8-tune.{B,C,D,E.small-only,F,G0,G1,G2,H} (all CLOSED-FINAL per ADR-0008; E.small-only remains OPTIONAL/medium out of scope of this closure)
- `SHUD_openMP_master_plan.md` §A5-infra (RETAINED)
- `SHUD_openMP_master_plan.md` §P9 (CLOSED per ADR-0009)
- `SHUD_openMP_master_plan.md` §P10 (DESIGN-ONLY, POST-P9-CLOSURE deferred, per this ADR)

### PR sequence (post-P8 closure chain)

- PR #418 — P8-tune.G0 capstone-merge (G0 NO-GO verdict landed on main)
- PR #420 — PR-X1 G0-RCA tolerance × EpsLin matrix (H4 refuted; AMG rescue hypothesis exhausted)
- PR #421 — PR-X2 ADR-0008 P8 solver-substitution line closure
- PR #422 — PR-Y1 A5 pipeline (standalone hydrology-acceptance validation)
- PR #423 — PR-Z1 `SHUD_CVODE_RELTOL` env hook + 2-cell heihe_x4 A5-gated spot check
- PR #424 — PR-Z2 ADR-0009 P9 CVODE outer-policy closure
- PR #425 — PR-Y2 A5 water_balance_residual unit-consistency fix + safe NaN fallback
- PR #426 — PR (this ADR) ADR-0010 consolidation + P10 decision framing

### Evidence directories (`.review-evidence/`)

- `.review-evidence/g0-amg-rca-pr-x1/` — 8-cell heihe_x4 90-day AMG_TOL × EpsLin sweep (Slurm 10248; H4 refutation source-of-truth)
- `.review-evidence/p9-spot-pr-z1/` — 2-cell heihe_x4 90-day P9 spot check (Slurm 10465; reltol 1e-4 vs 1e-3 verdict source-of-truth)
- `.review-evidence/p8tune-amg-pr-b/` + `.review-evidence/p8tune-amg-pr-c/` — 16-cell P8-tune.F BoomerAMG pattern-only sweep + aggregator verdict
- `.review-evidence/g0-amg-smoke-array-rerun/` — G0 4-cell integrated AMG smoke (source of G0 NO-GO wall inflation evidence)
- `.review-evidence/p8tune-klu-spike-pr-a/` + `.review-evidence/p8tune-klu-spike-pr-b/` — P8-tune.D 16-cell KLU pattern-only sweep + 3-axis verdict
- `.review-evidence/p8tune-spgmr-maxl-prd-60cell/` — P8-tune.C 60-cell SPGMR maxl PRD baseline

### Tooling (retained infrastructure)

- `tools/a5/` — A5 hydrology-acceptance validation pipeline (7 metrics; PR-Y1 + PR-Y2)
- `tools/p9.spot/` — P9 CVODE outer-policy spot-check aggregator + sbatch template (PR-Z1; retained as research knob infra)
- `tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp` — SUNLinSol_Hypre wrapper (retained in codebase as research knob; not production)
- `tools/p8tune.G0/spgmr_baseline_walls_g0.h` — case-specific SPGMR baselines
- `tools/p8tune.F/aggregate_amg_spike.sh` — P8-tune.F aggregator
- `tools/p8tune.D/klu_analyze_factor.cpp` — KLU spike binary (retained; not production)

### External

- SUNDIALS v6.0.0 user guide — CVODE BDF/Newton iteration + `CVodeSetReltol` + `SUNLinSol_SPGMR` + `SUNLinSol_KLU` + `SUNLinSol_Hypre` adapter contracts + step-controller theory + BDF stability regions
- Hypre 3.1.0 user guide — BoomerAMG `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve` + `HYPRE_BoomerAMGGet*` telemetry API
- SuiteSparse KLU user guide — `klu_analyze` + `klu_factor` + AMD/COLAMD ordering
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. §13 multigrid V-cycle bound (Axis 4 ≈ 2× operator_complexity expected) + §6 Krylov methods
- Henson, V. E. & Yang, U. M. (2002). "BoomerAMG: A parallel algebraic multigrid solver and preconditioner." *Applied Numerical Mathematics* 41(1): 155-177.
- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. (KLU AMD/COLAMD/BTF anchor + O(N^1.5) 2D-mesh PDE bound)
- Hindmarsh, A. C. et al. (2005). "SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers." *ACM Trans. Math. Softw.* 31(3): 363-396.
- Brown, P. N. & Saad, Y. (1990). "Hybrid Krylov methods for nonlinear systems of equations." *SIAM J. Sci. Stat. Comput.* 11(3): 450-481. — inexact Newton convergence theory (outer Newton + Krylov inner-tolerance interaction studied in PR-X1)
- Eisenstat, S. C. & Walker, H. F. (1996). "Choosing the forcing terms in an inexact Newton method." *SIAM J. Sci. Comput.* 17(1): 16-32. — forcing-term selection theory (CVODE EpsLin role probed in PR-X1)
- Hairer, E. & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer. §IV.8 step-size control for stiff BDF — anchors the "step-controller floor" argument in P9 closure.
- George, A. & Liu, J. W. H. (1981). *Computer Solution of Large Sparse Positive Definite Systems*. Prentice-Hall. — 2D mesh PDE nested-dissection O(N^1.5) flops bound (KLU NO-GO at production scale anchor per ADR-0005)

### Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-rca-pr-x1-runs/` — PR-X1 Slurm job 10248 sbatch output dir (8 cells; rsync'd to repo at `.review-evidence/g0-amg-rca-pr-x1/`)
- `/scratch/frd_muziyao/SHUD-OpenMP/.p9-spot-pr-z1-runs/` — PR-Z1 Slurm job 10465 sbatch output dir (2 cells; rsync'd to repo at `.review-evidence/p9-spot-pr-z1/`)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x4/` — heihe_x4 90-day-truncated AutoSHUD deployment (常驻; reused across P8-tune.D / P8-tune.F / P8-tune.G0 / PR-X1 / PR-Z1)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` — heihe_x16 90-day-truncated AutoSHUD deployment (常驻; reused across P8-tune.F / P8-tune.G0)

### Project brief (gitignored)

- `CLAUDE.md` §"项目速览" — mirrors this ADR's consolidated state (program status memo bullet added in this PR-#426; rsync'd to server per project convention)
