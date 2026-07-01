# ADR-0008: P8 Solver-Substitution Research Line Closure

## Status: Accepted (2026-06-30)

- **Date**: 2026-06-30
- **Deciders**: DankerMu + Claude orchestrator (per PR-X1 #420 G0-RCA evidence + retrospective synthesis of P8-tune.{B,C,D,E,F,G0} epic chain)
- **Owner**: SHUD-OpenMP 改造工程 / P8-tune retrospective closure
- **Tags**: p8tune / closure / solver-substitution / klu / boomeramg / hypre / spgmr / cvode / retrospective / forward-plan / a5-infra / cvode-policy / domain-decomposition
- **Supersedes**: none (consolidating closure ADR)
- **Superseded by**: none
- **Related**: ADR-0007 amg-spike-decision (predecessor; PR-X1 amendment appended) + ADR-0005 klu-spike-decision (KLU NO-GO chain trigger) + ADR-0004 maxl-sweep-decision (SPGMR maxl Optional opt-in baseline) + ADR-0003 precond-spike-decision (PREC_NONE production default) + ADR-0002 solver-path (P1e Path 1 selection) + master plan §P8-tune.{B,C,D,E,F,G0,G1,G2,H} all CLOSED-FINAL + `docs/p8tune/p8_retrospective_academic_summary.md` (academic-style retrospective)

---

## Context

The P8 solver-substitution research line was anchored after P1e (epic #283) closed via §4.6.2 partial-closure SHIP with `ExecPolicy::StrictOMP` RHS delivering 1.7× heihe_x4 / 1.066× heihe wall improvement under Path 1 (Serial NVector + StrictOMP RHS). The forward objective: identify a CPU-side linear-solver substitution that could deliver hydrology-acceptable acceleration on heihe_x4 (NumEle ≈ 40k, NumY ≈ 124k) and heihe_x16 (NumEle ≈ 160k, NumY ≈ 485k) production-target meshes within the SUNDIALS-CVODE 6.0.0 BDF/Newton framework.

Six successive sub-epics tested four solver substrates:

1. **P8-precond (P8-tune.B)** — physical block-diagonal preconditioner over SPGMR — closed by ADR-0003 PREC_NONE NO-GO (block-Jacobi `nfeLS / nfe` floor did not improve relative to PREC_NONE baseline).
2. **P8-tune.C** — SPGMR `maxl` 6-PR sweep — closed by ADR-0004 Optional-knob (`SHUD_SPGMR_MAXL=30` for small heihe N=1 Performance-tier opt-in; heihe_x4 regressed across all maxl ≥ 10).
3. **P8-tune.D** — KLU direct solve pattern-only spike — closed by ADR-0005 Case-aware (keliya/heihe pattern-feasible / heihe_x4 wall margin 1.87× / heihe_x16 wall margin 17.9× structurally infeasible).
4. **P8-tune.E.small-only** — KLU env-var mini-prototype for small cases — remains [OPEN, OPTIONAL/medium] per ADR-0005 §Forward action, untouched by this closure (small-case scope, no production-target gating).
5. **P8-tune.F** — BoomerAMG/Hypre pattern-only spike — closed by ADR-0007 strict NO-GO-both (Axis 4 hard-coded estimate trip) / amended GO (Axes 1/2/3/5 PASS at all 16 cells; Saad 2003 §13 V-cycle bound shows Axis 4 ≈ 2.0 is expected for pure V-cycle).
6. **P8-tune.G0** — Integrated CVODE + SUNLinSol_Hypre BoomerAMG smoke — closed by G0 verdict NO-GO-G0 (G0-4 heihe_x16 MALFORMED; heihe_x4 per-step PASS but total wall 15.1× WORSE due to 38.8× CVODE step inflation driven by `ncfn=100138` Newton control failures).

PR-X1 #420 (G0 root-cause analysis, merged main commit `39bd6b1`) tested the only remaining hypothesis for AMG path rescue: that the G0 ncfn explosion was a Hypre solve tolerance (`SHUD_AMG_TOL`) × CVODE EpsLin (`SHUD_CVODE_EPSLIN`) mismatch. An 8-cell heihe_x4 90-day Slurm array (job 10248) swept `AMG_TOL ∈ {1e-7, 1e-9, 1e-11, 1e-13}` × `EpsLin ∈ {0.05 default, 0.005}`. Outcome: across 4 orders-of-magnitude in `AMG_TOL` × 10× change in `EpsLin`, `ncfn` stays in a tight 98,286-104,795 window (vs SPGMR baseline 49); `ncfl=0` across all completed cells. Hypothesis REFUTED — the ncfn explosion is intrinsic to the CVODE outer Newton + step controller when AMG replaces SPGMR, not a tolerance mismatch addressable by inner-solve tuning.

PR-X1 closes the AMG path as a CPU-side production option. Combined with KLU NO-GO at heihe_x4 + heihe_x16 (ADR-0005), the full P8 solver-substitution research line — substituting CVODE's linear-solver substrate as the production CPU acceleration path — has exhausted its tested options.

This ADR formally records that closure decision and locks the forward direction.

---

## Decision

**Adopt closure of the P8 solver-substitution research line.**

No further KLU / BoomerAMG / SUNLinSol_Hypre productionization work is scheduled on the CPU side. The two existing alternative paths that were never the production substitution target stand:

1. **Production CPU baseline** = **P1e StrictOMP RHS** (`ExecPolicy::StrictOMP` + Serial NVector + PREC_NONE SPGMR maxl=5 default) + **`SHUD_SPGMR_MAXL=30` small-case Performance-tier opt-in** (per ADR-0004 Optional-knob). Per AC-S3 D7: 1.729× heihe_x4 / 1.066× heihe wall improvement vs B1b baseline at N=8.
2. **GPU sparse fallback (§P8-tune.H)** — also CLOSED-FINAL per this closure; GPU is NOT pursued (user direction in PR-X1 mandate). Re-opening would require new ADR.

Forward investigation moves to two **non-substitution** paths (anchored in this ADR §Forward action):

- **A5 validation infrastructure** (PR-Y1): standalone hydrology-acceptance pipeline (NSE/KGE/peak-timing/water-balance/runoff) usable to validate ANY acceleration tier, decoupled from solver-substitution scope.
- **CVODE outer policy tuning** (PR-Z1, §P9): `reltol`, `MaxStep`, `MaxNonlinIters`, `NonlinConvCoef`, vector `abstol` upper-bound spot check on SPGMR under A5 acceptance. This is **policy** tuning of the existing SPGMR baseline, NOT solver substitution.

- **Domain decomposition (§P10)** is logged DESIGN-ONLY with NO implementation commitment until PR-Z1 returns ≥1.5× wall improvement under A5; even then, the decision to start an implementation epic is gated on a fresh ADR.

The decision recognizes that the SHUD hydrology matrix shape — combining surface / unsat / GW / river / lake physics with strongly anisotropic time-scale coupling — does not present a linear-algebra structure where CPU-side direct or multigrid substrate substitution recovers acceleration. SPGMR with PREC_NONE remains the algorithmically correct default; further gains, if any, must come from CVODE outer-loop policy or from architecture-level restructuring (decomposition), not from inner linear solver substitution.

---

## Outcome Table

| Epic         | Date       | Solver / scope                                | Wall outcome (vs SPGMR PREC_NONE baseline)                    | Closure reason                                                                                                                                  |
|--------------|------------|-----------------------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| P8-tune.B    | 2026-06-27 | Block-Jacobi preconditioner over SPGMR        | `nfeLS / nfe` floor unchanged                                  | ADR-0003 PREC_NONE NO-GO; block-Jacobi did not improve Krylov convergence over PREC_NONE baseline                                              |
| P8-tune.C    | 2026-06-28 | SPGMR `maxl` sweep (5 → 50)                   | heihe N=1 +12% Optional; heihe_x4 all maxl ≥ 10 REGRESS         | ADR-0004 Optional-knob; Krylov saturates `maxl=5` at NumY ≥ 100K; `SHUD_SPGMR_MAXL=30` Performance opt-in only                                |
| P8-tune.D    | 2026-06-29 | KLU pattern-only spike (`klu_analyze_factor`) | heihe_x4 wall margin 1.87× over 0.7×SPGMR; heihe_x16 17.9×     | ADR-0005 Case-aware; KLU fill axis fine but AMD-reordered LU O(N^1.5) saturates at NumY ≥ 100K                                                |
| P8-tune.E.small | 2026-06-29 | KLU mini-prototype env-var hook (keliya/heihe) | Out of scope for this closure; remains [OPEN, OPTIONAL/medium] | NOT closed by this ADR; small-case scope only, no production-target gating                                                                     |
| P8-tune.F    | 2026-06-29 | BoomerAMG/Hypre pattern-only spike (16 cells)  | All 16 cells PASS hierarchy build + V-cycle apply             | ADR-0007 strict NO-GO-both (Axis 4 hardcoded estimate) / amended GO (4-axis); Axis 4 demoted to diagnostic; integrated verification deferred to G0 |
| P8-tune.G0   | 2026-06-30 | Integrated SUNLinSol_Hypre BoomerAMG smoke    | heihe_x4 per-step 0.39× faster; **total wall 15.1× WORSE**     | G0 NO-GO; G0-4 heihe_x16 MALFORMED + per-step PASS / total FAIL; `ncfn=100138` Newton control failures inflate nst 38.8×                       |
| P8-tune.G0-RCA (PR-X1) | 2026-06-30 | AMG_TOL × EpsLin sanity matrix (8-cell)    | ncfn ∈ [98,286, 104,795] across 4 orders-of-mag × 10× sweep      | Hypothesis REFUTED; tolerance mismatch is NOT the ncfn root cause; AMG path closed as CPU production option                                    |

Per-epic SPGMR PREC_NONE baseline at heihe_x4 N=1: per-step 0.226579 s (ADR-0004 60-cell PRD anchor) / 0.238369 s (case-specific G0 anchor) / total wall 1566.55 s (PR-X1 90-day SPGMR baseline).

---

## Evidence anchors

Post-merge of PR-X1 (main commit `39bd6b1`):

- **`.review-evidence/g0-amg-rca-pr-x1/`** (PR-X1, on main) — 8-cell heihe_x4 90-day RCA sweep:
  - `aggregate_rca.tsv` — 8-cell matrix (`task_id`, `amg_tol`, `epslin`, `verdict_class`, `wall_total_sec`, `n_cvode_steps`, `cvode_ncfn`, `cvode_ncfl`, `cvode_nfe`, `cvode_nli`, `cvode_netf`)
  - `README.md` — verdict block + matrix outcome + hypothesis-refuted narrative

- **`.review-evidence/g0-amg-smoke-array-rerun/`** (PR-B #416 G0 4-cell smoke) — G0 verdict source data (aggregate.tsv + per-cell logs + cell-keliya.telemetry.tsv)

- **`.review-evidence/p8tune-amg-pr-b/`** (P8-tune.F PR-B #403) — 16-cell pattern-only sweep (4 case × 4 (interp_type, coarsen_type))

- **`.review-evidence/p8tune-amg-pr-c/`** (P8-tune.F PR-C #404) — aggregator output + ADR-0007 byte-identical verdict block

- **`.review-evidence/p8tune-klu-spike-pr-a/`** + **`.review-evidence/p8tune-klu-spike-pr-b/`** (P8-tune.D #385 + #387) — 16-cell KLU pattern-only sweep + 3-axis verdict

- **`.review-evidence/p8tune-spgmr-maxl-prd-60cell/`** (P8-tune.C PR-D #373) — 60-cell SPGMR maxl PRD baseline

**Pinned SHUD submodule SHAs** (per merge sequence):
- PR-X1 closure (main): outer `39bd6b1` / SHUD per submodule pointer in that commit
- P8-tune.G0 capstone (PR-E #413): outer per merge-commit / SHUD `188854b` (PR-B Phase 6 drain-hook hot-fix)
- P8-tune.F capstone: outer per PR-D / SHUD `1ab61c0`
- P8-tune.D capstone: outer per PR-C #388 / SHUD `3aec657`-derived pattern
- P1e capstone (baseline anchor): SHUD `0b3998d`

**Pinned external dependencies**:
- SUNDIALS-CVODE 6.0.0 (unchanged across entire P8 epic chain)
- Hypre 3.1.0 (cn-node `/scratch/frd_muziyao/local/hypre-3.1.0/`; macOS brew Hypre 3.1.x)
- ColPack reused via P8-tune.D shell-out (P8-tune.G0 path does not link ColPack)
- SuiteSparse KLU pinned at version present on cn-nodes per P8-tune.D PR-0

---

## Consequences

### Positive

1. **P1e StrictOMP RHS is the production CPU baseline, formally locked**. Six successive solver-substitution explorations have NOT found a CPU-side substrate that beats P1e at production-target scale. This locks the engineering and forensic-debugging investment in the StrictOMP RHS path as the durable CPU baseline, freeing future epic budget for non-substitution directions (A5 infra / CVODE policy / decomposition design).
2. **Decisive negative-result corpus** — six closed sub-epic ADRs (0003/0004/0005/0007 + this 0008) + four detailed docs/p8tune/* academic summaries (`p8tune_d_academic_summary.md` + `p8tune_f_academic_summary.md` + `p8tune_g0_academic_summary.md` + `p8_retrospective_academic_summary.md`) form a publication-quality negative-result corpus on CPU sparse-solver substitution for hydrology BDF/Newton integration. The hypothesis-driven closure pattern (H1/H2/H3 stated → tested → refuted) makes the closure machine-readable and audit-traceable.
3. **`SHUD_LINSOL=spgmr` (default) preserved as zero-user-impact path**. Production users see zero behavior change from this closure. `SHUD_LINSOL=amg` opt-in remains in the codebase as a research knob for any future re-investigation (e.g., if Hypre 3.2+ adds case-aware interp_type heuristics or alternative coarsening combos demonstrate breakthrough).
4. **Forward direction is well-typed**. A5 infrastructure (PR-Y1) is the gating prerequisite for ANY future acceleration claim — without standardized NSE/KGE/peak/timing/runoff metrics, future epic comparisons against production SPGMR have no validation tier. P9 (PR-Z1) is a bounded scope CVODE policy sweep that delivers a definite answer (≥1.5× under A5, or NO-GO triggers P10 design re-evaluation). P10 design-only (no implementation commitment) avoids the meta-error of opening a 6-month decomposition epic before A5 + P9 establish whether further gains are even reachable.

### Negative

1. **CPU-side solver substitution as an acceleration substrate is empirically exhausted**. For SHUD's hydrology matrix shape at NumY ≥ 100K, neither direct (KLU) nor multigrid (BoomerAMG) substrates beat preconditioned SPGMR maxl=5 in integrated CVODE wall time. This negative result limits future acceleration work to (a) outer-loop policy tuning (P9), (b) architecture restructuring (P10 design / decomposition), or (c) GPU substrate (explicitly NOT pursued, requires new ADR if revisited). All three are higher-risk and higher-effort than the closed substitution paths.
2. **heihe_x16 remains untested under AMG**. G0-4 heihe_x16 MALFORMED was never upgraded to a clean wall measurement. The PR-X1 8-cell RCA covered heihe_x4 only. We do not have direct evidence for heihe_x16 AMG behavior; closure relies on the algorithmic argument (Newton control failure mode is intrinsic to outer-loop, not scale-specific to NumY).
3. **Symmetric verdict for the GPU path**. We explicitly close §P8-tune.H GPU sparse evaluation as part of this closure. This is a stronger position than the prior "fallback if AMG fails" framing in ADR-0007 §Forward action. The closure reasoning: user direction at PR-X1 closure mandates CPU-first, with GPU deferred indefinitely.
4. **A5 infrastructure is currently incomplete**. PR-Y1 has not been written; until it lands, future acceleration claims (including PR-Z1) cannot be validated to hydrology-acceptance standards. This is a known limitation that PR-Y1 is the gating fix for.

### Neutral

1. **No SHUD source change, no production behavior change** in this PR. ADR-0008 is documentation-only.
2. **P8-tune.E.small-only is NOT closed** by this ADR. It remains [OPEN, OPTIONAL/medium] per ADR-0005 §Forward action and may proceed independently. This closure ADR covers only the production-target solver substitution chain.
3. **`SHUD_AMG_TOL` and `SHUD_CVODE_EPSLIN` env hooks** introduced in PR-X1 remain in the codebase as research knobs. They are not deprecated; they simply have no production use-case after the AMG closure.
4. **OpenSpec changes for the closed epics** remain archived per their respective capstone PRs (P8-tune.D #388, P8-tune.F #404+#412 capstone-merge, P8-tune.G0 #413 capstone-merge). No new OpenSpec change is opened for this retrospective closure (closure = documentation; not an OpenSpec capability change).

---

## Acceptance criteria

PR-X2 acceptance (this PR):

- [x] `docs/adr/0008-p8-solver-substitution-closure.md` authored — §Status (Accepted 2026-06-30) + §Context + §Decision + §Outcome Table + §Evidence anchors + §Consequences + §Forward action + §References + §Amendment policy.
- [x] `docs/adr/0007-amg-spike-decision.md` amended — `### Amendment 2026-06-30 — G0-RCA outcome (PR-X1 #420)` block appended at the end of §Forward action; §Status (L3) + §Decision (L49-92) byte-identical to pre-PR-X2 state.
- [x] `docs/p8tune/p8_retrospective_academic_summary.md` authored — academic-style retrospective per the user pref (YAML + Abstract + §Introduction H1-H4 + §Related Work + §Methodology + §Experimental Setup + §Results + §Discussion + §Limitations + §Conclusion + §Future Work + §References).
- [x] `SHUD_openMP_master_plan.md` updated — §P8-tune.{G0,G1,G2,H} CLOSED-FINAL annotations + new §A5-infra + §P9 + §P10 sections; sub-section ordering preserved (G0 → G1 → G2 → H).
- [x] `CLAUDE.md` (gitignored) updated locally + rsync to server.
- [x] PR opened against `main` from `feat/p8-retrospective-adr-0008-closure`.

---

## Forward action

### Five-step plan (PR-Y1 / PR-Z1 / P10-design / GPU / SPGMR-retained)

1. **PR-Y1 — §A5-infra standalone validation pipeline**: implement NSE / KGE / peak magnitude / peak timing / runoff / water-balance metric extraction harness independent of any solver. Inputs: paired `*.rivqdown.dat` + `*.stage.dat` + forcing reference. Outputs: per-case metric report + machine-readable verdict (PASS/FAIL per indicator). Scope: hydrology-acceptance infrastructure usable by ANY future acceleration tier. Priority: HIGH (gates all subsequent acceleration claims). Estimated budget: 2-3 PR sequence.
2. **PR-Z1 — §P9 CVODE outer-policy tuning spot check**: 1-2 PR sequence sweeping `reltol ∈ {1e-4, 1e-5, 1e-6}` × `MaxStep ∈ {default, 3600s, 1800s}` × `MaxNonlinIters ∈ {3, 4, 5}` × `NonlinConvCoef ∈ {0.1, 0.05, 0.01}` × vector `abstol` (per-state-component) on SPGMR baseline. Validate against A5 (PR-Y1 infra). Gate: ≥1.5× heihe_x4 wall improvement under A5 PASS. Priority: MEDIUM. Estimated budget: 1-2 PR sequence.
3. **§P10 CPU domain decomposition (DESIGN-ONLY)**: document architectural scoping for subbasin / river-network natural decomposition + interface flux coupling + per-subdomain local KLU or SPGMR solver. NO implementation commitment. Re-evaluate IF (and only if) PR-Z1 fails to deliver ≥1.5× under A5. Priority: LOW (design ledger only). Estimated budget: 1 design-doc PR.
4. **GPU substrate NOT pursued** per user direction at PR-X1 closure. §P8-tune.H CLOSED-FINAL in this ADR. Re-opening requires a new ADR-NNNN with fresh GPU hardware audit (cn-node + gn01 + scale-out) + cost-benefit analysis.
5. **P1e StrictOMP RHS retained as production CPU baseline**. `SHUD_SPGMR_MAXL=30` small-case Performance opt-in remains valid per ADR-0004. `SHUD_LINSOL=spgmr` (default) and `SHUD_RHS_THREADS=<N>` env hooks unchanged. Zero behavior change from this closure for production users.

### Suppressed alternatives (documented for completeness)

The following candidate paths were considered and explicitly rejected by this closure:

- **Continue AMG investigation** (PR-X1 follow-up at alternative coarsen/interp combos beyond F-winner): rejected — H4 (PR-X1 hypothesis) refutation across 4 orders-of-magnitude tolerance × 10× EpsLin is sufficient evidence that the AMG-preconditioned outer Newton has an intrinsic control-failure mode unrelated to inner-solve quality. Further coarsen/interp exploration carries low expected value relative to PR-Y1 + PR-Z1.
- **MUMPS or alternative direct solver**: rejected — ADR-0005 §Forward action L242 already documented MUMPS as out-of-scope (heavier-weight dependency, no SUNDIALS adapter). Same algorithmic argument as KLU (O(N^1.5) fill at production scale) applies.
- **PETSc AMG or PETSc Krylov+precond**: rejected — ADR-0007 §Forward action L240 documented PETSc as out-of-scope (vendor sprawl, adapter shim required). Same algorithmic class as Hypre BoomerAMG; G0-RCA refutation generalizes.
- **AMG as smoother only (not preconditioner)**: rejected — would require deeper SUNDIALS Krylov-precond plumbing; if the underlying Newton control-failure mode is intrinsic, smoother-only deployment carries the same risk.
- **CUDA sparse solver fallback**: rejected per user direction (CPU-only forward).

---

## Amendment policy

§Status bullet and §Decision section are **byte-identical invariants** once this ADR is Accepted (mirror ADR-0007 amendment pattern). New evidence (e.g., future architectural-substrate ADRs, GPU re-evaluation, decomposition implementation results) does NOT modify §Status / §Decision — instead, a new dated `## Amendment YYYY-MM-DD — <topic>` section MUST be appended at the end of §Forward action, mirroring `docs/adr/0007-amg-spike-decision.md` §"Amendment 2026-06-30 (G0 verdict)" + "Amendment 2026-06-30 — G0-RCA outcome (PR-X1 #420)" pattern.

Verification command (post-merge of any future amendment):

```bash
# Confirm §Status + §Decision byte-identical to pre-amendment baseline
git diff <pre-amendment-sha> HEAD -- docs/adr/0008-p8-solver-substitution-closure.md | \
  awk '/^\+\+\+|^---/ {next} /^[+-]## Status:/||/^[+-]## Decision/||/^[+-]## Outcome Table/ {found=1; print} END {exit found}'
# Expected: empty output + exit 0
```

---

## References

### Internal (本仓库)

- `docs/adr/0007-amg-spike-decision.md` (predecessor; Amendment 2026-06-30 — G0-RCA outcome appended in PR-X2 this PR)
- `docs/adr/0005-klu-spike-decision.md` (KLU NO-GO chain trigger)
- `docs/adr/0004-maxl-sweep-decision.md` (SPGMR maxl Optional opt-in baseline; `SHUD_SPGMR_MAXL=30` Performance-tier)
- `docs/adr/0003-precond-spike-decision.md` (PREC_NONE production default)
- `docs/adr/0002-solver-path.md` (P1e Path 1 — Serial NVec + StrictOMP RHS; production baseline anchor)
- `docs/p8tune/p8_retrospective_academic_summary.md` (this PR — academic retrospective)
- `docs/p8tune/p8tune_g0_academic_summary.md` (G0 academic summary)
- `docs/p8tune/p8tune_f_academic_summary.md` (P8-tune.F academic summary)
- `docs/p8tune/p8tune_d_academic_summary.md` (P8-tune.D academic summary)
- `docs/p8tune/amg_g0_verdict.md` (G0 verdict source-of-truth)
- `docs/p8tune/amg_spike_verdict.md` (P8-tune.F verdict source-of-truth)
- `docs/p1e/p1e_academic_summary.md` (P1e baseline / template for academic summaries)
- `SHUD_openMP_master_plan.md` §P8-tune.{B,C,D,E,F,G0,G1,G2,H} (all CLOSED-FINAL per this ADR) + §A5-infra + §P9 + §P10 (new sections)
- `tools/p8tune.G0/aggregate_g0_smoke.sh` (G0 aggregator)
- `tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp` (SUNLinSol_Hypre wrapper; remains in codebase as research knob)
- `tools/p8tune.G0/spgmr_baseline_walls_g0.h` (case-specific SPGMR baselines)
- `tools/p8tune.F/aggregate_amg_spike.sh` (P8-tune.F aggregator)
- `tools/p8tune.D/klu_analyze_factor.cpp` (KLU spike binary)

### PR sequence (epic chain triggering this closure)

- PR #313/#315/#316 — P1e PR-F/G/H (production baseline establishment)
- PR #369-#373 + #368 + #376 — P8-tune.C 6-PR SPGMR maxl sweep + G7 amendment
- PR #384/#385/#387/#388 — P8-tune.D 4-PR KLU pattern-only spike
- PR #394/#402/#403/#404 + #412 (capstone-merge) — P8-tune.F 5-PR BoomerAMG pattern-only spike
- PR #414/#415/#416/#417 (PR-C amendment) + #413 (capstone-merge to main) — P8-tune.G0 4-PR integrated AMG smoke
- PR #418 (PR-D from baseline branch) + #419 (post-merge cleanup) — P8-tune.G0 epic close
- PR #420 — PR-X1 G0-RCA tolerance × EpsLin matrix (the closure trigger)
- This PR (PR-X2) — ADR-0008 retrospective closure + master plan §P8 CLOSED-FINAL + 5-step forward plan

### Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-rca-pr-x1-runs/` — PR-X1 Slurm job 10248 sbatch output dir (8 cells; rsync'd to repo at `.review-evidence/g0-amg-rca-pr-x1/`)
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0_amg_smoke.10014/` — G0 PR-A Slurm 10014 output (4 cells; rsync'd to repo at `.review-evidence/g0-amg-smoke-array-rerun/`)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x4/` — heihe_x4 90-day-truncated AutoSHUD deployment (常驻; reused across P8-tune.D / P8-tune.F / P8-tune.G0 / PR-X1)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` — heihe_x16 90-day-truncated AutoSHUD deployment (常驻; reused across P8-tune.F / P8-tune.G0)

### External

- SUNDIALS v6.0.0 user guide — CVODE BDF/Newton iteration + `CVodeGetNumNonlinSolvConvFails` (ncfn semantics) + `SUNLinSol_SPGMR` + `SUNLinSol_KLU` + `SUNLinSol_Hypre` adapter contracts
- Hypre 3.1.0 user guide — BoomerAMG `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve` + `HYPRE_BoomerAMGGet*` telemetry API
- SuiteSparse KLU user guide — `klu_analyze` + `klu_factor` + AMD/COLAMD ordering
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. §13 multigrid + §6 Krylov methods (V-cycle bound + GMRES analysis)
- Henson, V. E. & Yang, U. M. (2002). "BoomerAMG: A parallel algebraic multigrid solver and preconditioner." *Applied Numerical Mathematics* 41(1): 155-177.
- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. (KLU AMD/COLAMD/BTF anchor)
- Hindmarsh, A. C. et al. (2005). "SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers." *ACM Trans. Math. Softw.* 31(3): 363-396.

### Academic / theoretical

- Brown, P. N. & Saad, Y. (1990). "Hybrid Krylov methods for nonlinear systems of equations." *SIAM J. Sci. Stat. Comput.* 11(3): 450-481. — inexact Newton convergence theory (relevant to outer Newton + Krylov inner-tolerance interaction studied in PR-X1)
- Eisenstat, S. C. & Walker, H. F. (1996). "Choosing the forcing terms in an inexact Newton method." *SIAM J. Sci. Comput.* 17(1): 16-32. — forcing-term selection theory (relevant to CVODE EpsLin role probed in PR-X1)
- George, A. & Liu, J. W. H. (1981). *Computer Solution of Large Sparse Positive Definite Systems*. Prentice-Hall. — 2D mesh PDE nested-dissection O(N^1.5) flops bound (KLU NO-GO at production scale anchor per ADR-0005)
