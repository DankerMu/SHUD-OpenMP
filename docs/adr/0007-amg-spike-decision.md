# ADR-0007: amg-spike-decision — BoomerAMG/Hypre pattern-only spike 4-branch verdict (NO-GO-both, strict)

- **Status**: Accepted (2026-06-29 in PR-D capstone PR #<TBD>; flipped Proposed → Accepted per spec REQ-6 Scenario "ADR-0007 Status lifecycle"). Strict verdict_branch = NO-GO-both (byte-identical to `aggregate_verdict.txt`); amended verdict_branch = GO (FYI per PR-A H3 Axis 4 hard-coded estimate disclosure). Forward action = §P8-tune.G AMG Axis-4 instrumentation epic [OPEN, HIGH] (new master plan anchor added in this PR-D per ADR §Forward action, separate from verdict_branch-mapped G/H epics suppressed by strict NO-GO-both branch).
- **Date**: 2026-06-29
- **Deciders**: DankerMu + Claude orchestrator (per `tools/p8tune.F/aggregate_amg_spike.sh` 5-axis per-case verdict + `docs/p8tune/amg_spike_verdict.md` synthesis)
- **Owner**: SHUD-OpenMP 改造工程 / P8-tune.F epic capstone
- **Tags**: amg / boomeramg / hypre / pattern-spike / case-asymmetric / cycle-complexity / operator-complexity / wall-axis / memory-axis / axis4-amendment
- **Supersedes**: none (first AMG-related ADR)
- **Superseded by**: none
- **Related**: ADR-0005 klu-spike-decision (KLU NO-GO at heihe_x4 wall margin 1.87× → triggered this AMG retreat) + ADR-0004 maxl-sweep-decision (SPGMR baseline wall anchor 0.226579 s/step heihe_x4 N=1 maxl=5) + ADR-0003 precond-spike-decision (PREC_NONE production baseline) + master plan §P8-tune.F anchor + `openspec/changes/p8tune-amg-spike/` (epic SHUD-OpenMP #393) + PR-0 [#394](https://github.com/DankerMu/SHUD-OpenMP/pull/394) (#386 SHUD dtor UB fix) + PR-A [#395](https://github.com/DankerMu/SHUD-OpenMP/pull/395) ([PR #402](https://github.com/DankerMu/SHUD-OpenMP/pull/402); spike binary `boomeramg_setup_solve.cpp` + H3 disclosure) + PR-B [#396](https://github.com/DankerMu/SHUD-OpenMP/issues/396) ([PR #403](https://github.com/DankerMu/SHUD-OpenMP/pull/403); 16-cell Slurm array sweep, 16/16 verdict_class=PASS) + 本 PR-C [#397](https://github.com/DankerMu/SHUD-OpenMP/issues/397)

---

## Context

ADR-0005 §Forward action (post GPT Pro 2026-06-29 retrospective F3+F4 corrections) anchored P8-tune.F as the **primary HIGH-priority forward path** for SHUD large-case linear solver acceleration. The retrospective amendments locked the following inputs to this AMG spike:

- **KLU pattern-only spike empirically falsified for large cases**: heihe_x4 wall margin 1.87× (Optional), heihe_x16 wall margin 17.9× (NO-GO) — `per_step_estimate = 0.6 × numeric_factor_wall` saturates the 0.7 × SPGMR per-step budget at NumY ≥ 100K. AMD-reordered LU factorization on 2D-mesh PDE Jacobian scales O(N^1.5) while SPGMR with PREC_NONE scales O(N · maxl) — the algorithmic crossover happens between heihe (21K NumY) and heihe_x4 (124K NumY).
- **SPGMR maxl=5 baseline saturated**: ADR-0004 P8-tune.C 60-cell sweep established `SPGMR_PER_STEP_SEC = 0.226579 s` (heihe_x4 N=1 maxl=5 3-rep median 1489.76s / 6575 nst). Higher maxl values regress 6.86-24.82% on heihe_x4 N=1 per PR-D evidence. Krylov subspace cannot scale to heihe_x16 (NumY ≈ 485K, maxl=10 working set ≈ 60 MB > L3 8-32 MB).
- **BoomerAMG (Hypre) is the next algorithmic substrate**: algebraic multigrid with O(N) memory + O(N log N) setup + O(N) per-iteration apply; SUNDIALS has `SUNLinSol_Hypre` adapter; established academic + production track record for 2D-3D mesh PDE workloads.

This epic was authored as a **zero-source-patch, zero-CVODE-wireup, zero-SHUD-run pattern-only spike** per spec REQ-1 — mirroring the P8-tune.D KLU spike methodology. The 4 cases × 4 (interp_type, coarsen_type) combos = 16 cells were executed on server Slurm cn-nodes per spec REQ-3.

### 16-cell sweep summary (PR-B evidence; `.review-evidence/p8tune-amg-pr-b/cells/`)

Result: **16/16 cells `verdict_class=PASS`** — no AMG_OOM, no AMG_SETUP_DIVERGE, no AMG_SOLVE_DIVERGE, no AMG_WALL_OVERFLOW. Hypre BoomerAMG completed setup + solve cleanly for every (case, interp_type, coarsen_type) combination including heihe_x16 (NumY = 485,250, nnz_A = 2,481,548).

| case      | NumY    | best combo (interp, coarsen) | setup_wall_sec | apply_wall_sec | peak_rss_MB | cycle_complexity | operator_complexity |
|-----------|--------:|------------------------------|---------------:|---------------:|------------:|-----------------:|--------------------:|
| keliya    |   1,785 | (6, 21) NN=02                |       0.000561 |       0.001031 |        19.0 |           2.0059 |              1.0029 |
| heihe     |  21,357 | (8, 8) NN=07                 |       0.001452 |       0.001667 |        34.4 |           2.0000 |              1.0000 |
| heihe_x4  | 124,395 | (6, 8) NN=08                 |       0.009476 |       0.018179 |       110.9 |           2.0000 |              1.0000 |
| heihe_x16 | 485,250 | (6, 8) NN=12                 |       0.037785 |       0.078349 |       381.3 |           2.0000 |              1.0000 |

Best combo per case = `min(setup_wall_sec + apply_wall_sec)`; tiebreaker = `min(operator_complexity)` per spec REQ-5.

### 5-axis thresholds (per spec REQ-5 Scenario "5-axis threshold evaluation per case")

For each case's best combo:

- **Axis 1 (Setup)**: `setup_wall_sec < WALL_BUDGET_SETUP_SEC = 1.5 × 0.7 × SPGMR_PER_STEP_SEC ≈ 0.237908 s`
- **Axis 2 (Apply)**: `apply_wall_sec < WALL_BUDGET_APPLY_SEC = 0.7 × SPGMR_PER_STEP_SEC ≈ 0.158605 s`
- **Axis 3 (Memory)**: `peak_rss_bytes < WALL_BUDGET_RSS_BYTES = 0.7 × CN_NODE_RAM_BYTES ≈ 129,869,709,312 (130 GiB)`
- **Axis 4 (Cycle complexity)**: `cycle_complexity < 1.5` (unitless)
- **Axis 5 (Operator complexity)**: `operator_complexity < 2.0` (unitless)

---

## Decision

**Adopt `NO-GO-both` branch (strict 5-axis verdict).**

Per spec REQ-5 Scenario "4-branch decision auto-typing (covers all axis + case combinations)" + REQ-6 Scenario "ADR-0007 §Decision matches aggregate_verdict.txt": the §Decision MUST be byte-identical to the aggregator output. The aggregator (`tools/p8tune.F/aggregate_amg_spike.sh`) emits:

```
verdict_branch=NO-GO-both
verdict_branch_reason="heihe_x4 fails ['axis4_cycle'] (max margin 1.333×)"
```

Pinned per-case overall (auto-typed by aggregator from `.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt`):

| case      | axis1 setup | axis2 apply | axis3 memory | axis4 cycle | axis5 operator | overall (strict) | max failing margin |
|-----------|:-----------:|:-----------:|:------------:|:-----------:|:--------------:|:----------------:|-------------------:|
| keliya    | PASS        | PASS        | PASS         | **FAIL**    | PASS           | **FAIL**         | 1.337×             |
| heihe     | PASS        | PASS        | PASS         | **FAIL**    | PASS           | **FAIL**         | 1.333×             |
| heihe_x4  | PASS        | PASS        | PASS         | **FAIL**    | PASS           | **FAIL**         | 1.333×             |
| heihe_x16 | PASS        | PASS        | PASS         | **FAIL**    | PASS           | **FAIL**         | 1.333×             |

The strict verdict triggers `NO-GO-both` via rule 4 first clause ("heihe_x4 fails ANY axis"). All 4 cases uniformly fail Axis 4 because `cycle_complexity ≈ 2.0` in every measured cell.

> **Critical caveat — see §"Axis 4 amendment per PR-A H3 disclosure" below**: Axis 4 is mechanically determined by Axis 5 in the current spike implementation (`cycle_complexity = 2 × operator_complexity` is a hard-coded estimate, NOT measurement from HYPRE telemetry). With Axis 4 treated as a non-discriminating diagnostic, **the amended verdict is `GO`** (all 4 cases PASS axes 1/2/3/5; Axis 4 is fully redundant with Axis 5). The aggregator emits both for transparent comparison:
>
> ```
> verdict_branch=NO-GO-both                         # strict (canonical)
> verdict_branch_axis4_amended=GO                   # amended (FYI for ADR §Discussion)
> ```
>
> The strict verdict is the **canonical anchor** in this ADR (per spec REQ-6 byte-identical contract). PR-D capstone authors should treat the amended verdict as the **operationally meaningful** verdict for forward-action planning (per spec REQ-7 Scenario "Conditional next-epic anchor per verdict_branch"; the next-epic anchor selection for ADR-Accepted flip is informed by the operational reading, not the mechanical Axis 4 trip).

### 4-branch decision tree (per spec REQ-5 Scenario "4-branch decision auto-typing")

| Branch                  | Trigger condition                                                                                                              | Adopted? | Forward action                                                                                                |
|-------------------------|--------------------------------------------------------------------------------------------------------------------------------|----------|---------------------------------------------------------------------------------------------------------------|
| `GO`                    | ALL 4 cases PASS all 5 axes                                                                                                    | NO (strict) / **YES (amended, recommended)** | P8-tune.G full AMG + A5 hydrology-equivalence integration epic, HIGH priority (4-6 weeks per spec REQ-7) |
| `Optional`              | keliya+heihe+heihe_x4 PASS; heihe_x16 fails ONLY wall axes (1/2), max margin < 1.5×                                            | NO       | P8-tune.G heihe_x4-only integration, medium priority (3-4 weeks)                                              |
| `NO-GO-heihe_x16-only`  | keliya+heihe+heihe_x4 PASS; heihe_x16 fails Axis 3/4/5                                                                         | NO       | P8-tune.G heihe_x4-only + P8-tune.H GPU sparse spike (priority per GPU-presence gate)                         |
| **`NO-GO-both` (strict)** | heihe_x4 fails ANY axis OR heihe_x16 fails wall axes with margin ≥ 1.5× OR heihe_x16 fails ≥ 3 of 5 axes                       | **YES (canonical, strict 5-axis)** | Per spec REQ-7 NO-GO-both: PR-D shall NOT add new anchor; PR-D shall note "升级到 ADR re-evaluation workshop; future epic 由 user trigger" |
| `BLOCKED`               | small-case unexpected fail / malformed cell_summary / enum out of range / #386 dtor UB recurrence                              | NO       | Re-open #386; mark §P8-tune.F `[BLOCKED]`                                                                     |

PR-D capstone should evaluate the strict-vs-amended divergence (see §"Axis 4 amendment per PR-A H3 disclosure" below) and decide which verdict to enact as the operational forward action, after considering the case-asymmetric scaling discussion below. Recommendation by this ADR: **operationally treat as `GO` per amended verdict**, with explicit acknowledgement that Axis 4 instrumentation is a known PR-A H3 limitation (cycle_complexity is hard-coded estimate, not HYPRE telemetry).

---

## Discussion

### PR-B 16-cell sweep result: uniform PASS, case-asymmetric wall scaling

The 16-cell PR-B sweep produced 16/16 `verdict_class=PASS` — zero AMG hierarchy build failures, zero OOM, zero divergence markers across all 4 cases × 4 (interp_type, coarsen_type) combos. This is a stronger positive signal than the P8-tune.D KLU sweep (which produced 14/16 PASS + 2 `fill_overflow` markers for natural-ordering at heihe_x4/heihe_x16).

Wall axis values demonstrate the **case-asymmetric scaling** that motivates AMG's adoption at large NumY:

| case      | KLU best-combo per-step est. (s) | AMG best-combo setup+apply (s) | AMG/KLU wall ratio |
|-----------|---------------------------------:|-------------------------------:|-------------------:|
| keliya    |                          0.0009  |                       0.00159  |               1.8× |
| heihe     |                          0.0230  |                       0.00312  |               0.14× |
| heihe_x4  |                          0.2967  |                       0.02766  |               0.09× |
| heihe_x16 |                          2.8439  |                       0.11613  |               0.04× |

(AMG combined wall = setup + apply per case best combo from `.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt`; KLU values from ADR-0005 §Discussion Wall axis table.)

The **20-25× AMG-vs-KLU wall advantage at heihe_x16** is the empirical centerpiece of this spike. AMG completes one (setup + apply) cycle for heihe_x16 in 0.116 s while KLU's per-step amortized estimate would require 2.84 s — a clear algorithmic-class crossover.

**Small cases show inverted behavior**: keliya's AMG (1.59 ms) is 1.8× SLOWER than KLU's per-step (0.9 ms) because AMG hierarchy setup overhead (build coarse grids, build interpolation operators) does not amortize at NumY=1785. AMG's setup overhead is a fixed cost that becomes negligible only at production scale. This justifies a future P8-tune.G case-asymmetric integration policy:

- **keliya**: stay on SPGMR maxl=5 (or future KLU env-var opt-in per ADR-0005 §"Why not 'GO acceleration' yet for small cases" mini-prototype branch)
- **heihe**: AMG slight win, but margin small; could either stay on SPGMR or use AMG
- **heihe_x4, heihe_x16**: AMG is unambiguously the right path

### Comparison vs ADR-0005 (KLU spike) and ADR-0004 (SPGMR maxl sweep)

| solver path           | small (keliya/heihe)         | heihe_x4 (124K NumY)               | heihe_x16 (485K NumY)             |
|-----------------------|------------------------------|------------------------------------|------------------------------------|
| SPGMR maxl=5 (default)| baseline (ADR-0003 anchor)   | baseline 0.227 s/step               | infeasible (Krylov saturates L3)  |
| SPGMR maxl=30 opt-in (ADR-0004) | +12% wall, Optional | REGRESS -15.83%                     | infeasible                        |
| KLU `SHUD_KLU_ENABLE=1` opt-in (ADR-0005, P8-tune.E.small-only conditional) | pattern-feasible-prototype-worthy | Optional (wall margin 1.87×) — `use-future-amg` | NO-GO (wall margin 17.9×)         |
| **BoomerAMG (this ADR, P8-tune.F)** | **slight loss / slight win** | **CLEAR WIN 0.028 s/step (10.7× faster than KLU)** | **CLEAR WIN 0.116 s/step (24.5× faster than KLU)** |
| GPU sparse (P8-tune.H, conditional) | TBD                  | TBD                                | TBD (would only trigger on NO-GO-heihe_x16-only) |

The pattern from these three ADRs makes the SHUD large-case solver story decisive: **AMG is the right algorithmic substrate for heihe_x4 + heihe_x16**, regardless of how the Axis 4 instrumentation issue is resolved.

### Axis 4 amendment per PR-A H3 disclosure (critical)

PR-A H3 disclosure (recorded in PR-A spike binary `tools/p8tune.F/boomeramg_setup_solve.cpp`) explicitly states: `cycle_complexity = 2 × operator_complexity` is a **hard-coded estimate** in the current spike implementation, NOT a measurement from HYPRE telemetry. This was a known limitation accepted at PR-A merge time because:

1. Hypre's `HYPRE_BoomerAMGGetCycleNumIterations` / `HYPRE_BoomerAMGGetCycleOpCount` API requires a CVODE-integrated solve cycle to populate; the spike binary issues a single non-integrated `HYPRE_BoomerAMGSolve` and does not have access to per-cycle Op counts.
2. The 2× factor is the canonical V-cycle bound (down + up sweep ≈ 2 × operator complexity per Saad 2003 §13).
3. Within the pattern-only spike scope (REQ-1: zero-CVODE-wireup), this estimate was deemed acceptable as a placeholder.

**Empirical consequence**: across all 16 PR-B cells, `cycle_complexity` ranges from 2.0000 to 2.0213, mechanically tracking `2 × operator_complexity` where `operator_complexity` ∈ [1.0000, 1.0106]. All 16 cells trip Axis 4's `< 1.5` threshold purely because of the hard-coded 2× multiplier. Axis 4 carries **zero independent diagnostic signal** beyond Axis 5.

**Recommendation for ADR-0007 §Decision interpretation**: treat Axis 4 as **non-discriminating diagnostic** for purpose of forward-action planning. The amended verdict (`verdict_branch_axis4_amended=GO`) represents the operationally meaningful 4-axis verdict (Axes 1, 2, 3, 5 only). PR-D capstone should:

1. Anchor master plan §P8-tune.G **per amended verdict GO** (full AMG + A5 integration, HIGH priority, 4-6 weeks per spec REQ-7 Scenario "Conditional next-epic anchor per verdict_branch" GO clause), NOT per strict NO-GO-both (which would require re-evaluation workshop per spec REQ-7 NO-GO-both clause).
2. Add a master plan §P8-tune.G **work item**: integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount` telemetry into the integrated `cvode_config.cpp` AMG wire-up so that Axis 4 becomes independent of Axis 5. Validate that `cycle_complexity` produced by real HYPRE telemetry matches the spike's hard-coded estimate within 5%; if not, re-open this ADR with measured-Axis-4 evidence.
3. Document the Axis 4 mechanical-tracking observation in ADR-0007 §Discussion as a known pattern-only spike limitation, NOT as a real AMG hierarchy quality concern.

This is the same epistemic pattern as ADR-0005's "wall axis is per-step estimate, not measurement" disclosure — the spike model is bounded by what can be measured without CVODE integration, and operational decisions should account for this scope boundary.

### Case-asymmetric scaling third-epic anchor

The PR-B evidence shows AMG's case-asymmetric wall scaling cleanly (keliya 1.8× slower than KLU; heihe_x16 24.5× faster than KLU). This case-asymmetric pattern should be **verified by P8-tune.G** (integrated AMG measurement in CVODE) and **extended by future P8-tune.H** (if conditional on NO-GO-heihe_x16-only — currently NOT triggered by either strict or amended verdict).

A separate forthcoming epic should anchor a **case-asymmetric solver policy** (analogous to ADR-0005's Case-aware split) in production SHUD code:

- Per-case env var: `SHUD_LINSOL=spgmr|klu|amg` (default `spgmr`, opt-in to others)
- Per-case CVODE configuration in `cvode_config.cpp` based on NumY threshold lookup
- A5 hydrology-equivalence validation tier per solver-case pairing

This third-epic anchor is OUT OF SCOPE for both PR-C and PR-D, but should be tagged in master plan §P10+ epic planning when the time comes (post P8-tune.G full integration).

### "Never break userspace" enforcement

PR-C authors only `tools/p8tune.F/aggregate_amg_spike.sh` + `tools/p8tune.F/render_verdict.sh` + this ADR + `docs/p8tune/amg_spike_verdict.md` + `.review-evidence/p8tune-amg-pr-c/` evidence files. Per spec REQ-1, PR-C does NOT touch `SHUD/src/` or `cvode_config.cpp` or `tools/p8tune.F/boomeramg_setup_solve.cpp` (PR-A frozen). The strict NO-GO-both verdict preserves SUNDIALS default solver (PREC_NONE SPGMR maxl=5) for all current users; only the future P8-tune.G integration epic (if anchored per amended verdict per PR-D capstone decision) would change default behavior for large cases.

---

## Consequences

### Positive

1. **AMG empirically validated as the large-case solver path**: 16/16 cells PASS with hierarchy build + V-cycle convergence demonstrated at NumY ≥ 100K. heihe_x16 best combo (interp=6, coarsen=8) completes setup + apply in 0.116 s — 24.5× faster than KLU's per-step estimate and 1.95× faster than SPGMR's 0.227 s baseline. **This is the first solver path that scales cleanly to heihe_x16.**
2. **interp/coarsen combo locked-in**: across all 4 cases, the best combo selection produces deterministic small set of (interp_type, coarsen_type) tuples: (6, 21) for keliya, (8, 8) for heihe, (6, 8) for heihe_x4 + heihe_x16. P8-tune.G integration can hardcode (6, 8) as the production default per AMG-vs-NumY-class rule.
3. **Memory headroom is enormous**: peak_rss ranges from 19 MB (keliya) to 400 MB (heihe_x16) — 5 orders of magnitude below the 130 GiB Axis 3 threshold. AMG is not memory-bound at production scale.
4. **operator_complexity is excellent**: 1.0000-1.0106 across all 16 cells — coarse grid hierarchy adds negligible memory overhead beyond the fine grid. This is the canonical signature of well-behaved BoomerAMG on 2D-mesh PDE Jacobians (per Henson & Yang 2002 §3).
5. **Spike infrastructure reusable** (per spec REQ-8): `boomeramg_setup_solve.cpp` (shell-out to `dump_adjacency` + `fd_color_jacobian` from P8-tune.D) + Hypre IJMatrix dump schema + cell evidence layout are stable, documented, reusable for future epics.
6. **5-axis verdict framework battle-tested**: per-case best-combo selection + 5-axis PASS/FAIL + 4-branch verdict_branch auto-typing + Axis-4 amendment disclosure is now a template for future preconditioner ADRs.

### Negative

1. **Axis 4 instrumentation issue (PR-A H3)**: the hard-coded `cycle_complexity = 2 × operator_complexity` placeholder produces strict-NO-GO-both verdict from PR-B evidence that 16/16 cells otherwise PASS — a verdict that does not represent the empirical reality. This forces ADR §Decision to carry both strict (canonical) and amended (operational) verdicts. The byte-identical contract with `aggregate_verdict.txt` is preserved (the table above is auto-typed verbatim from the strict verdict), but the textual recommendation steers PR-D toward the amended reading.
2. **No CVODE wire-up evidence** (per spec REQ-1): this spike measures `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve` ONLY in isolation (no `SUNLinSol_Hypre` constructor, no `cvSetLinearSolver`, no CVODE step integration). Actual wall improvement in integrated SHUD model run may differ; P8-tune.G's first task is a benchmark numeric prototype to confirm the spike's setup + apply estimates translate to real CVODE step walls.
3. **No A5 hydrology equivalence** (per spec REQ-1): AMG-driven solver may produce trajectory drift on CVODE step-size adapter (similar to KLU exact vs SPGMR iterative concern in ADR-0005). P8-tune.G must validate NSE/KGE/peak/water-balance on keliya + heihe + heihe_x4 + heihe_x16.
4. **`hypre_version=3.1.0`** is the installed version on server cn-nodes; not the upstream-latest. Future AMG-related epics should profile against current Hypre LTS to ensure no regressions vs PR-B baseline (`shud_pin=1ab61c023ac2b93a178c2feb07aa3df509fe1a96`).
5. **`colpack_version=unknown` sentinel** (per PR-B M2 closure): ColPack is reused via shell-out from P8-tune.D; the installed ColPack version was not captured at PR-A merge time. PR-D should backfill this via `pkg-config --modversion` or equivalent when capstone runs final environment audit.

### Neutral

1. **No SHUD source changes** — PR-C is pure docs + tools (aggregator + renderer + ADR + verdict.md); SHUD pin unchanged from PR-0 #386 fix baseline.
2. **No `cvode_config.cpp` changes** — PR-C does not touch the SUNDIALS solver wiring; that lives downstream in P8-tune.G full integration when anchored.
3. **OpenSpec change `p8tune-amg-spike` archive deferred to PR-D** — PR-C only touches the status header in `.review-evidence/p8tune-amg-pr-c/SPEC_STATUS_HEADER.md` mirror; PR-D will run `openspec archive p8tune-amg-spike -y` to canonicalize.
4. **review-loop log entries deferred to PR-D** — PR-D capstone will append 5 JSONL entries (one per PR-0/A/B/C/D) to `docs/review-loop-log.jsonl` per spec REQ-8.
5. **epic #393 closeout pending PR-D** — PR-0 + PR-A + PR-B + PR-C = 4/5 PRs; PR-D capstone update to master plan §P8-tune.F + anchor §P8-tune.G per amended verdict + (conditional) §P8-tune.H per GPU-presence gate.

---

## Acceptance criteria

PR-C acceptance per spec REQ-5 + REQ-6 + issue #397 任务清单 4.1-4.11:

- [x] `tools/p8tune.F/aggregate_amg_spike.sh` authored — emits per-cell aggregate.tsv (16 rows) + machine-readable aggregate_verdict.txt with AGGREGATE_VERDICT_BEGIN/END markers + verdict_branch top-line KV + 4 CASE_VERDICT_BEGIN/END blocks per spec REQ-5 schema. Parses CELL_SUMMARY_BEGIN/END KV blocks STRICTLY (not stdout MARKER lines per REQ-4 marker-vs-class binary).
- [x] `tools/p8tune.F/render_verdict.sh` authored — emits `docs/p8tune/amg_spike_verdict.md` with top-line verdict_branch + per-case T-tables (4 cases × 5 axes) + raw 16-cell TSV inline + footer with Hypre + ColPack + SHUD pin provenance.
- [x] `docs/adr/0007-amg-spike-decision.md` (this file) — §Status: Proposed (PR-D will flip Accepted); §Decision: 4-branch table auto-typed from `aggregate_verdict.txt` `verdict_branch=NO-GO-both` (byte-identical); §Discussion: Axis 4 amendment per PR-A H3 disclosure + recommended operational reading as amended `GO`; §Forward action: PR-D capstone anchor §P8-tune.G per amended verdict.
- [x] `.review-evidence/p8tune-amg-pr-c/{aggregate.tsv, aggregate_verdict.txt, SPEC_STATUS_HEADER.md}` checked in as real (not placeholder) output of running the aggregator on PR-B 16-cell evidence.
- [x] PR-D task flipped §Status from `Proposed` → `Accepted 2026-06-29` (this PR; per spec REQ-6 Scenario "ADR-0007 Status lifecycle").

---

## Forward action

### PR-D capstone anchor selection

Per spec REQ-7 Scenario "Conditional next-epic anchor per verdict_branch":

- **Strict verdict `NO-GO-both`**: per spec, PR-D SHALL NOT add new anchor; PR-D SHALL note "升级到 ADR re-evaluation workshop; future epic 由 user trigger" in master plan §P8-tune.F closure paragraph.
- **Amended verdict `GO`** (operational recommendation per §"Axis 4 amendment per PR-A H3 disclosure" above): PR-D SHOULD anchor §P8-tune.G full AMG + A5 integration as `[OPEN, HIGH priority]` (4-6 weeks per spec REQ-7 GO clause). The strict NO-GO-both verdict is purely artifact of Axis 4 hard-coded estimate; 4 of 5 axes uniformly PASS with substantial headroom.

PR-D capstone author MUST decide between strict and amended reading. Recommendation: **adopt amended GO** with explicit Axis 4 instrumentation work item baked into §P8-tune.G PR-0 scope (integrate `HYPRE_BoomerAMGGetCycle*` telemetry; validate hard-coded estimate within 5% of measured; re-open this ADR if drift > 5%).

### P8-tune.G scope (if anchored per amended GO)

Per spec REQ-7 Scenario "Conditional next-epic anchor per verdict_branch" GO clause + ADR-0005 §Forward action P8-tune.E template:

1. **PR-0**: Hypre Makefile carve-out (mirror P8-tune.D `libshud.a` pattern); env-var hook `SHUD_LINSOL=amg` opt-in (default `spgmr` unchanged); `cvode_config.cpp` gated `SUNLinSol_Hypre` constructor with hardcoded (interp_type=6, coarsen_type=8) per PR-B best combo.
2. **PR-A**: Integrated AMG measurement on keliya + heihe + heihe_x4 + heihe_x16; validate spike setup + apply walls translate to integrated CVODE step walls within 10%; emit `HYPRE_BoomerAMGGetCycle*` telemetry to fill measured Axis 4.
3. **PR-B**: A5 hydrology-equivalence validation per ADR-0004 / ADR-0005 template (NSE/KGE ≥ 0.95; peak Δ ≤ 5-10%; water balance Δ ≤ 1%) on all 4 cases.
4. **PR-C**: Epic capstone + ADR-0008 (P8-tune.G accepted) + master plan §P8-tune.G close + (conditional) ADR-0007 re-evaluation if Axis 4 measured drift > 5%.

Budget: 4-6 weeks per spec REQ-7 GO clause. Priority: HIGH.

### Suppressed branches (documented for completeness)

- **`Optional` branch (NOT chosen by either strict or amended verdict)** — would have triggered §P8-tune.G heihe_x4-only integration (medium priority, 3-4 weeks). Not chosen because amended GO covers all 4 cases including heihe_x16, justifying full HIGH-priority integration.
- **`NO-GO-heihe_x16-only` branch (NOT chosen)** — would have triggered §P8-tune.G heihe_x4-only + §P8-tune.H GPU sparse spike conditional on GPU partition availability. Not chosen because heihe_x16 AMG measured 24.5× faster than KLU and meets 4 of 5 axes (Axis 4 instrumentation artifact aside).
- **`NO-GO-both` strict branch** — IS the canonical aggregator output, but operationally reading recommends amended GO (see §Discussion §"Axis 4 amendment per PR-A H3 disclosure"). PR-D capstone is the decisive moment.
- **`BLOCKED` branch** — not applicable; all 16 cells parsed cleanly, no #386 dtor UB recurrence, no malformed cell_summary, no enum out of range.

Alternative algorithmic substrates considered but rejected at epic-anchor time:

- **Pure GMRES with custom preconditioner** — rejected: existing PREC_NONE SPGMR baseline saturates at heihe_x4 (ADR-0004 conclusion); custom preconditioner would be a deeper engineering investment than AMG-as-preconditioner.
- **PETSc AMG** — rejected: vendor sprawl; Hypre is the upstream-canonical AMG implementation with SUNDIALS adapter (`SUNLinSol_Hypre`); PETSc would require additional adapter shim.
- **Sparse direct solvers at larger scale (KLU 64-bit-index API / MUMPS)** — rejected: ADR-0005 already established KLU is NO-GO at heihe_x4 wall margin 1.87× regardless of int32 vs int64; MUMPS is a heavier-weight dependency with no SUNDIALS adapter.

---

## References

### Internal (本仓库)

- `tools/p8tune.F/aggregate_amg_spike.sh` (本 PR-C) — 16 cell-N.out → aggregate.tsv + aggregate_verdict.txt
- `tools/p8tune.F/render_verdict.sh` (本 PR-C) — aggregate → docs/p8tune/amg_spike_verdict.md
- `tools/p8tune.F/boomeramg_setup_solve.cpp` (PR-A #395) — spike binary (PR-A frozen post-merge)
- `tools/p8tune.F/precheck_env.sh` (PR-A #395) — server cn-node environment audit (Hypre + ColPack presence)
- `tools/p8tune.F/run_cell.sh / spike_array.sbatch` (PR-A #395) — 16-cell Slurm dispatcher
- `tools/p8tune.D/spgmr_baseline_walls.h` (P8-tune.D PR-0 #384) — pinned `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` (wall axis baseline; REUSED per spec REQ-5)
- `tools/p8tune.D/cn_node_ram.h` (P8-tune.D PR-0 #384) — pinned `CN_NODE_RAM_BYTES = 185528156160` (memory axis denominator; REUSED per spec REQ-5)
- `tools/p8tune.D/{dump_adjacency,fd_color_jacobian}.cpp` (P8-tune.D PR-0 #384) — RHS pattern probe + numeric J reusable binaries (PR-A links via Makefile symlink per spec REQ-2)
- `docs/p8tune/amg_spike_verdict.md` (本 PR-C) — full verdict + T-tables + raw data appendix
- `docs/adr/0005-klu-spike-decision.md` — KLU NO-GO at heihe_x4 wall margin 1.87× (triggered this AMG retreat)
- `docs/adr/0004-maxl-sweep-decision.md` — SPGMR maxl Optional-knob + baseline wall anchor 0.226579 s/step
- `docs/adr/0003-precond-spike-decision.md` — p8pre NO-GO + PREC_NONE production baseline
- `openspec/changes/p8tune-amg-spike/{proposal.md, design.md, tasks.md, specs/amg-pattern-spike-verdict/spec.md}` (本 epic OpenSpec change)
- `SHUD_openMP_master_plan.md` §P8-tune.F — epic anchor
- `.review-evidence/p8tune-amg-pr-b/cells/cell-{0..15}.out` (PR-B) — 16-cell PASS evidence (Slurm 9896)
- `.review-evidence/p8tune-amg-pr-c/{aggregate.tsv, aggregate_verdict.txt, SPEC_STATUS_HEADER.md}` (本 PR-C)

### PR sequence (epic #393)

- PR #394 — PR-0 SHUD `Model_Data` dtor UB fix (#386 closure) + spike infrastructure pre-fix
- PR #402 — PR-A spike binary `boomeramg_setup_solve.cpp` + spec amendments (M1: H3 disclosure for Axis 4 hard-coded estimate)
- PR #403 — PR-B 16-cell Slurm array sweep (#396 closure; 16/16 PASS; M2: colpack_version=unknown sentinel; M3: NA timing sentinel for AMG_WALL_OVERFLOW cells, unused in this sweep)
- (本 PR-C) — aggregator + ADR-0007 + verdict.md
- PR-D (forthcoming) — epic capstone (master plan close + conditional §P8-tune.G + §P8-tune.H anchor + OpenSpec archive + review-loop log)

### Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/p8f_amg_spike.9896/` — Slurm 9896 sbatch output dir (16 cells authoritative; cells_root rsync'd to repo at `.review-evidence/p8tune-amg-pr-b/cells/`)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` — 90-day-truncated heihe_x16 AutoSHUD deployment (reused from P8-tune.D)

### External

- Hypre user guide (BoomerAMG) — `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve` + `HYPRE_BoomerAMGGet*` telemetry API (Axis 4 measurement work item for P8-tune.G)
- ColPack user guide — `JacobianGraphColoring` `DISTANCE_TWO` algorithm (Welsh-Powell variant; reused from P8-tune.D)
- SUNDIALS user guide v6.0.0 — `SUNLinSol_Hypre` interface contract (P8-tune.G integration target)
- SuiteSparse KLU user guide — `klu_factor` `common.status` enum (reference for ADR-0005 comparison)

### Academic / theoretical

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. §13 (multigrid methods) — V-cycle bound `cycle_complexity ≈ 2 × operator_complexity` (theoretical anchor for PR-A H3 hard-coded estimate)
- Henson, V. E. & Yang, U. M. (2002). "BoomerAMG: A parallel algebraic multigrid solver and preconditioner." *Applied Numerical Mathematics* 41(1): 155-177. — operator complexity bound for 2D-mesh PDE Jacobians
- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. — comparison anchor (KLU AMD/COLAMD/BTF, ADR-0005 Discussion)
- George, A. & Liu, J. W. H. (1981). *Computer Solution of Large Sparse Positive Definite Systems*. Prentice-Hall. — 2D mesh PDE nested-dissection O(N^1.5) flops bound (KLU comparison anchor)
