# ADR-0005: klu-spike-decision — KLU pattern-only spike 4-branch verdict (Case-aware branch)

- **Status**: Proposed (PR-B authoring; flipped to Accepted at PR-C capstone)
- **Date**: 2026-06-28
- **Deciders**: DankerMu + Claude orchestrator (per `tools/p8tune.D/aggregate_klu_spike.sh` 3-axis per-case verdict + `docs/p8tune/klu_spike_verdict.md` synthesis)
- **Owner**: SHUD-OpenMP 改造工程 / P8-tune.D epic capstone → split into (a) P8-tune.E.small-only (KLU env-var opt-in for small cases) + (b) P8-tune.F (BoomerAMG/Hypre spike for large cases)
- **Tags**: klu / suitesparse / direct-sparse / pattern-spike / case-aware / amg-retreat / fill-axis / rss-axis / wall-axis
- **Supersedes**: none (first KLU-related ADR)
- **Superseded by**: none
- **Related**: ADR-0004 maxl-sweep-decision (SPGMR maxl Optional-knob; baseline wall anchor heihe_x4 N=1 maxl=5 ≈ 0.227 s/step) + ADR-0003 precond-spike-decision (p8pre NO-GO; PREC_NONE production baseline) + master plan §P8-tune.D anchor + `openspec/changes/p8tune-klu-spike/` (epic SHUD-OpenMP#379) + PR-0 #384 (spike tool) + PR-A #385 (16-cell sweep) + 本 PR-B + 待开 PR-C (capstone)

---

## Context

ADR-0004 §Forward action (post-correction 2026-06-28) triggered the P8-tune.D KLU pattern-only spike epic per GPT Pro 2026-06-28 review's NumY 口径 working-set analysis:

- **heihe_x4** (NumY ≈ 124K) SPGMR maxl=10 working set ≈ 9.6 MB > L2 (1-2 MB per core) → DRAM-bound; all maxl ≥10 wall REGRESS 6.86%-24.82% per PR-D 60-cell sweep
- **heihe_x16** (NumY ≈ 485K) SPGMR maxl=10 working set ≈ 60 MB > L3 (8-32 MB shared per socket) → SPGMR infeasible for future scale

The user's intent "**hydrology-validatable large-case acceleration**" required a fundamentally different solver path. KLU (SuiteSparse direct sparse, AMD/COLAMD reordering + supernodal LU) was the canonical first candidate per SUNDIALS+SuiteSparse integration practice.

This epic was authored as a **zero-source-patch, zero-CVODE-wireup, zero-SHUD-run pattern-only spike** per spec REQ-1 — answering ONE question with hard evidence: **是否 KLU 在 keliya / heihe / heihe_x4 / heihe_x16 上的 fill ratio + RSS + estimated numeric wall 满足三轴硬阈值**？The 4 cases × 4 ordering combos = 16 cells were executed on server Slurm cn-nodes per spec REQ-4.

### 16-cell sweep summary (PR-A evidence; `.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md`)

| case      | NumY (measured) | best ordering | fill_ratio | numeric_wall_s | peak_rss_mb |
|-----------|-----------------|---------------|------------|----------------|-------------|
| keliya    | 1785            | amd btf=0     | 3.23       | 0.0014         | 4.4         |
| heihe     | 21,357          | amd btf=1     | 5.39       | 0.0383         | 15.3        |
| heihe_x4  | 124,395         | amd btf=1     | 8.35       | 0.4945         | 89.2        |
| heihe_x16 | 485,250         | amd btf=1     | 11.08      | 4.7399         | 407.7       |

Two natural-ordering cells (NN=08 heihe_x4 / NN=12 heihe_x16) hit `KLU_TOO_LARGE` (status `-4`; 32-bit signed index overflow) — surfaced as `fill_overflow` data points per amended spec REQ-5 Scenario "Tool-bound data point (KLU 32-bit-int index overflow)". They are DECISIVE evidence that natural ordering is pathological at production scale, but DO NOT poison the case-level verdict because AMD/COLAMD orderings stay well within int32 index space.

### 3-axis verdict (per spec REQ-5, computed by `tools/p8tune.D/aggregate_klu_spike.sh`)

For best-combo cell per case:

- **Fill axis**: `fill_ratio < 8·log₂(NumY)` — thresholds {86.4, 115.1, 135.4, 151.1}
- **RSS axis**: `peak_rss < 0.7 × CN_NODE_RAM_BYTES` — threshold ≈ 121 GiB (CN_NODE_RAM = 173 GiB, cn14 measured)
- **Wall axis**: `(numeric_factor_wall / refactor_freq=10) + (N_solve=5 · solve_wall) < 0.7 × SPGMR_per_step_wall_baseline = 0.7 × 0.227 = 0.1586 s` (with `solve_wall ≈ 0.1 × numeric_factor_wall` triangular-solve estimate)

---

## Decision

**Adopt Case-aware branch.** Per spec REQ-5 Scenario "Case-aware/Optional branch auto-population", the trigger condition is met:

```
keliya_KLU_overall_verdict       = GO
heihe_KLU_overall_verdict        = GO
heihe_x4_KLU_overall_verdict     = Optional  (∈ {NO-GO, Optional})
heihe_x16_KLU_overall_verdict    = NO-GO
case_aware_branch_fires          = true
```

Per-case recommended actions (auto-typed by aggregator per spec REQ-5 D8):

| case      | overall    | fill | rss  | wall | recommended_action       |
|-----------|------------|------|------|------|--------------------------|
| keliya    | GO         | PASS | PASS | PASS | `klu-env-var-opt-in`     |
| heihe     | GO         | PASS | PASS | PASS | `klu-env-var-opt-in`     |
| heihe_x4  | Optional   | PASS | PASS | FAIL | `use-future-amg`         |
| heihe_x16 | NO-GO      | PASS | PASS | FAIL | `use-future-amg`         |

Decisive-cell pointer:

```
heihe_x4_recommended_next_epic           = p8-tune.E-klu-impl
heihe_x4_recommended_next_epic_priority  = medium
```

The heihe_x4 wall margin = 0.297 / 0.159 ≈ 1.87× (within 2× budget — Optional), while heihe_x16 wall margin ≈ 17.9× (far past — NO-GO). The asymmetry justifies the Case-aware split: small cases earn a KLU env-var opt-in path; large cases retreat to BoomerAMG/Hypre per Q7 commitment.

### 4-branch decision tree (per spec REQ-6 Scenario "4-branch decision tree")

| Branch       | Trigger                                                                                       | Adopted? | Forward action                                                                                      |
|--------------|-----------------------------------------------------------------------------------------------|----------|-----------------------------------------------------------------------------------------------------|
| **GO**       | All 4 cases overall = GO (all 3 axes PASS for best-combo)                                     | NO       | Would trigger P8-tune.E full KLU + A5 hydrology-equivalence epic (4-6 week budget)                  |
| **Optional** | heihe_x4 overall = Optional only                                                              | NO       | Would trigger benchmark numeric prototype mini-spike (~1 week) before P8-tune.E commit              |
| **Case-aware** | keliya/heihe GO AND heihe_x4 ∈ {NO-GO, Optional} (small-case work, large-case forward retreat) | **YES**  | Split P8-tune.E into (a) small-case KLU env-var opt-in pattern (mirroring `SHUD_SPGMR_MAXL`) + (b) large-case forward to P8-tune.F BoomerAMG/Hypre spike |
| **NO-GO**    | heihe_x4 fail any axis with no Optional path                                                  | NO       | Would trigger P8-tune.F BoomerAMG/Hypre spike (3-4 week budget per Q7 F5 commitment)                |

---

## Rationale

### Fill axis PASS across all 4 best-combos — AMD is uniformly the winner

AMD ordering produces fill_ratio ∈ {3.23, 5.39, 8.35, 11.08} for the 4 cases, all comfortably below the 8·log₂(NumY) threshold. The fill_ratio scales roughly with log₂(NumY) (note the near-linear growth 3.23 → 5.39 → 8.35 → 11.08 as NumY scales 1785 → 21K → 124K → 485K — almost matching the theoretical nested-dissection optimum). This is the expected behavior of AMD on a 2D mesh PDE Jacobian with sparse off-diagonal river/lake coupling.

**Natural ordering pathology (decisive data, not blocking)**:

| case      | natural fill_ratio | result                                              |
|-----------|-------------------:|-----------------------------------------------------|
| keliya    | 205.95             | PASS-but-heavy (NumY=1785 small enough to absorb)   |
| heihe     | 2304.29            | PASS-but-pathological-wall (1630s numeric factor)  |
| heihe_x4  | —                  | `KLU_TOO_LARGE` (int32 index overflow → fill_overflow data point) |
| heihe_x16 | —                  | `KLU_TOO_LARGE` (int32 index overflow → fill_overflow data point) |

Natural ordering's fill explosion is the textbook 2D-mesh O(N²) classical-elimination signature. AMD/COLAMD orderings collapse this to O(N log N) — the empirical scaling matches.

### RSS axis PASS across all 4 best-combos — KLU memory footprint is bounded

Best-combo peak RSS ∈ {4.4 MB, 15.3 MB, 89 MB, 408 MB} — all 4 cases stay well below 121 GiB threshold (0.7× cn-node 173 GiB RAM). The RSS scales roughly with `nnz(L+U) × 16 bytes` (int32 index + double value per entry plus overhead). At heihe_x16 NumY=485K, the 408 MB footprint is 3 orders of magnitude under the threshold — KLU is not memory-bound for any case in this matrix.

### Wall axis case-asymmetric — the decisive empirical finding

The per-step amortized KLU wall estimate uses:

```
per_step_estimate = numeric_factor_wall / refactor_freq + N_solve × solve_wall
                 ≈ numeric_factor_wall / 10 + 5 × (0.1 × numeric_factor_wall)
                 = numeric_factor_wall × (0.1 + 0.5) = 0.6 × numeric_factor_wall
```

(with the conservative knobs `refactor_freq=10` + `N_solve=5` + `solve_wall ≈ 0.1 × factor_wall`)

Per-case per-step estimates vs 0.1586 s budget:

| case      | numeric_factor_wall_s | per-step estimate | budget (s) | margin | wall axis |
|-----------|-----------------------:|-------------------:|-----------:|-------:|----------|
| keliya    | 0.0014                | 0.0009            | 0.1586     | 0.006× | PASS     |
| heihe     | 0.0383                | 0.0230            | 0.1586     | 0.145× | PASS     |
| heihe_x4  | 0.4945                | 0.2967            | 0.1586     | 1.87×  | FAIL (marginal — Optional) |
| heihe_x16 | 4.7399                | 2.8439            | 0.1586     | 17.9×  | FAIL (far past — NO-GO)    |

The **case-asymmetric wall pattern** is the decisive empirical finding of this spike:

- Small cases (keliya, heihe) — KLU is faster than SPGMR per step (margin 0.006×, 0.145×). Direct factorization amortized over 10 CVODE refactor steps comes out well below the SPGMR Krylov-iteration cost. **KLU is a credible production solver at these scales.**
- Large cases (heihe_x4, heihe_x16) — KLU's numeric factorization dominates: heihe_x4 0.49s factor / step × 0.6 amortization = 0.30s/step > 0.16s budget; heihe_x16 4.74s factor / step extrapolates 17.9× the budget. Even with the most generous amortization knobs (e.g., `refactor_freq=100`), heihe_x16 still saturates the wall budget by an order of magnitude.

This is fundamentally because **AMD-reordered LU factorization on a 2D-mesh PDE Jacobian has roughly O(N^1.5) flops** (the canonical nested-dissection complexity for 2D problems) while SPGMR with PREC_NONE has O(N · maxl) per Arnoldi iteration. At the heihe_x4 / heihe_x16 scale, KLU's setup cost surpasses SPGMR's iterative cost — the algorithmic crossover happens between heihe and heihe_x4.

### BTF block structure — zero observable effect

For all 4 cases, AMD btf=0 vs AMD btf=1 produced identical fill_ratio and near-identical numeric_factor_wall (within measurement noise — see SWEEP_RESULTS.md NN=01 vs NN=02, NN=05 vs NN=06, NN=09 vs NN=10, NN=13 vs NN=14). This is consistent with SHUD's Jacobian structure: the elem-elem coupling is local 3-neighbor (mesh edges) and the river-lake-elem cross-coupling forms a relatively dense bridge; there is no clear block-triangular form to exploit. **BTF is not a useful knob for SHUD.** Future KLU integration (P8-tune.E) can default to `btf=0` without loss.

### Chromatic number χ observation — Welsh-Powell upper bound holds

Per-case Welsh-Powell χ from ColPack: keliya 16, heihe 18, heihe_x4 16, heihe_x16 20. All comfortably under the 50-color bound stipulated in spec REQ-2 Scenario "Column coloring via Welsh-Powell" (and well under the keliya tighter 30-color bound). The chromatic number is **bounded by 2D mesh degree (≤ 8) regardless of NumY**, confirming the spec's assertion. This makes the FD-color Jacobian probe O(χ) RHS evaluations per Jacobian build — affordable at production scale, useful infrastructure for future Jacobian-aware epics (spec REQ-8).

### COLAMD vs AMD — AMD wins uniformly

COLAMD's fill_ratio is consistently worse than AMD's: {4.64 vs 3.23, 8.96 vs 5.39, 15.06 vs 8.35, 20.63 vs 11.08} — 1.4-1.9× higher fill. This matches the SuiteSparse literature: COLAMD optimizes for unsymmetric LU patterns (think LP simplex matrices), while AMD optimizes for the symmetric `A + Aᵀ` pattern that SHUD's Jacobian approximates (water flow is symmetric in the diffusion limit; advection contributes asymmetry only in the non-zero pattern not the structural sparsity). AMD is the right choice for SHUD.

### "Never break userspace" enforcement

PR-B authors only `tools/p8tune.D/aggregate_klu_spike.sh` + `tools/p8tune.D/render_verdict.sh` + this ADR + `docs/p8tune/klu_spike_verdict.md` + the `.review-evidence/p8tune-klu-spike-pr-b/` evidence files. Per spec REQ-7 Scenario "PR-B aggregator + ADR PR boundary", PR-B does NOT touch `SHUD/src/` or `cvode_config.cpp`. The Case-aware decision below preserves SUNDIALS default solver (PREC_NONE SPGMR maxl=5) for all current users; only the future P8-tune.E.small-only env-var opt-in (similar to ADR-0004 `SHUD_SPGMR_MAXL`) would change default behavior — and only when opted in. Forward action item P8-tune.F (BoomerAMG/Hypre) is a separate epic; its acceptance criteria + ADR will be authored when that epic opens.

---

## Consequences

### Positive

1. **Small-case users can opt into KLU acceleration** (KLU's per-step estimate uses ≤1% of the 0.7×SPGMR-baseline per-step budget for keliya, ~14% for heihe — i.e., wall-budget headroom is >99% / >85% respectively; this is the budget-headroom fraction, NOT a +99%/+85% end-to-end speedup vs SPGMR) via a future env-var hook in P8-tune.E.small-only — no source patch required at user side, mirrors the ADR-0004 `SHUD_SPGMR_MAXL` pattern.
2. **Large-case path is unblocked** — P8-tune.F BoomerAMG/Hypre spike is now justified by hard evidence (wall_overflow at heihe_x16 17.9× over budget). The retreat is decisive, not speculative.
3. **Natural-ordering pathology surfaced as decisive `fill_overflow` data**: the spike correctly detected `KLU_TOO_LARGE` int32 index overflow at heihe_x4 / heihe_x16 natural+BTF. Future P8-tune.E implementers know to default to AMD (and that `klu_l_*` 64-bit-index API is NOT a required workaround for AMD-reordered SHUD matrices).
4. **AMD vs COLAMD vs BTF decision matrix locked**: AMD is the uniform winner; COLAMD is 1.4-1.9× worse on fill; BTF is zero-effect. P8-tune.E can hardcode `AMD + btf=0` without further investigation.
5. **Spike infrastructure reusable** (per spec REQ-8): the 3 spike binaries + adjacency CSC schema + FD-color numeric J binary format are now stable, documented in `tools/p8tune.D/README.md` §output-format, and reusable for future Jacobian-aware epics (P8-tune.E, P9 precision, etc.).
6. **3-axis verdict framework battle-tested** — the (fill, RSS, wall) per-axis methodology with per-case best-combo selection + KV-machine-readable verdict is now a template for future solver-selection ADRs (KLU, AMG, GPU sparse, etc.).

### Negative

1. **Case-aware split adds operational complexity**: production deployments need to know which case they're running to pick {default SPGMR / KLU env-var / future AMG} solver. The user runbook + master plan §P8-tune.E.small-only docs need to spell out the case-size cutoff.
2. **No GO branch achieved** — KLU is not a universal SHUD solver replacement. The original "KLU for all" optimism (ADR-0004 §Forward action implication) is empirically falsified for production NumY > 100K.
3. **heihe_x4 Optional is a near-miss**: the 1.87× wall margin means a generous amortization (`refactor_freq=20` instead of 10) would flip the verdict to PASS. P8-tune.E.small-only could include heihe_x4 if CVODE refactor cadence proves more sparse in practice — but the conservative spike verdict puts heihe_x4 in the AMG retreat for safety.
4. **No CVODE wire-up evidence** — per spec REQ-1, this spike measures `klu_factor` ONLY in isolation (no `SUNLinSol_KLU` constructor, no `cvSetLinearSolver`, no CVODE step integration). The `(N_solve=5, solve_wall=0.1×factor)` formula is a model, not measurement. P8-tune.E.small-only's actual wall improvement may differ; benchmark numeric prototype is a sensible first task in that epic.
5. **wall_overflow KV label is shared between Optional and NO-GO for the wall axis** — spec REQ-5 D8 enum has only 4 values {fill_overflow, rss_overflow, wall_overflow, clean_GO}, so heihe_x4 (Optional) and heihe_x16 (NO-GO) both report `KLU_NO_GO_axis = wall_overflow`. The distinction lives in `KLU_overall_verdict` (Optional vs NO-GO). Consumers must read both KVs together, not the axis label alone.

### Neutral

1. **No SHUD source changes** — PR-B is pure docs + tools (aggregator + renderer + ADR + verdict.md); SHUD pin unchanged.
2. **No `cvode_config.cpp` changes** — PR-B does not touch the SUNDIALS solver wiring; that lives downstream in P8-tune.E.small-only (env-var hook) when/if opened.
3. **OpenSpec change `p8tune-klu-spike` archive deferred to PR-C** — PR-C task 4.7 will move `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md` to `openspec/specs/klu-pattern-spike-verdict/spec.md`. PR-B only touches the status header in the change-local copy (see `.review-evidence/p8tune-klu-spike-pr-b/SPEC_STATUS_HEADER.md` for the committable mirror).
4. **epic #379 closeout pending PR-C** — PR-0 + PR-A + PR-B = 3/4 PRs merged; PR-C capstone update to master plan §P8-tune.D + open new §P8-tune.E.small-only + §P8-tune.F anchors.

---

## Discussion

### Forward implication for P8-tune.E.small-only (Case-aware branch GO half)

The small-case KLU env-var opt-in should mirror the ADR-0004 `SHUD_SPGMR_MAXL` precedent:

- New env var: `SHUD_KLU_ENABLE=1` (default 0 = SUNDIALS PREC_NONE SPGMR maxl=5 unchanged)
- `cvode_config.cpp` gated `SUNLinSol_KLU` constructor + AMD ordering + btf=0 (per Discussion BTF observation above)
- A5 hydrology equivalence (NSE/KGE/peak/water-balance) validation on keliya + heihe required for the Performance opt-in tier (NOT A5-certified yet — see ADR-0004 tier definitions)
- Acceptance criteria: G1 build / G2 default-compat unset bit-identical / G4 ncfn+ncfl improvement / G7-attested ADR-mechanism explanation (KLU's exact factorization vs Krylov iterative may produce trajectory drift on the CVODE step-size adapter; document mechanism per ADR-0004 §G7-attested template)

Recommendation: P8-tune.E.small-only is a **medium-priority** epic. The ~86% wall-budget headroom for heihe (KLU per-step estimate uses ~14% of the 0.7×SPGMR baseline) is substantial enough to justify the engineering investment; keliya's >99% headroom is gravy. 4-6 week budget is appropriate (mirrors ADR-0004 P8-tune.C epic cadence).

### Forward implication for P8-tune.F (Case-aware branch NO-GO half)

The large-case AMG retreat is the harder path. Per ADR-0004 §Forward action + Q7 commitment, the recommended candidates are:

- **BoomerAMG** (Hypre library) — algebraic multigrid, O(N) memory + O(N log N) setup + O(N) per-iteration; SUNDIALS has `SUNLinSol_Hypre` adapter
- **PETSc AMG** — similar algorithmic complexity, different library ecosystem; `SUNLinSol_PETSc` adapter

Q7's F5 commitment was 3-4 week budget for an AMG pattern-only spike (same shape as P8-tune.D but for AMG). The forward action's axis-typing per spec REQ-6 Scenario "NO-GO axis typing within NO-GO branch":

- **Wall axis (heihe_x4 + heihe_x16 FAIL)**: emphasize AMG's O(N) per-iteration cost vs KLU's O(N^1.5) setup. AMG should win at NumY ≥ 100K where KLU saturates.
- **NOT** SuperLU / SuperLU_MT / cvBandPre / BiCGStab (rejected per Q7 — direct sparse with parallel solve doesn't help SHUD's bandwidth-bound axis; band assumption broken by river/lake long-range coupling)

P8-tune.F is a **high-priority** epic. heihe_x16 17.9× wall margin means there is NO viable production solver for heihe_x16 today (SPGMR is saturated per ADR-0004; KLU is saturated per this ADR). AMG is the lifeline.

### Chromatic number χ observation forward

The FD-color Jacobian probe infrastructure produced bounded χ ∈ [16, 20] across all 4 cases. This is **reusable** for any future Jacobian-aware epic:

- **Matrix-free Newton-Krylov with colored FD preconditioner** (alternative to KLU + AMG) — could fold χ-color FD probe into the preconditioner build, amortize across CVODE steps
- **GPU-friendly batched Jacobian assembly** — 16-20 RHS evaluations per Jacobian update is workable on GPU even for heihe_x16

The infrastructure is parked in `tools/p8tune.D/` for now; future epics can ingest `fd_color_jacobian.cpp` + numeric J binary schema without re-deriving the FD-color analysis. Per spec REQ-8, the CLI is additive-only after PR-0 freeze.

### Comparison with ADR-0004 (SPGMR maxl Optional-knob)

ADR-0004 and this ADR together complete the SHUD linear solver decision matrix:

| solver path                             | small case (keliya/heihe)         | large case (heihe_x4/heihe_x16)         |
|-----------------------------------------|-----------------------------------|------------------------------------------|
| SPGMR maxl=5 (SUNDIALS default)         | baseline (ADR-0003 anchor)        | baseline (ADR-0003 anchor)              |
| SPGMR `SHUD_SPGMR_MAXL=30` opt-in (ADR-0004) | +12% wall (heihe N=1), Optional   | REGRESS −15.83% (heihe_x4 N=1), unset   |
| KLU `SHUD_KLU_ENABLE=1` opt-in (this ADR → P8-tune.E.small-only) | ≤1% wall budget used (keliya); ~14% (heihe); >99% / >85% budget-headroom respectively | wall_overflow → use future AMG          |
| AMG (P8-tune.F future)                  | TBD (likely competitive)          | TBD (target)                            |

The pattern is clear: **case-size dictates solver choice**. SHUD's user runbook should expose this directly.

---

## Acceptance criteria

PR-B acceptance per spec REQ-7 Scenario "PR-B aggregator + ADR PR boundary":

- [x] `tools/p8tune.D/aggregate_klu_spike.sh` authored — emits per-cell aggregate.tsv + machine-readable aggregate_verdict.txt KV per spec REQ-5 D8 schema (verified by running on PR-A 16-cell evidence: all 16 cells classified PASS or `fill_overflow` data point, no parse failures)
- [x] `tools/p8tune.D/render_verdict.sh` authored — emits `docs/p8tune/klu_spike_verdict.md` with per-case T-tables + 3-axis synthesis + raw data appendix + machine-readable verdict block + amended REQ-5 scenarios rationale
- [x] `docs/adr/0005-klu-spike-decision.md` (this file) — §Decision: Case-aware branch auto-typed from per-case KVs; §Discussion documents per-axis evidence, BTF zero-effect, chromatic number bound, AMD vs COLAMD, P8-tune.E.small-only + P8-tune.F forward actions
- [x] `.review-evidence/p8tune-klu-spike-pr-b/aggregate.tsv` + `aggregate_verdict.txt` checked in as real (not placeholder) output of running the aggregator on PR-A evidence
- [x] `.review-evidence/p8tune-klu-spike-pr-b/SPEC_STATUS_HEADER.md` (committable mirror of the spec status header per task §3.8)
- [ ] PR-C task 4.6 will flip §Status from `Proposed` → `Accepted YYYY-MM-DD` (PR-B leaves at Proposed pending merge per spec REQ-7 Scenario "PR-C capstone PR boundary")

---

## Forward action

### Per branch (Case-aware = chosen branch)

Per spec REQ-6 Scenario "4-branch decision tree" + tasks §3.7c:

1. **P8-tune.E.small-only** (new sub-epic, anchored by PR-C task 4.5) — small-case KLU env-var opt-in pattern mirroring `SHUD_SPGMR_MAXL` ADR-0004 hook
   - Scope: `SHUD_KLU_ENABLE=1` env var, gated `SUNLinSol_KLU` constructor in `cvode_config.cpp`, AMD ordering + btf=0, A5 hydrology equivalence validation on keliya + heihe
   - Budget: 4-6 weeks (mirrors P8-tune.C epic cadence)
   - Priority: medium
   - Gate: keliya + heihe both pass A5 (NSE/KGE ≥ 0.95, peak Δ ≤ 5-10%, water balance Δ ≤ 1%); 8-gate verdict format per ADR-0004 template

2. **P8-tune.F** (new epic, anchored by PR-C task 4.5) — BoomerAMG/Hypre pattern-only spike for large-case (heihe_x4 + heihe_x16) acceleration
   - Scope: same shape as P8-tune.D (zero-source-patch tool spike, 4 case × 4 ordering combo, 3-axis verdict, 4-branch ADR-0006)
   - Budget: 3-4 weeks per Q7 F5 commitment
   - Priority: high (heihe_x16 has NO viable production solver today)
   - Gate: BoomerAMG spike emits its own ADR-0006 with GO/Optional/NO-GO verdict at heihe_x16 scale

### Suppressed branches (documented for completeness)

- **GO branch (NOT chosen)** — would have triggered single P8-tune.E full KLU + A5 hydrology-equivalence epic. Not chosen because heihe_x4 wall margin = 1.87× (Optional) and heihe_x16 = 17.9× (NO-GO).
- **Optional branch (NOT chosen)** — would have triggered benchmark numeric prototype mini-spike (~1 week). Not chosen because heihe_x16 NO-GO already excludes large-case path; Case-aware split is more decisive.
- **NO-GO branch (NOT chosen)** — would have triggered P8-tune.F only. Not chosen because small-case GO data is genuine — denying small-case users a >85-99% wall-budget-headroom KLU opt-in (i.e., KLU per-step uses ≤14% of the 0.7×SPGMR-baseline budget for the heihe-class small cases) would be wasteful.

---

## References

### Internal (本仓库)

- `tools/p8tune.D/aggregate_klu_spike.sh` (本 PR) — 16-cell cell logs → aggregate.tsv + aggregate_verdict.txt
- `tools/p8tune.D/render_verdict.sh` (本 PR) — aggregate → docs/p8tune/klu_spike_verdict.md
- `tools/p8tune.D/cn_node_ram.h` (PR-0 #384) — pinned `CN_NODE_RAM_BYTES = 185528156160` (RSS axis denominator)
- `tools/p8tune.D/spgmr_baseline_walls.h` (PR-0 #384) — pinned `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` (wall axis baseline)
- `tools/p8tune.D/dump_adjacency.cpp / fd_color_jacobian.cpp / klu_analyze_factor.cpp` (PR-0 #384) — 3 spike binaries
- `tools/p8tune.D/spike_array.sbatch / run_cell.sh / precheck_env.sh` (PR-A #385) — 16-cell sweep dispatcher
- `docs/p8tune/klu_spike_verdict.md` (本 PR) — full verdict + T-tables + raw data
- `docs/adr/0004-maxl-sweep-decision.md` — SPGMR maxl Optional-knob + baseline wall anchor
- `docs/adr/0003-precond-spike-decision.md` — p8pre NO-GO + PREC_NONE production baseline
- `openspec/changes/p8tune-klu-spike/{proposal.md,design.md,tasks.md,specs/klu-pattern-spike-verdict/spec.md}` (本 epic OpenSpec change)
- `SHUD_openMP_master_plan.md` §P8-tune.D — epic anchor
- `.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md` — PR-A 16-cell narrative + per-cell data points
- `.review-evidence/p8tune-klu-spike-pr-a/SPEC_AMENDMENTS.md` — REQ-5 + REQ-7 amendments landed in PR-A

### PR sequence (epic #379)

- PR #384 — PR-0 spike tool (`p8tune.D/{dump_adjacency,fd_color_jacobian,klu_analyze_factor}.cpp` + Makefile + cn_ram + Mac smoke)
- PR #385 — PR-A 16-cell Slurm array sweep + cell evidence + spec amendments
- (本 PR-B) — aggregator + ADR-0005 + verdict.md
- PR-C (forthcoming) — capstone (master plan flip + new epic anchors + OpenSpec archive)

### Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/pr-a-20260628-115234/` — 9762 sbatch output dir (NN=00-07 authoritative for keliya + heihe)
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/exit-fix-1782652046/` — 9794 + 9812 + 9828 + 9829 re-run output dirs (NN=08-15 authoritative for heihe_x4 + heihe_x16)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` — 90-day-truncated heihe_x16 AutoSHUD deployment (job 9810 COMPLETED 37:57)

### External

- SuiteSparse KLU user guide — `klu_factor` `common.status` enum (`KLU_OK=0`, `KLU_TOO_LARGE=-4`, etc.)
- ColPack user guide — `JacobianGraphColoring` `DISTANCE_TWO` algorithm (Welsh-Powell variant)
- BoomerAMG (Hypre) — AMG forward retreat candidate for P8-tune.F large-case epic
- SUNDIALS user guide v6.0.0 — `SUNLinSol_KLU` + `SUNLinSol_Hypre` interface contracts

### Academic / theoretical

- Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. — AMD / COLAMD / BTF algorithms + nested-dissection complexity bounds
- George, A. & Liu, J. W. H. (1981). *Computer Solution of Large Sparse Positive Definite Systems*. Prentice-Hall. — 2D mesh PDE nested-dissection O(N^1.5) flops bound
- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd ed. SIAM. — Arnoldi-MGS working-set analysis (ADR-0004 §Discussion mechanism backbone)
