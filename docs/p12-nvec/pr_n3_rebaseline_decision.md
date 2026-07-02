# P12-nvec PR-N3 — Config E2 re-baseline decision + E2-vs-E wall measurement

```yaml
epic: P12-nvec
doc: pr_n3_rebaseline_decision
date: 2026-07-02
pr: PR-N3 (#445)
decision: RE-BASELINE ACCEPTED — Config E2 becomes the golden for the E2 lineage
shud_commit: f78e031            # p12-nvec (G-E4 evidence at ce4bcef; Note-1 abort fix + neutrality re-verify)
a3a_scope: A3a does NOT apply ACROSS the reduction-order change (C/E -> E2); applies WITHIN the E2 lineage thereafter
```

## Decision

Config E2 (fixed-tree deterministic reductions, `SHUD_NVEC_DETRED=1`, block
B=4096, Neumaier=0) passed the full G-E4 acceptance chain
(`pr_n3_ge4_verdict.md`): cross-thread bitwise (keliya production-B +
forced-small-B + heihe_x4 cross-N) → A4 ulp report → full A5 tightened
(nse=1.0000 / kge=0.9999 vs Config C, PASS). Therefore:

**A new golden is recorded for the Config E2 lineage.** Config E2 introduces a
ONE-TIME summation-order shift vs Config E / Config C (the cross-block binary
tree replaces the plain serial fold in the summation reductions), so its outputs
are a NEW reference, distinct from the Config C/E golden — certified once by A5,
not re-derived from the C/E golden.

This is a re-baseline of the SAME KIND as the P9 "mode-C-tune" precedent
(ADR-0009): a deliberate, gated trajectory change whose new reference is
established by hydrology-acceptance (A5), after which strict bitwise discipline
resumes WITHIN the new lineage.

### A3a applicability scoping (the load-bearing rule)

- **A3a does NOT apply ACROSS the reduction-order change.** Config E2 outputs are
  NOT bitwise-equal to Config C / Config E and MUST NOT be gated against the C/E
  golden with an A3a bitwise check. The A4 report documents the expected
  divergence; A5 certifies it. (Legacy §P9 discipline: never conflate the
  order-shift with a parallel bug — which is why G-E4 runs the cross-thread
  bitwise gate BEFORE the A4/A5 comparison.)
- **A3a DOES apply WITHIN the E2 lineage thereafter.** Config E2 is
  cross-thread-bitwise BY CONSTRUCTION (summation order is a pure function of NY
  and B, independent of thread count). Any future change on the E2 path is held
  to A3a bitwise-vs-the-E2-golden at the applicable thread counts — G-E4(1)
  proved E2 reproduces itself across N∈{1,2,4,8} (keliya, equal B) and N∈{8,16}
  (heihe_x4), so the E2 golden is a well-defined bitwise target.

### Scope of this decision

Records ONLY the E2-lineage golden decision (spec: "the re-baseline decision only
records THIS lineage decision"). It does NOT promote Config E2 to the release tag,
does NOT flip the production default, and does NOT execute the SHUD
`p12-nvec → openmp-baseline` merge-back — those are post-epic / capstone actions
per the PR-N3 out-of-scope boundary. Config E2 ships as an opt-in build leg:
`make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1 SHUD_NVEC_DETRED=1`.

## E2-vs-E wall measurement (heihe_x4 90-day, 3-run median, server)

Slurm 三铁律: sbatch from `/scratch`, `--output/--error` on `/scratch`,
bin+scripts on `/scratch`; each cell on its OWN exclusive node; NON-profile wall
runs; `N` = single cfg-`NUM_OPENMP` knob = `OMP_NUM_THREADS` (drives both StrictOMP
RHS threads and the OpenMP-NVector thread count); `SHUD_RHS_THREADS` unset. Wall
metric = runner-measured `date +%s.%N` delta around the `shud_omp` invocation
(model-process wall, excludes Slurm scheduling) — identical metric to PR-N2.
SHUD ce4bcef. Evidence: `.review-evidence/p12-nvec/pr-n3/server_matrix/`.

<!-- MATRIX_TABLE_START -->
| cell | 3 rep walls (s) | median wall (s) | nst | nfe | ncfn | ncfl | netf |
|---|---|---|---|---|---|---|---|
| E_n8   | 600.641 / 594.360 / 594.426 | **594.426** | 6575 | 6741 | 51 | 3620 | 0 |
| E_n16  | 496.970 / 493.611 / 496.338 | **496.338** | 6575 | 6741 | 51 | 3620 | 0 |
| E2_n8  | 435.305 / 431.689 / 429.214 | **431.689** | 6574 | 6728 | 50 | 3688 | 0 |
| E2_n16 | 361.641 / 365.530 / 362.653 | **362.653** | 6574 | 6728 | 50 | 3688 | 0 |
<!-- MATRIX_TABLE_END -->

- Config E counters (nst=6575/…/ncfl=3620) match PR-N2 Config C/E (bitwise, rivqdown
  `b5e4b0a2…`); Config E2 counters (nst=6574/…/ncfl=3688, rivqdown `f70209c4…`)
  reflect the certified summation-order shift. All within-cell reps SHA-stable.

### Wall gains + projection-vs-measured delta (informational)

<!-- WALL_START -->
- E2-vs-E wall gain @ N=8  = 594.426 / 431.689 = **1.3770×**
- E2-vs-E wall gain @ N=16 = 496.338 / 362.653 = **1.3686×**
- G-E3(iii) projection (additional wall over Config E @ N=16) = **1.3744×**
- measured @ N=16 = 1.3686× → delta **−0.42%** vs projection (measured slightly
  under the Amdahl projection; the projection assumed the full serial-reduction
  time parallelizes ideally — a −0.4% shortfall is well within tree-combine +
  fork/join overhead expectations)
<!-- WALL_END -->

Context (informational): E2_n16 vs the PR-N2 Config C reference (C_n16 median
694.343 s) = **1.9146×** wall — Config E2 recovers essentially all of the
reduction parallelism the Tier-1 serial overrides gave up, restoring Config-C-era
scaling while keeping cross-thread determinism.

### Node homogeneity

The 4 wall cells + the A5-pair generator ran on exclusive nodes cn10 / cn11 /
cn12 / cn14 / cn22 (cell→node: E_n8=cn22, E_n16=cn10, E2_n8=cn11, E2_n16=cn12,
a5pair=cn14). All five are in the PR-N2 homogeneity pool proven identical SKU
(Intel Xeon Gold 6133, 2×20 cores @ 2500 MHz ±<0.01%) by PR-N2 job 11344 —
`.review-evidence/p12-nvec/pr-n2/node_homogeneity.txt` (cn10/11/12/14/22 all
explicitly captured there). The cross-node E2/E wall ratios are therefore not
hardware-confounded, consistent with the PR-N2 discipline.

## References

- G-E4 verdict: `docs/p12-nvec/pr_n3_ge4_verdict.md`
- Spec: `openspec/changes/p12-nvec/specs/tier2-det-reduction/spec.md`
  (scenario "A5 certifies the new reference": re-baseline + mode-C-tune A3a scoping)
- Design: `design.md §D4` (re-baseline semantics + projection-vs-measured)
- Precedent: ADR-0009 (P9 mode-C-tune re-baseline); PR-N2 `tier1_verdict.md`
- Node pool: `.review-evidence/p12-nvec/pr-n2/node_homogeneity.txt`
- Evidence: `.review-evidence/p12-nvec/pr-n3/server_matrix/`
- SHUD commit: `f78e031` (p12-nvec; G-E4 evidence generated at `ce4bcef`, neutrality re-verified — see evidence README addendum)
