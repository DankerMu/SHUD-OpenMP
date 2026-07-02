# P12-nvec Tier-1 verdict (G-E2) + Tier-2 gate decision (G-E3)

```yaml
epic: P12-nvec
doc: tier1_verdict
date: 2026-07-02
pr: PR-N2 (#444)
matrix: heihe_x4 90-day, {Config C, Config E} × N∈{1,8,16}, 3-run median wall
tier1_verdict: TIER1_ADOPT   # G-E2
tier2_verdict: TIER2_GO      # G-E3
```

## Scope

This doc records the G-E2 Tier-1 ROI verdict and the G-E3 Tier-2 gate decision
from the PR-N2 server scaling matrix. All thresholds and their measurement
cells are pinned in `openspec/changes/p12-nvec/specs/tier1-scaling-verdict/spec.md`
and `docs/p12-nvec/spike_brief.md §Kill/ROI gates`; nothing is decided ad hoc
here. Raw matrix evidence: `.review-evidence/p12-nvec/pr-n2/`.

## Matrix (3-run medians)

Wall metric: the runner-measured wall around the `shud_omp` invocation
(`date +%s.%N` delta, i.e. model process wall excluding Slurm scheduling),
consistent across all 18 timed runs; `SHUD_RHS_THREADS` unset in every cell;
`N` = single cfg-`NUM_OPENMP` knob = `OMP_NUM_THREADS` (drives both StrictOMP
RHS threads and Config E's `N_VNew_OpenMP` NVector thread count). NON-profile
builds for the 18 wall runs.

<!-- MATRIX_TABLE_START -->
| cell | 3 rep walls (s) | median wall (s) | nst | nfe | ncfn | ncfl | netf |
|---|---|---|---|---|---|---|---|
| C_n1  | 1290.05 / 1288.95 / 1279.34 | **1288.953** | 6575 | 6741 | 51 | 3620 | 0 |
| C_n8  | 724.33 / 723.97 / 728.66    | **724.325**  | 6575 | 6741 | 51 | 3620 | 0 |
| C_n16 | 686.28 / 694.34 / 696.70    | **694.343**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n1  | 893.07 / 886.71 / 890.50    | **890.502**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n8  | 553.19 / 554.15 / 552.93    | **553.194**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n16 | 500.91 / 490.37 / 491.62    | **491.618**  | 6575 | 6741 | 51 | 3620 | 0 |
<!-- MATRIX_TABLE_END -->

Config C sp@8 = C_n1/C_n8 = 1.780×, sp@16 = C_n1/C_n16 = 1.856× (Amdahl
ceiling, cf RELEASE.md sp@8=1.804/sp@16=1.946 — same regime). Config E sp@8 =
E_n1/E_n8 = 1.610×, sp@16 = E_n1/E_n16 = 1.812×. All 18 wall runs rc==0.

## Bitwise cross-check (equal-N: C@N vs E@N)

Rule: at each N∈{1,8,16}, Config C and Config E must produce identical
cvode_stats counters (nst/nfe/ncfn/ncfl/netf) AND equal SHA of the
`heihe_x4.rivqdown.dat` output.

<!-- BITWISE_TABLE_START -->
| N | counters equal (nst/nfe/ncfn/ncfl/netf) | rivqdown SHA equal | verdict |
|---|---|---|---|
| 1  | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
| 8  | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
| 16 | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
<!-- BITWISE_TABLE_END -->

**Bitwise gate PASS at every N.** G-E1 (proven on keliya in PR-N1) transfers to
production x4 scale: Config E is byte-identical to Config C at N∈{1,8,16}.

Bonus (cross-N determinism, expected identical per Tier-1, recorded as evidence
not a gate): within each config the `rivqdown` SHA and cvode_stats counters are
identical across N∈{1,8,16} (distinct SHA count = 1, distinct counter-tuple
count = 1 for both C and E). A single SHA `b5e4b0a2…` covers all 18 wall runs +
the profile leg (19 runs).

## G-E2 verdict (TIER1_ADOPT / TIER1_CLOSE)

**Rule (verbatim, pinned — bar NOT moved):**

> TIER1_ADOPT iff speedup(E/C) ≥ 1.10 at N=8 or N=16 AND bitwise PASS, else TIER1_CLOSE

Measured ratios:

<!-- GE2_START -->
- speedup(E/C) @ N=8  = 724.325 / 553.194 = **1.3094×**  (≥ 1.10)
- speedup(E/C) @ N=16 = 694.343 / 491.618 = **1.4124×**  (≥ 1.10)
- bitwise PASS = **YES** (identical counters + rivqdown SHA at N∈{1,8,16})
<!-- GE2_END -->

**VERDICT: TIER1_ADOPT** — Config E adopted as an opt-in build leg
(`make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1`); release-tag
promotion deferred to a post-epic follow-up (release line cpu-accel-v1.0.x
untouched). Justification for the 1.10× bar (vs P9's 1.5×): Tier 1 is
trajectory-IDENTICAL (zero re-qualification, zero acceptance-criteria change,
~150 LOC), so a 10% wall win at production scale clears its cost (design D3).

keliya small-NY wall (informational only, per design R2 — keliya NY≈1.5k is a
correctness case below the OpenMP-NVector viability threshold, Config E is
opt-in and targeted at NY ≥ 100k): cited from the PR-N1 G-E1 legs; not run as a
perf cell here.

## G-E3 Tier-2 gate decision

**GO iff ALL three inputs pass** (each input's measurement cell pinned):

**(i) Tier-1 verdict = TIER1_ADOPT?**
→ TIER1_ADOPT (this doc, G-E2). **PASS.**

**(ii) PR-N0 reduction-op share ≥ 10% (threshold ≥ 10% on either — verbatim
either/OR aggregation: heihe_x4 OR heihe_x16 share ≥ 10% ⇒ (ii) PASS):**
- heihe_x4 Config C N=16 reduction share = **35.677%** (PR-N0 §5c, pinned)
- heihe_x16 Config C N=16 reduction share = **30.461%** (PR-N0 §5d, pinned)
- Both clear 10% (the gate needs only one). **PASS.**

**(iii) Amdahl projection ≥ 1.15× additional wall over Config E at N=16.**

Formula (verbatim, pinned):

> wall_E16 / (wall_E16 − t_red·(1 − 1/16))

Pinned inputs:
- `t_red` = **142,859,623,101 ns (142.860 s)** = PR-N0 heihe_x4 Config C N=16
  absolute reduction total_ns (PR-N0 §5c / §7). Serial reduction time is
  N-invariant and transfers to Config E under the bitwise-identical trajectory;
  cross-checked by this PR's Config E N=16 gate-on profile leg (see below).
- `wall_E16` = **491.618 s** = this matrix's Config E median wall at N=16.

<!-- GE3III_START -->
Projection = 491.618 / (491.618 − 142.860·(1 − 1/16)) = 491.618 / (491.618 − 133.931) = 491.618 / 357.687 = **1.3744×**  (≥ 1.15×). **PASS.**
<!-- GE3III_END -->

**G-E3 VERDICT: TIER2_GO** — all three inputs pass. PR-N3 (fixed-tree
deterministic reductions, `SHUD_NVEC_DETRED`, Config E2) blocked-by-G-E3 marker
is lifted; see issue #445.

### Config E N=16 gate-on profile leg — t_red cross-check (G-E3(iii))

Config E build WITH `SHUD_ENABLE_PROFILE=1`, run with `SHUD_NVEC_PROF=1`, N=16,
single run (bitwise-identical trajectory to E_n16: same rivqdown SHA + cvode
counters). Its reduction total_ns should be close to PR-N0's `t_red`
(142.860 s) since the trajectory is byte-identical. Share table:
`.review-evidence/p12-nvec/pr-n2/profile_leg/`.

<!-- PROFILE_LEG_START -->
- Config E N=16 reduction total_ns = **143,366,605,392 ns (143.367 s)**
  (`backend=hybrid`, NY=124395, nthreads=16, t_CVODE_raw=217.133 s; same 276,977
  reduction calls as Config C — bitwise trajectory)
- vs PR-N0 Config C N=16 t_red = 142.860 s → delta **+0.355%** (cross-check
  PASS — serial reduction time is N-invariant and transfers to Config E)
- Config E N=16 share of t_CVODE_raw: reduction **66.027%**, elementwise
  **8.710%** (elementwise dropped from Config C's 49.926% because Config E
  parallelizes it; reductions now dominate the smaller CVODE bucket — exactly
  the Tier-2 target). `derive_shares.py` output archived in evidence.
<!-- PROFILE_LEG_END -->

## Consequence executed

- G-E3 = **TIER2_GO** → issue #445 (PR-N3) blocked-by marker lifted: the
  `blocked` label removed + a gate-values comment posted (this PR is the sole
  owner of the GO/NO-GO consequence; the capstone task 5.1 only verifies). PR-N3
  fixed-tree deterministic reductions (`SHUD_NVEC_DETRED`, Config E2) may
  proceed under its own G-E4 acceptance chain.

## Verdict summary

**G-E2 = TIER1_ADOPT ; G-E3 = TIER2_GO.** Config E adopted as an opt-in build
leg; Tier-2 unblocked. Reproduce: `uv run analyze_matrix.py markers.txt results`
under `.review-evidence/p12-nvec/pr-n2/` (SUMMARY line: `TIER1_ADOPT ; TIER2_GO`).

## References

- Spec: `openspec/changes/p12-nvec/specs/tier1-scaling-verdict/spec.md`
- Design: `openspec/changes/p12-nvec/design.md` §D3 (G-E2 + 1.10× bar) + §D4
  (G-E3 三输入 + formula + t_red N-invariance)
- Spike brief: `docs/p12-nvec/spike_brief.md §Kill/ROI gates`
- PR-N0 evidence (G-E3 inputs): `.review-evidence/p12-nvec/pr-n0/README.md`
- PR-N1 evidence (G-E1 bitwise + Config E): `.review-evidence/p12-nvec/pr-n1/`
- This PR's matrix evidence: `.review-evidence/p12-nvec/pr-n2/`
- ADR-0011: `docs/adr/0011-p12-nvec-tier1-verdict-and-tier2-gate.md`
