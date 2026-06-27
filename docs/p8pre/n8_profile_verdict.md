---
title: "p8pre-spike Step 1 PR-B — N=8 Mode C profile ROI verdict"
date: 2026-06-27
version: 0.1
status: "branch: a (PROCEED — r_min=1.819 >= 1.5)"
related_docs:
  - openspec/changes/p8pre-spike/proposal.md
  - openspec/changes/p8pre-spike/tasks.md
  - openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md
  - tools/cvode_stats_diff/canonical_15_keys.yaml
  - docs/p1e/p1e_perf_baseline.md
  - docs/p1e/p1e_pr_i_strict_omp_verification.md
  - docs/p8pre/step1_prep.md
  - docs/p8pre/n8_profile_run.md
---

# Abstract

This PR-B verdict aggregates the 18-cell (2 cases × 3 N-values × 3 reps) Mode C
profile artifacts produced by PR-A (issue #341) into per-(case, N) medians of
the SHUD canonical 15-key CVODE stats set, the 7 emitted timer buckets, and
the 2 raw extras. It verifies cross-N invariance (Δ=0 strict) for the 5
counters `nst / nfe / nfeLS / nni / nsetups`, anchors the 4 absolute
baselines (`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 /
heihe_x4.nfe=6741`) imported from P1e PR-I, computes the `nfeLS/nfe`
ROI ratio per (case, N), and emits a single 4-branch verdict letter
(a/b/c/d) per the spec's exhaustive ROI tree. The verdict letter is the
input gate for the downstream PR-C capstone baseline doc (issue #343) and
ADR-0003 draft decision.

## §1 目的

Quantify whether the SHUD CVODE solver's linear-solver work (`nfeLS`)
relative to the nonlinear-solver work (`nfe`) is large enough at N=8 to
make a preconditioner (P8-precond) ROI-positive. The decision criterion is
spec n8-mode-c-profile-recheck Scenario "ROI verdict based on nfeLS/nfe
exhaustive branching" (L72-80):

- branch (a) `r_min ≥ 1.5` → PROCEED Step 2 P8-precond-0 identity spike
- branch (b) `r_max < 1.5` → NO-GO 转 P8-tune (maxl/restart/EpsLin)
- branch (c) `r_min < 1.5 ≤ r_max < 3.0` → case-aware precond design required
- branch (d) `r_min < 1.5 AND r_max ≥ 3.0` → case-aware + HIGH HETEROGENEITY

This document does NOT archive `wall_step1_baseline_median(case, N)` — that
is the PR-C (issue #343) capstone responsibility per task §4.1 and design D5.

## §2 数据来源

- Input: 18-cell rsync mirror at `/tmp/p8pre_n8_profile/` (= rsync mirror of
  `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/<cell>/`).
- PR-A run provenance: Slurm JID range = 9510..9527 (per
  `/tmp/p8pre_n8_profile/jid_table.txt`; full (case, N, rep, JID) table
  reproduced therein).
- SHUD pin: `7a1dc8f` (P1e ship pin, post-nested-Timer-fix), verified
  invariant across all 18 cells by PR-A pre-flight § per spec
  n8-mode-c-profile-recheck Scenario "SHUD pin unchanged across Step 1".
- Build: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`
  (Mode C: Serial NVector + StrictOMP RHS + Timer instrumentation).
- Partition pin: `heihe` → `cn14`, `heihe_x4` → `cn15` (per P1e PR-I
  SOP; keeps cross-cell CPU SKU comparable).

## §3 CVODE stats summary

### §3.1 Per (case, N) median table (15 canonical keys × 6 groups)

Median across 3 reps per cell, per spec L113 (median over 3 reps). All
values cite back to per-cell files at
`/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/cvode_stats.txt`.

| case | N | nfe | nfeLS | nni | nli | nsetups | netf | nst | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|------|---|-----|-------|-----|-----|---------|------|-----|-----|-----|------|------|-------|-------|---------|---------|
| heihe | 1 | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe_x4 | 1 | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |

### §3.2 Cross-N invariance Δ=0 strict (5 keys × 2 cases)

Spec n8-mode-c-profile-recheck Requirement "cross-N stats invariance check"
+ Scenarios L86-104. Each row: `X_N=1 = X_N=4 = X_N=8` (Δ=0 strict). Any
Δ != 0 blocks the verdict.

| case | key | N=1 | N=4 | N=8 | Δ (N=8 - N=1) | verdict |
|------|-----|-----|-----|-----|---------------|---------|
| heihe | nst | 6698 | 6698 | 6698 | 0 | PASS |
| heihe | nfe | 6943 | 6943 | 6943 | 0 | PASS |
| heihe | nfeLS | 12632 | 12632 | 12632 | 0 | PASS |
| heihe | nni | 6942 | 6942 | 6942 | 0 | PASS |
| heihe | nsetups | 0 | 0 | 0 | 0 | PASS |
| heihe_x4 | nst | 6575 | 6575 | 6575 | 0 | PASS |
| heihe_x4 | nfe | 6741 | 6741 | 6741 | 0 | PASS |
| heihe_x4 | nfeLS | 30509 | 30509 | 30509 | 0 | PASS |
| heihe_x4 | nni | 6740 | 6740 | 6740 | 0 | PASS |
| heihe_x4 | nsetups | 0 | 0 | 0 | 0 | PASS |

### §3.3 Absolute baseline anchor check (4 values)

Spec Scenarios L90 + L97 + tasks.md §3.3 absolute baseline anchors. Source
of anchor values: `docs/p1e/p1e_perf_baseline.md` §3.4 nst ladder
(heihe.nst=6698, heihe_x4.nst=6575) + `docs/p1e/p1e_pr_i_strict_omp_verification.md`
§3.1 nfe ladder (heihe.nfe=6943, heihe_x4.nfe=6741). Per invariance N=1=4=8 we
report the N=1 median.

| case | key | expected (P1e PR-I baseline) | actual (this run, N=1 median) | verdict |
|------|-----|------------------------------|-------------------------------|---------|
| heihe | nst | 6698 | 6698 | PASS |
| heihe | nfe | 6943 | 6943 | PASS |
| heihe_x4 | nst | 6575 | 6575 | PASS |
| heihe_x4 | nfe | 6741 | 6741 | PASS |

### §3.4 ROI ratios per (case, N)

- `r = nfeLS / nfe` per spec Scenario "nfeLS/nfe ratio reported per (case, N)" L63-70 (3 decimal places).
- `nli / nni` reported per spec L68. For SUNDIALS CVLS SPGMR with the
  default finite-difference Jacobian-vector product proxy, every GMRES
  inner iteration triggers exactly one RHS callback (counted into
  `nfeLS`), so the expected identity is `nfeLS = nli` (NOT
  `nfeLS = nfe + nli`). The last column `|nfeLS − nli|` is the
  sanity check; non-zero would indicate a non-FD-Jvp configuration or a
  counter ABI mismatch worth a follow-up.

| case | N | nfe_median | nfeLS_median | r = nfeLS/nfe | nni_median | nli_median | nli/nni | nfeLS − nli |
|------|---|------------|--------------|---------------|------------|------------|---------|-------------|
| heihe | 1 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe | 4 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe | 8 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe_x4 | 1 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |
| heihe_x4 | 4 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |
| heihe_x4 | 8 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |

### §3.5 Verdict header (branch letter + 4-branch tree application)

**branch: a (PROCEED — r_min=1.819 >= 1.5)**

Decision trace:
- `r_heihe_N=8` = 1.819
- `r_heihe_x4_N=8` = 4.526
- `r_min` = 1.819 ; `r_max` = 4.526
- Reason: r_min=1.819 >= 1.5

## §4 Timer bucket summary (7 buckets per (case, N))

Per spec Requirement "bucket breakdown using emitted timer buckets" L47-57.
`t_RHS_kernel` is EXCLUDED from the bucket-sum invariant because it is a
sub-bucket of `t_RHS_total` (containment relation per timer.cpp
`emit_extra` L176-184 derivation chain). The `t_wall_total` column is
the raw top-level wall and is the reference for the bucket-sum invariant
`t_RHS_total + t_CVODE_internal + t_forcing_io + t_ET + t_output + t_other
≈ t_wall_total` (±2% rounding + Timer overhead allowance per PR-A task
§2.3 acceptance check). Values are seconds, 9 decimal places per
`timer.cpp` `%.9f` format.

| case | N | t_RHS_kernel | t_RHS_total | t_CVODE_internal | t_forcing_io | t_ET | t_output | t_other | t_wall_total |
|------|---|--------------|-------------|------------------|--------------|------|----------|---------|--------------|
| heihe | 1 | 59.297105186 | 59.309465391 | 34.132480255 | 16.737010072 | 1.269606031 | 0.455737311 | 28.890957966 | 140.796860508 |
| heihe | 4 | 14.564996315 | 14.576597726 | 34.010878790 | 16.727701576 | 1.292510362 | 0.515063681 | 28.620022084 | 95.734229317 |
| heihe | 8 | 7.660043460 | 7.671164040 | 33.999476308 | 16.767080799 | 1.276299979 | 0.575878541 | 29.404578194 | 89.731664529 |
| heihe_x4 | 1 | 800.307229114 | 800.349873201 | 338.595027921 | 99.695411845 | 8.393876669 | 8.903034123 | 156.958198120 | 1412.895421879 |
| heihe_x4 | 4 | 226.466319826 | 226.514689354 | 343.939403780 | 102.545528716 | 8.734118574 | 9.025451759 | 158.157705906 | 849.704022841 |
| heihe_x4 | 8 | 119.911758937 | 119.963581421 | 344.555806220 | 102.507393472 | 8.688975633 | 8.841229155 | 159.076389259 | 743.552324144 |

## §5 ROI 论述

Branch (a) PROCEED. `r_min` = 1.819 ≥ 1.5, satisfying the spec's PROCEED
threshold uniformly across both cases at N=8. SUNDIALS `nfeLS` accumulates
RHS callbacks invoked by the Jacobian-vector-product proxy inside the GMRES
inner loop; an `r` of ~1.819–4.526 means every nonlinear
`nfe` evaluation is amplified by 1.819× to 4.526× through Krylov
iterations. For a preconditioner that reduces the average GMRES iteration
count by a factor `κ`, the expected Step 2 wall-time speedup is bounded
below by `(1 + (r − 1)/κ) / r` of the linear-solver portion, multiplied
by the linear-solver share of `t_CVODE_internal`. The spike binary
(PR-D §6.1-6.6) only needs to (i) keep the new linear-solver path bitwise
identical to the baseline (gate 2 `ncfn = 0` per spec) and (ii) prove the
preconditioner registration plumbing fires (gate 3 `nps > 0` AND
`npe > 0`). The expected Step 2 speedup window for a downstream non-trivial
preconditioner (e.g. element-block Jacobi) is therefore on the order of
10–25% of `t_CVODE_internal` per cell, depending on the `κ` achieved
and the linear-solver share of solver time visible in §4 above.

## §6 Limitations & threats to validity

- **90-day case truncation** per project rule (CLAUDE.md "所有 case ≤90 天截断"):
  both `heihe` and `heihe_x4` runs are truncated to 90 model-days, not the
  full 4-year forcing window. This is sufficient for OpenMP/CVODE stats
  signal but may underrepresent steady-state Krylov behavior for cases with
  long spin-up transients. Decision affected: §3.4 `r` ratios are
  representative of warm-up + steady-state mix, not pure steady-state.
- **2-case scope** (heihe @ 6335 NumEle + heihe_x4 @ 40046 NumEle): branches
  (c) / (d) "case-aware" terminology refers only to this pair; a third or
  fourth case (e.g. ksge / qhh / xinanjiang) MAY shift `r_min` or
  `r_max` outside the lattice cell observed here. ADR-0003 SHALL note the
  2-case sampling as a deferred-broadening hazard.
- **Single SHUD pin** `7a1dc8f` (P1e ship pin): the `r` ratio is sensitive
  to the CVLS `maxl` default (per SUNDIALS 6.0.0
  `CVodeSetMaxLSetup` semantics); upstream changes to the preconditioner
  registration default in CVLS would shift this baseline. Pinning the SHUD
  rev guards against silent drift between PR-B and PR-C.
- **Cross-N median over 3 reps**: 3 reps is sufficient for integer-valued
  CVODE counters where determinism makes Δ=0 the expected result, but for
  the float timer buckets (§4) it is below the noise floor for 5% wall
  comparisons. Step 2 PR-F gate 4 (5%/10% wall non-regression) inherits this
  noise floor.

## §7 引用

- spec: `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` (Requirements + Scenarios L3-122)
- tasks: `openspec/changes/p8pre-spike/tasks.md` §3 PR-B (L28-37)
- canonical keys: `tools/cvode_stats_diff/canonical_15_keys.yaml` (15-key + REJECT typo list)
- timer.cpp: `tools/profile/timer.cpp` L150-200 (7-bucket emit + extras shape)
- PR-A run doc: `docs/p8pre/n8_profile_run.md`
- PR-A jid_table: `/tmp/p8pre_n8_profile/jid_table.txt`
- p1e baseline anchors:
  - `docs/p1e/p1e_perf_baseline.md` §3.4 (heihe.nst=6698, heihe_x4.nst=6575)
  - `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 (heihe.nfe=6943, heihe_x4.nfe=6741)
- SUNDIALS CVODE 6.0.0 `cvode.h` (CVLS Krylov `nfeLS` semantics)
- aggregator script: `tools/p8pre/aggregate_n8_profile.sh`
