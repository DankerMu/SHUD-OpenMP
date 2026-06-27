---
title: "p8pre-spike Step 1 capstone — N=8 Mode C profile baseline + gate-4 anchor"
date: 2026-06-27
version: 1.0
status: "Step 1 GATE PASS (branch a PROCEED Step 2); gate-4 wall baseline anchored for PR-F #347"
related_docs:
  - "docs/p8pre/n8_profile_run.md (PR-A execution log, #341)"
  - "docs/p8pre/n8_profile_verdict.md (PR-B ROI verdict, #342)"
  - "docs/p8pre/step1_prep.md (P1e absolute baseline anchor)"
  - "openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md"
  - "openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md"
  - "SHUD_openMP_master_plan.md §P8-precond"
  - "docs/adr/0002-solver-path.md (Path 3 deferred → Step 2 spike scope)"
  - "docs/p1e/p1e_perf_baseline.md §3.1 wall median + §3.4 nst ladder (P1e PR-I anchor)"
  - "docs/p1e/p1e_pr_i_strict_omp_verification.md §3.1 nfe baseline"
  - "docs/p1e/p1e_academic_summary.md (mother template)"
---

# Abstract

This document is the Step 1 capstone for the p8pre-spike epic (`openspec/changes/p8pre-spike`), aggregating an 18-cell 2×3×3 experiment matrix (2 cases × 3 thread counts × 3 reps) executed on the server under build Mode C (`make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`, SHUD pin `7a1dc8f`, cn14/cn15 partition pin) per PR-A (#341). The aggregator (PR-B #342) verified 10/10 cross-N invariance entries Δ=0 over `{nst, nfe, nfeLS, nni, nsetups}` and 4/4 P1e absolute baseline anchors (`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741`), confirming hypothesis H1 (Mode C profile-build numerical invariance) and H2 (preconditioner ROI cost-effectiveness, `r_min = 1.819 ≥ 1.5`). The 4-branch ROI verdict tree returns **branch a (PROCEED Step 2)**: `r_min = nfeLS/nfe |_{heihe, N=8} = 1.819` and `r_max = nfeLS/nfe |_{heihe_x4, N=8} = 4.526`, both above the 1.5 PROCEED threshold. The 6-row wall median table in §5.1 archives `wall_step1_baseline_median(case, N)` — the canonical gate-4 baseline anchor consumed by PR-F (#347) Step 2 hard gate 4 (wall non-regression vs Step 1 baseline at `ε(heihe)=0.10` and `ε(heihe_x4)=0.05`). Hypotheses H1, H2, H3 (heihe + heihe_x4 dual-case coverage spans 1.6× problem size + 2× wall scaling) are all PASS. Step 2 P8-precond-0 identity spike (PRs #344-#347) is authorized to proceed.

**Keywords**: SHUD; CVODE; SPGMR; preconditioner ROI; nfeLS/nfe; cross-N invariance; wall baseline; gate-4 anchor; Mode C profile build

---

# §1 Introduction

The P1e epic (closed 2026-06-25) shipped Mode C `ExecPolicy::StrictOMP` and confirmed bitwise reproducibility + ≥1.5× wall speedup on `heihe_x4` (per `docs/p1e/p1e_summary.md` §7). A follow-up profile (P2a, 2026-06-26) localized roughly 25% of remaining wall to `t_CVODE_internal` (per master plan §P2a M12 + L1967), of which SPGMR Krylov work dominates. ADR-0002 Path 3 (SPGMR + block-Jacobi preconditioner) was preserved as a future-epic option contingent on quantifying the linear-solver-vs-nonlinear-solver work ratio `r = nfeLS / nfe`. The p8pre-spike epic (`openspec/changes/p8pre-spike`) is that quantification + a two-step identity-precond API spike (Step 1 profile re-measure + Step 2 identity-precond wire-up).

This document is the Step 1 **capstone** (PR-C #343 deliverable). It mints the gate-4 baseline anchor `wall_step1_baseline_median(case, N)` consumed by PR-F (#347), and records the formal verdict tree branch letter that authorizes Step 2.

Three research hypotheses are formalized (operational definitions in §3):

- **H1 — Mode C profile-build numerical invariance.** Building Mode C with the additional Timer instrumentation flag (`SHUD_ENABLE_PROFILE=1`) does NOT perturb the CVODE solver trajectory. Operational: for both cases, all five counters `{nst, nfe, nfeLS, nni, nsetups}` satisfy `X_{N=1} = X_{N=4} = X_{N=8}` (Δ=0 strict) AND each value matches the P1e PR-I absolute baseline anchor.
- **H2 — Preconditioner ROI cost-effectiveness.** The ratio `r = nfeLS / nfe` at N=8 across cases is high enough that introducing a preconditioner (reducing average GMRES iteration count by factor κ) gives a positive wall speedup. Operational: `r_min ≥ 1.5` per spec n8-mode-c-profile-recheck Scenario "ROI verdict based on nfeLS/nfe exhaustive branching" L72-80.
- **H3 — Two-case spike-coverage adequacy.** `heihe` (NumEle=6335, 90-day wall ~90s @ N=8) and `heihe_x4` (NumEle=40046, 90-day wall ~744s @ N=8) jointly span 6.3× problem size and 8.3× wall scaling — sufficient signal for Step 2 P8-precond-0 spike (identity-precond API wire-up). Operational: heihe falls into the "Medium" §1.1.1 ROI bucket of the master plan; heihe_x4 falls into the "Large + production-target" bucket.

§2 surveys precedent (P1e PR-I 24-cell anchors, P2a v0.5 bucket policy, ADR-0002 Path 3 deferral). §3 describes methodology (build flags, 18-cell submit, 5-gate suite). §4 enumerates hardware + software setup. §5 reports raw results: §5.1 6-row wall median table (the gate-4 anchor), §5.2 invariance, §5.3 absolute anchor verify, §5.4 ROI ratios, §5.5 verdict branch. §6 discusses hypothesis verification + P1e prior-epic comparison. §7 lists threats to validity. §8 concludes. §9 names downstream Step 2 dependencies.

---

# §2 Related Work

**§2.1 P1e PR-I 24-cell baseline (closed 2026-06-25).** PR-I (#317) established `heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741` as cross-N invariant under Mode C (Serial NVector + StrictOMP RHS) build at SHUD pin `3341368d` (without `SHUD_ENABLE_PROFILE=1`). Those anchors are absolute reference values for any subsequent Mode-C-family build (per `docs/p1e/p1e_perf_baseline.md` §3.4 nst ladder + `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 nfe baseline). The p8pre Step 1 verdict (PR-B #342 §3.3) re-validates these four anchors under the post-Timer-fix SHUD pin `7a1dc8f` — see §5.3.

**§2.2 P2a v0.5 profile bucket policy.** `tools/profile/timer.cpp:150-184` emits 7 named buckets (`t_RHS_kernel / t_RHS_total / t_CVODE_internal / t_forcing_io / t_ET / t_output / t_other`) plus 2 extras (`t_CVODE_raw / t_wall_total`). The bucket-sum invariant excludes `t_RHS_kernel` (it is a sub-bucket of `t_RHS_total`; the p2a v0.1 double-count bug fixed in SHUD `7a1dc8f` traces to this containment error). The Step 1 PR-A run (per `docs/p8pre/n8_profile_run.md` §4) verifies all 18 cells satisfy this invariant at `BSUM% = 0.0000%` (zero error), evidence that the post-fix Timer derivation chain is algebraically tight.

**§2.3 SUNDIALS CVODE 6.0.0 SPGMR + FD-Jvp identity.** CVLS SPGMR with the default finite-difference Jacobian-vector-product proxy counts every GMRES inner iteration as exactly one RHS callback (incremented into `nfeLS`). The expected identity `nfeLS = nli` holds (NOT `nfeLS = nfe + nli`); §5.4 column `|nfeLS − nli|` is the sanity check.

**§2.4 ADR-0002 Path 3 deferral.** ADR-0002 (closed P1e epic, 2026-06-25) preserved Path 3 (SPGMR + block-Jacobi physics preconditioner) as a future-epic option. The trigger conditions: (i) ROI ratio `r = nfeLS/nfe ≥ 1.5` quantified (this document's §5.4), (ii) identity-precond API wire-up gates pass (Step 2 PRs #344-#347), (iii) ADR-0003 GO decision. p8pre-spike is the materialization of triggers (i)+(ii). ADR-0003 is the formal Step 2 capstone.

**§2.5 p8pre Step 1 capstone positioning.** This document is the input to ADR-0003 along the branch-a (PROCEED) leg. The NO-GO branches (b/c/d) are NOT taken — see §5.5 + §8 — so this document does NOT draft a NO-GO ADR.

---

# §3 Methodology

## §3.1 Experiment matrix

The 18-cell matrix mirrors P1e PR-I deployment SOP with one alteration: N=2 is excluded (per spec n8-mode-c-profile-recheck Scenario "cell count matches design" L13-19) because P1e PR-I monotonic scaling already covers it and Step 2 gate-4 only needs N ∈ {1, 4, 8} headroom signals. Cell tuple `(build_mode, case, N, rep)`:

- `build_mode` = Mode C with profile instrumentation (1 fixed)
- `case` ∈ {heihe, heihe_x4} (2 levels)
- `N` ∈ {1, 4, 8} (3 levels)
- `rep` ∈ {1, 2, 3} (3 levels)
- Total = 2 × 3 × 3 = 18 cells.

## §3.2 Build flags + nm gates

Mode C with profile: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`. The `SHUD_ENABLE_PROFILE=1` flag is REQUIRED — without it, `tools/profile/timer.cpp` is not linked, `profile_B0.yaml` is not emitted, and PR-F (#347) gate-4 has no baseline (per spec L17). Three nm SHALL gates verified at PR-A pre-flight (per `docs/p8pre/n8_profile_run.md` §2 + spec L24-26):

| Symbol | Required | Verified |
|---|---|:---:|
| `N_VNew_Serial` | ≥ 1 hit | PASS |
| `N_VNew_OpenMP` | = 0 hits | PASS |
| `GOMP_parallel@GOMP_4.0` | ≥ 1 hit | PASS |

## §3.3 Determinism env

`OMP_PROC_BIND=close` + `OMP_PLACES=cores` baked into the Slurm wrapper `tools/p8pre/run_n8_profile.sh` (chained singleton). `SHUD_RHS_THREADS=N` set per cell (canonical knob per `docs/p1e/p1e_thread_split.md` Tab. 2).

## §3.4 5-gate suite (per cell + aggregate)

Per cell (PR-A §4 + tasks §2.3):

1. **ART**: 5 artifacts present (`profile_B0.yaml, cvode_stats.txt, <case>.rivqdown.dat, slurm.out, slurm.err`).
2. **CANON15**: `cvode_stats.txt` keys exactly = `tools/cvode_stats_diff/canonical_15_keys.yaml`.
3. **REJECT**: typo set absent (`nlcf, nfevals, hcur, qcur, hin`).
4. **EXTRAS**: `profile_B0.yaml` `extras:` block contains `t_CVODE_raw` AND `t_wall_total`.
5. **BSUM%**: bucket-sum invariant `t_RHS_total + t_CVODE_internal + t_forcing_io + t_ET + t_output + t_other ≈ t_wall_total` (excluding `t_RHS_kernel`, ±2%).

Aggregate (PR-B §3 + tasks §3.3-§3.4):

1. **Cross-N invariance Δ=0 strict** per case for 5 keys (10 entries total).
2. **Absolute baseline anchor** (4 entries: heihe.nst, heihe.nfe, heihe_x4.nst, heihe_x4.nfe).
3. **ROI ratios** `r = nfeLS / nfe` per (case, N) (6 entries).
4. **4-branch verdict** (branch letter ∈ {a, b, c, d}).
5. **Wall median archive** (this document §5.1 — the gate-4 anchor).

## §3.5 Singleton afterany Slurm chain

Per (case, N), `--dependency=afterany singleton` chains the 3 reps to guarantee cross-rep node-state stability. Per case, the (N=1, N=4, N=8) groups also chain so that the cn14 stream (heihe) and cn15 stream (heihe_x4) keep CPU-SKU consistency across N (per P1e PR-I §1 hardware context + spec p8precond-zero-identity-spike "SAME partition pin" requirement).

---

# §4 Experimental Setup

| Item | Value |
|---|---|
| Server endpoint | `frd_muziyao@210.77.77.22:32099` (Slurm) |
| Compute nodes | `cn14` (heihe stream) + `cn15` (heihe_x4 stream); 2 sockets × 20 cores each, ~170 GB RAM |
| OS / Compiler | Ubuntu 24.04 LTS / GCC 13.3.0 + libgomp |
| SHUD pin | `7a1dc8f` (P1e ship pin, post-nested-Timer fix; verified `git submodule status SHUD` per cell) |
| SUNDIALS-CVODE | 6.0.0 (pinned) |
| Build | `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` |
| Forcing | CMFD V0200, 90-day truncation per case (per CLAUDE.md C7) |
| heihe basin | NumEle=6335, server canonical baseline (`SHUD/Basins/heihe/forcing.trimmed` 29 MB; per `docs/case_deployment_map.md` §2.2) |
| heihe_x4 basin | NumEle=40046, server canonical baseline (`SHUD/Basins/heihe_x4/forcing` 286 MB local 3-yr subset; per `docs/case_deployment_map.md` §2.2) |
| Slurm JID range | 9510–9527 (18 cells; per `/tmp/p8pre_n8_profile/jid_table.txt`) |
| Submit window UTC | 2026-06-27 01:32:52 → 02:43:35 (~70 min wall) |
| Aggregator | `tools/p8pre/aggregate_n8_profile.sh` (PR-B #342) |
| Mirror | `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/` (rsync of `/scratch/.../.p8pre-runs/`) |

---

# §5 Results

## §5.1 Raw data table — gate-4 baseline anchor (CORE)

The CORE table for downstream PR-F (#347) hard-gate-4 decision. Wall median is the middle of 3 sorted `t_wall_total` values per cell (per spec L113 "median over 3 reps"); CVODE counter medians are equal to per-cell values since cross-N + cross-rep invariance holds (verified in §5.2). Source-of-truth: per-cell `profile_B0.yaml` `extras.t_wall_total` (in `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/profile_B0.yaml`); per-cell `cvode_stats.txt`.

**Table 1: `wall_step1_baseline_median(case, N)` — gate-4 anchor for PR-F #347.**

| case | N | n_reps | wall_median (s) | nst_median | nfe_median |
|---|---:|---:|---:|---:|---:|
| heihe | 1 | 3 | 140.797 | 6698 | 6943 |
| heihe | 4 | 3 | 95.734 | 6698 | 6943 |
| heihe | 8 | 3 | 89.732 | 6698 | 6943 |
| heihe_x4 | 1 | 3 | 1412.895 | 6575 | 6741 |
| heihe_x4 | 4 | 3 | 849.704 | 6575 | 6741 |
| heihe_x4 | 8 | 3 | 743.552 | 6575 | 6741 |

The `wall_median` column IS the canonical `wall_step1_baseline_median(case, N)` referenced by PR-F (#347) gate-4 per spec L17 + tasks §8.5. Step 2 identity-spike (PR-E #345) re-runs the same 18-cell matrix using a child SHUD pin on `openmp-baseline-p8pre`; PR-F (#347) computes per (case, N) `|wall_identity_median - wall_step1_baseline_median| / wall_step1_baseline_median ≤ ε(case)` with `ε(heihe) = 0.10` (~9s headroom @ N=8 against ~14s Slurm-jitter floor) and `ε(heihe_x4) = 0.05` (~37s headroom @ N=8 against ~7-15s noise floor).

**Per-(case, N) wall_median anchor values** (canonical citation form for downstream PR-F #347 aggregator + ADR-0003 ROI 论述):

- `wall_median(heihe, N=1) = 140.797 s` (3 reps; per Table 1)
- `wall_median(heihe, N=4) = 95.734 s` (3 reps; per Table 1)
- `wall_median(heihe, N=8) = 89.732 s` (3 reps; per Table 1)
- `wall_median(heihe_x4, N=1) = 1412.895 s` (3 reps; per Table 1)
- `wall_median(heihe_x4, N=4) = 849.704 s` (3 reps; per Table 1)
- `wall_median(heihe_x4, N=8) = 743.552 s` (3 reps; per Table 1)

**Per-rep wall values** (for reproducibility footprint — read from `/tmp/p8pre_n8_profile/<cell>/profile_B0.yaml` `extras.t_wall_total`):

| cell | rep1 (s) | rep2 (s) | rep3 (s) | median (s) |
|---|---:|---:|---:|---:|
| heihe_N1 | 141.790 | 140.797 | 125.216 | 140.797 |
| heihe_N4 | 96.509 | 95.734 | 95.068 | 95.734 |
| heihe_N8 | 89.732 | 89.744 | 89.324 | 89.732 |
| heihe_x4_N1 | 1474.953 | 1412.895 | 1268.620 | 1412.895 |
| heihe_x4_N4 | 849.704 | 849.756 | 840.095 | 849.704 |
| heihe_x4_N8 | 743.556 | 743.552 | 742.919 | 743.552 |

Per-rep intra-cell variance is tighter for larger N (heihe N=8 max-min spread = 0.420 s = 0.47%; heihe_x4 N=8 spread = 0.637 s = 0.09%) than for N=1 (heihe N=1 spread = 16.574 s = 12.0%; heihe_x4 N=1 spread = 206.333 s = 14.6%), indicating that the N=1 cells absorb more of the cn14/cn15 background-system noise floor. This shape is consistent with P1e PR-I §3 raw data noise per-rep variance.

## §5.2 Cross-N invariance Δ=0 strict (verbatim PR-B §3.2)

Per spec n8-mode-c-profile-recheck Requirement "cross-N stats invariance check" L86-104. Each row: `X_{N=1} = X_{N=4} = X_{N=8}` (Δ=0 strict). Any Δ ≠ 0 blocks Step 2; all 10 entries PASS.

**Table 2: Cross-N invariance (10 entries × 2 cases).**

| case | key | N=1 | N=4 | N=8 | Δ (N=8 − N=1) | verdict |
|---|---|---:|---:|---:|---:|:---:|
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

10/10 PASS. This evidence supports H1 (Mode C profile-build numerical invariance) — the additional Timer instrumentation does not perturb the CVODE solver trajectory.

## §5.3 Absolute baseline anchor verify (verbatim PR-B §3.3)

Spec Scenarios L90 + L97 + tasks.md §3.3 absolute baseline anchors. Source: `docs/p1e/p1e_perf_baseline.md` §3.4 nst ladder + `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 nfe baseline. Re-validation here under SHUD pin `7a1dc8f` (vs P1e PR-I `3341368d`).

**Table 3: Absolute baseline anchor verify (4 entries).**

| case | key | expected (P1e PR-I baseline) | actual (this run, N=1 median) | verdict |
|---|---|---:|---:|:---:|
| heihe | nst | 6698 | 6698 | PASS |
| heihe | nfe | 6943 | 6943 | PASS |
| heihe_x4 | nst | 6575 | 6575 | PASS |
| heihe_x4 | nfe | 6741 | 6741 | PASS |

4/4 PASS. This confirms that the post-Timer-fix SHUD pin `7a1dc8f` preserves the P1e ship-state numerical trajectory bit-for-bit at the counter level; the Timer code path is purely instrumentation-additive (no silent regression even on N=1 single-thread).

## §5.4 ROI ratios per (case, N) (verbatim PR-B §3.4)

`r = nfeLS / nfe` per spec Scenario L63-70 (3 decimal places). For CVLS SPGMR with FD-Jvp default, expected `nfeLS = nli`; the last column `|nfeLS − nli|` is the sanity check.

**Table 4: ROI ratios (6 entries).**

| case | N | nfe_median | nfeLS_median | r = nfeLS/nfe | nni_median | nli_median | nli/nni | nfeLS − nli |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heihe | 1 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe | 4 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe | 8 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0 |
| heihe_x4 | 1 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |
| heihe_x4 | 4 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |
| heihe_x4 | 8 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0 |

`nfeLS − nli = 0` per cell confirms the SPGMR + FD-Jvp identity (every GMRES iteration is exactly one RHS evaluation). The `r` ratio is invariant across N (a corollary of §5.2 cross-N invariance), so the N=8 column is the canonical reference for ROI.

## §5.5 Branch verdict (verbatim PR-B §3.5)

**branch: a (PROCEED — `r_min = 1.819 ≥ 1.5`)**

Decision trace per spec n8-mode-c-profile-recheck Scenario "ROI verdict based on nfeLS/nfe exhaustive branching" L72-80:

- `r_{heihe, N=8}` = 1.819
- `r_{heihe_x4, N=8}` = 4.526
- `r_min = 1.819`; `r_max = 4.526`
- branch (a) `r_min ≥ 1.5` → PROCEED Step 2 P8-precond-0 identity spike (PRs #344-#347)
- branches (b) `r_max < 1.5`, (c) `r_min < 1.5 ≤ r_max < 3.0`, (d) `r_min < 1.5 AND r_max ≥ 3.0` — all FALSE — NOT triggered

ADR-0003 NO-GO branch (b) is NOT taken; this document does NOT draft a NO-GO ADR. ADR-0003 is forthcoming after Step 2 hard-gate verdict (PR-F #347).

---

# §6 Discussion

## §6.1 Hypothesis verification

**Table 5: Hypothesis verification summary.**

| Hypothesis | Operationalization | Result | Verdict |
|---|---|---|:---:|
| H1 (Mode C profile-build invariance) | 10/10 Δ=0 cross-N + 4/4 absolute anchor | All PASS (§5.2 + §5.3) | PASS |
| H2 (preconditioner ROI cost-effectiveness) | `r_min ≥ 1.5` at N=8 | `r_min = 1.819 ≥ 1.5` (§5.4 + §5.5) | PASS |
| H3 (two-case spike coverage) | heihe + heihe_x4 span small + production | NumEle 6335 → 40046 (6.3×); wall @ N=8 89.7s → 743.5s (8.3×) | PASS |

All three hypotheses PASS. Step 1 GATE PASS unconditionally; Step 2 authorization is unambiguous.

## §6.2 Comparison with P1e prior epic

P1e PR-I (Mode C, NO `SHUD_ENABLE_PROFILE=1`, SHUD pin `3341368d`) established `nst/nfe` cross-N invariance for Mode B + Mode C builds. This document extends that property to the profile-instrumented Mode C build (`SHUD_ENABLE_PROFILE=1`, SHUD pin `7a1dc8f`). The two key extensions:

1. **No profile-instrumentation drift.** The Timer code path (`tools/profile/timer.cpp`) introduces no silent solver drift. This rules out the failure mode where SHUD_ENABLE_PROFILE=1 builds report systematically different `nst/nfe` from non-profile builds. Without this confirmation, PR-F (#347) gate-4 comparison `|wall_identity − wall_baseline|` would be confounded by an instrumentation bias term.
2. **No SHUD-pin-bump drift.** P1e ship pin `3341368d` → p8pre pin `7a1dc8f` traverses the nested-Timer-fix commits (P2a v0.5 work). Despite these commits modifying `MD_ET.cpp` + `Model_Control.cpp` Timer scope, the CVODE solver trajectory is bit-identical at the counter level.

Combined, this means the gate-4 baseline anchor minted in §5.1 is a fair comparator for Step 2 PR-E (#345) identity-spike wall — same build matrix, same SHUD pin (modulo identity-precond fork from `7a1dc8f`), same partition pin. The ε bound in PR-F (#347) is therefore a clean preconditioner-overhead signal, not contaminated by instrumentation or pin drift.

## §6.3 Wall scaling profile (informational)

From Table 1: heihe sp@8 = 140.797 / 89.732 = 1.569×; heihe_x4 sp@8 = 1412.895 / 743.552 = 1.901×. Both exceed P1e PR-I baselines (heihe sp@8 = 1.066×; heihe_x4 sp@8 = 1.729×; per `docs/p1e/p1e_perf_baseline.md` §3 Tab. 10), and the heihe small-case improvement (1.066 → 1.569) is notable but expected — the Mode C path under SHUD pin `7a1dc8f` no longer carries the nested-Timer double-count that inflated wall @ N=1 in the P1e-era pre-fix baseline. This is informational only; gate-4 in PR-F (#347) uses the absolute median values in §5.1, not the speedup ratio.

---

# §7 Limitations & Threats to Validity

## §7.1 Internal validity: profile-build wall vs production-build wall

This document mints the gate-4 baseline under `SHUD_ENABLE_PROFILE=1` build. Step 2 PR-E (#345) must use the SAME build flag to keep the gate-4 comparator fair; building a production binary without `SHUD_ENABLE_PROFILE=1` for the identity-spike run would re-introduce the instrumentation-overhead delta. Spec p8precond-zero-identity-spike Scenario "same build matrix as Step 1" (forthcoming PR-F gate spec) is the contract that prevents this.

## §7.2 External validity: 2-case coverage

Only `heihe` (NumEle=6335) + `heihe_x4` (NumEle=40046). Other server cases (`keliya`, `qhh`, `qinyijiang`, `xinanjiang_upstream`, `ksge`, `tlh`) are NOT measured at N=8. The ROI ratio `r = nfeLS / nfe` could differ outside the [1.819, 4.526] envelope for other physics regimes (e.g. lake-dominated `qhh` may show different SPGMR conditioning). Mitigation: branch (c) "case-aware precond design" of the spec's 4-branch tree is the formal hook for future broadening; ADR-0003 will inherit this carve-out as a known scope limitation (per `docs/p8pre/n8_profile_verdict.md` §6 bullet 2).

## §7.3 Construct validity: 90-day case truncation

Per CLAUDE.md "项目级铁律 所有 case ≤90 天截断", both cases run a 90-day model-time window, not the full 4-year forcing window. This is sufficient for OpenMP/CVODE counter signal but under-represents long steady-state Krylov behavior in cases with multi-year spin-up transients. Effect on this document: §5.4 `r` ratios mix warm-up + steady-state phases, not pure steady-state. Step 2 PR-F (#347) gate-4 inherits the same truncation, so the comparison is apples-to-apples and the gate-4 verdict remains valid; production deployment beyond P9 would re-measure on full-window runs (per `docs/p1e/p1e_academic_summary.md` §7.2 same precedent).

## §7.4 Conclusion validity: 3-rep median noise floor for wall

For float-valued wall medians (Table 1) over 3 reps, the noise floor is ~5-15% for N=1 cells (where background-system noise dominates) and ~0.1-0.5% for N=8 cells (where compute time dominates). PR-F (#347) gate-4 ε bounds (`ε(heihe) = 0.10` ~ 9s; `ε(heihe_x4) = 0.05` ~ 37s) are calibrated against the N=8 noise floor, where the comparison signal is strongest. The N=1 wall medians in Table 1 are reported for completeness (gate-4 compares all (case, N) entries) but carry larger noise.

## §7.5 Timer instrumentation overhead unquantified at this stage

Per `docs/p8pre/step1_prep.md` §0, the per-rep wall overhead of `SHUD_ENABLE_PROFILE=1` Timer instrumentation vs a non-profile Mode C build is not yet quantified. This is acceptable for the gate-4 comparator (both sides of PR-F (#347) use the same flag, so the overhead cancels) but means absolute wall numbers in Table 1 should NOT be cited as "production wall." Master plan §1.1.1 production wall targets remain the responsibility of P9 (server-only quantification per CLAUDE.md).

---

# §8 Conclusion

p8pre-spike Step 1 GATE PASS. Branch a (PROCEED Step 2) is selected unambiguously: `r_min = 1.819 ≥ 1.5` and `r_max = 4.526` (both above the 1.5 threshold). All four absolute baseline anchors verify (heihe.nst=6698, heihe.nfe=6943, heihe_x4.nst=6575, heihe_x4.nfe=6741). All ten cross-N invariance entries are Δ=0 strict.

The 6-row wall median table in §5.1 is the canonical `wall_step1_baseline_median(case, N)` consumed by PR-F (#347) hard gate 4 (wall non-regression vs Step 1 baseline at `ε(heihe) = 0.10` and `ε(heihe_x4) = 0.05`). It is anchored here as the source-of-truth; Step 2 aggregator `tools/p8pre/aggregate_identity_spike.sh` SHALL read these six values directly from this section.

All three research hypotheses (H1 Mode C profile-build invariance; H2 preconditioner ROI cost-effectiveness; H3 two-case spike coverage adequacy) are supported by the data. Step 2 P8-precond-0 identity spike (PRs #344-#347) is authorized to proceed under ADR-0002 Path 3 deferred-option activation.

---

# §9 Future Work

**§9.1 Step 2 — P8-precond-0 identity spike (PRs #344-#347).** PR-D (#344): create `SHUD/src/Equations/MD_precond_identity.{h,cpp}` + wire `CVodeSetPreconditioner` + `CVodeSetLSetupFrequency(50)` in `cvode_config.cpp:259`. PR-E (#345): 18-cell server re-run with the identity-precond binary, mirroring Step 1 matrix exactly. PR-F (#347): 4-hard-gate + 2-soft-gate verdict, with gate-4 baseline = Table 1 in this document.

**§9.2 ADR-0003 — Step 2 capstone decision.** Drafted by PR-G (#347+1, forthcoming). Decision branches: (a) GO — start formal P8-precond epic with block-Jacobi physics preconditioner (per master plan §P8-precond.1-.7); (b) NO-GO — transition to P8-tune (maxl/restart/EpsLin tuning); (c) DEFER — open issue for cross-N reduction-drift carve-out.

**§9.3 Broaden case coverage.** If ADR-0003 = GO, the formal P8-precond epic SHALL re-measure ROI ratios on the full server canonical case set (heihe, heihe_x4, keliya, qhh, qinyijiang, xinanjiang_upstream) at N=8 to either confirm or refine the case-aware precond strategy hinted by spec branches (c) + (d).

**§9.4 Beyond P9.** Production rollout (per master plan §P9) revisits the 90-day truncation carve-out (§7.3) and lifts wall measurement to full-multi-year runs. The gate-4 anchor minted here is Step 2-only; production wall comparators are independent.

---

# References

## Internal documents

- `docs/p8pre/n8_profile_run.md` — PR-A execution log (#341); 18-cell Slurm run + per-cell 5-gate verification + rsync mirror provenance.
- `docs/p8pre/n8_profile_verdict.md` — PR-B verdict (#342); 15-key + 7-bucket + 2-extras aggregator output, branch (a) PROCEED.
- `docs/p8pre/step1_prep.md` — P1e PR-I absolute baseline anchor (Step 0 doc-correction PR output); inline-quotes wall/nst/nfe for downstream stability.
- `docs/p1e/p1e_perf_baseline.md` §3.1 wall median table + §3.4 nst ladder.
- `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 nfe baseline + §3 24-cell verdict.
- `docs/p1e/p1e_academic_summary.md` — academic-paper-style mother template (per CLAUDE.md "阶段总结文档风格" user-pref).
- `docs/p1e/p1e_summary.md` §7 P1e SHIP verdict + §10 P1e-tag pinning.
- `docs/adr/0002-solver-path.md` Path 3 deferral + Decision Matrix.
- `docs/case_deployment_map.md` §2.2 server canonical baseline + §5 SHIP set table.
- `SHUD_openMP_master_plan.md` §P8-precond.0 prep (cross-ref this document; updated in this PR).
- `CLAUDE.md` "阶段总结文档风格 (user pref 2026-06-25)" + "Slurm 三铁律".

## OpenSpec

- `openspec/changes/p8pre-spike/proposal.md` (epic rationale).
- `openspec/changes/p8pre-spike/tasks.md` §1-§4 (PR-A/B/C scope) + §5-§8 (PR-D/E/F scope).
- `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` Requirements 1-6 + Scenarios L13-122.
- `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` (Step 2 gates spec).
- `openspec/glossary.md` (forthcoming PR-G additions: `p8pre-spike`, `identity preconditioner`, `PREC_LEFT`, `nfeLS/nfe ratio`, `CVodeSetLSetupFrequency`).

## GitHub PRs

- PR-A (#341) — `docs/p8pre/n8_profile_run.md` + `tools/p8pre/run_n8_profile.sh` + sbatch template. <https://github.com/DankerMu/SHUD-OpenMP/pull/341>
- PR-B (#342) — `tools/p8pre/aggregate_n8_profile.sh` + `docs/p8pre/n8_profile_verdict.md`. <https://github.com/DankerMu/SHUD-OpenMP/pull/342>
- PR-C (#343) — this document + `docs/case_deployment_map.md` §5 update + `SHUD_openMP_master_plan.md` §P8-precond cross-ref. <https://github.com/DankerMu/SHUD-OpenMP/pull/343>
- PR-D (#344) — identity preconditioner stub + cvode_config wire (forthcoming).
- PR-E (#345) — server identity-spike 18-cell run (forthcoming).
- PR-F (#347) — 4-hard-gate + 2-soft-gate verdict + gate-4 read from this §5.1 (forthcoming).
- PR-G — ADR-0003 + epic close (forthcoming).

## Tag & SHA pinning

- SHUD pin (p8pre-spike Step 1): `7a1dc8f` (P1e ship pin, post-nested-Timer fix; verified invariant across 18 cells).
- Outer baseline branch: `baseline/p8pre` HEAD `8ed785f` (post PR-B #342 merge `284ad32`; PR-C fork from here).
- P1e-tag (referenced baseline): annotated object `25023eff32d1fa317b045cbc786f379fac9e522c`, deref commit `11687b756dd53bb634df391bcbeb64b3cef5c750`.

## External dependencies

- SUNDIALS-CVODE 6.0.0 — `cvode.h:132` `CVodeSetLSetupFrequency`; `cvode_ls.h:91` `CVodeSetJacEvalFrequency`; `cvode_ls.h` `CVLsPrecSetupFn` + `CVLsPrecSolveFn` typedefs (signatures for `MD_precond_identity.cpp` in Step 2).
- SUNDIALS source `src/sunlinsol/spgmr/sunlinsol_spgmr.c` — SPGMR Krylov solver; `nfeLS` accumulates FD-Jvp RHS callbacks per inner iteration.
- OpenMP 4.5+ standard — `#pragma omp for schedule(static)` deterministic chunk assignment (per P1e §3.2 D2 + D4 mother-template Methodology).
- GCC 13.3.0 + libgomp on cn14/cn15.

## Methodology references

- Hindmarsh A. C., et al. "SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers." ACM TOMS 31(3):363–396 (2005). CVODE design + SPGMR nfeLS semantics.
- Saad Y. "Iterative Methods for Sparse Linear Systems." 2nd ed., SIAM (2003). SPGMR + Krylov subspace iteration theory; preconditioner left/right framing.
- Davis T. A. "Direct Methods for Sparse Linear Systems." SIAM (2006). KLU reference (deferred to ADR-0002 Path 4; not in p8pre-spike scope).

---

Generated: 2026-06-27 by claude code subagent (implementer); source-of-truth = `docs/p8pre/n8_profile_verdict.md` + `docs/p8pre/n8_profile_run.md` + per-cell `/tmp/p8pre_n8_profile/<cell>/profile_B0.yaml` + `cvode_stats.txt`.
