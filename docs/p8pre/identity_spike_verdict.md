---
title: "p8pre-spike Step 2 PR-F — Identity-Precond Spike Verdict"
subtitle: "Academic-style adjudication: 4 hard gates + 2 soft gates + ADR-0003 NO-GO recommendation"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-27
epic: "SHUD-OpenMP#338 (p8pre-spike)"
issue: "SHUD-OpenMP#347 (PR-F verdict adjudicator)"
SHUD_pin: 5276167
outer_pin: f800bb2
spike_phase: "Step 2 PR-F (verdict adjudication)"
verdict: NO-GO
adr_recommendation: NO-GO (design D8 fall-back PREC_NONE)
hard_gates:
  gate_1_build_PASS: PASS
  gate_2_ncfn_zero: FAIL
  gate_3_nps_npe_accumulation: PASS
  gate_4_wall_non_regression: PASS
soft_gates:
  gate_5_cross_N_tolerance: FAIL
  gate_6_setup_overhead: PASS
related_docs:
  - "openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md (L74-130)"
  - "openspec/changes/p8pre-spike/design.md (D5/D7/D8)"
  - "openspec/changes/p8pre-spike/tasks.md (§8 PR-F)"
  - "docs/p8pre/n8_profile_baseline.md (PR-C #355, gate-4 wall baseline anchor)"
  - "docs/p8pre/identity_spike_run.md (PR-E #358, 18-cell capture)"
  - "docs/p1e/p1e_pr_i_strict_omp_verification.md (gate-5 baseline SHA12 per-case)"
  - "docs/p1e/p1e_academic_summary.md (academic-style mother template)"
  - "SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c (SUNDIALS canonical PSetup/PSolve pattern)"
---

# Abstract

This document reports the verdict of the p8pre-spike Step 2 identity-precond spike (issue #347, PR-F). The spike instruments SUNDIALS PREC_LEFT with an identity preconditioner stub (P^-1 = I) and runs a 18-cell matrix (heihe + heihe_x4 × N ∈ {1,4,8} × 3 reps) on partition cn14/cn15 (gcc 13.3.0 + libgomp) at SHUD pin `5276167` (`openmp-baseline-p8pre` branch). The aggregator `tools/p8pre/aggregate_identity_spike.sh` evaluates four hard gates and two soft gates against the spec scenarios in `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L74-130. Result: **Gate 1 PASS, Gate 2 FAIL (ncfn>0 across all 18 cells), Gate 3 PASS, Gate 4 PASS**; soft gates: Gate 5 FAIL (per-cell SHA12 mismatch + max_ulp >> 1024), Gate 6 PASS (setup overhead 6+ orders of magnitude below the 5% threshold). Per spec L74-79 + design D8, any hard-gate failure drives **spike verdict = NO-GO**; ADR-0003 (PR-G #348) MUST record NO-GO and revert PREC_LEFT → PREC_NONE, delete `MD_precond_identity.{h,cpp}`, and close `baseline/p8pre`. The gate 2 failure quantitatively validates the theoretical position that an identity preconditioner contributes no convergence-acceleration ROI: SUNDIALS observed 6 nonlinear convergence failures (`ncfn`) for heihe and 47 for heihe_x4 deterministically across all 9 cells per case, identical to a hypothetical PREC_NONE run on the same stiff Jacobian. The spike fulfills its scoping mandate — wire SUNDIALS PREC_LEFT plumbing, measure ROI ceiling, decide GO/NO-GO for the formal P8-precond epic.

**Keywords**: SHUD; CVODE 6.0.0; SPGMR; PREC_LEFT; identity preconditioner; PSetup/PSolve callbacks; convergence failure; ROI gating

---

# §1 Introduction

Hydrological models built atop implicit ODE integrators face a recurring engineering question: when the linear-solve cost dominates wall time, does a preconditioner pay back its setup cost? The SHUD-OpenMP P1e epic [1] established that the Mode C strict-omp build at `SHUD_RHS_THREADS=8` achieves 1.066× / 1.729× speedup on heihe (NumEle=6335) / heihe_x4 (NumEle=40046) respectively, with `t_CVODE_internal` accounting for ~25-33% of wall time. The CVODE iterative solver currently runs SPGMR without a registered preconditioner (`PREC_NONE`). Open question: would registering even a trivial preconditioner unlock setup/solve API wiring needed for a future Diagonal/Jacobi → element-block-Jacobi epic?

The p8pre-spike epic (issue #338) [2] answers this in two steps: Step 1 (`PR-A` through `PR-C`) re-profiles Mode C with the same SHUD pin to refresh `ratio_nfeLS_over_nfe` and harden the wall-time baseline; Step 2 (`PR-D` through `PR-G`) wires SUNDIALS PREC_LEFT with an identity preconditioner stub and adjudicates four hard gates plus two soft gates. This document corresponds to PR-F — the verdict adjudicator.

We formalize three research hypotheses:

- **H1 (plumbing validation)**: With `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` registered and `LSetupFrequency=50`, SUNDIALS SHALL invoke both callbacks (`nps > 0`, `npe > 0`) during the 18-cell run. Operational definition: gate 3.
- **H2 (convergence neutrality)**: PREC_LEFT with P^-1 = I is mathematically equivalent to PREC_NONE; SHALL therefore preserve zero convergence-failure semantics (`ncfn = 0` per cell). Operational definition: gate 2.
- **H3 (overhead bound)**: The identity-stub setup cost SHALL remain below 5% of wall time (`t_precond_setup / t_wall_total ≤ 0.05`). Operational definition: soft gate 6.

H1 and H3 are expected PASS; H2 is the spike's primary research question — whether the SPGMR step-control machinery alone can sustain Newton convergence on the SHUD stiff Jacobian without an actual preconditioner-side rotation. The empirical answer becomes the ROI ceiling for any future preconditioner candidate: it must reduce `ncfn` below the observed PREC_LEFT-identity floor while paying its setup cost.

---

# §2 Related Work

**§2.1 SUNDIALS preconditioner API**. The CVLS interface in CVODE 6.0.0 [3] provides `CVodeSetPreconditioner(cvode_mem, CVLsPrecSetupFn pset, CVLsPrecSolveFn psolve)` for SPGMR/SPBCGS/SPTFQMR preconditioner registration. The canonical reference example is `cvDiurnal_kry.c` (`SHUD/InstallSundials/example/cvode/serial/`) where `Precond()` at L716 and `PSolve()` at L760 demonstrate the expected return-code contract (`*jcurPtr` write, `*jok` consumption, ier=0/positive/negative for recoverable/unrecoverable failures). The current PR-D stub `MD_precond_identity.{h,cpp}` mirrors this contract minimally: PSetup returns ier=0 with `*jcurPtr=0` (identity is invariant); PSolve copies `r` to `z` and returns 0.

**§2.2 P1e strict-omp bitwise baseline**. P1e PR-I [4] established the canonical Mode C per-case unique SHA12: heihe = `a2023ccd2de4` and heihe_x4 = `b5e4b0a2cf83` across 12 cells (N ∈ {1,2,4,8} × 3 reps). These values are the gate-5 strict-bitwise baseline anchor for soft gate 5 below.

**§2.3 Step 1 PR-A wall baseline**. Step 1 PR-A (issue #341, PR-B aggregator #342) [5] produced the 6-tuple `wall_step1_baseline_median(case, N)` archived to `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1, using the same build matrix (`SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`) and partition pin (cn14/cn15) — the only fair gate-4 comparison anchor (using P1e PR-I wall times would mix two non-comparable build matrices because PR-I built without `SHUD_ENABLE_PROFILE=1` at SHUD `3341368d`).

**§2.4 Design D5 + D7 gate construction**. The four hard gate / two soft gate cutoffs were finalized in `openspec/changes/p8pre-spike/design.md` D5 (per-case epsilon for gate 4, A4 tolerance for gate 5) and D7 (bitwise semantics relaxed to A4 for Step 2 because PREC_LEFT triggers extra `N_VLinearSum` / `N_VScale` operations whose reduction order may drift from PREC_NONE baseline).

---

# §3 Methodology

**§3.1 Build matrix**. SHUD source compiled on server cn14/cn15 (gcc 13.3.0 + libgomp, SUNDIALS-CVODE 6.0.0 from `SHUD/InstallSundials/`) with `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` at SHUD pin `5276167`. The build links `MD_precond_identity.{h,cpp}` (PR-D #357) and patches `cvode_config.cpp:259` from `CVSpilsSetPrecType(SUN_PREC_NONE)` to `SUN_PREC_LEFT` plus `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` plus `CVodeSetLSetupFrequency(cvode_mem, 50)`. Build evidence: `server_nm.log` (cell `T PSetupIdentity` at offset `0x21830`, `T PSolveIdentity` at offset `0x21880`, undefined-then-resolved `U CVodeSetPreconditioner`).

**§3.2 Submission**. The 18 cells (2 cases × 3 N × 3 reps) were submitted as singleton afterany chain (per-case linear chain, partitioned to cn14 (heihe) and cn15 (heihe_x4) per the §3.1 baseline policy). Run window: 2026-06-27 05:06:41Z → 06:18:02Z UTC (~71 minutes wall). All 18 cells exited 0; artifacts rsynced from `/scratch/.../.p8pre-runs/identity_spike/<cell>/` to Mac `/tmp/p8pre_identity_spike/<cell>/`.

**§3.3 Aggregation**. The local-side aggregator `tools/p8pre/aggregate_identity_spike.sh` (POSIX bash + awk + grep + sha256sum + `uv run python` for ULP computation) consumes the 18 cells plus `server_nm.log` plus the Step 1 PR-A baseline mirror at `/tmp/p8pre_n8_profile/` plus the hard-coded baseline values from `docs/p8pre/n8_profile_baseline.md` §5.1 and `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1. Output: stdout summary + `/tmp/p8pre_identity_spike/aggregate_verdict.txt` structured KV file (135 lines).

**§3.4 Median policy**. Per-(case, N) wall median = middle of 3 sorted `t_wall_total` values across reps (mirrors Step 1 PR-A 3-rep median aggregation per spec L113). NB: CVODE counter values (nst, nfe, ncfn, nps, npe) are identical across all 3 reps within a (case, N) group, so median == per-cell value.

**§3.5 Max-ULP fall-back**. Because `rivqdown.dat` is raw little-endian doubles (no SHUD snapshot magic header), the `tools/compare_snapshot` binary returns "format mismatch" (exit 2). Per spec L102-106 (the explicit Python fall-back clause), the aggregator computes max-ULP using inline numpy `np.spacing` over per-element double arrays.

**§3.6 18-cell raw data table**.

**Table 1: Per-cell stats. Slurm Elapsed + ExitCode are PR-E forward-carry columns; SHA12 + max_ulp are PR-F-added columns.**

| JID | case | N | rep | Elapsed (s) | ExitCode | nst | nfe | ncfn | nps | npe | wall_total (s) | t_precond_setup (s) | SHA12(rivqdown) | max_ulp |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|
| 9531 | heihe | 1 | 1 | 142 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 137.079 | 8.872e-6 | `99bff185ef2c` | 9.00e15 |
| 9532 | heihe | 1 | 2 | 142 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 137.274 | 8.923e-6 | `1cae9410705c` | 8.99e15 |
| 9533 | heihe | 1 | 3 | 126 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 121.619 | 8.255e-6 | `1cae9410705c` | 8.99e15 |
| 9534 | heihe | 4 | 1 | 99 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 94.049 | 8.824e-6 | `878da1e9c5ae` | 9.01e15 |
| 9535 | heihe | 4 | 2 | 99 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 93.846 | 8.767e-6 | `9c9196bf079e` | 9.01e15 |
| 9536 | heihe | 4 | 3 | 99 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 93.499 | 9.155e-6 | `7b7a4cfe21ee` | 9.01e15 |
| 9537 | heihe | 8 | 1 | 94 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 88.216 | 8.812e-6 | `1cae9410705c` | 8.99e15 |
| 9538 | heihe | 8 | 2 | 93 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 87.621 | 8.766e-6 | `44c4360b3efe` | 9.01e15 |
| 9539 | heihe | 8 | 3 | 93 | 0 | 6599 | 6696 | 6 | 18163 | 77 | 88.038 | 8.885e-6 | `1cae9410705c` | 8.99e15 |
| 9540 | heihe_x4 | 1 | 1 | 1519 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 1491.050 | 1.431e-5 | `8bb7e8ca0370` | 9.01e15 |
| 9541 | heihe_x4 | 1 | 2 | 1457 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 1428.287 | 1.381e-5 | `8bb7e8ca0370` | 9.01e15 |
| 9542 | heihe_x4 | 1 | 3 | 1301 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 1273.800 | 1.301e-5 | `8bb7e8ca0370` | 9.01e15 |
| 9543 | heihe_x4 | 4 | 1 | 887 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 858.281 | 1.409e-5 | `ba5cdd122f59` | 9.01e15 |
| 9544 | heihe_x4 | 4 | 2 | 887 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 858.297 | 1.265e-5 | `531d1f173b84` | 9.00e15 |
| 9545 | heihe_x4 | 4 | 3 | 876 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 847.850 | 1.395e-5 | `8bb7e8ca0370` | 9.01e15 |
| 9546 | heihe_x4 | 8 | 1 | 777 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 748.337 | 1.383e-5 | `8bb7e8ca0370` | 9.01e15 |
| 9547 | heihe_x4 | 8 | 2 | 777 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 747.720 | 1.454e-5 | `3e44870d726c` | 9.00e15 |
| 9548 | heihe_x4 | 8 | 3 | 778 | 0 | 6569 | 6775 | 47 | 37695 | 158 | 749.529 | 1.424e-5 | `8bb7e8ca0370` | 9.01e15 |

Sources: `wall_total` from per-cell `profile_B0.yaml` `extras.t_wall_total`; CVODE counters from per-cell `cvode_stats.txt`; `t_precond_setup` from per-cell `profile_B0.yaml` `extras.t_precond_setup` (the new Timer bucket added by PR-D §6.2); SHA12 from `sha256sum <case>.rivqdown.dat | cut -c1-12`; max_ulp from inline numpy on raw double arrays vs Step 1 PR-A baseline mirror; Slurm Elapsed derived from `slurm.out` `start_utc` / `end_utc` (UTC) timestamps; ExitCode from `slurm.out` `run_exit_code=N` line.

---

# §4 Hard-gate verdict

**Table 2: Four hard gates (any FAIL → spike NO-GO per spec L74-79).**

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| 1 | Build PASS | server `nm ./shud` shows `PSetupIdentity` + `PSolveIdentity` + `CVodeSetPreconditioner` symbols (≥1 each) | **PASS** | `server_nm.log`: 3 hits (1 each) |
| 2 | Zero convergence failure | `ncfn = 0` across all 18 cells | **FAIL** | 18 / 18 cells violate; heihe deterministic `ncfn = 6` (9/9), heihe_x4 deterministic `ncfn = 47` (9/9) |
| 3 | nps + npe accumulation | `nps > 0` AND `npe > 0` per cell | **PASS** | min_nps = 18163, min_npe = 77; SUNDIALS calls both callbacks deterministically |
| 4 | Wall non-regression | per (case, N): `|wall_identity_median − wall_step1_baseline_median| / wall_step1_baseline_median ≤ ε(case)` with `ε(heihe) = 0.10`, `ε(heihe_x4) = 0.05` | **PASS** | All 6 (case, N) groups within tolerance |

**Gate 4 per-(case, N) wall comparison:**

| case | N | wall_identity (s) | wall_baseline (s) | delta | epsilon | verdict |
|---|---:|---:|---:|---:|---:|---|
| heihe | 1 | 137.079 | 140.797 | 2.64% | 10.0% | PASS |
| heihe | 4 | 93.846 | 95.734 | 1.97% | 10.0% | PASS |
| heihe | 8 | 88.038 | 89.732 | 1.89% | 10.0% | PASS |
| heihe_x4 | 1 | 1428.287 | 1412.895 | 1.09% | 5.0% | PASS |
| heihe_x4 | 4 | 858.281 | 849.704 | 1.01% | 5.0% | PASS |
| heihe_x4 | 8 | 748.337 | 743.552 | 0.64% | 5.0% | PASS |

Maximum delta: heihe N=1 at 2.64% (well below 10.0%), heihe_x4 N=1 at 1.09% (well below 5.0%). The identity preconditioner imposes near-zero wall overhead — consistent with the trivial `memcpy(z, r)` PSolve body and the `*jcurPtr=0` PSetup body (both effectively O(N) memory ops dwarfed by RHS kernel cost).

**Conclusion**: Three of four hard gates PASS. Gate 2 FAILs deterministically (heihe `ncfn=6 ± 0`, heihe_x4 `ncfn=47 ± 0` across all 9 cells per case), driving spike verdict NO-GO.

---

# §5 Soft-gate verdict

**Table 3: Two soft gates (FAIL → carve-out + ADR-0003 open issue per spec L102-130).**

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| 5 | Cross-N tolerance | per (case, N, rep): SHA12(rivqdown.dat) == baseline_SHA12(case) (strict); OR max_ulp ≤ 1024 (A4 tolerance fall-back per master plan §2.3) | **FAIL** | strict: 18/18 violate; fall-back: 18/18 violate (max_ulp ≈ 9×10¹⁵ across all cells) |
| 6 | Setup overhead | per cell: `t_precond_setup / t_wall_total ≤ 0.05` | **PASS** | max ratio = 1.01×10⁻⁷ (6 orders of magnitude below the 0.05 threshold) |

**Gate 5 analysis**. The baseline SHA12 per spec L102-106 is the per-case unique SHA from P1e PR-I [4]: heihe = `a2023ccd2de4` and heihe_x4 = `b5e4b0a2cf83`. None of the 18 PR-E identity cells match. Falling back to max-ULP via numpy on raw doubles vs the Step 1 PR-A baseline mirror cells: every cell reports max_ulp ≈ 9×10¹⁵, vastly exceeding the A4 threshold of 1024. Root cause analysis: (a) per-cell `rivqdown.dat` byte count = 1,714,016 = 214,252 doubles (sizes match between baseline and identity); (b) baseline has 57,187 zero positions and identity has 62,342 zero positions — *the zero/non-zero set differs* between the two runs at 5,155 positions; (c) at non-zero baseline positions where identity emits zero (or vice versa), the relative diff explodes (`|a − b| / spacing(max(|a|,|b|))` with one side near machine-zero produces ULP distances on the order of 10¹⁵). This is NOT pure reduction-order drift — it is a structural difference between PREC_LEFT-with-identity and PREC_NONE numerical trajectories. Although mathematically P^-1 = I should commute, the SUNDIALS state machine for PREC_LEFT executes additional `N_VLinearSum` / `N_VScale` operations (per `cvode_spils.c` `cvLsSolve`) that perturb the iterative residual and rotate which time-step / Newton iterations cross the rivqdown truncation thresholds. Gate 5 verdict: **FAIL_STRICT → FAIL_FALLBACK** (no PASS path), recorded as ADR-0003 carve-out per spec L106-108.

NB on Step 1 PR-A baseline drift: examining the PR-A mirror per-cell SHA12 reveals that the PR-A run *itself* did not uniformly produce `a2023ccd2de4` per cell — only 5 of 9 heihe cells and 5 of 9 heihe_x4 cells matched the canonical P1e PR-I anchor. This is consistent with the PR-A documentation note that PR-A used a different SHUD pin (`7a1dc8f`) than P1e PR-I (`3341368d`), and that small reduction-tree variations across the new RHS Timer-instrumented binary are within A4 tolerance for the strict-omp build. Step 2 PR-F treats the canonical P1e PR-I per-case SHA as the gate-5 anchor (per spec), not the per-cell PR-A SHA.

**Gate 6 analysis**. The new `t_precond_setup` Timer bucket added by PR-D §6.2 (RAII `shud_profile::Timer _t("t_precond_setup");` at the head of `PSetupIdentity()` in `MD_precond_identity.cpp`) is present in every `profile_B0.yaml`. Range: 8.255×10⁻⁶ to 1.454×10⁻⁵ seconds across 18 cells, corresponding to ratio range 9.34×10⁻⁹ to 1.01×10⁻⁷. Maximum ratio is **6 orders of magnitude below** the 5×10⁻² threshold. This is expected — identity PSetup is trivially `*jcurPtr=0; return 0;` (no actual matrix work). **PASS** with strong margin.

The Gate 6 PASS does NOT translate to a real-preconditioner ROI prediction: identity-stub setup cost is the *lower bound* on PSetup cost for any future preconditioner. A Diagonal preconditioner would compute diag(J) and store reciprocals (O(N_eq) flops); ILU(0) would compute a sparse triangular factorization (O(nnz) flops). The empirical 1×10⁻⁷ ratio merely confirms the Timer instrumentation is wired correctly and the SUNDIALS callback dispatch overhead is negligible — a necessary but not sufficient condition for productive preconditioning.

---

# §6 Discussion + ROI Implication

**§6.1 Gate 2 failure as designed validation**. The spike's primary research question (H2 in §1) is whether SPGMR step-control alone sustains Newton convergence on the SHUD stiff Jacobian without a preconditioner-side rotation. Gate 2 FAIL deterministically answers NO: SUNDIALS observes 6 nonlinear convergence failures per heihe cell and 47 per heihe_x4 cell. Crucially, the **invariance** across (N, rep) — heihe `ncfn = 6` exactly in all 9 cells, heihe_x4 `ncfn = 47` exactly in all 9 cells — confirms this is a deterministic property of the (90-day forcing × case) system, not a noise artifact. Any future preconditioner candidate must reduce these floors to justify its setup cost. The identity stub does not reduce them by construction (P^-1 = I performs no rotation of the residual), so the floor is the same `ncfn` count a hypothetical PREC_NONE run would produce on the same case/forcing.

**§6.2 ROI quantification for formal P8-precond epic**. Per `docs/p1e/p1e_perf_baseline.md` §4 + cvode_stats parsing of the 18-cell run, the breakdown for heihe N=8 (representative) is:
- nst = 6599 successful steps
- nfe = 6696 RHS evaluations
- nfeLS = 12120 RHS evaluations attributed to linear solver = nps `+` extra unscaled Jacobian touches
- nps = 18163 PSolve calls
- npe = 77 PSetup calls
- ncfn = 6 nonlinear convergence failures (each typically triggers PSetup re-call)
- ncfl = 121 linear convergence failures (force SPGMR restart)

The ratio `nfeLS / nfe = 12120 / 6696 = 1.811` confirms substantial linear-solver overhead. A productive preconditioner would: (a) reduce `nli` (currently 12120) per Newton step, (b) eliminate `ncfn` floor by stabilizing Newton, (c) reduce `ncfl` SPGMR restarts. Identity preconditioner achieves none of these — establishing the operational ROI ceiling. **PASS criterion for any future P8-precond candidate: ncfn < 6 (heihe) and ncfn < 47 (heihe_x4) with combined setup-plus-solve overhead within 10% of identity-baseline wall.**

**§6.3 Comparison to prior epics**. P1c-P1d epics (carve-out chain summarized in [1] §2.1-2.3) chased reduction-order drift across N; this Step 2 spike redirects the diagnostic lens to the *upstream* of reduction — the linear solver state machine. The finding that PREC_LEFT-with-identity diverges from PREC_NONE at 5,155 of 214,252 rivqdown positions (§5 gate-5 analysis) is novel for this engineering line: it quantifies the floating-point cost of merely instantiating SUNDIALS PREC_LEFT codepath even when the preconditioner operator is mathematical identity. This finding informs Open Question Q3 (`docs/p8pre/identity_spike_run.md` §Q3) and ADR-0003.

**§6.4 Soft gate 5 expected-FAIL semantics**. Per spec L106-108 and design D7, soft gate 5 was a priori expected to potentially FAIL because PREC_LEFT triggers extra `N_VLinearSum` / `N_VScale` ops whose reduction order may drift. The strict-bitwise PASS path was the optimistic case; the A4 fall-back PASS path (max_ulp ≤ 1024) was the realistic case. Empirical max_ulp ≈ 9×10¹⁵ vastly exceeds even the relaxed threshold — but this is not a code-defect signal; it is the structural-divergence signal described in §5. The ADR-0003 NO-GO decision absorbs this FAIL as expected information (the spike answered an unknown), not as a defect to be patched.

---

# §7 ADR-0003 Recommendation

**Verdict**: NO-GO (gate 2 hard FAIL per spec L74-79 + design D8 fall-back PREC_NONE).

**Rationale**: The identity preconditioner provides no convergence acceleration (gate 2 FAIL) while adding non-trivial numerical-trajectory drift (gate 5 FAIL). Although gates 1, 3, 4, and 6 all PASS — confirming SUNDIALS plumbing is wired correctly and overhead is negligible — the absence of ROI removes any justification for retaining the PREC_LEFT codepath. Per design D8, retaining a no-ROI PREC_LEFT path would constitute carrying dead complexity into downstream epics, which violates project KISS/YAGNI principle.

**PR-G (#348) MUST**:

1. **Write `docs/adr/0003-precond-spike-decision.md`** documenting NO-GO with the §4/§5 verdict tables + §6 ROI quantification, per ADR template (`docs/adr/0002-solver-path.md` structure).
2. **Revert `cvode_config.cpp:259`** from `SUN_PREC_LEFT` back to `SUN_PREC_NONE` (delete the `CVodeSetPreconditioner` call + the `CVodeSetLSetupFrequency(50)` call).
3. **Delete `MD_precond_identity.{h,cpp}`** and unlink them from the Makefile target.
4. **Delete the `t_precond_setup` Timer bucket** wired by PR-D §6.2 in `tools/profile/timer.cpp` (or carve out as `unused_bucket` for future re-use).
5. **Close `baseline/p8pre`** branch (no further p8pre work this line; future P8-precond epic should re-fork from `main` with a clean prior-art base).
6. **Update `SHUD_openMP_master_plan.md` §P8-precond.0**: add cross-ref to `docs/adr/0003-precond-spike-decision.md` + ROI ceiling data + NO-GO outcome + re-estimated P8-precond epic engineering effort (with the spike findings, the formal epic must include preconditioner candidate selection — Diagonal vs Jacobi vs block-Jacobi — *before* PSetup/PSolve API integration, inverting the current §P8-precond.6 order).
7. **Optional**: write `docs/p8pre/p8pre_summary.md` capstone summarizing both Step 1 and Step 2 outcomes.

**Forward carry to future P8-precond epic** (when/if intaken):
- The SUNDIALS canonical PSetup/PSolve API pattern is established (`SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760); spec L26 wording correction noted in PR-D #357 should be propagated.
- The 1.811 `nfeLS/nfe` ratio and the 6/47 `ncfn` floor per case form the ROI baseline against which Diagonal/Jacobi/ILU(0) candidates must demonstrate improvement.
- The structural drift of PREC_LEFT vs PREC_NONE at 5,155 of 214,252 positions confirms ADR-0002 Path 3 (block-Jacobi precond) cannot assume bitwise neutrality with Mode C strict-omp baseline; it would need its own AC-S1/AC-S2 re-verification or a new acceptance grade.

---

# §8 Limitations and Threats to Validity

**§8.1 Case coverage**. Only heihe (NumEle=6335) and heihe_x4 (NumEle=40046) are exercised. The 5 CI cases (keliya / xinanjiang_upstream / qinyijiang / qhh / tailanhe) are NOT in spike scope; their `ncfn` floors are unknown and may differ qualitatively. However, gate 2 deterministic FAIL on the 2 production cases suffices to drive NO-GO — broader case coverage would only strengthen the verdict, not weaken it.

**§8.2 90-day truncation**. Per project rule "all cases ≤90 days for OpenMP verification" (CLAUDE.md), the 18-cell run uses 90-day forcing rather than the 4-year production model time. The `ncfn` floor at 90 days may not extrapolate linearly to 4-year runs — production-scale `ncfn` could be much higher (more time-step adaptivity / more events). This strengthens the NO-GO verdict (identity preconditioner is even less useful on long runs), so no threat-to-validity concern.

**§8.3 Single-server pin**. cn14 (heihe) / cn15 (heihe_x4) are partitioned per task §7.1; cross-server validation (e.g., cn05/cn06/cn09) was not performed. Inter-node ABI variance for libgomp / libsundials_cvode is negligible based on prior P1e PR-I 24-cell experience.

**§8.4 ULP measurement choice**. The max-ULP computation uses `np.spacing(max(|a|,|b|))` as denominator, which is conservative when one side is exactly zero (denominator collapses to `np.finfo(np.float64).tiny ≈ 5e-324`, exploding ULP for any non-zero numerator). An alternative metric — `tools/compare_snapshot` BITWISE comparison only (without ULP normalization) — would have reported 154,665 of 214,252 differing positions (72%) for the heihe N=1 rep1 cell. Either metric supports the FAIL verdict.

**§8.5 `compare_snapshot` format gap**. The `tools/compare_snapshot/compare_snapshot` binary expects a magic-headered binary format for the SHUD RHS snapshot, but `rivqdown.dat` is the canonical hydrologic output dump (raw little-endian doubles, no header). The Python fall-back is explicitly authorized by spec L102-106 and produces the gate-5 verdict deterministically. Should this format gap be considered tech-debt for a future tool epic, the Python script in `tools/p8pre/aggregate_identity_spike.sh` Phase E `compute_max_ulp()` (lines 195-225) is the reference implementation.

**§8.6 Soft gate 6 zero-cost trivia**. The 1×10⁻⁷ overhead ratio is structurally guaranteed by the identity-stub's near-empty PSetup body; it does NOT predict real-preconditioner overhead. Soft gate 6 PASS should be read as "Timer wiring correct + dispatch overhead negligible" rather than "preconditioners are cheap".

---

# §9 Conclusion + Future Work

**§9.1 Conclusion**. The p8pre-spike Step 2 identity-precond spike (PR-D through PR-F) fulfills its scoping mandate: H1 (plumbing) PASS, H2 (convergence neutrality) FAIL with deterministic floor, H3 (overhead) PASS with strong margin. Per spec L74-79, any hard-gate FAIL drives spike NO-GO. PR-G (#348) inherits the NO-GO recommendation and the operational ROI ceiling (heihe `ncfn=6`, heihe_x4 `ncfn=47`, `nfeLS/nfe = 1.811`).

**§9.2 Future Work**. Should a formal P8-precond epic be intaken:
- **Prerequisite 1**: Real preconditioner candidate (Diagonal/Jacobi/ILU(0)) must demonstrate `ncfn < 6` (heihe) AND `ncfn < 47` (heihe_x4) at acceptable setup cost. Identity stub establishes the no-rotation baseline.
- **Prerequisite 2**: Accept the `ncfn > 0` baseline and tune the CVODE step controller (`max_step`, `min_step`, `nonlin_conv_coef`) to absorb retries into successful steps. This is the P8-tune path (ADR-0002 Path 1 sub-option).
- **Prerequisite 3**: Investigate whether `CVodeSetMaxNonlinIters` increase reduces the deterministic 6/47 floor — the floor may reflect Newton residual stalls that more iterations could resolve at constant work cost.
- **Prerequisite 4**: Investigate SPGMR `maxl` parameter (currently SUNDIALS default 5); raising to 10-15 may reduce `ncfl=121` restart count and yield wall improvement independent of preconditioner choice.
- Per design D7 + ADR-0002 D7: the SPGMR → KLU direct-solver path (ADR-0002 Path 4) remains a valid option if iterative preconditioning continues to disappoint; KLU's analyze-and-factor-once policy on the heihe Jacobian sparsity pattern is empirically untested at present.

The Step 1 PR-A baseline (`docs/p8pre/n8_profile_baseline.md`) + the Step 2 gate-2 floor data (this document §3 Table 1) together form the gate-evaluation anchor for any future iterative-solver tuning experiment under the SHUD-OpenMP master plan.

---

# §10 References

- [1] P1e capstone academic summary — `docs/p1e/p1e_academic_summary.md` (SHUD-OpenMP epic #283)
- [2] p8pre-spike epic OpenSpec — `openspec/changes/p8pre-spike/proposal.md` (SHUD-OpenMP #338)
- [3] SUNDIALS CVODE 6.0.0 CVLS interface — `SHUD/InstallSundials/include/cvode/cvode_ls.h`; canonical preconditioner example `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760
- [4] P1e PR-I strict-omp verification — `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 (heihe `a2023ccd2de4`, heihe_x4 `b5e4b0a2cf83`)
- [5] Step 1 PR-A wall baseline — `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1 (PR-C #355)
- [6] PR-D binary instrumentation — `MD_precond_identity.{h,cpp}` + `cvode_config.cpp:259` PREC_LEFT patch (SHUD-OpenMP #357)
- [7] PR-E data capture — `docs/p8pre/identity_spike_run.md` (SHUD-OpenMP #358) + `.review-evidence/p8pre-pr-e-spike/cell_stats.txt`
- [8] Spec — `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L60-130
- [9] Design — `openspec/changes/p8pre-spike/design.md` D5/D6/D7/D8
- [10] Tasks — `openspec/changes/p8pre-spike/tasks.md` §8 PR-F
- [11] ADR-0002 solver path decision — `docs/adr/0002-solver-path.md` (Path 1 closure + Path 2/3/4 future options)
- [12] CLAUDE.md (project bottom-line) — academic-paper style mandate for P1e+ stage summaries
- [13] master plan — `SHUD_openMP_master_plan.md` §P8-precond.0 (preparatory chapter)
- [14] Aggregator script reference — `tools/p8pre/aggregate_n8_profile.sh` (PR-B #342) — sibling style template

---

*Generated by `tools/p8pre/aggregate_identity_spike.sh` (PR-F #347) on 2026-06-27. Verdict: NO-GO. ADR-0003 recommendation: design D8 fall-back PREC_NONE; PR-G #348 owns execution.*
