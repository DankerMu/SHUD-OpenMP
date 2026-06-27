---
title: "Clean PREC_NONE Baseline — p8tune Mode C Reference Set"
change: p8tune-spgmr-maxl
capability: clean-prec-none-baseline
SHUD_pin: 37be0fe
outer_pin: 2582352523b2631bc7d11008a339612c7b331ccd
plan: A
plan_a_evidence: revert-of-PR-D only (cvode_config.cpp:259 + trailing whitespace)
date: 2026-06-27
author: p8tune-pr-a (Mac aggregator + cn14 keliya smoke)
status: "PASS (Plan A) — keliya smoke integrated (rivqdown SHA12 1bfe6a30856e; 15-key snapshot)"
related_docs:
  - openspec/changes/p8tune-spgmr-maxl/specs/clean-prec-none-baseline/spec.md
  - openspec/changes/p8tune-spgmr-maxl/design.md
  - openspec/changes/p8tune-spgmr-maxl/tasks.md
  - docs/p8pre/n8_profile_verdict.md
  - docs/p8pre/capstone.md
  - docs/adr/0003-precond-spike-decision.md
  - openspec/glossary.md
  - tools/p8tune/aggregate_clean_baseline.sh
  - tools/cvode_stats_diff/canonical_15_keys.yaml
---

# Clean PREC_NONE Baseline — p8tune Mode C Reference Set

This document anchors the canonical `PREC_NONE` Mode C reference set on
cleaned SHUD pin `37be0fe`. It serves as:

1. The maxl-sweep decision criterion anchor (G4 / G6 / G7 of
   `maxl-sweep-verdict` capability).
2. The corrected future-candidate gate anchor (`ncfn_candidate ≤
   ncfn_PREC_NONE_baseline`) per `p8pre-doc-state-correction`.
3. The bit-identical reference for the `spgmr-maxl-env-hook` G3 default-compat
   gate via the keliya smoke artifact (§keliya-smoke-anchor).

The 18-cell 15-key CVODE counter set is sourced via **Plan A** — direct
extraction from `docs/p8pre/n8_profile_verdict.md` §3.1 (L70-77), justified
by the codepath-equivalence proof below (§codepath-equivalence).

## §codepath-equivalence

Per spec L19-26 + tasks §2.3, executed:

```bash
cd SHUD
git diff 7a1dc8f..37be0fe -- src/Equations/cvode_config.cpp \
                              src/Equations/*precond* \
                              src/Equations/*spgmr*
```

Output (full evidence at `.review-evidence/p8tune-pr-a/codepath_equivalence_diff.txt`):

```diff
diff --git a/src/Equations/cvode_config.cpp b/src/Equations/cvode_config.cpp
index 6e774fd..a6d1ba6 100644
--- a/src/Equations/cvode_config.cpp
+++ b/src/Equations/cvode_config.cpp
@@ -256,12 +256,12 @@ void SetCVODE(void * &cvode_mem, CVRhsFn f, Model_Data *MD,  N_Vector udata, SUN
     check_flag(&flag, "CVodeSStolerances", 1);

     //    LS = SUNSPGMR(udata, 0, 0); //v3.x
-    LS = SUNLinSol_SPGMR(udata, 0, 0, sunctx);
+    LS = SUNLinSol_SPGMR(udata, PREC_NONE, 0, sunctx);
     check_flag((void *)LS, "SUNLinSol_SPGMR", 0);
-
+

     flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);
     check_flag(&flag, "CVSpilsSetLinearSolver", 1);
-
+

     flag = CVodeSetMinStep(cvode_mem, 1E-6); //Minimum time interval in cvode.dt = t(i) - t(i - 1);
     check_flag(&flag, "CVodeSetMinStep", 1);
```

**Verdict: "revert-of-PR-D only" — PASS.**

Interpretation:

- The only semantic change is `cvode_config.cpp` line 259: literal `0` was
  replaced by named constant `PREC_NONE`. Verified at the SUNDIALS 6.0.0 ABI
  level via `SHUD/InstallSundials/include/sundials/sundials_iterative.h` L58
  `enum { PREC_NONE, PREC_LEFT, PREC_RIGHT, PREC_BOTH };` — `PREC_NONE` is
  the first enum element with integer value `0`. The change is therefore a
  zero-bit-change naming cleanup.
- Two trailing-whitespace-only hunks at lines 260 / 262 (cosmetic).
- `MD_precond_identity.{h,cpp}` files NEVER existed at either pin — verified
  via `git ls-tree 7a1dc8f src/Equations/` vs `git ls-tree 37be0fe src/Equations/`:
  identical 14-file rosters with no `MD_precond_identity*` entries in either
  tree. Those files lived transiently in intermediate commit `5276167`
  ("feat(p8pre): identity preconditioner stub + cvode_config PREC_LEFT wire")
  and were deleted by `37be0fe` ("revert(p8pre): design D8 PREC_NONE
  fall-back per ADR-0003 NO-GO"). The net diff `7a1dc8f..37be0fe` is
  therefore the "revert of PR-D state to PR-B state" — exactly the
  precondition that allows Plan A reuse.

**Consequence**: Step 1 PR-B verdict §3.1 data (gathered on SHUD `7a1dc8f`)
is reusable verbatim as the cleaned-PREC_NONE baseline reference set on
SHUD `37be0fe`. The 18 cells in §raw-18-cell-table below are CVODE-counter
bitwise-equivalent to a fresh `37be0fe` re-run (Plan B), avoiding ~5.5h
server compute per design D2.

## §submit-template-provenance

Spec L25 + L40 + tasks §2.2 requires verifying Plan B fallback template
existence at `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/submit_p1e_baseline_template.sbatch`.

Server SSH probe (executed by Mac-side implementer):

```text
$ ssh -p 32099 frd_muziyao@210.77.77.22 \
    'ls -l /scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/submit_p1e_baseline_template.sbatch'
ls: cannot access '/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/submit_p1e_baseline_template.sbatch':
    No such file or directory
```

**Template missing.** Per spec L25 + L40 the derivation provenance for a
fresh template (only needed if Plan B is exercised — currently NOT exercised
because Plan A codepath-equivalence PASSED) is:

| Component | Source | Notes |
|-----------|--------|-------|
| OpenMP wrapper | `tools/run_omp.sh` (S5d.4 / issue #182) | Provides `OMP_PROC_BIND` + `OMP_PLACES` + manifest binding per b1b §s5d-data-layout-soa-numa |
| Case deployment | `docs/case_deployment_map.md` heihe_x4 server entry | NumEle=40046, residency `/scratch/.../SHUD/Basins/heihe_x4/` (never regenerate per project rule §双端 + 90-day truncation) |
| Slurm template ancestor | `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/sbatch/Cheihe_x4_N1_rep1.sbatch` (rendered form) | Same N=1 / N=2 / N=4 / N=8 × 3-rep matrix; adapt `--output / --error` to `/scratch/.../p8tune-runs/clean_prec_none_baseline/cn_*.{out,err}` per Slurm 三铁律 |
| SHUD pin override | `37be0fe` (forward-only descendant of outer `e442ce8`) | Set via submodule pointer bump prior to build |
| Build flags | `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` | Mode C: Serial NVector + StrictOMP RHS + Timer instrumentation |

Plan B re-run scope (only if codepath-divergence is later observed): 18
cells = `heihe / heihe_x4` × `N ∈ {1, 4, 8}` × `rep ∈ {1, 2, 3}`, ~5.5h
total wall (heihe ~30s × 9 + heihe_x4 ~5min × 9 + Slurm queue).

## §raw-18-cell-table

Per spec L62-67. Source: `docs/p8pre/n8_profile_verdict.md` §3.1 L70-77
(per (case, N) median across 3 reps; cross-N invariance Δ=0 strict per §3.2
ensures intra-N rep determinism = median = min = max for integer counters).

Generated by `tools/p8tune/aggregate_clean_baseline.sh --plan-a` (T1):

| case | N | rep | wall.sec (note) | rivqdown SHA12 (note) | nfe | nfeLS | nni | nli | nsetups | netf | nst | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|------|---|-----|------------------|------------------------|-----|-------|-----|-----|---------|------|-----|-----|-----|------|------|-------|-------|---------|---------|
| heihe | 1 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 1 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 1 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6943 | 12632 | 6942 | 12632 | 0 | 0 | 6698 | 0 | 0 | 7 | 85 | 277730 | 53 | 256338 | 42 |
| heihe_x4 | 1 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 1 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 1 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | 1 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | 2 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | 3 | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | 6741 | 30509 | 6740 | 30509 | 0 | 0 | 6575 | 0 | 0 | 51 | 3620 | 1617224 | 53 | 1492794 | 42 |

**Statistics per (case, N)** (median + min + max across the 3 reps, per
spec L67): all three of median, min, and max are bit-identical (Δ=0) for
all 15 integer-valued counters within each (case, N) cell. This is the
intra-N rep determinism corollary of the §3.2 cross-N Δ=0 strict invariance
(verified at L86-96 of `n8_profile_verdict.md`).

**`wall.sec` per cell**: Not captured in `n8_profile_verdict.md` §3.1 (CVODE
counter table). Per (case, N) median wall is the §4 timer-bucket table at
`n8_profile_verdict.md` L155-162 (`t_wall_total` column):

| case | N | t_wall_total (s, median across 3 reps) |
|------|---|----------------------------------------|
| heihe | 1 | 140.796860508 |
| heihe | 4 | 95.734229317 |
| heihe | 8 | 89.731664529 |
| heihe_x4 | 1 | 1412.895421879 |
| heihe_x4 | 4 | 849.704022841 |
| heihe_x4 | 8 | 743.552324144 |

**`rivqdown SHA12` per cell**: Not captured in §3.1 (CVODE-counter-only
view). Per-cell `rivqdown.dat` SHA exists in per-cell artifacts under
`/scratch/.../p8pre-runs/<case>_N<n>_rep<r>/` (verified per spec for the
keliya smoke anchor in §keliya-smoke-anchor below). The heihe/heihe_x4
18-cell `rivqdown.dat` SHAs are out-of-scope for §3.1 reuse; they would be
captured by Plan B re-run if exercised (not exercised under Plan A).

## §cross-N-invariance-table

Per spec L69-73. Generated by `tools/p8tune/aggregate_clean_baseline.sh
--plan-a` (T2). Within each case, all 15 canonical CVODE counters are
bit-identical across N ∈ {1, 4, 8} (Δ=0). Counter divergence across N
within the same case is a P0 baseline issue (B1a S4 OMP-neutrality
regression detector).

**N=4 retention rationale** (per spec L73): N=4 baseline cells are retained
as OMP-neutrality regression detectors even though the downstream sweep
matrix omits N=4 (sweep uses N ∈ {1, 8} only — 60 cells already saturate
the maxl ROI signal at the boundary; N=4 adds compute without changing the
gate verdict).

| case | key | N=1 | N=4 | N=8 | Δ (N=8 − N=1) | verdict |
|------|-----|-----|-----|-----|---------------|---------|
| heihe | nfe | 6943 | 6943 | 6943 | 0 | PASS |
| heihe | nfeLS | 12632 | 12632 | 12632 | 0 | PASS |
| heihe | nni | 6942 | 6942 | 6942 | 0 | PASS |
| heihe | nli | 12632 | 12632 | 12632 | 0 | PASS |
| heihe | nsetups | 0 | 0 | 0 | 0 | PASS |
| heihe | netf | 0 | 0 | 0 | 0 | PASS |
| heihe | nst | 6698 | 6698 | 6698 | 0 | PASS |
| heihe | npe | 0 | 0 | 0 | 0 | PASS |
| heihe | nps | 0 | 0 | 0 | 0 | PASS |
| heihe | ncfn | 7 | 7 | 7 | 0 | PASS |
| heihe | ncfl | 85 | 85 | 85 | 0 | PASS |
| heihe | lenrw | 277730 | 277730 | 277730 | 0 | PASS |
| heihe | leniw | 53 | 53 | 53 | 0 | PASS |
| heihe | lenrwLS | 256338 | 256338 | 256338 | 0 | PASS |
| heihe | leniwLS | 42 | 42 | 42 | 0 | PASS |
| heihe_x4 | nfe | 6741 | 6741 | 6741 | 0 | PASS |
| heihe_x4 | nfeLS | 30509 | 30509 | 30509 | 0 | PASS |
| heihe_x4 | nni | 6740 | 6740 | 6740 | 0 | PASS |
| heihe_x4 | nli | 30509 | 30509 | 30509 | 0 | PASS |
| heihe_x4 | nsetups | 0 | 0 | 0 | 0 | PASS |
| heihe_x4 | netf | 0 | 0 | 0 | 0 | PASS |
| heihe_x4 | nst | 6575 | 6575 | 6575 | 0 | PASS |
| heihe_x4 | npe | 0 | 0 | 0 | 0 | PASS |
| heihe_x4 | nps | 0 | 0 | 0 | 0 | PASS |
| heihe_x4 | ncfn | 51 | 51 | 51 | 0 | PASS |
| heihe_x4 | ncfl | 3620 | 3620 | 3620 | 0 | PASS |
| heihe_x4 | lenrw | 1617224 | 1617224 | 1617224 | 0 | PASS |
| heihe_x4 | leniw | 53 | 53 | 53 | 0 | PASS |
| heihe_x4 | lenrwLS | 1492794 | 1492794 | 1492794 | 0 | PASS |
| heihe_x4 | leniwLS | 42 | 42 | 42 | 0 | PASS |

**30/30 PASS.** No P0 divergence.

## §roi-ratio-table

Per spec L75-80. Generated by `tools/p8tune/aggregate_clean_baseline.sh
--plan-a` (T3).

**Authoritative source for heihe_x4 `nfeLS = 30509`** is
`docs/p8pre/n8_profile_verdict.md` §3.1 (NOT the typo values `30518` in
ADR-0003 L22 / glossary L271, or `30517` in capstone §5.1 L161 — those are
corrected via `p8pre-doc-state-correction` PR-0; the PR-0 evidence
trail also includes the §5.1 ratio cell fix `4.527 → 4.526` bundled into
PR-A per PR-0 review-correctness deferred note).

`saturation_ratio = (nli / nni) / 5` flags rows where the Krylov subspace
is approaching the SUNDIALS default `maxl = 5` cap (threshold `≥ 0.8`).

| case | N | nfe | nfeLS | nfeLS/nfe | nni | nli | nli/nni | saturation_ratio | saturation_flag |
|------|---|-----|-------|-----------|-----|-----|---------|------------------|------------------|
| heihe | 8 | 6943 | 12632 | 1.819 | 6942 | 12632 | 1.820 | 0.364 | — |
| heihe_x4 | 8 | 6741 | 30509 | 4.526 | 6740 | 30509 | 4.527 | 0.905 | **>= 0.8 (Krylov saturating)** |

The `nli/nni = 4.527` rounding for heihe_x4 in the right-hand column derives
from `30509 / 6740` (nni denominator); the `nfeLS/nfe = 4.526` ratio uses
`30509 / 6741` (nfe denominator). Both citations resolve to source
`n8_profile_verdict.md` §3.4 L131.

## §solver-failure-table

Per spec L82-86. Generated by `tools/p8tune/aggregate_clean_baseline.sh
--plan-a` (T4). Both N=1 and N=8 included to provide the gate anchor for
`maxl-sweep-verdict` G6 hard-gate per sweep N ∈ {1, 8}.

| case | N | ncfn (median) | ncfl (median) | netf (median) |
|------|---|---------------|---------------|---------------|
| heihe | 1 | 7 | 85 | 0 |
| heihe | 8 | 7 | 85 | 0 |
| heihe_x4 | 1 | 51 | 3620 | 0 |
| heihe_x4 | 8 | 51 | 3620 | 0 |

**Cleaned-PREC_NONE production floors** (the authoritative anchor for the
maxl sweep G6 gate AND the corrected future-candidate gate per PR-0):

- heihe: `ncfn = 7`, `ncfl = 85`, `netf = 0`
- heihe_x4: `ncfn = 51`, `ncfl = 3620`, `netf = 0`

**These are the true production floor anchors per Step 1 PR-B verdict §3.1,
NOT the `PREC_LEFT + identity` floors `ncfn = 6 / ncfn = 47` from Step 2
PR-F** (those latter values are negative-control anchors only per ADR-0003
§Consequences after PR-0 correction; see `docs/p8pre/identity_spike_verdict.md`
§3 for the PREC_LEFT + identity numbers' provenance).

## §keliya-smoke-anchor

**Purpose**: bit-identical anchor for `spgmr-maxl-env-hook` G3 default-compat gate (PR-C 4-way CI gate compares `unset` / `""` / `"0"` / `"5"` invocations against this snapshot).

**Build env**:

| Field | Value |
|---|---|
| Server / node | `frd_muziyao@210.77.77.22:32099` / `cn14` |
| SHUD pin | `37be0fe92f729b8849834f9fc032faf86c642d3b` (`openmp-baseline` HEAD; outer `2582352523b2631bc7d11008a339612c7b331ccd` on `baseline/p8tune`) |
| Compiler | `gcc (Ubuntu 13.3.0-6ubuntu2~24.04) 13.3.0` |
| Linker | `libsundials_cvode.so.6 → SHUD/InstallSundials/lib/libsundials_cvode.so.6` |
| Build flags | `make shud SHUD_ENABLE_PROFILE=1 -j 8` |
| Slurm job ID | `9620` (cn14, ExitCode 0:0, Elapsed 00:01:47) |

**Run command**: `unset SHUD_SPGMR_MAXL && ./shud keliya` (serial-only smoke; OMP build `make shud_omp` is separately validated via PR-C G1 build gate, not by this anchor)

**Wall time**: 88s (build + 484-NumEle keliya run inside sbatch job).

**`rivqdown.dat` SHA12 anchor**:

```
rivqdown_sha12 = 1bfe6a30856e
```

**`cvode_stats.txt` 15-key snapshot** (per `tools/cvode_stats_diff/canonical_15_keys.yaml`):

| Key | Value |
|---|---|
| nfe | 112248 |
| nfeLS | 116421 |
| nni | 112247 |
| nli | 116421 |
| nsetups | 0 |
| netf | 5 |
| nst | 110917 |
| npe | 0 |
| nps | 0 |
| ncfn | 205 |
| ncfl | 42 |
| lenrw | 23294 |
| leniw | 53 |
| lenrwLS | 21474 |
| leniwLS | 42 |

**PR-C G3 gate contract**: any PR-C build invocation matching one of `{ unset, "", "0", "5" }` MUST produce `rivqdown.dat` SHA12 = `1bfe6a30856e` AND the 15-key snapshot above bit-identical (same gcc / SUNDIALS / cn14 toolchain reproducibility). Divergence on any single key = PR-C G3 FAIL → blocks PR-C merge.

**Raw artifact**: `.review-evidence/p8tune-pr-a/keliya_smoke_artifact.txt` (local; gitignored).
**Job stdout/stderr**: `.review-evidence/p8tune-pr-a/keliya_smoke_job.out`, `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/clean_prec_none_baseline/smoke_9620.{out,err}` (server).

<!--
Orchestrator integration note (Mac-side implementer cannot fill this in
because it requires a server build + run on cn14/cn15 owned by the parallel
A2 subagent):

This section will be filled in by the orchestrator with implementer-A2's
keliya cleaned-PREC_NONE smoke artifact captured on the server. Expected
content per spec L52-56 + tasks §2.5:

  ### Build environment
  - gcc version: <e.g. 13.3.0-6ubuntu2~24.04.1>
  - SUNDIALS: libsundials_cvode.so.6 (CVODE 6.0.0, from configure script)
  - OS: <ubuntu 24.04.x>
  - SHUD pin: 37be0fe (verified via `git -C SHUD rev-parse HEAD`)
  - Build flags: `make shud SHUD_ENABLE_PROFILE=1`
  - Host: cn14 OR cn15 (per Slurm 三铁律 — not login node)

  ### Run command
      unset SHUD_SPGMR_MAXL && ./shud keliya

  ### Artifacts
  - rivqdown.dat SHA12: <12-char hex prefix of sha256sum>
  - cvode_stats.txt 15-key snapshot (per canonical_15_keys.yaml order):
        nfe       = <value>
        nfeLS     = <value>
        nni       = <value>
        nli       = <value>
        nsetups   = <value>
        netf      = <value>
        nst       = <value>
        npe       = <value>
        nps       = <value>
        ncfn      = <value>
        ncfl      = <value>
        lenrw     = <value>
        leniw     = <value>
        lenrwLS   = <value>
        leniwLS   = <value>

  ### Purpose
  Bit-identical anchor for the `spgmr-maxl-env-hook` G3 default-compat gate
  (PR-C 4-way CI gate compares unset / empty / "0" / "5" invocations to
  this baseline; all 4 SHALL produce identical SHA + 15-key counters per
  tasks §3.7 + design D0 G3).
-->

## §decision-input-table

Per spec L88-92. Generated by `tools/p8tune/aggregate_clean_baseline.sh
--plan-a` (T5).

| input | value | satisfied? | source |
|-------|-------|------------|--------|
| Hard-evidence trigger `ncfl > 0 per cell` (heihe) | ncfl=85 | YES (>0) | n8_profile_verdict.md §3.1 |
| Hard-evidence trigger `ncfl > 0 per cell` (heihe_x4) | ncfl=3620 | YES (>0) | n8_profile_verdict.md §3.1 |
| Saturation indicator `nli/nni` (heihe_x4) | 4.527 | at threshold | n8_profile_verdict.md §3.4 (cited at L131) |
| Saturation indicator `nli/nni` (heihe) | 1.820 | below threshold (case-asymmetric ROI per design D13) | n8_profile_verdict.md §3.4 (cited at L128) |

**Verdict input: "hard-evidence satisfied → full 60-cell sweep"** per
`maxl-sweep-verdict` Requirement "Sweep entry condition" (capability gate
consumed by PR-B verdict-gate doc + PR-D sweep run).

## §mode-C-tune-reference

Per spec § "Plan A CVODE codepath equivalence verification" (L19-26) + tasks
§2.10. The codepath-equivalence proof in §codepath-equivalence above
constitutes the upstream evidence that SHUD pin `37be0fe` is the canonical
**mode-C-tune** reference set for cross-`maxl` comparison.

Reference: `openspec/glossary.md` term `mode-C-tune` (per p8pre-doc-state-
correction tasks §1.5 — definition includes per-(case, maxl) anchor
structure; A3a not required cross-maxl; A4 max_ulp cross-maxl AND cross-N
within-maxl; hydrology = A4 fallback only). The §codepath-equivalence proof
ratifies that this baseline doc captures the `maxl = 0` (SUNDIALS default,
which expands to `maxl = 5`) anchor cell at SHUD `37be0fe`, and is the
authoritative input for all downstream `maxl ∈ {5, 10, 15, 20, 30}` per-cell
cross-comparison cells produced by PR-D's 60-cell server sweep.

## References

- spec: `openspec/changes/p8tune-spgmr-maxl/specs/clean-prec-none-baseline/spec.md` (5 Requirements + Scenarios L3-92)
- tasks: `openspec/changes/p8tune-spgmr-maxl/tasks.md` §2 PR-A (tasks 2.1-2.10)
- design: `openspec/changes/p8tune-spgmr-maxl/design.md` D0 (risk triage) + D2 (why establish cleaned baseline first)
- aggregator script: `tools/p8tune/aggregate_clean_baseline.sh`
- canonical 15-key contract: `tools/cvode_stats_diff/canonical_15_keys.yaml`
- upstream Step 1 PR-B verdict: `docs/p8pre/n8_profile_verdict.md` (§3.1 + §3.2 + §3.4 + §4)
- upstream Step 2 PR-F identity spike (for negative-control anchor comparison): `docs/p8pre/identity_spike_verdict.md` (§3 + §6.2)
- ADR-0003 (precond spike decision): `docs/adr/0003-precond-spike-decision.md` (§Consequences post-PR-0)
- p8pre capstone (post-PR-0 §5.1 4.526 fix): `docs/p8pre/capstone.md` §5.1 L161
- glossary terms: `openspec/glossary.md` (`mode-C-tune`, `SHUD_SPGMR_MAXL`, per PR-0)
- evidence file: `.review-evidence/p8tune-pr-a/codepath_equivalence_diff.txt`
- SUNDIALS 6.0.0 enum (`PREC_NONE = 0`): `SHUD/InstallSundials/include/sundials/sundials_iterative.h` L58
