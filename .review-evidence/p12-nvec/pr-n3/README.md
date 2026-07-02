# P12-nvec PR-N3 evidence — Config E2 fixed-tree deterministic reductions + G-E4

**Issue:** #445 | **SHUD commit:** `ce4bcef` (p12-nvec) | **Outer base:** research/p12-nvec
**Verdict:** G-E4 PASS → Config E2 certified (re-baseline recorded).

## What Config E2 is

Compile-time-selectable ALTERNATIVE to the Config E Tier-1 serial reduction
overrides (`SHUD_NVEC_DETRED`). Summation reductions (dotprod / wrmsnorm[mask] /
wl2norm / l1norm + aliased `*local` + wsqrsum[mask]local + dotprodmultilocal) run
as: partition [0,NY) into fixed compile-time blocks of size **B=4096**
(independent of thread count) → accumulate each block SERIALLY IN INDEX ORDER
(carrying the `SHUD_NVEC_NOOPT` codegen pin so the vectorizer can't interleave)
→ combine block partials in a FIXED bottom-up binary tree (pure function of NY,B).
Cross-thread bitwise BY CONSTRUCTION. Non-summation reductions keep the Tier-1
serial bodies. Flag at 0 → byte-identical to Config E.

## Build commands (Config C / E / E2 / E2-b256)

```
# Config C (release default, Serial NVector)
make shud_omp
# Config E (Tier-1 serial reduction overrides)
make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1
# Config E2 (fixed-tree deterministic reductions, production B=4096)
make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1 SHUD_NVEC_DETRED=1
# Config E2 forced-small-B=256 (determinism leg — multi-level tree on keliya)
make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1 SHUD_NVEC_DETRED=1 \
     EXTRA_CXXFLAGS=-DSHUD_NVEC_DETRED_B=256
# Flag-matrix abort check (MUST fail loud):
make shud_omp SHUD_NVEC_DETRED=1        # -> $(error) requires SHUD_NVEC_HYBRID=1
```

Server build recipe (HYPRE/openblas/MPI paths) in `../pr-n2/scripts/build_matrix_bins.sh`
lineage; the PR-N3 build script is `.p12-nvec-runs/server_build_e2.sh` (login node).
Binary marker: compile-time-`#ifdef`-gated `[NVEC_HYBRID] Config E2 fixed-tree
DETERMINISTIC reduction` install banner (present only in E2 binaries: C=0, E=0,
E2=1, E2_b256=1) + runtime `NVEC config: Config E2 (...; B=4096, Neumaier=0)`.

## Thread knobs (both set per leg)

`N` is the SINGLE knob: cfg `NUM_OPENMP=N` (drives the OpenMP-NVector thread
count, hence the reduction block scheduling) AND `OMP_NUM_THREADS=N` (drives the
StrictOMP RHS via `omp_get_max_threads`). `SHUD_RHS_THREADS` UNSET in every leg
(PR-N2 discipline). Server legs add `OMP_PROC_BIND=close OMP_PLACES=cores`. The
`No of Threads` line is logged per run. (N=1 2-thread-floor footnote is N/A here —
the server matrix runs N∈{8,16} only; keliya determinism runs N∈{1,2,4,8} but at
N=1 the NVector floor does not affect the DETERMINISM SHA, only the wall, and wall
is not measured on keliya.)

## Wall metric

Runner-measured `date +%s.%N` delta around the `shud_omp` invocation (model
process wall, excludes Slurm scheduling) — identical to PR-N2. NON-profile builds
for all timed runs. 3-run median per cell, each cell on its own exclusive node.

## Directory map

| path | contents |
|---|---|
| `keliya_det/` | G-E4(1) keliya: 8 leg manifests (`*.manifest.sha`) + `SUMMARY.txt` (prod four-leg + forced-B256 multi-level + E2-off==C) |
| `server_matrix/` | G-E4(1) heihe_x4 cross-N + §4.3 wall matrix: `p12n3cell-*.out` (MARKER lines, all reps) + `E2_n8/`,`E2_n16/` (cvode_stats + rivqdown SHA per rep) |
| `a4_ulp/A4_REPORT.txt` | G-E4(2) A4 max_ulp report (keliya + heihe_x4) + Neumaier decision + basis |
| `a5report/E2_vs_C/`, `a5report/E_vs_C/` | G-E4(3) A5 verdict.md + metrics.json (tightened p12_tier2.yaml; E2 candidate + E control) |
| `a5pair_meta/` | a5-pair generator slurm log (C/E/E2 heihe_x4 full-output runs, identity + cvode) |
| `tools/a4_ulp.py` | the A4 ULP comparer (uv-run, reuses tools/a5 SHUD reader) |

## Results at a glance

- **G-E4(1) bitwise:** keliya prod N∈{1,2,4,8} all `801c2f79…`; keliya b256 N∈{1,8}
  both `8d742592…` (8 blocks/3 levels); E_n8==C_n8==`801c2f79…` (E2-off intact);
  heihe_x4 E2 N∈{8,16} identical counters (nst=6574/nfe=6728/ncfn=50/ncfl=3688/netf=0)
  + rivqdown `f70209c4…` (all 3 reps).
- **G-E4(2) A4:** keliya prod E2-vs-E max_ulp=0 (degenerate); heihe_x4 E2-vs-E
  rivqdown max_abs_diff=4.203e4 (step-sequence shift, not in-block rounding →
  Neumaier NOT enabled, justified). No threshold.
- **G-E4(3) A5:** E2-vs-C tightened → **PASS** nse=1.0000/kge=0.9999/peak_off=0/
  runoff=0.9999; control E-vs-C → PASS nse=1.0/kge=1.0. WB residual NaN==NaN
  (both `unavailable_no_mesh_metadata`, PR-Y2 informational; candidate ≤ reference
  trivially).
- **Walls (3-run median):** E_n8=594.426 E_n16=496.338 E2_n8=431.689 E2_n16=362.653;
  E2/E @N8=1.377× @N16=1.369× vs G-E3(iii) 1.3744× (delta −0.42%); E2_n16 vs
  PR-N2 C_n16=694.343 → 1.915×.

## Node map + homogeneity

Cell→node: E_n8=cn22, E_n16=cn10, E2_n8=cn11, E2_n16=cn12, a5pair=cn14 (jobs
11345–11349). All five in the PR-N2 homogeneity pool (Intel Xeon Gold 6133, 2×20
@ 2500 MHz ±<0.01%) — cn10/11/12/14/22 explicitly captured in
`../pr-n2/node_homogeneity.txt` (job 11344). Cross-node E2/E ratios not
hardware-confounded.

## Boundary audit (spec §8)

`f.cpp` / `MD_rhs_core.cpp` / `cvode_config.cpp` ZERO diff at ce4bcef (only
`Makefile`, `src/Model/MD_nvec_hybrid.{hpp,cpp}`, `src/Model/shud.cpp` changed).
See `../pr-n3/boundary_audit.txt`.

## Docs

- Verdict: `docs/p12-nvec/pr_n3_ge4_verdict.md`
- Re-baseline: `docs/p12-nvec/pr_n3_rebaseline_decision.md`
- Spec: `openspec/changes/p12-nvec/specs/tier2-det-reduction/spec.md`
