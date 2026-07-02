# P12-nvec PR-N2 — server scaling matrix + G-E2/G-E3 verdicts (issue #444)

Outer-only PR (SHUD submodule pointer UNCHANGED — this consumes the PR-N1
Config E binary at SHUD `d8f736c`; zero SHUD code change). heihe_x4 90-day
Slurm scaling matrix that decides the Tier-1 ROI verdict (G-E2) and the Tier-2
gate (G-E3). Verdict doc: `docs/p12-nvec/tier1_verdict.md`; decision ADR:
`docs/adr/0011-p12-nvec-tier1-verdict-and-tier2-gate.md`.

## 0. Run configuration (all pinned per spec + CLAUDE.md 三铁律)

- **Case**: heihe_x4 (`SHUD/Basins/heihe_x4`, NumEle 40046, NY≈124,395),
  90-day window (cfg.para START=1 END=91, delta=90 — NEVER regenerated, per
  CLAUDE.md 常驻 rule; the matrix runs against a PRIVATE per-cell copy of the
  12M `input/` dir + a private `output/`, so the canonical basin is never
  mutated and cells run concurrently on distinct nodes with zero cross-talk).
- **Builds** (SHUD `d8f736c`, gcc-13 on the server; HYPRE/OpenBLAS/MPI link
  paths per Makefile L460-461 + PR-N1 gcc_spot recipe):
  - Config C = `make shud_omp` → `shud_omp_C` (StrictOMP RHS + Serial NVector).
  - Config E = `make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1` →
    `shud_omp_E` (StrictOMP RHS + OpenMP element-wise NVector + serial reduction
    overrides).
  - Config E profile = `make shud_omp SHUD_USE_OPENMP_NVECTOR=1
    SHUD_NVEC_HYBRID=1 SHUD_ENABLE_PROFILE=1` → `shud_omp_E_prof` (the single
    gate-on profiler leg; adds the `t_CVODE_raw` bucket + `nvec_prof.csv`).
  - Full build recipe: `.p12-nvec-runs/matrix/build_matrix_bins.sh`. The 18 wall
    runs use the NON-profile binaries (`SHUD_ENABLE_PROFILE` NOT set); only the
    N=16 t_red cross-check leg uses `shud_omp_E_prof`.
- **N knob (single)**: `N` sets cfg `NUM_OPENMP=N` (drives Config E's
  `N_VNew_OpenMP(NY, cfg_NUM_OPENMP, sunctx)` NVector thread count) AND
  `export OMP_NUM_THREADS=N` (drives StrictOMP RHS via `omp_get_max_threads`).
  **`SHUD_RHS_THREADS` is UNSET in every cell** (falls back to
  `omp_get_max_threads()==OMP_NUM_THREADS`; logged `P1e startup:
  SHUD_RHS_THREADS=(unset)`). NUMA env: `OMP_PROC_BIND=close OMP_PLACES=cores`.
  `SHUD_LINSOL=spgmr`. This makes N a single knob for BOTH RHS and NVector
  threads (design R4; spec "N is the single cfg-numthreads knob").
- **Wall metric**: the runner-measured wall around the `shud_omp` invocation
  (`date +%s.%N` delta = model process wall, excluding Slurm scheduling/setup),
  consistent across all 18 timed runs. (sacct `Elapsed` per job is also
  available in `.p12-nvec-runs/matrix/slurm/`, but the per-run runner wall is
  the reported metric since a cell packs 3 reps + setup into one job.)
- **三铁律**: `sbatch` from `/scratch`; `--output`/`--error` on `/scratch`
  (`.p12-nvec-runs/matrix/slurm/`); all scripts + binaries on `/scratch`.
  `#SBATCH --exclusive`, partition CPU, one cell per node, **one timed run per
  node at a time** (a cell's 3 reps run sequentially within its single
  exclusive-node job → never two timed runs on a node concurrently).

### Job / node map (each cell = one exclusive node)

<!-- JOBMAP_START -->
| cell | job ID | node | binary | N |
|---|---|---|---|---|
| C_n1        | 11340 | cn14 | shud_omp_C      | 1  |
| E_n1        | 11341 | cn15 | shud_omp_E      | 1  |
| C_n8        | 11338 | cn11 | shud_omp_C      | 8  |
| E_n8        | 11339 | cn12 | shud_omp_E      | 8  |
| C_n16       | 11336 | cn22 | shud_omp_C      | 16 |
| E_n16       | 11337 | cn10 | shud_omp_E      | 16 |
| E_n16 prof  | 11342 | cn16 | shud_omp_E_prof | 16 |
<!-- JOBMAP_END -->

Node hardware homogeneity (load-bearing for cross-node C/E ratios): all 7
nodes are the identical Intel Xeon Gold 6133 @ 2.50GHz SKU (40 logical CPUs,
2500 MHz ± <0.01%), captured per-node via Slurm job 11344 →
[`node_homogeneity.txt`](node_homogeneity.txt). The exclusive-node isolation
above rules out contention; this rules out hardware confounding.

Reproduce: `sbatch --export=ALL,CONFIG={C|E},N={1|8|16}[,PROF=1,NREPS=1]
.p12-nvec-runs/matrix/matrix_cell.sbatch`.

## 1. Matrix (3-run median wall + cvode_stats)

<!-- MATRIX_TABLE_START -->
| cell | reps: walls (s) | median wall (s) | nst | nfe | ncfn | ncfl | netf |
|---|---|---|---|---|---|---|---|
| C_n1  | 1290.05 / 1288.95 / 1279.34 | **1288.953** | 6575 | 6741 | 51 | 3620 | 0 |
| C_n8  | 724.33 / 723.97 / 728.66    | **724.325**  | 6575 | 6741 | 51 | 3620 | 0 |
| C_n16 | 686.28 / 694.34 / 696.70    | **694.343**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n1  | 893.07 / 886.71 / 890.50    | **890.502**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n8  | 553.19 / 554.15 / 552.93    | **553.194**  | 6575 | 6741 | 51 | 3620 | 0 |
| E_n16 | 500.91 / 490.37 / 491.62    | **491.618**  | 6575 | 6741 | 51 | 3620 | 0 |
<!-- MATRIX_TABLE_END -->

All 18 wall runs rc==0 (sacct COMPLETED 0:0). `analyze_output.txt` is the full
`analyze_matrix.py` dump (SUMMARY: `TIER1_ADOPT ; TIER2_GO`).

Per-run artifacts (`results/<cell>/`): `run{1,2,3}.out/.err`,
`cvode_stats.run{1,2,3}.txt`, `rivqdown.run{1,2,3}.sha`. Median computed by
`analyze_matrix.py`; concatenated MARKER lines in `markers.txt`.

## 2. Bitwise cross-check (equal-N: C@N vs E@N)

Rule: identical cvode_stats counters + equal SHA of `heihe_x4.rivqdown.dat`.

<!-- BITWISE_TABLE_START -->
| N | counters equal | rivqdown SHA equal | verdict |
|---|---|---|---|
| 1  | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
| 8  | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
| 16 | YES (6575/6741/51/3620/0) | YES (`b5e4b0a2…`) | **PASS** |
<!-- BITWISE_TABLE_END -->

Bonus (cross-N determinism within each config, expected identical per Tier-1,
recorded as evidence not a gate): distinct rivqdown SHA across N∈{1,8,16} = 1
and distinct counter-tuples = 1 for BOTH Config C and Config E. A single SHA
`b5e4b0a2cf83b2a4b97d6be5b40ea7e0580d59fb6934db73a95f366f5ffc72b4` covers all 18
wall runs + the profile leg (19 runs). Per-run SHAs: `results/<cell>/rivqdown.run*.sha`.

## 3. Config E N=16 gate-on profile leg (G-E3(iii) t_red cross-check)

Config E + `SHUD_ENABLE_PROFILE=1` build, `SHUD_NVEC_PROF=1`, N=16, single run
(bitwise-identical trajectory to E_n16 — same rivqdown SHA + counters). Share
table via `uv run derive_shares.py profile_leg/nvec_prof.csv
profile_leg/profile_B0.yaml E_n16_prof`. Its reduction total_ns cross-checks
PR-N0's `t_red` (142.860 s).

<!-- PROFILE_LEG_START -->
- Config E N=16 reduction total_ns = **143,366,605,392 ns (143.367 s)**
  (`backend=hybrid`, NY=124395, nthreads=16, t_CVODE_raw=217.133 s, 276,977
  reduction calls — identical count to Config C's bitwise trajectory)
- vs PR-N0 Config C N=16 t_red = 142.860 s → delta **+0.355%** (cross-check PASS)
- reduction share of t_CVODE_raw = **66.027%** ; elementwise = **8.710%**
  (elementwise fell from Config C's 49.926% — Config E parallelizes it; the
  serial reduction now dominates the shrunken CVODE bucket → the Tier-2 target)
- artifacts: `profile_leg/{nvec_prof.csv,profile_B0.yaml,cvode_stats.txt}`;
  rivqdown SHA = `b5e4b0a2…` (same as E_n16 — profiler is bitwise-neutral)
<!-- PROFILE_LEG_END -->

## 4. Verdicts (computed by pinned rules — full detail in `docs/p12-nvec/tier1_verdict.md`)

<!-- VERDICT_START -->
- **G-E2** (TIER1_ADOPT iff speedup(E/C) ≥ 1.10 at N=8 or N=16 AND bitwise PASS):
  speedup @ N=8 = **1.3094×**, @ N=16 = **1.4124×**, bitwise PASS = **YES** →
  **TIER1_ADOPT**.
- **G-E3** (GO iff all three): (i) Tier-1=ADOPT **PASS**; (ii) reduction share
  heihe_x4 35.677% / heihe_x16 30.461% (≥10% on either) **PASS**; (iii) Amdahl
  `wall_E16/(wall_E16 − t_red·(1−1/16))` = 491.618/(491.618−142.860·0.9375) =
  **1.3744×** (≥1.15×) **PASS** → **TIER2_GO**.
- Consequence executed: issue #445 (PR-N3) `blocked` label removed + gate-values
  comment posted (TIER2_GO — Tier-2 unblocked).
<!-- VERDICT_END -->

## 5. Discipline checklist

- [x] SHUD submodule pointer unchanged (pure outer PR; Config E binary from
  PR-N1 @ SHUD `d8f736c`).
- [x] `SHUD_RHS_THREADS` unset in every cell (recorded above + in each
  `slurm/*.out` header).
- [x] 90-day truncation kept (each cell logs `delta=90`; canonical basin cfg
  never mutated — private per-cell input copy).
- [x] 三铁律: sbatch/output/scripts/binaries on `/scratch`; `--exclusive`
  nodes; one timed run per node at a time.
- [x] NON-profile binaries for the 18 wall runs; the single N=16 profile leg is
  the only `SHUD_ENABLE_PROFILE=1` run.
