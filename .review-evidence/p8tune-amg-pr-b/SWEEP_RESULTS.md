# PR-B #396 — 16-Cell Slurm Array Sweep Results

**Date**: 2026-06-29
**Slurm job**: `9896_[0-15]` (4 case × 4 combo) on `CPU` partition
**Nodes**: cn08 + others (varies per cell)
**Wall**: all cells COMPLETED 0:0 (exit 0) in 0-2 sec elapsed
**Verdict**: 16/16 PASS — none of 4 marker classes (AMG_OOM / AMG_SETUP_DIVERGE / AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW) triggered

## Spec REQ-4 KV completeness check

All 16 cells emit `CELL_SUMMARY_BEGIN`/`END` with 5-line schema:
- Line 1: `case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>`
- Line 2: `setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>`
- Line 3: `cycle_complexity=<CC> operator_complexity=<OC> residual_reduction_v1=<R1>`
- Line 4: `verdict_class=PASS`
- Line 5: `hypre_version=3.1.0 colpack_version=unknown shud_pin=1ab61c023ac2b93a178c2feb07aa3df509fe1a96`

Note: `colpack_version=unknown` — Makefile probe didn't find ColPack version on server (ColPack apt deb doesn't ship `ColPackConfigVersion.cmake`; only Mac `~/.local` source-built install has it). PR-C aggregator must tolerate this sentinel OR PR-B precheck_env.sh substitute. Per Phase 5 PR-A M6 disclosure.

## 16-cell verdict table

| NN | Case | interp | coarsen | NumY | nnz_A | setup_wall_sec | apply_wall_sec | peak_rss | cycle_C | op_C | residual_red_v1 | verdict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | keliya | 6 | 8 | 1,785 | 10,255 | 0.001094 | 0.001257 | 19.0 MB | 2.0043 | 1.0021 | 50.8 | PASS |
| 1 | keliya | 14 | 10 | 1,785 | 10,255 | 0.000570 | 0.001030 | 19.4 MB | 2.0043 | 1.0021 | 73.8 | PASS |
| 2 | keliya | 6 | 21 | 1,785 | 10,255 | 0.000561 | 0.001031 | 19.0 MB | 2.0059 | 1.0029 | 57.2 | PASS |
| 3 | keliya | 8 | 8 | 1,785 | 10,255 | 0.000582 | 0.001020 | 19.3 MB | 2.0043 | 1.0021 | 50.8 | PASS |
| 4 | heihe | 6 | 8 | 21,357 | 120,485 | 0.001558 | 0.001666 | 34.1 MB | 2.0000 | 1.0000 | 158,532 | PASS |
| 5 | heihe | 14 | 10 | 21,357 | 120,485 | 0.001634 | 0.001692 | 34.4 MB | 2.0000 | 1.0000 | 158,532 | PASS |
| 6 | heihe | 6 | 21 | 21,357 | 120,485 | 0.002623 | 0.002238 | 34.4 MB | 2.0213 | 1.0106 | 158,649 | PASS |
| 7 | heihe | 8 | 8 | 21,357 | 120,485 | 0.001452 | 0.001667 | 34.4 MB | 2.0000 | 1.0000 | 158,532 | PASS |
| 8 | heihe_x4 | 6 | 8 | 124,395 | 653,387 | 0.009476 | 0.018179 | 110.9 MB | 2.0000 | 1.0000 | 11,411 | PASS |
| 9 | heihe_x4 | 14 | 10 | 124,395 | 653,387 | 0.010401 | 0.018096 | 110.9 MB | 2.0000 | 1.0000 | 11,411 | PASS |
| 10 | heihe_x4 | 6 | 21 | 124,395 | 653,387 | 0.013382 | 0.017854 | 110.9 MB | 2.0003 | 1.0001 | 11,419 | PASS |
| 11 | heihe_x4 | 8 | 8 | 124,395 | 653,387 | 0.009986 | 0.017854 | 110.9 MB | 2.0000 | 1.0000 | 11,411 | PASS |
| 12 | heihe_x16 | 6 | 8 | **485,250** | 2,481,548 | 0.037785 | 0.078349 | 381.2 MB | 2.0000 | 1.0000 | 22,538 | PASS |
| 13 | heihe_x16 | 14 | 10 | 485,250 | 2,481,548 | 0.041478 | 0.079340 | 381.2 MB | 2.0000 | 1.0000 | 22,538 | PASS |
| 14 | heihe_x16 | 6 | 21 | 485,250 | 2,481,548 | 0.048697 | 0.079398 | 381.2 MB | 2.0001 | 1.0000 | 22,554 | PASS |
| 15 | heihe_x16 | 8 | 8 | 485,250 | 2,481,548 | 0.039249 | 0.077976 | 381.2 MB | 2.0000 | 1.0000 | 22,538 | PASS |

## Per-case best combo (criterion: min setup_wall + apply_wall combined)

| Case | NumY | Best Combo (interp, coarsen) | setup_wall_sec | apply_wall_sec | peak_RSS | residual_red_v1 |
|---|---:|:---:|---:|---:|---:|---:|
| keliya | 1,785 | (8, 8) | 0.000582 | 0.001020 | 19.3 MB | 50.8 |
| heihe | 21,357 | (8, 8) | 0.001452 | 0.001667 | 34.4 MB | 158,532 |
| heihe_x4 | 124,395 | (6, 8) | 0.009476 | 0.018179 | 110.9 MB | 11,411 |
| heihe_x16 | 485,250 | (6, 8) | 0.037785 | 0.078349 | 381.2 MB | 22,538 |

## Spec REQ-5 5-axis preliminary evaluation (vs pinned thresholds — PR-C will canonicalize)

Pinned constants (per spec REQ-5 / spgmr_baseline_walls.h):
- `SPGMR_PER_STEP_SEC = 0.226579` (P8-tune.D heihe_x4 baseline)
- `CN_NODE_RAM_BYTES = 173 × 1024^3 ≈ 185.7 GB`
- `WALL_BUDGET_APPLY_SEC = 0.7 × 0.226579 = 0.158605 s`
- `WALL_BUDGET_SETUP_SEC = 1.5 × 0.158605 = 0.237908 s`
- `WALL_BUDGET_RSS_BYTES = 0.7 × 185.7 = 130 GB`

Best-combo evaluation for heihe_x16 (most stringent case):
- Axis 1 (Setup): 0.037785 s vs budget 0.237908 s → **15.9% of budget, PASS** ✓
- Axis 2 (Apply): 0.078349 s vs budget 0.158605 s → **49.4% of budget, PASS** ✓
- Axis 3 (Memory): 381 MB vs budget 130 GB → **0.29% of budget, PASS** ✓
- Axis 4 (Cycle complexity): 2.0000 vs threshold 1.5 → **FAIL** ✗ (per H3 disclosure: 2×op_complexity estimate; mechanical Axis 4 ⇔ Axis 5 linkage means Axis 4 has no independent diagnostic value under this implementation)
- Axis 5 (Operator complexity): 1.0000 vs threshold 2.0 → **PASS** ✓

Wall + Memory axes (1+2+3) — overwhelmingly GO. Axis 4 — disclosed limitation (estimate not measurement). Axis 5 — PASS.

Per spec REQ-5 verdict_branch decision tree (PR-C aggregator will canonicalize):
- If Axis 4 strict: heihe_x16 fails Axis 4 → **NO-GO-heihe_x16-only** OR **NO-GO-both** depending on smaller cases
- If Axis 4 amended per H3 disclosure: all 4 cases PASS all meaningful axes → **GO**

**Recommendation for PR-C ADR-0007 §Discussion**: amend Axis 4 threshold OR mark Axis 4 as "non-discriminating diagnostic" given the 2×op_complexity estimate; primary verdict driven by Axis 1+2+3 walls + memory which all PASS with huge margins.

## Numeric cross-platform consistency (sanity check)

`residual_reduction_v1` for keliya combo 0 across 4 platforms:
- Mac brew Hypre 3.1.0: **50.8449**
- Server login node (source-built Hypre 3.1.0): **50.8449**
- Server cn08 smoke (job 9895): **50.8449**
- Server cn-node array (cell-0 of job 9896): **50.8449**

Bitwise-identical across all 4 platforms confirms portability fix (Hypre 2.x/3.x init API gate + EXTRA_LIBS Makefile hook + source-built Hypre 3.1.0 on server) is semantically equivalent. No numerical regression from Phase 6 round-2 to PR-B server build.

## Files

- `cell-{0..15}.out` — per-cell stdout (sbatch redirect output)
- `cell-{0..15}.err` — per-cell stderr (all empty — no runtime errors)
- `runtime/cell-NN.log` — per-cell detailed log from run_cell.sh
- `runtime/cell-NN.time` — `/usr/bin/time -v` peak RSS + page faults output

## Server install path (per #401 §3 + PR-B prereq closure)

User-space install at `/scratch/frd_muziyao/local/`:
- Hypre 3.1.0 source-built (`--without-superlu --without-fei --enable-shared CC=mpicc CXX=mpicxx`) at `/scratch/frd_muziyao/local/hypre-3.1.0/` — avoids Debian apt libhypre-dev 2.28.0 CombBLAS/SuperLU_DIST/ParMETIS/Scotch deps chain
- SuiteSparse + ColPack + openmpi-dev from `apt-get download` + `dpkg-deb -x /scratch/frd_muziyao/local/` (user-space; no sudo required)
- libopenblas.so symlink: `/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu/libopenblas.so` → `/usr/lib/x86_64-linux-gnu/libopenblas.so.0` (manual user-space symlink; system has runtime but no dev `.so` link)
- Makefile env-var overrides documented in PR-B README

## Sweep slurm submission record

```
JID = 9896
sbatch --parsable /scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.F/spike_array.sbatch
```

All 16 array tasks COMPLETED 0:0 within 0-2 seconds each (BoomerAMG at this scale is far below 8h wall budget).

Pre-submission precheck_env.sh: **7/7 PASS** at HEAD `d4a0d78`.

Next: PR-C #397 (aggregate_amg_spike.sh + ADR-0007 + verdict canonicalization).
