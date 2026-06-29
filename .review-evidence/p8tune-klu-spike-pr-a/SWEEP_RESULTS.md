# PR-A 16-cell sweep — final results

| Run dir | Source | Cells | Notes |
|---|---|---|---|
| `cells/run-9762/` | Original 9762 sbatch array (2026-06-28 ~07:00 UTC) | NN=0-7 authoritative (keliya + heihe), NN=8-15 stale (pre-`_exit` fix; SHUD `~FloodAlert` + Model_Data dtor UB crashed before KLU stage) | Historical. NN=0-7 kept as the authoritative source for keliya + heihe — they don't trigger the dtor UB. |
| `cells/exit-fix-1782652046/` | 9794 (NN=8-11) + 9812 (NN=12-15) (2026-06-28 ~21-23 UTC, after `_exit(0)` spike-binary fix in [commit a7eb922](https://github.com/DankerMu/SHUD-OpenMP/commit/a7eb922)) | NN=8-15 authoritative (heihe_x4 + heihe_x16) | All 8 large-case cells produce per-cell verdict data. |

## 16-cell summary table

Cell-id mapping: `NN = case_idx × 4 + ordering_idx`, where
`CASE_ARR=(keliya heihe heihe_x4 heihe_x16)` and
`(ordering, btf) = [(natural,1), (amd,0), (amd,1), (colamd,1)]`
per spec REQ-4 4-combo definition.

| NN | case | ordering | btf | fill_ratio | sym_wall_s | num_factor_wall_s | peak_rss_bytes | verdict-class |
|---|---|---|---|---|---|---|---|---|
| 00 | keliya     | natural | 1 | 205.95 | 0.0002 | 0.913 | 29.5 MB | PASS (heavy fill, as expected for natural) |
| 01 | keliya     | amd     | 0 | 3.23   | 0.0011 | 0.0014 | 4.6 MB  | **best for keliya** |
| 02 | keliya     | amd     | 1 | 3.23   | 0.0012 | 0.0015 | 4.6 MB  | PASS |
| 03 | keliya     | colamd  | 1 | 4.64   | 0.0016 | 0.0021 | 4.6 MB  | PASS |
| 04 | heihe      | natural | 1 | 2304.29| 0.0028 | **1630.14** | 3.34 GB | PASS-but-pathological-wall |
| 05 | heihe      | amd     | 0 | 5.39   | 0.0128 | 0.0384 | 15.9 MB | **best for heihe** |
| 06 | heihe      | amd     | 1 | 5.39   | 0.0146 | 0.0383 | 16.0 MB | PASS |
| 07 | heihe      | colamd  | 1 | 8.96   | 0.0173 | 0.0771 | 21.1 MB | PASS |
| 08 | heihe_x4   | natural | 1 | —      | —      | TIMEOUT 30 min | — | **wall_overflow data-point** |
| 09 | heihe_x4   | amd     | 0 | 8.35   | 0.0808 | 0.500  | 93.1 MB | **best for heihe_x4** |
| 10 | heihe_x4   | amd     | 1 | 8.35   | 0.0894 | 0.494  | 93.5 MB | PASS |
| 11 | heihe_x4   | colamd  | 1 | 15.06  | 0.0926 | 1.242  | 146.2 MB | PASS |
| 12 | heihe_x16  | natural | 1 | —      | —      | klu_factor status=-4 OOM | — | **rss_overflow data-point** |
| 13 | heihe_x16  | amd     | 0 | 11.08  | 0.4546 | 4.743  | 425.4 MB | **best for heihe_x16** |
| 14 | heihe_x16  | amd     | 1 | 11.08  | 0.5063 | 4.740  | 427.5 MB | PASS |
| 15 | heihe_x16  | colamd  | 1 | 20.63  | 0.4280 | 11.708 | 712.0 MB | PASS |

Mesh sizes:
- keliya: NumEle=484 → NumY ≈ 1500
- heihe: NumEle=6335 → NumY ≈ 19500
- heihe_x4: NumEle=40046 → NumY=124395 (measured)
- heihe_x16: NumEle=160331 → NumY=485250 (measured)

## Cell-class data points (per spec REQ-5 OOM-as-data-point)

- **NN=08 (heihe_x4 natural+BTF) = wall_overflow data point**: Slurm killed at 30-min wall.
  Natural ordering on NumY=124395 produces catastrophic fill (heihe natural at 1/3 the size
  already took 1630s; heihe_x4 natural extrapolates to ~6500s = 1h48m, exceeding our 30-min
  per-cell wall budget for the first attempt). Re-running with longer wall is not informative
  because (a) the verdict-axis comparison is against SPGMR-per-step wall (0.227s), and any
  factor wall above ~10s on this size is already >>> 7σ above-threshold; (b) the AMD reorderings
  at the same NumY give 0.50s, a 12,000× speedup. The TIMEOUT serves as a "natural ordering
  is pathological at scale" data point exactly as spec REQ-5 intends OOM-as-data-point.

- **NN=12 (heihe_x16 natural+BTF) = rss_overflow data point**: `klu_factor` returned
  `common.status=-4` (KLU_OUT_OF_MEMORY) at NumY=485250 with 64G mem cap and 16.8 GB
  measured peak RSS. KLU's internal malloc for the natural-ordering L+U factor exceeded the
  remaining 47G (= 64G - 16.8G) of the cell's allocation. This is the same pattern as NN=08
  but materializes as RSS instead of wall.

## Production summary

All 4 cases produce a clean best-combo:
- All 4 cases: **AMD ordering is the best by fill_ratio** (3.23 / 5.39 / 8.35 / 11.08).
- BTF flag has zero observable effect on fill / wall / RSS for any case (NN=01 vs NN=02
  identical; NN=05 vs NN=06 identical; NN=09 vs NN=10 identical; NN=13 vs NN=14 identical).
- Natural ordering is pathological at every size beyond keliya: 205 → 2304 → TIMEOUT → OOM
  fill explosion, consistent with O(N^2) classical-elimination prediction.
- COLAMD is always worse than AMD by fill (3.23→4.64 / 5.39→8.96 / 8.35→15.06 / 11.08→20.63),
  matching the literature that COLAMD optimizes for unsymmetric LU patterns while AMD
  optimizes for symmetric or nearly-symmetric A+A^T patterns (SHUD's Jacobian here).

This data feeds PR-B's `aggregate.tsv` + 3-axis per-case verdict + ADR-0005 decision tree.

## Workarounds applied during this sweep (in scope for PR-A only)

1. **SHUD `FloodAlert` uninit-pointer fix** (SHUD upstream commit 710c00a in `openmp-baseline`
   branch): `FloodAlert::~FloodAlert()` was reading uninitialized `itype` + `fid` pointers
   on Linux + glibc; macOS lenient malloc had been masking it. Default-init to NULL +
   guard `fclose`. **SHUD pointer bumped** in `0514e3b` / `bb51554` / etc.

2. **`_exit(0)` workaround in 3 spike binaries** ([commit a7eb922](https://github.com/DankerMu/SHUD-OpenMP/commit/a7eb922)):
   `~Model_Data` destructor chain triggered a second uninit-pointer UB at NumEle > ~30k on
   Linux + glibc (heihe_x4 + heihe_x16; keliya + heihe were below the threshold). Bisection
   (server jobids 9787-9793) confirmed all 16 FD probe colors complete cleanly + J binary
   writes cleanly; crash is purely in `delete MD` → FreeData → `~SubClass`. Spike binaries are
   one-shot processes; OS reclaims memory regardless. `_exit(0)` after `fflush` sidesteps the
   destructor without changing verdict semantics. Root-cause SHUD destructor audit tracked
   separately as issue [#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386).

3. **Build env hardening** (sbatch + Makefile commits in the PR-A branch):
   - `tools/p8tune.D/Makefile` CXXFLAGS `-O2 → -O1` ([commit ba4e781](https://github.com/DankerMu/SHUD-OpenMP/commit/ba4e781)): gcc 13 -O2 emits UB at
     NumY > 100k. Spike is not perf-critical (KLU library work dominates), so -O1 is the
     safe + shippable choice.
   - `tools/p8tune.D/spike_array.sbatch`: explicit `export PATH="${HOME}/.local/bin:${PATH}"`
     ([commit 788f447](https://github.com/DankerMu/SHUD-OpenMP/commit/788f447)) so cn-node `uv` is found (compute nodes don't source
     `~/.bashrc` for non-interactive sbatch).
   - `tools/p8tune.D/spike_array.sbatch`: 4-combo matrix corrected to spec REQ-4
     `(natural,1) (amd,0) (amd,1) (colamd,1)` (commit 251c509).

4. **Server data-path migration** (operational fix, not committed since it's a one-shot
   re-deploy procedure): `/volume/data/ForcingData/CMFD2.0/` was renamed to
   `/volume/data/ForcingData/CMFD20/` between the heihe_x4 AutoSHUD deploy (2026-06-17) and
   this sweep. Re-deploy of heihe_x16 (job 9810, COMPLETED 37:57) updated `LDAS_DATA` env
   accordingly. Pre-existing heihe_x4 deployment was not affected (forcing files already
   materialized to `SHUD/Basins/heihe_x4/forcing/*.csv` before the rename).

5. **GDAL/PROJ ABI fix** (sbatch env, operational): R 4.3.1's `etc/ldpaths` prepends
   `$R_HOME/lib:/usr/local/lib` to `LD_LIBRARY_PATH`. R's bundled libproj.so.25 (PROJ 9.2.0)
   lacks `proj_crs_has_point_motion_operation` symbol that system `/usr/bin/gdalwarp` (built
   against libgdal.so.34 = GDAL 3.8.4 = PROJ 9.4.0) requires. R's `system('gdalwarp ...')`
   child inherits LD_LIBRARY_PATH and gets the old libproj first.
   Fix: `export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libproj.so.25` in the deploy sbatch
   forces the system libproj for all children. This is an operational sbatch env, not a
   committed file in the PR.
