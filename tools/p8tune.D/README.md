# tools/p8tune.D — KLU pattern-only spike (PR-0)

Per openspec change [`p8tune-klu-spike`](../../openspec/changes/p8tune-klu-spike/) — pattern-only KLU spike epic. **NO** CVODE wire-up, **NO** SHUD source patch, **NO** model run, **NO** hydrology comparison. Only measures KLU fill ratio + RSS + numeric factor wall against a 4-case × 4-ordering matrix.

This directory implements **PR-0 only** (tool authoring). PR-A (server 16-cell Slurm sweep), PR-B (aggregator + ADR-0005), and PR-C (epic capstone) land in subsequent PRs per spec REQ-7.

---

## §build-environment-survey (task 1.0)

Before authoring, PR-0 task 1.0 surveyed the existing repo state to identify the build-system gap:

- **`SHUD/Makefile` `libshud.a` target**: **DID NOT EXIST** prior to PR-0.
  - Verified via `grep -nE '^(shud|libshud|.PHONY)' SHUD/Makefile` — only `shud` / `shud_omp` / `shud_asan` / `smoke_*` / `test_adjacency_fallback` targets present, no `libshud.a`.
  - **Solution**: PR-0 adds an additive `libshud.a` archive target in `SHUD/Makefile` (single hunk, NO modification to any existing target). The carve-out is documented at the target itself and in the openspec proposal §What Changes.
- **Top-level `Makefile`**: **DID NOT EXIST** prior to PR-0.
  - Verified via `find /Users/danker/Desktop/Hydro-SHUD/openMP -maxdepth 2 -name Makefile` — only `SHUD/Makefile` existed.
  - **Solution**: PR-0 adds a NEW top-level `Makefile` wiring `make shud_spike` to recurse into `SHUD/Makefile libshud.a` + `tools/p8tune.D/Makefile shud_spike`. The top-level Makefile does NOT interfere with `cd SHUD && make shud` / `cd SHUD && make shud_omp`.
- **SHUD anchor sources** (read-only; spec REQ-2/REQ-3 reference these):
  - `SHUD/src/Model/shud.cpp:118-121` — Model_Data init flow:
    ```cpp
    MD = new Model_Data(fin, fout);
    MD->loadinput();
    MD->initialize();
    ```
    (lowercase `loadinput` / `initialize`; `ReadInput` / `Initialize` do NOT exist as SHUD APIs)
  - `SHUD/src/Model/shud.cpp:139` — state vector layout: `N_VNew_Serial(NY = 3*NumEle + NumRiv + NumLake, sunctx)`
  - `SHUD/src/Model/MD_rhs_core.hpp:13-20` — `rhs_core(N_Vector Y, N_Vector DY, realtype t, ExecPolicy)` signature
  - `SHUD/src/ModelData/MD_readin.cpp:182-187` — `rivNode[]` allocation IS COMMENTED OUT (do NOT dereference `MD->rivNode[]` — would segfault)

---

## §link-line

Explicit link line (NOT pkg-config). Rationale:
- SuiteSparse Homebrew + Ubuntu apt packages do NOT ship `.pc` pkg-config files.
- ColPack CMake install ships `ColPackConfig.cmake` (CMake config), but no `.pc` file.

The spike tool's `tools/p8tune.D/Makefile` declares the link explicitly:

```
LIBS := \
  -lm \
  -lsundials_cvode \
  -lsundials_nvecserial \
  -lklu -lamd -lbtf -lcolamd -lsuitesparseconfig \
  -lColPack
```

plus platform-conditional OpenMP runtime (`-lomp` on Mac via libomp, `-lgomp` on Linux).

### Install prerequisites (Mac)

```bash
# 1. SuiteSparse via Homebrew
brew install suite-sparse

# 2. ColPack from source (no Homebrew formula)
mkdir -p $HOME/.local/src && cd $HOME/.local/src
git clone https://github.com/CSCsw/ColPack
cd ColPack && mkdir -p _build && cd _build
LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp" \
CXXFLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
CFLAGS="-I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp" \
cmake \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_INSTALL_PREFIX=$HOME/.local \
  -DBUILD_SHARED_LIBS=ON \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
  -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -L/opt/homebrew/opt/libomp/lib -lomp" \
  -DOpenMP_CXX_LIB_NAMES="omp" \
  -DCMAKE_SHARED_LINKER_FLAGS="-L/opt/homebrew/opt/libomp/lib -lomp" \
  ../build/cmake
cmake --build . -j8
cmake --install .
```

ColPack install layout (note non-standard paths on Mac):
- Headers: `$HOME/.local/include/ColPack_headers/`
- Shared libs: `$HOME/.local/lib/shared_library/libColPack.dylib`
- Static archive: `$HOME/.local/lib/archive/libColPack.a`

The `tools/p8tune.D/Makefile` defaults work for the Mac install above. Server install — see PR-A task 1.6.

### Install prerequisites (Server / Linux)

```bash
# SuiteSparse
apt install libsuitesparse-dev    # if sudo available

# Without sudo, build to /scratch/$USER/local (delegated to PR-A task 1.5)
```

---

## §output-format

### `<case>_adjacency.csc` (binary, written by `dump_adjacency`)

Little-endian, version 1:

| offset | type      | meaning                                                  |
|--------|-----------|----------------------------------------------------------|
| 0      | uint32    | magic = `0x434A4441` (`A` `D` `J` `C`)                   |
| 4      | uint32    | version = 1                                              |
| 8      | uint64    | NumEle                                                   |
| 16     | uint64    | NumRiv                                                   |
| 24     | uint64    | NumLake                                                  |
| 32     | uint64    | NumY = 3·NumEle + NumRiv + NumLake                       |
| 40     | uint64    | total_nnz                                                |
| 48     | uint64[25]| per_block_nnz (5×5 row-major: surf/unsat/gw/river/lake)  |
| 248    | uint64    | csc_col_count = NumY                                     |
| 256    | int64[NumY+1] | col_ptr (CSC: nonzero start per column)              |
| ...    | int32[total_nnz] | row_idx (CSC: row indices, sorted ascending per col) |

5-block layout matches state-vector layout per `SHUD/src/Model/shud.cpp:139`:

```
            surf       unsat      gw         river      lake
            (0..NE)    (NE..2NE)  (2NE..3NE) (3NE..3NE+NR) (3NE+NR..NY)
surf      [ J_ss       J_su       J_sg       J_sr          J_sl ]
unsat     [ J_us       J_uu       J_ug       0             0    ]
gw        [ J_gs       J_gu       J_gg       J_gr          J_gl ]
river     [ J_rs       0          J_rg       J_rr          0    ]
lake      [ J_ls       0          J_lg       0             J_ll ]
```

Determinism: `dump_adjacency` sorts CSC row indices ascending per column before write — bytewise identical output across re-runs.

### `<case>_numeric_J.bin` (binary, written by `fd_color_jacobian`)

Little-endian, version 1:

| offset | type             | meaning                                |
|--------|------------------|----------------------------------------|
| 0      | uint32           | magic = `0x4A4E444E` (`N` `D` `N` `J`) |
| 4      | uint32           | version = 1                            |
| 8      | uint64           | NumY                                   |
| 16     | uint64           | total_nnz                              |
| 24     | uint64           | n_colors (= χ from ColPack)            |
| 32     | int64[NumY+1]    | col_ptr (CSC)                          |
| ...    | int32[total_nnz] | row_idx (CSC)                          |
| ...    | double[total_nnz]| values (numeric J entries in CSC order)|

Determinism (per spec REQ-2 Scenario "FD probe determinism"): `fd_color_jacobian` pins `omp_set_num_threads(1)` at process entry — ColPack's parallel coloring + SHUD's `rhs_core` are then both single-threaded; ColPack's `SMALLEST_LAST` ordering + Curtis-Powell-Reid ε formula are deterministic. Bytewise identical output across re-runs.

### `klu_analyze_factor` stdout (text, parser-friendly KV format)

Per cell:
```
[klu] cell_summary
  case=<C> ordering=<O> btf=<B>
  NumY=<n>  Anz=<nnz>
  symbolic_lnz_est=<sym_lnz>  symbolic_unz_est=<sym_unz>  est_flops=<flops>
  nblocks=<n>  maxblock=<n>  nzoff=<n>  symmetry=<x>
  numeric_lnz=<n>  numeric_unz=<n>  max_lnz_block=<n>  max_unz_block=<n>
  fill_ratio=<x>  (numeric_lnz+unz)/Anz
  symbolic_wall_sec=<x>  numeric_factor_wall_sec=<x>
  peak_rss_bytes=<n>  (CN_NODE_RAM_BYTES=<n>)
```

OOM-as-data-point (per spec REQ-5 Scenario "OOM-as-data-point") — exit code 0, single line:
```
KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=<preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram>
```

### Pinned thresholds (PR-0)

- `tools/p8tune.D/cn_node_ram.h` — `CN_NODE_RAM_BYTES = 185528156160` (173.0 GiB, measured 2026-06-28 on cn14 via `probe_cn_ram.sbatch` job 9755)
- `tools/p8tune.D/spgmr_baseline_walls.h` — `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` (heihe_x4 N=1 maxl=5 3-rep median 1489.76s / nst 6575, pinned from epic #362 PR-D #373)

---

## §troubleshooting

### "cannot open libColPack.dylib"

ColPack install moves the shared library to `$HOME/.local/lib/shared_library/` (Mac) instead of the normal `lib/`. The Makefile adds an explicit `-L$(HOME)/.local/lib/shared_library` and `-Wl,-rpath,$(HOME)/.local/lib/shared_library`. If you installed ColPack to a non-default prefix, override `COLPACK_LIB` on the make CLI:

```bash
make shud_spike COLPACK_LIB=/my/custom/path/lib/shared_library
```

### "klu_factor failed (common.status=1)"

`KLU_SINGULAR` — the test matrix has a zero pivot. The spike tool constructs `M = I - γ·J` (γ=1) before factor; the diagonal-set step ensures every column has a non-zero diagonal. If you see this with the spike tool, the FD probe likely returned NaN/Inf for some entries — re-check `fd_color_jacobian` output for `|v|_max` reasonableness.

### "libomp not found" on Mac

```bash
brew install libomp
# then re-run `make shud_spike`
```

### χ > 30 for keliya (spec REQ-2 sanity bound)

PR-0 Mac smoke asserts χ ≤ 30 for keliya. If you see higher χ:
- Check the CSC adjacency: `dump_adjacency keliya` output `total_nnz` should be ~10000-12000 (keliya NumY=1785, average ~6 nonzeros per col).
- Verify ColPack install is the master branch (uses `PartialDistanceTwoColoring` with `COLUMN_PARTIAL_DISTANCE_TWO` mode).

### Non-deterministic numeric J binary

Re-run `fd_color_jacobian` MUST yield bytewise identical `<case>_numeric_J.bin`. If not, check:
- `omp_set_num_threads(1)` is being called at main entry (look for the line "color N/X (cols=...) probed" in stdout — re-runs should show the same column counts per color).
- ColPack version is master branch (older branches had non-deterministic SMPGCColoring tie-breaking).

---

## Files

| File                        | Role                                                      |
|-----------------------------|-----------------------------------------------------------|
| `Makefile`                  | tool build rules (explicit link line)                     |
| `dump_adjacency.cpp`        | in-process Model_Data init → 5-block CSC adjacency walk   |
| `fd_color_jacobian.cpp`     | ColPack PartialDistanceTwoColoring + Curtis-Powell-Reid FD|
| `klu_analyze_factor.cpp`    | KLU symbolic+numeric factor + RSS + OOM detection         |
| `cn_node_ram.h`             | pinned `CN_NODE_RAM_BYTES` from cn14 probe                |
| `spgmr_baseline_walls.h`    | pinned SPGMR per-step wall from epic #362 PR-D #373       |
| `probe_cn_ram.sbatch`       | one-shot cn14 `/proc/meminfo` probe                       |
| `spike_run.sh`              | per-cell driver (dump → fd_color → klu)                   |
| `verify_adjacency_keliya.py`| independent rSHUD-style Python ground-truth ref           |
| `dense_fd_cross_check.py`   | per-block finite/bounded sanity gate                      |
| `README.md`                 | this file                                                 |

## Format-version footer

§output-format format-version: **v1 (2026-06-28)**.

Per spec REQ-8 "Tool output format stability": the binary / text formats above SHALL remain backward-compatible across spike epic PR-A / PR-B; future-epic tools building on this spike (P8-tune.E full KLU integration, P9 / P10 Jacobian-aware epics) MAY extend with additional fields but SHALL NOT remove or renumber the version=1 fields.
