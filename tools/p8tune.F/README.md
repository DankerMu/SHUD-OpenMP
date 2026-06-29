# tools/p8tune.F — BoomerAMG/Hypre pattern-only spike (PR-A)

Per openspec change [`p8tune-amg-spike`](../../openspec/changes/p8tune-amg-spike/) — BoomerAMG/Hypre pattern-only spike epic. **NO** CVODE wire-up, **NO** SHUD source patch, **NO** model run, **NO** hydrology comparison. Only measures BoomerAMG setup + V-cycle apply + memory + hierarchy quality against a 4-case × 4-(interp_type, coarsen_type) combo matrix.

This directory implements **PR-A only** (tool authoring + Mac local 4-combo keliya smoke). PR-B (server 16-cell Slurm sweep), PR-C (aggregator + ADR-0007), PR-D (epic capstone) land in subsequent PRs per spec.

---

## §0 Purpose

Author the BoomerAMG spike tool `boomeramg_setup_solve` to:

1. Reuse P8-tune.D's `dump_adjacency` (5-block CSC) + `fd_color_jacobian` (Curtis-Powell-Reid FD colored Jacobian) via subprocess shell-out (REQ-2 Scenario "FD-color Jacobian reuse from P8-tune.D"). P8-tune.D source is frozen per REQ-1.
2. Wrap the numeric J binary into Hypre `IJMatrix` (`HYPRE_PARCSR` object type) as the BDF-equivalent test matrix M = I − γ·J (γ=1.0, mirror P8-tune.D `klu_analyze_factor.cpp` pattern).
3. Drive BoomerAMG `Setup` (timed) + `Solve` (timed, averaged over `--n-solve` iterations) with per-cell (interp_type, coarsen_type) combination.
4. Emit REQ-4 `cell_summary` KV block to stdout: `CELL_SUMMARY_BEGIN` ... `CELL_SUMMARY_END` with 5 KV lines (case + interp/coarsen, walls + RSS, complexity + residual reduction, verdict_class, version pins).

---

## §1 Hypre install + version reconciliation

### Mac (local PR-A 4-combo smoke)

```bash
brew install hypre
brew info hypre
# → hypre: stable 3.1.0 (bottled), HEAD
# → Installed (on request) From: Homebrew/homebrew-core
```

Brew currently ships **Hypre 3.1.0**. The spec design doc + issue title reference **Hypre 2.30.0**. The BoomerAMG API surface used here (`Create`/`Setup`/`Solve`/`SetInterpType`/`SetCoarsenType`/`GetNumIterations`/`GetCumNnzAP`/`GetFinalRelativeResidualNorm`) is **stable across 2.30 → 3.x**; 3.x adds GPU APIs without breaking the CPU BoomerAMG contract. Using 3.1.0 is acceptable; fallback would be a source-build of 2.30.0 (`git clone https://github.com/hypre-space/hypre && cd src && ./configure --prefix=… && make -j8 install`, ~20-40 min).

Version differences noted in code:
- `HYPRE_BIGINT 1` is the default brew build (`HYPRE_BigInt = long long int`). Our `boomeramg_setup_solve.cpp` uses `HYPRE_BigInt` directly for row/col indices.
- `HYPRE_Initialize()` is **Required** in 3.x (per `HYPRE_utilities.h` docstring). Called once before any other Hypre API; `HYPRE_Finalize()` at exit.
- MPI (`open-mpi`) is a runtime dependency of the brew Hypre dylib (see §2 symbol verification). We `MPI_Init` / `MPI_Finalize` with `MPI_COMM_SELF` for single-process spike.
- `HYPRE_BoomerAMGGetCycleComplexity` and `HYPRE_BoomerAMGGetOperatorComplexity` are **NOT in Hypre 3.1.0 public headers** (confirmed by `grep` in `HYPRE_parcsr_ls.h` + `nm libHYPRE.dylib`). We use `HYPRE_BoomerAMGSetCumNnzAP(solver, 1.0)` before `Setup` to enable nnz-A-and-P tracking, then `HYPRE_BoomerAMGGetCumNnzAP(solver, &cum_nnz_AP)` after `Setup`. `operator_complexity = cum_nnz_AP / nnz_A_fine` (Hypre canonical definition). `cycle_complexity = 2.0 × operator_complexity` (V-cycle standard estimate: pre + post smoothing on each level). See §6 below.

### Server (PR-B)

Blocked per follow-up [#401](https://github.com/DankerMu/SHUD-OpenMP/issues/401) (apt SuiteSparse + ColPack + valgrind + Hypre install). PR-B will resolve.

---

## §2 Symbol verification

```bash
# Header presence
ls /opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE.h
ls /opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE_parcsr_ls.h
ls /opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE_IJ_mv.h

# Symbol presence (BoomerAMG core API).
# Replace generic `grep | head -10` (which returned the first 10 alphabetically
# and didn't actually prove the required symbols were present — round-1 M11)
# with explicit regex enumerating the 11 symbols we link against. Should
# print exactly 11:
nm /opt/homebrew/opt/hypre/lib/libHYPRE.dylib | \
  grep -E '_HYPRE_BoomerAMG(Create|Destroy|Setup|Solve|SetInterpType|SetCoarsenType|SetMaxIter|SetTol|GetFinalRelativeResidualNorm|SetCumNnzAP|GetCumNnzAP)$' | \
  sort | wc -l
# expected: 11

# Plus HYPRE_Version (round-1 M5 runtime probe) — should print 1:
nm /opt/homebrew/opt/hypre/lib/libHYPRE.dylib | grep -E '_HYPRE_Version$' | wc -l
# expected: 1

# NOTE: HYPRE_BoomerAMGGetNumLevels is NOT in the Hypre 3.1.0 public
# dylib surface (confirmed: only HYPRE_BoomerAMGGetMaxLevels and
# HYPRE_BoomerAMGGetSmoothNumLevels are exported). The spec REQ-4
# AMG_SETUP_DIVERGE Scenario "num_levels == 0" trigger is implemented
# via the internal `hypre_ParAMGDataNumLevels` accessor macro on the
# solver opaque pointer (included from <_hypre_parcsr_ls.h>). PR-B may
# need to re-verify against a future Hypre version that adds the
# public getter, but the macro is stable across 3.x.

# Runtime dependency check
otool -L /opt/homebrew/opt/hypre/lib/libHYPRE.dylib
# expected:
#   /opt/homebrew/opt/open-mpi/lib/libmpi.40.dylib
#   /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib
```

If any required symbol is missing, fall back to Hypre 2.30.0 source build (see §1).

---

## §3 4-combo encoding table

The 16-cell sweep matrix (PR-B) factors as 4 cases × 4 (interp_type, coarsen_type) combos. PR-A keliya smoke runs all 4 combos against the small case to verify the binary + KV emission paths.

| Combo | interp_type | coarsen_type | description |
|-------|-------------|--------------|-------------|
| 0 | 6 (classical-extended) | 8 (HMIS) | Hypre default, robust baseline |
| 1 | 14 (extended+i) | 10 (HMIS) | aggressive interpolation (setup ↑ apply ↓) |
| 2 | 6 (classical-extended) | 21 (CGC) | alt-coarsening (sensitivity test) |
| 3 | 8 (standard) | 8 (HMIS) | fallback baseline if Combo 0 unstable |

(Mirror spec REQ-4 Scenario "4 case × 4 combo matrix" combo definitions.)

---

## §4 cell_summary KV schema

Emitted to stdout on every cell completion (PASS or marker class). Fixed line order for aggregator parser stability (REQ-4 Scenario "cell_summary KV block schema"):

```
CELL_SUMMARY_BEGIN
case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>
setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>
cycle_complexity=<CC> operator_complexity=<OC> residual_reduction_v1=<R1>
verdict_class=<PASS|AMG_OOM|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_WALL_OVERFLOW>
hypre_version=<HV> colpack_version=<CV> shud_pin=<SHA>
CELL_SUMMARY_END
```

Marker emission convention (REQ-4 marker-vs-class naming):
- Stdout marker line uses verb form: `MARKER:AMG_OOM_DETECTED` (suffix `_DETECTED` indicates emission action)
- KV value omits the suffix: `verdict_class=AMG_OOM`
- PASS has no marker emission, only KV value `verdict_class=PASS`

---

## §5 Build + smoke commands

### Build prerequisites

```bash
# 1. SHUD libshud.a (built once by P8-tune.D PR-0 carve-out)
cd /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD
make libshud.a -j4

# 2. P8-tune.D spike binaries (dump_adjacency + fd_color_jacobian)
cd /Users/danker/Desktop/Hydro-SHUD/openMP
make -C tools/p8tune.D all

# 3. Hypre + open-mpi + openblas + libomp via brew (one-time)
brew install hypre  # pulls open-mpi, openblas, libomp transitively
```

### Build P8-tune.F tool + symlinks

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP
make -C tools/p8tune.F clean
make -C tools/p8tune.F all
# expected: boomeramg_setup_solve + 2 symlinks created in tools/p8tune.F/

# Verify symlinks resolve
ls -la tools/p8tune.F/{dump_adjacency,fd_color_jacobian}
# expected: `dump_adjacency -> ../p8tune.D/dump_adjacency`
#           `fd_color_jacobian -> ../p8tune.D/fd_color_jacobian`
```

### Invocation convention

All 3 binaries (`dump_adjacency`, `fd_color_jacobian`, `boomeramg_setup_solve`) use a **default `--basin-root=../../SHUD/Basins`** that assumes the CWD is `tools/p8tune.F/` (or `tools/p8tune.D/`). Each binary **chdirs internally** into `<basin_root>/<case>/` before SHUD's relative-path IO + writes output files (adjacency CSC + numeric J binary) into that same directory. Run all three from `tools/p8tune.F/`:

### Pre-flight: produce adjacency + numeric J binaries (keliya)

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.F
rm -f ../../SHUD/Basins/keliya/keliya_adjacency.csc ../../SHUD/Basins/keliya/keliya_numeric_J.bin

# Shell out to symlinked p8tune.D binaries
./dump_adjacency --case keliya
./fd_color_jacobian --case keliya

# Verify outputs (files live in SHUD/Basins/keliya/ because each binary chdirs)
ls -la ../../SHUD/Basins/keliya/{keliya_adjacency.csc,keliya_numeric_J.bin}
```

### 4-combo smoke (keliya)

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.F
mkdir -p output

# Combo 0: interp=6  coarsen=8   (Hypre default)
# Combo 1: interp=14 coarsen=10  (aggressive)
# Combo 2: interp=6  coarsen=21  (alt-coarsening)
# Combo 3: interp=8  coarsen=8   (fallback)
for c in 0:6:8 1:14:10 2:6:21 3:8:8; do
  idx="${c%%:*}"
  rest="${c#*:}"
  interp="${rest%%:*}"
  coarsen="${rest#*:}"
  echo "=== Combo $idx (interp=$interp coarsen=$coarsen) ==="
  ./boomeramg_setup_solve --case keliya \
    --interp-type=$interp --coarsen-type=$coarsen \
    > output/cell-${idx}.log 2>&1
  echo "Exit: $?"
done
```

Note: `boomeramg_setup_solve` does **NOT** chdir (unlike the p8tune.D binaries) — it only reads the JBin file at `<basin_root>/<case>/<prefix>_numeric_J.bin`. Pre-flight is idempotent (skipped if the .csc and .bin already exist non-empty).

---

## §6 Acceptance

PR-A landed when all 4 combos:
1. Exit 0 (`echo $?` after each invocation)
2. Each `cell-<N>.log` contains `CELL_SUMMARY_BEGIN` + `CELL_SUMMARY_END` block
3. Each `cell-<N>.log` contains `verdict_class=PASS`
4. Each `cell-<N>.log` has `setup_wall_sec`, `apply_wall_sec`, `peak_rss_bytes` all non-zero

Quick grep verification:

```bash
for f in tools/p8tune.F/output/cell-*.log; do
  echo "=== $f ==="
  grep -E 'CELL_SUMMARY_(BEGIN|END)|verdict_class' "$f" | head -10
done
```

### Hypre version reconciliation (recap)

Spec asks for Hypre 2.30.0; brew ships 3.1.0. Decision: use 3.1.0 + document. Rationale:
- All required BoomerAMG public API symbols present (verified per §2)
- `HYPRE_BIGINT 1` (long long index) is compatible with our `HYPRE_BigInt` row/col usage
- `HYPRE_Initialize`/`HYPRE_Finalize` honored
- The complexity-getter gap (`GetCycleComplexity`/`GetOperatorComplexity` absent in 3.1.0 public API) is bridged via `SetCumNnzAP`/`GetCumNnzAP` → canonical operator complexity (Hypre design: `op_complexity = sum(nnz_A_level_l) / nnz_A_fine`); cycle complexity reported as `2 × op_complexity` per V-cycle pre+post smoothing standard convention
- The hierarchy-size accessor gap (`HYPRE_BoomerAMGGetNumLevels` absent in 3.1.0 public dylib) is bridged via the internal `hypre_ParAMGDataNumLevels((hypre_ParAMGData *)solver)` macro from `<_hypre_parcsr_ls.h>` — required by spec REQ-4 AMG_SETUP_DIVERGE Scenario "num_levels == 0" trigger (round-1 H1)
- Fallback if 3.1.0 binary surface issues surface in PR-B: source-build Hypre 2.30.0 (~20-40 min)

### cycle_complexity axis-independence caveat (round-1 H3 disclosure)

`cycle_complexity = 2 × operator_complexity` is a V-cycle estimate (1 pre-smoothing + 1 post-smoothing pass per level, weight=1), **NOT** an independent measurement. Hypre 3.1.0 has no `HYPRE_BoomerAMGGetCycleComplexity` public getter, so the spec's Axis 4 (`cycle_complexity < 1.5`) and Axis 5 (`operator_complexity < 2.0`) thresholds are **mechanically linked**: Axis 4 trips iff Axis 5 trips at `op_complex > 0.75`. PR-C ADR-0007 §Discussion + §Limitations MUST disclose this — Axis 4 and Axis 5 are NOT independent diagnostics under the Hypre-3.1.0-API constraint. The PR-C aggregator may need to amend the Axis 4 threshold (or drop Axis 4 as redundant) to reflect the 2× linkage. For W-cycles or aggressive coarsening the 2× estimate systematically under-counts the true cycle complexity — a Hypre source-build with the 2.30.0 public getter (or a Hypre 3.x version that re-introduces the getter) would unlock independent measurement.

---

## §7 Marker emission infrastructure

The binary implements all 4 marker paths (REQ-4 Scenarios 2-5) but at keliya scale (NumY=1500) only the PASS path triggers. Marker paths are exercised by PR-B at larger scales:

| Marker | Trigger | verdict_class |
|--------|---------|---------------|
| `MARKER:AMG_SETUP_DIVERGE_DETECTED` | `HYPRE_BoomerAMGSetup` returns non-zero | `AMG_SETUP_DIVERGE` |
| `MARKER:AMG_SOLVE_DIVERGE_DETECTED` | `HYPRE_BoomerAMGSolve` returns non-zero OR `GetFinalRelativeResidualNorm` > 1.0 | `AMG_SOLVE_DIVERGE` |
| `MARKER:AMG_OOM_DETECTED` | `std::bad_alloc` thrown OR `peak_rss > 8 GiB` (Mac pin; PR-B replaces with `CN_NODE_RAM_BYTES × 0.95`) | `AMG_OOM` |
| `MARKER:AMG_WALL_OVERFLOW_DETECTED` | SIGTERM from Slurm wall budget (PR-B `--time=08:00:00` overflow) | `AMG_WALL_OVERFLOW` |

All marker paths emit cell_summary KV + exit 0 (valid data point per REQ-4 design R2).

---

## §8 Refs

- [openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md) — REQ-1 / REQ-2 / REQ-4
- [openspec/changes/p8tune-amg-spike/tasks.md §2 PR-A](../../openspec/changes/p8tune-amg-spike/tasks.md) — subtasks 2.1-2.14
- [openspec/changes/p8tune-amg-spike/design.md](../../openspec/changes/p8tune-amg-spike/design.md) — design decisions D1-D8
- [docs/p8tune/p8tune_f_work_plan.md](../../docs/p8tune/p8tune_f_work_plan.md) §3.2 — PR-A scope detail
- [tools/p8tune.D/](../p8tune.D/) — KLU spike template (Makefile + JBin format + CN_NODE_RAM_BYTES probe)
- [Hypre BoomerAMG user manual](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html) — interp_type / coarsen_type semantics
