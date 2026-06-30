# G0-4 AMG integrated smoke 4-cell evidence — Slurm job 10014

PR-A #410 deferral fulfillment per `afb09f7` merge commitment. Slurm array
job 10014 (`tools/p8tune.G0/smoke_array.sbatch`) on server cn23 with
Phase-6-fix scripts.

## Per-cell results

| task | cell | exit | wall (s) | nst | verdict_class | notes |
|---|---|---|---|---|---|---|
| 0 | keliya | 0 | 323 | 124601 | **AMG_OK** | NumEle=484, 90-day SHORT |
| 1 | xinanjiang_upstream | 0 | 90 | 33223 | **AMG_OK** | NumEle=801, basin→project mapping (`xinanjiang_upstream` → `xinanjiang`) verified end-to-end |
| 2 | heihe_x4 | 0 | 23660 | 254756 | **AMG_OK** | NumEle=40046, completed within 8h budget |
| 3 | heihe_x16 | non-zero | ~28800 | n/a | **TIMEOUT** | NumEle=160331, Slurm SIGTERM at 8h + SIGKILL +30s; runner SIGTERM trap could not complete `MARKER:AMG_WALL_OVERFLOW_DETECTED` emission before SIGKILL |

## G0-4 per spec REQ semantics

Per `openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md`
G0-4 PASS criterion: each cell must `exit 0` AND emit `verdict_class=AMG_OK`
AND emit no divergence MARKER.

- keliya / xinanjiang_upstream / heihe_x4: **PASS** (3/4)
- heihe_x16: **FAIL** (Slurm-induced WALL_OVERFLOW, not runner-trap WALL_OVERFLOW)

G0-4 verdict (per spec): cells PASS-count = 3/4, FAIL-count = 1/4. Final G0-4
verdict declaration is **PR-C scope** (aggregator + verdict doc).

## Wall-signal preliminary observation (full evaluation = PR-B G0-5)

Per-step wall comparison vs SPGMR baselines (hot-patched at `c994a73`):

| cell | SPGMR per_step (s) | AMG per_step (s) | AMG nst | SPGMR nst | AMG/SPGMR per_step ratio | AMG/SPGMR nst ratio | AMG/SPGMR total wall ratio |
|---|---|---|---|---|---|---|---|
| heihe_x4 | 0.238369 | 0.0928 (23660s / 254756) | 254756 | 6572 | **0.39×** (AMG faster per step) | **38.8×** (AMG needs many more steps) | **15.1×** (AMG total wall MUCH WORSE) |
| heihe_x16 | 0.952489 | n/a (TIMEOUT) | n/a | 6556 | n/a | n/a | n/a |

**Preliminary observation**: AMG per-step IS faster than SPGMR on heihe_x4
(0.39× ratio is meaningful), but AMG's higher CVODE rejection rate (`ncfn`
control-failure counter) forces ~39× more integration steps, making total
wall ~15× WORSE. heihe_x16 hit 8h budget without completing 90-day SHORT.

This is the data PR-B G0-5 wall-signal benchmark will consume formally.
G0-5 PASS criterion = at least one case improves vs SPGMR. heihe_x4 does
NOT improve (15× worse total); heihe_x16 evidence inconclusive (TIMEOUT).

**G0-5 preliminary inference**: AMG not beneficial for these SHUD
hydrology matrix shapes at heihe_x4/x16 scale. Final G0-5 verdict
declaration is PR-B/C scope.

## Files

- `cell-<cell>.out`: stdout (model loop progress + AMG telemetry markers + CVODE final statistics + cell_summary KV block)
- `cell-<cell>.err`: stderr (NUMA / SHUD warnings / OMP messages)
- `precheck.log`: precheck_env.sh 7-condition gate output (TASK_ID=0 only per smoke_array.sbatch design)

## Slurm provenance

| field | value |
|---|---|
| job id | 10014 (array 0-3) |
| node | cn23 (single node, all 4 tasks share node) |
| partition | CPU |
| sbatch script | `/scratch/frd_muziyao/SHUD-OpenMP/tools/p8tune.G0/smoke_array.sbatch` (Phase-6-fix) |
| SHUD pin | `2ec4c00` on `openmp-baseline` (PR-A merge state) |
| outer commit | `860c4d0` (PR-A HEAD merged as `afb09f7`) |
| `SHUD_LINSOL` | `amg` |
| `OMP_NUM_THREADS` | 1 |
| `HYPRE_LIBDIR` | `/scratch/frd_muziyao/local/hypre-3.1.0/lib` |
| Hypre runtime | 3.1.0 (from-source) per `MARKER:AMG_TELEMETRY_REAL hypre_release=30100` |
