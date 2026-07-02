# PR-D1 bitwise-gate + self-check evidence (P11-osc, issue #434)

SHUD instrumentation: `SHUD_DIAG_DT` per-interval CVODE trace +
`SHUD_DIAG_OSC` state-delta flip counters. Both env gates are **strict `=1`
string predicates** (presence-only or `=0` must NOT enable). All evidence is
local, case **keliya** (deployed at `SHUD/Basins/keliya`, run from the basin
dir as `./shud keliya`, ~30 s).

## Build

`cd SHUD && make clean && make shud` — green, **zero new warnings**
(104 warning lines pre- and post-change; all pre-existing `sprintf`
deprecation + one `ld` search-path). `make shud_omp` (Config C) also green.

## Bitwise gate — the model-output SHA manifest

`sha/*.sha` are `shasum -a256` manifests of every file in
`SHUD/Basins/keliya/output/keliya.out/` **except**:
- `diag_*.csv` — the instrumentation artifacts (only exist when enabled);
- `*.time.csv` — wall-clock profiling log (`CPUTime_s`/`WallTime_s` columns
  vary run-to-run by construction; two clean *pre-change* runs already
  differ only here — see `leg-a-prerun-with-timecsv.sha` vs the baseline —
  so it is a timing artifact, never a model result, and is excluded from the
  neutrality gate exactly like the diag CSVs).

The baseline (leg a) was established by **two** clean pre-change runs whose
13 model-output files are byte-identical (determinism confirmed before the
manifest was trusted as the reference).

## Four legs (a/b/c/d)

| leg | condition | diag CSVs | model-output SHA | verdict |
|---|---|---|---|---|
| **a** | pre-change baseline (2 runs, deterministic) | n/a | reference (`leg-a-baseline.sha`, 13 files) | PASS — deterministic |
| **b** | post-change, both env **unset** | none created | == baseline | **PASS** |
| **c** | post-change, `SHUD_DIAG_DT=0 SHUD_DIAG_OSC=0` (same run) | none created | == baseline | **PASS** — strict `=1` proven (a presence-check would have emitted files here) |
| **d** | post-change, `SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1` | all 3 created | == baseline | **PASS** — read-only ⇒ bitwise-neutral **even when enabled** |

Leg c is the load-bearing strict-`=1` evidence: it exercises BOTH gates at
`=0` in one run against the shared keliya bitwise gate. Any presence-only
predicate would create the CSVs and fail this leg.

## Trace / flip self-check (leg d) — `selfcheck/selfcheck_output.txt`

Reproduce: `zsh selfcheck/selfcheck.sh <keliya.out dir> sha/leg-a-baseline.sha`

- CHECK1 Σ(delta_nst)=111130 == `cvode_stats.txt` nst=111130 — **PASS**
- CHECK2 trace rows=6480 == (END−START)·1440/SolverStep = 90·1440/20 = 6480
  (CVode driver-loop invocation count) — **PASS**
- CHECK3 every `delta_nst/nfe/ncfn/netf` ≥ 0 (0 negative rows) — **PASS**
- CHECK4 flip schema: 484 `ele` rows (id 1..484 = NumEle) + 333 `riv` rows
  (id 1..333 = NumRiv); `ele` rows have flips_stage=0, `riv` rows have
  element families=0; entity_id 1-based per index space — **PASS**
- CHECK5 daily `flips_total == sf+us+gw+stage` every row; Σ daily total
  (62382) == Σ per-entity total (62382) — **PASS**
- CHECK6 headers carry `project_name`/`solverstep_min` (+`epsilon_m=1e-06`
  on flip CSVs) — **PASS**
- CHECK7 enabled-run model-output SHA == baseline — **PASS**

## Source audit (osc-flip-counters spec)

- `src/Model/f.cpp` / `src/Model/MD_rhs_core.cpp`: **zero diff**.
- State reads go through `N_VGetArrayPointer(udata)` (accepted CVODE state)
  only. `uYsf`/`uYus`/`uYgw` appear **nowhere in code** — the single textual
  hit in `MD_osc_diag.hpp` is the documentation comment asserting they are
  never touched.

## Files changed (SHUD, branch `p11-osc`)

- `src/Model/MD_osc_diag.hpp` (new) — the two env-gated emitters.
- `src/Model/shud.cpp` — include + 3 call sites (construct+`begin` before
  the loop, `record` per interval at the accepted boundary, `finish` at run
  end); all behind `diag.any_on()`.

## Sample diag output — `diag_csv_heads/`

Heads/tails from the leg-d run (`diag_dt_trace.head.csv`,
`diag_osc_flips.headtail.csv`, `diag_osc_flips_daily.full.csv`).
