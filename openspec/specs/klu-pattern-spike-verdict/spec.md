# klu-pattern-spike-verdict Specification

> **Status**: Implemented via PR-0 [#384](https://github.com/DankerMu/SHUD-OpenMP/pull/384) + PR-A [#385](https://github.com/DankerMu/SHUD-OpenMP/pull/385) + PR-B [#387](https://github.com/DankerMu/SHUD-OpenMP/pull/387) + PR-C [#XXX](https://github.com/DankerMu/SHUD-OpenMP/pull/XXX) (epic [#379](https://github.com/DankerMu/SHUD-OpenMP/issues/379)).
> **Verdict (as of 2026-06-29)**: **Case-aware** — `keliya` + `heihe` = GO (KLU env-var opt-in path); `heihe_x4` = Optional (1.87× wall over 0.7×SPGMR budget; near-miss); `heihe_x16` = NO-GO (17.9× wall over budget; structural).
> **Forward actions**: P8-tune.E.small-only (medium priority — KLU env-var opt-in for keliya + heihe) + P8-tune.F (high priority — BoomerAMG/Hypre spike for heihe_x4 + heihe_x16).
> **Authoritative ADR**: [docs/adr/0005-klu-spike-decision.md](docs/adr/0005-klu-spike-decision.md) (Status: Accepted).
> **Authoritative verdict doc**: [docs/p8tune/klu_spike_verdict.md](docs/p8tune/klu_spike_verdict.md).

## Purpose

Pin the decisive GO / Optional / Case-aware / NO-GO verdict produced by the P8-tune.D KLU pattern-only spike epic (4 cases × 4 ordering combos = 16 cell Slurm array sweep), the 3-axis verdict methodology (fill_ratio + RSS + amortized wall), the machine-readable D8 KV schema consumed by ADR-0005, and the immutable scope guarantees (zero SHUD source patch, zero CVODE wire-up, zero hydrology-equivalence A5 test). This capability is the contract that P8-tune.E.small-only + P8-tune.F downstream epics build on.
## Requirements
### Requirement: P8-tune.D KLU pattern-only spike scope SHALL be zero-source-patch and zero-CVODE-wireup

The P8-tune.D KLU pattern-only spike epic SHALL produce decisive GO/Optional/Case-aware/NO-GO verdict for SuiteSparse KLU as SHUD large-case linear solver candidate by running standalone pattern analysis + symbolic + numeric factorization measurements against a 4-case × 4-ordering matrix WITHOUT modifying any SHUD `.c/.cpp/.h` source, WITHOUT wiring `SUNLinSol_KLU` into `cvode_config.cpp`, WITHOUT running any SHUD model integration, and WITHOUT introducing any hydrology-equivalence (A5) test.

#### Scenario: Tool authoring with no SHUD source patch

- **WHEN** PR-0 authors the spike tool (`tools/p8tune.D/{dump_adjacency,fd_color_jacobian,klu_analyze_factor}.cpp`)
- **THEN** the tool SHALL link only against compiled `libshud.a` archive and SuiteSparse + ColPack libraries
- **AND** the tool SHALL NOT modify any file under `SHUD/src/` or `SHUD/include/`
- **AND** the tool SHALL NOT modify `SHUD/Makefile` except adding the additive `libshud.a` archive target (the documented carve-out exception) and SHALL NOT delete or rename any existing target
- **AND** the tool SHALL be invoked from `tools/p8tune.D/` directory rooted at SHUD-OpenMP top-level, NOT from `SHUD/` submodule
- **AND** PR-0 SHALL NOT bump the SHUD submodule pointer EXCEPT for the additive `libshud.a` carve-out commit(s) on `openmp-baseline` per REQ-7 (pin advances 6ce17d6 → openmp-baseline HEAD; carve-out also includes a `.gitignore` amendment to suppress `libshud.a` + `_libshud_obj/` build artifacts from polluting fresh clones)

#### Scenario: Sweep execution with no CVODE wire-up and no model run

- **WHEN** PR-A executes the 16-cell Slurm array sweep on server cn-nodes
- **THEN** each cell SHALL produce: per-cell stdout/stderr log + numeric J binary dump + `klu_factor` symbolic-flops + numeric-wall + peak RSS from `/usr/bin/time -v`
- **AND** each cell SHALL NOT produce any `rivqdown.dat`, `wb.bin`, or any other SHUD model output file
- **AND** each cell SHALL NOT involve any CVODE step integration or `SUNLinSol_*` constructor invocation
- **AND** each cell SHALL be reproducible from raw input mesh + `cfg.para` without depending on any CVODE-side state

### Requirement: Spike tool SHALL acquire Jacobian sparsity via FD colored Jacobian using existing rhs_core dispatcher

The spike tool SHALL acquire the Jacobian sparsity pattern + numeric values via Curtis-Powell-Reid finite-difference colored Jacobian using Welsh-Powell column coloring from ColPack and probing `MD->rhs_core(Y + ε·v_color, DY, t, ExecPolicy::Serial)` via the existing SHUD RHS dispatcher, NOT via static analysis of `MD_rhs_core.cpp` source AND NOT via dense DQ Jacobian dump.

#### Scenario: Column coloring via Welsh-Powell

- **WHEN** `fd_color_jacobian.cpp` runs on a case adjacency dump
- **THEN** ColPack SHALL be invoked with `JacobianGraphColoring` mode and "DISTANCE_TWO" (or equivalent column-coloring) algorithm
- **AND** the chromatic number χ(G_J) SHALL be reported in spike output
- **AND** for SHUD's 2D local-coupling mesh, χ SHALL be bounded ≤ 50 for production-scale cases (heihe / heihe_x4 / heihe_x16) regardless of NumY scale
- **AND** for the keliya tool-correctness-gate case (NumY=1.5K, simpler mesh), χ SHALL be bounded ≤ 30 (tighter sanity bound; reflects empirical 2D mesh Welsh-Powell χ ≈ 2·deg+1 with degree ≈ 4-5 in keliya). This bound is asserted by the Mac smoke gate (task 1.13)

#### Scenario: FD probe via existing SHUD rhs_core

- **WHEN** `fd_color_jacobian.cpp` probes Jacobian column `k` (representing color group `c`)
- **THEN** it SHALL construct seed vector `v_c[i] = (color[i] == c ? 1 : 0)` for `i ∈ [0, NumY)`
- **AND** the initialization sequence prior to the probe SHALL mirror `SHUD/src/Model/shud.cpp:118-121` exactly: `MD = new Model_Data(fin, fout); MD->loadinput(); MD->initialize();` (lowercase `loadinput` / `initialize` — these are the actual public method names per `SHUD/src/ModelData/Model_Data.hpp:275-276`; `ReadInput` / `Initialize` are NOT valid SHUD API names)
- **AND** it SHALL call `MD->rhs_core(Y + ε·v_c, DY_perturbed, t0, ExecPolicy::Serial)` AND `MD->rhs_core(Y, DY_base, t0, ExecPolicy::Serial)`
- **AND** it SHALL emit numeric J values `J[i,k] ≈ (DY_perturbed[i] - DY_base[i]) / ε` for rows `i` where `color[i] == c`
- **AND** `ε` SHALL be `sqrt(machine_epsilon) * (|Y[k]| + 1)` per CPR standard practice

#### Scenario: FD probe determinism

- **WHEN** `fd_color_jacobian.cpp` is re-run with the same case + same adjacency dump + same compiler flags
- **THEN** it SHALL produce bytewise-identical numeric J binary output (deterministic ColPack ordering + deterministic ε formula yield deterministic numeric J)
- **AND** the determinism SHALL be verified at PR-0 Mac smoke by running `fd_color_jacobian keliya` twice and comparing output via `sha256sum`

### Requirement: Spike tool SHALL acquire mesh+river+lake adjacency via in-process libshud.a Init walk

The spike tool SHALL acquire the SHUD 5-block adjacency structure (surf/unsat/gw/river/lake × surf/unsat/gw/river/lake) via in-process `Model_Data` initialization (`MD->loadinput()` + `MD->initialize()`, mirroring `SHUD/src/Model/shud.cpp:118-121`) followed by a walk over (a) `MD->Ele[i].nabr[0..2] / lakenabr[0..2] / nabrToMe[0..2]` (Element AoS fields per `SHUD/src/classes/Element.hpp:25-27`), (b) `MD->Riv[i].down / toLake / frLake` (River AoS fields per `SHUD/src/classes/River.hpp:54-59`), (c) `MD->RivSeg[i]` (river-segment → element coupling per `SHUD/src/ModelData/MD_adjacency.hpp` S4.1/S4.2 `seg_by_riv` + `seg_by_ele`), and (d) `MD->io_riv / MD->io_lake` membership arrays — NOT via external mesh / river / lake file parsing AND NOT via static derivation from SHUD input file syntax (e.g. `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att`) AND NOT via `MD->rivNode[]` (which is declared but currently UNALLOCATED — see `SHUD/src/ModelData/MD_readin.cpp:182-187` where the `new _Node[NumRivNode]` block is commented out; dereferencing `rivNode` would segfault).

#### Scenario: In-process Init avoids re-implementing updateLakeElement logic

- **WHEN** `dump_adjacency.cpp` calls `MD->initialize()` (lowercase — actual SHUD API; `MD.Initialize()` does not exist)
- **THEN** SHUD's existing `updateLakeElement()` logic SHALL populate `MD->Ele[i].lakenabr[0..2]` + `MD->Ele[i].nabrToMe[0..2]` lake bank assignment automatically
- **AND** `dump_adjacency.cpp` SHALL read these populated members directly to determine elem→lake coupling sparsity
- **AND** the dumped adjacency SHALL match runtime RHS coupling structure exactly (no drift)
- **AND** the dump SHALL NOT dereference `MD->rivNode[]` (allocation commented out in `MD_readin.cpp:182-187`); elem↔river coupling SHALL be sourced from `MD->RivSeg[*]` (river-segment → element index) + `MD->Riv[i].down / toLake / frLake` per `MD_adjacency.cpp` S4.1-S4.5 build logic, which is the same coupling source that `MD_ElementFlux.cpp` + `MD_RiverFlux.cpp` use at runtime

#### Scenario: 5-block adjacency CSC output

- **WHEN** `dump_adjacency.cpp` completes the walk for a case
- **THEN** it SHALL emit a single CSC adjacency file `<case>_adjacency.csc` with 5 row-blocks {surf, unsat, gw, river, lake} × 5 col-blocks {surf, unsat, gw, river, lake} matching state-vector layout per `SHUD/src/Model/shud.cpp:139` (`N_VNew_Serial(NY = 3·NumEle + NumRiv + NumLake, sunctx)`)
- **AND** the file SHALL include header metadata: case name, NumEle, NumRiv, NumLake, NumY, total nnz, per-block nnz counts
- **AND** the file SHALL be deterministic (re-running yields bytewise identical output)

#### Scenario: keliya tool-correctness gate via independent ground-truth reference

- **WHEN** PR-0 Mac smoke executes `dump_adjacency keliya` (task 1.13)
- **THEN** the keliya adjacency dump per-5-block nnz SHALL be independently verified against a reference computed by `tools/p8tune.D/verify_adjacency_keliya.py` (an independent Python implementation that reads keliya `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att` files, constructs adjacency via rSHUD-equivalent logic, and computes per-block nnz)
- **AND** the per-5-block nnz comparison SHALL be exact-match (zero off-by-one tolerance) between the C++ dump and the Python reference
- **AND** in addition, the FD-probed numeric J for keliya SHALL be cross-checked against a brute-force dense FD on keliya (NumY ≈ 1.5K is small enough for an O(NumY) column-by-column dense baseline) — relative error ≤ 1e-6 per nonzero entry
- **AND** PR-0 SHALL NOT merge without keliya tool-correctness PASS (both nnz exact-match and dense FD cross-check)

### Requirement: Case + ordering matrix SHALL be 4 case × 4 ordering combos = 16 cells exactly

The sweep matrix SHALL contain exactly 4 cases × 4 ordering combinations = 16 cells, executed via Slurm array with 1 cell per node.

#### Scenario: 4-case definition

- **WHEN** PR-A submits the 16-cell Slurm array
- **THEN** the 4 cases SHALL be `keliya`, `heihe`, `heihe_x4`, `heihe_x16` exactly
- **AND** each case input SHALL be 90-day truncated `cfg.para` per CLAUDE.md project rule (NOT full case duration)
- **AND** `heihe_x16` SHALL be sourced from `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` (the deployed-at-PR-0 location verified by task 2.1; deployment is task 2.1's responsibility because CLAUDE.md L44 still says `heihe_x16 推到 P8` and master plan L832/L983 say `推到 P8 前补` — the `常驻` claim at master plan L2451 is a forward intent that becomes true ONLY after task 2.1 deployment lands; do NOT cite CLAUDE.md L86, which is the 反熵约定 paragraph and unrelated to heihe_x16)
- **AND** the matrix definition SHALL NOT be modified to drop or substitute any case without OpenSpec change amendment

#### Scenario: 4-combo definition

- **WHEN** PR-A executes a case
- **THEN** the 4 ordering combos SHALL be `(natural, +BTF)`, `(AMD, -BTF)`, `(AMD, +BTF)`, `(COLAMD, +BTF)` exactly
- **AND** each combo SHALL be a distinct Slurm array task ID (e.g., `--array=0-15`)
- **AND** the matrix definition SHALL NOT include `(natural, -BTF)`, `(COLAMD, -BTF)`, METIS-based, or CHOLMOD-based combos in this epic

#### Scenario: Slurm 三铁律 compliance

- **WHEN** PR-A submits the 16-cell Slurm array
- **THEN** `sbatch` SHALL be invoked from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/<run-id>/` (NOT from `/users/...`, per CLAUDE.md Slurm 三铁律 rule 1)
- **AND** `#SBATCH --output` and `#SBATCH --error` SHALL point to paths under `/scratch/...` (NOT `/tmp/` or any node-local path, per CLAUDE.md Slurm 三铁律 rule 2)
- **AND** any patch / hash / `run.sh` / `run_cell.sh` referenced from the sbatch SHALL reside under `/scratch/...` (per CLAUDE.md Slurm 三铁律 rule 3)
- **AND** `--partition=CPU` SHALL target idle nodes from `cn[05-06,09,14-19,23-24]` per master plan §P8-tune.D node list

#### Scenario: Pre-submission environment gate

- **WHEN** PR-A prepares the 16-cell Slurm array submission
- **THEN** the sbatch driver SHALL precheck (refusing submission with exit 1 + diagnostic if any check fails):
  - (a) each case under `SHUD/Basins/<case>/` exposes a valid `cfg.para` with `END - START = 90` days (per CLAUDE.md 项目级铁律)
  - (b) each case's `forcing.csv` references CMFD V0200 data accessible from compute node (per CLAUDE.md L93-95)
  - (c) `heihe_x16` is present at `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x16/` (matches `heihe_x16` row in CLAUDE.md / master plan once task 2.1 deployment lands)
  - (d) `CN_NODE_RAM_BYTES` from `tools/p8tune.D/cn_node_ram.h` is consistent with the value embedded in the aggregator threshold KV (build-time grep + diff)

### Requirement: Three-axis hard verdict SHALL apply (fill + RSS + wall) for per-case GO decision

The per-case verdict SHALL combine three hard-threshold axes such that GO requires ALL 3 axes PASS, and per-case axis flags SHALL be machine-readable in aggregator output.

#### Scenario: Fill axis threshold

- **WHEN** the aggregator computes per-cell fill ratio = `nnz(L + U) / nnz(A)` from `klu_factor` output
- **THEN** PASS threshold SHALL be `fill_ratio < 8 · log₂(NumY)` for the cell's case
- **AND** for heihe_x4 (NumY ~120K), threshold SHALL evaluate to `8 · log₂(120000) ≈ 136`
- **AND** for heihe_x16 (NumY ~760K), threshold SHALL evaluate to `8 · log₂(760000) ≈ 156`
- **AND** the rationale derivation (2D mesh PDE nested-dissection theoretical optimum ≈ log₂(NumY), 8× allowing real-world AMD/COLAMD deviation) SHALL be documented in `docs/p8tune/klu_spike_verdict.md`

#### Scenario: RSS axis threshold

- **WHEN** the aggregator computes per-cell peak RSS from `/usr/bin/time -v` "Maximum resident set size" output
- **THEN** PASS threshold SHALL be `peak_RSS < 0.7 × CN_NODE_RAM_BYTES`
- **AND** `CN_NODE_RAM_BYTES` SHALL be measured at PR-0 via `cat /proc/meminfo` on cn14 (or equivalent representative cn-node) and embedded BOTH in `tools/p8tune.D/cn_node_ram.h` (as a C++ `static constexpr size_t CN_NODE_RAM_BYTES = <measured>;` used by `klu_analyze_factor.cpp` for pre-flight check) AND in the aggregator threshold KV (used by `aggregate_klu_spike.sh` for verdict)
- **AND** the build-time consistency check SHALL grep the same numeric literal out of both `cn_node_ram.h` and the aggregator config and refuse build on mismatch
- **AND** rationale (allows multi-cell parallel execution without OOM) SHALL be documented

#### Scenario: OOM-as-data-point

- **WHEN** `klu_analyze_factor.cpp` runs a cell AND numeric factor exhausts cn-node RAM (either SuiteSparse `klu_factor` returns malloc-fail OR `/usr/bin/time -v` reports peak RSS exceeding `CN_NODE_RAM_BYTES`)
- **THEN** the tool SHALL exit with status 0 (NOT non-zero) AND SHALL emit a diagnostic line `KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N>` to stdout
- **AND** the aggregator SHALL classify that cell as `rss_overflow` data point (PASS-as-Slurm-task, FAIL-on-RSS-axis) — NOT as a Slurm task failure to be re-submitted
- **AND** the Slurm array SHALL NOT re-queue OOM cells (no `--requeue`-style retry); OOM is decisive data, not transient infra failure

#### Scenario: Tool-bound data point (KLU 32-bit-int index overflow)

- **WHEN** `klu_analyze_factor.cpp` runs a cell AND `klu_factor` returns with `common.status == KLU_TOO_LARGE` (status code `-4` in SuiteSparse KLU; "integer overflow has occurred" — SuiteSparse `klu.h`)
- **THEN** the tool SHALL exit with status 0 (NOT non-zero) AND SHALL emit a diagnostic line `KLU_INDEX_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=klu_factor_status_KLU_TOO_LARGE_int32_index_overflow` to stdout
- **AND** the aggregator SHALL classify that cell as `fill_overflow` data point (the 32-bit signed index space of `klu_factor` cannot hold `nnz(L+U)` for the cell's ordering; this is a fill-pathology surfaced as a tool-bound limit, equivalent to the fill axis hard-failing) — NOT as a Slurm task failure to be re-submitted
- **AND** the Slurm array SHALL NOT re-queue these cells; index overflow is decisive data, not transient infra failure
- **AND** the rationale (e.g., heihe_x16 natural+BTF at NumY=485250 produces `nnz(L+U)` exceeding `2^31`; switching to `klu_l_*` 64-bit-index API would be an implementation choice for P8-tune.E and is out of scope for the pattern-only spike) SHALL be documented in `docs/p8tune/klu_spike_verdict.md`

#### Scenario: Wall-budget data point (Slurm TIMEOUT)

- **WHEN** a cell exceeds its Slurm `--time` wall budget AND Slurm `SIGKILL`s the cell before `klu_analyze_factor.cpp` can emit a verdict line
- **THEN** the per-cell `run_cell.sh` trap handler SHALL detect the impending termination (via `trap '...' TERM` registered before pipeline invocation) and emit `KLU_WALL_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> elapsed_sec=<N> wall_budget_sec=<W>` to the cell log on a best-effort basis (Slurm's default `KillWait` of 30 s gives the trap a window to emit before SIGKILL)
- **AND** the aggregator SHALL classify that cell as `wall_overflow` data point (PASS-as-Slurm-task on the trap-emitted marker, FAIL-on-wall-axis) — NOT as a Slurm task failure to be re-submitted
- **AND** when the trap window is insufficient and no marker is emitted (raw TIMEOUT with empty `cell-NN.time` and truncated `cell-NN.log` mid-pipeline), the aggregator SHALL still classify the cell as `wall_overflow` if (a) `sacct -j <jobid> --format=State` reports `TIMEOUT` AND (b) the cell log shows pipeline progress past the dump_adjacency / fd_color_jacobian stages — this fallback enables wall_overflow classification even when the trap window was too small to land the marker
- **AND** the rationale (natural ordering on large NumY produces multi-hour factor walls; e.g., heihe natural+BTF at NumY=19500 took 1630s, so heihe_x4 natural+BTF at NumY=124395 extrapolates to >6500s — exceeding any reasonable per-cell wall budget while remaining strictly orderable as a wall-axis data point) SHALL be documented in `docs/p8tune/klu_spike_verdict.md`

#### Scenario: Wall axis threshold

- **WHEN** the aggregator computes per-cell estimated KLU per-step wall
- **THEN** PASS threshold SHALL be `(numeric_factor_wall / refactor_freq) + (N_solve · solve_wall) < 0.7 × SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline`
- **AND** `refactor_freq` SHALL be conservatively set to 10 (tuned in P8-tune.E if KLU adopted)
- **AND** `SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline` SHALL reference epic #362 (`p8tune-spgmr-maxl`) PR-D #373 60-cell sweep baseline — NOT this epic's PR-A, which is the new 16-cell KLU sweep. Numerical anchor: heihe_x4 N=1 maxl=5 median wall=1489.76s / nst=6575 ≈ 0.227 s/step
- **AND** the aggregator SHALL read the median wall from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv` (3-rep median for the row matching `case=heihe_x4`, `N=1`, `maxl=5`)
- **AND** to avoid re-parsing epic #362 artifacts at PR-B time, the numeric baseline value SHALL also be pinned into `tools/p8tune.D/spgmr_baseline_walls.h` (a header analogous to `cn_node_ram.h`) so aggregator can fall back to the pinned constant if `_summary.tsv` is unreachable
- **AND** N_solve SHALL be the per-step iteration count from CVODE step controller statistics

#### Scenario: Per-case axis machine-readable

- **WHEN** the aggregator emits `aggregate_verdict.txt` after a sweep run
- **THEN** for EACH case in {`keliya`, `heihe`, `heihe_x4`, `heihe_x16`}, it SHALL include the following KV block (canonical schema mirrored verbatim from design D8 and tasks §3.4):
  - `<case>_KLU_fill_axis            = PASS | FAIL`
  - `<case>_KLU_rss_axis             = PASS | FAIL`
  - `<case>_KLU_wall_axis            = PASS | FAIL`
  - `<case>_KLU_overall_verdict      = GO | Optional | Case-aware | NO-GO`
  - `<case>_KLU_NO_GO_axis           = fill_overflow | rss_overflow | wall_overflow | clean_GO`
  - `<case>_KLU_NO_GO_diagnostic     = "<verbatim concrete numbers, e.g. fill_ratio=85.2 >> 8·log₂(NumY)=136 threshold band>"`
  - `<case>_recommended_action       = klu-env-var-opt-in | use-spgmr-default | use-future-amg`
- **AND** in addition to the per-case block, it SHALL include the decisive-cell pointers:
  - `heihe_x4_recommended_next_epic           = p8-tune.E-klu-impl | p8-tune.F-amg-spike`
  - `heihe_x4_recommended_next_epic_priority  = high | medium | low`
- **AND** it SHALL include the embedded thresholds:
  - `CN_NODE_RAM_BYTES                                       = <measured at PR-0>`
  - `SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s  = <pinned, e.g. 0.227>`
- **AND** ADR-0005 GO / Optional / Case-aware / NO-GO branches SHALL all be auto-typeable from these KVs without manual interpretation

#### Scenario: Case-aware/Optional branch auto-population

- **WHEN** the aggregator emits `aggregate_verdict.txt` AND the heihe_x4 overall verdict is in {`Case-aware`, `Optional`}
- **THEN** the per-case `<case>_KLU_overall_verdict` and `<case>_recommended_action` KVs (specified above) SHALL be sufficient for ADR-0005 to auto-populate the Case-aware / Optional branch text — no additional manual interpretation required
- **AND** the Case-aware branch SHALL be triggered iff `keliya_KLU_overall_verdict = GO` AND `heihe_KLU_overall_verdict = GO` AND `heihe_x4_KLU_overall_verdict ∈ {NO-GO, Optional}` (i.e., small cases work, large case doesn't — pattern justifies env-var opt-in)
- **AND** the Optional branch SHALL be triggered iff `heihe_x4_KLU_overall_verdict = Optional` (e.g., 2 of 3 axes PASS but one axis is marginal) — recommendation is `benchmark numeric prototype mini-spike before committing P8-tune.E`

#### Scenario: Numeric factor determinism

- **WHEN** `klu_analyze_factor.cpp` is re-run with the same numeric J binary + the same ordering + the same BTF flag
- **THEN** the produced symbolic-flops count SHALL be bytewise identical across runs
- **AND** the produced `nnz(L+U)` SHALL be bytewise identical across runs
- **AND** the produced numeric-factor wall SHALL fall within ±5% jitter band of the median across 3 repeated runs (jitter band is the documented determinism tolerance for wall, since `/usr/bin/time -v` cannot be fully deterministic on a shared node, while flops/nnz can)

### Requirement: ADR-0005 SHALL adopt 4-branch decision tree with BoomerAMG/Hypre NO-GO retreat

The decision ADR SHALL classify the spike verdict into exactly 4 branches and document a concrete forward action per branch.

#### Scenario: 4-branch decision tree

- **WHEN** `docs/adr/0005-klu-spike-decision.md` is authored (PR-B)
- **THEN** the §Decision section SHALL define exactly 4 branches: GO, Optional, Case-aware, NO-GO
- **AND** GO branch SHALL trigger P8-tune.E full KLU + A5 hydrology-equivalence epic (4-6 week budget)
- **AND** Optional branch SHALL trigger benchmark numeric prototype mini-spike (~1 week) before P8-tune.E commit
- **AND** Case-aware branch SHALL trigger small-case KLU env-var opt-in pattern (mirroring `SHUD_SPGMR_MAXL`) + large-case forward to P8-tune.F
- **AND** NO-GO branch (heihe_x4 fail any axis) SHALL trigger P8-tune.F BoomerAMG/Hypre spike epic (3-4 week budget per Q7 commitment)

#### Scenario: NO-GO axis typing within NO-GO branch

- **WHEN** ADR-0005 NO-GO branch is selected
- **THEN** ADR-0005 §Discussion SHALL document which axis failed (`fill_overflow` / `rss_overflow` / `wall_overflow`) per `<case>_KLU_NO_GO_axis` KV
- **AND** the forward-action description SHALL adapt: for `fill_overflow` → emphasize AMG's O(N) memory advantage; for `wall_overflow` → emphasize AMG's O(N) factor wall; for `rss_overflow` → both
- **AND** ADR-0005 NO-GO branch SHALL NOT trigger SuperLU/SuperLU_MT/cvBandPre/BiCGStab (rejected alternatives per Q7)

### Requirement: PR sequence SHALL be 4-PR (PR-0 tool + PR-A sweep + PR-B aggregator/ADR + PR-C capstone) with 2-3 week budget

The spike epic SHALL be decomposed into exactly 4 sequential PRs with explicit module + ownership boundaries to enable subagent-workflow Phase 4 reviewer-pack scoping.

#### Scenario: PR-0 tool PR boundary

- **WHEN** PR-0 is created
- **THEN** it SHALL touch only: `tools/p8tune.D/*.cpp` + `tools/p8tune.D/Makefile` + top-level `Makefile` (new — wires `make shud_spike` target) + `SHUD/Makefile` (additive `libshud.a` archive target only — the documented carve-out) + `SHUD/.gitignore` (additive amendment to suppress `libshud.a` + `_libshud_obj/` build artifacts from polluting fresh clones — the documented carve-out) + `tools/p8tune.D/spike_run.sh` + `tools/p8tune.D/cn_node_ram.h` + `tools/p8tune.D/spgmr_baseline_walls.h` + `tools/p8tune.D/probe_cn_ram.sbatch` (one-shot cn-RAM probe distinct from the PR-A sweep sbatch; safe in PR-0 because it makes zero KLU/CVODE measurements) + `tools/p8tune.D/verify_adjacency_keliya.py` (independent tool-correctness reference) + `tools/p8tune.D/dense_fd_cross_check.py` (real brute-force dense FD vs colored FD cross-check per REQ-3 keliya tool-correctness gate) + `tools/p8tune.D/README.md` + `.review-evidence/p8tune-klu-spike-pr-0/` (Mac smoke + cn-RAM probe log evidence) + `SHUD_openMP_master_plan.md` §P8-tune.D status flip `[ACTIVE TRIGGER]` → `[OPEN, executing]` + PR-0 link (1-line edit; mirrors epic #362 PR-D `[OPEN, executing]` kickoff pattern — signals epic open to readers)
- **AND** it SHALL NOT touch any `SHUD/src/` source
- **AND** it SHALL NOT touch `tools/p8tune.D/spike_array.sbatch` or `tools/p8tune.D/run_cell.sh` (those are the PR-A sweep submission artifacts; `probe_cn_ram.sbatch` is the only `.sbatch` allowed in PR-0)
- **AND** it SHALL NOT touch any `docs/adr/` or `docs/p8tune/` (ADR + docs belong to PR-B; capstone status flip `[OPEN]→[CLOSE]` + next-epic anchor belong to PR-C — the PR-0 master plan edit is restricted to the §P8-tune.D `[ACTIVE TRIGGER]→[OPEN, executing]` line + PR-0 link)
- **AND** Acceptance SHALL include: Mac builds + Mac smoke runs `keliya` adjacency dump + χ-color FD probe (where χ is the Welsh-Powell chromatic number reported by ColPack, bounded ≤ 30 for keliya per Requirement 2 Scenario "Column coloring via Welsh-Powell") + KLU factor for `(AMD, +BTF)` combo + `verify_adjacency_keliya.py` exact-match check (per Requirement 3 keliya tool-correctness gate) + dense FD cross-check on keliya numeric J + cn-RAM probe log committed to `.review-evidence/p8tune-klu-spike-pr-0/cn_ram_probe.log`

#### Scenario: PR-A sweep PR boundary

- **WHEN** PR-A is created
- **THEN** it SHALL touch: `tools/p8tune.D/spike_array.sbatch` + `tools/p8tune.D/run_cell.sh` + `tools/p8tune.D/precheck_env.sh` (Requirement 4 Scenario "Pre-submission environment gate" implementation, new in PR-A) + `.review-evidence/p8tune-klu-spike-pr-a/` evidence directory
- **AND** it MAY additively touch the following operational workaround files (each carries an inline comment documenting the workaround rationale + a tracked-elsewhere root-cause anchor):
  - `tools/p8tune.D/dump_adjacency.cpp` and `tools/p8tune.D/fd_color_jacobian.cpp`: `_exit(0)` after stdout/stderr flush, bypassing SHUD's `~Model_Data` destructor chain which contains a second uninit-pointer UB (beyond the FloodAlert fix in the SHUD pointer bump) that surfaces at `NumEle > ~30k` on Linux+glibc. Spike binaries are one-shot processes; OS reclaims memory on exit; root-cause SHUD destructor audit tracked in [issue #386](https://github.com/DankerMu/SHUD-OpenMP/issues/386)
  - `tools/p8tune.D/klu_analyze_factor.cpp`: status-`-4` (KLU_TOO_LARGE) recognition + `KLU_INDEX_OVERFLOW_DETECTED` diagnostic, per the new Requirement 5 Scenario "Tool-bound data point (KLU 32-bit-int index overflow)" landed in this PR
  - `tools/p8tune.D/Makefile`: CXXFLAGS `-O2 → -O1` workaround for the gcc 13 -O2 heap-corruption UB at NumY > 100k. Spike binary perf is dominated by KLU library work, not main-loop arithmetic, so the perf cost is negligible
  - `tools/mesh_refine/heihe_x16.autoshud.txt` + `tools/mesh_refine/run_heihe_x16.sh`: the heihe_x16 AutoSHUD deployment artifact (`NumCells=101360` target, mesh-quality constraints inflate to measured `NumEle=160331`), required by tasks §2.1 as the prerequisite for case_idx=3 cells (NN=12-15). The PR-0 `heihe_x16 推到 P8` master plan annotation was an aspirational forward note; this PR-A is the actual deployment step
  - SHUD submodule pointer bump for the upstream `FloodAlert::~FloodAlert()` uninit-pointer fix (default-init pointers + guard `fclose`), pushed to `SHUD/openmp-baseline`
- **AND** the workaround additions above SHALL NOT change the spike binaries' verdict semantics (the per-cell PASS/data-point KVs in `cell-NN.log` are produced by exactly the same code path; `_exit(0)` happens AFTER all KVs are flushed to stdout)
- **AND** it SHALL produce 16 cell result artifacts laid out as flat per-cell files under `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/<run-id>/`: `cell-<NN>.out`, `cell-<NN>.err`, `cell-<NN>.log`, `cell-<NN>.J.bin` (one set per Slurm array task id `<NN>` ∈ {00..15}). This flat layout matches the sbatch `--output=cell-%a.out` / `--error=cell-%a.err` pattern in tasks §2.4 and the aggregator glob `cell-*.{log,err,J.bin}` in tasks §2.7
- **AND** Acceptance SHALL include: 16 cells PASS-or-data-point per Requirement 5 (any cell that surfaces as OOM / index-overflow / wall-overflow MUST be reported via its respective marker — `KLU_OOM_DETECTED` / `KLU_INDEX_OVERFLOW_DETECTED` / `KLU_WALL_OVERFLOW_DETECTED`) + per-cell stdout/stderr archived + per-cell numeric J + RSS measurement archived

#### Scenario: PR-B aggregator + ADR PR boundary

- **WHEN** PR-B is created
- **THEN** it SHALL touch: `tools/p8tune.D/aggregate_klu_spike.sh` + `tools/p8tune.D/render_verdict.sh` + `docs/adr/0005-klu-spike-decision.md` (new) + `docs/p8tune/klu_spike_verdict.md` (new) + `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md` (1-line status header note `Status: Implemented via PR-A + PR-B; verdict = <branch>` per tasks §3.8 — carried forward into the canonical archive in PR-C task 4.7) + `.review-evidence/p8tune-klu-spike-pr-b/aggregate.tsv` (raw per-cell aggregated data) + `.review-evidence/p8tune-klu-spike-pr-b/aggregate_verdict.txt` (machine-readable canonical KV per design D8)
- **AND** Acceptance SHALL include: aggregator emits 16-cell T-table + 3-axis verdict + per-case axis KV block per Requirement 5 Scenario "Per-case axis machine-readable" + ADR-0005 §Decision auto-populated for whichever branch fires (GO / Optional / Case-aware / NO-GO)

#### Scenario: PR-C capstone PR boundary

- **WHEN** PR-C is created
- **THEN** it SHALL touch: `SHUD_openMP_master_plan.md` §P8-tune.D status `[OPEN]→[CLOSE 2026-MM-DD]` + post-merge status paragraph + new §P8-tune.E (if GO) OR §P8-tune.F anchor (if NO-GO) (similar pattern to PR #378 §P8-tune.D anchor) + OpenSpec archive of `openspec/changes/p8tune-klu-spike/` to `openspec/specs/klu-pattern-spike-verdict/spec.md` with `Status: Implemented via PR-A + PR-B; verdict = <branch>` carried into the archived spec + `docs/adr/0005-klu-spike-decision.md` (§Status `Proposed` → `Accepted YYYY-MM-DD` ONLY per tasks §4.6; substantive ADR content is frozen in PR-B) + `docs/review-loop-log.jsonl` (1 append entry for PR-C merge per tasks §4.8)
- **AND** Acceptance SHALL include: master plan reflects ADR-0005 verdict + downstream epic anchored + canonical spec `openspec/specs/klu-pattern-spike-verdict/spec.md` published + review-loop-log entry appended

#### Scenario: Time budget

- **WHEN** the epic is scheduled
- **THEN** total time budget SHALL be 2-3 weeks from PR-0 open to PR-C merge
- **AND** if PR-0 slips ≥1 week, the user SHALL be escalated for "accept 3-4 week timeline vs defer epic to next quarter" — the escalation SHALL NOT offer cvBandPre as a fallback (cvBandPre is rejected per design D7: SHUD river/lake long-range coupling breaks the band assumption; per design R9 cvBandPre is a forward-forward action only, never a schedule fallback for this epic)
- **AND** the master plan §P8-tune.D anchor §Risk row SHALL document this escalation gate

### Requirement: Spike tool interface SHALL be reusable for future Jacobian-aware epics

The spike tool (adjacency dump + FD colored Jacobian + KLU analyze/factor wrapper) SHALL be authored as reusable infrastructure for future Jacobian-aware epics (P8-tune.E full KLU integration, P9 / P10 / any sparsity-needs epic), with stable interfaces and documented output formats.

#### Scenario: Tool output format stability

- **WHEN** the spike tool emits any output (adjacency CSC, numeric J binary, `klu_factor` symbolic-flops report)
- **THEN** each binary / text format SHALL conform to a schema documented in `tools/p8tune.D/README.md` §output-format
- **AND** CLI arguments for the three tool binaries (`dump_adjacency`, `fd_color_jacobian`, `klu_analyze_factor`) SHALL be additive-only after PR-0 freeze: new flags MAY be added without OpenSpec amendment; existing flags MUST NOT be removed or renamed without an OpenSpec change
- **AND** the README §output-format SHALL be checked into PR-0 and version-bumped via a footer line whenever the format changes

