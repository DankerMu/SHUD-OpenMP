# amg-pattern-spike-verdict Specification

> **Status**: Implemented via PR-0 [#394](https://github.com/DankerMu/SHUD-OpenMP/pull/394) + PR-A [#402](https://github.com/DankerMu/SHUD-OpenMP/pull/402) + PR-B [#403](https://github.com/DankerMu/SHUD-OpenMP/pull/403) + PR-C [#404](https://github.com/DankerMu/SHUD-OpenMP/pull/404) + PR-D #<TBD> (epic [#393](https://github.com/DankerMu/SHUD-OpenMP/issues/393)). ARCHIVED 2026-06-29.
> **Verdict (as of 2026-06-29)**: **strict=NO-GO-both** (canonical, byte-identical to `aggregate_verdict.txt`; trigger = "heihe_x4 fails ['axis4_cycle'] (max margin 1.333×)") / **amended=GO** (FYI per PR-A H3 disclosure — Axis 4 `cycle_complexity = 2 × operator_complexity` is a hard-coded estimate, NOT HYPRE telemetry measurement; all 4 cases PASS axes 1/2/3/5).
> **Forward actions**: P8-tune.G AMG Axis-4 instrumentation epic [OPEN, HIGH] (4-6w, integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount`); ADR-0007 re-evaluation workshop post-instrumentation (per spec REQ-7 NO-GO-both clause). Verdict_branch-mapped G/H epics SUPPRESSED by strict NO-GO-both.
> **Authoritative ADR**: [docs/adr/0007-amg-spike-decision.md](docs/adr/0007-amg-spike-decision.md) (Status: Accepted).
> **Authoritative verdict doc**: [docs/p8tune/amg_spike_verdict.md](docs/p8tune/amg_spike_verdict.md).

## Purpose

Pin the decisive `GO` / `Optional` / `NO-GO-heihe_x16-only` / `NO-GO-both` / `BLOCKED` verdict produced by the P8-tune.F BoomerAMG/Hypre pattern-only spike epic (4 cases × 4 (interp_type, coarsen_type) combos = 16 cell Slurm array sweep), the 5-axis verdict methodology (setup_wall + apply_wall + memory + cycle_complexity + operator_complexity) plus 4-branch decision tree auto-typed from `aggregate_verdict.txt`, the machine-readable cell_summary KV schema consumed by ADR-0007, and the immutable scope guarantees (zero SHUD source patch outside #386 dtor fix carve-out, zero CVODE wire-up of `SUNLinSol_Hypre`, zero hydrology-equivalence A5 test). This capability is the contract that downstream §P8-tune.G AMG Axis-4 instrumentation epic + ADR-0007 re-evaluation workshop + any forthcoming AMG productionization or GPU sparse epic build on.
## Requirements
### Requirement: P8-tune.F AMG pattern-only spike scope SHALL be zero-source-patch and zero-CVODE-wireup

The P8-tune.F BoomerAMG/Hypre pattern-only spike epic SHALL produce decisive `GO` / `Optional` / `NO-GO-heihe_x16-only` / `NO-GO-both` / `BLOCKED` verdict (verdict_branch token spelling is canonical hyphenated form, machine-readable for PR-C aggregator + PR-D ADR-0007 auto-fill) for Hypre BoomerAMG as SHUD large-case linear solver candidate by running standalone pattern analysis + numeric J probe + AMG hierarchy setup + V-cycle apply measurements against a 4-case × 4-(interp_type, coarsen_type)-combo matrix WITHOUT modifying any SHUD `.c/.cpp/.h` source, WITHOUT wiring `SUNLinSol_Hypre` into `cvode_config.cpp` in production code path, WITHOUT running any SHUD model integration, WITHOUT introducing any hydrology-equivalence (A5) test, WITHOUT promoting to A5-certified-tier (deferred to forthcoming P8-tune.G + ADR-0008), WITHOUT evaluating any GPU sparse path (deferred to forthcoming P8-tune.H, conditional on ADR-0007 NO-GO-heihe_x16-only branch), AND WITHOUT introducing cross-toolchain (Mac libomp / server libgomp) A5 (deferred to P9+ epic).

#### Scenario: Tool authoring with no SHUD source patch

- **WHEN** PR-A authors the spike tool (`tools/p8tune.F/boomeramg_setup_solve.cpp`)
- **THEN** the tool SHALL link only against compiled `libshud.a` archive and Hypre + ColPack libraries
- **AND** the tool SHALL NOT modify any file under `SHUD/src/` or `SHUD/include/` (excluding PR-0 #386 fix which is the SHUD root-cause repair, not a P8-tune.F scope addition)
- **AND** the tool SHALL NOT modify `SHUD/Makefile` (the `libshud.a` carve-out was already added by P8-tune.D PR-0 and is reused as-is)
- **AND** the tool SHALL be invoked from `tools/p8tune.F/` directory rooted at SHUD-OpenMP top-level, NOT from `SHUD/` submodule
- **AND** PR-A SHALL NOT bump the SHUD submodule pointer (any SHUD pointer bump is PR-0 #386 fix scope only)

#### Scenario: Sweep execution with no CVODE wire-up and no model run

- **WHEN** PR-B executes the 16-cell Slurm array sweep on server cn-nodes
- **THEN** each cell SHALL produce: per-cell stdout/stderr log + Hypre IJMatrix dump (optional, for re-run reproducibility) + `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve` timing + peak RSS from `/usr/bin/time -v`
- **AND** each cell SHALL NOT produce any `rivqdown.dat`, `wb.bin`, or any other SHUD model output file
- **AND** each cell SHALL NOT involve any CVODE step integration or `SUNLinSol_Hypre` constructor invocation in production code path
- **AND** each cell SHALL be reproducible from raw input mesh + `cfg.para` without depending on any CVODE-side state

### Requirement: Spike tool SHALL acquire Jacobian sparsity via FD colored Jacobian reused from P8-tune.D

The spike tool SHALL acquire the Jacobian sparsity pattern + numeric values via Curtis-Powell-Reid finite-difference colored Jacobian using Welsh-Powell column coloring from ColPack and probing `MD->rhs_core(Y + ε·v_color, DY, t, ExecPolicy::Serial)` via the existing SHUD RHS dispatcher, REUSING the `tools/p8tune.D/fd_color_jacobian.cpp` + `tools/p8tune.D/dump_adjacency.cpp` binaries unchanged via **shell-out invocation** (NOT in-process function call — `tools/p8tune.D/` source is frozen per REQ-1 scope so cannot be refactored to library API), linked via Makefile symlink in `tools/p8tune.F/Makefile` targeting `../p8tune.D/{fd_color_jacobian,dump_adjacency}`, NOT via static analysis AND NOT via dense DQ Jacobian dump AND NOT via re-implementation.

#### Scenario: FD-color Jacobian reuse from P8-tune.D

- **WHEN** `boomeramg_setup_solve` requires numeric J as input
- **THEN** it SHALL shell out to `tools/p8tune.D/fd_color_jacobian` (via subprocess) to produce CSR numeric J binary
- **AND** the column coloring SHALL be DISTANCE_TWO Welsh-Powell from ColPack (chromatic χ identical to P8-tune.D PR-0 baseline per case)
- **AND** the FD step size ε SHALL match P8-tune.D (sqrt(machine_eps) × max(|Y_i|, 1))
- **AND** the resulting numeric J SHALL pass byte-identical check against P8-tune.D PR-0 fd_color_jacobian output for the same case + same SHUD pin

#### Scenario: 5-block CSC adjacency reuse

- **WHEN** `boomeramg_setup_solve` requires adjacency structure for pre-build verification
- **THEN** it SHALL reuse `tools/p8tune.D/dump_adjacency` binary output unchanged
- **AND** adjacency SHALL be 5-block CSC (surf / unsat / gw / river / lake) matching P8-tune.D scheme

### Requirement: PR-0 SHALL fix #386 SHUD Model_Data destructor uninit-pointer UB as hard prereq

PR-0 SHALL be a dedicated SHUD source-level fix PR (not a tool-authoring PR) that resolves [#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386) (SHUD `Model_Data` 析构链 uninit-pointer UB causing `free(): invalid pointer` heap corruption at NumY > 100k) by identifying + initializing the offending pointer(s) in the `Model_Data` constructor and/or correcting the `delete[]` order in the destructor chain, validated by valgrind clean on keliya + heihe. After fix, PR-0 SHALL remove the `_exit(0)` workaround in `tools/p8tune.D/{fd_color_jacobian,dump_adjacency}.cpp` and verify dtor-full smoke on heihe_x4 + heihe_x16 cases. **This requirement is the explicit SHUD-source-fix carve-out excluded from REQ-1's zero-source-patch scope** (see REQ-1 Scenario "Tool authoring with no SHUD source patch" — `excluding PR-0 #386 fix which is the SHUD root-cause repair, not a P8-tune.F scope addition`).

#### Scenario: #386 root cause fix

- **WHEN** PR-0 author investigates #386
- **THEN** they SHALL grep `delete\[\]` and constructor patterns in `SHUD/src/ModelData/` (and any sibling `~Model_Data` chain) to locate the uninit ptr
- **AND** they SHALL fix by either (i) initializing the ptr to nullptr in constructor + null-check before delete in destructor, OR (ii) correcting the dtor order so all ptrs are initialized before any `delete[]` runs
- **AND** the SHUD source change SHALL be committed to `openmp-baseline` branch (per CLAUDE.md SHUD submodule 工作流;NEVER to `master`)
- **AND** outer repo SHALL contain a SHUD submodule pointer bump commit

#### Scenario: valgrind clean acceptance

- **WHEN** PR-0 reviewer runs `valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ./shud keliya`
- **THEN** valgrind SHALL report 0 errors AND 0 definite/indirect leaks
- **AND** same command on `./shud heihe` SHALL also report 0 errors AND 0 definite/indirect leaks
- **AND** `tools/p8tune.D/dump_adjacency heihe_x4` + `tools/p8tune.D/dump_adjacency heihe_x16` SHALL complete with normal exit (return 0, not `_exit(0)`) without heap corruption

#### Scenario: `_exit(0)` workaround removal

- **WHEN** PR-0 reviewer inspects `tools/p8tune.D/fd_color_jacobian.cpp` and `tools/p8tune.D/dump_adjacency.cpp`
- **THEN** any `_exit(0)` or `_exit(1)` call SHALL be replaced with `return 0` or `return 1` so the C++ destructor chain runs to completion
- **AND** the binary SHALL still exit 0 (success) or 1 (failure) but via normal main() return path

### Requirement: PR-B 16-cell Slurm array sweep SHALL emit verdict-class data per cell

PR-B SHALL execute a 4-case × 4-(interp_type, coarsen_type)-combo = 16-cell Slurm array sweep on server cn-nodes (`/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/`, NOT login-node, NOT `/users/`) with per-cell SIGTERM trap + `MARKER:AMG_WALL_OVERFLOW_DETECTED` stdout marker emission on Slurm kill, producing one of 5 KV `verdict_class` values: `PASS` / `AMG_OOM` / `AMG_SETUP_DIVERGE` / `AMG_SOLVE_DIVERGE` / `AMG_WALL_OVERFLOW` per cell with structured cell_summary KV block. **Naming convention**: stdout marker lines use verb form (`MARKER:AMG_OOM_DETECTED`, `MARKER:AMG_SETUP_DIVERGE_DETECTED`, etc., `_DETECTED` suffix indicates emission action), KV value uses class noun form (`verdict_class=AMG_OOM`, no `_DETECTED` suffix). PASS has no marker emission (only KV value `verdict_class=PASS`).

#### Scenario: 4 case × 4 combo matrix

- **WHEN** `spike_array.sbatch` is submitted with `--array=0-15`
- **THEN** cells SHALL be decoded as: `CASE = case_for(NN/4)` where `case_for(0)=keliya, case_for(1)=heihe, case_for(2)=heihe_x4, case_for(3)=heihe_x16`
- **AND** `COMBO = combo_for(NN%4)` where:
  - `combo_for(0) = (interp_type=6, coarsen_type=8)` Hypre default (classical-extended + HMIS)
  - `combo_for(1) = (interp_type=14, coarsen_type=10)` aggressive (extended+i + HMIS)
  - `combo_for(2) = (interp_type=6, coarsen_type=21)` alt-coarsening (classical-extended + CGC)
  - `combo_for(3) = (interp_type=8, coarsen_type=8)` fallback (standard + HMIS)

#### Scenario: Slurm 三铁律 + 7-condition precheck_env.sh gate

- **WHEN** sbatch is submitted
- **THEN** `tools/p8tune.F/precheck_env.sh` SHALL enforce 7 conditions explicitly (mirror klu spec REQ-4 Pre-submission environment gate pattern, no group-reference, every condition individually checked + logged):
  - (a) **cfg.para 90-day truncation** per case: `END = START + 90` (day-index), per CLAUDE.md 项目级铁律「所有 case ≤90 天截断」 — grep `cfg.para` per case + verify END-START = 90
  - (b) **CMFD V0200 forcing**: each forcing dir contains `V0200` filename pattern (CMFD V0106 已淘汰 per CLAUDE.md;NE 区域必须 V0200 否则 NA all)
  - (c) **heihe_x16 已部署**: `find /scratch/.../SHUD/Basins/heihe_x16/ -name '*.cfg.para'` 命中 + NumEle ≥ 25340 (per CLAUDE.md heihe_x16 mesh)
  - (d) **cn-node RAM ≥ 121 GiB**: `cat /proc/meminfo | grep MemTotal` value ≥ 127000000 KiB
  - (e) **sbatch from `/scratch/`**: `pwd` 起点 in `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/` (per CLAUDE.md Slurm 三铁律 rule 1)
  - (f) **`#SBATCH --output` + `--error` 路径在 `/scratch/`**: grep `#SBATCH --output=` + `--error=` 都以 `/scratch/` 开头 (per rule 2)
  - (g) **所有 referenced scripts 在 `/scratch/`**: grep `srun`/`run_cell.sh`/`fd_color_jacobian` 路径都 in `/scratch/` (per rule 3)
- **AND** precheck failure SHALL exit non-zero before sbatch submission attempt + emit which condition failed

#### Scenario: SIGTERM trap → AMG_WALL_OVERFLOW marker

- **WHEN** a cell's wall budget exceeds `#SBATCH --time=08:00:00` and Slurm sends SIGTERM
- **THEN** the SIGTERM trap SHALL emit `MARKER:AMG_WALL_OVERFLOW_DETECTED` marker line to cell-NN.log stdout
- **AND** the cell SHALL exit 0 (success — overflow is a valid data point, not a build failure) per P8-tune.D KillWait window convention
- **AND** cell_summary KV block SHALL include `verdict_class=AMG_WALL_OVERFLOW` field (no `_DETECTED` suffix in KV value)

#### Scenario: AMG_OOM marker on setup/solve memory failure

- **WHEN** `HYPRE_BoomerAMGSetup` or `HYPRE_BoomerAMGSolve` returns nonzero indicating memory exhaustion, OR `getrusage(RUSAGE_SELF, &ru).ru_maxrss × 1024 > CN_NODE_RAM_BYTES × 0.95` mid-run probe trips, OR std::bad_alloc thrown
- **THEN** the tool SHALL emit `MARKER:AMG_OOM_DETECTED` to cell-NN.log stdout
- **AND** the cell SHALL exit 0 (success — OOM is a valid data point per design R2)
- **AND** cell_summary KV block SHALL include `verdict_class=AMG_OOM` field

#### Scenario: AMG_SETUP_DIVERGE marker on hierarchy build failure

- **WHEN** `HYPRE_BoomerAMGSetup` returns nonzero status (non-OOM, non-bad_alloc) OR `HYPRE_BoomerAMGGetNumLevels(solver, &nlevels)` after setup returns nlevels = 0 OR setup_wall_sec > `2.0 × WALL_BUDGET_SETUP_SEC` (budget overflow but not yet Slurm SIGTERM)
- **THEN** the tool SHALL emit `MARKER:AMG_SETUP_DIVERGE_DETECTED` to cell-NN.log stdout
- **AND** the cell SHALL exit 0 (valid data point)
- **AND** cell_summary KV block SHALL include `verdict_class=AMG_SETUP_DIVERGE` field

#### Scenario: AMG_SOLVE_DIVERGE marker on convergence failure

- **WHEN** `HYPRE_BoomerAMGSolve` reports `residual_reduction_v1 < 2.0` (V-cycle 1 step residual reduction ratio below canonical convergence threshold, matching design R3 + proposal R3) OR `HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res)` returns res > 1.0 (iteration diverged)
- **THEN** the tool SHALL emit `MARKER:AMG_SOLVE_DIVERGE_DETECTED` to cell-NN.log stdout
- **AND** the cell SHALL exit 0 (valid data point)
- **AND** cell_summary KV block SHALL include `verdict_class=AMG_SOLVE_DIVERGE` field

#### Scenario: cell_summary KV block schema

- **WHEN** a cell completes successfully (PASS) or with marker (any of 4 AMG_* class)
- **THEN** cell-NN.log SHALL contain a `CELL_SUMMARY_BEGIN` ... `CELL_SUMMARY_END` block with KV lines (fixed order for aggregator parser stability):
  - `case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>`
  - `setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>`
  - `cycle_complexity=<CC> operator_complexity=<OC> residual_reduction_v1=<R1>`
  - `verdict_class=<PASS|AMG_OOM|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_WALL_OVERFLOW>` (no `_DETECTED` suffix; that suffix is for stdout marker only)
  - `hypre_version=<HV> colpack_version=<CV> shud_pin=<SHA>`
- **AND** `verdict_class` enum is canonical (PR-C aggregator parser SHALL reject any other value as malformed)

### Requirement: PR-C aggregator SHALL produce 5-axis verdict per case and auto-typed 4-branch decision

PR-C `tools/p8tune.F/aggregate_amg_spike.sh` SHALL parse all 16 cell-NN.log files, compute per-case best combo (criterion: minimum `setup_wall + apply_wall` combined, tiebreaker = lowest `operator_complexity`), evaluate each case's best combo against 5-axis verdict thresholds, and emit `aggregate_verdict.txt` with both per-case KV blocks and a top-line `verdict_branch` KV that auto-types ADR-0007 §Decision into one of 4 branches (GO / Optional / NO-GO-heihe_x16-only / NO-GO-both) or BLOCKED.

#### Scenario: Axis threshold constants pinned in shared header

- **WHEN** PR-A authors `tools/p8tune.F/aggregate_amg_spike.sh`
- **THEN** the aggregator SHALL reuse the existing P8-tune.D `tools/p8tune.D/spgmr_baseline_walls.h` (or sibling include) pinning `SPGMR_PER_STEP_SEC = 0.226579` (P8-tune.D heihe_x4 N=1 maxl=5 3-rep median baseline) + `CN_NODE_RAM_BYTES = 173 × 1024^3` (per P8-tune.D PR-0 cn-node RAM probe), UNCHANGED — not re-derive constants
- **AND** the aggregator SHALL compute derived thresholds as: `WALL_BUDGET_APPLY_SEC = 0.7 × SPGMR_PER_STEP_SEC ≈ 0.158605 s`; `WALL_BUDGET_SETUP_SEC = 1.5 × WALL_BUDGET_APPLY_SEC ≈ 0.237908 s`; `WALL_BUDGET_RSS_BYTES = 0.7 × CN_NODE_RAM_BYTES ≈ 130 GiB`
- **AND** if `spgmr_baseline_walls.h` is missing on server (P8-tune.D工具 deployment incomplete), aggregator SHALL fail with explicit "missing baseline header" error, not silently default

#### Scenario: 5-axis threshold evaluation per case

- **WHEN** aggregator evaluates a case's best combo (criterion: min `setup_wall_sec + apply_wall_sec` combined; tiebreaker = min `operator_complexity`)
- **THEN** it SHALL compute 5 boolean axis PASS/FAIL using the pinned derived thresholds:
  - **Axis 1 (Setup)**: `setup_wall_sec < WALL_BUDGET_SETUP_SEC` (≈ 0.237908 s; ratio threshold 1.5 × 0.7 × SPGMR_PER_STEP_SEC, amortize allowance)
  - **Axis 2 (Apply)**: `apply_wall_sec < WALL_BUDGET_APPLY_SEC` (≈ 0.158605 s; ratio threshold 0.7 × SPGMR_PER_STEP_SEC, same standard as P8-tune.D KLU per-step)
  - **Axis 3 (Memory)**: `peak_rss_bytes < WALL_BUDGET_RSS_BYTES` (≈ 130 GiB; ratio threshold 0.7 × CN_NODE_RAM_BYTES)
  - **Axis 4 (Cycle complexity)**: `cycle_complexity < 1.5` (unitless ratio, V-cycle internal op count / NumY)
  - **Axis 5 (Operator complexity)**: `operator_complexity < 2.0` (unitless ratio, sum coarse grid sizes / fine grid size)
- **AND** a case is `PASS` if all 5 axes PASS, else `FAIL` with `failing_axes` list (e.g., `[1, 4]` if both Setup and Cycle complexity fail)

#### Scenario: failing_margin computation per axis

- **WHEN** aggregator computes `failing_margin` for a case's failing axis
- **THEN** for any axis A with actual value V vs threshold T, `failing_margin SHALL be computed as V / T` (ratio form;`failing_margin > 1.0` indicates FAIL per definition;`failing_margin >= 1.5` indicates structural failure)
- **AND** for Axis 1 / Axis 2 (wall axes), V = `setup_wall_sec` or `apply_wall_sec`, T = corresponding `WALL_BUDGET_*_SEC` constant
- **AND** for Axis 3 (memory), V = `peak_rss_bytes`, T = `WALL_BUDGET_RSS_BYTES`
- **AND** for Axis 4 / Axis 5 (unitless complexity axes), V = `cycle_complexity` or `operator_complexity`, T = `1.5` or `2.0`

#### Scenario: 4-branch decision auto-typing (covers all axis + case combinations)

- **WHEN** aggregator computes verdict_branch (after per-case PASS/FAIL + failing_margin computed)
- **THEN** it SHALL apply (rules evaluated top-down, first matching rule wins):
  - **`GO`**: ALL 4 cases (keliya, heihe, heihe_x4, heihe_x16) PASS
  - **`Optional`**: keliya + heihe + heihe_x4 PASS AND heihe_x16 fails ONLY on wall axes (Axis 1 OR Axis 2 OR both, exclusive) with `max(failing_margin for failing axes) < 1.5×`
  - **`NO-GO-heihe_x16-only`**: keliya + heihe + heihe_x4 PASS AND heihe_x16 fails on Axis 3 (memory) OR Axis 4 (cycle complexity) OR Axis 5 (operator complexity) — i.e., any axis that grouping with hierarchy quality / structural retreat; treated together because design D2 establishes these axes diagnose the AMG hierarchy failing mode that requires GPU sparse retreat
  - **`NO-GO-both`**: heihe_x4 fails ANY axis OR heihe_x16 fails wall axes (Axis 1/2) with `max(failing_margin) ≥ 1.5×` OR heihe_x16 fails ≥ 3 of 5 axes (structural)
  - **`BLOCKED`**: any of:
    - any cell emits malformed cell_summary (missing verdict_class field, missing required KV)
    - any cell verdict_class enum value outside the canonical 5 (`PASS|AMG_OOM|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_WALL_OVERFLOW`)
    - any cell still shows heap corruption / dtor UB recurrence (indicating PR-0 #386 fix incomplete) per design R5
  - **Fallback** (no rule above matched): explicitly emit `BLOCKED` with reason "verdict_branch logic gap — case combination not covered, manual review required" (deterministic, no silent unset)
- **AND** small-case (keliya OR heihe) PASS gate: if keliya OR heihe FAIL any axis but heihe_x4 + heihe_x16 PASS, this is flagged as `BLOCKED` reason "small-case unexpected fail — tool instability suspected" (no production case has this verdict normally)

#### Scenario: aggregate_verdict.txt schema

- **WHEN** aggregator emits `aggregate_verdict.txt`
- **THEN** the file SHALL begin with `# AGGREGATE_VERDICT_BEGIN` and end with `# AGGREGATE_VERDICT_END`
- **AND** the file SHALL contain a single `verdict_branch=<GO|Optional|NO-GO-heihe_x16-only|NO-GO-both|BLOCKED>` top-line KV
- **AND** each case SHALL have a `CASE_VERDICT_BEGIN:<case>` ... `CASE_VERDICT_END:<case>` block with per-axis PASS/FAIL + best_combo (interp_type, coarsen_type) + 5 axis values
- **AND** the file SHALL be consumed by ADR-0007 §Decision auto-fill in PR-D capstone

### Requirement: ADR-0007 SHALL pin amg-spike-decision with auto-typed 4-branch verdict

PR-C SHALL produce `docs/adr/0007-amg-spike-decision.md` following ADR template (Status / Context / Decision / Discussion / References / Suppressed branches / Forward action), with §Decision table auto-filled from `aggregate_verdict.txt` `verdict_branch` KV, and PR-D SHALL flip Status from Proposed → Accepted on capstone-merge.

#### Scenario: ADR-0007 Status lifecycle

- **WHEN** PR-C is authored
- **THEN** ADR-0007 §Status SHALL be `Proposed`
- **AND** §Decision table values SHALL be auto-typed from `aggregate_verdict.txt` `verdict_branch` + per-case KV block
- **WHEN** PR-D capstone is opened
- **THEN** ADR-0007 §Status SHALL flip to `Accepted`
- **AND** §Forward action SHALL trigger forthcoming master plan section per verdict_branch (see REQ-7)

#### Scenario: ADR-0007 §Decision matches aggregate_verdict.txt

- **WHEN** PR-C Phase-4 cross-review verifies ADR-0007 §Decision
- **THEN** the reviewer SHALL confirm §Decision table values are byte-identical to `aggregate_verdict.txt` KV (no hand-curated drift)
- **AND** verdict_branch in §Decision text SHALL match `aggregate_verdict.txt verdict_branch=` line

### Requirement: PR-D capstone SHALL trigger conditional next-epic anchor in master plan

PR-D capstone SHALL flip master plan §P8-tune.F status from `[OPEN, anchor]` to `[CLOSED]`, append a post-merge status paragraph, AND conditionally introduce new master plan anchor sections for next-epic candidates based on ADR-0007 `verdict_branch`.

#### Scenario: Conditional next-epic anchor per verdict_branch

- **WHEN** ADR-0007 `verdict_branch = GO`
- **THEN** PR-D SHALL add master plan §P8-tune.G full AMG + A5 integration epic anchor `[OPEN, HIGH priority]` (4-6 weeks)
- **WHEN** ADR-0007 `verdict_branch = Optional`
- **THEN** PR-D SHALL add §P8-tune.G heihe_x4-only integration anchor `[OPEN, medium priority]` (3-4 weeks)
- **WHEN** ADR-0007 `verdict_branch = NO-GO-heihe_x16-only`
- **THEN** PR-D SHALL add §P8-tune.G heihe_x4-only anchor `[OPEN, medium priority]` (3-4w) AND §P8-tune.H GPU sparse spike anchor (priority + status per GPU-presence gate Scenario below)
- **WHEN** ADR-0007 `verdict_branch = NO-GO-both`
- **THEN** PR-D SHALL NOT add new anchor; instead PR-D SHALL note in master plan §P8-tune.F closure paragraph "升级到 ADR re-evaluation workshop; future epic 由 user trigger"
- **WHEN** ADR-0007 `verdict_branch = BLOCKED`
- **THEN** PR-D SHALL change §P8-tune.F status to `[BLOCKED]` AND re-open `[#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386)` with diagnostic comment

#### Scenario: GPU-presence gate for P8-tune.H anchor

- **WHEN** verdict_branch = `NO-GO-heihe_x16-only` AND PR-D capstone is being authored
- **THEN** PR-D author SHALL verify GPU partition availability on server (`sinfo -p GPU` 命中 `gn01` 或其他 GPU node)
- **IF** GPU partition available, PR-D SHALL anchor §P8-tune.H as `[OPEN, medium priority]` (~4w)
- **IF** GPU partition unavailable, PR-D SHALL anchor §P8-tune.H as `[OPEN, BLOCKED-on-GPU-availability]` + 显式注 "需 user 决定换 GPU 节点 / 等待 GPU partition 上线 / 直接 NO-GO heihe_x16"

#### Scenario: OpenSpec archive on capstone with mandatory Status header schema

- **WHEN** PR-D capstone merges
- **THEN** PR-D SHALL run `openspec archive p8tune-amg-spike -y` to move `openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md` to `openspec/specs/amg-pattern-spike-verdict/spec.md`
- **AND** the archived spec SHALL prepend a 5-line Status header (mirror P8-tune.D `openspec/specs/klu-pattern-spike-verdict/spec.md:1-7` archive pattern exactly), each line as a `> ` blockquote:
  - `> **Status**: Implemented via PR-0 #<N0> + PR-A #<NA> + PR-B #<NB> + PR-C #<NC> + PR-D #<ND> (epic #<EPIC>).`
  - `> **Verdict (as of <YYYY-MM-DD>)**: **<verdict_branch>** — <one-line summary with per-case PASS/FAIL/margin>`
  - `> **Forward actions**: <P8-tune.G + (conditional) P8-tune.H + anchor priority + budget per REQ-7 Conditional next-epic anchor Scenario>`
  - `> **Authoritative ADR**: [docs/adr/0007-amg-spike-decision.md](docs/adr/0007-amg-spike-decision.md) (Status: Accepted).`
  - `> **Authoritative verdict doc**: [docs/p8tune/amg_spike_verdict.md](docs/p8tune/amg_spike_verdict.md).`

### Requirement: review-loop log SHALL persist epic-level review accountability

PR-D capstone SHALL append at least 5 JSONL entries (one per PR-0/A/B/C/D) to `docs/review-loop-log.jsonl` recording fixture level, comprehensive rounds, `gate_net_catch`, verifier verdict counts, residual deferrals, and pre-merge skip blocks, mirroring P8-tune.D PR-D capstone log pattern.

#### Scenario: review-loop entries per PR

- **WHEN** each P8-tune.F PR (0/A/B/C/D) completes Phase-4.5 verifier gate
- **THEN** the orchestrator SHALL append one JSONL line to `docs/review-loop-log.jsonl` containing `{pr, epic, fixture_level, rounds, gate_net_catch, verdict_counts, residual_deferrals, pre_merge_skips}`
- **AND** PR-D capstone SHALL verify all 5 entries are present and SHA-anchored to the merged commits before opening capstone-merge PR

