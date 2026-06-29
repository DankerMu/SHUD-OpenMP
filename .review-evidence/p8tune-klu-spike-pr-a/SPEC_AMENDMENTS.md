# Spec amendments — PR-A scope additions to REQ-5 + REQ-7

> The canonical spec lives at `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md` (gitignored under `openspec/changes/`). This file mirrors the active-change amendments so they survive PR-A merge and are picked up by PR-C task 4.7 when the canonical spec is carried forward to `openspec/specs/klu-pattern-spike-verdict/spec.md`.
>
> The amendments here are landed in this PR-A; PR-B aggregator and PR-C capstone are downstream consumers.

## REQ-5 amendment — two new scenarios added after "OOM-as-data-point"

### Scenario: Tool-bound data point (KLU 32-bit-int index overflow)

- **WHEN** `klu_analyze_factor.cpp` runs a cell AND `klu_factor` returns with `common.status == KLU_TOO_LARGE` (status code `-4` in SuiteSparse KLU; "integer overflow has occurred" — SuiteSparse `klu.h`)
- **THEN** the tool SHALL exit with status 0 (NOT non-zero) AND SHALL emit a diagnostic line `KLU_INDEX_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=klu_factor_status_KLU_TOO_LARGE_int32_index_overflow` to stdout
- **AND** the aggregator SHALL classify that cell as `fill_overflow` data point (the 32-bit signed index space of `klu_factor` cannot hold `nnz(L+U)` for the cell's ordering; this is a fill-pathology surfaced as a tool-bound limit, equivalent to the fill axis hard-failing) — NOT as a Slurm task failure to be re-submitted
- **AND** the Slurm array SHALL NOT re-queue these cells; index overflow is decisive data, not transient infra failure
- **AND** the rationale (e.g., heihe_x16 natural+BTF at NumY=485250 produces `nnz(L+U)` exceeding `2^31`; switching to `klu_l_*` 64-bit-index API would be an implementation choice for P8-tune.E and is out of scope for the pattern-only spike) SHALL be documented in `docs/p8tune/klu_spike_verdict.md`

### Scenario: Wall-budget data point (Slurm TIMEOUT)

- **WHEN** a cell exceeds its Slurm `--time` wall budget AND Slurm `SIGKILL`s the cell before `klu_analyze_factor.cpp` can emit a verdict line
- **THEN** the per-cell `spike_array.sbatch` trap handler SHALL detect the impending termination (via `trap '...' TERM` registered before pipeline invocation) and emit `KLU_WALL_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> elapsed_sec=<N> wall_budget_sec=<W>` to the cell log on a best-effort basis (Slurm's default `KillWait` of 30 s gives the trap a window to emit before SIGKILL)
- **AND** the aggregator SHALL classify that cell as `wall_overflow` data point (PASS-as-Slurm-task on the trap-emitted marker, FAIL-on-wall-axis) — NOT as a Slurm task failure to be re-submitted
- **AND** when the trap window is insufficient and no marker is emitted (raw TIMEOUT with empty `cell-NN.time` and truncated `cell-NN.log` mid-pipeline), the aggregator SHALL still classify the cell as `wall_overflow` if (a) `sacct -j <jobid> --format=State` reports `TIMEOUT` AND (b) the cell log shows pipeline progress past the dump_adjacency / fd_color_jacobian stages — this fallback enables wall_overflow classification even when the trap window was too small to land the marker
- **AND** the rationale (natural ordering on large NumY produces multi-hour factor walls; e.g., heihe natural+BTF at NumY=19500 took 1630s, so heihe_x4 natural+BTF at NumY=124395 extrapolates to >6500s — exceeding any reasonable per-cell wall budget while remaining strictly orderable as a wall-axis data point) SHALL be documented in `docs/p8tune/klu_spike_verdict.md`

## REQ-7 amendment — PR-A sweep PR boundary expanded for workarounds

### Scenario: PR-A sweep PR boundary (full text replacement)

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

## Phase-4 review-cycle traceability

| Finding | Severity | Spec amendment | Code fix |
|---|---|---|---|
| F1: cell-12 status=-4 misclassified | HIGH | REQ-5 Scenario "Tool-bound data point" added | `klu_analyze_factor.cpp` recognizes KLU_TOO_LARGE |
| F2: NN=8 TIMEOUT not blessed | HIGH | REQ-5 Scenario "Wall-budget data point" added | `spike_array.sbatch` SIGTERM trap |
| F3: REQ-4 pre-submission gate unimplemented | MEDIUM | (no spec change — gate was already in REQ-4 since PR-0) | `precheck_env.sh` + `run_cell.sh` invocation |
| F4: PR-A boundary breach | MEDIUM | REQ-7 Scenario "PR-A sweep PR boundary" expanded with explicit workaround file list | (no code; spec change blesses existing touched files) |
| F5: aggregator parsing contract | MEDIUM | (covered by REQ-5 amendments F1+F2 — three markers now spec'd) | `run_cell.sh` greps both KLU_OOM_DETECTED + KLU_INDEX_OVERFLOW_DETECTED |
| F6: SWEEP_RESULTS narrative wrong cap | MEDIUM | (no spec change — narrative bug) | `SWEEP_RESULTS.md` 64G→170G + reclassification |
