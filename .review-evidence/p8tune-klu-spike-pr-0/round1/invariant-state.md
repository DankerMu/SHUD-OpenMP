Reviewer agent: review-invariant-state
Review round: round 1
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0

Summary: All P8-tune.D KLU spike PR-0 invariants hold — zero SHUD source patch (Makefile-only additive 45-line carve-out via 1 commit 6ce17d6→bc919f5), state machines (FD determinism, OOM-as-data-point, 3-binary pipeline) verified, REQ-3 rivNode invariant honored, evidence corpus complete. Two non-blocking compatibility gaps worth tracking for PR-A.

Invariant Matrix Coverage (high/broad-expanded only):
- Governing invariant (zero SHUD src patch): covered - `cd SHUD && git diff 6ce17d6..bc919f5 -- src/ tests/` returns 0 lines; only `Makefile` changed (+45 lines additive carve-out at L617-661); no header/source modification. Compiled `libshud.a` (3.1 MB) materialized from `_libshud_obj/*.o` via `SHUD_SRC_NOMAIN` filter (reuses existing L548 var from `smoke_strictomp`).
- Source-of-truth (SHUD pin lifecycle): covered - `git log --oneline 6ce17d6..bc919f5` shows exactly 1 commit (`bc919f5 feat(makefile): add libshud.a archive target`); outer pointer `SHUD` bumped (`M SHUD` staged in HEAD a65f3ca). Forward-only ancestry on `openmp-baseline` branch.
- Producers (dump_adjacency / fd_color_jacobian / klu_analyze_factor): covered - all 3 .cpp files present (445/401/310 LOC); each declares CLI contract (`--case` mandatory; `--basin-root`/`--in`/`--out`/`--ordering`/`--btf`/`--report-chi-only` per binary); binary output magics + version=1 documented in README §output-format; reusability invariant (REQ-8) met for downstream P8-tune.E / P9 epics.
- Validators (verify_adjacency_keliya.py / dense_fd_cross_check.py): covered - independent rSHUD-style Python ground-truth reads `keliya.sp.{mesh,riv,rivseg,att}` directly, compares per-5-block nnz against CSC header. Evidence TSV shows 25/25 blocks PASS, `total_nnz_dut=10255==total_nnz_ref` ZERO off-by-one. Dense FD cross-check shows 25/25 blocks PASS with `n_finite_viol=0` across all blocks.
- Public entrypoints (3 binaries CLI + spike_run.sh + probe_cn_ram.sbatch): covered - `spike_run.sh <case> <ordering> <btf>` honors 3-stage pipeline (dump → fd_color → klu) and propagates OOM-as-data-point via `set -euo pipefail` + KLU exit-0 convention. `probe_cn_ram.sbatch` follows Slurm 三铁律 (submit from `/scratch`, `--output` in `/scratch`, sbatch file in `/scratch/.../tools/`).
- Evidence/audit (11 evidence files + .review-evidence dir): covered - all expected artifacts present: `cn_ram_probe.log` (173.0 GiB cn14), `mac_smoke_keliya_{adjacency,fd_color,klu_factor,verify_nnz,dense_fd_cross_check,determinism,chi}.{log,tsv,txt}`, `maxl_sweep_summary.tsv` (epic#362 pin source). `mac_smoke_keliya_determinism.txt` shows run1==run2 sha256 (`e122cde9…ca7bd283`).
- Regression rows (keliya correctness gate + sha256 determinism + OOM-as-data-point): covered - keliya 5-block nnz PASS (evidence above); sha256 determinism PASS (above); OOM-as-data-point exit-0 path readable in `klu_analyze_factor.cpp:236-238` (preflight) + `:272-275` (klu_factor_OOM) + `:283-287` (post-factor RSS overshoot) — all 3 reasons emit `KLU_OOM_DETECTED` + return 0.

Findings:

- Severity: minor
  Failure class: compatibility — SHUD-side build artifacts un-ignored
  Contract or invariant: "Compatibility: existing SHUD build targets MUST still build unchanged" + state-invariant: outer `git status` MUST NOT show stray SHUD-tracked artifacts after `make shud_spike`.
  Scenario or repro: Fresh clone, then `make shud_spike`. After build, `cd SHUD && git status` shows `?? libshud.a` and `?? _libshud_obj/` because upstream `SHUD/.gitignore` lacks both patterns. Verified via `git show openmp-baseline:.gitignore | grep -E "(libshud|_libshud)"` → no match. Locally these are masked by `.git/modules/SHUD/info/exclude` (lines 22-23), but that file is per-clone and NOT shared.
  Required test or evidence: Add `libshud.a` and `_libshud_obj/` lines to `SHUD/.gitignore` in a follow-up SHUD-side commit (additive, single hunk), OR document in README §troubleshooting that fresh clones will see these untracked entries after `make shud_spike`. CI nodes / PR-A server checkout will surface them.
  Sibling surfaces: PR-A `spike_array.sbatch` on cn[05-24] — each compute node runs `make shud_spike`, leaves untracked stragglers in submodule.
  Blocks merge: no
  Impact: cosmetic — `git status` noise on every fresh clone after build; theoretical risk of `git add SHUD/libshud.a` (3 MB binary) being staged by accident if a future contributor runs `git add SHUD/.`.
  Requested fix: Either (a) follow-up patch to upstream `SHUD/.gitignore` adding `libshud.a` + `_libshud_obj/` (preferred; pushes ownership to source-of-truth); or (b) explicit README §troubleshooting bullet noting expected untracked entries after `make shud_spike`.

- Severity: minor
  Failure class: state-machine consistency — omp_set_num_threads(1) call asymmetry
  Contract or invariant: REQ-2 determinism state machine ("process start → omp_set(1) → ColPack → FD probe → bin output") is implemented in `fd_color_jacobian.cpp:196` only. Sibling binaries `dump_adjacency.cpp` and `klu_analyze_factor.cpp` do NOT call `omp_set_num_threads(1)`.
  Scenario or repro: `grep -n omp_set_num_threads tools/p8tune.D/*.cpp` returns only `fd_color_jacobian.cpp:196`. PR-A `spike_array.sbatch` may set `OMP_NUM_THREADS=8` for the Slurm allocation; `dump_adjacency` is a pure topology walk (deterministic regardless), but `klu_analyze_factor`'s SuiteSparse `klu_factor` reads `j.values` and produces `numeric_factor_wall_sec` which is a verdict axis per REQ-5 — wall measurement may be perturbed by background OMP threads even though KLU itself is single-threaded.
  Required test or evidence: Either (a) add `omp_set_num_threads(1)` at `main()` entry of `klu_analyze_factor.cpp` (1-line additive — defensive determinism, no behavioral risk for PR-0 since KLU is single-threaded); or (b) document the asymmetry in README §troubleshooting noting that `dump_adjacency` + `klu_analyze_factor` rely on caller's OMP env (PR-A `run_cell.sh` must `unset OMP_NUM_THREADS` or set `=1` per cell).
  Sibling surfaces: PR-A 16-cell sweep on cn-node multi-core Slurm allocation — each cell invocation MUST enforce thread=1 to make `numeric_factor_wall_sec` comparable across cells.
  Blocks merge: no
  Impact: PR-A wall-axis verdict precision; not a correctness issue for PR-0 (Mac smoke single-threaded by default).
  Requested fix: Add explicit guidance to README §troubleshooting OR add the 1-line `omp_set_num_threads(1)` to the other 2 binaries (1-line additive per file; matches existing pattern in `fd_color_jacobian.cpp:196` with its rationale comment).

Non-blocking notes:
- Master plan §P8-tune.D status flip (L2447) is a clean 1-line edit; section structure intact, downstream anchor references preserved.
- Tool-local `.gitignore` (`tools/p8tune.D/.gitignore`) correctly hides binaries / dSYM / output/ / *.csc / *.bin / *.o; outer-repo `.gitignore` does not duplicate (no conflict; if tool-local is dropped, outer needs absorption — currently `tools/p8tune.D/*` is not covered by outer wildcards).
- SHUD-side untracked files `shud_A` / `shud_C` (P1e era) are NOT staged in the pointer bump — outer `git status` shows only `M SHUD` for the bump itself; submodule's own staged change is the Makefile diff only.
- Design D2 invariant ("`MD->Ele[i].lakenabr[3] + nabrToMe[3]` populated during `MD->initialize()` — runtime truth, not file-syntax derivation") honored: `dump_adjacency.cpp:221-229` walks `MD->Ele[i].lakenabr[j]` post-`initialize()`; verify_adjacency_keliya.py:160 documents NumLake=0 for keliya so lake-coupling blocks are no-ops — matches `lake_*` rows showing 0 in verify_nnz.tsv.
- M = I - γ·J convention (Deviation #5): documented in `klu_analyze_factor.cpp:189-207` with `GAMMA=1.0`; `fill_ratio` formula at L292 uses `(numeric_lnz + unz) / Anz` where `Anz = j.total_nnz` (raw J nnz, not M nnz — but M and J share the same sparsity pattern since γ·J + I·diag doesn't introduce new fill, so they're equal). README §output-format documents the formula in the `klu_analyze_factor` stdout schema.
- Pre-flight estimate inversion test (REQ-5): `klu_analyze_factor.cpp:232` est_bytes uses 64-byte-per-nnz estimate which is a LOWER bound; the actual fall-through to `klu_factor` plus post-factor RSS check at L282-287 catches the inversion case (estimator says PASS but actual factor OOMs OR final RSS > CN_NODE_RAM_BYTES). Triple-guard is correct.
