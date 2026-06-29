# PR-A #395 (PR #402) — Phase 5 Cross-Review Round-1 Synthesis

**Date**: 2026-06-29
**Scope**: P8-tune.F PR-A — `tools/p8tune.F/` new spike tool (488 LOC cpp + 148 LOC Makefile + README + .gitignore)
**Branch**: `feat/issue-395-p8tune-amg-pr-a` HEAD `4a59306`
**Reviewers**: 6 (R1 Hypre API + cell_summary; R2 shell-out + REQ-1/2 + Makefile; R3 numeric J + A=I-γJ + RHS; R4 timing + RSS + version; R5 memory + dtor + signal; R6 spec/REQ + README + downstream PR risk)
**CI status**: all green (asan-ubsan keliya/qhh + build-and-compare + tools-tests + setup) at `4a59306`

## Pattern Escalation Decision

**Repair intensity**: HIGH (confirmed by reviewer findings density)
- 5 Critical findings (3 distinct from each other; 2 multi-reviewer convergent)
- 3 High findings (all spec REQ-4 marker emission semantic gaps)
- 11 Medium findings (4 spec semantic; 4 PR-B forward risk; 3 quality / portability)
- 12 Low/Suggestion

**Subagent-workflow Phase 5 verdict**: invoke Invariant Matrix HARD GATE — Phase 6 implementer MUST close all Critical + High before re-review. Selected Medium (M1, M5, M6) close in-scope per design REQ tightness; remaining Medium (M2, M3, M4, M7, M8, M9, M11) tracked to follow-up issue #401 §4 + flagged in PR-B/PR-C reviewer briefs.

**Reviewer verdict counts**:
- R1: OK_WITH_REVISION
- R2: REQUEST CHANGES
- R3: REQUEST CHANGES
- R4: REQUEST CHANGES (low-touch)
- R5: REQUEST CHANGES
- R6: APPROVE

5 REQUEST_CHANGES vs 1 APPROVE → confirms Phase 5 → Phase 6 round-1 needed.

## Failure Class Inventory

### Class A — Async/Loop semantic UB in Hypre solve path (1 Critical, root issue)

- **C1 (R3)** `tools/p8tune.F/boomeramg_setup_solve.cpp:573-576` — `HYPRE_IJVectorInitialize(x_ij)` called in the timed solve loop after the vector is assembled. Per Hypre 3.1.0 IJ docs, `Initialize` is for first-use only; calling on an assembled vector is UB. Effect: `apply_wall_sec` and `final_res_rel` are unreliable from iter 2 onwards. **All 4 keliya combos with N_solve=5 produce questionable apply walls** (cell-0.log apply_wall_sec=353μs may be wrong; correct value unknown without rerun).

**Containment**: spec REQ-4 cell_summary `apply_wall_sec` semantic ("BoomerAMG apply wall per solve, averaged over --n-solve"). Failure invalidates Axis 2 evaluation at PR-B scale → could mis-emit verdict_class=PASS when truly AMG_SOLVE_DIVERGE.

### Class B — Subprocess / shell injection (1 Critical, multi-reviewer convergent)

- **C2 (R2 CRIT + R5 SUGGEST + R1 LOW F1.9)** `tools/p8tune.F/boomeramg_setup_solve.cpp:223-232` — `shell_out_preflight()` builds `"./" + binary_name + " --case " + case_name + " --basin-root " + basin_root + " > /dev/null 2>&1"` and passes to `std::system()`. `case_name` and `basin_root` flow unsanitized from argv. Attacker-controlled case name `keliya; rm -rf $HOME` executes injected commands.

**Containment**: PR-A is local-tool spike (low threat), but PR-B server sweep + PR-C aggregator may pass case names from manifest files / CSV. Hardening before PR-B is mandatory.

### Class C — Make symlink target lifecycle (1 Critical, R2)

- **C3 (R2 CRIT)** `tools/p8tune.F/Makefile:139,152-164` — `dump_adjacency` and `fd_color_jacobian` targets are not `.PHONY` and have no prerequisites. Once a symlink exists, Make sees the target-file present + considers it up-to-date → if `../p8tune.D/dump_adjacency` is later deleted (`make -C tools/p8tune.D clean`), `make -C tools/p8tune.F all` is a no-op and keeps the dangling symlink. Silent build-correctness drift.

**Containment**: Adding `.PHONY: dump_adjacency fd_color_jacobian` to line 137 fixes. Alternative: declare upstream-binary as prereq.

### Class D — SIGTERM async-signal-safety (1 Critical, R5)

- **C4 (R5 CRIT)** `tools/p8tune.F/boomeramg_setup_solve.cpp:300-309` — `sigterm_handler` calls `emit_marker` → `std::printf` / `std::fflush`, then `emit_cell_summary` → `std::printf` + `std::string::c_str()`, then `peak_rss_bytes()` → `task_info` / `getrusage`. Per POSIX.1-2017 §2.4.3, `printf` / `fflush` / `malloc` / `std::string` accesses are NOT async-signal-safe. If SIGTERM arrives mid-`printf` from main thread (e.g. mid Hypre `PrintLevel=1` verbose output), handler recurses into stdio mutexes → deadlock or stream corruption.

**Containment**: PR-A keliya smoke does not exercise SIGTERM (Mac local, no Slurm). PR-B Slurm sweep WILL trigger this at wall-budget overflow. Fix: replace handler body with `volatile sig_atomic_t g_sigterm_pending = 1` flag check + emit-and-exit at safe points between solve iterations.

### Class E — JBin truncation silent acceptance (1 Critical, 4-reviewer convergent)

- **C5 (R5 CRIT + R1 MED F1.5 + R2 WARN + R3 WARN)** `tools/p8tune.F/boomeramg_setup_solve.cpp:163-184` — Every `std::fread` discards the return value. Truncated `.bin` files (NFS hiccup, disk-full during `fd_color_jacobian` crash, partial sync, SIGKILL mid-write) cause `row_idx` / `values` to contain stale heap garbage; `read_jbin` returns `true`. Hypre ingests garbage → either crash deep in `HYPRE_BoomerAMGSetup` (mis-attributed to `AMG_SETUP_DIVERGE`) or silent wrong-answer with PASS verdict.

**Containment**: Pre-flight idempotency check (line 365-380) is the wrong defense layer (only catches missing/zero-byte files). Real fix: check each `fread` return == expected count; reject on short reads. Pattern from `tools/p8tune.D/fd_color_jacobian.cpp:87,91` is the reference.

### Class F — Marker emission semantic gap vs spec REQ-4 (3 High, R1)

- **H1 (R1)** `tools/p8tune.F/boomeramg_setup_solve.cpp:555-566` — `AMG_SETUP_DIVERGE` Scenario incomplete. Spec REQ-4 Scenario "AMG_SETUP_DIVERGE marker on hierarchy build failure" (spec.md:108-113) lists 3 OR-ed triggers: (i) nonzero setup return, (ii) `HYPRE_BoomerAMGGetNumLevels(solver,&nlevels)==0`, (iii) `setup_wall_sec > 2.0 × WALL_BUDGET_SETUP_SEC`. cpp only checks (i). Effect: hierarchy collapse to single level silently emits PASS.

- **H2 (R1)** `tools/p8tune.F/boomeramg_setup_solve.cpp:597-615` — `AMG_SOLVE_DIVERGE` Scenario incomplete. Spec REQ-4 Scenario (spec.md:117) requires `residual_reduction_v1 < 2.0` OR `final_res > 1.0`. cpp only checks `final_res_rel > 1.0` + nonzero solve return. Stagnating V-cycle at residual_reduction=1.1 mis-emits PASS.

- **H3 (R1 + R6 SUGGEST)** `tools/p8tune.F/boomeramg_setup_solve.cpp:610-612` — `cycle_complexity = 2 × operator_complexity` is a pre-post-smoothing approximation, not a measurement. Hypre 3.1.0 public API has no direct `GetCycleComplexity`. Effect: Axis 4 (Cycle complexity < 1.5) and Axis 5 (Operator complexity < 2.0) become mechanically redundant (Axis 4 trips iff Axis 5 trips at op_complex < 0.75). Spec D2 rationale claims they're independent diagnostics. PR-C ADR-0007 §Discussion must disclose this.

### Class G — Spec semantic mismatch / sentinel values / quality (Medium cluster)

- **M1 (R1 + R3 + R4)** `boomeramg_setup_solve.cpp:587-595` — `residual_reduction_v1` is named `v1` but `HYPRE_BoomerAMGGetFinalRelativeResidualNorm` returns `|r_final|/|b|` after `MaxIter=100` cycles. cell-0.log shows `residual_reduction_v1=151303634` (≈1/6.6e-9 post-convergence) — meaningless as "1-step reduction". Either set MaxIter=1 for a separate timed solve to measure true V-cycle-1 reduction, or rename to `residual_reduction_final` + update spec/README. Hard close: spec REQ-4 Scenario expects v1 semantics; renaming requires spec amendment.

- **M5 (R4)** `boomeramg_setup_solve.cpp:143, 260-261` — `HYPRE_VERSION_STR "3.1.0"` hardcoded. Hypre 3.x provides `HYPRE_Version(char **)` runtime probe. Brew bump (3.1.1, 3.2.0) → silent KV lie.

- **M6 (R4 + R6 + R2 SUGGEST)** `boomeramg_setup_solve.cpp:147, 260-266` — `colpack_version=unknown` + `shud_pin=runtime` sentinel strings. Spec REQ-4 schema strictly requires actual values. No spec text guarantees PR-C aggregator tolerates sentinels. PR-C strict parser would reject all 16 cells. Fix-in-PR-A: embed SHA at compile time via Makefile `-DSHUD_PIN_SHA=...`; probe ColPack version similarly. Eliminates PR-B substitution dependency entirely.

- **M2 (R5 WARN)**: 4× duplicate cleanup (`HYPRE_*Destroy` blocks at AMG_OOM exit, SETUP_DIVERGE exit, success exit, bad_alloc catch). Defer to follow-up (cosmetic; functionality correct).

- **M3 (R5 WARN)**: AMG_OOM probe post-hoc. Hypre internal OOM not caught (returns NULL on macOS → segfault in Hypre internals before our handler). Acceptable for PR-A keliya scale; document as known limitation for PR-B.

- **M4 (R4 WARN)**: No warm-up before timed apply average. Cold-cache bias 10-30% at PR-B heihe_x4 scale. Defer to PR-B (PR-A walls μs-scale, below noise floor anyway).

- **M7 (R6 WARN)**: Per-row `HYPRE_IJMatrixSetValues` loop at NumY ≈ 485K → 485K syscalls + vector allocs before AMG starts. 10-60s setup_wall pollution at PR-B scale. Defer to PR-B (batched API optimization).

- **M8 (R2 WARN)**: `-lmpi` listed independently of Hypre's MPI dep → potential double-resolve on PR-B if server Hypre built `--with-MPI=no`. Defer to PR-B install step (gate `-lmpi` behind probe).

- **M9 (R2 WARN)**: Hard-coded `/opt/homebrew` brew prefix breaks Intel Mac (`/usr/local`) and non-brew installs. Defer to follow-up #401 (low impact; never tested by user).

- **M11 (R6 SUGGEST)**: README §2 `nm | grep | head -10` doesn't actually verify all listed symbols present. Fix-in-PR-A: replace with explicit grep regex.

### Class H — Low-severity (skip / minor)

- **L1-L7**: HYPRE_Initialize return-check (R1 F1.8); WIFEXITED for std::system rc (R2 SUGGEST); random RHS seed not mixed (R1 F1.6); _Exit(0) Finalize skip OK per POSIX (R1 F1.10); nnz_A==nnz_J diagonal-always invariant (R3 SUGGEST); cols_big O(NumY) alloc (R3 SUGGEST); LIFO cleanup order (R5).

## Invariant Surface Inventory

| Invariant | Current state | Phase 6 must close |
|---|---|---|
| **spec REQ-4 marker emission paths (3 of 4 incomplete)** | AMG_SETUP_DIVERGE + AMG_SOLVE_DIVERGE missing triggers; AMG_OOM partial; AMG_WALL_OVERFLOW unsafe | H1, H2, C4 |
| **spec REQ-4 cell_summary schema (sentinel-value parser gap)** | colpack/shud_pin sentinel; hypre_version hardcoded | M5, M6 |
| **Spec REQ-2 shell-out (subprocess invocation safety)** | shell injection via case_name | C2 |
| **Memory ownership / cleanup discipline (PR-0 #386 echo)** | 4× duplicate cleanup; bad_alloc handler good; SIGTERM handler unsafe | C4 (signal); M2 deferred |
| **JBin file reader integrity** | fread short-read silent acceptance | C5 |
| **Hypre solve loop semantics** | IJVectorInitialize re-init UB | C1 |
| **Make symlink target lifecycle** | missing .PHONY → dangling symlink mask | C3 |
| **Hypre 3.1.0 vs 2.30.0 API gap** | SetCumNnzAP/GetCumNnzAP bridge for op_complexity OK; cycle_complexity estimate biased | H3 (document); M5 |

## Regression Matrix

| Class | Pre-fix invariant violation | Post-fix verification |
|---|---|---|
| A (IJVectorInit UB) | apply_wall_sec iter 2+ may be wrong | Rerun 4 combos; verify apply_wall variance ≤ 5% across N_solve=5 |
| B (shell injection) | case_name attacker-controlled exec | Adversarial case name with `;` `$()` `&&` → must reject or fail-safe |
| C (Make symlink) | dangling symlink silent | `rm ../p8tune.D/dump_adjacency && make -C tools/p8tune.F all` must rebuild symlink (or error) |
| D (SIGTERM async-unsafe) | PR-B deadlock under load | Defer empirical to PR-B; ensure flag-based safe handler |
| E (JBin truncation) | garbage matrix → silent PASS | Truncate `.bin` file to 50%; rerun → must report read failure |
| F1 (SETUP_DIVERGE miss) | nlevels=0 collapse silent PASS | Force coarsening failure; verify MARKER:AMG_SETUP_DIVERGE_DETECTED |
| F2 (SOLVE_DIVERGE miss) | residual_reduction_v1 < 2 silent PASS | Force stagnation (Tol=1e-20 + MaxIter=1); verify MARKER |
| F3 (cycle_complexity bias) | Axis 4 + Axis 5 mechanical redundancy | Document in README §6 + cpp inline; flag for PR-C ADR §Discussion |

## Phase 6 Implementer Scope

**MUST CLOSE (Critical + High)** = C1, C2, C3, C4, C5, H1, H2, H3:
- C1: Remove `HYPRE_IJVectorInitialize` from solve loop. Use `SetValues(zeros) + Assemble` to reset x_par per iteration.
- C2: Replace `std::system()` with `fork() + execvp()` for shell-out subprocess; OR sanitize `case_name` via `[A-Za-z0-9_-]+` regex before interpolation.
- C3: Add `.PHONY: dump_adjacency fd_color_jacobian` to Makefile or declare symlink targets as PHONY rules.
- C4: Replace SIGTERM handler body with `volatile sig_atomic_t g_sigterm_pending = 1` flag; check flag at safe points (between solve iterations + after Setup); emit-and-exit there.
- C5: In `read_jbin`, check each `std::fread` return value == expected count; close + return false on mismatch.
- H1: After `HYPRE_BoomerAMGSetup`, call `HYPRE_BoomerAMGGetNumLevels` + compare `setup_wall_sec` against `WALL_BUDGET_SETUP_SEC` (~0.237908 s — pin via shared constant or hardcode for PR-A; PR-C aggregator will use spgmr_baseline_walls.h).
- H2: Compute `residual_reduction_v1` (M1 fix); add `|| (residual_reduction_v1 < 2.0)` to `solve_diverge` boolean.
- H3: Document `cycle_complexity` 2×op_complex estimate bias in cpp inline comment + README §6 + flag for PR-C ADR §Discussion as known limitation requiring axis-independence caveat.

**FIX IN-SCOPE (Medium subset)** = M1, M5, M6, M11:
- M1: Re-architect to measure true V-cycle-1 reduction. Run a separate `HYPRE_BoomerAMGSolve` with `MaxIter=1` + capture residual ratio. Keep current Solve loop for apply_wall_sec measurement at convergence.
- M5: Replace `HYPRE_VERSION_STR "3.1.0"` with runtime `HYPRE_Version()` probe; parse + emit.
- M6: Embed SHUD SHA at compile time via Makefile `-DSHUD_PIN_SHA=\"$(shell git -C ../../SHUD rev-parse HEAD)\"`. Similarly probe ColPack via grep on `ColPackHeaders.h` or pin via Makefile.
- M11: Replace README §2 `nm | grep | head -10` with explicit `grep -E` regex enumerating the 10 required symbols.

**DEFER TO FOLLOW-UP (Medium remainder + Low)** → append to issue #401 §4 + PR-B/PR-C reviewer briefs:
- M2: RAII cleanup refactor (PR-A → PR-B optional, cosmetic)
- M3: AMG_OOM Hypre-internal pre-check (PR-B install-time RSS pin)
- M4: Apply loop warm-up (PR-B perf)
- M7: Batched `HYPRE_IJMatrixSetValues` (PR-B perf, critical for verdict accuracy)
- M8: `-lmpi` double-resolve gate (PR-B install)
- M9: Intel Mac brew prefix (untested user platform)
- L1-L7: low priority, follow-up bin

## Phase 6 Acceptance

- All 5 Critical findings closed (code change + verification path noted)
- All 3 High findings closed (REQ-4 marker triggers + cycle_complexity disclosure)
- 4 selected Medium closed (semantic + sentinel + probe)
- 4-combo keliya smoke rerun → all 4 PASS + new `residual_reduction_v1` semantically correct (≤ ~10.0 typical for keliya) + `apply_wall_sec` stable across N_solve=5 (post-IJVectorInit fix)
- Adversarial smoke: truncated .bin → reject; injected case name → reject; missing upstream binary → rebuild symlink or error
- 7 follow-up Medium + 7 Low tracked to #401 §4

## Next Phases

- **Phase 6**: implementer (single subagent; high repair intensity but bounded fixes)
- **Phase 6.5**: round-2 cross-review (subset: R1 Hypre API + R3 numeric correctness + R5 memory/signal — these 3 had Critical findings; R2/R4/R6 already cleared with WARN-only)
- **Phase 7**: final adversarial review (single reviewer)
- **Phase 8**: pre-merge evidence gate (4 件套 + self-audit + oracle integrity) + squash-merge

## Critical findings traceability

| ID | Reviewer | File:line | Fix file:line (post-Phase-6) |
|---|---|---|---|
| C1 | R3 | boomeramg_setup_solve.cpp:573-576 | (TBD post-fix) |
| C2 | R2 (+ R5 + R1) | boomeramg_setup_solve.cpp:223-232 | (TBD post-fix) |
| C3 | R2 | Makefile:139,152-164 | (TBD post-fix) |
| C4 | R5 | boomeramg_setup_solve.cpp:300-309 | (TBD post-fix) |
| C5 | R5 (+ R1 + R2 + R3) | boomeramg_setup_solve.cpp:163-184 | (TBD post-fix) |

## CI status at synthesis

`4a59306` checks all PASS at https://github.com/DankerMu/SHUD-OpenMP/actions/runs/28371565209/:
- asan-ubsan (keliya) 38s pass
- asan-ubsan (qhh) 3s pass
- build-and-compare (1, keliya) 1m2s pass
- setup 2s pass
- tools-tests 10s pass

New tools/p8tune.F/ doesn't touch SHUD core or existing CI test paths; CI green is expected and uninformative for PR-A correctness. Phase 6 fixes won't regress CI either (same scope).
