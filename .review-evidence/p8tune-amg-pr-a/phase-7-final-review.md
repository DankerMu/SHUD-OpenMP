# PR-A #395 (PR #402) — Phase 7 Final Adversarial Review

**Date**: 2026-06-29
**Branch**: `feat/issue-395-p8tune-amg-pr-a` HEAD `5651a3e`
**Reviewer**: single adversarial (Phase 7 gate before Phase 8 merge)
**Verdict**: **APPROVE_WITH_FOLLOWUP**

## Closure summary

All Critical + High findings from Phase 5 closed and verified through Phase 6 + Phase 6.5 round-2 + round-2 repair:
- 5 Critical (C1-C5) closed; C1 restored to documented Hypre API after R3-round-2 self-retraction
- 3 High (H1-H3) closed
- 4 in-scope Medium (M1, M5, M6, M11) closed
- 7 deferred Medium + Low → follow-up (must be tracked in #396 before PR-B opens — see Phase 7 follow-up below)

## Spec REQ compliance

- **REQ-1 (zero-source-patch)**: CONFIRMED. `git diff baseline/p8tune-amg-spike..HEAD -- SHUD tools/p8tune.D/` empty. SHUD pointer unchanged at `1ab61c0`. p8tune.D unchanged.
- **REQ-2 (FD-color reuse via shell-out)**: CONFIRMED. `fork()+execvp()` pattern (cpp:306-363) eliminates shell injection. Symlinks `.PHONY` (Makefile:173) with executable existence guards. p8tune.D binaries invoked via symlinks.
- **REQ-4 (cell_summary KV schema + 4 marker paths)**: CONFIRMED. 5 KV lines in spec order (cpp:413-429). All 4 verdict marker paths present:
  - AMG_OOM: cpp:752-762 + bad_alloc 981
  - AMG_SETUP_DIVERGE: cpp:771-789 (all 3 OR'd triggers per H1)
  - AMG_SOLVE_DIVERGE: cpp:936-938 (all 3 OR'd triggers per H2)
  - AMG_WALL_OVERFLOW: cpp:481-498 (3-safe-point polling per C4)

## Hidden regressions

- **Phase 6 round-2 "bitwise-identical residual" claim**: CONFIRMED. residual_reduction_v1 values {50.8449, 73.8267, 57.2150, 50.8449} byte-for-byte match round-1 baseline across all 4 cells.
- **Wall variance**: setup 354-467μs (32% spread at μs-scale = OS jitter), apply 316-337μs (7% spread). Within expected envelope. No UB signal.
- **TODO/FIXME/XXX**: None in tools/p8tune.F/.
- **printf safety**: 30+ printf calls all use literal format strings; user input only in `%s` value position. No format-string vuln.
- **Signed/unsigned overflow at PR-B NumY ~485K scale**: int64_t col_ptr + HYPRE_BigInt = long long — overflow-safe.

## CI status

All 5 checks PASS at `5651a3e`: asan-ubsan keliya/qhh, build-and-compare, setup, tools-tests. New tools/p8tune.F/ doesn't intersect existing CI test paths — green is uninformative for PR-A correctness; absence of regressions confirmed.

## Round-2 retraction integrity

- **Hypre header citation**: VERIFIED at `/opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE_IJ_mv.h:556-561`. Exact text "This routine will also re-initialize an already assembled vector, allowing users to modify coefficient values" matches verification.md and cpp inline citations byte-for-byte.
- **Initialize restoration**: VERIFIED at cpp:819 (V-cycle-1 probe x reset) + cpp:913 (main solve loop reset). Both use documented `Initialize + SetValues + Assemble` pattern with Hypre header citations in surrounding comments.
- **verification.md retraction**: PRESENT in §C1 fix verified (lines 119-134) + §Round-2 repair Fix 1 (lines 184-189) + Fix 4 misinformation purge (lines 213-219).

## Schema compliance for PR-C aggregator

- cell-0.log byte-exact match to REQ-4: YES (lines 65-71)
- cell-3.log byte-exact match to REQ-4: YES (lines 63-69)
- All 4 cell logs share identical schema structure; only numeric values differ.

## PR-B / PR-C downstream tracking — GAP IDENTIFIED

**#401 covers**: SHUD-side leaks + server SuiteSparse/ColPack/valgrind install + CI hygiene.

**#401 does NOT cover** (gap): Phase 5 Medium deferrals M2-M11.

**#396 (PR-B issue) does NOT mention them either**.

**Pre-merge follow-up required** (Phase 7 recommends 2-min comment to #396):
- M2 RAII cleanup refactor (cosmetic; functionality correct)
- M3 AMG_OOM Hypre-internal pre-check (PR-B install + RSS pin; Hypre returns NULL on macOS → segfault → mis-attribute to AMG_SETUP_DIVERGE)
- M4 Apply loop warm-up (PR-B perf; cold-cache bias 10-30% at PR-B scale)
- **M7 Batched HYPRE_IJMatrixSetValues** (PR-B perf — CRITICAL at heihe_x16 NumY=485K; per-row loop adds 10-60s setup pollution → mis-emit AMG_SETUP_DIVERGE on healthy hierarchy)
- M8 `-lmpi` double-resolve gate (PR-B install if Hypre source-built `--with-MPI=no`)
- M9 Intel Mac brew prefix (untested user platform)
- M11 README §2 audit on fresh Mac install

## Adversarial PR-B perspective — likely failure mode

**M7 batched-SetValues unfix is the dominant risk**. At NumY=485K, the per-row loop adds 10-60s setup pollution exceeding `WALL_BUDGET_SETUP_SEC=0.237908`. Will mis-emit AMG_SETUP_DIVERGE on a hierarchy that's actually fine. PR-B reviewer must spot this OR PR-B must batch the SetValues call before sweep submission. PR-A's WALL_BUDGET hardcode (cpp:176) needs replacement with shared header `tools/p8tune.D/spgmr_baseline_walls.h` per REQ-5.

## Pre-merge gate (Phase 8 prep)

- **4-件套**: COMPLETE. `.review-evidence/p8tune-amg-pr-a/` contains 7 round-1 artifacts + cross-review-round-1/phase-5-synthesis.md + phase-6-rerun/(verification.md + build.log + 4 cell logs) + phase-7-final-review.md (this file).
- **Self-audit**: R1/R5 cleared at round-2; R3 self-retracted round-1 C1 with Hypre header citation; round-2 repair closed all.
- **Oracle integrity**: BITWISE-IDENTICAL residual_reduction_v1 across Phase 6 + Phase 6.5 round-2. Confirms Initialize-restore + Tol=0.0 + 5 hygiene fixes are semantically equivalent on host parcsr backend.

## Verdict

**APPROVE_WITH_FOLLOWUP** — Ready for Phase 8 squash-merge after 2 housekeeping items:
1. Append tracking comment to #396 (PR-B issue) enumerating M2-M11 with file:line anchors from phase-5-synthesis.md Class G/H — so PR-B reviewer inherits the brief
2. Update PR #402 body description to replace stale `system()` mention with `fork()/execvp()` (post-C2 fix language)

Neither blocks the merge itself. Without them, PR-B reviewer flies blind on 7 known Medium issues.
