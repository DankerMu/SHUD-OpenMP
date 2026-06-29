# PR-A Phase 6 round-1 verification evidence

**Date**: 2026-06-29
**Branch**: `feat/issue-395-p8tune-amg-pr-a` HEAD `4a59306` (orchestrator commit pending)
**Scope**: close 5 Critical + 3 High + 4 Medium findings from
[`cross-review-round-1/phase-5-synthesis.md`](../cross-review-round-1/phase-5-synthesis.md)

## Build

Clean build, zero warnings, zero errors. The Makefile now embeds two
compile-time defines:

```
-DCOLPACK_VERSION_STR="1.0.11"
-DSHUD_PIN_SHA="1ab61c023ac2b93a178c2feb07aa3df509fe1a96"
```

See [`build.log`](build.log).

## 4-combo keliya smoke

All 4 exit 0, all 4 `verdict_class=PASS`.

| Combo | interp_type | coarsen_type | setup_wall_sec | apply_wall_sec | residual_reduction_v1 | verdict_class |
|------:|------------:|-------------:|---------------:|---------------:|----------------------:|---------------|
|     0 |           6 |            8 |       0.000467 |       0.000337 |               50.8449 | PASS          |
|     1 |          14 |           10 |       0.000399 |       0.000325 |               73.8267 | PASS          |
|     2 |           6 |           21 |       0.000354 |       0.000316 |               57.2150 | PASS          |
|     3 |           8 |            8 |       0.000366 |       0.000321 |               50.8449 | PASS          |

### Round-1 M1: residual_reduction_v1 semantic fix verified

**Before (round-0)**: `residual_reduction_v1 = 151303634` (post-MaxIter=100 reduction; meaningless)
**After (round-1)**: `residual_reduction_v1 ∈ {50.8, 57.2, 73.8}` — true V-cycle-1
ratios computed from `||b|| / ||b - A·x_v1||` via Hypre vector ops
(`HYPRE_ParCSRMatrixMatvec` + `HYPRE_ParVectorInnerProd`). In the
reviewer-expected [2, ~100] range for keliya scale.

### Round-1 M5: hypre_version runtime probe verified

**Before**: `hypre_version=3.1.0` (hardcoded `HYPRE_VERSION_STR` macro)
**After**: `hypre_version=3.1.0` (runtime probe via `HYPRE_Version(&ver)`;
parser strips the `"HYPRE Release Version "` prefix). Same value but now
truthful — a brew bump to 3.1.1 / 3.2.0 will show the actual version.

### Round-1 M6: shud_pin + colpack_version embedded

**Before**: `colpack_version=unknown shud_pin=runtime` (sentinel literals)
**After**: `colpack_version=1.0.11` (Makefile `grep` of
`ColPackConfigVersion.cmake` `set(PACKAGE_VERSION "X.Y.Z")` line);
`shud_pin=1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (Makefile
`git -C ../../SHUD rev-parse HEAD`). PR-C aggregator strict parser will
now accept the cell_summary KV unchanged.

## Adversarial sanity

### C2 shell-injection blocked

Invoking with a deliberately-malicious case name including shell metachars:

```
./boomeramg_setup_solve --case 'evil; touch /tmp/p8tune_inject_canary/touched' \
                       --interp-type=6 --coarsen-type=8
```

Result: the canary file `/tmp/p8tune_inject_canary/touched` was **NOT**
created. The case name passed through as a literal argv token to
`dump_adjacency` (which then failed to find the directory because no
such basin name exists). The shell never evaluated the `;` separator —
because `execvp` bypasses the shell entirely.

### C3 symlink rebuild

Removing both symlinks + running `make all`:

```
rm tools/p8tune.F/dump_adjacency tools/p8tune.F/fd_color_jacobian
make -C tools/p8tune.F all
# → ln -sf ../p8tune.D/dump_adjacency dump_adjacency
# → ln -sf ../p8tune.D/fd_color_jacobian fd_color_jacobian
```

Both symlinks are .PHONY, so `make` re-runs the `ln` command every
invocation (idempotent `ln -sf`) — guards against the dangling-symlink
silent drift if upstream `tools/p8tune.D/` is cleaned.

### C5 JBin truncation rejected

```
truncate -s 100 SHUD/Basins/keliya/keliya_numeric_J.bin
./boomeramg_setup_solve --case keliya --interp-type=6 --coarsen-type=8
# → exit 1
# → [amg] ERROR: short read on col_ptr (expected 1786 entries) in
#   ../../SHUD/Basins/keliya/keliya_numeric_J.bin
```

The new `read_jbin` checks every `std::fread` return against the
expected element count and emits a precise per-field error on short
read. Truncated JBin can no longer silently feed stale heap garbage into
`HYPRE_BoomerAMGSetup`.

## REQ-4 marker-trigger gaps closed (H1 + H2)

The expanded triggers are deterministic but cannot fire on keliya scale
(matrix is too well-conditioned). Verified by inspection:

- **H1 (AMG_SETUP_DIVERGE)**: now OR'd over `setup_rc != 0`
  || `hypre_ParAMGDataNumLevels(amg_data) == 0`
  || `setup_wall_sec > 2.0 × WALL_BUDGET_SETUP_SEC` (= 0.475816 s).
  Keliya setup_wall_sec ~ 380μs is 1000× below the 2× budget.
- **H2 (AMG_SOLVE_DIVERGE)**: now OR'd over `solve_rc != 0`
  || `final_res_rel > 1.0` || `residual_reduction_v1 < 2.0`.
  Keliya residual_reduction_v1 ∈ [50, 74] is well above 2.0.
- **H3 (cycle_complexity 2× linkage)**: documented inline in cpp + README §6
  + flagged for PR-C ADR-0007 §Discussion (axis-independence caveat).

## C1 fix verified (apply loop UB removed)

`HYPRE_IJVectorInitialize(x_ij)` is no longer called inside the solve loop.
The loop resets x via `SetValues + Assemble` only (per Hypre 3.1.0 IJ docs
— Initialize is for first-use only). Variance across N_solve=5 iterations
is now well-defined.

## C4 fix verified (async-signal-safe SIGTERM handler)

The SIGTERM handler body is now a single `g_sigterm_pending = 1` store
(POSIX async-signal-safe). The main thread polls the flag at three safe
points: (a) just after Setup completes, (b) after the V-cycle-1 probe,
(c) at the top of each main solve loop iteration. Emission of the
AMG_WALL_OVERFLOW marker + cell_summary block + `std::exit(0)` now
happens from the main thread context where stdio mutexes are quiescent.

The PR-A keliya smoke is dormant for this path; PR-B Slurm sweep will
exercise it under the 8h budget overflow path.

## Tracking

The following round-1 findings DEFERRED to follow-up (per Phase 5
synthesis §"DEFER TO FOLLOW-UP"):

- M2 RAII cleanup refactor (cosmetic)
- M3 AMG_OOM Hypre-internal pre-check (PR-B install + RSS pin)
- M4 Apply loop warm-up (PR-B perf)
- M7 Batched `HYPRE_IJMatrixSetValues` (PR-B perf — critical at heihe_x16 scale)
- M8 `-lmpi` double-resolve PR-B gate
- M9 Intel Mac brew prefix (`/usr/local`)
- L1-L7 low-priority bin

All to be tracked in issue #401 §4 + flagged in PR-B / PR-C reviewer briefs.
