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

## C1 fix verified (apply loop reset uses documented Initialize pattern)

**Round-2 retraction notice**: The round-1 C1 framing — that calling
`HYPRE_IJVectorInitialize` on an already-assembled vector is UB and that
`SetValues + Assemble` alone is the correct reset — was incorrect.
Direct quote from Hypre 3.1.0 header
`/opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE_IJ_mv.h:556-561`:

> "Prepare a vector object for setting coefficient values. This routine
> will also **re-initialize an already assembled vector**, allowing users
> to modify coefficient values."

The documented contract IS to call `HYPRE_IJVectorInitialize` to re-zero
an assembled vector. Round-2 repair restores the documented pattern in
both reset paths (V-cycle-1 probe + main solve loop): `Initialize +
SetValues + Assemble + GetObject`. Variance across `N_solve=5` iterations
remains well-defined, and the residual_reduction_v1 values are bitwise
identical to the round-1 baseline (see Round-2 repair section below).

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

---

## Round-2 repair (post-cross-review)

Cross-review round-2 surfaced 1 critical retraction + 6 minor follow-ups
against the Phase 6 round-1 fixes. R3 self-reverted its round-1 C1 finding
after directly quoting Hypre 3.1.0 header
`/opt/homebrew/Cellar/hypre/3.1.0/include/HYPRE_IJ_mv.h:556-561`:

> "Prepare a vector object for setting coefficient values. This routine
> will also re-initialize an already assembled vector, allowing users to
> modify coefficient values."

The ORIGINAL pre-Phase-6 code (using `HYPRE_IJVectorInitialize` per iter)
was using the DOCUMENTED API. Phase 6 round-1 (acting on R3 round-1's
wrong premise) had swapped the documented pattern for an undocumented
`SetValues+Assemble`-on-assembled pattern that happens to work on the Mac
host parcsr backend but is not contractually supported by Hypre 3.1.0.

Round-2 restores the documented pattern + applies 6 other minor fixes.

### Fix 1: Restored `HYPRE_IJVectorInitialize` in both reset paths

| File:line | Path | Pre-round-2 | Round-2 |
|-----------|------|-------------|---------|
| `boomeramg_setup_solve.cpp:818` | V-cycle-1 probe x reset | `SetValues + Assemble` only | `Initialize + SetValues + Assemble` |
| `boomeramg_setup_solve.cpp:909` | Main solve loop x reset | `SetValues + Assemble` only | `Initialize + SetValues + Assemble` |

### Fix 2: Surgical `HYPRE_ClearError(HYPRE_ERROR_CONV)`

`boomeramg_setup_solve.cpp:887` — replaced `HYPRE_ClearAllErrors()` with
`HYPRE_ClearError(HYPRE_ERROR_CONV)` so the probe-induced convergence
flag is cleared without masking any legitimate AMG_OOM / generic memory
error bits that may have fired concurrently (more relevant at PR-B
heihe_x16 scale).

### Fix 3: `Tol=0.0` replaces `Tol=1e-300` denormal trick

`boomeramg_setup_solve.cpp:811` — verified via Hypre 3.1.0 header
`HYPRE_parcsr_ls.h:215-220`:

> "(Optional) Set the convergence tolerance, if BoomerAMG is used as a
> solver. If it is used as a preconditioner, it should be set to 0."

`Tol=0.0` IS the documented "no convergence check / preconditioner mode"
sentinel. Switched from `1e-300` to `0.0` for spec compliance. **Bitwise
identical** `residual_reduction_v1` values vs round-1 baseline (50.8449 /
73.8267 / 57.2150 / 50.8449), confirming the two values are semantically
equivalent for this probe but `Tol=0.0` is now contractually documented.

### Fix 4: Purged "Initialize on assembled is UB" misinformation

`boomeramg_setup_solve.cpp:54-61` (docblock) + `boomeramg_setup_solve.cpp:893-900`
(inline pre-loop comment) — removed claims that calling Initialize on an
assembled vector is UB. Replaced with explicit Hypre 3.1.0 header
citation. This verification.md `C1 fix verified` section above also
updated with the retraction notice.

### Fix 5: Hoisted `r_ij` to outer scope + bad_alloc cleanup

`boomeramg_setup_solve.cpp:638` — moved `HYPRE_IJVector r_ij = nullptr;`
declaration to the outer scope alongside `solver / A_ij / b_ij / x_ij`.
`boomeramg_setup_solve.cpp:858` — set `r_ij = nullptr;` after the
in-flight `IJVectorDestroy` so the catch arm skips double-destroy.
`boomeramg_setup_solve.cpp:976` — added `if (r_ij) HYPRE_IJVectorDestroy(r_ij);`
to the bad_alloc cleanup chain. At PR-B heihe_x16 scale (n ~ 485K,
~3.9 MB alloc per IJVector) a bad_alloc raised between the probe's
IJVectorCreate and IJVectorDestroy would have leaked this handle without
the hoist.

### Fix 6: Fixed `std::exit` destructor semantics comment

`boomeramg_setup_solve.cpp:489-497` — comment had claimed `std::exit`
runs destructors. Per C++ `[support.start.term]/p2`, `std::exit` runs
atexit handlers + static-storage-duration destructors but does NOT
unwind the stack, so local automatic-storage objects in `main()` are
NOT destroyed. Comment now states this accurately + explains why the
leak is benign in the SIGTERM-trap context (process about to die; kernel
reclaims everything).

### Fix 7: Fixed `hypre_version_runtime` ownership comment + cache pattern

`boomeramg_setup_solve.cpp:373-403` — comment had claimed Hypre's
`HYPRE_Version` returns a buffer that the library owns (no caller free).
Per Hypre 3.1.0 `src/utilities/HYPRE_version.c`:

```c
version = hypre_CTAlloc(char, len, HYPRE_MEMORY_HOST);
hypre_sprintf(version, "HYPRE Release Version %s", HYPRE_RELEASE_VERSION);
*version_ptr = version;
```

Fresh allocation per call, caller owns + must `hypre_TFree`. The previous
implementation leaked ~52 bytes per `cell_summary` emission. Round-2
implements Option A (static `std::string` cache) — first call parses the
dotted version triple and frees the Hypre-owned buffer; subsequent calls
return the cached `c_str()` without re-probing. Symbol `hypre_TFree` is
a macro (defined in `_hypre_utilities.h`, transitively included via
`_hypre_parcsr_ls.h` → `_hypre_parcsr_mv.h` → `_hypre_utilities.h`).

### Round-2 build + smoke verification

Build: clean, zero warnings, zero errors. See
`/tmp/phase6_round2_build.log`.

4-combo keliya smoke (all PASS, bitwise-identical `residual_reduction_v1`
vs round-1):

| Combo | interp_type | coarsen_type | setup_wall_sec | apply_wall_sec | residual_reduction_v1 | verdict_class |
|------:|------------:|-------------:|---------------:|---------------:|----------------------:|---------------|
|     0 |           6 |            8 |       0.001620 |       0.000333 |               50.8449 | PASS          |
|     1 |          14 |           10 |       0.000366 |       0.000319 |               73.8267 | PASS          |
|     2 |           6 |           21 |       0.000394 |       0.000298 |               57.2150 | PASS          |
|     3 |           8 |            8 |       0.000408 |       0.000329 |               50.8449 | PASS          |

The `cell-N.log` files have been regenerated (overwriting round-1).
`hypre_version=3.1.0 colpack_version=1.0.11 shud_pin=…` schema unchanged.
