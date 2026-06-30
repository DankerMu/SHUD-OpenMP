# sunlinsol-hypre-wrapper Specification

## Purpose
TBD - created by archiving change p8tune-g0-instrumented-amg-smoke. Update Purpose after archive.
## Requirements
### Requirement: Wrapper exposes SUNLinearSolver iterative-class interface

The wrapper MUST implement the `SUNLinearSolver_Ops` ABI defined in `SHUD/InstallSundials/include/sundials/sundials_linearsolver.h` for SUNDIALS 6.0, declaring type `SUNLINEARSOLVER_ITERATIVE` (NOT `SUNLINEARSOLVER_DIRECT`) and ID `SUNLINEARSOLVER_CUSTOM`. Constructor signature MUST be `SUNLinearSolver SUNLinSol_Hypre(N_Vector y, void *MD, int interp_type, int coarsen_type, SUNContext sunctx)` (5 arguments with explicit `Model_Data *MD` passed as `void *` to keep the header C-friendly). The `y` argument is the same `N_Vector` the caller passes from `SetCVODE` (matching the baseline `SUNLinSol_SPGMR(udata, ...)` call where `udata` is itself an N_Vector). The `MD` argument is the SHUD topology context; the wrapper stashes it for Setup-phase topology-restricted probe. Rationale for explicit-MD constructor (vs the earlier draft's `CVodeGetUserData(cvode_mem, ...)` retrieval): (a) type-safe — no `void *` round-trip through CVODE-internal `A_data` conventions that may drift across SUNDIALS releases; (b) less coupling — the wrapper does not need a `cvode_mem` stash mechanism via SetATimes; (c) lifetime — `MD` outlives the LS, enforced by SHUD driver SetCVODE invocation order.

#### Scenario: Constructor returns valid SUNLinearSolver

- **WHEN** caller invokes `SUNLinSol_Hypre(y, MD, 6, 8, sunctx)` with non-NULL `y` and valid `sunctx`
- **THEN** return value is non-NULL SUNLinearSolver handle whose `ops->gettype(LS)` returns `SUNLINEARSOLVER_ITERATIVE` and whose `ops->getid(LS)` returns `SUNLINEARSOLVER_CUSTOM`

#### Scenario: Constructor rejects NULL N_Vector

- **WHEN** caller invokes `SUNLinSol_Hypre(NULL, MD, 6, 8, sunctx)`
- **THEN** function returns NULL and writes a `[shud-amg] FATAL: ... y is NULL` line to stderr

#### Scenario: Constructor rejects unsupported (interp, coarsen) pair

- **WHEN** caller invokes `SUNLinSol_Hypre(y, MD, interp, coarsen, sunctx)` with `(interp, coarsen) != (6, 8)`
- **THEN** function returns NULL and writes a `[shud-amg] FATAL: ... (interp_type, coarsen_type) = (<got_interp>, <got_coarsen>), G0 requires (6, 8)` line to stderr (relax in G1)

### Requirement: All 15 SUNLinearSolver_Ops callbacks SHALL be implemented (no NULL slots)

The wrapper MUST provide implementations for all 15 function pointers of the on-disk `_generic_SUNLinearSolver_Ops` struct (`SHUD/InstallSundials/include/sundials/sundials_linearsolver.h:108-127`): `gettype`, `getid`, `setatimes`, `setpreconditioner`, `setscalingvectors`, `setzeroguess`, `initialize`, `setup`, `solve`, `numiters`, `resnorm`, `lastflag`, `space`, `resid`, `free`. No slot may be NULL; callbacks not exercised by BoomerAMG MUST be implemented as explicit no-ops returning the appropriate type (`SUNLS_SUCCESS` / `0` / `0.0` / NULL `N_Vector`). The preferred implementation pattern is `SUNLinSolNewEmpty(sunctx)` to obtain a default-NULL-filled ops table followed by explicit assignment of every one of the 15 slots.

#### Scenario: All 15 ops table slots are non-NULL after constructor

- **WHEN** `SUNLinSol_Hypre(y, 6, 8, sunctx)` returns a non-NULL `LS`
- **THEN** the underlying `LS->ops` struct MUST have non-NULL function pointers in all 15 slots: `gettype, getid, setatimes, setpreconditioner, setscalingvectors, setzeroguess, initialize, setup, solve, numiters, resnorm, lastflag, space, resid, free`; PR-0 build smoke MUST include a runtime assertion that fails the build smoke test if any slot is NULL

#### Scenario: SetPreconditioner is no-op (BoomerAMG IS the preconditioner)

- **WHEN** CVODE invokes `LS->ops->setpreconditioner(LS, P_data, Pset, Psol)`
- **THEN** wrapper returns `SUNLS_SUCCESS` and does NOT store the precondition callback (BoomerAMG IS the inner preconditioner; CVODE-supplied user preconditioner is ignored at G0)

#### Scenario: SetScalingVectors accepts but ignores at G0

- **WHEN** CVODE invokes `LS->ops->setscalingvectors(LS, s1, s2)` with non-NULL `s1, s2`
- **THEN** wrapper returns `SUNLS_SUCCESS`; the scale vectors are stashed for telemetry inspection but NOT applied to the AMG solve (G0 limitation, G1 may re-enable)

#### Scenario: SetZeroGuess stores the flag for SPILS optimization path

- **WHEN** CVODE invokes `LS->ops->setzeroguess(LS, onoff)` (SUNDIALS 6.0 SPILS optimization path)
- **THEN** wrapper stores the `booleantype onoff` flag in its content struct and returns `SUNLS_SUCCESS`; if `onoff = SUNTRUE`, subsequent Solve calls may skip the initial AMG iteration since the initial guess is known-zero

#### Scenario: ResNorm returns 0.0 (no published residual norm at G0)

- **WHEN** caller invokes `SUNLinSolResNorm(LS)`
- **THEN** wrapper returns `0.0` (BoomerAMG-as-direct-solver semantics; G0 does not publish a residual norm; G1 may re-enable)

#### Scenario: LastFlag returns wrapper's stored last return code

- **WHEN** caller invokes `SUNLinSolLastFlag(LS)`
- **THEN** wrapper returns the stored last return code from the most recent Solve callback (one of `SUNLS_SUCCESS`, `SUNLS_CONV_FAIL`, `SUNLS_PACKAGE_FAIL_UNREC`) as a `sunindextype`

#### Scenario: Space reports content struct sizes

- **WHEN** caller invokes `SUNLinSolSpace(LS, &lenrwLS, &leniwLS)`
- **THEN** wrapper writes `lenrwLS` = count of real-valued workspace doubles in the content struct (including ring buffer doubles) and `leniwLS` = count of integer-valued workspace ints (including ring buffer ints); returns `SUNLS_SUCCESS`

#### Scenario: Resid returns NULL (no published residual vector at G0)

- **WHEN** caller invokes `SUNLinSolResid(LS)`
- **THEN** wrapper returns NULL (no `N_Vector` residual published at G0; G1 may re-enable)

#### Scenario: Initialize bootstraps Hypre context per Hypre version AND pins Hypre threads under shud_omp

- **WHEN** wrapper's `Initialize(LS)` is invoked
- **THEN** wrapper calls `HYPRE_Initialize()` if `HYPRE_RELEASE_NUMBER >= 30000`, else calls `HYPRE_Init()` (legacy API). If the binary is `shud_omp` (detected via `omp_get_max_threads() > 1` or `getenv("OMP_NUM_THREADS")` returning a value > 1), wrapper forces Hypre to 1 thread via `HYPRE_SetGlobalOptions("default_thread_count=1")` (defense-in-depth alongside sbatch-side `OMP_NUM_THREADS=1` enforcement) AND emits exactly one line `[shud-amg] Hypre threads=1 mode=<single|nested>` to stdout

### Requirement: Setup callback builds Hypre matrix from CVODE Jacobian via topology-restricted probe

The wrapper's `Setup(LS, A)` callback MUST construct a `HYPRE_IJMatrix` from CVODE's current Jacobian. Since CVODE invokes the wrapper with `SUNMatrix A = NULL` (matrix-free path), Setup MUST probe the stashed `ATimes` callback to reconstruct sparse non-zeros, restricted to SHUD's topology connectivity pattern. The Jacobian dimension is `n = MD->NumY` (state-variable dimension, ~3× larger than `MD->NumEle`); the probe MUST be `O(NumY × bw_effective)` (NOT `O(NumY²)`) where `bw_effective ≤ 32` (conservative overcount; typical SHUD coupling is ≤12) derived from `MD->Ele[i].nabr[k]` (element neighbor field, k=0..2 per `SHUD/src/classes/Element.hpp:25`) plus river / lake couplings. The wrapper obtains `Model_Data *MD` from the **constructor stash** (set when `SUNLinSol_Hypre(y, MD, ...)` is invoked from `SetCVODE`); no `CVodeGetUserData(cvode_mem, ...)` round-trip is required.

Setup-vs-Solve build cadence (G0 baseline):
- `Setup(LS, NULL)` destroys any prior AMG/IJ handles and sets a `pending_setup_called` flag. Telemetry's `setup_wall_sec` column accumulates the destroy wall here.
- First `Solve` after a Setup performs the actual `HYPRE_BoomerAMGSetup` build (and the topology-restricted probe). Telemetry's `setup_wall_sec` accumulates the build wall here too; the Solve that consumes the pending flag emits the combined `setup_wall_sec` in its ring-buffer entry.

This deferred build is the G0 baseline because the wrapper has no access to a CVODE-managed `N_Vector` flavor template inside `Setup` (the `SUNMatrix A` argument is NULL on the SPILS path), whereas `Solve` receives the in-flight `b` vector and can `N_VClone(b)` for probe scratch. G1 may move the build back into Setup for explicit telemetry attribution if a stash mechanism for the template is added.

#### Scenario: Setup destroys prior handles + flags for rebuild; Solve performs actual build

- **WHEN** CVODE invokes `LS->ops->setup(LS, NULL)` for the K-th time during a solve session
- **THEN** wrapper destroys any prior `HYPRE_Solver` AMG handle + IJMatrix/IJVector handles, sets `pending_setup_called=1`, accumulates the destroy wall into `pending_setup_wall_sec`, and returns `SUNLS_SUCCESS`. The actual `HYPRE_BoomerAMGSetInterpType(amg, 6)` + `HYPRE_BoomerAMGSetCoarsenType(amg, 8)` + `HYPRE_BoomerAMGSetup(amg, par_A, par_b_dummy, par_x_dummy)` happens in the next Solve invocation (which has access to the in-flight `N_Vector b` for probe scratch).

#### Scenario: Setup probe restricted to topological bandwidth via MD->Ele[i].nabr

- **WHEN** wrapper's lazy-build (in first Solve after Setup) probes Jacobian columns
- **THEN** wrapper invokes `ATimes(e_col)` once per column `col` in `[0, NumY)` and reads out ONLY a small candidate row set derived from SHUD's element-neighbor topology (accessed via `MD->Ele[i].nabr[k]` for k=0..2 plus `MD->Ele[i].lakenabr[k]` plus `MD->Riv[i].down`), NOT all `NumY` rows. Total `ATimes` calls per Setup MUST be `≤ NumY` (one probe per column). Per-row candidate count MUST be `≤ bw_effective` with conservative cap `≤ 32`. The wrapper emits a one-time stdout line `[shud-amg] Setup probe bw_effective=<N> total_atimes_calls=<M>` recording the measured maximum per-column candidate count `N` and total `ATimes` call count `M` so PR-A smoke can confirm the topology restriction took effect.

#### Scenario: Setup uses constructor-stashed MD; no CVodeGetUserData chain

- **WHEN** wrapper's Setup needs SHUD topology to drive the probe
- **THEN** wrapper reads `Model_Data *MD` from its content struct (stashed at constructor time via the 5-arg `SUNLinSol_Hypre(y, MD, ...)` signature), then accesses `MD->Ele[i].nabr[k]` for topology; the wrapper does NOT invoke `CVodeGetUserData(cvode_mem, ...)` and does NOT dereference `udata` as a struct (which is an N_Vector solution vector with no `.y` or `.elem` member)

### Requirement: Solve callback delegates to HYPRE_BoomerAMGSolve and captures telemetry post-Solve

The wrapper's `Solve(LS, A, x, b, tol)` callback MUST invoke `HYPRE_BoomerAMGSolve(amg, par_A, par_b, par_x)`, copy the result back into the SUNDIALS `N_Vector x`, capture per-solve telemetry into the wrapper's internal ring buffer **immediately after the Solve completes** (NOT inside Setup — the Hypre `Get*` calls return values populated by `HYPRE_BoomerAMGSolve`, not by `HYPRE_BoomerAMGSetup`), and return the appropriate SUNDIALS 6.0 return code.

Hypre 3.1.0 public API substitution (spec amendment 2026-06-29): the earlier draft of this spec named `HYPRE_BoomerAMGGetCycleNumIterations` and `HYPRE_BoomerAMGGetCycleOpCount` as the telemetry source. These functions are NOT exposed by Hypre 3.1.0's public headers (verified via `grep` against `/opt/homebrew/include/HYPRE_parcsr_ls.h` on Mac brew 3.1.0). The wrapper uses the two available substitutes:
- `HYPRE_BoomerAMGGetNumIterations(amg, &n)` — per-Solve iteration count; dimensionally matches the original `CycleNumIterations` (a scalar iteration count populating the per-Solve `hypre_iters` ring-buffer field).
- `HYPRE_BoomerAMGGetCumNnzAP(amg, &nnz)` — cumulative nonzeros in the AP operator hierarchy. This is dimensionally a STATIC hierarchy-size count, NOT per-cycle work. Downstream consumers (PR-B aggregator) MUST treat `first_op_count` and `hypre_op_count` ring-buffer field as a hierarchy-size attestation, not a cycle-cost metric. Future Hypre releases that expose `CycleNumIterations` / `CycleOpCount` may be picked up via the `HYPRE_RELEASE_NUMBER` gate without further spec churn.

#### Scenario: Solve returns SUNLS_SUCCESS on Hypre convergence

- **WHEN** `HYPRE_BoomerAMGSolve` completes with Hypre return code `0`
- **THEN** wrapper returns `SUNLS_SUCCESS`; `LS->ops->numiters(LS)` returns the iteration count from `HYPRE_BoomerAMGGetNumIterations`

#### Scenario: Solve returns SUNLS_CONV_FAIL on Hypre divergence (recoverable)

- **WHEN** `HYPRE_BoomerAMGSolve` returns a divergence / non-convergence code (Hypre-specific positive int indicating max-iterations reached without convergence)
- **THEN** wrapper returns `SUNLS_CONV_FAIL` (SUNDIALS recoverable convergence failure, value defined at `sundials_linearsolver.h:204`); CVODE's outer Newton iteration may retry with smaller step

#### Scenario: Solve returns SUNLS_PACKAGE_FAIL_UNREC on Hypre structural failure

- **WHEN** `HYPRE_BoomerAMGSolve` returns a structural failure (NULL handle, OOM, internal Hypre error code distinct from non-convergence)
- **THEN** wrapper returns `SUNLS_PACKAGE_FAIL_UNREC` (SUNDIALS external-package unrecoverable failure, value defined at `sundials_linearsolver.h:199`), writes `MARKER:AMG_SOLVE_DIVERGE_DETECTED hypre_rc=<rc>` to stderr, and the cell_summary KV is emitted with `verdict_class=AMG_SOLVE_DIVERGE`

#### Scenario: Solve captures telemetry post-Solve (NOT in Setup)

- **WHEN** `HYPRE_BoomerAMGSolve` completes (success or non-convergent recoverable)
- **THEN** wrapper invokes `HYPRE_BoomerAMGGetNumIterations(amg, &iter_count)` AND `HYPRE_BoomerAMGGetCumNnzAP(amg, &op_count)` immediately after the Solve (Hypre 3.1.0 public-API substitutes for the unavailable `GetCycleNumIterations` / `GetCycleOpCount`), storing both values into the current ring-buffer entry along with `setup_wall_sec` (from the matching prior Setup call) and `solve_wall_sec`; these calls MUST NOT be made inside `Setup` (they return stale or uninitialized values on first invocation before any Solve has run)

### Requirement: Telemetry ring buffer captures per-Solve metrics with driver-side step context

The wrapper MUST maintain a fixed-size ring buffer (capacity `>= 100k entries`) inside its content struct, recording per-Solve metrics: `step_idx` (set by `SUNLinSol_Hypre_SetStepContext`), `t_sim` (set by `SUNLinSol_Hypre_SetStepContext`), `setup_called` (bool, true if a Setup callback fired since the last Solve), `hypre_iters` (from `HYPRE_BoomerAMGGetNumIterations` — Hypre 3.1.0 public-API substitute for the unavailable `GetCycleNumIterations`), `hypre_op_count` (from `HYPRE_BoomerAMGGetCumNnzAP` — Hypre 3.1.0 public-API substitute for the unavailable `GetCycleOpCount`; this is a STATIC hierarchy-size count, not a per-cycle work metric), `setup_wall_sec`, `solve_wall_sec`, `cvode_nli_step` (delta vs prior `SetStepContext` cumulative value), `cvode_nfeLS_step` (delta vs prior `SetStepContext` cumulative value). The buffer is drained by external telemetry harness in `tools/p8tune.G0/` via `SUNLinSol_Hypre_DrainTelemetry`.

The driver-side helper `SUNLinSol_Hypre_SetStepContext(SUNLinearSolver LS, long step_idx, realtype t_sim, long cvode_nli_cum, long cvode_nfeLS_cum)` is invoked by the SHUD driver loop (or by a `CVodeSetMonitorFn` callback if PR-0 task 1.5 confirms that API is present and viable in SUNDIALS 6.0) before each `CVode(...)` call to feed step-context into the wrapper. If `CVodeSetMonitorFn` is chosen, the wrapper attaches the monitor function during `Initialize` and drains step-context automatically; the driver does not call `SetStepContext` directly. PR-0 spike (task 1.5) records the chosen path in `PRE0_SPIKE_NOTES.md`.

#### Scenario: Ring buffer accepts entry and wraps on overflow

- **WHEN** wrapper Solve completes its 100001st call (capacity = 100k)
- **THEN** wrapper overwrites the oldest entry (entry index 0) and continues; an internal counter `entries_dropped_to_overflow` increments by 1

#### Scenario: External drain copies and clears ring buffer

- **WHEN** harness calls `SUNLinSol_Hypre_DrainTelemetry(LS, FILE* out)`
- **THEN** wrapper writes all valid entries as TSV rows to `out`, resets the buffer head/tail to zero, and resets `entries_dropped_to_overflow`

#### Scenario: SetStepContext provides driver-side CVODE counters per-step

- **WHEN** driver calls `SUNLinSol_Hypre_SetStepContext(LS, step_idx=K, t_sim=T, cvode_nli_cum=NL, cvode_nfeLS_cum=NFLS)` before `CVode(...)`
- **THEN** wrapper stores `(step_idx=K, t_sim=T)` for the next ring-buffer entry written by the upcoming Solve(s); the per-step `cvode_nli_step` = `NL - prev_NL`, `cvode_nfeLS_step` = `NFLS - prev_NFLS`; on the first call, `prev_NL = 0`, `prev_NFLS = 0`

### Requirement: Wrapper emits MARKER:AMG_TELEMETRY_REAL post-Solve

To distinguish G0 integrated telemetry from P8-tune.F's hard-coded estimate, the wrapper MUST emit `MARKER:AMG_TELEMETRY_REAL` to stdout exactly once per process lifetime, immediately after the first successful Solve that captures live HYPRE counters via `HYPRE_BoomerAMGGetNumIterations` AND `HYPRE_BoomerAMGGetCumNnzAP` (the Hypre 3.1.0 public-API substitutes for `GetCycleNumIterations` / `GetCycleOpCount`).

#### Scenario: Marker emits on first successful Solve

- **WHEN** wrapper's first call to `HYPRE_BoomerAMGGetNumIterations` returns a non-error value AND the matching `HYPRE_BoomerAMGGetCumNnzAP` call also succeeds, during the process lifetime
- **THEN** wrapper writes exactly the single line `MARKER:AMG_TELEMETRY_REAL hypre_release=<HYPRE_RELEASE_NUMBER> first_iters=<count> first_op_count=<count>` to stdout (where `first_op_count` is the `CumNnzAP` value — a STATIC hierarchy-size count, not a cycle-cost metric), and the marker is NOT re-emitted on subsequent Solves

#### Scenario: Marker absent if AMG path never executes

- **WHEN** a run completes with `SHUD_LINSOL=spgmr` (default) and no AMG Solve invocations
- **THEN** the run's stdout MUST NOT contain `MARKER:AMG_TELEMETRY_REAL`

### Requirement: Free callback releases all Hypre resources

The wrapper's `Free(LS)` callback MUST destroy the BoomerAMG solver handle, free the IJMatrix and IJVector handles, free the wrapper content struct (including ring buffer), and finalize Hypre via `HYPRE_Finalize()` if the wrapper holds the only outstanding initialization reference.

#### Scenario: Free leaves no Hypre allocations

- **WHEN** `SUNLinSolFree(LS)` is invoked on a wrapper that ran one or more successful Solves
- **THEN** after Free returns, `HYPRE_GetMemoryLocation`-tracked allocations attributable to this wrapper instance are zero

### Requirement: Wrapper code lives under SHUD submodule at SHUD/src/Equations/sunlinsol_hypre.{h,cpp}

The wrapper source MUST be placed at `SHUD/src/Equations/sunlinsol_hypre.h` (public header) + `SHUD/src/Equations/sunlinsol_hypre.cpp` (implementation). All SHUD submodule changes MUST be committed to the `openmp-baseline` long-lived branch (NEVER `master`); the outer `openMP` repository pointer is bumped via a follow-up commit.

#### Scenario: Wrapper header file path matches naming convention

- **WHEN** PR-0 lands
- **THEN** the SHUD submodule `openmp-baseline` branch contains a regular file at `src/Equations/sunlinsol_hypre.h` and `src/Equations/sunlinsol_hypre.cpp`

#### Scenario: Wrapper changes never pushed to SHUD master

- **WHEN** any G0 commit modifies `SHUD/src/Equations/sunlinsol_hypre.{h,cpp}`
- **THEN** the SHUD submodule branch where the commit lands MUST be `openmp-baseline` (verified via `git -C SHUD branch --show-current` at commit time); commit MUST NOT be pushed to SHUD `master`

