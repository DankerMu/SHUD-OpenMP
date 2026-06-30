# shud-linsol-selector Specification

## Purpose
TBD - created by archiving change p8tune-g0-instrumented-amg-smoke. Update Purpose after archive.
## Requirements
### Requirement: `SHUD_LINSOL` env var parsed at the top of SetCVODE, BEFORE CVodeCreate

The CVODE configuration setup function `SetCVODE` in `SHUD/src/Equations/cvode_config.cpp` MUST read the `SHUD_LINSOL` environment variable exactly once per process, **at the top of `SetCVODE` before the `CVodeCreate(CV_BDF, sunctx)` call at L310**. The parsed value selects the linear solver backend. Factory dispatch and `CVodeSetLinearSolver` invocation occur later at the existing L324/L327 site, but the parse + fatal-exit-on-unknown happens first so that no CVODE state has been allocated when the fatal fires.

#### Scenario: Default (env var unset) selects SPGMR with zero behavior change

- **WHEN** SHUD launches and `SHUD_LINSOL` is unset in the environment
- **THEN** `parse_linsol_env()` returns `LINSOL_SPGMR` at the top of `SetCVODE`, the configured linear solver MUST be `SUNLinSol_SPGMR` constructed with the same arguments as the pre-G0 baseline (`PREC_NONE`, `maxl = SHUD_SPGMR_MAXL env-var hook value`), and a downstream bit-identical run on the `keliya` SHORT 90-day case (or equivalent regression) MUST produce identical output bytes vs the pre-G0 baseline archive at `.review-evidence/g0-spgmr-baseline-90day-keliya/`

#### Scenario: `SHUD_LINSOL=spgmr` (explicit) selects SPGMR identically to unset

- **WHEN** SHUD launches and `SHUD_LINSOL=spgmr`
- **THEN** the configured linear solver and downstream run output MUST be byte-identical to the unset case

#### Scenario: `SHUD_LINSOL=amg` selects Hypre BoomerAMG via wrapper

- **WHEN** SHUD launches and `SHUD_LINSOL=amg`
- **THEN** the configured linear solver MUST be `SUNLinSol_Hypre(udata, 6, 8, sunctx)` (G0 hardcoded `interp_type=6, coarsen_type=8`), where `udata` is the `N_Vector` argument that `SetCVODE` receives directly (matching the existing baseline call `SUNLinSol_SPGMR(udata, ...)` at L324 — `udata` is itself an N_Vector, NOT a `UserData` struct, and has no `->y` member); `CVodeSetLinearSolver(cvode_mem, LS, NULL)` MUST be invoked with `SUNMatrix = NULL`

### Requirement: Unknown `SHUD_LINSOL` value rejected with explicit fatal error BEFORE CVodeCreate

The parser MUST refuse to start with an explicit error when `SHUD_LINSOL` is set to any value other than `spgmr` or `amg`. Silent fallback (e.g., defaulting to SPGMR on `SHUD_LINSOL=amG` typo) is forbidden. The fatal exit MUST occur before `CVodeCreate(CV_BDF, sunctx)` so that no CVODE state has been allocated when the process aborts.

#### Scenario: Unrecognized value triggers FATAL exit before CVodeCreate

- **WHEN** SHUD launches with `SHUD_LINSOL=amG` (case-sensitive mismatch) or `SHUD_LINSOL=KLU` or `SHUD_LINSOL=foobar`
- **THEN** the process MUST write to stderr the single line `[shud] FATAL: SHUD_LINSOL=<value> unrecognized; accepted: spgmr, amg` AND exit with non-zero status BEFORE `CVodeCreate(CV_BDF, sunctx)` is invoked (caller is the `parse_linsol_env()` helper inside `SetCVODE`, called above the `CVodeCreate` line at L310)

#### Scenario: Whitespace and empty string both treated as unset

- **WHEN** `SHUD_LINSOL=""` (empty string) or `SHUD_LINSOL="   "` (whitespace only)
- **THEN** behavior MUST match unset case (default to SPGMR); these are NOT treated as unrecognized values

### Requirement: Hypre runtime dylib precondition for AMG path

When `SHUD_LINSOL=amg`, the SHUD process MUST verify that the Hypre runtime dylib is loadable before any CVODE state is allocated. If the dylib cannot be resolved (missing `LD_LIBRARY_PATH` on a server cn-node, broken brew install on macOS, etc.), the process MUST fail fast with a clear diagnostic. When `SHUD_LINSOL` is unset or `=spgmr`, the wrapper makes zero Hypre calls and a missing Hypre runtime is irrelevant at the SHUD level (though the link-always policy means the dynamic loader itself may fail before `main()` — that is a separate platform-level error documented as a known trade-off of D2 link-always).

#### Scenario: AMG path fails fast on missing Hypre runtime

- **WHEN** SHUD launches with `SHUD_LINSOL=amg` AND the Hypre dylib cannot be dlopened (e.g., `LD_LIBRARY_PATH` does not include `/scratch/frd_muziyao/local/hypre-3.1.0/lib` on a server cn-node)
- **THEN** the process MUST write to stderr `[shud] FATAL: Hypre runtime missing (set LD_LIBRARY_PATH to <HYPRE_LIBDIR>)` AND exit non-zero BEFORE `CVodeCreate(CV_BDF, sunctx)` is invoked

#### Scenario: SPGMR default path tolerates absent Hypre at SHUD level

- **WHEN** SHUD launches with `SHUD_LINSOL=spgmr` (or unset) AND the Hypre dylib was loaded at process start (link-always) but is otherwise unused
- **THEN** SHUD MUST NOT emit any Hypre-related diagnostic, MUST NOT call `HYPRE_Init` or `HYPRE_Initialize`, and the SPGMR path completes exactly as under pre-G0 behavior

### Requirement: Enum-driven factory dispatch in cvode_config.cpp

The linsol selection MUST be implemented as an enum `linsol_t = { LINSOL_SPGMR = 0, LINSOL_AMG = 1, LINSOL_UNKNOWN = -1 }` plus per-backend factory functions (`create_spgmr_ls`, `create_amg_ls`). Both factories take `N_Vector y` as the first argument (matching the existing `SUNLinSol_SPGMR(udata, ...)` call signature at L324 where `udata` is itself an N_Vector). The call site at the existing `CVodeSetLinearSolver` location MUST invoke `LS = create_<selected>_ls(udata, sunctx)` then pass `LS` to CVODE. Adding a third backend in the future MUST require only a new enum value + new factory function, with no branching in the call site beyond the enum dispatch.

#### Scenario: Factory function returns valid LS for SPGMR

- **WHEN** `create_spgmr_ls(N_Vector y, SUNContext sunctx)` is invoked
- **THEN** function returns the same `SUNLinearSolver` handle that the pre-G0 baseline produced (constructor args matching the existing `SUNLinSol_SPGMR(y, PREC_NONE, get_spgmr_maxl_from_env(), sunctx)` call); the parameter is named `y` for clarity but accepts the same `udata` N_Vector the caller passes from `SetCVODE`

#### Scenario: Factory function returns valid LS for AMG

- **WHEN** `create_amg_ls(N_Vector y, SUNContext sunctx)` is invoked
- **THEN** function returns a `SUNLinearSolver` constructed via `SUNLinSol_Hypre(y, 6, 8, sunctx)` and never returns NULL on a healthy environment (Hypre linked, BoomerAMG available); on environment failure (e.g., Hypre dylib missing, which should be caught earlier by the Hypre-runtime precondition check but is defensively guarded), factory returns NULL and writes `[shud] FATAL: AMG factory failed; check Hypre install` to stderr before the caller exits non-zero

### Requirement: Existing `SHUD_SPGMR_MAXL` hook unchanged

The G0 changes MUST NOT modify the `SHUD_SPGMR_MAXL` env-var parsing logic at `SHUD/src/Equations/cvode_config.cpp:248-295`. That hook continues to feed the `maxl` argument to `create_spgmr_ls`. AMG path ignores `SHUD_SPGMR_MAXL`.

#### Scenario: `SHUD_SPGMR_MAXL=10` still respected under SPGMR path

- **WHEN** `SHUD_LINSOL=spgmr SHUD_SPGMR_MAXL=10` and SHUD runs
- **THEN** `create_spgmr_ls` calls `SUNLinSol_SPGMR(y, PREC_NONE, 10, sunctx)` (maxl=10, mirror of pre-G0 behavior under MAXL env var)

#### Scenario: `SHUD_SPGMR_MAXL` ignored under AMG path

- **WHEN** `SHUD_LINSOL=amg SHUD_SPGMR_MAXL=10` and SHUD runs
- **THEN** AMG factory ignores the `MAXL` env var (no warning needed); the factory produces the same AMG LS regardless of `SHUD_SPGMR_MAXL` value

### Requirement: Selector emits stdout marker identifying active backend, BEFORE CVodeCreate

At CVODE init time, after the linsol selector decides (and after the Hypre-runtime precondition check for AMG path), the configurator MUST emit a single stdout line `[shud] linsol=<spgmr|amg> source=<env|default>` for runtime traceability. The marker MUST precede `CVodeCreate(CV_BDF, sunctx)` so that post-mortem log inspection can confirm which backend was selected even on early crash.

#### Scenario: Default path emits source=default

- **WHEN** SHUD runs with unset `SHUD_LINSOL`
- **THEN** stdout contains exactly one line `[shud] linsol=spgmr source=default`

#### Scenario: Env-overridden AMG path emits source=env

- **WHEN** SHUD runs with `SHUD_LINSOL=amg`
- **THEN** stdout contains exactly one line `[shud] linsol=amg source=env`

#### Scenario: Marker emitted before CVode initialization

- **WHEN** `[shud] linsol=...` is emitted
- **THEN** no `CVode*` calls have yet been issued by the SHUD driver; the marker precedes the `CVodeCreate(CV_BDF, sunctx)` invocation at L310 (so post-mortem log inspection can confirm which backend was selected even on early crash)

