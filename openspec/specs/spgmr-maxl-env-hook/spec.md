# spgmr-maxl-env-hook Specification

## Purpose
TBD - created by archiving change p8tune-spgmr-maxl. Update Purpose after archive.
## Requirements
### Requirement: Env-var helper function

The system SHALL provide a `static int get_spgmr_maxl_from_env(void)` helper in `cvode_config.cpp` that returns the validated maxl value from environment variable `SHUD_SPGMR_MAXL`, with `0` (SUNDIALS default) returned when the variable is unset, empty, or invalid.

#### Scenario: Unset or empty env returns 0
- WHEN the env var `SHUD_SPGMR_MAXL` is unset OR set to empty string `""`
- THEN `get_spgmr_maxl_from_env()` SHALL return `0`
- AND `SUNLinSol_SPGMR(udata, PREC_NONE, 0, sunctx)` SHALL be invoked (SUNDIALS docs: maxl ≤ 0 → use default 5)
- AND no log line SHALL be emitted (silent default behavior)

#### Scenario: Explicit zero behaves identically to unset
- WHEN the env var `SHUD_SPGMR_MAXL` is explicitly set to `"0"`
- THEN `get_spgmr_maxl_from_env()` SHALL return `0`
- AND `SUNLinSol_SPGMR(udata, PREC_NONE, 0, sunctx)` SHALL be invoked (SUNDIALS docs: maxl ≤ 0 → use default 5)
- AND no log line SHALL be emitted (silent default behavior; bit-identical to unset case)
- AND the keliya smoke output SHALL be SHA12 + 15-key bit-identical to `unset SHUD_SPGMR_MAXL && ./shud keliya`

#### Scenario: Valid value is propagated
- WHEN the env var `SHUD_SPGMR_MAXL` is set to one of `{5, 10, 15, 20, 30}` (string form)
- THEN `get_spgmr_maxl_from_env()` SHALL parse and return the integer value
- AND `SUNLinSol_SPGMR(udata, PREC_NONE, <maxl>, sunctx)` SHALL be invoked
- AND a log line `[CVODE] SPGMR maxl=<maxl> pretype=PREC_NONE` SHALL be emitted to stdout for provenance

#### Scenario: Invalid value aborts startup
- WHEN the env var `SHUD_SPGMR_MAXL` is set to a value NOT in `{"", "0", "5", "10", "15", "20", "30"}` (e.g., `"7"`, `"50"`, `"foo"`, `"-1"`)
- THEN `get_spgmr_maxl_from_env()` SHALL emit `[CVODE] ERROR: SHUD_SPGMR_MAXL must be unset, 0, 5, 10, 15, 20, or 30 (got: <value>)` to stderr
- AND it SHALL call `myexit(ERRCVODE)` (project convention) to abort SHUD startup
- AND no SPGMR allocation SHALL proceed (fail-fast principle)

### Requirement: No regression to PREC_LEFT or preconditioner registration

The change SHALL preserve the cleaned `PREC_NONE` codepath without introducing any `PREC_LEFT` usage or preconditioner registration.

#### Scenario: PREC_LEFT absence verification
- WHEN PR-C is reviewed
- THEN `grep -rE 'PREC_LEFT' SHUD/src/Equations/cvode_config.cpp` SHALL return 0 matches
- AND `grep -rE 'CVodeSetPreconditioner' SHUD/src/Equations/cvode_config.cpp` SHALL return 0 matches
- AND `grep -rE 'CVodeSetLSetupFrequency' SHUD/src/Equations/cvode_config.cpp` SHALL return 0 matches
- AND no `MD_precond_identity.{h,cpp}` file SHALL be reintroduced

### Requirement: Default-unset bit-identical equivalence verification

The change SHALL include a CI-gated keliya smoke-test artifact proving that with env var unset, the resulting binary's keliya smoke run produces bit-identical CVODE counters and `rivqdown.dat` SHA12 to the cleaned `PREC_NONE` baseline established in capability `clean-prec-none-baseline`.

#### Scenario: keliya smoke 4-way equivalence
- WHEN PR-C build is produced and tested
- THEN orchestrator SHALL run 4 keliya smoke invocations on the patched SHUD binary:
  1. `unset SHUD_SPGMR_MAXL && ./shud keliya`
  2. `SHUD_SPGMR_MAXL= ./shud keliya` (empty string)
  3. `SHUD_SPGMR_MAXL=0 ./shud keliya`
  4. `SHUD_SPGMR_MAXL=5 ./shud keliya`
- AND collect `rivqdown.dat` SHA12 + `cvode_stats.txt` 15-key snapshot per run
- AND ALL 4 invocations SHALL produce bit-identical SHA12 + 15-key (per SUNDIALS docs: `maxl ≤ 0` → default 5; the env-var hook MUST honor this 4-way equivalence)
- AND ALL 4 invocations SHALL be bit-identical to the keliya cleaned-PREC_NONE smoke artifact from `docs/p8tune/clean_prec_none_baseline.md` §keliya-smoke-anchor
- AND any SHA12 mismatch OR counter-value mismatch in ANY of the 4 invocations SHALL block PR-C merge

### Requirement: Single-file source surface

The env-var hook implementation SHALL be confined to `SHUD/src/Equations/cvode_config.cpp`.

#### Scenario: Source change scope
- WHEN PR-C is reviewed
- THEN `git diff <merge_base>..HEAD --name-only` SHALL list only `src/Equations/cvode_config.cpp` (within SHUD submodule)
- AND no other SHUD source file SHALL be modified
- AND no SUNDIALS or external library headers SHALL be added or removed
- AND no Makefile changes SHALL be introduced

