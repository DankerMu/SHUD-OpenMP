# Integration review — PR #357 (p8pre PR-D impl)

Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 1e45e16
Summary: PR is integration-clean — pointer bump only, SHUD master untouched, API order conforms, smoke yaml carries t_precond_setup; one forward-compat note on PREC_LEFT deprecated alias.

Findings:
- None.

Non-blocking notes:

1. **SHUD upstream state (CONFIRMED)**: `origin/openmp-baseline-p8pre` HEAD = `5276167` (new commit, +137/-3 in src/Equations/), `origin/openmp-baseline` HEAD = `7a1dc8f` (untouched master discipline), `origin/master` unchanged. `.gitmodules` URL still `https://github.com/SHUD-System/SHUD.git`. C8 satisfied. SHUD log shows strict descendant chain: `7a1dc8f → 5276167` (single forward-only commit).

2. **Outer pointer bump format (CONFIRMED)**: Single commit `1e45e16` on `feat/issue-345-p8pre-pr-d-impl`; diff is exactly `SHUD | 2 +-` (one pointer line). Commit subject `feat(p8pre PR-D impl): …` matches `feat(p8pre)` prefix convention. Trailer `Closes #345` + `Refs #338` present. No other outer file modifications.

3. **PR-E #346 server build readiness (READY)**: Server clones via submodule URL + `git fetch origin openmp-baseline-p8pre` + checkout `5276167`. Same build invocation `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` as Step 1 PR-A (per `docs/p8pre/n8_profile_baseline.md` §3.2 + L116). Makefile `SRC = $(SRC_DIR)/Equations/*.cpp` wildcard (Makefile:383) auto-picks up `MD_precond_identity.cpp` — no Makefile edit needed.

4. **PR-F #347 gate-4 wall comparator parity (CONFIRMED)**: Build matrix invariant between Step 1 baseline (PR-A) and Step 2 identity-spike (PR-E) — both use `SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`. `Mode C profile` overhead cancels in `|wall_identity − wall_baseline|` per `n8_profile_baseline.md` §286.

5. **t_precond_setup forward compat (VERIFIED)**: Mac smoke yaml at `SHUD/Basins/keliya/output/keliya.out/profile_B0.yaml` contains `extras: t_precond_setup: 0.000037944` — surfaced by `tools/profile/timer.cpp` catch-all loop (L193-206) because not in `kKnownRawOrCanonical[]` (L187-191). PR-F #347 aggregator path confirmed.

6. **CVLS API call order (CONFORMS)**: `cvode_config.cpp:269-284` reads (1) SUNLinSol_SPGMR + check_flag → (2) CVodeSetLinearSolver + check_flag → (3) CVodeSetPreconditioner + check_flag → (4) CVodeSetLSetupFrequency + check_flag. Matches spec mandate. Preconditioner registration AFTER LinearSolver attach is correct per cvode_ls.h docs.

7. **PREC_LEFT alias forward concern (NOTE)**: SUNDIALS 6.0.0 `sundials_iterative.h:55-58` marks `PREC_LEFT` as deprecated alias for `SUN_PREC_LEFT` (both `=1`, enum-compatible). Works for 6.0.0 baseline but a future SUNDIALS bump (6.x.y / 7.x) could remove the alias. Not in PR-D scope; flag as forward concern for ADR-0003 / SUNDIALS upgrade epic.

8. **Mac vs server toolchain risk (LOW)**: Mac smoke built with Apple Clang + libomp; server PR-E will use gcc 13.3.0 + libgomp. PSetup/PSolve use only `extern "C"` linkage + SUNDIALS realtype/N_Vector ABI types — no C++ object-passing across the boundary. `shud_profile::Timer` is RAII-local inside PSetupIdentity, no cross-TU layout exposure. Build flag matrix identical to Step 1 PR-A which already ran on the same server toolchain. Risk: low; no special flag for PR-E.

9. **Sibling tools impact (NONE)**: `git diff HEAD~1..HEAD --stat` shows only `SHUD | 2 +-`. `tools/profile/timer.cpp`, `tools/cvode_stats_diff/`, `tools/p8pre/{run,aggregate,render}*` all unchanged. §6.0a pre-check holds.

10. **`openspec validate p8pre-spike --strict` (PASS)**: `Change 'p8pre-spike' is valid` (CLI exit 0).

11. **CI workflow compat (LOW RISK)**: SHUD `src/Equations/*.cpp` wildcard means CI rebuild picks up the new TU automatically. `PREC_LEFT` enum value `1` is well-defined in `sundials_iterative.h` — no compile-time risk on asan-ubsan keliya. New CVLS preconditioner call path may exercise more SUNDIALS code under sanitizers, but `N_VScale(1.0, r, z)` and `*jcurPtr` write are bounded ops. Worth a flag for PR-E's CI verifier as "needs verification" if asan-ubsan workflow trips.

12. **#341 /tmp mirror namespace (CLEAN)**: PR-E #346 will produce a separate `/tmp/p8pre_identity_spike/` namespace (per spec). PR-A's `/tmp/p8pre_n8_profile/` is read-only baseline. No overwrite.

Verdict: APPROVE — pointer bump diff is minimal and correct; SHUD changes conform to API + master-discipline contract; all downstream consumers (PR-E/F/G) have clean handoff.
