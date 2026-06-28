Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 1e45e16
Summary: SHUD diff (3 files +101/-3) implements identity preconditioner stub + PREC_LEFT wire exactly per spec; signatures, body, registration, build, and runtime stats all verify. APPROVE.

Findings:
- None.

Non-blocking notes:
- Signature accuracy (item 1): MD_precond_identity.h L27-34 PSetupIdentity/PSolveIdentity match SUNDIALS 6.0.0 cvode_ls.h:57-63 CVLsPrecSetupFn/CVLsPrecSolveFn typedefs verbatim. `extern "C"` wrapper L23/L37 correct for CVODE registration.
- Body accuracy (item 2): MD_precond_identity.cpp PSetupIdentity has `Timer _t("t_precond_setup")` at function head (L47 under SHUD_ENABLE_PROFILE guard L43, defensible per project convention), `*jcurPtr = jok ? SUNFALSE : SUNTRUE;` at L61, returns 0. PSolveIdentity L65-71 calls `N_VScale(SUN_RCONST(1.0), r, z)` and returns 0. jok-mirror matches cvDiurnal_kry.c canonical pattern L716/L724 (jok==SUNTRUE → SUNFALSE) and L729/L760 (jok==SUNFALSE → SUNTRUE).
- cvode_config.cpp edit (item 3): #include placed L10 after MD_diagnostics.hpp with clear comment block. SPGMR flipped to `(udata, PREC_LEFT, 0, sunctx)` at L264. CVodeSetPreconditioner + CVodeSetLSetupFrequency(50) inserted L282-285 with check_flag style matching surrounding code. Order alloc-LS → SetLinearSolver → SetPreconditioner matches SUNDIALS docs.
- Build sanity (item 4): `make -n shud SHUD_ENABLE_OPENMP_RHS=1` source list includes `src/Equations/MD_precond_identity.cpp` (auto-glob). nm: `_PSetupIdentity T`, `_PSolveIdentity T` defined; `_CVodeSetPreconditioner U` imported. Both build matrices verified.
- Runtime stats (item 5): cvode_stats.txt L8 `npe=1232`, L9 `nps=209227`; profile_B0.yaml L15 `t_precond_setup: 0.000037944` (~38µs total / ~31ns × 1232 calls, consistent). All three values match PR commit 5276167 verbatim.
- Bitwise neutrality (item 6): keliya is out of P8 spec scope (heihe + heihe_x4 only). Spec gates this PR exercises are: signatures match, body returns 0, nps/npe accumulate, t_precond_setup emits. All four PASS. Soft-gate-5 cross-N tolerance (max_ulp ≤ 1024) — non-blocker, defer to heihe smoke.
- Forward-only descendant (item 7): `git merge-base 5276167 7a1dc8f` = `7a1dc8f6ea9e5...` exactly. openmp-baseline-p8pre log shows linear descent. Strict gate satisfied.
- Scope discipline (item 8): exactly 3 expected files changed; rhs.cpp, shud.cpp, equations.h all untouched.
- Minor: SHUD_ENABLE_PROFILE guard around the Timer (L43-48) means t_precond_setup only emits under PROFILE=1; documented in header comment L28-33 matching f.cpp convention. Soft-gate-6 evidence requires PROFILE=1 — implementer's smoke was PROFILE=1, sound.
- Minor: PREC_LEFT (sundials_iterative.h L58) is the deprecated alias for SUN_PREC_LEFT. Both valid in 6.0.0 — consider cosmetic upgrade to SUN_PREC_LEFT in a future PR. Not actionable now.
