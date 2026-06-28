## Phase 4 Cross-Review Evidence Bundle (round 1, expanded fixture)

Reviewer agents: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf`
Review round: round 1
Reviewed head SHA: `1e45e16`
Local evidence: `.review-evidence/p8pre-pr-d-impl/{spec-compliance,correctness,integration,security-perf}.md`

### review-spec-compliance — APPROVE

Summary: All 10 checklist items PASS; spec/impl drift on `*jcurPtr` is implementer-correct (jok-mirror matches SUNDIALS canonical `cvDiurnal_kry.c` L716/L760), preserves spec gate-3 `nps>0 AND npe>0` intent.

Findings: **None.**

Key verification:
- Item 1 spec-drift resolution: spec L26 mandates unconditional `*jcurPtr = SUNFALSE`, but implementer used jok-mirror (`jok ? SUNFALSE : SUNTRUE`). Verified against SUNDIALS canonical `cvDiurnal_kry.c` L716/L760 — jok-mirror IS the canonical pattern. Spec internally inconsistent: literal L26 compliance would force npe=0 always (SUNDIALS only increments `npe` inside the cold-rebuild branch where jok==SUNFALSE), which would deterministically FAIL hard-gate-3 (spec L82-86 + D5 L107). Implementer honored the higher-priority gate-3 SHALL. Mac smoke npe=1232 confirms intent met.
- Item 2: profile_B0.yaml extras `t_precond_setup: 0.000037944` PASS at spec L108-113 location via timer.cpp L187-206 catch-all.
- Item 3: `merge-base 5276167 7a1dc8f = 7a1dc8f` (linear descendant verified).
- Items 4-7, 9-10 clean PASS. cvode_config.cpp L264-285 has all 4 SUNDIALS calls in canonical order with PREC_LEFT + LSetupFrequency(50); timer.cpp untouched; openspec/changes/, master plan, case_deployment_map, Makefile all empty diff.
- Item 8: SHUD commit 5276167 only on `openmp-baseline-p8pre` (local + remote); SHUD master tip unchanged — no master pollution.

### review-correctness — APPROVE

Summary: SHUD diff (3 files +101/-3) implements identity preconditioner stub + PREC_LEFT wire exactly per spec; signatures, body, registration, build, and runtime stats all verify.

Findings: **None.**

Key verification:
- Signatures (cvode_ls.h:57-63 vs MD_precond_identity.h L27-34): exact match; `extern "C"` wrapper correct.
- `PSetupIdentity` body: RAII Timer at L47, `*jcurPtr = jok ? SUNFALSE : SUNTRUE;` at L61, return 0. `PSolveIdentity` calls `N_VScale(SUN_RCONST(1.0), r, z)`.
- jok-mirror matches `cvDiurnal_kry.c` canonical L716/L724/L729/L760.
- cvode_config.cpp: #include L10, SPGMR PREC_LEFT at L264, CVodeSetPreconditioner + CVodeSetLSetupFrequency(50) inserted L282-285.
- Build: `make -n` picks up MD_precond_identity.cpp via auto-glob; nm shows both symbols T, CVodeSetPreconditioner U.
- Runtime stats: npe=1232, nps=209227, t_precond_setup=0.000037944 — all match PR commit message verbatim.
- Forward-only: `merge-base = 7a1dc8f` exactly; openmp-baseline-p8pre linear.
- Scope: exactly 3 expected files; rhs.cpp/shud.cpp/equations.h untouched.

Non-blocking note: PREC_LEFT is deprecated alias for SUN_PREC_LEFT in SUNDIALS 6.0.0 — both valid, consider future cosmetic upgrade.

### review-integration — APPROVE

Summary: PR is integration-clean — pointer bump only, SHUD master untouched, CVLS API call order conforms, smoke yaml carries t_precond_setup.

Findings: **None.**

Key verification:
- Outer diff exactly `SHUD | 2 +-`; single commit `feat(p8pre PR-D impl)` prefix; `Closes #345` trailer present.
- SHUD `origin/openmp-baseline-p8pre` HEAD = 5276167, strict descendant of 7a1dc8f; `origin/openmp-baseline` (master mirror) UNTOUCHED. C8 satisfied.
- Makefile glob `SRC = $(SRC_DIR)/Equations/*.cpp` (L383) auto-picks up `MD_precond_identity.cpp`.
- CVLS API call order in `cvode_config.cpp:269-284` correct: SUNLinSol_SPGMR(PREC_LEFT) → CVodeSetLinearSolver → CVodeSetPreconditioner → CVodeSetLSetupFrequency(50), each with check_flag.
- PR-E #346 build flag parity verified: same `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` as Step 1 PR-A (n8_profile_baseline.md §3.2). Mode C profile overhead cancels in PR-F gate-4 wall diff.
- Mac smoke yaml confirms `extras: t_precond_setup: 0.000037944`; soft-gate-6 path PASS via timer.cpp catch-all (L193-206).
- Mac vs server toolchain risk LOW: PSetup/PSolve are `extern "C"` with only SUNDIALS ABI types crossing boundary.
- /tmp namespace: PR-E uses `/tmp/p8pre_identity_spike/`, disjoint from PR-A's `/tmp/p8pre_n8_profile/`.

Forward concern: `PREC_LEFT` is a deprecated alias for `SUN_PREC_LEFT` in SUNDIALS 6.0.0 (`sundials_iterative.h:55-58`); works at 6.0.0 baseline but flag for ADR-0003 / future SUNDIALS bump.

### review-security-perf — APPROVE

Summary: Identity preconditioner stub is memory-safe, aliasing-safe, RAII-correct, and overhead-trivial (3.23e-7 of wall vs 5% gate); spike feasibility validated.

Findings: **None.**

Key verification:
- Memory safety: PSetupIdentity is POD-write + flag set; PSolveIdentity delegates to N_VScale_Serial whose source (`SHUD/cvode-6.0.0/src/nvector/serial/nvector_serial.c:531`) explicitly handles `z == x` aliasing.
- Timer RAII (timer.h:48-51) + atomic accumulation (timer.cpp:65) are thread-safe.
- Overhead: t_precond_setup = 37.944 μs over 209227 nps calls = 0.18 μs/call; 37.944 μs / 117.364 s = 3.23e-7 — 6 orders of magnitude headroom under 5% soft-gate-6 threshold; validates spike feasibility for future real preconditioners.
- `t_precond_setup` correctly NOT in `kKnownRawOrCanonical[]` (timer.cpp:187-190), auto-surfaces under `extras:` via catch-all at L204 — matches spec L108-113.

### Phase 4.5 Verifier — SKIPPED

Rationale: 0 PLAUSIBLE candidates — 4/4 reviewers APPROVE, all findings = None. Phase 4.5 requires verifier on PLAUSIBLE/CONFIRMED candidates, none exist.

### Round 1 verdict

**Clean.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE. Proceed to Phase 7.

### Forward action notes (not merge-blocking)

- Future openspec patch (carry to PR-F #347 / #349 openspec archive): cite SUNDIALS `cvDiurnal_kry.c` L716/L760 jok-mirror canonical in spec L26 to resolve internal inconsistency.
