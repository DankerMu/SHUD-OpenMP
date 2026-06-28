Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 01fd43fb0bd68795393aa81d5b07e982bb1c6d52
Summary: All 8 correctness checks PASS; doc citations exact, fork is degenerate forward-only descendant of 7a1dc8f, REJECT distinction technically sound.
Findings:
- None.
Non-blocking notes:
- Check 1 (4 grep API accuracy): All confirmed exact.
  - cvode.h:132 CVodeSetLSetupFrequency — PASS
  - cvode_ls.h:91 CVodeSetJacEvalFrequency — PASS
  - cvode_ls.h:57 typedef CVLsPrecSetupFn — PASS (doc also cites L99 for CVodeSetPreconditioner arg; L99 verified)
  - cvode_ls.h:61 typedef CVLsPrecSolveFn — PASS (doc also cites L100 arg; L100 verified)
- Check 2 (REJECT distinction): cvode.h:133 CVodeSetMaxConvFails confirmed adjacent to L132. Read of L128-142 confirms both as separate SUNDIALS_EXPORT decls. Signature contrast in doc §3 table is technically correct: msbp (long, step count, linear-solver setup loop) vs maxncf (int, failure count, nonlinear conv subsystem). Not interchangeable.
- Check 3 (SHUD fork): submodule HEAD = 7a1dc8f6ea9e5496f516255406ee3563d397959b. origin/openmp-baseline-p8pre = same. local branch = same. merge-base = same. All four 7a1dc8f.
- Check 4 (outer pointer): `git submodule status SHUD` shows ` 7a1dc8f...` (leading space = clean, no +/-). Pointer NOT bumped in this PR. Doc §5 L150-153 explicitly confirms .gitmodules untouched.
- Check 5 (forward-only math): Degenerate case (branch HEAD == fork base). merge-base(HEAD, 7a1dc8f) == 7a1dc8f is provably satisfied. Doc §5 L130-140 spells this out and correctly anticipates the post-PR-D state where rev-parse check fails (expected) but merge-base check still holds (required).
- Check 6 (cross-ref): spec.md L148-154 Scenario "Step 2 forward-only descendant extension" exists exactly at cited lines. Content matches the doc's quoted requirements.
- Check 7 (no SHUD source mod): branch log shows top commit 7a1dc8f only; 284 total commits is the full history reachable from 7a1dc8f (matches openmp-baseline lineage), no commits ahead of fork base.
- Check 8 (openspec strict): `openspec validate p8pre-spike --strict` → "Change 'p8pre-spike' is valid".
