## Phase 4 Cross-Review Evidence Bundle (round 1)

Reviewer agents: `review-correctness`, `review-integration`
Review round: round 1
Reviewed head SHA: `01fd43f`
Local evidence: `.review-evidence/p8pre-pr-d-preflight/{correctness,integration}.md`

### review-correctness — APPROVE

Summary: All 8 correctness checks PASS; doc citations exact, fork is degenerate forward-only descendant of 7a1dc8f, REJECT distinction technically sound.

Findings: **None.**

Verification:
- 4 grep refs all exact: `CVodeSetLSetupFrequency` cvode.h:132, `CVodeSetJacEvalFrequency` cvode_ls.h:91, `CVLsPrecSetupFn` cvode_ls.h:57, `CVLsPrecSolveFn` cvode_ls.h:61.
- REJECT distinction: cvode.h:133 `CVodeSetMaxConvFails` confirmed adjacent to L132 `CVodeSetLSetupFrequency`. Signature contrast (msbp long step-count linear-solver vs maxncf int fail-count nonlinear) is correct.
- Submodule HEAD / origin branch / local branch / merge-base all == `7a1dc8f6ea9e5496f516255406ee3563d397959b`.
- `git submodule status SHUD` shows clean pointer 7a1dc8f, NOT bumped.
- Doc §5 L130-140 correctly distinguishes pre-flight state (both strict checks PASS) from post-PR-D state (rev-parse fails as expected after commits land, merge-base must hold).
- spec.md L148-154 Scenario "Step 2 forward-only descendant extension" exists at cited lines and matches doc quotes.
- Branch log top = 7a1dc8f only; zero commits ahead of fork base.
- `openspec validate p8pre-spike --strict` → valid.

### review-integration — APPROVE

Summary: PR is a clean docs-only pre-flight; all 8 integration checklist items PASS independently; ready to merge as PR-D entry-evidence anchor.

Findings: **None.**

Non-blocking notes:
- **C8 submodule workflow PASS**: `.gitmodules` untouched (empty diff); SHUD pointer still 7a1dc8f; openmp-baseline-p8pre is a NEW origin branch (not master pollution), DankerMu push permission honored without fork URL change.
- **Fork base anchoring PASS**: `ls-remote` confirms 7a1dc8f exactly; doc §5 L135-140 correctly anticipates the criterion flip (strict-equality → merge-base-only) once PR-D impl #345 adds commits.
- **API verification independently reproduced**: all 5 grep lines match raw evidence. §3 REJECT rationale (msbp step-count vs maxncf failure-count) pre-empts a likely reviewer-confusion path correctly.
- **Doc structure PASS**: 169 lines, YAML + 6 §, lightweight evidence-log convention (NOT academic-paper style) per brief intent. Self-contained for ADR-0003 (#348) Step-2 citation.
- **Outer repo sanity PASS**: diff is exactly 1 file; no CI/code/spec/pointer side-effects.
- **Forward dep cross-ref PARTIAL-but-acceptable**: §1 + §5 cite #345 twice; #346/#347 not explicitly cited but they consume the binary not the doc; #348 ADR-0003 consumption is structural via §6 references list, appropriate for a pre-flight entry anchor.
- **CI compat PASS**: no workflow globs `docs/p8pre/`; `openspec validate p8pre-spike --strict` → valid.
- **Idempotency acceptable**: fork_evidence.txt L9-11 shows pre-check ls-remote returning empty; re-run with branch existing on origin would fail-loud on `git push -u` (desirable strictness, not worth gating).

### Phase 4.5 Verifier — SKIPPED

Rationale: 0 candidates with concrete failure scenarios — 2/2 reviewers APPROVE, all findings = None. Phase 4.5 precision-bias on compact fixture requires verifier on PLAUSIBLE/CONFIRMED candidates, none exist.

### Round 1 verdict

**Clean.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE. Proceed to Phase 7.
