Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 01fd43fb0bd68795393aa81d5b07e982bb1c6d52
Summary: PR is a clean docs-only pre-flight; all 8 integration checklist items PASS independently; ready to merge as PR-D entry-evidence anchor.

Findings:
- None.

Non-blocking notes:

1. SHUD submodule workflow (C8) — PASS
   - `git submodule status SHUD` confirms pointer at `7a1dc8f6...` (P1e ship pin, untouched by this PR).
   - `git diff main..HEAD -- .gitmodules` returns empty: `.gitmodules` untouched (`branch = openmp-baseline` field preserved per spec L154).
   - `openmp-baseline-p8pre` is a NEW branch on `origin` (SHUD-System/SHUD), forked from `7a1dc8f`, not pollution of `master` or `openmp-baseline`; push went through DankerMu's existing permission, no fork URL needed (fork_evidence.txt L18-21).

2. Fork base anchoring — PASS
   - `git ls-remote --heads origin openmp-baseline-p8pre` → `7a1dc8f6...` (matches expected fork base exactly).
   - Strict criterion `rev-parse openmp-baseline-p8pre == 7a1dc8f` holds (will naturally flip to merge-base-only criterion after PR-D impl #345 adds commits — doc §5 L135-140 correctly anticipates this).

3. API verification — independently reproduced
   - All 5 grep lines match raw evidence:
     - `cvode.h:132 CVodeSetLSetupFrequency` PASS
     - `cvode.h:133 CVodeSetMaxConvFails` (REJECT-control) PASS
     - `cvode_ls.h:57 CVLsPrecSetupFn` typedef PASS
     - `cvode_ls.h:61 CVLsPrecSolveFn` typedef PASS
     - `cvode_ls.h:91 CVodeSetJacEvalFrequency` PASS
     - `cvode_ls.h:99-100 CVodeSetPreconditioner` args PASS
   - §3 REJECT rationale (msbp vs maxncf subsystem distinction) is technically accurate and pre-empts a likely reviewer-confusion path.

4. Doc structure — PASS for lightweight evidence-log convention
   - 6 § + YAML metadata structure, ~169 lines; correctly NOT academic-paper-style (per brief). Self-contained for #348 ADR-0003 Step-2-API-entry citation.

5. Outer repo sanity — PASS
   - Diff is exactly 1 file (+169 lines): `docs/p8pre/api_verification.md`. No CI / code / spec / submodule-pointer side-effects.

6. Forward dep cross-ref — PARTIAL but acceptable
   - §1 + §5 cite `PR-D impl (#345)` (twice). PR-E #346 / PR-F #347 not explicitly cited, but the brief allows this — they consume the resulting binary, not this doc directly. ADR-0003 (#348) consumption is structural (§6 references list), not by issue number, which is appropriate for a pre-flight entry-evidence anchor that predates the ADR.

7. CI compat — PASS
   - No `.github/workflows/*.yml` glob references `docs/p8pre/`.
   - `openspec validate p8pre-spike --strict` → `Change 'p8pre-spike' is valid`.

8. Idempotency — accept, with minor observation
   - fork_evidence.txt L9-11 shows the implementer's pre-check `git ls-remote --heads origin openmp-baseline-p8pre` returning empty before branch creation (graceful detect path). If re-run with branch already on origin, `git push -u` would fail-loud rather than silently overwrite — this is desirable strictness for a fork anchor. Not worth gating.

Verdict: APPROVE — docs-only pre-flight slice; all integration constraints satisfied; sets up clean linear-descendant pointer chain for downstream #345/#346/#347/#348.
