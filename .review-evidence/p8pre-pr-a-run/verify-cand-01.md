# Verifier verdict: cand-01-gitignore-carry-forward

- **Candidate ID**: cand-01-gitignore-carry-forward
- **Originating reviewer**: review-integration (round 1)
- **Reviewed head SHA**: d8602d0fb4e609c106eaa5f79e973830182ac150
- **Verdict**: CONFIRMED

## Evidence

1. `.gitignore:16` declares `.s*-runs/` which `git check-ignore -v` confirms matches `.s5b-runs/x` (exit 0) — glob anchor is the literal `s` after the dot, so `.p8pre-runs/x` does NOT match (`git check-ignore` exit 1, verified live).
2. `.gitignore:23-25` explicitly enumerates P1d siblings (`.p1d-runs/`, `.p1d-pr-*-runs/`, `.p1d-pr-*-worktree/`) and L28-31 enumerates P1e (`.p1e-runs/`, `.p1e-pr-*-runs/`, `.p1e-pr-*-worktree/`). This is the precedent pattern reviewer cited; `.p8pre-runs/` is missing.
3. Fixture confirms `.p8pre-runs/` is the canonical scratch namespace: `tasks.md:18,25,26,69-73,78` + `specs/n8-mode-c-profile-recheck/spec.md:31,35,53,120` + `specs/p8precond-zero-identity-spike/spec.md:68,135` all anchor on `<scratch>/SHUD-OpenMP/.p8pre-runs/`. Both Step 1 (PR-A) AND Step 2 (PR-E) materialize subdirs under this root.
4. Failure scenario realistic: §2.4 rsyncs from `<scratch>/SHUD-OpenMP/.p8pre-runs/...` to local `/tmp/...`, but if any future hand-pull or worktree clone lands artifacts under repo root `.p8pre-runs/` (mirroring P1d/P1e local-Mac pattern visible in `ls`: `.p1d-pr-*-runs`, `.p1e-runs`), they leak into `git status` and risk `git add .` capture.
5. In-scope for PR #353: per fixture overall pattern of P1d/P1e siblings being added inline with their first PR, AND reviewer cites #340 carry-forward note explicitly placing this fix in #341 PR-A. PR #353 IS #341 PR-A run.

## Note

Trivial 1-line fix (`.p8pre-runs/` + optionally `.p8pre-pr-*-runs/` for forward-compat with future PR-B..PR-G). Merge-block per reviewer severity classification — sibling pattern is well-established and skipping it is a regression vs P1d/P1e discipline.
