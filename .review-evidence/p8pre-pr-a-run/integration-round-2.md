Reviewer agent: review-integration
Review round: round 2 (post Phase 6 fix)
Reviewed head SHA: 83a8864

Summary: Both round-1 Warnings RESOLVED; fixes are minimal, additive, and introduce no new integration risks.

Findings:
- None.

Non-blocking notes:
- openspec CLI not on PATH locally, so `openspec validate p8pre-spike --strict` could not be executed; treated as "needs verification at #349 archive" — text inspection of `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` shows well-formed `## ADDED Requirements` + `#### Scenario:` structure unchanged, only filename token substituted.
- `openspec/changes/` is project-gitignored (`.gitignore:13` "OpenSpec transient changes"); cand-02 fix lives only in local working tree on this branch — by design, will carry forward to `openspec/specs/` at #349 archive. Confirmed `openspec/specs/` currently empty (no stale `profile.yaml` to recall).
- `docs/` grep for `profile.yaml` returned no p8pre-spike-scoped hits, so source-of-truth surface area is clean.

Resolution status:
- cand-01: RESOLVED
  - `.gitignore` L22–24 adds block: comment `# p8pre spike (#341) runtime + per-PR slice` + patterns `.p8pre-runs/` and `.p8pre-pr-*-runs/`, placed adjacent to existing `.p1e-runs/` sibling (L18–20) — pattern parity preserved.
  - `git check-ignore .p8pre-runs/x .p8pre-pr-a-runs/y .p8pre-pr-b-runs/z .p1e-runs/foo` exits 0, all 4 paths matched; p1e entry intact (no regression).
  - Diff vs d8602d0 is +4 lines only, no deletions.
- cand-02: RESOLVED
  - `grep -rn "profile\.yaml" openspec/changes/p8pre-spike/` returns 0 hits.
  - `grep -rn "profile_B0\.yaml" openspec/changes/p8pre-spike/` returns 11 hits across 4 files: `proposal.md` (1), `tasks.md` (5, incl. §3.1), `specs/n8-mode-c-profile-recheck/spec.md` (4), `design.md` (1).
  - Spot-checked `specs/n8-mode-c-profile-recheck/spec.md`: L17 (Scenario "Profile emit names match canonical") and L53 (Scenario "Mode A vs Mode C tags differ only by suffix") both use `profile_B0.yaml` — matches actual SHUD emit observed in PR-A run.
  - `tasks.md` §3.1 references `profile_B0.yaml` consistent with reality.
  - No spec `## ADDED Requirements` heading removed, no `#### Scenario:` deleted, no AC weakened — substitution is lexical only.
