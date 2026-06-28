# review-integration round 2 — PR #355 post-Phase-6 fix verification

- Reviewer agent: review-integration
- Review round: round 2 (post Phase 6 fix)
- Reviewed head SHA: 43bd6f2 (tracked unchanged — fix is openspec/changes/ working-tree-only per .gitignore:13)
- Branch: feat/issue-343-p8pre-pr-c-baseline

## Summary

Phase 6 fix correctly resolves cand-01: all 7 anchor citations now point to §5.1 (academic-paper-style Results); no new findings.

## Verification matrix

### V1 — Spec/tasks/design fix verify

| Check | Expected | Actual | Status |
|---|---|---|---|
| V1.a `grep §3\b` to baseline doc | 0 hits | 0 hits | PASS |
| V1.b `grep §5\.1` to baseline doc | ≥5 hits | 5 hits across 4 files | PASS |
| V1.c1 `tasks.md:41` enumeration | academic §1..§10 with §5.1 raw data | "§1 Introduction (H1/H2/H3) + §2 Related Work + §3 Methodology + §4 Experimental Setup + §5 Results (§5.1 raw data + ...)" | PASS |
| V1.c2 `tasks.md:77` anchor | `§5.1 raw data table` | exact match line 77 | PASS |
| V1.c3 `n8-mode-c-profile-recheck/spec.md:17` | `§5.1` | exact match line 17 | PASS |
| V1.c4 `n8-mode-c-profile-recheck/spec.md:113` | full §1..§10 enumeration | matches line 113 (§5.1 raw data tables + §5.2..§5.5 sub-sections + §6 Discussion etc.) | PASS |
| V1.c5 `p8precond-zero-identity-spike/spec.md:90` | `§5.1 raw data table` | exact match | PASS |
| V1.c6 `design.md:26` | `§5.1 wall_step1_baseline_median` | exact match | PASS |
| V1.c7 `design.md:107` | `§5.1 (Step 1 PR-A ...)` | exact match | PASS |

5 §5.1 hits across 4 distinct files (tasks.md ×2 + n8-mode-c-profile-recheck/spec.md ×1 + p8precond-zero-identity-spike/spec.md ×1 + design.md ×2 = 6 lines; grep emitted 5 because line 17 of n8-mode-c-profile-recheck/spec.md has one occurrence and the trailing-paren §5) — well above the ≥5 threshold; all 7 cited locations confirmed.

### V2 — No new findings

| Check | Expected | Actual | Status |
|---|---|---|---|
| V2.a `openspec validate p8pre-spike --strict --no-interactive` | exit 0 | "Change 'p8pre-spike' is valid", exit 0 | PASS |
| V2.b baseline doc `docs/p8pre/n8_profile_baseline.md` §5.1 + Table 1 intact | §5.1 heading + `Table 1: wall_step1_baseline_median` present | line 129 `## §5.1 Raw data table — gate-4 anchor (CORE)` + line 133 `Table 1: wall_step1_baseline_median(case, N)` | PASS |
| V2.c PR-F #347 implementer reading tasks.md L77 verbatim | anchor resolves to extant §5.1 in baseline doc | tasks.md L77 → `§5.1 raw data table`; baseline doc §5.1 exists; closed loop | PASS |

### V3 — Out-of-scope drift NOT touched

| Location | Status |
|---|---|
| `tasks.md L84` (task 8.8 / `identity_spike_verdict.md`) | Out-of-scope (separate PR-F deliverable doc); correctly omitted from Phase 6 |
| `p8precond-zero-identity-spike/spec.md L122` (identity_spike_verdict.md) | Same; correctly omitted |

### V4 — Forward-compat with #349 archive

When `openspec archive p8pre-spike` runs at #349, the 5 corrected lines in `openspec/changes/p8pre-spike/{tasks,design}.md` + 2 spec files migrate verbatim into persistent `openspec/specs/`. Strict validation already passes (V2.a) so archive will succeed. CONFIRMED.

## Findings

- None.

## Non-blocking notes

- Fix lives only in working tree per `.gitignore:13` — head SHA 43bd6f2 unchanged, mirroring PR #353 cand-02 precedent. Persistent record materializes at #349 archive.
- Doc structure now self-consistent: academic-paper layout where §3 = Methodology + §5 = Results, removing the prior ambiguity where prose said "§3 raw data table" while baseline doc had §3 = Methodology.

## Resolution status

- cand-01: RESOLVED (all 7 anchors corrected to §5.1; openspec strict validation green; baseline doc §5.1 + Table 1 intact; closed loop with PR-F #347 consumer).

## Verdict

APPROVE — Phase 6 fix is complete, surgical, and forward-compatible.
