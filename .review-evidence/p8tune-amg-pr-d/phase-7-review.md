# PR-D #398 (PR #405) — Phase 7 Adversarial Review

**Date**: 2026-06-29
**Branch**: `feat/issue-398-p8tune-amg-pr-d` HEAD (post-M1+M2+L2 fixes)
**Reviewer**: single adversarial pass (Phase 7-style)
**Verdict**: **APPROVE_WITH_FOLLOWUP** → 4 findings closed inline → **MERGE-READY**

## Findings closure status

| Severity | ID | Description | Status |
|---|---|---|---|
| Medium | M1 | `merge_sha` typo `09e815d` (3 places: log L87 + academic L27 + L637) | **CLOSED** inline |
| Medium | M2 | PR-A/B/C placeholder merge_shas (`see-PR-402/3/4`) need real SHAs | **CLOSED** inline (`93e86a8` / `cfa944f` / `6a11039`) |
| Low | L1 | Stale "5-PR scope" + "ADR-0007 4-branch decision tree (forthcoming)" tables retained without superseded marker (master plan L2564-2585) | DEFERRED (cosmetic; both new + old content present; reader can navigate) |
| Low | L2 | Broken markdown link `(. review-evidence/...)` stray space (academic L632) | **CLOSED** inline |
| Praise | P1 | Strict-vs-amended verdict 2-form rendering transparently disclosed in all 3 anchors (ADR §Decision, spec Status header, master plan closure) | — |
| Praise | P2 | §P8-tune.G enabler-vs-verdict_branch-mapped distinction correctly disclosed in 3 places (master plan §F closure + §G header + spec Status header Forward actions) | — |

## Axis-by-axis verdict (per Phase 7 5-axis investigation)

- **Axis 1 (ADR Status flip)**: PASS — L3 reads "Accepted (2026-06-29 in PR-D capstone PR #<TBD>; flipped Proposed → Accepted per spec REQ-6 Scenario 'ADR-0007 Status lifecycle')"; verdict_branch=NO-GO-both byte-identical to aggregate_verdict.txt L17
- **Axis 2 (master plan close + G anchor)**: PASS — L2550 header `[CLOSED, 5-PR sequence MERGED]`; post-merge para L2552 cites 5-PR sequence; §P8-tune.G L2587 anchored `[OPEN, HIGH priority]` per ADR §Forward action; enabler distinction transparent
- **Axis 3 (OpenSpec archive)**: PASS — `openspec/specs/amg-pattern-spike-verdict/spec.md` exists; L3-7 5-line `>` blockquote Status header matches KLU template schema
- **Axis 4 (review-loop-log)**: PASS post-fix — 5 entries jq-valid; M1+M2 corrections preserve audit-trail (`git rev-parse 09a815d` + 3 PR-A/B/C SHAs all resolve)
- **Axis 5 (academic summary)**: PASS — 675 LOC; 10 § sections; YAML + H1/H2/H3 + Limitations + Future Work + References complete

## Spec REQ compliance (post-fix)

| Spec REQ | Status |
|---|---|
| REQ-6 ADR Status lifecycle | PASS |
| REQ-7 master plan anchor (NO-GO-both branch) | PASS |
| REQ-7 archive Status header schema | PASS |
| REQ-7 review-loop-log entries | PASS (post-fix) |
| REQ-8 academic summary | PASS |

## HARD GATE 5 conditions verification (for PR-E)

All 5 hold at PR-D HEAD post-fix:
1. PR-D would merge into baseline/p8tune-amg-spike ✓
2. master plan §P8-tune.F = [CLOSED] (L2550) ✓
3. ADR-0007 Status = Accepted (L3) ✓
4. `openspec/specs/amg-pattern-spike-verdict/spec.md` exists (git-tracked, 259 LOC) ✓
5. master plan §P8-tune.G conditional anchor present (L2587) ✓

## Pre-merge gate

- **4-件套** at `.review-evidence/p8tune-amg-pr-d/`:
  - phase-7-review.md (this file)
- **Self-audit**: 2 Medium + 1 Low closed inline; 1 Low deferred (cosmetic)
- **Oracle integrity**: `git rev-parse 09a815d 93e86a8 cfa944f 6a11039` all resolve to merged commits

## Verdict

**MERGE-READY** post-fix. PR-E (baseline → main capstone-merge) opens after PR-D merges.

## PR-E inheritance contract

After PR-D merges into baseline/p8tune-amg-spike:
1. Backfill PR-D entry in `docs/review-loop-log.jsonl` with actual `pr` # + `merge_sha` (orchestrator handles in PR-E or as a quick follow-up)
2. Open PR-E baseline/p8tune-amg-spike → main with **merge-commit strategy** (per CLAUDE.md PR-389 capstone pattern; preserves 5-PR baseline history)
3. PR-E final review + merge
4. Close #393 epic + verify #394/395/396/397/398 all closed
5. Verify master plan §P8-tune.F [CLOSED] + §P8-tune.G [OPEN, HIGH] anchors present in main HEAD
