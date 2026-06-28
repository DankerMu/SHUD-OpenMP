## Summary

- Adds `§verdict` section to `docs/p8tune/clean_prec_none_baseline.md` (between `§mode-C-tune-reference` and `§References`).
- Applies spec `maxl-sweep-verdict` Requirement "Sweep entry condition" scenario "Full sweep GO triggered by hard evidence" L11-15.
- **Decision**: FULL 60-cell sweep GO. Both heihe (`ncfl=85`) and heihe_x4 (`ncfl=3620`) satisfy `ncfl > 0` predicate per Step 1 PR-B verdict `n8_profile_verdict.md` §3.1.
- Decision-tree closure verified: input state matches "Full sweep GO" branch unambiguously; probe-only fallback + 2 NO-GO branches NOT exercised.
- Downstream PR-D contract: 60-cell matrix (`maxl ∈ {5, 10, 15, 20, 30}` × `case ∈ {heihe, heihe_x4}` × `N ∈ {1, 8}` × `rep ∈ {1, 2, 3}`), Slurm 三铁律 paths, PR-C env-var hook dependency.

## Why

PR-A baseline doc established the `§decision-input-table` (raw data: both cases `ncfl > 0`, heihe_x4 `nli/nni ≈ 4.527` saturation). PR-B applies the decision tree from spec `maxl-sweep-verdict` to that data and emits the explicit "full sweep GO" verdict — closing the sweep-mode adjudication before PR-D runs the actual 60-cell sweep.

This is doc-only verdict adjudication; no source / tool / server compute change. PR-D inherits this verdict to drive the sweep matrix; PR-E ADR-0004 will adjudicate the GO/NO-GO/Optional-knob/Diagnostic outcome separately on PR-D's sweep results.

OpenSpec change: `p8tune-spgmr-maxl` (capability `maxl-sweep-verdict`).

## Scope

1 file, +48 insertions, 0 deletions:
- `docs/p8tune/clean_prec_none_baseline.md` (new `§verdict` section)

No `.c`/`.cpp`/`.h`/`Makefile`/`.sh`/`.py` changes. No SHUD submodule pointer change (still `37be0fe`). No tool change. No new doc file. No PR-A content modification.

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] `git diff --name-only` → 1 doc file (`docs/p8tune/clean_prec_none_baseline.md`); no source/test/SHUD pointer
- [x] §verdict decision-input table cross-references PR-A `§decision-input-table` (both rows `ncfl > 0`)
- [x] §verdict branch adjudication enumerates all 4 spec decision-tree scenarios (Full sweep GO + probe-only + NO-GO no-entry + residual fallback) with explicit MATCH / NOT MATCH per current baseline state
- [x] §verdict downstream PR-D contract cites Slurm 三铁律 + 60-cell matrix dimensions + PR-C dependency
- [x] §verdict §Cross-ref explicitly disclaims ADR-0004 outcome adjudication (entry-condition input only, deferred to PR-E)
- [x] Phase 4 round 1: review-correctness + review-spec-compliance parallel @ `db82450` → both CLEAN, 0 findings, 5 non-blocking informational notes
- [x] Phase 4.5 verifier gate → empty verdict (0 candidates)
- [x] Phase 7 final review (Gap Sweep) @ `db82450` → CLEAN, APPROVE merge gate
- [x] CI 5/5 PASS @ `db82450` (setup 8s / build-and-compare keliya 59s / asan-ubsan keliya 34s / asan-ubsan qhh 5s / tools-tests 11s)
- [x] No SHA drift: single-SHA clean-round flow

## Agent Review

- **Reviewer agents used**: `reviewer` (Round 1 Phase 4 parallel: review-correctness + review-spec-compliance @ `db82450`); `reviewer` (Phase 7 Gap Sweep independent-final @ `db82450`)
- **Reviewed head SHA**: `db8245064ad80d061ec41a68ecbcfa3b1ef1acd8`
- **Review evidence**: consolidated bundle comment posted below
- **OpenSpec change**: `p8tune-spgmr-maxl`; fixture level: expanded (compact-doc-only for PR-B); selected risk packs: Documentation + Decision-tree logic correctness + Cross-PR contract integrity (PR-D entry-mode handoff + PR-E ADR-0004 scope boundary)
- **Phase 4.5 verifier verdict**: empty (Round 1 2 reviewers reported 0 actionable findings; nothing to verify); persisted at `.review-evidence/p8tune-pr-b/round1-phase45-verifier.md`
- **Key findings addressed**: 0 actionable findings. 5 non-blocking notes from Round 1 — all informational (no fixes needed). Clean single-round flow with no SHA drift.

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #365.
