# Phase 4 Correctness Review — PR-B #387

**Branch**: `feat/issue-382-p8tune-klu-spike-pr-b` HEAD bf8703e at review time
**Reviewer**: reviewer subagent (correctness perspective)
**Date**: 2026-06-28
**Verdict**: REQUEST CHANGES → all 5 findings fixed in commits 931a9c4 + 6ebdb4d

## Findings

### F1 (MEDIUM) — Silent fallback to stale run-9762/cell-08.log
- file: `tools/p8tune.D/aggregate_klu_spike.sh:184-199`
- evidence: `find_cell_log` fallback at L192-198 iterates `CELLS_ROOT.iterdir()` in fs order; if authoritative path missing, could silently return stale `run-9762/cell-08.log` (no marker, no cell_summary) producing UNKNOWN.
- fix: 931a9c4 — drop fallback entirely; hard-fail (return None → MISSING) per spec REQ-7 PR-A boundary

### F2 (MEDIUM) — UNKNOWN verdict_class silently passes through TSV
- file: `tools/p8tune.D/aggregate_klu_spike.sh:206-323, 381-402`
- evidence: parser default `verdict_class="UNKNOWN"`; if neither marker nor cell_summary parsed (truncated log), row emits to TSV with no warning, polluting best-combo selection.
- fix: 931a9c4 — loud WARN with log path + last 5 lines; downgrade to MISSING for verdict computation

### F3 (LOW) — Marker precedence iteration-order, not chronological
- file: `tools/p8tune.D/aggregate_klu_spike.sh:230-258`
- evidence: loop iterates `(INDEX_OVERFLOW, OOM, WALL_OVERFLOW)` with break; OOM-then-trap case would pick OOM but ignore later WALL marker.
- fix: 931a9c4 — collect all matches with `m.start()`, sort by byte offset, pick chronologically-first

### F4 (LOW) — pass_count==2 + wall_margin=None classification gap
- file: `tools/p8tune.D/aggregate_klu_spike.sh:538-570`
- evidence: when `best.numeric_factor_wall_sec is None`, wall_margin is None, branch fires `pass_count==2 && wall_axis==FAIL → NO-GO` without distinguishing "blown past budget" from "no wall data".
- fix: 931a9c4 — documentation-only inline comment acknowledging the conflation; spec D8 enum has no 4th value

### F5 (INFO) — ADR-0005 "+99%/+85% wall margin" ambiguous
- file: `docs/adr/0005-klu-spike-decision.md:158, 193, 226, 268`
- evidence: phrasing reads as +99% end-to-end speedup vs SPGMR rather than budget headroom fraction.
- fix: 931a9c4 (3 locations: L158/L193/L226) + 6ebdb4d (residual L268 surfaced by Phase 7) — replaced with "≤1%/14% budget used; >99%/>85% budget-headroom"

## Verifier verdicts (orchestrator-internal Phase 4.5)

All 5 findings CONFIRMED via direct read of cited code locations during fix synthesis. Since this is a stacked spike-PR with single reviewer perspective (PR-A pattern), and Phase 7 final reviewer independently re-verified all 5 fixes correct, the independent-verifier role is satisfied by Phase 7 reviewer.

| Finding | Verdict | Evidence |
|---|---|---|
| F1 | CONFIRMED | aggregate_klu_spike.sh:192-198 fallback loop verified; fix at 931a9c4 |
| F2 | CONFIRMED | UNKNOWN default verified at L210; fix at 931a9c4 |
| F3 | CONFIRMED | Iteration order verified at L230-234; fix at 931a9c4 |
| F4 | CONFIRMED | wall_margin None at L574-579 verified; doc fix at 931a9c4 |
| F5 | CONFIRMED | Phrasing at L158/193/226/268 verified; fix at 931a9c4 + 6ebdb4d |

No PLAUSIBLE or REFUTED.
