# Phase 0.5 fixture review evidence — p8tune-doc-correction

## Verdict: APPROVE-with-observations (3 non-blocking findings)

## Reviewer subagent ID: a6f8695a2bc5ad68a

## Per-check verdict matrix

| # | Check | Verdict |
|---|---|---|
| 1 | Internal consistency (proposal ↔ tasks ↔ spec) | PASS |
| 2 | NumY arithmetic plausibility (formula + per-case + working-set + cache-fit) | PASS — verified at `SHUD/src/ModelData/Model_Data.cpp:86` |
| 3 | Out-of-scope claim integrity | PASS |
| 4 | Spec delta validity (SHALL/MUST + Scenario blocks + openspec validate strict) | PASS |
| 5 | Caveat wording sufficiency | FINDING 1 (observation) |
| 6 | Master plan §P8-tune.D anchor scope match | PASS — matches Q1-Q9 grill |
| 7 | Reference integrity (file:line spot-check) | PASS |
| 8 | Risk pack selection | FINDING 3 (observation) |

## Findings

### Finding 1 (adopted into tasks.md §2.6) — ADR §Decision per-case table missing tier annotation
- File: `docs/adr/0004-maxl-sweep-decision.md:47-52` (highest-traffic spot in ADR)
- Resolution: tasks.md §2.6 added (annotate heihe N=1 row with `Performance opt-in (NOT A5-certified)`)

### Finding 2 (adopted into tasks.md §2.2) — Per-case NumY estimates need runtime-print backstop footnote
- File: ADR-0004 §Discussion per-case NumY table (to be added per §2.2)
- Resolution: tasks.md §2.2 extended with footer requirement: "NumRiv / NumLake estimates pending server runtime-print; replace with exact values in a future PR if precision needed for P8-tune.D fill-ratio formulae"

### Finding 3 (deferred to Phase 4 brief) — Risk pack could add `review-data-fidelity` lite for NumY table
- File: design.md D0 (L20)
- Resolution: NOT applied to D0; instead Phase 4 cross-review brief will explicitly include "Reviewer should verify the per-case NumY arithmetic against `SHUD/src/ModelData/Model_Data.cpp:86` formula + plausible NumRiv/NumLake estimates from heihe AutoSHUD config" as a focus area for `review-correctness`

## Pre-implementation status

OpenSpec change `p8tune-doc-correction` — Phase 0.5 fixture review CLEARED. Ready for Phase 1 implementation (docs edits per tasks.md §2-§4).
