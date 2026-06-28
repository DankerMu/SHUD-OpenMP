# Fixture Review — p8tune-spgmr-maxl (PR-0 entry gate)

- **Change**: p8tune-spgmr-maxl
- **Active issue**: #363 (PR-0 of 6)
- **Reviewed at**: feat/p8tune-pr-0-doc-correction off baseline/p8tune off main 95b9158
- **Subagent**: reviewer (a58e022df276bc78d)
- **openspec validate strict**: PASS

## Verdict: **pass** (all 8 checks)

| # | Check | Verdict |
|---|---|---|
| 1 | Fixture level correctness (expanded, not broad-expanded) | pass |
| 2 | Risk pack coverage (11 core + 7 domain marked) | pass |
| 3 | Selected-pack evidence map (each maps to tasks.md scenarios) | pass |
| 4 | Invariant Matrix deferral (to PR-C prep) | pass |
| 5 | Must-preserve concreteness (4 spec deltas have explicit input/output) | pass |
| 6 | PR-0 scope readiness (tasks §1.1-§1.8 cover acceptance) | pass |
| 7 | Cross-PR sequencing (matches D11) | pass |
| 8 | Project-profile alignment (SHUD-OpenMP domain packs/triggers/evidence) | pass |

## Out-of-scope informational notes (not blocking)

1. "17 requirements" count accurate (5+4+4+4 across 4 spec deltas).
2. Design D2 narrower reframing consistent with proposal.md.
3. PR-A spec named anti-pattern ("6/47 NOT future gate") strengthens reader traceability.
4. maxl-sweep-verdict decision-tree completeness scenario is rigorous precedent.
5. **Minor drift (non-blocking)**: proposal.md L21 says "3-way equivalence" while design D0+D8 G3 + spec spgmr-maxl-env-hook L51-61 + tasks §3.7 all say "4-way" (unset/empty/0/5). Author may align proposal.md L21 → "4-way" for editorial cleanliness; proposal text is non-normative once spec deltas land. Not in PR-0 acceptance criteria; deferred.

## Phase 0.5 closure

- D0 risk triage block authored by orchestrator in design.md (per Phase 0.5 step 6 allowed scope).
- Invariant Matrix explicitly deferred to PR-C prep per D0:48-50 rationale (PR-0/A/B/D/E do not modify CVODE source; only PR-C is high-intensity).
- Fixture is implementation-ready for PR-0.
