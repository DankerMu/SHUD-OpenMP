## Summary

- Corrects `ncfn < 6 ∧ ncfn < 47` future-gate citations across 3 merged p8pre docs (ADR-0003 / identity_spike_verdict / capstone) and adds a forward note to `docs/p8pre_summary.md`. The values 6/47 are PREC_LEFT+identity Step 2 spike floors; actual PREC_NONE production floors per `docs/p8pre/n8_profile_verdict.md` §3.1 are 7 (heihe) / 51 (heihe_x4). Corrected gate: `ncfn_candidate ≤ 7 ∧ ncfn_candidate ≤ 51`; 6/47 retained only as negative-control anchor.
- Fixes 3 `nfeLS = 30509` typos (ADR-0003 L22 + `openspec/glossary.md` L271 had `30518`; `capstone.md` §5.1 L161 had `30517`).
- Updates cleanup status across docs from "deferred" to "completed at outer `e442ce8` / SHUD `37be0fe` (2026-06-27)".
- Adds 2 glossary terms: `mode-C-tune` (per design D9 full semantics) and `SHUD_SPGMR_MAXL` (env-var values + 4 safety constraints).
- Adds `SHUD_openMP_master_plan.md` §P8-tune.C section: epic scope (4 capabilities × 6 PRs) + entry condition (hard-evidence already satisfied per §3.1) + 6-PR sequence + 8-gate verdict (G1-G8) + ADR-0004 6-branch verdict table.

## Why

p8pre-spike epic #338 closed NO-GO per [ADR-0003](docs/adr/0003-precond-spike-decision.md) at outer `e442ce8` / SHUD `37be0fe`. The closing docs cite `ncfn < 6 / 47` as the P8-tune future-candidate gate — those numbers come from the `PREC_LEFT + identity` Step 2 spike (which won't be used in production since PREC_LEFT was rejected for structural drift), not the cleaned `PREC_NONE` production baseline. Future-candidate gate must reference production floors `ncfn = 7 / 51` per `n8_profile_verdict.md` §3.1.

This PR is part of the [p8tune-spgmr-maxl](https://github.com/DankerMu/SHUD-OpenMP/issues/362) epic (6 PRs sequenced PR-0 → PR-A → PR-B → PR-C → PR-D → PR-E + conditional PR-F). PR-0 is the doc-state-correction foundation for the rest of the epic.

OpenSpec change: `p8tune-spgmr-maxl` (capability `p8pre-doc-state-correction`).

## Scope

Doc-only. 6 files, 108 insertions(+), 36 deletions(-):

- `SHUD_openMP_master_plan.md` (new §P8-tune.C section)
- `docs/adr/0003-precond-spike-decision.md` (future-gate + nfeLS typo + cleanup status + cross-ref)
- `docs/p8pre/capstone.md` (future-gate + nfeLS typo + cleanup status + forward link to ADR-0004)
- `docs/p8pre/identity_spike_verdict.md` (PASS-criterion + cleanup status + forward links)
- `docs/p8pre_summary.md` (additive §Forward note; grep-verified no in-text gate citation needed correction)
- `openspec/glossary.md` (nfeLS typo + 2 new terms: `mode-C-tune` + `SHUD_SPGMR_MAXL`)

No `.c`/`.cpp`/`.h`/`Makefile`/`.sh`/`.py`/`.yaml` changes. No SHUD submodule pointer change (still `37be0fe`).

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS (`Change 'p8tune-spgmr-maxl' is valid`)
- [x] `grep -nE 'ncfn ?< ?6|ncfn ?< ?47' docs/adr/0003-precond-spike-decision.md docs/p8pre/identity_spike_verdict.md docs/p8pre/capstone.md docs/p8pre_summary.md` → 0 matches
- [x] `grep -nE '30518|30517' docs/adr/0003-precond-spike-decision.md docs/p8pre/capstone.md openspec/glossary.md` → 0 matches
- [x] `grep -cE 'mode-C-tune|SHUD_SPGMR_MAXL' openspec/glossary.md` → 5 matches (≥ 2 expected)
- [x] `grep -c 'P8-tune.C' SHUD_openMP_master_plan.md` → 4 matches (≥ 1 expected)
- [x] `git diff --name-only` → 6 doc files only; no source/test/SHUD pointer changes
- [x] Phase 0.5 fixture review (orchestrator-spawned reviewer subagent) → PASS on all 8 checks
- [x] Phase 2 orchestrator scope audit → PASS
- [x] Phase 4 cross-review round 1: review-correctness + review-spec-compliance parallel → both CLEAN
- [x] Phase 4.5 verifier gate → empty verdict (0 candidates)
- [x] Phase 7 final review (Gap Sweep) → CLEAN, APPROVE merge gate
- [x] CI: 5/5 PASS (setup / build-and-compare / asan-ubsan keliya / asan-ubsan qhh / tools-tests)

## Agent Review

- **Reviewer agents used**: `reviewer` (Phase 0.5 fixture review), `reviewer` (Phase 4 Round 1: review-correctness), `reviewer` (Phase 4 Round 1: review-spec-compliance), `reviewer` (Phase 7 Gap Sweep independent-final)
- **Reviewed head SHA**: `e3aa6dbc977b2fad3fef49a6a099c74fe029b872`
- **Review evidence**: posted as consolidated bundle comment below
- **OpenSpec change**: `p8tune-spgmr-maxl`; fixture level: expanded (compact-doc-only for PR-0); selected risk packs: Documentation / migration notes
- **Phase 4.5 verifier verdict**: empty (Round 1 produced 0 actionable findings; nothing to verify); persisted at `.review-evidence/p8tune-pr-0/round1-phase45-verifier.md` (local)
- **Key findings addressed**: None — all 3 review passes CLEAN; 1 non-blocking note (`capstone.md` §5.1 ratio `4.527` vs authoritative `4.526`) deferred to PR-A `clean-prec-none-baseline` per cross-PR scope boundary

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #363.
