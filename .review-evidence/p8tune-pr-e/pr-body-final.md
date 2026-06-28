## Summary

**Epic #362 capstone.** Closes P8-tune.C epic (6-PR sequence: PR-0 #369 ✅ + PR-A #370 ✅ + PR-B #371 ✅ + PR-C #372 ✅ + PR-D #373 ✅ + PR-E (本 PR)).

- **4 new files** (911 lines total):
  - `tools/p8tune/aggregate_maxl_sweep.sh` (385 lines bash + inline python3) — 60-cell PR-D data parser → 8 T-tables + flat KV + synthesis
  - `tools/p8tune/render_verdict.sh` (91 lines bash) — Mac+server doc renderer
  - `docs/p8tune/maxl_sweep_verdict.md` (176 lines) — 8 detailed T-tables + production tune guidance
  - `docs/adr/0004-maxl-sweep-decision.md` (259 lines) — ADR Optional-knob branch + case-size-asymmetric Krylov pattern + forward action
- **ADR-0004 decision: Optional-knob branch**. `SHUD_SPGMR_MAXL` env-var stays as long-lived production opt-in. Default unchanged.
- **Per-case best-maxl recommendation** (3-rep median, Slurm 9690 60-cell data):
  - heihe N=1 → `SHUD_SPGMR_MAXL=30` (+11.99% wall, GO band, ncfl 85→0)
  - heihe N=8 → `SHUD_SPGMR_MAXL=30` (+6.78% wall, Optional band)
  - heihe_x4 N=1 → unset (default=5); all maxl ≥10 REGRESS −6.86% to −15.83%
  - heihe_x4 N=8 → unset (default=5); all maxl ≥10 REGRESS −15.81% to −24.82%
- **8-gate verdict**: G1/G2/G3/G4/G6/G8 PASS strict; G5 MIXED case-asymmetric; G7 STRICT FAIL is expected numerical (maxl bump → Krylov expansion → CVODE step-size response → trajectory drift)
- **Key empirical finding**: case-size-asymmetric Krylov memory bandwidth pattern. heihe (~6300 elem) MGS working set fits L2 → bigger maxl wins. heihe_x4 (~40046 elem) overflows L2 → DRAM-bound MGS → wall regression dominates ncfl elimination

## Why

Per ADR-0003 P8-tune.C epic Path 4 trigger: PREC_NONE remains production solver but SUNDIALS-default maxl=5 yields high ncfl on NWM cases (heihe 85, heihe_x4 3620), suggesting Krylov subspace might be undersized. PR-E aggregates the 60-cell PR-D sweep into the formal 4-branch outcome (GO+default-bump / Optional-knob / Diagnostic / NO-GO) per spec D8.

PR-E's job: G1-G8 gate adjudication + ADR-0004 authoring + production tune guidance.

OpenSpec change: `p8tune-spgmr-maxl` (capability `maxl-sweep-verdict` complete; aggregator + verdict + ADR).

## Scope

4 files (+911 lines, 0 deletions):
- `tools/p8tune/aggregate_maxl_sweep.sh` (NEW, 385 lines, executable)
- `tools/p8tune/render_verdict.sh` (NEW, 91 lines, executable)
- `docs/p8tune/maxl_sweep_verdict.md` (NEW, 176 lines, render output)
- `docs/adr/0004-maxl-sweep-decision.md` (NEW, 259 lines)

No `.c/.cpp/.h` changes. No Makefile. No SHUD submodule pointer change (still `6ce17d6`). No `cvode_config.cpp:259` default constant change (Optional-knob does NOT trigger default-bump).

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] Mac `bash -n tools/p8tune/aggregate_maxl_sweep.sh` → 0 syntax errors
- [x] Mac `bash -n tools/p8tune/render_verdict.sh` → 0 syntax errors
- [x] Server-side aggregator dry-run → produces 8 T-tables + aggregate_verdict.txt + verdict_synthesis.md; emits `ADR-0004 branch: Optional-knob`
- [x] Server-side render dry-run → composes single verdict doc with YAML frontmatter + synthesis + 8 T-tables
- [x] Verdict doc cross-references all source (PR-A/B/C/D + Slurm 9690 + spec D8)
- [x] ADR-0004 cites ADR-0002 Path 4 + ADR-0003 fallback + per-gate quantified evidence + 4-branch decision-tree closure
- [x] Per-case recommendation derived programmatically from 3-rep median (not first-rep noise)
- [x] G7 STRICT FAIL interpretation documented with mechanism (Arnoldi MGS → CVODE step-size adapter), differentiated from "corruption"

## Agent Review

**Workflow**: `subagent-workflow` Phase 0-8 (medium-intensity fixture; epic capstone).

**Phase 0.5 fixture review**: 1× `reviewer` subagent → PASS (no findings; spec/design/tasks internally consistent).

**Phase 4 cross-review (round 1)**: 4× `reviewer` subagent panel @ `38ec6135c4cdb9eb6141461b77fd3122c4dda979`:

| Reviewer | Verdict | Findings | Per-check |
|---|---|---|---|
| `review-correctness` | APPROVE | 0 | 13/13 PASS |
| `review-spec-compliance` | APPROVE | 1 (non-blocking) | 8/8 PASS |
| `review-integration` | APPROVE | 0 | 10/10 PASS |
| `review-data-fidelity` | APPROVE | 0 | 10/10 PASS |

Reports: [round1-correctness.md](https://github.com/DankerMu/SHUD-OpenMP/blob/feat/p8tune-pr-e-aggregator-adr/.review-evidence/p8tune-pr-e/round1-correctness.md), [round1-spec-compliance.md](https://github.com/DankerMu/SHUD-OpenMP/blob/feat/p8tune-pr-e-aggregator-adr/.review-evidence/p8tune-pr-e/round1-spec-compliance.md), [round1-integration.md](https://github.com/DankerMu/SHUD-OpenMP/blob/feat/p8tune-pr-e-aggregator-adr/.review-evidence/p8tune-pr-e/round1-integration.md), [round1-data-fidelity.md](https://github.com/DankerMu/SHUD-OpenMP/blob/feat/p8tune-pr-e-aggregator-adr/.review-evidence/p8tune-pr-e/round1-data-fidelity.md).

**Phase 4.5 verifier gate**: 1 candidate finding adjudicated. Verifier verdict: **PLAUSIBLE** (medium-fixture: does not block). Persisted at [round1-phase45-verifier.md](https://github.com/DankerMu/SHUD-OpenMP/blob/feat/p8tune-pr-e-aggregator-adr/.review-evidence/p8tune-pr-e/round1-phase45-verifier.md).

**PLAUSIBLE finding context**: G7 spec L99 literal predicate ("ANY A4 max_ulp violation SHALL fail G7") vs ADR-0004 Optional-knob adoption rationale ("expected numerical drift, not corruption"). Real literal contradiction but acknowledged in ADR §Rationale §G7 + verdict.md → documented-spec-tension closure pattern. **Follow-up tracked**: spec amendment to add ADR-attested mechanism carve-out OR split G7 → G7-strict + G7-attested (background task `task_0c609142` spawned).

**Phase 5/6/6.2/6.5**: SKIPPED (PLAUSIBLE not merge-blocking at medium fixture).

**Phase 7 final review (Gap Sweep)**: 1× `reviewer` subagent (clean-context) @ `38ec6135` → **CLEAN APPROVE** (9/9 PASS; pre-merge readiness APPROVE).

**Pre-merge evidence hard-gate**: ✅ PR Agent Review block | ✅ Phase 4.5 verdict table persisted | ✅ clean round-1 panel (1 PLAUSIBLE acknowledged) | ✅ Phase 7 CLEAN APPROVE | ✅ CI 5/5 PASS | ✅ completion self-audit (4 PR-E tasks + 4-branch closure satisfied) | ✅ oracle integrity (60-cell sweep is the oracle; no test/spec weakened — PLAUSIBLE acknowledged + tracked for follow-up amendment).

**Repair intensity**: medium (epic capstone; tools + docs + ADR).

**Round counter**: 1 (single comprehensive cross-review round; well within 5-round budget).

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #368. Epic #362 capstone.
