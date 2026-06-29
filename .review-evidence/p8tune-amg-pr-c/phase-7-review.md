# PR-C #397 (PR #404) — Phase 7 Adversarial Review

**Date**: 2026-06-29
**Branch**: `feat/issue-397-p8tune-amg-pr-c` HEAD (CI green, SUCCESS×3 across all checks)
**Reviewer**: single adversarial pass (Phase 7-style)
**Verdict**: **APPROVE_WITH_FOLLOWUP** → 1 Low deferred → **MERGE-READY**

## Findings closure status

| Severity | ID | Description | Status |
|---|---|---|---|
| Low | L1 | `run_dir` provenance comment at `aggregate_verdict.txt:162` is absolute-path-dependent (CLI-flag drift breaks bit-for-bit idempotency of metadata line; KV semantics + `verdict_branch=` anchor unaffected) | DEFERRED to PR-D / follow-up |
| Info | I1 | Rule precedence question: small-case BLOCKED gate requires `h4["all_pass"] and h16["all_pass"]` precondition (`aggregate_amg_spike.sh:523`), so falls through to Rule 4 NO-GO-both when both large cases fail — correct per spec REQ-5 | NO ACTION (confirmation) |
| Praise | P1 | Marker-vs-class binary STRUCTURALLY enforced: regex anchored inside `CELL_SUMMARY_BEGIN/END` block (`aggregate_amg_spike.sh:233-294`) — physically impossible to read stdout MARKER lines | — |
| Praise | P2 | Axis 4 amendment disclosure: aggregator emits BOTH strict `NO-GO-both` (canonical anchor) AND amended `GO` (FYI); ADR §Decision preserves spec REQ-6 byte-identical contract + §Discussion + §Forward action steer PR-D toward operational reading with instrumentation work item baked in | — |

## Axis-by-axis verdict (per Phase 7 5-axis investigation)

- **Axis 1 (parser strictness)**: PASS — block-anchored regex + `_DETECTED` rejection + 5-enum whitelist + NA sentinel + `colpack_version=unknown` sentinel all working as specified
- **Axis 2 (5-axis arithmetic)**: PASS — independent recomputation confirms cell-12 axes match `aggregate_verdict.txt:134-152` byte-identically (setup margin 0.158822, apply 0.493987, mem 0.003078, cycle 1.333333, op 0.500000); threshold derivations `0.7×0.226579 = 0.1586053` etc. match
- **Axis 3 (4-branch auto-typing)**: PASS — top-down evaluation matches spec REQ-5; Rule 4 first clause "heihe_x4 fails ANY axis" correctly triggers NO-GO-both; small-case BLOCKED gate has correct precondition (requires large cases PASS); fallback BLOCKED preserves determinism
- **Axis 4 (ADR byte-identical)**: PASS — `verdict_branch=NO-GO-both` verbatim at `aggregate_verdict.txt:17` and `docs/adr/0007-amg-spike-decision.md:56` (inside fenced code block under §Decision); Status=Proposed (`adr:3`); §Discussion mentions Axis 4 amendment (`adr:131-147`); §Forward action handles both strict and amended scenarios
- **Axis 5 (uv + bash quality)**: PASS — only `uv run python` invocations (no bare python/python3/pip); `bash -n` clean; `shellcheck -S warning` zero warnings; `set -euo pipefail`; missing spgmr_baseline_walls.h is fatal; all `$VAR` quoted; no `eval`

## Spec REQ compliance

| Spec REQ | Status |
|---|---|
| REQ-4 marker-vs-class binary | PASS |
| REQ-4 NA + colpack=unknown sentinel acceptance | PASS |
| REQ-5 5-axis evaluation | PASS |
| REQ-5 4-branch auto-typing | PASS |
| REQ-6 byte-identical ADR §Decision | PASS |

## Verification of strict NO-GO-both verdict

The strict `NO-GO-both` is spec-correct given uniform Axis 4 failure across all 4 cases (cycle_complexity ≈ 2.0 > 1.5 threshold). This is NOT a bug in BoomerAMG or in the spike binary — it is a known limitation of the current Axis 4 estimate (`2 × operator_complexity` hard-coded in `boomeramg_setup_solve.cpp`, disclosed in PR-A H3). With proper HYPRE telemetry instrumentation (PR-D / P8-tune.G follow-up per ADR-0007 §Forward action), Axis 4 becomes a real measurement.

The amended verdict `GO` (Axis 4 treated as non-discriminating) reflects the operationally meaningful reading: all 4 cases PASS axes 1/2/3/5 with substantial margins. Wall + memory axes are dispositive.

## Pre-merge gate

- **4-件套**: present at `.review-evidence/p8tune-amg-pr-c/`:
  - `aggregate.tsv` (16 cells raw data)
  - `aggregate_verdict.txt` (verdict_branch + 4 CASE_VERDICT blocks)
  - `SPEC_STATUS_HEADER.md` (PR-D consumption summary)
  - `phase-7-review.md` (this file)
- **Self-audit**: 1 Low surfaced; deferred (cosmetic, doesn't affect spec compliance)
- **Oracle integrity**: aggregator independently recomputed; cell-12 5-axis values byte-identical to aggregate_verdict.txt; threshold derivations confirmed

## Verdict

**MERGE-READY**. L1 deferred to PR-D / follow-up. No blockers.

## PR-D inheritance contract

Per reviewer L1 + recommendations:
1. **L1 cleanup**: normalize `run_dir` (+ `spgmr_baseline_walls.h`, `cn_node_ram.h`) provenance comments at `aggregate_amg_spike.sh:731-733` to REPO_ROOT-relative paths for full bit-for-bit reproducibility
2. **SPEC_STATUS_HEADER.md backfill**: replace `PR-D #<TBD>` placeholder with actual PR number after PR-D opens
3. **Strict-vs-amended verdict reconciliation**: carry into canonical OpenSpec archive header
4. **P8-tune.G PR-0 instrumentation reminder**: per ADR-0007 §Forward action, integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount` telemetry; if measured `cycle_complexity` drifts > 5% from hard-coded `2 × operator_complexity` estimate, ADR-0007 must be re-opened
