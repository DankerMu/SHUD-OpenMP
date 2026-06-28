# Spec-Compliance Review — PR #354 (p8pre PR-B aggregator)

Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed head SHA: 49a2d519f3d63a0b6e1cdd6e9f39e7bba138ffe8

## Summary
All 8 spec-compliance items PASS; aggregator faithfully implements every scenario in `n8-mode-c-profile-recheck/spec.md`, and live run output proves end-to-end gate behavior on the 18-cell mirror.

## Per-item trace

1. **REJECT typo keys (spec L38)** — PASS. `REJECT_KEYS=(nlcf nfevals hcur qcur hin)` at `tools/p8pre/aggregate_n8_profile.sh:65` is the exact 5-key set. Phase B (L128-149) iterates 18 cells, greps each REJECT key with start-of-line anchor `^${bad}=` (prevents substring FP), emits per-cell error `REJECT: typo key '$bad' found in <cell>/cvode_stats.txt`, exits 1 when hits>0.

2. **nst invariance (spec L86-91)** — PASS. Phase F (L362-376) covers both heihe + heihe_x4 across N∈{1,4,8}, asserts strict string equality. Live output L17-21 confirms invariance.

3. **nfe / nfeLS / nni / nsetups invariance (spec L94-104)** — PASS. `INVARIANCE_KEYS=(nst nfe nfeLS nni nsetups)` at L74 = the 5 spec-mandated keys exactly.

4. **Absolute baseline anchors (spec L90+L97)** — PASS. `BASELINE_NST[heihe]=6698 / heihe_x4=6575 / BASELINE_NFE[heihe]=6943 / heihe_x4=6741` at L78-82 match cited P1e sources. Phase G compares N=1 median (relying on N=1=N=4=N=8 from §F). Mismatch sets `block_exit=3` with Mode C regression flag (matches L32 contract).

5. **Branch a/b/c/d ROI tree (spec L75-80)** — PASS. Phase I (L441-480) uses awk float comparisons, evaluates precedence a→b→d→c:
   - a: `r_min ≥ 1.5` (precedes c when both apply).
   - b: `r_max < 1.5` (only reachable when a falsified; mutually exclusive with a).
   - d: `r_min < 1.5 AND r_max ≥ 3.0` (precedes c per task brief §7 + spec L79 explicit AND).
   - c: residual `r_min < 1.5 ≤ r_max < 3.0`.
   Coverage exhaustive + mutually exclusive over (r_min, r_max | r_min ≤ r_max). Live run produced `branch: a` (r_min=1.819 ≥ 1.5).

6. **Explicit branch letter emit (spec L80)** — PASS. Stdout L548-550 (`branch: $branch`); verdict doc §3.5 L770 (`**branch: a (PROCEED — r_min=1.819 >= 1.5)**`). Confirmed in run output L28 + `docs/p8pre/n8_profile_verdict.md` §3.5.

7. **18 cells × 24 metrics scope** — PASS. `2 cases × 3 N × 3 reps = 18` (L50-52). 15 keys (L58) + 7 buckets (L69) + 2 extras (L70) = 24. Run log L6 confirms `parse OK: 18 cells × 24 metrics`. N=2 explicitly absent per D4.

8. **Out-of-scope verification** — PASS. `git diff baseline/p8pre...HEAD --stat`: only 2 new files (+1130). Zero touches to `canonical_15_keys.yaml`, master plan, case_deployment_map, SHUD submodule.

## Findings

None.

## Non-blocking notes

- Canonical 15-key set is inline-duplicated from `tools/cvode_stats_diff/canonical_15_keys.yaml` rather than parsed (L55-57 acknowledges drift hazard, defers to yaml unit test as drift gate). Acceptable per stated flat-dependency rationale.
- REJECT_KEYS covers the 5 spec-mandated keys. The canonical yaml additionally lists `nFCall / nFCall1..5` (excluded_keys) and `hlast / qlast / t_rhs_*` (diagnostic_keys); spec L38 does not mandate these, so script is correctly scoped. Optional future hardening, not blocking.

## Verdict

APPROVE — all spec compliance items pass with live-run evidence; scope discipline clean; branch tree implementation matches spec L75-80 exhaustive + mutually exclusive contract.
