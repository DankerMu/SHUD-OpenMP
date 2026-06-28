Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 43bd6f2a5c76eda2b9bbc577a6121020bae12b71
Summary: All 12 correctness checks PASS; wall medians, ROI ratios, anchors, branch verdict, and ADR-0003 omission verified. No blocking findings.

Findings:
- None.

Non-blocking notes:
- baseline.md front-matter `date: 2026-06-27` while today is 2026-06-26; consistent with §4 submit window "2026-06-27 01:32:52 → 02:43:35 UTC" so the run completed in early UTC of 06-27 — acceptable but worth a one-line note in §4 to disambiguate timezone if reviewers in CST notice.
- §5.1 nested table schema (Case, N, rep, SHUD pin, wall, yaml) differs in column names from parent §5 (Case, Platform, SHUD pin, wall, yaml, 状态). Column count matches (6); semantic divergence is justified (per-rep raw data vs per-case canonical). Nesting under §5 is legitimate; suggest §5.1 sentence "Wall = profile_B0.yaml `extras.t_wall_total`" makes this explicit which it does.

Evidence:
- Wall medians (computed from /tmp/p8pre_n8_profile/<cell>/profile_B0.yaml extras.t_wall_total, sorted-middle of 3 reps):
  - heihe: N=1→140.797, N=4→95.734, N=8→89.732
  - heihe_x4: N=1→1412.895, N=4→849.704, N=8→743.552
  - All match §5.1 Table 1 + §5.1 per-(case,N) anchor bullets bit-for-bit (3 decimal places).
- Cross-N invariance (§5.2): 10/10 PASS verbatim from PR-B verdict.md §3.2.
- Absolute anchor (§5.3): heihe.nst=6698, heihe.nfe=6943, heihe_x4.nst=6575, heihe_x4.nfe=6741 — match step1_prep.md §3-§4 (P1e PR-I anchors).
- ROI ratios (§5.4): r_min=1.819 (heihe), r_max=4.526 (heihe_x4) — match PR-B §3.4 verbatim.
- Branch verdict (§5.5): branch a (PROCEED) — matches PR-B §3.5 header.
- ADR-0003: /docs/adr/ contains only 0001-soa-hot-fields.md + 0002-solver-path.md; NO 0003-*.md drafted — correct per branch-a logic.
- Hypothesis verification (§6.1 Table 5): H1 rooted in 10/10 + 4/4 (§5.2 + §5.3); H2 rooted in r_min=1.819 (§5.4); H3 rooted in NumEle 6335 → 40046 (6.3×) + wall 89.7 → 743.5 (8.3×). Each hypothesis traces to data.
- case_deployment_map §5.1 schema: 18-row table with consistent (Case, N, rep, SHUD pin, wall, yaml) header; per-rep wall values bit-match per-cell yaml extras.t_wall_total.
- Master plan §P8-precond.0 inserted at L2356 BEFORE §P8-precond.1 (L2368) — order correct; cross-refs to PR-A/B/C and openspec/changes/p8pre-spike resolve.
- `git diff baseline/p8pre...HEAD SHUD` = empty (SHUD pin unchanged).
- `git diff baseline/p8pre...HEAD -- openspec/changes/` = empty (orchestrator zone respected).
- Internal consistency: §6.3 wall scaling ratios 140.797/89.732 = 1.569 (matches) + 1412.895/743.552 = 1.901 (matches). §6.3 P1e comparison heihe 1.066× / heihe_x4 1.729× matches step1_prep.md §2 (504/473, 1340/775).

Verdict: APPROVE — Step 1 capstone is numerically faithful to PR-A/B inputs, anchors trace cleanly to P1e PR-I, gate-4 baseline is well-defined, and branch-a NO-GO-ADR-omission is correct per the spec's 4-branch tree.
