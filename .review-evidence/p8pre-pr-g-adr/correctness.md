Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 8cf527c
Summary: One Critical correctness defect: Step 1 ROI ratio is misattributed in capstone + p8pre_summary (claim r_min=1.811 — that is the Step 2 post-PREC_LEFT ratio; Step 1 source-of-truth records 1.819). Master plan §P8-precond.0 cites 1.819/1.811 correctly.

Findings:

- Critical: Step 1 ROI ratio attribution error in capstone + p8pre_summary
  - Locations: /Users/danker/Desktop/Hydro-SHUD/openMP/docs/p8pre/capstone.md L24, L165, L228, L290; /Users/danker/Desktop/Hydro-SHUD/openMP/docs/p8pre_summary.md L20, L45, L70
  - Source-of-truth `docs/p8pre/n8_profile_baseline.md` §5.4 Table 4 (Step 1 anchor): heihe N=8 nfeLS_median=12632 / nfe_median=6943 = **1.819**. §5.5 explicitly: "branch: a (PROCEED — r_min = 1.819 ≥ 1.5)".
  - Step 2 cvode_stats (verified at /tmp/p8pre_identity_spike/heihe_N8_rep1/cvode_stats.txt): nfeLS=12120 / nfe=6696 = **1.811** — this is the POST-PREC_LEFT ratio, NOT the Step 1 baseline.
  - capstone.md L167 has a confused self-reconciliation note (compares 11933/6943=1.719 vs 12120/6696=1.811 but never reconciles to the correct Step 1 anchor 12632/6943=1.819).
  - Master plan §P8-precond.0 (L2362, L2370) correctly cites 1.819 for Step 1 and 1.811 for Step 2 → internally consistent. Capstone + summary contradict it (checklist item 7 violation).
  - Required fix: in capstone.md and p8pre_summary.md change all "Step 1 r_min = 1.811" → "1.819" and clarify 1.811 is the Step 2 post-PREC_LEFT ratio.

Non-blocking notes:

- openspec validate --all --strict: 26/30 PASS; 4 FAIL on p1c-capstone / p1c-deterministic-reduction / p1d-capstone / p1d-numa-governance. None p8pre-related; appears pre-existing. Needs verification head SHA does not regress these vs baseline/p8pre.
- VERIFIED PASS: (2) Step 1 §5.1 capstone numbers verbatim match n8_profile_baseline.md §5.1 (wall + nst + nfe). (2) Step 2 §5.2 capstone numbers verbatim match identity_spike_verdict.md §3 Table 1 (nst=6599/6569, nfe=6696/6775, ncfn=6/47). (5) case_deployment_map §5.2 18-row table — 6 cases × 3 reps = 18 confirmed, all yaml paths use the correct prefix, JID 9531-9548 contiguous. (6) review-loop-log epic_summary line parses as JSON, total_prs=9, prs=[341..349] inclusive, verdict=NO-GO, final_SHUD_pin starts "5276167", total_rounds=1, workflow=subagent-workflow. (8) docs/adr/0002 + 0003 + SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c all exist. (9) `git diff baseline/p8pre...8cf527c -- SHUD` empty; submodule HEAD = 5276167 unchanged. (10) 6 files: 3 NEW docs + 2 EDIT docs + 1 EDIT master plan = matches stat.
- 6-gate verdict table PASS values identical across ADR §Context + capstone §5.3 + summary + master plan verdict table. ncfn=6/47, max_ulp ≈ 9×10¹⁵, 5,155/214,252 positions, SHUD pin 5276167 — all cross-doc consistent.
- ADR-0003 §References "GitHub PR sequence" L116-131 contains internally inconsistent PR labeling (L124 marks #344 as "PR-E" and #345 as "PR-F" while L127-129 say PR-E=#346 / PR-F=#347 / PR-G=#348). The "NB on numbering" at L131 partially explains, but the table itself is contradictory. Bookkeeping cleanup, not a data-accuracy blocker.
