Reviewer agent: review-documentation
Review round: round 1
Reviewed head SHA: 8cf527c
Summary: All four new docs structurally conform to checklist; NO-GO verdict consistent across ADR-0003 / capstone / summary / master plan; one numeric inconsistency on r_min (1.811 vs 1.819) and PR-A issue # ambiguity between Step 0 vs Step 1 nomenclature warrant clarification.

Findings:

- 🟡 Warning: `r_min` value drift between Step 1 verdict source and PR-G doc set
  `docs/adr/0003-precond-spike-decision.md:21` reports heihe N=8 `nfeLS/nfe = 12120 / 6696 = 1.811`; `docs/p8pre_summary.md:20` + `docs/p8pre/capstone.md:24,165` cite `r_min = 1.811`. But the upstream Step 1 verdict `docs/p8pre/n8_profile_verdict.md:5,135,138,140,166` and master plan §P8-precond.0 L2362+L2370 both quote `r_min = 1.819` (heihe `nfeLS=12632 / nfe=6943`). Capstone §5.1 NB para (L167) acknowledges the gap as "Step 1 SHUD pin 7a1dc8f post-fix anchor format 1.719" vs "Step 2 SHUD pin 5276167 post-PREC_LEFT 1.811" but does not reconcile vs `1.819` Step 1 PR-B aggregator output — the figure 1.819 (PR-B verdict) ≠ 1.719 (capstone §5.1 Table 1: 11933/6943) ≠ 1.811 (capstone §5.2 Step 2 heihe N=8: 12120/6696). Fix: add one footnote to ADR-0003 Context bullet + capstone §5.1 explicitly mapping the three ratios to their source builds (Step 1 PR-B verdict counter 12632 vs Table 1 counter 11933 vs Step 2 cvode_stats counter 12120) — currently the reader cannot tell whether 11933 in Table 1 is a typo or a legitimate counter divergence.

- 🟡 Warning: PR-A issue # double-binding inconsistent between ADR vs summary/capstone
  `docs/adr/0003-precond-spike-decision.md:120` says "PR-A (Step 1 prep): #340 / merge #352"; `docs/p8pre_summary.md:20` opens "Step 1 由 PR-A (#341, intake) + PR-A run (#341, …)" and `docs/p8pre/capstone.md:80,92` writes "Step 1 PR-A 18-cell 矩阵 (issue #341)". ADR's "PR-Step0 #339 / PR-A #340 / PR-B #341 / PR-C #342 / PR-D #343" mapping (L119-L123) contradicts summary+capstone's "PR-A #341 / PR-B #342 / PR-C #343" mapping (used 5+ times). The ADR L131 numbering-NB para also self-contradicts: it claims "Step 1 = #341/#342/#343 + Step 2 = #344/#346/#347/#348" — yet L120 above lists PR-A=#340. Fix: pick one binding (the summary/capstone #341/#342/#343/#344/#346/#347/#348 mapping appears canonical, matching review-loop-log `prs:[341..349]`) and align ADR L119-L130 + L131 NB para to it. Right now a reader cannot trace PR-A → GitHub issue.

- 🔵 Suggestion: Master plan §P8-precond.0 retains `r_min=1.819` in its PR-B summary line (L2362, L2370) while PR-G's new doc set canonicalizes 1.811 — recommend a one-line bridge sentence in §P8-precond.0 explaining that the Step 1 verdict aggregator-reported 1.819 vs the Step 2 cvode_stats representative 1.811 are the same epic, just different SHUD-pin builds; otherwise §P8-precond.0 reads as if there are two different ROI numbers floating around.

- 🟢 Praise: Academic-paper structure conformance is strong
  `docs/p8pre/capstone.md` cleanly delivers all 10 sections required by the P1e mother template — YAML frontmatter (L1-L20) + Abstract (L22-L27) + §1 Introduction with formal H1/H2/H3 hypotheses tied to operational gate definitions (L41-L43) + §2 Related Work + §3 Methodology + §4 Experimental Setup + §5 Results (3 subsections) + §6 Discussion (6 subsections) + §7 Limitations + §8 Conclusion + §9 Future Work + §10 References (with 4 reference subsections). 395 lines hits the 350-500 target.

- 🟢 Praise: ADR-0003 fully conforms to 0002-solver-path.md template
  Status / Date / Deciders / Owner / Tags / Supersedes / Superseded-by / Related (L3-L10) + Context (L14-L46) + Decision (L52-L60) + Consequences (Positive/Negative/Neutral L66-L82) + Forward action recommendations (L86-L110) + References (Epic+PRs / OpenSpec / Internal docs / SUNDIALS canonical / Previous ADR / Master plan / Mother template L116-L173). 173 lines fits the 100-200 envelope.

- 🟢 Praise: review-loop-log epic_summary JSON entry validates cleanly
  `tail -1 docs/review-loop-log.jsonl | python3 -m json.tool` succeeds; `kind=epic_summary`, `total_prs=9`, `prs=[341..349]`, `verdict=NO-GO`, `workflow=subagent-workflow` all match contract; includes useful extras `forward_to_future[]` + `verdict_driver` + `fixture_levels[]`.

- 🟢 Praise: case_deployment_map §5.2 18-row table internally consistent
  All 18 rows use SHUD pin `5276167`, JID range 9531-9548 noted in `Server source mirror` para (L164); yaml paths all rooted at `/tmp/p8pre_identity_spike/<case>_N<n>_rep<r>/profile_B0.yaml`; NO-GO archive carve-out explicit in §5.2 status para (L141).

- 🟢 Praise: Cross-doc NO-GO verdict consistency
  ADR-0003 Decision (L52) = capstone §8 Conclusion (L284-L288) = p8pre_summary §决策 (L37-L48) = master plan §P8-precond.0 Outcome (L2394) — all read "NO-GO option (b)" with same Rationale TL;DR (H2 deterministic FAIL + S5 PREC_LEFT inherent cost + KISS/YAGNI dead-codepath). All four cite PR-F #359 verdict adjudicator as primary input.

Non-blocking notes:
- Markdown integrity: 0 emoji in capstone (per project rule); tables render cleanly (verified §5.1, §5.2, Table 2/3 in capstone; H1-S6 table in ADR + master plan all 6-row aligned).
- Master plan §P8-precond.0 edit preserves original §P8-precond.0 prep title L2356 + appends p8pre-spike evidence below; original `P8-precond.1` to `.7` task numbering (L2396+) intact, so spec authors can still cross-reference.
- `openspec/changes/` diff is empty per `git diff origin/main...HEAD -- 'openspec/changes/**'` — orchestrator-zone untouched, correct.
- p8pre_summary.md (113 lines) is engineer-style without YAML — falls within CLAUDE.md "顶层简版 engineer-style 是 acceptable convention" carve-out alongside the academic capstone.
- ADR L75 mentions "p8pre-spike epic 7 PR (PR-A through PR-G)" — but the actual epic spans 8 letters (Step 0 + A/B/C/D/E/F/G + cleanup #349 = 9 PRs per review-loop-log). Minor count drift — not blocking but worth aligning when fixing the PR-A binding issue above.
