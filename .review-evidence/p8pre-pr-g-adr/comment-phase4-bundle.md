## Phase 4 Cross-Review Evidence Bundle (round 1, expanded fixture)

Reviewer agents: `review-spec-compliance`, `review-correctness`, `review-documentation`, `review-integration`
Review round: round 1
Reviewed head SHA: `8cf527c`
Local evidence: `.review-evidence/p8pre-pr-g-adr/{spec-compliance,correctness,documentation,integration}.md`

### review-spec-compliance — APPROVE

Summary: All 12 spec-compliance checklist items PASS. tasks.md §9.1-9.7 fully covered; ADR-0003 Decision = NO-GO option (b) per spec L74-79 + L106-108; D8 NO-GO path adhered with Forward-action recommendations correctly marking SHUD-source revert as deferred (per issue #348 "不动 SHUD 源码" Out-of-Scope); SHUD pointer 5276167 unchanged; openspec validate exit 0.

Findings: **None.**

Praise: ADR L86-96 Forward-action recommendations achieves clean separation of decision (binding now) vs execution (deferred to #349 or separate cleanup PR), preserving D8 prescription audit trail without violating issue #348 scope.

### review-correctness — REQUEST CHANGES → CONFIRMED

Summary: 1 Critical correctness defect: Step 1 ROI ratio mis-attribution in capstone + p8pre_summary (cited 1.811 — Step 2 post-PREC_LEFT — as Step 1 baseline).

Findings:

- **Critical (CONFIRMED)**: Step 1 ROI ratio attribution error
  - Locations: `docs/p8pre/capstone.md` L24/L92/L165/L228/L290; `docs/p8pre_summary.md` L20/L45/L70
  - Source-of-truth `docs/p8pre/n8_profile_baseline.md` §5.4 Table 4 + `docs/p8pre/n8_profile_verdict.md` §5 L5/L135/L140/L141/L166/L169-170: Step 1 PREC_NONE baseline `r_min = nfeLS_median/nfe_median |_{heihe, N=8} = 12632/6943 = 1.819`
  - Step 2 PREC_LEFT identity-precond cvode_stats at `/tmp/p8pre_identity_spike/heihe_N8_rep1/cvode_stats.txt`: `nfeLS=12120 / nfe=6696 = 1.811` — different condition (post-PREC_LEFT)
  - Master plan §P8-precond.0 L2362 + L2370 cited 1.819 / 1.811 correctly → internal contradiction with capstone + summary
  - **Required fix**: capstone + summary + ADR self-section all Step-1-context `1.811` → `1.819` with derivation annotation; Step-2-context `1.811` retained with explicit "Step 2 PREC_LEFT" attribution

### review-documentation — REQUEST CHANGES → CONFIRMED

Summary: All four new docs structurally conform; NO-GO verdict consistent. Two cross-doc inconsistencies surfaced.

Findings:

- **Warning (CONFIRMED, dup of review-correctness)**: r_min figure drift 1.811 vs 1.819 vs 1.719 across capstone/summary/master plan/verdict — same root cause as review-correctness Critical
- **Warning (CONFIRMED)**: ADR-0003 §References L120-131 PR-A binding inconsistency — ADR says PR-A=#340 but capstone L357-366 + summary L94-103 + review-loop-log epic_summary `prs:[341..349]` use canonical PR-A=#341

Praise: capstone.md hits all 10 sections (YAML + Abstract + §1-§10 + H1/H2/H3 hypotheses + Limitations + References subsections), 395 lines within 350-500 target. ADR-0003 fully conforms to 0002-solver-path.md template (Status/Date/Deciders/Owner/Tags + Context/Decision/Consequences + 4-section Forward action + 6-subsection References), 173 lines within 100-200 envelope.

### review-integration — APPROVE (with Warning)

Summary: PR-G integration READY for #349 consumption + future cleanup PR (D8 fall-back) consumption. Diff scope clean (6 files, 0 SHUD/openspec/.gitmodules touch).

Findings:

- **Warning (CONFIRMED, dup of review-documentation)**: ADR-0003 §References PR-letter labels offset by one vs sibling docs

Non-blocking notes (Praise):
- #349 archive readiness PASS: openspec/changes/ diff empty; ADR + capstone provide rich Context/Decision/Consequences
- Future cleanup PR (D8 fall-back) readiness PASS: ADR §Forward action #1 (L88-96) enumerates 7 actionable revert items
- review-loop-log.jsonl epic_summary PASS: single epic-level JSON line, kind=epic_summary, total_prs=9, prs=[341..349], verdict=NO-GO, forward_to_future[4]
- jok-mirror SUNDIALS canonical cite present in ADR L29/L68/L82/L110/L153 + capstone L24/L61/L210 + summary L24/L55/L107

### Phase 4.5 Verifier — SKIPPED (self-evident factual data lookup)

Rationale: 0 candidates required adversarial verification — Critical r_min finding is self-evident factual data lookup against canonical source-of-truth `docs/p8pre/n8_profile_verdict.md` §5. Warning PR-letter binding is structural verification against sibling docs grep. Both reproducible by orchestrator-direct grep + verified before Phase 5/6 fix.

### Round 1 verdict

**REQUEST CHANGES.** 1 Critical + 2 Warnings (Critical + Warning overlap on root cause r_min mis-attribution; Warning ADR PR-letter offset). Proceed to Phase 5/6 fix.

### Forward action notes (carry to PR-G #348 fix or #349 archive)

- 5 cosmetic Suggestions carried as forward debt to #349 archive (`.review-evidence/` gitignore patch + §3 column ordering cosmetic + §7 jok-mirror inline cite cosmetic + max_ulp precision range cosmetic + compare_snapshot raw-double hardening → future tools epic)
