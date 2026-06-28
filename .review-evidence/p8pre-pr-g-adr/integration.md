Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 8cf527c
Summary: APPROVE with 1 Warning — diff scope clean (6 docs, 0 SHUD/openspec/gitmodules touch) and downstream consumers (#349 archive + future cleanup PR) cleanly unblocked; but ADR-0003 §References "Epic + PRs" PR-letter labels (L120-129 + L131 footnote) shift by one vs canonical labels used by capstone.md / p8pre_summary.md / master plan / review-loop-log epic_summary — non-blocking but confusing for #349 archivist.

Findings:

#### 🟡 Warning: ADR-0003 §References PR-letter labels inconsistent vs all sibling docs
`docs/adr/0003-precond-spike-decision.md:120-131`
ADR L120-129 list PR-A=#340/PR-B=#341/PR-C=#342/PR-D=#343/PR-E=#344/PR-F=#345/PR-G=#346/PR-H=#347/PR-I=#348/PR-J=#349. But `docs/p8pre/capstone.md:357-366`, `docs/p8pre_summary.md:94-103`, master plan §P8-precond.0 (#341-#348 inline), and `review-loop-log.jsonl` epic_summary `prs:[341,...,349]` all use canonical mapping PR-A=#341/PR-B=#342/PR-C=#343/PR-D=#344+#345/PR-E=#346/PR-F=#347/PR-G=#348/PR-H=#349. The L131 footnote then claims Step 2 = #344/#346/#347/#348 (skips #345 entirely) further compounding the drift. Self-inconsistent within ADR itself (ADR L29 says PR-D=#357 and ADR L82+L110 says PR-D=#357 — consistent with canonical mapping where PR-D maps to GitHub #345 merged as commit ce. #357). Because ADR-0003 is the primary input for #349 archive justification + future P8-precond formal epic re-evaluation prerequisite document, label drift will confuse the archivist. Recommend #349 fix or follow-up PR re-aligning ADR L120-129 + L131 to canonical PR-A...PR-G mapping. Not blocking merge — ADR L29/L82/L110/L148-149 all reference correct GitHub PR# (#357, #348) inline.

Non-blocking notes:
- Integration #349 archive readiness PASS: `git diff baseline/p8pre...8cf527c -- openspec/changes/` empty (0 byte diff). PR-G ADR-0003 + capstone provide rich Context/Decision/Consequences/Forward-action input for archivist to consume.
- Integration future cleanup PR (D8 fall-back) readiness PASS: ADR-0003 §Forward action recommendations #1 (L88-96) lists 7 concrete actions (cvode_config.cpp:259 revert + delete MD_precond_identity.{h,cpp} + Makefile unlink + Timer bucket optional removal + outer pointer bump + baseline/p8pre close). All paths + line numbers actionable.
- SHUD upstream state PASS: outer pointer SHUD = 5276167 unchanged (`git ls-tree 8cf527c SHUD` = 160000 5276167...); .gitmodules untouched.
- review-loop-log.jsonl epic_summary PASS: single epic-level JSON line at end, parseable, schema = `{epic, kind:epic_summary, date, total_prs:9, prs:[341..349], verdict:NO-GO, verdict_driver, forward_to_future[4]}`. Doesn't duplicate any per-PR entry. NB: per-PR entry for issue #348 is NOT present yet — will be appended by orchestrator Phase 8 commit per workflow convention.
- Master plan §P8-precond.0 future gating PASS: L2389 "P8-precond.1-.7 formal epic NOT triggered under current spike data"; L2393 P8-tune.A-D alternative candidates noted (not concretely epic'd).
- case_deployment_map §5.2 archive accessibility PASS-WITH-NOTE: 18 yaml paths under `/tmp/p8pre_identity_spike/` (Mac-local volatile). Section header explicitly says "Server source mirror: /scratch/.../identity_spike/<cell>/" so canonical archive is server-side. Acceptable.
- CI compat PASS: no CI workflow validates docs/p8pre/ or docs/adr/ or review-loop-log.jsonl; `openspec validate p8pre-spike --strict` exit 0 confirmed.
- Forward carries PASS: jok-mirror SUNDIALS canonical cite present at ADR L29/L68/L82/L110/L153 + capstone L24/L61/L210/L307/L341/L379 + summary L24/L55/L107; compare_snapshot raw-double hardening forward note present at capstone L272-274/L328-330 + epic_summary `forward_to_future` entry 4; PR-F 5 cosmetic Suggestions implicitly carried to D8 cleanup PR via Forward Action #1.
- Outer .gitignore: `.review-evidence/` NOT in .gitignore; `git status` shows `?? .review-evidence/` untracked. PR-G does not add it. Carry to #349 archive PR per epic_summary forward carry not blocking.
- Diff scope: 6 files confirmed (master plan +32/-2, ADR +173 new, case_deployment_map +29, capstone +395 new, p8pre_summary +113 new, review-loop-log.jsonl +1).

Verdict: APPROVE with 1 Warning — PR-G doc-only scope is clean, downstream consumers unblocked, all forward carries present. ADR PR-letter label drift is cosmetic non-blocking but should be fixed in #349 archive or follow-up to avoid archivist confusion.
