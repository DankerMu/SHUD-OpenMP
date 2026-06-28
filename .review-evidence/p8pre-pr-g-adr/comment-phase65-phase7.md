## Phase 6.5 Round 2 Delta Review + Round 3 Follow-up + Phase 7 Final Review

### Phase 6 round 1 fix commit `a149c5f`

9 surgical edits per Phase 4 verified findings:
- 7 substitutions 1.811 → 1.819 in `docs/p8pre/capstone.md` (L24, L92, L165, L228, L290) + `docs/p8pre_summary.md` (L20, L45, L70), each annotated "Step 1 canonical" / "Step 1 baseline" / "Step 1 PR-B median verdict" disambiguation
- `docs/p8pre/capstone.md` L167 NB note rewritten to explicitly enumerate 3 nfeLS/nfe numbers + closure rule "本 capstone 在所有 'Step 1 ROI baseline' 语境用 canonical 1.819; '1.811' 仅在描述 Step 2 spike PREC_LEFT 实测时出现"
- `docs/adr/0003-precond-spike-decision.md` §References block rewritten to canonical PR-letter mapping (PR-A=#341 / ... / PR-G=#348 / archive=#349) + NB on #340 absorbed scope

### Phase 6.5 round 2 delta review (review-documentation @ a149c5f)

Verdict: **REQUEST CHANGES** — 3 new Critical findings exposed by the round-1 fix scope being incomplete.

Findings:

- **Critical F-R2-1** (CONFIRMED): `docs/adr/0003-precond-spike-decision.md` L21 cites `12120/6696=1.811` under header L18 "Step 1 (PR-A/B/C, #341-#343)" — section-header context says Step 1 but the data is Step 2 PR-E PREC_LEFT raw. Same bug round 1 flagged in capstone+summary, missed in ADR §Step 1.
- **Critical F-R2-2** (CONFIRMED): `docs/adr/0003-precond-spike-decision.md` L58 Rationale §3 "nfeLS/nfe = 1.811 ROI 仍然 promising" — ROI argument rests on Step 1 PREC_NONE baseline 1.819.
- **Critical F-R2-3** (CONFIRMED): `docs/adr/0003-precond-spike-decision.md` L69 Consequences Positive "nfeLS/nfe = {1.811, 4.526} 比值是任何 future preconditioner candidate 必须超越的 baseline" — Step 1 canonical anchor 1.819.

Praise (round 2):
- capstone L167 NB rewrite is exemplary — explicit 3-number (i)/(ii)/(iii) enumeration with derivations + closure rule disambiguates the data triple cleanly.
- capstone L24/L92/L165/L228/L290 + p8pre_summary L20/L45/L70 1.819 fixes all annotated with derivation ("12632/6943", "Step 1 canonical", "Step 1 baseline"); no orphan 1.811 in Step 1 context outside the NB.
- ADR §References Epic+PRs rewrite L116-131 cleanly aligned to canonical PR-A=#341 / ... / PR-G=#348.

### Phase 6 round 3 fix commit `29a6a06`

3 surgical ADR edits per Phase 6.5 verified Critical:
- L20-23 ROI block rewrite: heihe N=8 = `12632/6943=1.819` (Step 1 PREC_NONE baseline median per `docs/p8pre/n8_profile_verdict.md` §5 canonical) + inline NB clarifying Step 2 PR-E PREC_LEFT same-case ratio `12120/6696=1.811` is different condition
- L58: "Step 1 canonical `nfeLS/nfe = 1.819`" replacing "1.811"
- L69: "Step 1 PREC_NONE canonical `nfeLS/nfe = {1.819, 4.526}`" + explicit "ncfn floor (Step 2 spike 数据)" annotation for 6/47 pair

### Phase 7 final review (Gap Sweep @ 29a6a06)

Reviewer agent: phase-7-final-review
Verdict: **NEEDS DISCUSSION** — 1 new Warning surfaced

Findings:

- **Warning F-PH7-1** (CONFIRMED): `SHUD_openMP_master_plan.md` L2383 `nfeLS/nfe = 1.811 贴近 trigger threshold`. Counter set (`nst=6599 / nfe=6696 / nfeLS=12120`) is Step 2 PR-E PREC_LEFT identity-precond data. Structural placement under §Step 2 header L2372 provides context, but lacks explicit "Step 2 PREC_LEFT identity-precond" attribution NB that ADR L23 + capstone L167 carry. Same risk pattern Phase 6.5 caught on ADR L21/L58/L69, weaker risk here due to section-scoped context.

### Phase 6 round 3 follow-up fix commit `06746b8`

1 surgical master-plan edit:
- L2383 prefix "Step 2 PREC_LEFT identity-precond 实测 ROI 量化 (heihe N=8 representative, post-PREC_LEFT cvode_stats)" attribution + inline "Step 1 PREC_NONE canonical baseline anchor = `12632/6943 = 1.819` per `docs/p8pre/n8_profile_verdict.md` §5 — 不同 condition, CVODE step controller path differs nst 6698→6599" clarification

### Round 3.5 verification post-fix @ 06746b8

After 06746b8:
- 1.811 occurrences (all 4 doc paths): only 3 explicit Step-2-PREC_LEFT NB contexts (`docs/p8pre/capstone.md` L167 NB + `docs/adr/0003-precond-spike-decision.md` L23 NB + `SHUD_openMP_master_plan.md` L2383 NB)
- 1.819 occurrences (all 4 doc paths): all Step 1 canonical / PREC_NONE baseline / Step 1 PR-B median annotated
- CI 5/5 PASS @ 06746b8

### Phase 7 final-review verdict (post-fix @ 06746b8)

Clean (after F-PH7-1 fixed). 9/9 AC PASS + oracle integrity PASS + CI 5/5 PASS + mergeStateStatus=MERGEABLE + #349 archive readiness PASS + #348 design D8 future cleanup PR readiness PASS.

Cross-doc canonical agreement after 06746b8:
- Step 1 baseline canonical `r_min = 1.819` (from `n8_profile_verdict.md` §5 PREC_NONE) cited in: `docs/p8pre/capstone.md` L24/L92/L165/L228/L290 + `docs/p8pre_summary.md` L20/L45/L70 + `docs/adr/0003-precond-spike-decision.md` §Step 1 L21 + Rationale L58 + Consequences L69 + `SHUD_openMP_master_plan.md` §P8-precond.0 L2383 NB
- Step 2 PREC_LEFT spike ratio `1.811` cited explicit Step 2 attribution NB only: `docs/p8pre/capstone.md` L167 + `docs/adr/0003-precond-spike-decision.md` L23 + `SHUD_openMP_master_plan.md` L2383

**Verdict: Clean → proceed to Phase 8 evidence + merge.**
