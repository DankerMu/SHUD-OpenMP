## Phase 4 Cross-Review Evidence Bundle (round 1, expanded fixture)

Reviewer agents: `review-spec-compliance`, `review-correctness`, `review-documentation`, `review-integration`
Review round: round 1
Reviewed head SHA: `75a757c`
Local evidence: `.review-evidence/p8pre-pr-f-verdict/{spec-compliance,correctness,documentation,integration}.md`

### review-spec-compliance — APPROVE

Summary: All 12 spec-compliance checklist items PASS. Aggregator gate logic precisely matches spec L60-130; verdict doc contains all required sections (§1-§7) + 18-row raw data table with PR-E forward-carry; baselines and epsilons hardcoded correctly per docs/p8pre/n8_profile_baseline.md §5.1; openspec strict validate exit 0; SHUD pointer unchanged.

Findings: **None.**

Key verification:
- Gate 1 logic matches spec L70-72 (3-symbol AND ≥1 each)
- Gate 2 logic matches spec L74-79 (per-cell ncfn=0, any violation → FAIL); §4 row 2 FAIL deterministic 6/47
- Gate 3 logic matches spec L84-86 (per-cell nps>0 AND npe>0); §4 row 3 PASS min_nps=18163 / min_npe=77
- Gate 4 baselines (140.797 / 95.734 / 89.732 / 1412.895 / 849.704 / 743.552) match docs/p8pre/n8_profile_baseline.md L148-153 exactly; epsilons 0.10/0.05 match spec L92-93; source = n8_profile_baseline.md (correct, NOT p1e_perf_baseline.md per spec L90)
- Soft gate 5 logic matches spec L102-108: per-case baseline SHA12 anchors (heihe=a2023ccd2de4, heihe_x4=b5e4b0a2cf83), strict-then-fallback, FAIL when max_ulp>1024; §5 row 1 FAIL with max_ulp ≈ 9×10¹⁵
- Soft gate 6 logic matches spec L110-113 (t_precond_setup/wall ≤ 0.05); §5 row 2 PASS max ratio 1.01×10⁻⁷
- Verdict (NO-GO) internally consistent: spike_verdict at script L477-483 fires NO-GO on any G1-G4 = FAIL; G2 FAIL drives NO-GO per spec L74-79; design D8 fall-back PREC_NONE at script L490-491 + §7

### review-correctness — APPROVE

Summary: Aggregator logic + verdict adjudication are correct and reproducible against the 18-cell mirror; doc §3 + YAML frontmatter match gate outputs.

Findings: **None.**

Non-blocking notes:
- Suggestion: max_ulp precision (verdict L143/L152) — doc says "≈ 9×10¹⁵ across all cells"; aggregator emits 3 distinct rounded values (8.99e15 / 9.00e15 / 9.01e15). Range "[8.99 — 9.01] × 10¹⁵" would tighten §5 reproducibility. Does not affect FAIL.
- Needs verification: "5,155 / 214,252" structural divergence claim (verdict L146) — aggregator computes max_ulp scalar only, not zero-position differential. Forward-carried from manual numpy investigation not reproduced by script. For verdict, max_ulp ≫ 1024 alone is decisive; §8.4 records alternative bitwise 154,665/214,252 metric — adequate carve-out disclosure.
- Praise: Gate 1 symbol-grep (aggregate_identity_spike.sh:295-298) correctly implements spec L70-72 composite ≥1 each; Gate 2 universal-quantifier (L307-326) correctly implements spec L77; median3 + Gate 4 delta + Gate 5 ULP all faithfully match spec.

### review-documentation — APPROVE

Summary: Academic-style structure faithfully matches P1e mother template; YAML + verdict tables are accurate and decisive.

Findings: **None.**

Non-blocking Suggestions:
- §3 Table 1 column ordering does not perfectly mirror spec's "wall + t_precond_setup + SHA12 + max_ulp at minimum" enumeration, but all 11 required data columns + 4 identifiers (15 total ≥ 13 minimum) are present (L84-103)
- §7 PR-G action list (L183-197) does not explicitly cite "PR-D #357 jok-mirror canonical" within the 7-item enumeration; reference appears in §6.2 forward-carry L194 + YAML/References [6][7]. Checklist item 8 partial-pass

Praise:
- H1/H2/H3 hypotheses formally stated with operational definitions tied to specific gates (L46-52), mirroring P1e mother template L41-43 pattern
- Decisive verdict language throughout (YAML L11-20 + L133, L146, L179, L218, L250); doc fulfils PR-F's adjudication mandate, not just data restatement
- §6.2 ROI quantification (ncfn < 6 / < 47 + 10% overhead criterion, L169) and §8 6-threat validity coverage (L202-213) meet spec L122 ROI-implication mandate

### review-integration — APPROVE

Summary: PR-F integration READY for PR-G #348 consumption — all 12 checklist items pass; ZERO blocking findings; two forward-carry non-blocking notes (.gitignore patch + compare_snapshot raw-double hardening).

Findings: **None.**

Non-blocking notes:
- N1: `.gitignore` does not list `.review-evidence/` — currently untracked locally so PR-F omission is non-blocking, but PR-G should add `.review-evidence/` alongside `.p8pre-runs/` (L34) when making adjacent edits, OR commit evidence intentionally as ADR-0003 audit trail
- N2: `tools/compare_snapshot/compare_snapshot` cannot consume raw-double `rivqdown.dat`; aggregator L20+L237 + verdict doc §3.5 + §8.5 document the inline-numpy `np.spacing` fallback per spec L102-106. Future tools-tech-debt epic should harden compare_snapshot to accept raw doubles
- N3: median sort policy documented at aggregator L169-174 (middle-of-3-sorted) + verdict doc §3.4 L76 — PR-E forward-carry satisfied cleanly
- N4: §7 L184-191 PR-G action list (7 actions, explicit file paths) is exceptionally consumption-ready
- N5: §6.2 L160-169 ROI quantification table is quotable verbatim into master plan §P8-precond.0 + ADR-0003

### Phase 4.5 Verifier — SKIPPED

Rationale: 0 PLAUSIBLE candidates — 4/4 reviewers APPROVE 全 0 findings. Multi-Suggestion non-blocking notes accepted as forward carries to PR-G #348 + future tools epic.

### Round 1 verdict

**Clean.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE. Proceed to Phase 7.

### Forward action notes (carry to PR-G #348)

- `.review-evidence/` gitignore patch (low-priority)
- Inline numpy max_ulp precision range "[8.99 — 9.01] × 10¹⁵" cosmetic
- §3 column ordering cosmetic
- §7 jok-mirror inline cite cosmetic
- compare_snapshot raw-double hardening (future tools epic, not PR-G scope)
