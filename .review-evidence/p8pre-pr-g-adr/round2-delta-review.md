Reviewer agent: review-documentation (Phase 6.5 delta)
Review round: round 2
Reviewed head SHA: a149c5f
Delta range: 8cf527c..a149c5f
Summary: Fix incomplete — ADR §Step 1 (L21) still mis-attributes Step 2 number 12120/6696=1.811 as Step 1 ROI; remaining ADR 1.811 hits at L58/L69 also Step 1 context. capstone/summary fixes clean.

Findings:

- Critical (review-correctness, F-R2-1): `docs/adr/0003-precond-spike-decision.md:18-25` — section header is "**Step 1 (PR-A/B/C, #341-#343, #353-#355)** — N=8 Mode C profile recheck (18-cell 2×3×3 矩阵, SHUD pin `7a1dc8f`)" but L21 reports `heihe N=8: 12120 / 6696 = **1.811**`. This is **Step 2 PREC_LEFT data** per capstone L167 NB (Step 2 rep1 PREC_LEFT cvode_stats nfeLS=12120/nfe=6696) and per capstone L175 (Step 2 table). Step 1 PREC_NONE baseline median (per `docs/p8pre/n8_profile_verdict.md` §5 + capstone L167 (i) + p8pre_summary.md L20) is `12632/6943 = 1.819`. L24 of the same section correctly cites the Step 1 anchor `heihe.nfe=6943`, creating internal contradiction (numerator-denominator and anchor disagree). Fix: replace L21 with `heihe N=8: 12632 / 6943 = **1.819** (Step 1 PREC_NONE baseline median)`. This is the exact bug round 1 flagged — the fix landed in capstone+summary but NOT in ADR §Step 1.

- Critical (review-correctness, F-R2-2): `docs/adr/0003-precond-spike-decision.md:58` Rationale §3 — "`nfeLS/nfe = 1.811` ROI 仍然 promising" cites Step 1 ROI window as 1.811. Per ADR §Step 1 section, the Step 1 ROI baseline is the comparison anchor (1.819), not the Step 2 observed (1.811). Fix: replace 1.811 with 1.819 (or annotate as "Step 1 baseline 1.819 / Step 2 实测 1.811" if both are intended).

- Critical (review-correctness, F-R2-3): `docs/adr/0003-precond-spike-decision.md:69` Consequences Positive — "ROI ceiling 实测落地: `nfeLS/nfe = {1.811, 4.526}` 比值是任何 future preconditioner candidate 必须超越的 baseline。这些数字 + Step 1 PR-A `wall_step1_baseline_median` 共同形成 future iterative-solver tuning 的 anchor". This explicitly labels 1.811 as the Step 1 future-tuning anchor, but the canonical anchor per `docs/p8pre/n8_profile_verdict.md` §5 = 1.819. Fix: 1.811 → 1.819 (heihe_x4 4.526 already correct).

Non-blocking notes:

- Praise: capstone L167 NB rewrite is exemplary — explicit 3-number enumeration (i)/(ii)/(iii) with derivations + closure rule disambiguates a confusing data triple cleanly. Reusable template for future ROI-table NB notes.

- Praise: capstone L24/L92/L165/L228/L290 + p8pre_summary L20/L45/L70 fixes all clean — each 1.819 hit annotated with derivation "12632/6943" or "Step 1 canonical" or "Step 1 baseline" context, no orphan.

- Praise: ADR §References Epic+PRs rewrite (L116-131) cleanly aligned to canonical mapping PR-A=#341 / .../ PR-G=#348; NB on #340 absorbed scope (L133) is well-placed.

- openspec validate p8pre-spike --strict --no-interactive exit 0 (PASS).

- Line counts within range: capstone 395 (350-500), summary 113, ADR 175 — all unchanged from round 1 PASS items.

- Cross-doc NO-GO consistency holds (capstone §1 + ADR §Decision + summary §决策 all NO-GO).

Verdict: REQUEST CHANGES — 3 Critical r_min mis-attribution findings remaining in ADR-0003 (L21/L58/L69). Round 1 critical was only partially fixed: capstone + summary clean, but ADR (the actual decision document) still carries the bug. The fix must land in ADR §Step 1 to be consistent with capstone L167 NB closure rule that the commit itself introduced.
