# Spec-compliance review — PR #359 round 1

Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed head SHA: 75a757c
Summary: All 12 spec-compliance checklist items PASS. Aggregator gate logic precisely matches spec L60-130; verdict doc contains all required sections (§1-§7) + 18-row raw data table with PR-E forward-carry; baselines and epsilons hardcoded correctly per `docs/p8pre/n8_profile_baseline.md` §5.1; `openspec validate p8pre-spike --strict` exit 0; SHUD pointer unchanged.

Findings:
- None.

Non-blocking notes:

1. **Gate 1 (build PASS)** — `aggregate_identity_spike.sh:295-303` greps for all 3 spec-required symbols (PSetupIdentity / PSolveIdentity / CVodeSetPreconditioner) with `>= 1` per symbol per spec L70-71; verdict doc §4 row 1 cites `server_nm.log: 3 hits (1 each)`. Matches spec.

2. **Gate 2 (ncfn = 0)** — `aggregate_identity_spike.sh:307-326` iterates all 18 cells, FAILs on any ncfn>0 per spec L77-79; verdict doc §4 row 2 = FAIL with deterministic 6/47 ncfn floor (heihe / heihe_x4) across 9/9 cells per case. Matches spec L74-79 "any cell ncfn>0 → NO-GO".

3. **Gate 3 (nps+npe accumulation)** — `aggregate_identity_spike.sh:342-361` enforces `nps>0 AND npe>0` per cell per spec L84-86; verdict doc §4 row 3 = PASS with min_nps=18163, min_npe=77.

4. **Gate 4 (wall non-regression)** — `aggregate_identity_spike.sh:60-83` hardcodes `wall_step1_baseline_median` per spec L90: `BASELINE_WALL_MEDIAN[heihe_1]=140.797 / heihe_4=95.734 / heihe_8=89.732 / heihe_x4_1=1412.895 / heihe_x4_4=849.704 / heihe_x4_8=743.552` — exact match to `docs/p8pre/n8_profile_baseline.md:148-153`. Epsilons `GATE_4_EPSILON[heihe]=0.10` and `[heihe_x4]=0.05` match spec L92-93 exactly. Source = `n8_profile_baseline.md` §5.1, NOT `p1e_perf_baseline.md` §3.1 — correct per spec L90 explicit prohibition. Verdict doc §4 Gate-4 table shows all 6 (case, N) groups PASS with max delta 2.64% (heihe N=1).

5. **Soft gate 5 (cross-N tolerance)** — `aggregate_identity_spike.sh:66-71, 396-436` uses per-case baseline SHA12 (`heihe=a2023ccd2de4`, `heihe_x4=b5e4b0a2cf83`) from spec L103. Strict check first, max-ulp fallback (threshold=1024) per spec L106; FAIL classification when strict + fallback both violate. Verdict doc §5 row 1 = FAIL with max_ulp ≈ 9×10¹⁵ + ADR-0003 carve-out citation. Matches spec L106-108.

6. **Soft gate 6 (setup overhead)** — `aggregate_identity_spike.sh:73-74, 438-468` computes `t_precond_setup / wall ≤ 0.05` per cell per spec L110-113, handles DEFER case if bucket absent. Verdict doc §5 row 2 = PASS with max ratio 1.01×10⁻⁷ (6 orders below threshold). Matches spec L111.

7. **Verdict doc §3 raw data table** (verdict.md:84-103) — 18 rows × 15 columns: JID + case + N + rep + Elapsed + ExitCode + nst + nfe + ncfn + nps + npe + wall_total + t_precond_setup + SHA12 + max_ulp. Satisfies spec L122 + issue checklist 13-15 column tolerance.

8. **Verdict doc §4 + §5 verdict tables** (verdict.md:113-118, 141-144) — present; §4 explicitly marks each hard gate PASS/FAIL; §5 marks each soft gate PASS/FAIL with reasoning. Per-(case,N) gate-4 breakdown table also present (verdict.md:122-129). Matches spec L122-123.

9. **Verdict doc §6 ROI implication** (verdict.md:156-173) — present; §6.2 explicitly quantifies ROI ceiling (heihe ncfn=6 / heihe_x4 ncfn=47 / nfeLS/nfe=1.811) as gate for future P8-precond candidates. Matches spec L122-124.

10. **Verdict doc §7 ADR-0003 recommendation** (verdict.md:177-197) — NO-GO with design D8 PREC_NONE fall-back explicit; PR-G #348 ownership of 7-item execution checklist (revert PREC_LEFT, delete `MD_precond_identity.{h,cpp}`, close `baseline/p8pre`, etc.). Verdict driven by hard-gate FAIL (gate 2) per spec L74-79. Matches spec L117-118 + design D8 (lines 103, 110, 138).

11. **`openspec validate p8pre-spike --strict --no-interactive`** → "Change 'p8pre-spike' is valid" exit 0.

12. **SHUD pointer forward-only invariance** — outer diff (`git diff HEAD~1 HEAD`) shows ONLY 2 files: `tools/p8pre/aggregate_identity_spike.sh` (+569) + `docs/p8pre/identity_spike_verdict.md` (+250). `git ls-tree HEAD~1 -- SHUD` == `git ls-tree HEAD -- SHUD` == `5276167eea67184d801905f54dc805d2cd61db2d`. SHUD pin unchanged in PR-F per spec L140-154 invariant (PR-F = local-side aggregator + verdict adjudication only, no SHUD bump). Matches spec.

**Additional positive observation**: aggregator deliberately matches sibling `aggregate_n8_profile.sh` (PR-B #342) style (POSIX bash + awk + grep) per source-comment lines 17-19, and uses `uv run python` for max-ulp only (per project rule). Code structure (Phase A-G + verdict synthesis) is highly modular and follows the spec-derived gate ordering, easing audit.

**Word count**: ~470.
