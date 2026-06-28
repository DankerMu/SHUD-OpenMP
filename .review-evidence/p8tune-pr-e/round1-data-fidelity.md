Reviewer agent: review-data-fidelity
Review round: round 1
Reviewed head SHA: 38ec6135c4cdb9eb6141461b77fd3122c4dda979

Summary: All 10 data-fidelity checks PASS. Every quantitative claim in docs/p8tune/maxl_sweep_verdict.md and docs/adr/0004-maxl-sweep-decision.md reproduces exactly from raw 60-cell summary.tsv. Aggregator logic independently reimplemented + byte-identical results. No findings. APPROVE.

Findings: None.

Non-blocking notes:
- ADR-0004 L89 cites "−7% 至 −23%" for heihe_x4 maxl=10; exact figures are −6.86% and −22.97% — fair rounding
- ADR-0004 L95 estimates "heihe ~6300 elem"; actual 6335 — irrelevant for cache-pressure argument
- T7 max_ulp values (4427011158482776 / 4584600898546929175) not re-verified at byte level (would require rivqdown.dat binary recomputation); covered by review-correctness instead

Per-check assessment:
1. G4 delta verification:                 pass (4/4 exact: heihe Δncfl=−85 Δncfn=−2 ΔnfeLS=−51; heihe_x4 Δncfl=−3620 Δncfn=−51 ΔnfeLS=+6710 — matches T4)
2. G5 wall median verification:           pass (16/16 exact: medians + improvement_pct match T5 to two decimals; all 16 band classifications correct per threshold GO≥10/Optional 5-10/Diagnostic 1-5/Neutral ±1/REGRESSION <−1)
3. G6 sum_delta verification:             pass (16/16 exact: heihe sum_delta=−87, heihe_x4 sum_delta=−3671; all ≤ 0)
4. G8 cross-rep consistency:              pass (20/20 tuples bit-identical across 3 reps for 15 counter keys + rivqdown_sha12)
5. Per-case recommendation:               pass (4/4 exact: heihe N=1 argmax=30 @ +11.99%; heihe N=8 argmax=30 @ +6.78%; heihe_x4 N=1 best=−6.86% <1.0 → default 5; heihe_x4 N=8 best=−15.81% <1.0 → default 5)
6. ADR-0004 quantitative claims:          pass (heihe N=1 maxl=30 +11.99% exact; heihe_x4 N=1 maxl=30 −15.83% exact; heihe_x4 N=8 maxl=10 −22.97% exact; ncfl 85→0 + 3620→0 exact; ncfn 7→5 + 51→0 exact)
7. case-size-asymmetric claim:            pass (heihe=6335, heihe_x4=40046 per CLAUDE.md L92; ADR cite "~6300/~40046" fair; working-set arithmetic reproduces within rounding)
8. OMP-neutrality preservation:           pass (10/10 (case,maxl) pairs: N=1 sha12 == N=8 sha12)
9. No silent data truncation:             pass (60/60 rows; 0 rows wall_s ≤ 0)
10. Counter-vs-wall divergence narrative: pass (ADR cited spans match T5 exactly)

Verdict: APPROVE — verdict.md + ADR-0004 are 100% data-faithful to ground truth. Aggregator computation correct. No data truncation, no rounding cheats, no orphan claims. Safe to ship.
