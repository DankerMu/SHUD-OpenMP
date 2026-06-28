Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 49a2d519f3d63a0b6e1cdd6e9f39e7bba138ffe8
Summary: Aggregator numerics, REJECT logic, branch tree, and verdict doc all correct; no findings.

Findings:
- None.

Non-blocking notes:
- Median impl (aggregate_n8_profile.sh:301-307): 3-element sort-by-3-swaps is correct; final `b` is mid-rank. Verified manually with (6943,6943,6943)=6943. For 3 distinct values the swap chain `(if a>b swap)(if b>c swap)(if a>b swap)` yields a≤b≤c with b=median.
- Cross-N invariance (L370): integer string `!=` comparison (not `=`/numeric), correct for Δ=0 strict on integer counters. nst/nfe/nfeLS/nni/nsetups confirmed integer-emitted (no dot tokens).
- Ratio scale (L416): `awk printf "%.3f", a/b`. Spot-checks 12632/6943 = 1.819386 → "1.819" and 30509/6741 = 4.525886 → "4.526" match stdout/verdict doc exactly.
- Branch tree (L462-480): for r_min=1.819, r_max=4.526 → `ge_proceed=1` (1.819 ≥ 1.5) fires first → branch="a". Matches stdout `branch: a` and §3.5 header. Decision tree is mutually exclusive and exhaustive (a → b → d → c default), correctly distinguishes branch d (heterogeneity precedence) from branch c.
- REJECT typo regex (L137): `grep -qE "^${bad}="` anchored on start-of-line. `nlcf=` would match; `ncfl=` would NOT match the `nlcf=` pattern (different chars at positions 1-2). No false-positive risk for the valid `ncfl` key. Verified all 5 REJECT keys (nlcf/nfevals/hcur/qcur/hin) distinct from canonical 15. 0 hits across 18 cells confirmed.
- Absolute baseline check (L390-399): string `!=` on integer median. heihe nst=6698=expected, nfe=6943=expected; heihe_x4 nst=6575=expected, nfe=6741=expected. All PASS.
- nfeLS = nli identity (§3.4 doc + L758-760): aggregator data confirms `nfeLS − nli = 0` for all 6 (case, N) groups. Doc correctly attributes this to SUNDIALS CVLS SPGMR with FD-Jvp (one RHS call per GMRES inner iter). The `r = nfeLS/nfe` ROI metric is still well-defined as the linear-solver-work-per-nonlinear-eval amplification factor.
- §5 ROI 论述 expected speedup window "10–25% of t_CVODE_internal" is a forward-looking estimate for a future non-trivial preconditioner; PR-D §6 identity-precond spike will only verify API gating (ncfn=0, nps>0, npe>0), not perf — verdict doc framing is consistent with spec.
- §6 limitations correctly flag 90-day truncation (CLAUDE.md rule) and 2-case scope (heihe + heihe_x4).
- No SHUD pin change, no openspec/changes/p8pre-spike/ edits, no canonical_15_keys.yaml edits — diff cleanly scoped to 2 new files (+1130) as advertised.
- Minor stylistic: integer-detection in median awk (L310-314) is per-token (any dot → float). Robust against mixed-precision reps because timer.cpp always emits `%.9f` (always dotted) for floats, and CVODE stats always emit integers (no dot). No mode-switching footgun.

Verdict: APPROVE.
