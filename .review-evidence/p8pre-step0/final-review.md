Reviewer agent: phase-7-final-review
Review round: final
Reviewed head SHA: 5827e237c86150dd4a16e8f68eeffc211f66c46f
Summary: Gap Sweep clean. All 6 AC oracles PASS, no new defects beyond Phase 4 cand-01 (already verifier-REFUTED out-of-scope). APPROVE merge.

Gap Sweep findings (only items NOT already listed in Phase 4):
- None.

Pre-existing consumer compatibility:
- CI: `.github/workflows/serial-baseline.yml` does not grep `p1e_perf_baseline.md` / `pr-i-runs` / `p1e-i-runs` patterns (verified: `grep -n` returned empty). §3.5 sentence rewrite preserves `aggregator: tools/p1e_aggregate_pr_i_shall.sh` on `docs/p1e/p1e_perf_baseline.md:84` — no downstream tooling reference broken.
- New `docs/p8pre/step1_prep.md` §5 cross-ref table (L57-63) maps cleanly to tasks.md §1-§8 (PR-A self-check → §2 wall; PR-B aggregator → §3 nst + §4 nfe; PR-C capstone → §2-§4; PR-F gate-4 → §2 wall; ADR-0003 ROI → §2 sp@8). Anchor doc is self-contained.
- Historical archives under `.p1e-i-runs/<case>_N<n>_rep<r>/cvode_stats.txt` already emit canonical 15 keys (per `tools/cvode_stats_diff/canonical_15_keys.yaml`); doc rewrite re-aligns prose to existing on-disk reality — no migration / backfill required. Scope statement (issue #339 "PR-Step0: 单 PR 仅 docs, no code") is honest.
- Confirmed: no downstream consumer broken.

Completion self-audit:
- AC1 grep "pr-i-runs" in p1e_perf_baseline.md = 0: PASS
- AC2 grep "p1e-i-runs" in p1e_perf_baseline.md = 4: PASS (§3.5 L83 ×1 + §7.1 L156-157 ×2 + §7.3 L183 ×1)
- AC3 grep "nlcf" in p2a_profile_baseline.md = 0: PASS
- AC4 grep "nlcf" in p1e_perf_baseline.md = 0: PASS
- AC5 grep "lenrwLS" in p1e_perf_baseline.md = 1: PASS (§3.5 L83 canonical 15-key list)
- AC6 grep "p1e-i-runs" in p1e_pr_i_strict_omp_verification.md = 8: PASS (canonical anchor unchanged this PR)
- §0.2 sub-bullet 1 (§3.5 + §7.1 15-key canonical): PASS — `docs/p1e/p1e_perf_baseline.md:83` lists `nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS`, exact order matches `tools/cvode_stats_diff/canonical_15_keys.yaml`
- §0.2 sub-bullet 2 (§3.5 path fix): PASS — `docs/p1e/p1e_perf_baseline.md:83` `.p1e-i-runs/`
- §0.2 sub-bullet 3 (§7.1 path fix): PASS — `docs/p1e/p1e_perf_baseline.md:156-157` 2 sbatch lines fixed; §7.3 footprint table L183 also fixed (bonus consistency)
- §0.2 sub-bullet 4 (§9 v0.5 row nlcf removal in p2a): PASS — `grep "nlcf" docs/p2a/p2a_profile_baseline.md` = 0 (pre-existing state confirmed by AC oracle; PR diff does not touch this file but file was already correct per prior fix-up)
- §0.2 sub-bullet 5 (inline-quote §3.1 to step1_prep): PASS — `docs/p8pre/step1_prep.md:25-28` wall table + L39-42 nst table + L50-51 nfe baseline all inline-quoted with cross-ref to source-of-truth p1e docs

Oracle integrity: PASS — no AC weakened, no fixture rewritten, no test removed. Phase 0.5 fixture review (PASS) + Phase 4 round 1 (0 correctness findings, 1 integration finding REFUTED-out-of-scope) all remain valid at head 5827e23.

Final-review verdict: clean
