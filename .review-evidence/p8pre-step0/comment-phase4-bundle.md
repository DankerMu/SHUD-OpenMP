## Phase 4 Cross-Review Evidence Bundle (round 1)

Reviewer agents: `review-correctness`, `review-integration`
Review round: round 1
Reviewed head SHA: `5827e237c86150dd4a16e8f68eeffc211f66c46f`
Local evidence: `.review-evidence/p8pre-step0/{correctness,integration}.md`

### review-correctness

Summary: Diff faithfully replaces 15-key list (verbatim from `tools/cvode_stats_diff/canonical_15_keys.yaml`) + 4 path strings (`.pr-i-runs/` → `.p1e-i-runs/` at L83/156/157/183); new `docs/p8pre/step1_prep.md` inline-quoted wall/nst/nfe baselines exactly match source-of-truth rows; no collateral damage, no test/spec weakening.

Findings:
- **None.**

Non-blocking notes:
- Issue #339 In-Scope item 4 (p2a §9 v0.5 nlcf typo removal) is not visible in diff but is already satisfied at head SHA (`grep -c nlcf docs/p2a/p2a_profile_baseline.md = 0`; likely cleaned by prior commit `386da4d`). No correctness risk.

### review-integration

Summary: Edits internally clean, but two sibling P1e docs still cite the old `.pr-i-runs/` path → orphan cross-ref drift; flagged as warning, not blocker.

Findings:
- **Warning / Cross-doc drift** — `docs/p1e/p1e_academic_summary.md:219` quotes `/scratch/frd_muziyao/SHUD-OpenMP/.pr-i-runs/`, and `docs/p1e/p1e_summary.md:113` shows `sbatch --array=0-23 .pr-i-runs/run_pr_i_24cell.sbatch`. Identical fix mechanics as the four corrected lines. **Blocks merge: NO**. Requested fix: follow-up PR or note as out-of-scope.

Non-blocking notes:
- `step1_prep.md` relative links `../p1e/p1e_perf_baseline.md` resolve; §3.1/§3.4 anchors match real headers
- Sibling archive paths `.pr-d-runs/`, `.pr-j-runs/`, `.pr-c-runs/` at `p1e_perf_baseline.md:184-186` intentionally NOT renamed (canonical, different drift class)
- 15-key list ripple sweep clean: only `p1e_perf_baseline.md:83` cites the canonical list; no other doc/yaml/sh retains `nlcf/nfevals/hcur/qcur/hin`
- `canonical_15_keys.yaml` matches `tools/cvode_stats_diff/cvode_stats_diff.sh:54` `CANONICAL_KEYS` token-for-token; asserted in `test_15key_excludes_nfcall.py:73`
- `tools/p1e_aggregate_pr_i_shall.sh` is a real script filename (PR-I identifier, not archive-root typo) — intentionally unchanged
- `openspec/glossary.md` untouched — matches deferred-to-PR-G scope
- `SHUD_openMP_master_plan.md` and `.github/workflows/serial-baseline.yml` carry zero `.pr-i-runs/` occurrences

### Phase 4.5 Verifier Verdict (cand-01)

Reviewer: `verifier` subagent (independent of `review-integration`)
Verdict: **REFUTED-out-of-scope**

Evidence: Both cited lines exist verbatim — `p1e_academic_summary.md:219` and `p1e_summary.md:113` do contain `.pr-i-runs/`. However Issue #339 body "Module / Scope" + "In Scope" explicitly enumerates ONLY `docs/p1e/p1e_perf_baseline.md` (§3.5 + §7.1) and `docs/p2a/p2a_profile_baseline.md` (§9 v0.5 row). `openspec/changes/p8pre-spike/tasks.md` §0.2 (L4-10) mirrors this scope and never names the two flagged files. Per fixture-level `none` precision-bias rule (only CONFIRMED merge-blocks) AND scope discipline, the candidate is REFUTED for THIS PR.

Action: dropped from blocking set + spawned follow-up task chip (cross-ref hygiene cleanup in a separate single-PR follow-up; does not block or expand Step 0).

### Round 1 verdict

**Clean** (per `none` fixture-level precision bias: only CONFIRMED merge-blocks; 0 CONFIRMED + 0 merge-blocking PLAUSIBLE). Proceed to Phase 7.
