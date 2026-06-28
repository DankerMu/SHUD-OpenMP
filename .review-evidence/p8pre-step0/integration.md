Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 5827e237c86150dd4a16e8f68eeffc211f66c46f
Summary: Edits internally clean, but two sibling p1e docs still cite the old `.pr-i-runs/` path → orphan cross-ref drift; flagged as warning, not blocker.

Findings:
- Warning / Cross-doc drift / Single canonical archive-root naming `.p1e-i-runs/` / `docs/p1e/p1e_academic_summary.md:219` quotes `/scratch/frd_muziyao/SHUD-OpenMP/.pr-i-runs/` for the server artifact root, and `docs/p1e/p1e_summary.md:113` shows `sbatch --array=0-23 .pr-i-runs/run_pr_i_24cell.sbatch`. Issue #339 task §0.2 fixed §3.5/§7.1/§7.3 of `p1e_perf_baseline.md` but left these P1e siblings stale. A future reader following the cross-ref chain (`p8pre/step1_prep.md` §1 → `p1e_perf_baseline.md` §3.5 → P1e narrative) will hit `.pr-i-runs/` in the academic/summary docs and may re-introduce the drift / `Required test or evidence: grep -n "\.pr-i-runs" docs/p1e/*.md` returns these two lines and zero from `p1e_perf_baseline.md` (verified) / Sibling surfaces: identical fix mechanics as the four lines already corrected on §7.1/§7.3 / Blocks merge: NO (issue scope was scoped to `p1e_perf_baseline.md`; can defer to a follow-up doc-correction PR) / Impact: cosmetic if no rerun, latent if someone scripts archive lookup from academic_summary / Requested fix: replace 2 occurrences in a follow-up PR; OR explicitly note them as out-of-scope in this PR's body so the deferral is auditable.

Non-blocking notes:
- step1_prep.md relative links `../p1e/p1e_perf_baseline.md` and `../p1e/p1e_pr_i_strict_omp_verification.md` resolve (both real files at `docs/p1e/`); §3.1 and §3.4 anchors quoted in step1_prep.md §2/§3 match real headers at `docs/p1e/p1e_perf_baseline.md:37` and `:68`.
- Sibling archive paths `.pr-d-runs/`, `.pr-j-runs/`, `.pr-c-runs/` at `docs/p1e/p1e_perf_baseline.md:184-186` are intentionally NOT renamed; they are canonical (per `.review-evidence/p8pre-step0/fixture-review.md` scope = only `.pr-i-runs/` drift). They have own naming convention and only `.pr-i-runs/` was the typo.
- 15-key list ripple sweep: only the corrected line `p1e_perf_baseline.md:83` contains the canonical 15-key list. No other repo doc/yaml/sh under the canonical sweep cites the wrong (`nlcf/nfevals/hcur/qcur/hin`) list — clean.
- `canonical_15_keys.yaml` (15-key list) matches `cvode_stats_diff.sh:54` `CANONICAL_KEYS` token-for-token and is asserted-in-sync by `test_15key_excludes_nfcall.py:73`. The doc's "per canonical_15_keys.yaml" claim is accurate.
- `tools/p1e_aggregate_pr_i_shall.sh` is a real filename (not a path-drift artifact); the `pr_i` in the script name reflects the PR-I identifier, not the archive-root typo. Intentionally unchanged.
- `openspec/glossary.md` untouched in this PR; matches issue body's "glossary update deferred to PR-G" scope.
- `SHUD_openMP_master_plan.md` and `.github/workflows/serial-baseline.yml` carry zero `.pr-i-runs/` occurrences → no out-of-scope master-plan drift exposed.
