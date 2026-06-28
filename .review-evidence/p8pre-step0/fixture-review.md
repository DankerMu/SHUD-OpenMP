Fixture review: pass
Reviewed change: p8pre-spike
Reviewed scope: tasks.md §0 (Issue #339 slice)
Reviewed head SHA: 5827e237c86150dd4a16e8f68eeffc211f66c46f
Missing axes:
- None.
Required additions:
- None.
Notes:
- §0 is cleanly separable from §1–§10. §0.1 (epic intake) is the only cross-boundary item but it is GitHub-issue creation, not doc-correction; the doc-correction work in §0.2 + §0.3 is self-contained and does not depend on §1+ Slurm runs, SHUD pin bump, or aggregator scripts.
- All four Issue #339 AC items map 1:1 to tasks.md §0.2 sub-bullets (canonical-15-key at line 5, `.p1e-i-runs/` path at line 6 with verification command `grep -c "p1e-i-runs" docs/p1e/p1e_pr_i_strict_omp_verification.md ≥ 8` verbatim, §7.1 sync at line 7, nlcf-typo audit at line 8 with `grep -c "nlcf" docs/p2a/p2a_profile_baseline.md = 0` verbatim, inline-quote at line 9).
- Canonical source citation chain is clean: tasks.md:5 cites `tools/cvode_stats_diff/canonical_15_keys.yaml`; the yaml lists `ncfl` (line 23) and explicitly excludes `nFCall` family — the SUNDIALS canonical key authority is correctly anchored. tasks.md:8 explicitly states `ncfl` is canonical per `SHUD/src/Equations/cvode_config.cpp:104`, not the `nlcf` typo.
- Fixture level `none` defensible per `issue-risk-contract.md` line 59: §0 is docs-only, metadata-only path/key string replacements with no runtime behavior, no parser/writer/schema/auth/migration/solver/CVODE/OpenMP/snapshot/profile-timer code touched. Mandatory expanded triggers (lines 57–77) all map to §1–§10 scope (Slurm runs, MD_precond_identity.cpp, cvode_config.cpp:259 PREC_LEFT edit, Timer emit), correctly out of this PR.
- Documentation + Legacy-compatibility risk packs are the correct and only packs (no code change → no concurrency/auth/file-IO/public-API surface).
- proposal.md does not surface a per-PR "docs-only" boundary statement, but tasks.md:4 (`single PR touching only docs, no code`) plus design.md §Non-Goals (which is epic-level) plus Issue #339 body "In Scope" / "Out of Scope" block jointly establish the doc-only invariant. Sufficient, not redundant.
- Current branch state (`grep -c "pr-i-runs" docs/p1e/p1e_perf_baseline.md = 0`, `grep -c "nlcf" docs/p2a/p2a_profile_baseline.md = 0`, `grep -c "p1e-i-runs" docs/p1e/p1e_pr_i_strict_omp_verification.md = 8`) confirms the §0 spec is implementable and the AC thresholds are reachable — the §3.5 / §7.1 edits have already landed on this branch (lines 83, 156, 157, 183), and §9 v0.5 row already spells `ncfl` correctly at line 204.
