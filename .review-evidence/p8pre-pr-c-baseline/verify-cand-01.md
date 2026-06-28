Verifier agent: phase-4-5-verifier
Candidate ID: cand-01-spec-tasks-section-anchor-drift
Originating reviewer: review-integration
Reviewed head SHA: 43bd6f2a5c76eda2b9bbc577a6121020bae12b71
Verdict: CONFIRMED

Rationale: Drift is real and constructible. n8_profile_baseline.md uses academic ordering: §3 = Methodology (L57), §4 = Experimental Setup (L107), §5 = Results (L127), and Table 1 `wall_step1_baseline_median(case, N)` lives at §5.1 L129-142 — not in §3. Three openspec files cite "§3 raw data table" as the gate-4 baseline source, contradicting the doc. A PR-F implementer reading tasks.md verbatim would grep §3, find only the 5-gate methodology, and miss Table 1. Fix is a 5-line sed in already-gitignored openspec/changes/, mirroring PR #353 cand-02 in-PR precedent.

Evidence:
- `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md:113` claims "§3 raw data tables ... the §3 wall_median(case, N) table is the SOURCE OF TRUTH for `wall_step1_baseline_median(case, N)`"; actual doc at `docs/p8pre/n8_profile_baseline.md:129` is "§5.1 Raw data table — gate-4 baseline anchor (CORE)" containing Table 1 (L133-142). Same drift at `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md:90` ("§3 raw data table — NOT to PR-I") and `tasks.md:41` ("§3 raw data + §4 ratio analyses + §5 cross-N invariance"), `tasks.md:77` ("archived per task 4.1 ... §3 raw data table"), `tasks.md:81` (gate-4 baseline reference chain).
- Actual §3 content (L57-103) = Methodology only: §3.1 experiment matrix, §3.2 build flags + nm gates, §3.3 determinism env, §3.4 5-gate suite description, §3.5 Slurm chain. Zero raw data tables. Doc §3.4 item 5 even self-references "Wall median archive (this document §5.1 — the gate-4 anchor)" — proving the doc author intended §5.1, but specs/tasks lagged.
- `.gitignore:13` confirms `openspec/changes/` is gitignored — PR-F implementer reads from working tree, not git history; drift propagates without fix. Doc itself (`docs/p8pre/n8_profile_baseline.md`) is tracked and internally consistent; only the openspec fixture mis-labels the anchor section number.

Note: Reviewer called it "non-merge-blocking, foldable into PR-F/G", but per compact fixture precision-bias the drift IS constructible and the fix IS in-scope (single sed across 5 lines in this very PR's openspec fixture); CONFIRMED merge-blocks.
