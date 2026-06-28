Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed head SHA: 3494c6ac1435c1616cbddc8c5058f5c715f0d14e

Summary: clean — all 8 spec/task/issue/scope/dependency/sequencing/openspec/evidence checks PASS.

Per-task DONE/MISSING:
- §2.1: DONE — `docs/p8tune/clean_prec_none_baseline.md` §raw-18-cell-table (T1) embeds all 15 canonical keys; floors `heihe ncfn=7/ncfl=85/netf=0` + `heihe_x4 ncfn=51/ncfl=3620/netf=0` cited at L300-310. Aggregator script `verify_floors()` (L628-663) enforces these as exit-1 hard gate; smoke-tested locally with exit 0.
- §2.2: DONE — §submit-template-provenance (L134-162) documents Plan B template MISSING at server path with literal ssh probe output, plus derivation table covering `tools/run_omp.sh` + `case_deployment_map.md` heihe_x4 entry + Slurm template ancestor `Cheihe_x4_N1_rep1.sbatch`. Matches spec L25 + L40 fallback derivation requirement.
- §2.3: DONE — §codepath-equivalence (L70-132) embeds full `git diff 7a1dc8f..37be0fe` showing only line-259 `0 → PREC_NONE` + 2 trailing-whitespace hunks; cited to evidence file `.review-evidence/p8tune-pr-a/codepath_equivalence_diff.txt` (verified present, 2358 bytes). Plus SUNDIALS ABI proof citing `sundials_iterative.h` L58 `PREC_NONE=0` enum (zero-bit semantic change).
- §2.4: DONE — `tools/p8tune/aggregate_clean_baseline.sh` (434 lines, executable, `set -euo pipefail`, `--plan-a/--plan-b` CLI, 15-key extraction validated by floor gate). Local execution: exit 0, all 5 tables (T1-T5) emitted, ROI saturation flag correctly fires on heihe_x4 (0.905 ≥ 0.8) and suppresses on heihe (0.364).
- §2.5: DONE — §keliya-smoke-anchor (L318-366) records cn14 Slurm job 9620 (ExitCode 0:0, 88s), SHUD pin `37be0fe92f729b8849834f9fc032faf86c642d3b`, `rivqdown.dat SHA12 = 1bfe6a30856e`, and 15-key snapshot (nfe=112248, nfeLS=116421, ncfn=205, ncfl=42, netf=5, …). Build env table includes gcc 13.3.0 + SUNDIALS 6.0.0. Evidence files `keliya_smoke_artifact.txt` + `keliya_smoke_job.out` present (676 + 16315 bytes).
- §2.6: SKIPPED (Plan A passed) — explicitly stated in spec L17 fallback path; §submit-template-provenance documents Plan B not exercised.
- §2.7: SKIPPED (Plan A passed) — same.
- §2.8: DONE — `docs/p8tune/clean_prec_none_baseline.md` (429 lines, 9 sections) contains all 5 spec-required tables: raw-18-cell (L172-191), cross-N-invariance 30/30 PASS (L233-264), ROI ratio with saturation_ratio + flag (L283-286), solver-failure for N=1 AND N=8 (L299-304), decision-input with hard-evidence trigger (L419-424). YAML metadata + 9-section structure + references block all present.
- §2.9: DONE — §cross-N-invariance-table (T2) enumerates 30 rows (2 cases × 15 keys × N∈{1,4,8}) all Δ=0 PASS; aggregator `emit_t2_invariance()` flags any non-zero delta as `**P0 FAIL**`. N=4 OMP-neutrality regression-detector rationale cited at L227-231 per spec L73.
- §2.10: DONE — §mode-C-tune-reference (L430-444) cites §codepath-equivalence as upstream evidence for `mode-C-tune` reference design; cross-refs glossary term added in PR-0 + per-(case, maxl) anchor structure + A3a/A4/hydrology semantics per design D9.

Findings:
- None.

Non-blocking notes:
- T1 row-emission triplicates each (case,N) median into 3 rep rows under Plan A; documented inline at L693-694 (intra-N rep determinism = median == min == max for integer counters, derived from §3.2 cross-N Δ=0). Defensible per spec L67 "median + min + max statistics across the 3 reps".
- `rivqdown SHA12` and `wall.sec` per-cell columns are populated with prose annotations `(see §4 t_wall_total)` / `(Plan A: no per-cell SHA in §3.1)` rather than numeric values, because §3.1 source is CVODE-counter-only. Spec L64 lists these columns; the prose annotations + the §raw-18-cell-table follow-on wall-sec table at L203-210 satisfy the intent while being honest about Plan A's reuse-without-rerun stance. Plan B would populate numerically. No spec violation.
- PR diff includes 1 bonus PR-0 deferred fix (capstone.md §5.1 `4.527 → 4.526` ratio cell + provenance HTML comment + ROI sentence update); explicitly noted in PR body as bundled per PR-0 review-correctness deferred note. Diff = 10 insertions / 3 deletions in capstone.md, scope-aligned.

Per-check assessment:
1. Task DONE/MISSING:                       pass
2. Spec scenario coverage:                  pass — all 4 Requirements + 8 Scenarios (Plan A verdict-doc-extraction, Plan A codepath-equivalence + Plan B template safety net, codepath-divergence escalation [vacuously satisfied, Plan A passed], Plan B submit-template-adaptation [SKIPPED per spec L17], keliya-smoke artifact generation, 18-cell raw table, cross-N invariance, ROI ratio, solver-failure, decision-input) are covered with cited evidence.
3. Issue #364 acceptance:                   pass — all 8 acceptance bullets (5 tables present, codepath-equivalence section, keliya-smoke section, submit-template-provenance, aggregator script, decision-input with hard-evidence trigger satisfied, ROI saturation_ratio column, solver-failure floors with "NOT 6/47" note, cross-N invariance) verified.
4. Scope creep audit:                       pass — diff = 3 tracked files (1 new doc + 1 new tool + 1 capstone fix). Zero source/test/Makefile/SHUD-pointer change. Capstone §5.1 fix legitimately bundled per PR-0 deferred note.
5. Downstream PR dependency setup:          pass — PR-C G3 4-way CI gate anchor at §keliya-smoke-anchor (`1bfe6a30856e` + 15-key snapshot with explicit contract L363); PR-D 60-cell sweep decision-input at §decision-input-table (verdict input "hard-evidence satisfied → full 60-cell sweep" at L426-428); PR-E mode-C-tune reference at §mode-C-tune-reference cites baseline.
6. Cross-PR sequencing:                     pass — PR-A correctly uses corrected `7/51` floors (PR-0 dependency); old `6/47` floors only appear as negative-control anchor citations (L313-316 + aggregator T4 prose); decision-input table provided for PR-B; keliya smoke provided for PR-C.
7. openspec validate strict:                pass — `openspec validate p8tune-spgmr-maxl --strict` → "Change 'p8tune-spgmr-maxl' is valid".
8. Test/evidence coverage:                  pass — aggregator script exit 0 + 5 tables emitted (verified locally); baseline doc 429 lines / 9 sections present; keliya smoke artifact + job stdout files present (`.review-evidence/p8tune-pr-a/`); codepath diff evidence file present (2358 bytes).
