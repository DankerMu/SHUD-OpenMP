## Phase 4 Cross-Review Evidence Bundle (round 1)

Reviewer agents: `review-correctness`, `review-integration`
Review round: round 1
Reviewed head SHA: `20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d`
Local evidence: `.review-evidence/p8pre-pr-a-prep/{correctness,integration}.md`

### review-correctness — APPROVE

Summary: Template + wrapper + evidence doc are correct against design D4/D5/D7 and Slurm 三铁律; all 14 checklist items pass with one minor evidence-doc precision gap. No SHUD changes.

Findings:
- **Suggestion** (non-blocking): `docs/p8pre/pr_a_prep_evidence.md:60` — heihe_x4 forcing/ SHALL row reads "286M (per map §2.2)" — should be "≥ 200M (286M documented in map §2.2)" so future re-pack at e.g. 290M doesn't false-fail. Does not affect PR-A handoff.

Non-blocking notes:
- All 14 checklist items pass (substitution markers clean / Slurm 三铁律 met / determinism env exact / N=2 absent / cn14+cn15 pin correct / nm gates 1/0/1 / SHUD pin 7a1dc8f / case basin sizes / singleton chain wiring / no SHUD source change / POSIX bash+awk only)

### review-integration — APPROVE

Summary: Integration surfaces all clean — output path schema, per-cell artifact set, build-flag gating, CI, gitignore, render dry-run all align with downstream PR-B/C/F contracts.

Findings: **None.**

Non-blocking notes (carry-forward to #341 PR-A run):
1. **`.gitignore` doesn't auto-cover `.p8pre-runs/`** — pattern `.s*-runs/` + explicit `.p1*-runs/` enumerations don't catch `.p8pre-runs/`. Recommend folding `.p8pre-runs/` + `.p8pre-pr-*-runs/` entries into outer `.gitignore` in #341 before first server submission. Not blocking because Mac dry-run doesn't materialize the dir.
2. **JID placeholder substitution target clarification for #341 runner**: `__PREV_JID_<case>_N<n>_rep<r-1>__` placeholders appear in stdout sbatch invocation LINES (12 dependent reps), NOT in rendered `.sbatch` file bodies. Runner doc should call this out.

### Phase 4.5 Verifier — SKIPPED

Rationale: 0 candidates with concrete failure scenarios + concrete merge-blocking status. Per `compact` precision-bias (only CONFIRMED blocks merge), 1 Suggestion + 2 non-blocking notes do not require independent verification. Both notes carry-forward to #341.

### Round 1 verdict

**Clean.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE. Proceed to Phase 7.
