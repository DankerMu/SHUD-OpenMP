## Phase 4 Cross-Review Evidence Bundle (rounds 1 + 2)

Reviewer agents: `review-correctness`, `review-integration`
Review rounds: round 1 (initial) + round 2 (post Phase 6 fix integration)
Reviewed head SHA: `43bd6f2` (tracked; Phase 6 fix is openspec/changes/ working-tree-only per `.gitignore:13`)
Local evidence: `.review-evidence/p8pre-pr-c-baseline/{correctness,integration,integration-round-2}.md`

### Round 1 (head `43bd6f2`)

#### review-correctness — APPROVE

Summary: All 12 correctness checks PASS — wall medians, ROI ratios, anchors, branch verdict, ADR-0003 omission all verified. No blocking findings.

Findings: **None.**

Verification:
- Wall medians sorted-middle from `/tmp/p8pre_n8_profile/<cell>/profile_B0.yaml extras.t_wall_total`: heihe 140.797/95.734/89.732 + heihe_x4 1412.895/849.704/743.552. All 6 match §5.1 Table 1 to 3 decimal places.
- Cross-N invariance §5.2: 10/10 PASS verbatim from PR-B verdict §3.2.
- Absolute anchor §5.3: heihe nst=6698/nfe=6943, heihe_x4 nst=6575/nfe=6741 — match step1_prep.md §3-§4.
- ROI §5.4: r_min=1.819, r_max=4.526 — match PR-B §3.4.
- Branch verdict §5.5: a (PROCEED) — matches PR-B §3.5.
- ADR-0003 omission: `docs/adr/` contains only 0001 + 0002 — correct per branch-a logic.
- Master plan §P8-precond.0 prep at L2356 (before §P8-precond.1) — order correct.
- §6.3 wall scaling: 140.797/89.732 = 1.569× and 1412.895/743.552 = 1.901× (heihe N=8 vs N=1; heihe_x4 N=8 vs N=1).

Non-blocking notes:
- baseline.md front-matter `date: 2026-06-27` while local time was 2026-06-26 — consistent with §4 submit window "2026-06-27 01:32:52 → 02:43:35 UTC" (UTC midnight crossed).
- §5.1 nested table schema (Case, N, rep, SHUD pin, wall, yaml) differs in column names from parent case_deployment_map §5 (Case, Platform, SHUD pin, wall, yaml, 状态). Semantic divergence justified (per-rep raw data vs per-case canonical).

#### review-integration — APPROVE w/ 1 Warning (Phase 4.5 CONFIRMED → Phase 6 fix → round 2 RESOLVED)

Summary round 1: Integration contract intact and downstream-consumable; one cross-doc section-number drift (spec/tasks say §3, baseline doc puts table at §5.1) worth fixing.

Findings:
- **Warning cand-01**: `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` L17 + L113, `tasks.md` L41 + L77, `specs/p8precond-zero-identity-spike/spec.md` L90, `design.md` L26 + L107 cite "§3 raw data table" of `n8_profile_baseline.md`, but the actual `wall_step1_baseline_median(case, N)` Table 1 lives at §5.1. Doc internally consistent; spec/tasks/design mis-label the location. PR-F #347 implementer reading verbatim would grep §3, find Methodology, miss Table 1.

Non-blocking notes:
- Praise: API contract (6-row Table 1), 10/10+4/4 PASS verdicts, cross-doc number consistency, H1/H2/H3 hypothesis labels matching mother template, YAML metadata schema compat, master plan §P8-precond.0 position before §P8-precond.1.
- `openspec validate p8pre-spike --strict` PASS.
- N=1 wall per-rep spread documented self-aware (§5.1 + §7.4); ADR-0003 should note gate-4 strength is N=8 dominated.

### Phase 4.5 Verifier (round 1 candidate)

| candidate | verdict | rationale |
|---|---|---|
| cand-01 §3 vs §5.1 anchor drift | **CONFIRMED** | drift real + scenario realistic (PR-F implementer reads verbatim from working tree per .gitignore:13) + fix in-scope (5-line sed, gitignored openspec/changes/, mirrors PR #353 cand-02 precedent) |

### Phase 6 fix (orchestrator-direct, working-tree-only)

- 5 §3 → §5.1 substitutions across 4 files (gitignored openspec/changes/p8pre-spike/):
  - `tasks.md` L41 (section enumeration update to academic style)
  - `tasks.md` L77 (gate-4 baseline source ref)
  - `specs/n8-mode-c-profile-recheck/spec.md` L17 (baseline location ref)
  - `specs/n8-mode-c-profile-recheck/spec.md` L113 (section enumeration update to academic)
  - `specs/p8precond-zero-identity-spike/spec.md` L90 (Step 2 gate 4 source)
  - `design.md` L26 (Step 2 build-matrix comparison)
  - `design.md` L107 (gate-4 baseline source)
- Phase 6 fix is working-tree-only per `.gitignore:13` "OpenSpec transient changes" policy — tracked head SHA `43bd6f2` UNCHANGED. Fix archives to persistent `openspec/specs/` at #349. PR #353 cand-02 precedent confirms this is project convention.
- `openspec validate p8pre-spike --strict --no-interactive` exit 0 post-fix.

### Round 2 integration re-review (head `43bd6f2`, tracked unchanged)

#### review-integration — APPROVE (post-fix)

Summary: Phase 6 fix correctly resolves cand-01; all 7 anchor citations now point to §5.1 (academic-paper-style Results); strict validation green; no new findings.

Findings: **None.**

Resolution status:
- cand-01: **RESOLVED**
  - 0 stale `§3\b` refs to n8_profile_baseline.md across openspec/changes/
  - ≥ 5 grep-line hits for `§5.1` across 4 files covering all 7 cited locations
  - `openspec validate p8pre-spike --strict` exit 0
  - Baseline doc §5.1 + Table 1 intact at lines 129/133 (no regression in tracked doc)
  - PR-F #347 consumer loop closed (verbatim read now finds correct anchor)
  - Out-of-scope L84/L122 `identity_spike_verdict.md` refs correctly untouched (PR-F's own deliverable doc structure)
  - Archive at #349 will migrate fix verbatim into persistent `openspec/specs/`

### Cumulative verdict

**Clean.** Round 2 closes the loop on round-1 Warning. 0 outstanding findings. Proceed to Phase 7.
