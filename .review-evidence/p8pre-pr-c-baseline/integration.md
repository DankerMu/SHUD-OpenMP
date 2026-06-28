Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 43bd6f2a5c76eda2b9bbc577a6121020bae12b71
Summary: Integration contract intact and downstream-consumable; one cross-doc section-number drift (spec/tasks say §3, baseline doc puts table at §5.1) worth follow-up alignment but not blocking — anchor name + cell labels + values are unambiguous.

Findings:

1. Yellow / Warning — Section-number drift spec↔doc
   - `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md:108,113` + `tasks.md:41,77,81` + `specs/p8precond-zero-identity-spike/spec.md:90` all point at "§3 raw data table" of `n8_profile_baseline.md` for `wall_step1_baseline_median(case, N)`.
   - Actual location is **§5.1** (`n8_profile_baseline.md:129-153`) — PR-C adopted academic mother-template ordering (§3 Methodology / §5 Results) per `docs/p1e/p1e_academic_summary.md`.
   - Impact: PR-F (#347) / ADR-0003 / openspec-archive readers grepping "§3" miss-locate. Baseline doc is internally self-consistent (§5.1 / §8 / §9 all match). Spec/tasks now lie about the location.
   - Fix: bump spec L108/L113 + tasks §4.1/§8.1/§8.5 + p8precond-zero-identity-spike L90 from "§3 raw data table" → "§5.1 Raw data table". Foldable into PR-F or PR-G.

2. Green / Praise — `wall_step1_baseline_median(case, N)` API contract delivered exactly as PR-F needs
   - `n8_profile_baseline.md:135-142` Table 1: 6 rows × `(case, N, n_reps, wall_median, nst_median, nfe_median)`; case labels `heihe / heihe_x4`, N `1 / 4 / 8` — matches PR-F ε bounds (`ε(heihe)=0.10`, `ε(heihe_x4)=0.05`) from tasks §8.5.
   - `:148-153` redundant citation form `wall_median(heihe, N=1) = 140.797 s` — machine-greppable.

3. Green / Praise — 10/10 invariance + 4/4 absolute anchor PASS-tagged for ADR-0003
   - §5.2 Table 2 (10 rows verdict=PASS) + §5.3 Table 3 (4 rows verdict=PASS) + §6.1 Table 5 hypothesis summary "All PASS". PR-G grep-able verbatim.

4. Green / Praise — Cross-doc number consistency
   - baseline §5.1 / verdict (§3+§4) / run (§5) checked: nst/nfe/nfeLS/nni/nli + ratios 1.819/4.526 + per-rep wall all internally consistent. No silent drift across the 3 p8pre docs.

5. Green / Praise — Hypothesis labels match mother template
   - `n8_profile_baseline.md:35-37` uses `H1 / H2 / H3` (NOT H.1/H.2/H.3) — exact same form as `docs/p1e/p1e_academic_summary.md:41-43`.

6. Green / Praise — YAML metadata schema compat
   - `:1-17` title/date(`2026-06-27`)/version `1.0`/status/`related_docs` — same shape as p1e mother template `:1-21`. OpenSpec archive (#349) should parse without bespoke handling.

7. Green / Praise — `case_deployment_map §5.1` schema continuity
   - 18-row table (`case_deployment_map.md:115-132`) uses `| Case | N | rep | SHUD pin | wall (s) | yaml path |` — same column count + style as parent §5 SHIP set (`:99-106`). No drift; existing rows preserved.

8. Green / Praise — Master plan §P8-precond.0 prep position
   - `SHUD_openMP_master_plan.md:2356-2366` newly inserted §P8-precond.0 prep sits BEFORE §P8-precond.1 (`:2367`). Cross-refs PR-A/B/C docs + branch verdict + Step 2 entry; in scope per issue body.

Non-blocking notes:
- `openspec validate p8pre-spike --strict` PASS (re-verified locally).
- No `.github/workflows/` touched (`git diff main...HEAD -- .github/workflows/` empty).
- No spec deletion / weakening in `openspec/changes/p8pre-spike/` (diff empty — orchestrator zone respected).
- N=1 wall per-rep spread (heihe 12.0%, heihe_x4 14.6%) exceeds gate-4 ε(heihe)=10%; PR-F #347 strictly uses N=8 + ε on absolute median so this is documented self-aware (§5.1 + §7.4), not contract bug. ADR-0003 should acknowledge gate-4 strength is N=8 dominated.

Verdict: APPROVE — Finding 1 (spec/tasks section-number drift) is doc-only follow-up, foldable into PR-F or PR-G without blocking PR-C merge. All 8 integration checklist items verified.
