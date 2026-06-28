# p8pre-doc-state-correction Specification

## Purpose
TBD - created by archiving change p8tune-spgmr-maxl. Update Purpose after archive.
## Requirements
### Requirement: Future-candidate gate correction across 4 docs

The system SHALL replace `ncfn < 6` / `ncfn < 47` future-candidate gate wording with the corrected formulation across 4 merged p8pre-spike documents.

#### Scenario: ADR-0003 §Consequences correction (L70)
- WHEN PR-0 is applied
- THEN `docs/adr/0003-precond-spike-decision.md` §Consequences §Positive bullet at L70 (beginning "**ROI ceiling 实测落地**") SHALL no longer contain wording asserting `ncfn < 6 ∧ ncfn < 47` as future-candidate PASS criterion
- AND the corrected wording SHALL state: "Future preconditioner / solver-tune candidates must satisfy `ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)` (i.e., `≤ ncfn_PREC_NONE_cleaned_baseline` per `docs/p8pre/n8_profile_verdict.md` §3.1) for each case, while reducing `nfeLS/nfe`, `nli/nni`, or wall by the declared threshold. The identity floor 6/47 is retained ONLY as a `PREC_LEFT + identity` negative-control anchor, NOT as a production gate."
- AND a cross-ref to capability `clean-prec-none-baseline` SHALL be added to clarify where the cleaned baseline `ncfn` is established

#### Scenario: identity_spike_verdict.md §6.2 PASS criterion correction (L169)
- WHEN PR-0 is applied
- THEN `docs/p8pre/identity_spike_verdict.md` §6.2 final paragraph beginning "PASS criterion for any future P8-precond candidate" at L169 SHALL no longer contain `ncfn < 6 (heihe) and ncfn < 47 (heihe_x4)` wording
- AND the corrected wording SHALL state: "PASS criterion for any future P8-precond candidate: `ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)` (per `docs/p8pre/n8_profile_verdict.md` §3.1 cleaned-PREC_NONE baseline) with combined setup-plus-solve overhead within 10% of identity-baseline wall. These 6/47 are deterministic floors observed for `PREC_LEFT + identity`, a negative-control architecture, NOT for the production `PREC_NONE` codepath."

#### Scenario: capstone.md §6.4.3 + §Future Work corrections
- WHEN PR-0 is applied
- THEN `docs/p8pre/capstone.md` §6.4.3 paragraph at L234-238 (PASS criterion for future P8-precond candidate) SHALL no longer contain `ncfn < 6 (heihe)` AND `ncfn < 47 (heihe_x4)` wording
- AND the corrected wording at L236 SHALL state: "`ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)` per `docs/p8pre/n8_profile_verdict.md` §3.1 cleaned-PREC_NONE baseline (压 deterministic floor 下来)"
- AND the capstone §Future Work bullet at L313 ("**Prerequisite 1**: real preconditioner candidate ... `ncfn < 6 (heihe)` AND `ncfn < 47 (heihe_x4)`") SHALL be corrected to "`ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)`"
- AND the capstone SHALL include a forward reference to ADR-0004 (TBD) as the natural successor decision

#### Scenario: p8pre_summary.md forward note (no inline ncfn<6/47 wording present)
- WHEN PR-0 is applied
- THEN orchestrator SHALL grep `docs/p8pre_summary.md` for `ncfn < 6` / `ncfn < 47` / `ncfn<6` / `ncfn<47` (verified empty at change-authoring time)
- AND IF such wording exists, it SHALL be corrected per the same `ncfn_candidate ≤ 7 ∧ ncfn_candidate ≤ 51` formulation
- AND IF no such wording exists (current state), orchestrator SHALL add a §Forward note in `docs/p8pre_summary.md` cross-referring to p8tune-spgmr-maxl change + ADR-0004 (TBD) per the corrected future-candidate gate

### Requirement: nfeLS typo correction across 3 docs

The system SHALL correct the `heihe_x4 nfeLS` typos in 3 docs to the authoritative value `30509` from `docs/p8pre/n8_profile_verdict.md` §3.1 (Step 1 PR-B canonical aggregator).

#### Scenario: ADR-0003 L22 nfeLS typo correction
- WHEN PR-0 is applied
- THEN `docs/adr/0003-precond-spike-decision.md` L22 SHALL change `heihe_x4 N=8: 30518 / 6741 = 4.526` to `heihe_x4 N=8: 30509 / 6741 = 4.526`
- AND the corrected line SHALL cross-ref `n8_profile_verdict.md` §3.1 as authoritative source

#### Scenario: glossary L271 nfeLS typo correction
- WHEN PR-0 is applied
- THEN `openspec/glossary.md` L271 (`nfeLS/nfe ratio` entry) SHALL change `r = 30518/6741 = 4.526` to `r = 30509/6741 = 4.526`
- AND the corrected line SHALL cross-ref `n8_profile_verdict.md` §3.1 as authoritative source

#### Scenario: capstone §5.1 L161 nfeLS typo correction
- WHEN PR-0 is applied
- THEN `docs/p8pre/capstone.md` §5.1 table at L161 SHALL change `heihe_x4 | 1 | 1412.895 | 6575 | 6741 | 30517 | 4.527` to `heihe_x4 | 1 | 1412.895 | 6575 | 6741 | 30509 | 4.527`
- AND any other `30517` or `30518` references for heihe_x4 N=8 nfeLS in capstone SHALL be corrected to `30509`

### Requirement: Cleanup-deferred wording update

The system SHALL update cleanup-status wording across 3 docs to reflect completion at outer `e442ce8` / SHUD `37be0fe`.

#### Scenario: ADR-0003 cleanup status update
- WHEN PR-0 is applied
- THEN `docs/adr/0003-precond-spike-decision.md` §Decision §2 (D8 cleanup status) SHALL change from "fall-back PREC_NONE pending cleanup" wording to "fall-back PREC_NONE completed at outer `e442ce8` / SHUD `37be0fe` (cleanup pointer bump merged to `main` on 2026-06-27)"
- AND the merged-commit SHA SHALL be cited explicitly (outer `e442ce8`, SHUD `37be0fe`)
- AND the cleanup deletions (revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE; delete `MD_precond_identity.{h,cpp}`; remove `CVodeSetPreconditioner / CVodeSetLSetupFrequency` registrations) SHALL be enumerated for audit trail

#### Scenario: identity_spike_verdict.md cleanup status update
- WHEN PR-0 is applied
- THEN `docs/p8pre/identity_spike_verdict.md` §Forward action / §Cleanup SHALL include the same completion timestamp wording
- AND the doc SHALL note the spike artifact's archival location for future reference

#### Scenario: capstone.md cleanup status update
- WHEN PR-0 is applied
- THEN `docs/p8pre/capstone.md` §7.x (Future Work / Status) SHALL include "Design D8 cleanup completed at outer `e442ce8` / SHUD `37be0fe` (2026-06-27)" wording
- AND a note SHALL acknowledge: "P8-precond and P8-tune.C are NOT permanently rejected — Path 3 remains open for future restart per design pivot; see capability `clean-prec-none-baseline` + `maxl-sweep-verdict`"

### Requirement: Glossary mode-C-tune term and Master plan section

The system SHALL introduce `mode-C-tune` reference-set term in the glossary, plus a master plan §P8-tune.C section linking the corrected gate and the new reference set.

#### Scenario: Glossary mode-C-tune term added
- WHEN PR-0 is applied
- THEN `openspec/glossary.md` SHALL gain a new term `mode-C-tune` defined as: "Reference-set design pattern for SHUD CVODE solver-tune campaigns where Mode C strict-omp A3a SHA bitwise gate does not apply because the tune parameter (e.g., SPGMR `maxl`, `MaxNonlinIters`) directly affects reduction order. Per (case, maxl) tuple, `rivqdown.dat` SHA12 + `cvode_stats.txt` 15-key set (per `tools/cvode_stats_diff/canonical_15_keys.yaml`) form a separate reference anchor. A3a strict SHA verification does NOT apply across different maxl values. A4 max_ulp tolerance (≤1024 default; sweep-specific tolerance per maxl) applies BOTH between mode-C-tune(maxl=N) references AND between cross-N (N=1/4/8) within the same (case, maxl) tuple. Hydrology validation uses water-balance + hydrology-set indicator tolerance per G7 (currently A4 max_ulp fallback only; water-balance threshold TBD). Precedent: capability `maxl-sweep-verdict` (this change); future use: P8-tune.A/B/D + P8-precond reference design."
- AND a cross-ref entry SHALL be added: "see also: p8tune-spgmr-maxl change; ADR-0004 (TBD)"

#### Scenario: SHUD_SPGMR_MAXL glossary entry
- WHEN PR-0 is applied
- THEN `openspec/glossary.md` SHALL gain a new term `SHUD_SPGMR_MAXL` defined as: "Runtime env var introduced by capability `spgmr-maxl-env-hook` controlling SPGMR Krylov subspace dimension. Values: unset = SUNDIALS default (5); explicit `{5, 10, 15, 20, 30}` per maxl sweep matrix. NEVER opens PREC_LEFT, NEVER registers preconditioner."

#### Scenario: Master plan §P8-tune.C section added
- WHEN PR-0 is applied
- THEN `SHUD_openMP_master_plan.md` SHALL gain a new section §P8-tune.C (positioned after §P8-precond.0) documenting: epic scope (4 capabilities of this change), entry condition (cleaned `PREC_NONE` baseline anchors), 6-PR sequence (PR-0/A/B/C/D/E), 8-gate verdict (G1-G8 with band descriptions), ADR-0004 decision branches, forward path to KLU pattern-only if NO-GO

### Requirement: PR-0 doc-only no-code constraint

The PR-0 (doc state correction) SHALL not modify any code, build, or runtime artifact.

#### Scenario: PR-0 scope verification
- WHEN PR-0 is reviewed pre-merge
- THEN `git diff <merge_base>..HEAD --name-only` SHALL list only files matching `docs/**/*.md`, `openspec/glossary.md`, `SHUD_openMP_master_plan.md`
- AND NO files matching `**/*.{c,cpp,h,hpp,Makefile,*.sh}` SHALL appear
- AND `git submodule status SHUD` SHALL show no SHUD pointer change relative to outer main `e442ce8`
- AND no `.gitignore`, `.gitmodules`, build script, or CI workflow SHALL be modified

