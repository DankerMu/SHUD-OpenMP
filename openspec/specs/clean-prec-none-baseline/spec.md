# clean-prec-none-baseline Specification

## Purpose
TBD - created by archiving change p8tune-spgmr-maxl. Update Purpose after archive.
## Requirements
### Requirement: Plan A artifact reuse path

The system SHALL first attempt to reuse Step 1 PR-B verdict aggregator output from `docs/p8pre/n8_profile_verdict.md` §3.1, on the basis that SHUD `37be0fe` and SHUD `7a1dc8f` (Step 1 pin) produce bit-identical CVODE codepath behavior. The verdict §3.1 table already includes all 15 canonical counter keys per `tools/cvode_stats_diff/canonical_15_keys.yaml`.

#### Scenario: Plan A verdict-doc-extraction path
- WHEN orchestrator runs the Plan A extraction
- THEN it SHALL read `docs/p8pre/n8_profile_verdict.md` §3.1 (median per (case, N) 15-canonical-key table at L70-77)
- AND verify the table columns include the full 15 canonical keys per `tools/cvode_stats_diff/canonical_15_keys.yaml`: `nfe, nfeLS, nni, nli, nsetups, netf, nst, npe, nps, ncfn, ncfl, lenrw, leniw, lenrwLS, leniwLS`
- AND verify the expected cleaned-PREC_NONE floors are present: heihe `ncfn=7, ncfl=85, netf=0`; heihe_x4 `ncfn=51, ncfl=3620, netf=0`
- AND if all checks pass, it SHALL proceed to Plan A codepath-equivalence verification
- AND if any check fails (e.g., §3.1 table format changed), it SHALL escalate to Plan B fallback

#### Scenario: Plan A CVODE codepath equivalence verification
- WHEN orchestrator confirms §3.1 table contents
- THEN orchestrator SHALL document the CVODE codepath equivalence claim by running `git diff 7a1dc8f..37be0fe -- src/Equations/cvode_config.cpp src/Equations/*precond* src/Equations/*spgmr*` inside SHUD submodule
- AND the diff SHALL show only revert-of-PR-D modifications (`cvode_config.cpp:259` PREC_LEFT → PREC_NONE + removal of identity preconditioner registration + deletion of MD_precond_identity files)
- AND no other CVODE-affecting source change SHALL be present
- AND this equivalence proof SHALL be embedded in the baseline doc `docs/p8tune/clean_prec_none_baseline.md` §codepath-equivalence
- AND orchestrator SHALL ALSO verify Plan B template existence at `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/submit_p1e_baseline_template.sbatch` as fallback safety net; if template missing, baseline doc §submit-template-provenance SHALL document the alternative derivation from `tools/run_omp.sh` + `case_deployment_map.md`

#### Scenario: Plan A codepath divergence triggers Plan B escalation
- WHEN `git diff 7a1dc8f..37be0fe -- src/Equations/cvode_config.cpp src/Equations/*precond* src/Equations/*spgmr*` shows MORE than the revert-of-PR-D set
- THEN orchestrator SHALL abort Plan A
- AND it SHALL escalate to Plan B fallback with explicit rationale embedded in baseline doc §plan-b-escalation
- AND if Plan B is later run AND the resulting `cvode_stats.txt` counter values differ from the Plan A §3.1 extracted values at greater than 2-ULP per any 15-canonical-key field on the heihe N=8 rep1 cell, orchestrator SHALL flag this as an unresolved-bug investigation finding and pause the change pipeline

### Requirement: Plan B re-run fallback

The system SHALL provide an 18-cell server re-run fallback path when Plan A codepath-equivalence fails, using the Step 1 PR-A submit template adapted for SHUD pin `37be0fe`.

#### Scenario: Plan B submit template adaptation
- WHEN Plan A codepath-equivalence fails (diff shows more than revert-of-PR-D)
- THEN orchestrator SHALL copy `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/submit_p1e_baseline_template.sbatch` to `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/clean_prec_none_baseline/submit_clean_baseline_template.sbatch`
- AND if the Plan B template is missing at the source path, orchestrator SHALL derive a fresh template from `tools/run_omp.sh` + `case_deployment_map.md` heihe_x4 server entry, AND document the derivation provenance in baseline doc §submit-template-provenance
- AND it SHALL update `--output / --error` paths to `/scratch/.../p8tune-runs/clean_prec_none_baseline/cn_*.{out,err}` (per Slurm 三铁律: paths under `/scratch`, not `/tmp` or compute-node-local)
- AND it SHALL set SHUD pin = `37be0fe` (forward-only descendant of outer pointer at `e442ce8`)
- AND it SHALL set `SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` build flags
- AND it SHALL run 18 cells: `heihe / heihe_x4` × `N ∈ {1,4,8}` × `rep ∈ {1,2,3}`
- AND each cell SHALL emit `profile_B0.yaml`, `cvode_stats.txt` (with full 15 canonical keys), `rivqdown.dat`, `wall.sec`, `cell.meta`, stdout/stderr capture

### Requirement: keliya cleaned-PREC_NONE smoke anchor

The system SHALL produce a single keliya cleaned-PREC_NONE smoke artifact on SHUD `37be0fe` to serve as the bit-identical anchor for `spgmr-maxl-env-hook` G3 default-compat gate.

#### Scenario: keliya smoke artifact generation
- WHEN PR-A baseline doc is being authored
- THEN orchestrator SHALL build `shud` on SHUD `37be0fe` (`make shud SHUD_ENABLE_PROFILE=1`) and run a single keliya smoke (`./shud keliya`) with `SHUD_SPGMR_MAXL` unset
- AND it SHALL capture `rivqdown.dat` SHA12 + `cvode_stats.txt` 15-key snapshot
- AND embed these into `docs/p8tune/clean_prec_none_baseline.md` §keliya-smoke-anchor as the bit-identical reference for `spgmr-maxl-env-hook` G3 default-compat gate
- AND the smoke SHALL run on the server (Linux cn14/cn15) using the same toolchain as PR-A baseline (server `gcc 13.3.0-6ubuntu2~24.04.1 + libgomp + libsundials_cvode.so.6`)

### Requirement: 18-cell aggregator and baseline doc

The system SHALL produce a baseline document `docs/p8tune/clean_prec_none_baseline.md` containing 5 result tables that fully characterize the cleaned `PREC_NONE` reference set.

#### Scenario: 18-cell raw table
- WHEN aggregator processes 18 cells (Plan A: extracted from `docs/p8pre/n8_profile_verdict.md` §3.1; Plan B: from Plan B re-run output)
- THEN it SHALL emit a raw table with rows = 18 cells (case + N + rep), columns = (`wall.sec`, `rivqdown SHA12`, AND the 15 canonical CVODE counter keys per `tools/cvode_stats_diff/canonical_15_keys.yaml`: `nfe, nfeLS, nni, nli, nsetups, netf, nst, npe, nps, ncfn, ncfl, lenrw, leniw, lenrwLS, leniwLS`)
- AND the columns SHALL NOT include `njvtimes` or `ncfn_total` (these are not part of the 15-key canonical contract)
- AND each numeric value SHALL be cited from the source (`n8_profile_verdict.md` §3.1 for Plan A; per-cell `cvode_stats.txt` for Plan B)
- AND the table SHALL use median + min + max statistics across the 3 reps per (case, N)

#### Scenario: Cross-N invariance table
- WHEN baseline doc generated
- THEN it SHALL emit a cross-N invariance table verifying that within a single case, the 15 canonical CVODE counters are identical across N=1/4/8 (bitwise OMP-neutrality, per B1a §S4 PASS); cross-N invariance is a B1a S4 OMP-neutrality regression detector AND independent of downstream sweep matrix dimension
- AND counter divergence across N=1, N=4, or N=8 within same case SHALL be flagged as P0 baseline issue
- AND N=4 baseline cells are retained for OMP-neutrality regression coverage even though the downstream sweep matrix omits N=4 (sweep uses N ∈ {1, 8} only; rationale: 60 cells already saturate the maxl ROI signal at the boundary, N=4 adds compute without changing the gate verdict)

#### Scenario: ROI ratio table
- WHEN baseline doc generated
- THEN it SHALL emit a ROI table reporting per (case, N=8) the median `nfeLS/nfe` and `nli/nni` ratios
- AND it SHALL include a `saturation_ratio = (nli/nni) / 5` column
- AND it SHALL flag rows where `saturation_ratio ≥ 0.8` (Krylov subspace approaching default cap)
- AND it SHALL document that authoritative source for heihe_x4 `nfeLS = 30509` is `n8_profile_verdict.md` §3.1 (NOT the typo values `30518` in ADR-0003 L22 / glossary L271, or `30517` in capstone §5.1 L161 — those are corrected via `p8pre-doc-state-correction`)

#### Scenario: Solver-failure counter table
- WHEN baseline doc generated
- THEN it SHALL emit a solver-failure table with rows = (case, N), columns = (median `ncfn`, median `ncfl`, median `netf`) for BOTH N=1 AND N=8 (to provide the gate anchor for `maxl-sweep-verdict` G6 hard-gate per sweep N ∈ {1,8})
- AND it SHALL explicitly enumerate the cleaned-PREC_NONE floors: heihe `ncfn=7, ncfl=85, netf=0`; heihe_x4 `ncfn=51, ncfl=3620, netf=0`
- AND it SHALL state: "these are the true production floor anchors per Step 1 PR-B verdict §3.1, NOT the `PREC_LEFT + identity` floors 6/47 from Step 2 PR-F"

#### Scenario: Decision input table for maxl sweep
- WHEN baseline doc generated
- THEN it SHALL emit a decision-input table indicating: (a) hard-evidence trigger `ncfl > 0 per cell` is ALREADY SATISFIED — heihe `ncfl=85` (>0) AND heihe_x4 `ncfl=3620` (>0) per §3.1; (b) heihe_x4 `nli/nni = 4.527` (at saturation threshold per §3.4); (c) heihe `nli/nni = 1.820` (below saturation, used only for case-asymmetric ROI documentation per design D13)
- AND the table SHALL pre-compute the verdict input as "hard-evidence satisfied → full 60-cell sweep" per maxl-sweep-verdict Requirement "Sweep entry condition"

