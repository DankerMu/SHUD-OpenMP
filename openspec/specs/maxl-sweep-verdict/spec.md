# maxl-sweep-verdict Specification

## Purpose
TBD - created by archiving change p8tune-spgmr-maxl. Update Purpose after archive.
## Requirements
### Requirement: Sweep entry condition (probe vs full sweep)

The system SHALL gate maxl sweep run mode (probe-only 12-cell vs full 60-cell) on the baseline doc's decision-input table from capability `clean-prec-none-baseline`.

#### Scenario: Full sweep GO triggered by hard evidence
- WHEN the cleaned `PREC_NONE` baseline decision-input table reports `ncfl > 0` in ANY (case, N) cell (per Step 1 verdict §3.1: heihe `ncfl=85`, heihe_x4 `ncfl=3620` — this condition is already satisfied with current case set)
- THEN orchestrator SHALL execute the full 60-cell sweep
- AND output path SHALL be `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/`

#### Scenario: Probe-only mode (future-case fallback)
- WHEN the cleaned `PREC_NONE` baseline shows `ncfl = 0` in ALL cells AND heihe_x4 median `nli/nni ≥ 4.5`
- THEN orchestrator SHALL execute a probe-only 12-cell sweep first: `maxl ∈ {5, 10}` × `case ∈ {heihe, heihe_x4}` × `N=8` × `rep ∈ {1,2,3}`
- AND IF probe shows ≥5% wall reduction OR `nli` reduction ≥10% AT maxl=10 vs maxl=5 on heihe_x4
- THEN orchestrator SHALL escalate to full 60-cell sweep
- ELSE orchestrator SHALL close capability `maxl-sweep-verdict` with verdict "no sweep benefit; transition to KLU pattern-only"
- AND this branch is retained as a future-case fallback structure; with the current heihe + heihe_x4 case set both `ncfl > 0` per `n8_profile_verdict.md` §3.1, so this branch is NOT exercised in this change

#### Scenario: NO-GO with no entry conditions met
- WHEN the cleaned `PREC_NONE` baseline shows `ncfl = 0` in ALL cells AND heihe_x4 median `nli/nni < 4.5`
- THEN orchestrator SHALL skip the sweep
- AND ADR-0004 SHALL be authored with verdict "maxl sweep NO-GO; cleaned `PREC_NONE` baseline already optimal for current case set"
- AND transition recommendation SHALL be "consider P8-tune.D KLU pattern-only spike per ADR-0003 §Forward action §3.4"
- AND this branch is also a future-case fallback structure (current heihe + heihe_x4 data triggers full sweep)

#### Scenario: Decision tree completeness — residual case explicit fallback
- WHEN the cleaned `PREC_NONE` baseline shows any state NOT covered by the 3 scenarios above (e.g., `ncfl = 0` in all cells AND `nli/nni < 4.5` heihe_x4 but ≥ 4.0 heihe, or any other residual combination)
- THEN orchestrator SHALL default to the NO-GO branch with verdict "maxl sweep NO-GO; insufficient evidence; cleaned `PREC_NONE` baseline already optimal for current case set"
- AND ADR-0004 SHALL explicitly enumerate the observed (case, N) state that triggered this residual fallback
- AND this default-NO-GO branch ensures the entry-condition decision tree is total over all input states (no undefined orchestrator behavior)

### Requirement: 60-cell sweep matrix execution

The system SHALL execute a 60-cell server matrix using a sbatch submit template derived from the Step 1 PR-A pattern.

#### Scenario: Sweep matrix structure
- WHEN full sweep is triggered
- THEN the run matrix SHALL be `maxl ∈ {5, 10, 15, 20, 30}` × `case ∈ {heihe, heihe_x4}` × `N ∈ {1, 8}` × `rep ∈ {1, 2, 3}` = 60 cells
- AND each cell SHALL set `SHUD_SPGMR_MAXL=<maxl>` env var per the hook from capability `spgmr-maxl-env-hook`
- AND each cell SHALL emit `profile_B0.yaml`, `cvode_stats.txt` (15 canonical keys per `tools/cvode_stats_diff/canonical_15_keys.yaml`: `nfe, nfeLS, nni, nli, nsetups, netf, nst, npe, nps, ncfn, ncfl, lenrw, leniw, lenrwLS, leniwLS`), `rivqdown.dat`, `wall.sec`, `cell.meta`, stdout containing `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` provenance line
- AND the sweep matrix dimensions (N ∈ {1, 8}) intentionally differ from the baseline matrix (N ∈ {1, 4, 8}) per design D9: baseline N=4 cells serve only as cross-N OMP-neutrality regression detector AND are not consumed by any sweep gate

#### Scenario: Slurm three-rule compliance
- WHEN sbatch submitted
- THEN the submit script SHALL invoke `sbatch` from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/` (not from `/users/`)
- AND `#SBATCH --output / --error` SHALL point to paths under `/scratch/...` (not `/tmp` or compute-node-local)
- AND any patch / hash / run.sh referenced in the job SHALL reside under `/scratch/`
- AND `--partition=CPU` SHALL target idle nodes (cn05-06,09,14-19,23-24)

### Requirement: 8-gate verdict (G1-G8)

The system SHALL adjudicate the sweep result against 8 gates per `design.md` §D8.

#### Scenario: G1 build gate
- WHEN PR-C source builds
- THEN both `make shud` and `make shud_omp` SHALL exit 0
- AND `nm SHUD/build/shud_omp | grep -i spgmr` SHALL show SUNDIALS SPGMR symbols resolved
- AND build gate G1 SHALL PASS

#### Scenario: G2 no-PREC_LEFT-regression gate
- WHEN PR-C source is finalized
- THEN per Requirement: No regression to PREC_LEFT in capability `spgmr-maxl-env-hook`, all 3 grep checks SHALL return 0 matches
- AND G2 SHALL PASS

#### Scenario: G3 default-compatibility gate
- WHEN PR-C binary is tested
- THEN per Requirement: Default-unset bit-identical equivalence in capability `spgmr-maxl-env-hook`, both keliya smoke equivalence checks SHALL show bit-identical SHA12 and 15-key counter match
- AND G3 SHALL PASS

#### Scenario: G4 solver-work gate (soft, case-asymmetric)
- WHEN aggregator processes 60-cell results
- THEN for EACH of heihe N=8 and heihe_x4 N=8, the median `nli/nni` OR `nfeLS/nfe` OR `ncfl` at the best maxl SHALL be evaluated against the cleaned `PREC_NONE` baseline (which uses default maxl=5)
- AND per-case G4 verdict SHALL be PASS (≥5% reduction in any of the 3 metrics), MARGINAL (1-5% reduction), or FAIL (no reduction)
- AND the case-asymmetric merge rule SHALL be: if heihe_x4 PASS AND heihe FAIL → G4 PASS heihe_x4-only (ADR-0004 records case-asymmetry); if heihe_x4 FAIL → G4 FAIL regardless of heihe (heihe_x4 is primary production target per CLAUDE.md)

#### Scenario: G5 wall gate (soft, primary production driver, case-asymmetric)
- WHEN aggregator processes 60-cell results
- THEN for EACH of heihe N=8 and heihe_x4 N=8, the median wall.sec at the best maxl SHALL be compared to the cleaned `PREC_NONE` baseline median wall.sec
- AND heihe_x4 G5 verdict bands SHALL be: GO+default-bump (≥10% reduction), Optional-knob (5-10% reduction), Diagnostic (1-5% reduction), NO-GO (no reduction)
- AND heihe G5 verdict SHALL be reported separately; case-asymmetric merge per ADR-0004: heihe_x4 GO + heihe NO-GO → ADR-0004 GO + default-bump but documents heihe ROI as `marginal-or-none` in §case-asymmetry-note; heihe GO + heihe_x4 NO-GO → ADR-0004 Optional-knob (env-var recommended for heihe-class case only)

#### Scenario: G6 no-solver-regression gate (hard, full counter set)
- WHEN aggregator processes 60-cell results
- THEN for every (case, N, maxl), median `ncfn` SHALL be ≤ cleaned `PREC_NONE` baseline median `ncfn` for the corresponding (case, N), i.e., heihe ≤ 7 AND heihe_x4 ≤ 51
- AND median `ncfl` SHALL be ≤ cleaned `PREC_NONE` baseline median `ncfl` for the corresponding (case, N) (heihe ≤ 85 AND heihe_x4 ≤ 3620) OR within 10% relative increase (since `ncfl` is the primary Krylov restart-count target per ADR-0003)
- AND median `netf` SHALL not regress by more than 10% relative to baseline
- AND ANY violation SHALL fail G6 (hard gate; blocks GO and Optional-knob bands)

#### Scenario: G7 hydrology-A4 gate (soft, primary hydrology safety)
- WHEN aggregator processes 60-cell results
- THEN for every (case, N, maxl), `rivqdown.dat` SHALL be compared to the cleaned `PREC_NONE` baseline `rivqdown.dat` for the corresponding (case, N) using `tools/compare_snapshot/compare_snapshot`
- AND per-cell max_ulp SHALL be ≤ A4 threshold (1024 per ADR-0002 / spec §p1e A-grade fallback) — this is the strict gate predicate (**G7-strict**)
- AND a future-tightening hook (NOT a current gate) is the water-balance OR clause: if a water-balance threshold is later established by a P8-tune.* epic that defines `water_balance_error` quantitatively, that threshold may be used in lieu of the A4 max_ulp predicate; until then, G7-strict fallbacks to A4 only per design D9
- AND ANY A4 max_ulp violation SHALL fail G7-strict (blocks the GO+default-bump branch)

#### Scenario: G7-attested hydrology gate (soft for Optional-knob / Diagnostic branches, ADR-mechanism path)
- WHEN aggregator processes 60-cell results AND G7-strict fails
- THEN the violation MAY be reclassified as G7-attested PASS if a corresponding ADR documents the violation as **solver-tunable-sensitivity with mechanism analysis** (worked example: ADR-0004 documents `SHUD_SPGMR_MAXL` bump → SUNDIALS Arnoldi Modified Gram-Schmidt residual change → CVODE step-size adapter closed-loop response → trajectory drift; both pre-tune and post-tune outputs are valid PREC_NONE solutions on different step-size paths, not corruption)
- AND the ADR SHALL cite the specific tunable + mechanism chain + cross-reference back to this G7-attested scenario
- AND for Optional-knob / Diagnostic branches: G7-attested PASS is sufficient (G7-strict PASS is preferred but not required)
- AND for NO-GO hydrology branch: G7-strict FAIL without ADR-mechanism attestation → NO-GO (corruption)
- AND for GO+default-bump branch: G7-strict PASS is required (G7-attested is insufficient — default bump must not change trajectory for default users per 'never break userspace')

#### Scenario: G8 deterministic-repeatability gate (hard)
- WHEN aggregator processes 60-cell results
- THEN for each (case, N, maxl), the 3 reps SHALL show `rivqdown.dat` ULP delta within declared maxl-dependent tolerance (≤1024 ULP)
- AND `cvode_stats.txt` 15-key counter differences across 3 reps SHALL be exactly 0
- AND ANY violation SHALL fail G8 (blocks all positive verdicts)

### Requirement: Aggregator artifact and verdict tables

The system SHALL provide an aggregator script `tools/p8tune/aggregate_maxl_sweep.sh` parsing 60-cell per-cell artifacts into 8 verdict tables, with an output schema bound to the 8-gate matrix per design D8.

#### Scenario: Aggregator input schema
- WHEN aggregator script is invoked
- THEN input SHALL be per-cell artifact directories under `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/<case>_N<n>_maxl<m>_rep<r>/`
- AND each input directory SHALL contain `profile_B0.yaml`, `cvode_stats.txt` (15 canonical keys per `tools/cvode_stats_diff/canonical_15_keys.yaml`), `rivqdown.dat`, `wall.sec`, `cell.meta`
- AND `cell.meta` SHALL include `case`, `N`, `maxl`, `rep`, `host`, `start_time`, `end_time` provenance

#### Scenario: Aggregator output schema — 8 verdict tables
- WHEN aggregator completes
- THEN it SHALL emit 8 tables corresponding to D8 gates G1-G8 into `docs/p8tune/maxl_sweep_verdict.md`:
  - Table 1 (G1 build): build verification outcome from PR-C CI gates
  - Table 2 (G2 no-PREC_LEFT-regression): grep verification outcome from PR-C
  - Table 3 (G3 default-compat): 3-way bit-identical equivalence verification from PR-C CI
  - Table 4 (G4 solver-work, case-asymmetric): per-case (heihe, heihe_x4) median `nli/nni`, `nfeLS/nfe`, `ncfl` per (N, maxl) AND relative reduction vs cleaned-baseline
  - Table 5 (G5 wall, case-asymmetric): per-case (heihe, heihe_x4) median `wall.sec` per (N, maxl) AND relative reduction vs cleaned-baseline (heihe_x4 band: GO ≥10% / Optional 5-10% / Diagnostic 1-5% / NO-GO)
  - Table 6 (G6 no-solver-regression): per (case, N, maxl) median `ncfn`, `ncfl`, `netf` AND regression flag vs cleaned-baseline
  - Table 7 (G7 hydrology-A4): per (case, N, maxl) `rivqdown.dat` max_ulp vs cleaned-baseline (threshold 1024)
  - Table 8 (G8 deterministic-repeatability): per (case, N, maxl) 3-rep `rivqdown.dat` ULP delta + 15-key counter delta
- AND aggregator SHALL ALSO emit a flat KV file `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/aggregate_verdict.txt` mirroring the p8pre identity_spike_verdict pattern (one `key = value` per line; cross-referenceable from `docs/p8tune/maxl_sweep_verdict.md`)
- AND aggregator output SHALL include hydrology metrics from `tools/compare_snapshot/compare_snapshot` (max_ulp + zero/nonzero position drift) per (case, N, maxl) for G7 input

#### Scenario: Aggregator gate verdict propagation
- WHEN aggregator output is consumed by ADR-0004 authoring
- THEN the 8-table verdict SHALL directly drive the ADR-0004 branch selection per D8 ADR-0004 decision branches
- AND ADR-0004 SHALL cite specific table rows for each PASS/FAIL claim
- AND `docs/p8tune/maxl_sweep_verdict.md` §verdict-summary SHALL cross-reference ADR-0004 §Decision

### Requirement: ADR-0004 decision authoring

The system SHALL produce `docs/adr/0004-maxl-sweep-decision.md` per the ADR template (consistent with ADR-0002 / ADR-0003), recording the verdict band and forward action.

#### Scenario: ADR-0004 GO branch
- WHEN G5 ≥10% AND G6/G7/G8 PASS
- THEN ADR-0004 verdict SHALL be "GO + production default bump"
- AND ADR SHALL identify the maxl value (5, 10, 15, 20, or 30) at which heihe_x4 wall is minimized
- AND ADR SHALL require capability `spgmr-maxl-env-hook` follow-up: change `cvode_config.cpp` to use the chosen maxl as default constant (env var stays for override)
- AND ADR SHALL specify new SHA baseline lock under `mode-C-tune` reference set (per `design.md` §D9)

#### Scenario: ADR-0004 Optional-knob branch
- WHEN G5 5-10% AND G6/G7/G8 PASS
- THEN ADR-0004 verdict SHALL be "Optional knob; no default change"
- AND ADR SHALL document recommended maxl values for production users (per case profile)
- AND env var hook stays as-is; no source default change

#### Scenario: ADR-0004 Diagnostic branch
- WHEN G4 PASS but G5 <5% improvement
- THEN ADR-0004 verdict SHALL be "Diagnostic; close P8-tune.C"
- AND ADR SHALL document the counter-vs-wall split (solver work decreases but doesn't materialize in wall) and rationale
- AND forward action SHALL recommend P8-tune.D KLU pattern-only spike

#### Scenario: ADR-0004 NO-GO branch (hydrology fail)
- WHEN G7 FAIL
- THEN ADR-0004 verdict SHALL be "NO-GO; revert env var"
- AND ADR SHALL detail the hydrology A4 violation
- AND env-var hook SHALL be removed in a follow-up PR (return SHUD source to bit-identical `37be0fe`)

#### Scenario: ADR-0004 NO-GO branch (solver regression)
- WHEN G6 FAIL
- THEN ADR-0004 verdict SHALL be "NO-GO; revert env var"
- AND ADR SHALL detail the `ncfn` regression mechanism (likely: aggressive Krylov forcing Newton step accuracy beyond CVODE tolerance, causing Newton failures)
- AND env-var hook SHALL be removed in a follow-up PR

#### Scenario: ADR-0004 NO-GO branch (no improvement)
- WHEN G4 and G5 both NO-GO
- THEN ADR-0004 verdict SHALL be "NO-GO; close P8-tune.C; transition to KLU pattern-only"
- AND ADR SHALL recommend P8-tune.D KLU pattern-only spike opening as next epic

