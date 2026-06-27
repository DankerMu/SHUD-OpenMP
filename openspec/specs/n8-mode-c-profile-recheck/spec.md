# n8-mode-c-profile-recheck Specification

## Purpose
TBD - created by archiving change p8pre-spike. Update Purpose after archive.
## Requirements
### Requirement: N=8 Mode C profile experiment design

The system SHALL run Mode C (Serial NVector + StrictOMP RHS, build via `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` with SHUD pin `7a1dc8f`) profile experiments on server (Slurm cn14/cn15 chained singleton 2-stream, per P1e PR-I deployment SOP) covering exactly the cases heihe (NumEle=6335) and heihe_x4 (NumEle=40046), exactly the thread counts N ∈ {1, 4, 8}, and exactly 3 repetitions per cell (total 18 cells).

#### Scenario: cell count matches design

- **WHEN** the orchestrator submits Slurm jobs for the N=8 Mode C profile experiment
- **THEN** the total number of submitted cells equals 18 (= 2 cases × 3 N-values × 3 reps)
- **AND** the (case, N) coverage is exactly {(heihe, 1), (heihe, 4), (heihe, 8), (heihe_x4, 1), (heihe_x4, 4), (heihe_x4, 8)}
- **AND** N=2 is explicitly NOT submitted (per design D4 — P1e PR-I data already archived, monotonic scaling does not need N=2 interpolation)

#### Scenario: build target is Mode C with profile instrumentation

- **WHEN** the per-cell run script invokes `make` in `SHUD/`
- **THEN** the make target is exactly `shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` (the `SHUD_ENABLE_PROFILE=1` flag is REQUIRED so the binary links `tools/profile/timer.cpp` and emits profile_B0.yaml; the PR-I baseline binary at SHUD `3341368d` did NOT pass `SHUD_ENABLE_PROFILE=1` so its wall numbers are NOT a valid baseline for Step 2 gate-4 — Step 1 produces the new baseline `wall_step1_baseline_median(case, N)` archived in `docs/p8pre/n8_profile_baseline.md` §5.1)
- **AND** the resulting binary links `N_VNew_Serial` (`nm ./shud | grep N_VNew_Serial ≥ 1 hit`)
- **AND** the resulting binary does NOT link `N_VNew_OpenMP` (`nm ./shud | grep N_VNew_OpenMP = 0 hit`)
- **AND** the resulting binary links libgomp (`nm ./shud | grep GOMP_parallel ≥ 1 hit`)

#### Scenario: SHUD pin unchanged across Step 1

- **WHEN** the orchestrator queries the canonical submodule pointer source via `git submodule status SHUD` in the outer `baseline/p8pre` working tree
- **THEN** the recorded SHA equals `7a1dc8f` (P1e ship pin, post-nested-Timer-fix); the `.gitmodules` file does NOT carry the pointer SHA (it only stores path/URL/branch) so checking `.gitmodules` is not a valid invariance source
- **AND** `git rev-parse HEAD` inside `SHUD/` at run time also equals `7a1dc8f`
- **AND** this invariance holds across all 18 cells of Step 1 PR-A; Step 2 (PR-D onward) MAY bump the pointer to a child commit of `7a1dc8f` on the `openmp-baseline-p8pre` branch — the descendant relationship is verified separately by spec p8precond-zero-identity-spike Requirement "SHUD baseline preservation and forward-only extension"

### Requirement: SHUD canonical 15-key CVODE stats archive

The system SHALL archive the SHUD canonical 15-key CVODE stats per cell to `<scratch>/SHUD-OpenMP/.p8pre-runs/<case>_N<n>_rep<r>/cvode_stats.txt`, with keys EXACTLY matching the SHUD-emitted set per `SHUD/src/Equations/cvode_config.cpp:94-108` and `tools/cvode_stats_diff/canonical_15_keys.yaml`: `nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS`.

#### Scenario: per-cell stats file present and parseable

- **WHEN** an aggregator reads `<scratch>/SHUD-OpenMP/.p8pre-runs/heihe_N8_rep1/cvode_stats.txt`
- **THEN** the file contains exactly 15 key-value lines (one per stat key, in any order)
- **AND** all 15 keys are present from the canonical set `nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS` (per SHUD/src/Equations/cvode_config.cpp:94-108 `PrintFinalStats` printf order)
- **AND** the aggregator REJECTS as unknown keys any of `nlcf` (this is a typo; the SUNDIALS canonical name for linear-solver convergence failures is `ncfl`), `nfevals` (does not exist; `nfeLS` is the linear-solver RHS evals counter), `hcur` / `qcur` / `hin` (only emitted under `SHUD_ENABLE_DIAGNOSTICS` build, which is NOT enabled by this epic — see Non-Goals "do not modify SHUD pin")
- **AND** numeric values parse cleanly (no NaN, no truncated lines, no trailing whitespace mismatches)

#### Scenario: PREC_NONE baseline `nps = npe = 0`

- **WHEN** an aggregator reads `cvode_stats.txt` for any cell
- **THEN** `nps = 0` AND `npe = 0` (the baseline has no preconditioner registered, so SUNDIALS never calls `PSolve` or `PSetup`)
- **AND** this serves as the baseline for Step 2 spike's hard gate 3 (`nps > 0` AND `npe > 0` post-PREC_LEFT-identity-wire-up)

### Requirement: bucket breakdown using emitted timer buckets

The system SHALL collect timer bucket breakdown per cell using the 7 named buckets in the `buckets:` block emitted by `tools/profile/timer.cpp:158-167` dump() — `t_RHS_kernel` / `t_RHS_total` / `t_CVODE_internal` / `t_forcing_io` / `t_ET` / `t_output` / `t_other` — without introducing new sub-buckets in this epic.

#### Scenario: 7 emitted buckets present per cell yaml

- **WHEN** an aggregator reads the per-cell profile yaml at `<scratch>/SHUD-OpenMP/.p8pre-runs/<case>_N<n>_rep<r>/profile_B0.yaml`
- **THEN** all 7 timer buckets (`t_RHS_kernel`, `t_RHS_total`, `t_CVODE_internal`, `t_forcing_io`, `t_ET`, `t_output`, `t_other`) are present as fields under the top-level `buckets:` block
- **AND** the `extras:` block contains `t_CVODE_raw` (= raw measurement, wraps every `CVode()` solver call INCLUDING the RHS callbacks; emitted by `timer.cpp:183`) and `t_wall_total` (= top-level solver-loop wall, emitted by `timer.cpp:184`); per `timer.cpp:117-127` derived identities: `t_CVODE_internal = t_CVODE_raw - t_RHS_total` (net solver time, NOT a raw bucket — this is why p2a archives labeled the raw measurement `t_CVODE_raw` while the emitted bucket name is `t_CVODE_internal`; both refer to the SAME quantity once the derivation is applied)
- **AND** the bucket-sum invariant SHALL hold: `t_RHS_total + t_CVODE_internal + t_forcing_io + t_ET + t_output + t_other` ≈ `t_wall_total` (within ±2% rounding + Timer overhead allowance); NOTE `t_RHS_kernel` is EXCLUDED from this sum because `t_RHS_kernel ⊆ t_RHS_total` (RHS kernel is a sub-bucket of RHS total; including both double-counts kernel time — the p2a v0.1 nested-double-count bug fixed in SHUD `7a1dc8f` was caused by this exact containment error)
- **AND** no new sub-bucket like `t_SPGMR_Krylov` or `t_step_control` is introduced (per Open Question Q3 — sub-bucket profiling deferred to P8-precond formal epic)

### Requirement: nfeLS/nfe and nli/nni quantification

The system SHALL compute `nfeLS / nfe` and `nli / nni` ratios per (case, N) using median across 3 reps, and report these in `docs/p8pre/n8_profile_verdict.md`.

#### Scenario: nfeLS/nfe ratio reported per (case, N)

- **WHEN** the orchestrator reads `docs/p8pre/n8_profile_verdict.md` Table §3 "CVODE stats summary"
- **THEN** for each of the 6 (case, N) combinations `{(heihe, 1), (heihe, 4), (heihe, 8), (heihe_x4, 1), (heihe_x4, 4), (heihe_x4, 8)}`, the table contains:
  - `nfe_median`, `nfeLS_median`, `nfeLS/nfe`
  - `nni_median`, `nli_median`, `nli/nni`
  - `nsetups_median`, `netf_median`, `ncfn_median`
- **AND** ratios are formatted to 3 decimal places (e.g., `nfeLS/nfe = 4.582`)

#### Scenario: ROI verdict based on nfeLS/nfe exhaustive branching

- **WHEN** the orchestrator interprets the `nfeLS/nfe` ratios per case for N=8, denote `r_min = min(nfeLS/nfe over cases)` and `r_max = max(nfeLS/nfe over cases)`
- **THEN** the verdict branches are EXHAUSTIVE and MUTUALLY EXCLUSIVE over the (r_min, r_max) lattice:
  - branch (a): `r_min ≥ 1.5` → verdict = "P8-precond ROI 满足前置", proceed to Step 2 P8-precond-0 spike
  - branch (b): `r_max < 1.5` → verdict = "P8-precond ROI 不满足，转 P8-tune (maxl/restart/EpsLin)", ADR-0003 directly writes NOT 启动 precond epic
  - branch (c): `r_min < 1.5 ≤ r_max < 3.0` → verdict = "case-aware precond design required", Step 2 spike still proceeds for API gating, but ADR-0003 records open issue for case-aware precond strategy
  - branch (d): `r_min < 1.5 AND r_max ≥ 3.0` → verdict = "case-aware + HIGH HETEROGENEITY", Step 2 spike still proceeds, ADR-0003 records open issue AND flags heterogeneity for design (e.g. heihe nfeLS/nfe=4.0 / heihe_x4 nfeLS/nfe=1.0 would land here, demanding case-aware precond from day one)
- **AND** the 4 branches cover the full (r_min, r_max) ∈ ℝ² lattice with r_min ≤ r_max; the aggregator SHALL emit the branch letter (a/b/c/d) explicitly so the verdict is unambiguous

### Requirement: cross-N stats invariance check (5 counters)

The system SHALL verify that for each case, ALL five counters `nst / nfe / nfeLS / nni / nsetups` are invariant across N ∈ {1, 4, 8} when running Mode C (per P1e AC-S1 strict bitwise expectation extended to the CVODE stats domain).

#### Scenario: nst invariance per case

- **WHEN** the aggregator computes `nst_N=1 / nst_N=4 / nst_N=8` per case
- **THEN** for both heihe and heihe_x4, `nst_N=1 = nst_N=4 = nst_N=8` (Δ=0 strict)
- **AND** expected values per `docs/p1e/p1e_perf_baseline.md` §3.4 nst ladder: heihe nst=6698, heihe_x4 nst=6575
- **AND** if Δ > 0 observed, the verdict marks "P1e Mode C invariant regression detected", blocking Step 2 spike

#### Scenario: nfe invariance per case

- **WHEN** the aggregator computes `nfe_N=1 / nfe_N=4 / nfe_N=8` per case
- **THEN** for both heihe and heihe_x4, `nfe_N=1 = nfe_N=4 = nfe_N=8` (Δ=0 strict)
- **AND** expected baseline values from `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1: heihe nfe=6943, heihe_x4 nfe=6741
- **AND** if Δ > 0 observed, blocks Step 2 spike (Mode C strict-omp determinism regression)

#### Scenario: nfeLS / nni / nsetups invariance per case

- **WHEN** the aggregator computes `X_N=1 / X_N=4 / X_N=8` per case for each X ∈ {nfeLS, nni, nsetups}
- **THEN** for both heihe and heihe_x4 AND for each X, `X_N=1 = X_N=4 = X_N=8` (Δ=0 strict, mirroring nst + nfe per PR-I AC-S1 raw counter behavior)
- **AND** if Δ > 0 observed for ANY of the three counters in ANY case, blocks Step 2 spike — this confirms Mode C strict-omp is deterministic-by-construction independent of thread count for the full Krylov-iteration-counting suite, not only step-count

### Requirement: profile baseline document and gate-4 baseline archive

The system SHALL produce `docs/p8pre/n8_profile_baseline.md` containing the experimental setup, raw data tables (including `wall_step1_baseline_median(case, N)` as the gate-4 baseline source for Step 2 — see spec p8precond-zero-identity-spike Requirement "4 hard gates"), ratio analyses, and ROI verdict.

#### Scenario: profile baseline doc sections present

- **WHEN** a reader opens `docs/p8pre/n8_profile_baseline.md`
- **THEN** the doc contains: YAML metadata + Abstract + §1 Introduction (含 H1/H2/H3 hypotheses) + §2 Related Work + §3 Methodology + §4 Experimental Setup (含 SHUD pin / build / case / N / reps + partition pin cn14/cn15) + §5 Results 含 §5.1 raw data tables (wall median + 15-key CVODE stats per cell — the §5.1 `wall_step1_baseline_median(case, N)` table is the SOURCE OF TRUTH consumed by Step 2 gate 4) + §5.2 cross-N invariance check + §5.3 absolute baseline anchor + §5.4 nfeLS/nfe + nli/nni ROI ratios + §5.5 branch verdict + §6 Discussion + §7 Limitations + §8 Conclusion + §9 Future Work + §10 References
- **AND** §6 ROI verdict references `nfeLS/nfe` threshold per the Scenario "ROI verdict based on nfeLS/nfe exhaustive branching" above
- **AND** the doc is in academic-paper-style per CLAUDE.md "阶段总结文档风格" user-pref (P1e capstone style 母本 `docs/p1e/p1e_academic_summary.md`)

#### Scenario: local aggregation path binding

- **WHEN** the aggregator runs on Mac local (per task §3.1)
- **THEN** the aggregator reads input from `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/` (= rsync mirror of `<scratch>/SHUD-OpenMP/.p8pre-runs/<case>_N<n>_rep<r>/`)
- **AND** both paths point at the SAME 18-cell artifact set; the rsync layer exists so the Mac aggregator can iterate locally without ssh round-trips per cell (mirrors the PR-I aggregator pattern in `docs/p1e/p1e_pr_i_strict_omp_verification.md:297`)

