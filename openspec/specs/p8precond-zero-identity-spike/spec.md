# p8precond-zero-identity-spike Specification

## Purpose
TBD - created by archiving change p8pre-spike. Update Purpose after archive.
## Requirements
### Requirement: SUNDIALS preconditioner API wire-up

The system SHALL wire up SUNDIALS 6.0.0 `CVodeSetPreconditioner` and `CVodeSetLSetupFrequency` APIs in `SHUD/src/Equations/cvode_config.cpp` by changing the `SUNLinSol_SPGMR` call signature from `PREC_NONE` to `PREC_LEFT` and registering identity stub `PSetup` + `PSolve` callbacks declared in a new file `SHUD/src/Equations/MD_precond_identity.cpp`, with CVLS call ordering preserved per SUNDIALS 6.0.0 contract.

#### Scenario: cvode_config.cpp call ordering and signature change

- **WHEN** a reviewer reads `SHUD/src/Equations/cvode_config.cpp` lines 259-265 after the PR-D edits land
- **THEN** the post-edit call sequence is EXACTLY (in order, with `check_flag` between each call):
  1. `LS = SUNLinSol_SPGMR(udata, PREC_LEFT, 0, sunctx);` (line 259, changed from `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` where 0 = `PREC_NONE`)
  2. `check_flag((void *)LS, "SUNLinSol_SPGMR", 0);`
  3. `flag = CVodeSetLinearSolver(cvode_mem, LS, NULL);`
  4. `check_flag(&flag, "CVSpilsSetLinearSolver", 1);`
  5. `flag = CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity);`
  6. `check_flag(&flag, "CVodeSetPreconditioner", 1);`
  7. `flag = CVodeSetLSetupFrequency(cvode_mem, 50);` (50 = SUNDIALS default per design D6)
  8. `check_flag(&flag, "CVodeSetLSetupFrequency", 1);`
- **AND** the `CVodeSetPreconditioner` call MUST come AFTER `CVodeSetLinearSolver` (SUNDIALS 6.0.0 CVLS contract: the preconditioner registration is bound to the linear-solver memory allocated by `CVodeSetLinearSolver`; inserting `CVodeSetPreconditioner` before `CVodeSetLinearSolver` trips the `cvLsInitialize` cvLsMem-null precondition check at runtime — silent no-op or abort depending on SUNDIALS check level)
- **AND** the file includes `MD_precond_identity.h` (declarations of `PSetupIdentity` + `PSolveIdentity`)

#### Scenario: MD_precond_identity.cpp signatures

- **WHEN** a reviewer reads `SHUD/src/Equations/MD_precond_identity.cpp`
- **THEN** it defines exactly two extern "C" linkage functions:
  - `int PSetupIdentity(realtype t, N_Vector y, N_Vector fy, booleantype jok, booleantype *jcurPtr, realtype gamma, void *user_data)` returning 0 and setting `*jcurPtr = SUNFALSE` (jok-mirror canonical pattern per SUNDIALS `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716 `Precond` reference — write `*jcurPtr = jok` to mirror the call site's jok flag; identity spike specializes to `*jcurPtr = SUNFALSE` because identity holds no cached state across Newton iterations)
  - `int PSolveIdentity(realtype t, N_Vector y, N_Vector fy, N_Vector r, N_Vector z, realtype gamma, realtype delta, int lr, void *user_data)` returning 0 after copying `r` to `z` via `N_VScale(1.0, r, z)` (memcpy-style pattern per SUNDIALS `cvDiurnal_kry.c` L760 `PSolve` reference — identity solve is the degenerate case of the canonical memcpy/scale-by-one residual rotation)
- **AND** both signatures match SUNDIALS 6.0.0 `CVLsPrecSetupFn` and `CVLsPrecSolveFn` typedefs in `cvode/cvode_ls.h`

#### Scenario: MD_precond_identity.h declarations

- **WHEN** a reviewer reads `SHUD/src/Equations/MD_precond_identity.h`
- **THEN** it contains include guards (`#ifndef SHUD_MD_PRECOND_IDENTITY_H` / `#define ... / #endif`) + `#include <sundials/sundials_types.h>` + `#include <nvector/nvector_serial.h>` + `extern "C" { ... }` block declaring the two function prototypes matching the .cpp definitions
- **AND** the header is used by `cvode_config.cpp` via `#include "MD_precond_identity.h"`

#### Scenario: Makefile auto-pickup via glob (no OBJS edit)

- **WHEN** a reviewer reads `SHUD/Makefile` lines 380-383
- **THEN** the `SRC` variable definition uses globbing: `SRC = $(SRC_DIR)/classes/*.cpp $(SRC_DIR)/ModelData/*.cpp $(SRC_DIR)/Model/*.cpp $(SRC_DIR)/Equations/*.cpp`
- **AND** the `$(SRC_DIR)/Equations/*.cpp` glob automatically picks up the newly-added `MD_precond_identity.cpp` without any edit to the Makefile (verified by `make -n shud SHUD_ENABLE_OPENMP_RHS=1 | grep MD_precond_identity` showing a compile line)
- **AND** `make shud SHUD_ENABLE_OPENMP_RHS=1` succeeds without compile or link errors related to `PSetupIdentity` / `PSolveIdentity` / `CVodeSetPreconditioner`
- **AND** PR-D MUST NOT add an explicit `OBJS += MD_precond_identity.o` line (or equivalent) to the Makefile — the glob mechanism is the established SHUD convention and an explicit add would be a no-op + spurious diff

#### Scenario: no symbol collision

- **WHEN** PR-D pre-flight runs `grep -rnE "PSetup|PSolve" SHUD/src/`
- **THEN** the grep returns 0 hits before `MD_precond_identity.{h,cpp}` are added (master plan §4.17.2 confirms no existing preconditioner stubs in the SHUD tree)
- **AND** if pre-existing `PSetup` / `PSolve` / `PSetupIdentity` / `PSolveIdentity` symbol IS found, PR-D SHALL rename the new symbols to `PSetupP8preIdentity` / `PSolveP8preIdentity` to avoid the multi-definition link error

### Requirement: API existence pre-flight verification

The system SHALL verify that the correct SUNDIALS 6.0.0 setup-frequency and preconditioner APIs exist in the linked headers BEFORE proceeding with the spike implementation, per Open Question Q1 in design.md.

#### Scenario: grep cvode.h and cvode_ls.h confirm canonical API names

- **WHEN** the implementer runs `grep -n "CVodeSetLSetupFrequency" $(SHUD_PATH)/InstallSundials/include/cvode/cvode.h` AND `grep -n "CVodeSetJacEvalFrequency" $(SHUD_PATH)/InstallSundials/include/cvode/cvode_ls.h`
- **THEN** both greps return at least 1 hit each (verified 2026-06-26 in this repo: `CVodeSetLSetupFrequency` at cvode.h:132 + `CVodeSetJacEvalFrequency` at cvode_ls.h:91)
- **AND** `CVodeSetLSetupFrequency(void *cvode_mem, long int msbp)` IS the correct API to control SPGMR preconditioner setup frequency for the iterative-linear-solver code path (per SUNDIALS docs + design.md D6); `CVodeSetJacEvalFrequency` is the CVLS Jacobian-eval frequency for matrix-based solvers and is NOT applicable here
- **AND** the implementer SHALL reject `CVodeSetMaxConvFails` (cvode.h:133) as a fallback — that API controls the max number of nonlinear-iteration convergence failures per step (default 10) and is UNRELATED to setup frequency; pointing the implementer at this symbol misleads the wire-up
- **AND** if either grep returns 0 hits, PR-D BLOCKS — the SUNDIALS install is corrupted or wrong version; resolve before continuing

### Requirement: 4 hard gates for identity spike verdict

The system SHALL evaluate 4 hard gates against the server identity-spike-run data (heihe + heihe_x4 × N ∈ {1,4,8} × 3 reps × Mode C with PREC_LEFT identity, total 18 cells) and record verdict in `docs/p8pre/identity_spike_verdict.md`.

#### Scenario: gate 1 build PASS (server binary)

- **WHEN** the aggregator inspects the build evidence file `<scratch>/SHUD-OpenMP/.p8pre-runs/identity_spike/server_nm.log` produced by task §7.0 — this is the SERVER binary built with gcc 13.3.0 + libgomp, the same toolchain that produces the 18-cell artifacts
- **THEN** the log shows `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` exit code 0
- **AND** `nm ./shud | grep -E "PSetupIdentity|PSolveIdentity" | wc -l` returns ≥ 2 (both symbols linked)
- **AND** `nm ./shud | grep CVodeSetPreconditioner | wc -l` returns ≥ 1 (SUNDIALS preconditioner API symbol resolved)
- **AND** the Mac sanity build evidence from task §6.6 is NOT the gate-1 evidence (Mac Apple Clang + libomp differs from server gcc + libgomp ABI; Mac PASS does not imply server linker resolves identically)

#### Scenario: gate 2 zero convergence failure

- **WHEN** the aggregator reads `cvode_stats.txt` for all 18 cells
- **THEN** for every cell, `ncfn = 0` (CVODE nonlinear convergence failure count)
- **AND** the SHUD process exit code is 0 for every cell
- **AND** if ANY cell reports `ncfn > 0`, gate 2 FAIL and Step 2 spike verdict = NO-GO

#### Scenario: gate 3 nps and npe accumulation

- **WHEN** the aggregator reads `cvode_stats.txt` for all 18 cells
- **THEN** for every cell, `nps > 0` AND `npe > 0`
- **AND** this confirms SUNDIALS truly calls `PSolveIdentity` and `PSetupIdentity` (the API wire-up is alive)
- **AND** if ANY cell reports `nps = 0` OR `npe = 0`, gate 3 FAIL — API wire-up broken (e.g., callback pointers not stored)

#### Scenario: gate 4 wall non-regression vs Step 1 PR-A baseline

- **WHEN** the aggregator computes `wall_identity_median(case, N)` (median of 3 reps per cell, mirroring the Step 1 PR-A 3-rep median aggregation policy) and compares to `wall_step1_baseline_median(case, N)` from `docs/p8pre/n8_profile_baseline.md` §5.1 raw data table — NOT to PR-I baseline from `docs/p1e/p1e_perf_baseline.md` §3.1
- **THEN** for every (case, N) ∈ {(heihe, 1), (heihe, 4), (heihe, 8), (heihe_x4, 1), (heihe_x4, 4), (heihe_x4, 8)} the case-aware non-regression tolerance is:
  - `|wall_identity_median(heihe, N) - wall_step1_baseline_median(heihe, N)| / wall_step1_baseline_median(heihe, N) ≤ 0.10` (10% — heihe N=8 wall ~473s; 5% = 24s headroom is below Slurm submission jitter 2-3% × 473 ≈ 9-14s + intrinsic rep noise 1-2%; 10% gives 47s headroom)
  - `|wall_identity_median(heihe_x4, N) - wall_step1_baseline_median(heihe_x4, N)| / wall_step1_baseline_median(heihe_x4, N) ≤ 0.05` (5% — heihe_x4 N=8 wall ~775s; 5% = 39s headroom > intrinsic noise)
- **AND** both sides of the comparison MUST be built with the same matrix (`SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`) and partition-pinned to cn14 (heihe) or cn15 (heihe_x4) per task §7.1 — using PR-I wall median (built at SHUD `3341368d` WITHOUT `SHUD_ENABLE_PROFILE=1`) would mix two non-comparable build matrices (Timer instrumentation overhead + nested-Timer fix bias propagate to the wall delta and can swamp the 5% / 10% tolerance)
- **AND** if ANY (case, N) wall regression exceeds the case-aware tolerance, gate 4 FAIL — identity precond overhead unacceptable, dig into `nsetups * setup_time` attribution via soft gate 6

### Requirement: 2 soft gates for carve-out

The system SHALL evaluate 2 soft gates and record observations (PASS / FAIL with carve-out) in `docs/p8pre/identity_spike_verdict.md` §5.

#### Scenario: soft gate 5 cross-N tolerance

- **WHEN** the aggregator computes `<case>.rivqdown.dat` SHA12 per (case, N, rep) and compares to a SINGLE per-case baseline value `baseline_SHA12(case)` from `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 (the PR-I per-case unique SHA12 across N × rep — parameterized by case only because Mode C strict-omp produces one unique SHA12 per case across all 12 cells of N∈{1,2,4,8} × 3 reps; baseline values: heihe `a2023ccd2de4`, heihe_x4 `b5e4b0a2cf83`)
- **THEN** strict bitwise expectation: all 18 cells SHA12 == baseline_SHA12(case) (because PREC_LEFT identity is mathematically equivalent to PREC_NONE)
- **AND** if SHA12 mismatch observed, fall back to A4 tolerance via `tools/compare_snapshot/compare_snapshot <baseline_rivqdown> <new_rivqdown>` which returns exit code 0 + `max_ulp ≤ 1024` per master plan §2.3 (NOTE: `rivqdown.dat` may be text or binary depending on dump format; if `compare_snapshot` rejects text format, fall back to a Python script using `numpy.frombuffer` + `numpy.spacing` to compute element-wise ulp deltas)
- **AND** if `max_ulp ≤ 1024`, soft gate 5 = PASS with carve-out (record "SUNDIALS PREC_LEFT path triggers extra N_VLinearSum / N_VScale operations, reduction order non-bitwise, but within A4 §2.3 tolerance"); if `max_ulp > 1024`, soft gate 5 = FAIL, write ADR-0003 open issue

#### Scenario: soft gate 6 setup overhead via t_precond_setup bucket

- **WHEN** the aggregator computes `t_precond_setup_seconds / t_wall_total_seconds` per cell, where `t_precond_setup` is a NEW Timer bucket instrumented inside `PSetupIdentity` (RAII `shud_profile::Timer _t("t_precond_setup");` at the head of the function body in `MD_precond_identity.cpp`) and emitted to profile_B0.yaml `extras:` section per the `tools/profile/timer.cpp` `emit_extra` pattern at L176-184
- **THEN** the overhead ratio is ≤ 0.05 per cell, soft gate 6 = PASS
- **AND** if > 0.05, soft gate 6 = FAIL — record `LSetupFrequency` value used (e.g., 1 = every Newton iter, 50 = every 50 iters) and recommend adjusting LSetupFrequency upward in formal P8-precond epic
- **AND** if `t_precond_setup` bucket is absent from profile_B0.yaml (e.g., PR-D omitted the Timer instrumentation inside `PSetupIdentity`), soft gate 6 = DEFER and verdict doc records the deferral against Open Question Q3 — direct measurement is required for an actionable verdict; the back-of-envelope `nsetups × N_VLinearSum_cost` reasoning in design.md D5 is NOT a valid operational definition because there is no measured `N_VLinearSum_cost` baseline in this epic

### Requirement: spike verdict and ADR-0003 draft

The system SHALL produce `docs/p8pre/identity_spike_verdict.md` containing the 4 hard gate verdicts + 2 soft gate observations + recommendation for formal P8-precond epic intake, and draft `docs/adr/0003-precond-spike-decision.md` with formal decision.

#### Scenario: spike verdict doc sections present

- **WHEN** a reader opens `docs/p8pre/identity_spike_verdict.md`
- **THEN** the doc contains: YAML metadata + §1 目的 + §2 实验设置 + §3 raw data (18 cell wall + nps + npe + ncfn + SHA12 + per-cell t_precond_setup if instrumented) + §4 4 hard gate verdict table + §5 2 soft gate observation table + §6 ROI implication on formal P8-precond epic + §7 ADR-0003 推荐
- **AND** §4 verdict table marks each gate explicitly PASS or FAIL (no ambiguity); §5 marks each soft gate PASS / FAIL / DEFER
- **AND** §7 ADR-0003 推荐 includes the formal go/no-go for full P8-precond epic intake (Diagonal/Jacobi → GW element-Jacobi → river bidiagonal → lake dense → 必要时 ILU(0)) based on the combined Step 1 nfeLS/nfe + Step 2 hard gate verdict

#### Scenario: ADR-0003 draft produced

- **WHEN** a reader opens `docs/adr/0003-precond-spike-decision.md`
- **THEN** the doc follows the ADR template (per `docs/adr/0002-solver-path.md` structure: Status / Date / Deciders / Owner / Tags / Context / Decision / Consequences / References)
- **AND** Decision section explicitly states one of: (a) "GO 启动正式 P8-precond epic" with proposed first-PR scope, (b) "NO-GO 转 P8-tune" with rationale, (c) "DEFER 等独立 ADR" with the open issue (e.g., cross-N reduction drift) explicitly described

#### Scenario: local aggregation path binding

- **WHEN** the aggregator runs on Mac local (per task §8.1)
- **THEN** the aggregator reads input from `/tmp/p8pre_identity_spike/<case>_N<n>_rep<r>/` (= rsync mirror of `<scratch>/SHUD-OpenMP/.p8pre-runs/identity_spike/<case>_N<n>_rep<r>/`)
- **AND** both paths point at the SAME 18-cell artifact set; the rsync layer exists so the Mac aggregator can iterate locally without ssh round-trips per cell (mirrors the PR-I aggregator pattern in `docs/p1e/p1e_pr_i_strict_omp_verification.md:297`)

### Requirement: SHUD baseline preservation and forward-only extension

The system SHALL preserve the P1e SHIP state by ensuring the SHUD submodule pointer changes ONLY as a forward-only linear extension of the `7a1dc8f` P1e ship pin, and SHALL NOT modify the `baseline/P1e` D11-locked branch, and SHALL NOT trigger re-verification of P1e 3 SHALL gates (AC-S1/S2/S3) during this epic.

#### Scenario: Step 1 SHUD pointer invariance

- **WHEN** the orchestrator queries the SHUD submodule pointer SHA recorded in the outer tree object during Step 1 (PR-A → PR-C) via `git submodule status SHUD` or `git ls-tree HEAD -- SHUD` on `baseline/p8pre`
- **THEN** the recorded SHA equals `7a1dc8f` (no bump during Step 1 — Step 1 reads the P1e ship pin as-is)
- **AND** querying `.gitmodules` is NOT a valid SHA source (it only carries path/URL/branch fields, never a SHA; the pointer SHA lives exclusively in the outer commit tree object)

#### Scenario: Step 2 forward-only descendant extension

- **WHEN** the orchestrator inspects the SHUD submodule pointer SHA after PR-D bump (Step 2)
- **THEN** the new SHUD SHA is a LINEAR DESCENDANT of `7a1dc8f` on the `openmp-baseline-p8pre` branch, verified by `git -C SHUD merge-base <new SHA> 7a1dc8f` returning exactly `7a1dc8f`
- **AND** the new SHUD branch `openmp-baseline-p8pre` is forked from `openmp-baseline` at `7a1dc8f` (per master plan submodule workflow C8); commits land only on `openmp-baseline-p8pre`, NEVER on `openmp-baseline` master line
- **AND** `git log SHUD/ --oneline` shows the new SHUD pin as a single linear child of `7a1dc8f` (e.g., `<new SHA>: feat(p8pre): identity preconditioner stub + cvode_config PREC_LEFT wire`)
- **AND** the `.gitmodules` `branch = openmp-baseline` field remains unchanged (the working branch shift to `openmp-baseline-p8pre` is reflected by the pointer SHA bump in the outer tree, not by editing `.gitmodules`)

#### Scenario: baseline/P1e branch untouched

- **WHEN** the orchestrator inspects the `baseline/P1e` branch protection settings
- **THEN** `lock_branch = true` and `enforce_admins = true` remain
- **AND** no commit on `baseline/P1e` is added during this epic
- **AND** the P1e tag `P1e-tag` (annotated object `25023eff32d1`) is not modified

#### Scenario: no P1e SHALL re-verification

- **WHEN** the orchestrator reviews `docs/p8pre/*.md` produced by this epic
- **THEN** no document re-runs AC-S1 (cross-N SHA bitwise), AC-S2 (mode C vs mode A SHA equality), or AC-S3 (D7 per-case sp@8 ≥ threshold AND-gate) as a new SHALL gate
- **AND** any cross-reference to P1e SHALL gates is for context only (e.g., "PR-I per-case baseline SHA12 used as the soft-gate-5 strict-bitwise expected value in this epic")

