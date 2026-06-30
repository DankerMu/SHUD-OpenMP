# P8-tune.G0 PR-0 pre-spike notes (tasks 1.1–1.7)

Executed: 2026-06-29 (Mac local; `SHUD` pin `1ab61c0`).

This file resolves the 7 prereq spike questions before the linsol refactor
lands. Outcomes inform `SHUD/src/Equations/sunlinsol_hypre.{h,cpp}` and
`cvode_config.cpp` shape.

---

## 1.1 MD_adjacency reusability for Setup sparsity

**Decision: PROBE-DERIVE (topology-restricted ATimes probe).**

Rationale:

- `SHUD/src/ModelData/MD_adjacency.hpp` exposes 7 element/river/lake/segment
  list buckets (`seg_by_riv`, `seg_by_ele`, `upstream_by_down`,
  `riv_in_by_lake`, `ele_by_lake`, `lake_bank_edge_by_lake`, `edge_by_ele`)
  driven by `build_adjacency_lists(Model_Data*)`. These are deterministic
  gather lists for the RHS path, NOT a NumY×NumY CSR/CSC Jacobian pattern.
- The lists are keyed in **mesh-element / river / lake index space**, but
  the CVODE state vector is in **NumY = element-stripe + river + lake**
  ordering. A direct reuse would require a row-mapping layer that
  MD_adjacency does not ship today.
- The wrapper's Setup needs `(row_i → list of col_j)` in NumY space, with
  each row's coupling bounded by ≤ 8 neighbors per spec D5
  (3 element-edge neighbors × 3 stripes + river/lake edges).
- Building this pattern inline from `MD->Ele[i].nabr[k]` + river/lake
  topology at first Setup call is straightforward, deterministic, and
  decouples the wrapper from MD_adjacency's internal schema. The probe
  itself fills the values; the pattern is computed once and cached on
  the content struct.

MD_adjacency remains useful for the cross-validation in task 4.10
(symbolic vs probe nnz comparison), but is NOT in the Setup hot path.

## 1.2 SPGMR baseline setup-inclusion convention

**Decision: `setup_included`.**

The constant in `tools/p8tune.D/spgmr_baseline_walls.h`
(`SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579`) is
defined as `total_wall_s / total_nst` where `total_wall_s` is the 3-rep
median wall of `heihe_x4 N=1 maxl=5` (= 1489.76s) and `total_nst` is the
CVODE step count (= 6575). The numerator captures the FULL run wall
inclusive of any solver setup overhead.

For G0-5 comparability, the new `tools/p8tune.G0/spgmr_baseline_walls_g0.h`
case-specific anchors `SPGMR_PER_STEP_HEIHE_X4_S` / `SPGMR_PER_STEP_HEIHE_X16_S`
will be measured the same way (`total_wall_s / total_nst`) at task 4.9
(orchestrator scope), and AMG per-step measurements at PR-B aggregator
(task 7.4) will also include AMG Setup overhead in the per-step quotient.

The aggregator emits `wall_convention=setup_included` per row.

## 1.3 SUNDIALS 6.0 SUNLinearSolver_Ops ABI (on-disk)

**Confirmed 15 callbacks** at
`SHUD/InstallSundials/include/sundials/sundials_linearsolver.h:108-127`:

| # | Callback | Signature |
|---|---|---|
| 1 | gettype | `SUNLinearSolver_Type (*)(SUNLinearSolver)` |
| 2 | getid | `SUNLinearSolver_ID (*)(SUNLinearSolver)` |
| 3 | setatimes | `int (*)(SUNLinearSolver, void*, SUNATimesFn)` |
| 4 | setpreconditioner | `int (*)(SUNLinearSolver, void*, SUNPSetupFn, SUNPSolveFn)` |
| 5 | setscalingvectors | `int (*)(SUNLinearSolver, N_Vector, N_Vector)` |
| 6 | setzeroguess | `int (*)(SUNLinearSolver, booleantype)` |
| 7 | initialize | `int (*)(SUNLinearSolver)` |
| 8 | setup | `int (*)(SUNLinearSolver, SUNMatrix)` |
| 9 | solve | `int (*)(SUNLinearSolver, SUNMatrix, N_Vector, N_Vector, realtype)` |
| 10 | numiters | `int (*)(SUNLinearSolver)` |
| 11 | resnorm | `realtype (*)(SUNLinearSolver)` |
| 12 | lastflag | `sunindextype (*)(SUNLinearSolver)` |
| 13 | space | `int (*)(SUNLinearSolver, long int*, long int*)` |
| 14 | resid | `N_Vector (*)(SUNLinearSolver)` |
| 15 | free | `int (*)(SUNLinearSolver)` |

`SUNLinSolNewEmpty(SUNContext sunctx)` is declared at L143. Return-code
symbols at L187-209: `SUNLS_SUCCESS=0`, `SUNLS_MEM_NULL=-801`,
`SUNLS_ILL_INPUT=-802`, `SUNLS_MEM_FAIL=-803`, `SUNLS_ATIMES_NULL=-804`,
`SUNLS_ATIMES_FAIL_UNREC=-805`, `SUNLS_PSET_FAIL_UNREC=-806`,
`SUNLS_PSOLVE_NULL=-807`, `SUNLS_PSOLVE_FAIL_UNREC=-808`,
`SUNLS_PACKAGE_FAIL_UNREC=-809`, `SUNLS_GS_FAIL=-810`,
`SUNLS_QRSOL_FAIL=-811`, `SUNLS_VECTOROP_ERR=-812`,
`SUNLS_RES_REDUCED=801`, `SUNLS_CONV_FAIL=802`,
`SUNLS_ATIMES_FAIL_REC=803`, `SUNLS_PSET_FAIL_REC=804`,
`SUNLS_PSOLVE_FAIL_REC=805`, `SUNLS_PACKAGE_FAIL_REC=806`,
`SUNLS_QRFACT_FAIL=807`, `SUNLS_LUFACT_FAIL=808`.

**Confirmed absent in 6.0**: `SUN_SUCCESS`, `SUN_ERR_FAIL`,
`SUNLS_CONV_FAIL_UNREC`. Wrapper uses only the 6.0-listed return codes.

`SUNATimesFn` signature confirmed at
`SHUD/InstallSundials/include/sundials/sundials_iterative.h:95`:
`typedef int (*SUNATimesFn)(void *A_data, N_Vector v, N_Vector z);`

## 1.4 CVODE Setup-call cadence on keliya 90-day SHORT

**Measured: `nsetups = 0`** (zero Setup callbacks issued by CVODE under
the unmodified SPGMR `PREC_NONE` build on keliya 90-day SHORT;
2026-06-29 Mac local run, `SHUD` pin `1ab61c0`, total wall 27.6s,
nfe=112463, nfeLS=116751, nst=111130).

Source: `cvode_stats.txt` at
`.review-evidence/g0-spgmr-baseline-90day-keliya/mac/cvode_stats.txt`
emitted by `PrintFinalStats` (`CVodeGetNumLinSolvSetups`).

**Wall-budget implication**: with SPGMR + `PREC_NONE`, CVODE issues NO
`SUNLinSolSetup` invocations across an entire 90-day keliya run. The
existing `PrintFinalStats` reads `CVodeGetNumLinSolvSetups` (L52 in
`cvode_config.cpp`) — no temporary instrumentation was needed. For the
AMG path, the wrapper's `Setup` callback may equivalently never be
called by CVODE under matrix-free iterative LS conventions; the wrapper
MUST therefore lazy-build its AMG hierarchy at first `Solve` invocation
(or in `Initialize`) rather than relying on CVODE-issued Setup events.
This is reflected in the wrapper implementation:
- `Initialize`: does nothing except Hypre runtime init + thread pinning.
- `Solve` (first call): runs the probe-based Setup inline, builds
  `HYPRE_IJMatrix` and BoomerAMG hierarchy lazily, then invokes
  `HYPRE_BoomerAMGSolve` for the actual solve. Subsequent Solves reuse
  the cached hierarchy.
- `Setup` callback (CVODE-issued, if any): triggers a rebuild of the
  AMG hierarchy with the latest probe values. Under the measured
  cadence (nsetups=0), this path is never exercised on keliya; on
  larger cells with nontrivial nonlinear iteration, CVODE may issue
  Setup more often, and the rebuild is the design.md D4 baseline policy.

## 1.5 CVodeSetMonitorFn presence in SUNDIALS 6.0

**Confirmed PRESENT** at `SHUD/InstallSundials/include/cvode/cvode.h:141`:
`SUNDIALS_EXPORT int CVodeSetMonitorFn(void *cvode_mem, CVMonitorFn fn);`

**Decision: use driver-side `SUNLinSol_Hypre_SetStepContext` helper.**

Rationale: `CVodeSetMonitorFn` is available but its callback signature
(`CVMonitorFn`) only sees `cvode_mem` and doesn't have a direct, clean
plumbing path to the wrapper's content struct without a global registry.
A driver-side helper called from the SHUD step loop is the simpler,
more local pattern and matches design.md D6's primary description.
PR-A smoke runner integration can call
`SUNLinSol_Hypre_SetStepContext(LS, step_idx, t_sim, nli_cum, nfeLS_cum)`
before each `CVode(...)` call to populate the per-step context fields
in the ring buffer.

## 1.6 Server Hypre install matrix — SKIPPED (orchestrator scope)

Per Phase 1 brief, this is handled separately by the orchestrator
(server reachable, hypre 3.1.0 dylib confirmed at
`/scratch/frd_muziyao/local/hypre-3.1.0/lib/libHYPRE-3.1.0.so`). No
action in this PR-0 Phase 1.

## 1.7 ColPack participation

**Decision: NOT required for the integrated AMG path.**

Evidence (grep across `SHUD/src/` + `tools/p8tune.F/`):

- `SHUD/src/`: zero `ColPack`/`colpack` references — SHUD core has no
  ColPack dependency.
- `tools/p8tune.F/`: ColPack referenced in `Makefile` (linker `-lColPack`)
  and in `boomeramg_setup_solve.cpp` only for the `colpack_version=<CV>`
  KV provenance line emitted by the spike binary. The Hypre/AMG code
  path itself (lines 600–1000 of `boomeramg_setup_solve.cpp`) does not
  call any ColPack API.
- ColPack in P8-tune.F was a transitive build dep from `libshud.a`'s
  reference to graph-coloring utilities that existed in earlier rSHUD
  layers, NOT a runtime dependency of BoomerAMG.

**Consequence for PR-0**:
- Task 3.2: ColPack flag bundle is OMITTED from Makefile `LK_FLAGS` (no
  `-L${COLPACK_LIBDIR} -lColPack -Wl,-rpath,${COLPACK_LIBDIR}`).
- Task 3.8: ColPack install step is OMITTED from
  `.github/workflows/serial-baseline.yml` (orchestrator scope).
- Task 3.4: `make help` Hypre install matrix omits ColPack.

If a future G1 sweep needs ColPack for AMG coarsening tuning, it lands
in that PR via additive Makefile + workflow extensions.

---

## Summary table (one-line outcomes)

| Task | Outcome |
|---|---|
| 1.1 MD_adjacency reuse | probe-derive (topology-restricted ATimes) |
| 1.2 setup-inclusion | setup_included |
| 1.3 SUNDIALS 6.0 ABI | 15 callbacks confirmed, signatures in §1.3 |
| 1.4 CVODE Setup cadence | nsetups=0 on keliya 90-day (SPGMR PREC_NONE) → wrapper lazy-builds hierarchy at first Solve |
| 1.5 CVodeSetMonitorFn | present; driver-side SetStepContext chosen |
| 1.7 ColPack | NOT required for integrated AMG; flag bundle omitted |

(Task 1.6 deferred to orchestrator per brief.)
