# SHUD-OpenMP CPU Acceleration Release v1.1

**Current tag**: `cpu-accel-v1.1` (code anchor: outer `4529222` + SHUD `e37e26f`; the release branch carries docs-only polish on top)  
**Initial tag**: `cpu-accel-v1.0` (outer `479526b` + SHUD `6bae35d`, immutable v1.0 launch point)  
**SHUD submodule pin**: `openmp-baseline/56defd7` (P12-nvec Config E/E2 + user-friendly README runbook)  
**Released**: 2026-07-01 (v1.0) / 2026-07-02 (v1.1 — deterministic hybrid NVector, Config E/E2 opt-in)

## Patch history

| Patch | PR | Outer commit | SHUD commit | Summary |
|---|---|---|---|---|
| v1.0   | (init) | `479526b` | `6bae35d` | Initial release manifest (RELEASE.md + double tag) |
| v1.0.1 | [#429](https://github.com/DankerMu/SHUD-OpenMP/pull/429), [#430](https://github.com/DankerMu/SHUD-OpenMP/pull/430), [#431](https://github.com/DankerMu/SHUD-OpenMP/pull/431) | `6133a41` | `f8adea4` | `make shud_omp` defaults to Config C (no compile flag); real Config C scaling table (heihe_x4 N∈{1,2,4,8,16}, A5 all PASS, sp@8 = 1.80×); tag metadata + patch history |
| v1.0.2 | [#432](https://github.com/DankerMu/SHUD-OpenMP/pull/432) | (filled at tag time) | `db4ccdb` | SHUD README `§OpenMP parallel build (v1.0.1+)` — runtime threading runbook (env-var priority, thread-count guidance, Slurm template, common pitfalls, P1e A/B/D reproducibility). Docs-only; no runtime code change. |
| **v1.1** | [#447](https://github.com/DankerMu/SHUD-OpenMP/pull/447)–[#451](https://github.com/DankerMu/SHUD-OpenMP/pull/451) (epic [#441](https://github.com/DankerMu/SHUD-OpenMP/issues/441)) | (filled at tag time) | `e37e26f` | **P12-nvec deterministic hybrid NVector** — two compile-time opt-in legs: **Config E** (`SHUD_NVEC_HYBRID=1`: OpenMP element-wise NVector + serial reduction overrides, bitwise==C at every N; E/C **1.31×@N8 / 1.41×@N16**) and **Config E2** (`SHUD_NVEC_DETRED=1`: fixed-tree deterministic reductions B=4096, cross-thread bitwise by construction, one-time re-baselined golden, A5 nse=1.0000/kge=0.9999; E2/E **1.377×@N8 / 1.369×@N16**). Net `heihe_x4` @N16: **694 s (C) → 363 s (E2) = 1.915×** (≈3.55× vs serial). Default `make shud_omp` build byte-unchanged. Authority: ADR-0011 + `docs/p12-nvec/`. |

## v1.1 — deterministic hybrid NVector (Config E / E2, opt-in)

The v1.0 Amdahl ceiling (~2×, parallel fraction ~0.51) was set by the serial
CVODE-internal NVector work (element-wise + reductions ≈ 86% of raw CVODE time
at N=16 on `heihe_x4`, measured by the PR-N0 `SHUD_NVEC_PROF` profiler). v1.1
attacks exactly that remainder with two nested opt-in legs:

- **Config E** (`make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1`)
  parallelizes element-wise ops with the stock OpenMP NVector and pins all 20
  reduction slots to SHUD-owned serial generic-API overrides
  (`SHUD_NVEC_NOOPT` codegen pin: FMA-contraction/vectorization state matched
  to the vendored library — platform-verified on Apple clang/ARM and gcc/x86).
  **Determinism: bitwise-identical to Config C at every thread count** (G-E1:
  keliya clang N∈{1,2,4,8} + heihe gcc N∈{1,8} + heihe_x4 N∈{1,8,16}, single
  rivqdown SHA across 19 runs). Adopted per G-E2 (TIER1_ADOPT, ADR-0011).
- **Config E2** (`… SHUD_NVEC_DETRED=1`) additionally parallelizes reductions
  with fixed-tree deterministic summation (compile-time block size B=4096 via
  `SHUD_NVEC_DETRED_B`; serial in-block index-order folds; fixed binary-tree
  combine = pure function of (NY, B); dynamic thread→block mapping cannot
  affect order; malloc-failure path aborts loudly rather than order-shift).
  **Determinism: cross-thread bitwise by construction**, but a **one-time
  summation-order shift vs C/E** → new golden lineage, certified by G-E4
  (bitwise chain + A4 ulp report + tightened A5 `a5_thresholds.p12_tier2.yaml`:
  nse=1.0000, kge=0.9999, peak_off=0, runoff=0.9999). A3a bitwise acceptance
  applies WITHIN the E2 lineage, never across the C/E ↔ E2 boundary
  (`docs/p12-nvec/pr_n3_rebaseline_decision.md`).

Measured (`heihe_x4`, 90-day, 3-run medians, node-exclusive homogeneous
Xeon Gold 6133, same-batch pairings):

| Config | N=8 wall (s) | N=16 wall (s) | @N16 vs C | @N16 vs serial |
|---|---:|---:|---:|---:|
| C (v1.0 default) | 724.3 | 694.3 | 1.00× | 1.86× |
| E  | 553.2 | 491.6 | 1.41× | 2.62× |
| E2 | 431.7 | 362.7 | **1.915×** | **≈3.55×** |

E2/E gain landed within **−0.42%** of the G-E3(iii) Amdahl projection
(1.3744×) — the reduction-share model validated end-to-end. Thread knob for
E/E2: NVector threads read cfg.para `NUM_OPENMP` — set it and
`OMP_NUM_THREADS` to the same N (SHUD README §Config E / E2 runbook).

Engineering notes: per-call block-partial `malloc` in E2 reductions measured
at ~0.02% of E2@N16 wall (≈248 B × ~8×10⁵ calls) — a scratch-buffer hoist was
evaluated and **rejected** (static state + full re-verification cost for an
unmeasurable gain; KISS). Config E at cfg N=1 runs a 2-thread NVector floor
(documented, informational only — gate cells N∈{8,16} are matched-threads).

---

## Executive summary

This release ships the **first-phase CPU acceleration** for the SHUD fully-coupled hydrologic model, plus the standing validation infrastructure and full decision archive from the acceleration program.

Deliverables:

- **P1e StrictOMP RHS** — deterministic parallel RHS evaluation on `heihe_x4` (40,046 elements). Measured N∈{1,2,4,8,16} scaling on release Config C binary: **1.80× @ N=8, 1.95× @ N=16** with **A5 PASS** at every thread count (NSE=KGE=1.0000, trajectory-identical to serial reference). `make shud_omp` produces the Config C binary by default — no compile flags required. See §Scaling profile.
- **`SHUD_SPGMR_MAXL` small-case opt-in** — an env-driven knob for keliya-like small cases.
- **A5 hydrology-acceptance pipeline** (`tools/a5/`) — reusable NSE/KGE/peak/timing/runoff validator.
- **Decision archive** — ADR-0001 through ADR-0010 documenting what was tried, what was retained, what was closed, and what is deferred.

Everything else in the tree is either infrastructure (benchmarks, snapshot tools) or research trail retained for documentation value. Research artifacts are all env-gated and **inert by default** — no runtime impact when the corresponding env vars are unset.

---

## What ships (production paths)

### Runtime — SHUD-OpenMP core

- **P1e StrictOMP RHS** (SHUD `226e3ab` + follow-ups): OpenMP parallelization of the right-hand-side evaluation with owner-local gather + hot-field SoA + reordered element passes. Deterministic under strict tolerance; A3a bitwise vs B1b baseline @ 4 threads verified.
- **`SHUD_SPGMR_MAXL` env knob** (SHUD `6ce17d6`): overrides CVODE SPGMR `maxl` parameter. Default unset preserves baseline `maxl=5`. Documented improvement on `keliya` (484 elements) and similar small cases; **not recommended** as a global default (large-case effect is null-to-negative and correctness envelope is A5-informational only).
- **Memory hygiene** (SHUD `197269e`, `1ab61c0`, `056a1dc`, `710c00a`): Model_Data leak chain closure (5 new `delete[]` in `FreeData()`) + NSDMI nullptr defaults for raw pointer members + activated `_Lake` / `LakeBathymetry` dtor chain. Keliya B0 bitwise-neutral pre/post fix.

### Validation infrastructure

- **`tools/a5/`** (PR #422 + #425): standalone Python/uv project computing 7 hydrology metrics — NSE, KGE, peak magnitude, peak timing offset, total runoff volume ratio, monthly bias MAE, water balance residual (informational). YAML-driven thresholds. Machine-readable `MARKER:A5_VERDICT` output. 61/61 pytest coverage.
- **`tools/compare_snapshot/`**: bitwise-equivalent snapshot comparison for CI + local verification.
- **`tools/archive_b0_output.sh`**: 3-run bitwise archive helper for baseline capture.
- **`benchmarks/`**: 7 NWM case manifests (keliya / xinanjiang_upstream / qinyijiang / kashigeer / qhh / tailanhe / heihe [+ heihe_x4 40k / heihe_x16 252k on server]) + B0 archives + profile YAMLs.

### Docs of record (decision archive)

- `docs/adr/0001..0010` — full ADR series (macro orthogonality, KLU vs preconditioned SPGMR, hot-field SoA, AMG NO-GO, P8 closure, P9 closure, CPU program status).
- `docs/status_matrix.md` — canonical status board.
- `docs/b0_summary.md`, `docs/b1a_summary.md`, `docs/profile_decision.md`, `docs/case_deployment_map.md`, `docs/build_manifest.md` — living operations docs.
- `docs/p1e/p1e_academic_summary.md` — production baseline academic record.
- `SHUD_openMP_master_plan.md` — living program roadmap.
- `openspec/glossary.md` — canonical vocabulary.

### Ancillary

- `Script/`, `rAnalysis/`, `figures/` — analysis helpers (rSHUD-based post-processing, not runtime-critical).
- `.github/workflows/` — CI (asan-ubsan, build-and-compare, tools-tests).

---

## Scaling profile (`heihe_x4`, 40,046 elements, 90-day, SPGMR, Config C)

Measured 2026-07-01 on node-exclusive Linux `cn04/05/14/15/16` (Intel Xeon, 40 physical
cores/node), Slurm array 10764. Binary built with `make shud_omp HYPRE=1 …` — release
default Config C (Serial NVec + StrictOMP RHS). Every non-baseline cell validated
against the N=1 serial reference via the A5 hydrology-acceptance pipeline. Aggregator
verdict: **CONDITIONAL** — "all N A5=PASS and 1.5 ≤ speedup(N=8) < 2.5× — parallel
gain OK but modest".

| N (threads) | wall total (s) | speedup | efficiency | A5 verdict | NSE / KGE |
|---:|---:|---:|---:|:---|:---|
| 1  | 1317 | 1.000× |  100%  | reference | — |
| 2  |  973 | 1.353× |  67.7% | PASS | 1.0000 / 1.0000 |
| 4  |  824 | 1.598× |  40.0% | PASS | 1.0000 / 1.0000 |
| 8  |  730 | 1.804× |  22.6% | PASS | 1.0000 / 1.0000 |
| 16 |  677 | 1.946× |  12.2% | PASS | 1.0000 / 1.0000 |

CVODE solve invariants across all thread counts: `nst=6575, ncfn=51, ncfl=3620` —
identical trajectory, confirming StrictOMP is strictly deterministic (see ADR-0002).

**Interpretation.** Amdahl parallel fraction ≈ 0.51 back-solved from
`sp@16 = 1/(1−f + f/16) = 1.946`; residual sequential work (CVODE outer, non-RHS
kernels) caps the theoretical ceiling near 2×. Rationale for shipping despite
CONDITIONAL: the workload is Amdahl-bound (not a defect), determinism is exact
(A5 PASS at every N), and the alternative is single-threaded execution. Sp@8 =
1.80× closely tracks the P1e capstone measurement of 1.729× (heihe_x4, Phase 2
mode C, 36-cell design of experiments; see `docs/p1e/p1e_academic_summary.md`
§Configuration Matrix, Table 6).

**Evidence**: `.review-evidence/release-v1.0-scaling-configC/` — per-cell Slurm logs
(`slurm-10764_{0..4}.out`), CVODE stats (`cell-*/cvode_stats.txt`), A5 reports
(`a5-report-nthreads-{2,4,8,16}/`), and full aggregator output
(`scaling_verdict.txt`, `MARKER:RELEASE_V1_0_SCALING_VERDICT` block).

---

## What is NOT for production use (research trail — retained for documentation)

The following are present in this release **for reproducibility and documentation value only**. They are env-gated and produce zero runtime effect when the corresponding env vars are unset.

### Research SHUD env hooks (default-compat, inert by default)

- `SHUD_LINSOL=amg` (default unset → `spgmr`) — routes CVODE through the experimental Hypre BoomerAMG linear solver wrapper. Closed via ADR-0007 + ADR-0008.
- `SHUD_AMG_TOL` — Hypre BoomerAMG solve tolerance override. Only meaningful under `SHUD_LINSOL=amg`.
- `SHUD_CVODE_EPSLIN` — CVODE `EpsLin` linear convergence safety factor override.
- `SHUD_CVODE_RELTOL` — CVODE `reltol` override. Closed via ADR-0009 (2.5% ceiling on heihe_x4 does not clear 1.2× productionization gate).

All four hooks: env unset → bitwise-identical baseline. Malformed env values → `exit(2)` with clear stderr.

### Research tooling (opt-in only)

- `tools/p8tune.D/` — pattern-only structural AMG audits (KLU + ColPack dependency). Requires SuiteSparse.
- `tools/p8tune.G0/` — G0 6-gate integrated AMG smoke framework + Slurm array + telemetry.
- `tools/p8tune.G0-rca/` — G0-RCA 8-cell heihe_x4 tolerance sweep.
- `tools/p9.spot/` — P9 2-cell heihe_x4 spot check + A5 wiring + verdict aggregator.

None of these tools are invoked by the production pipeline. They exist to reproduce the ADR-0007/0008/0009 evidence tables.

### Research documentation + evidence

- `docs/p8tune/`, `docs/p9/`, `docs/adr/0007/0008/0009/0010.md` — research retrospectives + closure ADRs.
- `.review-evidence/` — evidence directories for reviewed PRs (byte-identical MARKER blocks + Slurm logs + A5 reports).
- `openspec/` — OpenSpec change + spec artifacts (per stage-change-pipeline).

---

## Explicit non-goals (out of scope for v1.0 and forward)

Per ADR-0010:

- **GPU acceleration** — not pursued in this release cycle or any future one without an explicit reversal of the CPU-only strategic direction.
- **Further global linear solver substitution on large cases** — the P8 line (SPGMR-preconditioned / KLU / AMG) is closed. Reopening requires new external evidence.
- **Further P9 reltol-family sweeps** at current `cfg.para reltol=1e-4` — bounded at ~2.5% by PR-Z1 evidence.

---

## Deferred (design-only, requires dedicated future planning turn)

- **P10 CPU domain decomposition** — the only remaining CPU-side acceleration path with a plausible non-trivial ceiling. Gated by ADR-0010 §Deferred criteria (P10-0 design-only feasibility spike + ≥1.5× credible A5-passing ceiling + explicit engineering commitment). This release does NOT authorize P10 implementation; it documents the gate.

---

## Build and run

### Requirements

- C++ compiler with OpenMP support (GCC ≥ 9, Clang ≥ 12, or AppleClang)
- SUNDIALS/CVODE 6.0.0 (via `SHUD/configure` — auto-downloads to `SHUD/InstallSundials/`)
- OpenBLAS (for the research AMG path only; production build does not need it)
- Optional: Hypre 3.1.0 (research AMG path only)

### Production build

```bash
cd SHUD
./configure          # downloads SUNDIALS + CVODE 6.0.0
make shud_omp        # production Config C: Serial NVec + StrictOMP RHS
```

`make shud_omp` produces the release Config C binary out of the box —
Serial NVector + StrictOMP RHS (ADR-0002 Path 1 winner). No compile-time
flags to remember.

Runtime:
```bash
export OMP_NUM_THREADS=<physical cores>
./shud_omp <case-name>
```

Optional: `SHUD_RHS_THREADS=<n>` overrides the RHS thread count explicitly
(default falls back to `omp_get_max_threads()`, which honors `OMP_NUM_THREADS`).

For `SHUD_SPGMR_MAXL` opt-in on small cases:
```bash
export SHUD_SPGMR_MAXL=30
./shud_omp keliya
```

### Validation

Bitwise gate (per case):
```bash
tools/archive_b0_output.sh <case> 3   # 3-run bitwise archive
```

Snapshot gate:
```bash
tools/compare_snapshot/compare_snapshot <golden.bin> <new.bin>
```

Hydrology-acceptance (streamflow trajectory equivalence):
```bash
cd tools/a5
uv sync
uv run a5 --reference <ref/output/case.out> \
          --candidate <cand/output/case.out> \
          --config config/a5_thresholds.default.yaml \
          --case-name <case> \
          --out <report-dir>
```

Emits `a5_metrics.json` + `a5_verdict.md` + `MARKER:A5_VERDICT` block.

### Research builds (not for production)

**AMG path** (ADR-0007/0008 evidence reproduction, requires Hypre + OpenBLAS):
```bash
cd SHUD
make shud_omp HYPRE=1 HYPRE_INCDIR=/path/to/hypre/include \
              HYPRE_LIBDIR=/path/to/hypre/lib \
              OPENBLAS_LIBDIR=/path/to/openblas/lib
export SHUD_LINSOL=amg
./shud_omp <case>
```

**P1e A/B/D configuration matrix** (ADR-0002 reproducibility):
```bash
make shud                                          # Config A (canonical serial)
make shud_omp SHUD_ENABLE_OPENMP_RHS=0             # Config A/B (serial RHS via shud_omp target)
make shud_omp SHUD_USE_OPENMP_NVECTOR=1            # Config D (Serial NVec off, OpenMP NVec on, StrictOMP RHS on)
make shud SHUD_ENABLE_OPENMP_RHS=1                 # Config C via shud target (equivalent to make shud_omp default)
```

Not recommended for production; retained for ADR-0007 reproducibility only.

---

## Version pinning + reproducibility

### Outer

- Tags: `cpu-accel-v1.0` (initial, `479526b`) and `cpu-accel-v1.0.1` (current, `181b8a1`)
- Branch: `release/cpu-accel-v1.0`
- `git describe --tags` after fetch resolves whichever tag applies

### SHUD submodule

- Tags: `cpu-accel-v1.0` (initial, `6bae35d`) and `cpu-accel-v1.0.1` (current, `f8adea4`) on `openmp-baseline`
- Upstream: `https://github.com/SHUD-System/SHUD.git`

To reproduce the current patch release (v1.0.1 — recommended):
```bash
git clone <outer-repo>
cd <outer-repo>
git checkout cpu-accel-v1.0.1
git submodule update --init --recursive
cd SHUD && git checkout cpu-accel-v1.0.1
```

To reproduce the immutable v1.0 launch point (older Config B `shud_omp` semantics):
```bash
git checkout cpu-accel-v1.0
git submodule update --init --recursive
cd SHUD && git checkout cpu-accel-v1.0
```

---

## Change log (PR history since P1e capstone)

| PR | Merge SHA | Purpose |
|---|---|---|
| #398 | (P1e capstone) | P1e StrictOMP RHS production baseline |
| #401 sub-tasks 1+2 (via #419) | `3a307b4` | SHUD Model_Data leak chain + `_Lake` dtor activation |
| #414 | `a490d80` | P8-tune.G0 PR-0 — SUNLinSol_Hypre wrapper + selector |
| #415 | `afb09f7` | P8-tune.G0 PR-A — 4-cell AMG integrated smoke |
| #416 | `2e0b750` | P8-tune.G0 PR-B — HYPRE Axis-4 telemetry |
| #417 | `db6672f` | P8-tune.G0 PR-C — G0 NO-GO verdict + ADR-0007 amendment |
| #418 | `33ff296` | P8-tune.G0 PR-E — capstone merge (NOT squash) |
| #420 | `39bd6b1` | PR-X1 — AMG_TOL + CVODE_EPSLIN 8-cell RCA (`amg_reopens=false`) |
| #421 | `e0191db` | PR-X2 — ADR-0008 P8 solver-substitution CLOSED-FINAL |
| #422 | `d0727dd` | PR-Y1 — A5 validation pipeline |
| #423 | `6cfbf8e` | PR-Z1 — `SHUD_CVODE_RELTOL` hook + 2-cell spot check tooling |
| #424 | `880fe0e` | PR-Z2 — ADR-0009 P9 CLOSED (wall_speedup=1.025) |
| #425 | `2c084ce` | PR-Y2 — A5 water_balance unit-consistent + NaN fallback |
| #426 | `d127edc` | ADR-0010 — CPU acceleration status consolidation + P10 decision |

Full ADR trail: `docs/adr/0001..0010`. Retrospectives: `docs/p1e/`, `docs/p8tune/`, `docs/p9/`.

---

## Support + maintenance policy

- **Bug fixes** to production paths (P1e RHS, `SHUD_SPGMR_MAXL`, A5 pipeline, memory hygiene) may be backported from `main` to `release/cpu-accel-v1.0` on request via cherry-pick.
- **Feature additions** (new research directions, GPU exploration, P10 implementation) will land on `main` and follow the standard PR pipeline — they will NOT be backported to v1.0.
- **Research trail retention**: research artifacts (`tools/p8tune.*/`, `tools/p9.spot/`, `docs/p8tune/`, `docs/p9/`, `.review-evidence/`) are retained on this release branch as-is. They are documentation, not runtime.
