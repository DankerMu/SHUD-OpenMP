# SHUD-OpenMP CPU Acceleration Release v1.0

**Tag**: `cpu-accel-v1.0`  
**Outer commit**: `d127edc` (release branch tip `release/cpu-accel-v1.0`)  
**SHUD submodule pin**: `openmp-baseline/6bae35d`  
**Released**: 2026-07-01

---

## Executive summary

This release ships the **first-phase CPU acceleration** for the SHUD fully-coupled hydrologic model, plus the standing validation infrastructure and full decision archive from the acceleration program.

Deliverables:

- **P1e StrictOMP RHS** — deterministic ~1.7× wall speedup on `heihe_x4` (40,046 elements), preserving bitwise correctness under strict tolerance.
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
make shud_omp        # production OpenMP build; HYPRE=0 default
```

Runtime:
```bash
export OMP_NUM_THREADS=<physical cores>
./shud_omp <case-name>
```

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

### Research AMG build (not for production)

```bash
cd SHUD
make shud_omp HYPRE=1 HYPRE_INCDIR=/path/to/hypre/include \
              HYPRE_LIBDIR=/path/to/hypre/lib \
              OPENBLAS_LIBDIR=/path/to/openblas/lib
export SHUD_LINSOL=amg
./shud_omp <case>
```

Not recommended for production; retained for ADR-0007 reproducibility only.

---

## Version pinning + reproducibility

### Outer

- Tag: `cpu-accel-v1.0`
- Branch: `release/cpu-accel-v1.0`
- Commit: `d127edc` (base) — check `git describe --tags` after fetch for authoritative resolution

### SHUD submodule

- Tag: `cpu-accel-v1.0` on `openmp-baseline`
- Commit: `6bae35d`
- Upstream: `https://github.com/SHUD-System/SHUD.git`

To reproduce this exact release:
```bash
git clone <outer-repo>
cd <outer-repo>
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
