# SHUD-OpenMP — CPU Acceleration for the SHUD Hydrologic Model

OpenMP shared-memory acceleration engineering for
[SHUD](https://github.com/SHUD-System/SHUD) (Simulator for Hydrologic
Unstructured Domains), a fully-coupled, physically-based distributed
hydrologic model solved with SUNDIALS/CVODE. The SHUD source lives here as
a git submodule (`SHUD/`, branch `openmp-baseline`); this outer repo holds
the acceleration program's benchmarks, validation tooling, evidence, and
decision records.

**Current release: `cpu-accel-v1.1`** (2026-07-02, branch
`release/cpu-accel-v1.1`, paired SHUD tag/branch of the same name).
Full manifest: [RELEASE.md](RELEASE.md).

## What you get

Measured on `heihe_x4` (40,046 elements, 90-day simulation, node-exclusive
Xeon, 3-run medians; every config A5 hydrology-acceptance PASS):

| Build | N=16 wall | vs Config C | vs serial | Determinism contract |
|---|---:|---:|---:|---|
| **Config E** — `make shud_omp` (default, v1.1.1) | 492 s | 1.41× | 2.62× | bitwise-identical **to Config C** at every thread count |
| **Config C** — `make shud_omp SHUD_USE_OPENMP_NVECTOR=0` (opt-out) | 694 s | 1.00× | 1.86× | bitwise-identical at every thread count |
| **Config E2** — `make shud_omp SHUD_NVEC_DETRED=1` | **363 s** | **1.915×** | **≈3.55×** | thread-count-invariant by construction; one-time re-baselined golden (A5: NSE=1.0000, KGE=0.9999 vs C) |

Since **v1.1.1** `make shud_omp` defaults to **Config E** — bitwise-identical
to the former Config C default at every thread count and 1.41× faster @N16, a
pure Pareto upgrade (nothing validated against Config C changes). The
Serial-NVector Config C binary is a single-flag opt-out
(`SHUD_USE_OPENMP_NVECTOR=0`). Same physics, same solver, no accuracy trade —
the speedups come from deterministic parallelization of the RHS evaluation
(v1.0) and of CVODE's internal vector operations (v1.1), never from reduced
tolerances or reordered non-deterministic floating-point sums.

## Quick start

```bash
git clone --recurse-submodules https://github.com/DankerMu/SHUD-OpenMP.git
cd SHUD-OpenMP
git checkout release/cpu-accel-v1.1
git submodule update --init

cd SHUD
./configure        # downloads + builds SUNDIALS/CVODE 6.0.0 locally
make shud_omp      # Config E default (v1.1.1); opt out to Config C with
                   #   SHUD_USE_OPENMP_NVECTOR=0, or E2 with SHUD_NVEC_DETRED=1

export OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores

# The default (Config E) and Config E2 read the NVector thread count from
# your project's .cfg.para NUM_OPENMP field, fixed at load time — the env
# var above only drives the RHS layer. Set NUM_OPENMP to the SAME N or you
# leave most of the gain on the table. (Results are correct and thread-
# count-invariant either way — a mismatch only costs wall time. Config C
# opt-out builds are unaffected: they have no OpenMP NVector layer.)
#   NUM_OPENMP  8

./shud_omp <your_project>
```

Which build to pick, thread-count guidance, Slurm templates, and common
pitfalls: **[SHUD/README.md §OpenMP parallel build](SHUD/README.md)** — the
operational runbook.

## Repository map

| Path | What it is |
|---|---|
| `SHUD/` | SHUD model source (submodule, branch `openmp-baseline`) — build and run here |
| `RELEASE.md` | Release manifest: what ships, scaling profiles, build/run/validation runbook (release branches only) |
| `SHUD_openMP_master_plan.md` | The program's single authoritative roadmap (phases, gates, disciplines) |
| `docs/adr/` | Decision archive ADR-0001…0011 — what was tried, adopted, closed, deferred, and why |
| `docs/` | Per-phase evidence + academic-style summaries (P1e, P9, P11-osc, P12-nvec, …) |
| `tools/a5/` | A5 hydrology-acceptance pipeline (NSE/KGE/peak/timing/runoff/water-balance) |
| `tools/` | Profiling, RHS snapshot/compare, mesh refine, benchmark bootstrap tooling |
| `benchmarks/` | 7-case benchmark manifests + frozen baseline output archives |
| `.review-evidence/` | Raw per-PR review evidence (SHA manifests, Slurm logs, disassembly, share tables) |

## Determinism, in one paragraph

The program's hard rule: parallelism must never make results depend on
thread count. Config C/E achieve this by keeping every cross-element
floating-point reduction in a fixed serial order (element-wise work
parallelizes freely — same per-element operations in any schedule). Config
E2 parallelizes the reductions too, via fixed-tree summation whose order is
a pure function of problem size — so it is also thread-count-invariant, but
its (single, deterministic) summation order differs from C/E, which is why
it ships as an explicitly re-baselined lineage certified by a tightened A5
gate. Details: `docs/adr/0011-*.md` + `docs/p12-nvec/`.

## Branch / tag model

- `main` — development default; epics merge here via capstone PRs.
- `release/cpu-accel-v1.0`, `release/cpu-accel-v1.1` — release lines
  (RELEASE.md lives here); annotated tags `cpu-accel-v1.0`…`v1.1` pin the
  exact outer+SHUD commit pairs.
- SHUD submodule: all acceleration commits live on `openmp-baseline`
  (upstream `master` untouched); `release/cpu-accel-v1.1` + matching tags
  mirror the release points.

## Status / roadmap

CPU acceleration is **complete through v1.1** (RHS + NVector layers; the
measured Amdahl remainder is now CVODE-sequential logic). GPU offload and
solver substitution were evaluated and closed with evidence (ADR-0006/0008);
multi-node domain decomposition (P10) is design-only and deferred
(ADR-0010). Minute-scale CVODE stepping on some basins was adjudicated as
real hydrologic dynamics, not numerical oscillation (P11-osc,
`docs/p11-osc/diagnosis_verdict.md`).

## Contact / provenance

SHUD model: Lele Shu ([shud.xyz](https://www.shud.xyz/)). Acceleration
engineering: this repository's ADRs + PR history are the audit trail; every
performance number above links to committed raw evidence under
`.review-evidence/`.
