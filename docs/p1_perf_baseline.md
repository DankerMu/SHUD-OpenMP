# P1 NUM_OPENMP scaling baseline — Mac dev-only + server §1.1.1 acceptance

P1 candidate verification of `MD_update.cpp` 3-pragma OpenMP stack
(element + river + lake), SHUD pin `07c677f` on outer commit
[`a6a9bd3`](https://github.com/DankerMu/SHUD-OpenMP/commit/a6a9bd3).

Two sections — Mac dev-only (this PR, NG1) and server §1.1.1 acceptance
(`heihe` + `heihe_x4`, deferred to PR-K2 / #223).

Per spec [`p1-state-update-parallel/spec.md` L164–L181][spec-scaling] the
per-cell verdict is **A3a (bitwise vs B1b golden)** or **A3b (ULP ≤ 4 +
max_abs_diff < 1e-12)**; design rationale D5 (NG3) allows A3b fallback
without blocking P1 lock.

[spec-scaling]: ../openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md

---

## 1. Mac dev-only scaling (NG1: not counted toward §1.1.1 go/no-go)

### 1.1 Scope

- Mac local 4 case × NUM_OPENMP ∈ {1, 2, 4, 8} = **16 cell**
- per cell: wall (s) + speedup vs N=1 + canonical snapshot SHA-256
  diff vs `benchmarks/<case>/B0_output/snapshot_t7776000.bin`
- Verdict per cell: A3a (bitwise) preferred; A3b (ULP ≤ 4 + max_abs <
  1e-12) fallback
- Driver: [`.s2-103/pr-k1/run_pr_k1_mac_scaling.sh`](../.s2-103/pr-k1/run_pr_k1_mac_scaling.sh)
- Outer commit: `a6a9bd3`; SHUD pin: `07c677f`

### 1.2 Mac platform

| Field | Value |
|---|---|
| CPU | Apple M4 Pro |
| Logical / physical cores | 14 / 14 |
| Compiler | Apple Clang 17.0.0 (clang-1700.6.3.2) |
| Target triple | arm64-apple-darwin24.6.0 |
| OMP runtime | libomp via `-Xpreprocessor -fopenmp` |
| Binary | `SHUD/shud_omp` |
| Window | 90 days (project-level `END = START + 90` truncation) |

> NG1 note: Mac is a dev/CI reference platform. §1.1.1 go/no-go
> acceptance numbers are sourced exclusively from the server
> `cn0X` / `heihe`+`heihe_x4` runs (see §2, populated by PR-K2 #223).

### 1.3 Matrix (16 cells)

| case (NumY) | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) | N=2 speedup | N=4 speedup | N=8 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| keliya (484) | 59 | 59 | 68 | 173 | 1.00× | 0.87× | 0.34× |
| xinanjiang_upstream (801) | 8 | 7 | 9 | 27 | 1.14× | 0.89× | 0.30× |
| qinyijiang (3155) | 243 | 241 | 243 | 482 | 1.01× | 1.00× | 0.50× |
| qhh (4773, +lake) | 66 | 60 | 73 | 97 | 1.10× | 0.90× | 0.68× |

### 1.4 Per-cell verdict (16 cells, A3a/A3b)

Each cell compares the case's canonical 90-day snapshot
(`snapshot_t<rel_sec>.bin` written via `SHUD_DUMP_T_VALUES`
hook at absolute minute = `(START + 90) * 1440`) against the
B1b-tag archived golden `benchmarks/<case>/B0_output/snapshot_t7776000.bin`.

| case | N | wall (s) | speedup | A3a/A3b | max_ulp | max_abs_diff | notes |
|---|---:|---:|---:|---|---:|---|---|
| keliya | 1 | 59 | 1.00× | A3a PASS | 0 | 0 | baseline |
| keliya | 2 | 59 | 1.00× | A3a PASS | 0 | 0 | — |
| keliya | 4 | 68 | 0.87× | A3a PASS | 0 | 0 | small-case OMP overhead |
| keliya | 8 | 173 | 0.34× | A3a PASS | 0 | 0 | severe oversubscription, bitwise still holds |
| xinanjiang_upstream | 1 | 8 | 1.00× | A3a PASS | 0 | 0 | baseline |
| xinanjiang_upstream | 2 | 7 | 1.14× | A3a PASS | 0 | 0 | — |
| xinanjiang_upstream | 4 | 9 | 0.89× | A3a PASS | 0 | 0 | — |
| xinanjiang_upstream | 8 | 27 | 0.30× | A3a PASS | 0 | 0 | smallest case, dominated by OMP launch + barrier |
| qinyijiang | 1 | 243 | 1.00× | A3a PASS | 0 | 0 | baseline |
| qinyijiang | 2 | 241 | 1.01× | A3a PASS | 0 | 0 | — |
| qinyijiang | 4 | 243 | 1.00× | A3a PASS | 0 | 0 | — |
| qinyijiang | 8 | 482 | 0.50× | A3a PASS | 0 | 0 | OMP_CUTOFF-relevant regime |
| qhh | 1 | 66 | 1.00× | A3a PASS | 0 | 0 | baseline (lake case) |
| qhh | 2 | 60 | 1.10× | A3a PASS | 0 | 0 | best Mac speedup, +lake loop |
| qhh | 4 | 73 | 0.90× | A3a PASS | 0 | 0 | — |
| qhh | 8 | 97 | 0.68× | A3a PASS | 0 | 0 | — |

**Aggregate**: 16 / 16 cells **A3a PASS** · 0 / 16 A3b fallback · 0 FAIL.

> Bitwise determinism across N ∈ {1, 2, 4, 8} on all four Mac cases
> is the **strong outcome** anticipated by design D5: per-element /
> per-river / per-lake updates carry no cross-iteration reductions
> in the current 3-pragma stack, so the OMP scheduling permutation
> does not perturb the floating-point trajectory at all (max_ulp = 0
> across 16 / 16 cells against B1b golden).

### 1.5 Observations

- **A3a holds universally on Mac.** All 16 (case, threads) snapshots
  are byte-identical to the B1b-tag canonical 90-day golden. No A3b
  fallback or "P7 final-fusion debug" annotation is needed for the
  Mac side; P1 candidate matches design D5's "ideal" lane.
- **Speedup is sub-linear and frequently anti-scaling on Mac.** Best
  speedup is qhh N=2 (1.10×); xinanjiang_upstream and keliya degrade
  to 0.30–0.34× at N=8. Roots are mundane:
  - Mac M4 Pro has 4P + 10E cores; once `OMP_NUM_THREADS` crosses the
    P-core count the OS schedules workers across mixed core types,
    so per-iteration latency variance dominates a per-element loop.
  - keliya (484 elements), xinanjiang_upstream (801) and the qinyijiang
    (3155) N=8 cell are sub-OMP_CUTOFF territory — `#pragma omp
    parallel for` overhead dwarfs the saved kernel time.
  - libomp on darwin does not honor `OMP_PROC_BIND` reliably without
    `OMP_PLACES` (D5 explicitly defers this knob to the server side
    via `run_omp.sh`).
- **NG1 is the right gate.** Mac numbers are useful for catching
  regressions in *bitwise* identity (A3a) and as a smoke-screen for
  the loop structure; they do not predict server speedup. §1.1.1
  go/no-go decisions wait on PR-K2 (#223) server `heihe` + `heihe_x4`
  scaling.

### 1.6 Aggregate wallclock cost

| segment | wall (s) | walls (min) |
|---|---:|---:|
| keliya × 4 | 59 + 59 + 68 + 173 | 6.5 |
| xinanjiang_upstream × 4 | 8 + 7 + 9 + 27 | 0.85 |
| qinyijiang × 4 | 243 + 241 + 243 + 482 | 20.2 |
| qhh × 4 | 66 + 60 + 73 + 97 | 4.9 |
| 16-cell driver total | 2135 s | **35.6 min** |

(Excludes per-case `fix_case_paths` + `forcing_trim` setup overhead,
~10 s × 4 = 40 s.)

---

## 2. Server §1.1.1 acceptance scaling — placeholder (PR-K2 #223)

This section is **intentionally empty** in PR-K1; it will be populated
by PR-K2 #223 with:

- `heihe` (NumY ≈ 6335) × NUM_OPENMP ∈ {1, 2, 4, 8} (4 cells)
- `heihe_x4` (NumY ≈ 25k, generated from heihe 4× rSHUD `a/4`
  refinement; see CLAUDE.md / §1.1.1) × NUM_OPENMP ∈ {1, 2, 4, 8}
  (4 cells)
- per-cell wall + speedup + Slurm jobid + A3a / A3b verdict
- 8-core speedup comparison vs §1.1.1 T-column targets (8-row table
  in tasks 5.9)
- delivered out of `/scratch/frd_muziyao/SHUD-OpenMP` under
  `cn0X` (CPU partition) per Slurm 3 rules in CLAUDE.md
- Slurm scripts: separate sbatch per case (5.8a heihe / 5.8b heihe_x4)

> §1.1.1 go/no-go is decided by **server** numbers; this PR's Mac
> matrix is dev-only (NG1) and does not affect the gate.

---

## 3. Sign-off

| field | value |
|---|---|
| signed_at | 2026-06-22 |
| signer | DankerMu |
| signed_against_outer_commit | a6a9bd3 |
| signed_against_SHUD_commit | 07c677f |
| binary | SHUD/shud_omp |
| Mac matrix | 16 / 16 A3a PASS (driver exit 0) |
| Mac wallclock | 35.6 min total |
| server matrix | (pending PR-K2 #223) |

Raw artifacts (gitignored, scratch-only):
- driver log: `.s2-103/pr-k1/driver.log`
- per-cell snapshot: `.s2-103/pr-k1/<case>_N<n>/snapshot_t<abs_min>.bin`
- wall TSV: `.s2-103/pr-k1/walls.tsv`
- verdict TSV: `.s2-103/pr-k1/results.tsv`
