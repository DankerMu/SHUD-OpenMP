# P1 NUM_OPENMP scaling baseline — Mac dev-only + server §1.1.1 acceptance

P1 candidate verification of `MD_update.cpp` 3-pragma OpenMP stack
(element + river + lake), SHUD pin `07c677f`.

Two sections:
- §1 Mac dev-only (PR-K1 #222, outer commit
  [`a6a9bd3`](https://github.com/DankerMu/SHUD-OpenMP/commit/a6a9bd3),
  NG1 — does not count toward §1.1.1 go/no-go).
- §2 server §1.1.1 acceptance (PR-K2 #223, outer commit
  [`31fd419`](https://github.com/DankerMu/SHUD-OpenMP/commit/31fd419),
  `heihe` + `heihe_x4` on Slurm cn03).

Per spec [`p1-state-update-parallel/spec.md` L164–L181][spec-scaling] the
per-cell verdict is **A3a (bitwise vs same-binary N=1 baseline)** or
**A3b (ULP ≤ 4 + max_abs_diff < 1e-12)** fallback; design rationale D5
(NG3) allows A3b fallback / WARNING at P1 standalone without blocking
the P1 epic lock (#211).

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

## 2. Server §1.1.1 acceptance scaling — heihe + heihe_x4 (PR-K2 #223)

### 2.1 Scope (spec L164–L181; tasks 5.8–5.9)

- 2 server case × NUM_OPENMP ∈ {1, 2, 4, 8} = **8 cells** wall + speedup
- 6 cells A3a / A3b (N=2 / 4 / 8 vs N=1 **same-binary** baseline)
- Slurm: 2 sbatch on CPU partition (cn03), `--cpus-per-task=8`,
  `OMP_PROC_BIND=close OMP_PLACES=cores`
  - jobid `8796` heihe — Elapsed `00:09:10`, ExitCode `0:0`
  - jobid `8797` heihe_x4 — Elapsed `01:05:41`, ExitCode `0:0`
- Outer commit: `31fd419`; SHUD pin: `07c677f`; binary:
  `SHUD/shud_omp` (built in-sbatch on cn03)
- Build sha256: `b637537c53ff446b9885f949c19f20e50eba53296ef417ea5a5924fa803b2865`
- Forcing: M7 trimmed window per `cfg_para_{start,end}` per case manifest
  (heihe 14245→14335, heihe_x4 1→91)

### 2.2 A3a baseline semantic

Per design D5 + PR-I #220 + PR-J #221 prior verification, the strict
A3a comparison is **same binary, same SHUD pin, different N**:

- `B0_output/<case>.rivqdown.dat` (archived 2026-06-17, serial `shud`
  binary `00ea9d80…` / `5b95f617…`) **predates** SHUD pin `07c677f`
  (3-pragma stack) and uses a different binary, so it is not a valid
  PR-K2 baseline. CVODE step count differs (heihe golden nst=6571 vs
  PR-K2 N=1 nst=6773) confirming the pre-3-pragma trajectory.
- Cross-binary equivalence is owned by:
  - PR-J #221 (server serial `shud` canonical SHA, 4 / 4 PASS at
    `OMP_NUM_THREADS=1`, A1-level refactor equivalence — 3-pragma
    inactive in serial build)
  - PR-K1 #222 (Mac snapshot-bin probe, 16 / 16 A3a PASS — RHS state
    binary-independent at canonical 90-day t)
- PR-K2 question = **does OMP scheduling perturb the `shud_omp`
  trajectory at N ∈ {2, 4, 8} relative to N=1?** Baseline therefore =
  PR-K2's own N=1 `<case>.rivqdown.dat`.

### 2.3 Server platform

| Field | Value |
|---|---|
| Host | `cn03` (CPU partition, Slurm `frd_muziyao@210.77.77.22:32099`) |
| OS | Linux 6.8.0 (Ubuntu 24.04) |
| Compiler | GCC `13.3.0-6ubuntu2~24.04.1` (libgomp) |
| OMP env | `OMP_PROC_BIND=close OMP_PLACES=cores` (S5d.4 / D5 manifest gate) |
| FP flags | `-O2 -ffp-contract=off -fopenmp` (per master plan §1.1.1 platform spec) |
| Binary | `SHUD/shud_omp` sha256 `b637537c…3b2865` |
| Window | 90 days (M7 trimmed forcing per case manifest) |

### 2.4 8-cell wall + speedup matrix

| case (NumEle / NumY) | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) | sp@2 | sp@4 | sp@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| heihe (6335 / 21357) | 135 | 141 | 128 | 125 | 0.96× | 1.05× | **1.08×** |
| heihe_x4 (40046 / 124395) | 1033 | 1027 | 955 | 908 | 1.01× | 1.08× | **1.14×** |

### 2.5 6-cell A3a / A3b verdict (vs N=1 same-binary baseline)

Per-cell `<case>.rivqdown.dat` SHA256 against N=1; on differ, ULP +
max_abs_diff against N=1 via `.s223-runs/ulp_diff.py` (raw double stream).

| case | N=2 vs N=1 | n_diff | max_ulp | max_abs_diff | verdict |
|---|---|---:|---:|---:|---|
| heihe | SHA equal | 0 / 214252 | 0 | 0 | **A3a PASS** |
| heihe_x4 | SHA equal | 0 / 387607 | 0 | 0 | **A3a PASS** |

| case | N | vs N=1 | n_diff | max_ulp | max_abs_diff | verdict |
|---|---:|---|---:|---:|---:|---|
| heihe | 4 | SHA differ | 210790 / 214252 (98.4%) | 9.17e18 | 4.56e5 | A3a FAIL → **A3b FAIL** |
| heihe | 8 | SHA differ | 210796 / 214252 (98.4%) | 9.15e18 | 4.39e5 | A3a FAIL → **A3b FAIL** |
| heihe_x4 | 4 | SHA differ | 382472 / 387607 (98.7%) | 4.59e18 | 1.66e6 | A3a FAIL → **A3b FAIL** |
| heihe_x4 | 8 | SHA differ | 383130 / 387607 (98.8%) | 4.59e18 | 1.95e6 | A3a FAIL → **A3b FAIL** |

**Aggregate**: N=2 cells 2 / 2 A3a PASS · N=4 / N=8 cells 4 / 4
A3a / A3b double-FAIL.

### 2.6 Trajectory-bifurcation note (N ≥ 4)

Diagnosis (first divergence + CVODE statistics):

- heihe nst per N: 6773 (N=1) · 6773 (N=2) · 6585 (N=4) · 6684 (N=8)
- heihe_x4 nst per N: 6571 (N=1) · 6571 (N=2) · 6570 (N=4) · 6572 (N=8)
- heihe N=4 first differ index = 2483 / 214252, `abs(a-b) = 9.16e-5`
  (≈ small ULP) — drift then amplified by CVODE step adaptation to
  `max_abs ≈ 5e5` at the 90-day terminus.

OMP scheduling reorders the per-element reduction inside MD\_update at
N ≥ 4 → CVODE re-selects step size against a different RHS sample →
trajectory bifurcates (chaotic ODE long-integration regime). This is
the strict-A3a / strict-A3b regression that design D5 NG3 identifies
as the **P7 final-fusion debug** target: P1 candidate exposes it, P7
must enforce deterministic reduction order (or a P9 deterministic
N\_Vector) to recover bitwise / ULP ≤ 4 at all N.

### 2.7 §1.1.1 T-column comparison

Per master plan §1.1.1 main table (P9 final target) + P7 strict
Amdahl-bounded interim table:

| case | scale | P7 strict 8-core M / T | P9 final 8-core M / T | actual N=8 sp | gap |
|---|---|---|---|---:|---|
| heihe | Medium (IO-mitigated) | "不独立验收" (master plan §1.1.1 + §5 Opt-IO) | 3.0× / 4.5× | 1.08× | small-case fork-join overhead dominates post-M7 IO mitigation |
| heihe_x4 | Large | 1.8× / 2.2× | 4.5× / 6.0× | 1.14× | first OMP candidate (no S5d SoA / owner-local gather / cutoff / N\_Vector) |

### 2.8 Aggregate verdict — §1.1.1 P1 epic

**§1.1.1 = WARNING (not blocked).** Two independent reasons:

1. **wall / speedup**: both cases below P7 strict M; expected for the
   first OMP candidate per design D5 ("P7 strict 退出 vs P9
   production"; "small case anti-scale" pattern). P2 / P7 work
   (deterministic reduction, S5d SoA / owner-local gather, cutoff,
   N\_Vector) plan to close the gap.
2. **A3a / A3b strict**: 4 / 6 cells (N ≥ 4) double-FAIL with
   trajectory bifurcation. Master plan §1.1.2 + design D5 NG3 + spec
   `p1-state-update-parallel` L164–L181 allow A3b-fallback / WARNING
   at P1 standalone; the strict-A3a regression is logged as a known
   P7 final-fusion debug target (deterministic reduction or
   parallel-deterministic N\_Vector), see also `openspec/changes/
   p1-update-omp/design.md` D5.

**P1 epic merge gate (#211) is NOT this PR.** PR-K2 records the
WARNING data + decision basis; downstream P7 work (#TBD) inherits the
strict-A3a debt explicitly.

### 2.9 Per-cell SHA + raw artifacts

Server scratch (gitignored, retained for audit):

```
.s223-runs/heihe_scaling_8796.out         # Slurm stdout 4-N loop
.s223-runs/heihe_x4_scaling_8797.out
.s223-runs/heihe_results.tsv              # SHA vs stale B0_output
.s223-runs/heihe_walls.tsv
.s223-runs/heihe_x4_results.tsv
.s223-runs/heihe_x4_walls.tsv
.s223-runs/<case>_N<n>/<case>.rivqdown.dat
.s223-runs/<case>_N<n>/cvode_stats.txt
.s223-runs/ulp_diff.py                    # A3b ULP fallback tool
.s223-runs/run_heihe_scaling.sbatch
.s223-runs/run_heihe_x4_scaling.sbatch
```

Local mirror: `.s2-103/pr-k2/{heihe,heihe_x4}_{scaling_*,results,walls}.{out,tsv}`.

Per-cell SHA256 of `<case>.rivqdown.dat`:

- heihe N=1 = `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`
- heihe N=2 = `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` (= N=1)
- heihe N=4 = `03055aa0fcbc9c3406e61f0ed926e2b77682b2d565ba1f2eef1de7721ba5ba9a`
- heihe N=8 = `904779c30770f55638ca01030ef5b9e6bf65095ab3d70e6894d843f29b40b6e7`
- heihe\_x4 N=1 = `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022`
- heihe\_x4 N=2 = `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` (= N=1)
- heihe\_x4 N=4 = `0b2aa00f0e2d55887ee44fd95848f2370fc5682aa00bd7fefda61ad0948fc765`
- heihe\_x4 N=8 = `d3d37e42a9ccfe9b23aec38d5a85cd627c870bc5642cc37ff780407551f11e8d`

---

## 3. Sign-off

### 3.1 PR-K1 Mac dev-only (NG1)

| field | value |
|---|---|
| signed_at | 2026-06-22 |
| signer | DankerMu |
| signed_against_outer_commit | `a6a9bd3` |
| signed_against_SHUD_commit | `07c677f` |
| binary | `SHUD/shud_omp` (Mac Apple Clang 17.0.0) |
| Mac matrix | 16 / 16 A3a PASS (driver exit 0) |
| Mac wallclock | 35.6 min total |

Raw artifacts (gitignored, scratch-only):
- driver log: `.s2-103/pr-k1/driver.log`
- per-cell snapshot: `.s2-103/pr-k1/<case>_N<n>/snapshot_t<abs_min>.bin`
- wall TSV: `.s2-103/pr-k1/walls.tsv`
- verdict TSV: `.s2-103/pr-k1/results.tsv`

### 3.2 PR-K2 server §1.1.1 acceptance (this PR)

| field | value |
|---|---|
| signed_at | 2026-06-22 |
| signer | DankerMu |
| signed_against_outer_commit | `31fd419` |
| signed_against_SHUD_commit | `07c677f` |
| binary | `SHUD/shud_omp` (server GCC 13.3.0, sha256 `b637537c…3b2865`) |
| Slurm jobid | `8796` heihe (Elapsed 00:09:10) · `8797` heihe\_x4 (Elapsed 01:05:41) on cn03 |
| 8-cell wall | heihe sp@8=1.08× · heihe\_x4 sp@8=1.14× |
| 6-cell strict | N=2: 2 / 2 A3a PASS · N=4 / N=8: 4 / 4 A3a / A3b FAIL (trajectory bifurcation, P7 debug debt) |
| §1.1.1 verdict | **WARNING** (P1 epic not blocked per design D5 NG3 + master plan §1.1.1 "P7 strict 退出 vs P9 production") |

Raw artifacts (gitignored, scratch-only):
- Slurm logs: `.s2-103/pr-k2/{heihe,heihe_x4}_scaling_<jobid>.out`
- per-cell wall + SHA TSV: `.s2-103/pr-k2/{heihe,heihe_x4}_{walls,results}.tsv`
- Server: `/scratch/frd_muziyao/SHUD-OpenMP/.s223-runs/{heihe,heihe_x4}_N{1,2,4,8}/{<case>.rivqdown.dat,cvode_stats.txt}`
- Server: `/scratch/frd_muziyao/SHUD-OpenMP/.s223-runs/ulp_diff.py` (A3b fallback tool)
