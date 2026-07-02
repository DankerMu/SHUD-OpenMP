# P12-nvec PR-N0 — `SHUD_NVEC_PROF` ops-wrapper profiler evidence (issue #442)

SHUD deliverable: env-gated (**strict `=1`**) NVector op-share profiler for
the coupled solver vectors. Wraps every POPULATED op pointer of `udata`/`du`'s
`ops` table at creation (before `CVodeInit`) with counting + monotonic-ns
shims that purely delegate to the original implementation; clone propagation
carries the shim table to all CVODE-internal temporaries. Dumps
`nvec_prof.csv` at run end.

SHUD branch `p12-nvec` @ `18f6319` (profiler `74026a8` + effective-N header
fix `18f6319`). Files: `src/Model/MD_nvec_prof.{hpp,cpp}` (new) +
`src/Model/shud.cpp` (include + install/clone-assert before `SetCVODE` +
dump at run end). `MD_nvec_prof.{cpp,hpp}` auto-compile via the
`src/Model/*.cpp` Makefile wildcard; the profiler builds WITH and WITHOUT
`SHUD_ENABLE_PROFILE`.

## 1. Strict `=1` gate + bitwise-neutrality legs (task 1.2) — keliya, Mac, serial `make shud`

Reproduce: `bash neutrality_legs.sh` (commit-based: builds the pre-patch
binary from parent `75afb2b`, the patched binary from `HEAD`, runs 5 legs).

Model-output SHA manifest (`sha/*.sha`, `shasum -a256`) covers every file in
`Basins/keliya/output/keliya.out/` **except**:
- `diag_*.csv` — P11-osc instrumentation artifacts;
- `nvec_prof.csv` — the P12 profiler artifact (only exists when gate on);
- `profile_B0.yaml` — `SHUD_ENABLE_PROFILE` bucket dump (profiling artifact,
  not a model output; absent in the serial neutrality build anyway);
- `*.time.csv` — wall-clock log (varies run-to-run by construction, PR-D1
  precedent).

| leg | condition | `nvec_prof.csv` | model-output SHA | verdict |
|---|---|---|---|---|
| **a** | pre-patch baseline (2 runs, deterministic), 13 files | n/a | reference (`leg-a-baseline.sha`) | PASS — deterministic (run1 == baseline) |
| **b** | patched, `SHUD_NVEC_PROF` **unset** | none created | == baseline | **PASS** |
| **c** | patched, `SHUD_NVEC_PROF=0` | none created | == baseline | **PASS** — strict `=1` proven (presence/`=0` does not enable) |
| **d** | patched, `SHUD_NVEC_PROF=1` | created | == baseline | **PASS** — pure delegation is bitwise-neutral even when enabled |

Leg c is the load-bearing strict-`=1` evidence: a presence-only or `!=0`
predicate would create the CSV here and change nothing else, but the gate is
`getenv && strcmp(v,"1")==0`, so no CSV is emitted and the SHA holds.

## 2. Clone-propagation smoke assert (task 1.2) — `selfcheck/nvec_prof_stdout.txt`

From the `=1` leg (and every profiled run), before `CVodeInit`:

```
[NVEC_PROF] clone-propagation smoke assert: base=31 N_VClone=31 N_VCloneEmpty=31 -> PASS
[NVEC_PROF] dump output/keliya.out/nvec_prof.csv : wrapped_entries=31 invoked(calls>0)=12 backend=serial NY=1785 nthreads=8
```

`base` = shims live on `udata`; `N_VClone` / `N_VCloneEmpty` = shims that
survived onto the clone (the ops table copy carried them). `base == N_VClone
== N_VCloneEmpty == wrapped_entries` (all 31 on the Serial backend) → every
CVODE-internal temporary is measured, and the debug count matches the CSV row
count exactly (spec "no unwrapped-op leakage" cross-check).

## 3. `nvec_prof.csv` pinned schema + fixed op→class mapping (task 1.1)

Header lines: `# project_name=…`, `# NY=…`, `# nthreads=…`,
`# backend=serial|openmp|hybrid`. Data rows: `op_name,op_class,calls,total_ns`.
Sample: `csv_sample/keliya_nvec_prof.csv` (serial neutrality build, so
`nthreads` falls back to the cfg `NUM_OPENMP`=8 when `_OPENMP` is undefined)
and `keliya_mac_C_n1/nvec_prof.csv` (Config C, `nthreads`=1 = effective N).

**Fixed op→class mapping table** (committed in `MD_nvec_prof.cpp`, the
`g_slot[]` registry — the audit deliverable):

| op_class | ops |
|---|---|
| **elementwise** | `nvlinearsum`, `nvconst`, `nvprod`, `nvdiv`, `nvscale`, `nvabs`, `nvinv`, `nvaddconst`, `nvcompare`; fused/array `nvlinearcombination`, `nvscaleaddmulti`, `nvlinearsumvectorarray`, `nvscalevectorarray`, `nvconstvectorarray` |
| **reduction** | `nvdotprod`, `nvmaxnorm`, `nvwrmsnorm`, `nvwrmsnormmask`, `nvmin`, `nvwl2norm`, `nvl1norm`, `nvinvtest`, `nvconstrmask`, `nvminquotient`; local `nvdotprodlocal`, `nvmaxnormlocal`, `nvminlocal`, `nvl1normlocal`, `nvinvtestlocal`, `nvconstrmasklocal`, `nvminquotientlocal`, `nvwsqrsumlocal`, `nvwsqrsummasklocal`; fused `nvdotprodmulti`, `nvwrmsnormvectorarray`, `nvwrmsnormmaskvectorarray`, `nvdotprodmultilocal` |
| **other** | `nvclone`, `nvcloneempty` (structural; carry op work but neither elementwise nor reduction) |

Classification rule: *elementwise* = output element `z[i]` is a fixed
per-element expression of inputs at the SAME index `i` (no cross-element
accumulation; bitwise-parallelizable). *reduction* = accumulates a scalar (or
per-element test) ACROSS elements (the class whose OpenMP `reduction(...)`
order varies with thread count — the Config D failure mode). The fused/array
entries are NULL by default in the SUNDIALS 6.0.0 Serial and OpenMP backends
(CVODE does not `N_VEnable*Ops` them here), so they carry `calls=0` when
present, or are absent entirely — either way there is no leakage.

## 4. Scope / boundary audit (acceptance criterion 7)

- `src/Model/f.cpp` / `src/Model/MD_rhs_core.cpp`: **zero diff** (verify:
  `git diff 75afb2b..18f6319 -- src/Model/f.cpp src/Model/MD_rhs_core.cpp`
  is empty).
- Decoupled 5-solver loop vectors (`SHUD_uncouple`, `shud.cpp` `N_VNew_Serial
  u1..u5 / du1..du5`): **NOT wrapped** — `nvec_prof_install` is called only on
  the coupled `udata`/`du` creation site.
- Composition order (design D1/D2): `install()` runs after creation and after
  any future PR-N1 hybrid override on the same vector, so shims wrap the
  EFFECTIVE table and delegate to the (overridden) pointer — PR-N0 ships no
  hybrid path, but the install site is composition-safe.

## 5. Share evidence (task 1.3) — measurement cells pinned

**N knob (recorded, matches the release cpu-accel-v1.0.2 scaling precedent
`tools/release_v1.0_omp_scaling/run_scaling_cell.sh`)**: N is set by
`export OMP_NUM_THREADS=N`; **`SHUD_RHS_THREADS` is UNSET in every leg** (it
falls back to `omp_get_max_threads()` == `OMP_NUM_THREADS`, logged as
`P1e startup: SHUD_RHS_THREADS=(unset) -> omp_set_num_threads(N)`). The cfg
`NUM_OPENMP` value only governs the (unused, Serial-NVector) thread hint in
Config C and is left at its deployed value; the effective N is what the CSV
`nthreads` header reports (the `18f6319` fix). All profile legs are built with
`SHUD_ENABLE_PROFILE=1` so `t_CVODE_raw` (the whole `CVode()` call, the share
denominator) is emitted in `profile_B0.yaml` under `extras:`.

Config C = `make shud_omp` (StrictOMP RHS + **Serial** NVector; `backend=serial`
in the CSV — Config C never links the OpenMP NVector backend). This is the
production baseline whose serial `t_CVODE_raw` remainder G-E3 decomposes.

Shares are of `t_CVODE_raw`:
`share(class) = Σ total_ns[class] / (t_CVODE_raw · 1e9)`; the non-NVector
remainder = `t_CVODE_raw − Σ(all NVector total_ns)`. Derivation:
`uv run derive_shares.py <nvec_prof.csv> <profile_B0.yaml> <label>`.

### 5a. keliya — Mac, Config C @ N=1 (informational sanity)

Artifacts: `keliya_mac_C_n1/`.

| class | calls | total_ns | share of t_CVODE_raw |
|---|---|---|---|
| elementwise | 4,097,497 | 6,598,170,000 | 25.506% |
| reduction | 1,323,850 | 7,499,291,000 | **28.989%** |
| other (clone) | 26 | 10,000 | 0.000% |
| non-NVector remainder | — | 11,771,763,589 | 45.505% |

t_CVODE_raw = 25.869 s, NY=1785, N=1. Informational only (keliya NY≈1.5k is a
correctness case, below the OpenMP-NVector viability threshold); overhead is
informational-only here per the spec.

### 5b. heihe_x4 — server Slurm, Config C @ N=1 (share leg)

Artifacts: `x4_n1_share_server/`. NY=124,395, N=1, t_CVODE_raw=1025.363 s,
run wall 1309.3 s. Job 11278 on cn21, shud_exit=0.

| class | calls | total_ns | share of t_CVODE_raw |
|---|---|---|---|
| elementwise | 545,581 | 186,097,645,461 | 18.149% |
| reduction | 276,977 | 133,673,147,259 | 13.037% |
| other (clone) | 26 | 279,095 | 0.000% |
| non-NVector remainder | — | 705,591,467,825 | 68.814% |

At N=1 the StrictOMP RHS runs single-threaded and dominates the CVode call
(non-NVector remainder 68.8%); NVector work is 31.2%.

### 5c. heihe_x4 — server Slurm, Config C @ N=16 (G-E3(ii)/(iii) share leg)

Artifacts: `x4_n16_share_server/` (= gate-on rep1 of job 11279; the 3 gate-on
reps agree to < 0.01% on reduction total_ns and < 0.15% on t_CVODE_raw, so
rep1 is representative). NY=124,395, N=16, t_CVODE_raw=400.430 s, nst=6575.

| class | calls | total_ns | share of t_CVODE_raw |
|---|---|---|---|
| elementwise | 545,581 | 199,918,838,102 | **49.926%** |
| reduction | 276,977 | 142,859,623,101 | **35.677%** |
| other (clone) | 26 | 286,199 | 0.000% |
| non-NVector remainder | — | 57,651,067,650 | 14.397% |

**N-dependence (why the gate leg is pinned to N=16, design D4/G-E3)**: from
N=1 → N=16 the StrictOMP RHS parallelizes, so t_CVODE_raw drops 1025.4 → 400.4 s
while the SERIAL NVector work stays ~N-invariant in absolute terms (reduction
133.7 → 142.9 s; elementwise 186.1 → 199.9 s) — so the NVector SHARE of
t_CVODE_raw grows sharply (reduction 13.0% → 35.7%; elementwise 18.1% → 49.9%).
This is exactly the Amdahl ceiling the epic attacks. **G-E3(iii)
t_red = 142,859,623,101 ns (142.860 s)** = the heihe_x4 Config C N=16 absolute
reduction total_ns.

### 5d. heihe_x16 — server Slurm, Config C @ N=16, SHORT 10-model-day window

Artifacts: `x16_n16_short_server/`. NY=485,250, N=16, t_CVODE_raw=152.259 s,
run wall 294.7 s (10-day window). Job 11280 on cn10, shud_exit=0.

| class | calls | total_ns | share of t_CVODE_raw |
|---|---|---|---|
| elementwise | 56,152 | 77,723,760,526 | 51.047% |
| reduction | 24,132 | 46,379,409,362 | **30.461%** |
| other (clone) | 26 | 223,964 | 0.000% |
| non-NVector remainder | — | 28,155,745,514 | 18.492% |

At NY≈485k the NVector ops are 81.5% of t_CVODE_raw, element-wise the largest
share. **G-E3(ii) heihe_x16 reduction share = 30.461%** (≥ 10%).

The heihe_x16 case is deployed at 90 days; the SHORT window is applied
**run-locally and non-destructively**: the sbatch backs up `cfg.para`, rewrites
`END = START + 10`, runs, and restores the original `cfg.para` on EXIT (via a
`trap`). The 90-day deployed asset is never left mutated (post-restore
`END-START=90` logged).

## 6. Profiler-overhead gate (task 1.3, NORMATIVE) — heihe_x4 Config C N=16

Job 11279, exclusive node cn22, same `SHUD_ENABLE_PROFILE=1` binary for both
arms (gate-off = `SHUD_NVEC_PROF` unset → shims never installed; gate-on =
`SHUD_NVEC_PROF=1`). Reproduce the verdict:
`uv run overhead_gate.py x4_n16_overhead/batch_log.txt`.

| arm | rep walls (s) | median (s) |
|---|---|---|
| gate-off (`SHUD_NVEC_PROF` unset) | 716.387 / 699.435 / 698.418 | 699.435 |
| gate-on (`SHUD_NVEC_PROF=1`) | 698.616 / 696.929 / 699.184 | 698.616 |

**overhead = (median_on − median_off) / median_off = −0.117%** → **PASS**
(≤ 2% NORMATIVE). The gate-on median is marginally below gate-off — the shim
cost (one counter increment + two monotonic-clock reads per op call) is lost
in run-to-run wall noise at NY≈124k, well within the bound. All 6 reps
shud_exit=0.

Gate: wall-clock overhead ≤ **2%** (3-run median, `SHUD_NVEC_PROF=1` vs unset,
same `SHUD_ENABLE_PROFILE=1` binary, N=16 — the same leg as the G-E3(ii)
share input). keliya/heihe_x16 overhead is informational-only.

**A-priori per-op distortion estimate (~1%)**: each shim adds one counter
increment + two monotonic-clock reads (~40–60 ns) per op call, vs ≥ µs-scale
op bodies at NY ≈ 120k → share distortion ~1% of op cost. Both bounds sit an
order of magnitude under the 10% G-E3 threshold, and G-E3 consumes SHARES, not
absolutes.

## 7. G-E3 / G-E2 input declaration

All shares below are of `t_CVODE_raw`, read from the pinned **Config C N=16**
`SHUD_ENABLE_PROFILE=1` profile legs (§5c, §5d).

- **G-E3(ii) — reduction-op share, ≥ 10% gate on EITHER case**:
  - heihe_x4 Config C N=16 reduction share = **35.677%** (§5c)
  - heihe_x16 Config C N=16 reduction share = **30.461%** (§5d)
  - Both clear the 10% threshold decisively → G-E3(ii) is satisfied on both
    cases (the gate needs only one).
- **G-E3(iii) — `t_red`**: heihe_x4 Config C N=16 absolute reduction
  `total_ns` = **142,859,623,101 ns (142.860 s)**. Feeds the Amdahl
  projection `wall_E16 / (wall_E16 − t_red·(1 − 1/16))`, computed in PR-N2
  with the measured Config E median wall at N=16. Note `t_red` is
  ~N-invariant (N=1 leg gives 133.673 s, §5b); the **pinned input is the
  N=16 value** per design D4.
- **G-E2 context — Tier-1 element-wise upside bound**: heihe_x4 Config C N=16
  elementwise share = **49.926%** (§5c); heihe_x16 = **51.047%** (§5d). This
  is the fraction of t_CVODE_raw that Tier-1 (Config E element-wise OpenMP
  NVector) can attack while keeping reductions serial → the headroom G-E2's
  1.10× ROI bar is measured against.

The Config E @ N=16 gate-on profile leg (the t_red cross-check under the
bitwise-identical Config E trajectory) lands with PR-N2 task 3.1 after PR-N1,
filed under `.review-evidence/p12-nvec/pr-n2/`.

## 8. Summary of measurement cells (all pinned per spec)

| case | endpoint | build | N | reduction share | elementwise share | t_CVODE_raw | role |
|---|---|---|---|---|---|---|---|
| keliya | Mac | Config C + PROFILE | 1 | 28.989% | 25.506% | 25.869 s | informational sanity |
| heihe_x4 | server | Config C + PROFILE | 1 | 13.037% | 18.149% | 1025.363 s | N-trend context |
| heihe_x4 | server | Config C + PROFILE | 16 | **35.677%** | 49.926% | 400.430 s | **G-E3(ii)+(iii) + G-E2** |
| heihe_x16 | server (10-day) | Config C + PROFILE | 16 | **30.461%** | 51.047% | 152.259 s | **G-E3(ii)** share-convergence |

Every leg: `SHUD_ENABLE_PROFILE=1` build, N via `OMP_NUM_THREADS`,
`SHUD_RHS_THREADS` unset (logged). Overhead gate (heihe_x4 C N=16): −0.117%
≤ 2% PASS.
