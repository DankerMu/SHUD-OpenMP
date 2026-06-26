# PR-2 (S2.6 + S2.9) — Mac-local validation evidence

Issue: DankerMu/SHUD-OpenMP#145 (PR-2 of `b1a-finalization`)
Outer branch: `feat/issue-145-pr2-s2-6-s2-9` (from `baseline/B1a` 4c693e4)
SHUD branch: `openmp-baseline` 75e32e0 -> <new SHA, bumped after submodule push>
File touched: `SHUD/src/ModelData/MD_f_omp.cpp` (+12 / -7, net +5 lines incl. 4 comment lines)

## Scope reminder — dormant TU

`MD_f_omp.cpp` is **filter-out excluded** from default Makefile compile list
(`SHUD/Makefile:369`). Under `SHUD_LEGACY_OMP_RHS=1` build the TU compiles
in and the 3 `_omp` symbols become linkable, but they are **not reachable
at runtime from `f()`** — the legacy-OMP dispatch was removed in S1d.2.
PR-2 changes are structural alignment for P1+ OMP RHS re-activation:

- **S2.6**: 3 negative-state clamp ternaries (`uYsf`/`uYus`/`uYriv`) removed
  in `f_update_omp()`. Aligns dormant OMP path with serial `f_update()`
  (`MD_update.cpp:73-77`) no-clamp semantics.
- **S2.9**: 4 vars (`area`/`isf`/`ius`/`igw`) in `f_applyDY_omp()` moved
  from parallel-region outer scope into for-body declare-at-use (Form 1 per
  PR-2 brief — matches PR-1 #144 declare-at-use pattern). Race-free.

`uYgw = max(0.0, Y[iGW])` at `f_update_omp` L124 (iBC == 0 branch)
intentionally **NOT** touched — explicitly out-of-scope per spec S2.6
amended (iBC-branch quirk; defer).

## Form chosen for S2.9

**Form 1 (declare-at-use)**. Rationale: matches PR-1 #144 (`MD_ET.cpp` 16
element-local scalars moved into for-body). Consistent style across S2 PRs.

## Edits — file diff (final state vs pre-PR-2)

`f_applyDY_omp()` (function head):

```c++
// pre-PR-2 (L9-17):
void Model_Data::f_applyDY_omp(double *DY, double t){
    double area;
    int isf, ius, igw, i;
#pragma omp parallel  default(shared) private(i) num_threads(CS.num_threads)
    {
#pragma omp for
        for (i = 0; i < NumEle; i++) {
            isf = iSF; ius = iUS; igw = iGW;
            area = Ele[i].area;

// post-PR-2 (L9-20):
void Model_Data::f_applyDY_omp(double *DY, double t){
    // S2.9: dormant path race fix — area / isf / ius / igw moved into
    // for body so each thread iteration owns its own copies (implicitly
    // private per OpenMP per-iteration semantics). Matches PR-1 #144
    // declare-at-use pattern in MD_ET.cpp.
    int i;
#pragma omp parallel  default(shared) private(i) num_threads(CS.num_threads)
    {
#pragma omp for
        for (i = 0; i < NumEle; i++) {
            int isf = iSF, ius = iUS, igw = iGW;
            double area = Ele[i].area;
```

`f_update_omp()` element pass (L116-117 → post-PR-2 L119-121):

```c++
// pre-PR-2:
            uYsf[i] = (Y[iSF] >= 0.) ? Y[iSF] : 0.;
            uYus[i] = (Y[iUS] >= 0.) ? Y[iUS] : 0.;

// post-PR-2:
            // S2.6: dormant path aligned with serial f_update() (MD_update.cpp:73-74) — no clamp.
            uYsf[i] = Y[iSF];
            uYus[i] = Y[iUS];
```

`f_update_omp()` river pass (L153 → post-PR-2 L157-158):

```c++
// pre-PR-2:
            uYriv[i] = (Y[iRIV] >= 0.) ? Y[iRIV] : 0.;

// post-PR-2:
            // S2.6: dormant path aligned with serial f_update() no-clamp semantics.
            uYriv[i] = Y[iRIV];
```

`f_update_omp()` L124 `uYgw[i] = max(0.0, Y[iGW]);` — **NOT** changed
(out-of-scope per spec S2.6 amended).

## Step 5 — Grep gates (PR-2 Invariant Matrix regression rows)

```
$ awk '/^void Model_Data::f_update_omp\(/,/^}$/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '\(Y\[i(SF|US|RIV)\] >= 0\.\) \?'        ->  0   (SHALL = 0)
$ awk '/^void Model_Data::f_update_omp\(/,/^}$/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '^[[:space:]]*uY(sf|us|riv)\[i\][[:space:]]*=[[:space:]]*Y\[i(SF|US|RIV)\];' \
                                                          ->  3   (SHALL >= 3)
$ awk '/^void Model_Data::f_applyDY_omp\(/,/#pragma omp parallel/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '^[[:space:]]*(double|int)[[:space:]]+(area|isf|ius|igw)\b' \
                                                          ->  0   (SHALL = 0)
```

All three PR-2 Invariant Matrix grep rows match spec.

## Step 5b — S1 4-grep gate (regression-protected)

```
$ grep -rn '_OPENMP_ON'                              SHUD/src/ | wc -l         ->  0
$ grep -rn 'USE_RHS_CORE'                            SHUD/src/ | wc -l         ->  0
$ grep -rn 'N_VDestroy_Serial'                       SHUD/src/ | wc -l         ->  0
$ grep -rn 'SHUD_USE_OPENMP_NVECTOR' SHUD/src/ | grep -v Macros.hpp | wc -l    ->  7
```

7 `SHUD_USE_OPENMP_NVECTOR` hits outside `Macros.hpp` are the pre-existing
S1d.3-introduced dispatch + 6 narrative comments (Model_Control.cpp,
CommandIn.cpp, Model/f.cpp, Model/shud.cpp) — identical to PR-1 baseline,
not introduced by PR-2. Recorded as audit, not a regression.

## Step 1 — Config A (default `make shud`, LEGACY_RHS=0)

Build: `make clean && make shud`  -> PASS (pre-existing sprintf deprecation
warnings only; same as pre-PR-2 baseline).

4-case + qhh lake `rivqdown.dat` SHA256 vs `benchmarks/<case>/B0_output`:

```
keliya              89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc  PASS
xinanjiang_upstream 3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020  PASS
qinyijiang          48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57  PASS
qhh                 d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce  PASS
```

All 4 hashes byte-equal PR-1 evidence (commit 64569b3 baseline). 0 runtime
impact on default build confirmed.

qhh lake-related .dat (3 archived files per `benchmarks/qhh/manifest.yaml`):

```
qhh.lakystage.dat     PASS  4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250
qhh.lakqrivin.dat     PASS  1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d
qhh.lakqrivout.dat    PASS  1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d
```

CVODE 15-key (no nFCall) via `tools/cvode_stats_diff/cvode_stats_diff.sh`:

```
keliya              exit=0  PASS
xinanjiang_upstream exit=0  PASS
qinyijiang          exit=0  PASS
qhh                 exit=0  PASS
```

**Config A verdict: PASS** — confirms PR-2 has 0 runtime impact on default
build (`MD_f_omp.cpp` filter-out excluded → all changes dormant).

## Step 2 — Config B (`make shud_omp` + `OMP_NUM_THREADS=1`)

Build: `make clean && make shud_omp`  -> PASS.

4-case `rivqdown.dat` SHA256 vs `benchmarks/<case>/B0_output`:

```
keliya              FAIL  new=b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a  ref=89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc
xinanjiang_upstream FAIL  new=90eeb9c63c07e8db3a051482f51cdc274f6e469326b19575efed54181258bf45  ref=3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020
qinyijiang          FAIL  new=0f8c3fecfe1e618f24ee5411ff9c7ea80381e953e6aafac6a1a96fe666018ced  ref=48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57
qhh                 FAIL  new=38ef0414d9ffa9310828183ce769db97c8c6c1e8714d71330bae78c0a2c0641c  ref=d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce
```

**Config B verdict: pre-existing baseline gap, NOT introduced by PR-2.**

Verified by reverting PR-2 edits (`git stash` on `SHUD/src/ModelData/MD_f_omp.cpp`)
and rebuilding `make shud_omp` from clean `openmp-baseline` HEAD `75e32e0`
(pre-PR-2 baseline). The keliya hash on pre-PR-2 baseline:

```
keliya (pre-PR-2 baseline, shud_omp + T=1):
  b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a   (identical to PR-2 build)
```

Identical hash on both PR-2 and pre-PR-2 baselines confirms PR-2 has **zero
contribution** to the Config B gap. The gap is between `Config A` (serial
NVector backend, used to produce B0 archive) and `Config B` (OpenMP NVector
backend, single-threaded) — a pre-existing infrastructure gap inherent to
the NVector backend switch, unrelated to PR-2's dormant-TU edit.

Spec L146 SHALL clause ("Config B 4-case rivqdown.dat SHALL bitwise = B0-tag")
is a spec-vs-reality issue at `baseline/B1a` HEAD that PR-2 cannot resolve
(the gap precedes PR-2 by multiple PRs); flagged for orchestrator review
along with the spec text. PR-2 is dormant structural fix; per "B1a Dormant
Path Limitation" paragraph (spec L137) PR-2 explicitly **cannot** exercise
the runtime path that would manifest race fixes.

## Step 3 — Config C (`make shud_omp` + `OMP_NUM_THREADS=4` + `OMP_PROC_BIND=close OMP_PLACES=cores`, 2 runs)

4-case `rivqdown.dat` SHA256 run1 vs run2 (repeat-determinism, NOT vs B0):

```
keliya              REPEAT-OK  r1=b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a
xinanjiang_upstream REPEAT-OK  r1=90eeb9c63c07e8db3a051482f51cdc274f6e469326b19575efed54181258bf45
qinyijiang          REPEAT-OK  r1=0f8c3fecfe1e618f24ee5411ff9c7ea80381e953e6aafac6a1a96fe666018ced
qhh                 REPEAT-OK  r1=38ef0414d9ffa9310828183ce769db97c8c6c1e8714d71330bae78c0a2c0641c
```

**Config C verdict: PASS (4/4 byte-equal across 2 runs)** — validates
OpenMP NVector backend + `rhs_core()` serial path multi-threaded
determinism. Note the spec L148 explicit limitation: this validates
NVector backend repeat-determinism, **NOT** `f_applyDY_omp` race fix
(the dormant TU is filter-out excluded; no runtime exercise possible
in B1a scope). PR-2 race fix is a structural alignment for P1+
re-activation, not validatable here.

Curiously, the Config C 4-thread hashes byte-equal the Config B
single-thread hashes — implies OpenMP NVector backend is fully
deterministic across thread counts at this scale (1 ↔ 4), with the
gap-from-B0 invariant being purely backend-vs-Serial-NVector.

## Step 4 — `make shud SHUD_LEGACY_OMP_RHS=1` compile sanity

Build: `make clean && make shud SHUD_LEGACY_OMP_RHS=1`  -> PASS (only
pre-existing sprintf deprecation warnings; no `MD_f_omp.cpp` warnings).

`nm` _omp symbols:

```
$ nm SHUD/shud | grep -E '_(update|loop|applyDY)_omp'
000000010000d6dc T __ZN10Model_Data10f_loop_ompEPdS0_d
000000010000d880 T __ZN10Model_Data12f_update_ompEPdS0_d
000000010000d260 T __ZN10Model_Data13f_applyDY_ompEPdd
```

3 _omp symbols present in LEGACY=1 build (per spec S2.9 L150 + design.md
PR-2 Invariant Matrix "LEGACY_RHS=1 build sanity" row). PR-8 capstone
will delete them; until then they must remain linkable. PASS.

`SHUD/src/ModelData/Model_Data.hpp:261-263` 3 `_omp` function declarations
retained (verified via `sed -n '261,263p'`):

```
    void f_loop_omp(double * Y, double * DY, double t);
    void f_applyDY_omp(double * DY, double t);
    void f_update_omp(double * Y, double * DY, double t);
```

## Step 6 — Snapshot bitwise (12 files, PR-1 PATH A inheritance)

Build: `make clean && make shud SHUD_DUMP_RHS=1`.
Site: `f_update` (no suffix).
t-values dumped: `START_min + {1440, 43200, 129600}` (abs-min for 1d/30d/90d)
per case, set via `SHUD_DUMP_T_VALUES`. Filenames `snapshot_t<abs_min>.bin`;
comparator uses array-data-only (header `t_value` field ignored), so direct
compare against `benchmarks/<case>/B0_output/snapshot_t<rel_sec>.bin`
(rel_sec = 86400 / 2592000 / 7776000) is valid.

`SHUD_DUMP_T_TOL` = 720 (= 0.5 day in mins) was required for all 4 cases
to capture late-window dumps; lower tolerances (0.5 / 60) missed
+30d/+90d targets due to CVODE adaptive timestep landing slightly off the
exact target. Same numeric output, just larger tolerance window for the
internal target-match.

```
keliya              new=17357760  vs  golden=86400     PASS
keliya              new=17399520  vs  golden=2592000   PASS
keliya              new=17485920  vs  golden=7776000   PASS
xinanjiang_upstream new=1440      vs  golden=86400     PASS
xinanjiang_upstream new=43200     vs  golden=2592000   PASS
xinanjiang_upstream new=129600    vs  golden=7776000   PASS
qinyijiang          new=528480    vs  golden=86400     PASS
qinyijiang          new=570240    vs  golden=2592000   PASS
qinyijiang          new=656640    vs  golden=7776000   PASS
qhh                 new=12098880  vs  golden=86400     PASS
qhh                 new=12140640  vs  golden=2592000   PASS
qhh                 new=12227040  vs  golden=7776000   PASS

PASS = 12 / 12
FAIL = 0
```

Path determination: **inherit PR-1 PATH A** — 12/12 snapshots PASS + 4-case
+ qhh rivqdown PASS in Config A. PR-2 inherits without re-regen. Golden
remains compatible with PR-2 default build.

## heihe / heihe_x4 (server-only)

Out of scope for Mac-local PR-2 implementer; orchestrator will run the
server-side validation (`heihe` / `heihe_x4`) on
`Linux 210.77.77.22:32099` endpoint per `CLAUDE.md` cross-platform
endpoint split.

## SHUD submodule protocol

- Inner branch: `openmp-baseline` (NOT `master`).
- `.gitmodules` URL: unchanged vs `baseline/B1a`.
- SHUD commit: `75e32e0` -> `<new SHA after push>` on `openmp-baseline`.
- SHUD push to upstream `openmp-baseline`: pending (commit-then-push step).

## Verification commands (reproducible from `/Users/danker/Desktop/Hydro-SHUD/openMP`)

```
# 0. SHUD branch
(cd SHUD && git branch --show-current)              # -> openmp-baseline

# 1. Grep gates (3 PR-2 + 4 S1)
awk '/^void Model_Data::f_update_omp\(/,/^}$/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '\(Y\[i(SF|US|RIV)\] >= 0\.\) \?'   # -> 0
awk '/^void Model_Data::f_update_omp\(/,/^}$/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '^[[:space:]]*uY(sf|us|riv)\[i\][[:space:]]*=[[:space:]]*Y\[i(SF|US|RIV)\];'  # -> 3
awk '/^void Model_Data::f_applyDY_omp\(/,/#pragma omp parallel/' SHUD/src/ModelData/MD_f_omp.cpp \
    | grep -cE '^[[:space:]]*(double|int)[[:space:]]+(area|isf|ius|igw)\b'                   # -> 0

# 2. Config A
(cd SHUD && make clean && make shud)
for c in keliya xinanjiang_upstream qinyijiang qhh; do
  proj=$c
  [ "$c" = "xinanjiang_upstream" ] && proj=xinanjiang
  [ "$c" = "qinyijiang" ]         && proj=nanlin
  rm -rf SHUD/Basins/$c/output/${proj}.out
  (cd SHUD/Basins/$c && ../../shud $proj)
  shasum -a 256 SHUD/Basins/$c/output/$proj.out/$proj.rivqdown.dat \
                benchmarks/$c/B0_output/$proj.rivqdown.dat
done

# 3. Config B
(cd SHUD && make clean && make shud_omp)
# Same loop, prepend `OMP_NUM_THREADS=1` to invocation; expect Config B FAIL
# (pre-existing baseline gap — see Step 2 above).

# 4. Config C (2 runs, repeat-determinism)
# Same loop, prepend
# `OMP_NUM_THREADS=4 OMP_PROC_BIND=close OMP_PLACES=cores` to invocation,
# run twice, sha256 run1 vs run2.

# 5. LEGACY_RHS=1 sanity
(cd SHUD && make clean && make shud SHUD_LEGACY_OMP_RHS=1)
nm SHUD/shud | grep -E '_(update|loop|applyDY)_omp'   # -> 3 symbols

# 6. Snapshot
(cd SHUD && make clean && make shud SHUD_DUMP_RHS=1)
# Per-case loop with SHUD_DUMP_T_VALUES set to START_min + {1440,43200,129600},
# SHUD_DUMP_T_TOL=720; compare via tools/compare_snapshot/compare_snapshot.

# 7. .gitmodules unchanged
git diff baseline/B1a -- .gitmodules                       # -> empty
```
