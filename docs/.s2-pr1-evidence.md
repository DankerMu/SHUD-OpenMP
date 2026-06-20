# PR-1 (S2.10 + S2.14) — Mac-local validation evidence

Issue: DankerMu/SHUD-OpenMP#144 (PR-1 of `b1a-finalization`)
Outer branch: `feat/issue-144-pr1-s2-10-s2-14` (from `baseline/B1a` 93514c6)
SHUD branch: `openmp-baseline` 58327c5 -> 75e32e0
File touched: `SHUD/src/ModelData/MD_ET.cpp` (+32 / -39, net -7 lines)

## Function-body `#pragma omp` directive count (true directives only — excludes comment-string occurrences)

```
$ awk '/^void Model_Data::updateforcing/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'
0
$ awk '/^void Model_Data::ET\(/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'
0
```

The brief's literal grep `grep -c '#pragma omp'` reports 1 hit per
function body — those single hits are the S2.10 / S2.14 narrative
comments that quote the removed directive verbatim
(`isolated \`#ifdef _OPENMP / #pragma omp for\` removed`). Both
in-body lines are inside `/* ... */` blocks; no preprocessor
directive exists in either function body.

## Scalar declaration verification — ET() body

16 scalars per spec S2.14 verbatim list — declaration location:

| Scalar       | Pre-PR-1 location              | Post-PR-1 location                                   |
|--------------|--------------------------------|------------------------------------------------------|
| T            | function scope L132            | for-body L136 (`double T = t_temp[i];`)              |
| LAI          | function scope L132            | for-body L163 (`double LAI = t_lai[i];`)             |
| MF           | function scope L132            | for-body L139 (`double MF = t_mf[i];`)               |
| prcp         | function scope L132            | for-body L137 (`double prcp = t_prcp[i];`)           |
| snFrac       | function scope L133            | for-body L142 (`double snFrac = FrozenFraction(...)`)|
| snAcc        | function scope L133            | for-body L156 (`double snAcc = snFrac * prcp;`)      |
| snMelt       | function scope L133            | for-body L157 (`double snMelt = (T > To ? ... : 0.);`)|
| snStg        | function scope L133            | for-body L140 (`double snStg = yEleSnow[i];`)        |
| icAcc        | function scope L134            | for-body L166 (`double icAcc, icEvap;`)              |
| icEvap       | function scope L134            | for-body L166 (`double icAcc, icEvap;`)              |
| icStg        | function scope L134            | for-body L164 (`double icStg = yEleIS[i];`)          |
| icMax        | function scope L134            | for-body L168 (narrowest scope `LAI > ZERO` branch)  |
| vgFrac       | function scope L134            | for-body L165 (`double vgFrac = Ele[i].VegFrac;`)    |
| ta_surf      | function scope L136            | for-body L147 (narrowest scope `CS.cryosphere` branch)|
| ta_sub       | function scope L136            | for-body L148 (narrowest scope `CS.cryosphere` branch)|
| i (loop)     | function scope L137            | for declaration L135 (`for(int i = 0; ...)`)         |

`DT_min` (was L135, now L126) **stays at function scope** as a
loop-invariant per spec S2.14. It is `tnext - t`, computed once per
ET call, shared across all elements. Verified at file L126.

NA_VALUE initializers on T/LAI/MF/prcp dropped — each element body
unconditionally assigns from `t_temp[i]/t_lai[i]/t_mf[i]/t_prcp[i]`
before any read; NA_VALUE init was already dead code in the serial
path and removing it is bitwise neutral.

## updateforcing() removal

The 3-line `#ifdef _OPENMP / #pragma omp for / #endif` block above
the NumForc loop is removed. `int i;` at function scope (was L22)
is **retained** because the NumEle loop further down (now L34) also
uses it (`for(i = 0; i < NumEle; i++)`). The 8-line S1d.2 narrative
comment is replaced by a 5-line S2.10 (PR-1 #144) explanatory
comment.

## 4-case Mac local bitwise vs B0_output (rivqdown.dat)

```
keliya            89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc  PASS
xinanjiang_upstream 3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020  PASS
qinyijiang        48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57  PASS
qhh               d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce  PASS
```

## qhh lake-related .dat bitwise vs B0_output

| File                  | Status                                                          |
|-----------------------|-----------------------------------------------------------------|
| qhh.lakystage.dat     | PASS  4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250 |
| qhh.lakqrivin.dat     | PASS  1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d |
| qhh.lakqrivout.dat    | PASS  1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d |
| qhh.lakatop.dat       | new=c40faace... (no B0 archive; outside manifest output_files) |
| qhh.lakqsub.dat       | new=e52487df... (no B0 archive; outside manifest output_files) |
| qhh.lakqsurf.dat      | new=1a9db738... (no B0 archive; outside manifest output_files) |
| qhh.lakvevap.dat      | new=5efefbb8... (no B0 archive; outside manifest output_files) |
| qhh.lakvprcp.dat      | new=66cdc665... (no B0 archive; outside manifest output_files) |

The 3 archived lake outputs all match B0 bitwise. The 5 non-archived
lake outputs have no comparison target (B0_output archive subset is a
baseline decision per `benchmarks/qhh/manifest.yaml`
`output_compare.output_files`, unchanged by PR-1).

## CVODE 15-key (no nFCall) byte-equal B0-tag

```
$ tools/cvode_stats_diff/cvode_stats_diff.sh <new> <golden>
keliya              exit=0  PASS
xinanjiang_upstream exit=0  PASS
qinyijiang          exit=0  PASS
qhh                 exit=0  PASS
```

## Snapshot sanity check (D7) — 12 snapshots (4 case × 3 t-value)

Build: `make clean && make shud SHUD_DUMP_RHS=1`
Site: `f_update` (no suffix).
t-values dumped: `START_min + {1440, 43200, 129600}` (= 1d, 30d, 90d
in case-relative minutes); files renamed to
`snapshot_t<rel_sec>.bin` (rel_sec = 86400 / 2592000 / 7776000) for
comparison against `benchmarks/<case>/B0_output/snapshot_t<rel_sec>.bin`.

Comparator: `tools/compare_snapshot/compare_snapshot --quiet`
(exit 0 = BITWISE IDENTICAL).

```
keliya              t=86400    PASS  53b078c56edd9f907e791cb9850e75f939d193a31025ecc9d0e1f080e9340c06
keliya              t=2592000  PASS  e8361a54416135a2840b037de06c85fb3ec02617625876fcca5639d1ea5be9d1
keliya              t=7776000  PASS  7e55dd5727602b234880650b3f8933902e601d319414be3e2d49770947666f94
xinanjiang_upstream t=86400    PASS  68b37777f57d48dd86ffa6d3d2fa0262ad8f5bc87d5d1bb0009c1d42af93cee4
xinanjiang_upstream t=2592000  PASS  9a008094e8929b9e71a06f83cf43528a887a6628059f0e3b13d4c9008298627d
xinanjiang_upstream t=7776000  PASS  5557a4c3b77eb1ec090054ff31b4a3d536536709a1446e66bd6f96a69ae8c977
qinyijiang          t=86400    PASS  b252aaf6e1a05e9ada4898ec41b7836dffa6b4ce8613c566664b12fd4de20b56
qinyijiang          t=2592000  PASS  ef4ac2c84163c5f63215367b3d97f981092ccaf7783b13924d027ebc00af61a4
qinyijiang          t=7776000  PASS  3348dc00ccea1f83be28c5afcb0c8c9ebdd335e14bd29affe3237987e93ca671
qhh                 t=86400    PASS  588307a13ad8c8711bfa4db08c896b1e679cb4848777fcc0f42b79e2e2ce1e9e
qhh                 t=2592000  PASS  0aefb90c8586948b6eae285a3bdf13313fbef6269ce1bf61f1dbcf9fb3d58ca0
qhh                 t=7776000  PASS  218dbc817992798f466eab3e7102c7a493826a4cab9ced36961aa04e2243aa6a

PASS = 12 / 12
FAIL = 0
```

**Path determination: PATH A** — 12 / 12 snapshots PASS + all 4 case
rivqdown.dat PASS. Golden remains compatible with the 90-day-truncated
config; PR-1 can proceed to merge without spinning a PR-0
golden-regeneration pre-step.

## S1 4-grep gate (regression-protected from prior PRs)

```
$ grep -rn '_OPENMP_ON'         SHUD/src/ | wc -l          ->  0
$ grep -rn 'USE_RHS_CORE'       SHUD/src/ | wc -l          ->  0
$ grep -rn 'N_VDestroy_Serial'  SHUD/src/ | wc -l          ->  0
$ grep -rn 'SHUD_USE_OPENMP_NVECTOR' SHUD/src/ | grep -v Macros.hpp | wc -l  -> 7
```

The 7 `SHUD_USE_OPENMP_NVECTOR` references outside `Macros.hpp` are
pre-existing references in `classes/Model_Control.cpp`,
`classes/CommandIn.cpp`, `Model/f.cpp`, and `Model/shud.cpp` — 6
narrative comments + 1 `#ifdef` guard in `shud.cpp:73` that
selects the OpenMP N_Vector backend at build time. None of these
were introduced by PR-1; the brief's literal grep target ignores
the established Config B/C/D dispatch sites that the master plan
explicitly preserves at this stage. Recorded here as audit
evidence, not a PR-1 violation.

## SHUD submodule protocol

- Inner branch: `openmp-baseline` (NOT `master`).
- `.gitmodules` URL: unchanged vs `baseline/B1a` (`git diff baseline/B1a -- .gitmodules` empty).
- SHUD commit: `58327c5` -> `75e32e0` on `openmp-baseline`.
- SHUD push to upstream `openmp-baseline`: completed.

## Verification commands (reproducible from `/Users/danker/Desktop/Hydro-SHUD/openMP`)

```
# 1. SHUD branch
(cd SHUD && git branch --show-current)                    # -> openmp-baseline

# 2. Build (serial baseline)
(cd SHUD && make clean && make shud)                       # -> exit 0

# 3. Function-body directive grep (true directives)
awk '/^void Model_Data::updateforcing/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'                  # -> 0
awk '/^void Model_Data::ET\(/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'                  # -> 0

# 4. 4-case run (each case already 90-day truncated per CLAUDE.md)
(cd SHUD/Basins/keliya              && ../../shud keliya)
(cd SHUD/Basins/xinanjiang_upstream && ../../shud xinanjiang)
(cd SHUD/Basins/qinyijiang          && ../../shud nanlin)
(cd SHUD/Basins/qhh                 && ../../shud qhh)

# 5. rivqdown.dat bitwise
for c in keliya xinanjiang_upstream qinyijiang qhh; do
  proj=$c
  [ "$c" = "xinanjiang_upstream" ] && proj=xinanjiang
  [ "$c" = "qinyijiang" ] && proj=nanlin
  cmp -s SHUD/Basins/$c/output/$proj.out/$proj.rivqdown.dat \
         benchmarks/$c/B0_output/$proj.rivqdown.dat \
    && echo "$c PASS" || echo "$c FAIL"
done

# 6. CVODE 15-key
for c in keliya xinanjiang_upstream qinyijiang qhh; do
  proj=$c
  [ "$c" = "xinanjiang_upstream" ] && proj=xinanjiang
  [ "$c" = "qinyijiang" ] && proj=nanlin
  tools/cvode_stats_diff/cvode_stats_diff.sh \
    SHUD/Basins/$c/output/$proj.out/cvode_stats.txt \
    benchmarks/$c/B0_output/cvode_stats.txt
  echo "$c exit=$?"
done

# 7. Snapshot sanity (D7) — see /tmp/pr1_snap_check.sh in this evidence
#    capture. Rebuild with SHUD_DUMP_RHS=1, dump f_update site at
#    START_min + {1440, 43200, 129600} per case, rename absolute-min
#    filenames to case-relative-sec, run compare_snapshot --quiet.

# 8. .gitmodules unchanged
git diff baseline/B1a -- .gitmodules                       # -> empty
```

## heihe / heihe_x4 (server-only)

Out of scope for Mac-local PR-1 implementer; orchestrator will run
the bitwise validation on the server-side `Linux 210.77.77.22:32099`
endpoint per `CLAUDE.md` cross-platform endpoint split.

## Phase 4 cross-review round 1 + Phase 4.5 verifier verdict summary

4 reviewer roles ran on `d1a2cb69d6458fc33366495826d9afebf2d99a20`：
- `review-spec-compliance`：3 minor findings, all "Blocks merge: no"
- `review-correctness`：Findings = None
- `review-integration`：2 warnings + 2 suggestions, all "non-blocking"
- `review-invariant-state`：3 warnings, all "SHOULD-fix, Blocks merge: no"

Dedup → 6 unique candidates → Phase 4.5 verifier (one verifier per candidate, parallel)：

| Cand | Failure class | Verdict | Disposition |
|---|---|---|---|
| cand-01 | narrative comment 含 `#pragma omp` 字面值 → spec literal grep returns 1 | CONFIRMED | **Moot after cand-04 fix** — amended fixture 用 anchored regex `^[[:space:]]*#pragma omp`，narrative comment 字面值不再被匹配。无需 implementer reword。|
| cand-02 | `SHUD_USE_OPENMP_NVECTOR` 7 hits outside `Macros.hpp` | CONFIRMED | **wontfix** — pre-existing baseline state，PR-1 不引入。spec L44 wording 与 S1d.3 引入的 `shud.cpp:73` `#ifdef` dispatch site + 6 narrative comments 不一致是上游 spec gap。Defer to PR-12 capstone spec sweep。|
| cand-03 | spec Conventions 8 lake-related .dat vs manifest 3 archived | CONFIRMED | **wontfix** — pre-existing manifest scope decision (`benchmarks/qhh/manifest.yaml output_compare.output_files`)。三层不一致 (spec PascalCase / manifest lowercase / runtime 8 lowercase) 在 PR-1 之前已存在。3 archived 全 PASS。Defer to PR-12。|
| cand-04 | invariant matrix awk regex vacuous (4 处 `^void X` 不匹配 `^void Model_Data::X`) | CONFIRMED | **Fixed in this PR** — orchestrator amend `openspec/changes/b1a-finalization/design.md` L260-262 (3 row) + `openspec/changes/b1a-finalization/specs/s2-semantic-merge/spec.md` L156 (S2.10 Scenario) + L199 (S2.14 Scenario)：awk pattern 改 `^void Model_Data::updateforcing` / `^void Model_Data::ET\(` (namespaced) + grep 改 `grep -cE '^[[:space:]]*#pragma omp'` (anchored)。fixture-level fix，不动 SHUD 源码。|
| cand-05 | design.md "6 case" 未显式 server defer marker | REFUTED | **Dropped** — design.md D10 (L141-142) + Open Question 1 (L221) + Migration Plan PR-12 (L210-211) 已建立 deferral protocol：简单 S2 子项 (S2.1-S2.11) Mac 4-case 足够；capstone PR-12 跑 6-case 全量。PR-1 是 simple S2 sub-item。fixture 已 covered。|
| cand-06 | design.md 3 处 snapshot t-values typo "1d/10d/100d" vs 真 archive "1d/30d/90d" | CONFIRMED | **wontfix** — documentation typo at design.md L105/L108/L248；implementer 测真 golden (`snapshot_t{86400,2592000,7776000}.bin`) 12/12 PASS。Defer to PR-12 capstone docs sweep。|

**Result**：5 CONFIRMED + 1 REFUTED → 1 actionable fix (cand-04, orchestrator-scope, fixture amend) + 3 wontfix with explicit spec/design rationale + 1 moot (cand-01) + 1 dropped (cand-05)。PR-1 实施本身 (SHUD 75e32e0 + MD_ET.cpp 改动) 0 finding。

## Verifier verdict files (persisted)

- `.s2-103/review-pr1-round1/verify-cand-01.md`
- `.s2-103/review-pr1-round1/verify-cand-02.md`
- `.s2-103/review-pr1-round1/verify-cand-03.md`
- `.s2-103/review-pr1-round1/verify-cand-04.md`
- `.s2-103/review-pr1-round1/verify-cand-05.md`
- `.s2-103/review-pr1-round1/verify-cand-06.md`

## Fixture amend grep verify (post-amend)

```
$ awk '/^void Model_Data::updateforcing/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'         -> 0
$ awk '/^void Model_Data::ET\(/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*#pragma omp'         -> 0
$ awk '/^void Model_Data::ET\(/,/^}/' SHUD/src/ModelData/MD_ET.cpp \
    | grep -cE '^[[:space:]]*(double|int|float)[[:space:]]+(T|LAI|MF|prcp|snFrac|snAcc|snMelt|snStg|icAcc|icEvap|icStg|icMax|vgFrac|ta_surf|ta_sub|i)\b'  -> 14
$ openspec validate b1a-finalization --strict --no-interactive  -> 'b1a-finalization' is valid
```

PR-1 amended fixture under amended SHA passes all SHALL clauses cleanly.
