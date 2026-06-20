# PR-5 (S2.3) — Record-only PR evidence

Issue: DankerMu/SHUD-OpenMP#148 (PR-5 of `b1a-finalization`)
Outer branch: `feat/issue-148-pr5-etflux-record` (from `baseline/B1a` `4da488c`)
SHUD branch: `openmp-baseline` HEAD unchanged at `8fa6b29` (no PR-5 SHUD commit)
File touched: **NONE in SHUD/** + `docs/.s2-pr5-evidence.md` (this file, NEW)

## TL;DR

PR-5 is **record-only** (cumulative pattern from PR-3 + PR-4). S2.3 ET flux 对非 lake element 调用 `f_etFlux()` 已经在 `rhs_core()` 内（S1b `rhs_flux()` PURE CARRY-OVER `f_loop()`）。PR-5 makes 0 SHUD source change; only commits this evidence record.

## S2.3 Requirement & verification

**Spec** (`openspec/changes/b1a-finalization/specs/s2-semantic-merge/spec.md`, S2.3): 工程 SHALL 在 `rhs_core()` 中显式对非 lake element 调用 `f_etFlux()`（`SHUD/src/ModelData/MD_ET.cpp:167-228`）；lake element 上 `f_etFlux()` SHALL 不被调用。

**Reality** (`MD_rhs_core.cpp` `rhs_flux()` body, within first NumEle loop):

```cpp
for (i = 0; i < NumEle; i++) {
    if(lakeon && Ele[i].iLake > 0){
        /* Lake elements */
        Ele[i].updateLakeElement();
        fun_Ele_lakeVertical(i, t);
        qLakeEvap[Ele[i].iLake - 1] += qEleEvapo[i] / lake[Ele[i].iLake - 1].NumEleLake;
        qLakePrcp[Ele[i].iLake - 1] += qElePrep[i] / lake[Ele[i].iLake - 1].NumEleLake;
    }else{
        f_etFlux(i, t);
        /*DO INFILTRATION FRIST, then do LATERAL FLOW.*/
        ...
    }
}
```

PURE CARRY-OVER source: `SHUD/src/ModelData/MD_f.cpp` `f_loop()` body L14-25 — byte-for-byte identical.

**Guard form note**: spec wording suggests `if (Ele[i].iLake > 0) continue;` skip, but actual code uses mutually-exclusive `if (lakeon && Ele[i].iLake > 0) { ... lake branch ... } else { f_etFlux(...); ... }`. Both forms semantically equivalent (non-lake-only call). No spec amend needed — equivalent control-flow OK.

## Grep gates (3/3 PASS)

```bash
# Gate 1: f_etFlux call exists in rhs_flux body
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'f_etFlux\(i, t\)'
# Expected: 1 ; Actual: 1 ✓

# Gate 2: call is in else-branch (preceded within 2 lines by `}else{`)
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -B 2 'f_etFlux' | grep -cE '\}else\{'
# Expected: 1 ; Actual: 1 ✓

# Gate 3: the `if(lakeon && Ele[i].iLake > 0)` guard sits 7 lines above `f_etFlux` (lake branch body = 5 lines + `}else{`)
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -B 7 'f_etFlux' | grep -cE 'if\(lakeon && Ele\[i\]\.iLake > 0\)'
# Expected: 1 ; Actual: 1 ✓
# This confirms: f_etFlux IS in the else-branch of the lake-guard if-else block (mutually-exclusive with lake-branch)
```

## Cross-validation: rhs_flux full body diff vs f_loop

Already executed in PR-4 evidence: `diff <(awk '...rhs_flux...' MD_rhs_core.cpp) <(awk '...f_loop...' MD_f.cpp)` = 1 line diff (function name only). Therefore f_etFlux call structure inherited byte-for-byte from S1b PURE CARRY-OVER source.

## Verified at PR-5 HEAD (= baseline/B1a HEAD `4da488c` + this evidence file)

- SHUD source diff vs `baseline/B1a 4da488c`: **0 lines**
- 4-case Mac local + qhh `rivqdown.dat` bitwise vs B0: **PASS** (inherits PR-4 merged state)
- CVODE 15-key (4-case): byte-equal B0-tag archive (inherits PR-4)
- LEGACY_RHS=1 build sanity: PASS (no SHUD source change)
- 12 snapshot bitwise: PASS (PR-1 PATH A inherits)
- S1 4-grep gate: 0/0/0/7 (unchanged from PR-1+PR-2+PR-3+PR-4 baseline)

## SHUD submodule discipline

- SHUD HEAD: `8fa6b29` (unchanged from PR-2)
- `.gitmodules` URL: `SHUD-System/SHUD.git` (unchanged)
- No SHUD branch push (record-only)

## Files in this PR

- `docs/.s2-pr5-evidence.md` (NEW, this file)

## Cumulative pattern continuation

PR-3 + PR-4 + PR-5 collectively confirm: **S2.1, S2.2, S2.3, S2.5, S2.7, S2.11** are all auto-satisfied by S1a/S1b/S1c PURE CARRY-OVER. PR-6 (S2.4 river DY) preview also confirmed record-only:

```bash
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'DY\[iRIV\][[:space:]]*=[[:space:]]*\(- QrivUp'
# Expected: 1 ; (already verified in PR-4 evidence end)
```

So PR-6 will be record-only too. Remaining S2 active code change: PR-7a/7b (record-only diff report) + PR-8 capstone (delete `MD_f_omp.cpp` + retire macro).

## Active code change remaining in S2 epic (after PR-5)

- PR-6 #149 [S2.4] — record-only (predicted)
- PR-7a #150 [S2.12/13/15/16] — record-only (4 items)
- PR-7b #151 [S2.17] — assert + DEBUG (only real code change: macro guard + assert)
- PR-8 #152 [S2-capstone] — delete `MD_f_omp.cpp` + retire `SHUD_LEGACY_OMP_RHS` (capstone semantic)
- PR-9 #153 [S3a + S3b + S2.8 D14] — dead-code + shared-write split (large semantic)

## Next

- Merge PR-5 → close #148
- Append `docs/review-loop-log.jsonl`
- PR-6 #149 record-only (predicted)
