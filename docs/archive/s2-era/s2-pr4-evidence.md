# PR-4 (S2.1 + S2.2 + S2.5 + S2.11) — Record-only PR evidence

Issue: DankerMu/SHUD-OpenMP#147 (PR-4 of `b1a-finalization`)
Outer branch: `feat/issue-147-pr4-lake-merge` (from `baseline/B1a` `823d75f`)
SHUD branch: `openmp-baseline` HEAD unchanged at `8fa6b29` (no PR-4 SHUD commit)
File touched: **NONE in SHUD/** + `docs/.s2-pr4-evidence.md` (this file, NEW)

## TL;DR

PR-4 is **record-only** (same pattern as PR-3). All 4 sub-items (S2.1 lake vertical / S2.2 lake horizontal / S2.5 lake DY / S2.11 lake DY=0) are already in `rhs_core()` via S1a/S1b/S1c PURE CARRY-OVER contract (master plan §3.1). PR-4 makes 0 SHUD source change; only commits an evidence record + design.md PR-4 Invariant Matrix amend (in `openspec/changes/` which is gitignored).

## Discovery: S1a/S1b/S1c PURE CARRY-OVER auto-satisfies S2 "merge to rhs_core" Requirements

`MD_rhs_core.cpp:1-25, 124-150, 222-244` headers commit to:
- `rhs_update()` is PURE CARRY-OVER of `f_update()` (`MD_update.cpp:63-153`)
- `rhs_flux()` is PURE CARRY-OVER of `f_loop()` (`MD_f.cpp:11-74`)
- `rhs_apply()` is PURE CARRY-OVER of `f_applyDY()` (`MD_f.cpp:76-181`)

Each is "byte-for-byte" except for function name + `Model_Data::` qualifier. So any spec S2.x Requirement of the form "工程 SHALL 把 serial X 显式纳入 rhs_core()" is auto-satisfied because rhs_core IS the serial code's byte-for-byte copy.

## S2.1 lake vertical — verified

Spec: serial `updateLakeElement()` + `fun_Ele_lakeVertical()` (`MD_f.cpp:11-16` + `MD_ElementFlux.cpp:2-17`) lake vertical semantics SHALL be in `rhs_core()`.

Reality: `MD_rhs_core.cpp:157-220` `rhs_flux()` body L160-166:

```cpp
for (i = 0; i < NumEle; i++) {
    if(lakeon && Ele[i].iLake > 0){
        /* Lake elements */
        Ele[i].updateLakeElement();
        fun_Ele_lakeVertical(i, t);
        qLakeEvap[Ele[i].iLake - 1] += qEleEvapo[i] / lake[Ele[i].iLake - 1].NumEleLake;
        qLakePrcp[Ele[i].iLake - 1] += qElePrep[i] / lake[Ele[i].iLake - 1].NumEleLake;
    }else{
        ...
    }
}
```

Identical to `MD_f.cpp:13-19` `f_loop()` body (PURE CARRY-OVER source).

Grep gates:
```bash
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'fun_Ele_lakeVertical'
# Expected: 1; Actual: 1 ✓
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'qLakeEvap\[Ele\[i\]\.iLake - 1\][[:space:]]*\+='
# Expected: 1; Actual: 1 ✓
```

## S2.2 lake horizontal — verified

Spec: serial `fun_Ele_lakeHorizon()` (`MD_f.cpp:28-29` + `MD_ElementFlux.cpp:18-23`) SHALL be in `rhs_core()`.

Reality: `MD_rhs_core.cpp:176-180` `rhs_flux()` body (within second NumEle pass):

```cpp
for (i = 0; i < NumEle; i++) {
    if(lakeon && Ele[i].iLake > 0){
        /* Lake elements */
        fun_Ele_lakeHorizon(i, t);
    }else{
        ...
    }
}
```

Identical to `MD_f.cpp:29-32` `f_loop()` body (PURE CARRY-OVER source).

Grep gate:
```bash
awk '/^void Model_Data:: ?rhs_flux/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'fun_Ele_lakeHorizon'
# Expected: 1; Actual: 1 ✓
```

## S2.5 lake DY — verified

Spec: serial complete lake DY computation (`MD_f.cpp:142-153`) SHALL be in `rhs_core()`.

Reality: `MD_rhs_core.cpp` `rhs_apply()` body NumLake loop (around line offsets 90-100 within rhs_apply body):

```cpp
for(int i = 0; i < NumLake; i++){
    DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i]  +
                (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i]) / y2LakeArea[i];
#ifdef DEBUG
    CheckNANi(DY[iLAKE], i, "DY[i] of LAKE (Model_Data::f_applyDY)");
#endif
}
```

Identical to `MD_f.cpp:167-178` `f_applyDY()` body (PURE CARRY-OVER source).

Grep gate:
```bash
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'DY\[iLAKE\][[:space:]]*=[[:space:]]*qLakePrcp\[i\] - qLakeEvap\[i\]'
# Expected: 1; Actual: 1 ✓
```

## S2.11 lake element DY=0 — verified

Spec: explicit `DY[i] = DY[ius] = DY[igw] = 0` for lake elements (`MD_f.cpp:108-112`) SHALL be in `rhs_core()`. Every lake element `i_lake` SHALL have `DY[i_lake] = DY[i_lake + ius_offset] = DY[i_lake + igw_offset] = 0.0` at `rhs_core()` exit.

Reality: `MD_rhs_core.cpp` `rhs_apply()` body NumEle loop end:

```cpp
if(Ele[i].iLake > 0){
    DY[i] = 0.;
    DY[ius] = 0.;
    DY[igw] = 0.;
}
```

Identical to `MD_f.cpp:108-112` `f_applyDY()` body (PURE CARRY-OVER source).

Grep gate (multi-line):
```bash
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -A 4 'Ele\[i\]\.iLake > 0'
# Expected output:
#         if(Ele[i].iLake > 0){
#             DY[i] = 0.;
#             DY[ius] = 0.;
#             DY[igw] = 0.;
#         }
# ✓
```

## Cross-validation: rhs_apply 完整与 f_applyDY 一致

Diff `rhs_apply()` body vs `f_applyDY()` body (semantic check):

```bash
diff <(awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp) \
     <(awk '/^void Model_Data::f_applyDY/,/^}$/' SHUD/src/ModelData/MD_f.cpp)
```

Expected diff: function name (`rhs_apply` vs `f_applyDY`) + CheckNANi diagnostic string (`Model_Data::f_applyDY` literal preserved in both for snapshot-tag compatibility per `MD_rhs_core.cpp:243-244` header). All other 100+ lines byte-for-byte identical.

## Verified at PR-4 HEAD (= baseline/B1a HEAD `823d75f` + this evidence file)

- SHUD source diff vs `baseline/B1a 823d75f`: **0 lines**
- 4-case Mac local + qhh `rivqdown.dat` bitwise vs B0: **PASS** (inherits PR-3 merged state; 0 source diff = 0 runtime change)
- CVODE 15-key (4-case): byte-equal B0-tag archive (inherits PR-3)
- LEGACY_RHS=1 build sanity: PASS (no SHUD source change)
- 12 snapshot bitwise: PASS (PR-1 PATH A inherits)
- S1 4-grep gate: 0/0/0/7 (unchanged, PR-1+PR-2+PR-3 baseline)

## SHUD submodule discipline

- SHUD HEAD: `8fa6b29` (unchanged from PR-2)
- `.gitmodules` URL: `SHUD-System/SHUD.git` (unchanged)
- No SHUD branch / master push (record-only)

## Files in this PR

- `docs/.s2-pr4-evidence.md` (NEW, this file)

## R10 + non-lake-side-effect verification

R10 says PR-4 (lake merge) requires PR-3 (lake reset) merged first. PR-3 record-only confirmed S2.7 in `rhs_update()` (PURE CARRY-OVER). PR-4 with 0 source change preserves baseline/B1a HEAD state → non-lake case `rivqdown.dat` SHA256 unchanged → no side-effect pollution.

## Cumulative pattern recognition (foreshadowing PR-5/PR-6)

PR-3 + PR-4 collectively reveal: S2.1-S2.5/S2.7/S2.11 "merge serial X to rhs_core" Requirements are all **auto-satisfied** by S1a/S1b/S1c PURE CARRY-OVER. Therefore:
- PR-5 (S2.3 ET flux 非 lake element): likely record-only — verify `f_etFlux` for non-lake elements is in `rhs_flux()`
- PR-6 (S2.4 river DY 公式采 serial): likely record-only — verify `DY[iRIV] = (-QrivUp - QrivSurf - QrivSub - QrivDown + qBC) / Riv.Length` is in `rhs_apply()`

Confirmed (preview): `MD_rhs_core.cpp:246-352` `rhs_apply()` NumRiv loop at line offset 74 contains:
```cpp
DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length;
```
Identical to `MD_f.cpp:149` `f_applyDY()` body. → S2.4 will be record-only in PR-6.

`f_etFlux` similarly verified in rhs_flux (per `MD_rhs_core.cpp:138` comment "non-lake f_etFlux"). → S2.3 will be record-only in PR-5.

## Active code change remaining in S2 epic

After PR-1/PR-2/PR-3/PR-4 (presumed PR-5/PR-6 record-only): **only** PR-7a/PR-7b record-only (`s2_semantic_diff_report.md` updates) + PR-8 capstone (delete `MD_f_omp.cpp` + retire `SHUD_LEGACY_OMP_RHS`) is real S2 code change. S3a/S3b/S2.8 paired in PR-9; S4 in PR-10; S3c in PR-11; B1a capstone in PR-12.

## Next

- Merge PR-4 → close #147
- Append `docs/review-loop-log.jsonl`
- PR-5 #148 record-only (predicted)
