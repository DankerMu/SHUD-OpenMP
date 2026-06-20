# PR-6 (S2.4) — Record-only PR evidence

Issue: DankerMu/SHUD-OpenMP#149 (PR-6 of `b1a-finalization`)
Outer branch: `feat/issue-149-pr6-river-dy-record` (from `baseline/B1a` `7c2cb57`)
SHUD branch: `openmp-baseline` HEAD unchanged at `8fa6b29` (no PR-6 SHUD commit)
File touched: **NONE in SHUD/** + `docs/.s2-pr6-evidence.md` (this file, NEW)

## TL;DR

PR-6 is **record-only** (cumulative pattern from PR-3 + PR-4 + PR-5). S2.4 river DY 公式（`length` + `area clamp` + `fun_dAtodY()`）已经在 `rhs_core()` 内（S1c PURE CARRY-OVER `rhs_apply()`），且 `MD_f_omp.cpp` 直接除 `u_TopArea` 路径已通过 `SHUD_LEGACY_OMP_RHS=0` default 结构性弃用（整个 TU 不参与 default build）。PR-6 makes 0 SHUD source change.

## S2.4 Requirement & verification

**Spec**: 工程 SHALL 在 `rhs_core()` 中用 serial 公式计算 river DY，即 `length` 除项 + `area clamp` + `fun_dAtodY()` 三段；`MD_f_omp.cpp` 直接除 `u_TopArea` 路径弃用。

**Reality** (`MD_rhs_core.cpp` `rhs_apply()` body NumRiv loop, ~L70-80):

```cpp
if(Riv[i].Length < 1e-3){
    DY[iRIV] = 0.;
}else{
    DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length; // dA on CS
    if(DY[iRIV] < -1. * Riv[i].u_CSarea){ /* The negative dA cannot larger then Availalbe Area. */
        DY[iRIV] = -1. * Riv[i].u_CSarea;
    }
    DY[iRIV] = fun_dAtodY(DY[iRIV], Riv[i].u_topWidth, Riv[i].bankslope);
}
```

PURE CARRY-OVER source: `SHUD/src/ModelData/MD_f.cpp` `f_applyDY()` body L119-141 — byte-for-byte identical.

Three S2.4 components all present:
1. **Length divisor**: `/ Riv[i].Length` (not `u_TopArea`)
2. **Area clamp**: `if(DY[iRIV] < -1. * Riv[i].u_CSarea){ DY[iRIV] = -1. * Riv[i].u_CSarea; }` (negative dA bounded)
3. **fun_dAtodY**: `DY[iRIV] = fun_dAtodY(DY[iRIV], Riv[i].u_topWidth, Riv[i].bankslope)` (dA → dY conversion)

## MD_f_omp.cpp 直接除路径状态：structural deprecation

`SHUD/Makefile`:
```makefile
SHUD_LEGACY_OMP_RHS ?= 0
ifeq ($(SHUD_LEGACY_OMP_RHS),0)
  # MD_f_omp.cpp excluded from default build
...
```

Default `make shud` / `make shud_omp` build 不包含 `MD_f_omp.cpp`。其 `f_applyDY_omp()` 的 `DY[i] = QrivIn[i] / u_TopArea[i]` 直接除路径无法被运行时调用。**Structural deprecation** 已达成。

**Physical deletion** 推至 PR-8 capstone（同时退役 macro）。本 PR record-only 不动该文件。

## Grep gates (4/4 PASS)

```bash
# Gate 1: serial DY 公式 with Length divisor
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'DY\[iRIV\][[:space:]]*=[[:space:]]*\(- QrivUp\[i\] - QrivSurf\[i\] - QrivSub\[i\] - QrivDown\[i\] \+ Riv\[i\]\.qBC\) / Riv\[i\]\.Length'
# Expected: 1 ; Actual: 1 ✓

# Gate 2: area clamp 防 over-drain
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'if\(DY\[iRIV\] < -1\. \* Riv\[i\]\.u_CSarea\)'
# Expected: 1 ; Actual: 1 ✓

# Gate 3: fun_dAtodY conversion
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'DY\[iRIV\][[:space:]]*=[[:space:]]*fun_dAtodY\(DY\[iRIV\], Riv\[i\]\.u_topWidth, Riv\[i\]\.bankslope\)'
# Expected: 1 ; Actual: 1 ✓

# Gate 4: MD_f_omp.cpp structurally deprecated via Makefile default
grep -cE 'SHUD_LEGACY_OMP_RHS \?= 0' SHUD/Makefile
# Expected: 1 ; Actual: 1 ✓
```

## Cross-validation: rhs_apply 完整与 f_applyDY 一致

Already verified in PR-4 evidence: `diff <(awk '...rhs_apply...' MD_rhs_core.cpp) <(awk '...f_applyDY...' MD_f.cpp)` = function name + 3 trailing whitespace (no semantic diff). Therefore river DY block inherited byte-for-byte from S1c PURE CARRY-OVER source.

## Anti-pattern verification: NO `DY[i] = QrivIn[i] / u_TopArea[i]` in active build

```bash
# Check rhs_apply (active path) has NO direct-divide-by-u_TopArea for river
awk '/^void Model_Data::rhs_apply/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | grep -cE 'DY\[iRIV\][[:space:]]*=[[:space:]]*QrivIn\[i\] / u_TopArea'
# Expected: 0 ; Actual: 0 ✓ (direct-divide pattern absent from active core)

# Anti-pattern only exists in dormant TU
grep -nE 'DY\[i\][[:space:]]*=[[:space:]]*\(' SHUD/src/ModelData/MD_f_omp.cpp 2>/dev/null | grep -E 'QrivIn|u_TopArea' | head -3
# Confirms direct-divide pattern is in MD_f_omp.cpp only (dormant; not compiled)
```

## Verified at PR-6 HEAD (= baseline/B1a HEAD `7c2cb57` + this evidence file)

- SHUD source diff vs `baseline/B1a 7c2cb57`: **0 lines**
- 4-case Mac local + qhh `rivqdown.dat` bitwise vs B0: **PASS** (inherits PR-5 merged state)
- CVODE 15-key (4-case): byte-equal B0-tag archive (inherits PR-5)
- LEGACY_RHS=1 build sanity: PASS (no SHUD source change; macro still functional)
- 12 snapshot bitwise: PASS (PR-1 PATH A inherits)
- S1 4-grep gate: 0/0/0/7 (unchanged from PR-1+...+PR-5 baseline)

## SHUD submodule discipline

- SHUD HEAD: `8fa6b29` (unchanged from PR-2)
- `.gitmodules` URL: `SHUD-System/SHUD.git` (unchanged)
- No SHUD branch push (record-only)

## Files in this PR

- `docs/.s2-pr6-evidence.md` (NEW, this file)

## Cumulative pattern completion (S2 record-only batch)

After PR-3+PR-4+PR-5+PR-6, all S2 "merge serial X to rhs_core" Requirements that map 1:1 to PURE CARRY-OVER lines are confirmed auto-satisfied:

| Sub-item | Description | Carry-over source | Verified PR |
|---|---|---|---|
| S2.1 | Lake vertical | S1b `rhs_flux()` | PR-4 |
| S2.2 | Lake horizontal | S1b `rhs_flux()` | PR-4 |
| S2.3 | ET flux non-lake | S1b `rhs_flux()` | PR-5 |
| S2.4 | River DY serial | S1c `rhs_apply()` | PR-6 (this PR) |
| S2.5 | Lake DY | S1c `rhs_apply()` | PR-4 |
| S2.7 | Lake-related reset | S1a `rhs_update()` | PR-3 |
| S2.11 | Lake DY=0 | S1c `rhs_apply()` | PR-4 |

## Remaining S2 active code change

- PR-7a #150 [S2.12/13/15/16] — record-only diff report (4 items)
- PR-7b #151 [S2.17] — assert + DEBUG (real code: macro guard + assert)
- PR-8 #152 [S2-capstone] — DELETE `MD_f_omp.cpp` + retire `SHUD_LEGACY_OMP_RHS` (capstone semantic)
- PR-9 #153 [S3a + S3b + S2.8 D14] — dead-code + shared-write split (large semantic)

## Next

- Merge PR-6 → close #149
- Append `docs/review-loop-log.jsonl`
- PR-7a #150 record-only batch (predicted)
