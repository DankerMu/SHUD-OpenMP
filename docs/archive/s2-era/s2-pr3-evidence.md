# PR-3 (S2.7 + S2.8) — Record-only PR + S2.8 DEFER to PR-9 evidence

Issue: DankerMu/SHUD-OpenMP#146 (PR-3 of `b1a-finalization`)
Outer branch: `feat/issue-146-pr3-s2-7-s2-8` (from `baseline/B1a` `60685a7`)
SHUD branch: `openmp-baseline` HEAD unchanged at `8fa6b29` (no PR-3 SHUD commit)
File touched: **NONE in SHUD/** + `docs/.s2-pr3-evidence.md` (this file, NEW)

## TL;DR

PR-3 is **record-only**. S2.7 spec criteria already satisfied by `MD_rhs_core.cpp:32-122` `rhs_update()` PURE CARRY-OVER (S1a contract). S2.8 spec premise ("PassValue zero is redundant") proven **wrong** by implementer's bitwise test: 4-case Mac local 4/4 `rivqdown.dat` FAIL vs B0 when PassValue zeros deleted. Root cause = `MD_RiverFlux.cpp:107-108,121-122` `fun_Seg_surface/sub` is a hidden writer that `+=` to the 4 target arrays in rhs_flux segment pass; deleting PassValue's zero causes double-counting. S2.8 implementation **deferred to PR-9** (#153) where it merges with S3a fun_Seg_* dead-`+=` removal.

## S2.7 — Record-only verification (PURE CARRY-OVER, no PR-3 code change)

Spec requirement: `rhs_core()` SHALL contain serial-path lake complete reset (`QLakeSurf` / `QLakeSub` / `QLakeRivIn` / `qLakeEvap` / `qLakePrcp`) every entry.

**Reality (verified)**:
- `MD_rhs_core.cpp:32-122` `rhs_update()` is **PURE CARRY-OVER** of `MD_update.cpp:63-153` `f_update()` per S1a `rhs_core` extraction contract (`MD_rhs_core.cpp:1-25` header commentary: "All variable names, loop bounds, branch predicates, expression trees, floating-point operation order, and global-state read / write timing are preserved byte-for-byte").
- `MD_rhs_core.cpp:104-114` `rhs_update()` lake-init loop:
  ```cpp
  for (int i = 0; i < NumLake; i++) {
      yLakeStg[i] = Y[iLAKE];
      lake[i].yStage = yLakeStg[i];
      lake[i].update();
      y2LakeArea[i] = lake[i].u_toparea;
      QLakeSub[i] = 0.;        // ✓ S2.7 #1
      QLakeSurf[i] = 0.;       // ✓ S2.7 #2
      qLakeEvap[i] = 0.;       // ✓ S2.7 #3 (lowercase q — spec typo fixed)
      qLakePrcp[i] = 0.;       // ✓ S2.7 #4 (lowercase q — spec typo fixed)
      QLakeRivIn[i] = 0.;      // ✓ S2.7 #5
      QLakeRivOut[i] = 0.;     // (extra, NOTE: not in spec list but needed by lake DY)
  }
  ```
- Identical block at `MD_update.cpp:135-146` `f_update()` (PURE CARRY-OVER source).

**S2.7 verdict**: spec criteria already satisfied (5/5 zeros present in `rhs_update()`); PR-3 makes 0 code change to SHUD. Verified via grep gate:

```bash
awk '/^void Model_Data::rhs_update\(/,/^}$/' SHUD/src/Model/MD_rhs_core.cpp | \
  grep -cE '^[[:space:]]*(QLakeSurf|QLakeSub|QLakeRivIn|qLakeEvap|qLakePrcp)\[[a-zA-Z_]+\][[:space:]]*=[[:space:]]*0\.;'
# Expected: 5; Actual: 5 ✓
```

## S2.8 — DEFER to PR-9 (with full root-cause diagnosis)

Spec requirement: delete `PassValue()` body zeros for 4 arrays (`QrivSurf` / `QrivSub` / `Qe2r_Surf` / `Qe2r_Sub`), assuming caller (`rhs_update` L95-103 or legacy `f_update` L127-134) entry-zero is sufficient.

### Phase 1 implementer attempt (FAILED bitwise vs B0)

Implementer applied the minimal-diff edit per task brief:
- `MD_f.cpp:187-188`: delete `QrivSurf[i] = 0.;` + `QrivSub[i] = 0.;` from `for(i<NumRiv)` loop body (retain `QrivUp[i] = 0.;` per spec minimal-diff carve-out)
- `MD_f.cpp:191-194`: delete entire `for(i<NumEle) { Qe2r_Surf[i] = 0.; Qe2r_Sub[i] = 0.; }` loop block
- Net: -6 source lines / +3 comment lines

Build PASS; bitwise vs B0-tag:

| case | verdict | new SHA256 | B0 reference |
|---|---|---|---|
| keliya | **FAIL** | `0e9d20df3827e2733c6ba5b3dc528a801078dd16e10765f871db2625840a120b` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` |
| xinanjiang_upstream | **FAIL** | `10ed83996f89b6ff21c8384ab36c1ac6b77628025ac6a78713bd97f8e6c74787` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` |
| qinyijiang | **FAIL** | `98cd352d120e9eca5ee3a1bdb2db1cb62d1f58838b4f6b2abf038d0368e7ff9d` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` |
| qhh | **FAIL** | `603ae3b658dd8422b070d7428579fa6f937694c7d0778a9ed6f493e7e5c94a3e` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` |
| qhh.lakystage.dat | **FAIL** | `43f8e70c…` | `4fcebe3a…` |
| qhh.lakqrivin.dat | PASS | `1a9db738…` | `1a9db738…` |
| qhh.lakqrivout.dat | PASS | `1a9db738…` | `1a9db738…` |

### Baseline counter-check (stash + rebuild)

Implementer reverted the edit (`git checkout -- src/ModelData/MD_f.cpp`), rebuilt clean on `openmp-baseline 8fa6b29` (PR-2 final state) and reran 4-case + qhh:

| case | verdict | SHA256 |
|---|---|---|
| keliya | PASS | `89686fb8…e99a8fc` |
| xinanjiang_upstream | PASS | `3794e7d3…4023020` |
| qinyijiang | PASS | `48036c5e…6dfbc57` |
| qhh | PASS | `d9a42798…a7b62ce` |

→ Baseline clean; PR-3's S2.8 deletion is the **sole** source of the FAIL.

### Root-cause analysis: hidden writers in `fun_Seg_surface/sub`

S2.8 premise "PassValue zero is redundant" assumes only entry-time (rhs_update / f_update) zeros write the 4 arrays. **Wrong**: `fun_Seg_surface()` / `fun_Seg_sub()` in `MD_RiverFlux.cpp` are hidden writers:

```cpp
// MD_RiverFlux.cpp:100-113 (fun_Seg_surface body)
void Model_Data::fun_Seg_surface(int iEle, int iRiv, int i){
    double isf = uYsf[iEle] - qEleInfil[iEle] + qEleExfil[iEle];
    isf = max(0., isf);
    QsegSurf[i] = WeirFlow_jtoi(...);
    QrivSurf[iRiv]    +=  QsegSurf[i];     // hidden writer #1
    Qe2r_Surf[iEle]   += -QsegSurf[i];     // hidden writer #2
}
// MD_RiverFlux.cpp:114-126 (fun_Seg_sub body)
void Model_Data::fun_Seg_sub( int iEle, int iRiv, int i){
    QsegSub[i] = flux_R2E_GW(...);
    QsegSub[i] *= fu_Sub[iEle];
    QrivSub[iRiv] += QsegSub[i];           // hidden writer #3
    Qe2r_Sub[iEle] += -QsegSub[i];         // hidden writer #4
}
```

These 4 writers are invoked from `rhs_flux()` segment pass (`MD_rhs_core.cpp:185-188`):

```cpp
for (i = 0; i < NumSegmt; i++) {
    fun_Seg_surface(RivSeg[i].iEle-1, RivSeg[i].iRiv-1, i);
    fun_Seg_sub(RivSeg[i].iEle-1, RivSeg[i].iRiv-1, i);
}
```

### Actual flow within a single RHS callback

1. `rhs_update()` entry: 4 arrays = 0 (`MD_rhs_core.cpp:95-103`)
2. `rhs_flux()` Element + Lake passes: do not touch the 4 arrays
3. `rhs_flux()` Segment pass: `fun_Seg_*` `+=` 4 arrays → 4 arrays = sum1 (of QsegSurf/QsegSub)
4. `PassValue()` step 1: re-zero 4 arrays (CLEARS sum1)
5. `PassValue()` step 2: NumSegmt loop `+=` 4 arrays → 4 arrays = sum2 (sum2 == sum1 by determinism)

If PR-3 deletes step 4 PassValue re-zero:
- step 5 NumSegmt loop `+=` on top of step 3's sum1 → 4 arrays = sum1 + sum2 = **2 × correct value**
- → 4-case bitwise FAIL (double-counted river segments)

Confirmed by implementer's empirical test (4/4 FAIL above).

### Why fixture review didn't catch this

Phase 0.5 fixture reviewer Pack scope ("D2 caller pre-zero invariant") traced `rhs_update` L95-103 + `f_update` L127-134 entry zeros + verified PassValue caller invocation order, but **did not trace into `fun_Seg_surface/sub` function bodies** — the 4 `+=` writes are 2 layers deep (`rhs_flux` → segment pass → `fun_Seg_*`). Lesson for future fixture reviews: trace ALL writers (forward dataflow analysis) between caller-entry zero and target function entry, not just direct call-site invariants.

### Resolution: defer S2.8 to PR-9

Per design.md **D14**:
- S2.8 spec amended to two scenarios: (1) PR-3 record-only + defer documentation; (2) PR-9 actual deletion paired with S3a `fun_Seg_*` 4 dead-`+=` removal.
- S3a + S2.8 combined diff in PR-9: `MD_RiverFlux.cpp` -4 lines (delete `QrivSurf/QrivSub/Qe2r_Surf/Qe2r_Sub +=` in `fun_Seg_*`) + `MD_f.cpp` PassValue body -6 lines (delete 4 zeros + empty NumEle loop) + comments.
- PR-9 bitwise gate proves S3a + S2.8 net change is invariant-preserving (rhs_update entry zero becomes single zero source; PassValue NumSegmt loop becomes single += source).

## Cross-platform split (per D10)

Mac local 4-case + qhh: this PR record-only (0 code change → baseline 4-case PASS inherits PR-2 evidence Step 1).
Server heihe / heihe_x4: deferred to PR-12 capstone per D10 cross-platform split.

## Verified at PR-3 HEAD (= baseline/B1a HEAD `60685a7` + this evidence file)

- SHUD source diff vs `baseline/B1a` `60685a7`: **0 lines**
- 4-case Mac local + qhh `rivqdown.dat` bitwise vs B0: **PASS** (inherits PR-2 evidence Step 1; 0 source diff means 0 runtime change)
- CVODE 15-key (4-case): byte-equal B0-tag archive (inherits PR-2 evidence Step 1)
- LEGACY_RHS=1 build sanity: PASS (no SHUD source change)
- 12 snapshot bitwise: PASS (PR-1 PATH A inherits)
- S1 4-grep gate: 0/0/0/7 (unchanged, PR-1+PR-2 baseline)

## SHUD submodule discipline

- SHUD HEAD: `8fa6b29` (unchanged from PR-2 merge)
- `.gitmodules` URL: `SHUD-System/SHUD.git` (unchanged)
- No SHUD branch / master push (record-only PR has nothing to commit on SHUD side)

## Files in this PR

- `docs/.s2-pr3-evidence.md` (NEW, this file)

## Form (record-only)

PR-3 is the rare "record-only" PR in this epic: 0 SHUD source diff + 0 SHUD pointer bump + 1 outer-repo docs commit. Differs from PR-7a / PR-7b (also record-only) which ALSO write `docs/b1b/s2_semantic_diff_report.md` entries. PR-3 currently has only this evidence file (S2.8 defer details belong here, not in the eventual `s2_semantic_diff_report.md` which is record-of-decision rather than implementation-evidence).

## R10 unblock confirmation for PR-4

design.md R10 says "S2.7 PR-3 SHALL be merged before PR-4 lake merge to prevent stale lake array reads". S2.7 already in place (5+1 zeros in `rhs_update()` since S1a). PR-3 merge formally records this state + closes #146; PR-4 (#147 lake merge) free to proceed.

## Next

- Merge PR-3 → close #146 (manual close per baseline/B1a base)
- Append review-loop-log.jsonl
- PR-9 #153 issue body amend: orchestrator append "PR-9 同时实施 S2.8（per D14 deferral from PR-3）"
- PR-4 #147 [S2.1+S2.2+S2.5+S2.11] proceed
