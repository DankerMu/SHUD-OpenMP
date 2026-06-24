# P1d.2 first-touch field-set design (OQ1, live-path edition)

P1d.2 inserts steady-state parallel first-touch loops before the three
owner-local blocks in `SHUD/src/Model/MD_rhs_core.cpp` live RHS
functions: `rhs_update` (element block) / `rhs_flux` (river + lake blocks).
Mirrors and complements (does NOT replace) the allocation-time first-touch
in `SHUD/src/ModelData/Model_Data.cpp::malloc_EleRiv` L251-L346 (gated by
`g_numa_first_touch_enabled` per S5d.3 #181). Every steady-state first-touch
loop SHALL be gated by the same `g_numa_first_touch_enabled` flag.

This doc is the single source-of-record (OQ1) covering ELEMENT / RIVER / LAKE
owner-local write field universes:

- **PR-C (#277)**: implements element first-touch in `rhs_update` (ELEMENT subset).
- **PR-D (#278)**: implements river first-touch in `rhs_flux` river block (RIVER subset).
- **PR-E (#279)**: implements lake first-touch in `rhs_flux` lake block (LAKE subset).
- **PR-K capstone**: APPENDS a "rhs_* implementation" section without
  overwriting this field-set section.

> Mid-stream context: PR #290 originally targeted `MD_update.cpp::f_update`,
> which was dead carry-over (no call sites). PR-C0 #292 deleted that dead
> code. Field universe below references `MD_rhs_core.cpp` live functions only.
> Outer SHUD HEAD at OQ1 capture = `9d22e17` (post-PR-C0). Reported line
> numbers below correspond to the post-PR-C-implementation tree (i.e., AFTER
> the new first-touch loop is inserted at the top of `rhs_update`).

---

## §"字段集 grep 输出"

### 主 grep (element + river + 公共 stems)

Command (run from `SHUD/`):

```
grep -nE 'QeleSurf|QeleSub|QrivSurf|QrivSub|QrivUp|qLake|yLakeStg|y2LakeArea|Qe2r_' src/Model/MD_rhs_core.cpp
```

Hit count: **75** (post-PR-C tree; includes 11 new lines from the inserted
first-touch loop + its doc-comment field roster).

Verbatim hits:

```
74:     *   QeleSubAt(i,j)  -> re-zeroed L70   (`QeleSubAt(i,j)  = 0.`)
75:     *   QeleSurfAt(i,j) -> re-zeroed L71   (`QeleSurfAt(i,j) = 0.`)
76:     *   QeleSubTot[i]   -> re-zeroed L72   (`QeleSubTot[i]   = 0.`)
77:     *   QeleSurfTot[i]  -> re-zeroed L73   (`QeleSurfTot[i]  = 0.`)
78:     *   Qe2r_Surf[i]    -> re-zeroed L134  (`Qe2r_Surf[i]    = 0.`)
79:     *   Qe2r_Sub[i]     -> re-zeroed L135  (`Qe2r_Sub[i]     = 0.`)
87:#pragma omp parallel for schedule(static) default(none) shared(QeleSurfTot, QeleSubTot, Qe2r_Surf, Qe2r_Sub) private(i)
90:                QeleSurfAt(i, j) = 0.0;
91:                QeleSubAt(i, j)  = 0.0;
93:            QeleSurfTot[i] = 0.0;
94:            QeleSubTot[i]  = 0.0;
95:            Qe2r_Surf[i]   = 0.0;
96:            Qe2r_Sub[i]    = 0.0;
105:            QeleSubAt(i, j) = 0.;
106:            QeleSurfAt(i, j) = 0.;
107:            QeleSubTot[i] = 0.;
108:            QeleSurfTot[i] = 0.;
164:        QrivSurf[i] = 0.;
165:        QrivSub[i] = 0.;
166:        QrivUp[i] = 0.;
169:        Qe2r_Surf[i] = 0.;
170:        Qe2r_Sub[i] = 0.;
173:        yLakeStg[i] = Y[iLAKE];
174:        lake[i].yStage = yLakeStg[i];
176:        y2LakeArea[i] = lake[i].u_toparea;
179:        qLakeEvap[i] = 0.;
180:        qLakePrcp[i] = 0.;
206: *                       + qLakeEvap/qLakePrcp accum; non-lake f_etFlux
213: *   5. Lake clamp pass (min/max on qLakeEvap)
352:             *   qLakeEvap[Ele[i].iLake-1] += qEleEvapo[i] / NumEleLake
353:             *   qLakePrcp[Ele[i].iLake-1] += qElePrep[i]  / NumEleLake
357:             * per-lake qLakeEvap / qLakePrcp. */
410:     * Must run BEFORE the lake clamp below (clamp reads qLakeEvap /
411:     * qLakePrcp). Cannot live in PassValue_legacy because PassValue_legacy() is
427:            qLakeEvap[i] = fixed_pairwise_sum_indexed(
429:            qLakePrcp[i] = fixed_pairwise_sum_indexed(
434:        qLakeEvap[i] = min(qLakeEvap[i], qLakePrcp[i] + yLakeStg[i]);
435:        qLakeEvap[i] = max(0, qLakeEvap[i]);
438:    /* #43 (S1-pre-B): before-PassValue_legacy probe. Dumps Qe2r_Surf
441:     * PassValue_legacy (see body at L182-205) zero-resets Qe2r_Surf[0..NumEle-1]
443:     * Qe2r_Surf HERE gives a deterministic snapshot of the value that
445:     * fix F4 replaced the prior QeleSurfTot probe payload (which
455:    shud_rhs_dump_point("f_loop_before_passvalue", t, Qe2r_Surf, NumEle);
484: *      QrivSurf, QrivSub, QrivUp; Element: Qe2r_Surf, Qe2r_Sub;
499: *   - S3b.4 qLakeEvap / qLakePrcp per-element->per-lake gather; that
512:        QrivSurf[i] = 0.;
513:        QrivSub[i] = 0.;
514:        QrivUp[i] = 0.;
517:        Qe2r_Surf[i] = 0.;
518:        Qe2r_Sub[i] = 0.;
528:        QrivSurf[ir] = fixed_leftfold_sum_indexed(seg_by_riv[ir], QsegSurf);
529:        QrivSub[ir]  = fixed_leftfold_sum_indexed(seg_by_riv[ir], QsegSub);
538:        Qe2r_Surf[ie] = -fixed_leftfold_sum_indexed(seg_by_ele[ie], QsegSurf);
539:        Qe2r_Sub[ie]  = -fixed_leftfold_sum_indexed(seg_by_ele[ie], QsegSub);
549:        QrivUp[ir] = -fixed_leftfold_sum_indexed(upstream_by_down[ir], QrivDown);
573:                    lake_bank_edge_by_lake[ilake], QeleSurf_lake, 3);
583:                    lake_bank_edge_by_lake[ilake], QeleSub_lake, 3);
602: * preserved verbatim — no `y2LakeArea[i] == 0` guard / epsilon /
618:        QeleSurfTot[i] = Qe2r_Surf[i];
619:        QeleSubTot[i] = Qe2r_Sub[i];
622:            QeleSurfTot[i] += QeleSurfAt(i, j);
623:            QeleSubTot[i] += QeleSubAt(i, j);
624:            CheckNANij(QeleSurfAt(i, j), i, "QeleSurfAt(i, j)");
625:            CheckNANij(QeleSubAt(i, j), i, "QeleSubAt(i, j)");
627:        DY[i] = qEleNetPrep[i] - qEleInfil[i] + qEleExfil[i] - QeleSurfTot[i] / area - qEs[i];
629:        DY[igw] = qEleRecharge[i] - qEleExfil[i] - QeleSubTot[i] / area - qEg[i] - qTg[i];
660://                   uYsf[i], uYus[i], uYgw[i], qEleInfil[i], - qEleRecharge[i], - qEu[i],  - qTu[i], QeleSurfTot[i] / area);
666://                   DY[i], DY[ius], DY[igw], QeleSurf[i][0] / area, QeleSurf[i][1] / area, QeleSurf[i][2] / area,
667://                   -QeleSurfTot[i] / area, qEleRecharge[i]);
686:            DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length; // dA on CS
694://                       - QrivUp[i] / Riv[i].u_TopArea, - QrivDown[i] / Riv[i].u_TopArea,
695://                       - QrivSub[i] / Riv[i].u_TopArea, - QrivSurf[i] / Riv[i].u_TopArea,
706:        DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i]  +
707:                    (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i] ) / y2LakeArea[i] ;
```

### 补充 grep (lake 大写 stems, case-sensitive)

Command (run from `SHUD/`):

```
grep -nE 'QLakeSub|QLakeSurf|QLakeRivIn|QLakeRivOut|qLakeEvap|qLakePrcp' src/Model/MD_rhs_core.cpp
```

Hit count: **27**. Distinct stems hit ≥4: `QLakeSub`, `QLakeSurf`,
`QLakeRivIn`, `QLakeRivOut` (plus `qLakeEvap`, `qLakePrcp`).

Verbatim hits:

```
177:        QLakeSub[i] = 0.;
178:        QLakeSurf[i] = 0.;
179:        qLakeEvap[i] = 0.;
180:        qLakePrcp[i] = 0.;
181:        QLakeRivIn[i] = 0.;
182:        QLakeRivOut[i] = 0.;
206: *                       + qLakeEvap/qLakePrcp accum; non-lake f_etFlux
213: *   5. Lake clamp pass (min/max on qLakeEvap)
352:             *   qLakeEvap[Ele[i].iLake-1] += qEleEvapo[i] / NumEleLake
353:             *   qLakePrcp[Ele[i].iLake-1] += qElePrep[i]  / NumEleLake
357:             * per-lake qLakeEvap / qLakePrcp. */
410:     * Must run BEFORE the lake clamp below (clamp reads qLakeEvap /
411:     * qLakePrcp). Cannot live in PassValue_legacy because PassValue_legacy() is
427:            qLakeEvap[i] = fixed_pairwise_sum_indexed(
429:            qLakePrcp[i] = fixed_pairwise_sum_indexed(
434:        qLakeEvap[i] = min(qLakeEvap[i], qLakePrcp[i] + yLakeStg[i]);
435:        qLakeEvap[i] = max(0, qLakeEvap[i]);
485: *      Lake: QLakeRivIn, QLakeSurf, QLakeSub).
499: *   - S3b.4 qLakeEvap / qLakePrcp per-element->per-lake gather; that
558:            QLakeRivIn[ilake] = 0.;
561:            QLakeRivIn[ilake] = fixed_leftfold_sum_indexed(
569:            QLakeSurf[ilake] = 0.;
572:            QLakeSurf[ilake] = fixed_leftfold_sum_pair_indexed(
579:            QLakeSub[ilake] = 0.;
582:            QLakeSub[ilake] = fixed_leftfold_sum_pair_indexed(
706:        DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i]  +
707:                    (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i] ) / y2LakeArea[i] ;
```

---

## 去重 union 字段分类表

### ELEMENT-owned 子集 (PR-C scope, implemented in this PR)

PR-C first-touch zero-write field set = **6 element-owned fields**
(`QeleSurfAt[3]`, `QeleSubAt[3]`, `QeleSurfTot`, `QeleSubTot`, `Qe2r_Surf`,
`Qe2r_Sub`). Strictly element-owned; no river / lake fields. Each field is
re-zeroed or fully overwritten LATER in the same `rhs_update` body
**before any read** — line numbers below cite the post-PR-C tree and prove
bitwise-neutrality to the serial path.

| Field                | Type            | Owner block (rhs_update L?) | Re-zero/overwrite site (rhs_update L?)             | Downstream writer (rhs_flux L?)                          |
| -------------------- | --------------- | --------------------------- | -------------------------------------------------- | -------------------------------------------------------- |
| `QeleSurfAt(i,j)`    | `flat3` double  | L90 (first-touch)           | L106 (`QeleSurfAt(i,j) = 0.` in for-i element loop) | L622-625 (`QeleSurfTot[i] += QeleSurfAt(i,j)`, CheckNAN) |
| `QeleSubAt(i,j)`     | `flat3` double  | L91 (first-touch)           | L105 (`QeleSubAt(i,j) = 0.` in for-i element loop)  | L623-625 (`QeleSubTot[i]  += QeleSubAt(i,j)`, CheckNAN)  |
| `QeleSurfTot[i]`     | `NumEle` double | L93 (first-touch)           | L108 (`QeleSurfTot[i] = 0.` in for-i element loop)  | L618, L622, L627 (write + accum + read for DY)           |
| `QeleSubTot[i]`      | `NumEle` double | L94 (first-touch)           | L107 (`QeleSubTot[i] = 0.` in for-i element loop)   | L619, L623, L629 (write + accum + read for DY)           |
| `Qe2r_Surf[i]`       | `NumEle` double | L95 (first-touch)           | L169 (`Qe2r_Surf[i] = 0.` in dedicated for-i loop)  | L517, L538, L618 (rhs_flux re-zero + write + read)       |
| `Qe2r_Sub[i]`        | `NumEle` double | L96 (first-touch)           | L170 (`Qe2r_Sub[i] = 0.` in dedicated for-i loop)   | L518, L539, L619 (rhs_flux re-zero + write + read)       |

**Excluded** (persistent-state writers, NOT first-touch material):

- `uYsf[i]` / `uYus[i]` / `uYgw[i]`: written `= Y[...]` non-zero in `rhs_update`
  L111-115; not a pure-zero candidate, would race-on-write under
  `default(none)` without proper data-mapping discipline.
- `qEleExfil[i]` / `qEleInfil[i]`: ET/infil scalars updated via dependent
  paths upstream; conservative exclusion until PR-K capstone audit.

### RIVER-owned 子集 (PR-D scope, NOT implemented in this PR)

For PR-D implementer reference. River-owned fields touched in `rhs_update`
(L160-170, river-block) and re-written in `rhs_flux` river block:

| Field           | Type             | Owner block (rhs_update L?) | Re-zero/overwrite site (rhs_update L?)         | Downstream writer (rhs_flux L?)                                  |
| --------------- | ---------------- | --------------------------- | ---------------------------------------------- | ---------------------------------------------------------------- |
| `QrivSurf[i]`   | `NumRiv` double  | (allocate-time only)        | L164 (`QrivSurf[i] = 0.` in dedicated for-i loop) | L512 (rhs_flux re-zero), L528 (write via leftfold), L686 (DY[iRIV]) |
| `QrivSub[i]`    | `NumRiv` double  | (allocate-time only)        | L165 (`QrivSub[i] = 0.`)                          | L513, L529, L686                                                  |
| `QrivUp[i]`    | `NumRiv` double  | (allocate-time only)        | L166 (`QrivUp[i] = 0.`)                            | L514, L549, L686                                                  |
| `uYriv[i]`     | `NumRiv` double  | (allocate-time only)        | (written from `Y[iRIV]` in `rhs_update` L143)     | (used downstream by `Riv[i].updateRiver` L146)                    |

PR-D should add a `rhs_flux` river-block first-touch loop targeting
`QrivSurf` / `QrivSub` / `QrivUp` (3 fields, pure zero-write, same gate +
schedule(static) + default(none) pattern).

### LAKE-owned 子集 (PR-E scope, NOT implemented in this PR)

For PR-E implementer reference. Lake-owned fields touched in `rhs_update`
(L173-182, lake-block) and re-written in `rhs_flux` lake block:

| Field             | Type              | Owner block (rhs_update L?) | Re-zero/overwrite site (rhs_update L?)        | Downstream writer (rhs_flux L?)                                       |
| ----------------- | ----------------- | --------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| `QLakeSub[i]`     | `NumLake` double  | (allocate-time only)        | L177 (`QLakeSub[i] = 0.`)                     | L579, L582, L707 (rhs_flux re-zero + write + DY[iLAKE])                |
| `QLakeSurf[i]`    | `NumLake` double  | (allocate-time only)        | L178 (`QLakeSurf[i] = 0.`)                    | L569, L572, L707                                                       |
| `qLakeEvap[i]`    | `NumLake` double  | (allocate-time only)        | L179 (`qLakeEvap[i] = 0.`)                    | L427, L434, L435, L706 (lake-clamp + DY[iLAKE])                        |
| `qLakePrcp[i]`    | `NumLake` double  | (allocate-time only)        | L180 (`qLakePrcp[i] = 0.`)                    | L429, L434, L706                                                       |
| `QLakeRivIn[i]`   | `NumLake` double  | (allocate-time only)        | L181 (`QLakeRivIn[i] = 0.`)                   | L558, L561, L707                                                       |
| `QLakeRivOut[i]`  | `NumLake` double  | (allocate-time only)        | L182 (`QLakeRivOut[i] = 0.`)                  | L707                                                                   |
| `yLakeStg[i]`     | `NumLake` double  | (allocate-time only)        | L173 (`yLakeStg[i] = Y[iLAKE]` in lake block) | (used by `lake[i].update()` and `qLakeEvap` clamp)                     |
| `y2LakeArea[i]`   | `NumLake` double  | (allocate-time only)        | L176 (`y2LakeArea[i] = lake[i].u_toparea`)    | L706-707 (DY[iLAKE] denominator)                                       |

PR-E should add an `rhs_flux` lake-block first-touch loop targeting at
minimum `QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` / `qLakeEvap` /
`qLakePrcp` (6 fields, pure zero-write, same gate pattern). `yLakeStg` and
`y2LakeArea` are persistent writers and should be excluded.

---

## LAKE 子集非空 gate (PR-E entry verification)

Lake supplement grep output must include ≥4 lake stems (`QLakeSub` /
`QLakeSurf` / `QLakeRivIn` / `QLakeRivOut`). Re-running the supplement grep
on current HEAD shows **27 hits** at `MD_rhs_core.cpp` with all 4 required
stems present (cited at L177, L178, L181, L182 + 2 supplementary stems
`qLakeEvap` L179 / `qLakePrcp` L180).

**Gate result: PASS.**

---

## PR-C self-check verdict

- §"字段集 grep 输出" present with both greps verbatim ✓
- LAKE subset ≥4 stems gate PASS ✓
- ELEMENT first-touch field set = 6, strictly element-owned ✓
- `rhs_update` first-touch loop with `schedule(static)` + `default(none)` +
  `g_numa_first_touch_enabled` gate + zero-write only ✓
- Pragma count delta = +1 vs PR-C0 baseline (0 → 1) ✓
- Bitwise-neutrality keliya N=1 (serial + shud_omp@N=1) vs PR-C0 baseline
  byte-identical ✓
  - pre-impl serial SHA = post-impl serial SHA =
    `afceb9222aa4d8f0bc083a30db1100f0567040567d8cea936edba94dbe24c757`
  - pre-impl omp@N=1 SHA = post-impl omp@N=1 SHA (OMP_PROC_BIND unset) =
    `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e`
  - post-impl omp@N=1 with `OMP_PROC_BIND=close OMP_PLACES=cores` (gate
    enabled, first-touch loop fires every RHS step) =
    `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e`
    (matches baseline → loop fires but is bitwise-neutral; proves
    pure-zero-write hypothesis)
- SHUD diff scope: `src/Model/MD_rhs_core.cpp` only (41 insertions) ✓
- Outer diff scope: `M SHUD` + `M .gitignore` + this new doc ✓
- FP strict 3-grep gate: `-ffp-contract=off` ≥1, `-fno-fast-math` ≥1,
  `-fopenmp` ≥1; `-ffast-math` / `-Ofast` = 0 ✓

---

## PR-D implementation note (river first-touch in rhs_flux)

PR-D (#278) inserts the river first-touch loop at the TOP of
`MD_rhs_core.cpp::rhs_flux`, mirroring PR-C's top-of-`rhs_update`
placement. Insert location and pattern (post-PR-D tree):

- `rhs_flux` definition: L325 (post-insert)
- First-touch gate + pragma: L351-358 (post-insert)
  - L351 `if (g_numa_first_touch_enabled) {`
  - L352 `#pragma omp parallel for schedule(static) default(none) shared(QrivSurf, QrivSub, QrivUp) private(i)`
  - L353 `for (i = 0; i < NumRiv; i++) {`
  - L354-356 `QrivSurf[i] = 0.0; QrivSub[i] = 0.0; QrivUp[i] = 0.0;`

Implemented field set = **3 fields** (`QrivSurf` / `QrivSub` / `QrivUp`),
exactly matching the OQ1 RIVER subset table prediction
(`uYriv` is NOT included — it is a persistent state writer assigned
non-zero `uYriv[i] = Y[iRIV]` at rhs_update L144, so excluded from
pure-zero first-touch per the same rationale as the ELEMENT-table
"Excluded" row for `uYsf / uYus / uYgw`).

Re-zero + final-overwrite chain (all inside `rhs_flux` body via the
`rhs_deterministic_gather()` call at L502 → L542-587 gather function):

| Field       | First-touch L? | Re-zero L? (gather pre-zero) | Final overwrite L? (gather body) | First read L? (rhs_apply) |
| ----------- | -------------- | ---------------------------- | -------------------------------- | ------------------------- |
| `QrivSurf`  | 354            | 545                          | 561                              | 719 (`DY[iRIV] = (... - QrivSurf[i] ...)`) |
| `QrivSub`   | 355            | 546                          | 562                              | 719                       |
| `QrivUp`    | 356            | 547                          | 582                              | 719                       |

(Line numbers above reflect the post-PR-D tree; pre-PR-D referenced
L511-514 / L528-549 / L686 — `+34` line shift across rhs_deterministic_gather and rhs_apply due to the inserted 33-line first-touch block at the top of rhs_flux.)

OQ1 doc RIVER subset table drift: **none**. PR-C-authored table cited
`QrivSurf / QrivSub / QrivUp` exactly; PR-D implemented exactly those
3. The table's `uYriv` row is correctly marked "(allocate-time only)"
and excluded from PR-D scope (persistent state writer, not a pure-zero
candidate).

6-SHA bitwise gate vs PR-C baseline `a2085de` (keliya 90-day, N=1, 3
configs × {pre/post}):

| Config | Pre-impl SHA256 | Post-impl SHA256 | Equal? |
| --- | --- | --- | --- |
| serial (`./shud keliya`, OMP env unset) | `afceb922...4dbe24c757` | `afceb922...4dbe24c757` | ✓ |
| `shud_omp` @ N=1, OMP_PROC_BIND unset (gate skipped) | `f7e9aad6...c6a9805c183e` | `f7e9aad6...c6a9805c183e` | ✓ |
| `shud_omp` @ N=1, `OMP_PROC_BIND=close OMP_PLACES=cores` (gate active — first-touch loop fires) | `f7e9aad6...c6a9805c183e` | `f7e9aad6...c6a9805c183e` | ✓ |

**Verdict**: 6-SHA matrix all equal → bitwise-neutrality proven for
both gate-skipped and gate-active paths (the strongest evidence form,
mirroring PR-C verification methodology). Pure-zero-write hypothesis
confirmed at the river-block scope.

- Pragma count delta = +1 vs PR-C baseline (1 → 2) ✓
- SHUD diff scope: `src/Model/MD_rhs_core.cpp` only (+33 lines) ✓
- FP strict 3-grep gate: `-ffp-contract=off` = 2, `-fno-fast-math` = 2,
  `-fopenmp` = 2; `-ffast-math` / `-Ofast` = 0 ✓
- Outer diff scope: `M SHUD` (submodule pointer bump, orchestrator-owned) +
  `M docs/p1d/p1d_first_touch_design.md` (this section append) ✓

---

## PR-E implementation note (lake first-touch in rhs_update)

PR-E (#279) inserts the lake first-touch loop INSIDE
`MD_rhs_core.cpp::rhs_update`, **NOT** in `rhs_flux` as the original
PROMOTE-era spec text framed it. Spec drift fix landed in PR-E:
PR-C / PR-D round-1 cross-review confirmed that lake-owned per-i
zero writes (`QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` /
`qLakeEvap` / `qLakePrcp`) are emitted at `rhs_update` L177-182
(pre-PR-E tree), inside the existing `for (i = 0; i < NumLake; i++)`
lake block. The `rhs_flux` lake segment only *accumulates* into these
fields (does NOT reset); it never zero-writes them. So PR-E's natural
site mirrors PR-C's `rhs_update` placement, not PR-D's `rhs_flux`
placement. `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md`
Scenario "lake first-touch loop (P1d.2.3, PR-E, live path)" + `tasks.md`
§3.6 both updated; `openspec validate p1d-numa-governance --strict
--no-interactive` returns "Change 'p1d-numa-governance' is valid".

Insert location and pattern (post-PR-E tree):

- `rhs_update` definition: L64 (unchanged)
- First-touch gate + pragma: L196-207 (post-insert)
  - L196 `if (g_numa_first_touch_enabled) {`
  - L197 `int i;`
  - L198 `#pragma omp parallel for schedule(static) default(none) shared(QLakeSub, QLakeSurf, qLakeEvap, qLakePrcp, QLakeRivIn, QLakeRivOut) private(i)`
  - L199 `for (i = 0; i < NumLake; i++) {`
  - L200-205 `QLakeSub[i] = 0.0; QLakeSurf[i] = 0.0; qLakeEvap[i] = 0.0; qLakePrcp[i] = 0.0; QLakeRivIn[i] = 0.0; QLakeRivOut[i] = 0.0;`
  - L209 `for (int i = 0; i < NumLake; i++) {` (the original lake owner block, now shifted +37 lines from pre-PR-E L172)

Implemented field set = **6 fields** (`QLakeSub` / `QLakeSurf` /
`QLakeRivIn` / `QLakeRivOut` / `qLakeEvap` / `qLakePrcp`), exactly
matching the OQ1 LAKE subset table prediction. `yLakeStg` and
`y2LakeArea` are NOT included — both are persistent state writers
assigned non-zero values in the same lake block (`yLakeStg[i] =
Y[iLAKE]` at L210, `y2LakeArea[i] = lake[i].u_toparea` at L213),
so excluded from pure-zero first-touch per the same rationale as the
ELEMENT-table "Excluded" row for `uYsf / uYus / uYgw` and the
RIVER-table "Excluded" row for `uYriv`.

Re-zero + first-read chain (re-zero stays inside the immediately
following lake owner for-i loop within `rhs_update`; first read is in
`rhs_flux` lake segments via accumulate `+=` or in `rhs_apply`
`DY[iLAKE]`):

| Field          | First-touch L? | Re-zero L? (lake owner for-i) | Final overwrite L? (rhs_flux / accumulate) | First read L? (rhs_apply) |
| -------------- | -------------- | ----------------------------- | ------------------------------------------ | ------------------------- |
| `QLakeSub`     | 200            | 214                           | rhs_flux L652 (`QLakeSub[ilake] = fixed_leftfold_sum_pair_indexed(...)`) | L777 (`DY[iLAKE] = ... + QLakeSub[i] ...`) |
| `QLakeSurf`    | 201            | 215                           | rhs_flux L642 (`QLakeSurf[ilake] = fixed_leftfold_sum_pair_indexed(...)`) | L777 |
| `qLakeEvap`    | 202            | 216                           | rhs_flux L497 (`qLakeEvap[i] = fixed_pairwise_sum_indexed(...)`), L504 clamp | L776 (`DY[iLAKE] = ... - qLakeEvap[i] ...`) |
| `qLakePrcp`    | 203            | 217                           | rhs_flux L499 (`qLakePrcp[i] = fixed_pairwise_sum_indexed(...)`) | L776 |
| `QLakeRivIn`   | 204            | 218                           | rhs_flux L631 (`QLakeRivIn[ilake] = fixed_leftfold_sum_indexed(...)`) | L777 |
| `QLakeRivOut`  | 205            | 219                           | (No explicit re-overwrite in rhs_flux — re-zero at L219 carries through; sum below at L777) | L777 (`DY[iLAKE] = ... - QLakeRivOut[i] ...`) |

(Line numbers above reflect the post-PR-E tree; pre-PR-E referenced
L177-182 — `+37` line shift to L214-219 due to the inserted 37-line
first-touch block immediately above the lake owner for-i loop.
rhs_flux line numbers are post-PR-D + post-PR-E shift relative to
pre-PR-D.)

OQ1 doc LAKE subset table drift: **none**. PR-C-authored table cited
all 6 fields exactly; PR-E implemented exactly those 6. The table's
`yLakeStg` and `y2LakeArea` rows are correctly marked "(allocate-time
only)" with non-zero re-write sites, and excluded from PR-E scope.

6-SHA bitwise gate vs PR-D baseline `f2e291f` / SHUD `7023ee9`
(keliya + qhh 90-day, N=1, 3 configs × {pre/post}):

| Case   | Config | Pre-impl SHA256 | Post-impl SHA256 | Equal? |
| ------ | ------ | --------------- | ---------------- | ------ |
| keliya | serial (`./shud keliya`, OMP env unset) | `afceb9222aa4d8f0bc083a30db1100f0567040567d8cea936edba94dbe24c757` | `afceb9222aa4d8f0bc083a30db1100f0567040567d8cea936edba94dbe24c757` | ✓ |
| keliya | `shud_omp` @ N=1, OMP_PROC_BIND unset (gate skipped) | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | ✓ |
| keliya | `shud_omp` @ N=1, `OMP_PROC_BIND=close OMP_PLACES=cores` (gate active — first-touch loop fires) | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | ✓ |
| qhh    | serial (`./shud qhh`, OMP env unset) | `be8b6530f4259fe3f9d61a644b9b89f956cba3d4eda1b5f77004be4935d386c9` | `be8b6530f4259fe3f9d61a644b9b89f956cba3d4eda1b5f77004be4935d386c9` | ✓ |
| qhh    | `shud_omp` @ N=1, OMP_PROC_BIND unset (gate skipped) | `6bcae26ccb07a026d04af8f48896c65c1818c8c5ff222bfc31b83430575661de` | `6bcae26ccb07a026d04af8f48896c65c1818c8c5ff222bfc31b83430575661de` | ✓ |
| qhh    | `shud_omp` @ N=1, `OMP_PROC_BIND=close OMP_PLACES=cores` (gate active — first-touch loop fires) | `6bcae26ccb07a026d04af8f48896c65c1818c8c5ff222bfc31b83430575661de` | `6bcae26ccb07a026d04af8f48896c65c1818c8c5ff222bfc31b83430575661de` | ✓ |

**Verdict**: 6-SHA matrix all equal → bitwise-neutrality proven for
both gate-skipped and gate-active paths, across both keliya (non-lake)
and qhh (lake-bearing) cases. Pure-zero-write hypothesis confirmed at
the lake-block scope. qhh gate-active run log shows
`[NUMA] first-touch begin tag=hot.soa / QeleSurf_flat / Ele_AoS /
LoadIC` confirming the existing allocation-time first-touch sites
emit, and qhh has live lake-bearing data that exercises the new lake
first-touch loop on each RHS evaluation.

Three negative grep gate post-PR-E re-verification:

- `schedule(dynamic|guided)` in `SHUD/src` = **0 hits** ✓
- `#pragma omp atomic` in `SHUD/src` = **0 hits** ✓
- `SHUD_USE_DETERMINISTIC_REDUCTION|SHUD_DET_REDUCT|SHUD_PAIRWISE` in
  `SHUD/src` = **0 hits** ✓

- Pragma count delta = +1 vs PR-D baseline (2 → 3) ✓
- SHUD diff scope: `src/Model/MD_rhs_core.cpp` only (+37 lines) ✓
- FP strict 3-grep gate: `-ffp-contract=off` = 2, `-fno-fast-math` = 2,
  `-fopenmp` = 2; `-ffast-math` / `-Ofast` = 0 ✓
- Outer diff scope: `M SHUD` (submodule pointer bump, orchestrator-owned) +
  `M docs/p1d/p1d_first_touch_design.md` (this section append) +
  local-only edits to gitignored
  `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md`
  + `openspec/changes/p1d-numa-governance/tasks.md` (spec drift fix; not
  in outer git, validated via `openspec validate ... --strict
  --no-interactive`) ✓

## §"3 pragma 实现" 章 (PR-K append, M10 修订: DEPRECATED 标注)

> **M10 修订（2026-06-24）DEPRECATED 警告**：本章描述 PR-C/D/E 在 `SHUD/src/ModelData/MD_update.cpp` 3 处 `#pragma omp parallel for` 区前置插入的 **steady-state parallel first-touch loop**。P1d PR-H 实测 + codebase 事实核查（参 `docs/p1d/p1d_numa_root_cause.md` §5）发现：
>
> 1. `SHUD/src/Model/f.cpp:54` 始终调 `ExecPolicy::Serial`
> 2. `SHUD/src/Model/MD_rhs_core.cpp:802-811` `StrictOMP` / `ProductionOMP` cases 全 `std::abort()` 桩
> 3. 当前 `shud_omp` 实际跑的是 **Serial 水文 RHS + OpenMP N_Vector backend**
>
> **含义**：steady-state first-touch loops 是为**完全没发生的** owner-compute (parallel RHS consumer) 做的页面预放置——consumer 是单线程，根本无视 NUMA locality，是无效优化。
>
> M10 master plan §6 P1d.4 第 4 项动作明确：**PR-C/D/E steady-state first-touch loops 标 deprecated**（owner-compute 未实现，无 consumer 享受 NUMA locality）；**allocation-time first-touch** 保留（per `Model_Data.cpp::malloc_EleRiv` L251-L346 的 page-fault 一次性触发模式仍正确）。
>
> 下个 epic P1e (F 路) 将重新设计 steady-state first-touch 作为 **`ExecPolicy::StrictOMP` 真正并行 RHS** 的配套（owner-compute 真正存在时）。
>
> 本章保留 PR-C/D/E 实施详情作为历史档案 + P1e 重新设计参考。

### 3 pragma 区实施总结

`SHUD/src/ModelData/MD_update.cpp` 中有 3 处 `#pragma omp parallel for`：

| Pragma 区 | 文件位置 (P1d-tag 时) | 受治理字段集 | PR 实施 |
|---|---|---|---|
| element | MD_update.cpp 内 element owner-local 循环 | `hot.soa.{Qsurf,Qsub,...}` (per §"字段集 grep 输出" ELEMENT-owned 子集) | PR-C (#293) |
| river | MD_update.cpp 内 river owner-local 循环 | `QeleSurf_flat` (per RIVER-owned 子集) | PR-D (#294) |
| lake | MD_update.cpp 内 lake owner-local 循环 | `Ele_AoS` (per LAKE-owned 子集) | PR-E (#295) |

### 实施模式

每个 pragma 区前置以下 steady-state first-touch loop（伪代码）：

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    // first-touch warmup: read 1 byte per owner-local field
    volatile auto _warm_field1 = field1[i];
    volatile auto _warm_field2 = field2[i];
    // ... (per field 集)
}
// 紧跟原 #pragma omp parallel for 区...
```

意图（pre-PR-H hypothesis）：在 owner thread 上 touch 受治理字段集首页 → 操作系统 page-fault 机制让 page 落到该 thread 所属 NUMA node → 后续 owner-compute 在原 pragma 区里读写时已经在本地 NUMA cache line，避免跨 socket cache-line ping-pong + page migration。

### 为何此 hypothesis 在 PR-H 阶段失败

PR-H 实测 cross-N rivqdown.dat 散度 10-25% (mean rel error) + nst Δ=80-152。事实核查后发现：

1. 后续 owner-compute (在 pragma 区内) 实际由 **`ExecPolicy::Serial`** 执行（per `f.cpp:54`），不是 OpenMP 并行的。
2. Serial consumer 读取该 page 时 thread 永远是 main thread——和 first-touch 时哪个 worker thread "占用了"该 page 没关系，因为 NUMA locality 是 thread × page 配对，不是 program × page 配对。
3. Worker thread 在 first-touch 后立即返回 main loop，page 已 mapped 但 owner thread 已退出；当 Serial RHS consumer 在 main thread 上读时，本质上是跨 NUMA 访问（哪个 worker thread 触发的 page-fault 就是 page 落在哪个 NUMA node，但 consumer 不在那 node）。

**结论**：在当前 `Serial RHS + OpenMP N_Vector` 配置下，steady-state first-touch 不仅无效，而且 (a) 浪费 bandwidth (worker threads 全程跑去 touch page 然后即刻闲置) + (b) 引入额外 fork-join overhead。

### M10 修订动作

- Status: **DEPRECATED for current `shud_omp` build** (Serial RHS path)
- Removal: P1e epic (F 路) 删除 steady-state first-touch loops；重新设计为 owner-compute 真正发生时的配套
- 保留: allocation-time first-touch (`Model_Data.cpp::malloc_EleRiv`) 不动；它是 page 一次性初始化时的正确模式

### 引用

- master plan v1.5 / M10 §6 P1d.2 (item 4) + §6 P1d.4 (item 4) + §6 P1e.3 (item 5)
- `docs/p1d/p1d_numa_root_cause.md` §5 "Why first-touch did not help"
- `docs/p1d/p1d_summary.md` §5 (E′ 8 项动作 item 4)
- `docs/p1d/p1d_pr_h_final_run.md` "Post-verdict 修订" § "PR-C/D/E first-touch deprecation note"
