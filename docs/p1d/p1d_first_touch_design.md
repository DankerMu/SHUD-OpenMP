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
