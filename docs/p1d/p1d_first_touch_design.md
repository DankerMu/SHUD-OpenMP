# P1d.2 first-touch field-set design (OQ1)

P1d.2 在 `SHUD/src/ModelData/MD_update.cpp::f_update`（每个 CVODE RHS 评估调用一次）的
三处 `#pragma omp parallel for`（element / river / lake owner-local 循环）前置 steady-state
parallel first-touch loop，使 hot 数组的 page-fault 在并行 region 内由对应 owner thread 触发，
与 `Model_Data.cpp::malloc_EleRiv` L251-L346 的 allocation-time first-touch **互补**（不替换）。

本 doc 是 OQ1 的单 source-of-record，覆盖 element / river / lake **三 pragma owner-local write 全集**：

- **PR-C（本 PR, #277）**：实施 element first-touch loop（本 doc 的 ELEMENT 子集）。
- **PR-D（#278 river）/ PR-E（#279 lake）**：复用本 doc 的 RIVER / LAKE 子集，**不再单独建 doc**。
- **PR-K（capstone）**：仅 APPEND 一节 "3 pragma 实现"，**不** pre-write。

设计模板见 `openspec/changes/p1d-numa-governance/design.md` D2；reduction-order 风险见 R1；
字段宇宙问题见 OQ1；FP grep gate 见 OQ4。

> Cross-ref（本分支布局）：P1c 文档在 `docs/` 根（如 `docs/p1c_summary.md` /
> `docs/p1c_a3a_root_cause.md` / `docs/p1c_kahan_patch.diff`），**不在** `docs/p1c/`。
> P1d 文档在 `docs/p1d/`（如 `docs/p1d/p1d_numa_env_runbook.md`）。

---

## §"字段集 grep 输出"

下面两条 grep 逐字执行于 PR-C 实施后（含本 PR 新增 element first-touch loop）的 SHUD 源树，
命中行完整粘贴；其后给出去重 union 分类表。

### 主 grep（element + river + 公共 stems）

命令：

```
grep -nE 'QeleSurf|QeleSub|QrivSurf|QrivSub|QrivUp|qLake|yLakeStg|y2LakeArea|Qe2r_' \
     SHUD/src/ModelData/MD_update.cpp SHUD/src/Model/MD_rhs_core.cpp
```

实际命中行（93 行）：

```
SHUD/src/ModelData/MD_update.cpp:37://                QrivSurf[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:38://                QrivSub[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:39:                QrivUp[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:72:     * f_update before any read: QeleSub / QeleSurf / QeleSurfTot /
SHUD/src/ModelData/MD_update.cpp:73:     * QeleSubTot by the element pragma just below (this loop), and
SHUD/src/ModelData/MD_update.cpp:74:     * Qe2r_Surf / Qe2r_Sub by the serial reset loop further down, so the
SHUD/src/ModelData/MD_update.cpp:79:#pragma omp parallel for schedule(static) default(none) shared(QeleSurfTot, QeleSubTot, Qe2r_Surf, Qe2r_Sub) private(i)
SHUD/src/ModelData/MD_update.cpp:82:            QeleSurfAt(i, j) = 0.0;
SHUD/src/ModelData/MD_update.cpp:83:            QeleSubAt(i, j)  = 0.0;
SHUD/src/ModelData/MD_update.cpp:85:        QeleSurfTot[i] = 0.0;
SHUD/src/ModelData/MD_update.cpp:86:        QeleSubTot[i]  = 0.0;
SHUD/src/ModelData/MD_update.cpp:87:        Qe2r_Surf[i]   = 0.0;
SHUD/src/ModelData/MD_update.cpp:88:        Qe2r_Sub[i]    = 0.0;
SHUD/src/ModelData/MD_update.cpp:90:#pragma omp parallel for schedule(static) default(none) shared(Y, t, uYsf, uYus, uYgw, qEleExfil, qEleInfil, QeleSurfTot, QeleSubTot) private(i)
SHUD/src/ModelData/MD_update.cpp:96:            QeleSubAt(i, j) = 0.;
SHUD/src/ModelData/MD_update.cpp:97:            QeleSurfAt(i, j) = 0.;
SHUD/src/ModelData/MD_update.cpp:98:            QeleSubTot[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:99:            QeleSurfTot[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:156:        QrivSurf[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:157:        QrivSub[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:158:        QrivUp[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:161:        Qe2r_Surf[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:162:        Qe2r_Sub[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:166:        yLakeStg[i] = Y[iLAKE];
SHUD/src/ModelData/MD_update.cpp:167:        lake[i].yStage = yLakeStg[i];
SHUD/src/ModelData/MD_update.cpp:169:        y2LakeArea[i] = lake[i].u_toparea;
SHUD/src/ModelData/MD_update.cpp:172:        qLakeEvap[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:173:        qLakePrcp[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:238:        yLakeStg[i] = Y5[i];
SHUD/src/ModelData/MD_update.cpp:240:    Sub2Global(yEleSurf, yEleUnsat, yEleGW, yRivStg, yLakeStg, NumEle, NumRiv, NumLake);
SHUD/src/ModelData/MD_update.cpp:278:            fprintf (fp, "%d\t%lf\n", i+1,yLakeStg[i]);
SHUD/src/Model/MD_rhs_core.cpp:64:            QeleSubAt(i, j) = 0.;
SHUD/src/Model/MD_rhs_core.cpp:65:            QeleSurfAt(i, j) = 0.;
SHUD/src/Model/MD_rhs_core.cpp:66:            QeleSubTot[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:67:            QeleSurfTot[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:123:        QrivSurf[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:124:        QrivSub[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:125:        QrivUp[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:128:        Qe2r_Surf[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:129:        Qe2r_Sub[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:132:        yLakeStg[i] = Y[iLAKE];
SHUD/src/Model/MD_rhs_core.cpp:133:        lake[i].yStage = yLakeStg[i];
SHUD/src/Model/MD_rhs_core.cpp:135:        y2LakeArea[i] = lake[i].u_toparea;
SHUD/src/Model/MD_rhs_core.cpp:138:        qLakeEvap[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:139:        qLakePrcp[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:165: *                       + qLakeEvap/qLakePrcp accum; non-lake f_etFlux
SHUD/src/Model/MD_rhs_core.cpp:172: *   5. Lake clamp pass (min/max on qLakeEvap)
SHUD/src/Model/MD_rhs_core.cpp:311:             *   qLakeEvap[Ele[i].iLake-1] += qEleEvapo[i] / NumEleLake
SHUD/src/Model/MD_rhs_core.cpp:312:             *   qLakePrcp[Ele[i].iLake-1] += qElePrep[i]  / NumEleLake
SHUD/src/Model/MD_rhs_core.cpp:316:             * per-lake qLakeEvap / qLakePrcp. */
SHUD/src/Model/MD_rhs_core.cpp:369:     * Must run BEFORE the lake clamp below (clamp reads qLakeEvap /
SHUD/src/Model/MD_rhs_core.cpp:370:     * qLakePrcp). Cannot live in PassValue_legacy because PassValue_legacy() is
SHUD/src/Model/MD_rhs_core.cpp:386:            qLakeEvap[i] = fixed_pairwise_sum_indexed(
SHUD/src/Model/MD_rhs_core.cpp:388:            qLakePrcp[i] = fixed_pairwise_sum_indexed(
SHUD/src/Model/MD_rhs_core.cpp:393:        qLakeEvap[i] = min(qLakeEvap[i], qLakePrcp[i] + yLakeStg[i]);
SHUD/src/Model/MD_rhs_core.cpp:394:        qLakeEvap[i] = max(0, qLakeEvap[i]);
SHUD/src/Model/MD_rhs_core.cpp:397:    /* #43 (S1-pre-B): before-PassValue_legacy probe. Dumps Qe2r_Surf
SHUD/src/Model/MD_rhs_core.cpp:400:     * PassValue_legacy (see body at L182-205) zero-resets Qe2r_Surf[0..NumEle-1]
SHUD/src/Model/MD_rhs_core.cpp:402:     * Qe2r_Surf HERE gives a deterministic snapshot of the value that
SHUD/src/Model/MD_rhs_core.cpp:404:     * fix F4 replaced the prior QeleSurfTot probe payload (which
SHUD/src/Model/MD_rhs_core.cpp:414:    shud_rhs_dump_point("f_loop_before_passvalue", t, Qe2r_Surf, NumEle);
SHUD/src/Model/MD_rhs_core.cpp:443: *      QrivSurf, QrivSub, QrivUp; Element: Qe2r_Surf, Qe2r_Sub;
SHUD/src/Model/MD_rhs_core.cpp:458: *   - S3b.4 qLakeEvap / qLakePrcp per-element->per-lake gather; that
SHUD/src/Model/MD_rhs_core.cpp:471:        QrivSurf[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:472:        QrivSub[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:473:        QrivUp[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:476:        Qe2r_Surf[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:477:        Qe2r_Sub[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:487:        QrivSurf[ir] = fixed_leftfold_sum_indexed(seg_by_riv[ir], QsegSurf);
SHUD/src/Model/MD_rhs_core.cpp:488:        QrivSub[ir]  = fixed_leftfold_sum_indexed(seg_by_riv[ir], QsegSub);
SHUD/src/Model/MD_rhs_core.cpp:497:        Qe2r_Surf[ie] = -fixed_leftfold_sum_indexed(seg_by_ele[ie], QsegSurf);
SHUD/src/Model/MD_rhs_core.cpp:498:        Qe2r_Sub[ie]  = -fixed_leftfold_sum_indexed(seg_by_ele[ie], QsegSub);
SHUD/src/Model/MD_rhs_core.cpp:508:        QrivUp[ir] = -fixed_leftfold_sum_indexed(upstream_by_down[ir], QrivDown);
SHUD/src/Model/MD_rhs_core.cpp:532:                    lake_bank_edge_by_lake[ilake], QeleSurf_lake, 3);
SHUD/src/Model/MD_rhs_core.cpp:542:                    lake_bank_edge_by_lake[ilake], QeleSub_lake, 3);
SHUD/src/Model/MD_rhs_core.cpp:561: * preserved verbatim — no `y2LakeArea[i] == 0` guard / epsilon /
SHUD/src/Model/MD_rhs_core.cpp:577:        QeleSurfTot[i] = Qe2r_Surf[i];
SHUD/src/Model/MD_rhs_core.cpp:578:        QeleSubTot[i] = Qe2r_Sub[i];
SHUD/src/Model/MD_rhs_core.cpp:581:            QeleSurfTot[i] += QeleSurfAt(i, j);
SHUD/src/Model/MD_rhs_core.cpp:582:            QeleSubTot[i] += QeleSubAt(i, j);
SHUD/src/Model/MD_rhs_core.cpp:583:            CheckNANij(QeleSurfAt(i, j), i, "QeleSurfAt(i, j)");
SHUD/src/Model/MD_rhs_core.cpp:584:            CheckNANij(QeleSubAt(i, j), i, "QeleSubAt(i, j)");
SHUD/src/Model/MD_rhs_core.cpp:586:        DY[i] = qEleNetPrep[i] - qEleInfil[i] + qEleExfil[i] - QeleSurfTot[i] / area - qEs[i];
SHUD/src/Model/MD_rhs_core.cpp:588:        DY[igw] = qEleRecharge[i] - qEleExfil[i] - QeleSubTot[i] / area - qEg[i] - qTg[i];
SHUD/src/Model/MD_rhs_core.cpp:619://                   uYsf[i], uYus[i], uYgw[i], qEleInfil[i], - qEleRecharge[i], - qEu[i],  - qTu[i], QeleSurfTot[i] / area);
SHUD/src/Model/MD_rhs_core.cpp:625://                   DY[i], DY[ius], DY[igw], QeleSurf[i][0] / area, QeleSurf[i][1] / area, QeleSurf[i][2] / area,
SHUD/src/Model/MD_rhs_core.cpp:626://                   -QeleSurfTot[i] / area, qEleRecharge[i]);
SHUD/src/Model/MD_rhs_core.cpp:645:            DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length; // dA on CS
SHUD/src/Model/MD_rhs_core.cpp:653://                       - QrivUp[i] / Riv[i].u_TopArea, - QrivDown[i] / Riv[i].u_TopArea,
SHUD/src/Model/MD_rhs_core.cpp:654://                       - QrivSub[i] / Riv[i].u_TopArea, - QrivSurf[i] / Riv[i].u_TopArea,
SHUD/src/Model/MD_rhs_core.cpp:665:        DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i]  +
SHUD/src/Model/MD_rhs_core.cpp:666:                    (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i] ) / y2LakeArea[i] ;
SHUD/src/Model/MD_rhs_core.cpp:668://            printf("%f: %g + %g\n",t, yLakeStg[i], DY[iLAKE]);
```

> 注：`qLake` 小写正则 stem **不**命中 `QLake*`（大写 Q），故主 grep 的 `QLakeSub` /
> `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` 仅以子串方式偶现于 MD_rhs_core.cpp 注释/DY 行
> （L443/L444/L666 等），lake owner-local 的 zero-write site 需补充 grep 才完整覆盖。

### 补充 grep（lake 大写 stems，case-sensitive）

命令：

```
grep -nE 'QLakeSub|QLakeSurf|QLakeRivIn|QLakeRivOut|qLakeEvap|qLakePrcp' \
     SHUD/src/ModelData/MD_update.cpp SHUD/src/Model/MD_rhs_core.cpp
```

实际命中行（33 行）：

```
SHUD/src/ModelData/MD_update.cpp:170:        QLakeSub[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:171:        QLakeSurf[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:172:        qLakeEvap[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:173:        qLakePrcp[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:174:        QLakeRivIn[i] = 0.;
SHUD/src/ModelData/MD_update.cpp:175:        QLakeRivOut[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:136:        QLakeSub[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:137:        QLakeSurf[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:138:        qLakeEvap[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:139:        qLakePrcp[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:140:        QLakeRivIn[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:141:        QLakeRivOut[i] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:165: *                       + qLakeEvap/qLakePrcp accum; non-lake f_etFlux
SHUD/src/Model/MD_rhs_core.cpp:172: *   5. Lake clamp pass (min/max on qLakeEvap)
SHUD/src/Model/MD_rhs_core.cpp:311:             *   qLakeEvap[Ele[i].iLake-1] += qEleEvapo[i] / NumEleLake
SHUD/src/Model/MD_rhs_core.cpp:312:             *   qLakePrcp[Ele[i].iLake-1] += qElePrep[i]  / NumEleLake
SHUD/src/Model/MD_rhs_core.cpp:316:             * per-lake qLakeEvap / qLakePrcp. */
SHUD/src/Model/MD_rhs_core.cpp:369:     * Must run BEFORE the lake clamp below (clamp reads qLakeEvap /
SHUD/src/Model/MD_rhs_core.cpp:370:     * qLakePrcp). Cannot live in PassValue_legacy because PassValue_legacy() is
SHUD/src/Model/MD_rhs_core.cpp:386:            qLakeEvap[i] = fixed_pairwise_sum_indexed(
SHUD/src/Model/MD_rhs_core.cpp:388:            qLakePrcp[i] = fixed_pairwise_sum_indexed(
SHUD/src/Model/MD_rhs_core.cpp:393:        qLakeEvap[i] = min(qLakeEvap[i], qLakePrcp[i] + yLakeStg[i]);
SHUD/src/Model/MD_rhs_core.cpp:394:        qLakeEvap[i] = max(0, qLakeEvap[i]);
SHUD/src/Model/MD_rhs_core.cpp:444: *      Lake: QLakeRivIn, QLakeSurf, QLakeSub).
SHUD/src/Model/MD_rhs_core.cpp:458: *   - S3b.4 qLakeEvap / qLakePrcp per-element->per-lake gather; that
SHUD/src/Model/MD_rhs_core.cpp:517:            QLakeRivIn[ilake] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:520:            QLakeRivIn[ilake] = fixed_leftfold_sum_indexed(
SHUD/src/Model/MD_rhs_core.cpp:528:            QLakeSurf[ilake] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:531:            QLakeSurf[ilake] = fixed_leftfold_sum_pair_indexed(
SHUD/src/Model/MD_rhs_core.cpp:538:            QLakeSub[ilake] = 0.;
SHUD/src/Model/MD_rhs_core.cpp:541:            QLakeSub[ilake] = fixed_leftfold_sum_pair_indexed(
SHUD/src/Model/MD_rhs_core.cpp:665:        DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i]  +
SHUD/src/Model/MD_rhs_core.cpp:666:                    (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i] ) / y2LakeArea[i] ;
```

---

## 去重 union 字段分类表

下表是两 grep 的去重字段 union，按 owner pragma 划分（owner pragma 行号引自
PR-C 实施后的 `MD_update.cpp`：element pragma = L90；river owner pragma = L109 + 串行 reset
L155-158；element 串行 reset = L161-162；lake owner pragma = L165）。"f_update zero-write site"
列指该字段在 `f_update` 内被零化/全覆盖的位置（用于 first-touch bitwise-neutrality 论证）；
"RHS 写者"列指 `MD_rhs_core.cpp` 内的最终计算写者（reduction / gather）。

### ELEMENT-owned 子集（PR-C 实施 scope）

| 字段 | 类型 | owner pragma | f_update zero-write site（再覆盖） | RHS 写者（MD_rhs_core.cpp） |
|---|---|---|---|---|
| `QeleSurf_flat`（`QeleSurfAt(i,j)`, flat3 j=0..2） | flat NumEle*3 hot flux | element L90 | element pragma L97（`= 0.`）| L65 zero + ElementFlux accum |
| `QeleSub_flat`（`QeleSubAt(i,j)`, flat3 j=0..2） | flat NumEle*3 hot flux | element L90 | element pragma L96（`= 0.`）| L64 zero + ElementFlux accum |
| `QeleSurfTot` | NumEle hot scratch | element L90 | element pragma L99（`= 0.`）| L577 `= Qe2r_Surf[i]` + L581 `+=` |
| `QeleSubTot` | NumEle hot scratch | element L90 | element pragma L98（`= 0.`）| L578 `= Qe2r_Sub[i]` + L582 `+=` |
| `Qe2r_Surf` | NumEle element→river flux | element 串行 reset L161 | 串行 reset L161（`= 0.`）| L497 `= -leftfold(...)`（全覆盖）|
| `Qe2r_Sub` | NumEle element→river flux | element 串行 reset L162 | 串行 reset L162（`= 0.`）| L498 `= -leftfold(...)`（全覆盖）|

**PR-C first-touch zero-write 字段集** = 上表 6 字段，严格 element-owned 子集，**不含**任何
river / lake 字段。每字段在 first-touch loop 写 `0.0` 后，于同一 `f_update` 调用内被再次
零化或全覆盖（见上表 "f_update zero-write site" 列），故 first-touch 对 serial numerical path
bitwise-neutral（per design R1）。

> 排除项（持久态，**禁止**入 first-touch loop，per task hard requirement）：
> `uYsf` / `uYus` / `uYgw` 在 element pragma body（L96-99 之后）由 `Y[...]` 赋**非零**值，
> 不是零化，若入 first-touch 会被自身后续 `= Y` 覆盖但语义上属持久态写者，明确排除以避免
> 误把状态写当 first-touch；`qEleExfil` / `qEleInfil` 虽零化但属 ET/infil 标量 scratch，
> 保守起见仅纳入上表 6 个明确的 hot flux/Tot/Qe2r 字段（与 task 候选集完全一致）。

### RIVER-owned 子集（PR-D scope，本 PR 不实施）

| 字段 | 类型 | owner pragma | f_update zero-write site | RHS 写者（MD_rhs_core.cpp） |
|---|---|---|---|---|
| `uYriv` | NumRiv river state | river L109（`shared(Y,t,uYriv)`）| L109 pragma body `= Y[iRIV]`（非零，持久态）| — |
| `Riv[i].qBC` / `Riv[i].yBC` | river BC scalar | river L109 | L109 pragma body | — |
| `QrivSurf` | NumRiv river flux | 串行 reset L156 | 串行 reset L156（`= 0.`）| L487 `= leftfold(...)` |
| `QrivSub` | NumRiv river flux | 串行 reset L157 | 串行 reset L157（`= 0.`）| L488 `= leftfold(...)` |
| `QrivUp` | NumRiv river flux | 串行 reset L158 | 串行 reset L158（`= 0.`）| L508 `= -leftfold(...)` |

> PR-D 建议（per spec scenario "river first-touch loop"）：将 L155-158 串行 `QrivSurf` /
> `QrivSub` / `QrivUp` 零化 loop 与 L161-162 串行 `Qe2r_Surf` / `Qe2r_Sub` 零化 loop 一并改
> `#pragma omp parallel for schedule(static)`，并把这 5 字段的 owner 责任记入本 doc。注意
> `Qe2r_Surf` / `Qe2r_Sub` 在本 PR-C 已归 ELEMENT 子集（element-owned 物理量，串行 reset 仅
> 位置上落在 river block 之后）；PR-D 改 L161-162 loop 时复用 ELEMENT 子集论证，不重复计字段。

### LAKE-owned 子集（PR-E scope，本 PR 不实施）

| 字段 | 类型 | owner pragma | f_update zero-write site | RHS 写者（MD_rhs_core.cpp） |
|---|---|---|---|---|
| `yLakeStg` | NumLake lake state | lake L165（`shared(Y,t)`）| L166 pragma body `= Y[iLAKE]`（非零，持久态）| — |
| `lake[i].yStage` | lake state scalar | lake L165 | L167 pragma body | — |
| `y2LakeArea` | NumLake lake geom | lake L165 | L169 pragma body `= lake[i].u_toparea`（非零）| — |
| `QLakeSub` | NumLake lake flux | lake L165 | L170 pragma body（`= 0.`）| L538/L541 `= leftfold_pair(...)` |
| `QLakeSurf` | NumLake lake flux | lake L165 | L171 pragma body（`= 0.`）| L528/L531 `= leftfold_pair(...)` |
| `qLakeEvap` | NumLake lake ET | lake L165 | L172 pragma body（`= 0.`）| L386 `= pairwise(...)` + L393-394 clamp |
| `qLakePrcp` | NumLake lake ET | lake L165 | L173 pragma body（`= 0.`）| L388 `= pairwise(...)` |
| `QLakeRivIn` | NumLake lake↔river flux | lake L165 | L174 pragma body（`= 0.`）| L517/L520 `= leftfold(...)` |
| `QLakeRivOut` | NumLake lake↔river flux | lake L165 | L175 pragma body（`= 0.`）| — (DY 消费 L666) |

---

## LAKE 子集非空 gate（PR-E 实施前置 verification gate，OQ1 决策落地）

spec scenario "element first-touch 字段集归档（PR-C）" 要求 LAKE 子集 SHALL 非空，含
`QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` **至少 4 stems** 命中行。

补充 grep 实测命中（lake owner-local zero-write site）：

| stem | MD_update.cpp lake owner 命中 | MD_rhs_core.cpp 命中 |
|---|---|---|
| `QLakeSub` | L170 | L136 / L538 / L541 |
| `QLakeSurf` | L171 | L137 / L528 / L531 |
| `QLakeRivIn` | L174 | L140 / L517 / L520 |
| `QLakeRivOut` | L175 | L141 |
| `qLakeEvap` | L172 | L138 / L386 / L393-394 |
| `qLakePrcp` | L173 | L139 / L388 |

**Gate 结果：PASS。** 4 个必需 stems（`QLakeSub` / `QLakeSurf` / `QLakeRivIn` /
`QLakeRivOut`）全部命中，且额外含 `qLakeEvap` / `qLakePrcp` 共 **6 stems** ≥ 4 门限。
PR-E 可凭此 doc 的 LAKE 子集直接实施 lake first-touch loop，无需另建 grep doc。

---

## PR-C 验收回执（self-check）

- §"字段集 grep 输出" 章存在，主 grep（93 行）+ 补充 grep（33 行）union 完整粘贴 ✓
- LAKE 子集非空，6 stems ≥ 4 门限，gate PASS ✓
- ELEMENT first-touch 字段集 = 6 字段，严格 element-owned，无 river/lake 字段 ✓
- element pragma 区新增 first-touch loop，`schedule(static)` + `default(none)` + 仅 `= 0.0`
  zero-write，无 reduction（实现见 `MD_update.cpp` L79-89，pragma count 3 → 4）✓
- bitwise-neutrality：keliya N=1 first-touch 前后 byte-identical（serial + shud_omp@N=1 双
  backend；详见 PR-C 验证回执 / `docs/p1d/` 后续 capstone）✓
</content>
</invoke>
