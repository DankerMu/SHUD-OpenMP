# P1c RHS Reduction Sites — In-scope / N/A / OMP-safe Classification

OpenSpec change `p1c-deterministic-reduction` 前置 audit (PR-A #244 §1.2 task).

## 来源 (baseline grep)

**分支模型 (P1c 阶段)**：`baseline/P1c` 是 P1c 阶段唯一活动集成线 (从 main HEAD `53d5294`
分出, 2026-06-22)。PR-A..PR-M 全部 base 该分支 (PR base != main, `Closes #N` 自动关闭
keyword 失效, orchestrator 手动 `gh issue close <N>`); PR-M 合并后 lock + sync main +
annotate `P1c-tag` (D11 immutable), per master plan §6 P1c.4 baseline lock 与 tag +
P1c.6 后续移交.

**SHUD submodule 工作流 (P1c §2.x SHUD code change)**：PR-B 起涉及 SHUD source 改造的 PR
SHALL 走 CLAUDE.md "SHUD submodule 工作流 (强制)" — `cd SHUD && git commit && git push origin
openmp-baseline` → `cd .. && git add SHUD && git commit` (pointer bump) → 外层 PR。禁 push
master / 禁 fork / 禁改 `.gitmodules`。PR-A 本身不动 SHUD source，pointer 与 baseline/P1c 一致
`07c677f` (verified)。

`docs/p1c/p1c_reduction_sites_baseline.txt` 为 frozen baseline，命令逐字：

```bash
grep -rnE '^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\[[^]]+\][[:space:]]*[+-]=' \
  SHUD/src/Model/ SHUD/src/ModelData/ \
  | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*)' \
  | grep -v _uncouple \
  > docs/p1c/p1c_reduction_sites_baseline.txt
```

附加 grep (覆盖性自检，不入 baseline.txt)：

```bash
grep -rn 'reduction(+:' SHUD/src/Model/ SHUD/src/ModelData/    # => 0 hit
grep -rn '#pragma omp atomic' SHUD/src/Model/ SHUD/src/ModelData/  # => 0 hit
```

SHUD pin = `07c677f`（PR-A #244 起始 + PR-K2 #223 三 pragma 栈，与 `baseline/P1c` SHUD pointer 一致）。

---

## 类 1 — 已覆盖 (in-scope, 本 change SHALL 改造)

8 个逻辑站点 = 10 line anchors，对应 `SHUD/src/Model/MD_rhs_core.cpp` 中
`Model_Data::rhs_flux()` (`qLakeEvap`/`qLakePrcp` 前置 pre-clamp gather, L278/L279)
和 `Model_Data::rhs_deterministic_gather()` (剩余 8 anchors)。

| # | file:line | 写目标 (acc_target) | 输入 (src) | 复用 S4 adjacency list |
|---|---|---|---|---|
| 1 | `SHUD/src/Model/MD_rhs_core.cpp:278` | `qLakeEvap[ilake]` | `qEleEvapo_lake[i]` | (隐式) B0 `for (i=0;i<NumEle;i++) if Ele[i].iLake>0` 顺序 (不 sort 重排) |
| 2 | `SHUD/src/Model/MD_rhs_core.cpp:279` | `qLakePrcp[ilake]` | `qElePrep_lake[i]` | 同上 |
| 3 | `SHUD/src/Model/MD_rhs_core.cpp:374` | `QrivSurf[ir]` | `QsegSurf[iseg]` | S4.1 `seg_by_riv[ir]` |
| 4 | `SHUD/src/Model/MD_rhs_core.cpp:375` | `QrivSub[ir]` | `QsegSub[iseg]` | S4.1 `seg_by_riv[ir]` |
| 5 | `SHUD/src/Model/MD_rhs_core.cpp:382` | `Qe2r_Surf[ie]` | `-QsegSurf[iseg]` | S4.2 `seg_by_ele[ie]` |
| 6 | `SHUD/src/Model/MD_rhs_core.cpp:383` | `Qe2r_Sub[ie]` | `-QsegSub[iseg]` | S4.2 `seg_by_ele[ie]` |
| 7 | `SHUD/src/Model/MD_rhs_core.cpp:392` | `QrivUp[ir]` | `-QrivDown[up]` | S4.3 `upstream_by_down[ir]` |
| 8a | `SHUD/src/Model/MD_rhs_core.cpp:406` | `QLakeRivIn[ilake]` | `QrivDown[iriv]` | S4.4 `riv_in_by_lake[ilake]` |
| 8b | `SHUD/src/Model/MD_rhs_core.cpp:420` | `QLakeSurf[ilake]` | `QeleSurf_lake[ie*3+j]` | S4.6 `lake_bank_edge_by_lake[ilake]` |
| 8c | `SHUD/src/Model/MD_rhs_core.cpp:433` | `QLakeSub[ilake]` | `QeleSub_lake[ie*3+j]` | S4.6 `lake_bank_edge_by_lake[ilake]` |

### Line-number 等价表 (B1b SHUD `0b3998d` ↔ current SHUD `07c677f`)

`docs/p1c/p1c_b1b_serial_order_dump.txt` 采用 B1b SHUD pin `0b3998d` 行号家族 (`site_tag` 后缀)，
sites.md 类1 表 / spec L39-L50 / `docs/p1c/p1c_a3a_root_cause.md` 使用 PR-K2 #223 SHUD pin
`07c677f` 行号家族。下游 §2.x PR-B/C/D/E 消费 dump 时按此表 cross-walk。

| 站点 | 写目标 | dump-family site_tag suffix (*) | class-1-family (`07c677f`) |
|---|---|---|---|
| 1 | `qLakeEvap[ilake]` | L278 (`qLakeEvap_L278`) (*) | L278 |
| 2 | `qLakePrcp[ilake]` | L279 (`qLakePrcp_L279`) (*) | L279 |
| 3 | `QrivSurf[ir]` | L311 (`QrivSurf_L311`) | L374 |
| 4 | `QrivSub[ir]` | L312 (`QrivSub_L312`) | L375 |
| 5 | `Qe2r_Surf[ie]` | L319 (`Qe2r_Surf_L319`) | L382 |
| 6 | `Qe2r_Sub[ie]` | L320 (`Qe2r_Sub_L320`) | L383 |
| 7 | `QrivUp[ir]` | L329 (`QrivUp_L329`) | L392 |
| 8a | `QLakeRivIn[ilake]` | L343 (`QLakeRivIn_L343`) | L406 |
| 8b | `QLakeSurf[ilake]` | L357 (`QLakeSurf_L357`) | L420 |
| 8c | `QLakeSub[ilake]` | L370 (`QLakeSub_L370`) | L433 |

(*) Rows 1-2 site_tag suffix uses current SHUD `07c677f` line number (L278/L279)
because B1b SHUD `0b3998d` source has the same statements at L222/L223 — the
instrumented code at dump producer time used the post-fix line number for the
lake aggregation rows; rows 3-10 site_tag suffix matches the B1b source line at
`0b3998d`. The column header "dump-family site_tag suffix" therefore reflects
the suffix literal as it appears in `docs/p1c/p1c_b1b_serial_order_dump.txt`,
not necessarily the B1b source line number.

**Cross-walk 验证**：`grep -c 'site=QrivSurf_L311' docs/p1c/p1c_b1b_serial_order_dump.txt` 期望 ≥ 1
(dump-family hit); `grep -nE '\| 3 \|' docs/p1c/p1c_reduction_sites.md` 期望命中本表 L374 行
(class-1-family hit)。

合并 8a/8b/8c 为单一 logical site "lake gathers 组" — 三者共用 `lake_bank_edge_by_lake`
或 `riv_in_by_lake` adjacency 结构 + 同一 iteration pattern (per spec L8-L10
Conventions)，改造路径一致。

B1b serial loop 在以上 anchors 的 (iter, acc_target, src_idx) 输入顺序参照
`docs/p1c/p1c_b1b_serial_order_dump.txt` (§1.3 产物，SHUD@0b3998d N=1 dump).
§2.x adjacency-list 消费 SHALL cross-check 该 dump。

---

## 类 2 — N/A 推迟 (out-of-scope, 显式 carve-out)

| file:line | 写目标 | 类别 | 原因 + 引用 |
|---|---|---|---|
| `SHUD/src/ModelData/MD_f.cpp:73` | `qLakeEvap[ilake]` | dead-code mirror | dead-code mirror of L278; `f_loop()` 在 SHUD 仓库内**无 active caller** (§1.4 grep 验证: `grep -rnE '(->|::)[[:space:]]*f_loop\(' SHUD/src/` 仅命中 `MD_f.cpp:15` 定义本身)，OMP 路径不可达，per design D1 + §1.4 显式验证。`f.cpp` callers (L78/L91/L104/L117/L130) 调用的是 `f_loop1`/`f_loop2`/.../`f_loop5` 而非 `f_loop()`。(regex 含 `[[:space:]]*` 兼容 `MD_f.cpp:15` 字面 `void Model_Data:: f_loop` 之间空格；sibling `f_applyDY` 无空格也命中。) |
| `SHUD/src/ModelData/MD_f.cpp:74` | `qLakePrcp[ilake]` | dead-code mirror | 同上 (L279 mirror) |
| `SHUD/src/ModelData/MD_f.cpp:147` | `DY[igw] += hot.QBC[i] / area` | dead-code mirror | `f_applyDY(double*, double)` 在 SHUD 仓库内**无 active caller** (§1.4 grep 验证: `grep -rnE '(->|::)[[:space:]]*f_applyDY\(' SHUD/src/` 仅命中 `MD_f.cpp:111` 定义本身)，`f.cpp` callers 调用的是 `f_applyDYi`/`f_applyDY_gw`/`f_applyDY_surf`/`f_applyDY_unsat`/`f_applyDY_river` 而非 `f_applyDY()`。OMP 路径不可达，per design D1。 |
| `SHUD/src/ModelData/MD_f.cpp:152` | `DY[isf] += hot.QSS[i] / area` | dead-code mirror | 同上 |
| `SHUD/src/ModelData/MD_f.cpp:154` | `DY[igw] += hot.QSS[i] / area` | dead-code mirror | 同上 |
| `SHUD/src/ModelData/MD_f.cpp:249` | `DY[iGW] += hot.QBC[i] / hot.area[i]` | applyBCSS 内部 | `applyBCSS(double*, int i)` 仅在 splitter-RHS 路径 (`f_applyDY_gw` 等) 被调用，per-i scalar accumulation，**无 cross-thread reduction** (每个 element owner 单独负责自身 `DY[iGW]`)。non-OMP-driven write，per design D6 carve-out (SPGMR/N_Vector 同类) 推 P9 deterministic N_Vector. |
| `SHUD/src/ModelData/MD_f.cpp:253` | `DY[iSF] += hot.QSS[i] / hot.area[i]` | applyBCSS 内部 | 同上 |
| `SHUD/src/ModelData/MD_f.cpp:255` | `DY[iGW] += hot.QSS[i] / hot.area[i]` | 同上 | 同上 |
| `SHUD/src/Model/MD_rhs_core.cpp:495` | `DY[igw] += Ele[i].QBC / area` | rhs_apply per-i BC | per-i scalar BC injection in element-owner-local outer loop; **无 cross-thread reduction** (每个 i 独占 `DY[igw]`，不存在多 thread 写同一 `DY[k]`)。已 OMP-safe 但归类于 "已 OMP-safe / 零改动"，见类 3。 |
| `SHUD/src/Model/MD_rhs_core.cpp:500` | `DY[isf] += Ele[i].QSS / area` | rhs_apply per-i SS | 同上 |
| `SHUD/src/Model/MD_rhs_core.cpp:502` | `DY[igw] += Ele[i].QSS / area` | 同上 | 同上 |

**SPGMR Gram-Schmidt orthogonalization** (SUNDIALS `sundials_spgmr.c::SUNLinSolSolve_SPGMR`) — 推 P9 deterministic N_Vector，per design D6 + master plan §6 P1c.1 候选 (c).

**`N_Vector` 内部 `N_VDotProd` / `N_VWrmsNorm`** (`SHUD_USE_OPENMP_NVECTOR=OFF` 编译下走 `N_VNew_Serial` 顺序确定) — 推 P9，per design D6 + master plan §6 P1c.1 候选 (c).

**`SHUD/src/Model/MD_ET.cpp` / `SHUD/src/Equations/...` / `SHUD/src/ModelData/TimeSeriesData.cpp`** — `grep -rnE '^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\[[^]]+\][[:space:]]*[+-]='` 对这两个子树命中 0 行 (验证下方 §"全量自检" 节)，无 N/A 行待登记。

---

## 类 3 — 已 OMP-safe / 零改动

| file:line | 写目标 | 输入 | OMP-safe 理由 |
|---|---|---|---|
| `SHUD/src/Model/MD_rhs_core.cpp:473` | `QeleSurfTot[i]` | `QeleSurfAt(i, j)` | `for i in 0..NumEle: for j in 0..3` inner-loop per-i 顺序累加 (i owner-local + j ∈ {0,1,2} 内层固定)，**不受 OMP 影响**；每个 i 独占 `QeleSurfTot[i]`，per spec L52 注释 |
| `SHUD/src/Model/MD_rhs_core.cpp:474` | `QeleSubTot[i]` | `QeleSubAt(i, j)` | 同上 |
| `SHUD/src/ModelData/MD_f.cpp:125` | `QeleSurfTot[i]` | `QeleSurfAt(i, j)` | dead-code mirror of L473 (§1.4 dead-code 证明) |
| `SHUD/src/ModelData/MD_f.cpp:126` | `QeleSubTot[i]` | `QeleSubAt(i, j)` | 同上 |
| `SHUD/src/Model/MD_rhs_core.cpp:495` | `DY[igw] += Ele[i].QBC / area` | per-i scalar | per-element-owner; 见类 2 说明 (重复登记便于 OMP-safe 类的完整性) |
| `SHUD/src/Model/MD_rhs_core.cpp:500` | `DY[isf] += Ele[i].QSS / area` | per-i scalar | 同上 |
| `SHUD/src/Model/MD_rhs_core.cpp:502` | `DY[igw] += Ele[i].QSS / area` | per-i scalar | 同上 |

注：L473-474 与 MD_f.cpp L125-126 是 dead-code mirror 对照；前者是 active path
(`rhs_apply()` 内)，后者是 dead-code 镜像 (`f_applyDY()` 内，§1.4 已证无
active caller)。两者均不构成 reduction 漂移源。

---

## 全量自检 (Acceptance Criteria 必备 diff 为空)

baseline.txt 全部 25 行 union vs 上述三类 union 差集为空。命令 (sed/awk
表格解析，提取每行中每一个匹配 `` `SHUD/.../*.cpp:N` `` 的 token)：

```bash
# 1) 从 reduction_sites.md 抽取所有 file:line tokens (覆盖类 1/2/3 三表)
awk 'BEGIN { p=0 }
     /^## 类 [123]/ { p=1; next }
     /^## / { p=0 }
     p && /`SHUD\// {
         while (match($0, /`SHUD\/[^`]+`/)) {
             site=substr($0, RSTART+1, RLENGTH-2);
             if (site ~ /\.cpp:[0-9]+$/) print site;
             $0=substr($0, RSTART+RLENGTH);
         }
     }' docs/p1c/p1c_reduction_sites.md | sort -u > /tmp/p1c_union.txt

# 2) 从 baseline.txt 抽取 file:line tokens
sort -u docs/p1c/p1c_reduction_sites_baseline.txt \
  | awk -F: '{print $1":"$2}' > /tmp/p1c_baseline_filelines.txt

# 3) diff (期望空白 + exit 0)
diff /tmp/p1c_baseline_filelines.txt /tmp/p1c_union.txt
echo "exit=$?"
```

## 实际 verification (PR-A 完成时记录)

```text
$ wc -l docs/p1c/p1c_reduction_sites_baseline.txt
      25 docs/p1c/p1c_reduction_sites_baseline.txt

$ awk '...' docs/p1c/p1c_reduction_sites.md | sort -u | wc -l
      25

$ diff /tmp/p1c_baseline_filelines.txt /tmp/p1c_union.txt
(空白)

$ echo "exit=$?"
0
```

⇒ baseline 25 lines = 三类 union 25 lines，差集为空，**全 RHS reduction
站点覆盖完整性 PASS** (per spec §"全 RHS reduction 站点 grep 清单完整覆盖"
Scenario L26 "差集为空")。

---

## 附 — `reduction(+:)` / `#pragma omp atomic` 双 grep

`grep -rn 'reduction(+:' SHUD/src/Model/ SHUD/src/ModelData/` ⇒ **0 hit** (不存在 OpenMP reduction clause).

`grep -rn '#pragma omp atomic' SHUD/src/Model/ SHUD/src/ModelData/` ⇒ **0 hit** (不存在 atomic 写).

⇒ 当前 SHUD 仓库内**无**任何 OpenMP 内置 reduction 路径；所有 reduction 都是手写 owner-local `+=` 形式，由 §1.1 anchored grep 完整覆盖。
