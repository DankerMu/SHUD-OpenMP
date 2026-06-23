## Conventions

### 站点计数定义

本 spec 中所述 "8 reduction 站点" 指 8 个**逻辑站点 (logical site)** = **10 line anchors**：

- 站点 1–7 各对应单一 line anchor (L278, L279, L374, L375, L382, L383, L392)；
- **站点 8 = "lake gathers 组" = 3 line anchors (L406 / L420 / L433)**：即 `QLakeRivIn` / `QLakeSurf` / `QLakeSub` 三个 lake 写目标合并为一个逻辑站点（理由：三者共用 `lake_bank_edge_by_lake` 或 `riv_in_by_lake` adjacency list，遍历结构与改造模式一致）。

后续凡 "8 站点"、"8 个 reduction 站点" 字样均按此定义；Scenario 内 "10 line anchors 落入 8 logical sites" 应作此理解。

### B0 serial loop 原始数组索引顺序

本 spec 中所述 "canonical order" / "稳定遍历顺序" 均指 **B0 serial loop 原始数组索引顺序** (per S3c.1 / master plan L1281) — 不是 element / segment / lake 的 id 升序。该顺序由 S4 既有 7 个 adjacency list (`seg_by_riv` / `seg_by_ele` / `upstream_by_down` / `riv_in_by_lake` / `ele_by_lake` / `lake_bank_edge_by_lake` / `edge_by_ele`) 锁定，构建时已保证内部 index 顺序等价于 B0 `for (i = 0; i < Num*; ++i)` 遍历顺序 (per s4-adjacency-topology spec L43-L45 / L64-L65)。**禁止**在 P1c 改造中按 id 排序、亦不构造任何并行 canonical-order 定义；所有 fixed-shape pairwise 配对结构 SHALL 复用 S4 既有 list。

---

## ADDED Requirements

### Requirement: 全 RHS reduction 站点 grep 清单完整覆盖

P1c 实施前 SHALL 产 `docs/p1c_reduction_sites.md` + `docs/p1c_reduction_sites_baseline.txt`（grep 输出的 frozen baseline, 入 git），列出 SHUD 仓库内**所有** RHS 路径的 reduction 站点：grep 锚定 (a) `reduction(+:` (b) `#pragma omp atomic` (c) anchored function-body accumulator `^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\[[^]]+\][[:space:]]*[+-]=` 排除注释/字符串 (d) 已知 owner-local gather 函数调用。每个命中站点 SHALL 标注：文件路径 + 行号 + 写目标变量 + 是否本 change 覆盖 + 若 N/A 说明理由 (例如 SPGMR / N_Vector 推 P9 per design D6；或 dead-code mirror per design D1)。

#### Scenario: grep 清单覆盖完整性 (anchored, comment/string excluded)

- **WHEN** `grep -rnE '^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*\[[^]]+\][[:space:]]*[+-]=' SHUD/src/Model/ SHUD/src/ModelData/ | grep -vE '^[^:]+:[0-9]+:[[:space:]]*(//|\*)' | grep -v _uncouple` 与 `docs/p1c_reduction_sites.md` 中"已覆盖" + "N/A 推迟" + "已 OMP-safe / 零改动" 三类站点之并集做差集
- **AND** 该 grep 输出与 `docs/p1c_reduction_sites_baseline.txt` (frozen baseline) line-for-line diff 为空
- **THEN** 差集为空 (所有 += 站点均显式分类，无遗漏，且基线无漂移)

#### Scenario: SPGMR / N_Vector / forcing / dead-code mirror 路径 N/A 说明

- **WHEN** 读 `docs/p1c_reduction_sites.md` 的 "N/A 推迟" / "dead-code mirror" 节
- **THEN** SPGMR Gram-Schmidt / N_Vector `N_VDotProd` / `MD_ET.cpp` / `TimeSeriesData.cpp` 等所有 N/A 站点 SHALL 各引用 design D6 carve-out 或 master plan §6 P1c.1 候选 (c) 推 P9 说明；MD_f.cpp `f_loop` L73-74 / `f_applyDY` 站点 SHALL 标 "dead-code mirror, OMP 路径不可达, per design D1 + §1.4 显式验证 grep 无 active caller"

---

### Requirement: MD_rhs_core.cpp 8 reduction 站点 (= 10 line anchors) fixed-shape pairwise 改造

`SHUD/src/Model/MD_rhs_core.cpp` 中以下 8 个 reduction 站点 (= 10 line anchors，per Conventions §"站点计数定义") SHALL 改造为 fixed-shape pairwise canonical reduction (即配对顺序与遍历顺序严格遵循 **B0 serial loop 原始数组索引顺序**, per Conventions §"B0 serial loop 原始数组索引顺序"；**全部复用** S4 既有 adjacency list，与 `NUM_OPENMP` 完全解耦)：

| 逻辑站点 | 行号 (line anchor) | 写目标 | 复用 S4 adjacency list |
|---|---|---|---|
| 1 | L278 | `qLakeEvap[ilake] += qEleEvapo_lake[i]` | (隐式) B0 `for (i = 0; i < NumEle; i++) if Ele[i].iLake > 0` 顺序 |
| 2 | L279 | `qLakePrcp[ilake] += qElePrep_lake[i]` | 同上 |
| 3 | L374 | `QrivSurf[ir] += QsegSurf[iseg]` | S4.1 `seg_by_riv` |
| 4 | L375 | `QrivSub[ir] += QsegSub[iseg]` | S4.1 `seg_by_riv` |
| 5 | L382 | `Qe2r_Surf[ie] += -QsegSurf[iseg]` | S4.2 `seg_by_ele` |
| 6 | L383 | `Qe2r_Sub[ie] += -QsegSub[iseg]` | S4.2 `seg_by_ele` |
| 7 | L392 | `QrivUp[ir] += -QrivDown[up]` | S4.3 `upstream_by_down` |
| 8 (lake gathers 组) | L406 / L420 / L433 | `QLakeRivIn` / `QLakeSurf` / `QLakeSub` lake gathers | S4.4 `riv_in_by_lake` / S4.6 `lake_bank_edge_by_lake` |

注：L473-474 `QeleSurfTot[i] += QeleSurfAt(i, j)` 与 `QeleSubTot[i] += QeleSubAt(i, j)` 位于 `rhs_apply()` 内 `for i in 0..NumEle: for j in 0..3` 的 inner-loop per-i 顺序累加 (i 已 owner-local，j ∈ {0,1,2} 内层固定)，**不受 OMP 影响，不在本 change 改造范围**；SHALL 在 `docs/p1c_reduction_sites.md` 中显式归入 "已 OMP-safe，零改动" 类。

**禁止**：

- 不引入 `#pragma omp reduction(+:sum)` (master plan §8.1 strict 禁止)
- 不引入 `#pragma omp atomic` (同上)
- 不改 schedule 类型 (继续 `schedule(static)`)
- 不动 fork-join 结构 (P7 final-fusion 范围)
- 不引入新编译宏 (design D8 — strict 阶段固定行为)
- 不按 element / segment / lake **id 升序**构造 canonical order；**不**构造任何并行 canonical-order 定义；必须复用 S4 既有 7 个 adjacency list（per Conventions §"B0 serial loop 原始数组索引顺序"）

#### Scenario: 10 line anchors 落入 8 logical sites (覆盖完整性)

- **WHEN** `grep -nE '[+-]=' SHUD/src/Model/MD_rhs_core.cpp | head -30`
- **THEN** 上述 10 line anchors (L278 / L279 / L374 / L375 / L382 / L383 / L392 / L406 / L420 / L433) 各对应一条命中；按 Conventions §"站点计数定义" 分组为 8 logical sites — 改造后行号可能漂移，但函数名 / 写目标变量 SHALL 与 `docs/p1c_reduction_sites.md` 中清单逐项对齐

#### Scenario: S4 adjacency list 复用 (无并行 canonical-order)

- **WHEN** `grep -n 'seg_by_riv\|seg_by_ele\|upstream_by_down\|riv_in_by_lake\|lake_bank_edge_by_lake' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** 改造后 5 个 S4 list (站点 3–8) 均被引用；改造前同一函数体内无任何 `std::sort` / `std::stable_sort` / `qsort` 出现 (确认未构造新 canonical-order)
- **AND** `docs/p1c_reduction_sites.md` 每个站点行 SHALL 显式标注其复用的 S4 list 名称 (与上表 col 4 一致)

#### Scenario: 编译期固定 (不引入新宏)

- **WHEN** `grep -rE "SHUD_USE_DETERMINISTIC_REDUCTION|SHUD_DET_REDUCT|SHUD_PAIRWISE" SHUD/`
- **THEN** 返回 0 hit (本 change 不引入任何形式的 deterministic-reduction 开关宏)

#### Scenario: schedule 类型 + fork-join 结构保持不变

- **WHEN** `grep -nE "schedule\(" SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** 所有 schedule 子句仍为 `static`（无 `dynamic` / `guided`）

#### Scenario: 禁 OMP atomic

- **WHEN** `grep -n '#pragma omp atomic' SHUD/src/`
- **THEN** 返回 0 hit

---

### Requirement: P1c.0 实验诊断 (前置 SHALL)

§2.x 站点改造**之前** SHALL 完成 P1c.0 实验诊断 (per design D9)：instrument PR-K2 N=1 vs N=4/8 的 RHS 调用栈，定位 `DY` 首次出现 byte-level 发散的 (step_index, array_name, ele/seg/riv index)；显式判定该位置是否在 8 站点 reduction 路径上；若不在，则**先**扩展 spec R2 表 in-scope 表 (或显式记录为 carve-out 推 P9, per design D6) **再**推进 §2.x。

#### Scenario: 诊断报告归档

- **WHEN** §2.x 改造任何提交进入 PR
- **THEN** `docs/p1c_a3a_root_cause.md` §"诊断结果" 章 SHALL 已存在并包含 4 项必备字段 (per p1c-capstone §"p1c_a3a_root_cause 吸收 F-K2-2 + 量化数据" Scenario)

#### Scenario: in-scope 表与诊断结果一致

- **WHEN** 诊断结果显示 DY 首发散点不在原 8 站点 reduction 路径
- **THEN** spec R2 表 SHALL 在 §2.x 任何代码改造前完成扩展 (新增对应站点行或显式 carve-out 推 P9 行)

---

### Requirement: Kahan 补偿求和兜底 (条件触发, server PR-K2 首跑 FAIL)

如 fixed-shape pairwise 改造完成后 **server PR-K2 首跑** (per design D4) 仍出现 N ∈ {1,2,4,8} bitwise FAIL（即 fixed-shape 不足以闭合 ULP 误差），P1c 实施 SHALL 在每个 owner accumulation 处叠加 Kahan / Neumaier 补偿求和（参 SUNDIALS user guide §3.1.1 deterministic dot-product 实现）。

若 server PR-K2 首跑即满足 A3a + nst 跨 N 全等 + 反向兼容三条件，Kahan 补偿 SHALL **不引入** (避免无谓的算术开销 + 代码复杂度上升)。

**禁止**：以 Mac local §2.5 16-cell snapshot 结果作为 Kahan 触发依据 (Mac snapshot 已知 pass-while-server-fails, per design D7)。

#### Scenario: 仅在必要时引入 Kahan

- **WHEN** server PR-K2 首跑 N ∈ {1,2,4,8} 全部 A3a PASS bitwise + nst 跨 N 全等 + 反向兼容
- **THEN** Kahan 补偿不引入；MD_rhs_core.cpp grep "kahan" / "neumaier" / "compensation" SHALL = 0 hit

#### Scenario: Kahan 引入位置正确 (条件触发后)

- **WHEN** server PR-K2 首跑 FAIL，第二轮叠加 Kahan
- **THEN** Kahan 补偿 SHALL 仅在上述 8 个 reduction 站点的 owner accumulation 处叠加；不在其他无关位置引入；每处叠加 SHALL 在 `docs/p1c_a3a_root_cause.md` 中说明触发理由（ULP-level 残留量化数据）

#### Scenario: Kahan 引入后必须 PR-K2 重新通过

- **WHEN** 触发条件满足并按 §4.7 注入 Kahan
- **THEN** PR-K2 二跑 (server cn0X, heihe + heihe_x4 各 4 N) SHALL 重新执行；二跑同样需满足 A3a bitwise + nst 跨 N 全等 + 反向兼容三条件，方可放行 P1c capstone；二跑仍 FAIL → P1c capstone NOT released

---

### Requirement: A3a 强制 — N ∈ {1, 2, 4, 8} bitwise DY

P1c 实施完成后 SHALL 满足 A3a 强制门：

| 项 | 标准 |
|---|---|
| 单次 RHS probe | `DY` 在 `NUM_OPENMP ∈ {1, 2, 4, 8}` 下 byte-identical (RHS snapshot probe per case) |
| 完整 CVODE run | 同 binary 在 `NUM_OPENMP ∈ {1, 2, 4, 8}` 下输出 canonical SHA byte-identical (heihe + heihe_x4) |
| 反向兼容 | `NUM_OPENMP=1` 路径与 B1b / B1-tag canonical SHA 仍 bitwise (P1 forced gate 不退化) |

#### Scenario: 单次 RHS probe 跨 N bitwise

- **WHEN** `tools/rhs_snapshot/rhs_snapshot <case> <NUM_OPENMP>` 各跑 N ∈ {1, 2, 4, 8}
- **AND** `compare_snapshot --quiet snapshot_N1.bin snapshot_N2.bin snapshot_N4.bin snapshot_N8.bin`
- **THEN** 全部 exit 0 (byte-equal across all 4 N for each case)

#### Scenario: 完整 CVODE run 跨 N canonical SHA bitwise (reproducible shell)

- **WHEN** `for N in 1 2 4 8; do OMP_NUM_THREADS=$N tools/archive_b0_output.sh <case> 1 --out .p1c-runs/N${N}_<case>; done`
- **AND** `sha256sum .p1c-runs/N{1,2,4,8}_<case>/output/<case>.out/<case>.rivqdown.dat | awk '{print $1}' | sort -u | wc -l`
- **THEN** 输出 = `1` (跨 4 N 字面相等)

#### Scenario: Mac N=1 反向兼容: 4 Mac case 与 P1-update-omp-tag Mac canonical SHA bitwise

- **WHEN** Mac local P1c 实施完成后 N=1 跑 keliya / xinanjiang_upstream / qinyijiang / qhh 4 case
- **THEN** `sha256sum output/<case>.out/<case>.rivqdown.dat` SHALL 与 `P1-update-omp-tag` 在同 case 同 N=1 下的 Mac canonical SHA 字面相等 (per `docs/p1_summary.md` §"§4 Mac canonical SHA" 表；qhh 注意 PR #188 rebase tag-chain caveat per p1-state-update-parallel L130)

#### Scenario: Server N=1 反向兼容: heihe + heihe_x4 与 P1-update-omp-tag server canonical SHA bitwise

- **WHEN** Server cn0X P1c 实施完成后 N=1 跑 heihe + heihe_x4
- **THEN** `sha256sum output/<case>.out/<case>.rivqdown.dat` SHALL 与 `P1-update-omp-tag` 在同 case 同 N=1 下的 server canonical SHA 字面相等 (per `docs/p1_fullrun_bitwise.md` 表)

---

### Requirement: CVODE `nst` 跨 N 完全相等 (heihe Δ=0 强制; heihe_x4 Δ=0 with ladder)

A3a 在单次 RHS 层之上，完整 CVODE 积分尺度 SHALL 进一步满足跨 N `nst` 完全相等 (修复 P1 实测的 heihe nst 6773/6773/6585/6684 漂移)：

| Case | 目标 (P1c 后) | 历史 (P1 实测) | Ladder |
|---|---|---|---|
| heihe | `nst` 跨 N ∈ {1, 2, 4, 8} **完全相等 (Δ=0)** | 6773 / 6773 / **6585** / **6684** (漂移) | 无 ladder (硬性 Δ=0) |
| heihe_x4 | `nst` 跨 N ∈ {1, 2, 4, 8} **完全相等 (Δ=0)** | 6571 / 6571 / **6570** / **6572** (微漂) | 若残留 \|Δ\|≤2 → 触发 SPGMR-noise 判定 (per design D6) |

附加要求：CVODE 15-key stats (nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS) SHALL 在 N ∈ {1, 2, 4, 8} 下逐 key 跨 N 完全相等 (heihe + heihe_x4 各 4 N)。

#### Scenario: CVODE 15-key 跨 N 字面相等 (reproducible shell)

- **WHEN** `for N in 1 2 4 8; do OMP_NUM_THREADS=$N tools/archive_b0_output.sh <case> 1 --out .p1c-runs/N${N}_<case>; done`
- **AND** `tools/cvode_stats_diff/cvode_stats_diff.sh .p1c-runs/N1_<case>/output/<case>.out/cvode_stats.txt .p1c-runs/N2_<case>/output/<case>.out/cvode_stats.txt .p1c-runs/N4_<case>/output/<case>.out/cvode_stats.txt .p1c-runs/N8_<case>/output/<case>.out/cvode_stats.txt`
- **THEN** 全部 exit 0 (15 keys × 4 N pairs 全部 byte-equal for each case)

#### Scenario: P1 实测 nst 漂移消除 (heihe Δ=0 强制)

- **WHEN** P1c 实施完成后 server 跑 heihe 各 4 N
- **AND** 读 `output/heihe.out/cvode_stats.txt`
- **THEN** heihe 各 N 的 `nst` 值字面相等 (即 P1 实测的 6585 / 6684 漂移已消除)

#### Scenario: heihe_x4 SPGMR-noise ladder (条件触发)

- **WHEN** P1c 实施完成后 server 跑 heihe_x4 4 N，残留 \|Δ_nst\| ∈ {1, 2}
- **AND** 以 `CVODE_RTOL/10` (e.g. 1e-7 → 1e-8) 复跑 heihe_x4 4 N
- **THEN** 残留 Δ 消失或同比例缩小 → 标记 "SPGMR-noise-attributed, deferred to P9" 入 `docs/p1c_a3a_root_cause.md` (per design D6)；残留 Δ 未缩小 → 视为 P1c FAIL → 触发 §4.7 conditional Kahan + PR-K2 二跑

---

### Requirement: 部分通过状态视为 FAIL

P1c success SHALL 当且仅当**同时**满足 A3a bitwise + nst 跨 N 全等 (heihe Δ=0 + heihe_x4 Δ=0 with optional SPGMR-noise ladder) + 反向兼容三条件 (per design D3)。任一不满足即 SHALL 视为 P1c FAIL，capstone MUST NOT 放行。

#### Scenario: 三条件任一 FAIL 即 P1c FAIL

- **WHEN** server PR-K2 任一轮 (首跑或 Kahan 二跑) 出现 {A3a bitwise, nst 跨 N 全等 (含 heihe_x4 SPGMR-ladder 判定), 反向兼容} 中的任一 FAIL
- **THEN** P1c capstone NOT released；PR-K2 evidence 须 re-collect；若 fixed-shape 已尝试且未叠 Kahan → 触发 §4.7；若已叠 Kahan 仍 FAIL → 归因到 design D9 决策分支 2/3 (in-scope 表扩展 / 退回 master plan §6 P1c.0 重新选路)

---

### Requirement: serial-baseline CI 强制 P1 N=1 不退化

P1c 实施期间所有 PR SHALL 通过 `.github/workflows/serial-baseline.yml` (keliya N=1 vs B0 bitwise)，确保任何 fixed-shape 改造错位**立即**被 CI 阻断而非延迟到 PR-K2 复跑时才发现。

#### Scenario: serial-baseline.yml 全程绿

- **WHEN** 任一 P1c 实施 PR 推送 commit
- **THEN** `.github/workflows/serial-baseline.yml` 三 build (DEBUG / RELEASE / PROFILE) + keliya 跑 + bitwise vs B0 全部 PASS

#### Scenario: §2.x 每个 site-class 改造后增量 CI

- **WHEN** §2.1 / §2.2 / §2.3 / §2.4 任一 site-class 改造完成
- **THEN** §2.1.v / §2.2.v / §2.3.v / §2.4.v 验证步骤 SHALL 立即执行 (keliya N=1 vs B0 bitwise，push branch 触发 `serial-baseline.yml`)；不可等到 §2.5 末端 verification 才 batch 验证 (per design R1 缓解)

---

### Requirement: Mac local 16-cell 4-case 辅助预筛 (SHOULD)

Mac local SHOULD 跑 keliya / xinanjiang_upstream / qinyijiang / qhh 各 N ∈ {1, 2, 4, 8} = 16 cell RHS snapshot probe + 4 N canonical SHA bitwise 验证作为早期 sanity check (per design D7)。**Mac 结果不阻 server 提交**，亦不作为 §4.7 conditional Kahan 触发依据；server cn0X PR-K2 复跑为唯一 SHALL 级 success gate。

#### Scenario: Mac local 4-case scan (SHOULD)

- **WHEN** §2.x 全部完成后准备 server 提交前
- **THEN** Mac local SHOULD 跑 4-case × 4-N = 16 cell；任何 PASS/FAIL 输出仅入 `docs/p1c_perf_baseline.md` §"Mac 辅助预筛" 节作为参考；不阻 server Slurm 提交

#### Scenario: Mac 结果不触发 Kahan

- **WHEN** Mac local 16-cell 4-case scan 出现任一 FAIL
- **THEN** 不触发 §4.7 conditional Kahan injection (per design D7)；仅记录 `docs/p1c_perf_baseline.md`；继续推进 server PR-K2 首跑作为唯一 Kahan 触发判据 (per design D4)
