# B1a Baseline 完成

> B1a = master plan §3 定义的"重构等价的单线程结果"，来自 **S0–S4 完成**。**stage = 工作阶段（S0/S1/S2/S3/S4）**；**baseline = 工作产物（B0/B1a）**；B1a 不是单一 stage 的产物，而是 **S0–S4 四个 stage 全部完成**之后才能签字的检查点。
>
> S0–S4 全部完成（PR-1 #144 through PR-11 #155 已 merged 进 `baseline/B1a`）。PR-12 #156 = B1a capstone：openspec archive + 6-case bitwise + 7 grep gate enforce + 文档收尾。

## 旧版 `s1_summary.md` 错在哪（历史复盘，已修正）

2026-06-18 当时的 `docs/s1_summary.md` 标题写 "S1（B1a refactor — 唯一 RHS core）完成于 2026-06-18。`B1a-tag` 已打。可以进入 S2"——**直接把 S1 等同于 B1a**。这违反 master plan §5：

| master plan B1a 契约（§3 + §5） | 实际 S1 完成时做了 | 缺什么（已在 PR-1..PR-11 补齐） |
|---|---|---|
| S0 锁 B0 baseline | 已完成 #3–#18，B0-tag = `884cfb13` / SHUD `78c37a1` | — |
| S1 抽取 serial RHS core | 已完成 #44 (S1a) + #45 (S1b) + #46 (S1c) + #47 (S1d.1) + #48 #49 #50 (S1d.2–S1d.5) | — |
| S2 语义对齐 + 合并 `_omp` 路径（**删 `f_update_omp / f_loop_omp / f_applyDY_omp` 三个独立函数**） | 原 0 commit | S2.1–S2.17 共 17 个子项已在 PR-1 #144 through PR-8 #152 落地 |
| S3 拆 flux compute + deterministic gather（删除 `PassValue()` 共享 `+=`） | 原 0 commit | S3a 4 项死代码 + S3b 4 项共享写拆分 + S3c 3 项 gather 重构 已在 PR-9 #153 + PR-11 #155 落地 |
| S4 固定拓扑顺序 + owner 映射（adjacency list） | 原 0 commit | S4.1–S4.7 共 7 个 adjacency list + topology manifest + `id == index+1` assert 已在 PR-10 #154 落地 |

旧 `status_matrix.md` L20 "B1a 行 PASS" 是**过早签证**——只验证了 S1 的 bitwise，根本没碰 S2/S3/S4 的合并 / gather / topology 工作。2026-06-20 capstone PR-12 #156 重新签证。

## `B1a-tag` 的处理

PR-12 capstone 之后：

- `B1a-tag` = `f7f992cabab5d5aec3bf08ab2db7c0669ef7fe75` / SHUD `0b3998d` — orchestrator 已 force-update from prior `64569b3` (S1d-end snapshot) at B1a capstone (2026-06-21)。
- 原 `64569b3` 时刻的 S1d snapshot 不再作为 B1a 实质引用；后续 strict 阶段（B1b/P1+）的"vs B1a-tag" 对比一律走 force-update 之后的新 tag。
- B1a-tag force-update 完成：旧 commit `64569b3` → 新 commit `f7f992c`，已 push 到 origin（`git ls-remote --tags origin | grep B1a-tag`）。

## B1a 完成时间线

PR-1 through PR-12 共 12 个 PR + 1 个 capstone（PR-7 拆 7a/7b 两个 PR），覆盖 S2.1–S2.17 + S3a/S3b/S3c + S4.1–S4.7 全部子项：

- PR-1 #144 [S2.10 + S2.14] — MD_ET 孤立 omp for 移除 + 16 标量移入 + snapshot sanity
- PR-2 #145 [S2.6 + S2.9] — 负状态 clamp + f_applyDY_omp data race
- PR-3 #146 [S2.7] — lake reset 前置（S2.8 deferred → PR-9 per D14）
- PR-4 #147 [S2.1 + S2.2 + S2.5 + S2.11] — lake-related 4 项（record-only）
- PR-5 #148 [S2.3] — ET flux non-lake（record-only）
- PR-6 #149 [S2.4] — river DY 公式（record-only）
- PR-7a #150 [S2.12/13/15/16] — record-only 4 项
- PR-7b #151 [S2.17] — assert + DEBUG 6 case
- PR-8 #152 [S2 capstone] — 删 `MD_f_omp.cpp` + 退役 `LEGACY_RHS` + `SHUD_LEGACY_OMP_RHS`
- PR-9 #153 [S3a + S3b + S2.8 D14] — 死代码 4 + 共享写拆分 4 + PassValue 临时扩 lake gather（8 commits）
- PR-10 #154 [S4.1-S4.7] — 7 adjacency lists + `docs/topology_manifest.yaml` + fallback unit test + CI gates
- PR-11 #155 [S3c.1+S3c.2+S3c.3] — PassValue → `rhs_deterministic_gather()`（3 commits）
- PR-12 #156 [B1a capstone] — archive specs + 6-case bitwise + B1a-tag force-update + branch lock

## S1 阶段做完的事（保留）

S1 把 SHUD 的 RHS 路径从 "serial / omp 两套并行演化" 收敛到 "一份 kernel + ExecPolicy 派生"，把 `_OPENMP_ON` 三义宏拆解成三正交宏。整个过程 Config A 默认 binary vs B0 bitwise neutrality：

- 抽 RHS core 到 `Model_Data::rhs_core(Y, DY, t, ExecPolicy)`。
- 三正交宏：`SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS`。
- 全树退役：`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 三个标识在 `SHUD/src/` 下 grep 结果归零。
- N_Vector 接口统一：`N_VGetArrayPointer` 取代 `NV_DATA_S / NV_DATA_OMP`；generic `N_VDestroy` 取代 `N_VDestroy_Serial`。
- CI 4-case × LEGACY_RHS 双轴 matrix 在 PR fast-feedback (2 jobs) 与 nightly cron (8 jobs) 都常开；snapshot 90d 窗口 HARD-fail gate、CVODE 15-key 全键 diff、4 个 grep gate 全部固化在 workflow。

S1 8 个 substage 时间线（S1d-end）：

- S1a (#44) — RHS update / f_update 抽取 + 周期性 IC backup probe。
- S1b (#45) — RHS flux（PassValue 边界）+ before-PassValue 12 张 snapshot golden。
- S1c (#46) — RHS apply / river DY + 4-case 8/8 + 24/24 snapshot；negative test 验 gate 起作用。
- S1d.1 (#47) — `LEGACY_RHS` 原子开关 + `rhs_core(ExecPolicy)` 落地。
- S1d.2 (#48 + #49) — 三正交宏 + `_OPENMP_ON` 退役 + N_Vector 统一 + 4 Config 矩阵固化为 Makefile target。
- S1-ci-A (#50) — CI 4-case × LEGACY_RHS 双轴 matrix + 4 grep gates。
- S1-ci-B (#51) — fetch-tags + B0-tag smoke + snapshot 90d HARD-fail + CVODE 15-key diff + label types 扩展。
- #52 — S1d-end snapshot（旧 "B1a-tag capstone"，事后判定为 S1d-end snapshot；2026-06-20 PR-12 重新签证 B1a）。

S1d-end 代码层面（vs B0）：SHUD `78c37a1` → `58327c5`；外层 commit `884cfb13` → `64569b3`。

## S2 / S3 / S4 后续 hand-off（PR-1..PR-11 已落地）

### S2 — 语义对齐 + 合并 `_omp` 路径（master plan L1084–L1198）

已删 `f_update_omp()` / `f_loop_omp()` / `f_applyDY_omp()` 三个独立函数（PR-8 capstone）。S2.1–S2.17 共 17 个子项分布在 PR-1 #144 through PR-7b #151；PR-8 #152 是 S2 capstone，删 `MD_f_omp.cpp` 整文件并退役 `LEGACY_RHS` / `SHUD_LEGACY_OMP_RHS` 两个 gating macro。S2 末点 vs B0 bitwise identical。

### S3 — 拆 flux compute + deterministic gather（master plan L1202–L1274）

`PassValue()` 整体替换为 `rhs_deterministic_gather()` —— PR-11 #155 收尾。S3a 4 项死代码 + S3b 4 项共享写拆分在 PR-9 #153 落地；S3c.1+S3c.2+S3c.3 在 PR-11 #155 落地。所有 gather 顺序 = B0 serial loop order，RHS bitwise identical。

### S4 — 固定拓扑顺序 + owner 映射（master plan L1277–L1321）

7 adjacency list（`seg_by_riv` / `seg_by_ele` / `upstream_by_down` / `riv_in_by_lake` / `ele_by_lake` / `lake_bank_edge_by_lake` / `edge_by_ele`）已在 PR-10 #154 落地，`docs/topology_manifest.yaml` 记录每个 list 排序规则 + B0 对应代码行号；`id == index+1` assert 已加；adjacency fallback unit test 落 CI。

## PR-12 capstone 验证结果（2026-06-20）

- 4-case Mac 本地 bitwise vs B0-tag：keliya / xinanjiang_upstream / qinyijiang / qhh rivqdown.dat 全 PASS。
- qhh 3 个 lake outputs（lakystage / lakqrivin / lakqrivout）bitwise vs B0-tag PASS。
- kashigeer：N/A（S0-13 deferred-upstream forcing-gap）。
- heihe / heihe_x4：直接服务器 Slurm bitwise PASS。orchestrator 通过免密 SSH (`210.77.77.22:32099`) 在 `cn08` 上跑 90 天截断（cfg.para 端口 START=14245 END=14335 for heihe / START=1 END=91 for heihe_x4，NUM_OPENMP=1，SHUD `0b3998d`）：
  - JobId 8537 heihe wall=500s SHA256 `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f` == B0-tag golden
  - JobId 8538 heihe_x4 wall=1290s SHA256 `f90601ef5738b972d688016ba1ee74f92ecb54faddaf46e4e2232f9d46567524` == B0-tag golden
  - follow-up issue #171 同步关闭（PR-12 直接覆盖了原打算延后的范围）
- 7+1 grep gate 0 hits：`MD_f_omp.cpp` absent / `PassValue\b` / `SHUD_LEGACY_OMP_RHS` / `LEGACY_RHS` / `_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` / bonus `f_update_omp|f_loop_omp|f_applyDY_omp`。
- openspec archive：4 specs（`b1a-capstone` / `s2-semantic-merge` / `s3-deterministic-gather` / `s4-adjacency-topology`）已 PROMOTE 为 system-level capability specs at `openspec/specs/<capability>/spec.md`（new tracked files），并将 change 文件夹 archive 到 `openspec/changes/archive/2026-06-20-b1a-finalization/`（local-only per `.gitignore`，匹配前两次 archive precedent `2026-06-19-s1-rhs-core-extraction/` + `2026-06-19-s2-pre-spec-housekeeping/`）。
- CI workflow `serial-baseline.yml` B1a 定版 final state read-only verified：1-axis matrix + S2 capstone grep gate + PR-11 PassValue gate + topology_manifest schema + adjacency fallback test + snapshot 90d HARD-fail + CVODE 15-key SHA256 diff 全在位。

## 验证 B1a-tag（force-update 后）

```
git ls-remote --tags origin | grep B1a-tag
# refs/tags/B1a-tag           -> <tag object SHA>（annotated）
# refs/tags/B1a-tag^{}        -> <new commit SHA, SHUD pin 0b3998d>
```

force-update 由 orchestrator 在 PR-12 squash-merge 后执行（在 in-PR 范围之外）。
