# B1a 基线总结

## 背景与定义

B1a 基线指 master plan §3 所定义的"重构等价的单线程结果 (refactor-equivalent serial result)"，即 SHUD 源码在完成 S0–S4 四个阶段重构之后、在默认编译配置下与 B0 保持比特一致 (bitwise identical) 的单线程产物。本项目中 **stage 指工作阶段** (S0/S1/S2/S3/S4)，**baseline 指工作产物** (B0/B1a)，二者不存在一一对应关系；B1a 并非任何单一 stage 的产物，而是 S0–S4 全部完成后才能签证的检查点。

截至 2026-06-21，S0–S4 全部完成（PR-1 #144 至 PR-11 #155 已合入 `baseline/B1a`），并由 PR-12 #156 作为 B1a capstone 完成 openspec 归档、6 案例 bitwise 比对、7 项 grep 守门 (grep gate) 强制启用与文档收尾。

## 历史复盘：旧版 `s1_summary.md` 错在哪（已修正）

2026-06-18 时点的 `docs/s1_summary.md` 标题写为"S1（B1a refactor — 唯一 RHS core）完成于 2026-06-18。`B1a-tag` 已打。可以进入 S2"——该陈述直接将 S1 等同于 B1a，违反 master plan §5 中关于 B1a 完成定义的契约。具体落差对照如下：

| master plan B1a 契约（§3 + §5） | 实际 S1 完成时已做 | 缺失项（已在 PR-1..PR-11 补齐） |
|---|---|---|
| S0 锁 B0 baseline | 已完成 #3–#18，`B0-tag` = `884cfb13` / SHUD `78c37a1` | — |
| S1 抽取 serial RHS core | 已完成 #44 (S1a) + #45 (S1b) + #46 (S1c) + #47 (S1d.1) + #48 #49 #50 (S1d.2–S1d.5) | — |
| S2 语义对齐 + 合并 `_omp` 路径（**删除 `f_update_omp / f_loop_omp / f_applyDY_omp` 三个独立函数**） | 原 0 commit | S2.1–S2.17 共 17 个子项已在 PR-1 #144 至 PR-8 #152 落地 |
| S3 拆分 flux compute + 确定性 gather（删除 `PassValue()` 共享 `+=`） | 原 0 commit | S3a 4 项死代码 + S3b 4 项共享写拆分 + S3c 3 项 gather 重构已在 PR-9 #153 + PR-11 #155 落地 |
| S4 固定拓扑顺序 + owner 映射（邻接表 adjacency list） | 原 0 commit | S4.1–S4.7 共 7 个 adjacency list + topology manifest + `id == index+1` 断言已在 PR-10 #154 落地 |

旧版 `status_matrix.md` L20 中"B1a 行 PASS"属于过早签证 (premature sign-off)：彼时仅验证了 S1 的 bitwise 性质，并未触及 S2/S3/S4 所要求的合并、gather 与 topology 工作。2026-06-20 由 capstone PR-12 #156 重新执行了完整签证。

## `B1a-tag` 的处理

PR-12 capstone 完成后，对 `B1a-tag` 作如下处理：

1. `B1a-tag` 当前指向 commit `f7f992cabab5d5aec3bf08ab2db7c0669ef7fe75` / SHUD pin `0b3998d`。orchestrator 在 B1a capstone (2026-06-21) 将其从原 `64569b3`（S1d-end snapshot）force-update 至上述新 commit。
2. 原 `64569b3` 时刻的 S1d snapshot 不再作为 B1a 的实质引用；后续 strict 阶段（B1b/P1+）的"vs B1a-tag"对比一律以 force-update 之后的新 tag 为准。
3. B1a-tag 的 force-update 已完成（旧 commit `64569b3` → 新 commit `f7f992c`），并已 push 至 origin，可通过 `git ls-remote --tags origin | grep B1a-tag` 校验。

## B1a 完成时间线

B1a 共由 12 个 PR 加 1 个 capstone 构成（其中 PR-7 拆分为 7a/7b 两个 PR），覆盖 S2.1–S2.17、S3a/S3b/S3c 与 S4.1–S4.7 全部子项：

- PR-1 #144 [S2.10 + S2.14]：MD_ET 孤立 omp for 移除 + 16 标量内联 + snapshot sanity。
- PR-2 #145 [S2.6 + S2.9]：负状态 clamp + `f_applyDY_omp` 数据竞争修复。
- PR-3 #146 [S2.7]：lake reset 前置（S2.8 deferred → PR-9 per D14）。
- PR-4 #147 [S2.1 + S2.2 + S2.5 + S2.11]：lake 相关 4 项（record-only）。
- PR-5 #148 [S2.3]：non-lake ET flux（record-only）。
- PR-6 #149 [S2.4]：river DY 公式（record-only）。
- PR-7a #150 [S2.12/13/15/16]：record-only 4 项。
- PR-7b #151 [S2.17]：assert + DEBUG 6 案例。
- PR-8 #152 [S2 capstone]：删除 `MD_f_omp.cpp` + 退役 `LEGACY_RHS` 与 `SHUD_LEGACY_OMP_RHS`。
- PR-9 #153 [S3a + S3b + S2.8 D14]：死代码 4 项 + 共享写拆分 4 项 + PassValue 临时扩 lake gather（8 commits）。
- PR-10 #154 [S4.1–S4.7]：7 个 adjacency list + `docs/topology_manifest.yaml` + fallback 单元测试 + CI gate。
- PR-11 #155 [S3c.1 + S3c.2 + S3c.3]：PassValue → `rhs_deterministic_gather()`（3 commits）。
- PR-12 #156 [B1a capstone]：specs 归档 + 6-case bitwise + B1a-tag force-update + 分支锁定。

## S1 阶段工作内容（保留）

S1 阶段将 SHUD 的 RHS 路径由"serial / omp 两套并行演化"收敛至"一份 kernel + ExecPolicy 派生"，同时将三义宏 `_OPENMP_ON` 拆解为三正交宏 (three orthogonal macros)。整个过程在 Config A 默认二进制 (binary) 配置下与 B0 保持 bitwise neutrality：

1. 将 RHS core 抽取至 `Model_Data::rhs_core(Y, DY, t, ExecPolicy)`。
2. 引入三正交宏：`SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS`。
3. 完成全树退役：`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 三项标识在 `SHUD/src/` 下的 grep 结果归零。
4. 统一 N_Vector 接口：以 `N_VGetArrayPointer` 取代 `NV_DATA_S / NV_DATA_OMP`；以 generic `N_VDestroy` 取代 `N_VDestroy_Serial`。
5. CI 固化：4-case × LEGACY_RHS 双轴矩阵在 PR fast-feedback（2 jobs）与 nightly cron（8 jobs）双通道常开；snapshot 90 日窗口 HARD-fail gate、CVODE 15-key 全键 diff 与 4 项 grep gate 全部纳入 workflow。

S1 共 8 个 substage，时间线（截至 S1d-end）如下：

- S1a (#44)：RHS update / `f_update` 抽取 + 周期性 IC backup probe。
- S1b (#45)：RHS flux（PassValue 边界）+ before-PassValue 12 张 snapshot golden。
- S1c (#46)：RHS apply / river DY + 4-case 8/8 + 24/24 snapshot；negative test 验证 gate 有效性。
- S1d.1 (#47)：`LEGACY_RHS` 原子开关 + `rhs_core(ExecPolicy)` 落地。
- S1d.2 (#48 + #49)：三正交宏 + `_OPENMP_ON` 退役 + N_Vector 统一 + 4 Config 矩阵固化为 Makefile target。
- S1-ci-A (#50)：CI 4-case × LEGACY_RHS 双轴矩阵 + 4 项 grep gate。
- S1-ci-B (#51)：fetch-tags + B0-tag smoke + snapshot 90 日 HARD-fail + CVODE 15-key diff + label types 扩展。
- #52：S1d-end snapshot（旧"B1a-tag capstone"，事后判定为 S1d-end snapshot；2026-06-20 由 PR-12 重新签证 B1a）。

S1d-end 在代码层面的位移（vs B0）：SHUD `78c37a1` → `58327c5`；外层 commit `884cfb13` → `64569b3`。

## S2 / S3 / S4 后续 hand-off（PR-1..PR-11 已落地）

### S2 — 语义对齐 + 合并 `_omp` 路径（master plan L1084–L1198）

已删除 `f_update_omp()` / `f_loop_omp()` / `f_applyDY_omp()` 三个独立函数（PR-8 capstone）。S2.1–S2.17 共 17 个子项分布于 PR-1 #144 至 PR-7b #151；PR-8 #152 为 S2 capstone，删除 `MD_f_omp.cpp` 整文件并退役 `LEGACY_RHS` 与 `SHUD_LEGACY_OMP_RHS` 两项 gating macro。S2 末点与 B0 bitwise identical。

### S3 — 拆分 flux compute + 确定性 gather（master plan L1202–L1274）

`PassValue()` 已整体替换为 `rhs_deterministic_gather()`，由 PR-11 #155 收尾。S3a 4 项死代码 + S3b 4 项共享写拆分于 PR-9 #153 落地；S3c.1 + S3c.2 + S3c.3 于 PR-11 #155 落地。所有 gather 顺序与 B0 serial loop 顺序一致，RHS 输出 bitwise identical。

### S4 — 固定拓扑顺序 + owner 映射（master plan L1277–L1321）

7 个 adjacency list（`seg_by_riv` / `seg_by_ele` / `upstream_by_down` / `riv_in_by_lake` / `ele_by_lake` / `lake_bank_edge_by_lake` / `edge_by_ele`）已在 PR-10 #154 落地。`docs/topology_manifest.yaml` 记录每个 list 的排序规则与 B0 对应代码行号；`id == index+1` 断言已加入；adjacency fallback 单元测试已纳入 CI。

## PR-12 capstone 验证结果（2026-06-20）

PR-12 capstone 的验证覆盖本地 Mac 与服务器双端，结果如下：

1. **Mac 本地 4-case bitwise vs B0-tag**：keliya / xinanjiang_upstream / qinyijiang / qhh 的 `rivqdown.dat` 全部 PASS。
2. **qhh 三项 lake outputs**（`lakystage` / `lakqrivin` / `lakqrivout`）bitwise vs B0-tag PASS。
3. **kashigeer**：N/A（属 S0-13 deferred-upstream forcing-gap）。
4. **heihe / heihe_x4**：直接在服务器 Slurm 上验证 bitwise PASS。orchestrator 通过免密 SSH (`210.77.77.22:32099`) 在 `cn08` 上执行 90 日截断运行（`cfg.para` 端口设定为 heihe `START=14245 END=14335` / heihe_x4 `START=1 END=91`，`NUM_OPENMP=1`，SHUD pin `0b3998d`）：
   - JobId 8537 heihe，wall=500 s，SHA256 `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f`，与 B0-tag golden 一致；
   - JobId 8538 heihe_x4，wall=1290 s，SHA256 `f90601ef5738b972d688016ba1ee74f92ecb54faddaf46e4e2232f9d46567524`，与 B0-tag golden 一致；
   - follow-up issue #171 同步关闭（PR-12 已覆盖其原计划的延后范围）。
5. **7+1 grep gate 0 hits**：覆盖 `MD_f_omp.cpp absent` / `PassValue\b` / `SHUD_LEGACY_OMP_RHS` / `LEGACY_RHS` / `_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` / bonus `f_update_omp|f_loop_omp|f_applyDY_omp`。
6. **openspec 归档**：4 份 specs（`b1a-capstone` / `s2-semantic-merge` / `s3-deterministic-gather` / `s4-adjacency-topology`）已 PROMOTE 为 system-level capability specs，存于 `openspec/specs/<capability>/spec.md`（新增 tracked 文件）；change 文件夹归档至 `openspec/changes/archive/2026-06-20-b1a-finalization/`（local-only per `.gitignore`，与前两次归档先例 `2026-06-19-s1-rhs-core-extraction/` + `2026-06-19-s2-pre-spec-housekeeping/` 保持一致）。
7. **CI workflow `serial-baseline.yml`** B1a 定版后只读复核 (read-only verified)：1-axis matrix + S2 capstone grep gate + PR-11 PassValue gate + topology_manifest schema + adjacency fallback test + snapshot 90 日 HARD-fail + CVODE 15-key SHA256 diff 全部到位。

## 验证 `B1a-tag`（force-update 之后）

```
git ls-remote --tags origin | grep B1a-tag
# refs/tags/B1a-tag           -> <tag object SHA>（annotated）
# refs/tags/B1a-tag^{}        -> <new commit SHA, SHUD pin 0b3998d>
```

force-update 由 orchestrator 在 PR-12 squash-merge 之后执行，处于 in-PR 范围之外。
