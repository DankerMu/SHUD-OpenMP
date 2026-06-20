# B1a Baseline 进度（IN PROGRESS）

> B1a = master plan §3 定义的"重构等价的单线程结果"，来自 **S0–S4 完成**。**stage = 工作阶段（S0/S1/S2/S3/S4）**；**baseline = 工作产物（B0/B1a）**；B1a 不是单一 stage 的产物，而是 **S0–S4 四个 stage 全部完成**之后才能签字的检查点。
>
> 当前进度：S0 ✅ + S1 ✅；**S2 / S3 / S4 全未做**。B1a baseline 未完成。

## 旧版 `s1_summary.md` 错在哪

2026-06-18 当时的 `docs/s1_summary.md` 标题写 "S1（B1a refactor — 唯一 RHS core）完成于 2026-06-18。`B1a-tag` 已打。可以进入 S2"——**直接把 S1 等同于 B1a**。这违反 master plan §5：

| master plan B1a 契约（§3 + §5） | 实际 S1 完成时做了 | 缺什么 |
|---|---|---|
| S0 锁 B0 baseline | ✅ #3–#18，B0-tag = `884cfb13` / SHUD `78c37a1` | — |
| S1 抽取 serial RHS core | ✅ #44 (S1a) + #45 (S1b) + #46 (S1c) + #47 (S1d.1) + #48 #49 #50 (S1d.2–S1d.5) | — |
| S2 语义对齐 + 合并 `_omp` 路径（**删 `f_update_omp / f_loop_omp / f_applyDY_omp` 三个独立函数**） | ❌ 0 commit | S2.1–S2.17 共 17 个子项 |
| S3 拆 flux compute + deterministic gather（删除 `PassValue()` 共享 `+=`） | ❌ 0 commit | S3a 4 项死代码 + S3b 4 项共享写拆分 + S3c 3 项 gather 重构 |
| S4 固定拓扑顺序 + owner 映射（adjacency list） | ❌ 0 commit | S4.1–S4.7 共 7 个 adjacency list + topology manifest + `id == index+1` assert |

`status_matrix.md` L20 "B1a 行 PASS" 是**过早签证**——只验证了 S1 的 bitwise，根本没碰 S2/S3/S4 的合并 / gather / topology 工作。

## `B1a-tag` 的处理

`B1a-tag = 64569b3` / SHUD `58327c5` **不等于** master plan §5 定义的真 B1a。其语义降级为 **"S1d-end snapshot"**：

- **保留作历史 reference**：S1 阶段确实完成，bitwise vs B0 也确实过了，可作 S1 完成时刻的 reproducible point。
- **不作 B1a 实质语义引用**：所有 "vs B1a-tag" 的对比都要重新定义——strict 阶段的 zero-参考点不是这里。
- **B1a-tag 重打时机**：S2 + S3 + S4 全部完成、签字 A1 bitwise vs B0 后，force-update tag 或新打 `B1a-final` tag。

## S1 阶段做完的事（事实陈述，不再宣称这是 B1a 完成）

S1 把 SHUD 的 RHS 路径从 "serial / omp 两套并行演化" 收敛到 "一份 kernel + ExecPolicy 派生"，把 `_OPENMP_ON` 三义宏拆解成三正交宏。整个过程 Config A 默认 binary vs B0 bitwise neutrality：

- 抽 RHS core 到 `Model_Data::rhs_core(Y, DY, t, ExecPolicy)`。
- 三正交宏：`SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS`。
- 全树退役：`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 三个标识在 `SHUD/src/` 下 grep 结果归零。
- N_Vector 接口统一：`N_VGetArrayPointer` 取代 `NV_DATA_S / NV_DATA_OMP`；generic `N_VDestroy` 取代 `N_VDestroy_Serial`。
- CI 4-case × LEGACY_RHS 双轴 matrix 在 PR fast-feedback (2 jobs) 与 nightly cron (8 jobs) 都常开；snapshot 90d 窗口 HARD-fail gate、CVODE 15-key 全键 diff、4 个 grep gate 全部固化在 workflow。

S1 8 个 substage 时间线：

- S1a (#44) — RHS update / f_update 抽取 + 周期性 IC backup probe。
- S1b (#45) — RHS flux（PassValue 边界）+ before-PassValue 12 张 snapshot golden。
- S1c (#46) — RHS apply / river DY + 4-case 8/8 + 24/24 snapshot；negative test 验 gate 起作用。
- S1d.1 (#47) — `LEGACY_RHS` 原子开关 + `rhs_core(ExecPolicy)` 落地。
- S1d.2 (#48 + #49) — 三正交宏 + `_OPENMP_ON` 退役 + N_Vector 统一 + 4 Config 矩阵固化为 Makefile target。
- S1-ci-A (#50) — CI 4-case × LEGACY_RHS 双轴 matrix + 4 grep gates。
- S1-ci-B (#51) — fetch-tags + B0-tag smoke + snapshot 90d HARD-fail + CVODE 15-key diff + label types 扩展。
- #52 — "B1a-tag capstone"（**这一签字事后被判定无效**，见上）。

代码层面（vs B0）：SHUD `78c37a1` → `58327c5`；外层 commit `884cfb13` → `64569b3`。

## 待做：S2 / S3 / S4

按 master plan §5 顺序门控，**每个 stage 通过 bitwise vs B0 才能进下一个 stage**。

### S2 — 语义对齐 + 合并 `_omp` 路径（master plan L1084–L1198）

目标：删 `f_update_omp()` / `f_loop_omp()` / `f_applyDY_omp()` 三个独立函数；逐项把"serial 缺失但 `_omp` 正确的语义"补进 core，绝不把"`_omp` 全部行为"照搬。每个 S2.x 子项独立 bitwise 验证。

17 个子项（每项明确 serial vs omp 差异 + 处置原则 + 文件行号）：S2.1 lake vertical / S2.2 lake horizontal / S2.3 ET flux / S2.4 river DY 公式 / S2.5 lake DY / S2.6 负状态 clamp / S2.7 lake 初始化 / S2.8 Qe2r/QrivSurf/Sub 清零 / S2.9 `f_applyDY_omp` data race / S2.10 `updateforcing()` 孤立 `omp for` / S2.11 lake element DY=0 / S2.12 uncoupled 路径 clamp（仅记录）/ S2.13 全局变量裸指针（仅记录）/ S2.14 `ET()` 孤立 `omp for` + 16 个标量 data race / S2.15 `AccTemperature.getACC()` 除零（仅记录，B1b 修复）/ S2.16 当前已使用 OpenMP N_Vector（已 S1d.3–S1d.5 解决）/ S2.17 `fun_Ele_sub()` lake 分支越界（仅 assert + 记录，B1b 修公式）。

S2 验收：所有改动后 vs B0 **bitwise identical**；语义 diff report 完成；输出改变类 bug fix（S2.15、S2.17）**不在 S2 实施**。

### S3 — 拆 flux compute + deterministic gather（master plan L1202–L1274）

目标：消除所有并行不安全的共享写，形成"纯计算 + owner-local gather"结构。

- S3a 删 4 项 `PassValue()` 已覆盖的死 `+=`（`MD_RiverFlux.cpp` L107/108/121/122）。
- S3b 拆 4 项不在 `PassValue()` 覆盖范围内的共享写（`QLakeRivIn` / `QLakeSurf` / `QLakeSub` / `qLakeEvap` 等）。
- S3c 把 `PassValue()` 整体替换为 `rhs_deterministic_gather()`，按 B0 serial loop order（即原始数组索引顺序）汇总。

S3 验收：所有 gather 顺序 = B0 serial loop order；RHS bitwise identical；水量平衡误差不变。**不允许**"顺序改变但锁定为新参考"——顺序变更属于 B1b 范畴。

### S4 — 固定拓扑顺序 + owner 映射（master plan L1277–L1321）

目标：构建并固定排序的 adjacency list，让所有 gather 有确定的 owner 和确定的贡献顺序。

7 个 adjacency list：`seg_by_riv` / `seg_by_ele` / `upstream_by_down` / `riv_in_by_lake` / `ele_by_lake` / `lake_bank_edge_by_lake` / `edge_by_ele`。排序规则：B0 serial loop 的**原始数组索引顺序**（不是 id 升序，除非 `id == index+1` assert 通过）。

S4 验收：adjacency list 后的 gather 与旧 `PassValue()` bitwise identical；topology manifest（YAML/JSON）记录每个 list 的排序规则 + B0 对应代码行号；所有 accumulator 有唯一 owner；`id == index+1` assert 已加入。

## 下一步动作

1. `status_matrix.md` L20 "B1a 行 PASS" 修订为 "B1a 行 IN-PROGRESS（S0+S1 PASS / S2-S4 PENDING）"。
2. 拉 S2 work：从 S2.1 开始（lake vertical 缺失），每个子项一个 PR，bitwise vs B0 验证。
3. GitHub `baseline/B1a` 分支 `lock_branch=true` 需要解开（前提是 user 同意此分支继续接受 S2-S4 commits）。
4. `B1a-tag` 处理：S4 完成后再决定 force-update 还是新打 `B1a-final`。

## 验证当前 `B1a-tag`（注：现语义为 S1d-end snapshot）

```
git ls-remote --tags origin | grep B1a-tag
# refs/tags/B1a-tag           -> 4fafb8e5... (annotated tag object)
# refs/tags/B1a-tag^{}        -> 64569b3f... (commit, SHUD pin 58327c5)
```
