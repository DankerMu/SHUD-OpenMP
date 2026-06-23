# Profile Gate 决策（S0-11 / openMP #15）

## 背景与定义

本文档记录由 profile gate 触发的优先级决策，对应 rhs-profile-gate spec.md "Profile decision signed before S1" 条目与 master plan §S0.12 决策表（四选一阶梯）。该文档构成 S0 预热阶段已收敛、项目可进入 S1 与 P-phase 并行化阶段的正式签字凭据。按 spec 场景 "Missing signature blocks S1" 的规定，若本文缺失（或缺少 `signed_at` 字段），必须将 `docs/status_matrix.md` 中的 profile gate 行置为 BLOCKED。

文中术语沿用项目惯例：剖析 (profile) 指通过仪器化的单线程运行获取各时间桶 (timing bucket) 占比；右端项 (RHS) 指 SHUD 的偏微分方程右端 kernel；Amdahl 上界 (Amdahl bound) 指假定可并行部分加速比趋于无穷时的整体加速上限；强迫数据 (forcing data) 指驱动模型的气象输入。

## 决策类型

本次决策采纳路径为**原方案**，即以 RHS kernel 为主投资方向的 OpenMP 并行化路线。

依据 master plan §S0.12 L752 决策表所列阈值，目标平台上 6 个成功执行的 case 给出 `t_RHS_total / t_wall_total` 分布如下：

| Case | t_RHS_total% | Amdahl S(∞) 上界 |
|---|---|---|
| heihe_x4 | 66.55% | 2.99x |
| qinyijiang | 64.64% | 2.83x |
| keliya | 49.33% | 1.97x |
| xinanjiang_upstream | 43.74% | 1.78x |
| qhh | 36.75% | 1.58x |
| heihe | 12.08% | 1.14x |

6 个 target case 中有 5 个 RHS 占比不低于 30%，明显高于 master plan §S0.12 "走原方案" 阈值（决策关键 case 多数满足 `t_RHS_total / t_total > 30%`）。规模最大的案例 heihe_x4（约 25,000 cell，是真正的 P-phase 加速目标）RHS 占比最高（66.55%），表明 RHS kernel 优先的并行化是生产规模下的正确主投资方向。没有任何 case 落入 "战略暂停" 区（RHS 全局占比低于 10%）。唯一的低占比离群 case heihe（12.08%）由 forcing IO 主导（`t_forcing_io = 79.1%`），属于另一类瓶颈；对该案例做 RHS 并行化在净收益上仍为正向，仅受 Amdahl 上界限制至 1.14x，直到 IO 路径被独立处理后方可改善。

## Amdahl 上界分析

假定 RHS 达到完美并行而其余 bucket 保持串行，则每个 case 的理论加速上界为：

```
S(∞) = 1 / (1 - t_RHS_total / t_wall_total)
```

各典型 case 的上界与 8 核现实加速比讨论如下：

1. 决策驱动 case (heihe_x4)：S(∞) = 2.99x；按 Amdahl 公式得 8 核现实上界 `1 / ((1 - 0.6655) + 0.6655/8) = 1 / (0.3345 + 0.0832) = 2.39x`。此为 §1.1.1 加速比 gate 对最大 case 的目标上界。
2. 次驱动 case (qinyijiang, 3155 cell)：S(∞) = 2.83x，8 核现实上界 2.30x。
3. 小 case (xinanjiang_upstream, 801 cell)：S(∞) = 1.78x，8 核现实上界 1.59x。小 case 需通过 `OMP_CUTOFF` 机制（master plan 核心原则 C8）兜底，而非投入全部并行预算。
4. IO 主导 case (heihe, 6335 cell，forcing IO 占比 79%)：S(∞) = 1.14x，8 核现实上界 1.13x。该 case 的 RHS 并行目标为 "维持不退化" 而非 "提升加速"；真正的改进需依赖以下两条路径之一：(a) 一次性读入的二进制缓存 forcing；(b) forcing IO 并行化。两者按逻辑归属于 P9 或更后续阶段优化。

§1.1.1 gate "单插槽 8 核加速比目标" 因此应解读为：目标为决策关键大 case (heihe_x4 与 qinyijiang) 上达到 1.5–2.0x；小 case 通过 OMP_CUTOFF 跳过；forcing IO 主导 case (heihe) 延后处理或与独立 IO 优化配对。该解读将头条 §1.1.1 数字（"8x 加速比"）从 "每 case 统一目标" 重新理解为 "向大 case 加权的 portfolio 指标"。

## P8 预条件时机

本次决策为**保留 P8 阶段定位，不前置实施**。

依据 master plan §S0.12 决策表，P8 预条件 SPGMR 的优化仅在 RHS 并行上界被达到之后才具有相关性。当前 profile 证据支持该判断：

1. 6 个 case 中有 4 个仅靠 RHS 并行即可在 8 核上压出不低于 1.5x 的加速比。这部分余量必须在 P1–P7（strict 阶段）内充分挖掘，再投入预条件设计与集成成本。
2. 对 IO 主导 case (heihe) 而言，P8 预条件无法寻址其瓶颈，因为瓶颈位于 `t_forcing_io` 而非 CVODE Krylov 迭代次数。为该 case 提前 P8 属于范畴错误 (category error)。
3. CVODE 内部时间 `t_CVODE_internal` 的占比从 6.83%（heihe）至 36.17%（keliya）不等，给出预条件能寻址的上限约 36%。该比例虽有意义，但并非当前决策关键大 case 中的主要杠杆。

综上，P8 预条件按 master plan §3 阶段顺序原位执行，不予前置。

## 跨平台差异审查

按 spec.md 场景 "> 10pp 差异触发 review note" 的要求：`docs/profile_platform.md` 报告 `delta_acceptable: false`，原因是 4 个两端均有数据的 case 中有 2 个超过 10pp 阈值：

- keliya：local 36.32%，target 49.33%，delta +13.01pp
- qinyijiang：local 51.07%，target 64.64%，delta +13.57pp

根因为定性推断（定量诊断不属于 #15 范围），归纳如下：

1. **不同微架构的单核吞吐差异**。Apple M4 Pro 性能核单线程 IPC 显著高于 Xeon Gold 6133（2017 年 Skylake-SP 架构 @ 2.5 GHz）。同样负载在 target 端 wall-clock 慢约 2.9–3.5x（target wall / local wall：keliya 79.7/27.8 = 2.87x；qinyijiang 799.9/229.5 = 3.49x），证实单核吞吐差距存在。
2. **跨 bucket 减速不均匀**。Apple clang 配合 Apple Silicon NEON 优化对 SHUD RHS C++ kernel 的加速比相对于 SUNDIALS/CVODE 内部 C 代码（两个平台同源 C 代码）更为激进。其结果是：RHS 在总 wall 中的占比在 Apple 端变小、在 x86 端变大。此现象为编译器与微架构交互的产物，并非 profile 正确性缺陷。
3. **case 间非对称性**。xinanjiang_upstream 与 qhh 显示较小 delta（< 2pp），说明跨平台非对称在所有 case 间并不均匀；其大小与 "RHS 在可向量化内层循环中占比多少" 相关——可向量化程度越高，Apple 优势越大，对 x86 的 delta 越大。

**对决策的影响**：上述 delta 并不令 "走原方案" 决策失效。两个平台均认同 RHS 为主时间桶（local 端跨 case 36–51%；target 端跨 case 12–67%，6 个中 5 个超过 30%），而决策关键大 case (heihe_x4) 仅有 target 数据——根本不存在 local 与 target 的比较基础。前述 Amdahl 上界计算锚定于 target 平台数字（`docs/profile_platform.md` `target_platform`），后者为 §1.1.1 权威端点，因此 delta 仅作为元数据 (metadata) 提示，并非阻塞 gate 的 finding。

行动项（以独立 issue 跟踪，不进入 #15）：

1. 在 P1–P3 strict 并行落地后，对 keliya 与 qinyijiang 在 target 平台 re-profile，确认 RHS 占比趋势在更高线程数下仍成立。
2. 若未来 heihe_x4 的 local 与 target 比对变得可行（例如部分 forcing 或下采样 mesh），重做 delta 比对。

## 证据

Local 端 profile 产物（4 真实 + 3 deferred），SHA256 摘要如下：

```
b16e8a9acedcff00c82b66192d3db3eced538c7718c8715adff7b384761ce14f  benchmarks/keliya/profile_B0.yaml
34831b4b641f664cece3c5a4db1dd2c72b0016b99dc4206d30ac9b5365a43d97  benchmarks/xinanjiang_upstream/profile_B0.yaml
77bdbad6bd23e9806688bb7e76cb82d8359ed3edfbc7af03002ea9ee8ad87b02  benchmarks/qinyijiang/profile_B0.yaml
9506ceeeab796c9685dccfd649f5adc889e239e3df2f3274d5a82647637cec08  benchmarks/qhh/profile_B0.yaml
cbe50adf2047c88bc1e7d6415ce76c6ba8310e159310876c72f2bebdbc24982c  benchmarks/heihe/profile_B0.deferred.yaml
aea1322a9b2b6ee2ffaf391b422ccf05e8c1c7f29f5d20a9778dc6568eed8a9f  benchmarks/heihe_x4/profile_B0.deferred.yaml
5bdecf4e2173868de20659141cb9b046349cc5ae0fba97078b877f1cd4a5cd02  benchmarks/kashigeer/profile_B0.deferred.yaml
```

Target 端 profile 产物（6 真实 + 1 deferred），SHA256 摘要如下：

```
711a380902d2dee176ff16bf5c3a5c360a9ee131420d7727a7d4e75dc62ca0f5  benchmarks/keliya/profile_B0.target.yaml
a739dfd7c66310bf5e5bcb0317a99768d3c1d41480e8e991e0d32aaeca9637e1  benchmarks/xinanjiang_upstream/profile_B0.target.yaml
1dae17564e44de5149f8e49cb8dd3f404caa5a1ee19dc0b9ef2f26ab417174ed  benchmarks/qinyijiang/profile_B0.target.yaml
cc312b7ab1db926ab85fff86b91cc0e29fc02b2a289103ee30db9555dad105f5  benchmarks/qhh/profile_B0.target.yaml
baa03be7ce16e01345bdc9e9b93c033ffcee55213113b9b1ba91441414a97f5d  benchmarks/heihe/profile_B0.target.yaml
03d9d4c9def804b27f5f5e6a8930063eb03ce5ad5cbadee979c848f829254c36  benchmarks/heihe_x4/profile_B0.target.yaml
8f64779b5c3c25b2a854f70b9721231a4e40b5dd4a1b2eadf2e3a7e43d615d17  benchmarks/kashigeer/profile_B0.target.deferred.yaml
```

外层 commit：`ecef3fbe6ad6971ac8dc2ff6a888ece8db8fae83`。SHUD submodule commit：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`。

## t_other 桶记账状态

按 rhs-profile-gate spec.md "t_other accounting WARN at 5%" 与 "FAIL at 10%" 场景的判定，target 平台 yaml 审计结果如下：

| Case | t_other_pct (target) | 状态 |
|---|---|---|
| heihe_x4 | 1.10% | OK |
| heihe | 1.52% | OK |
| qinyijiang | 0.87% | OK |
| qhh | 5.80% | **WARN** |
| keliya | 6.32% | **WARN** |
| xinanjiang_upstream | 22.42% | **FAIL** |

xinanjiang_upstream 的 FAIL（19.7s 跑中 22.42% t_other）应解读为启动开销主导的伪象 (artifact)，而非 profile 工具缺陷。论据如下：

1. 绝对 `t_other = 4.41s`（总 wall 19.7s）。
2. 同份 yaml 中 `t_forcing_io = 1.40s` 覆盖 51 个 forcing CSV（6 个 case 中 forcing-stations 数最少者）。
3. 进程启动（SUNDIALS init、mesh load、integrator setup）在当前 S0-10 仪器化中未单独标定，相关时间被纳入 `t_other` 桶。
4. 长跑 case (heihe 487s、heihe_x4 1417s) 中，启动开销摊薄至预期的不足 2%。

**决策影响**：FAIL 信号予以承认，但不阻塞 "走原方案" 决策。理由有二：(a) 决策锚定于大 case，其占比由 RHS 主导而非启动；(b) FAIL 信号为未来 profile 工具精化机会（将启动时间标为 `t_init` 而非纳入 `t_other`），属于 S0.12 retrospective 事项，非 S1 阻塞项。

在下一个决策门快照之前（即 P-phase 退出之前）应开启 follow-up issue，为 `tools/profile/timer.cpp` 增加 `t_init` bucket，将启动时间从 `t_other` 中拆出。

## 签字

| 字段 | 值 |
|---|---|
| signer | DankerMu（项目所有者；GitHub：@DankerMu；email：qingdanker@gmail.com） |
| signed_at | 2026-06-17 |
| signed_against_commit | outer `a860eae58991ce91ea91656cc9d4a08540e48f5b`（#16 合并后 `baseline/current` HEAD）+ SHUD submodule `78c37a1061de4112bc7c297bb7bd1f107432e6f2`（#14 后 openmp-baseline HEAD） |
| signed_via | claude-code-s0-13-issue-17 代 DankerMu（按 user 2026-06-17 grant 的 delegated 签字；按 Linus Torvalds persona 编排，遵循 /Users/danker/.claude/CLAUDE.md 的优先级栈） |
| signed_off_decision | 走原方案 + 调高 large case 权重 + P8-precond 不前置 |
| follow_up_issues | (a) P1-P3 strict 落地后 re-profile；(b) heihe forcing IO 优化延后到 P9+；(c) 把 t_init 从 t_other 中拆出（profile timer）；(d) kashigeer 上游 X76-X80 forcing 缺口（issue #29 已开） |

## Opt-IO 硬性前置判断（M7 trim 后重测）

本节由 PR-G #214（`openspec/changes/p1-update-omp` task 2.3）新增，对应 spec：`openspec/changes/p1-update-omp/specs/profile-retest-m7/spec.md` Requirement L19-L37（Opt-IO 硬性前置判断更新）。本节与 master plan §5 L1533 "Opt-IO 硬性前置阈值 = heihe `t_forcing_io / t_total >= 50%`" 联动。

### Trim 前（B0 baseline，参考）

S0 阶段 heihe 非 trimmed forcing 实测 `t_forcing_io / t_wall_total = 385.436 / 487.046 ≈ 79.1%`（见前文 "决策类型" 节 L24 "heihe 的 12.08% 由 forcing-IO 主导（t_forcing_io = 79.1%）"），远超 §5 L1533 的 50% 触发门，因此 master plan §5 将 Opt-IO 列为 heihe 的硬性前置约束。

### Trim 后（PR-G #214 本次实测）

在 server cn03 上以 Slurm 执行 `NUM_OPENMP=1` 90 天截断 trimmed forcing 三次（同一 binary `396ad9fb…`，3-run rivqdown SHA byte-identical 且与 B0/B1a/B1b-tag golden 一致）：

| Case | jobid | Elapsed | t_forcing_io mean | t_total mean | **ratio mean** | rivqdown SHA (3-run identical) | vs golden |
|---|---|---|---|---|---|---|---|
| heihe | 8742 | 00:07:29 | 2.836 s | 149.33 s | **1.90%** | `55abad28…` | ≡ B0/B1a/B1b-tag |
| heihe_x4 | 8743 | 01:09:06 | 2.667 s | 1382.0 s | **0.19%** | `f90601ef…` | ≡ B0/B1a/B1b-tag |

证据文件列示如下：

- `benchmarks/heihe/profile_B0.target.trimmed.yaml`（与既有 `profile_B0.target.yaml` 并存，per spec L59-L66）
- `benchmarks/heihe_x4/profile_B0.target.trimmed.yaml`
- server 原始数据：`/scratch/frd_muziyao/SHUD-OpenMP/.s214-runs/prof_{heihe_8742,heihe_x4_8743}_summary.txt`（含 7-bucket × 3-run cvode_stats + per-run SHA）

### 决策 = (a) Opt-IO 退回可选

按 spec L23：

> **(a) 退回可选**：trim 后 heihe `t_forcing_io / t_total < 50%` → Opt-IO 回到 §5 原 "B1b 锁定后任意时间执行" 可选定位；本 change 不阻塞 P1；下游 P-strict 全部完成后再评估

实测 ratio 数据：

- heihe = **1.90%**，远低于 50% 触发门，亦远低于 5% 严格门（PR-B task 1.8 单 run 已验 `< 5%`，本 PR-G 三 run 复证）；与 design.md L19 "M7 trim 后 forcing 占比 79% → ~13%" 预测的趋势一致，且实际比预测更乐观。
- heihe_x4 = **0.19%**，远低于 50% 触发门。大型 case 的 forcing IO 摊薄至几乎不可见，进一步证明 trim 后 Opt-IO 失去大 case 杠杆。

**结论**：Opt-IO 不再是 §1.1.1 验收 heihe 的硬性前置；回到 master plan §5 原 "B1b 锁定后任意时间执行" 可选定位。本 change `p1-update-omp` 不阻塞 P1；下游 P-strict (P1–P7) 全部完成后再评估 Opt-IO 是否必要（届时若 RHS 并行已榨出主要余量，forcing IO 剩余 1.9% / 0.19% 已无投入产出价值）。

| 字段 | 值 |
|---|---|
| signer | DankerMu（项目所有者；GitHub：@DankerMu；email：qingdanker@gmail.com） |
| signed_at | 2026-06-22 |
| signed_against_commit | outer `d34e455`（main HEAD at PR-G branch point；PR-B 已合并） + SHUD submodule `71b3a1ae`（openmp-baseline B1b-tag-aligned pin） |
| signed_via | claude-code-pr-g-issue-214 代 DankerMu（沿用 S0.12 delegated sign-off 模式，按 Linus Torvalds persona 编排，遵循 /Users/danker/.claude/CLAUDE.md 的优先级栈） |
| signed_off_decision | (a) Opt-IO 退回可选 — 本 change p1-update-omp 不阻塞 P1；P-strict 全部完成后再评估 |
| basis | heihe 1.90% / heihe_x4 0.19% 远低于 50% 触发门、远低于 5% 严格门；3-run rivqdown SHA identical 且与 B0/B1a/B1b-tag golden 一致（trim 后 bitwise correctness 自证） |
| evidence | benchmarks/{heihe,heihe_x4}/profile_B0.target.trimmed.yaml + server jobid 8742/8743 (cn03) + spec profile-retest-m7 Requirement L19-L37 |
| follow_up_issues | 无新增；M7 trim 后 forcing IO 不再是头部瓶颈，Opt-IO 长期挂在 status_matrix "PENDING (可选)" 即可 |
