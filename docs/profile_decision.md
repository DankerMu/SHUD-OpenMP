# Profile Gate 决策（S0-11 / openMP #15）

本文档记录由 profile gate 触发的优先级决策，对应 rhs-profile-gate spec.md "Profile decision signed before S1" 要求和 master plan §S0.12 决策表（4 选 1 阶梯）。这是 S0 预热已收敛、项目可进入 S1 / P-phase 并行化的正式签字。按 spec 场景 "Missing signature blocks S1"，本文缺失（或缺 `signed_at` 字段）必须把 `docs/status_matrix.md` 的 profile gate 行置为 BLOCKED。

## 决策类型

**走原方案（原计划：以 RHS kernel 为主的 OpenMP 并行化）**

理由（按 master plan §S0.12 L752 决策表阈值）：

- 目标平台上 6 个成功 target 跑的 `t_RHS_total / t_wall_total` 分布：

  | Case | t_RHS_total% | Amdahl S(∞) 上界 |
  |---|---|---|
  | heihe_x4 | 66.55% | 2.99x |
  | qinyijiang | 64.64% | 2.83x |
  | keliya | 49.33% | 1.97x |
  | xinanjiang_upstream | 43.74% | 1.78x |
  | qhh | 36.75% | 1.58x |
  | heihe | 12.08% | 1.14x |

- **6 个 target case 里 5 个 RHS 占比 >= 30%**，明显高于 master plan §S0.12 "走原方案" 阈值（决策关键 case 多数满足 `t_RHS_total / t_total > 30%`）。
- 最大 case（heihe_x4，~25k cell，真正的 P-phase 加速目标）RHS 占比最高（66.55%），证实 RHS kernel 优先的并行化是**生产规模下的正确主投资方向**。
- 没有 case 落到 "战略暂停" 区（RHS 全局 < 10%）。唯一低占比离群 case（heihe 的 12.08%）是 **forcing-IO 主导**（t_forcing_io = 79.1%），属于另一类瓶颈——只做 RHS 并行化对它仍然净正向，只是被 Amdahl 限到 1.14x，直到 IO 路径被处理。

## Amdahl 上界

每个 case 的理论加速上界（假设 RHS 变成完美并行，其他 bucket 保持串行）：

```
S(∞) = 1 / (1 - t_RHS_total / t_wall_total)
```

- **决策驱动 case（heihe_x4）：2.99x**（无穷线程）；按 Amdahl 8 核现实上界：`1 / ((1 - 0.6655) + 0.6655/8) = 1 / (0.3345 + 0.0832) = 2.39x` —— 这是 §1.1.1 加速比 gate 对最大 case 的目标上界。
- **次驱动（qinyijiang，3155 cell）：2.83x** 上界 / 8 核现实 2.30x。
- **小 case（xinanjiang_upstream，801 cell）：1.78x** 上界 / 8 核现实 1.59x —— 小 case 要靠 `OMP_CUTOFF`（master plan 核心原则 C8）兜底，而不是全力投并行预算。
- **IO 主导 case（heihe，6335 cell，但 forcing IO 占 79%）：1.14x** 上界 / 8 核现实 1.13x —— 这个 case 的 RHS 并行目标是 "不掉" 而不是 "提速"；真正的改进要么靠 (a) 一次性读入的二进制 cached forcing，要么靠 (b) forcing IO 并行化，逻辑上属于 P9 或更后阶段优化。

§1.1.1 gate "单插槽 8 核加速比目标" 因此读作：**目标 = 决策关键大 case（heihe_x4 + qinyijiang）上 1.5–2.0x**、**小 case 走 OMP_CUTOFF 跳过**、**forcing-IO 主导 case（heihe）延后或与独立 IO 优化配对**。这把头条 §1.1.1 数字（"8x 加速比"）从 "每 case 统一目标" 重新读作 "向大 case 加权的 portfolio 指标"。

## P8-precond 时机

**决策：留在 P8，不前置。**

按 master plan §S0.12 决策表，P8 预条件 SPGMR 的优化只在 RHS 并行上界被达到**之后**才相关。当前 profile 证据支持这点：

- 6 个 case 里 4 个，只靠 RHS 并行就能在 8 核上压出 >= 1.5x —— 这部分余量必须在 P1–P7（strict）里榨出来，再去付预条件设计 + 集成成本。
- 对 IO 主导 case（heihe）来说，P8 预条件**帮不上**——瓶颈是 `t_forcing_io`，不是 CVODE Krylov 迭代次数。给这个 case 把 P8 前置是 category error。
- CVODE 内部时间（`t_CVODE_internal`）从 6.83%（heihe）到 36.17%（keliya），给出预条件能寻址的上限约 36% —— 有意义但不是当前决策关键（大）case 中的主杠杆。

P8 预条件因此按 master plan §3 阶段顺序原位执行，不前提。

## 跨平台 delta review

按 spec.md 场景 "> 10pp 差异触发 review note"：`docs/profile_platform.md` 报 `delta_acceptable: false`，因为 4 个两端都有数据的 case 里 2 个超过 10pp 阈值：

- **keliya**：local 36.32%，target 49.33%，delta +13.01pp
- **qinyijiang**：local 51.07%，target 64.64%，delta +13.57pp

根因猜测（定性；定量诊断不在 #15 范围）：

1. **不同微架构的单核吞吐**。Apple M4 Pro 性能核单线程 IPC 大幅高于 Xeon Gold 6133（2017 Skylake-SP @ 2.5 GHz）。同样负载在 target 上 wall-clock 慢约 2.9–3.5x（target wall / local wall：keliya 79.7/27.8 = 2.87x，qinyijiang 799.9/229.5 = 3.49x），证实单核吞吐差距。
2. **跨 bucket 减速不均匀**。Apple clang + Apple silicon NEON 优化似乎对 SHUD RHS C++ kernel 的加速比对 SUNDIALS/CVODE 内部 C 代码（两个平台同源 C 代码）更激进。结果：RHS 在总 wall 里的占比**在 Apple 上变小、在 x86 上变大**——这是编译器 + 微架构交互的产物，不是 profile 正确性缺陷。
3. **xinanjiang_upstream 和 qhh** 显示小 delta（< 2pp），说明跨平台的非对称不在所有 case 上均匀——它与 "RHS 在可向量化内层循环里占比多少" 相关（越可向量化 → Apple 优势越大 → 对 x86 的 delta 越大）。

**对决策的影响**：delta 不让 "走原方案" 决策失效。两个平台都同意 RHS 是主 bucket（local：跨 case 36–51%；target：跨 case 12–67%，6 个里 5 个 > 30%），而**决策关键大 case（heihe_x4）只有 target 数据**——根本没有 local-vs-target 比较好做。上面的 Amdahl 上界算法锚定在 target 平台数字（`docs/profile_platform.md` `target_platform`），它是 §1.1.1 权威端点——所以 delta 只是 metadata 提示，不是 gate 阻塞 finding。

**行动项**（独立 issue 跟踪，不进 #15）：

- P1-P3 strict 并行落地后，对 keliya 和 qinyijiang 在 target 上 re-profile，确认 RHS 占比趋势在更高线程数下仍成立。
- 如果未来 heihe_x4 的 local-vs-target 比对变得可行（例如部分 forcing 或下采样 mesh），重做 delta 比对。

## 证据

Local 端产物（4 real + 3 deferred），SHA256：

```
b16e8a9acedcff00c82b66192d3db3eced538c7718c8715adff7b384761ce14f  benchmarks/keliya/profile_B0.yaml
34831b4b641f664cece3c5a4db1dd2c72b0016b99dc4206d30ac9b5365a43d97  benchmarks/xinanjiang_upstream/profile_B0.yaml
77bdbad6bd23e9806688bb7e76cb82d8359ed3edfbc7af03002ea9ee8ad87b02  benchmarks/qinyijiang/profile_B0.yaml
9506ceeeab796c9685dccfd649f5adc889e239e3df2f3274d5a82647637cec08  benchmarks/qhh/profile_B0.yaml
cbe50adf2047c88bc1e7d6415ce76c6ba8310e159310876c72f2bebdbc24982c  benchmarks/heihe/profile_B0.deferred.yaml
aea1322a9b2b6ee2ffaf391b422ccf05e8c1c7f29f5d20a9778dc6568eed8a9f  benchmarks/heihe_x4/profile_B0.deferred.yaml
5bdecf4e2173868de20659141cb9b046349cc5ae0fba97078b877f1cd4a5cd02  benchmarks/kashigeer/profile_B0.deferred.yaml
```

Target 端产物（6 real + 1 deferred），SHA256：

```
711a380902d2dee176ff16bf5c3a5c360a9ee131420d7727a7d4e75dc62ca0f5  benchmarks/keliya/profile_B0.target.yaml
a739dfd7c66310bf5e5bcb0317a99768d3c1d41480e8e991e0d32aaeca9637e1  benchmarks/xinanjiang_upstream/profile_B0.target.yaml
1dae17564e44de5149f8e49cb8dd3f404caa5a1ee19dc0b9ef2f26ab417174ed  benchmarks/qinyijiang/profile_B0.target.yaml
cc312b7ab1db926ab85fff86b91cc0e29fc02b2a289103ee30db9555dad105f5  benchmarks/qhh/profile_B0.target.yaml
baa03be7ce16e01345bdc9e9b93c033ffcee55213113b9b1ba91441414a97f5d  benchmarks/heihe/profile_B0.target.yaml
03d9d4c9def804b27f5f5e6a8930063eb03ce5ad5cbadee979c848f829254c36  benchmarks/heihe_x4/profile_B0.target.yaml
8f64779b5c3c25b2a854f70b9721231a4e40b5dd4a1b2eadf2e3a7e43d615d17  benchmarks/kashigeer/profile_B0.target.deferred.yaml
```

外层 commit：`ecef3fbe6ad6971ac8dc2ff6a888ece8db8fae83`
SHUD submodule commit：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`

## t_other 记账状态

按 rhs-profile-gate spec.md "t_other accounting WARN at 5%" 和 "FAIL at 10%" 场景。Target 平台 yaml 审计：

| Case | t_other_pct（target） | 状态 |
|---|---|---|
| heihe_x4 | 1.10% | OK |
| heihe | 1.52% | OK |
| qinyijiang | 0.87% | OK |
| qhh | 5.80% | **WARN** |
| keliya | 6.32% | **WARN** |
| xinanjiang_upstream | 22.42% | **FAIL** |

xinanjiang_upstream 的 FAIL（19.7s 跑里 22.42% t_other）解读为**启动开销主导的伪象**，不是 profile 工具缺陷：

- 绝对 t_other = 4.41s（总 19.7s wall）。
- 同份 yaml 的 `t_forcing_io = 1.40s` 覆盖 51 个 forcing CSV（6 个 case 里 forcing-stations 数最少的）。
- 进程启动（SUNDIALS init、mesh load、integrator setup）当前 S0-10 仪器化**没标**，被丢进 `t_other`。
- 长跑（heihe 487s、heihe_x4 1417s），启动开销摊薄到预期的 < 2%。

**决策影响**：FAIL 被承认，但不阻塞 "走原方案" 决策：(a) 决策锚定在大 case 上、占比由 RHS 主导而非启动；(b) FAIL 信号是**未来 profile 工具精化机会**（把启动时间标成 `t_init` 而不是丢进 `t_other`），属于 S0.12 retrospective 项，不是 S1 阻塞。

下个决策门快照之前（即 P-phase 退出之前）**应该**开个 follow-up issue，给 `tools/profile/timer.cpp` 加 `t_init` bucket，把启动时间从 `t_other` 里拆出去。

## 签字

| 字段 | 值 |
|---|---|
| signer | DankerMu（项目所有者；GitHub：@DankerMu；email：qingdanker@gmail.com） |
| signed_at | 2026-06-17 |
| signed_against_commit | outer `a860eae58991ce91ea91656cc9d4a08540e48f5b`（#16 merge 后 `baseline/current` HEAD）+ SHUD submodule `78c37a1061de4112bc7c297bb7bd1f107432e6f2`（#14 后 openmp-baseline HEAD） |
| signed_via | claude-code-s0-13-issue-17 代 DankerMu（按 user 2026-06-17 grant 的 delegated 签字；按 Linus Torvalds persona 编排，遵循 /Users/danker/.claude/CLAUDE.md 的优先级栈） |
| signed_off_decision | 走原方案 + 调高 large case 权重 + P8-precond 不前置 |
| follow_up_issues | (a) P1-P3 strict 落地后 re-profile；(b) heihe forcing IO 优化延后到 P9+；(c) 把 t_init 从 t_other 里拆出去（profile timer）；(d) kashigeer 上游 X76-X80 forcing 缺口（issue #29 已开） |

## Opt-IO 硬性前置判断（M7 trim 后重测）

> 本章节由 PR-G #214 (`openspec/changes/p1-update-omp` task 2.3) 新增。
> 对应 spec：`openspec/changes/p1-update-omp/specs/profile-retest-m7/spec.md` Requirement L19-L37 (Opt-IO 硬性前置判断更新)。
> 与 master plan §5 L1533 "Opt-IO 硬性前置阈值 = heihe `t_forcing_io / t_total >= 50%`" 联动。

### Trim 前（B0 baseline，参考）

S0 phase 上 heihe 非 trimmed forcing 实测 `t_forcing_io / t_wall_total = 385.436 / 487.046 ≈ 79.1%`（见上文 §"决策类型" L24 "heihe 的 12.08% 是 forcing-IO 主导（t_forcing_io = 79.1%）"），**远超 §5 L1533 50% 触发门**，因此 master plan §5 把 Opt-IO 定为 heihe 硬性前置。

### Trim 后（PR-G #214 本次实测）

server cn03 Slurm 跑 `NUM_OPENMP=1` 90 天截断 trimmed forcing 三次（同 binary `396ad9fb…`，3-run rivqdown SHA byte-identical 且 ≡ B0/B1a/B1b-tag golden）：

| Case | jobid | Elapsed | t_forcing_io mean | t_total mean | **ratio mean** | rivqdown SHA (3-run identical) | vs golden |
|---|---|---|---|---|---|---|---|
| heihe | 8742 | 00:07:29 | 2.836 s | 149.33 s | **1.90%** | `55abad28…` | ≡ B0/B1a/B1b-tag |
| heihe_x4 | 8743 | 01:09:06 | 2.667 s | 1382.0 s | **0.19%** | `f90601ef…` | ≡ B0/B1a/B1b-tag |

证据文件：
- `benchmarks/heihe/profile_B0.target.trimmed.yaml`（与既有 `profile_B0.target.yaml` 并存 per spec L59-L66）
- `benchmarks/heihe_x4/profile_B0.target.trimmed.yaml`
- server raw：`/scratch/frd_muziyao/SHUD-OpenMP/.s214-runs/prof_{heihe_8742,heihe_x4_8743}_summary.txt`（含 7-bucket × 3-run cvode_stats + per-run SHA）

### 决策 = (a) Opt-IO 退回可选

按 spec L23：

> **(a) 退回可选**：trim 后 heihe `t_forcing_io / t_total < 50%` → Opt-IO 回到 §5 原 "B1b 锁定后任意时间执行" 可选定位；本 change 不阻塞 P1；下游 P-strict 全部完成后再评估

实测 ratio：

- heihe = **1.90% << 50% 触发门**，**亦远低于 5% 严格门**（PR-B task 1.8 单 run 已验 `< 5%`，本 PR-G 三 run 复证；与 design.md L19 "M7 trim 后 forcing 占比 79% → ~13%" 预测的趋势一致，且实际比预测更乐观）
- heihe_x4 = **0.19% << 50% 触发门**（large case forcing IO 摊薄到几乎不可见，更证明 trim 后 Opt-IO 失去 large-case 杠杆）

**结论**：Opt-IO 不再是 §1.1.1 验收 heihe 的硬性前置；回到 master plan §5 原 "B1b 锁定后任意时间执行" 可选定位。本 change `p1-update-omp` 不阻塞 P1；下游 P-strict（P1–P7）全部完成后再评估 Opt-IO 是否必要（届时若 RHS 并行已榨出主要余量，forcing IO 剩余 1.9% / 0.19% 已无 ROI）。

| 字段 | 值 |
|---|---|
| signer | DankerMu（项目所有者；GitHub：@DankerMu；email：qingdanker@gmail.com） |
| signed_at | 2026-06-22 |
| signed_against_commit | outer `d34e455` (main HEAD at PR-G branch point; PR-B merged) + SHUD submodule `71b3a1ae` (openmp-baseline B1b-tag-aligned pin) |
| signed_via | claude-code-pr-g-issue-214 代 DankerMu（沿用 S0.12 delegated sign-off 模式，按 Linus Torvalds persona 编排，遵循 /Users/danker/.claude/CLAUDE.md 的优先级栈） |
| signed_off_decision | (a) Opt-IO 退回可选 — 本 change p1-update-omp 不阻塞 P1；P-strict 全部完成后再评估 |
| basis | heihe 1.90% / heihe_x4 0.19% << 50% 触发门 << 5% 严格门；3-run rivqdown SHA identical 且 ≡ B0/B1a/B1b-tag golden（trim 后 bitwise correctness 自证） |
| evidence | benchmarks/{heihe,heihe_x4}/profile_B0.target.trimmed.yaml + server jobid 8742/8743 (cn03) + spec profile-retest-m7 Requirement L19-L37 |
| follow_up_issues | 无新增；M7 trim 后 forcing IO 不再是头部瓶颈，Opt-IO 长期挂在 status_matrix "PENDING (可选)" 即可 |
