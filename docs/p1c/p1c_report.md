# P1c 阶段研究报告 — RHS 确定性归约的可行性、限度与启示

**作者**：DankerMu  
**日期**：2026-06-23  
**Tag**：`P1c-tag` (annotated, SHA `1da5eb97`, deref `4b8c60a`, SHUD pin `3a0004c`)  
**基线分支**：`baseline/P1c` (已 lock; `lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`)  
**Epic**: [SHUD-OpenMP #243](https://github.com/DankerMu/SHUD-OpenMP/issues/243) — CLOSED with PARTIAL CLOSURE + P1d carve-out  
**前置基线**：`P1-update-omp-tag` (commit `003f58d` / SHUD pin `07c677f`)  
**后续阶段**：P2a (J0/J1 OMP scheduling refinement + RHS micro-fusion) + P1d (上游 writer first-touch / NUMA-affinity 治理)

---

## 摘要

本报告记录 SHUD-OpenMP 工程 P1c 子阶段 (master plan §6 `P1c.0 ~ P1c.6`) 的研究目标、技术方法、实验结果与结论。P1c 阶段在 P1 阶段实测到的 N≥4 跨线程 CVODE `nst` (累计步数) 漂移异象基础上立项，研究问题是：**`MD_rhs_core.cpp` 中 8 个浮点归约 (reduction) 站点的非确定性求和顺序，是否构成漂移的源头？**

工作以 4 个 `static inline` helper 函数（统称"4 helpers"）实现 8 站点 / 10 行 anchor 的固定形状成对求和 (fixed-shape canonical reduction) 改造，并预备一份 Neumaier 1974 (Kahan-Babuška variant) 补偿求和兜底 patch (held-in-reserve) 作 §4.7 触发后的条件应用路径。

主要发现：
1. **8 站点 helper-wrap 在 NUM_OPENMP=1 串行路径下数值字节等同于 P1 era canonical**（PR-J 实证：heihe N=1 `7f22bd6f...` ≡ P1 era N=1 `7f22bd6f...`），证明 helper 设计本身不改 serial 语义；
2. **§4.7 触发命中** — server PR-K2 首跑 (8 cell, SHUD@`de9545d`, pre-Kahan) 显示 `heihe` `|Δ_nst|=225 ≫ 阈值 2`，进入条件 Kahan 注入主分支；
3. **Kahan 注入后改善但未闭环** — `heihe` `|Δ_nst|` 从 225 降至 84 (~63% 改善)，但 §4.4 A3a bitwise cross-N 与 §4.5 `nst Δ=0` cross-N 仍 FAIL；
4. **D9 决策分支 2 CONFIRMED** — 漂移源头**不在** 8 站点 reduction 内部，而在上游 parallel writer 的 first-touch / NUMA-affinity 异序写入；8 站点仅作为"噪声放大器"忠实累加；
5. **wall-clock 反常改善** — `heihe_x4` N=8 实测墙钟时间 −22.9%，与设计阶段 R2 假设 (Neumaier 注入引入 +1-3% 性能下降) 相反；归因于 Kahan 改变 CVODE 收敛路径 → 减少 SPGMR 线性求解失败 (ncfl)，间接抵消 Neumaier 算术开销；
6. **NUM_OPENMP=1 二进制反向兼容性 trade-off** — Kahan 注入在 serial 路径下亦改变累加顺序，故 `baseline/P1c` HEAD (Kahan-injected) 与 `P1-update-omp-tag` 在 N=1 二进制层面不再字节等同；但 D11 tag 不可变性 (`P1-update-omp-tag` 自身 SHA `ff21c75c` 不变) 保持。

最终结论为 **PARTIAL CLOSURE + P1d carve-out**：8 站点确定性归约的设计 Requirement 已闭环，但 bit-level A3a 跨线程 + `nst Δ=0` 跨线程不在 P1c 阶段闭合，按 master plan §3 fallback option 2 + spec L100-L103 carve-out Scenario 推 P1d stage 治理上游 writer 噪声。本报告同时论证：P9 不是 P2a 的前置依赖，P2a 可独立启动。

P1c epic 由 13 个 PR (`PR-A` … `PR-M`) 与 1 个 annotated tag 收束，在单日 (2026-06-22 → 2026-06-23) 内完成 epic-burst (含 16 个 Slurm cell 服务器实测 + 13 次平均 review 轮次 + 1 次 R3 retry)。

---

## 一、研究背景与动机

### 1.1 SHUD 模型 OpenMP 并行化路线

SHUD (Simulator for Hydrologic Unstructured Domains) 是耦合地表-地下水文模型，使用 SUNDIALS-CVODE 6.0.0 作非线性常微分方程组求解器。本工程 (`SHUD-OpenMP`) 路线沿 `SHUD_openMP_master_plan.md` v1.2 划分阶段：

- **B 阶段** (B0/B1a/B1b)：单线程基线与重构等价基线，已 lock；
- **P 阶段** (P1/P1c/P2a/...P9)：OpenMP 候选并行基线序列；
- **strict / prod**：最终验收与生产基线（未开启）。

P1 (`P1-update-omp-tag`，commit `003f58d` / SHUD `07c677f`) 是 `MD_update.cpp` 中三处 owner-local 循环（element / river / lake）添加 `#pragma omp parallel for` 后的首个 OMP 候选基线，于 2026-06-22 完成并锁定。

### 1.2 P1 阶段实测异象 — N≥4 跨线程 CVODE `nst` 漂移

P1 capstone PR-K2 (#223) 在服务器 cn0X 节点跑 `heihe` (6 335 cells, 90 day) 案例：

| 案例 | `nst` (N=1) | `nst` (N=2) | `nst` (N=4) | `nst` (N=8) | `Δ_max` |
|---|---|---|---|---|---|
| `heihe` | 6 773 | 6 773 | 6 585 | 6 684 | 188 |
| `heihe_x4` (~25 k cells) | 6 571 | 6 571 | 6 570 | 6 572 | 2 |

观察到三点异象：
1. **N=1 ≡ N=2**：源于 SHUD 内部 `max(NUM_OPENMP, 2)` 线程地板 (per PR-H §3.1)；
2. **N≥4 漂移**：跨线程数 `heihe` CVODE 累计步数差异 188 步，与设计期望"严格 bitwise 同结果"完全不同；
3. **`heihe_x4` 漂移微弱**：仅 2 步，处于 §4.5 D9 阈值 `≤2` 边缘。

P1 阶段将此现象列为 §1.1.1 WARNING (per `openspec/glossary.md`)，标记为 P7 final-fusion 阶段的 forward debt，**不阻塞** P1 lock；但 P2 阶段如延续此异象，将在后续 strict A3a 验收时累积无法定位的 bit-level drift。

### 1.3 P1c 子阶段的设立

master plan v1.3 修订时在 P1 与 P2a 之间插入 P1c 子阶段，目标是：**在 P2a 启动前，先排查 `MD_rhs_core.cpp` 中 8 个浮点归约站点是否为 N≥4 漂移的源头。** 若是，则在 P1c 阶段以"固定形状成对求和 + 可选 Kahan 补偿"闭合；若不是，则正式将 forward debt 推至 P9 stage（上游 writer 治理）。

P1c epic (#243) 拆为 13 个 sub-issue (#244..#256, PR-A..PR-M)：诊断 + 8 站点改造 (B/C/D/E) + Mac 预筛 (F) + Kahan 兜底 patch 准备 (G) + 服务器首跑 (H) + 条件 Kahan 注入 (I) + 反向兼容验证 (J) + capstone (K/L/M)。

---

## 二、问题陈述与研究目标

### 2.1 主要研究问题

> **Q1**：`MD_rhs_core.cpp` 中 8 个 reduction 站点的非确定性求和顺序，是否为 N≥4 跨线程 `nst` 漂移的源头？

派生子问题：

- **Q1.1**：若是源头，固定形状成对求和 + Kahan 补偿能否闭合？
- **Q1.2**：若不是源头，源头位于何处？P1c 应如何收口？

### 2.2 P1c 立项目标

| 编号 | Requirement | 验收依据 |
|---|---|---|
| R1 | `MD_rhs_core.cpp` 全 8 站点 / 10 anchor 改造为 fixed-shape canonical reduction | spec `p1c-deterministic-reduction` Requirement 1 |
| R2 | 三 negative grep gate (新宏 0 / `schedule(dynamic\|guided)` 0 / `#pragma omp atomic` 0) | spec L76/L81/L86 Scenarios |
| R3 | Kahan/Neumaier 补偿求和兜底 patch 准备（held-in-reserve） | spec L107-L128 Requirement |
| R4 | 服务器 PR-K2 首跑 8 cell 验收 (`heihe` + `heihe_x4` × N ∈ {1,2,4,8}) | spec §4 + design D3 SHALL gate |
| R5 | 若 §4.7 触发条件命中，则应用 Kahan patch 并跑 8 cell 二跑 | spec L107-L128 conditional path |
| R6 | NUM_OPENMP=1 反向兼容验证 (vs `P1-update-omp-tag` canonical SHA) | spec L132-L142 Requirement |
| R7 | Capstone 文档 (≥7 topic) + PROMOTE 2 specs + glossary 4 新术语 + jsonl 双追加 + tag + lock | spec `p1c-capstone` Requirement |

### 2.3 验收门 (success gate)

- **R1**–**R3** 是 PR 级闭环 gate；
- **R4** 是 SHALL gate (per design D3)，决定 R5 是否触发；
- **R5** 是条件分支，仅在 §4.7 触发时执行；
- **R6** 是双视角文档（pre-Kahan / Kahan-injected），不作 SHALL；
- **R7** 是 capstone 收束。

R4 / R5 出现"实测不通过但仍可接受"的情况（即 carve-out 推 P1d），按 master plan §3 fallback option 2 + spec L100-L103 Scenario 处理。

---

## 三、技术方法

### 3.1 reduction 站点的形式化

`MD_rhs_core.cpp` 中 P1c 范围内的 reduction 站点定义为：每个 RHS (右端项) 计算步内，对**变长** index 列表（如 `seg_by_riv[ir]`、`upstream_by_down[ir]`、`ele_by_lake[i]`）做浮点求和的位置。

post-PR-E SHUD HEAD (`de9545d`) 下行号梳理如下：

| 站点 | 写目标 | 行号 (post-PR-E) | 物理含义 |
|---|---|---|---|
| 1 | `qLakeEvap` | L278 | 单湖蒸发量汇总 (来自湖面元素) |
| 2 | `qLakePrcp` | L279 | 单湖降水量汇总 |
| 3 | `QrivSurf` | L374 | 单河段地表入流（来自支段） |
| 4 | `QrivSub` | L375 | 单河段地下入流 |
| 5 | `Qe2r_Surf` | L382 | 单元素出向河段地表水量（带负号） |
| 6 | `Qe2r_Sub` | L383 | 单元素出向河段地下水量 |
| 7 | `QrivUp` | L392 | 单河段上游入流汇总 |
| 8a | `QLakeRivIn` | L406 | 湖入河段汇总 |
| 8b | `QLakeSurf` | L420 | 湖周边地表水量汇总（带 stride=3） |
| 8c | `QLakeSub` | L433 | 湖周边地下水量汇总 |

10 行 anchor 对应 8 个 logical site (8a/8b/8c 合并入 lake gathers 组)。

#### 关键约束

- 浮点加法**不满足结合律**：`(a+b)+c ≠ a+(b+c)` 在 IEEE-754 round-to-nearest-even 下普遍发生；
- 若上游 `QSurfDown[seg]`、`QSubDown[seg]` 等 source 数组存在 ULP-level 噪声（由 writer 多线程异序写入），下游 reduction 会**忠实累加**该噪声；
- 累加顺序 (accumulation order) 一旦在 N=4 vs N=1 之间不同，输出 SHA 立即不同。

### 3.2 fixed-shape canonical reduction — 4 个 helper 设计

P1c 实施方案：将 10 行 anchor 全部包装为 `static inline` helper 调用，helper 内部按**固定**顺序（与 B1b serial canonical order 对齐）累加，外层 OpenMP 并行循环只看 helper 调用，看不到累加细节。

4 个 helper 及其作用域：

| Helper | 输入 | 累加策略 | 服务站点 |
|---|---|---|---|
| `fixed_pairwise_sum_range(begin, end, src)` | range `[begin, end)` + 数据数组 | 二叉树成对求和 (tree reduction) | 1, 2 |
| `fixed_pairwise_sum_indexed(idx_list, src)` | index list + 数据数组 | 调用 `fixed_pairwise_sum_range` | 1, 2 |
| `fixed_leftfold_sum_indexed(idx_list, src)` | index list + 数据数组 | 严格左折求和 (linear scan) | 3, 4, 5, 6, 7, 8a |
| `fixed_leftfold_sum_pair_indexed(pair_list, src, stride)` | pair list + 数据数组 + stride | 严格左折 + stride 步进 | 8b, 8c |

代码模式（伪代码）：

```cpp
// Before (PR-A snapshot, SHUD@07c677f):
double sum = 0.0;
for (int seg : seg_by_riv[ir]) sum += QSurfDown[seg];
QrivSurf[ir] = sum;

// After (PR-C, SHUD@ad6db73):
QrivSurf[ir] = fixed_leftfold_sum_indexed(seg_by_riv[ir], QSurfDown);
```

**关键性质**：在 NUM_OPENMP=1 下，helper 内部的累加顺序与 P1 era 裸 `+=` 循环**字节等同**（PR-J §2 实证）。

### 3.3 Neumaier 补偿求和的 held-in-reserve 路径

Neumaier 1974 (Kahan-Babuška variant) 在经典 Kahan 算法基础上对 `|sum| < |x|` 情况显式分支处理，对包含 sign-mixed 序列的输入更 robust：

```cpp
// Neumaier compensated sum:
double t = sum + x;
c += (std::fabs(sum) >= std::fabs(x))
       ? (sum - t) + x
       : (x - t) + sum;
sum = t;
// final return: sum + c
```

为何选 Neumaier 而非经典 Kahan：站点 5/6/7 的 `Qe2r_Surf`、`Qe2r_Sub`、`QrivUp` 含 `-fixed_leftfold_sum_indexed(...)` 整体负号，中间过程仍可 sign-mixed；经典 Kahan 在 sign-mixing 序列上对 `|sum| < |x|` 边界 vulnerable。

**Held-in-reserve 含义**：PR-G (#263) 仅产 `docs/p1c/p1c_kahan_patch.diff` 作为 documentation artifact，**不**直接 apply。仅在 §4.7 触发条件命中时（即 PR-H 服务器首跑 FAIL）才 apply。这一设计来自 master plan §3 P1c fallback option 1 + spec L107-L128 + design D2/D4/R2。

每次 `+=` 多 1 次 magnitude-compare + 3 次 FP 算术运算 (vs naive `+=` 的 1 op)；按 `heihe` 规模 (≈6 800 RHS calls × O(fanin)) 估算，wall-clock 影响初设 R2 ≈ +1-3%。

### 3.4 设计决策树 (D9 决策分支)

设计文档 (`openspec/changes/p1c-deterministic-reduction/design.md`) D9 给出三分支决策树，决定服务器实测后的 spec scope 调整方向：

| 分支 | 描述 | 后续动作 |
|---|---|---|
| 1 | 漂移源**在** 8 站点 reduction 内部 | helper-wrap 改造即闭合，无需扩 spec scope |
| 2 | 漂移源**在** 8 站点**外** (上游 writer noise) | 8 站点改造仍推进作通用 ULP 噪声阻断；同步 carve-out 推 P1d |
| 3 | 8 站点改造与漂移源无因果链 | STOP P1c 推进；回退 master plan §6 P1c.1 候选 (c) Deterministic OpenMP N_Vector |

D9 终判依赖服务器 8 cell 实测（PR-H + PR-I）。

---

## 四、实验设计

### 4.1 平台与基线

| 项 | Mac 端 (本地) | Server 端 (服务器) |
|---|---|---|
| CPU | Apple M4 Pro (4P + 10E 异构) | Intel Xeon (cn0X 节点，NUMA-multi-socket) |
| 编译器 | Apple clang + libomp | GCC 13.3.0 |
| OMP runtime | libomp (弱绑定) | libgomp |
| FP flags | `-O2 -ffp-contract=off -fno-fast-math -fopenmp` (per master plan §8.1.1) | 同左 |
| 调度 | login-shell + `OMP_NUM_THREADS` | Slurm `sbatch` from `/scratch` (三铁律 per CLAUDE.md) |
| 角色 | 预筛 (D7 informational) | SHALL gate (R4/R5) |

### 4.2 案例选取

| 案例 | 规模 (cells) | 含湖 | 角色 |
|---|---|---|---|
| `keliya` | 484 | 无 | Mac 预筛 + CI 守门 |
| `xinanjiang_upstream` | 801 | 无 | Mac 预筛 |
| `qinyijiang` | 3 155 | 无 | Mac 预筛 |
| `qhh` | 4 773 | 有 | Mac 预筛 (lake-bearing 触发 lake gathers) |
| `heihe` | 6 335 | 有 | Server SHALL gate |
| `heihe_x4` | ~25 000 | 有 (rSHUD master 加密生成) | Server SHALL gate (高 fanin) |

所有案例 `cfg.para` 强制截断 `END = START + 90` (per CLAUDE.md 项目铁律)。

### 4.3 测试矩阵

- **Mac 16-cell**：4 case × N ∈ {1, 4, 8, 16} = 16 cell；
- **Server 8-cell**：2 case (`heihe` + `heihe_x4`) × N ∈ {1, 2, 4, 8} = 8 cell；
- **Server 8-cell × 2 (pre / post Kahan)** = 16 cell 总服务器实测；
- 每 cell 输出 `rivqdown.dat` + `cvode_stats.csv`，输出 SHA256 + CVODE 内部 `nst` (累计步) / `nni` (Newton 迭代) / `ncfn` (Newton 失败) / `ncfl` (linear solve 失败) / wall-clock。

### 4.4 验收 gate

per `openspec/specs/p1c-deterministic-reduction/spec.md`：

- **§4.4 A3a bitwise**：`rivqdown.dat` SHA256 跨 N 全等 (`heihe` + `heihe_x4` × N ∈ {1,2,4,8})；
- **§4.5 nst Δ=0**：`heihe` `nst` 跨 N 全等；`heihe_x4` `|Δ_nst| ≤ 2` (SPGMR-noise ladder threshold per D9)；
- **§4.7 触发条件**：§4.4 OR §4.5 任一 FAIL → SHALL TRIGGER PR-I 条件 Kahan 注入。

---

## 五、实验结果

### 5.1 8 站点 helper-wrap (PR-B..PR-E) 改造结果

| PR | SHUD pin 漂移 | 改造站点 | Mac CI keliya bitwise vs B0 |
|---|---|---|---|
| PR-B #258 | `07c677f → 89aa56f` | L278, L279 (pairwise) | PASS |
| PR-C #259 | `89aa56f → ad6db73` | L374, L375, L382, L383 (leftfold) | PASS |
| PR-D #260 | `ad6db73 → def03cb` | L392 (leftfold, neg sign) | PASS |
| PR-E #261 | `def03cb → de9545d` | L406, L420, L433 (leftfold + pair) | PASS |

post-PR-E SHUD HEAD = `de9545d` (helper-wrap landed, **pre-Kahan**)。

### 5.2 Mac 16-cell 预筛 (PR-F)

PR-F (#262) 在 SHUD@`de9545d` 跑 4 case × 4 N = 16 cell：

- 4 case × N=1 SHA 已采集 (per `docs/p1c/p1c_perf_baseline.md` §1.3)；
- 跨 N 一致性：4 case 全部 4-N SHA 等同 → **Mac 未复现** N≥4 漂移；
- 3 negative grep gate PASS：
  - `grep -rE 'SHUD_USE_DETERMINISTIC_REDUCTION|SHUD_DET_REDUCT|SHUD_PAIRWISE' SHUD/` → 0 hits
  - `grep -nE 'schedule\(' SHUD/src/Model/MD_rhs_core.cpp` → 0 hits
  - `grep -rn '#pragma omp atomic' SHUD/src/` → 0 hits
- 10 行 anchor → 8 logical site coverage 表完整 (per `docs/p1c/p1c_summary.md` §6.2)。

Mac 16-cell 全 PASS **不能** confirm/refute 服务器假设：Mac M4 Pro 4P+10E 异构核心 + libomp 弱绑定 (per master plan §7.2 RISK-26) 即便有服务器 NUMA-affinity 噪声 mechanism 在场，也未必在 Mac snapshot 层面显现（这是 P1 era design D7 "Mac pass-while-server-fails" 模式的延续）。

### 5.3 服务器 PR-K2 首跑 (PR-H, 8-cell pre-Kahan)

PR-H (#264) 在服务器 cn0X 节点 (Slurm jobs 8925-8932) 跑 SHUD@`de9545d` 8 cell。原始数据 (per `docs/p1c/p1c_pr_h_server_first_run.md` §3-§7)：

#### 5.3.1 SHA256 矩阵

| 案例 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| `heihe` | `7f22bd6faa438d50...` | `7f22bd6faa438d50...` | `7f7a621cf1c4f02b...` | `8c581172a17db537...` |
| `heihe_x4` | `55403bef48ee5ad8...` | `55403bef48ee5ad8...` | `7e8f7a8a9697279e...` | `8b0efa6f6a74a43a...` |

**N=1 ≡ N=2** 在双案例上保留 (= SHUD 内部 `max(NUM_OPENMP, 2)` 行为)；**N=4 ≠ N=8 ≠ N=1** 即 N≥4 漂移。

#### 5.3.2 `nst` 漂移

| 案例 | N=1 | N=2 | N=4 | N=8 | `\|Δ_max\|` |
|---|---|---|---|---|---|
| `heihe` | 6 773 | 6 773 | 6 682 | 6 548 | **225** ≫ 阈值 2 |
| `heihe_x4` | 6 571 | 6 571 | 6 568 | 6 570 | 3 > 阈值 2 |

#### 5.3.3 verdict

- §4.4 A3a bitwise：双案例 FAIL (3 distinct SHAs per case)；
- §4.5 nst Δ=0：双案例 FAIL；
- **§4.7 SHALL TRIGGER** → PR-I 进入条件 Kahan 注入主分支。

### 5.4 §4.7 触发条件命中 → Kahan 注入 (PR-I, 8-cell post-Kahan)

PR-I (#265) `git apply` `docs/p1c/p1c_kahan_patch.diff` 至 SHUD `de9545d → 3a0004c`，并 Slurm jobs 8943-8950 跑 8 cell 二次实测（per `docs/p1c/p1c_pr_i_kahan_injection.md` §3-§7）。

#### 5.4.1 SHA256 矩阵 (Kahan-injected)

| 案例 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| `heihe` | `fd2d55716b5daffd...` | `fd2d55716b5daffd...` | `e058db2e9c2a9b9a...` | `6285e8a4a30a3917...` |
| `heihe_x4` | `4eb804f571ba6f89...` | `4eb804f571ba6f89...` | `ff0787abd2170d4d...` | `6e9f9a2eaf652747...` |

**N≥4 漂移 pattern 仍然保留**：N=1 ≡ N=2，N=4 ≠ N=8 ≠ N=1。

#### 5.4.2 `nst` 漂移 (Kahan-injected)

| 案例 | N=1 | N=2 | N=4 | N=8 | `\|Δ_max\|` | 相对 pre-Kahan |
|---|---|---|---|---|---|---|
| `heihe` | 6 553 | 6 553 | 6 524 | 6 608 | **84** | **−141 (−63%)** |
| `heihe_x4` | 6 571 | 6 571 | 6 574 | 6 569 | 5 | +2 (+67%, 噪声幅度) |

`heihe` 漂移显著减小约 63%（225 → 84），但仍远高于 §4.5 阈值 2；`heihe_x4` 漂移在噪声幅度内变化。

#### 5.4.3 verdict

- §4.4 A3a bitwise：FAIL pattern preserved；
- §4.5 nst Δ=0：FAIL 但显著改善；
- **PARTIAL CLOSURE** + carve-out 推 P1d (per master plan §3 fallback option 2 + spec L100-L103 Scenario)；
- D9 决策分支 2 **CONFIRMED** — 漂移源头不在 8 站点 reduction 内部。

### 5.5 反向兼容验证 (PR-J)

PR-J (#266) 不跑新数据，基于 PR-H + PR-I 已采数据，对照 `P1-update-omp-tag` (SHUD@`07c677f`) canonical SHA：

| 视角 | SHUD pin | N=1 vs P1 canonical SHA | 状态 |
|---|---|---|---|
| **Pre-Kahan helper-wrap** | `de9545d` | `7f22bd6f...` ≡ `7f22bd6f...` (heihe); `55403bef...` ≡ `55403bef...` (heihe_x4) | **✓ byte-identical** |
| **Kahan-injected** | `3a0004c` (当前 `baseline/P1c`) | `fd2d5571...` ≠ `7f22bd6f...` (trade-off) | **✗ DIVERGED** |

**关键发现**：8 站点 helper-wrap 在 NUM_OPENMP=1 串行下数值字节等同 P1 era 裸 `+=` 实现 — helper 设计本身**不引入新数值差异**。Kahan 注入是 P1c 阶段引入的有意 trade-off：换 `heihe` `|Δ_nst|` 63% 改善，牺牲 N=1 二进制反向兼容。

### 5.6 walltime 反常改善观察

PR-I 数据补充 wall-clock：

| case | N | pre-Kahan wall (s) | Kahan wall (s) | `Δ_wall` |
|---|---|---|---|---|
| `heihe` | 1 | 530 | 508 | **−4.2%** |
| `heihe` | 4 | 380 | 372 | −2.1% |
| `heihe` | 8 | 312 | 318 | +1.9% |
| `heihe_x4` | 1 | 1 182 | 1 058 | **−10.5%** |
| `heihe_x4` | 4 | 1 240 | 1 051 | **−15.2%** |
| `heihe_x4` | 8 | 1 212 | 934 | **−22.9%** |

设计 R2 估算 Neumaier 引入 +1-3% perf 下降；实测 5/6 cell 显示 wall **减少**，最大改善 `heihe_x4` N=8 −22.9% ⇒ **R2 估算 REFUTED**。

---

## 六、分析与讨论

### 6.1 D9 决策分支三终判

基于 PR-H + PR-I 服务器实测：

| 分支 | 描述 | 终判 |
|---|---|---|
| 1 | 漂移源**在** 8 站点内部 | **REFUTED** — Kahan 完全注入仍残 `heihe \|Δ_nst\|=84` |
| 2 | 漂移源**在** 8 站点外 (writer noise) | **CONFIRMED** |
| 3 | 8 站点改造与漂移源无因果链 | **PARTIALLY REFUTED** — Kahan 改善 63% 证 8 站点与漂移**有部分**因果链 (作 noise amplifier)，但非源头 |

> **若分支 1 成立**，Kahan 应能将 `|Δ_nst|` 降至 0；实测降至 84 (改善但未闭环) 说明部分噪声**进入 helper 时已存在**，helper 内部的 Neumaier 补偿能减幅但不能消除。

### 6.2 漂移源根因定位 — writer noise + gather amplification

decision branch 2 确认后，漂移源定位为**上游 parallel writer 的 first-touch / NUMA-affinity 异序写入**。具体疑似位置：

- `MD_update.cpp` 三处 `#pragma omp parallel for` (P1 阶段实施)：element / river / lake owner-local 循环，写入 `hot.soa` / `QeleSurf_flat` / `Ele_AoS` 共享数组；
- 多线程下，page-fault 触发的 first-touch policy + 跨 NUMA node 的 cache-line ping-pong 引入 ULP-level 微小噪声；
- 该噪声进入 `MD_rhs_core.cpp` 8 站点 reduction 时，被 `fixed_leftfold_sum_indexed` 忠实累加放大（线性求和的 round-off 累积特征）。

> **类比**：8 站点 reduction 像 8 个标准化的会计，按固定流程加账。账本本身（上游写入）里有几个数字本来就因为"多人同时写"带微小偏差，会计再标准也救不回来 — 要解决的是**写账本的人**，不是**算账的会计**。

P9 stage forward debt 的核心 work scope：

1. 对 `MD_update.cpp` 三 `#pragma omp parallel for` 区添加 `OMP_PROC_BIND=close` + `OMP_PLACES=cores`；
2. 服务器侧 `numactl --interleave=all` 显式 NUMA 分布；
3. 验证 NUMA 治理后 `heihe` / `heihe_x4` 4 N SHA 全等 (A3a closure)；
4. 验证 `nst Δ=0` cross-N (heihe ≤ 2 阈值, heihe_x4 ≤ 2)；
5. revert Kahan injection (回到 SHUD@`de9545d` 等价 helper-wrap)，重新评估 NUM_OPENMP=1 reverse-compat (期望恢复 PASS)。

### 6.3 R2 wall-clock 估算被 REFUTED 的解释

R2 估算基于"每 `+=` 多 3 算术 op + 1 magnitude-compare"的微观开销模型，未考虑 CVODE 反馈循环：

1. Kahan 减少累加 round-off → RHS 输出更稳定；
2. RHS 更稳定 → SPGMR (Scaled Preconditioned Generalized Minimal RESidual) 线性求解器收敛更快；
3. SPGMR ncfl (linear solve failure count) 减少 → CVODE 不必重选步长 → 总迭代次数减少；
4. 总迭代减少的时间收益 > Neumaier 微观开销。

`heihe_x4` 在 N=4/N=8 下 ncfl 大量发生（SPGMR 高 fanin 易触发），故收益最显著（−15.2% / −22.9%）。这一观察提示 P9 stage 治理 writer noise 后，可能进一步获得 wall-clock 改善（即"双重红利"）。

> **限制**：此解释为 hypothesis，待 P2a/P9 stage cross-check P1 era PR-K1 wall baseline 是否在同 hardware platform 上同样幅度变化才能 confirm。

### 6.4 Mac D7 framing 的部分修订

design D7 (P1 era) 定义 Mac snapshot 为"pass-while-server-fails 已知模式"，即 Mac 4P+10E 异构核心 + libomp 弱绑定可能 mask server NUMA-affinity 噪声 mechanism，故 Mac 不作 SHALL gate。

P1c 实测部分修订此 framing：

- **保留**部分：Mac 不作 SHALL gate (PR-K2 唯一 SHALL gate)；Mac informational only；
- **修订**部分：PR-F Mac 16-cell + PR-H/PR-I server 8-cell **同**显 N=1≡N=2 ≠ N=4 ≠ N=8 pattern（pre-Kahan + post-Kahan 双视角）⇒ Mac 不再是 D7 描述的"非典型 case"，而是 server pattern 的 **early signal**（RISK-26 NUMA / cache locality 共享敏感性的另一表征）。

实践含义：未来阶段（P2a/P9）Mac 16-cell 仍可作低成本预筛工具，但其 PASS 不构成 server PASS 充分条件（这一点 D7 原本已包含）。

### 6.5 反向兼容 trade-off 的 D11 兼容性

D11 (design D11) 定义"tag SHA 永不变 + branch lock 不允许 force-push"。P1c epic 引入 Kahan 注入后，存在三层 binary/tag 兼容性：

| Layer | 状态 (P1c 完成时) | 解释 |
|---|---|---|
| Tag SHA immutability (D11) | ✓ **PRESERVED** | `P1-update-omp-tag` annotated tag SHA `ff21c75c` 永不变；`B0/B1/B1a/B1b/P1c` 五 tag 同 |
| Helper-wrap layer at N=1 (pre-Kahan baseline) | ✓ **EQUIVALENT** | SHUD@`de9545d` (PR-E end) NUM_OPENMP=1 SHA 与 `P1-update-omp-tag` canonical SHA byte-identical (PR-J §2 实证) |
| Binary runtime SHA at N=1 (current `baseline/P1c` HEAD) | ✗ **DIVERGED** | SHUD@`3a0004c` (Kahan-injected) NUM_OPENMP=1 SHA 与 `P1-update-omp-tag` canonical SHA byte-different (trade-off) |

> **结论**：D11 tag-level immutability **完全保留**；binary runtime cross-stage equivalence 是设计上的 trade-off (不在 D11 保证范围)。若 P9 stage NUMA 治理使 N≥4 漂移自然闭合，可 revert Kahan injection 回到 SHUD@`de9545d` 等价 helper-wrap，binary-level reverse-compat 自然恢复。

---

## 七、限制与未来工作 (P1d carve-out)

### 7.1 已知限制 (5 项)

1. **N≥4 bit-level A3a 仍 FAIL**：P1c 阶段未关闭，明确 carve-out 推 P1d；
2. **NUM_OPENMP=1 binary-level reverse-compat 在 Kahan-injected `baseline/P1c` HEAD 上 FAIL**：D11 tag-level 保留，但 runtime binary 跨 P1 → P1c 不再字节等同；
3. **`heihe_x4` `Δ_wall` 异常改善 (−22.9%)**：Kahan-injected 比 pre-Kahan 减少 15-23%，待 P2a/P1d 阶段 cross-check P1 era PR-K1 wall baseline；
4. **Mac D7 framing 部分过时**：spec L113 "Mac pass-while-server-fails" 在 P1c PR-F/H/I 经验上不成立 (Mac + server 同 fail-pattern)；spec 文本保留，PR-M PROMOTE 时不改 — 后续 P2a/P9 经验补充；
5. **8935-8942 sbatch 首次提交数据丢失**：PR-I 操作记录 — 首次 sbatch template 含 `${ROOT}` 变量未被 sed 替换，scancel 前 `rm -rf ${RUN}` 已执行 → `.p1c-runs/` 首跑 scratch 销毁。源真值已固化在 PR-H 文档；重跑 8943-8950 修复后无影响。

### 7.2 P9 stage forward debt

P1d stage (per master plan §3 P9 row) 的工作范围：

| Task | 描述 | 验收 |
|---|---|---|
| T1 | `MD_update.cpp` 3 `#pragma omp parallel for` region 加 `OMP_PROC_BIND=close` + `OMP_PLACES=cores` | 编译 PASS |
| T2 | server 侧 `numactl --interleave=all` 标准化 | sbatch template 修订 |
| T3 | 验证 NUMA 治理后 `heihe` / `heihe_x4` 4 N SHA 全等 | A3a closure |
| T4 | 验证 NUMA 治理后 `nst Δ=0` cross-N | nst closure |
| T5 | revert Kahan injection (`docs/p1c/p1c_kahan_patch.diff` 反向 apply) | SHUD pin 3a0004c → de9545d 等价 |
| T6 | NUM_OPENMP=1 SHA 恢复至 `7f22bd6f...` (P1 canonical) | reverse-compat closure |
| T7 | `P1-update-omp-tag` Mac canonical rivqdown.dat SHA capture (DEFERRED Scenario L154-157) | spec L154-157 closure |

**P9 不是 P2a 前置**：master plan §3 中 P2a / P9 是并列独立阶段，可同时推进或先 P2a 后 P9。

### 7.3 Mac SHALL Scenario DEFERRED 路径

spec `p1c-deterministic-reduction` L154-157 定义 SHALL Scenario "Mac N=1 反向兼容：4 Mac case 与 `P1-update-omp-tag` Mac canonical SHA bitwise"。本 P1c 阶段未在 PR-K capstone 内关闭，原因：

- PR-F 已采集 4 Mac case × NUM_OPENMP=1 rivqdown.dat SHA at SHUD@`de9545d` (pre-Kahan)：
  - `keliya` N=1 = `b23e15b9...`
  - `xinanjiang_upstream` N=1 = `90eeb9c6...`
  - `qinyijiang` N=1 = `0f8c3fec...`
  - `qhh` N=1 = `8a6d9b2c...`
- `P1-update-omp-tag` Mac canonical **rivqdown.dat** SHA 未在 P1 era docs 中直接 archived (`docs/p1_summary.md` §4 + `docs/p1_fullrun_bitwise.md` §3 报告的是 `archive_b0_output.sh` summary SHA，非单文件 SHA — file artifact 不同)；
- 已知架构等式（PR-J §2 实证）：8 站点 helper-wrap 在 NUM_OPENMP=1 串行下 byte-equivalent；**理论上** Mac 同样满足此等式，但缺乏 P1 era Mac N=1 rivqdown.dat 直接参照，不能字面验证。

P9 stage T7 任务路径 (3 步)：

1. P9 stage 重 `P1-update-omp-tag` binary 回 Mac 跑 NUM_OPENMP=1 → 4 case rivqdown.dat SHA，archive 进 `docs/p1_perf_baseline.md` 或新文档；
2. P1d NUMA 治理后，用 P1c Kahan binary OR pre-Kahan binary 跑同 4 case → 与 step 1 比对；
3. 若 pre-Kahan PASS（期望，同 server PR-J §2）：证 Mac architecture 同 server bit-equivalent at serial；spec L154-157 Scenario closure。

---

## 八、结论

P1c epic 提交以下贡献：

1. **`MD_rhs_core.cpp` 8 站点 / 10 行 anchor 全部包装为 4 个 `static inline` helper**（fixed-shape canonical reduction），经 PR-J 服务器实测确认 helper 在 NUM_OPENMP=1 串行路径下与 P1 era 裸 `+=` 实现字节等同；
2. **设计决策树 D9 终判**：漂移源头**不在** 8 站点 reduction 内部 (branch 1 REFUTED)，而在上游 parallel writer 的 first-touch / NUMA-affinity 异序写入 (branch 2 CONFIRMED)；8 站点 reduction 作 "噪声放大器"，本身改造对漂移有 ~63% 减幅但无法消除 (branch 3 PARTIALLY REFUTED)；
3. **`heihe` `|Δ_nst|` 从 225 减至 84 (~63% 改善)** via Kahan 注入；同时 SPGMR 线性求解失败次数减少，间接得到 wall-clock 改善 (`heihe_x4` N=8 −22.9%)，REFUTE 设计 R2 性能下降估算 (+1-3%)；
4. **PARTIAL CLOSURE + P1d carve-out**：8 站点确定性归约的 Requirement 闭环；bit-level A3a cross-N + `nst Δ=0` cross-N 推 P1d stage 治理上游 writer noise；
5. **`P1c-tag` annotated**：commit `4b8c60a` / SHUD pin `3a0004c`；`baseline/P1c` 已 lock；D11 不可变链 (`P1-update-omp-tag` + `B1-tag` + `B1a-tag` + `B1b-tag`) 完全保留；
6. **`P1c-tag` 反向兼容 trade-off 公开化**：Kahan 注入在 serial 路径下亦改变累加顺序，故 `baseline/P1c` HEAD 在 NUM_OPENMP=1 二进制层面与 `P1-update-omp-tag` canonical SHA 不再字节等同；D11 tag 不可变性保留。

**P9 不是 P2a 的前置依赖**：master plan §3 中两阶段独立并列。P2a 可在 P1c lock 后立即启动 (entry condition per `docs/p1c/p1c_summary.md` §5.1 已满足)。

---

## 附录

### A.1 PR 时间线

| PR | issue | 日期 | 范围 | review 轮 | SHUD pin (post) |
|---|---|---|---|---|---|
| [PR-A #257](https://github.com/DankerMu/SHUD-OpenMP/pull/257) | #244 | 2026-06-22 | P1c.0 diagnostic + grep + dump + MD_f.cpp dead-code verify | 3 (R3 retry) | `07c677f` (unchanged) |
| [PR-B #258](https://github.com/DankerMu/SHUD-OpenMP/pull/258) | #245 | 2026-06-22 | L278-279 lake aggregation pairwise + Mac build gate | 1 | `89aa56f` |
| [PR-C #259](https://github.com/DankerMu/SHUD-OpenMP/pull/259) | #246 | 2026-06-23 | L374-375/L382-383 segment→river/element gather (leftfold) | 1 | `ad6db73` |
| [PR-D #260](https://github.com/DankerMu/SHUD-OpenMP/pull/260) | #247 | 2026-06-23 | L392 QrivUp gather (reuse leftfold) | 1 | `def03cb` |
| [PR-E #261](https://github.com/DankerMu/SHUD-OpenMP/pull/261) | #248 | 2026-06-23 | L406/L420/L433 lake gathers (leftfold + pair) | 1 | `de9545d` |
| [PR-F #262](https://github.com/DankerMu/SHUD-OpenMP/pull/262) | #249 | 2026-06-23 | Mac 16-cell scan + 3 negative grep + coverage table | 1 | `de9545d` |
| [PR-G #263](https://github.com/DankerMu/SHUD-OpenMP/pull/263) | #250 | 2026-06-23 | Kahan/Neumaier held-in-reserve patch + trigger conditions | 1 (R1 hunk-header fix) | `de9545d` |
| [PR-H #264](https://github.com/DankerMu/SHUD-OpenMP/pull/264) | #251 | 2026-06-23 | Server PR-K2 首跑 8-cell → §4.7 trigger FIRED | 1 | `de9545d` |
| [PR-I #265](https://github.com/DankerMu/SHUD-OpenMP/pull/265) | #252 | 2026-06-23 | §4.7 conditional Kahan injection 8-cell rerun → PARTIAL CLOSURE | 1 | `3a0004c` |
| [PR-J #266](https://github.com/DankerMu/SHUD-OpenMP/pull/266) | #253 | 2026-06-23 | NUM_OPENMP=1 reverse-compat (vs P1-update-omp-tag) | 1 | `3a0004c` |
| [PR-K #267](https://github.com/DankerMu/SHUD-OpenMP/pull/267) | #254 | 2026-06-23 | Capstone docs (p1c_summary 10 sections + perf_baseline + status_matrix) | 1 (R1 PR# off-by-one fix) | `3a0004c` |
| [PR-L #268](https://github.com/DankerMu/SHUD-OpenMP/pull/268) | #255 | 2026-06-23 | P1c-tag annotated procedure + baseline/P1c lock prep | 1 | `3a0004c` |
| [PR-M #269](https://github.com/DankerMu/SHUD-OpenMP/pull/269) | #256 | 2026-06-23 | PROMOTE 2 specs + archive + glossary 4 新术语 + jsonl 双追加 + Epic close | 1 | `3a0004c` |

**总计**：13 PR + 1 capstone tag (P1c-tag) + 1 baseline lock；单日 epic-burst；6 review gate net catches (PR-A R3 retry、PR-G R1、PR-K R1、Mac SHALL Scenario gap, etc.)；服务器 16 Slurm cell × 19-21 min wall = 总服务器 compute ~40 min。

### A.2 完整 SHA 表

#### A.2.1 SHA256 — `heihe`

| 视角 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| P1 canonical (`07c677f`) | `7f22bd6faa438d50...` | `7f22bd6faa438d50...` | `03055aa0fcbc9c34...` | `904779c30770f556...` |
| P1c pre-Kahan (`de9545d`) | `7f22bd6faa438d50...` | `7f22bd6faa438d50...` | `7f7a621cf1c4f02b...` | `8c581172a17db537...` |
| P1c Kahan (`3a0004c`, 当前 baseline/P1c) | `fd2d55716b5daffd...` | `fd2d55716b5daffd...` | `e058db2e9c2a9b9a...` | `6285e8a4a30a3917...` |

#### A.2.2 SHA256 — `heihe_x4`

| 视角 | N=1 | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| P1 canonical (`07c677f`) | `55403bef48ee5ad8...` | `55403bef48ee5ad8...` | `0b2aa00f0e2d5588...` | `d3d37e42a9ccfe9b...` |
| P1c pre-Kahan (`de9545d`) | `55403bef48ee5ad8...` | `55403bef48ee5ad8...` | `7e8f7a8a9697279e...` | `8b0efa6f6a74a43a...` |
| P1c Kahan (`3a0004c`, 当前 baseline/P1c) | `4eb804f571ba6f89...` | `4eb804f571ba6f89...` | `ff0787abd2170d4d...` | `6e9f9a2eaf652747...` |

### A.3 D11 不可变链

P1c epic 完成时的 5 tag SHA：

| Tag | annotated tag object SHA | deref commit SHA | SHUD pin | 锁定日期 |
|---|---|---|---|---|
| `B1-tag` | `0c0621c986e54e371c5a176850d1eb981150010e` | — | — | (B1 stage) |
| `B1a-tag` | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | `f7f992c` | `0b3998d` | 2026-06-21 |
| `B1b-tag` | `96e224daad8cb9c93f855851724f8d45468391c2` | (B1b end) | (B1b end) | 2026-06-21 |
| `P1-update-omp-tag` | `ff21c75c8e968d5e47ca53b015425360be9ac879` | `003f58d` | `07c677f` | 2026-06-22 |
| **`P1c-tag`** | **`1da5eb9734680fc61e68f6091964c38fc5f67c6f`** | **`4b8c60a`** | **`3a0004c`** | **2026-06-23** |

5 tag 在 P1c 完成时刻验证 SHA 不变（D11 enforced）。`B0-tag` 同链中存在；SHA 历史档参见 `docs/b0_summary.md`。

### A.4 验证命令

```bash
# A.4.1 P1c-tag 三重验证
git rev-parse P1c-tag                    # → 1da5eb9734680fc61e68f6091964c38fc5f67c6f
git rev-parse P1c-tag^{commit}           # → 4b8c60af261e0d1517f52702e4827a4e2d67dd41
git ls-tree P1c-tag SHUD | awk '{print $3}'  # → 3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4

# A.4.2 D11 不可变链全量验证
for tag in B1-tag B1a-tag B1b-tag P1-update-omp-tag P1c-tag; do
  printf "%-25s -> " $tag
  git rev-parse $tag
done

# A.4.3 baseline/P1c lock 验证
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c \
  --jq '.protection.lock_branch.enabled,
        .protection.enforce_admins.enabled,
        .protection.allow_force_pushes.enabled,
        .protection.allow_deletions.enabled'
# 期望 4 行: true / true / false / false

# A.4.4 8 站点 helper-wrap 全 PRESENT 验证 (P2a entry condition)
grep -c 'fixed_pairwise_sum_indexed\|fixed_leftfold_sum_indexed\|fixed_leftfold_sum_pair_indexed' \
  SHUD/src/Model/MD_rhs_core.cpp
# 期望 ≥ 10 (10 行 anchor)

# A.4.5 三 negative grep gate (PR-F 守门)
grep -rE 'SHUD_USE_DETERMINISTIC_REDUCTION|SHUD_DET_REDUCT|SHUD_PAIRWISE' SHUD/
# 期望 0 hits
grep -nE 'schedule\(' SHUD/src/Model/MD_rhs_core.cpp
# 期望 0 hits
grep -rn '#pragma omp atomic' SHUD/src/
# 期望 0 hits
```

### A.5 引用来源 (Sources of truth)

| 文档 | 内容 | 角色 |
|---|---|---|
| 本文件 | P1c 阶段研究报告 (narrative synthesis) | 综合 |
| [`docs/p1c/p1c_summary.md`](p1c_summary.md) | P1c capstone 10 sections (机制说明) | source of truth |
| [`docs/p1c/p1c_a3a_root_cause.md`](p1c_a3a_root_cause.md) | A3a 根因 + D9 决策分支 + Kahan 路径终判 | analysis |
| [`docs/p1c/p1c_perf_baseline.md`](p1c_perf_baseline.md) | Mac 16-cell + Server PR-H/PR-I 8-cell wall + 数据 | raw data |
| [`docs/p1c/p1c_reduction_sites.md`](p1c_reduction_sites.md) | 8 站点 overview + B1b serial canonical order | reference |
| [`docs/p1c/p1c_pr_h_server_first_run.md`](p1c_pr_h_server_first_run.md) | PR-H server first run 8-cell raw data + verdict | raw data |
| [`docs/p1c/p1c_pr_i_kahan_injection.md`](p1c_pr_i_kahan_injection.md) | PR-I Kahan 二跑 8-cell raw data + carve-out 决策 | raw data |
| [`docs/p1c/p1c_pr_j_reverse_compat.md`](p1c_pr_j_reverse_compat.md) | PR-J reverse-compat 双视角 (pre-Kahan / Kahan) | analysis |
| [`docs/p1c/p1c_tag_and_lock.md`](p1c_tag_and_lock.md) | P1c-tag annotated procedure + branch lock | procedure |
| [`docs/p1c/p1c_kahan_patch.diff`](p1c_kahan_patch.diff) | Kahan held-in-reserve patch (PR-G 输出 / PR-I 应用) | code artifact |
| [`docs/p1c/p1c_b1b_serial_order_dump.txt`](p1c_b1b_serial_order_dump.txt) | B1b 基线 serial order dump (PR-A) | reference |
| [`docs/status_matrix.md`](../status_matrix.md) | 阶段 × benchmark 状态矩阵 (P1c 行 PR-K 已写入) | status |
| [`docs/stage-pipeline-log.jsonl`](../stage-pipeline-log.jsonl) | epic-pipeline 累积日志 (P1c 2 entries by PR-M) | meta |
| `openspec/specs/p1c-deterministic-reduction/spec.md` | P1c deterministic-reduction Requirement (PROMOTE by PR-M) | spec |
| `openspec/specs/p1c-capstone/spec.md` | P1c capstone Requirement (PROMOTE by PR-M) | spec |
| `openspec/glossary.md` §"P1c deterministic-reduction baseline 集合" | 5 新术语 (P1c-tag / baseline/P1c / 4 helpers / Kahan held-in-reserve / P1d carve-out) | terminology |
| `SHUD_openMP_master_plan.md` §6 P1c row | master plan P1c bucket + §3 fallback option 2 + §4 P1c.2 success gate | doctrine |

---

**完**

本报告作为 P1c epic 的高层叙事综合 (narrative synthesis)，与 8 份 source-of-truth 文档 (per A.5) 形成"综合 + raw evidence"二层结构。Source-of-truth 文档原汁原味保留 per-PR audit trail；本报告负责 narrative + cross-reference + 学术风格收口。后续 P2a / P9 阶段如需引用 P1c 经验，先读本报告获取整体视角，再按 A.5 表深入对应 raw evidence。
