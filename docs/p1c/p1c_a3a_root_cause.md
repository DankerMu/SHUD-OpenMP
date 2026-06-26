# P1c.0 — A3a 实验诊断 + 根因分析

OpenSpec change `p1c-deterministic-reduction` Stage 1 (PR-A #244) §1.5 task —
P1c.0 实验诊断 (前置 SHALL, per design D9)。本文档在 PR-A 阶段建立结构 +
登记 Mac local sanity check 初步数据，server PR-K2 二跑 (PR-H stage) 补
最终量化数据 + decision branch 终判。

## 上下文

- **分支模型 (P1c 阶段)**: `baseline/P1c` 是 P1c 阶段活动集成线, PR-A..PR-M 全部 base 该分支; PR-H server 数据回填本文档时同样 base baseline/P1c, per master plan §6 P1c.4 baseline lock 与 tag + P1c.6 后续移交.
- **基线**: SHUD pin `07c677f` (B1b-tag SHUD + PR-D/PR-E/PR-F element/river/lake `#pragma omp parallel for` 三栈, PR-K2 #223 实测 state).
- **P1 epic 实测 (PR-K2 #223 / 2026-06-22)**: heihe `nst` 跨 N ∈ {1,2,4,8} = **6773 / 6773 / 6585 / 6684**，N≥4 漂移；heihe_x4 `nst` = 6571 / 6571 / 6570 / 6572 (微漂)。
- **疑似机理 (Stage 1 hypothesis, per proposal.md L3)**: P1 阶段已并行的 owner-local writer 路径 (Model_Data.cpp / MD_update.cpp 中的 `#pragma omp parallel for`) 在 N>2 下产生 first-touch / NUMA-affinity 异序写入 ULP 噪声 → 下游 `rhs_deterministic_gather()` 8 reduction 站点忠实累加噪声 → CVODE 自适应步长重选 → `nst` 跨 N 漂移。

## Kahan 候选路径 (held-in-reserve, §4.7 条件触发)

PR-G (#250) 产 `docs/p1c/p1c_kahan_patch.diff` 作为 held-in-reserve patch.
**NOT applied** in PR-G; conditional application only on §4.7 trigger
(server PR-K2 first run FAIL). Per spec L107-L128 `Kahan 补偿求和兜底
(条件触发, server PR-K2 首跑 FAIL)` Requirement + design D2 + D4 + R2.

### (a) 算法 — Neumaier 1974 (Kahan-Babuška variant)

经典 Kahan 算法:

```cpp
y = x - c;
t = sum + y;
c = (t - sum) - y;
sum = t;
```

Neumaier 改进: 处理 |sum| < |x| 时的 sign-bit asymmetry —

```cpp
t = sum + x;
c += (std::fabs(sum) >= std::fabs(x)) ? (sum - t) + x : (x - t) + sum;
sum = t;
// final return: sum + c
```

Neumaier 在 ULP-level 浮点累加意义上是 backward-stable
(compensation 始终捕获 next-bit-below 的 round-off). 选 Neumaier 而非
经典 Kahan 的理由: 站点 5/6/7 (Qe2r_Surf, Qe2r_Sub, QrivUp) 的 sigma
经 `-fixed_leftfold_sum_indexed(...)` 一次性 negate, 但中间过程在 call-
site chaining 仍可 sign-mixed; 经典 Kahan 在 sign-mixing 序列上
vulnerable, Neumaier 显式分支处理 |sum|<|x| 情况。

参考: SUNDIALS user guide §3.1.1 "deterministic dot-product" +
Neumaier 1974 ZAMM Vol 54.

### (b) 8 站点 × 10 anchor 叠加位置 (helper-level injection)

| 站点 | 写目标 | 当前 helper (post-PR-E) | Kahan 注入后 helper |
|---|---|---|---|
| 1 (L278) | qLakeEvap | fixed_pairwise_sum_indexed (PR-B) | fixed_pairwise_sum_indexed + Neumaier-compensated tree join |
| 2 (L279) | qLakePrcp | fixed_pairwise_sum_indexed (PR-B) | same |
| 3 (L374) | QrivSurf | fixed_leftfold_sum_indexed (PR-C) | fixed_leftfold_sum_indexed_kahan (Neumaier loop) |
| 4 (L375) | QrivSub | fixed_leftfold_sum_indexed (PR-C) | same |
| 5 (L382) | Qe2r_Surf | -fixed_leftfold_sum_indexed (PR-C) | -fixed_leftfold_sum_indexed_kahan |
| 6 (L383) | Qe2r_Sub | -fixed_leftfold_sum_indexed (PR-C) | same |
| 7 (L392) | QrivUp | -fixed_leftfold_sum_indexed (PR-D) | -fixed_leftfold_sum_indexed_kahan |
| 8a (L406) | QLakeRivIn | fixed_leftfold_sum_indexed (PR-E) | fixed_leftfold_sum_indexed_kahan |
| 8b (L420) | QLakeSurf | fixed_leftfold_sum_pair_indexed (PR-E) | fixed_leftfold_sum_pair_indexed_kahan |
| 8c (L433) | QLakeSub | fixed_leftfold_sum_pair_indexed (PR-E) | same |

实际 patch (`docs/p1c/p1c_kahan_patch.diff`) 在 4 个 helper def
(`fixed_pairwise_sum_range`, `fixed_leftfold_sum_indexed`,
`fixed_leftfold_sum_pair_indexed` × 2 use-sites) 内修改累加循环, 即 cover
所有 10 个 call sites — helper-wrap 结构允许 patch 不需逐 anchor 重写
call site, 比 inline-per-site 改造少 6 处 diff hunk。

注: 站点 1-2 (qLakeEvap / qLakePrcp) 通过 `fixed_pairwise_sum_indexed`
→ `fixed_pairwise_sum_range` 间接获 Neumaier 补偿; 站点 3-7 + 8a 通过
`fixed_leftfold_sum_indexed` 直接获补偿; 站点 8b-c 通过
`fixed_leftfold_sum_pair_indexed` 直接获补偿. 4 helper def 修改 cover
10 call sites (R1 Suggestion fold-in: 透明 transitive coverage 关系).

**Header include note**: post-PR-E HEAD (`de9545d`) 的 `MD_rhs_core.cpp`
不直接 include `<cmath>` (`grep -nE '#include' …` 验证), 注释掉的 L611/L627
legacy `fabs` 通过 `Model_Data.hpp` 链 transitive 拾起 `<cmath>`. Patch
**显式** add `#include <cmath>` — Neumaier `std::fabs(lo)` 分支是
hot-path use, 必须在 helper TU 内自洽 resolve, 不依赖 transitive.

### (c) 触发条件 / 非触发条件

**SHALL 触发** (§4.7 server PR-K2 首跑 FAIL):

- §4.4 A3a bitwise FAIL: heihe 4 N OR heihe_x4 4 N 任一 N pair 出现 byte
  diff.
- OR §4.5 nst across N FAIL: heihe nst 跨 N ∈ {1,2,4,8} 不全相等 (P1
  实测 6585/6684 漂移仍在则触发).
- OR §4.5 nst across N: heihe_x4 残留 `|Δ_nst|>2` (`≤2` 走 SPGMR-noise
  ladder per §4.5.x, 不直接触 Kahan).
- OR reverse-compat FAIL: heihe / heihe_x4 N=1 与 P1-update-omp-tag 不
  字面相等.

**SHALL NOT 触发** (per design D7):

- Mac local §2.5 16-cell scan 任一 FAIL (Mac pass-while-server-fails
  已知模式, Mac M4 Pro 4P+10E 异构核心 + libomp 弱绑定 per master plan
  §7.2 RISK-26 即便有 server NUMA-affinity 噪声 mechanism 在场也不会在
  Mac local snapshot 层面显现, 见 PR-F `docs/p1c/p1c_perf_baseline.md` §1).
- 单 PR (B / C / D / E) 增量 keliya bitwise vs B0 FAIL (这是 PR 自身
  gate, 应在 PR 级 fix, 不上升到 Kahan).

### (d) Wall-clock 影响估算 (per design R2)

- Neumaier 每 `+=` 增 1 magnitude-compare + 3 FP op (vs 1 op naive `+=`).
- 10 anchor × heihe-scale fanin (NumRiv × avg fanin + NumLake × avg
  lake-ele) per RHS call, overhead ratio < 5%.
- 经验估算: heihe ~6800 RHS calls × overhead ratio ≈ 1-3% total wall
  微降 (CVODE step memory-bound dominates).
- 若实测 wall 增 > 5%, 标 perf-regression-investigation, 入本文档跟踪
  条目, 但不阻 Kahan-injected 路径 A3a success gate; perf rollback 走独
  立 PR (不复用 PR-K2 二跑).

**Tracking 阈值**: 实测 wall delta > 5% vs pre-Kahan PR-E HEAD ⇒
record in `docs/p1c/p1c_perf_baseline.md` 跟踪条目 (perf-regression-investigation),
但不 block A3a success gate.

### (e) 条件应用流程 (§4.7 触发后, PR-K2 二跑)

**触发条件** (§4.7 server PR-K2 FIRST run FAIL, 任一):

- §4.4 A3a bitwise FAIL on heihe 4 N OR heihe_x4 4 N (任一 N-pair byte diff).
- OR §4.5 nst across N FAIL (heihe nst 跨 N ∈ {1,2,4,8} 不全相等).
- OR §4.5 heihe_x4 nst 残留 `|Δ_nst| > 2` (`≤2` 走 SPGMR-noise ladder,
  不直接触 Kahan).
- OR reverse-compat FAIL (heihe / heihe_x4 N=1 与 P1-update-omp-tag 不
  字面相等).

**非触发条件** (per design D7, MUST NOT trigger Kahan):

- Mac local §2.5 16-cell scan ANY FAIL (Mac pass-while-server-fails
  已知模式 — Mac M4 Pro 4P+10E 异构核心 + libomp 弱绑定 mask server
  NUMA-affinity noise, per master plan §7.2 RISK-26).
- 单 PR (B / C / D / E) 增量 keliya bitwise vs B0 FAIL (PR-level gate, fix
  at PR; 不上升到 Kahan).

**应用命令** (only when §4.7 trigger fires, PR-K2 二跑):

```bash
# Server cn0X, /scratch/frd_muziyao/SHUD-OpenMP/
cd SHUD && git apply ../docs/p1c/p1c_kahan_patch.diff  # verified PASS via
                                                    # `git apply --check`
                                                    # on PR-G; patch source
                                                    # pin = SHUD@de9545d
git commit -m "P1c §4.7 conditional Kahan injection (Neumaier 1974) — PR-K2 二跑"
git push origin openmp-baseline
# 外层 pointer bump
cd .. && git add SHUD
git commit -m "SHUD pointer bump: Kahan-injected (§4.7 PR-K2 二跑)"
# Slurm 三铁律 sbatch from /scratch
sbatch /scratch/frd_muziyao/SHUD-OpenMP/.<stage>-runs/p1c_pr_k2_run2.sbatch
```

Per CLAUDE.md "SHUD submodule 工作流 (强制)" + "Slurm 三铁律".

**Patch source pin**: SHUD@`de9545d` (post-PR-E HEAD on `openmp-baseline`,
= 外层 `baseline/P1c` HEAD `afffcb2` SHUD pointer). Patch 由 `git diff` 自动
生成 (临时分支 → reset 不污染 upstream), 自动计算 hunk header 行数;
PR-G 已 `git apply --check` clean verify. 若 SHUD upstream HEAD 漂移,
patch 再生于 `git apply` 失败前 SHALL 重新 rebase 到当前 HEAD (helper def
行号偏移; algorithm 内容不变).

**应用后 verification**:

- SHUD `make shud_omp` build PASS (no `<cmath>` symbol unresolved).
- PR-K2 二跑 A3a bitwise PASS on heihe + heihe_x4 × N ∈ {1,2,4,8}.
- nst across N all-equal (heihe) / `|Δ_nst| ≤ 2` (heihe_x4).
- Reverse-compat: N=1 byte-equal to P1-update-omp-tag.
- Wall delta ≤ 5% vs pre-Kahan PR-E HEAD (record in
  `docs/p1c/p1c_perf_baseline.md`).

---

## 诊断结果

下表 4 项必备字段 (per spec p1c-capstone §"p1c_a3a_root_cause 吸收 F-K2-2 +
量化数据" Scenario)。**PR-A 阶段 (本 PR)**: 字段 (a)/(b)/(c) Mac local 部分
有初步数据 + 字段 (d) 有 Mac local hypothesis 测试；server 完整 PR-K2 二跑
数据由 PR-H 补 (per fixture L226-235 "Mac local sanity check (本 PR 内) +
占位字段")。

### (a) pre-fix DY divergence first-occurrence

**Server PR-K2 #223 实测 reference** (heihe nst 数据见 `docs/p1_summary.md` + `docs/p1/p1_perf_baseline.md` + `docs/build_manifest.md`; heihe_x4 nst 数据见 `docs/p1/p1_perf_baseline.md`; F-K2-2 reviewer finding tracking in P1 fullrun review): heihe `nst` 跨 N ∈ {1,2,4,8} = 6773/6773/**6585**/**6684**, heihe_x4 `nst` = 6571/6571/**6570**/**6572**; PR-H 二跑 success gate 要求 spec L172 Δ=0 强制 (heihe) + heihe_x4 Δ=0 with optional SPGMR-noise ladder (per design D6).

**PR-H 首跑实测 (2026-06-22, SHUD@de9545d, post-PR-E HEAD)** (per `docs/p1c/p1c_pr_h_server_first_run.md` §3-§6):
- heihe nst {N=1,2,4,8} = {6773, 6773, **6682**, **6548**}, |Δ_max|=225 (≫ §4.5 D9 ≤2 阈值)
- heihe_x4 nst {N=1,2,4,8} = {6571, 6571, **6568**, **6570**}, |Δ_max|=3 (越 §4.5 D9 ≤2 阈值)
- A3a bitwise: 双 case 全 FAIL (3 distinct SHAs per case, N=1≡N=2 due to SHUD internal `max(NUM_OPENMP,2)` thread floor)
- **§4.7 SHALL TRIGGER**: PR-I (#252) 走条件 Kahan injection 主分支 (per PR-G `docs/p1c/p1c_kahan_patch.diff`)，PR-K2 二跑。

P1 era (B1b PR-K2 #223) 与 P1c PR-H 首跑 nst 数值不同 (heihe 6585/6684 → 6682/6548) 因 SHUD HEAD 漂移 (P1 era 07c677f → P1c post-PR-E de9545d 含 8 站点 helper-wrap)。Pattern 一致 (N≥4 漂移)，幅度变化。

| 字段 | Mac local 初步 | Server PR-H 首跑 (de9545d, pre-Kahan) |
|---|---|---|
| step_index (CVODE) | N/A — Mac local 6/6 BITWISE IDENTICAL | TBD via Kahan-injected 二跑 + per-site rhs_dump probe (deferred to PR-K diagnostic deep-dive) |
| array_name | N/A — no first-divergence anchor observed | TBD via 同上 |
| ele/seg/riv index | N/A — see Mac sanity check below | TBD via 同上 |
| bit-level diff | 0 byte (Mac 6/6 BITWISE IDENTICAL) | heihe N=1↔N=4 ≠ 0 byte (整 rivqdown.dat SHA mismatch, 90-day timeseries); heihe_x4 同。Per-step / per-site bit-diff 待 PR-I/K 二跑 + 加 instrumentation |

**Mac local sanity check** (本 PR §1.5 (1) 部分):

- **Build**: SHUD@07c677f (PR-A start state), `make shud_omp SHUD_DUMP_RHS=1`, strict FP flags (`-ffp-contract=off` + `-fno-fast-math` + `-fopenmp` 各 ≥1 hit in `/tmp/p1c_make_omp_dump.log`).
- **Run**: `keliya` 与 `qhh` 各 N ∈ {1, 4, 8} = 6 cell (90d cfg), `SHUD_DUMP_SITE=f_applyDY` snapshot at `abs_min=17357760` (keliya) / `abs_min=12098880` (qhh).
- **Comparison**: `tools/compare_snapshot/compare_snapshot <a.bin> <b.bin>`.
- **结果 (keliya, 3 N pair)**:
  - keliya N=1 vs N=4 ⇒ **BITWISE IDENTICAL** (exit 0)
  - keliya N=1 vs N=8 ⇒ **BITWISE IDENTICAL** (exit 0)
  - keliya N=4 vs N=8 ⇒ **BITWISE IDENTICAL** (exit 0)
- **结果 (qhh, 3 N pair)** — lake-bearing 4773 ele case，更可能触发上游 writer noise:
  - qhh N=1 vs N=4 ⇒ **BITWISE IDENTICAL** (exit 0)
  - qhh N=1 vs N=8 ⇒ **BITWISE IDENTICAL** (exit 0)
  - qhh N=4 vs N=8 ⇒ **BITWISE IDENTICAL** (exit 0)

⇒ **Mac local 未复现 N≥4 DY 漂移** at `f_applyDY` snapshot anchor for **both**
keliya (484 ele, no lake) and qhh (4773 ele, with lake) across N {1, 4, 8}。
这与 PR-K1 #222 Mac 16-cell A3a 全 PASS via snapshot probe 的历史结论
一致 (per proposal L17, design D7)。Mac M4 Pro 4P+10E 异构核心 + libomp 弱
绑定 (per master plan §7.2 RISK-26) 即便有 PR-K2 服务器观察到的 NUMA-affinity
噪声 mechanism 在场，也不会在 Mac local snapshot 层面显现 — 此现象与
design D7 "Mac snapshot 已知 pass-while-server-fails" 一致。

⇒ **服务器复跑 (PR-H stage) SHALL 提供 PR-K2 #223 等价 8-cell 数据** (heihe +
heihe_x4 × N ∈ {1,2,4,8} = 8 cell) 以确认 first-occurrence。Mac local 单 case
单 anchor 数据不足以定位 server cn0X N≥4 漂移源 (per design D7 "Mac snapshot
已知 pass-while-server-fails")。

### (b) post-fix same probe showing zero divergence

| 字段 | TBD via PR-H (post-§2.x 改造后) |
|---|---|
| step_index (CVODE) | TBD |
| array_name | TBD |
| ele/seg/riv index | TBD |
| bit-level diff | 期望 0 byte (post-fix bitwise identical) |

### (c) per-site ULP delta table

8 站点 × N {2, 4, 8} = 24 cells. PR-A 阶段无数据 (Mac local 9-of-10 anchors
PRESENT 但 N=1 baseline-only, 无 cross-N ULP delta probe; 1-of-10 anchor
即 L343 QLakeRivIn ABSENT 在 4 Mac case (per dump.txt R3 audit table),
SHALL 由 server PR-K2 二跑 heihe + heihe_x4 lake-bearing case 补)。
Server PR-K2 二跑 (PR-H) 期望矩阵：

| Site | N=2 ULP delta | N=4 ULP delta | N=8 ULP delta |
|---|---|---|---|
| L278 qLakeEvap | TBD | TBD | TBD |
| L279 qLakePrcp | TBD | TBD | TBD |
| L374 QrivSurf | TBD | TBD | TBD |
| L375 QrivSub | TBD | TBD | TBD |
| L382 Qe2r_Surf | TBD | TBD | TBD |
| L383 Qe2r_Sub | TBD | TBD | TBD |
| L392 QrivUp | TBD | TBD | TBD |
| L406 QLakeRivIn | TBD | TBD | TBD |
| L420 QLakeSurf | TBD | TBD | TBD |
| L433 QLakeSub | TBD | TBD | TBD |

### (d) tree-reduction-depth N>2 hypothesis confirm/refute

**Mac local 初步分析 (本 PR)**:

- 上述 keliya Mac local 数据 (N=1/4/8 bitwise identical at f_applyDY snapshot) **既不 confirm 也不 refute** master plan v1.3 §6 P7.3.5 "tree-reduction depth N>2 跃迁" hypothesis (proposal L3 已将该 hypothesis 修订为 owner-local writer first-touch / NUMA-affinity 异序)，因 Mac M4 Pro 4P+10E 异构核心 + libomp 弱绑定 (per master plan §7.2 RISK-26) 即便有该 hypothesis 在场，Mac snapshot 层面也未必触发。
- proposal.md L3 + design D9 已**显式修订**该 hypothesis: P1 阶段 `rhs_deterministic_gather()` 8 站点本身**仍为纯 serial 执行** (`grep -nE '#pragma omp' SHUD/src/Model/MD_rhs_core.cpp` 不命中 `#pragma omp parallel`)，因此漂移**不**由 8 站点的 OpenMP tree-reduction 直接触发；机理更可能是上游 parallel writer first-touch / NUMA-affinity 异序写入 ULP 噪声 → gather 站点忠实累加放大。
- **server PR-K2 二跑 (PR-H) 数据 SHALL 最终判定**: (i) heihe N=1/N=4 在哪个 step / 哪个 RHS-internal array 首先发散; (ii) 该 array 是否在 8 站点 reduction 路径之上 (= writer noise)，还是路径外 (= writer 自身 noise，gather 仅忠实复制)。

**初步判定 (per Mac local + Stage 1 grep)**:

- 倾向 hypothesis 修订版 (writer noise → gather amplification)，即 **decision branch 2** (诊断结果落在 8 站点 reduction 路径**外部**)；
- 故 §2.x 8 站点 fixed-shape pairwise 改造仍 SHALL 推进作为 ULP 噪声放大阻断手段 (per design D9 决策分支 2)；
- 同时 server PR-K2 数据 (PR-H) SHALL 验证: 若实际证据指向上游 writer noise 而非 gather noise，**spec R2 表 SHALL 在 capstone 锁前扩展 in-scope 表 OR 显式记录为 carve-out 推 P1d** (per design D6 + spec L100-L103 Scenario)。

---

## design D9 decision branch 判定 (PR-A 阶段初步)

| 分支 | 描述 | PR-A 阶段判定 | 终判 (PR-I 后) |
|---|---|---|---|
| 1 | 诊断结果落在 8 站点 reduction 路径 | **NOT confirmed** by Mac local | **REFUTED** — Kahan 完全注入 (PR-I §3-§6) 残 |Δ_nst|=84 / SHAs cross-N 不等 = drift 不在 8 站点内部 |
| 2 | 诊断结果落在 8 站点 reduction 路径**外部** (parallel writer noise) | **PROBABLE** per Stage 1 + Mac local + proposal L3 修订 hypothesis | **CONFIRMED** (PR-I 8-cell Kahan 数据) |
| 3 | 诊断显示 8 站点改造与漂移源无任何因果链 | NOT supported by Stage 1 grep | **PARTIALLY REFUTED** — Kahan 改善 heihe Δ_nst 225→84 (~63% 改善) 证 8 站点 reduction 与 drift 有部分因果链 (作为 noise 放大器), 但非源头 |

**PR-I 阶段终判结论**: **分支 2 CONFIRMED** (诊断结果落在 8 站点 reduction 路径外部)。
- ✓ §2.x fixed-shape pairwise 改造已完成 (PR-B/C/D/E)，作为通用 ULP 噪声阻断手段在 helper 层闭合;
- ✓ Kahan 兜底 (PR-G/PR-I) 已注入并测量, 部分改善但未关闭 §4.4 A3a + §4.5 nst Δ=0 门;
- ✓ **carve-out 推 P1d** (per design D6 + spec L100-L103 Scenario): 上游 writer first-touch / NUMA-affinity 治理 (OMP_PROC_BIND + OMP_PLACES + numactl interleave) 为 P1d stage 单独范围;
- 详 PR-I §8 "§4.7 二次决策 — PARTIAL CLOSURE + P1d CARVE-OUT" 三条结论 + carve-out scope 表 + extend 选项 cost 分析。
- 若 server 数据 confirm 分支 1 (落在 8 站点内)，§2.x 改造直接闭合，无需 spec 扩展。
- 若 server 数据 confirm 分支 3 (无因果链)，**STOP §2.x 推进**，P1c 范围重新评估，退回 master plan §6 P1c.1 候选 (c) Deterministic OpenMP N_Vector。

---

## Mac local sanity check 复现脚本 (本 PR §1.5 (1) 部分)

```bash
# 0) verify cfg.para 已 90d 截断 (项目铁律: 所有 case ≤90 天截断, per CLAUDE.md;
#    shipped Basins/keliya/input/keliya/keliya.cfg.para 已就位 90d)
grep -E '^START|^END' SHUD/Basins/keliya/input/keliya/keliya.cfg.para
# expected: START 12053 / END 12143 (END - START = 90 day-index 制)
# 若不等, 改 cfg.para 或用 `tools/fix_case_paths/fix_case_paths.sh` (该工具
# rewrite path + NUM_OPENMP, 不 truncate END; END 需手动改).
```

```bash
# 1) build shud_omp with SHUD_DUMP_RHS=1
cd SHUD && make clean && make shud_omp SHUD_DUMP_RHS=1 2>&1 \
  | tee /tmp/p1c_make_omp_dump.log
# strict FP gate
grep -c '\-ffp-contract=off' /tmp/p1c_make_omp_dump.log   # >= 1
grep -c '\-fno-fast-math'    /tmp/p1c_make_omp_dump.log   # >= 1
grep -c '\-fopenmp'           /tmp/p1c_make_omp_dump.log   # >= 1

# 2) run keliya across N
cd Basins/keliya
mkdir -p /tmp/p1c_dump/keliya_N{1,4,8}
for N in 1 4 8; do
  OMP_NUM_THREADS=$N \
    SHUD_DUMP_OUTPUT_DIR=/tmp/p1c_dump/keliya_N${N} \
    SHUD_DUMP_CASE_ID=keliya \
    SHUD_DUMP_SITE=f_applyDY \
    SHUD_DUMP_T_VALUES=17357760 \
    SHUD_DUMP_T_TOL=60 \
    ../../shud_omp keliya >/dev/null
done

# 3) compare
cd ../../..
./tools/compare_snapshot/compare_snapshot \
  /tmp/p1c_dump/keliya_N1/snapshot_t17357760.bin \
  /tmp/p1c_dump/keliya_N4/snapshot_t17357760.bin
# => BITWISE IDENTICAL (exit 0)

./tools/compare_snapshot/compare_snapshot \
  /tmp/p1c_dump/keliya_N1/snapshot_t17357760.bin \
  /tmp/p1c_dump/keliya_N8/snapshot_t17357760.bin
# => BITWISE IDENTICAL (exit 0)
```

qhh case (lake-bearing, 4773 ele) 同模式，`SHUD_DUMP_T_VALUES=12098880`
(START=8401 + 1 day = 8402 day × 1440 = 12098880 min).

---

## Hand-off → PR-H (server PR-K2 二跑) checklist

- [ ] **Re-dump 1 ABSENT anchor on server case** — Mac local dump 缺 L343 QLakeRivIn (= class-1 L406, per dump.txt R3-fixed L60-L64 (L343 ABSENT row) + L68-L76 (Implication footer) + Anchor hit count audit table + sites.md Line-number 等价表). 这是 4 Mac case 唯一真正 ABSENT 的 anchor (L319/L320 在 R3 retro fix 已修正为 PRESENT, 各 6240 body hits, 无需 server re-dump). server PR-K2 二跑 instrumentation SHALL 在 heihe + heihe_x4 case (lake-bearing case 才能触发 `riv_in_by_lake[ilake]` non-empty) 各 N ∈ {1,4,8} 复跑时补这 1 anchor 的 `[p1c_dump]` trace 行，写回 `docs/p1c/p1c_b1b_serial_order_dump.txt` 末尾 OR 新建 `docs/p1c/p1c_server_order_dump.txt` (二者皆可，capstone PR-K decide).
- [ ] **Slurm 三铁律 (server cn0X 提交 SHALL 遵循)** — (i) `sbatch` 从 `/scratch` 下提交 (不从 `/users/$USER`); (ii) `#SBATCH --output/--error` 路径必须在 `/scratch` 共享盘 (compute node `/tmp` node-local, 作业结束丢, sacct 显示 ExitCode 127); (iii) 作业脚本/patch/hash/run.sh 都放 `/scratch`. **禁 login node 跑 SHUD/keliya/heihe** (共享 CPU 30s 可膨胀到 30+ min). Per CLAUDE.md "Slurm 三铁律".
- [ ] **SHUD submodule 工作流 (PR-H 复跑 SHALL 遵循)** — PR-H 会 re-instrument SHUD source (插 8 站点 ULP delta probe + 1 ABSENT anchor dump i.e. L343 QLakeRivIn); 任何 SHUD source 改造 SHALL `cd SHUD && git commit && git push origin openmp-baseline` (长寿分支, 从 `3aec657` 派生) → `cd .. && git add SHUD && git commit` (pointer bump) → 外层 PR; **禁 push master / 禁 fork / 禁改 `.gitmodules`**. Per CLAUDE.md "SHUD submodule 工作流 (强制)".
- [ ] Server cn0X build `shud_omp SHUD_DUMP_RHS=1`，3-grep gate strict FP.
- [ ] heihe + heihe_x4 各 N ∈ {1, 4, 8} = 6 cell `f_applyDY` snapshot dump.
- [ ] `compare_snapshot` 跨 N 对 (heihe N=1 vs N=4 等)，定位首发散点；记入字段 (a)。
- [ ] §2.x fixed-shape pairwise 改造后 post-fix dump (同 case + 同 anchor), 记入字段 (b)。
- [ ] 在每个 reduction 站点入口 + 出口插 instrumented dump (ULP delta probe), 跑 N=2/4/8 各 8 站点 = 24 cell，填字段 (c) ULP delta 表。
- [ ] 基于 server 数据 update 字段 (d) tree-reduction hypothesis confirm/refute。
- [ ] 终判 D9 decision branch 1/2/3, update 本文档 §"design D9 decision branch 判定" 表的 "终判 (PR-H 后)" 列。
- [ ] 若 branch 2 或 branch 3, update spec R2 表 in-scope 扩展 OR 显式 carve-out 推 P1d (per design D6) **在 spec 锁前**完成。
