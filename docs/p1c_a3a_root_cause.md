# P1c.0 — A3a 实验诊断 + 根因分析

OpenSpec change `p1c-deterministic-reduction` Stage 1 (PR-A #244) §1.5 task —
P1c.0 实验诊断 (前置 SHALL, per design D9)。本文档在 PR-A 阶段建立结构 +
登记 Mac local sanity check 初步数据，server PR-K2 二跑 (PR-H stage) 补
最终量化数据 + decision branch 终判。

## 上下文

- **基线**: SHUD pin `07c677f` (B1b-tag SHUD + PR-D/PR-E/PR-F element/river/lake `#pragma omp parallel for` 三栈, PR-K2 #223 实测 state).
- **P1 epic 实测 (PR-K2 #223 / 2026-06-22)**: heihe `nst` 跨 N ∈ {1,2,4,8} = **6773 / 6773 / 6585 / 6684**，N≥4 漂移；heihe_x4 `nst` = 6571 / 6571 / 6570 / 6572 (微漂)。
- **疑似机理 (Stage 1 hypothesis, per proposal.md L3)**: P1 阶段已并行的 owner-local writer 路径 (Model_Data.cpp / MD_update.cpp 中的 `#pragma omp parallel for`) 在 N>2 下产生 first-touch / NUMA-affinity 异序写入 ULP 噪声 → 下游 `rhs_deterministic_gather()` 8 reduction 站点忠实累加噪声 → CVODE 自适应步长重选 → `nst` 跨 N 漂移。

## Kahan 候选路径

如 fixed-shape pairwise 改造 (§2.x) 完成后 server PR-K2 首跑仍 FAIL (A3a non-bitwise 或 nst 跨 N 非全等)，将触发条件 Kahan / Neumaier 兜底，per spec L107-L114 `Kahan 补偿求和兜底` Requirement + design D2 决策第二阶段。

Patch 设计草稿 (§3.1 任务): 8 个 owner accumulation 位上叠加 Neumaier 修正:

```cpp
// per-acc-target compensated sum (Neumaier variant of Kahan):
double sum = 0.0;
double c   = 0.0;
for (int iseg : seg_by_riv[ir]) {
    double y = QsegSurf[iseg] - c;
    double t = sum + y;
    c = (std::fabs(sum) >= std::fabs(y)) ? ((sum - t) + y) : ((y - t) + sum);
    sum = t;
}
QrivSurf[ir] = sum;  // or += sum if accumulator pre-zero already done
```

触发条件: server PR-K2 首跑 FAIL (per spec §"仅在必要时引入 Kahan" Scenario)。
**禁止**以 Mac local §2.5 16-cell snapshot 结果作为触发依据 (per design D7,
Mac snapshot 已知 pass-while-server-fails)。

---

## 诊断结果

下表 4 项必备字段 (per spec p1c-capstone §"p1c_a3a_root_cause 吸收 F-K2-2 +
量化数据" Scenario)。**PR-A 阶段 (本 PR)**: 字段 (a)/(b)/(c) Mac local 部分
有初步数据 + 字段 (d) 有 Mac local hypothesis 测试；server 完整 PR-K2 二跑
数据由 PR-H 补 (per fixture L226-235 "Mac local sanity check (本 PR 内) +
占位字段")。

### (a) pre-fix DY divergence first-occurrence

| 字段 | Mac local 初步 | Server PR-K2 (TBD via PR-H) |
|---|---|---|
| step_index (CVODE) | TBD | TBD |
| array_name | TBD | TBD |
| ele/seg/riv index | TBD | TBD |
| bit-level diff | TBD | TBD |

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

8 站点 × N {2, 4, 8} = 24 cells. PR-A 阶段无数据 (Mac local 7-of-10 anchors
未触发漂移 = 全 0 ULP delta)。Server PR-K2 二跑 (PR-H) 期望矩阵：

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
- 同时 server PR-K2 数据 (PR-H) SHALL 验证: 若实际证据指向上游 writer noise 而非 gather noise，**spec R2 表 SHALL 在 capstone 锁前扩展 in-scope 表 OR 显式记录为 carve-out 推 P9** (per design D6 + spec L100-L103 Scenario)。

---

## design D9 decision branch 判定 (PR-A 阶段初步)

| 分支 | 描述 | PR-A 阶段判定 | 终判 (PR-H 后) |
|---|---|---|---|
| 1 | 诊断结果落在 8 站点 reduction 路径 | **NOT confirmed** by Mac local | TBD |
| 2 | 诊断结果落在 8 站点 reduction 路径**外部** (parallel writer noise) | **PROBABLE** per Stage 1 + Mac local + proposal L3 修订 hypothesis | TBD |
| 3 | 诊断显示 8 站点改造与漂移源无任何因果链 | NOT supported by Stage 1 grep | TBD |

**PR-A 阶段判定结论**: 倾向 **分支 2** (诊断结果落在 8 站点 reduction 路径外部)。
- 推进 §2.x fixed-shape pairwise 改造作为通用 ULP 噪声阻断手段；
- 同时 server PR-K2 二跑 (PR-H stage) SHALL 提供 byte-level dump 跨 N=1/4/8 数据，定位 DY 首发散点；
- 若 server 数据 confirm 分支 2, **SHALL 在 spec R2 表锁前扩展 in-scope** OR 显式 carve-out 推 P9 (per design D6)。
- 若 server 数据 confirm 分支 1 (落在 8 站点内)，§2.x 改造直接闭合，无需 spec 扩展。
- 若 server 数据 confirm 分支 3 (无因果链)，**STOP §2.x 推进**，P1c 范围重新评估，退回 master plan §6 P1c.1 候选 (c) Deterministic OpenMP N_Vector。

---

## Mac local sanity check 复现脚本 (本 PR §1.5 (1) 部分)

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

- [ ] Server cn0X build `shud_omp SHUD_DUMP_RHS=1`，3-grep gate strict FP.
- [ ] heihe + heihe_x4 各 N ∈ {1, 4, 8} = 6 cell `f_applyDY` snapshot dump.
- [ ] `compare_snapshot` 跨 N 对 (heihe N=1 vs N=4 等)，定位首发散点；记入字段 (a)。
- [ ] §2.x fixed-shape pairwise 改造后 post-fix dump (同 case + 同 anchor), 记入字段 (b)。
- [ ] 在每个 reduction 站点入口 + 出口插 instrumented dump (ULP delta probe), 跑 N=2/4/8 各 8 站点 = 24 cell，填字段 (c) ULP delta 表。
- [ ] 基于 server 数据 update 字段 (d) tree-reduction hypothesis confirm/refute。
- [ ] 终判 D9 decision branch 1/2/3, update 本文档 §"design D9 decision branch 判定" 表的 "终判 (PR-H 后)" 列。
- [ ] 若 branch 2 或 branch 3, update spec R2 表 in-scope 扩展 OR 显式 carve-out 推 P9 (per design D6) **在 spec 锁前**完成。
