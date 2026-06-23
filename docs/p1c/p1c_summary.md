# P1c — capstone summary

P1c epic (`P1c.0 ~ P1c.6` per master plan §6) 总结。P1c 阶段 13 sub-issues (#244-#256, PR-A..PR-M) 全部 merged 到 `baseline/P1c` 分支。本文件作 P1c-tag (PR-L #255) + PROMOTE (PR-M #256) + P2a hand-off 的 source of truth。

## §1 完成定义

P1c "deterministic-reduction + capstone" 阶段在以下条件达成时视为 CLOSED:

| 条件 | Requirement | 实测 / 状态 | PR |
|---|---|---|---|
| (a) 8-site canonical-reduction landed | spec `p1c-deterministic-reduction` Requirement: 8 站点 / 10 anchor 全部 helper-wrap (`fixed_pairwise_sum_indexed` / `fixed_leftfold_sum_indexed` / `fixed_leftfold_sum_pair_indexed`) | ✓ COMPLETE | PR-A..PR-E |
| (b) 3 negative grep gates PASS | spec Scenarios L76/L81/L86: 新宏 0 / `schedule(dynamic\|guided)` 0 / `#pragma omp atomic` 0 | ✓ COMPLETE (per `docs/p1c/p1c_summary.md` §6.1) | PR-F |
| (c) 10-anchor coverage table 完整 | spec L63-L66 Scenario: 10 write target × 4 helper mapping | ✓ COMPLETE (per `docs/p1c/p1c_summary.md` §6.2) | PR-F |
| (d) Kahan held-in-reserve patch documented | spec L107-L128 Requirement: conditional Kahan + trigger + apply flow | ✓ COMPLETE (`docs/p1c/p1c_kahan_patch.diff` + a3a §"Kahan 候选路径") | PR-G |
| (e) Server PR-K2 首跑 (8-cell) executed + verdict | spec §4 + design D3 SHALL gate | ✓ COMPLETE → §4.7 trigger FIRED (A3a + nst 双 FAIL) | PR-H |
| (f) §4.7 Kahan injection 二跑 (8-cell) executed + verdict | spec L107-L128 conditional path | ✓ COMPLETE → PARTIAL CLOSURE + P1d carve-out | PR-I |
| (g) NUM_OPENMP=1 reverse-compat documented | spec L132-L142 Requirement | ✓ COMPLETE (pre-Kahan PASS, Kahan-injected FAIL trade-off) | PR-J |
| (h) Capstone docs (≥7 topic) | spec p1c-capstone Requirement | ✓ COMPLETE (本 PR) | PR-K |
| (i) P1c-tag annotated + baseline/P1c lock | spec L150-L160 + design D11 | △ PENDING (PR-L 下一 PR) | PR-L |
| (j) PROMOTE 2 specs + archive + jsonl | spec p1c-capstone "PROMOTE" Scenario | △ PENDING (PR-M 下一 PR) | PR-M |

**closure status**: (a)-(h) 8/8 done; (i)/(j) 顺序推进。**Carve-out scope (推 P9)**:
- ✗ Bit-level A3a cross-N (heihe / heihe_x4 N=1≡N=2 ≠ N=4 ≠ N=8 pattern)
- ✗ nst Δ=0 cross-N (heihe |Δ|=84 post-Kahan; heihe_x4 |Δ|=5)
- ✗ Upstream parallel writer first-touch / NUMA-affinity 治理 (OMP_PROC_BIND/OMP_PLACES + numactl interleave) — P1d stage 独立范围

## §2 旧版错误复盘 — N/A

P1c 为 new epic stage (sub-epic of master plan §3 P1c bucket), 不涉及旧版 P1c 错误复盘。  
但 P1c 阶段过程中触及的相关历史教训:
- B1b stage S5b "rhs serial canonical order" 验收 (PR-1 #191 ~ PR-16 #207) 提供本阶段 8-site 排序依据 (Stage 1 dump 用 B1b 基线源码生成)。
- P1 era PR-K2 #223 catastrophic N≥4 漂移 (heihe nst 6585/6684) 是 P1c motivation 来源。P1c 完成 8-site reduction 后, drift 量级从 P1 era ~6685 降至 P1c PR-I ~6608 (~1% 减小), 但 pattern (N=1≡N=2 ≠ N=4 ≠ N=8) **保留** — 证 drift origin 不在 8 站点内部 (D9 branch 2 CONFIRMED)。

## §3 P1c-tag 处理 (PR-L scope)

- **tag name**: `P1c-tag` (annotated, immutable per design D11)
- **deref SHA**: baseline/P1c HEAD post-PR-K (本 PR 合入后 SHA)
- **SHUD pin (tag deref)**: `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` (Kahan-injected; openmp-baseline pushed)
- **tag annotated message** (per PR-L 草拟):
  ```
  P1c deterministic-reduction + capstone (2026-06-22)
  - 8-site / 10-anchor helper-wrap (PR-B..PR-E) — bitwise neutral at NUM_OPENMP=1
  - Kahan (Neumaier) injection at 4 helpers — improves nst Δ but doesn't close A3a
  - PARTIAL CLOSURE: deterministic-reduction Requirement landed; bit-level A3a + nst Δ=0 carved-out to P1d (upstream writer first-touch / NUMA governance)
  - NUM_OPENMP=1 binary-level reverse-compat NOT preserved due to Kahan injection
  - D11 tag-immutability preserved (P1-update-omp-tag SHA unchanged)
  ```
- **baseline/P1c branch lock** (post-tag):
  - `lock_branch=true`
  - `enforce_admins=true`
  - `allow_force_pushes=false`

## §4 时间线

| 日期 | 事件 | PR / Issue |
|---|---|---|
| 2026-06-22 | PR-A merged: P1c.0 diagnostic + dump + grep | #257 / closes #244 |
| 2026-06-22 | PR-B merged: L278-279 lake pairwise | #258 / closes #245 |
| 2026-06-23 | PR-C merged: L374-375/L382-383 leftfold | #259 / closes #246 |
| 2026-06-23 | PR-D merged: L392 QrivUp leftfold neg | #260 / closes #247 |
| 2026-06-23 | PR-E merged: L406/L420/L433 lake leftfold + pair | #261 / closes #248 |
| 2026-06-23 | PR-F merged: Mac 16-cell scan + 3 grep + coverage | #262 / closes #249 |
| 2026-06-23 | PR-G merged: Kahan held-in-reserve patch | #263 / closes #250 |
| 2026-06-23 | PR-H merged: Server PR-K2 首跑 → §4.7 trigger FIRED | #264 / closes #251 |
| 2026-06-23 | PR-I merged: §4.7 Kahan injection + carve-out | #265 / closes #252 |
| 2026-06-23 | PR-J merged: reverse-compat documented | #266 / closes #253 |
| 2026-06-23 | PR-K merged: 本 PR capstone | #267 / closes #254 |
| 2026-06-23 (next) | PR-L: P1c-tag annotate + lock | / closes #255 |
| 2026-06-23 (next) | PR-M: PROMOTE 2 specs + archive + jsonl | / closes #256 + Epic #243 close |

13 sub-issues + 1 epic = 14 issues 在单日 epic-burst 完成 (含 8-cell server Slurm × 2 跑 = 40 min compute + R-review + 13 squash merges)。

## §5 Hand-off → P2a / P1d

### §5.1 P2a hand-off (next stage)

baseline/P1c HEAD locked as baseline/P2a starting point; SHUD pin `3a0004c` (Kahan-injected) inherited. P2a 范围 (per master plan §3): J0/J1 OMP scheduling refinement + RHS micro-fusion (per master plan §3 P2a row).

P2a entry condition 验证 (PR-L 验证命令 §7):
```bash
# baseline/P1c branch lock 确认
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c --jq '.protection.lock_branch'  # → true
# SHUD pin 与 P1c-tag deref 一致
git ls-tree P1c-tag SHUD --object-only | head -c 40  # → 3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4
# 8-site helper-wrap 全 PRESENT
grep -c 'fixed_pairwise_sum_indexed\|fixed_leftfold_sum_indexed\|fixed_leftfold_sum_pair_indexed' \
  SHUD/src/Model/MD_rhs_core.cpp  # → ≥10
```

### §5.2 P1d carve-out (deferred stage)

P1d stage (per master plan §3 P9 row) 负责:
1. **Upstream parallel writer first-touch / NUMA-affinity 治理**
   - MD_update.cpp 3 #pragma omp parallel for region (hot.soa / QeleSurf_flat / Ele_AoS) 加 `default(none)` + `OMP_PROC_BIND=close`
   - 标准化 `OMP_PLACES=cores` (server) / `OMP_PROC_BIND=close` (Mac)
   - `numactl --interleave=all` (server NUMA 显式)
2. **Bit-level A3a cross-N closure**
   - 验证 P1d NUMA 治理后 heihe / heihe_x4 4 N SHA 全等
   - 若 PASS: Kahan injection 可 revert (`docs/p1c/p1c_kahan_patch.diff` 反向 apply)
3. **nst Δ=0 cross-N closure**
   - 同上 NUMA 治理后 nst 跨 N 不变 (Δ=0)
4. **NUM_OPENMP=1 reverse-compat 恢复**
   - revert Kahan 后 N=1 SHA 应恢复至 `7f22bd6f...` (P1-update-omp-tag canonical) ✓
5. **Mac reframing 二次验证**
   - Mac 16-cell scan post-NUMA-fix 应 PASS (per `docs/p1c/p1c_perf_baseline.md` §1.6 hypothesis)

## §6 capstone 验证结果

### §6.1 三 negative grep gates (per spec Scenarios L76 / L81 / L86) — PR-F PASS

| Gate | Command | Expected | Actual | Verdict |
|---|---|---|---|---|
| (a) 新宏 | `grep -rE 'SHUD_USE_DETERMINISTIC_REDUCTION\|SHUD_DET_REDUCT\|SHUD_PAIRWISE' SHUD/` | 0 hits | 0 hits | **PASS** |
| (b) schedule | `grep -nE 'schedule\(' SHUD/src/Model/MD_rhs_core.cpp` | 0 hits OR static-only | 0 hits | **PASS** |
| (c) atomic | `grep -rn '#pragma omp atomic' SHUD/src/` | 0 hits | 0 hits | **PASS** |

### §6.2 10 line anchors → 8 logical sites coverage — PR-F PASS

per post-PR-E SHUD@de9545d 行号 (current SHUD@3a0004c 行号略变, helper 名称 / 调用结构不变):

| # | 写目标变量 | PR | helper 调用 | 来源 anchor |
|---|---|---|---|---|
| 1 | `qLakeEvap` | PR-B | `fixed_pairwise_sum_indexed(ele_by_lake[i], qEleEvapo_lake)` | L278 |
| 2 | `qLakePrcp` | PR-B | `fixed_pairwise_sum_indexed(ele_by_lake[i], qElePrep_lake)` | L279 |
| 3 | `QrivSurf` | PR-C | `fixed_leftfold_sum_indexed(seg_by_riv[ir], QSurfDown)` | L374 |
| 4 | `QrivSub` | PR-C | `fixed_leftfold_sum_indexed(seg_by_riv[ir], QSubDown)` | L375 |
| 5 | `Qe2r_Surf` | PR-C | `-fixed_leftfold_sum_indexed(seg_by_ele[ie], QSurfDown)` | L382 |
| 6 | `Qe2r_Sub` | PR-C | `-fixed_leftfold_sum_indexed(seg_by_ele[ie], QSubDown)` | L383 |
| 7 | `QrivUp` | PR-D | `-fixed_leftfold_sum_indexed(upstream_by_down[ir], QrivDown)` | L392 |
| 8a | `QLakeRivIn` | PR-E | `fixed_leftfold_sum_indexed(riv_in_by_lake[ilake], QrivDown)` | L406 |
| 8b | `QLakeSurf` | PR-E | `fixed_leftfold_sum_pair_indexed(lake_bank_edge_by_lake[ilake], QeleSurf_lake, 3)` | L420 |
| 8c | `QLakeSub` | PR-E | `fixed_leftfold_sum_pair_indexed(lake_bank_edge_by_lake[ilake], QeleSub_lake, 3)` | L433 |

10 write targets → 8 logical sites (8a/8b/8c 合并 lake gathers 组, per spec Conventions §"站点计数定义").

### §6.3 Server PR-K2 首跑 (8-cell) — PR-H FAIL → §4.7 trigger fired

8-cell heihe + heihe_x4 × N∈{1,2,4,8}:
- **§4.4 A3a bitwise**: heihe ✗ FAIL (3 distinct SHAs); heihe_x4 ✗ FAIL (3 distinct SHAs)
- **§4.5 nst**: heihe {6773, 6773, 6682, 6548}, |Δ|=225 ≫ D9 ≤2; heihe_x4 {6571, 6571, 6568, 6570}, |Δ|=3 > 2
- **§4.7 trigger**: SHALL TRIGGER → PR-I conditional Kahan injection

详 `docs/p1c/p1c_pr_h_server_first_run.md` §3-§7.

### §6.4 §4.7 Kahan injection 二跑 (8-cell) — PR-I PARTIAL CLOSURE

8-cell heihe + heihe_x4 × N∈{1,2,4,8} on SHUD@3a0004c (Kahan-injected):
- **A3a**: 仍 ✗ FAIL (3 distinct SHAs pattern 保留)
- **nst delta**: heihe |Δ| 225 → 84 (~63% 改善); heihe_x4 |Δ| 3 → 5 (噪声幅度)
- **Δ_wall**: 意外改善 (heihe_x4 N=8 -22.9%; heihe_x4 N=4 -15.2%; heihe_x4 全 cells 负 Δ; heihe N=1 -4.2%)。R2 +1-3% perf 降估算 REFUTED — Kahan 改 CVODE 收敛路径 → 减 ncfl linear-solve failure → 抵消 Neumaier overhead

详 `docs/p1c/p1c_pr_i_kahan_injection.md` §3-§7。

### §6.5 design D9 decision branch 终判

| 分支 | 描述 | 终判 (PR-I 后) |
|---|---|---|
| 1 | drift origin IN 8 sites | **REFUTED** (Kahan 注入 8 sites 仍 残 |Δ|=84) |
| 2 | drift origin OUTSIDE 8 sites (writer noise) | **CONFIRMED** |
| 3 | 8-site reduction 无因果链 | **PARTIALLY REFUTED** (8 sites 作 noise amplifier, ~63% 减 heihe Δ_nst) |

详 `docs/p1c/p1c_a3a_root_cause.md` §"design D9 decision branch 判定" 终判表。

### §6.6 NUM_OPENMP=1 reverse-compat — PR-J PARTIAL

| 视角 | SHUD pin | N=1 vs P1-tag canonical | 状态 |
|---|---|---|---|
| Pre-Kahan (helper-wrap only) | `de9545d` | byte-identical | ✓ PASS |
| Kahan-injected (current baseline/P1c) | `3a0004c` | byte-different (trade-off) | ✗ FAIL |

详 `docs/p1c/p1c_pr_j_reverse_compat.md` §2-§4。

### §6.7 Mac reframing (RISK-26 NUMA 二次观察)

PR-F Mac 16-cell (M4 Pro 4P+10E + libomp 弱绑定) + Server PR-H/PR-I (Intel/AMD 多核同质) 均显 **同** N=1≡N=2 ≠ N=4 ≠ N=8 pattern → Mac 不再是 D7 "pass-while-server-fails" 非典型 case, 而是 server pattern 的 early signal (RISK-26 NUMA / cache locality 共享敏感性)。D7 SHOULD/SHALL trigger 不对称仍保留 (Mac informational only, server PR-K2 是 SHALL gate)。

### §6.8 Mac N=1 SHALL Scenario (spec L154-157) — DEFERRED

spec `p1c-deterministic-reduction` L154-157 定义 SHALL Scenario "Mac N=1 反向兼容: 4 Mac case 与 P1-update-omp-tag Mac canonical SHA bitwise"。本 PR 调查发现:
- PR-F 已采集 4 Mac case × NUM_OPENMP=1 rivqdown.dat SHA at SHUD@de9545d (pre-Kahan):
  - keliya N=1 = `b23e15b9...` (per `docs/p1c/p1c_perf_baseline.md` §1.3 L40)
  - xinanjiang_upstream N=1 = `90eeb9c6...` (L44)
  - qinyijiang N=1 = `0f8c3fec...` (L48)
  - qhh N=1 = `8a6d9b2c...` (L52)
- P1-update-omp-tag Mac canonical **rivqdown.dat** SHA 未直接 archived 在 P1 era docs (`docs/p1_summary.md` §"§4 Mac canonical SHA" 表 + `docs/p1_fullrun_bitwise.md` §3 表均报告的是 archive_b0_output.sh **summary SHA**, 非单文件 rivqdown.dat SHA — file artifact 不同)。
- 已知架构等式 (server PR-J §2 实证): 8-site helper-wrap 在 NUM_OPENMP=1 (serial) bitwise-equivalent (P1 era rivqdown SHA == P1c pre-Kahan rivqdown SHA at server)。**理论上** Mac 同样应满足此等式, 但缺乏 P1 era Mac N=1 rivqdown.dat SHA 直接 reference, 不能字面验证。
- **决定**: 将 spec L154-157 Mac SHALL Scenario 标 DEFERRED, 推 P1d stage 同时:
  1. 在 P9 stage 重 P1-update-omp-tag binary 回 Mac 跑 NUM_OPENMP=1 → 4 case rivqdown.dat SHA, archive 进 `docs/p1_perf_baseline.md` 或 `docs/p1d/p1d_*.md`;
  2. P1d NUMA 治理后, 用 P1c Kahan binary OR pre-Kahan binary 跑同 4 case → 与 step (1) 比对;
  3. 若 pre-Kahan PASS (期望, 同 server PR-J §2): 证 Mac architecture 同 server bit-equivalent at serial; spec L154-157 Scenario 闭。

P1c epic 阶段不阻塞 — Mac N=1 reverse-compat 不是 capstone 二跑 SHALL gate (per design D7 Mac informational only + server PR-K2 唯一 SHALL gate)。Status matrix P1c 行 Mac 列标 "PARTIAL @ Mac (pre-Kahan only)" — pre-Kahan N=1 数据存档 (per PR-F + 本节 SHA citation), Kahan-injected 未跑 Mac 16-cell (R4 PR-I capstone 二跑 server-only)。

## §7 P1c-tag 验证命令

(由 PR-L #255 执行)

```bash
# §7.1 P1c-tag annotated 创建
git tag -a P1c-tag \
  -m "$(cat <<'EOF'
P1c deterministic-reduction + capstone (2026-06-22)

- 8-site / 10-anchor helper-wrap (PR-B..PR-E) — bitwise neutral at NUM_OPENMP=1
- Kahan (Neumaier) injection at 4 helpers — improves nst Δ but doesn't close A3a
- PARTIAL CLOSURE: deterministic-reduction Requirement landed; bit-level A3a +
  nst Δ=0 carved-out to P1d (upstream writer first-touch / NUMA governance)
- NUM_OPENMP=1 binary-level reverse-compat NOT preserved due to Kahan injection
- D11 tag-immutability preserved (P1-update-omp-tag SHA unchanged)
EOF
)" \
  <baseline/P1c HEAD SHA post-PR-K>
git push origin P1c-tag

# §7.2 baseline/P1c branch lock
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c/protection \
  --method PUT --field lock_branch=true \
  --field enforce_admins=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_pull_request_reviews=null \
  --field required_status_checks=null \
  --field restrictions=null

# §7.3 D11 immutability 验证
git tag --verify P1c-tag  # → "object <sha> immutable annotated tag"
git rev-parse P1c-tag^{commit}  # → baseline/P1c HEAD SHA post-PR-K
git ls-tree P1c-tag SHUD | grep -oE '[0-9a-f]{40}'  # → 3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4

# §7.4 历史 tag SHAs 不变验证 (D11 永不变)
for tag in P1-update-omp-tag B1-tag B1a-tag B1b-tag; do
  echo "$tag: $(git rev-parse $tag 2>/dev/null)"
done
# 应输出与 P1 stage 完成时刻 (pre-P1c) 完全一致的 SHAs
```

## §8 反向兼容判定 (per PR-J)

P1c epic 结束时 baseline/P1c HEAD (SHUD@3a0004c) NUM_OPENMP=1 binary SHA 与 P1-update-omp-tag canonical SHA 关系:

| Layer | 状态 | 解释 |
|---|---|---|
| Tag SHA immutability (D11) | ✓ PRESERVED | P1-update-omp-tag annotated tag SHA 永不变 |
| Binary runtime SHA at N=1 | ✗ DIVERGED | Kahan injection changes acc order |
| Helper-wrap layer at N=1 (pre-Kahan baseline) | ✓ EQUIVALENT | de9545d N=1 SHA 与 P1-update-omp-tag canonical SHA byte-identical (PR-J §2) |

**Trade-off accepted** (per master plan §3 fallback option 2 + spec L100-L103 carve-out Scenario): 为换 heihe Δ_nst 225 → 84 (~63% 改善), 牺牲 N=1 binary-level reverse-compat。Kahan patch held-in-reserve (`docs/p1c/p1c_kahan_patch.diff`), 可 P9 stage NUMA 治理后回退 + N=1 reverse-compat 恢复。

## §9 限制与已知问题

1. **N≥4 bit-level A3a 仍 FAIL** — P1c 阶段未关闭, 推 P1d (per §1 carve-out scope + §5.2 P9 hand-off)。
2. **NUM_OPENMP=1 binary-level reverse-compat 在 Kahan-injected baseline/P1c HEAD 上 FAIL** — D11 tag-level immutability 保留, 但 runtime binary 跨 P1→P1c 阶段不再 byte-identical。
3. **heihe_x4 Δ_wall 异常改善** — Kahan-injected wall 比 pre-Kahan 减少 15-23%, 与 R2 +1-3% perf 降估算相反。需 PR-K capstone 后 P2a/P1d 阶段 cross-check P1 era PR-K1 wall baseline 是否在同 hardware platform 上同样幅度。
4. **Mac D7 framing 部分过时** — spec L113 "Mac snapshot 已知 pass-while-server-fails per design D7" 在 P1c PR-F/PR-H/PR-I 经验上不成立 (Mac + server 同 fail-pattern)。spec 文本保留, PR-M PROMOTE 时由 reviewer 决定是否在 archive 标 "P1c era 经验补充"。
5. **8935-8942 sbatch 数据丢失** — PR-I 操作记录, 首次 sbatch template 含 `${ROOT}` 变量未被 sed 替换, scancel 前 `rm -rf ${RUN}` 已执行 → `.p1c-runs/` 首跑 scratch 销毁。源真值已固化在 PR-H 文档, 重跑 8943-8950 修复后无影响。

## §10 引用来源 (source of truth)

| 文档 | 内容 |
|---|---|
| [`docs/p1c/p1c_summary.md`](p1c_summary.md) | 本文件, P1c capstone source of truth |
| [`docs/p1c/p1c_a3a_root_cause.md`](p1c_a3a_root_cause.md) | A3a 根因分析 + decision branch + Kahan 候选路径 + 终判 |
| [`docs/p1c/p1c_perf_baseline.md`](p1c_perf_baseline.md) | Mac 16-cell + Server PR-H/PR-I 8-cell wall + 数据 |
| [`docs/p1c/p1c_reduction_sites.md`](p1c_reduction_sites.md) | 8-site reduction overview |
| [`docs/p1c/p1c_b1b_serial_order_dump.txt`](p1c_b1b_serial_order_dump.txt) | B1b 基线 serial order dump |
| [`docs/p1c/p1c_pr_h_server_first_run.md`](p1c_pr_h_server_first_run.md) | PR-H server first run 8-cell raw data |
| [`docs/p1c/p1c_pr_i_kahan_injection.md`](p1c_pr_i_kahan_injection.md) | PR-I Kahan 二跑 8-cell raw data + carve-out 决策 |
| [`docs/p1c/p1c_pr_j_reverse_compat.md`](p1c_pr_j_reverse_compat.md) | PR-J reverse-compat 双视角 |
| [`docs/p1c/p1c_kahan_patch.diff`](p1c_kahan_patch.diff) | Kahan held-in-reserve patch (PR-G 输出 + PR-I 应用) |
| [`docs/status_matrix.md`](status_matrix.md) | 阶段 × benchmark 状态矩阵 (P1c 行待 §11 add) |
| `openspec/changes/p1c-deterministic-reduction/specs/p1c-deterministic-reduction/spec.md` | P1c deterministic-reduction Requirement 来源 |
| `openspec/changes/p1c-deterministic-reduction/specs/p1c-capstone/spec.md` | P1c capstone Requirement 来源 |
| `openspec/changes/p1c-deterministic-reduction/design.md` | 设计文档 D1-D11 决策 + 风险 R1-R3 + NG1-NG5 |
| `SHUD_openMP_master_plan.md` | master plan §3 P1c row + §4 P1c.2 success gate + §6 stage hand-off |
