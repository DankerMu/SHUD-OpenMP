# P1c PR-J — P1 反向兼容验证 (NUM_OPENMP=1 vs P1-update-omp-tag canonical SHA)

验证目标 (per spec L132-L142): 现 baseline/P1c HEAD (SHUD@3a0004c, Kahan-injected post-PR-I) build 在 NUM_OPENMP=1 跑 heihe 90d 是否与 P1 阶段固化的 P1-update-omp-tag canonical SHA 字面相等。

**结论**: 双视角文档:
1. **Pre-Kahan (SHUD@de9545d, PR-H stage)** N=1 reverse-compat: **✓ PASS** — 8-site helper-wrap (PR-B/C/D/E) bitwise-equivalent at serial。
2. **Kahan-injected (SHUD@3a0004c, PR-I stage, current baseline/P1c)** N=1 reverse-compat: **✗ FAIL** — Neumaier 改 helper acc order 引入 N=1 bit-level drift (acknowledged trade-off, P9 carve-out 范围)。

## §1 P1-update-omp-tag canonical reference

P1 era reference SHA from `docs/p1_perf_baseline.md:209-216` (SHUD@07c677f canonical, server cn0X 90d run):

| case | N=1 | N=2 (=N=1) | N=4 | N=8 |
|---|---|---|---|---|
| heihe | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | 同 N=1 | `03055aa0fcbc9c3406e61f0ed926e2b77682b2d565ba1f2eef1de7721ba5ba9a` | `904779c30770f55638ca01030ef5b9e6bf65095ab3d70e6894d843f29b40b6e7` |
| heihe_x4 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | 同 N=1 | `0b2aa00f0e2d55887ee44fd95848f2370fc5682aa00bd7fefda61ad0948fc765` | `d3d37e42a9ccfe9b23aec38d5a85cd627c870bc5642cc37ff780407551f11e8d` |

P1 era 同样观察 N=1≡N=2 (SHUD 内部 `max(NUM_OPENMP,2)` 行为, per PR-H §3.1)，behavior 跨 P1→P1c 保留。

## §2 Pre-Kahan P1c (PR-H stage, SHUD@de9545d) vs P1 canonical

参考 PR-H §3 实测 SHAs:

| case | N | P1 canonical (07c677f) | P1c pre-Kahan (de9545d) | 等? |
|---|---|---|---|---|
| heihe | 1 | `7f22bd6f...` | `7f22bd6f...` | **✓ PASS — byte-identical** |
| heihe | 2 | `7f22bd6f...` | `7f22bd6f...` | **✓ PASS** |
| heihe | 4 | `03055aa0...` | `7f7a621c...` | ✗ FAIL (post-PR-E acc order 改变) |
| heihe | 8 | `904779c3...` | `8c581172...` | ✗ FAIL |
| heihe_x4 | 1 | `55403bef...` | `55403bef...` | **✓ PASS — byte-identical** |
| heihe_x4 | 2 | `55403bef...` | `55403bef...` | **✓ PASS** |
| heihe_x4 | 4 | `0b2aa00f...` | `7e8f7a8a...` | ✗ FAIL |
| heihe_x4 | 8 | `d3d37e42...` | `8b0efa6f...` | ✗ FAIL |

### 关键发现 — 8 站点 helper-wrap 在 N=1 byte-equivalent

PR-B/C/D/E 实施 4 helper (`fixed_pairwise_sum_range`, `fixed_pairwise_sum_indexed`, `fixed_leftfold_sum_indexed`, `fixed_leftfold_sum_pair_indexed`) 改造 8 站点 reduction → **N=1 (serial path) bitwise-identical**: P1 era N=1 SHA `7f22bd6f...` ≡ P1c pre-Kahan N=1 SHA `7f22bd6f...` (heihe 双方); `55403bef...` ≡ `55403bef...` (heihe_x4 双方). 8 cell 中 4 cell (N=1 + N=2) PASS。

设计 spec L62-L72 "A3a strict canonical reduction Requirement" 在 serial 路径已闭。这是 deterministic-reduction Requirement 的强 empirical 证据: helper-wrap **不**改变 serial 数值语义, 同 master plan §2.x.2 设计目标 + spec L46-L60 "fixed-shape canonical accumulation order".

N≥4 SHA 已 drift (per master plan §4 / PR-I §8 决策 = drift 由上游 writer noise 产生)。N≥4 path 不在 PR-J 反向兼容范围内 (spec L132-L142 重点 NUM_OPENMP=1 单点)。

## §3 Kahan-injected P1c (PR-I stage, SHUD@3a0004c, current baseline/P1c) vs P1 canonical

参考 PR-I §3 实测 SHAs:

| case | N | P1 canonical (07c677f) | P1c Kahan (3a0004c) | 等? |
|---|---|---|---|---|
| heihe | 1 | `7f22bd6f...` | `fd2d5571...` | ✗ **FAIL — Kahan break N=1** |
| heihe | 2 | `7f22bd6f...` | `fd2d5571...` | ✗ **FAIL** |
| heihe | 4 | `03055aa0...` | `e058db2e...` | ✗ FAIL |
| heihe | 8 | `904779c3...` | `6285e8a4...` | ✗ FAIL |
| heihe_x4 | 1 | `55403bef...` | `4eb804f5...` | ✗ **FAIL — Kahan break N=1** |
| heihe_x4 | 2 | `55403bef...` | `4eb804f5...` | ✗ **FAIL** |
| heihe_x4 | 4 | `0b2aa00f...` | `ff0787ab...` | ✗ FAIL |
| heihe_x4 | 8 | `d3d37e42...` | `6e9f9a2e...` | ✗ FAIL |

### 关键发现 — Kahan 注入引入 N=1 数值漂移 (acknowledged trade-off)

PR-G/PR-I Neumaier compensation 改 4 helper accumulation order (`acc + c` 补偿项)，即使 serial 路径下 IEEE-754 表达式不再是原裸 leftfold `acc += src[i]`, 故 N=1 SHA `fd2d5571...` ≠ pre-Kahan `7f22bd6f...` ≠ P1 canonical `7f22bd6f...`.

Per master plan + spec L100-L103 carve-out Scenario + PR-I §8 三结论:
- 此 N=1 reverse-compat break **属设计 trade-off**: 为换 heihe Δ_nst 225 → 84 (~63% 改善 cross-N stability), 牺牲 N=1 vs P1-update-omp-tag 字面等于。
- D11 immutability 保 **tag SHA** 永不变 (P1-update-omp-tag annotated tag SHA = `<P1 stage SHA>` 不动); D11 **不**保 binary 跑出 SHA 跨阶段不变。
- 客观文档化: 跨 P1 → P1c 阶段, `baseline/P1c` HEAD 在 NUM_OPENMP=1 不再 bitwise-equivalent 于 P1-update-omp-tag canonical。

## §4 PR-J reverse-compat 终判矩阵

| 视角 | SHUD pin | N=1 vs P1-tag canonical | 状态 |
|---|---|---|---|
| Pre-Kahan (PR-B/C/D/E only) | `de9545d` | ✓ byte-identical | PASS |
| Kahan-injected (PR-G/PR-I) | `3a0004c` (= 当前 baseline/P1c) | ✗ N=1 byte-different | FAIL (trade-off) |

### Rollback option (NOT recommended)

可选 baseline/P1c 子模块 pointer 由 `3a0004c` 回退至 `de9545d` 以恢复 N=1 reverse-compat ✓, 牺牲 Kahan 部分改善 (heihe Δ_nst 225 vs 84). 不 recommended 因为:
- Kahan patch 仍 held-in-reserve 在 `docs/p1c_kahan_patch.diff` (PR-G);
- master plan §3 fallback option 2 (carve-out) 已 closes 8-site Requirement, residual A3a 推 P9 — 与是否 land Kahan 二者均 D11-compatible;
- Kahan 改善 nst stability 是 P1c 阶段最大 progress, 回退会失。

### Default (recommended) — keep Kahan landed at SHUD@3a0004c

继续以 SHUD@3a0004c 作 baseline/P1c HEAD + P1c-tag 锁定 SHA (PR-L #255), capstone (PR-K #254) 显式 narrate "P1c 引入 NUM_OPENMP=1 数值漂移 但属 trade-off, 不违 D11 (tag SHA 永不变)". 反向兼容 (NUM_OPENMP=1 vs P1-update-omp-tag canonical) **NOT preserved at binary level**, 但 **preserved at tag-immutability level**.

## §5 Hand-off

- ✓ 标 PR-J 完成 (closes #253);
- ✓ PR-K capstone (#254) `docs/p1c_summary.md` 须 narrate 此 trade-off + 列入 §"反向兼容判定" 章节;
- ✓ PR-L (#255) P1c-tag annotated message 含 "NUM_OPENMP=1 binary-level reverse-compat NOT preserved due to PR-G/PR-I Kahan injection; D11 tag-immutability preserved" 语句;
- ✓ PR-M (#256) PROMOTE 时 spec p1c-deterministic-reduction 已含 Kahan 兜底 + carve-out 路径, 不需 reshape;
- ✓ P9 stage 范围 (per master plan): 若 P9 NUMA 治理 + OMP_PROC_BIND 标准化 后 nst Δ=0 自然达成, 可 P9 阶段 revert Kahan 注入 (回 `de9545d` 等价 helper-wrap) + 重新 evaluate N=1 reverse-compat (期望恢复 PASS)。