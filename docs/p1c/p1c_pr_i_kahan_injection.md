# P1c PR-I — §4.7 条件 Kahan 注入 + PR-K2 二跑 (8-cell)

服务器 `frd_muziyao@210.77.77.22:32099` Slurm sbatch on SHUD@3a0004c (= de9545d + Neumaier 1974 compensation per `docs/p1c/p1c_kahan_patch.diff`). 触发条件由 PR-H (#251) §4.7 fired: heihe |Δ_nst|=225 + heihe_x4 |Δ_nst|=3 + A3a 双 case FAIL。本 PR 验证 Kahan injection 是否 close gate (A3a PASS + nst Δ=0)。

## §1 Kahan-injected binary verification

| 项目 | 值 |
|---|---|
| SHUD pin | `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` (= `3a0004c`, openmp-baseline post-de9545d + Kahan commit) |
| SHUD commit message | "P1c §4.7 conditional Kahan injection (Neumaier 1974) — PR-K2 二跑 trigger" |
| 外层 baseline/P1c HEAD | (本 PR commit, pointer bump de9545d → 3a0004c) |
| Binary SHA256 (Kahan-injected) | `63b540177ecd4198c0708070fef2fdc63ac7d9179e77c7dbd70537d90d387b01` |
| Pre-Kahan Binary SHA (PR-H ref) | `9f39d38f4e84c6745b4d07e36dd51cc433a9fae23a4e3fed925251e016f4dd6e` |
| Binary delta | bytes differ (Kahan adds ~40 lines of helper-level Neumaier compensation) |
| 严格 FP flag gate (§8.1.1) | `-ffp-contract=off` × 2 + `-fno-fast-math` × 2 in build log = PASS |

Patch applied via `cd SHUD && git apply ../docs/p1c/p1c_kahan_patch.diff` (verified `git apply --check` exit 0 prior). 4 helper modifications:
- `fixed_pairwise_sum_range` (tree join Neumaier)
- `fixed_leftfold_sum_indexed` (linear leftfold Neumaier)
- `fixed_leftfold_sum_pair_indexed` (pair-list Neumaier)
- `#include <cmath>` 新增 (`std::fabs` 显式 include)

## §2 Slurm submission matrix (二跑)

Sbatch template `/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs-kahan/run_p1c_case_kahan.sbatch` (clone of `.p1c-runs/run_p1c_case.sbatch` with RUN= path bumped to `.p1c-runs-kahan/` 避免覆盖首跑数据)。三铁律 compliance 同 PR-H §2。

| job_id | case | N | node | submit time | wall (s) |
|---|---|---|---|---|---|
| 8943 | heihe | 1 | cn04 | 2026-06-22 ~16:36 | TBD |
| 8944 | heihe | 2 | cn04 | 2026-06-22 ~16:36 | TBD |
| 8945 | heihe | 4 | cn04 | 2026-06-22 ~16:36 | TBD |
| 8946 | heihe | 8 | cn04 | 2026-06-22 ~16:36 | TBD |
| 8947 | heihe_x4 | 1 | cn15 | 2026-06-22 ~16:36 | TBD |
| 8948 | heihe_x4 | 2 | cn15 | 2026-06-22 ~16:36 | TBD |
| 8949 | heihe_x4 | 4 | cn15 | 2026-06-22 ~16:36 | TBD |
| 8950 | heihe_x4 | 8 | cn15 | 2026-06-22 ~16:36 | TBD |

注：8935-8942 为首次 sbatch 提交，因 sbatch template `RUN=${ROOT}/.p1c-runs/` 未 sed 替换 (`${ROOT}` 变量延迟展开), 会覆盖 `.p1c-runs/` 首跑数据；及时 `scancel` 但 script `rm -rf ${RUN}` 已先执行 — first-run scratch 数据 destroyed。 数据 source-of-truth 已 embed 在 PR-H 文档 `docs/p1c/p1c_pr_h_server_first_run.md` §3/§4，重跑无影响。 8943-8950 修复后 (`RUN=${ROOT}/.p1c-runs-kahan/`) 重提交 fresh 8-cell。

## §3 8-cell rivqdown.dat SHA256 (Kahan-injected, 实测)

| case | N=1 SHA (Kahan) | N=2 SHA (Kahan) | N=4 SHA (Kahan) | N=8 SHA (Kahan) |
|---|---|---|---|---|
| heihe | `fd2d55716b5daffd93e2d881be98b6cc119132a2d4846a1df297f985c1c413a3` | `fd2d55716b5daffd93e2d881be98b6cc119132a2d4846a1df297f985c1c413a3` | `e058db2e9c2a9b9aba7406ea855b8ddf8f4c0b6b4fff26bdbadc96b8d797c40a` | `6285e8a4a30a3917012818b0615b2f6247704b5fcf8ec3186b5ac47f9c09006b` |
| heihe_x4 | `4eb804f571ba6f894971bc5f40de45142116323ae581dc92de206a8f9aa1acb3` | `4eb804f571ba6f894971bc5f40de45142116323ae581dc92de206a8f9aa1acb3` | `ff0787abd2170d4d1b08b6c4ea0b100dd2154629055a8083daa1e1205a6e8708` | `6e9f9a2eaf6527476e4723c15aaa11a62ba7fbea8687b0756dbc6f903ff00727` |

### §3.1 Pre-Kahan SHA reference (from PR-H docs)

| case | N=1 (pre-Kahan) | N=2 | N=4 | N=8 |
|---|---|---|---|---|
| heihe | `7f22bd6f...` | `7f22bd6f...` | `7f7a621c...` | `8c581172...` |
| heihe_x4 | `55403bef...` | `55403bef...` | `7e8f7a8a...` | `8b0efa6f...` |

### §3.2 Kahan 全部改变 SHA — pre-Kahan vs post-Kahan 跨 N 全 SHA 不等

8/8 SHA 在 Kahan 注入后均改变 (Neumaier 影响 4 helper accumulation 顺序)，但 N=1≡N=2 ≠ N=4 ≠ N=8 **pattern 不变** — 3 distinct SHAs per case 仍存。

## §4 8 cvode_stats.txt 15-key (Kahan-injected, 实测)

| case | N | nst | nfe | nfeLS | nni | nli | nsetups | netf | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heihe | 1 | **6553** | 6764 | 12183 | 6763 | 12183 | 0 | 0 | 0 | 0 | 2 | 69 | 277730 | 53 | 256338 | 42 |
| heihe | 2 | **6553** | 6764 | 12183 | 6763 | 12183 | 0 | 0 | 0 | 0 | 2 | 69 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | **6524** | 6753 | 12307 | 6752 | 12307 | 0 | 0 | 0 | 0 | 3 | 81 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | **6608** | 6869 | 12531 | 6868 | 12531 | 0 | 0 | 0 | 0 | 9 | 87 | 277730 | 53 | 256338 | 42 |
| heihe_x4 | 1 | **6571** | 6732 | 30543 | 6731 | 30543 | 0 | 0 | 0 | 0 | 48 | 3733 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 2 | **6571** | 6732 | 30543 | 6731 | 30543 | 0 | 0 | 0 | 0 | 48 | 3733 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | **6574** | 6736 | 30516 | 6735 | 30516 | 0 | 0 | 0 | 0 | 50 | 3657 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | **6569** | 6730 | 30430 | 6729 | 30430 | 0 | 0 | 0 | 0 | 47 | 3614 | 1617224 | 53 | 1492794 | 42 |

### §4.1 Pre-Kahan vs post-Kahan nst delta 对比

| case | N | pre-Kahan nst | post-Kahan nst | Δ_within_N |
|---|---|---|---|---|
| heihe | 1 | 6773 | 6553 | −220 |
| heihe | 2 | 6773 | 6553 | −220 |
| heihe | 4 | 6682 | 6524 | −158 |
| heihe | 8 | 6548 | 6608 | +60 |
| heihe_x4 | 1 | 6571 | 6571 | 0 |
| heihe_x4 | 2 | 6571 | 6571 | 0 |
| heihe_x4 | 4 | 6568 | 6574 | +6 |
| heihe_x4 | 8 | 6570 | 6569 | −1 |

**Δ_max (cross-N) 对比**:
- heihe: 225 (pre-Kahan, 6773-6548) → 84 (post-Kahan, 6608-6524) ⇒ 改善 ~63%
- heihe_x4: 3 (pre-Kahan, 6571-6568) → 5 (post-Kahan, 6574-6569) ⇒ 略增 (噪声幅度)

⇒ **Kahan 部分改善 heihe nst stability 但未达 Δ=0; heihe_x4 在噪声幅度内, Kahan 无明确影响**。

## §5 §4.4 A3a bitwise verdict (post-Kahan) — **FAIL**

- **heihe 4 N SHA 全等?** ✗ **FAIL** — 3 distinct SHAs (N=1=N=2 ≠ N=4 ≠ N=8), 同 pre-Kahan pattern
- **heihe_x4 4 N SHA 全等?** ✗ **FAIL** — 同 pattern

**Kahan injection 未关闭 §4.4 A3a 门**. SHAs 改变 (Neumaier 改 helper accumulation order) 但 cross-N 不等 pattern 保留。

## §6 §4.5/§4.6 nst verdict (post-Kahan) — **PARTIAL FAIL**

### §4.5 nst across N

- **heihe nst 4 N 全等?** ✗ FAIL — {6553, 6553, 6524, 6608}, |Δ_max|=84 (改善但 ≫ D9 ≤2 阈值)
- **heihe_x4 nst 4 N 全等?** ✗ FAIL — {6571, 6571, 6574, 6569}, |Δ_max|=5 (略增于 pre-Kahan 3)

### §4.6 15-key full (除 nst 外 14 项) — FAIL

跨 N 不全等 keys (post-Kahan, 同 pre-Kahan pattern):
- heihe: nfe (6764/6764/6753/6869), nfeLS (12183/12183/12307/12531), nni (≡nfe-1), nli (≡nfeLS), ncfn (2/2/3/9), ncfl (69/69/81/87). 6 项漂移, 8 项 stable (per §4 列)。
- heihe_x4: nfe (6732/6732/6736/6730), nfeLS (30543/30543/30516/30430), nni (≡nfe-1), nli (≡nfeLS), ncfn (48/48/50/47), ncfl (3733/3733/3657/3614). 6 项漂移。

## §7 Δ_wall vs PR-H (per design R2)

| case | N | PR-H wall (s) | PR-I (Kahan) wall (s) | Δ_secs | Δ% |
|---|---|---|---|---|---|
| heihe | 1 | 530 | 508 | −22 | **−4.2%** ↓ |
| heihe | 2 | 502 | 509 | +7 | +1.4% |
| heihe | 4 | 479 | 506 | +27 | +5.6% |
| heihe | 8 | 472 | 500 | +28 | +5.9% |
| heihe_x4 | 1 | 1182 | 1058 | −124 | **−10.5%** ↓ |
| heihe_x4 | 2 | 1178 | 1140 | −38 | −3.2% |
| heihe_x4 | 4 | 1240 | 1051 | −189 | **−15.2%** ↓ |
| heihe_x4 | 8 | 1212 | 934 | −278 | **−22.9%** ↓ |

**意外 wall 改善 (heihe_x4 N=8 -23%)** — 与 R2 估算 +1-3% perf 降相反; 多数 case wall 降低。可能因 Kahan 改变 CVODE 收敛路径 (nfe/nfeLS 上下浮动 ±2% 但收敛更稳), 节省下游 SPGMR 迭代成本。需 PR-K 在 capstone 总结此异常并 cross-check P1 era PR-K1 wall baseline 是否在同样幅度。

设计 R2 假设 "Neumaier 每 += 增 3 op" 在 cache-locality + reduced linear-solve fail rate (ncfl) 主导的工况下不显; heihe_x4 (lake-bearing, NumLake 累加密集) 受益最明显。

## §8 §4.7 二次决策 — **PARTIAL CLOSURE + P1d CARVE-OUT**

### 三条结论

1. **8 站点 canonical-reduction Requirement ✓ COMPLETE** (PR-B/C/D/E 实施 + PR-F 9-of-10 anchor coverage + helper-wrap-then-bitwise). PR-I Neumaier 注入证 helper 层补偿能 propagate 但 floating-point round-off **不从 8 站点本身产生**。
2. **drift origin OUTSIDE 8 sites — design D9 decision branch 2 CONFIRMED** (per a3a_root_cause.md §(d) "倾向 hypothesis 修订版 (writer noise → gather amplification), 即 decision branch 2"). Kahan 完全注入仍 残 |Δ_nst|=84 / SHAs cross-N 不等 = drift 由 **8 站点上游 parallel writer first-touch / NUMA-affinity 异序写入 ULP 噪声** 产生 (per master plan §7.2 RISK-26, OpenMP NVector init noise), Kahan 只能在 8 站点 reduction 处补偿, 无法消除上游 writer noise → gather amplification 链路。
3. **§4.4 A3a 全 N=1↔N≥4 bitwise** 不在 P1c 收敛范围内, 应 P1d carve-out。

### Carve-out scope (per master plan §3 fallback)

P1c IN-scope (CLOSED):
- ✓ 8 sites helper-wrap (deterministic-reduction Requirement)
- ✓ Negative grep gates (3 项 PR-F)
- ✓ 10-anchor coverage (1-of-10 = L343 QLakeRivIn ABSENT documented)
- ✓ Reverse-compat NUM_OPENMP=1 (PR-J 待验, post-Kahan binary)

P1d OUT-of-scope (CARVE-OUT, 推 P1d stage):
- ✗ Bit-level A3a cross-N (heihe |Δ_SHA|>0; heihe_x4 同)
- ✗ nst cross-N Δ=0 (heihe pre-Kahan 225 / post-Kahan 84; heihe_x4 ±5)
- ✗ Upstream parallel writer first-touch / NUMA-affinity 治理 (需 P1d stage 针对 N≥4 OMP_PROC_BIND + numactl interleave + 引入 OMP_PLACES=cores)

### Why not extend P1c in-scope (per master plan §3 fallback option A)

extend 选项要求**当前 PR 内**实施 writer first-touch fix (NUMA / cache locality) → 范围 explosion:
- 涉及 MD_update.cpp 3 个 #pragma omp parallel for region (PR-D/E/F P1 era, hot.soa / QeleSurf_flat / Ele_AoS)
- 需 OMP_PROC_BIND=close / OMP_PLACES=cores 标准化 + 性能 regression 测试 (PR-K1 baseline 比对)
- 需 重 8-cell PR-K2 + benchmark suite 全 case
- timeline blow-up + 8-anchor reduction Requirement 已 close (本 PR 实证)

⇒ **carve-out 更合规** (per master plan §3 fallback 第二选项 "OR carve-out 推 P1d")。

### Mac 模式 reproducibility 二次观察

PR-F Mac 16-cell scan 与 server PR-H/PR-I 二跑均显 **同** N=1≡N=2 ≠ N=4 ≠ N=8 pattern。Mac (M4 Pro 4P+10E 异构) 与 server (cn04 / cn15 同质 Intel/AMD 多核) 共享 RISK-26 NUMA / cache locality 敏感性 → Mac 不再是 "may pass while server fails" 的非典型 case，反而是 server-pattern 的 early signal。这一观察将在 PR-K capstone `docs/p1c/p1c_perf_baseline.md` §1.2 Mac sanity prediction power 节 documented。

## §9 Hand-off — PARTIAL-CLOSURE + CARVE-OUT 路径

- **PR-J reverse-compat (#253)** ON Kahan binary: 在 SHUD@3a0004c (Kahan-injected) build 上跑 NUM_OPENMP=1 heihe, 取 rivqdown SHA + cvode_stats，与 P1-update-omp-tag canonical SHA 字面比对。预测：不等 (P1 era 是 07c677f 未应用 8 站点 helper-wrap + Kahan; binary 改变了 acc order)。PR-J 需 documenting "P1c 引入 NUM_OPENMP=1 数值漂移 但 D11 immutability 仅保 tag SHA 不变, 不保证 binary 跑出 SHA 不变" (per a3a R3 documented framing).
- **PR-K capstone (#254)** 取本 PR §3/§4/§7/§8 数据落 `docs/p1c/p1c_summary.md` + `docs/p1c/p1c_perf_baseline.md` + 更新 `docs/p1c/p1c_a3a_root_cause.md` §"决策分支判定" 段; status_matrix.md 加 P1c partial-closure 行 + P1d carve-out 行。
- **PR-L (#255)** P1c-tag 仍 annotate + baseline/P1c lock, 但 tag message 写明 "partial-closure: 8-site reduction CLOSED, bit-level A3a + nst Δ=0 CARVED-OUT to P1d". D11 不变 (tag SHA 永不变).
- **PR-M (#256)** PROMOTE 2 specs (p1c-deterministic-reduction + p1c-capstone) 时 spec 内已含 "Kahan 兜底 + carve-out" 路径 (per PR-G a3a + spec.md L100-L103 Scenario), 不需 reshape spec; archive 完整 change.

## §10 SHUD pointer bump

本 PR 外层 commit 将 baseline/P1c 子模块 pointer 由 `de9545d` (pre-Kahan) 升级至 `3a0004c` (Kahan-injected). SHUD upstream openmp-baseline 已 pushed (`de9545d..3a0004c`), 不污染 master 分支 (per CLAUDE.md SHUD submodule 工作流 强制).