## 工作情况说明（Merge 前）

- 关联 Issue：#339
- PR：#350
- 冻结提交：`5827e237c86150dd4a16e8f68eeffc211f66c46f`
- 上游 Epic：#338 (p8pre-spike, N=8 Mode C profile recheck + identity precond spike + ADR-0003)

### 背景与目标

p8pre-spike epic 启动前置 Step 0 doc-correction PR，专修两类历史漂：

1. **15-key canonical drift**：`docs/p1e/p1e_perf_baseline.md` §3.5 L83 原列表含 4 个非 SUNDIALS-canonical 字段（`nlcf` 是 typo，正确为 `ncfl`；`nfevals/hcur/qcur/hin` 不在 SHUD 实际 emit 的 canonical set 内）。统一改为 `tools/cvode_stats_diff/canonical_15_keys.yaml` 权威的 15 键 `nfe/nfeLS/nni/nli/nsetups/netf/nst/npe/nps/ncfn/ncfl/lenrw/leniw/lenrwLS/leniwLS`。下游 PR-B aggregator 直接以这个 yaml 为单一权威源。
2. **scratch archive root drift**：`docs/p1e/p1e_perf_baseline.md` §3.5 / §7.1 / §7.3 archive table 4 处 `.pr-i-runs/` 漏 `p1e-` 前缀，与上游 canonical anchor `docs/p1e/p1e_pr_i_strict_omp_verification.md`（含 8 处 `.p1e-i-runs/`）不一致。本 PR 改回 canonical。

并新建 `docs/p8pre/step1_prep.md` inline 引用 P1e PR-I 24-cell wall median (§3.1) + nst ladder (§3.4) + nfe baseline (跨 doc) — 作 Step 1 (PR-A/B) + Step 2 (PR-F gate-4) + ADR-0003 ROI 论述的 absolute baseline anchor，未来读者不再依赖修正前的旧路径串。

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `docs/p1e/p1e_perf_baseline.md` | §3.5 L83: 错 15-key 列表 → canonical 15-key；§3.5 L83 / §7.1 L156-157 / §7.3 L183: `.pr-i-runs/` × 4 → `.p1e-i-runs/`；4 行 edit |
| `docs/p8pre/step1_prep.md` | 新建（70+ 行）：inline 引用 §3.1 wall (heihe 504/511/488/473 + heihe_x4 1340/1051/850/775) + §3.4 nst (6698/6575 ×5) + §4 nfe (6943/6741)，附 §5 下游消费方 cross-ref index |

无 SHUD submodule pin bump，无 .cpp/.h/.yaml 改动，无 CI rule 改动，无 case manifest 改动。

### 测试与验证

本 PR 为 doc-only，无 unit test 可加；验收 oracle 是 6 个 grep 指令：

| AC | command | 实测 |
|---|---|---|
| `pr-i-runs` in p1e_perf_baseline.md = 0 | `grep -c pr-i-runs docs/p1e/p1e_perf_baseline.md` | **0** |
| `p1e-i-runs` > 0 | `grep -c p1e-i-runs docs/p1e/p1e_perf_baseline.md` | **4** |
| `nlcf` in p2a = 0 | `grep -c nlcf docs/p2a/p2a_profile_baseline.md` | **0**（天然满足） |
| `nlcf` in p1e = 0 | `grep -c nlcf docs/p1e/p1e_perf_baseline.md` | **0** |
| canonical 15-key marker `lenrwLS` | `grep -c lenrwLS docs/p1e/p1e_perf_baseline.md` | **1** |
| 上游 canonical anchor `.p1e-i-runs` 未触 | `grep -c p1e-i-runs docs/p1e/p1e_pr_i_strict_omp_verification.md` | **8** ≥ 8 |

CI 全 PASS (5/5)：asan-ubsan (keliya / qhh)、build-and-compare (1, keliya)、setup、tools-tests。`openspec validate p8pre-spike --strict --no-interactive` exit 0。

### Review 与修复闭环

- **Phase 0.5 fixture review**: PASS — p8pre-spike fixture §0 与 #339 AC 1:1 对齐，fixture level `none` defensible
- **Phase 4 round 1 cross-review** (compact-level, 2 reviewers):
  - `review-correctness`: 0 findings + 1 traceability note
  - `review-integration`: 1 Warning finding（sibling-doc drift `p1e_academic_summary.md:219` + `p1e_summary.md:113`，标 Blocks merge=NO）
- **Phase 4.5 verifier**: cand-01 verdict = **REFUTED-out-of-scope**（drift facts 真实但严格在 #339 In-Scope enumeration 外）→ drop + spawn follow-up task chip
- **Phase 6 跳过**（Phase 5 rule: cross-review reports clean ⇒ skip）
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，6/6 AC PASS，oracle integrity PASS，APPROVE merge

### 兼容性、风险与已知限制

- 无 API / 数据格式 / 迁移兼容性影响（doc-only）
- 历史 `.p1e-i-runs/<case>_N<n>_rep<r>/cvode_stats.txt` 归档 bytes 不变，本 PR 仅 re-align prose 到 on-disk 现实
- **已知 out-of-scope follow-up**: `p1e_academic_summary.md:219` + `p1e_summary.md:113` 仍含同类 `.pr-i-runs/` drift（Phase 4.5 verifier 已 REFUTED-out-of-scope，已 spawn 独立 task chip 处理；不属于 #339 explicit In-Scope）

### 维护者关注点

- 无额外关注点。
