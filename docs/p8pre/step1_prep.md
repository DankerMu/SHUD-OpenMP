---
title: "p8pre-spike Step 1 prep — P1e PR-I baseline anchor for N=8 Mode C profile recheck"
date: 2026-06-26
version: 0.1 (Step 0 doc-correction PR, populated for downstream PR-A/B/F gate-4 use)
status: "Anchor doc inline-quoting P1e PR-I 24-cell median values so Step 1 aggregator + Step 2 PR-F gate-4 + ADR-0003 do NOT depend on the (now-corrected) `.p1e-i-runs/` path string in `docs/p1e/p1e_perf_baseline.md` §3.5/§7.1. Created in Step 0 PR per Epic #338 task §0.2 last bullet."
related_docs:
  - "docs/p1e/p1e_perf_baseline.md §3.1 (wall median table — source of truth)"
  - "docs/p1e/p1e_perf_baseline.md §3.4 (nst ladder — source of truth)"
  - "docs/p1e/p1e_pr_i_strict_omp_verification.md §3.1 + §5 (nfe + SHA12 baseline)"
  - "openspec/changes/p8pre-spike/tasks.md §1-§4 (Step 1 PR-A/B/C scope)"
---

# p8pre-spike Step 1 prep — P1e PR-I baseline anchor

## §1 目的

p8pre-spike Step 1 (PR-A/B/C) N=8 Mode C profile 复测要在 PR-B 聚合器 + PR-F gate-4 wall non-regression check + ADR-0003 ROI 论述里反复引用 P1e PR-I 24-cell 已 ship 的 wall + nst 基线。Step 0 doc-correction PR 把这些基线值 inline 落地到本 doc，避免下游读者必须穿透回 `docs/p1e/p1e_perf_baseline.md` §3.5/§7.1 的 `.p1e-i-runs/` 路径串（该串在 Step 0 之前误写作 `.pr-i-runs/`，已在同 PR 修正）。

source-of-truth 仍然是 `docs/p1e/p1e_perf_baseline.md` + `docs/p1e/p1e_pr_i_strict_omp_verification.md`；本 doc 仅 inline 镜像 + 加 cross-ref，未引入新数据。

## §2 wall median (s) per (case, N) — quoted from P1e §3.1 Table

P1e PR-I 24-cell: 3 reps per cell，取 median。Source: [`docs/p1e/p1e_perf_baseline.md`](../p1e/p1e_perf_baseline.md) §3.1。

| case | NumEle | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) |
|---|---:|---:|---:|---:|---:|
| heihe | 6335 | 504 | 511 | 488 | 473 |
| heihe_x4 | 40046 | 1340 | 1051 | 850 | 775 |

**Step 2 gate-4 用法** (per `openspec/changes/p8pre-spike/tasks.md` §8.5)：identity spike vs Step 1 baseline 同 build matrix (`SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` + SHUD pin `7a1dc8f` + 同 partition cn14/cn15) 比较，`ε(heihe) = 0.10` (~24s headroom) / `ε(heihe_x4) = 0.05` (~39s headroom)。P1e PR-I 列的 wall (本表) 因 build 在 SHUD `3341368d` 且未带 `SHUD_ENABLE_PROFILE=1`，不能直接当 Step 2 gate-4 baseline（Timer instrumentation overhead bias）；Step 1 PR-A 同 build 的 18-cell `wall_step1_baseline_median(case, N)` 才是 gate-4 真正 baseline。本表只用于：

1. Step 1 PR-A 自审 (复测 wall 应与本表同量级；若 Δ > 50% 触发 Mode C silent regression flag)
2. ADR-0003 ROI 论述里说明 P1e PR-H StrictOMP 已达到的实测 ROI (heihe 1.066×, heihe_x4 1.729×)

## §3 nst absolute baseline (15-key invariance anchor) — quoted from P1e §3.4

per [`docs/p1e/p1e_perf_baseline.md`](../p1e/p1e_perf_baseline.md) §3.4 nst ladder：

| case | ref nst (mode A) | N=1 nst | N=2 nst | N=4 nst | N=8 nst | max \|Δ\| |
|---|---:|---:|---:|---:|---:|---:|
| heihe | 6698 | 6698 | 6698 | 6698 | 6698 | 0 |
| heihe_x4 | 6575 | 6575 | 6575 | 6575 | 6575 | 0 |

**Step 1 aggregator 用法** (per `openspec/changes/p8pre-spike/tasks.md` §3.3)：除 relative cross-N invariance Δ=0 strict 之外，PR-B 还要 check absolute 锚定值 — heihe `nst_median = 6698` AND heihe_x4 `nst_median = 6575`。任一不匹配 → Mode C silent uniform regression flag (relative Δ=0 invariance alone 抓不到全 N 同步漂)。

## §4 nfe absolute baseline (跨文档引用)

per [`docs/p1e/p1e_pr_i_strict_omp_verification.md`](../p1e/p1e_pr_i_strict_omp_verification.md) §3.1：

- heihe `nfe_median = 6943`
- heihe_x4 `nfe_median = 6741`

Step 1 aggregator (PR-B §3.3) 同样 check absolute 锚定值 + cross-N Δ=0 strict invariance。

## §5 cross-ref index

| 下游消费方 | 用本 doc 哪节 | 用途 |
|---|---|---|
| PR-A self-check | §2 wall | 18-cell 复测同量级 sanity |
| PR-B aggregator (`tools/p8pre/aggregate_n8_profile.sh`) | §3 nst + §4 nfe | absolute baseline invariance check |
| PR-C capstone (`docs/p8pre/n8_profile_baseline.md`) | §2-§4 全部 | 与 P1e 对照表 |
| PR-F identity spike verdict (`docs/p8pre/identity_spike_verdict.md`) | §2 wall (forward-ref via PR-A 18-cell median) | gate-4 wall non-regression |
| ADR-0003 ROI 论述 | §2 sp@8 | StrictOMP 已实测 ROI 基线 |

## §6 引用

- [`docs/p1e/p1e_perf_baseline.md`](../p1e/p1e_perf_baseline.md) §3.1 wall median table + §3.4 nst ladder (source of truth)
- [`docs/p1e/p1e_pr_i_strict_omp_verification.md`](../p1e/p1e_pr_i_strict_omp_verification.md) §3.1 nfe baseline + §5 SHA12 baseline
- `tools/cvode_stats_diff/canonical_15_keys.yaml` (15-key 列表权威)
- Epic #338 (p8pre-spike) + Issue #339 (Step 0 doc-correction PR)
- `openspec/changes/p8pre-spike/tasks.md` §0 / §1-§4

---
Generated: 2026-06-26 by orchestrator (Step 0 doc-correction PR for p8pre-spike Epic #338)
