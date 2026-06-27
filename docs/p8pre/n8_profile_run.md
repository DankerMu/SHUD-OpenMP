---
title: "p8pre-spike Step 1 PR-A run — 18-cell N=8 Mode C profile execution log"
date: 2026-06-27
version: 0.1
status: "18-cell Slurm run COMPLETED; per-cell 5-artifact + 15-key canonical + extras + bucket-sum invariant + run_exit_code=0 all PASS; rsync mirror at /tmp/p8pre_n8_profile/. PR-B (#342) aggregator + ROI verdict pending."
related_docs:
  - "openspec/changes/p8pre-spike/proposal.md (epic rationale)"
  - "openspec/changes/p8pre-spike/tasks.md §2 (PR-A run scope) + §3 (PR-B aggregator scope, downstream consumer)"
  - "openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md"
  - "docs/p8pre/pr_a_prep_evidence.md (PR-A prep: build matrix + render dry-run)"
  - "docs/p8pre/step1_prep.md (P1e PR-I baseline anchor)"
  - "tools/cvode_stats_diff/canonical_15_keys.yaml (15-key gate single source of truth)"
  - "tools/profile/timer.cpp L150-184 (bucket derivation algebra)"
---

# p8pre-spike Step 1 PR-A run — execution log

## §1 目的

本 doc 记录 p8pre-spike Step 1 PR-A run 阶段 18-cell N=8 Mode C profile 执行情况,
仅作 execution log + per-cell artifact verification, 不评 ROI。

ROI verdict / cross-N nst Δ invariance / nfeLS/nfe scaling / nli/nni ratios → PR-B (#342)
`docs/p8pre/n8_profile_verdict.md`。N=8 baseline (median over reps, 含 SHA12 stability)
→ PR-C (#343) `docs/p8pre/n8_profile_baseline.md`。

## §2 实验执行

| 项 | 值 |
|---|---|
| Slurm submit 时间 (UTC) | 2026-06-27T01:32:52Z (JID 9510, 9513, 9516, 9519, 9522, 9525 起首) |
| Slurm complete 时间 (UTC) | 2026-06-27T02:43:35Z (JID 9521 末) |
| Total wall (clock end - first start) | ~70 min |
| JID range | 9510-9527 (18 个) |
| Node assignment | cn14 (heihe stream, JID 9510-9518) / cn15 (heihe_x4 stream, JID 9519-9527) |
| SHUD pin | `7a1dc8f` (Mode C build: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`) |
| Build matrix nm gates (PR-A prep §1) | PASS (`N_VNew_Serial`=1 / `N_VNew_OpenMP`=0 / `GOMP_parallel`=1) |
| All 18 cells `sacct State / ExitCode` | COMPLETED / `0:0` |
| 18-cell singleton dependency chain | OK (3 reps serial per (case, N) via `--dependency=afterany`) |

## §3 per-cell wall (per sacct Elapsed)

JID → cell mapping per `.p8pre-runs/jid_table.txt` (4 columns: `case N rep JID`).

| cell | JID | node | wall (mm:ss) |
|---|---:|---|---:|
| heihe_N1_rep1    | 9510 | cn14 | 02:28 |
| heihe_N1_rep2    | 9511 | cn14 | 02:27 |
| heihe_N1_rep3    | 9512 | cn14 | 02:11 |
| heihe_N4_rep1    | 9513 | cn14 | 01:43 |
| heihe_N4_rep2    | 9514 | cn14 | 01:42 |
| heihe_N4_rep3    | 9515 | cn14 | 01:41 |
| heihe_N8_rep1    | 9516 | cn14 | 01:37 |
| heihe_N8_rep2    | 9517 | cn14 | 01:36 |
| heihe_N8_rep3    | 9518 | cn14 | 01:36 |
| heihe_x4_N1_rep1 | 9519 | cn15 | 25:04 |
| heihe_x4_N1_rep2 | 9520 | cn15 | 24:02 |
| heihe_x4_N1_rep3 | 9521 | cn15 | 21:37 |
| heihe_x4_N4_rep1 | 9522 | cn15 | 14:38 |
| heihe_x4_N4_rep2 | 9523 | cn15 | 14:39 |
| heihe_x4_N4_rep3 | 9524 | cn15 | 14:30 |
| heihe_x4_N8_rep1 | 9525 | cn15 | 12:52 |
| heihe_x4_N8_rep2 | 9526 | cn15 | 12:54 |
| heihe_x4_N8_rep3 | 9527 | cn15 | 12:53 |

## §4 per-cell artifact + canonical-key + invariant verification

Raw output: `.review-evidence/p8pre-pr-a-run/verification.txt`. Five gates per cell:

| Gate | Definition | Source of truth |
|---|---|---|
| ART | 5 artifacts present (`profile_B0.yaml`, `cvode_stats.txt`, `<case>.rivqdown.dat`, `slurm.out`, `slurm.err`) | sbatch template L117-128 |
| CANON15 | `cvode_stats.txt` 行键集合 = canonical 15-key 全集 (`nfe nfeLS nni nli nsetups netf nst npe nps ncfn ncfl lenrw leniw lenrwLS leniwLS`) | `tools/cvode_stats_diff/canonical_15_keys.yaml` |
| REJECT | typo / wrong-name set 缺席 (`nlcf`, `nfevals`, `hcur`, `qcur`, `hin` 均不出现) | brief §1.b |
| EXTRAS | `profile_B0.yaml` 含 `extras:` block 内 `t_CVODE_raw` AND `t_wall_total` | `tools/profile/timer.cpp:183-184` |
| BSUM% | `(t_RHS_total + t_CVODE_internal + t_forcing_io + t_ET + t_output + t_other - t_wall_total) / t_wall_total × 100`，gate ≤ ±2% (排除 `t_RHS_kernel` 防 nested double-count，参 SHUD `7a1dc8f` p2a v0.1 修复) | `tools/profile/timer.cpp:152-156` 派生公式 |
| RC0 | `slurm.out` 含 `[p8pre_n8_profile] run_exit_code=0` 行 | sbatch template L105 |

**Result: 18 / 18 cells PASS all 5 gates (zero BLOCK).** Worst absolute bucket-sum
error = `0.0000%` (跨全 18 cells)。原因即 timer.cpp 的派生公式：
`t_other := t_wall_total - t_CVODE_raw - t_forcing_io - t_ET - t_output` (L152-156, 钳 ≥0)
+ `t_CVODE_internal := t_CVODE_raw - t_RHS_total` (L150-151, 钳 ≥0) → 代数恒等使
6 项之和精确等于 `t_wall_total` (前提是两钳均未触发, 即两个 raw 差均非负)。
**零钳触发 = 测量良性, 各 raw timer pair 不出现负值漂移**, 此乃 PR-A run 阶段
最强的 instrumentation 自洽证据。

完整 18 cells 表 (节选格式):

```
cell                   | ART | CANON15 | REJECT | EXTRAS |   BSUM% | RC0
heihe_N1_rep1          |  OK |      OK |     OK |     OK | -0.0000 |  OK
heihe_N1_rep2          |  OK |      OK |     OK |     OK | -0.0000 |  OK
heihe_N1_rep3          |  OK |      OK |     OK |     OK | +0.0000 |  OK
... (全 18 行均 PASS, 详见 verification.txt) ...
heihe_x4_N8_rep3       |  OK |      OK |     OK |     OK | +0.0000 |  OK
```

## §5 SHA12 + nst 初步抓取 (handoff to PR-B, 仅 median per (case, N))

per-rep median sample; **不评 Δ / invariance, PR-B (#342) §3.3-§3.4 scope**。
所有 CVODE 计数 (`nst / nfe / nfeLS / nni / nli`) 在每个 case 内部三 reps × 三 N
**完全一致** (bitwise 同值), 提示求解轨迹 cross-rep + cross-N invariant; rivqdown
SHA12 在同 (case, N) 三 reps 内出现部分 diverge — invariance 分析交 PR-B/PR-C。

| (case, N) | median wall (s) | nst | nfe | nfeLS | nli/nni | nfeLS/nfe | sample rivqdown SHA12 (rep 中位 wall) |
|---|---:|---:|---:|---:|---:|---:|---|
| (heihe, 1)    | 147  | 6698 | 6943 | 12632 | 1.8197 | 1.8194 | `a2023ccd2de4` (rep2) |
| (heihe, 4)    | 102  | 6698 | 6943 | 12632 | 1.8197 | 1.8194 | `04c6d51433f0` (rep2) |
| (heihe, 8)    |  96  | 6698 | 6943 | 12632 | 1.8197 | 1.8194 | `a01c5018939b` (rep2) |
| (heihe_x4, 1) | 1442 | 6575 | 6741 | 30509 | 4.5266 | 4.5260 | `b5e4b0a2cf83` (rep2) |
| (heihe_x4, 4) |  878 | 6575 | 6741 | 30509 | 4.5266 | 4.5260 | `103d1749e241` (rep2) |
| (heihe_x4, 8) |  773 | 6575 | 6741 | 30509 | 4.5266 | 4.5260 | `7e777ec1b6cd` (rep2) |

ratio columns 仅作 PR-B aggregator 输入 cross-check, 数值意义 / ROI 分支 (a/b/c/d
per spec §3.4) 由 PR-B 评。

## §6 rsync mirror

Server source: `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/`
Local mirror: `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/` (18 dirs + `jid_table.txt` + `render_stdout.txt`)
Rsync size: ~46 MB total (`du -sh` 46M)
Rsync time: 2026-06-27 (本地)
Rsync log: `.review-evidence/p8pre-pr-a-run/rsync_log.txt`
Iron rule compliance: rsync POST 所有 Slurm jobs COMPLETED (sacct 验过), 符合 P1e PR-D
"no rsync during runs" 约束。

包含 patterns:
- `*/cvode_stats.txt` (15 行)
- `*/profile_B0.yaml` (7 buckets + extras)
- `*/<case>.rivqdown.dat` (river discharge, gate-1 of PR-C SHA12 baseline)
- `*/slurm.out` + `*/slurm.err`
- `jid_table.txt` + `render_stdout.txt` (run provenance)

排除: `rendered/` (18 临时 .sbatch 已由 wrapper 渲染, 不再需要)

## §7 Handoff to PR-B (#342)

PR-B aggregator 入口 = `/tmp/p8pre_n8_profile/`, 期望 18 cell dirs + jid_table.txt 同
本 doc §3 行序一致。

PR-B (#342) scope (不在本 doc):
- median per (case, N) wall + nst/nfe/nfeLS/nli/nni 复算 (本 doc §5 仅作 lightweight sanity)
- cross-N nst Δ invariance gate (spec §3.3)
- nfeLS/nfe + nli/nni cross-N scaling ROI verdict (spec §3.4 branch a/b/c/d)
- ratio time-series + bucket attribution

PR-C (#343) scope (不在本 doc):
- N=8 baseline 锁: median wall + median CVODE counters + rivqdown SHA12 选种
- 与 P1e baseline 跨 epic 对比

## §8 引用

- openspec/changes/p8pre-spike/proposal.md
- openspec/changes/p8pre-spike/tasks.md §2 (PR-A run) + §3 (PR-B aggregator, downstream)
- openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md
- tools/p8pre/run_n8_profile.sh (server-side wrapper, 18-cell submit + JID chain)
- tools/p8pre/submit_n8_profile_template.sbatch (per-cell template)
- tools/cvode_stats_diff/canonical_15_keys.yaml (15-key gate single source of truth)
- tools/profile/timer.cpp:150-184 (`t_other` / `t_CVODE_internal` 派生算法 + extras 兜底)
- .review-evidence/p8pre-pr-a-run/verification.txt (raw 18-cell gate output)
- .review-evidence/p8pre-pr-a-run/rsync_log.txt (raw rsync transcript)
- /scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/jid_table.txt (server JID table, 18 行)
- Slurm 三铁律 → CLAUDE.md §"Slurm 三铁律"
- P1e PR-D rsync 时机约束 → docs/p1e_summary.md (history reference)
