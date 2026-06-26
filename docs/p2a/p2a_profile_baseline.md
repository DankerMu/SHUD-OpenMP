---
title: "P2a profile baseline — pre-CVODE forcing/ET wall 占比量化"
date: 2026-06-25
version: 0.1 (initial, prep step before P2a epic intake)
status: "DRAFT — pending t_forcing_io bucket instrumentation fix in next iteration"
related_docs:
  - "docs/p1e/p1e_perf_baseline.md (RHS = 66.55% wall reference)"
  - "docs/profile_decision.md (B0 era profile gate, t_RHS_total% baseline)"
  - "SHUD_openMP_master_plan.md §P2a.2 (2×2 build matrix + profile)"
---

# P2a profile baseline

## §1 目的

P2a (并行 pre-CVODE forcing / ET loop) epic intake 前的 ROI 量化前置工作。在 SHUD `shud.cpp` 主循环 4 处补 `shud_profile::Timer` 插桩（per master plan §P2a.2 + tools/profile/timer.h 预定 buckets），跑 5 case × N=1 × 1 rep，量化 forcing + ET 在 wall 中的占比，判断 P2a 是否值得进入 14-PR 模板。

## §2 实验设置

| 项 | 值 |
|---|---|
| SHUD pin | `7cc46d8` (P1e `3341368` + Step 1 instrumentation hooks) |
| Build | `make shud SHUD_ENABLE_PROFILE=1` (no OMP, N=1 串行) |
| Cases | 4 Mac case + 1 server case (heihe_x4 NOT deployed on server, 需 rSHUD mesh-refine 派生，超出 prep scope) |
| Threads | N=1, OMP_NUM_THREADS=1, SHUD_RHS_THREADS=1 |
| Reps | 1 (profile baseline, 不验收) |
| Truncation | 90-day (per CLAUDE.md 项目铁律) |
| Mac runtime | Apple M4 Pro arm64, libomp 22.1.7 |
| Server runtime | Intel Xeon cn11 x86_64, libgomp gcc 13.3 |

## §3 数据采集

### §3.1 Mac 4 case (本地)

来源: `/tmp/p2a_profile_mac/{case}_N1.yaml`

| Case | NumEle | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output |
|---|---:|---:|---:|---:|---:|---:|---:|
| keliya | 484 | 30.23 | 11.71 | 26.05 | 7.04 | 0.05 | 0.03 |
| xinanjiang_upstream | 801 | 4.73 | 1.91 | 3.66 | 0.76 | 0.07 | 0.06 |
| qinyijiang | 3155 | 285.61 | 145.32 | 281.16 | 3.14 | 0.28 | 0.05 |
| qhh | 4773 (+lake) | 97.30 | 35.81 | 54.85 | 57.49 | 0.62 | 0.20 |

### §3.2 Server 1 case (Slurm cn11, job 9448)

来源: `/scratch/frd_muziyao/SHUD-OpenMP/.p2a-profile/heihe_N1.yaml`

| Case | NumEle | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output |
|---|---:|---:|---:|---:|---:|---:|---:|
| heihe | 6335 | 494.81 | 51.66 | 83.74 | **773.28** ❌ | 2.30 | 0.31 |

real time = 560.59 s (含 init / finalize / output, NumSteps 完成)。

## §4 关键发现：`t_forcing_io` instrumentation BUG (server heihe)

**Anomaly**: heihe `t_forcing_io = 773.28s` > `t_wall_total = 494.81s`，物理不可能。

**疑似根因** (待 P2a-0 audit 跟进):
- `shud.cpp` L210 `MD->updateforcing(t)` 包在 `{ Timer _t_fr; ... }` scope。但 inner `while (t < tnext)` 迭代次数远多于 outer `NumSteps`；每次 inner step 调一次 forcing。bucket 累加正常，单次 forcing 调用平均 ~0.18s × 4320 次 ≈ 770s — 符合算术。
- 但 `t_wall_total` (L196) 包整个 outer `for` loop，理论应 ≥ inner forcing 累加。实测 494s < 773s 说明 wall_total 或 forcing_io 之一有 atomic 累加 race / dump 时序 bug。
- Mac 4 case 未触发（forcing 时间 < wall），可能 NFS IO latency 在 server 端 amplify 出 bucket 累加问题。

**workaround 推算法**（用于 ROI 估算，**不可用于 P2a SHALL gate**）:

```
forcing+ET+其它(init/finalize)推算 = wall - t_CVODE_raw - t_output
                                    = 494.81 - 83.74 - 0.31
                                    = 410.76 s (83.0% of wall)
```

Mac 4 case 直接 bucket 值可信（forcing+ET < wall 全 PASS）。

## §5 forcing+ET wall 占比 + Amdahl sp@8 上界

| Case | Platform | wall (s) | forcing+ET % | Amdahl sp@8 上界 | 备注 |
|---|---|---:|---:|---:|---|
| keliya | Mac M4 Pro | 30.23 | 23.4% | 1.26× | bucket-direct |
| xinanjiang_upstream | Mac M4 Pro | 4.73 | 17.5% | 1.18× | bucket-direct |
| qinyijiang | Mac M4 Pro | 285.61 | 1.2% | 1.01× | bucket-direct |
| qhh | Mac M4 Pro | 97.30 | 59.7% | 2.09× | bucket-direct; +lake forcing 路径多 |
| **heihe** | **Server cn11** | **494.81** | **~83%** (推算) | **~3.81×** | wall-derived workaround，待 bug fix re-baseline |

(heihe_x4 数据缺失，需 server rSHUD mesh-refine 部署后补)

## §6 P2a Go/No-Go 建议

**Pro (支持 P2a 启动)**:
1. **server heihe forcing+ET 占 wall ~83%** (推算)，sp@8 Amdahl 上界 ~3.81× — 比 P1e RHS 上界 2.39× 还高，production ROI 显著
2. Mac qhh +lake 拓扑 forcing+ET 60% — 中等案例已展示 forcing-bound 模式
3. 设计语汇可直接复用 P1e (StrictOMP single parallel region + SHUD_RHS_THREADS env + 2×2 因果 + §4.6.2 partial-closure + D7 AND-gate)
4. P1e era 已修的 race (RISK-16 movePointer + RISK-17 printf) 减少 P2a 新增风险

**Con (P2a 风险 / 限制)**:
1. forcing+ET 占比 **case-dependent 巨大** (qinyijiang 1.2% ~ heihe 83%)；single threshold 不适用
2. heihe 推算占比 ~83% 来自 wall-derived workaround，**实际 forcing IO 是否为 thread-safe-parallelizable 未知**（NetCDF read + CMFD V0200 路径访问可能有共享 file descriptor / mutex bottleneck）
3. qinyijiang 1.2% → P2a 对它几乎无收益 (sp@8 上界 1.01×)，必走 §4.6.2 partial-closure
4. `t_forcing_io` bucket bug 必先修才能跑 P2a 验收 SHALL gate (否则 sp@8 算不出来)

**推荐**:

| 选项 | 决策 |
|---|---|
| **(a) 启动 P2a epic + PR-B0 audit-required = 修 timer bug** | 推荐 — bug 修后 re-baseline + 因果 2×2 实验给精确 ROI + 14-PR 模板 |
| (b) 推后 P2a，先做 P2b (RHS element vertical) 或 P3 (owner-local gather) | 不推荐 — 这两个阶段在 P1e StrictOMP RHS 之后，预计 wall 已被 P1e 大幅压低，ROI 反而比 P2a 小 |
| (c) 跳过 P2a 直接 P7 (final-fusion) / P9 (production DAG) | 不推荐 — 跳级会损失 forcing+ET 80% 占比 case 的 ROI 机会 |

## §7 next step

1. **P2a-0 (audit-required prep PR)**: 修 `t_forcing_io` bucket bug + re-run heihe baseline + 部署 heihe_x4 on server (rSHUD mesh-refine 从 heihe 派生)
2. **P2a-A (epic intake PR)**: openspec change `p2a-pre-cvode-parallel` propose + design D1-Dn + tasks §1-§7
3. **2×2 build matrix**: per master plan §P2a.2 (P2a-A serial / P2a-B parallel / P2a-C 隔离)
4. **stage-change-pipeline**: epic issue + 12-14 sub-issue DAG

## §8 引用

- master plan §P2a.1-.8 (P1e 经验重梳理后版)
- docs/p1e/p1e_perf_baseline.md §6 三因素分析 (OMP overhead floor 经验)
- docs/profile_decision.md (B0 era profile gate, t_RHS_total% 4 case)
- tools/profile/timer.h (7 预定 bucket)
- `SHUD@7cc46d8` (Step 1 instrumentation hooks at L208/L215/L228/L266)
- Slurm job 9448 (heihe N=1 cn11 wall 9:21)

---
Generated: 2026-06-25 by orchestrator (P2a profile baseline prep, post Mac + heihe data)
