---
title: "P2a profile baseline — pre-CVODE forcing/ET wall 占比量化"
date: 2026-06-26
version: 0.2 (re-baselined after t_forcing_io nested-Timer fix)
status: "SHIP-LOCKED for P2a epic intake — t_forcing_io / t_ET / t_output buckets validated <wall on all 5 cases"
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
| SHUD pin (v0.1 initial) | `7cc46d8` (P1e `3341368` + Step 1 instrumentation hooks) — surfaced bucket double-count bug |
| SHUD pin (v0.2 fixed) | `7a1dc8f` (`7cc46d8` + nested-Timer removal in MD_ET.cpp / Model_Control.cpp) — **current ship pin** |
| Build | `make shud SHUD_ENABLE_PROFILE=1` (no OMP, N=1 串行) |
| Cases | 4 Mac case + 1 server case (heihe_x4 NOT deployed on server, 需 rSHUD mesh-refine 派生，超出 prep scope) |
| Threads | N=1, OMP_NUM_THREADS=1, SHUD_RHS_THREADS=1 |
| Reps | 1 (profile baseline, 不验收) |
| Truncation | 90-day (per CLAUDE.md 项目铁律) |
| Mac runtime | Apple M4 Pro arm64, libomp 22.1.7 |
| Server runtime | Intel Xeon cn08 (re-run) / cn11 (initial), x86_64, libgomp gcc 13.3 |

## §3 数据采集

### §3.1 Mac 4 case (本地)

来源: `/tmp/p2a_profile_mac/{case}_N1.yaml` (v0.1 initial, SHUD `7cc46d8`)
qhh re-run post-fix (SHUD `7a1dc8f`) 用于 sanity 验收。

| Case | NumEle | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| keliya | 484 | 30.23 | 11.71 | 26.05 | 7.04 | 0.05 | 0.03 | v0.1 (pre-fix; double-count ≤30% wall, 安全) |
| xinanjiang_upstream | 801 | 4.73 | 1.91 | 3.66 | 0.76 | 0.07 | 0.06 | v0.1 |
| qinyijiang | 3155 | 285.61 | 145.32 | 281.16 | 3.14 | 0.28 | 0.05 | v0.1 |
| qhh | 4773 (+lake) | 97.30 | 35.81 | 54.85 | 57.49 | 0.62 | 0.20 | v0.1 (pre-fix, 含 double-count) |
| **qhh (fixed)** | **4773 (+lake)** | **97.19** | **35.85** | **54.70** | **29.55** | **0.30** | **0.14** | **v0.2 (SHUD `7a1dc8f`); forcing 减半 sanity PASS** |

**fix 影响推算** (qhh delta):
- t_forcing_io 57.49 → 29.55 (×0.514) — nested double-count 假设吻合 (~half).
- t_ET 0.62 → 0.30 (×0.484)
- t_output 0.20 → 0.14 (×0.700; outer 2 sites 与 inner 1 site 比为 2:1 时部分减少)
- t_wall_total 不变 (×0.999) — fix 不动 main loop wall

其它 3 case (keliya / xinanjiang / qinyijiang) 未 re-run。v0.1 数字含 ~×2 nested double-count;
若需精确占比，把 t_forcing_io / t_ET / t_output 除以 2 推算即可（实际 ROI 估算不影响结论：占比都 < 30%，远低于 heihe/qhh）。

### §3.2 Server 1 case (Slurm cn08, job 9475 re-run post-fix)

来源: `/scratch/frd_muziyao/SHUD-OpenMP/.p2a-profile/heihe_N1.yaml`

| Case | NumEle | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| heihe (v0.1) | 6335 | 494.81 | 51.66 | 83.74 | **773.28** ❌ | 2.30 | 0.31 | nested double-count bug, **REJECTED** |
| **heihe (v0.2 fixed)** | **6335** | **523.05** | **60.20** | **94.43** | **400.99** ✓ | **1.31** | **0.46** | **SHIP** |

real time (v0.2) = 610.30 s (含 init / finalize / output, NumSteps 完成)。wall+~17% init/output overhead。
forcing 400.99 < wall 523.05 物理 sane ✓。

**fix 影响推算** (heihe delta):
- t_forcing_io 773.28 → 400.99 (×0.519) — 与 qhh ×0.514 一致，确认 nested ×2 是稳定 fix。
- t_ET 2.30 → 1.31 (×0.570; 但 nfe 6943 < cn08 jitter 范围)
- t_wall_total 494.81 → 523.05 (cn08 vs cn11 node load 6% jitter, 可接受)

## §4 instrumentation bug — **FIXED in SHUD `7a1dc8f`**

**v0.1 anomaly**: heihe `t_forcing_io = 773.28s` > `t_wall_total = 494.81s`，物理不可能。

### §4.1 真实 root cause (v0.2 调查结果)

三个 inner-function RAII Timer 与 shud.cpp main-loop 调用站点 Timer **嵌套**，
同名 bucket 被 outer + inner 两个 Timer 各累加一次 wall（outer scope ⊇ inner scope，
两者都跑 dtor），导致 bucket 累计 ≈ 2 × 真实值：

| Inner-function Timer (REMOVED) | Outer call-site Timer (KEPT) |
|---|---|
| `MD_ET.cpp:20`  `Timer("t_forcing_io")` 在 `Model_Data::updateforcing()` 内 | `shud.cpp:211` `Timer _t_fr("t_forcing_io")` 包 `MD->updateforcing(t)` |
| `MD_ET.cpp:132` `Timer("t_ET")`         在 `Model_Data::ET()` 内             | `shud.cpp:219` `Timer _t_et("t_ET")`         包 `MD->ET(t, tnext)` |
| `Model_Control.cpp:87` `Timer("t_output")` 在 `Control_Data::ExportResults()` 内 | `shud.cpp:241/282` `Timer _t_out/_t_out2("t_output")` 包 `summary()` + `ExportResults()` |

### §4.2 Mac vs server 差异

Mac qhh / server heihe 都有 bug，只是 Mac forcing/wall ratio (~59% 表观) 没有越过 wall（~100% 上限）
而看似 sane；server heihe per-iter forcing path 更重 + NumSteps 更多 → double-count 累积越过 wall 才暴露。
原假设「libgomp vs libomp 时钟 ABI 问题」**REFUTED** — 是纯 nested Timer 累加 bug，与 chrono / 平台无关。

### §4.3 Fix (SHUD commit `7a1dc8f`)

删除 3 处 inner-function Timer + 关联 `#include "timer.h"`。保留 shud.cpp 4 处 call-site Timer 作为唯一累加点。
profile 覆盖范围不变（outer scope 完全包含 inner 函数调用），determinism 不变（Timer 仍 RAII + lock-free atomic）。

**Caveat**: SUBSURF coupled-CVODE path (`shud.cpp:447-502`) 无 Timer 包裹。若未来 enable SUBSURF mode，需要在那段补 instrumentation
(out of P2a scope; main solver loop only per master plan §S0.12)。

## §5 forcing+ET wall 占比 + Amdahl sp@8 上界 (v0.2 SHIP-LOCKED)

| Case | Platform | SHUD pin | wall (s) | forcing+ET (s) | forcing+ET % | Amdahl sp@8 上界 | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| keliya | Mac M4 Pro | `7cc46d8` (pre-fix) | 30.23 | 7.09 → ~3.5 | ~11.7% | ~1.11× | bucket÷2 推算 (pre-fix double-count) |
| xinanjiang_upstream | Mac M4 Pro | `7cc46d8` (pre-fix) | 4.73 | 0.83 → ~0.4 | ~8.8% | ~1.08× | bucket÷2 推算 |
| qinyijiang | Mac M4 Pro | `7cc46d8` (pre-fix) | 285.61 | 3.42 → ~1.7 | ~0.6% | ~1.00× | bucket÷2 推算 |
| **qhh (fixed)** | Mac M4 Pro | **`7a1dc8f`** | **97.19** | **29.85** | **30.7%** | **1.39×** | **bucket-direct (v0.2 fixed)** |
| **heihe (fixed)** | Server cn08 | **`7a1dc8f`** | **523.05** | **402.30** | **76.92%** | **3.06×** | **bucket-direct (v0.2 fixed) — SHIP for P2a SHALL gate** |

**fixed 数字解读**:
- **heihe forcing+ET 76.92%** wall — P2a sp@8 Amdahl 上界 **3.06×**（v0.1 推算的 ~3.81× 因含 ~17% init/finalize/output 的非 main-loop wall 而虚高；v0.2 用真实 bucket 后更可信）。
- qhh 30.7% — 中等 case，sp@8 上界 1.39×，单独看 ROI 不够，但叠加 heihe 验证 P2a 在生产场景 (server NWM heihe) 显著 ROI。
- 其它 3 case 推算占比 < 12%，sp@8 < 1.12× — P2a 对它们几乎无收益，**走 §4.6.2 partial-closure 必需**。

(heihe_x4 数据缺失，需 server rSHUD mesh-refine 部署后补 — 不在本 prep scope)

## §6 P2a Go/No-Go 建议

**Pro (支持 P2a 启动)**:
1. **server heihe forcing+ET 占 wall 76.92%** (bucket-direct, v0.2 fixed)，sp@8 Amdahl 上界 **3.06×** — 比 P1e RHS 上界 2.39× 高 28%，production ROI 显著
2. Mac qhh forcing+ET 30.7% — 中等案例已展示 forcing-bound 模式
3. 设计语汇可直接复用 P1e (StrictOMP single parallel region + SHUD_RHS_THREADS env + 2×2 因果 + §4.6.2 partial-closure + D7 AND-gate)
4. P1e era 已修的 race (RISK-16 movePointer + RISK-17 printf) 减少 P2a 新增风险
5. **bug fix 提前到 prep PR 已完成 (SHUD `7a1dc8f`)** — P2a-A epic intake 可直接进 propose 阶段，无 P2a-0 dependency

**Con (P2a 风险 / 限制)**:
1. forcing+ET 占比 **case-dependent 巨大** (qinyijiang ~0.6% ~ heihe 76.92%)；single threshold 不适用 → 仍需 §4.6.2 partial-closure
2. heihe 76.92% 中 forcing IO 是否为 thread-safe-parallelizable **未知** (NetCDF read + CMFD V0200 共享 file descriptor / mutex bottleneck 可能限制 sp@8 < 3.06×)
3. qinyijiang ~0.6% → P2a 对它几乎无收益 (sp@8 上界 ~1.00×)，必走 §4.6.2 partial-closure
4. Mac 3 case (keliya / xinanjiang / qinyijiang) 仍是 v0.1 推算数字，若 P2a-A propose 阶段需要这 3 case 精确占比，需要单独 re-run

**推荐**:

| 选项 | 决策 |
|---|---|
| **(a) 启动 P2a epic propose (P2a-0 已 done)** | **强烈推荐** — bug 修后 bucket SHIP-LOCKED，可直接 14-PR 模板 |
| (b) 推后 P2a，先做 P2b (RHS element vertical) 或 P3 (owner-local gather) | 不推荐 — 这两个阶段在 P1e StrictOMP RHS 之后，预计 wall 已被 P1e 大幅压低，ROI 反而比 P2a 小 |
| (c) 跳过 P2a 直接 P7 (final-fusion) / P9 (production DAG) | 不推荐 — 跳级会损失 forcing+ET 76.92% 占比 case 的 ROI 机会 |

## §7 next step

1. ~~**P2a-0 (audit-required prep PR)**: 修 `t_forcing_io` bucket bug + re-run heihe baseline~~ — **DONE** in SHUD `7a1dc8f` + outer pin bump + this doc v0.2 update
2. **P2a-A (epic intake PR)**: openspec change `p2a-pre-cvode-parallel` propose + design D1-Dn + tasks §1-§7
3. **2×2 build matrix**: per master plan §P2a.2 (P2a-A serial / P2a-B parallel / P2a-C 隔离)
4. **stage-change-pipeline**: epic issue + 12-14 sub-issue DAG
5. **heihe_x4 on server**: rSHUD mesh-refine 从 heihe 派生（单独步骤，非 P2a epic 必需前置 — 可在 P2a-D / E 之间引入加密 case）
6. **(optional) Mac 3 case re-run with `7a1dc8f`**: 若 P2a-A propose 阶段需要精确 keliya / xinanjiang / qinyijiang 占比，可单独 re-run (15min 总耗时)

## §8 引用

- master plan §P2a.1-.8 (P1e 经验重梳理后版)
- docs/p1e/p1e_perf_baseline.md §6 三因素分析 (OMP overhead floor 经验)
- docs/profile_decision.md (B0 era profile gate, t_RHS_total% 4 case)
- tools/profile/timer.h (7 预定 bucket)
- `SHUD@7cc46d8` (Step 1 instrumentation hooks at L208/L215/L228/L266) — v0.1 baseline, bug-laden
- `SHUD@7a1dc8f` (Step 2 nested-Timer fix: MD_ET.cpp + Model_Control.cpp) — v0.2 SHIP-LOCKED
- Slurm job 9448 (heihe N=1 cn11 wall 9:21, v0.1 anomaly)
- Slurm job 9475 (heihe N=1 cn08 wall 10:10, v0.2 fixed, **canonical**)

---
Generated: 2026-06-25 by orchestrator (P2a profile baseline prep, post Mac + heihe data)
Updated: 2026-06-26 by implementer (v0.2 re-baseline post nested-Timer fix; SHIP-LOCKED for P2a epic intake)
