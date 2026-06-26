---
title: "P2a profile baseline — pre-CVODE forcing/ET wall 占比量化 (含 heihe_x4 production target 校准)"
date: 2026-06-26
version: 0.3 (heihe_x4 production target added — P2a NO-GO verdict)
status: "**P2a NO-GO** — heihe_x4 (production target, NumEle=40046) forcing+ET 仅 7.97% wall，sp@8 Amdahl 上界 1.075×。heihe (6335) 76.92% 是 outlier，由 small-case forcing IO NFS-bottleneck 主导。建议转 P2b (RHS element vertical) / P3 (owner-local gather) / P5 (KLU 替换 SPGMR) 攻 CVode+RHS 80.7% wall 真瓶颈"
related_docs:
  - "docs/p1e/p1e_perf_baseline.md (RHS = 66.55% wall reference)"
  - "docs/profile_decision.md (B0 era profile gate, t_RHS_total% baseline)"
  - "SHUD_openMP_master_plan.md §P2a.2 (2×2 build matrix + profile) — 本 doc 触发 P2a 不启动决策"
---

# P2a profile baseline (v0.3 含 heihe_x4)

## §1 目的

P2a (并行 pre-CVODE forcing / ET loop) epic intake 前的 ROI 量化前置工作。在 SHUD `shud.cpp` 主循环 4 处补 `shud_profile::Timer` 插桩（per master plan §P2a.2 + tools/profile/timer.h 预定 buckets），跑 5 case × N=1 × 1 rep，量化 forcing + ET 在 wall 中的占比，判断 P2a 是否值得进入 14-PR 模板。

v0.3 增补 server **heihe_x4 production target** 数据（之前 v0.1/v0.2 仅 heihe small case），反转 go/no-go 决策。

## §2 实验设置

| 项 | 值 |
|---|---|
| SHUD pin (v0.1 initial) | `7cc46d8` (P1e `3341368` + Step 1 instrumentation hooks) — surfaced bucket double-count bug |
| SHUD pin (v0.2/v0.3 fixed) | `7a1dc8f` (`7cc46d8` + nested-Timer removal in MD_ET.cpp / Model_Control.cpp) — **current ship pin** |
| Build | `make shud SHUD_ENABLE_PROFILE=1` (no OMP, N=1 串行) |
| Cases | 4 Mac case (keliya / xinanjiang / qinyijiang / qhh) + 2 server case (**heihe** + **heihe_x4**) |
| Threads | N=1, OMP_NUM_THREADS=1, SHUD_RHS_THREADS=1 |
| Reps | 1 (profile baseline, 不验收) |
| Truncation | 90-day (per CLAUDE.md 项目铁律) |
| Mac runtime | Apple M4 Pro arm64, libomp 22.1.7 |
| Server runtime | Intel Xeon cn08, x86_64, libgomp gcc 13.3 |
| **heihe_x4 case basin (server)** | **`/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x4/`** (常驻 2.3G, AutoSHUD 2026-06-17 生成, NumEle=40046, ~6.3× heihe 6335; **不可现场重生**) |

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

### §3.2 Server 2 case (Slurm cn08)

来源: `/scratch/frd_muziyao/SHUD-OpenMP/.p2a-profile/{heihe,heihe_x4}_N1.yaml`

| Case | NumEle | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output | t_other | 状态 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| heihe (v0.1) | 6335 | 494.81 | 51.66 | 83.74 | **773.28** ❌ | 2.30 | 0.31 | - | nested double-count bug, **REJECTED** |
| **heihe (v0.2)** | **6335** | **523.05** | **60.20** | **94.43** | **400.99** | **1.31** | **0.46** | - | **SHIP** (small-case outlier, NFS-bound) |
| **heihe_x4 (v0.3)** | **40046** | **1373.23** | **763.79** | **1107.52** | **101.22** | **8.25** | **8.38** | **147.85** | **SHIP** (production target, RHS-bound) |

heihe job 9475 (cn08 wall 10:10, real 610.30s)；heihe_x4 job 9483 (cn08 wall 23:22, real 1403.04s)。

### §3.3 heihe (6335) vs heihe_x4 (40046) scaling 反转分析

**关键反转**: NumEle 6.3× scaling 之后 forcing+ET 占比从 76.92% 暴降到 7.97%。

| 指标 | heihe (6335) | heihe_x4 (40046) | 比例 | 解读 |
|---|---:|---:|---:|---|
| wall_total | 523.05 | 1373.23 | 2.63× | sublinear (CVode internal step density 不变 + cache 友好) |
| t_RHS_kernel | 60.20 | 763.79 | **12.69×** | **superlinear** (cells × CVode internal step density 增加) |
| t_CVODE_raw | 94.43 | 1107.52 | 11.73× | 含 RHS, 与 RHS 同步 superlinear |
| t_forcing_io | 400.99 | 101.22 | **0.25×** | **逆向减少!** heihe NFS 远端读 bottleneck，heihe_x4 basin-local forcing/ cache 命中 |
| t_ET | 1.31 | 8.25 | 6.3× | linear cells scaling (per-element ET physics) |
| forcing+ET %wall | 76.92% | **7.97%** | -- | **完全反转** |
| t_CVODE_raw %wall | 18.05% | **80.65%** | -- | **CVode 成为绝对主导 bottleneck** |
| Amdahl sp@8 (P2a, forcing+ET only) | 3.06× | **1.075×** | -- | **P2a 对 production 几乎无收益** |

**heihe outlier 解释**:
- heihe basin (6335 cells) forcing 走 `/volume/data/ForcingData/CMFD2.0/Data_forcing_03hr_010deg/*.nc` NFS 远端读 (CMFD V0200 网格 0.1° × 0.1°，每个 timestep 读取 8 vars × 248 stations × 720 timesteps/month)
- heihe_x4 basin (40046 cells) basin 内 2.3G 含 `forcing/` 子目录 (canonical 路径硬编码 .tsd.forc 第 2 行，AutoSHUD 派生时 hard-copy)，本地 NVMe 命中
- 即 heihe 76.92% **不是 forcing IO 物理特性**，是 **IO 路径配置工件**，**不可外推到 production**

**结论**: production target heihe_x4 真实瓶颈在 **CVode + RHS (80.65% wall)**，**不是 forcing+ET**。

## §4 instrumentation bug — **FIXED in SHUD `7a1dc8f`**

(v0.2 内容保留，本 v0.3 不改)

**v0.1 anomaly**: heihe `t_forcing_io = 773.28s` > `t_wall_total = 494.81s`，物理不可能。

### §4.1 真实 root cause

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

## §5 forcing+ET wall 占比 + Amdahl sp@8 上界 (v0.3 SHIP-LOCKED)

| Case | Platform | NumEle | wall (s) | forcing+ET % | Amdahl sp@8 上界 (P2a) | t_CVODE_raw % | 备注 |
|---|---|---:|---:|---:|---:|---:|---|
| keliya | Mac | 484 | 30.23 | ~11.7% | ~1.11× | 86.2% | bucket÷2 推算 (pre-fix) |
| xinanjiang_upstream | Mac | 801 | 4.73 | ~8.8% | ~1.08× | 77.4% | bucket÷2 推算 |
| qinyijiang | Mac | 3155 | 285.61 | ~0.6% | ~1.00× | 98.4% | bucket÷2 推算 |
| qhh (fixed) | Mac | 4773 (+lake) | 97.19 | 30.7% | 1.39× | 56.3% | bucket-direct |
| **heihe (fixed)** | **Server** | **6335** | **523.05** | **76.92%** ⚠️ | **3.06×** | **18.0%** | **OUTLIER** — NFS forcing IO bottleneck, 不可外推 |
| **heihe_x4 (NEW)** | **Server** | **40046** | **1373.23** | **7.97%** | **1.075×** | **80.65%** | **PRODUCTION TARGET — P2a 无收益, CVode 主导** |

**verdict**: production-target heihe_x4 **forcing+ET 仅 7.97%** wall，P2a 并行 forcing/ET 无 ROI。**P2a NO-GO**。

## §6 P2a Go/No-Go 建议 — **NO-GO** (v0.3 反转)

v0.2 推荐 "强烈启动 P2a"，基于 heihe 76.92%。v0.3 加入 production target heihe_x4 后反转：

**反转理由**:
1. **production target heihe_x4 (NumEle=40046, sp@8 上界 1.075×)** — P2a 并行 forcing/ET 几乎无收益
2. **heihe 76.92% 是 outlier**：basin 内无 forcing/ 本地 cache，强制走 NFS `/volume/data/ForcingData/CMFD2.0/` 远端读，与 IO 路径配置工件相关，不是 forcing+ET 物理瓶颈
3. **真实 production bottleneck 是 CVode (heihe_x4 80.65% wall)**：含 RHS_kernel 55.6% + CVODE_internal 25%；RHS 已是 P1e 攻克目标 (P1e sp@8 = 1.729×)，CVODE_internal 25% 是 SPGMR + step control 子模块 → 应攻 solver-internal，不攻 pre-CVODE

**对 5 case 综合判断**:

| Case | forcing+ET %wall | sp@8 P2a 上界 | P2a 单 case ROI |
|---|---:|---:|---|
| qinyijiang | 0.6% | 1.00× | 0 |
| xinanjiang | 8.8% | 1.08× | 微 |
| keliya | 11.7% | 1.11× | 微 |
| qhh | 30.7% | 1.39× | 中 |
| heihe | 76.92% | 3.06× | 高 (但 outlier IO 配置工件) |
| **heihe_x4** | **7.97%** | **1.075×** | **微 (production target)** |

P2a 仅对 IO-bottleneck outlier (heihe) 有效，对其它 5 case (4 Mac + heihe_x4 production) ROI < 1.4×。

**推荐 (取代 v0.2 §6 选项 a)**:

| 选项 | 决策 | 理由 |
|---|---|---|
| **(NEW a) 跳过 P2a，转 P2b / P3 / P5** | **强烈推荐** | heihe_x4 CVode 80.65% wall 是真瓶颈。P2b (RHS element vertical, 已 P1e 部分落地剩 ~10%) + P3 (owner-local gather, sum reduction) + P5 (KLU 替换 SPGMR, 干掉 25% CVODE_internal) |
| (b) 仍启动 P2a 但 scope 仅 heihe outlier 修 | 不推荐 | IO 路径配置工件，正解是改 heihe basin 部署 forcing/ 本地 cache (复用 heihe_x4 的 hard-copy 模式)，而不是并行化 |
| (c) 重新派生 heihe 加 forcing/ 本地 cache | **推荐作为单独小修** | tools/mesh_refine 加 `--with-forcing-cache` flag，AutoSHUD 派生时 hard-copy CMFD 切片到 basin/forcing/。预计 wall 从 523s → ~200s（heihe forcing 400s 大半归零） |

## §7 next step

1. ~~**P2a-0 (audit-required prep PR)**: 修 `t_forcing_io` bucket bug + re-run heihe baseline~~ — **DONE** in SHUD `7a1dc8f` + outer pin bump
2. ~~**P2a-A (epic intake PR)**~~ — **CANCELLED per §6 NO-GO**
3. **master plan §P2a 重写**: 标 "P2a 不启动"，转 P2b / P3 / P5 优先；保留 §P2a 文献位置作为 historical alternative
4. **stage-change-pipeline**: 改为 P2b 或 P3 epic intake (待 user 确认转向)
5. **(optional) 单独小修 heihe basin forcing/ 本地 cache**: 复用 heihe_x4 部署模式 (hard-copy CMFD 切片入 basin/forcing/)，预计 heihe wall 从 523s → ~200s。非 P2a epic，仅 deployment 改进
6. **(optional) Mac 3 case re-run with `7a1dc8f`**: 若后续 P2b/P3 epic 需要精确 keliya / xinanjiang / qinyijiang 占比，可单独 re-run (15 min)

## §8 引用

- master plan §P2a.1-.8 (P1e 经验重梳理后版) — **本 doc 触发 §P2a 取消决策**
- docs/p1e/p1e_perf_baseline.md §6 三因素分析 (OMP overhead floor 经验)
- docs/profile_decision.md (B0 era profile gate, t_RHS_total% 4 case)
- tools/profile/timer.h (7 预定 bucket)
- `SHUD@7cc46d8` (Step 1 instrumentation hooks at L208/L215/L228/L266) — v0.1 baseline, bug-laden
- `SHUD@7a1dc8f` (Step 2 nested-Timer fix: MD_ET.cpp + Model_Control.cpp) — v0.2 / v0.3 SHIP-LOCKED
- Slurm job 9448 (heihe N=1 cn11 wall 9:21, v0.1 anomaly)
- Slurm job 9475 (heihe N=1 cn08 wall 10:10, v0.2 fixed)
- **Slurm job 9483 (heihe_x4 N=1 cn08 wall 23:22, v0.3 production target, canonical)**

---
Generated: 2026-06-25 by orchestrator (P2a profile baseline prep, post Mac + heihe data)
Updated: 2026-06-26 by implementer (v0.2 re-baseline post nested-Timer fix)
Updated: 2026-06-26 by orchestrator (v0.3 + heihe_x4 production target — **P2a NO-GO verdict 反转**)
