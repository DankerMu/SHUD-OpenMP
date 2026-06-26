---
title: "P2a profile baseline — pre-CVODE forcing/ET wall 占比量化 (heihe corrected + heihe_x4 production fair-compare)"
date: 2026-06-26
version: 0.4 (heihe forcing.trimmed fair-comparison — P2a NO-GO 严谨确认)
status: "**P2a NO-GO 严谨确认** — heihe corrected (forcing 29MB local via forcing_trim) forcing+ET **13.39%** wall sp@8 上界 **1.13×**；heihe_x4 production (286MB local) forcing+ET **7.97%** wall sp@8 上界 **1.07×**。两 case 上界均 < 1.15×。v0.3 推荐保持但 §6 'heihe outlier 由 NFS IO 路径 artifact' 解读错；实际是 v0.2 heihe basin 用 12GB CMFD V0200 全时段 (1951-2024) forcing dataset 工件 (vs heihe_x4 90-day window 内 csv subset)。corrected 后 heihe 与 heihe_x4 forcing %wall 同量级，CVODE 66.69% / 80.65% wall 才是真瓶颈。建议转 P2b (RHS element vertical) / P3 (owner-local gather) / P5 (KLU 替换 SPGMR)"
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

### §3.2 Server 3 case (Slurm cn08, 含 v0.4 heihe corrected)

来源: `/scratch/frd_muziyao/SHUD-OpenMP/.p2a-profile/{heihe,heihe_v4,heihe_x4}_N1.yaml`

| Case | NumEle | forcing 部署 | forcing/ size | wall (s) | t_RHS_kernel | t_CVODE_raw | t_forcing_io | t_ET | t_output | t_other | 状态 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| heihe (v0.1) | 6335 | NFS 12GB (74yr) | 12 GB | 494.81 | 51.66 | 83.74 | **773.28** ❌ | 2.30 | 0.31 | - | nested double-count bug, **REJECTED** |
| heihe (v0.2) | 6335 | NFS 12GB (74yr) | 12 GB | 523.05 | 60.20 | 94.43 | 400.99 | 1.31 | 0.46 | - | dataset size artifact, **REJECTED for fair-compare** |
| **heihe (v0.4 corrected)** | **6335** | **local trimmed** | **29 MB** | **134.87** | **55.97** | **89.95** | **16.78** | **1.28** | **0.44** | **26.42** | **SHIP** (fair-compare) |
| **heihe_x4 (v0.3)** | **40046** | **local 90d window** | **286 MB** | **1373.23** | **763.79** | **1107.52** | **101.22** | **8.25** | **8.38** | **147.85** | **SHIP** (production target) |

heihe v0.2 job 9475 (cn08 wall 10:10, real 610.30s, NFS-bound)；heihe v0.4 job 9487 (cn08 wall 2:21, real 140.29s, local-trimmed; **缩 75% wall**)；heihe_x4 job 9483 (cn08 wall 23:22, real 1403.04s)。

**heihe v0.4 部署**: `SHUD/Basins/heihe/` 含 input/heihe/ (copy from NWM canonical), forcing/ symlink → `/volume/data/nwm/Basins/heihe/forcing/`, **forcing.trimmed/ 29 MB** (1711 csv) via `tools/forcing_trim/forcing_trim.sh heihe 14245 14335`（M7 落地工具, bash+awk, bitwise-equivalent to 12GB on 90-day window），.tsd.forc 第 2 行 absolute 指 forcing.trimmed/。**常驻** SHUD/Basins/heihe/ 不重生。

### §3.3 三 case fair-compare scaling 分析 (v0.4 修订)

**v0.3 解读错** ("heihe NFS 路径 vs heihe_x4 local NVMe") **REFUTED**。实际 v0.2 heihe basin NFS 路径只是次要因素，主因是 **forcing dataset 时段长度差异**：
- v0.2 heihe forcing/: CMFD V0200 全 74 年 (1951-2024)，单 csv ~7 MB，total 12 GB → SHUD updateforcing 每 timestep 在大 csv 中线性扫描当前时间 row → 24× wall amplification
- v0.4 heihe forcing.trimmed/: 90-day window + 2-day buffer = 94d，单 csv ~17 KB，total 29 MB → SHUD 单次 lookup 极快
- heihe_x4 forcing/: AutoSHUD pipeline 派生时已 hardcopy 3yr 子集 (2003-01 → 2006-01)，单 csv ~170 KB，total 286 MB

3 case scaling (含 v0.4 fair-compare):

| 指标 | heihe v0.2 (12GB) | heihe v0.4 (29MB) | heihe_x4 (286MB) | v0.4 vs v0.2 | x4 vs v0.4 |
|---|---:|---:|---:|---:|---:|
| forcing/ size | 12 GB | 29 MB | 286 MB | 0.0024× | 9.86× |
| wall_total | 523.05 | **134.87** | 1373.23 | **0.258×** (缩 74.2%) | 10.18× |
| t_RHS_kernel | 60.20 | 55.97 | 763.79 | 0.93× (稳定) | 13.65× |
| t_CVODE_raw | 94.43 | 89.95 | 1107.52 | 0.95× (稳定) | 12.31× |
| t_forcing_io | 400.99 | **16.78** | 101.22 | **0.042×** (缩 95.8%) | 6.03× |
| t_ET | 1.31 | 1.28 | 8.25 | 0.98× (稳定) | 6.44× |
| forcing+ET %wall | 76.91% | **13.39%** | 7.97% | **fair-compare 后两 case 同量级** | -- |
| t_CVODE_raw %wall | 18.05% | **66.69%** | 80.65% | **真实瓶颈** | -- |
| Amdahl sp@8 (P2a) | 3.06× | **1.13×** | 1.07× | **P2a 几乎无收益** | -- |

**关键发现 (v0.4 修订)**:
1. **heihe outlier 已消除**: v0.4 13.39% forcing+ET 与 heihe_x4 7.97% 同量级
2. **heihe v0.4 wall 缩 74.2%** (523s → 135s) — forcing.trimmed 直接 IO 减负反映
3. **真实 production bottleneck = CVODE** (heihe v0.4 66.69% + heihe_x4 80.65% wall)
4. **CVODE 内 RHS_kernel 仅占 41%** (heihe v0.4 56/90 = 62%; heihe_x4 764/1108 = 69%)，**剩余 CVODE_internal 30-40% 是 SPGMR + step control** — 可被 P5 (KLU 替换) 攻克
5. v0.2 heihe basin /volume NFS 路径其实可工作（heihe v0.4 forcing/ 仍 symlink 到 NFS source，是 forcing_trim 输出本地），路径不是主因 → **v0.3 §6 'NFS-bottleneck' 解读 REFUTED**

**结论 (v0.4 严谨化)**:
- production target heihe_x4 + small case heihe corrected **两者** forcing+ET 占比都 < 15% wall，sp@8 上界 < 1.15×
- **P2a 对任意 NumEle production case 几乎无收益** (不仅是 production scale)
- **CVode (含 RHS + solver internal) 70-80% wall** 才是真瓶颈 → P2b / P5 优先

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

## §5 forcing+ET wall 占比 + Amdahl sp@8 上界 (v0.4 SHIP-LOCKED)

| Case | Platform | NumEle | forcing 部署 | wall (s) | forcing+ET % | Amdahl sp@8 上界 (P2a) | t_CVODE_raw % | 备注 |
|---|---|---:|---|---:|---:|---:|---:|---|
| keliya | Mac | 484 | local 12yr+ | 30.23 | ~11.7% | ~1.11× | 86.2% | bucket÷2 推算 (pre-fix) |
| xinanjiang_upstream | Mac | 801 | local 12yr+ | 4.73 | ~8.8% | ~1.08× | 77.4% | bucket÷2 推算 |
| qinyijiang | Mac | 3155 | local 12yr+ | 285.61 | ~0.6% | ~1.00× | 98.4% | bucket÷2 推算 |
| qhh (fixed) | Mac | 4773 (+lake) | local 12yr+ | 97.19 | 30.7% | 1.39× | 56.3% | bucket-direct |
| heihe (v0.2 REJECTED) | Server | 6335 | NFS 12GB 74yr | 523.05 | 76.91% | 3.06× | 18.05% | dataset size artifact — **不可作 fair-compare** |
| **heihe (v0.4 corrected)** | **Server** | **6335** | **local trimmed 29MB 94d** | **134.87** | **13.39%** | **1.13×** | **66.69%** | **SHIP fair-compare** |
| **heihe_x4** | **Server** | **40046** | **local 286MB 3yr** | **1373.23** | **7.97%** | **1.07×** | **80.65%** | **PRODUCTION TARGET** |

**verdict (v0.4 严谨化)**: production target heihe_x4 + fair-compare heihe corrected **两 case** forcing+ET %wall **都 < 15%**，Amdahl sp@8 上界 < 1.15×。**P2a NO-GO 严谨确认** (v0.3 已正确决策，v0.4 修正原因解读)。

## §6 P2a Go/No-Go 建议 — **NO-GO 严谨确认** (v0.4)

v0.3 决策 NO-GO 正确，但 §6 原因解读不准 ("NFS-bottleneck IO 路径配置 artifact"). v0.4 用 fair-compare heihe corrected (forcing_trim 12GB → 29MB) 重测后：

**严谨原因 (v0.4 修订)**:
1. **production target heihe_x4 (NumEle=40046)** + **fair-compare heihe (NumEle=6335 corrected)** 两 case forcing+ET %wall 都 < 15%，sp@8 上界 < 1.15×
2. **v0.2 heihe 76.91% 是 forcing dataset 时段长度 artifact**：CMFD V0200 全 74yr (12GB) vs trimmed 90-day window (29MB)，缩 413× → forcing wall 缩 24× (401s → 17s)
3. **NFS 路径不是主因**：heihe v0.4 forcing/ 仍 symlink 到 /volume NFS source（forcing_trim 输出本地，但读 source 也是 NFS），路径相同但 forcing wall 大幅降 → 证明 dataset size 才是关键变量
4. **真实 production bottleneck = CVode (heihe v0.4 66.69% + heihe_x4 80.65% wall)**:
   - RHS_kernel: heihe v0.4 41.5% / heihe_x4 55.6% wall → P2b 已部分攻克 (P1e RHS sp@8 1.729×)，剩余可优化空间
   - CVODE_internal (raw - RHS): heihe v0.4 25.2% / heihe_x4 25.1% wall → SPGMR + step control，P5 (KLU) 主目标
   - `t_other` (init/finalize): heihe v0.4 19.6% / heihe_x4 10.8% wall → 小 case 占比更高 (init overhead 不 amortize)

**对 7 case 综合判断 (v0.4 update)**:

| Case | forcing+ET %wall | sp@8 P2a 上界 | P2a 单 case ROI |
|---|---:|---:|---|
| qinyijiang | 0.6% | 1.00× | 0 |
| xinanjiang | 8.8% | 1.08× | 微 |
| keliya | 11.7% | 1.11× | 微 |
| qhh | 30.7% | 1.39× | 中 |
| heihe v0.2 (REJECTED) | 76.91% | 3.06× | -- (artifact) |
| **heihe v0.4** | **13.39%** | **1.13×** | **微** |
| **heihe_x4** | **7.97%** | **1.07×** | **微 (production target)** |

P2a 对所有 fair-compare case **均无 production ROI** (max sp@8 < 1.4×)，远不及 P1e RHS (sp@8 1.729×)。

**推荐**:

| 选项 | 决策 | 理由 |
|---|---|---|
| **(a) 跳过 P2a，转 P2b / P5** | **强烈推荐** | CVode 66-80% wall 是真瓶颈。P2b (RHS element vertical 剩 ~30%) + P5 (KLU 替换 SPGMR 干掉 CVODE_internal 25%) |
| (b) 启动 P2a 但 scope 仅大 case | 不推荐 | 所有 fair-compare case (含 heihe_x4 production) ROI < 1.15×，scope-limit 无意义 |
| (c) heihe basin 部署 forcing_trim 写入 master plan | **推荐作为单独小修** | tools/forcing_trim 已 M7 落地，但 heihe / 其它 server case 部署 SOP 需文档化。本工作已 prove forcing_trim 显著缩 wall (74.2%)，未来 server baseline 应默认 forcing.trimmed (不影响 SHALL gate, bitwise-equivalent per M7 spec) |
| (d) Mac 3 case re-run with 7a1dc8f | 可选 | 若 P2b/P5 epic 需要精确 keliya / xinanjiang / qinyijiang 占比 |

## §7 next step

1. ~~**P2a-0 (audit-required prep PR)**: 修 `t_forcing_io` bucket bug + re-run heihe baseline~~ — **DONE** in SHUD `7a1dc8f` + outer pin bump
2. ~~**P2a-A (epic intake PR)**~~ — **CANCELLED per §6 NO-GO 严谨确认**
3. **master plan §P2a 重写**: 标 "P2a 不启动 (forcing_trim 已 M7 落地解决 dataset size, P2a parallelization 无 ROI)"，转 P2b / P5 优先
4. **stage-change-pipeline**: P2b (RHS element vertical 剩余) 或 P5 (KLU 替换 SPGMR) epic intake (待 user 确认转向)
5. **heihe basin 部署 forcing_trim 写入 deployment SOP**: 本工作已 prove `tools/forcing_trim/forcing_trim.sh heihe 14245 14335` 输出 29MB 本地 trimmed forcing (vs 12GB NFS source)，wall 缩 74.2%。未来 server case baseline 部署默认 forcing.trimmed。**已常驻** SHUD/Basins/heihe/forcing.trimmed (29MB) **禁止重生**
6. **(optional) Mac 3 case re-run with `7a1dc8f`**: 若后续 P2b/P5 epic 需要精确 keliya / xinanjiang / qinyijiang 占比，可单独 re-run (15 min)

## §8 引用

- master plan §P2a.1-.8 (P1e 经验重梳理后版) — **本 doc 触发 §P2a 取消决策**
- docs/p1e/p1e_perf_baseline.md §6 三因素分析 (OMP overhead floor 经验)
- docs/profile_decision.md (B0 era profile gate, t_RHS_total% 4 case)
- tools/profile/timer.h (7 预定 bucket)
- `SHUD@7cc46d8` (Step 1 instrumentation hooks at L208/L215/L228/L266) — v0.1 baseline, bug-laden
- `SHUD@7a1dc8f` (Step 2 nested-Timer fix: MD_ET.cpp + Model_Control.cpp) — v0.2 / v0.3 SHIP-LOCKED
- Slurm job 9448 (heihe N=1 cn11 wall 9:21, v0.1 anomaly)
- Slurm job 9475 (heihe N=1 cn08 wall 10:10, v0.2, dataset size artifact, REJECTED for fair-compare)
- Slurm job 9483 (heihe_x4 N=1 cn08 wall 23:22, v0.3 production target, canonical)
- **Slurm job 9487 (heihe N=1 cn08 wall 2:21, v0.4 corrected via forcing_trim 12GB→29MB, fair-compare canonical)**
- `tools/forcing_trim/forcing_trim.sh` (M7 spec p1-update-omp/m7-forcing-trim, bash+awk POSIX, bitwise-equivalent on 90-day window)

---
Generated: 2026-06-25 by orchestrator (P2a profile baseline prep, post Mac + heihe data)
Updated: 2026-06-26 by implementer (v0.2 re-baseline post nested-Timer fix)
Updated: 2026-06-26 by orchestrator (v0.3 + heihe_x4 production target — P2a NO-GO verdict 反转)
Updated: 2026-06-26 by orchestrator (v0.4 + heihe forcing.trimmed fair-compare — **NO-GO 严谨确认 + v0.3 NFS-artifact 解读 REFUTED**)
