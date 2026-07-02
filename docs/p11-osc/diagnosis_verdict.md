---
title: "P11-osc 诊断裁决：qinyijiang / keliya 分钟级 CVODE 步进——数值振荡 vs 真实快动力学"
subtitle: "3-case 证据矩阵 + kill-gate 裁决 + REAL_DYNAMICS 闭合决定（诊断阶段 capstone）"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-07-01
version: 1.0 (P11-osc PR-D3 diagnosis verdict)
epic: "#433 (P11-osc); PR-D3 = #436"
verdict: REAL_DYNAMICS
control_sanity: PASS
gate_decision: "epic closure + SHUD p11-osc merge-back recommendation (instrumentation retained, default-off)"
related_docs:
  - "docs/p11-osc/spike_brief.md (kill-gate 单一来源 + case matrix + §Optimization 约束)"
  - "openspec/changes/p11-osc/design.md (§Verdict gate + §Risks R2/R3/R4 + §Execution topology)"
  - "openspec/changes/p11-osc/specs/osc-verdict-gate/spec.md (本 PR 的验收契约)"
  - "tools/osc_diag/README.md (analyzer CLI + verdict 决策函数)"
  - ".review-evidence/p11-osc-diag/pr-d1/README.md (bitwise gate + self-check)"
  - ".review-evidence/p11-osc-diag/pr-d2/README.md (emitter↔parser contract smoke)"
  - ".review-evidence/p11-osc-diag/pr-d3/ (本裁决的 3-case 原始证据)"
  - "docs/adr/0008-*.md / 0009-*.md / 0010-*.md (已关闭 CPU 加速线语境)"
---

# Abstract / 摘要

P11-osc spike 诊断 `qinyijiang` / `keliya` 两案例在 90 天 benchmark 窗口内
CVODE 以分钟尺度步进（较同窗 `heihe` 多约 19× 步数）的成因：究竟是可阻尼的
**数值通量振荡**（H-osc），还是**真实快水文动力学**（H-dyn）。方法为 PR-D1
植入的 read-only、default-bitwise-neutral 双诊断（`SHUD_DIAG_DT` 每-interval
dt/step 计数轨迹 + `SHUD_DIAG_OSC` accepted-boundary 状态符号翻转计数），经
PR-D2 分析器（`tools/osc_diag/`）产出 machine-readable kill-gate 裁决。3-case
本地 serial 证据矩阵（primary=qinyijiang→`nanlin`；corroborating=keliya；
healthy control=xinanjiang_upstream→`xinanjiang`）结果：**control sanity 通过**
（xinanjiang_upstream `burst_share_60s = 0.0000 < 0.05`），故 epic 裁决有效；
**epic 裁决 = REAL_DYNAMICS**（qinyijiang MARKER verbatim）。决定性判据是
top-1% element flip 空间集中度仅 **0.0366 < 0.50**——分钟级步进 burst 真实
（`burst_share_60s = 0.9942`），但翻转在全盆地 3,155 个 element 上近乎均匀
分布，而非局部化于少数振荡单元；per-day Spearman ρ = **−0.3753 < 0.5**，翻转
峰与 sub-60 s dt 日之间无正向机械联系。人工评估的 forcing-tracking 佐证与
machine 裁决一致：qinyijiang 翻转日 top-2（day 446/447）恰为降雨日 top-2
（26.2 / 30.6 mm·d⁻¹），翻转峰跟降雨事件走而非跟纯数值 dt-collapse 走。结论：
qinyijiang / keliya 的分钟级步进是**诚实的真实快动力学**，任何限速器都将
falsify physics——per spike_brief kill-gate，该 optimization 线**关闭**，
**零限速器代码落地**。SHUD `p11-osc` instrumentation 分支建议 merge 回
`openmp-baseline` 作为 opt-in 诊断保留（default-off、read-only、bitwise-neutral，
PR-D1 四腿证据已证）。裁决仅适用于 90 天 benchmark 窗口（R2 window
representativeness threat，见 §7）。

**Keywords**: SHUD; CVODE; minute-scale stepping; numerical oscillation; real
fast dynamics; burst share; flip counters; kill-gate; REAL_DYNAMICS

---

# §1 Introduction / 引言

## §1.1 问题背景

P1e epic 关闭后（StrictOMP RHS 生产基线），并行 RHS 攻击的是**每步成本**，
无法触及**步数**。B0 archive 证据（90 天窗口，serial Config A）显示步数
`nst` 是 forcing/physics 驱动而非 mesh 驱动：`heihe`（6,335 ele）与
`heihe_x4`（40,046 ele）取几乎相同的 ~6.57k 步。真正的 ≥10× CPU 机会（在
ADR-0008/0009/0010 三条线关闭后仅剩此一处）在于 `qinyijiang`（nst≈127k，
est.）/ `keliya`（nst≈101k）的分钟级步进：若 nst 能降到 heihe-like ~6.5k，
wall 缩约 19×。但这仅在**振荡为数值**时才是合法、physics-preserving 的加速；
若为真实快动力学，阻尼即 falsify physics，该线必须关闭。

关键悖论（spike_brief §Motivating evidence 第 3 点）：CVODE 在 qinyijiang 上
**并不 struggle**——netf=0、ncfn≈0.06%，误差控制器是"心满意足地"走 1 分钟步。
聚合统计无法区分两假设，需 read-only 的 dt/翻转诊断定位。

## §1.2 形式化假设（spike_brief）

- **H-osc（数值）**：inter-cell 通量振荡（横向 surface/subsurface 通量符号
  翻转、wet/dry 阈值 chatter、river–element 交换反向）使局部误差估计居高不下，
  把 dt 钉在分钟级。阻尼振荡是合法的 physics-preserving 加速。
- **H-dyn（物理）**：盆地水文响应在 benchmark 窗口内确实运行在分钟时间尺度
  （暴雨脉冲、薄土层、陡峭河道）。dt 是诚实的，阻尼将 falsify physics，该线关闭。

## §1.3 裁决聚合规则（design.md §Verdict gate / osc-verdict-gate spec）

- Epic 裁决 = **qinyijiang（project `nanlin`）MARKER 裁决 verbatim**；
- 仅当 **xinanjiang_upstream control sanity 通过**（`burst_share_60s < 0.05`）时
  epic 裁决**有效**（否则诊断本身可疑 → verdict BLOCKED）；
- **keliya 仅作 corroborating evidence**——keliya/qinyijiang 分歧在本文档讨论，
  但**不 override** 基于 qinyijiang 的 epic 裁决。

---

# §2 Related Work / 相关工作（已关闭线语境）

本 epic 刻意不占用任何已关闭线的 namespace：

- **ADR-0008（P8 CLOSED-FINAL）**：global linear solver substitution 线关闭
  （AMG-rescue 假设被 G0-RCA 反驳）。本 epic 不做 solver 替换。
- **ADR-0009（P9 CLOSED）**：reltol-family retuning 线关闭
  （wall_speedup=1.025 < 1.2× gate）。本 epic 不做 reltol 重调。
- **ADR-0010（program status; P10 design-gated）**：GPU / domain decomposition
  保持 design-gated。本 epic **不复用 P10 name、不触其 gate**。

P11-osc 的独特定位：不改 solver、不改 tolerance、不做并行——而是问"分钟级步进
本身是否可以（合法地）消除"。若 REAL_DYNAMICS，则答案是"不能"，与上述三条
关闭线一起，确认 P1e StrictOMP + SPGMR_MAXL small-case opt-in 是 CPU 加速的
生产终点。

---

# §3 Methodology / 方法论

## §3.1 诊断双探针（PR-D1，read-only、default-off）

- **D1 `SHUD_DIAG_DT=1`**：driver loop（`shud.cpp`，post-CVode-return）每
  `CS.SolverStep` interval 发一行 `t_next, delta_nst, delta_nfe, delta_ncfn,
  delta_netf, h_last, h_cur`。`CVodeGetNumSteps` / `GetLastStep` /
  `GetCurrentStep` 的 delta 均 read-only，无 solver 状态突变、无轨迹风险。
- **D2 `SHUD_DIAG_OSC=1`**：每 interval 边界 diff CVODE 状态向量
  （`N_VGetArrayPointer(udata)`，layout `[sf|us|gw|riv]`，**绝不**碰
  `uY*` RHS scratch globals），对 interval delta 的符号交替计数
  （rise→fall→rise = 1 flip；|delta| ≤ 1e-6 m epsilon dead-floor 保持前号）。
  输出 per-entity 累计（typed `ele`/`riv`）+ per-day 聚合（`day_index =
  floor(t_next/1440)`）。
- **strict `=1` 门 + bitwise 中性**：PR-D1 四腿证据（env unset / `=0`×2 /
  `=1`）证 keliya B0 model-output SHA pre/post 一致，enabled 时也 bitwise
  中性（read-only）。

## §3.2 kill-gate 决策函数（single source；pinned，nothing decided in-tool）

canonical 阈值取自 spike_brief §Kill-gate = design.md §Verdict gate =
osc-diag-analyzer spec（三处同文）：

```
OSC_CONFIRMED = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50 AND ρ >= 0.5
REAL_DYNAMICS = burst_share_60s <  0.50 OR  top1pct_concentration <  0.50
INCONCLUSIVE  = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50 AND ρ <  0.5
```

- **burst_share_60s**：mean-dt < 60 s 的 interval 承载的 nst 占比
  （`mean_dt_seconds = 60 * solverstep_min / delta_nst`；delta_nst=0 行跳过）。
  加速上界估计：消除 burst 的 wall 增益上限 = `1/(1 − burst_share)`。
- **top-1% element concentration**：population = elements only（per-element
  flips = sf+us+gw 求和）；river 行单独报告、排除在集中度分母外。集中度 =
  top `ceil(0.01 × NumEle)` element 承载的 element flips 占比。
- **per-day Spearman ρ**：daily `flips_total` vs daily sub-60 s interval count
  的秩相关（共享 day_index 键；trailing partial-day bucket 对称剔除）。
  NaN ρ（太少天/常数序列）fail ρ≥0.5 门 → 保守走 INCONCLUSIVE，绝不
  OSC_CONFIRMED。

## §3.3 forcing-tracking（human-assessed，NOT machine-verdict input）

无任何 forcing 序列进入分析器（design.md §Verdict gate 明确）。翻转峰是否
跟降雨峰走，是本文档记录的**人工评估佐证**，不改变 MARKER 裁决。方法：对每
case 读全部 station CSV（`SHUD/Basins/<basin>/forcing/X*Y*.csv`，`Precip_mm.d`
列，3-hourly cadence），`Time_interval` 列值即 model day_index（1:1 对齐，
explorer 确认），在 `cfg.para START..END` 窗口内求 basin-mean 日均降雨率，
与 daily flip totals / per-day sub-60 s count 对齐。

---

# §4 Experimental Setup / 实验设置

- **平台**：本地 Apple Silicon Mac（design.md §Execution topology：全本地、
  无 Slurm；诊断判定明确以本地为准）。
- **构建**：`SHUD/` submodule pin `75afb2b`（SHUD 分支 `p11-osc`），
  `cd SHUD && make shud` serial（Config A）——Config C 冗余，因 instrumentation
  在 RHS 之外且 P1e A3a bitwise 等价 pins serial == Config C 轨迹。
- **窗口**：全部 90 天（项目铁律；cfg.para 已部署截断，未修改）。
  qinyijiang(nanlin) day 366–456、keliya day 12053–12143、
  xinanjiang(xinanjiang) day 0–90。所有 case `SolverStep = 20 min`
  （由 trace header 读出，design R5）。
- **运行**：从各自 basin dir、双 env 门开：
  `cd SHUD/Basins/<basin> && SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1 ../../shud <project>`。
- **healthy-control 槽 = xinanjiang_upstream，非 heihe**：heihe 是
  `endpoint: server-only`（`benchmarks/heihe/manifest.yaml`），无 Mac 部署
  （`docs/case_deployment_map.md` §2.1）；xinanjiang_upstream 是同等健康 profile
  且本地可跑。
- **分析器**：`cd tools/osc_diag && uv run osc_diag --diag-dir <...>/<project>.out
  --case <case> --out <evidence-dir>`；`--case qinyijiang` 将 project_name
  `nanlin` 映射到 benchmark case（分析器 exit 0 / MARKER block）。

---

# §5 Results / 结果

## §5.1 3-case 证据矩阵（Tab.1）

**Tab.1** — 3-case gate inputs + verdict（本地 serial，双诊断开，90 天窗口）。
所有数值 verbatim 摘自 `.review-evidence/p11-osc-diag/pr-d3/<case>/marker_block.txt`
+ `osc_diag_summary.json`；cvode_stats 摘自各 run 的 `cvode_stats.txt`。

| case | role | NumEle | nst | wall (s) | netf | ncfn/nst | **burst_60s** | burst_10s | **top1% conc** | **Spearman ρ** | **verdict** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **qinyijiang** (`nanlin`) | **primary** | 3,155 | 156,580 | 293 | 0 | 0.07% | **0.9942** | 0.0000 | **0.0366** | **−0.3753** | **REAL_DYNAMICS** |
| keliya | corroborating | 484 | 111,130 | 33 | 7 | 0.20% | 0.0019 | 0.0000 | 0.3766 | 0.1285 | REAL_DYNAMICS |
| xinanjiang_upstream (`xinanjiang`) | **control** | 801 | 6,775 | 6 | 3 | 0.04% | **0.0000** | 0.0000 | 0.2294 | NaN | REAL_DYNAMICS |

裁决门评估（qinyijiang，epic verdict source）：
- Gate 1 `burst_share_60s = 0.9942 ≥ 0.50` → **PASS**（分钟级步进 burst 真实）。
- Gate 2 `top1pct_concentration = 0.0366 < 0.50` → **FAIL**（翻转空间弥散，
  非局部化）→ 依 `REAL_DYNAMICS = ... OR concentration < 0.50` 即判定。
- Gate 3 `spearman_rho = −0.3753 < 0.5` → **FAIL**（无正向机械联系；负号
  进一步反证 dt-collapse 驱动振荡）。
- 组合 → **REAL_DYNAMICS**。

## §5.2 MARKER blocks（verbatim）

**qinyijiang（epic verdict source）** —
`.review-evidence/p11-osc-diag/pr-d3/qinyijiang/marker_block.txt`：

```
MARKER:OSC_DIAG_VERDICT_BEGIN
case=qinyijiang
verdict=REAL_DYNAMICS
burst_share_60s=0.9942
burst_share_60s_threshold=0.5000
burst_share_10s=0.0000
top1pct_concentration=0.0366
top1pct_concentration_threshold=0.5000
spearman_rho=-0.3753
spearman_rho_threshold=0.5000
MARKER:OSC_DIAG_VERDICT_END
```

**keliya（corroborating only）** —
`.review-evidence/p11-osc-diag/pr-d3/keliya/marker_block.txt`：

```
MARKER:OSC_DIAG_VERDICT_BEGIN
case=keliya
verdict=REAL_DYNAMICS
burst_share_60s=0.0019
burst_share_60s_threshold=0.5000
burst_share_10s=0.0000
top1pct_concentration=0.3766
top1pct_concentration_threshold=0.5000
spearman_rho=0.1285
spearman_rho_threshold=0.5000
MARKER:OSC_DIAG_VERDICT_END
```

**xinanjiang_upstream（control sanity source）** —
`.review-evidence/p11-osc-diag/pr-d3/xinanjiang_upstream/marker_block.txt`：

```
MARKER:OSC_DIAG_VERDICT_BEGIN
case=xinanjiang_upstream
verdict=REAL_DYNAMICS
burst_share_60s=0.0000
burst_share_60s_threshold=0.5000
burst_share_10s=0.0000
top1pct_concentration=0.2294
top1pct_concentration_threshold=0.5000
spearman_rho=NaN
spearman_rho_threshold=0.5000
MARKER:OSC_DIAG_VERDICT_END
```

## §5.3 Control sanity gate（HARD）

xinanjiang_upstream `burst_share_60s = 0.0000 < 0.05` → **PASS**。健康 profile
确认：dt histogram 91.9% nst 落在 `[1200,∞)` s bin（mean dt > 20 min），仅
1 个 interval 在 `[60,300)` s，**0 个 sub-60 s interval**；ρ = NaN 正因无任何
sub-60 s 日（常数零序列，Spearman 未定义）——这本身即健康证据。故 epic
裁决（REAL_DYNAMICS）**有效**。

## §5.4 dt histogram 摘要（Tab.2）

**Tab.2** — 各 case 的 nst_share 分布（interval mean-dt bins，秒）。源：
`.review-evidence/p11-osc-diag/pr-d3/<case>/dt_histogram.csv`。

| bin (s) | qinyijiang nst_share | keliya nst_share | xinanjiang nst_share |
|---|---:|---:|---:|
| [0,10) | 0.0000 | 0.0000 | 0.0000 |
| [10,60) | **0.9942** | 0.0019 | 0.0000 |
| [60,300) | 0.0045 | **0.9891** | 0.0010 |
| [300,600) | 0.0013 | 0.0089 | 0.0165 |
| [600,1200) | 0.0000 | 0.0001 | 0.0632 |
| [1200,∞) | 0.0000 | 0.0000 | **0.9193** |

三案例形成清晰梯度：qinyijiang 极端分钟级（99.4% 在 [10,60) s）、keliya
"just-above-60 s"堆积（98.9% 在 [60,300) s，故 burst_share_60s 反而极低）、
xinanjiang 健康（91.9% > 20 min）。

## §5.5 flip 空间分布（Tab.3）

**Tab.3** — top-1% element flip 集中度 detail。源：各 case
`osc_diag_summary.json.concentration_detail`。

| case | NumEle | top_k (1%) | total_ele_flips | riv_flips (报告用) | top-1% concentration |
|---|---:|---:|---:|---:|---:|
| qinyijiang | 3,155 | 32 | 452,969 | 90,967 | **0.0366** |
| keliya | 484 | 5 | 52,367 | 10,015 | 0.3766 |
| xinanjiang | 801 | 9 | 221,425 | 14,748 | 0.2294 |

qinyijiang 的 452,969 element flips 里 top-32 element 仅承载 3.66%——翻转
在 3,155 个 element 上**近乎均匀弥散**，这是"全盆地真实快响应"而非"少数
振荡单元"的决定性空间证据。

## §5.6 human-assessed forcing-tracking 佐证（NOT a machine input）

> **声明**：本节为人工评估佐证，无 forcing 数据进入分析器；不改变 §5.1–5.3
> 的 MARKER 裁决。源：`.review-evidence/p11-osc-diag/pr-d3/<case>/{basin_mean_daily_precip.csv,
> perday_subdt_vs_flips.csv}`。

**qinyijiang（primary）**——翻转峰跟降雨峰走：

**Tab.4** — qinyijiang top flip days vs 同日 basin-mean 降雨率。

| day_index | flips_total | mean_precip (mm·d⁻¹) | 备注 |
|---:|---:|---:|---|
| 446 | 17,712 | 26.17 | flip #1 / precip #2 |
| 447 | 16,028 | 30.58 | flip #2 / **precip #1** |
| 445 | 14,594 | 9.78 | 降雨事件簇边缘 |
| 397 | 13,221 | 10.63 | 次级降雨日 |
| 448 | 11,960 | 19.56 | precip #5 |
| 451 | 11,742 | 7.10 | 事件簇尾 |

qinyijiang 翻转日 top-2 = 降雨日 top-2；day 438–452 是明显降雨事件簇，翻转
top-8 中 6 天（446/447/445/438/448/451）落于此簇（仅 397/405 在外）。
**翻转峰跟降雨事件走**，与 H-dyn（暴雨脉冲驱动全盆地
快响应）一致。同时须诚实指出：sub-60 s stepping 是**全程基线**（91/91 天皆有
sub-60 s interval，6,344/6,480 interval 为 sub-60 s），并非仅降雨日发生——即
盆地整体运行在分钟尺度，降雨事件只是在此基线上叠加翻转峰。两点都指向物理，
不指向可局部阻尼的数值振荡。

**keliya（corroborating）**——翻转峰同样跟降雨走，dt-collapse 与降雨反相关：

| 现象 | day_index | 数值 |
|---|---:|---:|
| flip 峰 top-2 | 12141 / 12142 | flips 3,199 / 2,440 |
| 同日降雨（= 全窗降雨 top-2） | 12141 / 12142 | 1.14 / 1.00 mm·d⁻¹ |
| 唯一 sub-60 s 日 | 12054 | 该日降雨仅 0.0067 mm·d⁻¹ |

keliya 翻转峰 = 全窗降雨峰；而唯一的 dt-collapse 日（12054）几乎无雨——
dt-collapse 与降雨**反相关**，说明少量 sub-60 s interval 并非降雨振荡所致。
这与 keliya machine 裁决（burst_share_60s=0.0019 极低 + concentration<0.50）
一致收敛到 REAL_DYNAMICS。

**xinanjiang_upstream（control）**——健康求解器面对真实降雨也不 collapse：
窗口内有显著降雨事件（day 72 达 34.2 mm·d⁻¹），但 **0 天**出现 sub-60 s
interval。证明诊断探针不会在健康 case 上误报 burst。

**诚实度标注**：qinyijiang 的 forcing-tracking 信号方向明确（翻转峰↔降雨峰
正向重合，且全程分钟级基线）；本佐证与三条 machine gate 独立收敛到同一结论，
不存在需要"和稀泥"的模糊信号。

---

# §6 Discussion / 讨论

## §6.1 假设验证：H-dyn 成立，H-osc 被反驳

三条独立证据链一致指向 H-dyn：
1. **空间弥散**（Tab.3）：top-1% concentration=0.0366，翻转遍布全盆地
   3,155 element——数值振荡应局部化于 wet/dry 界面或特定 edge 对，弥散分布
   与之矛盾。
2. **无正向时间联系**（§5.1 Gate 3）：ρ=−0.3753，翻转峰与 sub-60 s dt 日
   负相关——若振荡是 dt-collapse 的成因，应正相关。
3. **forcing-tracking**（§5.6）：翻转峰跟降雨事件走 + 全程分钟级基线——
   典型真实快响应盆地特征。

且 CVODE 聚合统计（netf=0、ncfn=0.07%）确认误差控制器并未 struggle——它
"心满意足地"走分钟步，因为**物理确实要求**分钟步，而非被振荡逼停。

D2 proxy 保守性（design R3）进一步加固结论：state-delta flip 在 SolverStep
分辨率**低估** sub-interval 振荡，偏差恒**against** OSC_CONFIRMED。即真实
振荡只会比观测更多才可能翻案——而观测的 concentration 已远低于门（0.0366 vs
0.50），保守偏差方向使 REAL_DYNAMICS 裁决更稳健。

## §6.2 keliya / qinyijiang 分歧分析（aggregation rule）

两案例 MARKER 均为 REAL_DYNAMICS，但**路径不同**，值得记录（spec
"aggregation on divergent secondary" 场景要求分歧须讨论）：
- **qinyijiang**：`burst_60s` 高（0.9942）但 concentration 低（0.0366）→
  经 Gate 2 判定。真·分钟级 + 弥散翻转。
- **keliya**：`burst_60s` 极低（0.0019）→ 经 Gate 1 判定。keliya 的 101k 步
  堆在 [60,300) s bin（98.9%，"just above" 60 s 阈值），并非 sub-60 s burst。

两者对 REAL_DYNAMICS 的支持机制不同，但都不满足 OSC_CONFIRMED 的三门合取。
per aggregation rule，keliya 仅 corroborating，**不 override** qinyijiang；
此处两者恰好同向，epic 裁决稳固。keliya 的 histogram 形态提示：即便未来解开
其分钟级，`burst_share` 加速上界（`1/(1−0.0019)≈1.002×`）几乎为零，本就
不值得限速器投入——与 REAL_DYNAMICS 决定一致。

## §6.3 与已关闭线的对比

P11-osc 是 ADR-0008/0009/0010 之后**最后一个 ≥10× CPU 机会假设**的检验。
REAL_DYNAMICS 裁决意味着：qinyijiang 的 wall 是 step-count-driven，而 step
count 是 physics-driven 且**不可合法压缩**。结合三条已关闭线，确认 **P1e
StrictOMP RHS + SHUD_SPGMR_MAXL small-case opt-in 是 CPU 加速的生产终点**，
GPU/domain-decomposition 保持 design-gated（ADR-0010，不在本 epic 触碰）。

---

# §7 Limitations & Threats to Validity / 限制与威胁

## §7.1 R2 window-representativeness threat（design.md §Risks R2）

**本裁决仅适用于 90 天 benchmark 窗口。** 三案例的 per-case START 季节
**未受控**（nanlin day 366–456、keliya day 12053–12143、xinanjiang day 0–90
落在各自不同的季节相位）。full-length（>90 天）多季节窗口重跑 **deferred 到
post-verdict follow-up**（本 epic Out of Scope）。具体威胁：若某 case 的
benchmark 窗口恰好落在低-flux 季节，可能低估其振荡；但 qinyijiang 窗口
（day 366–456）已包含明显降雨事件簇（day 438–452），且分钟级步进是**全程
基线**而非事件局限，故窗口偏置不太可能翻转 REAL_DYNAMICS 裁决。full-length
验证列为 follow-up 以进一步降低此威胁。

## §7.2 D2 proxy under-count（design.md §Risks R3）

D2 flip 计数是 accepted-boundary state-delta proxy（SolverStep 分辨率），
**保守低估** sub-interval 振荡（偏差恒 against OSC_CONFIRMED）。见 §6.1：
此偏差方向使 REAL_DYNAMICS 更稳健，不构成翻案风险，但意味着本诊断**不能
用于反向证明"完全无振荡"**——它只能证明"振荡不足以支撑 OSC_CONFIRMED 门"。

## §7.3 kashigeer gap（deferred-upstream，non-blocking）

`kashigeer` 无法纳入本证据矩阵：其 B0 archive 缺 `cvode_stats.txt`——
**blocked deferred-upstream**：X76 forcing band 在**两个 endpoint**（本地 Mac
+ 服务器）均缺失（`ksge.tsd.forc` 声明 941 站但 `forcing/` 仅 403 个 CSV，
缺失带覆盖整个 X76.*/X77–X79.* 列），`./shud ksge` 在 CVODE 构造前即以
exit 12 (FILEIO) abort，per `benchmarks/kashigeer/B0_output/DEFERRED.txt`。
**仅在上游 forcing 修复后**方可 opportunistic refresh；本 gap 不阻塞 P11-osc
裁决（qinyijiang 单案例已充分定 epic verdict，per aggregation rule）。

## §7.4 平台外推

诊断判定明确以本地 Mac 为准（design.md §Execution topology）。§1.1.1 量化
目标的服务器验收不适用于本诊断阶段——本裁决是"是否值得优化"的 go/no-go，
不是性能验收。heihe/heihe_x4 的服务器 profile（healthy，nst≈6.57k）作为
B0 evidence 表参照，与本地 control（xinanjiang）一致，交叉支持诊断探针在
健康 case 上不误报。

---

# §8 Conclusion / 结论

**Epic 裁决 = REAL_DYNAMICS**（qinyijiang MARKER verbatim；control sanity
PASS 故有效）。qinyijiang / keliya 的分钟级 CVODE 步进是**诚实的真实快
水文动力学**，非可阻尼的局部数值振荡：分钟级 burst 真实（burst_60s=0.9942）
但翻转空间弥散（top-1% concentration=0.0366 « 0.50），无正向时间联系
（ρ=−0.3753），且翻转峰跟降雨事件走。任何限速器都将 falsify physics。

**per spike_brief kill-gate：optimization 线关闭，本 change 零限速器代码落地
（spec "closure path" + "confirmed path creates no implementation" 均满足——
本 PR 既不实现限速器，也不因非-OSC_CONFIRMED 裁决而产 limiter stub）。**

---

# §9 Follow-up / gate 决定（REAL_DYNAMICS 分支 — RECORD ONLY）

> 按 task 3.4 + osc-verdict-gate spec "closure path"：REAL_DYNAMICS →
> verdict doc 兼 epic closure note + SHUD `p11-osc` merge-back 决定显式记录。
> **本节仅记录决定，不执行**（capstone 由 orchestrator 在 task 3.6 执行；
> 本 PR 不开 GitHub issue、不建 openspec change、不做 SHUD 分支实际 merge）。

## §9.1 Epic closure note

P11-osc 诊断阶段以 **REAL_DYNAMICS 闭合**。三条 machine gate + forcing-tracking
佐证一致收敛。无 optimization 后续、无 limiter candidate、无新 ADR（closure
由本 verdict doc 承载；ADR 仅在需要正式 program-status 修订时另提）。Epic #433
于 gate decision 后按 closure 收尾（capstone PR merge `research/p11-osc` →
`main`，随后 manual `gh issue close` 关闭 stacked PR 引用的 issue，per CLAUDE.md
branch model）。

## §9.2 SHUD `p11-osc` 分支 merge-back recommendation（保留 instrumentation）

**建议：将 SHUD `p11-osc` 分支 merge 回 `openmp-baseline`，instrumentation 作
opt-in 诊断保留。** 论据（从证据出发）：

1. **default-off 且 bitwise-neutral**：PR-D1 四腿证据（README leg a/b/c/d）
   证明——env unset（leg b）、`SHUD_DIAG_DT=0 SHUD_DIAG_OSC=0`（leg c，strict
   `=1` 门的 load-bearing 证据）、以及 `=1` enabled（leg d，read-only）三种
   条件下 keliya B0 的 13 个 model-output 文件 SHA 全部 == baseline。生产
   默认路径**零风险**。
2. **零 RHS 侵入**：`f.cpp` / `MD_rhs_core.cpp` zero diff（PR-D1 source
   audit）；状态读经 `N_VGetArrayPointer(udata)` accepted 状态，绝不碰
   `uY*` scratch globals。
3. **正向价值**：该 instrumentation 是未来任何"步数异常"case 分诊的
   现成、低成本工具（O(NY) copy + compare per interval）。保留它使后续
   诊断无需重新植入。
4. **无维护负担**：两个 emitter 集中在 `MD_osc_diag.hpp`（new）+ `shud.cpp`
   3 个 call site（全在 `diag.any_on()` 后），表面积极小。

merge-back **实际操作不在本 PR** —— per branch/merge model（spike_brief
§Branch/merge model），`p11-osc` → `openmp-baseline` 仅在 spike verdict 落地
**之后**执行；本节记录 recommendation，供 capstone turn 执行。

## §9.3 未选分支（留痕）

- **OSC_CONFIRMED 分支**：未触发（concentration 0.0366 < 0.50）。不产
  limiter candidate stub、不开 limiter design issue。spike_brief §Optimization
  的限速器约束（no f()-history / mass conservation / A5 tightened NSE·KGE≥0.99、
  peak≤1 interval、runoff≤1%、WB non-degrading / ROI nst↓≥30% AND wall≥1.5×）
  在本裁决下**不激活**。
- **INCONCLUSIVE 分支**：未触发（Gate 2 concentration 已 FAIL，不满足
  INCONCLUSIVE 的 `concentration ≥ 0.50` 前置）。不启动 refinement iteration
  （如 top-K per-edge flux / adjacency clustering）。

---

# References / 参考文献

## 内部文档
- `docs/p11-osc/spike_brief.md` — kill-gate 单一来源 + case matrix + §Optimization
- `openspec/changes/p11-osc/design.md` — §Verdict gate / §Risks R2/R3/R4 / §Execution topology
- `openspec/changes/p11-osc/specs/osc-verdict-gate/spec.md` — 本 PR 验收契约
- `openspec/changes/p11-osc/specs/osc-diag-analyzer/spec.md` — 分析器决策函数契约
- `tools/osc_diag/README.md` — analyzer CLI + verdict 阈值
- `benchmarks/kashigeer/B0_output/DEFERRED.txt` — kashigeer forcing gap（§7.3）
- `docs/case_deployment_map.md` §2.1 — case 部署权威（heihe server-only）

## 证据路径（本裁决原始数据）
- `.review-evidence/p11-osc-diag/pr-d1/README.md` — bitwise gate 四腿 + self-check
- `.review-evidence/p11-osc-diag/pr-d2/README.md` — emitter↔parser contract smoke
- `.review-evidence/p11-osc-diag/pr-d3/qinyijiang/` — primary（epic verdict source）
- `.review-evidence/p11-osc-diag/pr-d3/keliya/` — corroborating
- `.review-evidence/p11-osc-diag/pr-d3/xinanjiang_upstream/` — control（sanity source）

## GitHub / ADR
- Epic #433（P11-osc）；PR-D1 #434；PR-D2 #435；本 PR-D3 #436
- ADR-0008（P8 closure）/ ADR-0009（P9 closure）/ ADR-0010（program status; P10 gate）

## 外部依赖
- SUNDIALS-CVODE 6.0.0（`CVodeGetNumSteps` / `GetLastStep` / `GetCurrentStep`）
- OpenMP（诊断构建为 serial Config A；Config C 冗余 per A3a bitwise 等价）
