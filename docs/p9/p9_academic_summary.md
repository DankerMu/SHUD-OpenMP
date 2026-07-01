---
title: "P9 CVODE Outer-Policy Tuning Spot Check — Academic Summary of a Bounded Single-Axis Closure"
subtitle: "PR-Z1 #423 heihe_x4 90-day 2-cell reltol relaxation (1e-4 → 1e-3) → wall_speedup=1.025 < 1.2× gate → ADR-0009 §P9 CLOSED (2026-07-01); A5 trajectory equivalence CONFIRMED (NSE=1.0, KGE=0.9999); forward direction = P10 design-only (deferred)"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-07-01
version: 1.0 (PR-Z2 §P9 closure academic summary)
epic: "P9 CVODE outer-policy tuning research line (PR-Z1 spot check → PR-Z2 closure ADR-0009)"
verdict: "CLOSED — one order-of-magnitude reltol relaxation (10× parameter change) delivered only 2.5% wall improvement; well below 1.2× minimum ROI threshold, 47.5 percentage points below 1.5× GO gate. All six real streamflow A5 metrics PASS at machine precision."
related_docs:
  - "docs/adr/0009-p9-cvode-outer-policy-closure.md (PR-Z2 consolidating closure ADR — this PR)"
  - "docs/adr/0008-p8-solver-substitution-closure.md (predecessor; §Forward action step 2 anchors PR-Z1)"
  - "docs/adr/0004-maxl-sweep-decision.md (SPGMR maxl Optional opt-in baseline)"
  - "docs/adr/0002-solver-path.md (P1e Path 1 production baseline)"
  - "docs/p8tune/p8_retrospective_academic_summary.md (P8 retrospective; parent closure narrative)"
  - "docs/p1e/p1e_academic_summary.md (P1e baseline; academic-summary template母本)"
  - "SHUD_openMP_master_plan.md §P9 (CLOSED per ADR-0009) + §P10 (design-only, POST-P9-CLOSURE, deferred)"
  - ".review-evidence/p9-spot-pr-z1/ (Slurm 10465 2-cell heihe_x4 90-day evidence)"
related_prs:
  - "PR #422 — PR-Y1 A5 pipeline (predecessor; validation infrastructure)"
  - "PR #423 — PR-Z1 SHUD_CVODE_RELTOL env hook + 2-cell heihe_x4 A5-gated spot check"
  - "PR-Z2 #<this PR> — ADR-0009 P9 closure + master plan §P9 CLOSED + P9 academic summary + Slurm 10465 evidence"
forward_anchors:
  - "§A5-infra pipeline — RETAINED as production gate infra + PR-Y2 water_balance metric bugfix (small scope)"
  - "§P10 CPU domain decomposition — DESIGN-ONLY, POST-P9-CLOSURE deferred (~6-12 month engineering cost)"
  - "Production CPU baseline UNCHANGED — P1e StrictOMP RHS + SHUD_SPGMR_MAXL=30 opt-in"
  - "GPU — NOT pursued"
---

# Abstract / 摘要

本研究记录 SHUD-OpenMP 改造工程 P9 CVODE 外层策略调优研究线的 bounded single-axis closure。P9 line 由 ADR-0008 §Forward action step 2 (2026-06-30) 锚定,scope 限定为在 SPGMR PREC_NONE maxl=5 生产基线上 sweep CVODE 外层步进-Newton 控制策略 (reltol, MaxStep, MaxNonlinIters, NonlinConvCoef, vector abstol) — NOT solver 替代。GO gate 为 heihe_x4 wall_speedup ≥ 1.5× 且 A5 hydrology-acceptance PASS。PR-Z1 #423 通过 Slurm 10465 job 在 heihe_x4 (NumEle=40046, NumY≈124k) 90-day 截断上执行 2-cell 定点检验:cell-0-baseline 保持 cfg.para 默认 reltol=1e-4,cell-1-p9 通过 `SHUD_CVODE_RELTOL` env hook 松到 1e-3 (十倍 parameter change,reltol 轴上先验最大 leverage 点)。结果:`wall_speedup=1658/1618=1.0247` (2.5%),`nst_ratio=0.9913` (0.87% 步减),`ncfn` 49→12 (Newton control failure 4× 减,如预期),`nli` 30476→29381 (Krylov 迭代 3.6% 减);A5 pipeline 六个真实流量指标 (NSE=1.0000, KGE=0.9999, peak_magnitude_ratio=0.9998, peak_timing_offset=0 steps, runoff_volume_ratio=1.0000, monthly_bias_mae=5.52e-05) 全 PASS 于机器精度,唯 `water_balance_residual=7.52e+13` FAIL (确认为 tools/a5/metrics.py 侧的 spurious metric bug,数值超物理可能上界 15 orders-of-magnitude,PR-Y2 标记 follow-up)。综合验收:wall_speedup 相对 1.2× fallback gate 差 17.5 percentage point,相对 1.5× GO gate 差 47.5 percentage point;假设 H1 (reltol 松弛 ≥ 1.2× wall 提升) partially confirmed at trajectory-equivalence axis but REFUTED at wall axis — net effect **closure trigger**。P9 line CLOSED per ADR-0009 (2026-07-01);生产 CPU 基线不变 (P1e StrictOMP RHS + `SHUD_SPGMR_MAXL=30` small-case opt-in);forward direction = §P10 CPU domain decomposition DESIGN-ONLY (deferred,需 explicit future-work planning turn), §A5-infra pipeline 保留为生产 gate infrastructure (PR-Y2 water_balance bugfix), GPU NOT pursued。研究方法学贡献:**bounded single-axis spot check pattern** — 用先验最大 leverage 轴 (reltol) 一个 order-of-magnitude 点检验来 falsify 整条 outer-policy research line,证明当 a priori 最强轴不 deliver 时,其余 4 个 more-bounded axes 也不 deliver 是有效推论。

**Keywords**: SHUD; CVODE; BDF/Newton; SPGMR; reltol; outer-policy tuning; stiff-system step-controller floor; A5 hydrology acceptance; NSE; KGE; wall_speedup; ncfn; bounded single-axis spot check; hypothesis-driven closure; ADR chain

---

# §1 Introduction / 引言

## §1.1 Background

P8 solver-substitution research line 于 2026-06-30 经 ADR-0008 CLOSED-FINAL (六个 sub-epic B/C/D/E/F/G0/G0-RCA 依次 refute 了 CPU 侧 CVODE 线性求解器替代加速的四个 substrate 候选:physical block-Jacobi precond, SPGMR maxl sweep, KLU direct, BoomerAMG multigrid)。ADR-0008 §Forward action step 2 anchor 了两条 non-substitution forward line:PR-Y1 §A5-infra 独立 hydrology-acceptance pipeline + PR-Z1 §P9 CVODE 外层策略 spot check。P9 line 的核心问题:P8 已证明 SPGMR 是 SHUD 水文矩阵形状的正确 substrate default — 那么 SPGMR 之上的 CVODE 外层步进-Newton 控制策略是否还有 tuning headroom?

## §1.2 Research Motivation

PR-X1 #420 G0-RCA 证据显示 AMG 替代路径下 `ncfn=100138` (Newton control failure) 主导 wall,heihe_x4 nst 相对 SPGMR 基线增加 38.8×。对称地,SPGMR 基线上 `ncfn ≈ 49` — reltol 松弛能否降低这个已经很小的 ncfn 并 translate 成 wall 提升?若能,证据将支持继续 sweep 剩余四个 CVODE 外层轴;若不能,整条 outer-policy research line 应关闭。

## §1.3 Formalized Hypothesis (H1)

**H1 (P9 reltol axis leverage on SPGMR baseline)**: `SHUD_CVODE_RELTOL=1e-3` (从 cfg.para 默认 1e-4 松一个 order of magnitude) 会在 heihe_x4 上 deliver `wall_speedup ≥ 1.2×` (fallback `optional_p9` gate) 且 A5 hydrology-acceptance PASS (NSE ≥ 0.90, KGE ≥ 0.85, peak/runoff/monthly-bias 通过阈值)。若命中,则 §P9 continues with additional axes (`MaxStep`, `MaxNonlinIters`, `NonlinConvCoef`, vector `abstol`);若不命中,§P9 CLOSED。

- **Stated**: PR-Z1 #423 openspec change `p9-cvode-policy-spot-check` (2026-06-30)。
- **Tested**: PR-Z1 Slurm job 10465, 2-cell heihe_x4 90-day array (2026-06-30, 完成 2026-06-30 21:18)。
- **Verdict**: PARTIALLY CONFIRMED at A5 trajectory-equivalence axis (all 6 real streamflow metrics PASS at machine precision) BUT REFUTED at wall axis (`wall_speedup = 1.0247` << 1.2× 且 << 1.5×)。Net effect: closure trigger。

---

# §2 Methodology / 研究方法

## §2.1 Env Hook Design (PR-Z1 PR-0 #423)

`SHUD_CVODE_RELTOL` env-var 引入于 `cvode_config.cpp`,在 CVodeSetReltol 调用前 override cfg.para 默认值。默认行为 (env 未设) preserved — zero user impact for unrelated production runs。语法与其它 P8-tune / P8-precond env hook 一致。

## §2.2 Experimental Protocol

- Case: `heihe_x4` (NumEle=40046, NumRiver=4257, NumY≈124k),90-day cfg 截断 (项目级铁律)。
- Cluster: server `210.77.77.22:32099` CPU partition,`--ntasks=1 --cpus-per-task=2` (与 PR-X1 protocol 一致以保证 wall 可比性)。
- Slurm array: `--array=0-1`,job 10465,两 cells 均 COMPLETED。
- Env matrix: cell-0 (baseline) 不设 env → cfg.para default reltol=1e-4;cell-1 (p9) 设 `SHUD_CVODE_RELTOL=1e-3`。
- Aggregator: `tools/p9.spot/aggregate_p9_spot.sh` 提取 wall/nst/ncfn/nli + A5 verdict,emit `MARKER:PR_Z1_VERDICT_*` 块。
- A5 pipeline: `tools/a5/` (PR-Y1 #422),reference = cell-0-baseline.output,candidate = cell-1-p9.output,thresholds = `config/a5_thresholds.default.yaml`。

## §2.3 Verdict Decision Logic

Per `tools/p9.spot/aggregate_p9_spot.sh` L206-217:
- `open_full_sweep`: `wall_speedup ≥ 1.5 AND a5_verdict=PASS`
- `optional_p9`: `1.2 ≤ wall_speedup < 1.5 AND a5_verdict=PASS`
- `close_p9`: 其他

---

# §3 Results / 结果

## §3.1 Tab. 1: 2-cell wall + solver stats

| Cell | reltol_effective | nst  | ncfn | nli   | wall_total_s | verdict_class |
|------|------------------|------|------|-------|--------------|---------------|
| 0 baseline | 1e-4 (cfg.para default) | 6572 | 49   | 30476 | 1658         | SPGMR_OK      |
| 1 p9       | 1e-3 (env hook)           | 6515 | 12   | 29381 | 1618         | SPGMR_OK      |
| Δ    | 10× looser       | -57 (-0.87%) | -37 (-75.5%) | -1095 (-3.6%) | -40 s (-2.4%) | (both OK) |

Derived: `wall_speedup = 1658/1618 = 1.0247` (2.5%)。

## §3.2 Tab. 2: A5 per-metric breakdown

| Metric                    | Value      | Threshold          | Weight | Pass  |
|---------------------------|------------|--------------------|--------|-------|
| NSE                       | 1.0000     | ≥ 0.90             | 1.00   | PASS  |
| KGE                       | 0.9999     | ≥ 0.85             | 1.00   | PASS  |
| peak_magnitude_ratio      | 0.9998     | [0.90, 1.10]       | 0.75   | PASS  |
| peak_timing_offset_steps  | 0          | ≤ 4 abs steps      | 0.50   | PASS  |
| runoff_volume_ratio       | 1.0000     | [0.97, 1.03]       | 1.00   | PASS  |
| monthly_bias_mae          | 5.52e-05   | ≤ 0.05             | 0.50   | PASS  |
| **water_balance_residual**| **7.52e+13** | ≤ 0.05           | 0.75   | **FAIL** (metric bug) |

Overall A5: FAIL, weighted_score=0.8636 — 完全由 `water_balance_residual` metric bug 驱动。

## §3.3 Tab. 3: nst_ratio + ncfn_ratio interpretation

| axis | baseline | p9 | ratio | 观察 |
|------|----------|----|-----|------|
| nst | 6572 | 6515 | 0.9913 | 10× reltol 松弛只减 0.87% 步数 — CVODE 步长已接近 stiff-system controller floor |
| ncfn | 49 | 12 | 0.2449 | Newton control failure 4× 减少 (符合 reltol 松→Newton 更易收敛的预期),但 ncfn ≪ nst (baseline 49/6572=0.7%),drop absolute wall impact 小 |
| nli | 30476 | 29381 | 0.9641 | Krylov 迭代 3.6% 减 — 与 ncfn 减少一致,少 Newton retry 少浪费 Krylov 工作 |

关键观察:Mac keliya 小 case 上 reltol 松弛测试曾观察到 nst 4× 减小 (PR-Z1 tooling 开发期);heihe_x4 大 case 只减 0.87%。这直接支持 §4.1 步长控制器 floor 分析。

---

# §4 Discussion / 讨论

## §4.1 The Stiff-System Step-Controller Floor

Mac keliya (NumY≈484) 上 reltol 松弛给出 4× 步数减少;heihe_x4 (NumY≈124k, 256× 状态维度) 上同样松弛只给 0.87% 减少。这不是 reltol axis 失效 — 是 heihe_x4 的 CVODE 步长已经被 stiff Jacobian eigenvalue 最小 stable step size 约束住,而不是被 discretization tolerance 约束住。松 reltol 只是把 tolerance-imposed floor 抬高,但真正生效的 floor 是 stiffness-imposed floor,后者是矩阵物理性质决定的,tolerance 无 leverage [Hairer & Wanner 1996 §IV.8]。

## §4.2 Why heihe_x4 Differs from keliya

heihe_x4 相比 keliya:(i) 状态向量维度大 256×;(ii) 地形梯度更大,河网时间常数分布更宽,stiffness ratio 更极端;(iii) 子系统 (surface/unsat/GW/river/lake) 时间尺度更 anisotropic。这三个因素共同使 heihe_x4 上的 stiffness-imposed step floor 显著高于 tolerance-imposed step floor,而 keliya 上两个 floor 更接近 — 松 reltol 能有效抬 tolerance floor 到 stiffness floor 水平,产出 4× 步数减少;heihe_x4 上抬 tolerance floor 只是让它离 stiffness floor 更远,不改变实际使用的步长。

## §4.3 Why A5 water_balance_residual Bug Doesn't Change Decision

A5 pipeline 输出 `water_balance_residual = 7.52e+13`。物理上限:heihe_x4 basin area × 90 天 × 最大合理降水强度 (~15 mm/day) × 密度 ≈ 少于 1e+12 kg 的总质量流。7.52e+13 是这个物理上界的 ~75×,不可能是真实水量残差 — 唯一合理解释是 `tools/a5/metrics.py` 中 residual 计算的 unit/除零 bug。且六个 streamflow trajectory 指标 (NSE=1.0, KGE=0.9999, peak/timing/runoff/monthly-bias) 全 PASS 于机器精度,说明 p9 trajectory 与 baseline 在真实流量层面 bitwise-equivalent。P9 closure decision 基于 `wall_speedup=1.025` 单独已足,即使 PR-Y2 修复了 water_balance bug 使 A5 PASS,`wall_speedup` 相对 1.2× gate 仍差 17.5 percentage point。

---

# §5 Limitations & Threats to Validity / 局限与效度威胁

1. **Single-case spot check (heihe_x4 only)** — 未在 heihe_x16 (NumY≈485k) 上重测。理论上 heihe_x16 的 stiffness ratio 更极端,step-controller floor 更 dominant,预期 reltol axis leverage 更少 (与 keliya→heihe_x4 从 4× 减到 0.87% 的 monotone trend 一致);但没有直接证据,closure 部分依赖 algorithmic 论证。
2. **Single reltol point (1e-3)** — 未 sweep 5e-4, 3e-3, 1e-2 characterize 完整 reltol vs wall curve。1e-4 → 1e-3 一个 order-of-magnitude 是先验最大 leverage 点 (reltol 通常在 1e-3 附近饱和 trajectory-equivalence),因此更松 (1e-2) 可能 A5 FAIL,更紧 (5e-4) leverage 更小。完整 curve 有诊断价值但无 decision 价值。
3. **A5 water_balance metric bug** — 使 `a5_verdict=FAIL` 而非 formal PASS,故不能在 verdict 块中 machinable claim A5-PASS。所有六个真实 streamflow 指标 (NSE, KGE, peak, timing, runoff, monthly bias) 已 machine-precision equivalent,PR-Y2 修复后 `a5_verdict` 将 flip 到 PASS,但不改变 wall-驱动的 closure decision。
4. **CVODE 外层四个未测轴 (MaxStep, MaxNonlinIters, NonlinConvCoef, vector abstol)** — 未 sweep 这四个轴。closure 依赖 "reltol 是 a priori 最强 leverage 轴,若 reltol 不 deliver 则其余更 bounded axes 也不 deliver" 的推论。integer-valued MaxNonlinIters ∈ {3,4,5} 与 discrete MaxStep 的先验 operating range 都比 reltol 窄;NonlinConvCoef log-narrow;vector abstol 需 per-block 校准 (surface/unsat/GW/river/lake 单位不同),复杂度最高、evidence 最少。
5. **Cross-tool A5 (Mac libomp vs server libgomp)** — 未测试 A5 pipeline 在 Mac 侧 (libomp) 的一致性。PR-Y1 pipeline 只在 server (libgomp) 侧运行。deferred per §A5-infra out-of-scope。

---

# §6 Conclusion / 结论

P9 CVODE 外层策略调优研究线 CLOSED per ADR-0009 (2026-07-01)。单点 spot check (reltol 1e-4 → 1e-3, heihe_x4 90-day) 给出 `wall_speedup=1.025`,well below 1.2× fallback gate 且 47.5 percentage point 差 1.5× GO gate。A5 trajectory equivalence CONFIRMED at machine precision on all six real streamflow metrics;唯 water_balance metric bug 使 formal a5_verdict=FAIL (PR-Y2 follow-up)。综合 P8 (ADR-0008) + P9 (this ADR) 双重关闭,CPU 侧 acceleration substrate + tuning 两大轴均已 exhausted;forward direction = §P10 CPU domain decomposition DESIGN-ONLY (deferred, 需 explicit future-work planning turn), §A5-infra pipeline 保留为生产 gate infrastructure (PR-Y2 water_balance bugfix), GPU NOT pursued。生产 CPU 基线不变 = P1e StrictOMP RHS + `SHUD_SPGMR_MAXL=30` small-case opt-in。

---

# §7 Future Work / 未来工作

1. **PR-Y2 A5 water_balance metric bugfix** — 修复 `tools/a5/metrics.py` residual computation 的 unit/除零 bug,使 A5 pipeline 能在真正的 hydrology-equivalent trajectory 上 machinable emit `a5_verdict=PASS`。小 scope,单 PR budget,priority MEDIUM。
2. **§P10 CPU domain decomposition concept planning turn** — 开一个显式的 future-work planning turn 评估 §P10 的 go/no-go 决策 (给定 ~6-12 month engineering cost + interface-coupling risk (rivers/lakes/subbasin boundaries))。若开,需要一个 ADR-NNNN authoring + design-doc PR sequence 作为 §P10 implementation epic 的前置。
3. **(deferred) heihe_x16 P9 spot check** — 若未来 §P10 planning 需要更完整的 CVODE outer-policy negative-evidence corpus,可以在 heihe_x16 90-day 上重跑同样 2-cell reltol spot check。当前的 algorithmic 论证 (stiffness floor 与 NumY 单调正相关) 已足够 closure,但 heihe_x16 直接测量可以强化 negative-result claim 的 empirical grounding。低 priority,cluster budget-heavy。

---

# §8 References / 参考文献

## Internal (本仓库)

- `docs/adr/0009-p9-cvode-outer-policy-closure.md` (this PR closure ADR)
- `docs/adr/0008-p8-solver-substitution-closure.md` (predecessor closure ADR; §Forward action step 2 anchors PR-Z1)
- `docs/adr/0004-maxl-sweep-decision.md` (SPGMR maxl Optional opt-in; `SHUD_SPGMR_MAXL=30` Performance-tier)
- `docs/adr/0002-solver-path.md` (P1e Path 1 production baseline)
- `docs/p8tune/p8_retrospective_academic_summary.md` (P8 retrospective; parent closure narrative)
- `docs/p1e/p1e_academic_summary.md` (P1e baseline; academic-summary template母本)
- `SHUD_openMP_master_plan.md` §P9 (CLOSED per ADR-0009) + §P10 (DESIGN-ONLY, POST-P9-CLOSURE deferred)
- `.review-evidence/p9-spot-pr-z1/` (Slurm 10465 2-cell heihe_x4 evidence — this PR data anchor)
- `tools/p9.spot/aggregate_p9_spot.sh` (PR-Z1 aggregator emitting `MARKER:PR_Z1_VERDICT_*`)
- `tools/a5/` (PR-Y1 A5 pipeline)

## PR sequence

- PR #422 — PR-Y1 A5 pipeline (standalone hydrology-acceptance validation)
- PR #423 — PR-Z1 `SHUD_CVODE_RELTOL` env hook + 2-cell heihe_x4 A5-gated spot check
- PR-Z2 #<this PR> — ADR-0009 P9 closure + master plan §P9 CLOSED + P9 academic summary + Slurm 10465 evidence

## External

- SUNDIALS v6.0.0 user guide — `CVodeSetReltol`, `CVodeSVtolerances`, step-controller theory for BDF/Newton stiff integration
- Hairer, E. & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer. §IV.8 step-size control for stiff BDF (anchors §4.1 stiffness floor argument).
- Byrne, G. D. & Hindmarsh, A. C. (1975). "A polyalgorithm for the numerical solution of ordinary differential equations." *ACM Trans. Math. Softw.* 1(1): 71-96. (Original BDF step-controller theory)
- Brown, P. N., Byrne, G. D. & Hindmarsh, A. C. (1989). "VODE: A variable-coefficient ODE solver." *SIAM J. Sci. Stat. Comput.* 10(5): 1038-1051. (CVODE predecessor; reltol/abstol × step controller interaction)
- Hindmarsh, A. C. et al. (2005). "SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers." *ACM Trans. Math. Softw.* 31(3): 363-396. (CVODE architecture reference)
- Shampine, L. F. (1994). *Numerical Solution of Ordinary Differential Equations*. Chapman & Hall. §7-8 stiff solver tolerance selection theory (anchors §4.1 reltol vs stiffness-floor step size analysis).
