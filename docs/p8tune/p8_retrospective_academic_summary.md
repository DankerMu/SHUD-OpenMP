---
title: "P8 Solver-Substitution Research Line — Full Retrospective: A Six-Epic Chain of Negative Results on CPU-Side CVODE Linear-Solver Substitution for SHUD Hydrology BDF/Newton Integration"
subtitle: "学术风格 retrospective — B → C → D → E → F → G0 → G0-RCA(PR-X1) → PR-X2 closure ADR-0008; 4 formalized hypotheses (H1-H4) stated-tested-refuted 全链;forward direction A5 infra + CVODE policy tuning + decomposition-design-only + GPU-not-pursued"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-30
version: 1.0 (PR-X2 retrospective closure ADR-0008 academic summary)
epic: "P8 solver-substitution research line (P8-tune.B/C/D/E/F/G0/G0-RCA chain, closing via PR-X2 ADR-0008)"
verdict: "CLOSED-FINAL — no CPU linear-solver substitution beats SPGMR PREC_NONE maxl=5 at production-target scale (heihe_x4 NumY ≈ 124k, heihe_x16 NumY ≈ 485k)"
related_docs:
  - "docs/adr/0008-p8-solver-substitution-closure.md (PR-X2 consolidating closure ADR — this PR)"
  - "docs/adr/0007-amg-spike-decision.md (P8-tune.F predecessor; Amendment 2026-06-30 — G0-RCA outcome appended in PR-X2)"
  - "docs/adr/0005-klu-spike-decision.md (P8-tune.D KLU spike Case-aware verdict)"
  - "docs/adr/0004-maxl-sweep-decision.md (P8-tune.C SPGMR maxl Optional-knob; SHUD_SPGMR_MAXL=30 Performance-tier)"
  - "docs/adr/0003-precond-spike-decision.md (P8-tune.B PREC_NONE NO-GO baseline)"
  - "docs/adr/0002-solver-path.md (P1e Path 1 — Serial NVec + StrictOMP RHS; production baseline)"
  - "docs/p8tune/amg_g0_verdict.md (G0 verdict source-of-truth)"
  - "docs/p8tune/amg_spike_verdict.md (P8-tune.F verdict source-of-truth)"
  - "docs/p8tune/p8tune_g0_academic_summary.md (G0 academic summary; template母本)"
  - "docs/p8tune/p8tune_f_academic_summary.md (P8-tune.F academic summary)"
  - "docs/p8tune/p8tune_d_academic_summary.md (P8-tune.D academic summary)"
  - "docs/p1e/p1e_academic_summary.md (P1e baseline / academic-summary template母本)"
  - "SHUD_openMP_master_plan.md §P8-tune.{B,C,D,E,F,G0,G1,G2,H} (all CLOSED-FINAL) + §A5-infra + §P9 + §P10 (new anchors)"
  - "openspec/changes/p8tune-{spgmr-maxl,klu-spike,amg-spike,g0-instrumented-amg-smoke}/ (four archived epic-specs)"
  - ".review-evidence/g0-amg-rca-pr-x1/ (PR-X1 8-cell tolerance × EpsLin sanity matrix)"
  - ".review-evidence/g0-amg-smoke-array-rerun/ (G0 4-cell integrated smoke)"
  - ".review-evidence/p8tune-amg-pr-b/ + p8tune-amg-pr-c/ (F 16-cell pattern-only + verdict)"
  - ".review-evidence/p8tune-klu-spike-pr-a/ + p8tune-klu-spike-pr-b/ (D KLU pattern-only)"
  - ".review-evidence/p8tune-spgmr-maxl-prd-60cell/ (C SPGMR maxl PRD baseline)"
related_prs:
  - "PR #313/#315/#316 — P1e PR-F/G/H (production baseline establishment)"
  - "PR #369-#373 + #368 + #376 — P8-tune.C 6-PR SPGMR maxl sweep"
  - "PR #384/#385/#387/#388 — P8-tune.D 4-PR KLU pattern-only spike"
  - "PR #394/#402/#403/#404 + #412 — P8-tune.F 5-PR BoomerAMG pattern-only spike + capstone-merge"
  - "PR #414/#415/#416 + #417 (PR-C amendment) + #418 (PR-D) + #413 (capstone-merge) + #419 (post-merge cleanup) — P8-tune.G0 4-PR integrated smoke"
  - "PR #420 — PR-X1 G0-RCA tolerance × EpsLin matrix (the closure trigger)"
  - "PR-X2 #<this PR> — ADR-0008 retrospective closure + master plan §P8 CLOSED-FINAL + 5-step forward plan"
forward_anchors:
  - "§A5-infra — Hydrology-Acceptance Validation Pipeline (PR-Y1, HIGH priority, ~2-3 PR)"
  - "§P9 — CVODE Outer-Policy Tuning spot check (PR-Z1, MEDIUM priority, ~1-2 PR)"
  - "§P10 — CPU Domain Decomposition (DESIGN-ONLY; NO implementation commitment)"
  - "§P8-tune.H — GPU sparse fallback (CLOSED-FINAL; NOT pursued)"
  - "Production CPU baseline preserved — P1e StrictOMP RHS + SHUD_SPGMR_MAXL=30 opt-in"
---

# Abstract / 摘要

本研究系统回顾 SHUD-OpenMP 改造工程 P1e capstone (2026-06-25) 之后启动的 P8 solver-substitution research line — 一条为期约 5 天、跨 4 个 architectural-substrate 候选、6 个 sub-epic、总共 ≥25 个 PR 的 CPU 侧 CVODE 线性求解器替代加速研究链。研究背景:P1e Path 1 (`ExecPolicy::StrictOMP` + Serial NVec + PREC_NONE SPGMR maxl=5) 建立了 1.7× heihe_x4 / 1.066× heihe wall improvement 的生产基线,但 heihe_x16 (NumY ≈ 485k) 与 heihe_x4 (NumY ≈ 124k) 生产目标网格仍存在加速余量;Amdahl 反推显示 heihe_x4 serial fraction ≈ 63%,进一步 wall improvement 必须走 non-RHS 路径。P8 line 由 ADR-0002 决策树引导,依次评估四种 linear-solver substrate:(i) 物理块对角预条件器 + SPGMR (P8-tune.B) — ADR-0003 PREC_NONE NO-GO;(ii) SPGMR maxl sweep (P8-tune.C, 60-cell PRD) — ADR-0004 Optional-knob (`SHUD_SPGMR_MAXL=30` Performance opt-in);(iii) KLU direct solve pattern-only spike (P8-tune.D, 16-cell) — ADR-0005 Case-aware (小 case pattern-feasible / heihe_x4 wall margin 1.87× / heihe_x16 17.9× NO-GO);(iv) BoomerAMG/Hypre 通过 pattern-only spike (P8-tune.F, 16-cell) + 集成 CVODE smoke (P8-tune.G0, 4-cell) + tolerance-EpsLin RCA (PR-X1, 8-cell) 三阶段递进 — ADR-0007 strict NO-GO-both / amended GO + ADR-0007 Amendment 2026-06-30 G0 verdict NO-GO + ADR-0007 Amendment 2026-06-30 G0-RCA H4-REFUTED。

研究形式化 4 个跨阶段研究假设 (H1 - H4) 作为 P8 line 的 falsifiable criteria:H1 (P8-tune.B - E, pattern-only 结构 AMG audits 识别 operator/cycle complexity < 2.0 → wall improvement);H2 (P8-tune.F 16-cell 的 (interp=6, coarsen=8) winner combo 会 translate 到集成 CVODE);H3 (P8-tune.G0 集成 SUNLinSol_Hypre 会 deliver ≥10% heihe_x4 / heihe_x16 wall improvement vs SPGMR baseline);H4 (PR-X1: G0 `ncfn=100138` 是 Hypre solve tol + CVODE EpsLin mismatch 造成,tightening 双端 tolerance 可 rescue AMG path)。四假设的 stated → tested → refuted 全链依次记录于 2026-06-27 至 2026-06-30 五天。

关键数值结果:**(i)** P8-tune.C 60-cell SPGMR maxl 3-rep median 建立 heihe_x4 N=1 per-step baseline `0.226579 s / 6575 nst`,heihe_x4 全 maxl ≥ 10 regress 6.86-24.82%,小 heihe N=1 maxl=30 improve 12% (`SHUD_SPGMR_MAXL=30` Performance-tier ship);**(ii)** P8-tune.D KLU 16-cell (`klu_analyze_factor`) 显示 heihe_x4 numeric_factor wall 1.87× 超 0.7×SPGMR budget、heihe_x16 17.9× 超 (structurally infeasible);**(iii)** P8-tune.F 16-cell pattern-only (BoomerAMG) 全 16/16 hierarchy build + V-cycle apply PASS,heihe_x16 best combo (interp=6, coarsen=8) 单 V-cycle setup+apply 0.116 s vs KLU 2.84 s per-step (24.5× 优),但 Axis 4 `cycle_complexity = 2 × operator_complexity` 是 hardcoded 估值,strict-vs-amended verdict 二分;**(iv)** P8-tune.G0 4-cell 集成 CVODE smoke 显示 heihe_x4 AMG per-step `0.092873 s` 比 SPGMR `0.238369 s` 快 0.39× (per-step G0-5 PASS),但 total wall `23660 s` vs SPGMR `1566.55 s` **15.1× WORSE**,因 CVODE nst 258× (254756 vs 6572, 38.8× step 增) 主导,heihe_x16 8h Slurm 预算 SIGKILL MALFORMED — g0_verdict_branch=NO-GO-G0;**(v)** PR-X1 8-cell heihe_x4 90-day RCA 在 `AMG_TOL ∈ {1e-7, 1e-9, 1e-11, 1e-13}` × `EpsLin ∈ {0.05, 0.005}` 8-cell 矩阵中,`ncfn` 稳定在 98,286-104,795 (6% 窗) — 4 orders-of-mag × 10× 双端 tolerance 变化对 outer Newton control failure rate 零 leverage,H4 REFUTED。综合验收 P8 solver-substitution research line CLOSED-FINAL per ADR-0008;production CPU baseline retained = P1e StrictOMP RHS + `SHUD_SPGMR_MAXL=30` small-case opt-in;forward direction 为 §A5-infra (PR-Y1) + §P9 CVODE outer policy tuning (PR-Z1) + §P10 decomposition design-only + GPU 明确 NOT pursued。

研究方法学贡献:**六 sub-epic 递进 pattern (pattern-only spike → integrated smoke → RCA sanity matrix) + hypothesis-driven closure ledger (H1-H4 stated-tested-refuted 全链) + append-only ADR amendment pattern (§Status + §Decision byte-identical invariant + Amendment 分层堆叠) + case-asymmetric wall scaling 表征 (小 case OMP setup floor vs 大 case algorithmic crossover) + integrated-CVODE-level 反驳强于 pattern-only 二分 verdict**。这一 methodology 为未来 architectural-substrate 研究提供 pattern-to-integration-to-closure 的完整工程模板,同时也建立了一个可发表规模的 CPU 稀疏求解替代 negative-result corpus。

**Keywords**: SHUD; CVODE; BDF/Newton; SPGMR; PREC_NONE; SUNLinSol_Hypre; BoomerAMG; Hypre; SuiteSparse KLU; solver substitution; hydrology; ncfn; Newton control failure; step controller; case-asymmetric scaling; hypothesis-driven closure; ADR chain; retrospective; negative result; P1e StrictOMP RHS; SHUD_LINSOL; SHUD_SPGMR_MAXL; SHUD_AMG_TOL; SHUD_CVODE_EPSLIN

---

# §1 Introduction / 引言

## §1.1 Background

SHUD-OpenMP 改造工程自 B0 baseline (SHUD `3aec657`) 起在 SUNDIALS-CVODE 6.0.0 隐式 BDF/Newton 框架下经历五个大 epic 阶段:B1a (S0-S4) → B1b (S5-S6) → P1c (10-anchor / 8-site deterministic reduction) → P1d (NUMA first-touch 假设推翻) → P1e (`ExecPolicy::StrictOMP` 生产基线)。P1e capstone (2026-06-25) 通过 ADR-0002 Path 1 建立了 (i) heihe_x4 (NumY ≈ 124k) 1.7× wall improvement, (ii) heihe (NumY ≈ 21k) 1.066× wall improvement, (iii) 6 case × 4 N × 3 rep 跨平台 SHA 矩阵 bitwise 稳定的生产基线。ADR-0002 Path 1 selection rationale:reproducibility 直接闭环 + speedup 路径清晰 + cost/risk/reversibility 三项最优 [1]。

P1e 关闭后即刻面临下一阶段问题:heihe_x4 加速比 1.7× 已 approaches Amdahl serial-fraction 上界 (serial fraction ≈ 63% per Amdahl 反推),heihe_x16 (NumY ≈ 485k) 生产目标网格未验证。进一步 wall improvement 必须走 non-RHS 路径,而 SHUD CVODE profile 显示 linear-solver Setup + Solve 是 RHS 之外的第二大 wall consumer (per ADR-0003 §Context 60% RHS + 25% linear-solver + 15% other 三方拆分)。P8 solver-substitution research line 因此成为 P1e 之后自然的 forward line — 目标:验证是否存在一个 CPU 侧 CVODE 线性求解器替代 substrate 能在 heihe_x4 + heihe_x16 上 deliver ≥ 10% wall improvement,并通过 hydrology-acceptance (A5) 验证。

## §1.2 Research Motivation

P8 research line 的核心问题:在 SHUD 水文 ODE 系统的隐式 BDF/Newton 框架下,已知 SPGMR PREC_NONE maxl=5 是 ADR-0003/0004 carve-out chain 锁定的 default,那么是否存在一个更强的替代 substrate — 具体来说 (i) 更强的预条件器 (P8-tune.B 物理块对角 precond),(ii) 更长的 Krylov 子空间 (P8-tune.C maxl sweep),(iii) 直接稀疏求解 (P8-tune.D KLU),(iv) 代数多重网格 (P8-tune.F/G0/PR-X1 BoomerAMG/Hypre) — 能在 production-target 网格规模 (heihe_x4 NumY 124k, heihe_x16 NumY 485k) 上提供有意义的 wall reduction?

## §1.3 Formalized Hypotheses (H1-H4)

跨 6 个 sub-epic 递进,4 个 falsifiable 研究假设作为 P8 line 的 verdict gate:

- **H1 (P8-tune.B - E: pattern-only structural AMG/KLU audits)**: pattern-only 结构 audits (dump adjacency + FD-color Jacobian + KLU analyze_factor / BoomerAMG setup_solve) 能识别出 `operator_complexity < 2.0` (AMG 侧) / `fill_ratio` 合理 (KLU 侧) 的 hierarchy/factorization → integrated wall improvement 应可预期。**Stated**: P8-tune.C proposal 2026-06-27 (ADR-0004 §Context)。**Tested**: P8-tune.C 60-cell + P8-tune.D 16-cell + P8-tune.F 16-cell (依次)。**Verdict**: PARTIAL — pattern-only assertions PASS (KLU fill_ratio 1.18-1.92, BoomerAMG operator_complexity 1.00-1.01) 但 wall axis 在 heihe_x4/x16 均超预算 (KLU 1.87× / 17.9×; BoomerAMG 有 pattern-only setup+apply 数字但未 integrated verified)。H1 partial-refuted at wall axis。
- **H2 (P8-tune.F integrated-translation)**: P8-tune.F 16-cell pattern-only sweep 识别的 large-case best combo `(interp_type=6, coarsen_type=8)` 会 translate 到集成 CVODE 阶段仍是最优。**Stated**: P8-tune.F PR-D 2026-06-29 capstone verdict 中 hardcoded 到 P8-tune.G0 wrapper。**Tested**: P8-tune.G0 4-cell 集成 CVODE smoke。**Verdict**: 结构层 CONFIRMED (keliya `operator_complexity=1.002945` 集成端与 P8-tune.F 16-cell sweep 1.0029 0.0% drift, hierarchy 结构 well-translated), 但集成 wall axis 反而 REFUTED (总 wall 15.1× regress despite per-step 0.39× 优)。
- **H3 (P8-tune.G0 integrated wall improvement)**: 集成 SUNLinSol_Hypre + BoomerAMG hierarchy 会在至少一个 {heihe_x4, heihe_x16} 上 deliver ≥ 10% wall improvement vs SPGMR baseline。**Stated**: ADR-0007 §Forward action Amendment 2026-06-29 三段 gate framing G0/G1/G2。**Tested**: P8-tune.G0 PR-A 4-cell Slurm array (Slurm 10014)。**Verdict**: REFUTED at heihe_x4 (per-step 0.39× 优但 total wall 15.1× regress, ncfn=100138 主导 nst 38.8× 增),heihe_x16 MALFORMED 无法直接测量但 heihe_x4 结果已充分。
- **H4 (PR-X1 tolerance-EpsLin rescue)**: G0 `ncfn=100138` 是 Hypre solve tolerance `SHUD_AMG_TOL` (~1e-7 pre-X1 default) 与 CVODE `EpsLin` (0.05 SUNDIALS default) mismatch 造成 — Hypre "early-outs" 出 bad Newton correction → Newton 反复 fail → step retry → nst 暴增。Tightening 双端 tolerance 可 rescue AMG path。**Stated**: PR-X1 #420 openspec change 2026-06-30。**Tested**: PR-X1 8-cell heihe_x4 90-day Slurm array (Slurm 10248) — `AMG_TOL ∈ {1e-7, 1e-9, 1e-11, 1e-13}` × `EpsLin ∈ {0.05, 0.005}`。**Verdict**: REFUTED — `ncfn ∈ [98286, 104795]` 4 orders-of-mag × 10× sweep 内保持 6% 窗内,`ncfl=0` 全 cell (Hypre 内核每次收敛),tolerance mismatch 不是 ncfn 根因。

H4 REFUTED 触发 PR-X2 (本 retrospective) 的整体 solver-substitution research line closure。

## §1.4 Contribution Scope

本 retrospective 综合前 5 个 sub-epic ADR + 3 个 academic summary + PR-X1 evidence, 形式化 P8 line 的整体 negative result + closure rationale + forward direction。§2 综述 SHUD-OpenMP 各 epic carve-out chain 上下文接续;§3 综述 6 个 sub-epic 各自 methodology;§4 综述实验设置 (硬件 + 软件栈 + benchmark cases + Slurm 三铁律);§5 汇报 6 个 sub-epic 关键数值 + PR-X1 outcome;§6 综合 H1-H4 假设验证 + 与前序 P1e Path 1 baseline 对比 + ncfn 主导 outer-Newton failure mode 的算法层含义 + integrated-vs-pattern-only verdict 分歧;§7 明确 limitations + threats to validity (3/8 cells timeout on cn23 memory-bandwidth contention + heihe_x16 未直接测 + F-winner combo 之外 coarsen/interp 未覆盖);§8 P8 solver-substitution line CLOSED-FINAL 结论;§9 未来工作 §A5-infra + §P9 CVODE policy + §P10 decomposition design-only + GPU-not-pursued;§10 参考。

---

# §2 Related Work / 相关工作

## §2.1 SHUD-OpenMP carve-out chain (P1e 之前)

- **B0 baseline** (SHUD `3aec657`): pinned 上游 fresh clone,SUNDIALS-CVODE 6.0.0,SPGMR PREC_NONE maxl=5 native default,serial RHS + serial NVec (`shud` binary; `shud_omp` 加 `NVECTOR_OPENMP` 但 RHS 未并行)。
- **B1a stage** (S0-S4, PR-1 至 PR-12 #156, 2026-06-21 capstone): OpenMP wire-up + kernel level guardrails + bitwise B0 baseline lock (`B1a-tag = f7f992c / SHUD 0b3998d`)。
- **B1b stage** (S5-S6): 结构改造 + S6b bug fix。
- **P1c epic** (10-anchor / 8-site deterministic reduction): 8-site canonical fold order + Kahan (Neumaier) injection。**结论**: cross-N bit-level `A3a pattern (N=1≡N=2 ≠ N=4 ≠ N=8)` 保留,drift origin 不在 8 站点内部。Carve-out 推 P1d NUMA 假设。
- **P1d epic** (NUMA first-touch): 5 项 fact-check 全部推翻 P1d 假设 — RHS 始终 Serial path,writer first-touch hypothesis 根本不适用。E' containment closure (production default `OMP_NUM_THREADS=1` serial fallback)。Carve-out "真正应并行的 RHS 还没并行" 推 P1e。
- **P1e epic** (`ExecPolicy::StrictOMP` + Serial NVec + PREC_NONE SPGMR, ADR-0002 Path 1): production baseline established。heihe_x4 1.729× (H3 PASS) / heihe 1.066× (H3 FAIL for small case);§4.6.2 partial-closure SHIP。See `docs/p1e/p1e_academic_summary.md`。

P8 solver-substitution research line 承接 P1e capstone 直接下游 forward line。

## §2.2 P8-tune sub-epic chain

- **P8-tune.B — 物理块对角预条件器 (P8-precond, closed by ADR-0003 PREC_NONE NO-GO)**: 5-block precond (surface/unsat/GW/river/lake) 通过 CVODE `PSetup`/`PSolve` callback wire-up。**Trigger**: ADR-0002 §Discussion "block-Jacobi precond as future option"。**Outcome**: `nfeLS / nfe` floor 与 PREC_NONE baseline 持平,precond setup overhead 抵消 Krylov iter 减少的收益。ADR-0003 PREC_NONE NO-GO,`SUNLinSol_SPGMR(udata, PREC_NONE, 0, sunctx)` retained as production default。
- **P8-tune.C — SPGMR maxl sweep (closed by ADR-0004 Optional-knob)**: 6-PR sequence + 60-cell PRD (Performance-Regression-Detection) baseline。`SHUD_SPGMR_MAXL ∈ {5, 10, 20, 30, 50}` × 4 case × 3 rep。**Outcome**: heihe_x4 全 maxl ≥ 10 regress 6.86-24.82% (Krylov 子空间超过 L2 cache);小 heihe N=1 maxl=30 improve 12% (`SHUD_SPGMR_MAXL=30` Performance opt-in tier ship, NOT A5-certified)。ADR-0004 Optional-knob。
- **P8-tune.D — KLU pattern-only spike (closed by ADR-0005 Case-aware)**: 4-PR sequence + 16-cell Slurm array (4 case × 4 ordering)。KLU analyze + KLU factor wall + RSS via `/usr/bin/time -v`。**Outcome**: keliya + heihe 3-axis PASS (fill_ratio + RSS + wall);heihe_x4 Optional (wall margin 1.87×);heihe_x16 NO-GO (wall margin 17.9×)。ADR-0005 Case-aware split。See `docs/p8tune/p8tune_d_academic_summary.md`。
- **P8-tune.E.small-only — KLU env-var mini-prototype for small cases**: [OPEN, OPTIONAL/medium];**NOT closed** by this retrospective — 只覆盖小 case scope,不影响 P8 solver-substitution 主线 closure 决策。See ADR-0005 §Forward action。
- **P8-tune.F — BoomerAMG/Hypre pattern-only spike (closed by ADR-0007 strict NO-GO-both / amended GO)**: 5-PR sequence + 16-cell Slurm array (4 case × 4 (interp_type, coarsen_type))。BoomerAMG setup_solve pattern-only spike;Axis 4 (`cycle_complexity`) 硬编码 estimate `= 2 × operator_complexity` per Saad 2003 §13 V-cycle theoretical bound。**Outcome**: 全 16/16 cells PASS hierarchy build + V-cycle apply;heihe_x4 best combo (interp=6, coarsen=8) setup+apply 0.028 s;heihe_x16 setup+apply 0.116 s (24.5× 优于 KLU per-step estimate 2.84 s);但 Axis 4 hardcoded estimate 触发 strict NO-GO-both / amended GO 二分。ADR-0007 §Discussion 揭示 Axis 4 与 Axis 5 mechanical 同构 (carries zero independent diagnostic signal),recommend operationally 走 amended GO 并把 integrated verification 推给 §P8-tune.G0。See `docs/p8tune/p8tune_f_academic_summary.md`。
- **P8-tune.G0 — Integrated CVODE + SUNLinSol_Hypre BoomerAMG smoke (closed by G0 NO-GO-G0 verdict)**: 4-PR sequence + 4-cell 90-day SHORT Slurm array。**Outcome**: G0-1 default-compat PASS + G0-2 build PASS + G0-3 telemetry-real PASS + **G0-4 integrated-completes FAIL** (heihe_x16 MALFORMED SIGKILL at 8h budget) + G0-5 per-step wall PASS (heihe_x4 AMG 0.093 s < SPGMR 0.238 s) + G0-6 solver-stats PASS。BUT total wall on heihe_x4 15.1× WORSE (23660 s vs 1566.55 s),因 CVODE nst 38.8× 增 (254756 vs 6572),`ncfn=100138` (Newton control failure ≈ nfe 的 20%) 主导。`g0_verdict_branch=NO-GO-G0`;AMG-not-beneficial at integrated-CVODE level。See `docs/p8tune/p8tune_g0_academic_summary.md`。

## §2.3 PR-X1 (G0-RCA) 的定位

PR-X1 是 G0 NO-GO 后 用户直接指定的 last-hypothesis 检验 spike。核心问题:G0 `ncfn=100138` 究竟是 (i) AMG-preconditioned Newton 系统结构性 stiff 使 outer Newton 反复 fail (intrinsic),还是 (ii) Hypre 内核解到过松 tolerance 导致 Newton correction 数值 noisy 使 Newton fail (tolerance mismatch)。这两个 hypothesis 有 vastly different implication:(i) 意味着 AMG path 结构性 closed;(ii) 意味着 tighten 双端 tolerance 可 rescue AMG。PR-X1 显式测 (ii) hypothesis,`AMG_TOL ∈ {1e-7, 1e-9, 1e-11, 1e-13}` × `EpsLin ∈ {0.05 default, 0.005}` 8-cell 矩阵;refutation 直接给到 solver-substitution line 关闭的 last-call rationale。

## §2.4 与外部工作的对比

SHUD 水文模型 CVODE BDF/Newton 集成层的 CPU 稀疏求解 substitution 是 一个 domain-specific 且 rarely-formally-published 的问题空间。相近的 published work:

- **Hindmarsh et al. 2005 [2]** SUNDIALS suite 通用文档,但未涵盖 hydrology BDF/Newton 场景下 KLU/AMG vs SPGMR 的 empirical comparison。
- **Henson & Yang 2002 [3]** BoomerAMG 通用性能 characterization,但目标是 elliptic Poisson-like PDE,不是 hydrology parabolic-hyperbolic mixed system。
- **Davis 2006 [4]** SuiteSparse KLU 直接稀疏求解,包括 AMD/COLAMD ordering + BTF 算法讨论,但未 empirical characterize hydrology Jacobian 上 KLU wall scaling。
- **Brown & Saad 1990 [5] + Eisenstat & Walker 1996 [6]** inexact Newton + forcing-term selection 理论,PR-X1 H4 hypothesis 的核心 theoretical anchor (但 empirically 显示 forcing-term selection 无 leverage on ncfn)。

本 retrospective 的贡献:一个 domain-specific negative-result corpus, 明确记录 CPU 稀疏求解替代在 hydrology BDF/Newton 集成层的 empirical scaling ceiling。

---

# §3 Methodology / 方法论

本章综述 6 个 P8 sub-epic 各自 methodology,聚焦跨 epic 递进 (pattern-only → integrated → RCA sanity matrix) 与 hypothesis-driven closure ledger。

## §3.1 6-epic 方法学递进 pattern

**Tab. 1: P8 solver-substitution research line methodology 递进**

| Sub-epic | 方法学 tier | Scope | 验收 axes | Verdict semantics |
|---|---|---|---|---|
| P8-tune.B (P8-precond) | integrated CVODE PSetup/PSolve callback | 5-block precond over SPGMR | `nfeLS / nfe` ratio + wall improvement | ADR-0003 PREC_NONE NO-GO baseline set |
| P8-tune.C (SPGMR maxl) | integrated CVODE + PRD 60-cell | 5-value maxl sweep × 4 case × 3 rep | wall (per case + per maxl) + G7 split-gate | ADR-0004 Optional-knob (SHUD_SPGMR_MAXL=30 Performance-tier) |
| P8-tune.D (KLU pattern-only) | pattern-only spike (binary shell-out; NO SHUD run, NO CVODE wire-up) | 4 case × 4 ordering (AMD, COLAMD, natural, custom) 16 cells | 3-axis: fill_ratio + RSS + numeric_factor wall | ADR-0005 Case-aware (small=feasible, x4=Optional, x16=NO-GO) |
| P8-tune.F (BoomerAMG pattern-only) | pattern-only spike + hardcoded Axis 4 estimate | 4 case × 4 (interp_type, coarsen_type) 16 cells | 5-axis: setup_wall + apply_wall + RSS + cycle_complexity + operator_complexity | ADR-0007 strict NO-GO-both / amended GO |
| P8-tune.G0 (integrated smoke) | integrated CVODE + SUNLinSol_Hypre + real Hypre telemetry | 4 case × 90-day SHORT 4 cells | 6-gate: default-compat + build + telemetry-real + integrated-completes + wall-signal + solver-stats | G0 NO-GO-G0 verdict |
| PR-X1 (G0-RCA) | integrated CVODE + SUNLinSol_Hypre tolerance × EpsLin matrix | heihe_x4 × 2 AMG_TOL 4× × 2 EpsLin 2× 8 cells (90-day SHORT) | H4 hypothesis (ncfn range across tolerance sweep) | H4 REFUTED |

方法学递进的核心特征:(i) B/C 是 integrated CVODE 直接测 (P8 line 起点);(ii) D/F 退回到 pattern-only spike (由于 KLU/AMG integrated wire-up 需要显著工程投入,先 pattern-only 判断可行性);(iii) G0 是 D/F 结论触发的 integrated 后续 (只对 F winner combo 集成 CVODE 验证);(iv) PR-X1 是 G0 last-hypothesis RCA (在 heihe_x4 上 8-cell sanity 矩阵)。这一 pattern-only → integrated → RCA sanity matrix 三级递进适应 CPU 稀疏求解 substitution 的调研效率约束。

## §3.2 Hypothesis-driven closure ledger

跨 epic H1-H4 假设的 stated-tested-refuted 链是 P8 line 的核心 traceable audit trail。**Tab. 2 (§5 Results §5.5 also cross-references)**:

| Hypothesis | Sub-epic anchor | Stated in | Test evidence | Verdict | Closure date |
|---|---|---|---|---|---|
| H1 pattern-only structural audits identify wall-improvement candidates | P8-tune.B - E | ADR-0004 §Context (2026-06-27) | P8-tune.C 60-cell + P8-tune.D 16-cell + P8-tune.F 16-cell (pattern axes PASS but wall axis FAIL) | PARTIAL REFUTED | 2026-06-29 (post-P8-tune.F PR-D capstone) |
| H2 P8-tune.F best combo translates to integrated CVODE | P8-tune.F PR-D → G0 | P8-tune.F PR-D 2026-06-29 capstone | G0 telemetry structure CONFIRMED (0.0% drift) BUT integrated wall REFUTED (15.1× regress) | REFUTED (integrated wall) / CONFIRMED (structure) | 2026-06-30 (G0 NO-GO verdict) |
| H3 integrated AMG delivers ≥ 10% wall improvement | P8-tune.G0 | ADR-0007 §Forward action Amendment 2026-06-29 | G0 4-cell smoke: heihe_x4 per-step 0.39× 优 BUT total wall 15.1× regress | REFUTED | 2026-06-30 (G0 NO-GO verdict) |
| H4 tolerance mismatch is the ncfn root cause; tightening rescues AMG | PR-X1 | PR-X1 #420 openspec change 2026-06-30 | PR-X1 8-cell heihe_x4: ncfn ∈ [98,286, 104,795] across 4 orders-of-mag × 10× sweep | REFUTED | 2026-06-30 (PR-X1 aggregate_rca.tsv MARKER:PR_X1_VERDICT_BEGIN..END) |

H4 refutation is the immediate trigger for PR-X2 (this retrospective) — the last salvage hypothesis is empirically dead, closing the entire solver-substitution research line。

## §3.3 ADR chain append-only pattern

跨 sub-epic ADR (0003 - 0007) + 本 PR-X2 ADR-0008 采用 append-only amendment pattern:每个 ADR 的 §Status + §Decision 一旦 Accepted 就 byte-identical invariant,新证据只以 dated `## Amendment YYYY-MM-DD — <topic>` 分层 append 到 §Forward action 末尾。这一 pattern 有三个显式好处:(i) 可 mechanically 通过 `git diff` verify no re-litigation;(ii) 保持 decision history readable — 读者可按时序追踪 amendment 堆叠;(iii) 避免 rewrite history 引起 audit confusion。ADR-0007 的 amendment 堆叠 (2026-06-29 framing amendment + 2026-06-30 G0 verdict + 2026-06-30 G0-RCA outcome) 是这一 pattern 的典范。

## §3.4 Case-asymmetric scaling 表征

P8 line 反复出现的 methodological finding: 求解器加速比是 **case-asymmetric** 的,即小 case (keliya NumY 1.8K / heihe NumY 21K) vs 大 case (heihe_x4 NumY 124K / heihe_x16 NumY 485K) 的加速比 vs 求解器 substitution 呈算法-类不同的 scaling regime:

- 小 case: OMP runtime 固定 overhead + cache locality 反转 + NUMA migration 物理 limit 主导 (P1e `docs/p1e/p1e_perf_baseline.md` §6 v0.2 GPT Pro fact-check);任何求解器 substitution 会因 setup overhead 反而慢或持平。
- 大 case: L2/L3 cache 饱和 + Krylov subspace 超 cache + AMD LU ordering O(N^1.5) 主导;这里才是求解器 substitution 的算法优势 zone。

这一 case-asymmetric scaling 表征在 ADR-0005 § Discussion (KLU case-aware) + ADR-0007 §Discussion (BoomerAMG small-case invert) + G0 keliya-vs-heihe_x4 setup overhead 数据 中都独立观察到。P8 methodological anchor: **求解器 substitution 决策必须 case-asymmetric 化,不能全 case 用同一 verdict**。

## §3.5 Integrated-vs-pattern-only verdict 分歧

P8 methodology 的一个核心 lesson 是 pattern-only spike verdict 与 integrated CVODE verdict 之间的分歧 potential:

- P8-tune.F 16-cell pattern-only 显示 heihe_x16 AMG best combo setup+apply 0.116 s vs KLU 2.84 s (24.5× 优),暗示 AMG 是大 case 唯一 viable substrate。
- P8-tune.G0 4-cell 集成 CVODE smoke 显示 heihe_x4 AMG 集成端 wall 15.1× 差 (per-step 0.39× 优被 nst 38.8× 增抹掉)。

Pattern-only spike 只测 single V-cycle apply,没有 Newton iteration + step controller 反馈循环 — 而 integrated CVODE 的 wall consumption 主导来自 outer-loop (Newton + step controller),而非 inner-loop (Krylov subspace)。这一 methodology lesson 直接产生两个 anchor:(i) 未来 architectural-substrate 研究 SHALL 有 integrated-CVODE gate,不能只停在 pattern-only;(ii) integrated wall 反驳 (在充分证据下) 比 pattern-only 二分 verdict 强。

---

# §4 Experimental Setup / 实验设置

## §4.1 硬件平台

**Tab. 3: 双端硬件 + 软件栈**

| 项 | 服务器 (P8-tune.C/D/F/G0/PR-X1 权威) | Mac (P1e + P8-tune.G0 PR-0 wrapper + PR-X1 pre-flight) |
|---|---|---|
| Endpoint | `frd_muziyao@210.77.77.22:32099` (cn05-06,09,14-19,23-24 CPU partition; gn01 GPU partition) | Apple M4 Pro local |
| OS / Kernel | Ubuntu 24.04.2 LTS, Linux 6.8.0-57-generic | Darwin 24.6.0 (macOS Sequoia 15) |
| CPU | Intel Xeon dual-socket NUMA (cn14/cn23 verified 173 GiB RAM;各 cn-node 40-core 2-NUMA-node) | Apple M4 Pro 14-core (4P + 10E) |
| Compiler | GCC 13.3.0 | Apple Clang 17.0.0 |
| OMP runtime | libgomp.so.1 | libomp 22.1.7 (Homebrew) |
| SUNDIALS | 6.0.0 (pinned, unchanged 全 P8 chain) | 同 |
| Hypre | 3.1.0 (`/scratch/frd_muziyao/local/hypre-3.1.0/`) | brew Hypre 3.1.x |
| SuiteSparse (KLU) | apt-installed cn-node 版本 (P8-tune.D pinned) | brew SuiteSparse |
| Scheduler | Slurm sbatch from `/scratch` + `--output/--error` in `/scratch` | local shell |
| 关键 Slurm 三铁律 | (1) 从 `/scratch` 下 sbatch (policy 拦 `/users/$USER`);(2) `--output/--error` 在 `/scratch`;(3) 作业脚本引用 patch/hash/run.sh 都放 `/scratch` | n/a |

## §4.2 Benchmark cases

**Tab. 4: P8 benchmark cases roster** (per CLAUDE.md `docs/case_deployment_map.md` 唯一权威)

| Case | NumEle | NumY (est.) | 常驻位置 | P8-tune 覆盖 | 90-day truncated |
|---|---:|---:|---|---|---|
| keliya | 484 | 1,785 | Mac + Server `Basins/keliya/` | B/C/D/F/G0 (small ref) | YES |
| xinanjiang_upstream | 801 | ~3,000 | Server `Basins/xinanjiang_upstream/` (project name `xinanjiang`) | G0 (medium ref, replaces P8-tune.F heihe) | YES |
| heihe | 6,335 | 21,357 | Server `Basins/heihe/` | B/C/D/F (secondary) | YES |
| heihe_x4 | 40,046 | 124,395 | Server `Basins/heihe_x4/` (常驻; AutoSHUD pipeline 2026-06-17) | B/C/D/F/G0/PR-X1 (**primary decisive**) | YES |
| heihe_x16 | 160,331 | 485,250 | Server `Basins/heihe_x16/` (常驻) | D/F/G0 (production-target ceiling) | YES |

heihe_x4 是 P8 line 全阶段的 decisive case — ADR-0005 KLU verdict, ADR-0007 AMG-strict verdict 触发条件, G0 integrated smoke primary, PR-X1 RCA scope 都以 heihe_x4 为主。heihe_x16 是 production-target ceiling,pattern-only 阶段可覆盖但 integrated 阶段常因 8h Slurm budget SIGKILL 而 MALFORMED (G0-4)。

## §4.3 项目级铁律

- **所有 case ≤ 90 天截断** per CLAUDE.md 项目级铁律。理由:OpenMP 并行验证 + bitwise neutrality + golden 生成不需要 4 年 model time,3 个月足够给信号。
- **CMFD forcing V0200** (1951-01 → 2024-12) 强制;V0106 淘汰 (heihe NE 16 站 forcing.csv 全空)。
- **Python `uv run` / `uv pip`** 强制,禁 `python` / `python3` / `pip`。
- **双端 CLAUDE.md 同步**: rsync 到 server 每次本地改动。

## §4.4 Slurm 三铁律 (P8 全阶段 服务器实验)

**§4.4.1 从 `/scratch` 下 sbatch** (policy 拦 `/users/$USER` 提交)。

**§4.4.2 `#SBATCH --output/--error` 路径必须在 `/scratch`** (compute node 的 `/tmp` node-local,作业结束会丢, sacct 显示 ExitCode 127)。

**§4.4.3 作业脚本里引用的 patch / hash / run.sh 都放 `/scratch`** (不能放 `/tmp`)。

P8 提交位:`/scratch/frd_muziyao/SHUD-OpenMP/.<stage>-runs/` dot-prefixed scratch 子目录:
- P8-tune.C: `.p8tune.C-runs/`
- P8-tune.D: `.p8tune.D-runs/`
- P8-tune.F: `.p8tune.F-runs/p8f_amg_spike.9896/`
- P8-tune.G0: `.p8tune.G0-runs/g0_amg_smoke.10014/` (PR-A) + `g0_amg_smoke_rerun.<jobid>/` (PR-B rerun)
- PR-X1: `.p8tune-rca-pr-x1-runs/` (Slurm 10248)

## §4.5 Reproducibility footprint

完整 reproducibility footprint (per PR-X1 as most recent example):

```bash
# 1. Sync to main + verify commit
git checkout main
git pull --ff-only --recurse-submodules
git rev-parse HEAD  # expected 39bd6b1 or later at PR-X2 time

# 2. Build SHUD (server)
ssh frd_muziyao@210.77.77.22 -p 32099
cd /scratch/frd_muziyao/SHUD-OpenMP
cd SHUD && make clean && make shud_omp HYPRE=1

# 3. Access review evidence
ls .review-evidence/g0-amg-rca-pr-x1/
cat .review-evidence/g0-amg-rca-pr-x1/README.md
awk '/PR_X1_VERDICT_BEGIN/,/PR_X1_VERDICT_END/' .review-evidence/g0-amg-rca-pr-x1/README.md
column -t -s$'\t' .review-evidence/g0-amg-rca-pr-x1/aggregate_rca.tsv
```

---

# §5 Results / 结果

本章按 sub-epic 时序汇报 P8 line 关键数值 + P8-tune.G0 6-gate verdict + PR-X1 H4 refutation。

## §5.1 P8-tune.B/C 生产 SPGMR baseline anchor

- P8-tune.B (P8-precond): 5-block precond `nfeLS / nfe` ratio 与 PREC_NONE baseline 持平 (`nfeLS / nfe ≈ 1.8` on heihe_x4),precond setup + solve overhead 抵消 Krylov iter 减少收益。ADR-0003 PREC_NONE NO-GO。
- P8-tune.C (SPGMR maxl sweep 60-cell PRD, 3-rep median):
  - heihe_x4 N=1 maxl=5 baseline: `1489.76 s / 6575 nst = 0.226579 s / step` (ADR-0004 §Discussion anchor)。
  - heihe_x4 N=1 maxl=10 至 50: 6.86-24.82% regress (Krylov 子空间超 L2 cache)。
  - heihe N=1 maxl=30: 12% improve (`SHUD_SPGMR_MAXL=30` Performance opt-in tier ship, NOT A5-certified)。

生产 SPGMR PREC_NONE baseline 从 P8-tune.C 起锁定,后续所有 P8 sub-epic 的 wall reference 都用此 baseline。G0 更新到 case-specific baseline `SPGMR_PER_STEP_HEIHE_X4_S = 0.238369` + `SPGMR_PER_STEP_HEIHE_X16_S = 0.952489` (PR-A 实测 + PR-B hot-patched)。

## §5.2 P8-tune.D KLU 16-cell 三轴 verdict (case-aware)

**Tab. 5: P8-tune.D KLU 16-cell pattern-only 3-axis verdict** (per ADR-0005 §Decision):

| Case | fill_ratio | RSS (fraction of 0.7×cn-RAM budget) | numeric_factor wall (fraction of 0.7×SPGMR budget) | 3-axis verdict |
|---|---:|---:|---:|---|
| keliya | 1.184 | 0.001× | 0.056× | GO |
| heihe | 1.923 | 0.006× | 0.140× | GO |
| heihe_x4 | 2.487 | 0.032× | **1.867×** | Optional (near-miss) |
| heihe_x16 | 3.056 | 0.202× | **17.89×** | NO-GO |

heihe_x4 fill_ratio (2.487) + RSS 都 fine (32× headroom on RAM);但 numeric_factor wall (**1.867×** budget) 超预算,KLU wall scaling 在 NumY ≥ 100K 处不可行。heihe_x16 一致 (17.89× 更严),结构 NO-GO。ADR-0005 Case-aware split。See `docs/p8tune/p8tune_d_academic_summary.md`。

## §5.3 P8-tune.F BoomerAMG 16-cell 五轴 verdict (strict-vs-amended)

**Tab. 6: P8-tune.F BoomerAMG 16-cell pattern-only 5-axis verdict** (per ADR-0007 §Decision):

| Case | best combo (interp, coarsen) | setup_wall_sec | apply_wall_sec | peak_rss_MB | operator_complexity | cycle_complexity (hardcoded = 2 × op) | 5-axis strict verdict | 4-axis amended (drop Axis 4) |
|---|---|---:|---:|---:|---:|---:|---|---|
| keliya | (6, 21) NN=02 | 0.000561 | 0.001031 | 19.0 | 1.0029 | 2.0059 | **FAIL** (Axis 4) | PASS |
| heihe | (8, 8) NN=07 | 0.001452 | 0.001667 | 34.4 | 1.0000 | 2.0000 | **FAIL** (Axis 4) | PASS |
| heihe_x4 | (6, 8) NN=08 | 0.009476 | 0.018179 | 110.9 | 1.0000 | 2.0000 | **FAIL** (Axis 4) | PASS |
| heihe_x16 | (6, 8) NN=12 | 0.037785 | 0.078349 | 381.3 | 1.0000 | 2.0000 | **FAIL** (Axis 4) | PASS |

strict verdict = NO-GO-both (`heihe_x4` fails Axis 4);amended verdict = GO (Axes 1/2/3/5 PASS at all cases)。ADR-0007 §Discussion "Axis 4 amendment per PR-A H3 disclosure" 指明 Axis 4 = `2 × operator_complexity` 是 hardcoded estimate (Saad 2003 §13 V-cycle bound 直接抄) — 全 16 cells cycle_complexity mechanically 跟踪 Axis 5,carries zero independent diagnostic signal。ADR §Forward action Amendment 2026-06-29 (post-Linus-review framing amendment) demote Axis 4 从 hard blocker 到 hierarchy-quality diagnostic + split §P8-tune.G into three sequential gates G0 → G1 → G2。See `docs/p8tune/p8tune_f_academic_summary.md`。

## §5.4 P8-tune.G0 integrated CVODE smoke 6-gate verdict

**Tab. 7: P8-tune.G0 6-gate verdict** (per `docs/p8tune/amg_g0_verdict.md` byte-identical anchor):

| Gate | Result | One-line 证据 |
|---|---|---|
| G0-1 default-compat | PASS | `SHUD_LINSOL` unset + `=spgmr` bit-identical vs pre-G0 SPGMR baseline (Mac + server per-platform anchor) |
| G0-2 build | PASS | `make shud_omp HYPRE=1` green Mac (brew Hypre 3.1.x) + Ubuntu CI (apt Hypre 3.1.0) + server (`/scratch/.../hypre-3.1.0/`) |
| G0-3 telemetry-real | PASS | keliya `cycle_complexity=1.000000` + `operator_complexity=1.002945` from real Hypre API (`HYPRE_BoomerAMGGetCumNnzAP` + `HYPRE_BoomerAMGGetOperatorComplexity`) |
| G0-4 integrated-completes | **FAIL** | heihe_x16 MALFORMED — Slurm SIGKILL at 8h budget fired before SIGTERM trap could emit cell_summary |
| G0-5 wall-signal | PASS (per-step) | heihe_x4 AMG per-step `0.092873 s` < SPGMR `0.238369 s` (0.39×);G0-5 OR-gate satisfied |
| G0-6 solver-stats | PASS | `cvode_nfe / nli / nfeLS / ncfn / ncfl / netf` non-NA for 3 AMG_OK cells;heihe_x16 MALFORMED exempt |
| **G0_OVERALL** | **NO-GO** | G0-4 FAIL blocks GO-G0 branch; heihe_x4 total wall 15.1× regress independently confirms AMG-not-beneficial at integrated-CVODE level |

**Tab. 8: G0 wall-signal per-cell**:

| Cell | AMG per-step (s) | AMG nst | SPGMR nst | AMG/SPGMR nst ratio | Total wall ratio (AMG / SPGMR) | wall_convention |
|---|---:|---:|---:|---:|---|---|
| keliya | 0.001432 | 124913 | n/a | n/a | n/a | telemetry_truncated |
| xinanjiang_upstream | 0.002709 | 33223 | n/a | n/a | n/a | wall_total_proxy |
| heihe_x4 | 0.092873 | 254756 | 6572 | **38.8×** | **15.1× WORSE** (23660 s vs 1566.55 s) | wall_total_proxy |
| heihe_x16 | n/a (MALFORMED) | n/a | 6556 | n/a | n/a | nst_unavailable |

heihe_x4 CVODE solver stats:`cvode_nfe=502982, cvode_nli=983928, cvode_nfeLS=124395, cvode_ncfn=100138, cvode_ncfl=0, cvode_netf=2`。`ncfn` 主导 (Newton control failure 100138 ≈ nfe 的 20%);`ncfl=0` 全 cell (AMG 内核每 Krylov call 收敛)。See `docs/p8tune/p8tune_g0_academic_summary.md`。

## §5.5 PR-X1 (G0-RCA) H4 refutation matrix

**Tab. 9: PR-X1 8-cell tolerance × EpsLin sanity matrix** (per `.review-evidence/g0-amg-rca-pr-x1/aggregate_rca.tsv`):

| Cell | AMG_TOL | EpsLin | State | nst | ncfn | ncfl | wall_total (s) | Node |
|---|---|---|---|---:|---:|---:|---:|---|
| 0 | 1e-7 | 0.05 (default) | COMPLETED | 251,883 | **98,286** | 0 | 27,908 | cn23 (7-way contended) |
| 1 | 1e-9 | 0.05 | COMPLETED | 253,558 | **99,095** | 0 | 28,117 | cn23 |
| 2 | 1e-11 | 0.05 | TIMEOUT | NA | NA | NA | >28,800 | cn23 |
| 3 | 1e-13 | 0.05 | TIMEOUT | NA | NA | NA | >28,800 | cn23 |
| 4 | 1e-7 | 0.005 | COMPLETED | 257,610 | **104,795** | 0 | 28,142 | cn23 |
| 5 | 1e-9 | 0.005 | COMPLETED | 256,525 | **103,970** | 0 | 28,553 | cn23 |
| 6 | 1e-11 | 0.005 | TIMEOUT | NA | NA | NA | >28,800 | cn23 |
| 7 | 1e-13 | 0.005 | COMPLETED | 256,138 | **104,390** | 0 | 22,871 | cn08 (alone) |

**G0 baseline reference (PR-A, heihe_x4 alone)**: `nst=254,756, ncfn=100,138, wall=23,660 s`。
**SPGMR baseline reference (PR-0, heihe_x4)**: `nst=6,572, ncfn=49, wall=1,566.55 s`。

**MARKER:PR_X1_VERDICT_BEGIN**:
```
best_cell=0
best_ncfn=98286
best_nst_ratio=38.3267
best_wall_ratio=17.8149
amg_reopens=false
num_present=8
num_amg_ok=5
spgmr_baseline_ncfn=49
spgmr_baseline_nst=6572
spgmr_baseline_wall_sec=1566.5518531799316
```
**MARKER:PR_X1_VERDICT_END**

**H4 hypothesis REFUTED**: 5 completed cells 全部 `ncfn ∈ [98,286, 104,795]` (6% 窗内);跨 4 orders-of-magnitude `AMG_TOL` (1e-7 → 1e-13) × 10× `EpsLin` (0.05 → 0.005) 8-cell 矩阵 `ncfn` 全部远超 SPGMR 单位数 baseline (49 vs ≈100k);`ncfl=0` 全 cell (Hypre 内核每次收敛,failure 在 outer Newton)。tolerance mismatch 不是 ncfn 根因。`amg_reopens=false` 明确记录 AMG path 不 re-opens。

## §5.6 H1-H4 hypothesis verification summary

**Tab. 10: P8 line 全链 H1-H4 假设验证 summary** (mirror of §3.2 with results):

| Hypothesis | Statement | Test | Verdict | Closure trigger |
|---|---|---|---|---|
| H1 | pattern-only structural audits identify wall-improvement candidates | P8-tune.C/D/F pattern axes | PARTIAL REFUTED (pattern PASS, wall FAIL) | ADR-0005 + ADR-0007 |
| H2 | P8-tune.F best combo translates to integrated CVODE | G0 telemetry vs pattern-only | STRUCTURE CONFIRMED + INTEGRATED WALL REFUTED | G0 NO-GO verdict |
| H3 | integrated AMG delivers ≥10% wall improvement | G0 4-cell smoke | REFUTED (heihe_x4 15.1× regress) | G0 NO-GO verdict |
| H4 | tolerance mismatch is ncfn root cause; tightening rescues AMG | PR-X1 8-cell heihe_x4 90-day | REFUTED (ncfn 6% window across 4 orders-of-mag) | ADR-0008 (this closure) |

H4 refutation is the terminal closure signal for the P8 solver-substitution research line。

---

# §6 Discussion / 讨论

## §6.1 Why H1-H3 all eventually failed at integration boundary

Pattern-only spike (H1) 与集成 CVODE (H2/H3) verdict 之间的 systematic 分歧是 P8 line 最重要的 methodological finding。P8-tune.F 16-cell pattern-only 显示 BoomerAMG 单 V-cycle apply 在 heihe_x16 上 0.116 s (vs KLU per-step 2.84 s, 24.5× 优),这是 hierarchical algorithm 的自然算法优势 — O(N) 每 V-cycle vs O(N^1.5) LU factor。但 P8-tune.G0 集成 CVODE 显示 AMG per-step 0.093 s 于 heihe_x4 (vs SPGMR 0.238 s, 0.39×) 仍在 pattern-only 数量级,BUT total wall 15.1× regress (23660 s vs 1566.55 s)。

分歧根源 = **outer-loop 反馈循环**。Pattern-only spike 只测单 V-cycle 或单 Krylov call,没有 CVODE Newton iteration + step controller 反馈:
- Pattern-only: `wall = setup + apply` (one call each)
- Integrated: `wall ≈ nst × (Newton iterations × Krylov apply + step retry cost + Jacobian rebuild)`

在集成 CVODE 上,AMG-preconditioned Newton system 显示 `ncfn ≈ 100,138` (Newton 控制失败率 = nfe 的 20%),`ncfl = 0` (内核 Krylov 每次收敛);SPGMR-preconditioned 显示 `ncfn ≈ 49` (Newton 失败几乎不发生),`ncfl = 0` (同)。inner-loop 都健康,但 AMG-preconditioned outer Newton 存在结构性 stiff behavior — 具体来说,Newton correction 在 AMG-preconditioned Jacobian 上产生的 y 更新反复 fail CVODE 的 step size adaptor,触发 step retry + smaller dt + 反复 rebuild Jacobian + 反复 setup AMG hierarchy。这是 pattern-only spike 无法测的。

**Methodological lesson**: 未来 architectural-substrate 研究 MUST 有 integrated CVODE gate,不能停在 pattern-only。P8 methodology 的 pattern-only → integrated → RCA 三级递进设计正是应对这一 lesson。

## §6.2 What per-step PASS / total wall FAIL means (G0)

G0-5 gate spec 写 `amg_wall_per_step < spgmr_wall_per_step`,heihe_x4 per-step 0.093 < 0.238 满足,G0-5 nominally PASS。但 total wall 15.1× regress 使 G0_OVERALL=NO-GO,分歧显示 spec letter-of-the-law verdict 与 production wall reality 冲突。

这是 P8 methodology 的关键 refinement:**per-step wall benefit 在 integrated CVODE 中可能被 outer-Newton control failure 抹掉**。Future G1 gate (即使 AMG path 未来 re-open) MUST 以 total wall 为度量,而非 per-step。G0-5 spec 需 amend 但因 AMG path 结构性 closed,不再重要 — 记录为 methodological footnote 即可。

## §6.3 What ncfn=100k means algorithmically

`ncfn` = Newton nonlinear convergence failures counter (`CVodeGetNumNonlinSolvConvFails`)。在 CVODE 6.0.0 BDF/Newton 框架下,`ncfn` 每次 increment 表示:
1. Newton iteration 在 given step 上 fail convergence within `MaxNonlinIters` iterations (default 3, per SUNDIALS default);
2. Step controller triggers step-size reduction (通常 `dt := dt * 0.25`) + Newton retry;
3. 若 Newton 仍 fail multiple times,step controller reduce dt further or abort。

`ncfn ≈ 100,138` on heihe_x4 意味着:AMG-preconditioned Newton 在 254,756 total steps 中 有约 20% 需要 retry (不含 successful step 内部的 Newton iterations)。这不是 inner-loop Krylov failure (`ncfl=0` 显示 inner solves 每次 converge),而是 outer Newton correction 与 step controller adaptor 之间的 systematic mismatch。

algorithmically,可能的解释:
- AMG-preconditioned Jacobian 保持稀疏结构但改变 numerical conditioning,产生的 Newton correction 步长 not-well-aligned with CVODE step controller 的 error estimation。
- Newton iteration matrix `M = I - γJ` (γ = current step scale) 在 AMG preconditioner 下 spectral properties 与 SPGMR PREC_NONE 显著不同,`MaxNonlinIters=3` 默认在 AMG 下可能不够。
- CVODE step controller 依赖 `NonlinConvCoef` (nonlinear convergence tolerance 相对因子, default 0.1) 与 solver 内部 Newton iteration norm 阈值的组合;AMG 改变 Newton iteration 内 residual reduction pattern,该 default 参数组合不再合适。

PR-X1 只 tested 一个 axis (`AMG_TOL` × `EpsLin`),没测 `MaxNonlinIters` / `NonlinConvCoef` / `reltol` — 这些 axes 是 §P9 (PR-Z1) 的 scope,但 §P9 只对 SPGMR baseline tune (P8 solver-substitution line 关闭,AMG path 不再 investigated)。

## §6.4 与 prior epic P1e 对比

P1e Path 1 生产基线 (`ExecPolicy::StrictOMP` + Serial NVec + PREC_NONE SPGMR maxl=5) 通过 architectural correctness (single parallel region + phase-based for + `SHUD_RHS_THREADS` env split) 实现 1.729× heihe_x4 加速,architectural pattern = "并行正确的层次而非算法替代"。

P8 solver-substitution research line 尝试通过算法替代进一步加速,失败。这一 P1e-vs-P8 对比揭示 SHUD-OpenMP 加速的深层结构:**RHS 并行 (P1e Path 1) 是正确的加速抽象层次;linear-solver 替代 (P8) 是错误的**。原因:SHUD RHS 是 embarrassingly-parallel 类 (element bucket 独立计算),Amdahl 分子分母都清晰;CVODE linear solver 是 strong-coupling 类 (Newton iteration + step controller 全局反馈),Amdahl 分析不适用,pattern-only spike verdict 会 systematically 高估。

**Methodological lesson for future acceleration research**: 优先加速 **embarrassingly-parallel 类** phases (RHS, output write, forcing read),避免 **strong-coupling 类** phases (linear solver, Newton iteration) 除非有直接算法-类改进 (如从 CPU 走到 GPU,或 SPGMR 走到 preconditioned SPGMR — 但 P8-tune.B 已证 block-Jacobi 不 work)。

## §6.5 Case-asymmetric scaling 的三次独立观察

P8 line 三个 sub-epic 独立观察到 case-asymmetric scaling:
1. **ADR-0005 (P8-tune.D)**: keliya + heihe 3-axis PASS,heihe_x4 Optional,heihe_x16 NO-GO — KLU fill scaling O(N^1.5) 主导。
2. **ADR-0007 (P8-tune.F) §Discussion**: keliya AMG (1.59 ms) 比 KLU (0.9 ms) SLOWER 1.8×,heihe_x16 AMG 比 KLU 24.5× 优。BoomerAMG hierarchy setup overhead 在 NumY=1785 处不 amortize。
3. **P8-tune.G0 (§5.4 Tab. 8)**: keliya AMG per-step 1.4 ms (setup+solve only) vs xinanjiang_upstream 2.7 ms (wall_total_proxy)。小 case AMG per-step 高于 medium case,setup overhead 反 dominant。

三次独立 case-asymmetric 观察确认 **求解器 substitution decision 必须 case-asymmetric**。ADR-0008 §Decision 明确记录 case-asymmetric 结论:小 case 走 SPGMR maxl=5 default OR `SHUD_SPGMR_MAXL=30` opt-in (ADR-0004);大 case 走 SPGMR maxl=5 (KLU + AMG 都无 wall benefit)。

## §6.6 Why the closure is DEFINITIVE (not "let's try one more axis")

有效性上,可能会问:PR-X1 只 tested 一个 axis (`AMG_TOL` × `EpsLin`),为什么不再 tested `MaxNonlinIters` × `NonlinConvCoef` × `reltol` axes 再决定?这一 concern legitimate,但 cost-benefit 分析显示:

- **AMG 已投入 4 个 sub-epic + ≥25 个 PR (F 5-PR + G0 4-PR + PR-X1 1-PR + ancillary)**;每额外 axis 需要 4-8 cell Slurm sweep + 2-3 天 investigation。
- **PR-X1 refutation 已经足够 orthogonal**: `AMG_TOL` × `EpsLin` 是 inner-solve tolerance axes,`MaxNonlinIters` × `NonlinConvCoef` × `reltol` 是 outer-Newton control axes。inner-solve tolerance 无 leverage 强烈暗示 outer Newton 结构性 stiff,而不是 tolerance mismatch。
- **PR-X1 evidence 已经足够 crisp**: `ncfn ∈ [98286, 104795]` 6% 窗内,不是 borderline 数据 (若是 60-80k range 才 borderline);SPGMR ncfn=49 与 AMG ncfn=100k 是 3 orders-of-magnitude 差,tolerance tweaking 完全无 leverage。
- **user direction explicit**: 用户在 PR-X1 closure 时明确 mandate 转向 A5 infra + CVODE policy tuning + decomposition design,GPU NOT pursued。

Therefore, PR-X2 closure is definitive under a bounded-effort-per-hypothesis principle。Further AMG investigation carries low expected value relative to §A5-infra (PR-Y1) + §P9 (PR-Z1) + §P10 design 三个 forward paths。

---

# §7 Limitations & Threats to Validity / 限制与效度威胁

## §7.1 PR-X1 3/8 cells timeout on cn23 (memory-bandwidth contention)

PR-X1 8 cells 部署在 Slurm 10248 array,job scheduler 把 7 cells (0-6) 集中到 cn23 单节点,cell 7 单独在 cn08。7-way concurrent AMG-preconditioned CVODE run 在 cn23 上是 memory-bandwidth-bound (sparse AMG hierarchy build 反复 stream matrix data through L3 cache),导致 wall time 显著膨胀。3 cells (2, 3, 6) 8h Slurm 预算超时,MARKED as MALFORMED。

**Impact on H4 refutation strength**: 这 3 cells 的数据丢失不改变 H4 refutation verdict,因为 completed cells (0/1/4/5/7) 已经 span AMG_TOL ∈ {1e-7, 1e-9, 1e-13} × EpsLin ∈ {0.05, 0.005} 5 组合,`ncfn ∈ [98286, 104795]` 数据充分。缺失的 3 cells (2, 3, 6) 是 tighter AMG_TOL 组合;若 tighter AMG_TOL 有 leverage,理应 monotone 影响 ncfn — 但 completed cells 显示 tolerance 完全无 leverage,tighter tolerance 只会增 per-solve Hypre iters 而不改 ncfn。3 cells 的 timeout 本身已经是 tighter tolerance 更慢 (per-solve iters 更多) 的 confirmation。

**Threat to validity level**: LOW。若未来 re-run 让 3 cells complete,verdict 不会改变。但如实记录 3/8 cells 数据缺失是 audit trail 完整性要求。

## §7.2 heihe_x16 (252k NumEle) 从未 completed under AMG

G0-4 (P8-tune.G0) + PR-X1 (heihe_x4 only scope) 都没有 heihe_x16 clean integrated CVODE 数据。G0 heihe_x16 SIGKILL at 8h budget,MALFORMED。PR-X1 明确 heihe_x4 only scope。

**Impact on P8 line closure strength**: heihe_x16 未 AMG-completed 是 admitted evidence gap。理论上 heihe_x16 AMG 完成后,per-step 可能仍 0.39× 优 (or better),但 total wall 15.1× regress pattern 由 outer Newton control failure driven,ncfn scaling 与 NumY 关系是弱的 (Newton control failure mode 是 outer-loop 现象,不是 inner-loop NumY scaling)。因此从算法 argument 推断 heihe_x16 AMG 也 15× 级别 regress,但直接 empirical 数据缺失。

**Threat to validity level**: MEDIUM。P8 line closure 仰赖 (i) heihe_x4 直接证据 + (ii) 算法 generalize argument。若未来有直接 heihe_x16 AMG 数据,可能 subtly 修改 closure 强度 (unlikely 会 re-open,但可能修改 forward-action wording)。

## §7.3 F-winner combo (interp=6, coarsen=8) 之外 coarsen/interp 未 tested at G0-RCA scope

P8-tune.F 16-cell 在 4 (interp_type, coarsen_type) 上 sweep;G0 hardcode F-winner combo (interp=6, coarsen=8);PR-X1 8-cell 也 hardcode F-winner combo。因此 H4 (tolerance rescue) 只在 F-winner combo 上被 refuted,理论上其他 coarsen/interp 组合可能有 different ncfn behavior。

**Impact on closure strength**: 这不是 fatal weakness。Argument:(i) F-winner combo 是 pattern-only spike 最优 setup+apply combo,若 ncfn 主导是 combo-specific,则 F-winner 应 minimal ncfn — 但 empirically 是 100k+;(ii) BoomerAMG hierarchy 的 setup/interp choice 影响 hierarchy quality (Axes 1/2/3/5),不改变 outer Newton feedback loop 的 stiff mode (这需 preconditioner spectrum 与 CVODE step controller adaptor 层面的 co-design,单换 coarsen/interp 不覆盖)。

**Threat to validity level**: LOW-MEDIUM。cost-benefit 视角:若未来有人 challenge 这一 conclusion,需要 4-cell 90-day AMG smoke × 4 alternate coarsen/interp = 16 cells (~3 天 Slurm) 才能反驳。当前 evidence + algorithmic argument 已足够 closure。

## §7.4 §P8-tune.E.small-only 未 closed by this retrospective

ADR-0008 明确不 close §P8-tune.E.small-only (KLU env-var mini-prototype for keliya + heihe small cases)。这是 scope 边界:E.small-only 只覆盖 keliya + heihe (NumY ≤ 21K),不影响 production-target heihe_x4 / heihe_x16。ADR-0005 §Forward action 已经 case-aware verdict 明确 small case 走 E.small-only。P8 line 主线 closure 不受 E.small-only status 影响。

**Threat to validity level**: NONE。E.small-only 是 explicit out-of-scope,不是 methodology gap。

## §7.5 SPGMR-only PR-X1 baseline

PR-X1 aggregate 中的 `spgmr_baseline_ncfn=49, spgmr_baseline_nst=6572, spgmr_baseline_wall_sec=1566.55` 是 heihe_x4 90-day SPGMR PREC_NONE maxl=5 单次 run 的数据 (per PR-0 pin)。这是 single-run baseline,不是 3-rep median (P8-tune.C PRD 60-cell median 是 `0.226579 s / step`,与 PR-X1 baseline `1566.55 / 6572 = 0.2383 s / step` 差异反映 90-day 长 run vs PRD 短 run 的 step size 差异,以及 case-specific vs 60-cell median 差异)。

**Threat to validity level**: NONE。SPGMR baseline 稳定性已由 P8-tune.C 60-cell PRD 独立 validate;PR-X1 直接 baseline 用于 ratio 比较,不需要 baseline 3-rep 精度。

## §7.6 Server node contention as confound

PR-X1 8 cells 中 7 cells co-located on cn23,cell 7 单独 on cn08。cell 7 wall (22871 s) vs cn23-co-located cells (27908-28553 s) 显示 memory-bandwidth 反 cause 22-25% wall inflation on cn23。但 `ncfn` (measurement of interest) 不受 wall inflation 影响 — `ncfn` 是 CVODE 内部 counter,是 numerical outcome,与 wall time 独立。

**Impact on H4 refutation**: NONE — ncfn is the diagnostic axis;wall inflation 是 audit trail 上的 honest disclosure,不影响 verdict。

**Threat to validity level**: NONE (对 H4 refutation);LOW (对 wall-related derived metrics if any)。

---

# §8 Conclusion / 结论

**P8 solver-substitution research line is CLOSED-FINAL** per ADR-0008 (2026-06-30)。跨 6 个 sub-epic (P8-tune.B/C/D/F/G0 + PR-X1) 5 天 empirical work + ≥25 PR sequence + 4 formalized hypotheses (H1-H4) 全部 stated → tested → refuted 后,CPU 侧 CVODE 线性求解器替代作为 SHUD hydrology BDF/Newton production-target 加速路径,empirically exhausted。

**关键 closure findings**:
1. P8-tune.B (block-Jacobi precond over SPGMR): `nfeLS/nfe` floor 与 PREC_NONE baseline 持平,ADR-0003 PREC_NONE NO-GO。
2. P8-tune.C (SPGMR maxl sweep 60-cell PRD): heihe_x4 全 maxl ≥ 10 regress,`SHUD_SPGMR_MAXL=30` Performance opt-in for small heihe only,ADR-0004 Optional-knob。
3. P8-tune.D (KLU pattern-only): fill scaling O(N^1.5) 主导,heihe_x4 wall margin 1.87× / heihe_x16 17.9× (structural),ADR-0005 Case-aware。
4. P8-tune.F (BoomerAMG pattern-only 16-cell): 全 16/16 hierarchy build + V-cycle apply PASS,但 Axis 4 硬编码 estimate 触发 strict-vs-amended verdict,ADR-0007。
5. P8-tune.G0 (integrated CVODE + SUNLinSol_Hypre 4-cell smoke): heihe_x4 per-step 0.39× 优但 total wall 15.1× regress,`ncfn=100138` Newton control failure 主导 nst 38.8× 增,G0 NO-GO-G0 verdict。
6. PR-X1 (G0-RCA 8-cell tolerance × EpsLin matrix): H4 hypothesis (tolerance mismatch as ncfn root cause) REFUTED,`ncfn` 稳定在 6% 窗内跨 4 orders-of-mag × 10× sweep,tolerance tweaking 完全无 leverage。

**Production CPU baseline preserved**: **P1e StrictOMP RHS** (`ExecPolicy::StrictOMP` + Serial NVec + PREC_NONE SPGMR maxl=5) + **`SHUD_SPGMR_MAXL=30`** small-case Performance opt-in per ADR-0004。零 user-facing 行为变化 from this closure。`SHUD_LINSOL=amg` opt-in 保留在 codebase 作 research knob 但 NOT production 推荐。

**Methodological 贡献**: (i) 6-epic 递进 pattern-only → integrated → RCA sanity matrix 方法学; (ii) H1-H4 hypothesis-driven closure ledger; (iii) ADR chain append-only amendment pattern; (iv) case-asymmetric scaling 三次独立观察; (v) integrated-CVODE-level 反驳 > pattern-only 二分 verdict 的 methodological lesson。这一 corpus 为未来 SHUD-OpenMP 及类似水文数值 project 的 CPU 加速研究提供 auditable negative-result reference。

---

# §9 Future Work / 未来工作

Per ADR-0008 §Forward action:

## §9.1 PR-Y1 — §A5-infra Hydrology-Acceptance Validation Pipeline [HIGH]

Standalone NSE / KGE / peak / timing / runoff / water-balance metric extraction pipeline decoupled from any solver substrate。~2-3 PR sequence,~2-3 weeks budget。See master plan §A5-infra 新 anchor。作 PR-Z1 (§P9) + 任何未来 acceleration claim 的 gating 前置条件。

## §9.2 PR-Z1 — §P9 CVODE Outer-Policy Tuning Spot Check [MEDIUM]

Bounded sweep of `CVodeSetReltol` / `CVodeSetMaxStep` / `CVodeSetMaxNonlinIters` / `CVodeSetNonlinConvCoef` / vector `N_VAbstol` on existing SPGMR PREC_NONE baseline。~1-2 PR sequence,~1-2 weeks budget。Gate: ≥ 1.5× heihe_x4 wall improvement under A5 PASS。Below 1.5× → close §P9 no-action → re-evaluate §P10 design。See master plan §P9 新 anchor。

## §9.3 §P10 — CPU Domain Decomposition (DESIGN-ONLY) [LOW]

Document architectural scoping for subbasin/river-network natural decomposition + interface flux coupling + per-subdomain local solver。NO implementation commitment 直到 PR-Y1 merged + PR-Z1 returns NO-GO/Optional。design-doc scope only,1 PR。See master plan §P10 新 anchor。

## §9.4 §P8-tune.H GPU Sparse Fallback — CLOSED-FINAL (NOT pursued)

Per ADR-0008 §Forward action item 4 + user direction at PR-X1 closure。Re-opening requires fresh ADR-NNNN + GPU hardware audit + cost-benefit analysis vs CPU §P9 + §P10 paths。See master plan §P8-tune.H closure。

## §9.5 §P8-tune.E.small-only KLU env-var mini-prototype — REMAINS OPEN [OPTIONAL/medium]

Not covered by this closure。ADR-0005 §Forward action remains valid — small case (keliya + heihe) KLU env-var mini-prototype for research knob use。See master plan §P8-tune.E.small-only anchor 未变。

---

# §10 References / 参考

## Internal (本仓库)

- `docs/adr/0008-p8-solver-substitution-closure.md` — 本 retrospective 的 consolidating ADR
- `docs/adr/0007-amg-spike-decision.md` — P8-tune.F ADR + Amendment 2026-06-30 (G0 verdict) + Amendment 2026-06-30 (G0-RCA outcome, appended in PR-X2)
- `docs/adr/0005-klu-spike-decision.md` — P8-tune.D KLU spike ADR (Case-aware verdict)
- `docs/adr/0004-maxl-sweep-decision.md` — P8-tune.C SPGMR maxl ADR (Optional-knob;`SHUD_SPGMR_MAXL=30` Performance-tier opt-in)
- `docs/adr/0003-precond-spike-decision.md` — P8-tune.B block-Jacobi precond ADR (PREC_NONE NO-GO)
- `docs/adr/0002-solver-path.md` — P1e Path 1 (Serial NVec + StrictOMP RHS) ADR
- `docs/p8tune/amg_g0_verdict.md` — P8-tune.G0 verdict source-of-truth
- `docs/p8tune/amg_spike_verdict.md` — P8-tune.F verdict source-of-truth
- `docs/p8tune/p8tune_g0_academic_summary.md` — G0 academic summary (methodology 母本)
- `docs/p8tune/p8tune_f_academic_summary.md` — P8-tune.F academic summary
- `docs/p8tune/p8tune_d_academic_summary.md` — P8-tune.D academic summary
- `docs/p1e/p1e_academic_summary.md` — P1e academic summary (template 母本)
- `SHUD_openMP_master_plan.md` §P8-tune.{B,C,D,E,F,G0,G1,G2,H} (all CLOSED-FINAL) + §A5-infra + §P9 + §P10 (new anchors)
- `tools/p8tune.C/` — SPGMR maxl sweep tooling
- `tools/p8tune.D/{dump_adjacency,fd_color_jacobian,klu_analyze_factor}.cpp` — KLU pattern-only spike binaries
- `tools/p8tune.F/{boomeramg_setup_solve.cpp, aggregate_amg_spike.sh}` — BoomerAMG pattern-only spike + aggregator
- `tools/p8tune.G0/{sunlinsol_hypre_wrapper.cpp, aggregate_g0_smoke.sh, spgmr_baseline_walls_g0.h}` — G0 wrapper + aggregator + case-specific baselines
- `.review-evidence/g0-amg-rca-pr-x1/` — PR-X1 RCA evidence (already on main, do not modify)
- `.review-evidence/g0-amg-smoke-array-rerun/` — G0 4-cell integrated smoke evidence
- `.review-evidence/p8tune-amg-pr-b/` + `p8tune-amg-pr-c/` — P8-tune.F 16-cell pattern-only + verdict evidence
- `.review-evidence/p8tune-klu-spike-pr-a/` + `p8tune-klu-spike-pr-b/` — P8-tune.D KLU pattern-only evidence
- `.review-evidence/p8tune-spgmr-maxl-prd-60cell/` — P8-tune.C SPGMR maxl PRD 60-cell evidence

## PR sequence

- PR #313 / #315 / #316 — P1e PR-F/G/H (production baseline establishment)
- PR #369 / #370 / #371 / #372 / #373 / #368 / #376 — P8-tune.C 6-PR SPGMR maxl sweep + G7 amendment
- PR #384 / #385 / #387 / #388 — P8-tune.D 4-PR KLU pattern-only spike
- PR #394 / #402 / #403 / #404 / #412 — P8-tune.F 5-PR BoomerAMG pattern-only spike + capstone-merge
- PR #414 / #415 / #416 / #417 (PR-C amendment) / #418 (PR-D) / #413 (capstone-merge) / #419 (post-merge cleanup) — P8-tune.G0 4-PR integrated smoke
- PR #420 — PR-X1 G0-RCA tolerance × EpsLin matrix (the closure trigger)
- **PR-X2 #<this PR>** — ADR-0008 retrospective closure + master plan §P8 CLOSED-FINAL + 5-step forward plan (本 retrospective)

## Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-rca-pr-x1-runs/` — PR-X1 Slurm 10248 output dir (8 cells)
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0_amg_smoke.10014/` — G0 PR-A Slurm 10014 output (4 cells)
- `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x4/` + `heihe_x16/` — 90-day-truncated AutoSHUD deployment (常驻)

## External / academic

1. Hindmarsh, A. C., Brown, P. N., Grant, K. E., Lee, S. L., Serban, R., Shumaker, D. E., & Woodward, C. S. (2005). SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers. *ACM Transactions on Mathematical Software*, 31(3), 363-396.
2. Hindmarsh, A. C., et al. (2005). Op. cit. (SUNDIALS canonical reference)
3. Henson, V. E., & Yang, U. M. (2002). BoomerAMG: A parallel algebraic multigrid solver and preconditioner. *Applied Numerical Mathematics*, 41(1), 155-177.
4. Davis, T. A. (2006). *Direct Methods for Sparse Linear Systems*. SIAM. (SuiteSparse KLU AMD/COLAMD/BTF)
5. Brown, P. N., & Saad, Y. (1990). Hybrid Krylov methods for nonlinear systems of equations. *SIAM Journal on Scientific and Statistical Computing*, 11(3), 450-481.
6. Eisenstat, S. C., & Walker, H. F. (1996). Choosing the forcing terms in an inexact Newton method. *SIAM Journal on Scientific Computing*, 17(1), 16-32.
7. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. (V-cycle bound + GMRES analysis)
8. George, A., & Liu, J. W. H. (1981). *Computer Solution of Large Sparse Positive Definite Systems*. Prentice-Hall. (2D mesh PDE nested-dissection O(N^1.5) bound)
9. Hindmarsh, A. C., & Serban, R. (2020). User Documentation for CVODE v6.0.0. Lawrence Livermore National Laboratory Technical Report UCRL-SM-208108, current at time of P8 chain.
10. Falgout, R. D., Cleary, A., Jones, J., Chow, E., Henson, V. E., Baldwin, C., Brown, P. N., Vassilevski, P., & Yang, U. M. (2020). Hypre User's Manual (version 2.20.0 series, superseded by 3.1.0 used in P8-tune.F/G0). Center for Applied Scientific Computing, Lawrence Livermore National Laboratory.
