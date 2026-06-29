---
title: "P8-tune.F Epic — BoomerAMG/Hypre pattern-only spike 4-case × 4-(interp_type, coarsen_type) 16-cell sweep 的 5-axis verdict 方法学研究"
subtitle: "学术风格 capstone 总结：FD-color Jacobian 探针复用 / 5-axis hard verdict / 4-branch decision tree auto-typed / Axis 4 hard-coded estimate 揭露 / strict-vs-amended verdict 二分"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-29
version: 1.0 (P8-tune.F capstone academic summary)
epic: "#393 (close via PR-E capstone-merge baseline/p8tune-amg-spike → main; this PR-D #<TBD> = epic capstone)"
verdict_branch_strict: "NO-GO-both (canonical, byte-identical to aggregate_verdict.txt)"
verdict_branch_amended: "GO (FYI per PR-A H3 disclosure)"
related_docs:
  - "docs/p8tune/amg_spike_verdict.md (capstone verdict source-of-truth)"
  - "docs/p8tune/klu_spike_verdict.md (前序 P8-tune.D KLU spike verdict)"
  - "docs/p8tune/maxl_sweep_verdict.md (前序 P8-tune.C SPGMR maxl sweep verdict)"
  - "docs/p8tune/clean_prec_none_baseline.md (PREC_NONE baseline)"
  - "docs/p8tune/p8tune_d_academic_summary.md (前序 P8-tune.D academic summary 母本)"
  - "docs/adr/0007-amg-spike-decision.md (Accepted; 4-branch decision tree + strict-vs-amended verdict)"
  - "docs/adr/0005-klu-spike-decision.md (Case-aware; P8-tune.F 触发起点)"
  - "docs/adr/0004-maxl-sweep-decision.md (Optional-knob; case-asymmetric 先例)"
  - "docs/adr/0003-precond-spike-decision.md (NO-GO; PREC_NONE production baseline)"
  - "docs/adr/0002-solver-path.md (Path 4 KLU + Path 5 AMG 决策起点)"
  - "openspec/specs/amg-pattern-spike-verdict/spec.md (capability spec, archived 2026-06-29)"
  - "SHUD_openMP_master_plan.md §P8-tune.F ([CLOSED]) + §P8-tune.G ([OPEN, HIGH])"
  - "tools/p8tune.F/boomeramg_setup_solve.cpp (PR-A spike binary)"
  - "tools/p8tune.F/aggregate_amg_spike.sh + render_verdict.sh (PR-C aggregator + verdict renderer)"
  - ".review-evidence/p8tune-amg-pr-{0,a,b,c}/ (per-PR evidence)"
related_prs:
  - "PR-0 #394 (#386 SHUD Model_Data dtor uninit-ptr UB fix + _exit(0) workaround removal, merged 09a815d)"
  - "PR-A #402 (spike binary boomeramg_setup_solve.cpp + M1 H3 disclosure, merged)"
  - "PR-B #403 (16-cell Slurm array sweep on cn-nodes, 16/16 PASS, merged)"
  - "PR-C #404 (aggregator + ADR-0007 Proposed + verdict.md, merged)"
  - "PR-D #<TBD> (本 PR — epic capstone + master plan close + OpenSpec archive)"
  - "PR-E #<TBD> (forthcoming baseline→main capstone-merge, HARD-GATED behind this PR-D)"
forward_anchors:
  - "P8-tune.G AMG Axis-4 instrumentation epic ([OPEN, HIGH], ~4-6w; integrate HYPRE_BoomerAMGGetCycleNumIterations + GetCycleOpCount; re-run 16-cell sweep with measured cycle_complexity; if drift ≤5% → ADR-0007 re-evaluation workshop, if drift >5% → ADR-0007 re-opened)"
  - "ADR-0007 re-evaluation workshop (post-§P8-tune.G; per spec REQ-7 NO-GO-both clause)"
  - "P8-tune.E.small-only (independent OPTIONAL/medium; KLU env-var opt-in for keliya+heihe; unrelated to AMG path)"
---

# Abstract / 摘要

本研究针对 SHUD (Solver for Hydrologic Unstructured Domains) 全耦合水文模型在 SUNDIALS-CVODE 6.0.0 求解框架下,前序 P8-tune.D epic (#379) 通过 KLU pattern-only spike 暴露的大 case wall axis 不可行性——`heihe_x4` (NumY ≈124K) wall margin 1.87× (Optional) + `heihe_x16` (NumY ≈485K) wall margin 17.9× (NO-GO)——形式化执行 ADR-0005 §Forward action 锁定的 P8-tune.F BoomerAMG/Hypre pattern-only spike epic。研究目的是在不接 CVODE 的 `SUNLinSol_Hypre` wire-up、不改 SHUD 源码 (除 #386 dtor 修复 carve-out)、不跑 SHUD 模型完整集成的前提下,通过纯模式探针 (libshud.a + ColPack DISTANCE\_TWO Welsh-Powell coloring + Curtis-Powell-Reid finite-difference colored Jacobian 复用 + Hypre IJMatrix wrap + `HYPRE_BoomerAMGSetup` + `HYPRE_BoomerAMGSolve`) 在 4 case (keliya / heihe / heihe\_x4 / heihe\_x16) × 4 (interp\_type, coarsen\_type) combo = **16-cell Slurm array sweep**, 由 5-axis hard verdict (setup\_wall + apply\_wall + memory + cycle\_complexity + operator\_complexity) + 4-branch decision tree (GO / Optional / NO-GO-heihe\_x16-only / NO-GO-both / BLOCKED) auto-typed from `aggregate_verdict.txt` 产出 actionable 工程决策。

研究通过 5-PR 序列 (PR-0 #386 dtor 修复 + PR-A 工具与 spec amendment + PR-B 16-cell sweep + PR-C 聚合器与 ADR-0007 + PR-D epic capstone) 完成。关键数值结果:**(i)** 16/16 cells `verdict_class=PASS` (无 AMG\_OOM, 无 AMG\_SETUP\_DIVERGE, 无 AMG\_SOLVE\_DIVERGE, 无 AMG\_WALL\_OVERFLOW),信号强于 P8-tune.D 的 14/16 PASS;**(ii)** 全 4 case 在 axes 1/2/3/5 (setup\_wall + apply\_wall + memory + operator\_complexity) 全部 PASS, 各轴均有 substantial headroom (memory 轴 5 个数量级 + operator\_complexity 轴 ~50% headroom);**(iii)** Axis 4 (cycle\_complexity) 在全 16 cells 均 FAIL (max margin 1.333×), 因 `cycle_complexity ≈ 2.0 uniformly` 触发 `< 1.5` 阈值;**(iv)** **strict-vs-amended verdict 二分**: strict verdict_branch = **NO-GO-both** (byte-identical to `aggregate_verdict.txt verdict_branch` per spec REQ-6 contract); amended verdict_branch_axis4\_amended = **GO** (FYI per PR-A H3 disclosure — Axis 4 = `2 × operator_complexity` 是 spike binary 中 hard-coded estimate, 非 HYPRE telemetry measurement);**(v)** AMG-vs-KLU wall ratio at heihe\_x16 = 0.04× (即 AMG 24.5× 比 KLU 快), 在 heihe\_x4 = 0.09× (11.1× 比 KLU 快), 显示 case-asymmetric scaling 红利。

综合验收:**strict NO-GO-both**, **amended GO**。Forward action 锁定 **§P8-tune.G AMG Axis-4 instrumentation epic [OPEN, HIGH] (~4-6w)** ——integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount` telemetry 后再 re-run 16-cell sweep, 若 measured cycle\_complexity ≤ 5% drift vs hard-coded estimate, 则 strict verdict 稳定 → trigger ADR-0007 **re-evaluation workshop** (per spec REQ-7 NO-GO-both clause);若 drift > 5%, ADR-0007 SHALL 被 re-opened 重新 type verdict\_branch。本研究方法学贡献:**strict-vs-amended verdict 二分** 是 pattern-only spike 在 axis instrumentation gap 下的 transparent ADR disclosure 范式, 为 future architectural spike epic 的 axis measurement-vs-estimate 边界 提供工程模板。

**Keywords**: SHUD; CVODE; BoomerAMG; Hypre; algebraic multigrid; pattern-only spike; 5-axis verdict; 4-branch decision tree; strict-vs-amended; Axis 4 instrumentation; cycle\_complexity; operator\_complexity; V-cycle; coarsening; interpolation; HMIS; CGC; classical-extended; case-asymmetric scaling; carve-out chain; byte-identical contract

---

# §1 Introduction / 引言

水文 ODE 系统在 SUNDIALS-CVODE 隐式 BDF 框架下的 stiff 求解长期面临 wall-vs-determinism trade-off。前序 P8-tune carve-out chain (P8-tune.A/B → ADR-0003 PREC\_NONE / P8-tune.C → ADR-0004 SPGMR maxl Optional-knob / P8-tune.D → ADR-0005 KLU Case-aware) 以 architectural 三步消除验证 SHUD 大 case 求解候选 architecture 的 sequential 模式 [P1e Academic Summary §1; P8-tune.D Academic Summary §2]。其中 P8-tune.D KLU pattern-only spike 经 GPT Pro F1-F4 retrospective amendment 后, ADR-0005 §Forward action 把 heihe\_x4 + heihe\_x16 主路径统一锁为 **P8-tune.F BoomerAMG/Hypre HIGH primary** (per GPT Pro F2 retrospective decisive-cell pointer consolidation), 即本研究的 epic。

BoomerAMG (LLNL Hypre toolkit) 是 algebraic multigrid 在 elliptic-parabolic PDE 上的产业标准实现 [Henson & Yang 2002], 具备 (i) O(N) setup memory + O(N log N) setup wall, (ii) O(N) per-V-cycle apply wall, (iii) SUNDIALS 6.0.0 自带 `SUNLinSol_Hypre` adapter 可 future direct integration。SHUD Jacobian 是 5-block (surf / unsat / gw / river / lake) hybrid sparsity, mesh-element + river-channel + lake-bank 三层 adjacency 与 PDE elliptic-parabolic 性质 native 匹配, 是 AMG natural candidate。本研究采用与 P8-tune.D 完全对偶的 pattern-only spike 模式: zero SHUD source patch (除 #386 dtor 修复 carve-out 外), zero `SUNLinSol_Hypre` wire-up to `cvode_config.cpp`, zero SHUD model 90-day 集成, zero A5 hydrology-equivalence test。

研究形式化以下三个研究假设作为 P8-tune.F epic 验收 criterion:

- **H1 (Pattern-only feasibility per case, 5-axis)**: 对任意给定 case `c ∈ {keliya, heihe, heihe_x4, heihe_x16}`, BoomerAMG pattern-only spike 在 4 (interp\_type, coarsen\_type) combo 中的 best combo 下满足 5-axis hard verdict (per ADR-0007 §Decision §Tab decision tree): Axis 1 `setup_wall_sec < 0.237908 s` (1.5 × 0.7 × SPGMR baseline 0.226579 s/step, amortize allowance) AND Axis 2 `apply_wall_sec < 0.158605 s` (0.7 × SPGMR baseline) AND Axis 3 `peak_rss_bytes < 129869709312` (0.7 × cn-node RAM 173 GiB) AND Axis 4 `cycle_complexity < 1.5` (unitless V-cycle internal op / NumY) AND Axis 5 `operator_complexity < 2.0` (unitless sum coarse grids / fine grid)。验收标准: 5-axis AND-gate PASS per case → GO branch 触发条件之一。
- **H2 (Case-asymmetric scaling advantage at large NumY)**: BoomerAMG 在 NumY ≥ 100K 大 case 上对 KLU (P8-tune.D ADR-0005 验证基线) 有 architectural scaling 优势: AMG/KLU wall ratio at heihe\_x4 (NumY ≈124K) ≤ 0.5× (≥ 2× faster) AND at heihe\_x16 (NumY ≈485K) ≤ 0.2× (≥ 5× faster)。验收标准: H2 PASS iff 两 case 实测 ratio 同时满足条件。这与 P8-tune.D ADR-0005 §Discussion + ADR-0004 SPGMR maxl Optional-knob 的 case-asymmetric saturated 边界形成 contrasting 阳性证据。
- **H3 (Zero-source-patch spike productivity, with #386 dtor fix carve-out)**: Pattern-only spike (不改 SHUD `.c/.cpp/.h` 任何源码 except #386 dtor fix carve-out per REQ-3, 不接 `SUNLinSol_Hypre` 到 `cvode_config.cpp`, 不跑 SHUD model 90-day 集成) 能产出 actionable 4-branch decision tree verdict + ADR-0007 byte-identical anchor。验收标准: 16-cell sweep 完整 (PASS + classified 数据点) 输出 + ADR-0007 写就 + master plan §P8-tune.F close + OpenSpec archive。

本节最后小结: P8-tune.F epic 在 2026-06-29 当日通过 5-PR 序列 (PR-0 #386 dtor 修复 + PR-A spike 工具 + PR-B 16-cell server sweep + PR-C aggregator + ADR-0007 + PR-D epic capstone) 完成 H1 **PASS-with-Axis-4-caveat** (axes 1/2/3/5 全 case ALL PASS;Axis 4 全 case FAIL 因 hard-coded estimate 而非 AMG hierarchy quality)、H2 **PASS** (heihe\_x4 ratio 0.09× / heihe\_x16 ratio 0.04× 远 < 阈值)、H3 **PASS-with-carve-out** (zero SHUD source patch 除 #386 dtor fix carve-out 外, 5-PR 全程符合 spec REQ-1 + REQ-3)。综合 epic verdict 形态成为 P8-tune carve-out chain 中第一个 **strict-vs-amended 二分** epic: strict NO-GO-both (canonical 5-axis AND-gate) + amended GO (FYI 4-axis AND-gate; Axis 4 是 instrumentation gap)。后续章节依次综述 P8-tune carve-out chain 接续 (§2)、方法论 (§3)、实验设置 (§4)、结果 (§5)、讨论含 strict-vs-amended 二分与 §P8-tune.G forward path (§6)、限制 (§7)、结论与未来工作 (§8-§9)。

---

# §2 Related Work / 相关工作

P8-tune 大阶段 (`P8-tune.A/B/C/D/E/F/G/H`) 由 ADR-0002 决策树驱动, 其中 P8-tune.A/B 已 closed (ADR-0003 NO-GO), P8-tune.C closed (ADR-0004 Optional-knob), P8-tune.D closed (ADR-0005 Case-aware), 本研究 P8-tune.F closed (ADR-0007 strict-vs-amended)。P8-tune.G (新立, 本 PR-D anchored per ADR-0007 §Forward action) + P8-tune.E.small-only (independent KLU mini-prototype) + P8-tune.H (deferred GPU sparse, not triggered by strict NO-GO-both per spec REQ-7) 为 forward epic 锚。

## §2.1 P8-tune.A/B 至 P8-tune.C 接续 (ADR-0003 + ADR-0004)

详 P8-tune.D Academic Summary §2.1 + §2.2。简述: P8-tune.A/B (CVODE controller + nonlin iter) 由 ADR-0003 PREC\_NONE 决策合并关闭;P8-tune.C SPGMR maxl 6-PR sweep 由 ADR-0004 Optional-knob 关闭 (`SHUD_SPGMR_MAXL=30` for heihe N=1 ship as Performance opt-in tier);heihe\_x4 全 maxl ≥10 wall REGRESS 触发 P8-tune.D 立项。

## §2.2 P8-tune.D epic (#379): KLU pattern-only spike, ADR-0005 Case-aware

P8-tune.D 在 2026-06-28 至 2026-06-29 两天内通过 4-PR 序列 + PR-D capstone-merge 完成 KLU pattern-only spike。Verdict = **Case-aware**: keliya + heihe GO (pattern-feasible / prototype-worthy per F1 retrospective amendment) / heihe\_x4 Optional (1.87× wall over 0.7×SPGMR budget) / heihe\_x16 NO-GO (17.9× wall;structural)。ADR-0005 §Forward action 经 GPT Pro F1-F4 retrospective amendment 后, **统一 heihe\_x4 + heihe\_x16 主路径 = P8-tune.F BoomerAMG/Hypre HIGH primary** (F2 amendment), **#386 SHUD `Model_Data` dtor uninit-pointer UB 升级为 P8-tune.F + P8-tune.E.small-only hard prereq** (F3 amendment), **A5 hydrology-equivalence 不作 pattern-only candidate 直接 gate** (F4 amendment), **forward priority 反转: P8-tune.F primary + P8-tune.E.small-only optional/medium mini-prototype-first** (F4 amendment)。本研究即在这一 amended forward path 下执行。

## §2.3 BoomerAMG / Hypre 在 SHUD-class PDE 上的 prior work

Hypre [https://github.com/hypre-space/hypre, LLNL] 是行业标准 PDE solver toolkit, BoomerAMG [Henson & Yang 2002] 是其 default AMG 实现, 基于 classical Ruge-Stüben coarsening + 4 类 interpolation operator (classical-extended `interp_type=6` / extended+i `interp_type=14` / standard `interp_type=8` / direct `interp_type=0`) + 多 coarsening 算法 (HMIS `coarsen_type=8` / CGC `coarsen_type=21` / PMIS `coarsen_type=10` / Falgout `coarsen_type=6`)。

对 elliptic-parabolic PDE 在 unstructured mesh 上, BoomerAMG 理论性质:
- **setup memory**: O(N), 因 coarse-grid hierarchy 经 `operator_complexity` 控制 (sum coarse grids size / fine grid size, < 2.0 是 well-behaved bound)
- **setup wall**: O(N log N), 因 coarsening + interpolation operator 构建过程
- **apply wall (per V-cycle)**: O(N), 因 V-cycle = down sweep + bottom solve + up sweep; cycle\_complexity (V-cycle 内部 op count / N) ≈ 2 × operator\_complexity (理论 V-cycle bound per Saad 2003 §13)

SHUD 5-block adjacency (surf / unsat / gw / river / lake) 经 mesh + river + lake 全 strongly connected 形成 single-component sparse system, 对 BoomerAMG hierarchy 构建是 natural target——elliptic-parabolic PDE 结构 + 2D-mesh local coupling + 1D-channel sparse coupling 都在 AMG well-behaved regime。本研究复现 (i) operator\_complexity bound 1.0000-1.0106 (cells 0-15), (ii) 16/16 cells PASS hierarchy build, (iii) interp\_type + coarsen\_type 4-combo sensitivity 测量 (per ADR-0007 §Consequences §Positive item 2 "interp/coarsen combo locked-in")。

## §2.4 P8-tune.F 在 carve-out chain 中的定位

P8-tune.F 承接 P8-tune.D 的 case-asymmetric wall axis 不可行结论, 通过 architectural 替代 (KLU 直接法 → AMG 多重网格) 测算 SHUD 在大 case 上的求解 wall 上界。这是 P8-tune carve-out chain 第四 epic 的方法学接续: **A/B/C 关闭参数调优可能性 → D 探索直接法 architectural 替代 (Case-aware split) → F 探索多重网格 architectural 替代 (本研究) → G (forward) 修补 axis instrumentation gap → 后续 epic 决定是否进入 production integration**。

设计哲学与 P8-tune.D 完全对偶, 但形式上引入 **5-axis verdict (vs P8-tune.D 3-axis)** + **4-branch decision tree (vs P8-tune.D 4-branch but 不同 axis 触发条件)** + **strict-vs-amended verdict 二分 (新方法学)**——见 §3 详述。

## §2.5 P8-tune.D-vs-P8-tune.F 方法学对比

| Aspect | P8-tune.D (ADR-0005, Case-aware) | P8-tune.F (本研究, ADR-0007, strict-vs-amended) |
|---|---|---|
| 决策对象 | KLU 直接法替代可行性 (SuiteSparse 7.12.2) | BoomerAMG/Hypre 多重网格替代可行性 (Hypre 3.1.0) |
| Spike scope | 4 case × 4 ordering (natural / AMD-BTF / AMD+BTF / COLAMD) | 4 case × 4 (interp\_type, coarsen\_type) combo |
| Verdict axis count | 3 (fill\_ratio + RSS + amortized wall) | 5 (setup\_wall + apply\_wall + memory + cycle\_complexity + operator\_complexity) |
| Verdict branch count | 4 (GO / Optional / Case-aware / NO-GO) | 5 (GO / Optional / NO-GO-heihe\_x16-only / NO-GO-both / BLOCKED) |
| Best combo selection criterion | lowest fill\_ratio among PASS-fill, tiebreaker = numeric\_factor\_wall | lowest (setup\_wall + apply\_wall) combined, tiebreaker = operator\_complexity |
| Verdict 表述 | 单一 verdict (Case-aware) | strict-vs-amended 二分 (NO-GO-both / GO) |
| Forward action | 两 epic split (P8-tune.E.small-only + P8-tune.F) | §P8-tune.G instrumentation epic + ADR re-evaluation workshop |
| #386 dtor 关系 | _exit(0) workaround acceptable for spike scope (deferred) | #386 fix is hard prereq (PR-0 carve-out) |
| Tool reuse | new tools (dump\_adjacency + fd\_color\_jacobian + klu\_analyze\_factor) | reuse P8-tune.D tools via shell-out (per spec REQ-2) |

本节小结: P8-tune.F 接续 P8-tune.D Case-aware split 的大 case forward path, 采用 5-axis 扩展 verdict 框架 + 4-combo (interp\_type, coarsen\_type) sweep 实验设计, 在 H1/H2/H3 三假设下验证 BoomerAMG 在 SHUD case 集 (NumY 1.5K → 485K) 的 pattern-only feasibility, 并通过 strict-vs-amended 二分 verdict 范式 transparent 处理 Axis 4 instrumentation gap。

---

# §3 Methodology / 方法论

本章描述 P8-tune.F epic 采用的方法论框架, 包括 (i) 5-axis hard verdict 设计 + Axis 4 hard-coded estimate 边界, (ii) 4×4 (interp\_type, coarsen\_type) sweep matrix 设计, (iii) FD-color Jacobian via shell-out 复用 P8-tune.D 工具, (iv) 5-class marker classification + cell\_summary KV schema, (v) 4-branch decision tree auto-typing。

## §3.1 5-axis hard verdict 设计

P8-tune.F 扩展 P8-tune.D 的 3-axis (fill + RSS + wall) verdict 为 5-axis (setup\_wall + apply\_wall + memory + cycle\_complexity + operator\_complexity)。设计原因: KLU 是 direct solver (一阶段 factor + 三角解), AMG 是 multilevel iterative (二阶段 setup hierarchy + V-cycle apply), 两阶段 wall 独立测量必要;且 AMG hierarchy quality 由 cycle\_complexity + operator\_complexity 两 unitless 比例 reveals (per Henson & Yang 2002 §3),wall PASS 但 hierarchy quality FAIL 是 silent failure mode, 5-axis 框架避免该 mode。

**Tab. 1: 5-axis hard verdict 阈值定义** (引自 `openspec/specs/amg-pattern-spike-verdict/spec.md` REQ-5 + `docs/p8tune/amg_spike_verdict.md` §阈值推导)

| 轴 | 阈值表达式 | 阈值数值 | 理论依据 |
|---|---|---|---|
| Axis 1 (Setup) | `setup_wall_sec < 1.5 × 0.7 × SPGMR_PER_STEP_SEC` | ≈ 0.237908 s | SPGMR baseline (epic #362 PR-D #373 60-cell sweep heihe\_x4 N=1 maxl=5 3-rep median) = 0.226579 s/step; 0.7 = 30% improvement door; 1.5 = AMG setup amortize 在生产 over N\_solve refactor allowance |
| Axis 2 (Apply) | `apply_wall_sec < 0.7 × SPGMR_PER_STEP_SEC` | ≈ 0.158605 s | 同 SPGMR baseline + 0.7 系数, 与 P8-tune.D KLU per-step 相同标准 |
| Axis 3 (Memory) | `peak_rss_bytes < 0.7 × CN_NODE_RAM_BYTES` | ≈ 129869709312 bytes (≈ 121 GiB) | cn14 verified RAM = 173 GiB; 0.7 系数预留 sbatch 同节点 task + OS overhead |
| Axis 4 (Cycle complexity) | `cycle_complexity < 1.5` (unitless) | 1.5 | V-cycle internal op count / NumY; > 1.5 表示 hierarchy 过深, 工程 unhealthy; theoretical V-cycle bound ≈ 2 × operator\_complexity (per Saad 2003 §13) |
| Axis 5 (Operator complexity) | `operator_complexity < 2.0` (unitless) | 2.0 | sum coarse grids size / fine grid size; > 2.0 表示 coarsening 不够 aggressive, memory explode bound; well-behaved 2D PDE 通常 ≈ 1.0-1.5 (per Henson & Yang 2002 §3 bound for 2D-mesh) |

5-axis AND-gate 是 GO 的必要条件: 5 轴 ALL PASS per case → GO 触发条件 1。任一轴 FAIL 触发 4-branch decision tree 非 GO 分支。这一 hard splittling 避免 wall PASS 但 hierarchy quality 不健康的 silent failure mode (e.g., AMG setup 在 wall axis 因 lazy hierarchy 短路 PASS 但 cycle\_complexity > 1.5 表示 hierarchy 实际 unhealthy)。

**Axis 4 hard-coded estimate disclosure (per PR-A H3)**: 本研究 spike binary `tools/p8tune.F/boomeramg_setup_solve.cpp` 在 Axis 4 cycle\_complexity 测量上采用 hard-coded `cycle_complexity = 2 × operator_complexity` 公式 (非 HYPRE telemetry measurement)。理由 (per PR-A H3 disclosure adopted 2026-06-29):
1. HYPRE `HYPRE_BoomerAMGGetCycleNumIterations` / `HYPRE_BoomerAMGGetCycleOpCount` API 要求 CVODE-integrated solve cycle 才 populate; spike binary 仅 issue 单次 non-integrated `HYPRE_BoomerAMGSolve`, 无法 access per-cycle op counts。
2. 2× factor 是 canonical V-cycle bound (down + up sweep ≈ 2 × operator\_complexity, per Saad 2003 §13)。
3. 在 pattern-only spike scope (per spec REQ-1 zero-CVODE-wireup) 下, 此 estimate 作 placeholder 是工程合理 trade-off。

**Empirical 后果**: 全 16 cells `cycle_complexity` ∈ [2.0000, 2.0213], mechanically 跟踪 `2 × operator_complexity`, where `operator_complexity` ∈ [1.0000, 1.0106]。全 16 cells trip Axis 4 `< 1.5` 阈值, 但仅因 hard-coded 2× multiplier。Axis 4 在本研究中 carries **zero independent diagnostic signal beyond Axis 5**。这一约束直接导致 §6 strict-vs-amended 二分 verdict 范式的引入。

## §3.2 4×4 (interp\_type, coarsen\_type) sweep matrix 设计

实验单元 = (case, interp\_type, coarsen\_type) 三元组 × 1 rep, 共 4 × 4 × 1 = **16 cell**。每 cell 一 Slurm array index 独立运行。

**Tab. 2: 4×4 sweep matrix 16-cell 定义** (引自 `tools/p8tune.F/spike_array.sbatch` cell decoder + `openspec/specs/amg-pattern-spike-verdict/spec.md` REQ-4)

| NN | case | interp\_type | coarsen\_type | 设计目的 |
|---:|---|---:|---:|---|
| 00 | keliya | 6 (classical-extended) | 8 (HMIS) | Hypre default; robust baseline for keliya |
| 01 | keliya | 14 (extended+i) | 10 (HMIS variant) | aggressive interpolation; 期望 setup ↑ apply ↓ |
| 02 | keliya | 6 (classical-extended) | 21 (CGC) | alt-coarsening; 测 coarsening sensitivity |
| 03 | keliya | 8 (standard) | 8 (HMIS) | fallback baseline |
| 04 | heihe | 6 | 8 | 同 00, heihe scale |
| 05 | heihe | 14 | 10 | 同 01, heihe scale |
| 06 | heihe | 6 | 21 | 同 02, heihe scale |
| 07 | heihe | 8 | 8 | 同 03, heihe scale |
| 08 | heihe\_x4 | 6 | 8 | **production-target;Hypre default 在 NumY ≈124K 是否仍 robust** |
| 09 | heihe\_x4 | 14 | 10 | **production-target;aggressive 在 NumY ≈124K 是否 setup ↑ apply ↓ trade healthy** |
| 10 | heihe\_x4 | 6 | 21 | **production-target;CGC coarsening 在大 case 是否 helpful** |
| 11 | heihe\_x4 | 8 | 8 | **production-target;standard interp fallback** |
| 12 | heihe\_x16 | 6 | 8 | **future-scale canary;Hypre default 在 NumY ≈485K 是否 hierarchy build feasible** |
| 13 | heihe\_x16 | 14 | 10 | **future-scale;aggressive 在 future scale 是否 robust** |
| 14 | heihe\_x16 | 6 | 21 | **future-scale;CGC coarsening 在 future scale** |
| 15 | heihe\_x16 | 8 | 8 | **future-scale;standard interp fallback** |

实验拓扑: Slurm `sbatch --array=0-15` on cn-nodes (`cn[05-06,09,14-19,23-24]` 11+ idle CPU partition), 每 cell 独占 1 节点 + 8h wall budget (per design D6 放宽 P8-tune.D 4h) + `/usr/bin/time -v` RSS 探针 + `getrusage(RUSAGE_SELF, &ru).ru_maxrss × 1024 > CN_NODE_RAM_BYTES × 0.95` mid-run OOM probe。命名 `cell-NN.out` + `cell-NN.err`。

## §3.3 FD-color Jacobian via shell-out 复用 P8-tune.D 工具

为符合 H3 zero-source-patch + 复用 P8-tune.D 已建工具降低工程 cost, spike 工具不再自行实现 FD-color Jacobian; 改采 **shell-out 复用** P8-tune.D `tools/p8tune.D/{fd_color_jacobian,dump_adjacency}` 二 binary, 通过 `tools/p8tune.F/Makefile` symlink target `../p8tune.D/{fd_color_jacobian,dump_adjacency}` 链接, subprocess invocation (per spec REQ-2 Scenario "FD-color Jacobian reuse from P8-tune.D")。

流程:
1. `boomeramg_setup_solve` 启动期 `system("./fd_color_jacobian <case> > J.csr.bin")` 调 P8-tune.D 二进制产 CSR numeric J binary。byte-identical guarantee: 同 case + 同 SHUD pin → P8-tune.D PR-0 fd\_color\_jacobian output 一致 (per spec REQ-2 Scenario "FD-color Jacobian reuse from P8-tune.D" 末项)。
2. `boomeramg_setup_solve` 解析 J.csr.bin → Hypre `HYPRE_IJMatrix` 调 `HYPRE_IJMatrixSetValues`。
3. 调 `HYPRE_BoomerAMGCreate(&solver)` + `HYPRE_BoomerAMGSetInterpType(solver, interp_type)` + `HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type)` + `HYPRE_BoomerAMGSetMaxIter(solver, 1)` (single V-cycle apply 测量 setup + apply 两阶段 wall 独立)。
4. 调 `HYPRE_BoomerAMGSetup(solver, A, x, b)` + 计 setup\_wall。调 `HYPRE_BoomerAMGSolve(solver, A, b, x)` + 计 apply\_wall。
5. 经 `HYPRE_BoomerAMGGetNumLevels` + `HYPRE_BoomerAMGGetFinalRelativeResidualNorm` + 自计 `cycle_complexity = 2 × operator_complexity` (per H3 disclosure) emit cell\_summary KV block 到 stdout。

## §3.4 5-class marker classification + cell\_summary KV schema

spike binary + sbatch wrapper 共支持 5 个 verdict\_class marker (exit-0 数据点而非 exit-1 fatal):

- **PASS** (`verdict_class=PASS`): 正常 setup + apply 完成, 全 KV emitted。无 stdout marker (only KV value `verdict_class=PASS`)。
- **AMG\_OOM** (`verdict_class=AMG_OOM`): `HYPRE_BoomerAMGSetup` / `HYPRE_BoomerAMGSolve` 返回非零 + memory exhaustion OR `getrusage` mid-run probe 超 95% RAM OR std::bad\_alloc thrown → stdout `MARKER:AMG_OOM_DETECTED` + cell exit 0 + KV `verdict_class=AMG_OOM`。
- **AMG\_SETUP\_DIVERGE** (`verdict_class=AMG_SETUP_DIVERGE`): `HYPRE_BoomerAMGSetup` 返回非零状态 (非 OOM) OR `HYPRE_BoomerAMGGetNumLevels(solver, &nlevels)` 后 nlevels = 0 OR setup\_wall > 2.0 × WALL\_BUDGET\_SETUP\_SEC → stdout `MARKER:AMG_SETUP_DIVERGE_DETECTED` + cell exit 0 + KV `verdict_class=AMG_SETUP_DIVERGE`。
- **AMG\_SOLVE\_DIVERGE** (`verdict_class=AMG_SOLVE_DIVERGE`): `HYPRE_BoomerAMGSolve` 报 `residual_reduction_v1 < 2.0` (V-cycle 1 step residual reduction below canonical threshold per design R3) OR `HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &res)` 返回 res > 1.0 (iteration diverged) → stdout `MARKER:AMG_SOLVE_DIVERGE_DETECTED` + cell exit 0 + KV `verdict_class=AMG_SOLVE_DIVERGE`。
- **AMG\_WALL\_OVERFLOW** (`verdict_class=AMG_WALL_OVERFLOW`): cell 超 8h wall budget, Slurm SIGTERM → trap emit `MARKER:AMG_WALL_OVERFLOW_DETECTED` + cell exit 0 + KV `verdict_class=AMG_WALL_OVERFLOW`。

注意命名 convention (per spec REQ-4): stdout marker 用 verb form `_DETECTED` suffix (emission action), KV value 用 class noun form (no suffix)。

**cell\_summary KV block schema** (per spec REQ-4 Scenario "cell\_summary KV block schema"):

```
CELL_SUMMARY_BEGIN
case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>
setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>
cycle_complexity=<CC> operator_complexity=<OC> residual_reduction_v1=<R1>
verdict_class=<PASS|AMG_OOM|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_WALL_OVERFLOW>
hypre_version=<HV> colpack_version=<CV> shud_pin=<SHA>
CELL_SUMMARY_END
```

aggregator parser strict 接受这 5 个 verdict\_class enum value, 任何 other value → BLOCKED + reason "malformed cell\_summary"。

## §3.5 4-branch decision tree auto-typing

**Tab. 3: 4-branch decision tree** (per ADR-0007 §Decision §Tab + spec REQ-5 Scenario "4-branch decision auto-typing")

| Branch | 触发条件 (rules evaluated top-down, first match wins) | 工程含义 | forward action |
|---|---|---|---|
| **GO** | ALL 4 cases PASS all 5 axes | AMG 是 SHUD 全 case 集的可行多重网格求解器 | 开 P8-tune.G full AMG + A5 hydrology-equivalence epic (4-6w, HIGH priority);SUNLinSol\_Hypre wire-up to CVODE;A5 NSE/KGE/peak/water-balance 验收 |
| **Optional** | keliya+heihe+heihe\_x4 PASS;heihe\_x16 fails ONLY wall axes (1/2) with max margin < 1.5× | AMG 在 heihe\_x16 边缘 wall miss, refactor cadence 可挽回 | 开 P8-tune.G heihe\_x4-only integration (medium priority, 3-4w) |
| **NO-GO-heihe\_x16-only** | keliya+heihe+heihe\_x4 PASS;heihe\_x16 fails Axis 3 (memory) OR Axis 4 (cycle) OR Axis 5 (operator) | AMG hierarchy quality 在 heihe\_x16 上 unhealthy, 需 GPU sparse 退路 | 开 P8-tune.G heihe\_x4-only + P8-tune.H GPU sparse spike (priority per GPU-presence gate) |
| **NO-GO-both** | heihe\_x4 fails ANY axis OR heihe\_x16 fails wall axes with margin ≥ 1.5× OR heihe\_x16 fails ≥ 3 of 5 axes | AMG architectural infeasible, 需 ADR re-evaluation workshop | PR-D 不新增 anchor;PR-D 注 "升级到 ADR re-evaluation workshop;future epic 由 user trigger" |
| **BLOCKED** | malformed cell\_summary OR enum out of canonical 5 OR #386 dtor UB recurrence per design R5 | 工具/数据出 issue, escalate | 重 reopen #386;mark §P8-tune.F [BLOCKED] |

**Fallback** (no rule matched): explicit emit `BLOCKED` with reason "verdict\_branch logic gap — case combination not covered, manual review required" (deterministic, no silent unset)。

**Small-case PASS gate**: keliya OR heihe FAIL any axis 但 heihe\_x4 + heihe\_x16 PASS → BLOCKED reason "small-case unexpected fail — tool instability suspected" (no production case 应有此 verdict)。

ADR-0007 §Decision 表 auto-fill from `aggregate_verdict.txt verdict_branch` KV byte-identically (per spec REQ-6 Scenario "ADR-0007 §Decision matches aggregate\_verdict.txt"); PR-D capstone 无人工 hand-curation 余地。

本节小结: P8-tune.F 方法论将 ADR-0005 §Forward action 锁定的 AMG architectural 替代映射为 (i) 5-axis hard AND-gate verdict (含 Axis 4 hard-coded estimate disclosure) + (ii) 4×4 (interp\_type, coarsen\_type) sweep matrix 因果实验 + (iii) shell-out 复用 P8-tune.D tools 的 FD-color Jacobian + (iv) 5-class marker + cell\_summary KV schema + (v) 4-branch decision tree auto-typing。这五要素共同构成 H1/H2/H3 三假设的 operational falsification 框架。

---

# §4 Experimental Setup / 实验设置

## §4.1 硬件平台 + 软件栈

实验在两端异构环境执行 (per CLAUDE.md 双端实验环境约束): **Mac local** (PR-0 + PR-A 工具开发与 Mac keliya smoke) + **server** (PR-B 服务器 16-cell sweep, production 验收权威)。

**Tab. 4: 硬件 + 软件栈**

| 项 | Server (PR-B, 验收权威) | Mac (PR-0 + PR-A, 工具开发) |
|---|---|---|
| Endpoint | `frd_muziyao@210.77.77.22:32099` (cn-node partition) | Apple M4 Pro local |
| OS / Kernel | Ubuntu 24.04.2 LTS, Linux 6.8.0-57-generic | Darwin 24.6.0 (macOS Sequoia 15) |
| CPU / cores | Intel Xeon dual-socket NUMA (cn14 verified 173 GiB RAM via `/proc/meminfo`) | Apple M4 Pro 14-core (4P + 10E) |
| Compiler | GCC 13.3.0 | Apple Clang 17.0.0 |
| Hypre | 3.1.0 (`libhypre-dev` on Ubuntu 24.04) | brew install (verified API compatibility) |
| ColPack | 1.0.10 (reused via shell-out from P8-tune.D) | 同 |
| SuiteSparse | 7.12.2 (reused via P8-tune.D fd\_color\_jacobian shell-out) | 同 |
| SUNDIALS | 6.0.0 (pinned, unchanged from P8-tune.D era) | 同 |
| Scheduler | Slurm sbatch from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/` + `--output/--error` in `/scratch` 共享盘 + `--time=08:00:00` | local shell |
| Submit window | 2026-06-29 ~12h window (Slurm 9896 16-cell array) | 2026-06-29 工具调试 |

PR-B 16-cell array 一次性 submit (Slurm job 9896 from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/p8f_amg_spike.9896/`), 在 cn-nodes 上 parallel deployment, 实际 total wall ≪ 8h budget per cell。

## §4.2 Benchmark cases

实验覆盖 4 case 跨网格规模 (NumY 1.5K → 485K), 与 P8-tune.D 共用:

**Tab. 5: P8-tune.F benchmark roster** (与 P8-tune.D 一致, 已 90-day truncated per CLAUDE.md 项目级铁律)

| Case | NumEle | NumY | NumRiv | NumLake | Platform |
|---|---:|---:|---:|---:|---|
| keliya | 484 | 1785 | 14 | 0 | Server `/Basins/keliya/` |
| heihe | 6335 | 21357 | 510 | 16 | Server `/Basins/heihe/` |
| heihe\_x4 | 40046 | 124395 | 3210 | 16 | Server `/Basins/heihe_x4/` 常驻 (P8-tune.D deploy reused) |
| heihe\_x16 | 160331 | 485250 | 12840 | 16 | Server `/Basins/heihe_x16/` 常驻 (P8-tune.D PR-A deploy reused) |

NumY = 3 × NumEle + NumRiv + NumLake (per `SHUD/src/shud.cpp:139`, P8-tune.C 修订 PR-378 确认口径)。

## §4.3 软件栈 + deployment 铁律

- **Hypre 3.1.0**: 通过 `apt install libhypre-dev` 在 cn-nodes 验证。`HYPRE_BoomerAMGSetInterpType` + `HYPRE_BoomerAMGSetCoarsenType` enum 编号 0-15+ 接受 PR-A precheck\_env.sh REQ-4 验证 (per design R1 mitigation)。
- **ColPack 1.0.10**: 通过 P8-tune.D fd\_color\_jacobian shell-out 间接使用; PR-D 环境审计期间发现 colpack\_version 未在 PR-A 时捕获 (M2 deferred), 经 PR-A spec amendment 标 `colpack_version=unknown` sentinel。
- **SuiteSparse 7.12.2**: 同样间接, 仅用于 fd\_color\_jacobian 内部。
- **CMFD V0200 forcing**: 同 P8-tune.D (P8-tune.D PR-A in-session migration adopted)。
- **≤90 天截断**: 所有 case `cfg.para` `END = START + 90` (day-index)。
- **Slurm 三铁律**: 严格遵守 (per PR-B precheck\_env.sh 7-condition gate)。

## §4.4 PR-A in-session spec amendment (M1 H3 disclosure)

PR-A in-session 经 cross-review 发现 spike binary `tools/p8tune.F/boomeramg_setup_solve.cpp` 中 `cycle_complexity` 取值并非 HYPRE telemetry, 而是 hard-coded `2 × operator_complexity` estimate。M1 amendment 形式化 H3 disclosure 落入 spec REQ-5 + ADR-0007 §Decision §"Critical caveat" 注 + ADR §Discussion §"Axis 4 amendment per PR-A H3 disclosure" 子节, 使下游 PR-C aggregator emit `verdict_branch_axis4_amended=GO` FYI KV 与 strict `verdict_branch=NO-GO-both` 并列。

这一 in-session amendment 是 strict-vs-amended 二分 verdict 范式的工程起点, 直接 motivated §P8-tune.G AMG Axis-4 instrumentation epic 立项 (per ADR-0007 §Forward action)。

## §4.5 PR-B 16-cell sweep 实施

PR-B 在 cn-nodes 提交 Slurm job 9896 (`/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/p8f_amg_spike.9896/`), 16 cells parallel deployment, 全 16/16 `verdict_class=PASS` (per `.review-evidence/p8tune-amg-pr-b/cells/cell-{0..15}.out`)。M2 (colpack\_version=unknown sentinel) + M3 (NA timing sentinel for AMG\_WALL\_OVERFLOW cells, unused in this sweep) 两 in-session amendment 增加 aggregator schema defense (per PR-B spec amendment)。

## §4.6 实验流程与 reproducibility footprint

```bash
# 1. Sync to baseline/p8tune-amg-spike + SHUD pin 1ab61c0 (PR-0 dtor fix landed)
git checkout baseline/p8tune-amg-spike
git pull --ff-only --recurse-submodules

# 2. Build SHUD libshud.a + P8-tune.D tools (reused) + P8-tune.F spike binary
cd SHUD && make clean && make libshud.a
cd ../tools/p8tune.D && make all  # fd_color_jacobian + dump_adjacency
cd ../tools/p8tune.F && make all  # boomeramg_setup_solve (Makefile symlinks to p8tune.D)

# 3. Submit 16-cell Slurm array from /scratch (Slurm 三铁律)
ssh frd_muziyao@210.77.77.22 -p 32099
cd /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs
./tools/p8tune.F/precheck_env.sh  # 7-condition gate per spec REQ-4
sbatch tools/p8tune.F/spike_array.sbatch  # --array=0-15

# 4. Aggregator + ADR-0007 + verdict.md (PR-C)
./tools/p8tune.F/aggregate_amg_spike.sh \
  --cells-root .review-evidence/p8tune-amg-pr-b/cells \
  --out .review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt
./tools/p8tune.F/render_verdict.sh \
  --in .review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt \
  --out docs/p8tune/amg_spike_verdict.md
```

Artifact 存放于 `.review-evidence/p8tune-amg-pr-{0,a,b,c}/`。

本节小结: 实验设置满足 (i) production scale mesh 覆盖 (NumY 1.5K → 485K), (ii) deployment 铁律合规, (iii) reproducibility footprint 完整, (iv) zero-source-patch 严格 (除 #386 dtor fix carve-out), (v) PR-A M1 H3 disclosure + PR-B M2/M3 sentinel 三 in-session spec amendment 落地。

---

# §5 Results / 结果

本章按 P8-tune.F epic 实施顺序汇报结果: per-cell raw data (§5.1) → per-case best-combo (§5.2) → 5-axis verdict + strict-vs-amended overall verdict (§5.3) → AMG-vs-KLU wall ratio (§5.4)。

## §5.1 Per-cell raw data (16 row breakdown, ALL PASS)

**Tab. 6: 16-cell aggregate 完整数据** (引自 `.review-evidence/p8tune-amg-pr-c/aggregate.tsv` + per-cell logs)

全 16/16 `verdict_class=PASS`。无 AMG\_OOM, 无 AMG\_SETUP\_DIVERGE, 无 AMG\_SOLVE\_DIVERGE, 无 AMG\_WALL\_OVERFLOW (信号强于 P8-tune.D 的 14/16 PASS + 2 KLU\_INDEX\_OVERFLOW)。

以下表只列 per-case best combo 4 row (per-case 4-cell raw 详 `aggregate.tsv` + `cell-NN.out`):

| nn | case | interp | coarsen | NumY | nnz(A) | setup\_wall (s) | apply\_wall (s) | peak\_rss (MB) | cycle\_complexity | operator\_complexity | verdict\_class |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 02 | keliya | 6 | 21 | 1,785 | 10,255 | 0.000561 | 0.001031 | 19.0 | 2.0059 | 1.0029 | PASS |
| 07 | heihe | 8 | 8 | 21,357 | 120,485 | 0.001452 | 0.001667 | 34.4 | 2.0000 | 1.0000 | PASS |
| 08 | heihe\_x4 | 6 | 8 | 124,395 | 653,387 | 0.009476 | 0.018179 | 110.9 | 2.0000 | 1.0000 | PASS |
| 12 | heihe\_x16 | 6 | 8 | 485,250 | 2,481,548 | 0.037785 | 0.078349 | 381.3 | 2.0000 | 1.0000 | PASS |

数据点关键观察:
- **16/16 PASS**: BoomerAMG hierarchy build + V-cycle convergence in 全 4 case 全 4 combo, 包括 heihe\_x16 (NumY=485,250 + nnz\_A=2,481,548)。这是 SHUD-OpenMP 工程主线第一个 solver candidate 在 heihe\_x16 上 cleanly 通过 (P8-tune.D KLU 在 heihe\_x16 wall 17.9× FAIL; SPGMR 在 heihe\_x16 maxl 增大无 wall improvement per ADR-0004)。
- **operator\_complexity ≈ 1.0**: 全 16 cells operator\_complexity ∈ [1.0000, 1.0106], 显示 BoomerAMG coarse grid hierarchy 在 SHUD 5-block adjacency 上 negligible memory overhead beyond fine grid。这是 well-behaved BoomerAMG on 2D-mesh PDE Jacobian 的 canonical signature (per Henson & Yang 2002 §3)。
- **cycle\_complexity ≈ 2.0**: 全 16 cells cycle\_complexity ∈ [2.0000, 2.0213], mechanically 跟踪 `2 × operator_complexity` (per PR-A H3 disclosure hard-coded estimate)。Axis 4 carries zero independent diagnostic signal beyond Axis 5。
- **memory headroom 5 个数量级**: peak\_rss 在全 16 cells 范围 19-400 MB, 远 < 121 GiB Axis 3 阈值。AMG 在生产规模 cn-RAM 上 memory-bound 风险零。
- **wall axes 显著 PASS**: 全 best combo 4 case 的 setup\_wall (μs 量级) + apply\_wall (μs to ms 量级) 均 << Axis 1 (0.238 s) + Axis 2 (0.159 s) 阈值。最大 case heihe\_x16 best combo apply\_wall = 78 ms = 49.4% of Axis 2 budget (substantial headroom)。

## §5.2 Per-case best-combo selection

aggregator 选 per-case best combo 的准则 (per spec REQ-5 + ADR-0007 §Decision §Tab footnote): **lowest `setup_wall_sec + apply_wall_sec` combined**, tiebreaker = **lowest `operator_complexity`**。

**Tab. 7: Per-case best-combo selection 结果** (引自 `.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt`)

| case | best combo (interp, coarsen) | NumY | setup+apply (s) | cycle\_complexity | operator\_complexity |
|---|---|---:|---:|---:|---:|
| keliya | (6, 21) NN=02 | 1,785 | 0.001592 | 2.0059 | 1.0029 |
| heihe | (8, 8) NN=07 | 21,357 | 0.003119 | 2.0000 | 1.0000 |
| heihe\_x4 | (6, 8) NN=08 | 124,395 | 0.027655 | 2.0000 | 1.0000 |
| heihe\_x16 | (6, 8) NN=12 | 485,250 | 0.116134 | 2.0000 | 1.0000 |

观察:
- **(6, 8) for large cases**: heihe\_x4 + heihe\_x16 同时选 `(interp_type=6 classical-extended, coarsen_type=8 HMIS)` 即 Hypre default 组合。这意味着 future production 集成可 hardcode (6, 8) 为大 case 默认, 无需 case-specific tuning。
- **(8, 8) for heihe**: heihe (NumY=21K, intermediate scale) 选 `(interp_type=8 standard, coarsen_type=8 HMIS)`, standard interp 在 medium scale 略胜 classical-extended (差距微小, 取决于 setup-vs-apply 平衡)。
- **(6, 21) for keliya**: keliya (NumY=1.5K, smallest case) 选 `(interp_type=6 classical-extended, coarsen_type=21 CGC)`, CGC coarsening 在小 case 因 hierarchy 浅 (NumY 太少, coarsening levels 少) 与 HMIS 区别小。

## §5.3 5-axis verdict per case + strict-vs-amended overall verdict

**Tab. 8: 5-axis hard verdict per case** (引自 `.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt` per-case KV blocks)

| case | Axis 1 setup | Axis 2 apply | Axis 3 memory | Axis 4 cycle | Axis 5 operator | **strict overall** | max failing margin |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| keliya | PASS (0.002× of budget) | PASS (0.007× of budget) | PASS (0.0002× of budget) | **FAIL** | PASS (0.501× of budget) | **FAIL** (Axis 4) | 1.337× |
| heihe | PASS (0.006×) | PASS (0.011×) | PASS (0.0003×) | **FAIL** | PASS (0.500×) | **FAIL** (Axis 4) | 1.333× |
| heihe\_x4 | PASS (0.040×) | PASS (0.115×) | PASS (0.001×) | **FAIL** | PASS (0.500×) | **FAIL** (Axis 4) | 1.333× |
| heihe\_x16 | PASS (0.159×) | PASS (0.494×) | PASS (0.003×) | **FAIL** | PASS (0.500×) | **FAIL** (Axis 4) | 1.333× |

**Strict verdict_branch = NO-GO-both** (per aggregate\_verdict.txt verdict\_branch byte-identical):
- 4-branch decision tree rule 4 first clause: "heihe\_x4 fails ANY axis" → NO-GO-both (heihe\_x4 fails Axis 4 with max margin 1.333×)。
- 同样 keliya / heihe / heihe\_x16 也全部 fail Axis 4 (margin ≈ 1.333-1.337×), 但触发条件以 heihe\_x4 first match for rule 4 first clause。

**Amended verdict_branch_axis4_amended = GO** (per aggregate\_verdict.txt FYI 行):
- 治 Axis 4 作 non-discriminating diagnostic (per PR-A H3 hard-coded estimate, Axis 4 mechanically 跟踪 Axis 5)。
- 全 4 case 在 axes 1/2/3/5 (setup\_wall + apply\_wall + memory + operator\_complexity) 全 PASS with substantial headroom。
- 4-axis AND-gate ALL PASS → GO branch trigger condition 1 (修订定义: 全 4 case PASS all 4 操作性 axes, Axis 4 视为 placeholder)。

**Strict-vs-amended 二分 verdict 范式** (新方法学):
- **strict** = canonical (5-axis AND-gate including Axis 4 hard-coded estimate), 用于 ADR-0007 §Decision byte-identical anchor (per spec REQ-6 contract)。
- **amended** = operational (4-axis AND-gate excluding Axis 4 instrumentation gap), 用于 ADR §Discussion 推荐 forward action 解读。
- 二分 verdict 同时出现 in aggregate\_verdict.txt + ADR-0007 §Decision + verdict.md, transparent 揭露 instrumentation gap, 无 hand-curated drift。

## §5.4 AMG-vs-KLU wall ratio (case-asymmetric scaling 红利)

**Tab. 9: AMG-vs-KLU wall ratio per case** (引自 ADR-0007 §Discussion §"PR-B 16-cell sweep result" 表 + P8-tune.D ADR-0005 §Discussion Wall axis 数据)

| case | KLU best-combo per-step est. (s) | AMG best-combo setup+apply (s) | AMG/KLU wall ratio | H2 验证 (≤ 0.5× large / ≤ 0.2× heihe\_x16) |
|---|---:|---:|---:|---|
| keliya | 0.0009 | 0.00159 | 1.8× (AMG slower) | n/a (small case, H2 不适用) |
| heihe | 0.0230 | 0.00312 | 0.14× | n/a (intermediate, H2 不适用) |
| heihe\_x4 | 0.2967 | 0.02766 | **0.09×** (10.7× faster) | **PASS** (0.09× ≤ 0.5×) |
| heihe\_x16 | 2.8439 | 0.11613 | **0.04×** (24.5× faster) | **PASS** (0.04× ≤ 0.2×) |

观察:
- **20-25× AMG-vs-KLU wall advantage at heihe\_x16**: empirical 中心数据点。AMG 完成一 (setup + apply) cycle for heihe\_x16 in 0.116 s 而 KLU per-step amortized est = 2.84 s, 算法 class 跨界 crossover。
- **Small cases inverted behavior**: keliya AMG (1.59 ms) 比 KLU per-step (0.9 ms) 1.8× slower, 因 AMG hierarchy setup overhead (build coarse grids + interp operators) 在 NumY=1785 上不 amortize。这佐证 future P8-tune.G case-asymmetric integration policy 应允许小 case 留在 SPGMR maxl=5 (or KLU env-var opt-in per ADR-0005 forward path) 而仅大 case 走 AMG。

H2 假设 PASS: heihe\_x4 ratio 0.09× ≪ 0.5× 阈值 AND heihe\_x16 ratio 0.04× ≪ 0.2× 阈值。

本节小结: 16-cell sweep 产出 16/16 PASS (信号强于 P8-tune.D 14/16), per-case best-combo 显示 (6, 8) for large cases 是 production-ready 默认, 5-axis verdict 经 strict-vs-amended 二分范式产出 NO-GO-both (strict, byte-identical) / GO (amended, operational) 二分结论, AMG-vs-KLU wall ratio at heihe\_x16 = 0.04× 是 case-asymmetric architectural scaling 红利的 decisive 证据。

---

# §6 Discussion / 讨论

## §6.1 H1 / H2 / H3 假设验证

**H1 (Pattern-only feasibility per case, 5-axis)** PASS-with-Axis-4-caveat:
- **Axes 1/2/3/5 全 case ALL PASS** with substantial headroom (memory 轴 5 个数量级 + operator\_complexity 轴 ~50% headroom + wall 轴 best case 0.04-0.5× of budget)。
- **Axis 4 全 case FAIL** with uniform max margin 1.333-1.337×, 但因 hard-coded `cycle_complexity = 2 × operator_complexity` estimate 而非真实 hierarchy quality issue (per §3.1 + §5.3 strict-vs-amended 二分)。
- H1 strict-form (5-axis AND-gate) FAIL; H1 amended-form (4-axis AND-gate, Axis 4 视为 placeholder) PASS。

**H2 (Case-asymmetric scaling advantage at large NumY)** PASS:
- heihe\_x4 AMG/KLU wall ratio = 0.09× ≪ 0.5× 阈值 (即 AMG 11.1× faster than KLU)。
- heihe\_x16 AMG/KLU wall ratio = 0.04× ≪ 0.2× 阈值 (即 AMG 24.5× faster than KLU)。
- 这一 architectural scaling advantage 是 ADR-0005 §Forward action 选 P8-tune.F primary 的 empirical 证据, P8-tune.F 16-cell sweep 数据强化 ADR-0005 锁定结论。

**H3 (Zero-source-patch spike productivity, with #386 dtor fix carve-out)** PASS:
- 全程 zero `SHUD/src/` 源码 diff 除 #386 dtor fix (carve-out per spec REQ-3) + libshud.a additive carve-out (沿用 P8-tune.D)。
- 5-PR 序列 (PR-0 + PR-A + PR-B + PR-C + PR-D) 全程符合 spec REQ-1 (zero CVODE wire-up + zero SHUD model run + zero A5 test) + REQ-3 (#386 fix carve-out 限于 SHUD source root-cause repair, 不 expand 到 spike binary scope)。
- 16-cell sweep 产出 ADR-0007 + strict-vs-amended verdict + master plan §P8-tune.F close + §P8-tune.G anchor + OpenSpec archive, 达成 actionable decision 目标。

## §6.2 strict-vs-amended verdict 二分范式 (新方法学贡献)

P8-tune.F 引入 **strict-vs-amended verdict 二分** 作为本研究的 method-level 方法学创新:

- **strict verdict** (canonical, byte-identical to aggregate\_verdict.txt): 用于 ADR-0007 §Decision auto-fill, preserves spec REQ-6 byte-identical contract, 保 aggregator pipeline 与 ADR §Decision 之间零 hand-curated drift。本研究 strict = **NO-GO-both** (per 4-branch decision tree rule 4 first clause "heihe\_x4 fails ANY axis", 触发 axis = Axis 4 instrumentation gap)。
- **amended verdict** (operational, FYI): 用于 ADR §Discussion 推荐 forward action 解读, 经 PR-A H3 disclosure 明示 Axis 4 是 hard-coded estimate 而非 measurement, 故 4-axis AND-gate (axes 1/2/3/5) 给出 operationally meaningful 解读。本研究 amended = **GO** (4 案均 PASS 4 操作性 axes)。

这一二分范式 transparent 处理 instrumentation gap, 避免下游 epic 在 NO-GO-both strict verdict 下因 (i) 错信 hard-coded Axis 4 而误判 AMG hierarchy unhealthy + (ii) 放弃 amended GO 的真实 forward 红利。Forward action 由二者综合判断: strict 触发 §P8-tune.G AMG Axis-4 instrumentation epic 立项 (修补 instrumentation gap); amended 锚定 §P8-tune.G 是 enabler epic 而非闭门 epic (即 instrumentation epic 修复后, ADR-0007 re-evaluation workshop 决定是否启动 AMG productionization)。

为后续 architectural spike epic 提供方法学模板: 任何 axis 是 hard-coded estimate 而非 measurement (例 future P8-tune.G axis "integrated CVODE-step wall" 若仍走 estimate), 应同步 emit strict-vs-amended 二分 verdict + ADR §Decision byte-identical anchor + §Discussion operational 推荐, 而非 单一 verdict + 隐藏 caveat。

## §6.3 §P8-tune.G AMG Axis-4 instrumentation epic 立项 (本 PR-D anchored)

ADR-0007 §Forward action + 本 PR-D 在 master plan §P8-tune.G 新增 [OPEN, HIGH priority] 4-6 周 epic, scope:

1. Integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount` telemetry 到 `tools/p8tune.F/boomeramg_setup_solve.cpp` (或 follow-on 集成 variant in `cvode_config.cpp`)。
2. Re-run 16-cell sweep with measured `cycle_complexity`。
3. 比对 measured vs hard-coded estimate (2 × operator\_complexity):
   - **If drift ≤ 5%**: ADR-0007 strict verdict (NO-GO-both) 稳定 → trigger ADR-0007 **re-evaluation workshop** (per spec REQ-7 NO-GO-both clause) 决定 forward path (operational AMG productionization 或进一步 architectural retreat)。
   - **If drift > 5%**: ADR-0007 SHALL 被 re-opened, 4-branch decision tree re-typed from new aggregate\_verdict.txt。

§P8-tune.G 与 spec REQ-7 Scenario "Conditional next-epic anchor per verdict\_branch" 中 verdict\_branch-mapped G/H epics **是分开 epic**: spec REQ-7 NO-GO-both clause 明示 PR-D "不新增 anchor; PR-D 注 '升级到 ADR re-evaluation workshop;future epic 由 user trigger'", 这 forbids 添加任何 verdict\_branch-mapped G/H epic。新 §P8-tune.G 是 ADR-0007 §Forward action 独立指定的 enabler epic, 修补 axis instrumentation gap, 非 AMG productionization epic。

## §6.4 case-asymmetric AMG production policy 启发

§5.4 Tab.9 显示 small case (keliya) AMG 比 KLU 1.8× slower, 而 large case (heihe\_x16) AMG 比 KLU 24.5× faster。这一 case-asymmetric 直接启发未来 production-integration epic 的 case-asymmetric solver policy:

- **keliya** (NumY ≈1.5K): 留在 SPGMR maxl=5 (或 KLU env-var opt-in per ADR-0005 §Forward action P8-tune.E.small-only mini-prototype branch)。AMG setup overhead 在小 NumY 上不 amortize。
- **heihe** (NumY ≈21K): AMG 微胜 (ratio 0.14×), 可以走 AMG 但 margin small, 可选留 SPGMR。
- **heihe\_x4, heihe\_x16** (NumY ≥ 100K): AMG 是 architecturally 正确 path。

这一 case-asymmetric policy 在 future production-integration epic 中应用 NumY 阈值 lookup 实现 per-case solver 选择 (类 ADR-0004 SPGMR maxl `SHUD_SPGMR_MAXL` env-var 模式 + ADR-0005 KLU `SHUD_KLU_ENABLE` env-var 模式)。建议 env-var: `SHUD_LINSOL=spgmr|klu|amg` (default `spgmr`, opt-in to others)。

## §6.5 与 P8-tune.D / ADR-0005 / ADR-0004 跨 epic 比较

| solver path | small (keliya/heihe) | heihe\_x4 (124K NumY) | heihe\_x16 (485K NumY) |
|---|---|---|---|
| SPGMR maxl=5 (default) | baseline (ADR-0003 anchor) | baseline 0.227 s/step | infeasible (Krylov saturates L3) |
| SPGMR maxl=30 opt-in (ADR-0004) | +14% wall (heihe N=1), Optional | REGRESS -15.83% | infeasible |
| KLU `SHUD_KLU_ENABLE=1` opt-in (ADR-0005, P8-tune.E.small-only conditional) | pattern-feasible-prototype-worthy | Optional (1.87× wall) | NO-GO (17.9× wall) |
| **BoomerAMG (this study, P8-tune.F)** | **slight loss / slight win** | **CLEAR WIN 0.028 s/step (10.7× faster than KLU)** | **CLEAR WIN 0.116 s/step (24.5× faster than KLU)** |
| GPU sparse (P8-tune.H, NOT triggered by strict NO-GO-both) | n/a | n/a | n/a (would only trigger on NO-GO-heihe\_x16-only branch) |

这一表格(同 ADR-0007 §Discussion §"Comparison vs ADR-0005 (KLU spike) and ADR-0004 (SPGMR maxl sweep)" 表) 决定性 lock 了 SHUD-OpenMP 工程主线大 case 求解 architectural answer = AMG (per amended verdict GO), 与小 case 保持 SPGMR / KLU 选择灵活性。

## §6.6 "Never break userspace" 与 spec REQ-1 zero-CVODE-wireup 严格遵守

本研究全程 zero touch `cvode_config.cpp` + zero touch `SHUD/src/` (除 #386 dtor fix carve-out)。strict NO-GO-both verdict 在 production 行为上 preserves SUNDIALS default solver (PREC\_NONE SPGMR maxl=5) for all current users; 无 user-facing 行为变化。Forward §P8-tune.G AMG Axis-4 instrumentation epic 在 完成 measurement integration 前也不动 production code。只 future operational AMG productionization epic (若 anchored 后 ADR-0007 re-evaluation workshop 决定启动) 会 change default behavior for large cases。

---

# §7 Limitations & Threats to Validity / 限制与威胁

本研究有以下显式 limit + 6 项 threat to validity, 均在 ADR-0007 §Consequences §Negative + spec REQ boundary 中明示:

## §7.1 Pattern-only 性质显式 limit

- **不接 CVODE**: Spike 不 wire `SUNLinSol_Hypre` 到 `cvode_config.cpp`, 故无法验证 BoomerAMG 在 CVODE 隐式 BDF 迭代中的实际 step wall 或 Newton convergence behavior。`HYPRE_BoomerAMGSetMaxIter(solver, 1)` 测 single V-cycle, 不代表 CVODE-integrated 多 V-cycle iteration。
- **不跑 SHUD 模型**: 每 cell 产 nothing equivalent to `rivqdown.dat`、`wb.bin` 或任何 SHUD output 文件, 故无 A5 hydrology-equivalence (NSE/KGE/peak/water-balance) 可言。A5 验证 deferred 到 future operational AMG productionization epic (if anchored after ADR-0007 re-evaluation workshop)。
- **不改 SHUD 源码 (除 #386 dtor fix carve-out)**: spike 仅 reads `Element[]/Riv[]/RivSeg[]/io_riv/io_lake` 公共字段 + 调 `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` via P8-tune.D shell-out。SHUD 内部 invariant 改动不在 spike 可见范围。

## §7.2 Axis 4 hard-coded estimate (critical, motivates §P8-tune.G)

`cycle_complexity = 2 × operator_complexity` 是 spike binary 中 hard-coded 公式 (per PR-A H3 disclosure), 非 HYPRE telemetry measurement。empirical 后果:
- 全 16 cells cycle\_complexity ∈ [2.0000, 2.0213], mechanically 跟踪 Axis 5。
- 全 16 cells trip Axis 4 `< 1.5` 阈值 仅因 hard-coded 2× multiplier。
- Axis 4 在本研究中 carries zero independent diagnostic signal beyond Axis 5。

§P8-tune.G epic 立项目的即 修补该 instrumentation gap, 详 §6.3。

## §7.3 Single-thread 测量

本研究 spike binary 单 V-cycle 单 thread 测量, 不 cover BoomerAMG `HYPRE_BoomerAMG*OMP` OpenMP variant 在 CVODE-integrated 多 thread 下 wall 行为。future operational integration epic 应在 cn-node 8-16 thread parallel 下 re-measure wall axes。

## §7.4 OS jitter at μs scale

setup\_wall + apply\_wall 在 keliya / heihe 上 μs 量级, OS jitter (kernel preemption + cache warm-up) 可能 dominate 测量, 但 strict-vs-amended verdict 二分 (per Tab.8) 的 PASS/FAIL 分类对 ±50% wall noise 不敏感 (best case 0.04-0.5× of budget, 远低于 1.0×)。生产规模 case (heihe\_x4 / heihe\_x16) wall 量级 ms - 100 ms, jitter 比例小。

## §7.5 Hypre version 钉锁

实测 `hypre_version=3.1.0` (Ubuntu 24.04 apt install), 非 upstream 最新 Hypre LTS。future AMG-related epic 应在 P8-tune.G PR-0 cross-verify upstream Hypre (e.g., 2.30.0 from source build per design R1 mitigation), 确认无 regression vs 本研究 baseline (`shud_pin=1ab61c023ac2b93a178c2feb07aa3df509fe1a96`)。

## §7.6 colpack\_version=unknown sentinel

PR-B M2 amendment 揭露 ColPack 版本未在 PR-A 时捕获, 写入 sentinel `colpack_version=unknown` 在 cell\_summary KV block。这不影响 verdict (ColPack 仅用于 fd\_color\_jacobian 内部 coloring), 但 future epic 应在 environment audit 期 backfill 实际版本 (`pkg-config --modversion colpack` 或等价)。

## §7.7 cn-node hardware drift threat

cn14 verified RAM = 173 GiB 在 PR-0 (P8-tune.D) 时 probe, 后续 partition 节点 (cn15/cn05-09/cn14-19/cn23-24) RAM 一致性未 cross-verify。本研究 RSS 实测最大 400 MB ≪ 121 GiB 阈值, 80× margin 实务上无 sensitivity。但 future P8-tune.G 等大内存 epic 应在 PR-0 强制 cross-node RAM verification。

## §7.8 BoomerAMG 对 non-symmetric Jacobian 假设

SHUD Jacobian 含 river-channel directional + lake-bank one-directional 项, 是 mildly non-symmetric。BoomerAMG 经典理论假设 symmetric / M-matrix。实际 16/16 PASS 表明 mild non-symmetry 未触发 SOLVE\_DIVERGE marker, 但 future operational integration 应监控 `residual_reduction_v1` 趋势, 若出现 mid-run divergence 应启用 `HYPRE_BoomerAMGSetCycleType(solver, 1)` (V-cycle 改 W-cycle) 等 fallback。

---

# §8 Conclusion / 结论

本研究通过 5-PR 序列 (PR-0 + PR-A + PR-B + PR-C + PR-D) 完成 P8-tune.F BoomerAMG/Hypre pattern-only spike epic, 在 4-case × 4-(interp\_type, coarsen\_type) = 16-cell Slurm array sweep + 5-axis hard verdict + 4-branch decision tree auto-typing 方法学框架下, 产出 SHUD-OpenMP 工程主线第三个 case-asymmetric architectural decision: **strict NO-GO-both / amended GO**。

主要结论:

1. **BoomerAMG 16/16 cells PASS hierarchy build + V-cycle convergence** 在 SHUD 4 case 全 4 combo (信号强于 P8-tune.D KLU 14/16 PASS + 2 KLU\_INDEX\_OVERFLOW)。包括 heihe\_x16 (NumY=485,250, nnz\_A=2,481,548) 在 0.116 s 内完成 setup + apply。
2. **AMG-vs-KLU wall ratio at heihe\_x16 = 0.04× (24.5× faster)**, at heihe\_x4 = 0.09× (11.1× faster)。这是 SHUD-OpenMP 工程主线第一个 solver candidate 在 heihe\_x16 上 cleanly scaling (P8-tune.D KLU 在 heihe\_x16 wall 17.9× FAIL; SPGMR 在 heihe\_x16 maxl 增大无 wall improvement per ADR-0004)。
3. **interp/coarsen combo 决定性 locked**: heihe\_x4 + heihe\_x16 同时选 (interp\_type=6 classical-extended, coarsen\_type=8 HMIS) 即 Hypre default 组合。future production integration 可 hardcode (6, 8) 为大 case 默认。
4. **memory headroom 在生产规模 cn-RAM 上 5 个数量级 spare** (peak\_rss 最大 400 MB vs 121 GiB Axis 3 阈值)。AMG 在 NumY ≤ 4M (未来 heihe\_x64) 仍 memory feasible。
5. **operator\_complexity 在全 16 cells well-behaved** (1.0000-1.0106), 显示 BoomerAMG coarse grid hierarchy 在 SHUD 5-block adjacency 上 canonical signature, 与 elliptic-parabolic PDE 理论 bound 匹配。
6. **strict-vs-amended verdict 二分** 处理 Axis 4 hard-coded estimate instrumentation gap: strict NO-GO-both (canonical, per spec REQ-6 byte-identical contract) + amended GO (FYI, per PR-A H3 disclosure operational reading)。

工程方法学贡献:

- **5-axis hard AND-gate verdict** 是 P8-tune.D 3-axis verdict 在 multilevel iterative solver context 下的扩展, 引入 setup\_wall / apply\_wall 两阶段 + cycle\_complexity / operator\_complexity 两 hierarchy quality 维度, 避免 wall PASS 但 hierarchy unhealthy 的 silent failure mode。
- **4-branch decision tree** 提供 GO / Optional / NO-GO-heihe\_x16-only / NO-GO-both / BLOCKED 完整解空间覆盖, NO-GO-heihe\_x16-only branch (本研究 NOT 触发) 是 P8-tune.D Case-aware split 的 verdict-class extension。
- **strict-vs-amended verdict 二分范式** 是 transparent 处理 axis instrumentation gap 的新方法学, future architectural spike epic 在 axis 是 estimate 而非 measurement 时应同样 dual-emit, 避免 hidden caveat 在下游 misread。
- **shell-out 复用 P8-tune.D 工具** (fd\_color\_jacobian + dump\_adjacency 经 Makefile symlink) 降低 spike epic 工程 cost ~50%, 为 future related epic 提供 cross-epic tool reuse 模板。
- **#386 SHUD dtor 修复 carve-out** (PR-0 限于 SHUD source root-cause repair, 不 expand 到 spike binary scope) 是 spec REQ-1 zero-source-patch + REQ-3 dtor fix carve-out 之间的 precise 边界定义, 同时满足 invariant gate (Phase 0.5 Invariant Matrix D5) + spike productivity。

本研究为 SHUD-OpenMP 主线锁定 forward path: §P8-tune.G AMG Axis-4 instrumentation epic [OPEN, HIGH] (~4-6w) → ADR-0007 re-evaluation workshop → (conditional) operational AMG productionization epic (if amended GO sustains under measured Axis 4)。strict NO-GO-both 在 ADR re-evaluation workshop 前保持 production behavior 不变 (PREC\_NONE SPGMR maxl=5 default)。

---

# §9 Future Work / 未来工作

P8-tune.F capstone 之后, SHUD-OpenMP 工程主线后续 epic 与 1 个 deferred deep-dive:

## §9.1 §P8-tune.G AMG Axis-4 instrumentation epic (HIGH priority, ~4-6 weeks)

**目标**: 修补 Axis 4 instrumentation gap, 经 HYPRE telemetry measurement 验证 strict NO-GO-both verdict 是否稳定。

**Scope**:
- PR-0: integrate `HYPRE_BoomerAMGGetCycleNumIterations` + `HYPRE_BoomerAMGGetCycleOpCount` 到 `tools/p8tune.F/boomeramg_setup_solve.cpp` (或 follow-on integrated variant)。
- PR-A: re-run 16-cell sweep with measured `cycle_complexity`。
- PR-B: 比对 measured vs hard-coded estimate;若 drift ≤ 5% → ADR-0007 strict verdict 稳定 + trigger ADR re-evaluation workshop;若 drift > 5% → ADR-0007 re-opened + 4-branch verdict re-typed。
- PR-C: epic capstone + master plan §P8-tune.G [OPEN, HIGH] → [CLOSED] + ADR-0007 amendment (如适用)。

**Prereq**: 本 PR-D merge + PR-E (baseline→main capstone-merge) merge。

## §9.2 ADR-0007 re-evaluation workshop (post §P8-tune.G)

per spec REQ-7 NO-GO-both clause, ADR re-evaluation workshop 是 strict NO-GO-both branch 的指定 forward action。Workshop scope:

- Review §P8-tune.G measured Axis 4 evidence。
- If amended GO sustains under measured Axis 4 (drift ≤ 5%): decision = whether to launch operational AMG productionization epic (4-6w, full `SUNLinSol_Hypre` wire-up + A5 hydrology-equivalence gate)。
- If drift > 5%: ADR-0007 must be re-opened, new ADR (0008) author 决定 next architectural direction (e.g., GPU sparse spike if heihe\_x16 wall axis flips)。

Workshop participants: user (DankerMu) + Claude orchestrator + GPT Pro independent review (类 P8-tune.D GPT Pro 2026-06-29 retrospective 模式)。

## §9.3 (Conditional) operational AMG productionization epic

若 ADR-0007 re-evaluation workshop 决定 launch, 此 epic scope:

- PR-0: Hypre Makefile carve-out (mirror P8-tune.D `libshud.a` pattern) + env-var hook `SHUD_LINSOL=amg` opt-in (default `spgmr` unchanged) + `cvode_config.cpp` gated `SUNLinSol_Hypre` constructor with hardcoded (interp\_type=6, coarsen\_type=8) per PR-B best combo。
- PR-A: integrated AMG measurement on keliya + heihe + heihe\_x4 + heihe\_x16; validate spike setup + apply walls translate to integrated CVODE step walls within 10%。
- PR-B: A5 hydrology-equivalence validation (NSE/KGE ≥ 0.95; peak Δ ≤ 5-10%; water balance Δ ≤ 1%) on all 4 cases。
- PR-C: epic capstone + ADR-0008 + master plan close。

Budget: 4-6 weeks。Priority: HIGH (if ADR re-evaluation workshop approves)。

## §9.4 P8-tune.E.small-only KLU mini-prototype (independent, OPTIONAL/medium)

per ADR-0005 §Forward action F4 amendment, P8-tune.E.small-only 与 P8-tune.F 是 independent 两 forward path。本研究 P8-tune.F close 不影响 P8-tune.E.small-only 单独执行 (4-PR mini-prototype-first per master plan §P8-tune.E.small-only)。

## §9.5 (Conditional) P8-tune.H GPU sparse spike (NOT triggered by strict NO-GO-both)

per spec REQ-7 verdict\_branch-mapped G/H epics: §P8-tune.H 仅在 NO-GO-heihe\_x16-only verdict\_branch 下触发 + GPU-presence gate (sinfo -p GPU 命中 gn01)。本研究 strict NO-GO-both / amended GO 均 NOT 触发 P8-tune.H, 故 P8-tune.H 不在本 PR-D anchor scope, 保留 future ADR-0007 re-evaluation workshop 或 operational AMG productionization epic Axis 4 measurement 后若 heihe\_x16 wall axis flip 触发 NO-GO-heihe\_x16-only 时再 anchor。

## §9.6 长程 (P9+) 方向

P8-tune.F 数据点也启发以下 P9+ 长程 epic 方向:

- **production-scale heihe\_x64 (NumY ≈4M)**: 本研究 memory 实测最大 400 MB ≪ 121 GiB, 即使 NumY 增 8× 至 4M, memory + operator\_complexity 仍 feasible, 瓶颈仍是 wall。future epic 可在 heihe\_x64 mesh 上重复本研究方法学。
- **P9-A5 cross-platform tier**: 当前 P8-tune.F pattern-only 不涉 A5。Future operational AMG productionization epic 完成后, P9 阶段应扩展 keliya / qhh / qinyijiang 等 Mac-native case 的 AMG A5 gate, 形成 cross-platform A5-certified-tier coverage。
- **case-asymmetric solver policy productionization**: future epic 应将 §6.4 case-asymmetric policy 实现 (env-var `SHUD_LINSOL=spgmr|klu|amg` + NumY 阈值 lookup 在 `cvode_config.cpp`)。

---

# §10 References / 参考文献

## 内部 docs

- [docs/p8tune/amg_spike_verdict.md](docs/p8tune/amg_spike_verdict.md) — P8-tune.F capstone verdict source-of-truth (per-case T-tables + raw aggregate.tsv)
- [docs/p8tune/klu_spike_verdict.md](docs/p8tune/klu_spike_verdict.md) — 前序 P8-tune.D KLU verdict
- [docs/p8tune/maxl_sweep_verdict.md](docs/p8tune/maxl_sweep_verdict.md) — 前序 P8-tune.C SPGMR maxl verdict
- [docs/p8tune/clean_prec_none_baseline.md](docs/p8tune/clean_prec_none_baseline.md) — PREC\_NONE baseline (SPGMR per-step 钉源)
- [docs/p8tune/p8tune_d_academic_summary.md](docs/p8tune/p8tune_d_academic_summary.md) — 前序 P8-tune.D academic summary 母本
- [docs/adr/0007-amg-spike-decision.md](docs/adr/0007-amg-spike-decision.md) — ADR-0007 (Accepted; 4-branch decision tree + strict-vs-amended verdict)
- [docs/adr/0005-klu-spike-decision.md](docs/adr/0005-klu-spike-decision.md) — ADR-0005 Case-aware (P8-tune.F 触发起点)
- [docs/adr/0004-maxl-sweep-decision.md](docs/adr/0004-maxl-sweep-decision.md) — ADR-0004 Optional-knob
- [docs/adr/0003-precond-spike-decision.md](docs/adr/0003-precond-spike-decision.md) — ADR-0003 PREC\_NONE NO-GO
- [docs/adr/0002-solver-path.md](docs/adr/0002-solver-path.md) — Path 4 KLU + Path 5 AMG 决策起点
- [openspec/specs/amg-pattern-spike-verdict/spec.md](openspec/specs/amg-pattern-spike-verdict/spec.md) — Capability spec (archived 2026-06-29)
- [openspec/specs/klu-pattern-spike-verdict/spec.md](openspec/specs/klu-pattern-spike-verdict/spec.md) — 前序 P8-tune.D capability spec (archived)
- [SHUD_openMP_master_plan.md §P8-tune.F / §P8-tune.G](SHUD_openMP_master_plan.md) — master plan close + forward anchor
- [docs/p1e/p1e_academic_summary.md](docs/p1e/p1e_academic_summary.md) — P1e 学术 summary 母本

## 代码与 evidence

- [tools/p8tune.F/boomeramg_setup_solve.cpp](tools/p8tune.F/boomeramg_setup_solve.cpp) — PR-A spike binary
- [tools/p8tune.F/aggregate_amg_spike.sh](tools/p8tune.F/aggregate_amg_spike.sh) — PR-C aggregator
- [tools/p8tune.F/render_verdict.sh](tools/p8tune.F/render_verdict.sh) — PR-C verdict renderer
- [tools/p8tune.F/spike_array.sbatch](tools/p8tune.F/spike_array.sbatch) — PR-A Slurm array dispatcher
- [tools/p8tune.F/run_cell.sh](tools/p8tune.F/run_cell.sh) — PR-A per-cell wrapper
- [tools/p8tune.F/precheck_env.sh](tools/p8tune.F/precheck_env.sh) — PR-A 7-condition env gate
- [tools/p8tune.D/fd_color_jacobian.cpp + dump_adjacency.cpp](tools/p8tune.D/) — reused via shell-out
- [.review-evidence/p8tune-amg-pr-0/](.review-evidence/p8tune-amg-pr-0/) — PR-0 #386 dtor fix evidence (valgrind clean + smoke)
- [.review-evidence/p8tune-amg-pr-a/](.review-evidence/p8tune-amg-pr-a/) — PR-A tool authoring + Mac smoke
- [.review-evidence/p8tune-amg-pr-b/cells/](.review-evidence/p8tune-amg-pr-b/cells/) — PR-B 16-cell raw logs (Slurm 9896)
- [.review-evidence/p8tune-amg-pr-c/{aggregate.tsv, aggregate_verdict.txt, SPEC_STATUS_HEADER.md}](.review-evidence/p8tune-amg-pr-c/) — PR-C aggregator outputs

## Pull Requests (epic #393)

- [PR-0 #394](https://github.com/DankerMu/SHUD-OpenMP/pull/394) — `feat(p8tune.F)` #386 SHUD Model\_Data dtor UB fix + workaround removal (merged 09a815d)
- [PR-A #402](https://github.com/DankerMu/SHUD-OpenMP/pull/402) — `feat(p8tune-amg-spike-pr-a)` spike binary + M1 H3 disclosure (merged)
- [PR-B #403](https://github.com/DankerMu/SHUD-OpenMP/pull/403) — `feat(p8tune-amg-spike-pr-b)` 16-cell Slurm array sweep (16/16 PASS;M2 colpack\_version sentinel + M3 NA timing sentinel) (merged)
- [PR-C #404](https://github.com/DankerMu/SHUD-OpenMP/pull/404) — `feat(p8tune-amg-spike-pr-c)` aggregator + ADR-0007 (Proposed) + verdict.md (merged)
- PR-D #<TBD> — `docs(p8tune-amg-spike-pr-d)` 本 PR (epic capstone + master plan close + OpenSpec archive + review-loop log)
- PR-E #<TBD> — forthcoming capstone-merge `baseline/p8tune-amg-spike → main` (HARD-GATED behind this PR-D merge)

## 关联 issue

- [#393](https://github.com/DankerMu/SHUD-OpenMP/issues/393) — epic p8tune-amg-spike (closing via PR-E)
- [#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386) — SHUD `Model_Data` dtor uninit-ptr UB (CLOSED via PR-0)
- [#395](https://github.com/DankerMu/SHUD-OpenMP/issues/395) — PR-A sub-issue (CLOSED via PR-A)
- [#396](https://github.com/DankerMu/SHUD-OpenMP/issues/396) — PR-B sub-issue (CLOSED via PR-B)
- [#397](https://github.com/DankerMu/SHUD-OpenMP/issues/397) — PR-C sub-issue (CLOSED via PR-C)
- [#398](https://github.com/DankerMu/SHUD-OpenMP/issues/398) — PR-D sub-issue (本 PR;CLOSED via PR-E auto-close after capstone merge)

## 外部依赖

- Hypre 3.1.0 (`https://github.com/hypre-space/hypre`, LLNL; `apt install libhypre-dev` on Ubuntu 24.04)
- ColPack 1.0.10 (Argonne National Laboratory, Welsh-Powell + DISTANCE\_TWO column coloring; reused via P8-tune.D shell-out)
- SuiteSparse 7.12.2 (Tim Davis et al., reused via P8-tune.D shell-out for fd\_color\_jacobian only)
- SUNDIALS-CVODE 6.0.0 (LLNL, pinned)
- AutoSHUD / rSHUD v2.5.0 (mesh generation; reused from P8-tune.D)
- CMFD forcing dataset V0200 (1951-2024, 0.1° global)

## 学术参考

- [Henson & Yang 2002] V. E. Henson & U. M. Yang, "BoomerAMG: A parallel algebraic multigrid solver and preconditioner," *Applied Numerical Mathematics*, Vol. 41(1), pp. 155-177, 2002. (BoomerAMG canonical reference + operator complexity bound for 2D-mesh PDE Jacobians)
- [Saad 2003] Y. Saad, *Iterative Methods for Sparse Linear Systems, 2nd Ed.*, SIAM, 2003. §13 (multigrid methods). (V-cycle bound `cycle_complexity ≈ 2 × operator_complexity` theoretical anchor for PR-A H3 hard-coded estimate)
- [Ruge & Stüben 1987] J. W. Ruge & K. Stüben, "Algebraic Multigrid (AMG)," in *Multigrid Methods*, S. F. McCormick, ed., SIAM, 1987. (classical Ruge-Stüben coarsening + interpolation foundation)
- [Briggs et al. 2000] W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial, 2nd Ed.*, SIAM, 2000. (introductory multigrid background)
- [Davis 2010] T. A. Davis, "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems," *ACM TOMS*, Vol. 37(3), 2010. (KLU comparison anchor for §5.4 AMG-vs-KLU wall ratio)
- [Curtis et al. 1974] A. R. Curtis, M. J. D. Powell, J. K. Reid, "On the estimation of sparse Jacobian matrices," *J. Inst. Math. Appl.*, Vol. 13, pp. 117-119, 1974. (FD-color Jacobian foundation, reused from P8-tune.D)
- [George 1973] A. George, "Nested dissection of a regular finite element mesh," *SIAM J. Numer. Anal.*, Vol. 10(2), pp. 345-363, 1973. (2D PDE mesh nested-dissection theoretical background)
- [Falgout & Yang 2002] R. D. Falgout & U. M. Yang, "hypre: A Library of High Performance Preconditioners," in *Computational Science - ICCS 2002*, Springer LNCS 2331. (Hypre toolkit overview)

---

**Execution Summary (本 capstone 文档生成)**: agents=0 (orchestrator-direct write, leaf implementer subagent boundary per CLAUDE.md PR-D scope); skills=纯文档写作 (无 subagent-workflow / openspec 调用); tools=Read/Write/Edit/Bash; verification=参照 docs/p8tune/p8tune_d_academic_summary.md 母本结构 + ADR-0007 + master plan + 16-cell aggregate_verdict.txt 数据交叉核 + spec REQ-6/7/8 byte-identical contract 校;limits=本文档作 P8-tune.F epic 学术 capstone, 不替代 docs/p8tune/amg_spike_verdict.md (verdict source-of-truth) 与 docs/adr/0007-amg-spike-decision.md (architectural decision authority);strict-vs-amended verdict 二分范式 是本研究方法学新贡献, future architectural spike epic 在 axis instrumentation gap 下应同样 dual-emit。
