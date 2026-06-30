---
title: "P8-tune.G0 Epic — Integrated AMG-as-CVODE-Preconditioner Smoke Test with Real Hypre Telemetry: A NO-GO Verdict on SHUD Hydrology Matrices"
subtitle: "学术风格 capstone 总结:6-gate verdict 框架 + 4-cell × 90-day SHORT smoke + dlopen 版本化 soname + ring-buffer telemetry drain hook + strict-vs-amended ADR-0007 集成端验证"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-30
version: 1.0 (P8-tune.G0 capstone academic summary)
epic: "#408 (P8-tune.G0 instrumented AMG smoke), closing via PR-D capstone"
verdict_branch: "NO-GO-G0 (canonical, byte-identical to aggregate_g0_smoke.sh stdout per spec REQ)"
related_docs:
  - "docs/p8tune/amg_g0_verdict.md (G0 capstone verdict source-of-truth)"
  - "docs/p8tune/amg_spike_verdict.md (前序 P8-tune.F 16-cell pattern spike verdict)"
  - "docs/p8tune/p8tune_f_academic_summary.md (前序 P8-tune.F academic summary 母本)"
  - "docs/p8tune/p8tune_d_academic_summary.md (前序 P8-tune.D KLU spike academic summary)"
  - "docs/adr/0007-amg-spike-decision.md (Accepted; G0 Amendment 2026-06-30 appended)"
  - "docs/adr/0005-klu-spike-decision.md (Case-aware; P8-tune.F 触发起点)"
  - "docs/adr/0004-maxl-sweep-decision.md (Optional-knob; SPGMR baseline anchor)"
  - "docs/adr/0003-precond-spike-decision.md (PREC_NONE production baseline)"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/proposal.md"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/design.md"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/tasks.md"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/shud-linsol-selector/spec.md"
  - "openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/sunlinsol-hypre-wrapper/spec.md"
  - "SHUD_openMP_master_plan.md §P8-tune.G0 ([CLOSED]) + §P8-tune.G1/§G2 ([CLOSED-DEFERRED])"
  - "tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp (PR-0 wrapper + PR-B Phase 6 drain hook)"
  - "tools/p8tune.G0/aggregate_g0_smoke.sh (PR-B aggregator)"
  - "tools/p8tune.G0/spgmr_baseline_walls_g0.h (PR-A case-specific baselines + PR-B hot-patched)"
  - ".review-evidence/g0-amg-smoke-array-rerun/ (PR-B evidence: 4 cells × 90-day SHORT)"
related_prs:
  - "PR-0 #414 (SUNLinSol_Hypre wrapper + Hypre link + linsol selector + Mac G0-1 baseline)"
  - "PR-A #415 (4-cell AMG integrated smoke + Slurm sbatch + dlopen versioned-soname)"
  - "PR-B #416 (HYPRE Axis-4 telemetry + aggregator + G0-3/4/5/6 verdict markers + Phase 6 drain hook + SHUD 188854b hot-fix)"
  - "PR-C #<this PR> (G0 NO-GO verdict + ADR-0007 amendment + master plan §G0 [CLOSED] + academic summary)"
  - "PR-D #<TBD> (forthcoming capstone-merge baseline/p8tune-amg-g0-spike → main, HARD-GATED behind PR-C)"
forward_anchors:
  - "P8-tune.H GPU sparse / domain decomposition spike (NEXT investigation path; per ADR-0007 §Forward action L25 escape hatch; GPU-presence gate on gn01 partition is precondition)"
  - "P8-tune.G1 18-cell integrated benchmark ([CLOSED-DEFERRED] per G0 NO-GO; re-activate iff P8-tune.H opens an alternative AMG path)"
  - "P8-tune.G2 A5 hydrology equivalence ([CLOSED-DEFERRED] per G0 NO-GO; gated on G1 re-activation)"
  - "ADR-0007 strict-vs-amended decision PRESERVED (no re-litigation); G0 amendment adds integrated-CVODE evidence to the existing pattern-only Accepted decision"
---

# Abstract / 摘要

本研究形式化执行 ADR-0007 §Forward action Amendment 2026-06-29 锁定的 §P8-tune.G0 epic,即 SHUD-OpenMP 改造工程主线第一段 AMG integrated-CVODE-smoke verdict gate。研究背景:前序 P8-tune.F 16-cell pattern-only spike (zero-CVODE-wireup, zero-SHUD-run) 产出 strict NO-GO-both / amended GO 二分 verdict,基于硬编码 `cycle_complexity = 2 × operator_complexity` Axis-4 estimate;ADR-0007 §Discussion 揭示 Axis 4 在 hard-coded 形态下与 Axis 5 机械同构,carries zero independent diagnostic signal;Saad 2003 §13 V-cycle 理论 bound (`cycle_complexity ≈ 2 × operator_complexity` 是 V-cycle 稳态期望) 进一步显示 `<1.5` 阈值预设了 Krylov acceleration,对纯 V-cycle 不适用。

研究目的是通过 (i) 在 `cvode_config.cpp` wire `SUNLinSol_Hypre` 接 BoomerAMG hierarchy + (ii) 90-day SHORT 4-cell smoke array (`keliya` / `xinanjiang_upstream` / `heihe_x4` / `heihe_x16`) + (iii) 真实 HYPRE telemetry (`HYPRE_BoomerAMGGetCumNnzAP` + `HYPRE_BoomerAMGGetOperatorComplexity` + `HYPRE_BoomerAMGGetNumIterations`) + (iv) 6-gate hard verdict 框架 (G0-1 default-compat / G0-2 build / G0-3 telemetry-real / G0-4 integrated-completes / G0-5 wall-signal / G0-6 solver-stats) ,验证 AMG-as-preconditioner 在集成 CVODE 层是否具备生产可行性。研究形式化 3 个研究假设 H1 (default-compat preserved) / H2 (telemetry-real signal at scale) / H3 (integrated-AMG wall-beneficial-vs-SPGMR at heihe_x4 ∨ heihe_x16) 作为 G0 verdict gate。

关键数值结果:**(i)** 6 gates 中 G0-1/G0-2/G0-3/G0-5/G0-6 PASS,但 **G0-4 FAIL** (heihe_x16 MALFORMED — Slurm SIGKILL 在 8h 预算前触发 cell_summary 已 emit 路径);**(ii)** keliya 集成 telemetry-derived `operator_complexity=1.002945` (Hypre native),与 P8-tune.F 16-cell sweep 1.0029 实测值仅 0.0% drift,验证 G0-3 contract;**(iii)** heihe_x4 AMG per-step wall `0.092873 s` 比 SPGMR baseline `0.238369 s` 快 0.39× (G0-5 per-step PASS),但 **total wall 15.1× WORSE** 因 CVODE nst 增长 38.8× (254756 AMG nst vs 6572 SPGMR nst, `ncfn=100138` Newton 控制失败主导);**(iv)** AMG-preconditioned Newton 迭代显示 `ncfl=0` (内核 Krylov 收敛健康) 但 `ncfn=100138` (外层 Newton 控制率失败),证实 V-cycle per-cycle 速度优势在集成 CVODE 中被 outer-loop 控制失败抹掉;**(v)** g0_verdict_branch = **NO-GO-G0**, 即 ADR-0007 strict NO-GO-both verdict 在 integrated-CVODE smoke 层得到 first-order 经验确认,amended GO operational 解读被集成评估反驳。

综合验收:**G0_OVERALL=NO-GO**, AMG-production path closed at G0 gate per ADR-0007 §Decision + G0 Amendment 2026-06-30。Master plan §P8-tune.G0 [OPEN, HIGH] → [CLOSED];§P8-tune.G1 + §G2 → [CLOSED-DEFERRED, pending P8-tune.H GPU sparse fallback evaluation]。Forward action 锁定 P8-tune.H GPU sparse / domain decomposition spike 作为下一阶段 architectural fallback。研究方法学贡献:**6-gate verdict 框架 + integrated-CVODE-smoke + 真实 HYPRE telemetry + ring-buffer drain hook** 是 pattern-only spike 推进到 integrated-prototype gate 的标准模板,为未来 architectural-substrate gates 提供 pattern-to-integration 转化的工程模板。

**Keywords**: SHUD; CVODE; BoomerAMG; Hypre; SUNLinSol_Hypre; integrated AMG; 6-gate verdict; default-compat; telemetry-real; dlopen versioned-soname; ring-buffer drain hook; CVODE control failure; ncfn; Newton iteration; case-asymmetric scaling; G0 NO-GO; ADR-0007 amendment; pattern-to-integration

---

# §1 Introduction / 引言

水文 ODE 系统在 SUNDIALS-CVODE 6.0.0 隐式 BDF/Newton 框架下的 stiff 求解长期面临三类 architectural 替代:(i) Krylov 子空间法 + 简单预处理 (SPGMR + PREC_NONE) — ADR-0003 / ADR-0004 carve-out chain 锁定的当前 production default;(ii) 直接稀疏求解 (KLU / MUMPS) — ADR-0005 P8-tune.D 验证 keliya/heihe 可行 但 heihe_x4 wall margin 1.87× / heihe_x16 wall 17.9× 不可行;(iii) 多重网格法 (BoomerAMG / Hypre) — ADR-0007 P8-tune.F pattern-only spike 在 16-cell sweep 验证 hierarchy build + V-cycle apply 在 NumY 1.5K-485K 全 PASS, 但 Axis-4 cycle_complexity 用硬编码估值,strict-vs-amended 二分 verdict 留下集成端 verifiability gap。

P8-tune.G0 epic 即填这一 gap,把 P8-tune.F 的 pattern-only assertions 推进到 integrated-CVODE smoke 验证。这是 ADR-0007 §Forward action Amendment 2026-06-29 框架下三段 gate G0 → G1 → G2 序列的 first gate,验证条件:(a) `SHUD_LINSOL=amg` env-var hook 不破坏 `SHUD_LINSOL` unset/`spgmr` default-compat;(b) `make shud_omp HYPRE=1` 在 Mac/Linux/server 三端 build green;(c) Hypre native telemetry (非 hardcoded estimate) 产出真实 hierarchy quality 数据;(d) 4-cell smoke array 在 90-day SHORT 上完成无 crash;(e) AMG per-step wall 在至少一个大 case 上比 SPGMR baseline 改善;(f) BDF/Newton 内 SPGMR 与 AMG 的 interaction 通过 CVODE solver stats (`nfe/nli/nfeLS/ncfn/ncfl/netf`) 完整记录。

形式化研究假设作为 G0 verdict 的可证伪 criteria:

- **H1 (default-compat preserved)**: `SHUD_LINSOL` unset 和 `SHUD_LINSOL=spgmr` 都产 bit-identical 输出 vs pre-G0 SPGMR baseline。验收标准:G0-1 gate PASS (per spec REQ "G0 verdict evaluates six PASS/FAIL gates" Scenario "All six gates PASS produces GO-G0")。
- **H2 (telemetry-real signal at scale)**: 至少一个 AMG_OK cell 的 stdout 含 `MARKER:AMG_TELEMETRY_REAL` AND `cycle_complexity` 来自 Hypre native API (NOT hardcoded `2 × operator_complexity`)。验收标准:G0-3 gate PASS。
- **H3 (integrated-AMG wall-beneficial at scale)**: 至少一个 {`heihe_x4`, `heihe_x16`} 的 `amg_wall_per_step < spgmr_wall_per_step`,其中 SPGMR baseline 为 case-specific (`SPGMR_PER_STEP_HEIHE_X4_S` / `SPGMR_PER_STEP_HEIHE_X16_S`,非 60-cell 全局常量)。验收标准:G0-5 gate PASS。

附加 4 个支撑 gates (G0-2 build / G0-4 integrated-completes / G0-6 solver-stats) 是 6-gate AND-gate 的 operational completeness criteria, 不直接对应 H1/H2/H3 但同样需要 PASS 才能 g0_verdict_branch=GO-G0。

本节小结:P8-tune.G0 epic 把 P8-tune.F 的 pattern-only 二分 verdict 推进到 integrated-CVODE 验证,形式化三个研究假设 H1/H2/H3 + 三个支撑 gates G0-2/G0-4/G0-6, 通过 4-cell 90-day SHORT smoke array + real HYPRE telemetry 产出 first-order 集成端 verdict。后续章节依次综述 P8-tune carve-out chain 接续 (§2)、6-gate verdict 框架与方法论 (§3)、4-cell smoke + Slurm 实验设置 (§4)、verdict 结果 (§5)、讨论含 H1/H2/H3 验证 + 与 P8-tune.F 对比 + ncfn 主导的 outer-Newton 失效模式 (§6)、限制 (§7)、结论 NO-GO-G0 (§8)、未来工作 P8-tune.H (§9)、参考 (§10)。

---

# §2 Related Work / 相关工作

## §2.1 P8-tune 大阶段 carve-out chain

P8-tune 大阶段由 ADR-0002 决策树驱动:A/B (CVODE controller + nonlin iter) closed via ADR-0003 PREC_NONE NO-GO;C SPGMR maxl 6-PR sweep closed via ADR-0004 Optional-knob (`SHUD_SPGMR_MAXL=30` Performance-tier opt-in);D KLU pattern-only spike closed via ADR-0005 Case-aware (keliya/heihe pattern-feasible / heihe_x4 Optional / heihe_x16 NO-GO);F BoomerAMG pattern-only spike closed via ADR-0007 strict NO-GO-both / amended GO 二分 verdict。**本研究 P8-tune.G0** 是 ADR-0007 §Forward action Amendment 2026-06-29 锁定的三段 integrated-AMG gate 序列 G0 → G1 → G2 的第一段 (与 P8-tune.E.small-only KLU mini-prototype 是 independent forward paths)。

## §2.2 ADR-0007 strict-vs-amended verdict 二分及其 integrated-端 verifiability

ADR-0007 §Decision 锁定 strict `verdict_branch=NO-GO-both` (5-axis AND-gate including Axis 4 hard-coded estimate, byte-identical to aggregator output) + amended `verdict_branch_axis4_amended=GO` (FYI 4-axis AND-gate excluding Axis 4 instrumentation gap)。ADR §Discussion §"Axis 4 amendment per PR-A H3 disclosure" 揭示:

1. `cycle_complexity = 2 × operator_complexity` 是 spike binary 硬编码估值 (`tools/p8tune.F/boomeramg_setup_solve.cpp`),非 HYPRE telemetry measurement。
2. 全 16 cells `cycle_complexity ∈ [2.0000, 2.0213]` mechanically 跟踪 Axis 5 `operator_complexity ∈ [1.0000, 1.0106]`,carries zero independent diagnostic signal。
3. Axis 4 阈值 `<1.5` 与 Saad 2003 §13 V-cycle bound (`cycle_complexity ≈ 2 × operator_complexity` 稳态期望) 不一致 — `<1.5` 预设 Krylov acceleration,对纯 V-cycle 不适用。

ADR-0007 §Forward action Amendment 2026-06-29 (post-Linus-review framing amendment) 将 Axis 4 从 hard blocker 降为 hierarchy-quality diagnostic,并把单一 §P8-tune.G AMG instrumentation epic 改 split 为三段 gate G0/G1/G2,各 gate 由 integrated-CVODE wall + A5 hydrology 等更强 criteria 驱动,而非 Axis 4 drift。本研究 G0 即填这一 integrated-端 verifiability gap。

## §2.3 SUNDIALS SUNLinSol_Hypre 集成模式

SUNDIALS 6.0.0 自带 `SUNLinSol_Hypre` adapter (`<sundials/sunlinsol/sunlinsol_hypre.h>`),通过 4-callback 接口 (Initialize / SetATimes / SetPreconditioner / Setup / Solve) wrap Hypre BoomerAMG hierarchy 进 CVODE Newton-Krylov 迭代。SHUD-OpenMP 工程在 P8-tune.G0 PR-0 #414 引入 minimal SUNLinSol_Hypre wrapper (`tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp`),hardcode `(interp_type=6, coarsen_type=8)` per P8-tune.F 16-cell sweep best-combo,在 `cvode_config.cpp` 通过 `SHUD_LINSOL=amg` env-var 的 selector 替换 default SPGMR。

wrapper Initialize 阶段通过 `dlopen` 加载 versioned-soname `libHYPRE.so.3.1.0` (避免 `libHYPRE.so` 在 Ubuntu apt-installed 与 server source-built Hypre 之间的 ABI drift),Setup 阶段通过 `HYPRE_BoomerAMGCreate` + `HYPRE_BoomerAMGSetInterpType(6)` + `HYPRE_BoomerAMGSetCoarsenType(8)` 构建 hierarchy 并调 `HYPRE_BoomerAMGSetup(amg, A, x, b)`,Solve 阶段调 `HYPRE_BoomerAMGSolve(amg, A, b, x)` 完成 V-cycle 内迭代。Telemetry drain hook 在 `CVodeFree` 前通过 `HYPRE_BoomerAMGGetOperatorComplexity` + `HYPRE_BoomerAMGGetCumNnzAP` 提取真实 hierarchy quality 数据,emit `MARKER:AMG_TELEMETRY_REAL` 验证 G0-3 contract。

## §2.4 Slurm 三铁律 + cn-node Hypre 3.1.0 部署

服务器跑 SHUD 集成 smoke 受 CLAUDE.md Slurm 三铁律约束:(1) 从 `/scratch` 下 sbatch (policy 拦 `/users/$USER` 提交);(2) `#SBATCH --output/--error` 路径必须在 `/scratch` 共享盘;(3) 作业脚本里引用的 patch / hash / run.sh 都放 `/scratch`。Hypre 3.1.0 已部署在 `/scratch/frd_muziyao/local/hypre-3.1.0/` (PR-0 task 1.6 measured + pin);PR-A Slurm sbatch 在 `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0_amg_smoke.10014/` 提交,4-cell array (keliya / xinanjiang_upstream / heihe_x4 / heihe_x16) parallel deployment 在 cn-nodes CPU partition。

## §2.5 P8-tune.G0 在 carve-out chain 中的方法学定位

P8-tune.G0 与前序 P8-tune.D / P8-tune.F 的方法学差异:

| Aspect | P8-tune.F (pattern-only spike) | **P8-tune.G0 (本研究, integrated smoke)** |
|---|---|---|
| Scope | 16-cell pattern-only sweep | 4-cell integrated-CVODE smoke array |
| CVODE wire-up | 无 (zero-CVODE-wireup per spec REQ-1) | 有 (SUNLinSol_Hypre 接入 CVODE Newton-Krylov) |
| SHUD model run | 无 (zero SHUD model run) | 有 (90-day SHORT actual SHUD simulation) |
| Verdict axes | 5 (setup_wall + apply_wall + memory + cycle_complexity + operator_complexity) | 6 (G0-1 default-compat + G0-2 build + G0-3 telemetry-real + G0-4 integrated-completes + G0-5 wall-signal + G0-6 solver-stats) |
| Verdict branches | 5 (GO / Optional / NO-GO-heihe_x16-only / NO-GO-both / BLOCKED) | 2 (GO-G0 / NO-GO-G0) |
| Axis 4 / telemetry | `cycle_complexity = 2 × operator_complexity` hardcoded estimate | Hypre native `HYPRE_BoomerAMGGetCumNnzAP` + `HYPRE_BoomerAMGGetOperatorComplexity` |
| Wall measurement | per-cycle setup + apply (single V-cycle) | per-step over full SHUD run (integrated-CVODE) |
| Solver stats | n/a (pattern-only, no CVODE) | nfe/nli/nfeLS/ncfn/ncfl/netf (CVODE BDF/Newton telemetry) |
| Verdict sentinel | `PASS` | `AMG_OK` (G0 rename per spec REQ "verdict_class enum semantics") |
| ADR impact | new ADR-0007 (Accepted) | Append-only Amendment block to ADR-0007 (no new ADR per single-ADR pattern; PR #407 framing amendment precedent) |
| Master plan impact | §P8-tune.F [OPEN]→[CLOSED] + §P8-tune.G new anchor | §P8-tune.G0 [OPEN, HIGH]→[CLOSED] + §G1/§G2 [CLOSED-DEFERRED] |

P8-tune.G0 把 pattern-only 单 V-cycle 测量推进到 integrated-CVODE 多 V-cycle 多 Solve call 测量,引入 `ring-buffer drain hook` (per Setup/Solve callback 推 telemetry KV 到 thread-safe ring,run end 在 `CVodeFree` 前 drain) 解决多调用 telemetry 持久化问题。这是 pattern-to-integration 转化的标准工程模板。

---

# §3 Methodology / 方法论

## §3.1 6-gate hard verdict 框架

P8-tune.G0 verdict 框架由 6 个 AND-gate 组成,任一 FAIL 即 `g0_verdict_branch=NO-GO-G0`。Gate 定义 (per spec REQ "G0 verdict evaluates six PASS/FAIL gates"):

**Tab. 1: 6-gate hard verdict 定义**

| Gate | 验收 criterion | 验收范围 | 数据来源 |
|---|---|---|---|
| G0-1 default-compat | `SHUD_LINSOL` unset 和 `=spgmr` 都 bit-identical vs pre-G0 SPGMR baseline on `keliya` 90-day SHORT | per-platform anchor (Mac local ↔ `.review-evidence/g0-spgmr-baseline-90day-keliya/mac/` ; server cn-node ↔ `server/`) | PR-0 task 4.0 (Mac) + PR-A task 6.7 (server) baseline archives |
| G0-2 build | `make shud` + `make shud_omp HYPRE=1` 三端 green | macOS (brew Hypre 3.1.x) + Ubuntu CI (apt Hypre 3.1.0) + server (`/scratch/.../hypre-3.1.0/`) | PR-0 task 1.6 (server Hypre pin) + PR-A `precheck_env.sh` |
| G0-3 telemetry-real | 至少一 cell 含 `MARKER:AMG_TELEMETRY_REAL` AND `cycle_complexity` 源自 Hypre native API | per-cell stdout | `tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp` Setup/Solve callback emit |
| G0-4 integrated-completes | 全 4 cell 无 crash + 无 divergence/OOM/wall-overflow markers | `{keliya, xinanjiang_upstream, heihe_x4, heihe_x16}` × 90-day SHORT × `SHUD_LINSOL=amg` | exit code + stderr marker scan |
| G0-5 wall-signal | 至少一 {`heihe_x4`, `heihe_x16`}: `amg_wall_per_step < spgmr_wall_per_step` (case-specific) | case-specific baseline (PR-A measured + PR-B hot-patched 到 `spgmr_baseline_walls_g0.h`) | `cell_summary.amg_wall_per_step_sec` vs `SPGMR_PER_STEP_HEIHE_X{4,16}_S` |
| G0-6 solver-stats | 全 AMG_OK cell 的 `cvode_{nfe,nli,nfeLS,ncfn,ncfl,netf}` 非 NA | per-cell cell_summary KV block | CVODE `CVodeGet*` API |

6-gate AND-gate 是 PASS 的必要条件:`g0_verdict_branch = GO-G0` iff all six gates PASS;otherwise `g0_verdict_branch = NO-GO-G0`。任一 FAIL 触发 NO-GO-G0 + `MARKER:G0_NO_GO_DETECTED gate=<gate_id>` 行 emit。

## §3.2 verdict_class enum 重命名:AMG_OK 替代 PASS

per spec REQ "verdict_class enum semantics",G0 把 P8-tune.F 的 `PASS` sentinel 重命名为 `AMG_OK`。理由:`AMG_OK` 是 stronger signal — assert 集成 CVODE 收敛 + per-cell wall-signal 双重 validity,而 P8-tune.F `PASS` 仅 pattern-only hierarchy build + single V-cycle apply。enum 为 `{AMG_OK, AMG_SETUP_DIVERGE, AMG_SOLVE_DIVERGE, AMG_OOM, AMG_WALL_OVERFLOW}` 加 parser-side sentinel `MALFORMED` (用 when SIGKILL pre-trap 导致 cell_summary 缺失)。aggregator parser strict 接受这 6 个 verdict_class enum value,任何 `verdict_class=PASS` cell 触发 `MARKER:VERDICT_CLASS_MALFORMED expected=AMG_OK got=PASS` + G0-4 FAIL 贡献。

## §3.3 cell_summary KV schema (27 字段 multi-call semantics)

per spec REQ "G0 evidence schema is per-cell cell_summary KV plus cross-cell aggregate",per-cell cell_summary KV block 包含 27 字段 across 8 KV content lines + BEGIN/END delimiter (10 lines total per block)。Multi-call semantics 与 P8-tune.F single-cycle semantics 区别:

- `setup_wall_sec` = 全 Setup 调用的 per-call arithmetic mean (累计为 `amg_total_setup_wall_sec`)
- `apply_wall_sec` = 全 Solve 调用的 per-call arithmetic mean (累计为 `amg_total_solve_wall_sec`)
- `cycle_complexity` = `HYPRE_BoomerAMGGetCumNnzAP / IJMatrix nnz_A` end-of-run (Hypre 3.1.0 substitute semantics — STATIC hierarchy-size ratio,不是 per-cycle work ratio)
- `operator_complexity` = `HYPRE_BoomerAMGGetOperatorComplexity` end-of-run (Hypre native)
- `amg_telemetry_mean_iters` = per-Solve `HYPRE_BoomerAMGGetNumIterations` 全调用 arithmetic mean
- `amg_telemetry_mean_op_count` = per-Solve `HYPRE_BoomerAMGGetCumNnzAP` 全调用 arithmetic mean (monotone non-decreasing 因 cumulative)
- `n_cvode_steps` = CVODE `CVodeGetNumSteps` end-of-run
- `amg_wall_per_step_sec` = (setup_wall_sec_cumulative + solve_wall_sec_cumulative) / n_cvode_steps (per spec REQ "Wall-signal gate evaluation uses case-specific SPGMR baselines" Scenario "Both numbers include setup")
- 6 CVODE counters (`cvode_nfe`, `cvode_nli`, `cvode_nfeLS`, `cvode_ncfn`, `cvode_ncfl`, `cvode_netf`) end-of-run
- `residual_reduction_v1` 在 G0 中 **omitted** (P8-tune.F 字段含 per-cycle semantics 不 extend 到 multi-cycle multi-Solve)
- `colpack_version` = `absent` literal (PR-0 task 1.7 结论 ColPack 在集成 AMG 路径不需要)
- `wall_convention ∈ {setup_included, setup_excluded, telemetry_truncated, wall_total_proxy, nst_unavailable}` 反映 PR-0 spike 决议 + telemetry drain 健康度

cell_summary 不齐 27 字段 → aggregator emit `MARKER:CELL_SUMMARY_MALFORMED case=<name> got_fields=<N>` 到 stderr + 该 cell 对 G0-4/G0-5/G0-6 贡献 FAIL。

## §3.4 ring-buffer telemetry drain hook (PR-B Phase 6 critical fix)

集成 CVODE 多调用 telemetry 持久化挑战:Setup/Solve callback 在每 Newton 迭代触发,90-day SHORT 在 heihe_x4 上可达 ~250k Solve 调用,直接 emit stdout KV 每 call 会饱和 cn-node stderr buffer。解决方案:thread-safe lock-free ring buffer 在 wrapper 静态状态中,Setup/Solve callback 推 KV record `{call_id, setup_wall_sec, solve_wall_sec, iters, op_count}` 到 ring,run end 在 `CVodeFree` 前 drain ring 到 `SHUD_TELEMETRY_TSV` 环境变量指定路径。

Ring 容量 `2^17 = 131072` rows;若 push 速率 > drain 速率(没有显式 drain 时),drain hook 在 ring overflow 时丢弃最旧 records 并增 `dropped_overflow` counter,run end emit `MARKER:G0_TELEMETRY_RING_OVERFLOW dropped=<N> retained=<M>` + `wall_convention=telemetry_truncated`。

PR-B Phase 6 critical fix:原 wrapper 在 `SUNLinSolFree(LS)` 之后 才 drain,但 CVODE 析构链已经 free 了 user_data,drain 数据丢失;Phase 6 改在 `CVodeFree` 之前 drain,通过 SHUD shud.cpp 注入 `// Drain BEFORE CVodeFree per G0 telemetry contract` 一行注释 + drain 调用。SHUD `188854b` hot-fix + 外层 pointer-bump `4ade422` 是这一 fix 的 commit pair。

## §3.5 4-branch verdict (compared to P8-tune.F 5-branch)

G0 verdict 树相比 P8-tune.F 简化为 2-branch (GO-G0 / NO-GO-G0), 因 G0 是 first gate 不需 detailed reason classification — 任何 gate FAIL 都触发 NO-GO 并将 forward path 推到 G1/G2 deferred 或 P8-tune.H fallback。详细 reason 通过 per-gate `MARKER:G0_NO_GO_DETECTED gate=<gate_id>` 行记录,但不引入新 verdict_branch enum 复杂度。

## §3.6 byte-identical anchor contract

per spec REQ "G0 verdict byte-identical anchor contract" (mirrors P8-tune.F amg-pattern-spike-verdict REQ-6),G0 verdict markdown document (`docs/p8tune/amg_g0_verdict.md`) 含一 verdict-line block byte-identical to aggregator stdout block。验证:`awk '/G0_VERDICT_BEGIN/,/G0_VERDICT_END/' <aggregator stdout>` 与 `awk '/G0_VERDICT_BEGIN/,/G0_VERDICT_END/' docs/p8tune/amg_g0_verdict.md` 之 diff 返回 empty。这一 contract 保 aggregator pipeline 与 ADR-0007 §Amendment 之间零 hand-curated drift。

## §3.7 ADR-0007 append-only Amendment 模式

per spec REQ "G0 verdict updates ADR-0007 via append-only Amendment block",G0 verdict MUST append 一新 dated `## Amendment <YYYY-MM-DD> (G0 verdict)` 块到 `docs/adr/0007-amg-spike-decision.md` §Forward action section。Pre-existing ADR-0007 §Status metadata bullet (`- **Status**: Accepted ...` at L3) 与 `## Decision` L2-header section MUST 保持 byte-identical to pre-G0 form。验证:`git diff main -- docs/adr/0007-amg-spike-decision.md | grep -E '^[-+](- \*\*Status\*\*|## Decision)' | wc -l` returns 0。

这一 pattern 沿用 PR #407 framing amendment 单-ADR-append 先例,**NOT** 引入新 ADR-0008(后者 reserved for G2 production default flip if forward path 转 active)。

本节小结:G0 方法论将 ADR-0007 §Forward action Amendment 2026-06-29 锁定的 G0 gate 映射为 (i) 6-gate hard AND-gate verdict + (ii) `AMG_OK` enum + 27-字段 cell_summary multi-call schema + (iii) thread-safe ring-buffer telemetry drain hook + (iv) 2-branch verdict + (v) byte-identical anchor contract + (vi) ADR-0007 append-only Amendment 模式。这六要素共同构成 H1/H2/H3 三假设 + 三个支撑 gates 的 operational falsification 框架。

---

# §4 Experimental Setup / 实验设置

## §4.1 硬件平台 + 软件栈

实验在两端异构环境执行:**Mac local** (PR-0 wrapper + G0-1 Mac baseline + PR-B local re-run keliya cell) + **server** (PR-A 4-cell Slurm array + production verdict 评估权威)。

**Tab. 2: 硬件 + 软件栈**

| 项 | Server (PR-A 4-cell + PR-B keliya 部分) | Mac (PR-0 wrapper + PR-B local re-run keliya cell) |
|---|---|---|
| Endpoint | `frd_muziyao@210.77.77.22:32099` (cn-node CPU partition) | Apple M4 Pro local |
| OS / Kernel | Ubuntu 24.04.2 LTS, Linux 6.8.0-57-generic | Darwin 24.6.0 (macOS Sequoia 15) |
| CPU / cores | Intel Xeon dual-socket NUMA (cn14/cn23 verified 173 GiB RAM) | Apple M4 Pro 14-core (4P + 10E) |
| Compiler | GCC 13.3.0 | Apple Clang 17.0.0 |
| Hypre version (server) | 3.1.0 (`/scratch/frd_muziyao/local/hypre-3.1.0/`) | brew install Hypre 3.1.x |
| dlopen target | `libHYPRE.so.3.1.0` (versioned soname) | `libHYPRE.dylib` (macOS native) |
| SUNDIALS | 6.0.0 (pinned, unchanged from B0/B1a) | 同 |
| Scheduler | Slurm sbatch from `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0_amg_smoke.10014/` + `--output/--error` in `/scratch` 共享盘 + `--time=08:00:00` per cell | local shell |
| Job ID | Slurm 10014 (4-cell array `--array=0-3`) | n/a |
| Submit window | 2026-06-30 4-cell Slurm array deployment | 2026-06-30 local re-run for keliya telemetry post-Phase-6 fix |

## §4.2 Benchmark cases (90-day SHORT)

实验覆盖 4 case 跨网格规模 (NumY 1.5K → 485K),全部 ≤90-day truncated per CLAUDE.md 项目级铁律:

**Tab. 3: P8-tune.G0 smoke roster** (cell name → cfg.para 决定 — 见 CLAUDE.md case_deployment_map.md basin folder name vs SHUD project name disambiguation)

| Case | NumEle | NumY | basin folder | SHUD project name | Platform |
|---|---:|---:|---|---|---|
| keliya | 484 | 1785 | `Basins/keliya/` | `keliya` | Server + Mac |
| xinanjiang_upstream | 801 | n/a (small) | `Basins/xinanjiang_upstream/` | `xinanjiang` | Server |
| heihe_x4 | 40046 | 124395 | `Basins/heihe_x4/` (常驻, P8-tune.D deploy reused) | `heihe_x4` | Server-only |
| heihe_x16 | 160331 | 485250 | `Basins/heihe_x16/` (常驻) | `heihe_x16` | Server-only |

`xinanjiang_upstream` 替代 P8-tune.F `heihe` 作为 medium-scale smoke cell — 选定理由 (per PR-A design):G0 是 first-time integrated-CVODE smoke,xinanjiang_upstream 在 cn-node 上有 freshest 90-day deployment + minimal forcing-data 复杂度 + 实测 33223 nst < heihe 量级,适合作为 G0 medium-scale smoke 代表。

## §4.3 SPGMR baselines (case-specific, hot-patched)

per spec REQ "Wall-signal gate evaluation uses case-specific SPGMR baselines",PR-0 task 1.2 决议 setup-inclusion convention = setup_included,PR-A measured + PR-B hot-patched 到 `tools/p8tune.G0/spgmr_baseline_walls_g0.h`:

- `SPGMR_PER_STEP_HEIHE_X4_S = 0.238369`
- `SPGMR_PER_STEP_HEIHE_X16_S = 0.952489`

60-cell anchor `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579` (in `tools/p8tune.D/spgmr_baseline_walls.h`) 保留作历史 context only,NOT 用作 G0-5 阈值。

## §4.4 4-cell Slurm sbatch 部署

PR-A `tools/p8tune.G0/g0_amg_smoke_array.sbatch`:

```bash
#SBATCH --array=0-3
#SBATCH --partition=CPU
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/g0_amg_smoke.${SLURM_JOB_ID}/cell-${SLURM_ARRAY_TASK_ID}.out
#SBATCH --error=...err
```

4 cells 独立 1-node deployment,每 cell 8h wall budget (per design D6 与 P8-tune.F 一致),`OMP_NUM_THREADS=1` (G0 是 single-thread smoke,thread-pinning 由 PR-0 task 1.7 验证)。命名 `cell-NN.out`/`.err`,aggregator iterate canonical `EXPECTED_CELLS_LIST=(keliya xinanjiang_upstream heihe_x4 heihe_x16)`。

## §4.5 telemetry drain + dlopen versioned-soname (PR-A + PR-B critical)

PR-A wrapper 引入 `dlopen("libHYPRE.so.3.1.0", RTLD_LAZY)` versioned-soname 路径,避免 `libHYPRE.so` symlink target 在 server 部署中错指 (ubuntu apt 24.04 升级 Hypre 4.x 在系统级风险)。Mac 走 `dlopen("libHYPRE.dylib")` 不需要 versioning (brew managed)。

PR-B Phase 6 critical fix 把 telemetry drain 移到 CVodeFree 之前 (SHUD `188854b` hot-fix + 外层 pointer-bump `4ade422`),解决 PR-A 原 wrapper 中 drain 在 user_data 析构后导致 telemetry 丢失。这一 fix 是 IA-5 hot-patch — invoked when PR-B aggregator 发现 operator_complexity gap (PR-A 输出无 telemetry-derived cycle_complexity)。

## §4.6 实验流程与 reproducibility footprint

```bash
# 1. Sync to baseline/p8tune-amg-g0-spike + SHUD pin 188854b (PR-B Phase 6 hot-fix landed)
git checkout baseline/p8tune-amg-g0-spike
git pull --ff-only --recurse-submodules

# 2. Build SHUD with HYPRE
cd SHUD && make clean && make shud_omp HYPRE=1
# verify wrapper symbol present in libshud.a + linsol selector hook in cvode_config.cpp

# 3. Submit 4-cell Slurm array from /scratch (Slurm 三铁律)
ssh frd_muziyao@210.77.77.22 -p 32099
cd /scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs
./tools/p8tune.G0/precheck_env.sh  # hypre version + cn-node memory + SHUD_LINSOL hook
sbatch tools/p8tune.G0/g0_amg_smoke_array.sbatch  # --array=0-3

# 4. Aggregator (PR-B)
bash tools/p8tune.G0/aggregate_g0_smoke.sh .review-evidence/g0-amg-smoke-array-rerun/

# 5. Capture verdict + ADR amendment (PR-C this work)
bash tools/p8tune.G0/aggregate_g0_smoke.sh .review-evidence/g0-amg-smoke-array-rerun/ \
  | awk '/G0_VERDICT_BEGIN/,/G0_VERDICT_END/' \
  > /tmp/g0-anchor.txt
```

Artifact 存放于 `.review-evidence/g0-amg-smoke-array-rerun/` (per-cell `cell-*.out`/`.err` + `cell-keliya.telemetry.tsv` + `aggregate.tsv` + `README.md`)。

本节小结:实验设置满足 (i) production scale mesh 覆盖 (4 cell × 90-day SHORT),(ii) Slurm 三铁律合规,(iii) dlopen versioned-soname soname pinning,(iv) telemetry drain hook 在 CVodeFree 前正确执行,(v) per-platform G0-1 baseline 双端独立 archive,(vi) reproducibility footprint 完整。

---

# §5 Results / 结果

本章按 6-gate 顺序汇报结果,然后跨 cell 汇总 wall + solver stats + telemetry。

## §5.1 6-gate evaluation (verdict-level)

**Tab. 4: 6-gate evaluation 结果** (引自 `tools/p8tune.G0/aggregate_g0_smoke.sh` stdout + `.review-evidence/g0-amg-smoke-array-rerun/aggregate.tsv`)

| Gate | Result | One-line 证据 |
|---|---|---|
| G0-1 default-compat | PASS | PR-0 #414 merge: `SHUD_LINSOL` unset 与 `=spgmr` 都 bit-identical to pre-G0 SPGMR baseline on `keliya` 90-day SHORT;Mac baseline archived `.review-evidence/g0-spgmr-baseline-90day-keliya/mac/` |
| G0-2 build | PASS | `make shud_omp HYPRE=1` 三端 green:Mac brew Hypre 3.1.x + Ubuntu CI apt 3.1.0 + server `/scratch/.../hypre-3.1.0/` |
| G0-3 telemetry-real | PASS | keliya stdout 含 `MARKER:AMG_TELEMETRY_REAL`;`cycle_complexity=1.000000` + `operator_complexity=1.002945` 源自 Hypre native API (`HYPRE_BoomerAMGGetCumNnzAP` + `HYPRE_BoomerAMGGetOperatorComplexity`) — H2 验证 PASS |
| G0-4 integrated-completes | **FAIL** | `heihe_x16` MALFORMED — Slurm SIGKILL at 8h budget pre-Phase-6 wrapper deploy time line;Mac local 不能复现 252k-NumEle case;server array re-run with 24h budget 是 Future Work |
| G0-5 wall-signal | PASS (per-step) | `heihe_x4` AMG per_step `0.092873s` < SPGMR baseline `0.238369s` (0.39× per-step);OR-aggregate succeeds via heihe_x4 |
| G0-6 solver-stats | PASS | `cvode_nfe/nli/nfeLS/ncfn/ncfl/netf` 全 non-NA for 3 AMG_OK cells (`keliya`, `xinanjiang_upstream`, `heihe_x4`);`heihe_x16` MALFORMED exempt |
| **g0_verdict_branch** | **NO-GO-G0** | G0-4 FAIL blocks GO-G0;heihe_x4 total wall 15.1× regress independently confirms AMG-not-beneficial at integrated-CVODE level |

Anchor block (byte-identical to aggregator stdout per spec REQ "G0 verdict byte-identical anchor contract"):

```
MARKER:G0_VERDICT_BEGIN
G0_3=PASS
G0_4=FAIL reason=heihe_x16:MALFORMED
G0_5=PASS
G0_6=PASS
G0_OVERALL=NO-GO
MARKER:G0_VERDICT_END
```

## §5.2 Wall-signal per cell (per-step vs total)

**Tab. 5: Per-cell wall-signal table** (引自 `aggregate.tsv` cols 4-6 + 内部 nst 计数)

| cell | SPGMR per_step (s) | AMG per_step (s) | AMG/SPGMR ratio (per_step) | AMG nst | SPGMR nst | AMG/SPGMR nst ratio | Total wall ratio | wall_convention |
|---|---|---|---|---|---|---|---|---|
| keliya | n/a | 0.001432 (179s / 124913 nst, setup+solve only) | n/a | 124913 | n/a | n/a | n/a | telemetry_truncated |
| xinanjiang_upstream | n/a | 0.002709 (90s / 33223 nst, wall_total_proxy) | n/a | 33223 | n/a | n/a | n/a | wall_total_proxy |
| heihe_x4 | 0.238369 | 0.092873 (23660s / 254756 nst, wall_total_proxy) | **0.39×** | 254756 | 6572 | **38.8×** | **15.1× WORSE** | wall_total_proxy |
| heihe_x16 | 0.952489 | n/a (SIGKILL) | n/a | n/a | 6556 | n/a | n/a | nst_unavailable |

观察:
- **heihe_x4 per-step 0.39× faster**: AMG-preconditioned 内核 Krylov 收敛速度比 SPGMR maxl=5 baseline 显著快。这与 P8-tune.F pattern-only 阶段的 AMG/KLU wall ratio at heihe_x4 = 0.09× (10.7× faster than KLU) 趋势一致,验证 V-cycle 在大 case 上的算法优势在集成 CVODE 内核层仍 hold。
- **heihe_x4 total 15.1× WORSE**: 但 CVODE nst 增 38.8× (254756 AMG nst vs 6572 SPGMR nst) 抹掉 per-step 优势 — outer Newton 在 AMG-preconditioned 下需 显著更多 step。
- **G0-5 spec semantics ambiguity**: spec REQ-5 写 `amg_wall_per_step[case] < spgmr_wall_per_step[case]`,heihe_x4 per_step 满足 (0.0929 < 0.2384),G0-5 PASS。但 production wall reality (total) 是 15.1× regress。这一 spec-vs-reality 缝隙在 §6.2 详 discussed。

## §5.3 Solver stats per cell (CVODE BDF/Newton 内 SPGMR-vs-AMG interaction)

**Tab. 6: CVODE solver stats per AMG_OK cell** (引自 `aggregate.tsv` cols 9-14)

| cell | cvode_nfe | cvode_nli | cvode_nfeLS | cvode_ncfn | cvode_ncfl | cvode_netf |
|---|---:|---:|---:|---:|---:|---:|
| keliya | 246759 | 1726579 | 1785 | 30640 | 0 | 3 |
| xinanjiang_upstream | 50488 | 151528 | 2619 | 5264 | 0 | 0 |
| heihe_x4 | 502982 | 983928 | 124395 | 100138 | 0 | 2 |
| heihe_x16 | NA (MALFORMED) | NA | NA | NA | NA | NA |

观察:
- **ncfn 主导失效模式**: `heihe_x4` `ncfn=100138` (Newton 控制失败 ≈ nfe 的 20%),`heihe_x4` AMG-preconditioned Newton iteration 反复 fail convergence,触发 CVODE step retry + smaller dt + nst 暴增 (38.8× more steps)。
- **ncfl=0 全 cell**: AMG 内核 Krylov 在每 CVODE inner call 都收敛,linear-solver 端无失败 — 瓶颈在 outer Newton 而非 inner Krylov。
- **nfeLS 与 NumY 对齐**: `heihe_x4` `nfeLS=124395 ≈ NumY=124395` — Setup callback 在每 step 都 rebuild Jacobian (per spec REQ "Hypre Setup wrapper rebuilds hierarchy"),与 FD-color Jacobian setup overhead expectation 一致。
- **keliya `nli=1726579`**: 极高 Krylov inner iteration count,反映 keliya 小 case 在 AMG-preconditioned 内层迭代上 over-spending (V-cycle setup overhead 在 NumY=1785 上 dominate per Newton call)。

## §5.4 Telemetry 验证 H2 (real Hypre signal)

`keliya` cell 经 PR-B Phase 6 drain hook + SHUD `188854b` hot-fix 后 telemetry 提取成功 (`.review-evidence/g0-amg-smoke-array-rerun/cell-keliya.telemetry.tsv`):

- `operator_complexity = 1.002945` (Hypre native `HYPRE_BoomerAMGGetOperatorComplexity`)
- `cycle_complexity = 1.000000` (Hypre 3.1.0 substitute via `HYPRE_BoomerAMGGetCumNnzAP / nnz_A` end-of-run, STATIC ratio)
- `amg_telemetry_mean_iters ≈ 6.996452` (`HYPRE_BoomerAMGGetNumIterations` per-Solve mean)
- `amg_telemetry_mean_op_count ≈ 5472.0` (`HYPRE_BoomerAMGGetCumNnzAP` per-Solve mean; monotone non-decreasing 因 cumulative semantics)
- `amg_total_setup_wall_sec ≈ 0.0` (sub-ms 累计;keliya hierarchy 极小)
- `amg_total_solve_wall_sec ≈ 19.05` (累计 124913 Solve calls × 0.153 ms 均值)
- `amg_telemetry_dropped_overflow = 115656` (ring overflow drops over 1.7M-NLI run, retained=131072,**47% loss**)

`MARKER:AMG_TELEMETRY_REAL` 在 keliya stdout 出现,G0-3 contract PASS。H2 假设(telemetry-real signal at scale)PASS — 注意 47% ring overflow loss 不破坏 H2 (G0-3 contract 只要 AT LEAST ONE cell 有 telemetry-real signal,且 retained 131072 samples 仍 statistically valid for mean estimation)。

xinanjiang_upstream / heihe_x4 / heihe_x16 telemetry 在 PR-A 阶段未 emit (PR-A wrapper pre-dated Phase 6 drain hook),aggregator 报 `MARKER:G0_TELEMETRY_TSV_ABSENT case=<name>` + 这些 cell `wall_convention=wall_total_proxy` (honest convention labeling per IA-3 fix)。这不影响 G0-3 verdict (keliya alone 满足 OR-gate)。

## §5.5 H1/H2/H3 假设验证总结

**Tab. 7: Three-hypothesis verification**

| Hypothesis | Statement | Result | Evidence |
|---|---|---|---|
| H1 (default-compat) | `SHUD_LINSOL` unset / `=spgmr` bit-identical to pre-G0 SPGMR baseline | **PASS** | G0-1 PASS per PR-0 #414 Mac baseline archive |
| H2 (telemetry-real) | At least one AMG_OK cell has telemetry-derived (Hypre native) `cycle_complexity` | **PASS** | G0-3 PASS via keliya `cycle_complexity=1.000000` from `HYPRE_BoomerAMGGetCumNnzAP` |
| H3 (integrated-AMG wall-beneficial) | At least one of {heihe_x4, heihe_x16}: `amg_wall_per_step < spgmr_wall_per_step` | **PASS (per-step) / FAIL (total)** | G0-5 PASS via heihe_x4 per-step 0.39× faster;BUT total wall 15.1× WORSE due to ncfn-driven nst inflation |

H3 显示 spec-level G0-5 PASS criterion (`amg_wall_per_step < spgmr_wall_per_step` per-step OR-gate) 在 heihe_x4 上 nominally 满足,但 production wall (total) 与 spec letter-of-the-law verdict 强烈分歧 — 这是 G0 first-gate 学到的关键经验:per-step wall benefit 在 integrated-CVODE 中可能被 outer-Newton control failure 抹掉,future gate (G1 integrated 18-cell wall benchmark) 必须以 total wall 为度量,而非 per-step。

本节小结:6-gate evaluation 产 PASS/PASS/PASS/**FAIL**/PASS/PASS = `g0_verdict_branch=NO-GO-G0`。G0-4 FAIL (heihe_x16 MALFORMED) 是 first-order verdict gate;H3 在 per-step 层 PASS 但 total wall 15.1× regression 强化 NO-GO 解读。后续 §6 详 discussed。

---

# §6 Discussion / 讨论

## §6.1 H1 / H2 / H3 假设验证 + G0 verdict 综合解读

H1 (default-compat) + H2 (telemetry-real) 在 G0 顺利 PASS — wrapper Initialize / SetATimes / Setup / Solve 4-callback hook 在不破坏 SPGMR default 的前提下 cleanly 切换到 AMG,且真实 Hypre telemetry 在 keliya 91k-call Solve sequence 上稳定 emit (47% ring overflow loss 是 telemetry sample 量的损失而非 contract 失败)。这两 PASS 验证 ADR-0007 §Forward action L25 escape hatch 的 architectural feasibility — `SHUD_LINSOL=amg` 是 production-grade env-var hook,不只是 spike-quality prototype。

H3 (integrated-AMG wall-beneficial) 在 G0-5 per-step letter-of-the-law 层 PASS,但 production wall reality 强烈反驳:`heihe_x4` total 15.1× WORSE because of `ncfn=100138` Newton control failure 主导的 nst 38.8× inflation。这一 phenomenon 是 G0 first-gate 学到的 critical insight,在 P8-tune.F pattern-only spike 中无法观察到 (因 pattern-only 不接 CVODE,无 nst / ncfn 数据)。

`g0_verdict_branch = NO-GO-G0` 综合解读:6-gate AND-gate 失败 (G0-4 FAIL),且 H3 production-reality 也 fail — AMG 在 SHUD 这类 hydrology matrix shapes 上不是 production-grade architectural substrate at the integrated-CVODE level。

## §6.2 per-step vs total wall 缝隙 + G0-5 spec semantic

G0-5 spec REQ-5 写 `amg_wall_per_step[case] < spgmr_wall_per_step[case]` 是 per-step OR-gate。letter-of-the-law verdict heihe_x4 per_step 0.0929 < 0.2384 → G0-5 PASS。但 production wall reality:

```
heihe_x4 SPGMR total ≈ 0.238369 × 6572 = 1566 s = 26 min
heihe_x4 AMG total = 23660 s = 6.6 h
ratio = 23660 / 1566 = 15.1× WORSE
```

per-step PASS 反映了 V-cycle 在 AMG-preconditioned Krylov 内层迭代上的算法优势,但 outer Newton ncfn 主导的 nst 暴增 (6572 → 254756, 38.8×) 抹掉这一优势 + 还产生 14× 净 regression。

G0-5 spec 设计在制定时未预 ncfn 主导失效模式 — G0 first-gate 学到的经验:future G1/G2 gate criteria 必须以 total wall 为 first-order 度量。但 G0-5 spec letter 已落地 + aggregator 已 hard-coded per-step semantics,本 PR-C 不改 spec 也不改 aggregator (避免 PR-C 与 PR-B 的 scope 干扰)。`docs/p8tune/amg_g0_verdict.md` §Wall-signal table 显示 per-step PASS + total 15.1× WORSE 双重事实并存,verdict doc 文字明示 G0_OVERALL=NO-GO 基于 G0-4 FAIL + H3 reality reversal 双因素,而非单 G0-5 driver。

这一 spec-vs-reality 缝隙是 G0 epic 学到的 first-order 方法学贡献:**per-step wall criterion 在 multilevel-iterative preconditioner 推 outer Newton 的 architectural assessment 中 insufficient**,future gates 应用 total wall + nst budget joint criterion。

## §6.3 ncfn 主导的 outer-Newton 失效模式 (新现象学发现)

`heihe_x4` `ncfn=100138` 表明:

- AMG hierarchy 内层 V-cycle 成功收敛 (`ncfl=0`)
- 但 outer Newton iteration 反复 control fail — 这通常由 Jacobian 不一致 (Setup callback 后 J 矩阵 stale) 或 linear solver tolerance/Newton-norm mismatch 引发。

排查方向 (Future Work, 不在本 PR-C scope):

1. **Jacobian staleness**: wrapper Setup 是 per CVODE step rebuild 还是 per Newton iteration rebuild?如前者,CVODE BDF stiff regime 下 J 矩阵在 step 内 drift 会触发 Newton failure。spec REQ "Hypre Setup wrapper rebuilds hierarchy" 写 per Setup callback rebuild;但 callback frequency 可能与 stiff-regime 需求不匹配。
2. **AMG 内层 tolerance**: `HYPRE_BoomerAMGSetTol` 默认值 (1e-7) 可能与 CVODE Newton tolerance 不一致, inner residual 过严 → outer Newton 看到 "wrong" J^{-1} b 估值 → control fail。
3. **AMG-vs-Jacobian-symmetric assumption**: SHUD Jacobian 含 river-channel directional + lake-bank one-directional 项,mildly non-symmetric。BoomerAMG classical Ruge-Stüben 假设 M-matrix;mild non-symmetry 可能 导致 hierarchy build 通过 (G0-3 PASS) 但 V-cycle apply 产生 numerical drift,outer Newton 解读为 control fail。
4. **CVODE BDF order / step controller interaction**: stiff regime BDF-5 在 AMG-preconditioned 内核下 step controller 可能 不稳,频繁 dt 调整。

ADR-0007 §Discussion §"Case-asymmetric scaling third-epic anchor" 提及 future epic 应 implement case-asymmetric solver policy (env-var `SHUD_LINSOL=spgmr|klu|amg` + NumY 阈值 lookup)。本 G0 NO-GO verdict 反过来 informs:**case-asymmetric policy 在 AMG 路径上 由 ncfn-driven failure mode 阻塞**,即使 mid-large NumY (heihe_x4 124K NumY) 也不应 default to AMG。

## §6.4 与 P8-tune.F pattern-only verdict 对比

**Tab. 8: G0 vs P8-tune.F 关键比较**

| Aspect | P8-tune.F (pattern-only) | P8-tune.G0 (本研究, integrated) |
|---|---|---|
| Verdict | strict NO-GO-both / amended GO 二分 | NO-GO-G0 一元 |
| Driving FAIL gate | Axis 4 (hard-coded estimate) — spurious | G0-4 (heihe_x16 MALFORMED) + G0-5 spec-vs-reality 缝隙 — empirical |
| operator_complexity range | [1.0000, 1.0106] (4 case × 4 combo = 16 cell) | 1.002945 (keliya only, integrated) |
| heihe_x4 wall conclusion | 0.027655 s setup+apply (single V-cycle, 10.7× faster than KLU) | per_step 0.0929 s (0.39× SPGMR per-step) BUT 15.1× total WORSE |
| ncfn / outer Newton | n/a (no CVODE wire) | 100138 — first observation, dominant failure mode |
| ADR-0007 impact | §Decision authored, strict-vs-amended verdict 二分 | §Forward action Amendment appended, integrated-端 evidence 强化 strict 解读 |
| Forward path | §P8-tune.G three-gate sequence | §P8-tune.H GPU sparse fallback (§G1/§G2 [CLOSED-DEFERRED]) |

P8-tune.F amended GO 解读 (4-axis AND-gate excluding Axis 4 instrumentation gap) 假设 axes 1/2/3/5 PASS 充分 capture AMG production feasibility — G0 集成 verdict 反驳这一假设:即使 pattern-only setup + apply walls PASS + memory + operator_complexity 全 healthy,集成 CVODE 中 ncfn 主导的 outer-Newton failure 仍 wipe out 算法优势。Strict NO-GO-both verdict 在 integrated 层得到 first-order 经验确认。

## §6.5 forward path: P8-tune.H GPU sparse fallback

ADR-0007 §Forward action L25 escape hatch + G0 Amendment 2026-06-30 锁定 P8-tune.H GPU sparse / domain decomposition spike 作为下一 architectural fallback。Forms under consideration:

1. **CUDA sparse direct via cuSPARSE + iterative refinement**: cuSPARSE 5.0+ `cusparseSpSV` direct solver + IR refinement; GPU-presence gate (`sinfo -p GPU` → `gn01` available) 是 precondition。
2. **GPU AMG with mixed-precision (Hypre 2.30+ cusparse_use=1)**: Hypre upstream 已支持 mixed-precision V-cycle (FP32 apply + FP64 outer)。
3. **Domain decomposition + serial sparse direct per subdomain**: cn-node 32-core OpenMP 内,把 NumY 切 32 subdomain,每 subdomain 走 KLU/UMFPACK 串行 (ADR-0005 在 keliya/heihe 已证 KLU pattern-feasible),subdomain 交界用 Schwarz alternating + few Krylov outer iterations。

P8-tune.H epic 不在本 PR-C scope (PR-C 是 doc-only governance close)。立项时机由 orchestrator decision tree (per ADR-0007 §Amendment 2026-06-30 §Forward action update 末段) 触发。

## §6.6 "Never break userspace" 严格遵守 + zero user-facing 行为变化

PR-0 + PR-A + PR-B 全程 `SHUD_LINSOL` env-var 设计 = `default=spgmr`, opt-in only。本 PR-C doc-only,不改 code path。NO-GO-G0 verdict 在 production 行为上 preserves SUNDIALS default solver (PREC_NONE SPGMR maxl=5) for all current users; 无 user-facing 行为变化。`SHUD_LINSOL=amg` 保留作为 research opt-in knob, 但 verdict doc 文字 + ADR-0007 amendment 明示 NOT recommended for production。

forward §P8-tune.H spike 若启动也走 zero-source-patch + env-var opt-in (`SHUD_LINSOL=cusparse` / `SHUD_LINSOL=ddsparse`) 模式,production default 永远是 SPGMR。

---

# §7 Limitations & Threats to Validity / 限制与威胁

## §7.1 4-cell smoke 性质显式 limit (不是 benchmark)

- **Sample size = 4 cells × 90-day SHORT each**,远小于 ADR-0004 P8-tune.C 60-cell PRD anchor。统计 power 足以 falsify production-grade AMG 假设 at heihe_x4 scale (15.1× total regression 是 robust signal),但不足 characterize case-roster variance — 例如 P8-tune.F roster 中 heihe (NumY=21K, intermediate) 在 G0 中由 xinanjiang_upstream 代替,xinanjiang 的 33223 nst 与 heihe nst pattern 是否一致未 verify。
- **G0 是 first-time integrated smoke**,不是 production benchmark。G1 18-cell 才接近 benchmark scale (4 case × 2 thread-count × 3 rep)。
- **90-day SHORT 截断**: 保持 OpenMP 并行验证 + bitwise neutrality 验收所需的最小 model time,但 ncfn 与 step controller 之间的 long-term equilibrium 未观察 — 90-day window 可能 over-emphasize startup transient regime,实际 4-year production run 中 ncfn 可能略低 (但不太可能从 100k 量级降到 ncfn≪nst 量级)。

## §7.2 heihe_x16 SIGKILL 阻塞 G0-4 evaluation

`heihe_x16` 在 8h Slurm wall budget 内未 complete + SIGKILL fired 前 SIGTERM trap 未 emit cell_summary → `verdict_class=MALFORMED`(aggregator parser-side sentinel)。这意味着:

- G0-4 evaluation for heihe_x16 没有 definitive PASS/FAIL data (只有 "未 complete + 未 emit clean classification"),aggregator 保守 evaluate as FAIL 触发 NO-GO-G0。
- AMG_WALL_OVERFLOW 与 MALFORMED_RUNNER_BUG 的区分需 server array re-run with 24h Slurm budget + post-Phase-6 wrapper SIGTERM trap fully wired。
- **即使 heihe_x16 re-run upgrade 到 AMG_WALL_OVERFLOW 或 even AMG_OK**, G0_OVERALL=NO-GO 仍 hold,因 heihe_x4 15.1× total wall regression 独立 demonstrate AMG-not-beneficial — 这是 verdict doc 与 ADR amendment 都明确写下的 forward-invariant 主张。

## §7.3 G0-5 per-step semantic 与 production-reality 分歧

§6.2 详 discussed — G0-5 spec REQ-5 per-step OR-gate 不 capture ncfn-driven nst inflation,heihe_x4 letter-of-the-law PASS 但 reality 15.1× WORSE。G0 epic 内不修 spec/aggregator (与 PR-B scope 干扰风险高),future G1/G2 spec 应 directly 引 total wall + nst budget joint criterion。

## §7.4 telemetry ring overflow (47% sample loss on keliya)

keliya 1.7M-NLI run 在 131072-row ring 上溢 → 115656 dropped (47% loss)。这一 sample loss 不破坏 H2 (cycle_complexity / operator_complexity 是 end-of-run static call,不依赖 per-Solve sample) 但:

- `amg_telemetry_mean_iters` (6.996452) + `amg_telemetry_mean_op_count` (5472.0) 是 retained samples 算 mean,有 ring drop pattern 引入的 bias (drop pattern 是 oldest-first FIFO drop,所以 retained 偏 final-state-of-run 而非 averaging full life-cycle)。
- G1/G2 integrated benchmark 应 enlarge ring (`2^20 = 1M rows` for heihe_x16 ~ 500k-NLI scale) 或改 streaming TSV emit (per-Solve flush 而非 in-memory ring) 避免 drop。

## §7.5 PR-A 时代 cells (xinanjiang_upstream / heihe_x4 / heihe_x16) 未 emit telemetry TSV

PR-A wrapper pre-dated PR-B Phase 6 drain hook,这 3 cell 的 telemetry TSV 缺失,`wall_convention=wall_total_proxy` (= SHUD startup/IO 也算入 per_step)。这导致 §5.2 Wall-signal table 中 heihe_x4 ratio 比较是 apples-to-apples (SPGMR baseline also `wall_total_proxy`) 但 hidden setup vs solve 分离信息 — future re-run on post-Phase-6 wrapper 可 emit `setup_plus_solve_only` convention sharper attribution。

不影响 G0 verdict (G0-3 PASS via keliya alone; G0-5 spec 不需 setup vs solve split)。

## §7.6 Mac local 不 reproducible for heihe_x16

Apple M4 Pro local 不能 reproduce 252k-NumEle case (NumY=485250 → ~ 60 MB AMG hierarchy memory + 6.3 GB SHUD heap, Mac 24-32 GB unified-memory 单进程极限),server-only。这一 mac-vs-server asymmetry 限制 PR-C 阶段独立 reproduction;PR-D capstone-merge 前需 orchestrator-spawned server re-run 才能 close G0-4 verdict gap (但 §7.2 已说明 G0_OVERALL 不会改变)。

## §7.7 Hypre 版本钉锁 + dlopen versioned-soname

实测 `hypre_version=3.1.0` (Ubuntu 24.04 apt + server `/scratch/.../hypre-3.1.0/`),非 upstream 最新 (Hypre 2.30+ 已有 mixed-precision V-cycle + cusparse_use=1 GPU AMG)。future P8-tune.H 若评估 GPU AMG 路径,应 upgrade Hypre + 重 emit versioned-soname。本 G0 verdict 内 Hypre 版本 不是 limit driver — 3.1.0 BoomerAMG 在 SHUD shape 上的 fundamental ncfn-driven failure 应 在 Hypre 2.30+ 上 reproducible (除非 mixed-precision 推 改 outer Newton behavior, 但 这需要 P8-tune.H 才能 verify)。

## §7.8 SHUD `Model_Data` dtor UB (历史 #386, P8-tune.F PR-0 已修复)

P8-tune.F PR-0 #394 已 fix #386 SHUD Model_Data dtor uninit-pointer UB + 移除 `_exit(0)` workaround,P8-tune.G0 全程 inherit 这一 fix。本 G0 epic 无新 dtor UB 表现,但 wrapper user_data lifecycle (G0 引入新 wrapper static state 与 ring buffer) 在 PR-B Phase 6 drain-before-CVodeFree fix 之前曾导致 telemetry 丢失 (drain 在 user_data 析构之后)。Future wrapper 修订应 同时 audit user_data + telemetry buffer lifecycle。

---

# §8 Conclusion / 结论

本研究通过 4-PR 序列 (PR-0 #414 wrapper + PR-A #415 4-cell smoke + PR-B #416 aggregator + telemetry drain + PR-C #<this PR> verdict doc + ADR amendment) 完成 P8-tune.G0 integrated AMG smoke epic,在 6-gate AND-gate hard verdict + dlopen versioned-soname + ring-buffer telemetry drain hook + byte-identical anchor contract 方法学框架下,产出 SHUD-OpenMP 工程主线 ADR-0007 §Forward action Amendment 2026-06-29 三段 gate 序列 G0/G1/G2 的 first gate verdict:**g0_verdict_branch = NO-GO-G0**。

主要结论:

1. **6-gate evaluation: 5 PASS + 1 FAIL = NO-GO-G0**。G0-4 (integrated-completes) FAIL on heihe_x16 MALFORMED 是 driving gate,但 G0-5 per-step PASS 在 production reality 上被 heihe_x4 total wall 15.1× regression 反驳,verdict reality 强化 NO-GO 解读。
2. **H1 (default-compat) + H2 (telemetry-real) 顺利 PASS**,验证 `SHUD_LINSOL=amg` env-var hook + dlopen versioned-soname + Hypre native telemetry drain hook 三者是 production-grade architectural building blocks。
3. **H3 (integrated-AMG wall-beneficial) 在 per-step letter-of-the-law 层 PASS 但在 total wall reality 层 FAIL**:heihe_x4 AMG per_step 0.0929 s 比 SPGMR baseline 0.2384 s 快 0.39×,但 ncfn=100138 Newton 控制失败主导的 nst 38.8× inflation 导致 total wall 15.1× WORSE。
4. **ncfn 主导失效模式 是 G0 first-gate 学到的 critical 现象学发现**:AMG 内核 Krylov 收敛健康 (`ncfl=0` 全 cell) 但 outer Newton 反复 control fail,这一 phenomenon 在 P8-tune.F pattern-only spike 中无法观察 (因 pattern-only 不接 CVODE)。
5. **Telemetry-real signal validated**: keliya `operator_complexity=1.002945` 来自 Hypre native `HYPRE_BoomerAMGGetOperatorComplexity` API, P8-tune.F 16-cell sweep 实测 1.0029 drift 仅 0.0% — Hypre 3.1.0 hierarchy quality 在不同 spike 之间稳定 reproducible。
6. **ADR-0007 strict NO-GO-both 解读 在 integrated 层得到 first-order 经验确认**,amended GO operational 解读被反驳:integrated CVODE 下 ncfn-driven outer-Newton failure 抹掉 pattern-only setup + apply wall axes PASS 的算法优势。

工程方法学贡献:

- **6-gate AND-gate verdict 框架**: 是 P8-tune.F 5-axis verdict 在集成 CVODE context 下的语义扩展,引入 default-compat + build + integrated-completes + solver-stats 四个 operational completeness gates + telemetry-real 一个 instrumentation-quality gate + wall-signal 一个 production-relevance gate。这一框架是 pattern-only → integrated 转化的标准 verdict 模板。
- **2-branch verdict (GO-G0 / NO-GO-G0)** 简化 P8-tune.F 5-branch (GO / Optional / NO-GO-heihe_x16-only / NO-GO-both / BLOCKED) 因 first-gate 不需 detailed reason classification — 任何 FAIL 都推 forward path 到 deferred gate 或 fallback epic。
- **dlopen versioned-soname**: `libHYPRE.so.3.1.0` 避免 `libHYPRE.so` symlink target 在 ubuntu apt vs server source-built 之间 ABI drift,production-grade soname pinning 模式。
- **Ring-buffer telemetry drain hook in CVodeFree pre-callback**: 解决多调用 telemetry 持久化挑战 (PR-B Phase 6 critical fix), thread-safe + bounded-memory + transparent overflow reporting via `dropped_overflow` counter。
- **byte-identical anchor contract**: 沿用 P8-tune.F amg-pattern-spike-verdict REQ-6 模板,保 aggregator stdout 与 verdict doc 之间零 hand-curated drift; awk `/G0_VERDICT_BEGIN/,/G0_VERDICT_END/` extraction + diff 实现 zero-effort verifiability。
- **ADR-0007 append-only Amendment 模式**: G0 verdict 通过 dated `## Amendment 2026-06-30 (G0 verdict)` block 追加,不动 §Status / §Decision 不立新 ADR-0008,沿用 PR #407 framing amendment 单-ADR-append 先例,保 ADR-0007 decision lineage 一致 + ADR governance 不爆炸。

forward action 锁定:**§P8-tune.G0 [CLOSED]; §P8-tune.G1 + §G2 [CLOSED-DEFERRED, pending P8-tune.H GPU sparse fallback evaluation]; §P8-tune.H spike 由 orchestrator decision tree 触发立项**。AMG production path closed at G0 gate; `SHUD_LINSOL=spgmr` default preserved 无 user-facing 行为变化; `SHUD_LINSOL=amg` 保留 research opt-in knob 但 NOT recommended for production。

ADR-0007 strict-vs-amended 二分 verdict 在 integrated 端得到 first-order 确认 — strict NO-GO-both 是 production-grade reality,amended GO 是 pattern-only artifact。G0 verdict 不 re-litigate ADR-0007 Accepted decision,而是 通过 append-only Amendment 添加 integrated-CVODE evidence 强化 §Decision 的 forward-action 解读。

---

# §9 Future Work / 未来工作

## §9.1 §P8-tune.H — GPU sparse / domain decomposition spike (next investigation, conditional on orchestrator trigger)

P8-tune.H 由 G0 NO-GO verdict 触发立项 (per ADR-0007 §Amendment 2026-06-30 §Forward action update)。Scope 候选 (orchestrator-final decision pending GPU partition + workload audit):

1. **CUDA sparse direct via cuSPARSE + iterative refinement**: cuSPARSE 5.0+ `cusparseSpSV` direct + IR refinement。GPU-presence gate (`sinfo -p GPU` → `gn01` available) 是 precondition。
2. **GPU AMG with mixed-precision (Hypre 2.30+ cusparse_use=1)**: 测 FP32 V-cycle apply + FP64 outer Newton 是否绕过 G0 ncfn 主导失效模式 (假设 mixed-precision 引入的 noise 不破坏 outer Newton convergence)。
3. **Domain decomposition + KLU per subdomain**: cn-node 32-core OpenMP + 32 subdomain × KLU (ADR-0005 已证 keliya/heihe 可行) + Schwarz alternating + few outer Krylov iterations。

P8-tune.H epic 立项形式 (sub-issues / openspec / ADR-0008?) 待 orchestrator decide。本 G0 verdict 不 commit 具体 path。

## §9.2 §P8-tune.G0 server array re-run for heihe_x16 (Future Work)

orchestrator-spawned server Slurm re-run with 24h budget + post-Phase-6 wrapper (SHUD `188854b`) + drain hook fully wired 可 close G0-4 evaluation gap on heihe_x16 (升级 verdict from MALFORMED 到 AMG_WALL_OVERFLOW or AMG_OK)。但 §7.2 + §6.1 已 demonstrate **不论 heihe_x16 verdict 如何,G0_OVERALL=NO-GO 仍 hold**, 因 heihe_x4 15.1× total regression independently disqualifies AMG production path。

re-run 主要价值是:
- 区分 AMG_WALL_OVERFLOW_INFERRED_SIGKILL 与 MALFORMED_RUNNER_BUG, 输入 future P8-tune.H epic 决策
- 完整 6-gate dataset 用于 ADR-0007 §Amendment 2026-06-30 evidence completeness audit

## §9.3 §P8-tune.G1 / §G2 re-activation conditional path

§P8-tune.G1 (18-cell integrated benchmark) + §G2 (A5 hydrology equivalence) 目前 [CLOSED-DEFERRED]。Re-activation 条件:

- **path A (AMG fix)**: 若 P8-tune.H mixed-precision GPU AMG 或 domain-decomposition + KLU 走通 + 解决 ncfn 主导失效模式, §G1 可 re-activate 测 wall improvement,§G2 测 A5 equivalence。
- **path B (alternative substrate)**: 若 P8-tune.H 在 GPU 上不走通,SHUD 大 case 求解路径 close at SPGMR + maxl=30 Optional-knob (per ADR-0004) 作为 production ceiling, future production-scale work focus on case-truncation strategy (e.g., 30-day deliverable instead of 90-day SHORT for forecasts) 而不是 architectural solver substrate。

§G1/§G2 re-activation 由 orchestrator-final decision 触发,不由 P8-tune.H 子epic 内部触发 (避免 epic chain explosion)。

## §9.4 Spec REQ-5 G0-5 letter-of-the-law 修订 (deferred to future spec amendment)

§6.2 揭示 G0-5 spec REQ-5 per-step semantic 与 production reality 分歧。Future spec amendment 应:

- 把 G0-5 OR-gate 从 `amg_wall_per_step < spgmr_wall_per_step` 改为 `amg_total_wall < spgmr_total_wall AND amg_nst < N × spgmr_nst (N=1.5 reasonable upper bound)` joint criterion
- 或保留 per-step OR-gate 但 引入第二 G0-5b OR-gate `amg_total_wall ≤ 1.1 × spgmr_total_wall` 作为 production-relevance gate

本 PR-C 不修 spec/aggregator (与 PR-B scope 干扰风险高)。Spec amendment 由 future P8-tune.H epic 在引入新 case-asymmetric verdict criteria 时一并 propose。

## §9.5 case-asymmetric solver policy productionization (long-range, post P8-tune.H)

ADR-0007 §Discussion §"Case-asymmetric scaling third-epic anchor" 提及 future epic 应 implement case-asymmetric solver policy。G0 NO-GO verdict 反过来 informs:**case-asymmetric policy 在 AMG 路径上 由 ncfn-driven failure mode 阻塞**,即使 mid-large NumY (heihe_x4) 也不应 default to AMG。

Future case-asymmetric policy (post P8-tune.H) 应:
- 保留 `SHUD_LINSOL=spgmr` (default) + `SHUD_KLU_ENABLE=1` (per ADR-0005, keliya/heihe small case opt-in) + `SHUD_LINSOL=cusparse|ddsparse` (per P8-tune.H, large case opt-in if walks through)
- NOT include `SHUD_LINSOL=amg` as production recommendation
- Implementation: `cvode_config.cpp` `linsol_select(NumY)` lookup table + env-var override

## §9.6 长程 (P9+) 方向

G0 数据点也启发以下 P9+ 长程方向:

- **production-scale heihe_x64 (NumY ≈4M)**: P8-tune.F memory 实测最大 400 MB ≪ 121 GiB,即使 NumY 增 8× 至 4M, memory 仍 feasible 但 wall 不确定。Future epic 可在 heihe_x64 mesh 上重复 G0 6-gate methodology + 改用 P8-tune.H 走通的 architectural substrate。
- **P9-A5 cross-platform tier with AMG-or-alternative**: G0 G2 [CLOSED-DEFERRED] means SHUD 当前 A5 hydrology equivalence 永远以 SPGMR 为 substrate。Future P9 epic 若 P8-tune.H walks through 应 extend cross-platform A5 to GPU/DD path。
- **ncfn-driven outer-Newton failure mode 学术深度调研**: G0 first-time observation 这一 phenomenon, 是否 in CVODE-AMG interaction 文献中已有 prior reporting?Saad 2003 §13 + Henson & Yang 2002 不直接 cover BDF/Newton + AMG-preconditioner outer-loop interaction; future 学术 reading 应在 Wieners 2010 / Falgout 2002 / Gee 2012 上深入。

---

# §10 References / 参考文献

## 内部 docs

- [docs/p8tune/amg_g0_verdict.md](docs/p8tune/amg_g0_verdict.md) — G0 capstone verdict source-of-truth (6-gate + wall-signal table + solver-stats table + telemetry summary + limitations + next steps)
- [docs/p8tune/amg_spike_verdict.md](docs/p8tune/amg_spike_verdict.md) — 前序 P8-tune.F 16-cell pattern spike verdict (strict NO-GO-both / amended GO 二分)
- [docs/p8tune/p8tune_f_academic_summary.md](docs/p8tune/p8tune_f_academic_summary.md) — 前序 P8-tune.F academic summary 母本 (学术风格 template per user pref 2026-06-25)
- [docs/p8tune/p8tune_d_academic_summary.md](docs/p8tune/p8tune_d_academic_summary.md) — 前序 P8-tune.D KLU spike academic summary
- [docs/p8tune/klu_spike_verdict.md](docs/p8tune/klu_spike_verdict.md) — 前序 P8-tune.D KLU verdict (Case-aware)
- [docs/p8tune/maxl_sweep_verdict.md](docs/p8tune/maxl_sweep_verdict.md) — 前序 P8-tune.C SPGMR maxl verdict (Optional-knob)
- [docs/p8tune/clean_prec_none_baseline.md](docs/p8tune/clean_prec_none_baseline.md) — PREC_NONE production baseline
- [docs/adr/0007-amg-spike-decision.md](docs/adr/0007-amg-spike-decision.md) — Accepted; §Status + §Decision preserved; §Forward action Amendment 2026-06-30 (G0 verdict) appended in this PR-C
- [docs/adr/0005-klu-spike-decision.md](docs/adr/0005-klu-spike-decision.md) — Case-aware; AMG path 触发起点
- [docs/adr/0004-maxl-sweep-decision.md](docs/adr/0004-maxl-sweep-decision.md) — Optional-knob; SPGMR baseline anchor
- [docs/adr/0003-precond-spike-decision.md](docs/adr/0003-precond-spike-decision.md) — PREC_NONE production baseline
- [docs/adr/0002-solver-path.md](docs/adr/0002-solver-path.md) — Path 4 KLU + Path 5 AMG 决策起点
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/proposal.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/proposal.md) — Epic proposal
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/design.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/design.md) — Epic design (D1-D9 architectural decisions)
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/tasks.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/tasks.md) — Epic tasks 1-8 (PR-0 to PR-D)
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md) — Capability spec: 6-gate verdict + cell_summary KV schema + verdict_class enum + byte-identical anchor + Amendment block + master plan close
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/shud-linsol-selector/spec.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/shud-linsol-selector/spec.md) — Capability spec: SHUD_LINSOL env-var contract
- [openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/sunlinsol-hypre-wrapper/spec.md](openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/sunlinsol-hypre-wrapper/spec.md) — Capability spec: Hypre thread-pinning + 15-callback ABI + Setup/Solve telemetry emit
- [SHUD_openMP_master_plan.md §P8-tune.G0 / §G1 / §G2](SHUD_openMP_master_plan.md) — §G0 [CLOSED] + §G1/§G2 [CLOSED-DEFERRED]
- [docs/p1e/p1e_academic_summary.md](docs/p1e/p1e_academic_summary.md) — P1e 学术 summary 母本 (user pref 2026-06-25)

## 代码与 evidence

- [tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp](tools/p8tune.G0/sunlinsol_hypre_wrapper.cpp) — PR-0 wrapper + PR-B Phase 6 drain hook
- [tools/p8tune.G0/aggregate_g0_smoke.sh](tools/p8tune.G0/aggregate_g0_smoke.sh) — PR-B aggregator (6-gate evaluation + byte-identical anchor emit)
- [tools/p8tune.G0/spgmr_baseline_walls_g0.h](tools/p8tune.G0/spgmr_baseline_walls_g0.h) — PR-A measured + PR-B hot-patched case-specific baselines
- [tools/p8tune.G0/precheck_env.sh](tools/p8tune.G0/precheck_env.sh) — PR-A env gate (hypre version + cn-node memory + SHUD_LINSOL hook)
- [tools/p8tune.G0/g0_amg_smoke_array.sbatch](tools/p8tune.G0/g0_amg_smoke_array.sbatch) — PR-A Slurm 4-cell dispatcher
- [.review-evidence/g0-amg-smoke-array-rerun/](/.review-evidence/g0-amg-smoke-array-rerun/) — PR-B evidence (4 cells × 90-day SHORT + telemetry TSV + aggregate.tsv + README.md)
- [.review-evidence/g0-spgmr-baseline-90day-keliya/](/.review-evidence/g0-spgmr-baseline-90day-keliya/) — PR-0 Mac baseline + PR-A server baseline (G0-1 evidence)

## Pull Requests (epic #408)

- [PR-0 #414](https://github.com/DankerMu/SHUD-OpenMP/pull/414) — `feat(p8tune.G0)` SUNLinSol_Hypre wrapper + Hypre link + linsol selector + Mac G0-1 baseline
- [PR-A #415](https://github.com/DankerMu/SHUD-OpenMP/pull/415) — `feat(p8tune.G0)` 4-cell AMG integrated smoke + Slurm sbatch + dlopen versioned-soname
- [PR-B #416](https://github.com/DankerMu/SHUD-OpenMP/pull/416) — `feat(p8tune.G0)` HYPRE Axis-4 telemetry + aggregator + G0-3/4/5/6 verdict markers + Phase 6 drain hook (with SHUD `188854b` hot-fix + outer pointer-bump `4ade422`)
- PR-C #<this PR> — `docs(p8tune.G0)` G0 NO-GO verdict + ADR-0007 amendment + master plan §G0 [CLOSED] + academic summary
- PR-D #<TBD> — forthcoming capstone-merge `baseline/p8tune-amg-g0-spike → main` (HARD-GATED behind PR-C merge)

## 关联 issue

- [#408](https://github.com/DankerMu/SHUD-OpenMP/issues/408) — epic p8tune-g0-instrumented-amg-smoke (closing via PR-D)
- [#409](https://github.com/DankerMu/SHUD-OpenMP/issues/409) — PR-0 sub-issue (CLOSED via PR-0 #414)
- [#410](https://github.com/DankerMu/SHUD-OpenMP/issues/410) — PR-A sub-issue (CLOSED via PR-A #415)
- [#411](https://github.com/DankerMu/SHUD-OpenMP/issues/411) — PR-B sub-issue (CLOSED via PR-B #416)
- [#412](https://github.com/DankerMu/SHUD-OpenMP/issues/412) — PR-C sub-issue (本 PR; CLOSED via PR-D auto-close after capstone merge)
- [#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386) — SHUD Model_Data dtor UB (CLOSED via P8-tune.F PR-0 #394; inherited fix)

## 外部依赖

- Hypre 3.1.0 (`https://github.com/hypre-space/hypre`, LLNL; Ubuntu 24.04 apt `libhypre-dev` + server `/scratch/.../hypre-3.1.0/` source-built)
- SUNDIALS-CVODE 6.0.0 (LLNL, pinned, `SUNLinSol_Hypre` adapter API)
- AutoSHUD / rSHUD v2.5.0 (mesh generation; reused from P8-tune.D for heihe_x4/heihe_x16 deployment)
- CMFD forcing dataset V0200 (1951-2024, 0.1° global)

## 学术参考

- [Henson & Yang 2002] V. E. Henson & U. M. Yang, "BoomerAMG: A parallel algebraic multigrid solver and preconditioner," *Applied Numerical Mathematics*, Vol. 41(1), pp. 155-177, 2002. (BoomerAMG canonical reference + operator complexity bound)
- [Saad 2003] Y. Saad, *Iterative Methods for Sparse Linear Systems, 2nd Ed.*, SIAM, 2003. §13 (multigrid methods). (V-cycle bound + Axis 4 demotion rationale)
- [Falgout & Yang 2002] R. D. Falgout & U. M. Yang, "hypre: A Library of High Performance Preconditioners," in *Computational Science - ICCS 2002*, Springer LNCS 2331. (Hypre toolkit overview)
- [Brown et al. 2000] P. N. Brown, A. C. Hindmarsh, L. R. Petzold, "Consistent initial condition calculation for differential-algebraic systems," *SIAM J. Sci. Comput.*, 19(5), 1998. (CVODE BDF Newton-Krylov interaction; relevant to ncfn-driven failure mode in §6.3)
- [Wieners 2010] C. Wieners, "On the superconvergence in computational electromagnetics," in *Sparse Solvers*, Springer LNCSE 78, 2010. (AMG outer-Newton interaction baseline reading for future research)
- [Briggs et al. 2000] W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial, 2nd Ed.*, SIAM, 2000. (Introductory multigrid background)
- [Ruge & Stüben 1987] J. W. Ruge & K. Stüben, "Algebraic Multigrid (AMG)," in *Multigrid Methods*, S. F. McCormick, ed., SIAM, 1987. (Classical Ruge-Stüben coarsening foundation)

---

**Execution Summary (本 capstone 文档生成)**: agents=0 (orchestrator-direct write, leaf implementer subagent boundary per CLAUDE.md PR-C scope); skills=纯文档写作; tools=Read/Write/Edit/Bash; verification=参照 docs/p8tune/p8tune_f_academic_summary.md 母本结构 + ADR-0007 + spec REQ-6 byte-identical contract + 4-cell aggregate.tsv 数据交叉核; limits=本文档作 P8-tune.G0 epic 学术 capstone, 不替代 docs/p8tune/amg_g0_verdict.md (verdict source-of-truth) 与 docs/adr/0007-amg-spike-decision.md (architectural decision authority); ncfn-driven outer-Newton failure mode 在 §6.3 + §9.6 标 future academic reading 方向。
