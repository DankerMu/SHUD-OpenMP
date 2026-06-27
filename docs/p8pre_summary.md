---
title: "p8pre-spike epic capstone — engineer-style summary"
date: 2026-06-27
status: "Epic CLOSED 2026-06-27 — Step 1 PROCEED PASS + Step 2 NO-GO per ADR-0003"
epic: "SHUD-OpenMP#338"
related_docs:
  - "docs/p8pre/capstone.md (epic-level academic-paper-style capstone, parallel doc)"
  - "docs/adr/0003-precond-spike-decision.md (NO-GO decision rationale)"
  - "docs/p8pre/n8_profile_baseline.md (Step 1 PR-C capstone, gate-4 anchor)"
  - "docs/p8pre/identity_spike_verdict.md (Step 2 PR-F verdict adjudicator)"
  - "SHUD_openMP_master_plan.md §P8-precond.0 (本 epic 在 master plan 的 anchor section)"
---

# 概览

p8pre-spike epic (SHUD-OpenMP#338) 是 P1e epic close (2026-06-25) 后接续启动的 ROI 量化 spike, 目的是回答 ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) 是否值得在 P2 阶段后续 epic 启动的问题。epic 通过两步实证: Step 1 量化 SPGMR 工作量 ROI 比 `r = nfeLS / nfe` + 锁 gate-4 wall baseline anchor, Step 2 wire SUNDIALS PREC_LEFT identity preconditioner stub 测 4 hard gate + 2 soft gate。**结论 NO-GO** — identity preconditioner 提供零 SPGMR convergence 加速 ROI; design D8 fall-back PREC_NONE 还原推 #349 archive 或 separate cleanup PR (PR-G #348 doc-only, 不动 SHUD 源码)。

# Step 1 PR-A baseline summary

Step 1 由 PR-A (#341, intake) + PR-A run (#341, 18-cell Slurm Mode C profile execution) + PR-B (#342, aggregator + ROI verdict) + PR-C (#343, academic-paper-style capstone) 4 PR 组成。在 SHUD pin `7a1dc8f` (Step 0 #350 fix doc + .pr-i-runs/→.p1e-i-runs/ rename + profile bucket-sum invariant 修复) 之后, server cn14/cn15 用 `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` build, 跑 18-cell (2 case heihe + heihe_x4 × 3 N {1,4,8} × 3 rep) 矩阵。aggregator 验证 10/10 cross-N invariance Δ=0 strict (`{nst, nfe, nfeLS, nni, nsetups}` 跨 N1/N4/N8 完全相同), 4/4 P1e absolute baseline anchor 满足 (`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741`)。ROI 量化: `r_min = nfeLS_median/nfe_median |_{heihe, N=8} = 12632/6943 = 1.819`, `r_max = nfeLS_median/nfe_median |_{heihe_x4, N=8} = 4.526`, 均超 ADR-0002 Path 3 trigger threshold 1.5 → **branch a (PROCEED Step 2)**。`wall_step1_baseline_median(case, N)` 6-row 表 archived to `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1, 作 Step 2 hard gate 4 wall non-regression 的 anchor。

# Step 2 spike outcome (4 hard gate + 2 soft gate mini-table)

Step 2 由 PR-D #357 (impl: `MD_precond_identity.{h,cpp}` 40+61 lines + `cvode_config.cpp:259` PREC_NONE→PREC_LEFT + `CVodeSetPreconditioner` + `CVodeSetLSetupFrequency(50)`; SHUD pin `7a1dc8f → 5276167` forward-only descendant) + PR-E #358 (18-cell Slurm spike run JID 9531-9548 全 ExitCode 0) + PR-F #359 (aggregator + 6-gate verdict) 3 PR 组成。SHUD canonical reference: `cvDiurnal_kry.c` L716/L760 (jok-mirror PSetup + memcpy PSolve pattern)。

| # | Gate | Result | 关键证据 |
|---|---|---|---|
| H1 | Build (3-symbol nm) | **PASS** | `PSetupIdentity` + `PSolveIdentity` + `CVodeSetPreconditioner` 同时 resolved at `server_nm.log` |
| H2 | `ncfn = 0` 跨 18 cell | **FAIL** | heihe `ncfn=6` × 9/9 + heihe_x4 `ncfn=47` × 9/9 deterministic 不随 N 或 rep 变 |
| H3 | `nps > 0 ∧ npe > 0` per cell | **PASS** | min_nps=18163, min_npe=77 |
| H4 | wall non-regression vs Step 1 baseline | **PASS** | 6/6 (case, N) within ε(heihe)=0.10 + ε(heihe_x4)=0.05 (max heihe N=1 = 2.64%; max heihe_x4 N=1 = 1.09%) |
| S5 | cross-N tolerance (strict SHA OR max_ulp ≤ 1024) | **FAIL** | strict 18/18 violate; A4 fall-back 18/18 violate (max_ulp ≈ 9×10¹⁵; 5,155/214,252 positions structurally diverge) |
| S6 | setup overhead ratio ≤ 0.05 | **PASS** | max 1.01×10⁻⁷ (6 数量级 below threshold) |

H2 hard FAIL deterministic 是 NO-GO 主因 — identity P⁻¹=I 不旋转 residual, 对 SHUD stiff Jacobian 上的 Newton 收敛失败完全无作用。S5 FAIL 揭示 PREC_LEFT 状态机 inherent cost: 仅 wire 通即触发 SUNDIALS `cvLsSolve` 内部额外 `N_VLinearSum` / `N_VScale` ops (per `cvode_spils.c`), 扰动 90 天积分轨迹 ≈9×10¹⁵ ULP 远超 A4 阈值 1024。H1/H3/H4/S6 PASS 仅证 plumbing 已正确接通 + overhead 可忽略, 这些是 "preconditioner 框架可用" 的必要条件而非 "preconditioner 带来收益" 的充分条件 — H2 FAIL 语境下 PASS 的四项无法挽救 NO-GO 结论。

# 决策 (NO-GO per ADR-0003)

PR-G #348 PR 写入 [`docs/adr/0003-precond-spike-decision.md`](adr/0003-precond-spike-decision.md), 决策 NO-GO option (b) per `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L74-79 strict + L106-108 fall-back: identity precond 不进入 production, p8pre-spike epic 关闭, design D8 fall-back PREC_NONE 还原推后续 cleanup PR。

**Rationale (TL;DR)**:

1. H2 deterministic FAIL (heihe 6 + heihe_x4 47 floor) 证 identity P⁻¹=I zero ROI
2. S5 FAIL 揭示 PREC_LEFT 状态机 inherent fp64 drift (5,155/214,252 positions ≈9×10¹⁵ ULP)
3. Step 1 canonical `nfeLS/nfe = 1.819` ROI window 存在但需 real preconditioner candidate (not identity)
4. KISS / YAGNI 不允许 dead PREC_LEFT codepath 留 production
5. spec L74-79 hard FAIL → spike NO-GO → formal P8-precond.1-.7 epic NOT to be opened under current data

# 后续 work

**Immediate (deferred to separate cleanup PR or #349 archive)**:

- design D8 fall-back PREC_NONE 还原: revert `cvode_config.cpp:259` 回 `PREC_NONE` + 删 `CVodeSetPreconditioner` + 删 `CVodeSetLSetupFrequency(50)` + 删 `MD_precond_identity.{h,cpp}` + Makefile unlink + (optional) 删 `t_precond_setup` Timer bucket; bump outer pointer 从 `5276167` 到新 SHUD HEAD (forward-only descendant)
- close `baseline/p8pre` 分支 (HEAD `df45deb`); future P8-precond epic 应 re-fork from `main` 干净 prior-art base
- spec L26 wording correction (jok-mirror canonical cite SUNDIALS `cvDiurnal_kry.c` L716/L760) per PR-D #357 review-loop-log F-R2-3 deferred → #349 archive scope

**Medium-term P8-precond formal epic re-evaluation (NOT to be opened under current spike data)**:

- pre-spike 选 real preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi physics-based) + 量化对 `ncfn` floor 降幅预期
- re-evaluate cost/risk (原 ADR-0002 Path 3 2-3 epic-week 估算含 identity 路径 ROI 不为零假设; ROI 实测=0 后需重估)
- 接受 S5 PREC_LEFT vs PREC_NONE structural drift baseline (≈9×10¹⁵ ULP); 设计 mode C-precond reference SHA 重新固化路径

**Alternative P8-tune candidate path** (per ADR-0003 §Forward action recommendations §3 + spec §9 Future Work):

- P8-tune.A: CVODE step controller (`max_step` / `min_step` / `nonlin_conv_coef`) 调优, 把 `ncfn` retry 吸收进 successful steps
- P8-tune.B: `CVodeSetMaxNonlinIters` 增加, 看 6/47 deterministic floor 是否 Newton residual stall 可由更多迭代 resolve
- P8-tune.C: SPGMR `maxl` 参数 sweep (SUNDIALS 默认 5; raise to 10-15 压 `ncfl=121` 重启次数)
- P8-tune.D: ADR-0002 Path 4 KLU direct solver pattern-only spike (per ADR-0002 §References ADR-0003 KLU spike forthcoming; 与本 ADR 是 distinct ADR)

**epic value summary**: 虽然 NO-GO, p8pre-spike epic 价值 = (i) framework readiness (PSetup/PSolve canonical SUNDIALS pattern + Timer instrumentation 已固化, future preconditioner candidate 可直接复用骨架), (ii) ROI ceiling 数据库 (`ncfn` floor 6/47 + Step 1 canonical `nfeLS/nfe` 比值 1.819/4.526 + S5 drift baseline 5,155/214,252 positions), (iii) Negative result 形式化记录 (avoids future epic 重复试错 identity 路径)。

# Forward note (post-cleanup 2026-06-27)

design D8 fall-back PREC_NONE 还原已 completed at outer `e442ce8` / SHUD `37be0fe` (cleanup pointer bump merged to `main` 2026-06-27)。`PREC_LEFT + identity` 路径的 `ncfn = 6 (heihe) / 47 (heihe_x4)` floor 仅作 negative-control anchor 保留; 任何 future preconditioner / solver-tune candidate 的 PASS gate 应使用 cleaned-PREC_NONE baseline `ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)` per `docs/p8pre/n8_profile_verdict.md` §3.1。下一步 P8-tune.C SPGMR `maxl` sweep 在 change [`p8tune-spgmr-maxl`](../openspec/changes/p8tune-spgmr-maxl/proposal.md) (4 capabilities: `p8pre-doc-state-correction` / `clean-prec-none-baseline` / `spgmr-maxl-env-hook` / `maxl-sweep-verdict`) 中形式化, 决策入口 ADR-0004 (TBD, by PR-E)。

# References

## Internal docs

- [`docs/adr/0003-precond-spike-decision.md`](adr/0003-precond-spike-decision.md) (PR-G #348, NO-GO 决策 rationale)
- [`docs/p8pre/capstone.md`](p8pre/capstone.md) (PR-G #348, epic-level academic-paper-style capstone, 与本 doc 并列)
- [`docs/p8pre/n8_profile_baseline.md`](p8pre/n8_profile_baseline.md) (Step 1 PR-C #355 capstone, gate-4 anchor)
- [`docs/p8pre/n8_profile_run.md`](p8pre/n8_profile_run.md) (Step 1 PR-A #353 18-cell execution log)
- [`docs/p8pre/n8_profile_verdict.md`](p8pre/n8_profile_verdict.md) (Step 1 PR-B #354 ROI verdict aggregator, branch a PROCEED)
- [`docs/p8pre/identity_spike_run.md`](p8pre/identity_spike_run.md) (Step 2 PR-E #358 18-cell execution log)
- [`docs/p8pre/identity_spike_verdict.md`](p8pre/identity_spike_verdict.md) (Step 2 PR-F #359 verdict adjudicator)
- [`docs/adr/0002-solver-path.md`](adr/0002-solver-path.md) (Path 3 deferred 状态; ADR-0003 是 Path 3 trigger 第 3 条件的 NO-GO 决策记录, Path 3 仍保留 `P2 optimization` 标位)
- [`SHUD_openMP_master_plan.md`](../SHUD_openMP_master_plan.md) §P8-precond.0 (本 epic 在 master plan 的 anchor section)

## OpenSpec

- [`openspec/changes/p8pre-spike/proposal.md`](../openspec/changes/p8pre-spike/proposal.md)
- [`openspec/changes/p8pre-spike/design.md`](../openspec/changes/p8pre-spike/design.md) D5/D7/D8
- [`openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md`](../openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md) L74-130

## GitHub PR sequence

- Step 0 (intake bookkeeping fix): #339 / merge #350
- Step 1 PR-A (intake + 18-cell run): #341 / merge #353
- Step 1 PR-B (aggregator + ROI verdict): #342 / merge #354
- Step 1 PR-C (Step 1 capstone): #343 / merge #355
- Step 2 PR-D (pre-flight): #344 / merge #356
- Step 2 PR-D (impl): #345 / merge #357
- Step 2 PR-E (data capture): #346 / merge #358
- Step 2 PR-F (verdict adjudication): #347 / merge #359
- Step 2 PR-G (ADR-0003 + master plan + epic capstone — 本 PR): #348 (this PR)
- Step 2 PR-H (epic archive): #349 (forthcoming)

## External

- SUNDIALS-CVODE 6.0.0 canonical: `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760 (jok-mirror PSetup + memcpy PSolve pattern; identity stub follows same contract)
- SUNDIALS-CVODE 6.0.0 `src/sundials/cvode_spils.c` `cvLsSolve` (extra `N_VLinearSum` / `N_VScale` op triggered by `PREC_LEFT` 状态机, S5 structural drift root cause)
- Davis T.A. *Direct Methods for Sparse Linear Systems* SIAM (2006) — ADR-0002 Path 4 KLU 基础参考 (alternative P8-tune.D 候选)

---

*Generated by PR-G #348 implementer 2026-06-27 (epic SHUD-OpenMP#338 capstone)*
