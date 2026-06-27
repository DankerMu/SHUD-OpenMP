# ADR-0003: precond-spike-decision — identity preconditioner NO-GO + design D8 PREC_NONE fall-back

- **Status**: Accepted (p8pre-spike Step 2 close, 2026-06-27)
- **Date**: 2026-06-27
- **Deciders**: DankerMu + Claude orchestrator (per `tools/p8pre/aggregate_identity_spike.sh` verdict + `docs/p8pre/identity_spike_verdict.md` adjudication)
- **Owner**: SHUD-OpenMP 改造工程 / p8pre-spike epic capstone → P8-precond formal epic intake (NOT opened — NO-GO)
- **Tags**: precond / spike / NO-GO / PREC_LEFT / PREC_NONE / SPGMR / ROI-gating
- **Supersedes**: none (first precond-related ADR)
- **Superseded by**: none
- **Related**: ADR-0002 Path 3 (deferred → spike trigger) + master plan §P8-precond.0 + `openspec/changes/p8pre-spike/` (epic SHUD-OpenMP#338) + PR-F #359 verdict adjudicator + PR-G #348 (本 ADR 所在 PR)

---

## Context

p8pre-spike epic (SHUD-OpenMP#338) 在 P1e epic close 后立项，目的是回答 ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) 是否值得作为 P2 阶段后续 epic 启动的问题。Path 3 在 ADR-0002 中标记为 `P2 optimization (paired with Path 1)`，trigger condition 为 (i) ROI 量化 `r = nfeLS/nfe ≥ 1.5`、(ii) identity-precond API wire-up 5+1 hard/soft gate PASS、(iii) ADR-0003 GO 决策。本 epic 通过两步实证回答前两个 trigger，并产出本 ADR 关闭第三个。

**Step 1 (PR-A/B/C, #341-#343, #353-#355)** — N=8 Mode C profile recheck (18-cell 2×3×3 矩阵, SHUD pin `7a1dc8f`, server cn14/cn15)：

- 实测 ROI 比 `r = nfeLS / nfe`:
  - heihe N=8: 12120 / 6696 = **1.811** (基本贴近 ADR-0002 Path 3 trigger threshold 1.5)
  - heihe_x4 N=8: 30518 / 6741 = **4.526** (远超 trigger threshold)
- `wall_step1_baseline_median(case, N)` (gate-4 anchor) 6-row 表落地于 `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1
- 4 baseline counter anchor 全部满足 (`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741`)
- **Step 1 verdict**: branch a (PROCEED Step 2) — Path 3 ROI trigger 第一个条件 PASS

**Step 2 (PR-D/E/F, #344-#347, #356-#359)** — identity precond stub + cvode_config PREC_LEFT wire + 4-hard-gate + 2-soft-gate verdict：

- PR-D #357: `MD_precond_identity.{h,cpp}` 40+61 lines (RAII Timer `t_precond_setup` + jok-mirror PSetup matching SUNDIALS `cvDiurnal_kry.c` L716 pattern + memcpy PSolve matching L760 pattern) + `cvode_config.cpp:259` patch (`SUN_PREC_NONE` → `SUN_PREC_LEFT` + `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` + `CVodeSetLSetupFrequency(cvode_mem, 50)`). SHUD pin bumped `7a1dc8f` → `5276167` (forward-only descendant).
- PR-E #358: 18-cell Slurm spike run (cn14 heihe + cn15 heihe_x4 × N∈{1,4,8} × 3 reps, JIDs 9531-9548), 全 18 cell ExitCode 0; 数据归档于 `/tmp/p8pre_identity_spike/<cell>/` + `.review-evidence/p8pre-pr-e-spike/cell_stats.txt`.
- PR-F #359: `tools/p8pre/aggregate_identity_spike.sh` (569 行 POSIX bash + awk + sha256sum + `uv run python` numpy ULP) 计算 4 hard gate + 2 soft gate verdict, 详 `docs/p8pre/identity_spike_verdict.md` §4-§5.

**6 gate 实测结果**：

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| H1 | Build PASS | server `nm ./shud` 3 symbol (`PSetupIdentity` + `PSolveIdentity` + `CVodeSetPreconditioner`) | **PASS** | `server_nm.log` 3 hits |
| H2 | Zero convergence failure | `ncfn = 0` 跨 18 cell | **FAIL** | heihe `ncfn=6` 跨 9/9 cell + heihe_x4 `ncfn=47` 跨 9/9 cell (deterministic 不随 N 或 rep 变) |
| H3 | nps + npe accumulation | `nps > 0` AND `npe > 0` 每 cell | **PASS** | min_nps=18163, min_npe=77 |
| H4 | Wall non-regression | per (case, N): `|wall_identity − wall_step1| / wall_step1 ≤ ε(case)`, `ε(heihe)=0.10`, `ε(heihe_x4)=0.05` | **PASS** | 6/6 (case, N) within tolerance (max heihe N=1 = 2.64% < 10%; max heihe_x4 N=1 = 1.09% < 5%) |
| S5 | Cross-N tolerance | per (case, N, rep): SHA12 == baseline strict; OR max_ulp ≤ 1024 | **FAIL** | strict: 18/18 violate; fall-back: 18/18 violate (max_ulp ≈ 9×10¹⁵ ≫ 1024; 5,155/214,252 positions structurally diverge) |
| S6 | Setup overhead | per cell: `t_precond_setup / t_wall_total ≤ 0.05` | **PASS** | max ratio 1.01×10⁻⁷ (6 数量级 below threshold) |

H2 / S5 双 FAIL deterministically：H2 揭示 identity P⁻¹=I 提供零 SPGMR 收敛加速 ROI (preconditioner 仅旋转 residual，identity 不旋转 → SPGMR 走 N_VDotProd / N_VWSqrSumLocal 时 Newton 仍然 stall 同一频次)；S5 揭示 PREC_LEFT 仅仅 wire 通即在 SUNDIALS `cvLsSolve` 内部触发额外 `N_VLinearSum` / `N_VScale` op (per `cvode_spils.c`), 在 fp64 浮点累计意义上扰动 `rivqdown.dat` 5,155/214,252 = 2.4% 位置发生 nonzero-zero rotation — 结构性差异 (非纯 reduction-order drift)。

H1 / H3 / H4 / S6 PASS 仅证 (i) plumbing 已正确接通、(ii) identity stub overhead 可忽略——但这些都是 "preconditioner 框架可用" 的**必要**条件，不是 "preconditioner 带来收益" 的**充分**条件。在 H2 hard FAIL 的语境下，PASS 的四项无法挽救 NO-GO 结论。

---

## Decision

**采纳 NO-GO option (b)** per `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L74-79 strict + L106-108 fall-back：**identity precond 不进入 production，PR-G 关闭 p8pre-spike epic，并在 future cleanup PR 执行 design D8 fall-back PREC_NONE 还原**。

**Rationale**：

1. **Hard gate 2 deterministic FAIL**：`ncfn` 跨 18 cell 完全不变 (heihe 6/6/6 跨 N1/N4/N8 × 3rep, heihe_x4 47/47/47 跨 N1/N4/N8 × 3rep), 证明 identity P⁻¹=I 对 SHUD stiff Jacobian 上的 Newton 收敛失败完全无作用——这与理论一致 (identity 不旋转 residual)。任何 future preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi) 必须把 `ncfn` 压到这两个 floor 以下才有 ROI ground，identity 路径不提供这种 ground (实际是 PREC_NONE 等价数学行为)。
2. **Soft gate 5 FAIL 揭示 PREC_LEFT 状态机 inherent cost**：仅仅启用 `SUN_PREC_LEFT` 即在 SUNDIALS 内部触发额外 N_VLinearSum + N_VScale ops, 扰动 90 天积分轨迹 ≈9×10¹⁵ ULP, 远超 A4 阈值 1024。这是 SUNDIALS 内部架构的客观 cost — future preconditioner candidate 在 evaluate 时必须接受这个 baseline drift, 或者其 mode A reference 必须重新固化 (per S5 carve-out 推 PR-G + #349 archive)。
3. **`nfeLS/nfe = 1.811` ROI 仍然 promising 但 identity stub 无法 unlock**：实测比值贴 ADR-0002 Path 3 1.5 trigger threshold, 证明 SHUD stiff Jacobian 上 SPGMR 工作量充足 → real preconditioner ROI window 存在 — 但 identity 不是真的 preconditioner, 不能用 identity gate 推断 real candidate 行为。结论是 ROI window 存在但需要不同实现路径 (per Consequences §3 alternative path 候选)。
4. **KISS / YAGNI 不允许携带 dead PREC_LEFT codepath**：如果留 identity 在 production 路径, 必然产生未来 maintainer 误读 "已有 preconditioner" 的语义陷阱; 而留 PREC_NONE + Timer 占位的 dead bucket 同样违反 YAGNI。design D8 fall-back 是 clean 还原, 与 ADR-0002 Path 3 `weak alone` 标注 (which assumes future paired Path 1) 一致。
5. **形式 P8-precond epic 不应在当前数据下开**：spec L74-79 + design D8 写明 hard FAIL → spike NO-GO → 不进 formal epic。Step 1 PR-C capstone (`docs/p8pre/n8_profile_baseline.md` §5.5 + §8) 已 anchor branch a PROCEED 条件, 但 Step 2 PR-F 在 branch a 内部走到 sub-branch NO-GO; 不撤销 Step 1 PROCEED, 只关 Step 2 + 推后 P8-precond epic 触发条件。

---

## Consequences

### Positive

- **plumbing 已验证可用 + canonical SUNDIALS pattern 已固化**: PR-D 的 PSetup/PSolve 实现遵循 SUNDIALS canonical `cvDiurnal_kry.c` L716/L760 jok-mirror + memcpy pattern, 当 future preconditioner candidate (Diagonal / Jacobi / ILU(0)) 启动时可直接复用骨架——不必重新发现 jok 旗语 + ier 三档 return code contract。
- **ROI ceiling 实测落地**: `ncfn = {6 (heihe), 47 (heihe_x4)}` floor 与 `nfeLS/nfe = {1.811, 4.526}` 比值是任何 future preconditioner candidate 必须超越的 baseline。这些数字 + Step 1 PR-A `wall_step1_baseline_median` 共同形成 future iterative-solver tuning 的 anchor (per `docs/p8pre/identity_spike_verdict.md` §6.2 PASS criterion: `ncfn < 6 ∧ ncfn < 47 ∧ wall_overhead ≤ 10%`).
- **SUNDIALS PREC_LEFT 状态机 drift 量化**: S5 carve-out 揭示 PREC_LEFT vs PREC_NONE 在 fp64 意义上结构差异 = 5,155/214,252 positions (2.4%), 这是 ADR-0002 Path 3 future epic 在 cross-mode SHA verification 时不可忽略的实证 baseline。Path 3 实施时不能假设 bitwise neutrality with Mode C strict-omp 当前 baseline.
- **ADR-0002 Path 3 不被否决, 仅 trigger 第三个条件 NO-GO**: ADR-0002 Path 3 trigger 三条件 (ROI + spike + ADR-0003 GO) 中前两个仍然客观存在 (ROI 量化 + plumbing 验证), 仅 ADR-0003 GO 不发出。Path 3 在 ADR-0002 内保留 `P2 optimization` 标位; 但 P8-precond formal epic 重新启动需要新的 design 输入 (per Consequences §3)。

### Negative

- **p8pre-spike epic 7 PR (PR-A through PR-G) + cleanup PR 全部投入未带来直接 production speedup**: epic 总投入 ~3 epic-weeks (intake + Step 1 + Step 2 + capstone), `wall_step1_baseline_median` 是唯一 production-usable artifact (gate-4 anchor for future tuning experiments); 主要价值在 (i) framework readiness + (ii) ROI ceiling 数据库 + (iii) Negative result 形式化记录避免后续重复试错。
- **SHUD pin 暂留 `5276167` 含 identity 代码**: 直到 design D8 fall-back PR 执行前, `5276167` 含未使用的 `MD_precond_identity.{h,cpp}` + `cvode_config.cpp:259` PREC_LEFT 行。short-term cosmetic debt, 但 PR-G 已 doc-only 不动 SHUD 源码; cleanup 推迟到 separate PR 或 #349 archive 完成时。
- **baseline/p8pre 分支未关**: 与 SHUD pin 同 reason 推迟; baseline/p8pre 仍指向 `df45deb` (含 PR-D 后续 commit). PR-G 仅 doc, 不动 branch state.

### Neutral

- **P8-precond.0 prep section in master plan 保留**: §P8-precond.0 仍指向 p8pre-spike epic outcome (本 ADR + Step 1 anchor + Step 2 verdict), 仅其 `unlock condition` 由 "Step 2 ADR-0003 GO 后" 变更为 "future 不同 preconditioner candidate 重新开 epic 后"。formal epic §P8-precond.1-.7 仍保留作为 future-epic 设计骨架 (per Consequences §3)。
- **spec L26 wording correction (jok-mirror canonical cite SUNDIALS `cvDiurnal_kry.c` L716/L760) 推 #349 archive**: PR-D #357 review-loop-log F-R2-3 deferred 仍在 forward debt list, 不在本 PR scope。

---

## Forward action recommendations (NOT executed in PR-G — out of scope per issue #348 "不动 SHUD 源码")

1. **Execute design D8 PREC_NONE fall-back in separate cleanup PR or #349 archive 完成时**:
   - Revert `SHUD/src/Equations/cvode_config.cpp:259` from `SUN_PREC_LEFT` 回 `SUN_PREC_NONE`
   - Remove the `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` call
   - Remove the `CVodeSetLSetupFrequency(cvode_mem, 50)` call
   - Delete `SHUD/src/Model/MD_precond_identity.h` + `SHUD/src/Model/MD_precond_identity.cpp`
   - Unlink them from the `SHUD/Makefile` `shud` / `shud_omp` target
   - Optional: delete the `t_precond_setup` Timer bucket from `tools/profile/timer.cpp` (or 留 `unused_bucket` for future reuse)
   - Bump outer pointer from `5276167` to new SHUD HEAD (forward-only descendant; `openmp-baseline` push)
   - Close `baseline/p8pre` branch (future P8-precond epic 应 re-fork from `main` with clean prior-art base)

2. **P8-precond formal epic NOT to be opened under current design assumptions**: 任何 future P8-precond epic intake 必须先做以下 design pivot:
   - Diagonal / Jacobi / ILU(0) / block-Jacobi physics-based candidate selection (NOT identity)
   - Pre-spike 量化 candidate 对 `ncfn` floor 的预期降幅 (e.g. literature reference + 小规模 prototype)
   - Re-evaluate ADR-0002 Path 3 cost/risk 估计 (原 2-3 epic-week 假设含 identity 路径 ROI 不为零; 实测 ROI = 0 后 cost/risk 需重估)
   - Accept the S5 PREC_LEFT vs PREC_NONE structural drift baseline (≈9×10¹⁵ ULP); 设计 mode C-precond reference SHA 重新固化路径

3. **Alternative P8-tune path 候选 (per spec §9.2 Prerequisite 2-4)**: 如果 P8-precond epic 推迟启动, future epic 可考虑替代路径:
   - **P8-tune.A**: CVODE step controller (`max_step`, `min_step`, `nonlin_conv_coef`) 调优, 把 `ncfn` retry 吸收进 successful steps (per ADR-0002 Path 1 sub-option)
   - **P8-tune.B**: `CVodeSetMaxNonlinIters` 增加, 看 6/47 deterministic floor 是否 Newton residual stall 可由更多迭代 resolve
   - **P8-tune.C**: SPGMR `maxl` 参数 sweep (SUNDIALS 默认 5; raise to 10-15 试图压 `ncfl=121` 重启次数)
   - **P8-tune.D**: ADR-0002 Path 4 KLU direct solver pattern-only spike (per ADR-0002 Path 4 deferred 状态 + ADR-0003 KLU spike forthcoming in ADR-0002 §References)

4. **Spec L26 wording correction (jok-mirror canonical cite)**: PR-D #357 review-loop-log deferred F-R2-3 在 #349 archive scope 内执行——补 `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L26 引用 `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760 作 jok-mirror canonical reference。

---

## References

### Epic + PRs

- Epic: SHUD-OpenMP#338 (p8pre-spike)
- PR-Step0 (intake doc fix): #339 / merge #350
- PR-A (Step 1 prep): #340 / merge #352
- PR-B (Step 1 18-cell run): #341 / merge #353
- PR-C (Step 1 verdict aggregator): #342 / merge #354
- PR-D (Step 1 capstone): #343 / merge #355
- PR-E (Step 2 PR-D impl 实质 = PR for impl, 见 numbering note): #344 / merge #356
- PR-F (Step 2 PR-D impl SHUD changes): #345 / merge #357
- PR-G (Step 2 PR-E data capture): #346 / merge #358
- PR-H (Step 2 PR-F verdict): #347 / merge #359
- PR-I (Step 2 PR-G capstone — 本 PR): #348 (this PR)
- PR-J (epic archive): #349 (forthcoming)

NB on numbering: epic 内 informal PR letter (Step1: PR-A/B/C, Step2: PR-D/E/F/G) 与 GitHub PR # 之间 1:1 mapping per Step 1 = #341/#342/#343 + Step 2 = #344/#346/#347/#348。intake bookkeeping fix #339 是 Step 0 (precedes Step 1 PR-A)。

### OpenSpec

- `openspec/changes/p8pre-spike/proposal.md`
- `openspec/changes/p8pre-spike/design.md` D5 (gate construction) + D7 (gate 5 fall-back to A4) + D8 (PREC_NONE fall-back path)
- `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` (Step 1 spec)
- `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L74-79 (hard gate fall verdict NO-GO) + L102-108 (soft gate 5 A4 fall-back + carve-out) + L120-130 (overall NO-GO criteria)
- `openspec/changes/p8pre-spike/tasks.md` §8 PR-F + §9 PR-G

### Internal docs

- `docs/p8pre/n8_profile_baseline.md` (PR-C #355 Step 1 capstone, gate-4 anchor)
- `docs/p8pre/n8_profile_run.md` (PR-A #353 18-cell execution log)
- `docs/p8pre/n8_profile_verdict.md` (PR-B #354 ROI verdict aggregator, branch a PROCEED)
- `docs/p8pre/identity_spike_run.md` (PR-E #358 18-cell execution log, neutral data + provenance)
- `docs/p8pre/identity_spike_verdict.md` (PR-F #359 verdict adjudicator, 本 ADR 的 primary input)
- `docs/p8pre_summary.md` (PR-G #348 顶层 engineer-style summary, 与本 ADR 并列)
- `docs/p8pre/capstone.md` (PR-G #348 epic-level academic-paper-style capstone)

### SUNDIALS canonical reference

- `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716 (`Precond` jok-mirror PSetup pattern) + L760 (`PSolve` memcpy pattern)
- `SHUD/InstallSundials/include/cvode/cvode_ls.h` (`CVLsPrecSetupFn` + `CVLsPrecSolveFn` typedefs)
- `SHUD/InstallSundials/include/cvode/cvode.h` L132 (`CVodeSetLSetupFrequency`)
- SUNDIALS-CVODE 6.0.0 `src/sundials/cvode_spils.c` `cvLsSolve` (extra `N_VLinearSum` / `N_VScale` op triggered by `PREC_LEFT` 状态机, S5 drift root cause)

### Previous ADR

- `docs/adr/0002-solver-path.md`:
  - Path 1 (Serial NVec + StrictOMP RHS) IMPLEMENTED P1e epic close 2026-06-25
  - Path 3 (SPGMR + block-Jacobi precond) trigger 第 (iii) 条件 ADR-0003 GO 决策 = 本 ADR NO-GO
  - Path 4 (SPGMR → KLU) ADR-0003 forthcoming KLU spike status 保留 (not in p8pre-spike scope)

### Master plan

- `SHUD_openMP_master_plan.md` §P8-precond.0 (本 ADR 触发的 prep section update, PR-G #348 同 PR 一起更新)
- §6 P2a M12 (post-P1e profile localization 25% wall in `t_CVODE_internal`, ROI 量化的 prior epic input)
- §1.1.1 strict 量化目标 (precond ROI 目标 `nfeLS/nfe 降 ≥30% → 等价 wall 降 30%`, 本 spike NO-GO 不否决但推迟该目标)

### Mother template

- `docs/adr/0002-solver-path.md` (ADR 结构 mother template, 本 ADR 沿用 8-section 结构 + Context/Decision/Consequences/References 框架)
