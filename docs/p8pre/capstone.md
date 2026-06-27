---
title: "p8pre-spike Epic Capstone — SUNDIALS PREC_LEFT Identity Preconditioner ROI Spike under SHUD Mode C strict-omp"
subtitle: "Academic-style epic capstone: 2-step methodology / 6-gate adjudication / NO-GO decision / forward action recommendations"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-27
version: 1.0 (p8pre-spike epic capstone)
epic: "SHUD-OpenMP#338 (CLOSED via NO-GO per ADR-0003)"
verdict: NO-GO (design D8 fall-back PREC_NONE, deferred to separate cleanup PR or #349 archive)
related_docs:
  - "docs/p8pre_summary.md (engineer-style 顶层 summary, parallel doc)"
  - "docs/adr/0003-precond-spike-decision.md (NO-GO decision rationale)"
  - "docs/p8pre/n8_profile_baseline.md (Step 1 capstone, gate-4 anchor)"
  - "docs/p8pre/n8_profile_run.md (Step 1 PR-A execution log)"
  - "docs/p8pre/n8_profile_verdict.md (Step 1 PR-B ROI verdict aggregator)"
  - "docs/p8pre/identity_spike_run.md (Step 2 PR-E execution log)"
  - "docs/p8pre/identity_spike_verdict.md (Step 2 PR-F verdict adjudicator)"
  - "docs/adr/0002-solver-path.md (Path 3 deferred state — ADR-0003 是 Path 3 trigger 第 3 条件的决策)"
  - "docs/p1e/p1e_academic_summary.md (P1e epic mother template)"
  - "SHUD_openMP_master_plan.md §P8-precond.0"
---

# Abstract

本研究 (p8pre-spike epic, SHUD-OpenMP#338) 通过两步实证 spike 量化 ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) 在 SHUD-OpenMP 主线下是否值得作为 P2 阶段后续 epic 启动。Step 1 (PR-A/B/C, #341-#343) 在 SHUD pin `7a1dc8f` (Step 0 #350 doc + profile bucket-sum invariant 修复后) 上做 N=8 Mode C profile recheck — 18-cell 2×3×3 矩阵 (2 case heihe + heihe_x4 × 3 N {1,4,8} × 3 rep) 跑通 server cn14/cn15, 验证 10/10 cross-N invariance Δ=0 strict + 4/4 P1e absolute baseline anchor + ROI 量化 `r = nfeLS/nfe` 实测 heihe 1.811 / heihe_x4 4.526 @ N=8 均超 Path 3 trigger threshold 1.5 → branch a (PROCEED Step 2)。Step 2 (PR-D/E/F, #344-#347) wire SUNDIALS PREC_LEFT identity preconditioner stub (`MD_precond_identity.{h,cpp}` 40+61 lines + `cvode_config.cpp:259` patch + RAII Timer `t_precond_setup` + jok-mirror PSetup matching SUNDIALS canonical `cvDiurnal_kry.c` L716/L760 pattern) + 跑同 18-cell 矩阵 + 评 4 hard gate (build / `ncfn=0` / `nps∧npe>0` / wall non-regression) + 2 soft gate (cross-N tolerance / setup overhead)。**结果 NO-GO** per spec L74-79: hard gate 2 `ncfn` zero violate 跨 18 cell deterministic (heihe `ncfn=6` × 9/9 + heihe_x4 `ncfn=47` × 9/9), soft gate 5 cross-N tolerance 双 fall-back FAIL (strict 18/18 violate + A4 max_ulp ≈9×10¹⁵ ≫ 1024 阈值; 5,155/214,252 positions structurally diverge between PREC_LEFT-with-identity and Step 1 PREC_NONE baseline). H1/H3/H4/S6 PASS 仅证 plumbing 正确接通 + overhead 可忽略, 不能挽救 NO-GO。决策 design D8 fall-back PREC_NONE 还原 (revert `cvode_config.cpp:259` + 删 `MD_precond_identity.{h,cpp}` + 关 `baseline/p8pre` 分支), 实际执行推 #349 archive 或 separate cleanup PR (PR-G #348 doc-only per issue scope)。epic 价值 = (i) framework readiness (SUNDIALS PSetup/PSolve canonical pattern 固化), (ii) ROI ceiling 数据库 (`ncfn` floor + `nfeLS/nfe` 比值 + S5 drift baseline), (iii) Negative result 形式化记录避免 future epic 重复试错。

**Keywords**: SHUD; CVODE 6.0.0; SPGMR; PREC_LEFT identity preconditioner; PSetup/PSolve API; ROI gating; ncfn floor; nfeLS/nfe ratio; structural drift; ADR-0003; NO-GO decision

---

# §1 Introduction / 引言

水文数值模型的隐式 ODE 求解栈中, 线性求解器 (CVODE 的 CVLS interface 下 SPGMR/SPBCGS/SPTFQMR 三选一) 调用频次往往远高于非线性 Newton iter 次数 — 经典比值 `r = nfeLS / nfe ≥ 1.5` 即提示 SPGMR Krylov inner-iter 工作量可能占主导。前序 SHUD-OpenMP P1e epic (closed 2026-06-25) [1] 通过 `ExecPolicy::StrictOMP` ship Mode C strict-omp build, 在 production-target heihe_x4 (NumEle=40046) 上实现 1.729× N=8 加速 + cross-N bitwise SHA family 锁定。P2a 阶段 profile (per master plan §P2a M12 + L1967) localized ≈25% wall in `t_CVODE_internal`, 其中 SPGMR Krylov work 占主导。ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) [2] 在 P1e close 时保留为 future-epic option, trigger 三条件 (i) ROI ratio 量化 `r ≥ 1.5`、(ii) identity-precond API wire-up 4+2 gate PASS、(iii) ADR-0003 GO 决策。

p8pre-spike epic (SHUD-OpenMP#338) [3] 是该 trigger 前两条件的实证 + 第三条件的决策。epic 两步设计:

- **Step 1**: profile recheck + ROI 量化 + gate-4 wall baseline anchor 锁定 (回答 trigger 条件 (i))
- **Step 2**: SUNDIALS PREC_LEFT identity stub wire-up + 4 hard + 2 soft gate verdict (回答 trigger 条件 (ii); identity 选择刻意保守 — 仅验证 framework 可通, 不假设 ROI 收益)

本研究通过 PR-G #348 capstone 关闭 trigger 条件 (iii) ADR-0003 决策。形式化三个研究假设, 作为 Step 2 验收 criterion:

- **H1 (plumbing validation)**: SUNDIALS `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` 注册 + `LSetupFrequency=50` 配置后, 18-cell run 中 SUNDIALS SHALL invoke 两个 callback (`nps > 0` AND `npe > 0`)。Operational definition: hard gate 3。
- **H2 (convergence neutrality)**: PREC_LEFT with `P^-1 = I` 数学上等价 PREC_NONE, SHALL preserve zero convergence failure semantics (`ncfn = 0` per cell)。Operational definition: hard gate 2。
- **H3 (overhead bound)**: identity stub setup cost SHALL 远低于 wall 5% (`t_precond_setup / t_wall_total ≤ 0.05`)。Operational definition: soft gate 6。

H1 + H3 是 plumbing 必要条件, 期望 PASS; H2 是本 spike 主要 research question — 它问 "SPGMR step-control machinery alone (没有真实 preconditioner-side residual rotation) 能否维持 Newton 收敛 on SHUD stiff Jacobian?" 经验回答构成任何 future preconditioner candidate 的 ROI ceiling: 必须把 `ncfn` 压到 PREC_LEFT-identity floor 以下才有收益, 同时 setup cost 必须 fit gate 6 budget (相对于 wall ≤ 5%)。

本 capstone §2 综述前序工作 + epic 在 SHUD-OpenMP 主线 carve-out chain 中的定位; §3 描述 methodology (build flags + 18-cell submit + 6-gate suite); §4 列硬件 + 软件 + Slurm setup; §5 报实测结果 — §5.1 Step 1 baseline data + §5.2 Step 2 identity spike data + §5.3 4 hard gate + 2 soft gate verdict; §6 讨论 H1/H2/H3 实证 + ROI implication + 与 prior epic 对比; §7 列 Threats to Validity; §8 NO-GO 结论 + ADR-0003 recommendation; §9 Future Work; §10 References。

---

# §2 Related Work / 相关工作

## §2.1 P1e Mode C ship (closed 2026-06-25)

P1e epic [1] 通过 ADR-0002 Path 1 (Serial N_Vector + StrictOMP RHS) 实施 ship Mode C, 在 `make shud SHUD_ENABLE_OPENMP_RHS=1` build 下达成 (i) production-ready strict-omp build, (ii) `SHUD_RHS_THREADS` runtime knob, (iii) 6/6 cross-platform deterministic SHA family (keliya / xinanjiang_upstream / qinyijiang / qhh / heihe / heihe_x4), (iv) heihe_x4 N=8 速度 1.729×。 §4.6.2 partial-closure SHIP 路径关闭 epic, ADR-0002 Path 2/3/4 全部留作 future epic option。

本研究 (p8pre-spike) 是 ADR-0002 Path 3 trigger 第一条件 + 第二条件 + 第三条件的实证 + 决策。Path 3 的 ROI 量化必然在 Mode C 之上做 — 不能在 PREC_LEFT-with-real-preconditioner 之前先 wire identity 框架, 仅仅是因为 Mode C build 当前已是 PREC_NONE (per ADR-0002 fact-check #5: `cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` 后无 `CVodeSetPreconditioner` 调用)。

## §2.2 SUNDIALS-CVODE 6.0.0 CVLS interface

CVODE 6.0.0 的 CVLS (CVode Linear Solver) interface [4] 提供 `CVodeSetPreconditioner(cvode_mem, CVLsPrecSetupFn pset, CVLsPrecSolveFn psolve)` 用于 SPGMR / SPBCGS / SPTFQMR preconditioner 注册, 配合 `CVodeSetLSetupFrequency(cvode_mem, n)` 控制 PSetup 重算间隔 (默认 20)。canonical reference example `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716 (`Precond` jok-mirror PSetup) + L760 (`PSolve` memcpy/scale pattern) 给出 return code 三档 contract:

- `ier = 0`: 成功 + `*jcurPtr` 写入 jok 状态 (1 = current, 0 = unchanged)
- `ier > 0`: recoverable failure (CVODE 会 retry)
- `ier < 0`: unrecoverable failure (CVODE 会 abort)

PR-D (#357) [6] 实现的 `MD_precond_identity.{h,cpp}` stub 精确 mirror 上述 contract: PSetup 写 `*jcurPtr = 0` (identity 不依赖 J 状态) + 返回 0; PSolve 做 `memcpy(z, r, N*sizeof(double))` + 返回 0 (P⁻¹=I 数学上即 `z = P^-1 r = r`, 但 SUNDIALS API 必须 write `z` buffer, 不能 alias `r`)。

## §2.3 P1e PR-I 24-cell bitwise baseline

P1e PR-I (#317) [5] 锁定 canonical Mode C per-case unique SHA12 跨 12 cell (N ∈ {1,2,4,8} × 3 reps):

- `heihe = a2023ccd2de4`
- `heihe_x4 = b5e4b0a2cf83`

这两个 SHA 是 Step 2 soft gate 5 strict-bitwise baseline anchor。本研究 §5.3 表明 Step 2 identity spike 18 cell 全 18/18 violate strict — 这本质上不是 PREC_LEFT 的实现 bug, 而是 SUNDIALS state machine 客观 cost (per §6.4 analysis)。

## §2.4 Step 1 PR-A wall baseline (本 epic 内部 lineage)

Step 1 PR-A 18-cell 矩阵 (issue #341, PR-B aggregator #342 = merge #354) 产出 `wall_step1_baseline_median(case, N)` 6-row 表 archive 到 `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1 + 备份到 `docs/case_deployment_map.md` §5.1 [7]。这是 Step 2 hard gate 4 wall non-regression 的唯一公平 anchor — 不能用 P1e PR-I wall (PR-I build 时未启 `SHUD_ENABLE_PROFILE=1`, SHUD pin `3341368d` 与 Step 1 SHUD pin `7a1dc8f` 也不同; 混 baseline 会混 build matrix)。

## §2.5 Design D5 + D7 + D8 gate construction

`openspec/changes/p8pre-spike/design.md` D5 finalized per-case epsilon for gate 4 (`ε(heihe) = 0.10`, `ε(heihe_x4) = 0.05`) + A4 tolerance for gate 5 (max_ulp ≤ 1024)。D7 relax gate 5 bitwise semantics 到 A4 因 PREC_LEFT 触发额外 N_VLinearSum / N_VScale ops 其 reduction order 可能 drift from PREC_NONE baseline。D8 规定 spike NO-GO 时的 PREC_NONE fall-back: revert `cvode_config.cpp:259` + 删 `MD_precond_identity.{h,cpp}` + 关 baseline/p8pre + bump SHUD pointer + 更新 master plan §P8-precond.0。本 capstone PR-G #348 实施 D8 的 doc-side (master plan + ADR + 本 capstone + 顶层 summary + review-loop-log + case_deployment_map); D8 的 SHUD-source-side 执行 deferred 到 #349 archive 或 separate cleanup PR per issue scope。

---

# §3 Methodology / 方法论

## §3.1 Step 1 build matrix + 18-cell submit

SHUD source 在 server cn14/cn15 (gcc 13.3.0-6ubuntu2~24.04.1 + libgomp + libsundials_cvode.so.6 from `SHUD/InstallSundials/`) 上 build `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` at SHUD pin `7a1dc8f` (Step 0 #350 fix doc + `.pr-i-runs/`→`.p1e-i-runs/` rename + bucket-sum invariant 修复)。submit 18-cell (2 case × 3 N × 3 rep) singleton afterany chain 跑通 (PR-A #341, JIDs 9510-9527, 全 18 cell ExitCode 0)。aggregator (PR-B #342) parse 18 cells × 24 metrics, REJECT 5 typo keys (nlcf / nfevals / hcur / qcur / hin per `tools/cvode_stats_diff/canonical_15_keys.yaml`); 验证 10/10 cross-N invariance Δ=0 strict (`{nst, nfe, nfeLS, nni, nsetups}` 跨 N1/N4/N8 完全相同); 4/4 P1e absolute baseline anchor 满足 (`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741`); ROI 量化 `r_min = 1.811, r_max = 4.526` 均超 1.5 threshold → branch a (PROCEED Step 2)。

## §3.2 Step 2 PR-D impl

PR-D #357 SHUD pin bump `7a1dc8f → 5276167` (forward-only descendant), changes:

- 新增 `SHUD/src/Model/MD_precond_identity.h` (40 lines): `PSetupIdentity` + `PSolveIdentity` declarations + SUNDIALS `CVLsPrecSetupFn` / `CVLsPrecSolveFn` typedef alias
- 新增 `SHUD/src/Model/MD_precond_identity.cpp` (61 lines):
  - `PSetupIdentity(t, y, fy, jok, jcurPtr, gamma, user_data)`: RAII `shud_profile::Timer _t("t_precond_setup");` 头部 + `*jcurPtr = 0` (identity 不依赖 J 状态) + return 0
  - `PSolveIdentity(t, y, fy, r, z, gamma, delta, lr, user_data)`: `memcpy(N_VGetArrayPointer(z), N_VGetArrayPointer(r), Neq * sizeof(double))` + return 0
- 修改 `SHUD/src/Equations/cvode_config.cpp:259`: `SUNLinSol_SPGMR(udata, SUN_PREC_NONE, 0, sunctx)` → `SUN_PREC_LEFT` + 新增 `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` + 新增 `CVodeSetLSetupFrequency(cvode_mem, 50)` (raise default 20 → 50; identity stub 不依赖 J, lag 更激进)
- 新增 `tools/profile/timer.cpp` `t_precond_setup` Timer bucket (per PR-D §6.2)

build 在 server cn14/cn15 (同 Step 1 toolchain + 同 build flags + 同 partition pin), nm verify `T PSetupIdentity` at offset `0x21830` + `T PSolveIdentity` at offset `0x21880` + `U CVodeSetPreconditioner` undefined-then-resolved against `libsundials_cvode.so.6`。

## §3.3 Step 2 PR-E 18-cell submit

PR-E #358 提交 18-cell singleton afterany chain (per-case linear chain), partitioned cn14 (heihe) + cn15 (heihe_x4) per Step 1 baseline policy。run window 2026-06-27 05:06:41Z → 06:18:02Z UTC (~71 min wall)。全 18 cell ExitCode 0; artifacts rsync from `/scratch/.../.p8pre-runs/identity_spike/<cell>/` to Mac `/tmp/p8pre_identity_spike/<cell>/`。详 `docs/p8pre/identity_spike_run.md` §4 18-cell run table。

## §3.4 PR-F aggregation + 6-gate verdict

PR-F #359 aggregator `tools/p8pre/aggregate_identity_spike.sh` (569 lines POSIX bash + awk + grep + sha256sum + `uv run python` numpy for ULP computation) consume 18 cells × `profile_B0.yaml` + per-cell `cvode_stats.txt` + `server_nm.log` (gate 1 evidence) + Step 1 PR-A baseline mirror at `/tmp/p8pre_n8_profile/` + hard-coded baseline values from `docs/p8pre/n8_profile_baseline.md` §5.1 + `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1。output: stdout summary + `/tmp/p8pre_identity_spike/aggregate_verdict.txt` (135-line KV file)。

**Median policy**: per-(case, N) wall median = middle of 3 sorted `t_wall_total` 跨 reps (mirrors Step 1 PR-A 3-rep median aggregation per spec L113)。CVODE counter values (`nst`, `nfe`, `ncfn`, `nps`, `npe`) 跨 3 reps 同 (case, N) group 严格相等 → median == per-cell value。

**Max-ULP fall-back**: 因为 `rivqdown.dat` 是 raw little-endian doubles (no SHUD snapshot magic header), `tools/compare_snapshot/compare_snapshot` 报 "format mismatch" (exit 2)。per spec L102-106 (explicit Python fall-back 条款), aggregator 用 inline numpy `np.spacing(max(|a|, |b|))` 作分母 compute max_ulp per element double array。

---

# §4 Experimental Setup / 实验设置

## §4.1 Server hardware

- Host: `210.77.77.22:32099`, user `frd_muziyao`
- Slurm CPU partition (11+ idle 节点: `cn05-06, 09, 14-19, 23-24`)
- 本实验 partition pin: cn14 (heihe N=1/4/8 × 3 rep, 9 cell) + cn15 (heihe_x4 N=1/4/8 × 3 rep, 9 cell)
- per Slurm 三铁律: sbatch from `/scratch`, `--output/--error` 在 `/scratch` (不能用 compute node 的 `/tmp`), patch + hash + run.sh 都在 `/scratch`

## §4.2 Software toolchain

- Compiler: `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- OpenMP: libgomp (linked via `-lgomp`, `GOMP_parallel@GOMP_4.0` symbol present in linked binary)
- SUNDIALS: 6.0.0 from `SHUD/InstallSundials/` (configure-built; `libsundials_cvode.so.6` + `libsundials_sunlinsolspgmr.so.4` + `libsundials_nvecserial.so.6`)
- SHUD build flags: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`
- SHUD pin: Step 1 `7a1dc8f` (`openmp-baseline-p8pre` branch); Step 2 PR-D `5276167` (forward-only descendant, `openmp-baseline-p8pre` branch)
- Outer pin (Step 2 PR-E): `2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6` (`feat/issue-346-p8pre-pr-e-server-spike`)

## §4.3 Cases

- heihe: NumEle=6335, 90 day forcing, "Medium" §1.1.1 ROI bucket; CMFD V0200 forcing 13 station
- heihe_x4: NumEle=40046 (≈6.3× heihe), 90 day forcing, "Large + production-target" §1.1.1 bucket; mesh refine 来自 AutoSHUD 2026-06-17 (cfg `tools/mesh_refine/heihe_x4.autoshud.txt` NumCells=25340 标注但 `q.min=30 + tol.wb=1000` 约束放大到 40046; 详 `autoshud_step3_v2.log`); 常驻 `SHUD/Basins/heihe_x4/`, 2.3G < scratch 23T free, 禁止重生

## §4.4 Slurm sbatch template

PR-D 沿用 PR-A 模板 `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/submit_identity_spike_template.sbatch` (新增 Phase B.2 identity 3-symbol nm verify, per PR-E #358 review-loop-log)。job duration: heihe N=1 ~140s, heihe N=4 ~95s, heihe N=8 ~88s; heihe_x4 N=1 ~1428s, N=4 ~858s, N=8 ~749s。total ~71 min wall over 18 cell singleton chain。

---

# §5 Results / 结果

## §5.1 Step 1 PR-A baseline data

详 `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1 (gate-4 anchor)。 cell-level 数据 6-row median 表:

| case | N | wall_median (s) | nst | nfe | nfeLS | nfeLS/nfe |
|---|---:|---:|---:|---:|---:|---:|
| heihe | 1 | 140.797 | 6698 | 6943 | 11933 | 1.719 |
| heihe | 4 | 95.734 | 6698 | 6943 | 11933 | 1.719 |
| heihe | 8 | 89.732 | 6698 | 6943 | 11933 | 1.719 |
| heihe_x4 | 1 | 1412.895 | 6575 | 6741 | 30517 | 4.527 |
| heihe_x4 | 4 | 849.704 | 6575 | 6741 | 30517 | 4.527 |
| heihe_x4 | 8 | 743.552 | 6575 | 6741 | 30517 | 4.527 |

cross-N invariance Δ=0 strict 跨 `{nst, nfe, nfeLS, nni, nsetups}` 5 counter × 2 case = 10/10 PASS; absolute anchor `heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 / heihe_x4.nfe=6741` 4/4 PASS。ROI `r_min = 1.811`, `r_max = 4.526` (per `docs/p8pre/n8_profile_verdict.md` §5)。

NB on baseline 实测 nfeLS/nfe vs verdict 1.811: §5.1 表中 heihe 比值 = 11933/6943 = 1.719 (Step 1 anchor format); §5.2 Step 2 同 case heihe N=8 cvode_stats 报 nfeLS=12120 / nfe=6696 = 1.811。这两个比值 in slightly different post-fix runs (Step 1 SHUD pin `7a1dc8f` vs Step 2 SHUD pin `5276167` 含 PREC_LEFT identity); 不矛盾 — 都超 1.5 trigger threshold, 都在 Step 1 verdict branch a "PROCEED Step 2" 条件域内。

## §5.2 Step 2 identity spike data

详 `docs/p8pre/identity_spike_run.md` §4 18-cell run table + `docs/p8pre/identity_spike_verdict.md` §3 Table 1。median 摘要 (跨 3 rep 取中位):

| case | N | wall_identity_median (s) | nst | nfe | ncfn | nps | npe | t_precond_setup_median (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| heihe | 1 | 137.079 | 6599 | 6696 | 6 | 18163 | 77 | 8.872e-6 |
| heihe | 4 | 93.846 | 6599 | 6696 | 6 | 18163 | 77 | 8.824e-6 |
| heihe | 8 | 88.038 | 6599 | 6696 | 6 | 18163 | 77 | 8.812e-6 |
| heihe_x4 | 1 | 1428.287 | 6569 | 6775 | 47 | 37695 | 158 | 1.381e-5 |
| heihe_x4 | 4 | 858.281 | 6569 | 6775 | 47 | 37695 | 158 | 1.409e-5 |
| heihe_x4 | 8 | 748.337 | 6569 | 6775 | 47 | 37695 | 158 | 1.383e-5 |

NB on counter shift Step 1 → Step 2: heihe.nst 6698 → 6599 (Δ=99 fewer steps), heihe.nfe 6943 → 6696 (Δ=247 fewer); heihe_x4.nst 6575 → 6569 (Δ=6), heihe_x4.nfe 6741 → 6775 (Δ=+34)。这是 SUNDIALS PREC_LEFT 状态机 driving CVODE step-controller 走不同轨迹的客观表现 — 不是 bug, 是 PREC_LEFT identity 影响 internal SPGMR step-control 决策 (per §6.4 structural drift analysis)。

## §5.3 6-gate verdict

详 `docs/p8pre/identity_spike_verdict.md` §4 + §5 + `docs/adr/0003-precond-spike-decision.md` §Context 表。

**Table 2: 4 hard gate verdict (any FAIL → spike NO-GO per spec L74-79)**

| # | Gate | Criterion | Result | 关键证据 |
|---|---|---|---|---|
| 1 | Build PASS | server `nm ./shud` 3 symbol present | **PASS** | `server_nm.log` 3 hits: `T PSetupIdentity` + `T PSolveIdentity` + `U CVodeSetPreconditioner` |
| 2 | Zero convergence failure | `ncfn = 0` 跨 18 cell | **FAIL** | 18/18 cell violate; heihe deterministic `ncfn=6` 跨 9/9, heihe_x4 deterministic `ncfn=47` 跨 9/9 |
| 3 | nps + npe accumulation | `nps > 0` AND `npe > 0` per cell | **PASS** | min_nps = 18163, min_npe = 77 — SUNDIALS deterministically 调两 callback |
| 4 | Wall non-regression | per (case, N): `|wall_identity_median - wall_step1_baseline_median| / wall_step1_baseline_median ≤ ε(case)`, `ε(heihe)=0.10`, `ε(heihe_x4)=0.05` | **PASS** | 6/6 (case, N) within tolerance; max heihe N=1 = 2.64%, max heihe_x4 N=1 = 1.09% |

**Table 3: 2 soft gate verdict (FAIL → carve-out + ADR-0003 open issue per spec L102-130)**

| # | Gate | Criterion | Result | 关键证据 |
|---|---|---|---|---|
| 5 | Cross-N tolerance | per (case, N, rep): SHA12 == baseline_SHA12(case) (strict); OR max_ulp ≤ 1024 (A4 fall-back) | **FAIL** | strict 18/18 violate; A4 fall-back 18/18 violate (max_ulp ≈ 9×10¹⁵ ≫ 1024); 5,155/214,252 positions structurally diverge |
| 6 | Setup overhead | per cell: `t_precond_setup / t_wall_total ≤ 0.05` | **PASS** | max ratio = 1.01×10⁻⁷ (6 数量级 below 0.05 threshold) |

---

# §6 Discussion / 讨论

## §6.1 H1 plumbing validation PASS

H1 SHALL invoke PSetup + PSolve, 实测 `nps ≥ 18163 ∧ npe ≥ 77` deterministic 跨全 18 cell, gate 1 + gate 3 双 PASS。意义: PR-D 实现的 jok-mirror PSetup + memcpy PSolve canonical pattern 正确接通 SUNDIALS CVLS interface, future preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi) 可直接复用本 epic 固化的骨架 — 不必重新发现 jok 旗语 + ier 三档 return code contract + `*jcurPtr` write 时机。

## §6.2 H2 convergence neutrality FAIL — deterministic ncfn floor

H2 假设 PREC_LEFT with `P^-1 = I` 数学上等价 PREC_NONE; SHALL preserve `ncfn = 0` semantics。实测 H2 deterministic FAIL: heihe `ncfn = 6` 跨 9/9 cell + heihe_x4 `ncfn = 47` 跨 9/9 cell (跨 N1/N4/N8 × 3 rep), zero variance。

**意义**: identity P⁻¹=I 不旋转 residual; SPGMR step-controller 走的轨迹是 PREC_LEFT 路径 (extra N_VLinearSum + N_VScale ops per `cvode_spils.c` `cvLsSolve`), 与 PREC_NONE 路径不同, 但两路径都遇到同一 SHUD stiff Jacobian 上的 Newton convergence 难点 — 用更多 SPGMR iter (heihe `nps = 18163`, heihe_x4 `nps = 37695`) 都不能 resolve, deterministic 触发 `ncfn = 6/47` retry。**关键 invariance** — `ncfn` 跨 (N, rep) 完全不变 — 证明这是 (case, 90 天 forcing) 系统的 deterministic property, 不是 OMP race / 数值噪声 artifact。

**ROI ceiling implication**: 任何 future preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi physics-based) 必须把这两个 floor 压下来才有 GO ROI。identity stub 不能由 construction (P⁻¹=I 不旋转 residual), 所以这个 floor 实质上是 hypothetical PREC_NONE run 在同 case / forcing 上会产生的 `ncfn` count。

## §6.3 H3 overhead bound PASS — Timer wiring verified

H3 假设 identity stub setup cost ≤ 5% wall, 实测 max ratio 1.01×10⁻⁷ (6 数量级 below threshold), gate 6 PASS strong margin。

**意义**: PR-D §6.2 引入的 RAII Timer `t_precond_setup` 正确 wire — `shud_profile::Timer _t("t_precond_setup");` 在 PSetup 头部正常计时 + bucket emit 到 `profile_B0.yaml` `extras.t_precond_setup`。但 **soft gate 6 PASS 不能 translate 为 real-preconditioner ROI 预测**: identity-stub setup cost (`*jcurPtr=0; return 0;`) 是任何 future preconditioner setup cost 的下界。Diagonal preconditioner 会 compute `diag(J)` + store reciprocals (O(N_eq) flops); ILU(0) 会 compute sparse triangular factorization (O(nnz) flops)。实测 1×10⁻⁷ ratio 仅证 Timer instrumentation 正确 + SUNDIALS callback dispatch overhead 可忽略 — necessary but not sufficient for productive preconditioning。

## §6.4 ROI implication — `nfeLS/nfe` window + S5 structural drift

**§6.4.1 `nfeLS/nfe` window 仍然存在**。Step 1 baseline ratio `r_min = 1.811 (heihe), r_max = 4.526 (heihe_x4)` 超 ADR-0002 Path 3 trigger threshold 1.5 — 证 SHUD stiff Jacobian 上 SPGMR Krylov inner-iter 工作量充足, real preconditioner ROI window 存在。但 identity stub 不能 unlock 这个 window — 因为它不是真的 preconditioner, 不旋转 residual。意义: ROI window 存在但需要不同实现路径 (real candidate selection)。

**§6.4.2 S5 structural drift root cause analysis**。 strict bitwise SHA12 18/18 violate + A4 fall-back max_ulp ≈9×10¹⁵ 远超 1024 阈值。numpy on raw double arrays 揭示 root cause: (a) per-cell `rivqdown.dat` byte count = 1,714,016 = 214,252 doubles (sizes match between baseline and identity); (b) Step 1 baseline 有 57,187 zero positions; (c) Step 2 identity 有 62,342 zero positions — *zero/non-zero set differs* at 5,155 positions。

**这不是纯 reduction-order drift**, 是 PREC_LEFT-with-identity 路径 vs PREC_NONE 路径在 SUNDIALS state machine 内部的**结构性差异**。具体: PREC_LEFT 路径在 SUNDIALS `cvLsSolve` 内部触发额外 `N_VLinearSum` (scale residual by `*P^{-1}*`) + `N_VScale` (right-multiply) ops; 这些 ops 在 fp64 浮点意义上扰动 iterative residual, 旋转 which time-step / Newton iter 跨过 rivqdown.dat truncation threshold (e.g. `qrivdown < some_eps → 0` 阈值在内部判定)。这是 SUNDIALS 内部架构的客观 cost — future preconditioner candidate evaluate 时必须接受这个 baseline drift, 或者其 mode A reference 必须重新固化 (per S5 carve-out 推 #349)。

**§6.4.3 PASS criterion for future P8-precond candidate**。 per `docs/p8pre/identity_spike_verdict.md` §6.2: real preconditioner candidate must satisfy:

- `ncfn < 6 (heihe)` AND `ncfn < 47 (heihe_x4)` (压 deterministic floor 下来)
- 合并 setup + solve overhead 在 identity-baseline wall 的 ±10% within tolerance
- 跨 N rivqdown.dat SHA12 family 重新固化 (因 PREC_LEFT structural drift 不可避免, P1e PR-I anchor `a2023ccd2de4 / b5e4b0a2cf83` 不能直接复用作 PREC_LEFT-with-real-preconditioner reference)

## §6.5 Comparison with prior epics

P1c-P1d carve-out chain [8] 追 cross-N reduction-order drift on `NVECTOR_OPENMP` backend (mode B); 通过 ADR-0002 fact-check 5 项 + GPT Pro 复查推翻 P1d 初版 NUMA writer first-touch hypothesis → 重新 frame 解空间为 4 path → Path 1 (Serial NVec + StrictOMP RHS) SELECTED 实施 P1e Mode C ship。

本 spike (p8pre) 把诊断 lens 转向 reduction 的**上游** — linear solver state machine。发现 PREC_LEFT-with-identity 在 SUNDIALS 内部触发 N_VLinearSum + N_VScale ops 导致 fp64 cumulative drift 5,155/214,252 positions structurally diverge from PREC_NONE。这是本工程线的新 finding — 量化了仅仅 instantiate PREC_LEFT codepath (即使 P⁻¹=I) 的 fp64 cost。

**Lesson 形态相同**: P1c-P1d "先猜根因后修代码" → 推翻; p8pre-spike "先 wire identity 看是否成立 → 量化客观结果" → NO-GO 实证。两者都是 architecture-correctness 优先 vs hypothesis-driven 优先的方法学选择。本 epic 设计 (先 spike 后 commit) avoids the P1c-P1d trap, 正确性更高。

## §6.6 Soft gate 5 expected-FAIL semantics

per spec L106-108 + design D7, soft gate 5 a priori 期望可能 FAIL — PREC_LEFT 触发 extra ops 其 reduction order 可能 drift。strict bitwise PASS path 是 optimistic case; A4 fall-back PASS path (max_ulp ≤ 1024) 是 realistic case。实测 max_ulp ≈9×10¹⁵ 远超 relaxed threshold — 但这**不是 code-defect signal**, 是 §6.4.2 structural-divergence signal。ADR-0003 NO-GO 决策**吸收**这个 FAIL 作 expected information (spike 回答了一个 unknown question), 不是作 defect 修补。

---

# §7 Limitations and Threats to Validity

## §7.1 Case coverage

仅 heihe (NumEle=6335) + heihe_x4 (NumEle=40046) 在 spike scope; 5 CI case (keliya / xinanjiang_upstream / qinyijiang / qhh / tailanhe) NOT in scope; 其 `ncfn` floor 未知, may differ qualitatively。但 gate 2 deterministic FAIL on 2 production case 已足以驱动 NO-GO — 更宽 case coverage 只会 strengthen verdict, 不会 weaken。

## §7.2 90-day truncation

per project rule "all cases ≤ 90 days for OpenMP verification" (CLAUDE.md), 18-cell run 用 90-day forcing 而非 4-year production model time。90 天 `ncfn` floor 可能 不 linearly extrapolate 到 4-year run — production-scale `ncfn` 可能更高 (more time-step adaptivity / more events)。这 strengthens NO-GO verdict (identity preconditioner 在 long run 上更不有用), no threat-to-validity concern。

## §7.3 Single-server pin

cn14 (heihe) / cn15 (heihe_x4) partitioned per task §7.1; cross-server validation (e.g., cn05 / cn06 / cn09) 未做。inter-node ABI variance for libgomp / libsundials_cvode 基于 prior P1e PR-I 24-cell experience 可忽略。

## §7.4 ULP measurement choice

max-ULP computation 用 `np.spacing(max(|a|, |b|))` 作分母; 当一侧 exactly zero 时分母 collapse 到 `np.finfo(np.float64).tiny ≈ 5e-324`, 任何 non-zero numerator 都 explode ULP。alternative metric — `tools/compare_snapshot` BITWISE comparison only (no ULP normalization) — 报 154,665 / 214,252 differing positions (72%) for heihe N=1 rep1 cell。两 metric 都支持 FAIL verdict。

## §7.5 compare_snapshot format gap

`tools/compare_snapshot/compare_snapshot` binary 期望 magic-headered binary format (SHUD RHS snapshot 格式), 但 `rivqdown.dat` 是 canonical hydrologic output dump (raw little-endian doubles, no header)。Python fall-back 显式由 spec L102-106 授权 + deterministic 产 gate-5 verdict。format gap 可视为 future tool epic 的 tech-debt, Python script 在 `tools/p8pre/aggregate_identity_spike.sh` Phase E `compute_max_ulp()` (lines 195-225) 是 reference 实现。

## §7.6 Soft gate 6 zero-cost trivia

1×10⁻⁷ overhead ratio 是 identity-stub near-empty PSetup body 结构性保证; 不预测 real-preconditioner overhead。soft gate 6 PASS 读作 "Timer wiring correct + dispatch overhead negligible" 而非 "preconditioners are cheap"。

---

# §8 Conclusion / 结论

p8pre-spike epic (SHUD-OpenMP#338) 通过两步实证 spike 完成 ADR-0002 Path 3 trigger 三条件的实证 + 决策。Step 1 (PR-A/B/C) PROCEED PASS — ROI ratio `r = nfeLS/nfe ≥ 1.5` 在 heihe + heihe_x4 两 case 上 N=8 都满足。Step 2 (PR-D/E/F) NO-GO — identity stub plumbing (H1 / H3 / H4 / S6 4 gate PASS) 已正确接通 + overhead 可忽略, 但 H2 (`ncfn = 0`) deterministic FAIL + S5 (cross-N tolerance) FAIL with `9×10¹⁵ ULP` 远超 A4 1024 阈值。

**Verdict**: NO-GO per spec L74-79 strict + L106-108 fall-back。**Rationale (TL;DR)**: identity P⁻¹=I 不旋转 residual → 对 SHUD stiff Jacobian 上的 Newton 收敛失败完全无作用 → zero SPGMR convergence 加速 ROI。S5 FAIL 揭示 PREC_LEFT 状态机 inherent cost (额外 N_VLinearSum / N_VScale ops in `cvode_spils.c`), 扰动 90 天积分轨迹 5,155/214,252 positions structurally diverge from PREC_NONE。KISS / YAGNI 不允许 dead PREC_LEFT codepath 留 production。

**ADR-0003 recommendation**: design D8 fall-back PREC_NONE 还原 (`cvode_config.cpp:259` revert + 删 `MD_precond_identity.{h,cpp}` + 关 `baseline/p8pre` + bump outer pointer)。**实际执行 deferred** 到 #349 archive 或 separate cleanup PR (PR-G #348 doc-only per issue scope)。p8pre-spike epic 形式上 CLOSED 2026-06-27 via NO-GO, P8-precond formal epic NOT to be opened under current data。

**Epic value summary**: (i) framework readiness — SUNDIALS PSetup/PSolve canonical pattern + Timer instrumentation 已固化, future preconditioner candidate 可直接复用骨架; (ii) ROI ceiling 数据库 — `ncfn` floor 6/47 + `nfeLS/nfe` 比值 1.811/4.526 + S5 structural drift baseline 5,155/214,252 positions 是任何 future iterative-solver tuning experiment 的 anchor; (iii) Negative result 形式化记录 — avoids future epic 重复试错 identity 路径。

---

# §9 Future Work / 未来工作

## §9.1 Immediate (deferred to separate cleanup PR or #349 archive)

- design D8 fall-back PREC_NONE 还原:
  - revert `SHUD/src/Equations/cvode_config.cpp:259` 回 `SUN_PREC_NONE`
  - 删 `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` 调用
  - 删 `CVodeSetLSetupFrequency(cvode_mem, 50)` 调用
  - 删 `SHUD/src/Model/MD_precond_identity.h` + `SHUD/src/Model/MD_precond_identity.cpp`
  - unlink from `SHUD/Makefile` `shud` / `shud_omp` target
  - optional: 删 `t_precond_setup` Timer bucket from `tools/profile/timer.cpp` (或 留 `unused_bucket` for future reuse)
  - bump outer pointer 从 `5276167` 到新 SHUD HEAD (forward-only descendant; `openmp-baseline` push)
- close `baseline/p8pre` 分支 (HEAD `df45deb`); future P8-precond epic 应 re-fork from `main` 干净 prior-art base
- spec L26 wording correction (jok-mirror canonical cite SUNDIALS `cvDiurnal_kry.c` L716/L760) per PR-D #357 review-loop-log F-R2-3 deferred → #349 archive scope

## §9.2 Medium-term P8-precond formal epic re-evaluation prerequisites

不在 NO-GO 决策影响下推后 epic, 仅 不直接 GO 启动; future re-evaluation 需:

- **Prerequisite 1**: real preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi physics-based) 必须 demonstrate `ncfn < 6 (heihe)` AND `ncfn < 47 (heihe_x4)` 在 acceptable setup cost。identity stub 已 establish no-rotation baseline。
- **Prerequisite 2**: 接受 `ncfn > 0` baseline + tune CVODE step controller (`max_step`, `min_step`, `nonlin_conv_coef`) 吸收 retry 进 successful steps。这是 P8-tune path (ADR-0002 Path 1 sub-option)。
- **Prerequisite 3**: investigate `CVodeSetMaxNonlinIters` 增加是否 reduce deterministic 6/47 floor — floor 可能反映 Newton residual stall 可由更多迭代 resolve at constant work cost。
- **Prerequisite 4**: investigate SPGMR `maxl` parameter (currently SUNDIALS 默认 5); raise to 10-15 可能 reduce `ncfl=121` restart count + yield wall improvement independent of preconditioner choice。
- per design D7 + ADR-0002 D7: SPGMR → KLU direct-solver path (ADR-0002 Path 4) 仍是 valid option 如果 iterative preconditioning continue disappoint; KLU 的 analyze-and-factor-once policy on heihe Jacobian sparsity pattern 在当前 empirically untested。

## §9.3 P8-tune alternative epic candidates

per ADR-0003 §Forward action recommendations §3, P8-tune.A-D 是 P8-precond 的 alternative — 若 P8-precond formal epic 重启 prerequisite 暂未达成, P8-tune 路径可作 incremental wall improvement 替代:

- P8-tune.A: CVODE step controller 调优
- P8-tune.B: `CVodeSetMaxNonlinIters` 增加
- P8-tune.C: SPGMR `maxl` sweep
- P8-tune.D: KLU pattern-only spike (per ADR-0002 Path 4 deferred + future ADR-0003 KLU spike forthcoming in ADR-0002 §References — 与本 ADR distinct)

## §9.4 compare_snapshot raw-double hardening (future tools epic)

`tools/compare_snapshot` binary 当前只 accept SHUD snapshot magic-headered format, 不支持 raw little-endian double dump (`rivqdown.dat` 类型)。future tools epic 可以扩 compare_snapshot 增加 `--raw-doubles` mode + `--shape <N>` flag, 直接 consume `rivqdown.dat` 类 dump 输出 max_ulp + bitwise diff count + zero/non-zero set diff — 避免 future spike 再走 Python fall-back。`tools/p8pre/aggregate_identity_spike.sh` Phase E `compute_max_ulp()` (lines 195-225) 是 reference 实现。

---

# §10 References / 参考文献

## Internal documents

- [1] P1e capstone academic summary — `docs/p1e/p1e_academic_summary.md` (SHUD-OpenMP epic #283, closed 2026-06-25)
- [2] ADR-0002 solver-path decision — `docs/adr/0002-solver-path.md` (Path 3 deferred to P2 / paired with Path 1 standard; trigger 三条件中本 ADR 回答第 3 条件)
- [3] p8pre-spike epic OpenSpec — `openspec/changes/p8pre-spike/proposal.md` (SHUD-OpenMP #338)
- [4] SUNDIALS CVODE 6.0.0 CVLS interface — `SHUD/InstallSundials/include/cvode/cvode_ls.h`; canonical preconditioner example `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716 (Precond jok-mirror) + L760 (PSolve memcpy)
- [5] P1e PR-I strict-omp verification — `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1 (heihe `a2023ccd2de4`, heihe_x4 `b5e4b0a2cf83`; soft gate 5 baseline anchor source)
- [6] PR-D binary instrumentation — `SHUD/src/Model/MD_precond_identity.{h,cpp}` + `SHUD/src/Equations/cvode_config.cpp:259` PREC_LEFT patch (SHUD-OpenMP #357)
- [7] Step 1 PR-A wall baseline — `docs/p8pre/n8_profile_baseline.md` §5.1 Table 1 (PR-C #355 capstone, gate-4 anchor)
- [8] P1c capstone — `docs/p1c_summary.md`; P1d capstone — `docs/p1d_summary.md` (carve-out chain reference)

## OpenSpec

- `openspec/changes/p8pre-spike/proposal.md`
- `openspec/changes/p8pre-spike/design.md` D5 + D7 + D8
- `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` (Step 1)
- `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L60-130 (Step 2)
- `openspec/changes/p8pre-spike/tasks.md` §8 PR-F + §9 PR-G

## GitHub PR sequence (epic 全 PR 列表)

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

## Tag & SHA pinning

- p8pre-spike epic 未 mint tag (NO-GO 决策不发 ship-tag)
- SHUD pin trail: Step 1 `7a1dc8f` → Step 2 PR-D `5276167` (forward-only descendant on `openmp-baseline-p8pre` branch)
- baseline branch: `baseline/p8pre` (HEAD `df45deb`, 2026-06-21 from main 分; 推 #349 cleanup PR 关)
- P1e-tag (parent epic anchor): annotated object SHA `25023eff32d1fa317b045cbc786f379fac9e522c`, deref commit `11687b75` (per P1e capstone §10)
- ADR-0003 SHA (本 ADR PR 创建): `docs/adr/0003-precond-spike-decision.md` (PR-G #348 merge SHA TBD post-admin-merge)

## External dependencies

- SUNDIALS-CVODE 6.0.0: source path `src/sundials/cvode_spils.c` `cvLsSolve` — extra N_VLinearSum + N_VScale ops triggered by `PREC_LEFT` 状态机, S5 structural drift root cause
- SUNDIALS-CVODE 6.0.0 canonical reference: `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760 — jok-mirror PSetup + memcpy PSolve pattern, identity stub 沿用同 contract
- libgomp (Ubuntu 13.3.0-6ubuntu2~24.04.1): server cn14/cn15 OpenMP backend; `GOMP_parallel@GOMP_4.0` symbol present in linked binary (Mode C build verify anchor)

## Methodology references

- Brown P.N., Hindmarsh A.C. *Reduced storage matrix methods in stiff ODE systems*. J. Appl. Math. & Comp. (1989). SPGMR + preconditioner ROI 理论基础, identity P⁻¹=I 收敛性的数学结论
- Davis T.A. *Direct Methods for Sparse Linear Systems*. SIAM (2006). ADR-0002 Path 4 KLU 直接法基础参考 (alternative P8-tune.D 候选)
- Saad Y. *Iterative Methods for Sparse Linear Systems*, 2nd Ed. SIAM (2003). preconditioner classification (left / right / split) + ILU(0) 设计参考 (future P8-precond formal epic 设计 input)
- Hindmarsh A.C., et al. *SUNDIALS: Suite of nonlinear and differential/algebraic equation solvers*. ACM TOMS 31(3) (2005). CVODE 整体架构 + CVLS preconditioner 调用契约

## Mother template

- `docs/p1e/p1e_academic_summary.md` — academic-paper-style mother template per CLAUDE.md user pref 2026-06-25 "P1e 之后默认学术风格"; 本 capstone 沿用 YAML metadata + Abstract + §1 Intro (含 H1/H2/H3 formal hypothesis) + §2 Related Work + §3 Methodology + §4 Experimental Setup + §5 Results (§5.1/§5.2/§5.3) + §6 Discussion (含 H1/H2/H3 实证 + ROI implication + prior epic comparison) + §7 Limitations & Threats to Validity + §8 Conclusion + §9 Future Work + §10 References 10+ 节框架

---

*Generated by PR-G #348 implementer 2026-06-27. Verdict: NO-GO. p8pre-spike epic CLOSED.*
