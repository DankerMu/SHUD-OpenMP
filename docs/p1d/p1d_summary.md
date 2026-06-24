# P1d — capstone summary

P1d epic (`P1d.1 ~ P1d.7` per master plan §6 v1.5 M10) 总结。P1d 阶段 13 sub-PR + PR-C0 insertion 全部 merged 到 `baseline/P1d` 分支（HEAD `a19fb5e`，master plan M10 已 merge）。本文件作 P1d-tag (PR-L 后续) + PROMOTE (PR-M 后续) + P1e hand-off 的 source of truth。

## §1 Status

| Field | Value |
|---|---|
| Epic | #274 |
| Status | **PARTIAL CLOSURE via E′ containment path** |
| Date | 2026-06-24 |
| SHUD pin trail | P1c `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` → P1d `210ac191...` (post-PR-G Kahan revert, `openmp-baseline` pushed; net +7/-47 vs `6aada88` PR-E baseline) |
| Outer HEAD | `a19fb5e` (master plan v1.5 / M10 merged) |
| Master plan revision | v1.5 / M10（2026-06-24 PR-H 实测 + GPT Pro 双重复查 + codebase 事实核查驱动） |
| 4-mode 重写后 spec | `serial` / `strict-omp` / `det-omp` / `fast-omp` 四模式（详 §5 + §8） |

P1d 不是简单 E（"ship serial-only, walk away from OMP"），是 **E′ containment closure**：保留全部 P1d 工程结果 + 4-mode spec 把 strict 承诺限定到正确 mode + 把 P1e (F 路) 作为下个 epic 把 "真正应并行的 RHS 还没并行" 这件事补上。详 §5 + §9。

## §2 Epic scope

13 sub-PR + PR-C0 insertion = **14 PR**，于 2026-06-22 → 2026-06-24 三天内完成：

| PR | # | Scope | Status |
|---|---|---|---|
| PR-C0 | #292 | Delete dead legacy `f_update` / `f_loop` / `f_applyDY` definitions + 3 declarations | MERGED |
| PR-C | #293 | `MD_rhs_core.cpp::rhs_update` element pragma steady-state first-touch | MERGED |
| PR-D | #294 | `MD_rhs_core.cpp::rhs_flux` river pragma steady-state first-touch | MERGED |
| PR-E | #295 | `MD_rhs_core.cpp::rhs_update` lake pragma steady-state first-touch + 3 negative grep gate + qhh self-verify | MERGED |
| PR-F | #296 | Server intermediate 8-cell Kahan IN baseline run + `numactl --interleave=all` anti-pattern finding | MERGED |
| PR-G | #297 | SHUD Kahan revert (4 surgical reverts) + first-touch stacked + Mac 9-SHA matrix proves clean | MERGED |
| PR-H | #298 | Server final 8-cell 3 SHALL gate verdict = FAIL + E′ rewrite post-verdict revision | MERGED |
| PR-I | #299 | Mac independent worktree `P1-update-omp-tag` reference 6-cell anchor | MERGED |
| PR-K | (this) | Capstone docs — 4 new + 2 update + 1 evidence | (this PR) |
| PR-L | (next) | `P1d-tag` annotated + `baseline/P1d` lock (D11 6-tag chain) | PENDING |
| PR-M | (last) | PROMOTE 2 specs (`p1d-numa-governance` + `p1d-capstone`) + archive + Epic close + propose `p1e-strict-omp-rhs` | PENDING |

> Note：原 epic 草案有 PR-A (proposal) / PR-B (NUMA env runbook) / PR-J (Mac comparison) 三个站。PR-A/B 已在 epic intake / PR-B #276 完成；PR-J 在 PR-H FAIL + E′ path 决策后取消（Mac PR-J comparison 没必要在 server FAIL 后还做）。最终序列见上表。

## §3 3 SHALL gate verdict（per spec.md L126-150）

| Gate | Threshold | PR-H 实测 | Verdict |
|---|---|---|---|
| L123 Kahan revert canonical | heihe N=1 SHA == `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | byte-identical 全 64-hex | **PASS** |
| L130 A3a bitwise cross-N | heihe + heihe_x4 每 case N∈{1,2,4,8} 4 cell SHA 全等 | 双 case 各 3 distinct SHA（N=1≡N=2 ≠ N=4 ≠ N=8） | **FAIL** |
| L139 nst Δ + ladder | heihe Δ=0 strict + heihe_x4 \|Δ\| ≤ 2 | heihe Δ=80@N=4 / 152@N=8；heihe_x4 Δ=11@N=4 / 4@N=8 | **FAIL** |
| L145 N=1 reverse-compat | 6-case N=1 SHA == `P1-update-omp-tag` canonical | server heihe PARTIAL（spec 仅预写 heihe N=1 canonical） | **PARTIAL PASS** |

PR-H FAIL trigger → 按 spec L150 epic 应标 BLOCKED 不进 PR-I/J/K/L/M。**用户决策（2026-06-24，两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查支持）走 E′ containment path**：保留 PR-H FAIL verdict + burst 继续 + 4-mode spec rewrite + P1e (F 路) 作为下个 epic。详 §5。

## §4 Empirical findings

### §4.1 rivqdown.dat 实测散度（PR-H, 90-day heihe / heihe_x4）

| Case / N | mean_rel | RMSE | max_rel | n_diff / n_total |
|---|---|---|---|---|
| heihe N=2 vs N=1 | 0 | 0 | 0 | 0 / 214252 |
| heihe N=4 vs N=1 | **3.8%** | 1.79e+03 | 1010× | 210774 / 214252 |
| heihe N=8 vs N=1 | **10.0%** | 2.76e+03 | 12534× | 210779 / 214252 |
| heihe_x4 N=2 vs N=1 | 0 | 0 | 0 | 0 / 387607 |
| heihe_x4 N=4 vs N=1 | **17.6%** | 8.63e+03 | 2214× | 383130 / 387607 |
| heihe_x4 N=8 vs N=1 | **25.1%** | 7.59e+03 | 2908× | 382116 / 387607 |

### §4.2 P1c era 对照（Kahan IN，no NUMA env，no first-touch）

| Case / N | mean_rel | max_rel |
|---|---|---|
| heihe N=8 vs N=1 | 10.0% | 1446× |
| heihe_x4 N=8 vs N=1 | 20.3% | 66120× |

P1c PR-I 早就以 10-25% N≥4 流量散度上线，**当时未补做这个测算**。PR-H 实测把 P1c 遗留 finding 显式化。Kahan 注入只压住 nst step count，没修水文输出散度——orthogonal axes。`max_rel` 1000-66000× 部分被极小 denominator 放大，待后续审计是否引入物理 floor；`mean_rel` 10-25% 是工程层面的硬实锤。

### §4.3 wall + speedup（server）

| Case | N=1 wall | N=2 | N=4 | N=8 | Speedup S8 | 并行效率 |
|---|---|---|---|---|---|---|
| heihe | 513s | 513s | 503s | 456s | **1.13×** | 14.1% |
| heihe_x4 | 1187s | 1169s | 1096s | 937s | **1.27×** | 15.9% |

### §4.4 Amdahl 反推（recalculated, supersedes PR-H 初版 "f~72%" 估算）

由 `f = (1/S - 1/N)/(1 - 1/N)`，N=8 + S=1.13 → `f ≈ 0.869`；S=1.27 → `f ≈ 0.757`。

| Case | Amdahl serial fraction `f` |
|---|---|
| heihe | **86.9%** |
| heihe_x4 | **75.7%** |

注释：`f` 不能全部归因 CVODE 内部——含 (a) serial RHS（fact-check #1+#2：当前 `shud_omp` 走 `ExecPolicy::Serial` 路径）(b) N_Vector fork/join 开销（fact-check #3：`shud_omp` 硬编码 `NVECTOR_OPENMP`）(c) memory bandwidth 饱和。

**关键洞察**：早期 B1a profile 显示 heihe_x4 RHS 占 wall **66.55%**，理想 8 核 Amdahl 上界 = `1 / (1 - 0.6655 + 0.6655/8) = 2.39×`。当前 1.13-1.27× 的瓶颈**不是 "Amdahl 已极限"**，是 **真正应并行的 RHS 还没并行**（StrictOMP 路径还是 abort 桩，per fact-check #2）。

## §5 E′ containment closure — 8 项动作

| # | 动作 | 状态 |
|---|---|---|
| 1 | production 默认 `cfg.para NUM_OPENMP=1`（serial path 实测仅比 N=8 慢 11-12%，但可 reproducible；ROI 优于跑 N=8 拿 10-25% mean rel error） | spec rewrite per PR-M |
| 2 | `shud_omp` 标 `fast-omp experimental, non-production`（保留 build + CI；不进 production cfg.para 默认值） | spec rewrite per PR-M |
| 3 | 3 SHALL gate 重写为 **4-mode spec**（不弱化 strict 承诺，限定到正确 mode）：`serial` / `strict-omp` (待 P1e) / `det-omp` (后续) / `fast-omp` (=current `shud_omp`) | spec rewrite per PR-M |
| 4 | PR-C/D/E **steady-state first-touch loops 标 deprecated**（owner-compute 未实现，无 consumer 享受 NUMA locality）；allocation-time first-touch 保留；P1e 重新设计为 owner-compute 配套 | M10 master plan + 本 doc + p1d_numa_root_cause.md |
| 5 | **PR-G Kahan revert 保留**（事实核查证明 revert 干净 + N=1 byte-identical 到 pre-Kahan canonical；为 P1e 做 baseline 准备） | 本 doc + p1d_kahan_revert.md |
| 6 | **PR-K capstone docs 诚实记录** NVECTOR_OPENMP 是当前 cross-N 散度根因 + Serial RHS 是真正 bottleneck + first-touch deprecation | 本 PR |
| 7 | **PR-L `P1d-tag` annotated message** 写 containment closure narrative + 指向 P1e；`baseline/P1d` lock_branch=true | PR-L (next) |
| 8 | **PR-M PROMOTE 2 specs**（含 4-mode 重写） + Epic #274 close + propose `p1e-strict-omp-rhs` openspec change | PR-M (last) |

## §6 5/5 codebase 事实核查（PR-H post-verdict 修订）

| # | 断言 | 核查 | 结论 |
|---|---|---|---|
| 1 | `f()` 始终调 Serial RHS | `SHUD/src/Model/f.cpp:54` `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` | ✅ |
| 2 | `StrictOMP/ProductionOMP` 是 abort 桩 | `SHUD/src/Model/MD_rhs_core.cpp:802-811` 三 case 全 `std::abort()` | ✅ |
| 3 | `shud_omp` build 硬编码 NVECTOR_OPENMP | `SHUD/Makefile` shud_omp target: `-DSHUD_USE_OPENMP_NVECTOR=1 -lsundials_nvecopenmp` | ✅ |
| 4 | `SHUD_USE_OPENMP_NVECTOR` 与 `SHUD_ENABLE_OPENMP_RHS` 正交，后者默认 0 | `SHUD/Makefile:140` `SHUD_ENABLE_OPENMP_RHS ?= 0` | ✅ |
| 5 | SPGMR 没注册 preconditioner | `SHUD/src/Equations/cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` 后无 `CVodeSetPreconditioner` 调用 | ✅ |

**含义**：`shud_omp` 当前实际跑的是 **Serial 水文 RHS + OpenMP N_Vector backend**，**不是** 真正的 hydrology RHS OpenMP 并行。PR-C/D/E 添加的 steady-state first-touch loops 是为**完全没发生的** parallel RHS owner-compute 做的页面预放置——consumer 是单线程，根本无视 NUMA locality，是**无效优化 + 带宽浪费**。

## §7 初版 4 个错误诊断的纠正

PR-H 初版 verdict 把 FAIL 解读为 "nst Δ 超 spec ladder + 根因在 SPGMR multi-threaded preconditioner + KLU 唯一根治"。**两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查全部推翻**。本节 4 项更正：

| 初版（错） | 修订（对） |
|---|---|
| "drift origin 在 SPGMR multi-threaded preconditioner" | SPGMR 没有 preconditioner（fact-check #5）；drift 是 `N_VDotProd_OpenMP` 的 `reduction(+:sum) schedule(static)` 跨 N reduction tree 顺序不固定（SUNDIALS 6.0.0 `nvector_openmp.c`）。WRMS norm 同样过 OpenMP reduction |
| "Amdahl serial fraction ~72%" | 由 wall 反推 `f = (1/S - 1/N)/(1 - 1/N)`：heihe ~**87%**，heihe_x4 ~**76%**（§4.4）；且不能全部归因 CVODE——含 serial RHS + N_Vector fork/join 开销 + memory bandwidth 饱和 |
| "reltol=1e-4 但实测 10% → solver 承诺被打穿四个数量级" | CVODE reltol 控制的是**每步状态 WRMS local error**，**不是** 90 天派生流量轨迹的全局相对误差上界。10% mean_rel 不可接受是工程结论，不是 CVODE 实现错误的证据 |
| "KLU 是唯一根治路径" | KLU 不能单独 fix。CVODE WRMS norm 还是过 N_Vector reduction，换 KLU 不换 N_Vector 仍然漂。Determinism 与 solver 选择**正交**。KLU 推到 ADR-0002 作 4 路对比之一 |

详 `docs/p1d/p1d_numa_root_cause.md` §7。

## §8 K=200 vs K=0 在流量层面的真实含义

| Mode | nst Δ | 流量散度 | 含义 |
|---|---|---|---|
| K=0 (true strict) | 0 | ≤ULP | 真 reproducible |
| K=200 (PR-H 实测) | 80-152 (heihe) | mean 10-25% rel error | **另一条 CVODE 轨迹**，非 ULP 近似 |

对照量级：

| 参照 | 量级 | 与 PR-H mean_rel 10-25% 关系 |
|---|---|---|
| CMFD forcing 不确定性 | 10-20% | 同量级 |
| 流量计 gauge 精度 | 5-10% | **超过测量精度** |
| Manning's n 标定带宽 | ~50% | 量级以下 |

工程结论：K=200 的 10-25% 散度不能 claim "in tolerance with CVODE rtol"；它是 **另一条轨迹**，不是噪声。这是 4-mode 重写把 `fast-omp` 标 non-production 的依据。

## §9 Forward handoff → P1e（F 路）

### §9.1 P1e 启动前置

- P1d 全部 closure (本 doc + p1d_pr_h_final_run.md 的 E′ post-verdict 修订)
- `baseline/P1d` 已 lock + `P1d-tag` 已 push（PR-L）
- ADR `docs/adr/0002-solver-path.md` 已建立（4 路对比：Serial N_Vector + StrictOMP RHS / Deterministic NVECTOR_REPRO_OMP / SPGMR + block-Jacobi precond / KLU sparse direct）— Phase 2(e) 并行 agent owns
- openspec `p1e-strict-omp-rhs` change 已 propose — Phase 2(e) 并行 agent owns

### §9.2 P1e 技术主线（F 路）

**Serial N_Vector + StrictOMP RHS**：CVODE/SPGMR 继续用 Serial N_Vector（保证 reduction 顺序确定 → cross-N bitwise 自然成立），SHUD 水文 RHS 真正并行（替换 abort 桩 → 真正去吃 RHS 66.55% wall → 理想 2.39× Amdahl 上界可达）。

### §9.3 启动前必跑 2×2 build 因果实验

| Build | N_Vector | RHS | 目的 |
|---|---|---|---|
| A | `N_VNew_Serial` | Serial | canonical reference |
| B | `N_VNew_OpenMP` | Serial | = 当前 `shud_omp`，复现 10-25% 散度作 control |
| C | **`N_VNew_Serial`** | **StrictOMP** | **P1e production 候选** |
| D | `N_VNew_OpenMP` | StrictOMP | research only |

每 build × N∈{1,2,4,8} × 3 repeats，hash `CV_Y` + `rivqdown.dat` + capture `nst/nfe/nli/nni/netf/ncfn` 15-key set。判据：

- **A 同 build 同 N 重复 3 次 bitwise** → solver 本身是 deterministic 的，前提
- **B 跨 N 不同 而 A 跨 N 相同** → 确认 NVECTOR_OPENMP reduction 是主因（不是 RHS race）
- **C 跨 N bitwise + nst Δ=0 + 加速** → F 路成立
- **C 跨 N 也分叉** → 查 RHS race / 共享状态 / phase dependency；可能更细 owner-compute 分解
- **C 加速 < 1.5×** → 进 ADR-0002 评估 (block-Jacobi precond / NVECTOR_REPRO_OMP / KLU)

### §9.4 P1e 实施要点

1. **`ExecPolicy::StrictOMP` 路径替换 `std::abort()` 桩**（`MD_rhs_core.cpp:802` 当前 case）
2. **单 parallel region + phase-based for + `default(none)`** + 隐式 barrier
3. **复用现有 `rhs_deterministic_gather()` 基础设施**：并行 owner 外层 + canonical fold 内层（fixed B0 顺序 left-fold，spec 已设计好）
4. **配置项拆**：`NUM_RHS_THREADS` (RHS 并行度) + `NUM_NVECTOR_THREADS` (默认 1 = Serial NVector)
5. **删 PR-C/D/E steady-state first-touch loops**（M10 deprecated）；保留 allocation-time first-touch（`Model_Data.cpp::malloc_EleRiv` L251-L346 模式）
6. **`rivqdown.dat` 输出缓存 audit**（per Pro2 警示）：确认输出代码是从 `tout` 状态重算 flux，**不是** 直接写 solver 内部最后一次 RHS 留下的 `FluxRiv` 缓存（CVODE `CV_NORMAL` 模式下 internal step 可能超过 output time）

### §9.5 P2a 启动前置改链

原 "P1c-tag 已 lock + P1c.3 A3a 全通过" 不再充分。新前置 = **P1e (F 路) 完成 strict-omp mode 实现 + 3 SHALL gate 在 strict-omp mode 内通过 + 加速比 ≥ 1.5× + `P1e-tag` 已 push + `baseline/P1e` lock + ADR-0002 close out**。详 master plan §6 P1d.7 + §6 P2a preamble。

## §10 P1c carve-out closure

spec `p1c-deterministic-reduction` L100-L103 carve-out Scenario + L154-157 Mac SHALL Scenario 在 P1d 完成时**间接闭合**，但不是按原 hypothesis（"NUMA writer governance + first-touch 治理"）闭，而是 **E′ containment closure 把问题归类到正确层**：

- 原 carve-out "upstream parallel writer first-touch / NUMA-affinity governance" — fact-check #2 证明 consumer 是单线程（StrictOMP abort 桩），所以 upstream first-touch 是无效优化；问题不是 "NUMA 治理" 治不好的，问题是 **RHS 真正没并行**
- Mac SHALL Scenario L154-157 — P1d PR-I 在独立 worktree 切 `P1-update-omp-tag` 跑 Mac 6-cell（keliya + qhh × {serial, omp@N=1, omp@N=8}）并 archive SHA，作为 future P1e Mac 比对的 anchor reference；详 `docs/p1d/p1d_pr_i_p1_update_omp_reference.md`

per D11 PROMOTE 历史 immutability：保留 P1c spec 字面 "carve-out 推 P9 行" 不动；P1d closure 通过 `docs/p1d/p1d_summary.md`（本 doc）+ `docs/p1d/p1d_numa_root_cause.md` 显式记录 "P9 字面 = P1d 语义同义" 映射。

## §11 三 negative grep gate 保留（P1d-wide）

per `openspec/specs/p1c-deterministic-reduction/spec.md` L76/L81/L86 已建立的 3 gate，P1d 全 epic 期间所有 SHUD 改造 commit 验证：

| Gate | Command | 期望 | PR-K post-K verify |
|---|---|---|---|
| 新宏 0 | `grep -rE 'SHUD_USE_DETERMINISTIC_REDUCTION\|SHUD_DET_REDUCT\|SHUD_PAIRWISE' SHUD/` | 0 hits | 0 hits ✓ |
| schedule | `grep -nE 'schedule\((dynamic\|guided)\)' SHUD/src/Model/MD_rhs_core.cpp` | 0 hits | 0 hits ✓ |
| atomic | `grep -rn '#pragma omp atomic' SHUD/src/` | 0 hits | 0 hits ✓ |

PR-K capstone 显式 re-verify per spec L221-L224 Scenario。

## §12 References

| 文档 | 内容 |
|---|---|
| 本文件 (`docs/p1d/p1d_summary.md`) | P1d capstone source of truth |
| `docs/p1d/p1d_perf_baseline.md` | 8-cell wall + Amdahl + Mac 6-cell + CPU-hour cost |
| `docs/p1d/p1d_numa_root_cause.md` | NVECTOR_OPENMP reduction 根因分析 + first-touch 失效原因 |
| `docs/p1d/p1d_first_touch_design.md` | PR-C/D/E first-touch 设计（M10 后 steady-state 部分标 DEPRECATED） |
| `docs/p1d/p1d_numa_env_runbook.md` | NUMA env standardization (PR-B) |
| `docs/p1d/p1d_kahan_revert.md` | PR-G SHUD Kahan revert + 9-SHA matrix |
| `docs/p1d/p1d_pr_f_intermediate_run.md` | PR-F intermediate Kahan IN 8-cell + `--interleave=all` anti-pattern finding |
| `docs/p1d/p1d_pr_h_final_run.md` | PR-H final 8-cell verdict + E′ post-verdict 修订 |
| `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` | PR-I Mac `P1-update-omp-tag` worktree reference 6-cell anchor |
| `docs/p1d/p1d_pr_k_capstone_run.md` | 本 PR-K self-evidence |
| `docs/p1d/p1d_tag_and_lock.md` | PR-L `P1d-tag` + branch lock 程序 (PR-L 作) |
| `docs/p1d/p1d_report.md` | epic executive report |
| `SHUD_openMP_master_plan.md` v1.5 / M10 | §6 P1d + §6 P1e + §7.2 RISK-NEW1/2 + §7.3 stage rows + §8.1 4-mode block |
| `docs/adr/0002-solver-path.md` | 4 路 solver 对比 ADR (Phase 2(e) 并行 agent owns) |
| `openspec/changes/p1d-numa-governance/` | P1d openspec change (PR-M PROMOTE) |
| `openspec/changes/p1e-strict-omp-rhs/` | P1e openspec change (Phase 2(e) 并行 agent owns) |
| `docs/status_matrix.md` | P1d 行 + P1e 行 PENDING（本 PR 更新） |
| `docs/build_manifest.md` | SHUD pin trail P1d 段 + FP gate 3-flag 形 (本 PR 更新) |

## §13 验证 P1d-tag（PR-M 实测填实）

**Status**: FILLED — PR-M post-PR-L-merge 实测 (2026-06-24)

PR-L 创建本 placeholder 章；PR-L 合并后 orchestrator 执行 `git tag -a P1d-tag <HEAD> -m '<msg>'` 创建 tag；PR-M 编辑本章填入实测 tag-object SHA + deref commit SHA。

### §13.1 P1d-tag 创建命令（orchestrator post-merge 执行）

详 `docs/p1d/p1d_tag_and_lock.md` §4 P1d-tag 创建命令。

### §13.2 D11 5 historical tag SHA 不变 + P1d-tag 新增 = 6 chain（PR-M 实测）

| Tag | tag-object SHA (PR-L 时刻) | tag-object SHA (PR-M 验证) | deref commit SHA (PR-M 验证) | 状态 |
|---|---|---|---|---|
| B1-tag | `0c0621c986e54e371c5a176850d1eb981150010e` | `0c0621c986e54e371c5a176850d1eb981150010e` | `ed054b417101bca35d2e10cd262d3333187b983d` | byte-identical immutable |
| B1a-tag | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | `f7f992cabab5d5aec3bf08ab2db7c0669ef7fe75` | byte-identical immutable |
| B1b-tag | `96e224daad8cb9c93f855851724f8d45468391c2` | `96e224daad8cb9c93f855851724f8d45468391c2` | `18a0c9085f494d1cf228c7be4adf27d9132d05dd` | byte-identical immutable |
| P1-update-omp-tag | `ff21c75c8e968d5e47ca53b015425360be9ac879` | `ff21c75c8e968d5e47ca53b015425360be9ac879` | `003f58dc079116ef2161d2f96006228ef0e013d0` | byte-identical immutable |
| P1c-tag | `1da5eb9734680fc61e68f6091964c38fc5f67c6f` | `1da5eb9734680fc61e68f6091964c38fc5f67c6f` | `4b8c60af261e0d1517f52702e4827a4e2d67dd41` | byte-identical immutable |
| P1d-tag | n/a (PR-L 时刻未创建) | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` | `f88f2dc2cad1adbe3797b89fbe247aa12bf8c0a9` | NEW (PR-L post-merge, 6th chain) |

D11 immutability 验证通过：5 historical SHA 全部 byte-identical 至 P1c epic close 时刻记录值（per `docs/p1c_tag_and_lock.md` + `openspec/glossary.md` L168/L200 entries）；P1d-tag 仅追加，不动 5 historical。

> 注：`docs/p1d/p1d_tag_and_lock.md` §5 表（PR-L 录入版）SHA 值与本节略有错位；PR-M 已在 `p1d_tag_and_lock.md` §5.1 添加 SHA correction note 说明。本节 §13.2 取 PR-M `git rev-parse <tag>` 实测输出作权威 source-of-truth。

### §13.3 P1d-tag 内容验证（PR-M 实测）

| Item | 实测值 |
|---|---|
| `git rev-parse P1d-tag` (tag-object SHA) | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` |
| `git rev-parse P1d-tag^{}` (deref commit SHA) | `f88f2dc2cad1adbe3797b89fbe247aa12bf8c0a9` |
| `git show P1d-tag --format=%s` (subject line) | `P1d epic capstone — E' containment closure` |
| `git cat-file -p P1d-tag \| wc -l` (message lines) | 66 |
| `git ls-tree P1d-tag SHUD` (SHUD submodule pin at tag) | `160000 commit 210ac193a4f09f700a9c0b20010d19f788948c32	SHUD` |

### §13.4 baseline/P1d 分支 lock 验证（PR-M 后置）

**Status**: PENDING PR-M merge + branch lock command execution.

| Item | 实测值 |
|---|---|
| `gh api repos/.../branches/baseline/P1d --jq '.protection.lock_branch.enabled'` | TBD (post-PR-M-merge, expected true) |
| `gh api .../protection --jq '.enforce_admins.enabled'` | TBD (post-PR-M-merge, expected true) |
| `gh api .../protection --jq '.allow_force_pushes.enabled'` | TBD (post-PR-M-merge, expected false) |
| `gh api .../protection --jq '.allow_deletions.enabled'` | TBD (post-PR-M-merge, expected false) |

> 注：lock 命令在 PR-M merge 后由 orchestrator 立即执行（per `docs/p1d/p1d_tag_and_lock.md` §6）。本表 post-lock 由 orchestrator 填回实测 `gh api` 输出。

### §13.5 main fast-forward 验证（PR-M 后置）

**Status**: PENDING PR-M merge + main fast-forward command execution.

| Item | 实测值 |
|---|---|
| `git rev-parse origin/main` | TBD (post-PR-M-merge) |
| `git rev-parse origin/baseline/P1d` | TBD (post-PR-M-merge) |
| Match | TBD (post-PR-M-merge, expected true) |

> 注：main fast-forward 在 PR-M lock 完成后执行（per `docs/p1d/p1d_tag_and_lock.md` §7）。本表 post-fast-forward 由 orchestrator 填回实测 `git rev-parse` 输出。

### §13.6 P1d 6-tag chain post-PR-L summary table

PR-L 合并 + P1d-tag 创建 + push 后，完整 D11 6-tag chain（含 tag-object SHA + deref commit SHA 双视图）：

| Tag | tag-object SHA (annotated) | deref commit SHA |
|---|---|---|
| B1-tag | `0c0621c986e54e371c5a176850d1eb981150010e` | `ed054b417101bca35d2e10cd262d3333187b983d` |
| B1a-tag | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | `f7f992cabab5d5aec3bf08ab2db7c0669ef7fe75` |
| B1b-tag | `96e224daad8cb9c93f855851724f8d45468391c2` | `18a0c9085f494d1cf228c7be4adf27d9132d05dd` |
| P1-update-omp-tag | `ff21c75c8e968d5e47ca53b015425360be9ac879` | `003f58dc079116ef2161d2f96006228ef0e013d0` |
| P1c-tag | `1da5eb9734680fc61e68f6091964c38fc5f67c6f` | `4b8c60af261e0d1517f52702e4827a4e2d67dd41` |
| P1d-tag | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` | `f88f2dc2cad1adbe3797b89fbe247aa12bf8c0a9` |

D11 NG7：6 tag 全 annotated；`git tag --verify <tag>` 全 6 PASS；force-update 禁止（per design D11 + master plan §6 P1d.5）。

### References

- `docs/p1d/p1d_tag_and_lock.md` (P1d-tag procedure + §5.1 SHA correction note)
- `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` § "baseline/P1d 分支 + P1d-tag"
- master plan v1.5 / M10 §6 P1d.5 (D11 6-tag chain) + §6 P1d.6 (Go/No-Go → P1e)
- `docs/p1d/p1d_go_nogo_verdict.md` (PR-M 综合 verdict)

## §14 时间线（per P1c §4 模板，补齐对齐）

| 日期 | 事件 | PR / Issue |
|---|---|---|
| 2026-06-23 | PR-A merged: epic intake + baseline/P1d 分支 + `.gitignore .p1d-runs/` scratch | #288 / closes #275 |
| 2026-06-23 | PR-B merged: NUMA env 标准化 runbook + server sbatch `OMP_PROC_BIND=close + OMP_PLACES=cores` | #289 / closes #276 |
| 2026-06-24 | PR-C0 merged: SHUD submodule pointer bump `3a0004c → 9d22e17`（delete dead legacy `f_update/f_loop/f_applyDY`） | #292 / closes #291 |
| 2026-06-24 | PR-C merged: `rhs_update.cpp` element 块 steady-state first-touch loop + OQ1 doc append | #293 / closes #277 |
| 2026-06-24 | PR-D merged: `rhs_flux.cpp` river 块 first-touch + OQ1 append | #294 / closes #278 |
| 2026-06-24 | PR-E merged: `rhs_update.cpp` lake 块 first-touch + 3 negative grep gate + qhh 自验证 | #295 / closes #279 |
| 2026-06-24 | PR-F merged: server intermediate 8-cell（NUMA env + first-touch IN，Kahan IN）+ `numactl --interleave=all` 实证有害 | #296 / closes #280 |
| 2026-06-24 | PR-G merged: SHUD Kahan revert（`3a0004c → 210ac19`，4 surgical reverts 共 +7/-47）+ Mac 9-SHA matrix 证明 clean revert | #297 / closes #281 |
| 2026-06-24 | PR-H merged: server final 8-cell 三 SHALL gate verdict **FAIL** + post-verdict E′ rewrite（5/5 事实核查 + 4 错误纠正 + F 路 next epic）| #298 / closes #282 |
| 2026-06-24 | PR-I merged: Mac 独立 worktree at `P1-update-omp-tag` reference 采集（keliya + qhh × 3 mode = 6 cell SHA + nst + wall） | #299 / closes #283 |
| 2026-06-24 | master plan v1.5 / M10 sync revision: +180/-7 additive；新增 §6 P1d + §6 P1e + §7.2 RISK-NEW1/2 + §8.1 4-mode | #300 |
| 2026-06-24 | PR-K merged: capstone docs（4 new + 2 update + first_touch_design §"3 pragma 实现" DEPRECATED note + go_nogo placeholder）| #301 / closes #285 |
| 2026-06-24 | ADR-0002 (solver-path) merged: 4 路对比（Path 1 SELECTED for P1e）+ 15 文件 cross-ref 修正（ADR-0001 被 SoA hot fields 占用，renumber 0001 → 0002） | #302 |
| 2026-06-24 | PR-L merged: P1d-tag annotated procedure 草拟 + `p1d_summary.md §13` placeholder | #303 / closes #286 |
| 2026-06-24 | PR-M merged: Go/No-Go verdict 填实 + PROMOTE 2 specs（byte-identical）+ glossary 4 新术语 + 2 既存 term 更新 + jsonl 2 entries + PR-L §5 SHA 修正 | #304 / closes #287 |
| 2026-06-24 | post-PR-L merge: `git tag -a P1d-tag f88f2dc -F /tmp/p1d-tag-msg.txt && git push origin P1d-tag` → tag-object `a82bf336...` + deref `f88f2dc2...` | n/a |
| 2026-06-24 | post-PR-M merge: `openspec/changes/p1d-numa-governance → openspec/changes/archive/2026-06-24-p1d-numa-governance/`（gitignored）+ `gh api .../branches/baseline/P1d/protection PUT lock_branch=true enforce_admins=true ...` | n/a |
| 2026-06-24 | main promote PR: 因 main 含 4 个 P1c-era 独立 commit 不可 FF，改 merge PR + `--ours` 解决 3 处冲突（master plan / status_matrix / glossary 取 M10 v1.5 一侧），main 至 `4a520bb` 含 `baseline/P1d` ancestor + 4 P1c housekeeping commits | #306 |
| 2026-06-24 | Epic #274 CLOSED（auto-close via "Closes #274" in PR-M body） | #274 close |

15 PR（含 PR-A through PR-M 共 13 + master plan + ADR-0002 + main promote）+ 1 epic = 16 issues / PR 在 2 日 burst（2026-06-23 启动 + 2026-06-24 全部 close）完成。PR-J (#284) 跳过（per E′ narrative，PR-I + PR-G Mac 9-SHA matrix 已覆盖 Mac reference scope）。

## §15 反向兼容判定（per PR-I + PR-G Mac 9-SHA matrix，对应 P1c §8 模板）

P1d epic 结束时 `baseline/P1d` HEAD（outer `85fc567` / SHUD `210ac19`）NUM_OPENMP=1 binary SHA 与 `P1-update-omp-tag` canonical SHA 关系：

| Layer | 状态 | 解释 |
|---|---|---|
| Tag SHA immutability (D11) | ✓ PRESERVED | `P1-update-omp-tag` annotated tag SHA 永不变（D11 6-tag chain 5 historical 全部字面一致） |
| Mac N=1 binary SHA（serial mode）| ✓ EQUIVALENT | PR-G Mac 9-SHA matrix + PR-I Mac 6-cell reference 双侧实证：`keliya serial SHA = 89686fb8c97a3852...e99a8fc` byte-identical 至 `P1-update-omp-tag` 期 SHUD@`07c677f` build；`qhh serial` 同样 byte-identical |
| Mac N=1 binary SHA（omp@N=1 path）| ✓ EQUIVALENT | `keliya omp@N=1 PROC_BIND=close = b23e15b9...033a51a1a` byte-identical 至 pre-Kahan (`de9545d`) → 证明 Kahan revert + first-touch 叠加在 N=1 path 上数学等价于 `P1-update-omp-tag` 期 |
| Server N=1 binary SHA（serial / omp@N=1）| ✓ PASS at canonical heihe | PR-H server heihe N=1 SHA = `7f22bd6faa438d50...` byte-identical 至 spec L123 预写 canonical（全 64-hex 一致） |
| Server N=1 binary SHA（heihe_x4 / 其它 case）| ⚠ PARTIAL（spec 未预写）| `heihe_x4 N=1 = 55403bef48ee5ad8...` 无 spec 预写 canonical reference；性质上仍 N=1 deterministic（强 implication of N=1 reverse-compat），但缺正式 baseline 比对 → 标 PARTIAL pending P1e 期补 reference baseline |
| N≥2 binary SHA cross-N（fast-omp mode = `shud_omp` 当前 build）| ✗ DIVERGED（设计内） | `shud_omp` = Serial RHS + NVECTOR_OPENMP backend，跨 N reduction tree 顺序非 deterministic → 跨 N rivqdown.dat mean rel error 10-25% / max rel 1000-12500× / nst Δ=80-152；E′ closure 已把 `shud_omp` 标 `fast-omp experimental, non-production` |
| Helper-wrap layer（owner-local gather, P1c 引入）| ✓ EQUIVALENT to pre-Kahan | PR-G revert SHUD `3a0004c → 210ac19` 4 surgical reverts 后，所有 8 site helper-wrap 数学等价于 `de9545d`（pre-Kahan helper-wrap baseline）→ 还原 P1c 之前的 owner-gather FP 序 |

**Trade-off accepted**（per master plan §6 P1d.4 E′ closure narrative）：production 默认 `cfg.para NUM_OPENMP=1`（serial mode）→ 完整 N=1 reverse-compat preserved；`shud_omp`（fast-omp mode）保留作 research / experimental 但明确 non-production。Strict cross-N bitwise + nst Δ=0 待 P1e（F 路）`strict-omp` mode 实现（Serial N_Vector + StrictOMP RHS 替换 `MD_rhs_core.cpp:802-811` abort 桩）后真正 close。

## §16 限制与已知问题（per P1c §9 模板）

1. **N≥4 cross-N A3a bitwise + nst Δ=0 SHALL gate FAIL（设计内 PARTIAL CLOSURE）** — `shud_omp` (fast-omp mode) 当前 build 跨 N 不可复现，根因 NVECTOR_OPENMP `N_VDotProd_OpenMP` `reduction(+:sum) schedule(static)` 顺序不固定 + Serial RHS 没真正受益于并行；待 P1e (F 路) `strict-omp` mode 实现真正 close（per master plan §6 P1e + ADR-0002 Path 1 SELECTED）。
2. **PR-C/D/E steady-state first-touch loops 在 current `shud_omp` build 下无效** — `f.cpp:54` 始终调 `ExecPolicy::Serial`，consumer 是单线程，无 NUMA locality 收益；M10 标 DEPRECATED + 保留作历史档案。P1e 删除并重新设计为 owner-compute（StrictOMP）配套。**allocation-time first-touch (`Model_Data.cpp::malloc_EleRiv` L251-L346) 仍有效保留**。
3. **`heihe_x4 N=1` 缺 spec 预写 canonical SHA reference** — spec L123 仅预写 `heihe` 的 canonical；`heihe_x4 N=1 = 55403bef48ee5ad8...` 标记 PARTIAL；P1e 期补 reference baseline（per §15 反向兼容表）。
4. **Server PR-H wall + speedup 数据有限**（仅 heihe + heihe_x4 各 4-N，无 keliya / qhh / qinyijiang server-side baseline）— 当前 Amdahl 反推 87% / 76% 仅适用 heihe / heihe_x4 mesh；其它 case server-side 加速比待 P1e 2×2 build matrix 因果实验补齐（含 keliya / qhh / heihe / heihe_x4 全 4 case × 4 build × 4 N × 3 reps）。
5. **PR-L agent §5 historical SHA 表 off-by-one 错位** — PR-L commit 历史含错 SHA 表（5 行 shift by one row）；PR-M（#304）修正 + §5.1 SHA correction note 入档（per `docs/p1d/p1d_tag_and_lock.md`）。本错位不影响 D11 chain immutability，因 P1d-tag 创建命令使用的是 `git rev-parse <baseline/P1d HEAD>` 实测 SHA 而非 PR-L doc 中错位值。
6. **main 不可 FF 至 baseline/P1d** — main 有 4 个独立 P1c-era commits (`859e8e7` P9→P1d rename / `525d54c` tail fix / `7174ec6` docs/p1c/ 归档 / `d8cc666` p1c_report.md)，不在 `baseline/P1d` 祖先链上 → 不能直接 `git push origin baseline/P1d:main`；通过 #306 merge PR + `--ours` 3 处冲突解决合入 main（master plan / status_matrix / glossary 取 M10 v1.5 一侧）。这是 P1c epic 期间 main 与 baseline/P1c 路由偏离的遗留问题，P1d 期间被动消化；P1e 期间应通过严格 baseline → main FF 协议避免再次出现。
7. **`openspec/changes/p1e-strict-omp-rhs/` 5 文件 local drafts 未入 git**（per project 约定 `openspec/changes/` 是 gitignored 工作区）— ADR-0002 已合入 main 作 forward handoff，但 p1e openspec change 4 件（proposal/design/tasks/spec）+ capstone spec 仍是 Mac local 待 P1e epic 启动时迁移；不影响 P1d closure 完整性，但若工作机迁移需手动复制保留。
8. **PR-J (#284) skipped** — 原 plan PR-J 为 "P1d-binary Mac 比对"；E′ closure 后 PR-I + PR-G Mac 9-SHA matrix 已覆盖 Mac reference scope；issue #284 保留 OPEN 状态，可选 close 为 "skipped per E′" 或 retarget 至 P1e 期作 strict-omp mode Mac 验证。
