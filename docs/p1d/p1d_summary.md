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
