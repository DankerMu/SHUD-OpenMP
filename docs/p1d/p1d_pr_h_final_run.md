# P1d PR-H — server final 8-cell three SHALL gate verdict (#282)

## Verdict: **FAIL** (A3a + nst Δ=0 gates FAIL at N≥4) — burst closure via E′ path

Per `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` L149-150:
- **WHEN** PR-H 三 SHALL gate 任一项 FAIL
- **THEN** PR-H verdict SHALL 为 FAIL + epic issue 标 blocked + 不进入 PR-I/J/K/L/M

**Post-verdict update (2026-06-24)**: 用户决策后，P1d 走 **E′ closure path**——burst 继续但 narrative 重写：PR-H 保留 FAIL verdict + 4-mode spec rewrite + `shud_omp` 标 `fast-omp experimental, non-production`；下个 epic（P1e，新建）走 **F 路：Serial N_Vector + StrictOMP RHS**。Master plan M10 同步修订。详见本文档末尾 "post-verdict 修订" 章节。

## Scope

Production server (`frd_muziyao@210.77.77.22:32099`) 8-cell matrix:
- Cases: `heihe` (NumEle=6335) + `heihe_x4` (NumEle ~25000)
- Thread counts: N ∈ {1, 2, 4, 8}
- Per `docs/p1d/p1d_pr_f_intermediate_run.md` finding: **DROP** `numactl --interleave=all` (PR-B runbook prescription is anti-pattern with first-touch active).
- Env: `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `OMP_NUM_THREADS=${N}` only.

## Environment

- Outer baseline/P1d HEAD: `21de1e2` (PR-G merge commit)
- SHUD openmp-baseline HEAD: `210ac19` (post-PR-G Kahan revert; 3 reduction helpers restored to naive form; PR-C/D/E first-touch loops preserved byte-identical)
- Server build: `make clean && make shud_omp` PASS; FP strict 3-grep `-ffp-contract=off + -fno-fast-math + -fopenmp` upheld; `-ffast-math / -Ofast` 0
- Slurm 三铁律 honored: sbatch + run dirs + logs all under `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/`
- Cluster: CPU partition, 1 node × 8 cpus-per-task per job; jobs ran in parallel on `cn03` + `cn07` (8-cell wall-clock end-to-end ~20 min)
- Slurm job IDs: 9041-9048

## NUMA gate confirmation

Every cell's stdout emits the `[NUMA]` tokens proving `g_numa_first_touch_enabled = 1`:

```
[NUMA] OMP_PROC_BIND=close
[NUMA] first-touch begin tag=hot.soa
[NUMA] first-touch begin tag=QeleSurf_flat
[NUMA] first-touch begin tag=Ele_AoS
[NUMA] first-touch begin tag=LoadIC
```

## PR-H SHA matrix (full 64-hex)

| Cell | PR-H SHA256 (post-Kahan-revert + first-touch + no --interleave) | wall (s) |
|---|---|---|
| heihe N=1 | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | 513 |
| heihe N=2 | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | 513 |
| heihe N=4 | `fe7f1f071810d519617f3edad8918a799f237fb5f93cc727367000955318659a` | 503 |
| heihe N=8 | `e67b6592fa75d90afb077e72448436dd040e21871378e862576777b8eff1fe83` | 456 |
| heihe_x4 N=1 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | 1187 |
| heihe_x4 N=2 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | 1169 |
| heihe_x4 N=4 | `81e13d7aed9bb4739e77eedf2b0ee143129445251262ebd3dc3068c573fb2f86` | 1096 |
| heihe_x4 N=8 | `2099a1f7d303a87e4b791f050638969628f6ecbbd569178eeabf7c2d4ff13281` | 937 |

## PR-H CVODE stats (nst, nfe)

| Cell | nst | nfe |
|---|---|---|
| heihe N=1 | 6773 | 7035 |
| heihe N=2 | 6773 | 7035 |
| heihe N=4 | 6693 | 6949 |
| heihe N=8 | 6621 | 6901 |
| heihe_x4 N=1 | 6571 | 6733 |
| heihe_x4 N=2 | 6571 | 6733 |
| heihe_x4 N=4 | 6582 | 6749 |
| heihe_x4 N=8 | 6575 | 6742 |

## Three SHALL gate verdict (spec.md L126-150)

### Gate 0 — Kahan revert canonical SHA (spec L123 Scenario)

> **THEN** 输出 `output/heihe.out/heihe.rivqdown.dat` SHA256 SHALL 等同 `P1-update-omp-tag` canonical SHA（`7f22bd6faa438d50...`）

- Spec canonical SHA prefix: `7f22bd6faa438d50...`
- PR-H heihe N=1 actual: `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`
- **Verdict: PASS** — byte-identical to spec canonical. Proves PR-G revert restored the pre-Kahan SHUD trajectory; first-touch stacked above doesn't disturb it at N=1.

### Gate 1 — §4.4 A3a bitwise across N (spec L130 Scenario, server only)

> **THEN** heihe + heihe_x4 每 case 的 N ∈ {1,2,4,8} 4 cell SHA256 SHALL 全等

| Case | N=1 SHA (12) | N=2 SHA (12) | N=4 SHA (12) | N=8 SHA (12) | All equal? |
|---|---|---|---|---|---|
| heihe | `7f22bd6faa43` | `7f22bd6faa43` | `fe7f1f071810` | `e67b6592fa75` | **NO (3 distinct)** |
| heihe_x4 | `55403bef48ee` | `55403bef48ee` | `81e13d7aed9b` | `2099a1f7d303` | **NO (3 distinct)** |

**Verdict: FAIL** for both cases. N=1/N=2 are stable (low-thread bitwise reproducible), but N=4 and N=8 each produce distinct SHA. This residual variation is the SAME pattern P1c era exhibited (with Kahan IN) — Kahan revert did NOT change the residual structure.

### Gate 2 — §4.5 nst Δ=0 + ladder (spec L139 Scenario, server only)

> **THEN** heihe `nst` 跨 N ∈ {1,2,4,8} SHALL 全等（Δ=0 严格 hard gate）
> **AND** heihe_x4 `nst` 跨 N SHALL `|Δ_nst| ≤ 2`

| Case | nst N=1 | nst N=2 | nst N=4 | nst N=8 | Δ N=2 | Δ N=4 | Δ N=8 | Strict criterion | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| heihe | 6773 | 6773 | 6693 | 6621 | 0 | **80** | **152** | Δ=0 strict | **FAIL** |
| heihe_x4 | 6571 | 6571 | 6582 | 6575 | 0 | **11** | 4 | \|Δ\|≤2 ladder | **FAIL** (N=4: 11>2; N=8: 4>2) |

**Verdict: FAIL** for both cases at N≥4.

### Gate 3 — N=1 reverse-compat (spec L145 Scenario, 6-case)

> **THEN** 6 case × NUM_OPENMP=1 SHA256 SHALL byte-identical 至 `P1-update-omp-tag` canonical SHA

Server portion (this PR):
- heihe N=1: `7f22bd6faa438d50...` == spec canonical `7f22bd6faa438d50...` ✓ **PASS**
- heihe_x4 N=1: `55403bef48ee5ad8...` — no spec canonical SHA pre-written; need P1-update-omp-tag era server heihe_x4 N=1 reference. **DEFERRED** (not in spec L123 pre-written value).

Mac portion (PR-J): pending PR-I reference + PR-J comparison; NOT executed per spec L150 (PR-H FAIL stops burst).

**Verdict: PARTIAL PASS** (heihe N=1 server portion satisfies the only spec-pre-written canonical SHA). Other server/Mac cells deferred per BLOCKED stop protocol.

## Comparison with P1c Kahan baseline (informational)

P1c reference (`/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs-kahan/`, Kahan IN, no NUMA env, no first-touch):

| Cell | P1c nst | PR-H nst | Δ_nst | P1c SHA(12) | PR-H SHA(12) | SHA equal? |
|---|---|---|---|---|---|---|
| heihe N=1 | 6553 | 6773 | +220 | `fd2d55716b5d` | `7f22bd6faa43` | NO |
| heihe N=2 | 6553 | 6773 | +220 | `fd2d55716b5d` | `7f22bd6faa43` | NO |
| heihe N=4 | 6524 | 6693 | +169 | `e058db2e9c2a` | `fe7f1f071810` | NO |
| heihe N=8 | 6608 | 6621 | +13 | `6285e8a4a30a` | `e67b6592fa75` | NO |
| heihe_x4 N=1 | 6571 | 6571 | 0 | `4eb804f571ba` | `55403bef48ee` | NO |
| heihe_x4 N=2 | 6571 | 6571 | 0 | `4eb804f571ba` | `55403bef48ee` | NO |
| heihe_x4 N=4 | 6574 | 6582 | +8 | `ff0787abd217` | `81e13d7aed9b` | NO |
| heihe_x4 N=8 | 6569 | 6575 | +6 | `6e9f9a2eaf65` | `2099a1f7d303` | NO |

Observations:
- PR-H heihe N=1 ≠ P1c heihe N=1 (expected: Kahan IN/OUT alters CVODE adaptive trajectory; Mac PR-G already demonstrated this with keliya: pre-K2 vs Kahan-IN differ at every config). Server pattern matches Mac evidence.
- PR-H heihe_x4 N=1 nst Δ=0 vs P1c (both = 6571) but SHA differs (different CVODE trajectory still produces different downstream FP sequence).
- PR-H heihe N=1 == spec L123 pre-written canonical `7f22bd6faa438d50...` (this PR's PASS evidence).

## Comparison with PR-F v2 (Kahan IN baseline, --interleave=all)

PR-F v2 SHA same as P1c (Kahan IN preserves trajectory at N=1/N=2). PR-F v2 walls vs PR-H walls — confirms PR-F finding that `--interleave=all` is anti-pattern (PR-H without it gets heihe_x4 walls back to v1 range ~1187-937s vs PR-F v2 ~1305-1037s).

| Cell | PR-F v2 wall | PR-H wall | Δ |
|---|---|---|---|
| heihe_x4 N=1 | 1305 | 1187 | -118 (-9%) |
| heihe_x4 N=8 | 1037 | 937 | -100 (-10%) |

Drop of `--interleave=all` regained 9-10% wall on heihe_x4. PR-F finding confirmed.

## Root cause analysis (post-fact-check, 2026-06-24)

> ⚠️ **本节根因分析已修订**。初版基于 4 个错误前提（multi-threaded preconditioner / Amdahl 72% / KLU 唯一根治 / "CVODE tolerance 被打穿四个数量级"），经过两轮独立 GPT Pro 复查 + codebase 事实核查全部推翻。正确根因见 "post-verdict 修订" 章节。

Design D9 大方向是对的（drift origin 不在 P1c §4.7 8 站点 reduction helpers 内部），但具体定位错了——drift 不是 "SPGMR multi-threaded preconditioner"（事实：SPGMR 没注册任何 preconditioner），而是 **SUNDIALS 6.0.0 `NVECTOR_OPENMP` 内的 `N_VDotProd_OpenMP` + `N_VWSqrSumLocal_OpenMP` 用 `reduction(+:sum) schedule(static)`，跨 N reduction tree 顺序不固定**。

## Status

- PR-H verdict: **FAIL**
- Epic #274: marked BLOCKED via gh issue comment
- PR-I (Mac worktree reference): completed but work product held (not committed/pushed) pending user decision
- PR-J/K/L/M: NOT started per spec L150

## Artifacts (server, gitignored)

- sbatch: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/run_p1d_pr_h.sbatch`
- 8 run dirs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/{heihe,heihe_x4}_N{1,2,4,8}/`
- Each run dir contains: `rivqdown.sha256`, `wall.txt`, `cvode_stats.txt` (15-key set), `output_listing.txt`, `done.txt`
- Slurm logs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/logs/p1d_pr_h_{9041..9048}.{out,err}`

## Post-verdict 修订（2026-06-24, fact-check + E′ + F 路）

初版 verdict 把 PR-H FAIL 解读为 "nst Δ 超 spec ladder + 根因在 SPGMR multi-threaded preconditioner + KLU 唯一根治"。**两轮独立 GPT Pro 复查 + codebase 事实核查全部推翻这个解读**。本节为唯一权威修订。

### 关键事实核查（5/5 全部支持错误诊断纠正）

| # | 断言 | 核查 | 结论 |
|---|---|---|---|
| 1 | `f()` 始终调 Serial RHS | `SHUD/src/Model/f.cpp:54` `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` | ✅ |
| 2 | `StrictOMP/ProductionOMP` 是 abort 桩 | `SHUD/src/Model/MD_rhs_core.cpp:802-811` 三 case 全 `std::abort()` | ✅ |
| 3 | `shud_omp` build 硬编码 NVECTOR_OPENMP | `SHUD/Makefile` shud_omp target: `-DSHUD_USE_OPENMP_NVECTOR=1 -lsundials_nvecopenmp` | ✅ |
| 4 | `SHUD_USE_OPENMP_NVECTOR` 与 `SHUD_ENABLE_OPENMP_RHS` 正交，后者默认 0 | `SHUD/Makefile:140` `SHUD_ENABLE_OPENMP_RHS ?= 0` | ✅ |
| 5 | SPGMR 没注册 preconditioner | `SHUD/src/Equations/cvode_config.cpp:259` `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` 后无 `CVodeSetPreconditioner` 调用 | ✅ |

**含义**：`shud_omp` 当前实际跑的是 **Serial 水文 RHS + OpenMP N_Vector backend**，**不是** 真正的 hydrology RHS OpenMP 并行。PR-C/D/E 添加的 first-touch loops 是为**完全没发生的** parallel RHS owner-compute 做的页面预放置——consumer 是单线程，根本无视 NUMA locality。

### 初版 4 个错误结论的更正

| 初版（错） | 修订（对） |
|---|---|
| "drift origin 在 SPGMR multi-threaded preconditioner" | SPGMR 没有 preconditioner；drift 是 `N_VDotProd_OpenMP` 的 `reduction(+:sum) schedule(static)` 跨 N reduction tree 顺序不固定（确认源码 [SUNDIALS 6.0.0 `nvector_openmp.c`](https://github.com/LLNL/sundials/blob/v6.0.0/src/nvector/openmp/nvector_openmp.c)）。WRMS norm 同样过 OpenMP reduction |
| "Amdahl serial fraction ~72%" | 由 wall 反推 `f = (1/S - 1/N)/(1 - 1/N)`：heihe ~**87%**，heihe_x4 ~**76%**。且不能全部归因 CVODE——含 serial RHS + N_Vector fork/join 开销 + memory bandwidth 饱和 |
| "reltol=1e-4 但实测 10% → solver 承诺被打穿四个数量级" | CVODE reltol 控制的是**每步状态 WRMS local error**，不是 90 天派生流量轨迹的全局相对误差上界。10% 不可接受是工程结论，不是 CVODE 实现错误的证据 |
| "KLU 是唯一根治路径" | KLU 不能单独 fix。CVODE WRMS norm 还是过 N_Vector reduction，换 KLU 不换 N_Vector 仍然漂。Determinism 与 solver 选择**正交** |

### 实测 rivqdown.dat 数值散度（PR-H, 90-day heihe / heihe_x4）

| Case / N | mean_rel | RMSE | max_rel | n_diff / n_total |
|---|---|---|---|---|
| heihe N=2 vs N=1 | 0 | 0 | 0 | 0 / 214252 |
| heihe N=4 vs N=1 | **3.8%** | 1.79e+03 | 1010× | 210774 / 214252 |
| heihe N=8 vs N=1 | **10.0%** | 2.76e+03 | 12534× | 210779 / 214252 |
| heihe_x4 N=2 vs N=1 | 0 | 0 | 0 | 0 / 387607 |
| heihe_x4 N=4 vs N=1 | **17.6%** | 8.63e+03 | 2214× | 383130 / 387607 |
| heihe_x4 N=8 vs N=1 | **25.1%** | 7.59e+03 | 2908× | 382116 / 387607 |

**P1c era 对照**（已上线版本，Kahan IN，no NUMA env，no first-touch）：

| Case / N | P1c mean_rel | P1c max_rel |
|---|---|---|
| heihe N=8 vs N=1 | 10.0% | 1446× |
| heihe_x4 N=8 vs N=1 | 20.3% | 66120× |

P1c 早就以 10-25% N≥4 流量散度上线，**当时未补做这个测算**。PR-H 实测把 P1c 遗留 finding 显式化。Kahan 注入只压住 nst step count，没修水文输出散度。注意 `max_rel` 1000-66000× 部分是被极小 denominator (近 0 流量) 放大，需要后续审计是否引入物理 floor；但 `mean_rel` 10-25% 是工程层面的硬实锤。

### 并行加速比 + Amdahl 反推

| Case | N=1 wall | N=8 wall | Speedup `S8` | 并行效率 `S8/8` | CPU-hour cost |
|---|---|---|---|---|---|
| heihe | 513s | 456s | 1.13× | 14.1% | 6.32× serial |
| heihe_x4 | 1187s | 937s | 1.27× | 15.9% | 6.30× serial |

8 cores 换 1.13-1.27× 加速 + 10-25% 流量误差 → **ROI 为负**。

**关键洞察**：项目早期 profile 显示 heihe_x4 **RHS 占 wall 66.55%**，理想 8 核 Amdahl 上界约 **2.39×**。当前 1.27× 的主要问题不是 "Amdahl 已到极限"，而是 **真正应并行的 RHS 还没并行**（StrictOMP 路径还是 abort 桩）。

### K=200 vs K=0 在流量层面的真实含义

| Mode | nst Δ | 流量散度 | 含义 |
|---|---|---|---|
| K=0 (true strict) | 0 | ≤ULP | 真 reproducible |
| K=200 (PR-H 实测) | 80-152 (heihe) | mean 10-25% rel error | **另一条 CVODE 轨迹**，非 ULP 近似 |

对照：vs CMFD forcing 不确定性 (10-20%) → 同量级；vs gauge 精度 (5-10%) → 超过测量精度；vs Manning's n 标定带宽 (~50%) → 量级以下。

### Remediation 路径决策（实测后修订）

| 路径 | 描述 | 评估 |
|---|---|---|
| A | 放宽 ladder K=200 | ❌ false advertising "strict" |
| B | SPGMR → KLU 重构 | P2/P3 ADR 评估，**不是 P1d 必选**——KLU 单独不能 fix N_Vector reduction |
| C | SUNDIALS 内部 deterministic patch | ❌ 验收周期长，未必彻底 fix |
| D | N=1/2 bitwise + N≥4 ladder 双 mode | 配合 E′ |
| **E′** | **P1d containment closure** | **当前 P1d 收尾路线**（不是简单 E，含 4-mode spec + 错误叙事更正 + first-touch deprecation note） |
| **F** | **Serial N_Vector + StrictOMP RHS** | **下个 epic (P1e) 技术主线**——真正完成 P1d 原计划应该完成的事 |

### E′ 路（P1d closure）具体动作

1. **production 默认 `cfg.para NUM_OPENMP=1`**（serial path 实测仅比 N=8 慢 11%，但可 reproducible）
2. **`shud_omp` 标 `fast-omp experimental, non-production`**（保留 build + CI）
3. **PR-K capstone docs 诚实记录**：
   > P1d 完成 NUMA env standardization + 数值清理（PR-C/D/E first-touch loops + PR-G Kahan revert）；**但 hydrology RHS 仍是 Serial**——`ExecPolicy::StrictOMP` / `ProductionOMP` 路径仍为 `std::abort()` 桩，等待 P1e (F 路) 实现。当前 `shud_omp` 实际跑的是 `Serial RHS + NVECTOR_OPENMP backend`，跨 N 不可复现，不推荐 production 使用。
4. **三 SHALL gate 4-mode 重写**（不弱化 strict 承诺，限定到正确 mode）：
   - `serial` mode (N=1): SHALL canonical bitwise vs P1-update-omp-tag
   - `strict-omp` mode (待 P1e 实现): SHALL 跨 N bitwise + nst Δ=0 + N=1 reverse-compat **strict** (这才是原始 SHALL gate 真正适用的 mode)
   - `fast-omp` mode (=当前 `shud_omp`): MAY 不可复现，明确 non-production
5. **PR-C/D/E first-touch deprecation note**：steady-state first-touch 是 owner-compute 配套；owner-compute 没实现，所以这些 loops 在当前 build 下是无效优化 + 带宽浪费。Allocation-time first-touch 保留。下个 epic P1e 重新设计为 owner-compute 配套。
6. **P1d-tag annotated message** 写 containment closure narrative + 指向 P1e
7. **PR-I 数据 commit**（N=1 6-case Mac reference anchor 仍有 reproducibility 价值）

### F 路（P1e 新 epic）顶层 plan

技术主线：**Serial N_Vector + StrictOMP RHS**。CVODE/SPGMR 继续用 Serial N_Vector（保证 reduction 顺序确定），SHUD 水文 RHS 真正并行（替换 abort 桩）。

启动前**必须先跑 2×2 因果实验**：

| Build | N_Vector | RHS | 目的 |
|---|---|---|---|
| A | Serial | Serial | canonical reference |
| B | OpenMP | Serial | =当前 `shud_omp`，复现 10-25% 散度作 control |
| C | **Serial** | **StrictOMP** | **P1e 候选 production** |
| D | OpenMP | StrictOMP | research only |

每 build × N∈{1,2,4,8} × 3 repeats，hash `CV_Y` + `rivqdown.dat` + capture `nst/nfe/nli/nni/netf/ncfn`。判据：
- B 跨 N 不同而 A 相同 → 确认 NVECTOR_OPENMP reduction 是主因
- C 跨 N bitwise + nst Δ=0 + 加速 → F 路成立
- C 也分叉 → 查 RHS race / 共享状态 / phase dependency

F 路 epic 具体实现要点：
- `ExecPolicy::StrictOMP` 路径替换 `std::abort()` 桩
- 单 parallel region + phase-based for + `default(none)` + 隐式 barrier
- 复用现有 `rhs_deterministic_gather()`：并行 owner 外层 + canonical fold 内层
- 拆 `NUM_RHS_THREADS` + `NUM_NVECTOR_THREADS` 配置项
- 删 steady-state first-touch；保留 allocation-time first-touch（per Pro 共识）
- `rivqdown.dat` 输出缓存 audit（per Pro2 警示：要确认输出是从 `tout` 状态重算，不是直接写最后一次 RHS 内部缓存）

### 后续 ADR 路线图（不阻塞 F）

- `docs/adr/0002-solver-path.md`（新建）：4 路对比 — Serial N_Vector + StrictOMP RHS / Deterministic NVECTOR_REPRO_OMP / SPGMR + block-Jacobi precond / KLU sparse direct prototype。F 路是首选，KLU 推后做 pattern-only spike 量化 fill ratio + memory peak + factor wall

## Master plan 同步修订

P1d 实测结论与 v1.4 / M9 master plan 偏离很大，已通过 **v1.5 / M10 修订** 同步：
- 顶部 M10 修订要点 quote block
- §1.1.2 加 P1d / P1e 档 + 4-mode 表述
- §3 路线图加 P1d → P1e 节点
- §4.13 + §4.17 加事实核查补充
- §6 新增 P1d + P1e 章节
- §6 P1c.5 / §6 P2a 启动前置 改链路 (P1c → P1d → P1e → P2a)
- §7.2 加 RISK-NEW (NVECTOR_OPENMP reduction order)
- §7.3 加 P1d / P1e 行
- §8.1 strict 模式按 4-mode 拆

详见 `SHUD_openMP_master_plan.md` v1.5 M10 修订要点。
