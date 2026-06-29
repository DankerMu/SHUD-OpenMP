---
title: "P8-tune.F Epic — BoomerAMG/Hypre pattern-only spike work plan"
subtitle: "前置准备 (#386 fix) + 5-PR scope + ADR-0007 4-branch decision tree + 风险与缓解 + 预算与接续 epic triggers"
authors: ["SHUD-OpenMP 改造工程组"]
date: 2026-06-29
version: 1.0 (epic 启动前 actionable work plan)
status: "ANCHORED — 待 trigger (#386 fix 后启动 PR-0)"
priority: "HIGH (primary forward line per ADR-0005 §Forward action + GPT Pro F4 retrospective)"
epic: "p8tune-amg-spike (forthcoming;openspec change 待开 + GitHub epic issue 待开)"
budget: "~4 epic-weeks (5 PR;详 §6)"
prereq:
  - "[#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386) SHUD Model_Data 析构链 uninit-pointer fix (P8-tune.F PR-0 第一个 task,与 P8-tune.E.small-only PR-0 shared prereq)"
  - "P8-tune.D CLOSE (已完成 2026-06-29 PR-D #389 merge)"
related_docs:
  - "docs/adr/0005-klu-spike-decision.md §Forward action + §GPT Pro retrospective F4 (本 epic trigger source-of-truth)"
  - "docs/p8tune/klu_spike_verdict.md (P8-tune.D 16-cell verdict)"
  - "docs/p8tune/p8tune_d_academic_summary.md §10.5 (forward path 修订)"
  - "SHUD_openMP_master_plan.md §P8-tune.F (5-PR scope 锚)"
  - "docs/adr/0002-solver-path.md (Path 5 BoomerAMG/Hypre 决策起点 — forthcoming)"
related_prior_epics:
  - "P8-tune.D #379 (KLU pattern spike; wall axis 否决大 case)"
  - "P8-tune.C #362 (SPGMR maxl Optional-knob; Krylov saturated 大 case)"
  - "P8-tune.A/B (ADR-0003 PREC_NONE NO-GO)"
forward_anchors:
  - "P8-tune.G (full AMG + A5 hydrology-equivalence integration epic; trigger by ADR-0007 GO)"
  - "P8-tune.H (GPU sparse spike; trigger by ADR-0007 NO-GO heihe_x16 only)"
---

# §0 摘要 / Executive summary

P8-tune.F 是 SHUD-OpenMP 工程主线对大 case (`heihe_x4` NumY ≈124K, `heihe_x16` NumY ≈485K) wall 加速主线的 **primary forward 路径**, 由 ADR-0005 Case-aware verdict + GPT Pro 2026-06-29 F4 retrospective 二者共同确定。前序 ADR-0004 (SPGMR maxl Optional-knob) + ADR-0005 (KLU pattern-only spike) 分别在 Krylov vector working-set 与 numeric factor wall 两轴 saturated/NO-GO 大 case, 工程主线必须转向 algebraic multigrid (AMG) 求解器 — 该方法在 elliptic-parabolic PDE 上 O(N) memory + scalable 是 SHUD water flow domain 的 algebraically-correct 退路。

**目标**: 以 ~4 epic-week 投入 pattern/numeric-only spike (mirror P8-tune.D zero-source-patch + zero-CVODE-wireup discipline) 产出 ADR-0007 4-branch verdict (GO / Optional / NO-GO / BLOCKED), 决定下一 epic 是否启动 P8-tune.G (full AMG CVODE integration + A5).

**5-PR scope**:
- **PR-0**: [#386](https://github.com/DankerMu/SHUD-OpenMP/issues/386) SHUD Model_Data 析构链 uninit-pointer fix + P8-tune.D 工具链稳定性确认 (~0.5-2 days)
- **PR-A**: Hypre/BoomerAMG build + Mac keliya smoke (~3-5 days)
- **PR-B**: 服务器 12-16 cell Slurm array sweep (4 case × 4 (interp_type, coarsen_type) combo) (~5-7 days)
- **PR-C**: aggregator + ADR-0007 verdict (3-axis + AMG-specific cycle_complexity + operator_complexity + residual_reduction_per_V-cycle) (~3-5 days)
- **PR-D**: epic capstone + master plan close + OpenSpec archive + (conditional) trigger P8-tune.G or P8-tune.H (~2-3 days)

**Out of scope**: SHUD 源码 patch (Hypre spike 仅链 libshud.a, 类 P8-tune.D PR-0 carve-out); CVODE integration (deferred to P8-tune.G); A5 hydrology-equivalence (deferred to P8-tune.G — pattern-only spike 同 P8-tune.D 纪律).

**关键决策点**: AMD ordering 默认 (per P8-tune.D ADR-0005 ordering lock); 4 BoomerAMG (interp_type, coarsen_type) combo 由 Hypre best-practice 选取; ADR-0007 决策树 4-branch 见 §4。

---

# §1 Why now (trigger condition + 上游 ADR)

## §1.1 ADR-0005 §Forward action 触发 (per 2026-06-29 retrospective amendment)

ADR-0005 §Decision + §GPT Pro retrospective corrections 明确:

> **heihe_x4** = Optional (wall margin 1.87× over 0.7×SPGMR budget); **heihe_x16** = NO-GO (wall margin 17.9×). Forward action: **P8-tune.F BoomerAMG/Hypre = HIGH priority, primary line for 大 case 加速 main objective**.

KLU 在大 case 上 wall axis 否决的根因是 numeric factor 三角分解 wall 与 NumY^1.5-2.0 super-linear 增长(详 [`docs/p8tune/p8tune_d_academic_summary.md`](docs/p8tune/p8tune_d_academic_summary.md) §6.4)。AMG 的目标正好对应:setup wall O(N), apply wall O(N log N), iteration convergence 收敛率 ≈ V-cycle 1-3 step 内 residual 10× 衰减。这与 KLU 的 factor-once-solve-many 完全不同的工程 trade-off,适配 unsteady RHS frequent refactor 的 SHUD use case。

## §1.2 Prerequisite: #386 fix (per GPT Pro F3)

`Model_Data` 析构链 (`~Model_Data → FreeData → ~SubClass`) uninit-pointer UB 在 NumY > 100k 触发 `free(): invalid pointer` heap corruption。P8-tune.D spike 用 `_exit(0)` 绕过 dtor 调用, 该 workaround **不适用** P8-tune.F:

- P8-tune.F PR-A 工具链复用 `dump_adjacency` + `fd_color_jacobian` (P8-tune.D 的两个 spike 二进制), 这两 binary 内部仍依赖 `Model_Data` init 路径 — 若 init 后 spike main 添加 BoomerAMG setup/solve loop (用 RAII Hypre objects), spike 退出时若 Hypre dtor 在 Model_Data dtor 之前 fire, 会再次触发 UB
- 即使 `_exit(0)` 仍能屏蔽 dtor, 长期持续依赖 spike binary 用 `_exit(0)` workaround 是工程债务, 任何 future P8-tune.E.small-only / P8-tune.G full integration epic 必须根本修

故 #386 fix = P8-tune.F PR-0 第一个 task, 0.5-2 days 估时 (per ADR-0005 §9.3 investigation plan)。

## §1.3 与 P8-tune.D 工程哲学一致

P8-tune.F 复用 P8-tune.D 的 pattern-only spike 模式 — zero source patch (libshud.a 链), zero CVODE wireup (无 SUNLinSol_Hypre), zero SHUD model run (无 rivqdown.dat output), zero A5 hydrology-equivalence (deferred to P8-tune.G full integration epic)。这一模式由 P8-tune.D 经验证 cost-effective (~2-3 epic-week 产出 actionable architectural decision), 为 P8-tune.F 4-week 预算的基础。

---

# §2 Forthcoming dependencies (软件栈 + Hypre 选型)

## §2.1 Hypre 库选型 (BoomerAMG = Hypre 主要 AMG 实现)

| 项 | 选择 | 理由 |
|---|---|---|
| Library | **Hypre 2.30.0** (Latest stable; LLNL maintained) | 行业标准 PDE solver toolkit; 含 BoomerAMG + Hybrid + GMRES preconditioners; SUNDIALS 6.0.0 +SUNLinSol_Hypre interface (forthcoming P8-tune.G 复用) |
| 求解器 | **BoomerAMG** (classical Ruge-Stüben + extended interpolation) | 对 elliptic-parabolic PDE 在 unstructured mesh 上 setup O(N) + apply O(N log N); 是 Hypre 默认 AMG 推荐 |
| Backend | CPU-only (序列 + OpenMP) | P8-tune.F 是 pattern-only spike 不必考虑 GPU; GPU 由 forthcoming P8-tune.H spike 评估 |
| Install | `apt install libhypre-dev` (Ubuntu) / `brew install hypre` (Mac) | 类 P8-tune.D SuiteSparse install 模式 |
| Symbol verification | `nm libhypre.a | grep -E 'HYPRE_BoomerAMGCreate\|HYPRE_BoomerAMGSetup\|HYPRE_BoomerAMGSolve'` | PR-A 验收 |

**Alternative considered + rejected**:
- **PETSc HYPRE wrapper**: 引入额外 PETSc dependency 增加 工程 cost 2-3 week; PR-0 build complexity 不必要
- **AMGx (NVIDIA)**: GPU-only; 由 P8-tune.H spike 评估
- **MueLu (Trilinos)**: Trilinos build dependency tree 过深 (boost + teuchos + tpetra + epetra), 工程 risk 高
- **plain GMRES + ILU(K)** preconditioner: 与 SPGMR 同类 Krylov 方法 — P8-tune.C ADR-0004 已 saturated, AMG 才是 architecturally distinct retreat

## §2.2 ColPack reuse (per P8-tune.D PR-0 carve-out)

`tools/p8tune.D/fd_color_jacobian.cpp` 已实现 ColPack DISTANCE_TWO Welsh-Powell column coloring + CPR finite-difference Jacobian probe。P8-tune.F PR-A 复用该工具 + numeric J 输出, 不重新实现 coloring 或 FD probe。

## §2.3 libshud.a carve-out (per P8-tune.D PR-0 documented exception)

`SHUD/Makefile libshud.a` archive target 已建 (PR-0 #384 carve-out)。P8-tune.F PR-A `tools/p8tune.F/Makefile` 直接 `-L../SHUD -lshud -lhypre -lcolpack` 链接, 不修 SHUD 任何 .c/.cpp/.h 源。

## §2.4 OpenSpec change `p8tune-amg-spike` (forthcoming)

P8-tune.F PR-0 之前应建立:

```
openspec/changes/p8tune-amg-spike/
├── .openspec.yaml
├── proposal.md (why now + trigger condition)
├── design.md (D1-D8 design decisions; 类 p8tune-klu-spike 结构)
├── tasks.md (5-PR task breakdown)
└── specs/
    └── amg-pattern-spike-verdict/
        └── spec.md (REQ-1 to REQ-8 类 klu-pattern-spike-verdict)
```

由 PR-0 commit 前先 author + `openspec validate p8tune-amg-spike --strict`。详 §5 PR-0 deliverables。

---

# §3 5-PR scope 详细分解

## §3.1 PR-0: #386 fix + 工具链稳定性

**Scope**:
1. **#386 root-cause fix** (per ADR-0005 §9.3 investigation plan):
   - `grep -rn 'delete\[\]' SHUD/src/ModelData/` 找未初始化 ptr
   - `valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all` repro on keliya (NumY=1500 fast)
   - 修 uninit pointer (常见模式: 构造函数未初始化某 ptr, 析构 `delete[]` 时 invalid)
   - heihe (NumY=19515) 二次 valgrind 验证 — 若 keliya 通过但 heihe 失败, 说明 NumY-dependent path 存在 (e.g. lake 模块条件 init)
   - SHUD submodule push to `openmp-baseline` 分支 (per CLAUDE.md SHUD submodule 工作流)
   - SHUD pointer bump in outer commit
2. **CI 加强**: `.github/workflows/ci.yml` 的 `asan-ubsan (keliya)` job 显式 enable destructor coverage — 即跑完后不用 `_exit` 而是正常 dtor 路径(检查现 CI 是否已经全 dtor 路径,若是则只确认覆盖)
3. **spike binary 移除 `_exit(0)` workaround**: `tools/p8tune.D/{fd_color_jacobian,dump_adjacency}.cpp` 末把 `_exit(ok ? 0 : 1)` 改回 `return ok ? 0 : 1` — 等 #386 修了之后这一改动应不再触发 UB
4. **heihe_x4 + heihe_x16 二次 sanity smoke**: 跑 `dump_adjacency heihe_x4` + `dump_adjacency heihe_x16` (现有工具链)确认无 heap corruption — 这是 P8-tune.F 工具链稳定性 acceptance

**Deliverables**:
- SHUD `openmp-baseline` 分支 commit (uninit ptr fix)
- 外层 SHUD pointer bump commit
- `tools/p8tune.D/` 移除 `_exit(0)` workaround
- `.github/workflows/ci.yml` ASan destructor coverage 确认 enable
- 新 ADR-0006 (可选, 仅记录 #386 root cause + fix); 或 直接 commit message + master plan note 即可
- `.review-evidence/p8tune-amg-pr-0/{valgrind_keliya.log, valgrind_heihe.log, x4_smoke.log, x16_smoke.log}` evidence
- openspec change `p8tune-amg-spike` skeleton (proposal + design + tasks + spec stub)

**Acceptance**:
- `valgrind ./shud keliya` 0 errors 0 leaks
- `valgrind ./shud heihe` 0 errors 0 leaks
- `dump_adjacency heihe_x4` / `dump_adjacency heihe_x16` 完整 init+dtor 路径 0 corruption
- `openspec validate p8tune-amg-spike --strict` PASS
- CI all green

**Budget**: 0.5-2 days (depends on uninit ptr 根因复杂度;若 1 行 fix 半天, 若涉及 ownership 链 1-2 天)

**Dependencies**: P8-tune.D CLOSE (已完成)

## §3.2 PR-A: Hypre build + spike tool authoring + Mac smoke

**Scope**:
1. **Hypre install + symbol verification**:
   - Ubuntu cn-node: `apt install libhypre-dev` (检查 version; 若 < 2.20 build from source)
   - Mac local: `brew install hypre`
   - `nm libhypre.so | grep HYPRE_BoomerAMG` 验收 + 记录 SHA / version 进 evidence

2. **新工具二进制 `tools/p8tune.F/boomeramg_setup_solve.cpp`**:
   - 输入: numeric J (CSR 或 IJMatrix format, 复用 P8-tune.D PR-A fd_color_jacobian output)
   - Hypre API: `HYPRE_IJMatrixCreate` → `HYPRE_IJMatrixSetValues` → `HYPRE_IJMatrixAssemble` → `HYPRE_BoomerAMGCreate(&solver)` → `HYPRE_BoomerAMGSetInterpType(solver, interp_type)` → `HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type)` → `HYPRE_BoomerAMGSetup(solver, A, b, x)` (timed) → `HYPRE_BoomerAMGSolve(solver, A, b, x)` (timed; N_solve=5)
   - 输出 emit per spec REQ-5 (forthcoming) 的 cell_summary KV block:
     - `case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>`
     - `setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>`
     - `cycle_complexity=<C> operator_complexity=<OC> residual_reduction_v1=<R1>`
     - (option) `iteration_count_for_residual_10x_reduction=<IT>`

3. **`tools/p8tune.F/dump_adjacency` + `fd_color_jacobian` 复用**: 通过 `Makefile` symlink 或 `make -C tools/p8tune.D dump_adjacency fd_color_jacobian` 重用, 不复制源码

4. **Mac keliya smoke** (类 P8-tune.D PR-0 验收):
   - `make -C tools/p8tune.F all` build (Mac libhypre + libcolpack + libshud.a)
   - `./dump_adjacency keliya` 产 adjacency CSC
   - `./fd_color_jacobian keliya` 产 numeric J
   - `./boomeramg_setup_solve keliya --interp-type=6 --coarsen-type=8` 跑 BoomerAMG default config (classical extended interpolation + HMIS coarsening, Hypre 推荐 default)
   - `./boomeramg_setup_solve keliya --interp-type=14 --coarsen-type=10` 测 alt combo (extended+i + AGG-2-PMIS aggressive)

**Deliverables**:
- `tools/p8tune.F/Makefile` 新建 (链 libshud.a + libhypre + libcolpack)
- `tools/p8tune.F/boomeramg_setup_solve.cpp` 新建 (~300-500 行;Hypre setup+solve+timing+RSS 探针)
- `tools/p8tune.F/README.md` (Hypre symbol verification 步骤 + interp/coarsen combo 编号表 + cell_summary KV schema)
- `.review-evidence/p8tune-amg-pr-a/{mac_smoke_keliya_*, hypre_version.log, nm_libhypre.log}`
- master plan §P8-tune.F PR-A scope 行 status 更新

**Acceptance**:
- Mac smoke 4 combo runs 全部 exit 0 + emit cell_summary KV block
- `setup_wall_sec` + `apply_wall_sec` 非零 + `peak_rss_bytes` 合理 (~10-50 MB for keliya)
- chromatic χ + nnz(A) 与 P8-tune.D PR-0 验收 byte-identical (复用 fd_color_jacobian)
- CI all green (asan-ubsan on Mac smoke command — 注意 Hypre 内部 release 可能 trip ASan, 需要 ASan whitelist 或仅在 keliya 上验证)

**Budget**: 3-5 days (Hypre API 学习曲线 + interp/coarsen combo 编号查 Hypre docs + Mac brew build edge cases)

**Dependencies**: PR-0 merged

## §3.3 PR-B: 服务器 12-16 cell Slurm array sweep

**Scope**:
1. **Sweep matrix 设计** (类 P8-tune.D 4×4 matrix 但 AMG dimension 不同):

   **4 case** = keliya / heihe / heihe_x4 / heihe_x16 (同 P8-tune.D)
   
   **4 (interp_type, coarsen_type) combo** = Hypre BoomerAMG best-practice 选取:
   - Combo 0: (interp=6 classical extended, coarsen=8 HMIS) — Hypre default, robust baseline
   - Combo 1: (interp=14 extended+i, coarsen=10 HMIS) — agressive interpolation, 期望 setup ↑ apply ↓
   - Combo 2: (interp=6, coarsen=21 CGC) — alternate coarsening, 测 coarsening sensitivity
   - Combo 3: (interp=8 standard, coarsen=8) — fallback baseline if Combo 0 unstable
   
   总 4 × 4 = **16 cell**, 部分 cell (如 heihe_x16 × Combo 1 setup wall) 可能 timeout (需 8h wall budget per cell, 比 P8-tune.D 4h 更宽松因 Hypre setup O(N) 大 case 仍 wall-bound)

2. **`tools/p8tune.F/spike_array.sbatch`**:
   - `#SBATCH --array=0-15` 同 P8-tune.D 模式
   - cell decoder: `CASE = case_for(NN/4)`, `COMBO = combo_for(NN%4)`
   - SIGTERM trap → emit `AMG_WALL_OVERFLOW_DETECTED` marker (类 P8-tune.D KLU_WALL_OVERFLOW)
   - `/usr/bin/time -v` RSS 探针
   - 输出 `cell-NN.log` 含 cell_summary KV + 可能的 marker (AMG_OOM / AMG_SETUP_DIVERGE / AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW)

3. **执行 Slurm 三铁律 严守** (per CLAUDE.md):
   - 从 `/scratch` `sbatch`
   - `--output / --error` 路径在 `/scratch`
   - 所有 referenced patch / hash / runscript 在 `/scratch`
   - 提交位 `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/`

4. **cn-RAM probe re-verify**: 在 PR-B 任务 1.3.1 重跑 `cat /proc/meminfo` 确认 cn-node RAM 仍 173 GiB (P8-tune.D PR-0 一次 probe;PR-B 二次确认排除节点 drift)

5. **Per-cell raw evidence collection**: 跑完后 `rsync` 拉 cell-NN.log 到本地 `.review-evidence/p8tune-amg-pr-b/cells/`

**Deliverables**:
- `tools/p8tune.F/spike_array.sbatch` (类 P8-tune.D `spike_array.sbatch`)
- `tools/p8tune.F/run_cell.sh` (single-cell wrapper, 类 P8-tune.D)
- `tools/p8tune.F/precheck_env.sh` (类 P8-tune.D REQ-4 gate — 4 条件: cfg.para 90 天, V0200 forcing, heihe_x16 部署, cn-node RAM)
- `.review-evidence/p8tune-amg-pr-b/{cells/cell-{00..15}.log, SWEEP_RESULTS.md}` (类 P8-tune.D PR-A 结构)

**Acceptance**:
- 16/16 cell 产 verdict-class data (PASS / AMG_OOM / AMG_SETUP_DIVERGE / AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW)
- 4 case 每 case 至少 1 个 combo PASS (否则 ADR-0007 直接 NO-GO 该 case)
- cell-NN.log 完整记录 setup_wall + apply_wall + peak_rss + cycle_complexity + operator_complexity + residual_reduction_v1
- evidence rsync 完整, byte 一致

**Budget**: 5-7 days (16 cell 服务器 Slurm submit + monitor + 中断重跑 buffer; 类 P8-tune.D PR-A 实际 7h 跨节点 parallel + 中断 fix 8h)

**Dependencies**: PR-A merged

## §3.4 PR-C: aggregator + ADR-0007 verdict + verdict docs

**Scope**:
1. **`tools/p8tune.F/aggregate_amg_spike.sh`** (类 P8-tune.D `aggregate_klu_spike.sh`):
   - 解析 16 cell-NN.log
   - 计算 per-case best combo (准则: 最低 setup+apply combined wall; tiebreaker = operator_complexity)
   - 3-axis AMG verdict + 2 AMG-specific axis:
     - **Axis 1: Setup**: `setup_wall < 1.5 × 0.7 × SPGMR_per_step` (允许 setup 比 KLU factor 慢 1.5× 因 AMG setup 在生产是 amortized over N_solve refactor)
     - **Axis 2: Apply**: `apply_wall < 0.7 × SPGMR_per_step` (= 0.158605 s/step)
     - **Axis 3: Memory**: `peak_rss < 0.7 × cn_node_ram` (= 121 GiB, 同 P8-tune.D)
     - **Axis 4 (AMG-specific)**: `cycle_complexity < 1.5` (V-cycle 内部 op 数与 NumY 比;> 1.5 表示 AMG hierarchy 过深, 工程不 healthy)
     - **Axis 5 (AMG-specific)**: `operator_complexity < 2.0` (sum coarse grids size / fine grid size; > 2.0 表示 coarsening 不够 aggressive, 内存爆)
   - emit `aggregate.tsv` (16 行) + `aggregate_verdict.txt` (4 case × 5 axis KV block)

2. **`tools/p8tune.F/render_verdict.sh`** (类 P8-tune.D `render_verdict.sh`): 渲染 `docs/p8tune/amg_spike_verdict.md` (top-line verdict + per-case T-tables + raw TSV)

3. **`docs/adr/0007-amg-spike-decision.md`** (新): ADR follows ADR-0005 模板:
   - §Status: Proposed (PR-C authoring; 待 PR-D capstone Accepted flip)
   - §Context: P8-tune.D KLU 在大 case wall axis NO-GO; SPGMR Krylov 已 saturated; AMG 是 algebraically-correct retreat
   - §Decision: 4-branch decision tree
     - **GO**: heihe_x4 + heihe_x16 都 PASS (5 axis 全过) → 触发 P8-tune.G full AMG + A5 integration epic (4-6 weeks)
     - **Optional**: heihe_x4 PASS, heihe_x16 FAIL → 触发 P8-tune.G heihe_x4-only integration (~3-4 weeks)
     - **NO-GO heihe_x16-only**: heihe_x4 PASS, heihe_x16 FAIL with Memory or Operator-complexity axis 否决 → 触发 P8-tune.H GPU sparse spike for heihe_x16 + P8-tune.G heihe_x4-only
     - **NO-GO both**: heihe_x4 + heihe_x16 都 FAIL → reconsider solver architecture (domain decomposition / multigrid hybrid / GPU AMG); 升级到 ADR re-do or workshop
   - §Discussion: 与 ADR-0005 (KLU) + ADR-0004 (SPGMR maxl) 对比;case-asymmetric scaling pattern 第三 epic 验证 (B / C / D); methodology lesson learnt
   - §References + §Suppressed branches + §Forward action

4. **`.review-evidence/p8tune-amg-pr-c/{aggregate.tsv, aggregate_verdict.txt, SPEC_STATUS_HEADER.md}`** (类 P8-tune.D PR-B)

**Deliverables**:
- `tools/p8tune.F/aggregate_amg_spike.sh` (~500-700 lines bash + python)
- `tools/p8tune.F/render_verdict.sh` (~200-300 lines)
- `docs/adr/0007-amg-spike-decision.md` (~250-350 lines)
- `docs/p8tune/amg_spike_verdict.md` (~200-250 lines, rendered)
- `.review-evidence/p8tune-amg-pr-c/` (3 files: aggregate + verdict + spec header)

**Acceptance**:
- aggregator 解析 16 cell-NN.log 全部成功 (无 UNKNOWN class, 0 silent fallback)
- ADR-0007 §Decision auto-typed 与 aggregate_verdict.txt KV block 一致
- 4 verdict branch 之一明确 trigger (GO / Optional / NO-GO heihe_x16-only / NO-GO both)
- `openspec validate p8tune-amg-spike --strict` PASS

**Budget**: 3-5 days (类 P8-tune.D PR-B 实际 1 天 implementer + 0.5 天 Phase-4 review iterate)

**Dependencies**: PR-B merged

## §3.5 PR-D: epic capstone + master plan close + OpenSpec archive + 接续 trigger

**Scope**:
1. **ADR-0007 Status flip**: Proposed → Accepted (本 PR capstone)
2. **master plan §P8-tune.F**: [OPEN, anchor] → [CLOSED] + post-merge status para (类 P8-tune.D §P8-tune.D close 模式)
3. **master plan 新增 §P8-tune.G**: 若 ADR-0007 GO 或 Optional, 新加 §P8-tune.G full AMG + A5 integration epic anchor (类 §P8-tune.E.small-only)
4. **master plan 新增 §P8-tune.H** (如有需要): 若 ADR-0007 NO-GO heihe_x16-only, 新加 §P8-tune.H GPU sparse spike anchor
5. **OpenSpec archive**: `openspec archive p8tune-amg-spike -y` → `openspec/specs/amg-pattern-spike-verdict/spec.md`
6. **docs/review-loop-log.jsonl**: 追加 PR-0/A/B/C/D 5 条 review-loop entries
7. **学术 capstone summary** (可选): `docs/p8tune/p8tune_f_academic_summary.md` (类 `p8tune_d_academic_summary.md`, ~500-700 行)
8. **epic capstone-merge PR-E**: baseline/p8tune-amg-spike → main, merge-commit strategy (类 P8-tune.D PR-D #389)

**Deliverables**:
- ADR-0007 Status: Accepted
- master plan 3 sections (P8-tune.F CLOSE + P8-tune.G OPEN + P8-tune.H OPEN conditionally)
- `openspec/specs/amg-pattern-spike-verdict/spec.md` (archived)
- `docs/review-loop-log.jsonl` (5 entries)
- (optional) `docs/p8tune/p8tune_f_academic_summary.md`
- epic capstone-merge PR opened

**Acceptance**:
- master plan §P8-tune.F [CLOSED] + 接续 epic anchor 写入
- `openspec list --specs | grep amg-pattern-spike-verdict` 命中
- review-loop log 5 entries 完整
- CI all green
- merge per pre-authorization

**Budget**: 2-3 days

**Dependencies**: PR-C merged

---

# §4 ADR-0007 4-branch decision tree (详细)

## §4.1 Branch GO (heihe_x4 + heihe_x16 5 axis 全 PASS)

**Trigger**: 全 4 case (keliya / heihe / heihe_x4 / heihe_x16) 各自 best combo 在 5 axis (setup + apply + memory + cycle_complexity + operator_complexity) ALL PASS。

**Forward action**:
- 立即 trigger **P8-tune.G full AMG + A5 integration epic** (4-6 weeks)
- scope: `SUNLinSol_Hypre` wire-up to `cvode_config.cpp` (default OFF, opt-in via `SHUD_AMG_ENABLE=1`)
- A5 hydrology-equivalence gate: rivqdown.dat NSE/KGE/peak/water-balance vs PREC_NONE B1b baseline on heihe_x4 (production target) + heihe_x16 (future scale)
- ADR-0008 promotion: Performance opt-in tier → A5-certified tier (如 wall improvement ≥ 30% + A5 PASS)

**期望产出**: 生产 production-target heihe_x4 wall ↓ 30-50% + heihe_x16 from non-viable → viable;A5 PASS 给水文学发表 certificate。

## §4.2 Branch Optional (heihe_x4 PASS, heihe_x16 FAIL 但 close-margin)

**Trigger**: heihe_x4 5 axis ALL PASS, heihe_x16 失败 axis 是 Setup 或 Apply wall 但 margin ≤ 1.5×;Memory + cycle/operator complexity axis PASS。

**Forward action**:
- trigger **P8-tune.G heihe_x4-only integration** (3-4 weeks)
- scope: 同 GO branch 但仅 heihe_x4 production target 验收 A5;heihe_x16 deferred
- ADR-0008 Optional-tier (类 ADR-0004 SHUD_SPGMR_MAXL Optional-knob 模式): `SHUD_AMG_ENABLE=1` 在 heihe_x4 cn-node 上 ship, 在 heihe_x16 上 fall-through to SPGMR

**期望产出**: 生产 heihe_x4 加速;heihe_x16 仍由 P8-tune.H GPU sparse spike 评估

## §4.3 Branch NO-GO heihe_x16-only (heihe_x4 PASS, heihe_x16 Memory or Operator-complexity FAIL)

**Trigger**: heihe_x4 5 axis ALL PASS, heihe_x16 失败 axis 是 Memory (peak_rss > 0.7 × cn-RAM = 121 GiB) 或 Operator-complexity (> 2.0 表示 coarsening 不够 aggressive, 内存 explode)。

**Forward action**:
- trigger **P8-tune.G heihe_x4-only integration** (3-4 weeks, 同 Optional branch)
- 并行 trigger **P8-tune.H GPU sparse spike for heihe_x16** (~4 weeks)
- P8-tune.H scope: NVIDIA AmgX (单节点 cuSparse + cuBLAS backed AMG) 或 ROCSparse hipsparse-amg 替代;reusing dump_adjacency + fd_color_jacobian (P8-tune.D 工具链)

**期望产出**: heihe_x4 CPU AMG ship + heihe_x16 GPU AMG 评估;后续 P8-tune.I 或 P9 epic 整合

## §4.4 Branch NO-GO both (heihe_x4 + heihe_x16 都 FAIL)

**Trigger**: 即使 best combo, heihe_x4 失败任一 axis 或 heihe_x16 失败任一 axis (即 heihe_x4 不属 PASS branch)。

**Forward action**:
- **不立即 trigger 下一 epic**
- 升级到 ADR re-evaluation workshop: 重审 ADR-0002 solver-path decision tree, 考虑:
  - Domain decomposition (heihe_x4 / heihe_x16 切成 sub-domain, 每 sub-domain 单独 KLU 或 SPGMR + Schwarz coupling)
  - Multigrid hybrid: AMG (coarse) + KLU/SPGMR (fine) 二层混合
  - Substantial re-architecting: 换 CVODE 求解器 backend (e.g. SUNDIALS IDAS, ARKODE) — 但 SHUD use case 紧耦合 implicit BDF 不易换
- 可能产出新 ADR-0009 SDM (Substantial Decision Memo) 决策

**期望产出**: 给 user + GPT Pro 联合复审 architectural 路径

## §4.5 Branch BLOCKED (工具链不稳定)

**Trigger**: PR-0 #386 fix 之后, PR-B sweep 仍出现 heap corruption / 非 marker exit / cell-NN.log incomplete。

**Forward action**:
- 暂停 P8-tune.F, 回到 #386 root cause re-investigate (可能 ASan 没覆盖到的 path)
- 不进 ADR-0007 verdict

---

# §5 风险与缓解

## §5.1 R1: Hypre 安装失败 / version 不匹配 (Mac/Linux)

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| Hypre version < 2.20 (老 Ubuntu apt) 缺 BoomerAMG-extended interp | PR-A 不能跑 (combo 1/2 失败) | Medium | PR-A 任务 1.1 强制 `nm libhypre | grep HYPRE_BoomerAMGSetInterpType` 检测;若失败 build from source (libhypre-2.30.0 from LLNL git) |
| Mac brew hypre 与 Linux apt hypre 微版本差异 | Mac smoke PASS, Linux sweep 不同结果 | Low | PR-A evidence 记录两端 SHA + version;PR-B sweep 用 Linux server 是验收权威 |

## §5.2 R2: AMG setup wall 在 heihe_x16 上 OOM

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| heihe_x16 NumY=485K Hypre BoomerAMG setup 内存超 cn-RAM 173 GiB | cell-12/13/14/15 全 AMG_OOM_DETECTED marker | Medium | spec REQ-5 把 AMG_OOM 作 exit-0 数据点 (类 KLU_OOM);ADR-0007 NO-GO heihe_x16-only branch 接此 outcome |
| Coarsening 不够 aggressive 导致 operator_complexity > 5 | RSS 爆 + apply wall 爆 | Medium | Combo 1 (extended+i, HMIS) 设计为 aggressive baseline;若 Combo 0 OOM, Combo 1 应可 PASS |

## §5.3 R3: BoomerAMG 收敛性差 (residual_reduction < 5× per V-cycle)

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| SHUD Jacobian 是 non-symmetric (river-channel + lake-bank 路径), AMG 假设 symmetric/M-matrix 在 SHUD 上失效 | residual_reduction_v1 < 2×, AMG_SOLVE_DIVERGE marker | Low-Medium | P8-tune.D 的 fd_color Jacobian 已 verified well-formed;若 SHUD 矩阵真的 non-symmetric AMG-hostile, 走 NO-GO branch + 评估 GMRES + ILU(K) preconditioner |
| AMG 在 unsteady RHS 下 Jacobian 频繁更新 — 不利于 amortize setup wall | setup_wall 占总 wall 80%+, AMG 不再优于 KLU | Medium | aggregator 的 Setup axis 阈值 1.5 × 0.7 × SPGMR per-step 已考虑此 — 若 setup overflow, NO-GO branch |

## §5.4 R4: PR-0 #386 fix 失败 / 部分修复

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| uninit ptr 根因复杂 (涉及 ownership 链), 0.5-2 days 估时不足 | PR-0 阻塞 P8-tune.F 启动 | Medium | 时延 1-2 weeks 内 acceptable;若 > 2 weeks, escalate user 决定是否仍走 `_exit(0)` workaround 但只在 P8-tune.F PR-A spike 用 (production-not-touched)；P8-tune.E.small-only 必须等真 fix |
| Fix 只在 keliya 通过, heihe / heihe_x4 仍 corrupt | Production code 仍未稳定, P8-tune.G integration 又走回 _exit(0) workaround | Medium | PR-0 acceptance 强制 `valgrind heihe` PASS (keliya + heihe 双 case);若 NumY > 100k 仍 corrupt, ESH (Extended Scope Hold) issue 重新 reopen + 升级到 P8-tune.G prereq (P8-tune.F PR-A spike 仍可走 `_exit` workaround因 spike scope 只用 init) |

## §5.5 R5: Slurm 三铁律违反 (continuation P8-tune.D 经验)

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| 误用 login-node submit, 或 `--output` 路径在 `/users/$USER` 而非 `/scratch` | sacct 显示 ExitCode 127, 数据丢失 | Low (P8-tune.D 教训记) | PR-B `precheck_env.sh` 强制 4 条件 + 加 5 条件 "sbatch from /scratch path"; scancel 后 sbatch on cn-node retry |

## §5.6 R6: cn-node 资源竞争 (多 epic 同时跑)

| Risk | 影响 | 概率 | 缓解 |
|---|---|---|---|
| P8-tune.F PR-B 与 P8-tune.E.small-only PR-A 同时占 cn-node | Slurm queue 等 + cell 跑慢 | Medium | P8-tune.F primary, P8-tune.E.small-only 不应抢占 cn-node;P8-tune.E.small-only 启动 condition = P8-tune.F PR-B sweep 完成 (cn-node 空出) |

---

# §6 预算与时间线

## §6.1 5-PR budget 估算

| PR | 估时 (calendar days) | 估时 (engineering hours) | 关键路径 |
|---|---:|---:|---|
| PR-0 | 0.5-2 | 4-16 | uninit ptr root cause 复杂度 |
| PR-A | 3-5 | 24-40 | Hypre API + ColPack/CSR 转换 + Mac smoke 4 combo |
| PR-B | 5-7 | 8 (orchestrator) + 40-56 (server wall) | 16-cell Slurm sweep, 部分 cell 8h wall |
| PR-C | 3-5 | 24-40 | aggregator + ADR-0007 起草 |
| PR-D | 2-3 | 16-24 | capstone docs + OpenSpec archive + epic merge PR-E |
| **总计** | **13.5-22 cal days (~3-4.4 weeks)** | **116-176 engineering hours** | ~4 epic-week 预算 |

实际 calendar duration 取决于 (i) PR-B Slurm cell 是否需要重跑 (P8-tune.D 经验 ~7h wall + ~16h 重跑 buffer); (ii) Phase-4 review-loop iterate 次数 (P8-tune.D 经验 PR-A 2 rounds + PR-B 1 round + PR-C 1 round)。

## §6.2 关键里程碑

| Milestone | Date estimate | Trigger |
|---|---|---|
| #386 fix landed | T + 0.5-2 days | PR-0 merge to baseline/p8tune-amg-spike |
| Mac smoke 4 combo PASS | T + 4-7 days | PR-A merge |
| Server 16-cell sweep complete | T + 9-14 days | PR-B merge |
| ADR-0007 Accepted | T + 12-19 days | PR-C merge |
| epic CLOSED + capstone-merge | T + 14-22 days | PR-D + PR-E merge |
| (conditional) P8-tune.G epic START | T + 15-25 days | ADR-0007 GO or Optional branch |

T = #386 fix 启动日 (即 P8-tune.F PR-0 开 day 1)。

---

# §7 接续 epic triggers

按 ADR-0007 4-branch 自动 trigger:

```
P8-tune.F epic close (PR-D + PR-E capstone-merge)
       │
       ├── if ADR-0007 = GO
       │     → P8-tune.G full AMG + A5 integration epic (4-6w)
       │       (heihe_x4 + heihe_x16 production support; A5 NSE/KGE PASS;
       │        ADR-0008 promotion to A5-certified-tier)
       │
       ├── if ADR-0007 = Optional
       │     → P8-tune.G heihe_x4-only integration (3-4w)
       │       (heihe_x4 production opt-in;heihe_x16 仍 SPGMR fall-through)
       │
       ├── if ADR-0007 = NO-GO heihe_x16-only
       │     → P8-tune.G heihe_x4-only integration (3-4w) (并行)
       │       + P8-tune.H GPU sparse spike (heihe_x16, ~4w) (并行)
       │
       └── if ADR-0007 = NO-GO both
             → ADR re-evaluation workshop;考虑 domain decomp / hybrid solver
               / 新 SDM (ADR-0009);P8-tune 阶段可能 close 进 P9
```

P8-tune.E.small-only (KLU mini-prototype) 不在本路径上 — 仅 small case keliya/heihe 优化, 与本 epic 大 case 主目标无关。可与 P8-tune.F 并行启动 (共享 #386 prereq)。

---

# §8 与 P8-tune.D 对比 (复用 + 不同)

## §8.1 复用 P8-tune.D 的工程方法学

| 项 | P8-tune.D | P8-tune.F (本 epic) |
|---|---|---|
| Spike 模式 | Pattern-only + zero-source-patch + zero-CVODE-wireup | 同 |
| libshud.a carve-out | Yes (P8-tune.D PR-0 introduce) | 复用 |
| FD-color Jacobian + ColPack DISTANCE_TWO | `tools/p8tune.D/fd_color_jacobian.cpp` | 复用 (Makefile symlink) |
| 5-block CSC adjacency dump | `tools/p8tune.D/dump_adjacency.cpp` | 复用 |
| Slurm array sweep + SIGTERM trap | `tools/p8tune.D/spike_array.sbatch` | 模板复用 |
| Aggregator + Per-case best-combo + N-axis verdict | `tools/p8tune.D/aggregate_klu_spike.sh` | 模板复用 (axis 数从 3 增至 5) |
| ADR 4-branch decision tree | ADR-0005 (Case-aware) | ADR-0007 (类似 4-branch + GO 子分支) |
| review-loop log + Phase 4-7 cross-review | `docs/review-loop-log.jsonl` | 继续 append |
| Slurm 三铁律 严守 | PR-A in-session 教训 | precheck_env.sh + sbatch dry-run |
| 90-day case 截断 | `cfg.para END=91` | 同 |
| CMFD V0200 forcing | 同 | 同 |
| SHUD submodule push to `openmp-baseline` | 同 | 同 |

## §8.2 P8-tune.F 不同点

| 项 | P8-tune.D | P8-tune.F | 原因 |
|---|---|---|---|
| 求解器 | SuiteSparse KLU (32-bit Int API) | Hypre BoomerAMG (16-byte float + 8-byte int) | AMG 是 iterative, 无 fill-overflow 风险 |
| Verdict axes | 3 (fill + RSS + wall) | 5 (setup + apply + memory + cycle_complexity + operator_complexity) | AMG 有独立 setup 阶段 + iteration complexity 维度 |
| 0.7× 系数 wall budget | 单一 `WALL_BUDGET_S = 0.7 × heihe_x4 SPGMR per-step` | Apply 同 0.7×, Setup 用 1.5 × 0.7 × (因生产 amortize over N_solve refactor) | AMG setup 是 amortized cost |
| Per-cell wall budget | 4h (Slurm) | 8h (Slurm) | AMG setup 在 heihe_x16 可能 ≥ 1h, 需更长 wall |
| Marker classes | KLU_OOM / KLU_INDEX_OVERFLOW / KLU_WALL_OVERFLOW (3) | AMG_OOM / AMG_SETUP_DIVERGE / AMG_SOLVE_DIVERGE / AMG_WALL_OVERFLOW (4) | AMG 有独立 setup/solve divergence 失败模式 |
| Ordering decision | AMD lock (hardcoded in tool config) | (interp_type, coarsen_type) combo sweep | AMG 有更大 hyperparameter 解空间 |
| Test cases | 4 (keliya + heihe + heihe_x4 + heihe_x16) | 同 | 沿用 P8-tune.D matrix 保 across-epic comparability |
| Sweep matrix size | 4 × 4 = 16 cell | 4 × 4 = 16 cell | 同 |

---

# §9 Open questions (待 user decision)

## §9.1 Q1: P8-tune.F PR-0 是否单独从 P8-tune.D 工具链拆出?

**Option A**: P8-tune.F PR-0 仅修 #386, 保留 spike 工具在 `tools/p8tune.D/`, P8-tune.F PR-A 在 `tools/p8tune.F/` 复用 P8-tune.D 工具(Makefile 引)
- Pro: 不打扰 P8-tune.D 历史 baseline
- Con: 工具 ownership 分散 (`tools/p8tune.D/` 是 KLU spike 历史, 复用 in P8-tune.F 有点 awkward)

**Option B**: P8-tune.F PR-0 把 P8-tune.D 的 dump_adjacency + fd_color_jacobian 移到 `tools/common/spike/` 共享目录, P8-tune.D + P8-tune.F 各自的 specialized 工具留在自己目录
- Pro: 工具 ownership 清晰
- Con: 改 P8-tune.D 历史 baseline 不雅;merge 时多一层 commit

**推荐**: Option A (default), 等 P8-tune.G integration 才考虑共享目录 refactor。

## §9.2 Q2: A5 hydrology-equivalence 是否仍是 P8-tune.G 单独 gate?

GPT Pro F4 retrospective 已指出 A5 不应作 pattern-only candidate direct gate, 仅作 integrated solver candidate 验收 gate。P8-tune.F PR-D ADR-0007 GO branch 应明确 P8-tune.G 包含 A5 gate 还是只做 wall improvement + counter delta + ADR-0008 promotion 推到 future epic?

**推荐**: P8-tune.G 包含 A5 gate (A5 是 hydrology-equivalence 验收必经, 任何 production 路径 ship 前都应 A5 PASS;promotion to A5-certified-tier 是 ship 必要条件)。但 A5 阈值可放宽 (initial: NSE ≥ 0.9 而非 0.95, 允许 P9-A5 epic 后续 tighten)。

## §9.3 Q3: P8-tune.H GPU sparse spike 是否必须本 epic 启动?

ADR-0007 NO-GO heihe_x16-only branch 触发 P8-tune.H GPU sparse spike, 但 cn-node 是否有 GPU (CLAUDE.md 提到 `GPU` 分区 `gn01`) 需 user confirm。若 cn-node GPU 资源不足, P8-tune.H 可延后或换 server。

**推荐**: PR-D capstone 时 trigger 只是 anchor, 不立即启动;由 user 在 ADR-0007 NO-GO heihe_x16-only branch 出现后 separately trigger。

---

# §10 References

## 内部 docs

- [docs/adr/0005-klu-spike-decision.md §Forward action + §GPT Pro retrospective F4](docs/adr/0005-klu-spike-decision.md) — 本 epic trigger source-of-truth
- [docs/p8tune/klu_spike_verdict.md](docs/p8tune/klu_spike_verdict.md) — P8-tune.D verdict (KLU 在大 case wall axis NO-GO 数据)
- [docs/p8tune/p8tune_d_academic_summary.md §10.5](docs/p8tune/p8tune_d_academic_summary.md) — forward path 修订 (P8-tune.F primary)
- [SHUD_openMP_master_plan.md §P8-tune.F (5-PR scope 锚)](SHUD_openMP_master_plan.md)
- [docs/adr/0002-solver-path.md](docs/adr/0002-solver-path.md) — Path 5 BoomerAMG/Hypre 决策起点
- [docs/adr/0004-maxl-sweep-decision.md](docs/adr/0004-maxl-sweep-decision.md) — case-asymmetric 先例 (SPGMR maxl Optional-knob)
- [docs/adr/0003-precond-spike-decision.md](docs/adr/0003-precond-spike-decision.md) — ADR-0003 PREC_NONE NO-GO (P8-tune.A/B closure)

## 代码与 evidence

- [tools/p8tune.D/dump_adjacency.cpp](tools/p8tune.D/dump_adjacency.cpp) — P8-tune.F PR-A 复用
- [tools/p8tune.D/fd_color_jacobian.cpp](tools/p8tune.D/fd_color_jacobian.cpp) — P8-tune.F PR-A 复用
- [tools/p8tune.D/spike_array.sbatch](tools/p8tune.D/spike_array.sbatch) — P8-tune.F PR-B 模板复用
- [tools/p8tune.D/aggregate_klu_spike.sh](tools/p8tune.D/aggregate_klu_spike.sh) — P8-tune.F PR-C 模板复用 (axis 数 3 → 5)
- [tools/p8tune.D/render_verdict.sh](tools/p8tune.D/render_verdict.sh) — P8-tune.F PR-C 模板复用
- [tools/p8tune.D/precheck_env.sh](tools/p8tune.D/precheck_env.sh) — P8-tune.F PR-B 复用 + 加 1 条 (sbatch from /scratch path)

## 外部依赖

- **Hypre 2.30.0** (Lawrence Livermore National Laboratory, `https://github.com/hypre-space/hypre`)
- **SuiteSparse 7.12.2** (Tim Davis, P8-tune.D 已 pinned, P8-tune.F 不引入新版)
- **ColPack 1.0.10** (Argonne, P8-tune.D 已 pinned)
- **SUNDIALS-CVODE 6.0.0** (LLNL, pinned P1e era)

## 学术参考

- [Yang 2002] U. M. Yang, "Parallel Algebraic Multigrid Methods — High Performance Preconditioners," *Numerical Solution of Partial Differential Equations on Parallel Computers*, Lecture Notes in Computational Science and Engineering 51, Springer, 2006. (BoomerAMG 设计基础)
- [Henson & Yang 2002] V. E. Henson, U. M. Yang, "BoomerAMG: A parallel algebraic multigrid solver and preconditioner," *Applied Numerical Mathematics*, Vol. 41, pp. 155-177, 2002.
- [Falgout et al. 2006] R. D. Falgout, J. E. Jones, U. M. Yang, "The Design and Implementation of hypre, a Library of Parallel High Performance Preconditioners," *Numerical Solution of Partial Differential Equations on Parallel Computers*, Springer, 2006.
- [De Sterck et al. 2008] H. De Sterck, R. D. Falgout, J. W. Nolting, U. M. Yang, "Distance-two interpolation for parallel algebraic multigrid," *Numerical Linear Algebra with Applications*, Vol. 15, pp. 115-139, 2008. (extended interpolation type 14)
- [Briggs et al. 2000] W. L. Briggs, V. E. Henson, S. F. McCormick, *A Multigrid Tutorial, 2nd Ed.*, SIAM, 2000.

---

**Execution Summary (本 work plan 文档生成)**: agents=0 (orchestrator-direct write); skills=纯文档写作 + 参照 P8-tune.D ADR-0005 + master plan + 学术 §10.5 forward path; tools=Read/Write; verification=与 docs/adr/0005-klu-spike-decision.md + SHUD_openMP_master_plan.md §P8-tune.F + GPT Pro F4 retrospective 三方交叉核; limits=本文档作 P8-tune.F epic actionable work plan, 5-PR scope + 4-branch decision tree + budget 是 PR-0 启动前的 ANCHOR, 实际 PR scope 启动时可微调 (e.g. Hypre interp/coarsen combo 编号可能在 PR-A Mac smoke 后调整)。
