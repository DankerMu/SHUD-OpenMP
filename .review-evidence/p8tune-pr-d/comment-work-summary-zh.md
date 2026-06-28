## PR-D 工作总结（中文）

### 实际工作

新增 2 个 sweep 驱动脚本 + 在服务器执行 60-cell SPGMR maxl 扫描：

1. **`tools/p8tune/run_maxl_cell.sh`** (222 行) — 单 cell 驱动：
   - 4 参数白名单（case / N / maxl / rep）+ per-cell 隔离工作目录（`${CELL_DIR}/work/` symlink-forest 避免 `Basins/<case>/output/` 11-way 并发竞态）
   - 环境变量：`SHUD_SPGMR_MAXL=<k>` (PR-C hook) + `SHUD_RHS_THREADS=<N>` (canonical Mode C thread knob per `shud.cpp:152-167`) + `OMP_NUM_THREADS=<N>` + `OMP_PROC_BIND=spread` + `OMP_PLACES=cores`
   - `/usr/bin/time -f "%e"` wall 捕获（解耦 SHUD profiler，给 PR-E G5 独立 wall metric）
   - 5 mandatory artifacts: profile_B0.yaml + cvode_stats.txt + rivqdown.dat + wall.sec + cell.meta
   - 退出前 grep `[CVODE] SPGMR maxl=${MAXL} pretype=PREC_NONE` 验证 PR-C contract

2. **`tools/p8tune/submit_maxl_sweep_template.sbatch`** (83 行) — Slurm array 模板：
   - `--array=1-60%11`（11-way 并发匹配 CPU 分区 idle 节点数）
   - `--partition=CPU` + `--cpus-per-task=8` + `--time=03:00:00`
   - `--output/--error` 落在 `/scratch/.../maxl_sweep/_slurm/`（Slurm 三铁律 ②）
   - 决定性 decoder: `case_idx = idx/30` / `N_idx = (idx%30)/15` / `maxl_idx = (idx%15)/3` / `rep_idx = idx%3`（生成 60 unique cells）

### 执行结果（Slurm job 9690）

- **状态**: 60/60 COMPLETED on cn[06,09,11,14-19,21-22,24]
- **总 wall**: ~80 min（11-way concurrency cap）
- **artifact 完整性**: 60/60 cells 产出 5 必需文件 + provenance log line（`prov_log_count=1`）
- **summary.tsv**: 61 行 × 22 列（header + 60 cells × case/N/maxl/rep + wall_s + 15 canonical CVODE keys + rivqdown_sha12 + prov_log_count）

### 跨 maxl saturation 关键发现（preview，PR-E adjudicate）

**rivqdown.dat SHA12 跨 maxl 分组**（每 (case, N) tuple）：

| (case, N) | maxl=5 | maxl ∈ {10, 15, 20, 30} |
|---|---|---|
| heihe N=1 | `a2023ccd2de4` | `e4e9721cf667` (4 个完全相同) |
| heihe N=8 | `a2023ccd2de4` | `e4e9721cf667` |
| heihe_x4 N=1 | `b5e4b0a2cf83` | `d3d03476b6b7` (4 个完全相同) |
| heihe_x4 N=8 | `b5e4b0a2cf83` | `d3d03476b6b7` |

**3 个独立信号**：

1. **Saturation at maxl=10**: 4 个 maxl 值 (10/15/20/30) 产出 bit-identical rivqdown.dat → 增大 Krylov subspace 越过 10 之后无收益
2. **maxl=5 vs maxl=10 hydrology divergence**: SHA12 不同 → maxl=5 因 ncfl 触发的 step-size adjustment 使解轨迹漂移
3. **Solver failure 消除**:
   - heihe N=1: ncfl `85 → 0`, ncfn `7 → 5`（轻微）
   - heihe_x4 N=1: ncfl `3620 → 0`, ncfn `51 → 0`（戏剧性）

**Wall 双向**：
- heihe N=1: maxl=10 wall +1%（essentially free 改善）
- heihe_x4 N=1: maxl=10 wall −0.6%（甚至更快）
- **heihe_x4 N=8: maxl=10 wall +19%** ⚠ counter 改善但 wall 变慢 → PR-E G5 attention

### 不变量保留

- **PR-A baseline**: maxl=5 cells 在 ncfn/ncfl/nfe/nfeLS/nst/lenrw/lenrwLS 上 bit-identical PR-A 18-cell floor（heihe ncfn=7 ncfl=85，heihe_x4 ncfn=51 ncfl=3620 验证通过）→ env-unset = SUNDIALS default = maxl=5 codepath 不变
- **OMP-neutrality (B1a S4)**: 所有 10 个 (case, maxl) 组合下 N=1 vs N=8 rivqdown SHA12 IDENTICAL → 无 OMP 退化
- **PREC_NONE preserved (PR-C contract)**: 60 cells 全部 stdout 含 `pretype=PREC_NONE` provenance line
- **SHUD pointer 无漂移**: 仍 `6ce17d6` (PR-C merge state)

### Slurm 三铁律遵守

| 铁律 | 证据 |
|---|---|
| ① sbatch 从 /scratch | submit from `/scratch/.../maxl_sweep/` |
| ② --output/--error 在 /scratch | `--output=/scratch/.../maxl_sweep/_slurm/cell_%A_%a.out` |
| ③ 引用脚本在 /scratch | `bash /scratch/.../tools/p8tune/run_maxl_cell.sh` |

### review/fix 闭合

- **Phase 0.5 fixture review**: 1× reviewer → PASS (spec/design/tasks/IM D15 一致)
- **Phase 4 cross-review (round 1)**: 3 × reviewer 并行（correctness + spec-compliance + integration）→ 3/3 CLEAN APPROVE, **0 actionable findings** (28/28 per-check PASS), 14 informational/positive notes
- **Phase 4.5 verifier gate**: 0 候选 → 空 verdict 表 (无 verifier 子任务必要)
- **Phase 5/6/6.2/6.5**: SKIP (round 1 CLEAN)
- **Phase 7 Gap Sweep**: 1 × independent-final 子任务 (clean-context 未读前 reports) → **CLEAN APPROVE** (10/10 PASS)
- **CI**: 5/5 PASS (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)
- **Pre-merge hard-gate 7-check**: ✅ Agent Review block + ✅ Phase 4.5 verdict 持久化 + ✅ clean panel + ✅ Phase 7 CLEAN + ✅ CI PASS + ✅ self-audit + ✅ oracle integrity

### 风险与已知限制

- **PR-D 不 adjudicate PR-E outcome**: cross-maxl saturation finding 仅 preview，G1-G8 verdict + ADR-0004 GO/Optional-knob/Diagnostic/NO-GO 决策完全留给 PR-E（PR body 与 reviewers 一致确认 no premature adjudication）
- **heihe_x4 N=8 wall asymmetry**: maxl=10 比 maxl=5 慢 19%，但 counter 改善 → PR-E G5 wall gate 需 explicit treat（可能 trigger Optional-knob 而非 default-bump=10）
- **3 reps 样本量小**: per-cell wall 噪声 ±5-10s 在 heihe (~150s) 处约 ±3-7%，在 heihe_x4 (~1500s) 处 <1%；PR-E 聚合应 report median 而非 mean，并标 N=3 显著性局限
- **`compare_snapshot` binary 未运行**: PR-D 用 sha256sum 作为 byte-equality 谓词（功能等价），PR-E 聚合阶段会用 `tools/compare_snapshot/compare_snapshot` 跑 max_ulp 数值
- **server-resident artifacts 不入 git**: 60 cell × 5 artifacts (300 files) 留在 `/scratch/.../maxl_sweep/`，PR-E 仍可从同位置消费；Mac 仅保留 summary.tsv + 3 个 sample cells tar (`.review-evidence/p8tune-pr-d/`，gitignored)

### 下一步

PR-D merge → PR-E #368 启动：
- 消费 `/scratch/.../maxl_sweep/_summary.tsv` + 60 cell.meta + per-cell cvode_stats.txt + rivqdown.dat
- 跑 G1-G8 verdict pipeline (G1 build × G2 启动 × G3 bit-identical × G4 counter floor × G5 wall × G6 speedup × G7 hydrology × G8 provenance)
- 写 `docs/p8tune/maxl_sweep_verdict.md` + `docs/adr/0004-maxl-sweep-decision.md`
- Outcome 4-branch: GO+default-bump=10 / Optional-knob (per case asymmetry) / Diagnostic only / NO-GO
