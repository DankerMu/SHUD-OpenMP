# P1d — perf baseline

P1d epic capstone perf source-of-truth。涵盖 server PR-H 8-cell（heihe + heihe_x4 × N∈{1,2,4,8}）+ Mac PR-I 6-cell（keliya + qhh × {serial, omp@N=1, omp@N=8}）实测数据，配以 Amdahl 反推 + 早期 profile 上界对照 + ROI 分析。本 doc 是 PR-L `P1d-tag` annotated message + PR-M PROMOTE archive 的 perf 引用。

## §1 Hardware contexts

| 项 | Server (go/no-go 权威) | Mac (dev-only reference, per CLAUDE.md §1.1.1) |
|---|---|---|
| Endpoint | `frd_muziyao@210.77.77.22:32099` | Apple M4 Pro local |
| OS / Kernel | Linux Ubuntu 24.04 (cn03/cn07) | Darwin 24.6.0 |
| CPU | dual-socket Intel/AMD CPU 分区 (24 cpus per node × 2 socket = 48 logical) | Apple M4 Pro 14-core (4P + 10E) |
| NUMA | 2 nodes (per `numactl --hardware`, node 0 size 95338 MB / node 1 size 96714 MB, distances 0→1=21 vs 0→0=10) | single-socket UMA, NUMA N/A |
| Compiler | GCC 13.3.0-6ubuntu2~24.04.1 | Apple Clang 17.0.0 (clang-1700.6.3.2) |
| Project root | `/scratch/frd_muziyao/SHUD-OpenMP` | `/Users/danker/Desktop/Hydro-SHUD/openMP` |
| Forcing | CMFD V0200 1951-01 → 2024-12 (`/volume/data/ForcingData/CMFD2.0/`) | 同 server (rsync 镜像) |
| 截断纪律 | 90-day cap per CLAUDE.md "所有 case ≤90 天截断" 项目铁律 | 同 |

## §2 Build matrix

P1d 期间 build matrix 仅 2 种（M10 修订前后未变）：

| Target | 命令 | 说明 |
|---|---|---|
| `shud` (serial) | `cd SHUD && make shud` | Serial path；`N_VNew_Serial` + Serial RHS |
| `shud_omp` | `cd SHUD && make shud_omp` | per fact-check #1+#2+#3：实际 = **Serial 水文 RHS + OpenMP N_Vector backend**（StrictOMP/ProductionOMP RHS path 为 `std::abort()` 桩，从未执行；`-DSHUD_USE_OPENMP_NVECTOR=1` + `-lsundials_nvecopenmp` 决定 N_Vector 用 OpenMP 后端） |

FP strict gate（master plan §8.1）：

| Flag | shud | shud_omp |
|---|---|---|
| `-O2` | ≥1 | ≥1 |
| `-ffp-contract=off` | ≥1 | ≥1 |
| `-fno-fast-math` | ≥1 | ≥1 |
| `-fopenmp` (linux) / `-Xpreprocessor -fopenmp` (mac) | — | ≥1 |
| `-ffast-math` / `-Ofast` | 0 | 0 |

PR-G post-revert + PR-K verify：3-grep ≥1 + 0-grep =0 全通过。

## §3 Server PR-H wall + speedup matrix（90-day, post-Kahan-revert + first-touch + no `--interleave=all`, per `docs/p1d/p1d_pr_h_final_run.md`）

| Case | N=1 wall (s) | N=2 | N=4 | N=8 | Speedup S2 | S4 | **S8** |
|---|---|---|---|---|---|---|---|
| heihe (NumEle=6335) | 513 | 513 | 503 | **456** | 1.00× | 1.02× | **1.13×** |
| heihe_x4 (NumEle~25000) | 1187 | 1169 | 1096 | **937** | 1.02× | 1.08× | **1.27×** |

并行效率 `S8 / 8`：heihe 14.1%、heihe_x4 15.9%。

cf. PR-F v2 (Kahan IN + `--interleave=all` per PR-B runbook 原 prescription) heihe_x4 N=8 wall = 1037s vs PR-H 937s → PR-F 实测发现 `--interleave=all` 是 anti-pattern with active first-touch（详 PR-F doc §"Wall interpretation"），PR-H 已 drop `--interleave=all` 回收 10% wall。

## §4 Amdahl 反推（recalculated, supersedes PR-H 初版 "f~72%" 估算）

由公式 `f = (1/S - 1/N)/(1 - 1/N)`，代入 N=8 + 实测 speedup：

| Case | S8 | `1/S` | `1/N` | 分子 `1/S - 1/N` | 分母 `1 - 1/N` | **`f`** |
|---|---|---|---|---|---|---|
| heihe | 1.13× | 0.8850 | 0.125 | 0.7600 | 0.875 | **0.869 (86.9%)** |
| heihe_x4 | 1.27× | 0.7874 | 0.125 | 0.6624 | 0.875 | **0.757 (75.7%)** |

注释：`f` 不能全部归因 CVODE 内部 — 含三项加和：

1. **serial RHS**（fact-check #1+#2：当前 `shud_omp` 走 `ExecPolicy::Serial` 路径，hydrology RHS 完全单线程）
2. **N_Vector fork/join 开销**（fact-check #3：`shud_omp` 硬编码 NVECTOR_OPENMP，每次 CVODE 调 dot product / WRMS norm 都 fork-join，N=8 时这部分本身有 ~5-10% overhead）
3. **memory bandwidth 饱和**（heihe_x4 大数据集尤甚）

## §5 Theoretical upper bound（B1a profile 早期数据对照）

`benchmarks/heihe_x4/profile_B0.target.yaml` (Project profile_decision sign-off SHUD@`78c37a1` era) 实测 heihe_x4 RHS 占 wall **66.55%**。理想 Amdahl 8 核上界：

```
S_max = 1 / (1 - 0.6655 + 0.6655 / 8)
      = 1 / (0.3345 + 0.0832)
      = 1 / 0.4177
      = 2.39×
```

| 当前 (PR-H) | 理想 Amdahl | 差距 |
|---|---|---|
| heihe_x4 S8 = 1.27× | 2.39× | -1.12× (-47%) |

**关键洞察**：当前 1.27× 的瓶颈 **不是 "Amdahl 已极限"**（理想上界还有 2.39×），是 **真正应并行的 RHS 还没并行**（per fact-check #2 StrictOMP path 是 `std::abort()` 桩）。差距 ~1.12× 大致对应 "真正应该跑 OMP 的 RHS 仍是 serial" 这件事被遮盖：profile 算 66.55% RHS，假设 RHS 100% 并行，但实际 RHS 0% 并行（仅 NVector ops 跑 OMP fork-join）。

故 P1e (F 路) 的目标是 **真正去吃这个 1.12× 差距**，把 S8 推到 ≥1.5× 作为最小可接受门，T 目标 2.39× (heihe_x4 ideal)。

## §6 rivqdown.dat 散度矩阵（PR-H 实测 vs P1c era 对照）

per `docs/p1d/p1d_pr_h_final_run.md` §"Post-verdict 修订" §"实测 rivqdown.dat 数值散度"：

### §6.1 PR-H (Kahan OUT + first-touch + NUMA env)

| Case / N | mean_rel | RMSE | max_rel | n_diff / n_total |
|---|---|---|---|---|
| heihe N=2 vs N=1 | 0 | 0 | 0 | 0 / 214252 |
| heihe N=4 vs N=1 | **3.8%** | 1.79e+03 | 1010× | 210774 / 214252 |
| heihe N=8 vs N=1 | **10.0%** | 2.76e+03 | 12534× | 210779 / 214252 |
| heihe_x4 N=2 vs N=1 | 0 | 0 | 0 | 0 / 387607 |
| heihe_x4 N=4 vs N=1 | **17.6%** | 8.63e+03 | 2214× | 383130 / 387607 |
| heihe_x4 N=8 vs N=1 | **25.1%** | 7.59e+03 | 2908× | 382116 / 387607 |

### §6.2 P1c era (Kahan IN, no NUMA env, no first-touch)

| Case / N | mean_rel | max_rel |
|---|---|---|
| heihe N=8 vs N=1 | 10.0% | 1446× |
| heihe_x4 N=8 vs N=1 | 20.3% | 66120× |

**结论**：Kahan 注入只压住 nst step count（P1c heihe |Δ_nst|=84 vs PR-H 152；改善 ~45%），**没修水文输出散度**。两套数据相同 N=8 mean_rel 同量级（10% vs 10%、25% vs 20%）→ Kahan 是 orthogonal axis。`max_rel` 1000-66000× 部分被极小 denominator 放大，待 P1e audit `rivqdown.dat` 输出代码是否引入物理 floor（per Pro2 警示 §4.9）；`mean_rel` 10-25% 是工程层面的硬实锤。

## §7 Mac PR-I 6-cell reference matrix（`P1-update-omp-tag` anchor era, per `docs/p1d/p1d_pr_i_p1_update_omp_reference.md`）

worktree at `.p1d-pr-i-worktree/P1-update-omp-anchor`（outer `003f58d` / SHUD `07c677f`），独立 build：

### §7.1 keliya (NumEle=484, 90-day START=12053 END=12143)

| Mode | SHA256 (first 16) | nst | wall (s) |
|---|---|---|---|
| serial | `89686fb8c97a3852` | 101188 | 27.23 |
| omp@N=1 | `b23e15b94c0f67be` | 111272 | 65.24 |
| omp@N=8 | `c7a052803a5bae1a` | 97217 | 183.77 |

### §7.2 qhh (NumEle=4773 +lake, 90-day START=8401 END=8491)

| Mode | SHA256 (first 16) | nst | wall (s) |
|---|---|---|---|
| serial | `d9a42798eb649dce` | 13000 | 83.82 |
| omp@N=1 | `38ef0414d9ffa931` | 13000 | 85.11 |
| omp@N=8 | `e184a6c5e0807fb9` | 13031 | 127.55 |

### §7.3 PR-G Mac 9-SHA matrix cross-reference（confirms PR-G clean revert）

per `docs/p1d/p1d_kahan_revert.md` §"SHA matrix"：

| Config | post-PR-G SHA (head 16) | == pre-K2 `de9545d`? | == P1-update-omp-tag (PR-I anchor)? |
|---|---|---|---|
| `./shud keliya` serial | `89686fb8c97a3852` | ✓ byte-identical | ✓ byte-identical (PR-I keliya serial) |
| `shud_omp` @ N=1 OMP_PROC_BIND unset | `b23e15b94c0f67be` | ✓ byte-identical | ✓ byte-identical (PR-I keliya omp@N=1) |
| `shud_omp` @ N=1 `OMP_PROC_BIND=close OMP_PLACES=cores` | `b23e15b94c0f67be` | ✓ byte-identical | ✓ byte-identical |

**意义**：PR-G Kahan revert 把 Mac keliya 全 3 config 回到 pre-K2 (de9545d) byte-identical，且 pre-K2 = P1-update-omp-tag era → 证明 PR-G revert 干净 + first-touch loops 在 N=1 是 bitwise-neutral。

### §7.4 Mac wall 解读（informational only per CLAUDE.md §1.1.1）

- **keliya** (NumEle=484): N=8 wall (183.77s) > serial (27.23s) by 6.7× — small case dominated by OMP overhead on Apple Silicon
- **qhh** (NumEle=4773 +lake): N=8 wall (127.55s) > serial (83.82s) by 1.5× — still oversubscribed
- Mac libomp 弱绑定 + UMA + 14-core asymmetric (4P + 10E) → Mac wall **不计入 §1.1.1 go/no-go**

Mac SHA / nst 字段保留作 future P1e Mac 比对的 anchor reference（per `docs/p1d/p1d_pr_i_p1_update_omp_reference.md`）。

## §8 CPU-hour cost ROI analysis

8-core 跑 `shud_omp` vs 1-core `shud` 的 CPU-hour cost 比：

| Case | serial (1×513s) CPU-hr | N=8 (8×456s) CPU-hr | CPU cost ratio | Speedup | **ROI** |
|---|---|---|---|---|---|
| heihe | 0.142 | 1.013 | **7.13×** | 1.13× | -6.31× （强负） |
| heihe_x4 | 0.330 | 2.082 | **6.31×** | 1.27× | -4.97× （强负） |

含义：8 核跑 `shud_omp` 比 serial 多花 6.3-7.1× CPU 时间（hyperthreading + N_Vector fork-join 开销 + serial RHS 不并行白吃 cores），换来 1.13-1.27× wall 加速 + 10-25% mean_rel 流量误差。

production 决策（E′ 第 1 项）：**默认 `cfg.para NUM_OPENMP=1` (serial)**。理由：
1. wall 仅比 N=8 慢 11-12%
2. 完全 reproducible（byte-identical run-to-run + cross-N N/A）
3. CPU 资源换 6.3-7.1× 节约
4. 0% 流量误差（vs N=8 10-25% mean_rel）

## §9 工程不确定性带宽对照

PR-H 10-25% mean_rel 落在哪个量级？

| 参照 | 量级 | 与 PR-H mean_rel 10-25% 关系 |
|---|---|---|
| CMFD V0200 forcing 不确定性（per CMFD 文档 + 项目 §1.1.1 调研） | 10-20% | **同量级**（即 OMP 散度等同 forcing 数据本身的不确定性 — 不可接受 claim "比 forcing 精度高") |
| 流量计 gauge 测量精度 | 5-10% | **超过测量精度**（OMP 散度 > 校准数据精度 — 不能作为 calibration 基准） |
| Manning's n 物理参数标定带宽 | ~50% | 量级以下（OMP 散度未必影响 parameter 校准 acceptance, 但仍混淆 sensitivity analysis） |

结论：PR-H 10-25% mean_rel 跨过 gauge 精度门 + 同 forcing 不确定性等价 → **不能** 作为 production；fast-omp mode 仅适合 experimental / parameter sweep / draft 阶段，per E′ 4-mode spec.

## §10 P1e (F 路) 目标

### §10.1 strict-omp mode 加速比验收（量化目标加 §1.1.1 strict-omp 行 by PR-M / P1e proposal）

| Case 规模 | NumEle | N=8 加速比 M (最小可接受) | N=8 加速比 T (目标) | 说明 |
|---|---|---|---|---|
| Small (keliya) | <1k | 1.0× | 1.3× | overhead-dominated；strict 仅要 reproducible，不阻 ship |
| Medium (qhh + heihe ~6k) | 1k-10k | **1.5×** | 2.0× | M 目标 = "真正吃到 RHS 并行的最小信号" |
| Large (heihe_x4 ~25k) | 10k-100k | **1.8×** | 2.4× | T 目标 = B1a profile 66.55% RHS 的 Amdahl 8 核理想上界 |
| XLarge (heihe_x16 ~100k+) | >100k | 2.0× | 3.0× | 留 P2+ |

### §10.2 fallback 路径（per ADR-0002, by Phase 2(e) 并行 agent）

若 P1e mode C（Serial N_Vector + StrictOMP RHS）跨 N 不可达 strict bitwise：

| 路径 | 描述 | 评估顺序 |
|---|---|---|
| NVECTOR_REPRO_OMP custom backend | serial-order left-fold reduction (确定性 wrapper at N_Vector layer) | 第一 fallback (per Pro2 推荐) |
| SPGMR + block-Jacobi physics-based precond | element/river/lake 3×3 小块独立 setup + solve | 第二 (per Pro2 推荐) |
| KLU sparse direct | full Jacobian 直接 LU 分解 | 第三 (per ADR-0002, 仅在前两个不 work + 量化 fill ratio + memory peak + factor wall 后单独决策) |

## §11 References

| 文档 | 用途 |
|---|---|
| `docs/p1d/p1d_summary.md` | epic capstone (§1-§12) |
| `docs/p1d/p1d_pr_h_final_run.md` §"Post-verdict 修订" | 实测散度 + Amdahl 反推 + E′ + F 路决策 |
| `docs/p1d/p1d_pr_f_intermediate_run.md` | `--interleave=all` anti-pattern finding |
| `docs/p1d/p1d_kahan_revert.md` §"SHA matrix" | PR-G Mac 9-SHA proves clean revert |
| `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` | Mac anchor 6-cell |
| `docs/p1d/p1d_numa_root_cause.md` | 散度根因技术解释 (本 doc 的 supplement) |
| `docs/profile_decision.md` | B1a era profile_B0.target.yaml RHS 66.55% 上界依据 |
| `SHUD_openMP_master_plan.md` v1.5 §1.1.1 + §6 P1d + §6 P1e | 量化加速比目标 + P1d 章 + P1e 章 |
| `CLAUDE.md` §1.1.1 + §"Slurm 三铁律" | 量化目标只在服务器验收 + 90 天截断纪律 |
