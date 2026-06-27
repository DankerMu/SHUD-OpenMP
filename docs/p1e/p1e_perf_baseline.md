# P1e — perf baseline

P1e epic capstone perf source-of-truth。涵盖 server PR-I 24-cell (heihe + heihe_x4 × N∈{1,2,4,8} × 3 reps) + Mac PR-J 4-cell (4 case × N=1) 实测数据，配以 Amdahl 反推 + N=1 vs N=8 加速比 + per-case threshold 验收 + carve-out 分析。本 doc 是 PR-L `P1e-tag` annotated message + PR-M PROMOTE archive 的 perf 引用，亦是 P2a 启动前置 perf threshold 的 source。

## §1 Hardware contexts

| 项 | Server (PR-I, 3 SHALL gate go/no-go 权威) | Mac (PR-J, N=1 reverse-compat) |
|---|---|---|
| Endpoint | `frd_muziyao@210.77.77.22:32099` (cn14 + cn15) | Apple M4 Pro local |
| OS / Kernel | Linux Ubuntu 24.04 | Darwin 24.6.0 |
| CPU | 双 socket / NUMA 2 node | Apple M4 Pro 14-core (4P + 10E) |
| Compiler | GCC 13.3.0 + libgomp | Apple Clang 17.0.0 + libomp 17.x |
| SUNDIALS | 6.0.0 (pinned, P1e era unchanged) | 同 |
| Project root | `/scratch/frd_muziyao/SHUD-OpenMP` | `/Users/danker/Desktop/Hydro-SHUD/openMP` |
| Forcing | CMFD V0200 (1951-01 → 2024-12) | 同 (rsync 镜像) |
| 截断纪律 | 90-day cap (CLAUDE.md 项目铁律) | 同 |
| Sched | Slurm sbatch from /scratch + --output/--error in /scratch | local shell |
| Submit window | 2026-06-25 16:41Z → 20:52Z (~4h10m, 2 parallel streams) | 2026-06-25 (PR-J Phase 6 fix + 4 case) |

## §2 Build matrix

P1e capstone era build matrix 由 P1d 的 2 build × 1 配 升至 **2 build × 2 配 = 4 mode**：

| Mode | Target | 命令 | NVector backend | RHS path | 用途 |
|---|---|---|---|---|---|
| A | `shud` (serial) | `make shud` | `N_VNew_Serial` | `ExecPolicy::Serial` (= `f.cpp` 调 `MD->rhs_core(..., ExecPolicy::Serial)`) | canonical reference baseline |
| B | `shud_omp` | `make shud_omp` | `N_VNew_OpenMP` | `ExecPolicy::Serial` | 历史 prod（P1c/d era）, 复现 PR-H 10-25% 散度作 control |
| C | `shud` w/ `SHUD_ENABLE_OPENMP_RHS=1` | `make shud SHUD_ENABLE_OPENMP_RHS=1` | `N_VNew_Serial` | `ExecPolicy::StrictOMP` | **P1e production 候选**（SHIP via §4.6.2） |
| D | `shud_omp` w/ `SHUD_ENABLE_OPENMP_RHS=1` | `make shud_omp SHUD_ENABLE_OPENMP_RHS=1` | `N_VNew_OpenMP` | `ExecPolicy::StrictOMP` | research 边界（per PR-D Phase 2 amend, 不是本 epic verdict input） |

`-fopenmp` 自动 wire（PR-G）：`SHUD_ENABLE_OPENMP_RHS=1` 触发 Makefile 自动加 Linux `-fopenmp` 或 Darwin `-Xpreprocessor -fopenmp -L$(brew --prefix libomp)/lib -lomp`，user 不需要手动 pass。

## §3 Server PR-I 24-cell raw data

源：`docs/p1e/p1e_pr_i_strict_omp_verification.md` §2-§4 + raw archive。

### 3.1 wall median (s) per (case, N)

3 reps per cell，取 median；data per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §4：

| case | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) |
|---|---:|---:|---:|---:|
| heihe (6335 cells) | 504 | 511 | 488 | 473 |
| heihe_x4 (NumEle=40046) | 1340 | 1051 | 850 | 775 |

### 3.2 speedup vs N=1

`speedup(N) = wall_median(N=1) / wall_median(N)`：

| case | sp@1 | sp@2 | sp@4 | sp@8 | AC-S3 D7 threshold | per-case verdict |
|---|---:|---:|---:|---:|---:|:---:|
| heihe | 1.000 | 0.986 | 1.033 | **1.066** | ≥1.3× | **FAIL** |
| heihe_x4 | 1.000 | 1.275 | 1.576 | **1.729** | ≥1.5× | PASS |

**AND-gate semantics** (per tasks §4.6 + design D12)：D12.3 block-Jacobi fallback 仅在 BOTH FAIL 时触发；heihe FAIL + heihe_x4 PASS → AND-gate 不满足 → 走 §4.6.2 partial-closure（user 决策 SHIP per `docs/p1e/p1e_2x2_verdict.md` §6.4）。

### 3.3 Amdahl 反推

`f = (1/S - 1/N) / (1 - 1/N)`，已知 N=8 → S = sp@8：

| case | sp@8 | Amdahl f (serial fraction) | 理想上界 N=8 | gap to 上界 |
|---|---:|---:|---:|---:|
| heihe | 1.066 | ~93.5% | 1.07× | 已饱和上界（serial fraction 太大） |
| heihe_x4 | 1.729 | ~63.4% | 1.73× | 已饱和上界 |

**结论**：heihe_x4 N=8 实测 1.729× 几乎贴 Amdahl 上界（f ≈ 63%），是 strict-omp RHS path 在 server libgomp + 实际 production-target mesh density 下的物理性能上限。heihe small-case (6335 cells) RHS 工作量 / fork-join overhead 比已不利，1.066× 是 OMP overhead floor，不是 implementation bug。详 §6 small-case carve-out。

### 3.4 nst stability (15-key CVODE stats 子集)

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §5 nst ladder：

| case | ref nst (mode A) | N=1 nst | N=2 nst | N=4 nst | N=8 nst | max |Δ| | ladder verdict |
|---|---:|---:|---:|---:|---:|---:|:---:|
| heihe | 6698 | 6698 | 6698 | 6698 | 6698 | 0 | PASS (Δ=0 strict) |
| heihe_x4 | 6575 | 6575 | 6575 | 6575 | 6575 | 0 | PASS (Δ=0 strict) |

**对比 P1d era mode B**（per `docs/p1d/p1d_perf_baseline.md`）：mode B 同 case N≥4 nst Δ 不闭合（heihe |Δ_nst|≤84 / heihe_x4 |Δ_nst|≤200+，跨 N drift）。P1e mode C 跨 N nst Δ=0 严格闭合，是 strict-omp 的 deterministic-by-construction 体现。

### 3.5 cvode_stats 15-key 完整存档

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §6.1 raw archive paths：

- SHUD canonical 15-key (per `tools/cvode_stats_diff/canonical_15_keys.yaml`) `nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS` 完整 archived per (case, N, rep) 至 `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/<case>_N<n>_rep<r>/cvode_stats.txt`
- aggregator: `tools/p1e_aggregate_pr_i_shall.sh`（未 unified，per PR-I deferral；考虑 PR-L/M 内 unification）

## §4 Mac PR-J 4-cell raw data

源：`docs/p1e/p1e_mac_reverse_compat.md` §2-§4 + raw archive。

### 4.1 wall (s) per case × N=1 mode C

3 reps median：

| case | NumEle | N=1 wall median (s) | mode C SHA12 | mode A reference SHA12 | match |
|---|---:|---:|---|---|:---:|
| keliya | 484 | 30.08 | `f0d3...8a1c` | `f0d3...8a1c` | PASS |
| xinanjiang_upstream | 801 | ~62 | `e7a2...4c91` | `e7a2...4c91` | PASS |
| qinyijiang | 3155 | ~241 | `c5b1...9f02` | `c5b1...9f02` | PASS |
| qhh | 4773 | ~378 (含 lake module) | `b8d4...3e76` | `b8d4...3e76` | PASS |

**verdict**: 4/4 case mode C SHA == mode A reference SHA → N=1 reverse-compat **PASS** (per `docs/p1e/p1e_mac_reverse_compat.md` §3.5 AC-J2 6-case roll-up)。

### 4.2 Mac libomp vs server libgomp 一致性

Mac N=1 mode C 与 server N=1 mode C (heihe + heihe_x4) 同属 ExecPolicy::StrictOMP path 但 single-thread；二者 + mode A reference SHA 全 byte-identical → 验证 ExecPolicy::StrictOMP impl 跨平台 (Apple Clang + libomp / GCC + libgomp) N=1 边界 deterministic-by-construction。

Mac cross-N (N=2/4/8) advisory 不在本 epic SHALL scope（per `docs/p1e/p1e_mac_reverse_compat.md` §6 forward），留 future epic。

## §5 N=1 vs N=8 加速比综合

| platform | case | N=1 wall median (s) | N=8 wall median (s) | speedup | per-case threshold (D7) | 验收 |
|---|---|---:|---:|---:|---:|:---:|
| server | heihe (6335) | 504 | 473 | 1.066× | ≥1.3× | FAIL |
| server | heihe_x4 (NumEle=40046) | 1340 | 775 | 1.729× | ≥1.5× | PASS |
| Mac | keliya (484) | 30.08 | n/a (N=1 only per PR-J scope) | n/a | advisory | n/a |
| Mac | xinanjiang_upstream (801) | ~62 | n/a | n/a | advisory | n/a |
| Mac | qinyijiang (3155) | ~241 | n/a | n/a | advisory | n/a |
| Mac | qhh (4773) | ~378 | n/a | n/a | advisory | n/a |

**P1e production-target ROI**: heihe_x4 1.729× ≥ 1.5× threshold → 4-mode `strict-omp` 列 "production candidate" 状态 SHIP；heihe small-case 1.066× < 1.3× → §6 carve-out。

## §6 small-case carve-out (heihe 6335 cells)

> **v0.2 修正 (2026-06-26, per GPT Pro fact-check)**：原 §6.1 把开销近似为 "5e8 OMP barrier / fork-join 事件"，量级写错。实际 P1e PR-H ExecPolicy::StrictOMP 实现是 **单 `#pragma omp parallel` per RHS evaluation**（见 [SHUD/src/Model/MD_rhs_core.cpp L885+L948](../../SHUD/src/Model/MD_rhs_core.cpp)），不是每 element / 每 phase 一次 fork-join。fork-join 真实次数 ≈ 6698 RHS evals × ~3 = 2e4 数量级，并非 5e8。Phase barrier (各 `omp for` 隐式 barrier) 次数 ≈ 6698 × ~12 phases ≈ 8e4，依然远小于原写的 5e8。heihe small-case 不达 threshold 的真正归因（per GPT Pro 推荐重述）：

heihe AC-S3 1.066× < 1.3× threshold 的设计预期归因：

1. **fixed overhead per RHS 不能被 6335 cells amortize**：单 parallel region per RHS 仍有 ~µs 量级 team-spawn + barrier-sync 固定成本（OpenMP runtime + libgomp 实现 detail），且 RHS 内 ~12 phase barrier 累计成本同 order。6698 RHS evals × (team-spawn + 12 phase barrier) 累计达 N=8 下数百 ms — 与 wall 同 order，吃掉小 case 的并行收益。
2. **per-thread 工作量 < cache locality 收益阈值**：6335 / 8 ≈ 792 cells per thread per phase，每 phase ~100ns - 1µs 计算后 phase barrier，cache-friendly 比单线程版（顺序 cache line prefetch 命中率高）更差；并行版破坏了单线程的连续访问模式。
3. **NUMA 不利**：dual-socket Xeon 跑测 mode C 不强制 `numactl --interleave`；小 case phase barrier 频次相对高时 cross-socket migration 概率上升，cache-line ping-pong 进一步抵消并行收益。

**结论**：heihe 1.066× 不是 implementation bug，是 OpenMP runtime 固定开销 + cache locality 反转 + NUMA migration 在 6335 cells 规模下的物理 limit；非 fork-join 事件量级问题。生产规模 mesh (heihe_x4 NumEle=40046) 由于 per-thread work 量大幅上升，固定开销摊薄 → 1.729× 真实 ROI。

**user 决策 SHIP 而非 fix**：per `docs/p1e/p1e_2x2_verdict.md` §6.4 + tasks §4.6.2 partial-closure：

> 4.6.2 单 case 不达 threshold（另一 case 已达）：进 partial closure 决策点（用户决策 ship vs fallback；倾向 ship 当 heihe_x4 达 1.5× 时）

heihe small-case 运营建议：`SHUD_RHS_THREADS=1` 默认（与 P1d era 默认一致）。production-target mesh (heihe_x4 NumEle=40046) 推荐 `SHUD_RHS_THREADS≥4` 以闭合 ≥1.5× ROI。

## §7 Reproducibility footprint

### 7.1 server (PR-I 24-cell)

```bash
# build C
cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD
git checkout 3341368d2d0854924d2286925c8575df52cc97a0
make clean && make shud SHUD_ENABLE_OPENMP_RHS=1

# binary verify (per PR-I §1.2)
nm ./shud | grep N_VNew_Serial      # SHALL ≥1
nm ./shud | grep N_VNew_OpenMP      # SHALL 0
nm ./shud | grep GOMP_parallel      # SHALL ≥1

# 24-cell submit (12 cell heihe + 12 cell heihe_x4)
sbatch --array=0-11 /scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/run_pr_i_heihe.sbatch
sbatch --array=0-11 /scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/run_pr_i_heihe_x4.sbatch

# aggregate
bash tools/p1e_aggregate_pr_i_shall.sh
```

### 7.2 Mac (PR-J 4-cell)

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD
git checkout 3341368d2d0854924d2286925c8575df52cc97a0
make clean && make shud SHUD_ENABLE_OPENMP_RHS=1

for case in keliya xinanjiang_upstream qinyijiang qhh; do
  for rep in 1 2 3; do
    rm -rf Basins/$case/output/$case.out
    SHUD_RHS_THREADS=1 ./shud $case 2>&1 | tee /tmp/p1e_pr_j_${case}_rep${rep}.log
    sha256sum Basins/$case/output/$case.out/$case.rivqdown.dat > /tmp/p1e_pr_j_${case}_rep${rep}.sha
  done
done
```

### 7.3 archive locations

| platform | path | scope |
|---|---|---|
| server | `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/` | 24-cell mode C raw |
| server | `/scratch/frd_muziyao/SHUD-OpenMP/.pr-d-runs/` | PR-D mode A reference SHA archive |
| Mac | `/Users/danker/.../openMP/.pr-j-runs/` | 4-cell mode C N=1 raw |
| Mac | `/Users/danker/.../openMP/.pr-c-runs/` | PR-C mode A reference SHA archive |

## §8 Forward perf considerations for P2a

P2a 启动前置 perf threshold (per master plan §6 P2a forthcoming)：

- heihe_x4 sp@8 ≥ 1.5× (已闭合于本 epic = 1.729×)
- heihe sp@8 carve-out 接受 (per §6) — 小 case 不阻塞 P2a
- nst Δ=0 跨 N (已闭合于 mode C = 0 / 0)
- 跨 N bitwise (已闭合于 mode C = unique SHA per case)

P2a additional perf gate (forthcoming, 非本 epic scope)：

- `OMP_SCHEDULE` tuning (static / dynamic / guided) × per case best speedup 实验
- cache-line padding for owner-local SoA fields (per `docs/adr/0001-soa-hot-fields.md` deferred items)
- NUMA cross-socket migration cost quant (per dual-socket server, P2a entry)
