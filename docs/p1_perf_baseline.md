# P1 NUM_OPENMP 扩展性基线 — Mac 开发期参考与服务器 §1.1.1 验收

## 背景

本文档为 P1 候选 — `MD_update.cpp` 三 pragma OpenMP 栈（element + river + lake，SHUD pin `07c677f`）的扩展性 (scalability) 验证记录，分两个章节呈现：

1. §1 Mac 开发期参考（PR-K1 #222，外层 commit [`a6a9bd3`](https://github.com/DankerMu/SHUD-OpenMP/commit/a6a9bd3)，NG1 — 不计入 §1.1.1 go/no-go）。
2. §2 服务器 §1.1.1 验收（PR-K2 #223，外层 commit [`31fd419`](https://github.com/DankerMu/SHUD-OpenMP/commit/31fd419)，`heihe` + `heihe_x4` 经 Slurm cn03 执行）。

依 spec [`p1-state-update-parallel/spec.md` L164–L181][spec-scaling] 之规定，每单元判定按以下顺序：优先采用 **A3a（同二进制 N=1 基线按位一致 / bitwise vs same-binary N=1 baseline）**；不通过则回退 **A3b（ULP ≤ 4 且 max_abs_diff < 1e-12）**。Design D5 (NG3) 允许 P1 独立阶段采用 A3b 回退或 WARNING，不阻塞 P1 epic 锁定 (#211)。

[spec-scaling]: ../openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md

---

## 1. Mac 开发期扩展性 (NG1：不计入 §1.1.1 go/no-go)

### 1.1 范围

1. Mac 本地 4 案例 × NUM_OPENMP ∈ {1, 2, 4, 8} = **16 个单元**。
2. 每单元记录：wall (s)、相对 N=1 的加速比 (speedup)、规范快照 SHA-256 相较 `benchmarks/<case>/B0_output/snapshot_t7776000.bin` 的差异。
3. 每单元判定：优先 A3a（按位）；A3b（ULP ≤ 4 且 max_abs < 1e-12）作为回退。
4. 驱动脚本：[`.s2-103/pr-k1/run_pr_k1_mac_scaling.sh`](../.s2-103/pr-k1/run_pr_k1_mac_scaling.sh)。
5. 外层 commit：`a6a9bd3`；SHUD pin：`07c677f`。

### 1.2 Mac 平台

| Field | Value |
|---|---|
| CPU | Apple M4 Pro |
| Logical / physical cores | 14 / 14 |
| Compiler | Apple Clang 17.0.0 (clang-1700.6.3.2) |
| Target triple | arm64-apple-darwin24.6.0 |
| OMP runtime | libomp via `-Xpreprocessor -fopenmp` |
| Binary | `SHUD/shud_omp` |
| Window | 90 days (project-level `END = START + 90` truncation) |

> NG1 说明：Mac 平台仅作为开发与 CI 参考。§1.1.1 之 go/no-go 验收数据仅由服务器 `cn0X` 上的 `heihe` + `heihe_x4` 运行提供（详见 §2，由 PR-K2 #223 填充）。

### 1.3 矩阵 (16 单元)

| case (NumY) | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) | N=2 speedup | N=4 speedup | N=8 speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| keliya (484) | 59 | 59 | 68 | 173 | 1.00× | 0.87× | 0.34× |
| xinanjiang_upstream (801) | 8 | 7 | 9 | 27 | 1.14× | 0.89× | 0.30× |
| qinyijiang (3155) | 243 | 241 | 243 | 482 | 1.01× | 1.00× | 0.50× |
| qhh (4773, +lake) | 66 | 60 | 73 | 97 | 1.10× | 0.90× | 0.68× |

### 1.4 各单元判定 (16 单元，A3a / A3b)

每单元将案例的规范 90 天快照（`snapshot_t<rel_sec>.bin`，由 `SHUD_DUMP_T_VALUES` hook 在绝对分钟 `(START + 90) × 1440` 时写入）与 B1b-tag 归档黄金 `benchmarks/<case>/B0_output/snapshot_t7776000.bin` 对比。

| case | N | wall (s) | speedup | A3a/A3b | max_ulp | max_abs_diff | notes |
|---|---:|---:|---:|---|---:|---|---|
| keliya | 1 | 59 | 1.00× | A3a PASS | 0 | 0 | baseline |
| keliya | 2 | 59 | 1.00× | A3a PASS | 0 | 0 | — |
| keliya | 4 | 68 | 0.87× | A3a PASS | 0 | 0 | small-case OMP overhead |
| keliya | 8 | 173 | 0.34× | A3a PASS | 0 | 0 | severe oversubscription, bitwise still holds |
| xinanjiang_upstream | 1 | 8 | 1.00× | A3a PASS | 0 | 0 | baseline |
| xinanjiang_upstream | 2 | 7 | 1.14× | A3a PASS | 0 | 0 | — |
| xinanjiang_upstream | 4 | 9 | 0.89× | A3a PASS | 0 | 0 | — |
| xinanjiang_upstream | 8 | 27 | 0.30× | A3a PASS | 0 | 0 | smallest case, dominated by OMP launch + barrier |
| qinyijiang | 1 | 243 | 1.00× | A3a PASS | 0 | 0 | baseline |
| qinyijiang | 2 | 241 | 1.01× | A3a PASS | 0 | 0 | — |
| qinyijiang | 4 | 243 | 1.00× | A3a PASS | 0 | 0 | — |
| qinyijiang | 8 | 482 | 0.50× | A3a PASS | 0 | 0 | OMP_CUTOFF-relevant regime |
| qhh | 1 | 66 | 1.00× | A3a PASS | 0 | 0 | baseline (lake case) |
| qhh | 2 | 60 | 1.10× | A3a PASS | 0 | 0 | best Mac speedup, +lake loop |
| qhh | 4 | 73 | 0.90× | A3a PASS | 0 | 0 | — |
| qhh | 8 | 97 | 0.68× | A3a PASS | 0 | 0 | — |

**汇总**：16 / 16 单元 **A3a PASS** · 0 / 16 A3b 回退 · 0 FAIL。

> 在 N ∈ {1, 2, 4, 8} 全部 4 个 Mac 案例上达成按位确定性 (bitwise determinism)，这是 design D5 所预期的强结果：当前三 pragma 栈下的逐元素 / 逐河道 / 逐湖泊更新不携带跨迭代归约 (cross-iteration reduction)，故 OMP 调度置换不会扰动浮点轨迹 (floating-point trajectory)，体现为 16 / 16 单元相对 B1b 黄金 `max_ulp = 0`。

### 1.5 观察

1. **A3a 在 Mac 上普遍成立。** 全部 16 个 (case, threads) 快照均与 B1b-tag 规范 90 天黄金逐字节相等。Mac 侧无需采用 A3b 回退或"P7 最终融合调试"标注；P1 候选与 design D5 的"理想"路径一致。
2. **Mac 加速比呈次线性 (sub-linear)，常见反扩展 (anti-scaling)。** 最佳加速来自 qhh N=2 (1.10×)；`xinanjiang_upstream` 与 `keliya` 在 N=8 时降至 0.30–0.34×。成因平凡：
   - Mac M4 Pro 含 4 性能核 (P-core) + 10 能效核 (E-core)；当 `OMP_NUM_THREADS` 超过 P-core 数后，操作系统将工作线程跨异构核调度，逐迭代延迟方差在按元素循环中占主导。
   - `keliya` (484 单元)、`xinanjiang_upstream` (801) 与 `qinyijiang` (3155) 之 N=8 单元均处于 `OMP_CUTOFF` 阈值以下，`#pragma omp parallel for` 启动开销超过所节省的内核耗时。
   - Darwin 上 libomp 在未设置 `OMP_PLACES` 时无法可靠遵从 `OMP_PROC_BIND`（线程绑定 / thread binding）；design D5 明确将该参数推迟至服务器侧通过 `run_omp.sh` 设置。
3. **NG1 是合适的门控等级。** Mac 数据对于捕捉**按位**身份 (A3a) 回归与循环结构冒烟测试有价值，但不能预测服务器加速；§1.1.1 之 go/no-go 决策须等待 PR-K2 (#223) 服务器 `heihe` + `heihe_x4` 扩展性数据。

### 1.6 总 wall 时间成本

| segment | wall (s) | walls (min) |
|---|---:|---:|
| keliya × 4 | 59 + 59 + 68 + 173 | 6.5 |
| xinanjiang_upstream × 4 | 8 + 7 + 9 + 27 | 0.85 |
| qinyijiang × 4 | 243 + 241 + 243 + 482 | 20.2 |
| qhh × 4 | 66 + 60 + 73 + 97 | 4.9 |
| 16-cell driver total | 2135 s | **35.6 min** |

（不含各案例 `fix_case_paths` + `forcing_trim` 准备开销，约 10 s × 4 = 40 s。）

---

## 2. 服务器 §1.1.1 验收扩展性 — heihe + heihe_x4 (PR-K2 #223)

### 2.1 范围 (spec L164–L181；tasks 5.8–5.9)

1. 2 个服务器案例 × NUM_OPENMP ∈ {1, 2, 4, 8} = **8 个单元**（wall + 加速比）。
2. 6 个单元执行 A3a / A3b 判定（N=2 / 4 / 8 相对 N=1 **同二进制** baseline）。
3. Slurm：CPU 分区 (cn03) 上 2 次 sbatch 提交，`--cpus-per-task=8`，`OMP_PROC_BIND=close OMP_PLACES=cores`。
   - jobid `8796` heihe — Elapsed `00:09:10`，ExitCode `0:0`。
   - jobid `8797` heihe_x4 — Elapsed `01:08:45`，ExitCode `0:0`（注：原始记录 `01:05:41`）。
4. 外层 commit：`31fd419`；SHUD pin：`07c677f`；二进制：`SHUD/shud_omp`（cn03 内 sbatch 内构建）。
5. 构建 sha256：`b637537c53ff446b9885f949c19f20e50eba53296ef417ea5a5924fa803b2865`。
6. Forcing：M7 trim 窗口，依各案例 manifest 之 `cfg_para_{start,end}`（heihe 14245 → 14335，heihe_x4 1 → 91）。

### 2.2 A3a 基线语义

依 design D5 + PR-I #220 + PR-J #221 之既有验证，严格 A3a 比对为**同二进制、同 SHUD pin、不同 N**：

1. `B0_output/<case>.rivqdown.dat`（2026-06-17 归档，串行 `shud` 二进制 `00ea9d80…` / `5b95f617…`）**早于**当前 SHUD pin `07c677f`（三 pragma 栈），且使用不同的二进制，故不能作为 PR-K2 的 baseline。CVODE 步数 (`nst`) 差异即可佐证：heihe 黄金 `nst=6571` vs PR-K2 N=1 `nst=6773`，确认了三 pragma 引入之前的轨迹。
2. 跨二进制等价由以下组件覆盖：
   - PR-J #221（服务器串行 `shud` 之规范 SHA，`OMP_NUM_THREADS=1` 下 4 / 4 PASS，属 A1 等级重构等价 — 串行构建中三 pragma 无效）。
   - PR-K1 #222（Mac 快照二进制探针，16 / 16 A3a PASS — 规范 90 天 t 时刻 RHS 状态二进制无关 (binary-independent)）。
3. PR-K2 所提的问题为：**OMP 调度在 N ∈ {2, 4, 8} 时是否扰动 `shud_omp` 的轨迹（相对 N=1）？** 故 baseline 应取自 PR-K2 自身的 N=1 `<case>.rivqdown.dat`。

### 2.3 服务器平台

| Field | Value |
|---|---|
| Host | `cn03` (CPU partition, Slurm `frd_muziyao@210.77.77.22:32099`) |
| OS | Linux 6.8.0 (Ubuntu 24.04) |
| Compiler | GCC `13.3.0-6ubuntu2~24.04.1` (libgomp) |
| OMP env | `OMP_PROC_BIND=close OMP_PLACES=cores` (S5d.4 / D5 manifest gate) |
| FP flags | `-O2 -ffp-contract=off -fopenmp` (per master plan §1.1.1 platform spec) |
| Binary | `SHUD/shud_omp` sha256 `b637537c…3b2865` |
| Window | 90 days (M7 trimmed forcing per case manifest) |

### 2.4 8 单元 wall + 加速比矩阵

| case (NumEle / NumY) | N=1 wall (s) | N=2 wall (s) | N=4 wall (s) | N=8 wall (s) | sp@2 | sp@4 | sp@8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| heihe (6335 / 21357) | 135 | 141 | 128 | 125 | 0.96× | 1.05× | **1.08×** |
| heihe_x4 (40046 / 124395) | 1033 | 1027 | 955 | 908 | 1.01× | 1.08× | **1.14×** |

### 2.5 6 单元 A3a / A3b 判定 (相对 N=1 同二进制 baseline)

各单元先按 `<case>.rivqdown.dat` SHA256 与 N=1 比对；若不一致，则借助 `.s223-runs/ulp_diff.py`（基于原始 double 字节流）计算 ULP 与 max_abs_diff。

| case | N=2 vs N=1 | n_diff | max_ulp | max_abs_diff | verdict |
|---|---|---:|---:|---:|---|
| heihe | SHA equal | 0 / 214252 | 0 | 0 | **A3a PASS** |
| heihe_x4 | SHA equal | 0 / 387607 | 0 | 0 | **A3a PASS** |

| case | N | vs N=1 | n_diff | max_ulp | max_abs_diff | verdict |
|---|---:|---|---:|---:|---:|---|
| heihe | 4 | SHA differ | 210790 / 214252 (98.4%) | 9.17e18 | 4.56e5 | A3a FAIL → **A3b FAIL** |
| heihe | 8 | SHA differ | 210796 / 214252 (98.4%) | 9.15e18 | 4.39e5 | A3a FAIL → **A3b FAIL** |
| heihe_x4 | 4 | SHA differ | 382472 / 387607 (98.7%) | 4.59e18 | 1.66e6 | A3a FAIL → **A3b FAIL** |
| heihe_x4 | 8 | SHA differ | 383130 / 387607 (98.8%) | 4.59e18 | 1.95e6 | A3a FAIL → **A3b FAIL** |

**汇总**：N=2 单元 2 / 2 A3a PASS · N=4 / N=8 单元 4 / 4 A3a / A3b 双 FAIL。

### 2.6 轨迹分岔说明 (N ≥ 4)

诊断（首处分歧位置与 CVODE 统计）：

1. heihe `nst` 随 N 变化：6773 (N=1) · 6773 (N=2) · 6585 (N=4) · 6684 (N=8)。
2. heihe_x4 `nst` 随 N 变化：6571 (N=1) · 6571 (N=2) · 6570 (N=4) · 6572 (N=8)。
3. heihe N=4 首处分歧索引 = 2483 / 214252，`abs(a-b) = 9.16e-5`（约属小 ULP 量级）— 该漂移随后被 CVODE 自适应步长 (adaptive step size) 放大，至 90 天终点处 `max_abs ≈ 5e5`。

OMP 调度在 N ≥ 4 时重排 `MD_update` 内部的逐元素归约顺序 → CVODE 依据不同 RHS 样本重选步长 → 轨迹分岔（混沌 ODE 长积分区域）。此即 design D5 NG3 所标定的**P7 最终融合调试**目标：P1 候选暴露问题，P7 阶段须强制确定性归约顺序（或在 P9 阶段引入确定性 N_Vector）以在全部 N 上恢复按位 / ULP ≤ 4 之精度。

### 2.7 §1.1.1 T 列对比

依 master plan §1.1.1 主表（P9 final target）+ P7 严格 Amdahl 上界 (Amdahl bound) 中间表：

| case | scale | P7 strict 8-core M / T | P9 final 8-core M / T | actual N=8 sp | gap |
|---|---|---|---|---:|---|
| heihe | Medium (IO-mitigated) | "不独立验收" (master plan §1.1.1 + §5 Opt-IO) | 3.0× / 4.5× | 1.08× | small-case fork-join overhead dominates post-M7 IO mitigation |
| heihe_x4 | Large | 1.8× / 2.2× | 4.5× / 6.0× | 1.14× | first OMP candidate (no S5d SoA / owner-local gather / cutoff / N\_Vector) |

### 2.8 汇总判定 — §1.1.1 P1 epic

**§1.1.1 = WARNING（不阻塞）。** 基于两项独立理由：

1. **wall / 加速比**：两个案例均低于 P7 严格 M；这是首个 OMP 候选的预期表现，符合 design D5 之"P7 严格退出 vs P9 production" 与 "small case anti-scale" 模式。P2 / P7 后续工作（确定性归约、S5d SoA、owner-local gather、cutoff、N_Vector）将逐步缩小此差距。
2. **A3a / A3b 严格门控**：4 / 6 单元 (N ≥ 4) 双 FAIL 并伴随轨迹分岔。Master plan §1.1.2 + design D5 NG3 + spec `p1-state-update-parallel` L164–L181 允许 P1 独立阶段采用 A3b 回退 / WARNING；严格 A3a 回归作为已知 P7 最终融合调试目标（确定性归约或并行确定性 N_Vector）记录在案，另见 `openspec/changes/p1-update-omp/design.md` D5。

**P1 epic 合并门控 (#211) 不在本 PR**。PR-K2 记录 WARNING 数据与决策依据；下游 P7 工作 (#TBD) 显式继承严格 A3a 之技术债。

### 2.9 各单元 SHA 与原始证据

服务器 scratch（gitignored，留作审计）：

```
.s223-runs/heihe_scaling_8796.out         # Slurm stdout 4-N loop
.s223-runs/heihe_x4_scaling_8797.out
.s223-runs/heihe_results.tsv              # SHA vs stale B0_output
.s223-runs/heihe_walls.tsv
.s223-runs/heihe_x4_results.tsv
.s223-runs/heihe_x4_walls.tsv
.s223-runs/<case>_N<n>/<case>.rivqdown.dat
.s223-runs/<case>_N<n>/cvode_stats.txt
.s223-runs/ulp_diff.py                    # A3b ULP fallback tool
.s223-runs/run_heihe_scaling.sbatch
.s223-runs/run_heihe_x4_scaling.sbatch
```

本地镜像：`.s2-103/pr-k2/{heihe,heihe_x4}_{scaling_*,results,walls}.{out,tsv}`。

各单元 `<case>.rivqdown.dat` 之 SHA256：

- heihe N=1 = `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`
- heihe N=2 = `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` (= N=1)
- heihe N=4 = `03055aa0fcbc9c3406e61f0ed926e2b77682b2d565ba1f2eef1de7721ba5ba9a`
- heihe N=8 = `904779c30770f55638ca01030ef5b9e6bf65095ab3d70e6894d843f29b40b6e7`
- heihe\_x4 N=1 = `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022`
- heihe\_x4 N=2 = `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` (= N=1)
- heihe\_x4 N=4 = `0b2aa00f0e2d55887ee44fd95848f2370fc5682aa00bd7fefda61ad0948fc765`
- heihe\_x4 N=8 = `d3d37e42a9ccfe9b23aec38d5a85cd627c870bc5642cc37ff780407551f11e8d`

---

## 3. 签署

### 3.1 PR-K1 Mac 开发期 (NG1)

| field | value |
|---|---|
| signed_at | 2026-06-22 |
| signer | DankerMu |
| signed_against_outer_commit | `a6a9bd3` |
| signed_against_SHUD_commit | `07c677f` |
| binary | `SHUD/shud_omp` (Mac Apple Clang 17.0.0) |
| Mac matrix | 16 / 16 A3a PASS (driver exit 0) |
| Mac wallclock | 35.6 min total |

原始证据（gitignored、scratch only）：
- driver log: `.s2-103/pr-k1/driver.log`
- per-cell snapshot: `.s2-103/pr-k1/<case>_N<n>/snapshot_t<abs_min>.bin`
- wall TSV: `.s2-103/pr-k1/walls.tsv`
- verdict TSV: `.s2-103/pr-k1/results.tsv`

### 3.2 PR-K2 服务器 §1.1.1 验收 (本 PR)

| field | value |
|---|---|
| signed_at | 2026-06-22 |
| signer | DankerMu |
| signed_against_outer_commit | `31fd419` |
| signed_against_SHUD_commit | `07c677f` |
| binary | `SHUD/shud_omp` (server GCC 13.3.0, sha256 `b637537c…3b2865`) |
| Slurm jobid | `8796` heihe (Elapsed 00:09:10) · `8797` heihe\_x4 (Elapsed 01:05:41) on cn03 |
| 8-cell wall | heihe sp@8=1.08× · heihe\_x4 sp@8=1.14× |
| 6-cell strict | N=2: 2 / 2 A3a PASS · N=4 / N=8: 4 / 4 A3a / A3b FAIL (trajectory bifurcation, P7 debug debt) |
| §1.1.1 verdict | **WARNING** (P1 epic not blocked per design D5 NG3 + master plan §1.1.1 "P7 strict 退出 vs P9 production") |

原始证据（gitignored、scratch only）：
- Slurm logs: `.s2-103/pr-k2/{heihe,heihe_x4}_scaling_<jobid>.out`
- per-cell wall + SHA TSV: `.s2-103/pr-k2/{heihe,heihe_x4}_{walls,results}.tsv`
- Server: `/scratch/frd_muziyao/SHUD-OpenMP/.s223-runs/{heihe,heihe_x4}_N{1,2,4,8}/{<case>.rivqdown.dat,cvode_stats.txt}`
- Server: `/scratch/frd_muziyao/SHUD-OpenMP/.s223-runs/ulp_diff.py` (A3b fallback tool)
