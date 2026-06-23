# P1 阶段全运行输出回归按位一致性验证 — Mac 4 案例

## 验证范围 (tasks.md L52-L55 / spec L143-L161)

依 `openspec/changes/p1-update-omp/tasks.md` L54：

> 5.3 Mac 本地 4 case 用 P1 候选 commit 跑 `tools/archive_b0_output.sh <case> 3`
>     完成 3-run 自洽 + 完整 run canonical summary SHA ≡ B1b/B1-tag baseline
>     + CVODE 15-key stats identical

精度等级：**A1（重构等价 / refactor equivalence）**，依 master plan §2.2。以 P1 候选 commit 编译产出的串行 (serial) `shud` 二进制须按位再现 B1b 规范 (canonical) 汇总 SHA。此门控与 B0 / B1a / B1b 归档阶段所采用的规范 SHA 门控完全一致（参见 `docs/status_matrix.md` L13 / L64 / L93 / L132 / L155）。

1. 4 个 Mac 案例 × `tools/archive_b0_output.sh <case> 3` 三轮归档脚本。
2. 每个案例设两项门控：
   - **G1 自洽 (self-determinism)**：3 轮运行的规范汇总 SHA 互相一致。
   - **G2 对标 B1b**：规范汇总 SHA 等于 `benchmarks/<case>/B0_output/repeatability.txt` 字段 `sha256_run1`。
3. SHUD pin：`07c677f`（即三 pragma 栈 — PR-D element + PR-E river + PR-F lake）。
4. 二进制：`SHUD/shud`（串行；以 `make shud` 构建）。CI `serial-baseline / build-and-compare` 通过 `.github/workflows/serial-baseline.yml` L699 / L917 调用同一二进制与同一门控。

## Tag 链与黄金来源

1. `B0 ≡ B1a ≡ B1b ≡ B1`，依 `docs/status_matrix.md` L107-L113 + L155（4 Mac case canonical SHA chain）。
2. 黄金来源：`benchmarks/<case>/B0_output/repeatability.txt` 之 `sha256_run1` 字段，于 B0-tag 归档时设定，至 B1a / B1b 期间冻结。
3. 规范 SHA 算法（即 `tools/archive_b0_output.sh` 第 109-117 行 + 318-362 行所定义）：先按 manifest `output_files` 集合对此次运行实际产出的文件 + `cvode_stats.txt` 各取 SHA256，按 `<hash>  <path>` 每行一条写入 `/tmp/<case>_run<N>.sha256`；规范汇总即对该哈希清单文件字节内容再取一次 SHA256。

## 归档子集诚实披露

manifest 字段 `output_compare.output_files` 列出各案例 6 / 6 / 6 / 9 个 dat 文件（即 §S0.13 spec 层前向占位）。`cfg.para DT_*=0` 在部署默认下禁用了大部分通道，故规范 SHA 实际覆盖的**物理产出**子集为：

| case | files actually hashed under canonical SHA |
|---|---|
| keliya | `keliya.rivqdown.dat` + `cvode_stats.txt`                            (2) |
| xinanjiang_upstream | `xinanjiang.rivqdown.dat` + `xinanjiang.eleygw.dat` + `cvode_stats.txt` (3) |
| qinyijiang | `nanlin.rivqdown.dat` + `cvode_stats.txt`                            (2) |
| qhh | `qhh.rivqdown.dat` + `qhh.lakystage.dat` + `qhh.lakqrivin.dat` + `qhh.lakqrivout.dat` + `cvode_stats.txt` (5) |

manifest 中其余条目（`DT_*=0` 状态下的 `eleysurf` / `eleyunsat` / `eleysnow` / `eleygw`、`rivystage`、`flood.csv`）在当前 `cfg.para` 配置下不会由 SHUD 产出 — `archive_b0_output.sh`（第 340-348 行）及 B0-tag 之 `missing_manifest_files` 字段已记录此 NWM 部署遗留缺口（详见 S0-4 / issue #11）。单一规范汇总 SHA 仅覆盖实际产出的文件；此即 B0 / B1a / B1b 历史门控的范畴，也是 `sha256_run1` 所记录的对象。

## 4 案例 × 2 门控矩阵

| case | new canonical SHA | golden sha256_run1 | G1 self-det | G2 vs B1b |
|---|---|---|---|---|
| keliya | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | PASS | PASS |
| xinanjiang_upstream | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | PASS | PASS |
| qinyijiang | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | PASS | PASS |
| qhh | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | PASS | PASS |

**汇总：G1 自洽 4 / 4 PASS · G2 对标 B1b 4 / 4 PASS = 8 / 8 PASS。**

## CVODE 15 键统计按位一致 (issue #220 验收项 L9)

以 `tools/cvode_stats_diff/cvode_stats_diff.sh <new> <golden>` 对每个案例新生成的 `cvode_stats.txt`（取自 `archive_b0_output.sh <case> 3` 的第 3 轮运行）与 `benchmarks/<case>/B0_output/cvode_stats.txt` 进行比对：

| case | cvode_stats_diff exit | verdict |
|---|---:|---|
| keliya | 0 | PASS |
| xinanjiang_upstream | 0 | PASS |
| qinyijiang | 0 | PASS |
| qhh | 0 | PASS |

CVODE 全部 15 个规范键 (`nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS`，依 `tools/cvode_stats_diff/canonical_15_keys.yaml` + design D10) 在全部 4 个案例上、新运行与 B1b 黄金之间逐字节相同。

## 各案例 wall 时间 (90 天 × 3 轮，NUM_OPENMP=1，串行 shud)

| case | run1 (s) | run2 (s) | run3 (s) |
|---|---:|---:|---:|
| keliya | 28 | 28 | 27 |
| xinanjiang_upstream | 4 | 5 | 4 |
| qinyijiang | 242 | 239 | 238 |
| qhh | 83 | 86 | 91 |

（宿主：Darwin 24.6.0 arm64，Apple Silicon Mac 本地；属开发期参考数据，不构成 §1.1.1 验收数据。）

## 交叉链接 — A0 / A1 验证三股独立证据流

1. **PR-H 规范 RHS 快照按位一致**（中段状态，`docs/p1_rhs_snapshot_bitwise.md`）— Mac 4 案例在 NUM_OPENMP=1 下对标 B1b RHS dumps 取得 12 / 12 PASS（以 `SHUD_DUMP_RHS=1` 编入的 `shud_omp`）。
2. **PR-I 全运行规范汇总 SHA 按位一致**（本文档）— Mac 4 案例在 NUM_OPENMP=1 下对标 B1b 归档 `sha256_run1` 取得 4 / 4 PASS（串行 `shud`，经 `tools/archive_b0_output.sh` 调用）。
3. **CI `serial-baseline / build-and-compare(keliya)`** — 在 PR-D / PR-E / PR-F / PR-H 历次合并上保持 GREEN（≥ 7 次 GREEN 运行），采用同一 `make shud` 编译路径与逐 `*.dat` SHA256 门控（`.github/workflows/serial-baseline.yml` L905-L955）。

综合三股证据，三 pragma 栈（PR-D 单元循环 + PR-E 河道循环 + PR-F 湖泊循环）下 A0 / A1 基线保持性 (baseline preservation) 由三个正交门控共同佐证：中段 RHS 快照按位一致、运行结束规范汇总按位一致、以及 CI 逐文件字节比对。

## OMP runtime 子注（非门控项）

为求完整性补充：`shud_omp @ OMP_NUM_THREADS=1`（即链入 `-fopenmp` 的 OpenMP runtime 二进制 + 启用三 pragma 栈）在 4 个案例上**不**与串行 `sha256_run1` 黄金按位一致（实测规范 SHA：keliya `9365d812…`、xj_up `b49722dc…`、qinyijiang `98a91323…`、qhh `8ef63a56…`；均满足 3 轮自洽）。此乃预期现象，不构成回归：

1. B0 / B1a / B1b 黄金均由串行 `shud` 二进制归档（各 `repeatability.txt` 字段 `binary: …/SHUD/shud` 已记载）。
2. `shud_omp` 链入 OMP runtime 并在单线程下仍激活 `#pragma omp parallel for` 区域 — 归约树、循环调度器、FMA 选择与无 `-fopenmp` 串行路径不同。
3. Master plan §2.2 之 A0 / A1 门控仅比对**串行 vs 串行**；OMP 二进制全运行按位一致属 A3a 门控（P7 严格阶段），即便如此 §2.2 也要求"**同线程数**按位一致"（NUM_OPENMP=N vs NUM_OPENMP=N），并非 NUM_OPENMP=1 (OMP) 对 serial。
4. `shud_omp @ N=1` 与 B1b 的对比在 RHS 快照（中段状态）层面已由 PR-H 完成（12 / 12 PASS）— 即 master plan §2.2 中 P1 / A2 精度门控之所在。

本子注用以记录两种二进制的数值相等边界 (numerical equality boundary)，不计入 PR-I 判定。

## 复现

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD && make shud
cd /Users/danker/Desktop/Hydro-SHUD/openMP
for c in keliya xinanjiang_upstream qinyijiang qhh; do
  tools/fix_case_paths/fix_case_paths.sh "$c"
  tools/archive_b0_output.sh "$c" 3
  # archive script writes B0_output/repeatability.txt with sha256_run1..3;
  # PASS iff sha256_run1 == golden's sha256_run1 (pre-script snapshot).
done
```

驱动脚本及各案例归档日志保留于 `.s2-103/pr-i/`（gitignored scratch tree）。

## 签署

- signed_at: 2026-06-22
- signer: DankerMu
- signed_against_outer_commit: `3fb14ee`（PR-I 起始时 main HEAD；PR-H 收口 commit `docs(p1): review-loop-log — #219 PR-H capstone`）
- signed_against_SHUD_commit: `07c677f`（P1 三 pragma 栈；PR-F `[#218 PR-F] MD_update.cpp lake loop #pragma omp parallel for`）
- gate: A1 重构等价（串行 `shud` vs B1b）— 4 / 4 案例规范汇总 SHA 按位 PASS，4 / 4 案例 3 轮自洽 PASS = **8 / 8 PASS**。

---

## 服务器章节 (heihe + heihe_x4) — PR-J #221

### 范围 (spec L136-L142；tasks 5.5-5.6)

依 `openspec/changes/p1-update-omp/tasks.md` L57-L58（服务器全运行按位 vs B1b @ NUM_OPENMP=1）：

> 5.5 server cn0X 跑 `tools/archive_b0_output.sh heihe 3` + canonical
>     summary SHA ≡ B1b golden
> 5.6 server cn0X 跑 `tools/archive_b0_output.sh heihe_x4 3` + canonical
>     summary SHA ≡ B1b golden

精度等级：**A1（重构等价）**，依 master plan §2.2。服务器 GCC 严格浮点工具链上的串行 `shud` 须在 **2 个服务器案例**（`heihe`，NumEle = 6335；`heihe_x4` 约 25k，由 rSHUD v2.5 4× 网格加密生成）上按位再现 B1b 之 `sha256_run1`。算法与脚本与 Mac PR-I 完全相同 (`tools/archive_b0_output.sh`)，并在 NUM_OPENMP=1 下间接演练三 pragma 栈（串行构建未链入 `-fopenmp`，参见下方三 grep 门控）。

本章节完成 P1 §1.1.1 之服务器侧验收支：Mac PR-I 覆盖 4 个开发期案例的重构等价；PR-J 覆盖 2 个生产规模 (production-scale) 案例的**服务器平台 A1 验证**。

### Slurm 三铁律合规

依 `CLAUDE.md` 服务器策略：

1. ✓ `sbatch` 由 `/scratch/frd_muziyao/SHUD-OpenMP/.s221-runs/` 提交（非 `/users/$USER`，否则触发 policy 拦截）。
2. ✓ `#SBATCH --output / --error` 路径位于 `/scratch` 共享文件系统（非 compute node 的 `/tmp`，否则作业结束即丢失，表现为 ExitCode 127）。
3. ✓ 脚本与二进制引用 (`tools/archive_b0_output.sh`、`SHUD/shud`) 均位于 `/scratch` 而非 `/tmp`。

Scratch 树：`/scratch/frd_muziyao/SHUD-OpenMP/.s221-runs/`（dot-prefixed → 自动 gitignored）。stdout 与新旧 `repeatability.txt` 本地镜像置于 `.s2-103/pr-j/`。

### 2 案例 × 2 门控矩阵

| case | NumEle | new canonical sha256_run1 | golden sha256_run1 | G1 self-det | G2 vs B1b | jobid | Elapsed | node |
|---|---:|---|---|---|---|---|---|---|
| heihe | 6335 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | PASS | PASS | 8794 | 00:27:18 | cn03 |
| heihe_x4 | ~25000 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | PASS | PASS | 8795 | 01:08:45 | cn03 |

**汇总：G1 自洽 2 / 2 PASS · G2 对标 B1b 2 / 2 PASS = 4 / 4 PASS。**

两个案例 3 轮归档的 SHA 完全一致（G1），且与 B1b-tag 期归档的黄金 `sha256_run1` 按位一致（G2）— 该哈希字段与原归档时 `benchmarks/<case>/B0_output/repeatability.txt` 所记录的字段相同。

### 各案例 3 轮 wall 时间 (90 天 × NUM_OPENMP=1，串行 shud 于 cn03)

| case | run1 (s) | run2 (s) | run3 (s) | mean (s) |
|---|---:|---:|---:|---:|
| heihe | 545 | 534 | 537 | 539 |
| heihe_x4 | 1367 | 1366 | 1370 | 1368 |

cn03 与原 B1b 归档宿主 (Linux 6.8.0-90 / 6.8.0-49) 属同一内核家族，wall 时间偏离黄金期 522 / 505 / 528 s (heihe) 与 1216 / 1211 / 1214 s (heihe_x4) 约 5%。Wall 偏差与数值输出无关（按位 PASS 已覆盖数值等价）。

### 服务器编译 — 严格浮点三 grep 门控

`make clean && make shud` 于服务器 (`/scratch/frd_muziyao/SHUD-OpenMP/SHUD`) 执行，日志保留于 `.s221-runs/make_shud_server.log`：

| flag | grep -c | required | verdict |
|---|---:|---:|---|
| `-fopenmp` | 0 | 0 (serial build, no OMP runtime) | PASS |
| `-ffp-contract=off` | 2 | ≥ 1 (disable FMA contraction) | PASS |
| `-fno-fast-math` | 2 | ≥ 1 (disable relaxed FP) | PASS |

编译命令样例：`g++ -O2 -g -ffp-contract=off -fno-fast-math -std=c++14 …`，浮点确定性旗标集合与 Mac 工具链完全一致；GCC 与 clang 产生不同二进制 SHA（服务器 `3e9e5629…` vs 原 B1b 归档 `00ea9d80…`），然而 NUM_OPENMP=1 下两编译器在同一严格浮点封套内均产出**按位一致的输出**。

### 交叉链接 — 服务器三股独立证据流

1. **PR-B #213 trim 按位 vs B0-tag**（单文件 `rivqdown.dat` SHA，与规范汇总属不同哈希层级）：
   - heihe：`55abad28…`
   - heihe_x4：`f90601ef…`
2. **PR-J #221 全运行规范汇总 SHA 按位 vs B1b**（本章节）：
   - heihe：`675c927c…b95ba8a67`
   - heihe_x4：`3fbcbd5c…fea646c06`
3. **CI `serial-baseline / build-and-compare(keliya)`** 于 PR-D / PR-E / PR-F / PR-H / PR-I 历次合并均保持 GREEN。

PR-B 演练单输出通道的逐文件数值保真度；PR-J 演练整个归档之规范汇总（哈希清单的再哈希），覆盖所有实际产出通道 + `cvode_stats.txt`。两者哈希算法不同，但 A1 基线保持性结论一致。M7 forcing trim 90 天窗口未扰动数值输出 → 在 PR-A / B forcing trim 管线 + PR-D / E / F 三 pragma 栈下保持按位稳定。

### 复现

```bash
# On server, from /scratch (Slurm three-iron-rule):
ssh -p 32099 frd_muziyao@210.77.77.22
cd /scratch/frd_muziyao/SHUD-OpenMP
git fetch origin && git checkout main && git pull --recurse-submodules origin main
# Verify pin: main HEAD ≡ fac3da0 + SHUD ≡ 07c677f
cd .s221-runs
sbatch run_heihe.sbatch       # jobid 8794
sbatch run_heihe_x4.sbatch    # jobid 8795
# After completion: each sbatch backups B0_output/repeatability.txt,
# runs archive_b0_output.sh <case> 3, validates new sha256_run1 == golden
# sha256_run1 + 3-run identical, then restores golden (keeps git tracked
# file clean). New-run repeatability saved to .s221-runs/<case>_new_repeatability.txt.
```

sbatch 脚本、Slurm 日志、以及新旧 `repeatability.txt` 保留于服务器 `.s221-runs/`，并镜像至本地 `.s2-103/pr-j/`（均为 gitignored scratch tree）。

### 签署 — 服务器章节

- signed_at: 2026-06-22
- signer: DankerMu
- signed_against_outer_commit: `fac3da0`（PR-J 起始时 main HEAD；PR-I 收口 commit `docs(p1): review-loop-log — #220 PR-I capstone (Mac 4-case full-run 8/8 PASS)`）
- signed_against_SHUD_commit: `07c677f`（P1 三 pragma 栈；PR-F `[#218 PR-F] MD_update.cpp lake loop #pragma omp parallel for`）
- 服务器二进制：`SHUD/shud`（串行，无 `-fopenmp`），`sha256 = 3e9e56295528b0399aff928d1b44d708da87b37777ea81e0de216a3d12a975f3`，于 cn03（Linux 6.8.0-90-generic x86_64，GCC 严格浮点）。
- gate: A1 重构等价（服务器串行 `shud` vs B1b）— 2 / 2 案例规范汇总 SHA 按位 PASS，2 / 2 案例 3 轮自洽 PASS = **4 / 4 PASS**。
