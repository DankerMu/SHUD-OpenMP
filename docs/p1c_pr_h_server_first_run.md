# P1c PR-H — Server PR-K2 首跑 (8-cell success gate)

服务器 `frd_muziyao@210.77.77.22:32099` (xnode login + cn0X compute partition) Slurm sbatch
SHUD@de9545d (post-PR-E HEAD, openmp-baseline) `make shud_omp` build. 测两 case × 四 N (heihe + heihe_x4 × N∈{1,2,4,8}) = **8 cell** bitwise + nst 全等门控。本 PR 是 P1c 阶段 **SHALL** 级验收点，对应 master plan §4 + 设计 D3 + 设计 D7 (Mac vs Server 分级).

## §1 Server build verification

| 项目 | 值 |
|---|---|
| Project path | `/scratch/frd_muziyao/SHUD-OpenMP/` |
| SHUD pin | `de9545d5fbe43a5e6cbd18ca31b8d4d3617f2e11` (= `de9545d`, post-PR-E) |
| Outer baseline/P1c HEAD | `934944e` (post-PR-G merge) |
| Binary SHA256 | `9f39d38f4e84c6745b4d07e36dd51cc433a9fae23a4e3fed925251e016f4dd6e` |
| Binary path | `SHUD/shud_omp` (size 2077120 bytes) |
| Build host | xnode (frontend login, build-only per 三铁律) |
| Build time (UTC+8) | 2026-06-22 ~15:18 |

### 严格 FP flag 二项 gate (master plan §8.1.1 canonical)

`make shud_omp` build log grep 必含 (两项均须 present)：

| flag | present? | grep occurrences |
|---|---|---|
| `-ffp-contract=off` | PASS | 2 (single + multi-CXX compile) |
| `-fno-fast-math` | PASS | 2 |

Build log path `/tmp/make_shud_server_pr_h.log` (compute-node ephemeral, snapshotted at build time).

并 forbidden flags `-ffast-math` / `-Ofast` 不出现 (per §8.1.1).

### 8-cell 部署条件

| case | source | element count | basin disk size | cfg.para 截断 | NUM_OPENMP 模板值 |
|---|---|---|---|---|---|
| heihe | `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe/` | 6335 | 37 MiB | START=14245, END=14335 (= 90 days) | 1 → sed 替换 |
| heihe_x4 | `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/heihe_x4/` | ~25344 (rSHUD v2.5.0 4× 加密 from heihe) | 2.3 GiB | START=1, END=91 (= 90 days, day-index 制) | 1 → sed 替换 |

CMFD V0200 forcing (per CLAUDE.md 铁律) 已 deployed in `/volume/data/ForcingData/CMFD2.0/Data_forcing_03hr_010deg/{LRad,Prec,Pres,RHum,SHum,SRad,Temp,Wind}/`，basin 端 symlink。

## §2 Slurm submission matrix

Sbatch template `/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs/run_p1c_case.sbatch` per CLAUDE.md 三铁律:
- (1) sbatch from `/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs/` ✓ (不从 `/users/$USER`)
- (2) `--output` / `--error` 路径 `/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs/logs/%x_%j.{out,err}` ✓ (compute node ephemeral `/tmp` 禁用)
- (3) NEVER run SHUD on login node ✓ (执行行 `OMP_NUM_THREADS=${N} shud_omp ${CASE}` 在 compute node 内)

每 (case, N) 独立 run dir `${ROOT}/.p1c-runs/${CASE}_N${N}/`，cfg.para 复制+sed `NUM_OPENMP\t${N}`，其余文件 symlink 至源 basin (避免 2.3 GiB heihe_x4 × 4 = 9.2 GiB 复制开销)。

| job_id | case | N | node | submit time | 备注 |
|---|---|---|---|---|---|
| 8925 | heihe | 1 | cn03 | 2026-06-22 ~15:46 | 首跑探针 (test before fan-out) |
| 8926 | heihe | 2 | cn03 | 2026-06-22 ~15:54 | |
| 8927 | heihe | 4 | cn03 | 2026-06-22 ~15:54 | |
| 8928 | heihe | 8 | cn03 | 2026-06-22 ~15:54 | |
| 8929 | heihe_x4 | 1 | cn08 | 2026-06-22 ~15:54 | |
| 8930 | heihe_x4 | 2 | cn08 | 2026-06-22 ~15:54 | |
| 8931 | heihe_x4 | 4 | cn08 | 2026-06-22 ~15:54 | |
| 8932 | heihe_x4 | 8 | cn08 | 2026-06-22 ~15:54 | |

8 个 sbatch 提交全 ACCEPT，节点分配 heihe 全 cn03 / heihe_x4 全 cn08。两个节点各承 4 jobs × cpus-per-task=8 = 32 CPU (cn0X 物理 32C，不超订阅)。

## §3 8-cell rivqdown.dat SHA256 (实测)

每 run dir `${ROOT}/.p1c-runs/${CASE}_N${N}/rivqdown.sha256` 由 sbatch 末尾 `sha256sum` 写出。SHA full 32-byte 64-hex 完整保留。

| case | N=1 SHA | N=2 SHA | N=4 SHA | N=8 SHA |
|---|---|---|---|---|
| heihe | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | `7f7a621cf1c4f02bb762b3171af4b3b2f647e153735559c7bc1b9bf1f388fc52` | `8c581172a17db5371a7a8537aba7b72442ce487698a11d767dd561151c767889` |
| heihe_x4 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | `7e8f7a8a9697279ed92bf46bf06a8ccd7e3842c65a82194ca4973f0cdb3f347d` | `8b0efa6f6a74a43ad8d74697fbd25bb291a41be5fb2fd5b4e7f4e5fe1da00ac9` |

### §3.1 N=1 ≡ N=2 现象 (内部 2-thread floor)

`shud_omp` stdout 显示 cfg.para `NUM_OPENMP=1` 实跑 OpenMP NVector "No of Threads = 2"; cfg.para `NUM_OPENMP=2` 同样 "No of Threads = 2"; cfg.para `NUM_OPENMP=4/8` 真实为 4/8。SHUD 内部 `max(NUM_OPENMP, 2)` 入 NVector 初始化, 故 cfg=1 与 cfg=2 在 OMP 层 byte-identical, SHA 必匹配。

| job_id | NUM_OPENMP (cfg) | 实跑 threads | wall (s) |
|---|---|---|---|
| 8925 | 1 | 2 | 530 |
| 8926 | 2 | 2 | 502 |
| 8927 | 4 | 4 | 479 |
| 8928 | 8 | 8 | 472 |
| 8929 | 1 (heihe_x4) | 2 | 1182 |
| 8930 | 2 (heihe_x4) | 2 | 1178 |
| 8931 | 4 (heihe_x4) | 4 | 1240 |
| 8932 | 8 (heihe_x4) | 8 | 1212 |

PR-J 反向兼容 (NUM_OPENMP=1 vs P1-update-omp-tag canonical SHA) 评估须考虑此 2-thread floor (P1 update omp tag canonical run 同行为则为 P1-relative 一致, 不同则定 bahavior 漂移)。

## §4 8 cvode_stats.txt — 15-key 实测

`cvode_stats.txt` 实际 column 名: nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS (= 15 keys per CVODE 6.0.0)。

| case | N | nst | nfe | nfeLS | nni | nli | nsetups | netf | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heihe | 1 | **6773** | 7035 | 12885 | 7034 | 12885 | 0 | 0 | 0 | 0 | 6 | 93 | 277730 | 53 | 256338 | 42 |
| heihe | 2 | **6773** | 7035 | 12885 | 7034 | 12885 | 0 | 0 | 0 | 0 | 6 | 93 | 277730 | 53 | 256338 | 42 |
| heihe | 4 | **6682** | 6947 | 12710 | 6946 | 12710 | 0 | 0 | 0 | 0 | 9 | 91 | 277730 | 53 | 256338 | 42 |
| heihe | 8 | **6548** | 6780 | 12320 | 6779 | 12320 | 0 | 0 | 0 | 0 | 6 | 78 | 277730 | 53 | 256338 | 42 |
| heihe_x4 | 1 | **6571** | 6733 | 30559 | 6732 | 30559 | 0 | 0 | 0 | 0 | 48 | 3702 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 2 | **6571** | 6733 | 30559 | 6732 | 30559 | 0 | 0 | 0 | 0 | 48 | 3702 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 4 | **6568** | 6730 | 30483 | 6729 | 30483 | 0 | 0 | 0 | 0 | 47 | 3604 | 1617224 | 53 | 1492794 | 42 |
| heihe_x4 | 8 | **6570** | 6721 | 30568 | 6720 | 30568 | 0 | 0 | 0 | 0 | 47 | 3795 | 1617224 | 53 | 1492794 | 42 |

**heihe Δ_nst** = max(6773, 6773, 6682, 6548) - min(同) = **225** (catastrophic ≫ 2)
**heihe_x4 Δ_nst** = max(6571, 6571, 6568, 6570) - min(同) = **3** (just over D9 ≤2 boundary)

## §5 §4.4 A3a bitwise verdict — **FAIL**

- **heihe 4 N SHA 全等?** ✗ **FAIL** — 3 distinct SHAs (N=1=N=2 ≠ N=4 ≠ N=8)
- **heihe_x4 4 N SHA 全等?** ✗ **FAIL** — 3 distinct SHAs (相同 pattern: N=1=N=2 ≠ N=4 ≠ N=8)

Per master plan §4.4 + spec L62-L72 strict gate: 一 case 内任一 N pair 字节不等 → A3a FAIL → §4.7 trigger TRUE。

## §6 §4.5/§4.6 nst + 15-key verdict — **FAIL**

### §4.5 nst across N — FAIL

- **heihe nst 4 N 全等?** ✗ FAIL — {6773, 6773, 6682, 6548}, |Δ_max| = 225 (≫ 2)
- **heihe_x4 nst 4 N 全等?** ✗ FAIL — {6571, 6571, 6568, 6570}, |Δ_max| = 3 (> 2)

设计 D9 边界:
- heihe: |Δ_nst|=225 ≫ 2 → 直接 §4.7 Kahan 注入 (主分支)，不走 SPGMR-noise ladder。
- heihe_x4: |Δ_nst|=3 > 2 → 同样直接 Kahan 注入 (理论 ≤2 走 SPGMR-ladder, 此处 3 越界)。

### §4.6 15-key (除 nst 外 14 项) — FAIL

跨 N 不全等 keys:
- heihe: nfe (7035/7035/6947/6780), nfeLS (12885/12885/12710/12320), nni (≡nfe-1), nli (≡nfeLS), ncfn (6/6/9/6), ncfl (93/93/91/78). 14 项中 6 项漂移 (剩 8 项 lenrw/leniw/lenrwLS/leniwLS/nsetups/netf/npe/nps 全等)。
- heihe_x4: nfe (6733/6733/6730/6721), nfeLS (30559/30559/30483/30568), nni (≡nfe-1), nli (≡nfeLS), ncfn (48/48/47/47), ncfl (3702/3702/3604/3795). 同 6 项漂移。

`nfe / nni / nli / nfeLS` 漂移 = OMP 跨 N 不同的 CVODE 收敛步数, 与 nst 漂移同源 (RHS 浮点 round-off 顺序差异)。`ncfn / ncfl` 漂移 = convergence failure / linear-system failure 计数 跨 N 不同 = OMP 路径在 SPGMR/Newton 迭代上分歧。

## §7 §4.7 Kahan injection trigger decision — **TRIGGER PR-I**

判定:
- §4.4 A3a FAIL (heihe + heihe_x4 双 case) ✓
- §4.5 nst FAIL (heihe |Δ|=225, heihe_x4 |Δ|=3) ✓
- §4.6 15-key 6 项漂移 ✓

→ **§4.7 SHALL TRIGGER**：PR-I (#252) 走条件 Kahan injection 主分支 (per PR-G `docs/p1c_kahan_patch.diff` + `docs/p1c_a3a_root_cause.md` §"Kahan 候选路径" §(e) 应用流程)，PR-K2 二跑。

二跑命令 (PR-I 内执行):
```bash
cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD
git apply ../docs/p1c_kahan_patch.diff
git commit -m "P1c §4.7 conditional Kahan injection (Neumaier 1974) — PR-K2 二跑 trigger"
git push origin openmp-baseline
cd .. && git add SHUD && git commit -m "SHUD pointer bump: Kahan-injected"
make -C SHUD clean && make -C SHUD shud_omp
# 重提交 8-cell sbatch (clone of run_p1c_case.sbatch, output to .p1c-runs-kahan/)
```

### Mac 模式 vs Server 模式 reproducibility 观察

PR-F `docs/p1c_perf_baseline.md` §1 Mac 16-cell scan: 全 FAIL with 同 "N=1=N=2 / N=4 / N=8 = 3 distinct SHAs" pattern。Server PR-H 复现同 pattern。设计 D7 假设 "Mac fail-pattern 不代表 server 行为" 此处不成立 — server 与 Mac 在 catastrophic RHS round-off 上同 vulnerable. Mac 仍 SHOULD-level (per D7 不是 SHALL gate)，但其 fail 是 server fail 的 early signal。a3a_root_cause.md §"Kahan 候选路径" §(c) 触发条件不变 — 仍只 server PR-K2 触发 (这里), Mac 不触发 (Mac results 信息性 only)。

## §8 Hand-off — FAIL 路径

- **PR-J reverse-compat (#253)** 暂缓: 待 Kahan-injected 二跑 PASS 后用 Kahan binary 跑 NUM_OPENMP=1 vs P1-update-omp-tag canonical SHA (因 P1 update omp tag canonical 是 pre-Kahan binary 跑出的, 真实 reverse-compat 需 evaluate "Kahan binary @ N=1 是否与 pre-Kahan binary @ N=1 字面相等"; 若否, 需 master plan §3 标 "P1c 引入 NUM_OPENMP=1 数值漂移" 路径, 设计 D11 immutability 仅保 tag SHA 不变, 不保证 binary 跑出 SHA 不变)。
- **PR-I (#252)** 由 NOT-TRIGGERED 路径切到 TRIGGER 主分支: 写 Kahan injection 二跑 8-cell 结果文档。
- **PR-K capstone (#254)** 改为采集 Kahan-injected 数据 + Δ_wall 1-3% 实测 (per design R2)。

完成判定: **PR-H 任务完成 — §4.7 trigger fired, hand-off PR-I with TRIGGER signal**。本 PR 不强制 PASS-A3a, 而是 documenting 实测结果 + fire trigger gate, 让 PR-I 处理 Kahan injection。
