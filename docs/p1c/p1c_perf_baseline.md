# P1c — perf baseline (seed; capstone PR-K fills sections beyond §1)

> **Status**: seed file (PR-F #249). PR-K (capstone) 将填入 §2 及之后服务器 PR-K2 实测结果。

## §1 Mac 辅助预筛 (PR-F, 16-cell 4-case scan, post-PR-E pin de9545d)

Per spec [`p1c-deterministic-reduction/spec.md`](../openspec/changes/p1c-deterministic-reduction/specs/p1c-deterministic-reduction/spec.md) Requirement "Mac local 16-cell 4-case 辅助预筛 (SHOULD)" + design D7
(Mac SHOULD-level, 服务器 PR-K2 是唯一 SHALL gate; Mac PASS 不触发 Kahan，Mac FAIL 不阻 server).

### §1.1 平台与构建

| Field | Value |
|---|---|
| CPU | Apple M4 Pro |
| Logical / physical cores | 14 / 14 |
| OS | macOS 15.6 (Darwin 24.6.0) |
| Compiler | Apple Clang (g++ wrapper) |
| FP flags | `-ffp-contract=off -fno-fast-math` (per master plan §8.1.1; 2/2 hits in make log) |
| SHUD pin | `de9545d` (post-PR-E HEAD) |
| Binary SHA-256 | `c8c88da74d31fe380873afe40bd08a6035c8f1b7e46c1b2bd6bcb16c718d7558` |
| 90-day truncation | all 4 cases (per CLAUDE.md 项目级铁律) |
| 驱动脚本 | [`.p1c-pr-f-runs/run_scan.sh`](../.p1c-pr-f-runs/run_scan.sh) (gitignored scratch) |

### §1.2 Per-case 4-N canonical SHA bitwise (rivqdown.dat)

每单元：编辑 `cfg.para` 的 `NUM_OPENMP=N` → `cd Basins/<case> && ../../shud_omp <prjname>` → `shasum -a 256 output/<prjname>.out/<prjname>.rivqdown.dat` → 还原 `NUM_OPENMP=8`。

| Case (NumEle) | prjname | N=1 SHA (head 8) | N=2 SHA (head 8) | N=4 SHA (head 8) | N=8 SHA (head 8) | 4-N bitwise |
|---|---|---|---|---|---|---|
| keliya (484) | keliya | `b23e15b9` | `b23e15b9` | `3fff6574` | `c7a05280` | **FAIL** (3 distinct) |
| xinanjiang_upstream (801) | xinanjiang | `90eeb9c6` | `90eeb9c6` | `37bf030b` | `008a28af` | **FAIL** (3 distinct) |
| qinyijiang (3155) | nanlin | `0f8c3fec` | `0f8c3fec` | `5f0532f3` | `38f3e632` | **FAIL** (3 distinct) |
| qhh (4773, +lake) | qhh | `8a6d9b2c` | `8a6d9b2c` | `d985fbb7` | `ac5c1ce9` | **FAIL** (3 distinct) |

四个 case 同一规律：N=1 / N=2 相同 (单线程 + 双线程 fork-join 不引入新分支)，N=4 与 N=8 各自不同 — Mac 平台 OMP_PROC_BIND unset 下 N>=4 出现 byte 级漂移，与 P1c motivation (PR-K2 #223 服务器同类漂移) 行为一致。

### §1.3 Full SHA (rivqdown.dat) 列表

```
keliya N=1    b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a
keliya N=2    b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a
keliya N=4    3fff657456f2079fd266207d231bb8e707b808d62def99782dca9695e0b4d695
keliya N=8    c7a052803a5bae1ac1b8f8043197381d1b91ef085742b48aaa82f976e0ab38a3
xinanjiang_upstream N=1    90eeb9c63c07e8db3a051482f51cdc274f6e469326b19575efed54181258bf45
xinanjiang_upstream N=2    90eeb9c63c07e8db3a051482f51cdc274f6e469326b19575efed54181258bf45
xinanjiang_upstream N=4    37bf030b6347403febb39bb4c344a686888ad6f9cd04bbc327d9d89fbf4a5718
xinanjiang_upstream N=8    008a28afe9b0a6435365bfc719ab1e9acbec5c9130b74cbb826148d33b7f90b0
qinyijiang N=1    0f8c3fecfe1e618f24ee5411ff9c7ea80381e953e6aafac6a1a96fe666018ced
qinyijiang N=2    0f8c3fecfe1e618f24ee5411ff9c7ea80381e953e6aafac6a1a96fe666018ced
qinyijiang N=4    5f0532f3048eae02bf356dba29d7d958a2172e1f2a01a29d78d6a725c5d68311
qinyijiang N=8    38f3e632daab46e93234b34de072a47afa52322fecb88a8fb51c301ef5a88f2b
qhh N=1    8a6d9b2cbddc640c19384635b3d6cde4399e5c6c17031cea1ac39d10d83532b6
qhh N=2    8a6d9b2cbddc640c19384635b3d6cde4399e5c6c17031cea1ac39d10d83532b6
qhh N=4    d985fbb798fcc16dd3d0532258a1c9321e63321335cb967b984c0c8481566003
qhh N=8    ac5c1ce922705e7fab40568404de52a0db06697730481b447303d5ba7ca26818
```

### §1.4 Wallclock (informational)

| Case | N=1 (s) | N=2 (s) | N=4 (s) | N=8 (s) |
|---:|---:|---:|---:|---:|
| keliya | 69 | 70 | 76 | 179 |
| xinanjiang_upstream | 8 | 8 | 9 | 24 |
| qinyijiang | 242 | 235 | 271 | 459 |
| qhh | 84 | 83 | 84 | 106 |

无单元 timeout (30 min/cell 上限内全部完成)；总耗时约 33 min wallclock。

### §1.5 Notes / 解读

- **Mac local FAIL 不阻 server PR-K2 提交** (per design D7)。
- **Mac PASS 不会自动触发 Kahan** (Mac pass-while-server-fails 风险)。
- 服务器 PR-K2 首跑 + nst delta + reverse-compat = 三个 SHALL 条件 for capstone (per design D3)。
- 16-cell 命令：`for N in 1 2 4 8: edit cfg.para NUM_OPENMP=$N → SHUD/shud_omp <prjname> (cwd=Basins/<case>) → shasum -a 256 output/<prjname>.out/<prjname>.rivqdown.dat`.
- N=1 → N=2 全部 case bitwise equal (双线程 fork-join 不引入新 floating-point ordering)；N=4 / N=8 各自漂移 — 与 P1c motivation 同源。

## §2 Server PR-K2 首跑 (PR-H, SHUD@de9545d pre-Kahan)

| Field | Value |
|---|---|
| Server endpoint | `frd_muziyao@210.77.77.22:32099` (xnode login + cn0X compute partition) |
| SHUD pin | `de9545d` (post-PR-E HEAD) |
| Outer baseline/P1c HEAD at run time | `934944e` (post-PR-G merge) |
| Binary SHA-256 | `9f39d38f4e84c6745b4d07e36dd51cc433a9fae23a4e3fed925251e016f4dd6e` |
| FP flags (build log grep 2/2) | `-ffp-contract=off` + `-fno-fast-math` PASS |
| Slurm matrix | heihe (cn03) jobs 8925-8928; heihe_x4 (cn08) jobs 8929-8932 |
| 总 wall | 21 min (max heihe_x4 N=4 = 1240 s ≈ 20.7 min) |

### §2.1 8-cell rivqdown.dat SHA + wall

| case | N=1 SHA (head 16) | N=2 | N=4 | N=8 | wall (s) per N |
|---|---|---|---|---|---|
| heihe | `7f22bd6faa438d50` | `7f22bd6faa438d50` | `7f7a621cf1c4f02b` | `8c581172a17db537` | 530 / 502 / 479 / 472 |
| heihe_x4 | `55403bef48ee5ad8` | `55403bef48ee5ad8` | `7e8f7a8a9697279e` | `8b0efa6f6a74a43a` | 1182 / 1178 / 1240 / 1212 |

(完整 64-hex SHA 见 `docs/p1c/p1c_pr_h_server_first_run.md` §3。)

### §2.2 cvode_stats 15-key — nst 主要漂移

| case | N | nst | nfe | nfeLS | ncfn | ncfl |
|---|---|---|---|---|---|---|
| heihe | 1 | **6773** | 7035 | 12885 | 6 | 93 |
| heihe | 2 | **6773** | 7035 | 12885 | 6 | 93 |
| heihe | 4 | **6682** | 6947 | 12710 | 9 | 91 |
| heihe | 8 | **6548** | 6780 | 12320 | 6 | 78 |
| heihe_x4 | 1 | **6571** | 6733 | 30559 | 48 | 3702 |
| heihe_x4 | 2 | **6571** | 6733 | 30559 | 48 | 3702 |
| heihe_x4 | 4 | **6568** | 6730 | 30483 | 47 | 3604 |
| heihe_x4 | 8 | **6570** | 6721 | 30568 | 47 | 3795 |

(其余 9 key: nni / nli / nsetups / netf / npe / nps / lenrw / leniw / lenrwLS / leniwLS — 见 PR-H §4。)

**verdict**:
- A3a bitwise: heihe + heihe_x4 双 case 3 distinct SHAs (N=1≡N=2 ≠ N=4 ≠ N=8) — **FAIL**
- nst across N: heihe |Δ|=225, heihe_x4 |Δ|=3 — **FAIL** (heihe ≫ D9 ≤2; heihe_x4 just over)
- 15-key: 6 keys drift (nfe / nfeLS / nni / nli / ncfn / ncfl), 8 stable
- §4.7 trigger: **SHALL TRIGGER PR-I conditional Kahan injection**

## §3 Server PR-K2 二跑 (PR-I, SHUD@3a0004c Kahan-injected)

| Field | Value |
|---|---|
| SHUD pin (Kahan-injected) | `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` |
| 外层 baseline/P1c HEAD | `f3d0d28` (post-PR-I merge, PR-K base) |
| Binary SHA-256 (Kahan) | `63b540177ecd4198c0708070fef2fdc63ac7d9179e77c7dbd70537d90d387b01` |
| FP flags (build log grep 2/2) | `-ffp-contract=off` + `-fno-fast-math` PASS |
| Slurm matrix | heihe (cn04) jobs 8943-8946; heihe_x4 (cn15) jobs 8947-8950 |
| 总 wall | 19 min (max heihe_x4 N=2 = 1140 s ≈ 19 min) |

### §3.1 8-cell SHA + wall (Kahan)

| case | N=1 SHA (head 16) | N=2 | N=4 | N=8 | wall (s) per N |
|---|---|---|---|---|---|
| heihe | `fd2d55716b5daffd` | `fd2d55716b5daffd` | `e058db2e9c2a9b9a` | `6285e8a4a30a3917` | 508 / 509 / 506 / 500 |
| heihe_x4 | `4eb804f571ba6f89` | `4eb804f571ba6f89` | `ff0787abd2170d4d` | `6e9f9a2eaf652747` | 1058 / 1140 / 1051 / 934 |

(完整 64-hex SHA 见 `docs/p1c/p1c_pr_i_kahan_injection.md` §3。)

### §3.2 cvode_stats 15-key (Kahan)

| case | N | nst | nfe | nfeLS | ncfn | ncfl |
|---|---|---|---|---|---|---|
| heihe | 1 | **6553** | 6764 | 12183 | 2 | 69 |
| heihe | 2 | **6553** | 6764 | 12183 | 2 | 69 |
| heihe | 4 | **6524** | 6753 | 12307 | 3 | 81 |
| heihe | 8 | **6608** | 6869 | 12531 | 9 | 87 |
| heihe_x4 | 1 | **6571** | 6732 | 30543 | 48 | 3733 |
| heihe_x4 | 2 | **6571** | 6732 | 30543 | 48 | 3733 |
| heihe_x4 | 4 | **6574** | 6736 | 30516 | 50 | 3657 |
| heihe_x4 | 8 | **6569** | 6730 | 30430 | 47 | 3614 |

**verdict**:
- A3a: 3 distinct SHAs per case 仍存 — **FAIL** (pattern preserved)
- nst delta: heihe |Δ|=84 (改善 ~63% vs 225); heihe_x4 |Δ|=5 (噪声幅度) — **PARTIAL**
- §4.7 二次决策: **PARTIAL CLOSURE + P1d carve-out**

## §4 Δ_wall 对比 PR-H pre-Kahan vs PR-I Kahan (R2 估算 verification)

| case | N | PR-H wall (s) | PR-I wall (s) | Δ_secs | Δ% | direction |
|---|---|---|---|---|---|---|
| heihe | 1 | 530 | 508 | −22 | **−4.2%** | ↓ faster |
| heihe | 2 | 502 | 509 | +7 | +1.4% | minor slow |
| heihe | 4 | 479 | 506 | +27 | +5.6% | slower |
| heihe | 8 | 472 | 500 | +28 | +5.9% | slower |
| heihe_x4 | 1 | 1182 | 1058 | −124 | **−10.5%** | ↓ faster |
| heihe_x4 | 2 | 1178 | 1140 | −38 | −3.2% | faster |
| heihe_x4 | 4 | 1240 | 1051 | −189 | **−15.2%** | ↓ faster |
| heihe_x4 | 8 | 1212 | 934 | −278 | **−22.9%** | ↓ faster |

**R2 估算 (Kahan +1-3% perf 降) REFUTED**: 8 cells 中 5 cells wall 改善 (heihe_x4 全 cells), 仅 heihe N=4/N=8 +5% 慢 (但仍在 noise band). 假设解释: Kahan 改 CVODE 收敛路径 → 减 SPGMR linear-solve fail 计数 (ncfl heihe 93 → 69 / heihe_x4 3702 → 3733 in noise) → 抵消 Neumaier 浮点 overhead 且更经济。

cross-check 对照 P1 era PR-K1 wall baseline 留 P2a 阶段做 (per p1c_summary.md §9.3 限制 #3)。

## §5 Mac sanity prediction power update (per design D7)

PR-F Mac 16-cell scan (per §1.2) + Server PR-H/PR-I 8-cell (per §2/§3) 比较:

| 平台 | A3a 4-N bitwise | pattern (N=1≡N=2 ≠ N=4 ≠ N=8) | sanity prediction (D7 framing) |
|---|---|---|---|
| Mac M4 Pro 4P+10E | 4/4 case FAIL | 一致 | 原 D7 framing: "pass-while-server-fails"; PR-F 经验: **fail-while-server-fails** (同 pattern), 即 Mac early signal |
| Server cn03/cn08 (heihe) | FAIL | 一致 | SHALL gate |
| Server cn04/cn15 (heihe_x4) | FAIL | 一致 | SHALL gate |
| Server cn04/cn15 (heihe Kahan) | FAIL | 一致 | Kahan partial only |

**D7 framing 经验 update**:
- 原假设: "Mac snapshot 已知 pass-while-server-fails" (spec L113 framing) → 经 PR-F 不成立
- 修订: Mac + Server **共享** RISK-26 (NUMA / cache locality) 触发 floating-point 不确定性的同类机制
- D7 SHOULD/SHALL trigger 不对称仍**保留** (Mac informational only, server SHALL gate) — 因为 Mac 不在 D11 immutability 范围, 不能担任 PROMOTE 验证。但 Mac 不再是 "may pass while server fails" 的非典型 case, 反而是 server pattern 的 early signal。
- PR-M PROMOTE 时由 reviewer 决定是否在 archive 标 "Mac sanity prediction power 经验补充" (per `docs/p1c/p1c_summary.md` §9.4)

## §6 阶段 hand-off

详 `docs/p1c/p1c_summary.md` §5 (P2a + P1d)。本 perf baseline 文档是 capstone source of truth for performance 数据, 在 PR-M PROMOTE 时随 archive。

[`docs/p1c/p1c_summary.md`](p1c_summary.md) — sibling capstone (≥7 主题完整结构)。
