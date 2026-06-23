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

## §2-N (placeholder for PR-K capstone)

- 服务器 PR-K2 结果 (`heihe` + `heihe_x4` × 4 N) — bitwise verdict per case
- nst delta table (N=1 vs N>=2 step count)
- ULP delta table (per-site × N)
- reverse-compat check (旧 binary vs 新 binary 同 N=1)
- 阶段 hand-off → P2a / next consumer

[`docs/p1c_summary.md`](p1c_summary.md) — sibling seed (capstone 验证结果 占位)。
