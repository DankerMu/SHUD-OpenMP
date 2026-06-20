## 背景

#126 P7-Gates 验收暴露 **archived B0/B1a anchor 跨平台不可复现**:

- archived `benchmarks/<case>/B0_output/<case>.rivqdown.dat` SHA256 `89686fb8...` (keliya) / `3794e7d3...` (xinanjiang_upstream) / `48036c5e...` (qinyijiang) / `d9a42798...` (qhh) 是 **Mac (Apple Silicon, Apple Clang arm64) 生成**
- Server (gcc 13.3.0 x86_64) 任何 build 都不 bit-match anchor — 8455 FP diag 已实测 6 build variants 全输出同 SHA `fe4f9c99...` (keliya 90d), 包括 Config A pure baseline (`make shud` 无任何 OMP define)
- ⇒ 不是 rhs_core_omp 6-stage fusion bug; Mac A3a 4/4 PASS 反证源码层 refactor-equivalence
- ⇒ A3a vs B1a-tag bitwise gate 在 server gcc 13.3.0 平台不可达

详见 [#126 评论](https://github.com/DankerMu/SHUD-OpenMP/issues/126#issuecomment-4757292206) + `.s2-103/diag/A3a_platform_finding.md` + `.s2-103/p7-acceptance/server_fp_diag_8455.txt`。

## 范围

在 server (cn03/cn05) 用 Config A pure baseline (`make shud`) 跑 4 case 90 天截断, 生成 `<case>.rivqdown.dat.sha256.gcc13.x86_64` 平台分支 anchor:

| Case | 当前 archived (Mac) anchor | 待生成 (gcc13.x86_64) anchor |
|------|----------------------------|------------------------------|
| keliya | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | (待跑, 预计 `fe4f9c99...` per 8455) |
| xinanjiang_upstream | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | (待跑, 预计 `542e5ed1...` per 8450) |
| qinyijiang | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | (待跑, 预计 `40e86772...` per 8450) |
| qhh | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | (待跑, 预计 `cb81463f...` per 8450) |

### 工作清单

1. **服务器跑 Slurm**: `make shud` Config A pure × 4 case × 3 reps (repeatability 验证) — 90 天截断, 单 thread
2. **archive layout**: `benchmarks/<case>/B0_output/` 下加平台分支 sha256 文件 (e.g., `keliya.rivqdown.dat.sha256.gcc13.x86_64`); 现有 `keliya.rivqdown.dat` (Mac-bin) 重命名为 `keliya.rivqdown.dat.apple-clang.arm64` 或加 `Mac/` 子目录; gcc13.x86_64 dat 入 `gcc13/` 子目录
3. **CI matrix 改造**: invariant-sweep / strict-omp-acceptance-gates 按 runner OS+compiler 切换 anchor 选择 (e.g., `if [[ "$(uname)" == "Linux" && "$(g++ -dumpversion | cut -d. -f1)" -ge 13 ]]; then use_gcc13_anchor; fi`)
4. **spec 同步**: `openspec/specs/strict-omp-acceptance-gates/` 加平台-分支 anchor 说明 + `docs/build_manifest.md` IEEE-754 章节加 anchor 不跨编译器/架构可复现的注脚
5. **验收**: 服务器 A3a 4/4 PASS vs `gcc13.x86_64` anchor; Mac A3a 4/4 PASS vs `apple-clang.arm64` anchor

## 验收 gate

- A3a vs B1a-tag bitwise PASS on **both** platforms (Mac 4/4 + Server 4/4 against their respective platform anchors)
- Repeatability: server 3-rep bitwise 自洽
- CI invariant-sweep 在 Linux gcc 13 runner 跑 PASS
- spec `strict-omp-acceptance-gates` 修订 PR 通过 review

## 不在范围

- 跨平台 bitwise 收敛 (e.g., 通过 `-fno-tree-vectorize` 或 cross-compiler flag 让 Mac/Server 输出相同 SHA) — 经 8455 实测 gcc 13 -O0/-O2/-fno-tree-vectorize/-fno-tree-slp-vectorize/no-vec-both **全 6 variants 都不 reproduce Mac anchor**; 跨平台 bit-level FP convergence 不属本 issue 范围, 留 future research
- 重新追踪 B0 anchor 历史变更 (S0 archived 时是 Mac local 跑出的, 此为既成事实)

## 影响 / 阻塞 / 依赖

- 阻塞: server A3a vs B1a-tag bitwise gate (#126 deferral)
- 不阻塞: A3b cross-thread (已 PASS), wallclock (独立)
- 依赖: 无 (独立 sub-issue)

## 配套数据

- 8455 FP diag (6 build variants): `.s2-103/p7-acceptance/server_fp_diag_8455.txt`
- 8450 A3a server 4-case JSON: `.s2-103/p7-acceptance/A3a_server.json`
- 平台诊断报告: `.s2-103/diag/A3a_platform_finding.md`
- Mac A3a 4/4 PASS evidence: 即 archived `benchmarks/<case>/B0_output/<case>.rivqdown.dat` SHA + Mac local 重跑 4/4 实测 ([#126 评论 table](https://github.com/DankerMu/SHUD-OpenMP/issues/126#issuecomment-4757292206))

## Labels

- `sibling`
- `s2-strict`
- `acceptance-gates`
- `priority:p1`
- `runs-on:server`
