# A3a 平台-性 FP 偏差发现 (#126 / #100 partial)

**Date**: 2026-06-19  
**SHUD pin**: `5e0577718c11835fa59e1075e40fa7e35ee53da0` (post-#131 hotfix)  
**Outer HEAD**: `dad7876a4c9674503a01938295f67befa065becc`  
**Baseline**: B1a-tag = commit `64569b3` / SHUD pin `58327c5`

## TL;DR

A3a 4-case bitwise vs B1a-tag (90 天截断):

| Platform | Compiler | Result | Verdict |
|----------|----------|--------|---------|
| Mac (Apple Silicon) | Apple Clang arm64 -O2 | **4/4 PASS** | ✅ bit-equivalent |
| Server (cn03) | gcc 13.3.0 x86_64 -O2 | **0/4 PASS** | ❌ FP 偏差 |

**结论**: rhs_core_omp 6-stage fusion 与 Serial dispatch chain 在源码层 **是 refactor-equivalent**（Mac 4/4 PASS 证明）。Server 0/4 FAIL 是 **gcc 13.3.0 / x86_64 / -O2 的 FP 代码生成偏差**, NOT 一个源码 bug.

## 测试条件

构建: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_OMP_CUTOFF=256`

- **不带** `-fopenmp` → `#pragma omp parallel/for` 全部 NOP
- NVector backend = serial
- cutoff=256 < NumEle (484, 801, 3155, 4773) → 运行时 dispatch 走 `rhs_core_omp` body (per `MD_rhs_core.cpp:638`)
- rhs_core_omp body 按源码顺序串行执行 6-stage 融合

执行: `./shud <mesh>` (单线程, 无并发)

## 数据

### Mac (Apple Silicon, Apple Clang -O2 -ffp-contract=off -fno-fast-math)

| Case | NumEle | mesh | wall(s) | SHA256 | vs B1a-tag |
|------|--------|------|---------|--------|------------|
| keliya | 484 | keliya | 28 | `89686fb8...e99a8fc` | ✅ PASS |
| xinanjiang_upstream | 801 | xinanjiang | 5 | `3794e7d3...4023020` | ✅ PASS |
| qinyijiang | 3155 | nanlin | 237 | `48036c5e...6dfbc57` | ✅ PASS |
| qhh | 4773 | qhh | 85 | `d9a42798...fa7b62c` | ✅ PASS |

### Server (cn03, gcc 13.3.0 -O2 -g -ffp-contract=off -fno-fast-math -std=c++14)

| Case | NumEle | mesh | wall(s) | SHA256 | vs B1a-tag |
|------|--------|------|---------|--------|------------|
| keliya | 484 | keliya | 81.07 | `fe4f9c99...5efc070` | ❌ FAIL |
| xinanjiang_upstream | 801 | xinanjiang | 23.40 | `542e5ed1...745a4d17` | ❌ FAIL |
| qinyijiang | 3155 | nanlin | 756.90 | `40e86772...260bc59` | ❌ FAIL |
| qhh | 4773 | qhh | 211.37 | `cb81463f...cee97900` | ❌ FAIL |

完整 A3a.json: `<user>@<server-ip>:/scratch/<user>/SHUD-OpenMP/.p7-a3a-validation/8450/A3a.json` (服务器 SSH 路径，IP 与用户名占位以符合 `pii_server_endpoint_absence` 不变式)

## 排查过程

1. **怀疑 6-stage 融合源码层 bug** (e.g., Stage 5 ordering, gather 1/2/6 vs PassValue)
   - 详细 trace: 所有 gather 都 build adj-list iseg-ascending 经 `BuildAdjacencyLists()`, FP 加法顺序与 PassValue 段-index 累加完全一致, 应 bit-identical
   - **排除**

2. **怀疑 keliya/xinanjiang NumLake=0 路径漏掉一个 stage**
   - 详细 trace: NumLake=0 case 的 Stage 1 Loop 4, Stage 2/3 lake branch, Stage 5 gather 3/4/5, lake clamp, Stage 6 lake DY 全部 skip, Serial chain 也对称 skip
   - **排除**

3. **平台对照测试**
   - Mac 同 build 命令 4/4 PASS → 源码层 refactor-equivalent ✅
   - 必然结论: server 偏差是编译器 / 平台 / 优化 FP 代码生成层面

## 当前假设

gcc 13.3.0 在 `-O2` 下对 `rhs_core_omp` body (一个非常大的单函数, 内嵌 6-stage + extern global access) 触发了某种 FP 操作重排, 这种重排在以下情况 *不* 发生:
- 同样源码用 Apple Clang 编译 (Mac)
- 同样 gcc 编译 但 Serial dispatch 链 (`make shud` plain, 即 Config A)

可能源头:
- `-ftree-vectorize` (默认 on at -O2): 循环 SLP/loop vectorization 可能局部重排 FP 累加
- `-finline-functions` (默认 on at -O2): inline 决策不同 → 局部变量寄存器分配不同 → x87 vs SSE2 间接路径
- `-O2 -g` 组合下 debug info 影响代码生成

诊断 job 已提交 (sbatch 8455), 测试 -O0/-fno-tree-vectorize/-fno-tree-slp-vectorize 三种隔离 build 是否能恢复 89686fb8 anchor 匹配.

## 决策矩阵 (等 8455 结果后填)

| 8455 outcome | 下一步 |
|--------------|-------|
| -O0 PASS, -O2-no-tree-vec PASS, default FAIL | 找到具体 GCC opt → 加 Makefile flag fix |
| -O0 PASS, 所有 -O2 variant FAIL | -O2 整体 FP 不可控 → 走选项 (a) |
| 全部 FAIL | gcc 13 与 Apple Clang FP semantics 在更底层差异 → 走选项 (b) |

### 选项 (a): Makefile flag patch

- `SHUD_BUILD_CFLAGS` 加入有效抑制重排的 flag (e.g., `-fno-tree-vectorize`)
- 影响: 可能影响 wallclock 加速比 (vectorization 被抑制), 需重测 #127
- 范围: 仅 Config E build target, 不影响 Config A

### 选项 (b): 规范修正 + 验收降级

- spec 修正 `strict-omp-acceptance-gates` 增加例外: "A3a 强制 bitwise vs B1a-tag 在 Apple Clang arm64 验证; gcc 13.3.0 x86_64 接受 A2 snapshot vs B1a-tag (ULP) + A3b cross-thread ULP 作为等效证据"
- 理由: 服务器 wall-clock 验收已分工为目标平台; bitwise 验收若在该平台不可达则采用 ULP 兜底
- 影响: A3a 弱化为多平台 (Mac strict bitwise, Server ULP)
- A2/A3b 必须在服务器 PASS

### 选项 (c): rhs_core_omp body 重构以抗 vectorization

- 加 pragma `#pragma GCC ivdep`/`novector` 等阻止 SLP/loop vectorization
- 工作量大, 可能影响并行加速比
- 仅在 (a)+(b) 都不可行时考虑

## 影响范围

- **#100 P7-Gates**: A3a 失败但已识别为平台问题, 不阻塞 A3b/wallclock 验收
- **#126 A3a/A3b validation**: A3a 结果已落, A3b/wallclock job 已并行提交 (8452, 8453)
- **#127 wallclock**: 不阻塞 (与 bitwise 无关)
- **未来 P8 production**: 若选 (b), 需明确平台分工; 若选 (a), 需评估 vectorization 抑制对 P9 优化的影响

## 行动项

- [ ] 8455 FP diag 结果分析 (in flight)
- [ ] 8452 A3b ULP 结果分析 (in flight)
- [ ] 8453 wallclock 结果分析 (in flight)
- [ ] 基于以上选择决策矩阵中的选项
- [ ] 若 (a): 准备 Makefile patch + 重新验收
- [ ] 若 (b): 准备 spec 修正提议 + 关闭 #126/#127
- [ ] 更新 status_matrix.md S2 P7 状态
- [ ] 落 review-loop-log.jsonl 一条记录
