## P7-Gates 综合验收 — Partial PASS

### 子 issue 状态

| Sub-issue | 类型 | Verdict | 数据 |
|-----------|------|---------|------|
| #126 | A3a + A3b | ✅ **PASS** (Mac A3a 4/4 + Server A3b max_ulp=0×2) | [#126 comment](https://github.com/DankerMu/SHUD-OpenMP/issues/126) |
| #127 | Wall-clock | ⚠️ **PARTIAL** (heihe PASS, heihe_x4 1.04x defer P8) | [#127 comment](https://github.com/DankerMu/SHUD-OpenMP/issues/127) |

### 验收 4 个 P7 gate 状态总结

1. **grep gate** (单 parallel region in rhs_core_omp): ✅ 已在 #99 PR 验证, CI invariant-sweep job 持续守门
2. **invariants** (P5 owner-local + adjacency, P7 cutoff, OMP_NUM_THREADS env, ExecPolicy dispatch): ✅ 各 sub-issue PR 已验证, CI invariant-sweep 持续守门
3. **cutoff boundary** (Mac 3-cutoff self-consistency, Server cutoff fallback validation): ✅ 在 #97 PR 验证
4. **A3a + A3b + wallclock**: 上表 #126/#127 已覆盖

### 核心结论

**P7 strict-OMP 阶段 correctness 完整**, 由以下硬证据支撑:
- A3b qinyijiang (NumEle=3155): cross-thread max_ulp=0 / 29154 elements
- A3b heihe_x4 (NumEle=40046): cross-thread max_ulp=0 / 387607 elements
- 这两份数据证明 P5 owner-local gather + P7 6-stage fusion 在 8 threads 下零数据竞争, FP 操作顺序 thread-deterministic, 达到 A3c (cross-thread bitwise) 加分项

**P7 性能未达 heihe_x4 ≥1.5x target** (实测 1.04x), profile (#127 评论) 证实 RHS kernel 跨 8 threads scaling 0.99x — `rhs_core_omp` body memory-bandwidth bound, 优化属 P8 范畴。

**A3a anchor 平台问题**: B0/B1a anchor 是 Mac (Apple Clang arm64) 生成, server (gcc 13.3.0 x86_64) 任何 build (Config A / Config E / 各 -O variants) 都不 bit-match anchor — anchor 平台错配, 不是 rhs_core_omp bug (#126 评论的 8455 FP 诊断 6 build variants 全输出同 SHA `fe4f9c99...`, Config A baseline 都不 match anchor)。Mac A3a 4/4 PASS 反证源码层 refactor-equivalence。

### S2 strict-OMP 阶段闭合

P7 fusion ship as-is, S2 strict-OMP 阶段在 correctness 维度达到 strict-omp-acceptance-gates spec 要求。

P7→P8 转入触发条件:
- ✅ A3a (correctness, Mac 4/4 + Server cross-thread max_ulp=0 兜底)
- ✅ A3b (强制 ULP ≤ 4, 实测 max_ulp=0)
- ✅ A3c 加分 (cross-thread bitwise, 实测 max_ulp=0 = bitwise)
- ✅ wallclock heihe ≥ 0.95x (实测 1.009x)
- ⚠️ wallclock heihe_x4 ≥ 1.5x → 转 P8 sibling backlog (memory-layout 优化)

### P8 backlog 起点

P8 sibling change 已开新 issue [#133](https://github.com/DankerMu/SHUD-OpenMP/issues/133) (`[S2 P8-sibling] heihe_x4 wall-clock 1.04x→1.5x via memory-layout`):
- profile 数据: t_RHS_kernel = 57.5% wall, scaling 0.99x → 改 SoA + cache blocking + NUMA first-touch
- Amdahl ceiling: 2.04x 理论 (RHS 完美 8x), 1.5x target 数学可达
- 候选: SoA layout 抽取热字段 / 更大 stage fusion 减 barriers / `#pragma omp simd` 显式向量化 / NUMA first-touch policy

### 关闭

P7 gates 已交付, 关闭本 epic-level issue。
