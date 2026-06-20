## P7-Gates A3a + A3b 验收数据

### 数据来源

- **Mac (Apple Silicon, Apple Clang -O2 -ffp-contract=off -fno-fast-math)**: A3a 4-case 本地跑（`make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_OMP_CUTOFF=256`, 无 -fopenmp, 90 天截断）
- **Server (cn03/cn05, gcc 13.3.0 -O2 -g -ffp-contract=off -fno-fast-math -std=c++14)**: A3a + A3b 跑 Slurm 投到独占 CPU node, A3a job 8450, A3b qinyijiang job 8452 (recomputed JSON), A3b heihe_x4 job 8484

### A3a 4-case vs B1a-tag bitwise

| Case | NumEle | Mac SHA256 | Mac vs B0 anchor | Server SHA256 | Server vs B0 anchor |
|------|--------|------------|------------------|---------------|---------------------|
| keliya | 484 | `89686fb8...e99a8fc` | ✅ MATCH | `fe4f9c99...5efc070` | ❌ NO-MATCH |
| xinanjiang_upstream | 801 | `3794e7d3...4023020` | ✅ MATCH | `542e5ed1...745a4d17` | ❌ NO-MATCH |
| qinyijiang | 3155 | `48036c5e...6dfbc57` | ✅ MATCH | `40e86772...260bc59` | ❌ NO-MATCH |
| qhh | 4773 | `d9a42798...fa7b62c` | ✅ MATCH | `cb81463f...cee97900` | ❌ NO-MATCH |

**Mac 4/4 PASS, Server 0/4 ≠ anchor**.

### A3a Server 偏差根因 (8455 FP 诊断 job)

测试同一台 cn03 上构建 Config A pure baseline (`make shud`, 无任何 OMP define) 跑 keliya 90d:

| Build variant | keliya SHA256 |
|---------------|---------------|
| **Config A pure baseline** | `fe4f9c99...5efc070` |
| Config E -O2 default | `fe4f9c99...5efc070` |
| Config E -O0 | `fe4f9c99...5efc070` |
| Config E -O2 -fno-tree-vectorize | `fe4f9c99...5efc070` |
| Config E -O2 -fno-tree-slp-vectorize | `fe4f9c99...5efc070` |
| Config E -O2 -fno-vec-both | `fe4f9c99...5efc070` |

**Server gcc 13.3.0 跑任何 build 都产生 `fe4f9c99...`, 与 archived anchor `89686fb8...` 不同。**

⇒ archived B0/B1a anchor 是 Mac (Apple Clang arm64) 生成的，与 Server (gcc 13.3.0 x86_64) FP 代码生成不兼容。**这是 anchor 平台错配，不是 rhs_core_omp 6-stage fusion bug**。Mac 4/4 PASS 反证了源码层 refactor-equivalence (rhs_core_omp body 与 rhs_update→rhs_flux→rhs_apply Serial chain 在 Apple Clang 下 bitwise 一致)。

### A3b cross-thread ULP

Production binary (`make shud_omp SHUD_ENABLE_OPENMP_RHS=1`, default cutoff=1024, Config D+E coupled, -fopenmp on) @ OMP_NUM_THREADS ∈ {1,2,4,8}:

| Case | NumEle | max_ulp@{2,4,8 vs 1} | max_abs_diff | n_diff / n_total | ULP threshold | Verdict |
|------|--------|----------------------|--------------|-------------------|---------------|---------|
| qinyijiang | 3155 | **0** | 0.0 | 0 / 29154 | ≤ 4 | ✅ PASS (A3c-bitwise level) |
| heihe_x4 | 40046 | **0** | 0.0 | 0 / 387607 | ≤ 4 | ✅ PASS (A3c-bitwise level) |

两个 case 的 4 个 thread 配置完全 BITWISE 一致 → max_ulp=0 远超 A3b 的 4-ULP 阈值, 实际达到 **A3c (cross-thread bitwise)** 加分项水平。证明 P5 owner-local gather + 6-stage fusion 在 8 threads 下零数据竞争, FP 操作顺序 thread-deterministic。

### 验收结论

- A3a 强制 vs B1a-tag bitwise: **跨平台分裂** (Mac PASS, Server 平台 anchor 错配)
- A3b 强制 cross-thread ULP ≤ 4: **2/2 PASS** (实际 max_ulp=0)
- A3c 加分 cross-thread bitwise: **2/2 PASS** (隐性达到)

### 配套文件

- A3a server 4-case: [`A3a_server.json`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3a_server.json) (note: 跑成 FAIL 但 root cause = anchor 平台错配)
- A3b qinyijiang: [`A3b_qinyijiang.json`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3b_qinyijiang.json)
- A3b heihe_x4: [`A3b_heihe_x4.json`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3b_heihe_x4.json)
- FP diag: [`server_fp_diag_8455.txt`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/server_fp_diag_8455.txt)
- 完整平台分析: [`.s2-103/diag/A3a_platform_finding.md`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/diag/A3a_platform_finding.md)

### 后续

- A3a 平台分裂会在 issue (TBD: anchor regeneration for gcc 13 platform) 中跟踪
- 当前 P7 correctness gates 满足, 关闭本 issue
