## P7-Gates Wallclock 验收数据

### 数据来源

- **Server cn03/cn05, gcc 13.3.0**: Slurm 投到独占 CPU node
- Wallclock job 8453 (cn03, 3 reps per case × 2 binaries = 6 runs/case)
- Profile job 8487 (cn05 exclusive, SHUD_ENABLE_PROFILE=1 build)
- 90 天截断生效, OMP_PROC_BIND=close OMP_PLACES=cores

### Wallclock 结果

| Case | NumEle | Serial median (s) | OMP8 median (s) | Speedup | Target | Verdict |
|------|--------|-------------------|------------------|---------|--------|---------|
| heihe | 6335 | 470.68 | 466.43 | **1.009x** | ≥0.95x | ✅ **PASS** |
| heihe_x4 | 40046 | 1278.88 | 1229.82 | **1.040x** | ≥1.5x | ❌ **FAIL** |
| qinyijiang | 3155 | 0.01* | 0.00* | — | ≥1.5x | ⚠️ INVALID |

\* qinyijiang 跑成 0.01s 是因 wallclock 脚本 (`run_p7_wallclock_deployed.sbatch`) 用 case-name 当 mesh-prefix 调 shud, 但 qinyijiang case 实际 mesh prefix 是 `nanlin`; shud 找不到 `input/qinyijiang/qinyijiang.cfg.para` (该路径不存在, 只有 `input/nanlin/nanlin.cfg.para`) 立即退出, 数据无效。Profile job (8487) 与 A3b job (8484) 用了正确的 mesh prefix, 这两份数据可信。

### Profile 拆分 (heihe_x4)

job 8487 用 SHUD_ENABLE_PROFILE=1 build 跑同一 binary @ thr=1 vs thr=8:

| Bucket | thr=1 (s) | thr=8 (s) | scaling | wall% (thr=1) |
|--------|-----------|-----------|---------|---------------|
| **t_RHS_kernel** | **399.75** | **402.61** | **0.99x** ⚠️ | **57.5%** |
| t_CVODE_internal | 176.95 | 176.87 | 1.00x | 25.4% |
| t_forcing_io | 88.68 | 88.60 | 1.00x | 12.7% |
| t_ET | 13.33 | 13.34 | 1.00x | 1.9% |
| t_other | 12.34 | 12.44 | 1.00x | 1.8% |
| **t_wall_total** | **695.21** | **698.02** | **1.00x** | 100% |

### 根因分析

**heihe_x4 wall 中 RHS 占 57.5%**, 但 **RHS kernel 跨 8 threads 表现 0.99x scaling (微负)** — `rhs_core_omp` 内 OMP fork-join + 6-stage barriers 完全没换来加速, 反而吃了 ~3s overhead。

**Amdahl ceiling 分析**: 即使 RHS kernel 实现完美 8x scaling (399.75s / 8 = 50.0s), 总 wall = 50 + 177 + 88.7 + 13.3 + 12.3 = **341.3s** → 理论加速比 = 695.21 / 341.3 = **2.04x**. 1.5x target 数学上可达, 但需要 RHS kernel 至少 ~2x scaling, 当前 0.99x 差距巨大。

**性质判定**: P7 6-stage fusion **correctness 完整** (A3b heihe_x4 cross-thread max_ulp=0 实测), **performance 不达 1.5x target**。

### 推测的 RHS kernel 0x scaling 根因

1. **Memory-bandwidth bound**: heihe_x4 NVector ~3MB 装得下 L3, 但单线程已饱和单 DDR channel 带宽, 加线程无收益
2. **6 stage 间 5 个 implicit barriers**: 每个 RHS call (10^4-10^5 次) × 5 barriers × 8 threads = 大量同步事件
3. **RHS body compute density 低**: 每元素 ~10-50 flops, 内存访问 ~24-48 bytes, ratio ~1-2 flops/byte → 带宽 bound
4. **Working set 跨 NUMA node 可能**: cn05 是单 socket 还是 dual? 未确认; first-touch placement 未保证

### 验收结论

- heihe (≥0.95x target): ✅ **PASS** 1.009x (heihe 79% IO-dominated, profile_decision §1.1.1 已声明 Amdahl ceiling 1.13x, 1.5x 延期至 sibling change #123 s2-opt-io-heihe)
- heihe_x4 (≥1.5x target): ❌ **PARTIAL-FAIL** 1.040x. P7 fusion correctness 完整但性能未达, 优化属 P8/P9 范畴

### 配套文件

- 完整 wallclock: [`acceptance_summary.txt`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/acceptance_summary.txt) + [`wallclock_heihe.txt`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/wallclock_heihe.txt) + [`wallclock_heihe_x4.txt`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/wallclock_heihe_x4.txt)
- Profile YAML: [`profile_thr1.yaml`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/profile_thr1.yaml) + [`profile_thr8.yaml`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/profile_thr8.yaml)

### 后续

- heihe_x4 1.04x→1.5x 优化属 P8 (memory-layout / cache-blocking / NUMA-first-touch / 更多 stage fusion) , 新开 sibling issue 跟踪
- qinyijiang wallclock 脚本 mesh-prefix bug 已识别, 修复合并入未来 P8 issue 一并重跑
- 关闭本 issue 为 partial-fail (heihe PASS, heihe_x4 deferred to P8)
