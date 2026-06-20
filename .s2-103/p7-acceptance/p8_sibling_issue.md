## P8 sibling: heihe_x4 wall-clock 1.04x→1.5x via memory-layout

### 上下文

S2 P7 strict-OMP 阶段已交付 (#100 关闭), correctness gates 全部满足。**唯一遗留** P7 acceptance 是 heihe_x4 wallclock target ≥1.5x:

- 实测 1.04x (#127 wallclock job 8453: serial 1278.88s, omp8 1229.82s)
- profile (job 8487) 拆分: t_RHS_kernel = 399.75s (thr=1) / 402.61s (thr=8) → **scaling 0.99x**
- t_RHS_kernel 占 wall 57.5% → 是性能瓶颈

### Amdahl ceiling

```
wall(perfect RHS 8x) = (399.75 / 8) + 176.95 + 88.68 + 13.33 + 12.34 = 341.30s
理论加速比 = 695.21 / 341.30 = 2.04x
```

⇒ 1.5x target **数学可达**, 但需 RHS kernel 至少 ~2x scaling, 当前 0.99x 差距巨大。

### 根因推测 (验收 P8 时需确认)

`rhs_core_omp` 6-stage fusion **memory-bandwidth bound**:

1. **NVector 3MB** (387607 elements × 8B), L3 装得下但单线程已饱和单 DDR channel
2. **6 stage × 5 barriers** 同步开销: 10^4-10^5 RHS calls × 5 implicit barriers × 8 threads
3. **RHS body compute/memory ratio 1-2 flops/byte**: 远低于现代 CPU 计算能力, 带宽 bound
4. **NUMA placement 未保证**: cn05 多 socket 时跨 socket 访问开销大

### 优化候选方案

按预期收益排序:

1. **SoA layout 抽取热字段** (`Ele[i].x` → `x[i]`): 减少 cache line 浪费, 提升空间局部性 (预期 1.2-1.5x)
2. **Stage fusion 进一步合并** (减 barriers): 把 owner-local 操作合到同一 `#pragma omp for`, 仅在跨索引依赖处保留 barrier (预期 1.05-1.10x)
3. **NUMA first-touch placement**: 在 init 阶段 first-touch 每个 thread 的 owner-local 内存区间 (预期 1.1-1.3x dual-socket)
4. **`#pragma omp simd` 显式向量化**: 提升 compute density, 缓解带宽压力 (预期 1.1-1.2x)
5. **Cache blocking**: 把大循环拆成 L2/L3 容量的 tile, 多次重用 cache 数据 (预期 1.1-1.3x)

### 验收 gate

- 服务器 heihe_x4 wallclock ≥ 1.5x (gcc 13.3.0 x86_64, 90 天截断)
- 跨 thread bitwise 不退化 (保持 max_ulp=0 across {1,2,4,8})
- 所有现有 invariant CI 保持绿

### 配套数据

- profile thr=1: [`.s2-103/p7-acceptance/profile_thr1.yaml`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/profile_thr1.yaml)
- profile thr=8: [`.s2-103/p7-acceptance/profile_thr8.yaml`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/profile_thr8.yaml)
- wallclock raw: [`.s2-103/p7-acceptance/wallclock_heihe_x4.txt`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/wallclock_heihe_x4.txt)
- A3b cross-thread baseline: [`.s2-103/p7-acceptance/A3b_heihe_x4.json`](https://github.com/DankerMu/SHUD-OpenMP/blob/baseline/current/.s2-103/p7-acceptance/A3b_heihe_x4.json)

### 备注

- 此 issue 在 P7 #100 关闭后 spawn, 不阻塞 P7 阶段闭合
- 实施时 wallclock 脚本里的 qinyijiang mesh-prefix bug (`run_p7_wallclock_deployed.sbatch:case=qinyijiang` 用 case-name 当 mesh prefix 但应为 `nanlin`) 一并修复
- 优先级 P1 (P7 ship 后立即开干)

### Labels

- `s2-sibling`
- `p8` (如未存在则新建 label)
- `priority:p1`
- `runs-on:server`
