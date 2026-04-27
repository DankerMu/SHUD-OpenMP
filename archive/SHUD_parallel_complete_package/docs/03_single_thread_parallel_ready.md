# 03 单线程 parallel-ready 预优化路线

本节回答：**为了保证更高并行效率和精度，为什么要先优化单线程 SHUD？具体优化什么？**

这里的“单线程优化”不是单纯追求单线程速度，而是把 SHUD 整理成后续并行可以安全接管的 reference implementation。

## 1. 单线程预优化的定位

### 不应做的事

单线程预优化阶段不应混入：

- 新物理过程；
- 新 forcing 插值方案；
- 新 CVODE 线性求解器；
- 新误差容差策略；
- pairwise / Kahan / superaccumulator 等新求和算法；
- 基于 OpenMP reduction 的并行汇总。

### 应做的事

单线程预优化阶段应完成：

1. 统一 RHS 路径；
2. 复用当前 serial 语义；
3. 拆分纯计算与汇总；
4. 固定拓扑顺序；
5. 降低 I/O 和重复更新开销；
6. 建立强验证工具；
7. 锁定 B1。

## 2. 为什么这一步能提升并行效率

如果不先做 parallel-ready 预优化，后续并行会遇到三个效率瓶颈：

### 2.1 共享累加导致 atomic/critical 开销

如果继续使用：

```cpp
QrivSurf[ir] += QsegSurf[i];
Qe2r_Surf[ie] += -QsegSurf[i];
```

后续并行时只能使用 atomic、critical 或私有副本合并。atomic/critical 会造成严重同步开销；私有副本合并如果无固定顺序，会破坏 bitwise reproducibility。

预优化通过 deterministic gather 把这类开销改为 owner-local loop。

### 2.2 两套 RHS 路径导致并行调试成本暴涨

如果继续维护 `f_loop()` 和 `f_loop_omp()` 两套实现，每次发现误差，都要同时排查：

```text
路径是否少算了过程？
数组是否没清零？
浮点顺序是否变化？
线程是否 data race？
```

统一 RHS core 后，strict 并行阶段只需要排查执行策略和写入所有权。

### 2.3 forcing I/O 和重复状态更新会掩盖真正计算热点

当前 forcing 读取存在反复打开文件和跳行的模式；同时某些状态 update 可能在 forcing update 和 RHS loop 中重复执行。单线程阶段先做缓存和 update 合并，能让后续并行 profiler 更真实地看到计算热点。

## 3. 为什么这一步能保证并行精度

并行精度的核心不是“每个线程都算得对”，而是：

```text
每个状态变量、通量变量和 DY 的写入 owner 明确；
每个浮点求和的贡献顺序明确；
每个过程只执行一次，且执行顺序可验证。
```

单线程预优化会把这些规则先落实到 B1。后续 strict OpenMP 只在 B1 上加 execution policy。

## 4. 单线程预优化阶段清单

| 阶段 | 目标 | 主要改造 | 是否允许改变 B0 输出 | 交付物 |
|---|---|---|---|---|
| S0 | 锁定 B0 | benchmark、RHS snapshot、CVODE stats | 不涉及 | B0 report |
| S1 | 唯一 RHS core | 合并 serial/omp 逻辑，统一过程顺序 | 原则上不允许 | `rhs_core()` |
| S2 | 语义对齐 | lake、ET、river DY、update/init 对齐 | 原则上不允许；bug fix 需单列 | semantic diff report |
| S3 | flux/gather 拆分 | pure compute + deterministic gather | 不允许，除非求和顺序变化被锁为 B1 | topology/gather report |
| S4 | 拓扑固定 | owner mapping、adjacency list、排序 manifest | 不允许 | topology manifest |
| S5 | scratch/状态整理 | owner-local arrays、thread-safe diagnostics | 不允许 | ownership map |
| S6 | forcing cache | 语义等价 cache/preload | 不允许 | forcing equivalence report |
| S7 | profile/instrument | wall-clock、RHS、I/O、CVODE stats | 不允许 | profile report |
| S8 | 锁定 B1 | B1 benchmark set | B1 可与 B0 相同或带解释差异 | B1 reference |

## 5. 推荐的单线程架构形态

### 5.1 RHS core 结构

建议把 RHS 拆成以下阶段：

```cpp
void rhs_core(double* Y, double* DY, double t, ExecPolicy policy) {
    rhs_reset_and_update_state(Y, DY, t, policy);
    rhs_element_vertical(t, policy);
    rhs_element_horizontal_flux(t, policy);
    rhs_segment_river_flux(t, policy);
    rhs_river_downflow(t, policy);
    rhs_lake_flux(t, policy);
    rhs_deterministic_gather(policy);
    rhs_apply_dy(DY, t, policy);
}
```

在 B1 阶段，`policy = Serial`，但所有函数已经具备 owner-local 和 fixed-order 结构。

### 5.2 flux storage

推荐把“通量计算结果”和“汇总结果”明确分开：

```text
QsegSurf/QsegSub       = segment flux slots
QedgeSurf/QedgeSub     = element-edge flux slots, if needed
QLakeEleSurf/Sub       = lake-element flux slots, if needed
QrivSurf/QrivSub       = river owner gather result
Qe2r_Surf/Qe2r_Sub     = element owner gather result
QrivUp                 = downstream river owner gather result
QeleSurfTot/SubTot     = element owner gather result
```

### 5.3 topology manifest

B1 需要保存拓扑排序规则，例如：

```yaml
segment_order: original_input_order
seg_by_riv_order: segment_id_ascending_matching_B0_loop
seg_by_ele_order: segment_id_ascending_matching_B0_loop
upstream_by_down_order: river_id_ascending_matching_B0_loop
lake_ele_order: element_id_ascending_matching_B0_loop
edge_owner_rule: min(element_id_i, element_id_j)
```

这个 manifest 不只是文档，也是回归测试的一部分。

### 5.4 diagnostics

RHS 内部不应直接输出大量日志。建议形成：

```text
RHS diagnostic buffer
  → RHS 结束后 serial dump
  → full run report
```

这样可以避免并行后文件输出顺序造成非确定性。

## 6. 单线程预优化中的精度标准

### 6.1 纯重构阶段

如果只是重排代码结构，但保持公式和求和顺序，要求：

```text
RHS bitwise identical with B0
full run bitwise identical with B0
CVODE stats identical with B0
```

### 6.2 fixed-order gather 阶段

如果 deterministic gather 的顺序与 B0 完全一致，仍要求 bitwise identical。

如果为了更适合并行而改变了 gather 顺序，应当：

1. 不在 strict 并行阶段做；
2. 先以单线程形式建立新 B1；
3. 报告 B0→B1 差异；
4. 后续 strict 并行只对 B1 做 bitwise identical。

### 6.3 forcing cache 阶段

forcing cache 必须逐点验证：

```text
for all tested t, col:
    old_getX(t, col) == cached_getX(t, col) bitwise
```

### 6.4 intentional bug fix 阶段

如果修复了 serial 本身的 bug，不能用“bitwise identical”作为验收，而应使用：

```text
局部 RHS 差异解释
水量平衡不恶化
关键输出差异可解释
回归报告记录为 B1 intentional change
```

## 7. 单线程预优化完成后的 go/no-go 条件

进入并行阶段前，必须满足：

- [ ] B0 已锁定；
- [ ] B1 已锁定；
- [ ] B1 单线程多次运行 bitwise identical；
- [ ] RHS snapshot 工具可用；
- [ ] full run 对比工具可用；
- [ ] topology manifest 可用；
- [ ] 所有 shared accumulation 已拆为 deterministic gather；
- [ ] 所有 process path 已在唯一 RHS core 中定义；
- [ ] forcing cache 与 B0/B1 语义一致；
- [ ] 编译选项固定且无 fast-math；
- [ ] 后续并行阶段的目标数组 owner 已明确。

如果其中任一项未满足，不建议进入 OpenMP 阶段。
