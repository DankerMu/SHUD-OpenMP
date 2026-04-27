# 04 后续并行阶段：每阶段改造哪些计算

本节回答：**在 B1 parallel-ready serial reference 建立后，每个并行阶段要改造哪些计算？**

原则：

```text
先并行无共享写的 local map；
再并行 pure flux compute；
再并行 owner-local deterministic gather；
最后才进入 CVODE 内部并行和 production reduction。
```

## P0. 并行执行策略框架

### 目标

在不改变 RHS core 语义的前提下，引入 execution policy。

### 改造对象

```cpp
enum class ExecPolicy {
    Serial,
    StrictOMP,
    ProductionOMP
};
```

所有 RHS 子过程都接收 policy：

```cpp
rhs_element_vertical(t, policy);
rhs_segment_river_flux(t, policy);
rhs_deterministic_gather(policy);
```

### 并行方式

P0 不真正并行，只建立接口。

### 精度标准

```text
policy = Serial 时，结果与 B1 bitwise identical。
```

---

## P1. 并行 reset / state update / initialization

### 目标

并行化最安全的 owner-local state update。

### 可改造计算

| 计算 | 并行方式 | owner |
|---|---|---|
| `DY[i] = 0` | `parallel for schedule(static)` | state index |
| `uYsf/uYus/uYgw` 更新 | element loop | element |
| element BC/SS 更新 | element loop | element |
| `qEleExfil/qEleInfil/...` 清零 | element loop | element |
| `uYriv`、`Riv[i].updateRiver()` | river loop | river |
| river BC 更新 | river loop | river |
| lake stage、area、lake flux 清零 | lake loop | lake |

### 不允许做的事

- 不允许在 update 阶段汇总跨 element/river/lake 的 flux；
- 不允许 debug print；
- 不允许对共享全局计数器做非原子写。

### 精度标准

```text
P1 RHS snapshot 与 B1 bitwise identical；
完整 run 与 B1 bitwise identical；
CVODE stats identical。
```

### 风险

`Ele[i].updateElement()`、`Riv[i].updateRiver()`、`lake[i].update()` 若内部写共享对象，会破坏并行安全。需要先审查这些函数是否只改自身对象。

---

## P2. 并行 element vertical processes

### 目标

并行化 element-local 垂向过程。

### 可改造计算

| 计算 | 说明 | owner |
|---|---|---|
| `f_etFlux(i,t)` | ET / canopy / snow / forcing local calculation | element |
| `Ele[i].updateElement()` | hydraulic properties update | element |
| `fun_Ele_Infiltraion(i,t)` | infiltration | element |
| `fun_Ele_Recharge(i,t)` | recharge | element |
| lake element vertical local terms | lake element 上的 local vertical flux | element |

### 前提条件

这些函数必须只写：

```text
Ele[i] 自身字段；
qEle*[i]；
yEle*[i]；
局部 scratch。
```

如果会写 `qLakeEvap[lake] += ...` 或其他 shared accumulator，必须改成：

```text
qLakeEvap_by_ele[i] = ...
qLakePrcp_by_ele[i] = ...
```

然后交给 P5 owner-local gather。

### 精度标准

strict 阶段：

```text
P2 与 B1 RHS bitwise identical。
```

由于 element vertical 是 owner-local，理论上最容易达到 bitwise identical。

### 风险

ET 或 snow 过程可能依赖 forcing pointer 状态。如果 forcing pointer 在 RHS 内被修改，必须保证 pointer update 在 RHS 前串行完成，或保证所有线程只读当前 forcing 值。

---

## P3. 并行 element horizontal / edge flux compute

### 目标

并行化 element-element surface/subsurface lateral flux 计算。

### 可改造计算

| 计算 | 说明 |
|---|---|
| `fun_Ele_surface(i,t)` | element surface lateral flux |
| `fun_Ele_sub(i,t)` | element subsurface lateral flux |
| edge-owner flux compute | 每条 edge 只计算一次 |

### 推荐策略

#### 策略 A：保持 element-owner + 固定 j loop

如果现有 `QeleSurf[i][j]` / `QeleSub[i][j]` 的写入只由 element `i` 自己负责，并且 `j=0..2` 顺序固定，则可以：

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < NumEle; ++i) {
    fun_Ele_surface(i, t);
    fun_Ele_sub(i, t);
}
```

前提是函数内部不写邻居的 `QeleSurf[inabr][jnabr]`。

#### 策略 B：改为 edge-owner flux slots

如果函数会同时写两侧 element，推荐改为 edge-owner：

```text
for each edge e:
    compute flux once
    QedgeSurf[e] = flux
    QedgeSub[e]  = flux

for each element i:
    gather its three edge fluxes in j=0..2 order
```

### 精度标准

如果使用策略 A 且写入顺序与 B1 相同：

```text
bitwise identical。
```

如果改为 edge-owner，单线程 B1 必须先建立并验证；并行阶段仍需与 B1 bitwise identical。

### 风险

element-element flux 是并行中最容易发生“双写邻居”的部分。必须先用 instrumentation 检查每个 `QeleSurf[i][j]` 和 `QeleSub[i][j]` 是否唯一写入。

---

## P4. 并行 segment-river pure flux compute

### 目标

并行化 river segment 与 element 的交换通量计算，但不做汇总。

### 可改造计算

| 计算 | 改造目标 |
|---|---|
| `fun_Seg_surface(iEle, iRiv, iSeg)` | 只写 `QsegSurf[iSeg]` |
| `fun_Seg_sub(iEle, iRiv, iSeg)` | 只写 `QsegSub[iSeg]` |

### 推荐结构

```cpp
#pragma omp parallel for schedule(static)
for (int iseg = 0; iseg < NumSegmt; ++iseg) {
    int ie = RivSeg[iseg].iEle - 1;
    int ir = RivSeg[iseg].iRiv - 1;
    QsegSurf[iseg] = compute_seg_surface(ie, ir, iseg, t);
    QsegSub[iseg]  = compute_seg_sub(ie, ir, iseg, t);
}
```

### 不允许做的事

```cpp
QrivSurf[ir] += QsegSurf[iseg];
Qe2r_Surf[ie] += -QsegSurf[iseg];
```

这些必须放到 P5 gather。

### 精度标准

```text
QsegSurf/QsegSub 与 B1 bitwise identical；
RHS snapshot 与 B1 bitwise identical。
```

### 风险

如果原函数内部同时做 compute 和 accumulate，拆分时要确保物理公式和符号方向完全一致。

---

## P5. 并行 owner-local deterministic gather

### 目标

并行化汇总，但保持每个 owner 内的浮点加法顺序固定。

### 可改造计算

| gather | owner | 贡献顺序 |
|---|---|---|
| segment → river | river | `seg_by_riv[ir]` |
| segment → element | element | `seg_by_ele[ie]` |
| upstream → downstream river | downstream river | `upstream_by_down[ir]` |
| lake element → lake | lake | `ele_by_lake[ilake]` |
| element edge → element total | element | `j=0..2` |

### 推荐结构

```cpp
#pragma omp parallel for schedule(static)
for (int ir = 0; ir < NumRiv; ++ir) {
    double surf = 0.0;
    double sub  = 0.0;
    for (int k = 0; k < seg_by_riv[ir].size(); ++k) {
        int iseg = seg_by_riv[ir][k];
        surf += QsegSurf[iseg];
        sub  += QsegSub[iseg];
    }
    QrivSurf[ir] = surf;
    QrivSub[ir]  = sub;
}
```

### 为什么不用 OpenMP reduction

OpenMP reduction 中值的组合位置和组合顺序未指定，不能保证与串行 bitwise identical。见规范：https://www.openmp.org/spec-html/5.0/openmpsu107.html

### 精度标准

strict 阶段：

```text
所有 gather 输出数组与 B1 bitwise identical。
```

需要比较：

```text
Qe2r_Surf/Qe2r_Sub
QrivSurf/QrivSub/QrivUp
QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut/qLakeEvap/qLakePrcp
QeleSurfTot/QeleSubTot
```

### 风险

如果 owner 内贡献顺序与 B1 不一致，结果可能出现 1–若干 ULP 差异。strict 阶段不接受此差异。

---

## P6. 并行 applyDY

### 目标

并行化 element、river、lake 的 DY 写入。

### 可改造计算

| 计算 | owner | 风险 |
|---|---|---|
| element DY | element | 低 |
| river DY | river | 中，必须使用 B1 serial 公式 |
| lake DY | lake | 中，需 lake flux gather 已完成 |

### 推荐结构

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < NumEle; ++i) {
    apply_dy_element(i, DY, t);
}

#pragma omp parallel for schedule(static)
for (int i = 0; i < NumRiv; ++i) {
    apply_dy_river(i, DY, t);
}

#pragma omp parallel for schedule(static)
for (int i = 0; i < NumLake; ++i) {
    apply_dy_lake(i, DY, t);
}
```

### 精度标准

```text
DY 全量与 B1 bitwise identical。
```

### 风险

当前 `f_applyDY_omp()` 的 river DY 公式与 serial 不一致，P6 必须使用 B1 的统一公式，不能继承旧 `_omp` 公式。

---

## P7. 完整 RHS OpenMP + serial CVODE

### 目标

在 CVODE 仍使用 serial `N_Vector` / 原线性求解器的前提下，只并行 SHUD RHS 应用层。

### 改造对象

- `rhs_core(..., ExecPolicy::StrictOMP)`；
- 所有 RHS 子阶段使用 OpenMP；
- CVODE vector 层暂不换 OpenMP `N_Vector`；
- 不换线性求解器；
- 不改容差。

### 精度标准

如果 P1–P6 均通过，P7 应达到：

```text
RHS snapshots 与 B1 bitwise identical；
完整 run 输出与 B1 bitwise identical；
CVODE internal stats identical；
RHS call count identical。
```

### 风险

如果完整 run 不一致，但 RHS snapshots 一致，问题可能来自：

- CVODE vector data access 差异；
- 未初始化变量；
- 输出/summary 的并行副作用；
- 非确定性 debug/log；
- 编译器因 OpenMP 开关改变 floating behavior。

---

## P8. CVODE vector / linear solver 并行与稀疏求解

### 目标

进入高性能 production 阶段，优化 CVODE 内部向量运算和线性求解层。

### 可改造计算

| 改造 | 目的 | 备注 |
|---|---|---|
| OpenMP `N_Vector` | 加速 CVODE vector ops | 可能改变 norm/dot product reduction 顺序 |
| Sparse matrix / KLU | 中小规模稀疏直接解 | 需要 Jacobian/sparsity 支持 |
| GMRES / FGMRES | 大规模刚性系统 | 需要预条件器 |
| physics-block preconditioner | 降低 Krylov 迭代 | 设计复杂 |
| vector absolute tolerance | 按变量尺度控制误差 | 精度路线相关 |

SUNDIALS/CVODE 文档指出其支持多种 `N_Vector` 实现，包括 OpenMP/Pthreads/MPI，也支持 GMRES、FGMRES 等 Krylov 方法；对于大规模刚性系统，Krylov 方法通常更可行，预条件器很关键。文档见：https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html

### 精度标准

P8 不再要求与 B1 bitwise identical，而进入 deterministic tolerance：

```text
同一配置多次运行 deterministic；
与 B1 的状态差异在 CVODE 容差尺度内；
水量守恒不恶化；
NSE/KGE/peak/balance 等水文指标差异可接受；
CVODE stats 变化可解释。
```

### 风险

CVODE 内部 reduction 顺序改变可能导致 error test 结果和内部步长序列改变。即使 RHS 本身 bitwise identical，完整 run 也可能不再 bitwise identical。

---

## P9. production deterministic reduction / compensated summation

### 目标

在生产模式下进一步提高数值稳定性和并行效率。

### 可选策略

| 策略 | 作用 | 是否与 B1 bitwise identical |
|---|---|---|
| fixed pairwise summation | 降低求和误差，固定顺序 | 通常否 |
| Kahan / Neumaier summation | 降低累计误差 | 否 |
| binned / superaccumulator | 强可复现 | 否，成本高 |
| deterministic tree reduction | 多线程稳定复现 | 否，除非 B1 也是同树 |
| OpenMP reduction | 快，但顺序 unspecified | 不适合作 strict |

### 精度标准

P9 的标准是“确定性 + 可解释容差”，不是 bitwise B1：

```text
同一线程数、多次运行 bitwise identical 或严格 deterministic；
不同线程数之间差异低于 tolerance；
与 B1 的水文指标差异低于设定阈值；
如果数值误差更低，可作为 new numerical reference 单独立项。
```

### 风险

更好的求和算法可能使结果偏离 B1，但这不是错误。问题在于不能和并行 bug 混在一起。因此 P9 必须晚于 P7。

## 并行阶段总表

| 阶段 | 并行对象 | 是否 strict bitwise | 主要风险 |
|---|---|---|---|
| P1 | reset/update/init | 是 | update 函数写共享状态 |
| P2 | element vertical | 是 | forcing pointer / lake accumulator |
| P3 | element horizontal edge flux | 是 | 双写邻居 flux |
| P4 | segment-river flux compute | 是 | compute 与 accumulate 未拆干净 |
| P5 | deterministic gather | 是 | owner 内顺序不一致 |
| P6 | applyDY | 是 | river/lake DY 公式不一致 |
| P7 | full RHS OpenMP + serial CVODE | 是 | 编译/未初始化/日志副作用 |
| P8 | CVODE vector / linear solver | 否，tolerance | CVODE 内部步长路径变化 |
| P9 | production summation | 否，tolerance | 新求和算法改变参考结果 |
