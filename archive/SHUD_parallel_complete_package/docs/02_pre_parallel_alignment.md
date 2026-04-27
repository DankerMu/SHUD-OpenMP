# 02 并行前必须完成的对齐工作

本节回答：**在真正开始并行前，SHUD 需要先对齐什么？**

关键原则是：并行阶段只能改“执行方式”，不应同时修“模型路径”。因此，凡是会导致 serial/parallel 语义不一致、会影响浮点加法顺序、会造成共享写入风险的工作，都应前移到单线程 parallel-ready 阶段。

## A0. 锁定 B0 historical serial base

### 目标

在任何代码改造前，先锁定当前单线程 SHUD 的行为。

### 要做的事

1. 选定最小标准算例：
   - 无 lake、小流域、短时段；
   - 有 lake、小流域、短时段；
   - 中等流域、含 river network；
   - 边界条件 / source-sink 激活算例；
   - dry/wet transition 算例。
2. 固定编译选项：
   - 禁止 `-ffast-math`；
   - 禁止非受控 FMA contraction；
   - 固定优化级别，例如 `-O2`；
   - 固定 SUNDIALS 版本。
3. 记录输出：
   - 所有 model output；
   - CVODE stats；
   - RHS call count；
   - wall-clock 和 I/O 时间；
   - 关键 flux/DY snapshots。

### 精度标准

B0 是历史基线，不需要和其他结果比。它的要求是：

```text
同一二进制、同一输入、多次单线程运行 bitwise identical。
```

如果 B0 自身不可复现，必须先解决 I/O、未初始化变量或非确定性路径问题。

### 风险

如果 B0 算例太少，会导致后续 B1 和并行验收覆盖不够。尤其需要覆盖 lake，因为当前 serial/omp 路径在 lake 处理上存在明显差异。

---

## A1. 建立唯一 RHS core

### 目标

把当前 `f_update/f_loop/f_applyDY` 与 `_omp` 路径整合为同一套 RHS core。

### 要做的事

当前入口：

```text
OpenMP: f_update_omp → f_loop_omp → f_applyDY_omp
Serial: f_update     → f_loop     → f_applyDY
```

应重构为：

```cpp
int f(double t, N_Vector CV_Y, N_Vector CV_Ydot, void *DS) {
    double* Y  = get_data(CV_Y);
    double* DY = get_data(CV_Ydot);
    MD->rhs_core(Y, DY, t, ExecPolicy::Serial);      // B1 阶段
    // 后续 strict OpenMP:
    // MD->rhs_core(Y, DY, t, ExecPolicy::StrictOMP);
}
```

`rhs_core()` 内部的过程顺序只写一份：

```text
reset/update state
  → element vertical processes
  → element/lake horizontal processes
  → segment-river flux compute
  → river downflow compute
  → deterministic gather
  → apply DY
```

### 精度标准

A1 完成后，如果只是函数组织调整，要求：

```text
RHS snapshots 与 B0 bitwise identical；
完整 run 与 B0 bitwise identical；
CVODE stats 与 B0 identical。
```

如果发现旧 `_omp` 路径与 serial 路径不同，不应把 `_omp` 路径作为 B1 依据；B1 首先继承 serial 语义。

### 风险

把代码抽成统一 core 时容易改变过程顺序。尤其是：

- ET 与 infiltration/recharge 的相对顺序；
- lake vertical/horizon 的相对顺序；
- `PassValue()` 调用位置；
- river downflow 与 segment flux 的先后关系。

这些顺序必须先按 B0 serial 固定。

---

## A2. 对齐已存在的 serial/omp 语义差异

### 目标

把当前 `_omp` 路径中缺失或不一致的逻辑，统一回 B1 serial RHS core 中。重点不是“修 OpenMP”，而是明确 B1 的唯一语义。

### 必须检查和对齐的事项

| 项 | 当前风险 | B1 处理原则 |
|---|---|---|
| lake vertical process | serial `f_loop()` 有 lake element 分支，`f_loop_omp()` 不等价 | B1 core 必须显式包含 lake vertical |
| lake horizon process | serial 对 lake element 调 `fun_Ele_lakeHorizon()` | B1 core 必须包含同等逻辑 |
| lake evaporation/precip accumulation | serial 中 `qLakeEvap/qLakePrcp += ...` | B1 先保持 serial 顺序；后续拆为 deterministic gather |
| ET | serial 普通 element 调 `f_etFlux()` | B1 必须明确 ET 调用位置 |
| river DY | serial 有 length、area clamp、`fun_dAtodY()`；omp 直接除 `u_TopArea` | B1 采用 serial 语义，除非另立数值修正变更 |
| lake DY | serial `f_applyDY()` 写 lake DY | B1 必须包含 lake DY |
| update/init | serial 清零 lake flux 和更新 lake area；omp 不等价 | B1 必须完整 reset |
| boundary/source-sink | serial/omp 细节需逐项比较 | B1 固定为单一实现 |
| negative state clipping | serial 和 omp 对 `Y` 是否 `max(0, Y)` 有差异 | 必须明确 B1 采用哪种语义，并解释 |

### 精度标准

如果 A2 只是把后续 parallel-ready core 明确继承 serial 语义，则 B1 仍应与 B0 bitwise identical。

如果 A2 修复了明确 bug，例如 `_omp` 路径缺 lake，但 B1 仍使用 serial 逻辑，则 B0 不变。只有当 serial 本身有明确 bug 并被修复时，才允许 B1 与 B0 不一致。

### 风险

A2 最容易把“并行对齐”误做成“物理修正”。建议规则：

```text
凡是会改变 B0 serial 输出的修复，必须单独立项，不混入并行对齐。
```

---

## A3. 拆分 compute flux 与 gather

### 目标

把所有“计算通量时顺手累加到 owner”的代码拆成两步：

```text
pure compute:     每条 edge/segment/lake-element 只写唯一 flux slot
deterministic gather: 每个 owner 按固定顺序汇总自己的贡献
```

### 需要拆的对象

| 对象 | 当前/潜在风险 | 改造方向 |
|---|---|---|
| segment → river | `QrivSurf[ir] += QsegSurf[i]` | `seg_by_riv[ir]` 固定顺序 gather |
| segment → element | `Qe2r_Surf[ie] += -QsegSurf[i]` | `seg_by_ele[ie]` 固定顺序 gather |
| upstream → downstream river | `QrivUp[down] += -QrivDown[i]` | `upstream_by_down[down]` 固定顺序 gather |
| lake element → lake | `qLakeEvap[lake] += ...` | `ele_by_lake[lake]` 固定顺序 gather |
| element neighbor flux | 可能双写邻居 flux | edge-owner 或 owner-local fixed j loop |
| global summaries | 水量平衡统计 | 单独 deterministic summary pass |

### 推荐模式

不推荐：

```cpp
#pragma omp parallel for
for (int i = 0; i < NumSegmt; ++i) {
    int ir = RivSeg[i].iRiv - 1;
    #pragma omp atomic
    QrivSurf[ir] += QsegSurf[i];
}
```

推荐：

```cpp
#pragma omp parallel for schedule(static)
for (int iseg = 0; iseg < NumSegmt; ++iseg) {
    compute_segment_flux(iseg);  // 只写 QsegSurf[iseg], QsegSub[iseg]
}

#pragma omp parallel for schedule(static)
for (int ir = 0; ir < NumRiv; ++ir) {
    double surf = 0.0;
    double sub  = 0.0;
    for (int k = 0; k < nseg_by_riv[ir]; ++k) {
        int iseg = seg_by_riv[ir][k]; // 固定升序
        surf += QsegSurf[iseg];
        sub  += QsegSub[iseg];
    }
    QrivSurf[ir] = surf;
    QrivSub[ir]  = sub;
}
```

### 精度标准

A3 在单线程中完成时，应先按 B0 的贡献顺序构造 adjacency list。若排序与 B0 循环顺序一致，则应达到：

```text
RHS snapshots 与 B0 bitwise identical。
```

如果 owner-local gather 的顺序与 B0 的全局 loop 顺序不同，会产生浮点 bit 差异。strict 阶段不允许这种差异，除非把 B1 明确锁定为新的 fixed-order reference。

### 风险

拆分后数组数量增加，内存上升；但这是换取 deterministic parallelism 的必要成本。

---

## A4. 固定拓扑顺序和 owner 映射

### 目标

让所有汇总都有确定的 owner 和确定的贡献顺序。

### 需要建立的 adjacency / owner list

| 名称 | 用途 | 排序规则 |
|---|---|---|
| `seg_by_riv[ir]` | river 汇总来自 segment 的 surface/sub flux | 原始 segment id 升序，或与 B0 loop 等价 |
| `seg_by_ele[ie]` | element 汇总来自 river segment 的交换 flux | 原始 segment id 升序，或与 B0 loop 等价 |
| `upstream_by_down[ir]` | downstream river 汇总 upstream downflow | upstream river id 升序，或与 B0 loop 等价 |
| `ele_by_lake[ilake]` | lake 汇总湖面降水/蒸发/湖岸 flux | element id 升序，或与 B0 loop 等价 |
| `edge_by_ele[ie]` | element total 汇总三个邻边 flux | 保持 `j=0..2` |
| `edge_owner[e]` | 每条 element-element edge 的唯一计算 owner | 固定规则，例如较小 element id owns |

### 精度标准

拓扑 list 构造本身不改变结果。用这些 list 替换原循环后要求：

```text
若贡献顺序等价于 B0：bitwise identical；
若贡献顺序不等价：必须先锁定为 B1，并记录差异。
```

### 风险

拓扑 list 的排序规则一旦改变，会改变浮点求和顺序。必须把排序规则写入 manifest，并纳入回归测试。

---

## A5. 整理 scratch arrays 与共享状态

### 目标

把所有临时数组和状态更新变成 owner-local 或 thread-local，避免隐式共享副作用。

### 要检查的对象

1. `qEle*`、`Qele*`、`Qseg*`、`Qriv*`、`QLake*`；
2. `Ele[i]`、`Riv[i]`、`lake[i]` 内部 update 函数是否写共享对象；
3. global variables：`uYsf/uYus/uYgw/uYriv/uYlake/globalY/timeNow/lakeon`；
4. debug / print / flood warning 是否在 RHS 内写共享文件；
5. NaN check 是否有非线程安全输出。

### 改造原则

```text
每个数组元素只能有唯一 owner 写入；
跨 owner 的贡献先写 flux slot，不直接写 accumulator；
线程内部临时变量放 stack 或 thread-local scratch；
RHS 内不做文件输出；
诊断输出放在 RHS 外部，或 strict serial diagnostic mode。
```

### 精度标准

A5 只整理写入所有权，不应改变公式和顺序。要求：

```text
RHS snapshots 与 B0/B1 bitwise identical。
```

### 风险

`Ele[i].updateElement()`、`Riv[i].updateRiver()`、`lake[i].update()` 若内部依赖全局变量或写共享对象，需要进一步拆出 pure update 或明确 owner。

---

## A6. forcing cache 与输入访问优化

### 目标

减少 forcing 反复读文件造成的 I/O 开销，同时不改变 `getX()` 语义。

### 背景

当前 `_TimeSeriesData::read_csv()` 每次 refill 都会重新打开文件，并跳过已读队列行。`getX()` 返回当前 queue 行的值。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/TimeSeriesData.cpp

### 要做的事

1. 先实现语义等价缓存：
   - 保持同样的时间单位转换；
   - 保持同样的 `movePointer()` 逻辑；
   - 保持 `getX()` 零阶取值，不引入插值。
2. 对每个 forcing 文件建立只读缓存或流式 reader：
   - 小文件可 preload；
   - 大文件可 memory-map / buffered sequential reader；
   - 禁止在 RHS 中反复打开文件。
3. 在 RHS snapshot 中比较所有 `tsd_*getX(t, col)` 返回值。

### 精度标准

```text
同一 t、同一 col，cache getX() 与 B0 getX() bitwise identical；
完整 run 与 B0/B1 bitwise identical。
```

### 风险

如果同时引入 forcing interpolation，会改变数值解；这属于精度路线，不属于 strict 并行前置优化。

---

## A7. 建立 RHS snapshot 和完整 run 验证工具

### 目标

后续每次改造都可以定位差异来自哪里。

### RHS snapshot 至少导出

```text
Y
DY
uYsf/uYus/uYgw/uYriv/uYlake
qElePrep/qEleNetPrep/qEleInfil/qEleExfil/qEleRecharge
qEs/qEu/qEg/qTu/qTg/qEleETP/qEleETA
QeleSurf/QeleSub/QeleSurfTot/QeleSubTot
QsegSurf/QsegSub
Qe2r_Surf/Qe2r_Sub
QrivSurf/QrivSub/QrivUp/QrivDown
QLakeSurf/QLakeSub/QLakeRivIn/QLakeRivOut/qLakeEvap/qLakePrcp
```

### 验证方法

1. 固定 `t` 和 `Y`，直接调用 RHS；
2. 对每个数组做：
   - bitwise compare；
   - max absolute error；
   - max relative error；
   - ULP histogram；
   - first mismatch index；
3. 完整 run 对比：
   - 输出文件；
   - CVODE stats；
   - RHS call count；
   - internal steps；
   - error test failures；
   - linear solver iterations。

### 精度标准

A7 是工具阶段，不改变模型。工具自身需要通过自测：同一文件和自身比较必须全通过；故意扰动一个数组值时必须能定位。

---

## A8. 锁定 B1 parallel-ready serial reference

### 目标

B1 是后续并行的唯一 base。

### B1 必须具备的性质

1. 唯一 RHS core；
2. flux compute 与 gather 已拆分；
3. 拓扑顺序固定；
4. forcing 访问语义锁定；
5. scratch arrays owner 明确；
6. 单线程完整 run 可复现；
7. strict instrumentation 可定位差异。

### B1 验收标准

理想目标：

```text
B1 与 B0 bitwise identical。
```

允许例外：如果 B1 包含明确 bug fix 或路径修复，则必须提供：

```text
B1_CHANGELOG.md
B0_vs_B1_RHS_report.md
B0_vs_B1_full_run_report.md
water_balance_report.md
hydrograph_metric_report.md
```

只有 B1 被锁定后，才进入并行阶段。
