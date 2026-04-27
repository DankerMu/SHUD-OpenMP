# 01 源码观察：为什么必须先做单线程 parallel-ready 对齐

本节只记录与“并行前对齐”和“并行阶段划分”直接相关的源码观察。它不是完整代码审查，而是为并行路线提供依据。

## 1. RHS 入口已经分成两套实现

`src/Model/f.cpp` 中，OpenMP 和 serial 条件编译调用不同函数链：

```text
OpenMP: f_update_omp(Y, DY, t) → f_loop_omp(Y, DY, t) → f_applyDY_omp(DY, t)
Serial: f_update(Y, DY, t)     → f_loop(t)            → f_applyDY(DY, t)
```

源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/Model/f.cpp

这说明当前 OpenMP 路径不是“同一 RHS 的并行执行策略”，而是另一套实现。因此，当前 OpenMP 结果和串行结果不一致时，不应先归咎于浮点加法顺序。

## 2. `f_loop()` 与 `f_loop_omp()` 的过程覆盖不一致

串行 `f_loop()` 中存在 lake element 分支：

```text
if lakeon && Ele[i].iLake > 0:
    updateLakeElement()
    fun_Ele_lakeVertical()
    qLakeEvap += ...
    qLakePrcp += ...
else:
    f_etFlux()
    updateElement()
    fun_Ele_Infiltraion()
    fun_Ele_Recharge()
```

之后还会对 lake element 调用 `fun_Ele_lakeHorizon()`，普通 element 调用 `fun_Ele_surface()` / `fun_Ele_sub()`；并在 river/segment/lake 处理后调用 `PassValue()`。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp

当前 `f_loop_omp()` 主要覆盖普通 element 的 update/infiltration/recharge、surface/sub、segment、river downflow，然后调用 `PassValue()`；从当前公开源码看，它没有等价覆盖串行路径中的 lake vertical/horizon 和 `f_etFlux()` 分支。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp

这意味着 lake case、ET case 和普通 element case 在 serial/omp 路径中可能不是同一套 RHS。

## 3. `f_applyDY()` 与 `f_applyDY_omp()` 的 river DY 公式不一致

串行 `f_applyDY()` 中，river DY 先按 reach length 计算截面积变化，再限制负向变化不能超过可用截面积，最后通过 `fun_dAtodY()` 转换为水深变化：

```text
DY[iRIV] = (-QrivUp - QrivSurf - QrivSub - QrivDown + qBC) / Riv[i].Length
if DY[iRIV] < -Riv[i].u_CSarea:
    DY[iRIV] = -Riv[i].u_CSarea
DY[iRIV] = fun_dAtodY(DY[iRIV], Riv[i].u_topWidth, Riv[i].bankslope)
```

源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp

当前 `f_applyDY_omp()` 的 river DY 则直接除以 `Riv[i].u_TopArea`，没有同样的可用截面积限制和 `fun_dAtodY()` 转换。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp

因此，在 river stage 上，当前 OpenMP 路径不是串行路径的浮点加法重排，而是方程/变量转换路径不同。

## 4. `f_update()` 与 `f_update_omp()` 的初始化覆盖不一致

串行 `f_update()` 会执行：

- 清零 element flux arrays；
- 更新 `uYsf/uYus/uYgw`；
- 处理 element BC；
- 清零 `qEleExfil/qEleInfil`；
- 更新 river stage 和 river BC；
- 清零 `QrivSurf/QrivSub/QrivUp`、`Qe2r_*`；
- 更新 lake stage、lake area；
- 清零 lake 相关 flux；
- 清零 `DY`。

源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_update.cpp

当前 `f_update_omp()` 覆盖 element、river、DY 清零，但从当前公开源码看没有等价覆盖 lake 更新和 lake flux 清零。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp

这说明并行前必须先统一 update/init 语义，避免“脏数组”“未清零”“lake 状态未更新”等差异。

## 5. `PassValue()` 是共享浮点累加的集中风险点

串行 `PassValue()` 会执行三类汇总：

```text
segment → river:
    QrivSurf[ir] += QsegSurf[i]
    QrivSub[ir]  += QsegSub[i]

segment → element:
    Qe2r_Surf[ie] += -QsegSurf[i]
    Qe2r_Sub[ie]  += -QsegSub[i]

upstream river → downstream river:
    QrivUp[iDownStrm] += -QrivDown[i]
```

源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp

如果直接把这些循环改成 OpenMP 并行，并使用 `atomic +=` 或普通 `reduction`，虽然可以避免 data race，但无法保证与单线程相同的浮点加法顺序。strict 阶段必须把它改成 owner-local deterministic gather。

## 6. `fun_Seg_surface()` / `fun_Seg_sub()` 不能继续把“通量计算”和“汇总”绑在一起

`Model_Data.hpp` 显示 SHUD 当前有 `QsegSurf/QsegSub`、`QrivSurf/QrivSub`、`Qe2r_Surf/Qe2r_Sub`、`QLake*` 等 flux arrays。segment-river 交换如果在 flux 函数内部直接写入 river/element accumulator，就会形成并行共享写风险。相关结构定义见：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/Model_Data.hpp

后续应把 segment-river 计算拆成两步：

```text
compute_segment_flux(i) → 只写 QsegSurf[i], QsegSub[i]
gather_segment_flux(owner) → 按固定顺序写 Qriv*, Qe2r_*
```

## 7. forcing 读取本身适合单线程预优化，但不能在 strict 并行阶段改变语义

`TimeSeriesData::read_csv()` 当前每次 refill 会重新打开文件，跳过 `MAXQUE * nQue + 2` 行，再读下一段；`getX()` 直接返回当前行值，不做插值。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/TimeSeriesData.cpp

forcing cache / preload 是加速方向，但它应放在单线程预优化阶段完成，并且在 B1 锁定前证明：

```text
同一 t、同一 col 下 getX(t, col) 与 B0 完全一致
```

如果后续要引入 forcing interpolation，那属于精度路线或 production numerical improvement，不能混入 strict 并行。

## 8. CVODE 相关优化应晚于 RHS strict 并行

`Control_Data` 当前主要暴露 `num_threads`、`abstol`、`reltol`、`InitStep`、`MaxStep` 等控制量。源码依据：https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/Model_Control.hpp

SUNDIALS/CVODE 支持多种 `N_Vector` 实现，包括 serial、MPI、OpenMP、Pthreads 等；也支持 GMRES、FGMRES、Bi-CGStab、TFQMR、PCG 等 Krylov 方法，并指出对大规模刚性系统 Krylov 方法通常优于直接法，预条件器通常很关键。文档依据：https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html

但这些改变会影响 CVODE 内部 norm、dot product、误差测试和线性求解路径，可能改变自适应步长序列。因此必须晚于 RHS strict 并行。

## 9. 结论

当前 SHUD 并行改造不能直接从 `#pragma omp parallel for` 开始。必须先做：

1. 唯一 RHS core；
2. serial/omp 语义对齐；
3. flux compute 与 gather 拆分；
4. fixed-order topology；
5. forcing/cache 语义锁定；
6. B1 parallel-ready serial reference。

只有这些完成后，后续并行阶段才能真正回答：

```text
并行是否正确？
并行是否快？
差异是否来自合理的 production 数值策略？
```
