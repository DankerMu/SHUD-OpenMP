# P1e PR-A — rivqdown.dat 输出缓存 audit

## 概要

- **审计范围**：`SHUD/src/Model/shud.cpp` MainLoop + `SHUD/src/classes/Model_Control.cpp` PCtrl 数据流 + `SHUD/src/ModelData/MD_RiverFlux.cpp` `QrivDown` 写入路径 + `SHUD/src/Model/MD_rhs_core.cpp` 内 `Flux_RiverDown` 调用 + `SHUD/src/Model/f.cpp` `f()` RHS 入口。
- **判据**（spec L264-275 + design.md D5）：
  - 直接出现在 output stream → **internal cache** (unsafe — `CV_NORMAL` 模式下内部步可超过 `tout`，cache 内容对应 `t_internal ≠ tnext`，非确定)
  - 先 `recompute_flux(Y, tout, ...)` 再 output → **tout recompute** (OK)
- **结论**：**internal cache** — `QrivDown[]` 是 RHS 内部副作用 buffer，PCtrl 直接通过 `*PrintVar[]` 指针读取，无 tout recompute。
- **PR-B0 触发**：**yes** — design D5 要求 P1e-strict-OMP 阶段必须改造为按 `tout` 边界 recompute，否则 PR-C/PR-D 2×2 mode A 同 build 同 N × 3 reps 守门可能因 OMP 子句下 RHS 调用序列改变 → 不同 `QrivDown` 缓存值而 FAIL。

## Audit 命令 + 原始证据

```bash
$ grep -nE 'rivqdown|riv_Q_down' src/Model/*.cpp src/ModelData/*.cpp src/classes/*.cpp
src/ModelData/MD_initialize.cpp:374:        CS.PCtrl[ip++].Init(ForcStartTime, NumRiv, pf_out->riv_Q_down, CS.dt_Qr_down, QrivDown, 1, io_riv);
src/classes/IO.cpp:110:     i.g. pj0.eleysurf.dat/ prject001.rivqdown.dat
src/classes/IO.cpp:127:    sprintf(riv_Q_down, "%s/%s%s.rivqdown", outpath, projectname, suffix);
src/classes/IO.cpp:303:    strcpy(   riv_Q_down, p->riv_Q_down );

$ grep -nE "QrivDown" src/**/*.cpp src/**/*.hpp
src/ModelData/MD_initialize.cpp:374:        CS.PCtrl[ip++].Init(..., QrivDown, 1, io_riv);
src/ModelData/MD_readin.cpp:591:    delete[]    QrivDown;
src/ModelData/MD_RiverFlux.cpp:23:            QrivDown[i] = ManningEquation(CSarea, n, R, s);
src/ModelData/MD_RiverFlux.cpp:38:        QrivDown[i] = ManningEquation(CSarea, n, R, s);
src/ModelData/MD_RiverFlux.cpp:53:                QrivDown[i] = ManningEquation(CSarea, n, R, s);
src/ModelData/MD_RiverFlux.cpp:57:                QrivDown[i] = Riv[i].u_CSarea * sqrt(GRAV * uYriv[i]) * 60.;
src/ModelData/MD_RiverFlux.cpp:65:    CheckNANi(QrivDown[i], i, "RiverFlux Down");
src/Model/MD_rhs_core.cpp:341: *   (`DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + ...)`),
src/Model/MD_rhs_core.cpp:579:        QrivUp[ir] = -fixed_leftfold_sum_indexed(upstream_by_down[ir], QrivDown);
src/Model/MD_rhs_core.cpp:592:                    riv_in_by_lake[ilake], QrivDown);
src/Model/MD_rhs_core.cpp:716:            DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length;
src/ModelData/Model_Data.cpp:164:    QrivDown    = new double[NumRiv];
src/ModelData/Model_Data.cpp:427:    flood->InitPointer(yRivStg, QrivDown);
src/ModelData/MD_f_uncouple.cpp:102:            QrivUp[iDownStrm] += - QrivDown[i];
src/ModelData/MD_f_uncouple.cpp:199:            DY[i] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].u_TopArea;
src/ModelData/MD_update.cpp:40:                QrivDown[i] = 0.;
```

## 输出代码段定位

### 输出注册：PCtrl 在 init 阶段绑定 `QrivDown` 指针

File: `SHUD/src/ModelData/MD_initialize.cpp:374`

```cpp
if (CS.dt_Qr_down > 0)
    CS.PCtrl[ip++].Init(ForcStartTime, NumRiv, pf_out->riv_Q_down,
                        CS.dt_Qr_down, QrivDown, 1, io_riv);
```

`QrivDown` 是 `Model_Data` 内 `new double[NumRiv]` 分配的堆数组（`Model_Data.cpp:164`），生命周期 = Model_Data 对象。注册后 PCtrl 保存的是该数组首地址，每次 PrintData 读取当前内存内容。

### MainLoop：CVode advance 后立即 ExportResults，无 recompute

File: `SHUD/src/Model/shud.cpp:191-197`

```cpp
flag = CVode(mem, tnext, udata, &t, CV_NORMAL);
check_flag(&flag, "CVode", 1);
...
MD->summary(udata);
MD->CS.ExportResults(t);     // <-- 直接出库，无 recompute_flux(Y, tnext, ...) 调用
MD->flood->FloodWarning(t);
```

### PrintData 实现：从指针读取 + 时间平均 + fwrite

File: `SHUD/src/classes/Model_Control.cpp:469-491`

```cpp
void Print_Ctrl::PrintData(double dt, double t){
    long    t_long;
    NumUpdate++;                                /* 每次 ExportResults tick 累加 */
    for (int i = 0; i < NumVar; i++){
        buffer[i] += *(PrintVar[i]);            /* <-- 直接解引用 QrivDown[i]，无 recompute */
    }
    t_long = (long int)t;
    if ((t_long % Interval) == 0){              /* tau-aligned tick */
        for (int i = 0; i < NumVar; i++){
            buffer[i] *= tau / NumUpdate ;      /* 时间平均 */
        }
        NumUpdate = 0;
        if(Binary){
            fun_printBINARY(t-Interval, dt);    /* fwrite 到 .rivqdown.dat */
        }
        for (int i = 0; i < NumVar; i++){
            buffer[i] = 0.;
        }
    }
}
```

### QrivDown 写入：仅在 `f()` → `rhs_core` → `Flux_RiverDown` 内更新

File: `SHUD/src/Model/MD_rhs_core.cpp:436-438`

```cpp
for (i = 0; i < NumRiv; i++) {
    Flux_RiverDown(t, i);                       /* 触发 QrivDown[i] = ManningEquation(...) */
}
```

File: `SHUD/src/ModelData/MD_RiverFlux.cpp:5,23,38,53,57`

```cpp
void Model_Data::Flux_RiverDown(double t, int i){
    ...
    QrivDown[i] = ManningEquation(CSarea, n, R, s);   /* 多个分支均直接写 QrivDown[i] */
    ...
}
```

File: `SHUD/src/Model/f.cpp:8-65`（关键证据：`f()` 是 CVode 的 RHS callback，CV_NORMAL 模式下 CVode 可能内部多次调用 `f()`，最后一次 `f()` 的内部时间 `t_internal` 不一定等于 `tnext`）

```cpp
int f(double t, N_Vector CV_Y, N_Vector CV_Ydot, void *DS){
    ...
    MD->rhs_core(Y, DY, t, ExecPolicy::Serial);   /* 触发 Flux_RiverDown → QrivDown[] 写入 */
    MD->nFCall++;
    return 0;
}
```

## 数据源分析

1. `QrivDown[]` 主要写位置 = `Flux_RiverDown` (`MD_RiverFlux.cpp:23,38,53,57`)，即唯一以 Manning 方程结果更新 cache 的路径。另有两处共享写：(a) `MD_update.cpp:40` 在 `f_update` 路径下零-reset (`QrivDown[i] = 0.`，split-RHS uncoupled path 才执行；本审计主路径 `f()` → `rhs_core` 不经此分支)，(b) `Model_Data.cpp:427` 经 `flood->InitPointer(yRivStg, QrivDown)` 把指针别名给 FloodWarning pipeline。两者均不引入 tout-aware recompute，因此 internal cache 分类与 PR-B0 trigger 结论不变。
2. `Flux_RiverDown` 唯一被调位置：`rhs_core` (MD_rhs_core.cpp:437)。
3. `rhs_core` 由 `f()` callback (f.cpp:8) 调用，`f()` 注册为 CVode RHS 函数。
4. CVode 在 `CV_NORMAL` 模式下 internal step pattern：可能 take internal steps that overshoot `tout = tnext`，然后用 BDF interpolant 返回 `udata = Y(tnext)`。**但 `f()` 最后一次被调用时的 `t` 参数是 `t_internal ≥ tnext`，不是 `tnext` 本身**。
5. 因此 `QrivDown[]` 在 `ExportResults(t = tnext)` 被读取时，反映的是 `t_internal` 时刻的 Manning 方程结果，而 `udata` 则是 `Y(tnext)` 插值返回。两者时间不一致。
6. PCtrl `PrintData` 直接 `buffer[i] += *(PrintVar[i])`，无任何 `recompute(udata, tnext)` 调用 → 缓存值直接写盘。

## 结论

- **数据源类别**：**internal cache** （unsafe，per spec L264-275 / design D5）
- **理由**：
  1. `QrivDown[]` 是 `f()` RHS 内部副作用 buffer，非 Y 状态直接派生量。
  2. PCtrl 通过指针 `*PrintVar[i]` 读取该 buffer，无 tout 边界 recompute call。
  3. CV_NORMAL 模式 CVode internal step pattern 决定 `f()` 最后一次的 `t_internal`，不等于 ExportResults 看到的 `tnext`。
  4. 在 P1e StrictOMP RHS 上线后，OMP 并行下 RHS 求值序列 / 调用次数可能微变，进而 `QrivDown[]` 缓存内容微变 → 输出非比特一致。

## 修复 routing

- **internal cache** → SHALL 新增 **PR-B0** 重写 `rivqdown.dat` 输出路径为按 `tout` 边界 recompute（per design D5）。
- 候选实现路径（PR-B0 设计时再细化）：
  - **选项 1**：在 `ExportResults(t)` 内部、`PrintData` 调用之前，对 river 通道先调用 `recompute_river_flux(Y, t, QrivDown)`（独立从 `udata = N_VGetArrayPointer(udata)` 派生 → 调用 Manning 方程更新 `QrivDown`，与 RHS 内部缓存解耦）。
  - **选项 2**：将 `QrivDown` 在 `PrintData` 路径上换成 "snapshot at tout" 的备用 buffer (`QrivDown_out`)，避免污染 RHS 路径。
  - 选项 1 实施成本低，建议优先。

## 影响下游 PR

- **PR-B (2×2 driver script)**：driver 假定 `.rivqdown.dat` 是 deterministic（同 build / 同 N × 3 reps SHA256 一致 + 跨 N 同 case + 跨 mode bitwise 等价）。**mode A 内 reps-only 守门**仍 deterministic（串行 CVode 在固定 binary 下 `t_internal` 序列由 method order + state 唯一决定），cache 路由问题不在此处显化。**若 PR-B0 未先落地，PR-C/D 跨 mode (A vs C/D) 比特一致性审可能 spurious FAIL**（mode C/D 引入 OMP 线程后 RHS 调用序列变化 → `QrivDown` 缓存与 mode A baseline 不一致；即使 mode C/D 各自 reps 守门通过）。
- **PR-F (StrictOMP RHS 实施)**：strict OMP RHS 改变 RHS 求值序列 → 进一步放大 internal cache 漂移。**必须 PR-B0 先落地，才能保证 PR-F 上线时 mode C/D rivqdown 与 mode A 比特一致**。
- **PR-G (build flag + Makefile -fopenmp wiring)**：与本 audit 无直接耦合，但 PR-B0 修复 routing 时需注意 `Flux_RiverDown` 路径若也被纳入 StrictOMP for 循环，需保证 `QrivDown_out` recompute buffer 不参与 OMP 并行（避免重复跑同样的数据竞争问题）。

## 后续

- 本审计结论 → **PR-B0 强制要求**，PR-B0 应在 PR-B 之前进入（或 PR-B 阶段同步引入）。
- spec.md 内 `rivqdown.dat 输出缓存 audit (P1e.0, PR-A + 必要时 PR-B0)` 的 "必要时 PR-B0" 条件 = `internal cache` → **本审计已 trigger，PR-B0 必要**。
