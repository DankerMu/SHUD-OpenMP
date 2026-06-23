# P1.0 预审计：updateElement / updateRiver / lake.update 与 f_updatei case 1–5

## 背景与定义

本文档为 `openspec/changes/p1-update-omp/` 之 PR-C (issue #215) 所要求的 reviewer-only 预审计 (pre-audit)。其目的在于：在 `MD_update.cpp` 三处所有者循环 (owner loop) 添加 OpenMP pragma 之前，先以静态阅读 (static reading) 方式确认 5 个 update 函数与其调用图谱满足线程安全 (thread-safety) 条件。本审计**不修改任何 SHUD 源码**，其结论用于绑定 `design.md` 中 D9 决策 (path (a) / (b.1) / (b.2)) 的路径选择，并作为 PR-D / PR-E / PR-F 实现的前置门控 (gate)。

## 1. 审计对象与范围

本次审计共涵盖 5 个函数 + 1 个分派器 (dispatcher)，每项均按 read set / write set / shared object writes / RNG-time-IO / verdict 五维列出。审计对象如下：

1. `_Element::updateElement(double, double, double)` — `SHUD/src/classes/Element.cpp:257`
2. `_River::updateRiver(double)` — `SHUD/src/classes/River.cpp:49`
3. `_Lake::update()` — `SHUD/src/classes/Lake.cpp:104`
4. `Model_Data::f_updatei(Y, DY, t, flag)` case 1–5 — `SHUD/src/ModelData/MD_update.cpp:6-62`
5. `Model_Data::f_update(Y, DY, t)` 三处 owner loop — `SHUD/src/ModelData/MD_update.cpp:63-154`

审计类型：reviewer-only，零源码改动。关联 spec 为 `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md` 之 "P1.0 pre-audit" 需求项；决策绑定为 `design.md` 中 D9 决策的 (a) / (b.1) / (b.2) 路径选择。

审计锚点 (anchor) 如下：

- 外层 commit：`008913be8bb2b9be3720dbbfa01e309a9a34ee22` (main HEAD, 2026-06-22)
- SHUD submodule：`017c629e0359845821e51bb0b172ad02452a2541`

### 1.1 已确认为纯函数 / 可重入的辅助函数

经检视，5 个目标函数内部所调用的全部辅助函数均为无状态 (stateless) 的纯函数 (输入 → 输出，无隐藏状态、无 IO、无 RNG、无全局变量改动)。详见下表。

| Helper | 定义位置 | 性质 |
| --- | --- | --- |
| `effKH(Ygw, aqDepth, MacD, Kmac, AF, Kmx)` | `Equations.cpp:116` | 纯函数；`myexit` 仅在终止性错误下触发 (每个线程持有各自 Ele 的几何参数) |
| `satKfun(elemSatn, n)` | `Equations.cpp:136` | 纯函数 |
| `sat2psi(elemSatn, alpha, n)` | `Equations.hpp:31` (inline) | 纯函数 |
| `fixMaxValue(x, defVal)` | `functions.hpp:183` (inline) | 纯函数 |
| `fun_TopWidth / fun_CrossArea / fun_CrossPerem / fun_EqWidth` | `River.hpp:115-127` (inline) | 纯函数 |
| `LakeBathymetry::toparea(y)` | `Lake.cpp:59-78` | 仅读取 `this->yi[], ai[], nvalue` (`this` 即各 lake 实例自身成员)，不存在改动，安全 |
| `_TimeSeriesData::getX(t, col)` | `TimeSeriesData.cpp:122-125` | 返回 `ts[iNow][col]`；纯读；依据 S5a #176 契约 ("`movePointer` 之后 thread-safe read-only") 满足线程安全 |

`movePointer` 是 `_TimeSeriesData` 上唯一的改动者 (mutator)，但其**不出现于**本次审计涉及的 5 个函数体内部。它在每个外层时间步内仅在 RHS / update 并行候选区之外被调用一次。该结论已由对 `MD_update.cpp` 的 grep 检索确认。

### 1.2 索引宏均为 owner-local

`SHUD/src/Model/Macros.hpp:44-48` 中定义的索引宏如下：

```
#define iSF     i
#define iUS     i + NumEle
#define iGW     i + 2 * NumEle
#define iRIV    i + 3 * NumEle
#define iLAKE   i + 3 * NumEle + NumRiv
```

每个宏展开后均为以 per-iteration 变量 `i` 为键 (key) 的唯一偏移量。因此不同迭代步触及的 `Y[]` 槽位 (slot) 互不相交，线程间不存在数组槽位的伪共享 (false sharing)。

## 2. `_Element::updateElement(double Ysurf, double Yunsat, double Ygw)` 审计表

| Aspect | Finding |
| --- | --- |
| Signature | `void _Element::updateElement(double Ysurf, double Yunsat, double Ygw)` |
| Caller args | 来自 `f_update` element 循环的 `uYsf[i], uYus[i], uYgw[i]` (`MD_update.cpp:74,75,78,82`)；以及 `f_updatei` case 1/2/3 中由 `Y[]` 派生的值 |
| Read set (member) | `AquiferDepth`, `macD`, `macKsatH`, `geo_vAreaF`, `KsatH`, `infKsatV`, `hAreaF`, `macKsatV`, `ThetaS`, `ThetaR`, `Alpha`, `Beta` (全部在 `copyGeol/copySoil/copyLandc` 初始化之后即只读) |
| Read set (param) | `Ysurf` (函数体内未使用 — 形参绑定但仅消费 `Yunsat`, `Ygw`)，`Yunsat`, `Ygw` |
| Write set (member) | `u_effKH`, `u_deficit`, `Kmax`, `u_satn`, `u_theta`, `u_satKr`, `u_phius`, `u_effkInfi` — **全部为 `this` 实例成员** |
| Global read | 无 |
| Global write | 无 |
| RNG / time / IO | 无 (无 `rand`，无 `time`，除 `#ifdef DEBUG` 中已注释的 `fprintf` 外亦无 IO) |
| External calls | `effKH`, `satKfun`, `sat2psi`, `max` — 全部为纯函数 (见 §1.1) |
| Shared object write | **无** — 全部写入经由 `this->u_*` private / public 标量成员；每个线程持有不同的 `&Ele[i]`，因而 `this` 为 owner-local |
| Thread-safety verdict | **safe** |

附注：`Ysurf` 为悬空形参 (dead parameter)，即声明后函数体内未引用。此处仅作观察记录，不视为缺陷。

## 3. `_River::updateRiver(double newY)` 审计表

| Aspect | Finding |
| --- | --- |
| Signature | `void _River::updateRiver(double newY)` |
| Caller arg | 来自 `f_update` river 循环的 `uYriv[i]` (`MD_update.cpp:111`)；以及 `f_updatei` case 4 (`MD_update.cpp:41`) |
| Read set (member) | `BottomWidth`, `bankslope`, `Length` (全部在 `applyParameter/initialRiver` 初始化之后即只读) |
| Read set (param) | `newY` |
| Write set (member) | `u_Ystage`, `u_topWidth`, `u_CSarea`, `u_CSperem`, `u_eqWidth`, `u_TopArea` — **全部为 `this` 实例成员** |
| Global read | 无 |
| Global write | 无 |
| RNG / time / IO | 无 |
| External calls | `fun_TopWidth`, `fun_CrossArea`, `fun_CrossPerem`, `fun_EqWidth`, `fixMaxValue` — 全部为内联纯函数 (见 §1.1) |
| Cross-river dependency | **无** — 函数体从未对 `Riv[neighbor]` 进行索引，亦从未读取 `down`、`RivOut` 或任何拓扑字段。状态更新严格局限于 `Riv[i]`。(对照：`updateFrDownstream` 确实读取 `DownRiv[idown]`，但其仅在初始化阶段调用，不在 update 阶段触发。) |
| Shared object write | **无** |
| Thread-safety verdict | **safe** |

## 4. `_Lake::update()` 审计表

| Aspect | Finding |
| --- | --- |
| Signature | `void _Lake::update()` (无形参；读取 `this->yStage`, `this->zmin`) |
| Caller setup | `f_update` lake 循环在调用 `lake[i].update()` **之前**先执行 `yLakeStg[i] = Y[iLAKE]; lake[i].yStage = yLakeStg[i];` (`MD_update.cpp:137-138`) |
| Read set (member) | `yStage`, `zmin`, `bathymetry` (LakeBathymetry 实例由 `this` 拥有) |
| Write set (member) | `u_toparea` — owner-local |
| Global read | 无 |
| Global write | 无 |
| RNG / time / IO | 无 |
| External calls | `bathymetry.toparea(y)` — 作用于 `this->bathymetry` (per-lake 实例)，对其自身 `yi/ai/nvalue` 数组进行纯读 (见 §1.1) |
| Shared object write | **无** |
| Caller-side audit | L137 `yLakeStg[i] = Y[iLAKE]` — owner-local 数组写 (`iLAKE = i + 3·NumEle + NumRiv`，per-`i` 互不相交)。L138 `lake[i].yStage = yLakeStg[i]` — 在互不相同的 `&lake[i]` 上的 owner-local 成员写。二者皆安全。 |
| Thread-safety verdict | **safe** |

## 5. `f_update` 三处 owner loop (`MD_update.cpp:63-154`)

| Loop | Range | Reads | Writes (owner-local only) | External calls | Safety |
| --- | --- | --- | --- | --- | --- |
| **element** L64-L105 | `i = 0 .. NumEle-1` | `Y[iSF], Y[iUS], Y[iGW]`, `Ele[i].iBC`, `tsd_eyBC/eqBC` (经 `getX` 纯读), `t` | `QeleSubAt(i,j)`, `QeleSurfAt(i,j)` (j = 0..2, flat[3*i+j]，owner-local)；`QeleSubTot[i]`, `QeleSurfTot[i]`；`uYsf[i]`, `uYus[i]`, `uYgw[i]`；`Ele[i].QBC`, `Ele[i].yBC`；`qEleExfil[i]`, `qEleInfil[i]` | `tsd_eyBC.getX`, `tsd_eqBC.getX` (纯函数，见 §1.1)；`max` (纯函数) | **safe** — 所有写入均以 `i` 为键；`Ele[i]` 在不同线程之间为不同对象 |
| **river-update** L107-L125 | `i = 0 .. NumRiv-1` | `Y[iRIV]`, `Riv[i].BC`, `tsd_rqBC/ryBC` (经 `getX` 纯读), `t` | `uYriv[i]`；`Riv[i].updateRiver(uYriv[i])` — 经 §3 证实为完全 owner-local；`Riv[i].qBC`, `Riv[i].yBC` | `Riv[i].updateRiver` (§3)；`tsd_rqBC.getX`, `tsd_ryBC.getX` (纯函数)；`CheckNANi` (仅 DEBUG) | **safe** |
| **river-clear** L127-L131 | `i = 0 .. NumRiv-1` | 无 (write-only 清零) | `QrivSurf[i], QrivSub[i], QrivUp[i]` — owner-local | 无 | **safe** |
| **element-clear** L132-L135 | `i = 0 .. NumEle-1` | 无 | `Qe2r_Surf[i], Qe2r_Sub[i]` — owner-local | 无 | **safe** |
| **lake** L136-L147 | `i = 0 .. NumLake-1` | `Y[iLAKE]` | `yLakeStg[i]`；`lake[i].yStage`；`lake[i].update()` (§4 — owner-local)；`y2LakeArea[i]`；`QLakeSub[i], QLakeSurf[i], qLakeEvap[i], qLakePrcp[i], QLakeRivIn[i], QLakeRivOut[i]` — 全部 owner-local | `lake[i].update` (§4) | **safe** |
| **DY-clear** L148-L150 | `i = 0 .. NumY-1` | 无 | `DY[i]` — owner-local | 无 | **safe** (虽超出 P1 范围，仍列出以求完整) |

各循环均不存在跨迭代依赖 (即未出现 `Ele[i+k]` / `Riv[neighbor]` / `lake[j]` 在 `j != i` 处的读取)；不涉及归约 (reduction)；循环体内无 IO (仅有 `shud_rhs_dump_point` 位于循环体外、并行候选区之外)。所有边界条件 (BC) 读取均经由 `_TimeSeriesData::getX`，其线程安全性由 S5a 契约保证 (见 §1.1)，前提是 `movePointer` 串行执行并在他处调用，本文已验证。

## 6. `f_updatei` case 1–5 审计与 `f_update` 对照映射

`f_updatei(Y, DY, t, flag)` 是用于局部状态刷新 (partial state refresh) 的备用入口点。依据 spec NG7 条款，本入口**不在 PR-D / E / F 并行化范围之内**，本节仅作为 P2a reviewer 的前置参考资料。

| Case | Range | Semantics | Reads | Writes (owner-local) | vs `f_update` 对应项 | Safety |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | `i = 0 .. NumEle-1` (L8-12) | `uYsf` 刷新 (含 floor = 0) | `Y[i]` | `uYsf[i]` | 为 f_update element 循环写集的子集；floor-at-zero 语义一致 (L74 处无 floor — 微小差异，但此为 `flag` 设计上的语义分离) | **safe** |
| **2** | `i = 0 .. NumEle-1` (L13-17) | `uYus` 刷新 (含 floor = 0) | `Y[i]` | `uYus[i]` | 为 f_update element 循环子集 (L75 处无 floor) | **safe** |
| **3** | `i = 0 .. NumEle-1` (L18-32) | `uYgw` 刷新 + Element BC 更新 | `Y[i]`, `Ele[i].iBC`, `tsd_eyBC/eqBC.getX`, `t` | `uYgw[i]`, `Ele[i].QBC`, `Ele[i].yBC` | 与 f_update element 循环之 BC 子块 (L76-86) 一致；`getX` 纯读相同；per-`Ele[i]` 成员写相同 | **safe** |
| **4** | `i = 0 .. NumRiv-1` (L33-53) | `uYriv` + `Riv[i].updateRiver` + Riv BC | `Y[i]`, `Riv[i].BC`, `tsd_rqBC/ryBC.getX`, `t` | `uYriv[i]`, `QrivUp[i]=0`, `QrivDown[i]=0`, `Riv[i].updateRiver(uYriv[i])` 写集 (§3), `Riv[i].qBC`, `Riv[i].yBC` | 与 f_update river-update 循环 (L107-125) 写集一致；`updateRiver` 的线程安全条件相同 (经 §3 证明) | **safe** |
| **5** | `i = 0 .. NumLake-1` (L54-58) | `uYlake` 刷新 (含 floor = 0) | `Y[i]` | `uYlake[i]` | 为 f_update lake 循环子集 (L137 处无 floor；`f_updatei` 施加 floor) | **safe** |

**关于 `f_updatei` 的结论**：每个 case 的读 / 写集均为对应 `f_update` owner loop 写集的子集 (或与之相同)。因此线程安全性由 §2–§5 继承而来。case 4 中的 `updateRiver` 调用与 `f_update` river 循环中同名调用承担相同的安全条件 — 二者皆依赖 `_River::updateRiver` 为 owner-local-only，该条件已由 §3 证立。

`f_updatei` 各 case 与 `f_update` 的功能差异 (即是否在 `(Y[i] >= 0.) ? Y[i] : 0.` 处施加 floor) 属语义层 (semantics-level) 的设计取舍，**与线程安全性无关**。本审计**不**就 `f_updatei` 自身的并行化提出建议 — 该决策依据 spec NG7 延迟至 P2a 阶段。

## 7. 结论与签收

### 7.1 Verdict

**(a) safe** — 5 个函数 + `f_updatei` case 1–5 + `f_update` 三处 owner loop 全部仅执行严格的 owner-local 成员 / 数组写入。具体证据如下：

- **不存在共享对象写**：每个写入目标或是 (i) 由循环变量 `i` 索引的数组槽位 (`uYsf/uYus/uYgw/uYriv/uYlake`、`QeleSub*/QeleSurf*/Qe2r_*`、`QrivSurf/Sub/Up/Down`、`QLake*`、`qLakeEvap/Prcp`、`yLakeStg`、`y2LakeArea`、`qEleExfil/Infil`)，或是 (ii) per-iteration 对象 `Ele[i]` / `Riv[i]` / `lake[i]` 的成员 (`QBC`, `yBC`, `qBC`, `yStage`, `u_*`)。
- **不存在全局改动**：未写入 `tsd_*` (仅经 `getX` 纯读)；未写入模型层标量 (`NumEle`, `NumRiv`, `NumLake` 等均为只读)；未观察到 static-local 改动。
- **不存在 RNG / time / IO**：循环体内无此类调用 (DEBUG 模式下的 `CheckNANi` / `CheckNANij` 仅作只读 sanity check；`shud_rhs_dump_point` 在 `#ifdef SHUD_DUMP_RHS` 保护下位于循环体之外)。
- **不存在跨迭代数据依赖**：循环体内未出现 `Ele[i+k]`、`Riv[neighbor]`、`lake[j != i]` 等读取。邻接拓扑 (`down`, `nabr[]`, `lakenabr[]`, `iEleBank[]`) 仅由 `*Flux` 例程读取，本审计涉及的 update 函数不读取拓扑。
- **全部外部被调用方均为纯函数 / 可重入**：§1.1 已枚举所用的 12 个 helper，全部为无状态函数，或仅对 per-instance 自有数据进行只读访问，线程间无共享。

### 7.2 design D9 路径选择

**采纳路径 (a)**：判定为 in-scope 且无需 PR-Cfix。PR-D / PR-E / PR-F 可直接对 `f_update` 三处 owner loop (element / river-update / lake) 添加 `#pragma omp parallel for`，无需对所审计函数进行任何源码层重构。

`f_updatei` case 1–5 依据 spec NG7 **不在** PR-D / E / F 并行化范围之内；本审计对其的结论仅作为 P2a reviewer 的前置参考。

### 7.3 五函数线程安全性一行式汇总

| # | Function | Verdict | Rationale |
| --- | --- | --- | --- |
| 1 | `_Element::updateElement` | **safe** | 仅在 owner-local `Ele[i]` 上写入 `this->u_*`；helper 全部为纯函数 |
| 2 | `_River::updateRiver` | **safe** | 仅在 owner-local `Riv[i]` 上写入 `this->u_*`；不访问 `Riv[neighbor]` |
| 3 | `_Lake::update` | **safe** | 仅在 owner-local `lake[i]` 上写入 `this->u_toparea`；`bathymetry.toparea` 为纯读 |
| 4 | `f_updatei` case 1–5 | **safe** | 读 / 写集为 f_update 对应项子集；BC 读取均经 `getX` 纯函数 |
| 5 | `f_update` 三循环 | **safe** | 所有写入以 `i` 为键；无跨迭代读；BC 读取经纯函数 `tsd_*.getX` |

### 7.4 签收元数据

- signed_at: 2026-06-22
- signer: DankerMu (project owner)
- signed_against_commit: 外层 `008913be8bb2b9be3720dbbfa01e309a9a34ee22` + SHUD `017c629e0359845821e51bb0b172ad02452a2541`
- task ref: `openspec/changes/p1-update-omp/tasks.md` task 3.1–3.5 + 3.5b
- spec ref: `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md` Requirement "P1.0 pre-audit"
- design ref: `openspec/changes/p1-update-omp/design.md` decision D9 path (a)
- PR: #215 (PR-C)
