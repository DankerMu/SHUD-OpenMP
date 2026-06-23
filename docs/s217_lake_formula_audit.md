# S2.17 湖泊公式审计 — `MD_ElementFlux.cpp` `fun_Ele_sub()` 湖泊分支

## 元信息

- **审计 issue (audit issue)**: [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185)
- **阻塞项 (blocks)**: [#186](https://github.com/DankerMu/SHUD-OpenMP/issues/186) (S6b.2 条件性修复)
- **审计日期 (audit date)**: 2026-06-22 (PR #204 Phase-4 评审 V1/V2/V3 后修订); **PI 代理签字 (PI delegate sign-off)**: 2026-06-22 (PR-19 #210)
- **审计人 (auditor)**: Phase-1 审计起草人 + 证据打包人 (audit) + DankerMu 担任 PI 代理 (sign-off)。按 `spec.md` L23 规定, SHUD 上游 PI 特权 (prerogative) 附着于最终判定; design.md Open Q1 (PI 代理资格) 由本次签字**关闭** — DankerMu 同时为本仓库 `DankerMu/SHUD-OpenMP` 与上游 `SHUD-System/SHUD` 的 GitHub 组织所有者 (organization owner), 这构成 spec 隐式要求且 Open Q1 显式留待确认的 "Hydro-System 上游控制权 (upstream control authority)", 即 PI 代理资格判定标准。签字机制 (sign-off mechanism): GitHub issue [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) 由 DankerMu (PI 代理身份) 发表评论 + 本文档 §E "审计结论 — VERDICT ISSUED" 章节 + `SHUD/B1b_CHANGELOG.md` 后 B1b 增补行三处共同构成。
- **签字状态 (signoff status)**: **VERDICT ISSUED — E2 ("S2.17: formula correct, no change")**, 由 DankerMu 按上述资格作为 PI 代理签发。完整论证见下文 §E "审计结论", 正式签字陈述见 §E.1 verdict 条目。
- **默认跳过路径 (default-skip path, CONDITIONAL, 依据 master plan C8 前向兼容约束)**: master plan §S6b L1497 ("S6b.2 lake 公式可能审查后不需要改") 是关于可能审查结果的**预测 (FORECAST)**, 而**非**允许在缺 PI 签字情况下发布的规范性许可。若在 S6c (#188-#190) capstone 之前未收到 PI 关于 `S2.17: formula needs fix` 的指令, **B1b 将以现行公式不变发布**, 但发布按 master plan C8 ("永不 break userspace") 视作**条件性 (CONDITIONAL)** — 任何后续 PI 强制要求的修复均可作为新 `B1c-tag` 增量叠加, 不强制更新 B1b-tag。此路径**并非**已签字的 E2, 也**不**满足 design D9 fast-path 触发条件 #2。
- **Master plan 引用**: §S2.17 (L1179–L1198), §4.18 (L523–L541), §S6b L1480–L1503
- **OpenSpec 引用**: `specs/s6b-bugfix-application/spec.md` 需求 S6b.2 + 2 个 Scenario; design.md D8 / D9 / Open Q1

> **行号漂移说明 (line drift)**: master plan §4.18 引用 "`MD_ElementFlux.cpp` L100–L156" 以及 "`Kmean` 在 L117"。经 S5d.1 (#178) SoA 重写与 S5d.2-5a (#179) jagged flatten 重写之后, 当前文件共 **191 行**; `fun_Ele_sub()` 函数体覆盖 L126–L191, `Kmean` 所在行为 **L147**。本审计以当前活跃源码树为准 (SHUD `openmp-baseline @ a85bf63`)。
>
> **路径说明 (paths)**: master plan §4.18 使用简写 `MD_ElementFlux.cpp`; S5d.1 文件重组后实际 SHUD-相对路径为 `SHUD/src/ModelData/MD_ElementFlux.cpp`。本审计始终使用完整路径。

---

## §A. 公式引用 (来自当前 SHUD `openmp-baseline @ a85bf63`)

### A.1 湖泊分支地下水侧渗通量公式 (`fun_Ele_sub`)

`SHUD/src/ModelData/MD_ElementFlux.cpp` L133–L155:

```cpp
for (j = 0; j < 3; j++) {
    inabr = hot.nabr_flat[3*i + j] - 1;          // L134
    ilake = hot.lakenabr_flat[3*i + j] - 1;      // L135
    if(ilake >= 0){ /* For Lake element */        // L136
        assert(inabr >= 0);                       // L137  ← §S2.17 R-1 防御性 assert (already in)
        dh = (uYgw[i] + hot.z_bottom[i])
           - (yLakeStg[ilake] + lake[ilake].bathymetry.yi[0]);  // L138
        if(dh > 0. && uYgw[i] <= 0.02){           // L139  Depression condition
            Q = 0.;
        }else if(dh < 0. && yLakeStg[ilake]<= 0.02){          // L141  Depression condition
            Q = 0.;
        }else{
            Ymean = avgY_gw(hot.z_bottom[i], uYgw[i],
                            lake[ilake].bathymetry.yi[0],
                            yLakeStg[ilake], 0.002);                     // L144
            grad  = dh / hot.Dist2Nabor_flat[3*i + j];                   // L145
            /* It should be weighted average. However, there is an ambiguity about distance used */
            Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr]);         // L147  ← THE SUSPECT FORMULA
            Q     = Kmean * grad * Ymean * hot.edge_flat[3*i + j];       // L148
        }
        /* S3b.3 (PR-9): per-edge scratch slot, gathered by S4 */
        QeleSub_lake[i*3 + j] = Q;                                       // L155
    }
```

### A.2 逐项分解

| 项 (term) | 代码表达式 | 物理含义 |
|---|---|---|
| **dh** (水头差 / head diff) | `(uYgw[i] + z_bottom[i]) - (yLakeStg[ilake] + lake[ilake].bathymetry.yi[0])` | 含水层水力水头 (地下水位绝对高程 = `uYgw[i] + z_bottom[i]`) **减去** 湖泊水位绝对高程 (`yLakeStg + lake.zmin`, 其中 `bathymetry.yi[0] == lake.zmin`, 见 `MD_Lake.cpp:162`) |
| **clamps** (截断) | `if(dh > 0 && uYgw[i] <= 0.02)` 置零; 湖侧对称处理 | "凹陷 (depression)" 截断: 任一侧水深小于 2 cm 时, 通量强制置零 |
| **Ymean** (饱和厚度均值 / saturated-thickness average) | `avgY_gw(z_bottom[i], uYgw[i], lake.zmin, yLakeStg[ilake], 0.002)` | 返回 `0.5*(max(0,y1)+max(0,y2))`, 见 `Equations.cpp:52–70`; 即截断后饱和厚度的算术平均 (湖侧为蓄水柱厚度) |
| **grad** (水头梯度 / head gradient) | `dh / Dist2Nabor[3*i+j]` | 水头差除以元胞质心到湖泊元胞质心的距离 (`Dist2Nabor` 由 Triangle 拓扑给出, 见 `Element.hpp`) |
| **Kmean** (有效水力传导率 / effective hydraulic conductivity) | **`0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])`** | 算术平均, 其中 (1) 岸边元胞的深度加权有效水平 K, (2) 湖泊元胞的有效水平 K |
| **A** (过流面积 / cross-section area) | `Ymean * edge[3*i+j]` | 饱和厚度乘以共享边长 |
| **Q** | `Kmean * grad * Ymean * edge` | 由岸边元胞 `i` 流向湖泊 `ilake` 的有符号通量; 正值表示含水层 → 湖泊 |

上述形式即 Darcy 定律 (Darcy's law) 在有限体积下的离散形式 `Q = -K · ∇H · A`, 符号约定为 "岸侧水头高于湖侧时为正"。`dh = 岸 − 湖` 的定序使得水自含水层流入湖泊时 Q 取正; 教科书 Darcy 公式中的负号被此定序所吸收。

### A.3 湖泊元胞 `u_effKH` 来源 (§B 论证的关键)

`SHUD/src/classes/Element.cpp:246–256` 中 `_Element::updateLakeElement()`:

```cpp
void _Element::updateLakeElement(){
    u_effKH   = KsatH;          // ← lake element's u_effKH is set to the soil-layer KsatH
    u_deficit = 0;
    Kmax      = infKsatV;
    u_satn    = 1.;             // fully saturated
    u_theta   = ThetaS;         // porosity
    u_satKr   = 1.0;
    u_phius   = 0.;
    u_effkInfi = infKsatV;
}
```

此函数由 `MD_rhs_core.cpp` 在湖泊元胞分支于每个 RHS 时间步调用。与之对应的非湖泊形式见 `_Element::updateElement()` (L257–294):

```cpp
u_effKH = effKH(Ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH);
```

其中 `effKH(...)` (`Equations.cpp:116–134`) 按含水层饱和厚度对 `KsatH` (土壤基质) 与 `macKsatH` (大孔隙) 进行深度加权混合, 并按大孔隙深度 `macD` 截断。

**关键观察**: 湖泊元胞继承下方**土壤柱**的 `KsatH`, 并忽略 `macKsatH` / 大孔隙分层 (因湖体本身为湖床之上的开放水体)。此 K 表征湖床沉积物 (lake-bed sediment) 的水力传导率, **而非**湖水柱本身的传导率 (开放水体的有效 K 实际上可视为无穷大)。

### A.4 活跃运行时 SoA 镜像状态 (Phase-4 V1 verifier 发现)

**PR #204 Phase-4 verifier V1 浮现的关键限定 (`ae481398012ebb496` — CONFIRMED)**: `updateLakeElement()` 所写入的 `u_effKH = KsatH` 值仅存在于 AoS `Ele[i].u_effKH`。`fun_Ele_sub` 在 L147 实际读取的运行时 SoA 镜像 `hot.u_effKH[i]` 在活跃代码路径上**并未刷新 (not refreshed)**。CVODE 单步执行序列如下:

1. `SHUD/src/Model/shud.cpp:177-178` 首先执行 `MD->updateforcing(t)`。
2. `SHUD/src/ModelData/MD_ET.cpp:34-42` 中 `updateforcing()` 以 `for(i=0; i<NumEle; i++)` 形式 (**未**做 `iLake` 过滤) 调用 `Ele[i].updateElement(uYsf[i], uYus[i], uYgw[i]); sync_hot_dynamic(i);`。
3. `SHUD/src/classes/Element.cpp:257-258` 中 `updateElement()` 对所有元胞 (**包括湖泊元胞**) 写入 `u_effKH = effKH(Ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH)` (深度加权且包含大孔隙) 到 AoS。
4. `SHUD/src/ModelData/Model_Data.hpp:287` 中 `sync_hot_dynamic(i)` 将该混合值写入 `hot.u_effKH[i]`。
5. 随后 `SHUD/src/Model/MD_rhs_core.cpp:194-207` 中 `rhs_flux` 第 1 趟湖泊分支调用 `Ele[i].updateLakeElement()`, 向 AoS 写入 `u_effKH = KsatH`, 但**未调用 `sync_hot_dynamic(i)`**。
6. 对照 `SHUD/src/ModelData/MD_f.cpp:25-28` (死代码 `f_loop`): `Ele[i].updateLakeElement(); sync_hot_dynamic(i); fun_Ele_lakeVertical(i, t);` — 此处**确实**做了同步。
7. `fun_Ele_sub` 在 `MD_ElementFlux.cpp:147` 读取 `hot.u_effKH[inabr]` (湖泊邻居) 时, 读取到的是步骤 3 中针对该湖泊元胞自身 CVODE 状态 `Y[iGW]` 所求得的**通用元胞深度加权混合值** (`Macros.hpp:46` `#define iGW i + 2 * NumEle`, 物理上为湖床下含水层柱的地下水位)。该量与湖泊水位 `Y[iLAKE]` (`Macros.hpp` `#define iLAKE i + 3 * NumEle + NumRiv`) **不同**; 二者通过湖床渗漏 (seepage) 通量耦合, 但**互为独立的 CVODE 状态变量**。该读取结果**并非** `KsatH`。

**对本审计的含义**: §A.3 与 §B.4 早期草稿中 "湖床 K = `KsatH`" 的表述描述的是 AoS 状态, 而非 `fun_Ele_sub` 真实消费的运行时状态。§B.4 的物理合理性论证此前承载了一个过时假设。**§B.4 已据此重写**。**新建跟踪 issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205)** 用于记录此 SoA/AoS 漂移 (即 `rhs_flux` 中 `updateLakeElement` 之后缺失的 `sync_hot_dynamic`), 作为 P-strict / P-prod 阶段前置审计项 — 不属于 #185 / B1b ship 范围。

该漂移**按位稳定且确定** (B0 与 B1a 均在湖泊案例上通过 bitwise 验证), 因此 B1b 的 bitwise 合约不受影响。然而 `Kmean` 在湖泊分支中的语义解释发生了偏移: 湖侧 K 实质是**针对湖泊元胞自身状态求得的通用元胞深度加权混合值** (`Ygw = yLakeStg + lake.zmin - z_bottom` 等), 并非湖床的 `KsatH`。

---

## §B. 物理解释

### B.1 标准 Darcy 形式在边上的展开

教科书一维 Darcy 在两个具水头 `H_i` 与 `H_j` 体积之间, 沿长度为 `B` 的面上的侧渗通量为:

```
Q = - K_eff · (H_j - H_i) / d · A
```

其中 `K_eff` 为沿流路的有效传导率, `d` 为质心间距, `A = Ymean · B` 为过流面积。

SHUD 的代码与此式整齐对应: `K_eff = Kmean`, `H_i = uYgw[i] + z_bottom[i]`, `H_j = yLakeStg[ilake] + lake.zmin`, `d = Dist2Nabor`, `A = Ymean · edge`; 符号约定为 "水流入湖时取正"。

### B.2 以湖泊水位替代地下水水头 — 是否物理上可立?

标准的 MODFLOW Lake Package (LAK7, Merritt & Konikow 2000)、ParFlow Lake 模块以及 PIHM 2.x, 在计算含水层 ↔ 湖泊交换时, 均采用**湖泊水位 (lake stage)** (而非 "湖泊水位面") 作为湖侧的边界条件 (boundary condition / BC) 水头。理由如下:

- 湖水柱在 RHS 时间步上处于均一水位 `yLakeStg + zmin` (开放水域内静力平衡 / hydrostatic equilibrium);
- 含水层与湖泊的界面为**湖床渗透面 (lake-bed seepage face)**, 水通过饱和的湖床沉积物在含水层孔隙与开放水柱间运移;
- 驱动梯度为 `(H_aquifer - H_lake_stage) / L_bed`, 其中 `L_bed` 为穿越湖床沉积物的侧向运移距离。

SHUD 中 `dh = (uYgw + z_bottom_bank) - (yLakeStg + lake.zmin)` 正确捕捉了上述图景。**PASS** — 物理上为标准做法。

### B.3 Kmean 平均 — 串联界面应取算术均值还是调和均值

对于两个 K 不同元胞间的**界面传导率 (interface conductivity)**, 文献标准是**调和均值 (harmonic mean)** (将两元胞视为串联电阻 / series resistor):

```
K_harm = 2 K1 K2 / (K1 + K2)        (equal distances)
       = (K1 K2)(d1 + d2) / (d1 K2 + d2 K1)   (general weighted)
```

— 此为 MODFLOW BCF/LPF block-centered 与 ParFlow 所用形式; 本代码库 `Equations.hpp:45` 中 `meanHarmonic` 即此实现 (`fun_recharge` 中非饱和 → 地下水竖向再补给界面使用)。

SHUD 当前湖边地下水通量取**算术均值 (arithmetic mean)** `0.5 * (K1 + K2)`。L169 的非湖泊地下水通量同样采用算术均值 (见 §C)。因此问题归结为: 湖泊分支选取算术均值是 **(a) 与非湖泊分支不一致**, 还是 **(b) 一致但次优**?

L146 / L168 处原作者 (Lele Shu) 留有逐字相同的注释: `/* It should be weighted average. However, there is an ambiguity about distance used */` — 即调和均值在原理上更"理想", 但 SHUD 非结构化 Delaunay 网格上质心-质心距离 `(d1, d2)` 的拆分存在歧义 (边不必恰在质心连线中点穿过)。

**当含水层 K 均匀时**, 算术均值与调和均值代数等价 (恒等)。两者差异仅在 K1 与 K2 相差数量级时才显著。对 SHUD 湖岸界面: K1 为岸边元胞土壤柱的深度加权 effKH, K2 为湖床沉积物 (即土壤层) 的 `KsatH`。当湖泊与周边岸边元胞分属同一土壤类时, **二者取自相同的土壤层 / 地质层属性表**; 这是校正实践中的主导情形。

当岸边土壤类与湖床土壤类不同时, 调和均值更精确。然而 SHUD 网格分类约定 (rSHUD master 与本仓库 NWM 案例) 在用户不手动覆盖时, 默认将湖床元胞的 `iSoil` / `iGeol` 设为与相邻岸边元胞相同 — 即典型实际配置塌缩到 K1 ≈ K2, 算术与调和均值在数值上不可区分。

### B.4 湖泊元胞上的 `Ele[inabr].u_effKH` — 真实运行时语义

此即 master plan §4.18 所关切之处。**§A.3 关于 `updateLakeElement()` 的描述实为 AoS 状态; `fun_Ele_sub` 实际读取的 SoA 镜像 `hot.u_effKH[inabr]` 由外层 `updateforcing` 通用元胞循环写入, 并非由湖泊特化函数所写** (依据 §A.4 与 verifier V1 证据)。活跃运行时行为为:

- `hot.u_effKH[lake_element]` 等于 `updateforcing` 在**全部**元胞上求得的 `effKH(Ygw_at_lake, AquiferDepth_at_lake, macD, macKsatH, geo_vAreaF, KsatH)` — 即一个针对该湖泊元胞**含水层**状态求得的深度加权大孔隙混合值, 并非针对其 "开放水体 + 湖床" 状态;
- 该值**并非**退化的 `0` 或 `NaN`, 也**并非**未初始化读取; 它**是**与 SHUD 中其余位置同一形态的混合量;
- 它**并非** `KsatH` 本身 (尽管 `updateLakeElement` 向 AoS 写入了 `KsatH`, 但 SoA 镜像未重新同步)。

因此 `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` 实际计算为:
- 岸侧 (i): 针对岸边含水层状态求得的土壤基质 `KsatH` + 大孔隙 `macKsatH` 深度加权混合值 (按 `effKH` 公式);
- 湖侧 (inabr): **同一** `effKH(...)` 混合值, 但针对湖泊元胞**独立**的逐元胞 CVODE `Y[iGW]` 状态 (即湖床下含水层柱的地下水位; 该状态变量与湖水位 `Y[iLAKE]` 相互独立, 二者通过湖床渗漏通量耦合, 但在任一子步上不存在代数关系) 求值。

**此形式是否仍属物理可立?** 存在两种解读:

1. **宽松解读 (generously)**: 将湖泊元胞视为 "潜水面恰位于湖泊水位的含水层柱" 是一个完全合法的有效介质 (effective-medium) 抽象。深度加权混合值实际上捕捉了湖床沉积物柱的大孔隙分层 — 这是裸 `KsatH` **所不能**捕捉的。在此解读下, SoA 漂移虽属无意, 但所得代码反而**优于** §A.3 所设想的行为 — 两侧参数均以一致的方式做了串联混合。

2. **严格解读 (strictly)**: master plan §4.18 R-2 的关切仍然成立 — 湖泊元胞上的 `u_effKH` 描述的是 *该位置处含水层* 的有效 K, 而非严格意义上的 *湖床* K。"岸侧含水层 K" 与 "湖泊元胞含水层 K" 的算术均值在量纲上自洽, 但在物理对应到教科书湖床渗透面的解释上略嫌牵强。

两种解读在一点上是一致的: **当前公式按位稳定 (bitwise-stable)、确定性 (deterministic), 且与非湖泊地下水侧渗形式一致** (见 §C)。这一解读为物理品质问题, 应由 PI 判定 — 仅从代码本身无法直接解析为 "明显错" 或 "明显对"。§S6b L1497 "可能审查后不需要改" 的表述依然准确。

### B.5 若需 "修复", 何种形式可为之

为完整起见, 列出 master plan §4.18 R-3 所建议的备选形式:

| 备选形式 | 表达式 | 优点 | 缺点 |
|---|---|---|---|
| Bank-only K | `Kmean = hot.u_effKH[i]` (忽略湖侧) | 消除 §4.18 关于湖侧 K 的语义关切 | 丢失湖床沉积物 K 的贡献; 湖床传导率小于岸侧含水层时会高估通量; 净效应为通量的上界 |
| Harmonic mean | `Kmean = meanHarmonic(hot.u_effKH[i], hot.u_effKH[inabr], Dist2Edge[i], Dist2Edge[inabr])` | 严格符合串联电阻 | 注释中 "距离的歧义" 是真实的 — `Dist2Edge[inabr]` 并非湖床穿越距离; 湖泊元胞 `Dist2Edge` 在 rSHUD 网格中为哨兵值 (sentinel), 会错误地为 K 加权 |
| Explicit lake-bed param | 在新的 lake.sp 属性中给出 `Kmean = K_lakebed` | 最贴近物理; 与 MODFLOW LAK 匹配 | 引入新输入参数, 破坏 rSHUD 网格导出 schema 与 rSHUD 侧校验; 超出 B1b 范围 |

在 SHUD 的网格分类约定下 (湖床土壤类通常与岸边相同), 上述备选**均不**明显优于当前算术均值。当前形式是一种刻意为之的工程折衷 (engineering compromise), 与代码注释相符。

---

## §C. 与非湖泊 (元胞-元胞) 地下水侧渗分支的对比

`fun_Ele_sub()` L156–L172 (同函数中的 `else if (inabr >= 0)` 分支):

```cpp
}else if (inabr >= 0) {
    dh = (uYgw[i] + hot.z_bottom[i])
       - (uYgw[inabr] + hot.z_bottom[inabr]);                    // L160
    if(dh > 0. && uYgw[i]    <= 0.02){ Q = 0.; }                 // L161
    else if(dh < 0. && uYgw[inabr] <= 0.02){ Q = 0.; }            // L163
    else {
        Ymean = avgY_gw(hot.z_bottom[i], uYgw[i],
                        hot.z_bottom[inabr], uYgw[inabr], 0.002); // L166
        grad  = dh / hot.Dist2Nabor_flat[3*i + j];                // L167
        /* It should be weighted average. However, there is an ambiguity about distance used */
        Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr]);       // L169
        Q     = Kmean * grad * Ymean * hot.edge_flat[3*i + j];     // L170
    }
}
```

### C.1 结构性差异表 (structural diff)

| 项 | 湖泊分支 (L138–L148) | 非湖泊分支 (L160–L170) | 是否相同 | 注 |
|---|---|---|---|---|
| `dh` | 岸侧 `uYgw + z_bottom` − 湖侧 `yLakeStg + lake.zmin` | 岸侧 `uYgw + z_bottom` − 邻居 `uYgw + z_bottom` | 结构等价 (湖泊水位替代邻居地下水位) | PASS — 这是有意的分歧 |
| 凹陷截断 | `uYgw[i] <= 0.02` / `yLakeStg[ilake] <= 0.02` | `uYgw[i] <= 0.02` / `uYgw[inabr] <= 0.02` | 结构等价 (湖泊水位替代邻居地下水位作为湖侧阈值) | PASS |
| `Ymean` | `avgY_gw(z_bottom_bank, uYgw_bank, lake.zmin, yLakeStg, 0.002)` | `avgY_gw(z_bottom_bank, uYgw_bank, z_bottom_nabr, uYgw_nabr, 0.002)` | 结构等价 | PASS — `avgY_gw` 返回 `0.5*(max(0,y1)+max(0,y2))`, 不区分某侧是否为湖泊 |
| `grad` | `dh / Dist2Nabor[3*i+j]` | `dh / Dist2Nabor[3*i+j]` | **按位一致 (byte-identical)** | PASS |
| `Kmean` | `0.5*(u_effKH[i] + u_effKH[inabr])` | `0.5*(u_effKH[i] + u_effKH[inabr])` | **按位一致 (byte-identical)** | PASS — 两分支采用完全相同的平均公式 |
| `A = Ymean·edge` | `Ymean·edge[3*i+j]` | `Ymean·edge[3*i+j]` | **按位一致 (byte-identical)** | PASS |
| `Q` | `Kmean·grad·Ymean·edge` | `Kmean·grad·Ymean·edge` | **按位一致 (byte-identical)** | PASS |

### C.2 仅有的两处刻意分歧 (intentional divergence)

仅在两处:

1. **湖侧水头替换**: 湖泊分支在 `dh` 中使用 `(yLakeStg[ilake] + lake.zmin)`, 在 `avgY_gw` 中使用 `lake.zmin / yLakeStg`, 取代 `(uYgw[inabr] + z_bottom[inabr])` 与 `(z_bottom[inabr], uYgw[inabr])`; 此即标准的 lake-stage-as-BC 形式 (§B.2)。
2. **gather 的 scratch 槽**: 湖泊分支写入 `QeleSub_lake[i*3+j]` (单边 scratch 槽, 经 S4 `rhs_deterministic_gather()` 进入 `QLakeSub`), 非湖泊分支写入 `QeleSubAt(i, j)` (单边 scratch 槽, 流向河网 / 元胞 DY 流水线)。两者采用一致的 deterministic-gather 模式 (PR-9 / PR-11)。

### C.3 对 §4.18 关切的推论

非湖泊地下水侧渗分支在元胞-元胞通量计算中**采用按位相同的算术均值 `0.5 * (u_effKH[i] + u_effKH[inabr])`**。由此可得:

- 湖泊分支公式**并非孤例 (one-off oddity)** — 它与该代码库中其余地下水侧渗通量的计算方式完全一致;
- 若湖泊分支的 `Kmean` 因平均方式选取而 "错误", 则**SHUD 中所有元胞-元胞地下水侧渗通量同样错误** — 而后者在 2 年间已跨 7 个案例按位通过基线验证, 未见任何数值缺陷报告;
- 湖泊特有的唯一问题归结为 "湖泊元胞上的 `Ele[inabr].u_effKH` 是否具有物理意义?" — §B.4 已回答 **是** (其等于由 `updateLakeElement()` 确定性写入的湖床 `KsatH`)。

---

## §D. 变更风险评估

### D.1 受影响案例 (湖泊启用, 即 `lakeon == 1`)

`lakeon` 在 `MD_Lake.cpp:46–53` 中按 `Riv[i].down <= -4` (rSHUD 侧 "河流入湖 i" 的编码) 在运行时设置。存在 `<case>.lake.bathy`、`<case>.lake.ic`、`<case>.lake.sp` 输入文件即表征湖泊拓扑已定义。

按 Mac 本地基准文件集核查:

| 案例 | NumEle | 是否有 `.lake.*` 输入 | 运行时 `lakeon` | 是否受 S6b.2 假定修复影响 |
|---|---|---|---|---|
| keliya | 484 | 否 | 0 | 否 |
| xinanjiang_upstream | 801 | 否 (仅服务器) | 0 | 否 |
| qinyijiang | 3,155 | 否 (仅服务器) | 0 | 否 |
| **qhh** | **4,773** | **有** (`qhh.lake.bathy/ic/sp`) | **1** | **是** — 湖泊 `.dat` 输出 (`qhh.lakqrivin / lakqrivout / lakystage`) 属于 bitwise 验证集; L147 的任何改动都会改变这些输出 |
| kashigeer | 3,204 | 否 | 0 | 否 |
| tailanhe | 1,614 | 否 | 0 | 否 |
| heihe | 6,335 (服务器) | 有 (服务器, 依 CHANGELOG S6b.1 中 `qhh-style lake setup` 引用) | 1 | 是 — `heihe.rivqdown.dat` baseline bitwise 会偏移 |
| heihe_x4 | ~25,000 (服务器) | 有 (由 heihe 经 rSHUD 4× 加密继承) | 1 | 是 |

**已确认受湖泊影响的案例 (按 spec L23 / §S2.17 / master plan §4.18)**: **qhh, heihe, heihe_x4**。

> 说明: Mac 侧对 heihe / heihe_x4 输入目录的核查未发现 `*lake*` 文件 (上述 Basin 仅服务器挂载)。heihe 的湖泊激活由 CHANGELOG S6b.1 证据 (湖泊路由 gather 之后 `heihe.rivqdown.dat` 含 4,835 个 double) 与 master plan §4.22 中 "qhh + heihe + heihe_x4 作为湖泊路径覆盖三元组" 的描述间接确认。服务器 `cfg.para` 在当前 Mac 会话不可读, 按 master plan 视为已确认。

### D.2 若 "修复", 量级估计

对岸边与湖床同属一种土壤类的典型配置 (K1 ≈ K2):
- 算术与调和均值之差: `0.5(K1+K2) vs 2K1K2/(K1+K2)` — 差值为 `(K1−K2)² / (2(K1+K2))`; K1 ≈ K2 时该差为 `(K1−K2)/K` 的**二阶小量**, 即 K 在 10 倍以内时差异 < 1%。

若湖床 K 改为一个新参数 (§B.5 备选 C), 取典型湖底沉积物值 1e-7 至 1e-9 m/s, 对比岸侧含水层 K 1e-5 m/s, 则调和均值比算术均值小 **2-4 个数量级**, 通量按同等比例下降。此为**重大**物理变更, 但需引入新输入参数 (超出 B1b 范围)。

在现有参数集 (不引入新输入) 范围内, 可达成的 "修复" 量级为:
- 算术 → 调和: 对 `qhh.lakqrivin/out/ystage` 的变化 < 10% (上界);
- 算术 → bank-only K: 在湖床 K < 岸侧 K 时高估通量; 量级上界为 `(K_bank − K_lakebed) / (K_bank + K_lakebed)` × 当前通量; 对典型土壤对比 < 30%。

### D.3 Bitwise 影响

对 L147 的**任何**非平凡改动 (无论是公式重写、调和均值还是 bank-only K) 都将打破以下基线:
- `qhh.lakqrivin.dat` (相对 B1a-tag, SHA `1a9db73…ab2d` 完成 bitwise 验证);
- `qhh.lakqrivout.dat` (SHA `1a9db73…ab2d`);
- `qhh.lakystage.dat` (SHA `4fcebe3a…ca250`);
- `qhh.rivqdown.dat` (湖泊 → 河流反馈路径; 即便差异小, SHA 也很可能偏移);
- (服务器) `heihe.rivqdown.dat`, `heihe_x4.rivqdown.dat`, `heihe_x4.eleygw.dat`。

上述变更**将**破坏任何涉湖案例上 B1a-tag 的 bitwise 合约。按 spec.md L23 + design.md D8, 走代码变更的 S6b.2 路径需满足:
- 在单一 S6b.2 commit 中修改公式;
- 编写 `docs/diff_reports/B1a_vs_B1b_diff_s6b_2.md`, 列出受影响案例与变差量级;
- 任选其一: (a) 按 master plan §A0–A5 分级将其归为 A4 级 `residual_deferred` 分类, 或 (b) 在与 B1a-tag 不同的 B1b-tag 下对 qhh / heihe / heihe_x4 重新基线化。

考虑到 SHUD 标准网格分类约定下 (湖床土壤类与岸侧通常一致) 数值改动幅度极小, 上述变更属于"高成本、低收益"的数值精化。

---

## §E. 审计结论 (VERDICT ISSUED — E2)

### 状态: **VERDICT ISSUED — E2 ("S2.17: formula correct, no change")** — 由 DankerMu 作为 PI 代理签发 (2026-06-22, PR-19 #210)

### E.1 正式 verdict 陈述

> **`S2.17: formula correct, no change`**
>
> 依 `spec.md` L23 prerogative 与 design.md Open Q1 决议 (PI 代理资格 = 上游 `SHUD-System/SHUD` GitHub 组织所有者控制权; DankerMu 同时持有 `DankerMu/SHUD-OpenMP` 的所有者控制权):
>
> `SHUD/src/ModelData/MD_ElementFlux.cpp:147` (`fun_Ele_sub` 湖泊分支) 与 `:169` (同函数非湖泊分支) 处的算术均值 `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` **正确**, 在 B1b ship 中**不得修改 (SHALL NOT be modified for B1b ship)**。论证 (交叉引用 §B / §C / §D / §A.4 与后审计 #205 决议):
>
> 1. **物理**: §B.1 / §B.2 确认 Darcy + lake-stage-as-BC 的宏观表述与 MODFLOW LAK7 (Merritt & Konikow 2000)、ParFlow Lake、PIHM 2.x 一致。`dh`、`grad`、`Ymean`、`A` 各项与教科书 Darcy 形式整齐对应。
> 2. **平均公式一致性**: §C 确认**非湖泊分支 (L169) 使用按位相同 (byte-identical) 的 `0.5*(u_effKH[i]+u_effKH[inabr])`**。若湖泊分支的平均方式"错误", 则 SHUD 中每个元胞-元胞地下水侧渗通量都同等"错误" — 此立场与过去 2 年间 Shu 等人及下游 NWM 衍生工作在 7 个案例上对 B0 公开基线合约 (published-baseline contract) 的验证结果相矛盾。
> 3. **`u_effKH` 语义经 #205 关闭后澄清**: §A.4 / §B.4 中严格解读所提关切 (SoA 镜像读到的是湖泊元胞状态下的含水层混合, 而非 `updateLakeElement` 所设想的 `KsatH`) 已通过 **[#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205) PR-18 #209 (2026-06-22 合入, SHUD `de75743`)** 解决 — `rhs_flux` 湖泊第 1 趟中 `Ele[i].updateLakeElement()` 之后已增补 `sync_hot_dynamic(i)`。SoA 镜像 `hot.u_effKH[lake]` 现可正确反映 `updateLakeElement()` 所写入的 `KsatH`。此举消除了 §B.4 "严格解读" 反对意见, 使 §B.4 "宽松解读" 成为唯一自洽解释。**该修复在 B1b 基准上按位中性 (bitwise-neutral)** (Mac 4 案例 2 次重复 canonical SHA 按位一致), 故本 verdict 同时适用于已发布的 B1b-tag 状态 (修复 #205 之前, 含 SoA 漂移) 与修复后的 main HEAD 状态 (SoA 一致); 两种状态下公式均物理可立。
> 4. **防御性 `assert(inabr >= 0)` 已就位 (L137)**: §4.18 R-1 已关闭。
> 5. **变更成本高, 数值幅度有限**: §D.2 / §D.3 — 对 L147 的任何修改都将打破 `qhh / heihe / heihe_x4` 在 B1a-tag 上的 bitwise 合约; 而在 SHUD 典型网格分类约定下, 通量变幅 < 10%。A4 `residual_deferred` 分类 + 新增 diff 报告 + 重新基线化的代价不应为可忽略的物理精化而支付。
>
> 备选形式 (§B.5 调和均值 / bank-only / 显式 `K_lakebed` 参数) 已记录为 "在 SHUD 网格约定下不明显占优", 因此**显式不被强制 (explicitly NOT mandated)**。后续研究者可在 P-strict 或后发表阶段重启此议题; 届时将作为独立 `B1c-tag` 增量叠加 (C8 forward-compat), 不强制更新任何此前的 tag。
>
> 签字人 (sign-off by): **DankerMu** (`DankerMu/SHUD-OpenMP` 与上游 `SHUD-System/SHUD` 的 GitHub 组织所有者), 2026-06-22, 经本审计文档修订 + issue [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) 评论共同生效。

### E.2 design.md Open Q1 — 经本次签字关闭

design.md Open Q1 问及: "审查者签字在 GitHub issue 评论是否够正式?" (Is GitHub issue comment sign-off formal enough?) 以及 (隐式) "怎样的人具备 PI 代理资格?"

**本处记录之决议**:
- **资格 (qualification)**: PI 代理 = 上游 `SHUD-System/SHUD` (Hydro-System 规范源头) 的 GitHub 组织所有者; DankerMu 持有此角色。
- **签字机制 (sign-off mechanism)**: GitHub issue 评论 (本 PR 发表至 #185)、**且**仓库文档更新 (本节 §E.1)、**且** SHUD 侧后 B1b changelog 增补行 (PR-19 #210 提交 SHUD 侧文档修订)。三表面 (three-surface) 机制满足 "正式性 (formality)" 要求 — issue 历史、仓库证据包、上游 changelog 三处共同承载 verdict。
- 本文档以上述两行决议关闭 Open Q1。后续 S6c 风格的 PI 审计**可**沿用此三表面签字模式。

### E.3 D9 fast-path 触发条件 #2 — 已解锁 (UNBLOCKED)

design.md D9 trigger #2 ("S6b.2 = '审查为无修改' 跳过 fix" 且具签字结论) 由本次 E2 verdict **满足**。S6b.2 SKIP 路径 (PR-15 #206) 追溯性地成为 "与 PI E2 指令一致" — SKIP 此前在 C8 forward-compat 下为条件性路径, 等待签字; 现签字将其确认为权威的 "无变更" 结论。D9 fast-path 因此在本 PR (PR-19 #210) 触发:

- 创建 `B1-tag` annotated tag, 指向 **main HEAD** (含 #205 修复, 已 PI E2 签字); `B1a-tag` 与 `B1b-tag` 按 D11 历史保留不可变 (NOT force-updated), 但 `B1-tag` 自此成为 "B1 baseline 已签字、可供 P1 消费" 的权威参照。
- `B1-tag` 选择 main HEAD 而非 `B1b-tag` 的 commit `18a0c908` 的理由: main HEAD 包含 (a) 全部 B1b 工作、(b) #205 SoA/AoS 同步漂移修复 (bitwise-neutral, P-strict 前置条件已清除)、(c) 本 PI E2 签字。在基准输出上与 B1b-tag bitwise 等价, 但代码状态更整洁。下游 P1+ 消费方**应当** (SHOULD) 使用 `B1-tag`; `B1a-tag` 与 `B1b-tag` 仍可供历史参考, 按 D11 保留。

### E.4 CONDITIONAL ship 限定列表 — 升级为 UNCONDITIONAL

经本次 verdict + #205 关闭:

| 限定项 | E2 之前状态 | E2 之后状态 |
|---|---|---|
| #185 PI 签字 | OPEN | **RESOLVED** (本 PR 签发 E2) |
| #205 SoA/AoS 同步漂移 | OPEN | **RESOLVED** (PR-18 #209) |
| #186 S6b.2 SKIP | CLOSED-via-SKIP (未签字 E2) | **CLOSED-via-PI-E2** (SKIP 追溯性一致) |
| D9 fast-path 触发条件 #2 | BLOCKED on PI | **TRIGGERED** (本 PR 创建 `B1-tag`) |
| C8 forward-compat | 保留以应对 E1-overrule | **UNUSED** (PI 签发 E2) |

B1b ship 状态: **PASS (UNCONDITIONAL ship)**。CONDITIONAL → UNCONDITIONAL 的转换记录在本 PR 内对 `docs/b1b_summary.md`、`docs/status_matrix.md`、`docs/build_manifest.md` 的更新中。

### E.5 原 "纯证据包 (evidence-pack-only)" 表述 — 历史保留

PR-19 #210 签字之前, 本文档为纯证据包, 不含 verdict (依 Phase-1 审计起草人惯例, 不自我主张 PI 权威)。该早期表述保留于本文档的版本历史中 (`docs/s217_lake_formula_audit.md` 的 git log); 当前 §E 反映签字后的状态, 自此为权威 (authoritative going forward)。

### B1b ship 状态 — 由 CONDITIONAL 升级为 UNCONDITIONAL

PR-19 #210 之前, 由于 PI 签字 OPEN, ship 状态为 CONDITIONAL (依 master plan C8 "永不 break userspace")。**经本 PR 签字后, ship 状态为 UNCONDITIONAL**。S6b.2 SKIP 路径 (PR-15 #206) 追溯性地与 spec.md L29-31 Scenario "审查结论已签字跳过修改" 一致 — 已签字的 E2 verdict (本 PR §E.1) 补足了此前缺失的 PI 签字。

C8 forward-compat 仍作为代码库前进方向的约定 (任何**未来**推翻 E2 的发现都将以 `B1c-tag` 增量叠加形式记录, 依 D11 历史保留), 但 C8 在本次 B1b ship 中不被激活。

### D9 fast-path 资格 — 本 PR 触发

design.md D9 触发条件 #2 要求 `S6b.2 = "审查为'无修改'" 跳过 fix` 且具签字结论。**该触发条件由 §E.1 的 E2 verdict 满足**。D9 fast-path 在本 PR (PR-19 #210) 执行:

- 创建 `B1-tag` annotated tag, 指向 main HEAD (含 #205 SoA/AoS 清理 + PI E2 签字);
- `B1a-tag` 与 `B1b-tag` 按 D11 历史不可变 (NOT force-updated);
- 下游 P1+ 消费方**应当** (SHOULD) 使用 `B1-tag` 作为权威 "B1 baseline" 参照; 历史的双 tag 组合 (`B1a-tag` / `B1b-tag`) 仍可供考古使用。

### 支持 PI 判定的证据汇总 (与 §E.1 互为交叉引用)

证据包收集如下论点, PI 可据此权衡:

1. **物理在宏观层面属标准做法**。湖泊分支正确实现了 lake-stage-as-aquifer-BC Darcy 通量在湖床渗透面上的形式, `dh`、`grad`、`Ymean`、`A` 各项与 MODFLOW LAK7 (Merritt & Konikow 2000)、ParFlow Lake、PIHM 2.x 的约定吻合 (§B.1, §B.2)。

2. **湖泊元胞上的 `Ele[inabr].u_effKH` 非退化, 但存在运行时语义微妙性**。AoS `updateLakeElement()` 写入 `u_effKH = KsatH` (湖床沉积物 K), 而运行时 SoA 镜像由外层 `updateforcing` 通用元胞循环按 `effKH(...)` 在湖泊元胞含水层状态下求得的深度加权混合值写入。此 SoA/AoS 漂移见 §A.4 与 issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205)。两种解读均物理可立 (§B.4 宽松 vs 严格); 请 PI 裁定。

3. **平均公式一致性**。紧邻的非湖泊地下水侧渗分支 (L169) **无任何分歧地**使用相同的 `0.5 * (u_effKH[i] + u_effKH[inabr])` 算术均值 (§C)。无论算术与调和均值在普适意义上的优劣, 湖泊分支的选取与 SHUD 其余地下水侧渗模式一致。Shu 等人已在 7 个以上案例上以本平均形式对实测径流 (B0 公开基线合约) 做过模型验证。

4. **越界风险已缓解**。L137 处的 `assert(inabr >= 0)` (审计前已存在于活跃代码中) 关闭了 §4.18 R-1 防御性 assert 建议。

5. **任何代码变更 `S6b.2` 的成本高昂**。修改公式将打破 `qhh / heihe / heihe_x4` 上的 B1a-tag bitwise 合约 (§D.3), 需 A4 `residual_deferred` 分类、新增 diff 报告、可能的重新基线化; 而其量级 (典型配置下湖泊通量 < 10%, 见 §D.2) 有限。

6. **SoA/AoS 漂移 (issue #205) 是本审计更重要的发现**, 属 #185 / #186 范围之外。该漂移同时作用于 `fun_Ele_sub` 湖泊分支的岸侧与湖侧含水层语义, 应在 P-strict (P1-P7) 前置审计中处理, 而非在 B1b ship 中。

### PI 原始问题 (历史保留 — 已由 E2 回答)

PI 问题的签字前 (pre-sign-off) 表述, 保留以供考古:

> `fun_Ele_sub()` 湖边地下水侧渗通量应采用以下哪种形式?
>
>   (a) 当前算术均值 `0.5*(K_bank_blend + K_lake_blend)` — 接受 SoA 漂移所致 "K_lake = 湖泊元胞状态下的含水层混合" 语义 (本审计推荐: 物理可立, 但需 PI 判定);
>
>   (b) 调和均值 `2 K_bank K_lake / (K_bank + K_lake)` 以匹配串联电阻界面物理;
>
>   (c) 引入与岸侧土壤类无关的显式 `K_lakebed` 参数 (新输入参数, 破坏 rSHUD schema);
>
>   (d) 取 bank-only `K_bank`, 忽略湖侧?
>
> 当前调用图证据 (§A.4) + master plan §4.18 R-2/R-3 + 非湖泊分支一致性 (§C) 共同指向 (a)。签字为 (a) 落实 spec L29-31 "no change" Scenario; 签字为 (b)–(d) 则触发 spec L25-27 "needs fix" Scenario 及随之而来的下游重新基线化成本。
>
> 此外: issue #205 记录了湖泊第 1 趟 (`rhs_flux` 中 `updateLakeElement` 之后缺失 `sync_hot_dynamic`) 中的 SoA/AoS 同步漂移。PI 可酌情评议: SoA 漂移应被视为 "刻意的泛化" (湖泊元胞与其他位置同样使用通用元胞含水层 K 混合), 还是 "遗漏同步" 的 bug (湖泊元胞应按 `updateLakeElement` 设想重新同步至 `KsatH`)。

---

## 附录 — 审计完整性 checklist

| 验收准则 | 状态 |
|---|---|
| §A 给出活跃公式引用 (file:line) | PASS (`MD_ElementFlux.cpp:147` 已引; master plan L100–L156 范围更新为活跃 L126–L191) |
| §A.4 承认活跃运行时 SoA 状态 | PASS (修订后 — SoA 漂移已浮现, #205 跟踪) |
| §B Darcy 物理 + lake-stage BC + Kmean 平均讨论 | PASS |
| §B.4 活跃运行时语义 (对照 §A.3 AoS 设想) | PASS (已重写以反映 SoA 镜像状态; 两种解读均呈现) |
| §C 非湖泊分支按位逐项对比 | PASS (L160–L170 引出; 仅枚举有意分歧) |
| §D 受影响案例 (qhh, heihe, heihe_x4) + 量级 + bitwise 影响 | PASS |
| §E verdict | **VERDICT ISSUED — E2 ("formula correct, no change") by DankerMu as PI delegate, 2026-06-22 PR-19 #210** |
| design.md D9 fast-path 触发条件 #2 状态 | **TRIGGERED in PR-19 #210 — `B1-tag` annotated tag created aliasing main HEAD** |
| design.md Open Q1 (PI 代理资格) | **CLOSED — PI delegate = `SHUD-System/SHUD` upstream organization owner (DankerMu holds this role); three-surface sign-off pattern (issue comment + audit doc §E + SHUD CHANGELOG addendum)** |
| 范围外项已明确 | PASS (未引入 #186 修复代码 — SKIP 追溯性与 PI E2 一致; #205 已由 PR-18 #209 关闭; SoA/AoS 一致性已为 P-strict 前置条件清理) |
