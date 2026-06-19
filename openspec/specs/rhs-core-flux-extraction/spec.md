# rhs-core-flux-extraction Specification

## Purpose
TBD - created by archiving change s1-rhs-core-extraction. Update Purpose after archive.
## Requirements
### Requirement: rhs_flux pure carry-over from serial f_loop

`SHUD/src/Model/MD_rhs_core.cpp` 新增的 `rhs_flux()` 函数体 SHALL 是 serial `f_loop()` 的**逐行原样搬运**（`SHUD/src/ModelData/MD_f.cpp:11-74 (f_loop)`，PR #43 后实际函数体范围 L11 → L74 含 closing brace，包含 #ifdef SHUD_DUMP_RHS 探针块 + after-PassValue no-op 钩子）。逐行 diff SHALL 满足：

- 任何**逻辑分支** / **循环条件** / **比较运算符** / **赋值** / **函数调用参数** 与 legacy `f_loop()` **完全一致**
- **不重命名**任何局部变量、循环索引、临时量（保持 `i` 不改 `idx` / `iele` 等）
- **不调整调用语义**：每个 `fun_Ele_*` / `fun_Seg_*` / `Flux_RiverDown` / `Ele[i].updateLakeElement` / `Ele[i].updateElement` / `PassValue` 的参数顺序与 legacy 完全一致
- **不引入新的局部变量**用于"清晰化"或"重构"（如不能把 `Ele[i].iLake - 1` 提取成 `int ilake_idx`）
- **不内联或外提**任何子表达式

S1b 的唯一目标是"搬运不变 + 逐步 bitwise 验证"，任何"顺手清理"都构成范围越界，必须留待 S2 / S3 / S5 / S6。

#### Scenario: Diff legacy f_loop vs new rhs_flux

- **WHEN** 对比 `SHUD/src/ModelData/MD_f.cpp:11-74` 中 legacy `f_loop()` 函数体（不含函数签名与 `#ifdef SHUD_DUMP_RHS` 块）与 `SHUD/src/Model/MD_rhs_core.cpp` 中 `rhs_flux()` 函数体
- **THEN** 二者 SHALL 在 token 级别**字符完全一致**（允许的差异仅限于函数签名命名空间限定 `Model_Data::f_loop` → `Model_Data::rhs_flux`）
- **AND** PR description SHALL 附 `diff legacy_block.cpp new_block.cpp` 命令的零输出截图作为搬运证据

#### Scenario: No sub-function split inside rhs_flux

- **WHEN** code review 检查 `rhs_flux()` 的内部结构
- **THEN** rhs_flux 体内 SHALL **不存在** `rhs_element_vertical()` / `rhs_element_horizontal()` / `rhs_segment_compute()` / `rhs_river_downflow()` 等子函数调用——这些是 S3 的范畴，S1b PR 提前拆即违反 design D2
- **AND** rhs_flux 体内可调用的函数 SHALL 仅限 legacy `f_loop` 已调用的集合：`updateLakeElement` / `fun_Ele_lakeVertical` / `fun_Ele_lakeHorizon` / `f_etFlux` / `updateElement` / `fun_Ele_Infiltraion` / `fun_Ele_Recharge` / `fun_Ele_surface` / `fun_Ele_sub` / `fun_Seg_surface` / `fun_Seg_sub` / `Flux_RiverDown` / `min` / `max` / `PassValue`

### Requirement: Process order element-pass1 → element-pass2 → segment → river → lake-clamp → PassValue preserved exactly

`rhs_flux()` SHALL 严格按以下 6 步顺序执行，**任何**重排序、合并、拆分、循环倒置都被禁止（命名约定：master plan §S1b 简称 "lake → ET → element → segment → river → PassValue" 对应 `f_loop()` 实际代码中两次 element 大循环内的 lake-element + ET 子分支与 non-lake 处理；本 Requirement 标题展开为代码实测顺序——element-pass1 = lake-element + non-lake ET / updateElement / Infiltraion / Recharge；element-pass2 = lake horizon + non-lake surface / sub——与 master plan 简称等价但更贴近实现）：

1. **Element pass 1**（`for i in [0, NumEle)`）：lake 元素 → `updateLakeElement` + `fun_Ele_lakeVertical` + `qLakeEvap/qLakePrcp` 累加；非 lake 元素 → `f_etFlux` + `updateElement` + `fun_Ele_Infiltraion` + `fun_Ele_Recharge`
2. **Element pass 2**（`for i in [0, NumEle)`）：lake 元素 → `fun_Ele_lakeHorizon`；非 lake 元素 → `fun_Ele_surface` + `fun_Ele_sub`
3. **Segment pass**（`for i in [0, NumSegmt)`）：`fun_Seg_surface` + `fun_Seg_sub`
4. **River pass**（`for i in [0, NumRiv)`）：`Flux_RiverDown`
5. **Lake clamp pass**（`for i in [0, NumLake)`）：`qLakeEvap = min(qLakeEvap, qLakePrcp + yLakeStg)` + `max(0, ...)`
6. **PassValue**（`PassValue()` 调用，无参数）

任何"合并 pass 1 与 pass 2 为一次循环"、"把 segment pass 与 river pass 调序"、"把 lake clamp 上移到 element pass 1 之内"等微改 SHALL 视为破坏不变量，PR 不可合并。

#### Scenario: Reorder regression must fail bitwise gate

- **WHEN** 实验性修改 `rhs_flux()` 把 **element pass 1（步骤 1）与 element pass 2（步骤 2）** 调换顺序（注：PR #62 实测发现原方案 "segment↔river swap" 是 bitwise no-op —— segment pass 写 `QsegSurf`/`QsegSub`（仅 `PassValue` 消费），river pass 写 `QrivDown`/`QLakeRivIn`/`QLakeRivOut`（仅 `f_applyDY` 消费），两输出集 disjoint，swap 后 bitwise PASS 与负向测试目的相悖；改 element pass 1↔2 swap 与下一 Scenario "Pass 1 / Pass 2 cannot be merged" 同源）
- **THEN** keliya / xinanjiang_upstream / qinyijiang / qhh 任意一个 case 的完整 run 输出 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` SHA256 **不一致**，CI bitwise gate SHALL FAIL
- **AND** 该 negative test SHALL 以 patch 形式落地 `SHUD/tests/s1b_reorder_regression.patch`（**不入** main commit；patch + revert 一起留档）；S1b PR 描述 MUST 附以下命令：`git apply tests/s1b_reorder_regression.patch && make shud && ./shud qinyijiang`，并写明预期 fail SHA256（pinned 进 PR description）+ revert 命令 `git apply -R tests/s1b_reorder_regression.patch`

#### Scenario: Pass 1 / Pass 2 cannot be merged

- **WHEN** code review 检查 `rhs_flux()` 是否把两次 `for i in [0, NumEle)` 合并成一次循环
- **THEN** rhs_flux SHALL 保留**两次独立的 element 循环**（与 legacy `f_loop()` MD_f.cpp L13 与 L29 两个 for 一致），合并循环会改变 `updateElement` / `fun_Ele_surface` 之间 cross-element 数据可见性时序，破坏 bitwise

### Requirement: PassValue call position MUST equal legacy f_loop

`rhs_flux()` 内 `PassValue()` 调用 SHALL 位于**严格 6 步过程的最后一步**（即 river downflow → lake clamp → PassValue），与 legacy `SHUD/src/ModelData/MD_f.cpp:51`（`PassValue();` 紧随 `for (i = 0; i < NumLake; i++)` lake clamp 循环之后）位置精确一致。

提前调用（如把 PassValue 移到 segment pass 之前）或推迟调用（如移到 `rhs_apply` 入口）SHALL 视为破坏 flux 累加语义：

- master plan §4.7 已记录 `PassValue()` 会先清零 `QrivSurf` / `QrivSub` / `QrivUp` / `Qe2r_Surf` / `Qe2r_Sub`，再从 `QsegSurf` / `QsegSub` 重新累加
- 提前 PassValue 会使 `Flux_RiverDown` 计算时 `QrivSurf` / `QrivSub` 仍是上一时步状态
- 推迟 PassValue 会使 `f_applyDY` / `rhs_apply` 读到 `fun_Seg_*` 中被 `PassValue` 视为"死代码"的 `+=` 累积值（与 B0 不一致）

S1b 的 `rhs_flux()` 中 PassValue 仍是 legacy 的清零+累加实现，不在 S1b 改写；S3 才会替换为确定性 gather。

#### Scenario: PassValue placement bitwise probe

- **WHEN** 实验性修改 `rhs_flux()` 把 `PassValue()` 上移到 segment pass 之前
- **THEN** keliya 与 qinyijiang case 的完整 run 输出 SHALL 与 B0-tag SHA256 不一致，CI bitwise gate SHALL FAIL
- **AND** 该破坏路径的 stderr SHALL 不报错（属"沉默语义破坏"），故 bitwise gate 是唯一防线

### Requirement: S1b mixed-mode dispatch (rhs_update + rhs_flux new; rhs_apply legacy)

S1b 阶段 `USE_RHS_CORE=1` 编译时，`f.cpp` 入口 SHALL 经 `rhs_core(Y, DY, t)`（S1a 三参签名，**不**带 `ExecPolicy`）调度，调度内部 SHALL 按以下混合策略执行：

1. **`rhs_update()`**：走 S1a 已抽取的新路径
2. **`rhs_flux()`**：走 S1b 本阶段新抽取路径
3. **DY 写入**：仍 fallback 调原 `f_applyDY()`（legacy），等 S1c 抽取完成

混合模式存在的唯一理由是细粒度 A/B 验证：S1b 失败时能立刻定位到 `rhs_flux` 内部（rhs_update 已在 S1a 验证过），而不需在三函数耦合状态下定位。S1b 阶段 SHALL NOT 提前调用 `rhs_apply()` 桩函数（即使已存在），否则失败定位难度翻倍。

S1b 阶段 `USE_RHS_CORE=0`（默认）编译时，`f.cpp` 入口 SHALL 仍走原 `f_update` / `f_loop` / `f_applyDY` 三步串联路径，与 B0 二进制语义完全一致。

#### Scenario: USE_RHS_CORE=1 routes flux through rhs_flux

- **WHEN** `make shud USE_RHS_CORE=1` 编译后，单步 RHS 评估的执行栈被 gdb 断点 trace
- **THEN** 调用栈 SHALL 依次出现 `rhs_core` → `rhs_update` → `rhs_flux` → `f_applyDY`（注意第三步仍是 legacy 函数名，**不是** `rhs_apply`）

#### Scenario: USE_RHS_CORE=0 default keeps legacy path

- **WHEN** `make shud`（无 `USE_RHS_CORE` define）编译后跑 keliya 90 天
- **THEN** 输出 `.dat` 文件 SHALL 与 `git show B0-tag:benchmarks/keliya/B0_output/<file>` SHA256 完全一致
- **AND** 调用栈 trace SHALL 不出现 `rhs_core` / `rhs_update` / `rhs_flux` 任意一个新符号

### Requirement: Intermediate flux arrays bitwise vs legacy f_loop

`rhs_flux()` 执行完成后（即 PassValue 已调用、`rhs_flux` 即将 return 的瞬间），全局 flux 数组的内存状态 SHALL 与同输入下 legacy `f_loop()` 执行完成后的内存状态**字节相等**。

被检查的数组集合（master plan §S1b 验收门控 + design R2 风险驱动）：

- `QeleSurf[NumEle][3]`（element 表面侧向 flux）
- `QeleSub[NumEle][3]`（element 地下侧向 flux）
- `QrivSurf[NumRiv]`（river 表面输入 flux，PassValue 清零后重累加）
- `QrivSub[NumRiv]`（river 地下输入 flux，同上）
- `QrivUp[NumRiv]`（river 上游 flux，PassValue 累加）
- `Qe2r_Surf[NumEle]`（element 到 river 表面 flux，PassValue 清零后重累加）
- `Qe2r_Sub[NumEle]`（element 到 river 地下 flux，同上）
- `QsegSurf[NumSegmt]` / `QsegSub[NumSegmt]`（segment 中间 flux）
- `qLakeEvap[NumLake]` / `qLakePrcp[NumLake]`（lake 收支）
- `QLakeRivIn[NumLake]` / `QLakeRivOut[NumLake]` / `QLakeSub[NumLake]` / `QLakeSurf[NumLake]`（lake 交换 flux）
- `flux_ET` 相关数组（`qEleEvapo` / `qElePrep` / `qEleNetPrep` / `qEleInfil` / `qEleExfil` / `qEleRecharge` / `qEs` / `qEu` / `qTu` / `qTg`）

数组的 `memcmp` SHALL 返回 0，且 `SHA256` SHALL 一致。

#### Scenario: keliya rhs_flux exit state bitwise

- **WHEN** keliya 90 天 run 在 `t_values = [86400, 2592000, 7776000]`（1d / 30d / 90d，与 task 0.1a 归档 golden 对齐）做 RHS dump probe，分别用 (a) legacy `f_loop()` 完整调用 与 (b) `rhs_flux()` 完整调用（`rhs_update` 已替换）两种 binary
- **THEN** 上述所有 flux 数组的 `memcmp` SHALL 返回 0，`compare_snapshot` SHALL 报 "BITWISE IDENTICAL" 且 exit code 0

### Requirement: Full run bitwise vs B0-tag for keliya / qinyijiang / qhh

`USE_RHS_CORE=1`（S1b 新 `rhs_update` + 新 `rhs_flux` + legacy `f_applyDY`）编译后的二进制，对以下 case 跑 90 天 truncation 完整 run，所有输出 `.dat` 文件的 SHA256 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 一致：

- **keliya**（484 cells，PR fast-feedback baseline）
- **qinyijiang**（3155 cells，NumRiv/NumEle ≈ 0.10，密集河网压测 PassValue 重累加路径）
- **qhh**（4773 cells + lake，唯一 lake case，压测 rhs_flux 中 lake DY 分支）

keliya 单 case PASS SHALL **不构成** S1b merge 充分条件——密集河网（qinyijiang）+ lake 路径（qhh）的覆盖是强制的，缺一即门控阻塞。

不在 S1b 验收范围：

- **kashigeer**：N/A `deferred-upstream`（master plan §A0.6 / `docs/status_matrix.md` 已标注），CI 不跑
- **heihe / heihe_x4**（server-only）：S1b PR merge 进 baseline/current 后 24h 内由开发者在服务器手动跑一次归档（与 scaffolding / applydy spec 一致的 per-substage cadence；见 `tasks.md` 2.9），不阻塞 PR merge
- **xinanjiang_upstream**：CI 范围内但非 S1b 阻塞强制项（OMP_CUTOFF 边界 case，S1b 不涉及并行）

#### Scenario: qinyijiang dense river network bitwise PASS

- **WHEN** PR CI 跑 `USE_RHS_CORE=1` 编译的 binary 跑 qinyijiang 90 天
- **THEN** `benchmarks/qinyijiang/B0_output/*.dat` 的 SHA256 SHALL 与 `git show B0-tag:benchmarks/qinyijiang/B0_output/<file>` 完全一致
- **AND** qinyijiang 失败而 keliya 成功的组合 SHALL 视为"PassValue 累加路径破坏"信号，PR 不可合并

#### Scenario: qhh lake path bitwise PASS

- **WHEN** PR CI 跑 `USE_RHS_CORE=1` 编译的 binary 跑 qhh 90 天
- **THEN** `benchmarks/qhh/B0_output/*.dat` 的 SHA256 SHALL 与 B0-tag 完全一致
- **AND** qhh 失败而 keliya / qinyijiang 成功 SHALL 视为"rhs_flux 中 lake DY / `updateLakeElement` / `fun_Ele_lakeVertical` / `fun_Ele_lakeHorizon` 任一分支破坏"信号

### Requirement: Snapshot probe at PassValue boundary (before + after)

S1b 阶段 `rhs_snapshot` hook SHALL 在 `rhs_flux()` 内部 PassValue 调用的**前后两个时点**各 dump 一次快照，对每个 t_value（统一采用 `t_values = [86400, 2592000, 7776000]`，即 1d / 30d / 90d，单位 seconds，与 scaffolding / applydy spec 对齐；全部 ≤ 90 天截断窗口）分别保存：

- `snapshot_t<v>_before_passvalue.bin`：lake clamp 循环结束、`PassValue()` 调用**之前**的 flux 数组全集
- `snapshot_t<v>_after_passvalue.bin`：`PassValue()` 调用**返回之后**的 flux 数组全集（与 legacy `f_loop()` 返回时点相同）

两个 probe 各自 SHALL 通过 `compare_snapshot` 与 B0-tag 对应 golden snapshot 字节相等：

- **before** 探针对应 B0-tag 同时点的 "pre-PassValue" 状态（master plan §S1b 风险 R2 要求）
- **after** 探针对应 task 0.1a 在 pre-S1a 重新归档的 B0-tag golden snapshot（`benchmarks/<case>/B0_output/snapshot_t<v>.bin`，t_values 统一为 `[86400, 2592000, 7776000]`）；不再使用 S0-7 4-year-run goldens（4 年时间戳在 90 天截断窗口内不可达）

任一 probe 出现差异（`compare_snapshot` exit code ≠ 0）SHALL 让 CI 失败，且 stderr 报告 `ulp_max` / `first_diff_index` 以辅助定位是 PassValue 内部清零阶段、累加阶段，还是 PassValue 之前的 segment / river pass 出错。

**Before-PassValue golden 创建**：PassValue 之前的 golden 由 `tasks.md` task 0.1a + 0.1b 在 **pre-S1a** 一并归档（同 binary B0-tag，4 case × 3 t_values × 2 probe 点 = 24 张 golden），落在 `benchmarks/<case>/B0_output/snapshot_t<v>_before_passvalue.bin`；S1b PR 不再延后此决策。注：原 task 0.1 在 round-2 fix 中已拆分为 0.1a (12 after-PassValue) + 0.1b (12 before-PassValue + `tools/rhs_snapshot/` 工具扩展)；24 = 12 + 12 = 0.1a + 0.1b 合并产物。

#### Scenario: PassValue before / after both byte-equal

- **WHEN** S1b CI 跑 keliya 90 天，分别在 `t = 86400 / 2592000 / 7776000`（1d / 30d / 90d）的 PassValue 调用前后各 dump 一份 snapshot
- **THEN** `compare_snapshot snapshot_t<v>_before_passvalue.bin <golden_before>.bin` 与 `compare_snapshot snapshot_t<v>_after_passvalue.bin <golden_after>.bin` SHALL 各自 exit code 0 报 "BITWISE IDENTICAL"
- **AND** before-probe 与 after-probe 之间的 diff 报告 SHALL 显示 `QrivSurf` / `QrivSub` / `QrivUp` / `Qe2r_Surf` / `Qe2r_Sub` 数组值在 PassValue 调用后被清零并按 `QsegSurf` / `QsegSub` 重新累加（确认 PassValue 行为符合 master plan §4.7 描述）

### Requirement: CVODE stats invariance (15-key canonical set) and _omp path untouched

S1b 阶段验证 SHALL 同时保证：

1. **CVODE 统计 15 件套完全相等**：完整集合 `nfe` / `nfeLS` / `nni` / `nli` / `nsetups` / `netf` / `nst` / `npe` / `nps` / `ncfn` / `ncfl` / `lenrw` / `leniw` / `lenrwLS` / `leniwLS` 在 `USE_RHS_CORE=1` 与 B0 二进制下 SHALL 完全一致；任何统计漂移即使输出 `.dat` 字节相等也 SHALL 视为 FAIL（master plan §S1c 验收门控延伸至 S1b，design R5 风险条目）。比对统一调用 `tools/cvode_stats_diff/cvode_stats_diff.sh`（由 `tasks.md` task 0.2 提供，exit code 0 表 15 键全等）
2. **`_omp` 三函数零改动**：`Model_Data::f_applyDY_omp`（`SHUD/src/ModelData/MD_f_omp.cpp:9`）/ `Model_Data::f_loop_omp`（`MD_f_omp.cpp:69`）/ `Model_Data::f_update_omp`（`MD_f_omp.cpp:104`）源码在 S1b PR 中 SHALL 行级未改（`git diff` 显示对应函数 0 行变更），保留作为 S2 语义对齐参照（design D3）
3. **`_OPENMP_ON` 默认 OFF**：S1b 编译矩阵 SHALL 保持 `SHUD_ENABLE_OPENMP_RHS=0` / `SHUD_USE_OPENMP_NVECTOR=0` / `SHUD_LEGACY_OMP_RHS=0`（三宏拆分在 S1d 落地，S1b 期间 `_OPENMP_ON` 单宏依然 OFF），`_omp` 路径运行时不可触达

S1b PR 不涉及任何 OpenMP 并行（rhs_flux 内部仍纯 serial），任何 `#pragma omp parallel for` 出现在 `rhs_flux()` 体内 SHALL 视为越界。

#### Scenario: CVODE stats 15-key identical across USE_RHS_CORE switch

- **WHEN** keliya 90 天分别用 (a) `make shud` baseline 与 (b) `make shud USE_RHS_CORE=1` 各跑一次，对比落盘 `cvode_stats.txt`，执行 `tools/cvode_stats_diff/cvode_stats_diff.sh <baseline> <new>`
- **THEN** 脚本 SHALL 对 15 键（`nfe` / `nfeLS` / `nni` / `nli` / `nsetups` / `netf` / `nst` / `npe` / `nps` / `ncfn` / `ncfl` / `lenrw` / `leniw` / `lenrwLS` / `leniwLS`）全等返回 exit code 0；任意键不等（含 ±1 误差）SHALL 让 S1b FAIL

#### Scenario: _omp path source diff is zero

- **WHEN** S1b PR `git diff baseline/current...HEAD -- SHUD/src/ModelData/MD_f_omp.cpp`
- **THEN** diff SHALL 不包含 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 三函数体内任何行变更
- **AND** 即使整文件被 reformat，三函数体内 token 流 SHALL 与 baseline/current 一致
- **NOTE**：本 Scenario 仅适用于 S1b PR 上下文；post-S1d.2 由 `openmp-macro-decoupling` capability 退役 `_OPENMP_ON`（拆分为三正交宏），本 Scenario 被替代

