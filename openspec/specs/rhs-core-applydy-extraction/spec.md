# rhs-core-applydy-extraction Specification

## Purpose

S1c 阶段交付。把 `f_applyDY` 的 integrated DY 累加（element DY + river DY + lake DY + `SHUD_DUMP_RHS` hook）从 SHUD 主路径逐行搬运到 `rhs_core` 的 `rhs_apply()` 新函数；S1c 完成后 `rhs_core(Y, DY, t)` 依次调用 `rhs_update` → `rhs_flux` → `rhs_apply` 三段新路径，混合 fallback 完全退役。24 张 snapshot golden（4 case × 3 t_values × 2 probe-point：before-PassValue + after-PassValue）覆盖 bitwise + CVODE stats 15-key 验收。

## Scope

**S1c 阶段签名约定**：本 capability 全程 `rhs_core()` 为**三参版本** `rhs_core(double* Y, double* DY, double t)`，且 `rhs_apply()` 为**双参版本** `rhs_apply(double* DY, double t)`——**不**带 `ExecPolicy` 参数。`ExecPolicy` 枚举与四参 `rhs_core(Y, DY, t, policy)` / 三参 `rhs_apply(DY, t, policy)` 重载由 `exec-policy-enum` capability 在 **S1d.1** 引入；S1c spec 不引入 `ExecPolicy`，避免阶段越界。

**头文件命名约定**：实现文件 `SHUD/src/Model/MD_rhs_core.cpp`；header `SHUD/src/Model/MD_rhs_core.hpp`（与 scaffolding / flux spec 一致）。

**Temporal scope note**：本 capability 中所有 `_OPENMP_ON` 相关 Scenarios 仅适用于 S1c PR 评审上下文；post-S1d.2 由 `openmp-macro-decoupling` capability 退役 `_OPENMP_ON`（拆分为 `SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS` 三正交宏），相关 Scenarios 被 superseded。
## Requirements
### Requirement: rhs_apply body is verbatim copy of serial f_applyDY

`SHUD/src/Model/MD_rhs_core.cpp` 中新增的 `Model_Data::rhs_apply(double *DY, double t)` SHALL 把 `Model_Data::f_applyDY()`（`SHUD/src/ModelData/MD_f.cpp::f_applyDY`）的循环体（element DY + river DY + lake DY + 可选 `SHUD_DUMP_RHS` hook）逐行搬运到新函数体，不得改写任何表达式、不得调整三个 for 循环的执行顺序、不得新增 / 删除任何全局变量读写。被搬运的代码 MUST 与 `git show B0-tag:SHUD/src/ModelData/MD_f.cpp` 中 `f_applyDY()` 函数体逻辑等价；唯一允许的差异是函数名（`f_applyDY` → `rhs_apply`）+ `#ifdef SHUD_DUMP_RHS` hook 字符串 tag。`rhs_apply` 保持 `Model_Data::` 成员函数前缀（与 S1a `rhs_update` / S1b `rhs_flux` 一致；S1a 实施时已确认 `iSF`/`iUS`/`iGW`/`iRIV`/`iLAKE` 宏需要 `this` scope，详 PR #58 工作总结）。

#### Scenario: Diff vs B0 f_applyDY shows only signature changes

- **WHEN** reviewer 在 S1c PR 上对 `git show B0-tag:SHUD/src/ModelData/MD_f.cpp` 的 `f_applyDY()` 函数体（`MD_f.cpp::f_applyDY`）与 `SHUD/src/Model/MD_rhs_core.cpp` 中 `rhs_apply()` 函数体做逐行 diff
- **THEN** 输出 SHALL 只包含函数签名 / `MD->` 前缀 / `#ifdef SHUD_DUMP_RHS` hook 名称三类差异
- **AND** 任何表达式 / 控制流 / 循环边界 / 浮点常量 / 数学函数调用 MUST 字符级一致

#### Scenario: Global variable read/write set unchanged

- **WHEN** 静态扫描 `rhs_apply()` 中所有 SHUD 全局变量出现位置（`uYsf` / `uYus` / `uYgw` / `qEleNetPrep` / `qEleInfil` / `qEleExfil` / `qEleRecharge` / `qEs` / `qEu` / `qEg` / `qTu` / `qTg` / `QeleSurf` / `QeleSub` / `Qe2r_Surf` / `Qe2r_Sub` / `QeleSurfTot` / `QeleSubTot` / `QrivUp` / `QrivSurf` / `QrivSub` / `QrivDown` / `qLakePrcp` / `qLakeEvap` / `QLakeRivIn` / `QLakeRivOut` / `QLakeSub` / `QLakeSurf` / `y2LakeArea`）
- **THEN** 集合 SHALL 与 B0-tag `f_applyDY()` 完全相同
- **AND** `rhs_apply()` MUST NOT 在入口 / 出口处对全局变量做任何额外清零 / 重置 / 镜像

### Requirement: River DY uses serial formula with length + area clamp + fun_dAtodY

`rhs_apply()` 在 river DY 计算分支 SHALL 严格沿用 serial `Model_Data::f_applyDY()`（`SHUD/src/ModelData/MD_f.cpp::f_applyDY` 内 river DY 段）的三步公式：① `DY[iRIV] = (- QrivUp[i] - QrivSurf[i] - QrivSub[i] - QrivDown[i] + Riv[i].qBC) / Riv[i].Length;` ② `if(DY[iRIV] < -1. * Riv[i].u_CSarea) DY[iRIV] = -1. * Riv[i].u_CSarea;` ③ `DY[iRIV] = fun_dAtodY(DY[iRIV], Riv[i].u_topWidth, Riv[i].bankslope);`。MUST NOT 使用 `_omp` 路径（`SHUD/src/ModelData/MD_f_omp.cpp::f_applyDY_omp` 内 river DY 段）中"直接除以 `u_TopArea`"的简化公式，原因详见 master plan §4.3。Neumann 边界条件分支（`Riv[i].BC > 0` → `DY[iRIV] = 0.`）也 MUST 完整保留。

#### Scenario: River DY divisor is Riv[i].Length not u_TopArea

- **WHEN** 静态 grep `rhs_apply()` 中 river 循环体
- **THEN** SHALL 出现 `/ Riv[i].Length`，MUST NOT 出现 `/ Riv[i].u_TopArea`
- **AND** SHALL 出现 `fun_dAtodY(` 调用，调用参数顺序为 `(DY[iRIV], Riv[i].u_topWidth, Riv[i].bankslope)`

#### Scenario: qinyijiang dense river network bitwise PASS

- **WHEN** 在 `qinyijiang`（NumRiv/NumEle = 0.10，3155 cells，密集河网）以 `USE_RHS_CORE=1` 完整跑 90 天
- **THEN** `benchmarks/qinyijiang/B0_output/` 下所有 river 相关输出 (`*.rivStg.dat` / `*.rivQ.dat` / `*.rivFlx*.dat`) 的 SHA256 SHALL 与 `git show B0-tag:benchmarks/qinyijiang/B0_output/<file>` 完全一致

#### Scenario: Negative — simpler omp formula would break qinyijiang bitwise

- **WHEN** 反例：把 river DY 临时改为 `_omp` 公式 `DY[iRIV] = (...) / Riv[i].u_TopArea;` 并删掉 area clamp + `fun_dAtodY()` 调用
- **THEN** `qinyijiang` 第 1 天 snapshot SHA256 SHALL 立即与 B0 golden 不一致，CI bitwise check MUST 失败
- **AND** 此反例 SHALL 以 patch 形式落地 `SHUD/tests/s1c_river_dy_omp_negative.patch`（**不入** main commit；patch + revert 一起留档）；S1c PR 描述 MUST 附以下命令：`git apply tests/s1c_river_dy_omp_negative.patch && make shud && ./shud qinyijiang`，并写明预期 fail SHA256（pinned 进 PR description）+ revert 命令 `git apply -R tests/s1c_river_dy_omp_negative.patch`

### Requirement: Lake DY computation preserved

`rhs_apply()` 在 lake DY 计算分支 SHALL 完整保留 serial `Model_Data::f_applyDY()`（`SHUD/src/ModelData/MD_f.cpp::f_applyDY` 内 lake DY 段）的 lake DY 公式：`DY[iLAKE] = qLakePrcp[i] - qLakeEvap[i] + (QLakeRivIn[i] - QLakeRivOut[i] + QLakeSub[i] + QLakeSurf[i]) / y2LakeArea[i];`，循环上界保持 `i < NumLake`，DEBUG 模式 `CheckNANi(DY[iLAKE], ...)` 同步保留。`_omp` 路径（`SHUD/src/ModelData/MD_f_omp.cpp::f_applyDY_omp`）完全缺失 lake DY 计算（master plan §S2.5 / §4.2），但 S1c MUST 保留 serial lake 语义不变。

**Lake DY divide-by-zero 保留 UB 说明**：serial `f_applyDY()` lake DY 公式中存在 `y2LakeArea[i] == 0` 时直接除零的 undefined behavior 路径；S1c 阶段 **bitwise-preserve** 这一 UB（即 `rhs_apply()` 不得加保护性 guard / fallback / 检查），与 B0 二进制行为完全一致。numerical-stability 修复（如 epsilon-guard / 安全除法）属 **S5 / S6** 范畴，S1c 阶段不引入。

#### Scenario: Lake DY loop present with serial formula

- **WHEN** 静态扫描 `rhs_apply()` 函数体
- **THEN** SHALL 存在以 `NumLake` 为上界的循环
- **AND** 循环体 SHALL 写 `DY[iLAKE]`，表达式与 B0-tag serial `f_applyDY()` 字符级一致

#### Scenario: Lake DY divide-by-zero UB preserved

- **WHEN** 在 `rhs_apply()` 函数体内 `grep` lake DY computation 段（写 `DY[iLAKE]` 的 `for (i = 0; i < NumLake; i++)` 循环）
- **THEN** SHALL NOT 出现 `if (y2LakeArea[i] == 0)` / `if (y2LakeArea[i] < epsilon)` / `fabs(y2LakeArea[i])` / 任何 guard pattern（包括 ternary `y2LakeArea[i] != 0 ? ... : 0` 或显式 `assert(y2LakeArea[i] > 0)` 之类的 defensive check）
- **AND** divide-by-zero UB SHALL bitwise-preserve serial path semantics（即直接执行 `/ y2LakeArea[i]`，与 B0-tag serial `f_applyDY()` 字符级一致）
- **AND** numerical-stability 修复（epsilon-guard / 安全除法 / IEEE 754 inf 处理）属 **S5 / S6** 范畴，S1c 阶段任何 guard 引入 SHALL 让 reviewer 否决

#### Scenario: qhh lake case bitwise PASS

- **WHEN** 在 `qhh`（4773 cells + lake）以 `USE_RHS_CORE=1` 完整跑 90 天
- **THEN** `benchmarks/qhh/B0_output/*.lakeStg.dat` 与 `*.lakeQ*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/qhh/B0_output/<file>` 完全一致
- **AND** snapshot probe 在 `t_values = [86400, 2592000, 7776000]`（1d / 30d / 90d，统一与 scaffolding / flux spec 对齐）处 dump 的 DY 数组（含 `iLAKE` 索引段）SHALL 与 golden snapshot bitwise 一致

### Requirement: rhs_core full new path after S1c

S1c 完成后，`f.cpp` 中 `#ifdef USE_RHS_CORE` 启用时 `rhs_core(Y, DY, t)`（三参签名）SHALL 依次调用 `rhs_update(Y, DY, t)` + `rhs_flux(t)` + `rhs_apply(DY, t)` 三个新路径函数，覆盖完整 RHS 评估（Y → uY* → flux → DY）。MUST NOT 出现"新路径 + legacy fallback"混合调用（S1a / S1b 的混合 fallback 在 S1c 完成时彻底移除）。dispatch 入口 `rhs_core()` MUST NOT 回调任何 `f_update` / `f_loop` / `f_applyDY` 旧函数。`ExecPolicy` 枚举与四参重载由 S1d.1 引入，S1c 不涉及。

#### Scenario: rhs_core body invokes only new path

- **WHEN** 静态扫描 `SHUD/src/Model/MD_rhs_core.cpp` 中 `rhs_core(Y, DY, t)` 函数体
- **THEN** 函数调用 SHALL 严格为 `rhs_update(Y, DY, t)` → `rhs_flux(t)` → `rhs_apply(DY, t)` 三行，按此顺序
- **AND** MUST NOT 出现对 `f_update` / `f_loop` / `f_applyDY`（任何成员函数形式）的直接调用
- **AND** MUST NOT 出现 `ExecPolicy` 参数 / `switch (policy)` 分支（S1d.1 才引入）

#### Scenario: USE_RHS_CORE=1 build links rhs_apply

- **WHEN** `make shud USE_RHS_CORE=1` 编译并 `nm` / `objdump` 检查 binary 符号表
- **THEN** SHALL 存在 `rhs_apply` 符号
- **AND** legacy `f_applyDY` 符号在 `USE_RHS_CORE=1` 时 SHALL 仍存在（作为对比基线），但不被 `rhs_core()` 调用栈触及

### Requirement: Bitwise equality vs B0 across local cases

S1c 完整 binary（`USE_RHS_CORE=1`，serial 路径全开）在 keliya / xinanjiang_upstream / qinyijiang / qhh 四个 local case 上各跑一次 90 天 deployment（START → START+90d，依 `tools/fix_case_paths/` 自动截断），所有 `benchmarks/<case>/B0_output/*.dat` 输出 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` SHA256 完全相等。任何一个 case 任何一个 `.dat` 文件 SHA256 不一致即门控 FAIL，不允许"主要输出 PASS 次要输出 FAIL"的部分通过。

#### Scenario: keliya fast-feedback bitwise PASS

- **WHEN** PR 触发 CI，CI 编译 `USE_RHS_CORE=1` 后跑 `keliya` 90 天
- **THEN** `benchmarks/keliya/B0_output/` 下每个 `.dat` 文件的 SHA256 SHALL 与 `git show B0-tag:benchmarks/keliya/B0_output/<file>` 输出完全一致
- **AND** CI job 退出码 SHALL 为 0

#### Scenario: 4-case full bitwise PASS

- **WHEN** PR 标 `full-bitwise` label 或 nightly run，CI 依次跑 keliya / xinanjiang_upstream / qinyijiang / qhh
- **THEN** 每个 case 的所有 `*.dat` 输出 SHA256 SHALL 与对应 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全相等
- **AND** 任一 case 任一文件不一致 SHALL 阻塞 PR merge

#### Scenario: Single .dat mismatch fails entire S1c gate

- **WHEN** 任意 case 的某个低优先级输出（例如 `*.eleETvap.dat`）SHA256 与 golden 不一致
- **THEN** S1c PR gate SHALL 标记为 FAIL，即使主要 hydraulic 输出（`*.eleSurf.dat` / `*.rivStg.dat`）全部 PASS
- **AND** PR 描述 MUST NOT 以"主要输出对齐，次要差异已知"为由请求 merge

### Requirement: CVODE stats 15-key canonical equality

S1c 完成后，每个 case 的完整 run 在结束时 SUNDIALS CVODE 报告的 **canonical 15-key set** 统计 SHALL 与 B0-tag run 完全相等（整数 bitwise）。15 个键完整集合定义详 `openspec/glossary.md` §CVODE canonical 15-key set（全部来自 SUNDIALS CVODE / CVSpils API 的 `PrintFinalStats` 输出，与 SHUD 实际归档对齐；`nFCall` 由独立 capability 跟踪，详 design.md D10 F19 修订）。stats 漂移即使 `.dat` 输出 SHA256 一致也算门控 FAIL，原因详见 design.md R5："物理上没改但 f.cpp 入口分支语义改了，如果 RHS 评估时机有微差（比如 fallback 期间多一次空调用），nfe 会变"。比较 MUST 是精确相等而非近似（差 1 也算 FAIL）。比对统一调用 `tools/cvode_stats_diff/cvode_stats_diff.sh`（由 `tasks.md` task 0.2 提供，exit code 0 表 canonical 15-key 全等）。

#### Scenario: CVODE stats 15-key bitwise equal on xinanjiang_upstream

- **WHEN** `xinanjiang_upstream`（801 cells，OMP_CUTOFF 边界）以 `USE_RHS_CORE=1` 完整跑 90 天，落盘 `cvode_stats.txt`，执行 `tools/cvode_stats_diff/cvode_stats_diff.sh <B0_archive>/cvode_stats.txt <S1c_run>/cvode_stats.txt`
- **THEN** canonical 15-key set（详 `openspec/glossary.md` §CVODE canonical 15-key set）SHALL 全部精确相等，脚本返回 exit code 0

#### Scenario: nfe drift by 1 is FAIL

- **WHEN** 任意 case 报告 `nfe(S1c) = nfe(B0) + 1` 或 `- 1`（或 canonical 15-key 中其它任意键差 ≥ 1）
- **THEN** S1c gate SHALL 标记为 FAIL
- **AND** PR 描述 MUST NOT 以"差异 < 0.01%"为由请求 merge；MUST 回溯定位为 R5 场景（fallback 期间额外空调用 / dispatch 分支多绕一次 RHS）并修复

#### Scenario: Stats logged into archive for diff

- **WHEN** S1c PR 跑完 4 个 local case
- **THEN** PR comment 或 CI artifact SHALL 包含一张表格列出 4 case × canonical 15-key 字段（详 `openspec/glossary.md` §CVODE canonical 15-key set）的 S1c 值 vs B0-tag 值
- **AND** 表格 MUST 显式标注每一格"= B0"或具体差值

### Requirement: Snapshot probe bitwise vs golden after rhs_apply

`tools/rhs_snapshot/` 在统一 `t_values = [86400, 2592000, 7776000]`（1d / 30d / 90d，单位 seconds，全部 ≤ 90 天截断；与 scaffolding / flux spec 对齐；12 张 golden 由 `tasks.md` task 0.1a 在 pre-S1a 重新归档）SHALL 在 `rhs_apply()` 完成后 dump `DY[0:NumY]` 完整数组（含 element / river / lake 三段），与 `benchmarks/<case>/B0_output/snapshot_t<value>.bin` 调用 `tools/compare_snapshot/compare_snapshot` 比对 MUST 返回 exit code 0（即 "BITWISE IDENTICAL"）。snapshot 时机 MUST 在 `rhs_apply` 完成后、`rhs_core` 函数返回前；MUST NOT 在 `rhs_flux` 之后立即 dump（否则捕获到的是 flux 状态而非 DY）。

#### Scenario: All 3 snapshots per case PASS

- **WHEN** keliya / xinanjiang_upstream / qinyijiang / qhh 各跑一次 `USE_RHS_CORE=1`，按统一 `t_values = [86400, 2592000, 7776000]` 触发 snapshot dump
- **THEN** `tools/compare_snapshot/compare_snapshot <golden> <new>` SHALL 对 4 × 3 = 12 张 snapshot 全部返回 exit code 0
- **AND** stdout SHALL 含 "BITWISE IDENTICAL" 行

#### Scenario: Snapshot timing is post-rhs_apply

- **WHEN** 用 gdb 或 printf 追踪 snapshot dump 调用栈
- **THEN** 调用栈 SHALL 在 `rhs_apply()` 返回之后、`rhs_core()` 返回之前进入 dump hook
- **AND** dumped DY 数组 MUST 包含 element / river / lake 三段（长度 = `NumY`）

#### Scenario: Compare exit code propagated to CI

- **WHEN** 任意一张 snapshot compare 返回 exit code 非 0
- **THEN** CI job SHALL 失败并把 `compare_snapshot` 的 stdout / stderr 完整附在 job log

### Requirement: _omp path untouched

S1c 全程 SHALL NOT 修改 `SHUD/src/ModelData/MD_f_omp.cpp` 中 `Model_Data::f_applyDY_omp` / `Model_Data::f_loop_omp` / `Model_Data::f_update_omp` 任何一行。`f_applyDY_omp` 的 `_omp` river DY 公式（直接除 `u_TopArea`，缺面积 clamp 和 `fun_dAtodY`）、缺失的 lake DY 段、`f_applyDY_omp` 局部变量 data race（master plan §4.6）三项已知问题 MUST 全部留在原位等 S2 处理。`SHUD_LEGACY_OMP_RHS` 宏 OFF 时 `_omp` 三个函数 MUST NOT 进入编译；ON 时进入编译但 S1c 期间不允许测试触达。

#### Scenario: MD_f_omp.cpp diff is empty in S1c PR

- **WHEN** 检查 S1c PR 中 `SHUD/src/ModelData/MD_f_omp.cpp` 的 git diff
- **THEN** diff SHALL 完全为空（0 lines changed）
- **AND** 任何对 `_omp` 函数体的"顺手优化 / 顺手对齐"MUST 在 review 中被打回
- **NOTE**：本 Scenario 仅适用于 S1c PR 上下文；post-S1d.2 由 `openmp-macro-decoupling` capability 退役 `_OPENMP_ON`（拆分为三正交宏），本 Scenario 被替代

#### Scenario: SHUD_LEGACY_OMP_RHS=0 strips _omp from binary

- **WHEN** `make shud SHUD_LEGACY_OMP_RHS=0` 编译并 `nm` 检查 binary 符号表
- **THEN** `Model_Data::f_applyDY_omp` / `f_loop_omp` / `f_update_omp` 符号 SHALL NOT 出现
- **AND** binary 链接 MUST NOT 依赖 `nvector_openmp.h` 提供的符号

### Requirement: rhs_core complete after S1c — three-parameter only

S1c 完成时 `rhs_core(Y, DY, t)`（三参签名）SHALL 是唯一已实现的执行入口，覆盖完整 RHS 评估三步（`rhs_update` + `rhs_flux` + `rhs_apply`）。`ExecPolicy` 枚举与四参重载 `rhs_core(Y, DY, t, ExecPolicy policy)` 由 `exec-policy-enum` capability 在 **S1d.1** 引入，S1c spec 不引入 ExecPolicy / StrictOMP / ProductionOMP 任何相关符号；`LEGACY_RHS` 编译宏 + `USE_RHS_CORE` 脚手架删除属 S1d 范围；S1c PR MUST NOT 触碰这两项。

#### Scenario: rhs_core has only three-parameter signature

- **WHEN** 检查 S1c PR 中 `SHUD/src/Model/MD_rhs_core.hpp` 头文件声明
- **THEN** SHALL 仅存在 `rhs_core(double* Y, double* DY, double t)` 单一声明
- **AND** MUST NOT 出现 `ExecPolicy` 类型 / 四参重载 / `switch (policy)` 分支
- **AND** S1c 调用栈正常完成 `rhs_update` + `rhs_flux` + `rhs_apply` 三步即视为 PASS

#### Scenario: USE_RHS_CORE scaffolding still present

- **WHEN** 检查 S1c PR 中 `SHUD/src/Model/f.cpp` 的 git diff
- **THEN** `#ifdef USE_RHS_CORE` 调度块 SHALL 仍存在
- **AND** S1c PR MUST NOT 删除 `USE_RHS_CORE` 宏或将其改为 `LEGACY_RHS`

#### Scenario: ExecPolicy enum not introduced in S1c

- **WHEN** 检查 S1c PR 中 `SHUD/src/Model/MD_rhs_core.cpp` / `MD_rhs_core.hpp` / `f.cpp`
- **THEN** SHALL NOT 出现 `enum class ExecPolicy` / `ExecPolicy::Serial` / `ExecPolicy::StrictOMP` / `ExecPolicy::ProductionOMP` 任何 token
- **AND** ExecPolicy 引入由 `exec-policy-enum` capability 在 S1d.1 单独 PR 完成

### Requirement: Server-only case manual validation within 24h post-merge

S1c PR merge 到 `baseline/current` 后，operator SHALL 在 24 小时内于服务器 `/scratch/frd_muziyao/SHUD-OpenMP` 用 Slurm 在计算节点跑 `heihe`（6335 cells，本地不下 forcing）和 `heihe_x4`（≈25k cells，rSHUD v2.5.0 master 加密生成）各一次 90 天，对 `USE_RHS_CORE=1` 与 B0-tag 的 `.dat` 输出做 SHA256 全量比对，bitwise FAIL MUST 回滚外层 submodule pointer。验证结果 MUST 以服务器 artifact + SHA256 列表形式归档到 `.s1-server-validation/` 目录（dot-prefixed scratch，自动 gitignored），并在 PR comment 留 link。

#### Scenario: heihe Slurm job bitwise PASS

- **WHEN** S1c PR merge 后 ≤ 24h，operator 在服务器 `/scratch/frd_muziyao/SHUD-OpenMP/.s1-server-validation/` 下 `sbatch` 提交 heihe 90 天 run（`USE_RHS_CORE=1`，`OMP_NUM_THREADS=1` serial）
- **THEN** `benchmarks/heihe/B0_output/*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/heihe/B0_output/<file>` 完全相等
- **AND** Slurm job ExitCode SHALL 为 0
- **AND** stdout / stderr MUST 写入 `/scratch` 共享盘（按 CLAUDE.md "三条铁律"）

#### Scenario: heihe_x4 Slurm job bitwise PASS

- **WHEN** S1c PR merge 后 ≤ 24h，operator 在服务器对 `heihe_x4` 跑同样 `USE_RHS_CORE=1` 90 天 run
- **THEN** `benchmarks/heihe_x4/B0_output/*.dat` SHA256 SHALL 与 B0-tag golden 完全相等

#### Scenario: Server FAIL triggers submodule pointer revert

- **WHEN** heihe 或 heihe_x4 任一 `.dat` SHA256 不一致
- **THEN** operator SHALL 在外层 `baseline/current` 上 revert 该 PR 引入的 SHUD submodule pointer bump
- **AND** SHUD `openmp-baseline` 分支 SHALL `git revert` 对应 commit 并重 push
- **AND** S1c gate SHALL 重新标 FAIL，S1d 不得启动

