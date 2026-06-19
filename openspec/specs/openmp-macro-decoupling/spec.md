# openmp-macro-decoupling Specification

## Purpose

S1d.2 阶段交付。把过载三义的 `_OPENMP_ON` 单宏拆成三个正交宏（`SHUD_ENABLE_OPENMP_RHS` 控制 rhs_core OMP 分支编译 / `SHUD_USE_OPENMP_NVECTOR` 控制 N_Vector 后端 + `-lsundials_nvecopenmp` 链接 / `SHUD_LEGACY_OMP_RHS` 控制 `_omp` 三函数编译），全树退役 `_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial`；`f.cpp` 6 处 `NV_DATA_OMP` / `NV_DATA_S` 统一为 generic `N_VGetArrayPointer`，`Macros.hpp` 的 `SET_VALUE` 改 `N_VGetArrayPointer(v)[i]`；S1d-part2 独立 sub-PR，4 case bitwise vs B0 PASS。

## Requirements
### Requirement: 三正交宏取代 `_OPENMP_ON` 单宏

构建系统 SHALL 引入三个独立正交宏 `SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS` 替代原 `_OPENMP_ON` 单宏。三宏 MUST 互相独立、可任意组合（共 8 个配置组合，本 capability 重点验证其中 4 个），且 MUST 在 `SHUD/Makefile` 与 `SHUD/src/Model/Macros.hpp` 中默认全部为 OFF（值 0 或 undefined）。三宏 MUST 分别对应三个独立关注点：

- `SHUD_ENABLE_OPENMP_RHS`：控制 `rhs_core(StrictOMP)` / `rhs_core(ProductionOMP)` 分支是否编译进 binary（S1 阶段两分支为 stub）
- `SHUD_USE_OPENMP_NVECTOR`：控制 `shud.cpp` 中 N_Vector 创建走 `N_VNew_OpenMP` 还是 `N_VNew_Serial`，并决定链接阶段是否引入 `-lsundials_nvecopenmp`
- `SHUD_LEGACY_OMP_RHS`：控制 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 是否参与编译（S1 阶段保留作 S2 对齐参照）

S1 默认（三宏全 OFF）SHALL 与 B0 binary 完全等价的 serial 路径运行；任一组合 ON 的 binary MUST 至少能 smoke-compile 通过（不强制运行成功，stub 路径在 runtime 可 assert）。

#### Scenario: Config A 默认全 OFF 与 B0 bitwise

- **WHEN** 开发者执行 `make shud SHUD_ENABLE_OPENMP_RHS=0 SHUD_USE_OPENMP_NVECTOR=0 SHUD_LEGACY_OMP_RHS=0`（或等价的 `make shud` 默认调用），并在 keliya / xinanjiang_upstream / qinyijiang / qhh 4 个本地 case 上跑完 90 天截断 run
- **THEN** 4 case 的全部 `B0_output/*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全相等，CVODE stats 通过 `tools/cvode_stats_diff/cvode_stats_diff.sh` 对 canonical 15-key set（`nfe` / `nfeLS` / `nni` / `nli` / `nsetups` / `netf` / `nst` / `npe` / `nps` / `ncfn` / `ncfl` / `lenrw` / `leniw` / `lenrwLS` / `leniwLS`）全键 diff exit code = 0（F19 修订：归档不含 `nFCall`，由独立 capability 跟踪）

#### Scenario: Config B SHUD_LEGACY_OMP_RHS=1 编译入 _omp 符号（value-add = binary 含 _omp 符号 + Serial 路径无 perturbation）

Config B 的 value-add **不是**单纯"与 B0 bitwise 相等"（runtime 默认仍走 `rhs_core(Serial)`，`_omp` 三函数仅编译进 binary 不被调用，bitwise 平凡成立），而是**验证两件事**：① binary 符号表中确有 `_omp` 三符号（compile + link 路径在 S2 启用前已就位）；② 编译这些额外 translation unit MUST NOT 扰动 Serial 路径的代码生成（如 static init order、模板实例化顺序、symbol layout 等）——bitwise PASS 即为 smoke check：Config B 的 Serial 输出 SHA256 与 B0 完全相等，反过来证明加 `_omp` TU 不引入隐式副作用。

- **WHEN** 开发者执行 `make shud SHUD_ENABLE_OPENMP_RHS=0 SHUD_USE_OPENMP_NVECTOR=0 SHUD_LEGACY_OMP_RHS=1`
- **THEN**（primary value-add）`nm shud | grep -E 'f_update_omp|f_loop_omp|f_applyDY_omp' | wc -l` SHALL 输出 `3`（三个 `_omp` 符号在 binary 符号表中确实就位，type 为 `T` / `t`，text section），这是本 Config 的核心 value-add——验证 compile + link 路径在 S2 启用前已就位
- **AND**（smoke check）在 4 本地 case 上跑 90 天截断 run，全部 `B0_output/*.dat` SHA256 与 B0-tag 完全相等——验证编译时引入 `_omp` symbols 未扰动 Serial 路径的代码生成 / 静态初始化顺序 / TU 布局（runtime 默认仍走 `rhs_core(Serial)`，`_omp` 三函数仅编译进 binary 不被调用，bitwise 平凡成立；反过来 bitwise PASS 即证明加 `_omp` TU 不引入隐式副作用）
- **AND**（smoke check）`tools/cvode_stats_diff/cvode_stats_diff.sh` 15-key 对比 exit code = 0（同上 smoke check 语义，CVODE stats 15 键全键不被 `_omp` TU 编译入扰动）

#### Scenario: Config C SHUD_ENABLE_OPENMP_RHS=1 smoke-compile 通过 + runtime abort

- **WHEN** 开发者执行 `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_USE_OPENMP_NVECTOR=0 SHUD_LEGACY_OMP_RHS=0`
- **THEN** `make` 命令 SHALL 成功（exit code 0），binary 中 `rhs_core(StrictOMP)` 分支 SHALL 编译入符号表；运行时若 `f()` 调度到 StrictOMP 分支 SHALL 调用 `std::abort()` 终止进程（S1 阶段 stub 行为，不使用 `assert(false)`）

#### Scenario: Config D SHUD_ENABLE_OPENMP_RHS=1 + SHUD_USE_OPENMP_NVECTOR=1 smoke-compile 通过

- **WHEN** 开发者执行 `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_USE_OPENMP_NVECTOR=1 SHUD_LEGACY_OMP_RHS=0`
- **THEN** `make` 命令 SHALL 成功，binary 链接阶段 SHALL 含 `-lsundials_nvecopenmp`，`N_VNew_OpenMP` 符号 SHALL 出现在 binary 符号表中（`nm shud | grep N_VNew_OpenMP` 非空）；运行时若 `f()` 调度到 ProductionOMP 分支 SHALL 调用 `std::abort()` 终止进程

### Requirement: `_OPENMP_ON` 宏从源码全面退役（S1d.2 完成时）

`_OPENMP_ON` 宏在 S1a / S1b / S1c 期间 SHALL 保持原样存在（这三个 substage 不动 OpenMP 宏分派，仅做 RHS 三段提取）；本 capability（S1d.2）完成后，整个 SHUD submodule 源码（`SHUD/src/**` 与 `SHUD/Makefile`）MUST NOT 再含有 `_OPENMP_ON` 文字标识符。所有历史引用 `_OPENMP_ON` 的代码 MUST 迁移至三宏中**对应正确关注点**的新宏：

- `SHUD/src/Model/f.cpp` 中 `f` / `f_surf` / `f_unsat` / `f_gw` / `f_river` / `f_lake` 的 RHS 分派 → `SHUD_ENABLE_OPENMP_RHS`（S1 阶段统一改走 `rhs_core(Serial)`，相关 ifdef 移除）
- `SHUD/src/Model/shud.cpp` N_Vector 创建（L55–L75）→ `SHUD_USE_OPENMP_NVECTOR`
- `SHUD/src/Model/Macros.hpp` `#include "omp.h"` / `#include "nvector/nvector_openmp.h"` / `SET_VALUE` → 由 `SHUD_USE_OPENMP_NVECTOR` 控制 header 引入，`SET_VALUE` 走 generic 接口（见下条 Requirement）
- `SHUD/Makefile` `CXX_OPENMP_DEFINE = -D_OPENMP_ON` → 删除该行，按三宏分别注入 `-DSHUD_ENABLE_OPENMP_RHS=1` / `-DSHUD_USE_OPENMP_NVECTOR=1` / `-DSHUD_LEGACY_OMP_RHS=1`

#### Scenario: 本 capability landed 后源码不再含 `_OPENMP_ON`（grep 全树零命中）

- **WHEN** 本 capability merge 后，在 SHUD 仓库 working tree 上执行 `grep -r '_OPENMP_ON' SHUD/src/`
- **THEN** 命令 SHALL 返回 ZERO 匹配（exit code 1，无 stdout 输出）
- **AND** 在 `SHUD/Makefile` 上执行 `grep -n '_OPENMP_ON' SHUD/Makefile` 同样 SHALL 返回 ZERO 匹配
- **AND** 该 grep 退出码作为 CI 强 gate（详见 `b0-tag-ci-integration` capability 的 macro removal grep gate Requirement）

#### Scenario: S1a/S1b/S1c 期间 `_OPENMP_ON` 仍然存在（不强制提前退役）

- **WHEN** 检查 S1a / S1b / S1c 任一 PR merge 后的 SHUD 源码
- **THEN** `_OPENMP_ON` 文字标识符 SHALL 仍然存在于 `SHUD/src/Model/f.cpp` 与 `SHUD/Makefile`（不在这些 substage 的清理范围）
- **AND** 仅 S1d.2 才强制全树退役；提前删 = 跨 capability scope creep

#### Scenario: f.cpp 六个 RHS 入口函数不再含 `#ifdef _OPENMP_ON`

- **WHEN** 在 SHUD 源码上执行 `grep -n '#ifdef _OPENMP_ON\|#ifndef _OPENMP_ON' SHUD/src/Model/f.cpp`
- **THEN** 命令 SHALL 返回 ZERO 匹配；6 个 RHS 入口函数（`f` / `f_surf` / `f_unsat` / `f_gw` / `f_river` / `f_lake`）中均不再有 N_Vector 类型分支 ifdef

#### Scenario: Makefile 不再注入 `-D_OPENMP_ON`

- **WHEN** 执行 `grep -n '_OPENMP_ON' SHUD/Makefile`
- **THEN** 命令 SHALL 返回 ZERO 匹配；原 `CXX_OPENMP_DEFINE = -D_OPENMP_ON` 行 SHALL 已删除或替换为按三宏 ON/OFF 注入对应 `-DSHUD_*` 的逻辑

#### Scenario: Config A binary 中无 `_OPENMP_ON` 痕迹

- **WHEN** 在 Config A binary 上执行 `strings shud | grep _OPENMP_ON` 或反汇编 grep
- **THEN** 命令 SHALL 返回 ZERO 匹配；任何后续基于该 binary 的 debug / strings 检索均查不到旧宏字符串残留

### Requirement: N_Vector 数组访问统一为 `N_VGetArrayPointer`

`SHUD/src/Model/f.cpp` 中所有 6 处 `NV_DATA_OMP(v)` / `NV_DATA_S(v)` 调用 MUST 统一替换为 SUNDIALS generic 接口 `N_VGetArrayPointer(v)`。替换后 `f.cpp` 中 MUST NOT 再有任何 `#ifdef` 围绕 N_Vector 数组指针访问的条件分支。Generic 接口 SHALL 在 SUNDIALS 6.0.0 中自动 dispatch 到 `N_VGetArrayPointer_Serial` / `N_VGetArrayPointer_OpenMP` 实现，且在 `-O2` 优化下 SHALL inline 等价于直接展开（无运行时开销）。

#### Scenario: f.cpp 无 NV_DATA_OMP / NV_DATA_S 残留

- **WHEN** 在 SHUD 源码上执行 `grep -rn 'NV_DATA_OMP\|NV_DATA_S' SHUD/src/Model/f.cpp`
- **THEN** 命令 SHALL 返回 ZERO 匹配

#### Scenario: f.cpp 中 6 处 N_Vector 解包统一调用 N_VGetArrayPointer

- **WHEN** 在 `f.cpp` 上执行 `grep -cn 'N_VGetArrayPointer' SHUD/src/Model/f.cpp`
- **THEN** 命令 SHALL 返回 12 行匹配（6 个 RHS 入口函数 × 每个 2 次：`CV_Y` + `CV_Ydot`），且每次调用前后均不再被 `#ifdef _OPENMP_ON` / `#ifdef SHUD_*` 包裹

#### Scenario: N_VGetArrayPointer perf invariance（informational only，非 merge gate）

- **WHEN** 在同一硬件 / 同一工具链 / 同一 locked flags 下，编译 Config A binary 与 B0-tag binary，并各跑 keliya 90 天截断 3 次取中位数
- **THEN** Config A binary size SHALL 与 B0 相差 ≤ 2%（generic 接口 inline 后展开等价于直接宏），中位 wall time SHALL 与 B0 中位 ±5% 内（generic 接口在 `-O2` 下 inline 无开销）
- **AND** 该 Scenario 仅作 informational 记录（提示 `N_VGetArrayPointer` 引入未引起退步），**不作为 merge gate**——bitwise PASS（前述 4 case SHA256 + 15-key CVODE diff）是本 capability 唯一 merge 准入条件（per design.md R6 mitigation）
- **AND** 若 size / wall time 超 ±2% / ±5%，CI SHALL 以 `::warning` 提示而非 `::error` fail

### Requirement: `Macros.hpp` 中 `SET_VALUE` 宏迁移至 `N_VGetArrayPointer(v)[i]`

`SHUD/src/Model/Macros.hpp` L11–L18 中由 `#ifdef _OPENMP_ON` 二选一定义的 `SET_VALUE(v, i)` 宏（`NV_Ith_OMP(v,i)` vs `NV_Ith_S(v,i)`）MUST 改为 SUNDIALS generic 调用 **`N_VGetArrayPointer(v)[i]`**（与 `f.cpp` 中 6 处 `NV_DATA_OMP`/`NV_DATA_S` → `N_VGetArrayPointer` 替换对称统一；**不**使用 `NV_Ith(v,i)`——后者在 SUNDIALS 6.0.0 中是 type-specific 派生宏，无 generic 等价物，会破坏一致性）。改造后 `Macros.hpp` 中 MUST NOT 再含任何由 N_Vector 后端类型决定的 ifdef 分支；`#include "omp.h"` 与 `#include "nvector/nvector_openmp.h"` MUST 由 `SHUD_USE_OPENMP_NVECTOR=1` 守卫，OFF 时仅引入 `#include "nvector/nvector_serial.h"`。

#### Scenario: SET_VALUE 统一使用 `N_VGetArrayPointer(v)[i]` 形式

- **WHEN** 阅读 `SHUD/src/Model/Macros.hpp` 中 `SET_VALUE` 宏定义
- **THEN** 宏体 SHALL 展开为 `(N_VGetArrayPointer(v))[i] = (val)` 形式
- **AND** MUST NOT 使用 `NV_Ith(v, i, val)` / `NV_Ith_S(v,i)` / `NV_Ith_OMP(v,i)` 任一形式
- **AND** 与 `f.cpp` 6 处 `N_VGetArrayPointer` 替换对称（同一 generic 访问形式）

#### Scenario: Macros.hpp 中 SET_VALUE 不再走 N_Vector 类型 ifdef

- **WHEN** 在 `SHUD/src/Model/Macros.hpp` 上执行 `grep -n 'NV_Ith_OMP\|NV_Ith_S' SHUD/src/Model/Macros.hpp`
- **THEN** 命令 SHALL 返回 ZERO 匹配（已统一为 `N_VGetArrayPointer(v)[i]`）

#### Scenario: SET_VALUE 调用点 bitwise 行为不变

- **WHEN** 编译 Config A binary 并跑 4 个本地 case 90 天截断
- **THEN** 全部输出 SHA256 SHALL 与 B0-tag 完全相等（`SET_VALUE` 的语义改造为 refactor-equivalent）

#### Scenario: SHUD_USE_OPENMP_NVECTOR=0 时 Macros.hpp 不引入 omp 后端 header

- **WHEN** 编译 Config A（`SHUD_USE_OPENMP_NVECTOR=0`），用 `g++ -E -DSHUD_USE_OPENMP_NVECTOR=0 SHUD/src/Model/Macros.hpp 2>&1 | grep -c 'nvector_openmp'`
- **THEN** 命令 SHALL 返回 0（preprocessor 输出不含 nvector_openmp.h 引入痕迹），即使 `nvector_openmp.h` 在 `SHUD/InstallSundials/include/nvector/` 不存在 binary 也能 compile

#### Scenario: SHUD_USE_OPENMP_NVECTOR=1 时 Macros.hpp 引入 omp 后端 header

- **WHEN** 编译 Config D（`SHUD_USE_OPENMP_NVECTOR=1`），用 `g++ -E -DSHUD_USE_OPENMP_NVECTOR=1 SHUD/src/Model/Macros.hpp 2>&1 | grep -c 'nvector_openmp'`
- **THEN** 命令 SHALL 返回非 0（preprocessor 输出含 nvector_openmp.h 引入），binary 编译时该 header MUST 已被 SUNDIALS 6.0.0 安装提供

### Requirement: N_Vector 创建按 `SHUD_USE_OPENMP_NVECTOR` 分派

`SHUD/src/Model/shud.cpp` L55–L75 内 N_Vector 创建调用（`udata` / `du` 在 implicit coupled 路径）MUST 改为按 `SHUD_USE_OPENMP_NVECTOR` 决定后端。`SHUD_uncouple` 路径的 u1–u5 / du1–du5 在 S1d.2 阶段**保留 hardcoded `N_VNew_Serial`**，作 S1 阶段 scope 边界（uncouple path 在 S1 全程不参与 OpenMP NVector 验证，P8-NVector 启用时一次性补齐）；该决定在源码处加一行 comment 说明 scope intent。

- `SHUD_USE_OPENMP_NVECTOR=1` → 走 `N_VNew_OpenMP(NY, MD->CS.num_threads, sunctx)`
- `SHUD_USE_OPENMP_NVECTOR=0` → 走 `N_VNew_Serial(NY, sunctx)`（S1 阶段默认）

S1 全程默认 `SHUD_USE_OPENMP_NVECTOR=0`，vector 创建一律走 Serial 后端。Makefile MUST 仅在 `SHUD_USE_OPENMP_NVECTOR=1` 时把 `-lsundials_nvecopenmp` 加入 link flags；OFF 时 link line MUST NOT 含 `-lsundials_nvecopenmp` token。

#### Scenario: Config A binary 不链接 -lsundials_nvecopenmp

- **WHEN** 编译 Config A 后执行 `nm shud | grep -c nvecopenmp` 或 `otool -L shud | grep -c nvecopenmp`（macOS）/ `ldd shud | grep -c nvecopenmp`（Linux）
- **THEN** 命令 SHALL 返回 0；binary 不依赖 `libsundials_nvecopenmp.{so,dylib}`，即使该库从 `SHUD/InstallSundials/lib/` 移除 binary 仍能加载运行

#### Scenario: Config D binary 链接 -lsundials_nvecopenmp

- **WHEN** 编译 Config D 后执行 `nm shud | grep N_VNew_OpenMP` 或 `otool -L shud | grep nvecopenmp` / `ldd shud | grep nvecopenmp`
- **THEN** 命令 SHALL 返回非空；binary 含 `N_VNew_OpenMP` 符号引用，并依赖 `libsundials_nvecopenmp.{so,dylib}`

#### Scenario: Config A runtime 创建 Serial vector

- **WHEN** 运行 Config A binary 在 keliya 上跑 1 个 SolverStep，并在 N_Vector 创建后插入 type-id 探针（debug 用），dump 出 `N_VGetVectorID(udata)`
- **THEN** 探针 SHALL 输出 `SUNDIALS_NVEC_SERIAL`（type-id 对应 Serial 后端）

#### Scenario: Config D runtime 创建 OpenMP vector

- **WHEN** 运行 Config D smoke 测试 binary（即使后续因 ProductionOMP stub assert abort 也允许），在 N_VNew 之后立即 dump `N_VGetVectorID(udata)`
- **THEN** 探针 SHALL 输出 `SUNDIALS_NVEC_OPENMP`（type-id 对应 OpenMP 后端）

### Requirement: `N_VDestroy_Serial` 统一为 generic `N_VDestroy`

`SHUD/src/Model/shud.cpp` L111–L112 与 L323–L333 中全部 `N_VDestroy_Serial(v)` 调用 MUST 替换为 generic `N_VDestroy(v)`。Generic 接口 SHALL 按 vector type-tag 自动 dispatch 到正确 destructor（Serial / OpenMP），消除 master plan §4.19 识别的 "N_VNew_OpenMP 创建的 vector 用 N_VDestroy_Serial 销毁" 的 UB 风险。S1 阶段虽然默认仍走 Serial 后端（`SHUD_USE_OPENMP_NVECTOR=0`），但本修复 MUST 一次落地以保所有后续阶段（P8-NVector 启用 OpenMP NVector 时）安全。

#### Scenario: shud.cpp 不再含 N_VDestroy_Serial 残留

- **WHEN** 在 SHUD 源码上执行 `grep -rn 'N_VDestroy_Serial' SHUD/src/`
- **THEN** 命令 SHALL 返回 ZERO 匹配

#### Scenario: Config A 销毁路径走 generic 入口

- **WHEN** 跑 Config A keliya 90 天截断完整 run，CVODE 结束后由 `SHUD()` / `SHUD_uncouple()` 销毁 vector
- **THEN** binary SHALL 调用 `N_VDestroy(v)`，运行过程无 heap corruption（valgrind / address sanitizer 在 free 阶段无 warning），最终输出 SHA256 与 B0 完全相等

#### Scenario: Config D 销毁 OpenMP vector 不触发 type-tag 不匹配 UB

- **WHEN** 跑 Config D smoke 测试 binary（即使 ProductionOMP 路径 assert abort，N_Vector 已在 abort 前创建），程序退出时 destructor SHALL 被调用
- **THEN** `N_VDestroy(v)` SHALL 按 OpenMP type-tag dispatch 到 `N_VDestroy_OpenMP`，无 heap corruption；valgrind 在 exit 阶段 SHALL 无 `invalid free` warning（与 `N_VDestroy_Serial` 销毁 OpenMP vector 时会触发的 type 不匹配 UB 形成对比）

#### Scenario: §4.19 风险一次性解决

- **WHEN** 后续任意阶段（P8-NVector 等）翻转 `SHUD_USE_OPENMP_NVECTOR=1` 并启用 OpenMP NVector 后端
- **THEN** 销毁路径 SHALL 无需任何源码改动即可正确 dispatch（generic `N_VDestroy` 在 S1d 已落地），不再触发 §4.19 标识的 type-tag 不匹配 UB 风险

### Requirement: 多文件回归风险拆为两个 sub-PR

S1d Step 2 涉及 `Macros.hpp` / `f.cpp` / `shud.cpp` / `Makefile` 4 个文件 + 6 处宏改造 + N_Vector 接口统一，回归面广。变更交付 MUST 按 master plan R4 风险缓解拆为两个独立 sub-PR：

- **S1d-part1**：ExecPolicy 枚举接入 + `USE_RHS_CORE` 脚手架删除（已由 `exec-policy-enum` capability 覆盖）
- **S1d-part2（本 capability）**：三宏拆分 + N_Vector 访问统一 + N_VNew 分派 + N_VDestroy 改 generic

两个 sub-PR MUST 各自独立提交、独立通过 B0-tag bitwise CI、独立 review；中间任一步 bitwise 失败 MUST 阻塞下一 sub-PR 合入。每个 sub-PR 的 SHUD submodule commit MUST push 到 `SHUD-System/SHUD` 的 `openmp-baseline` 分支；外层 PR 必须 bump submodule pointer。

#### Scenario: S1d-part2 PR 独立通过 bitwise CI

- **WHEN** S1d-part2 PR（本 capability 的实现 PR）提交到 `baseline/current`，CI workflow `.github/workflows/serial-baseline.yml` 在 4 个本地 case 上自动跑 B0-tag bitwise check
- **THEN** CI SHALL 全 case PASS（4 case × 全 `B0_output/*.dat` SHA256 = B0-tag 对应文件 SHA256），CVODE stats 完全一致；CI fail 即 PR 无法 merge

#### Scenario: S1d-part2 SHUD submodule push 进 openmp-baseline 分支

- **WHEN** 实现者在 SHUD submodule 内完成 S1d-part2 改动并 push
- **THEN** push 目标 SHALL 是 `SHUD-System/SHUD` 的 `openmp-baseline` 分支（`cd SHUD && git branch --show-current` SHALL 输出 `openmp-baseline`），SHALL NOT push 到 `master`；外层 PR 描述 SHALL 明确说明 submodule pointer bump 的 from-SHA / to-SHA

#### Scenario: S1d-part1 未合入时禁止 S1d-part2 开 PR

- **WHEN** S1d-part1（ExecPolicy 枚举 + USE_RHS_CORE 脚手架删除）尚未 merge，开发者试图先开 S1d-part2 PR
- **THEN** PR review SHALL 退回，依据是 S1d-part2 依赖 `rhs_core(Serial)` 入口已稳定（part1 的产物）；强行合入 SHALL 触发 bitwise CI 失败（part2 改的 `f.cpp` ifdef 分支无 part1 的 `rhs_core()` 调度入口可走）

#### Scenario: 两个 sub-PR 合入后 status_matrix B1a 行可标 PASS

- **WHEN** S1d-part1 + S1d-part2 均已 merge，4 case CI bitwise PASS，服务器手动 heihe / heihe_x4 bitwise 也 PASS（per Migration Plan）
- **THEN** `docs/status_matrix.md` B1a 行 SHALL 从 PENDING 改为 PASS，并 SHALL 打 `B1a-tag`（annotated）锚定外层 commit + SHUD submodule pin

