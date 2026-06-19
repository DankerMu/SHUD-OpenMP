# exec-policy-enum Specification

## Purpose

S1d.1 阶段交付。引入 `enum class ExecPolicy { Serial, StrictOMP, ProductionOMP }` 枚举 + `rhs_core(Y, DY, t, ExecPolicy)` 四参重载 + `switch (policy)` 分派；`StrictOMP` / `ProductionOMP` 两分支为 `std::abort()` stub（禁用 `assert(false)` 防 `-DNDEBUG` 静默 fall-through），留给 S2 P1 实现真实并行。同时落地 `USE_RHS_CORE` 脚手架删除 + `LEGACY_RHS` A/B 编译宏引入（A/B atomic landing），本阶段只 land 接口形状与 Serial 路径走通。

## Scope

**Temporal scope note**: 本 capability requirements apply at S1d.1 completion; S1a/S1b/S1c interim states (three-parameter `rhs_core(double*, double*, double)`, `USE_RHS_CORE` 脚手架仍在) 分别由 `rhs-core-scaffolding` / `rhs-core-flux-extraction` / `rhs-core-applydy-extraction` capabilities 治理。本 capability 的 `rhs_core(Y, DY, t, ExecPolicy)` 四参重载 + switch dispatch 在 S1d.1 引入。
## Requirements
### Requirement: ExecPolicy 枚举定义与 rhs_core 调度入口

`MD_rhs_core.cpp` / 对应头文件 SHALL 定义 `enum class ExecPolicy { Serial, StrictOMP, ProductionOMP }`，`rhs_core()` 函数签名 MUST 接受 `ExecPolicy policy` 参数并在函数体内以 `switch (policy)` 分派到各分支。MUST NOT 使用 C++ 模板特化或虚函数实现 policy 分派（避免 binary 路径膨胀与 vtable 开销，且与 SHUD 既有 C 风格 enum 调度一致）。

#### Scenario: ExecPolicy 枚举在头文件可见且包含三个值

- **WHEN** 在 SHUD 源码中 grep `enum class ExecPolicy`
- **THEN** 命中 `MD_rhs_core.cpp` 或其头文件中的定义
- **AND** 枚举体恰好包含 `Serial` / `StrictOMP` / `ProductionOMP` 三个值，无其他成员
- **AND** `rhs_core` 函数签名以 `ExecPolicy policy` 作为参数之一

#### Scenario: rhs_core 内部以 switch 分派 policy

- **WHEN** 阅读 `MD_rhs_core.cpp` 中 `rhs_core()` 函数体
- **THEN** 函数体以 `switch (policy)` 形式分派
- **AND** 三个 `case` 标签分别对应 `ExecPolicy::Serial` / `::StrictOMP` / `::ProductionOMP`
- **AND** 源码中无 `template <ExecPolicy>` 特化与 `virtual ... rhs_core` 声明

### Requirement: S1 阶段仅 Serial 分支可执行；OMP 分支为 std::abort stub

S1 阶段 `rhs_core()` SHALL 仅在 `case ExecPolicy::Serial` 分支中实现真实调用链（`rhs_update` → `rhs_flux` → `rhs_apply`）；`StrictOMP` 与 `ProductionOMP` 两个 `case` MUST 在分支入口调用 `std::abort()` 或 `__builtin_trap()`（**禁用 `assert(false)`** —— `-DNDEBUG` 会把 `assert` 整段 strip 成 no-op，导致 release build 静默 fall-through，违背 stub 的本意）。MUST NOT 让 OMP 分支以任何方式静默 fall-through 到 Serial（避免误把 OMP 路径当 Serial 跑出"绿"的 bitwise）。

#### Scenario: Serial 分支调用三段抽取函数

- **WHEN** 传入 `ExecPolicy::Serial` 调用 `rhs_core(Y, DY, t, ExecPolicy::Serial)`
- **THEN** 依次执行 `rhs_update` / `rhs_flux` / `rhs_apply`
- **AND** 不命中任何 abort
- **AND** 无 fallback 调用原 `Model_Data::f_update` / `f_loop` / `f_applyDY`（USE_RHS_CORE 脚手架已删，见后述）

#### Scenario: StrictOMP / ProductionOMP 分支 runtime abort

- **WHEN** 在 `SHUD_ENABLE_OPENMP_RHS=1` 编译产物中 runtime 传入 `ExecPolicy::StrictOMP` 或 `ExecPolicy::ProductionOMP`
- **THEN** 进入对应 `case` 后立即调用 `std::abort()` 终止进程
- **AND** MUST NOT 使用 `assert(false)`（会被 `-DNDEBUG` strip）
- **AND** MUST NOT 静默调用 Serial 分支充数
- **AND** 该路径仅作 S1 编译可达性证据，不参与 bitwise 验收

#### Scenario: NDEBUG 编译下 stub 仍然 abort（regression guard）

- **WHEN** 用 `make shud SHUD_ENABLE_OPENMP_RHS=1 EXTRA_CXXFLAGS=-DNDEBUG` 在 locked-flag 工具链下编译，并在单元测试中调用 `rhs_core(Y, DY, t, ExecPolicy::StrictOMP)`
- **THEN** 子进程 MUST 以 SIGABRT 退出（`waitpid` 返回的 status 满足 `WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT`）
- **AND** stub 行为 SHALL 与未带 `-DNDEBUG` 时一致
- **AND** 该场景显式守护"`assert(false)` 退役为 `std::abort()`"的决策，杜绝回退到 `assert`

#### Scenario: Config C runtime stub 进程级 abort 烟囱测试

- **WHEN** 编译 `tests/s1d_strictomp_assert_smoke.cpp`（fork 子进程调用 `rhs_core(..., ExecPolicy::StrictOMP)`，parent 用 `waitpid` 取 status）
- **THEN** 父进程 MUST 观测到 `WIFSIGNALED(status) == 1` 且 `WTERMSIG(status) == SIGABRT`
- **AND** ProductionOMP 分支同等行为
- **AND** 子进程 MUST NOT 以正常 `exit(0)` 终止（说明 stub 起作用）

### Requirement: USE_RHS_CORE 脚手架在 S1d.2 后完全移除

S1d.2 完成后，`f.cpp` SHALL 默认无条件调用 `rhs_core(ExecPolicy::Serial)`，MUST NOT 再通过 `#ifdef USE_RHS_CORE` / `#ifndef USE_RHS_CORE` 在新旧路径间二选一。S1a-S1c 期间使用的混合 fallback（部分新 + 部分 legacy）SHALL 不再出现在最终 S1d 提交里。

#### Scenario: 源码中 USE_RHS_CORE 宏完全消失

- **WHEN** 在 SHUD 源码内 `grep -rn 'USE_RHS_CORE' SHUD/src/`
- **THEN** 在 `SHUD/src/Model/f.cpp` 与 `SHUD/src/Model/MD_rhs_core.cpp`（或对应头文件）中无任何命中
- **AND** Makefile / CMakeLists 中无 `-DUSE_RHS_CORE` 定义
- **AND** docs / examples 中如有提到 USE_RHS_CORE 仅作历史脚手架说明，不再控制实际编译

#### Scenario: f.cpp 默认路径直达 rhs_core(Serial)

- **WHEN** 阅读 `SHUD/src/Model/f.cpp` 中 `f()` 函数体
- **THEN** 默认（无任何宏定义）走 `rhs_core(Y, DY, t, ExecPolicy::Serial)` 一条路径
- **AND** 无 `if (use_rhs_core) { ... } else { Model_Data::f_update(); Model_Data::f_loop(); Model_Data::f_applyDY(); }` 这类运行时切换
- **AND** legacy 路径仅在下述 LEGACY_RHS 编译宏下激活

#### Scenario: S1d.1 Step 4.4 + 4.5 atomic landing enforced at review time

- **WHEN** S1d.1 PR diff 中包含 `-#ifdef USE_RHS_CORE` 删除 hits 但**没有**对应的 `+#ifdef LEGACY_RHS` 在同一 commit 引入
- **THEN** PR MUST 被拒绝（legacy 路径 SHALL 始终可达；`USE_RHS_CORE` 已删但 `LEGACY_RHS` 尚未引入的中间状态 MUST NOT 出现在 commit history 中）
- **AND** 反方向同样禁止：引入 `+#ifdef LEGACY_RHS` 但**未**删除 `-#ifdef USE_RHS_CORE` 的中间状态在 S1d.1 close 时 SHALL 同样被拒绝
- **AND** Step 4.4（删 USE_RHS_CORE 脚手架）与 Step 4.5（引入 LEGACY_RHS A/B 宏）MUST 作为同一 atomic commit 提交，reviewer SHALL grep PR diff 确认两类宏变更同时存在

### Requirement: LEGACY_RHS 编译宏保留 A/B 对比能力

`LEGACY_RHS` 编译宏 SHALL 保留以支持 B0 路径与 B1a 路径在同一份源码下 A/B bitwise 对比。`LEGACY_RHS=1` 编译产物 MUST 让 `f()` 入口路由到原始 `Model_Data::f_update` / `f_loop` / `f_applyDY`（B0 二进制等价路径，源码位于 `SHUD/src/ModelData/MD_update.cpp` / `MD_f.cpp`）；`LEGACY_RHS=0`（默认）MUST 路由到 `rhs_core(ExecPolicy::Serial)`（B1a 路径）。两种路径在 keliya / xinanjiang_upstream / qinyijiang / qhh 四个 local case 上 SHALL 与 `B0-tag` 归档的输出 bitwise identical（SHA256 完全相等）。CVODE 统计量按 canonical 15-key set（详 `openspec/glossary.md` §CVODE canonical 15-key set）通过 `tools/cvode_stats_diff/cvode_stats_diff.sh` 全键对比，两种编译下 MUST 与 B0 一致（F19 修订：归档不含 `nFCall`，由独立 capability 跟踪）。

#### Scenario: LEGACY_RHS=0 默认编译 vs B0 bitwise PASS（4 local case）

- **WHEN** 用 `make shud LEGACY_RHS=0`（即默认）编译，逐一跑 keliya / xinanjiang_upstream / qinyijiang / qhh
- **THEN** 每个 case 的 `B0_output/*.dat` SHA256 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全相等
- **AND** `tools/cvode_stats_diff/cvode_stats_diff.sh <run.txt> <(git show B0-tag:benchmarks/<case>/B0_output/cvode_stats.txt)` exit code = 0（canonical 15-key set 全键命中且数值完全相等；详 `openspec/glossary.md` §CVODE canonical 15-key set）

#### Scenario: LEGACY_RHS=1 编译 vs B0 bitwise PASS（4 local case）

- **WHEN** 用 `make shud LEGACY_RHS=1` 编译，逐一跑同样四个 local case
- **THEN** 每个 case 输出与 `B0-tag` 归档 SHA256 完全相等
- **AND** `tools/cvode_stats_diff/cvode_stats_diff.sh` canonical 15-key 对比 exit code = 0
- **AND** 该路径走原 `Model_Data::f_update` / `f_loop` / `f_applyDY`（`SHUD/src/ModelData/`），不进入 `rhs_core`

### Requirement: LEGACY_RHS=0 binary 保留 legacy 符号以便复现

`LEGACY_RHS=0` 默认编译产物 SHALL 保留 `Model_Data::f_update` / `Model_Data::f_loop` / `Model_Data::f_applyDY` 三个符号在 binary 符号表中（供 A/B 复现与 future debugging 直接调用，无需重编 LEGACY_RHS=1）。源码 MUST NOT 给这三个函数加 `__attribute__((unused))` / `[[maybe_unused]]` 等会触发 linker GC 的属性；Makefile MUST NOT 给该 binary 加 `-ffunction-sections` 与 `--gc-sections` 组合（否则 unreferenced legacy 函数会被裁掉）。

#### Scenario: LEGACY_RHS=0 binary 仍含 legacy 符号

- **WHEN** 用 `make shud LEGACY_RHS=0` 编译产生 `shud` binary，并执行 `nm shud | grep -E 'f_update|f_loop|f_applyDY' | grep -v _omp`
- **THEN** 命令 SHALL 至少返回 3 行（三个 legacy 函数符号各 1 条）
- **AND** 三符号 type MUST 是 `T` / `t`（text section），不是 `U`（undefined）也不是缺失

#### Scenario: Linker GC 未误裁 legacy 函数

- **WHEN** 在 LEGACY_RHS=0 binary 的 link map 上 grep `f_update` / `f_loop` / `f_applyDY`
- **THEN** 三函数 SHALL 出现在 `text` section
- **AND** Makefile MUST NOT 同时含 `-ffunction-sections` 与 `--gc-sections`/`-Wl,--gc-sections`

### Requirement: SHUD_ENABLE_OPENMP_RHS 宏控制 OMP 分支编译可见性

S1 默认编译（`SHUD_ENABLE_OPENMP_RHS=0`）MUST 通过 `#ifdef SHUD_ENABLE_OPENMP_RHS` 把 `StrictOMP` / `ProductionOMP` 两个 `case` 与其内部代码整段排除出 translation unit；最终 binary MUST NOT 包含 OMP 路径相关符号。`SHUD_ENABLE_OPENMP_RHS=1` 编译 SHALL 让 OMP 分支代码进入 binary 并保留 `std::abort()` stub（-DNDEBUG 安全：与 `assert(false)` 不同，`std::abort()` 在 release build 下仍立即 SIGABRT，不会被 strip 成 no-op），编译 MUST 成功，runtime 调用 MUST abort。`SHUD_ENABLE_OPENMP_RHS=1` 仅作 S1 smoke compile 测试，不参与 bitwise 验收。

#### Scenario: 默认编译 binary 不含 OMP 路径符号

- **WHEN** 用 `make shud`（默认 `SHUD_ENABLE_OPENMP_RHS=0`）编译产生 `shud` binary
- **THEN** `nm shud | grep -i 'StrictOMP\|ProductionOMP'` 无命中
- **AND** `rhs_core` 函数体内对应 OMP 分支已被预处理器移除
- **AND** binary size 与 LEGACY_RHS=0 默认 baseline 无显著膨胀

#### Scenario: SHUD_ENABLE_OPENMP_RHS=1 编译成功且 runtime abort

- **WHEN** 用 `make shud SHUD_ENABLE_OPENMP_RHS=1` 编译
- **THEN** 编译成功，binary 产出
- **AND** 该 binary 若被调用并以 `ExecPolicy::StrictOMP` / `::ProductionOMP` 进入 `rhs_core`，runtime 触发 `std::abort()` 并 abort（-DNDEBUG 编译下仍立即 SIGABRT，不静默 fall-through）
- **AND** 该模式不跑 bitwise vs B0，仅做编译可达性 smoke test

### Requirement: _omp 路径在 S1 全程不变

`f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 三个 legacy OpenMP 函数源码 SHALL 在 S1 全阶段（含 S1d）保持不变，作为 S2 语义对齐参照。S1 阶段 `SHUD_LEGACY_OMP_RHS=0` 默认让这三个函数不参与编译（不进入 binary），但源码 MUST 保留在仓库内。MUST NOT 在 S1 任何 PR 内对这三个函数做哪怕一处修改（包括注释、空格、include 顺序调整）。

#### Scenario: S1 期间 _omp 三函数源码与 B0-tag 完全一致

- **WHEN** 在 S1d 完成时执行 `git diff B0-tag -- SHUD/src/ModelData/MD_f_omp.cpp` 仅检查 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 三个函数体
- **THEN** 三函数体 diff 为空
- **AND** 函数签名、include 依赖、函数顺序与 B0-tag 完全相同
- **AND** `SHUD_LEGACY_OMP_RHS=0` 默认编译 binary 不含这三个 `_omp` 符号

#### Scenario: SHUD_LEGACY_OMP_RHS=1 编译能保留 _omp 路径作 S2 参照

- **WHEN** 用 `make shud SHUD_LEGACY_OMP_RHS=1` 编译
- **THEN** 编译成功，binary 含 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 符号
- **AND** 该 binary 仅作 S2 启动前的 legacy 对比参照，不在 S1 验收清单内
- **AND** S1 期间该宏不开启用于 bitwise 验证

