# S2 语义差异报告 (Semantic Diff Report) — record-only 子项处置决策

## 背景与定义

本文档依据 `openspec/changes/b1a-finalization/specs/s2-semantic-merge/spec.md` 中"`docs/b1b/s2_semantic_diff_report.md` schema" Requirement 的规定编写。S2 阶段对 SHUD 源码进行语义合并 (semantic merge) 审计，其中部分子项属于"仅记录 (record-only)"类别——即识别出差异但暂不在 B1a 阶段实施代码修改，理由是相关改动可能破坏 B1a 与 B0 之间的按位一致 (bitwise identical) 约束，或属于跨阶段决策耦合 (cross-stage decision coupling) 范畴，应推迟至 B1b 或更后续阶段处理。

每个 record-only 子项依统一的 7 字段 schema 描述：master plan 引用、文件行号、风险与观测、处置决策、推迟理由、验证手段、关联未来 issue。

**PR 范围说明**：

- **PR-7a**：S2.12、S2.13、S2.15、S2.16 共 4 个 record-only 子项。
- **PR-7b** (本 PR)：S2.17 (含 lake 分支 assert、DEBUG 模式 qhh 运行、RELEASE 模式 4 算例 bitwise 验证及记录)，五个 section 全部完成。

## S2.12 uncoupled 路径 clamp 差异

**Master plan reference:**
> **S2.12 — uncoupled 路径 clamp** — **差异**：`f_updatei()` 统一做 `max(0,Y)` clamp；coupled `f_update()` 不 clamp ——**原则**：记录差异；当前方案以 coupled 路径为主线，uncoupled 暂不纳入并行改造 (master plan §5 S2.12)

**File:line(s):** `SHUD/src/ModelData/MD_update.cpp:3-59` (`f_updatei()`，uncoupled 路径) 与 `SHUD/src/ModelData/MD_update.cpp:60-147` (`f_update()`，coupled 路径)。

**Risk / observation:** uncoupled 路径中 `f_updatei()` 对状态向量 Y[*] 实施 `max(0, Y)` 截断 (clamp) 以防止负状态出现，而 coupled 路径 `f_update()` 不做此截断，依赖 CVODE 数值积分器在时间步内部维持解的非负性。两条路径在数值处理上语义不一致。若 B1a 至 B1b 阶段切换主线 (例如 P1+ 阶段引入并行 uncoupled 路径)，则需要先行统一截断策略。

**Decision (record-only / defer to B1b / etc.):** record-only。uncoupled 路径不纳入 B1a 阶段的并行改造范围，截断差异保持原状。

**Reason for deferral:** 当前 B1a 主线为 coupled 路径 (`f_update`)；uncoupled 路径仅在特定配置下触发 (即 `Forcing.csv` 单元素脱耦合配置)，并非 NWM/SHUD 的主流用例 (use case)。统一截断策略需要先行收敛 B1b 阶段的负状态修复方案 (S2.6 carve-out 与 uYgw 对齐议题 #159)；过早合并会引入跨阶段决策耦合。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/ModelData/MD_update.cpp` 应不包含 L3-59 范围内的实质改动 (PR-7a 不触及该文件)。

**Linked future issue (if any):** 无。B1b Epic 启动时再开 follow-up issue，可参照 #159 uYgw P1+ alignment 模板。

## S2.13 全局变量裸指针

**Master plan reference:**
> **S2.13 — 全局变量裸指针（记录，不迁移）** — **差异**：`uYsf/uYus/uYgw/uYriv/uYlake/timeNow` 是全局变量，非 `Model_Data` 成员 —— **P1-P7 影响**：strict OpenMP 下 CVODE 仍单线程调 `f()`，全局指针在 RHS 入口赋值后在 parallel region 内只读，无竞争。**不是 P1-P7 前置条件** —— **S2 阶段原则**：仅**记录**风险和当前使用模式（哪些函数读、哪些写、赋值时机），不做迁移 —— **迁移时机**：推迟到 P8+ / LibSHUD / Reentrant RHS 阶段 (master plan §5 S2.13)

**File:line(s):**
- `uYsf` / `uYus` / `uYgw` — `SHUD/src/Model/Macros.hpp` (global `extern` decl, L126-134) 加 `SHUD/src/Model/MD_rhs_core.cpp` `rhs_update()` (write in RHS entry, read in flux/apply)
- `uYriv` / `uYlake` — same pattern (`SHUD/src/Model/Macros.hpp`)
- `timeNow` — global time variable (`SHUD/src/Model/Macros.hpp`), written in `f()` entry, read by all flux functions

**Risk / observation:** 上述 6 个全局裸指针 (raw pointer) 是 RHS 入口 (`f()` 或 `rhs_core()`) 对 `N_Vector y` 执行 `N_VGetArrayPointer` 之后的便捷别名 (alias)。在 strict OpenMP (P1-P7) 模式下，CVODE 单线程调用 `f()`，并行区 (parallel region) 在内部 NumEle/NumRiv 循环内启动，全局指针在并行区内仅作只读访问，不存在竞争 (race)。然而，在可重入 RHS (Reentrant RHS，P8+ 阶段，例如 Jacobian 并行差分估计需要并发调用多次 `f()`) 场景下，全局可变指针将产生冲突。

| 变量 | 读 by | 写 by | 赋值时机 |
|---|---|---|---|
| `uYsf` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iSF_offset` |
| `uYus` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iUS_offset` |
| `uYgw` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iGW_offset` |
| `uYriv` | `rhs_flux()` / `rhs_apply()` (river loop) | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iRIV_offset` |
| `uYlake` | `rhs_flux()` / `rhs_apply()` (lake loop) | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iLAKE_offset` |
| `timeNow` | 所有 flux 函数 | `f()` / `rhs_core()` entry | RHS entry，赋 `t` 参数 |

**Decision (record-only / defer to B1b / etc.):** record-only。6 个全局指针保持现状，不收编为 `Model_Data` 类成员。

**Reason for deferral:** 推迟至 **P8+ / LibSHUD / Reentrant RHS** 阶段处理。迁移涉及 `Macros.hpp` 的宏重构以及大量函数签名变更 (所有 flux 函数需新增 `Model_Data&` 参数)，改动面巨大。当前 B1a 与 P1-P7 阶段并不需要此项重构 (CVODE 单线程调用 `f()` 已足够)，提前实施违反 KISS 与 YAGNI 原则。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/Model/Macros.hpp` 应不包含 6 个全局变量声明的实质改动；`grep -E 'extern.*(uYsf|uYus|uYgw|uYriv|uYlake|timeNow)' SHUD/src/Model/Macros.hpp` 应保持 B0-tag 状态。

**Linked future issue (if any):** 无。P8 Reentrant RHS 阶段启动时再开 Epic 跟踪。

## S2.15 AccTemperature 除零

**Master plan reference:**
> **S2.15 — `AccTemperature.getACC()` 除零** — **差异**：`que.size()==0` 时除零 → NaN —— **S2 阶段原则**：仅**记录**此 bug，不在 S2 修复——修复会改变输出，违反 B1a == B0 的约束 —— **B1b 阶段修复**（S6a/S6b.1）：加 guard `que.empty() ? 0.0 : ACC/que.size()` (master plan §5 S2.15 + §6 S6b.1)

**File:line(s):** `SHUD/src/classes/AccTemperature.hpp:60-62` (`getACC()` method)

**Risk / observation:** 当 `que.size() == 0` 时 (此情形发生于初始化早期，即冰冻圈 (cryosphere) 模块启用且模拟前 1440 分钟内 24 小时平均队列尚未填满)，`return ACC / que.size()` 会触发整数除零，返回 NaN 或 Inf。该 NaN 可沿冻土温度依赖路径传播至 `qElePrep` 与 `qEleInfil`，污染前 1440 分钟内的整体输出。

**Decision (record-only / defer to B1b / etc.):** 推迟至 B1b 阶段 S6b.1 修复。

**Reason for deferral:** 增加守卫 (guard) 将改变模型输出 (修前为 NaN，修后为 0.0)，违反 B1a 与 B0 之间按位再现性 (bitwise reproducibility) 约束。此修复属于"输出改变类 bug fix"范畴 (master plan §5 表中与 S2.15、S2.17 同类)，必须在独立的 B1b 阶段实施，避免与 OpenMP 并行加速 (B1a 主线) 改动混合，造成审计时难以归因输出变化的来源。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/classes/AccTemperature.hpp` 应不包含 L60-62 范围内的实质改动；`getACC()` 方法体保持 B0-tag 状态 (仍存在潜在的除零路径)。

**Linked future issue (if any):** 无。B1b Epic 启动时开 issue 跟踪 S6b.1。

## S2.16 OpenMP N_Vector 当前使用

**Master plan reference:**
> **S2.16 — 当前已使用 OpenMP N_Vector** — **差异**：`shud.cpp` L58-59 已用 `N_VNew_OpenMP` —— **原则**：**已由 S1d.3-S1d.5 宏解耦解决**。N_Vector 后端由 `SHUD_USE_OPENMP_NVECTOR` 独立控制，S1 阶段默认 OFF → `N_VNew_Serial`；P8-NVector 才允许打开。S2 阶段无需额外处理此项 (master plan §5 S2.16 + §4.21 + §4.13)

**File:line(s):** `SHUD/src/Model/shud.cpp:58-59` (`N_Vector y = N_VNew_OpenMP(NEQ, NumThreads, sunctx)` 与 `N_VNew_Serial(NEQ, sunctx)` 的切换点)。

**Risk / observation:** B0 阶段 `shud.cpp:58-59` 默认硬编码使用 `N_VNew_OpenMP`，与 SUNDIALS-CVODE 的 OpenMP 后端 (backend) 静态绑定，引发跨平台 (serial-only build) 编译错误，并影响数值再现性 (即 D13 OpenMP NVector backend baseline gap)。

**Decision (record-only / defer to B1b / etc.):** **无需新增动作 (zero action)**。该议题已由 S1d.3 至 S1d.5 三步宏解耦工作解决：

- **S1d.3** 引入 `SHUD_USE_OPENMP_NVECTOR` 编译宏，默认 OFF，对应 `N_VNew_Serial`。
- **S1d.4** 统一所有 `N_VGetArrayPointer` 调用，避免后端特化 (specialization)。
- **S1d.5** 以 `N_VDestroy` 通用接口替代 `N_VDestroy_OpenMP`。

S2 阶段无需额外处理；本文档仅记录"已解决"状态以备审计追溯。

**Reason for deferral:** 不适用 (已 resolved)。开启 OpenMP N_Vector 后端的工作推迟至 P8-NVector 阶段，并需配套 D13 baseline regression 验收。

**Verification:**
- `grep -nE 'SHUD_USE_OPENMP_NVECTOR' SHUD/src/Model/shud.cpp` 结果应 ≥ 1 (S1d.3 宏存在)。
- `grep -nE 'SHUD_USE_OPENMP_NVECTOR \?= 0' SHUD/Makefile` 结果应 = 1 (默认 OFF)。
- `git diff B0-tag HEAD -- SHUD/src/Model/shud.cpp | grep -cE '^\+.*N_VNew_OpenMP'` 应 ≤ N_VNew_Serial 分支数 (S1d.3 宏 if-else 双分支保留)。

**Linked future issue (if any):** D13 (位于 `openspec/changes/b1a-finalization/design.md` D13 条目，即 OpenMP NVector backend reduction-order baseline gap)；P8-NVector Epic 启动时开 follow-up。

## S2.17 fun_Ele_sub lake 分支 assert

**Master plan reference:**
> **S2.17 — `fun_Ele_sub()` lake 分支 `Ele[inabr].u_effKH` 越界/语义风险**（⚠️ blocker） — **问题**：`MD_ElementFlux.cpp` L107–L121，lake 分支进入条件是 `ilake >= 0`，但 L117 计算 `Kmean` 时使用了 `Ele[inabr].u_effKH`（`inabr = Ele[i].nabr[j] - 1`），**未检查 `inabr >= 0`** —— **S2 阶段原则**：在 `fun_Ele_sub()` lake 分支入口加 `assert(inabr >= 0)` 防御性检查（不影响 release 输出）+ 审查并**记录** lake element 的 `u_effKH` 赋值来源和物理意义分析结论 (master plan §5 S2.17 + §4.18)

**File:line(s):** `SHUD/src/ModelData/MD_ElementFlux.cpp:107-121` (lake 分支位于 `fun_Ele_sub()`，源 L101 起)。

**Risk / observation:** lake 分支 (L107 `if(ilake >= 0)`) 仅检查 `ilake` 是否合法，未检查 `inabr`；但在 L119 `Kmean = 0.5 * (Ele[i].u_effKH + Ele[inabr].u_effKH)` 处直接索引 `Ele[inabr]`，其中 `inabr = Ele[i].nabr[j] - 1` (源 L105)。若 `Ele[i].nabr[j] == 0` 则 `inabr == -1`，触发未定义行为 (UB, `Ele[-1].u_effKH`)。依据数据流分析 (master plan §4.18)，`MD_Lake.cpp` L133–L144 在为 `lakenabr[j]` 赋值前已检查 `inabr >= 0`，因此当前数据流下 `inabr` 在 lake 分支内**应当**合法；但该保证为隐式约束，源码层缺乏显式断言。另存有物理语义疑问：lake element 的 `u_effKH` 是否具备物理意义？取算术平均 `0.5 * (Ele[i] + Ele[inabr])` 是否合理？

**Decision (record-only / defer to B1b / etc.):** S2 阶段添加 `assert(inabr >= 0)` 作为防御性 DEBUG 检查；`u_effKH` 公式变更推迟至 B1b/S6a 阶段处理。

**Reason for deferral:** 修改 `Kmean` 公式 (例如改为仅使用 `Ele[i].u_effKH`，或引入 lake-bed 导水率参数) 将改变 lake 算例的输出，违反 B1a 与 B0 之间的按位再现性约束。`assert(inabr >= 0)` 仅在 DEBUG 构建 (即未定义 `NDEBUG` 时) 生效；在 RELEASE 构建 (`-DNDEBUG`) 下，编译器会将该断言剥离为空操作 (no-op)，因此**不影响输出按位等价性**。公式审查及物理语义结论归入 B1b/S6a 阶段 (master plan §6 S6b.2)。

**Verification:**
- DEBUG 构建 (默认 `make shud`，未定义 `NDEBUG`) 下执行 qhh 90 天运行 (qhh 是本地唯一含 lake 的算例，`NumLake=1`，`NumEle=4773`)：`assert(inabr >= 0)` 未触发，正常退出 0；此为该步骤的成功判据 (即 sentinel 未触发表明数据流分析正确，`inabr` 在 lake 分支内确实合法)。其余 4 个 Mac 本地算例 (keliya、xinanjiang_upstream、qinyijiang、kashigeer) 不含 lake element，assert 路径为死代码 (dead code)，可跳过 DEBUG 运行。Server 侧 heihe 与 heihe_x4 的 DEBUG/RELEASE 验证推迟至 PR-12 capstone (依据 `openspec/changes/b1a-finalization/design.md` D10)。
- RELEASE 构建 (`make shud EXTRA_CXXFLAGS=-DNDEBUG`) 下完成 Mac 本地 4 算例 (keliya、xinanjiang_upstream、qinyijiang、qhh；kashigeer 因上游 forcing 缺失而 deferred — 参见 `benchmarks/kashigeer/B0_output/DEFERRED.txt`) 验证：所有 B0_output `.dat` 文件 (含全部算例的 rivqdown.dat、qhh 的 lakystage.dat / lakqrivin.dat / lakqrivout.dat 及 xinanjiang_upstream 的 eleygw.dat) 的 SHA256 应等于 B0-tag 归档值；cvode_stats 15 项关键字段亦按位等于 B0-tag 归档。
- `git diff B0-tag HEAD -- SHUD/src/ModelData/MD_ElementFlux.cpp` 应仅包含 `#include <cassert>` (新增于 L2) 与 `assert(inabr >= 0)` (新增于 L109) 两处改动，不含 L117 `Kmean` 公式或其他 lake 分支计算逻辑的变化。

**Linked future issue (if any):** 无。B1b/S6a Epic 启动时开 issue 跟踪 `u_effKH` 公式 review；master plan §6 S6b.2 已将该项列入"输出改变类 bug fix"待修清单。

## Section 完整性验证

PR-7b 落地后的完整性状态如下：

1. 全部 5 个 section (S2.12 / S2.13 / S2.15 / S2.16 / S2.17) 齐全，每个 section 的 7 字段完整。
2. PR-7a 涵盖 4 个 record-only section，对应 0 处 SHUD 源码改动。
3. PR-7b 包含 1 个 section (S2.17) 加 1 处 SHUD 源码改动 (即 DEBUG 模式专属的防御性 `assert`，含 `#include <cassert>` 与 1 行 assert)；RELEASE 构建相对 B0 的按位等价性予以保持。

`git diff B0-tag <PR-7b HEAD> -- SHUD/src/` 仅应包含 `MD_ElementFlux.cpp` 的 2 行新增 (`cassert` include 及 assert)；其余 PR-7a record-only 涉及的 file:line 范围内不应出现实质改动。

## Master plan 一致性

每个 section 的 **Master plan reference** 字段逐字 (verbatim) 引用 master plan §5 S2.x 段落的开头一句话，以确保 record 内容与上游 spec 保持一致。master plan §5 的任何更新均需通过后续 docs sync PR 在本文档对应 section 中同步修订。

## S2 capstone 时间线

PR-8 (#152) 为 S2 capstone landing PR，负责完成 `_omp` RHS receivers 的实装以及退役分叉 (退役-fork) 的实际删除工作，将 S2.1 至 S2.17 的 record-only 决策落实到代码层。

**SHUD submodule (`openmp-baseline` 分支) 源码层改动：**

1. 删除 `SHUD/src/ModelData/MD_f_omp.cpp` 整个文件 (共 176 行；该文件保存 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 三个 legacy `_omp` 函数体，S1 阶段冻结，至 PR-8 阶段退役)。
2. 删除 `SHUD/src/ModelData/Model_Data.hpp` 中 3 个 `_omp` 方法声明 (L261–263)。
3. 简化 `SHUD/src/Model/f.cpp`：`f()` 不再依 `#ifdef LEGACY_RHS` 分叉，无条件调用 `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)`；PURE CARRY-OVER 性质的 `rhs_update / rhs_flux / rhs_apply` 仍是 `f_update / f_loop / f_applyDY` 的逐字节 (byte-for-byte) 复制源，B0 baseline 予以保留。
4. 清理 `SHUD/src/Model/MD_rhs_core.cpp`、`MD_rhs_core.hpp`、`Macros.hpp`、`classes/CommandIn.cpp` 中所有提及 `LEGACY_RHS` 或 `SHUD_LEGACY_OMP_RHS` 的注释。
5. 退役 `SHUD/Makefile` 中 `LEGACY_RHS ?= 0` 与 `SHUD_LEGACY_OMP_RHS ?= 0` 两个宏块及其 `$(LEGACY_RHS_DEFINE)` / `$(SHUD_LEGACY_OMP_DEFINE)` 编译行引用、`make shud LEGACY_RHS=…` 与 `make shud SHUD_LEGACY_OMP_RHS=…` help 文字以及 MD_f_omp.cpp filter-out 逻辑；保留 `SHUD_USE_OPENMP_NVECTOR`、`SHUD_ENABLE_OPENMP_RHS`、`SHUD_DUMP_RHS`、`SHUD_ENABLE_PROFILE` 等与本 capstone 正交的宏。

**树范围 grep 门禁 (deterministic, 0 hits)：**

- `ls SHUD/src/ModelData/MD_f_omp.cpp` — file not found (整个 TU 已删除)。
- `grep -rn 'f_update_omp\|f_loop_omp\|f_applyDY_omp' SHUD/src/` — 0 hits。
- `grep -rn 'SHUD_LEGACY_OMP_RHS\|LEGACY_RHS' SHUD/src/ SHUD/Makefile` — 0 hits。

**CI workflow 更新 (`.github/workflows/serial-baseline.yml`)：**

1. 编译矩阵由 2 轴 (`rhs_path × case`) 折叠为 1 轴 (`case` only)。fast-feedback PR 运行 keliya 单作业；full-bitwise 与 nightly cron 运行 4 个 case 4 作业。
2. 新增 `S2 capstone grep gate` 步骤：上述 3 项 grep 门禁在 CI 每次运行中执行 (data-independent，仅做源码层检查)。
3. 移除所有 `${{ matrix.rhs_path }}` 字符串引用以及 `make … LEGACY_RHS=…` 编译参数；message、log、artifact 命名同步去除 `axis=` 标记。

**Verification：**

1. 默认 `make shud` 在 Mac 本地 (Apple Silicon) 编译成功。
2. Mac 本地 4 算例 (`keliya / xinanjiang_upstream / qinyijiang / qhh`) 90 天窗口运行：`rivqdown.dat` 的 SHA256 等于 B0-tag `benchmarks/<case>/B0_output/*.dat`；`qhh` lake 输出 (`lakystage.dat` / `lakqrivin.dat` / `lakqrivout.dat`) 的 SHA256 等于 B0-tag 归档。CVODE 15 项关键字段不变性在 PROFILE=1 构建下应同样成立 (PR-12 capstone 再行验证)。
3. `kashigeer` 上游 forcing-gap 的既有 deferred 状态保持不变 (参见 `benchmarks/kashigeer/B0_output/DEFERRED.txt`，S0-13 N/A 重分类)。
4. Server 侧 `heihe` 与 `heihe_x4` 的 DEBUG/RELEASE 验证依 `openspec/changes/b1a-finalization/design.md` D10 推迟至 PR-12 capstone。
