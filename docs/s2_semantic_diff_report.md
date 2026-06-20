# S2 Semantic Diff Report — record-only 子项处置决策

本文档按 `openspec/changes/b1a-finalization/specs/s2-semantic-merge/spec.md`
"`docs/s2_semantic_diff_report.md` schema" Requirement 编写。每个 record-only 子项
按 7 字段 schema 描述。

**PR 范围说明**：
- **PR-7a (本 PR)**：S2.12 / S2.13 / S2.15 / S2.16 共 4 项 record-only
- **PR-7b**：S2.17（lake 分支 assert + DEBUG 6-case run + record） — 单独 PR

## S2.12 uncoupled 路径 clamp 差异

**Master plan reference:**
> **S2.12 — uncoupled 路径 clamp** — **差异**：`f_updatei()` 统一做 `max(0,Y)` clamp；coupled `f_update()` 不 clamp ——**原则**：记录差异；当前方案以 coupled 路径为主线，uncoupled 暂不纳入并行改造 (master plan §5 S2.12)

**File:line(s):** `SHUD/src/ModelData/MD_update.cpp:3-59` (`f_updatei()`，uncoupled 路径) vs `SHUD/src/ModelData/MD_update.cpp:60-147` (`f_update()`，coupled 路径)

**Risk / observation:** uncoupled 路径 `f_updatei()` 对 Y[*] 做 `max(0, Y)` clamp（防负状态），coupled 路径 `f_update()` 不做 clamp（依赖 CVODE 数值积分保持非负）。两条路径语义不一致，若 B1a → B1b 阶段切换主线（如 P1+ 引入并行 uncoupled），需统一 clamp 策略。

**Decision (record-only / defer to B1b / etc.):** record-only — uncoupled 路径不纳入 B1a 并行改造，clamp 差异保持原状。

**Reason for deferral:** 当前 B1a 主线 = coupled 路径 (`f_update`)；uncoupled 路径只在特定 case (`Forcing.csv` 单元素脱耦合配置) 触发，不属于 NWM/SHUD 主流 use case。统一 clamp 策略需要先收敛 B1b 阶段的负状态修复策略 (S2.6 carve-out + uYgw alignment issue #159)；提前合并会引入跨阶段决策耦合。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/ModelData/MD_update.cpp` SHALL 不包含 L3-59 范围内的实质改动（PR-7a 不动该文件）。

**Linked future issue (if any):** 无（B1b Epic 启动时再开 follow-up issue，可参考 #159 uYgw P1+ alignment 模板）。

## S2.13 全局变量裸指针

**Master plan reference:**
> **S2.13 — 全局变量裸指针（记录，不迁移）** — **差异**：`uYsf/uYus/uYgw/uYriv/uYlake/timeNow` 是全局变量，非 `Model_Data` 成员 —— **P1-P7 影响**：strict OpenMP 下 CVODE 仍单线程调 `f()`，全局指针在 RHS 入口赋值后在 parallel region 内只读，无竞争。**不是 P1-P7 前置条件** —— **S2 阶段原则**：仅**记录**风险和当前使用模式（哪些函数读、哪些写、赋值时机），不做迁移 —— **迁移时机**：推迟到 P8+ / LibSHUD / Reentrant RHS 阶段 (master plan §5 S2.13)

**File:line(s):**
- `uYsf` / `uYus` / `uYgw` — `SHUD/src/ModelData/Macros.hpp` (global decl) + `SHUD/src/Model/MD_rhs_core.cpp` `rhs_update()` (write in RHS entry, read in flux/apply)
- `uYriv` / `uYlake` — same pattern
- `timeNow` — global time variable, written in `f()` entry, read by all flux functions

**Risk / observation:** 6 个全局裸指针是 RHS 入口 (`f()` / `rhs_core()`) 对 `N_Vector y` 做 `N_VGetArrayPointer` 后的便捷别名。在 strict OpenMP (P1-P7) 下 CVODE 单线程调 `f()`，parallel region 在内部 NumEle/NumRiv loop 内启动，全局指针在 parallel region 内**只读**，无竞争。但在 Reentrant RHS (P8+) 下 (如 Jacobian 并行差分估计需要并发调多次 `f()`)，全局可变指针会冲突。

| 变量 | 读 by | 写 by | 赋值时机 |
|---|---|---|---|
| `uYsf` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iSF_offset` |
| `uYus` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iUS_offset` |
| `uYgw` | `rhs_flux()` / `rhs_apply()` | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iGW_offset` |
| `uYriv` | `rhs_flux()` / `rhs_apply()` (river loop) | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iRIV_offset` |
| `uYlake` | `rhs_flux()` / `rhs_apply()` (lake loop) | `f()` / `rhs_core()` entry | RHS entry，`N_VGetArrayPointer(y) + iLAKE_offset` |
| `timeNow` | 所有 flux 函数 | `f()` / `rhs_core()` entry | RHS entry，赋 `t` 参数 |

**Decision (record-only / defer to B1b / etc.):** record-only — 6 个全局指针保持现状；不收编入 `Model_Data` 类。

**Reason for deferral:** 推迟到 **P8+ / LibSHUD / Reentrant RHS** 阶段。迁移涉及 `Macros.hpp` 宏重构 + 大量函数签名变更 (所有 flux 函数需加 `Model_Data&` 参数)，改动面巨大；当前 B1a/P1-P7 不需要这次重构 (CVODE 单线程调用 `f()` 即可)，提前做违反 KISS/YAGNI。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/ModelData/Macros.hpp` SHALL 不包含 6 个全局变量声明的实质改动；`grep -E 'extern.*(uYsf|uYus|uYgw|uYriv|uYlake|timeNow)' SHUD/src/ModelData/Macros.hpp` SHALL 保持 B0-tag 状态。

**Linked future issue (if any):** 无（P8 Reentrant RHS 启动时再开 Epic 跟踪）。

## S2.15 AccTemperature 除零

**Master plan reference:**
> **S2.15 — `AccTemperature.getACC()` 除零** — **差异**：`que.size()==0` 时除零 → NaN —— **S2 阶段原则**：仅**记录**此 bug，不在 S2 修复——修复会改变输出，违反 B1a == B0 的约束 —— **B1b 阶段修复**（S6a/S6b.1）：加 guard `que.empty() ? 0.0 : ACC/que.size()` (master plan §5 S2.15 + §6 S6b.1)

**File:line(s):** `SHUD/src/ModelData/AccTemperature.hpp:60-62` (`getACC()` method)

**Risk / observation:** 当 `que.size() == 0` (初始化早期，cryosphere 启用且模拟前 1440 min 未填满 24h average queue) `return ACC / que.size()` 触发整数除零 → 返回 NaN/Inf；NaN 通过 frozen-soil 温度依赖路径传播到 `qElePrep` / `qEleInfil`，可能污染 1440 min 内的整体输出。

**Decision (record-only / defer to B1b / etc.):** defer to B1b/S6b.1 修复。

**Reason for deferral:** 加 guard 会改变输出（修前 NaN vs 修后 0.0），违反 B1a == B0 的 bitwise reproducibility 约束。修复属 "输出改变类 bug fix" (master plan §5 表，与 S2.15/S2.17 同类)，必须在独立 B1b 阶段做，避免与 OpenMP 并行加速 (B1a 主线) 改动混合，难以审计哪个改动引起输出变化。

**Verification:** `git diff B0-tag HEAD -- SHUD/src/ModelData/AccTemperature.hpp` SHALL 不包含 L60-62 范围内的实质改动；`getACC()` 方法体保持 B0-tag 状态（仍含潜在除零）。

**Linked future issue (if any):** 无（B1b Epic 启动时开 issue 跟踪 S6b.1）。

## S2.16 OpenMP N_Vector 当前使用

**Master plan reference:**
> **S2.16 — 当前已使用 OpenMP N_Vector** — **差异**：`shud.cpp` L58-59 已用 `N_VNew_OpenMP` —— **原则**：**已由 S1d.3-S1d.5 宏解耦解决**。N_Vector 后端由 `SHUD_USE_OPENMP_NVECTOR` 独立控制，S1 阶段默认 OFF → `N_VNew_Serial`；P8-NVector 才允许打开。S2 阶段无需额外处理此项 (master plan §5 S2.16 + §4.21 + §4.13)

**File:line(s):** `SHUD/src/Model/shud.cpp:58-59` (`N_Vector y = N_VNew_OpenMP(NEQ, NumThreads, sunctx)` / `N_VNew_Serial(NEQ, sunctx)` 切换)

**Risk / observation:** B0 阶段 `shud.cpp:58-59` 默认硬编码 `N_VNew_OpenMP`，与 SUNDIALS-CVODE 的 OpenMP backend 绑定，引发跨平台 (serial-only build) 编译错误 + 数值再现性影响 (D13 OpenMP NVector backend baseline gap)。

**Decision (record-only / defer to B1b / etc.):** **零 action** — 已由 S1d.3-S1d.5 三步宏解耦解决：
- **S1d.3** 引入 `SHUD_USE_OPENMP_NVECTOR` 编译宏，默认 OFF → `N_VNew_Serial`
- **S1d.4** 统一所有 `N_VGetArrayPointer` 调用避免后端 specialization
- **S1d.5** 用 `N_VDestroy` generic 替代 `N_VDestroy_OpenMP`

S2 阶段无需额外处理；记录"已解决"状态供审计追溯。

**Reason for deferral:** 不适用（已 resolved）。打开 OpenMP N_Vector 后端推迟到 P8-NVector 阶段，需配套 D13 baseline regression 验收。

**Verification:**
- `grep -nE 'SHUD_USE_OPENMP_NVECTOR' SHUD/src/Model/shud.cpp` SHALL ≥ 1 (S1d.3 宏存在)
- `grep -nE 'SHUD_USE_OPENMP_NVECTOR \?= 0' SHUD/Makefile` SHALL = 1 (默认 OFF)
- `git diff B0-tag HEAD -- SHUD/src/Model/shud.cpp | grep -cE '^\+.*N_VNew_OpenMP'` SHALL ≤ N_VNew_Serial branch (S1d.3 宏 if-else 双分支保留)

**Linked future issue (if any):** D13 (in `openspec/changes/b1a-finalization/design.md` D13 entry — OpenMP NVector backend reduction-order baseline gap)；P8-NVector Epic 启动时开 follow-up。

## Section completeness verification

PR-7a 落地后:
- 本 PR：4 sections (S2.12 / S2.13 / S2.15 / S2.16) 齐全，每 section 7 字段完整
- PR-7b：追加 1 section (S2.17)；总 5 sections

`git diff B0-tag <PR-7a HEAD> -- SHUD/src/` SHALL 不包含本 PR 4 项 record-only 涉及的 file:line 范围内的实质改动 — 本 PR 0 SHUD 源码改动。

## Master plan 一致性

每个 section 的 **Master plan reference** 字段 verbatim 引用 master plan §5 S2.x 段落开头一句话，确保 record 内容与上游 spec 一致；任何 master plan §5 update 都需要本文档对应 section 同步 amend (作为后续 docs sync PR)。
