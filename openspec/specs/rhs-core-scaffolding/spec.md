# rhs-core-scaffolding Specification

## Purpose

S1a 阶段交付。把 SHUD `f_update` / `f_loop` / `f_applyDY` 的入口与签名收敛到统一 `Model_Data::rhs_core(Y, DY, t)` 入口点（三参版本），引入 `MD_rhs_core.cpp` / `MD_rhs_core.hpp` 骨架 + `rhs_update()` 纯搬运 + `f.cpp` 内 `#ifdef USE_RHS_CORE` 编译分支，bitwise neutral vs B0。混合模式仅 `rhs_update` 走新路径，`f_loop` / `f_applyDY` 仍 fallback 到 legacy，S1b / S1c 后续逐段抽取。

## Scope

**S1a 阶段签名约定**：本 capability 全程 `rhs_core()` 为**三参版本** `rhs_core(double* Y, double* DY, double t)`，**不**带 `ExecPolicy` 参数。`ExecPolicy` 枚举与四参重载 `rhs_core(Y, DY, t, ExecPolicy policy)` 由 `exec-policy-enum` capability 在 **S1d.1** 引入；S1a / S1b / S1c spec 不引入 `ExecPolicy`，避免阶段越界。

**头文件命名约定**：实现文件 `SHUD/src/Model/MD_rhs_core.cpp`；header `SHUD/src/Model/MD_rhs_core.hpp`（与 SHUD 既有 `Macros.hpp` / `Element.hpp` 命名风格一致），不使用 `.h` / `rhs_core.hpp` 等异名。

**Temporal scope note**：本 capability 中所有 `_OPENMP_ON` 相关 Scenarios 仅适用于 S1a PR 评审上下文；post-S1d.2 由 `openmp-macro-decoupling` capability 退役 `_OPENMP_ON`（拆分为 `SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS` 三正交宏），相关 Scenarios 被 superseded。
## Requirements
### Requirement: New `MD_rhs_core.cpp` skeleton with `rhs_update()` carry-over

SHUD submodule SHALL 在 `SHUD/src/Model/MD_rhs_core.cpp`（新建）+ 配套 header `SHUD/src/Model/MD_rhs_core.hpp` 中定义 `rhs_core()` 调度入口骨架与 `rhs_update()` 函数。`rhs_update()` SHALL 是 serial `Model_Data::f_update`（`SHUD/src/ModelData/MD_update.cpp::f_update`）的**纯搬运**：变量名、循环顺序、`uYsf/uYus/uYgw/uYriv/uYlake/yLakeStg/y2LakeArea` 全局/成员赋值、`Ele[i].iBC` 分支、`Riv[i].BC` 分支、`QeleSub/QeleSurf/qEleExfil/qEleInfil` 归零、`Qe2r_Surf/Qe2r_Sub` 归零、`DY[i] = 0.` 收尾、`SHUD_DUMP_RHS` hook 调用——全部 byte-for-byte 一致，MUST NOT 改任何逻辑、表达式或浮点运算顺序。

#### Scenario: Source carry-over diff is structural-only

- **WHEN** reviewer 对 `git diff` 比对 `SHUD/src/ModelData/MD_update.cpp::f_update` 与 `SHUD/src/Model/MD_rhs_core.cpp::rhs_update` 函数体
- **THEN** 差异 SHALL 仅限于函数签名 / 命名空间 / `Model_Data::` 限定符与新文件 include 头；函数体语句序列 MUST 与原 `f_update` 一一对应，无新增条件分支、无循环顺序调整、无表达式重写
- **AND** `SHUD_DUMP_RHS` hook 调用 SHALL 保留原 dump tag 字符串 `"f_update"` 不变（避免与 task 0.1a 重新归档 snapshot 元数据脱锚；tag 字符串与 `tools/rhs_snapshot/` header 中 record name 对齐，rename 会让 snapshot 元数据 lookup fail）

#### Scenario: `rhs_core()` dispatch skeleton compiles

- **WHEN** `SHUD/src/Model/MD_rhs_core.cpp` 与 `SHUD/src/Model/MD_rhs_core.hpp` 首次 commit
- **THEN** 文件 SHALL 包含 `rhs_core(double* Y, double* DY, double t)` 三参函数声明（无 `ExecPolicy` 参数；`ExecPolicy` 由 S1d.1 引入），函数体 SHALL 按 S1a 混合模式调用 `rhs_update()` + legacy `MD->f_loop()` + legacy `MD->f_applyDY()`；`SHUD_ENABLE_OPENMP_RHS` 未定义时整段仍能编译成功

#### Scenario: Header is included only by `f.cpp` and `MD_rhs_core.cpp`

- **WHEN** S1a 完成时
- **THEN** `SHUD/src/Model/MD_rhs_core.hpp` SHALL 仅被 `SHUD/src/Model/f.cpp` 与 `SHUD/src/Model/MD_rhs_core.cpp` include；其它 `MD_*.cpp` / `shud.cpp` / `main.cpp` SHALL 不引入新 header，避免编译面扩散

---

### Requirement: `f.cpp` introduces `#ifdef USE_RHS_CORE` branch with B0 fallback default

`SHUD/src/Model/f.cpp::f()` SHALL 引入 `#ifdef USE_RHS_CORE` 编译期分支：定义时 `f()` 调用 `rhs_core(Y, DY, t)`（S1a 三参签名）；未定义时（**默认**）`f()` 行为与 B0 完全等价——仍走原 `MD->f_update / f_loop / f_applyDY`（serial，定义于 `SHUD/src/ModelData/MD_update.cpp` 与 `SHUD/src/ModelData/MD_f.cpp`）或 `MD->f_update_omp / f_loop_omp / f_applyDY_omp`（`_OPENMP_ON`，定义于 `SHUD/src/ModelData/MD_f_omp.cpp`）路径。`USE_RHS_CORE` 默认 OFF 是 S1a–S1c 的强制约束；测试时由 CI / 开发者 explicit `make shud USE_RHS_CORE=1` 开启。

#### Scenario: Default build (USE_RHS_CORE undefined) is B0-tag bitwise

- **WHEN** `make shud` 不显式定义 `USE_RHS_CORE` 且 `_OPENMP_ON` 关闭
- **THEN** 编译产物在 keliya / xinanjiang_upstream / qinyijiang / qhh 上跑出的 `B0_output/*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全相等

#### Scenario: `USE_RHS_CORE` macro position is single-source

- **WHEN** S1a PR 落入 baseline/current
- **THEN** `f.cpp` 中 `#ifdef USE_RHS_CORE` 分支 SHALL 仅出现在 `f()` 函数体内；其它 5 个入口（`f_surf` / `f_unsat` / `f_gw` / `f_river` / `f_lake`）MUST NOT 引入该宏，保持 B0 路径不变

---

### Requirement: S1a mixed-mode dispatch — only `rhs_update` migrated, `flux`/`apply` fallback

S1a 阶段 `rhs_core(Y, DY, t)` 调度 SHALL 仅运行新的 `rhs_update()`；后续 flux 与 apply 步骤 MUST 仍调用 legacy `MD->f_loop(t)`（`SHUD/src/ModelData/MD_f.cpp::f_loop`）与 `MD->f_applyDY(DY, t)`（`SHUD/src/ModelData/MD_f.cpp::f_applyDY`）。这是 S1a 唯一允许的混合模式；S1b / S1c 完成前不得在 S1a 内一次性搬运 flux / apply。

#### Scenario: `rhs_core` body matches S1a mixed-mode spec

- **WHEN** S1a PR 内 `rhs_core()` 函数体被审查
- **THEN** 函数体 SHALL 按顺序为：`rhs_update(Y, DY, t)` → `MD->f_loop(t)` → `MD->f_applyDY(DY, t)`；不得调用 `rhs_flux` / `rhs_apply`（S1a 阶段二者尚不存在）；MUST NOT 引入 `ExecPolicy` 参数（S1d.1 才引入）

#### Scenario: USE_RHS_CORE=1 in S1a is bitwise to B0

- **WHEN** `make shud USE_RHS_CORE=1` 编译 + 在本地 4 个 CI case 跑 90 天截断 run
- **THEN** 输出 `B0_output/*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全一致；mixed mode（新 update + legacy loop/apply）不引入任何 bit 差

---

### Requirement: Global state read/write contract preserved

`rhs_update()` SHALL 读写**完全相同**的全局变量集合（`uYsf` / `uYus` / `uYgw` / `uYriv` / `uYlake` / `yLakeStg` / `y2LakeArea` / `timeNow`，参见 master plan §4.10）以及相同的 `Model_Data` 成员（`QeleSub` / `QeleSurf` / `QeleSubTot` / `QeleSurfTot` / `qEleExfil` / `qEleInfil` / `Qe2r_Surf` / `Qe2r_Sub` / `QrivSurf` / `QrivSub` / `QrivUp` / `QLakeSub` / `QLakeSurf` / `qLakeEvap` / `qLakePrcp` / `QLakeRivIn` / `QLakeRivOut` 等）作为 legacy `f_update()`。读写时机 MUST 与 legacy 一致；`rhs_core()` 入口 / 出口 MUST NOT 私自重置 / 缓存 / 拷贝任何全局指针。`timeNow` 赋值 SHALL 仍由 `f()` 在 `rhs_core()` 调用前完成（保持 `f()` 入口处单点赋值的语义），`rhs_update()` 内部 MUST NOT 重复写 `timeNow`。

#### Scenario: PR description enumerates global symbol delta

- **WHEN** S1a PR 描述
- **THEN** 描述 SHALL 列出 `rhs_update()` 引用的全局变量与 `Model_Data` 成员清单，并 explicit 标注 "与 legacy `f_update` 完全一致，无新增、无遗漏"

#### Scenario: Grep audit confirms no extra writes

- **WHEN** 对新 `MD_rhs_core.cpp` 执行 `grep -nE 'uYsf|uYus|uYgw|uYriv|uYlake|yLakeStg|y2LakeArea|timeNow' MD_rhs_core.cpp` 与对原 `f_update` 同样命令做行级对比
- **THEN** 两者写入位置（左值出现处）数量 SHALL 相同，读取位置（右值出现处）数量 SHALL 相同；任何一边出现新的写入位置即视为契约破坏，PR 不得 merge

#### Scenario: `timeNow` not double-written

- **WHEN** S1a 编译产物运行
- **THEN** `timeNow` SHALL 仅由 `f.cpp::f()` 在 `rhs_core()` 调用之前写一次（保留 `timeNow = t;` 在 `f()` 入口处单点赋值的语义）；`rhs_update()` 函数体内 MUST NOT 出现 `timeNow =` 赋值语句

---

### Requirement: Single-call DY snapshot bitwise to legacy `f_update()` snapshot

S1a 完成时，**单次** RHS 评估 DY snapshot——通过 `tools/rhs_snapshot/`（参见 S0 spec `rhs-snapshot-tooling`）在 `rhs_update()` 路径与 legacy `f_update()` 路径各跑一次后 dump——SHALL 经 `tools/compare_snapshot/` 比对返回 `BITWISE IDENTICAL`（exit code 0）。snapshot probe 触发时点遵循 `benchmarks/<case>/manifest` 中 `snapshot_probe.t_values`，统一为 `t_values = [86400, 2592000, 7776000]`（即 1d / 30d / 90d，单位 seconds，全部 ≤ 90 天截断窗口）。12 张 golden 由 `tasks.md` task 0.1a 在 pre-S1a 重新归档（替代 S0-7 时期生成的 4-year-run timestamp 不可达 golden）。

#### Scenario: Snapshot probe at 3 t_values byte-equal

- **WHEN** keliya / xinanjiang_upstream / qinyijiang / qhh 4 case 各在 `t_values = [86400, 2592000, 7776000]`（1d / 30d / 90d，单位 seconds）触发 `SHUD_DUMP_RHS` hook（USE_RHS_CORE=1 一次，USE_RHS_CORE 关闭一次）
- **THEN** `compare_snapshot <new>.bin <legacy>.bin` SHALL exit 0 + 输出 `BITWISE IDENTICAL`；任意一个 case 任意一个 t_value 失败即 S1a FAIL

#### Scenario: Snapshot golden also matches B0 golden

- **WHEN** USE_RHS_CORE=1 跑出的 snapshot 与 `benchmarks/<case>/B0_output/snapshot_t<value>.bin` 比对（其中 `<value>` ∈ `{86400, 2592000, 7776000}`，对应 task 0.1a 在 pre-S1a 重新归档的 12 张 golden，直接读 working tree 文件、不通过 `git show B0-tag:` 间接寻址）
- **THEN** `compare_snapshot` SHALL exit 0；这覆盖"新 update + legacy loop/apply"组合对 B0 DY 状态的 bitwise 保持
- **NOTE**：新 t-value goldens（`[86400, 2592000, 7776000]`）由 task 0.1a 在 #53 merge 时落入 `baseline/current`；不在 `B0-tag` tree 内（B0-tag 内仅含 S0-7 abs-minute goldens 作 historical reference）。S1 CI bitwise gate 直接对比 working tree 文件，不通过 `git show B0-tag:` 间接寻址

#### Scenario: First-call-after-CVODE-reinit semantics

- **WHEN** 任意 case（建议 keliya）通过 **deployment-layer override**（部署期 `cfg.para` 直接覆写 `Update_IC_STEP = 43200`（30 day × 1440 min/day；SHUD 解析的 token 是带下划线的 `Update_IC_STEP`，单位是 minute——见 `SHUD/src/classes/Model_Control.cpp::Control_Data::read` 内 `Update_IC_STEP` 解析分支 + `SHUD/src/classes/Model_Control.hpp::Control_Data::UpdateICStep` 成员声明；cfg.para 已 gitignored，不污染 SHUD submodule）触发 90 day 窗口内 mid-window IC backup（`SHUD/src/ModelData/MD_update.cpp::PrintInit`，不是 CVODE state re-injection——SHUD 无 `CVodeReInit` 路径），并以 `USE_RHS_CORE=1` 跑 90 天截断
- **THEN** re-init 之后**第一次** `rhs_core()` 调用产生的 DY SHALL 与 legacy `f_update + f_loop + f_applyDY` 路径**字节相等**；所有 3 个 t_value snapshot SHALL PASS（`compare_snapshot` exit 0）
- **AND** 该 Scenario 防回归 master plan §S1a 风险 R6（warm restart 后 `MD_rhs_core.cpp` 内部状态可能与 legacy 不同步）
- **AND** 部署期 keliya cfg.para override 由 `tasks.md` task 1.5.a 负责（pre-S1a 落地，与 task 0.1a golden 归档配套使用）

### Requirement: Full-run bitwise vs B0-tag across 4 local CI cases

S1a 完成时，`make shud USE_RHS_CORE=1` 编译产物在 keliya / xinanjiang_upstream / qinyijiang / qhh 4 case 上跑完整 90 天截断 run，所有 `B0_output/*.dat` SHA256 SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 完全相等。kashigeer N/A（A0 deferred-upstream，不在 S1 CI 必跑集合）；heihe / heihe_x4（server-only）在 S1a PR merge 后 24h 内由开发者在服务器手动跑一次归档（master plan §S1a Migration Plan）。

#### Scenario: 4-case CI gate passes

- **WHEN** S1a PR 触发 `.github/workflows/serial-baseline.yml` 的 4-case bitwise job
- **THEN** 每个 case 的 `sha256sum B0_output/*.dat` SHALL 与 `git show B0-tag:benchmarks/<case>/B0_output/` 对应文件完全相等；任何一个 case 失败 SHALL 让 CI 失败、PR 不得 merge

#### Scenario: Server-only cases archived within 24h post-merge

- **WHEN** S1a PR merge 进 baseline/current 后 24 小时内
- **THEN** 开发者 SHALL 在服务器 `/scratch/frd_muziyao/SHUD-OpenMP/` 通过 Slurm 跑 heihe + heihe_x4（USE_RHS_CORE=1，90 天截断），把 SHA256 归档至外层 PR 评论；若与 B0-tag 不一致 SHALL 触发回滚（外层 revert pointer bump）

#### Scenario: kashigeer is explicitly skipped with rationale

- **WHEN** S1a CI 配置 / status_matrix 引用 case 集合
- **THEN** kashigeer SHALL 标注为 `endpoint: deferred-upstream`，并附 A0 阶段 SKIP 理由链接；MUST NOT 在 S1 bitwise 集合内静默剔除

---

### Requirement: CVODE statistics invariance (15-key canonical set)

S1a `USE_RHS_CORE=1` 完整 run 收尾时 CVODE 统计 **canonical 15-key set** SHALL 与 B0（USE_RHS_CORE 未定义）对应 case 的值完全相等。15 个键完整集合定义详 `openspec/glossary.md` §CVODE canonical 15-key set（全部来自 SUNDIALS CVODE / CVSpils API 的 `PrintFinalStats` 输出，与 SHUD 实际归档对齐——`nFCall` 是 SHUD 内部计数器但未写入 `cvode_stats.txt`，由独立 capability 跟踪，详 design.md D10 F19 修订）。即使 `B0_output/*.dat` bitwise 通过，任一键漂移仍视为 FAIL（master plan §S1 风险 R5）。比对统一调用共享脚本 `tools/cvode_stats_diff/cvode_stats_diff.sh`（由 `tasks.md` task 0.2 提供），脚本 exit code 0 表示 canonical 15-key 全等。

#### Scenario: Stats 15-key equality per case via shared diff utility

- **WHEN** keliya / xinanjiang_upstream / qinyijiang / qhh 各跑 USE_RHS_CORE=1 与 USE_RHS_CORE 未定义两次，提取 `cvode_stats.txt`，对每个 case 执行 `tools/cvode_stats_diff/cvode_stats_diff.sh <new_stats.txt> <baseline_stats.txt>`（参数顺序为 `<new> <golden>`，与脚本 `Usage:` 一致；见 `tools/cvode_stats_diff/cvode_stats_diff.sh:4-5`）
- **THEN** 脚本 SHALL 对 canonical 15-key set（详 `openspec/glossary.md` §CVODE canonical 15-key set）逐一比较，全等返回 exit code 0；任意键不等 SHALL 让脚本 exit ≠ 0 并让 S1a FAIL，即使 `*.dat` SHA256 全等

### Requirement: `_omp` path source untouched in this capability

S1a capability MUST NOT 修改 `Model_Data::f_update_omp`、`Model_Data::f_loop_omp`、`Model_Data::f_applyDY_omp` 的任何源代码（三者全部定义于 `SHUD/src/ModelData/MD_f_omp.cpp`；参见 design.md D3）。`_OPENMP_ON` 宏分支与 `f.cpp::f()` 内部 `_OPENMP_ON` ifdef block 在 S1a 阶段保持完全等价于 B0；S2 才合并 `_omp` 语义差异。

#### Scenario: `_omp` function bodies unchanged

- **WHEN** 对 S1a PR 执行 `git diff baseline/current..HEAD -- SHUD/src/ModelData/MD_f_omp.cpp`，并 grep `f_update_omp|f_loop_omp|f_applyDY_omp`
- **THEN** 三个函数的函数体 SHALL 显示为零行新增、零行删除；任何 hunk 命中 `_omp` 函数体 SHALL 让 reviewer 否决

#### Scenario: `_OPENMP_ON` branch in `f.cpp` not regressed

- **WHEN** S1a 后查看 `f.cpp::f()` 内部 `_OPENMP_ON` ifdef block
- **THEN** `#ifdef _OPENMP_ON` 分支的 `NV_DATA_OMP` 与 `f_update_omp / f_loop_omp / f_applyDY_omp` 调用 SHALL 保持原样；`USE_RHS_CORE` 与 `_OPENMP_ON` MUST NOT 在 S1a 阶段产生新的组合分支（两者组合下的行为属 S1d / S2 范畴）
- **NOTE**：本 Scenario 仅适用于 S1a / S1b / S1c PR 上下文；post-S1d.2 由 `openmp-macro-decoupling` capability 退役 `_OPENMP_ON`（拆分为三正交宏），本 Scenario 被替代

#### Scenario: Macro decoupling deferred

- **WHEN** S1a PR 范围审查
- **THEN** `_OPENMP_ON` 拆分为 `SHUD_ENABLE_OPENMP_RHS` / `SHUD_USE_OPENMP_NVECTOR` / `SHUD_LEGACY_OMP_RHS` 三个正交宏 SHALL 不在 S1a 范围内；该改动属 `openmp-macro-decoupling` capability（S1d Step 2）

