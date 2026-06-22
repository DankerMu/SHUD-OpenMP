## Purpose

规约 S5c CVODE solver diagnostics（7 stats hook + SHUD_ENABLE_DIAGNOSTICS gate + 15-key CI diff）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/b1b-baseline-completion/specs/<capability>/spec.md PROMOTE 而来（#190 S6c-12c capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。

## Requirements

### Requirement: 接入 SUNDIALS CVODE stats 7 个 API

系统 SHALL 在 CVODE 求解完成后通过 SUNDIALS 公共 API 读取以下 7 项统计并落地到诊断 channel：`CVodeGetNumSteps` / `CVodeGetNumRhsEvals` / `CVodeGetNumErrTestFails` / `CVodeGetNumNonlinSolvIters` / `CVodeGetNumLinIters` / `CVodeGetLastStep` / `CVodeGetLastOrder`。落地格式 SHALL 与 PR-12 现有 15-key snapshot 模式兼容（每项一行 `<name>: <value>`，便于 SHA256 diff）。

#### Scenario: 7 项 CVODE stats 全部出现在 15-key snapshot 中
- **WHEN** 在 S5c 完成 commit 上跑 keliya 90 天截断并 dump 15-key CVODE snapshot
- **THEN** 输出 includes `nst` `nfe` `netf` `nni` `nli` `hlast` `qlast` 7 个 key 且值非空

#### Scenario: 7 项 stats 仅依赖 SUNDIALS 6.0.0 公共 API
- **WHEN** grep S5c 新增代码对 CVODE 内部符号的引用
- **THEN** 仅出现 `CVodeGetNum*` / `CVodeGetLast*` 公共 API；无 `cv_mem->` 类内部 struct 访问

---

### Requirement: 自定义 RHS 子阶段 timer 必须覆盖 7 个 bucket

系统 SHALL 实现 RHS 子阶段 wall-clock timer，按 7 个 bucket 分类：`update` / `ET` / `lateral` / `segment` / `river` / `gather` / `applyDY`。timer 实现 SHALL 满足：(a) 高精度时钟（`std::chrono::steady_clock` 或同等）；(b) 单 case 单 run 输出累计 ns 与百分比；(c) 命名与 master plan §S5c L1366 一致；(d) 不引入 RHS 路径浮点运算变化。

#### Scenario: 7 个 bucket 输出完整时间分布
- **WHEN** 在 S5c 完成 commit 上跑 heihe 90 天截断并启用诊断 timer
- **THEN** 输出 includes 7 个 bucket 的累计 ns 与百分比之和 ∈ [99.5%, 100.5%]

#### Scenario: timer 不影响 bitwise
- **WHEN** S5c 完成 commit 与 B1a-tag 上跑同 case 同 cfg 的 90 天 rivqdown.dat SHA256
- **THEN** 两 SHA256 完全一致（timer 仅做时间测量，无浮点路径变更）

---

### Requirement: forcing I/O 单独 timer 必须与 RHS 子阶段 timer 分离

`TimeSeriesData::read_csv()` 启动期 I/O 累计 wall-clock SHALL 单独 timer 上报，不混入 RHS 子阶段 7 bucket。该 timer 与 master plan M7 forcing trim 的判定指标 `t_forcing_io / t_total` 直接对应。

#### Scenario: forcing I/O timer 独立 channel
- **WHEN** 在 heihe_x4 90 天截断（未 trim forcing）上跑 S5c 完成 commit
- **THEN** 输出 includes `t_forcing_io` 单独 bucket，且其值 ≈ 780s（master plan L1567 M7 实测值，±10% 容差）

---

### Requirement: 诊断开关默认关闭，开关切换不改 bitwise

系统 SHALL 通过编译宏 `SHUD_ENABLE_DIAGNOSTICS`（或同等命名）控制诊断 timer 与 CVODE stats 收集。开关默认关闭；开启时不得改变 RHS 路径输出。

#### Scenario: 开关关闭时与 B1a-tag bitwise 一致
- **WHEN** 关闭 `SHUD_ENABLE_DIAGNOSTICS` 编译并跑 6 case 90 天截断
- **THEN** 6 case rivqdown.dat SHA256 全 PASS vs B1a-tag

#### Scenario: 开关开启时与 B1a-tag bitwise 一致
- **WHEN** 开启 `SHUD_ENABLE_DIAGNOSTICS` 编译并跑 6 case 90 天截断
- **THEN** 6 case rivqdown.dat SHA256 全 PASS vs B1a-tag（诊断仅测量，不修改路径）

---

### Requirement: nFCall 与 CVODE 15-key nfe 严格分离

`Model_Data` 的 `nFCall` 计数器（`Model_Data.hpp` L58）SHALL 作 §C8 RHS 调用归一指标独立追踪：(a) 自增点限制在 RHS kernel 入口；(b) **不进入** PR-12 CI snapshot 的 15-key invariance gate；(c) 与 CVODE `nfe` 之间允许差异，差异 ≠ CI fail，差异需在 `B1b_CHANGELOG.md` 解释（F19 round-2 design D10 决策）。nFCall 与 nfe 的差异本身**无上限阈值**（free-running counter）；CI 仅强制两件事 = 不入 15-key gate + 每 case 一行 changelog 解释。

#### Scenario: 15-key snapshot 不包含 nFCall
- **WHEN** 检查 S5c 完成后 CI workflow `serial-baseline.yml` 的 15-key SHA256 diff 步骤
- **THEN** diff 输入 key 集合不含 `nFCall` 标识；`nfe` 仅指 `CVodeGetNumRhsEvals`；CI script 中 15-key 列表通过常量/yaml 声明，nFCall 显式排除

#### Scenario: nFCall 独立 channel 上报
- **WHEN** 在 S5c 完成 commit 上跑任意 case
- **THEN** 输出有独立 `nFCall: <value>` 行，与 15-key snapshot 文件分开

#### Scenario: nFCall != nfe 时 changelog 强制解释（无数值阈值）
- **WHEN** 任一 case S5c 完成 run 上 `nFCall != nfe`
- **THEN** `B1b_CHANGELOG.md` S5c 段含一行 `case=<name> nFCall=<n1> nfe=<n2> reason=<rationale>`；差异**无**上限阈值（free-running，不阻 CI），但缺少 changelog 行视为 fail

---

### Requirement: S5c 完成后完整 run 与 B1a bitwise identical（开关任意位置）

S5c 收尾时，无论诊断开关开/关，6 case Mac + 服务器 endpoint SHA256 SHALL 与 B1a-tag golden 完全一致。

#### Scenario: 双开关位置 6 case bitwise 全通过
- **WHEN** 分别用 `SHUD_ENABLE_DIAGNOSTICS=ON` 与 `OFF` 编译 + 跑 6 case 90 天截断
- **THEN** 两组 SHA256 均 PASS vs B1a-tag golden
