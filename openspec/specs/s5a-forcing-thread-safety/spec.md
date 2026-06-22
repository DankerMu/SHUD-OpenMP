## Purpose

规约 S5a forcing-pipeline 线程安全 audit 与文档落地（audit-only，不触 SHUD/src 数值代码）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/b1b-baseline-completion/specs/<capability>/spec.md PROMOTE 而来（#190 S6c-12c capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。

## Requirements

### Requirement: movePointer 必须在 RHS 并行区外串行调用

`TimeSeriesData::movePointer()` 是 forcing 数据的 single-thread 状态 mutate 入口；该方法 SHALL 仅在主循环 RHS 并行区**外**串行调用，不得在任何 `#pragma omp parallel`/`#pragma omp for` 上下文中触发。系统 SHALL 在审计阶段定位所有现存调用点，记录每个调用点的文件与行号到 `B1b_CHANGELOG.md` 的 S5a 段，证明调用位置满足串行约束。

#### Scenario: 现存 movePointer 调用点全部在并行区外
- **WHEN** 审计 `SHUD/src/TimeSeriesData.cpp` 与所有调用方（grep 结果）的 `movePointer` 调用上下文
- **THEN** 每个调用点均位于 `#pragma omp` 之外的串行代码段，审计记录入 `B1b_CHANGELOG.md`

#### Scenario: 接口注释明示串行调用契约
- **WHEN** 阅读 `SHUD/src/TimeSeriesData.hpp` 中 `movePointer()` 的声明
- **THEN** 声明上方注释明确写 "single-thread mutate; MUST be called outside any RHS parallel region"

---

### Requirement: getX 在 RHS 并行区内必须是 thread-safe read-only

`TimeSeriesData::getX(t, col)` 在 `movePointer()` 之后 SHALL 保持 thread-safe read-only 语义：(a) 不修改任何 `TimeSeriesData` 实例状态；(b) 不写入任何共享缓冲；(c) zero-order hold 取值规则不变。系统 SHALL 在 `SHUD/src/TimeSeriesData.cpp` L102–L105 周边代码上提供 audit 评论，并在 `SHUD/src/TimeSeriesData.hpp` 接口上标注 thread-safe read-only 契约。

#### Scenario: getX 与 B0 bitwise 等价
- **WHEN** 在 B1b 候选代码上对 4 case 的任意 forcing 时间点 `t` 与列 `col` 调用 `getX(t, col)`
- **THEN** 返回值与 B0-tag 对应调用点的浮点表示完全一致（bitwise）

#### Scenario: 接口注释明示 thread-safe read-only 契约
- **WHEN** 阅读 `SHUD/src/TimeSeriesData.hpp` 中 `getX(double t, int col)` 的声明
- **THEN** 声明上方注释明确写 "thread-safe read-only after movePointer; zero-order hold; no shared write"

---

### Requirement: S5a 不修改任何 I/O 逻辑

S5a SHALL 仅执行审计、注释与契约标注，不得修改 `read_csv()` 缓冲逻辑、不得修改文件 stream 生命周期、不得引入 forcing cache、不得改 refill 时机。Opt-IO 阶段（master plan §5）属独立 change 范畴。

#### Scenario: S5a 完成时 I/O 路径与 B1a 一致
- **WHEN** 检查 S5a 完成后 `SHUD/src/TimeSeriesData.cpp` 与 B1a 在 `read_csv()` 入口及 file stream 生命周期上的 diff
- **THEN** I/O 逻辑 diff 为空（仅有注释行新增），代码运行路径未变更

---

### Requirement: S5a 完成后完整 run 与 B1a bitwise identical

S5a 收尾时，对 7 case 中 4 case 直接 endpoint（keliya / xinanjiang_upstream / qinyijiang / qhh）+ 服务器侧 2 case (heihe / heihe_x4) Slurm 验证，完整 run 输出（rivqdown.dat / lakeoutputs 等）SHA256 SHALL 与 B1a-tag golden 完全一致。kashigeer 维持 deferred-upstream N/A。

#### Scenario: 4 case + 2 server case 全 bitwise 通过
- **WHEN** S5a 完成 commit 上跑 6 case bitwise vs B1a-tag (NUM_OPENMP=1, 90 天截断)
- **THEN** 4 Mac case rivqdown.dat + qhh 3 lake outputs 全 PASS；server heihe + heihe_x4 SHA256 全 PASS
