## ADDED Requirements

### Requirement: S6b.1 AccTemperature 除零 guard

系统 SHALL 在 `SHUD/src/AccTemperature.hpp` L60–L62 区段对 `AccTemperature::getACC()` 加除零 guard：原表达式 `ACC / que.size()` SHALL 改为 `que.empty() ? 0.0 : ACC / que.size()`（master plan §4.12 / S2.15）。该 fix SHALL 独立一个 commit，commit message 包含 `S6b.1` 标识。影响范围 SHALL 在 `B1b_CHANGELOG.md` 标注："仅影响 cryosphere 启用且模拟前 1440 min 的 NaN 传播路径"。

#### Scenario: divide-zero guard 表达式正确
- **WHEN** 检查 S6b.1 commit 后 `AccTemperature.hpp` L60–L62
- **THEN** `getACC()` body 包含 `que.empty() ? 0.0 :` 条件

#### Scenario: bitwise-validation case 不变（cryosphere=1 非雪期窗 + qhh cryosphere=0）
- **WHEN** 在 S6b.1 commit 上跑 4 case（keliya / xinanjiang_upstream / qinyijiang，`CRYOSPHERE=1` 但 90 天截断窗口不命中 AccTemperature 1440 min 初始 NaN 路径；qhh，`CRYOSPHERE=0` AccTemperature 路径完全跳过）90 天截断
- **THEN** 8 个 golden 文件 SHA256 与 B1a-tag 完全一致（zero-impact fix）

#### Scenario: cryosphere case 前 1440 min NaN 消除
- **WHEN** 在 S6b.1 commit 上跑 heihe（`CRYOSPHERE=1`，命中初始 AccTemperature NaN 路径）前 1440 min
- **THEN** AccTemperature 输出无 NaN；变更原因记入 `B1a_vs_B1b_diff_s6b_1.md`（qhh 此 Scenario 不适用，已由上一条 cover；keliya cryosphere-on 1-day 辅助证据在 Mac 端 `.s6b-1-runs/` 并入 changelog）

---

### Requirement: S6b.2 lake formula 仅在 S2.17 审查为"需修正"时执行

`fun_Ele_sub()` lake 分支公式（`SHUD/src/MD_ElementFlux.cpp` L117，master plan §4.18 / S2.17）SHALL 仅在 S2.17 审查结论为"公式需修正"时执行修改；否则 SHALL 写入 `B1b_CHANGELOG.md` 为"S2.17 审查结论 = 公式正确，无修改"并跳过。审查结论 SHALL 由 SHUD 上游 PI 签字或 PI delegate 在 GitHub issue 上评论后生效。

#### Scenario: 审查结论已签字执行修改
- **WHEN** S2.17 审查 issue 上有 PI/PI delegate 评论 `S2.17: formula needs fix`
- **THEN** S6b.2 commit 修改 `MD_ElementFlux.cpp` L117 公式，commit message 包含 `S6b.2`，影响范围记入 `B1b_CHANGELOG.md`

#### Scenario: 审查结论已签字跳过修改
- **WHEN** S2.17 审查 issue 上有 PI/PI delegate 评论 `S2.17: formula correct, no change`
- **THEN** S6b.2 不修改代码，`B1b_CHANGELOG.md` 写入 `S6b.2: S2.17 reviewed, no change` + 引用评论 URL

---

### Requirement: S6b.3 其他 S2 记录的待修 bug 逐项独立评估

S6b.3 SHALL 涵盖 S2 阶段记录但推迟到 B1b 的所有 bug（已知候选 = GitHub issue #159 S2.6 follow-up + 任何在 PR-1..PR-11 标 `S6b-followup` 的 issue）。每项 SHALL：(a) 独立 commit；(b) commit message 包含 `S6b.3.<seq>` 标识；(c) 影响范围记入 `B1b_CHANGELOG.md`；(d) 若评估结论为"非 bug" 或"延后到 P1+ 处理"，仍 SHALL 写入 changelog 解释。

#### Scenario: #159 评估结论 zero-impact
- **WHEN** S6b.3 评估 issue #159 (S2.6 follow-up)
- **THEN** 评估结果（修改 / 非 bug / 延后）记入 `B1b_CHANGELOG.md` S6b.3.1 段；issue 上同步评论结论

#### Scenario: S6b.3 候选 issue 数量与 S2 record 表对齐
- **WHEN** 审计 S2 阶段 PR-1..PR-11 commit message 与 GitHub issue 标 `S6b-followup` 的清单
- **THEN** S6b.3 评估清单等于该清单（不漏项 + 不无中生有）

#### Scenario: S6b.3 候选清单为空时不留悬空
- **WHEN** audit 完成且 grep 命中 0 候选
- **THEN** `B1b_CHANGELOG.md` S6b.3 段含字面行 `S6b.3: 0 candidates after S2 follow-up audit (<audit-date>)`；不允许 changelog 空段

---

### Requirement: 每个 fix 独立 commit + 独立 diff report

S6b.1 / S6b.2 / S6b.3.<seq> 每项 SHALL：(a) 单 commit（不与其他 fix 混 commit）；(b) 产出 `docs/diff_reports/B1a_vs_B1b_diff_<fix_id>.md` diff report 描述哪些输出变了（固定目录 `docs/diff_reports/`）；(c) 若不影响任何 benchmark 输出（如 cryosphere 未启用的 case），SHALL 在 diff report 标 "zero-impact fix"；(d) zero-impact fix SHALL 单独标，不与 non-zero-impact fix 混报。

#### Scenario: S6b.1 diff report 完整
- **WHEN** 检查 `docs/diff_reports/B1a_vs_B1b_diff_s6b_1.md`
- **THEN** 报告包含：影响 case 列表 / 不影响 case 列表 / 变化量描述 / 物理解释 / 4 case 是否标 zero-impact

#### Scenario: S6b.2 跳过时 diff report 仍存在
- **WHEN** S6b.2 跳过（S2.17 审查为"无修改"）
- **THEN** `docs/diff_reports/B1a_vs_B1b_diff_s6b_2.md` 仍生成，内容 = "S2.17 reviewed as correct; no code change; no diff"

---

### Requirement: B1b_CHANGELOG.md 单源汇总

`B1b_CHANGELOG.md` SHALL 是 S6b 所有 fix 的单源 changelog，禁止多份 changelog。每个 fix 一行表：fix id / commit SHA / 影响范围 / zero-impact 标记 / diff report 链接。

#### Scenario: changelog 表覆盖 S6b 全项
- **WHEN** 检查 S6c 锁定前 `B1b_CHANGELOG.md` 内容
- **THEN** 表行数 = S6b.1 + S6b.2 + 全部 S6b.3.<seq> 项数；每行有 commit SHA + diff report 链接

#### Scenario: 无重复 changelog
- **WHEN** 在 repo 内 grep `B1.*CHANGELOG`
- **THEN** 仅 `B1b_CHANGELOG.md` 存在；无 `B1b_CHANGELOG_v2.md` 等并存
