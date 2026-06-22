## Purpose

记录 B1b baseline 收尾验收契约（S5+S6b 全部完成 + bitwise vs B0/B1a-tag + Go/No-Go 7 项 + CONDITIONAL ship caveats）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/b1b-baseline-completion/specs/<capability>/spec.md PROMOTE 而来（#190 S6c-12c capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。

## Requirements

### Requirement: B1b 单线程多次运行 bitwise 自洽

B1b 候选 commit (`<new-commit>`) SHALL 在 NUM_OPENMP=1 配置下连续 3 次跑 6 case 90 天截断，每 case 三轮输出 SHA256 SHALL 三次完全一致；6 case = keliya / xinanjiang_upstream / qinyijiang / qhh (含 3 lake outputs) + 服务器侧 heihe + heihe_x4；kashigeer 维持 deferred-upstream N/A。

#### Scenario: 4 case Mac 三次自洽 PASS
- **WHEN** 在 B1b 候选 commit 上 `tools/archive_b0_output.sh <case> 3` for 每个 Mac 4 case
- **THEN** 每 case 三轮 SHA256 完全一致

#### Scenario: 2 case 服务器三次自洽 PASS
- **WHEN** 在 cn08 或任一 CPU 分区双 socket Xeon idle node (cn05-06,09,14-19,23-24) 上 Slurm 提交 heihe + heihe_x4 各 3 次 90 天截断 NUM_OPENMP=1，sbatch 从 `/scratch` 下提交，`--output/--error` 路径在 `/scratch/frd_muziyao/SHUD-OpenMP/.b1b-server-runs/`
- **THEN** 每 case 三轮 SHA256 完全一致；运行节点 + jobid 记录在 `docs/b1b_summary.md`

---

### Requirement: B1b_CHANGELOG.md 完整且每项可归因

`B1b_CHANGELOG.md` SHALL 在 B1b-tag 锁定时刻完整：(a) S6b 全部 fix 一行不漏；(b) 每行有 commit SHA + 影响范围 + diff report 链接；(c) zero-impact fix 显式标注；(d) `B1a_vs_B1b_RHS_report` 内每个差异均能映射到 changelog 某一行（无 unaccounted diff）。

#### Scenario: changelog 与 RHS report 完全映射
- **WHEN** 比对 `B1b_CHANGELOG.md` 行集合与 `B1a_vs_B1b_RHS_report.md` 中所有差异条目
- **THEN** 差异条目集合 ⊆ changelog 行集合；不存在 changelog 未列的差异

#### Scenario: zero-impact 快速路径触发时合并 tag
- **WHEN** 所有 S6b fix 均标 zero-impact，且 **S6b.2 = "S2.17 审查为'无修改'"** 或 **S6b.2 = "S2.17 fix 实施后 zero-impact 验证 PASS"**
- **THEN** `B1a_vs_B1b_RHS_report` 内容 = "全 benchmark 无差异"；允许 B1a/B1b 合并为单 `B1-tag`（master plan L1497 快速路径，design D9）；不强制产出 `B1a_vs_B1b_full_run_report`；触发时 `B1b_CHANGELOG.md` 必含 S2.17 审查结论引用 (issue URL 或 commit SHA)

---

### Requirement: 水量平衡不恶化对比 B0

`B0_vs_B1b_water_balance_report` SHALL 在 B1b-tag 锁定前完成，至少覆盖 4 case Mac + 2 case 服务器。每 case 报告含：(a) 输入降水累计；(b) 输出径流累计；(c) 储水变化；(d) 闭合误差 ≤ B0 闭合误差（不恶化）。

#### Scenario: 6 case 水量平衡不恶化
- **WHEN** 计算 B1b 6 case 90 天截断的闭合误差
- **THEN** 每 case 闭合误差 ≤ B0 同 case 闭合误差（相对 0.1% 容差）

---

### Requirement: B1b-tag annotated 且禁止 force-update

B1b-tag SHALL 是 annotated tag，message 包含 (a) commit SHA；(b) SHUD submodule pin；(c) `B1a → B1b structural & bugfix complete`；(d) zero-impact 快速路径触发与否标识。tag 创建后 SHALL push 到 `origin`，`baseline/B1b` 分支 SHALL 锁（`lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`），不允许后续 force-update（与 B1a 实操历史不同，B1b 一次锁死）。

#### Scenario: B1b-tag annotated + SHA + SHUD pin
- **WHEN** `git show B1b-tag` 查看 tag 内容
- **THEN** 输出含 tag message + commit SHA + SHUD submodule SHA

#### Scenario: origin 远端 tag 同步
- **WHEN** `git ls-remote --tags origin | grep B1b-tag`
- **THEN** 返回 annotated tag 对象 SHA + dereferenced commit SHA 两行

#### Scenario: baseline/B1b 分支锁
- **WHEN** `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/B1b/protection`
- **THEN** `lock_branch=true`，`enforce_admins=true`，`allow_force_pushes=false`，`allow_deletions=false`

---

### Requirement: Go/No-Go → P1 七项 checklist

B1b-tag 锁定 commit 上 SHALL 验证 master plan §S6c L1511–L1525 (A1b 验收 + Go/No-Go 合并) 7 项 checklist 全 PASS：(1) B1b 已锁定 (L1513)；(2) B1b 单线程多次 bitwise (L1514)；(3) 所有 shared accumulation 已拆为 deterministic gather (PR-11，L1523)；(4) 编译选项固定且无 fast-math (L1524)；(5) `schedule(static)` 规则确定 (L1525)；(6) `B1b_CHANGELOG.md` 完整 (L1515)；(7) 水量守恒不恶化 (L1517)。L1519–L1525 §S6c "Go/No-Go" 明确含 (1)(2)(3)(4)(5)；(6)(7) 来自 L1511–L1517 §A1b acceptance，本 spec 把两段合并为统一 7 项验收门。任一不 PASS SHALL 阻断 P1 启动。

#### Scenario: 7 项 checklist 全 PASS
- **WHEN** 检查 docs/b1b_summary.md 的 Go/No-Go 节
- **THEN** 7 项每项有 evidence（grep / 文件路径 / SHA / 报告链接），全 PASS

#### Scenario: fast-math 编译 flag 已禁
- **WHEN** grep `SHUD/Makefile` 与 CI workflow `serial-baseline.yml` 的编译 flag
- **THEN** 无 `-ffast-math` / `-Ofast` / `-funsafe-math-optimizations`

---

### Requirement: 文档同步与 status_matrix 更新

B1b-tag 锁定 SHALL 同步：(a) 新建 `docs/b1b_summary.md`（仿 `docs/b1a_summary.md` 模板，含 S5a/b/c/d + S6b + capstone 完成时间线）；(b) `docs/status_matrix.md` B1b 行从 PENDING / IN-PROGRESS 更新为 PASS；(c) `docs/build_manifest.yaml` 加 `B1b-tag` 节；(d) openspec change `b1b-baseline-completion` archive 到 `openspec/changes/archive/<date>-b1b-baseline-completion/`；(e) capability specs PROMOTE 到 `openspec/specs/<capability>/spec.md`（与 B1a capstone PR-12 模式一致）。

#### Scenario: docs/b1b_summary.md 模板对齐
- **WHEN** 比对 `docs/b1b_summary.md` 与 `docs/b1a_summary.md` 章节结构
- **THEN** 章节顺序一致：完成定义 / 旧版错误复盘（不适用 → 简短一行）/ B1b-tag 处理 / 时间线 / S5+S6 后续 hand-off / capstone 验证结果 / 验证 B1b-tag

#### Scenario: status_matrix B1b 行 PASS
- **WHEN** 检查 `docs/status_matrix.md` B1b 行
- **THEN** 状态 = PASS + 引用 B1b-tag commit SHA + b1b_summary.md 链接

#### Scenario: openspec archive 完成
- **WHEN** 检查 `openspec/changes/archive/<date>-b1b-baseline-completion/` 存在
- **THEN** 含 proposal/design/specs/tasks 4 文件；同时 `openspec/specs/` 下有 6 个新 capability spec 文件

---

### Requirement: review-loop-log 与 stage-pipeline-log 双追加

B1b-tag 锁定后 SHALL 在两个 jsonl 各追加 1 行：(a) `docs/review-loop-log.jsonl` 追加 subagent-workflow 维度 capstone 行（capstone_commit / verdicts / rounds / verdict=clean）；(b) `docs/stage-pipeline-log.jsonl` 追加 stage-change-pipeline 维度行（change=b1b-baseline-completion / rounds / gate_net_catch / p0/p1 / verdict）。两文件 append-only 不覆写历史。

#### Scenario: review-loop-log 新增 B1b 行
- **WHEN** B1b-tag 锁定后 `tail -1 docs/review-loop-log.jsonl`
- **THEN** 行 JSON 含 `b1b_capstone_commit` / `verdicts.confirmed` / `verdicts.refuted` / `verdicts.plausible` / `verdict: clean`

#### Scenario: stage-pipeline-log 新增 B1b 行
- **WHEN** B1b-tag 锁定后 `tail -1 docs/stage-pipeline-log.jsonl`
- **THEN** 行 JSON 含 `change: "b1b-baseline-completion"` / `rounds` / `gate_net_catch` / `verdict`
