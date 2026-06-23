## Purpose

规约 P1 capstone 收尾契约：`P1-update-omp-tag` annotated tag 创建 + push origin、`baseline/P1` 分支创建 + lock (`lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`)、`docs/p1_summary.md` 模板对齐 `docs/b1b_summary.md`、`docs/status_matrix.md` P1 行 PENDING → PASS（6 case + kashigeer N/A）、`docs/build_manifest.md` 加 "P1-update-omp-tag 应用状态" 节、4 capability spec PROMOTE → `openspec/specs/<capability>/spec.md`、change archive 到 `openspec/changes/archive/<date>-p1-update-omp/`、`openspec/glossary.md` 加 P1 术语条目、`docs/review-loop-log.jsonl` + `docs/stage-pipeline-log.jsonl` 双追加、Epic + sub-issue 关闭。本 capability 是 P1 epic 的 Phase D（PR-L #224 tag-only + PR-M #225 docs + PR-N #226 PROMOTE）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/p1-update-omp/specs/<capability>/spec.md PROMOTE 而来（#226 P1 capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。
- D11 immutable：`P1-update-omp-tag` 一次锁死禁止 force-update（与 B1a-tag force-update 历史**不同**，与 B1b-tag 一致）；后续 retroactive 更新走 forward-compat `P1c-tag stacking` / `P2-* stacking`（master plan C8）。
- `docs/p1_summary.md` 模板锚定 `docs/b1b_summary.md` 7-topic 结构（**at least 7 topics required, deeper structure allowed**）：完成定义 / 旧版错误复盘 / P1-update-omp-tag 处理 / 时间线 / P1 后续 hand-off (→ P2a) / capstone 验证结果 / 验证 P1-update-omp-tag 验证命令；实际写作 SHALL **不少于** 这 7 个 topic，但可按需切分子节、增补 SHUD pin trail / 3-pragma 详情 / Validation evidence anchors / §1.1.1 WARNING framing / B-chain history / forward debts 等 deeper sub-section。
- 4 capability spec PROMOTE 模板 = Purpose + Conventions + Requirements 三段；`## ADDED Requirements` → `## Requirements` 机械替换；保留原 Requirement / Scenario 结构。
- Spec wording 修补（如 F-K2-1 / F-M-2）**仅在 PROMOTE step 应用**到 promoted spec；archive 保留 change-side spec 原文作为历史证据。

## Requirements

### Requirement: P1-update-omp-tag annotated + push origin

P1 capstone PR squash-merge 到 `main` 后，SHALL 创建 annotated tag `P1-update-omp-tag` aliasing 该 merge commit，tag message 含：
- commit SHA
- SHUD submodule pin（openmp-baseline branch HEAD at P1 capstone）
- "P1 — state update parallel complete (MD_update.cpp 三 owner loop)" 简述
- 7-bullet P1 fix 列表（element loop / river loop / lake loop / P1.0 audit clear / RHS snapshot vs B1b PASS / full-run vs B1b PASS / CVODE stats identical）
- Mac 4 case canonical SHA + server 2 case rivqdown SHA 表
- M7 forcing trim 状态（6-case trim PASS + kashigeer N/A + Opt-IO 决策结果引用）
- scaling 实测概要（NUM_OPENMP=1/2/4/8 加速比 + A3a/A3b verdict 引用 `docs/p1_perf_baseline.md`）

tag 创建后 SHALL 立即 push 到 origin。

#### Scenario: P1-update-omp-tag annotated + 含必要字段

- **WHEN** `git show P1-update-omp-tag --no-patch --format=full`
- **THEN** tag 是 annotated（非 lightweight），含上述所有字段

#### Scenario: origin 远端 tag 同步

- **WHEN** `git ls-remote --tags origin | grep P1-update-omp-tag`
- **THEN** 返回 annotated tag object SHA + dereferenced commit SHA 两行

---

### Requirement: baseline/P1 分支创建 + lock

`baseline/P1` 分支 SHALL 从 `main` 的 P1 capstone merge commit 分出（与 P1-update-omp-tag aliasing 同一 commit），并启用 protection rule `lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false`（与 baseline/B1b 一致，D11 enforced）。

#### Scenario: baseline/P1 分支存在 + 锁定

- **WHEN** `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'`
- **THEN** `{lock_branch:true, enforce_admins:true, allow_force_pushes:false, allow_deletions:false}`

#### Scenario: baseline/P1 HEAD = P1-update-omp-tag commit

- **WHEN** `git rev-parse origin/baseline/P1` vs `git rev-list -n 1 P1-update-omp-tag`
- **THEN** 两 SHA 一致

---

### Requirement: docs/p1_summary.md 模板对齐 b1b_summary.md（at least 7 topics required, deeper structure allowed）

`docs/p1_summary.md` SHALL 仿 `docs/b1b_summary.md` 结构，**at least 7 topics required, deeper structure allowed**（per F-M-2 PROMOTE upgrade #226：原 "7-section schema 严格对齐" 模板与实际 capstone summary 12-section 深结构 (SHUD pin trail / 3-pragma 详情 / Validation evidence anchors / §1.1.1 WARNING framing / B-chain history / forward debts 等) 漂移；本 Requirement 允许 PR-M #225 实际写作 12-section 深结构作为合法实现）。最少必备 7 topic：
1. 完成定义
2. 旧版错误复盘（不适用 → 简短一行）
3. P1-update-omp-tag 处理
4. 时间线
5. P1 后续 hand-off (→ P2a)
6. capstone 验证结果
7. 验证 P1-update-omp-tag (git ls-remote 命令 + branch protection 验证命令)

实际写作 SHALL **不少于** 这 7 个 topic（每个 topic 至少 1 节）；**可** 按需切分子节、增补 SHUD pin trail / 3-pragma 详情 / Validation evidence anchors / §1.1.1 WARNING framing / B-chain history / forward debts 等 deeper sub-section（如 PR-M #225 `docs/p1_summary.md` 实际 12-section 写法）。topic 顺序 SHALL 与 b1b_summary 7-topic 顺序对齐（深结构子节 inline 在所属 topic 内）。

#### Scenario: 7 topic 必备 + 顺序对齐

- **WHEN** 比对 `docs/p1_summary.md` 与 `docs/b1b_summary.md` 章节结构
- **THEN** 7 必备 topic（完成定义 / 旧版错误复盘 / P1-update-omp-tag 处理 / 时间线 / 后续 hand-off / capstone 验证结果 / 验证 P1-update-omp-tag）全部出现；topic 出现顺序与 b1b_summary 一致；deeper sub-section 数量不计（允许超过 7 节）

#### Scenario: deeper structure 允许（不阻 PASS）

- **WHEN** `docs/p1_summary.md` 含超过 7 个 `##` 顶级节（如 PR-M #225 实际 12 节包含 SHUD pin trail / 3-pragma stack 详情 / Validation evidence 3-anchor / §1.1.1 WARNING / B-chain history 等）
- **THEN** 不阻 PASS；deeper sub-section 视为对应 7 必备 topic 的 inline 扩展（如 "SHUD pin trail" → "时间线" topic 内；"Validation evidence" → "capstone 验证结果" topic 内）

#### Scenario: 验证命令完整

- **WHEN** 检查 `docs/p1_summary.md` "验证 P1-update-omp-tag" topic（不限于第 7 节）
- **THEN** 含 `git ls-remote --tags origin | grep P1-update-omp-tag` + `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection` + `git show P1-update-omp-tag --no-patch` 三命令

---

### Requirement: status_matrix.md P1 行 PENDING → PASS

`docs/status_matrix.md` SHALL 更新：
- P1 行：6 case (keliya / xinanjiang_upstream / qinyijiang / qhh / heihe / heihe_x4) 列从 PENDING 改为 PASS；kashigeer 列保持 N/A deferred-upstream
- aggregate 列：从 PENDING → PASS
- "最近一次更新" 注释更新含 P1 capstone PR + P1-update-omp-tag commit SHA + Mac 4 case canonical SHA + server 2 case rivqdown SHA + 引用 `docs/p1_summary.md`
- P1 行证据节新增（仿 B1b 行证据节），含每 case PASS 引用

#### Scenario: P1 行 6 case PASS + kashigeer N/A

- **WHEN** 检查 `docs/status_matrix.md` P1 行
- **THEN** keliya / xinanjiang_upstream / qinyijiang / qhh / heihe / heihe_x4 全部 PASS；kashigeer N/A；aggregate PASS

#### Scenario: P1 行证据节存在

- **WHEN** grep "## P1 行证据" `docs/status_matrix.md`
- **THEN** 节存在，含 6 case 证据表（canonical SHA + 引用 PR）

---

### Requirement: build_manifest.md 加 P1-update-omp-tag 节

`docs/build_manifest.md` SHALL 新增 "P1-update-omp-tag 应用状态" 节，含：
- tag-applied: true
- tag-date
- tag-object-sha (annotated)
- tag-commit-sha (capstone merge SHA)
- tag-SHUD-pin (openmp-baseline branch HEAD)
- conditional-ship: no (B1b 是 conditional → unconditional 升级历史，P1 一开始就 unconditional)
- M7 trim status: 6-case trim PASS (kashigeer N/A) + Opt-IO decision

#### Scenario: P1-tag 节齐全

- **WHEN** grep "P1-update-omp-tag 应用状态" `docs/build_manifest.md`
- **THEN** 节存在，含上述 7 字段

---

### Requirement: 4 capability specs PROMOTE 到 openspec/specs/

P1 capstone 完成后，SHALL 把本 change 的 4 capability specs `openspec/changes/p1-update-omp/specs/<capability>/spec.md` PROMOTE 到 `openspec/specs/<capability>/spec.md`，转换格式 `## ADDED Requirements` → `## Purpose / ## Conventions / ## Requirements` 三段（与 B1b PROMOTE 模式一致）。capability 列表：

- `m7-forcing-trim`
- `profile-retest-m7`
- `p1-state-update-parallel`
- `p1-capstone`

#### Scenario: 4 specs 已 PROMOTE

- **WHEN** `ls openspec/specs/{m7-forcing-trim,profile-retest-m7,p1-state-update-parallel,p1-capstone}/spec.md`
- **THEN** 4 文件存在

#### Scenario: PROMOTE 格式转换正确

- **WHEN** `head -10 openspec/specs/m7-forcing-trim/spec.md`
- **THEN** 含 `## Purpose` / `## Conventions` / `## Requirements` 三段头

---

### Requirement: openspec change archive

`openspec/changes/p1-update-omp/` SHALL archive 到 `openspec/changes/archive/<date>-p1-update-omp/`，含原 proposal / design / specs / tasks 4 文件原样保留；archive 后原 `openspec/changes/p1-update-omp/` 删除。

#### Scenario: change archive 完整

- **WHEN** `ls openspec/changes/archive/<date>-p1-update-omp/`
- **THEN** 含 proposal.md / design.md / tasks.md + specs/ 子目录（4 capability spec.md）

---

### Requirement: review-loop-log + stage-pipeline-log 双追加

P1 capstone 完成后 SHALL 在两个 jsonl 各追加 1 行：

- `docs/review-loop-log.jsonl`：append 行含 `capstone_commit / pr / change / milestone / verdicts / rounds / verdict: clean` 等字段（仿 B1b PR-17 #190 capstone 行模式）
- `docs/stage-pipeline-log.jsonl`：append 行含 `change: "p1-update-omp" / rounds / gate_net_catch / p0/p1 / verdict` 等字段

两文件 append-only 不覆写历史。

#### Scenario: review-loop-log 新增 P1 capstone 行

- **WHEN** P1 capstone PR merge 后 `tail -1 docs/review-loop-log.jsonl`
- **THEN** 行 JSON 含 `change: "p1-update-omp"` / `capstone_commit` / `verdict: clean`

#### Scenario: stage-pipeline-log 新增 P1 行

- **WHEN** P1 capstone PR merge 后 `tail -1 docs/stage-pipeline-log.jsonl`
- **THEN** 行 JSON 含 `change: "p1-update-omp"` / `rounds` / `gate_net_catch`

---

### Requirement: Epic / sub-issue 关闭 + glossary 更新

P1 capstone 完成后 SHALL：
- Epic P1 issue 关闭（含 `Closes Epic` 链接说明）
- 所有 sub-issue 已自动关闭（PR base=main 触发 `Closes #N`）
- `openspec/glossary.md` 加术语条目：M7-forcing-trim / P1-update-omp-tag / baseline/P1 / forcing.trimmed / forcing_trimmed=1

#### Scenario: Epic + sub-issues 全 closed

- **WHEN** `gh issue list --label P1 --state open`
- **THEN** 返回 `[]`

#### Scenario: glossary 新增条目

- **WHEN** grep "P1-update-omp-tag\|M7-forcing-trim\|forcing.trimmed" `openspec/glossary.md`
- **THEN** 三术语条目均存在
