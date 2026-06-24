## ADDED Requirements

### Requirement: docs/p1d/ 归档目录建立

P1d epic capstone SHALL 建立 `docs/p1d/` 目录归档 P1d epic 期间产出的 source-of-truth 文档（仿 `docs/p1c/` 组织模式）。

#### Scenario: docs/p1d/ ≥11 份必备文档

- **WHEN** P1d epic capstone PR-K 合并后 `ls docs/p1d/`
- **THEN** SHALL 含至少：
  - `p1d_summary.md`: 10 section capstone summary
  - `p1d_perf_baseline.md`: 8-cell server + Mac 4-cell + nst 数据（perf wall benchmark）
  - `p1d_numa_root_cause.md`: NUMA writer noise diagnosis + first-touch design rationale
  - `p1d_first_touch_design.md`: 3 pragma 区 first-touch loop 实现细节（含 PR-C §"字段集 grep 输出" 章 + PR-K §"3 pragma 实现" 章；PR-B 的 Mac OMP env 已独立到 `p1d_numa_env_runbook.md`）
  - `p1d_numa_env_runbook.md`: server sbatch template + Mac OMP env runbook（per tasks 2.3，独立文件避免与 first_touch_design 双写）
  - `p1d_kahan_revert.md`: PR-G Kahan revert + reverse-compat 双视角
  - `p1d_pr_f_intermediate_run.md`: PR-F intermediate 8-cell + NUMA env + first-touch only raw data (Kahan 仍在)
  - `p1d_pr_h_final_run.md`: PR-H 8-cell + 三 SHALL gate verdict raw data
  - `p1d_mac_reference.md`: P1-update-omp-tag Mac N=1 reference + P1d-binary Mac per-case 比对结果（含 PR-J Mac SHALL closure 全部 verification 数据，与 PR-I reference 同 doc 不再单独 split）
  - `p1d_tag_and_lock.md`: P1d-tag annotated procedure + baseline/P1d lock
- **AND** 同目录 `p1d_report.md` 学术报告综合（仿 `docs/p1c/p1c_report.md`）

#### Scenario: docs/p1d/ 跨 doc 引用一致性

- **WHEN** `grep -rln 'docs/p1d_[a-z]' docs/ openspec/specs/ openspec/glossary.md`
- **THEN** SHALL 返回 0 hits（命名约定: 所有 P1d 期文档均以 `docs/p1d/p1d_*` 子目录形式存在，无 `docs/p1d_xxx` 散落根路径）
- **AND** `docs/stage-pipeline-log.jsonl` 内若含 `docs/p1d_<old>` 字符串属于 immutable historical record（仿 P1c jsonl 实际记录格式：jsonl 一般记录 `"change":"p1d-numa-governance"` 等结构化字段，正常不应含散落路径字面）；如出现仅作历史归档不修订

### Requirement: openspec PROMOTE 2 spec

P1d epic capstone PR-M SHALL PROMOTE 2 spec 进 `openspec/specs/`：

#### Scenario: p1d-numa-governance spec PROMOTE

- **WHEN** PR-M 合并后 `ls openspec/specs/p1d-numa-governance/`
- **THEN** SHALL 含 `spec.md`，byte-identical 至 `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md`
- **AND** `diff -q` 报告无差异

#### Scenario: p1d-capstone spec PROMOTE

- **WHEN** PR-M 合并后 `ls openspec/specs/p1d-capstone/`
- **THEN** SHALL 含 `spec.md`，byte-identical 至 `openspec/changes/p1d-numa-governance/specs/p1d-capstone/spec.md`

#### Scenario: openspec/changes/p1d-numa-governance archive

- **WHEN** PR-M 阶段 archive local (gitignored)
- **THEN** `mv openspec/changes/p1d-numa-governance openspec/changes/archive/2026-06-XX-p1d-numa-governance/` (XX = actual date)
- **AND** archive 目录 gitignored 不入 git

### Requirement: glossary 4 新术语

P1d epic capstone PR-M SHALL 在 `openspec/glossary.md` §"P1d NUMA governance baseline 集合"（新章节）追加 4 新术语。其中 `steady-state first-touch (P1d)` 与既存 `first-touch / NUMA`（glossary L147-L148，S5d 引入）明确区分：既存 term 描述 allocation-time first-touch；P1d 新增 term 描述 RHS 评估期每步 warm-up first-touch。

#### Scenario: 4 新术语 必备

- **WHEN** `grep -nE '^\*\*P1d-tag\*\*|^\*\*baseline/P1d\*\*|^\*\*steady-state first-touch \(P1d\)\*\*|^\*\*P1d NUMA env' openspec/glossary.md`
- **THEN** SHALL 4 命中
- **AND** 每术语含 definition + `_Avoid_` 句子
- **AND** `steady-state first-touch (P1d)` term 内 SHALL cross-ref 既存 `first-touch / NUMA` term（glossary L147-L148），明确二者互补关系（allocation-time vs steady-state）

#### Scenario: glossary 命名空间不冲突 + 既存 first-touch term 增补

- **WHEN** glossary 已存在术语 `**first-touch / NUMA**`（glossary L147，S5d 引入）
- **THEN** P1d epic capstone PR-M SHALL 在该现有 term 末尾追加补充段：`"P1d 在 SHUD/src/ModelData/MD_update.cpp 三 pragma 前置 steady-state first-touch loop（区别于 Model_Data.cpp::malloc_EleRiv 内 allocation-time first-touch），详见 steady-state first-touch (P1d) term。"`
- **AND** 不删除原 first-touch / NUMA term 定义，保留 narrative continuity

#### Scenario: P1d carve-out term status 更新

- **WHEN** glossary 已存在术语 `**P1d carve-out (writer noise governance)**`（glossary L216-L218，P1c epic forward debt entry，PR base 对应 P1c PR-M repo PR #256）
- **THEN** P1d epic capstone PR-M SHALL 更新该 term 末尾 status: 由 `**non-blocking P1c** per master plan §3 fallback option 2` → 追加新行 `**Status (P1d epic close)**: CLOSED via P1d epic <date>，参见 docs/p1d/p1d_summary.md §"P1c carve-out closure"`
- **AND** 不删除原 term key，narrative 注明源于 P1c epic forward debt（per glossary L217）

### Requirement: stage-pipeline-log.jsonl 双追加

P1d epic capstone PR-M SHALL 在 `docs/stage-pipeline-log.jsonl` 追加 2 entries：

#### Scenario: pipeline summary entry

- **WHEN** PR-M 阶段
- **THEN** SHALL 追加 1 line JSON：`{"change":"p1d-numa-governance","date":"<date>","rounds":<n>,"gate_net_catch":<n>,"verdict":"closure|partial",...}` 含 P1d epic 全 PR list + verdict + 严格 hard gate verification

#### Scenario: Epic close-out entry

- **WHEN** PR-M 阶段
- **THEN** SHALL 追加 1 line JSON：`{"change":"p1d-numa-governance","epic":"...","tag":"P1d-tag","tag_object":"...","tag_deref":"...","SHUD_pin":"...","baseline_branch":"baseline/P1d",...}`

### Requirement: Epic close-out

P1d epic capstone PR-M 合并后 SHALL `gh issue close <P1d epic issue> --reason completed`（其 sub-issues 通过 PR Closes 关键字自动关闭，per CLAUDE.md "PR base 非 default 时 close-keywords 失效" 处理）。

#### Scenario: Epic + sub-issues 全 closed

- **WHEN** PR-M post-merge
- **THEN** `gh issue list --repo DankerMu/SHUD-OpenMP --state open --search 'P1d'` SHALL 返回空 `[]`
- **AND** P1d epic issue 状态 = CLOSED + sub-issues 全部状态 = CLOSED

### Requirement: status_matrix 更新

P1d epic capstone PR-K SHALL 在 `docs/status_matrix.md` 更新 P1d row：

#### Scenario: P1d row PENDING → verdict

- **WHEN** PR-K 阶段
- **THEN** P1d row 由 "PENDING (2026-06-23 新增 carve-out 阶段)" 更新为 实际 verdict（CLOSURE / PARTIAL / FAIL）
- **AND** 8-column matrix（6 case + verdict + remarks）填实测 SHA 数据 + 三 SHALL gate verdict

#### Scenario: P2a row 准备

- **WHEN** P1d closure (verdict = CLOSURE)
- **THEN** P2a row "PENDING (P1d 闭环后启动, per §7.3 修订 2026-06-23)" 状态保持 PENDING 但 prerequisites 已满足，可由 P2a epic 启动文档触发

### Requirement: main fast-forward 至 baseline/P1d

P1d epic capstone PR-M 合并后 SHALL fast-forward main 分支至 baseline/P1d HEAD（仿 P1c 模式）。

#### Scenario: main fast-forward

- **WHEN** PR-M 合并 + post-merge action
- **THEN** `git push origin baseline/P1d:main` SHALL succeed (fast-forward, no force)
- **AND** `git rev-parse origin/main` ≡ `git rev-parse origin/baseline/P1d`

### Requirement: D11 immutability final verification

P1d epic capstone PR-M post-merge action SHALL 最终验证 D11 6 tag chain SHA 全部不变。

#### Scenario: D11 6 tag chain verify

- **WHEN** P1d epic capstone 全完成时
- **THEN** 6 tag SHA `git rev-parse <tag>` 均与 P1d epic 启动时刻 (pre-P1d) 一致 (B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag) + P1d-tag 新增
- **AND** `git tag --verify P1d-tag` 报告 annotated tag immutable

### Requirement: subagent-workflow phase 0-8 全程执行

P1d epic 13 PR 全程 SHALL 遵循 subagent-workflow Phase 0-8 流程（仿 P1c epic 模式 — P1c 13 PR 实际选择记录在 `docs/stage-pipeline-log.jsonl` P1c entry；P1d 继承同一选择，相同 epic 内不混用其它 workflow）。允许范围：实施 / 审核 / 验证 subagent 可由 subagent-workflow **或** cc-cx-workflow 二者之一驱动；具体选择由 capstone PR-M `docs/stage-pipeline-log.jsonl` Epic close-out entry 显式记录。

#### Scenario: Phase 4 risk-adaptive cross-review + Phase 4.5 verifier

- **WHEN** 任一 P1d sub-issue 实施 PR
- **THEN** SHALL 经 reviewer subagent cross-review + verifier subagent CONFIRMED/PLAUSIBLE/REFUTED 三态裁决
- **AND** Phase 8 evidence + Chinese work summary + CI + merge

#### Scenario: workflow 选择记录入 jsonl

- **WHEN** PR-M Epic close-out entry 写入 `docs/stage-pipeline-log.jsonl`
- **THEN** SHALL 含 `"workflow":"subagent-workflow"` 或 `"workflow":"cc-cx-workflow"` 字段明示 P1d epic 实际驱动方式（仿 P1c jsonl entry 字段格式）

#### Scenario: subagent-workflow 不嵌套

- **WHEN** subagent 被 spawn 执行 implementer / reviewer / verifier 任务
- **THEN** subagent SHALL 不调 stage-change-pipeline / 不 spawn 嵌套 subagent / 不 invoke 本流水线

### Requirement: E' containment closure capstone capture (M10 修订)

P1d epic capstone PR-M SHALL 显式 capture E' containment closure 决策路径所伴生的 capstone artifacts：master plan v1.5 / M10 sync revision + ADR-0002 solver-path 4-way 评估 + p1e-strict-omp-rhs openspec local drafts + 4-mode spec capstone rewrite。该 Requirement 仅适用于 E' closure framing（PR-H FAIL 后用户决策 path），原 closure path（3 SHALL gate 全 PASS）不触发。

#### Scenario: master plan v1.5 / M10 sync revision capstone

- **WHEN** PR-M 合并时
- **THEN** `SHUD_openMP_master_plan.md` SHALL 已包含 M10 sync revision 段 (+180 / -7 lines additive, per `docs/p1d/p1d_summary.md` §1 reference)
- **AND** §6 P1d.4 (E′ 8 actions) + §6 P1d.5 (D11 6-chain) + §8.1 (4-mode block) SHALL 全部 in place
- **AND** §6 P1e (F path next epic preamble) SHALL 已建立 (forward handoff to next epic)

#### Scenario: ADR-0002 solver-path 4-way 评估 forward handoff

- **WHEN** PR-M 合并时
- **THEN** `docs/adr/0002-solver-path.md` SHALL 已建立 (Phase 2(e) parallel agent owns merge before PR-M)
- **AND** ADR 含 4-way solver path 对比：Path 1 (Serial N_Vector + StrictOMP RHS) / Path 2 (NVECTOR_REPRO_OMP) / Path 3 (SPGMR + block-Jacobi precond) / Path 4 (KLU sparse direct)
- **AND** **Path 1 SELECTED** for P1e implementation
- **AND** ADR 含 2x2 build matrix causal experiment design (per `docs/p1d/p1d_summary.md` §9.3)

#### Scenario: p1e-strict-omp-rhs openspec change local drafts forward handoff

- **WHEN** PR-M 合并时
- **THEN** `openspec/changes/p1e-strict-omp-rhs/` SHALL 已 local-draft 建立 (gitignored, Phase 2(e) parallel agent owns final propose)
- **AND** drafts 含 proposal.md (F path narrative) + design.md (Path 1 SELECTED reference) + tasks.md (P1e PR breakdown)
- **AND** P1e epic 启动前置 = P1d-tag push 完成 + baseline/P1d lock + p1e-strict-omp-rhs change propose + ADR-0002 close out (per `docs/p1d/p1d_summary.md` §9.1)

#### Scenario: 4-mode spec capstone rewrite capture

- **WHEN** PR-M 合并时
- **THEN** PROMOTEd `openspec/specs/p1d-numa-governance/spec.md` SHALL 含顶部 "Mode taxonomy (M10 修订, E' closure)" 4-mode block (serial / strict-omp / det-omp / fast-omp)
- **AND** 3 SHALL gate Requirements SHALL 含 Mode binding sub-clause (per spec amendment Task 4)
- **AND** 新增 Scenario "P1d E' containment closure verdict" SHALL 在 spec 内出现 (per spec amendment Task 4)
- **AND** `docs/p1d/p1d_summary.md` §5 (E′ 8 actions) + §1 (Status: PARTIAL CLOSURE via E′ containment path) + §10 (P1c carve-out closure narrative) SHALL 全部 cross-ref 4-mode spec capstone
