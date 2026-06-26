# p1e-capstone Specification

## Purpose
TBD - created by archiving change p1e-strict-omp-rhs. Update Purpose after archive.
## Requirements
### Requirement: docs/p1e/ 归档目录建立

P1e epic capstone SHALL 建立 `docs/p1e/` 目录归档 P1e epic 期间产出的 source-of-truth 文档（仿 `docs/p1d/` 组织模式）。

#### Scenario: docs/p1e/ ≥14 份必备文档

- **WHEN** P1e epic capstone PR-K 合并后 `ls docs/p1e/`
- **THEN** SHALL 含至少以下 14 份文档（12 unconditional + 2 conditional with placeholder fallback）：
  - `p1e_summary.md`: 10 section capstone summary (§"验证 P1e-tag" 章 placeholder 在 PR-K 创建, SHA 由 PR-M post-PR-L-merge amend 填实)
  - `p1e_perf_baseline.md`: server 24-cell + Mac 4-cell perf wall + nst 数据 + N=1 vs N=8 加速比 (per-case threshold)
  - `p1e_2x2_experiment.md`: 2×2 build matrix 192 cell 综合表 (Phase 1: PR-C/D mode A/B + Phase 2: PR-G post-merge mode C/D + verdict + decision branch routing)
  - `p1e_2x2_verdict.md`: Phase 1 + Phase 2 综合 verdict + D12 routing 决策（Phase 1 由 PR-E 起草 + Phase 2 由 PR-I 在 mode C/D 数据完成后 amend; 含 4 mode × 4 N × 3 reps × 4 case 全数据 + D12.1/.2/.3/.4 routing 实际触发分支）
  - `p1e_strict_omp_design.md`: `ExecPolicy::StrictOMP` 实现细节 + 单 parallel region rationale + phase-based for + complexity analysis
  - `p1e_thread_split.md`: `SHUD_RHS_THREADS` vs `OMP_NUM_THREADS` runbook + build flag `SHUD_ENABLE_OPENMP_RHS` 使用说明 (含 PR-G Makefile -fopenmp 自动 wire)
  - `p1e_rivqdown_cache_audit.md`: rivqdown.dat 输出缓存 audit 报告 (per PR-A §1.7 起草, PR-K 终稿)
  - `p1e_first_touch_removal.md`: PR-H 3 处 steady-state first-touch removal 记录 + before/after diff + allocation-time + load-time first-touch 保留 verify
  - `p1e_pr_c_2x2_mac.md`: PR-C Mac 96 cell raw evidence (Phase 1 + Phase 2 sections)
  - `p1e_pr_d_2x2_server.md`: PR-D server 96 cell raw evidence (Phase 1 + Phase 2 sections)
  - `p1e_pr_i_strict_omp_verification.md`: PR-I 3 SHALL gate + 加速比 per-case threshold 验收 raw evidence
  - `p1e_mac_reverse_compat.md`: PR-J Mac N=1 reverse-compat per-case 验证 + Mac advisory cross-N (SHOULD)
  - `p1e_tag_and_lock.md`: P1e-tag annotated procedure + baseline/P1e lock
  - `p1e_toolchain_investigation.md`: 仅在 mode A Phase 1 FAIL 触发 (conditional doc)，否则可以 placeholder note "not triggered"
  - `p1e_pr_n_block_jacobi.md`: 仅在 D12.3 PR-N 触发 (conditional doc, per spec p1e-strict-omp-rhs L368 + tasks §8.4)，否则可以 placeholder note "not triggered (D12.3 fallback path not exercised)"
- **AND** 同目录 `p1e_report.md` 学术报告综合（仿 `docs/p1d/p1d_report.md`）

#### Scenario: tag 验证章 amend by PR-M

- **WHEN** PR-L merged AND PR-M 启动前
- **THEN** PR-K 已写 `docs/p1e/p1e_summary.md` §"验证 P1e-tag" placeholder 章框架（含 "_capstone-time SHA: TBD by PR-M post-merge_"）
- **AND** PR-M SHALL 编辑该章填实 tag-object SHA (`git rev-parse P1e-tag`) + deref commit SHA (`git rev-parse P1e-tag^{}`)，与 §7.15 jsonl entry 同来源
- **AND** P1e-tag deref commit 是 pre-PROMOTE HEAD（仿 P1d 模式，避免 PR-K 写后 PR-L 再 amend 循环）

#### Scenario: docs/p1e/ 跨 doc 引用一致性

- **WHEN** `grep -rln 'docs/p1e_[a-z]' docs/ openspec/specs/ openspec/glossary.md`
- **THEN** SHALL 返回 0 hits（命名约定: 所有 P1e 期文档均以 `docs/p1e/p1e_*` 子目录形式存在，无 `docs/p1e_xxx` 散落根路径）
- **AND** `docs/stage-pipeline-log.jsonl` 内若含 `docs/p1e_<old>` 字符串属于 immutable historical record（正常不应含散落路径字面）

### Requirement: openspec PROMOTE 2 spec

P1e epic capstone PR-M SHALL PROMOTE 2 spec 进 `openspec/specs/`：

#### Scenario: p1e-strict-omp-rhs spec PROMOTE

- **WHEN** PR-M 合并后 `ls openspec/specs/p1e-strict-omp-rhs/`
- **THEN** SHALL 含 `spec.md`，byte-identical 至 `openspec/changes/p1e-strict-omp-rhs/specs/p1e-strict-omp-rhs/spec.md`
- **AND** `diff -q` 报告无差异

#### Scenario: p1e-capstone spec PROMOTE

- **WHEN** PR-M 合并后 `ls openspec/specs/p1e-capstone/`
- **THEN** SHALL 含 `spec.md`，byte-identical 至 `openspec/changes/p1e-strict-omp-rhs/specs/p1e-capstone/spec.md`

#### Scenario: openspec/changes/p1e-strict-omp-rhs archive

- **WHEN** PR-M 阶段 archive local (gitignored)
- **THEN** `mv openspec/changes/p1e-strict-omp-rhs openspec/changes/archive/2026-06-XX-p1e-strict-omp-rhs/` (XX = actual date)
- **AND** archive 目录 gitignored 不入 git

### Requirement: glossary 4 新术语

P1e epic capstone PR-M SHALL 在 `openspec/glossary.md` §"P1e strict-omp F-path baseline 集合"（新章节）追加 4 新术语：

- `P1e-tag`
- `baseline/P1e`
- `strict-omp mode` (区分 P1d era 4-mode 表中的 4 mode；本术语指 master plan §8.1 strict-omp 行的 production candidate mode = Serial NVec + StrictOMP RHS + 跨 N bitwise + nst Δ=0)
- `2×2 build matrix` (= 本 epic P1e.2 因果实验的 4 build × 4 N × 3 reps × 4 case = 192 cell 矩阵)

#### Scenario: 4 新术语 必备

- **WHEN** `grep -nE '^\*\*P1e-tag\*\*|^\*\*baseline/P1e\*\*|^\*\*strict-omp mode\*\*|^\*\*2×2 build matrix\*\*' openspec/glossary.md`
- **THEN** SHALL 4 命中
- **AND** 每术语含 definition + `_Avoid_` 句子

#### Scenario: glossary 命名空间不冲突 + 既存 P1d carve-out term status 更新

- **WHEN** glossary 已存在术语 `**P1d carve-out**`（P1d epic capstone PR-M 入册）
- **THEN** P1e epic capstone PR-M SHALL 更新该 term 末尾 status: 追加新行 `**Status (P1e epic close)**: CLOSED via P1e epic <date>，参见 docs/p1e/p1e_summary.md §"P1d carve-out closure"`
- **AND** 不删除原 term key，narrative 注明源于 P1d epic forward debt

#### Scenario: strict-omp mode 术语区分

- **WHEN** glossary `strict-omp mode` term 内 SHALL cross-ref master plan §8.1 4-mode 表
- **THEN** 明确说明本术语指 production candidate mode = (N_Vector: Serial / RHS: StrictOMP / Bitwise: 跨 N + nst Δ=0 + N=1 reverse-compat strict / Build: `make shud SHUD_ENABLE_OPENMP_RHS=1`)
- **AND** 与其它 3 mode (`serial` / `det-omp` / `fast-omp`) 明确区分

### Requirement: stage-pipeline-log.jsonl 双追加

P1e epic capstone PR-M SHALL 在 `docs/stage-pipeline-log.jsonl` 追加 2 entries：

#### Scenario: pipeline summary entry

- **WHEN** PR-M 阶段
- **THEN** SHALL 追加 1 line JSON：`{"change":"p1e-strict-omp-rhs","date":"<date>","rounds":<n>,"gate_net_catch":<n>,"verdict":"closure|partial",...}` 含 P1e epic 全 PR list + verdict + 严格 hard gate verification + 2×2 实验 routing 决策分支

#### Scenario: Epic close-out entry

- **WHEN** PR-M 阶段
- **THEN** SHALL 追加 1 line JSON：`{"change":"p1e-strict-omp-rhs","epic":"...","tag":"P1e-tag","tag_object":"...","tag_deref":"...","SHUD_pin":"...","baseline_branch":"baseline/P1e","workflow":"subagent-workflow",...}`

### Requirement: Epic close-out

P1e epic capstone PR-M 合并后 SHALL `gh issue close <P1e epic issue> --reason completed`（其 sub-issues 通过 PR Closes 关键字自动关闭，per CLAUDE.md "PR base 非 default 时 close-keywords 失效" 处理）。

#### Scenario: Epic + sub-issues 全 closed

- **WHEN** PR-M post-merge
- **THEN** `gh issue list --repo DankerMu/SHUD-OpenMP --state open --search 'P1e'` SHALL 返回空 `[]`
- **AND** P1e epic issue 状态 = CLOSED + sub-issues 全部状态 = CLOSED

### Requirement: status_matrix 更新

P1e epic capstone PR-K SHALL 在 `docs/status_matrix.md` 更新 P1e row + P2a row prerequisites：

#### Scenario: P1e row 新增 + verdict

- **WHEN** PR-K 阶段
- **THEN** P1e row 新增（master plan §6 P1e 起 P1e 是新 stage）+ 填实 verdict（CLOSURE / PARTIAL / FAIL）+ 8-column matrix（6 case + verdict + remarks）
- **AND** 3 SHALL gate verdict + 加速比实测填实

#### Scenario: P2a row prerequisites 由 P1d → P1e 更新

- **WHEN** PR-K 阶段
- **THEN** P2a row prerequisites 由 "P1d 闭环后启动" 更新为 "P1e 闭环后启动 (3 SHALL gate strict-omp PASS + 加速比 ≥ 1.5× + P1e-tag lock + ADR-0002 close)"

### Requirement: ADR-0002 (solver-path) close out

P1e epic capstone PR-K SHALL 更新 `docs/adr/0002-solver-path.md` Status 与 Implementation closure 节（PR-K 内执行，PR-M 仅 verify — 避免循环精度依赖 per tasks §6.7）。**Note**：KLU pattern-only spike 是 forthcoming **ADR-0003**（独立 ADR，不再 alias 为 ADR-0002 自身后续 spike）— 若 D12.4 触发，引用 ADR-0003 而非 ADR-0002 自身。

#### Scenario: ADR-0002 Status 更新

- **WHEN** PR-K 实施
- **THEN** `docs/adr/0002-solver-path.md` 头部 `- Status: Accepted` SHALL 更新为 `- Status: Implemented (P1e epic close, <date>)`
- **AND** 末尾 SHALL 追加 "Implementation closure" 节，引用：
  - `docs/p1e/p1e_summary.md` (capstone summary)
  - `docs/p1e/p1e_2x2_verdict.md` (2×2 实验结论)
  - 实际触发的 decision branch (D12.1 / D12.2 / D12.3 / D12.4)
  - P1e-tag SHA + baseline/P1e lock confirmation (post-PR-L 信息，PR-M post-PR-L-merge amend 此节填实 SHA)
- **AND** 若实际触发 D12.4 (Path 4 fallback), 引用 forthcoming **ADR-0003 (KLU spike)** 而非 ADR-0002 自身（避免 ADR self-reference）

#### Scenario: ADR-0002 与 P1e 实施一致性

- **WHEN** P1e epic 完成时
- **THEN** ADR-0002 Decision Matrix 中 Path 1 (Selected) 与 P1e 实施实际路径 SHALL 一致
- **AND** 若实际触发 D12.2 fallback (Path 2)，ADR-0002 SHALL 追加 "Note: Path 2 triggered via P1e D12.2 fallback" 段
- **AND** ADR-0002 L301 "Forthcoming ADRs - ADR-0002 (forthcoming): KLU pattern-only spike" 行 SHALL 修订为 "Forthcoming ADRs - ADR-0003 (KLU spike, forthcoming)"（消除 ADR self-reference）

### Requirement: main fast-forward 至 baseline/P1e

P1e epic capstone PR-M 合并后 SHALL fast-forward main 分支至 baseline/P1e HEAD（仿 P1c / P1d 模式）。

#### Scenario: main fast-forward

- **WHEN** PR-M 合并 + post-merge action
- **THEN** `git push origin baseline/P1e:main` SHALL succeed (fast-forward, no force)
- **AND** `git rev-parse origin/main` ≡ `git rev-parse origin/baseline/P1e`

### Requirement: D11 immutability final verification (7 tag chain)

P1e epic capstone PR-M post-merge action SHALL 最终验证 D11 7 tag chain SHA 全部不变。

#### Scenario: D11 7 tag chain verify

- **WHEN** P1e epic capstone 全完成时
- **THEN** 7 tag SHA `git rev-parse <tag>` 均与 P1e epic 启动时刻 (pre-P1e) 一致：
  - B1-tag, B1a-tag, B1b-tag, P1-update-omp-tag, P1c-tag, P1d-tag 6 historical 全 SHA 不变
  - P1e-tag 新增（annotated tag object SHA + deref commit SHA 入 jsonl）
- **AND** `git tag --verify P1e-tag` 报告 annotated tag immutable

### Requirement: subagent-workflow phase 0-8 全程执行

P1e epic 13 PR 全程 SHALL 遵循 subagent-workflow Phase 0-8 流程（仿 P1c / P1d epic 模式）。允许范围：实施 / 审核 / 验证 subagent 可由 subagent-workflow **或** cc-cx-workflow 二者之一驱动；具体选择由 capstone PR-M `docs/stage-pipeline-log.jsonl` Epic close-out entry 显式记录。

#### Scenario: Phase 4 risk-adaptive cross-review + Phase 4.5 verifier

- **WHEN** 任一 P1e sub-issue 实施 PR
- **THEN** SHALL 经 reviewer subagent cross-review + verifier subagent CONFIRMED/PLAUSIBLE/REFUTED 三态裁决
- **AND** Phase 8 evidence + Chinese work summary + CI + merge

#### Scenario: workflow 选择记录入 jsonl

- **WHEN** PR-M Epic close-out entry 写入 `docs/stage-pipeline-log.jsonl`
- **THEN** SHALL 含 `"workflow":"subagent-workflow"` 或 `"workflow":"cc-cx-workflow"` 字段明示 P1e epic 实际驱动方式（仿 P1c/P1d jsonl entry 字段格式）

#### Scenario: subagent-workflow 不嵌套

- **WHEN** subagent 被 spawn 执行 implementer / reviewer / verifier 任务
- **THEN** subagent SHALL 不调 stage-change-pipeline / 不 spawn 嵌套 subagent / 不 invoke 本流水线

