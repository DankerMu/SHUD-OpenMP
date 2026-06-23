# S6b.3 候选审计 (Candidate Audit) — S2 follow-up Bug 名录

> **Status (as of 2026-06-22)**: **完结历史文档**. S6b.3 audit closed at the date below; all downstream D9 fast-path gating has resolved — see §"Closing note" 末尾 update for current state. 仅留作 B1b epoch 历史证据。

## 背景与定义

本文档记录 S6b.3 阶段对"S2 阶段已识别但延后 (deferred) 至 B1b 处理"类候选 (candidate) bug 的审计 (audit) 工作。审计目的是将 S2 阶段保留的 record-only 子项汇总归集，逐一评估其在 B1b 阶段的处置 (disposition) 方案，并对零影响 (zero-impact) 类候选作出最终结论。本工作完成 spec.md §S6b.3 Requirement L35–L49 及 design.md R5 所规定的审计任务。

| Field | Value |
|---|---|
| Audit date | 2026-06-21 |
| Auditor | Phase-1 implementer agent (issue #187) |
| Scope | All "S2 阶段记录但推迟到 B1b" bug候选 — completes spec.md §S6b.3 Requirement L35–L49 + design.md R5 |
| Outer branch at audit | `feat/issue-187-b1b-s6b-3` (base `baseline/B1b@e044cbe`) |
| SHUD HEAD at audit | `bd7714a` on `openmp-baseline` |
| Candidate count | **1** (issue #159 only) |
| Empty-fallback line required | NO (count > 0 — spec L47–L49 Scenario applies to 0-candidate case) |
| Final disposition status | **CLOSED** — D9 fast-path chain resolved (B1b lock + B1-tag fast-path #2 trigger via PR-19 #210) |

## 审计方法

依据 spec.md L43–L46 Scenario "S6b.3 候选 issue 数量与 S2 record 表对齐" 的规定，候选名录 (roster) 为以下三类来源的并集 (union)：

1. `baseline/B1b` 上但 `baseline/B1a` 上不存在、且 commit subject 提及 `S6b-followup` 或 `S2 follow-up` 的提交；
2. `DankerMu/SHUD-OpenMP` 中带 `S6b-followup` 标签的 GitHub issue；
3. `docs/review-loop-log.jsonl` 中来自 PR-1 至 PR-12、且明确关联 S2 / S6b 语义的 `residual_deferred` 条目。

### 1. Commit message grep (外层仓库)

```text
$ git log --grep="S6b-followup" --oneline baseline/B1b ^baseline/B1a
(empty)

$ git log --grep="S2.*follow-up" --oneline baseline/B1b ^baseline/B1a
(empty)
```

结论：B1b 提交链中没有任何 commit 携带上述标签 token。spec.md 中虽规定了该命名约定，但历史提交未实际遵循。

### 2. GitHub issue 标签扫描

```text
$ gh issue list --label "S6b-followup" --repo DankerMu/SHUD-OpenMP --state all --json number,title,state
[]

$ gh label list --repo DankerMu/SHUD-OpenMP | grep -i "follow\|S6b\|S2"
s2-strict   S2 strict OpenMP 阶段 (P1-P7)   #5319e7
```

结论：`S6b-followup` 标签在 spec.md 中被引用，但实际未在 GitHub 仓库中创建。仓库中唯一与 S2 相关的标签为 `s2-strict`，其用途为 P1-P7 strict 模式跟踪，与本次审计关注的"延后 bug"语义正交。

### 3. review-loop-log 中 `residual_deferred` 条目人工挖掘

对 `docs/review-loop-log.jsonl` 中 PR-1 至 PR-12 的所有 `residual_deferred > 0` 条目做人工扫描，仅列出语义匹配 S2 或 S6b 的条目：

| PR | issue | residual_deferred note excerpt | Maps to a candidate? |
|---|---|---|---|
| PR-2 | #145 | "Pack 4 cross-review CLEAN x3 (1 WARN uYgw asymmetry PLAUSIBLE per verifier, deferred to **#159 follow-up**)" + "residual_deferred = D13 backend gap + **uYgw P1+ alignment**" | YES — this IS issue #159 |
| PR-12 | #156 | "**#159 (S2.6 P1+ pre-req) retained out of B1a scope**" | YES — same issue #159 |
| PR-1 | #144 | "cand-02/03/06 wontfix defer PR-12" + "server heihe/heihe_x4 deferred to PR-12 capstone" | NO — orchestrator-fixture candidates + server-run sequencing; PR-12 capstone closed both; no S2 bug-fix semantic |
| PR-3 | #146 | "S2.7 already satisfied by S1a rhs_update PURE CARRY-OVER" + "S2.8 implementation deferred to PR-9 #153 paired with S3a fun_Seg_* dead-+= rem" | NO — implementation-sequencing within S2 chain, closed by PR-9; not a B1b-deferred bug |
| PR-10 | #182 | "S5d.4 review-fix M1/M2/M3/m4 follow-up + residual deferred list" | NO — S5d.4 review residuals were dispositioned within PR-10 + S5d capstone PR-11 |
| PR-12 (S6b.1) | #184 | none — clean; `residual_deferred=0` is implied by absence | NO |

结论：仅 issue **#159** 属于真正意义上的"由 B1a 延后至 B1b 结构性修复窗口的 S2 类 bug"。其余 `residual_deferred` 条目要么属于 orchestrator-scope，要么属于 S2 链内部的实施排序 (implementation-sequencing) 议题，要么属于服务器侧待运行 (server-pending) 事项，且均已在所属 Stage 链内闭合。

### 4. spec.md 与 design.md 交叉引用

`openspec/changes/b1b-baseline-completion/specs/s6b-bugfix-application/spec.md` L37 与 `design.md` L227 均明确将 "#159 (S2.6 follow-up)" 列为唯一已知候选，并承认名录的其余部分需通过审计 (即本文档) 发现。两份文档均预留 0 至 3 个候选的容量。

综合以上四步，审计收敛至唯一 1 个候选，即 issue **#159**。

## 候选清单

### Candidate S6b.3.1 — issue #159 (S2.6 follow-up: `f_update_omp` `uYgw` `iBC == 0` asymmetry)

| Field | Value |
|---|---|
| Source PR/issue | `DankerMu/SHUD-OpenMP#159`, opened from PR-2 #145 Pack 1 Finding 2 / Phase 4.5 verifier verdict PLAUSIBLE |
| Original target file | `SHUD/src/ModelData/MD_f_omp.cpp` (L124, `iBC == 0` branch, `uYgw[i] = max(0.0, Y[iGW]);`) |
| Symptom described | Dormant TU `MD_f_omp.cpp::f_update_omp` had a 3-aligned-1-asymmetric form: `uYsf` / `uYus` / `uYriv` matched serial `f_update` direct-alias (`Y[iSF/iUS/iRIV]`) but `uYgw` retained the older `max(0.0, Y[iGW])` clamp. spec carve-out at PR-2 spec.md L109 + design.md L290 marked the asymmetry as "dormant-path historical quirk", deferred to P1+ OMP re-activation. |
| Default-build impact | ZERO. `MD_f_omp.cpp` was filtered out of the default build (Makefile L366-370 of the PR-2 era). |
| Evaluation conclusion | **NOT A BUG (auto-resolved by S2 capstone PR-8 #152)** |

#### 证据链

1. **PR-8 #152 (S2 capstone) 已物理删除 `MD_f_omp.cpp`**：

   ```text
   $ cd SHUD && git log openmp-baseline --diff-filter=D --name-only --oneline | grep -B1 -A0 "MD_f_omp" | head
   22777e5 S2 capstone: delete MD_f_omp.cpp + retire LEGACY_RHS + SHUD_LEGACY_OMP_RHS (PR-8 #152)
   src/ModelData/MD_f_omp.cpp
   ```

   该不对称所对应的目标 TU 已不存在于 `openmp-baseline` 分支的源码树中，因此无任何代码位点 (code site) 需要做对齐。

2. **`MD_f_omp.cpp` 在当前 SHUD 源码树中已确认缺失**：

   ```text
   $ find /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD -name "MD_f_omp*"
   (empty)
   $ ls /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_f*.cpp
   /.../SHUD/src/ModelData/MD_f.cpp
   /.../SHUD/src/ModelData/MD_f_uncouple.cpp
   ```

3. **`iBC == 0` 分支中现存两处 `uYgw[i] = ...` 位点均已采用 serial 直接别名形式**，即 #159 当时倡导的目标形式：

   - `SHUD/src/ModelData/MD_update.cpp:63-86` `Model_Data::f_update`：

     ```cpp
     if(Ele[i].iBC == 0){ // NO BC
     //   uYgw[i] = max(0.0, Y[iGW]);
         uYgw[i] = Y[iGW];
         Ele[i].QBC = 0.;
     }
     ```

   - `SHUD/src/Model/MD_rhs_core.cpp:55-86` `Model_Data::rhs_update`：

     ```cpp
     if(Ele[i].iBC == 0){ // NO BC
     //   uYgw[i] = max(0.0, Y[iGW]);
         uYgw[i] = Y[iGW];
         Ele[i].QBC = 0.;
     }
     ```

   注释化的 `max(0.0, Y[iGW])` 被保留为历史注释，便于后续读者追溯演化脉络；当前实际生效的代码即 #159 主张的直接别名形式。

#### 范围外的 live `max(0.0, Y[i])` 位点 (forward defense)

PR #203 review MN1 (`af5eb35f22058f59e`) 指出 `SHUD/src/ModelData/MD_update.cpp:22` 在 `Model_Data::f_updatei` case 3 (`iBC == 0` 分支) 内仍存在一处 LIVE 站点，使用 `uYgw[i] = max(0.0, Y[i]);`。该回调 `f_gw` 通过 `SHUD/src/Model/shud.cpp:336,389` 注册至 `CVode(mem3, ...)`，仅当 SHUD 以 `-DSHUD_uncouple` 编译且运行时携带 CLI `-g` 参数 (即 uncoupled GW-only 模式) 时方可到达。

此站点**不在 #159 的范围内**——#159 指向的是已退役的 `_omp` 变体 `MD_f_omp.cpp::f_update_omp`，而非 uncoupled 模式下的 `f_updatei`。该不对称模式在此处幸存，是因为 uncoupled 模式不在 B1b 阶段的 coupled-RHS 主代码路径上 (B1b 90 天 baseline 与 7 个基准算例全部以 coupled 模式运行，从不使用 `-g`)。后续若有工作触及 uncoupled 模式，应同步评估其与 `f_update` 及 `rhs_update` 的对齐情况；此处记录以备下一次审计捕获。

#### 处置 (依 spec.md L37 子句 d)

spec L37 规定"若评估结论为'非 bug' 或'延后到 P1+ 处理'，仍 SHALL 写入 changelog 解释"。本审计将 #159 记录为 **NOT A BUG (auto-resolved by PR-8 #152)**：不对称形式已通过结构层 (即目标文件删除) 消除，而幸存的 serial 站点亦已采用所期望的直接别名形式。S6b.3.1 无需任何代码改动。处置内容写入以下位置：

- `SHUD/B1b_CHANGELOG.md` 中 S6b.3.1 章节 (本 PR)；
- GitHub issue #159 评论，附带本 PR 的 CHANGELOG SHA 链接；
- `docs/diff_reports/B1a_vs_B1b_diff_s6b_3_1.md`，作为"zero-impact / no-code-change"报告 (本 PR)。

Issue #159 将在 PR 合入时关闭，遵循 PR-12 收尾约定 ("Closes #N" 加在 PR base 非 `main` 时手动执行 `gh issue close`)。

## 名录汇总

| Seq | Source | Title | Conclusion | Action |
|---|---|---|---|---|
| S6b.3.1 | issue #159 | f_update_omp uYgw iBC==0 asymmetry | NOT A BUG — auto-resolved by PR-8 #152 (`MD_f_omp.cpp` deleted; serial path already direct-alias) | CHANGELOG explanation + diff-report stub + #159 comment + close on PR merge |

候选总数：**1**。需要代码改动的候选数：**0**。可归因于 S6b.3 的 B1a → B1b 数值输出差异：**0**。

## Closing note

依据 design.md D9 fast-path 触发条件 #3 ("S6b.3 全部候选评估结论为 zero-impact (含 0 候选 / 非 bug / 延后到 B1b 之外)")，本审计向 D9 触发链贡献了一个"非 bug 自动解决 (non-bug auto-resolved)"判决。结合 S6b.1 的 zero-impact 结论 (经 PR-12 #184/#202 验证) 以及 S6b.2 的状态 (#186，待 PI review #185 裁定)，D9 fast-path 资格仅取决于 S6b.2 的结论。

### 2026-06-22 update — D9 fast-path chain RESOLVED

- **#185 PI review CLOSED** — PR-19 #210 capstone delivered S2.17 lake formula PI E2 sign-off (`docs/s217_lake_formula_audit.md` §E final verdict) + B1-tag annotated tag created aliasing main HEAD at fast-path trigger #2.
- **#186 RESOLVED** — see PR-19 #210 / s217 audit §E disposition.
- **D9 fast-path triggers #1/#2/#3 全部 resolved**: B1b-tag (PR-16 #207) + B1-tag (PR-19 #210) + this S6b.3 audit (PR merge time) 全部 successful。
- **S6b.3 itself**: candidate #159 verdict NOT-A-BUG auto-resolved + #159 closed on PR-12 #156 merge time per closing convention.
- **本 audit doc**: historical record. 本 doc 之后任何 S2 follow-up / S6b 类 bug 不再走 this roster — 直接走 P2+ issue 与 spec。
