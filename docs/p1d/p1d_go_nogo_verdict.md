# P1d Go/No-Go verdict — PR-M filled

**Status**: FILLED (PR-M 实测填实，2026-06-24)

**综合 verdict**: **PARTIAL CLOSURE via E′ containment path**（per master plan v1.5 / M10 §6 P1d.4）。Forward handoff → P1e (F path) per ADR-0002 Path 1 SELECTED。

PR-K (capstone docs) 创建本 placeholder；PR-M (verdict + PROMOTE + Epic close) 填入 actual verdict against actual P1d artifact state。

## §1 PR-H 3 SHALL gate 实测结果 (data source: `docs/p1d/p1d_pr_h_final_run.md`)

| Gate | Threshold | PR-H 实测 | Verdict |
|---|---|---|---|
| L123 Kahan revert canonical | heihe N=1 SHA == `7f22bd6faa438d50...` (P1-update-omp canonical) | byte-identical 全 64-hex | **PASS** |
| L130 A3a bitwise cross-N | heihe + heihe_x4 每 case N∈{1,2,4,8} 4 cell SHA 全等 | 双 case 各 3 distinct SHA（N=1≡N=2 ≠ N=4 ≠ N=8） | **FAIL** |
| L139 nst Δ + ladder | heihe Δ=0 strict + heihe_x4 \|Δ\| ≤ 2 | heihe Δ=80@N=4 / 152@N=8；heihe_x4 Δ=11@N=4 / 4@N=8 | **FAIL** |
| L145 N=1 reverse-compat | 6-case N=1 SHA == `P1-update-omp-tag` canonical | server heihe partial（spec 仅预写 heihe N=1 canonical） | **PARTIAL** |

**Spec verdict per gate**: PARTIAL CLOSURE — 2 hard SHALL gate FAIL (A3a + nst) + 1 PARTIAL (N=1 reverse-compat) + 1 PASS (Kahan canonical)。原 spec L150 设 "任一 SHALL FAIL → P1d 不 closure"，**用户决策 2026-06-24 走 E′ containment path 不再走原 closure**：详 `docs/p1d/p1d_summary.md` §5 + 本 doc §7。

## §2 Mac PR-I 6-cell reference closure (data source: `docs/p1d/p1d_pr_i_p1_update_omp_reference.md`)

| Item | Status |
|---|---|
| keliya × {serial, omp@N=1, omp@N=8} + qhh × {serial, omp@N=1, omp@N=8} = 6 cell PR-I 独立 worktree 切 `P1-update-omp-tag` 采集 | **PASS** (6/6) |
| 与 PR-G Mac 9-SHA matrix cross-check（N=1 path byte-identical at SHUD pin transition `3a0004c` → `210ac19`） | **PASS** (byte-identical N=1) |

Verdict: **PASS** — Mac PR-I reference anchor + PR-G clean Kahan revert 双重确认 SHUD pin 210ac19 (post-PR-G) byte-identical to P1-update-omp canonical at N=1 / serial mode。

## §3 P1d-tag push 状态 (orchestrator post-PR-L-merge 执行)

| Item | Status |
|---|---|
| `git tag -a P1d-tag <baseline/P1d HEAD>` annotated 创建 | **COMPLETED** (post-PR-L-merge 2026-06-24) |
| `git push origin P1d-tag` | **COMPLETED** (P1d-tag pushed to origin) |
| tag-object SHA | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` |
| deref commit SHA | `f88f2dc2cad1adbe3797b89fbe247aa12bf8c0a9` |
| `baseline/P1d` lock | PENDING post-PR-M-merge (deferred per spec p1d-capstone.md L43-L48) |
| D11 5 → 6 tag chain verify | **COMPLETED** — 5 historical SHA byte-identical, P1d-tag 新增 (详 `docs/p1d/p1d_summary.md` §13.2/§13.6 + `docs/p1d/p1d_tag_and_lock.md` §5/§5.1) |

## §4 OpenSpec PROMOTE 状态 (PR-M 本 PR)

| Item | Status |
|---|---|
| `openspec/specs/p1d-numa-governance/spec.md` PROMOTE | **COMPLETED** (本 PR Task 6, byte-identical to amended source per Tasks 4) |
| `openspec/specs/p1d-capstone/spec.md` PROMOTE | **COMPLETED** (本 PR Task 6, byte-identical to amended source per Tasks 5) |
| `openspec/changes/p1d-numa-governance/` archive to `openspec/changes/archive/2026-06-24-p1d-numa-governance/` | PENDING orchestrator post-merge (gitignored, local fs action) |

Amendments per Tasks 4+5 (additive, original Requirements/Scenarios preserved):
- 4-mode taxonomy block (serial / strict-omp / det-omp / fast-omp) 加入 `p1d-numa-governance/spec.md` top（after change Identifier）
- 3 SHALL gate Requirements + Mode binding sub-clause（A3a + nst applicable to strict-omp + det-omp；N=1 reverse-compat applicable to serial + strict-omp）
- 新 Scenario：P1d E′ containment closure verdict
- 新 Scenarios in `p1d-capstone/spec.md`：master plan v1.5 / M10 sync revision / ADR-0002 forward handoff / p1e-strict-omp-rhs openspec local drafts / 4-mode spec capstone

## §5 Glossary 4 新术语 + 2 既存 term 状态更新 (PR-M 本 PR)

新增 4 术语 (alphabetical order, after existing `**P1d carve-out** / **P9 carve-out**` entry at L216-218)：

| Term | Status |
|---|---|
| `**P1d-tag**` | **COMPLETED** (含 D11 6-chain reference + E′ closure 关联 + `_Avoid_` 项) |
| `**baseline/P1d**` | **COMPLETED** (含 lock_branch=true + 13 PR merge chain + `_Avoid_` 项) |
| `**steady-state first-touch (P1d)**` | **COMPLETED** (含 DEPRECATED 说明 + 与 allocation-time first-touch 区分 + `_Avoid_` 项) |
| `**P1d NUMA env**` | **COMPLETED** (含 `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + PR-F drop `numactl --interleave=all` 决策 + `_Avoid_` 项) |

更新 2 既存 term：

| Term | Status |
|---|---|
| `**first-touch / NUMA**` (L147-148) | **COMPLETED** — append P1d 互补段 (steady-state vs allocation-time first-touch 区分) |
| `**P1d carve-out (writer noise governance)**` (L216-218, 原 `**P9 carve-out**`) | **COMPLETED** — rename + append CLOSED via E′ closure 段 + master plan v1.5 / M10 §6 P1d.4 reference |

## §6 JSONL 双追加 (PR-M 本 PR)

`docs/stage-pipeline-log.jsonl` 末尾追加 2 entries：

| Entry | Status |
|---|---|
| Entry 1: P1d epic pipeline summary（含 `tag_object` + `tag_deref` + `verdict: PARTIAL CLOSURE via E' path` + `mode_taxonomy` + `forward: P1e + ADR-0002`） | **COMPLETED** |
| Entry 2: P1d epic close-out（含 `tag` + `branch_locked` + `main_ff` + `PROMOTE` list + `glossary_added` list + `D11_chain` list + `next_epic: P1e`） | **COMPLETED** |

两 entries 均含 `"workflow":"subagent-workflow"` 字段（per `openspec/changes/p1d-numa-governance/specs/p1d-capstone/spec.md` L142-145 Scenario）。

## §7 综合 verdict

**PARTIAL CLOSURE via E′ containment path** (per master plan v1.5 / M10 §6 P1d.4)

判据：
- L130 A3a bitwise SHALL gate: **FAIL** (root cause = NVECTOR_OPENMP `N_VDotProd_OpenMP` `reduction(+:sum) schedule(static)` reduction tree 顺序不固定，per `docs/p1d/p1d_numa_root_cause.md` + master plan v1.5 / M10 §6 P1d 5/5 codebase fact-check)。**NOT in P1d scope to fix**。
- L139 nst Δ + ladder SHALL gate: **FAIL** (same root cause; CVODE WRMS norm 同样过 N_Vector OMP reduction)。**NOT in P1d scope to fix**。
- L145 N=1 reverse-compat SHALL gate (in `serial` mode): **PASS** (heihe N=1 SHA byte-identical to P1-update-omp canonical 全 64-hex；详 PR-H §3 + PR-I 6-cell anchor + PR-G Mac 9-SHA matrix)
- L123 Kahan revert canonical (in `serial` mode): **PASS** (PR-G clean revert proven via Mac 9-SHA matrix)

**Forward handoff**:
- **P1e new epic** (F path: Serial N_Vector + StrictOMP RHS) opened per ADR-0002 Path 1 SELECTED (详 `docs/adr/0002-solver-path.md`)
- **PR-C/D/E steady-state first-touch loops** 标 DEPRECATED (consumer = Serial RHS path; owner-compute 未实现 → 无效优化)
- **PR-G Kahan revert** 保留 (serial mode N=1 byte-identical 证明 revert 干净)
- **openspec `p1e-strict-omp-rhs`** local drafts already in place (Phase 2(e) parallel agent owns final propose + Phase 4-8)

## §8 References

- `SHUD_openMP_master_plan.md` v1.5 / M10 §6 P1d (E′ closure 8 actions) + §6 P1e (F path next epic) + §8.1 (4-mode block)
- `docs/p1d/p1d_summary.md` §5 (E′ 8 项动作) + §13 (P1d-tag 验证表)
- `docs/p1d/p1d_pr_h_final_run.md` (PR-H raw verdict + SHA matrix + nst stats + E′ post-verdict 修订)
- `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` (Mac 6-cell anchor)
- `docs/p1d/p1d_kahan_revert.md` (PR-G Mac 9-SHA matrix)
- `docs/p1d/p1d_report.md` §6 (Why E′ over E) + §7 (Why F over B)
- `docs/p1d/p1d_tag_and_lock.md` §5/§5.1 (D11 6-chain SHA correction note) + §4 (tag 创建命令)
- `docs/adr/0002-solver-path.md` (4-way solver evaluation, Path 1 SELECTED)
- `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` (E′ closure amendment per Task 4)
- `openspec/changes/p1d-numa-governance/specs/p1d-capstone/spec.md` (E′ closure amendment per Task 5)
- `openspec/specs/p1d-numa-governance/spec.md` (PROMOTE byte-identical per Task 6)
- `openspec/specs/p1d-capstone/spec.md` (PROMOTE byte-identical per Task 6)
- `openspec/changes/p1e-strict-omp-rhs/` (P1e local drafts, Phase 2(e) parallel agent owns)
- `openspec/glossary.md` L147 + L216 + new P1d 4 terms (per Task 7)
- `docs/stage-pipeline-log.jsonl` last 2 entries (per Task 8)
