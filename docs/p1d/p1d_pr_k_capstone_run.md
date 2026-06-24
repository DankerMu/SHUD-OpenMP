# P1d PR-K — capstone docs PR self-evidence

PR-K 是 P1d epic 的 capstone docs PR。本 doc 是 PR-K 自身的 evidence trail（仿 PR-F / PR-H / PR-I 的 self-evidence 模式）。详细 P1d 内容见同 PR 创建的 `docs/p1d/p1d_summary.md` + `docs/p1d/p1d_perf_baseline.md` + `docs/p1d/p1d_numa_root_cause.md` + `docs/p1d/p1d_report.md`。

## §1 Scope

PR-K 仅 docs 改动，不改 SHUD submodule pointer / OpenSpec specs / 代码 / 测试 / CI workflow。

**boundary**：

| 范围 | PR-K 是否包含 |
|---|---|
| 创建 `docs/p1d/p1d_summary.md` | ✓ |
| 创建 `docs/p1d/p1d_perf_baseline.md` | ✓ |
| 创建 `docs/p1d/p1d_numa_root_cause.md` | ✓ |
| 创建 `docs/p1d/p1d_report.md` | ✓ |
| 创建 `docs/p1d/p1d_pr_k_capstone_run.md` (本 doc) | ✓ |
| 更新 `docs/status_matrix.md` (P1d 行 + P1e 行) | ✓ |
| 更新 `docs/build_manifest.md` (SHUD pin trail P1d 段) | ✓ |
| 修改 SHUD submodule pointer | ✗ (P1d-tag 在 PR-L 创建) |
| 修改 `openspec/specs/` (PROMOTE spec) | ✗ (PR-M scope) |
| 创建 `docs/p1d/p1d_tag_and_lock.md` | ✗ (PR-L scope) |
| 创建 `docs/adr/0001-solver-path.md` | ✗ (Phase 2(e) 并行 agent owns) |
| 创建 `openspec/changes/p1e-strict-omp-rhs/` | ✗ (Phase 2(e) 并行 agent owns) |

## §2 Files created / modified

| Path | Status | Lines (approx) | 用途 |
|---|---|---|---|
| `docs/p1d/p1d_summary.md` | NEW | ~280 | P1d capstone source of truth (§1-§12) |
| `docs/p1d/p1d_perf_baseline.md` | NEW | ~230 | wall + Amdahl + Mac 6-cell + CPU-hour cost ROI |
| `docs/p1d/p1d_numa_root_cause.md` | NEW | ~340 | 技术 autopsy (§1-§13) |
| `docs/p1d/p1d_report.md` | NEW | ~230 | epic executive report (§1-§12) |
| `docs/p1d/p1d_pr_k_capstone_run.md` | NEW | ~110 | PR-K self-evidence (本 doc) |
| `docs/status_matrix.md` | MODIFY | +2-3 lines (P1d row 新增 + P1e row 新增 + preamble update) | 状态矩阵更新 |
| `docs/build_manifest.md` | MODIFY | +20-30 lines (P1d-tag 章节 + SHUD pin trail 更新) | 构建清单更新 |

共 **4 NEW + 2 MODIFY + 1 PR-K self-evidence = 7 file 操作**。

## §3 Cross-reference grep verification

5 examples 证明本 PR 创建的 docs 与 master plan v1.5 / M10 + 已有 P1d docs 的引用对齐：

### §3.1 master plan §6 P1d ↔ p1d_summary.md

| master plan v1.5 §6 P1d 节 | p1d_summary 对应 § |
|---|---|
| §6 P1d.2 5/5 fact-check | §6 5/5 codebase 事实核查（5 entry 表 — line-for-line 一致）|
| §6 P1d.3 SHALL gate verdict 表 | §3 3 SHALL gate verdict（4 entry 表）|
| §6 P1d.4 E′ 8 项动作 | §5 E′ containment closure 8 项动作（数对齐）|
| §6 P1d.5 baseline lock + tag (D11 6-tag chain) | §12 References 含 P1d-tag forward |
| §6 P1d.7 forward 移交 → P1e | §9.1-9.5 P1e 启动前置 + 技术主线 + 2×2 build matrix + 实施要点 + P2a 启动前置改链 |

### §3.2 M10 quote block ↔ p1d_numa_root_cause.md

M10 quote 含 `(a) f.cpp:54 + (b) MD_rhs_core.cpp:802-811 + (c) Makefile + (d) cvode_config.cpp:259 + (e) reduction(+:sum) schedule(static)` 5 项 fact-check；p1d_numa_root_cause.md §3 5/5 表 1:1 对齐，§4 详细解释每条 fact 的含义 + SUNDIALS 6.0.0 source path。

### §3.3 PR-H verdict ↔ p1d_report.md

PR-H final run doc (`docs/p1d/p1d_pr_h_final_run.md`) §"Three SHALL gate verdict"（4 gate 表）↔ p1d_report.md §5 What was NOT delivered（4 项 FAIL 表）+ p1d_summary.md §3（3 SHALL gate verdict 表）— 三处数据一致：

- heihe N=1 SHA `7f22bd6faa438d50...` PASS
- heihe N=4 nst Δ=80 / N=8 nst Δ=152 FAIL
- heihe_x4 N=4 \|Δ\|=11 / N=8 \|Δ\|=4 FAIL
- N=1 reverse-compat PARTIAL

### §3.4 PR-G Mac 9-SHA matrix ↔ p1d_perf_baseline.md

PR-G doc (`docs/p1d/p1d_kahan_revert.md` §"SHA matrix" 表) Mac 3×3 = 9 SHA 数据 ↔ p1d_perf_baseline.md §7.3 PR-G Mac 9-SHA matrix cross-reference 表 — 三 config × {PR-E baseline / pre-K2 / post-PR-G} = 9 SHA 完整对齐：

- serial keliya post-PR-G == pre-K2 `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` ✓
- omp@N=1 OMP_PROC_BIND unset post-PR-G == pre-K2 `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` ✓
- omp@N=1 OMP_PROC_BIND=close post-PR-G == pre-K2 `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` ✓
- 同时 post-PR-G == PR-I anchor (P1-update-omp-tag canonical) — 证明 PR-G clean revert + first-touch loops bitwise-neutral at N=1

### §3.5 早期 profile RHS 66.55% ↔ Amdahl 上界 2.39× ↔ P1e 目标

- `docs/profile_decision.md` heihe_x4 profile_B0.target.yaml RHS 占 wall 66.55% (B1a era)
- p1d_perf_baseline.md §5 计算理想 8 核 Amdahl 上界 `1 / (1 - 0.6655 + 0.6655/8) = 2.39×`
- p1d_perf_baseline.md §10.1 P1e strict-omp mode 加速比验收 Medium (heihe ~6k) M=1.5× / Large (heihe_x4 ~25k) T=2.4×
- p1d_summary.md §4.4 + §9.2 当前 1.27× 距 2.39× 差距 = "真正应并行的 RHS 还没并行"
- master plan v1.5 §6 P1e.1 设计意图相同 "理想上界 2.39× 加速可达"

5 处算式 + 数据一致。

## §4 Boundary surface

PR-K 是 **docs-only** PR：

- **SHUD submodule pointer**: 不动（保持 PR-G `210ac19` Kahan revert + first-touch stacked）
- **OpenSpec specs**: 不动（`openspec/specs/` 不变；`openspec/changes/p1d-numa-governance/specs/` 已是 PROMOTE candidate, PR-M PROMOTE 时移到 `openspec/specs/`）
- **Code / src / tests**: 不动
- **CI workflow**: 不动（`.github/workflows/serial-baseline.yml` 不变）
- **Master plan**: 不动（v1.5 / M10 已在前序 commit 合入 `a19fb5e`）
- **`docs/adr/0001-solver-path.md`**: 不创建（Phase 2(e) 并行 agent owns）
- **`openspec/changes/p1e-strict-omp-rhs/`**: 不创建（Phase 2(e) 并行 agent owns）
- **`docs/p1d/p1d_tag_and_lock.md`**: 不创建（PR-L scope）

## §5 CI expectation

预期 PASS（doc-only PR）：

- `serial-baseline.yml` keliya N=1 vs B0 bitwise — PASS（无 SHUD pointer 变化）
- `asan-ubsan keliya + qhh` — N/A（无 source 变化）
- `tools-tests` — N/A（无 tools/ 变化）

新增 doc 不触发任何 CI gate。Markdown lint (if any) PASS — 本 PR 使用标准 Markdown 表格 + 代码块 + 链接 + 中英文混排（technical terms in English）。

## §6 Style consistency check

| 维度 | 检查 |
|---|---|
| Markdown 风格 | header 用 `##` / `###` 层级；表格用 `|---|` 对齐；代码块用三反引号 + 语言标 |
| 中英文混排 | 项目 convention bilingual；technical terms / file paths / commands 用 English；narrative 用中文 |
| File path 引用 | 绝对路径或相对 from `repo root`（如 `SHUD/src/Model/f.cpp:54`） |
| Cross-ref | 引用 master plan 用 `§N` 表示；引用 sibling docs 用 `docs/p1d/<name>.md` 相对路径 |
| SHA 引用 | 全 64-hex 或明确截断（如 "first 16" / "head 12"） |
| 数据 source | 全部 empirical data 来自 P1d brief 明确给出的 7 个表（rivqdown 散度 / wall+speedup / Amdahl recalc / Mac PR-I 6-cell / PR-G Mac 9-SHA / 5/5 fact-check / 3 SHALL gate verdict） |
| 不引入新事实 | 仅引用已存在 doc / source / fact-check；不做新断言 |

## §7 Limits

- 本 doc 不审计 SHUD source 实际行号是否 drift（用 brief 提供的 `f.cpp:54` / `MD_rhs_core.cpp:802-811` / `Makefile:140` / `cvode_config.cpp:259` 行号 verbatim；若 SHUD pin 后续变化导致行号 shift，本 doc 引用过期，但事实 #1-#5 内容不变）
- 本 doc 不验证 master plan v1.5 / M10 字面是否与本 PR 7 doc 内容**完全**对齐到字（仅做 §3.1-§3.5 高层对齐验证；line-for-line audit 留 PR-M PROMOTE 阶段 reviewer 做）
- ADR-0001 + p1e-strict-omp-rhs openspec change 由 Phase 2(e) 并行 agent owns；本 doc 标记为 "forthcoming" 但不创建

## §8 References

- `docs/p1d/p1d_summary.md` — capstone narrative
- `docs/p1d/p1d_perf_baseline.md` — perf data
- `docs/p1d/p1d_numa_root_cause.md` — 技术 autopsy
- `docs/p1d/p1d_report.md` — executive report
- `docs/status_matrix.md` — P1d 行 + P1e 行 (本 PR 更新)
- `docs/build_manifest.md` — SHUD pin trail (本 PR 更新)
- `SHUD_openMP_master_plan.md` v1.5 / M10 — §6 P1d + §6 P1e
- `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` — 3 SHALL gate canonical
- `docs/p1d/p1d_pr_h_final_run.md` — PR-H verdict + E′ post-verdict 修订
- `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` — Mac anchor reference
- `docs/p1d/p1d_kahan_revert.md` — PR-G clean revert evidence
