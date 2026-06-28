## 工作情况说明（Merge 前）

- 关联 Issue：#340
- PR：#352
- 冻结提交：`20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d`
- 上游 Epic：#338 (p8pre-spike Step 1)
- 前序 PR：#350 (Step 0 doc-correction merged 2026-06-27)

### 背景与目标

p8pre-spike Step 1 PR-A prep slice（precedes #341 PR-A run）。本 PR 完成 4 件事：

1. **写 Slurm 18-cell N=8 Mode C profile 模板** — 基于 `tools/p1e_2x2_sbatch_template.sbatch` 原型，drop `__BUILD__` marker (Mode C 固定)，保留 `__CASE__/__N__/__REP__/__NODE__`，加 cn14/cn15 case-pin。
2. **写 wrapper** — POSIX bash+awk，materialize 18-cell matrix (2 case × 3 N × 3 rep, N=2 排除 per design D4) + singleton afterany chain per (case, N)。
3. **服务器 cn14 build verify** — srun 实跑 `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`，3 nm SHALL gates 全 PASS。
4. **案例 basin verify** — heihe forcing.trimmed 29M + heihe_x4 basin 2.3G 全部 ≥ SHALL gate。

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `tools/p8pre/submit_n8_profile_template.sbatch` (新, 123 行) | Mode C 单 cell Slurm 模板，4 substitution marker，Slurm 三铁律 + determinism env + provenance log |
| `tools/p8pre/render_n8_profile.sh` (新, 147 行, exec) | wrapper expand 18 cells，POSIX bash+awk，cn14/cn15 case-pin，singleton chain，print-only (not auto-submit) |
| `docs/p8pre/pr_a_prep_evidence.md` (新, 151 行) | Academic-style evidence doc — build/nm/case/render 5 §s |

无 SHUD submodule pin bump，无 SHUD 源码改动，无 case manifest 改动，无 CI rule 改动。

### 测试与验证

**本地**：
- `bash -n` 两 script exit 0
- `bash tools/p8pre/render_n8_profile.sh | grep -c "^sbatch"` = 18 ✓
- 6 unique (case, N) group × 3 reps，N=2 absent，cn14:cn15 9:9 split ✓
- `openspec validate p8pre-spike --strict --no-interactive` exit 0

**服务器 cn14 (srun + Mode C build)**:
- `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` exit 0
- `nm ./shud | grep -c N_VNew_Serial` = 1 (Serial NVec linked) ✓
- `nm ./shud | grep -c N_VNew_OpenMP` = 0 (OpenMP NVec absent) ✓
- `nm ./shud | grep -c GOMP_parallel` = 1 (libgomp linked) ✓
- SHUD pin == `7a1dc8f` ✓
- heihe forcing.trimmed = 29M + heihe_x4 basin = 2.3G + forcing/ = 286M ✓

**CI**: 已通过 (详见 PR Checks)。

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过 Phase 0.5)
- **Phase 4 round 1 cross-review** (compact, 2 parallel reviewers):
  - `review-correctness`: APPROVE，0 blocking + 1 Suggestion (evidence-doc L60 SHALL 行 wording — "286M" → "≥ 200M"，non-blocking)
  - `review-integration`: APPROVE，0 blocking + 2 non-blocking notes (`.gitignore` 缺 `.p8pre-runs/` + JID placeholder clarification 都 carry-forward 到 #341)
- **Phase 4.5 verifier**: SKIPPED (0 concrete blocking candidates per compact precision-bias)
- **Phase 5/6 fix pass**: SKIPPED (cross-review clean)
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，11/11 AC PASS，oracle integrity PASS，APPROVE merge

### 兼容性、风险与已知限制

- 无 API / 数据格式 / 迁移兼容性影响（infra prep PR）
- `.p8pre-runs/` namespace 与 `.p1e-i-runs/` distinct，downstream PR-B aggregator 不会 mis-glob
- **carry-forward 到 #341**:
  1. `.gitignore` 补 `.p8pre-runs/` + `.p8pre-pr-*-runs/`
  2. PR-A runner doc 明确 JID 替换 target 是 stdout sbatch 行不是 .sbatch 文件体
- **accepted as-is**: evidence-doc L60 "286M" wording — 不影响 PR-A handoff，下游若需精确化在 PR-C capstone 调整

### 维护者关注点

- 无额外关注点。下一步 #341 PR-A run 将 ssh server 实跑 18-cell ~4.5h Slurm wall。
