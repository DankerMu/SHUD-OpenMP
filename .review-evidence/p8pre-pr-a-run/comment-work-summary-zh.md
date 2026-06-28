## 工作情况说明（Merge 前）

- 关联 Issue：#341
- PR：#353
- 冻结提交：`83a8864` (Phase 6 fix 后)
- 上游 Epic：#338 (p8pre-spike Step 1)
- 前序 PR：#350 (Step 0 doc-correction merged 2026-06-27) + #352 (Step 1 PR-A prep merged 2026-06-27)

### 背景与目标

p8pre-spike Step 1 PR-A **run slice**（precedes #342 PR-B aggregator）。本 PR 完成 5 件事：

1. **写 server-side runner script** — `tools/p8pre/run_n8_profile.sh`：pre-flight Mode C 二进制 nm verify（auto-rebuild on miss），mkdir 18 cell artifact dir，invoke render wrapper，parse + JID 替换（`__PREV_JID_*` placeholder dependency 链）+ sbatch `--parsable`，append jid_table.txt
2. **修复 #340 模板 2 处 bug**（PR #352 deliverable 实跑时暴露）：
   - **bug-1 cwd**：原 `cd SHUD/ && ./shud __CASE__` 失败 — heihe_x4 无 `SHUD/input/heihe_x4/` demo dir（只 heihe 有）。改为 P1e PR-B convention `cd Basins/<case>/ && $SHUD_DIR/shud <case>`。
   - **bug-2 output dir + filename**：SHUD 实际写到 `<cwd>/output/<case>.out/` 文件名 `profile_B0.yaml` + `<case>.rivqdown.dat`，非原模板假设的 `output/<case>/ + profile.yaml + rivqdown.dat`。
3. **服务器 18 cell Slurm 实跑 + 验收** — JID 9510-9527 全 COMPLETED ExitCode 0:0：
   - heihe (cn14): wall 1:36-2:28 × 9 cells
   - heihe_x4 (cn15): wall 12:52-25:04 × 9 cells
   - 总 wall ~67 min（singleton-chain max heihe_x4 N=1 dominant）
4. **per-cell 6 gate verification** — 18 cells × {ART, CANON15, REJECT, EXTRAS, BSUM±2%, RC0} 全 PASS，rsync mirror 18 dir (~46MB) → /tmp/p8pre_n8_profile/
5. **写 execution log doc** — `docs/p8pre/n8_profile_run.md` 学术风格 164 行 8 §s

并修复 Phase 4.5 round 1 暴露的 2 个 CONFIRMED candidate（详见 review 闭环段）。

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `tools/p8pre/run_n8_profile.sh` (新, 271 行, exec) | Server-side runner: Mode C nm pre-flight + render + sbatch --parsable + JID 替换 + jid_table.txt |
| `tools/p8pre/submit_n8_profile_template.sbatch` (修, 2-bug fix) | cwd → `Basins/<case>/` + 输出文件 `profile_B0.yaml` + `<case>.rivqdown.dat` |
| `docs/p8pre/n8_profile_run.md` (新, 164 行) | Academic-style execution log，8 §s，含 wall + JID + 18-cell verification 表 + cross-N CVODE 观测 |
| `.gitignore` (修, +4 行) | Phase 6 fix cand-01: 新增 `.p8pre-runs/` + `.p8pre-pr-*-runs/` 与 p1e block 同序 |

无 SHUD submodule pin bump（保持 `7a1dc8f`），无 SHUD 源码改动，无 case manifest 改动，无 CI rule 改动。

### Cross-N CVODE counter bitwise-identical（PR-B #342 input）

- **heihe** (9 cells N∈{1,4,8} × 3 reps): `nst=6698, nfe=6943, nfeLS=12632, nni=6942, nli=12632` strict
- **heihe_x4** (9 cells): `nst=6575, nfe=6741, nfeLS=30509, nni=6740, nli=30509` strict

均严格匹配 P1e PR-I absolute baseline anchor（`docs/p8pre/step1_prep.md` §3 nst ladder + §4 nfe baseline）→ Mode C 无 silent 数值回退。

### 测试与验证

**本地**：
- `bash -n` runner + template exit 0
- `git check-ignore .p8pre-runs/x` exit 0 (Phase 6 fix 后)
- `grep -rn "profile\.yaml" openspec/changes/p8pre-spike/` = 0 (cand-02 fix 后)
- `openspec validate p8pre-spike --strict --no-interactive` exit 0
- `/tmp/p8pre_n8_profile/` 18 cell dir 完整

**服务器 cn14 + cn15 (Slurm 实跑)**:
- 18 JIDs (9510-9527) 全 sacct State=COMPLETED ExitCode=0:0
- Wall: heihe 1:36-2:28 × 9 / heihe_x4 12:52-25:04 × 9 / 总 ~67min
- Slurm 三铁律 全 3 条遵守

**CI**: 5/5 PASS (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (expanded, 4 parallel reviewers):
  - `review-spec-compliance`: APPROVE，0 findings (6 spec scenarios 全 PASS)
  - `review-correctness`: APPROVE，0 findings + 2 non-blocking notes (BSUM 算法恒等 + 退出码 doc/code 漂移)
  - `review-integration`: REQUEST CHANGES，2 Warning candidate（→ Phase 4.5）
  - `review-security-perf`: APPROVE，0 findings + 7 non-blocking notes (defensive code + 与 P1e PR-I wall delta 由 Timer 仪表 bias 解释)
- **Phase 4.5 verifier** (2 parallel):
  - cand-01 `.gitignore` carry-forward → **CONFIRMED** (`.s*-runs/` 字面 anchor，不 cover `.p8pre-runs/`)
  - cand-02 openspec `profile.yaml` filename 漂移 → **CONFIRMED** (SHUD 实际 emit `profile_B0.yaml`，PR-B implementer 按 tasks 文本会 glob 0 个文件)
- **Phase 6 fix pass** (单 implementer):
  - cand-01: `.gitignore` +4 行 (1 注释 + 2 模式)
  - cand-02: 9 处文本替换跨 4 个 openspec/changes/ 文件（注：`openspec/changes/` per `.gitignore:13` 为 transient gitignored，fix 在 local working tree 生效；epic 内后续 implementer 全部从 my working tree fire → 不会被误导；至 #349 archive 才 land 入 persistent `openspec/specs/`）
- **Phase 6.5 round 2 cross-review** (focused: integration):
  - `review-integration` round 2: APPROVE，cand-01 + cand-02 **RESOLVED**，0 new findings
  - spec-compliance/correctness/security-perf round 1 已 clean，未复跑
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，12/12 AC PASS，oracle integrity PASS，APPROVE merge

### 兼容性、风险与已知限制

- 无 API 兼容性影响（runner + 模板 + 文档）
- 模板 bug 修复改变模板 cwd + 输出路径 schema：与 PR-B #342 aggregator 设计契约一致（`profile_B0.yaml` + `<case>.rivqdown.dat`），不与 P1e PR-I 路径冲突（不同 namespace）
- **carry-forward 已无**（cand-01 + cand-02 已在本 PR 闭环）
- **forward-known limit**: openspec/changes/ 是 transient gitignored，cand-02 fix 不在 git tracked diff 中；reviewer 需理解此 project convention（已在 PR body Phase 6 commit message 透露）

### 维护者关注点

- 无额外关注点。下一步 #342 PR-B aggregator：从 /tmp/p8pre_n8_profile/ 算 median per (case, N) + nfeLS/nfe + nli/nni + nst Δ cross-N + ROI a/b/c/d 4 分支判定。
