## 工作情况说明（Merge 前）

- 关联 Issue：#344
- PR：#356
- 冻结提交：`01fd43f`
- 上游 Epic：#338 (p8pre-spike Step 2 entry)
- 前序 PR：#350/#352/#353/#354/#355（Step 0 + Step 1 capstone 4 PRs，branch a PROCEED）

### 背景与目标

p8pre-spike **Step 2 P8-precond-0 spike 启动 PR-D pre-flight slice**（precedes #345 PR-D impl 落 identity precond stub）。本 PR 完成 3 件事：

1. **SUNDIALS 6.0.0 preconditioner API 头文件 verify** — 4 grep PASS (`CVodeSetLSetupFrequency` cvode.h:132, `CVodeSetJacEvalFrequency` cvode_ls.h:91, `CVLsPrecSetupFn` typedef cvode_ls.h:57, `CVLsPrecSolveFn` typedef cvode_ls.h:61) + REJECT distinction (`CVodeSetMaxConvFails` cvode.h:133 不是 setup frequency 替代品，targets 非线性 conv-fail 阈值不同子系统)
2. **SHUD submodule upstream fork `openmp-baseline-p8pre`** — from MANDATORY base `7a1dc8f` (P1e ship pin per spec p8precond-zero-identity-spike Scenario "Step 2 forward-only descendant extension" L148-154) + push to upstream + 验证 forward-only descendant strict criteria (rev-parse + merge-base 均 == 7a1dc8f)
3. **写 lightweight evidence log doc** — `docs/p8pre/api_verification.md` 169 行 6 §s（NOT academic style；pre-flight evidence anchor for #348 ADR-0003 cite）

### 本次具体改动

| 文件 / 上游状态 | 改动概要 |
|---|---|
| `docs/p8pre/api_verification.md` (新, 169 行) | Lightweight evidence log: §1 目的 + §2 4-grep PASS table + §3 REJECT distinction + §4 fork command transcript + §5 forward-only descendant strict criteria + §6 引用 |
| **SHUD upstream NEW branch** `openmp-baseline-p8pre` | origin SHA = `7a1dc8f6ea9e5496f516255406ee3563d397959b` (P1e ship pin exactly) |

无 SHUD outer submodule pointer bump（仍 `7a1dc8f`；pointer bump 留 PR-D impl #345），无 SHUD 源码 commit on `openmp-baseline-p8pre`（fork-only this PR），无 `.gitmodules` 改动，无 CI rule 改动。

### 测试与验证

**4 API grep PASS table** (Mac local @ SHUD/InstallSundials/include/cvode/):

| # | Symbol | Header | Line |
|---:|---|---|---:|
| 1 | `CVodeSetLSetupFrequency` | cvode.h | 132 |
| 2 | `CVodeSetJacEvalFrequency` | cvode_ls.h | 91 |
| 3 | `CVLsPrecSetupFn` typedef | cvode_ls.h | 57 |
| 4 | `CVLsPrecSolveFn` typedef | cvode_ls.h | 61 |

**REJECT 区分**: `CVodeSetMaxConvFails` (cvode.h:133) — 控制 SUNNonlinSol 非线性 convergence-failure 阈值 — 与 `CVodeSetLSetupFrequency` (cvode.h:132) 控制 CVODE step controller setup 频率属不同子系统，两者都是独立 SUNDIALS function，不能互相替代。

**SHUD branch fork verification**:
- Pre-fork SHUD HEAD = `7a1dc8f` (P1e ship pin)
- `git checkout -b openmp-baseline-p8pre 7a1dc8f && git push -u origin` 成功
- Post-push origin SHA = `7a1dc8f` (strict equality)
- Forward-only strict criteria (degenerate case at pre-flight):
  - `rev-parse openmp-baseline-p8pre == rev-parse 7a1dc8f` → IDENTICAL
  - `merge-base openmp-baseline-p8pre 7a1dc8f == rev-parse 7a1dc8f` → IDENTICAL

**openspec**: `openspec validate p8pre-spike --strict --no-interactive` exit 0
**CI**: 5/5 PASS (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (compact, 2 parallel reviewers, **全 APPROVE 0 findings**):
  - `review-correctness`: 8/8 checks PASS（4 grep accuracy + REJECT distinction + fork SHA verify + outer pointer unchanged + forward-only math + doc cross-ref + no source modification on new branch + openspec strict）
  - `review-integration`: 8/8 checks PASS（C8 submodule workflow + fork base anchoring + API verification reproduced + doc structure + outer repo sanity + forward dep cross-ref + CI compat + idempotency）
- **Phase 4.5 verifier**: SKIPPED (0 PLAUSIBLE candidates)
- **Phase 5/6**: SKIPPED (cross-review clean)
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，10/10 AC PASS（含 forward-only + .gitmodules 未触 2 bonus AC），oracle integrity PASS

### 兼容性、风险与已知限制

- 无 API 兼容性影响（pre-flight 验证；fork-only）
- SHUD upstream `openmp-baseline` master 分支未触（C8 不污染）
- `openmp-baseline-p8pre` 是 long-lived working branch — PR-D impl #345 / PR-E #346 / PR-F #347 都在此 branch 上工作
- **forward-known limit**: pre-flight degenerate case（branch HEAD == fork base，0 commits ahead）；#345 commits 落地后 rev-parse strict equality 会失效，但 merge-base equality 仍必须 hold per forward-only guarantee

### 维护者关注点

- 无额外关注点。下一步 **#345 PR-D impl**：
  - 建 `SHUD/src/Equations/MD_precond_identity.{h,cpp}` 在 `openmp-baseline-p8pre` branch
  - 编辑 `SHUD/src/Equations/cvode_config.cpp:259` (`PREC_NONE` → `PREC_LEFT`) + `CVodeSetPreconditioner` + `CVodeSetLSetupFrequency` wire-up
  - SHUD commit + push to openmp-baseline-p8pre
  - 外层 outer pointer bump 验证 forward-only descendant
  - Mac local build sanity + smoke test `./shud keliya` (验 nps>0 + npe>0)
