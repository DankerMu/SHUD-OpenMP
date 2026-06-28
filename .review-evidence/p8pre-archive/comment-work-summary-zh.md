## 工作情况说明（Merge 前）

- 关联 Issue：#349
- PR：#361
- 冻结提交：`6eb1a27`
- 上游 Epic：#338 (p8pre-spike Step 4 PR-archive 顶点 closeout)
- 前序 PR：#360（PR-G ADR-0003 + master plan + epic capstone）

### 背景与目标

p8pre-spike **Step 4 PR-archive 顶点 closeout 切片**。 epic 最末一个 PR。本 PR 完成 4 件事:

1. **glossary +7 canonical terms + ADR-0003 cross-ref** — `openspec/glossary.md` 252→288 lines
2. **openspec archive p8pre-spike** — PROMOTE 12 reqs (6 + 6) 到 `openspec/specs/{n8-mode-c-profile-recheck,p8precond-zero-identity-spike}/spec.md` + 移 change dir 到 `openspec/changes/archive/2026-06-27-p8pre-spike/` (gitignored 不入 git tree)
3. **spec L29 jok-mirror cite 修正** — PR-D #357 forward-debt F-R2-3 resolved
4. **post-archive openspec validate --specs --strict exit 0** verified

### 4 文件落地

| 文件 | 变化 |
|---|---|
| `openspec/glossary.md` (EDIT, +36 lines) | 7 canonical terms (p8pre-spike / identity preconditioner / PREC_LEFT / nfeLS/nfe ratio / CVodeSetLSetupFrequency / .p1e-i-runs/ / ncfl) + ADR-0003 cross-ref entry |
| `openspec/specs/n8-mode-c-profile-recheck/spec.md` (NEW PROMOTED) | 6 reqs PROMOTED from p8pre-spike change |
| `openspec/specs/p8precond-zero-identity-spike/spec.md` (NEW PROMOTED) | 6 reqs PROMOTED + L29-30 jok-mirror canonical cite (SUNDIALS cvDiurnal_kry.c L716/L760) + L29 wording fix `*jcurPtr = jok ? SUNFALSE : SUNTRUE` matching SHUD impl MD_precond_identity.cpp L61 |
| `openspec/changes/archive/2026-06-27-p8pre-spike/` (MOVED, gitignored) | full change dir archived per openspec convention |

### Spec L29 jok-mirror correction key technical finding

实现器 round 1 第一版 spec L29 写: "setting *jcurPtr = SUNFALSE (jok-mirror canonical pattern; identity spike specializes to *jcurPtr = SUNFALSE because identity holds no cached state)" — 这是错误的。

实际 SHUD `MD_precond_identity.cpp` L61 写: `*jcurPtr = jok ? SUNFALSE : SUNTRUE;` — true jok-mirror canonical 模式。

错误 spec 版本会导致后人误读 — 因为 unconditional `*jcurPtr = SUNFALSE` 实测产生 `npe=0` 违反 gate 3 (PR-F #359 验证)。jok-mirror 模式才能使 `CVodeGetNumPrecEvals → npe` counter increment 正常 (gate 3 PASS 关键)。

Round 1 fix @ `6eb1a27` 修正 spec L29 为: `*jcurPtr = jok ? SUNFALSE : SUNTRUE` + 完整 jok-mirror semantics 解释 + PR-D #357 implementer-deviation empirical verification 引用。spec/impl/SUNDIALS canonical cite 三方 triangulation 一致。

### Review 与修复闭环

- **Phase 4 round 1 compact cross-review** (single-reviewer `review-archive-closeout` combined spec-compliance + correctness + documentation lenses @ `6eb1a27`): **APPROVE 0 findings**
  - Archive operation completeness PASS (12 reqs PROMOTE + archive dir move + validate exit 0)
  - Glossary +7 terms + ADR-0003 cross-ref PASS (format consistent + cross-ref well-placed)
  - Spec L29 jok-mirror cite triangulation PASS (spec/impl/canonical 3-way一致, F-R2-3 forward debt CLOSED)
  - Out-of-scope discipline PASS (SHUD unchanged + docs/adr/0003 + docs/p8pre/* + master plan 全部 untouched)
  - Forward debt closeout PASS (PR-D F-R2-3 resolved; 5 PR-G cosmetic Suggestions correctly forward-carried to future tools epic)
- **Phase 4.5 verifier**: SKIPPED (0 candidates需 adversarial verification)
- **Phase 5/6/6.5/7**: SKIPPED (cross-review clean + compact fixture single-reviewer pack)
- **CI 5/5 PASS** @ `6eb1a27` (setup 3s / tools-tests 10s / build-and-compare keliya 1m3s / asan-ubsan keliya 38s / asan-ubsan qhh 5s)

### 兼容性、风险与已知限制

- 无 API 兼容性破坏 (pure doc + openspec archive 操作)
- SHUD upstream `openmp-baseline` master 未触 (C8 不污染)
- SHUD pin `5276167` unchanged (D8 fall-back 仍 deferred)
- `openspec/changes/p8pre-spike/` 已 archived 到 `openspec/changes/archive/2026-06-27-p8pre-spike/` (gitignored 不入 git tree, 但 working-tree 内可访问历史)
- baseline/p8pre branch + SHUD openmp-baseline-p8pre branch deletion = orchestrator-only post-merge actions per ADR-0003 NO-GO

### 维护者关注点

- Epic #338 close + branch deletions 由 orchestrator post-merge 执行 (per ADR-0003 NO-GO option (b) decision path)
- baseline/p8pre 删除后, 所有 9 个 PR (#350 + PR-A #341..PR-G #348 + PR-archive #349) 的 squash commits 仍 accessible via git reflog + GitHub PR 历史
- SHUD upstream `openmp-baseline-p8pre` branch 删除后, 上游 `openmp-baseline` master 仍 stable at SHUD pin 5276167 — 包含完整 P8-precond identity spike impl 留作 future epic reference (不进 production codepath)
- 5 PR-G cosmetic Suggestions 已 carry forward 到 future tools epic:
  - `.review-evidence/` gitignore patch
  - §3 column ordering cosmetic
  - max_ulp precision range cosmetic
  - compare_snapshot raw-double hardening
- Design D8 fall-back (revert cvode_config.cpp:259 + delete MD_precond_identity.{h,cpp}) — separate cleanup PR scope (非 #349 archive scope, 留 future work item)
