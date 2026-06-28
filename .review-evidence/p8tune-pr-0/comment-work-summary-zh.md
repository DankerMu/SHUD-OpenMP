## 工作情况说明（Merge 前）

- **关联 Issue**：#363（p8tune-spgmr-maxl epic #362 的 PR-0 of 6）
- **PR**：#369
- **冻结提交**：`e3aa6dbc977b2fad3fef49a6a099c74fe029b872`
- **PR base**：`baseline/p8tune`（off `main` 95b9158；epic 集成基线）

### 背景与目标

p8pre-spike epic #338 NO-GO 关闭后（ADR-0003 在 outer `e442ce8` / SHUD `37be0fe`），三处合并 doc（ADR-0003 L70 / identity_spike_verdict §6.2 L169 / capstone §6.4.3 L236）把未来候选 gate 错引为 `ncfn < 6 ∧ ncfn < 47`。这些 6/47 数字来自 Step 2 `PREC_LEFT + identity` spike 的 floor（PREC_LEFT 因 S5 结构漂移已被拒，不会进入 production），而真正的 `PREC_NONE` production floor 是 7/51（per `n8_profile_verdict.md` §3.1）。

PR-0 是 p8tune-spgmr-maxl epic 的 doc-state-correction 基础：把所有 future-candidate gate citations 改正到 `ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)`，6/47 仅保留为 negative-control anchor。同时修正 `nfeLS = 30509` 的 3 处 typo + 更新 cleanup 状态措辞 + 新增 2 个 glossary 术语 + master plan 新增 §P8-tune.C 章节。

### 本次具体改动

| 模块 | 文件 | 改动 |
|---|---|---|
| ADR | `docs/adr/0003-precond-spike-decision.md` | L22 nfeLS 30518→30509；L53 §Decision §2 cleanup 状态 "deferred → completed at outer `e442ce8` / SHUD `37be0fe` (2026-06-27)" + 4-item audit trail；L70 §Consequences positive bullet 重写（6/47 = negative-control anchor / future gate = ≤ 7/51 per §3.1）；§Negative bullet 2 cleanup 措辞；§Forward action §1 checklist [x] 完成 items 1-6 |
| Verdict | `docs/p8pre/identity_spike_verdict.md` | L169 §6.2 PASS-criterion 改正；L193-196 Cleanup status update sub-paragraph + forward link to spgmr-maxl-env-hook + maxl-sweep-verdict + ADR-0004 TBD；L224 §9.2 Prerequisite 1 改正 |
| Capstone | `docs/p8pre/capstone.md` | L161-163 §5.1 表 heihe_x4 nfeLS 30517→30509（3 行）；L234-238 §6.4.3 PASS-criterion bullet 1 改正 + bullet 4 mode-C-tune 引用 + forward link；L313 §9.2 Prerequisite 1 改正；§9.1 cleanup section 重命名为 "design D8 cleanup COMPLETED at outer e442ce8 / SHUD 37be0fe (2026-06-27)" + 6 actions [x] |
| Summary | `docs/p8pre_summary.md` | 新增 §Forward note (post-cleanup 2026-06-27) 在 epic value summary 与 References 之间，记 cleanup completion + 6/47 negative-control 澄清 + 未来 PASS gate = ncfn_candidate ≤ 7/51 + cross-ref to p8tune-spgmr-maxl + ADR-0004 TBD |
| Glossary | `openspec/glossary.md` | L271 nfeLS ratio 30518→30509 + authoritative cross-ref；新增 "### p8tune-spgmr-maxl (SPGMR maxl tune) 集合"：`mode-C-tune` 术语（per design D9：per-(case, maxl) anchor / A3a not required cross-maxl / A4 max_ulp cross-maxl AND cross-N within-maxl / hydrology = A4 fallback only）+ `SHUD_SPGMR_MAXL` 术语（4 safety constraints: 不开 PREC_LEFT / 不注册 preconditioner / default-unset bit-identical SHUD 37be0fe / invalid abort via myexit(ERRCVODE)） |
| Master plan | `SHUD_openMP_master_plan.md` | L2394 §P8-precond.0 Outcome "cleanup deferred → COMPLETED at outer e442ce8 / SHUD 37be0fe (2026-06-27)" + pointer to §P8-tune.C；L2396-2436 新增 §P8-tune.C（epic 范围 4 capabilities × 6 PR + conditional PR-F；entry condition: hard-evidence already satisfied per §3.1 heihe ncfl=85 / heihe_x4 ncfl=3620；6-PR sequence 表；8-gate verdict 表 G1-G8；ADR-0004 6-branch verdict 表） |

**Diff stats**：6 files, 108 insertions(+), 36 deletions(-)
**Scope guarantee**：0 .c/.cpp/.h/.sh/Makefile/.py/.yaml 改动；0 SHUD submodule pointer 改动；SHUD pin 保持 `37be0fe`

### 测试与验证

| 验证 | 命令 | 结果 |
|---|---|---|
| openspec strict | `openspec validate p8tune-spgmr-maxl --strict` | PASS (`Change 'p8tune-spgmr-maxl' is valid`) |
| ncfn 6/47 残留 | `grep -nE 'ncfn ?< ?6\|ncfn ?< ?47' docs/` | 0 matches |
| 30518/30517 typo 残留 | `grep -rnE '30518\|30517' docs/ openspec/glossary.md` | 0 matches |
| Glossary 新术语 | `grep -cE 'mode-C-tune\|SHUD_SPGMR_MAXL' openspec/glossary.md` | 5 matches (≥ 2 expected) |
| Master plan §P8-tune.C | `grep -c 'P8-tune.C' SHUD_openMP_master_plan.md` | 4 matches (≥ 1 expected) |
| Scope 不含源码 | `git diff --name-only \| grep -E '\.(c\|cpp\|h\|sh\|py\|yaml)$\|Makefile'` | 0 matches |
| SHUD pointer 未变 | `git diff --name-only \| grep -E '^SHUD$'` | 0 matches |
| CI | 5 checks (setup / build-and-compare / asan-ubsan keliya / asan-ubsan qhh / tools-tests) | 5/5 PASS |

### Review 与修复闭环

| Phase | Reviewer | 轮次 | 结论 |
|---|---|---|---|
| 0.5 fixture review | reviewer subagent | — | PASS (8/8 checks) |
| 4 cross-review | review-correctness + review-spec-compliance（并行） | round 1 | 双 CLEAN，0 actionable findings |
| 4.5 verifier gate | (空：0 candidates) | round 1 | — |
| 6/6.2/6.5 | — | — | SKIPPED（无 findings，无 pattern escalation） |
| 7 final review (Gap Sweep) | independent-final | — | CLEAN, Reject-When 精度门应用，APPROVE merge |

**Phase 6 修复**：无（review 全 CLEAN，无 fix 轮）。

### 兼容性、风险与已知限制

- **Doc-only change**：无 API / runtime / 数据 / 二进制 兼容性影响
- **SHUD submodule 未触**：pin 保持 `37be0fe`（cleaned PREC_NONE state，p8pre design D8 cleanup tail）
- **下游 PR 依赖已 setup**：
  - PR-A 可直接引用 corrected 7/51 floors（spec `clean-prec-none-baseline` §codepath-equivalence 用到）
  - PR-C 可使用 glossary `SHUD_SPGMR_MAXL` 标准化命名（spec `spgmr-maxl-env-hook` cross-ref）
  - PR-D/E 可引用 mode-C-tune reference set（spec `maxl-sweep-verdict` 用到）
  - ADR-0004 forward refs 已铺设（PR-E 会 author）
- **已知 limitation（informational, 不阻塞 merge）**：
  - `docs/p8pre/capstone.md` §5.1 表 column `4.527` 与权威 ratio `30509/6741 = 4.526` 小差异；属 PR-A `clean-prec-none-baseline` capability 的 §3.1 re-anchor 范围；PR-0 spec 仅约束 integer nfeLS 字段，不约束 derived ratio 数字；已 deferred 到 PR-A
  - `openspec/changes/p8tune-spgmr-maxl/proposal.md` L21 "3-way equivalence" 与 design+spec+tasks "4-way" 微弱 drift；proposal text 在 spec deltas land 后非 normative；非阻塞，deferred
  - Invariant Matrix 未在 PR-0 添加；per D0:48-50 explicitly defer 到 PR-C prep（PR-0/A/B/D/E 都不改 CVODE source，只 PR-C 改）

### 维护者关注点

- PR-0 是 epic 第 1 个 PR；后续 5 个 PR（#364 PR-A / #365 PR-B / #366 PR-C / #367 PR-D / #368 PR-E）会按依赖图顺序展开
- 一旦 PR-0 merge 到 `baseline/p8tune`，PR-A 即可从 `baseline/p8tune` 分支启动（依赖 PR-0 corrected floors）
- Merge 后 close issue #363 via PR description `Closes #363`
- 无额外人工关注点

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
