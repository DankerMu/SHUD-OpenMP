## Summary

p8pre-spike Step 2 PR-D pre-flight slice. Verifies SUNDIALS 6.0.0 preconditioner API exists in `cvode.h` / `cvode_ls.h` headers (4 grep PASS + REJECT distinction) + forks SHUD `openmp-baseline-p8pre` working branch from `7a1dc8f` (P1e ship pin) with forward-only descendant guarantee. Closes #344, refs #338.

### Deliverables

| 文件 / 上游状态 | 用途 |
|---|---|
| `docs/p8pre/api_verification.md` (NEW, 169 行) | Lightweight evidence log (6 §s); 4-grep PASS table + REJECT distinction + SHUD fork command transcript + forward-only descendant strict criteria |
| **SHUD upstream NEW branch** `openmp-baseline-p8pre` | origin SHA = `7a1dc8f` (P1e ship pin exactly) — long-lived working line for Step 2 spike commits |

### Acceptance Criteria (全 PASS @ head `01fd43f`)

| AC | 实测 |
|---|---|
| 3 grep PASS: CVodeSetLSetupFrequency / CVodeSetJacEvalFrequency / CVLsPrec(Setup\|Solve)Fn | ✓ 4/4 (typedefs are 2 separate symbols) |
| REJECT CVodeSetMaxConvFails (not setup frequency) | ✓ cvode.h:133 documented as distinct subsystem |
| openmp-baseline-p8pre branch 创建 + push success **from `7a1dc8f` exactly** | ✓ origin SHA = 7a1dc8f6ea9e5496f516255406ee3563d397959b |
| `git rev-parse openmp-baseline-p8pre == git rev-parse 7a1dc8f` strict | ✓ IDENTICAL |
| `git merge-base openmp-baseline-p8pre 7a1dc8f == git rev-parse 7a1dc8f` strict | ✓ IDENTICAL |
| `docs/p8pre/api_verification.md` 记录 grep + fork SHA + merge-base | ✓ 6 §s |
| SHUD outer pointer unchanged (still 7a1dc8f) | ✓ pointer bump deferred to PR-D impl #345 |
| openspec validate p8pre-spike --strict | ✓ exit 0 |
| `.gitmodules` 未触 (C8 不污染 master) | ✓ |
| SHUD `openmp-baseline` master 未触 | ✓ |

### 4 API grep PASS

| # | Symbol | Header | Line |
|---:|---|---|---:|
| 1 | `CVodeSetLSetupFrequency` | cvode.h | 132 |
| 2 | `CVodeSetJacEvalFrequency` | cvode_ls.h | 91 |
| 3 | `CVLsPrecSetupFn` typedef | cvode_ls.h | 57 |
| 4 | `CVLsPrecSolveFn` typedef | cvode_ls.h | 61 |

### REJECT distinction

`CVodeSetMaxConvFails` (cvode.h:133) 控制 **SUNNonlinSol 非线性 convergence-failure 阈值** — NOT a substitute for `CVodeSetLSetupFrequency` (cvode.h:132) 控制 **CVODE step controller setup 频率**. 两者属不同 SUNDIALS 子系统。

### Forward-only descendant guarantee

Per spec p8precond-zero-identity-spike Scenario "Step 2 forward-only descendant extension" L148-154 + design D6 + CLAUDE.md C8: fork base MUST be `7a1dc8f` exactly (P1e ship pin) so PR-D impl #345 pointer bump is provably a linear descendant. Degenerate case at pre-flight (0 commits ahead) — strict equality holds. Post-PR-D impl, strict rev-parse equality will fail (commits land) but merge-base equality MUST continue to hold.

## Agent Review

- Reviewer agents used: `review-correctness`, `review-integration` (Phase 4 round 1) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED (0 PLAUSIBLE candidates — 2/2 APPROVE 全 0 findings)
- Reviewed head SHA: `01fd43f`
- Review evidence: see this PR's comments — Phase 4 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `compact`; selected risk packs: Documentation + Spec compliance (forward-only descendant) + Legacy-compatibility (SHUD submodule workflow C8)
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking. 多 non-blocking technical notes (carry as observations only)

## Test plan

- [x] 4 API grep verify in SHUD/InstallSundials/include/cvode/{cvode.h, cvode_ls.h}
- [x] SHUD branch fork from 7a1dc8f + push to origin
- [x] Forward-only strict criteria (rev-parse + merge-base both IDENTICAL to 7a1dc8f)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 4 round 1 compact cross-review (2/2 APPROVE)
- [x] Phase 7 independent final review: clean
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [ ] Auto-merge after pre-merge evidence hard-gate
