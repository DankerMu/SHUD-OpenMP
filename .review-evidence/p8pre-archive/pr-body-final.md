## Summary

p8pre-spike Step 4 PR-archive (epic closeout, doc-only). Closes #349, refs #338. After merge: orchestrator closes Epic #338 + deletes `baseline/p8pre` + deletes SHUD `openmp-baseline-p8pre` per ADR-0003 NO-GO option (b).

### Deliverables

| 文件 | 变化 |
|---|---|
| `openspec/glossary.md` (EDIT, +36 lines, 252→288) | 7 canonical terms (p8pre-spike / identity preconditioner / PREC_LEFT / nfeLS/nfe ratio / CVodeSetLSetupFrequency / .p1e-i-runs/ / ncfl) + ADR-0003 cross-ref entry |
| `openspec/specs/n8-mode-c-profile-recheck/spec.md` (NEW PROMOTED) | 6 reqs PROMOTED from p8pre-spike change |
| `openspec/specs/p8precond-zero-identity-spike/spec.md` (NEW PROMOTED) | 6 reqs PROMOTED + L29-30 jok-mirror canonical cite (SUNDIALS cvDiurnal_kry.c L716/L760) + L29 wording fix `*jcurPtr = jok ? SUNFALSE : SUNTRUE` matching SHUD impl L61 |
| `openspec/changes/archive/2026-06-27-p8pre-spike/` (MOVED, gitignored) | full change dir archived per openspec convention |

### Key technical finding: spec L29 jok-mirror correction

Implementer's initial spec L29 wording said setting `*jcurPtr = SUNFALSE` unconditionally — contradicts the actual SHUD impl at `MD_precond_identity.cpp:61` which uses `*jcurPtr = jok ? SUNFALSE : SUNTRUE` (jok-mirror canonical pattern per SUNDIALS `cvDiurnal_kry.c` L716/L760).

The unconditional SUNFALSE variant produces `npe=0` and violates gate 3 "nps and npe accumulation" — verified empirically in PR-D #357 where implementer chose jok-mirror over spec literal text after discovering gate 3 FAIL with the literal interpretation.

Round 1 fix at `6eb1a27` corrects spec L29 to accurately reflect canonical jok-mirror code + why it matters + reference to PR-D #357 empirical verification. **PR-D #357 forward-debt F-R2-3 CLOSED**.

### Acceptance Criteria (PASS @ head `6eb1a27`)

| AC | 实测 |
|---|---|
| openspec validate strict exit 0 | ✓ both promoted specs PASS per `openspec validate --specs --strict` |
| 12 reqs PROMOTE 成功 (6+6) | ✓ verified via grep `### Requirement:` |
| glossary +7 canonical terms + ADR-0003 cross-ref | ✓ L258/L262/L266/L270/L274/L278/L282/L286 |
| change directory moved to `openspec/changes/archive/` | ✓ openspec archive command exit 0 + dir `2026-06-27-p8pre-spike/` present |
| spec L26 jok-mirror canonical cite (PR-D #357 forward-debt) | ✓ L29-30 of promoted spec + L29 wording matches SHUD impl L61 |
| SHUD pin 5276167 unchanged | ✓ no SHUD source change in this PR |
| docs/adr/0003 + docs/p8pre/* + master plan untouched | ✓ all empty diff |

### Out of Scope (orchestrator-only post-merge actions per ADR-0003 NO-GO)

- Epic #338 close (post-merge orchestrator action)
- baseline/p8pre branch deletion (orchestrator-only, per ADR-0003 §Forward action §1.7)
- SHUD openmp-baseline-p8pre branch deletion (orchestrator-only)
- Design D8 fall-back (revert cvode_config.cpp + delete MD_precond_identity.{h,cpp}) — separate cleanup PR scope

### Forward-debt status

- **CLOSED in this PR**: PR-D #357 F-R2-3 (jok-mirror canonical cite) → spec L29-30 corrected
- **Carried to future tools epic** (per PR-G 5 cosmetic Suggestions): `.review-evidence/` gitignore patch + §3 column ordering cosmetic + max_ulp precision range cosmetic + compare_snapshot raw-double hardening
- **Carried to separate cleanup PR**: Design D8 PREC_NONE fall-back execution (SHUD source revert + outer pointer bump)

## Agent Review

- Reviewer agent used: `review-archive-closeout` (compact fixture single-reviewer combined spec-compliance + correctness + documentation lenses)
- Reviewed head SHA: `6eb1a27`
- Review evidence: this PR's comments — Phase 4 review summary + Chinese work summary
- OpenSpec change: `p8pre-spike` (archived); fixture level: `compact`; selected risk packs: Archive integrity + Spec compliance + Documentation
- Key findings: 0 CONFIRMED, 0 PLAUSIBLE merge-blocking. Round 1 APPROVE clean.

## Test plan

- [x] glossary +7 terms + ADR-0003 cross-ref
- [x] openspec archive p8pre-spike --yes (12 reqs PROMOTE + dir move)
- [x] post-archive openspec validate --specs --strict exit 0
- [x] spec L29 jok-mirror cite + wording fix matching SHUD impl L61
- [x] Phase 4 round 1 compact cross-review APPROVE 0 findings
- [x] CI 5/5 PASS @ 6eb1a27
- [ ] Manual merge + Epic #338 close + branch deletions
