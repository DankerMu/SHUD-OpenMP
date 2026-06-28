## Summary

p8pre-spike Step 3 PR-G doc-only capstone. Documents Step 2 spike NO-GO verdict + design D8 PREC_NONE fall-back recommendation (NOT executed; deferred to #349 archive or separate cleanup PR per issue 不动 SHUD 源码 Out-of-Scope). 6 doc files. Closes #348, refs #338.

### Deliverables

| 文件 | 行数 | 用途 |
|---|---:|---|
| `docs/adr/0003-precond-spike-decision.md` (NEW) | 175 | ADR template per 0002-solver-path.md structure; Decision = NO-GO option (b); cites PR-F #359 verdict gate 2 ncfn FAIL + gate 5 max_ulp ≈ 9×10¹⁵ FAIL |
| `SHUD_openMP_master_plan.md` §P8-precond.0 (EDIT) | +29 | NO-GO outcome + ROI data (Step 1 canonical nfeLS/nfe=1.819 + Step 2 PREC_LEFT 1.811 with explicit attribution NB) + 6-gate verdict table + P8-tune.A/B/C/D candidates + Outcome para |
| `docs/p8pre_summary.md` (NEW) | 113 | 顶层 engineer-style summary |
| `docs/p8pre/capstone.md` (NEW) | 395 | epic-level academic-paper-style per CLAUDE.md user pref + docs/p1e/p1e_academic_summary.md mother template |
| `docs/review-loop-log.jsonl` (APPEND) | +1 | epic-level cross-run summary entry (kind=epic_summary, total_prs=9) |
| `docs/case_deployment_map.md` §5.2 (EDIT) | +35 | 18-row identity-spike yaml paths table + NO-GO archive note |

### ADR-0003 Decision

**NO-GO option (b)** per spec L74-79 strict + L106-108 fall-back + design D5 NO-GO path. Decision rationale 5 points:

1. Hard gate 2 deterministic `ncfn` floor 6/47 跨 18/18 cells PROVE identity P⁻¹=I 对 SHUD stiff Jacobian zero SPGMR convergence 加速
2. Soft gate 5 max_ulp ≈ 9×10¹⁵ ≫ 1024 揭示 PREC_LEFT 状态机 inherent cost (5,155/214,252 = 2.4% positions structural divergence)
3. Step 1 canonical `nfeLS/nfe = 1.819` ROI window 仍 promising 但 identity stub 无法 unlock — 需要 real preconditioner candidate
4. KISS/YAGNI 禁 dead PREC_LEFT codepath
5. spec L74-79 hard FAIL → formal P8-precond epic NOT-to-be-opened

### Forward action recommendations (from ADR-0003 §Consequences §Forward action)

1. Design D8 PREC_NONE fall-back (deferred to #349 archive OR separate cleanup PR):
   - Revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE
   - Delete `MD_precond_identity.{h,cpp}`
   - Bump outer SHUD pointer to forward-only descendant of 5276167
   - Close `baseline/p8pre` branch
2. P8-tune alternative epic candidates (P8-tune.A/B/C/D)
3. spec L26 wording correction (jok-mirror canonical, cite SUNDIALS cvDiurnal_kry.c L716/L760) → #349 archive scope
4. compare_snapshot raw-double hardening → future tools epic

### Cross-doc canonical agreement post-fix @ 06746b8

**Step 1 baseline `r_min = 1.819`** (canonical per `docs/p8pre/n8_profile_verdict.md` §5 PREC_NONE) consistent across:
- capstone L24, L92, L165, L228, L290 + p8pre_summary L20, L45, L70 + ADR §Step 1 L21 + Rationale §3 L58 + Consequences Positive L69 + master plan §P8-precond.0 L2383

**Step 2 PREC_LEFT identity-precond ratio `1.811`** cited only in 3 explicit Step 2 attribution NBs (capstone L167 NB + ADR L23 NB + master plan L2383 NB).

### Acceptance Criteria (PASS @ head `06746b8`)

| AC | 实测 |
|---|---|
| `docs/adr/0003-precond-spike-decision.md` 完成 (Status / Decision / Consequences / References齐全) | ✓ 175 lines, 8 sections per ADR-0002 template |
| ADR-0003 Decision = NO-GO option (b) 明确 | ✓ L53 |
| master plan §P8-precond.0 prep section 更新含 ROI 实测 + gate verdict + 工程量重估 | ✓ +29 lines + L2383 NB |
| `docs/p8pre_summary.md` (顶层) | ✓ 113 lines engineer-style |
| `docs/p8pre/capstone.md` (academic) | ✓ 395 lines academic-paper-style per CLAUDE.md user pref |
| `docs/review-loop-log.jsonl` epic-level entry append | ✓ +1 line JSON validated |
| `docs/case_deployment_map.md` §5 18 cell yaml paths 补完 | ✓ §5.2 +35 lines |
| openspec validate strict | ✓ exit 0 |
| SHUD pin 5276167 unchanged | ✓ no SHUD source change (revert deferred) |
| openspec/changes/ untouched | ✓ (留 #349) |

## Agent Review

- Reviewer agents used: 4 parallel Phase 4 round 1 (review-spec-compliance, review-correctness, review-documentation, review-integration) + 1 Phase 6.5 round 2 delta (review-documentation) + 1 Phase 7 final review (phase-7-final-review)
- Reviewed head SHAs: 8cf527c (round 1) → a149c5f (round 2) → 29a6a06 (round 3) → 06746b8 (final post-fix)
- Phase 4.5 verifier: SKIPPED (self-evident factual data lookup vs source-of-truth `docs/p8pre/n8_profile_verdict.md` §5)
- 3 rounds of review/fix cycle: r1 caught 1 Critical + 2 Warnings → fix → r2 caught 3 more Critical (my fix incomplete) → fix → Phase 7 caught 1 more Warning → fix → r3 post-fix clean
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Documentation + Spec compliance + Numerical reasoning

## Test plan

- [x] 6 doc files written/edited per issue checklist
- [x] ADR-0003 Decision = NO-GO option (b)
- [x] Master plan §P8-precond.0 has ROI + 6-gate verdict + P8-tune alternatives + L2383 Step 2 attribution NB
- [x] review-loop-log epic_summary JSON validated
- [x] case_deployment_map §5.2 18-row yaml paths table
- [x] openspec validate p8pre-spike --strict --no-interactive exit 0
- [x] Phase 4 round 1 expanded cross-review (1 Critical + 2 Warnings confirmed)
- [x] Phase 5/6 round 1 fix @ a149c5f (9 edits)
- [x] Phase 6.5 round 2 delta review (3 more Critical caught)
- [x] Phase 6 round 3 fix @ 29a6a06 (3 ADR edits)
- [x] Phase 7 final review (1 Warning caught on master plan L2383)
- [x] Phase 6 round 3 follow-up fix @ 06746b8 (1 master plan edit)
- [x] Phase 7 post-fix re-audit clean
- [x] CI 5/5 PASS @ 06746b8 (setup, tools-tests, build-and-compare keliya, asan-ubsan keliya + qhh)
- [ ] Manual merge after pre-merge evidence hard-gate
