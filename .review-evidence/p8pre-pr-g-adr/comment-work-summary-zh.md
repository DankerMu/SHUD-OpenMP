## 工作情况说明（Merge 前）

- 关联 Issue：#348
- PR：#360
- 冻结提交：`06746b8`
- 上游 Epic：#338 (p8pre-spike Step 3 PR-G 顶点 capstone)
- 前序 PR：#359（PR-F 4-hard + 2-soft gate verdict adjudicator）

### 背景与目标

p8pre-spike **Step 3 PR-G doc-only capstone 切片**。基于 PR-F #359 verdict NO-GO 决议, 实现 6 个 doc 文件 + 5 cosmetic Suggestion 收口（forward-debt 推 #349 archive）。issue 明确"不动 SHUD 源码"——design D8 fall-back (revert `cvode_config.cpp:259` + delete `MD_precond_identity.{h,cpp}` + bump SHUD pointer + close `baseline/p8pre`) 执行**推 #349 或 separate cleanup PR**, 本 PR 仅 doc。

### 6 文件落地

| 文件 | 行数 | 性质 |
|---|---:|---|
| `docs/adr/0003-precond-spike-decision.md` (NEW) | 175 | ADR template per 0002-solver-path.md (Status/Date/Deciders/Owner/Tags/Context/Decision/Consequences/References 完整 8 section); Decision = **NO-GO option (b)** per spec L74-79 + L106-108; Forward-action recommendations 4 段 (D8 fall-back deferred / P8-precond NOT-to-open / P8-tune.A-D alternatives / spec L26 jok-mirror correction) |
| `docs/p8pre/capstone.md` (NEW) | 395 | epic-level academic-paper-style 10 section (YAML metadata + Abstract + Keywords + §1 Intro 含 H1/H2/H3 formal hypotheses + §2 Related Work + §3 Methodology + §4 Setup + §5 Results + §6 Discussion + §7 Limitations + §8 Conclusion + §9 Future Work + §10 References) per CLAUDE.md user-pref 母本 `docs/p1e/p1e_academic_summary.md` |
| `docs/p8pre_summary.md` (NEW) | 113 | 顶层 engineer-style summary, 含 概览 + Step 1/2 summary + 决策 + 后续 + References |
| `SHUD_openMP_master_plan.md` §P8-precond.0 (EDIT) | +29 | NO-GO outcome + ROI data + 6-gate verdict table + P8-tune.A/B/C/D alternative candidates + Outcome paragraph (design D8 deferred) |
| `docs/case_deployment_map.md` §5.2 (EDIT) | +35 | 18-row identity-spike yaml paths table (heihe + heihe_x4 × N∈{1,4,8} × 3 rep) + JID 9531-9548 + NO-GO archive carve-out note |
| `docs/review-loop-log.jsonl` (APPEND) | +1 | epic-level cross-run summary entry (`kind=epic_summary`, `total_prs=9`, `prs=[341..349]`, `verdict=NO-GO`, `forward_to_future[4]`) |

### ADR-0003 Decision

**NO-GO option (b)** per:
- spec L74-79 strict (任一 hard gate FAIL → spike NO-GO)
- spec L106-108 fall-back (max_ulp ≫ 1024 → fall-back FAIL)
- design D5 ANY hard gate FAIL → NO-GO design path

**Decision rationale 5 points**:
1. Hard gate 2 `ncfn` deterministic floor 6/47 跨 18/18 cells PROVE identity P⁻¹=I 对 SHUD stiff Jacobian SPGMR convergence 完全无作用
2. Soft gate 5 max_ulp ≈ 9×10¹⁵ ≫ 1024 揭示 PREC_LEFT 状态机 inherent cost (额外 N_VLinearSum / N_VScale ops, 5,155/214,252 = 2.4% positions structural divergence)
3. Step 1 canonical `nfeLS/nfe = 1.819` ROI window 仍 promising 但 identity stub 无法 unlock — 需要 real preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi) 不同实现路径
4. KISS/YAGNI 禁 dead PREC_LEFT codepath
5. spec L74-79 hard FAIL → formal P8-precond epic NOT-to-be-opened

### 关键技术发现 (epic-value-summary triple)

p8pre-spike epic 价值即使在 NO-GO 下仍有 3 维 anchor 落地:

1. **framework readiness**: SUNDIALS PSetup/PSolve canonical pattern (jok-mirror per `cvDiurnal_kry.c` L716/L760) + Timer instrumentation 已固化; future real-preconditioner candidate (Diagonal / Jacobi / ILU(0) / block-Jacobi physics-based) 可直接复用骨架
2. **ROI ceiling database**: `ncfn` floor 6/47 (Step 2 spike) + Step 1 PREC_NONE canonical `nfeLS/nfe` 比值 `{1.819, 4.526}` + S5 structural drift baseline 5,155/214,252 positions 是任何 future iterative-solver tuning experiment 的 reproducible anchor
3. **Negative result 形式化记录**: avoids future epic 重复试错 identity 路径

### Review 与修复闭环

- **Phase 4 round 1 expanded cross-review** (4 reviewers @ `8cf527c`): 1 Critical (r_min 1.811 vs 1.819 attribution in capstone+summary) + 2 Warnings (r_min drift + ADR PR-letter binding inconsistency #340 vs canonical #341)
- **Phase 5/6 round 1 fix** @ `a149c5f`: 9 surgical edits (7 substitutions 1.811→1.819 capstone+summary + capstone L167 NB 三组数字 enumeration + ADR §References block rewrite to canonical mapping)
- **Phase 6.5 round 2 delta review** (review-documentation @ `a149c5f`): caught 3 more Critical (我漏 ADR §Step 1 L21 + Rationale L58 + Consequences L69 仍 cited 1.811)
- **Phase 6 round 3 fix** @ `29a6a06`: 3 surgical ADR edits (L20-23 + L58 + L69) + 完整 Step 1 vs Step 2 attribution split
- **Phase 7 Gap Sweep** (phase-7-final-review @ `29a6a06`): caught 1 more Warning (master plan §P8-precond.0 L2383 同 pattern, 结构性 §Step 2 header 提供 context 但缺 explicit attribution NB)
- **Phase 6 round 3 follow-up fix** @ `06746b8`: 1 surgical master plan edit L2383 + 完整 attribution NB
- **Phase 7 post-fix re-audit** @ `06746b8`: Clean (1.811 仅 3 处 explicit Step 2 NB; 1.819 ≥ 11 处 explicit Step 1 canonical; CI 5/5 PASS; oracle integrity PASS)

### Cross-doc canonical agreement (post @ 06746b8)

**Step 1 baseline `r_min = 1.819`** (canonical per `docs/p8pre/n8_profile_verdict.md` §5 PREC_NONE) 一致出现 11+ 处:
- `docs/p8pre/capstone.md` L24, L92, L165, L228, L290 (5×)
- `docs/p8pre_summary.md` L20, L45, L70 (3×)
- `docs/adr/0003-precond-spike-decision.md` §Step 1 L21 + Rationale §3 L58 + Consequences Positive L69 (3×)
- `SHUD_openMP_master_plan.md` §P8-precond.0 L2383 (1×)

**Step 2 PREC_LEFT identity-precond 实测 `1.811`** 仅出现 3 处 explicit Step 2 attribution NB:
- `docs/p8pre/capstone.md` L167 NB
- `docs/adr/0003-precond-spike-decision.md` L23 NB
- `SHUD_openMP_master_plan.md` L2383 NB

### 兼容性、风险与已知限制

- 无 API 兼容性破坏 (pure doc-only, 6 files)
- SHUD upstream `openmp-baseline` master 未触 (C8 不污染)
- SHUD pin `5276167` unchanged (design D8 revert deferred 推 #349 / cleanup PR)
- `openspec/changes/p8pre-spike/specs/*` 未触 (留 #349 archive)
- baseline/p8pre branch 未关 (留 #349 / cleanup PR)
- 5 cosmetic Suggestions carry-forward 到 #349:
  - `.review-evidence/` gitignore patch
  - §3 column ordering cosmetic
  - §7 jok-mirror inline cite cosmetic
  - max_ulp precision range cosmetic
  - compare_snapshot raw-double hardening (future tools epic, 非 PR-G scope)

### 维护者关注点

- 无额外关注点。下一步 **#349**:
  - openspec archive `p8pre-spike` change
  - spec L26 wording correction → cite jok-mirror canonical (SUNDIALS `cvDiurnal_kry.c` L716/L760)
  - 收 5 cosmetic Suggestions
  - 执行 design D8 NO-GO tail (1 revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE / 2 delete `MD_precond_identity.{h,cpp}` from SHUD on openmp-baseline-p8pre branch / 3 bump outer pointer / 4 close baseline/p8pre)
  - Epic #338 close
