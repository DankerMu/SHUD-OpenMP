## 工作情况说明（Merge 前）

- 关联 Issue：#347
- PR：#359
- 冻结提交：`75a757c`
- 上游 Epic：#338 (p8pre-spike Step 2 PR-F)
- 前序 PR：#358（PR-E 18-cell server data capture）

### 背景与目标

p8pre-spike **Step 2 PR-F verdict adjudication slice**（依赖 #346 PR-E 18-cell identity spike data）。本 PR 完成 2 件事：

1. **实现 `tools/p8pre/aggregate_identity_spike.sh`** — 569 行 POSIX bash + awk + sha256sum + uv-python numpy aggregator，parse 18 cells × profile_B0.yaml + cvode_stats.txt + .rivqdown.dat，computes per (case, N) wall_median + nps_median + npe_median + ncfn_max + per-cell SHA12 + max_ulp。Evaluates 4 hard-gate + 2 soft-gate per spec L60-130
2. **写 `docs/p8pre/identity_spike_verdict.md`** — 250 行 academic-paper-style 11 sections（YAML + Abstract + §1-§10）per CLAUDE.md user pref + 母本 `docs/p1e/p1e_academic_summary.md`

### 验证 verdict（NO-GO）

**Hard gates**:

| # | Gate | 标准 | 实测 |
|---|---|---|---|
| 1 | Build PASS | server nm 3-symbol | **PASS** server_nm.log 1 hit 每 |
| 2 | Zero conv failure | `ncfn=0` per cell (18/18) | **FAIL** heihe `ncfn=6` (9/9) / heihe_x4 `ncfn=47` (9/9) deterministic |
| 3 | nps + npe accumulation | nps>0 AND npe>0 per cell | **PASS** min_nps=18163 / min_npe=77 全 18 |
| 4 | Wall non-regression | (\|identity - baseline\|/baseline ≤ ε(case)) per (case, N) | **PASS** max delta heihe=2.64% (<10%) / heihe_x4=1.09% (<5%) |

**Soft gates**:

| # | Gate | 标准 | 实测 |
|---|---|---|---|
| 5 | Cross-N tolerance | SHA12 strict OR max_ulp ≤ 1024 | **FAIL** 18/18 strict 不匹配（baseline `a2023ccd2de4` / `b5e4b0a2cf83`）+ fallback max_ulp ≈ 9×10¹⁵ 远超阈值；structural divergence at 5,155 of 214,252 rivqdown positions |
| 6 | Setup overhead | t_precond_setup / wall ≤ 0.05 per cell | **PASS** max ratio = 1.01×10⁻⁷，6 orders below threshold |

**Spike verdict: NO-GO** per spec L74-79 + L106-108：任一 hard gate FAIL（gate 2）+ soft gate 5 FAIL → NO-GO + design D8 PREC_NONE fall-back。

### ADR-0003 recommendation（PR-G #348 scope）

verdict 文档 §7 明确推荐 **NO-GO**：
1. 写 `docs/adr/0003-precond-spike-decision.md` 文档化 NO-GO + rationale
2. Revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE
3. Delete `MD_precond_identity.{h,cpp}`
4. Update master plan §P8-precond.0 with NO-GO outcome
5. Close `baseline/p8pre` 分支
6. Cite jok-mirror SUNDIALS canonical（cvDiurnal_kry.c L716/L760）for spec L26 wording correction (#349 archive scope)

### 本次具体改动

| 文件 | 行数 | 用途 |
|---|---:|---|
| `tools/p8pre/aggregate_identity_spike.sh` (新) | 569 | aggregator script，4 hard-gate + 2 soft-gate logic + uv-python numpy max_ulp fallback |
| `docs/p8pre/identity_spike_verdict.md` (新) | 250 | academic-paper-style verdict doc 11 sections，18×15 raw data table，§7 ADR-0003 NO-GO + 7-action list |

无 SHUD 源码改动，无 `.gitmodules` / `openspec/changes/` / master plan / case_deployment_map / CI rule 改动。SHUD pin `5276167` unchanged。

### 关键技术发现

**Gate 2 FAIL 根本原因**：identity precond P^-1=I 提供零 SPGMR convergence help on stiff Jacobian，propagate 到 CVODE Newton retries。`ncfn=6/47` deterministic 表明这是 reproducible 现象，非 flaky 数值。

**Gate 5 max_ulp ≈ 9×10¹⁵ 根本原因**：5,155 of 214,252 rivqdown positions 显示 structural divergence（zero/non-zero set differs between identity-PREC_LEFT and baseline-PREC_NONE）。不是 pure floating-point reduction order drift，而是 CVODE step controller 的整数级 step path divergence (nst heihe 6698→6599 / heihe_x4 6575→6569)。

**Gate 4 wall PASS 但 Gate 2 FAIL 的 ROI implication**：identity precond 数值 overhead ≈ 0（gate 6 6 orders 余量）；但 zero convergence improvement（gate 2 FAIL）说明 future real preconditioner 必须 demonstrably reduce ncfn AND fit in 5% setup budget。spike 验证了 plumbing 但 ROI 假设 H2 + H3 都 FAIL。

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (expanded, 4 parallel reviewers @ `75a757c`, **全 APPROVE 0 findings**):
  - `review-spec-compliance`: 12/12 checklist PASS（gate logic 与 spec L60-130 字符级一致 + baseline + epsilon + SHA12 anchors 全对）
  - `review-correctness`: 0 findings + 2 Suggestions non-blocking（max_ulp 精度范围 + 5155/214252 来自 manual numpy 而非 script，但 max_ulp 单标量 decisive）+ Praise gate 1/2/3/4/5 logic 全对
  - `review-documentation`: 0 findings + 2 Suggestions non-blocking（§3 列序 cosmetic + §7 7-action list 未显式 inline jok-mirror，但 §6.2 + Refs 有）+ Praise H1/H2/H3 formal + 决定性 verdict 措辞 + §6.2 ROI quantification 满足 spec L122
  - `review-integration`: 0 findings + 5 non-blocking notes（`.review-evidence/` gitignore 延后 PR-G + compare_snapshot raw-double 加固延后 future tools epic + median sort 已 cite + §7 7-action 高度 consumption-ready + §6.2 quotable 进 master plan/ADR）
- **Phase 4.5 verifier**: SKIPPED (0 PLAUSIBLE candidates)
- **Phase 5/6**: SKIPPED (cross-review clean)
- **Phase 7 final review** (Gap Sweep): **clean**，9/9 AC PASS + oracle integrity PASS + CI 5/5 PASS + mergeStateStatus=MERGEABLE + PR-G readiness PASS + aggregator reproducibility PASS

### 兼容性、风险与已知限制

- 无 API 兼容性破坏（pure addition: 1 script + 1 doc）
- SHUD upstream `openmp-baseline` master 未触（C8 不污染）
- SHUD pin 5276167 unchanged（PR-G NO-GO tail 才 revert PREC_LEFT → PREC_NONE）
- **forward note (5 cosmetic Suggestions)** carried to PR-G #348：
  - `.review-evidence/` gitignore patch
  - §3 column ordering cosmetic
  - §7 jok-mirror inline cite cosmetic
  - max_ulp precision range cosmetic
  - compare_snapshot raw-double hardening（future tools epic，非 PR-G scope）

### 维护者关注点

- 无额外关注点。下一步 **#348 PR-G**：
  - 写 `docs/adr/0003-precond-spike-decision.md` academic-style ADR
  - 写 `docs/p8pre/p8pre_summary.md` academic-paper-style capstone per CLAUDE.md user pref
  - Update master plan §P8-precond.0 with NO-GO outcome + ROI quantification
  - 执行 design D8 NO-GO tail：
    1. Revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE
    2. Delete `MD_precond_identity.{h,cpp}` from SHUD on openmp-baseline-p8pre
    3. Bump outer pointer to new SHUD HEAD (forward-only descendant of 5276167)
  - Cite jok-mirror canonical（PR-D forward carry）
  - 收 PR-F 5 cosmetic Suggestions
