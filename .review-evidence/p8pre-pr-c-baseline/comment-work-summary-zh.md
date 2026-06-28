## 工作情况说明（Merge 前）

- 关联 Issue：#343
- PR：#355
- 冻结提交：`43bd6f2`（tracked；Phase 6 openspec/changes fix 是 working-tree-only per `.gitignore:13`）
- 上游 Epic：#338 (p8pre-spike Step 1 capstone)
- 前序 PR：#350 (Step 0) + #352 (PR-A prep) + #353 (PR-A run) + #354 (PR-B aggregator branch a verdict)

### 背景与目标

p8pre-spike Step 1 **capstone PR-C slice**（Step 1 末尾，定 Step 2 gate-4 baseline anchor，precedes #344-#347 Step 2 P8-precond-0 identity spike）。本 PR 完成 3 件事：

1. **写 academic-paper-style baseline doc** — `docs/p8pre/n8_profile_baseline.md` 367 行 10 §s（母本 `docs/p1e/p1e_academic_summary.md`）：YAML metadata + Abstract + §1 Introduction (H1/H2/H3 形式化假设) + §2 Related Work + §3 Methodology + §4 Experimental Setup + §5 Results (§5.1 raw data + §5.2 invariance + §5.3 baseline anchor + §5.4 ROI ratios + §5.5 branch verdict) + §6 Discussion + §7 Limitations & Threats to Validity + §8 Conclusion + §9 Future Work + §10 References
2. **archive gate-4 baseline** — §5.1 6 行 `wall_step1_baseline_median(case, N)` table（heihe 140.797/95.734/89.732s, heihe_x4 1412.895/849.704/743.552s 中位数）为 PR-F #347 hard-gate 4 wall non-regression 的 single source of truth
3. **update 2 living docs** — `docs/case_deployment_map.md` §5.1 18-row table 補 N=8 yaml paths + `SHUD_openMP_master_plan.md` §P8-precond.0 prep cross-ref block @ L2356（指向 3 个 p8pre docs + branch a PROCEED）

并修复 Phase 4.5 round 1 暴露的 1 个 CONFIRMED candidate（详见 review 闭环段）。

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `docs/p8pre/n8_profile_baseline.md` (新, 367 行) | Academic-paper-style capstone doc, 10 § + Abstract + H1/H2/H3 hypothesis verification, §5.1 IS Step 2 gate-4 anchor |
| `docs/case_deployment_map.md` (修, +27 行) | §5.1 18-row table archive N=8 Mode C profile yaml paths under `/tmp/p8pre_n8_profile/` |
| `SHUD_openMP_master_plan.md` (修, +12 行) | §P8-precond.0 prep cross-ref block at L2356 (BEFORE §P8-precond.1), 指向 3 p8pre docs + Step 1 GATE PASS + branch a |

**Phase 6 working-tree-only fix**: 5 处 §3 → §5.1 substitution 跨 4 个 gitignored openspec/changes/ 文件（`tasks.md` L41+L77 + `specs/n8-mode-c-profile-recheck/spec.md` L17+L113 + `specs/p8precond-zero-identity-spike/spec.md` L90 + `design.md` L26+L107），per `.gitignore:13` "OpenSpec transient changes" policy 不在 tracked diff 中 — 同 PR #353 cand-02 precedent，archive 至 persistent `openspec/specs/` at #349。

无 SHUD submodule pin bump（保持 `7a1dc8f`），无 SHUD 源码改动，无 CI rule 改动。

### Step 1 GATE 再确认（branch a PROCEED Step 2）

| 项 | 值 |
|---|---|
| r_min (heihe N=8) | 1.819 |
| r_max (heihe_x4 N=8) | 4.526 |
| Branch | **a (PROCEED Step 2)** |
| 4 absolute baseline anchors | PASS (heihe nst=6698/nfe=6943 + heihe_x4 nst=6575/nfe=6741) |
| 10 invariance Δ=0 checks | PASS (N=1=N=4=N=8 strict per case per key) |
| ADR-0003 NO-GO branch | NOT taken (branch a path) |

### Wall median table (gate-4 baseline anchor, §5.1)

```
case        N    wall_median(s)    nst_median    nfe_median
heihe       1    140.797           6698          6943
heihe       4     95.734           6698          6943
heihe       8     89.732           6698          6943
heihe_x4    1   1412.895           6575          6741
heihe_x4    4    849.704           6575          6741
heihe_x4    8    743.552           6575          6741
```

来源：`/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/profile_B0.yaml` `extras.t_wall_total`，sorted-middle median per (case, N)。PR-F #347 gate-4 直接 cite 此列。

### 假设验证（H1/H2/H3）

- **H1**: Mode C build (SHUD_ENABLE_OPENMP_RHS=1 + SHUD_ENABLE_PROFILE=1) preserves CVODE numerical invariance → **VERIFIED**（10/10 Δ=0 per §5.2）
- **H2**: ROI ratio `r = nfeLS/nfe ≥ 1.5` indicates preconditioner introduction cost-effective → **VERIFIED**（r_min=1.819 per §5.4）
- **H3**: heihe + heihe_x4 dual-case spans 6.3× problem size scale + 8-10× wall scale，覆盖 small + large → **VERIFIED**（per §5.1）

### 测试与验证

**本地**：
- `openspec validate p8pre-spike --strict --no-interactive` exit 0
- `grep -c "wall_median" docs/p8pre/n8_profile_baseline.md` = 9 ≥ 6
- `grep -c "branch a" docs/p8pre/n8_profile_baseline.md` ≥ 2（YAML metadata + Abstract + §5.5/§8/References 总 5 处）
- All 13/14 cited file paths resolve（1 forward ref to PR-F #347 `aggregate_identity_spike.sh` 仍未存在，预期 forward dep）

**CI**: 5/5 PASS (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (compact, 2 parallel reviewers):
  - `review-correctness`: APPROVE，0 findings + 2 non-blocking notes（date front-matter UTC 跨日 + §5.1 schema 与 case_deployment_map §5 列名不同 justified）
  - `review-integration`: APPROVE w/ 1 Warning candidate（cand-01: openspec/changes/ 引用 baseline doc "§3 raw data table" 但实际 §5.1 academic Results 结构，PR-F #347 verbatim 读会找错位置）
- **Phase 4.5 verifier**: cand-01 verdict = **CONFIRMED + merge-blocking**（compact precision-bias + trivial 5-line sed fix in-scope，mirrors PR #353 cand-02 precedent）
- **Phase 6 fix pass** (orchestrator-direct on gitignored openspec/changes/):
  - 5 处 §3 → §5.1 substitution 跨 4 文件（per drift location verified）
  - openspec validate strict 仍 PASS
  - 同 PR #353 cand-02 precedent: gitignored fix 在 working tree 生效，后续 p8pre-spike implementer 从 my working tree fire → 不会误读
- **Phase 6.5 round 2 cross-review** (focused integration):
  - `review-integration` round 2: APPROVE，cand-01 **RESOLVED**，0 new findings
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，11/11 AC PASS，oracle integrity PASS，APPROVE merge

### 兼容性、风险与已知限制

- 无 API / 数据格式 / 迁移兼容性影响（doc PR）
- §5.1 wall_median table 是 Step 2 PR-F #347 gate-4 single source of truth — 数值与 PR-A run /tmp mirror profile_B0.yaml extras.t_wall_total 严格一致
- Phase 6 openspec/changes/ fix 在 working tree only (`.gitignore:13` transient policy)，archive at #349
- **forward-known limit**: 90-day case truncation (CLAUDE.md C7) + 2-case scope (design D4) — Step 2 同样使用，不构成 ADR-0003 拦截
- **expected forward dep**: `tools/p8pre/aggregate_identity_spike.sh` (PR-F #347 deliverable) referenced in §5.1 + §9，目前不存在；ADR-0003 决策时 PR-F 将认领

### 维护者关注点

- 无额外关注点。下一步 **Step 2 P8-precond-0 spike** 启动：
  - #344 PR-D pre-flight: SUNDIALS API verify (CVodeSetLSetupFrequency in cvode.h:132, CVodeSetJacEvalFrequency in cvode_ls.h:91)
  - #345 PR-D impl: MD_precond_identity.{h,cpp} + cvode_config.cpp:259 PREC_NONE → PREC_LEFT + CVodeSetPreconditioner wire + SHUD `openmp-baseline-p8pre` fork from `7a1dc8f` + Mac sanity
  - #346 PR-E: server 18-cell identity spike (~6h compute wait)
  - #347 PR-F: aggregator + 4 hard-gate + 2 soft-gate verdict
  - **Step 2 HARD GATE**: 任 FAIL → ADR-0003 NO-GO
  - #348 PR-G: ADR-0003 draft + p8pre_summary.md
  - #349 openspec archive + Epic #338 close
