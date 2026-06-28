## 工作情况说明（Merge 前）

- 关联 Issue：#342
- PR：#354
- 冻结提交：`49a2d51`
- 上游 Epic：#338 (p8pre-spike Step 1)
- 前序 PR：#350 (Step 0 doc-correction merged 2026-06-27) + #352 (Step 1 PR-A prep merged 2026-06-27) + #353 (Step 1 PR-A run merged 2026-06-27)

### 背景与目标

p8pre-spike Step 1 PR-B **aggregator + ROI verdict slice**（precedes #343 PR-C 学术 baseline doc + Step 2 P8-precond-0 spike）。本 PR 完成 3 件事：

1. **写 aggregator script** — `tools/p8pre/aggregate_n8_profile.sh` POSIX bash + awk，从 #341 PR-A run rsync mirror `/tmp/p8pre_n8_profile/` 读 18 cells，per cell parse 15 canonical CVODE keys + 7 timer buckets + 2 extras = 24 metrics
2. **执行 5 类 gate check + verdict 决策**：
   - REJECT typo keys (`nlcf` / `nfevals` / `hcur` / `qcur` / `hin`) — exit 1
   - per (case, N) median over 3 reps × 24 metrics
   - cross-N invariance Δ=0 strict × {nst, nfe, nfeLS, nni, nsetups} × {heihe, heihe_x4} (10 checks) — exit 2
   - absolute baseline anchor × 4 (heihe nst=6698/nfe=6943, heihe_x4 nst=6575/nfe=6741) — exit 3 (Mode C regression)
   - ROI 4-branch tree (a/b/c/d per spec L75-80): r_min/r_max(nfeLS/nfe) 决策
3. **写 verdict doc** — `docs/p8pre/n8_profile_verdict.md` 学术风格 219 行 7 §s（NOT baseline，留 PR-C #343）

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `tools/p8pre/aggregate_n8_profile.sh` (新, 911 行, exec) | POSIX bash + awk aggregator: 5 gate (REJECT typo / invariance Δ=0 / absolute baseline / ROI ratios / branch verdict) + branch letter 双发 stdout + verdict doc |
| `docs/p8pre/n8_profile_verdict.md` (新, 219 行) | Academic-style verdict doc: Abstract + §1 目的 + §2 数据来源 + §3.1-§3.5 stats/invariance/baseline/ratios/verdict header + §4 timer + §5 Step 2 expected + §6 限制 + §7 引用 |

无 SHUD submodule pin bump（保持 `7a1dc8f`），无 SHUD 源码改动，无 `tools/cvode_stats_diff/canonical_15_keys.yaml` 改动，无 master plan 改动，无 CI rule 改动。

### ROI verdict 结果（branch a PROCEED Step 2）

```
case       N   n_reps  nfe_median   nfeLS_median r=nfeLS/nfe nst_median
heihe      1   3       6943         12632        1.819   6698
heihe      4   3       6943         12632        1.819   6698
heihe      8   3       6943         12632        1.819   6698
heihe_x4   1   3       6741         30509        4.526   6575
heihe_x4   4   3       6741         30509        4.526   6575
heihe_x4   8   3       6741         30509        4.526   6575

ROI ratios r = nfeLS/nfe at N=8:
  heihe:    r = 1.819
  heihe_x4: r = 4.526
  r_min = 1.819 ; r_max = 4.526

branch: a (PROCEED — r_min=1.819 >= 1.5)
```

**Step 1 GATE PASS** — r_min ≥ 1.5 阈值满足，可进入 Step 2 P8-precond-0 identity spike。r_max=4.526 高 ratio 提示 heihe_x4 case SPGMR Krylov 子空间 dominant cost，identity preconditioner spike 可验证 SUNDIALS API 接通 + 性能契约。

### 测试与验证

**本地**：
- `bash -n tools/p8pre/aggregate_n8_profile.sh` exit 0
- aggregator 在 /tmp/p8pre_n8_profile/ mirror 跑 exit 0, wall ~1.6s (security-perf reviewer 测)
- 4 absolute baselines + 10 invariance Δ=0 + 6 ROI ratios 全 PASS
- branch letter `a` 双发 stdout + verdict doc §3.5 header
- `openspec validate p8pre-spike --strict --no-interactive` exit 0

**CI**: 5/5 PASS (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (expanded, 4 parallel reviewers, **全 APPROVE 0 findings**):
  - `review-spec-compliance`: 8 spec scenarios 全 PASS + 2 non-blocking notes
  - `review-correctness`: numerics + REJECT regex + branch tree + verdict doc 全 OK + 多个 non-blocking technical notes
  - `review-integration`: 11 integration items 全 OK + branch letter forward compat 验 + ADR-0003 hint
  - `review-security-perf`: 0 injection vector + 1.6s wall + perf microopt 建议
- **Phase 4.5 verifier**: SKIPPED (0 PLAUSIBLE candidates，4/4 APPROVE 无 finding)
- **Phase 5/6 fix pass**: SKIPPED (cross-review clean)
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，12/12 AC PASS，4-branch tree 数学穷尽性验，oracle integrity PASS，APPROVE merge

### 兼容性、风险与已知限制

- 无 API / 数据格式 / 迁移兼容性影响
- aggregator input schema 与 #341 PR-A run output 严格一致（`profile_B0.yaml` + `cvode_stats.txt` × 18 cells）
- verdict doc 输出格式与 PR-C #343 baseline doc + PR-F #347 gate-4 verdict + #348 PR-G ADR-0003 draft generator 三个 downstream consumer 契约一致：`branch:` token grep-parseable (frontmatter + Abstract + §3.5 三处)
- **forward-known limit**: 90-day case truncation（per CLAUDE.md C7 项目铁律）+ 2-case scope（design D4）— 不构成 ADR-0003 拦截
- branch d (r_max ≥ 3.0 AND r_min < 1.5) precondition 不触发（r_min=1.819 ≥ 1.5），即使 heihe_x4 r=4.526 显著高于 3.0 阈值 — 数学正确，spec L75-80 branch tree 优先级 a→b→d→c default 实施无误

### 维护者关注点

- 无额外关注点。下一步 #343 PR-C：写学术风格 `docs/p8pre/n8_profile_baseline.md` 作 Step 2 gate-4 anchor + master plan §P8-precond-0 prep。
