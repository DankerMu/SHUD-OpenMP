## Summary

p8pre-spike Step 1 PR-B slice. Aggregator script + ROI verdict doc consume the 18-cell rsync mirror from #341 PR-A run, decide 4-branch verdict per spec, emit `branch: a (PROCEED Step 2)`. Closes #342, refs #338.

### Deliverables

| 文件 | 用途 |
|---|---|
| `tools/p8pre/aggregate_n8_profile.sh` (NEW, 911 行) | POSIX bash + awk aggregator: parse 18 × profile_B0.yaml + cvode_stats.txt; REJECT typo; per (case, N) median × 24 metrics; cross-N invariance Δ=0 strict × 5 keys × 2 cases; absolute baseline anchor × 4 values; 4-branch verdict tree; explicit branch letter emit |
| `docs/p8pre/n8_profile_verdict.md` (NEW, 219 行) | Academic-style verdict doc; 7 §s; branch letter `a` in YAML metadata + Abstract + §3.5 header |

### Acceptance Criteria (全 PASS @ head `49a2d51`)

| AC | 实测 |
|---|---|
| aggregator 解析 18 cell × 24 metrics | ✓ |
| aggregator REJECT typo keys | ✓ exit 1 on nlcf/nfevals/hcur/qcur/hin |
| cross-N invariance Δ=0 strict × 5 keys × 2 cases | ✓ 10/10 PASS |
| absolute baseline 4 anchor: heihe {nst=6698, nfe=6943}, heihe_x4 {nst=6575, nfe=6741} | ✓ exact integer match |
| ROI ratios per (case, N): 6 values | ✓ heihe r=1.819 × 3, heihe_x4 r=4.526 × 3 |
| Branch letter emit stdout AND verdict doc §3 header | ✓ `branch: a` 三处 (frontmatter + Abstract + §3.5) |
| 4-branch tree exhaustive application | ✓ branch a (r_min=1.819 ≥ 1.5) PROCEED |
| openspec validate strict | ✓ exit 0 |
| SHUD pin 7a1dc8f unchanged | ✓ |
| Verdict doc internal consistency | ✓ §3.1 ↔ §3.4 ↔ §3.5 cohere |

### ROI verdict 概要

```
case       N   nfe_median  nfeLS_median  r=nfeLS/nfe  nst_median
heihe      1   6943        12632         1.819        6698
heihe      4   6943        12632         1.819        6698
heihe      8   6943        12632         1.819        6698
heihe_x4   1   6741        30509         4.526        6575
heihe_x4   4   6741        30509         4.526        6575
heihe_x4   8   6741        30509         4.526        6575

r_min = 1.819 (heihe)
r_max = 4.526 (heihe_x4)

branch: a (PROCEED — r_min=1.819 >= 1.5)
```

### Step 1 GATE → PROCEED

`r_min = 1.819 ≥ 1.5` → **Step 1 GATE PASS**。Next: PR-C (#343 academic-style baseline doc) → Step 2 P8-precond-0 spike (#344 / #345 / #346 / #347 / ADR-0003).

`r_max = 4.526 ≥ 3.0` 但 branch d 不触发（precondition `r_min < 1.5` 不满足，spec L75-80 4-branch tree precedence a→b→d→c default）。Heihe_x4 高 r 提示 SPGMR Krylov 子空间 dominant cost — identity preconditioner spike 仍是合理 Step 2 起点验证 SUNDIALS API 接通+性能契约，不构成 ADR-0003 NO-GO 论据。

### Sanity column doc fix

§3.4 sanity 列 `r-1 vs nli/nni` 修正为 `nfeLS = nli`（SUNDIALS CVLS SPGMR + FD-Jvp identity，而非 textbook `nfeLS = nfe + nli`）。数据吻合：heihe `nfeLS=12632 == nli=12632` strict，heihe_x4 `nfeLS=30509 == nli=30509` strict。

## Agent Review

- Reviewer agents used: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf` (Phase 4 round 1) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED (0 PLAUSIBLE candidates — 4/4 APPROVE 全 0 findings)
- Reviewed head SHA: `49a2d51`
- Review evidence: see this PR's comments — Phase 4 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Public API/CLI + File IO + Numerical stability + Spec compliance + Documentation
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking. 多 non-blocking technical/perf-microopt notes (carry as observations only)

## Test plan

- [x] aggregator script `bash -n` PASS
- [x] aggregator exit 0 on /tmp/p8pre_n8_profile/ mirror
- [x] 4 absolute baselines + 10 invariance Δ=0 + 6 ROI ratios verified
- [x] branch letter 三发 (stdout + verdict doc frontmatter + Abstract + §3.5 header)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 4 round 1 expanded cross-review (4 reviewers 全 APPROVE)
- [x] Phase 7 independent final review: clean, APPROVE merge
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [ ] Auto-merge after pre-merge evidence hard-gate
