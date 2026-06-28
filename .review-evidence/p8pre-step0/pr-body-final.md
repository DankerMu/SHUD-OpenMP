## Summary

p8pre-spike Epic #338 Step 0 doc-correction PR (preceding PR-A). Doc-only, no code / no SHUD pin bump.

Closes #339
Refs #338

### 改动

| 文件 | 改动 |
|---|---|
| [docs/p1e/p1e_perf_baseline.md](docs/p1e/p1e_perf_baseline.md) | §3.5 错 15-key 列表 → SHUD canonical (per `tools/cvode_stats_diff/canonical_15_keys.yaml`); §3.5/§7.1/§7.3 archive table `.pr-i-runs/` ×4 → `.p1e-i-runs/` |
| [docs/p8pre/step1_prep.md](docs/p8pre/step1_prep.md) | 新建 — inline 引用 P1e §3.1 wall median + §3.4 nst ladder + §4 跨 doc nfe，作 Step 1 PR-A/B/F + ADR-0003 baseline anchor |

### Acceptance Criteria (全 PASS @ head `5827e23`)

| AC | command | 实测 |
|---|---|---|
| `pr-i-runs` in p1e_perf_baseline.md = 0 | `grep -c pr-i-runs docs/p1e/p1e_perf_baseline.md` | **0** |
| `p1e-i-runs` in p1e_perf_baseline.md > 0 | `grep -c p1e-i-runs docs/p1e/p1e_perf_baseline.md` | **4** |
| `nlcf` in p2a_profile_baseline.md = 0 | `grep -c nlcf docs/p2a/p2a_profile_baseline.md` | **0** (天然) |
| `nlcf` in p1e_perf_baseline.md = 0 | `grep -c nlcf docs/p1e/p1e_perf_baseline.md` | **0** |
| canonical 15-key marker `lenrwLS` 落地 | `grep -c lenrwLS docs/p1e/p1e_perf_baseline.md` | **1** |
| 上游 canonical anchor 未触 | `grep -c p1e-i-runs docs/p1e/p1e_pr_i_strict_omp_verification.md` | **8** ≥ 8 |

### Out-of-scope follow-up

`.pr-i-runs/` drift 在 `docs/p1e/p1e_academic_summary.md:219` + `docs/p1e/p1e_summary.md:113` 仍存在（同 drift 类）。Phase 4.5 verifier 判定 REFUTED-out-of-scope（不在 #339 issue body In-Scope 列表内），已 spawn follow-up task chip。

## Agent Review

- Reviewer agents used: `review-correctness`, `review-integration` (Phase 4 round 1) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Verifier agent: `verifier` (Phase 4.5 single candidate)
- Reviewed head SHA: `5827e237c86150dd4a16e8f68eeffc211f66c46f`
- Review evidence: see this PR's comments — Phase 0.5 fixture review / Phase 4 cross-review bundle / Phase 4.5 verifier verdict / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `none`; selected risk packs: Documentation / migration notes, Legacy compatibility / examples
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking PLAUSIBLE. 1 candidate (cross-doc drift in sibling P1e docs) verifier-REFUTED out-of-scope → follow-up task spawned.

## Test plan

- [x] grep AC matrix 全 PASS (6/6)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 0.5 fixture review PASS
- [x] Phase 4 round 1 cross-review (correctness clean / integration 1 Warning out-of-scope)
- [x] Phase 4.5 verifier verdict on cand-01: REFUTED-out-of-scope
- [x] Phase 7 independent final review: clean, APPROVE merge
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [ ] Auto-merge after pre-merge evidence hard-gate (pending Phase 8)
