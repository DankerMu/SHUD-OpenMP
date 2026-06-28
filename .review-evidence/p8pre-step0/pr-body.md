## Summary

p8pre-spike Epic #338 Step 0 doc-correction PR (preceding PR-A). Doc-only, no code / no SHUD pin bump.

Closes #339
Refs #338

### 改动

| 文件 | 改动 |
|---|---|
| [docs/p1e/p1e_perf_baseline.md](docs/p1e/p1e_perf_baseline.md) | §3.5 错 15-key 列表 → SHUD canonical (per `tools/cvode_stats_diff/canonical_15_keys.yaml`); §3.5/§7.1/§7.3 archive table `.pr-i-runs/` ×4 → `.p1e-i-runs/` |
| [docs/p8pre/step1_prep.md](docs/p8pre/step1_prep.md) | 新建 — inline 引用 P1e §3.1 wall median + §3.4 nst ladder + §4 跨 doc nfe，作 Step 1 PR-A/B/F + ADR-0003 baseline anchor |

### Acceptance Criteria (全 PASS)

| AC | command | 实测 |
|---|---|---|
| `pr-i-runs` in p1e_perf_baseline.md = 0 | `grep -c pr-i-runs docs/p1e/p1e_perf_baseline.md` | **0** |
| `p1e-i-runs` in p1e_perf_baseline.md > 0 | `grep -c p1e-i-runs docs/p1e/p1e_perf_baseline.md` | **4** |
| `nlcf` in p2a_profile_baseline.md = 0 | `grep -c nlcf docs/p2a/p2a_profile_baseline.md` | **0** (天然) |
| `nlcf` in p1e_perf_baseline.md = 0 | `grep -c nlcf docs/p1e/p1e_perf_baseline.md` | **0** |
| canonical 15-key marker `lenrwLS` 落地 | `grep -c lenrwLS docs/p1e/p1e_perf_baseline.md` | **1** |
| 上游 canonical anchor 未触 | `grep -c p1e-i-runs docs/p1e/p1e_pr_i_strict_omp_verification.md` | **8** ≥ 8 |

### Test plan

- [x] grep AC matrix 全 PASS (本地, head 5827e23)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 0.5 fixture review PASS — `.review-evidence/p8pre-step0/fixture-review.md`
- [ ] Phase 4 cross-review (compact-level doc accuracy / cross-ref integrity)
- [ ] Phase 7 independent final review
- [ ] Auto-merge after CI / pre-merge evidence hard-gate
