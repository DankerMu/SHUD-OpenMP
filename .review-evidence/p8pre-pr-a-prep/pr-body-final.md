## Summary

p8pre-spike Step 1 PR-A prep slice. Authors Slurm 18-cell N=8 Mode C profile template + wrapper + server cn14 build/nm/case verify. Precedes #341 PR-A run.

Closes #340
Refs #338

### Deliverables

| 文件 | 用途 |
|---|---|
| `tools/p8pre/submit_n8_profile_template.sbatch` | 单 cell Slurm Mode C 模板，`__CASE__`/`__N__`/`__REP__`/`__NODE__` substitution + Slurm 三铁律 + determinism env |
| `tools/p8pre/render_n8_profile.sh` | POSIX bash+awk wrapper，render 18-cell matrix (2 case × 3 N × 3 rep) + cn14/cn15 case-pin + singleton chain |
| `docs/p8pre/pr_a_prep_evidence.md` | Academic-style evidence doc — build/nm/case raw 数据归档 |

### Acceptance Criteria (全 PASS @ head `20a7ec1`)

| AC | 实测 |
|---|---|
| `baseline/p8pre` 分支存在 (unlocked) | ✓（Step 0 PR #350 已建） |
| 服务器 cn14 build Mode C exit 0 | ✓ srun build PASS |
| nm Serial NVec ≥ 1 | ✓ (1) |
| nm OpenMP NVec = 0 | ✓ (0) |
| nm GOMP_parallel ≥ 1 (libgomp 实链) | ✓ (1) |
| heihe forcing.trimmed = 29M | ✓ |
| heihe_x4 basin ≥ 200M | ✓ (2.3G total, forcing/ 286M) |
| Slurm 三铁律 遵守 | ✓ template + wrapper 三条全 |
| 18-cell coverage + cn14/cn15 partition pin | ✓ 18 lines dry-run, 9:9 split |
| SHUD pin 7a1dc8f unchanged | ✓ |
| `openspec validate p8pre-spike --strict` | ✓ PASS |

### Carry-forward to #341 PR-A run

1. `.gitignore` 补 `.p8pre-runs/` + `.p8pre-pr-*-runs/`（本 PR Mac dry-run 不材化该路径，不阻塞）
2. PR-A runner doc 明确 JID 替换 target = stdout sbatch 行 not .sbatch 文件体

## Agent Review

- Reviewer agents used: `review-correctness`, `review-integration` (Phase 4 round 1) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED (0 concrete blocking candidates per `compact` precision-bias)
- Reviewed head SHA: `20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d`
- Review evidence: see this PR's comments — Phase 4 cross-review bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `compact`; selected risk packs: Public API/CLI + Documentation + Legacy-compatibility
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking. 1 Suggestion + 2 non-blocking notes → 2 carry-forward to #341 + 1 accepted as-is

## Test plan

- [x] grep AC matrix 全 PASS (11/11)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 4 round 1 cross-review (correctness APPROVE / integration APPROVE)
- [x] Phase 7 independent final review: clean, APPROVE merge
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [x] Local dry-run 18-cell sanity PASS
- [x] Server cn14 srun Mode C build + 3 nm gates PASS
- [ ] Auto-merge after pre-merge evidence hard-gate
