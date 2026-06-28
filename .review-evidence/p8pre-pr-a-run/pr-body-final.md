## Summary

p8pre-spike Step 1 PR-A run slice. Executes Slurm 18-cell N=8 Mode C profile recheck, verifies per-cell 6 gates, rsync mirror to local /tmp/. Closes #341, refs #338.

### Deliverables

| 文件 | 用途 |
|---|---|
| `tools/p8pre/run_n8_profile.sh` (NEW) | Server-side runner: pre-flight Mode C nm verify, render + sbatch --parsable 18 cells with JID dep chain |
| `tools/p8pre/submit_n8_profile_template.sbatch` (MODIFIED) | 2-bug fix from PR #352: cwd → `Basins/<case>/` + output dir → `<cwd>/output/<case>.out/` + filenames `profile_B0.yaml` + `<case>.rivqdown.dat` |
| `docs/p8pre/n8_profile_run.md` (NEW, 164 行) | Academic-style execution log: wall + JID + verification table + cross-N bitwise observation |
| `.gitignore` (MODIFIED, +4) | Phase 6 fix cand-01: cover `.p8pre-runs/` + `.p8pre-pr-*-runs/` |

### Acceptance Criteria (全 PASS @ head `83a8864`)

| AC | 实测 |
|---|---|
| 18 cell 全 COMPLETED, ExitCode 0:0 (per sacct) | ✓ JIDs 9510-9527 |
| per-cell 5 artifact 齐全 (profile_B0.yaml + cvode_stats.txt + `<case>.rivqdown.dat` + slurm.out + slurm.err) | ✓ 18/18 |
| cvode_stats.txt 15 keys = SHUD canonical (REJECT typo `nlcf`/`nfevals`/`hcur`/`qcur`/`hin`) | ✓ |
| profile_B0.yaml `extras:` 含 `t_CVODE_raw` + `t_wall_total` | ✓ |
| bucket-sum invariant ±2% EXCLUDE `t_RHS_kernel` | ✓ worst \|Δ\| = 0.0000% (algebraic identity, 无 p2a v0.1 nested double-count bug recurrence) |
| `run_exit_code=0` in slurm.out | ✓ 18/18 |
| 本地 rsync mirror 18 cell 完整 | ✓ /tmp/p8pre_n8_profile/ |
| Slurm 三铁律 (submit from /scratch + output/error in /scratch + no rsync during runs) | ✓ |
| SHUD pin 7a1dc8f unchanged | ✓ |
| `openspec validate p8pre-spike --strict` | ✓ |
| `.gitignore` carry-forward (Phase 6 fix) | ✓ |
| Cross-N bitwise CVODE observation matches P1e PR-I absolute baseline | ✓ |

### Wall summary

- Total Slurm clock: ~1h10min (heihe stream cn14 done in ~10min, heihe_x4 stream cn15 dominant ~67min)
- heihe (9 cells): 1:36-2:28 per cell
- heihe_x4 (9 cells): 12:52-25:04 per cell

### Cross-N CVODE counter bitwise-identical (handoff to PR-B #342)

- **heihe** (9 cells N∈{1,4,8} × 3 reps): `nst=6698, nfe=6943, nfeLS=12632, nni=6942, nli=12632` strict — matches P1e PR-I absolute baseline anchored in [step1_prep.md](docs/p8pre/step1_prep.md) §3 + §4 → no Mode C silent regression
- **heihe_x4** (9 cells): `nst=6575, nfe=6741, nfeLS=30509, nni=6740, nli=30509` strict — matches P1e PR-I absolute baseline

## Agent Review

- Reviewer agents used: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf` (Phase 4 round 1) + `review-integration` (Phase 6.5 round 2 post-fix) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: 2 parallel verifiers — cand-01 + cand-02 both CONFIRMED → Phase 6 fix → round 2 both RESOLVED
- Reviewed head SHA: `83a8864` (final, post Phase 6 fix; round 1 was `d8602d0`)
- Review evidence: see this PR's comments — Phase 4 + 4.5 + 6.5 cross-review bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Public API/CLI + File IO + Concurrency + Numerical stability + Server/local partition + Documentation
- Key findings addressed: 2 CONFIRMED → 2 RESOLVED in Phase 6 + 1 traceability disclosure (openspec/changes/ gitignored, fix in local working tree, archives at #349)

## Test plan

- [x] 18 cells PASS 6-gate verification (server Slurm + sacct + per-cell artifact verify)
- [x] rsync mirror 完整 (18 dirs + jid_table + render_stdout)
- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0
- [x] Phase 4 round 1 expanded cross-review (4 reviewers)
- [x] Phase 4.5 verifier on 2 PLAUSIBLE candidates (both CONFIRMED)
- [x] Phase 6 fix: `.gitignore` + 9 openspec text substitutions
- [x] Phase 6.5 round 2 integration re-review (both RESOLVED, 0 new findings)
- [x] Phase 7 independent final review: clean, APPROVE merge
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [ ] Auto-merge after pre-merge evidence hard-gate
