# Phase 7 Final Review — PR #353 Gap Sweep

- Head SHA: 83a8864
- Branch: feat/issue-341-p8pre-pr-a-run
- PR mergeable: MERGEABLE
- CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)

## Diff scope (vs baseline/p8pre)

4 files, +457/-13. Exactly the expected set:
- tools/p8pre/run_n8_profile.sh (new, +270)
- tools/p8pre/submit_n8_profile_template.sbatch (+19/-13, 2-bug fix)
- docs/p8pre/n8_profile_run.md (new, +164)
- .gitignore (+4)
No openspec/ tracked diff (gitignored, by design — fix in working tree, archives at #349).

## AC matrix self-audit @ 83a8864

| AC | Evidence | Verdict |
|---|---|---|
| 18 cells COMPLETED, ExitCode 0:0 | `slurm.out:run_exit_code=0` on all 18 cells | PASS |
| Per-cell 5 artifacts | `ls /tmp/p8pre_n8_profile/<cell>/` = profile_B0.yaml + cvode_stats.txt + <case>.rivqdown.dat + slurm.out + slurm.err on heihe_N8_rep1 and heihe_x4_N8_rep1 | PASS |
| 15-key canonical (rej `nlcf`/`nfevals`/`hcur`/`qcur`/`hin`) | nst/nfe/nfeLS/nni/nli present, no typos | PASS |
| extras: t_CVODE_raw + t_wall_total | profile_B0.yaml grep confirms both | PASS |
| Bucket-sum invariant ±2% excl t_RHS_kernel | doc §4 worst \|Δ\|=0%; correctness review flagged this is algebraic identity (non-blocking) | PASS |
| run_exit_code=0 (RC0) | 18/18 slurm.out greps | PASS |
| Local rsync mirror complete | 18 cell dirs + jid_table.txt + render_stdout.txt at /tmp/p8pre_n8_profile/ | PASS |
| Slurm 三铁律 | sbatch from /scratch (run_n8_profile.sh L23-26 enforces), #SBATCH output/error pinned /scratch (template L48-49), no rsync calls in submitter (grep clean) | PASS |
| SHUD pin 7a1dc8f unchanged | submodule status confirms 7a1dc8f6 | PASS |
| openspec validate p8pre-spike --strict | "Change 'p8pre-spike' is valid" | PASS |
| .gitignore carry-forward (Phase 6 fix) | `.p8pre-runs/` + `.p8pre-pr-*-runs/` added L33-34 | PASS |
| Cross-N bitwise-identical (PR body cite) | heihe N1/N8: nst=6698 nfe=6943 strict; heihe_x4 N4/N8: nst=6575 nfe=6741 strict — matches PR body verbatim | PASS |

## Gap Sweep findings (NEW, not in Phase 4)

None. Spot-checks did not surface new defects:
- Doc/code RC0 path: doc body cites `run_exit_code=0` line from slurm.out; the YAML extras: block does NOT contain `run_exit_code:` (only t_CVODE_raw / t_wall_total / cvode_stats). PR doc consistently describes RC0 as slurm.out-sourced, not yaml-sourced — no drift.
- Carry-forward to PR-B: PR body §"Cross-N CVODE counter bitwise-identical (handoff to PR-B #342, no analysis here)" + doc §7 "Handoff to PR-B (#342)" with explicit `/tmp/p8pre_n8_profile/` entry-point + scope exclusion list. Adequate; downstream PR-B has clear contract.
- CLAUDE.md project rules: no Python in scripts (bash+awk); SHUD pin unchanged so submodule push rule N/A; 90-day rule untouched (PR doesn't modify cfg.para — heihe/heihe_x4 cfg under SHUD/Basins/, excluded from submodule git tree by design).

## Oracle integrity

PASS. No spec or test weakened. openspec/changes/p8pre-spike/ requirements intact, strict validation green. tools/p8pre/* additions only. No AC deletion.

## CI status

5/5 PASS (asan-ubsan keliya 38s, asan-ubsan qhh 2s, build-and-compare keliya 1m8s, setup 4s, tools-tests 11s). No failing, no pending.

## Final-review verdict

**Clean** — proceed to Phase 8 evidence + auto-merge.
