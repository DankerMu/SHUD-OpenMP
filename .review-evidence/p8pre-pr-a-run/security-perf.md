Reviewer agent: review-security-perf
Review round: round 1
Reviewed head SHA: d8602d0fb4e609c106eaa5f79e973830182ac150

Summary: PR is security-clean and performance-prudent; all checklist items pass with one defensive-code note on unquoted sbatch arg expansion and one observation re P1e wall delta (already explained by step1_prep.md §0).

Findings:
- None.

Non-blocking notes:

1. **Defensive: unquoted `$sbatch_args` expansion** — `tools/p8pre/run_n8_profile.sh:215` does `sbatch --parsable $sbatch_args` with intentional word-splitting. Source = `render_n8_profile.sh` stdout, a trusted internal tool emitting fixed-alphabet strings (`sbatch --job-name=p8pre_<case>_N<n>_singleton [--dependency=afterany:<JID>] <abs-path>.sbatch`); case ∈ {`heihe`,`heihe_x4`}, N ∈ {1,4,8}, rep ∈ {1,2,3}, JID = digits validated by `[[ "$jid" =~ ^[0-9]+$ ]]` at L228 before being stashed into `PREV_JID`. No injection vector. If future PRs let `--job-name` or args come from less-trusted sources, switch to bash array (`sbatch --parsable "${args[@]}"`).

2. **Slurm 三铁律 verified** — submit_log.txt L1 `cwd=/scratch/...` (rule #1); template L48-49 `#SBATCH --output/--error` under `/scratch/.../.p8pre-runs/<cell>/` (rule #2); rendered .sbatch + binary + cfg under `/scratch` (rule #3). Pre-flight rebuild path (L94-103) runs on host `xnode` (login) — per CLAUDE.md "5-min verify build acceptable on login node". Compliant; revisit if heihe_x4 build grows beyond 5 min.

3. **No credentials in evidence** — submit_log.txt, rsync_log.txt, monitor_log.txt, jid_table.txt, verification.txt scanned: zero tokens, keys, or passwords. ssh/scp inferred to use existing passwordless key per CLAUDE.md two-host sync convention.

4. **Singleton chain wall efficient (design D4)** — per (case, N) 3 reps serial via `--dependency=afterany:<prev_JID>`. monitor_log shows 70min total wall, dominated by heihe_x4 N=1 chain (3×~24min ≈ 72min sacct sum). 6 parallel chains (2 cases × 3 N) on cn14/cn15 → max 24/40 cpus utilized per node at N=8, no oversubscription. Memory: heihe_x4 ~40k cells × 3 jobs ≈ 1.5GB/node vs 170GB capacity — comfortable.

5. **Wall vs P1e PR-I delta is expected, not regression** — heihe N=1 147s (this run) vs P1e PR-I 504s (`docs/p1e/p1e_perf_baseline.md` §3.1) → 3.4× faster, prima facie alarming. Resolution: `docs/p8pre/step1_prep.md` §0 explicitly states P1e PR-I built at SHUD `3341368d` WITHOUT `SHUD_ENABLE_PROFILE=1`, while this run uses SHUD `7a1dc8f` WITH PROFILE. Doc explicitly says "不能直接当 Step 2 gate-4 baseline (Timer instrumentation overhead bias)"; PR-A run IS the new gate-4 baseline. heihe_x4 aligns tightly (N=1: 1442 vs 1340s +7.6%; N=8: 773 vs 775s -0.3%), supporting the build-config-difference hypothesis (heihe small case more sensitive to per-step constants). No action.

6. **rsync destination namespaced** — `/tmp/p8pre_n8_profile/`, 47.7 MB, 18 cell dirs + jid_table.txt + render_stdout.txt only. No path-traversal vectors; all source paths under `/scratch/.../.p8pre-runs/`.

7. **Strict-mode hygiene OK** — `set -euo pipefail` in run_n8_profile.sh L37, render_n8_profile.sh L40, template L55. Intentional bypasses at run_n8_profile.sh L214 and template L99-102 are necessary to capture exit codes for triage; both restore strict mode immediately. Acceptable.
