## Summary

- **2 new Mac-side tools** (305 lines total):
  - `tools/p8tune/run_maxl_cell.sh` — per-cell driver (env-var setup + isolated working dir + /usr/bin/time wall + 5-artifact collection + provenance log verification)
  - `tools/p8tune/submit_maxl_sweep_template.sbatch` — Slurm array template `--array=1-60%11` with deterministic (case,N,maxl,rep) decoder + Slurm 三铁律 compliance
- **60-cell sweep EXECUTED on cn14 + idle siblings** (Slurm job 9690, 60/60 COMPLETED):
  - matrix: `maxl ∈ {5, 10, 15, 20, 30}` × `case ∈ {heihe, heihe_x4}` × `N ∈ {1, 8}` × `rep ∈ {1, 2, 3}` = 60 cells
  - 11-way parallel concurrency cap → wall ~80 min total
  - All 60 cells produce {profile_B0.yaml, cvode_stats.txt, rivqdown.dat, wall.sec, cell.meta} + stdout provenance log per spec L45
- **Cross-maxl saturation finding (preview, full adjudication is PR-E)**:
  - `rivqdown.dat` SHA12 IDENTICAL across maxl ∈ {10, 15, 20, 30} within each (case, N) tuple
  - maxl=5 produces different SHA12 (separate group)
  - heihe N=1: ncfl `85 → 0` at maxl=10 (solver failures eliminated); ncfn 7→5
  - heihe_x4 N=1: ncfl `3620 → 0` at maxl=10; ncfn `51 → 0`
  - Wall delta marginal at N=1 (heihe +1%, heihe_x4 −0.6%) → counter improvement essentially free
  - **N=8 heihe_x4 case-asymmetric**: maxl=10 is +19% slower than maxl=5 → counter-vs-wall divergence for PR-E G5 attention
- **OMP-neutrality preserved**: SHA12 identical for N=1 vs N=8 within same (case, maxl) → B1a S4 OMP-neutrality regression detector PASS
- **PREC_NONE preserved**: PR-C env-var hook honored (provenance log line in each cell stdout)

## Why

PR-B verdict gated "full 60-cell sweep GO" (hard-evidence trigger satisfied). PR-C provided the runtime `SHUD_SPGMR_MAXL` env-var hook. PR-D executes the actual sweep and produces the artifact set that PR-E will aggregate into the G1-G8 verdict + ADR-0004 outcome adjudication.

PR-D's job: drive the sweep + capture raw artifacts. PR-E's job: aggregate + adjudicate.

OpenSpec change: `p8tune-spgmr-maxl` (capability `maxl-sweep-verdict` partial; PR-D executes §4.3-§4.6).

## Scope

2 tool files (+305 lines, 0 deletions):
- `tools/p8tune/run_maxl_cell.sh` (NEW, 222 lines, executable)
- `tools/p8tune/submit_maxl_sweep_template.sbatch` (NEW, 83 lines, executable)

No `.c/.cpp/.h` changes. No Makefile. No SHUD submodule pointer change (still `6ce17d6`). No doc-set change in this PR (PR-E will author `docs/p8tune/maxl_sweep_verdict.md` + `docs/adr/0004-maxl-sweep-decision.md`).

Sweep artifacts (server-resident, NOT in PR):
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/{heihe,heihe_x4}_N{1,8}_maxl{5,10,15,20,30}_rep{1,2,3}/` (60 cell dirs × 5 artifacts each = 300 files; full rivqdown.dat stays on server for PR-E)
- `/scratch/.../_summary.tsv` (61-row × 22-col flat KV; built by `_summary.sh` on server)
- `/scratch/.../_slurm/cell_9690_*.{out,err}` (Slurm per-task logs)

Local-only (gitignored, not in PR diff):
- `.review-evidence/p8tune-pr-d/{summary.tsv, sweep_evidence.tar.gz}` — summary + selected sample cell artifacts

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] Mac `bash -n tools/p8tune/run_maxl_cell.sh` → 0 syntax errors
- [x] Mac `bash -n tools/p8tune/submit_maxl_sweep_template.sbatch` → 0 syntax errors
- [x] Mac DRY-RUN of (case, N, maxl, rep) decoder for task_id ∈ [1, 60] → all 60 unique cells, axis coverage verified
- [x] Server preflight: SHUD pin = `6ce17d6` (PR-C live), heihe + heihe_x4 cfg.para 90-day truncated, idle nodes available, output dir created
- [x] Server single-cell smoke (heihe N=1 maxl=10 rep=1 via srun cn14) → CELL OK, wall=131s
- [x] Server 60-cell sweep submit (Slurm 9690, sbatch from `/scratch/.../p8tune-runs/maxl_sweep/`) → 60/60 COMPLETED
- [x] All 60 cells produce required 5 artifacts → verified by per-cell `[ -f wall.sec ] && [ -s wall.sec ] && [ -f cvode_stats.txt ] && [ -f rivqdown.dat ] && [ -f profile_B0.yaml ] && [ -f cell.meta ]` loop
- [x] All 60 cells emit `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` provenance log line (PR-C contract) → `prov_log_count=1` per row of summary.tsv
- [x] Cross-maxl bit-divergence map confirms saturation at maxl=10 (rivqdown.dat SHA12 identical across {10,15,20,30} within each (case, N))
- [x] N=1 vs N=8 within same (case, maxl) → SHA12 identical (OMP-neutrality)
- [x] Slurm 三铁律 compliance: sbatch from `/scratch/`, `--output/--error` under `/scratch/`, referenced script `run_maxl_cell.sh` at `/scratch/.../tools/p8tune/`

## Agent Review

(Populated by Phase 8.)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #367.
