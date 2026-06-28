Reviewer agent: phase-7-final-review
Review round: final
Reviewed head SHA: 20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d
Summary: Gap Sweep clean. All 11 ACs verified; Phase 4 non-blocking notes correctly scoped as #341 carry-forwards.

Gap Sweep findings (items NOT in Phase 4 reports):
- None.

Pre-existing consumer compatibility:
- `.gitignore` `.p1e-runs/` / `.p1e-pr-*-runs/` / `.s*-runs/` namespaces distinct from `.p8pre-runs/`. Confirmed `git check-ignore --stdin -v` returns exit 1 for `.p8pre-runs/foo` (not ignored). Phase 4 integration carry-forward disposition is accurate: this PR writes nothing under `.p8pre-runs/` on Mac (wrapper skips when /scratch absent); #341 will add the entry when files materialize.

Completion self-audit:
- AC1 baseline/p8pre branch: PASS (Step 0 PR #350 per tasks §1.0; SHUD pin `7a1dc8f` verified)
- AC2 cn14 build Mode C exit 0: PASS (cn14_build_evidence.log:42)
- AC3 nm Serial NVec ≥ 1: PASS (log:46 = 1)
- AC4 nm OpenMP NVec = 0: PASS (log:48-49 = 0,0)
- AC5 nm GOMP_parallel ≥ 1: PASS (log:51 = 1)
- AC6 heihe forcing.trimmed 29M: PASS (cn14_case_evidence.log:2)
- AC7 heihe_x4 ≥ 200M: PASS (log:10 = 2.3G basin, log:17 = 286M forcing/)
- AC8 Slurm 三铁律: PASS — Rule 1: wrapper L31-33 requires `cd /scratch` pre-sbatch; Rule 2: template L48-49 `#SBATCH --output/--error` absolute under `/scratch/.../.p8pre-runs/`; Rule 3: L57-58 `REPO=/scratch/...`, L93 `./shud` under `$SHUD_DIR`
- AC9 18-cell coverage + cn14/cn15 pin: PASS — `bash tools/p8pre/render_n8_profile.sh | grep -c "^sbatch"` = 18; 9 heihe→cn14 + 9 heihe_x4→cn15 + 6 first-rep + 12 deps + 0 N=2
- AC10 SHUD pin 7a1dc8f unchanged: PASS (`git submodule status SHUD` → 7a1dc8f...)
- AC11 openspec validate strict: PASS ("Change 'p8pre-spike' is valid")

Slurm 三铁律 deep audit: template logs `hostname`, `numa`, `outer_git`, `SHUD pin`, `OMP_*`, `SHUD_RHS_THREADS`, `start/end_utc`, `run_exit_code` — slurm.out self-contained for sacct cross-ref.

Determinism env: `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `OMP_NUM_THREADS=__N__` + `SHUD_RHS_THREADS=__N__` (template L67-71) — byte-identical to `tools/p1e_2x2_sbatch_template.sbatch`. No deviation.

Timer.cpp bucket contract: `tools/profile/timer.cpp` L161-168 emits the 7 canonical buckets (`t_RHS_kernel`, `t_RHS_total`, `t_CVODE_internal`, `t_forcing_io`, `t_ET`, `t_output`, `t_other`) the template comment claims. L183-184 emit `t_CVODE_raw` + `t_wall_total` to `extras:`. Catch-all L193-206 supports §6.0a `t_precond_setup` auto-emit.

Hardcoded `frd_muziyao` paths: scoped per Epic #338 + CLAUDE.md (DankerMu/frd_muziyao sole production runner). Not a defect.

Failure mode (cn14 down): no auto-fallback, but D4 pins cn14/cn15 *intentionally* for cross-cell CPU SKU comparability. Wrapper is dry-run only in this PR; operational fallback correctly deferred to #341 runner.

Oracle integrity: PASS — no AC weakened, no test/spec deleted, no fixture rewritten. `docs/p1e/*.md` edits are tasks §0.2 typo fixes (`.pr-i-runs/` → `.p1e-i-runs/` + canonical 15-key set).

Final-review verdict: clean
