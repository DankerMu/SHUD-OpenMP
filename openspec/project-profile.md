# SHUD-OpenMP Project Profile

Living artifact (Phase 0.0 bootstrap → maintained Phase 0.5). Copies SHUD Solver template + adds OpenMP-baseline / cross-endpoint surfaces specific to this repo.

Project profile: SHUD-OpenMP
Entry surfaces:
- SHUD submodule: `SHUD/src/Model*`, `SHUD/src/ModelData/MD_*`, `SHUD/src/Equations/cvode_config.cpp`, `SHUD/Makefile`, `SHUD/configure`
- Case data layout: `SHUD/Basins/<case>/input/<case>/<case>.{tsd.forc,cfg.para,...}`
- Benchmark registry: `benchmarks/<case>/manifest.yaml`, `benchmarks/<case>/B0_output/`
- Tooling: `tools/{rhs_snapshot,compare_snapshot,profile,fix_case_paths,bootstrap_check,archive_b0_output,check_manifest}/`
- CI + docs: `.github/workflows/serial-baseline.yml`, `docs/{build_manifest,status_matrix,profile_platform,profile_decision}.md`
- Branch topology: `baseline/current` long-lived + `B0-tag`

Contracts:
- Case manifest schema (master plan §S0.2 fields: endpoint / NumEle / has_lake / snapshot_probe.t_values / output_compare.output_files)
- SHUD output schema (`output/<case>/*.dat` per `has_lake`)
- CVODE PrintFinalStats key=value (nfe / nfeLS / nni / nli / nsetups / netf)
- RHS snapshot binary `format.h` version (40-byte header + per-array layout)
- Compile switches `SHUD_DUMP_RHS` / `SHUD_ENABLE_PROFILE` — default 0 must be bitwise-neutral vs upstream pin `3aec657`
- Status matrix cell vocabulary `PASS / FAIL / BLOCKED / N/A / PENDING`
- A0 acceptance checklist 9 项

Risk axes:
- Bitwise reproducibility across runs / threads / platforms (Apple Silicon vs target Linux)
- CVODE / SUNDIALS 6.x version pin enforced link-time
- Compile flag drift (`-ffast-math` / `-Ofast` / `-funsafe-math-optimizations` disallowed)
- OpenMP fork-join correctness; `NumEle < OMP_CUTOFF` serial fallback
- Profile timer overhead ≤ 1% (compile-switch neutrality)
- Server vs local data partition (heihe / heihe_x4 are server-only; no 12G forcing in PRs)
- Credential / absolute-path leak into committed files

Typical evidence:
- 3-run repeatability SHA256 (per-case `repeatability.txt`)
- `compare_snapshot` bitwise diff vs golden + ULP report
- `check_manifest.py` schema validator pass
- CVODE stats file with 6 required keys
- `profile_B0.yaml` `t_other_pct ≤ 5%`
- Server rsync receipts (placeholders only, no credentials in PR body)

Domain risk packs (added to core packs):
- Numerical stability / conservation / NaN
- Solver runtime / performance / threading
- Bitwise reproducibility / compile-switch neutrality / cross-platform
- Server/local data partition / endpoint discipline

Domain expanded-triggers (added to core triggers):
solver, CVODE, SUNDIALS, OpenMP, thread, RHS, snapshot, profile timer, compile switch, baseline, B0-tag, server-only, rSHUD, manifest, benchmark, bitwise
