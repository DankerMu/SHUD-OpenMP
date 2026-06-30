# g0-amg-smoke-array-rerun — PR-B #416 Phase 6 dogfood re-run

Mac local re-run of keliya 90-day AMG with the PR-B #416 Phase 6 fixed
wrapper (SHUD `188854b`) + aggregator (`tools/p8tune.G0/aggregate_g0_smoke.sh`).

## Provenance

- **keliya**: fresh local run 2026-06-30 with `SHUD_LINSOL=amg` +
  `SHUD_TELEMETRY_TSV=...cell-keliya.telemetry.tsv`. The wrapper emitted
  the new 10-column TSV schema including `operator_complexity` derived
  from Hypre private-header macros (`hypre_ParAMGDataAArray` +
  `hypre_ParCSRMatrixNumNonzeros`).
  - exit_code=0, wall_total_sec=179, nst=124913, verdict_class=AMG_OK
  - operator_complexity=1.002945 (Hypre native, NOT a proxy)
  - cycle_complexity=1.000000 (telemetry-derived; biased toward 1.0 by
    ring-buffer truncation — 115656 entries dropped over the 1.7M-NLI
    run; surfaced via `MARKER:G0_TELEMETRY_RING_OVERFLOW` and
    `wall_convention=telemetry_truncated`).

- **xinanjiang_upstream / heihe_x4 / heihe_x16**: copied from PR-A
  `.review-evidence/g0-amg-smoke-array/` (no re-run — PR-A wrapper did
  not emit telemetry TSV, so these cells stay in `wall_convention=
  wall_total_proxy` per the new honest convention labeling).

## G0 Verdict (post-fix)

```
G0_3=PASS                            # earned via keliya telemetry-derived CC
G0_4=FAIL reason=heihe_x16:MALFORMED # heihe_x16 SIGKILL pre-Phase-6 (Mac
                                     # local re-run unable to reproduce
                                     # 252k-NumEle case; server re-run
                                     # is orchestrator scope)
G0_5=PASS                            # heihe_x4 improves vs SPGMR (0.092873
                                     # vs 0.238369), wall_convention=
                                     # wall_total_proxy (honest)
G0_6=PASS                            # CVODE solver-stats present for all
                                     # AMG_OK cells
G0_OVERALL=NO-GO                     # G0-4 gate (heihe_x16 evidence)
                                     # blocks; this fixture is leaf-agent
                                     # scope, server array re-run is
                                     # orchestrator follow-up
```

## Issue-body finding fixes verified

- **P0-1 (G0-3 unearned PASS)**: with the tightened semantics
  (`MARKER_PRESENT && HAS_TELEMETRY_CC`), G0-3 now requires at least
  ONE AMG_OK cell to have telemetry-derived cycle_complexity. Pre-fix
  fixture (`g0-amg-smoke-array/aggregate.tsv`) had `cycle_complexity=NA`
  for all 3 AMG_OK cells and would now FAIL with reason
  `G0_3_NO_AMG_OK_CELL_HAS_TELEMETRY`. Post-fix fixture (this dir)
  earns PASS via keliya's `1.000000` telemetry-derived CC.

- **P0-2 (operator_complexity gap)**: wrapper now derives the value
  via private Hypre macros. Mac local Hypre 3.1.0 produces
  `operator_complexity=1.002945` for keliya — a sane value for
  interp_type=6 / coarsen_type=8 BoomerAMG on a 124k-NumY system.

- **P0-3 (wall_convention mislabel)**: per-row label now distinguishes
  `setup_plus_solve_only` (telemetry-derived) vs `wall_total_proxy`
  (fallback, SHUD startup/IO included) vs `telemetry_truncated`
  (ring overflow). The pinned SPGMR baselines remain `wall_total_proxy`
  per the spike header (no header mutation per IA-4).

- **P1-1**: `MARKER:G0_5_AMG_VALUE_UNAVAILABLE case=heihe_x16` emitted
  (distinct from `G0_5_BASELINE_UNKNOWN`).

- **P1-2**: `amg_telemetry_dropped_overflow=115656` for keliya in
  aggregate.tsv col 22, plus `MARKER:G0_TELEMETRY_RING_OVERFLOW`.

- **P1-3 (subsumed by P0-1)**: `MARKER_PRESENT` loop now restricted
  to AMG_OK cells.

- **P1-4 (SIGKILL distinction)**: heihe_x16 in this fixture does NOT
  have a `CANCELLED AT ... DUE TO TIME LIMIT` stderr line so it stays
  labeled MALFORMED (cell_summary_missing). Server re-run with Slurm
  is the only path to exercise the new
  `AMG_WALL_OVERFLOW_INFERRED_SIGKILL` branch.

- **P1-5**: aggregator now iterates the canonical
  `EXPECTED_CELLS_LIST=(keliya xinanjiang_upstream heihe_x4 heihe_x16)`
  (file-glob iteration replaced).

- **P1-6**: 1-line code comment in `SHUD/src/Model/shud.cpp` justifies
  drain-BEFORE-CVodeFree.
