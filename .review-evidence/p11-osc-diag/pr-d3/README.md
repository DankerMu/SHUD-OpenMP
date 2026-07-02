# PR-D3 3-case evidence matrix (P11-osc, issue #436)

The evidence + verdict PR. 3-case 90-day **local Mac serial** diagnostic runs
with both env gates on (`SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1`), the PR-D2 analyzer
(`tools/osc_diag/`) applied per case, and `docs/p11-osc/diagnosis_verdict.md`.
**No SHUD / tools code changes** — this PR consumes what PR-D1/D2 built. SHUD
submodule pinned at `75afb2b` (SHUD branch `p11-osc`); no pointer bump.

## Build

`cd SHUD && make shud` — serial, green (pre-existing sprintf/ld warnings only,
zero new). Binary `SHUD/shud` at submodule `75afb2b`.

## Run invocation (per case, from its basin dir)

```bash
cd SHUD/Basins/<basin>
SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1 ../../shud <project>
# -> output/<project>.out/{diag_dt_trace,diag_osc_flips,diag_osc_flips_daily}.csv
```

Basin folder ≠ SHUD project name: `qinyijiang→nanlin`, `keliya→keliya`,
`xinanjiang_upstream→xinanjiang` (discovered via
`find SHUD/Basins/<basin>/input/ -name '*.cfg.para'`). 90-day truncation was
already deployed in every `cfg.para` (nanlin 366–456, keliya 12053–12143,
xinanjiang 0–90); configs were NOT modified.

## Analyzer invocation (per case)

```bash
cd tools/osc_diag
uv run osc_diag --diag-dir <...>/<project>.out --case <case> --out <this-dir>/<basin>
# exit 0 + MARKER block on stdout for all three.
```

`--case qinyijiang` maps project_name `nanlin` → benchmark case qinyijiang
(recorded per osc-verdict-gate spec "evidence completeness": qinyijiang traces
carry `project_name=nanlin` in their headers).

## Results (Tab.1 of the verdict doc)

| case | role | nst | wall (s) | burst_60s | burst_10s | top1% conc | Spearman ρ | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **qinyijiang** (`nanlin`) | **primary** | 156,580 | 293 | **0.9942** | 0.0000 | **0.0366** | −0.3753 | **REAL_DYNAMICS** |
| keliya | corroborating | 111,130 | 33 | 0.0019 | 0.0000 | 0.3766 | 0.1285 | REAL_DYNAMICS |
| xinanjiang_upstream (`xinanjiang`) | **control** | 6,775 | 6 | **0.0000** | 0.0000 | 0.2294 | NaN | REAL_DYNAMICS |

- **Control sanity gate (HARD): PASS** — xinanjiang_upstream
  `burst_share_60s = 0.0000 < 0.05` → epic verdict valid.
- **Epic verdict = REAL_DYNAMICS** (qinyijiang MARKER verbatim; per aggregation
  rule keliya is corroborating only).
- Decisive input: qinyijiang top-1% element flip concentration 0.0366 « 0.50 —
  minute-scale bursts are real but flips are spatially diffuse across all 3,155
  elements, not localized numerical oscillation. Negative Spearman ρ corroborates
  (no positive flip↔dt-collapse coupling).
- Gate decision (RECORD ONLY): REAL_DYNAMICS → epic closure + SHUD `p11-osc`
  merge-back recommendation (instrumentation retained, default-off,
  bitwise-neutral). No limiter stub, no GitHub issue, no openspec change created
  by this PR (see `diagnosis_verdict.md` §9).

## Per-case artifacts (`<basin>/`)

Each of `qinyijiang/`, `keliya/`, `xinanjiang_upstream/` contains:

- `diag_dt_trace.csv`, `diag_osc_flips.csv`, `diag_osc_flips_daily.csv` — the
  three raw diag CSVs (full).
- `dt_histogram.csv`, `osc_diag_summary.json` — analyzer outputs.
- `marker_block.txt` — the verbatim `MARKER:OSC_DIAG_VERDICT` block (stdout capture).
- `analyzer_stdout.txt`, `analyzer_exit.txt` — full analyzer stdout + exit code (0).
- `cvode_stats.txt` — end-of-run CVODE stats (SHUD-emitted).
- `run_log_tail.txt`, `wall_time.txt` — run log tail + wall-clock start/end/seconds.
- `perday_subdt_vs_flips.csv` — per-day sub-60 s / sub-10 s interval count joined
  with daily flip totals (forcing-tracking prep; NOT a machine input).
- `basin_mean_daily_precip.csv` — basin-mean daily precip (`Precip_mm.d`) over the
  run window, keyed by day_index (human-assessed forcing-tracking corroboration
  material; NOT a machine input — no forcing enters the analyzer).

## Human-assessed forcing-tracking (verdict doc §5.6, NOT a machine input)

- qinyijiang: top-2 flip days (446/447) = top-2 precip days (26.2 / 30.6 mm·d⁻¹);
  sub-60 s stepping is a **whole-window baseline** (91/91 days) — real fast-response
  basin, flip peaks ride precip events.
- keliya: flip peaks (12141/12142) = window precip top-2; the sole sub-60 s day
  (12054) had ~0 precip → dt-collapse anti-correlates with precip.
- xinanjiang (control): a 34 mm·d⁻¹ precip event (day 72) produced **0** sub-60 s
  intervals — healthy solver does not collapse dt under real forcing.

## Not in this PR

- No SHUD submodule pointer change (pinned 75afb2b).
- No limiter / smoothing / deadband code (REAL_DYNAMICS → hard-excluded).
- No `docs/p11-osc/spike_brief.md` status-field edit or master-plan cross-ref
  (task 3.5, separate) — only `diagnosis_verdict.md` (task 3.3) + gate-decision
  record (task 3.4) here.
- kashigeer excluded: cvode_stats blocked deferred-upstream (X76 forcing band
  missing on both endpoints per `benchmarks/kashigeer/B0_output/DEFERRED.txt`);
  refresh only after the upstream forcing fix (verdict doc §7.3).
