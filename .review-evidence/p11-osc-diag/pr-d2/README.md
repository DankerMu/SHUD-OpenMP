# PR-D2 emitter↔parser contract smoke (P11-osc, issue #435)

Task 2.5: run the `tools/osc_diag` analyzer on the **real keliya diagnostic
CSVs** (not the truncated PR-D1 evidence heads) and prove exit 0 + a MARKER
block. This validates only the emitter↔parser **contract** — it is NOT the
epic verdict (that is PR-D3, qinyijiang primary; per design.md §Verdict gate
keliya is corroborating evidence only).

All local Mac, serial build, submodule pinned at `75afb2b` (SHUD branch
`p11-osc`). No SHUD submodule pointer change in PR-D2.

## How the CSVs were regenerated

```bash
# 1. Serial build at the PR-D1 submodule pin (75afb2b).
cd SHUD && make clean && make shud            # green

# 2. Run keliya FROM THE BASIN DIR with both env gates on (~30 s).
#    Running from SHUD/ fails with a FILEIO error 12; the basin-dir
#    invocation is the one pinned in the PR-D1 evidence README.
cd SHUD/Basins/keliya
SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1 ../../shud keliya   # exit 0

# -> SHUD/Basins/keliya/output/keliya.out/{diag_dt_trace,diag_osc_flips,
#    diag_osc_flips_daily}.csv
```

## Analyzer invocation

```bash
cd tools/osc_diag
uv run osc_diag \
    --diag-dir /path/to/SHUD/Basins/keliya/output/keliya.out \
    --case     keliya \
    --out      ./osc-report/keliya
# exit 0; MARKER block on stdout; dt_histogram.csv + osc_diag_summary.json
# written to --out.
```

## MARKER block produced (verbatim — see `marker_block.txt`)

```
MARKER:OSC_DIAG_VERDICT_BEGIN
case=keliya
verdict=REAL_DYNAMICS
burst_share_60s=0.0019
burst_share_60s_threshold=0.5000
burst_share_10s=0.0000
top1pct_concentration=0.3766
top1pct_concentration_threshold=0.5000
spearman_rho=0.1285
spearman_rho_threshold=0.5000
MARKER:OSC_DIAG_VERDICT_END
```

**Exit code 0**, MARKER block emitted → contract smoke PASS.

Do NOT read this `REAL_DYNAMICS` as the epic verdict: keliya is corroborating
only, the epic verdict is the qinyijiang MARKER verdict (PR-D3), and this run
exercises the emitter↔parser path only.

## Cross-checks against the PR-D1 self-check evidence

The analyzer's parsed totals reproduce the PR-D1 leg-d self-check numbers
exactly (see `analyzer_out/osc_diag_summary.json`), confirming the parsers
read the real emitter output faithfully:

| quantity | analyzer summary | PR-D1 self-check |
|---|---:|---:|
| trace rows | 6480 | 6480 (CHECK2) |
| Σ delta_nst (total_nst) | 111130 | 111130 (CHECK1) |
| NumEle (n_ele) | 484 | 484 ele rows (CHECK4) |
| element flips + river flips | 52367 + 10015 = 62382 | Σ daily 62382 (CHECK5) |
| rows skipped (delta_nst=0) | 0 | — |

top-1% concentration population: `top_k = ceil(0.01 × 484) = 5`; river flips
(10015) reported separately, excluded from the concentration denominator.

## Fencepost handling (PR-D1 reviewer finding) — confirmed on real data

The keliya daily CSV has **91** day rows (`day_index` 12053–12143), the last
being the trailing partial-day bucket `12143,4,0,0,0,4`. The analyzer dropped
`day_index=12143` from BOTH Spearman join sides symmetrically and computed ρ
over the **90** complete days:

```
"spearman_detail": {
    "n_days_joined": 90,
    "dropped_trailing_partial_day_index": 12143
}
```

Rationale + the deterministic rule are documented in
`tools/osc_diag/README.md` §"Fencepost handling". The drop affects only the
per-day Spearman join; burst share and the histogram still count all 6480
non-skipped intervals (total_nst=111130 includes the trailing interval).

## dt histogram (real keliya) — `analyzer_out/dt_histogram.csv`

```
bin_lo_s,bin_hi_s,interval_count,nst_share
0,10,0,0
10,60,10,0.001889678755
60,300,6180,0.9890758571
300,600,282,0.008890488617
600,1200,8,0.0001439755242
1200,inf,0,0
```

Interval counts sum to 6480 (all non-skipped rows); nst shares sum to 1.0.
99% of keliya's nst sits in the `[60,300)` s bin — mean dt clusters just above
the 60 s burst threshold, so burst_share_60s is ~0.2% (the keliya step pile-up
is spread evenly across intervals rather than concentrated in sub-60 s bursts).

## Files

- `marker_block.txt` — verbatim MARKER block (stdout capture).
- `analyzer_out/dt_histogram.csv`, `analyzer_out/osc_diag_summary.json` — the
  full analyzer artifacts.
- `diag_csv/diag_dt_trace.head.csv` — head of the 413 KB real trace (6480 rows).
- `diag_csv/diag_osc_flips.headtail.csv` — head+tail of the 484 ele + 333 riv rows.
- `diag_csv/diag_osc_flips_daily.full.csv` — the full 91-row daily CSV (shows
  the trailing partial-day bucket at 12143).
