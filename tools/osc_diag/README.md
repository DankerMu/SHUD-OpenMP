# osc_diag — P11-osc CVODE stepping oscillation analyzer

**Version:** `OSC_DIAG-v1.0.0`
**Location:** `tools/osc_diag/`
**Language:** Python 3.11+, managed by [`uv`](https://docs.astral.sh/uv/) per the
repo-wide `Python 一律用 uv` rule (see `CLAUDE.md`).

`osc_diag` is the P11-osc spike analyzer (design.md §D3). It consumes the three
env-gated diagnostic CSVs emitted by the SHUD driver (PR-D1,
`SHUD/src/Model/MD_osc_diag.hpp`) and emits a machine-readable verdict on
whether `qinyijiang` / `keliya` minute-scale CVODE stepping is numerical flux
oscillation (`OSC_CONFIRMED`) or real fast dynamics (`REAL_DYNAMICS`), or is
`INCONCLUSIVE`.

The verdict is a **total decision function** whose thresholds are pinned in the
`osc-diag-analyzer` spec / spike-brief kill-gate — nothing is decided in-tool.

## Inputs (pinned schemas)

Emitted by a SHUD run with `SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1`:

| file | header | data columns |
|---|---|---|
| `diag_dt_trace.csv` | `# project_name=<name> solverstep_min=<min>` | `t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur` |
| `diag_osc_flips.csv` | `# project_name=<name> solverstep_min=<min> epsilon_m=<m>` | `entity_type,entity_id,flips_sf,flips_us,flips_gw,flips_stage` (typed `ele`\|`riv` rows) |
| `diag_osc_flips_daily.csv` | (same as flips) | `day_index,flips_sf,flips_us,flips_gw,flips_stage,flips_total` |

Rows with `delta_nst = 0` are skipped by the analyzer: 0 contribution to burst
numerators **and** to the nst denominator (dt-step-trace spec skip rule).

## Outputs

- **stdout**: `MARKER:OSC_DIAG_VERDICT_BEGIN..END` block (grep-friendly) with
  all gate inputs + thresholds + verdict.
- `<out>/dt_histogram.csv` — fixed mean-dt bins, interval count + nst share.
- `<out>/osc_diag_summary.json` — full gate inputs, concentration detail,
  Spearman detail, histogram, and the D2 proxy-limitation note.

Exit codes:
- `0` — analysis completed, MARKER block emitted.
- `2` — malformed / missing / truncated input (**fail closed**: no MARKER
  block is ever written from partial data).

## Install & run

```bash
cd tools/osc_diag
uv sync                            # creates .venv, installs deps
uv run pytest -v                   # requirement-driven test suite

# Against real SHUD diag CSVs (all three in one directory):
uv run osc_diag \
    --diag-dir /path/to/keliya.out \
    --case     keliya \
    --out      ./osc-report/keliya

# Or point at each file explicitly:
uv run osc_diag \
    --trace       .../diag_dt_trace.csv \
    --flips       .../diag_osc_flips.csv \
    --flips-daily .../diag_osc_flips_daily.csv \
    --case        qinyijiang \
    --out         ./osc-report/qinyijiang
```

### Why `--case` is required

The trace header carries `project_name` — the `./shud <arg>` name, the only
name SHUD knows. The benchmark case name can differ (qinyijiang runs as
`nanlin`). `--case` labels all outputs with the benchmark case; `project_name`
is preserved separately in `osc_diag_summary.json`.

`solverstep_min` is read **only** from the trace header (never `cfg.para`,
never a hardcode — design R5). The minutes→seconds conversion for mean-dt is
owned by this analyzer.

## Metrics

Canonical mean-dt formula (single source, verbatim from the dt-step-trace
spec):

```
mean_dt_seconds = 60 * solverstep_min / delta_nst
```

- **dt histogram** (`dt_histogram.csv`): fixed mean-dt bins in seconds —
  `[0,10), [10,60), [60,300), [300,600), [600,1200), [1200,inf)` — with the
  interval count and per-bin nst share (delta_nst=0 rows excluded). Interval
  counts sum to the number of non-skipped rows; nst shares sum to 1.
- **burst share** {60 s, 10 s}: fraction of total nst carried by intervals
  with mean dt below the threshold. Speedup-ceiling estimator: eliminating
  bursts caps wall gain at `1 / (1 - burst_share)`.
- **top-1% element flip concentration**: population = **elements only**;
  per-element flips = `flips_sf + flips_us + flips_gw` summed; concentration =
  share of total element flips carried by the top `ceil(0.01 × NumEle)`
  elements. River (`riv`) rows are reported separately and are **excluded**
  from the concentration population and its denominator.
- **per-day Spearman ρ**: rank correlation between daily `flips_total`
  (`diag_osc_flips_daily.csv`) and the daily count of sub-60 s mean-dt
  intervals (D1 rows bucketed on the shared `day_index = floor(t_next/1440)`
  key). Pass threshold ρ ≥ 0.5 (applied by the verdict, pinned in spec).

## Verdict — total decision function

Thresholds pinned in the `osc-diag-analyzer` spec / spike_brief kill-gate.
Every input combination maps to exactly one verdict; no catch-all branch:

```
OSC_CONFIRMED = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50 AND ρ >= 0.5
REAL_DYNAMICS = burst_share_60s <  0.50 OR  top1pct_concentration <  0.50
INCONCLUSIVE  = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50 AND ρ <  0.5
```

A NaN ρ (Spearman undefined — too few days or a constant series) fails the
ρ ≥ 0.5 gate, so a bursty + localized case with an undefined temporal link
resolves to `INCONCLUSIVE`, never `OSC_CONFIRMED` (conservative, matching the
D2 proxy under-count direction).

**Forcing-tracking is NOT a machine-verdict input** — no forcing data enters
the analyzer. Whether flip spikes track forcing peaks is human-assessed
corroborating evidence recorded in `docs/p11-osc/diagnosis_verdict.md`
(PR-D3), and does not change the MARKER verdict.

## D2 proxy limitation (design R3)

The D2 flip counts are an accepted-boundary state-delta proxy at SolverStep
resolution. They **conservatively under-count** sub-interval oscillation — the
bias is always **against** `OSC_CONFIRMED`, never toward it. This note is
emitted in `osc_diag_summary.json` (`proxy_limitation`) on every run.

## Fencepost handling — trailing partial day (PR-D1 reviewer finding)

An N-day run emits **N+1** day rows: the final SolverStep interval's
`t_next = END × 1440` floors exactly onto day `END`, producing a trailing
degenerate **single-interval** bucket (real keliya evidence: 91 daily rows,
`day_index` 12053–12143, last row `12143,4,0,0,0,4`). Both the `flips_daily`
side and the D1-row day-bucketing side derive from the *same* `t_next` stream,
so the trailing bucket appears consistently on **both** join sides — but it is
a partial day carrying a single interval.

**Decision (deterministic): drop the maximum `day_index` bucket from BOTH
Spearman join sides symmetrically** before ranking. Rationale:

- The bucket is a partial day (one interval), not a representative model day;
  including it injects a low-information outlier point into the rank
  correlation.
- The drop is applied identically to both variables (daily `flips_total` and
  daily sub-60 s interval count), so it **cannot bias ρ** toward either
  variable — it only removes the shared incomplete boundary point.
- It affects **only** the per-day Spearman join. The histogram, burst share,
  and concentration metrics consume all non-skipped trace rows / all element
  rows and are **unchanged** by the fencepost (the trailing interval's nst is
  still counted in burst share and the histogram — it is a real interval).

The dropped `day_index` is reported in
`osc_diag_summary.json.spearman_detail.dropped_trailing_partial_day_index`.
The drop can be disabled programmatically (`drop_trailing_partial_day=False`
in `metrics.daily_flip_dt_spearman`) for testing.

## Scope

This tool is the PR-D2 analyzer only. Out of scope (PR-D3): the 3-case
evidence matrix, the epic verdict (= qinyijiang MARKER verdict gated on the
xinanjiang_upstream control sanity), `docs/p11-osc/diagnosis_verdict.md`, and
any limiter design. Adjacency clustering of top-K flip elements is **deferred**
(D1/D2 CSVs carry no mesh topology) — a candidate for the single INCONCLUSIVE
refinement iteration.

## References

- `openspec/changes/p11-osc/specs/osc-diag-analyzer/spec.md` — analyzer contract
- `openspec/changes/p11-osc/specs/dt-step-trace/spec.md` — trace input schema
- `openspec/changes/p11-osc/specs/osc-flip-counters/spec.md` — flips input schema
- `openspec/changes/p11-osc/design.md` §D3 + §Verdict gate
- `docs/p11-osc/spike_brief.md` §D3 + §Kill-gate
- `SHUD/src/Model/MD_osc_diag.hpp` — the emitter (PR-D1)
- `tools/a5/` — uv project layout template
