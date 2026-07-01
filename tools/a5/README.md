# A5 — hydrology-acceptable validation pipeline

**Version:** `A5-v1.0.0`
**Location:** `tools/a5/`
**Language:** Python 3.11+, managed by [`uv`](https://docs.astral.sh/uv/) per the
repo-wide `Python 一律用 uv` rule (see `CLAUDE.md`).

A5 is the reusable *hydrology-acceptance gate* for SHUD solver-substitution and
policy-tuning studies. Given a reference SHUD run and a candidate SHUD run, it
computes seven hydrology metrics, compares each against configurable
thresholds, and returns a machine-readable PASS/FAIL verdict.

Delivered by PR-Y1. Consumers today:
- **PR-Z1 (P9 spot-check)** — invokes `a5` on the seven NWM cases after any
  solver / policy substitution to verify hydrology fidelity survives.
- **Future P10 DDM epic** — same CLI, potentially different threshold config.

## Contract

Input:
1. Two SHUD output directories (`*.out/` folders containing `*.rivqdown.dat`,
   `*.elevprcp.dat`, `*.eleveta.dat`, `*.eleygw.dat`).
2. A thresholds YAML file (default: `config/a5_thresholds.default.yaml`).
3. A case name and an output directory.

Output:
- `<out>/a5_metrics.json` — machine-readable per-metric values + overall verdict.
- `<out>/a5_verdict.md` — human-readable markdown report with a per-metric table.
- **stdout**: MARKER block (grep-friendly) for CI aggregation.

Exit codes:
- `0` — overall verdict PASS
- `1` — overall verdict FAIL
- `2` — malformed thresholds YAML (unusable config, distinct from a data FAIL)
- `3` — I/O error reading SHUD outputs

## Install & run

```bash
cd tools/a5
uv sync                           # creates .venv, installs deps
uv run pytest -v                  # runs the requirement-driven test suite

# Full invocation against real SHUD output directories:
uv run a5 \
    --reference /path/to/ref/case.out \
    --candidate /path/to/cand/case.out \
    --config    config/a5_thresholds.default.yaml \
    --case-name heihe_x4 \
    --out       ./a5-report/heihe_x4
```

## The seven metrics

| # | Metric                       | Formula / interpretation                                             | Family      | Default weight |
|---|------------------------------|----------------------------------------------------------------------|-------------|----------------|
| 1 | `nse`                        | Nash-Sutcliffe: `1 - Σ(cand-ref)² / Σ(ref-mean)²`. Perfect = 1.       | min-only    | 1.00 (critical) |
| 2 | `kge`                        | Kling-Gupta: `1 - √((r-1)² + (α-1)² + (β-1)²)`. Perfect = 1.          | min-only    | 1.00 (critical) |
| 3 | `peak_magnitude_ratio`       | `cand.max / ref.max`. Ideal = 1.                                     | interval    | 0.75 (critical) |
| 4 | `peak_timing_offset`         | `argmax(cand) - argmax(ref)` in timesteps.                           | abs-max     | 0.50            |
| 5 | `runoff_volume_ratio`        | `Σ(cand·dt) / Σ(ref·dt)`. Ideal = 1.                                 | interval    | 1.00 (critical) |
| 6 | `monthly_bias_mae`           | Mean absolute relative error on monthly aggregates.                  | max-only    | 0.50            |
| 7 | `water_balance_residual`     | `|Q_out - (P - ET - ΔS)| / P` over the run window.                   | max-only    | 0.75 (critical) |

Metric family semantics:
- **min-only**: PASS iff `value >= min`.
- **max-only**: PASS iff `value <= max`.
- **interval**: PASS iff `min <= value <= max`.
- **abs-max**: PASS iff `|value| <= max_abs_steps`.

Degenerate inputs (constant reference, zero total precipitation, etc.) produce
`NaN`, which the verdict layer treats as a FAIL with an explanatory reason.

## Overall verdict rule

Given per-metric PASS/FAIL results and weights:

```
weighted_score = Σ (w_i · pass_i) / Σ w_i
verdict = PASS  iff  weighted_score >= 0.85
                AND  every metric with weight >= 0.75 individually passed
```

Any metric with `weight >= 0.75` is treated as **critical**: a single critical
failure forces overall FAIL regardless of the aggregate score. This defeats
weighted-average masking of shape-preserving-but-magnitude-wrong regressions.

## Thresholds config

See `config/a5_thresholds.default.yaml` for the canonical starting point. A JSON
Schema (yaml-encoded) accompanies it at `config/thresholds.schema.yaml` for
editor autocomplete / CI lint hooks.

To tune thresholds for a specific case, copy the default file and edit — the
CLI accepts any path via `--config`.

## MARKER block

Emitted on stdout at the end of every run, regardless of verdict:

```
MARKER:A5_VERDICT_BEGIN
case=<case-name>
verdict=PASS|FAIL
weighted_score=<0.0000>
nse=<value|NaN>
kge=<value|NaN>
peak_mag_ratio=<value|NaN>
peak_timing_offset_steps=<int|NA>
runoff_volume_ratio=<value|NaN>
monthly_bias_mae=<value|NaN>
water_balance_residual=<value|NaN>
MARKER:A5_VERDICT_END
```

The line ordering, key names, and float precision are load-bearing — PR-Z1 and
future aggregators grep for these tokens. Any change is a semver-major bump
(new `A5-vN.0.0`).

## SHUD binary format the reader expects

Per `SHUD/src/classes/Model_Control.cpp::Print_Ctrl::open_file` (version >= 2):

```
offset (bytes)  content                        dtype
0               header (project metadata)      char[1024]
1024            start_time (YYYYMMDD)          float64
1032            NumVar (nc)                    float64
1040            column indices idx[nc]         float64[nc]
1040 + 8*nc     row 0: [t_min, x1..xnc]        float64[nc+1]
...             row k: [t_min, x1..xnc]        float64[nc+1]
```

All doubles are little-endian. The reader validates this layout and surfaces
clear errors on truncation or wrong endianness.

## Handoff to PR-Z1

PR-Z1 will:
1. For each of the seven NWM cases, run the reference solver (B1b baseline)
   and the candidate solver, producing two output directories.
2. Invoke `a5` on each pair, capturing the MARKER block per case.
3. Aggregate the MARKER blocks across cases into a summary table (`a5.tsv`)
   for the P9 verdict.

A5 itself does not know about "cases plural" — orchestration lives in the P9
runner script. Keeping A5 single-case makes it reusable for any future epic.

## Extending A5

- **New metric?** Add function to `src/a5/metrics.py`, wire into `_compute_metrics`
  in `cli.py`, add threshold family in `verdict.py`, extend MARKER block, and add
  requirement tests. Bump to `A5-v1.1.0`.
- **New metric family (e.g. "value must be within k standard deviations")?**
  Extend `_check_one` in `verdict.py`.
- **Different SHUD output format?** Add a version parameter to
  `shud_output.read_series()`; the current implementation targets `ver >= 2.0`
  (matching the on-disk SHUD writer).

## Non-goals

A5 is deliberately **not**:
- A solver benchmark harness (that is P9's job).
- A build-system integration test (that is `tools/check_manifest` etc.).
- A per-cell diagnostic tool (per-cell shape mismatches will be visible via
  `snapshot_repeatability` and `compare_snapshot`, not A5).
