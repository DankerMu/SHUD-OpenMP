# A5 hydrology-acceptance verdict — heihe_x4

- Schema version: `A5-v1.0.0`
- Reference dir: `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p9-spot-pr-z1/cell-0-baseline.output`
- Candidate dir: `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p9-spot-pr-z1/cell-1-p9.output`
- Thresholds file: `config/a5_thresholds.default.yaml`

## Overall verdict: **FAIL** (weighted score = 0.8636)

| Metric | Value | Threshold | Weight | Pass | Reason |
| --- | ---: | --- | ---: | :---: | --- |
| nse | 1 | min=0.9 | 1.00 | PASS |  |
| kge | 0.99994 | min=0.85 | 1.00 | PASS |  |
| peak_magnitude_ratio | 0.999803 | min=0.9, max=1.1 | 0.75 | PASS |  |
| peak_timing_offset | 0 | max_abs_steps=4 | 0.50 | PASS |  |
| runoff_volume_ratio | 0.999973 | min=0.97, max=1.03 | 1.00 | PASS |  |
| monthly_bias_mae | 5.52264e-05 | max=0.05 | 0.50 | PASS |  |
| water_balance_residual | 7.52011e+13 | max=0.05 | 0.75 | FAIL | value 7.52011e+13 > max 0.05 |

## Interpretation

Candidate does NOT meet the hydrology-acceptance bar. See the per-metric reasons column for what regressed. A single critical-weight metric failure is sufficient to force FAIL, regardless of the aggregate weighted score.
