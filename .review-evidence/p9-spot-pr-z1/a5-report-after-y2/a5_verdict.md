# A5 hydrology-acceptance verdict — heihe_x4

- Schema version: `A5-v1.0.0`
- Reference dir: `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p9-spot-pr-z1/cell-0-baseline.output`
- Candidate dir: `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p9-spot-pr-z1/cell-1-p9.output`
- Thresholds file: `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/a5/config/a5_thresholds.default.yaml`

## Overall verdict: **PASS** (weighted score = 1.0000)

| Metric | Value | Threshold | Weight | Pass | Reason |
| --- | ---: | --- | ---: | :---: | --- |
| nse | 1 | min=0.9 | 1.00 | PASS |  |
| kge | 0.99994 | min=0.85 | 1.00 | PASS |  |
| peak_magnitude_ratio | 0.999803 | min=0.9, max=1.1 | 0.75 | PASS |  |
| peak_timing_offset | 0 | max_abs_steps=4 | 0.50 | PASS |  |
| runoff_volume_ratio | 0.999973 | min=0.97, max=1.03 | 1.00 | PASS |  |
| monthly_bias_mae | 5.52264e-05 | max=0.05 | 0.50 | PASS |  |
| water_balance_residual | NaN | max=0.05 | 0.75 | PASS | informational: metric returned NaN (mesh metadata unavailable for area-weighted closure); excluded from weighted score per PR-Y2 policy |

## Interpretation

Candidate meets the hydrology-acceptance bar: the weighted PASS score 1.0000 clears the 0.85 gate and no critical-weight (>= 0.75) metric individually failed.
