# A5 hydrology-acceptance verdict — heihe_x4

- Schema version: `A5-v1.0.0`
- Reference dir: `../../.p12-nvec-runs/a5pair/C.out`
- Candidate dir: `../../.p12-nvec-runs/a5pair/E.out`
- Thresholds file: `config/a5_thresholds.p12_tier2.yaml`

## Overall verdict: **PASS** (weighted score = 1.0000)

| Metric | Value | Threshold | Weight | Pass | Reason |
| --- | ---: | --- | ---: | :---: | --- |
| nse | 1 | min=0.99 | 1.00 | PASS |  |
| kge | 1 | min=0.99 | 1.00 | PASS |  |
| peak_magnitude_ratio | 1 | min=0.9, max=1.1 | 0.75 | PASS |  |
| peak_timing_offset | 0 | max_abs_steps=1 | 0.50 | PASS |  |
| runoff_volume_ratio | 1 | min=0.99, max=1.01 | 1.00 | PASS |  |
| monthly_bias_mae | 0 | max=0.05 | 0.50 | PASS |  |
| water_balance_residual | NaN | max=0.05 | 0.75 | PASS | informational: metric returned NaN (mesh metadata unavailable for area-weighted closure); excluded from weighted score per PR-Y2 policy |

## Interpretation

Candidate meets the hydrology-acceptance bar: the weighted PASS score 1.0000 clears the 0.85 gate and no critical-weight (>= 0.75) metric individually failed.
