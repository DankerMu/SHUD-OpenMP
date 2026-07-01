"""A5 hydrology-acceptable validation pipeline.

Standalone, reusable validation gate for SHUD solver-substitution and
policy-tuning studies. Given a reference and a candidate SHUD output
directory (or synthetic numpy arrays), compute seven hydrology metrics
(NSE, KGE, peak magnitude, peak timing, runoff volume, monthly bias,
water balance residual), compare them against configurable thresholds,
and emit a machine-readable MARKER block plus a human-readable report.

Consumers:
    - PR-Z1 (P9 spot-check gate on solver substitution): calls the CLI.
    - Future P10 DDM epic: same CLI, potentially different thresholds.

Public surface:
    a5.metrics      - the seven metric functions
    a5.verdict      - threshold gate + MARKER block builder
    a5.shud_output  - SHUD .dat binary reader
    a5.report       - JSON + markdown report emitters
    a5.cli.main     - argparse entry point (installed as `a5` console script)
"""

__version__ = "1.0.0"
