"""P11-osc CVODE stepping oscillation diagnostic analyzer.

Consumes the three env-gated diagnostic CSVs emitted by the SHUD driver
(PR-D1, `SHUD/src/Model/MD_osc_diag.hpp`):

    diag_dt_trace.csv         - per-SolverStep-interval CVODE counter deltas
    diag_osc_flips.csv        - per-entity cumulative state-delta sign flips
    diag_osc_flips_daily.csv  - per-day aggregate flip totals

and emits, per the osc-diag-analyzer spec:

    dt_histogram.csv          - fixed mean-dt bins with interval count + nst share
    burst share {60 s, 10 s}  - fraction of total nst in sub-threshold intervals
    top-K flip elements       - top-1% element flip concentration (elements only)
    per-day Spearman rho      - daily flips_total vs daily sub-60s interval count
    MARKER:OSC_DIAG_VERDICT    - machine-readable verdict block (total decision fn)

The verdict is computed by the TOTAL decision function in `verdict.py`;
thresholds are pinned in the spec, nothing is decided in-tool. The analyzer
fails closed: missing / truncated / unparsable input -> non-zero exit, no
MARKER block.

Public surface:
    osc_diag.parsers  - the three CSV parsers (pinned schemas)
    osc_diag.metrics  - burst share, histogram, flip concentration, Spearman
    osc_diag.verdict  - total decision function + MARKER block builder
    osc_diag.report   - dt_histogram.csv + summary JSON emitters
    osc_diag.cli.main - argparse entry point (installed as `osc_diag`)
"""

__version__ = "1.0.0"
