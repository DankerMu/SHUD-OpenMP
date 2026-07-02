"""Artifact emitters: dt_histogram.csv + osc_diag_summary.json.

The MARKER block itself is built in `verdict.format_marker_block` and
written to stdout by the CLI; these emitters produce the on-disk artifacts
that sit alongside it in the analyzer output directory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .metrics import ConcentrationResult, DtHistogram, SpearmanResult
from .verdict import VerdictResult

# D2 proxy limitation, surfaced in every summary (design R3): state-delta
# flips at SolverStep resolution UNDER-COUNT sub-interval oscillation. The
# under-count is conservative — it biases AGAINST OSC_CONFIRMED, never
# toward it.
PROXY_LIMITATION_NOTE = (
    "D2 flip counts are an accepted-boundary state-delta proxy at SolverStep "
    "resolution; they conservatively UNDER-count sub-interval oscillation "
    "(design R3), biasing against OSC_CONFIRMED, never toward it."
)


def write_histogram_csv(out_dir: Path, hist: DtHistogram) -> Path:
    """Write dt_histogram.csv: one row per fixed mean-dt bin.

    Columns: bin_lo_s, bin_hi_s, interval_count, nst_share. The top bin's
    upper edge is written as `inf`. Interval counts sum to the number of
    non-skipped trace rows; nst shares sum to 1 within float tolerance
    (spec scenario "histogram written and consistent").
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dt_histogram.csv"
    lines = ["bin_lo_s,bin_hi_s,interval_count,nst_share"]
    for b in hist.bins:
        hi = "inf" if math.isinf(b.hi) else f"{b.hi:g}"
        lines.append(f"{b.lo:g},{hi},{b.interval_count},{b.nst_share:.10g}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build_summary(
    case: str,
    project_name: str,
    solverstep_min: float,
    trace_rows: int,
    trace_rows_skipped: int,
    hist: DtHistogram,
    concentration: ConcentrationResult,
    spearman: SpearmanResult,
    verdict: VerdictResult,
) -> dict[str, Any]:
    """Assemble the osc_diag_summary.json payload."""

    def _num(v: float) -> Any:
        return "NaN" if isinstance(v, float) and math.isnan(v) else v

    return {
        "schema_version": "OSC_DIAG-v1.0.0",
        "case": case,
        "project_name": project_name,
        "solverstep_min": solverstep_min,
        "trace": {
            "rows_total": trace_rows,
            "rows_skipped_delta_nst_zero": trace_rows_skipped,
            "rows_used": trace_rows - trace_rows_skipped,
            "total_nst": hist.total_nst,
        },
        "gate_inputs": {
            "burst_share_60s": _num(verdict.burst_share_60s),
            "burst_share_10s": _num(verdict.burst_share_10s),
            "top1pct_concentration": _num(verdict.top1pct_concentration),
            "spearman_rho": _num(verdict.rho),
        },
        "concentration_detail": {
            "n_ele": concentration.n_ele,
            "top_k": concentration.top_k,
            "total_ele_flips": concentration.total_ele_flips,
            "riv_flips_total_reported_separately": concentration.riv_flips_total,
            "top_element_ids": list(concentration.top_element_ids),
            "top_element_flips": list(concentration.top_element_flips),
        },
        "spearman_detail": {
            "n_days_joined": spearman.n_days,
            "dropped_trailing_partial_day_index": spearman.dropped_trailing_day,
        },
        "histogram": [
            {
                "bin": b.label,
                "interval_count": b.interval_count,
                "nst_share": _num(b.nst_share),
            }
            for b in hist.bins
        ],
        "verdict": verdict.verdict,
        "proxy_limitation": PROXY_LIMITATION_NOTE,
    }


def write_summary_json(out_dir: Path, summary: dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "osc_diag_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=False)
        f.write("\n")
    return path
