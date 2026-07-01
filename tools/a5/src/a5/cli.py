"""A5 CLI entry point.

Usage:
    a5 --reference REF_OUTPUT_DIR --candidate CAND_OUTPUT_DIR \
       --config THRESHOLDS_YAML --case-name NAME --out OUTPUT_DIR

The reference and candidate directories are SHUD project output dirs
(the `.out/` folders that contain `*.rivqdown.dat`, `*.elevprcp.dat`,
`*.eleveta.dat`, `*.eleygw.dat`). The reader locates them by scanning
the directory for files matching `*.<keyword>.dat`.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from . import metrics as _m
from .report import write_reports
from .shud_output import ShudSeries, read_series
from .verdict import (
    MalformedThresholdError,
    evaluate,
    format_marker_block,
    load_thresholds,
)

EXIT_OK = 0
EXIT_FAIL_VERDICT = 1
EXIT_MALFORMED_CONFIG = 2
EXIT_IO_ERROR = 3


def _find_dat(output_dir: Path, keyword: str) -> Optional[Path]:
    """Locate `*.<keyword>.dat` in output_dir. Returns None if absent."""
    matches = sorted(output_dir.glob(f"*.{keyword}.dat"))
    if not matches:
        return None
    if len(matches) > 1:
        # Multiple candidates — pick the shortest project-prefix (most
        # canonical) but stay deterministic.
        matches.sort(key=lambda p: (len(p.name), p.name))
    return matches[0]


def _load_case_series(output_dir: Path) -> dict[str, ShudSeries]:
    """Load all SHUD outputs relevant to A5 (rivqdown, elevprcp, eleveta, eleygw)."""
    keywords = ["rivqdown", "elevprcp", "eleveta", "eleygw"]
    out: dict[str, ShudSeries] = {}
    for kw in keywords:
        p = _find_dat(output_dir, kw)
        if p is None:
            continue
        try:
            out[kw] = read_series(p)
        except (ValueError, FileNotFoundError) as exc:
            print(f"warning: failed to read {p}: {exc}", file=sys.stderr)
    return out


def _extract_outlet(series: ShudSeries) -> np.ndarray:
    """Return outlet-column values (last column). Empty array if no data."""
    if series.values.size == 0:
        return np.zeros(0, dtype=np.float64)
    return series.values[:, -1].copy()


def _basin_mean(series: ShudSeries) -> np.ndarray:
    """Return basin-averaged (mean across columns) timeseries."""
    if series.values.size == 0:
        return np.zeros(0, dtype=np.float64)
    return series.values.mean(axis=1)


def _month_labels(timestamps: np.ndarray) -> np.ndarray:
    """Convert datetime64 timestamps to YYYYMM integer labels."""
    if timestamps.size == 0:
        return np.zeros(0, dtype=np.int64)
    # datetime64[us] -> Python datetime -> YYYYMM
    labels = np.empty(timestamps.size, dtype=np.int64)
    for i, ts in enumerate(timestamps):
        dt = ts.astype("datetime64[us]").astype(datetime)
        labels[i] = dt.year * 100 + dt.month
    return labels


def _compute_metrics(
    ref: dict[str, ShudSeries], cand: dict[str, ShudSeries]
) -> dict[str, float]:
    """Compute the seven A5 metrics from paired reference / candidate series."""
    if "rivqdown" not in ref or "rivqdown" not in cand:
        raise RuntimeError(
            "rivqdown.dat missing in either reference or candidate; A5 needs "
            "discharge to compute NSE/KGE/peak/runoff-volume/monthly-bias"
        )
    ref_q = _extract_outlet(ref["rivqdown"])
    cand_q = _extract_outlet(cand["rivqdown"])
    if ref_q.size != cand_q.size:
        raise RuntimeError(
            f"reference discharge length {ref_q.size} != candidate {cand_q.size}; "
            "align run periods before A5 evaluation"
        )
    if ref_q.size == 0:
        raise RuntimeError(
            "reference discharge series is empty; nothing to evaluate"
        )

    # Uniform daily dt (SHUD default output cadence). Downstream cases
    # with non-uniform output should pass dt via DY.dat — TODO PR-Z1.
    dt_days = 1.0

    results: dict[str, float] = {
        "nse": _m.nse(ref_q, cand_q),
        "kge": _m.kge(ref_q, cand_q),
        "peak_magnitude_ratio": _m.peak_magnitude_ratio(ref_q, cand_q),
        "peak_timing_offset": float(_m.peak_timing_offset(ref_q, cand_q)),
        "runoff_volume_ratio": _m.runoff_volume_ratio(ref_q, cand_q, dt=dt_days),
    }

    month_labels = _month_labels(ref["rivqdown"].timestamps)
    if month_labels.size == ref_q.size:
        results["monthly_bias_mae"] = _m.monthly_bias_mae(
            ref_q, cand_q, month_labels
        )
    else:
        # Guard for reader/labeler mismatch — surface NaN + warning.
        print(
            "warning: month-label array length mismatch; skipping monthly_bias_mae",
            file=sys.stderr,
        )
        results["monthly_bias_mae"] = float("nan")

    # Water balance: needs P, ET, and storage change from the candidate.
    # Storage change = Δ(GW head * area). Because we do not have per-cell
    # area at this layer, approximate ΔS as diff of basin-mean groundwater.
    # This is a coarse proxy suitable for regression comparison against a
    # candidate that shares the identical mesh; absolute closure will not
    # match true basin volume balance. PR-Z1 refines this once we have
    # element-area weighting.
    if all(k in cand for k in ("elevprcp", "eleveta", "eleygw")):
        p_mean = _basin_mean(cand["elevprcp"])
        e_mean = _basin_mean(cand["eleveta"])
        gw = _basin_mean(cand["eleygw"])
        q_mean = _basin_mean(cand["rivqdown"])
        n = min(p_mean.size, e_mean.size, gw.size, q_mean.size)
        if n >= 2:
            ds = np.diff(gw[:n])
            # Align: use step-wise change; pad first step so sizes match.
            ds = np.concatenate([[0.0], ds])
            results["water_balance_residual"] = _m.water_balance_residual(
                q_mean[:n], p_mean[:n], e_mean[:n], ds[:n]
            )
        else:
            results["water_balance_residual"] = float("nan")
    else:
        results["water_balance_residual"] = float("nan")

    return results


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="a5",
        description=(
            "A5 hydrology-acceptance validation pipeline. Compare a candidate "
            "SHUD output directory against a reference and emit a PASS/FAIL "
            "verdict against configurable thresholds."
        ),
    )
    p.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Reference SHUD output directory (e.g. .../basins/keliya/output/keliya.out)",
    )
    p.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Candidate SHUD output directory (same layout as --reference)",
    )
    p.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Thresholds YAML file (see config/a5_thresholds.default.yaml)",
    )
    p.add_argument(
        "--case-name",
        required=True,
        help="Case identifier written into the MARKER block (e.g. heihe_x4)",
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for a5_metrics.json + a5_verdict.md",
    )
    return p


def run(args: argparse.Namespace) -> int:
    try:
        thresholds = load_thresholds(str(args.config))
    except MalformedThresholdError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MALFORMED_CONFIG

    try:
        ref = _load_case_series(args.reference)
        cand = _load_case_series(args.candidate)
    except (OSError, ValueError) as exc:
        print(f"error: failed to load SHUD output: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    try:
        metric_values = _compute_metrics(ref, cand)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_IO_ERROR

    verdict = evaluate(metric_values, thresholds)
    write_reports(
        out_dir=args.out,
        case=args.case_name,
        reference_dir=str(args.reference),
        candidate_dir=str(args.candidate),
        thresholds_file=str(args.config),
        verdict=verdict,
    )
    marker = format_marker_block(args.case_name, verdict)
    sys.stdout.write(marker)
    sys.stdout.flush()
    return EXIT_OK if verdict.verdict == "PASS" else EXIT_FAIL_VERDICT


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
