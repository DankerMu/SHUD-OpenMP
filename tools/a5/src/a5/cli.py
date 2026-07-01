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


def _try_load_mesh_metadata(output_dir: Path) -> Optional[dict[str, np.ndarray]]:
    """Attempt to locate + parse SHUD mesh metadata (element areas, porosity).

    Water balance closure requires converting per-element flux rates
    (elevprcp, eleveta in m/s) to basin-integrated volumes. That in turn
    requires per-element area (m^2). Storage-change closure additionally
    requires per-element porosity to translate GW head change (m) to
    volume change.

    This function searches for the SHUD mesh files (`<case>.sp.dat`,
    `<case>.mesh`, `<case>.att.dat`) in `output_dir`, its parent, and its
    grand-parent. If found, it parses them and returns a dict with keys:

        element_area:  shape (n_ele,) float64, m^2 per element
        porosity:      shape (n_ele,) float64, dimensionless (optional)

    Returns None if any required file is missing OR if the SHUD .sp format
    parser is not yet implemented (see TODO(PR-Y3)). Callers must treat
    None as "cannot compute area-weighted water balance; fall back to NaN".

    Note (PR-Y2 scope): the .sp/.mesh binary format is non-trivial and
    varies across SHUD builds. This shipping version returns None
    unconditionally, wiring the safe NaN fallback. The Tier-2 parser is
    tracked as PR-Y3 (see docs/adr/future).
    """
    # TODO(PR-Y3): implement full .sp.mesh parser for area-weighted water
    # balance. For now, return None so callers use the safe NaN fallback.
    # Search paths: output_dir/*.sp.dat, output_dir/../*.sp.dat, etc.
    # Once implemented, the parser must return element_area (m^2) at
    # minimum, plus porosity if available.
    _ = output_dir  # explicit no-op to document intended future signature
    return None


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
    ref: dict[str, ShudSeries],
    cand: dict[str, ShudSeries],
    mesh_search_dir: Optional[Path] = None,
) -> tuple[dict[str, float], dict[str, str]]:
    """Compute the seven A5 metrics from paired reference / candidate series.

    Args:
        ref, cand:        SHUD output series loaded via `_load_case_series`.
        mesh_search_dir:  optional directory to search for mesh metadata
                          (.sp.dat / .att.dat / .mesh) needed for the
                          area-weighted water balance closure. When None or
                          when no metadata is found, water_balance_residual
                          is emitted as NaN (informational-only per verdict
                          logic). See `_try_load_mesh_metadata`.

    Returns:
        (metric_values, status)
            metric_values: dict[metric_name, float] passed to verdict.evaluate.
            status:        dict[metric_name, str] side-channel diagnostic
                           messages (e.g. `water_balance_status =
                           "unavailable_no_mesh_metadata"`). Not consumed by
                           verdict logic; surfaced in report JSON for audit.
    """
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
    status: dict[str, str] = {}

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

    # Water balance closure needs volume-consistent inputs (m^3 per timestep):
    # discharge at outlet integrated over dt, precip/ET area-integrated over
    # the basin, storage change (dGW * area * porosity) area-integrated.
    # That requires per-element area (and ideally porosity) from mesh
    # metadata. If mesh metadata is unavailable in the output tree, we
    # cannot produce a dimensionally meaningful residual — the pre-PR-Y2
    # implementation subtracted basin-mean rates from basin-mean discharge
    # which is dimensionally inconsistent and could blow up to 1e13-scale
    # nonsense (see PR-Z1 evidence for a concrete example). Safe fallback:
    # emit NaN + explicit `water_balance_status` sentinel so verdict logic
    # can downgrade the metric to informational.
    #
    # Tier-1 (this PR-Y2): fallback = NaN whenever mesh metadata absent.
    # Tier-2 (PR-Y3):      when metadata IS available, compute proper
    #                      area-weighted volumes and produce dimensionless
    #                      residual bounded to <= 0.05 for a well-posed run.
    mesh_meta = (
        _try_load_mesh_metadata(mesh_search_dir)
        if mesh_search_dir is not None
        else None
    )
    if mesh_meta is None:
        results["water_balance_residual"] = float("nan")
        status["water_balance_residual_status"] = "unavailable_no_mesh_metadata"
    elif not all(k in cand for k in ("elevprcp", "eleveta", "eleygw", "rivqdown")):
        results["water_balance_residual"] = float("nan")
        status["water_balance_residual_status"] = "unavailable_missing_series"
    else:
        # Tier-2 path (unreachable while _try_load_mesh_metadata returns None).
        # Left as a scaffold so PR-Y3 can wire the area-weighted volumes here
        # without changing the metric signature or verdict logic.
        try:
            area = mesh_meta["element_area"]  # m^2, shape (n_ele,)
            # Rate * area * dt (in seconds) -> m^3 per timestep. SHUD daily
            # cadence: dt = 86400 s. TODO(PR-Y3): honor DY.dat non-uniform dt.
            dt_sec = 86400.0
            n = min(
                cand["elevprcp"].values.shape[0],
                cand["eleveta"].values.shape[0],
                cand["eleygw"].values.shape[0],
                cand["rivqdown"].values.shape[0],
            )
            if n < 2:
                results["water_balance_residual"] = float("nan")
                status["water_balance_residual_status"] = "unavailable_short_series"
            else:
                p_vol = (cand["elevprcp"].values[:n] * area).sum(axis=1) * dt_sec
                e_vol = (cand["eleveta"].values[:n] * area).sum(axis=1) * dt_sec
                # Outlet discharge (m^3/s at last river column) * dt -> m^3/step
                q_vol = cand["rivqdown"].values[:n, -1] * dt_sec
                gw = cand["eleygw"].values[:n]
                porosity = mesh_meta.get(
                    "porosity", np.ones(gw.shape[1], dtype=np.float64)
                )
                dgw = np.diff(gw, axis=0, prepend=gw[:1])
                ds_vol = (dgw * area * porosity).sum(axis=1)
                results["water_balance_residual"] = _m.water_balance_residual(
                    q_vol, p_vol, e_vol, ds_vol
                )
                status["water_balance_residual_status"] = "computed"
        except (KeyError, ValueError, RuntimeError) as exc:
            results["water_balance_residual"] = float("nan")
            status["water_balance_residual_status"] = f"error:{type(exc).__name__}:{exc}"

    return results, status


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
        metric_values, metric_status = _compute_metrics(
            ref, cand, mesh_search_dir=args.reference
        )
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
        metric_status=metric_status,
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
