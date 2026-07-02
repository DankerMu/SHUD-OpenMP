"""osc_diag CLI entry point.

Usage:
    osc_diag --diag-dir DIR --case CASE --out OUTDIR
    osc_diag --trace T.csv --flips F.csv --flips-daily D.csv --case CASE --out OUTDIR

Consumes the three P11-osc diagnostic CSVs (emitted by SHUD with
SHUD_DIAG_DT=1 SHUD_DIAG_OSC=1), computes the gate inputs (burst share,
top-1% element flip concentration, per-day Spearman rho), writes
`dt_histogram.csv` + `osc_diag_summary.json` into --out, and prints the
`MARKER:OSC_DIAG_VERDICT_BEGIN..END` block to stdout.

--case is REQUIRED and explicit because the trace header carries
`project_name` (the ./shud argument, e.g. `nanlin`), which differs from the
benchmark case name (`qinyijiang`). All outputs are labeled with --case.

FAIL CLOSED: any missing / truncated / unparsable input raises
MalformedTraceError -> non-zero exit (EXIT_MALFORMED_INPUT) and NO MARKER
block is written. A verdict is never produced from partial data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import metrics as _m
from .parsers import (
    MalformedTraceError,
    parse_dt_trace,
    parse_osc_flips,
    parse_osc_flips_daily,
)
from .report import build_summary, write_histogram_csv, write_summary_json
from .verdict import decide, format_marker_block

EXIT_OK = 0
EXIT_MALFORMED_INPUT = 2

# Default diag CSV filenames (as emitted by MD_osc_diag.hpp).
_TRACE_NAME = "diag_dt_trace.csv"
_FLIPS_NAME = "diag_osc_flips.csv"
_DAILY_NAME = "diag_osc_flips_daily.csv"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="osc_diag",
        description=(
            "P11-osc CVODE stepping oscillation analyzer. Consumes the three "
            "SHUD diagnostic CSVs and emits a machine-readable "
            "MARKER:OSC_DIAG_VERDICT block plus dt_histogram.csv."
        ),
    )
    p.add_argument(
        "--case",
        required=True,
        help=(
            "Benchmark case name written into all outputs. Explicit because "
            "the trace header carries project_name (e.g. nanlin), which "
            "differs from the benchmark case (qinyijiang)."
        ),
    )
    p.add_argument(
        "--diag-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing the three diag CSVs "
            f"({_TRACE_NAME} / {_FLIPS_NAME} / {_DAILY_NAME}). Mutually "
            "exclusive convenience for the three explicit --trace/--flips/"
            "--flips-daily flags."
        ),
    )
    p.add_argument("--trace", type=Path, default=None, help=f"Path to {_TRACE_NAME}.")
    p.add_argument("--flips", type=Path, default=None, help=f"Path to {_FLIPS_NAME}.")
    p.add_argument(
        "--flips-daily", type=Path, default=None, help=f"Path to {_DAILY_NAME}."
    )
    p.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for dt_histogram.csv + osc_diag_summary.json.",
    )
    return p


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    """Resolve the three CSV paths from --diag-dir and/or explicit flags.

    Explicit flags override the --diag-dir default per file. Raises
    MalformedTraceError (fail closed) if any of the three paths cannot be
    determined.
    """
    base = args.diag_dir
    trace = args.trace or (base / _TRACE_NAME if base else None)
    flips = args.flips or (base / _FLIPS_NAME if base else None)
    daily = args.flips_daily or (base / _DAILY_NAME if base else None)

    missing = [
        name
        for name, val in (
            ("--trace/--diag-dir", trace),
            ("--flips/--diag-dir", flips),
            ("--flips-daily/--diag-dir", daily),
        )
        if val is None
    ]
    if missing:
        raise MalformedTraceError(
            "input paths not fully specified; provide --diag-dir or all of "
            f"--trace/--flips/--flips-daily (unresolved: {', '.join(missing)})"
        )
    return trace, flips, daily


def run(args: argparse.Namespace) -> int:
    try:
        trace_path, flips_path, daily_path = _resolve_inputs(args)
        trace = parse_dt_trace(trace_path)
        flips = parse_osc_flips(flips_path)
        daily = parse_osc_flips_daily(daily_path)
    except MalformedTraceError as exc:
        # FAIL CLOSED: no MARKER block, non-zero exit.
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_MALFORMED_INPUT

    # All three parsed successfully — compute gate inputs. From here on the
    # inputs are structurally valid; metrics degrade gracefully (e.g. NaN
    # rho) rather than fabricating a verdict.
    hist = _m.compute_histogram(trace)
    bs60 = _m.burst_share(trace, _m.BURST_THRESHOLD_60S)
    bs10 = _m.burst_share(trace, _m.BURST_THRESHOLD_10S)
    conc = _m.flip_concentration(flips)
    spear = _m.daily_flip_dt_spearman(trace, daily)

    result = decide(
        burst_share_60s=bs60,
        burst_share_10s=bs10,
        top1pct_concentration=conc.top1pct_concentration,
        rho=spear.rho,
    )

    rows_skipped = int((trace.delta_nst == 0).sum())
    summary = build_summary(
        case=args.case,
        project_name=trace.project_name,
        solverstep_min=trace.solverstep_min,
        trace_rows=trace.n_rows,
        trace_rows_skipped=rows_skipped,
        hist=hist,
        concentration=conc,
        spearman=spear,
        verdict=result,
    )
    write_histogram_csv(args.out, hist)
    write_summary_json(args.out, summary)

    marker = format_marker_block(args.case, result)
    sys.stdout.write(marker)
    sys.stdout.flush()
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
