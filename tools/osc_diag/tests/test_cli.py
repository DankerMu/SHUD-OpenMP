"""End-to-end CLI tests: exit codes, MARKER emission, artifacts, fail-closed.

Covers spec scenarios:
  - histogram written and consistent (dt_histogram.csv exists + sums)
  - truncated CSV rejected -> non-zero exit AND no MARKER:OSC_DIAG_VERDICT_BEGIN
  - --case labels outputs (project_name != case name)
"""

from __future__ import annotations

import json

import pytest

from osc_diag.cli import EXIT_MALFORMED_INPUT, EXIT_OK, main

from .conftest import (
    write_dt_trace,
    write_osc_flips,
    write_osc_flips_daily,
)

MINUTES_PER_DAY = 1440.0


def _build_valid_diag_dir(dir_path, project_name="nanlin"):
    """Write a minimally valid trio of diag CSVs into dir_path."""
    dir_path.mkdir(parents=True, exist_ok=True)
    # Two complete days + a trailing partial day for the fencepost path.
    trace_rows = [
        (10 * MINUTES_PER_DAY + 60.0, 30),
        (10 * MINUTES_PER_DAY + 80.0, 20),
        (11 * MINUTES_PER_DAY + 60.0, 30),
        (12 * MINUTES_PER_DAY, 4),  # trailing partial-day interval
    ]
    write_dt_trace(
        dir_path / "diag_dt_trace.csv",
        trace_rows,
        solverstep_min=20.0,
        project_name=project_name,
    )
    write_osc_flips(
        dir_path / "diag_osc_flips.csv",
        ele_family_rows=[(5, 0, 0), (1, 0, 0), (0, 0, 0)],
        riv_stage=[3, 2],
        project_name=project_name,
    )
    write_osc_flips_daily(
        dir_path / "diag_osc_flips_daily.csv",
        [(10, 100, 0, 0, 0), (11, 50, 0, 0, 0), (12, 4, 0, 0, 0)],
        project_name=project_name,
    )
    return dir_path


def test_cli_happy_path_exit0_and_marker(tmp_path, capsys):
    diag = _build_valid_diag_dir(tmp_path / "diag")
    out = tmp_path / "out"
    rc = main(["--diag-dir", str(diag), "--case", "qinyijiang", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == EXIT_OK
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" in captured.out
    assert "MARKER:OSC_DIAG_VERDICT_END" in captured.out
    assert "case=qinyijiang" in captured.out  # --case, NOT project_name


def test_cli_writes_histogram_and_summary_artifacts(tmp_path, capsys):
    diag = _build_valid_diag_dir(tmp_path / "diag")
    out = tmp_path / "out"
    rc = main(["--diag-dir", str(diag), "--case", "keliya", "--out", str(out)])
    assert rc == EXIT_OK

    hist_path = out / "dt_histogram.csv"
    summary_path = out / "osc_diag_summary.json"
    assert hist_path.exists()
    assert summary_path.exists()

    # Histogram consistency: interval counts sum to non-skipped rows (4),
    # nst shares sum to 1.
    lines = hist_path.read_text().strip().splitlines()
    assert lines[0] == "bin_lo_s,bin_hi_s,interval_count,nst_share"
    counts = [int(ln.split(",")[2]) for ln in lines[1:]]
    shares = [float(ln.split(",")[3]) for ln in lines[1:]]
    assert sum(counts) == 4
    assert sum(shares) == pytest.approx(1.0, abs=1e-9)

    summary = json.loads(summary_path.read_text())
    assert summary["case"] == "keliya"
    assert summary["project_name"] == "nanlin"  # header project name preserved
    assert "proxy_limitation" in summary
    assert summary["spearman_detail"]["dropped_trailing_partial_day_index"] == 12


def test_cli_truncated_trace_fails_closed_no_marker(tmp_path, capsys):
    diag = _build_valid_diag_dir(tmp_path / "diag")
    # Truncate the trace to header-only (no data rows).
    (diag / "diag_dt_trace.csv").write_text(
        "# project_name=nanlin solverstep_min=20\n"
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    rc = main(["--diag-dir", str(diag), "--case", "keliya", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == EXIT_MALFORMED_INPUT
    assert rc != 0
    # FAIL CLOSED: no MARKER block on stdout.
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" not in captured.out


def test_cli_missing_file_fails_closed_no_marker(tmp_path, capsys):
    diag = _build_valid_diag_dir(tmp_path / "diag")
    (diag / "diag_osc_flips.csv").unlink()
    out = tmp_path / "out"
    rc = main(["--diag-dir", str(diag), "--case", "keliya", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == EXIT_MALFORMED_INPUT
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" not in captured.out


def test_cli_unresolved_inputs_fails_closed(tmp_path, capsys):
    # Neither --diag-dir nor the explicit trio -> fail closed, no MARKER.
    out = tmp_path / "out"
    rc = main(["--case", "keliya", "--out", str(out)])
    captured = capsys.readouterr()
    assert rc == EXIT_MALFORMED_INPUT
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" not in captured.out


def test_cli_explicit_paths_override_diag_dir(tmp_path, capsys):
    diag = _build_valid_diag_dir(tmp_path / "diag")
    out = tmp_path / "out"
    rc = main(
        [
            "--trace", str(diag / "diag_dt_trace.csv"),
            "--flips", str(diag / "diag_osc_flips.csv"),
            "--flips-daily", str(diag / "diag_osc_flips_daily.csv"),
            "--case", "keliya",
            "--out", str(out),
        ]
    )
    captured = capsys.readouterr()
    assert rc == EXIT_OK
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" in captured.out


# ---------------------------------------------------------------------------
# All-three-verdict-branches via the CLI (end-to-end construction)
# ---------------------------------------------------------------------------

def _write_burst_localized_trace(dir_path, sub60_by_day, project_name="nanlin"):
    """Trace with a high 60s burst share (delta_nst=240 => mean_dt=5s)."""
    dir_path.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, n in sub60_by_day.items():
        base = d * MINUTES_PER_DAY + 60.0
        for k in range(n):
            rows.append((base + k * 20.0, 240))  # mean_dt 5s -> burst
    # trailing partial day
    maxd = max(sub60_by_day) + 1
    rows.append((maxd * MINUTES_PER_DAY, 4))
    write_dt_trace(dir_path / "diag_dt_trace.csv", rows,
                   solverstep_min=20.0, project_name=project_name)
    return maxd


def test_cli_verdict_confirmed_branch(tmp_path, capsys):
    diag = tmp_path / "diag"
    sub60 = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5}
    maxd = _write_burst_localized_trace(diag, sub60)
    # Localized flips: element 1 carries all flips (concentration = 1.0).
    write_osc_flips(diag / "diag_osc_flips.csv",
                    ele_family_rows=[(1000, 0, 0)] + [(0, 0, 0)] * 99)
    # Daily flips monotone with sub60 count -> rho = 1.0. Plus trailing day.
    day_rows = [(d, 100 * n, 0, 0, 0) for d, n in sub60.items()]
    day_rows.append((maxd, 4, 0, 0, 0))
    write_osc_flips_daily(diag / "diag_osc_flips_daily.csv", day_rows)

    rc = main(["--diag-dir", str(diag), "--case", "c", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "verdict=OSC_CONFIRMED" in out


def test_cli_verdict_real_dynamics_branch(tmp_path, capsys):
    # High burst but uniform flips (concentration < 0.50) -> REAL_DYNAMICS.
    diag = tmp_path / "diag"
    sub60 = {10: 3, 11: 3, 12: 3}
    maxd = _write_burst_localized_trace(diag, sub60)
    # 100 elements each carrying 10 flips -> top-1% (1 ele) = 10/1000 = 0.01.
    write_osc_flips(diag / "diag_osc_flips.csv",
                    ele_family_rows=[(10, 0, 0)] * 100)
    day_rows = [(d, 100 * n, 0, 0, 0) for d, n in sub60.items()]
    day_rows.append((maxd, 4, 0, 0, 0))
    write_osc_flips_daily(diag / "diag_osc_flips_daily.csv", day_rows)

    rc = main(["--diag-dir", str(diag), "--case", "c", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "verdict=REAL_DYNAMICS" in out


def test_cli_verdict_inconclusive_branch(tmp_path, capsys):
    # High burst AND localized flips, but flips ANTI-correlated with sub60
    # counts -> rho < 0.5 -> INCONCLUSIVE.
    diag = tmp_path / "diag"
    sub60 = {d: (20 - d) for d in range(10, 20)}  # decreasing
    maxd = _write_burst_localized_trace(diag, sub60)
    write_osc_flips(diag / "diag_osc_flips.csv",
                    ele_family_rows=[(1000, 0, 0)] + [(0, 0, 0)] * 99)
    # flips INCREASING while sub60 DECREASING -> rho = -1.0.
    day_rows = [(d, 100 * (d - 9), 0, 0, 0) for d in range(10, 20)]
    day_rows.append((maxd, 4, 0, 0, 0))
    write_osc_flips_daily(diag / "diag_osc_flips_daily.csv", day_rows)

    rc = main(["--diag-dir", str(diag), "--case", "c", "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == EXIT_OK
    assert "verdict=INCONCLUSIVE" in out
