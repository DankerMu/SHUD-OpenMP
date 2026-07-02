"""Requirement-driven tests for the pinned-schema parsers + fail-closed.

Covers spec scenarios:
  - truncated CSV rejected (fail closed on malformed input)
  - header solverstep_min read from trace (never hardcoded)
  - schema validation (typed ele|riv rows, daily flips_total invariant)
"""

from __future__ import annotations

import pytest

from osc_diag.parsers import (
    MalformedTraceError,
    parse_dt_trace,
    parse_osc_flips,
    parse_osc_flips_daily,
)

from .conftest import (
    rows_for_delta_nst,
    write_dt_trace,
    write_osc_flips,
    write_osc_flips_daily,
)


# ---------------------------------------------------------------------------
# Happy path — real emitter shape round-trips
# ---------------------------------------------------------------------------

def test_parse_dt_trace_reads_header_and_rows(tmp_path):
    p = write_dt_trace(
        tmp_path / "diag_dt_trace.csv",
        rows_for_delta_nst([9, 6, 8]),
        solverstep_min=20.0,
        project_name="keliya",
    )
    trace = parse_dt_trace(p)
    assert trace.project_name == "keliya"
    assert trace.solverstep_min == 20.0
    assert trace.n_rows == 3
    assert trace.delta_nst.tolist() == [9, 6, 8]


def test_parse_osc_flips_typed_rows(tmp_path):
    p = write_osc_flips(
        tmp_path / "diag_osc_flips.csv",
        ele_family_rows=[(4, 0, 0), (0, 0, 0)],
        riv_stage=[23, 17, 21],
        project_name="keliya",
    )
    flips = parse_osc_flips(p)
    assert flips.n_ele == 2
    assert flips.n_riv == 3
    assert flips.ele_flips_total.tolist() == [4, 0]
    assert flips.riv_flips_stage.tolist() == [23, 17, 21]


def test_parse_daily_computes_and_validates_total(tmp_path):
    p = write_osc_flips_daily(
        tmp_path / "diag_osc_flips_daily.csv",
        [(12053, 743, 0, 0, 47), (12054, 723, 53, 0, 40)],
    )
    daily = parse_osc_flips_daily(p)
    assert daily.day_index.tolist() == [12053, 12054]
    assert daily.flips_total.tolist() == [790, 816]


# ---------------------------------------------------------------------------
# Fail closed — truncation / missing / malformed
# ---------------------------------------------------------------------------

def test_missing_trace_file_raises(tmp_path):
    with pytest.raises(MalformedTraceError, match="does not exist"):
        parse_dt_trace(tmp_path / "nope.csv")


def test_truncated_trace_header_only_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20\n"
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="truncated"):
        parse_dt_trace(p)


def test_trace_missing_solverstep_header_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text(
        "# project_name=keliya\n"
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur\n"
        "60.0,20,0,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="solverstep_min"):
        parse_dt_trace(p)


def test_trace_wrong_columns_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20\n"
        "t_next,delta_nst,WRONG\n"
        "60.0,20,0\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="column header"):
        parse_dt_trace(p)


def test_trace_nonnumeric_field_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20\n"
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur\n"
        "60.0,notanint,0,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="non-numeric"):
        parse_dt_trace(p)


def test_trace_negative_delta_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20\n"
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur\n"
        "60.0,-1,0,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="negative"):
        parse_dt_trace(p)


def test_flips_bad_entity_type_raises(tmp_path):
    p = tmp_path / "diag_osc_flips.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20 epsilon_m=1e-06\n"
        "entity_type,entity_id,flips_sf,flips_us,flips_gw,flips_stage\n"
        "lake,1,0,0,0,0\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="entity_type"):
        parse_osc_flips(p)


def test_flips_ele_row_with_nonzero_stage_raises(tmp_path):
    p = tmp_path / "diag_osc_flips.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20 epsilon_m=1e-06\n"
        "entity_type,entity_id,flips_sf,flips_us,flips_gw,flips_stage\n"
        "ele,1,4,0,0,9\n",
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="flips_stage"):
        parse_osc_flips(p)


def test_daily_total_mismatch_raises(tmp_path):
    p = tmp_path / "diag_osc_flips_daily.csv"
    p.write_text(
        "# project_name=keliya solverstep_min=20 epsilon_m=1e-06\n"
        "day_index,flips_sf,flips_us,flips_gw,flips_stage,flips_total\n"
        "12053,743,0,0,47,999\n",  # 999 != 790
        encoding="utf-8",
    )
    with pytest.raises(MalformedTraceError, match="flips_total"):
        parse_osc_flips_daily(p)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "diag_dt_trace.csv"
    p.write_text("", encoding="utf-8")
    with pytest.raises(MalformedTraceError, match="empty"):
        parse_dt_trace(p)
