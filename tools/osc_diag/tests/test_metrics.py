"""Requirement-driven tests for the analysis metrics.

Covers spec scenarios:
  - burst share on synthetic fixture (known 60% -> within 1% of 0.60)
  - concentration on synthetic fixture (3/300 carry 70% -> within 1% of 0.70)
  - correlation on synthetic fixture (known rho: monotone=1.0, shuffled~0)
  - delta_nst=0 degenerate-row handling (0 contribution to numerator + denom)
  - histogram consistency (counts sum to non-skipped rows; shares sum to 1)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from osc_diag import metrics as _m
from osc_diag.parsers import parse_dt_trace, parse_osc_flips, parse_osc_flips_daily

from .conftest import rows_for_delta_nst


# ---------------------------------------------------------------------------
# mean-dt formula (dt-step-trace spec: solverstep_min=20, delta_nst=20 -> 60s)
# ---------------------------------------------------------------------------

def test_mean_dt_formula_60s(make_trace):
    p = make_trace(rows_for_delta_nst([20]), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    dt = _m.mean_dt_seconds(trace.solverstep_min, trace.delta_nst)
    assert dt[0] == pytest.approx(60.0)


def test_mean_dt_reads_solverstep_from_header_not_hardcoded(make_trace):
    # A different solverstep_min in the header must change the reported dt.
    p = make_trace(rows_for_delta_nst([10]), solverstep_min=30.0)
    trace = parse_dt_trace(p)
    # 60 * 30 / 10 = 180 s
    dt = _m.mean_dt_seconds(trace.solverstep_min, trace.delta_nst)
    assert dt[0] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# burst share
# ---------------------------------------------------------------------------

def test_burst_share_60_percent_within_1pct(make_trace):
    # solverstep_min=20 => mean_dt<60s iff delta_nst>20.
    # 10 burst intervals @ delta_nst=30 (mean_dt=40s) carry 300 nst.
    # 10 non-burst @ delta_nst=20 (mean_dt=60s, NOT <60) carry 200 nst.
    # burst_share_60s = 300 / 500 = 0.60 exactly.
    deltas = [30] * 10 + [20] * 10
    p = make_trace(rows_for_delta_nst(deltas), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    bs = _m.burst_share(trace, _m.BURST_THRESHOLD_60S)
    assert abs(bs - 0.60) <= 0.01


def test_burst_share_10s_threshold(make_trace):
    # mean_dt < 10s iff delta_nst > 120 (60*20/10). One interval @ 240 nst
    # (mean_dt=5s) + one @ 20 nst (60s). burst_10s = 240/260 ~ 0.923.
    deltas = [240, 20]
    p = make_trace(rows_for_delta_nst(deltas), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    bs10 = _m.burst_share(trace, _m.BURST_THRESHOLD_10S)
    assert bs10 == pytest.approx(240.0 / 260.0, abs=1e-9)


def test_burst_share_boundary_is_strict_less_than(make_trace):
    # delta_nst=20 => mean_dt exactly 60s => NOT < 60 => not a burst.
    p = make_trace(rows_for_delta_nst([20, 20]), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    assert _m.burst_share(trace, _m.BURST_THRESHOLD_60S) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# delta_nst == 0 degenerate rows
# ---------------------------------------------------------------------------

def test_delta_nst_zero_rows_excluded_from_burst_and_denominator(make_trace):
    # Same 60% construction, plus two delta_nst=0 rows. The skip rule means
    # burst_share stays exactly 0.60 (0 contribution to numerator AND denom).
    deltas = [30] * 10 + [20] * 10 + [0, 0]
    p = make_trace(rows_for_delta_nst(deltas), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    bs = _m.burst_share(trace, _m.BURST_THRESHOLD_60S)
    assert bs == pytest.approx(0.60, abs=1e-9)


def test_delta_nst_zero_rows_excluded_from_histogram_counts(make_trace):
    deltas = [30, 20, 0, 0, 0]
    p = make_trace(rows_for_delta_nst(deltas), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    hist = _m.compute_histogram(trace)
    # Only 2 non-skipped rows contribute.
    assert hist.total_intervals == 2
    assert hist.interval_count_sum() == 2


def test_delta_nst_zero_rows_excluded_from_daily_sub60_count(make_trace):
    # One sub-60 (delta=30) + one delta_nst=0, both in day 0. Count = 1.
    p = make_trace(rows_for_delta_nst([30, 0]), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    counts = _m.daily_sub60_interval_count(trace)
    assert counts == {0: 1}


# ---------------------------------------------------------------------------
# histogram consistency (spec scenario)
# ---------------------------------------------------------------------------

def test_histogram_counts_sum_to_nonskipped_rows_and_shares_sum_to_one(make_trace):
    rng = np.random.default_rng(7)
    deltas = rng.integers(low=1, high=400, size=200).tolist()
    deltas += [0] * 13  # degenerate rows the histogram must ignore
    p = make_trace(rows_for_delta_nst(deltas), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    hist = _m.compute_histogram(trace)

    assert hist.interval_count_sum() == 200  # non-skipped rows only
    assert hist.total_intervals == 200
    assert hist.nst_share_sum() == pytest.approx(1.0, abs=1e-9)


def test_histogram_bin_labels_and_edges(make_trace):
    p = make_trace(rows_for_delta_nst([5]), solverstep_min=20.0)
    trace = parse_dt_trace(p)
    hist = _m.compute_histogram(trace)
    labels = [b.label for b in hist.bins]
    assert labels == [
        "[0,10)",
        "[10,60)",
        "[60,300)",
        "[300,600)",
        "[600,1200)",
        "[1200,inf)",
    ]


# ---------------------------------------------------------------------------
# concentration (spec scenario: 3 of 300 carry 70%)
# ---------------------------------------------------------------------------

def test_concentration_70pct_within_1pct(make_flips):
    # 300 elements. top_k = ceil(0.01*300) = 3. Put 700 flips across the 3
    # heaviest elements and 300 flips across the other 297 -> 0.70.
    ele = []
    # 3 heavy elements: (sf, us, gw) summing to ~233 each (700 total).
    heavy = [(233, 0, 0), (233, 0, 0), (234, 0, 0)]
    ele.extend(heavy)
    # 297 light elements sharing 300 flips (~1 each, remainder on the first).
    for i in range(297):
        ele.append((2 if i < 3 else 1, 0, 0))  # 3*2 + 294*1 = 300
    p = make_flips(ele)
    flips = parse_osc_flips(p)
    res = _m.flip_concentration(flips)
    assert res.n_ele == 300
    assert res.top_k == 3
    assert res.total_ele_flips == 1000
    assert abs(res.top1pct_concentration - 0.70) <= 0.01


def test_concentration_excludes_rivers_from_denominator(make_flips):
    # Rivers carry huge flip counts but must NOT enter the concentration
    # population or denominator. Element concentration must be unaffected.
    ele = [(50, 0, 0), (50, 0, 0)] + [(0, 0, 0)] * 98  # 100 ele, 100 flips
    riv = [10000, 20000]  # rivers reported separately
    p = make_flips(ele, riv_stage=riv)
    flips = parse_osc_flips(p)
    res = _m.flip_concentration(flips)
    assert res.n_ele == 100
    assert res.total_ele_flips == 100
    assert res.riv_flips_total == 30000
    # top_k = ceil(0.01*100) = 1 -> the single heaviest (50) / 100 = 0.50.
    assert res.top_k == 1
    assert res.top1pct_concentration == pytest.approx(0.50)


def test_concentration_sums_families_per_element(make_flips):
    # per-element flips = sf+us+gw summed (spec).
    ele = [(10, 20, 30)] + [(0, 0, 0)] * 99  # element 1 carries 60
    p = make_flips(ele)
    flips = parse_osc_flips(p)
    res = _m.flip_concentration(flips)
    assert res.total_ele_flips == 60
    assert res.top_element_flips[0] == 60


def test_concentration_zero_flips_is_zero_not_nan(make_flips):
    ele = [(0, 0, 0)] * 50
    p = make_flips(ele)
    res = _m.flip_concentration(parse_osc_flips(p))
    assert res.top1pct_concentration == 0.0


# ---------------------------------------------------------------------------
# Spearman correlation (spec scenario: known rho)
# ---------------------------------------------------------------------------

def _daily_rows_and_matching_trace(flips_by_day, sub60_by_day, solverstep_min=20.0):
    """Build daily rows (flips_sf carries the total) + a trace whose per-day
    sub-60s interval counts equal `sub60_by_day`.

    An extra trailing PARTIAL day is appended to both sides so the fencepost
    drop is exercised; the correlation target is computed over the complete
    days only.
    """
    days = sorted(flips_by_day)
    day_rows = [(d, flips_by_day[d], 0, 0, 0) for d in days]

    trace_rows = []
    for d in days:
        n_sub60 = sub60_by_day[d]
        # sub-60 intervals: delta_nst=30 (mean_dt=40s). Plus one non-burst
        # (delta_nst=20 => 60s) so not every day is all-burst.
        for _ in range(n_sub60):
            trace_rows.append(30)
        trace_rows.append(20)
    return day_rows, days, trace_rows


def test_spearman_monotone_rho_is_one(make_trace, make_daily):
    # Monotone-increasing flips AND sub60 counts across days -> rho = 1.0.
    flips_by_day = {10: 100, 11: 200, 12: 300, 13: 400, 14: 500}
    sub60_by_day = {10: 1, 11: 2, 12: 3, 13: 4, 14: 5}
    day_rows, days, per_day_deltas = _daily_rows_and_matching_trace(
        flips_by_day, sub60_by_day
    )

    # Build a trace covering these days, then append a trailing partial day.
    trace_rows = []
    for d in days:
        n_sub60 = sub60_by_day[d]
        base = d * _m.MINUTES_PER_DAY + 60.0
        k = 0
        for _ in range(n_sub60):
            trace_rows.append((base + k * 20.0, 30))
            k += 1
        trace_rows.append((base + k * 20.0, 20))
    # Trailing partial day (day 15): single interval landing on the boundary.
    trace_rows.append((15 * _m.MINUTES_PER_DAY, 4))
    day_rows.append((15, 4, 0, 0, 0))

    tp = make_trace(trace_rows, solverstep_min=20.0)
    dp = make_daily(day_rows, solverstep_min=20.0)
    trace = parse_dt_trace(tp)
    daily = parse_osc_flips_daily(dp)

    res = _m.daily_flip_dt_spearman(trace, daily)
    assert res.dropped_trailing_day == 15
    assert res.n_days == 5  # the 5 complete days
    assert res.rho == pytest.approx(1.0, abs=0.05)


def test_spearman_shuffled_rho_near_zero(make_trace, make_daily):
    # Anti-correlated construction: flips increase with day while sub60
    # counts DECREASE -> rho = -1.0 (|reported - constructed| <= 0.05).
    flips_by_day = {d: 100 * (d - 9) for d in range(10, 20)}  # increasing
    sub60_by_day = {d: (20 - d) for d in range(10, 20)}       # decreasing

    days = sorted(flips_by_day)
    day_rows = [(d, flips_by_day[d], 0, 0, 0) for d in days]
    trace_rows = []
    for d in days:
        base = d * _m.MINUTES_PER_DAY + 60.0
        k = 0
        for _ in range(sub60_by_day[d]):
            trace_rows.append((base + k * 20.0, 30))
            k += 1
        trace_rows.append((base + k * 20.0, 20))

    tp = make_trace(trace_rows, solverstep_min=20.0)
    dp = make_daily(day_rows, solverstep_min=20.0)
    res = _m.daily_flip_dt_spearman(
        parse_dt_trace(tp), parse_osc_flips_daily(dp), drop_trailing_partial_day=False
    )
    assert res.rho == pytest.approx(-1.0, abs=0.05)


def test_spearman_constant_series_is_nan(make_trace, make_daily):
    # Constant sub60 counts -> rank variance zero -> Spearman undefined.
    days = [10, 11, 12]
    day_rows = [(d, 100 * (d - 9), 0, 0, 0) for d in days]
    trace_rows = []
    for d in days:
        base = d * _m.MINUTES_PER_DAY + 60.0
        # exactly one sub-60 interval each day -> constant y series
        trace_rows.append((base, 30))
    tp = make_trace(trace_rows, solverstep_min=20.0)
    dp = make_daily(day_rows, solverstep_min=20.0)
    res = _m.daily_flip_dt_spearman(
        parse_dt_trace(tp), parse_osc_flips_daily(dp), drop_trailing_partial_day=False
    )
    assert math.isnan(res.rho)


# ---------------------------------------------------------------------------
# Fencepost handling (PR-D1 reviewer finding)
# ---------------------------------------------------------------------------

def test_fencepost_trailing_partial_day_dropped_symmetrically(make_trace, make_daily):
    # Two complete days + a trailing partial day. The trailing bucket must be
    # dropped from BOTH join sides; the correlation uses the 2 complete days.
    day_rows = [(10, 100, 0, 0, 0), (11, 200, 0, 0, 0), (12, 4, 0, 0, 0)]
    trace_rows = [
        (10 * _m.MINUTES_PER_DAY + 60.0, 30),
        (10 * _m.MINUTES_PER_DAY + 80.0, 30),
        (11 * _m.MINUTES_PER_DAY + 60.0, 30),
        (12 * _m.MINUTES_PER_DAY, 4),  # trailing partial-day interval
    ]
    tp = make_trace(trace_rows, solverstep_min=20.0)
    dp = make_daily(day_rows, solverstep_min=20.0)
    res = _m.daily_flip_dt_spearman(parse_dt_trace(tp), parse_osc_flips_daily(dp))
    assert res.dropped_trailing_day == 12
    assert res.n_days == 2
