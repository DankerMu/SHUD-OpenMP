"""Requirement-driven tests for the TOTAL verdict decision function + MARKER.

Covers spec scenarios:
  - verdict logic — confirmed
  - verdict logic — real dynamics
  - verdict logic — inconclusive
  - totality (every input combination maps to exactly one verdict)
"""

from __future__ import annotations

import math

from osc_diag.verdict import (
    VERDICT_CONFIRMED,
    VERDICT_INCONCLUSIVE,
    VERDICT_REAL_DYNAMICS,
    decide,
    format_marker_block,
)


# ---------------------------------------------------------------------------
# The three branches (spec scenarios)
# ---------------------------------------------------------------------------

def test_verdict_confirmed():
    # burst >= 0.50 AND conc >= 0.50 AND rho >= 0.5
    r = decide(burst_share_60s=0.60, burst_share_10s=0.30,
               top1pct_concentration=0.55, rho=0.7)
    assert r.verdict == VERDICT_CONFIRMED


def test_verdict_real_dynamics_low_burst():
    # burst < 0.50 (regardless of conc / rho)
    r = decide(burst_share_60s=0.40, burst_share_10s=0.10,
               top1pct_concentration=0.99, rho=0.99)
    assert r.verdict == VERDICT_REAL_DYNAMICS


def test_verdict_real_dynamics_low_concentration():
    # burst >= 0.50 but conc < 0.50 (spatially ~uniform flips)
    r = decide(burst_share_60s=0.80, burst_share_10s=0.50,
               top1pct_concentration=0.40, rho=0.99)
    assert r.verdict == VERDICT_REAL_DYNAMICS


def test_verdict_inconclusive():
    # burst >= 0.50 AND conc >= 0.50 AND rho < 0.5
    r = decide(burst_share_60s=0.80, burst_share_10s=0.50,
               top1pct_concentration=0.60, rho=0.30)
    assert r.verdict == VERDICT_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Boundary behavior (thresholds are >=, strict for rho gate)
# ---------------------------------------------------------------------------

def test_verdict_boundary_all_exactly_at_threshold_is_confirmed():
    # 0.50 / 0.50 / 0.5 all satisfy the >= gates -> CONFIRMED.
    r = decide(0.50, 0.10, 0.50, 0.5)
    assert r.verdict == VERDICT_CONFIRMED


def test_verdict_burst_just_below_threshold_is_real_dynamics():
    r = decide(0.4999, 0.10, 0.99, 0.99)
    assert r.verdict == VERDICT_REAL_DYNAMICS


def test_verdict_nan_rho_counts_as_below_threshold_inconclusive():
    # NaN rho fails the rho >= 0.5 gate; bursty+localized -> INCONCLUSIVE
    # (never CONFIRMED). Conservative per design R3.
    r = decide(0.80, 0.50, 0.60, math.nan)
    assert r.verdict == VERDICT_INCONCLUSIVE


# ---------------------------------------------------------------------------
# Totality: every combination lands in exactly one verdict, no catch-all
# ---------------------------------------------------------------------------

def test_verdict_is_total_over_grid():
    valid = {VERDICT_CONFIRMED, VERDICT_REAL_DYNAMICS, VERDICT_INCONCLUSIVE}
    values = [0.0, 0.4999, 0.50, 0.5001, 1.0]
    rhos = [-1.0, 0.0, 0.4999, 0.5, 0.5001, 1.0, math.nan]
    for b in values:
        for c in values:
            for rho in rhos:
                v = decide(b, 0.0, c, rho).verdict
                assert v in valid
                # Cross-check the region logic explicitly.
                if b < 0.50 or c < 0.50:
                    assert v == VERDICT_REAL_DYNAMICS
                elif (not (isinstance(rho, float) and math.isnan(rho))) and rho >= 0.5:
                    assert v == VERDICT_CONFIRMED
                else:
                    assert v == VERDICT_INCONCLUSIVE


# ---------------------------------------------------------------------------
# MARKER block
# ---------------------------------------------------------------------------

def test_marker_block_byte_exact():
    r = decide(0.6123, 0.3050, 0.5500, 0.7200)
    out = format_marker_block("qinyijiang", r)
    expected = (
        "MARKER:OSC_DIAG_VERDICT_BEGIN\n"
        "case=qinyijiang\n"
        "verdict=OSC_CONFIRMED\n"
        "burst_share_60s=0.6123\n"
        "burst_share_60s_threshold=0.5000\n"
        "burst_share_10s=0.3050\n"
        "top1pct_concentration=0.5500\n"
        "top1pct_concentration_threshold=0.5000\n"
        "spearman_rho=0.7200\n"
        "spearman_rho_threshold=0.5000\n"
        "MARKER:OSC_DIAG_VERDICT_END\n"
    )
    assert out == expected


def test_marker_block_nan_rho_renders_as_string():
    r = decide(0.80, 0.50, 0.60, math.nan)
    out = format_marker_block("keliya", r)
    assert "spearman_rho=NaN" in out
    assert "verdict=INCONCLUSIVE" in out
    assert "MARKER:OSC_DIAG_VERDICT_BEGIN" in out
    assert "MARKER:OSC_DIAG_VERDICT_END" in out


def test_marker_block_contains_all_gate_inputs_and_thresholds():
    r = decide(0.55, 0.20, 0.51, 0.6)
    out = format_marker_block("c", r)
    for key in (
        "burst_share_60s=",
        "burst_share_60s_threshold=",
        "burst_share_10s=",
        "top1pct_concentration=",
        "top1pct_concentration_threshold=",
        "spearman_rho=",
        "spearman_rho_threshold=",
    ):
        assert key in out
