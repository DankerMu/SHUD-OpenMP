"""Requirement-driven tests for the seven A5 hydrology metrics.

Coverage map (see SKILL.md `<testing>` rubric):
    happy path:   perfect match -> ideal value
    edge:         constant reference (NSE denominator = 0)
    edge:         proportional cand for peak-magnitude and volume ratios
    edge:         symmetric timing offsets
    error:        shape mismatch, empty input
    error:        zero total precipitation in water balance

Each function's behavior is anchored to a small closed-form example so
regressions are easy to diagnose.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from a5 import metrics as m


# ---------- NSE ---------------------------------------------------------------


def test_nse_perfect_match() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.nse(ref, ref) == pytest.approx(1.0)


def test_nse_less_than_one_for_perturbed_series() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = ref + 0.5  # constant offset
    val = m.nse(ref, cand)
    assert val < 1.0
    # Closed-form: numerator = 5*0.25 = 1.25, denom = 10 -> NSE = 0.875
    assert val == pytest.approx(0.875, abs=1e-9)


def test_nse_constant_reference_returns_nan() -> None:
    ref = np.full(5, 3.0)
    cand = np.array([3.0, 3.5, 2.9, 3.1, 3.2])
    assert math.isnan(m.nse(ref, cand))


def test_nse_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        m.nse(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))


# ---------- KGE ---------------------------------------------------------------


def test_kge_perfect_match() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert m.kge(ref, ref) == pytest.approx(1.0)


def test_kge_zero_std_reference_returns_nan() -> None:
    ref = np.full(5, 2.0)
    cand = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isnan(m.kge(ref, cand))


def test_kge_scaled_candidate_gives_expected_alpha() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    cand = 2.0 * ref  # r=1, beta=2, alpha=2 -> KGE = 1 - sqrt(0 + 1 + 1) = 1 - sqrt(2)
    val = m.kge(ref, cand)
    assert val == pytest.approx(1.0 - math.sqrt(2.0), abs=1e-9)


# ---------- Peak magnitude ratio ---------------------------------------------


def test_peak_magnitude_ratio_scaled() -> None:
    ref = np.array([1.0, 5.0, 3.0])
    cand = 1.1 * ref
    assert m.peak_magnitude_ratio(ref, cand) == pytest.approx(1.1)


def test_peak_magnitude_ratio_zero_reference_returns_nan() -> None:
    ref = np.zeros(3)
    cand = np.array([1.0, 2.0, 3.0])
    assert math.isnan(m.peak_magnitude_ratio(ref, cand))


# ---------- Peak timing offset -----------------------------------------------


def test_peak_timing_offset_positive() -> None:
    ref = np.array([0.0, 5.0, 0.0, 0.0])
    cand = np.array([0.0, 0.0, 5.0, 0.0])  # peaks one step later
    assert m.peak_timing_offset(ref, cand) == 1


def test_peak_timing_offset_negative_symmetric() -> None:
    ref = np.array([0.0, 0.0, 5.0, 0.0])
    cand = np.array([0.0, 5.0, 0.0, 0.0])
    assert m.peak_timing_offset(ref, cand) == -1


def test_peak_timing_offset_flat_series_is_zero() -> None:
    ref = np.zeros(5)
    cand = np.zeros(5)
    assert m.peak_timing_offset(ref, cand) == 0


# ---------- Runoff volume ratio ----------------------------------------------


def test_runoff_volume_ratio_uniform_dt_matches_sum_ratio() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    cand = np.array([1.1, 2.2, 3.3])
    # dt cancels -> ratio = 1.1
    assert m.runoff_volume_ratio(ref, cand, dt=1.0) == pytest.approx(1.1)


def test_runoff_volume_ratio_nonuniform_dt() -> None:
    ref = np.array([1.0, 2.0, 3.0])
    cand = np.array([1.0, 2.0, 3.0])
    # cand == ref -> ratio must be 1.0 regardless of dt weights
    dt = np.array([0.5, 1.0, 2.0])
    assert m.runoff_volume_ratio(ref, cand, dt=dt) == pytest.approx(1.0)


def test_runoff_volume_ratio_zero_reference_returns_nan() -> None:
    ref = np.zeros(3)
    cand = np.array([1.0, 2.0, 3.0])
    assert math.isnan(m.runoff_volume_ratio(ref, cand))


# ---------- Monthly bias MAE --------------------------------------------------


def test_monthly_bias_mae_perfect_match() -> None:
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = ref.copy()
    labels = np.array([202601, 202601, 202602, 202602])
    assert m.monthly_bias_mae(ref, cand, labels) == pytest.approx(0.0)


def test_monthly_bias_mae_known_value() -> None:
    # Two months:
    #   Jan totals: ref=3, cand=3.3 -> rel_err = 0.1
    #   Feb totals: ref=7, cand=6.3 -> rel_err = 0.1
    # MAE = 0.1
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    cand = np.array([1.1, 2.2, 2.7, 3.6])
    labels = np.array([202601, 202601, 202602, 202602])
    assert m.monthly_bias_mae(ref, cand, labels) == pytest.approx(0.1, abs=1e-9)


def test_monthly_bias_mae_zero_month_totals_excluded() -> None:
    ref = np.array([0.0, 0.0, 5.0, 5.0])
    cand = np.array([1.0, 1.0, 5.5, 4.5])  # first month ref=0 -> excluded
    labels = np.array([202601, 202601, 202602, 202602])
    # Only Feb counts: ref=10, cand=10 -> rel_err = 0
    assert m.monthly_bias_mae(ref, cand, labels) == pytest.approx(0.0)


def test_monthly_bias_mae_all_zero_ref_returns_nan() -> None:
    ref = np.zeros(4)
    cand = np.array([1.0, 2.0, 3.0, 4.0])
    labels = np.array([202601, 202601, 202602, 202602])
    assert math.isnan(m.monthly_bias_mae(ref, cand, labels))


# ---------- Water balance residual -------------------------------------------


def test_water_balance_residual_perfect_closure() -> None:
    # Q_out = P - ET - dS -> residual = 0
    p = np.array([10.0, 10.0, 10.0])
    e = np.array([2.0, 2.0, 2.0])
    ds = np.array([1.0, 0.0, -1.0])  # sum = 0
    q = p - e - ds
    assert m.water_balance_residual(q, p, e, ds) == pytest.approx(0.0)


def test_water_balance_residual_5_percent_gap() -> None:
    # Total P = 30, LHS deficit = 1.5 -> residual = 1.5 / 30 = 0.05
    p = np.array([10.0, 10.0, 10.0])
    e = np.array([2.0, 2.0, 2.0])
    ds = np.array([0.0, 0.0, 0.0])
    q = np.array([8.0 - 0.5, 8.0 - 0.5, 8.0 - 0.5])
    assert m.water_balance_residual(q, p, e, ds) == pytest.approx(0.05, abs=1e-9)


def test_water_balance_residual_zero_precipitation_returns_nan() -> None:
    p = np.zeros(3)
    e = np.zeros(3)
    ds = np.zeros(3)
    q = np.array([1.0, 1.0, 1.0])
    assert math.isnan(m.water_balance_residual(q, p, e, ds))


def test_water_balance_residual_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        m.water_balance_residual(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
        )
