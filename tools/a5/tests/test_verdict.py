"""Tests for verdict threshold logic + MARKER block emission."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from a5.verdict import (
    MalformedThresholdError,
    evaluate,
    format_marker_block,
    load_thresholds,
)


# ---------- Threshold YAML loading ------------------------------------------


@pytest.fixture()
def default_thresholds_path() -> Path:
    return Path(__file__).parent.parent / "config" / "a5_thresholds.default.yaml"


def test_load_default_thresholds(default_thresholds_path: Path) -> None:
    t = load_thresholds(str(default_thresholds_path))
    # Every metric family present
    for name in [
        "nse",
        "kge",
        "peak_magnitude_ratio",
        "peak_timing_offset",
        "runoff_volume_ratio",
        "monthly_bias_mae",
        "water_balance_residual",
    ]:
        assert name in t
    assert t["nse"]["min"] == 0.90
    assert t["water_balance_residual"]["max"] == 0.05


def test_load_thresholds_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MalformedThresholdError, match="not found"):
        load_thresholds(str(tmp_path / "nope.yaml"))


def test_load_thresholds_unknown_metric(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("bogus_metric:\n  min: 1.0\n  weight: 1.0\n")
    with pytest.raises(MalformedThresholdError, match="unknown metric"):
        load_thresholds(str(p))


def test_load_thresholds_missing_weight(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("nse:\n  min: 0.9\n")
    with pytest.raises(MalformedThresholdError, match="weight"):
        load_thresholds(str(p))


def test_load_thresholds_interval_min_gt_max(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        dedent(
            """
            peak_magnitude_ratio:
              min: 1.5
              max: 0.5
              weight: 0.75
            """
        )
    )
    with pytest.raises(MalformedThresholdError, match="min .* > max"):
        load_thresholds(str(p))


def test_load_thresholds_non_dict_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just_a_list_item\n")
    with pytest.raises(MalformedThresholdError, match="must be a mapping"):
        load_thresholds(str(p))


# ---------- evaluate() ------------------------------------------------------


@pytest.fixture()
def default_thresholds(default_thresholds_path: Path) -> dict:
    return load_thresholds(str(default_thresholds_path))


def _perfect_metrics() -> dict[str, float]:
    return {
        "nse": 1.0,
        "kge": 1.0,
        "peak_magnitude_ratio": 1.0,
        "peak_timing_offset": 0.0,
        "runoff_volume_ratio": 1.0,
        "monthly_bias_mae": 0.0,
        "water_balance_residual": 0.0,
    }


def test_evaluate_all_pass(default_thresholds: dict) -> None:
    v = evaluate(_perfect_metrics(), default_thresholds)
    assert v.verdict == "PASS"
    assert v.weighted_score == pytest.approx(1.0)
    assert all(r.passed for r in v.per_metric.values())


def test_evaluate_critical_weight_fail_forces_overall_fail(
    default_thresholds: dict,
) -> None:
    # NSE weight = 1.0 (critical). Fail NSE alone.
    metrics = _perfect_metrics()
    metrics["nse"] = 0.1
    v = evaluate(metrics, default_thresholds)
    assert v.verdict == "FAIL"
    assert not v.per_metric["nse"].passed


def test_evaluate_low_weight_fail_alone_may_still_pass(
    default_thresholds: dict,
) -> None:
    # peak_timing_offset weight = 0.5 (non-critical). Fail it alone; other
    # metrics carry the weighted score above 0.85.
    metrics = _perfect_metrics()
    metrics["peak_timing_offset"] = 10.0  # exceeds max_abs_steps=4
    v = evaluate(metrics, default_thresholds)
    # Weighted score = (1+1+0.75+0+1+0.5+0.75) / (1+1+0.75+0.5+1+0.5+0.75)
    #                = 5.0 / 5.5 ~ 0.909  >= 0.85 -> PASS
    assert v.verdict == "PASS"
    assert not v.per_metric["peak_timing_offset"].passed


def test_evaluate_missing_metric_treated_as_fail(default_thresholds: dict) -> None:
    metrics = _perfect_metrics()
    del metrics["nse"]
    v = evaluate(metrics, default_thresholds)
    assert not v.per_metric["nse"].passed
    assert "not provided" in v.per_metric["nse"].reason


def test_evaluate_nan_metric_treated_as_fail(default_thresholds: dict) -> None:
    metrics = _perfect_metrics()
    metrics["nse"] = float("nan")
    v = evaluate(metrics, default_thresholds)
    assert not v.per_metric["nse"].passed
    assert "NaN" in v.per_metric["nse"].reason


# ---------- MARKER block ----------------------------------------------------


def test_marker_block_byte_exact(default_thresholds: dict) -> None:
    # Deterministic input -> deterministic MARKER string. If format is
    # changed, this test fails and the version bumps.
    metrics = {
        "nse": 0.973,
        "kge": 0.891,
        "peak_magnitude_ratio": 1.02,
        "peak_timing_offset": -1.0,
        "runoff_volume_ratio": 0.994,
        "monthly_bias_mae": 0.031,
        "water_balance_residual": 0.024,
    }
    v = evaluate(metrics, default_thresholds)
    out = format_marker_block("heihe_x4", v)
    expected = (
        "MARKER:A5_VERDICT_BEGIN\n"
        "case=heihe_x4\n"
        "verdict=PASS\n"
        f"weighted_score={v.weighted_score:.4f}\n"
        "nse=0.9730\n"
        "kge=0.8910\n"
        "peak_mag_ratio=1.0200\n"
        "peak_timing_offset_steps=-1\n"
        "runoff_volume_ratio=0.9940\n"
        "monthly_bias_mae=0.0310\n"
        "water_balance_residual=0.0240\n"
        "MARKER:A5_VERDICT_END\n"
    )
    assert out == expected


def test_marker_block_nan_metric_renders_as_string(
    default_thresholds: dict,
) -> None:
    metrics = _perfect_metrics()
    metrics["nse"] = float("nan")
    v = evaluate(metrics, default_thresholds)
    out = format_marker_block("case_nan", v)
    assert "nse=NaN" in out
    assert "verdict=FAIL" in out


# ---------- CLI-level malformed config exit ---------------------------------


def test_cli_malformed_yaml_exits_2(tmp_path: Path) -> None:
    from a5.cli import main

    bad = tmp_path / "bad.yaml"
    bad.write_text("nse:\n  min: 0.9\n")  # missing weight
    rc = main(
        [
            "--reference",
            str(tmp_path),
            "--candidate",
            str(tmp_path),
            "--config",
            str(bad),
            "--case-name",
            "unit",
            "--out",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2
