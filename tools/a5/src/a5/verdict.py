"""Threshold gate + weighted PASS/FAIL verdict for A5 metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import yaml

# Metrics whose *value* is inside [min, max] is a PASS.
_INTERVAL_METRICS = {
    "peak_magnitude_ratio",
    "runoff_volume_ratio",
}
# Metrics that PASS when value >= min (higher-is-better).
_MIN_ONLY_METRICS = {"nse", "kge"}
# Metrics that PASS when value <= max (lower-is-better).
_MAX_ONLY_METRICS = {
    "monthly_bias_mae",
    "water_balance_residual",
}
# Metrics whose absolute value must be <= max_abs_steps.
_ABS_MAX_METRICS = {"peak_timing_offset"}

_ALL_METRICS = (
    _MIN_ONLY_METRICS | _INTERVAL_METRICS | _MAX_ONLY_METRICS | _ABS_MAX_METRICS
)

_OVERALL_SCORE_MIN = 0.85
_CRITICAL_WEIGHT_MIN = 0.75


class MalformedThresholdError(ValueError):
    """Raised when the thresholds YAML fails schema validation."""


@dataclass(frozen=True)
class MetricResult:
    name: str
    value: float
    passed: bool
    threshold: dict[str, Any]
    weight: float
    reason: str = ""


@dataclass(frozen=True)
class OverallVerdict:
    verdict: str  # "PASS" | "FAIL"
    weighted_score: float
    per_metric: dict[str, MetricResult] = field(default_factory=dict)


def load_thresholds(path: str) -> dict[str, dict[str, Any]]:
    """Load and validate a thresholds YAML file.

    Raises MalformedThresholdError with an actionable message when the
    file is not readable, is not a dict, references an unknown metric,
    lacks required keys, or has a non-numeric weight / bound.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise MalformedThresholdError(f"thresholds file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise MalformedThresholdError(
            f"thresholds YAML at {path} failed to parse: {exc}"
        ) from exc

    if not isinstance(data, dict):
        raise MalformedThresholdError(
            f"thresholds YAML at {path} must be a mapping at the top level; "
            f"got {type(data).__name__}"
        )

    unknown = set(data.keys()) - _ALL_METRICS
    if unknown:
        raise MalformedThresholdError(
            f"thresholds YAML at {path} references unknown metrics: "
            f"{sorted(unknown)}. Known metrics: {sorted(_ALL_METRICS)}"
        )

    for name, spec in data.items():
        if not isinstance(spec, dict):
            raise MalformedThresholdError(
                f"threshold entry for {name!r} must be a mapping; got "
                f"{type(spec).__name__}"
            )
        if "weight" not in spec:
            raise MalformedThresholdError(
                f"threshold entry for {name!r} is missing required key 'weight'"
            )
        try:
            w = float(spec["weight"])
        except (TypeError, ValueError) as exc:
            raise MalformedThresholdError(
                f"threshold {name!r} has non-numeric weight: {spec['weight']!r}"
            ) from exc
        if not 0.0 <= w <= 1.0:
            raise MalformedThresholdError(
                f"threshold {name!r} weight {w} out of [0.0, 1.0]"
            )

        if name in _MIN_ONLY_METRICS:
            if "min" not in spec:
                raise MalformedThresholdError(
                    f"threshold {name!r} must define 'min'"
                )
        elif name in _MAX_ONLY_METRICS:
            if "max" not in spec:
                raise MalformedThresholdError(
                    f"threshold {name!r} must define 'max'"
                )
        elif name in _INTERVAL_METRICS:
            if "min" not in spec or "max" not in spec:
                raise MalformedThresholdError(
                    f"threshold {name!r} must define both 'min' and 'max'"
                )
            if float(spec["min"]) > float(spec["max"]):
                raise MalformedThresholdError(
                    f"threshold {name!r}: min {spec['min']} > max {spec['max']}"
                )
        elif name in _ABS_MAX_METRICS:
            if "max_abs_steps" not in spec:
                raise MalformedThresholdError(
                    f"threshold {name!r} must define 'max_abs_steps'"
                )

    return data


def _check_one(name: str, value: float, spec: dict[str, Any]) -> tuple[bool, str]:
    """Return (passed, reason). NaN values FAIL with an explanatory reason."""
    weight = float(spec["weight"])
    if isinstance(value, float) and math.isnan(value):
        return False, f"metric returned NaN (undefined; weight={weight})"

    if name in _MIN_ONLY_METRICS:
        minv = float(spec["min"])
        if value >= minv:
            return True, ""
        return False, f"value {value:.6g} < min {minv:.6g}"

    if name in _MAX_ONLY_METRICS:
        maxv = float(spec["max"])
        if value <= maxv:
            return True, ""
        return False, f"value {value:.6g} > max {maxv:.6g}"

    if name in _INTERVAL_METRICS:
        minv = float(spec["min"])
        maxv = float(spec["max"])
        if minv <= value <= maxv:
            return True, ""
        return False, f"value {value:.6g} outside [{minv:.6g}, {maxv:.6g}]"

    if name in _ABS_MAX_METRICS:
        max_abs = float(spec["max_abs_steps"])
        if abs(value) <= max_abs:
            return True, ""
        return False, f"|value|={abs(value):.6g} > max_abs_steps {max_abs:.6g}"

    # Should never reach: load_thresholds guarantees metric name is known.
    return False, f"unhandled metric family for {name!r}"


def evaluate(
    metrics: dict[str, float], thresholds: dict[str, dict[str, Any]]
) -> OverallVerdict:
    """Compute per-metric PASS/FAIL and overall weighted verdict.

    Overall rule:
        weighted_score = sum(w_i * pass_i) / sum(w_i) over metrics that
                         appear in *both* metrics and thresholds
        verdict = PASS iff weighted_score >= 0.85
                  AND every metric with weight >= 0.75 individually PASSed.
    """
    per: dict[str, MetricResult] = {}
    numer = 0.0
    denom = 0.0
    critical_fail = False

    for name, spec in thresholds.items():
        if name not in metrics:
            # Threshold declared but metric not computed — treat as FAIL
            # with an explicit reason. Avoids silently ignoring gaps.
            weight = float(spec["weight"])
            per[name] = MetricResult(
                name=name,
                value=math.nan,
                passed=False,
                threshold=dict(spec),
                weight=weight,
                reason="metric not provided by pipeline",
            )
            denom += weight
            if weight >= _CRITICAL_WEIGHT_MIN:
                critical_fail = True
            continue

        value = float(metrics[name])
        weight = float(spec["weight"])
        passed, reason = _check_one(name, value, spec)
        per[name] = MetricResult(
            name=name,
            value=value,
            passed=passed,
            threshold=dict(spec),
            weight=weight,
            reason=reason,
        )
        denom += weight
        if passed:
            numer += weight
        elif weight >= _CRITICAL_WEIGHT_MIN:
            critical_fail = True

    weighted_score = numer / denom if denom > 0.0 else 0.0
    verdict = (
        "PASS"
        if (weighted_score >= _OVERALL_SCORE_MIN and not critical_fail)
        else "FAIL"
    )
    return OverallVerdict(
        verdict=verdict, weighted_score=weighted_score, per_metric=per
    )


def format_marker_block(case: str, verdict: OverallVerdict) -> str:
    """Return the machine-readable MARKER block, exactly as PR-Z1 will grep for.

    The lines and order below are load-bearing: downstream aggregators
    parse them by regex. Do not reorder without bumping A5-vN.
    """
    per = verdict.per_metric

    def _v(name: str) -> str:
        if name not in per:
            return "NA"
        val = per[name].value
        if isinstance(val, float) and math.isnan(val):
            return "NaN"
        return f"{val:.4f}"

    def _vi(name: str) -> str:
        if name not in per:
            return "NA"
        val = per[name].value
        if isinstance(val, float) and math.isnan(val):
            return "NaN"
        return f"{int(round(val))}"

    lines = [
        "MARKER:A5_VERDICT_BEGIN",
        f"case={case}",
        f"verdict={verdict.verdict}",
        f"weighted_score={verdict.weighted_score:.4f}",
        f"nse={_v('nse')}",
        f"kge={_v('kge')}",
        f"peak_mag_ratio={_v('peak_magnitude_ratio')}",
        f"peak_timing_offset_steps={_vi('peak_timing_offset')}",
        f"runoff_volume_ratio={_v('runoff_volume_ratio')}",
        f"monthly_bias_mae={_v('monthly_bias_mae')}",
        f"water_balance_residual={_v('water_balance_residual')}",
        "MARKER:A5_VERDICT_END",
    ]
    return "\n".join(lines) + "\n"
