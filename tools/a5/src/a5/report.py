"""JSON + markdown report emitters for A5.

Produces two artifacts inside the --out directory plus a MARKER block
on stdout (see verdict.format_marker_block).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .verdict import OverallVerdict


def _sanitize(v: Any) -> Any:
    """Convert NaN to a JSON-friendly sentinel string; leave others untouched."""
    if isinstance(v, float) and math.isnan(v):
        return "NaN"
    return v


def build_metrics_json(
    case: str,
    reference_dir: str,
    candidate_dir: str,
    thresholds_file: str,
    verdict: OverallVerdict,
    metric_status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble the a5_metrics.json payload.

    Args:
        metric_status: optional side-channel dict of diagnostic strings keyed
            by metric name (e.g. `water_balance_status =
            "unavailable_no_mesh_metadata"`). Merged into each metric's block
            under a `status` field; absent when the metric has no status.
    """
    metrics_block: dict[str, dict[str, Any]] = {}
    for name, r in verdict.per_metric.items():
        entry: dict[str, Any] = {
            "value": _sanitize(r.value),
            "pass": bool(r.passed),
            "threshold": r.threshold,
            "weight": r.weight,
            "reason": r.reason,
        }
        # Water balance follow-up (PR-Y2): if the metric has a side-channel
        # status message (e.g. "unavailable_no_mesh_metadata"), attach it.
        status_key = f"{name}_status"
        if metric_status and status_key in metric_status:
            entry["status"] = metric_status[status_key]
        metrics_block[name] = entry
    return {
        "schema_version": "A5-v1.0.0",
        "case": case,
        "reference_dir": reference_dir,
        "candidate_dir": candidate_dir,
        "thresholds_file": thresholds_file,
        "metrics": metrics_block,
        "overall": {
            "verdict": verdict.verdict,
            "weighted_score": verdict.weighted_score,
        },
    }


def build_verdict_md(
    case: str,
    reference_dir: str,
    candidate_dir: str,
    thresholds_file: str,
    verdict: OverallVerdict,
) -> str:
    """Assemble the human-readable a5_verdict.md contents."""
    lines: list[str] = []
    lines.append(f"# A5 hydrology-acceptance verdict — {case}")
    lines.append("")
    lines.append(f"- Schema version: `A5-v1.0.0`")
    lines.append(f"- Reference dir: `{reference_dir}`")
    lines.append(f"- Candidate dir: `{candidate_dir}`")
    lines.append(f"- Thresholds file: `{thresholds_file}`")
    lines.append("")
    lines.append(
        f"## Overall verdict: **{verdict.verdict}** "
        f"(weighted score = {verdict.weighted_score:.4f})"
    )
    lines.append("")
    lines.append("| Metric | Value | Threshold | Weight | Pass | Reason |")
    lines.append("| --- | ---: | --- | ---: | :---: | --- |")
    for name, r in verdict.per_metric.items():
        if isinstance(r.value, float) and math.isnan(r.value):
            val_str = "NaN"
        elif name == "peak_timing_offset":
            val_str = f"{int(round(r.value))}"
        else:
            val_str = f"{r.value:.6g}"
        thr_str = ", ".join(
            f"{k}={v}"
            for k, v in r.threshold.items()
            if k != "weight"
        )
        pass_str = "PASS" if r.passed else "FAIL"
        reason = r.reason or ""
        lines.append(
            f"| {name} | {val_str} | {thr_str} | {r.weight:.2f} | "
            f"{pass_str} | {reason} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if verdict.verdict == "PASS":
        lines.append(
            "Candidate meets the hydrology-acceptance bar: the weighted PASS "
            f"score {verdict.weighted_score:.4f} clears the 0.85 gate and no "
            "critical-weight (>= 0.75) metric individually failed."
        )
    else:
        lines.append(
            "Candidate does NOT meet the hydrology-acceptance bar. See the "
            "per-metric reasons column for what regressed. A single "
            "critical-weight metric failure is sufficient to force FAIL, "
            "regardless of the aggregate weighted score."
        )
    lines.append("")
    return "\n".join(lines)


def write_reports(
    out_dir: Path,
    case: str,
    reference_dir: str,
    candidate_dir: str,
    thresholds_file: str,
    verdict: OverallVerdict,
    metric_status: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Write a5_metrics.json + a5_verdict.md into out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "a5_metrics.json"
    md_path = out_dir / "a5_verdict.md"

    payload = build_metrics_json(
        case=case,
        reference_dir=reference_dir,
        candidate_dir=candidate_dir,
        thresholds_file=thresholds_file,
        verdict=verdict,
        metric_status=metric_status,
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")

    md_text = build_verdict_md(
        case=case,
        reference_dir=reference_dir,
        candidate_dir=candidate_dir,
        thresholds_file=thresholds_file,
        verdict=verdict,
    )
    md_path.write_text(md_text, encoding="utf-8")
    return json_path, md_path
