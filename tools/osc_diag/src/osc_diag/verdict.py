"""Total verdict decision function + machine-readable MARKER block.

Thresholds are PINNED here verbatim from the osc-diag-analyzer spec /
spike_brief kill-gate / design.md verdict gate (single source; nothing is
decided in-tool — this module only applies the fixed thresholds). Every
input combination maps to exactly one verdict; there is no catch-all
branch.

    OSC_CONFIRMED = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50
                    AND rho >= 0.5
    REAL_DYNAMICS = burst_share_60s <  0.50 OR  top1pct_concentration <  0.50
    INCONCLUSIVE  = burst_share_60s >= 0.50 AND top1pct_concentration >= 0.50
                    AND rho <  0.5

A NaN rho (Spearman undefined) counts as rho < 0.5 — it fails the >= 0.5
gate, so a bursty+localized case with an undefined temporal link resolves
to INCONCLUSIVE, never OSC_CONFIRMED (conservative, matching the D2 proxy
under-count bias in design R3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Pinned gate thresholds (spec / spike_brief kill-gate).
BURST_SHARE_60S_THRESHOLD = 0.50
CONCENTRATION_THRESHOLD = 0.50
RHO_THRESHOLD = 0.5

VERDICT_CONFIRMED = "OSC_CONFIRMED"
VERDICT_REAL_DYNAMICS = "REAL_DYNAMICS"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class VerdictResult:
    verdict: str
    burst_share_60s: float
    burst_share_10s: float
    top1pct_concentration: float
    rho: float


def _rho_passes(rho: float) -> bool:
    """rho >= 0.5, with NaN treated as failing the gate (conservative)."""
    if isinstance(rho, float) and math.isnan(rho):
        return False
    return rho >= RHO_THRESHOLD


def decide(
    burst_share_60s: float,
    burst_share_10s: float,
    top1pct_concentration: float,
    rho: float,
) -> VerdictResult:
    """Apply the TOTAL decision function. Returns exactly one verdict.

    Region partition (total, no overlaps, no gaps):
      - burst < 0.50 OR conc < 0.50            -> REAL_DYNAMICS
      - (burst >= 0.50 AND conc >= 0.50):
          rho >= 0.5   -> OSC_CONFIRMED
          rho <  0.5   -> INCONCLUSIVE   (NaN rho lands here)
    """
    bursty = burst_share_60s >= BURST_SHARE_60S_THRESHOLD
    localized = top1pct_concentration >= CONCENTRATION_THRESHOLD

    if not bursty or not localized:
        verdict = VERDICT_REAL_DYNAMICS
    elif _rho_passes(rho):
        verdict = VERDICT_CONFIRMED
    else:
        verdict = VERDICT_INCONCLUSIVE

    return VerdictResult(
        verdict=verdict,
        burst_share_60s=burst_share_60s,
        burst_share_10s=burst_share_10s,
        top1pct_concentration=top1pct_concentration,
        rho=rho,
    )


def _fmt(value: float) -> str:
    """Format a gate input; NaN renders as the literal token `NaN`."""
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    return f"{value:.4f}"


def format_marker_block(case: str, result: VerdictResult) -> str:
    """Return the machine-readable MARKER block.

    Line ordering, key names, and float precision are load-bearing —
    downstream aggregators (PR-D3) grep for these tokens. All gate inputs
    and their thresholds are emitted so the verdict is fully reproducible
    from the block alone.
    """
    lines = [
        "MARKER:OSC_DIAG_VERDICT_BEGIN",
        f"case={case}",
        f"verdict={result.verdict}",
        f"burst_share_60s={_fmt(result.burst_share_60s)}",
        f"burst_share_60s_threshold={BURST_SHARE_60S_THRESHOLD:.4f}",
        f"burst_share_10s={_fmt(result.burst_share_10s)}",
        f"top1pct_concentration={_fmt(result.top1pct_concentration)}",
        f"top1pct_concentration_threshold={CONCENTRATION_THRESHOLD:.4f}",
        f"spearman_rho={_fmt(result.rho)}",
        f"spearman_rho_threshold={RHO_THRESHOLD:.4f}",
        "MARKER:OSC_DIAG_VERDICT_END",
    ]
    return "\n".join(lines) + "\n"
