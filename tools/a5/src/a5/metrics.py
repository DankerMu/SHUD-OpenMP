"""Seven hydrology-acceptance metrics for A5.

Each function follows the shape:

    def metric(ref: np.ndarray, cand: np.ndarray, **kwargs) -> float

Ref and cand are 1-D discharge (or precipitation / storage) timeseries of
identical length, aligned in time. When degenerate input makes a metric
mathematically undefined (division by zero, zero variance in reference),
the function returns np.nan; callers must treat NaN as a signal that the
metric could not be evaluated, not as a passing verdict.
"""

from __future__ import annotations

import math

import numpy as np


def _validate_pair(ref: np.ndarray, cand: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if ref.ndim != 1 or cand.ndim != 1:
        raise ValueError(
            f"ref and cand must be 1-D; got ref.ndim={ref.ndim}, cand.ndim={cand.ndim}"
        )
    if ref.shape != cand.shape:
        raise ValueError(
            f"ref and cand must have equal length; got {ref.shape} vs {cand.shape}"
        )
    if ref.size == 0:
        raise ValueError("ref and cand must be non-empty")
    return ref, cand


def nse(ref: np.ndarray, cand: np.ndarray) -> float:
    """Nash-Sutcliffe efficiency.

    NSE = 1 - sum((cand - ref)**2) / sum((ref - mean(ref))**2)

    Range: (-inf, 1]. Perfect = 1.0. Zero = candidate is no better than
    reference-mean. Returns NaN when the reference is constant
    (denominator = 0), which makes NSE undefined.
    """
    ref, cand = _validate_pair(ref, cand)
    ref_mean = ref.mean()
    denom = float(np.sum((ref - ref_mean) ** 2))
    if denom == 0.0:
        return math.nan
    numer = float(np.sum((cand - ref) ** 2))
    return 1.0 - numer / denom


def kge(ref: np.ndarray, cand: np.ndarray) -> float:
    """Kling-Gupta efficiency (Gupta et al. 2009 formulation).

    KGE = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        r     = Pearson correlation between ref and cand
        alpha = cand.std / ref.std   (variability ratio)
        beta  = cand.mean / ref.mean (bias ratio)

    Range: (-inf, 1]. Perfect = 1.0. Returns NaN when ref.std == 0
    (r undefined) or ref.mean == 0 (beta undefined).
    """
    ref, cand = _validate_pair(ref, cand)
    ref_std = float(ref.std(ddof=0))
    ref_mean = float(ref.mean())
    if ref_std == 0.0 or ref_mean == 0.0:
        return math.nan
    cand_std = float(cand.std(ddof=0))
    cand_mean = float(cand.mean())
    # Pearson r via numpy — guard against a constant candidate (cand_std == 0):
    # in that case r is undefined; we surface NaN.
    if cand_std == 0.0:
        return math.nan
    # Compute r manually to avoid RuntimeWarnings from np.corrcoef on
    # nearly-constant series.
    cov = float(np.mean((ref - ref_mean) * (cand - cand_mean)))
    r = cov / (ref_std * cand_std)
    alpha = cand_std / ref_std
    beta = cand_mean / ref_mean
    return 1.0 - math.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)


def peak_magnitude_ratio(ref: np.ndarray, cand: np.ndarray) -> float:
    """cand.max() / ref.max(). Ideal = 1.0.

    Returns NaN if ref.max() == 0.
    """
    ref, cand = _validate_pair(ref, cand)
    ref_peak = float(ref.max())
    if ref_peak == 0.0:
        return math.nan
    return float(cand.max()) / ref_peak


def peak_timing_offset(ref: np.ndarray, cand: np.ndarray) -> int:
    """argmax(cand) - argmax(ref), in timesteps.

    Positive => candidate peaks later than reference. Ties resolved by
    numpy.argmax (first occurrence). A completely flat series has a
    well-defined argmax (index 0), so the offset is 0 for two flat
    series — that is the intended behavior per spec.
    """
    ref, cand = _validate_pair(ref, cand)
    return int(np.argmax(cand)) - int(np.argmax(ref))


def runoff_volume_ratio(
    ref: np.ndarray, cand: np.ndarray, dt: np.ndarray | float = 1.0
) -> float:
    """Total runoff volume ratio = sum(cand * dt) / sum(ref * dt).

    Args:
        ref, cand: aligned 1-D discharge series (m^3/day or comparable).
        dt: either a scalar timestep (in days) or a 1-D array of per-step
            durations. When ref and cand share dt, the dt cancels out and
            this reduces to sum(cand)/sum(ref) — but we keep the
            multiplication explicit so non-uniform dt is honored.

    Returns NaN if the integrated reference volume is 0.
    """
    ref, cand = _validate_pair(ref, cand)
    if np.isscalar(dt):
        dt_arr = np.full_like(ref, float(dt))
    else:
        dt_arr = np.asarray(dt, dtype=np.float64)
        if dt_arr.shape != ref.shape:
            raise ValueError(
                f"dt shape {dt_arr.shape} does not match ref shape {ref.shape}"
            )
    ref_vol = float(np.sum(ref * dt_arr))
    if ref_vol == 0.0:
        return math.nan
    cand_vol = float(np.sum(cand * dt_arr))
    return cand_vol / ref_vol


def monthly_bias_mae(
    ref: np.ndarray, cand: np.ndarray, month_index: np.ndarray
) -> float:
    """Mean absolute relative error on monthly aggregates.

    Aggregates ref and cand by month_index label, then computes:

        MAE = mean( |(cand_monthly - ref_monthly) / ref_monthly| )

    Months where the reference total is 0 are excluded from the mean
    (rather than propagating NaN). If every month has ref_monthly == 0,
    returns NaN.

    Args:
        ref, cand:   aligned 1-D series.
        month_index: shape (T,) integer or datetime64 labels grouping
                     timesteps into months. Ties (same label) are summed.
    """
    ref, cand = _validate_pair(ref, cand)
    month_index = np.asarray(month_index)
    if month_index.shape != ref.shape:
        raise ValueError(
            f"month_index shape {month_index.shape} does not match ref shape {ref.shape}"
        )
    # Group by unique label preserving order of first appearance
    unique_labels, inverse = np.unique(month_index, return_inverse=True)
    n_months = unique_labels.size
    ref_monthly = np.zeros(n_months, dtype=np.float64)
    cand_monthly = np.zeros(n_months, dtype=np.float64)
    np.add.at(ref_monthly, inverse, ref)
    np.add.at(cand_monthly, inverse, cand)

    mask = ref_monthly != 0.0
    if not np.any(mask):
        return math.nan
    rel_err = np.abs((cand_monthly[mask] - ref_monthly[mask]) / ref_monthly[mask])
    return float(rel_err.mean())


def water_balance_residual(
    runoff_out: np.ndarray,
    precip: np.ndarray,
    et: np.ndarray,
    storage_change: np.ndarray,
) -> float:
    """Water-balance residual = |Q_out - (P - ET - dS)| summed, divided by total P.

    All four arrays share the same aligned time axis. Each entry is a
    volume (or volume-flux) in consistent units.

        residual = |sum(runoff_out) - (sum(precip) - sum(et) - sum(storage_change))|
                   / sum(precip)

    Ideal <= 0.05 (5% closure error).

    Returns NaN when total precipitation is 0.
    """
    q = np.asarray(runoff_out, dtype=np.float64)
    p = np.asarray(precip, dtype=np.float64)
    e = np.asarray(et, dtype=np.float64)
    ds = np.asarray(storage_change, dtype=np.float64)
    shapes = {q.shape, p.shape, e.shape, ds.shape}
    if len(shapes) != 1:
        raise ValueError(
            "runoff_out, precip, et, storage_change must all share the same shape; "
            f"got {q.shape}, {p.shape}, {e.shape}, {ds.shape}"
        )
    total_p = float(p.sum())
    if total_p == 0.0:
        return math.nan
    lhs = float(q.sum())
    rhs = total_p - float(e.sum()) - float(ds.sum())
    return abs(lhs - rhs) / total_p
