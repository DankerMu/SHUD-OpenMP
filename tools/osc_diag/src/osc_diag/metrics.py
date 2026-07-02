"""Analysis metrics for the P11-osc diagnostic CSVs.

All thresholds live in `verdict.py`; this module computes the RAW gate
inputs only (burst share, top-1% element concentration, per-day Spearman
rho) plus the dt histogram artifact. Nothing here decides a verdict.

Canonical mean-dt formula (single source, verbatim from dt-step-trace spec
and design.md D1):

    mean_dt_seconds = 60 * solverstep_min / delta_nst

`solverstep_min` comes from the trace header (never cfg.para, never a
hardcode). Rows with delta_nst == 0 are skipped: 0 contribution to burst
numerators AND to the nst denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .parsers import DailyFlips, DtTrace, FlipCounts

# Fixed mean-dt histogram bins in seconds (osc-diag-analyzer spec:
# "dt histogram artifact"). Half-open [lo, hi); the last bin is [1200, inf).
HISTOGRAM_BIN_EDGES_S: tuple[float, ...] = (0.0, 10.0, 60.0, 300.0, 600.0, 1200.0)

# Burst-share thresholds in seconds (spec: {60 s, 10 s}).
BURST_THRESHOLD_60S = 60.0
BURST_THRESHOLD_10S = 10.0

# Minutes per model day (SHUD model-time convention; day_index = floor(t/1440)).
MINUTES_PER_DAY = 1440.0


def mean_dt_seconds(solverstep_min: float, delta_nst: np.ndarray) -> np.ndarray:
    """Canonical interval-mean dt in seconds for each trace row.

    `delta_nst == 0` rows yield inf (they are excluded downstream; the value
    is never consumed). Callers must apply the skip mask first.
    """
    delta = np.asarray(delta_nst, dtype=np.float64)
    with np.errstate(divide="ignore"):
        out = 60.0 * solverstep_min / delta
    return out


def _nonzero_mask(delta_nst: np.ndarray) -> np.ndarray:
    """Boolean mask of trace rows that contribute (delta_nst > 0)."""
    return np.asarray(delta_nst) > 0


@dataclass(frozen=True)
class HistogramBin:
    lo: float
    hi: float  # math.inf for the open top bin
    interval_count: int
    nst_share: float

    @property
    def label(self) -> str:
        hi = "inf" if math.isinf(self.hi) else f"{self.hi:g}"
        return f"[{self.lo:g},{hi})"


@dataclass(frozen=True)
class DtHistogram:
    bins: tuple[HistogramBin, ...]
    total_intervals: int  # non-skipped rows (delta_nst > 0)
    total_nst: int        # sum of delta_nst over non-skipped rows

    def interval_count_sum(self) -> int:
        return sum(b.interval_count for b in self.bins)

    def nst_share_sum(self) -> float:
        return float(sum(b.nst_share for b in self.bins))


def compute_histogram(trace: DtTrace) -> DtHistogram:
    """Bin non-skipped intervals by mean-dt into the fixed bins.

    Per-bin nst_share = (sum of delta_nst in bin) / (total non-skipped nst).
    Interval counts sum to the number of non-skipped rows; nst shares sum to
    1 within float tolerance (spec scenario "histogram written and
    consistent").
    """
    mask = _nonzero_mask(trace.delta_nst)
    dnst = trace.delta_nst[mask].astype(np.float64)
    dt_s = mean_dt_seconds(trace.solverstep_min, trace.delta_nst)[mask]
    total_nst = float(dnst.sum())

    edges = list(HISTOGRAM_BIN_EDGES_S) + [math.inf]
    bins: list[HistogramBin] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if math.isinf(hi):
            in_bin = dt_s >= lo
        else:
            in_bin = (dt_s >= lo) & (dt_s < hi)
        count = int(np.count_nonzero(in_bin))
        nst_in = float(dnst[in_bin].sum())
        share = nst_in / total_nst if total_nst > 0.0 else 0.0
        bins.append(HistogramBin(lo=lo, hi=hi, interval_count=count, nst_share=share))

    return DtHistogram(
        bins=tuple(bins),
        total_intervals=int(mask.sum()),
        total_nst=int(round(total_nst)),
    )


def burst_share(trace: DtTrace, threshold_s: float) -> float:
    """Fraction of total nst carried by intervals with mean dt < threshold_s.

    Speedup-ceiling estimator: eliminating all sub-threshold bursts caps the
    achievable wall speedup at 1 / (1 - burst_share). Returns 0.0 when there
    is no contributing nst (degenerate trace).
    """
    mask = _nonzero_mask(trace.delta_nst)
    dnst = trace.delta_nst[mask].astype(np.float64)
    dt_s = mean_dt_seconds(trace.solverstep_min, trace.delta_nst)[mask]
    total_nst = float(dnst.sum())
    if total_nst <= 0.0:
        return 0.0
    burst_nst = float(dnst[dt_s < threshold_s].sum())
    return burst_nst / total_nst


@dataclass(frozen=True)
class ConcentrationResult:
    top1pct_concentration: float
    n_ele: int
    top_k: int                    # ceil(0.01 * n_ele)
    total_ele_flips: int
    top_element_ids: tuple[int, ...]   # 1-based ids of the top-K elements
    top_element_flips: tuple[int, ...]
    riv_flips_total: int          # rivers reported separately (NOT in denom)


def flip_concentration(flips: FlipCounts, top_frac: float = 0.01) -> ConcentrationResult:
    """Top-1% element flip concentration.

    Population = elements only; per-element flips = flips_sf+us+gw summed
    (done in the parser). top_k = ceil(top_frac * n_ele). Concentration =
    (flips carried by the top_k elements) / (total element flips). River
    rows are reported separately and are NOT in the population or the
    denominator (spec: "top-K flip elements and the top-1% concentration").

    Returns 0.0 concentration when there are no element flips (degenerate).
    """
    ele = np.asarray(flips.ele_flips_total, dtype=np.int64)
    n_ele = int(ele.size)
    top_k = max(1, math.ceil(top_frac * n_ele))
    total = int(ele.sum())

    # Descending sort; ties broken by ascending 1-based id for determinism.
    order = np.lexsort((np.arange(n_ele), -ele))
    top_idx = order[:top_k]
    top_flips = int(ele[top_idx].sum())
    conc = (top_flips / total) if total > 0 else 0.0

    return ConcentrationResult(
        top1pct_concentration=conc,
        n_ele=n_ele,
        top_k=top_k,
        total_ele_flips=total,
        top_element_ids=tuple(int(i) + 1 for i in top_idx),  # 1-based
        top_element_flips=tuple(int(ele[i]) for i in top_idx),
        riv_flips_total=int(np.asarray(flips.riv_flips_stage, dtype=np.int64).sum()),
    )


def daily_sub60_interval_count(trace: DtTrace) -> dict[int, int]:
    """Count of sub-60 s mean-dt intervals per day_index.

    D1 trace rows are bucketed by the shared key day_index = floor(t_next /
    1440) — the same key the daily flips CSV uses. delta_nst == 0 rows are
    skipped (spec skip rule). Returns {day_index: count}.
    """
    mask = _nonzero_mask(trace.delta_nst)
    t_next = trace.t_next[mask]
    dt_s = mean_dt_seconds(trace.solverstep_min, trace.delta_nst)[mask]
    day_idx = np.floor(t_next / MINUTES_PER_DAY).astype(np.int64)
    is_sub60 = dt_s < BURST_THRESHOLD_60S

    counts: dict[int, int] = {}
    for d, hit in zip(day_idx.tolist(), is_sub60.tolist()):
        counts[d] = counts.get(d, 0) + (1 if hit else 0)
    return counts


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    n_days: int                  # number of joined day points used
    dropped_trailing_day: int | None  # day_index dropped as partial (or None)


def daily_flip_dt_spearman(
    trace: DtTrace, daily: DailyFlips, drop_trailing_partial_day: bool = True
) -> SpearmanResult:
    """Per-day Spearman rank correlation rho.

    Between daily flips_total (from diag_osc_flips_daily.csv) and the daily
    count of sub-60 s mean-dt intervals (D1 rows bucketed on the shared
    day_index key). Pass threshold rho >= 0.5 is applied in verdict.py.

    FENCEPOST HANDLING (PR-D1 reviewer finding): an N-day run emits N+1 day
    rows because the final interval's t_next = END*1440 floors exactly onto
    day END, producing a trailing degenerate single-interval bucket (real
    keliya evidence: last daily row `12143,4,0,0,0,4`). That trailing bucket
    is a PARTIAL day and appears on BOTH join sides (both derive from the
    same t_next stream). We drop the maximum day_index from BOTH sides
    symmetrically before ranking, so the correlation is computed over the
    complete-day window only. The drop is deterministic (always the max
    day_index) and documented in the README. It is symmetric, so it cannot
    bias the rank correlation toward either variable.

    rho is NaN when fewer than 2 joined points remain or when either ranked
    series is constant (Spearman undefined). verdict.py maps a NaN rho to
    "fails the rho >= 0.5 gate".
    """
    sub60 = daily_sub60_interval_count(trace)

    dropped: int | None = None
    daily_days = daily.day_index.tolist()
    daily_totals = daily.flips_total.tolist()

    if drop_trailing_partial_day and len(daily_days) > 0:
        # The shared trailing bucket = the max day_index present across the
        # union of both sides (they share the same t_next stream, so the
        # daily CSV max is the join max).
        trailing = max(max(daily_days), max(sub60) if sub60 else daily_days[-1])
        dropped = int(trailing)
        daily_pairs = [
            (d, t) for d, t in zip(daily_days, daily_totals) if d != trailing
        ]
        sub60 = {d: c for d, c in sub60.items() if d != trailing}
    else:
        daily_pairs = list(zip(daily_days, daily_totals))

    # Join on shared day_index: iterate the (post-drop) daily rows and pull
    # the matching sub-60s count (0 if that day had no sub-60s intervals).
    x_flips: list[float] = []
    y_sub60: list[float] = []
    for day, total in daily_pairs:
        x_flips.append(float(total))
        y_sub60.append(float(sub60.get(int(day), 0)))

    n = len(x_flips)
    if n < 2:
        return SpearmanResult(rho=math.nan, n_days=n, dropped_trailing_day=dropped)

    x = np.asarray(x_flips, dtype=np.float64)
    y = np.asarray(y_sub60, dtype=np.float64)
    # Spearman is undefined when either series is constant (zero rank var).
    if np.all(x == x[0]) or np.all(y == y[0]):
        return SpearmanResult(rho=math.nan, n_days=n, dropped_trailing_day=dropped)

    rho, _ = stats.spearmanr(x, y)
    return SpearmanResult(rho=float(rho), n_days=n, dropped_trailing_day=dropped)
