"""Shared pytest fixtures + synthetic CSV builders for osc_diag tests.

The builders write text CSVs byte-compatible with the real emitter
`SHUD/src/Model/MD_osc_diag.hpp`, so tests exercise the exact parser path a
real keliya/qinyijiang run would. Each builder lets a test dial a KNOWN
burst share / concentration / Spearman rho so the metric can be checked
against a construction target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pytest

MINUTES_PER_DAY = 1440.0


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def write_dt_trace(
    path: Path,
    rows: Sequence[tuple[float, int]],
    solverstep_min: float = 20.0,
    project_name: str = "synthcase",
    *,
    extra_cols: tuple[int, int, int, float, float] = (0, 0, 0, 0.0, 0.0),
) -> Path:
    """Write a diag_dt_trace.csv.

    Args:
        rows: sequence of (t_next_minutes, delta_nst). Companion deltas +
              h_last/h_cur are filled from `extra_cols` (delta_nfe, ncfn,
              netf, h_last, h_cur) — they do not affect any tested metric.
    """
    d_nfe, d_ncfn, d_netf, h_last, h_cur = extra_cols
    lines = [
        f"# project_name={project_name} solverstep_min={solverstep_min:.10g}",
        "t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur",
    ]
    for t_next, d_nst in rows:
        lines.append(
            f"{t_next:.6f},{d_nst},{d_nfe},{d_ncfn},{d_netf},"
            f"{h_last:.17g},{h_cur:.17g}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_osc_flips(
    path: Path,
    ele_family_rows: Sequence[tuple[int, int, int]],
    riv_stage: Iterable[int] = (),
    solverstep_min: float = 20.0,
    epsilon_m: float = 1e-6,
    project_name: str = "synthcase",
) -> Path:
    """Write a diag_osc_flips.csv.

    Args:
        ele_family_rows: per-element (flips_sf, flips_us, flips_gw) tuples;
                         entity_id is 1-based row order.
        riv_stage:       per-river flips_stage counts; entity_id 1-based.
    """
    lines = [
        f"# project_name={project_name} solverstep_min={solverstep_min:.10g} "
        f"epsilon_m={epsilon_m:.10g}",
        "entity_type,entity_id,flips_sf,flips_us,flips_gw,flips_stage",
    ]
    for i, (f_sf, f_us, f_gw) in enumerate(ele_family_rows, start=1):
        lines.append(f"ele,{i},{f_sf},{f_us},{f_gw},0")
    for j, f_stage in enumerate(riv_stage, start=1):
        lines.append(f"riv,{j},0,0,0,{f_stage}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_osc_flips_daily(
    path: Path,
    day_rows: Sequence[tuple[int, int, int, int, int]],
    solverstep_min: float = 20.0,
    epsilon_m: float = 1e-6,
    project_name: str = "synthcase",
) -> Path:
    """Write a diag_osc_flips_daily.csv.

    Args:
        day_rows: (day_index, flips_sf, flips_us, flips_gw, flips_stage);
                  flips_total is computed = sum of the four families.
    """
    lines = [
        f"# project_name={project_name} solverstep_min={solverstep_min:.10g} "
        f"epsilon_m={epsilon_m:.10g}",
        "day_index,flips_sf,flips_us,flips_gw,flips_stage,flips_total",
    ]
    for day, f_sf, f_us, f_gw, f_stage in day_rows:
        total = f_sf + f_us + f_gw + f_stage
        lines.append(f"{day},{f_sf},{f_us},{f_gw},{f_stage},{total}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Helpers to construct trace rows with a target burst share
# ---------------------------------------------------------------------------

def rows_for_delta_nst(
    deltas: Sequence[int], solverstep_min: float = 20.0, day: int = 0
) -> list[tuple[float, int]]:
    """Build (t_next, delta_nst) rows within a single day.

    t_next is monotone increasing inside `day` (so all rows floor to the
    same day_index). Values are spaced far from the day boundary to avoid
    accidental cross-day bucketing.
    """
    base = day * MINUTES_PER_DAY + 60.0
    return [(base + k * solverstep_min, int(d)) for k, d in enumerate(deltas)]


@pytest.fixture()
def make_trace(tmp_path: Path):
    def _factory(rows, **kw) -> Path:
        return write_dt_trace(tmp_path / "diag_dt_trace.csv", rows, **kw)

    return _factory


@pytest.fixture()
def make_flips(tmp_path: Path):
    def _factory(ele_family_rows, riv_stage=(), **kw) -> Path:
        return write_osc_flips(
            tmp_path / "diag_osc_flips.csv", ele_family_rows, riv_stage, **kw
        )

    return _factory


@pytest.fixture()
def make_daily(tmp_path: Path):
    def _factory(day_rows, **kw) -> Path:
        return write_osc_flips_daily(
            tmp_path / "diag_osc_flips_daily.csv", day_rows, **kw
        )

    return _factory
