"""Shared pytest fixtures for A5 tests.

Also generates the synthetic SHUD binary fixtures used by
`test_shud_output.py` when they are missing (so the repo does not have
to check in binary blobs).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"

_HEADER = "a5-test-fixture-v1"
_HEADER_BYTES_TOTAL = 1024
_START_TIME = 20260101  # 2026-01-01
_N_COLS = 10
_N_STEPS = 5


def _write_synth_dat(path: Path, values: np.ndarray) -> None:
    """Serialize a small SHUD-format binary fixture.

    Args:
        path:   destination file path.
        values: shape (N_STEPS, N_COLS) float64 matrix.
    """
    assert values.ndim == 2 and values.shape[1] == _N_COLS
    n_steps = values.shape[0]

    with open(path, "wb") as f:
        header_bytes = _HEADER.encode("ascii")
        pad = _HEADER_BYTES_TOTAL - len(header_bytes)
        f.write(header_bytes + b"\x00" * pad)

        st = np.array([_START_TIME], dtype="<f8")
        nc = np.array([_N_COLS], dtype="<f8")
        idx = np.arange(1, _N_COLS + 1, dtype="<f8")  # 1-based column ids
        f.write(st.tobytes())
        f.write(nc.tobytes())
        f.write(idx.tobytes())

        # Rows: [t_min, x1, ..., xN_COLS]
        for k in range(n_steps):
            row = np.zeros(_N_COLS + 1, dtype="<f8")
            row[0] = 1440.0 * (k + 1)  # daily cadence, minutes
            row[1:] = values[k]
            f.write(row.tobytes())


@pytest.fixture(scope="session")
def synth_ref_dat() -> Path:
    """Generate (once per session) a synthetic reference SHUD .dat fixture."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "ref_synthetic.dat"
    if not path.exists():
        rng = np.random.default_rng(seed=42)
        vals = rng.uniform(low=100.0, high=1000.0, size=(_N_STEPS, _N_COLS))
        _write_synth_dat(path, vals.astype(np.float64))
    return path


@pytest.fixture(scope="session")
def synth_cand_dat(synth_ref_dat: Path) -> Path:
    """Generate a candidate fixture = reference * 1.05 (5% uniform inflation).

    Used to give deterministic non-zero deltas for read-round-trip tests.
    """
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIR / "cand_synthetic.dat"
    if not path.exists():
        # Read the reference back and inflate. We re-run the writer on the
        # (deterministic) same seed so cand == 1.05 * ref exactly.
        rng = np.random.default_rng(seed=42)
        vals = rng.uniform(low=100.0, high=1000.0, size=(_N_STEPS, _N_COLS))
        cand = (vals * 1.05).astype(np.float64)
        _write_synth_dat(path, cand)
    return path


@pytest.fixture(scope="session")
def n_cols() -> int:
    return _N_COLS


@pytest.fixture(scope="session")
def n_steps() -> int:
    return _N_STEPS


@pytest.fixture(scope="session")
def start_time_yyyymmdd() -> int:
    return _START_TIME


@pytest.fixture(scope="session")
def header_text() -> str:
    return _HEADER
