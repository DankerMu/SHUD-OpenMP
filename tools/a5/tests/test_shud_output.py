"""Tests for the SHUD .dat binary reader."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from a5.shud_output import read_dt, read_rivqdown, read_series


def test_read_series_roundtrip(
    synth_ref_dat: Path,
    n_cols: int,
    n_steps: int,
    start_time_yyyymmdd: int,
    header_text: str,
) -> None:
    s = read_series(synth_ref_dat)
    assert s.nc == n_cols
    assert s.values.shape == (n_steps, n_cols)
    assert s.timestamps.shape == (n_steps,)
    assert s.start_time.year == start_time_yyyymmdd // 10000
    assert s.header.startswith(header_text)
    # First step is 1440 minutes = 1 day after start_time
    assert s.t_minutes[0] == pytest.approx(1440.0)
    # Values must be strictly positive per fixture seed
    assert (s.values > 0).all()


def test_read_series_candidate_is_scaled_reference(
    synth_ref_dat: Path, synth_cand_dat: Path
) -> None:
    ref = read_series(synth_ref_dat)
    cand = read_series(synth_cand_dat)
    ratio = cand.values / ref.values
    # Deterministic 5% inflation by conftest
    assert np.allclose(ratio, 1.05, atol=1e-12)


def test_read_rivqdown_outlet_last_column_by_default(synth_ref_dat: Path) -> None:
    ts, q = read_rivqdown(synth_ref_dat)
    ref = read_series(synth_ref_dat)
    assert ts.shape == q.shape
    assert np.array_equal(q, ref.values[:, -1])


def test_read_rivqdown_explicit_outlet_index(synth_ref_dat: Path) -> None:
    ref = read_series(synth_ref_dat)
    _, q = read_rivqdown(synth_ref_dat, outlet=0)
    assert np.array_equal(q, ref.values[:, 0])


def test_read_rivqdown_bad_outlet_raises(synth_ref_dat: Path) -> None:
    with pytest.raises(IndexError):
        read_rivqdown(synth_ref_dat, outlet=999)


def test_read_series_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_series(tmp_path / "does_not_exist.dat")


def test_read_series_truncated_file(tmp_path: Path) -> None:
    p = tmp_path / "truncated.dat"
    p.write_bytes(b"\x00" * 100)  # far shorter than 1024 header
    with pytest.raises(ValueError, match="truncated"):
        read_series(p)


def test_read_series_wrong_endianness_flagged(tmp_path: Path) -> None:
    # Write header + a big-endian nc value that decodes to a huge little-endian
    # negative-magnitude float; the reader should reject with a NumVar error.
    p = tmp_path / "wrong_endian.dat"
    payload = bytearray()
    payload.extend(b"a5-test-fixture-v1".ljust(1024, b"\x00"))
    payload.extend(struct.pack(">d", 20260101))  # start_time BE
    payload.extend(struct.pack(">d", 10.0))  # nc BE — LE reader sees garbage
    p.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="NumVar"):
        read_series(p)


def test_read_dt_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "DY.dat"
    arr = np.array([0.1, 0.2, 0.3, 0.4], dtype="<f8")
    p.write_bytes(arr.tobytes())
    got = read_dt(p)
    assert np.allclose(got, arr)


def test_read_dt_bad_size(tmp_path: Path) -> None:
    p = tmp_path / "DY.dat"
    p.write_bytes(b"\x00" * 5)
    with pytest.raises(ValueError, match="not a multiple of 8"):
        read_dt(p)


def test_read_dt_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_dt(tmp_path / "nope.dat")
