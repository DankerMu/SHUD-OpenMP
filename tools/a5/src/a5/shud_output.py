"""SHUD binary .dat output reader.

Format (version >= 2, matches SHUD/src/classes/Model_Control.cpp
Print_Ctrl::open_file() and rSHUD/R/readout.R):

    Offset (bytes)  | Content                              | dtype
    ----------------+--------------------------------------+---------
    0               | header (project metadata, 1024 char) | char[1024]
    1024            | start_time (YYYYMMDD as float64)     | float64
    1032            | NumVar (nc, as float64)              | float64
    1040            | column indices idx[nc]               | float64[nc]
    1040 + 8*nc     | row 0: [t_minutes, x1..xnc]          | float64[nc+1]
    ...             | row k: ...                           | float64[nc+1]

All doubles are little-endian (SHUD runs on x86_64 / arm64 hosts;
big-endian hosts are unsupported by the reference implementation).

t_minutes is minutes elapsed since start_time (parsed as YYYYMMDD).

Every element output file (elevprcp, elevet*, elevgw, eleysurf, ...) and
river output file (rivqdown, rivystage, ...) shares this layout — they
differ only in NumVar (= number of elements or number of river segments).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

HEADER_BYTES = 1024
HEADER_DOUBLES = HEADER_BYTES // 8  # 128
_DOUBLE_BYTES = 8


@dataclass(frozen=True)
class ShudSeries:
    """Parsed SHUD output series.

    Attributes:
        header:     the raw 1024-byte header, decoded as ASCII (nulls stripped).
        start_time: the model start-time as a UTC datetime.
        nc:         NumVar — number of columns (elements or river segments).
        idx:        column index array of length nc (1-based, per SHUD convention).
        t_minutes:  shape (nrows,) — minutes elapsed since start_time.
        timestamps: shape (nrows,) — datetime[64] UTC absolute timestamps.
        values:     shape (nrows, nc) — the payload matrix.
    """

    header: str
    start_time: datetime
    nc: int
    idx: np.ndarray
    t_minutes: np.ndarray
    timestamps: np.ndarray
    values: np.ndarray


def _parse_start_time(st_double: float) -> datetime:
    """Decode SHUD start-time double (YYYYMMDD encoded as float64) to datetime."""
    # SHUD writes StartTime as `long` cast to double. Values look like
    # 19510101.0 for 1951-01-01. Round to nearest int to defeat any
    # float representation drift.
    st_int = int(round(st_double))
    s = f"{st_int:08d}"
    try:
        return datetime.strptime(s, "%Y%m%d").replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise ValueError(
            f"SHUD start_time double {st_double!r} did not parse as YYYYMMDD "
            f"(rendered as {s!r}): {exc}"
        ) from exc


def _read_series(path: Path) -> ShudSeries:
    """Low-level reader: parse header + st + nc + idx + data matrix."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SHUD .dat file not found: {path}")
    if not path.is_file():
        raise ValueError(f"SHUD .dat path exists but is not a regular file: {path}")

    raw = path.read_bytes()
    if len(raw) < HEADER_BYTES + 2 * _DOUBLE_BYTES:
        raise ValueError(
            f"SHUD .dat {path} is truncated: {len(raw)} bytes < minimum "
            f"{HEADER_BYTES + 2 * _DOUBLE_BYTES} (header + st + nc)"
        )

    header_bytes = raw[:HEADER_BYTES]
    # Header is a fixed-width null-padded C string.
    header_str = header_bytes.split(b"\x00", 1)[0].decode("ascii", errors="replace")

    arr = np.frombuffer(raw, dtype="<f8")
    if arr.size < HEADER_DOUBLES + 2:
        raise ValueError(
            f"SHUD .dat {path}: fewer than {HEADER_DOUBLES + 2} doubles after "
            f"header; file corrupted"
        )

    st_double = float(arr[HEADER_DOUBLES])
    nc_double = float(arr[HEADER_DOUBLES + 1])
    nc = int(round(nc_double))
    if nc <= 0:
        raise ValueError(
            f"SHUD .dat {path}: parsed NumVar={nc_double} which rounds to "
            f"nc={nc}. Suspected wrong endianness or file corruption."
        )

    idx_start = HEADER_DOUBLES + 2
    idx_end = idx_start + nc
    if arr.size < idx_end:
        raise ValueError(
            f"SHUD .dat {path}: expected {nc} column-index doubles but only "
            f"{arr.size - idx_start} present"
        )
    idx = arr[idx_start:idx_end].copy()

    remaining = arr[idx_end:]
    row_len = nc + 1
    nrows_float = remaining.size / row_len
    nrows = remaining.size // row_len
    if nrows_float != nrows:
        # Partial final row -> data drop, but do not fail hard: the R
        # readout() emits a `message` and returns the truncated matrix.
        # Mirror that behavior.
        pass
    if nrows == 0:
        # Header-only file (e.g., simulation aborted before first flush).
        t_min = np.zeros(0, dtype=np.float64)
        values = np.zeros((0, nc), dtype=np.float64)
    else:
        mat = remaining[: nrows * row_len].reshape(nrows, row_len)
        t_min = mat[:, 0].copy()
        values = mat[:, 1:].copy()

    start_time = _parse_start_time(st_double)
    # Convert to naive datetimes (drop tz) before np.datetime64 to silence
    # numpy's tz-drop UserWarning; the tz is preserved in `start_time`.
    _start_naive = start_time.replace(tzinfo=None)
    timestamps = np.array(
        [_start_naive + timedelta(minutes=float(m)) for m in t_min],
        dtype="datetime64[us]",
    )

    return ShudSeries(
        header=header_str,
        start_time=start_time,
        nc=nc,
        idx=idx,
        t_minutes=t_min,
        timestamps=timestamps,
        values=values,
    )


def read_series(path: str | Path) -> ShudSeries:
    """Read a SHUD .dat output file. See ShudSeries for the returned fields."""
    return _read_series(Path(path))


def read_rivqdown(path: str | Path, outlet: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read a *.rivqdown.dat and return (timestamps, discharge_at_outlet).

    Args:
        path:   path to the .rivqdown.dat file.
        outlet: 0-based column index of the outlet. Default: last column
                (SHUD writes rivers in strahler-descending order, so the
                highest-index segment is typically the outlet).

    Returns:
        (timestamps: datetime64[us] shape (T,), discharge: float64 shape (T,))
    """
    series = read_series(path)
    if series.values.size == 0:
        return series.timestamps, np.zeros(0, dtype=np.float64)
    if outlet is None:
        outlet = series.nc - 1
    if not 0 <= outlet < series.nc:
        raise IndexError(
            f"outlet column {outlet} out of range [0, {series.nc})"
        )
    return series.timestamps, series.values[:, outlet].copy()


def read_dt(path: str | Path) -> np.ndarray:
    """Read the DY.dat timestamp array (one double per emitted timestep).

    DY.dat is written by SHUD alongside the .dat outputs; it stores the
    ODE integrator step used at each output flush (in days). Returns the
    raw double array; callers may cumulatively sum to derive elapsed time.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DY.dat not found: {path}")
    raw = path.read_bytes()
    if len(raw) % _DOUBLE_BYTES != 0:
        raise ValueError(
            f"DY.dat {path} size {len(raw)} not a multiple of 8; corrupted"
        )
    return np.frombuffer(raw, dtype="<f8").copy()
