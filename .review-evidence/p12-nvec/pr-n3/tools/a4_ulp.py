#!/usr/bin/env python3
"""P12-nvec PR-N3 — A4 max-ULP comparer (Config E2 vs Config E).

No pass/fail threshold at this step (spec G-E4(2)): reports the max ULP
distance between two SHUD binary outputs so the PR can DECIDE whether Neumaier
compensation is needed. Reuses the tools/a5 SHUD reader (same on-disk format,
ver>=2). Compares the full value matrix (all columns, all rows) of a chosen
`*.<keyword>.dat` output.

ULP distance is computed on the IEEE-754 float64 bit pattern (monotone integer
ordering trick): reinterpret each double as int64, map negatives to a monotone
range, and take |a_int - b_int|. 0 == bitwise identical.

Usage (from tools/a5 so the a5 package imports):
  uv run python /path/to/a4_ulp.py REF_OUT_DIR CAND_OUT_DIR [keyword]
    REF_OUT_DIR  = Config E  .out dir (baseline for the A4 report)
    CAND_OUT_DIR = Config E2 .out dir (candidate)
    keyword      = rivqdown (default) | elevprcp | eleveta | eleygw | ...
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# a5 is on sys.path when run via `uv run` from tools/a5
from a5.shud_output import read_series


def _find_dat(d: Path, kw: str) -> Path:
    m = sorted(d.glob(f"*.{kw}.dat"), key=lambda p: (len(p.name), p.name))
    if not m:
        raise SystemExit(f"no *.{kw}.dat in {d}")
    return m[0]


def _monotone_key(x: np.ndarray) -> np.ndarray:
    """Map float64 bit patterns to a monotone uint64 ordering key so that
    integer distance == ULP distance. Non-negative floats: raw bits are
    already monotone. Negative floats: flip to `0x8000... - bits` computed in
    UINT64 arithmetic (avoids the int64 overflow of 2**63)."""
    bits = x.astype(np.float64).view(np.uint64).copy()
    neg = (bits >> np.uint64(63)) == np.uint64(1)  # sign bit set
    SIGN = np.uint64(0x8000000000000000)
    bits[neg] = SIGN - bits[neg]
    return bits


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise ULP distance between two float64 arrays (>=0, 0==bitwise)."""
    ka = _monotone_key(a).astype(object)  # object -> Python bigints, no overflow
    kb = _monotone_key(b).astype(object)
    return np.abs(ka - kb).astype(np.float64)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    ref_dir = Path(sys.argv[1])
    cand_dir = Path(sys.argv[2])
    kw = sys.argv[3] if len(sys.argv) > 3 else "rivqdown"

    ref = read_series(_find_dat(ref_dir, kw))
    cand = read_series(_find_dat(cand_dir, kw))
    ra, ca = ref.values, cand.values
    if ra.shape != ca.shape:
        # align on the common prefix (defensive; expected identical)
        r = min(ra.shape[0], ca.shape[0])
        c = min(ra.shape[1], ca.shape[1])
        ra, ca = ra[:r, :c], ca[:r, :c]
        print(f"WARN: shape mismatch, compared common block {ra.shape}")

    d = ulp_distance(ra.ravel(), ca.ravel())
    absdiff = np.abs(ra.ravel() - ca.ravel())
    n_ident = int((d == 0).sum())
    total = d.size
    max_ulp = float(d.max()) if total else 0.0
    # relative magnitude of the max-ulp element
    imax = int(d.argmax()) if total else 0
    ref_at = float(ra.ravel()[imax]) if total else 0.0
    cand_at = float(ca.ravel()[imax]) if total else 0.0

    print(f"A4_ULP keyword={kw}")
    print(f"  shape={ra.shape} elements={total}")
    print(f"  bitwise_identical_elements={n_ident}/{total} ({100.0*n_ident/total:.3f}%)")
    print(f"  max_ulp={max_ulp:.0f}")
    print(f"  max_abs_diff={float(absdiff.max()):.6e}")
    print(f"  at_max_ulp: ref={ref_at:.17g} cand={cand_at:.17g}")
    print(f"MARKER:A4_ULP keyword={kw} max_ulp={max_ulp:.0f} identical={n_ident}/{total} max_abs_diff={float(absdiff.max()):.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
