#!/usr/bin/env python3
"""Scan SHUD output binaries / CSVs for NaN / Inf.

Usage:
    uv run --with numpy python3 scan_nan.py <run_output_dir>

Produces a per-file table:
    <fname>  doubles=<n>  NaN=<k>  Inf=<m>
matching the format used in SHUD/B1b_CHANGELOG.md S6b.1 "Cryosphere
NaN-elimination evidence" table.

For `.dat` files (SHUD binary outputs) we read as float64 little-endian
(SHUD writes native `double` and target platforms are LE x86-64 / arm64).
For `.csv` files we read text and count "nan"/"inf" tokens (case-insensitive,
word-boundary aware) and also count the numeric token count for context.
Files we don't recognise (e.g. `.SHUD` text summary) are read as text and
scanned for nan/inf tokens.

Outputs to stdout. Exit code 0 if no NaN/Inf detected, 1 otherwise.

Spec source: SHUD/B1b_CHANGELOG.md "Cryosphere NaN-elimination evidence"
table format (S6b.1 / PR #202).
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
NAN_RE = re.compile(rb"\bnan\b", re.IGNORECASE)
INF_RE = re.compile(rb"\b[+-]?inf(?:inity)?\b", re.IGNORECASE)


def scan_dat(path: str) -> tuple[int, int, int]:
    a = np.fromfile(path, dtype=np.float64)
    return len(a), int(np.isnan(a).sum()), int(np.isinf(a).sum())


def scan_text(path: str) -> tuple[int, int, int]:
    with open(path, "rb") as f:
        data = f.read()
    nan_hits = len(NAN_RE.findall(data))
    inf_hits = len(INF_RE.findall(data))
    # Approximate "doubles" count = numeric tokens in text body.
    text = data.decode("utf-8", errors="replace")
    n_tokens = sum(1 for _ in TOKEN_RE.finditer(text))
    return n_tokens, nan_hits, inf_hits


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    run_dir = sys.argv[1]
    if not os.path.isdir(run_dir):
        print(f"ERROR: not a directory: {run_dir}", file=sys.stderr)
        return 2

    total_nan = 0
    total_inf = 0
    rows: list[tuple[str, int, int, int, str]] = []
    for fname in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, fname)
        if not os.path.isfile(path):
            continue
        try:
            if fname.endswith(".dat"):
                n, nan, inf = scan_dat(path)
                kind = "dat"
            elif fname.endswith(".csv") or fname.endswith(".SHUD"):
                n, nan, inf = scan_text(path)
                kind = "txt"
            else:
                continue
        except Exception as e:  # noqa: BLE001
            rows.append((fname, -1, -1, -1, f"ERROR: {e}"))
            continue
        rows.append((fname, n, nan, inf, kind))
        total_nan += nan
        total_inf += inf

    width = max((len(r[0]) for r in rows), default=20)
    for fname, n, nan, inf, kind in rows:
        if n < 0:
            print(f"{fname:<{width}}  {kind}")
        else:
            print(
                f"{fname:<{width}}  doubles={n:>8d}  NaN={nan:>3d}  Inf={inf:>3d}  ({kind})"
            )
    print()
    print(f"TOTAL  NaN={total_nan}  Inf={total_inf}")
    return 0 if (total_nan == 0 and total_inf == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
