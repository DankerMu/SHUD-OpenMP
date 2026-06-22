#!/usr/bin/env python3
"""check_numa_token_order.py — S5d.3 (#181) [NUMA] log-token ordering gate.

Asserts spec L75-77 ordering on a SHUD run log:

    min_line('[NUMA] OMP_PROC_BIND=') < min_line('[NUMA] first-touch begin')

If the second pattern has zero matches (OMP_PROC_BIND unset path) the
ordering assertion is vacuously satisfied — we only require the first
token to be present. If --mode set is supplied we additionally require
the second pattern to have AT LEAST 1 match; if --mode unset we
additionally require the second pattern to have ZERO matches.

Usage:
    python3 check_numa_token_order.py <run.log> [--mode set|unset]

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import sys
from pathlib import Path


def find_min_line(lines, prefix):
    for i, line in enumerate(lines, start=1):
        if line.startswith(prefix):
            return i
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_log", type=Path, help="path to SHUD run log (stdout capture)")
    p.add_argument(
        "--mode",
        choices=("set", "unset", "any"),
        default="any",
        help="OMP_PROC_BIND env mode the log was produced under",
    )
    args = p.parse_args()

    if not args.run_log.is_file():
        print(f"FAIL: log not found: {args.run_log}")
        return 1

    text = args.run_log.read_text(errors="replace").splitlines()

    bind_line = find_min_line(text, "[NUMA] OMP_PROC_BIND=")
    touch_line = find_min_line(text, "[NUMA] first-touch begin")

    if bind_line is None:
        print(f"FAIL: '[NUMA] OMP_PROC_BIND=' line missing from {args.run_log}")
        return 1

    if args.mode == "set":
        if touch_line is None:
            print(
                f"FAIL: --mode set requires '[NUMA] first-touch begin' present, "
                f"none found in {args.run_log}"
            )
            return 1
        if bind_line >= touch_line:
            print(
                f"FAIL: bind_line={bind_line} >= touch_line={touch_line} "
                f"(ordering violated) in {args.run_log}"
            )
            return 1
        print(
            f"PASS: bind_line={bind_line} < touch_line={touch_line} "
            f"({args.run_log.name})"
        )
        return 0

    if args.mode == "unset":
        if touch_line is not None:
            print(
                f"FAIL: --mode unset expects zero '[NUMA] first-touch begin' lines, "
                f"found one at line {touch_line} in {args.run_log}"
            )
            return 1
        print(
            f"PASS: bind_line={bind_line}, no '[NUMA] first-touch begin' lines "
            f"({args.run_log.name})"
        )
        return 0

    # mode=any
    if touch_line is None:
        print(
            f"PASS (vacuous): bind_line={bind_line}, no '[NUMA] first-touch begin' "
            f"lines (skip mode or no parallel path) ({args.run_log.name})"
        )
        return 0
    if bind_line >= touch_line:
        print(
            f"FAIL: bind_line={bind_line} >= touch_line={touch_line} "
            f"in {args.run_log}"
        )
        return 1
    print(
        f"PASS: bind_line={bind_line} < touch_line={touch_line} "
        f"({args.run_log.name})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
