#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# ///
"""P12-nvec PR-N0 profiler-overhead gate (heihe_x4 Config C N=16).

Reads the batch .out log (MARKER:P12_CELL lines carrying wall_s per rep) for
3 gate-off reps (SHUD_NVEC_PROF unset) and 3 gate-on reps (SHUD_NVEC_PROF=1),
computes the median wall of each arm, and the overhead ratio.

Gate: overhead = (median_on - median_off) / median_off <= 2% (NORMATIVE).

Usage: overhead_gate.py <batch_log.out>
Parses lines like:
  MARKER:P12_CELL label=off-rep1 case=heihe_x4 N=16 prof=0 wall_s=671.2 rc=0
  MARKER:P12_CELL label=on-rep1  case=heihe_x4 N=16 prof=1 wall_s=678.9 rc=0
"""
import sys
import re
import statistics


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    off, on = [], []
    pat = re.compile(r"MARKER:P12_CELL .*prof=(\d).*wall_s=([0-9.]+).*rc=(\d)")
    with open(sys.argv[1]) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            prof, wall, rc = int(m.group(1)), float(m.group(2)), int(m.group(3))
            if rc != 0:
                print(f"WARN: non-zero rc in line: {line.strip()}")
                continue
            (on if prof == 1 else off).append(wall)

    print(f"gate-off reps (prof=0): {off}")
    print(f"gate-on  reps (prof=1): {on}")
    if len(off) < 1 or len(on) < 1:
        raise SystemExit("need >=1 rep per arm")

    med_off = statistics.median(off)
    med_on = statistics.median(on)
    overhead = (med_on - med_off) / med_off * 100.0
    print("-" * 56)
    print(f"median gate-off wall = {med_off:.3f} s")
    print(f"median gate-on  wall = {med_on:.3f} s")
    print(f"overhead = (on-off)/off = {overhead:+.3f}%")
    print(f"NORMATIVE gate (<= 2.0%): {'PASS' if overhead <= 2.0 else 'FAIL'}")


if __name__ == "__main__":
    main()
