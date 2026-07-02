#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# ///
"""P12-nvec PR-N0 share-table derivation.

Reads a nvec_prof.csv (op counts + total_ns per op, classed
elementwise/reduction/other) plus the sibling profile_B0.yaml (for the
t_CVODE_raw denominator, in seconds), and prints the per-case share table:

    elementwise share  = sum(total_ns of elementwise ops) / t_CVODE_raw_ns
    reduction   share  = sum(total_ns of reduction   ops) / t_CVODE_raw_ns
    other (clone) share= sum(total_ns of other       ops) / t_CVODE_raw_ns
    non-NVector remainder = 1 - (elementwise + reduction + other)

All shares are of t_CVODE_raw (the whole CVode() call). The reduction
absolute total_ns is the G-E3(iii) t_red input; the reduction share is the
G-E3(ii) input; the elementwise share is the G-E2 Tier-1 upside bound.

Usage:
    derive_shares.py <nvec_prof.csv> <profile_B0.yaml> <label>
"""
import sys
import csv


def read_tcvode_raw_ns(yaml_path):
    """Extract extras.t_CVODE_raw (seconds) from the profile YAML → ns."""
    t_cvode_raw_s = None
    with open(yaml_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("t_CVODE_raw:"):
                t_cvode_raw_s = float(s.split(":", 1)[1].strip())
                break
    if t_cvode_raw_s is None:
        raise SystemExit(f"t_CVODE_raw not found in {yaml_path}")
    return t_cvode_raw_s, int(round(t_cvode_raw_s * 1e9))


def read_csv(csv_path):
    header = {}
    rows = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#"):
                # "# key=value"
                kv = line[1:].strip()
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    header[k.strip()] = v.strip()
                continue
            break
    with open(csv_path) as f:
        # skip header/comment lines, then DictReader over the data rows
        data_lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(data_lines)
    for r in reader:
        rows.append(r)
    return header, rows


def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    csv_path, yaml_path, label = sys.argv[1], sys.argv[2], sys.argv[3]

    header, rows = read_csv(csv_path)
    t_raw_s, t_raw_ns = read_tcvode_raw_ns(yaml_path)

    cls_ns = {"elementwise": 0, "reduction": 0, "other": 0}
    cls_calls = {"elementwise": 0, "reduction": 0, "other": 0}
    for r in rows:
        c = r["op_class"]
        cls_ns[c] = cls_ns.get(c, 0) + int(r["total_ns"])
        cls_calls[c] = cls_calls.get(c, 0) + int(r["calls"])

    nvec_ns = sum(cls_ns.values())
    remainder_ns = t_raw_ns - nvec_ns

    def share(ns):
        return 100.0 * ns / t_raw_ns if t_raw_ns > 0 else float("nan")

    print(f"=== share table: {label} ===")
    print(f"project_name = {header.get('project_name')}")
    print(f"NY           = {header.get('NY')}")
    print(f"nthreads (N) = {header.get('nthreads')}   backend = {header.get('backend')}")
    print(f"t_CVODE_raw  = {t_raw_s:.6f} s  ({t_raw_ns} ns)")
    print("-" * 68)
    print(f"{'class':<14}{'calls':>14}{'total_ns':>18}{'share_of_tCVraw':>18}")
    for c in ("elementwise", "reduction", "other"):
        print(f"{c:<14}{cls_calls[c]:>14}{cls_ns[c]:>18}{share(cls_ns[c]):>17.3f}%")
    print(f"{'NVector sum':<14}{'':>14}{nvec_ns:>18}{share(nvec_ns):>17.3f}%")
    print(f"{'non-NVector':<14}{'':>14}{remainder_ns:>18}{share(remainder_ns):>17.3f}%")
    print("-" * 68)
    print(f"G-E3(ii)  reduction share = {share(cls_ns['reduction']):.3f}% of t_CVODE_raw")
    print(f"G-E3(iii) t_red (reduction total_ns) = {cls_ns['reduction']} ns "
          f"({cls_ns['reduction']/1e9:.6f} s)")
    print(f"G-E2      elementwise upside share = {share(cls_ns['elementwise']):.3f}% of t_CVODE_raw")
    print()


if __name__ == "__main__":
    main()
