#!/usr/bin/env -S uv run python
"""dense_fd_cross_check.py — brute-force dense FD vs colored FD cross-check.

P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
`p8tune-klu-spike` spec REQ-3 Scenario "keliya tool-correctness gate
via independent ground-truth reference":

    In addition, the FD-probed numeric J for keliya SHALL be cross-checked
    against a brute-force dense FD on keliya (NumY ~= 1.5K is small enough
    for an O(NumY) column-by-column dense baseline) -- relative error
    <= 1e-6 per nonzero entry.

The dense FD baseline is produced by `fd_color_jacobian --brute-force-dense`
(emits `<case>_numeric_J_dense.bin` next to the colored `<case>_numeric_J.bin`).
This script loads BOTH binaries and computes the real per-entry relative
error per 5-block pair:

    max_rel_err = max over (i,k) in CSC pattern of
                  |dut[i,k] - ref[i,k]| / max(|ref[i,k]|, |dut[i,k]|, 1e-12)

PASS criterion: max_rel_err <= 1e-6 per block (per spec).

(PR-0 reviewer round 1 finding c01 / F1 fix: prior version hardcoded
max_rel_err = 0.0 and never invoked an independent probe.)

Usage:
    uv run dense_fd_cross_check.py [--case keliya] [--jbin PATH] [--jbin-dense PATH] [--tsv PATH]
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path


def default_jbin(case: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "SHUD" / "Basins" / case / f"{case}_numeric_J.bin"
    )


def default_jbin_dense(case: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "SHUD" / "Basins" / case / f"{case}_numeric_J_dense.bin"
    )


def default_tsv(case: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / ".review-evidence"
        / "p8tune-klu-spike-pr-0"
        / f"mac_smoke_{case}_dense_fd_cross_check.tsv"
    )


BLOCKS = ["surf", "unsat", "gw", "river", "lake"]

# Per-case (NumEle, NumRiv, NumLake) for block_of_row indexing.
# keliya: NumEle=484, NumRiv=333 (from dump_adjacency), NumLake=0.
# (keliya.sp.riv has 333 rivers; rivseg count differs.)
CASE_DIMS = {
    "keliya": (484, 333, 0),
}


def block_of_row(row: int, NumEle: int, NumRiv: int, NumLake: int) -> int:
    if row < NumEle:
        return 0
    if row < 2 * NumEle:
        return 1
    if row < 3 * NumEle:
        return 2
    if row < 3 * NumEle + NumRiv:
        return 3
    return 4


def read_jbin(path: Path) -> dict:
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x4A4E444E:
            raise ValueError(f"bad magic 0x{magic:08x}; expected 0x4A4E444E (NDNJ) in {path}")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"unsupported version {version} in {path}")
        NumY, total_nnz, n_colors = struct.unpack("<QQQ", f.read(24))
        col_ptr = list(struct.unpack(f"<{NumY+1}q", f.read(8 * (NumY + 1))))
        row_idx = list(struct.unpack(f"<{total_nnz}i", f.read(4 * total_nnz)))
        values = list(struct.unpack(f"<{total_nnz}d", f.read(8 * total_nnz)))
    return {
        "NumY": NumY,
        "total_nnz": total_nnz,
        "n_colors": n_colors,
        "col_ptr": col_ptr,
        "row_idx": row_idx,
        "values": values,
    }


def cross_check_real(
    dut: dict,
    ref: dict,
    NumEle: int,
    NumRiv: int,
    NumLake: int,
) -> dict:
    """Per-entry relative-error comparison across all pattern entries,
    grouped by 5-block (block_row, block_col) pair.

    Both `dut` (colored FD) and `ref` (dense FD) MUST share identical
    CSC layout (NumY + total_nnz + col_ptr + row_idx). If they diverge,
    raise ValueError -- that itself is a tool-correctness failure.
    """
    if dut["NumY"] != ref["NumY"]:
        raise ValueError(f"NumY mismatch: dut={dut['NumY']} ref={ref['NumY']}")
    if dut["total_nnz"] != ref["total_nnz"]:
        raise ValueError(f"total_nnz mismatch: dut={dut['total_nnz']} ref={ref['total_nnz']}")
    if dut["col_ptr"] != ref["col_ptr"]:
        raise ValueError("col_ptr mismatch between dut and ref")
    if dut["row_idx"] != ref["row_idx"]:
        raise ValueError("row_idx mismatch between dut and ref")

    out = {(r, c): {"max_rel_err": 0.0, "n_nonzero": 0, "n_zero": 0,
                     "n_finite_viol": 0, "max_abs_dut": 0.0}
           for r in range(5) for c in range(5)}
    NumY = dut["NumY"]
    for col in range(NumY):
        s, e = dut["col_ptr"][col], dut["col_ptr"][col + 1]
        bc = block_of_row(col, NumEle, NumRiv, NumLake)
        for p in range(s, e):
            row = dut["row_idx"][p]
            v_dut = dut["values"][p]
            v_ref = ref["values"][p]
            br = block_of_row(row, NumEle, NumRiv, NumLake)
            d = out[(br, bc)]
            if (math.isnan(v_dut) or math.isinf(v_dut)
                    or math.isnan(v_ref) or math.isinf(v_ref)):
                d["n_finite_viol"] += 1
                continue
            if v_dut == 0 and v_ref == 0:
                d["n_zero"] += 1
            else:
                d["n_nonzero"] += 1
                if abs(v_dut) > d["max_abs_dut"]:
                    d["max_abs_dut"] = abs(v_dut)
                denom = max(abs(v_ref), abs(v_dut), 1e-12)
                rel_err = abs(v_dut - v_ref) / denom
                if rel_err > d["max_rel_err"]:
                    d["max_rel_err"] = rel_err
    return out


def write_tsv(path: Path, summary: dict, dut: dict, ref: dict, threshold: float) -> bool:
    """Emit TSV; return True if every block has max_rel_err <= threshold AND no finite violation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [["block_row", "block_col", "n_nonzero", "n_zero", "max_abs_dut",
             "n_finite_viol", "max_rel_err", "verdict"]]
    all_pass = True
    for r in range(5):
        for c in range(5):
            d = summary[(r, c)]
            ok_finite = d["n_finite_viol"] == 0
            # If no nonzeros in the block, max_rel_err stays 0 and trivially passes.
            ok_relerr = d["max_rel_err"] <= threshold
            verdict = "PASS" if (ok_finite and ok_relerr) else "FAIL"
            if verdict == "FAIL":
                all_pass = False
            rows.append([BLOCKS[r], BLOCKS[c], str(d["n_nonzero"]), str(d["n_zero"]),
                         f"{d['max_abs_dut']:.3e}", str(d["n_finite_viol"]),
                         f"{d['max_rel_err']:.3e}", verdict])
    rows.append([])
    rows.append(["NumY", str(dut["NumY"]), "", "", "", "", "", ""])
    rows.append(["total_nnz", str(dut["total_nnz"]), "", "", "", "", "", ""])
    rows.append(["n_colors_dut", str(dut["n_colors"]), "", "", "", "", "", ""])
    rows.append(["n_colors_ref_dense", str(ref["n_colors"]), "", "", "", "", "", ""])
    rows.append(["threshold", f"{threshold:.1e}", "", "", "", "", "", ""])
    rows.append(["overall", "PASS" if all_pass else "FAIL", "", "", "", "", "", ""])
    with path.open("w") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    return all_pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--case", type=str, default="keliya",
                   help="case name (default: keliya)")
    p.add_argument("--jbin", type=Path, default=None,
                   help="colored FD numeric J binary (default: SHUD/Basins/<case>/<case>_numeric_J.bin)")
    p.add_argument("--jbin-dense", type=Path, default=None,
                   help="dense FD numeric J binary reference (default: SHUD/Basins/<case>/<case>_numeric_J_dense.bin)")
    p.add_argument("--tsv", type=Path, default=None,
                   help="TSV output (default: .review-evidence/p8tune-klu-spike-pr-0/mac_smoke_<case>_dense_fd_cross_check.tsv)")
    p.add_argument("--threshold", type=float, default=1e-6,
                   help="per-block max_rel_err threshold (default: 1e-6 per spec REQ-3 L79)")
    args = p.parse_args()

    case = args.case
    jbin_path = args.jbin if args.jbin else default_jbin(case)
    jbin_dense_path = args.jbin_dense if args.jbin_dense else default_jbin_dense(case)
    tsv_path = args.tsv if args.tsv else default_tsv(case)

    if case not in CASE_DIMS:
        print(f"ERROR: dims for case '{case}' not registered in CASE_DIMS", file=sys.stderr)
        return 1

    NumEle, NumRiv, NumLake = CASE_DIMS[case]

    if not jbin_path.exists():
        print(f"ERROR: --jbin {jbin_path} not found (run fd_color_jacobian first)", file=sys.stderr)
        return 1
    if not jbin_dense_path.exists():
        print(f"ERROR: --jbin-dense {jbin_dense_path} not found (run fd_color_jacobian --brute-force-dense first)", file=sys.stderr)
        return 1

    dut = read_jbin(jbin_path)
    ref = read_jbin(jbin_dense_path)
    print(f"[dense_fd_cross_check] loaded DUT (colored): NumY={dut['NumY']} total_nnz={dut['total_nnz']} n_colors={dut['n_colors']}")
    print(f"[dense_fd_cross_check] loaded REF (dense):   NumY={ref['NumY']} total_nnz={ref['total_nnz']} n_colors={ref['n_colors']}")

    summary = cross_check_real(dut, ref, NumEle, NumRiv, NumLake)
    ok = write_tsv(tsv_path, summary, dut, ref, args.threshold)
    print(f"[dense_fd_cross_check] TSV: {tsv_path}")
    # Print per-block max_rel_err summary for quick eyeballing.
    print("[dense_fd_cross_check] per-block max_rel_err:")
    for r in range(5):
        for c in range(5):
            d = summary[(r, c)]
            if d["n_nonzero"] > 0:
                print(f"  {BLOCKS[r]:>5s} x {BLOCKS[c]:<5s}  nnz={d['n_nonzero']:>4d}  max_rel_err={d['max_rel_err']:.3e}")
    if ok:
        print(f"[dense_fd_cross_check] PASS: every block has max_rel_err <= {args.threshold:.1e}")
        return 0
    print(f"[dense_fd_cross_check] FAIL: at least one block exceeds threshold {args.threshold:.1e}; see TSV")
    return 1


if __name__ == "__main__":
    sys.exit(main())
