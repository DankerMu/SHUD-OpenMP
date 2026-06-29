#!/usr/bin/env -S uv run python
"""verify_adjacency_keliya.py — independent Python ground-truth reference.

P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
`p8tune-klu-spike` spec REQ-3 Scenario "keliya tool-correctness gate via
independent ground-truth reference":

    PR-0 Mac smoke executes `dump_adjacency keliya` and SHALL verify the
    per-5-block nnz against this Python reference. The C++ dump nnz MUST
    exact-match the Python nnz (zero off-by-one tolerance).

This script reads keliya input files (mesh / riv / rivseg) and replicates
the adjacency-build logic of `tools/p8tune.D/dump_adjacency.cpp` in pure
Python — independent codepath, no SHUD libs linked. Emits a TSV under
.review-evidence/p8tune-klu-spike-pr-0/.

Usage:
    uv run verify_adjacency_keliya.py [--basins-root PATH] [--case keliya]
        [--csc PATH]   (path to dump_adjacency output for comparison)
        [--tsv PATH]   (TSV output path)

Exit code:
    0  exact-match (or no CSC supplied: emits ref TSV only)
    1  mismatch / IO error

Refs:
    openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-3
    tools/p8tune.D/dump_adjacency.cpp (mirror logic)
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_BASINS = Path(__file__).resolve().parents[2] / "SHUD" / "Basins"
DEFAULT_TSV = (
    Path(__file__).resolve().parents[2]
    / ".review-evidence"
    / "p8tune-klu-spike-pr-0"
    / "mac_smoke_keliya_verify_nnz.tsv"
)

# 5-block names matching dump_adjacency.cpp BlockId enum.
BLOCKS = ["surf", "unsat", "gw", "river", "lake"]


def read_table(path: Path) -> tuple[list[str], list[list[str]]]:
    """Read whitespace-separated SHUD input table: first line `<N>\\t<NCOLS>`,
    second line column names, then N data rows."""
    with path.open() as f:
        first = f.readline().split()
        n_rows = int(first[0])
        header = f.readline().split()
        rows = []
        for _ in range(n_rows):
            line = f.readline()
            if not line:
                break
            rows.append(line.split())
        return header, rows


def block_of_row(row: int, NumEle: int, NumRiv: int, NumLake: int) -> int:
    if row < NumEle:
        return 0  # surf
    if row < 2 * NumEle:
        return 1  # unsat
    if row < 3 * NumEle:
        return 2  # gw
    if row < 3 * NumEle + NumRiv:
        return 3  # river
    return 4  # lake


def build_keliya_adjacency(input_dir: Path):
    mesh_path = input_dir / "keliya.sp.mesh"
    riv_path = input_dir / "keliya.sp.riv"
    rivseg_path = input_dir / "keliya.sp.rivseg"
    att_path = input_dir / "keliya.sp.att"

    mesh_hdr, mesh_rows = read_table(mesh_path)
    riv_hdr, riv_rows = read_table(riv_path)
    rivseg_hdr, rivseg_rows = read_table(rivseg_path)
    att_hdr, att_rows = read_table(att_path)

    NumEle = len(mesh_rows)
    NumRiv = len(riv_rows)
    NumSegmt = len(rivseg_rows)

    # NumLake from att column iLake max (col 'LAKE')
    lake_col = att_hdr.index("LAKE")
    iLake_vals = [int(r[lake_col]) for r in att_rows]
    NumLake = max(iLake_vals) if iLake_vals and max(iLake_vals) > 0 else 0

    NumY = 3 * NumEle + NumRiv + NumLake

    # Build per-column sets of row indices for unique-counting.
    cols = defaultdict(set)

    def add_sym(a: int, b: int) -> None:
        cols[a].add(b)
        cols[b].add(a)

    def row_surf(i: int) -> int:
        return i

    def row_unsat(i: int) -> int:
        return NumEle + i

    def row_gw(i: int) -> int:
        return 2 * NumEle + i

    def row_riv(i: int) -> int:
        return 3 * NumEle + i

    def row_lake(i: int) -> int:
        return 3 * NumEle + NumRiv + i

    # diagonal
    for k in range(NumY):
        cols[k].add(k)

    # intra-element: surf↔unsat↔gw (3 sym pairs per elem)
    for i in range(NumEle):
        add_sym(row_surf(i), row_unsat(i))
        add_sym(row_surf(i), row_gw(i))
        add_sym(row_unsat(i), row_gw(i))

    # inter-element via Nabr1/Nabr2/Nabr3 (cols index in mesh header)
    nabr_cols = [mesh_hdr.index(c) for c in ("Nabr1", "Nabr2", "Nabr3")]
    for i, row in enumerate(mesh_rows):
        for nj in nabr_cols:
            inabr = int(row[nj]) - 1  # 1-based → 0-based; -1 = boundary (raw value 0)
            if 0 <= inabr < NumEle:
                add_sym(row_surf(i), row_surf(inabr))
                add_sym(row_gw(i), row_gw(inabr))

    # elem ↔ river via RivSeg (iRiv, iEle columns)
    iRiv_col = rivseg_hdr.index("iRiv")
    iEle_col = rivseg_hdr.index("iEle")
    for row in rivseg_rows:
        ir = int(row[iRiv_col]) - 1
        ie = int(row[iEle_col]) - 1
        if 0 <= ir < NumRiv and 0 <= ie < NumEle:
            add_sym(row_riv(ir), row_surf(ie))
            add_sym(row_riv(ir), row_gw(ie))

    # river ↔ river downstream (Riv 'Down' col)
    down_col = riv_hdr.index("Down")
    for i, row in enumerate(riv_rows):
        idown = int(row[down_col]) - 1
        if 0 <= idown < NumRiv:
            add_sym(row_riv(i), row_riv(idown))

    # NumLake = 0 for keliya; skip lake-coupling blocks (they'd be no-ops anyway).

    # per-block nnz tally
    per_block = [[0] * 5 for _ in range(5)]
    total_nnz = 0
    for col, rows in cols.items():
        total_nnz += len(rows)
        blk_col = block_of_row(col, NumEle, NumRiv, NumLake)
        for r in rows:
            blk_row = block_of_row(r, NumEle, NumRiv, NumLake)
            per_block[blk_row][blk_col] += 1

    return {
        "NumEle": NumEle,
        "NumRiv": NumRiv,
        "NumLake": NumLake,
        "NumY": NumY,
        "total_nnz": total_nnz,
        "per_block_nnz": per_block,
    }


def read_csc_header(path: Path) -> dict:
    """Parse the binary CSC header per dump_adjacency.cpp output schema."""
    with path.open("rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != 0x434A4441:
            raise ValueError(f"bad magic 0x{magic:08x}; expected 0x434A4441 (ADJC)")
        version = struct.unpack("<I", f.read(4))[0]
        if version != 1:
            raise ValueError(f"unsupported version {version}")
        NumEle, NumRiv, NumLake, NumY = struct.unpack("<QQQQ", f.read(32))
        total_nnz = struct.unpack("<Q", f.read(8))[0]
        per_block_flat = struct.unpack("<25Q", f.read(8 * 25))
        per_block = [list(per_block_flat[r * 5 : (r + 1) * 5]) for r in range(5)]
    return {
        "NumEle": NumEle,
        "NumRiv": NumRiv,
        "NumLake": NumLake,
        "NumY": NumY,
        "total_nnz": total_nnz,
        "per_block_nnz": per_block,
    }


def write_tsv(path: Path, ref: dict, dut: dict | None) -> bool:
    """Emit TSV; return True if match (or dut is None / no comparison)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    rows.append(["block_row", "block_col", "ref_nnz", "dut_nnz", "match"])
    all_match = True
    for r in range(5):
        for c in range(5):
            ref_nnz = ref["per_block_nnz"][r][c]
            if dut is not None:
                dut_nnz = dut["per_block_nnz"][r][c]
                match = "PASS" if ref_nnz == dut_nnz else "FAIL"
                if ref_nnz != dut_nnz:
                    all_match = False
            else:
                dut_nnz = ""
                match = ""
            rows.append([BLOCKS[r], BLOCKS[c], str(ref_nnz), str(dut_nnz), match])
    # summary footer
    rows.append([])
    rows.append(["case", "keliya", "", "", ""])
    rows.append(["NumEle", str(ref["NumEle"]), "", "", ""])
    rows.append(["NumRiv", str(ref["NumRiv"]), "", "", ""])
    rows.append(["NumLake", str(ref["NumLake"]), "", "", ""])
    rows.append(["NumY", str(ref["NumY"]), "", "", ""])
    rows.append(["total_nnz_ref", str(ref["total_nnz"]), "", "", ""])
    if dut is not None:
        rows.append(["total_nnz_dut", str(dut["total_nnz"]), "", "", ""])
        rows.append(["total_match", "PASS" if all_match else "FAIL", "", "", ""])
    with path.open("w") as f:
        for row in rows:
            f.write("\t".join(row) + "\n")
    return all_match


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--basins-root", type=Path, default=DEFAULT_BASINS)
    p.add_argument("--case", default="keliya")
    p.add_argument("--csc", type=Path, default=None, help="dump_adjacency CSC output")
    p.add_argument("--tsv", type=Path, default=DEFAULT_TSV)
    args = p.parse_args()

    if args.case != "keliya":
        print(
            f"ERROR: this script is keliya-specific; got case={args.case!r}", file=sys.stderr
        )
        return 1

    input_dir = args.basins_root / args.case / "input" / args.case
    if not input_dir.exists():
        print(f"ERROR: {input_dir} not found", file=sys.stderr)
        return 1

    ref = build_keliya_adjacency(input_dir)
    print(
        f"[verify_adjacency_keliya] reference: NumEle={ref['NumEle']} NumRiv={ref['NumRiv']} "
        f"NumLake={ref['NumLake']} NumY={ref['NumY']} total_nnz={ref['total_nnz']}"
    )

    dut = None
    if args.csc is not None:
        if not args.csc.exists():
            print(f"ERROR: --csc {args.csc} not found", file=sys.stderr)
            return 1
        dut = read_csc_header(args.csc)
        print(
            f"[verify_adjacency_keliya] CSC dut:     NumEle={dut['NumEle']} NumRiv={dut['NumRiv']} "
            f"NumLake={dut['NumLake']} NumY={dut['NumY']} total_nnz={dut['total_nnz']}"
        )

    match = write_tsv(args.tsv, ref, dut)
    print(f"[verify_adjacency_keliya] TSV written: {args.tsv}")
    if dut is not None:
        if match:
            print("[verify_adjacency_keliya] PASS: per-5-block nnz exact-match (ZERO off-by-one)")
            return 0
        else:
            print("[verify_adjacency_keliya] FAIL: per-5-block nnz mismatch (see TSV)")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
