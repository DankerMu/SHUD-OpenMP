#!/usr/bin/env bash
# tools/p8tune.D/aggregate_klu_spike.sh
#
# P8-tune.D KLU pattern-only spike (PR-B). Per openspec change
# `p8tune-klu-spike` tasks.md §3.1-3.4 + spec REQ-5 Scenario
# "Per-case axis machine-readable".
#
# Scans PR-A 16-cell sweep results, parses each cell-NN.log for the
# klu_analyze_factor `cell_summary` KV block OR one of the three failure
# markers (KLU_OOM_DETECTED / KLU_INDEX_OVERFLOW_DETECTED /
# KLU_WALL_OVERFLOW_DETECTED), emits:
#
#   1. ${OUT_DIR}/aggregate.tsv         — per-cell flat TSV (16 rows)
#   2. ${OUT_DIR}/aggregate_verdict.txt — machine-readable KV per spec D8
#
# Per-case 3-axis verdict (spec REQ-5):
#   fill_axis  PASS iff fill_ratio  < 8 * log2(NumY)  for best-combo cell
#   rss_axis   PASS iff peak_rss    < 0.7 * CN_NODE_RAM_BYTES
#   wall_axis  PASS iff (numeric_factor_wall / refactor_freq) +
#                       (N_solve * solve_wall)
#                       < 0.7 * SPGMR_per_step_wall (≈ 0.227 s)
#
# Verdict mapping (3-axis):
#   3/3 PASS                              -> GO
#   2/3 PASS (wall is borderline)         -> Optional
#   1/3 PASS or per-case asymmetric       -> Case-aware
#   0/3 PASS                              -> NO-GO
#
# Inputs (env vars / defaults):
#   CELLS_ROOT  .review-evidence/p8tune-klu-spike-pr-a/cells
#               (NN=00-07 under run-9762/; NN=08-15 under exit-fix-1782652046/)
#   OUT_DIR     .review-evidence/p8tune-klu-spike-pr-b
#   CN_NODE_RAM_BYTES_HDR  tools/p8tune.D/cn_node_ram.h
#   SPGMR_BASELINE_HDR     tools/p8tune.D/spgmr_baseline_walls.h
#
# Configurable wall-axis interpretation (env vars):
#   WALL_REFACTOR_FREQ          default 10  (refactor frequency in CVODE steps)
#   WALL_N_SOLVE                default 5   (per-step Newton iters)
#   WALL_SOLVE_FACTOR           default 0.1 (solve_wall as fraction of factor_wall)
#   WALL_SPGMR_BUDGET_FRACTION  default 0.7 (0.7 × SPGMR per-step wall is the budget)
#
# Outputs:
#   ${OUT_DIR}/aggregate.tsv           — 16-row TSV (header + 16 data rows)
#   ${OUT_DIR}/aggregate_verdict.txt   — flat KV verdict (consumer-friendly)
#
# Usage:
#   tools/p8tune.D/aggregate_klu_spike.sh
#   CELLS_ROOT=/path/to/cells OUT_DIR=/tmp/out tools/p8tune.D/aggregate_klu_spike.sh
#
# Exit codes:
#   0  OK
#   1  missing inputs / parse failure
#   2  python3 missing
#
# Refs:
#   openspec/changes/p8tune-klu-spike/tasks.md §3.1-3.4
#   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5/REQ-6
#   tools/p8tune/aggregate_maxl_sweep.sh (PR-D #373 reference template)

set -euo pipefail

# Default paths assume cwd = repo root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CELLS_ROOT="${CELLS_ROOT:-${REPO_ROOT}/.review-evidence/p8tune-klu-spike-pr-a/cells}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/.review-evidence/p8tune-klu-spike-pr-b}"
CN_NODE_RAM_BYTES_HDR="${CN_NODE_RAM_BYTES_HDR:-${SCRIPT_DIR}/cn_node_ram.h}"
SPGMR_BASELINE_HDR="${SPGMR_BASELINE_HDR:-${SCRIPT_DIR}/spgmr_baseline_walls.h}"

# Wall-axis tunables.
WALL_REFACTOR_FREQ="${WALL_REFACTOR_FREQ:-10}"
WALL_N_SOLVE="${WALL_N_SOLVE:-5}"
WALL_SOLVE_FACTOR="${WALL_SOLVE_FACTOR:-0.1}"
WALL_SPGMR_BUDGET_FRACTION="${WALL_SPGMR_BUDGET_FRACTION:-0.7}"

if [[ ! -d "${CELLS_ROOT}" ]]; then
    echo "ERROR: missing CELLS_ROOT directory: ${CELLS_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${CN_NODE_RAM_BYTES_HDR}" ]]; then
    echo "ERROR: missing cn_node_ram.h: ${CN_NODE_RAM_BYTES_HDR}" >&2
    exit 1
fi
if [[ ! -f "${SPGMR_BASELINE_HDR}" ]]; then
    echo "ERROR: missing spgmr_baseline_walls.h: ${SPGMR_BASELINE_HDR}" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 missing" >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

# Parse C++ headers for the pinned constants (single-source-of-truth).
# cn_node_ram.h: static constexpr std::size_t CN_NODE_RAM_BYTES = 185528156160ULL;
CN_NODE_RAM_BYTES="$(grep -oE 'CN_NODE_RAM_BYTES\s*=\s*[0-9]+' "${CN_NODE_RAM_BYTES_HDR}" | grep -oE '[0-9]+')"
# spgmr_baseline_walls.h: static constexpr double SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579;
SPGMR_PER_STEP_WALL_S="$(grep -oE 'SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S\s*=\s*[0-9.]+' "${SPGMR_BASELINE_HDR}" | grep -oE '[0-9.]+$')"

if [[ -z "${CN_NODE_RAM_BYTES}" || -z "${SPGMR_PER_STEP_WALL_S}" ]]; then
    echo "ERROR: failed to parse constants from C++ headers" >&2
    echo "  CN_NODE_RAM_BYTES = '${CN_NODE_RAM_BYTES}'" >&2
    echo "  SPGMR_PER_STEP_WALL_S = '${SPGMR_PER_STEP_WALL_S}'" >&2
    exit 1
fi

echo "INFO: CN_NODE_RAM_BYTES = ${CN_NODE_RAM_BYTES}"
echo "INFO: SPGMR_PER_STEP_WALL_S = ${SPGMR_PER_STEP_WALL_S}"
echo "INFO: WALL_REFACTOR_FREQ = ${WALL_REFACTOR_FREQ}"
echo "INFO: WALL_N_SOLVE = ${WALL_N_SOLVE}"
echo "INFO: WALL_SOLVE_FACTOR = ${WALL_SOLVE_FACTOR}"
echo "INFO: WALL_SPGMR_BUDGET_FRACTION = ${WALL_SPGMR_BUDGET_FRACTION}"
echo "INFO: CELLS_ROOT = ${CELLS_ROOT}"
echo "INFO: OUT_DIR = ${OUT_DIR}"

# Export env vars for python (PR-A flat-file layout + tunable constants).
export CELLS_ROOT OUT_DIR
export CN_NODE_RAM_BYTES SPGMR_PER_STEP_WALL_S
export WALL_REFACTOR_FREQ WALL_N_SOLVE WALL_SOLVE_FACTOR WALL_SPGMR_BUDGET_FRACTION

python3 - <<'PYEOF'
import os
import re
import sys
import math
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
CELLS_ROOT = Path(os.environ["CELLS_ROOT"])
OUT_DIR = Path(os.environ["OUT_DIR"])
CN_NODE_RAM_BYTES = int(os.environ["CN_NODE_RAM_BYTES"])
SPGMR_PER_STEP_WALL_S = float(os.environ["SPGMR_PER_STEP_WALL_S"])
WALL_REFACTOR_FREQ = float(os.environ["WALL_REFACTOR_FREQ"])
WALL_N_SOLVE = float(os.environ["WALL_N_SOLVE"])
WALL_SOLVE_FACTOR = float(os.environ["WALL_SOLVE_FACTOR"])
WALL_SPGMR_BUDGET_FRACTION = float(os.environ["WALL_SPGMR_BUDGET_FRACTION"])

WALL_BUDGET_S = WALL_SPGMR_BUDGET_FRACTION * SPGMR_PER_STEP_WALL_S

# Cell decoder per spec REQ-4 / spike_array.sbatch L108-111.
CASE_ARR = ["keliya", "heihe", "heihe_x4", "heihe_x16"]
ORDERING_ARR = ["natural", "amd", "amd", "colamd"]
BTF_ARR = [1, 0, 1, 1]

def decode_cell(nn):
    case_idx = nn // 4
    ord_idx = nn % 4
    return CASE_ARR[case_idx], ORDERING_ARR[ord_idx], BTF_ARR[ord_idx]

# Cell-NN file discovery: NN=00-07 in run-9762/, NN=08-15 in
# exit-fix-1782652046/. Per SWEEP_RESULTS.md table line 5: run-9762
# also contains stale NN=08-15 cells (pre-`_exit` fix; SHUD ~FloodAlert
# + Model_Data dtor UB crashed before KLU stage) — they must NOT be
# preferred. We follow the authoritative mapping verbatim.
#
# Per spec REQ-7 PR-A boundary (post-amendment), both layouts use the
# canonical flat cell-NN.{log,err,out,time} filenames. Single-digit
# cell-N.{err,out} files in run-9762 are stale Slurm-side stdout from
# before `_exit` fix and are NOT consulted.
AUTHORITATIVE_DIR = {
    0: "run-9762",
    1: "run-9762",
    2: "run-9762",
    3: "run-9762",
    4: "run-9762",
    5: "run-9762",
    6: "run-9762",
    7: "run-9762",
    8: "exit-fix-1782652046",
    9: "exit-fix-1782652046",
    10: "exit-fix-1782652046",
    11: "exit-fix-1782652046",
    12: "exit-fix-1782652046",
    13: "exit-fix-1782652046",
    14: "exit-fix-1782652046",
    15: "exit-fix-1782652046",
}

def find_cell_log(nn):
    nn_str = f"{nn:02d}"
    subdir_name = AUTHORITATIVE_DIR.get(nn)
    if subdir_name is None:
        return None
    candidate = CELLS_ROOT / subdir_name / f"cell-{nn_str}.log"
    if candidate.exists():
        return candidate
    # Fallback: scan all subdirs (covers future re-runs / renamed dirs).
    for subdir in CELLS_ROOT.iterdir():
        if not subdir.is_dir():
            continue
        candidate = subdir / f"cell-{nn_str}.log"
        if candidate.exists():
            return candidate
    return None

# ---------------------------------------------------------------------------
# Per-cell parser: extract klu_analyze_factor `cell_summary` KV OR
# one of three markers KLU_OOM_DETECTED / KLU_INDEX_OVERFLOW_DETECTED /
# KLU_WALL_OVERFLOW_DETECTED per spec REQ-5 amended scenarios.
# ---------------------------------------------------------------------------
def parse_cell_log(path):
    """Return a dict with parsed KVs and verdict_class."""
    result = {
        "log_path": str(path),
        "verdict_class": "UNKNOWN",
        "case": None,
        "ordering": None,
        "btf": None,
        "NumY": None,
        "Anz": None,
        "fill_ratio": None,
        "symbolic_wall_sec": None,
        "numeric_factor_wall_sec": None,
        "peak_rss_bytes": None,
        "numeric_lnz": None,
        "numeric_unz": None,
        "nblocks": None,
        "chromatic_number": None,
        "marker_line": None,
    }
    with open(path, "r", errors="replace") as f:
        text = f.read()

    # Markers (each is exit-0 data point per amended REQ-5).
    for marker in (
        "KLU_INDEX_OVERFLOW_DETECTED",
        "KLU_OOM_DETECTED",
        "KLU_WALL_OVERFLOW_DETECTED",
    ):
        m = re.search(r"^.*" + marker + r".*$", text, re.MULTILINE)
        if m:
            result["marker_line"] = m.group(0).strip()
            # Parse the marker line: case=<C> ordering=<O> btf=<B>
            #   peak_rss_bytes=<N> [reason=...] [elapsed_sec=N wall_budget_sec=W]
            mc = re.search(r"case=(\S+)", m.group(0))
            mo = re.search(r"ordering=(\S+)", m.group(0))
            mb = re.search(r"btf=(\d+)", m.group(0))
            mrss = re.search(r"peak_rss_bytes=(\d+)", m.group(0))
            if mc:
                result["case"] = mc.group(1)
            if mo:
                result["ordering"] = mo.group(1)
            if mb:
                result["btf"] = int(mb.group(1))
            if mrss:
                result["peak_rss_bytes"] = int(mrss.group(1))
            if marker == "KLU_INDEX_OVERFLOW_DETECTED":
                result["verdict_class"] = "fill_overflow"
            elif marker == "KLU_OOM_DETECTED":
                result["verdict_class"] = "rss_overflow"
            else:
                result["verdict_class"] = "wall_overflow"
            break

    # Also harvest NumY from any dump_adjacency line — useful when the
    # marker line lacks it (markers don't carry NumY).
    m = re.search(r"loaded case=(\S+):.*NumY=(\d+)", text)
    if m:
        if result["case"] is None:
            result["case"] = m.group(1)
        result["NumY"] = int(m.group(2))

    # chromatic_number from fd_color stage.
    m = re.search(r"chromatic_number=(\d+)", text)
    if m:
        result["chromatic_number"] = int(m.group(1))

    # If we already classified by marker, return (no cell_summary expected).
    if result["verdict_class"] != "UNKNOWN":
        return result

    # PASS path — parse cell_summary block.
    # Block structure (klu_analyze_factor.cpp output):
    #   [klu] cell_summary
    #     case=<C> ordering=<O> btf=<B>
    #     NumY=<N> Anz=<N>
    #     symbolic_lnz_est=<N> symbolic_unz_est=<N> est_flops=<N>
    #     nblocks=<N> maxblock=<N> nzoff=<N> symmetry=<N>
    #     numeric_lnz=<N> numeric_unz=<N> max_lnz_block=<N> max_unz_block=<N>
    #     fill_ratio=<F> (numeric_lnz+unz)/Anz
    #     symbolic_wall_sec=<F> numeric_factor_wall_sec=<F>
    #     peak_rss_bytes=<N> (CN_NODE_RAM_BYTES=<N>)
    block_match = re.search(
        r"\[klu\] cell_summary(.*?)(?:===|\Z)", text, re.DOTALL
    )
    if block_match:
        block = block_match.group(1)
        m = re.search(r"case=(\S+)\s+ordering=(\S+)\s+btf=(\d+)", block)
        if m:
            result["case"] = m.group(1)
            result["ordering"] = m.group(2)
            result["btf"] = int(m.group(3))
        m = re.search(r"NumY=(\d+)\s+Anz=(\d+)", block)
        if m:
            result["NumY"] = int(m.group(1))
            result["Anz"] = int(m.group(2))
        m = re.search(r"nblocks=(\d+)", block)
        if m:
            result["nblocks"] = int(m.group(1))
        m = re.search(r"numeric_lnz=(\d+)\s+numeric_unz=(\d+)", block)
        if m:
            result["numeric_lnz"] = int(m.group(1))
            result["numeric_unz"] = int(m.group(2))
        m = re.search(r"fill_ratio=([\d.eE+-]+)", block)
        if m:
            result["fill_ratio"] = float(m.group(1))
        m = re.search(r"symbolic_wall_sec=([\d.eE+-]+)", block)
        if m:
            result["symbolic_wall_sec"] = float(m.group(1))
        m = re.search(r"numeric_factor_wall_sec=([\d.eE+-]+)", block)
        if m:
            result["numeric_factor_wall_sec"] = float(m.group(1))
        m = re.search(r"peak_rss_bytes=(\d+)", block)
        if m:
            result["peak_rss_bytes"] = int(m.group(1))
        result["verdict_class"] = "PASS"

    return result

# ---------------------------------------------------------------------------
# Walk all 16 cells.
# ---------------------------------------------------------------------------
cells = []
for nn in range(16):
    log_path = find_cell_log(nn)
    expected_case, expected_ordering, expected_btf = decode_cell(nn)
    if log_path is None:
        cells.append({
            "nn": nn,
            "case": expected_case,
            "ordering": expected_ordering,
            "btf": expected_btf,
            "verdict_class": "MISSING",
            "log_path": None,
            "NumY": None, "Anz": None, "fill_ratio": None,
            "symbolic_wall_sec": None, "numeric_factor_wall_sec": None,
            "peak_rss_bytes": None, "numeric_lnz": None, "numeric_unz": None,
            "nblocks": None, "chromatic_number": None, "marker_line": None,
        })
        print(f"WARN: cell NN={nn:02d} ({expected_case}/{expected_ordering}/btf{expected_btf}) — log missing", file=sys.stderr)
        continue
    parsed = parse_cell_log(log_path)
    parsed["nn"] = nn
    # Backfill case/ordering/btf from decoder if the log didn't carry them
    # (e.g. early failures before the cell_summary block).
    if parsed["case"] is None:
        parsed["case"] = expected_case
    if parsed["ordering"] is None:
        parsed["ordering"] = expected_ordering
    if parsed["btf"] is None:
        parsed["btf"] = expected_btf
    # Consistency check: warn (not fail) if decoder ≠ log content.
    if parsed["case"] != expected_case or parsed["ordering"] != expected_ordering or int(parsed["btf"]) != expected_btf:
        print(
            f"WARN: cell NN={nn:02d} decoder ({expected_case}/{expected_ordering}/btf{expected_btf})"
            f" != log content ({parsed['case']}/{parsed['ordering']}/btf{parsed['btf']})",
            file=sys.stderr,
        )
    cells.append(parsed)

# ---------------------------------------------------------------------------
# Per-cell aggregate.tsv emission.
# ---------------------------------------------------------------------------
tsv_path = OUT_DIR / "aggregate.tsv"
with open(tsv_path, "w") as f:
    cols = [
        "nn", "case", "ordering", "btf",
        "NumY", "Anz",
        "fill_ratio", "numeric_lnz_plus_unz", "nnz_A",
        "symbolic_wall_s", "numeric_wall_s",
        "peak_rss_mb",
        "chromatic_number",
        "verdict_class",
    ]
    f.write("\t".join(cols) + "\n")
    for c in cells:
        lpu = (c["numeric_lnz"] or 0) + (c["numeric_unz"] or 0) if c["numeric_lnz"] is not None else ""
        rss_mb = ""
        if c["peak_rss_bytes"] is not None:
            rss_mb = f"{c['peak_rss_bytes'] / 1024.0 / 1024.0:.2f}"
        row = [
            f"{c['nn']:02d}",
            c["case"] or "",
            c["ordering"] or "",
            str(c["btf"]) if c["btf"] is not None else "",
            str(c["NumY"]) if c["NumY"] is not None else "",
            str(c["Anz"]) if c["Anz"] is not None else "",
            f"{c['fill_ratio']:.4f}" if c["fill_ratio"] is not None else "",
            str(lpu),
            str(c["Anz"]) if c["Anz"] is not None else "",
            f"{c['symbolic_wall_sec']:.6f}" if c["symbolic_wall_sec"] is not None else "",
            f"{c['numeric_factor_wall_sec']:.6f}" if c["numeric_factor_wall_sec"] is not None else "",
            rss_mb,
            str(c["chromatic_number"]) if c["chromatic_number"] is not None else "",
            c["verdict_class"],
        ]
        f.write("\t".join(row) + "\n")
print(f"OK: wrote {tsv_path}")

# ---------------------------------------------------------------------------
# Per-case best-combo selection + 3-axis verdict.
# Per spec REQ-5 + tasks §3.2: "lowest fill_ratio among PASS-fill combos
# OR lowest numeric_wall if all PASS-fill".
# In practice, AMD ordering wins per-case in the PR-A sweep — both AMD btf=0
# and AMD btf=1 cells have identical fill_ratio, so the tiebreaker is wall.
# ---------------------------------------------------------------------------
by_case = defaultdict(list)
for c in cells:
    by_case[c["case"]].append(c)

def per_case_verdict(case_name, case_cells):
    """Return dict with per-axis PASS/FAIL + overall verdict for one case."""
    # Best combo: lowest fill_ratio amongst cells that PASSed; break tie by
    # numeric_factor_wall_sec.
    pass_cells = [c for c in case_cells if c["verdict_class"] == "PASS" and c["fill_ratio"] is not None]
    failed_classes = [c["verdict_class"] for c in case_cells if c["verdict_class"] != "PASS"]
    best = None
    if pass_cells:
        best = sorted(
            pass_cells,
            key=lambda c: (c["fill_ratio"], c.get("numeric_factor_wall_sec") or float("inf")),
        )[0]
    # NumY (case-level) — prefer best-combo NumY; fall back to any cell.
    NumY = None
    for c in case_cells:
        if c["NumY"] is not None:
            NumY = c["NumY"]
            break
    # Fill axis threshold: 8 * log2(NumY)
    fill_threshold = 8.0 * math.log2(NumY) if NumY else None

    # ---------- Fill axis ----------
    fill_axis = "FAIL"
    fill_diag = ""
    if "fill_overflow" in failed_classes:
        # If ANY combo failed the index overflow, log it. But fill axis is
        # judged by the BEST combo — natural-ordering pathology doesn't
        # poison the case if AMD passes.
        if best is None:
            fill_axis = "FAIL"
            fill_diag = "all combos failed (fill_overflow on every ordering)"
        else:
            if best["fill_ratio"] < fill_threshold:
                fill_axis = "PASS"
                fill_diag = (
                    f"best-combo {best['ordering']}+btf{best['btf']} "
                    f"fill_ratio={best['fill_ratio']:.2f} < threshold=8·log₂({NumY})={fill_threshold:.1f}"
                    f"; natural-ordering pathology surfaced as fill_overflow data point (decisive, not blocking)"
                )
            else:
                fill_axis = "FAIL"
                fill_diag = (
                    f"best-combo {best['ordering']}+btf{best['btf']} "
                    f"fill_ratio={best['fill_ratio']:.2f} >= threshold=8·log₂({NumY})={fill_threshold:.1f}"
                )
    elif best is not None and fill_threshold is not None:
        if best["fill_ratio"] < fill_threshold:
            fill_axis = "PASS"
            fill_diag = (
                f"best-combo {best['ordering']}+btf{best['btf']} "
                f"fill_ratio={best['fill_ratio']:.2f} < threshold=8·log₂({NumY})={fill_threshold:.1f}"
            )
        else:
            fill_axis = "FAIL"
            fill_diag = (
                f"best-combo {best['ordering']}+btf{best['btf']} "
                f"fill_ratio={best['fill_ratio']:.2f} >= threshold=8·log₂({NumY})={fill_threshold:.1f}"
            )
    else:
        fill_axis = "FAIL"
        fill_diag = "no PASS combo to evaluate fill"

    # ---------- RSS axis ----------
    rss_axis = "FAIL"
    rss_diag = ""
    rss_threshold_bytes = 0.7 * CN_NODE_RAM_BYTES
    if best is not None and best["peak_rss_bytes"] is not None:
        if best["peak_rss_bytes"] < rss_threshold_bytes:
            rss_axis = "PASS"
            rss_diag = (
                f"best-combo peak_rss={best['peak_rss_bytes']/(1024**3):.3f} GiB"
                f" < 0.7×CN_NODE_RAM={rss_threshold_bytes/(1024**3):.1f} GiB"
            )
        else:
            rss_axis = "FAIL"
            rss_diag = (
                f"best-combo peak_rss={best['peak_rss_bytes']/(1024**3):.3f} GiB"
                f" >= 0.7×CN_NODE_RAM={rss_threshold_bytes/(1024**3):.1f} GiB"
            )
    else:
        rss_diag = "no PASS combo to evaluate RSS"

    # ---------- Wall axis ----------
    # estimate = (numeric_factor_wall / refactor_freq) + (N_solve * solve_wall)
    # solve_wall = WALL_SOLVE_FACTOR * numeric_factor_wall (estimated triangular solve cost)
    # threshold = 0.7 * SPGMR_per_step_wall (~ 0.1589 s for 0.7×0.227)
    wall_axis = "FAIL"
    wall_diag = ""
    if best is not None and best["numeric_factor_wall_sec"] is not None:
        nfw = best["numeric_factor_wall_sec"]
        solve_wall = WALL_SOLVE_FACTOR * nfw
        per_step_estimate = (nfw / WALL_REFACTOR_FREQ) + (WALL_N_SOLVE * solve_wall)
        if per_step_estimate < WALL_BUDGET_S:
            wall_axis = "PASS"
            wall_diag = (
                f"per-step estimate=({nfw:.4f}/{WALL_REFACTOR_FREQ:.0f}) + "
                f"({WALL_N_SOLVE:.0f}·{WALL_SOLVE_FACTOR:.2f}·{nfw:.4f}) = {per_step_estimate:.4f} s"
                f" < budget=0.7·{SPGMR_PER_STEP_WALL_S:.4f}={WALL_BUDGET_S:.4f} s"
            )
        else:
            wall_axis = "FAIL"
            wall_diag = (
                f"per-step estimate=({nfw:.4f}/{WALL_REFACTOR_FREQ:.0f}) + "
                f"({WALL_N_SOLVE:.0f}·{WALL_SOLVE_FACTOR:.2f}·{nfw:.4f}) = {per_step_estimate:.4f} s"
                f" >= budget=0.7·{SPGMR_PER_STEP_WALL_S:.4f}={WALL_BUDGET_S:.4f} s"
            )
    else:
        wall_diag = "no PASS combo to evaluate wall"

    # ---------- Overall verdict ----------
    # Per spec REQ-5 D8 + Scenario "Case-aware/Optional branch
    # auto-population":
    #   3/3 PASS                                      -> GO
    #   2/3 PASS + wall marginal (≤ 2× budget)        -> Optional
    #   2/3 PASS + wall blown (> 2× budget)           -> NO-GO (wall_overflow)
    #   <2/3 PASS or fill/rss FAIL                    -> NO-GO
    # "Marginal" 2× definition documented inline; rationale: Optional
    # signals "mini-spike before commit"; NO-GO signals "wall axis is
    # decisive — go to AMG retreat".
    pass_count = sum(1 for x in (fill_axis, rss_axis, wall_axis) if x == "PASS")

    # Compute wall_margin if wall axis is being evaluated.
    wall_margin = None
    if best is not None and best["numeric_factor_wall_sec"] is not None and WALL_BUDGET_S > 0:
        nfw = best["numeric_factor_wall_sec"]
        solve_wall = WALL_SOLVE_FACTOR * nfw
        per_step_est = (nfw / WALL_REFACTOR_FREQ) + (WALL_N_SOLVE * solve_wall)
        wall_margin = per_step_est / WALL_BUDGET_S

    if pass_count == 3:
        overall = "GO"
        no_go_axis = "clean_GO"
    elif pass_count == 2 and wall_axis == "FAIL" and wall_margin is not None and wall_margin <= 2.0:
        overall = "Optional"  # marginal wall axis (within 2× budget)
        no_go_axis = "wall_overflow"
    elif pass_count == 2 and wall_axis == "FAIL":
        overall = "NO-GO"     # wall axis blown well past budget
        no_go_axis = "wall_overflow"
    elif pass_count == 0:
        overall = "NO-GO"
        if fill_axis == "FAIL":
            no_go_axis = "fill_overflow"
        elif rss_axis == "FAIL":
            no_go_axis = "rss_overflow"
        else:
            no_go_axis = "wall_overflow"
    else:
        # 1 or 2 axes FAIL where fill or rss are among them — NO-GO.
        overall = "NO-GO"
        if fill_axis == "FAIL":
            no_go_axis = "fill_overflow"
        elif rss_axis == "FAIL":
            no_go_axis = "rss_overflow"
        else:
            no_go_axis = "wall_overflow"

    # Recommended action.
    if overall == "GO":
        rec_action = "klu-env-var-opt-in"
    elif overall == "Optional":
        rec_action = "klu-env-var-opt-in"
    elif overall == "NO-GO":
        rec_action = "use-future-amg"
    else:
        rec_action = "use-spgmr-default"

    # Diagnostic — verbatim concrete numbers per spec REQ-5 D8.
    diag_pieces = []
    if fill_axis == "FAIL":
        diag_pieces.append(fill_diag)
    if rss_axis == "FAIL":
        diag_pieces.append(rss_diag)
    if wall_axis == "FAIL":
        diag_pieces.append(wall_diag)
    if not diag_pieces:
        diag_pieces.append(f"all 3 axes PASS for best-combo {best['ordering']}+btf{best['btf']}")
    diag_text = " | ".join(diag_pieces)

    return {
        "best": best,
        "NumY": NumY,
        "fill_threshold": fill_threshold,
        "fill_axis": fill_axis,
        "rss_axis": rss_axis,
        "wall_axis": wall_axis,
        "overall": overall,
        "no_go_axis": no_go_axis,
        "diag": diag_text,
        "rec_action": rec_action,
        "fill_diag": fill_diag,
        "rss_diag": rss_diag,
        "wall_diag": wall_diag,
    }

verdicts = {}
for case in CASE_ARR:
    verdicts[case] = per_case_verdict(case, by_case[case])

# ---------------------------------------------------------------------------
# Case-aware overall branch (spec REQ-5 Scenario "Case-aware/Optional
# branch auto-population"): Case-aware iff keliya GO AND heihe GO AND
# heihe_x4 in {NO-GO, Optional}.
# ---------------------------------------------------------------------------
keliya_v = verdicts["keliya"]["overall"]
heihe_v = verdicts["heihe"]["overall"]
heihe_x4_v = verdicts["heihe_x4"]["overall"]
heihe_x16_v = verdicts["heihe_x16"]["overall"]

if (keliya_v == "GO" and heihe_v == "GO"
        and heihe_x4_v in ("NO-GO", "Optional")):
    case_aware_branch = True
else:
    case_aware_branch = False

# Per spec REQ-5 amendment: when Case-aware fires, small cases SHALL recommend
# `klu-env-var-opt-in`, large cases SHALL recommend `use-future-amg`.
if case_aware_branch:
    verdicts["keliya"]["rec_action"] = "klu-env-var-opt-in"
    verdicts["heihe"]["rec_action"] = "klu-env-var-opt-in"
    verdicts["heihe_x4"]["rec_action"] = "use-future-amg"
    verdicts["heihe_x16"]["rec_action"] = "use-future-amg"

# Next-epic pointer (decisive cell = heihe_x4).
if heihe_x4_v == "GO":
    next_epic = "p8-tune.E-klu-impl"
    next_epic_priority = "high"
elif heihe_x4_v == "Optional":
    next_epic = "p8-tune.E-klu-impl"
    next_epic_priority = "medium"
else:  # NO-GO
    next_epic = "p8-tune.F-amg-spike"
    next_epic_priority = "high"

# ---------------------------------------------------------------------------
# Emit aggregate_verdict.txt per spec REQ-5 Scenario "Per-case axis
# machine-readable" + design D8.
# ---------------------------------------------------------------------------
verdict_path = OUT_DIR / "aggregate_verdict.txt"
with open(verdict_path, "w") as f:
    f.write("# p8tune-klu-spike aggregate verdict (PR-B)\n")
    f.write("# Generated by tools/p8tune.D/aggregate_klu_spike.sh\n")
    f.write("# Spec: openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5 + REQ-6\n")
    f.write("#\n")
    f.write(f"# Wall-axis interpretation parameters (env-tunable):\n")
    f.write(f"#   WALL_REFACTOR_FREQ          = {WALL_REFACTOR_FREQ}\n")
    f.write(f"#   WALL_N_SOLVE                = {WALL_N_SOLVE}\n")
    f.write(f"#   WALL_SOLVE_FACTOR           = {WALL_SOLVE_FACTOR}\n")
    f.write(f"#   WALL_SPGMR_BUDGET_FRACTION  = {WALL_SPGMR_BUDGET_FRACTION}\n")
    f.write(f"#   WALL_BUDGET_S (derived)     = {WALL_BUDGET_S:.6f}\n")
    f.write("#\n")
    # Per-case KV blocks. Per-line key width = max(case_len)=9 ("heihe_x16")
    # + ("_KLU_NO_GO_diagnostic" = 21 char) = total key prefix 30 char,
    # leave 1 space pad before "=".
    KEY_WIDTH = 40
    for case in CASE_ARR:
        v = verdicts[case]
        def kw(k):
            return f"{case}_{k}".ljust(KEY_WIDTH)
        f.write(f"{kw('KLU_fill_axis')}= {v['fill_axis']}\n")
        f.write(f"{kw('KLU_rss_axis')}= {v['rss_axis']}\n")
        f.write(f"{kw('KLU_wall_axis')}= {v['wall_axis']}\n")
        f.write(f"{kw('KLU_overall_verdict')}= {v['overall']}\n")
        f.write(f"{kw('KLU_NO_GO_axis')}= {v['no_go_axis']}\n")
        # Diagnostic — escape internal " by using single quote wrap.
        diag = v["diag"].replace('"', "'")
        f.write(f'{kw("KLU_NO_GO_diagnostic")}= "{diag}"\n')
        # Per-axis sub-diagnostics (additive to D8; lets render walk
        # per-axis without splitting the merged NO_GO_diagnostic).
        for axis_label, sub_diag in (
            ("KLU_fill_axis_diagnostic", v["fill_diag"]),
            ("KLU_rss_axis_diagnostic", v["rss_diag"]),
            ("KLU_wall_axis_diagnostic", v["wall_diag"]),
        ):
            sub = sub_diag.replace('"', "'")
            f.write(f'{kw(axis_label)}= "{sub}"\n')
        f.write(f"{kw('recommended_action')}= {v['rec_action']}\n")
        f.write("\n")

    # Decisive-cell pointers.
    LK = lambda k: k.ljust(56)
    f.write(f"{LK('heihe_x4_recommended_next_epic')}= {next_epic}\n")
    f.write(f"{LK('heihe_x4_recommended_next_epic_priority')}= {next_epic_priority}\n")
    f.write("\n")

    # Case-aware branch flag.
    f.write(f"{LK('case_aware_branch_fires')}= {'true' if case_aware_branch else 'false'}\n")
    f.write("\n")

    # Embedded thresholds (per spec REQ-5 D8).
    f.write(f"{LK('CN_NODE_RAM_BYTES')}= {CN_NODE_RAM_BYTES}\n")
    f.write(f"{LK('SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s')}= {SPGMR_PER_STEP_WALL_S:.6f}\n")
print(f"OK: wrote {verdict_path}")

# ---------------------------------------------------------------------------
# Final stdout summary (human-friendly).
# ---------------------------------------------------------------------------
print()
print("=== Per-case verdicts ===")
for case in CASE_ARR:
    v = verdicts[case]
    print(f"  {case:<10s}: fill={v['fill_axis']} rss={v['rss_axis']} wall={v['wall_axis']} -> {v['overall']}")
print()
print(f"Case-aware branch fires: {case_aware_branch}")
print(f"Decisive-cell next epic: {next_epic} (priority={next_epic_priority})")
PYEOF
