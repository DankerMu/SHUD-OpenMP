#!/usr/bin/env bash
# tools/p8tune.F/aggregate_amg_spike.sh
#
# P8-tune.F BoomerAMG pattern-only spike (PR-C). Per openspec change
# `p8tune-amg-spike` spec REQ-5 (5-axis per-case evaluation + 4-branch
# verdict_branch auto-typing) + REQ-6 (ADR-0007 §Decision byte-identical
# anchor) + issue #397 任务清单 4.1-4.5.
#
# Scans PR-B 16-cell sweep results, parses each `cell-N.out` file for the
# CELL_SUMMARY_BEGIN/END KV block, extracts verdict_class strictly from
# that block (NOT from MARKER:* stdout lines per spec REQ-4 marker-vs-class
# binary), and emits:
#
#   1. ${OUT_DIR}/aggregate.tsv          — per-cell flat TSV (16 rows)
#   2. ${OUT_DIR}/aggregate_verdict.txt  — machine-readable verdict per spec
#
# Per-case 5-axis verdict (spec REQ-5):
#   Axis 1 (Setup):  setup_wall_sec  < WALL_BUDGET_SETUP_SEC (≈ 0.237908 s
#                                                = 1.5 × 0.7 × SPGMR_PER_STEP_SEC)
#   Axis 2 (Apply):  apply_wall_sec  < WALL_BUDGET_APPLY_SEC (≈ 0.158605 s
#                                                = 0.7 × SPGMR_PER_STEP_SEC)
#   Axis 3 (Memory): peak_rss_bytes  < WALL_BUDGET_RSS_BYTES (≈ 130 GiB
#                                                = 0.7 × CN_NODE_RAM_BYTES)
#   Axis 4 (Cycle):  cycle_complexity < 1.5
#   Axis 5 (Operator): operator_complexity < 2.0
#
# 4-branch verdict_branch auto-typing (top-down, first match wins):
#   GO                       — all 4 cases all 5 axes PASS
#   Optional                 — keliya+heihe+heihe_x4 PASS; heihe_x16 fails
#                              ONLY wall axes (1/2), max margin < 1.5×
#   NO-GO-heihe_x16-only     — keliya+heihe+heihe_x4 PASS; heihe_x16 fails
#                              on Axis 3 / 4 / 5
#   NO-GO-both               — heihe_x4 fails ANY axis OR heihe_x16 fails
#                              wall axes with margin ≥ 1.5× OR heihe_x16
#                              fails ≥ 3 of 5 axes
#   BLOCKED                  — small-case (keliya OR heihe) unexpected
#                              fail (tool instability suspected) OR
#                              malformed cell_summary OR enum out of range
#                              OR PR-0 #386 dtor UB recurrence
#   BLOCKED (fallback)       — no rule matched (deterministic, no silent unset)
#
# Per-case best combo: min(setup_wall_sec + apply_wall_sec); tiebreaker =
# min operator_complexity (per spec REQ-5).
#
# Inputs (env vars / CLI):
#   --run-dir <path>     directory containing cell-N.out files (default
#                        .review-evidence/p8tune-amg-pr-b/cells)
#   --output-dir <path>  directory for aggregate.tsv + aggregate_verdict.txt
#                        (default .review-evidence/p8tune-amg-pr-c)
#   SPGMR_BASELINE_HDR   tools/p8tune.D/spgmr_baseline_walls.h (REUSED per
#                        spec REQ-5 Scenario "Axis threshold constants
#                        pinned in shared header"; missing = fatal)
#   CN_NODE_RAM_BYTES_HDR  tools/p8tune.D/cn_node_ram.h (REUSED, ditto)
#
# Outputs:
#   ${OUT_DIR}/aggregate.tsv           — 16-row TSV (header + 16 data rows)
#   ${OUT_DIR}/aggregate_verdict.txt   — AGGREGATE_VERDICT_BEGIN/END +
#                                        verdict_branch top-line + 4
#                                        CASE_VERDICT_BEGIN/END blocks
#
# Usage:
#   tools/p8tune.F/aggregate_amg_spike.sh
#   tools/p8tune.F/aggregate_amg_spike.sh \
#       --run-dir .review-evidence/p8tune-amg-pr-b/cells \
#       --output-dir .review-evidence/p8tune-amg-pr-c
#
# Exit codes:
#   0  OK
#   1  missing inputs / parse failure / malformed cell_summary
#   2  uv missing (CLAUDE.md mandates `uv run python` for all Python)
#
# Refs:
#   openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md
#       REQ-4 marker-vs-class binary; REQ-5 5-axis + 4-branch; REQ-6 ADR-0007
#   tools/p8tune.D/aggregate_klu_spike.sh (PR-B #387 reference template)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_DIR="${REPO_ROOT}/.review-evidence/p8tune-amg-pr-b/cells"
OUT_DIR="${REPO_ROOT}/.review-evidence/p8tune-amg-pr-c"

# ---- CLI parse ----
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail$/p' "$0" | head -n -2
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 [--run-dir <path>] [--output-dir <path>]" >&2
            exit 1
            ;;
    esac
done

SPGMR_BASELINE_HDR="${SPGMR_BASELINE_HDR:-${REPO_ROOT}/tools/p8tune.D/spgmr_baseline_walls.h}"
CN_NODE_RAM_BYTES_HDR="${CN_NODE_RAM_BYTES_HDR:-${REPO_ROOT}/tools/p8tune.D/cn_node_ram.h}"

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: missing --run-dir directory: ${RUN_DIR}" >&2
    exit 1
fi
if [[ ! -f "${SPGMR_BASELINE_HDR}" ]]; then
    # Per spec REQ-5 Scenario "Axis threshold constants pinned in shared
    # header": missing baseline header is fatal, NOT silent fallback.
    echo "ERROR: missing spgmr_baseline_walls.h: ${SPGMR_BASELINE_HDR}" >&2
    echo "ERROR: aggregator REQUIRES P8-tune.D pinned constants per spec REQ-5;" >&2
    echo "ERROR: no silent default permitted." >&2
    exit 1
fi
if [[ ! -f "${CN_NODE_RAM_BYTES_HDR}" ]]; then
    echo "ERROR: missing cn_node_ram.h: ${CN_NODE_RAM_BYTES_HDR}" >&2
    exit 1
fi
# CLAUDE.md mandates Python via `uv run` only — no bare python/python3/pip.
if ! command -v uv >/dev/null 2>&1; then
    echo 'ERROR: uv missing (CLAUDE.md mandates "uv run python" for all Python)' >&2
    exit 2
fi

mkdir -p "${OUT_DIR}"

# Parse C++ headers for pinned constants (single-source-of-truth).
# spgmr_baseline_walls.h:
#   static constexpr double SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579;
# cn_node_ram.h:
#   static constexpr std::size_t CN_NODE_RAM_BYTES = 185528156160ULL;
SPGMR_PER_STEP_SEC="$(grep -oE 'SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S\s*=\s*[0-9.]+' "${SPGMR_BASELINE_HDR}" | grep -oE '[0-9.]+$')"
CN_NODE_RAM_BYTES="$(grep -oE 'CN_NODE_RAM_BYTES\s*=\s*[0-9]+' "${CN_NODE_RAM_BYTES_HDR}" | grep -oE '[0-9]+')"

if [[ -z "${SPGMR_PER_STEP_SEC}" || -z "${CN_NODE_RAM_BYTES}" ]]; then
    echo "ERROR: failed to parse constants from C++ headers" >&2
    echo "  SPGMR_PER_STEP_SEC = '${SPGMR_PER_STEP_SEC}'" >&2
    echo "  CN_NODE_RAM_BYTES  = '${CN_NODE_RAM_BYTES}'" >&2
    exit 1
fi

echo "INFO: RUN_DIR              = ${RUN_DIR}"
echo "INFO: OUT_DIR              = ${OUT_DIR}"
echo "INFO: SPGMR_PER_STEP_SEC   = ${SPGMR_PER_STEP_SEC}  (P8-tune.D pinned baseline)"
echo "INFO: CN_NODE_RAM_BYTES    = ${CN_NODE_RAM_BYTES}  (P8-tune.D pinned baseline)"

export RUN_DIR OUT_DIR SPGMR_PER_STEP_SEC CN_NODE_RAM_BYTES SPGMR_BASELINE_HDR CN_NODE_RAM_BYTES_HDR

# All axis arithmetic + JSON intermediate logic lives in python (cleaner
# than bash float ops). Invoked via `uv run python` per CLAUDE.md mandate.
uv run python - <<'PYEOF'
import os
import re
import sys
import math
from collections import OrderedDict
from pathlib import Path

# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------
RUN_DIR = Path(os.environ["RUN_DIR"])
OUT_DIR = Path(os.environ["OUT_DIR"])
SPGMR_PER_STEP_SEC = float(os.environ["SPGMR_PER_STEP_SEC"])
CN_NODE_RAM_BYTES = int(os.environ["CN_NODE_RAM_BYTES"])
SPGMR_BASELINE_HDR = os.environ["SPGMR_BASELINE_HDR"]
CN_NODE_RAM_BYTES_HDR = os.environ["CN_NODE_RAM_BYTES_HDR"]

# Derived thresholds per spec REQ-5 Scenario "Axis threshold constants
# pinned in shared header".
WALL_BUDGET_APPLY_SEC = 0.7 * SPGMR_PER_STEP_SEC      # ≈ 0.158605
WALL_BUDGET_SETUP_SEC = 1.5 * WALL_BUDGET_APPLY_SEC   # ≈ 0.237908
WALL_BUDGET_RSS_BYTES = 0.7 * CN_NODE_RAM_BYTES       # ≈ 130 GiB

# Axis 4/5 unitless thresholds per spec REQ-5 Scenario "5-axis threshold
# evaluation per case".
CYCLE_COMPLEXITY_THRESHOLD = 1.5
OPERATOR_COMPLEXITY_THRESHOLD = 2.0

# Canonical verdict_class enum (per spec REQ-4 + REQ-5 Scenario
# "cell_summary KV block schema"). Aggregator rejects any value outside.
VALID_VERDICT_CLASSES = {
    "PASS",
    "AMG_OOM",
    "AMG_SETUP_DIVERGE",
    "AMG_SOLVE_DIVERGE",
    "AMG_WALL_OVERFLOW",
}

# 4-case canonical order per spec REQ-4 cell decode table.
CASE_ARR = ["keliya", "heihe", "heihe_x4", "heihe_x16"]

# ---------------------------------------------------------------------------
# Per-cell parser: extract CELL_SUMMARY_BEGIN/END KV block ONLY.
# Per spec REQ-4 marker-vs-class binary: aggregator MUST NOT read
# MARKER:AMG_*_DETECTED from stdout lines outside the KV block. The marker
# is a STDOUT verb (emitted live during run); verdict_class is a KV noun
# (canonical summary). They must be byte-distinct per spec.
# ---------------------------------------------------------------------------
def parse_cell_log(path):
    """Return a dict with parsed cell KVs. Raises ValueError if malformed."""
    result = {
        "log_path": str(path),
        "case": None,
        "interp_type": None,
        "coarsen_type": None,
        "NumY": None,
        "nnz_A": None,
        "setup_wall_sec": None,
        "apply_wall_sec": None,
        "peak_rss_bytes": None,
        "cycle_complexity": None,
        "operator_complexity": None,
        "residual_reduction_v1": None,
        "verdict_class": None,
        "hypre_version": None,
        "colpack_version": None,
        "shud_pin": None,
    }
    with open(path, "r", errors="replace") as f:
        text = f.read()

    # Extract ONLY the CELL_SUMMARY_BEGIN/END block. Per spec REQ-4
    # marker-vs-class binary, this is the SOLE source of verdict_class.
    block_match = re.search(
        r"CELL_SUMMARY_BEGIN\s*\n(.*?)\nCELL_SUMMARY_END",
        text,
        re.DOTALL,
    )
    if not block_match:
        raise ValueError(
            f"missing CELL_SUMMARY_BEGIN/END block in {path}"
        )
    block = block_match.group(1)

    # Line 1: case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>
    m = re.search(
        r"case=(\S+)\s+interp_type=(\d+)\s+coarsen_type=(\d+)\s+NumY=(\d+)\s+nnz_A=(\d+)",
        block,
    )
    if m:
        result["case"] = m.group(1)
        result["interp_type"] = int(m.group(2))
        result["coarsen_type"] = int(m.group(3))
        result["NumY"] = int(m.group(4))
        result["nnz_A"] = int(m.group(5))
    # Line 2: setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>
    # NA sentinel accepted per PR-B M3 (AMG_WALL_OVERFLOW cells lack
    # numeric timing — value 'NA' allowed; aggregator treats as None).
    m = re.search(
        r"setup_wall_sec=([\d.eE+-]+|NA)\s+apply_wall_sec=([\d.eE+-]+|NA)\s+peak_rss_bytes=(\d+|NA)",
        block,
    )
    if m:
        result["setup_wall_sec"] = None if m.group(1) == "NA" else float(m.group(1))
        result["apply_wall_sec"] = None if m.group(2) == "NA" else float(m.group(2))
        result["peak_rss_bytes"] = None if m.group(3) == "NA" else int(m.group(3))
    # Line 3: cycle_complexity=<CC> operator_complexity=<OC>
    # residual_reduction_v1=<R1>
    m = re.search(
        r"cycle_complexity=([\d.eE+-]+|NA)\s+operator_complexity=([\d.eE+-]+|NA)\s+residual_reduction_v1=([\d.eE+-]+|NA)",
        block,
    )
    if m:
        result["cycle_complexity"] = None if m.group(1) == "NA" else float(m.group(1))
        result["operator_complexity"] = None if m.group(2) == "NA" else float(m.group(2))
        result["residual_reduction_v1"] = None if m.group(3) == "NA" else float(m.group(3))
    # Line 4: verdict_class=<enum>
    m = re.search(r"verdict_class=(\S+)", block)
    if m:
        vc = m.group(1)
        # Per spec REQ-4: verdict_class has NO `_DETECTED` suffix
        # (that suffix is for stdout marker only). Aggregator rejects
        # any value with `_DETECTED` appearing in the KV.
        if vc.endswith("_DETECTED"):
            raise ValueError(
                f"malformed verdict_class='{vc}' in {path}: "
                f"`_DETECTED` suffix is for stdout MARKER lines only, "
                f"not for the KV noun (spec REQ-4 marker-vs-class binary)"
            )
        if vc not in VALID_VERDICT_CLASSES:
            raise ValueError(
                f"verdict_class='{vc}' in {path} not in canonical enum "
                f"{sorted(VALID_VERDICT_CLASSES)}"
            )
        result["verdict_class"] = vc
    # Line 5: hypre_version=<HV> colpack_version=<CV> shud_pin=<SHA>
    # colpack_version=unknown sentinel accepted per PR-B M2 closure.
    m = re.search(
        r"hypre_version=(\S+)\s+colpack_version=(\S+)\s+shud_pin=(\S+)",
        block,
    )
    if m:
        result["hypre_version"] = m.group(1)
        result["colpack_version"] = m.group(2)
        result["shud_pin"] = m.group(3)

    # Final required-field validation.
    required = [
        "case", "interp_type", "coarsen_type", "NumY", "nnz_A",
        "verdict_class",
    ]
    for k in required:
        if result[k] is None:
            raise ValueError(
                f"missing required KV '{k}' in CELL_SUMMARY block of {path}"
            )
    return result


# ---------------------------------------------------------------------------
# Walk all 16 cells. PR-B layout uses single-digit `cell-N.out` filenames
# (NOT zero-padded NN). Discovery is by directory scan + NN extraction.
# ---------------------------------------------------------------------------
cells = []
for nn in range(16):
    candidate = RUN_DIR / f"cell-{nn}.out"
    if not candidate.exists():
        print(
            f"ERROR: missing cell-{nn}.out in {RUN_DIR} — aggregator "
            f"requires all 16 cells per spec REQ-3 + PR-B 16/16 PASS",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        parsed = parse_cell_log(candidate)
    except ValueError as e:
        print(f"ERROR: parse failure at NN={nn}: {e}", file=sys.stderr)
        sys.exit(1)
    parsed["nn"] = nn
    cells.append(parsed)

# ---------------------------------------------------------------------------
# Emit aggregate.tsv (16 raw rows + header).
# ---------------------------------------------------------------------------
tsv_path = OUT_DIR / "aggregate.tsv"
tsv_cols = [
    "nn", "case", "interp_type", "coarsen_type", "NumY", "nnz_A",
    "setup_wall_sec", "apply_wall_sec", "peak_rss_bytes",
    "cycle_complexity", "operator_complexity", "residual_reduction_v1",
    "verdict_class", "hypre_version", "colpack_version", "shud_pin",
]
with open(tsv_path, "w") as f:
    f.write("\t".join(tsv_cols) + "\n")
    for c in cells:
        def fmt(v):
            if v is None:
                return "NA"
            if isinstance(v, float):
                # Preserve full precision from cell log for downstream
                # arithmetic determinism.
                return f"{v:.6f}".rstrip("0").rstrip(".") if "." in f"{v}" else f"{v}"
            return str(v)
        row = [
            f"{c['nn']:02d}",
            c["case"],
            str(c["interp_type"]),
            str(c["coarsen_type"]),
            str(c["NumY"]),
            str(c["nnz_A"]),
            fmt(c["setup_wall_sec"]),
            fmt(c["apply_wall_sec"]),
            fmt(c["peak_rss_bytes"]),
            fmt(c["cycle_complexity"]),
            fmt(c["operator_complexity"]),
            fmt(c["residual_reduction_v1"]),
            c["verdict_class"],
            c["hypre_version"] or "NA",
            c["colpack_version"] or "NA",
            c["shud_pin"] or "NA",
        ]
        f.write("\t".join(row) + "\n")
print(f"OK: wrote {tsv_path}")

# ---------------------------------------------------------------------------
# Per-case best combo selection + 5-axis evaluation.
# Per spec REQ-5: best combo = min(setup_wall_sec + apply_wall_sec);
# tiebreaker = min operator_complexity.
# ---------------------------------------------------------------------------
def best_combo_for_case(case_name):
    """Return cell dict for the best combo of `case_name`, or None."""
    pass_cells = [
        c for c in cells
        if c["case"] == case_name
        and c["verdict_class"] == "PASS"
        and c["setup_wall_sec"] is not None
        and c["apply_wall_sec"] is not None
    ]
    if not pass_cells:
        return None
    return sorted(
        pass_cells,
        key=lambda c: (
            (c["setup_wall_sec"] + c["apply_wall_sec"]),
            (c["operator_complexity"] if c["operator_complexity"] is not None else float("inf")),
        ),
    )[0]


def evaluate_5_axes(cell):
    """Return per-axis dict {axis_N: {pass, value, threshold, margin}}.

    Per spec REQ-5 Scenario "5-axis threshold evaluation per case" +
    "failing_margin computation per axis".

    margin = V / T (ratio form; margin > 1.0 indicates FAIL).
    """
    axes = OrderedDict()
    # Axis 1: Setup
    v1 = cell["setup_wall_sec"]
    t1 = WALL_BUDGET_SETUP_SEC
    axes["axis1_setup"] = {
        "name": "Setup wall",
        "value": v1, "threshold": t1, "unit": "sec",
        "pass": (v1 is not None and v1 < t1),
        "margin": (v1 / t1) if (v1 is not None and t1 > 0) else None,
    }
    # Axis 2: Apply
    v2 = cell["apply_wall_sec"]
    t2 = WALL_BUDGET_APPLY_SEC
    axes["axis2_apply"] = {
        "name": "Apply wall",
        "value": v2, "threshold": t2, "unit": "sec",
        "pass": (v2 is not None and v2 < t2),
        "margin": (v2 / t2) if (v2 is not None and t2 > 0) else None,
    }
    # Axis 3: Memory
    v3 = cell["peak_rss_bytes"]
    t3 = WALL_BUDGET_RSS_BYTES
    axes["axis3_memory"] = {
        "name": "Peak RSS",
        "value": v3, "threshold": t3, "unit": "bytes",
        "pass": (v3 is not None and v3 < t3),
        "margin": (v3 / t3) if (v3 is not None and t3 > 0) else None,
    }
    # Axis 4: Cycle complexity
    v4 = cell["cycle_complexity"]
    t4 = CYCLE_COMPLEXITY_THRESHOLD
    axes["axis4_cycle"] = {
        "name": "Cycle complexity",
        "value": v4, "threshold": t4, "unit": "ratio",
        "pass": (v4 is not None and v4 < t4),
        "margin": (v4 / t4) if (v4 is not None and t4 > 0) else None,
    }
    # Axis 5: Operator complexity
    v5 = cell["operator_complexity"]
    t5 = OPERATOR_COMPLEXITY_THRESHOLD
    axes["axis5_operator"] = {
        "name": "Operator complexity",
        "value": v5, "threshold": t5, "unit": "ratio",
        "pass": (v5 is not None and v5 < t5),
        "margin": (v5 / t5) if (v5 is not None and t5 > 0) else None,
    }
    return axes


# Compute per-case best combos + 5-axis verdicts.
case_results = OrderedDict()
for case in CASE_ARR:
    best = best_combo_for_case(case)
    if best is None:
        case_results[case] = {
            "best": None,
            "axes": None,
            "all_pass": False,
            "failing_axes": ["axis_unknown"],
            "max_failing_margin": float("inf"),
            "wall_axes_only_failing": False,
        }
        continue
    axes = evaluate_5_axes(best)
    failing = [k for k, ax in axes.items() if not ax["pass"]]
    margins = [ax["margin"] for ax in axes.values() if not ax["pass"] and ax["margin"] is not None]
    max_margin = max(margins) if margins else 0.0
    wall_only = bool(failing) and all(k in ("axis1_setup", "axis2_apply") for k in failing)
    case_results[case] = {
        "best": best,
        "axes": axes,
        "all_pass": (not failing),
        "failing_axes": failing,
        "max_failing_margin": max_margin,
        "wall_axes_only_failing": wall_only,
    }

# ---------------------------------------------------------------------------
# 4-branch verdict_branch auto-typing per spec REQ-5 Scenario "4-branch
# decision auto-typing (covers all axis + case combinations)".
# Rules evaluated top-down; first match wins.
# ---------------------------------------------------------------------------
def compute_verdict_branch():
    """Return (verdict_branch, reason_str) per spec REQ-5 rules."""
    # Short aliases.
    kel = case_results["keliya"]
    heihe = case_results["heihe"]
    h4 = case_results["heihe_x4"]
    h16 = case_results["heihe_x16"]

    # Pre-flight BLOCKED gate: any best=None means parse failure /
    # malformed cell_summary path (the parser raised ValueError above,
    # but this catches the "all combos for a case failed verdict_class
    # != PASS" path — would mean the spike fundamentally crashed for
    # that case).
    for case_name, r in case_results.items():
        if r["best"] is None:
            return (
                "BLOCKED",
                f"case {case_name} has no PASS cell — spike fundamentally "
                f"failed (no best combo selectable)",
            )

    # Small-case BLOCKED gate (per spec REQ-5 Scenario "4-branch decision
    # auto-typing" trailing clause): keliya OR heihe FAIL any axis but
    # heihe_x4 + heihe_x16 PASS — would indicate tool instability since
    # small cases should be trivially fast for AMG.
    if (not kel["all_pass"] or not heihe["all_pass"]) and h4["all_pass"] and h16["all_pass"]:
        return (
            "BLOCKED",
            "small-case (keliya OR heihe) unexpected fail while heihe_x4 + "
            "heihe_x16 PASS — tool instability suspected; no production "
            "case has this verdict normally",
        )

    # Rule 1: GO — all 4 cases all 5 axes PASS.
    if all(r["all_pass"] for r in case_results.values()):
        return ("GO", "all 4 cases all 5 axes PASS for best combo")

    # Rule 2: Optional — keliya+heihe+heihe_x4 PASS AND heihe_x16 fails
    # ONLY on wall axes (1/2) with max margin < 1.5×.
    if (
        kel["all_pass"] and heihe["all_pass"] and h4["all_pass"]
        and not h16["all_pass"]
        and h16["wall_axes_only_failing"]
        and h16["max_failing_margin"] < 1.5
    ):
        return (
            "Optional",
            f"keliya/heihe/heihe_x4 PASS; heihe_x16 fails only on wall "
            f"axes with max margin = {h16['max_failing_margin']:.3f}× "
            f"(<1.5× marginal)",
        )

    # Rule 3: NO-GO-heihe_x16-only — keliya+heihe+heihe_x4 PASS AND
    # heihe_x16 fails on Axis 3/4/5.
    if (
        kel["all_pass"] and heihe["all_pass"] and h4["all_pass"]
        and not h16["all_pass"]
    ):
        # Any failure not exclusively on wall axes → axis 3/4/5 grouping.
        if not h16["wall_axes_only_failing"]:
            return (
                "NO-GO-heihe_x16-only",
                f"keliya/heihe/heihe_x4 PASS; heihe_x16 fails on "
                f"{sorted(h16['failing_axes'])} (includes hierarchy / "
                f"memory axes)",
            )
        # Else: wall-only with margin ≥ 1.5× falls through to NO-GO-both.

    # Rule 4: NO-GO-both — heihe_x4 fails ANY axis OR heihe_x16 fails
    # wall axes with margin ≥ 1.5× OR heihe_x16 fails ≥ 3 of 5 axes.
    h16_failing_count = len(h16["failing_axes"]) if h16["failing_axes"] else 0
    if (
        not h4["all_pass"]
        or (
            not h16["all_pass"]
            and h16["wall_axes_only_failing"]
            and h16["max_failing_margin"] >= 1.5
        )
        or (h16_failing_count >= 3)
    ):
        reasons = []
        if not h4["all_pass"]:
            reasons.append(
                f"heihe_x4 fails {sorted(h4['failing_axes'])} "
                f"(max margin {h4['max_failing_margin']:.3f}×)"
            )
        if (
            not h16["all_pass"]
            and h16["wall_axes_only_failing"]
            and h16["max_failing_margin"] >= 1.5
        ):
            reasons.append(
                f"heihe_x16 wall axes margin "
                f"{h16['max_failing_margin']:.3f}× ≥ 1.5×"
            )
        if h16_failing_count >= 3:
            reasons.append(
                f"heihe_x16 fails {h16_failing_count} of 5 axes (structural)"
            )
        return ("NO-GO-both", "; ".join(reasons))

    # Fallback BLOCKED — no rule above matched.
    return (
        "BLOCKED",
        "verdict_branch logic gap — case combination not covered by "
        "rules 1-4; manual review required",
    )


verdict_branch, verdict_reason = compute_verdict_branch()

# ---------------------------------------------------------------------------
# Compute hypothetical "Axis-4-amended" verdict for Discussion / ADR.
# Per PR-A H3 disclosure: cycle_complexity = 2 × operator_complexity is a
# hard-coded estimate (NOT measurement) in current spike implementation.
# Axis 4 mechanically tracks Axis 5; recommend treating Axis 4 as a
# non-discriminating diagnostic. The aggregator makes the STRICT call but
# documents the amended verdict for transparent comparison.
# ---------------------------------------------------------------------------
def compute_verdict_branch_axis4_amended():
    """As compute_verdict_branch but Axis 4 is treated as always-PASS."""
    saved_axes = {}
    for case, r in case_results.items():
        if r["axes"] is None:
            continue
        saved_axes[case] = (r["axes"]["axis4_cycle"]["pass"], r["axes"]["axis4_cycle"]["margin"])
        r["axes"]["axis4_cycle"]["pass"] = True
        # Re-compute failing_axes + max_failing_margin.
        r["failing_axes"] = [k for k, ax in r["axes"].items() if not ax["pass"]]
        margins = [ax["margin"] for ax in r["axes"].values() if not ax["pass"] and ax["margin"] is not None]
        r["max_failing_margin"] = max(margins) if margins else 0.0
        r["all_pass"] = (not r["failing_axes"])
        r["wall_axes_only_failing"] = bool(r["failing_axes"]) and all(
            k in ("axis1_setup", "axis2_apply") for k in r["failing_axes"]
        )
    amended_branch, amended_reason = compute_verdict_branch()
    # Restore.
    for case, (was_pass, was_margin) in saved_axes.items():
        case_results[case]["axes"]["axis4_cycle"]["pass"] = was_pass
        failing = [k for k, ax in case_results[case]["axes"].items() if not ax["pass"]]
        margins = [ax["margin"] for ax in case_results[case]["axes"].values() if not ax["pass"] and ax["margin"] is not None]
        case_results[case]["failing_axes"] = failing
        case_results[case]["max_failing_margin"] = max(margins) if margins else 0.0
        case_results[case]["all_pass"] = (not failing)
        case_results[case]["wall_axes_only_failing"] = bool(failing) and all(
            k in ("axis1_setup", "axis2_apply") for k in failing
        )
    return (amended_branch, amended_reason)


verdict_branch_axis4_amended, verdict_reason_axis4_amended = compute_verdict_branch_axis4_amended()

# ---------------------------------------------------------------------------
# Emit aggregate_verdict.txt per spec REQ-5 Scenario "aggregate_verdict.txt
# schema".
# ---------------------------------------------------------------------------
verdict_path = OUT_DIR / "aggregate_verdict.txt"
with open(verdict_path, "w") as f:
    f.write("# AGGREGATE_VERDICT_BEGIN\n")
    f.write("# p8tune-amg-spike aggregate verdict (PR-C)\n")
    f.write("# Generated by tools/p8tune.F/aggregate_amg_spike.sh\n")
    f.write("# Spec: openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md\n")
    f.write("#       REQ-4 marker-vs-class binary; REQ-5 5-axis + 4-branch; REQ-6 ADR-0007\n")
    f.write("#\n")
    f.write(f"# Pinned baseline constants (P8-tune.D, REUSED per spec REQ-5):\n")
    f.write(f"#   SPGMR_PER_STEP_SEC         = {SPGMR_PER_STEP_SEC:.6f}\n")
    f.write(f"#   CN_NODE_RAM_BYTES          = {CN_NODE_RAM_BYTES}\n")
    f.write(f"# Derived thresholds:\n")
    f.write(f"#   WALL_BUDGET_SETUP_SEC      = {WALL_BUDGET_SETUP_SEC:.6f}\n")
    f.write(f"#   WALL_BUDGET_APPLY_SEC      = {WALL_BUDGET_APPLY_SEC:.6f}\n")
    f.write(f"#   WALL_BUDGET_RSS_BYTES      = {WALL_BUDGET_RSS_BYTES:.0f}\n")
    f.write(f"#   CYCLE_COMPLEXITY_THRESHOLD = {CYCLE_COMPLEXITY_THRESHOLD}\n")
    f.write(f"#   OPERATOR_COMPLEXITY_THRESHOLD = {OPERATOR_COMPLEXITY_THRESHOLD}\n")
    f.write("#\n")

    # Top-line verdict_branch KV (canonical, byte-identical anchor for
    # ADR-0007 §Decision per spec REQ-6).
    f.write(f"verdict_branch={verdict_branch}\n")
    f.write(f"verdict_branch_reason=\"{verdict_reason}\"\n")
    f.write("\n")

    # Axis 4 amendment disclosure (per PR-A H3 + issue #397 deliverable).
    f.write(
        "# Axis 4 amendment disclosure (per PR-A H3 + issue #397):\n"
        "# cycle_complexity = 2 × operator_complexity is a hard-coded estimate\n"
        "# in current boomeramg_setup_solve.cpp implementation, NOT measurement\n"
        "# from HYPRE telemetry. Axis 4 mechanically tracks Axis 5 in all 16\n"
        "# observed cells (cycle ≈ 2.0 uniformly). Recommend treating Axis 4\n"
        "# as a non-discriminating diagnostic in ADR-0007 §Discussion. The\n"
        "# strict verdict above stands as the auto-typed canonical anchor;\n"
        "# the amended branch below is FYI for ADR §Discussion.\n"
    )
    f.write(f"verdict_branch_axis4_amended={verdict_branch_axis4_amended}\n")
    f.write(f"verdict_branch_axis4_amended_reason=\"{verdict_reason_axis4_amended}\"\n")
    f.write("\n")

    # Per-case CASE_VERDICT_BEGIN/END blocks.
    for case in CASE_ARR:
        r = case_results[case]
        f.write(f"# CASE_VERDICT_BEGIN:{case}\n")
        if r["best"] is None:
            f.write(f"{case}_best_combo=NONE\n")
            f.write(f"{case}_overall=FAIL\n")
            f.write(f"{case}_reason=\"no PASS cell — spike crashed for this case\"\n")
            f.write(f"# CASE_VERDICT_END:{case}\n\n")
            continue
        best = r["best"]
        axes = r["axes"]
        f.write(f"{case}_best_combo_nn={best['nn']:02d}\n")
        f.write(f"{case}_best_combo_interp_type={best['interp_type']}\n")
        f.write(f"{case}_best_combo_coarsen_type={best['coarsen_type']}\n")
        f.write(f"{case}_best_combo_NumY={best['NumY']}\n")
        f.write(f"{case}_best_combo_nnz_A={best['nnz_A']}\n")
        # 5 axes
        for axis_key in (
            "axis1_setup", "axis2_apply", "axis3_memory",
            "axis4_cycle", "axis5_operator",
        ):
            ax = axes[axis_key]
            status = "PASS" if ax["pass"] else "FAIL"
            f.write(f"{case}_{axis_key}_status={status}\n")
            f.write(f"{case}_{axis_key}_value={ax['value']}\n")
            f.write(f"{case}_{axis_key}_threshold={ax['threshold']}\n")
            margin_str = "NA" if ax["margin"] is None else f"{ax['margin']:.6f}"
            f.write(f"{case}_{axis_key}_margin={margin_str}\n")
        f.write(f"{case}_all_pass={'true' if r['all_pass'] else 'false'}\n")
        f.write(f"{case}_failing_axes={','.join(r['failing_axes']) if r['failing_axes'] else 'none'}\n")
        f.write(f"{case}_max_failing_margin={r['max_failing_margin']:.6f}\n")
        f.write(f"{case}_overall={'PASS' if r['all_pass'] else 'FAIL'}\n")
        f.write(f"# CASE_VERDICT_END:{case}\n\n")

    # Footer.
    f.write(f"# Provenance:\n")
    f.write(f"#   spgmr_baseline_walls.h = {SPGMR_BASELINE_HDR}\n")
    f.write(f"#   cn_node_ram.h          = {CN_NODE_RAM_BYTES_HDR}\n")
    f.write(f"#   run_dir                = {RUN_DIR}\n")
    f.write(f"#   cells_parsed           = 16/16\n")
    # Pull hypre + colpack + shud_pin from cell 0 (PR-B 16-cell sweep is
    # single-run uniform — all cells share identical env per spec REQ-3).
    c0 = cells[0]
    f.write(f"#   hypre_version          = {c0['hypre_version']}\n")
    f.write(f"#   colpack_version        = {c0['colpack_version']}\n")
    f.write(f"#   shud_pin               = {c0['shud_pin']}\n")
    f.write("# AGGREGATE_VERDICT_END\n")

print(f"OK: wrote {verdict_path}")

# ---------------------------------------------------------------------------
# Final stdout summary (human-friendly).
# ---------------------------------------------------------------------------
print()
print("=== Per-case 5-axis results (best combo) ===")
for case in CASE_ARR:
    r = case_results[case]
    if r["best"] is None:
        print(f"  {case:<12s}: NONE (spike crashed)")
        continue
    best = r["best"]
    axes_str = " ".join(
        f"{k[:6]}={('P' if ax['pass'] else 'F')}"
        for k, ax in r["axes"].items()
    )
    print(
        f"  {case:<12s}: NN={best['nn']:02d} interp={best['interp_type']:>2d} "
        f"coarsen={best['coarsen_type']:>2d}  {axes_str}  "
        f"overall={'PASS' if r['all_pass'] else 'FAIL'}"
    )
print()
print(f"verdict_branch                = {verdict_branch}")
print(f"verdict_branch (Axis4-amended) = {verdict_branch_axis4_amended}  (FYI)")
print(f"reason                        : {verdict_reason}")
PYEOF
