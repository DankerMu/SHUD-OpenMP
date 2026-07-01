#!/bin/bash
# tools/p9.spot/aggregate_p9_spot.sh
#
# PR-Z1 — P9 spot-check aggregator. Consumes:
#   - ${RUN_DIR}/cell-0-baseline.out          (SPGMR baseline reltol)
#   - ${RUN_DIR}/cell-1-p9.out                (SPGMR reltol=1e-3)
#   - ${RUN_DIR}/a5-report/a5_metrics.json    (A5 gate verdict)
# and emits the PR_Z1_VERDICT MARKER block deciding whether to open a
# full P9 sweep or close P9 as unpromising.
#
# Decision logic (per PR-Z1 brief §Stage 2):
#   - wall_speedup >= 1.5 AND a5_verdict=PASS  -> open_full_sweep
#   - 1.2 <= wall_speedup < 1.5 AND a5_verdict=PASS -> optional_p9
#   - otherwise -> close_p9
#
# Wall speedup = baseline_wall / p9_wall (larger is better).
# nst_ratio    = p9_nst / baseline_nst   (smaller is better; expected <1
#                if reltol relaxation is taking bigger steps).
#
# Usage:
#   ./aggregate_p9_spot.sh <run_dir>
#
# Args:
#   run_dir  directory containing cell-0-baseline.{out,err} +
#            cell-1-p9.{out,err} + a5-report/a5_metrics.json
#
# Exit code:
#   0  aggregator succeeded (verdict emitted; decision may be any of
#      open_full_sweep / optional_p9 / close_p9).
#   2  unrecoverable parse error (missing sidecars, missing A5 JSON,
#      cannot compute wall speedup because a required cell wall is NA).

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run_dir>" >&2
    exit 2
fi

RUN_DIR="$1"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[aggregate-P9-SPOT] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 2
fi

echo "[aggregate-P9-SPOT] PR-Z1 2-cell heihe_x4 SPGMR spot-check aggregator"
echo "[aggregate-P9-SPOT] run_dir: ${RUN_DIR}"
echo ""

# ---------- Helpers -------------------------------------------------------

# Grab a KV field (`KEY=VAL`) from a cell_summary block.
extract_kv() {
    local cell_out="$1"
    local key="$2"
    grep -oE "${key}=[^[:space:]]+" "${cell_out}" 2>/dev/null \
        | head -n 1 \
        | awk -F= '{print $2}'
}

# Extract a CVODE Final Statistics value (`KEY = VAL` layout) — mirrors
# aggregate_rca.sh::extract_cvode_stat for parse-shape parity.
extract_cvode_stat() {
    local cell_out="$1"
    local key="$2"
    awk -v key="${key}" '
        {
            for (i=1; i<=NF; i++) {
                if ($i == key) {
                    if (i+1 <= NF && $(i+1) == "=") {
                        if (i+2 <= NF) {
                            print $(i+2)
                            exit
                        }
                    } else if (i+1 <= NF && $(i+1) ~ /^=/) {
                        v = substr($(i+1), 2)
                        if (v != "") {
                            print v
                            exit
                        }
                        if (i+2 <= NF) {
                            print $(i+2)
                            exit
                        }
                    }
                }
            }
        }
    ' "${cell_out}" 2>/dev/null
}

# Ratio a/b as printf "%.4f" (or NA if either operand is missing / b==0).
ratio() {
    local a="$1"
    local b="$2"
    if [[ -z "${a}" || "${a}" == "NA" || -z "${b}" || "${b}" == "NA" || "${b}" == "0" ]]; then
        echo "NA"
    else
        awk -v a="${a}" -v b="${b}" 'BEGIN { printf "%.4f\n", a / b }'
    fi
}

# ---------- Parse baseline cell -------------------------------------------
BASELINE_OUT="${RUN_DIR}/cell-0-baseline.out"
P9_OUT="${RUN_DIR}/cell-1-p9.out"

BASELINE_PRESENT=1
P9_PRESENT=1

if [[ ! -f "${BASELINE_OUT}" ]]; then
    echo "MARKER:PR_Z1_CELL_ABSENT cell=baseline path=${BASELINE_OUT}" >&2
    BASELINE_PRESENT=0
fi
if [[ ! -f "${P9_OUT}" ]]; then
    echo "MARKER:PR_Z1_CELL_ABSENT cell=p9 path=${P9_OUT}" >&2
    P9_PRESENT=0
fi

# ---- baseline ----
if [[ ${BASELINE_PRESENT} -eq 1 ]]; then
    B_WALL=$(extract_kv "${BASELINE_OUT}" "wall_total_sec")
    B_NST=$(extract_cvode_stat "${BASELINE_OUT}" "nst")
    B_NCFN=$(extract_cvode_stat "${BASELINE_OUT}" "ncfn")
    B_NLI=$(extract_cvode_stat "${BASELINE_OUT}" "nli")
    B_VERDICT_CLASS=$(extract_kv "${BASELINE_OUT}" "verdict_class")
else
    B_WALL="NA"; B_NST="NA"; B_NCFN="NA"; B_NLI="NA"; B_VERDICT_CLASS="ABSENT"
fi

# ---- p9 ----
if [[ ${P9_PRESENT} -eq 1 ]]; then
    P_WALL=$(extract_kv "${P9_OUT}" "wall_total_sec")
    P_NST=$(extract_cvode_stat "${P9_OUT}" "nst")
    P_NCFN=$(extract_cvode_stat "${P9_OUT}" "ncfn")
    P_NLI=$(extract_cvode_stat "${P9_OUT}" "nli")
    P_VERDICT_CLASS=$(extract_kv "${P9_OUT}" "verdict_class")
else
    P_WALL="NA"; P_NST="NA"; P_NCFN="NA"; P_NLI="NA"; P_VERDICT_CLASS="ABSENT"
fi

# Fallback for missing stats -> NA
: "${B_WALL:=NA}"; : "${B_NST:=NA}"; : "${B_NCFN:=NA}"; : "${B_NLI:=NA}"; : "${B_VERDICT_CLASS:=UNKNOWN}"
: "${P_WALL:=NA}"; : "${P_NST:=NA}"; : "${P_NCFN:=NA}"; : "${P_NLI:=NA}"; : "${P_VERDICT_CLASS:=UNKNOWN}"

# ---------- Compute ratios ------------------------------------------------
# wall_speedup = baseline_wall / p9_wall (>=1 means p9 is faster)
WALL_SPEEDUP=$(ratio "${B_WALL}" "${P_WALL}")
NST_RATIO=$(ratio "${P_NST}" "${B_NST}")

# ---------- Parse A5 verdict ---------------------------------------------
A5_JSON="${RUN_DIR}/a5-report/a5_metrics.json"
A5_MARKER_LOG="${RUN_DIR}/a5-marker.log"

A5_VERDICT="UNKNOWN"
A5_WEIGHTED_SCORE="NA"

if [[ -f "${A5_JSON}" ]]; then
    # Parse verdict + weighted_score from JSON. Prefer python (uv-native
    # A5 project already needs Python) for correctness; fall back to grep
    # if python is unavailable.
    # A5 JSON schema (A5-v1.0.0): { ..., "overall": { "verdict": ..., "weighted_score": ... } }.
    # Prefer python for correctness; grep fallback nests same keys so it also works.
    if command -v python3 >/dev/null 2>&1; then
        A5_VERDICT=$(python3 -c '
import json
try:
    with open("'"${A5_JSON}"'") as f:
        d = json.load(f)
    ov = d.get("overall") or {}
    # Accept either nested (canonical) or top-level (defensive for older A5 versions)
    v = ov.get("verdict") if isinstance(ov, dict) and "verdict" in ov else d.get("verdict", "UNKNOWN")
    print(v or "UNKNOWN")
except Exception:
    print("UNKNOWN")
' 2>/dev/null)
        A5_WEIGHTED_SCORE=$(python3 -c '
import json
try:
    with open("'"${A5_JSON}"'") as f:
        d = json.load(f)
    ov = d.get("overall") or {}
    s = ov.get("weighted_score") if isinstance(ov, dict) and "weighted_score" in ov else d.get("weighted_score")
    print(f"{s:.4f}" if isinstance(s, (int, float)) else "NA")
except Exception:
    print("NA")
' 2>/dev/null)
    else
        # Grep fallback (works for both nested + top-level layouts since first
        # match of `"verdict":"..."` is the one we want in the current schema).
        A5_VERDICT=$(grep -oE '"verdict"[[:space:]]*:[[:space:]]*"[^"]+"' "${A5_JSON}" 2>/dev/null | head -n 1 | awk -F'"' '{print $4}')
        A5_WEIGHTED_SCORE=$(grep -oE '"weighted_score"[[:space:]]*:[[:space:]]*[0-9.]+' "${A5_JSON}" 2>/dev/null | head -n 1 | awk -F':' '{gsub(/[[:space:]]/,"",$2); print $2}')
    fi
elif [[ -f "${A5_MARKER_LOG}" ]]; then
    # Marker-log fallback if JSON absent (A5 exited before writing report
    # but MARKER may still be in stdout log).
    A5_VERDICT=$(grep -E '^verdict=' "${A5_MARKER_LOG}" 2>/dev/null | head -n 1 | awk -F= '{print $2}')
    A5_WEIGHTED_SCORE=$(grep -E '^weighted_score=' "${A5_MARKER_LOG}" 2>/dev/null | head -n 1 | awk -F= '{print $2}')
else
    echo "MARKER:PR_Z1_A5_ARTIFACTS_MISSING run_dir=${RUN_DIR}" >&2
fi

: "${A5_VERDICT:=UNKNOWN}"
: "${A5_WEIGHTED_SCORE:=NA}"

# ---------- Decision logic -----------------------------------------------
# open_full_sweep : wall_speedup >= 1.5 AND a5_verdict=PASS
# optional_p9     : 1.2 <= wall_speedup < 1.5 AND a5_verdict=PASS
# close_p9        : otherwise
P9_DECISION="close_p9"
if [[ "${A5_VERDICT}" == "PASS" && "${WALL_SPEEDUP}" != "NA" ]]; then
    DECISION=$(awk -v s="${WALL_SPEEDUP}" 'BEGIN {
        if (s >= 1.5) print "open_full_sweep"
        else if (s >= 1.2) print "optional_p9"
        else print "close_p9"
    }')
    P9_DECISION="${DECISION}"
fi

# ---------- Emit MARKER block --------------------------------------------
cat <<EOF
MARKER:PR_Z1_VERDICT_BEGIN
baseline_wall_sec=${B_WALL}
p9_wall_sec=${P_WALL}
wall_speedup=${WALL_SPEEDUP}
baseline_nst=${B_NST}
p9_nst=${P_NST}
nst_ratio=${NST_RATIO}
baseline_ncfn=${B_NCFN}
p9_ncfn=${P_NCFN}
baseline_nli=${B_NLI}
p9_nli=${P_NLI}
baseline_verdict_class=${B_VERDICT_CLASS}
p9_verdict_class=${P_VERDICT_CLASS}
a5_verdict=${A5_VERDICT}
a5_weighted_score=${A5_WEIGHTED_SCORE}
p9_decision=${P9_DECISION}
MARKER:PR_Z1_VERDICT_END
EOF

exit 0
