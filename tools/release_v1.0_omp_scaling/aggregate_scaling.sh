#!/bin/bash
# tools/release_v1.0_omp_scaling/aggregate_scaling.sh
#
# Release v1.0 OMP scaling aggregator. Consumes:
#   - ${RUN_DIR}/cell-<TASK>-nthreads-<N>.out       (5 cells: N=1,2,4,8,16)
#   - ${RUN_DIR}/a5-report-nthreads-<N>/a5_metrics.json (4 reports: N=2,4,8,16)
# and emits MARKER:RELEASE_V1_0_SCALING_VERDICT block with per-N wall
# breakdown, speedup ratios, parallel efficiency, A5 verdict pass-through,
# and an overall scaling verdict.
#
# Decision logic (per brief):
#   PRODUCTION_APPROVED : all N A5=PASS AND n8_speedup_total >= 2.5
#   CONDITIONAL         : all N A5=PASS AND 1.5 <= n8_speedup_total < 2.5
#                         OR n8_speedup_total < 1.5 with A5 PASS
#                         (diminishing returns, still safe)
#   BLOCKED             : ANY N A5=FAIL (thread count breaks trajectory
#                         equivalence -- unsafe for production)
#
# Usage:
#   ./aggregate_scaling.sh <run_dir>
#
# Args:
#   run_dir  directory containing cell-{0..4}-nthreads-*.{out,err} +
#            a5-report-nthreads-{2,4,8,16}/a5_metrics.json
#
# Exit code:
#   0  aggregator succeeded (verdict emitted; may be any of PRODUCTION_APPROVED
#      / CONDITIONAL / BLOCKED)
#   2  unrecoverable parse error (missing sidecars, cannot compute speedup
#      because reference N=1 wall is NA)

set -uo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run_dir>" >&2
    exit 2
fi

RUN_DIR="$1"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[aggregate-scaling] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 2
fi

echo "[aggregate-scaling] Release v1.0 heihe_x4 OMP scaling aggregator"
echo "[aggregate-scaling] run_dir: ${RUN_DIR}"
echo ""

# ---------- Helpers -------------------------------------------------------

# Extract a KV field (`KEY=VAL`) from a cell_summary block. Uses grep
# for line-level match then awk for value isolation.
extract_kv() {
    local cell_out="$1"
    local key="$2"
    grep -oE "^${key}=[^[:space:]]+" "${cell_out}" 2>/dev/null \
        | head -n 1 \
        | awk -F= '{print $2}'
}

# Extract a CVODE Final Statistics value (`KEY = VAL` layout, whitespace-
# separated). Same shape as tools/p9.spot/aggregate_p9_spot.sh.
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

# Divide a ratio by an integer (parallel efficiency = speedup / N).
divide_by() {
    local a="$1"
    local b="$2"
    if [[ -z "${a}" || "${a}" == "NA" || -z "${b}" || "${b}" == "0" ]]; then
        echo "NA"
    else
        awk -v a="${a}" -v b="${b}" 'BEGIN { printf "%.4f\n", a / b }'
    fi
}

# Parse A5 verdict from a5_metrics.json (nested schema; falls back to
# marker log if JSON absent).
parse_a5_verdict() {
    local a5_json="$1"
    local marker_log="$2"
    if [[ -f "${a5_json}" ]] && command -v python3 >/dev/null 2>&1; then
        python3 -c '
import json, sys
try:
    with open("'"${a5_json}"'") as f:
        d = json.load(f)
    ov = d.get("overall") or {}
    v = ov.get("verdict") if isinstance(ov, dict) and "verdict" in ov else d.get("verdict", "UNKNOWN")
    print(v or "UNKNOWN")
except Exception:
    print("UNKNOWN")
' 2>/dev/null
    elif [[ -f "${a5_json}" ]]; then
        # Grep fallback (works for nested or top-level layouts).
        grep -oE '"verdict"[[:space:]]*:[[:space:]]*"[^"]+"' "${a5_json}" 2>/dev/null \
            | head -n 1 | awk -F'"' '{print $4}'
    elif [[ -f "${marker_log}" ]]; then
        grep -E '^verdict=' "${marker_log}" 2>/dev/null | head -n 1 | awk -F= '{print $2}'
    else
        echo "UNKNOWN"
    fi
}

# Parse A5 weighted_score from a5_metrics.json.
parse_a5_score() {
    local a5_json="$1"
    local marker_log="$2"
    if [[ -f "${a5_json}" ]] && command -v python3 >/dev/null 2>&1; then
        python3 -c '
import json
try:
    with open("'"${a5_json}"'") as f:
        d = json.load(f)
    ov = d.get("overall") or {}
    s = ov.get("weighted_score") if isinstance(ov, dict) and "weighted_score" in ov else d.get("weighted_score")
    print(f"{s:.4f}" if isinstance(s, (int, float)) else "NA")
except Exception:
    print("NA")
' 2>/dev/null
    elif [[ -f "${a5_json}" ]]; then
        grep -oE '"weighted_score"[[:space:]]*:[[:space:]]*[0-9.]+' "${a5_json}" 2>/dev/null \
            | head -n 1 | awk -F: '{gsub(/[[:space:]]/,"",$2); print $2}'
    elif [[ -f "${marker_log}" ]]; then
        grep -E '^weighted_score=' "${marker_log}" 2>/dev/null | head -n 1 | awk -F= '{print $2}'
    else
        echo "NA"
    fi
}

# ---------- Parse per-cell wall / CVODE stats -----------------------------
NTHREADS_LIST=(1 2 4 8 16)

# Baseline (N=1) parse. All downstream ratios reference this.
BASELINE_OUT="${RUN_DIR}/cell-0-nthreads-1.out"
if [[ ! -f "${BASELINE_OUT}" ]]; then
    echo "MARKER:RELEASE_V1_SCALING_CELL_ABSENT cell=nthreads=1 path=${BASELINE_OUT}" >&2
    echo "[aggregate-scaling] FATAL: baseline cell (N=1) sidecar missing; cannot compute speedups" >&2
    exit 2
fi

N1_WALL_TOTAL=$(extract_kv "${BASELINE_OUT}" "wall_total_sec")
N1_WALL_SETUP=$(extract_kv "${BASELINE_OUT}" "wall_setup_sec")
N1_WALL_SOLVE=$(extract_kv "${BASELINE_OUT}" "wall_solve_sec")
N1_NST=$(extract_cvode_stat "${BASELINE_OUT}" "nst")
N1_NCFN=$(extract_cvode_stat "${BASELINE_OUT}" "ncfn")
N1_VERDICT_CLASS=$(extract_kv "${BASELINE_OUT}" "verdict_class")

: "${N1_WALL_TOTAL:=NA}"; : "${N1_WALL_SETUP:=NA}"; : "${N1_WALL_SOLVE:=NA}"
: "${N1_NST:=NA}"; : "${N1_NCFN:=NA}"; : "${N1_VERDICT_CLASS:=UNKNOWN}"

if [[ "${N1_WALL_TOTAL}" == "NA" ]]; then
    echo "MARKER:RELEASE_V1_SCALING_BASELINE_UNPARSED baseline_out=${BASELINE_OUT}" >&2
    echo "[aggregate-scaling] FATAL: baseline wall_total_sec unparsed; cannot compute speedups" >&2
    exit 2
fi

# Per-N parse. Stored in name-indexed vars for readable MARKER emission.
declare -A W_TOTAL W_SETUP W_SOLVE NST NCFN VERDICT_CLASS
declare -A SPEEDUP_TOTAL SPEEDUP_SOLVE PAR_EFF A5_VERDICT A5_SCORE

W_TOTAL[1]="${N1_WALL_TOTAL}"
W_SETUP[1]="${N1_WALL_SETUP}"
W_SOLVE[1]="${N1_WALL_SOLVE}"
NST[1]="${N1_NST}"
NCFN[1]="${N1_NCFN}"
VERDICT_CLASS[1]="${N1_VERDICT_CLASS}"
SPEEDUP_TOTAL[1]="1.0000"
SPEEDUP_SOLVE[1]="1.0000"
PAR_EFF[1]="1.0000"
A5_VERDICT[1]="N/A_REFERENCE"
A5_SCORE[1]="N/A"

ANY_A5_FAIL=0
ANY_A5_UNKNOWN=0

for TASK in 1 2 3 4; do
    N="${NTHREADS_LIST[${TASK}]}"
    CELL_OUT="${RUN_DIR}/cell-${TASK}-nthreads-${N}.out"

    if [[ ! -f "${CELL_OUT}" ]]; then
        echo "MARKER:RELEASE_V1_SCALING_CELL_ABSENT cell=nthreads=${N} path=${CELL_OUT}" >&2
        W_TOTAL[${N}]="NA"; W_SETUP[${N}]="NA"; W_SOLVE[${N}]="NA"
        NST[${N}]="NA"; NCFN[${N}]="NA"; VERDICT_CLASS[${N}]="ABSENT"
        SPEEDUP_TOTAL[${N}]="NA"; SPEEDUP_SOLVE[${N}]="NA"; PAR_EFF[${N}]="NA"
        A5_VERDICT[${N}]="UNKNOWN"
        A5_SCORE[${N}]="NA"
        ANY_A5_UNKNOWN=1
        continue
    fi

    W_TOTAL[${N}]=$(extract_kv "${CELL_OUT}" "wall_total_sec")
    W_SETUP[${N}]=$(extract_kv "${CELL_OUT}" "wall_setup_sec")
    W_SOLVE[${N}]=$(extract_kv "${CELL_OUT}" "wall_solve_sec")
    NST[${N}]=$(extract_cvode_stat "${CELL_OUT}" "nst")
    NCFN[${N}]=$(extract_cvode_stat "${CELL_OUT}" "ncfn")
    VERDICT_CLASS[${N}]=$(extract_kv "${CELL_OUT}" "verdict_class")

    : "${W_TOTAL[${N}]:=NA}"; : "${W_SETUP[${N}]:=NA}"; : "${W_SOLVE[${N}]:=NA}"
    : "${NST[${N}]:=NA}"; : "${NCFN[${N}]:=NA}"; : "${VERDICT_CLASS[${N}]:=UNKNOWN}"

    SPEEDUP_TOTAL[${N}]=$(ratio "${N1_WALL_TOTAL}" "${W_TOTAL[${N}]}")
    SPEEDUP_SOLVE[${N}]=$(ratio "${N1_WALL_SOLVE}" "${W_SOLVE[${N}]}")
    PAR_EFF[${N}]=$(divide_by "${SPEEDUP_TOTAL[${N}]}" "${N}")

    A5_JSON="${RUN_DIR}/a5-report-nthreads-${N}/a5_metrics.json"
    A5_MARKER_LOG="${RUN_DIR}/a5-report-nthreads-${N}/a5-marker.log"
    A5_VERDICT[${N}]=$(parse_a5_verdict "${A5_JSON}" "${A5_MARKER_LOG}")
    A5_SCORE[${N}]=$(parse_a5_score "${A5_JSON}" "${A5_MARKER_LOG}")
    : "${A5_VERDICT[${N}]:=UNKNOWN}"
    : "${A5_SCORE[${N}]:=NA}"

    case "${A5_VERDICT[${N}]}" in
        PASS) ;;
        FAIL) ANY_A5_FAIL=1 ;;
        *)    ANY_A5_UNKNOWN=1 ;;
    esac
done

# ---------- Decision logic -----------------------------------------------
#   BLOCKED             : ANY N A5=FAIL
#   PRODUCTION_APPROVED : all N A5=PASS AND n8_speedup_total >= 2.5
#   CONDITIONAL         : all N A5=PASS AND speedup axis below 2.5
OVERALL_VERDICT="BLOCKED"
OVERALL_REASONING="one or more A5 verdicts UNKNOWN or missing evidence"

N8_SPEEDUP="${SPEEDUP_TOTAL[8]:-NA}"

if [[ ${ANY_A5_FAIL} -eq 1 ]]; then
    OVERALL_VERDICT="BLOCKED"
    OVERALL_REASONING="A5 FAIL at one or more thread counts -- trajectory equivalence broken; unsafe for production"
elif [[ ${ANY_A5_UNKNOWN} -eq 1 ]]; then
    OVERALL_VERDICT="BLOCKED"
    OVERALL_REASONING="A5 verdict UNKNOWN at one or more thread counts (missing artifacts or A5 harness error)"
else
    # all A5 verdicts PASS
    if [[ "${N8_SPEEDUP}" == "NA" ]]; then
        OVERALL_VERDICT="CONDITIONAL"
        OVERALL_REASONING="all N A5=PASS but N=8 speedup unparseable; caveat production approval pending manual review"
    else
        DECISION=$(awk -v s="${N8_SPEEDUP}" 'BEGIN {
            if (s >= 2.5)      print "PRODUCTION_APPROVED"
            else if (s >= 1.5) print "CONDITIONAL_STRONG"
            else               print "CONDITIONAL_WEAK"
        }')
        case "${DECISION}" in
            PRODUCTION_APPROVED)
                OVERALL_VERDICT="PRODUCTION_APPROVED"
                OVERALL_REASONING="all N A5=PASS and N=8 speedup>=2.5x -- production baseline for release v1.0"
                ;;
            CONDITIONAL_STRONG)
                OVERALL_VERDICT="CONDITIONAL"
                OVERALL_REASONING="all N A5=PASS and 1.5<=N=8 speedup<2.5x -- parallel gain OK but modest"
                ;;
            CONDITIONAL_WEAK)
                OVERALL_VERDICT="CONDITIONAL"
                OVERALL_REASONING="all N A5=PASS but N=8 speedup<1.5x -- diminishing returns; safe but limited parallel benefit"
                ;;
        esac
    fi
fi

# ---------- Emit MARKER block --------------------------------------------
cat <<EOF
MARKER:RELEASE_V1_0_SCALING_VERDICT_BEGIN
case=heihe_x4
cfg_reltol=1e-4
cfg_period_days=90
n1_wall_total_sec=${W_TOTAL[1]}
n1_wall_setup_sec=${W_SETUP[1]}
n1_wall_solve_sec=${W_SOLVE[1]}
n1_nst=${NST[1]}
n1_ncfn=${NCFN[1]}
n1_verdict_class=${VERDICT_CLASS[1]}
n2_wall_total_sec=${W_TOTAL[2]}
n2_wall_setup_sec=${W_SETUP[2]}
n2_wall_solve_sec=${W_SOLVE[2]}
n2_nst=${NST[2]}
n2_ncfn=${NCFN[2]}
n2_verdict_class=${VERDICT_CLASS[2]}
n2_speedup_total=${SPEEDUP_TOTAL[2]}
n2_speedup_solve=${SPEEDUP_SOLVE[2]}
n2_parallel_efficiency=${PAR_EFF[2]}
n2_a5_verdict=${A5_VERDICT[2]}
n2_a5_weighted_score=${A5_SCORE[2]}
n4_wall_total_sec=${W_TOTAL[4]}
n4_wall_setup_sec=${W_SETUP[4]}
n4_wall_solve_sec=${W_SOLVE[4]}
n4_nst=${NST[4]}
n4_ncfn=${NCFN[4]}
n4_verdict_class=${VERDICT_CLASS[4]}
n4_speedup_total=${SPEEDUP_TOTAL[4]}
n4_speedup_solve=${SPEEDUP_SOLVE[4]}
n4_parallel_efficiency=${PAR_EFF[4]}
n4_a5_verdict=${A5_VERDICT[4]}
n4_a5_weighted_score=${A5_SCORE[4]}
n8_wall_total_sec=${W_TOTAL[8]}
n8_wall_setup_sec=${W_SETUP[8]}
n8_wall_solve_sec=${W_SOLVE[8]}
n8_nst=${NST[8]}
n8_ncfn=${NCFN[8]}
n8_verdict_class=${VERDICT_CLASS[8]}
n8_speedup_total=${SPEEDUP_TOTAL[8]}
n8_speedup_solve=${SPEEDUP_SOLVE[8]}
n8_parallel_efficiency=${PAR_EFF[8]}
n8_a5_verdict=${A5_VERDICT[8]}
n8_a5_weighted_score=${A5_SCORE[8]}
n16_wall_total_sec=${W_TOTAL[16]}
n16_wall_setup_sec=${W_SETUP[16]}
n16_wall_solve_sec=${W_SOLVE[16]}
n16_nst=${NST[16]}
n16_ncfn=${NCFN[16]}
n16_verdict_class=${VERDICT_CLASS[16]}
n16_speedup_total=${SPEEDUP_TOTAL[16]}
n16_speedup_solve=${SPEEDUP_SOLVE[16]}
n16_parallel_efficiency=${PAR_EFF[16]}
n16_a5_verdict=${A5_VERDICT[16]}
n16_a5_weighted_score=${A5_SCORE[16]}
overall_scaling_verdict=${OVERALL_VERDICT}
overall_reasoning=${OVERALL_REASONING}
MARKER:RELEASE_V1_0_SCALING_VERDICT_END
EOF

exit 0
