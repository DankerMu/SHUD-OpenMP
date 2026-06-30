#!/bin/bash
# tools/p8tune.G0-rca/aggregate_rca.sh
#
# P8-tune.G0 PR-X1 — aggregator binary for the 8-cell heihe_x4 AMG
# tolerance + CVODE EpsLin RCA matrix per the PR-X1 brief.
#
# Background: G0 verdict (#412) NO-GO root-cause analysis. heihe_x4
# AMG baseline showed ncfl=0 + ncfn=100138 (Newton control failure
# explosion) + nst_ratio=38.8x vs SPGMR + 15.1x wall regression.
# This aggregator processes the 8-cell tolerance sweep results and
# emits a verdict deciding whether AMG can be conditionally reopened.
#
# Per-cell expectations (8 rows total, see amg_rca_array.sbatch):
#   TASK 0: AMG_TOL=1e-7  EPSLIN=default
#   TASK 1: AMG_TOL=1e-9  EPSLIN=default
#   TASK 2: AMG_TOL=1e-11 EPSLIN=default
#   TASK 3: AMG_TOL=1e-13 EPSLIN=default
#   TASK 4: AMG_TOL=1e-7  EPSLIN=0.005
#   TASK 5: AMG_TOL=1e-9  EPSLIN=0.005
#   TASK 6: AMG_TOL=1e-11 EPSLIN=0.005
#   TASK 7: AMG_TOL=1e-13 EPSLIN=0.005
#
# All cells are heihe_x4 (the PR-X1 scope-locked case). Per-cell sidecar
# filenames are `cell-<TASK_ID>-tol<AMG_TOL>-eps<EPSLIN>.{out,err,telemetry.tsv}`.
#
# SPGMR baseline reference (hardcoded — from `.review-evidence/
# g0-spgmr-baselines-heihe/heihe_x4/`):
#   SPGMR_NCFN          = 49
#   SPGMR_NST           = 6572
#   SPGMR_WALL_TOTAL_SEC = 1566.5
#
# These are pinned constants for the PR-X1 verdict math; they are NOT
# re-derived from spgmr_baseline_walls_g0.h (which only pins per-step
# walls). Drift detection is out of scope here (PR-X2 handles).
#
# Output:
#   - <run_dir>/aggregate_rca.tsv   12-column matrix (8 data rows)
#   - stdout PR_X1_VERDICT_BEGIN/END block
#
# Usage:
#   ./aggregate_rca.sh <run_dir>
#
# Args:
#   run_dir  directory containing 8 sidecars:
#            cell-<TASK_ID>-tol<AMG_TOL>-eps<EPSLIN>.{out,err}
#            and optional .telemetry.tsv files.
#
# Exit code:
#   0  aggregator succeeded (verdict emitted; amg_reopens may be
#      true or false depending on cell quality).
#   2  unrecoverable parse error (e.g., run_dir missing, no cells
#      found).
#
# Refs:
#   - PR-X1 brief (RCA for G0 verdict #412 NO-GO)
#   - tools/p8tune.G0/aggregate_g0_smoke.sh (template; this PR-X1
#     extension swaps the cell axis to the (AMG_TOL, EPSLIN) matrix
#     and adds 5 vs-SPGMR ratio columns)
#   - .review-evidence/g0-spgmr-baselines-heihe/heihe_x4/cvode_stats.txt
#     (SPGMR baseline source)

set -uo pipefail

# ---------- CLI -------------------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run_dir>" >&2
    echo "  run_dir   directory with cell-<TASK_ID>-tol<AMG_TOL>-eps<EPSLIN>.{out,err}" >&2
    exit 2
fi

RUN_DIR="$1"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[aggregate-RCA] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 2
fi

# ---------- SPGMR baseline constants (hardcoded per PR-X1 brief) ----------
# Source: .review-evidence/g0-spgmr-baselines-heihe/heihe_x4/cvode_stats.txt
# and wall-total.txt. Pinned here because PR-X1 is RCA-only — drift detection
# vs the live spgmr_baseline_walls_g0.h is PR-X2 scope.
SPGMR_NCFN=49
SPGMR_NST=6572
SPGMR_WALL_TOTAL_SEC=1566.5518531799316

echo "[aggregate-RCA] PR-X1 8-cell heihe_x4 RCA aggregator"
echo "[aggregate-RCA] run_dir:              ${RUN_DIR}"
echo "[aggregate-RCA] SPGMR baseline:       ncfn=${SPGMR_NCFN} nst=${SPGMR_NST} wall=${SPGMR_WALL_TOTAL_SEC}s"
echo ""

# ---------- Expected matrix coordinates (per amg_rca_array.sbatch) --------
EXPECTED_CONFIGS=(
    "0:1e-7:default"
    "1:1e-9:default"
    "2:1e-11:default"
    "3:1e-13:default"
    "4:1e-7:0.005"
    "5:1e-9:0.005"
    "6:1e-11:0.005"
    "7:1e-13:0.005"
)

# ---------- Helpers ---------------------------------------------------------

# Extract a `<KEY>=<VAL>` field from a cell_summary block.
extract_kv() {
    local cell_out="$1"
    local key="$2"
    grep -oE "${key}=[^[:space:]]+" "${cell_out}" 2>/dev/null \
        | head -n 1 \
        | awk -F= '{print $2}'
}

# Extract a CVODE Final Statistics value (`KEY     = VAL`) from raw stdout.
# Mirrors aggregate_g0_smoke.sh::extract_cvode_stat to keep parse behavior
# bit-aligned with the G0 aggregator.
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

# Compute a/b as printf "%.4f" (or NA if either is missing).
ratio() {
    local a="$1"
    local b="$2"
    if [[ -z "${a}" || "${a}" == "NA" || -z "${b}" || "${b}" == "NA" || "${b}" == "0" ]]; then
        echo "NA"
    else
        awk -v a="${a}" -v b="${b}" 'BEGIN { printf "%.4f\n", a / b }'
    fi
}

# ---------- Per-cell parse + accumulation ---------------------------------
declare -a CELL_TASK_ID
declare -a CELL_AMG_TOL
declare -a CELL_EPSLIN
declare -a CELL_RUN_LABEL
declare -a CELL_VERDICT_CLASS
declare -a CELL_EXIT_CODE
declare -a CELL_WALL_TOTAL_SEC
declare -a CELL_NST
declare -a CELL_NFE
declare -a CELL_NLI
declare -a CELL_NCFN
declare -a CELL_NCFL
declare -a CELL_NETF
declare -a CELL_NCFN_RATIO
declare -a CELL_NST_RATIO
declare -a CELL_WALL_RATIO
declare -a CELL_HYPRE_RELEASE
declare -a CELL_PRESENT

NUM_PRESENT=0
NUM_AMG_OK=0

for CONFIG in "${EXPECTED_CONFIGS[@]}"; do
    IFS=':' read -r TASK_ID AMG_TOL EPSLIN <<< "${CONFIG}"
    RUN_LABEL="${TASK_ID}-tol${AMG_TOL}-eps${EPSLIN}"
    CELL_OUT="${RUN_DIR}/cell-${RUN_LABEL}.out"
    CELL_ERR="${RUN_DIR}/cell-${RUN_LABEL}.err"

    CELL_TASK_ID+=("${TASK_ID}")
    CELL_AMG_TOL+=("${AMG_TOL}")
    CELL_EPSLIN+=("${EPSLIN}")
    CELL_RUN_LABEL+=("${RUN_LABEL}")

    if [[ ! -f "${CELL_OUT}" ]]; then
        echo "MARKER:PR_X1_CELL_REJECTED_MALFORMED task=${TASK_ID} amg_tol=${AMG_TOL} epslin=${EPSLIN} reason=cell_out_absent"
        CELL_PRESENT+=("0")
        CELL_VERDICT_CLASS+=("MALFORMED")
        CELL_EXIT_CODE+=("NA")
        CELL_WALL_TOTAL_SEC+=("NA")
        CELL_NST+=("NA")
        CELL_NFE+=("NA")
        CELL_NLI+=("NA")
        CELL_NCFN+=("NA")
        CELL_NCFL+=("NA")
        CELL_NETF+=("NA")
        CELL_NCFN_RATIO+=("NA")
        CELL_NST_RATIO+=("NA")
        CELL_WALL_RATIO+=("NA")
        CELL_HYPRE_RELEASE+=("NA")
        continue
    fi

    CELL_PRESENT+=("1")
    NUM_PRESENT=$((NUM_PRESENT + 1))

    # cell_summary block check
    if ! grep -q "^CELL_SUMMARY_BEGIN" "${CELL_OUT}" 2>/dev/null; then
        # Distinguish SIGKILL-from-Slurm vs hard malformed.
        if [[ -f "${CELL_ERR}" ]] && \
           grep -qE "CANCELLED AT.*DUE TO TIME LIMIT|slurmstepd: error.*CANCELLED" "${CELL_ERR}" 2>/dev/null; then
            echo "MARKER:PR_X1_CELL_REJECTED_MALFORMED task=${TASK_ID} reason=wall_overflow_inferred_sigkill"
            CELL_VERDICT_CLASS+=("AMG_WALL_OVERFLOW_INFERRED_SIGKILL")
        else
            echo "MARKER:PR_X1_CELL_REJECTED_MALFORMED task=${TASK_ID} reason=cell_summary_missing"
            CELL_VERDICT_CLASS+=("MALFORMED")
        fi
    else
        VC=$(extract_kv "${CELL_OUT}" "verdict_class")
        CELL_VERDICT_CLASS+=("${VC:-MISSING}")
    fi

    # exit_code + wall_total_sec from the runner footer line
    #   [shud-G0-RCA] cell=heihe_x4 run_label=<L> ... exit_code=<X> wall_total_sec=<W> ...
    FOOTER_LINE=$(grep -E "^\[shud-G0-RCA\] cell=" "${CELL_OUT}" 2>/dev/null | tail -n 1)
    if [[ -n "${FOOTER_LINE}" ]]; then
        EX=$(echo "${FOOTER_LINE}" | grep -oE "exit_code=[0-9]+" | awk -F= '{print $2}')
        WT=$(echo "${FOOTER_LINE}" | grep -oE "wall_total_sec=[0-9]+" | awk -F= '{print $2}')
    else
        EX="NA"
        WT="NA"
    fi
    CELL_EXIT_CODE+=("${EX:-NA}")
    CELL_WALL_TOTAL_SEC+=("${WT:-NA}")

    # CVODE Final Statistics (raw text)
    NST=$(extract_cvode_stat "${CELL_OUT}" "nst")
    NFE=$(extract_cvode_stat "${CELL_OUT}" "nfe")
    NLI=$(extract_cvode_stat "${CELL_OUT}" "nli")
    NCFN=$(extract_cvode_stat "${CELL_OUT}" "ncfn")
    NCFL=$(extract_cvode_stat "${CELL_OUT}" "ncfl")
    NETF=$(extract_cvode_stat "${CELL_OUT}" "netf")
    CELL_NST+=("${NST:-NA}")
    CELL_NFE+=("${NFE:-NA}")
    CELL_NLI+=("${NLI:-NA}")
    CELL_NCFN+=("${NCFN:-NA}")
    CELL_NCFL+=("${NCFL:-NA}")
    CELL_NETF+=("${NETF:-NA}")

    # Ratios vs SPGMR baseline
    NCFN_R=$(ratio "${NCFN:-NA}" "${SPGMR_NCFN}")
    NST_R=$(ratio "${NST:-NA}" "${SPGMR_NST}")
    WALL_R=$(ratio "${WT:-NA}" "${SPGMR_WALL_TOTAL_SEC}")
    CELL_NCFN_RATIO+=("${NCFN_R}")
    CELL_NST_RATIO+=("${NST_R}")
    CELL_WALL_RATIO+=("${WALL_R}")

    # Hypre release from MARKER:AMG_TELEMETRY_REAL
    HR=$(grep -E "^MARKER:AMG_TELEMETRY_REAL" "${CELL_OUT}" 2>/dev/null | head -n 1 \
         | grep -oE "hypre_release=[0-9]+" | awk -F= '{print $2}')
    CELL_HYPRE_RELEASE+=("${HR:-NA}")

    # Tally
    if [[ "${CELL_VERDICT_CLASS[$((${#CELL_VERDICT_CLASS[@]}-1))]}" == "AMG_OK" ]]; then
        NUM_AMG_OK=$((NUM_AMG_OK + 1))
    fi
done

echo "[aggregate-RCA] discovered ${NUM_PRESENT}/8 cells, ${NUM_AMG_OK} AMG_OK"
echo ""

# ---------- aggregate_rca.tsv emission ------------------------------------
AGGREGATE_TSV="${RUN_DIR}/aggregate_rca.tsv"
{
    # 16 columns. Naming kept close to aggregate_g0_smoke.sh where overlap
    # exists, with PR-X1-specific columns (amg_tol, epslin,
    # ncfn_ratio_vs_spgmr, nst_ratio_vs_spgmr, wall_ratio_vs_spgmr)
    # inserted around the SPGMR comparators.
    printf "task_id\tamg_tol\tepslin\trun_label\tverdict_class\texit_code"
    printf "\twall_total_sec\twall_ratio_vs_spgmr_baseline"
    printf "\tn_cvode_steps\tnst_ratio_vs_spgmr_baseline"
    printf "\tcvode_ncfn\tncfn_ratio_vs_spgmr_baseline"
    printf "\tcvode_ncfl\tcvode_nfe\tcvode_nli\tcvode_netf\thypre_release\n"

    for i in "${!CELL_TASK_ID[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s\t%s" \
            "${CELL_TASK_ID[$i]}" \
            "${CELL_AMG_TOL[$i]}" \
            "${CELL_EPSLIN[$i]}" \
            "${CELL_RUN_LABEL[$i]}" \
            "${CELL_VERDICT_CLASS[$i]}" \
            "${CELL_EXIT_CODE[$i]}"
        printf "\t%s\t%s" \
            "${CELL_WALL_TOTAL_SEC[$i]}" \
            "${CELL_WALL_RATIO[$i]}"
        printf "\t%s\t%s" \
            "${CELL_NST[$i]}" \
            "${CELL_NST_RATIO[$i]}"
        printf "\t%s\t%s" \
            "${CELL_NCFN[$i]}" \
            "${CELL_NCFN_RATIO[$i]}"
        printf "\t%s\t%s\t%s\t%s\t%s\n" \
            "${CELL_NCFL[$i]}" \
            "${CELL_NFE[$i]}" \
            "${CELL_NLI[$i]}" \
            "${CELL_NETF[$i]}" \
            "${CELL_HYPRE_RELEASE[$i]}"
    done
} > "${AGGREGATE_TSV}"

echo "[aggregate-RCA] wrote ${AGGREGATE_TSV}"
echo ""

# ---------- PR-X1 verdict ---------------------------------------------------
# Judge criteria per PR-X1 brief:
#   "amg_reopens = true iff ANY cell has ncfn<5000 AND nst_ratio<2x AND
#    wall<=SPGMR"
# (i.e. all three pass thresholds simultaneously on at least one cell).
#
# "best_cell" = cell with lowest ncfn among AMG_OK cells with valid metrics.
# If no cell is AMG_OK or all metrics are NA, best_cell=none.

BEST_TASK="none"
BEST_NCFN="NA"
BEST_NST_RATIO="NA"
BEST_WALL_RATIO="NA"
AMG_REOPENS="false"

# First pass: find the cell with lowest ncfn among AMG_OK cells.
for i in "${!CELL_TASK_ID[@]}"; do
    if [[ "${CELL_VERDICT_CLASS[$i]}" != "AMG_OK" ]]; then continue; fi
    if [[ "${CELL_NCFN[$i]}" == "NA" || -z "${CELL_NCFN[$i]}" ]]; then continue; fi

    if [[ "${BEST_NCFN}" == "NA" ]]; then
        BEST_TASK="${CELL_TASK_ID[$i]}"
        BEST_NCFN="${CELL_NCFN[$i]}"
        BEST_NST_RATIO="${CELL_NST_RATIO[$i]}"
        BEST_WALL_RATIO="${CELL_WALL_RATIO[$i]}"
    else
        # awk integer compare (handles string-as-int safely)
        IS_LOWER=$(awk -v a="${CELL_NCFN[$i]}" -v b="${BEST_NCFN}" \
            'BEGIN { print (a < b) ? "true" : "false" }')
        if [[ "${IS_LOWER}" == "true" ]]; then
            BEST_TASK="${CELL_TASK_ID[$i]}"
            BEST_NCFN="${CELL_NCFN[$i]}"
            BEST_NST_RATIO="${CELL_NST_RATIO[$i]}"
            BEST_WALL_RATIO="${CELL_WALL_RATIO[$i]}"
        fi
    fi
done

# Second pass: amg_reopens iff ANY AMG_OK cell meets all 3 criteria.
for i in "${!CELL_TASK_ID[@]}"; do
    if [[ "${CELL_VERDICT_CLASS[$i]}" != "AMG_OK" ]]; then continue; fi

    NCFN="${CELL_NCFN[$i]}"
    NST_R="${CELL_NST_RATIO[$i]}"
    WALL_R="${CELL_WALL_RATIO[$i]}"

    if [[ "${NCFN}" == "NA" || "${NST_R}" == "NA" || "${WALL_R}" == "NA" ]]; then
        continue
    fi

    PASSES=$(awk -v n="${NCFN}" -v nr="${NST_R}" -v wr="${WALL_R}" \
        'BEGIN { print (n < 5000 && nr < 2.0 && wr <= 1.0) ? "true" : "false" }')
    if [[ "${PASSES}" == "true" ]]; then
        AMG_REOPENS="true"
        break
    fi
done

# Emit verdict block
cat <<EOF
MARKER:PR_X1_VERDICT_BEGIN
best_cell=${BEST_TASK}
best_ncfn=${BEST_NCFN}
best_nst_ratio=${BEST_NST_RATIO}
best_wall_ratio=${BEST_WALL_RATIO}
amg_reopens=${AMG_REOPENS}
num_present=${NUM_PRESENT}
num_amg_ok=${NUM_AMG_OK}
spgmr_baseline_ncfn=${SPGMR_NCFN}
spgmr_baseline_nst=${SPGMR_NST}
spgmr_baseline_wall_sec=${SPGMR_WALL_TOTAL_SEC}
MARKER:PR_X1_VERDICT_END
EOF

exit 0
