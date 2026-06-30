#!/bin/bash
# tools/p8tune.G0/aggregate_g0_smoke.sh
#
# P8-tune.G0 PR-B (#411) — aggregator binary for the 4-cell AMG
# integrated smoke evidence emitted by PR-A `run_smoke_cell.sh` +
# PR-B `SUNLinSol_Hypre_DrainTelemetry` shutdown hook.
#
# Per issue #411 tasks 7.3-7.8 + spec
# `openspec/changes/p8tune-g0-instrumented-amg-smoke/specs/amg-integrated-smoke-verdict/spec.md`
# REQ "G0 verdict evaluates six PASS/FAIL gates":
#   - Parse per-cell `cell_summary` KV blocks (28-field schema with
#     BEGIN/END delimiters) from `cell-<NAME>.out`.
#   - Parse raw CVODE Final Statistics block (nst/nfe/nli/nfeLS/
#     nsetups/netf/nni/ncfn/ncfl) from the same file (PR-A runner
#     emits these as NA sentinels in cell_summary; raw text holds them).
#   - Parse per-cell telemetry TSV (`cell-<NAME>.telemetry.tsv`) if
#     present (PR-B drain hook output); compute amg_telemetry_*,
#     amg_total_setup_wall_sec, amg_total_solve_wall_sec; derive
#     cycle_complexity = mean(hypre_op_count) / first_op_count from
#     MARKER:AMG_TELEMETRY_REAL.
#   - Reject cells emitting verdict_class=PASS as malformed (G0 sentinel
#     is AMG_OK; spec REQ "verdict_class enum semantics").
#   - Evaluate G0-3 telemetry-real, G0-4 integrated-completes, G0-5
#     wall-signal (case-specific SPGMR baselines from
#     spgmr_baseline_walls_g0.h), G0-6 solver-stats-documented.
#   - Emit `<run_dir>/aggregate.tsv` cross-cell summary.
#   - Emit MARKER:G0_VERDICT_BEGIN ... MARKER:G0_VERDICT_END block to
#     stdout (PR-C consumes byte-identically for verdict doc anchor).
#
# IA-4 (per issue #411 §Out of Scope): MUST NOT mutate
# `tools/p8tune.G0/spgmr_baseline_walls_g0.h`. The header is parsed
# literally (`grep #define`) — no shell-side recomputation.
#
# Usage:
#   ./aggregate_g0_smoke.sh <run_dir>
#
# Args:
#   run_dir   directory containing PR-A `cell-<NAME>.{out,err}`,
#             optional PR-B `cell-<NAME>.telemetry.tsv`, and
#             `precheck.log`. The 4 expected cells are
#             {keliya, xinanjiang_upstream, heihe_x4, heihe_x16}.
#
# Outputs:
#   <run_dir>/aggregate.tsv               (cross-cell summary table)
#   stdout MARKER:G0_VERDICT_BEGIN/END block
#   stdout MARKER:G0_CELL_REJECTED_MALFORMED / MARKER:G0_TELEMETRY_TSV_ABSENT
#          / MARKER:G0_BASELINE_DRIFT_CHECK_REQUESTED diagnostics
#
# Exit code:
#   0  on aggregator success (verdict can be GO-G0 or NO-GO-G0; aggregator
#      itself didn't fail). Caller inspects MARKER:G0_OVERALL=... for the
#      verdict branch.
#   2  unrecoverable parse error (e.g., run_dir missing, no cells found).

set -uo pipefail

# ---------- CLI -------------------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <run_dir>" >&2
    echo "  run_dir   directory with cell-<NAME>.{out,err}[.telemetry.tsv]" >&2
    exit 2
fi

RUN_DIR="$1"
if [[ ! -d "${RUN_DIR}" ]]; then
    echo "[aggregate-G0] FATAL: run_dir not found: ${RUN_DIR}" >&2
    exit 2
fi

# Resolve absolute path for the SPGMR baseline header. The aggregator
# script lives at tools/p8tune.G0/; the header is its sibling.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASELINE_HEADER="${SCRIPT_DIR}/spgmr_baseline_walls_g0.h"

if [[ ! -f "${BASELINE_HEADER}" ]]; then
    echo "[aggregate-G0] FATAL: baseline header not found: ${BASELINE_HEADER}" >&2
    exit 2
fi

# ---------- Parse pinned SPGMR baselines (literal #define) ------------------
# Per IA-4: aggregator parses header literally; mutation forbidden.
# Sentinel WALL_SIGNAL_UNKNOWN expands to (NAN) (math.h); detect both
# the WALL_SIGNAL_UNKNOWN token form and the (NAN) expansion form.
parse_baseline() {
    local sym="$1"
    local line
    line=$(grep -E "^[[:space:]]*#define[[:space:]]+${sym}[[:space:]]" "${BASELINE_HEADER}" | head -n 1)
    if [[ -z "${line}" ]]; then
        echo "MISSING"
        return
    fi
    # Strip "#define <SYM>" prefix + leading/trailing parens/whitespace
    local val
    val=$(echo "${line}" | sed -E "s/^[[:space:]]*#define[[:space:]]+${sym}[[:space:]]+//; s/[[:space:]]*\/\*.*$//; s/^\(//; s/\)$//; s/^[[:space:]]+//; s/[[:space:]]+$//")
    if [[ "${val}" == "WALL_SIGNAL_UNKNOWN" || "${val}" == "NAN" ]]; then
        echo "WALL_SIGNAL_UNKNOWN"
    else
        echo "${val}"
    fi
}

SPGMR_HEIHE_X4=$(parse_baseline "SPGMR_PER_STEP_HEIHE_X4_S")
SPGMR_HEIHE_X16=$(parse_baseline "SPGMR_PER_STEP_HEIHE_X16_S")

echo "[aggregate-G0] baseline header: ${BASELINE_HEADER}"
echo "[aggregate-G0] SPGMR_PER_STEP_HEIHE_X4_S  = ${SPGMR_HEIHE_X4}"
echo "[aggregate-G0] SPGMR_PER_STEP_HEIHE_X16_S = ${SPGMR_HEIHE_X16}"
echo ""

# ---------- Discover cells in run_dir ---------------------------------------
CELLS=()
for cellfile in "${RUN_DIR}"/cell-*.out; do
    [[ -f "${cellfile}" ]] || continue
    base="$(basename "${cellfile}")"
    cell="${base#cell-}"
    cell="${cell%.out}"
    CELLS+=("${cell}")
done

if [[ ${#CELLS[@]} -eq 0 ]]; then
    echo "[aggregate-G0] FATAL: no cell-*.out found in ${RUN_DIR}" >&2
    exit 2
fi

echo "[aggregate-G0] discovered ${#CELLS[@]} cell(s): ${CELLS[*]}"
echo ""

# ---------- Per-cell parse + accumulation -----------------------------------
# Associative arrays keyed by cell name; bash 4+ required (Mac default
# bash is 3.2 — we use indexed arrays + helper functions to stay portable).

# Index-aligned arrays:
declare -a CELL_NAMES
declare -a CELL_VERDICT_CLASS
declare -a CELL_EXIT_CODE
declare -a CELL_WALL_TOTAL_SEC
declare -a CELL_NST
declare -a CELL_NFE
declare -a CELL_NLI
declare -a CELL_NFELS
declare -a CELL_NCFN
declare -a CELL_NCFL
declare -a CELL_NETF
declare -a CELL_NNI
declare -a CELL_NSETUPS
declare -a CELL_HYPRE_RELEASE
declare -a CELL_FIRST_OP_COUNT
declare -a CELL_FIRST_ITERS
declare -a CELL_TELEMETRY_PRESENT
declare -a CELL_AMG_TELEMETRY_MEAN_ITERS
declare -a CELL_AMG_TELEMETRY_MEAN_OP_COUNT
declare -a CELL_AMG_TELEMETRY_SETUP_CALLS
declare -a CELL_AMG_TOTAL_SETUP_WALL_SEC
declare -a CELL_AMG_TOTAL_SOLVE_WALL_SEC
declare -a CELL_AMG_WALL_PER_STEP_SEC
declare -a CELL_CYCLE_COMPLEXITY
declare -a CELL_OPERATOR_COMPLEXITY
declare -a CELL_WALL_CONVENTION
declare -a CELL_WALL_SIGNAL
declare -a CELL_MALFORMED

# Returns the first numeric value extracted from `<KEY>=<VAL>`-style KV lines
# in the cell stdout cell_summary block. Used for fields embedded inside the
# cell_summary block (case=, verdict_class=, n_cvode_steps=, hypre_version=).
extract_kv() {
    local cell_out="$1"
    local key="$2"
    grep -oE "${key}=[^[:space:]]+" "${cell_out}" 2>/dev/null \
        | head -n 1 \
        | awk -F= '{print $2}'
}

# Returns the first numeric value matching `^[[:space:]]*<KEY>[[:space:]]*=`
# from the raw CVODE Final Statistics block (NOT the cell_summary KV).
# SHUD prints `nst     = NNN` (tab/space separated), one per line, with
# some lines containing TWO KVs (`nfe     = X     nfeLS   = Y`).
extract_cvode_stat() {
    local cell_out="$1"
    local key="$2"
    awk -v key="${key}" '
        # Match either `KEY = VAL` or `... KEY = VAL ...` (anywhere on line).
        # Tokens are space/tab separated. Find token equal to KEY, then the
        # next "=" then next numeric. SHUD prints both KEY=VAL pairs on the
        # same line for nfe/nfeLS/nni/nli/nsetups/netf/ncfn/ncfl pairs.
        {
            for (i=1; i<=NF; i++) {
                if ($i == key) {
                    # next token should be "=", then the value (allow direct
                    # adjacency too, e.g., "ncfn=N" though SHUD always uses
                    # padded "ncfn    = N" form)
                    if (i+1 <= NF && $(i+1) == "=") {
                        if (i+2 <= NF) {
                            print $(i+2)
                            exit
                        }
                    } else if (i+1 <= NF && $(i+1) ~ /^=/) {
                        # form "= VAL" attached to "key"
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

# Parse telemetry TSV. Columns:
#   step_idx<TAB>t_sim<TAB>setup_called<TAB>hypre_iters<TAB>hypre_op_count
#   <TAB>setup_wall_sec<TAB>solve_wall_sec<TAB>cvode_nli_step<TAB>cvode_nfeLS_step
# Header row + N data rows. We compute mean iters/op_count + sum
# setup/solve walls + count of setup events.
parse_telemetry_tsv() {
    local tsv="$1"
    # Outputs (whitespace-separated, single line):
    #   mean_iters mean_op_count setup_calls total_setup_wall_sec total_solve_wall_sec n_rows
    awk -F'\t' '
        NR == 1 { next }       # skip header
        /^#/    { next }       # skip overflow-count comment line
        NF >= 7 {
            iters_sum  += $4
            op_sum     += $5
            setup_wall += $6
            solve_wall += $7
            if ($3 == "1") setup_calls++
            n++
        }
        END {
            if (n == 0) {
                print "NA NA 0 NA NA 0"
            } else {
                printf "%.6f %.6f %d %.6e %.6e %d\n",
                    iters_sum/n, op_sum/n, setup_calls,
                    setup_wall, solve_wall, n
            }
        }
    ' "${tsv}" 2>/dev/null
}

CELL_REJECTED_COUNT=0
NUM_AMG_OK=0
NUM_FAIL=0

for CELL in "${CELLS[@]}"; do
    CELL_OUT="${RUN_DIR}/cell-${CELL}.out"
    CELL_TSV="${RUN_DIR}/cell-${CELL}.telemetry.tsv"
    CELL_NAMES+=("${CELL}")

    # Detect cell_summary BEGIN/END delimiters.
    if ! grep -q "^CELL_SUMMARY_BEGIN" "${CELL_OUT}" 2>/dev/null; then
        echo "MARKER:G0_CELL_REJECTED_MALFORMED case=${CELL} reason=cell_summary_missing"
        CELL_REJECTED_COUNT=$((CELL_REJECTED_COUNT + 1))
        CELL_VERDICT_CLASS+=("MALFORMED")
        CELL_MALFORMED+=("1")
    else
        CELL_MALFORMED+=("0")

        # Extract verdict_class. PR-A runner sentinel is AMG_OK; spec
        # G0 rejects legacy PASS sentinel as malformed.
        VC=$(extract_kv "${CELL_OUT}" "verdict_class")
        if [[ "${VC}" == "PASS" ]]; then
            echo "MARKER:G0_CELL_REJECTED_MALFORMED case=${CELL} expected=AMG_OK got=PASS"
            CELL_REJECTED_COUNT=$((CELL_REJECTED_COUNT + 1))
            VC="MALFORMED_PASS"
        fi
        CELL_VERDICT_CLASS+=("${VC:-MISSING}")
    fi

    # exit_code + wall_total_sec from the runner's footer line
    #   [shud-G0] cell=<NAME> exit_code=<X> wall_total_sec=<W> verdict_class=<V>
    FOOTER_LINE=$(grep -E "^\[shud-G0\] cell=${CELL} exit_code=" "${CELL_OUT}" 2>/dev/null | tail -n 1)
    if [[ -n "${FOOTER_LINE}" ]]; then
        EX=$(echo "${FOOTER_LINE}" | grep -oE "exit_code=[0-9]+" | awk -F= '{print $2}')
        WT=$(echo "${FOOTER_LINE}" | grep -oE "wall_total_sec=[0-9]+" | awk -F= '{print $2}')
    else
        EX="NA"
        WT="NA"
    fi
    CELL_EXIT_CODE+=("${EX:-NA}")
    CELL_WALL_TOTAL_SEC+=("${WT:-NA}")

    # CVODE Final Statistics (raw text, NOT cell_summary KV)
    NST=$(extract_cvode_stat "${CELL_OUT}" "nst")
    NFE=$(extract_cvode_stat "${CELL_OUT}" "nfe")
    NLI=$(extract_cvode_stat "${CELL_OUT}" "nli")
    NFELS=$(extract_cvode_stat "${CELL_OUT}" "nfeLS")
    NCFN=$(extract_cvode_stat "${CELL_OUT}" "ncfn")
    NCFL=$(extract_cvode_stat "${CELL_OUT}" "ncfl")
    NETF=$(extract_cvode_stat "${CELL_OUT}" "netf")
    NNI=$(extract_cvode_stat "${CELL_OUT}" "nni")
    NSETUPS=$(extract_cvode_stat "${CELL_OUT}" "nsetups")
    CELL_NST+=("${NST:-NA}")
    CELL_NFE+=("${NFE:-NA}")
    CELL_NLI+=("${NLI:-NA}")
    CELL_NFELS+=("${NFELS:-NA}")
    CELL_NCFN+=("${NCFN:-NA}")
    CELL_NCFL+=("${NCFL:-NA}")
    CELL_NETF+=("${NETF:-NA}")
    CELL_NNI+=("${NNI:-NA}")
    CELL_NSETUPS+=("${NSETUPS:-NA}")

    # MARKER:AMG_TELEMETRY_REAL hypre_release=<N> first_iters=<I> first_op_count=<C>
    MARKER_LINE=$(grep -E "^MARKER:AMG_TELEMETRY_REAL" "${CELL_OUT}" 2>/dev/null | head -n 1)
    HR=$(echo "${MARKER_LINE}" | grep -oE "hypre_release=[0-9]+" | awk -F= '{print $2}')
    FI=$(echo "${MARKER_LINE}" | grep -oE "first_iters=[0-9]+" | awk -F= '{print $2}')
    FOC=$(echo "${MARKER_LINE}" | grep -oE "first_op_count=[0-9]+" | awk -F= '{print $2}')
    CELL_HYPRE_RELEASE+=("${HR:-NA}")
    CELL_FIRST_ITERS+=("${FI:-NA}")
    CELL_FIRST_OP_COUNT+=("${FOC:-NA}")

    # Operator complexity is currently not emitted by the wrapper (gap
    # documented in PR-B summary, deferred to wrapper hot-fix); cell_summary
    # KV `operator_complexity=NA` from PR-A runner.
    CELL_OPERATOR_COMPLEXITY+=("NA")

    # Telemetry TSV parse
    if [[ -f "${CELL_TSV}" ]]; then
        CELL_TELEMETRY_PRESENT+=("1")
        TSV_PARSED=$(parse_telemetry_tsv "${CELL_TSV}")
        # Format: mean_iters mean_op_count setup_calls total_setup_wall total_solve_wall n_rows
        read -r MI MOC SC TSU TSO NR <<< "${TSV_PARSED}"
        CELL_AMG_TELEMETRY_MEAN_ITERS+=("${MI}")
        CELL_AMG_TELEMETRY_MEAN_OP_COUNT+=("${MOC}")
        CELL_AMG_TELEMETRY_SETUP_CALLS+=("${SC}")
        CELL_AMG_TOTAL_SETUP_WALL_SEC+=("${TSU}")
        CELL_AMG_TOTAL_SOLVE_WALL_SEC+=("${TSO}")

        # cycle_complexity = mean(hypre_op_count) / first_op_count
        # Spec amendment 2026-06-29 semantics: STATIC hierarchy-size ratio,
        # NOT per-cycle work ratio. CumNnzAP is cumulative-across-Setup so
        # the mean over the run is a "hierarchy stability + amortized
        # bloat" attestation; first_op_count anchors to the first Solve.
        if [[ "${MOC}" != "NA" && "${FOC}" != "NA" && "${FOC}" != "0" && "${MOC}" != "" ]]; then
            CC=$(awk -v moc="${MOC}" -v foc="${FOC}" 'BEGIN { printf "%.6f\n", moc / foc }')
            CELL_CYCLE_COMPLEXITY+=("${CC}")
        else
            CELL_CYCLE_COMPLEXITY+=("NA")
        fi
    else
        echo "MARKER:G0_TELEMETRY_TSV_ABSENT case=${CELL}"
        CELL_TELEMETRY_PRESENT+=("0")
        CELL_AMG_TELEMETRY_MEAN_ITERS+=("NA")
        CELL_AMG_TELEMETRY_MEAN_OP_COUNT+=("NA")
        CELL_AMG_TELEMETRY_SETUP_CALLS+=("NA")
        CELL_AMG_TOTAL_SETUP_WALL_SEC+=("NA")
        CELL_AMG_TOTAL_SOLVE_WALL_SEC+=("NA")
        CELL_CYCLE_COMPLEXITY+=("NA")
    fi

    # amg_wall_per_step_sec — convention setup_included per PR-0 spike 1.2:
    #   if telemetry TSV present:
    #     = (total_setup_wall + total_solve_wall) / nst
    #   else (PR-A evidence only, no TSV yet):
    #     fall back to wall_total_sec / nst (best available signal; not
    #     strictly setup_included since wall_total includes process
    #     startup + output write, but this is the only signal pre-PR-B
    #     drain hook and is documented as such in the wall_signal column)
    CELL_WALL_CONVENTION+=("setup_included")
    if [[ "${NST:-NA}" != "NA" && "${NST}" != "0" ]]; then
        if [[ "${CELL_AMG_TOTAL_SETUP_WALL_SEC[$((${#CELL_AMG_TOTAL_SETUP_WALL_SEC[@]}-1))]}" != "NA" && "${CELL_AMG_TOTAL_SOLVE_WALL_SEC[$((${#CELL_AMG_TOTAL_SOLVE_WALL_SEC[@]}-1))]}" != "NA" ]]; then
            APS=$(awk -v su="${CELL_AMG_TOTAL_SETUP_WALL_SEC[$((${#CELL_AMG_TOTAL_SETUP_WALL_SEC[@]}-1))]}" \
                       -v so="${CELL_AMG_TOTAL_SOLVE_WALL_SEC[$((${#CELL_AMG_TOTAL_SOLVE_WALL_SEC[@]}-1))]}" \
                       -v nst="${NST}" \
                       'BEGIN { printf "%.6f\n", (su + so) / nst }')
            CELL_WALL_SIGNAL+=("telemetry")
        elif [[ "${WT:-NA}" != "NA" ]]; then
            APS=$(awk -v wt="${WT}" -v nst="${NST}" \
                       'BEGIN { printf "%.6f\n", wt / nst }')
            CELL_WALL_SIGNAL+=("wall_total_proxy")
        else
            APS="NA"
            CELL_WALL_SIGNAL+=("unavailable")
        fi
    else
        APS="NA"
        CELL_WALL_SIGNAL+=("nst_unavailable")
    fi
    CELL_AMG_WALL_PER_STEP_SEC+=("${APS}")

    # Verdict tally
    if [[ "${CELL_VERDICT_CLASS[$((${#CELL_VERDICT_CLASS[@]}-1))]}" == "AMG_OK" ]]; then
        NUM_AMG_OK=$((NUM_AMG_OK + 1))
    else
        NUM_FAIL=$((NUM_FAIL + 1))
    fi
done

# ---------- G0-3 telemetry-real assertion ----------------------------------
# Spec G0-3 PASS = MARKER:AMG_TELEMETRY_REAL present in ≥1 cell AND for each
# AMG_OK cell, cycle_complexity is telemetry-derived (TSV available + value
# non-NA). Cells without TSV are NA — they don't fail the assertion if no
# AMG_OK cell has TSV, but the marker must still be present.
MARKER_PRESENT=0
for i in "${!CELL_NAMES[@]}"; do
    if [[ "${CELL_FIRST_OP_COUNT[$i]}" != "NA" ]]; then
        MARKER_PRESENT=1
        break
    fi
done

# Stricter: if any AMG_OK cell has TSV but CC is NA, that's a parse fail.
# Per spec amendment + IA: the AMG_OK cells without TSV are exempt from
# the cycle_complexity assertion; only AMG_OK cells WITH TSV are required
# to have a telemetry-derived (non-NA) cycle_complexity.
CC_DERIVED_OK=1
for i in "${!CELL_NAMES[@]}"; do
    if [[ "${CELL_VERDICT_CLASS[$i]}" == "AMG_OK" && "${CELL_TELEMETRY_PRESENT[$i]}" == "1" ]]; then
        if [[ "${CELL_CYCLE_COMPLEXITY[$i]}" == "NA" ]]; then
            CC_DERIVED_OK=0
            break
        fi
    fi
done

if [[ "${MARKER_PRESENT}" == "1" && "${CC_DERIVED_OK}" == "1" ]]; then
    G0_3="PASS"
    G0_3_REASON=""
elif [[ "${MARKER_PRESENT}" == "0" ]]; then
    G0_3="FAIL"
    G0_3_REASON="AMG_TELEMETRY_MARKER_ABSENT"
else
    G0_3="FAIL"
    G0_3_REASON="CYCLE_COMPLEXITY_NOT_DERIVED_FROM_TELEMETRY"
fi

# ---------- G0-4 integrated-completes --------------------------------------
# Spec G0-4 PASS = all 4 expected cells {keliya, xinanjiang_upstream,
# heihe_x4, heihe_x16} AMG_OK. The aggregator does NOT enforce that ALL
# 4 expected cells are present (the run_dir may be a partial dogfood) —
# instead it evaluates on the cells actually found. PR-B dogfood may run
# with 1-4 cells. The strict 4-cell requirement is anchored by PR-C
# verdict doc which checks the cell set.
G0_4_FAIL_CELLS=()
EXPECTED_CELLS=(keliya xinanjiang_upstream heihe_x4 heihe_x16)
for ec in "${EXPECTED_CELLS[@]}"; do
    found=0
    found_vc=""
    for i in "${!CELL_NAMES[@]}"; do
        if [[ "${CELL_NAMES[$i]}" == "${ec}" ]]; then
            found=1
            found_vc="${CELL_VERDICT_CLASS[$i]}"
            break
        fi
    done
    if [[ "${found}" == "0" ]]; then
        G0_4_FAIL_CELLS+=("${ec}:MISSING")
    elif [[ "${found_vc}" != "AMG_OK" ]]; then
        G0_4_FAIL_CELLS+=("${ec}:${found_vc}")
    fi
done
if [[ ${#G0_4_FAIL_CELLS[@]} -eq 0 ]]; then
    G0_4="PASS"
    G0_4_REASON=""
else
    G0_4="FAIL"
    G0_4_REASON="$(IFS=,; echo "${G0_4_FAIL_CELLS[*]}")"
fi

# ---------- G0-5 wall-signal -----------------------------------------------
# Spec G0-5 PASS = (amg_improves[heihe_x4]) OR (amg_improves[heihe_x16]).
# amg_improves[case] = amg_wall_per_step_sec[case] < SPGMR_PER_STEP_<CASE>_S.
# Cells with WALL_SIGNAL_UNKNOWN baseline are excluded from the OR.

AMG_X4_IMPROVE="unknown"
AMG_X16_IMPROVE="unknown"
AMG_X4_APS="NA"
AMG_X16_APS="NA"

for i in "${!CELL_NAMES[@]}"; do
    if [[ "${CELL_NAMES[$i]}" == "heihe_x4" ]]; then
        AMG_X4_APS="${CELL_AMG_WALL_PER_STEP_SEC[$i]}"
    elif [[ "${CELL_NAMES[$i]}" == "heihe_x16" ]]; then
        AMG_X16_APS="${CELL_AMG_WALL_PER_STEP_SEC[$i]}"
    fi
done

# heihe_x4 evaluation
if [[ "${SPGMR_HEIHE_X4}" == "WALL_SIGNAL_UNKNOWN" || "${SPGMR_HEIHE_X4}" == "MISSING" ]]; then
    echo "MARKER:G0_5_BASELINE_UNKNOWN case=heihe_x4 reason=spgmr_baseline_wall_signal_unknown"
    AMG_X4_IMPROVE="excluded"
elif [[ "${AMG_X4_APS}" == "NA" ]]; then
    AMG_X4_IMPROVE="excluded"
else
    AMG_X4_IMPROVE=$(awk -v amg="${AMG_X4_APS}" -v spgmr="${SPGMR_HEIHE_X4}" \
        'BEGIN { print (amg < spgmr) ? "true" : "false" }')
    echo "MARKER:G0_5_WALL_PER_STEP case=heihe_x4 amg=${AMG_X4_APS} spgmr=${SPGMR_HEIHE_X4} improves=${AMG_X4_IMPROVE} wall_convention=setup_included"
fi

# heihe_x16 evaluation
if [[ "${SPGMR_HEIHE_X16}" == "WALL_SIGNAL_UNKNOWN" || "${SPGMR_HEIHE_X16}" == "MISSING" ]]; then
    echo "MARKER:G0_5_BASELINE_UNKNOWN case=heihe_x16 reason=spgmr_baseline_wall_signal_unknown"
    AMG_X16_IMPROVE="excluded"
elif [[ "${AMG_X16_APS}" == "NA" ]]; then
    AMG_X16_IMPROVE="excluded"
else
    AMG_X16_IMPROVE=$(awk -v amg="${AMG_X16_APS}" -v spgmr="${SPGMR_HEIHE_X16}" \
        'BEGIN { print (amg < spgmr) ? "true" : "false" }')
    echo "MARKER:G0_5_WALL_PER_STEP case=heihe_x16 amg=${AMG_X16_APS} spgmr=${SPGMR_HEIHE_X16} improves=${AMG_X16_IMPROVE} wall_convention=setup_included"
fi

if [[ "${AMG_X4_IMPROVE}" == "true" || "${AMG_X16_IMPROVE}" == "true" ]]; then
    G0_5="PASS"
    G0_5_REASON=""
elif [[ "${AMG_X4_IMPROVE}" == "excluded" && "${AMG_X16_IMPROVE}" == "excluded" ]]; then
    G0_5="FAIL"
    G0_5_REASON="WALL_SIGNAL_UNKNOWN_BOTH_CELLS"
else
    G0_5="FAIL"
    G0_5_REASON="NEITHER_HEIHE_X4_NOR_HEIHE_X16_IMPROVES"
fi

# G0-5 contingent drift evidence per task 7.5 + IA-4. Triggered when both
# cells have measurable AMG values + baselines but neither improves.
if [[ "${AMG_X4_IMPROVE}" == "false" && "${AMG_X16_IMPROVE}" == "false" ]]; then
    echo "MARKER:G0_BASELINE_DRIFT_CHECK_REQUESTED case=heihe_x4 pinned=${SPGMR_HEIHE_X4} amg_observed=${AMG_X4_APS}"
    echo "MARKER:G0_BASELINE_DRIFT_CHECK_REQUESTED case=heihe_x16 pinned=${SPGMR_HEIHE_X16} amg_observed=${AMG_X16_APS}"
    # Drift evidence file emission. Per task 7.5: aggregator emits markdown
    # documenting drift hypothesis + recommends a PR-0-style hotfix.
    # MUST NOT mutate spgmr_baseline_walls_g0.h.
    DRIFT_DOC="${SCRIPT_DIR}/spgmr_baseline_drift_detected.md"
    {
        echo "# G0-5 baseline drift evidence — both heihe_x4 + heihe_x16 AMG fail to improve"
        echo ""
        echo "Generated by \`tools/p8tune.G0/aggregate_g0_smoke.sh\` (PR-B)."
        echo ""
        echo "## Pinned PR-0 baselines (\`spgmr_baseline_walls_g0.h\`)"
        echo ""
        echo "- \`SPGMR_PER_STEP_HEIHE_X4_S\` = \`${SPGMR_HEIHE_X4}\`"
        echo "- \`SPGMR_PER_STEP_HEIHE_X16_S\` = \`${SPGMR_HEIHE_X16}\`"
        echo ""
        echo "## Aggregator-observed AMG per-step walls"
        echo ""
        echo "- heihe_x4: \`amg_wall_per_step_sec = ${AMG_X4_APS}\` (convention: setup_included)"
        echo "- heihe_x16: \`amg_wall_per_step_sec = ${AMG_X16_APS}\` (convention: setup_included)"
        echo ""
        echo "## Drift hypothesis"
        echo ""
        echo "Both cells report \`amg < spgmr\` = false. Either:"
        echo "1. The pinned SPGMR baselines are stale (drift since PR-0 measurement);"
        echo "   or"
        echo "2. AMG genuinely under-performs SPGMR per-step on these cells."
        echo ""
        echo "## Recommended follow-up"
        echo ""
        echo "File a PR-0-style hotfix on \`openmp-baseline\` re-running"
        echo "\`SHUD_LINSOL=spgmr\` SHORT 90-day on heihe_x4 + heihe_x16, then"
        echo "re-pinning \`SPGMR_PER_STEP_HEIHE_X{4,16}_S\` if the measured values"
        echo "differ materially from the pinned values."
        echo ""
        echo "PR-B does NOT mutate \`spgmr_baseline_walls_g0.h\` (IA-4: PR-0 has"
        echo "exclusive ownership of that header)."
    } > "${DRIFT_DOC}"
    echo "[aggregate-G0] G0-5 drift evidence written: ${DRIFT_DOC}"
fi

# ---------- G0-6 solver-stats-documented -----------------------------------
# Spec G0-6 PASS = for every AMG_OK cell, all 6 CVODE solver-stat fields
# (nfe / nli / nfeLS / ncfn / ncfl / netf) are non-NA.
G0_6_MISSING=()
for i in "${!CELL_NAMES[@]}"; do
    if [[ "${CELL_VERDICT_CLASS[$i]}" != "AMG_OK" ]]; then continue; fi
    for field_pair in "nfe:${CELL_NFE[$i]}" "nli:${CELL_NLI[$i]}" \
                      "nfeLS:${CELL_NFELS[$i]}" "ncfn:${CELL_NCFN[$i]}" \
                      "ncfl:${CELL_NCFL[$i]}" "netf:${CELL_NETF[$i]}"; do
        field="${field_pair%:*}"
        val="${field_pair#*:}"
        if [[ "${val}" == "NA" || -z "${val}" ]]; then
            G0_6_MISSING+=("${CELL_NAMES[$i]}:${field}")
        fi
    done
done
if [[ ${#G0_6_MISSING[@]} -eq 0 ]]; then
    G0_6="PASS"
    G0_6_REASON=""
else
    G0_6="FAIL"
    G0_6_REASON="$(IFS=,; echo "${G0_6_MISSING[*]}")"
fi

# ---------- aggregate.tsv emission -----------------------------------------
AGGREGATE_TSV="${RUN_DIR}/aggregate.tsv"
{
    # Header (20 columns)
    printf "cell\tverdict_class\texit_code\twall_total_sec\tn_cvode_steps"
    printf "\tamg_wall_per_step_sec\tcycle_complexity\toperator_complexity"
    printf "\tcvode_nfe\tcvode_nli\tcvode_nfeLS\tcvode_ncfn\tcvode_ncfl\tcvode_netf"
    printf "\tamg_telemetry_mean_iters\tamg_telemetry_mean_op_count"
    printf "\tamg_total_setup_wall_sec\tamg_total_solve_wall_sec"
    printf "\twall_convention\twall_signal\thypre_release\n"

    for i in "${!CELL_NAMES[@]}"; do
        printf "%s\t%s\t%s\t%s\t%s" \
            "${CELL_NAMES[$i]}" \
            "${CELL_VERDICT_CLASS[$i]}" \
            "${CELL_EXIT_CODE[$i]}" \
            "${CELL_WALL_TOTAL_SEC[$i]}" \
            "${CELL_NST[$i]}"
        printf "\t%s\t%s\t%s" \
            "${CELL_AMG_WALL_PER_STEP_SEC[$i]}" \
            "${CELL_CYCLE_COMPLEXITY[$i]}" \
            "${CELL_OPERATOR_COMPLEXITY[$i]}"
        printf "\t%s\t%s\t%s\t%s\t%s\t%s" \
            "${CELL_NFE[$i]}" \
            "${CELL_NLI[$i]}" \
            "${CELL_NFELS[$i]}" \
            "${CELL_NCFN[$i]}" \
            "${CELL_NCFL[$i]}" \
            "${CELL_NETF[$i]}"
        printf "\t%s\t%s\t%s\t%s" \
            "${CELL_AMG_TELEMETRY_MEAN_ITERS[$i]}" \
            "${CELL_AMG_TELEMETRY_MEAN_OP_COUNT[$i]}" \
            "${CELL_AMG_TOTAL_SETUP_WALL_SEC[$i]}" \
            "${CELL_AMG_TOTAL_SOLVE_WALL_SEC[$i]}"
        printf "\t%s\t%s\t%s\n" \
            "${CELL_WALL_CONVENTION[$i]}" \
            "${CELL_WALL_SIGNAL[$i]}" \
            "${CELL_HYPRE_RELEASE[$i]}"
    done
} > "${AGGREGATE_TSV}"

echo ""
echo "[aggregate-G0] wrote ${AGGREGATE_TSV}"
echo ""

# ---------- G0 verdict block ------------------------------------------------
# Overall = GO iff all 4 gates PASS. Note: G0-1 + G0-2 are out of scope
# for this aggregator (G0-1 is bit-identical SPGMR baseline anchored at
# PR-0; G0-2 is build matrix anchored at PR-0/PR-A CI). PR-C wires those
# into the final verdict block. This aggregator emits the 4 gates it owns.
if [[ "${G0_3}" == "PASS" && "${G0_4}" == "PASS" && "${G0_5}" == "PASS" && "${G0_6}" == "PASS" ]]; then
    G0_OVERALL="GO"
else
    G0_OVERALL="NO-GO"
fi

cat <<EOF
MARKER:G0_VERDICT_BEGIN
G0_3=${G0_3}${G0_3_REASON:+ reason=${G0_3_REASON}}
G0_4=${G0_4}${G0_4_REASON:+ reason=${G0_4_REASON}}
G0_5=${G0_5}${G0_5_REASON:+ reason=${G0_5_REASON}}
G0_6=${G0_6}${G0_6_REASON:+ reason=${G0_6_REASON}}
G0_OVERALL=${G0_OVERALL}
MARKER:G0_VERDICT_END
EOF

exit 0
