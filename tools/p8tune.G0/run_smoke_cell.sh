#!/bin/bash
# tools/p8tune.G0/run_smoke_cell.sh
#
# P8-tune.G0 PR-A — single-cell integrated AMG smoke runner. Invoked
# per cell ∈ {keliya, xinanjiang_upstream, heihe_x4, heihe_x16} from
# smoke_array.sbatch (TASK_ID -> CELL mapping) and from the Mac local
# dry-run dogfood path. Sets SHUD_LINSOL=amg + OMP_NUM_THREADS=1
# (defense-in-depth alongside wrapper-side HYPRE_SetGlobalOptions per
# design.md D10), verifies 90-day cfg.para truncation (IA-7 fix), then
# invokes `./shud_omp <cell>` and emits a per-cell cell_summary KV
# block per spec REQ "G0 evidence schema is per-cell cell_summary KV".
#
# Per CLAUDE.md 项目级铁律 "所有 case ≤90 天截断" + issue #410 task 6.1.
#
# Usage:
#   bash run_smoke_cell.sh <cell_name>
#
# Args:
#   cell_name  ∈ {keliya, xinanjiang_upstream, heihe_x4, heihe_x16}
#
# Layout (per SHUD basin-relative cwd convention):
#   - caller cd's to the basin project dir Basins/<cell_name>/ first
#     (or the script auto-cd's if invoked from SHUD/ root)
#   - cfg.para path: input/<cell_name>/<*>.cfg.para
#   - binary path:   ../../shud_omp (basin-relative) OR $SHUD_BIN absolute
#   - per-cell sidecars:
#       $RUN_DIR/cell-<cell_name>.out                              (stdout)
#       $RUN_DIR/cell-<cell_name>.err                              (stderr)
#
# Exit code:
#   0  PASS — `shud_omp` exited 0 + cell_summary emitted
#   2  cfg.para 90-day precondition failed (IA-7 fix)
#   *  non-zero — propagated from shud_omp (real divergence / crash /
#      OOM / SIGTERM); caller may inspect $RUN_DIR/cell-<cell>.{out,err}
#
# Refs:
#   - issue #410 §In Scope §1 (cell_summary KV + IA-7 90-day check)
#   - spec amg-integrated-smoke-verdict REQ "G0 evidence schema is
#     per-cell cell_summary KV" (27-field schema; this PR-A emits the
#     minimal subset + NA sentinels for un-measured-in-PR-A fields;
#     telemetry aggregation is PR-B scope)
#   - CLAUDE.md 项目级铁律 90-day truncation
#   - CLAUDE.md Slurm 三铁律 (sbatch ancestor)

set -euo pipefail

# ---------- CLI -------------------------------------------------------------
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <cell_name>" >&2
    echo "  cell_name ∈ {keliya, xinanjiang_upstream, heihe_x4, heihe_x16}" >&2
    exit 2
fi

CELL="$1"

case "${CELL}" in
    keliya|xinanjiang_upstream|heihe_x4|heihe_x16) ;;
    *)
        echo "[shud-G0] FATAL: unknown cell_name='${CELL}' " \
             "(expected one of keliya|xinanjiang_upstream|heihe_x4|heihe_x16)" >&2
        exit 2
        ;;
esac

# ---------- Basin folder -> SHUD project name mapping -----------------------
# Per CLAUDE.md §双端实验环境 + docs/case_deployment_map.md §1: SHUD `./shud_omp
# <name>` expects the *project* name, not the *basin folder* name. They match
# for keliya/heihe_x4/heihe_x16; xinanjiang_upstream maps to project name
# `xinanjiang` (its cfg.para lives at input/xinanjiang/xinanjiang.cfg.para).
declare -A SHUD_PROJECT_NAME=(
    ["keliya"]="keliya"
    ["xinanjiang_upstream"]="xinanjiang"
    ["heihe_x4"]="heihe_x4"
    ["heihe_x16"]="heihe_x16"
)
PROJECT_NAME="${SHUD_PROJECT_NAME[${CELL}]:-${CELL}}"

# ---------- Environment -----------------------------------------------------
# AMG path selector (defense-in-depth: also set in smoke_array.sbatch).
# Wrapper-side reads SHUD_LINSOL via parse_linsol_env() in cvode_config.cpp.
export SHUD_LINSOL=amg

# OMP_NUM_THREADS=1 contract per issue #410 task 6.1 + 10.5; wrapper-side
# `HYPRE_SetGlobalOptions(hypre_threads=1)` is the primary enforcement,
# this export is belt-and-suspenders for the host process.
export OMP_NUM_THREADS=1

# ---------- Locate cell artifacts -------------------------------------------
# SHUD CLI invokes `./shud_omp <project>` from inside the basin project dir
# (Basins/<cell>/), where it expects `input/<project>/<project>.cfg.para`.
# Auto-cd to the basin dir if currently sitting one level up at SHUD/. Per
# CLAUDE.md convention, cfg.para is auto-discovered via `find input/`
# (handles basin-folder ≠ SHUD project name like xinanjiang_upstream=xinanjiang).
if [[ -d "Basins/${CELL}" && ! -d "input/${PROJECT_NAME}" ]]; then
    cd "Basins/${CELL}" || {
        echo "[shud-G0] FATAL: cannot cd to Basins/${CELL}" >&2
        exit 2
    }
fi

if [[ ! -d "input/${PROJECT_NAME}" ]]; then
    echo "[shud-G0] FATAL: input/${PROJECT_NAME}/ not found under $(pwd) — case not deployed (cwd should be basin project dir; cell=${CELL} project=${PROJECT_NAME})" >&2
    exit 2
fi

CFG_PARA="$(find "input/${PROJECT_NAME}" -name '*.cfg.para' 2>/dev/null | head -n 1)"
if [[ -z "${CFG_PARA}" || ! -f "${CFG_PARA}" ]]; then
    echo "[shud-G0] FATAL: no *.cfg.para under input/${PROJECT_NAME}/ — case not deployed (cell=${CELL})" >&2
    exit 2
fi

# ---------- IA-7 fix: 90-day cfg.para safety check --------------------------
# Per issue #410 task 6.1 / 10.4 + CLAUDE.md 项目级铁律 "所有 case ≤90 天截断":
# verify cfg.para END - START == 90.
#
# Note: SHUD cfg.para uses `KEY<whitespace>VALUE` (NOT `KEY=VALUE`); the
# awk pattern pinned in issue body §6.1 used `-F'='` which would never
# match this file format. Adapted the implementation to whitespace-
# separated fields while preserving the same `kv["END"] - kv["START"]
# == 90` semantics. Same awk shape is mirrored in precheck_env.sh.
END_START_DELTA=$(awk '
    /^[[:space:]]*(START|END)[[:space:]]/ {
        kv[$1] = $2
    }
    END {
        if (!("START" in kv) || !("END" in kv)) {
            print "NA"
        } else {
            print (kv["END"] - kv["START"]) + 0
        }
    }' "${CFG_PARA}")

if [[ "${END_START_DELTA}" != "90" ]]; then
    echo "[shud-G0] FATAL: 90-day truncation precondition failed for case=${CELL} END-START=${END_START_DELTA}" >&2
    echo "[shud-G0]        cfg.para=${CFG_PARA}" >&2
    echo "[shud-G0]        expected END-START=90 (per CLAUDE.md 项目级铁律)" >&2
    exit 2
fi

# ---------- Banner ----------------------------------------------------------
echo "=== run_smoke_cell.sh cell=${CELL} ==="
echo "date_utc:        $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "hostname:        $(hostname)"
echo "cwd:             $(pwd)"
echo "cell:            ${CELL}"
echo "project_name:    ${PROJECT_NAME}"
echo "cfg_para:        ${CFG_PARA}"
echo "end_start_delta: ${END_START_DELTA} (90-day precondition PASS)"
echo "shud_linsol:     ${SHUD_LINSOL}"
echo "omp_num_threads: ${OMP_NUM_THREADS}"

# ---------- Resolve RUN_DIR for tee output ---------------------------------
# Slurm-managed --output captures stdout/stderr at sbatch level too; the
# per-cell .out / .err are an authoritative second copy for the aggregator
# parser. When invoked from Mac local dogfood (no SLURM_*), RUN_DIR falls
# back to $(pwd)/.g0-smoke-local/ so logs land somewhere predictable.
if [[ -n "${RUN_DIR:-}" ]]; then
    RUN_DIR_RESOLVED="${RUN_DIR}"
else
    RUN_DIR_RESOLVED="$(pwd)/.g0-smoke-local"
fi
mkdir -p "${RUN_DIR_RESOLVED}"
CELL_OUT="${RUN_DIR_RESOLVED}/cell-${CELL}.out"
CELL_ERR="${RUN_DIR_RESOLVED}/cell-${CELL}.err"
echo "run_dir:         ${RUN_DIR_RESOLVED}"
echo "cell_out:        ${CELL_OUT}"
echo "cell_err:        ${CELL_ERR}"
echo ""

# ---------- Binary preflight -----------------------------------------------
# Resolve shud_omp: explicit $SHUD_BIN > basin-relative ../../shud_omp >
# ./shud_omp (when caller already placed binary at cwd, e.g. tests).
if [[ -n "${SHUD_BIN:-}" && -x "${SHUD_BIN}" ]]; then
    SHUD_BIN_RESOLVED="${SHUD_BIN}"
elif [[ -x "../../shud_omp" ]]; then
    SHUD_BIN_RESOLVED="../../shud_omp"
elif [[ -x "./shud_omp" ]]; then
    SHUD_BIN_RESOLVED="./shud_omp"
else
    echo "[shud-G0] FATAL: shud_omp not found (checked \$SHUD_BIN, ../../shud_omp, ./shud_omp) under $(pwd)" >&2
    exit 2
fi
echo "shud_bin:        ${SHUD_BIN_RESOLVED}"
echo ""

# ---------- Capture wall + invoke shud_omp ----------------------------------
START_WALL=$(date +%s)
# Tee stdout and stderr to per-cell files while preserving exit code via
# pipefail. Use process substitution so both streams are recorded plus
# also surfaced to the Slurm-managed --output / --error capture.
#
# NOTE: shud_omp is invoked with the SHUD *project name* (PROJECT_NAME), not
# the basin folder name (CELL). They differ for xinanjiang_upstream
# (basin=xinanjiang_upstream → project=xinanjiang). The SHUD CLI hands its
# argument to sprintf "input/%s" in CommandIn.cpp, so passing CELL there
# would resolve to a non-existent input/xinanjiang_upstream/ directory.
set +e
"${SHUD_BIN_RESOLVED}" "${PROJECT_NAME}" \
    > >(tee "${CELL_OUT}") \
    2> >(tee "${CELL_ERR}" >&2)
RC=$?
# Block until both background tee processes finish flushing their output to
# the sidecar files. Without this wait, the subsequent grep extractors at
# L210+ may run before tee drains the FIFO and miss real markers/fields.
wait
set -e
END_WALL=$(date +%s)
WALL_TOTAL=$((END_WALL - START_WALL))

# ---------- Extract minimal cell_summary fields -----------------------------
# Per spec "Cell with verdict_class != AMG_OK fills NA sentinels": PR-A
# emits the minimal subset of the 27-field schema (cell_name, verdict_class,
# exit_code, wall_total_sec, nst, linsol, omp_threads, hypre_release). The
# rest are NA sentinels — telemetry aggregation lands in PR-B.
#
# All extraction steps below MUST tolerate grep/awk no-match (rc=1) without
# aborting the script; `set +e` covers the whole extract + emit block so
# cell_summary always reaches stdout even when none of the regexes match.
set +e

# nst (CVODE step count) — parse from PrintFinalStats output. SHUD prints
# `nst     = NNN` (no leading "NumSteps" label); also accept the long form
# for backward compat.
NST=$(grep -E '^[[:space:]]*(nst|NumSteps)[[:space:]]*=' "${CELL_OUT}" 2>/dev/null | head -n 1 | awk -F'=' '{gsub(/[[:space:]]/,"",$2); print $2}')
if [[ -z "${NST}" ]]; then
    NST="NA"
fi

# hypre_release — wrapper prints `[shud-amg] hypre_release=<X.Y.Z>` (or
# similar provenance line); fall back to NA.
HYPRE_RELEASE=$(grep -oE 'hypre[_ ]release=([0-9]+\.[0-9]+\.[0-9]+)' "${CELL_OUT}" 2>/dev/null \
                | head -n 1 | awk -F= '{print $2}')
if [[ -z "${HYPRE_RELEASE}" ]]; then
    HYPRE_RELEASE="NA"
fi

# verdict_class — derive from exit code + marker presence in stdout.
# PR-A semantics (IA-3 fix per issue task 6.6):
#   exit 0 + no divergence MARKER → AMG_OK
#   any divergence MARKER         → corresponding class (NOT AMG_OK)
#   exit non-zero w/o marker      → leave as AMG_OK + RC != 0 will be
#                                    flagged by aggregator (cell_summary
#                                    must still emit a class token)
VERDICT_CLASS="AMG_OK"
if grep -q '^MARKER:AMG_SETUP_DIVERGE_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_SETUP_DIVERGE"
elif grep -q '^MARKER:AMG_SOLVE_DIVERGE_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_SOLVE_DIVERGE"
elif grep -q '^MARKER:AMG_OOM_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_OOM"
elif grep -q '^MARKER:AMG_WALL_OVERFLOW_DETECTED' "${CELL_OUT}" 2>/dev/null; then
    VERDICT_CLASS="AMG_WALL_OVERFLOW"
fi
set -e

# ---------- Emit cell_summary KV block --------------------------------------
# Per spec REQ-G0-evidence-schema: BEGIN/END delimiters + 27 fields total
# (8 KV content lines). PR-A emits the minimal subset; fields not measured
# in PR-A (telemetry, peak_rss, AMG-internal counters, CVODE solver stats)
# are NA sentinels. PR-B aggregator fills the remaining fields by parsing
# the wrapper's ring-buffer telemetry emissions.
{
    echo ""
    echo "CELL_SUMMARY_BEGIN"
    echo "case=${CELL} interp_type=6 coarsen_type=8 NumY=NA nnz_A=NA"
    echo "setup_wall_sec=NA apply_wall_sec=NA peak_rss_bytes=NA"
    echo "cycle_complexity=NA operator_complexity=NA"
    echo "verdict_class=${VERDICT_CLASS}"
    echo "hypre_version=${HYPRE_RELEASE} colpack_version=absent shud_pin=NA"
    echo "cvode_nfe=NA cvode_nli=NA cvode_nfeLS=NA cvode_ncfn=NA cvode_ncfl=NA cvode_netf=NA"
    echo "amg_telemetry_mean_iters=NA amg_telemetry_mean_op_count=NA amg_telemetry_setup_calls=NA amg_total_setup_wall_sec=NA amg_total_solve_wall_sec=NA"
    echo "n_cvode_steps=${NST} amg_wall_per_step_sec=NA"
    echo "CELL_SUMMARY_END"
    echo ""
    echo "[shud-G0] cell=${CELL} exit_code=${RC} wall_total_sec=${WALL_TOTAL} verdict_class=${VERDICT_CLASS}"
} | tee -a "${CELL_OUT}"

exit ${RC}
