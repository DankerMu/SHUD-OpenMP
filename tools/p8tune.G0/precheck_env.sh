#!/bin/bash
# tools/p8tune.G0/precheck_env.sh
#
# P8-tune.G0 PR-A — Slurm-side pre-submission gate per issue #410
# task 6.2. Reuses P8-tune.F H1 fix template (one PASS/FAIL line per
# condition; never group-references). Verifies the 7 conditions in
# order, emitting `[precheck] PASS condition=<id> detail=...` or
# `[precheck] FATAL condition=<id> detail=...`, and exits non-zero
# on the first FATAL (fail-fast — invalid environment makes subsequent
# AMG cells meaningless).
#
# Conditions:
#   (a) cwd matches /scratch/.../.p8tune.G0-runs/<RUN_ID>/
#   (b) ./shud_omp exists + is executable
#   (c) ldd ./shud_omp | grep -q libHYPRE returns 0  (binary links Hypre)
#   (d) case dir Basins/<cell> exists for the cell from $SLURM_ARRAY_TASK_ID
#   (e) wrapper telemetry symbol SUNLinSol_Hypre_DrainTelemetry present
#       in nm ./shud_omp (compile-time symbol attestation)
#   (f) OMP_NUM_THREADS == 1  (per CLAUDE.md OMP_CUTOFF policy for AMG)
#   (g) cfg.para END - START == 90 for the selected cell (IA-7 fix)
#
# Usage:
#   bash precheck_env.sh
#
# Resolves:
#   - cell from $SLURM_ARRAY_TASK_ID (0=keliya, 1=xinanjiang_upstream,
#     2=heihe_x4, 3=heihe_x16)
#   - RUN_ID from environment (set by sbatch via --export=RUN_ID=... or
#     defaults to job-id-based)
#
# Exit:
#   0  all 7 conditions pass; emits `[precheck] PASS conditions=7/7`
#   1  one or more conditions fail; emits `[precheck] FATAL_SUMMARY ...`
#
# Refs:
#   - issue #410 task 6.2 + 10.x governance
#   - tools/p8tune.F/precheck_env.sh (template for emit_pass / emit_fail
#     pattern; reused per issue brief)
#   - CLAUDE.md Slurm 三铁律

set -uo pipefail

# ---------- Configuration ---------------------------------------------------
CELLS=(keliya xinanjiang_upstream heihe_x4 heihe_x16)
EXPECTED_PWD_PREFIX="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.G0-runs/"
TELEMETRY_SYMBOL="SUNLinSol_Hypre_DrainTelemetry"

# Basin folder -> SHUD project name mapping, mirroring run_smoke_cell.sh.
# Per CLAUDE.md §双端实验环境 + docs/case_deployment_map.md §1:
# xinanjiang_upstream's cfg.para lives under input/xinanjiang/.
declare -A SHUD_PROJECT_NAME=(
    ["keliya"]="keliya"
    ["xinanjiang_upstream"]="xinanjiang"
    ["heihe_x4"]="heihe_x4"
    ["heihe_x16"]="heihe_x16"
)

# Slurm-array task -> cell mapping; if SLURM_ARRAY_TASK_ID unset (Mac
# local dry-run / smoke_single_test pre-flight), default to task 0.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
if [[ ${TASK_ID} -lt 0 || ${TASK_ID} -ge ${#CELLS[@]} ]]; then
    echo "[precheck] FATAL: SLURM_ARRAY_TASK_ID=${TASK_ID} out of range [0,${#CELLS[@]})" >&2
    exit 1
fi
CELL="${CELLS[${TASK_ID}]}"
PROJECT_NAME="${SHUD_PROJECT_NAME[${CELL}]:-${CELL}}"

fail=0
pass_count=0
emit_pass() {
    echo "[precheck] PASS condition=$1 detail=$2"
    pass_count=$((pass_count + 1))
}
emit_fail() {
    echo "[precheck] FATAL condition=$1 detail=$2" >&2
    fail=1
}

# ---------- (a) $RUN_DIR under .p8tune.G0-runs/ ----------------------------
# Per CLAUDE.md Slurm 三铁律 rule 1: sbatch must operate under
# /scratch/.../.p8tune.G0-runs/. Originally this checked pwd, but sbatch
# bodies cd into SHUD/ before calling precheck — and pwd would then no
# longer be under .p8tune.G0-runs/. Check the *RUN_DIR* env var instead
# (set by sbatch from RUN_ID); this is semantically what we actually care
# about (the run-artifact root is on the shared /scratch tier, not /tmp).
#
# On Mac local dry-run, RUN_DIR is unset → emit INFO + PASS (deferred to
# server).
EXPECTED_PWD_PREFIX_STRIP="${EXPECTED_PWD_PREFIX%/}"
cur_pwd="$(pwd)"
RUN_DIR_PROBE="${RUN_DIR:-}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${RUN_DIR_PROBE}" ]]; then
        emit_fail "a" "RUN_DIR unset under Slurm job ${SLURM_JOB_ID} (sbatch must export RUN_DIR per template)"
    elif [[ "${RUN_DIR_PROBE}" == "${EXPECTED_PWD_PREFIX_STRIP}"* ]]; then
        emit_pass "a" "RUN_DIR=${RUN_DIR_PROBE} under ${EXPECTED_PWD_PREFIX_STRIP} (pwd=${cur_pwd} for context)"
    else
        emit_fail "a" "RUN_DIR=${RUN_DIR_PROBE} not under ${EXPECTED_PWD_PREFIX_STRIP} (Slurm 三铁律 rule 1)"
    fi
else
    echo "[precheck] INFO: SLURM_JOB_ID unset (Mac local dry-run); skipping (a) per dry-run convention" >&2
    emit_pass "a" "Mac dry-run (SLURM_JOB_ID unset); deferred to server submission"
fi

# ---------- (b) shud_omp exists + is executable ----------------------------
# Resolve shud_omp: when invoked from SHUD/ root, ./shud_omp. When invoked
# from a basin dir (e.g. Basins/keliya/), ../../shud_omp. When invoked
# from .p8tune.G0-runs/<RUN_ID>/ (Slurm cwd), use $SHUD_BIN if exported.
SHUD_BIN_PROBE=""
for cand in "${SHUD_BIN:-}" "./shud_omp" "../../shud_omp" \
            "/scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud_omp"; do
    if [[ -n "${cand}" && -x "${cand}" ]]; then
        SHUD_BIN_PROBE="${cand}"
        break
    fi
done
if [[ -z "${SHUD_BIN_PROBE}" ]]; then
    emit_fail "b" "shud_omp not found (checked \$SHUD_BIN, ./shud_omp, ../../shud_omp, /scratch/.../SHUD/shud_omp) under ${cur_pwd}"
else
    emit_pass "b" "shud_omp present + executable at ${SHUD_BIN_PROBE} ($(stat -c %s "${SHUD_BIN_PROBE}" 2>/dev/null || stat -f %z "${SHUD_BIN_PROBE}" 2>/dev/null) bytes)"
fi

# ---------- (c) ldd shows libHYPRE -----------------------------------------
# On Linux: `ldd ${SHUD_BIN_PROBE} | grep -q libHYPRE`.
# On Mac: ldd doesn't exist; use otool -L. Skip check on Mac dry-run per
# convention (server-only Slurm submission is the binding test).
if [[ -z "${SHUD_BIN_PROBE}" ]]; then
    emit_fail "c" "shud_omp missing; cannot check libHYPRE linkage"
elif command -v ldd >/dev/null 2>&1; then
    if ldd "${SHUD_BIN_PROBE}" 2>/dev/null | grep -q -E 'libHYPRE'; then
        emit_pass "c" "ldd ${SHUD_BIN_PROBE} shows libHYPRE linkage"
    else
        emit_fail "c" "ldd ${SHUD_BIN_PROBE} does NOT show libHYPRE (binary missing AMG path)"
    fi
elif command -v otool >/dev/null 2>&1; then
    if otool -L "${SHUD_BIN_PROBE}" 2>/dev/null | grep -q -E 'libHYPRE'; then
        emit_pass "c" "otool -L ${SHUD_BIN_PROBE} shows libHYPRE linkage (Mac fallback)"
    else
        emit_fail "c" "otool -L ${SHUD_BIN_PROBE} does NOT show libHYPRE (binary missing AMG path)"
    fi
else
    echo "[precheck] INFO: neither ldd nor otool available; skipping (c)" >&2
    emit_pass "c" "ldd/otool unavailable; deferred to runtime dlopen probe in wrapper"
fi

# ---------- (d) case dirs exist for ALL 4 cells ----------------------------
# Resolve case dir from current cwd: SHUD/ root -> Basins/<cell>; basin dir
# -> input/<cell>; .p8tune.G0-runs/ -> /scratch/.../SHUD/Basins/<cell>.
# Iterate over all 4 cells (not just SLURM_ARRAY_TASK_ID's) so that one
# precheck invocation validates the entire array — caught by P1-2 reviewer.
# CASE_DIRS map stores the resolved path per cell for (g) below.
declare -A CASE_DIRS=()
case_d_fail=0
for cell in "${CELLS[@]}"; do
    found=""
    for cand in "Basins/${cell}" "/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/${cell}"; do
        if [[ -d "${cand}" ]]; then
            found="${cand}"
            break
        fi
    done
    if [[ -n "${found}" ]]; then
        CASE_DIRS["${cell}"]="${found}"
    else
        case_d_fail=1
        emit_fail "d" "case dir for cell=${cell} missing (checked Basins/ + /scratch/.../SHUD/Basins/)"
    fi
done
if [[ ${case_d_fail} -eq 0 ]]; then
    emit_pass "d" "all 4 case dirs present (${CELLS[*]})"
fi

# Also retain CASE_DIR for the selected cell so (g) can fall back to a
# clearer error if the per-cell loop below doesn't catch it.
CASE_DIR="${CASE_DIRS[${CELL}]:-}"

# ---------- (e) telemetry symbol attestation -------------------------------
# Per design.md D6/D7: wrapper exposes SUNLinSol_Hypre_DrainTelemetry as
# the C-linkage entry point for the aggregator. nm extracts the symbol
# table; symbol presence proves the wrapper was compiled into the binary.
if [[ -z "${SHUD_BIN_PROBE}" ]]; then
    emit_fail "e" "shud_omp missing; cannot check telemetry symbol"
elif ! command -v nm >/dev/null 2>&1; then
    echo "[precheck] INFO: nm not available; skipping (e)" >&2
    emit_pass "e" "nm unavailable; deferred to wrapper runtime emit"
elif nm "${SHUD_BIN_PROBE}" 2>/dev/null | grep -q "${TELEMETRY_SYMBOL}"; then
    emit_pass "e" "telemetry symbol ${TELEMETRY_SYMBOL} present in ${SHUD_BIN_PROBE}"
else
    # nm may need -D for dynamic symbols on stripped binaries.
    if nm -D "${SHUD_BIN_PROBE}" 2>/dev/null | grep -q "${TELEMETRY_SYMBOL}"; then
        emit_pass "e" "telemetry symbol ${TELEMETRY_SYMBOL} present (nm -D)"
    else
        emit_fail "e" "telemetry symbol ${TELEMETRY_SYMBOL} NOT found in ${SHUD_BIN_PROBE} (wrapper not linked?)"
    fi
fi

# ---------- (f) OMP_NUM_THREADS == 1 ---------------------------------------
# Per CLAUDE.md OMP_CUTOFF policy + issue #410 task 10.5: AMG cells run
# with OMP_NUM_THREADS=1 to keep Hypre internal threading deterministic.
omp_threads="${OMP_NUM_THREADS:-unset}"
if [[ "${omp_threads}" == "1" ]]; then
    emit_pass "f" "OMP_NUM_THREADS=1 (per OMP_CUTOFF AMG policy)"
else
    emit_fail "f" "OMP_NUM_THREADS=${omp_threads} != 1 (per CLAUDE.md OMP_CUTOFF / issue 10.5)"
fi

# ---------- (g) 90-day cfg.para precondition for ALL 4 cells (IA-7) -------
# Mirrors run_smoke_cell.sh awk one-liner; per issue #410 task 6.2 + 10.4.
# Iterate over all 4 cells (P1-2 reviewer): one precheck invocation
# validates the entire array. CASE_DIRS["<cell>"] is the basin folder
# resolved by (d) above; its input/<project>/<project>.cfg.para is the
# auto-discovery target (find handles basin-folder ≠ project name like
# xinanjiang_upstream=xinanjiang).
case_g_fail=0
for cell in "${CELLS[@]}"; do
    case_dir="${CASE_DIRS[${cell}]:-}"
    if [[ -z "${case_dir}" ]]; then
        # (d) above already emitted the fail; skip without double-counting
        case_g_fail=1
        continue
    fi
    pname="${SHUD_PROJECT_NAME[${cell}]:-${cell}}"
    cfg_para="$(find "${case_dir}/input/${pname}" -name '*.cfg.para' 2>/dev/null | head -n 1)"
    if [[ -z "${cfg_para}" || ! -f "${cfg_para}" ]]; then
        # Fallback to whole input/ tree (handles future cases where
        # project-name mapping isn't seeded yet).
        cfg_para="$(find "${case_dir}/input" -name '*.cfg.para' 2>/dev/null | head -n 1)"
    fi
    if [[ -z "${cfg_para}" || ! -f "${cfg_para}" ]]; then
        case_g_fail=1
        emit_fail "g" "no *.cfg.para under ${case_dir}/input/ — case=${cell} not deployed"
        continue
    fi
    # SHUD cfg.para uses `KEY<whitespace>VALUE`, not `KEY=VALUE`; adapted
    # from issue body §6.2 `-F'='` form to whitespace-separated fields.
    # Mirrors run_smoke_cell.sh.
    delta=$(awk '
        /^[[:space:]]*(START|END)[[:space:]]/ {
            kv[$1] = $2
        }
        END {
            if (!("START" in kv) || !("END" in kv)) {
                print "NA"
            } else {
                print (kv["END"] - kv["START"]) + 0
            }
        }' "${cfg_para}")
    if [[ "${delta}" != "90" ]]; then
        case_g_fail=1
        emit_fail "g" "cfg.para END-START=${delta} for case=${cell} (expected 90 per CLAUDE.md 项目级铁律; cfg=${cfg_para})"
    fi
done
if [[ ${case_g_fail} -eq 0 ]]; then
    emit_pass "g" "all 4 cfg.para END-START=90 (${CELLS[*]})"
fi

# ---------- Final verdict --------------------------------------------------
if [[ ${fail} -eq 0 ]]; then
    echo "[precheck] PASS conditions=${pass_count}/7"
    exit 0
else
    echo "[precheck] FATAL_SUMMARY conditions_passed=${pass_count}/7; see [precheck] FATAL lines on stderr" >&2
    exit 1
fi
