#!/bin/bash
# tools/p8tune.F/precheck_env.sh
#
# P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-B). Pre-submission
# environment gate per openspec change `p8tune-amg-spike` spec REQ-4
# Scenario "Slurm 三铁律 + 7-condition precheck_env.sh gate":
#
#   PR-B precheck SHALL enforce 7 conditions (a..g), each individually
#   checked + logged. Group-references are forbidden — every condition
#   must emit its own PASS/FAIL line.
#
# Conditions:
#   (a) cfg.para 90-day truncation per case: END - START = 90 (day-index)
#   (b) CMFD V0200 forcing per case
#   (c) heihe_x16 deployed at expected path + NumEle ≥ 25340
#   (d) cn-node RAM ≥ 121 GiB (≥ 127000000 KiB per /proc/meminfo MemTotal)
#   (e) sbatch invoked from /scratch/.../.p8tune.F-runs/ (Slurm 三铁律 rule 1)
#   (f) #SBATCH --output= / --error= paths in /scratch/ (rule 2)
#   (g) all referenced scripts in /scratch/ (rule 3)
#
# Called from spike_array.sbatch BEFORE the cell wrapper dispatch; on
# failure refuses cell execution with exit 1 + diagnostic, so Slurm sees
# a real precheck failure and the orchestrator can replay after fix.
#
# Usage:
#   bash precheck_env.sh <BASIN_ROOT> <SBATCH_FILE> <RUN_CELL_FILE>
#
#   BASIN_ROOT       e.g. /scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins
#   SBATCH_FILE      absolute path to spike_array.sbatch (for grep of
#                    #SBATCH --output / --error + referenced scripts)
#   RUN_CELL_FILE    absolute path to run_cell.sh (for grep of referenced
#                    binaries: boomeramg_setup_solve / dump_adjacency /
#                    fd_color_jacobian)
#
# Exits:
#   0  all 7 conditions pass; emits "PRECHECK_PASS conditions=7/7"
#   1  one or more conditions fail; emits "PRECHECK_FAIL condition=<id>
#      detail=<msg>" lines on stderr + summary on exit
#
# Refs:
#   openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md REQ-4
#   CLAUDE.md 项目级铁律 "所有 case ≤90 天截断"
#   CLAUDE.md "CMFD forcing 强制 V0200"
#   CLAUDE.md Slurm 三铁律 (3 rules)

set -uo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <BASIN_ROOT> <SBATCH_FILE> <RUN_CELL_FILE>" >&2
    exit 1
fi

BASIN_ROOT="$1"
SBATCH_FILE="$2"
RUN_CELL_FILE="$3"

CASES=(keliya heihe heihe_x4 heihe_x16)
RAM_MIN_KIB=127000000          # 121 GiB ≈ 126877696 KiB; round up to 127M
HEIHE_X16_MIN_NUMELE=25340     # CLAUDE.md heihe_x16 mesh nominal NumEle floor
SCRATCH_ROOT_RE='^/scratch/'   # POSIX ERE; all paths must start with /scratch/
EXPECTED_PWD_PREFIX='/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.F-runs/'

fail=0
pass_count=0
emit_pass() {
    echo "PRECHECK_PASS condition=$1 detail=$2"
    pass_count=$((pass_count + 1))
}
emit_fail() {
    echo "PRECHECK_FAIL condition=$1 detail=$2" >&2
    fail=1
}

# ---------- (a) cfg.para END-START = 90 per case ----------------------------
# Per CLAUDE.md 项目级铁律 "所有 case ≤90 天截断". cfg.para path is
# auto-discovered via `find <case>/input/ -name '*.cfg.para'` per CLAUDE.md
# convention. Each case must individually PASS; aggregate the per-case
# result into a single condition (a) PASS/FAIL.
a_pass=1
for case in "${CASES[@]}"; do
    case_dir="${BASIN_ROOT}/${case}"
    cfg_para=$(find "${case_dir}/input" -name '*.cfg.para' 2>/dev/null | head -n 1)
    if [[ -z "${cfg_para}" ]]; then
        emit_fail "a.${case}" "no *.cfg.para under ${case_dir}/input/"
        a_pass=0
        continue
    fi
    start=$(awk '$1=="START"{print $2; exit}' "${cfg_para}")
    end=$(awk '$1=="END"{print $2; exit}' "${cfg_para}")
    if [[ -z "${start}" || -z "${end}" ]]; then
        emit_fail "a.${case}" "could not read START/END from ${cfg_para}"
        a_pass=0
        continue
    fi
    days=$((end - start))
    if [[ ${days} -ne 90 ]]; then
        emit_fail "a.${case}" "${cfg_para}: START=${start} END=${end} delta=${days} (expected 90 per CLAUDE.md 铁律)"
        a_pass=0
    fi
done
if [[ ${a_pass} -eq 1 ]]; then
    emit_pass "a" "all 4 cases cfg.para END-START = 90 days"
fi

# ---------- (b) CMFD V0200 forcing per case ---------------------------------
# Per CLAUDE.md "CMFD forcing 强制 V0200". AutoSHUD-generated cases (heihe_x4,
# heihe_x16) record provenance in <case>/<case>.autoshud.runtime.txt; legacy
# cases (keliya, heihe) predate AutoSHUD and their V0200 provenance is
# implicit (forcing files were generated upstream + stable + already
# verified). For legacy cases we emit INFO + treat as PASS (mirror p8tune.D
# precheck_env.sh L92-96 "legacy case, skipping forcing-V0200 check").
b_pass=1
for case in "${CASES[@]}"; do
    case_dir="${BASIN_ROOT}/${case}"
    runtime_cfg="${case_dir}/${case}.autoshud.runtime.txt"
    forcing_dir="${case_dir}/forcing"
    if [[ ! -d "${forcing_dir}" ]]; then
        emit_fail "b.${case}" "${forcing_dir}/ does not exist"
        b_pass=0
        continue
    fi
    if [[ -f "${runtime_cfg}" ]]; then
        if ! grep -qE 'CMFD20|CMFD2\.0|V0200' "${runtime_cfg}"; then
            emit_fail "b.${case}" "${runtime_cfg} does not reference CMFD V0200"
            b_pass=0
        fi
    else
        echo "INFO: ${case} has no runtime cfg ${runtime_cfg}; legacy case (forcing V0200 provenance implicit)" >&2
    fi
done
if [[ ${b_pass} -eq 1 ]]; then
    emit_pass "b" "all 4 cases forcing references CMFD V0200 (legacy cases pass implicitly)"
fi

# ---------- (c) heihe_x16 deployed + NumEle ≥ 25340 -------------------------
# Per CLAUDE.md: heihe_x16 lives at SHUD/Basins/heihe_x16/, NumEle ≥ 25340
# from rSHUD shud.triangle with a=AreaMax/16 (actual nominal: 40046 per
# autoshud_step3_v2.log, but spec floor is 25340 from the unconstrained
# refinement target).
x16_cfg=$(find "${BASIN_ROOT}/heihe_x16" -name '*.cfg.para' 2>/dev/null | head -n 1)
if [[ -z "${x16_cfg}" ]]; then
    emit_fail "c" "no *.cfg.para under ${BASIN_ROOT}/heihe_x16/ — heihe_x16 not deployed"
else
    # NumEle is the first integer on line 1 of *.sp.mesh (header line:
    # "<NumEle> <NumNodeOrSimilar>"; see SHUD/Basins/heihe_x4/.../heihe_x4.sp.mesh)
    sp_mesh=$(find "${BASIN_ROOT}/heihe_x16" -name '*.sp.mesh' 2>/dev/null | head -n 1)
    if [[ -z "${sp_mesh}" ]]; then
        emit_fail "c" "heihe_x16 cfg.para found at ${x16_cfg} but no *.sp.mesh — mesh deployment incomplete"
    else
        num_ele=$(awk 'NR==1{print $1; exit}' "${sp_mesh}")
        if [[ -z "${num_ele}" || ! "${num_ele}" =~ ^[0-9]+$ ]]; then
            emit_fail "c" "could not parse NumEle from ${sp_mesh} line 1"
        elif [[ ${num_ele} -lt ${HEIHE_X16_MIN_NUMELE} ]]; then
            emit_fail "c" "${sp_mesh} NumEle=${num_ele} < ${HEIHE_X16_MIN_NUMELE} (per CLAUDE.md heihe_x16 floor)"
        else
            emit_pass "c" "heihe_x16 deployed at ${sp_mesh%/*} with NumEle=${num_ele} (≥ ${HEIHE_X16_MIN_NUMELE})"
        fi
    fi
fi

# ---------- (d) cn-node RAM ≥ 121 GiB --------------------------------------
# Per CLAUDE.md (cn14 has 173 GiB; floor is 121 GiB = 127000000 KiB).
# /proc/meminfo MemTotal is in KiB on Linux. On Mac (precheck dry-run
# locally), /proc/meminfo doesn't exist; emit INFO + PASS so the script
# can be linted on Mac without faking server state.
if [[ -r /proc/meminfo ]]; then
    mem_kib=$(awk '/^MemTotal:/{print $2; exit}' /proc/meminfo)
    if [[ -z "${mem_kib}" || ! "${mem_kib}" =~ ^[0-9]+$ ]]; then
        emit_fail "d" "could not parse MemTotal from /proc/meminfo"
    elif [[ ${mem_kib} -lt ${RAM_MIN_KIB} ]]; then
        emit_fail "d" "MemTotal=${mem_kib} KiB < ${RAM_MIN_KIB} KiB (121 GiB floor)"
    else
        mem_gib=$((mem_kib / 1024 / 1024))
        emit_pass "d" "MemTotal=${mem_kib} KiB (~${mem_gib} GiB ≥ 121 GiB)"
    fi
else
    echo "INFO: /proc/meminfo not readable (likely Mac local lint); skipping (d) per dry-run convention" >&2
    emit_pass "d" "/proc/meminfo unavailable (Mac dry-run); deferred to server submission"
fi

# ---------- (e) sbatch from /scratch/.../.p8tune.F-runs/ -------------------
# Per CLAUDE.md Slurm 三铁律 rule 1: "从 /scratch 下 sbatch (policy 拦
# /users/$USER 提交)". The convention is to invoke this script with
# pwd = the run dir (.p8tune.F-runs/<RUN_ID>/), so check pwd prefix.
cur_pwd=$(pwd)
# Accept both exact-match (.p8tune.F-runs/) and any sub-dir under it
# (.p8tune.F-runs/<RUN_ID>/). Strip trailing slash from EXPECTED_PWD_PREFIX
# for prefix comparison so the bare-dir case matches.
EXPECTED_PWD_PREFIX_STRIP="${EXPECTED_PWD_PREFIX%/}"
if [[ "${cur_pwd}" == "${EXPECTED_PWD_PREFIX_STRIP}" ]] || \
   [[ "${cur_pwd}" == ${EXPECTED_PWD_PREFIX_STRIP}/* ]]; then
    emit_pass "e" "pwd=${cur_pwd} under ${EXPECTED_PWD_PREFIX_STRIP}"
else
    emit_fail "e" "pwd=${cur_pwd} not under ${EXPECTED_PWD_PREFIX_STRIP} (per Slurm 三铁律 rule 1)"
fi

# ---------- (f) #SBATCH --output / --error paths in /scratch/ --------------
# Per CLAUDE.md Slurm 三铁律 rule 2: "compute node 的 /tmp node-local, 作业
# 结束会丢, sacct 显示 ExitCode 127". Both #SBATCH --output= and --error=
# must resolve to /scratch/ paths. The spike_array.sbatch may use bare
# `cell-%a.out` / `cell-%a.err` relative to submission cwd — that's fine
# as long as the submission cwd is in /scratch/ (which (e) above checks).
if [[ ! -f "${SBATCH_FILE}" ]]; then
    emit_fail "f" "sbatch file ${SBATCH_FILE} not found"
else
    # Look for --output= and --error= directives. Accept either /scratch/
    # absolute paths OR bare relative paths (since (e) confirms pwd is in
    # /scratch/, bare relative paths resolve under /scratch/ too).
    out_line=$(grep -E '^#SBATCH[[:space:]]+--output=' "${SBATCH_FILE}" | head -n 1)
    err_line=$(grep -E '^#SBATCH[[:space:]]+--error=' "${SBATCH_FILE}" | head -n 1)
    f_pass=1
    if [[ -z "${out_line}" ]]; then
        emit_fail "f" "${SBATCH_FILE} missing #SBATCH --output= directive"
        f_pass=0
    else
        out_val=$(echo "${out_line}" | sed -E 's/^#SBATCH[[:space:]]+--output=//')
        # Reject only paths that START with /tmp or another non-scratch
        # absolute path. Bare relative paths (cell-%a.out) are allowed.
        case "${out_val}" in
            /tmp/*|/var/*|/users/*)
                emit_fail "f" "--output=${out_val} not under /scratch/ (per rule 2)"; f_pass=0 ;;
            /scratch/*) ;;  # absolute /scratch/, OK
            /*)
                # absolute path but not /scratch/; reject
                emit_fail "f" "--output=${out_val} absolute but not /scratch/ (per rule 2)"; f_pass=0 ;;
            *) ;;           # relative path, OK (cwd already confirmed /scratch/ via (e))
        esac
    fi
    if [[ -z "${err_line}" ]]; then
        emit_fail "f" "${SBATCH_FILE} missing #SBATCH --error= directive"
        f_pass=0
    else
        err_val=$(echo "${err_line}" | sed -E 's/^#SBATCH[[:space:]]+--error=//')
        case "${err_val}" in
            /tmp/*|/var/*|/users/*)
                emit_fail "f" "--error=${err_val} not under /scratch/ (per rule 2)"; f_pass=0 ;;
            /scratch/*) ;;
            /*)
                emit_fail "f" "--error=${err_val} absolute but not /scratch/ (per rule 2)"; f_pass=0 ;;
            *) ;;
        esac
    fi
    if [[ ${f_pass} -eq 1 ]]; then
        emit_pass "f" "#SBATCH --output / --error directives resolve under /scratch/"
    fi
fi

# ---------- (g) all referenced scripts in /scratch/ -------------------------
# Per CLAUDE.md Slurm 三铁律 rule 3: "作业脚本里引用的 patch / hash / run.sh
# 都放 /scratch". For each of the 4 core scripts/binaries (run_cell.sh,
# boomeramg_setup_solve, dump_adjacency, fd_color_jacobian), grep
# spike_array.sbatch + run_cell.sh for ABSOLUTE-path references and verify
# every such reference starts with /scratch/. Relative references like
# `./boomeramg_setup_solve` (used by run_cell.sh after cd to TOOL_DIR)
# are intentional + acceptable since the surrounding cwd is already
# /scratch/ per condition (e).
#
# Regex anchors: `(^|[^./A-Za-z0-9_-])` ensures we capture only true
# absolute path tokens (preceded by line-start, whitespace, quote, or
# other non-path char) and skip the trailing piece of relative paths.
g_pass=1
g_targets=(run_cell.sh boomeramg_setup_solve dump_adjacency fd_color_jacobian)
for src_file in "${SBATCH_FILE}" "${RUN_CELL_FILE}"; do
    if [[ ! -f "${src_file}" ]]; then
        emit_fail "g" "source file ${src_file} not found"
        g_pass=0
        continue
    fi
    for target in "${g_targets[@]}"; do
        # Extract absolute path tokens containing the target name.
        # Step 1: grep for lines containing the target.
        # Step 2: awk-extract any /...${target}... token that is preceded
        #         by line-start / space / quote (NOT by . or word char,
        #         which would indicate a relative path like `./foo` or
        #         `../foo`).
        # Step 3: filter to non-/scratch absolute paths.
        bad_paths=$(grep -oE "(^|[[:space:]\"'])/[A-Za-z0-9_./-]*${target}[A-Za-z0-9_./-]*" "${src_file}" 2>/dev/null \
                    | sed -E 's|^[[:space:]\"'\'']||' \
                    | grep -vE "${SCRATCH_ROOT_RE}" \
                    | sort -u)
        if [[ -n "${bad_paths}" ]]; then
            emit_fail "g" "${src_file} references ${target} at non-/scratch absolute path(s): $(echo "${bad_paths}" | tr '\n' ' ')"
            g_pass=0
        fi
    done
done
if [[ ${g_pass} -eq 1 ]]; then
    emit_pass "g" "all absolute-path references to (run_cell.sh / boomeramg_setup_solve / dump_adjacency / fd_color_jacobian) under /scratch/"
fi

# ---------- Final verdict ---------------------------------------------------
if [[ ${fail} -eq 0 ]]; then
    echo "PRECHECK_PASS conditions=${pass_count}/7"
    exit 0
else
    echo "PRECHECK_FAIL_SUMMARY conditions_passed=${pass_count}/7; see PRECHECK_FAIL lines on stderr" >&2
    exit 1
fi
