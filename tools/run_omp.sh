#!/usr/bin/env bash
# tools/run_omp.sh — S5d.4 (#182) OpenMP thread-binding wrapper for SHUD
#
# Spec: openspec/changes/b1b-baseline-completion/specs/s5d-data-layout-soa-numa
#   Requirement "线程绑定 run script 与 manifest 字段必填"
#   Scenario   "run_omp.sh 提供完整 OMP 环境"
# Design: openspec/changes/b1b-baseline-completion/design.md D5
#   "OMP_PROC_BIND 通过 run script + manifest 双锚，不强制程序覆盖"
#
# What this wrapper does (and intentionally NOT does):
#   - EXPORTS OMP_PROC_BIND / OMP_PLACES / OMP_NUM_THREADS to the
#     defaults expected by S5d.3 first-touch + S6b RHS hot path, but
#     ONLY if not already set in the caller's environment. This lets
#     SLURM/PBS users override per-job binding (design D5 rationale
#     #1-2: do not strip user testing knobs).
#   - Prints the final OMP_* state to STDERR (NOT stdout) before
#     invoking the binary, so the operator sees what binding is in
#     effect without polluting the SHUD stdout that downstream tools
#     (cvode_stats_diff, [NUMA] log-token grep gates) parse.
#   - Forwards every remaining argv element to ./shud so any case
#     name / driver flag passes through unchanged.
#
# What this wrapper does NOT do:
#   - It does NOT call `omp_set_num_threads()` from inside the program
#     (forbidden by design D5; the env approach keeps the override
#     surface uniform across login / SLURM / interactive shells).
#   - It does NOT auto-detect socket / core counts. The default
#     OMP_NUM_THREADS=1 is the B1b serial bitwise baseline; P1+
#     workflows are expected to override OMP_NUM_THREADS explicitly.
#
# Usage:
#   tools/run_omp.sh ./shud keliya
#   OMP_NUM_THREADS=8 tools/run_omp.sh ./shud heihe
#   OMP_PROC_BIND=spread tools/run_omp.sh ./shud heihe   # user testing knob
#
# Exit status: returns the SHUD binary's exit status verbatim.

set -euo pipefail

# Argv check FIRST — fail fast before any env echo / export so a bare
# `tools/run_omp.sh` (no command) emits ONLY the ERROR on stderr without
# polluting the operator log with state lines for a run that never starts.
if [[ $# -eq 0 ]]; then
    echo "[OMP] ERROR: no command provided. Usage: tools/run_omp.sh ./shud <case>" 1>&2
    exit 2
fi

# Defaults satisfy spec Scenario L93-95: >= 3 export OMP_* + 1 ./shud
# invocation + 1 stderr state echo.
#
# Why `=` (assign-if-unset, colon-less) instead of `:=`:
#   - Caller's env wins (SLURM/PBS scripts already exporting OMP_*
#     keep their values).
#   - Empty string is treated as set (so `OMP_PROC_BIND=` from the
#     caller is NOT overwritten — operator override discipline per
#     design D5 "do not strip user testing knobs"). The colon form
#     `:=` would treat empty as unset and silently substitute the
#     default, defeating the override.
: "${OMP_PROC_BIND=close}"
: "${OMP_PLACES=cores}"
: "${OMP_NUM_THREADS=1}"

export OMP_PROC_BIND OMP_PLACES OMP_NUM_THREADS

# stderr state echo — operator-facing. Single line, prefix matches
# `[OMP]` taxonomy used by the in-program WARNING (shud.cpp emit_numa_token
# stderr branch). Downstream log parsers should look for `[OMP] ` lines on
# stderr; the run.log captured by the orchestrator should `2>&1` merge
# both streams when grep ordering matters.
printf '[OMP] PROC_BIND=%s, PLACES=%s, NUM_THREADS=%s\n' \
    "$OMP_PROC_BIND" "$OMP_PLACES" "$OMP_NUM_THREADS" 1>&2

# Forward all remaining args to the SHUD binary (or whichever exe the
# caller provides as $1; this lets the wrapper sit in front of e.g.
# shud_asan or shud_profile too).
exec "$@"
