#!/bin/bash
#SBATCH --job-name=shud-p7-wallclock
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/<server-scratch-root>/.p7-wallclock-validation/shud-p7-wallclock.%j.out
#SBATCH --error=/scratch/<server-scratch-root>/.p7-wallclock-validation/shud-p7-wallclock.%j.err
#
# run_p7_wallclock.sbatch — P7 exit wall-clock acceptance for the
# strict-OMP build (Config E: SHUD_ENABLE_OPENMP_RHS=1) on the server
# (single-socket 8-core x86_64 Linux per master plan §1.1.1 / §5
# cross-platform 验收铁律 + design.md §R6).
#
# This template is committed to the repo under
#   tools/server_validation/run_p7_wallclock.sbatch
# but is NOT deployed automatically. Manual deploy steps mirror
# `tools/server_validation/archive_s1_bitwise.sh`; the path is contract-
# fixed for s2-strict-omp-full #100 task 8.11 (do NOT rename).
#
# CLAUDE.md "三铁律" compliance (compute-node policy):
#   1. Submitted from /scratch (NOT /users/$USER) — orchestrator MUST
#      `cd /scratch/<server-scratch-root>/.p7-wallclock-validation/`
#      before `sbatch run_p7_wallclock.sbatch`.
#   2. --output / --error paths above are under /scratch (compute-node
#      /tmp is node-local; sacct would show ExitCode 127 if /tmp).
#   3. All referenced files (this sbatch + cfg snippets + log/median
#      sinks) live under /scratch; no /tmp / /users references here.
#
# Placeholders to substitute before deployment:
#   <server-scratch-root>  e.g. `<user>/<project>` (mirrors archive_s1_bitwise.sh)
#   <user>                 your username on the server
#
# Acceptance contract (master plan §1.1.1 + spec strict-omp-acceptance-gates
# Requirement 7):
#   - heihe_x4   @ OMP_NUM_THREADS=8 vs B1a-tag serial SHALL be >= 1.5x
#   - qinyijiang @ OMP_NUM_THREADS=8 vs B1a-tag serial SHALL be >= 1.5x
#   - heihe      @ OMP_NUM_THREADS=8 vs B1a-tag serial SHALL be >= 0.95x
#       (heihe is 79% IO-dominated, Amdahl upper bound 1.13x; sibling
#        change `s2-opt-io-heihe` owns the >= 1.5x goal — Opt-IO scope).
#
# Methodology:
#   - >= 3 repeats per case at OMP_NUM_THREADS=8, record median wall-clock.
#   - Serial baseline = B1a-tag (commit 64569b3 / SHUD pin 58327c5) Config A
#     binary rerun at the same Slurm job on the same node so hardware /
#     filesystem / SUNDIALS install are matched. Serial median taken from
#     the same >= 3-rep methodology.
#   - 90-day truncation (CLAUDE.md project rule) — wall-clock NOT compared
#     in absolute seconds; speedup normalized per-model-time-step (design.md
#     §R6). Both serial + OMP runs use the same 90-day cfg.para.
#   - Cases:
#       * heihe       (NumEle = 6335, server-only forcing)
#       * heihe_x4    (NumEle ~ 25k, server-only forcing, rSHUD-generated)
#       * qinyijiang  (NumEle = 3155, mirrored from local case set;
#                     mesh-prefix = `nanlin` — see CASE_TO_MESH map below.
#                     rSHUD generated the mesh under input/nanlin/, so SHUD
#                     binary must be invoked as `./shud_omp nanlin` from
#                     Basins/qinyijiang/. Identity fallback is forbidden
#                     per spec wallclock-script-mesh-mapping and #135.)
#     keliya / xinanjiang_upstream are NumEle < OMP_CUTOFF (1024) so they
#     would fall through to serial fallback — excluded from wall-clock
#     acceptance per design §D13 logic (mirrors A3b case-selection rules).
#
# Workflow:
#   1. Sync repo (recurse-submodules) into /scratch/<server-scratch-root>
#      (operator-side; run_p7_wallclock.sbatch only consumes the synced
#      tree).
#   2. Build SHUD twice (serial baseline + OMP):
#        - Serial: make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 shud (Config A,
#          B1a-tag binary expected; preserved as $ARCHIVE_DIR/shud_serial)
#        - OMP:    make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 \
#                       SHUD_ENABLE_OPENMP_RHS=1 shud (Config E, the
#                       P7 strict build; preserved as $ARCHIVE_DIR/shud_omp)
#   3. For each case in (heihe heihe_x4 qinyijiang):
#        - Run shud_serial 3x: record wall_serial_i (i = 1..3) -> median
#        - Run shud_omp    3x @ OMP_NUM_THREADS=8 OMP_PROC_BIND=close
#                              OMP_PLACES=cores: record wall_omp_i -> median
#        - Compute speedup = median(serial) / median(omp); persist to
#          $ARCHIVE_DIR/wallclock_<case>.txt with raw + median + variance
#          (sample stddev / mean fraction).
#   4. Gate check against acceptance contract above; tee a final
#      $ARCHIVE_DIR/acceptance_summary.txt with PASS / FAIL per case + a
#      `RESULT:` line so downstream tooling (#100 task 8.11) can grep.
#
# Exit codes:
#   0  all 3 cases meet their target speedup
#   1  any case below target (FAIL gate)
#   2  preflight / build failure
#
# Owned by: openspec change s2-strict-omp-full (task 9.4 #101 scaffold;
#   server execution = #100 task 8.11).

set -euo pipefail
shopt -s nullglob

PROJECT_ROOT="/scratch/<server-scratch-root>"
ARCHIVE_DIR="${PROJECT_ROOT}/.p7-wallclock-validation/${SLURM_JOB_ID:-manual}"
mkdir -p "$ARCHIVE_DIR"

# Repeats per case (>= 3 per master plan §1.1.1 + spec Scenario "P7
# wall-clock heihe_x4 >= 1.5x" 'THEN 3 次重复取中位数').
REPEATS=3

# Cases under wall-clock acceptance (mirrors spec strict-omp-acceptance-gates
# Requirement 7 + design §D13 case-selection):
CASES_OMP_TARGET="qinyijiang heihe_x4"  # both expect >= 1.5x
CASE_HEIHE="heihe"                       # expects >= 0.95x (non-regressive)

# Case → mesh-prefix map (rSHUD case dir-name may differ from mesh prefix).
# qinyijiang case stores its mesh under input/nanlin/ (per #135 + spec
# wallclock-script-mesh-mapping); heihe / heihe_x4 are identity-mapped.
# Missing entries fail-fast via :? expansion below — NO silent identity
# fallback (that fallback was the original bug; #127 job 8453 wrote 0.01s).
declare -A CASE_TO_MESH=(
    [qinyijiang]=nanlin
    [heihe]=heihe
    [heihe_x4]=heihe_x4
)

# Preflight: scratch path sanity (mirrors run_s1_bitwise.sbatch).
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[preflight] FAIL: PROJECT_ROOT=$PROJECT_ROOT not found" >&2
    exit 2
fi
if [ ! -d "$PROJECT_ROOT/SHUD" ]; then
    echo "[preflight] FAIL: SHUD submodule not found in $PROJECT_ROOT" >&2
    exit 2
fi

cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Build phase: produce 2 binaries (Serial baseline + OMP P7 build).
# -----------------------------------------------------------------------------

# Build serial baseline (Config A, defaults). Expected to be byte-equivalent
# to B1a-tag's binary when run from the B1a-tag pin; if the current outer
# repo is post-B1a, the Serial build is treated as a fresh measurement
# (still serial code-path; OMP gain measured against it).
echo "[build] SERIAL: make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 shud"
cd SHUD
make clean
if ! make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 shud > "$ARCHIVE_DIR/build_serial.log" 2>&1; then
    echo "[build] FAIL: serial build; see $ARCHIVE_DIR/build_serial.log" >&2
    exit 2
fi
cp shud "$ARCHIVE_DIR/shud_serial"

# Build OMP (Config E: SHUD_ENABLE_OPENMP_RHS=1). This is the P7 strict
# build target. ExecPolicy::StrictOMP dispatch must be real here (i.e.
# the f.cpp routing must call into rhs_core(... StrictOMP) not std::abort).
# If still on the pre-P7 stub, this build will SUCCEED but a subsequent
# shud run will SIGABRT — preflight failure surfaces in run logs below.
echo "[build] OMP:    make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 SHUD_ENABLE_OPENMP_RHS=1 shud"
make clean
if ! make SHUD_DUMP_RHS=0 SHUD_ENABLE_PROFILE=0 SHUD_ENABLE_OPENMP_RHS=1 shud > "$ARCHIVE_DIR/build_omp.log" 2>&1; then
    echo "[build] FAIL: omp build; see $ARCHIVE_DIR/build_omp.log" >&2
    exit 2
fi
cp shud "$ARCHIVE_DIR/shud_omp"
cd "$PROJECT_ROOT"

# -----------------------------------------------------------------------------
# Run + measure phase: REPEATS x (serial + omp) per case.
# -----------------------------------------------------------------------------

# median <space-separated-numbers>
# Prints the median of numeric stdin args. Sample size is REPEATS (3); for
# an odd N the median is the (N+1)/2-th sorted value. (Generic for future
# REPEATS > 3.)
median() {
    printf '%s\n' "$@" | sort -n | awk -v n=$# '
        NR == int((n+1)/2) { printf "%s", $1; exit }
    '
}

# variance_pct <space-separated-numbers>
# Prints sample stddev / mean * 100 (%) as a coarse stability indicator.
# Wall-clock variance > ~5% on the same node usually indicates a noisy
# neighbor or thermal throttling — flag in the summary for operator review.
variance_pct() {
    printf '%s\n' "$@" | awk '
        { sum += $1; sq += $1*$1; n++ }
        END {
            if (n < 2) { print "0.0"; exit }
            mean = sum / n
            var = (sq - n * mean * mean) / (n - 1)
            if (var < 0) var = 0
            sd = sqrt(var)
            if (mean == 0) { print "inf"; exit }
            printf "%.2f", sd / mean * 100
        }
    '
}

# assert_real_s_sane <real_s>
# Fail-fast if SHUD's reported `time -p` real wall is < 1.0s under
# 90-day truncation — any 90-day legit run is >= ~30s on the smallest
# case (keliya local) and >= ~480s on qinyijiang (mesh=nanlin) server
# per #135 design.md §D3. Sub-second values mean SHUD early-exited
# (EXIT 12 missing cfg, SIGABRT, etc.) but `time -p` still wrote a row
# — see #127 job 8453 wallclock_qinyijiang.txt 0.01s INVALID.
# Per spec wallclock-script-mesh-mapping Requirement "SHUD anomalously
# short run SHALL fail-fast".
assert_real_s_sane() {
    arss_real="$1"
    if awk -v r="$arss_real" 'BEGIN { exit !(r < 1.0) }'; then
        echo "[run] FAIL: anomalously short run real_s=${arss_real}s < 1.0s threshold — likely SHUD early-exit (EXIT 12 / SIGABRT); inspect rep_log for 'Fatal Error' / 'EXIT' lines" >&2
        return 1
    fi
    return 0
}

# run_case <case> <proj> <binary_path> [omp_threads]
# Runs the binary REPEATS times, parses wall-clock from a SHUD-produced
# log line or wraps with `time -p`. Emits "wall_1 wall_2 wall_3" on stdout.
# omp_threads: when non-empty, exports OMP_NUM_THREADS + OMP_PROC_BIND +
# OMP_PLACES per master plan §1.1.1 hardware contract.
run_case() {
    rc_case="$1"
    rc_proj="$2"
    rc_bin="$3"
    rc_threads="${4:-}"
    rc_case_dir="$PROJECT_ROOT/SHUD/Basins/$rc_case"
    if [ ! -d "$rc_case_dir" ]; then
        echo "[run] FAIL: case dir $rc_case_dir missing" >&2
        return 1
    fi
    rc_label="serial"
    if [ -n "$rc_threads" ]; then
        rc_label="omp${rc_threads}"
    fi
    rc_walls=""
    rep=1
    while [ "$rep" -le "$REPEATS" ]; do
        cd "$rc_case_dir"
        rm -rf "output/$rc_proj.out"
        # Use `time -p` (POSIX real / user / sys seconds) and grep `^real`.
        # `time` writes to stderr; capture both streams to log + tee real
        # wall to a per-rep file so the grep is robust to SHUD stdout.
        rep_log="$ARCHIVE_DIR/run_${rc_case}_${rc_label}_rep${rep}.log"
        if [ -n "$rc_threads" ]; then
            { OMP_NUM_THREADS="$rc_threads" OMP_PROC_BIND=close OMP_PLACES=cores \
              time -p "$rc_bin" "$rc_proj" >> "$rep_log" 2>&1 ; } 2>> "$rep_log"
        else
            { time -p "$rc_bin" "$rc_proj" >> "$rep_log" 2>&1 ; } 2>> "$rep_log"
        fi
        real_s=$(awk '/^real / {print $2; exit}' "$rep_log")
        if [ -z "$real_s" ]; then
            echo "[run] FAIL: $rc_case $rc_label rep${rep} no 'real' line in $rep_log" >&2
            cd "$PROJECT_ROOT"
            return 1
        fi
        if ! assert_real_s_sane "$real_s"; then
            echo "[run] FAIL: $rc_case $rc_label rep${rep} rep_log preserved: $rep_log" >&2
            cd "$PROJECT_ROOT"
            return 1
        fi
        rc_walls="$rc_walls $real_s"
        rep=$((rep + 1))
    done
    cd "$PROJECT_ROOT"
    # Trim leading space, emit.
    echo "${rc_walls# }"
}

# Acceptance gate helper. awk float compare; prints PASS/FAIL + speedup.
# gate_check <case> <speedup> <target>
gate_check() {
    gc_case="$1"
    gc_speedup="$2"
    gc_target="$3"
    gc_pass=$(awk -v s="$gc_speedup" -v t="$gc_target" 'BEGIN { print (s >= t) ? "1" : "0" }')
    if [ "$gc_pass" = "1" ]; then
        printf "PASS  %-12s speedup=%sx  target>=%sx\n" "$gc_case" "$gc_speedup" "$gc_target"
        return 0
    fi
    printf "FAIL  %-12s speedup=%sx  target>=%sx\n" "$gc_case" "$gc_speedup" "$gc_target"
    return 1
}

# -----------------------------------------------------------------------------
# Measure each case.
# -----------------------------------------------------------------------------
FAIL=0

measure_case() {
    mc_case="$1"
    mc_target="$2"
    mc_proj="${CASE_TO_MESH[$mc_case]:?case→mesh mapping missing for $mc_case (add entry to CASE_TO_MESH map; identity fallback forbidden per spec wallclock-script-mesh-mapping)}"
    echo ""
    echo "=== measure: $mc_case target=${mc_target}x repeats=$REPEATS ==="

    serial_walls=$(run_case "$mc_case" "$mc_proj" "$ARCHIVE_DIR/shud_serial") || {
        echo "[measure] FAIL: serial run failed for $mc_case" >&2
        FAIL=$((FAIL + 1))
        return 1
    }
    omp_walls=$(run_case "$mc_case" "$mc_proj" "$ARCHIVE_DIR/shud_omp" 8) || {
        echo "[measure] FAIL: omp run failed for $mc_case" >&2
        FAIL=$((FAIL + 1))
        return 1
    }

    serial_median=$(median $serial_walls)
    omp_median=$(median $omp_walls)
    serial_var=$(variance_pct $serial_walls)
    omp_var=$(variance_pct $omp_walls)
    speedup=$(awk -v s="$serial_median" -v o="$omp_median" \
                  'BEGIN { if (o == 0) { print "inf"; exit } printf "%.3f", s / o }')

    summary="$ARCHIVE_DIR/wallclock_${mc_case}.txt"
    {
        printf "case=%s\n" "$mc_case"
        printf "repeats=%d\n" "$REPEATS"
        printf "serial_walls=%s\n" "$serial_walls"
        printf "serial_median_s=%s\n" "$serial_median"
        printf "serial_variance_pct=%s\n" "$serial_var"
        printf "omp_walls=%s\n" "$omp_walls"
        printf "omp_median_s=%s\n" "$omp_median"
        printf "omp_variance_pct=%s\n" "$omp_var"
        printf "speedup=%s\n" "$speedup"
        printf "target=%s\n" "$mc_target"
    } > "$summary"
    cat "$summary"

    if ! gate_check "$mc_case" "$speedup" "$mc_target" >> "$ARCHIVE_DIR/acceptance_summary.txt"; then
        FAIL=$((FAIL + 1))
    fi
}

# Wipe / re-create the summary at the top of the run.
: > "$ARCHIVE_DIR/acceptance_summary.txt"

# `|| true` 防止 measure_case 内层 fail-fast (run_case return 1 →
# `return 1`) 经 set -euo pipefail 触发 bash errexit, 跳过 Summary 块。
# FAIL 计数由 measure_case 内部累计, 最终 exit code 由 line 351 if 决定。
# 详 spec wallclock-script-result-always-emitted / #137.
for case in $CASES_OMP_TARGET; do
    measure_case "$case" "1.5" || true
done
measure_case "$CASE_HEIHE" "0.95" || true

# -----------------------------------------------------------------------------
# Summary.
# -----------------------------------------------------------------------------
echo ""
echo "=== Summary ==="
cat "$ARCHIVE_DIR/acceptance_summary.txt"
echo "Archive dir: $ARCHIVE_DIR"
if [ "$FAIL" -ne 0 ]; then
    echo "RESULT: FAIL ($FAIL case(s) below target; see wallclock_*.txt)" \
        | tee -a "$ARCHIVE_DIR/acceptance_summary.txt" >&2
    exit 1
fi
echo "RESULT: PASS (heihe_x4 + qinyijiang >= 1.5x, heihe >= 0.95x)" \
    | tee -a "$ARCHIVE_DIR/acceptance_summary.txt"
exit 0
