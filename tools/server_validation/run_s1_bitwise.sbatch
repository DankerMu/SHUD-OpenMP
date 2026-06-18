#!/bin/bash
#SBATCH --job-name=shud-s1-bitwise
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/<server-scratch-root>/.s1-server-validation/shud-s1-bitwise.%j.out
#SBATCH --error=/scratch/<server-scratch-root>/.s1-server-validation/shud-s1-bitwise.%j.err
#
# run_s1_bitwise.sbatch — server-side B0/B1a bitwise validation for
# heihe + heihe_x4 (server-only cases per CLAUDE.md endpoint split).
#
# This template is committed to the repo under
#   tools/server_validation/run_s1_bitwise.sbatch
# but is NOT deployed automatically. Manual deploy steps are documented
# in `tools/server_validation/archive_s1_bitwise.sh`.
#
# CLAUDE.md "三铁律" compliance (compute-node policy):
#   1. Submitted from /scratch (NOT /users/$USER) — orchestrator MUST
#      `cd /scratch/<server-scratch-root>/.s1-server-validation/` before
#      `sbatch run_s1_bitwise.sbatch`.
#   2. --output / --error paths above are under /scratch (compute-node
#      /tmp is node-local; sacct would show ExitCode 127 if we used /tmp).
#   3. All referenced files (this sbatch + patch / hash / run.sh) live
#      under /scratch; no /tmp / /users references in this file.
#
# Placeholders to substitute before deployment:
#   <server-scratch-root>  e.g. `<user>/<project>`
#   <user>                 your username on the server
#
# Workflow:
#   1. Sync repo (recurse-submodules) into /scratch/<server-scratch-root>
#   2. Build SHUD with the binary mode under test (DUMP=0 for *.dat
#      bitwise; DUMP=1 + env for snapshot generation)
#   3. Run heihe + heihe_x4 with 90-day-truncated cfg.para
#   4. SHA256 all benchmarks/<case>/B0_output/*.dat against B0-tag
#   5. SHA256 all snapshot_t<v>{,_before_passvalue}.bin against
#      docs/build_manifest.md listed values
#   6. Persist archive at /scratch/.s1-server-validation/<job-id>/
#
# Exit codes:
#   0  all bitwise comparisons PASS
#   1  any case mismatch (FAIL) — see *.err log for details
#   2  preflight failure (missing data / binary / scratch path)
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.3).

set -euo pipefail
# F7 fix: nullglob so a no-match `*.dat` glob expands to empty instead
# of the literal "*.dat" string (which would cause defensive code below
# to compare the literal as a path and silently skip).
shopt -s nullglob

PROJECT_ROOT="/scratch/<server-scratch-root>"
ARCHIVE_DIR="${PROJECT_ROOT}/.s1-server-validation/${SLURM_JOB_ID:-manual}"
mkdir -p "$ARCHIVE_DIR"

# Preflight: scratch path sanity.
if [ ! -d "$PROJECT_ROOT" ]; then
    echo "[preflight] FAIL: PROJECT_ROOT=$PROJECT_ROOT not found" >&2
    exit 2
fi
if [ ! -d "$PROJECT_ROOT/SHUD" ]; then
    echo "[preflight] FAIL: SHUD submodule not found in $PROJECT_ROOT" >&2
    exit 2
fi

cd "$PROJECT_ROOT"

# Build SHUD (DUMP=0 + USE_RHS_CORE=1 first for *.dat bitwise + new-path exercise;
# DUMP=1 next for snapshots).
# USE_RHS_CORE=1 is required for S1a (openMP #44) server validation: without it
# the build hits the legacy B0 fallback path and trivially matches B0-tag SHA256
# without ever exercising the new rhs_core() dispatch — false-PASS hazard. See
# spec rhs-core-scaffolding "Server-only cases archived within 24h post-merge".
echo "[build] make clean && make SHUD_DUMP_RHS=0 USE_RHS_CORE=1 SHUD_ENABLE_PROFILE=0 shud"
cd SHUD
make clean
if ! make SHUD_DUMP_RHS=0 USE_RHS_CORE=1 SHUD_ENABLE_PROFILE=0 shud > "$ARCHIVE_DIR/build_default.log" 2>&1; then
    echo "[build] FAIL: SHUD_DUMP_RHS=0 USE_RHS_CORE=1 build; see $ARCHIVE_DIR/build_default.log" >&2
    exit 2
fi
cp shud "$ARCHIVE_DIR/shud_default"

# Run heihe + heihe_x4 with default binary; SHA256 vs B0-tag.
FAIL=0
for case in heihe heihe_x4; do
    proj="$case"  # heihe and heihe_x4 both use proj=case as default
    echo "[run] case=$case proj=$proj (DUMP=0)"
    case_dir="$PROJECT_ROOT/SHUD/Basins/$case"
    if [ ! -d "$case_dir" ]; then
        echo "[run] FAIL: case dir $case_dir missing" >&2
        FAIL=1
        continue
    fi
    cd "$case_dir"
    rm -rf "output/$proj.out"
    if ! "$ARCHIVE_DIR/shud_default" "$proj" > "$ARCHIVE_DIR/run_${case}_default.log" 2>&1; then
        echo "[run] FAIL: $case run nonzero; see $ARCHIVE_DIR/run_${case}_default.log" >&2
        FAIL=1
        cd "$PROJECT_ROOT"
        continue
    fi
    cd "$PROJECT_ROOT"
    # Compare each *.dat to B0-tag archive. F7 fix: with nullglob set,
    # a no-match glob expands to empty; count matched files so we can
    # detect the "no goldens in B0_output/" failure mode explicitly
    # instead of silently looping zero times.
    matched_count=0
    for f in benchmarks/$case/B0_output/*.dat; do
        matched_count=$((matched_count + 1))
        bname=$(basename "$f")
        new_path="SHUD/Basins/$case/output/$proj.out/$bname"
        if [ ! -f "$new_path" ]; then
            # F6 fix: missing output is a real test failure, not a WARN.
            echo "[compare] FAIL: $case/$bname not produced" | tee -a "$ARCHIVE_DIR/compare.log" >&2
            FAIL=$((FAIL + 1))
            continue
        fi
        new_sha=$(sha256sum "$new_path" | awk '{print $1}')
        b0_sha=$(git show B0-tag:"$f" 2>/dev/null | sha256sum | awk '{print $1}')
        if [ "$new_sha" = "$b0_sha" ]; then
            echo "[compare] PASS $case/$bname  $new_sha" | tee -a "$ARCHIVE_DIR/compare.log"
        else
            echo "[compare] FAIL $case/$bname  new=$new_sha b0=$b0_sha" | tee -a "$ARCHIVE_DIR/compare.log"
            FAIL=$((FAIL + 1))
        fi
    done
    if [ "$matched_count" -eq 0 ]; then
        echo "[compare] FAIL: $case has zero benchmarks/$case/B0_output/*.dat goldens (empty glob)" \
            | tee -a "$ARCHIVE_DIR/compare.log" >&2
        FAIL=$((FAIL + 1))
    fi
done

# Build SHUD with DUMP=1 + USE_RHS_CORE=1 to generate snapshots from the new
# rhs_update() path (unsuffixed + before_passvalue). USE_RHS_CORE=1 must match
# the default-build above so the snapshot tag string "f_update" goldens come
# from the same dispatch path.
echo "[build] make clean && make SHUD_DUMP_RHS=1 USE_RHS_CORE=1 SHUD_ENABLE_PROFILE=0 shud"
cd SHUD
make clean
if ! make SHUD_DUMP_RHS=1 USE_RHS_CORE=1 SHUD_ENABLE_PROFILE=0 shud > "$ARCHIVE_DIR/build_dump.log" 2>&1; then
    echo "[build] FAIL: SHUD_DUMP_RHS=1 USE_RHS_CORE=1 build; see $ARCHIVE_DIR/build_dump.log" >&2
    exit 2
fi
cp shud "$ARCHIVE_DIR/shud_dump"
cd "$PROJECT_ROOT"

# Generate snapshots at unified t = [1d, 30d, 90d] for heihe / heihe_x4.
# t-values in seconds → abs-minutes done in shell; START_day is read from cfg.para.
for case in heihe heihe_x4; do
    proj="$case"
    case_dir="$PROJECT_ROOT/SHUD/Basins/$case"
    cfg="$case_dir/input/$proj/$proj.cfg.para"
    if [ ! -f "$cfg" ]; then
        echo "[snap] WARN: $cfg missing, skipping snapshots for $case"
        continue
    fi
    start_day=$(awk '$1=="START" {print $2; exit}' "$cfg")
    start_min=$((start_day * 1440))
    # 1d=1440 min, 30d=43200 min, 90d=129600 min
    abs_t="$((start_min + 1440)),$((start_min + 43200)),$((start_min + 129600))"

    for site_suffix in "f_update:" "f_loop_before_passvalue:before_passvalue"; do
        site="${site_suffix%:*}"
        suffix="${site_suffix##*:}"
        staging="$ARCHIVE_DIR/snap_${case}_${site}"
        mkdir -p "$staging"
        cd "$case_dir"
        rm -rf "output/$proj.out"
        echo "[snap] case=$case site=$site suffix='$suffix' abs_t=$abs_t"
        # F6 fix: dump binary failure was silently swallowed by `|| true`;
        # now it FAILs the run + bumps the FAIL counter so the summary
        # surfaces the issue instead of misreporting PASS.
        if ! SHUD_DUMP_T_VALUES="$abs_t" \
             SHUD_DUMP_T_TOL=60 \
             SHUD_DUMP_OUTPUT_DIR="$staging" \
             SHUD_DUMP_CASE_ID="$proj" \
             SHUD_DUMP_SITE="$site" \
             SHUD_DUMP_FNAME_SUFFIX="$suffix" \
                 "$ARCHIVE_DIR/shud_dump" "$proj" > "$ARCHIVE_DIR/run_${case}_${site}.log" 2>&1; then
            echo "[snap] FAIL: dump binary failed for $case (site=$site); see $ARCHIVE_DIR/run_${case}_${site}.log" \
                | tee -a "$ARCHIVE_DIR/snap.log" >&2
            FAIL=$((FAIL + 1))
            cd "$PROJECT_ROOT"
            continue
        fi
        cd "$PROJECT_ROOT"
        # Rename + SHA256.
        for rel_sec in 86400 2592000 7776000; do
            case "$rel_sec" in
                86400) abs_min=$((start_min + 1440)) ;;
                2592000) abs_min=$((start_min + 43200)) ;;
                7776000) abs_min=$((start_min + 129600)) ;;
            esac
            if [ -z "$suffix" ]; then
                src="$staging/snapshot_t${abs_min}.bin"
                dst_name="snapshot_t${rel_sec}.bin"
            else
                src="$staging/snapshot_t${abs_min}_${suffix}.bin"
                dst_name="snapshot_t${rel_sec}_${suffix}.bin"
            fi
            if [ -f "$src" ]; then
                cp "$src" "$ARCHIVE_DIR/$dst_name"
                sha=$(sha256sum "$ARCHIVE_DIR/$dst_name" | awk '{print $1}')
                echo "[snap] PRODUCED $case/$dst_name  $sha" | tee -a "$ARCHIVE_DIR/snap.log"
            else
                # F6 fix: missing snapshot was previously a silent WARN that
                # didn't move the exit gate; promote to FAIL so the summary
                # gate reflects reality.
                echo "[snap] FAIL: $case/$dst_name not produced (src=$src missing)" \
                    | tee -a "$ARCHIVE_DIR/snap.log" >&2
                FAIL=$((FAIL + 1))
            fi
        done
    done
done

echo ""
echo "=== Summary ==="
echo "Archive dir: $ARCHIVE_DIR"
echo "*.dat compare results: $ARCHIVE_DIR/compare.log"
echo "Snapshot generation results: $ARCHIVE_DIR/snap.log"
if [ "$FAIL" -ne 0 ]; then
    # F6 fix: summary message now covers BOTH *.dat bitwise mismatch and
    # snapshot-generation failure (was "see compare.log" only).
    echo "RESULT: FAIL ($FAIL failure(s) in *.dat bitwise compare and/or snapshot generation; see compare.log + snap.log)" >&2
    exit 1
fi
echo "RESULT: PASS (all 4-case *.dat bitwise + snapshot generation OK)"
exit 0
