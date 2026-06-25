#!/usr/bin/env bash
# tools/p1e_2x2_runner.sh
#
# P1e PR-B (issue #310) — 2x2 build matrix single-cell driver.
#
# Runs one (build, case, N, rep) cell of the P1e 192-cell 2x2 matrix and
# captures the 4 deliverables per cell:
#
#   1. cvode_stats.txt   — verbatim 15-key CVODE stats from SHUD's
#                          output/<project>.out/cvode_stats.txt
#   2. <case>.rivqdown.sha256  — SHA256 of the post-PR-B0 deterministic
#                                output/<project>.out/<project>.rivqdown.dat
#   3. wall.sec          — wall-clock seconds (integer) of the SHUD run
#   4. cv_y_hash.txt     — per-tout SHA256 of each cv_y_<t>.bin produced by
#                          the SHUD_DUMP_CV_Y=1 env-gated dump in shud.cpp;
#                          format: "<tout-as-printed>  <sha256>" one per line
#
# Build modes (per design D6, master plan §6 P1e.2):
#   A: make shud                                  -> ./shud           (Serial + Serial RHS)
#   B: make shud_omp                              -> ./shud_omp       (OpenMP NVECTOR + Serial RHS)
#   C: make shud SHUD_ENABLE_OPENMP_RHS=1         -> ./shud           (Serial NVECTOR + StrictOMP RHS)
#   D: make shud_omp SHUD_ENABLE_OPENMP_RHS=1     -> ./shud_omp       (OpenMP NVECTOR + StrictOMP RHS)
#
# Mode C/D pre-PR-G: StrictOMP RHS is a std::abort() stub in
# MD_rhs_core.cpp:802-811; the driver builds + runs are allowed for cell
# completeness but the SHUD run will abort. Cell wall.sec records the
# abort wall (typically <1 s); cvode_stats.txt + rivqdown.sha256 + cv_y_hash
# may be empty or absent (recorded as MISSING). Post-PR-G mode C/D runs
# produce all 4 outputs.
#
# Usage:
#     tools/p1e_2x2_runner.sh --case <name> --build <A|B|C|D> \
#                             --N <int> --rep <int> [--keep] [--out-dir <dir>]
#
# Outputs (default --out-dir = .p1e-runs/2x2/<build>_<case>_N<N>_rep<r>/):
#   - cvode_stats.txt
#   - <case>.rivqdown.sha256
#   - wall.sec
#   - cv_y_hash.txt
#   - cell.meta (provenance: SHUD pin, binary sha, env, build cmd, etc.)
#   - run.log (full SHUD stdout+stderr; useful for debugging)
#
# Without --keep, the working SHUD/Basins/<case>/output/<project>.out
# tree is cleared before the run and left in place after (so it can be
# inspected if needed); the cv_y_*.bin dumps are removed unless --keep
# is passed.
#
# Provenance fields written to cell.meta:
#   shud_pin             git rev-parse HEAD inside SHUD/
#   binary_sha256        sha256 of the SHUD binary just used
#   built_at             ISO timestamp of binary mtime
#   host                 uname -srm
#   omp_num_threads      effective OMP_NUM_THREADS used (= --N)
#   cmd                  exact invocation
#
# Exit codes:
#   0  cell complete + all 4 outputs written (or recorded MISSING with reason)
#   1  caller error (bad args / missing case / missing binary)
#   2  SHUD build failed
#   3  SHUD run failed AND we couldn't even record wall.sec (script crash)
#
# Notes:
#   - Reuses sha256_of_file pattern from tools/archive_b0_output.sh.
#   - Mac + server universal: auto-detects sha256sum vs `shasum -a 256`.
#   - Does NOT modify cfg.para; assumes the 90-day truncation is already
#     in place (run tools/archive_b0_output.sh once per case beforehand if
#     not). This keeps the driver pure-observational.
#   - Does NOT decide cell selection / matrix expansion — that's the
#     downstream wrapper's job (PR-C/D will iterate over manifest cells).

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
SHUD_ROOT="$REPO_ROOT/SHUD"
BASINS_ROOT="$SHUD_ROOT/Basins"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

CASE=""
BUILD=""
NTHREADS=""
REP=""
KEEP_CV_Y=0
OUT_DIR=""

usage() {
    cat <<EOF
Usage: $0 --case <name> --build <A|B|C|D> --N <int> --rep <int> [--keep] [--out-dir <dir>]

P1e PR-B 2x2 build matrix single-cell driver. Runs one (build, case, N, rep)
cell and writes 4 deliverables (cvode_stats.txt / <case>.rivqdown.sha256 /
wall.sec / cv_y_hash.txt) into --out-dir.

Arguments:
  --case <name>     SHUD case name under SHUD/Basins/<name>/ (e.g. keliya, qhh,
                    heihe, heihe_x4)
  --build <A..D>    Build mode:
                      A = make shud                            -> ./shud
                      B = make shud_omp                        -> ./shud_omp
                      C = make shud SHUD_ENABLE_OPENMP_RHS=1   -> ./shud (StrictOMP RHS)
                      D = make shud_omp SHUD_ENABLE_OPENMP_RHS=1 -> ./shud_omp (StrictOMP RHS)
  --N <int>         Thread count (sets OMP_NUM_THREADS; must be >= 1)
  --rep <int>       Repetition index (>= 1; freeform tag, not validated against
                    a manifest)

Options:
  --keep            Keep the cv_y_<tout>.bin dump files in
                    SHUD/Basins/<case>/output/<project>.out/ after the run
                    (default removes them to save disk)
  --out-dir <dir>   Override cell output directory
                    (default: \$REPO_ROOT/.p1e-runs/2x2/<build>_<case>_N<N>_rep<r>/)
  --help            This message.

Build mode -> binary mapping (per design D6, master plan §6 P1e.2):
  A: Serial NVECTOR + Serial RHS    (canonical reference baseline)
  B: OpenMP NVECTOR + Serial RHS    (= current shud_omp; PR-H 10-25% drift baseline)
  C: Serial NVECTOR + StrictOMP RHS (P1e production candidate)
  D: OpenMP NVECTOR + StrictOMP RHS (research boundary)

NOTE: Pre-PR-G, mode C/D RHS is a std::abort() stub. Cells will still be
run for matrix completeness; outputs may be MISSING (recorded as such in
cell.meta). Post-PR-G mode C/D produce all 4 outputs.

Exit codes:
  0  cell complete (all 4 outputs, or recorded MISSING with reason)
  1  caller error (bad args / missing case / missing binary)
  2  SHUD build failed
  3  driver crash before wall.sec written
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --case)    CASE="$2"; shift 2;;
        --build)   BUILD="$2"; shift 2;;
        --N)       NTHREADS="$2"; shift 2;;
        --rep)     REP="$2"; shift 2;;
        --keep)    KEEP_CV_Y=1; shift;;
        --out-dir) OUT_DIR="$2"; shift 2;;
        --help|-h) usage; exit 0;;
        *) echo "error: unknown arg '$1'" >&2; usage >&2; exit 1;;
    esac
done

if [[ -z "$CASE" || -z "$BUILD" || -z "$NTHREADS" || -z "$REP" ]]; then
    echo "error: --case --build --N --rep all required" >&2
    usage >&2
    exit 1
fi
if ! [[ "$BUILD" =~ ^[ABCD]$ ]]; then
    echo "error: --build must be A|B|C|D (got '$BUILD')" >&2
    exit 1
fi
if ! [[ "$NTHREADS" =~ ^[0-9]+$ ]] || [[ "$NTHREADS" -lt 1 ]]; then
    echo "error: --N must be integer >= 1 (got '$NTHREADS')" >&2
    exit 1
fi
if ! [[ "$REP" =~ ^[0-9]+$ ]] || [[ "$REP" -lt 1 ]]; then
    echo "error: --rep must be integer >= 1 (got '$REP')" >&2
    exit 1
fi

CASE_DIR="$BASINS_ROOT/$CASE"
if [[ ! -d "$CASE_DIR" ]]; then
    echo "error: case dir not deployed: $CASE_DIR" >&2
    echo "       hint: deploy NWM case manifest + cfg.para 90d truncation first" >&2
    exit 1
fi

# Build -> binary name + make command
case "$BUILD" in
    A) BIN_NAME="shud";     MAKE_CMD=("make" "shud");;
    B) BIN_NAME="shud_omp"; MAKE_CMD=("make" "shud_omp");;
    C) BIN_NAME="shud";     MAKE_CMD=("make" "shud" "SHUD_ENABLE_OPENMP_RHS=1");;
    D) BIN_NAME="shud_omp"; MAKE_CMD=("make" "shud_omp" "SHUD_ENABLE_OPENMP_RHS=1");;
esac
BIN_PATH="$SHUD_ROOT/$BIN_NAME"

# Output directory
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$REPO_ROOT/.p1e-runs/2x2/${BUILD}_${CASE}_N${NTHREADS}_rep${REP}"
fi
mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------------
# SHA256 portable wrapper (reused from tools/archive_b0_output.sh)
# -----------------------------------------------------------------------------

sha256_of_file() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    else
        shasum -a 256 "$f" | awk '{print $1}'
    fi
}

# -----------------------------------------------------------------------------
# Determine project name (parse manifest or fall back to case name)
# -----------------------------------------------------------------------------

# Per CLAUDE.md the canonical source is benchmarks/<case>/manifest.yaml
# project_name; fall back to <case> when no manifest exists (server-side
# heihe / heihe_x4 may be deployed without a manifest copy).
PROJECT_NAME="$CASE"
MANIFEST="$REPO_ROOT/benchmarks/$CASE/manifest.yaml"
if [[ -f "$MANIFEST" ]]; then
    pn_line="$(grep -E '^project_name:' "$MANIFEST" | head -1 || true)"
    if [[ -n "$pn_line" ]]; then
        # match either   project_name: foo   or   project_name: "foo"
        pn_val="$(echo "$pn_line" | sed -E 's/^project_name:[[:space:]]*"?([^"]*)"?[[:space:]]*$/\1/')"
        if [[ -n "$pn_val" ]]; then
            PROJECT_NAME="$pn_val"
        fi
    fi
fi
OUTPUT_DIR_REL="output/${PROJECT_NAME}.out"
OUTPUT_DIR_ABS="$CASE_DIR/$OUTPUT_DIR_REL"

echo "[p1e_2x2_runner] cell: build=$BUILD case=$CASE N=$NTHREADS rep=$REP"
echo "[p1e_2x2_runner]   project=$PROJECT_NAME bin=$BIN_NAME out_dir=$OUT_DIR"

# -----------------------------------------------------------------------------
# Build phase (idempotent: skip if binary fresher than newest SHUD/src/ file)
# -----------------------------------------------------------------------------

NEED_BUILD=1
SENTINEL_FILE="$SHUD_ROOT/.last_build_mode_${BIN_NAME}"
if [[ -x "$BIN_PATH" ]]; then
    # Find newest src file mtime; bash portable approach with find -newer
    newest_src=$(find "$SHUD_ROOT/src" -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' \) -newer "$BIN_PATH" -print -quit 2>/dev/null || true)
    if [[ -z "$newest_src" ]]; then
        # Also check Makefile freshness
        if [[ "$SHUD_ROOT/Makefile" -ot "$BIN_PATH" ]]; then
            NEED_BUILD=0
            echo "[p1e_2x2_runner]   binary fresh; skipping rebuild"
        fi
    fi
fi

# F-R2-1 (Phase 4.5 verifier): A↔C share `shud` binary, B↔D share `shud_omp`.
# mtime-only freshness check misses build-flag changes (SHUD_ENABLE_OPENMP_RHS),
# so a cell silently runs the WRONG binary across A↔C / B↔D rotations. The
# sentinel records which build mode last produced the binary; mismatch forces
# a rebuild. Sentinel lives under SHUD/ (alongside the binary) and is excluded
# via SHUD/.git/info/exclude pattern `.last_build_mode_*`.
if [[ "$NEED_BUILD" -eq 0 ]]; then
    LAST_BUILD=$(cat "$SENTINEL_FILE" 2>/dev/null || echo "")
    if [[ "$LAST_BUILD" != "$BUILD" ]]; then
        NEED_BUILD=1
        echo "[p1e_2x2_runner]   build mode changed (${LAST_BUILD:-<none>} -> ${BUILD}); forcing rebuild"
    fi
fi

if [[ "$NEED_BUILD" -eq 1 ]]; then
    echo "[p1e_2x2_runner]   building: ${MAKE_CMD[*]}"
    # `make clean` is intentionally omitted to keep cells fast; if the user
    # wants a clean rebuild they should run `cd SHUD && make clean` once
    # before invoking the driver, OR delete the binary.
    if ! ( cd "$SHUD_ROOT" && "${MAKE_CMD[@]}" >"$OUT_DIR/build.log" 2>&1 ); then
        echo "error: SHUD build failed; see $OUT_DIR/build.log" >&2
        tail -30 "$OUT_DIR/build.log" >&2 || true
        exit 2
    fi
    # F-R2-1 (Phase 4.5 verifier): record which build mode produced this
    # binary so the next invocation can detect A↔C / B↔D flag changes.
    echo "$BUILD" > "$SENTINEL_FILE"
fi

if [[ ! -x "$BIN_PATH" ]]; then
    echo "error: SHUD binary not present after build: $BIN_PATH" >&2
    exit 2
fi

# -----------------------------------------------------------------------------
# Prepare clean output directory in the case tree
# -----------------------------------------------------------------------------

# F-R1-4 (Phase 4.5 verifier): defensive guard. PROJECT_NAME parse failure
# would leave OUTPUT_DIR_ABS at <case_dir>/output/.out — refusing to wipe
# protects against silent destruction of unrelated SHUD output trees.
if [[ -z "$PROJECT_NAME" ]] || [[ "$OUTPUT_DIR_ABS" == */output/.out ]]; then
    echo "error: refusing to clear suspicious OUTPUT_DIR_ABS=$OUTPUT_DIR_ABS" >&2
    exit 1
fi
# F-R2-4 (Phase 4.5 verifier): wipes per-cell SHUD output tree. --keep only
# preserves cv_y_*.bin until the NEXT cell run wipes this directory; cell-local
# copies under $OUT_DIR are the durable artifacts.
rm -rf "$OUTPUT_DIR_ABS"
mkdir -p "$OUTPUT_DIR_ABS"

# -----------------------------------------------------------------------------
# Run phase
# -----------------------------------------------------------------------------

# Capture provenance up-front so even a crash leaves a usable cell.meta
SHUD_PIN="$(cd "$SHUD_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
BIN_MTIME="$(date -r "$BIN_PATH" -u +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || stat -c %y "$BIN_PATH" 2>/dev/null || echo unknown)"
BIN_SHA="$(sha256_of_file "$BIN_PATH")"
HOST="$(uname -srm)"

{
    echo "cell:           ${BUILD}_${CASE}_N${NTHREADS}_rep${REP}"
    echo "build:          $BUILD"
    echo "case:           $CASE"
    echo "project_name:   $PROJECT_NAME"
    echo "N:              $NTHREADS"
    echo "rep:            $REP"
    echo "binary:         $BIN_PATH"
    echo "binary_sha256:  $BIN_SHA"
    echo "binary_mtime:   $BIN_MTIME"
    echo "shud_pin:       $SHUD_PIN"
    echo "host:           $HOST"
    echo "cmd:            cd $CASE_DIR && SHUD_DUMP_CV_Y=1 OMP_NUM_THREADS=$NTHREADS $BIN_PATH $PROJECT_NAME"
} > "$OUT_DIR/cell.meta"

# Wall-clock measurement: use $SECONDS for integer s; portable + no perl/date math
RUN_LOG="$OUT_DIR/run.log"
T0=$SECONDS

set +e
(
    cd "$CASE_DIR" && \
    SHUD_DUMP_CV_Y=1 \
    OMP_NUM_THREADS="$NTHREADS" \
    "$BIN_PATH" "$PROJECT_NAME"
) >"$RUN_LOG" 2>&1
RUN_RC=$?
set -e

T1=$SECONDS
WALL_SEC=$(( T1 - T0 ))
echo "$WALL_SEC" > "$OUT_DIR/wall.sec"
echo "exit_code:      $RUN_RC" >> "$OUT_DIR/cell.meta"
echo "wall_sec:       $WALL_SEC" >> "$OUT_DIR/cell.meta"

echo "[p1e_2x2_runner]   run rc=$RUN_RC wall=${WALL_SEC}s"

# -----------------------------------------------------------------------------
# Capture cvode_stats.txt
# -----------------------------------------------------------------------------

CVS_SRC="$OUTPUT_DIR_ABS/cvode_stats.txt"
if [[ -f "$CVS_SRC" ]]; then
    cp -f "$CVS_SRC" "$OUT_DIR/cvode_stats.txt"
else
    : > "$OUT_DIR/cvode_stats.txt"
    echo "# MISSING: $CVS_SRC not produced (SHUD run rc=$RUN_RC)" > "$OUT_DIR/cvode_stats.txt"
fi

# -----------------------------------------------------------------------------
# Capture rivqdown.dat SHA256
# -----------------------------------------------------------------------------

RIV_SRC="$OUTPUT_DIR_ABS/${PROJECT_NAME}.rivqdown.dat"
RIV_SHA_OUT="$OUT_DIR/${CASE}.rivqdown.sha256"
if [[ -f "$RIV_SRC" ]]; then
    RIV_SHA="$(sha256_of_file "$RIV_SRC")"
    printf '%s  %s\n' "$RIV_SHA" "${PROJECT_NAME}.rivqdown.dat" > "$RIV_SHA_OUT"
else
    echo "# MISSING: $RIV_SRC not produced (SHUD run rc=$RUN_RC)" > "$RIV_SHA_OUT"
fi

# -----------------------------------------------------------------------------
# Capture CV_Y per-tout SHA256 (cv_y_<t>.bin produced by SHUD_DUMP_CV_Y=1)
# -----------------------------------------------------------------------------

CV_Y_OUT="$OUT_DIR/cv_y_hash.txt"
: > "$CV_Y_OUT"

# Use python-style globbing: list cv_y_*.bin sorted by name (numeric tout in
# filename gives natural lexicographic order because we zero-pad to 20.6f).
shopt -s nullglob
cv_y_files=("$OUTPUT_DIR_ABS"/cv_y_*.bin)
shopt -u nullglob

if [[ "${#cv_y_files[@]}" -eq 0 ]]; then
    echo "# MISSING: no cv_y_*.bin files in $OUTPUT_DIR_ABS (SHUD_DUMP_CV_Y dump did not fire — likely RHS abort or env lost)" > "$CV_Y_OUT"
else
    # Sort the file list to ensure deterministic ordering across platforms
    IFS=$'\n' cv_y_sorted=($(printf '%s\n' "${cv_y_files[@]}" | sort))
    unset IFS
    for f in "${cv_y_sorted[@]}"; do
        bn="$(basename "$f")"
        # Strip "cv_y_" prefix and ".bin" suffix to recover the tout string
        tout="${bn#cv_y_}"
        tout="${tout%.bin}"
        sha="$(sha256_of_file "$f")"
        printf '%s  %s\n' "$tout" "$sha" >> "$CV_Y_OUT"
    done
fi
echo "cv_y_count:     ${#cv_y_files[@]}" >> "$OUT_DIR/cell.meta"

# -----------------------------------------------------------------------------
# Cleanup CV_Y dumps unless --keep
# -----------------------------------------------------------------------------

if [[ "$KEEP_CV_Y" -eq 0 ]]; then
    rm -f "$OUTPUT_DIR_ABS"/cv_y_*.bin
fi

echo "[p1e_2x2_runner] DONE cell=${BUILD}_${CASE}_N${NTHREADS}_rep${REP}"
echo "    out_dir:            $OUT_DIR"
echo "    cvode_stats.txt:    $OUT_DIR/cvode_stats.txt"
echo "    rivqdown.sha256:    $RIV_SHA_OUT"
echo "    wall.sec:           $OUT_DIR/wall.sec ($WALL_SEC s)"
echo "    cv_y_hash.txt:      $CV_Y_OUT (${#cv_y_files[@]} timesteps)"

exit 0
