#!/bin/bash
# tools/p8pre/run_identity_spike.sh
#
# p8pre-spike Step 2 PR-E run (issue #346) — server-side runner that
# submits the 18-cell identity-precond spike matrix to Slurm.
#
# Matrix per design D4: 2 cases × 3 N-values × 3 reps = 18 cells.
#   cases = heihe (cn14) + heihe_x4 (cn15)
#   N     = 1, 4, 8 (N=2 explicitly SKIPPED)
#   reps  = 1, 2, 3 (singleton chain afterany per (case, N) group)
#
# Pipeline:
#   1. Pre-flight: verify SHUD/shud is Mode C (nm gates) — rebuild on miss.
#   1b. Pre-flight identity nm gate-1: PSetupIdentity / PSolveIdentity /
#       CVodeSetPreconditioner symbols present (per design D5, evidence per
#       issue #346) — Mac sanity 不能替代 server gcc 13.3.0 + libgomp ABI.
#   2. mkdir 18 per-cell artifact dirs under .p8pre-runs/identity_spike/.
#   3. Invoke tools/p8pre/render_identity_spike.sh — captures 18 sbatch lines
#      AND materializes 18 rendered .sbatch files under .p8pre-runs/identity_spike/rendered/.
#   4. Substitute __PREV_JID_*__ placeholders with captured JIDs, execute
#      each `sbatch --parsable` in order, record (case, N, rep, JID) into
#      .p8pre-runs/identity_spike/jid_table.txt.
#   5. Emit summary + `squeue --me -h | wc -l` to stdout.
#
# Slurm 三铁律 (CLAUDE.md):
#   1. MUST be invoked FROM /scratch/frd_muziyao/SHUD-OpenMP (sbatch policy
#      blocks submissions from /users/$USER). This script enforces via cd.
#   2. #SBATCH --output/--error are under /scratch (set by template).
#   3. All artifacts (binary, cfg, rendered .sbatch) live under /scratch.
#
# Exit code:
#   0   all 18 submitted, jid_table.txt has 18 lines, summary printed.
#   2   pre-flight failed (nm gates miss after rebuild OR identity gate-1 fail).
#   3   render wrapper failed (non-18 cell count).
#   4   sbatch submit failed mid-stream (partial jid_table.txt left as-is
#       for triage; caller can re-run after canceling residual jobs).
#
# Reference style: tools/p8pre/run_n8_profile.sh (proven Step 1 PR-A).
# POSIX bash + awk + grep; no Python dependency.

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

REPO="/scratch/frd_muziyao/SHUD-OpenMP"
SHUD_DIR="$REPO/SHUD"
SHUD_BIN="$SHUD_DIR/shud"
RUNS_DIR="$REPO/.p8pre-runs/identity_spike"
RENDERED_DIR="$RUNS_DIR/rendered"
JID_TABLE="$RUNS_DIR/jid_table.txt"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_WRAPPER="$SCRIPT_DIR/render_identity_spike.sh"

CASES=(heihe heihe_x4)
NS=(1 4 8)
REPS=(1 2 3)

# ----------------------------------------------------------------------------
# Phase A: Enforce /scratch cwd (Slurm 铁律 #1)
# ----------------------------------------------------------------------------

cd "$REPO"
echo "[run_identity_spike] cwd=$(pwd)"
echo "[run_identity_spike] host=$(hostname)"
echo "[run_identity_spike] outer_git=$(git rev-parse --short HEAD 2>/dev/null || echo n/a)"
echo "[run_identity_spike] SHUD pin=$(cd "$SHUD_DIR" && git rev-parse --short HEAD 2>/dev/null || echo n/a)"

# ----------------------------------------------------------------------------
# Phase B: Pre-flight — verify Mode C binary
# ----------------------------------------------------------------------------

verify_mode_c() {
    # Returns 0 if nm gates pass, non-zero otherwise. Stdout: gate counts.
    if [ ! -x "$SHUD_BIN" ]; then
        echo "[verify_mode_c] MISS: $SHUD_BIN not executable" >&2
        return 1
    fi
    local serial_count omp_nvec_count gomp_count
    serial_count=$(nm "$SHUD_BIN" 2>/dev/null | grep -c 'N_VNew_Serial' || true)
    omp_nvec_count=$(nm "$SHUD_BIN" 2>/dev/null | grep -c 'N_VNew_OpenMP' || true)
    gomp_count=$(nm -D "$SHUD_BIN" 2>/dev/null | grep -c 'GOMP_parallel' || true)
    echo "[verify_mode_c] N_VNew_Serial=$serial_count (need>=1)"
    echo "[verify_mode_c] N_VNew_OpenMP=$omp_nvec_count (need==0)"
    echo "[verify_mode_c] GOMP_parallel=$gomp_count (need>=1)"
    if [ "$serial_count" -ge 1 ] && [ "$omp_nvec_count" -eq 0 ] && [ "$gomp_count" -ge 1 ]; then
        return 0
    fi
    return 1
}

echo "[run_identity_spike] Phase B: pre-flight nm Mode C verify"
if verify_mode_c; then
    echo "[run_identity_spike] pre-flight PASS: existing Mode C binary"
else
    echo "[run_identity_spike] pre-flight MISS: rebuilding Mode C"
    cd "$SHUD_DIR"
    make clean
    make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1
    cd "$REPO"
    if ! verify_mode_c; then
        echo "[run_identity_spike] FATAL: rebuild did not satisfy Mode C nm gates" >&2
        exit 2
    fi
    echo "[run_identity_spike] pre-flight PASS after rebuild"
fi

# Phase B.2: identity spike 3-symbol nm gate (gate-1 evidence per issue #346 + design D5)
verify_identity_symbols() {
    if [ ! -x "$SHUD_BIN" ]; then return 1; fi
    local psetup psolve setprec
    psetup=$(nm "$SHUD_BIN" 2>/dev/null | grep -c 'PSetupIdentity' || true)
    psolve=$(nm "$SHUD_BIN" 2>/dev/null | grep -c 'PSolveIdentity' || true)
    setprec=$(nm "$SHUD_BIN" 2>/dev/null | grep -c 'CVodeSetPreconditioner' || true)
    echo "[verify_identity_symbols] PSetupIdentity=$psetup (need>=1)"
    echo "[verify_identity_symbols] PSolveIdentity=$psolve (need>=1)"
    echo "[verify_identity_symbols] CVodeSetPreconditioner=$setprec (need>=1)"
    if [ "$psetup" -ge 1 ] && [ "$psolve" -ge 1 ] && [ "$setprec" -ge 1 ]; then return 0; fi
    return 1
}
echo "[run_identity_spike] Phase B.2: identity 3-symbol nm gate-1 evidence"
if verify_identity_symbols; then
    echo "[run_identity_spike] identity nm gate PASS"
    # Capture full nm output as gate-1 evidence file
    nm "$SHUD_BIN" 2>/dev/null | grep -E "PSetupIdentity|PSolveIdentity|CVodeSetPreconditioner" > "$RUNS_DIR/server_nm.log"
    echo "[run_identity_spike] gate-1 evidence: $RUNS_DIR/server_nm.log"
else
    echo "[run_identity_spike] FATAL: identity 3-symbol gate FAIL after Mode C build" >&2
    exit 2
fi

# ----------------------------------------------------------------------------
# Phase C: mkdir 18 per-cell artifact dirs
# ----------------------------------------------------------------------------

echo "[run_identity_spike] Phase C: mkdir per-cell artifact dirs"
mkdir -p "$RUNS_DIR"
mkdir -p "$RENDERED_DIR"
mkdir_count=0
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for rep in "${REPS[@]}"; do
            cell_dir="$RUNS_DIR/${case}_N${n}_rep${rep}"
            mkdir -p "$cell_dir"
            mkdir_count=$((mkdir_count + 1))
        done
    done
done
if [ "$mkdir_count" -ne 18 ]; then
    echo "[run_identity_spike] FATAL: expected 18 cell dirs, got $mkdir_count" >&2
    exit 3
fi
echo "[run_identity_spike] mkdir cells=$mkdir_count"

# ----------------------------------------------------------------------------
# Phase D: Invoke render wrapper, capture stdout (18 sbatch lines)
# ----------------------------------------------------------------------------

echo "[run_identity_spike] Phase D: render 18 cells"
if [ ! -x "$RENDER_WRAPPER" ]; then
    echo "[run_identity_spike] FATAL: render wrapper not executable: $RENDER_WRAPPER" >&2
    exit 3
fi

# Capture render stdout into temp file. Stderr passes through.
RENDER_STDOUT="$RUNS_DIR/render_stdout.txt"
"$RENDER_WRAPPER" > "$RENDER_STDOUT"
render_lines=$(wc -l < "$RENDER_STDOUT" | tr -d ' ')
echo "[run_identity_spike] render emitted $render_lines sbatch lines"
if [ "$render_lines" -ne 18 ]; then
    echo "[run_identity_spike] FATAL: render emitted $render_lines lines, expected 18" >&2
    exit 3
fi

rendered_count=$(find "$RENDERED_DIR" -maxdepth 1 -name 'submit_*.sbatch' -type f | wc -l | tr -d ' ')
echo "[run_identity_spike] rendered .sbatch files=$rendered_count"
if [ "$rendered_count" -ne 18 ]; then
    echo "[run_identity_spike] FATAL: $rendered_count rendered .sbatch files, expected 18" >&2
    exit 3
fi

# ----------------------------------------------------------------------------
# Phase E: Parse + submit with JID substitution
#
# Each render stdout line looks like one of:
#   sbatch --job-name=p8pre_identity_<case>_N<n>_singleton <path>.sbatch
#   sbatch --job-name=p8pre_identity_<case>_N<n>_singleton --dependency=afterany:__PREV_JID_<case>_N<n>_rep<r-1>__ <path>.sbatch
#
# Strategy: maintain assoc array prev_jid[<case>_N<n>_rep<r>] = <JID>. For each
# line, if dependency placeholder present, look up + substitute the actual JID
# captured from the prior submit, then run `sbatch --parsable` to capture the
# new JID.
#
# Output: jid_table.txt with 4 space-separated columns per line: case N rep JID.
# ----------------------------------------------------------------------------

echo "[run_identity_spike] Phase E: submit 18 cells with JID substitution"

declare -A PREV_JID

# Truncate jid_table.txt to start fresh
: > "$JID_TABLE"

submit_count=0
fail_count=0

# Parse lines in render order (heihe N=1 rep1..3, heihe N=4 rep1..3, ... heihe_x4 N=8 rep3)
while IFS= read -r line; do
    # Extract (case, N, rep) from rendered_path (last field of the line).
    rendered_path=$(awk '{print $NF}' <<< "$line")
    rendered_base=$(basename "$rendered_path")
    # rendered_base format: submit_<case>_N<n>_rep<r>.sbatch
    # Parse via regex; case can contain underscore (heihe_x4) — anchor on _N pattern.
    if [[ "$rendered_base" =~ ^submit_(.+)_N([0-9]+)_rep([0-9]+)\.sbatch$ ]]; then
        case="${BASH_REMATCH[1]}"
        n="${BASH_REMATCH[2]}"
        rep="${BASH_REMATCH[3]}"
    else
        echo "[run_identity_spike] FATAL: cannot parse cell from $rendered_base" >&2
        exit 4
    fi

    # Substitute __PREV_JID_<case>_N<n>_rep<R-1>__ placeholder if present
    if [[ "$line" =~ __PREV_JID_([A-Za-z0-9_]+)__ ]]; then
        placeholder_key="${BASH_REMATCH[1]}"
        prev_jid="${PREV_JID[$placeholder_key]:-}"
        if [ -z "$prev_jid" ]; then
            echo "[run_identity_spike] FATAL: missing prior JID for placeholder $placeholder_key (cell=${case}_N${n}_rep${rep})" >&2
            exit 4
        fi
        line=$(printf '%s\n' "$line" | sed "s/__PREV_JID_${placeholder_key}__/${prev_jid}/g")
    fi

    # Strip leading "sbatch " — we want to invoke `sbatch --parsable <args>` so
    # we can capture just the JID on stdout.
    sbatch_args=${line#sbatch }
    echo "[run_identity_spike] submit cell=${case}_N${n}_rep${rep} cmd: sbatch --parsable $sbatch_args"

    # Submit; capture JID. --parsable emits just `<JID>` or `<JID>;<cluster>`.
    set +e
    raw_out=$(sbatch --parsable $sbatch_args 2>&1)
    rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[run_identity_spike] FATAL: sbatch failed rc=$rc for cell ${case}_N${n}_rep${rep}: $raw_out" >&2
        fail_count=$((fail_count + 1))
        # Continue logging what we got so triage is easier. Then exit at end of loop.
        printf '%s %s %s ERR_rc%d\n' "$case" "$n" "$rep" "$rc" >> "$JID_TABLE"
        continue
    fi
    # Strip cluster suffix if present
    jid=${raw_out%%;*}
    # Validate JID is positive integer
    if ! [[ "$jid" =~ ^[0-9]+$ ]]; then
        echo "[run_identity_spike] FATAL: sbatch returned non-numeric JID '$raw_out' for cell ${case}_N${n}_rep${rep}" >&2
        fail_count=$((fail_count + 1))
        printf '%s %s %s ERR_nonnum\n' "$case" "$n" "$rep" >> "$JID_TABLE"
        continue
    fi

    # Record (case, N, rep, JID)
    printf '%s %s %s %s\n' "$case" "$n" "$rep" "$jid" >> "$JID_TABLE"
    submit_count=$((submit_count + 1))

    # Stash JID under the placeholder key the NEXT rep in this (case, N) chain
    # will reference. Render wrapper uses key pattern: <case>_N<n>_rep<r>.
    PREV_JID["${case}_N${n}_rep${rep}"]="$jid"

done < "$RENDER_STDOUT"

echo "[run_identity_spike] submit summary: submit_count=$submit_count fail_count=$fail_count"
table_lines=$(wc -l < "$JID_TABLE" | tr -d ' ')
echo "[run_identity_spike] jid_table.txt lines=$table_lines (path=$JID_TABLE)"

# ----------------------------------------------------------------------------
# Phase F: Summary + queue state
# ----------------------------------------------------------------------------

echo "[run_identity_spike] Phase F: queue state"
queue_total=$(squeue --me -h 2>/dev/null | wc -l | tr -d ' ' || echo 0)
echo "[run_identity_spike] squeue --me total=$queue_total"

# JID range (min/max) for human eyeballing
if [ "$submit_count" -gt 0 ]; then
    jid_min=$(awk '$4 ~ /^[0-9]+$/ {print $4}' "$JID_TABLE" | sort -n | head -1)
    jid_max=$(awk '$4 ~ /^[0-9]+$/ {print $4}' "$JID_TABLE" | sort -n | tail -1)
    echo "[run_identity_spike] JID range: $jid_min .. $jid_max"
fi

if [ "$submit_count" -ne 18 ] || [ "$fail_count" -ne 0 ]; then
    echo "[run_identity_spike] PARTIAL: submit_count=$submit_count fail_count=$fail_count (expected 18 / 0)" >&2
    exit 4
fi

echo "[run_identity_spike] OK: all 18 cells submitted"
exit 0
