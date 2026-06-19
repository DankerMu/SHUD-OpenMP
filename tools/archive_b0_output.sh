#!/usr/bin/env bash
# tools/archive_b0_output.sh
#
# B0 baseline output archive driver (issue #11, S0-8b).
#
# Purpose: for a given benchmark case, run SHUD `<N>` times under the locked
# B0 binary, verify SHA256 repeatability of the manifest-listed output set,
# and copy run 1's outputs into `benchmarks/<case>/B0_output/` together with
# a `repeatability.txt` audit trail. This script realizes:
#
#   - openspec/changes/s0-baseline-lock/specs/b0-archive/spec.md
#       "Output archive per benchmark", "Three-run repeatability gate"
#   - CLAUDE.md "90 天截断政策": cfg.para `END = START + 90` for B0 archive runs
#     (snapshot probe in #9 is exempt; uses 110d for the 100d probe).
#
# Usage:
#     tools/archive_b0_output.sh <case-name> [num-runs (default 3)]
#
# Behavior:
#   1. Parse benchmarks/<case>/manifest.yaml for:
#        project_name, has_lake, output_compare.output_files
#   2. Snapshot SHUD/Basins/<case>/input/<project>/<project>.cfg.para to
#      `.cfg.para.preB0archive` (idempotent: only on first run) then rewrite
#      END = START + 90 if not already truncated.
#   3. For each run 1..N: clear output dir, run SHUD, sha256 the manifest
#      output_files set plus the cvode_stats.txt sibling. Hash file goes to
#      a mktemp-allocated path under /tmp (random 6-char suffix; cleaned up
#      via EXIT trap so no stale files remain even on crash).
#   4. Diff the N hash files: any disagreement → exit 1 with the diff dump.
#   5. On agreement, copy run 1's outputs into benchmarks/<case>/B0_output/
#      (preserving the file basenames; lake-mentioning files skipped when
#      manifest has_lake=false), then write repeatability.txt with the
#      per-run summary SHA256 and PASS verdict.
#
# Conventions:
#   - Project root = parent dir of the dir containing this script.
#   - Standard POSIX tooling only (bash, awk, sed, sha256sum-or-shasum, diff,
#     cp, rm). No Python (CLAUDE.md uv policy).
#   - sha256sum on Linux, `shasum -a 256` on macOS (auto-detected).
#   - `set -euo pipefail` strict mode.
#
# Exit codes:
#   0  success — archive written, repeatability PASS
#   1  argument / manifest / case-dir problem (caller error)
#   2  SHUD run failed
#   3  repeatability FAILED (hashes diverge across runs)

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
SHUD_ROOT="$REPO_ROOT/SHUD"
BASINS_ROOT="$SHUD_ROOT/Basins"
BENCH_ROOT="$REPO_ROOT/benchmarks"
SHUD_BIN="$SHUD_ROOT/shud"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <case-name> [num-runs (default 3)]" >&2
    exit 1
fi

CASE_NAME="$1"
NUM_RUNS="${2:-3}"

if ! [[ "$NUM_RUNS" =~ ^[0-9]+$ ]] || [[ "$NUM_RUNS" -lt 2 ]]; then
    echo "error: num-runs must be an integer >= 2 (got '$NUM_RUNS')" >&2
    exit 1
fi

MANIFEST="$BENCH_ROOT/$CASE_NAME/manifest.yaml"
CASE_DIR="$BASINS_ROOT/$CASE_NAME"
ARCHIVE_DIR="$BENCH_ROOT/$CASE_NAME/B0_output"

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "error: case directory not deployed: $CASE_DIR" >&2
    exit 1
fi
if [[ ! -x "$SHUD_BIN" ]]; then
    echo "error: SHUD binary not built/executable: $SHUD_BIN" >&2
    echo "       hint: cd SHUD && make shud" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# SHA256 portable wrapper
# -----------------------------------------------------------------------------

sha256_of_file() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    else
        shasum -a 256 "$f" | awk '{print $1}'
    fi
}

# Summary hash of a sha256 manifest file (deterministic = sha256 of its bytes).
sha256_of_manifest() {
    local mf="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$mf" | awk '{print $1}'
    else
        shasum -a 256 "$mf" | awk '{print $1}'
    fi
}

# -----------------------------------------------------------------------------
# Path sanitizer for committable audit trails
# -----------------------------------------------------------------------------
# Replace personal scratch / home / users roots with `<server-scratch-root>` /
# `<server-home-root>` placeholders so `repeatability.txt` (which lands in
# git history) cannot leak an operator's account name. Idempotent: a path
# that already has placeholders, or one outside the known scratch trees,
# passes through unchanged. No-op on local Mac (HOME / scratch paths there
# don't match the patterns).
sanitize_path() {
    local p="$1"
    p="${p//\/scratch\/$USER/<server-scratch-root>}"
    p="${p//\/home\/$USER/<server-home-root>}"
    p="${p//\/users\/$USER/<server-home-root>}"
    echo "$p"
}

# -----------------------------------------------------------------------------
# Minimal YAML parser (purpose-built for our manifest shape)
# Extracts:
#   PROJECT_NAME          = project_name
#   HAS_LAKE              = has_lake (true/false)
#   OUTPUT_FILES[]        = output_compare.output_files (list of strings)
# Layout assumption matches benchmarks/<case>/manifest.yaml: 2-space indent,
# bullet list under `output_compare:` → `output_files:`. Comments and blank
# lines tolerated.
# -----------------------------------------------------------------------------

parse_manifest() {
    local f="$1"
    PROJECT_NAME=""
    HAS_LAKE=""
    OUTPUT_FILES=()

    local in_oc=0           # inside `output_compare:` block?
    local in_oc_files=0     # inside `  output_files:` list?
    local line key val

    while IFS= read -r line; do
        # strip trailing CR
        line="${line%$'\r'}"
        # strip inline comments (`# ...`) only when not inside quoted string.
        # Manifest doesn't use quoted `#` so safe to strip.
        case "$line" in
            *'#'*) line="${line%%#*}";;
        esac
        # rstrip
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue

        # Top-level keys: project_name / has_lake / output_compare:
        if [[ "$line" =~ ^project_name:[[:space:]]*\"?([^\"]*)\"?[[:space:]]*$ ]]; then
            PROJECT_NAME="${BASH_REMATCH[1]}"
            in_oc=0; in_oc_files=0
            continue
        fi
        if [[ "$line" =~ ^has_lake:[[:space:]]*([A-Za-z]+)[[:space:]]*$ ]]; then
            HAS_LAKE="${BASH_REMATCH[1]}"
            in_oc=0; in_oc_files=0
            continue
        fi
        if [[ "$line" == "output_compare:" ]]; then
            in_oc=1; in_oc_files=0
            continue
        fi

        if [[ "$in_oc" -eq 1 ]]; then
            # Detect a new top-level key (no leading space) -> exit output_compare
            if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*: ]]; then
                in_oc=0; in_oc_files=0
                # process this line as top-level by re-entering on next read?
                # we already consumed; just reset and let next iter handle it
                # but since project_name/has_lake/output_compare patterns
                # checked above only matched leading-no-space already, we
                # need to handle here too:
                if [[ "$line" =~ ^project_name:[[:space:]]*\"?([^\"]*)\"?[[:space:]]*$ ]]; then
                    PROJECT_NAME="${BASH_REMATCH[1]}"
                elif [[ "$line" =~ ^has_lake:[[:space:]]*([A-Za-z]+)[[:space:]]*$ ]]; then
                    HAS_LAKE="${BASH_REMATCH[1]}"
                fi
                continue
            fi

            # Inside output_compare block.
            if [[ "$line" =~ ^[[:space:]]+output_files: ]]; then
                in_oc_files=1
                continue
            fi
            # Any non-list sibling key under output_compare cancels the list mode
            if [[ "$line" =~ ^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*: ]]; then
                in_oc_files=0
                continue
            fi

            if [[ "$in_oc_files" -eq 1 ]]; then
                # Expect: `    - "output/<proj>.out/<name>.dat"`
                if [[ "$line" =~ ^[[:space:]]+-[[:space:]]*\"?([^\"]+)\"?[[:space:]]*$ ]]; then
                    OUTPUT_FILES+=("${BASH_REMATCH[1]}")
                fi
            fi
        fi
    done < "$f"

    if [[ -z "$PROJECT_NAME" ]]; then
        echo "error: parse_manifest: project_name not found in $f" >&2
        return 1
    fi
    if [[ -z "$HAS_LAKE" ]]; then
        echo "error: parse_manifest: has_lake not found in $f" >&2
        return 1
    fi
    if [[ "${#OUTPUT_FILES[@]}" -eq 0 ]]; then
        echo "error: parse_manifest: output_compare.output_files empty in $f" >&2
        return 1
    fi
}

# -----------------------------------------------------------------------------
# cfg.para 90-day truncation (CLAUDE.md policy)
#
# Files & idempotency:
#   - cfg.para path: SHUD/Basins/<case>/input/<project>/<project>.cfg.para
#   - On first invocation we save the current cfg.para to
#     `<project>.cfg.para.preB0archive` as a one-shot snapshot of "whatever
#     state cfg.para was in before this script ever touched it". Subsequent
#     invocations leave that snapshot alone (it is the restore point).
#   - END field is rewritten to START + 90 only if not already at that value.
# -----------------------------------------------------------------------------

truncate_cfg_para_to_90d() {
    local cfg="$1"
    local backup="${cfg}.preB0archive"

    if [[ ! -f "$cfg" ]]; then
        echo "error: cfg.para not found: $cfg" >&2
        return 1
    fi

    if [[ ! -f "$backup" ]]; then
        cp "$cfg" "$backup"
    fi

    local start end target
    start=$(awk '$1=="START" {print $2; exit}' "$cfg")
    end=$(awk '$1=="END"   {print $2; exit}'   "$cfg")
    if [[ -z "$start" || -z "$end" ]]; then
        echo "error: cfg.para missing START/END: $cfg" >&2
        return 1
    fi
    target=$((start + 90))

    if [[ "$end" -eq "$target" ]]; then
        echo "    cfg.para already truncated: START=$start END=$end (target=$target) — skip"
        return 0
    fi

    # In-place rewrite of the END line, preserving the leading 'END\t' format.
    # The cfg.para uses tab-separated `KEY\tVALUE`. Use awk to keep the tab.
    local tmp
    tmp="$(mktemp)"
    awk -v t="$target" 'BEGIN{OFS="\t"} {
        if ($1 == "END") { print $1, t }
        else             { print $0 }
    }' "$cfg" > "$tmp"
    mv "$tmp" "$cfg"

    echo "    cfg.para rewrote: START=$start END=$end -> $target (90d window)"
}

# -----------------------------------------------------------------------------
# SHUD run wrapper.
# Uses manifest convention: cd SHUD/Basins/<case> && ../../shud <project>
# (matches benchmarks/<case>/manifest.yaml `run_command` exactly).
# -----------------------------------------------------------------------------

run_shud() {
    local case_dir="$1"
    local project="$2"

    # Clear output dir first so we don't reuse stale binary dumps.
    rm -rf "$case_dir/output/${project}.out"

    local t0 t1
    t0="$(date +%s)"
    if ! ( cd "$case_dir" && "$SHUD_BIN" "$project" >/dev/null 2>&1 ); then
        echo "error: SHUD run failed: case=$CASE_NAME project=$project" >&2
        return 1
    fi
    t1="$(date +%s)"
    echo "$((t1 - t0))"
}

# -----------------------------------------------------------------------------
# SHA256 manifest for one run: lists hashes of each manifest output_file +
# cvode_stats.txt sibling. The sibling file lives at the same dir as the
# manifest output_files (output/<project>.out/cvode_stats.txt).
# Skips lake-named files when manifest has_lake=false.
# -----------------------------------------------------------------------------

write_sha_manifest() {
    local case_dir="$1"
    local manifest_file="$2"      # destination /tmp/<case>_run<N>.<rand>.sha256 (mktemp)

    local record_missing="${3:-0}"
    : > "$manifest_file"
    local f abs h has_lake_lc
    has_lake_lc="$(printf '%s' "$HAS_LAKE" | tr '[:upper:]' '[:lower:]')"
    # MISSING_FILES is a global accumulator. Only run 1 records into it
    # (record_missing=1); subsequent runs just rebuild the hash manifest.
    for f in "${OUTPUT_FILES[@]}"; do
        if [[ "$has_lake_lc" == "false" ]]; then
            # SHUD lake outputs follow the `<proj>.lak<channel>.dat` convention
            # (see SHUD/src/classes/IO.cpp:169-176). Anchor the skip pattern on
            # `.lak<letter>...dat` to avoid false positives on unrelated
            # channels containing the substring "lak" (e.g. `polakov.dat`,
            # `flakecount.dat`). PR #26 review follow-up.
            case "$f" in
                *.lak[a-z]*.dat) continue;;
            esac
        fi
        abs="$case_dir/$f"
        if [[ ! -f "$abs" ]]; then
            # Manifest lists this file but cfg.para DT_* disables it.
            # S0-8b documents this as a pre-existing deployment vs manifest gap
            # (NWM cases inherited DT_*=0 for most output channels). Track and
            # skip rather than abort.
            if [[ "$record_missing" -eq 1 ]]; then
                MISSING_FILES+=("$f")
            fi
            continue
        fi
        h="$(sha256_of_file "$abs")"
        printf '%s  %s\n' "$h" "$f" >> "$manifest_file"
    done

    # cvode_stats.txt sibling
    local cvode_stats="$case_dir/output/${PROJECT_NAME}.out/cvode_stats.txt"
    if [[ ! -f "$cvode_stats" ]]; then
        echo "error: cvode_stats.txt missing after run: $cvode_stats" >&2
        return 1
    fi
    h="$(sha256_of_file "$cvode_stats")"
    printf '%s  %s\n' "$h" "output/${PROJECT_NAME}.out/cvode_stats.txt" >> "$manifest_file"
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

echo "[archive_b0] case=$CASE_NAME runs=$NUM_RUNS"
echo "[archive_b0] parsing manifest: $MANIFEST"
parse_manifest "$MANIFEST"
echo "    project_name=$PROJECT_NAME has_lake=$HAS_LAKE n_output_files=${#OUTPUT_FILES[@]}"

CFG_PARA="$CASE_DIR/input/$PROJECT_NAME/${PROJECT_NAME}.cfg.para"
echo "[archive_b0] cfg.para 90d truncate: $CFG_PARA"
truncate_cfg_para_to_90d "$CFG_PARA"

# Tally walltimes & hash files
WALLS=()
HASH_FILES=()
SUM_HASHES=()
# Manifest output_files that DID NOT actually exist after the run. Caused by
# NWM-case cfg.para DT_* toggles being 0 (output disabled) while manifest lists
# the channel; not a script bug. Populated by write_sha_manifest and reset
# below before run 1.
MISSING_FILES=()

# Per F20 / F30 precedent (PR #54 round-3): replace predictable /tmp paths
# with mktemp's atomic O_CREAT|O_EXCL + 6-char random suffix to close the
# symlink-attack / pre-create TOCTOU window on shared hosts. We accumulate
# every allocated temp into TMP_FILES and clean them all up in a single
# EXIT trap so neither the success nor the failure path leaks scratch state.
TMP_FILES=()
# shellcheck disable=SC2317  # invoked via `trap ... EXIT`, not unreachable
cleanup_tmp_files() {
    if [[ "${#TMP_FILES[@]}" -gt 0 ]]; then
        rm -f "${TMP_FILES[@]}"
    fi
}
trap cleanup_tmp_files EXIT

for i in $(seq 1 "$NUM_RUNS"); do
    echo "[archive_b0] run $i / $NUM_RUNS ..."
    if ! WT="$(run_shud "$CASE_DIR" "$PROJECT_NAME")"; then
        exit 2
    fi
    if ! HF="$(mktemp "/tmp/${CASE_NAME}_run${i}.XXXXXX.sha256")"; then
        echo "error: mktemp failed for hash manifest (run $i)" >&2
        exit 2
    fi
    TMP_FILES+=("$HF")
    # Only run 1 populates MISSING_FILES; later runs just rebuild hashes.
    if [[ "$i" -eq 1 ]]; then
        MISSING_FILES=()
        if ! write_sha_manifest "$CASE_DIR" "$HF" 1; then
            exit 2
        fi
    else
        if ! write_sha_manifest "$CASE_DIR" "$HF" 0; then
            exit 2
        fi
    fi
    SUM="$(sha256_of_manifest "$HF")"
    WALLS+=("$WT")
    HASH_FILES+=("$HF")
    SUM_HASHES+=("$SUM")
    echo "    wall=${WT}s summary_sha256=$SUM ($HF)"
done

if [[ "${#MISSING_FILES[@]}" -gt 0 ]]; then
    echo "[archive_b0] WARN: ${#MISSING_FILES[@]} manifest output_files not produced by SHUD"
    echo "             (DT_* in cfg.para = 0; pre-existing deployment vs manifest gap):"
    for mf in "${MISSING_FILES[@]}"; do
        echo "    - $mf"
    done
fi

# -----------------------------------------------------------------------------
# Diff the N hash files: must all be byte-identical
# -----------------------------------------------------------------------------

echo "[archive_b0] verifying ${NUM_RUNS}-run SHA256 identity ..."
DIVERGED=0
for i in $(seq 2 "$NUM_RUNS"); do
    # mktemp + TMP_FILES + EXIT trap (see top-of-loop comment): closes the
    # /tmp/<case>_diff_1_<i>.txt TOCTOU/symlink window.
    if ! DIFF_FILE="$(mktemp "/tmp/${CASE_NAME}_diff_1_${i}.XXXXXX.txt")"; then
        echo "error: mktemp failed for diff dump (run $i)" >&2
        exit 2
    fi
    TMP_FILES+=("$DIFF_FILE")
    if ! diff -u "${HASH_FILES[0]}" "${HASH_FILES[i-1]}" >"$DIFF_FILE" 2>&1; then
        DIVERGED=1
        echo "    FAIL: run 1 vs run $i diverged — $DIFF_FILE:" >&2
        sed 's/^/      /' "$DIFF_FILE" >&2
    fi
done

if [[ "$DIVERGED" -ne 0 ]]; then
    echo "[archive_b0] repeatability FAILED for case=$CASE_NAME" >&2
    exit 3
fi

# -----------------------------------------------------------------------------
# Archive run 1 outputs + write repeatability.txt
# -----------------------------------------------------------------------------

echo "[archive_b0] all ${NUM_RUNS} runs identical; archiving to $ARCHIVE_DIR"
mkdir -p "$ARCHIVE_DIR"

HAS_LAKE_LC="$(printf '%s' "$HAS_LAKE" | tr '[:upper:]' '[:lower:]')"
for f in "${OUTPUT_FILES[@]}"; do
    if [[ "$HAS_LAKE_LC" == "false" ]]; then
        # See write_sha_manifest() above for the rationale.
        case "$f" in
            *.lak[a-z]*.dat) continue;;
        esac
    fi
    if [[ ! -f "$CASE_DIR/$f" ]]; then
        continue   # disabled by cfg.para DT_*; surfaced earlier in MISSING_FILES
    fi
    cp -f "$CASE_DIR/$f" "$ARCHIVE_DIR/$(basename "$f")"
done
cp -f "$CASE_DIR/output/${PROJECT_NAME}.out/cvode_stats.txt" "$ARCHIVE_DIR/cvode_stats.txt"

REPEAT_FILE="$ARCHIVE_DIR/repeatability.txt"
DATE_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
{
    echo "case: $CASE_NAME"
    echo "runs: $NUM_RUNS"
    for i in $(seq 1 "$NUM_RUNS"); do
        echo "sha256_run${i}: ${SUM_HASHES[i-1]}"
    done
    echo "verdict: PASS"
    echo "date: $DATE_ISO"
    echo "wall_times_sec: $(IFS=,; echo "${WALLS[*]}")"
    echo "binary: $(sanitize_path "$SHUD_BIN")"
    echo "binary_sha256: $(sha256_of_file "$SHUD_BIN")"
    echo "host: $(uname -srm)"
    echo "cfg_window_days: 90"
    if [[ "${#MISSING_FILES[@]}" -gt 0 ]]; then
        echo "missing_manifest_files: ${#MISSING_FILES[@]}"
        echo "missing_manifest_files_note: cfg.para DT_*=0 for these channels; pre-existing NWM deployment gap, see S0-4 / issue #11"
        for mf in "${MISSING_FILES[@]}"; do
            echo "missing_file: $mf"
        done
    else
        echo "missing_manifest_files: 0"
    fi
} > "$REPEAT_FILE"

echo "[archive_b0] DONE case=$CASE_NAME verdict=PASS"
echo "    archive: $ARCHIVE_DIR"
echo "    repeatability: $REPEAT_FILE"

exit 0
