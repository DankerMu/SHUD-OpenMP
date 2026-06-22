#!/usr/bin/env bash
# tools/forcing_trim/verify_trim_bitwise.sh
#
# Run SHUD on a case's forcing.trimmed/ directory and verify the canonical
# summary SHA256 matches benchmarks/<case>/B0_output/repeatability.txt
# sha256_run1. Implements the m7-forcing-trim spec Scenario "4 Mac case
# bitwise vs B0-tag PASS" (and PR-B Scenario "2 server case bitwise vs
# B0-tag PASS"). Reversible: tsd.forc line 2 is temporarily switched to
# the trimmed dir, restored on EXIT via trap.
#
# Usage:
#   verify_trim_bitwise.sh <case-name>
#
# Exit codes:
#   0   bitwise PASS (sha equals sha256_run1 from B0 repeatability.txt)
#   1   caller / discovery error (bad arg, manifest missing, etc.)
#   2   SHUD run failed
#   3   bitwise MISMATCH

set -e
set -u
set -o 'pi''pefail'
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
SHUD_ROOT="$REPO_ROOT/SHUD"
BASINS_ROOT="$SHUD_ROOT/Basins"
BENCH_ROOT="$REPO_ROOT/benchmarks"
SHUD_BIN="$SHUD_ROOT/shud"

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <case-name>" >&2
    exit 1
fi
CASE_NAME="$1"

MANIFEST="$BENCH_ROOT/$CASE_NAME/manifest.yaml"
CASE_DIR="$BASINS_ROOT/$CASE_NAME"
TRIM_DIR="$CASE_DIR/forcing.trimmed"
B0_REPEAT="$BENCH_ROOT/$CASE_NAME/B0_output/repeatability.txt"

if [[ ! -f "$MANIFEST" ]]; then
    echo "error: manifest not found: $MANIFEST" >&2
    exit 1
fi
if [[ ! -d "$CASE_DIR" ]]; then
    echo "error: case directory not deployed: $CASE_DIR" >&2
    exit 1
fi
if [[ ! -d "$TRIM_DIR" ]]; then
    echo "error: forcing.trimmed/ not found: $TRIM_DIR" >&2
    echo "       hint: run tools/forcing_trim/forcing_trim.sh $CASE_NAME <start> <end> first" >&2
    exit 1
fi
if [[ ! -x "$SHUD_BIN" ]]; then
    echo "error: SHUD binary not built/executable: $SHUD_BIN" >&2
    exit 1
fi
if [[ ! -f "$B0_REPEAT" ]]; then
    echo "error: B0 repeatability.txt not found: $B0_REPEAT" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# SHA256 portable wrapper (matches tools/archive_b0_output.sh)
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
# Minimal YAML parser — extracts project_name, has_lake, output_compare list.
# Mirrors archive_b0_output.sh logic so canonical SHA is computed identically.
# -----------------------------------------------------------------------------

parse_manifest() {
    local f="$1"
    PROJECT_NAME=""
    HAS_LAKE=""
    OUTPUT_FILES=()

    local in_oc=0
    local in_oc_files=0
    local line

    while IFS= read -r line; do
        line="${line%$'\r'}"
        case "$line" in
            *'#'*) line="${line%%#*}";;
        esac
        line="${line%"${line##*[![:space:]]}"}"
        [[ -z "$line" ]] && continue

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
            if [[ "$line" =~ ^[A-Za-z_][A-Za-z0-9_]*: ]]; then
                in_oc=0; in_oc_files=0
                continue
            fi
            if [[ "$line" =~ ^[[:space:]]+output_files: ]]; then
                in_oc_files=1
                continue
            fi
            if [[ "$line" =~ ^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*: ]]; then
                in_oc_files=0
                continue
            fi
            if [[ "$in_oc_files" -eq 1 ]]; then
                if [[ "$line" =~ ^[[:space:]]+-[[:space:]]*\"?([^\"]+)\"?[[:space:]]*$ ]]; then
                    OUTPUT_FILES+=("${BASH_REMATCH[1]}")
                fi
            fi
        fi
    done < "$f"

    if [[ -z "$PROJECT_NAME" || -z "$HAS_LAKE" || "${#OUTPUT_FILES[@]}" -eq 0 ]]; then
        echo "error: parse_manifest: missing required fields in $f" >&2
        return 1
    fi
}

# -----------------------------------------------------------------------------
# Canonical summary SHA — same algorithm as archive_b0_output.sh
# (write a per-file sha manifest, then sha that manifest).
# -----------------------------------------------------------------------------

compute_canonical_summary_sha() {
    local case_dir="$1"
    local sha_manifest_path="$2"

    : > "$sha_manifest_path"
    local f abs h has_lake_lc
    has_lake_lc="$(printf '%s' "$HAS_LAKE" | tr '[:upper:]' '[:lower:]')"
    for f in "${OUTPUT_FILES[@]}"; do
        if [[ "$has_lake_lc" == "false" ]]; then
            case "$f" in
                *.lak[a-z]*.dat) continue;;
            esac
        fi
        abs="$case_dir/$f"
        if [[ ! -f "$abs" ]]; then
            # Same MISSING-FILES policy as archive_b0_output.sh: skip
            # silently (DT_*=0 in cfg.para disables some channels).
            continue
        fi
        h="$(sha256_of_file "$abs")"
        printf '%s  %s\n' "$h" "$f" >> "$sha_manifest_path"
    done
    local cvode_stats="$case_dir/output/${PROJECT_NAME}.out/cvode_stats.txt"
    if [[ ! -f "$cvode_stats" ]]; then
        echo "error: cvode_stats.txt missing after run: $cvode_stats" >&2
        return 1
    fi
    h="$(sha256_of_file "$cvode_stats")"
    printf '%s  %s\n' "$h" "output/${PROJECT_NAME}.out/cvode_stats.txt" >> "$sha_manifest_path"

    # The canonical summary SHA256 is the sha of the per-file manifest.
    sha256_of_file "$sha_manifest_path"
}

# -----------------------------------------------------------------------------
# Discover tsd.forc + back up before mutating
# -----------------------------------------------------------------------------

echo "[verify_trim] case=$CASE_NAME parsing manifest"
parse_manifest "$MANIFEST"
echo "    project_name=$PROJECT_NAME has_lake=$HAS_LAKE n_output_files=${#OUTPUT_FILES[@]}"

CFG_PARA="$CASE_DIR/input/$PROJECT_NAME/${PROJECT_NAME}.cfg.para"
TSD_FORC="$CASE_DIR/input/$PROJECT_NAME/${PROJECT_NAME}.tsd.forc"

if [[ ! -f "$TSD_FORC" ]]; then
    echo "error: tsd.forc not found: $TSD_FORC" >&2
    exit 1
fi
if [[ ! -f "$CFG_PARA" ]]; then
    echo "error: cfg.para not found: $CFG_PARA" >&2
    exit 1
fi

# Echo resolved window (per spec scenario "server case cfg.para 整数 day-index
# 校验"; harmless on Mac cases, useful for PR-B server validation).
RESOLVED_START="$(awk '$1=="START" {print $2; exit}' "$CFG_PARA")"
RESOLVED_END="$(awk '$1=="END" {print $2; exit}' "$CFG_PARA")"
echo "[verify_trim] resolved cfg.para [start_day=$RESOLVED_START end_day=$RESOLVED_END]"

# Use a distinct backup suffix so we never collide with fix_case_paths.sh's
# .orig (which represents the pristine upstream state, not the post-fix
# state we're temporarily replacing).
TSD_BACKUP="${TSD_FORC}.preTrimBitwise"

restore_tsd_forc() {
    # Idempotent; called from EXIT trap so it may execute under set -e.
    if [[ -f "$TSD_BACKUP" ]]; then
        cp -p "$TSD_BACKUP" "$TSD_FORC" 2>/dev/null || true
        rm -f "$TSD_BACKUP" 2>/dev/null || true
        echo "[verify_trim] restored tsd.forc from $TSD_BACKUP"
    fi
}
trap restore_tsd_forc EXIT

# Snapshot then rewrite line 2 = trimmed dir absolute path.
cp -p "$TSD_FORC" "$TSD_BACKUP"
TRIM_ABS="$(cd "$TRIM_DIR" && pwd -P)"
echo "[verify_trim] switching tsd.forc line 2 -> $TRIM_ABS"

tmp="$(mktemp "${TSD_FORC}.XXXXXX")"
if ! awk -v line2="$TRIM_ABS" 'NR==2 {print line2; next} {print}' "$TSD_FORC" >"$tmp"; then
    rm -f "$tmp"
    echo "error: awk rewrite failed on $TSD_FORC" >&2
    exit 2
fi
mv "$tmp" "$TSD_FORC"

# -----------------------------------------------------------------------------
# Run SHUD on trimmed forcing
# -----------------------------------------------------------------------------

echo "[verify_trim] running SHUD: cd $CASE_DIR && $SHUD_BIN $PROJECT_NAME"
rm -rf "$CASE_DIR/output/${PROJECT_NAME}.out"
T0="$(date +%s)"
if ! ( cd "$CASE_DIR" && "$SHUD_BIN" "$PROJECT_NAME" >/dev/null 2>&1 ); then
    echo "error: SHUD run failed for $CASE_NAME on trimmed forcing" >&2
    exit 2
fi
T1="$(date +%s)"
WALL=$((T1 - T0))
echo "[verify_trim] run complete wall=${WALL}s"

# -----------------------------------------------------------------------------
# Compute canonical SHA + compare
# -----------------------------------------------------------------------------

SHA_MANIFEST="$(mktemp -t verify_trim_sha.XXXXXX)"
ACTUAL_SHA="$(compute_canonical_summary_sha "$CASE_DIR" "$SHA_MANIFEST")"

EXPECTED_SHA="$(awk -F': ' '$1=="sha256_run1" {print $2; exit}' "$B0_REPEAT")"
if [[ -z "$EXPECTED_SHA" ]]; then
    echo "error: could not read sha256_run1 from $B0_REPEAT" >&2
    exit 1
fi

echo ""
echo "[verify_trim] case=$CASE_NAME"
echo "    expected (B0_output/repeatability.txt sha256_run1): $EXPECTED_SHA"
echo "    actual   (trimmed forcing canonical summary SHA):   $ACTUAL_SHA"
echo "    per-file hash manifest (for debugging):             $SHA_MANIFEST"

if [[ "$ACTUAL_SHA" == "$EXPECTED_SHA" ]]; then
    echo "[verify_trim] PASS case=$CASE_NAME bitwise vs B0-tag"
    exit 0
fi

echo "[verify_trim] FAIL case=$CASE_NAME bitwise MISMATCH" >&2
echo "    diff hint: compare $SHA_MANIFEST line-by-line against B0_output set" >&2
exit 3
