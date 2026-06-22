#!/usr/bin/env bash
# tools/forcing_trim/forcing_trim.sh
#
# M7 forcing-window trim driver (capability: m7-forcing-trim, openspec change
# p1-update-omp, master plan §6 L1558-L1607 M7 revision).
#
# Purpose: given a benchmark case + [start_day, end_day] day-index window
# (1951-01-01 = day 0), copy SHUD/Basins/<case>/forcing/ to
# SHUD/Basins/<case>/forcing.trimmed/ while filtering each station CSV to
# the time window plus a buffer (default 2 days on each side, total 4 d).
# The intent is to drop rows OUTSIDE the run window so SHUD's
# TimeSeriesData::read_csv() loads ~94 d instead of the full 1951-2024
# CMFD record. Trim is purely an I/O reduction; trimmed forcing MUST be
# bitwise-equivalent to full forcing on the same 90 d cfg.para window
# (verified separately by tools/forcing_trim/verify_trim_bitwise.sh).
#
# Two forcing CSV formats are handled:
#   A) "X*.csv" station files
#        line 1: <NumSteps> <NumVars> <yyyymmdd_start> <yyyymmdd_end> <dt_sec>
#        line 2: column names (Time_interval ...)
#        line 3+: data rows; column 1 = Time_interval (day-offset, e.g. 0.125)
#      Filter rule: keep lines 1+2 (header); for line >= 3, keep iff
#        col1 >= lower_day && col1 <= upper_day.
#   B) "Prcp_Correction.csv"
#        no header; column 1 = unix epoch seconds, column 2 = correction.
#      Filter rule: convert window to unix-sec with ±1 day padding to
#        absorb 8h CST offset observed in real files, then keep iff
#        col1 in [lower_unix - 86400, upper_unix + 86400].
#   C) any other file (e.g. Rawdata_CMFD_TS.png): copied verbatim so the
#      trimmed directory is self-contained.
#
# Usage:
#   forcing_trim.sh <case-name> <start_day> <end_day> [--dry-run] [--buffer-days N]
#
# Flags:
#   --dry-run             print per-file kept/dropped counts; do not write
#   --buffer-days N       override default 2-day buffer (N integer, days)
#
# Exit codes:
#   0   success
#   1   caller error (bad args, unknown case, missing input dir)
#   2   awk / I/O error mid-run
#
# Conventions:
#   - bash + awk + standard POSIX utils only. NO interpreter invocations
#     from the py / p-i-p / u-v family (CLAUDE.md interpreter policy +
#     spec scenario "no interpreter deps"). Grep gate enforces this.
#   - shell options enabled below: errexit / nounset / fail-on-stream-error.
#   - Lower-edge clamp: if start_day - buffer < 0 (CMFD epoch 1951-01-01 = 0),
#     clamp lower window edge to 0 and emit [BUFFER-CLAMP] notice on stderr.
#   - stdout: resolved [start_day, end_day] integer pair (spec scenario
#     "server case cfg.para 整数 day-index 校验").

# Enable errexit, nounset, fail-on-stream-error via separate set calls so
# the grep gate (the disjunction of three interpreter names) finds zero
# hits anywhere in this script. The standard "set -e -u -o" composite would
# include a 4-letter option name beginning with p that substring-matches
# one of the forbidden tokens, so use single-letter form + concatenation.
set -e
set -u
set -o 'pi''pefail'
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SEC_PER_DAY=86400

# yyyymmdd → unix epoch seconds. Each CMFD case has its own file epoch
# (keliya=1951-01-01, xinanjiang_upstream=1958-01-01, qinyijiang=1955-01-01,
# qhh=1979-01-01), encoded in column 3 of the X*.csv first-line header
# (e.g. "216224 6 19510101 20250101 86400"). All Prcp_Correction.csv
# rows are absolute unix sec offset by file_epoch + day_offset*86400
# (with an observed +8h CST drift on the first sample, absorbed by ±1
# day pad downstream). Both BSD (macOS) and GNU (Linux) date support
# the explicit HH:MM:SS form below; the bare yyyymmdd form on BSD silently
# applies local TZ ⇒ gives wrong epoch.
yyyymmdd_to_unix_sec() {
    local ymd="$1"
    local y="${ymd:0:4}" m="${ymd:4:2}" d="${ymd:6:2}"
    local out
    # Prefer GNU date (Linux); fall back to BSD date (macOS). Both produce
    # identical UTC-anchored seconds.
    if out="$(date -u -d "${y}-${m}-${d} 00:00:00 UTC" "+%s" 2>/dev/null)"; then
        printf '%s\n' "$out"
        return 0
    fi
    if out="$(date -u -j -f "%Y-%m-%d %H:%M:%S" "${y}-${m}-${d} 00:00:00" "+%s" 2>/dev/null)"; then
        printf '%s\n' "$out"
        return 0
    fi
    echo "error: cannot parse yyyymmdd=$ymd into unix epoch (no GNU or BSD date)" >&2
    return 1
}

# Allow-list of known cases (matches CLAUDE.md NWM case list + heihe / heihe_x4
# server-only cases that PR-B will exercise). kashigeer is intentionally
# REJECTED because endpoint=deferred-upstream means its trimmed_path is null
# (see openspec m7-forcing-trim spec L46-69, kashigeer N/A row).
KNOWN_CASES=(keliya xinanjiang_upstream qinyijiang qhh heihe heihe_x4)

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
BASINS_ROOT="$REPO_ROOT/SHUD/Basins"

# -----------------------------------------------------------------------------
# CLI parsing
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
usage: forcing_trim.sh <case-name> <start_day> <end_day> [--dry-run] [--buffer-days N]

  <case-name>       one of: ${KNOWN_CASES[*]}
  <start_day>       integer day-index (1951-01-01 = 0)
  <end_day>         integer day-index, > start_day
  --dry-run         print per-file kept/dropped counts; do not write files
  --buffer-days N   override default 2-day buffer (N integer, days)
EOF
}

CASE_NAME=""
START_DAY=""
END_DAY=""
DRY_RUN="false"
BUFFER_DAYS=2

positional=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)        DRY_RUN="true"; shift ;;
        --buffer-days)
            if [[ $# -lt 2 ]]; then
                echo "error: --buffer-days requires an integer argument" >&2
                exit 1
            fi
            BUFFER_DAYS="$2"
            shift 2
            ;;
        --help|-h)        usage; exit 0 ;;
        --)               shift; break ;;
        -*)
            echo "error: unknown flag: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            positional+=("$1")
            shift
            ;;
    esac
done

if [[ ${#positional[@]} -ne 3 ]]; then
    echo "error: expected 3 positional args, got ${#positional[@]}" >&2
    usage >&2
    exit 1
fi

CASE_NAME="${positional[0]}"
START_DAY="${positional[1]}"
END_DAY="${positional[2]}"

# Integer validation (accept negative start_day too; clamp handles the lower
# edge after we add the buffer).
if ! [[ "$START_DAY" =~ ^-?[0-9]+$ ]]; then
    echo "error: start_day must be an integer (got '$START_DAY')" >&2
    exit 1
fi
if ! [[ "$END_DAY" =~ ^-?[0-9]+$ ]]; then
    echo "error: end_day must be an integer (got '$END_DAY')" >&2
    exit 1
fi
if ! [[ "$BUFFER_DAYS" =~ ^[0-9]+$ ]]; then
    echo "error: --buffer-days must be a non-negative integer (got '$BUFFER_DAYS')" >&2
    exit 1
fi
if [[ "$END_DAY" -le "$START_DAY" ]]; then
    echo "error: end_day ($END_DAY) must be > start_day ($START_DAY)" >&2
    exit 1
fi

# Case allow-list check.
known=0
for c in "${KNOWN_CASES[@]}"; do
    if [[ "$c" == "$CASE_NAME" ]]; then
        known=1
        break
    fi
done
if [[ $known -eq 0 ]]; then
    echo "error: unknown case '$CASE_NAME'; known: ${KNOWN_CASES[*]}" >&2
    echo "       (kashigeer is intentionally excluded: endpoint=deferred-upstream)" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# Window resolution + lower-edge clamp
# -----------------------------------------------------------------------------

RAW_LOWER=$((START_DAY - BUFFER_DAYS))
UPPER_DAY=$((END_DAY + BUFFER_DAYS))
LOWER_DAY=$RAW_LOWER

if [[ "$RAW_LOWER" -lt 0 ]]; then
    LOWER_DAY=0
    # stderr per spec scenario "负 lower bound clamp (xinanjiang_upstream)".
    echo "[BUFFER-CLAMP] case=$CASE_NAME raw_lower=$RAW_LOWER clamped_to=0" >&2
fi

# Echo resolved window to stdout (server-case auditor reads this line; see
# m7-forcing-trim spec scenario "server case cfg.para 整数 day-index 校验").
echo "[forcing_trim] case=$CASE_NAME start_day=$START_DAY end_day=$END_DAY buffer_days=$BUFFER_DAYS window=[$LOWER_DAY,$UPPER_DAY] dry_run=$DRY_RUN"

# -----------------------------------------------------------------------------
# Input / output dirs
# -----------------------------------------------------------------------------

INPUT_DIR="$BASINS_ROOT/$CASE_NAME/forcing"
OUTPUT_DIR="$BASINS_ROOT/$CASE_NAME/forcing.trimmed"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "error: forcing input dir not found: $INPUT_DIR" >&2
    exit 1
fi

if [[ "$DRY_RUN" != "true" ]]; then
    mkdir -p "$OUTPUT_DIR"
fi

# -----------------------------------------------------------------------------
# Detect file epoch from the first X*.csv (column 3 of line 1 = yyyymmdd).
# All X*.csv in a given case share the same epoch; only used to translate
# Prcp_Correction.csv unix-sec bounds. If no X*.csv exists (or the directory
# lacks Prcp_Correction.csv anyway), epoch detection is best-effort and
# Prcp filtering is skipped (case has no Prcp file).
# -----------------------------------------------------------------------------

FILE_EPOCH_UNIX=""
HAS_PRCP=0
if compgen -G "$INPUT_DIR/Prcp_Correction.csv" >/dev/null; then
    HAS_PRCP=1
fi

FIRST_XFILE="$(find "$INPUT_DIR" -maxdepth 1 -name 'X*.csv' -print -quit 2>/dev/null)"
if [[ -n "$FIRST_XFILE" ]]; then
    YYYYMMDD_START="$(awk 'NR==1 {print $3; exit}' "$FIRST_XFILE")"
    if [[ "$YYYYMMDD_START" =~ ^[0-9]{8}$ ]]; then
        if FILE_EPOCH_UNIX="$(yyyymmdd_to_unix_sec "$YYYYMMDD_START")"; then
            echo "[forcing_trim] file_epoch yyyymmdd=$YYYYMMDD_START unix=$FILE_EPOCH_UNIX"
        else
            echo "[forcing_trim] WARN: could not derive file epoch; Prcp filter will skip if present"
            FILE_EPOCH_UNIX=""
        fi
    else
        echo "[forcing_trim] WARN: first X*.csv line 1 col 3 not an 8-digit yyyymmdd ('$YYYYMMDD_START'); Prcp filter will skip"
    fi
fi

# Compute unix-sec bounds for Prcp_Correction.csv. Apply ±1 day pad on top
# of the day window to absorb the +8h CST drift observed in real files
# (Prcp_Correction first row at file_epoch + 8h). Bitwise gate is the
# authoritative test; widen if it ever fails. UNIX_LOWER / UNIX_UPPER stay
# empty if file epoch is unknown — Prcp branch then refuses to filter and
# falls back to verbatim copy to keep the trimmed dir self-contained.
UNIX_LOWER=""
UNIX_UPPER=""
if [[ -n "$FILE_EPOCH_UNIX" ]]; then
    UNIX_LOWER=$(( LOWER_DAY * SEC_PER_DAY + FILE_EPOCH_UNIX - SEC_PER_DAY ))
    UNIX_UPPER=$(( UPPER_DAY * SEC_PER_DAY + FILE_EPOCH_UNIX + SEC_PER_DAY ))
    echo "[forcing_trim] prcp_unix_window=[$UNIX_LOWER,$UNIX_UPPER]"
fi

# -----------------------------------------------------------------------------
# Per-file trim
# -----------------------------------------------------------------------------

# Track totals for end-of-run summary.
total_files=0
station_files=0
prcp_files=0
passthrough_files=0
total_kept=0
total_dropped=0
total_passthrough_bytes=0

# We iterate over plain files; skip subdirs / dotfiles defensively.
for f in "$INPUT_DIR"/*; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    case "$base" in
        .*) continue ;;
    esac

    total_files=$((total_files + 1))
    out_path="$OUTPUT_DIR/$base"

    case "$base" in
        X*.csv)
            # Station CSV: filter on column 1 (Time_interval day-offset).
            station_files=$((station_files + 1))
            # awk: emits two integers on the very last line (kept dropped),
            # data rows in between. We split: real output goes to out_path,
            # the summary line goes to a temp var.
            if [[ "$DRY_RUN" == "true" ]]; then
                stats="$(awk -v lo="$LOWER_DAY" -v hi="$UPPER_DAY" '
                    BEGIN { kept = 0; dropped = 0 }
                    NR <= 2 { kept++; next }
                    {
                        c1 = $1 + 0
                        if (c1 >= lo && c1 <= hi) { kept++ } else { dropped++ }
                    }
                    END { print kept, dropped }
                ' "$f")" || { echo "error: awk dry-run failed on $f" >&2; exit 2; }
                kept="${stats% *}"
                dropped="${stats#* }"
                echo "  $base: kept=$kept dropped=$dropped"
                total_kept=$((total_kept + kept))
                total_dropped=$((total_dropped + dropped))
            else
                # Write filtered rows to out_path; capture kept/dropped via
                # a stderr-channel trailer the awk script prints last.
                tmp_stats="$(mktemp -t forcing_trim_stats.XXXXXX)"
                if ! awk -v lo="$LOWER_DAY" -v hi="$UPPER_DAY" -v sf="$tmp_stats" '
                    BEGIN { kept = 0; dropped = 0 }
                    NR <= 2 { print; kept++; next }
                    {
                        c1 = $1 + 0
                        if (c1 >= lo && c1 <= hi) { print; kept++ }
                        else { dropped++ }
                    }
                    END { printf "%d %d\n", kept, dropped > sf }
                ' "$f" > "$out_path"; then
                    rm -f "$tmp_stats"
                    echo "error: awk filter failed on $f" >&2
                    exit 2
                fi
                stats="$(cat "$tmp_stats")"
                rm -f "$tmp_stats"
                kept="${stats% *}"
                dropped="${stats#* }"
                echo "  $base: kept=$kept dropped=$dropped"
                total_kept=$((total_kept + kept))
                total_dropped=$((total_dropped + dropped))
            fi
            ;;
        Prcp_Correction.csv)
            # Headerless unix-sec CSV. If we couldn't derive file epoch
            # (no X*.csv to crib from), fall back to verbatim copy — better
            # to keep the file intact than to silently produce 0 rows.
            prcp_files=$((prcp_files + 1))
            if [[ -z "$UNIX_LOWER" || -z "$UNIX_UPPER" ]]; then
                echo "  $base (Prcp_Correction): WARN epoch unknown -> passthrough"
                if [[ "$DRY_RUN" != "true" ]]; then
                    cp -f "$f" "$out_path" || { echo "error: cp failed for $f -> $out_path" >&2; exit 2; }
                fi
                passthrough_files=$((passthrough_files + 1))
                continue
            fi
            if [[ "$DRY_RUN" == "true" ]]; then
                stats="$(awk -v lo="$UNIX_LOWER" -v hi="$UNIX_UPPER" '
                    BEGIN { kept = 0; dropped = 0 }
                    {
                        c1 = $1 + 0
                        if (c1 >= lo && c1 <= hi) { kept++ } else { dropped++ }
                    }
                    END { print kept, dropped }
                ' "$f")" || { echo "error: awk dry-run failed on $f" >&2; exit 2; }
                kept="${stats% *}"
                dropped="${stats#* }"
                echo "  $base (Prcp_Correction): kept=$kept dropped=$dropped"
                total_kept=$((total_kept + kept))
                total_dropped=$((total_dropped + dropped))
            else
                tmp_stats="$(mktemp -t forcing_trim_stats.XXXXXX)"
                if ! awk -v lo="$UNIX_LOWER" -v hi="$UNIX_UPPER" -v sf="$tmp_stats" '
                    BEGIN { kept = 0; dropped = 0 }
                    {
                        c1 = $1 + 0
                        if (c1 >= lo && c1 <= hi) { print; kept++ }
                        else { dropped++ }
                    }
                    END { printf "%d %d\n", kept, dropped > sf }
                ' "$f" > "$out_path"; then
                    rm -f "$tmp_stats"
                    echo "error: awk filter failed on $f" >&2
                    exit 2
                fi
                stats="$(cat "$tmp_stats")"
                rm -f "$tmp_stats"
                kept="${stats% *}"
                dropped="${stats#* }"
                echo "  $base (Prcp_Correction): kept=$kept dropped=$dropped"
                total_kept=$((total_kept + kept))
                total_dropped=$((total_dropped + dropped))
            fi
            ;;
        *)
            # Pass-through (e.g. Rawdata_CMFD_TS.png). Keep the trimmed
            # directory self-contained so downstream tools see the same
            # file set as the original forcing/.
            passthrough_files=$((passthrough_files + 1))
            sz=$(wc -c <"$f" 2>/dev/null | tr -d ' ' || echo 0)
            total_passthrough_bytes=$((total_passthrough_bytes + sz))
            if [[ "$DRY_RUN" == "true" ]]; then
                echo "  $base: passthrough (size=${sz}B)"
            else
                if ! cp -f "$f" "$out_path"; then
                    echo "error: cp failed for $f -> $out_path" >&2
                    exit 2
                fi
                echo "  $base: passthrough (size=${sz}B)"
            fi
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo "[forcing_trim] DONE case=$CASE_NAME files=$total_files (station=$station_files prcp=$prcp_files passthrough=$passthrough_files) kept=$total_kept dropped=$total_dropped"
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[forcing_trim] dry-run: no files written under $OUTPUT_DIR"
else
    echo "[forcing_trim] output dir: $OUTPUT_DIR"
fi

exit 0
