#!/usr/bin/env bash
# tools/p1e_cv_y_hash/p1e_cv_y_hash.sh
#
# P1e PR-B (issue #310) — CV_Y state vector hash CLI.
#
# Reads cv_y_<tout>.bin files produced by SHUD's SHUD_DUMP_CV_Y=1
# env-gated dump (see SHUD/src/Model/shud.cpp post-summary block) and
# emits one line per timestep with:
#
#     <tout>  <sha256>
#
# Used to verify cross-build (mode A vs B vs C vs D) solver-state
# byte-equality at every tout boundary. Distinct from the rivqdown.dat
# SHA256 channel (which reflects derived output): cv_y is the solver
# state vector Y at tout, before any output-time recompute.
#
# Usage:
#     tools/p1e_cv_y_hash/p1e_cv_y_hash.sh <dir-with-cv_y-files> \
#         [--expected <expected.txt>]
#
# Arguments:
#   <dir>             Path to a directory containing cv_y_*.bin files.
#                     Typically:
#                       SHUD/Basins/<case>/output/<project>.out/
#                     when the run was invoked with SHUD_DUMP_CV_Y=1 and
#                     --keep was passed to tools/p1e_2x2_runner.sh, OR
#                     the .p1e-runs/2x2/<cell>/ directory if cell-local
#                     keep is used.
#
# Options:
#   --expected <file> Compare line-for-line against an expected file
#                     (same format: <tout>  <sha>). Exit 1 if any line
#                     differs, 0 if all match.
#   --help, -h        This message.
#
# Output format:
#   stdout         "<tout-as-printed>  <sha256>" (two spaces between tout
#                  and sha, matching `sha256sum` / `shasum -a 256` style)
#                  sorted by filename (natural lexicographic order since
#                  tout is zero-padded to 20.6f by the SHUD dump)
#
# Exit codes:
#   0  success (and, if --expected given, all matches)
#   1  caller error (bad args / dir not found / no cv_y_*.bin files)
#   2  --expected mismatch (lines diverged)

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

usage() {
    cat <<EOF
Usage: $0 <dir-with-cv_y-files> [--expected <expected.txt>]

P1e PR-B CV_Y state vector hash CLI. Emits "<tout>  <sha256>" per cv_y_*.bin.

Arguments:
  <dir>             Directory containing cv_y_*.bin files (typically
                    SHUD/Basins/<case>/output/<project>.out/ after a run
                    with SHUD_DUMP_CV_Y=1).

Options:
  --expected <file> Compare against an expected hash file (same format).
                    Exit 2 if any line diverges, 0 if all match.
  --help, -h        This message.

Output (stdout, one per line, sorted by filename):
  <tout-as-printed>  <sha256>

Exit codes:
  0  success (and all expected matches if --expected given)
  1  caller error
  2  --expected mismatch
EOF
}

DIR=""
EXPECTED=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --expected)
            EXPECTED="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "error: unknown option '$1'" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [[ -z "$DIR" ]]; then
                DIR="$1"
                shift
            else
                echo "error: unexpected positional arg '$1'" >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$DIR" ]]; then
    echo "error: <dir> required" >&2
    usage >&2
    exit 1
fi
if [[ ! -d "$DIR" ]]; then
    echo "error: not a directory: $DIR" >&2
    exit 1
fi

# Portable SHA256 wrapper (reused from tools/archive_b0_output.sh)
sha256_of_file() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    else
        shasum -a 256 "$f" | awk '{print $1}'
    fi
}

# Enumerate cv_y_*.bin sorted by name (filenames are zero-padded so sort
# is also natural numeric on tout).
shopt -s nullglob
files=("$DIR"/cv_y_*.bin)
shopt -u nullglob

if [[ "${#files[@]}" -eq 0 ]]; then
    echo "error: no cv_y_*.bin files in $DIR" >&2
    echo "       hint: run SHUD with SHUD_DUMP_CV_Y=1 (per SHUD/src/Model/shud.cpp L196-220 env-gated dump)" >&2
    exit 1
fi

IFS=$'\n' files=($(printf '%s\n' "${files[@]}" | sort))
unset IFS

# Emit to stdout (or accumulate for --expected compare)
TMP_ACTUAL=""
if [[ -n "$EXPECTED" ]]; then
    TMP_ACTUAL="$(mktemp)"
    # trap to clean up tmp on any exit
    trap 'rm -f "$TMP_ACTUAL"' EXIT
fi

for f in "${files[@]}"; do
    bn="$(basename "$f")"
    tout="${bn#cv_y_}"
    tout="${tout%.bin}"
    sha="$(sha256_of_file "$f")"
    line="$(printf '%s  %s\n' "$tout" "$sha")"
    if [[ -n "$EXPECTED" ]]; then
        printf '%s\n' "$line" >> "$TMP_ACTUAL"
    fi
    printf '%s\n' "$line"
done

# Compare against expected if requested
if [[ -n "$EXPECTED" ]]; then
    if [[ ! -f "$EXPECTED" ]]; then
        echo "error: --expected file not found: $EXPECTED" >&2
        exit 1
    fi
    if diff -u "$EXPECTED" "$TMP_ACTUAL" >/dev/null 2>&1; then
        echo "# p1e_cv_y_hash: MATCH (${#files[@]} timesteps vs $EXPECTED)" >&2
        exit 0
    else
        echo "# p1e_cv_y_hash: MISMATCH vs $EXPECTED" >&2
        diff -u "$EXPECTED" "$TMP_ACTUAL" >&2 || true
        exit 2
    fi
fi

exit 0
