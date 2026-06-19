#!/usr/bin/env bash
# tools/check_invariant_sweep.sh — repo-wide invariant grep meta-tool.
#
# Reads tools/invariants.yaml (small purpose-built YAML; parsed with awk +
# bash to avoid pulling in PyYAML / yq), iterates each invariant, expands
# its `expected_absent_in` globs, and asserts the `token` does NOT appear
# in any non-comment line of any matched file. Emits `[OK]` per invariant
# on PASS and `[MISMATCH]` on FAIL (with offending file:line dumped).
#
# Comment-line stripping: this tool intentionally ignores `^[[:space:]]*#`
# leading-hash lines (covers .sh / .yml / .yaml / .py / .md headers and
# bash-style # comments). C-style `//` / `/* */` comments are NOT stripped;
# scope your token narrowly enough to avoid false-positives, or exclude
# C/C++ source files from `expected_absent_in`. The `# ` filter is the
# pragmatic 80% solution for the tools/ + .github/ + .md surface area
# this gate covers.
#
# Usage:
#   bash tools/check_invariant_sweep.sh                 # check all invariants
#   bash tools/check_invariant_sweep.sh <invariant-id>  # check one invariant
#
# Exit codes:
#   0  all invariants PASS
#   1  one or more invariants MISMATCH (offending hits printed)
#   2  usage error / invariants.yaml missing / parse error
#
# Owned by: PR <p3-followups-tooling> (issue #56 Part B).

set -eu

PROG=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/.." && pwd)
SPEC="$REPO/tools/invariants.yaml"

usage() {
    cat <<EOF >&2
Usage: $PROG [<invariant-id>]

Checks every invariant defined in tools/invariants.yaml. If <invariant-id>
is given, only that invariant is checked.

Exit codes:
  0  all checked invariants PASS
  1  one or more invariants MISMATCH
  2  usage error / spec problem
EOF
    exit 2
}

[ -r "$SPEC" ] || { printf "%s: cannot read %s\n" "$PROG" "$SPEC" >&2; exit 2; }

FILTER_ID="${1:-}"
[ "${FILTER_ID:-}" = "--help" ] || [ "${FILTER_ID:-}" = "-h" ] && usage

# -----------------------------------------------------------------------------
# Minimal YAML parser (purpose-built for this file's exact shape).
#
# Layout we accept:
#   invariants:
#     - id: <slug>
#       token: '<literal>'                # OR hex_token: '<lowercase-hex>'
#       hex_token: '<lowercase-hex>'      # OR token: '<literal>' (XOR; exactly one)
#       regex: false|true                 # optional, default false
#       expected_absent_in:
#         - '<glob>'
#         - '<glob>'
#       rationale: |
#         <free text>
#
# hex_token rationale: for PII / endpoint / credentials-adjacent invariants,
# embedding the literal token here would itself violate the gate (the spec
# is a tracked file). hex_token sidesteps that by carrying an `xxd -r -p`-
# decodable hex string; the decoded literal is materialized only in memory
# at scan time. See `pii_server_endpoint_absence` in tools/invariants.yaml.
#
# State machine over single-pass read; rationale block is consumed-and-discarded
# (the tool emits only OK/MISMATCH, not the rationale).
#
# Emits to stdout, one record per invariant, format:
#   <id>|<regex>|<token>|<hex_token>|<glob1>,<glob2>,...
# (pipe-separated; globs comma-joined). Caller splits on this. Exactly one
# of <token> / <hex_token> is populated per record.
# -----------------------------------------------------------------------------

parse_invariants() {
    awk '
        BEGIN {
            in_inv = 0       # inside "invariants:" block
            in_one = 0       # inside one - id: ... record
            in_paths = 0     # inside expected_absent_in list
            in_rationale = 0 # inside rationale: | block (skip lines)
            id = ""; token = ""; hex_token = ""; regex = "false"; paths = ""
        }

        function flush() {
            if (id != "") {
                # Emit 5 fields: id | regex | token | hex_token | paths_csv
                # Consumer decodes hex_token (if non-empty) to derive the
                # effective token, so the spec file itself never has to carry
                # the literal PII / secret string.
                print id "|" regex "|" token "|" hex_token "|" paths
            }
            id = ""; token = ""; hex_token = ""; regex = "false"; paths = ""
            in_paths = 0
            in_rationale = 0
        }

        # Strip CRLF
        { sub(/\r$/, "") }

        # Skip empty lines + full-line comments
        /^[[:space:]]*$/ { next }
        /^[[:space:]]*#/ { next }

        # rationale: | swallow until next top-level list entry / key
        in_rationale && /^[[:space:]]{6,}[^[:space:]]/ { next }
        in_rationale {
            # exits rationale on any line with indentation <= 4 spaces (i.e.
            # a new record or sibling key at -id level which is 4 spaces)
            in_rationale = 0
            # fall through to normal handlers
        }

        /^invariants:[[:space:]]*$/ {
            in_inv = 1
            next
        }

        !in_inv { next }

        /^[[:space:]]+-[[:space:]]+id:/ {
            # New record — flush prior
            flush()
            in_one = 1
            sub(/^[[:space:]]+-[[:space:]]+id:[[:space:]]*/, "")
            gsub(/^[\"\x27]|[\"\x27]$/, "")
            id = $0
            next
        }

        in_one && /^[[:space:]]+token:/ {
            sub(/^[[:space:]]+token:[[:space:]]*/, "")
            # Strip a single layer of leading + trailing single OR double quotes
            gsub(/^[\"\x27]|[\"\x27]$/, "")
            token = $0
            in_paths = 0
            next
        }

        # hex_token: <lowercase-hex>   (PII-safe alternative to token; decoded
        # in the caller via xxd -r -p). Exactly one of token / hex_token MUST
        # be set; check_one() enforces. (Quote-free comment: this awk program
        # itself is wrapped in single quotes by the bash heredoc-like form, so
        # any embedded single quote inside the awk source closes the string.)
        in_one && /^[[:space:]]+hex_token:/ {
            sub(/^[[:space:]]+hex_token:[[:space:]]*/, "")
            gsub(/^[\"\x27]|[\"\x27]$/, "")
            hex_token = $0
            in_paths = 0
            next
        }

        in_one && /^[[:space:]]+regex:/ {
            sub(/^[[:space:]]+regex:[[:space:]]*/, "")
            gsub(/[[:space:]]+$/, "")
            regex = $0
            in_paths = 0
            next
        }

        in_one && /^[[:space:]]+expected_absent_in:[[:space:]]*$/ {
            in_paths = 1
            next
        }

        in_paths && /^[[:space:]]+-[[:space:]]+/ {
            sub(/^[[:space:]]+-[[:space:]]+/, "")
            gsub(/^[\"\x27]|[\"\x27]$/, "")
            if (paths == "") paths = $0
            else paths = paths "," $0
            next
        }

        # rationale: | (block scalar)
        in_one && /^[[:space:]]+rationale:[[:space:]]*\|[[:space:]]*$/ {
            in_paths = 0
            in_rationale = 1
            next
        }

        # Any other top-of-key line at the record-sibling indent closes the paths list
        in_one && /^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*:/ {
            in_paths = 0
            # fall through (next iteration will reclassify)
        }

        END { flush() }
    ' "$SPEC"
}

# -----------------------------------------------------------------------------
# Glob expansion under REPO root — bash-3.2 portable (macOS default), uses
# `find` instead of `shopt -s globstar` (bash 4+ only).
#
# Expansion rules:
#   - '**/*'                        → whole-repo sweep, gitignore-honored via
#                                     `git ls-files` (skips .git, build dirs,
#                                     generated binaries, etc.)
#   - '<dir>/**/<name-pattern>'     → `find <dir> -type f -name <pattern>`
#                                     (recursive; <name-pattern> may include
#                                     * for fnmatch-style matching, e.g.
#                                     '*.sh' or '*.cpp')
#   - '<literal-path>'              → literal file check (treated as one file)
#
# Output: one absolute path per line.
# -----------------------------------------------------------------------------

expand_glob() {
    local pat="$1"
    local hits=()
    local p

    if [ "$pat" = "**/*" ]; then
        # Whole-repo sweep — git ls-files honors .gitignore, skips submodule
        # .git internals + generated binaries. Skip .workplans/ + binary-ish
        # tracked artifacts that would produce noise.
        while IFS= read -r p; do
            case "$p" in
                .workplans/*) continue;;
                *.png|*.jpg|*.jpeg|*.pdf|*.bin) continue;;
            esac
            hits+=("$REPO/$p")
        done < <(cd "$REPO" && git ls-files)
    elif [[ "$pat" == *"**/"* ]]; then
        # 'prefix/**/name-pat' → `find prefix -type f -name name-pat`
        local prefix="${pat%%/\*\*/*}"
        local name="${pat##*\*\*/}"
        if [ -d "$REPO/$prefix" ]; then
            while IFS= read -r p; do
                hits+=("$p")
            done < <(find "$REPO/$prefix" -type f -name "$name" 2>/dev/null)
        fi
    else
        # Literal path
        if [ -f "$REPO/$pat" ]; then
            hits+=("$REPO/$pat")
        fi
    fi

    if [ "${#hits[@]}" -gt 0 ]; then
        printf '%s\n' "${hits[@]}"
    fi
}

# -----------------------------------------------------------------------------
# Comment-line stripper.
#
# Reads file on $1, prints to stdout only lines that are NOT entirely a
# `^[[:space:]]*#` shell/yaml/python-style comment. C-style `//` is NOT
# stripped; scope your invariant token accordingly.
# -----------------------------------------------------------------------------

strip_comments() {
    local f="$1"
    awk '!/^[[:space:]]*#/' "$f"
}

# -----------------------------------------------------------------------------
# Per-invariant check.
#
# Returns 0 = PASS, 1 = MISMATCH (prints offending file:line on FAIL).
# -----------------------------------------------------------------------------

check_one() {
    local id="$1" regex="$2" token="$3" hex_token="$4" paths_csv="$5"
    local grep_flag="-F"
    if [ "$regex" = "true" ]; then
        grep_flag="-E"
    fi

    # Exactly-one source check (token XOR hex_token); aborts the run so a
    # malformed spec entry never silently becomes a no-op gate.
    if [ -n "$hex_token" ] && [ -n "$token" ]; then
        printf "[MISMATCH] %-32s  spec error: both 'token:' and 'hex_token:' set; pick exactly one\n" "$id" >&2
        return 1
    fi
    if [ -z "$hex_token" ] && [ -z "$token" ]; then
        printf "[MISMATCH] %-32s  spec error: neither 'token:' nor 'hex_token:' set\n" "$id" >&2
        return 1
    fi
    # Decode hex_token (PII-safe path) at scan time so the spec file itself
    # never carries the literal secret/PII string.
    if [ -n "$hex_token" ]; then
        if ! token="$(printf '%s' "$hex_token" | xxd -r -p)" || [ -z "$token" ]; then
            printf "[MISMATCH] %-32s  spec error: hex_token '%s' failed xxd decode\n" "$id" "$hex_token" >&2
            return 1
        fi
    fi

    # CSV-aware split that preserves literal `**/`, `**/*`, and other glob
    # metachars. The old `local globs=( $paths_csv )` form ran a pathname
    # expansion on each comma-segment BEFORE the array assignment, snapshotting
    # whatever happened to be on disk at parse time (and silently dropping
    # `**` since bash <4 has no globstar). expand_glob() never received the
    # literal pattern, so its `**/` / `**/*` branches were dead.
    local globs=()
    local IFS_SAVE="$IFS"
    IFS=','
    read -ra globs <<< "$paths_csv"
    IFS="$IFS_SAVE"
    # Trim leading/trailing whitespace per entry (CSV-without-quotes friendly)
    local i
    for i in "${!globs[@]}"; do
        globs[i]="${globs[i]#"${globs[i]%%[![:space:]]*}"}"
        globs[i]="${globs[i]%"${globs[i]##*[![:space:]]}"}"
    done

    local seen_files=()
    local pat
    for pat in "${globs[@]}"; do
        while IFS= read -r f; do
            [ -n "$f" ] && seen_files+=( "$f" )
        done < <(expand_glob "$pat")
    done

    if [ "${#seen_files[@]}" -eq 0 ]; then
        printf "[MISMATCH] %-32s  no files matched any expected_absent_in glob (paths_csv=%s)\n" "$id" "$paths_csv"
        return 1
    fi

    local found=0
    local f hits
    for f in "${seen_files[@]}"; do
        # Skip this script itself — it documents the token via comments and
        # CLI examples. The SPEC file is NOT self-excluded: invariants.yaml
        # entries use hex_token for PII (see F3 / PR #71) so the literal
        # string never lives in the spec; if it ever appears as a `token:`
        # field the gate SHOULD fire.
        case "$f" in
            "$SCRIPT_DIR/check_invariant_sweep.sh") continue;;
        esac
        if hits=$(strip_comments "$f" | grep -n "$grep_flag" -- "$token" 2>/dev/null); then
            if [ -n "$hits" ]; then
                if [ $found -eq 0 ]; then
                    printf "[MISMATCH] %-32s  token=%s\n" "$id" "$token"
                    found=1
                fi
                printf "%s\n" "$hits" | while IFS= read -r line; do
                    printf "             %s: %s\n" "${f#"$REPO"/}" "$line"
                done
            fi
        fi
    done

    if [ $found -eq 0 ]; then
        printf "[OK]       %-32s  token=%s absent in %d file(s)\n" "$id" "$token" "${#seen_files[@]}"
        return 0
    fi
    return 1
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

OVERALL=0
COUNT=0
while IFS='|' read -r id regex token hex_token paths_csv; do
    [ -z "$id" ] && continue
    if [ -n "$FILTER_ID" ] && [ "$id" != "$FILTER_ID" ]; then
        continue
    fi
    COUNT=$((COUNT + 1))
    if ! check_one "$id" "$regex" "$token" "$hex_token" "$paths_csv"; then
        OVERALL=1
    fi
done < <(parse_invariants)

if [ "$COUNT" -eq 0 ]; then
    if [ -n "$FILTER_ID" ]; then
        printf "%s: invariant id '%s' not found in %s\n" "$PROG" "$FILTER_ID" "$SPEC" >&2
    else
        printf "%s: no invariants parsed from %s\n" "$PROG" "$SPEC" >&2
    fi
    exit 2
fi

if [ $OVERALL -ne 0 ]; then
    printf "\nFAIL: %d invariant(s) MISMATCH; see [MISMATCH] lines above\n" "$COUNT" >&2
    exit 1
fi
printf "\nPASS: %d invariant(s) all OK\n" "$COUNT"
exit 0
