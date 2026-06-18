#!/bin/sh
# check_goldens.sh — verify all 24 snapshot golden SHA256s vs docs manifest.
#
# Parses `docs/build_manifest.md` for the two SHA tables (after-PassValue
# 12 + before-PassValue 12), computes sha256 of each listed file, and
# emits per-row [OK] / [MISMATCH] lines.
#
# Exit 0 iff all 24 rows match. Exit non-zero on any mismatch or missing
# file. Run from anywhere — the script resolves the repo root from its
# own location.
#
# Owned by: openspec change s1-rhs-core-extraction (PR #54 round-1 F12).

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
MANIFEST="$REPO_ROOT/docs/build_manifest.md"

[ -r "$MANIFEST" ] || { printf "FATAL: %s missing\n" "$MANIFEST" >&2; exit 2; }

# Pick a sha256 tool: prefer GNU sha256sum, fall back to BSD shasum -a 256.
if command -v sha256sum >/dev/null 2>&1; then
    SHA_CMD='sha256sum'
elif command -v shasum >/dev/null 2>&1; then
    SHA_CMD='shasum -a 256'
else
    printf "FATAL: no sha256sum or shasum in PATH\n" >&2
    exit 2
fi

# Parse rows from build_manifest.md. Both target tables share the same
# 4-pipe shape:
#   | `<case>` | `<t_value>` | `<file_path>` | `<sha256>` |
# Filter to rows that mention an snapshot bin path; awk extracts the
# fields. We deliberately match on the bin filename suffix to avoid
# picking up other tables that happen to share the same column count.
PASS=0
FAIL=0
TOTAL=0

# awk extracts file_path + expected_sha. The `|` split yields:
#   $1 = "" (leading)
#   $2 = " `<case>` "
#   $3 = " `<t_value>` "
#   $4 = " `<file_path>` "
#   $5 = " `<sha256>` "
#   $6 = "" (trailing)
# We strip leading/trailing space + surrounding backticks.
trim() {
    # $1 = string to trim of surrounding whitespace + backticks
    echo "$1" | sed -e 's/^[[:space:]]*`//' -e 's/`[[:space:]]*$//'
}

while IFS= read -r row; do
    file_field=$(printf '%s\n' "$row" | awk -F'|' '{print $4}')
    sha_field=$(printf '%s\n' "$row"  | awk -F'|' '{print $5}')
    file_path=$(trim "$file_field")
    expected=$(trim "$sha_field")
    [ -z "$file_path" ] && continue
    [ -z "$expected" ] && continue
    TOTAL=$((TOTAL + 1))
    abs_path="$REPO_ROOT/$file_path"
    if [ ! -f "$abs_path" ]; then
        printf '[MISSING] %s  (expected %s)\n' "$file_path" "$expected"
        FAIL=$((FAIL + 1))
        continue
    fi
    actual=$($SHA_CMD "$abs_path" | awk '{print $1}')
    if [ "$actual" = "$expected" ]; then
        printf '[OK]       %s  %s\n' "$file_path" "$actual"
        PASS=$((PASS + 1))
    else
        printf '[MISMATCH] %s  expected=%s actual=%s\n' "$file_path" "$expected" "$actual"
        FAIL=$((FAIL + 1))
    fi
done <<EOF
$(grep -E '^\| `[a-z_]+` *\| `[0-9]+` *\| `benchmarks/[a-z_]+/B0_output/snapshot_t[0-9_a-z]+\.bin` *\| `[0-9a-f]{64}` *\|' "$MANIFEST")
EOF

printf '\nGOLDEN CHECK SUMMARY  total=%d  pass=%d  fail=%d\n' "$TOTAL" "$PASS" "$FAIL"
if [ "$TOTAL" -lt 24 ]; then
    printf 'WARNING: expected 24 golden rows, parsed %d. Check docs/build_manifest.md table layout.\n' "$TOTAL" >&2
    exit 1
fi
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
