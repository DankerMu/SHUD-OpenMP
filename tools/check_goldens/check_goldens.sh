#!/bin/sh
# check_goldens.sh — verify all 24 snapshot golden SHA256s vs docs manifest
# + 12 unique-bin SHAs from per-case repeatability_snapshots.txt files.
#
# Parses `docs/build_manifest.md` for the two SHA tables (after-PassValue
# 12 + before-PassValue 12), computes sha256 of each listed file, and
# emits per-row [OK] / [MISMATCH] lines.
#
# F27c (PR #54 round-3): also parses 4 × `benchmarks/<case>/B0_output/
# repeatability_snapshots.txt`. For each `run<N>  <filename>  <sha>` row,
# verifies the SHA matches `sha256sum benchmarks/<case>/B0_output/<filename>`.
# This catches drift between the 3-run repeatability evidence and the
# shipped before-PassValue golden bins.
#
# Exit 0 iff 24 build_manifest.md rows + 12 unique-bin rows from
# repeatability files all match (36 entries total). Exit non-zero on any
# mismatch or missing file. Run from anywhere — the script resolves the
# repo root from its own location.
#
# Owned by: openspec change s1-rhs-core-extraction (PR #54 round-1 F12 +
# round-3 F27c).

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
    # F21 (PR #54 round-2): use printf '%s\n' instead of echo to avoid
    # backslash / leading-dash interpretation differences between BSD and
    # GNU echo (file_path entries are safe today, but printf is the POSIX
    # contract-compliant way to emit arbitrary user data).
    printf '%s\n' "$1" | sed -e 's/^[[:space:]]*`//' -e 's/`[[:space:]]*$//'
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

printf '\nMANIFEST CHECK SUMMARY  total=%d  pass=%d  fail=%d\n' "$TOTAL" "$PASS" "$FAIL"
if [ "$TOTAL" -lt 24 ]; then
    printf 'WARNING: expected 24 golden rows, parsed %d. Check docs/build_manifest.md table layout.\n' "$TOTAL" >&2
    exit 1
fi

# -----------------------------------------------------------------------------
# F27c — repeatability_snapshots.txt drift detector.
#
# For each of the 4 cases (keliya / xinanjiang_upstream / qinyijiang / qhh),
# parse `benchmarks/<case>/B0_output/repeatability_snapshots.txt` rows of
# the form `run<N>  snapshot_t<v>_before_passvalue.bin  <sha>`. Take the
# 3 unique-bin SHAs (one per t-value; 3 runs × 3 t-values = 9 rows, but
# determinism collapses to 3 unique SHAs per file). Verify each matches
# `sha256sum benchmarks/<case>/B0_output/<filename>`.
#
# Total: 4 cases × 3 t-values = 12 unique-bin checks.
# -----------------------------------------------------------------------------
REPEAT_PASS=0
REPEAT_FAIL=0
REPEAT_TOTAL=0

# mktemp-based scratch file for subshell-tally workaround
# (a `... | while` pipeline runs in subshell, so tally vars can't escape;
# pipe to file, then count in parent). F30-pattern mktemp + EXIT trap so
# the scratch file is cleaned up even on early exit.
REPEAT_SCRATCH=$(mktemp -t check_goldens_repeat.XXXXXX) || {
    printf 'FATAL: mktemp failed for repeatability scratch file\n' >&2
    exit 2
}
trap 'rm -f "$REPEAT_SCRATCH"' EXIT

for case_name in keliya xinanjiang_upstream qinyijiang qhh; do
    repeat_file="$REPO_ROOT/benchmarks/$case_name/B0_output/repeatability_snapshots.txt"
    if [ ! -f "$repeat_file" ]; then
        printf '[MISSING-REPEAT] benchmarks/%s/B0_output/repeatability_snapshots.txt\n' "$case_name"
        REPEAT_FAIL=$((REPEAT_FAIL + 1))
        continue
    fi
    # Parse `run<N>  <filename>  <sha>` rows, take unique (filename, sha) pairs.
    # awk: skip blank lines + comment lines (^#), emit "filename<TAB>sha".
    # `sort -u` collapses 3 runs into 3 unique rows when deterministic.
    awk '/^run[0-9]+[[:space:]]+snapshot_t[0-9]+_before_passvalue\.bin[[:space:]]+[0-9a-f]+/ {
            print $2 "\t" $3
        }' "$repeat_file" | sort -u | while IFS="$(printf '\t')" read -r fname expected_sha; do
        [ -z "$fname" ] && continue
        abs_path="$REPO_ROOT/benchmarks/$case_name/B0_output/$fname"
        if [ ! -f "$abs_path" ]; then
            printf '[MISSING-REPEAT] %s/B0_output/%s (expected %s)\n' "$case_name" "$fname" "$expected_sha"
            printf '__REPEAT_FAIL__\n'
            continue
        fi
        actual=$($SHA_CMD "$abs_path" | awk '{print $1}')
        if [ "$actual" = "$expected_sha" ]; then
            printf '[OK-REPEAT] %s/%s  %s\n' "$case_name" "$fname" "$actual"
            printf '__REPEAT_OK__\n'
        else
            printf '[MISMATCH-REPEAT] %s/%s  expected=%s actual=%s\n' "$case_name" "$fname" "$expected_sha" "$actual"
            printf '__REPEAT_FAIL__\n'
        fi
    done > "$REPEAT_SCRATCH"
    while IFS= read -r line; do
        case "$line" in
            __REPEAT_OK__)
                REPEAT_PASS=$((REPEAT_PASS + 1))
                REPEAT_TOTAL=$((REPEAT_TOTAL + 1))
                ;;
            __REPEAT_FAIL__)
                REPEAT_FAIL=$((REPEAT_FAIL + 1))
                REPEAT_TOTAL=$((REPEAT_TOTAL + 1))
                ;;
            "[OK-REPEAT]"*|"[MISMATCH-REPEAT]"*|"[MISSING-REPEAT]"*)
                # Emit per-row diagnostic to stdout (was previously written
                # to scratch interleaved with sentinels; print here for the
                # user-visible output).
                printf '%s\n' "$line"
                ;;
        esac
    done < "$REPEAT_SCRATCH"
done

printf '\nREPEATABILITY CHECK SUMMARY  total=%d  pass=%d  fail=%d\n' \
    "$REPEAT_TOTAL" "$REPEAT_PASS" "$REPEAT_FAIL"

if [ "$REPEAT_TOTAL" -lt 12 ]; then
    printf 'WARNING: expected 12 repeatability rows (4 cases × 3 t-values), parsed %d. Check repeatability_snapshots.txt files.\n' "$REPEAT_TOTAL" >&2
    exit 1
fi

TOTAL_GRAND=$((TOTAL + REPEAT_TOTAL))
PASS_GRAND=$((PASS + REPEAT_PASS))
FAIL_GRAND=$((FAIL + REPEAT_FAIL))
printf '\nGRAND TOTAL  total=%d  pass=%d  fail=%d\n' "$TOTAL_GRAND" "$PASS_GRAND" "$FAIL_GRAND"

if [ "$FAIL" -ne 0 ] || [ "$REPEAT_FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
