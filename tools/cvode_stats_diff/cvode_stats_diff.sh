#!/bin/sh
# cvode_stats_diff.sh — strict 16-key bitwise diff for CVODE final stats.
#
# Usage:
#   cvode_stats_diff.sh <new.txt> <golden.txt>
#
# Behavior (contract; do not soften without spec change):
#   Exit 0 iff ALL 16 canonical keys are present in BOTH files and the
#   values are byte-equal pairwise. Any deviation = non-zero exit:
#     - key MISSING in <file>  (key not present in either file)
#     - key UNKNOWN in <file>: <key>  (key present but not in canonical 16)
#     - key: golden=<X>, new=<Y>  (key present in both, values differ)
#
# Canonical 16-key set (per openspec/.../design.md D10; order matters
# only for stable diagnostic output, not for equality):
#   nfe, nfeLS, nni, nli, nsetups, netf, nst, npe, nps, ncfn, ncfl,
#   lenrw, leniw, lenrwLS, leniwLS, nFCall
#
# Input format: `key=value` lines (one key per line), as written by
# SHUD's PrintFinalStats (see SHUD/src/Equations/cvode_config.cpp:90-106).
# Both `key=value` and `key: value` and `key = value` forms are
# accepted (we strip whitespace + `=` + `:` to a canonical token);
# downstream B0 archive uses `key=value` form.
#
# POSIX shell compliant (no bash-isms; tested with /bin/sh + busybox).
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.2).

set -eu

PROG=$(basename "$0")

usage() {
    printf "Usage: %s <new.txt> <golden.txt>\n" "$PROG" >&2
    exit 2
}

[ $# -eq 2 ] || usage

NEW="$1"
GOLDEN="$2"

[ -r "$NEW" ]    || { printf "%s: cannot read NEW file: %s\n"    "$PROG" "$NEW"    >&2; exit 2; }
[ -r "$GOLDEN" ] || { printf "%s: cannot read GOLDEN file: %s\n" "$PROG" "$GOLDEN" >&2; exit 2; }

CANONICAL_KEYS="nfe nfeLS nni nli nsetups netf nst npe nps ncfn ncfl lenrw leniw lenrwLS leniwLS nFCall"

# parse_value <file> <key>
# Emits the value (rhs of `<key>=` or `<key>:` or `<key> =`) to stdout.
# Empty stdout = key absent. Uses awk for portability.
#
# F3 fix: detects duplicate-key files. If the key appears MORE than once
# in <file>, parse_value emits nothing to stdout and exits non-zero so the
# caller can route the file to the dup-key diagnostic path. (Previously
# the `exit` on first match silently picked the first occurrence, which
# could hide a corruption.)
parse_value() {
    file="$1"
    key="$2"
    awk -v k="$key" '
        # Normalize: strip whitespace around `=` / `:` so `nfe= 100`,
        # `nfe : 100`, `nfe=100` all parse identically.
        {
            line=$0
            delim=0
            # Find first delimiter (= or :).
            for (i=1; i<=length(line); i++) {
                c=substr(line,i,1)
                if (c=="=" || c==":") { delim=i; break }
            }
            if (delim==0) next
            kpart=substr(line, 1, delim-1)
            vpart=substr(line, delim+1)
            # trim
            gsub(/^[ \t]+|[ \t]+$/, "", kpart)
            gsub(/^[ \t]+|[ \t]+$/, "", vpart)
            if (kpart == k) { hits++; captured=vpart }
        }
        END {
            if (hits == 1) { print captured; exit 0 }
            else if (hits == 0) { exit 0 }   # absent: caller treats empty as MISSING
            else { exit 2 }                   # duplicate: caller signals dup-key
        }
    ' "$file"
}

# list_keys <file>
# Emits one key per line (extracted as the lhs of `=` or `:`).
list_keys() {
    file="$1"
    awk '
        {
            line=$0
            delim=0
            for (i=1; i<=length(line); i++) {
                c=substr(line,i,1)
                if (c=="=" || c==":") { delim=i; break }
            }
            if (delim==0) next
            kpart=substr(line, 1, delim-1)
            gsub(/^[ \t]+|[ \t]+$/, "", kpart)
            if (kpart != "") print kpart
        }
    ' "$file"
}

# is_canonical_key <key>
# Returns 0 (truthy) iff key is in the 16-key canonical set.
#
# Implementation note: POSIX shell `for` loops do NOT scope the iteration
# variable, so a function-local `for k in ...` would leak its final value
# back to the caller and corrupt outer loops that also use `k`. Use a
# `_ck` (function-private) name to avoid that aliasing.
is_canonical_key() {
    _ck_needle="$1"
    for _ck in $CANONICAL_KEYS; do
        [ "$_ck" = "$_ck_needle" ] && return 0
    done
    return 1
}

FAIL=0

# 1) Check missing keys (canonical 16, in either file) + dup-key detection.
# F3 fix: parse_value now returns exit 2 on duplicate key; if either file
# has the same canonical key listed more than once, treat the whole file as
# corrupt and fail-fast with a DUPLICATE diagnostic.
#
# `set +e` brackets the parse_value capture: awk's non-zero exit on dup-key
# (rc=2) propagates into `$(...)`, and `set -e` (line 29) would otherwise
# abort the script before we can route it.
for k in $CANONICAL_KEYS; do
    set +e
    new_v=$(parse_value "$NEW"    "$k"); new_rc=$?
    gld_v=$(parse_value "$GOLDEN" "$k"); gld_rc=$?
    set -e
    if [ "$new_rc" -eq 2 ]; then
        printf "DUPLICATE key in %s: %s\n" "$NEW" "$k"
        FAIL=1
    fi
    if [ "$gld_rc" -eq 2 ]; then
        printf "DUPLICATE key in %s: %s\n" "$GOLDEN" "$k"
        FAIL=1
    fi
    if [ "$new_rc" -eq 0 ] && [ -z "$new_v" ]; then
        printf "%s MISSING in %s\n" "$k" "$NEW"
        FAIL=1
    fi
    if [ "$gld_rc" -eq 0 ] && [ -z "$gld_v" ]; then
        printf "%s MISSING in %s\n" "$k" "$GOLDEN"
        FAIL=1
    fi
done

# 2) Check unknown keys (present in file but not canonical 16).
for f in "$NEW" "$GOLDEN"; do
    list_keys "$f" | while IFS= read -r k; do
        if ! is_canonical_key "$k"; then
            printf "%s UNKNOWN in %s: %s\n" "key" "$f" "$k"
            # Cannot set FAIL from subshell pipe; emit sentinel file.
            : > /tmp/cvode_stats_diff.unknown.$$
        fi
    done
done
if [ -f /tmp/cvode_stats_diff.unknown.$$ ]; then
    FAIL=1
    rm -f /tmp/cvode_stats_diff.unknown.$$
fi

# 3) For canonical keys present in BOTH files, value-compare.
# Skip dup-key files here (already diagnosed in pass 1; their parse_value
# returns empty so a value-mismatch error would be misleading noise).
for k in $CANONICAL_KEYS; do
    set +e
    new_v=$(parse_value "$NEW"    "$k"); new_rc=$?
    gld_v=$(parse_value "$GOLDEN" "$k"); gld_rc=$?
    set -e
    [ "$new_rc" -ne 0 ] && continue
    [ "$gld_rc" -ne 0 ] && continue
    [ -z "$new_v" ] && continue
    [ -z "$gld_v" ] && continue
    if [ "$new_v" != "$gld_v" ]; then
        printf "%s: golden=%s, new=%s\n" "$k" "$gld_v" "$new_v"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
