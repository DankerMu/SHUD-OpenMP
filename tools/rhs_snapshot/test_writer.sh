#!/bin/sh
# test_writer.sh — 6-scenario acceptance test for compose_snapshot_filename().
#
# Scenarios (PR #54 round-1 F9 fix; PR #54 round-2 F17 adds (e)/(f)):
#   (a) Empty suffix → "snapshot_t<t>.bin" (legacy form, PR #53 back-compat)
#   (b) Non-empty suffix → "snapshot_t<t>_<suffix>.bin"
#   (c) `/` in suffix → rejected (compose returns empty → harness exit 1)
#   (d) `\\` in suffix → rejected (path-traversal guard)
#   (e) Suffix == 64 chars → accepted (boundary case for F5 length cap)
#   (f) Suffix == 65 chars → rejected (F5 length cap; +1 over boundary)
#
# Mirrors the test pattern in tools/cvode_stats_diff/test_cvode_stats_diff.sh.
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.1b.i).

set -eu

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
HARNESS_SRC="$SCRIPT_DIR/test_writer_main.cpp"
WRITER_SRC="$SCRIPT_DIR/writer.cpp"

[ -r "$HARNESS_SRC" ] || { printf "FATAL: %s missing\n" "$HARNESS_SRC" >&2; exit 2; }
[ -r "$WRITER_SRC" ]  || { printf "FATAL: %s missing\n" "$WRITER_SRC"  >&2; exit 2; }

# Pick a compiler: prefer g++, fall back to clang++ (matches the project's
# Linux-vs-macOS convention without requiring Makefile plumbing).
CXX="${CXX:-}"
if [ -z "$CXX" ]; then
    if command -v g++ >/dev/null 2>&1; then
        CXX=g++
    elif command -v clang++ >/dev/null 2>&1; then
        CXX=clang++
    else
        printf "FATAL: no g++ or clang++ in PATH\n" >&2
        exit 2
    fi
fi

TMP=$(mktemp -d -t rhs_snapshot_test_writer.XXXXXX)
trap 'rm -rf "$TMP"' EXIT

BIN="$TMP/test_writer"
"$CXX" -std=c++17 -O0 -Wall -I"$SCRIPT_DIR" \
    -o "$BIN" "$HARNESS_SRC" "$WRITER_SRC"

PASS=0
FAIL=0

assert_exit() {
    expect="$1"; shift
    label="$1"; shift
    actual="$1"; shift
    if [ "$actual" = "$expect" ]; then
        printf "PASS  %s  (exit=%s)\n" "$label" "$actual"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %s  (expect_exit=%s actual=%s)\n" "$label" "$expect" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

assert_stdout() {
    label="$1"; shift
    expect="$1"; shift
    actual="$1"; shift
    if [ "$actual" = "$expect" ]; then
        printf "PASS  %s  (stdout=%s)\n" "$label" "$actual"
        PASS=$((PASS + 1))
    else
        printf "FAIL  %s  (expect_stdout=%s actual=%s)\n" "$label" "$expect" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

# -----------------------------------------------------------------------------
# Scenario (a): empty suffix → legacy filename
# -----------------------------------------------------------------------------
set +e
out_a=$("$BIN" 86400 "" 2>&1); rc_a=$?
set -e
assert_exit 0 "scenario(a) empty suffix → exit 0" "$rc_a"
assert_stdout "scenario(a) empty suffix → legacy filename" "snapshot_t86400.bin" "$out_a"

# -----------------------------------------------------------------------------
# Scenario (b): non-empty suffix → snapshot_t<t>_<suffix>.bin
# -----------------------------------------------------------------------------
set +e
out_b=$("$BIN" 2592000 "before_passvalue" 2>&1); rc_b=$?
set -e
assert_exit 0 "scenario(b) non-empty suffix → exit 0" "$rc_b"
assert_stdout "scenario(b) suffixed filename" "snapshot_t2592000_before_passvalue.bin" "$out_b"

# -----------------------------------------------------------------------------
# Scenario (c): `/` in suffix → rejected
# -----------------------------------------------------------------------------
set +e
out_c=$("$BIN" 86400 "bad/path" 2>&1); rc_c=$?
set -e
assert_exit 1 "scenario(c) suffix with '/' → exit 1 (rejected)" "$rc_c"

# -----------------------------------------------------------------------------
# Scenario (d): `\\` in suffix → rejected
# -----------------------------------------------------------------------------
set +e
out_d=$("$BIN" 86400 "bad\\sep" 2>&1); rc_d=$?
set -e
assert_exit 1 "scenario(d) suffix with '\\\\' → exit 1 (rejected)" "$rc_d"

# -----------------------------------------------------------------------------
# Scenario (e): suffix == 64 chars → accepted (boundary case for F5 cap)
# -----------------------------------------------------------------------------
# F17 (PR #54 round-2): cover the F5 length-cap boundary explicitly so
# regressions to `> 63` / `>= 64` accidental tightening get caught.
SFX_64=$(printf 'a%.0s' $(seq 1 64))
[ "${#SFX_64}" -eq 64 ] || { printf "FATAL: SFX_64 length=%d, want 64\n" "${#SFX_64}" >&2; exit 2; }
set +e
out_e=$("$BIN" 86400 "$SFX_64" 2>&1); rc_e=$?
set -e
assert_exit 0 "scenario(e) suffix 64 chars (boundary) → exit 0 (accepted)" "$rc_e"
assert_stdout "scenario(e) suffix 64 chars yields suffixed filename" "snapshot_t86400_${SFX_64}.bin" "$out_e"

# -----------------------------------------------------------------------------
# Scenario (f): suffix == 65 chars → rejected (F5 length cap; +1 over boundary)
# -----------------------------------------------------------------------------
SFX_65=$(printf 'a%.0s' $(seq 1 65))
[ "${#SFX_65}" -eq 65 ] || { printf "FATAL: SFX_65 length=%d, want 65\n" "${#SFX_65}" >&2; exit 2; }
set +e
out_f=$("$BIN" 86400 "$SFX_65" 2>&1); rc_f=$?
set -e
assert_exit 1 "scenario(f) suffix 65 chars → exit 1 (F5 length-cap reject)" "$rc_f"
# F22 (PR #54 round-2): writer.cpp now emits stderr diagnostic on length-cap
# reject; verify the message reaches the harness's combined out (2>&1 above).
if printf '%s' "$out_f" | grep -q 'exceeds 64-char limit'; then
    printf "PASS  scenario(f) writer stderr diagnostic present\n"
    PASS=$((PASS + 1))
else
    printf "FAIL  scenario(f) writer stderr diagnostic missing (got: %s)\n" "$out_f"
    FAIL=$((FAIL + 1))
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
printf "\nTEST SUMMARY  PASS=%d  FAIL=%d\n" "$PASS" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
    exit 1
fi
exit 0
