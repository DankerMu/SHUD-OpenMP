#!/bin/bash
#
# test_mesh_map.sh — static negative tests for run_p7_wallclock.sbatch
# (case→mesh map + assert_real_s_sane), per openspec change
# s2-wallclock-script-mesh-map tasks.md §3.
#
# Usage: ./tools/server_validation/test_mesh_map.sh
# Exit 0 = PASS, non-zero = FAIL.
#
# This runs the static layer only (no SHUD build, no Slurm submit). It
# isolates `declare -A CASE_TO_MESH=(...)` and `assert_real_s_sane()`
# from run_p7_wallclock.sbatch via awk-windowed extraction so we never
# trigger any SBATCH-only side effects (mkdir on server scratch dirs,
# Makefile clean, etc.).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH="$SCRIPT_DIR/run_p7_wallclock.sbatch"

[ -r "$SBATCH" ] || { echo "FAIL: sbatch not readable: $SBATCH" >&2; exit 1; }

# ---- Test 1: bash -n syntax ----
bash -n "$SBATCH" || { echo "FAIL: bash -n syntax check failed" >&2; exit 1; }
echo "PASS: bash -n syntax"

# ---- Test 2: CASE_TO_MESH map contents ----
# Extract the `declare -A CASE_TO_MESH=(...)` block by sourcing a fenced subset.
# We isolate JUST the map declaration + assert_real_s_sane via grep window so
# the test does not run any sbatch-only logic (SBATCH dirs / make build).
TMP=$(mktemp)
awk '/^declare -A CASE_TO_MESH=/,/^\)/' "$SBATCH" > "$TMP"
[ -s "$TMP" ] || { echo "FAIL: no CASE_TO_MESH block found in sbatch" >&2; rm -f "$TMP"; exit 1; }
# shellcheck disable=SC1090
source "$TMP"
rm -f "$TMP"

[ "${CASE_TO_MESH[qinyijiang]:-}" = "nanlin" ] || { echo "FAIL: [qinyijiang]=nanlin expected, got '${CASE_TO_MESH[qinyijiang]:-}'" >&2; exit 1; }
[ "${CASE_TO_MESH[heihe]:-}" = "heihe" ] || { echo "FAIL: [heihe]=heihe expected" >&2; exit 1; }
[ "${CASE_TO_MESH[heihe_x4]:-}" = "heihe_x4" ] || { echo "FAIL: [heihe_x4]=heihe_x4 expected" >&2; exit 1; }
echo "PASS: CASE_TO_MESH has qinyijiang→nanlin + heihe identity + heihe_x4 identity"

# ---- Test 3: missing-key fail-fast via `:?` ----
unset 'CASE_TO_MESH[qinyijiang]'
# Run in a subshell with `set -u`; expect non-zero.
if ( set -u; : "${CASE_TO_MESH[qinyijiang]:?case→mesh mapping missing for qinyijiang}" ) 2>/dev/null; then
    echo "FAIL: missing-key did NOT trigger non-zero exit" >&2
    exit 1
fi
echo "PASS: missing CASE_TO_MESH[qinyijiang] triggers non-zero exit via :?"

# ---- Test 4: assert_real_s_sane behaviour ----
TMP2=$(mktemp)
awk '/^assert_real_s_sane\(\)/,/^\}/' "$SBATCH" > "$TMP2"
[ -s "$TMP2" ] || { echo "FAIL: no assert_real_s_sane function found" >&2; rm -f "$TMP2"; exit 1; }
# shellcheck disable=SC1090
source "$TMP2"
rm -f "$TMP2"

# 0.01s -> fail
if assert_real_s_sane 0.01 2>/dev/null; then
    echo "FAIL: assert_real_s_sane(0.01) should have returned non-zero" >&2
    exit 1
fi
echo "PASS: assert_real_s_sane(0.01) returns non-zero"

# 30.0s -> ok
if ! assert_real_s_sane 30.0 2>/dev/null; then
    echo "FAIL: assert_real_s_sane(30.0) should have returned zero" >&2
    exit 1
fi
echo "PASS: assert_real_s_sane(30.0) returns zero"

# 1.0s exact -> ok (boundary; not strictly < 1.0)
if ! assert_real_s_sane 1.0 2>/dev/null; then
    echo "FAIL: assert_real_s_sane(1.0) should have returned zero (boundary)" >&2
    exit 1
fi
echo "PASS: assert_real_s_sane(1.0) returns zero (boundary)"

# 0.99s -> fail (just below)
if assert_real_s_sane 0.99 2>/dev/null; then
    echo "FAIL: assert_real_s_sane(0.99) should have returned non-zero" >&2
    exit 1
fi
echo "PASS: assert_real_s_sane(0.99) returns non-zero"

echo ""
echo "ALL TESTS PASS"
exit 0
