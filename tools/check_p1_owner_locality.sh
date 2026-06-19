#!/usr/bin/env bash
# tools/check_p1_owner_locality.sh — S2 P1 owner-locality grep gate
#
# Enforces openspec/changes/s2-strict-omp-full/specs/p1-rhs-update-owner-local-omp/spec.md
# Scenario "Element state update only writes Ele[i] owner":
#   - Element loop body writes only use loop index `i`
#   - No `Ele[i+1]` / `Ele[i-1]` / `Ele[inabr]` / `Ele[Ele[i].nabr[j]-1]` patterns
#   - No reduction / atomic clauses on the owner-local pragma
#
# Exit 0 on PASS; exit 1 on any forbidden pattern found.
# Run from the outer repo root.

set -euo pipefail

SCOPE="SHUD/src/ModelData/MD_update.cpp"
TARGET_FN="Model_Data::f_update"  # the no-suffix one (L63 onward), not f_updatei

if [ ! -f "$SCOPE" ]; then
    echo "FAIL: $SCOPE not found"
    exit 1
fi

fail=0

# Extract the f_update() body — from "void Model_Data::f_update(...)" up to the
# matching close brace. Simple line-range carving: find function start, extract
# down to the "}//end of for j=1:NumEle" comment that anchors end of element loop.
start=$(grep -n '^void Model_Data::f_update(' "$SCOPE" | head -1 | cut -d: -f1)
if [ -z "$start" ]; then
    echo "FAIL: f_update() function start not found in $SCOPE"
    exit 1
fi
# End of element loop is the "}//end of for j=1:NumEle" comment line (L104 baseline).
end=$(awk -v s="$start" 'NR>s && /end of for j=1:NumEle/{print NR; exit}' "$SCOPE")
if [ -z "$end" ]; then
    echo "FAIL: cannot locate end of element loop comment in f_update()"
    exit 1
fi

body=$(sed -n "${start},${end}p" "$SCOPE")

# Strip C++ line-comment-only lines (`^\s*//…`) before structural checks so the
# legacy dead-code blocks (e.g. the `// Ele[Ele[i].nabr[j]-1]` lines kept as
# mass-balance reference) do not false-positive on Check 1 / Check 2. Block
# comments (`/* … */`) are also stripped on the start / continuation lines.
# This filter only affects Check 1 / Check 2; Check 3 keeps the anchored
# `^[[:space:]]*#pragma omp` so block-comment pragma mentions stay excluded.
body_live=$(echo "$body" | grep -vE '^[[:space:]]*//' | grep -vE '^[[:space:]]*\*' | grep -vE '^[[:space:]]*/\*')

# Check 1: no Ele[i+1] / Ele[i-1] / Ele[i+anything] / Ele[i-anything]
if echo "$body_live" | grep -E 'Ele\[i[[:space:]]*[\+\-]' >/dev/null; then
    echo "FAIL: found Ele[i+/-…] cross-element index in f_update() element loop"
    echo "$body_live" | grep -nE 'Ele\[i[[:space:]]*[\+\-]'
    fail=1
fi

# Check 2: no Ele[inabr] / Ele[nabr] / Ele[Ele[i].nabr[…]-1] etc. — any 'nabr'
# inside Ele[…]. Allow `Ele[i].nabr` as field READ (used inside conditionals,
# not as write target). Disallow `Ele[…nabr…]` as the index expression.
if echo "$body_live" | grep -E 'Ele\[[^]]*nabr[^]]*\]' >/dev/null; then
    bad_lines=$(echo "$body_live" | grep -nE 'Ele\[[^]]*nabr[^]]*\]' || true)
    # Filter out Ele[i].nabr accesses (the dot indicates it's a field of Ele[i], not an index)
    bad_real=$(echo "$bad_lines" | grep -vE 'Ele\[i\]\.nabr' || true)
    if [ -n "$bad_real" ]; then
        echo "FAIL: found Ele[neighbor-index] cross-element write/access in f_update() element loop"
        echo "$bad_real"
        fail=1
    fi
fi

# Check 3: pragma must be #pragma omp for schedule(static), no reduction /
# atomic / critical / dynamic / guided clauses. Anchor on `^[[:space:]]*#pragma`
# so block-comment lines (` * #pragma omp …`) and line-comment lines
# (`// #pragma omp …`) are excluded from the validator.
pragma=$(echo "$body" | grep -E '^[[:space:]]*#pragma omp' || true)
if [ -n "$pragma" ]; then
    if ! echo "$pragma" | grep -E 'for[[:space:]]+schedule\(static\)' >/dev/null; then
        echo "FAIL: pragma found but not '#pragma omp for schedule(static)' in f_update() element loop"
        echo "$pragma"
        fail=1
    fi
    if echo "$pragma" | grep -E 'reduction\(|atomic|critical|schedule\(dynamic|schedule\(guided' >/dev/null; then
        echo "FAIL: forbidden clause (reduction/atomic/critical/dynamic/guided) on f_update() element loop pragma"
        echo "$pragma"
        fail=1
    fi
fi

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "P1 element-owner locality check FAIL"
    exit 1
fi

echo "P1 element-owner locality check PASS"
echo "  function: $TARGET_FN @ lines $start..$end"
echo "  pragma: ${pragma:-<not yet present — Config A defaults>}"
exit 0
