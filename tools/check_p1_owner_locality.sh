#!/usr/bin/env bash
# tools/check_p1_owner_locality.sh — S2 P1 owner-locality grep gate
#
# Enforces openspec/changes/s2-strict-omp-full/specs/p1-rhs-update-owner-local-omp/spec.md
# Scenario "Element state update only writes Ele[i] owner":
#   - Element loop body writes only use loop index `i`
#   - No `Ele[i+1]` / `Ele[i-1]` / `Ele[inabr]` / `Ele[Ele[i].nabr[j]-1]` patterns
#   - No reduction / atomic / critical / nowait clauses on the owner-local pragma
#
# SCOPE retargeted (R1 cand-02): grep gate now scopes the canonical runtime
# `Model_Data::rhs_update()` in SHUD/src/Model/MD_rhs_core.cpp (NOT the legacy
# `f_update()` in MD_update.cpp). The default Config A + Config E builds dispatch
# `f.cpp::f()` → `rhs_core(Y, DY, t, ExecPolicy::Serial)` → `rhs_update()`;
# the legacy path is only reachable when `LEGACY_RHS=1` Makefile macro is
# explicitly set, which is not the default CI Config matrix.
#
# Exit 0 on PASS; exit 1 on any forbidden pattern found.
# Run from the outer repo root.

set -euo pipefail

SCOPE="SHUD/src/Model/MD_rhs_core.cpp"
TARGET_FN="Model_Data::rhs_update"

if [ ! -f "$SCOPE" ]; then
    echo "FAIL: $SCOPE not found"
    exit 1
fi

fail=0

# Extract the rhs_update() body — from "void Model_Data::rhs_update(...)" up to
# the "}//end of for j=1:NumEle" comment that anchors end of element loop.
start=$(grep -n '^void Model_Data::rhs_update(' "$SCOPE" | head -1 | cut -d: -f1)
if [ -z "$start" ]; then
    echo "FAIL: rhs_update() function start not found in $SCOPE"
    exit 1
fi
end=$(awk -v s="$start" 'NR>s && /end of for j=1:NumEle/{print NR; exit}' "$SCOPE")
if [ -z "$end" ]; then
    echo "FAIL: cannot locate end of element loop comment in rhs_update()"
    exit 1
fi

body=$(sed -n "${start},${end}p" "$SCOPE")

# Strip C++ line-comment-only lines (`^\s*//…`) and block-comment continuation
# lines (`^\s*\*`) before structural checks so the legacy dead-code blocks
# (e.g. the `// Ele[Ele[i].nabr[j]-1]` lines kept as mass-balance reference)
# do not false-positive on Check 1 / Check 2. Inline block-comment spans
# (`/* … */` on the same line as code) are stripped in place so a bypass like
# `/* dead code */ Ele[i+1].QBC = 1.0;` is correctly caught (R1 cand-04).
# Multi-line `/* … */` spans are rare in this codebase and not handled here.
# This filter only affects Check 1 / Check 2; Check 3 keeps the anchored
# `^[[:space:]]*#pragma omp` so block-comment pragma mentions stay excluded.
body_live=$(echo "$body" | sed -E 's|/\*[^*]*\*/||g' | grep -vE '^[[:space:]]*//' | grep -vE '^[[:space:]]*\*')

# Check 1: no Ele[i+1] / Ele[i-1] / Ele[i+anything] / Ele[i-anything]
if echo "$body_live" | grep -E 'Ele\[i[[:space:]]*[\+\-]' >/dev/null; then
    echo "FAIL: found Ele[i+/-…] cross-element index in rhs_update() element loop"
    echo "$body_live" | grep -nE 'Ele\[i[[:space:]]*[\+\-]'
    fail=1
fi

# Check 2: every `Ele[` subscript must be exactly `Ele[i]` (own element index).
# Forbidden forms include `Ele[Ele[i].nabr[0]-1]` (nested-bracket neighbor
# lookup), `Ele[inabr]`, `Ele[nabr]`, and anything else where the outer
# subscript is not the bare loop index `i`. `Ele[i].nabr[j]` is allowed as
# field READ (the outer subscript IS `Ele[i]`; the trailing `.nabr[j]` is a
# field/array access on the resolved Ele struct, used inside conditionals
# like `Ele[i].nabr[j] > 0`).
#
# R1 cand-01 fix: the previous `Ele\[[^]]*nabr[^]]*\]` pattern uses a
# `[^]]`-anchored class that stops at the first `]`, so nested-bracket forms
# like `Ele[Ele[i].nabr[0]-1]` silently slipped through. New strategy: strip
# allowed `Ele[i]` substrings (the ONLY allowed outer subscript form), then
# any residual `Ele[` indicates a forbidden indexing form. Check 1 already
# catches `Ele[i+/-N]` separately so we don't need to special-case those here.
body_filtered=$(echo "$body_live" | sed -E 's|Ele\[i\]||g')
if echo "$body_filtered" | grep -E 'Ele\[' >/dev/null; then
    bad_real=$(echo "$body_filtered" | grep -nE 'Ele\[' || true)
    if [ -n "$bad_real" ]; then
        echo "FAIL: found Ele[…] subscript that is not the bare Ele[i] owner index in rhs_update() element loop"
        echo "(showing post-Ele[i]-strip view to highlight the offending residual subscript)"
        echo "$bad_real"
        fail=1
    fi
fi

# Check 3: pragma must be #pragma omp for schedule(static), no reduction /
# atomic / critical / dynamic / guided / nowait clauses. Anchor on
# `^[[:space:]]*#pragma` so block-comment lines (` * #pragma omp …`) and
# line-comment lines (`// #pragma omp …`) are excluded from the validator.
#
# R1 cand-03 fix: `nowait` added to the forbidden-clause regex. Per spec L20
# + design §D11 P5 三层依赖模型, nowait on the element-owner stage would
# break the barrier sequencing that P7 fusion (#99) depends on — downstream
# stages read `uYsf[i] / uYus[i] / uYgw[i] / Ele[i].QBC` and must observe
# the element-owner stage's writes before starting.
pragma=$(echo "$body" | grep -E '^[[:space:]]*#pragma omp' || true)
if [ -n "$pragma" ]; then
    if ! echo "$pragma" | grep -E 'for[[:space:]]+schedule\(static\)' >/dev/null; then
        echo "FAIL: pragma found but not '#pragma omp for schedule(static)' in rhs_update() element loop"
        echo "$pragma"
        fail=1
    fi
    if echo "$pragma" | grep -E 'reduction\(|atomic|critical|schedule\(dynamic|schedule\(guided|nowait\b' >/dev/null; then
        echo "FAIL: forbidden clause (reduction/atomic/critical/dynamic/guided/nowait) on rhs_update() element loop pragma"
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
