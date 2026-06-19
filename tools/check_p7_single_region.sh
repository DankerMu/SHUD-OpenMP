#!/usr/bin/env bash
# tools/check_p7_single_region.sh — S2 P7 parallel-region count + pragma gates
#
# Spec authority:
#   openspec/changes/s2-strict-omp-full/specs/p7-final-fusion-omp-cutoff/spec.md
#     Requirement L45  "Single parallel region in MD_rhs_core.cpp (C7 strict gate)"
#       Scenario   L49-L54  grep -cE '^#pragma omp parallel\s*$|^#pragma omp parallel\s+default' == 1
#     Requirement L56  "`#pragma omp for` only, ban `parallel for` and dynamic/guided"
#       Scenario   L60-L63  grep -cE '^#pragma omp parallel for' == 0
#
# Why this script exists alongside tools/invariants.yaml:
#   The repo-wide invariant sweep tool (tools/check_invariant_sweep.sh) strips
#   lines whose first non-whitespace char is `#` before grep-ing. That filter
#   is correct for shell / yaml / python comments but ALSO strips C++ preprocessor
#   directives (`#pragma`, `#include`, `#define`), which makes the sweep blind
#   to `#pragma omp parallel for` / `#pragma omp critical` regressions. The
#   YAML entries `p7_single_parallel_region_strict` + `p5_no_reduction_atomic_critical`
#   remain in tools/invariants.yaml as the documented spec identifier (so a
#   reader of invariants.yaml sees the full 4-invariant table from issue #100);
#   this helper script is the ACTUAL enforcement for the `#pragma`-prefixed
#   regressions that the sweep tool cannot see.
#
# What this script enforces (4 gates):
#   1. EXACTLY 1 `#pragma omp parallel default(none)` region in MD_rhs_core.cpp
#      (positive count == 1; spec p7 Req L45 Scenario L51-L52)
#   2. ZERO `#pragma omp parallel for` (no second fork-join; spec L56 Scenario L60-L63)
#   3. ZERO `#pragma omp critical` (spec p5 owner-local gather + p7 Req L57)
#   4. ZERO `assert(false);` statement-form in OMP stubs (spec exec-policy-enum
#      Req L37-L40 NDEBUG safety — actual code form, not doc/comment mention)
#
# Gates 3+4 overlap with the YAML invariants `p5_no_reduction_atomic_critical`
# + `production_omp_abort_stub_ndebug_safe`; gate 4 IS effectively the YAML
# entry (token does not start with `#` so the sweep already catches it).
# Gate 3 is the duplicate that the YAML cannot see — kept here to make this
# script a single one-stop CI gate for the 4-tuple.
#
# Usage:
#   bash tools/check_p7_single_region.sh        # check all 4 gates
#
# Exit codes:
#   0  all 4 gates PASS
#   1  one or more gates FAIL (diagnostic to stderr)
#   2  setup error (source file missing)

set -eu

SCOPE="SHUD/src/Model/MD_rhs_core.cpp"

if [ ! -f "$SCOPE" ]; then
    echo "FAIL: $SCOPE not found" >&2
    exit 2
fi

fail=0

# -----------------------------------------------------------------------------
# Gate 1 — exactly 1 `#pragma omp parallel default(...)` region (spec p7 Req
# L45 + Scenario L49-L54). Pattern matches:
#   `^#pragma omp parallel$`                — bare parallel
#   `^#pragma omp parallel default(...)`    — parallel with default(none)
# but explicitly NOT `^#pragma omp parallel for ...` (which is gate 2).
# -----------------------------------------------------------------------------
count_parallel=$(grep -cE '^#pragma omp parallel\s*$|^#pragma omp parallel\s+default' "$SCOPE" || true)
if [ "$count_parallel" -ne 1 ]; then
    echo "FAIL: gate 1: expected exactly 1 \`#pragma omp parallel default(...)\` region in $SCOPE, found $count_parallel" >&2
    grep -nE '^#pragma omp parallel\s*$|^#pragma omp parallel\s+default' "$SCOPE" >&2 || true
    fail=1
fi

# -----------------------------------------------------------------------------
# Gate 2 — zero `#pragma omp parallel for` (spec p7 Req L56 + Scenario L60-L63).
# A `parallel for` would create a SECOND fork-join breaking C7.
# -----------------------------------------------------------------------------
count_parallel_for=$(grep -cE '^[[:space:]]*#pragma omp parallel for' "$SCOPE" || true)
if [ "$count_parallel_for" -ne 0 ]; then
    echo "FAIL: gate 2: \`#pragma omp parallel for\` reappeared in $SCOPE (expected 0, found $count_parallel_for)" >&2
    grep -nE '^[[:space:]]*#pragma omp parallel for' "$SCOPE" >&2 || true
    fail=1
fi

# -----------------------------------------------------------------------------
# Gate 3 — zero `#pragma omp critical` (spec p5 owner-local gather + p7 Req L57).
# Critical sections serialize a single thread at a time and break the bitwise
# A3a vs B1a-tag gate (FP reorder across threads under release-order). Scoped
# strictly to MD_rhs_core.cpp; MD_ET.cpp's pre-existing named critical block
# (#86) is OUT of scope.
# -----------------------------------------------------------------------------
count_critical=$(grep -cE '^[[:space:]]*#pragma omp critical' "$SCOPE" || true)
if [ "$count_critical" -ne 0 ]; then
    echo "FAIL: gate 3: \`#pragma omp critical\` reappeared in $SCOPE (expected 0, found $count_critical)" >&2
    grep -nE '^[[:space:]]*#pragma omp critical' "$SCOPE" >&2 || true
    fail=1
fi

# -----------------------------------------------------------------------------
# Gate 4 — zero `assert(false);` statement-form in $SCOPE (spec exec-policy-enum
# Req L37-L40 NDEBUG safety). The OMP stubs MUST use `std::abort()` because
# `-DNDEBUG` strips `assert(...)` to a no-op. Pattern uses trailing `;` to
# discriminate ACTUAL statement form from doc/comment mention like
# `assert(false) is forbidden ...` in the file header.
# -----------------------------------------------------------------------------
count_assert_false=$(grep -cE 'assert\(false\)[[:space:]]*;' "$SCOPE" || true)
if [ "$count_assert_false" -ne 0 ]; then
    echo "FAIL: gate 4: \`assert(false);\` statement-form reappeared in $SCOPE (expected 0, found $count_assert_false). Use std::abort() instead per spec exec-policy-enum L37-L40 NDEBUG safety." >&2
    grep -nE 'assert\(false\)[[:space:]]*;' "$SCOPE" >&2 || true
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "" >&2
    echo "P7 single-region gate FAIL — see gate-specific diagnostics above" >&2
    exit 1
fi

echo "P7 single-region gate PASS"
echo "  scope: $SCOPE"
echo "  gate 1: 1 \`#pragma omp parallel default(...)\` region (expected 1)"
echo "  gate 2: 0 \`#pragma omp parallel for\` directives"
echo "  gate 3: 0 \`#pragma omp critical\` directives"
echo "  gate 4: 0 \`assert(false);\` statement-form (use std::abort())"
exit 0
