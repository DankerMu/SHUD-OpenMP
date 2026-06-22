#!/usr/bin/env bash
# tools/check_sizeof/check_sizeof.sh — S5d 汇总验收 (#183) T8.1
#
# Spec: openspec/changes/b1b-baseline-completion/specs/s5d-data-layout-soa-numa/spec.md
#   Requirement "S5d 汇总验收 cache miss + NUMA 加速比"
#   Scenario "ElementHotData 字节占比":
#     sizeof(ElementHotData) / sizeof(_Element) < 0.20
#
# Compile + run the 1-TU sizeof emission program against the SHUD worktree
# headers (no SHUD lib link needed). Returns the program's exit code which
# is 0 iff ratio < 0.20.
#
# Usage:
#   bash tools/check_sizeof/check_sizeof.sh
# Or invoked from the outer Makefile / CI gate.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Allow optional CXX override (e.g. CXX=g++-12 on macOS); default works on
# both Apple clang and stock g++ on Linux.
export CXX="${CXX:-c++}"

# Clean rebuild for deterministic ratio (no stale .o from prior runs).
make -C "$HERE" clean >/dev/null
make -C "$HERE" all   >/dev/null

# Run; capture exit code from the program (0 = PASS, 1 = FAIL).
"$HERE/check_sizeof"
