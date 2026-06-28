#!/bin/bash
# tools/p8tune.D/spike_run.sh
#
# P8-tune.D KLU pattern-only spike (PR-0). Per-cell driver that chains the
# 3 spike binaries (dump_adjacency → fd_color_jacobian → klu_analyze_factor)
# for one (case, ordering, btf) cell.
#
# Per openspec change `p8tune-klu-spike` tasks.md §1.12.
#
# Usage:
#   bash spike_run.sh <case> <ordering> <btf>
#
# Example (Mac smoke):
#   cd tools/p8tune.D
#   ./spike_run.sh keliya amd 1
#
# Args:
#   case      ∈ {keliya, heihe, heihe_x4, heihe_x16}   per spec REQ-4
#   ordering  ∈ {natural, amd, colamd}                  per spec REQ-4
#   btf       ∈ {0, 1}                                  per spec REQ-4
#
# Env vars (optional):
#   OUTPUT_DIR  where per-cell logs land (default: ./output)
#   BASIN_ROOT  parent of Basins/<case>/ (default: ../../SHUD/Basins)
#
# Output (under $OUTPUT_DIR):
#   cell-<case>_<ordering>_btf<btf>.log     (combined stdout+stderr)
#
# Exit code: 0 if all 3 stages PASS or KLU_OOM_DETECTED (per REQ-5 OOM-as-data-point);
#            1 if dump_adjacency / fd_color_jacobian / klu_analyze_factor failed
#            with non-zero exit + no OOM detected.
#
# Refs:
#   openspec/changes/p8tune-klu-spike/tasks.md §1.12
#   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 <case> <ordering> <btf>" >&2
    echo "  case      ∈ {keliya, heihe, heihe_x4, heihe_x16}" >&2
    echo "  ordering  ∈ {natural, amd, colamd}" >&2
    echo "  btf       ∈ {0, 1}" >&2
    exit 2
fi

CASE="$1"
ORDERING="$2"
BTF="$3"

# Case-name whitelist (PR-0 reviewer round 1 finding c12 / F10 fix).
# Per spec REQ-4 Scenario "4-case definition", the spike matrix is restricted
# to exactly these 4 cases.
case "$CASE" in
    keliya|heihe|heihe_x4|heihe_x16) ;;
    *) echo "ERROR: case must be one of {keliya, heihe, heihe_x4, heihe_x16}, got '$CASE'" >&2; exit 2 ;;
esac
case "$ORDERING" in
    natural|amd|colamd) ;;
    *) echo "ERROR: ordering must be natural|amd|colamd, got '$ORDERING'" >&2; exit 2 ;;
esac
case "$BTF" in
    0|1) ;;
    *) echo "ERROR: btf must be 0|1, got '$BTF'" >&2; exit 2 ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-./output}"
BASIN_ROOT="${BASIN_ROOT:-../../SHUD/Basins}"
mkdir -p "$OUTPUT_DIR"

CELL_ID="${CASE}_${ORDERING}_btf${BTF}"
LOG="${OUTPUT_DIR}/cell-${CELL_ID}.log"
: > "$LOG"

# Spike binaries must be on PATH or in the cwd. Default: cwd (Mac smoke).
DUMP=./dump_adjacency
FDCJ=./fd_color_jacobian
KLU=./klu_analyze_factor

for b in "$DUMP" "$FDCJ" "$KLU"; do
    if [[ ! -x "$b" ]]; then
        echo "ERROR: $b not found or not executable; build via 'make shud_spike' first" | tee -a "$LOG" >&2
        exit 1
    fi
done

echo "=== spike_run.sh ${CELL_ID} ===" | tee -a "$LOG"
echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG"
echo "hostname: $(hostname)" | tee -a "$LOG"
echo "basin_root: $BASIN_ROOT" | tee -a "$LOG"

# Stage 1: dump_adjacency
echo "--- stage 1: dump_adjacency ---" | tee -a "$LOG"
"$DUMP" --case "$CASE" --basin-root "$BASIN_ROOT" 2>&1 | tee -a "$LOG"

# Stage 2: fd_color_jacobian (full FD probe, NOT --report-chi-only)
echo "--- stage 2: fd_color_jacobian ---" | tee -a "$LOG"
"$FDCJ" --case "$CASE" --basin-root "$BASIN_ROOT" 2>&1 | tee -a "$LOG"

# Stage 2b: brute-force dense FD + cross-check (keliya tool-correctness gate ONLY).
# Per spec REQ-3 Scenario "keliya tool-correctness gate via independent
# ground-truth reference": the FD-probed numeric J for keliya SHALL be
# cross-checked against a brute-force dense FD with relative error ≤ 1e-6
# per nonzero entry. NumY is small enough (1785) for an O(NumY) dense probe
# (~30s on Mac). Skipped for other cases where dense FD is intractable.
# Triggered ONLY for the tool-correctness cell (keliya, amd, btf=1).
# (PR-0 reviewer round 1 finding c01 / F1 fix.)
if [[ "$CASE" == "keliya" && "$ORDERING" == "amd" && "$BTF" == "1" ]]; then
    echo "--- stage 2b: fd_color_jacobian --brute-force-dense (keliya tool-correctness gate) ---" | tee -a "$LOG"
    "$FDCJ" --case "$CASE" --basin-root "$BASIN_ROOT" --brute-force-dense 2>&1 | tee -a "$LOG"
    echo "--- stage 2c: dense_fd_cross_check.py ---" | tee -a "$LOG"
    uv run python "$(dirname "$0")/dense_fd_cross_check.py" --case "$CASE" 2>&1 | tee -a "$LOG"
fi

# Stage 3: klu_analyze_factor
echo "--- stage 3: klu_analyze_factor ---" | tee -a "$LOG"
"$KLU" --case "$CASE" --basin-root "$BASIN_ROOT" --ordering "$ORDERING" --btf "$BTF" 2>&1 | tee -a "$LOG"

echo "=== spike_run.sh ${CELL_ID} done ===" | tee -a "$LOG"
echo "log: $LOG"
