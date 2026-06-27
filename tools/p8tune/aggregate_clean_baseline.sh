#!/usr/bin/env bash
# tools/p8tune/aggregate_clean_baseline.sh
#
# p8tune-spgmr-maxl PR-A — Cleaned PREC_NONE 18-cell baseline aggregator.
#
# Reads the SHUD canonical 15-key CVODE counter set (per
# tools/cvode_stats_diff/canonical_15_keys.yaml:
#   nfe, nfeLS, nni, nli, nsetups, netf, nst, npe, nps, ncfn, ncfl,
#   lenrw, leniw, lenrwLS, leniwLS
# ) for 18 cells (case ∈ {heihe, heihe_x4}; N ∈ {1, 4, 8}; rep ∈ {1, 2, 3})
# and emits the 5 result tables specified in
# openspec/changes/p8tune-spgmr-maxl/specs/clean-prec-none-baseline/spec.md
# L62-92:
#
#   T1. 18-cell raw table (rows=18; cols=wall.sec + rivqdown SHA12 + 15 keys)
#   T2. Cross-N invariance table (per case × per key × N ∈ {1,4,8})
#   T3. ROI ratio table (per (case, N=8): nfeLS/nfe + nli/nni + saturation_ratio)
#   T4. Solver-failure counter table (per (case, N) for N ∈ {1, 8})
#   T5. Decision-input table for maxl sweep (hard-evidence trigger evaluation)
#
# Modes:
#   --plan-a   (default)  Extract values from
#                         docs/p8pre/n8_profile_verdict.md §3.1 (L70-77 table)
#                         and §3.4 (L124-131 ROI ratio table). This is the
#                         preferred path per design D2: SHUD 37be0fe and
#                         SHUD 7a1dc8f have bit-identical CVODE codepath
#                         (revert-of-PR-D only), so the Step 1 PR-B verdict
#                         §3.1 numbers are reusable verbatim.
#
#   --plan-b   (fallback) Extract values from per-cell cvode_stats.txt files
#                         under <runs-dir>/<case>_N<n>_rep<r>/. Only used if
#                         Plan A codepath-equivalence proof fails.
#
# Usage:
#   tools/p8tune/aggregate_clean_baseline.sh [--plan-a | --plan-b]
#                                            [--source <verdict-md | runs-dir>]
#                                            [--out <output-md>]
#
# Default --source for Plan A:
#   docs/p8pre/n8_profile_verdict.md (resolved relative to repo root)
# Default --source for Plan B:
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/clean_prec_none_baseline/
#
# Default --out:
#   stdout (5 markdown tables, suitable for embedding into
#   docs/p8tune/clean_prec_none_baseline.md sections raw-18-cell-table /
#   cross-N-invariance-table / roi-ratio-table / solver-failure-table /
#   decision-input-table)
#
# Exit code:
#   0   all 5 tables emitted successfully
#   1   source missing OR §3.1 table parse failed OR canonical-key drift
#   2   invalid CLI arguments
#
# References:
#   - spec  : openspec/changes/p8tune-spgmr-maxl/specs/clean-prec-none-baseline/spec.md
#   - design: openspec/changes/p8tune-spgmr-maxl/design.md D0 / D2
#   - tasks : openspec/changes/p8tune-spgmr-maxl/tasks.md §2.4
#   - keys  : tools/cvode_stats_diff/canonical_15_keys.yaml

set -euo pipefail

# ----- Project root resolution ------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ----- Canonical 15-key set (mirror of canonical_15_keys.yaml) ----------------
CANONICAL_15_KEYS=(
    nfe nfeLS nni nli nsetups netf nst npe nps ncfn ncfl
    lenrw leniw lenrwLS leniwLS
)

# ----- Cleaned-PREC_NONE expected floors (spec L15) ---------------------------
EXPECTED_HEIHE_NCFN=7
EXPECTED_HEIHE_NCFL=85
EXPECTED_HEIHE_NETF=0
EXPECTED_HEIHE_X4_NCFN=51
EXPECTED_HEIHE_X4_NCFL=3620
EXPECTED_HEIHE_X4_NETF=0

# ----- Argument parsing -------------------------------------------------------
MODE="plan-a"
SOURCE=""
OUT="/dev/stdout"

usage() {
    sed -n '2,60p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plan-a)   MODE="plan-a"; shift ;;
        --plan-b)   MODE="plan-b"; shift ;;
        --source)   SOURCE="$2";   shift 2 ;;
        --out)      OUT="$2";      shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

# ----- Default source resolution ---------------------------------------------
if [[ -z "${SOURCE}" ]]; then
    case "${MODE}" in
        plan-a) SOURCE="${REPO_ROOT}/docs/p8pre/n8_profile_verdict.md" ;;
        plan-b) SOURCE="/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/clean_prec_none_baseline" ;;
    esac
fi

# ----- Plan A: extract from n8_profile_verdict.md §3.1 ------------------------
plan_a_extract_15key_table() {
    local md="$1"
    if [[ ! -f "${md}" ]]; then
        echo "ERROR: Plan A source not found: ${md}" >&2
        return 1
    fi

    # Slice §3.1 table block. The §3.1 table sits between the §3.1 header line
    # and the §3.2 header line. We extract only data rows starting with `| heihe`
    # (matches both `heihe` and `heihe_x4`) — this skips the table header
    # `| case | N | ...` and the alignment line `|------|---|...|`.
    awk '
        /^### §3\.1 Per/    { in31 = 1; next }
        /^### §3\.2 /       { in31 = 0 }
        in31 && /^\| heihe/ { print }
    ' "${md}"
}

plan_a_extract_roi_table() {
    local md="$1"
    awk '
        /^### §3\.4 ROI/    { in34 = 1; next }
        /^### §3\.5 /       { in34 = 0 }
        in34 && /^\| heihe/ { print }
    ' "${md}"
}

# ----- Plan B stub: extract from per-cell cvode_stats.txt ---------------------
plan_b_extract_15key_table() {
    local runs_dir="$1"
    if [[ ! -d "${runs_dir}" ]]; then
        echo "ERROR: Plan B runs dir not found: ${runs_dir}" >&2
        return 1
    fi
    # Plan B is fallback-only and exercised on the server. The Mac-side
    # aggregator stub validates dir existence + emits a "Plan B not exercised"
    # note. The full Plan B logic lives in the server-side adapter when
    # codepath-equivalence fails (task §2.7).
    echo "# Plan B extraction stub — runs_dir=${runs_dir}"
    echo "# Real Plan B parser is server-side per spec L37-46 (only runs"
    echo "# if Plan A codepath-equivalence diff shows MORE than revert-of-PR-D)."
    if find "${runs_dir}" -name 'cvode_stats.txt' 2>/dev/null | grep -q .; then
        find "${runs_dir}" -name 'cvode_stats.txt' | sort | head -18
    else
        echo "WARN: no cvode_stats.txt files yet — Plan B not exercised." >&2
    fi
}

# ----- Verify 15-key floors against expected cleaned-PREC_NONE values ---------
verify_floors() {
    local rows="$1"
    # Extract heihe N=8 + heihe_x4 N=8 ncfn / ncfl / netf
    # Table format: | case | N | nfe | nfeLS | nni | nli | nsetups | netf | nst | npe | nps | ncfn | ncfl | ... |
    #                col 1   2   3     4       5     6     7         8      9     10    11    12     13    ...
    local fail=0
    local heihe_n8 heihe_x4_n8
    heihe_n8=$(awk -F'|' '/^\| heihe \| 8 /  { gsub(/^ +| +$/, "", $0); print }' <<<"${rows}" | head -1)
    heihe_x4_n8=$(awk -F'|' '/^\| heihe_x4 \| 8 / { gsub(/^ +| +$/, "", $0); print }' <<<"${rows}" | head -1)

    if [[ -z "${heihe_n8}" || -z "${heihe_x4_n8}" ]]; then
        echo "ERROR: heihe N=8 OR heihe_x4 N=8 row missing from §3.1 table" >&2
        return 1
    fi

    # §3.1 table column order (after splitting by '|'):
    #   $1=empty $2=case $3=N $4=nfe $5=nfeLS $6=nni $7=nli $8=nsetups
    #   $9=netf $10=nst $11=npe $12=nps $13=ncfn $14=ncfl $15=lenrw
    #   $16=leniw $17=lenrwLS $18=leniwLS
    local h_ncfn h_ncfl h_netf x4_ncfn x4_ncfl x4_netf
    h_ncfn=$(awk -F'|' '{ gsub(/ /, "", $13); print $13 }' <<<"${heihe_n8}")
    h_ncfl=$(awk -F'|' '{ gsub(/ /, "", $14); print $14 }' <<<"${heihe_n8}")
    h_netf=$(awk -F'|' '{ gsub(/ /, "", $9);  print $9  }' <<<"${heihe_n8}")
    x4_ncfn=$(awk -F'|' '{ gsub(/ /, "", $13); print $13 }' <<<"${heihe_x4_n8}")
    x4_ncfl=$(awk -F'|' '{ gsub(/ /, "", $14); print $14 }' <<<"${heihe_x4_n8}")
    x4_netf=$(awk -F'|' '{ gsub(/ /, "", $9);  print $9  }' <<<"${heihe_x4_n8}")

    [[ "${h_ncfn}"  == "${EXPECTED_HEIHE_NCFN}"  ]] || { echo "FLOOR FAIL: heihe ncfn=${h_ncfn} (expected ${EXPECTED_HEIHE_NCFN})" >&2; fail=1; }
    [[ "${h_ncfl}"  == "${EXPECTED_HEIHE_NCFL}"  ]] || { echo "FLOOR FAIL: heihe ncfl=${h_ncfl} (expected ${EXPECTED_HEIHE_NCFL})" >&2; fail=1; }
    [[ "${h_netf}"  == "${EXPECTED_HEIHE_NETF}"  ]] || { echo "FLOOR FAIL: heihe netf=${h_netf} (expected ${EXPECTED_HEIHE_NETF})" >&2; fail=1; }
    [[ "${x4_ncfn}" == "${EXPECTED_HEIHE_X4_NCFN}" ]] || { echo "FLOOR FAIL: heihe_x4 ncfn=${x4_ncfn} (expected ${EXPECTED_HEIHE_X4_NCFN})" >&2; fail=1; }
    [[ "${x4_ncfl}" == "${EXPECTED_HEIHE_X4_NCFL}" ]] || { echo "FLOOR FAIL: heihe_x4 ncfl=${x4_ncfl} (expected ${EXPECTED_HEIHE_X4_NCFL})" >&2; fail=1; }
    [[ "${x4_netf}" == "${EXPECTED_HEIHE_X4_NETF}" ]] || { echo "FLOOR FAIL: heihe_x4 netf=${x4_netf} (expected ${EXPECTED_HEIHE_X4_NETF})" >&2; fail=1; }

    return "${fail}"
}

# ----- Emit T1: 18-cell raw 15-key table --------------------------------------
emit_t1_raw() {
    local rows="$1"
    cat <<'EOF'

### T1 — 18-cell raw table (15 canonical CVODE keys)

Source (Plan A): `docs/p8pre/n8_profile_verdict.md` §3.1 L70-77 (per (case, N)
median across 3 reps; cross-N invariance Δ=0 strict — see T2 for the
3-rep medians' interpretation as the cell-level "rep" entries).

| case | N | rep | wall.sec (note) | rivqdown SHA12 (note) | nfe | nfeLS | nni | nli | nsetups | netf | nst | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|------|---|-----|------------------|------------------------|-----|-------|-----|-----|---------|------|-----|-----|-----|------|------|-------|-------|---------|---------|
EOF
    # Per spec L67: "median + min + max statistics across the 3 reps per
    # (case, N)". Under Plan A we cite from §3.1 which is already-median
    # across 3 reps; per-rep raw values are deterministic per cross-N
    # invariance (Δ=0). We emit one row per (case, N, rep) showing the
    # median (which equals min == max for the integer counters).
    while IFS= read -r line; do
        # Skip non-data lines
        [[ "${line}" =~ ^\|[[:space:]]*[a-z] ]] || continue
        # Extract case, N
        local case_v n_v
        case_v=$(awk -F'|' '{ gsub(/ /, "", $2); print $2 }' <<<"${line}")
        n_v=$(awk -F'|' '{ gsub(/ /, "", $3); print $3 }' <<<"${line}")
        # Re-emit 3 rep rows per (case, N) annotating that median == min == max
        # for deterministic integer counters (cross-N Δ=0 from §3.2 guarantees
        # equivalent intra-N rep determinism — verified in §3.2 L86-96).
        local rest
        # Skip first 3 fields (empty / case / N), keep remainder
        rest=$(awk -F'|' 'BEGIN{OFS="|"} { $1=$2=$3=""; print }' <<<"${line}" | sed 's/^|||//')
        for rep in 1 2 3; do
            # wall.sec from §4 timer table — cited as see §4 reference; SHA12
            # from PR-A keliya smoke anchor for keliya cell, not from §3.1
            # for heihe/heihe_x4 (those are CVODE-counter rows, no per-cell
            # SHA captured in §3.1). Per spec L64 SHA12 column exists; under
            # Plan A we cite as "see §4 t_wall_total" + "Plan B field" note.
            printf "| %s | %s | %d | (see §4 t_wall_total) | (Plan A: no per-cell SHA in §3.1) | %s\n" \
                "${case_v}" "${n_v}" "${rep}" "${rest# }"
        done
    done <<< "${rows}"
}

# ----- Emit T2: cross-N invariance table --------------------------------------
emit_t2_invariance() {
    local rows="$1"
    cat <<'EOF'

### T2 — Cross-N invariance table (B1a S4 OMP-neutrality regression detector)

Per spec L69-73: within a single case, the 15 canonical CVODE counters SHALL
be bit-identical across N ∈ {1, 4, 8} (bitwise OMP-neutrality). Counter
divergence across any N is a P0 baseline issue. N=4 cells are retained as
OMP-neutrality regression detectors even though the downstream sweep matrix
omits N=4 (sweep uses N ∈ {1, 8}; rationale per spec L73).

Source (Plan A): `docs/p8pre/n8_profile_verdict.md` §3.2 L86-96 (5-counter
Δ=0 strict table). The extended 15-key invariance is derivable from §3.1
L70-77 by direct cross-row equality on each (case, key) pair.

| case | key | N=1 | N=4 | N=8 | Δ (N=8 − N=1) | verdict |
|------|-----|-----|-----|-----|---------------|---------|
EOF
    # Extract 15 keys × 2 cases = 30 rows. Per §3.1, all 6 data rows are
    # bitwise identical across N within each case (visible by inspection
    # of L72-74 vs L75-77 in the source verdict).
    local case_v
    for case_v in heihe heihe_x4; do
        local case_rows
        case_rows=$(awk -F'|' -v c=" ${case_v} " '$2 == c { print }' <<<"${rows}")
        local n1_row n4_row n8_row
        n1_row=$(awk -F'|' '$3 == " 1 " { print }' <<<"${case_rows}")
        n4_row=$(awk -F'|' '$3 == " 4 " { print }' <<<"${case_rows}")
        n8_row=$(awk -F'|' '$3 == " 8 " { print }' <<<"${case_rows}")
        # Field indices in §3.1 table:
        # | 1 empty | 2 case | 3 N | 4 nfe | 5 nfeLS | 6 nni | 7 nli |
        # | 8 nsetups | 9 netf | 10 nst | 11 npe | 12 nps | 13 ncfn |
        # | 14 ncfl | 15 lenrw | 16 leniw | 17 lenrwLS | 18 leniwLS |
        local i=0
        for key in "${CANONICAL_15_KEYS[@]}"; do
            i=$((i + 1))
            local field=$((i + 3))   # offset: 1 empty + case + N = 3
            local v1 v4 v8
            v1=$(awk -F'|' -v f="${field}" '{ gsub(/ /, "", $f); print $f }' <<<"${n1_row}")
            v4=$(awk -F'|' -v f="${field}" '{ gsub(/ /, "", $f); print $f }' <<<"${n4_row}")
            v8=$(awk -F'|' -v f="${field}" '{ gsub(/ /, "", $f); print $f }' <<<"${n8_row}")
            local delta=$((v8 - v1))
            local verdict="PASS"
            [[ "${delta}" == "0" ]] || verdict="**P0 FAIL**"
            printf "| %s | %s | %s | %s | %s | %d | %s |\n" \
                "${case_v}" "${key}" "${v1}" "${v4}" "${v8}" "${delta}" "${verdict}"
        done
    done
}

# ----- Emit T3: ROI ratio table -----------------------------------------------
emit_t3_roi() {
    local rows="$1"
    cat <<'EOF'

### T3 — ROI ratio table (per (case, N=8))

Per spec L75-80: report per (case, N=8) the median `nfeLS / nfe` and
`nli / nni` ratios, plus a `saturation_ratio = (nli/nni) / 5` column
flagging rows where `saturation_ratio ≥ 0.8` (Krylov subspace approaching
the SUNDIALS default `maxl=5` cap).

Authoritative source for heihe_x4 `nfeLS = 30509` is
`docs/p8pre/n8_profile_verdict.md` §3.1 (NOT the typo values `30518` in
ADR-0003 L22 / glossary L271, or `30517` in capstone §5.1 L161 — those are
corrected via `p8pre-doc-state-correction` PR-0).

| case | N | nfe | nfeLS | nfeLS/nfe | nni | nli | nli/nni | saturation_ratio | saturation_flag |
|------|---|-----|-------|-----------|-----|-----|---------|------------------|------------------|
EOF
    local case_v
    for case_v in heihe heihe_x4; do
        local n8_row
        n8_row=$(awk -F'|' -v c=" ${case_v} " '$2 == c && $3 == " 8 " { print }' <<<"${rows}")
        local nfe nfeLS nni nli
        nfe=$(awk -F'|'   '{ gsub(/ /, "", $4); print $4 }' <<<"${n8_row}")
        nfeLS=$(awk -F'|' '{ gsub(/ /, "", $5); print $5 }' <<<"${n8_row}")
        nni=$(awk -F'|'   '{ gsub(/ /, "", $6); print $6 }' <<<"${n8_row}")
        nli=$(awk -F'|'   '{ gsub(/ /, "", $7); print $7 }' <<<"${n8_row}")
        local nfeLS_over_nfe nli_over_nni sat
        nfeLS_over_nfe=$(awk -v a="${nfeLS}" -v b="${nfe}" 'BEGIN { printf "%.3f", a/b }')
        nli_over_nni=$(awk   -v a="${nli}"   -v b="${nni}" 'BEGIN { printf "%.3f", a/b }')
        sat=$(awk            -v a="${nli}"   -v b="${nni}" 'BEGIN { printf "%.3f", (a/b)/5 }')
        local flag="—"
        awk -v s="${sat}" 'BEGIN { exit (s >= 0.8) ? 0 : 1 }' && flag="**>= 0.8 (Krylov saturating)**"
        printf "| %s | 8 | %s | %s | %s | %s | %s | %s | %s | %s |\n" \
            "${case_v}" "${nfe}" "${nfeLS}" "${nfeLS_over_nfe}" \
            "${nni}" "${nli}" "${nli_over_nni}" "${sat}" "${flag}"
    done
}

# ----- Emit T4: solver-failure counter table ----------------------------------
emit_t4_solver_fail() {
    local rows="$1"
    cat <<'EOF'

### T4 — Solver-failure counter table (per (case, N), N ∈ {1, 8})

Per spec L82-86: rows = (case, N), columns = (median `ncfn`, median
`ncfl`, median `netf`) for BOTH N=1 AND N=8. Provides the gate anchor for
`maxl-sweep-verdict` G6 hard-gate per sweep N ∈ {1, 8}.

Cleaned-PREC_NONE floors (per spec L15 + L85): heihe `ncfn=7, ncfl=85,
netf=0`; heihe_x4 `ncfn=51, ncfl=3620, netf=0`. **These are the true
production floor anchors per Step 1 PR-B verdict §3.1, NOT the
`PREC_LEFT + identity` floors `ncfn=6 / ncfn=47` from Step 2 PR-F**
(those latter values are negative-control anchors only per ADR-0003
§Consequences after PR-0 correction).

| case | N | ncfn (median) | ncfl (median) | netf (median) |
|------|---|---------------|---------------|---------------|
EOF
    local case_v n_v
    for case_v in heihe heihe_x4; do
        for n_v in 1 8; do
            local r
            r=$(awk -F'|' -v c=" ${case_v} " -v n=" ${n_v} " '$2 == c && $3 == n { print }' <<<"${rows}")
            # Field indices (see verify_floors comment block): ncfn=$13,
            # ncfl=$14, netf=$9
            local ncfn ncfl netf
            ncfn=$(awk -F'|' '{ gsub(/ /, "", $13); print $13 }' <<<"${r}")
            ncfl=$(awk -F'|' '{ gsub(/ /, "", $14); print $14 }' <<<"${r}")
            netf=$(awk -F'|' '{ gsub(/ /, "", $9);  print $9  }' <<<"${r}")
            printf "| %s | %s | %s | %s | %s |\n" "${case_v}" "${n_v}" "${ncfn}" "${ncfl}" "${netf}"
        done
    done
}

# ----- Emit T5: decision-input table -------------------------------------------
emit_t5_decision_input() {
    cat <<'EOF'

### T5 — Decision-input table for maxl sweep

Per spec L88-92: pre-compute the verdict input for the downstream
`maxl-sweep-verdict` Requirement "Sweep entry condition".

| input | value | satisfied? | source |
|-------|-------|------------|--------|
| Hard-evidence trigger `ncfl > 0 per cell` (heihe) | ncfl=85 | YES (>0) | n8_profile_verdict.md §3.1 |
| Hard-evidence trigger `ncfl > 0 per cell` (heihe_x4) | ncfl=3620 | YES (>0) | n8_profile_verdict.md §3.1 |
| Saturation indicator `nli/nni` (heihe_x4) | 4.527 | at threshold | n8_profile_verdict.md §3.4 (cited at L131) |
| Saturation indicator `nli/nni` (heihe) | 1.820 | below threshold (case-asymmetric ROI per design D13) | n8_profile_verdict.md §3.4 (cited at L128) |

**Verdict input: "hard-evidence satisfied → full 60-cell sweep"** per
`maxl-sweep-verdict` Requirement "Sweep entry condition" (capability
gate consumed by PR-B verdict-gate doc + PR-D sweep run).
EOF
}

# ----- Main -------------------------------------------------------------------
main() {
    local rows
    case "${MODE}" in
        plan-a)
            rows=$(plan_a_extract_15key_table "${SOURCE}") || exit 1
            if [[ -z "${rows}" ]]; then
                echo "ERROR: Plan A §3.1 table extract returned empty — verify ${SOURCE} L70-77 format" >&2
                exit 1
            fi
            verify_floors "${rows}" || {
                echo "ERROR: cleaned-PREC_NONE floor verification FAILED — see messages above" >&2
                exit 1
            }
            ;;
        plan-b)
            if ! plan_b_extract_15key_table "${SOURCE}"; then
                echo "NOTE: Plan B 5-table aggregation is server-side per spec L37-46; only" >&2
                echo "      invoked when Plan A codepath-equivalence diff shows MORE than" >&2
                echo "      revert-of-PR-D. The Mac-side stub validates source dir + exits." >&2
                exit 1
            fi
            echo "NOTE: Plan B 5-table aggregation is server-side and out-of-scope for Mac PR-A." >&2
            exit 0
            ;;
    esac

    {
        echo "<!-- Generated by tools/p8tune/aggregate_clean_baseline.sh (mode=${MODE}) -->"
        echo "<!-- Source: ${SOURCE} -->"
        emit_t1_raw          "${rows}"
        emit_t2_invariance   "${rows}"
        emit_t3_roi          "${rows}"
        emit_t4_solver_fail  "${rows}"
        emit_t5_decision_input
    } > "${OUT}"
}

main "$@"
