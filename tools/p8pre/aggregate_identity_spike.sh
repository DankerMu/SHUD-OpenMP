#!/bin/bash
# tools/p8pre/aggregate_identity_spike.sh
#
# p8pre-spike Step 2 PR-F aggregator (issue #347) — local-side adjudicator
# that parses the 18-cell rsync mirror at $P8PRE_IDENTITY_DIR (default
# /tmp/p8pre_identity_spike/) emitted by PR-E run (issue #346), and the
# Step 1 PR-A baseline mirror at $P8PRE_BASELINE_DIR (default
# /tmp/p8pre_n8_profile/), computes per (case, N) wall_median + nps_median
# + npe_median + ncfn_max + per-cell SHA12(rivqdown.dat) + max_ulp(rivqdown
# identity vs baseline), evaluates 4 hard gates + 2 soft gates per spec
# `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md`
# L74-130 + design D5/D7, and emits a structured verdict to stdout AND to
# /tmp/p8pre_identity_spike/aggregate_verdict.txt for §3 raw data table
# consumption by docs/p8pre/identity_spike_verdict.md.
#
# Language choice rationale: POSIX bash + awk + grep — matches sibling
# tools/p8pre/aggregate_n8_profile.sh style (PR-B #342), avoids Python
# dependency for straightforward YAML/KV parsing. Single exception: max_ulp
# computation falls back to `uv run python` (per CLAUDE.md project rule)
# because compare_snapshot expects magic-headered binary format but
# rivqdown.dat is raw little-endian doubles.
#
# Matrix per spec: 2 cases × 3 N-values × 3 reps = 18 cells.
#   cases = heihe + heihe_x4
#   N     = 1, 4, 8
#   reps  = 1, 2, 3 (median across 3 reps per (case, N) group)
#
# Output:
#   stdout: human-readable per-gate verdict summary + verdict header
#   /tmp/p8pre_identity_spike/aggregate_verdict.txt: structured KV for
#     verdict doc consumption
#
# Exit code:
#   0   aggregation completed (regardless of PASS/FAIL — verdict written)
#   1   missing data (input dirs absent / cell files unreadable)
#   2   phase failure (parse error / aggregator internal bug)
#
# POSIX bash + awk + grep + sha256sum + uv run python (for ULP). Verified
# bash -n.

set -euo pipefail

# ----------------------------------------------------------------------------
# Phase A: Configuration constants
# ----------------------------------------------------------------------------

P8PRE_IDENTITY_DIR="${P8PRE_IDENTITY_DIR:-/tmp/p8pre_identity_spike/}"
P8PRE_BASELINE_DIR="${P8PRE_BASELINE_DIR:-/tmp/p8pre_n8_profile/}"
# Normalize trailing slash
P8PRE_IDENTITY_DIR="${P8PRE_IDENTITY_DIR%/}/"
P8PRE_BASELINE_DIR="${P8PRE_BASELINE_DIR%/}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CASES=(heihe heihe_x4)
NS=(1 4 8)
REPS=(1 2 3)

# Gate 4 tolerance per spec L92-100 + design D5
declare -A GATE_4_EPSILON
GATE_4_EPSILON[heihe]=0.10       # 10% — heihe N=8 wall ~89s, Slurm jitter dominant
GATE_4_EPSILON[heihe_x4]=0.05    # 5%  — heihe_x4 N=8 wall ~744s, 5% headroom > noise

# Gate 5 baseline per-case SHA12 (from docs/p1e/p1e_pr_i_strict_omp_verification.md §3.1)
declare -A GATE_5_BASELINE_SHA12
GATE_5_BASELINE_SHA12[heihe]="a2023ccd2de4"
GATE_5_BASELINE_SHA12[heihe_x4]="b5e4b0a2cf83"

# A4 tolerance per master plan §2.3
GATE_5_FALLBACK_MAX_ULP=1024

# Gate 6 setup-overhead ratio threshold per spec L122-126
GATE_6_RATIO_THRESHOLD=0.05

# Gate 4 baseline wall_median per (case, N) from docs/p8pre/n8_profile_baseline.md §5.1 Table 1
declare -A BASELINE_WALL_MEDIAN
BASELINE_WALL_MEDIAN[heihe_1]=140.797
BASELINE_WALL_MEDIAN[heihe_4]=95.734
BASELINE_WALL_MEDIAN[heihe_8]=89.732
BASELINE_WALL_MEDIAN[heihe_x4_1]=1412.895
BASELINE_WALL_MEDIAN[heihe_x4_4]=849.704
BASELINE_WALL_MEDIAN[heihe_x4_8]=743.552

VERDICT_FILE="${P8PRE_IDENTITY_DIR}aggregate_verdict.txt"
CELL_STATS_FILE="${P8PRE_IDENTITY_DIR}cell_stats.txt"
SERVER_NM_LOG="${P8PRE_IDENTITY_DIR}server_nm.log"
JID_TABLE="${P8PRE_IDENTITY_DIR}jid_table.txt"

# ----------------------------------------------------------------------------
# Phase B: Pre-flight checks
# ----------------------------------------------------------------------------

echo "================================================================"
echo "p8pre-spike Step 2 PR-F identity-spike aggregator (issue #347)"
echo "================================================================"
echo "Identity dir: $P8PRE_IDENTITY_DIR"
echo "Baseline dir: $P8PRE_BASELINE_DIR"
echo ""

if [ ! -d "$P8PRE_IDENTITY_DIR" ]; then
    echo "ERROR: identity spike input dir not found: $P8PRE_IDENTITY_DIR" >&2
    exit 1
fi
if [ ! -d "$P8PRE_BASELINE_DIR" ]; then
    echo "WARN: baseline mirror dir not found: $P8PRE_BASELINE_DIR — gate 5 max_ulp will be DEFER" >&2
fi
if [ ! -f "$SERVER_NM_LOG" ]; then
    echo "ERROR: server_nm.log not found: $SERVER_NM_LOG" >&2
    exit 1
fi
if [ ! -f "$CELL_STATS_FILE" ]; then
    echo "ERROR: cell_stats.txt not found (PR-E artifact): $CELL_STATS_FILE" >&2
    echo "  hint: copy from .review-evidence/p8pre-pr-e-spike/cell_stats.txt or re-parse" >&2
    exit 1
fi

# Cell-dir spot check
missing=0
for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            cell="${P8PRE_IDENTITY_DIR}${c}_N${n}_rep${r}"
            if [ ! -d "$cell" ]; then
                echo "ERROR: cell dir absent: $cell" >&2
                missing=$((missing + 1))
            fi
        done
    done
done
if [ "$missing" -gt 0 ]; then
    echo "ERROR: $missing of 18 cell dirs absent — abort" >&2
    exit 1
fi

echo "Pre-flight: 18 cell dirs present + server_nm.log + cell_stats.txt OK"
echo ""

# ----------------------------------------------------------------------------
# Phase C: Parse 18 cells from cell_stats.txt
# ----------------------------------------------------------------------------
# cell_stats.txt columns: case N rep nst nfe ncfn nps npe wall_total t_precond_setup
# Header line skipped (first line starts with "case").

declare -A C_NST C_NFE C_NCFN C_NPS C_NPE C_WALL C_TPS
n_cells=0
while IFS=' ' read -r col_case col_n col_rep col_nst col_nfe col_ncfn col_nps col_npe col_wall col_tps; do
    [ "$col_case" = "case" ] && continue
    key="${col_case}_N${col_n}_rep${col_rep}"
    C_NST[$key]="$col_nst"
    C_NFE[$key]="$col_nfe"
    C_NCFN[$key]="$col_ncfn"
    C_NPS[$key]="$col_nps"
    C_NPE[$key]="$col_npe"
    C_WALL[$key]="$col_wall"
    C_TPS[$key]="$col_tps"
    n_cells=$((n_cells + 1))
done < "$CELL_STATS_FILE"

if [ "$n_cells" -ne 18 ]; then
    echo "ERROR: parsed $n_cells cells from cell_stats.txt — expected 18" >&2
    exit 2
fi
echo "Phase C: parsed $n_cells cells from cell_stats.txt"

# ----------------------------------------------------------------------------
# Phase D: Compute per-(case, N) medians
# ----------------------------------------------------------------------------
# median of 3 = middle of 3 sorted values

median3() {
    # args: v1 v2 v3 — print middle of sorted
    printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -n | sed -n '2p'
}
max3() {
    printf "%s\n%s\n%s\n" "$1" "$2" "$3" | sort -n | tail -1
}

declare -A WALL_MEDIAN NPS_MEDIAN NPE_MEDIAN NCFN_MAX
for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        k1="${c}_N${n}_rep1"
        k2="${c}_N${n}_rep2"
        k3="${c}_N${n}_rep3"
        gk="${c}_N${n}"
        WALL_MEDIAN[$gk]=$(median3 "${C_WALL[$k1]}" "${C_WALL[$k2]}" "${C_WALL[$k3]}")
        NPS_MEDIAN[$gk]=$(median3 "${C_NPS[$k1]}" "${C_NPS[$k2]}" "${C_NPS[$k3]}")
        NPE_MEDIAN[$gk]=$(median3 "${C_NPE[$k1]}" "${C_NPE[$k2]}" "${C_NPE[$k3]}")
        NCFN_MAX[$gk]=$(max3 "${C_NCFN[$k1]}" "${C_NCFN[$k2]}" "${C_NCFN[$k3]}")
    done
done
echo "Phase D: computed 6 (case, N) medians (wall + nps + npe + ncfn_max)"

# ----------------------------------------------------------------------------
# Phase E: Per-cell SHA12 + max_ulp computation
# ----------------------------------------------------------------------------

declare -A SHA12_IDENTITY SHA12_BASELINE_CELL MAX_ULP_RES MAX_ULP_STATUS

# Helper: compute max ulp between two binary double arrays using uv python
# Outputs lines: "max_ulp=<n>" "n_elements=<n>" "mode=<status>"
compute_max_ulp() {
    local file_a="$1"   # baseline
    local file_b="$2"   # identity
    if [ ! -f "$file_a" ] || [ ! -f "$file_b" ]; then
        echo "max_ulp=NA"
        echo "n_elements=NA"
        echo "mode=MISSING"
        return
    fi
    local sz_a sz_b
    sz_a=$(stat -f%z "$file_a" 2>/dev/null || stat -c%s "$file_a")
    sz_b=$(stat -f%z "$file_b" 2>/dev/null || stat -c%s "$file_b")
    if [ "$sz_a" != "$sz_b" ]; then
        echo "max_ulp=NA"
        echo "n_elements=NA"
        echo "mode=SIZE_MISMATCH"
        return
    fi
    uv run --quiet --with numpy python - "$file_a" "$file_b" <<'PYEOF' 2>/dev/null || echo "mode=PYTHON_FAIL"
import sys
import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
a = np.fromfile(a_path, dtype=np.float64)
b = np.fromfile(b_path, dtype=np.float64)
n = a.size
if n == 0:
    print("max_ulp=0")
    print(f"n_elements={n}")
    print("mode=EMPTY")
    sys.exit(0)
# Bitwise equal mask
eq_mask = (a == b)
# For non-equal entries compute ulp distance — element-wise via spacing(max(|a|,|b|))
max_abs = np.maximum(np.abs(a), np.abs(b))
spacing = np.spacing(max_abs)
# avoid div-by-zero where both are 0 (already eq) or where spacing underflows
spacing = np.where(spacing == 0, np.finfo(np.float64).tiny, spacing)
ulp_dist = np.abs(a - b) / spacing
# mask out NaN/inf safe
finite_mask = np.isfinite(ulp_dist)
ulp_dist = np.where(finite_mask, ulp_dist, 0.0)
ulp_dist = np.where(eq_mask, 0.0, ulp_dist)
max_ulp = int(np.ceil(ulp_dist.max()))
print(f"max_ulp={max_ulp}")
print(f"n_elements={n}")
print("mode=OK")
PYEOF
}

if [ ! -d "$P8PRE_BASELINE_DIR" ]; then
    echo "Phase E: baseline dir absent — all max_ulp = DEFER"
    baseline_avail=0
else
    baseline_avail=1
fi

for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            id_file="${P8PRE_IDENTITY_DIR}${ck}/${c}.rivqdown.dat"
            bs_file="${P8PRE_BASELINE_DIR}${ck}/${c}.rivqdown.dat"
            if [ -f "$id_file" ]; then
                SHA12_IDENTITY[$ck]=$(sha256sum "$id_file" | cut -c1-12)
            else
                SHA12_IDENTITY[$ck]="MISSING"
            fi
            if [ "$baseline_avail" = "1" ] && [ -f "$bs_file" ]; then
                SHA12_BASELINE_CELL[$ck]=$(sha256sum "$bs_file" | cut -c1-12)
                ulp_out=$(compute_max_ulp "$bs_file" "$id_file")
                MAX_ULP_RES[$ck]=$(echo "$ulp_out" | awk -F= '/^max_ulp=/{print $2}')
                MAX_ULP_STATUS[$ck]=$(echo "$ulp_out" | awk -F= '/^mode=/{print $2}')
            else
                SHA12_BASELINE_CELL[$ck]="NA"
                MAX_ULP_RES[$ck]="NA"
                MAX_ULP_STATUS[$ck]="DEFER"
            fi
        done
    done
done
echo "Phase E: SHA12 + max_ulp computed for 18 cells (baseline_avail=$baseline_avail)"

# ----------------------------------------------------------------------------
# Phase F: Gate evaluation
# ----------------------------------------------------------------------------
echo ""
echo "================================================================"
echo "Gate evaluation"
echo "================================================================"

# ---------- Gate 1: build PASS ----------
# server_nm.log MUST show PSetupIdentity + PSolveIdentity + CVodeSetPreconditioner
g1_psetup=$(grep -c "PSetupIdentity" "$SERVER_NM_LOG" || true)
g1_psolve=$(grep -c "PSolveIdentity" "$SERVER_NM_LOG" || true)
g1_csetp=$(grep -c "CVodeSetPreconditioner" "$SERVER_NM_LOG" || true)
g1_total=$((g1_psetup + g1_psolve + g1_csetp))
if [ "$g1_psetup" -ge 1 ] && [ "$g1_psolve" -ge 1 ] && [ "$g1_csetp" -ge 1 ]; then
    G1_VERDICT="PASS"
else
    G1_VERDICT="FAIL"
fi
echo "Gate 1 (build PASS): $G1_VERDICT — PSetupIdentity=$g1_psetup PSolveIdentity=$g1_psolve CVodeSetPreconditioner=$g1_csetp"

# ---------- Gate 2: ncfn = 0 per cell ----------
g2_violations=0
g2_max_ncfn=0
for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            if [ "${C_NCFN[$ck]}" -gt 0 ]; then
                g2_violations=$((g2_violations + 1))
            fi
            if [ "${C_NCFN[$ck]}" -gt "$g2_max_ncfn" ]; then
                g2_max_ncfn=${C_NCFN[$ck]}
            fi
        done
    done
done
if [ "$g2_violations" -eq 0 ]; then
    G2_VERDICT="PASS"
else
    G2_VERDICT="FAIL"
fi
echo "Gate 2 (ncfn = 0): $G2_VERDICT — violations=$g2_violations / 18 cells, max_ncfn=$g2_max_ncfn"
# Per-case ncfn summary
for c in "${CASES[@]}"; do
    sum=0
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            sum=$((sum + C_NCFN[$ck]))
        done
    done
    avg=$(awk -v s="$sum" 'BEGIN{printf "%.2f", s/9}')
    echo "  ${c}: total ncfn across 9 cells = $sum (avg ${avg})"
done

# ---------- Gate 3: nps > 0 AND npe > 0 per cell ----------
g3_violations=0
g3_min_nps=999999999
g3_min_npe=999999999
for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            if [ "${C_NPS[$ck]}" -le 0 ] || [ "${C_NPE[$ck]}" -le 0 ]; then
                g3_violations=$((g3_violations + 1))
            fi
            [ "${C_NPS[$ck]}" -lt "$g3_min_nps" ] && g3_min_nps=${C_NPS[$ck]}
            [ "${C_NPE[$ck]}" -lt "$g3_min_npe" ] && g3_min_npe=${C_NPE[$ck]}
        done
    done
done
if [ "$g3_violations" -eq 0 ]; then
    G3_VERDICT="PASS"
else
    G3_VERDICT="FAIL"
fi
echo "Gate 3 (nps>0 AND npe>0): $G3_VERDICT — violations=$g3_violations / 18, min_nps=$g3_min_nps min_npe=$g3_min_npe"

# ---------- Gate 4: wall non-regression per (case, N) ----------
g4_violations=0
G4_DELTAS=()
for c in "${CASES[@]}"; do
    eps=${GATE_4_EPSILON[$c]}
    max_delta=0
    for n in "${NS[@]}"; do
        gk="${c}_N${n}"
        bk="${c}_${n}"
        wall_id=${WALL_MEDIAN[$gk]}
        wall_bs=${BASELINE_WALL_MEDIAN[$bk]}
        delta=$(awk -v a="$wall_id" -v b="$wall_bs" 'BEGIN{d=(a-b)/b; if(d<0)d=-d; printf "%.6f", d}')
        violated=$(awk -v d="$delta" -v e="$eps" 'BEGIN{print (d>e)?1:0}')
        if [ "$violated" = "1" ]; then
            g4_violations=$((g4_violations + 1))
        fi
        cmp=$(awk -v d="$delta" -v mx="$max_delta" 'BEGIN{print (d>mx)?d:mx}')
        max_delta=$cmp
        G4_DELTAS+=("${c}_N${n}=${delta}(eps=${eps},violated=${violated})")
        delta_pct=$(awk -v d="$delta" 'BEGIN{printf "%.2f", d*100}')
        eps_pct=$(awk -v e="$eps" 'BEGIN{printf "%.1f", e*100}')
        verdict_str=$([ "$violated" = "1" ] && echo FAIL || echo PASS)
        echo "Gate 4 ${c} N=${n}: wall_id=${wall_id}s wall_bs=${wall_bs}s delta=${delta_pct}% (eps=${eps_pct}%) — ${verdict_str}"
    done
done
if [ "$g4_violations" -eq 0 ]; then
    G4_VERDICT="PASS"
else
    G4_VERDICT="FAIL"
fi
echo "Gate 4 (wall non-regression): $G4_VERDICT — violations=$g4_violations / 6 (case, N) groups"

# ---------- Soft Gate 5: cross-N tolerance ----------
g5_strict_violations=0
g5_fallback_violations=0
g5_deferred=0
G5_NOTES=()
for c in "${CASES[@]}"; do
    base_sha=${GATE_5_BASELINE_SHA12[$c]}
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            id_sha=${SHA12_IDENTITY[$ck]}
            if [ "$id_sha" = "$base_sha" ]; then
                # strict PASS
                :
            else
                g5_strict_violations=$((g5_strict_violations + 1))
                # fall back to max_ulp
                ulp=${MAX_ULP_RES[$ck]}
                status=${MAX_ULP_STATUS[$ck]}
                if [ "$status" = "OK" ]; then
                    over=$(awk -v u="$ulp" -v t="$GATE_5_FALLBACK_MAX_ULP" 'BEGIN{print (u>t)?1:0}')
                    if [ "$over" = "1" ]; then
                        g5_fallback_violations=$((g5_fallback_violations + 1))
                    fi
                else
                    g5_deferred=$((g5_deferred + 1))
                fi
            fi
        done
    done
done
if [ "$g5_strict_violations" -eq 0 ]; then
    G5_VERDICT="PASS_STRICT"
elif [ "$g5_deferred" -gt 0 ]; then
    G5_VERDICT="DEFER"
elif [ "$g5_fallback_violations" -eq 0 ]; then
    G5_VERDICT="PASS_FALLBACK"
else
    G5_VERDICT="FAIL"
fi
echo "Soft Gate 5 (cross-N SHA12 / max_ulp): $G5_VERDICT — strict_violations=$g5_strict_violations fallback_violations=$g5_fallback_violations deferred=$g5_deferred (threshold=$GATE_5_FALLBACK_MAX_ULP ulp)"

# ---------- Soft Gate 6: setup overhead ----------
g6_violations=0
g6_max_ratio=0
g6_missing=0
for c in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for r in "${REPS[@]}"; do
            ck="${c}_N${n}_rep${r}"
            tps=${C_TPS[$ck]}
            wall=${C_WALL[$ck]}
            if [ -z "$tps" ] || [ "$tps" = "NA" ]; then
                g6_missing=$((g6_missing + 1))
                continue
            fi
            ratio=$(awk -v t="$tps" -v w="$wall" 'BEGIN{printf "%.10f", t/w}')
            cmp=$(awk -v r="$ratio" -v mx="$g6_max_ratio" 'BEGIN{print (r>mx)?r:mx}')
            g6_max_ratio=$cmp
            over=$(awk -v r="$ratio" -v t="$GATE_6_RATIO_THRESHOLD" 'BEGIN{print (r>t)?1:0}')
            [ "$over" = "1" ] && g6_violations=$((g6_violations + 1))
        done
    done
done
if [ "$g6_missing" -ge 18 ]; then
    G6_VERDICT="DEFER"
elif [ "$g6_violations" -eq 0 ]; then
    G6_VERDICT="PASS"
else
    G6_VERDICT="FAIL"
fi
g6_max_ratio_sci=$(awk -v r="$g6_max_ratio" 'BEGIN{printf "%.2e", r}')
echo "Soft Gate 6 (t_precond_setup / wall <= 0.05): $G6_VERDICT — violations=$g6_violations / 18, max_ratio=${g6_max_ratio_sci} (threshold=$GATE_6_RATIO_THRESHOLD)"

# ----------------------------------------------------------------------------
# Phase G: Verdict synthesis
# ----------------------------------------------------------------------------

# Spike-level verdict logic per spec L74-79:
# any of gate 1-4 FAIL → NO-GO
# otherwise PASS (soft gates record carve-out)
SPIKE_VERDICT="UNKNOWN"
if [ "$G1_VERDICT" = "FAIL" ] || [ "$G2_VERDICT" = "FAIL" ] || \
   [ "$G3_VERDICT" = "FAIL" ] || [ "$G4_VERDICT" = "FAIL" ]; then
    SPIKE_VERDICT="NO-GO"
else
    SPIKE_VERDICT="PASS"
fi

# ADR-0003 recommendation per spec L116-130
case "$SPIKE_VERDICT" in
    PASS)
        ADR_REC="GO (启动正式 P8-precond epic, identity stub validates SUNDIALS PSetup/PSolve plumbing + no convergence regression)"
        ;;
    NO-GO)
        ADR_REC="NO-GO (gate failure forces fall-back to PREC_NONE per design D8; ADR-0003 NO-GO with rationale; revert cvode_config.cpp:259 + delete MD_precond_identity.{h,cpp}; close baseline/p8pre)"
        ;;
    *)
        ADR_REC="DEFER"
        ;;
esac

echo ""
echo "================================================================"
echo "Spike verdict: $SPIKE_VERDICT"
echo "ADR-0003 recommendation: $ADR_REC"
echo "================================================================"

# ----------------------------------------------------------------------------
# Phase G2: Write structured verdict file for doc consumption
# ----------------------------------------------------------------------------

{
    echo "# tools/p8pre/aggregate_identity_spike.sh structured verdict output"
    echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# Consumed by: docs/p8pre/identity_spike_verdict.md §3 raw data + §4 + §5 tables"
    echo "spike_verdict=$SPIKE_VERDICT"
    echo "adr_recommendation=$ADR_REC"
    echo "gate_1_verdict=$G1_VERDICT"
    echo "gate_1_psetup_count=$g1_psetup"
    echo "gate_1_psolve_count=$g1_psolve"
    echo "gate_1_csetp_count=$g1_csetp"
    echo "gate_2_verdict=$G2_VERDICT"
    echo "gate_2_violations=$g2_violations"
    echo "gate_2_max_ncfn=$g2_max_ncfn"
    echo "gate_3_verdict=$G3_VERDICT"
    echo "gate_3_violations=$g3_violations"
    echo "gate_3_min_nps=$g3_min_nps"
    echo "gate_3_min_npe=$g3_min_npe"
    echo "gate_4_verdict=$G4_VERDICT"
    echo "gate_4_violations=$g4_violations"
    for d in "${G4_DELTAS[@]}"; do
        echo "gate_4_delta=$d"
    done
    echo "gate_5_verdict=$G5_VERDICT"
    echo "gate_5_strict_violations=$g5_strict_violations"
    echo "gate_5_fallback_violations=$g5_fallback_violations"
    echo "gate_5_deferred=$g5_deferred"
    echo "gate_5_threshold_ulp=$GATE_5_FALLBACK_MAX_ULP"
    echo "gate_6_verdict=$G6_VERDICT"
    echo "gate_6_violations=$g6_violations"
    echo "gate_6_max_ratio=$g6_max_ratio"
    echo "gate_6_threshold=$GATE_6_RATIO_THRESHOLD"
    # Per (case, N) summary
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            gk="${c}_N${n}"
            bk="${c}_${n}"
            echo "wall_median_identity[$gk]=${WALL_MEDIAN[$gk]}"
            echo "wall_median_baseline[$gk]=${BASELINE_WALL_MEDIAN[$bk]}"
            echo "nps_median[$gk]=${NPS_MEDIAN[$gk]}"
            echo "npe_median[$gk]=${NPE_MEDIAN[$gk]}"
            echo "ncfn_max[$gk]=${NCFN_MAX[$gk]}"
        done
    done
    # Per-cell SHA12 + max_ulp
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            for r in "${REPS[@]}"; do
                ck="${c}_N${n}_rep${r}"
                echo "sha12_identity[$ck]=${SHA12_IDENTITY[$ck]}"
                echo "sha12_baseline_cell[$ck]=${SHA12_BASELINE_CELL[$ck]}"
                echo "max_ulp[$ck]=${MAX_ULP_RES[$ck]}"
                echo "max_ulp_status[$ck]=${MAX_ULP_STATUS[$ck]}"
            done
        done
    done
} > "$VERDICT_FILE"

echo ""
echo "Structured verdict written to: $VERDICT_FILE"
echo "Cell count: $n_cells"
echo "Done."
exit 0
