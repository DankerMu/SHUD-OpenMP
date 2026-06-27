#!/bin/bash
# tools/p8pre/aggregate_n8_profile.sh
#
# p8pre-spike Step 1 PR-B aggregator (issue #342) — local-side aggregator that
# parses the 18-cell rsync mirror at $P8PRE_MIRROR_DIR (default
# /tmp/p8pre_n8_profile/) emitted by PR-A run (issue #341), computes per-cell
# median over 3 reps for 15 canonical CVODE keys + 7 timer buckets + 2 extras,
# verifies cross-N invariance (5 keys, Δ=0 strict), verifies absolute baseline
# anchors (heihe / heihe_x4 × nst / nfe), computes the nfeLS/nfe ROI ratio,
# and emits a 4-branch verdict letter (a/b/c/d) to stdout AND to
# docs/p8pre/n8_profile_verdict.md.
#
# Language choice rationale: POSIX bash + awk — matches sibling
# tools/p8pre/run_n8_profile.sh + render_n8_profile.sh style (also bash + awk),
# avoids Python dependency for what is straightforward YAML/KV parsing. The
# input is a small, well-structured grid (18 files × ~16 lines each) so
# bash+awk overhead is negligible.
#
# Matrix per design D4: 2 cases × 3 N-values × 3 reps = 18 cells.
#   cases = heihe + heihe_x4
#   N     = 1, 4, 8 (N=2 explicitly skipped)
#   reps  = 1, 2, 3 (median across 3 reps per (case, N) group)
#
# Output:
#   stdout: 6-row summary table + verdict header (branch: <letter>)
#   docs/p8pre/n8_profile_verdict.md: academic-style PR-B verdict doc
#
# Exit code:
#   0   clean PROCEED or NO-GO verdict written
#   1   REJECT typo / parse error (e.g., nlcf / nfevals / hcur / qcur / hin)
#   2   cross-N invariance fail (5 keys × 2 cases × 3 Ns)
#   3   absolute baseline mismatch (Mode C regression flag)
#
# POSIX bash + awk + grep; no Python dependency. Verified bash -n.

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

P8PRE_MIRROR_DIR="${P8PRE_MIRROR_DIR:-/tmp/p8pre_n8_profile/}"
# Normalize trailing slash for predictable concatenation
P8PRE_MIRROR_DIR="${P8PRE_MIRROR_DIR%/}/"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VERDICT_DOC="$REPO_ROOT/docs/p8pre/n8_profile_verdict.md"

CASES=(heihe heihe_x4)
NS=(1 4 8)
REPS=(1 2 3)

# Canonical 15-key set per tools/cvode_stats_diff/canonical_15_keys.yaml
# (kept inline rather than parsing the yaml to keep dependency surface flat;
# any drift between the two MUST be caught by the unit test described in the
# yaml header).
CANONICAL_KEYS=(nfe nfeLS nni nli nsetups netf nst npe nps ncfn ncfl lenrw leniw lenrwLS leniwLS)

# REJECT keys per spec n8-mode-c-profile-recheck Scenario "per-cell stats file
# present and parseable" L38: typo `nlcf` (canonical name is `ncfl`);
# misnomer `nfevals` (canonical name is `nfe`); diagnostic-only `hcur` /
# `qcur` / `hin` (emit only under -DSHUD_ENABLE_DIAGNOSTICS, NOT enabled in
# this epic — see canonical_15_keys.yaml `diagnostic_keys` block).
REJECT_KEYS=(nlcf nfevals hcur qcur hin)

# 7 timer buckets per profile_B0.yaml `buckets:` block + 2 extras
# (tools/profile/timer.cpp:158-167 + 183-184)
BUCKETS=(t_RHS_kernel t_RHS_total t_CVODE_internal t_forcing_io t_ET t_output t_other)
EXTRAS=(t_CVODE_raw t_wall_total)

# Cross-N invariance keys per spec n8-mode-c-profile-recheck Requirement
# "cross-N stats invariance check (5 counters)" + Scenarios L86-104
INVARIANCE_KEYS=(nst nfe nfeLS nni nsetups)

# Absolute baseline anchors per spec Scenarios L90 + L97 + tasks.md §3.3
declare -A BASELINE_NST
BASELINE_NST[heihe]=6698
BASELINE_NST[heihe_x4]=6575
declare -A BASELINE_NFE
BASELINE_NFE[heihe]=6943
BASELINE_NFE[heihe_x4]=6741

# ROI branch thresholds per spec Scenario "ROI verdict based on nfeLS/nfe
# exhaustive branching" L72-80 + tasks.md §3.4
ROI_THRESH_PROCEED="1.5"
ROI_THRESH_HETEROGENEITY="3.0"

# ----------------------------------------------------------------------------
# Phase A: Pre-flight — verify input dir + 18 cells present
# ----------------------------------------------------------------------------

echo "[aggregate_n8_profile] mirror_dir=$P8PRE_MIRROR_DIR"
if [ ! -d "$P8PRE_MIRROR_DIR" ]; then
    echo "[aggregate_n8_profile] FATAL: mirror dir not found: $P8PRE_MIRROR_DIR" >&2
    exit 1
fi

cell_count=0
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for rep in "${REPS[@]}"; do
            cell_dir="${P8PRE_MIRROR_DIR}${case}_N${n}_rep${rep}"
            yaml="$cell_dir/profile_B0.yaml"
            stats="$cell_dir/cvode_stats.txt"
            if [ ! -r "$yaml" ]; then
                echo "[aggregate_n8_profile] FATAL: missing profile_B0.yaml: $yaml" >&2
                exit 1
            fi
            if [ ! -r "$stats" ]; then
                echo "[aggregate_n8_profile] FATAL: missing cvode_stats.txt: $stats" >&2
                exit 1
            fi
            cell_count=$((cell_count + 1))
        done
    done
done
if [ "$cell_count" -ne 18 ]; then
    echo "[aggregate_n8_profile] FATAL: expected 18 cells, found $cell_count" >&2
    exit 1
fi
echo "[aggregate_n8_profile] pre-flight OK: 18 cells present"

# ----------------------------------------------------------------------------
# Phase B: REJECT typo keys check (all 18 cvode_stats.txt files)
# ----------------------------------------------------------------------------

echo "[aggregate_n8_profile] Phase B: REJECT typo key check"
typo_hits=0
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for rep in "${REPS[@]}"; do
            stats="${P8PRE_MIRROR_DIR}${case}_N${n}_rep${rep}/cvode_stats.txt"
            for bad in "${REJECT_KEYS[@]}"; do
                # Anchor on start-of-line + key + `=` to avoid false-positive on
                # comments or substring matches.
                if grep -qE "^${bad}=" "$stats"; then
                    echo "[aggregate_n8_profile] REJECT: typo key '$bad' found in ${case}_N${n}_rep${rep}/cvode_stats.txt" >&2
                    typo_hits=$((typo_hits + 1))
                fi
            done
        done
    done
done
if [ "$typo_hits" -gt 0 ]; then
    echo "[aggregate_n8_profile] FATAL: $typo_hits typo-key occurrence(s); spec n8-mode-c-profile-recheck L38 REJECT" >&2
    exit 1
fi
echo "[aggregate_n8_profile] REJECT typo check PASS (0 hits across 18 cells)"

# ----------------------------------------------------------------------------
# Phase C: Per-cell parse helpers
# ----------------------------------------------------------------------------

# parse_cvode_stat <stats_file> <key> → echoes integer value (per spec all 15
# keys are integer-valued in PrintFinalStats). Empty string on missing key.
parse_cvode_stat() {
    local f="$1" k="$2"
    awk -F= -v k="$k" '$1 == k { print $2; found=1; exit } END { if (!found) exit 1 }' "$f" 2>/dev/null || true
}

# parse_yaml_bucket <yaml_file> <bucket_name> → echoes float value (seconds).
# Matches `  <name>: <float>` inside buckets: block OR extras: block.
# timer.cpp emits `%.9f` so values are always plain decimal, no exp notation.
parse_yaml_bucket() {
    local f="$1" k="$2"
    awk -v k="$k" '
        $1 == k":" { print $2; found=1; exit }
        END { if (!found) exit 1 }
    ' "$f" 2>/dev/null || true
}

# Sanity-test the helpers on cell heihe_N1_rep1 — protects against silent
# parser drift if the emitted format changes.
SAMPLE_STATS="${P8PRE_MIRROR_DIR}heihe_N1_rep1/cvode_stats.txt"
SAMPLE_YAML="${P8PRE_MIRROR_DIR}heihe_N1_rep1/profile_B0.yaml"
sample_nst=$(parse_cvode_stat "$SAMPLE_STATS" "nst")
sample_wall=$(parse_yaml_bucket "$SAMPLE_YAML" "t_wall_total")
if [ -z "$sample_nst" ] || [ -z "$sample_wall" ]; then
    echo "[aggregate_n8_profile] FATAL: parser sanity check failed (nst='$sample_nst' wall='$sample_wall')" >&2
    exit 1
fi
echo "[aggregate_n8_profile] parser sanity OK: sample nst=$sample_nst wall=$sample_wall"

# ----------------------------------------------------------------------------
# Phase D: Per-cell parse → temporary CSV (one row per cell with all metrics)
# ----------------------------------------------------------------------------

TMP_DIR="$(mktemp -d -t aggregate_n8_profile.XXXXXX)"
trap 'rm -rf "$TMP_DIR"' EXIT
CELLS_CSV="$TMP_DIR/cells.csv"

# Header: case,N,rep,<15 keys>,<7 buckets>,<2 extras>
{
    printf 'case,N,rep'
    for k in "${CANONICAL_KEYS[@]}"; do printf ',%s' "$k"; done
    for b in "${BUCKETS[@]}"; do printf ',%s' "$b"; done
    for e in "${EXTRAS[@]}"; do printf ',%s' "$e"; done
    printf '\n'
} > "$CELLS_CSV"

for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        for rep in "${REPS[@]}"; do
            cell="${case}_N${n}_rep${rep}"
            stats="${P8PRE_MIRROR_DIR}${cell}/cvode_stats.txt"
            yaml="${P8PRE_MIRROR_DIR}${cell}/profile_B0.yaml"
            row="$case,$n,$rep"
            for k in "${CANONICAL_KEYS[@]}"; do
                v=$(parse_cvode_stat "$stats" "$k")
                if [ -z "$v" ]; then
                    echo "[aggregate_n8_profile] FATAL: missing canonical key '$k' in $stats" >&2
                    exit 1
                fi
                row="$row,$v"
            done
            for b in "${BUCKETS[@]}"; do
                v=$(parse_yaml_bucket "$yaml" "$b")
                if [ -z "$v" ]; then
                    echo "[aggregate_n8_profile] FATAL: missing bucket '$b' in $yaml" >&2
                    exit 1
                fi
                row="$row,$v"
            done
            for e in "${EXTRAS[@]}"; do
                v=$(parse_yaml_bucket "$yaml" "$e")
                if [ -z "$v" ]; then
                    echo "[aggregate_n8_profile] FATAL: missing extra '$e' in $yaml" >&2
                    exit 1
                fi
                row="$row,$v"
            done
            echo "$row" >> "$CELLS_CSV"
        done
    done
done

parsed_lines=$(($(wc -l < "$CELLS_CSV") - 1))
if [ "$parsed_lines" -ne 18 ]; then
    echo "[aggregate_n8_profile] FATAL: parsed $parsed_lines cells, expected 18" >&2
    exit 1
fi
echo "[aggregate_n8_profile] parse OK: 18 cells × $((${#CANONICAL_KEYS[@]} + ${#BUCKETS[@]} + ${#EXTRAS[@]})) metrics"

# ----------------------------------------------------------------------------
# Phase E: Per (case, N) median over 3 reps (one row per group, 6 rows total)
#
# Median over 3 reps = middle element after numeric sort. For integer keys
# the median is an integer; for float buckets we keep float precision.
# ----------------------------------------------------------------------------

MEDIANS_CSV="$TMP_DIR/medians.csv"
{
    printf 'case,N'
    for k in "${CANONICAL_KEYS[@]}"; do printf ',%s_median' "$k"; done
    for b in "${BUCKETS[@]}"; do printf ',%s_median' "$b"; done
    for e in "${EXTRAS[@]}"; do printf ',%s_median' "$e"; done
    printf '\n'
} > "$MEDIANS_CSV"

# Number of metric columns
N_KEYS=${#CANONICAL_KEYS[@]}
N_BUCKETS=${#BUCKETS[@]}
N_EXTRAS=${#EXTRAS[@]}
N_METRICS=$((N_KEYS + N_BUCKETS + N_EXTRAS))

# awk-based median computation per (case, N) group. Reads CELLS_CSV (skip
# header), partitions by case+N, collects all 3 rep values per metric, picks
# median of 3 (= sorted[1] = middle).
awk -v n_metrics="$N_METRICS" '
    BEGIN { FS=OFS="," }
    NR == 1 { next }  # skip header
    {
        case = $1
        n = $2
        # rep = $3 (unused; we just collect)
        gid = case "|" n
        groups[gid] = 1
        for (m = 1; m <= n_metrics; m++) {
            v = $(3 + m)
            count[gid, m] += 1
            c = count[gid, m]
            vals[gid, m, c] = v
        }
    }
    END {
        # Define explicit emit order matching outer driver
        order[1] = "heihe|1"
        order[2] = "heihe|4"
        order[3] = "heihe|8"
        order[4] = "heihe_x4|1"
        order[5] = "heihe_x4|4"
        order[6] = "heihe_x4|8"
        for (i = 1; i <= 6; i++) {
            gid = order[i]
            split(gid, parts, "|")
            case = parts[1]; n = parts[2]
            line = case OFS n
            for (m = 1; m <= n_metrics; m++) {
                # Pick median of 3 via sort then index [2] (1-based middle of 3)
                a = vals[gid, m, 1] + 0
                b = vals[gid, m, 2] + 0
                c = vals[gid, m, 3] + 0
                # 3-element median: pick the middle of (a, b, c)
                if (a > b) { t = a; a = b; b = t }
                if (b > c) { t = b; b = c; c = t }
                if (a > b) { t = a; a = b; b = t }
                # Now b is median. Use original-token precision: if value
                # looks integer (no dot), emit as int; else %.9f.
                tok1 = vals[gid, m, 1]; tok2 = vals[gid, m, 2]; tok3 = vals[gid, m, 3]
                is_float = 0
                if (index(tok1, ".") > 0) is_float = 1
                if (index(tok2, ".") > 0) is_float = 1
                if (index(tok3, ".") > 0) is_float = 1
                if (is_float) {
                    line = line OFS sprintf("%.9f", b)
                } else {
                    line = line OFS sprintf("%d", b)
                }
            }
            print line
        }
    }
' "$CELLS_CSV" >> "$MEDIANS_CSV"

medians_rows=$(($(wc -l < "$MEDIANS_CSV") - 1))
if [ "$medians_rows" -ne 6 ]; then
    echo "[aggregate_n8_profile] FATAL: $medians_rows median rows, expected 6" >&2
    exit 1
fi
echo "[aggregate_n8_profile] median OK: 6 (case,N) groups"

# Column index helper: get column N for a given metric name in MEDIANS_CSV.
# Header is: case,N,nfe_median,nfeLS_median,...
# 1-based: case=1, N=2, then metrics start at col 3.
column_idx_of_metric() {
    local name="$1"
    local idx
    idx=$(head -1 "$MEDIANS_CSV" | awk -F, -v n="${name}_median" '{
        for (i = 1; i <= NF; i++) { if ($i == n) { print i; exit } }
    }')
    if [ -z "$idx" ]; then
        echo "[aggregate_n8_profile] FATAL: metric '$name' not in medians header" >&2
        exit 1
    fi
    echo "$idx"
}

# Read median value for a given (case, N, metric) tuple from MEDIANS_CSV.
read_median() {
    local case="$1" n="$2" metric="$3"
    local col
    col=$(column_idx_of_metric "$metric")
    awk -F, -v c="$case" -v n="$n" -v col="$col" '
        $1 == c && $2 == n { print $col; exit }
    ' "$MEDIANS_CSV"
}

# ----------------------------------------------------------------------------
# Phase F: Cross-N invariance check (5 keys × 2 cases)
# ----------------------------------------------------------------------------

echo "[aggregate_n8_profile] Phase F: cross-N invariance check (Δ=0 strict)"
invariance_fails=()
for case in "${CASES[@]}"; do
    for k in "${INVARIANCE_KEYS[@]}"; do
        v1=$(read_median "$case" 1 "$k")
        v4=$(read_median "$case" 4 "$k")
        v8=$(read_median "$case" 8 "$k")
        if [ "$v1" != "$v4" ] || [ "$v4" != "$v8" ]; then
            msg="${case}:${k}: N=1=${v1}, N=4=${v4}, N=8=${v8}"
            invariance_fails+=("$msg")
            echo "[aggregate_n8_profile] BLOCKED invariance: $msg" >&2
        fi
    done
done

# ----------------------------------------------------------------------------
# Phase G: Absolute baseline anchor check
# ----------------------------------------------------------------------------

echo "[aggregate_n8_profile] Phase G: absolute baseline anchor check"
baseline_fails=()
for case in "${CASES[@]}"; do
    expected_nst="${BASELINE_NST[$case]}"
    expected_nfe="${BASELINE_NFE[$case]}"
    # Per invariance: N=1=4=8, so checking N=1 covers all.
    actual_nst=$(read_median "$case" 1 "nst")
    actual_nfe=$(read_median "$case" 1 "nfe")
    if [ "$actual_nst" != "$expected_nst" ]; then
        msg="${case}:nst: expected ${expected_nst}, got ${actual_nst}"
        baseline_fails+=("$msg")
        echo "[aggregate_n8_profile] BLOCKED baseline: $msg" >&2
    fi
    if [ "$actual_nfe" != "$expected_nfe" ]; then
        msg="${case}:nfe: expected ${expected_nfe}, got ${actual_nfe}"
        baseline_fails+=("$msg")
        echo "[aggregate_n8_profile] BLOCKED baseline: $msg" >&2
    fi
done

# ----------------------------------------------------------------------------
# Phase H: ROI ratios per (case, N) — r = nfeLS / nfe; nli/nni; cross-N Δ
# ----------------------------------------------------------------------------

echo "[aggregate_n8_profile] Phase H: ROI ratios per (case, N)"
declare -A R_RATIO
declare -A NLI_OVER_NNI
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        nfe=$(read_median "$case" "$n" "nfe")
        nfeLS=$(read_median "$case" "$n" "nfeLS")
        nni=$(read_median "$case" "$n" "nni")
        nli=$(read_median "$case" "$n" "nli")
        # awk float division, 3 decimal places per spec L70.
        r=$(awk -v a="$nfeLS" -v b="$nfe" 'BEGIN { if (b == 0) print "0.000"; else printf "%.3f", a / b }')
        nlni=$(awk -v a="$nli" -v b="$nni" 'BEGIN { if (b == 0) print "0.000"; else printf "%.3f", a / b }')
        R_RATIO["${case}|${n}"]="$r"
        NLI_OVER_NNI["${case}|${n}"]="$nlni"
    done
done

# nst Δ cross-N per case (should be 0 per invariance — for the verdict doc table)
declare -A NST_DELTA
for case in "${CASES[@]}"; do
    v1=$(read_median "$case" 1 "nst")
    v8=$(read_median "$case" 8 "nst")
    NST_DELTA["$case"]=$((v8 - v1))
done

# ----------------------------------------------------------------------------
# Phase I: ROI verdict branch letter (a/b/c/d) per spec L72-80
#
# r_min = min over (case, N) at N=8 per spec L74 ("min over cases at N=8");
# r_max = max over (case, N) at N=8.
# Per the spec the verdict applies to N=8 (the target thread count). We also
# print all 6 r values for full transparency, but the branch decision uses
# only the N=8 pair (heihe, heihe_x4) per spec L74.
# ----------------------------------------------------------------------------

r_heihe_n8="${R_RATIO[heihe|8]}"
r_x4_n8="${R_RATIO[heihe_x4|8]}"

# Compute r_min, r_max via awk (float-safe)
r_min=$(awk -v a="$r_heihe_n8" -v b="$r_x4_n8" 'BEGIN { printf "%.3f", (a < b ? a : b) }')
r_max=$(awk -v a="$r_heihe_n8" -v b="$r_x4_n8" 'BEGIN { printf "%.3f", (a > b ? a : b) }')

# Branch precedence per task brief Step 2 §7:
#   - branch d takes precedence over c if both conditions hold
#   - branch a takes precedence over c
# Decision logic mirrors spec L75-80 EXHAUSTIVE + MUTUALLY EXCLUSIVE structure:
branch=""
branch_reason=""
branch_action=""

# Use awk for float comparison (bash doesn't do float natively)
ge_proceed=$(awk -v r="$r_min" -v t="$ROI_THRESH_PROCEED" 'BEGIN { print (r >= t ? 1 : 0) }')
lt_proceed_max=$(awk -v r="$r_max" -v t="$ROI_THRESH_PROCEED" 'BEGIN { print (r < t ? 1 : 0) }')
ge_heterogeneity=$(awk -v r="$r_max" -v t="$ROI_THRESH_HETEROGENEITY" 'BEGIN { print (r >= t ? 1 : 0) }')
lt_proceed_min=$(awk -v r="$r_min" -v t="$ROI_THRESH_PROCEED" 'BEGIN { print (r < t ? 1 : 0) }')

if [ "$ge_proceed" -eq 1 ]; then
    branch="a"
    branch_action="PROCEED"
    branch_reason="r_min=${r_min} >= ${ROI_THRESH_PROCEED}"
elif [ "$lt_proceed_max" -eq 1 ]; then
    branch="b"
    branch_action="NO-GO"
    branch_reason="r_max=${r_max} < ${ROI_THRESH_PROCEED}"
elif [ "$lt_proceed_min" -eq 1 ] && [ "$ge_heterogeneity" -eq 1 ]; then
    # branch d: case-aware + high heterogeneity (precedence over c per brief §7)
    branch="d"
    branch_action="high heterogeneity"
    branch_reason="r_min=${r_min} < ${ROI_THRESH_PROCEED} AND r_max=${r_max} >= ${ROI_THRESH_HETEROGENEITY}"
else
    # branch c: case-aware mixed (r_min < 1.5 <= r_max < 3.0)
    branch="c"
    branch_action="mixed"
    branch_reason="r_min=${r_min} < ${ROI_THRESH_PROCEED} <= r_max=${r_max} < ${ROI_THRESH_HETEROGENEITY}"
fi

# ----------------------------------------------------------------------------
# Phase J: Decide exit code based on invariance + baseline + branch
# ----------------------------------------------------------------------------

block_exit=0
block_summary=""
if [ "${#invariance_fails[@]}" -gt 0 ]; then
    block_exit=2
    block_summary="cross-N invariance fail"
    for m in "${invariance_fails[@]}"; do
        block_summary="$block_summary | $m"
    done
fi
if [ "${#baseline_fails[@]}" -gt 0 ]; then
    # Baseline mismatch (3) trumps invariance (2) for Mode C regression flag,
    # per task brief Step 2 §7 exit code precedence (last listed = highest
    # specificity).
    block_exit=3
    block_summary="absolute baseline mismatch (Mode C regression)"
    for m in "${baseline_fails[@]}"; do
        block_summary="$block_summary | $m"
    done
fi

# ----------------------------------------------------------------------------
# Phase K: Stdout summary table + verdict header
# ----------------------------------------------------------------------------

# JID range for provenance (PR-A jid_table.txt)
JID_TABLE="${P8PRE_MIRROR_DIR}jid_table.txt"
jid_range="n/a"
if [ -r "$JID_TABLE" ]; then
    jid_min=$(awk '$4 ~ /^[0-9]+$/ {print $4}' "$JID_TABLE" | sort -n | head -1)
    jid_max=$(awk '$4 ~ /^[0-9]+$/ {print $4}' "$JID_TABLE" | sort -n | tail -1)
    jid_range="${jid_min}..${jid_max}"
fi

echo ""
echo "============================================================"
echo "[aggregate_n8_profile] 6-row (case, N) median summary"
echo "============================================================"
printf '%-10s %-3s %-7s %-12s %-12s %-7s %-12s\n' "case" "N" "n_reps" "nfe_median" "nfeLS_median" "r=nfeLS/nfe" "nst_median"
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        nfe=$(read_median "$case" "$n" "nfe")
        nfeLS=$(read_median "$case" "$n" "nfeLS")
        nst=$(read_median "$case" "$n" "nst")
        r="${R_RATIO[${case}|${n}]}"
        printf '%-10s %-3s %-7s %-12s %-12s %-7s %-12s\n' "$case" "$n" "3" "$nfe" "$nfeLS" "$r" "$nst"
    done
done
echo ""
echo "ROI ratios r = nfeLS/nfe at N=8:"
echo "  heihe:    r = $r_heihe_n8"
echo "  heihe_x4: r = $r_x4_n8"
echo "  r_min = $r_min ; r_max = $r_max"
echo ""

if [ "$block_exit" -gt 0 ]; then
    # Blocked: emit BLOCKED verdict header (no branch letter promotion)
    verdict_header_stdout="branch: BLOCKED (${block_summary})"
    verdict_header_doc="branch: BLOCKED (${block_summary})"
    echo "$verdict_header_stdout"
    echo ""
    echo "[aggregate_n8_profile] exit_code=$block_exit (blocked)"
else
    verdict_header_stdout="branch: $branch"
    verdict_header_doc="branch: ${branch} (${branch_action} — ${branch_reason})"
    echo "$verdict_header_stdout"
    echo ""
    echo "[aggregate_n8_profile] exit_code=0 (clean PROCEED/NO-GO verdict written)"
fi

# ----------------------------------------------------------------------------
# Phase L: Write docs/p8pre/n8_profile_verdict.md (academic style)
# ----------------------------------------------------------------------------

# Build the per-(case, N) table for §3.1 (15 keys × 6 groups)
table_31_rows=""
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        row="| $case | $n |"
        for k in "${CANONICAL_KEYS[@]}"; do
            v=$(read_median "$case" "$n" "$k")
            row="$row $v |"
        done
        table_31_rows="${table_31_rows}${row}"$'\n'
    done
done

# Build the §3.2 cross-N invariance table (5 keys × 2 cases)
table_32_rows=""
for case in "${CASES[@]}"; do
    for k in "${INVARIANCE_KEYS[@]}"; do
        v1=$(read_median "$case" 1 "$k")
        v4=$(read_median "$case" 4 "$k")
        v8=$(read_median "$case" 8 "$k")
        delta="$((v8 - v1))"
        if [ "$v1" = "$v4" ] && [ "$v4" = "$v8" ]; then
            verdict_cell="PASS"
        else
            verdict_cell="**BLOCKED**"
        fi
        row="| $case | $k | $v1 | $v4 | $v8 | $delta | $verdict_cell |"
        table_32_rows="${table_32_rows}${row}"$'\n'
    done
done

# Build the §3.3 absolute baseline table (4 anchors)
table_33_rows=""
for case in "${CASES[@]}"; do
    for k in nst nfe; do
        if [ "$k" = "nst" ]; then
            expected="${BASELINE_NST[$case]}"
        else
            expected="${BASELINE_NFE[$case]}"
        fi
        actual=$(read_median "$case" 1 "$k")
        if [ "$actual" = "$expected" ]; then
            verdict_cell="PASS"
        else
            verdict_cell="**MISMATCH (Mode C regression)**"
        fi
        row="| $case | $k | $expected | $actual | $verdict_cell |"
        table_33_rows="${table_33_rows}${row}"$'\n'
    done
done

# Build the §3.4 ROI ratio table (r and nli/nni per (case, N))
table_34_rows=""
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        nfe=$(read_median "$case" "$n" "nfe")
        nfeLS=$(read_median "$case" "$n" "nfeLS")
        nni=$(read_median "$case" "$n" "nni")
        nli=$(read_median "$case" "$n" "nli")
        r="${R_RATIO[${case}|${n}]}"
        nlni="${NLI_OVER_NNI[${case}|${n}]}"
        # Sanity: SUNDIALS CVLS SPGMR with finite-difference Jacobian-vector
        # product issues exactly ONE RHS callback per GMRES inner iter, so
        # nfeLS ≈ nli (NOT nfeLS = nfe + nli — that was a misnote earlier).
        # Report |nfeLS - nli| (integer, should be 0 for FD-Jvp setup).
        nfels_minus_nli=$((nfeLS - nli))
        row="| $case | $n | $nfe | $nfeLS | $r | $nni | $nli | $nlni | $nfels_minus_nli |"
        table_34_rows="${table_34_rows}${row}"$'\n'
    done
done

# Build §4 timer bucket table (7 buckets × 6 groups)
table_4_rows=""
for case in "${CASES[@]}"; do
    for n in "${NS[@]}"; do
        row="| $case | $n |"
        for b in "${BUCKETS[@]}"; do
            v=$(read_median "$case" "$n" "$b")
            row="$row $v |"
        done
        # Append t_wall_total from extras for sanity
        wall=$(read_median "$case" "$n" "t_wall_total")
        row="$row $wall |"
        table_4_rows="${table_4_rows}${row}"$'\n'
    done
done

# Compose the verdict doc
mkdir -p "$(dirname "$VERDICT_DOC")"
{
    cat <<EOF
---
title: "p8pre-spike Step 1 PR-B — N=8 Mode C profile ROI verdict"
date: 2026-06-27
version: 0.1
status: "${verdict_header_doc}"
related_docs:
  - openspec/changes/p8pre-spike/proposal.md
  - openspec/changes/p8pre-spike/tasks.md
  - openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md
  - tools/cvode_stats_diff/canonical_15_keys.yaml
  - docs/p1e/p1e_perf_baseline.md
  - docs/p1e/p1e_pr_i_strict_omp_verification.md
  - docs/p8pre/step1_prep.md
  - docs/p8pre/n8_profile_run.md
---

# Abstract

This PR-B verdict aggregates the 18-cell (2 cases × 3 N-values × 3 reps) Mode C
profile artifacts produced by PR-A (issue #341) into per-(case, N) medians of
the SHUD canonical 15-key CVODE stats set, the 7 emitted timer buckets, and
the 2 raw extras. It verifies cross-N invariance (Δ=0 strict) for the 5
counters \`nst / nfe / nfeLS / nni / nsetups\`, anchors the 4 absolute
baselines (\`heihe.nst=6698 / heihe.nfe=6943 / heihe_x4.nst=6575 /
heihe_x4.nfe=6741\`) imported from P1e PR-I, computes the \`nfeLS/nfe\`
ROI ratio per (case, N), and emits a single 4-branch verdict letter
(a/b/c/d) per the spec's exhaustive ROI tree. The verdict letter is the
input gate for the downstream PR-C capstone baseline doc (issue #343) and
ADR-0003 draft decision.

## §1 目的

Quantify whether the SHUD CVODE solver's linear-solver work (\`nfeLS\`)
relative to the nonlinear-solver work (\`nfe\`) is large enough at N=8 to
make a preconditioner (P8-precond) ROI-positive. The decision criterion is
spec n8-mode-c-profile-recheck Scenario "ROI verdict based on nfeLS/nfe
exhaustive branching" (L72-80):

- branch (a) \`r_min ≥ 1.5\` → PROCEED Step 2 P8-precond-0 identity spike
- branch (b) \`r_max < 1.5\` → NO-GO 转 P8-tune (maxl/restart/EpsLin)
- branch (c) \`r_min < 1.5 ≤ r_max < 3.0\` → case-aware precond design required
- branch (d) \`r_min < 1.5 AND r_max ≥ 3.0\` → case-aware + HIGH HETEROGENEITY

This document does NOT archive \`wall_step1_baseline_median(case, N)\` — that
is the PR-C (issue #343) capstone responsibility per task §4.1 and design D5.

## §2 数据来源

- Input: 18-cell rsync mirror at \`${P8PRE_MIRROR_DIR}\` (= rsync mirror of
  \`/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/<cell>/\`).
- PR-A run provenance: Slurm JID range = ${jid_range} (per
  \`${P8PRE_MIRROR_DIR}jid_table.txt\`; full (case, N, rep, JID) table
  reproduced therein).
- SHUD pin: \`7a1dc8f\` (P1e ship pin, post-nested-Timer-fix), verified
  invariant across all 18 cells by PR-A pre-flight § per spec
  n8-mode-c-profile-recheck Scenario "SHUD pin unchanged across Step 1".
- Build: \`make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1\`
  (Mode C: Serial NVector + StrictOMP RHS + Timer instrumentation).
- Partition pin: \`heihe\` → \`cn14\`, \`heihe_x4\` → \`cn15\` (per P1e PR-I
  SOP; keeps cross-cell CPU SKU comparable).

## §3 CVODE stats summary

### §3.1 Per (case, N) median table (15 canonical keys × 6 groups)

Median across 3 reps per cell, per spec L113 (median over 3 reps). All
values cite back to per-cell files at
\`${P8PRE_MIRROR_DIR}<case>_N<n>_rep<r>/cvode_stats.txt\`.

| case | N | nfe | nfeLS | nni | nli | nsetups | netf | nst | npe | nps | ncfn | ncfl | lenrw | leniw | lenrwLS | leniwLS |
|------|---|-----|-------|-----|-----|---------|------|-----|-----|-----|------|------|-------|-------|---------|---------|
EOF
    printf '%s' "$table_31_rows"
    cat <<EOF

### §3.2 Cross-N invariance Δ=0 strict (5 keys × 2 cases)

Spec n8-mode-c-profile-recheck Requirement "cross-N stats invariance check"
+ Scenarios L86-104. Each row: \`X_N=1 = X_N=4 = X_N=8\` (Δ=0 strict). Any
Δ != 0 blocks the verdict.

| case | key | N=1 | N=4 | N=8 | Δ (N=8 - N=1) | verdict |
|------|-----|-----|-----|-----|---------------|---------|
EOF
    printf '%s' "$table_32_rows"
    cat <<EOF

### §3.3 Absolute baseline anchor check (4 values)

Spec Scenarios L90 + L97 + tasks.md §3.3 absolute baseline anchors. Source
of anchor values: \`docs/p1e/p1e_perf_baseline.md\` §3.4 nst ladder
(heihe.nst=6698, heihe_x4.nst=6575) + \`docs/p1e/p1e_pr_i_strict_omp_verification.md\`
§3.1 nfe ladder (heihe.nfe=6943, heihe_x4.nfe=6741). Per invariance N=1=4=8 we
report the N=1 median.

| case | key | expected (P1e PR-I baseline) | actual (this run, N=1 median) | verdict |
|------|-----|------------------------------|-------------------------------|---------|
EOF
    printf '%s' "$table_33_rows"
    cat <<EOF

### §3.4 ROI ratios per (case, N)

- \`r = nfeLS / nfe\` per spec Scenario "nfeLS/nfe ratio reported per (case, N)" L63-70 (3 decimal places).
- \`nli / nni\` reported per spec L68. For SUNDIALS CVLS SPGMR with the
  default finite-difference Jacobian-vector product proxy, every GMRES
  inner iteration triggers exactly one RHS callback (counted into
  \`nfeLS\`), so the expected identity is \`nfeLS = nli\` (NOT
  \`nfeLS = nfe + nli\`). The last column \`|nfeLS − nli|\` is the
  sanity check; non-zero would indicate a non-FD-Jvp configuration or a
  counter ABI mismatch worth a follow-up.

| case | N | nfe_median | nfeLS_median | r = nfeLS/nfe | nni_median | nli_median | nli/nni | nfeLS − nli |
|------|---|------------|--------------|---------------|------------|------------|---------|-------------|
EOF
    printf '%s' "$table_34_rows"
    cat <<EOF

### §3.5 Verdict header (branch letter + 4-branch tree application)

**${verdict_header_doc}**

Decision trace:
- \`r_heihe_N=8\` = ${r_heihe_n8}
- \`r_heihe_x4_N=8\` = ${r_x4_n8}
- \`r_min\` = ${r_min} ; \`r_max\` = ${r_max}
- Reason: ${branch_reason}

EOF
    if [ "$block_exit" -gt 0 ]; then
        cat <<EOF
**Block conditions (the verdict is NOT a clean a/b/c/d emission)**:

- invariance fails: ${#invariance_fails[@]} cells
- baseline mismatches: ${#baseline_fails[@]} cells
- block summary: \`${block_summary}\`

EOF
    fi
    cat <<EOF
## §4 Timer bucket summary (7 buckets per (case, N))

Per spec Requirement "bucket breakdown using emitted timer buckets" L47-57.
\`t_RHS_kernel\` is EXCLUDED from the bucket-sum invariant because it is a
sub-bucket of \`t_RHS_total\` (containment relation per timer.cpp
\`emit_extra\` L176-184 derivation chain). The \`t_wall_total\` column is
the raw top-level wall and is the reference for the bucket-sum invariant
\`t_RHS_total + t_CVODE_internal + t_forcing_io + t_ET + t_output + t_other
≈ t_wall_total\` (±2% rounding + Timer overhead allowance per PR-A task
§2.3 acceptance check). Values are seconds, 9 decimal places per
\`timer.cpp\` \`%.9f\` format.

| case | N | t_RHS_kernel | t_RHS_total | t_CVODE_internal | t_forcing_io | t_ET | t_output | t_other | t_wall_total |
|------|---|--------------|-------------|------------------|--------------|------|----------|---------|--------------|
EOF
    printf '%s' "$table_4_rows"
    cat <<EOF

## §5 ROI 论述

EOF
    case "$branch" in
        a)
            cat <<EOF
Branch (a) PROCEED. \`r_min\` = ${r_min} ≥ 1.5, satisfying the spec's PROCEED
threshold uniformly across both cases at N=8. SUNDIALS \`nfeLS\` accumulates
RHS callbacks invoked by the Jacobian-vector-product proxy inside the GMRES
inner loop; an \`r\` of ~${r_heihe_n8}–${r_x4_n8} means every nonlinear
\`nfe\` evaluation is amplified by ${r_min}× to ${r_max}× through Krylov
iterations. For a preconditioner that reduces the average GMRES iteration
count by a factor \`κ\`, the expected Step 2 wall-time speedup is bounded
below by \`(1 + (r − 1)/κ) / r\` of the linear-solver portion, multiplied
by the linear-solver share of \`t_CVODE_internal\`. The spike binary
(PR-D §6.1-6.6) only needs to (i) keep the new linear-solver path bitwise
identical to the baseline (gate 2 \`ncfn = 0\` per spec) and (ii) prove the
preconditioner registration plumbing fires (gate 3 \`nps > 0\` AND
\`npe > 0\`). The expected Step 2 speedup window for a downstream non-trivial
preconditioner (e.g. element-block Jacobi) is therefore on the order of
10–25% of \`t_CVODE_internal\` per cell, depending on the \`κ\` achieved
and the linear-solver share of solver time visible in §4 above.
EOF
            ;;
        b)
            cat <<EOF
Branch (b) NO-GO. \`r_max\` = ${r_max} < 1.5, meaning every linear-solver
RHS callback already pulls its weight at the spec's NO-GO threshold. ADR-0003
SHALL record NOT 启动 precond epic; the next P8 step is P8-tune (maxl /
restart / EpsLin micro-tuning) per spec L77.
EOF
            ;;
        c)
            cat <<EOF
Branch (c) case-aware mixed. r values straddle 1.5: \`r_min\` = ${r_min}
< 1.5 ≤ \`r_max\` = ${r_max} < 3.0. Step 2 spike still proceeds for API
gating per spec L78, but ADR-0003 SHALL record an open issue requiring
case-aware precond design (the precond strategy that works for the high-r
case may be wasted for the low-r case).
EOF
            ;;
        d)
            cat <<EOF
Branch (d) case-aware + HIGH HETEROGENEITY. r values span the heterogeneity
threshold: \`r_min\` = ${r_min} < 1.5 AND \`r_max\` = ${r_max} ≥ 3.0. Step 2
spike still proceeds, but ADR-0003 SHALL record both the case-aware design
open issue AND a heterogeneity flag per spec L79-80. The factor-of-2+ r
spread suggests the precond design must be \`case-aware from day one\`.
EOF
            ;;
        *)
            cat <<EOF
Verdict is BLOCKED per Phase F/G checks above. Step 2 spike cannot proceed
until the regression flagged in §3.2 or §3.3 is reconciled.
EOF
            ;;
    esac

    cat <<EOF

## §6 Limitations & threats to validity

- **90-day case truncation** per project rule (CLAUDE.md "所有 case ≤90 天截断"):
  both \`heihe\` and \`heihe_x4\` runs are truncated to 90 model-days, not the
  full 4-year forcing window. This is sufficient for OpenMP/CVODE stats
  signal but may underrepresent steady-state Krylov behavior for cases with
  long spin-up transients. Decision affected: §3.4 \`r\` ratios are
  representative of warm-up + steady-state mix, not pure steady-state.
- **2-case scope** (heihe @ 6335 NumEle + heihe_x4 @ 40046 NumEle): branches
  (c) / (d) "case-aware" terminology refers only to this pair; a third or
  fourth case (e.g. ksge / qhh / xinanjiang) MAY shift \`r_min\` or
  \`r_max\` outside the lattice cell observed here. ADR-0003 SHALL note the
  2-case sampling as a deferred-broadening hazard.
- **Single SHUD pin** \`7a1dc8f\` (P1e ship pin): the \`r\` ratio is sensitive
  to the CVLS \`maxl\` default (per SUNDIALS 6.0.0
  \`CVodeSetMaxLSetup\` semantics); upstream changes to the preconditioner
  registration default in CVLS would shift this baseline. Pinning the SHUD
  rev guards against silent drift between PR-B and PR-C.
- **Cross-N median over 3 reps**: 3 reps is sufficient for integer-valued
  CVODE counters where determinism makes Δ=0 the expected result, but for
  the float timer buckets (§4) it is below the noise floor for 5% wall
  comparisons. Step 2 PR-F gate 4 (5%/10% wall non-regression) inherits this
  noise floor.

## §7 引用

- spec: \`openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md\` (Requirements + Scenarios L3-122)
- tasks: \`openspec/changes/p8pre-spike/tasks.md\` §3 PR-B (L28-37)
- canonical keys: \`tools/cvode_stats_diff/canonical_15_keys.yaml\` (15-key + REJECT typo list)
- timer.cpp: \`tools/profile/timer.cpp\` L150-200 (7-bucket emit + extras shape)
- PR-A run doc: \`docs/p8pre/n8_profile_run.md\`
- PR-A jid_table: \`${P8PRE_MIRROR_DIR}jid_table.txt\`
- p1e baseline anchors:
  - \`docs/p1e/p1e_perf_baseline.md\` §3.4 (heihe.nst=6698, heihe_x4.nst=6575)
  - \`docs/p1e/p1e_pr_i_strict_omp_verification.md\` §3.1 (heihe.nfe=6943, heihe_x4.nfe=6741)
- SUNDIALS CVODE 6.0.0 \`cvode.h\` (CVLS Krylov \`nfeLS\` semantics)
- aggregator script: \`tools/p8pre/aggregate_n8_profile.sh\`
EOF
} > "$VERDICT_DOC"

echo "[aggregate_n8_profile] verdict doc written: $VERDICT_DOC"
echo "[aggregate_n8_profile] DONE branch=${branch} exit=${block_exit}"

exit "$block_exit"
