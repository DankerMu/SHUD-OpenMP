#!/usr/bin/env bash
# tools/p1e_aggregate_pr_i_shall.sh
#
# P1e PR-I (issue #317) — server 24-cell mode-C SHALL gate aggregator.
#
# Idempotent re-aggregation of per-cell artifacts produced on the server by
# tools/p1e_2x2_runner.sh into the tables embedded in
# docs/p1e/p1e_pr_i_server_shall.md.
#
# Input layout (default --runs /tmp/p1e_pr_i_server_runs):
#   <runs>/C<case>_N<N>_rep<rep>/
#     cvode_stats.txt
#     <case>.rivqdown.sha256
#     wall.sec
#     cv_y_hash.txt
#     cell.meta
#
# Cell naming is concatenated `C<case>_N<N>_rep<rep>` (only mode C in PR-I).
#
# Modes:
#   roster       — full 24-row markdown table (default)
#   acs1         — AC-S1 mode C cross-N SHA stability per case (SHALL gate)
#   acs2         — AC-S2 mode C SHA == PR-D mode A reference (SHALL gate)
#   acs3         — AC-S3 D7 speedup AND-gate (SHALL gate)
#   speedup      — full wall-time matrix + per-case speedup
#   nst          — nst delta vs PR-D mode A reference (informational + ladder per case)
#   all          — run every gate + table sequentially
#
# Exit code is 0 unless an AC-S* SHALL gate FAILS.
#
# Usage:
#   tools/p1e_aggregate_pr_i_shall.sh [--runs <dir>] [mode]
#
# Reference SHAs (LOCKED from PR-D #312 mode A, baseline/P1e):
#   heihe    rivqdown.dat sha256 = a2023ccd2de43543a675df8eee46aa72ef1f9437764d28863dda62a4029e7798
#   heihe_x4 rivqdown.dat sha256 = b5e4b0a2cf83b2a4b97d6be5b40ea7e0580d59fb6934db73a95f366f5ffc72b4
#
# Reference nst (LOCKED from PR-D #312 mode A):
#   heihe    nst = 6698
#   heihe_x4 nst = 6575

set -euo pipefail

RUNS="/tmp/p1e_pr_i_server_runs"
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        roster|acs1|acs2|acs3|speedup|nst|all) MODE="$1"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$RUNS" ]]; then
    echo "error: runs dir not found: $RUNS" >&2
    exit 2
fi

CASES=(heihe heihe_x4)
NS=(1 2 4 8)
REPS=(1 2 3)

# Reference SHA + nst from PR-D #312 mode A (LOCKED baseline/P1e):
REF_SHA_HEIHE="a2023ccd2de43543a675df8eee46aa72ef1f9437764d28863dda62a4029e7798"
REF_SHA_HEIHE_X4="b5e4b0a2cf83b2a4b97d6be5b40ea7e0580d59fb6934db73a95f366f5ffc72b4"
REF_NST_HEIHE=6698
REF_NST_HEIHE_X4=6575

# D7 speedup thresholds (per design D7 + tasks §4.3):
SPEEDUP_THRESHOLD_HEIHE="1.3"
SPEEDUP_THRESHOLD_HEIHE_X4="1.5"

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

cell_dir() {
    # PR-I: only mode C; cell name = C<case>_N<N>_rep<rep>
    echo "$RUNS/C${1}_N${2}_rep${3}"
}

rivq_sha() {
    local f="$(cell_dir "$1" "$2" "$3")/${1}.rivqdown.sha256"
    [[ -f "$f" ]] && awk '{print $1}' "$f" || echo "MISSING"
}

wall_sec() {
    local f="$(cell_dir "$1" "$2" "$3")/wall.sec"
    [[ -f "$f" ]] && tr -d '[:space:]' < "$f" || echo "?"
}

cv_y_first_hash() {
    local f="$(cell_dir "$1" "$2" "$3")/cv_y_hash.txt"
    [[ -f "$f" ]] && awk 'NR==1{print $2}' "$f" || echo "MISSING"
}

cvode_field() {
    local key="$1"
    local f="$(cell_dir "$2" "$3" "$4")/cvode_stats.txt"
    [[ -f "$f" ]] && awk -F= -v k="$key" '$1==k{print $2}' "$f" || echo "?"
}

ref_sha() {
    case "$1" in
        heihe)    echo "$REF_SHA_HEIHE" ;;
        heihe_x4) echo "$REF_SHA_HEIHE_X4" ;;
        *)        echo "UNKNOWN_CASE" ;;
    esac
}

ref_nst() {
    case "$1" in
        heihe)    echo "$REF_NST_HEIHE" ;;
        heihe_x4) echo "$REF_NST_HEIHE_X4" ;;
        *)        echo "?" ;;
    esac
}

threshold_for() {
    case "$1" in
        heihe)    echo "$SPEEDUP_THRESHOLD_HEIHE" ;;
        heihe_x4) echo "$SPEEDUP_THRESHOLD_HEIHE_X4" ;;
        *)        echo "1.0" ;;
    esac
}

# Median of 3 reps (integer seconds), takes (case, N)
median_wall() {
    local c="$1" n="$2"
    local v1 v2 v3
    v1=$(wall_sec "$c" "$n" 1)
    v2=$(wall_sec "$c" "$n" 2)
    v3=$(wall_sec "$c" "$n" 3)
    if [[ "$v1" =~ ^[0-9]+$ && "$v2" =~ ^[0-9]+$ && "$v3" =~ ^[0-9]+$ ]]; then
        printf '%s\n%s\n%s\n' "$v1" "$v2" "$v3" | sort -n | sed -n '2p'
    else
        echo "?"
    fi
}

# --------------------------------------------------------------------------
# Roster (24 rows)
# --------------------------------------------------------------------------

emit_roster() {
    echo "| cell | case | N | rep | wall_sec | cvode_nst | cvode_nfe | cvode_netf | rivqdown_sha12 | cv_y_h0_sha12 |"
    echo "|------|------|--:|--:|---------:|----------:|----------:|-----------:|----------------|---------------|"
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            for r in "${REPS[@]}"; do
                local cell="C${c}_N${n}_rep${r}"
                local w=$(wall_sec "$c" "$n" "$r")
                local nst=$(cvode_field nst "$c" "$n" "$r")
                local nfe=$(cvode_field nfe "$c" "$n" "$r")
                local netf=$(cvode_field netf "$c" "$n" "$r")
                local rsha=$(rivq_sha "$c" "$n" "$r")
                local csha=$(cv_y_first_hash "$c" "$n" "$r")
                echo "| $cell | $c | $n | $r | $w | $nst | $nfe | $netf | \`${rsha:0:12}\` | \`${csha:0:12}\` |"
            done
        done
    done
}

# --------------------------------------------------------------------------
# AC-S1 — mode C cross-N bitwise per case (SHALL gate)
#         For each case, all 12 cells (4 N × 3 reps) must share one SHA.
# --------------------------------------------------------------------------

emit_acs1() {
    local fails=0
    echo "| case | unique_SHAs | rep1_N1 | rep2_N1 | rep3_N1 | rep1_N2 | rep1_N4 | rep1_N8 | verdict |"
    echo "|------|-----------:|---------|---------|---------|---------|---------|---------|:-------:|"
    for c in "${CASES[@]}"; do
        local all_shas=""
        for n in "${NS[@]}"; do
            for r in "${REPS[@]}"; do
                all_shas+="$(rivq_sha "$c" "$n" "$r")"$'\n'
            done
        done
        # Strip trailing empty line from heredoc accumulator; without grep -v,
        # identical SHAs report as 2 unique instead of 1 (bash here-string adds \n)
        local uniq=$(echo "$all_shas" | grep -v '^$' | sort -u | wc -l | tr -d ' ')
        local s11=$(rivq_sha "$c" 1 1)
        local s12=$(rivq_sha "$c" 1 2)
        local s13=$(rivq_sha "$c" 1 3)
        local s21=$(rivq_sha "$c" 2 1)
        local s41=$(rivq_sha "$c" 4 1)
        local s81=$(rivq_sha "$c" 8 1)
        local verdict="PASS"
        if [[ "$uniq" != "1" ]]; then verdict="**FAIL**"; fails=$((fails+1)); fi
        echo "| $c | $uniq | \`${s11:0:12}\` | \`${s12:0:12}\` | \`${s13:0:12}\` | \`${s21:0:12}\` | \`${s41:0:12}\` | \`${s81:0:12}\` | $verdict |"
    done
    echo
    if [[ $fails -eq 0 ]]; then
        echo "**AC-S1 verdict: PASS** (mode C SHA bitwise-identical across all N and reps for both cases)"
    else
        echo "**AC-S1 verdict: FAIL** ($fails case(s) split; strict-omp RHS race or shared state)"
    fi
    return $fails
}

# --------------------------------------------------------------------------
# AC-S2 — mode C SHA == PR-D mode A reference SHA (SHALL gate)
# --------------------------------------------------------------------------

emit_acs2() {
    local fails=0
    echo "| case | mode_C_SHA(N=1,rep=1) | reference_SHA(PR-D mode A) | match | verdict |"
    echo "|------|------------------------|----------------------------|:-----:|:-------:|"
    for c in "${CASES[@]}"; do
        local actual=$(rivq_sha "$c" 1 1)
        local expected=$(ref_sha "$c")
        local match="diff"
        local verdict="**FAIL**"
        if [[ "$actual" == "$expected" ]]; then
            match="same"
            verdict="PASS"
        else
            fails=$((fails+1))
        fi
        echo "| $c | \`${actual:0:16}\` | \`${expected:0:16}\` | $match | $verdict |"
    done
    echo
    if [[ $fails -eq 0 ]]; then
        echo "**AC-S2 verdict: PASS** (mode C matches PR-D mode A canonical → strict-omp RHS == serial RHS bitwise on server libgomp)"
    else
        echo "**AC-S2 verdict: FAIL** ($fails case(s) diverged from mode A reference; server libgomp diverges from Mac libomp finding in PR-H)"
    fi
    return $fails
}

# --------------------------------------------------------------------------
# AC-S3 — D7 speedup AND-gate (SHALL gate)
#         heihe N1/N8 ≥ 1.3× AND heihe_x4 N1/N8 ≥ 1.5×
# --------------------------------------------------------------------------

emit_acs3() {
    local fails=0
    local h_n1 h_n8 hx_n1 hx_n8 h_sp hx_sp h_pass hx_pass and_pass
    h_n1=$(median_wall heihe 1)
    h_n8=$(median_wall heihe 8)
    hx_n1=$(median_wall heihe_x4 1)
    hx_n8=$(median_wall heihe_x4 8)

    h_sp="?"
    hx_sp="?"
    if [[ "$h_n1" =~ ^[0-9]+$ && "$h_n8" =~ ^[0-9]+$ && "$h_n8" != "0" ]]; then
        h_sp=$(awk -v a="$h_n1" -v b="$h_n8" 'BEGIN{printf "%.3f", a/b}')
    fi
    if [[ "$hx_n1" =~ ^[0-9]+$ && "$hx_n8" =~ ^[0-9]+$ && "$hx_n8" != "0" ]]; then
        hx_sp=$(awk -v a="$hx_n1" -v b="$hx_n8" 'BEGIN{printf "%.3f", a/b}')
    fi

    h_pass=$(awk -v s="$h_sp" -v t="$SPEEDUP_THRESHOLD_HEIHE" 'BEGIN{print (s+0 >= t+0) ? "PASS" : "FAIL"}')
    hx_pass=$(awk -v s="$hx_sp" -v t="$SPEEDUP_THRESHOLD_HEIHE_X4" 'BEGIN{print (s+0 >= t+0) ? "PASS" : "FAIL"}')

    echo "| case | N=1 wall_median (s) | N=8 wall_median (s) | speedup | threshold | per-case verdict |"
    echo "|------|---------------------:|---------------------:|--------:|---------:|:----------------:|"
    echo "| heihe    | $h_n1  | $h_n8  | ${h_sp}x  | ${SPEEDUP_THRESHOLD_HEIHE}x  | $h_pass  |"
    echo "| heihe_x4 | $hx_n1 | $hx_n8 | ${hx_sp}x | ${SPEEDUP_THRESHOLD_HEIHE_X4}x | $hx_pass |"
    echo

    # AND-gate (per tasks §4.6: D12.3 only triggers if BOTH < threshold).
    # AC-S3 "PASS" requires BOTH per-case PASS.
    if [[ "$h_pass" == "PASS" && "$hx_pass" == "PASS" ]]; then
        and_pass="PASS"
        echo "**AC-S3 D7 AND-gate verdict: PASS** (both cases meet own threshold → D12.1 happy path)"
    elif [[ "$h_pass" == "FAIL" && "$hx_pass" == "FAIL" ]]; then
        and_pass="D12.3-fail"
        fails=1
        echo "**AC-S3 D7 AND-gate verdict: FAIL (D12.3)** (both cases below own threshold → D12.3 block-Jacobi fallback triggered)"
    else
        and_pass="partial"
        echo "**AC-S3 D7 AND-gate verdict: PARTIAL** (exactly one case meets threshold → tasks §4.6.2 partial-closure user decision point)"
        echo "  - heihe: $h_pass; heihe_x4: $hx_pass"
        echo "  - per tasks §4.6.2: prefer ship when heihe_x4 ≥ ${SPEEDUP_THRESHOLD_HEIHE_X4}x"
    fi
    return $fails
}

# --------------------------------------------------------------------------
# wall-time matrix (mean of 3 reps + median)
# --------------------------------------------------------------------------

emit_speedup() {
    echo "Mean of 3 reps (integer seconds) per cell:"
    echo
    echo "| case | N=1 mean | N=2 mean | N=4 mean | N=8 mean | N1/N8 mean |"
    echo "|------|---------:|---------:|---------:|---------:|----------:|"
    for c in "${CASES[@]}"; do
        local m1=$(mean_wall "$c" 1)
        local m2=$(mean_wall "$c" 2)
        local m4=$(mean_wall "$c" 4)
        local m8=$(mean_wall "$c" 8)
        local sp="?"
        if [[ "$m1" =~ ^[0-9.]+$ && "$m8" =~ ^[0-9.]+$ && "$m8" != "0" ]]; then
            sp=$(awk -v a="$m1" -v b="$m8" 'BEGIN{printf "%.3fx", a/b}')
        fi
        echo "| $c | $m1 | $m2 | $m4 | $m8 | $sp |"
    done
    echo
    echo "Median of 3 reps (integer seconds) per cell (used by AC-S3):"
    echo
    echo "| case | N=1 median | N=2 median | N=4 median | N=8 median | N1/N8 median |"
    echo "|------|-----------:|-----------:|-----------:|-----------:|-------------:|"
    for c in "${CASES[@]}"; do
        local m1=$(median_wall "$c" 1)
        local m2=$(median_wall "$c" 2)
        local m4=$(median_wall "$c" 4)
        local m8=$(median_wall "$c" 8)
        local sp="?"
        if [[ "$m1" =~ ^[0-9]+$ && "$m8" =~ ^[0-9]+$ && "$m8" != "0" ]]; then
            sp=$(awk -v a="$m1" -v b="$m8" 'BEGIN{printf "%.3fx", a/b}')
        fi
        echo "| $c | $m1 | $m2 | $m4 | $m8 | $sp |"
    done
    echo
    echo "Per-cell raw wall (sec):"
    echo
    echo "| case | N | rep1 | rep2 | rep3 |"
    echo "|------|--:|-----:|-----:|-----:|"
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            local w1=$(wall_sec "$c" "$n" 1)
            local w2=$(wall_sec "$c" "$n" 2)
            local w3=$(wall_sec "$c" "$n" 3)
            echo "| $c | $n | $w1 | $w2 | $w3 |"
        done
    done
}

mean_wall() {
    local c="$1" n="$2"
    local sum=0 count=0 v
    for r in "${REPS[@]}"; do
        v=$(wall_sec "$c" "$n" "$r")
        if [[ "$v" =~ ^[0-9]+$ ]]; then
            sum=$((sum + v))
            count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        awk -v s="$sum" -v n="$count" 'BEGIN{printf "%.1f", s/n}'
    else
        echo "?"
    fi
}

# --------------------------------------------------------------------------
# nst informational: mode C nst vs PR-D mode A reference, ladder check
# --------------------------------------------------------------------------

emit_nst() {
    echo "Mode C nst vs PR-D mode A reference (informational + nst ladder):"
    echo
    echo "| case | ref_nst (mode A) | N=1 rep1 | N=2 rep1 | N=4 rep1 | N=8 rep1 | max_delta_to_ref | ladder_OK |"
    echo "|------|-----------------:|---------:|---------:|---------:|---------:|-----------------:|:---------:|"
    for c in "${CASES[@]}"; do
        local refn=$(ref_nst "$c")
        local n1=$(cvode_field nst "$c" 1 1)
        local n2=$(cvode_field nst "$c" 2 1)
        local n4=$(cvode_field nst "$c" 4 1)
        local n8=$(cvode_field nst "$c" 8 1)
        local maxd=0
        for v in $n1 $n2 $n4 $n8; do
            if [[ "$v" =~ ^[0-9]+$ ]]; then
                local d=$(( v - refn )); [[ $d -lt 0 ]] && d=$((-d))
                [[ $d -gt $maxd ]] && maxd=$d
            fi
        done
        # ladder per spec p1d-numa-governance: heihe Δ=0 strict; heihe_x4 |Δ|≤2
        local ladder_thresh=0
        [[ "$c" == "heihe_x4" ]] && ladder_thresh=2
        local ladder_ok="PASS"
        if [[ $maxd -gt $ladder_thresh ]]; then ladder_ok="**FAIL** (delta=${maxd} > ${ladder_thresh})"; fi
        echo "| $c | $refn | $n1 | $n2 | $n4 | $n8 | $maxd | $ladder_ok |"
    done
}

# --------------------------------------------------------------------------
# Main dispatch
# --------------------------------------------------------------------------

emit_all() {
    local rc=0
    echo "## Roster (24 cells)"
    echo
    emit_roster
    echo
    echo "## AC-S1 (SHALL): mode C cross-N bitwise per case"
    echo
    emit_acs1 || rc=$?
    echo
    echo "## AC-S2 (SHALL): mode C SHA == PR-D mode A reference SHA per case"
    echo
    emit_acs2 || rc=$?
    echo
    echo "## AC-S3 (SHALL): D7 speedup AND-gate"
    echo
    emit_acs3 || rc=$?
    echo
    echo "## Speedup tables (informational)"
    echo
    emit_speedup
    echo
    echo "## nst ladder (informational)"
    echo
    emit_nst
    return $rc
}

case "$MODE" in
    roster)   emit_roster ;;
    acs1)     emit_acs1 ;;
    acs2)     emit_acs2 ;;
    acs3)     emit_acs3 ;;
    speedup)  emit_speedup ;;
    nst)      emit_nst ;;
    all)      emit_all ;;
esac
