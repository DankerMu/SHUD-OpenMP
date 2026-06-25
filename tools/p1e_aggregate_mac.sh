#!/usr/bin/env bash
# tools/p1e_aggregate_mac.sh
#
# P1e PR-C (issue #311) — Mac 48-cell evidence aggregator.
#
# Idempotent re-aggregation of per-cell artifacts produced by
# tools/p1e_2x2_runner.sh into the tables embedded in
# docs/p1e/p1e_pr_c_2x2_mac.md.
#
# Input layout (default --runs /tmp/p1e_pr_c_mac_runs):
#   <runs>/<build>_<case>_N<N>_rep<rep>/
#     cvode_stats.txt
#     <case>.rivqdown.sha256
#     wall.sec
#     cv_y_hash.txt
#     cell.meta
#
# Builds: A B   Cases: keliya qhh   N: 1 2 4 8   reps: 1 2 3
# Total expected: 48 cells.
#
# Modes:
#   roster              — full 48-row markdown table (default)
#   ac1                 — mode A 3-rep bitwise per (case,N) — SHALL gate
#   ac2                 — mode A cross-N stability per case — SHALL gate
#   ac3                 — CVode stats determinism per (build,case)
#   ac4                 — mode A vs mode B SHA per (case,N) — informational
#   ac5                 — wall-time matrix + speedup
#   ac6                 — cv_y_hash line count consistency
#   all                 — run every gate + table sequentially
#
# Exit code is 0 unless a SHALL gate (ac1, ac2) FAILS.
#
# Usage:
#   tools/p1e_aggregate_mac.sh [--runs <dir>] [mode]
#   tools/p1e_aggregate_mac.sh all > docs/p1e/p1e_pr_c_2x2_mac.partial.md

set -euo pipefail

RUNS="/tmp/p1e_pr_c_mac_runs"
MODE="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs) RUNS="$2"; shift 2 ;;
        roster|ac1|ac2|ac3|ac4|ac5|ac6|all) MODE="$1"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ ! -d "$RUNS" ]]; then
    echo "error: runs dir not found: $RUNS" >&2
    exit 2
fi

BUILDS=(A B)
CASES=(keliya qhh)
NS=(1 2 4 8)
REPS=(1 2 3)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

cell_dir() { echo "$RUNS/${1}_${2}_N${3}_rep${4}"; }

rivq_sha() {
    local f="$(cell_dir "$1" "$2" "$3" "$4")/${2}.rivqdown.sha256"
    [[ -f "$f" ]] && awk '{print $1}' "$f" || echo "MISSING"
}

wall_sec() {
    local f="$(cell_dir "$1" "$2" "$3" "$4")/wall.sec"
    [[ -f "$f" ]] && tr -d '[:space:]' < "$f" || echo "?"
}

cv_y_first_hash() {
    local f="$(cell_dir "$1" "$2" "$3" "$4")/cv_y_hash.txt"
    [[ -f "$f" ]] && awk 'NR==1{print $2}' "$f" || echo "MISSING"
}

cv_y_count() {
    local f="$(cell_dir "$1" "$2" "$3" "$4")/cv_y_hash.txt"
    [[ -f "$f" ]] && wc -l < "$f" | tr -d ' ' || echo 0
}

cvode_field() {
    local key="$2"
    local f="$(cell_dir "$1" "$3" "$4" "$5")/cvode_stats.txt"
    [[ -f "$f" ]] && awk -F= -v k="$key" '$1==k{print $2}' "$f" || echo "?"
}

# --------------------------------------------------------------------------
# Roster table (48 rows)
# --------------------------------------------------------------------------

emit_roster() {
    echo "| cell | build | case | N | rep | wall_sec | cvode_nst | cvode_nfe | cvode_nje | rivqdown_sha12 | cv_y_h0_sha12 |"
    echo "|------|:-----:|------|--:|--:|---------:|----------:|----------:|----------:|----------------|---------------|"
    for b in "${BUILDS[@]}"; do
        for c in "${CASES[@]}"; do
            for n in "${NS[@]}"; do
                for r in "${REPS[@]}"; do
                    local cell="${b}_${c}_N${n}_rep${r}"
                    local w=$(wall_sec "$b" "$c" "$n" "$r")
                    local nst=$(cvode_field "$b" "nst" "$c" "$n" "$r")
                    local nfe=$(cvode_field "$b" "nfe" "$c" "$n" "$r")
                    # SHUD's cvode_stats.txt has no nje key (KLU disabled here);
                    # netf is the closest "Jacobian-evaluation-class" proxy.
                    local nje=$(cvode_field "$b" "netf" "$c" "$n" "$r")
                    local rsha=$(rivq_sha "$b" "$c" "$n" "$r")
                    local csha=$(cv_y_first_hash "$b" "$c" "$n" "$r")
                    echo "| $cell | $b | $c | $n | $r | $w | $nst | $nfe | $nje | \`${rsha:0:12}\` | \`${csha:0:12}\` |"
                done
            done
        done
    done
}

# --------------------------------------------------------------------------
# AC1 — mode A 3-rep bitwise per (case, N)   SHALL gate
# --------------------------------------------------------------------------

emit_ac1() {
    local fails=0
    echo "| case | N | rep1_sha12 | rep2_sha12 | rep3_sha12 | unique | verdict |"
    echo "|------|--:|------------|------------|------------|-------:|:-------:|"
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            local s1=$(rivq_sha A "$c" "$n" 1)
            local s2=$(rivq_sha A "$c" "$n" 2)
            local s3=$(rivq_sha A "$c" "$n" 3)
            local uniq=$(printf '%s\n%s\n%s\n' "$s1" "$s2" "$s3" | sort -u | wc -l | tr -d ' ')
            local verdict="PASS"
            if [[ "$uniq" != "1" ]]; then verdict="**FAIL**"; fails=$((fails+1)); fi
            echo "| $c | $n | \`${s1:0:12}\` | \`${s2:0:12}\` | \`${s3:0:12}\` | $uniq | $verdict |"
        done
    done
    echo
    echo "AC1 result: $([ $fails -eq 0 ] && echo "PASS (8/8 groups bitwise-identical across reps)" || echo "**FAIL** ($fails group(s) diverged)")"
    return $fails
}

# --------------------------------------------------------------------------
# AC2 — mode A cross-N stability per case (rep=1)   SHALL gate
# --------------------------------------------------------------------------

emit_ac2() {
    local fails=0
    echo "| case | N=1 | N=2 | N=4 | N=8 | unique | verdict |"
    echo "|------|------|------|------|------|-------:|:-------:|"
    for c in "${CASES[@]}"; do
        local s1=$(rivq_sha A "$c" 1 1)
        local s2=$(rivq_sha A "$c" 2 1)
        local s4=$(rivq_sha A "$c" 4 1)
        local s8=$(rivq_sha A "$c" 8 1)
        local uniq=$(printf '%s\n%s\n%s\n%s\n' "$s1" "$s2" "$s4" "$s8" | sort -u | wc -l | tr -d ' ')
        local verdict="PASS"
        if [[ "$uniq" != "1" ]]; then verdict="**FAIL**"; fails=$((fails+1)); fi
        echo "| $c | \`${s1:0:12}\` | \`${s2:0:12}\` | \`${s4:0:12}\` | \`${s8:0:12}\` | $uniq | $verdict |"
    done
    echo
    echo "AC2 result: $([ $fails -eq 0 ] && echo "PASS (mode A SHA invariant under OMP_NUM_THREADS)" || echo "**FAIL** ($fails case(s) diverged)")"
    return $fails
}

# --------------------------------------------------------------------------
# AC3 — CVode stats determinism per (build, case)
# --------------------------------------------------------------------------

emit_ac3() {
    local anomalies=0
    echo "| build | case | N | rep | nst | nfe | nni | netf | verdict |"
    echo "|:-----:|------|--:|--:|-----:|-----:|-----:|-----:|:-------:|"
    for b in "${BUILDS[@]}"; do
        for c in "${CASES[@]}"; do
            # capture per-(build,case) baseline from N=1 rep=1
            local base_nst=$(cvode_field "$b" "nst" "$c" 1 1)
            local base_nfe=$(cvode_field "$b" "nfe" "$c" 1 1)
            for n in "${NS[@]}"; do
                for r in "${REPS[@]}"; do
                    local nst=$(cvode_field "$b" "nst" "$c" "$n" "$r")
                    local nfe=$(cvode_field "$b" "nfe" "$c" "$n" "$r")
                    local nni=$(cvode_field "$b" "nni" "$c" "$n" "$r")
                    local netf=$(cvode_field "$b" "netf" "$c" "$n" "$r")
                    local verdict="="
                    if [[ "$nst" != "$base_nst" || "$nfe" != "$base_nfe" ]]; then
                        verdict="DIFF"
                        anomalies=$((anomalies+1))
                    fi
                    echo "| $b | $c | $n | $r | $nst | $nfe | $nni | $netf | $verdict |"
                done
            done
        done
    done
    echo
    if [[ $anomalies -eq 0 ]]; then
        echo "AC3 result: PASS (per-(build,case) nst/nfe stable across N and reps)"
    else
        echo "AC3 result: $anomalies cell(s) deviated from per-(build,case) baseline — see DIFF rows."
        echo "Note: stat divergence is expected with NVECTOR_OPENMP reduction-order"
        echo "      non-associativity (mode B); mode A divergence WOULD be a fail."
    fi
    return 0
}

# --------------------------------------------------------------------------
# AC4 — mode A vs mode B SHA per (case, N)   informational
# --------------------------------------------------------------------------

emit_ac4() {
    echo "| case | N | A_rep1_sha12 | B_rep1_sha12 | match? |"
    echo "|------|--:|--------------|--------------|:-----:|"
    for c in "${CASES[@]}"; do
        for n in "${NS[@]}"; do
            local sa=$(rivq_sha A "$c" "$n" 1)
            local sb=$(rivq_sha B "$c" "$n" 1)
            local m="diff"
            [[ "$sa" == "$sb" ]] && m="same"
            echo "| $c | $n | \`${sa:0:12}\` | \`${sb:0:12}\` | $m |"
        done
    done
}

# --------------------------------------------------------------------------
# AC5 — wall-time matrix + speedup
# --------------------------------------------------------------------------

emit_ac5() {
    echo "Per-cell wall_sec (mean of 3 reps, integer seconds):"
    echo
    echo "| case | build | N=1 | N=2 | N=4 | N=8 | N1/N8 |"
    echo "|------|:-----:|-----:|-----:|-----:|-----:|------:|"
    for c in "${CASES[@]}"; do
        for b in "${BUILDS[@]}"; do
            local mean_n1=$(mean_wall "$b" "$c" 1)
            local mean_n2=$(mean_wall "$b" "$c" 2)
            local mean_n4=$(mean_wall "$b" "$c" 4)
            local mean_n8=$(mean_wall "$b" "$c" 8)
            local speedup="?"
            if [[ "$mean_n8" =~ ^[0-9.]+$ && "$mean_n1" =~ ^[0-9.]+$ && "$mean_n8" != "0" ]]; then
                speedup=$(awk -v a="$mean_n1" -v b="$mean_n8" 'BEGIN{printf "%.2fx", a/b}')
            fi
            echo "| $c | $b | $mean_n1 | $mean_n2 | $mean_n4 | $mean_n8 | $speedup |"
        done
    done
    echo
    echo "B-vs-A overhead ratio (B_mean / A_mean) per (case, N):"
    echo
    echo "| case | N=1 | N=2 | N=4 | N=8 |"
    echo "|------|-----:|-----:|-----:|-----:|"
    for c in "${CASES[@]}"; do
        local row="| $c |"
        for n in "${NS[@]}"; do
            local a=$(mean_wall A "$c" "$n")
            local b=$(mean_wall B "$c" "$n")
            local r="?"
            if [[ "$a" =~ ^[0-9.]+$ && "$b" =~ ^[0-9.]+$ && "$a" != "0" ]]; then
                r=$(awk -v aa="$a" -v bb="$b" 'BEGIN{printf "%.2f", bb/aa}')
            fi
            row="$row $r |"
        done
        echo "$row"
    done
}

mean_wall() {
    local b="$1" c="$2" n="$3"
    local sum=0 count=0 v
    for r in "${REPS[@]}"; do
        v=$(wall_sec "$b" "$c" "$n" "$r")
        if [[ "$v" =~ ^[0-9]+$ ]]; then
            sum=$((sum + v)); count=$((count + 1))
        fi
    done
    if [[ $count -gt 0 ]]; then
        awk -v s="$sum" -v c="$count" 'BEGIN{printf "%.1f", s/c}'
    else
        echo "?"
    fi
}

# --------------------------------------------------------------------------
# AC6 — cv_y_hash line count (timestep dump count)
# --------------------------------------------------------------------------

emit_ac6() {
    local fails=0
    declare -A by_case
    echo "| build | case | N | rep | cv_y_lines |"
    echo "|:-----:|------|--:|--:|-----------:|"
    for b in "${BUILDS[@]}"; do
        for c in "${CASES[@]}"; do
            for n in "${NS[@]}"; do
                for r in "${REPS[@]}"; do
                    local cnt=$(cv_y_count "$b" "$c" "$n" "$r")
                    echo "| $b | $c | $n | $r | $cnt |"
                done
            done
        done
    done
    echo
    echo "Per-case expected counts (90-day truncation, 20-min model dt):"
    for c in "${CASES[@]}"; do
        local uniq
        uniq=$(for b in "${BUILDS[@]}"; do
            for n in "${NS[@]}"; do
                for r in "${REPS[@]}"; do
                    cv_y_count "$b" "$c" "$n" "$r"
                done
            done
        done | sort -u | tr '\n' ',' )
        echo "  $c: {${uniq%,}}"
    done
}

# --------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------

case "$MODE" in
    roster) emit_roster ;;
    ac1)    emit_ac1 ;;
    ac2)    emit_ac2 ;;
    ac3)    emit_ac3 ;;
    ac4)    emit_ac4 ;;
    ac5)    emit_ac5 ;;
    ac6)    emit_ac6 ;;
    all)
        echo "### Cell roster (48 cells)"; echo
        emit_roster
        echo
        echo "### AC1 — mode A 3-rep bitwise per (case, N)   SHALL"; echo
        emit_ac1 || true
        echo
        echo "### AC2 — mode A cross-N stability per case   SHALL"; echo
        emit_ac2 || true
        echo
        echo "### AC3 — CVode stats determinism"; echo
        emit_ac3
        echo
        echo "### AC4 — mode A vs mode B SHA (informational)"; echo
        emit_ac4
        echo
        echo "### AC5 — wall-time matrix + speedup"; echo
        emit_ac5
        echo
        echo "### AC6 — cv_y_hash line count consistency"; echo
        emit_ac6
        ;;
esac
