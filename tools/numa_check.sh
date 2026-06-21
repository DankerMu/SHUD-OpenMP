#!/usr/bin/env bash
# tools/numa_check.sh — S5d.4 (#182) NUMA topology probe + run summary emit
#
# Spec: openspec/changes/b1b-baseline-completion/specs/s5d-data-layout-soa-numa
#   Requirement "NUMA 探测工具与 run log 落盘"
#   Scenarios  "numa_check.sh 输出包含硬件拓扑"
#              "本地 Mac 单 socket UMA 跳过 NUMA 验收"
#
# What this probe does:
#   1. Captures `numactl --hardware` output to <run_dir>/numa_topo.log.
#      On hosts without numactl (Apple Silicon, BSD), captures a synthetic
#      one-line note so the per-run summary downstream stays uniform.
#   2. Extracts the socket / node count from the capture and emits a
#      summary block to stderr (operator-facing) + appends it to
#      <run_dir>/numa_summary.txt (run-log persistent).
#   3. Cross-checks `OMP_PROC_BIND` env on multi-socket hosts; if multi-
#      socket AND OMP_PROC_BIND is unset, flags `numa_first_touch:
#      WARNING` per spec L109 design R3 mitigation #2.
#
# Usage:
#   tools/numa_check.sh <run_dir>
#
# Exit status:
#   0  — probe completed (happy path, both single- and multi-socket).
#   1  — script-side error (missing $1, run_dir not writable).
#
# Design notes (Linus-grade tooling discipline):
#   - The probe is idempotent: re-running overwrites numa_topo.log and
#     numa_summary.txt in $run_dir, no append-explosion risk.
#   - Apple Silicon path: numactl is BSD-userland only; macOS has no
#     numactl in Homebrew (libnuma is a Linux interface). We emit
#     `socket_count: 1` and `numa_first_touch: N/A (single-socket UMA)`
#     directly, matching spec Scenario L115-117 verbatim.
#   - The script avoids parsing `lscpu` because servers without
#     `numactl --hardware` are not in the supported endpoint set (per
#     CLAUDE.md, the dual-socket Xeon idle nodes cn05-06,09,14-19,23-24
#     all ship numactl from a stock Linux distro).

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "[NUMA] ERROR: usage: tools/numa_check.sh <run_dir>" 1>&2
    echo "[NUMA] ERROR: <run_dir> will receive numa_topo.log + numa_summary.txt" 1>&2
    exit 1
fi

run_dir="$1"
if ! mkdir -p "$run_dir" 2>/dev/null; then
    echo "[NUMA] ERROR: cannot create / write to run_dir=$run_dir" 1>&2
    exit 1
fi

topo_log="$run_dir/numa_topo.log"
summary_log="$run_dir/numa_summary.txt"

# Step 1 — capture topology
if command -v numactl >/dev/null 2>&1; then
    # Linux + numactl path. `numactl --hardware` is allowed to exit
    # non-zero on hosts with libnuma stub but no NUMA support; we
    # tolerate that with `|| true` so the summary still lands.
    numactl --hardware >"$topo_log" 2>&1 || \
        echo "[NUMA] numactl --hardware exited non-zero; see captured output" >>"$topo_log"
else
    # macOS / BSD / minimal Linux without numactl. We still write the
    # log file so downstream consumers (run summary aggregators) find
    # a deterministic artifact at the canonical path.
    {
        printf 'numactl: not available on this host (likely Apple Silicon / macOS UMA).\n'
        printf 'Fallback: single-socket UMA assumed; NUMA acceptance N/A per spec L115-117.\n'
    } >"$topo_log"
fi

# Step 2 — extract socket / node count
# `numactl --hardware` first line on Linux is `available: N nodes (0-...)`;
# the canonical anchor we grep for is `^available:`. Per spec L113
# "summary `socket_count: 2`", the value we surface IS the NUMA node count,
# which on multi-socket Xeon equates 1:1 with sockets (numactl labels each
# CPU package as a NUMA node).
socket_count=1
if grep -qE '^available: ' "$topo_log" 2>/dev/null; then
    # Extract the first integer that follows `available:`. awk handles
    # the "available: 2 nodes (0-1)" pattern with field index $2.
    socket_count=$(awk '/^available:/ {print $2; exit}' "$topo_log")
    if [[ -z "$socket_count" || ! "$socket_count" =~ ^[0-9]+$ ]]; then
        socket_count=1
    fi
fi

# Step 3 — emit summary + cross-check OMP_PROC_BIND
{
    printf 'socket_count: %s\n' "$socket_count"
    if [[ "$socket_count" -le 1 ]]; then
        # Single-socket UMA path. Spec L115-117 mandates `socket_count: 1`
        # + `N/A (single-socket UMA)` label.
        printf 'numa_first_touch: N/A (single-socket UMA)\n'
    else
        # Multi-socket path. Cross-check OMP_PROC_BIND in the calling
        # env; an unset binding on a multi-socket host means first-touch
        # is structurally ineffective (PR #181 skip path triggers and
        # NUMA pages bind to whatever thread happens to touch them
        # first — typically all on socket 0).
        bind_val="${OMP_PROC_BIND:-}"
        if [[ -z "$bind_val" ]]; then
            printf 'numa_first_touch: WARNING (OMP_PROC_BIND unset on %s-socket host)\n' \
                "$socket_count"
            printf '[NUMA] WARNING: OMP_PROC_BIND unset on %s-socket host; first-touch ineffective. Use tools/run_omp.sh.\n' \
                "$socket_count" 1>&2
        else
            printf 'numa_first_touch: OK (OMP_PROC_BIND=%s)\n' "$bind_val"
        fi
    fi
} | tee "$summary_log" 1>&2

exit 0
