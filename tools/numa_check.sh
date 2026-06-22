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
#   0  — probe completed (single-socket UMA, or multi-socket with
#        OMP_PROC_BIND set — first-touch is actionable).
#   1  — script-side error (missing $1, run_dir not writable).
#   3  — multi-socket host with OMP_PROC_BIND unset; first-touch
#        structurally ineffective. Calling sbatch / CI gate SHOULD fail
#        on this exit code so the operator notices the misconfiguration.
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
#
# socket_count guard — must remain numeric (the warning_state computation
# below uses string comparisons `== "0"` / `== "1"` for the single-socket
# branch). If you simplify or rewrite the extraction, also update the
# string-comparison check below; otherwise a non-numeric value could
# silently fall through to the multi-socket branch.
socket_count=1
if grep -qE '^available: ' "$topo_log" 2>/dev/null; then
    # Extract the first integer that follows `available:`. awk handles
    # the "available: 2 nodes (0-1)" pattern with field index $2.
    socket_count=$(awk '/^available:/ {print $2; exit}' "$topo_log")
    if [[ -z "$socket_count" || ! "$socket_count" =~ ^[0-9]+$ ]]; then
        socket_count=1
    fi
fi

# Step 3a — compute warning state BEFORE the emit pipeline.
#
# Why outside the pipe: in bash the LHS of a pipeline runs in a subshell,
# so any variable assignment inside `{ ... } | tee` is lost after the
# pipe completes. To propagate the WARNING decision to the final exit
# code (3 on multi-socket + OMP_PROC_BIND unset, 0 otherwise) we resolve
# the state here, then drive both the emit branch and the exit branch
# from the same single source of truth.
warning_state="ok"
if [[ "$socket_count" != "0" && "$socket_count" != "1" ]] && \
   [[ -z "${OMP_PROC_BIND:-}" ]]; then
    warning_state="warning"
fi

# Step 3b — emit summary + cross-check OMP_PROC_BIND.
#
# The producer block writes EVERY line that should land in both
# numa_summary.txt (via `tee`) and the operator stderr channel
# (via `1>&2` redirection of tee's stdout). The previous version
# duplicated the `[NUMA] WARNING:` line by emitting it both INSIDE
# the producer (via tee) AND OUTSIDE via a stderr-only printf —
# operators saw it twice. Now there is exactly one source: the
# producer block. tee mirrors it to both summary file and stderr.
{
    printf 'socket_count: %s\n' "$socket_count"
    case "$warning_state" in
        ok)
            if [[ "$socket_count" == "0" || "$socket_count" == "1" ]]; then
                # Single-socket UMA path. Spec L115-117 mandates
                # `socket_count: 1` + `N/A (single-socket UMA)` label.
                printf 'numa_first_touch: N/A (single-socket UMA)\n'
            else
                # Multi-socket WITH OMP_PROC_BIND set — first-touch is
                # actionable; report OK with the bind value for traceability.
                printf 'numa_first_touch: OK (OMP_PROC_BIND=%s on %s-socket host)\n' \
                    "${OMP_PROC_BIND}" "$socket_count"
            fi
            ;;
        warning)
            # Multi-socket WITHOUT OMP_PROC_BIND. First-touch is
            # structurally ineffective (PR #181 skip path triggers and
            # NUMA pages bind to whatever thread happens to touch them
            # first — typically all on socket 0). Emit both the
            # summary-line (for numa_summary.txt) AND the actionable
            # `[NUMA] WARNING:` (for operator log greps). Both lines
            # land in numa_summary.txt + stderr in one pass.
            printf 'numa_first_touch: WARNING (OMP_PROC_BIND unset on %s-socket host)\n' \
                "$socket_count"
            printf '[NUMA] WARNING: OMP_PROC_BIND unset on %s-socket host; first-touch ineffective. Use tools/run_omp.sh.\n' \
                "$socket_count"
            ;;
    esac
} | tee "$summary_log" 1>&2

# Step 4 — exit code reflects the WARNING decision computed in Step 3a.
#   0 — happy path (single-socket UMA, or multi-socket with binding set).
#   3 — multi-socket host with OMP_PROC_BIND unset (first-touch
#       structurally ineffective; calling sbatch / CI gate SHOULD fail).
case "$warning_state" in
    warning) exit 3 ;;
    *) exit 0 ;;
esac
