#!/bin/sh
# archive_s1_bitwise.sh — manual-deploy wrapper for run_s1_bitwise.sbatch.
#
# This wrapper exists because S1 server-only validation is NOT automated:
# heihe / heihe_x4 forcing data lives only on the server, and Slurm
# submission must follow CLAUDE.md "三铁律" (submit from /scratch,
# --output/--error under /scratch, all referenced files under /scratch).
#
# Usage:
#   1. Preflight: verify the sbatch template references /scratch only.
#        ./archive_s1_bitwise.sh --preflight
#   2. Print the deploy + submit commands for the operator (rsync from
#      local to server, then ssh + sbatch).
#        ./archive_s1_bitwise.sh --deploy-instructions
#   3. (Optional) Substitute placeholders in the sbatch template and
#      write the resolved file to a temp path; the operator scp's it.
#        ./archive_s1_bitwise.sh --resolve <server-scratch-root> <user>
#
# This script never runs sbatch itself — submission is an operator-side
# action so that the user retains explicit consent over compute-cluster
# resource usage. The script's job is to make the deploy steps safe +
# predictable.
#
# Exit codes:
#   0  success (preflight passed / commands printed)
#   1  preflight failure (sbatch references non-scratch paths)
#   2  usage error
#
# Owned by: openspec change s1-rhs-core-extraction (Group 0 task 0.3).

set -eu

PROG=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
SBATCH_TEMPLATE="$SCRIPT_DIR/run_s1_bitwise.sbatch"

usage() {
    cat <<EOF >&2
Usage:
  $PROG --preflight
  $PROG --deploy-instructions
  $PROG --resolve <server-scratch-root> <user> [<output-path>]

  --preflight             Verify sbatch template "三铁律" compliance
                          (--output/--error under /scratch only).

  --deploy-instructions   Print rsync + ssh + sbatch commands for the
                          operator. No remote action is taken.

  --resolve               Substitute placeholders <server-scratch-root>
                          and <user> in the template; write to
                          <output-path> (default: stdout).

Placeholders in the template:
  <server-scratch-root>   e.g. frd_muziyao/SHUD-OpenMP
  <user>                  your username on the server
EOF
    exit 2
}

[ $# -ge 1 ] || usage

case "$1" in
    --preflight)
        # 三铁律 check: --output and --error must be under /scratch.
        [ -r "$SBATCH_TEMPLATE" ] || { printf "FATAL: %s missing\n" "$SBATCH_TEMPLATE" >&2; exit 1; }
        bad=$(grep -E '^#SBATCH (--output|--error)' "$SBATCH_TEMPLATE" | grep -v '/scratch/' || true)
        if [ -n "$bad" ]; then
            printf "FAIL: sbatch --output/--error not under /scratch:\n" >&2
            printf "%s\n" "$bad" >&2
            exit 1
        fi
        # 第三条铁律: referenced files (patch / hash / run.sh) must be /scratch.
        # In our template, the only referenced fs paths are PROJECT_ROOT
        # and ARCHIVE_DIR — both rooted at /scratch/<server-scratch-root>.
        # Check that no /tmp / /users/ references slipped in.
        bad_paths=$(grep -nE '"(\\/tmp\\/|\\/users\\/)' "$SBATCH_TEMPLATE" || true)
        if [ -n "$bad_paths" ]; then
            printf "FAIL: sbatch references non-scratch paths:\n" >&2
            printf "%s\n" "$bad_paths" >&2
            exit 1
        fi
        printf "PASS: %s satisfies 三铁律 (CLAUDE.md compute-node policy)\n" "$SBATCH_TEMPLATE"
        ;;

    --deploy-instructions)
        cat <<'EOF'
# Manual deploy steps for S1 server-side bitwise validation
# (heihe + heihe_x4 cases; CLAUDE.md compute-node policy compliant).
#
# 1. From the local checkout, rsync the sbatch template to the server.
#    Replace <server-scratch-root> with the actual scratch path used,
#    e.g. `frd_muziyao/SHUD-OpenMP`. <user> = your server username.
#
#    rsync -avh -e "ssh -p 32099" \
#        tools/server_validation/run_s1_bitwise.sbatch \
#        <user>@210.77.77.22:/scratch/<server-scratch-root>/.s1-server-validation/
#
# 2. SSH into the server and resolve the <server-scratch-root>/<user>
#    placeholders in the sbatch file:
#
#    ssh -p 32099 <user>@210.77.77.22
#    cd /scratch/<server-scratch-root>/.s1-server-validation/
#    sed -i 's|<server-scratch-root>|<actual-path>|g; s|<user>|<your-username>|g' run_s1_bitwise.sbatch
#
# 3. Pull latest outer repo + SHUD submodule (do NOT push the resolved
#    sbatch back to the repo — placeholders stay in the committed template):
#
#    cd /scratch/<server-scratch-root>
#    git pull --recurse-submodules
#
# 4. Submit (MUST be from /scratch, not from /users/$USER):
#
#    cd /scratch/<server-scratch-root>/.s1-server-validation/
#    sbatch run_s1_bitwise.sbatch
#
# 5. Monitor + harvest results:
#
#    squeue -u <your-username>
#    tail -f /scratch/<server-scratch-root>/.s1-server-validation/shud-s1-bitwise.*.out
#    # On success, archive lives at /scratch/.s1-server-validation/<job-id>/
#    # PASS/FAIL summary printed at the end of the .out log.
#
# 6. Pull SHA256 results back to the local PR comment:
#
#    rsync -avh -e "ssh -p 32099" \
#        <user>@210.77.77.22:/scratch/<server-scratch-root>/.s1-server-validation/<job-id>/compare.log \
#        ./tools/server_validation/last-server-run-compare.log
EOF
        ;;

    --resolve)
        [ $# -ge 3 ] || usage
        SCRATCH_ROOT="$2"
        SERVER_USER="$3"
        OUT_PATH="${4:--}"  # default to stdout
        if [ "$OUT_PATH" = "-" ]; then
            sed -e "s|<server-scratch-root>|$SCRATCH_ROOT|g" \
                -e "s|<user>|$SERVER_USER|g" \
                "$SBATCH_TEMPLATE"
        else
            sed -e "s|<server-scratch-root>|$SCRATCH_ROOT|g" \
                -e "s|<user>|$SERVER_USER|g" \
                "$SBATCH_TEMPLATE" > "$OUT_PATH"
            printf "Wrote resolved sbatch to %s\n" "$OUT_PATH"
        fi
        ;;

    *)
        usage
        ;;
esac
