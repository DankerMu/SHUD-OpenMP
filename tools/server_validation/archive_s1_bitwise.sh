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
  $PROG --selftest-preflight
  $PROG --deploy-instructions
  $PROG --resolve <server-scratch-root> <user> [<target-deployment-server>] [<ssh-port>] [<output-path>]

  --preflight             Verify sbatch template "三铁律" compliance
                          (--output/--error under /scratch only).

  --selftest-preflight    Run positive-test sanity self-check of the
                          preflight regex: (1) unmodified sbatch
                          produces empty bad_paths; (2) mangled sbatch
                          (scratch placeholder → /users/foo) produces
                          non-empty bad_paths.

  --deploy-instructions   Print rsync + ssh + sbatch commands for the
                          operator. No remote action is taken.

  --resolve               Substitute placeholders <server-scratch-root>,
                          <user>, <target-deployment-server>, <ssh-port>
                          in the template; write to <output-path>
                          (default: stdout). Server + port default to
                          \$SHUD_DEPLOY_HOST / \$SHUD_DEPLOY_SSH_PORT if
                          unset on the command line; if neither source is
                          supplied the placeholder stays literal so the
                          operator must fill it in by hand before sbatch.

Placeholders in the template:
  <server-scratch-root>      e.g. <user>/<project>
  <user>                     your username on the server
  <target-deployment-server> e.g. cluster.example.org (host/IP only)
  <ssh-port>                 numeric SSH port for the cluster
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
        # (F2 fix: original pattern had over-escaped backslashes that
        # required a literal `"\` byte sequence which never appears in
        # sbatch content; simplified to match raw substrings. We also
        # skip lines starting with `#` so documentation comments
        # describing the rule itself don't false-positive.)
        bad_paths=$(grep -nE '(/tmp/|/users/)' "$SBATCH_TEMPLATE" \
                    | grep -vE '^[0-9]+:[[:space:]]*#' || true)
        if [ -n "$bad_paths" ]; then
            printf "FAIL: sbatch references non-scratch paths:\n" >&2
            printf "%s\n" "$bad_paths" >&2
            exit 1
        fi
        printf "PASS: %s satisfies 三铁律 (CLAUDE.md compute-node policy)\n" "$SBATCH_TEMPLATE"
        ;;

    --selftest-preflight)
        # F2 fix: positive-test sanity self-check for the preflight regex.
        # Two assertions:
        #   1. Unmodified sbatch → bad_paths empty (regex doesn't false-positive
        #      on comment lines that document the /tmp/ /users/ rule)
        #   2. Mangled sbatch (scratch placeholder → /users/foo) → bad_paths
        #      non-empty (regex catches real leaks in non-comment lines)
        [ -r "$SBATCH_TEMPLATE" ] || { printf "FATAL: %s missing\n" "$SBATCH_TEMPLATE" >&2; exit 1; }

        # Assertion 1: unmodified template passes preflight (regex empty).
        # Mirror the preflight comment-skip filter.
        bad_paths_clean=$(grep -nE '(/tmp/|/users/)' "$SBATCH_TEMPLATE" \
                          | grep -vE '^[0-9]+:[[:space:]]*#' || true)
        if [ -n "$bad_paths_clean" ]; then
            printf "FAIL: selftest assertion 1 — unmodified sbatch flagged by regex (non-comment hit):\n" >&2
            printf "%s\n" "$bad_paths_clean" >&2
            exit 1
        fi
        printf "PASS: selftest assertion 1 — unmodified sbatch produces empty bad_paths\n"

        # Assertion 2: mangled sbatch (scratch placeholder swapped to /users/foo)
        # produces a non-empty match. Use a temp file under /tmp.
        #
        # F20: switch from "/tmp/...$$.sbatch" (TOCTOU-vulnerable;
        # predictable name another user can pre-create or symlink) to
        # mktemp's atomic O_CREAT|O_EXCL on a 6-char random suffix.
        # The EXIT trap also cleans up the sed-generated .bak sidecar.
        TMP_MANGLED=$(mktemp -t archive_s1_bitwise_selftest.XXXXXX) || {
            printf "FATAL: mktemp failed for selftest scratch file\n" >&2
            exit 1
        }
        trap 'rm -f "$TMP_MANGLED" "$TMP_MANGLED.bak"' EXIT
        cp "$SBATCH_TEMPLATE" "$TMP_MANGLED"
        # Mangle ONLY the scratch placeholder occurrences (not /tmp/ in
        # comments) — substitute "/scratch/<server-scratch-root>" → "/users/foo".
        sed -i.bak 's|/scratch/<server-scratch-root>|/users/foo|g' "$TMP_MANGLED"
        bad_paths_mangled=$(grep -nE '(/tmp/|/users/)' "$TMP_MANGLED" \
                            | grep -vE '^[0-9]+:[[:space:]]*#' || true)
        if [ -z "$bad_paths_mangled" ]; then
            printf "FAIL: selftest assertion 2 — mangled sbatch (placeholder→/users/foo) produced empty bad_paths; regex failed to catch real leak\n" >&2
            exit 1
        fi
        printf "PASS: selftest assertion 2 — mangled sbatch correctly flagged (%d match line(s))\n" "$(printf '%s\n' "$bad_paths_mangled" | wc -l | tr -d ' ')"
        printf "PASS: --selftest-preflight (2/2 assertions PASS)\n"
        ;;

    --deploy-instructions)
        cat <<'EOF'
# Manual deploy steps for S1 server-side bitwise validation
# (heihe + heihe_x4 cases; CLAUDE.md compute-node policy compliant).
#
# Placeholders below stay literal in the committed template; replace
# in the operator's local shell before running.  Server / port values
# come from your environment, never from this committed script.
#
# 1. From the local checkout, rsync the sbatch template to the server.
#    Replace <server-scratch-root> with the actual scratch path used,
#    e.g. `<user>/<project>`. <user> = your server username.
#
#    rsync -avh -e "ssh -p <ssh-port>" \
#        tools/server_validation/run_s1_bitwise.sbatch \
#        <user>@<target-deployment-server>:/scratch/<server-scratch-root>/.s1-server-validation/
#
# 2. SSH into the server and resolve the <server-scratch-root>/<user>
#    placeholders in the sbatch file:
#
#    ssh -p <ssh-port> <user>@<target-deployment-server>
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
#    rsync -avh -e "ssh -p <ssh-port>" \
#        <user>@<target-deployment-server>:/scratch/<server-scratch-root>/.s1-server-validation/<job-id>/compare.log \
#        ./tools/server_validation/last-server-run-compare.log
EOF
        ;;

    --resolve)
        [ $# -ge 3 ] || usage
        SCRATCH_ROOT="$2"
        SERVER_USER="$3"
        # F14: <target-deployment-server> and <ssh-port> get filled from
        # CLI args 4/5 first, falling back to env vars
        # SHUD_DEPLOY_HOST / SHUD_DEPLOY_SSH_PORT.  If neither source is
        # populated the literal placeholder stays in the rendered file
        # (so the operator catches it before running rsync/ssh).
        DEPLOY_HOST="${4:-${SHUD_DEPLOY_HOST:-<target-deployment-server>}}"
        SSH_PORT="${5:-${SHUD_DEPLOY_SSH_PORT:-<ssh-port>}}"
        OUT_PATH="${6:--}"  # default to stdout
        if [ "$OUT_PATH" = "-" ]; then
            sed -e "s|<server-scratch-root>|$SCRATCH_ROOT|g" \
                -e "s|<user>|$SERVER_USER|g" \
                -e "s|<target-deployment-server>|$DEPLOY_HOST|g" \
                -e "s|<ssh-port>|$SSH_PORT|g" \
                "$SBATCH_TEMPLATE"
        else
            sed -e "s|<server-scratch-root>|$SCRATCH_ROOT|g" \
                -e "s|<user>|$SERVER_USER|g" \
                -e "s|<target-deployment-server>|$DEPLOY_HOST|g" \
                -e "s|<ssh-port>|$SSH_PORT|g" \
                "$SBATCH_TEMPLATE" > "$OUT_PATH"
            printf "Wrote resolved sbatch to %s\n" "$OUT_PATH"
        fi
        ;;

    *)
        usage
        ;;
esac
