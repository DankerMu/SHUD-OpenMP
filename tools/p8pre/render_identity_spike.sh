#!/bin/bash
# tools/p8pre/render_identity_spike.sh
#
# p8pre-spike Step 2 PR-E (issue #346) — wrapper that materializes the
# 18-cell identity-precond spike matrix by substituting markers in
# tools/p8pre/submit_identity_spike_template.sbatch and writing the rendered
# files to /scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/identity_spike/rendered/.
#
# Matrix per design D4: 2 cases × 3 N-values × 3 reps = 18 cells.
#   cases = heihe (cn14 pin) + heihe_x4 (cn15 pin)
#   N     = 1, 4, 8 (N=2 explicitly skipped)
#   reps  = 1, 2, 3
#
# Singleton chain per (case, N): rep 2 depends afterany on rep 1, rep 3
# depends afterany on rep 2 — keeps the 3 reps within a (case, N) group
# serial on the pinned node so intra-host contention does not inflate
# wall variance.
#
# Phase 1 dry-run: this script PRINTS 18 `sbatch ...` invocations to stdout
# instead of auto-submitting. The PR-E run scope (issue #346) is responsible
# for actually piping these into `sbatch`. Printing means a reviewer can eye-
# ball the matrix + dependency wiring before any cycles burn.
#
# Output dir for rendered .sbatch files:
#   /scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/identity_spike/rendered/
# (under /scratch per Slurm 三铁律 #3; .p8pre-runs/ matches the .p1e-runs/
# convention and is auto-gitignored by the outer repo's .gitignore '.s*-runs/'
# pattern style.)
#
# Slurm 三铁律 compliance (CLAUDE.md):
#   1. The rendered sbatch files must be submitted FROM /scratch (this
#      wrapper merely renders; the runner script in #346 PR-E run scope
#      SHALL cd /scratch/frd_muziyao/SHUD-OpenMP before sbatch).
#   2. #SBATCH --output/--error in the template are pinned to /scratch.
#   3. SHUD binary path in the template is pinned under
#      /scratch/frd_muziyao/SHUD-OpenMP/SHUD/.
#
# POSIX bash + awk only; no Python dependency.

set -euo pipefail

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Locate template relative to this script (works both for local dry-run on
# Mac and for server invocation under /scratch/.../tools/p8pre/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/submit_identity_spike_template.sbatch"

# Output dir on server. Created with mkdir -p; auto-gitignored by .s*-runs/
# pattern in outer repo .gitignore (matches .p8pre-runs/ glob).
OUT_DIR="/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/identity_spike/rendered"

# Matrix definition (per design D4 + tasks.md §1.5)
CASES=(heihe heihe_x4)
NS=(1 4 8)              # NB: N=2 explicitly SKIPPED — not a typo
REPS=(1 2 3)

# Per-case node pin (per P1e PR-I deployment SOP — pins cross-cell CPU SKU)
declare -A NODE_OF
NODE_OF[heihe]=cn14
NODE_OF[heihe_x4]=cn15

# ----------------------------------------------------------------------------
# Sanity
# ----------------------------------------------------------------------------

if [ ! -r "$TEMPLATE" ]; then
    echo "ERROR: template not readable: $TEMPLATE" >&2
    exit 1
fi

# Detect server vs local environment. On server we mkdir -p OUT_DIR; on Mac
# local dry-run we still print but do NOT create /scratch/... (it doesn't
# exist on Mac). The 18 rendered filenames are predictable from the matrix.
if [ -d "$(dirname "$OUT_DIR")" ]; then
    mkdir -p "$OUT_DIR"
    WRITE_RENDERED=1
else
    WRITE_RENDERED=0
fi

# ----------------------------------------------------------------------------
# Render loop — 18 cells, with singleton dependency chain per (case, N).
#
# Strategy: emit `sbatch ...` lines to stdout. For rep>1, append
# --dependency=afterany:<prev_jobid_placeholder> so the actual #346 PR-E
# runner can resolve the JID by capturing sbatch stdout (`Submitted batch
# job <JID>`). We use placeholder `__PREV_JID_<case>_N<n>_rep<r-1>__` which
# the runner SHALL substitute via shell variable after each rep 1+2 submit.
#
# Job names embed __singleton suffix so Slurm scheduler can also enforce
# at-most-one-running per (case, N) group as a belt-and-braces guard
# against runner JID-capture bugs.
# ----------------------------------------------------------------------------

CELL_COUNT=0
for case in "${CASES[@]}"; do
    node="${NODE_OF[$case]}"
    for n in "${NS[@]}"; do
        prev_placeholder=""
        for rep in "${REPS[@]}"; do
            CELL_COUNT=$((CELL_COUNT + 1))
            rendered_name="submit_${case}_N${n}_rep${rep}.sbatch"
            rendered_path="$OUT_DIR/$rendered_name"

            if [ "$WRITE_RENDERED" -eq 1 ]; then
                # Substitute markers and write rendered .sbatch
                awk -v case_v="$case" -v n_v="$n" -v rep_v="$rep" -v node_v="$node" '
                    {
                        gsub(/__CASE__/, case_v);
                        gsub(/__N__/, n_v);
                        gsub(/__REP__/, rep_v);
                        gsub(/__NODE__/, node_v);
                        print;
                    }
                ' "$TEMPLATE" > "$rendered_path"
                chmod +x "$rendered_path"
            fi

            # Build sbatch invocation line for stdout (dry-run print)
            singleton_name="p8pre_identity_${case}_N${n}_singleton"
            if [ -z "$prev_placeholder" ]; then
                # First rep in (case, N) group: no dependency
                echo "sbatch --job-name=${singleton_name} ${rendered_path}"
            else
                # rep 2 or rep 3: depend afterany on previous rep's JID
                echo "sbatch --job-name=${singleton_name} --dependency=afterany:${prev_placeholder} ${rendered_path}"
            fi

            # Update placeholder for next rep in same (case, N) group
            prev_placeholder="__PREV_JID_${case}_N${n}_rep${rep}__"
        done
    done
done

# ----------------------------------------------------------------------------
# Self-check on stderr (does not pollute stdout sbatch lines)
# ----------------------------------------------------------------------------

if [ "$CELL_COUNT" -ne 18 ]; then
    echo "ERROR: expected 18 cells but rendered $CELL_COUNT" >&2
    exit 2
fi

echo "[render_identity_spike] rendered $CELL_COUNT cells; write_rendered=$WRITE_RENDERED; out_dir=$OUT_DIR" >&2
