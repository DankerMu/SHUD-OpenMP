#!/bin/bash
# run.sh — 3-run repeatability harness for before-PassValue snapshots.
#
# Generates `benchmarks/<case>/B0_output/repeatability_snapshots.txt`:
# 3 independent SHUD runs × 3 t-values = 9 SHA256 rows; deterministic iff
# the 3 per-row SHAs are identical across runs. F27b (PR #54 round-3)
# promoted this from a one-shot .workplans/ harness to a committed tool
# so any reviewer can re-run the evidence on any machine + any case.
#
# Usage:
#   bash tools/snapshot_repeatability/run.sh <case>
#
# Supported cases (must match SHUD/Basins/<case> layout + benchmarks/<case>):
#   keliya                 proj=keliya
#   xinanjiang_upstream    proj=xinanjiang
#   qinyijiang             proj=nanlin
#   qhh                    proj=qhh
#
# START / END day values are parsed from cfg.para at runtime (F33, PR #54
# round-4); no per-case hardcoding.
#
# Prerequisites (the user is responsible for):
#   1. SHUD binary built with SHUD_DUMP_RHS=1
#        cd SHUD && make clean && make SHUD_DUMP_RHS=1 shud
#   2. Case forcing data populated under SHUD/Basins/<case>/
#   3. cfg.para END = START + 90 (project-level truncation rule)
#
# Outputs:
#   benchmarks/<case>/B0_output/repeatability_snapshots.txt
#     (overwritten on each invocation; 9 SHA rows + header + verdict)
#
# Exit codes:
#   0  3 runs complete, all 9 snapshots produced, file written
#   1  any run failed / any expected snapshot missing
#   2  usage error / unsupported case
#
# Owned by: PR #54 round-3 F27 (promoted from .workplans/run_f13_repeatability.sh).

set -eu

PROG=$(basename "$0")
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

usage() {
    cat <<EOF >&2
Usage: $PROG <case>

Supported cases: keliya / xinanjiang_upstream / qinyijiang / qhh
EOF
    exit 2
}

[ $# -eq 1 ] || usage

CASE="$1"

# Per-case parameters: only PROJ (SHUD project key in input/<proj>/, which is
# the filename prefix for cfg.para and other inputs). START / END day values
# are parsed from cfg.para at runtime (F33, PR #54 round-4) so any drift in
# cfg.para — including the project-level 90-day truncation rule — is honored
# without script edits.
case "$CASE" in
    keliya)
        PROJ=keliya
        ;;
    xinanjiang_upstream)
        PROJ=xinanjiang
        ;;
    qinyijiang)
        PROJ=nanlin
        ;;
    qhh)
        PROJ=qhh
        ;;
    *)
        printf "%s: unsupported case '%s'\n\n" "$PROG" "$CASE" >&2
        usage
        ;;
esac

CASEDIR="$REPO/SHUD/Basins/$CASE"
OUTFILE="$REPO/benchmarks/$CASE/B0_output/repeatability_snapshots.txt"
CFG_PARA="$CASEDIR/input/$PROJ/$PROJ.cfg.para"

[ -d "$CASEDIR" ] || {
    printf "%s: case dir %s not found (forcing data not deployed)\n" "$PROG" "$CASEDIR" >&2
    exit 2
}
[ -d "$(dirname "$OUTFILE")" ] || {
    printf "%s: archive dir %s not found\n" "$PROG" "$(dirname "$OUTFILE")" >&2
    exit 2
}
[ -x "$REPO/SHUD/shud" ] || {
    printf "%s: SHUD/shud binary not built (run: cd SHUD && make SHUD_DUMP_RHS=1 shud)\n" "$PROG" >&2
    exit 2
}

# F33 (PR #54 round-4): parse START/END from cfg.para instead of hardcoding.
# cfg.para format: "START<TAB>12053" / "END<TAB>12143" (whitespace-separated).
# Bails if either key is missing so a stale / malformed cfg surfaces immediately
# (was: hardcoded constants silently mis-targeted snapshot abs-min and SHUD
# dump silently never fired).
[ -f "$CFG_PARA" ] || {
    printf "%s: cfg.para not found: %s\n" "$PROG" "$CFG_PARA" >&2
    exit 2
}
START_DAY=$(awk '$1=="START"{print $2; exit}' "$CFG_PARA")
END_DAY=$(awk '$1=="END"{print $2; exit}' "$CFG_PARA")
if [ -z "$START_DAY" ] || [ -z "$END_DAY" ]; then
    printf "%s: cannot parse START/END from %s (START='%s' END='%s')\n" \
        "$PROG" "$CFG_PARA" "$START_DAY" "$END_DAY" >&2
    exit 2
fi

# Pick a sha256 tool: prefer GNU sha256sum, fall back to BSD shasum -a 256.
if command -v sha256sum >/dev/null 2>&1; then
    SHA_CMD='sha256sum'
elif command -v shasum >/dev/null 2>&1; then
    SHA_CMD='shasum -a 256'
else
    printf "%s: no sha256sum or shasum in PATH\n" "$PROG" >&2
    exit 2
fi

START_MIN=$((START_DAY * 1440))
ABS_T1=$((START_MIN + 1440))      # 1 day
ABS_T2=$((START_MIN + 43200))     # 30 day
ABS_T3=$((START_MIN + 129600))    # 90 day
REL_T=(86400 2592000 7776000)     # archive convention (case-relative seconds)
ABS_TS=("$ABS_T1" "$ABS_T2" "$ABS_T3")

DATE=$(git -C "$REPO" log -1 --format=%ci HEAD)
SHUD_HEAD=$(git -C "$REPO/SHUD" rev-parse --short=12 HEAD)

printf "===== %s (proj=%s, start_day=%d, end_day=%d) =====\n" "$CASE" "$PROJ" "$START_DAY" "$END_DAY"

# Write header.
{
    echo "# 3-run repeatability for before-PassValue snapshots"
    echo "# case: $CASE"
    echo "# date: $DATE"
    echo "# SHUD pin: $SHUD_HEAD"
    echo "# cfg.para END = START + 90 (gitignored deployment cfg)"
    echo "# t_values (abs min): ${ABS_T1},${ABS_T2},${ABS_T3}"
    echo "# rel t_values (sec) for filename: ${REL_T[0]},${REL_T[1]},${REL_T[2]}"
    echo "# columns: run_label  filename_as_archived  sha256"
} > "$OUTFILE"

for i in 1 2 3; do
    STAGING="$CASEDIR/.snapshot-repeatability-run$i"
    rm -rf "$STAGING"
    mkdir -p "$STAGING"
    rm -rf "$CASEDIR/output/${PROJ}.out"

    printf "  run%d ...\n" "$i"
    (
        cd "$CASEDIR"
        SHUD_DUMP_T_VALUES="$ABS_T1,$ABS_T2,$ABS_T3" \
        SHUD_DUMP_T_TOL=60 \
        SHUD_DUMP_OUTPUT_DIR="$STAGING" \
        SHUD_DUMP_CASE_ID="$PROJ" \
        SHUD_DUMP_SITE="f_loop_before_passvalue" \
        SHUD_DUMP_FNAME_SUFFIX="before_passvalue" \
        ../../shud "$PROJ" >"$STAGING/run.log" 2>&1
    )

    # Verify all 3 expected files produced.
    PRODUCED=0
    for ABS in "${ABS_TS[@]}"; do
        F="$STAGING/snapshot_t${ABS}_before_passvalue.bin"
        if [ ! -f "$F" ]; then
            printf "FATAL: %s run%d missing %s\n" "$CASE" "$i" "$F" >&2
            tail -10 "$STAGING/run.log" >&2 || true
            exit 1
        fi
        PRODUCED=$((PRODUCED + 1))
    done
    [ "$PRODUCED" -eq 3 ] || {
        printf "FATAL: %s run%d produced=%d\n" "$CASE" "$i" "$PRODUCED" >&2
        exit 1
    }

    # SHA256 each, write row using case-rel sec filename (archive convention).
    for idx in 0 1 2; do
        ABS="${ABS_TS[$idx]}"
        REL="${REL_T[$idx]}"
        ARCHIVE_FNAME="snapshot_t${REL}_before_passvalue.bin"
        SHA=$($SHA_CMD "$STAGING/snapshot_t${ABS}_before_passvalue.bin" | awk '{print $1}')
        printf "run%d  %s  %s\n" "$i" "$ARCHIVE_FNAME" "$SHA" >> "$OUTFILE"
    done
done

# Verify determinism: 3 unique-bin SHAs across all 3 runs (3 per row × 3 rows = 9 rows; deterministic ↔ 3 unique).
# F31 (PR #54 round-4): filter $3 to SHA256-shaped tokens only. Without the
# regex guard, awk picks up the trailing blank line ($3 = "") and the comment
# row "# All 3 per-row SHAs identical across runs = deterministic." ($3 = "3"),
# inflating UNIQ from 3 to 5 and flipping the verdict to NON-DETERMINISTIC on
# every re-run against a known-good file.
UNIQ=$(awk 'NR>8 && $3 ~ /^[0-9a-f]{64}$/ {print $3}' "$OUTFILE" | sort -u | wc -l | tr -d ' ')
if [ "$UNIQ" -eq 3 ]; then
    printf "  %s: DETERMINISTIC (3 unique SHAs across 9 rows)\n" "$CASE"
    echo "" >> "$OUTFILE"
    echo "# All 3 per-row SHAs identical across runs = deterministic." >> "$OUTFILE"
else
    printf "  %s: NON-DETERMINISTIC (%d unique SHAs, expected 3)\n" "$CASE" "$UNIQ" >&2
    echo "" >> "$OUTFILE"
    printf "# WARNING: %d unique SHAs across 9 rows (expected 3 = deterministic). Inspect rows.\n" "$UNIQ" >> "$OUTFILE"
    exit 1
fi

# Cleanup stagings.
rm -rf "$CASEDIR/.snapshot-repeatability-run1" "$CASEDIR/.snapshot-repeatability-run2" "$CASEDIR/.snapshot-repeatability-run3"

printf "===== %s DONE =====\n" "$CASE"
exit 0
