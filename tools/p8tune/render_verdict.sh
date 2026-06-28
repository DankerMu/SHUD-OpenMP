#!/usr/bin/env bash
# render_verdict.sh — concatenate T1-T8 markdown tables + verdict synthesis into
# a single `docs/p8tune/maxl_sweep_verdict.md` companion document.
#
# Inputs (defaults):
#   SWEEP_ROOT  /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep
#               (must contain T1_*.md ... T8_*.md + verdict_synthesis.md from
#                aggregate_maxl_sweep.sh)
#   OUT         docs/p8tune/maxl_sweep_verdict.md (relative to repo root)
#
# Output: single markdown verdict doc with YAML frontmatter + synthesis + 8 T-tables
#
# Usage:
#   render_verdict.sh                  # default paths
#   SWEEP_ROOT=/path OUT=docs/foo.md render_verdict.sh
#
# Exit codes:
#   0 OK
#   1 missing T-tables

set -euo pipefail

SWEEP_ROOT="${SWEEP_ROOT:-/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep}"
OUT="${OUT:-docs/p8tune/maxl_sweep_verdict.md}"

REQUIRED=(
    "${SWEEP_ROOT}/T1_G1_build.md"
    "${SWEEP_ROOT}/T2_G2_no_prec_left_regression.md"
    "${SWEEP_ROOT}/T3_G3_default_compat_4way.md"
    "${SWEEP_ROOT}/T4_G4_solver_work.md"
    "${SWEEP_ROOT}/T5_G5_wall.md"
    "${SWEEP_ROOT}/T6_G6_no_solver_regression.md"
    "${SWEEP_ROOT}/T7_G7_hydrology.md"
    "${SWEEP_ROOT}/T8_G8_determinism.md"
    "${SWEEP_ROOT}/verdict_synthesis.md"
)
for f in "${REQUIRED[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f (run aggregate_maxl_sweep.sh first)" >&2; exit 1; }
done

mkdir -p "$(dirname "${OUT}")"

cat > "${OUT}" <<EOF
---
title: P8-tune.C maxl sweep — 8-gate verdict
status: verdict-final
epic: p8tune-spgmr-maxl (#362)
pr_sequence: PR-E (#368)
adr_xref: docs/adr/0004-maxl-sweep-decision.md
data_source: /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv
data_provenance: Slurm 9690 (60/60 COMPLETED), tools/p8tune/aggregate_maxl_sweep.sh
---

# P8-tune.C maxl sweep — 8-gate verdict + ADR-0004 cross-reference

EOF

cat "${SWEEP_ROOT}/verdict_synthesis.md" >> "${OUT}"

echo -e "\n---\n## Detailed gate tables\n" >> "${OUT}"

for f in "${REQUIRED[@]:0:8}"; do
    cat "$f" >> "${OUT}"
    echo "" >> "${OUT}"
done

cat >> "${OUT}" <<EOF

---

## Cross-references

- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR
- [openspec/changes/p8tune-spgmr-maxl/](../../openspec/changes/p8tune-spgmr-maxl/) — OpenSpec change
- [docs/p8tune/clean_prec_none_baseline.md](clean_prec_none_baseline.md) — PR-A 18-cell PREC_NONE baseline + PR-B verdict gate
- [Slurm job 9690 summary.tsv](file:///scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv) — server-resident 60-cell raw data
- [aggregate_verdict.txt](file:///scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/aggregate_verdict.txt) — flat KV mirror of this doc

## Production tune guidance (Optional knob branch)

| (case, N) | recommended SHUD_SPGMR_MAXL | rationale (3-rep median wall vs unset baseline) |
|---|---|---|
| heihe N=1 | **=30** RECOMMENDED | +11.99% wall improvement (GO band), ncfl 85 → 0 (solver failures eliminated) |
| heihe N=8 | **=30** Optional | +6.78% wall improvement (Optional band), counter improvement |
| heihe_x4 N=1 | unset (default=5) | maxl ≥10 all REGRESS wall (−6.86% at 10 to −15.83% at 30); ncfl gain (3620 → 0) does NOT outweigh wall cost |
| heihe_x4 N=8 | unset (default=5) | maxl ≥10 all REGRESS wall (−15.81% to −24.82%); large-case + high-thread combo amplifies Krylov-vector memory bandwidth cost |

**Case-size-asymmetric pattern** (key finding): small case (heihe ~6300 elements) benefits from larger Krylov subspace; large case (heihe_x4 ~40046 elements) suffers because each Krylov vector occupies more memory bandwidth, and bigger maxl multiplies bandwidth pressure during Arnoldi orthogonalization. See ADR-0004 §discussion for mechanistic analysis.
EOF

echo "OK: rendered ${OUT}"
