#!/usr/bin/env bash
# tools/p8tune.F/render_verdict.sh
#
# P8-tune.F BoomerAMG pattern-only spike (PR-C). Per openspec change
# `p8tune-amg-spike` issue #397 §3 + spec REQ-5 (aggregate_verdict.txt
# consumer) + REQ-6 (ADR-0007 anchor).
#
# Reads aggregate_verdict.txt (produced by aggregate_amg_spike.sh) and
# emits docs/p8tune/amg_spike_verdict.md with:
#
#   1. Top-line verdict_branch
#   2. Per-case T-tables (4 cases × best combo + 5-axis result)
#   3. Raw 16-cell TSV inline
#   4. Footer: hypre + colpack + shud_pin provenance
#
# Inputs (env vars / CLI):
#   --verdict-txt <path>   aggregate_verdict.txt path
#   --tsv <path>           aggregate.tsv path (default = same dir as verdict)
#   --output <path>        docs/p8tune/amg_spike_verdict.md output
#
# Usage:
#   tools/p8tune.F/render_verdict.sh \
#       --verdict-txt .review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt \
#       --output docs/p8tune/amg_spike_verdict.md
#
# Exit codes:
#   0  OK
#   1  missing inputs
#   2  uv missing (CLAUDE.md mandates `uv run python`)
#
# Refs:
#   openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md
#       REQ-5 aggregate_verdict.txt schema; REQ-6 ADR-0007 byte-identical
#   tools/p8tune.D/render_verdict.sh (PR-B #387 reference template)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VERDICT_TXT="${REPO_ROOT}/.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt"
TSV=""
OUT="${REPO_ROOT}/docs/p8tune/amg_spike_verdict.md"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --verdict-txt)
            VERDICT_TXT="$2"
            shift 2
            ;;
        --tsv)
            TSV="$2"
            shift 2
            ;;
        --output)
            OUT="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '1,/^set -euo pipefail$/p' "$0" | head -n -2
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 [--verdict-txt <path>] [--tsv <path>] [--output <path>]" >&2
            exit 1
            ;;
    esac
done

# Default TSV = sibling of verdict file.
if [[ -z "${TSV}" ]]; then
    TSV="$(dirname "${VERDICT_TXT}")/aggregate.tsv"
fi

if [[ ! -f "${VERDICT_TXT}" ]]; then
    echo "ERROR: missing --verdict-txt: ${VERDICT_TXT}" >&2
    echo "ERROR: run aggregate_amg_spike.sh first." >&2
    exit 1
fi
if [[ ! -f "${TSV}" ]]; then
    echo "ERROR: missing --tsv: ${TSV}" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
    echo 'ERROR: uv missing (CLAUDE.md mandates "uv run python")' >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT}")"

export VERDICT_TXT TSV OUT

uv run python - <<'PYEOF'
import os
import re
from pathlib import Path
from collections import OrderedDict

VERDICT_TXT = Path(os.environ["VERDICT_TXT"])
TSV = Path(os.environ["TSV"])
OUT = Path(os.environ["OUT"])

# ---- Load aggregate_verdict.txt (preserve comment lines for context) ----
verdict_lines_raw = VERDICT_TXT.read_text().splitlines()
kv = {}
for line in verdict_lines_raw:
    if line.startswith("#") or "=" not in line:
        continue
    k, _, v = line.partition("=")
    kv[k.strip()] = v.strip().strip('"')

# ---- Load aggregate.tsv ----
tsv_lines = TSV.read_text().splitlines()
tsv_header = tsv_lines[0].split("\t")
tsv_rows = []
for line in tsv_lines[1:]:
    cols = line.split("\t")
    if len(cols) != len(tsv_header):
        continue
    tsv_rows.append(dict(zip(tsv_header, cols)))

CASE_ARR = ["keliya", "heihe", "heihe_x4", "heihe_x16"]

# Extract pinned constants from verdict.txt comments for caption / footer.
def extract_comment_kv(needle):
    for line in verdict_lines_raw:
        if line.startswith("#") and needle in line and "=" in line:
            # E.g.  "#   SPGMR_PER_STEP_SEC         = 0.226579"
            _, _, val = line.partition("=")
            return val.strip()
    return None


spgmr_per_step = extract_comment_kv("SPGMR_PER_STEP_SEC ")
cn_node_ram = extract_comment_kv("CN_NODE_RAM_BYTES ")
wall_budget_setup = extract_comment_kv("WALL_BUDGET_SETUP_SEC")
wall_budget_apply = extract_comment_kv("WALL_BUDGET_APPLY_SEC")
wall_budget_rss = extract_comment_kv("WALL_BUDGET_RSS_BYTES")
cycle_thr = extract_comment_kv("CYCLE_COMPLEXITY_THRESHOLD")
op_thr = extract_comment_kv("OPERATOR_COMPLEXITY_THRESHOLD")

hypre_version = extract_comment_kv("hypre_version") or "unknown"
colpack_version = extract_comment_kv("colpack_version") or "unknown"
shud_pin = extract_comment_kv("shud_pin") or "unknown"

verdict_branch = kv.get("verdict_branch", "?")
verdict_reason = kv.get("verdict_branch_reason", "")
verdict_branch_amended = kv.get("verdict_branch_axis4_amended", "?")
verdict_reason_amended = kv.get("verdict_branch_axis4_amended_reason", "")

# ---- Emit verdict markdown ----
out = []
out.append("---")
out.append("title: P8-tune.F BoomerAMG pattern-only spike — 5-axis per-case verdict")
out.append(f"status: verdict-final-PR-C")
out.append("epic: p8tune-amg-spike (#393)")
out.append("pr_sequence: PR-C")
out.append("adr_xref: docs/adr/0007-amg-spike-decision.md")
out.append("data_source: .review-evidence/p8tune-amg-pr-b/cells/")
out.append("data_provenance: Slurm 9896 16-cell array sweep (PR-B #396 / PR #403)")
out.append("aggregator: tools/p8tune.F/aggregate_amg_spike.sh")
out.append("renderer: tools/p8tune.F/render_verdict.sh")
out.append("---")
out.append("")
out.append("# P8-tune.F BoomerAMG pattern-only spike — verdict")
out.append("")
out.append("> **Generated** by `tools/p8tune.F/render_verdict.sh` from PR-B 16-cell sweep evidence.")
out.append("> **Spec** anchor: [openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md) §REQ-5 + §REQ-6.")
out.append("> **ADR**: [docs/adr/0007-amg-spike-decision.md](../adr/0007-amg-spike-decision.md).")
out.append("")
out.append("## Top-line verdict")
out.append("")
out.append(f"**verdict_branch = `{verdict_branch}`**")
out.append("")
if verdict_reason:
    out.append(f"> {verdict_reason}")
    out.append("")
out.append(
    f"**Axis 4 amended verdict (FYI for ADR §Discussion)**: `{verdict_branch_amended}`"
)
out.append("")
if verdict_reason_amended:
    out.append(f"> {verdict_reason_amended}")
    out.append("")
out.append(
    "Per PR-A H3 disclosure, `cycle_complexity = 2 × operator_complexity` "
    "is a hard-coded estimate in current `boomeramg_setup_solve.cpp` "
    "implementation (NOT measurement from HYPRE telemetry). Axis 4 "
    "mechanically tracks Axis 5 across all 16 observed cells. The "
    "aggregator emits the **strict** verdict as the canonical anchor; "
    "the amended verdict above is provided to inform ADR-0007 "
    "§Discussion treatment of Axis 4 as a non-discriminating diagnostic."
)
out.append("")

# Threshold rationale.
out.append("## Threshold rationale")
out.append("")
out.append(
    "Per spec REQ-5 Scenario \"Axis threshold constants pinned in shared "
    "header\" + \"5-axis threshold evaluation per case\":"
)
out.append("")
out.append("| Axis | Metric | Threshold | Derivation |")
out.append("|------|--------|-----------|------------|")
out.append(
    f"| 1 | `setup_wall_sec` | `< {wall_budget_setup}` s | "
    f"1.5 × 0.7 × SPGMR_PER_STEP_SEC = 1.5 × WALL_BUDGET_APPLY_SEC "
    f"(setup amortization allowance) |"
)
out.append(
    f"| 2 | `apply_wall_sec` | `< {wall_budget_apply}` s | "
    f"0.7 × SPGMR_PER_STEP_SEC ({spgmr_per_step} s, P8-tune.D pinned "
    f"baseline heihe_x4 N=1 maxl=5 3-rep median) |"
)
out.append(
    f"| 3 | `peak_rss_bytes` | `< {wall_budget_rss}` bytes (≈ 130 GiB) | "
    f"0.7 × CN_NODE_RAM_BYTES ({cn_node_ram} bytes ≈ 173 GiB cn-node RAM) |"
)
out.append(
    f"| 4 | `cycle_complexity` | `< {cycle_thr}` (ratio) | "
    f"V-cycle internal op count / NumY (canonical AMG hierarchy quality bound) |"
)
out.append(
    f"| 5 | `operator_complexity` | `< {op_thr}` (ratio) | "
    f"sum coarse grid sizes / fine grid size (canonical AMG memory bound) |"
)
out.append("")

# Per-case T-tables.
out.append("## Per-case T-tables")
out.append("")
out.append(
    "Best combo per case = `min(setup_wall_sec + apply_wall_sec)`; "
    "tiebreaker = `min(operator_complexity)` per spec REQ-5."
)
out.append("")

# Helper for KV lookup with fallback.
def gk(case, key, default="?"):
    return kv.get(f"{case}_{key}", default)

for case in CASE_ARR:
    nn = gk(case, "best_combo_nn")
    interp = gk(case, "best_combo_interp_type")
    coarsen = gk(case, "best_combo_coarsen_type")
    numy = gk(case, "best_combo_NumY")
    nnz = gk(case, "best_combo_nnz_A")
    overall = gk(case, "overall")
    failing = gk(case, "failing_axes")
    max_margin = gk(case, "max_failing_margin")

    out.append(f"### {case}")
    out.append("")
    if nn == "?" or interp == "?":
        out.append("**Best combo**: NONE (spike crashed for this case)")
        out.append("")
        continue
    out.append(
        f"**Best combo**: NN={nn}, interp_type={interp}, coarsen_type={coarsen}, "
        f"NumY={numy}, nnz_A={nnz}"
    )
    out.append("")
    out.append("| Axis | Value | Threshold | Margin (V/T) | Status |")
    out.append("|------|-------|-----------|--------------|--------|")
    for axis_key, axis_label in (
        ("axis1_setup", "1 Setup wall (s)"),
        ("axis2_apply", "2 Apply wall (s)"),
        ("axis3_memory", "3 Peak RSS (bytes)"),
        ("axis4_cycle", "4 Cycle complexity"),
        ("axis5_operator", "5 Operator complexity"),
    ):
        v = gk(case, f"{axis_key}_value")
        t = gk(case, f"{axis_key}_threshold")
        m = gk(case, f"{axis_key}_margin")
        s = gk(case, f"{axis_key}_status")
        out.append(f"| {axis_label} | {v} | {t} | {m} | {s} |")
    out.append("")
    out.append(
        f"**Overall (strict)**: `{overall}`  |  failing axes: `{failing}`  |  "
        f"max failing margin: `{max_margin}`"
    )
    out.append("")

# Raw TSV appendix.
out.append("## Raw 16-cell aggregate TSV")
out.append("")
out.append(
    f"Full data at `.review-evidence/p8tune-amg-pr-c/aggregate.tsv`:"
)
out.append("")
out.append("```tsv")
out.append(TSV.read_text().rstrip())
out.append("```")
out.append("")

# Machine-readable verdict block.
out.append("## Machine-readable verdict block")
out.append("")
out.append(
    "Canonical KV block consumed by ADR-0007 §Decision auto-fill per "
    "spec REQ-6:"
)
out.append("")
out.append("```")
out.append(VERDICT_TXT.read_text().rstrip())
out.append("```")
out.append("")

# Footer / provenance.
out.append("## Provenance")
out.append("")
out.append(f"- **Hypre version**: `{hypre_version}`")
out.append(f"- **ColPack version**: `{colpack_version}` (per PR-B M2 closure sentinel — ColPack reused via shell-out from P8-tune.D)")
out.append(f"- **SHUD pin**: `{shud_pin}`")
out.append(f"- **SPGMR baseline**: `{spgmr_per_step}` s/step (P8-tune.D pinned in `tools/p8tune.D/spgmr_baseline_walls.h`)")
out.append(f"- **CN node RAM**: `{cn_node_ram}` bytes (P8-tune.D pinned in `tools/p8tune.D/cn_node_ram.h`)")
out.append("")

# Cross-references.
out.append("## Cross-references")
out.append("")
out.append("- [docs/adr/0007-amg-spike-decision.md](../adr/0007-amg-spike-decision.md) — ADR-0007 (4-branch decision; this verdict is the §Decision anchor)")
out.append("- [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md) — ADR-0005 (KLU NO-GO at heihe_x4; triggered this AMG retreat)")
out.append("- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR-0004 (SPGMR baseline anchor)")
out.append("- [openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md) — capability spec")
out.append("- [.review-evidence/p8tune-amg-pr-b/](../../.review-evidence/p8tune-amg-pr-b/) — PR-B 16-cell evidence directory")
out.append("- [.review-evidence/p8tune-amg-pr-c/aggregate.tsv](../../.review-evidence/p8tune-amg-pr-c/aggregate.tsv) — this verdict's raw aggregate data")
out.append("- [.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt](../../.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt) — machine-readable verdict")
out.append("- [tools/p8tune.D/spgmr_baseline_walls.h](../../tools/p8tune.D/spgmr_baseline_walls.h) — pinned SPGMR baseline constant")
out.append("- [tools/p8tune.D/cn_node_ram.h](../../tools/p8tune.D/cn_node_ram.h) — pinned cn-node RAM constant")
out.append("")

OUT.write_text("\n".join(out))
print(f"OK: wrote {OUT}")
PYEOF
