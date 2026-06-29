#!/usr/bin/env bash
# tools/p8tune.D/render_verdict.sh
#
# P8-tune.D KLU pattern-only spike (PR-B). Per openspec change
# `p8tune-klu-spike` tasks.md §3.5 + spec REQ-5 Scenario "Per-case axis
# machine-readable".
#
# Reads aggregate.tsv + aggregate_verdict.txt (produced by
# aggregate_klu_spike.sh) and emits docs/p8tune/klu_spike_verdict.md
# with per-case T-tables + 3-axis synthesis + raw data appendix +
# machine-readable verdict block.
#
# Inputs (env vars / defaults):
#   AGG_DIR     .review-evidence/p8tune-klu-spike-pr-b
#   OUT         docs/p8tune/klu_spike_verdict.md
#
# Usage:
#   tools/p8tune.D/render_verdict.sh
#   AGG_DIR=/tmp/agg OUT=/tmp/verdict.md tools/p8tune.D/render_verdict.sh
#
# Exit codes:
#   0 OK
#   1 missing inputs
#   2 python3 missing
#
# Refs:
#   openspec/changes/p8tune-klu-spike/tasks.md §3.5
#   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5
#   tools/p8tune/render_verdict.sh (PR-D #373 reference template)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

AGG_DIR="${AGG_DIR:-${REPO_ROOT}/.review-evidence/p8tune-klu-spike-pr-b}"
OUT="${OUT:-${REPO_ROOT}/docs/p8tune/klu_spike_verdict.md}"

TSV="${AGG_DIR}/aggregate.tsv"
KV="${AGG_DIR}/aggregate_verdict.txt"

if [[ ! -f "${TSV}" ]]; then
    echo "ERROR: missing ${TSV} (run aggregate_klu_spike.sh first)" >&2
    exit 1
fi
if [[ ! -f "${KV}" ]]; then
    echo "ERROR: missing ${KV} (run aggregate_klu_spike.sh first)" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 missing" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUT}")"

export AGG_DIR OUT TSV KV

python3 - <<'PYEOF'
import os
import math
import re
from pathlib import Path
from collections import defaultdict

TSV = Path(os.environ["TSV"])
KV = Path(os.environ["KV"])
OUT = Path(os.environ["OUT"])

# ---- Load aggregate.tsv ----
rows = []
with open(TSV) as f:
    header = f.readline().rstrip("\n").split("\t")
    for line in f:
        cols = line.rstrip("\n").split("\t")
        if len(cols) != len(header):
            continue
        d = dict(zip(header, cols))
        rows.append(d)

# ---- Load aggregate_verdict.txt ----
kv = {}
with open(KV) as f:
    for line in f:
        if line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        kv[k.strip()] = v.strip().strip('"')

CASE_ARR = ["keliya", "heihe", "heihe_x4", "heihe_x16"]
CN_NODE_RAM_BYTES = int(kv["CN_NODE_RAM_BYTES"])
SPGMR_PER_STEP_WALL_S = float(kv["SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s"])

# Per-case best-combo lookup (lowest fill_ratio amongst PASS cells).
by_case = defaultdict(list)
for r in rows:
    by_case[r["case"]].append(r)

def best_combo(case):
    pass_rows = [r for r in by_case[case] if r["verdict_class"] == "PASS" and r["fill_ratio"]]
    if not pass_rows:
        return None
    return sorted(pass_rows, key=lambda r: (float(r["fill_ratio"]), float(r["numeric_wall_s"] or "inf")))[0]

def get_numy(case):
    for r in by_case[case]:
        if r["NumY"]:
            return int(r["NumY"])
    return None

def fmt_or(v, fmt, default="—"):
    if v in (None, "", "—"):
        return default
    try:
        return fmt.format(float(v))
    except (ValueError, TypeError):
        return str(v) if v else default

# ---- Emit verdict markdown ----
out_lines = []
out_lines.append("---")
out_lines.append("title: P8-tune.D KLU pattern-only spike — 3-axis per-case verdict")
out_lines.append("status: verdict-final")
out_lines.append("epic: p8tune-klu-spike (#379)")
out_lines.append("pr_sequence: PR-B")
out_lines.append("adr_xref: docs/adr/0005-klu-spike-decision.md")
out_lines.append("data_source: .review-evidence/p8tune-klu-spike-pr-a/cells/")
out_lines.append("data_provenance: Slurm 9762 (NN=00-07) + 9794+9812+9828+9829 re-runs (NN=08-15)")
out_lines.append("aggregator: tools/p8tune.D/aggregate_klu_spike.sh")
out_lines.append("---")
out_lines.append("")
out_lines.append("# P8-tune.D KLU pattern-only spike — verdict")
out_lines.append("")
out_lines.append("> **Generated** by `tools/p8tune.D/render_verdict.sh` from PR-A 16-cell sweep evidence.")
out_lines.append("> **Spec** anchor: [openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md) §REQ-5 / §REQ-6.")
out_lines.append(f"> **ADR**: [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md).")
out_lines.append("")
out_lines.append("## Top-line verdict")
out_lines.append("")
out_lines.append("| case      | fill axis | RSS axis | wall axis | overall | recommended action  |")
out_lines.append("|-----------|-----------|----------|-----------|---------|---------------------|")
for case in CASE_ARR:
    f_ax = kv.get(f"{case}_KLU_fill_axis", "?")
    r_ax = kv.get(f"{case}_KLU_rss_axis", "?")
    w_ax = kv.get(f"{case}_KLU_wall_axis", "?")
    overall = kv.get(f"{case}_KLU_overall_verdict", "?")
    rec = kv.get(f"{case}_recommended_action", "?")
    out_lines.append(f"| {case:<9} | {f_ax:<9} | {r_ax:<8} | {w_ax:<9} | {overall:<7} | {rec:<19} |")
out_lines.append("")
out_lines.append(f"**Case-aware branch fires**: `{kv.get('case_aware_branch_fires', '?')}` (per spec REQ-5 Scenario \"Case-aware/Optional branch auto-population\" — small cases GO, large cases NO-GO/Optional on wall axis).")
out_lines.append("")
out_lines.append(f"**Decisive cell**: heihe_x4 → recommended next epic = `{kv.get('heihe_x4_recommended_next_epic', '?')}` (priority = `{kv.get('heihe_x4_recommended_next_epic_priority', '?')}`).")
out_lines.append("")

# Threshold rationale.
out_lines.append("## Threshold rationale")
out_lines.append("")
out_lines.append("Per spec REQ-5 Scenarios \"Fill axis threshold\" / \"RSS axis threshold\" / \"Wall axis threshold\":")
out_lines.append("")
out_lines.append("### Fill axis: `fill_ratio < 8 · log₂(NumY)`")
out_lines.append("")
out_lines.append("Rationale: 2D mesh PDE nested-dissection theoretical optimum ≈ log₂(NumY); the 8× factor allows real-world AMD/COLAMD deviation while still bounding LU pivoting cost. Per-case thresholds:")
out_lines.append("")
out_lines.append("| case      | NumY (measured) | 8·log₂(NumY) threshold |")
out_lines.append("|-----------|-----------------|------------------------|")
for case in CASE_ARR:
    numy = get_numy(case)
    if numy:
        out_lines.append(f"| {case:<9} | {numy:<15} | {8 * math.log2(numy):.1f}                  |")
out_lines.append("")
out_lines.append("### RSS axis: `peak_RSS < 0.7 × CN_NODE_RAM_BYTES`")
out_lines.append("")
out_lines.append(f"`CN_NODE_RAM_BYTES` = `{CN_NODE_RAM_BYTES}` (measured at PR-0 via `cat /proc/meminfo` on cn14; ≈ {CN_NODE_RAM_BYTES / 1024**3:.1f} GiB total node RAM).")
out_lines.append("")
out_lines.append(f"Threshold: 0.7 × {CN_NODE_RAM_BYTES / 1024**3:.1f} GiB ≈ {0.7 * CN_NODE_RAM_BYTES / 1024**3:.1f} GiB. Rationale: allows multi-cell parallel execution on a cn-node without OOM (per spec REQ-5 Scenario \"RSS axis threshold\").")
out_lines.append("")
out_lines.append("### Wall axis: `(numeric_factor_wall / refactor_freq) + (N_solve · solve_wall) < 0.7 × SPGMR_per_step_wall`")
out_lines.append("")
out_lines.append(f"`SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s` = `{SPGMR_PER_STEP_WALL_S:.6f}` (heihe_x4 N=1 maxl=5 3-rep median = 1489.76s / nst=6575 ≈ 0.227 s/step; pinned in `tools/p8tune.D/spgmr_baseline_walls.h`).")
out_lines.append("")
out_lines.append(f"Budget: 0.7 × {SPGMR_PER_STEP_WALL_S:.4f} = {0.7 * SPGMR_PER_STEP_WALL_S:.4f} s/step.")
out_lines.append("")
out_lines.append("KLU per-step estimate uses `refactor_freq=10` (conservative — refactor every 10 CVODE steps) and `N_solve=5` (typical per-step Newton iterations) with `solve_wall ≈ 0.1 × numeric_factor_wall` (triangular L/U solves are ~10× cheaper than factorization for KLU). All knobs are env-var-tunable in the aggregator (`WALL_REFACTOR_FREQ`, `WALL_N_SOLVE`, `WALL_SOLVE_FACTOR`, `WALL_SPGMR_BUDGET_FRACTION`).")
out_lines.append("")

# Per-case T-tables.
out_lines.append("## Per-case T-tables (all 4 ordering combos per case)")
out_lines.append("")
for case in CASE_ARR:
    case_rows = sorted(by_case[case], key=lambda r: int(r["nn"]))
    best = best_combo(case)
    best_nn = best["nn"] if best else None
    out_lines.append(f"### {case}")
    out_lines.append("")
    out_lines.append("| NN | ordering | btf | fill_ratio | numeric_wall_s | peak_rss_mb | verdict_class | note            |")
    out_lines.append("|----|----------|-----|------------|----------------|-------------|---------------|-----------------|")
    for r in case_rows:
        note = ""
        if r["nn"] == best_nn:
            note = "**best-combo**"
        elif r["verdict_class"] == "fill_overflow":
            note = "fill_overflow data point"
        elif r["verdict_class"] == "rss_overflow":
            note = "rss_overflow data point"
        elif r["verdict_class"] == "wall_overflow":
            note = "wall_overflow data point"
        out_lines.append(
            "| {nn} | {ordering:<8} | {btf}   | {fill:>10} | {wall:>14} | {rss:>11} | {vc:<13} | {note} |".format(
                nn=r["nn"],
                ordering=r["ordering"],
                btf=r["btf"],
                fill=fmt_or(r["fill_ratio"], "{:.4f}"),
                wall=fmt_or(r["numeric_wall_s"], "{:.6f}"),
                rss=fmt_or(r["peak_rss_mb"], "{:.2f}"),
                vc=r["verdict_class"],
                note=note,
            )
        )
    out_lines.append("")
    # 3-axis synthesis for this case (axis-specific diagnostics).
    out_lines.append("**3-axis synthesis**:")
    out_lines.append("")
    out_lines.append(f"- **Fill axis**: `{kv.get(f'{case}_KLU_fill_axis', '?')}` — {kv.get(f'{case}_KLU_fill_axis_diagnostic', '?')}")
    out_lines.append(f"- **RSS axis**: `{kv.get(f'{case}_KLU_rss_axis', '?')}` — {kv.get(f'{case}_KLU_rss_axis_diagnostic', '?')}")
    out_lines.append(f"- **Wall axis**: `{kv.get(f'{case}_KLU_wall_axis', '?')}` — {kv.get(f'{case}_KLU_wall_axis_diagnostic', '?')}")
    out_lines.append(f"- **Overall verdict**: `{kv.get(f'{case}_KLU_overall_verdict', '?')}`")
    out_lines.append(f"- **NO-GO axis** (when applicable): `{kv.get(f'{case}_KLU_NO_GO_axis', '?')}`")
    out_lines.append(f"- **Recommended action**: `{kv.get(f'{case}_recommended_action', '?')}`")
    out_lines.append("")

# Amended REQ-5 scenarios — explain index_overflow + wall_overflow markers.
out_lines.append("## Amended REQ-5 scenarios (PR-A landed)")
out_lines.append("")
out_lines.append("### Tool-bound data point (KLU 32-bit-int index overflow)")
out_lines.append("")
out_lines.append("Per spec REQ-5 Scenario \"Tool-bound data point (KLU 32-bit-int index overflow)\" (amended in PR-A):")
out_lines.append("")
out_lines.append("- **Trigger**: `klu_factor` returns `common.status == KLU_TOO_LARGE` (status code `-4` in SuiteSparse KLU; \"integer overflow has occurred\" per `klu.h`).")
out_lines.append("- **Tool behavior**: emit `KLU_INDEX_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=klu_factor_status_KLU_TOO_LARGE_int32_index_overflow` and exit 0 (NOT non-zero).")
out_lines.append("- **Aggregator classification**: `fill_overflow` data point — the 32-bit signed index space of `klu_factor` cannot hold `nnz(L+U)` for this ordering, i.e. equivalent to fill-axis hard-failing as a tool-bound limit. NOT a Slurm task failure.")
out_lines.append("- **Observed in PR-A**: NN=08 (heihe_x4 natural+BTF) and NN=12 (heihe_x16 natural+BTF). Both natural-ordering cells hit the int32 index cap; AMD/COLAMD orderings stay well within. The 64-bit-index `klu_l_*` API would be an implementation choice for P8-tune.E and is out of scope for this pattern-only spike.")
out_lines.append("")
out_lines.append("### Wall-budget data point (Slurm TIMEOUT)")
out_lines.append("")
out_lines.append("Per spec REQ-5 Scenario \"Wall-budget data point (Slurm TIMEOUT)\" (amended in PR-A):")
out_lines.append("")
out_lines.append("- **Trigger**: cell exceeds Slurm `--time` wall budget; `spike_array.sbatch` SIGTERM trap emits `KLU_WALL_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> elapsed_sec=<N> wall_budget_sec=<W>` to the cell log on a best-effort basis.")
out_lines.append("- **Tool behavior**: exit 0 with the marker (or sacct fallback when the trap window is insufficient).")
out_lines.append("- **Aggregator classification**: `wall_overflow` data point — PASS as Slurm task on marker, FAIL on wall axis. NOT a Slurm task failure.")
out_lines.append("- **Observed in PR-A**: NN=08 had an initial TIMEOUT on the 9794 sbatch (which had `--time=00:30:00` override; the natural-ordering factor extrapolates >6500s). After widening to 4h the re-run (job 9828) completed in 2h01m, but with `KLU_TOO_LARGE` (re-classified as fill_overflow). No clean wall_overflow marker fires in the final 16-cell evidence set; the wall axis verdict is derived per the formula above from the AMD best-combo `numeric_factor_wall_sec`.")
out_lines.append("")

# Machine-readable verdict block.
out_lines.append("## Machine-readable verdict block (canonical KV)")
out_lines.append("")
out_lines.append("```")
with open(KV) as f:
    out_lines.append(f.read().rstrip())
out_lines.append("```")
out_lines.append("")

# Raw data appendix.
out_lines.append("## Raw data appendix")
out_lines.append("")
out_lines.append("Full 16-cell aggregate TSV (`.review-evidence/p8tune-klu-spike-pr-b/aggregate.tsv`):")
out_lines.append("")
out_lines.append("```tsv")
with open(TSV) as f:
    out_lines.append(f.read().rstrip())
out_lines.append("```")
out_lines.append("")

# Cross-references.
out_lines.append("## Cross-references")
out_lines.append("")
out_lines.append("- [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md) — ADR-0005 (4-branch decision tree from this verdict)")
out_lines.append("- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR-0004 (SPGMR maxl Optional-knob; baseline wall anchor)")
out_lines.append("- [openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md) — capability spec (gitignored under `openspec/changes/`)")
out_lines.append("- [.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md](../../.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md) — PR-A 16-cell evidence narrative")
out_lines.append("- [.review-evidence/p8tune-klu-spike-pr-a/SPEC_AMENDMENTS.md](../../.review-evidence/p8tune-klu-spike-pr-a/SPEC_AMENDMENTS.md) — REQ-5 + REQ-7 amendments landed in PR-A")
out_lines.append("- [tools/p8tune.D/cn_node_ram.h](../../tools/p8tune.D/cn_node_ram.h) — pinned `CN_NODE_RAM_BYTES` constant (RSS axis denominator)")
out_lines.append("- [tools/p8tune.D/spgmr_baseline_walls.h](../../tools/p8tune.D/spgmr_baseline_walls.h) — pinned SPGMR per-step wall constant (wall axis numerator anchor)")
out_lines.append("")

with open(OUT, "w") as f:
    f.write("\n".join(out_lines))

print(f"OK: wrote {OUT}")
PYEOF
