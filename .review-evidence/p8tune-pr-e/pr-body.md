## Summary

**Epic #362 capstone.** Closes P8-tune.C epic (6-PR sequence: PR-0 #369 ✅ + PR-A #370 ✅ + PR-B #371 ✅ + PR-C #372 ✅ + PR-D #373 ✅ + PR-E (本 PR)).

- **4 new files** (911 lines total):
  - `tools/p8tune/aggregate_maxl_sweep.sh` (385 lines bash + inline python3) — 60-cell PR-D data parser → 8 T-tables + flat KV + synthesis
  - `tools/p8tune/render_verdict.sh` (91 lines bash) — Mac+server doc renderer
  - `docs/p8tune/maxl_sweep_verdict.md` (176 lines) — 8 detailed T-tables + production tune guidance
  - `docs/adr/0004-maxl-sweep-decision.md` (259 lines) — ADR Optional-knob branch + case-size-asymmetric Krylov pattern + forward action
- **ADR-0004 decision: Optional-knob branch**. `SHUD_SPGMR_MAXL` env-var stays as long-lived production opt-in. Default unchanged.
- **Per-case best-maxl recommendation** (3-rep median, Slurm 9690 60-cell data):
  - heihe N=1 → `SHUD_SPGMR_MAXL=30` (+11.99% wall, GO band, ncfl 85→0)
  - heihe N=8 → `SHUD_SPGMR_MAXL=30` (+6.78% wall, Optional band)
  - heihe_x4 N=1 → unset (default=5); all maxl ≥10 REGRESS −6.86% to −15.83%
  - heihe_x4 N=8 → unset (default=5); all maxl ≥10 REGRESS −15.81% to −24.82%
- **8-gate verdict**: G1/G2/G3/G4/G6/G8 PASS strict; G5 MIXED case-asymmetric; G7 STRICT FAIL is expected numerical (maxl bump → Krylov expansion → CVODE step-size response → trajectory drift)
- **Key empirical finding**: case-size-asymmetric Krylov memory bandwidth pattern. heihe (~6300 elem) MGS working set fits L2 → bigger maxl wins. heihe_x4 (~40046 elem) overflows L2 → DRAM-bound MGS → wall regression dominates ncfl elimination

## Why

Per ADR-0003 P8-tune.C epic Path 4 trigger: PREC_NONE remains production solver but SUNDIALS-default maxl=5 yields high ncfl on NWM cases (heihe 85, heihe_x4 3620), suggesting Krylov subspace might be undersized. PR-E aggregates the 60-cell PR-D sweep into the formal 4-branch outcome (GO+default-bump / Optional-knob / Diagnostic / NO-GO) per spec D8.

PR-E's job: G1-G8 gate adjudication + ADR-0004 authoring + production tune guidance.

OpenSpec change: `p8tune-spgmr-maxl` (capability `maxl-sweep-verdict` complete; aggregator + verdict + ADR).

## Scope

4 files (+911 lines, 0 deletions):
- `tools/p8tune/aggregate_maxl_sweep.sh` (NEW, 385 lines, executable)
- `tools/p8tune/render_verdict.sh` (NEW, 91 lines, executable)
- `docs/p8tune/maxl_sweep_verdict.md` (NEW, 176 lines, render output)
- `docs/adr/0004-maxl-sweep-decision.md` (NEW, 259 lines)

No `.c/.cpp/.h` changes. No Makefile. No SHUD submodule pointer change (still `6ce17d6`). No `cvode_config.cpp:259` default constant change (Optional-knob does NOT trigger default-bump).

Server-resident artifacts (NOT in PR):
- `/scratch/.../maxl_sweep/aggregate_verdict.txt` (flat KV mirror)
- `/scratch/.../maxl_sweep/T{1..8}_*.md` (8 T-tables, source for render)
- `/scratch/.../maxl_sweep/verdict_synthesis.md` (synthesis block, embedded in verdict.md)

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] Mac `bash -n tools/p8tune/aggregate_maxl_sweep.sh` → 0 syntax errors
- [x] Mac `bash -n tools/p8tune/render_verdict.sh` → 0 syntax errors
- [x] Server-side aggregator dry-run (SWEEP_ROOT=`/scratch/.../maxl_sweep`) → produces 8 T-tables + aggregate_verdict.txt + verdict_synthesis.md; emits `ADR-0004 branch: Optional-knob`
- [x] Server-side render dry-run (OUT=`docs/p8tune/maxl_sweep_verdict.md`) → composes single verdict doc with YAML frontmatter + synthesis + 8 T-tables
- [x] Verdict doc cross-references all source: PR-A/B/C/D PR numbers + commit SHAs + Slurm job 9690 + spec D8 4-branch + production tune guidance table
- [x] ADR-0004 cites ADR-0002 Path 4 trigger + ADR-0003 fallback context + per-gate quantified evidence + decision rationale + 4-branch decision-tree closure
- [x] Per-case recommendation table data-derived from 3-rep median wall (not first-rep noise); programmatic generation in aggregator
- [x] G7 STRICT FAIL interpretation documented with mechanism (Arnoldi MGS → CVODE step-size adapter response), explicitly differentiated from "corruption"

## Agent Review

(Populated by Phase 8.)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #368. Epic #362 capstone.
