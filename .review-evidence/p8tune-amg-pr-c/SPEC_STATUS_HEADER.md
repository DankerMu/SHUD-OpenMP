# Spec status header — PR-C mirror (P8-tune.F amg-pattern-spike-verdict)

This file mirrors the status header to be inserted into the change-local spec
at `openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md`
(gitignored under `openspec/changes/`). Per spec REQ-7 Scenario "OpenSpec
archive on capstone with mandatory Status header schema", the header is
carried forward by PR-D when the canonical archive
`openspec/specs/amg-pattern-spike-verdict/spec.md` is created.

PR-C consumes the strict verdict `NO-GO-both` from
`.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt` (per spec REQ-6
byte-identical contract with ADR-0007 §Decision). The operational reading
(amended `GO`) is recorded in ADR-0007 §Discussion and is the recommended
input for PR-D capstone anchor selection.

## Header text (verbatim insertion immediately under the spec H1)

```
> **Status**: Implemented via PR-0 #394 + PR-A #395 ([PR #402](https://github.com/DankerMu/SHUD-OpenMP/pull/402)) + PR-B #396 ([PR #403](https://github.com/DankerMu/SHUD-OpenMP/pull/403)) + PR-C #397 + PR-D #<TBD> (epic #393).
> **Verdict (as of 2026-06-29)**: **NO-GO-both** (strict 5-axis verdict; canonical anchor per spec REQ-6 byte-identical contract). All 4 cases fail Axis 4 `cycle_complexity < 1.5` because spike implementation hard-codes `cycle_complexity = 2 × operator_complexity` per PR-A H3 disclosure (not HYPRE telemetry). Amended verdict treating Axis 4 as non-discriminating: **GO** (all 4 cases all axes 1/2/3/5 PASS with substantial headroom; heihe_x16 best combo setup+apply = 0.116s vs SPGMR baseline 0.227s = 1.95× faster; AMG-vs-KLU wall ratio 0.04× at heihe_x16 demonstrates clear large-case win). See ADR-0007 §Discussion §"Axis 4 amendment per PR-A H3 disclosure" for full reconciliation.
> **Forward actions**: PR-D capstone author to evaluate strict-vs-amended divergence and select master plan §P8-tune.G anchor. Recommended per ADR-0007 §Forward action: adopt amended `GO` → anchor §P8-tune.G full AMG + A5 integration as `[OPEN, HIGH priority]` (4-6 weeks per spec REQ-7 GO clause) with Axis 4 instrumentation work item (integrate `HYPRE_BoomerAMGGetCycle*` telemetry; re-open this ADR if measured drift > 5%).
> **Authoritative ADR**: [docs/adr/0007-amg-spike-decision.md](docs/adr/0007-amg-spike-decision.md) (Status: Proposed at PR-C; PR-D will flip to Accepted).
> **Authoritative verdict doc**: [docs/p8tune/amg_spike_verdict.md](docs/p8tune/amg_spike_verdict.md).
```

## PR-D carry-forward instruction (per spec REQ-7)

When PR-D executes the OpenSpec archive operation:

```
openspec archive p8tune-amg-spike -y
```

(which moves `openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md`
→ `openspec/specs/amg-pattern-spike-verdict/spec.md`), the above 5-line Status
header MUST be preserved in the canonical archive AND the PR-D field
`PR-D #<TBD>` MUST be backfilled with the actual PR-D PR number.

The PR-C edit to the change-local spec is the source of truth; this mirror
exists only because the change-local copy is gitignored and cannot be
observed by reviewers of PR-C itself.

## Pinned summary for PR-D consumption

| field | value | source |
|-------|-------|--------|
| epic | #393 | issue label `p8-tune.F` |
| spec | `amg-pattern-spike-verdict` | `openspec/changes/p8tune-amg-spike/specs/` |
| ADR | 0007 (Proposed → Accepted at PR-D) | `docs/adr/0007-amg-spike-decision.md` |
| verdict doc | `docs/p8tune/amg_spike_verdict.md` | this PR-C |
| strict verdict | `NO-GO-both` (canonical anchor) | `aggregate_verdict.txt verdict_branch=` |
| amended verdict | `GO` (FYI, operationally meaningful) | `aggregate_verdict.txt verdict_branch_axis4_amended=` |
| cells PASS | 16/16 | `.review-evidence/p8tune-amg-pr-b/cells/cell-{0..15}.out` |
| best combo keliya | (6, 21) NN=02 | aggregate_verdict.txt §CASE_VERDICT_BEGIN:keliya |
| best combo heihe | (8, 8) NN=07 | aggregate_verdict.txt §CASE_VERDICT_BEGIN:heihe |
| best combo heihe_x4 | (6, 8) NN=08 | aggregate_verdict.txt §CASE_VERDICT_BEGIN:heihe_x4 |
| best combo heihe_x16 | (6, 8) NN=12 | aggregate_verdict.txt §CASE_VERDICT_BEGIN:heihe_x16 |
| heihe_x16 setup+apply | 0.116 s | aggregate.tsv NN=12 |
| SPGMR baseline | 0.226579 s/step | `tools/p8tune.D/spgmr_baseline_walls.h` |
| AMG-vs-SPGMR ratio heihe_x16 | 0.51× (1.95× faster) | derived |
| AMG-vs-KLU ratio heihe_x16 | 0.04× (24.5× faster) | derived vs ADR-0005 KLU per-step est |
| Hypre version | 3.1.0 | cell-0.out cell_summary block |
| ColPack version | unknown (PR-B M2 sentinel) | cell-0.out cell_summary block |
| SHUD pin | 1ab61c023ac2b93a178c2feb07aa3df509fe1a96 | cell-0.out cell_summary block |

## Cross-PR cross-references (for PR-D audit)

- **PR-0 #394** — SHUD `Model_Data` dtor UB fix (#386 closure, hard prereq per ADR-0005 F3 retrospective). Required for both P8-tune.F + future P8-tune.G integration to avoid recurrence.
- **PR-A #395 (PR #402)** — `tools/p8tune.F/boomeramg_setup_solve.cpp` spike binary + Makefile carve-out reusing `dump_adjacency` + `fd_color_jacobian` from P8-tune.D via symlink (spec REQ-2 shell-out reuse). H3 disclosure recorded the Axis 4 `cycle_complexity = 2 × operator_complexity` hard-coded estimate limitation.
- **PR-B #396 (PR #403)** — 16-cell Slurm array sweep (job 9896). All 16 cells `verdict_class=PASS`. M1: H3 disclosure carried; M2: `colpack_version=unknown` sentinel accepted (ColPack version not captured at PR-A merge time); M3: NA timing sentinel acceptance for `AMG_WALL_OVERFLOW` cells (unused in this all-PASS sweep but parser accepts).
- **PR-C #397** (本 PR) — aggregator + ADR-0007 + verdict.md + this header.
- **PR-D #<TBD>** — epic capstone: master plan §P8-tune.F close + §P8-tune.G + (conditional) §P8-tune.H anchor per amended verdict + OpenSpec archive + ADR-0007 Status flip Proposed → Accepted + review-loop log entries (5 JSONL per spec REQ-8).

## Provenance

- Authored: 2026-06-29 in PR-C (epic #393, issue #397)
- Source: `tools/p8tune.F/aggregate_amg_spike.sh` → `.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt` machine-readable verdict block
- ADR cross-reference: `docs/adr/0007-amg-spike-decision.md` (§Status: Proposed; flipped to Accepted by PR-D)
- Verdict doc cross-reference: `docs/p8tune/amg_spike_verdict.md`
