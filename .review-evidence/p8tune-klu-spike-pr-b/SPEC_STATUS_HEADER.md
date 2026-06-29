# Spec status header — PR-B mirror

This file mirrors the status header inserted into the change-local spec
at `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md`
(gitignored under `openspec/changes/`). Per tasks §3.8, the marker is
carried forward by PR-C task 4.7 when the canonical archive
`openspec/specs/klu-pattern-spike-verdict/spec.md` is created.

## Header text (verbatim insertion immediately under the `# Spec` H1)

```
> **Status**: Implemented via PR-A + PR-B; verdict = Case-aware (per ADR-0005, `docs/adr/0005-klu-spike-decision.md`). Small cases (keliya, heihe) overall=GO; heihe_x4 overall=Optional (wall margin 1.87×); heihe_x16 overall=NO-GO (wall margin 17.9×). Forward action split into P8-tune.E.small-only (KLU env-var opt-in for small cases) + P8-tune.F (BoomerAMG/Hypre spike for large cases).
```

## PR-C carry-forward instruction (per tasks §4.7)

When PR-C executes the OpenSpec archive operation (move
`openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md`
→ `openspec/specs/klu-pattern-spike-verdict/spec.md`), the above status
header MUST be preserved in the canonical archive. The PR-B edit to the
change-local spec is the source of truth; this mirror exists only because
the change-local copy is gitignored and cannot be observed by reviewers
of PR-B itself.

## Provenance

- Authored: 2026-06-28 in PR-B (epic #379, issue #382)
- Source: `tools/p8tune.D/aggregate_klu_spike.sh` → `.review-evidence/p8tune-klu-spike-pr-b/aggregate_verdict.txt` machine-readable verdict block
- ADR cross-reference: `docs/adr/0005-klu-spike-decision.md` (§Status: Proposed; flipped to Accepted in PR-C task 4.6)
