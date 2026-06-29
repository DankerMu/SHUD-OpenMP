# Round 3 Cross-Review — Integration (regression-only audit)

Repo: `/Users/danker/Desktop/Hydro-SHUD/openMP`
PR: #384 (feat/issue-380-p8tune-klu-spike-pr-0)
Round 2 head: `50d2a4b`
Round 3 head: `7fc325b` (single commit: "fix(p8tune.D): address PR-0 round 2 findings G1-G4 (#380)")
Mode: post round-2-fix regression check — confirm G1–G4 outer fix did not regress build / submodule / file-layout invariants.
Round 2 status carried in: **APPROVE** with no new findings (round 1 → round 2 was net-clean).

## Summary

No regression detected. Round-2 fix touches only the outer source/doc/evidence triplet; submodule pointer, top-level Makefile, and `tools/p8tune.D/Makefile` are byte-identical to `50d2a4b`. Clean rebuild succeeds end-to-end. Openspec fixture validates `--strict`. Working-tree-only fixture amendments (tasks.md / design.md) are confirmed under `.gitignore` per project convention, so absence-from-PR is intended, not an oversight.

## Verification matrix

| Task | Check | Result |
|------|-------|--------|
| T1 | `make clean_spike && make shud_spike` | **PASS** — all 3 spike binaries (`dump_adjacency`, `fd_color_jacobian`, `klu_analyze_factor`) re-link clean; only pre-existing `sprintf` deprecation warning in `SHUD/src/Equations/functions.hpp:67` (unrelated; lives in SHUD pinned source). |
| T2 | `git diff 50d2a4b..7fc325b -- SHUD` | **PASS** — empty diff; SHUD pointer `41d9a17` (v1.0-263-g41d9a17) unchanged. Fix is outer-only as intended. |
| T3 | Changed-files envelope | **PASS** — exactly 3 files: `tools/p8tune.D/klu_analyze_factor.cpp`, `tools/p8tune.D/README.md`, `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log`. All within spec REQ-7 envelope. No stray edits to spec/, docs/, or other tools. |
| T4 | `openspec validate p8tune-klu-spike --strict --no-interactive` | **PASS** — "Change 'p8tune-klu-spike' is valid". Working-tree tasks.md / design.md amendments (per implementer note) are gitignored: `git check-ignore -v` confirms both paths matched by `.gitignore:13` (`openspec/changes/`). This is the documented project convention (status_matrix authoritative, openspec/changes scaffolding is local-only), so non-tracking is intended and does not break validation. |
| T5 | New evidence file location + git-clean | **PASS** — `mac_smoke_keliya_klu_ordering_matrix.log` lives at `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log` (the canonical PR-0 evidence dir), tracked at HEAD `7fc325b`. Status clean (only untracked items are sibling `round1/` and `round2/` review-output dirs unrelated to this PR). |
| T6 | Top-level + spike Makefile untouched | **PASS** — `git diff 50d2a4b..7fc325b -- Makefile tools/p8tune.D/Makefile` returns empty. |

## Spot-check on the actual G1 fix

Confirmed `tools/p8tune.D/klu_analyze_factor.cpp:273` now gates the `est_after_analyze_bytes` preflight on `ordering_id == 0 && Symbolic->lnz > 0 && Symbolic->unz > 0`, with a documented `else` branch emitting `PREFLIGHT_AFTER_ANALYZE skipped (... AMD-only); relying on post-factor RSS check`. The new evidence log shows the three orderings behave as fixed:
- `amd btf=1`: preflight runs, `symbolic_lnz+unz=33088`, est_bytes=1191168, well under budget; numeric factor PASS.
- `colamd btf=1`: preflight **skipped** (lnz=-1 unz=-1); numeric factor proceeds, fill_ratio=4.64, peak_rss=3.5 MB.
- `natural btf=0`: preflight skipped; numeric factor proceeds, fill_ratio=205.95 (expected — natural ordering is catastrophic on this graph), peak_rss=35.7 MB (still << CN_NODE_RAM).

The undefined-behavior cast `static_cast<size_t>(-2.0)` that produced the round-2 spurious OOMs and the corresponding spurious PASS verdicts on the 8/16 non-AMD PR-A cells is genuinely eliminated. README OOM-as-data-point reason enum is correctly updated from `preflight_estimate` to `preflight_after_analyze` so docs match emitted strings.

## Praise

- The fix is **minimal and surgical** — three small hunks (cpp guard + cpp log-format tightening + README enum tweak), no scope creep into adjacent code. Exactly the right shape for a round-2 fix.
- Inline comment at lines 264–272 cites both the upstream root cause (KLU 7.12.2 contract on `Symbolic->lnz/unz`) and the prior-round provenance (`PR-0 reviewer round 2 finding e01 / G1 fix`). Future readers won't have to re-derive this.

## Verdict

**APPROVE** — no new findings, no regressions. Round-3 commit is a clean realization of the round-2 fix plan; integration invariants (build, submodule pointer, Makefile, fixture validation, evidence layout) all hold.
