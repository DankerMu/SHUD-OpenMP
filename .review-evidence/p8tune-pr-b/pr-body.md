## Summary

- Adds `§verdict` section to `docs/p8tune/clean_prec_none_baseline.md` (between `§mode-C-tune-reference` and `§References`).
- Applies spec `maxl-sweep-verdict` Requirement "Sweep entry condition" scenario "Full sweep GO triggered by hard evidence" L11-15.
- **Decision**: FULL 60-cell sweep GO. Both heihe (`ncfl=85`) and heihe_x4 (`ncfl=3620`) satisfy `ncfl > 0` predicate per Step 1 PR-B verdict `n8_profile_verdict.md` §3.1.
- Decision-tree closure verified: input state matches "Full sweep GO" branch unambiguously; probe-only fallback + 2 NO-GO branches NOT exercised.
- Downstream PR-D contract: 60-cell matrix (`maxl ∈ {5, 10, 15, 20, 30}` × `case ∈ {heihe, heihe_x4}` × `N ∈ {1, 8}` × `rep ∈ {1, 2, 3}`), Slurm 三铁律 paths, PR-C env-var hook dependency.

## Why

PR-A baseline doc established the `§decision-input-table` (raw data: both cases `ncfl > 0`, heihe_x4 `nli/nni ≈ 4.527` saturation). PR-B applies the decision tree from spec `maxl-sweep-verdict` to that data and emits the explicit "full sweep GO" verdict — closing the sweep-mode adjudication before PR-D runs the actual 60-cell sweep.

This is doc-only verdict adjudication; no source / tool / server compute change. PR-D inherits this verdict to drive the sweep matrix; PR-E ADR-0004 will adjudicate the GO/NO-GO/Optional-knob/Diagnostic outcome separately on PR-D's sweep results.

OpenSpec change: `p8tune-spgmr-maxl` (capability `maxl-sweep-verdict`).

## Scope

1 file, +48 insertions, 0 deletions:
- `docs/p8tune/clean_prec_none_baseline.md` (new `§verdict` section)

No `.c`/`.cpp`/`.h`/`Makefile`/`.sh`/`.py` changes. No SHUD submodule pointer change (still `37be0fe`). No tool change. No new doc file.

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] `git diff --name-only` → 1 doc file (`docs/p8tune/clean_prec_none_baseline.md`); no source/test/SHUD pointer
- [x] §verdict decision-input table cross-references PR-A `§decision-input-table` (L386-398 of same doc, both rows `ncfl > 0`)
- [x] §verdict branch adjudication enumerates all 4 spec decision-tree scenarios (Full sweep GO + probe-only + NO-GO no-entry + residual fallback) with explicit MATCH / NOT MATCH per current baseline state
- [x] §verdict downstream PR-D contract cites Slurm 三铁律 + 60-cell matrix dimensions + PR-C dependency

## Agent Review

(Populated by Phase 8.)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #365.
