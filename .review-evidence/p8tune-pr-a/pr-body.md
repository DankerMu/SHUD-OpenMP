## Summary

- **Plan A path** (preferred, ~zero compute reuse): SHUD `7a1dc8f..37be0fe` codepath equivalence verified = revert-of-PR-D only. Step 1 PR-B verdict `n8_profile_verdict.md` §3.1 18-cell table reused verbatim as cleaned-PREC_NONE baseline; no fresh server compute needed for the §3.1 reproduction.
- **tools/p8tune/aggregate_clean_baseline.sh**: emits 5 markdown tables (T1 raw 18-cell + T2 cross-N invariance + T3 ROI ratio + T4 solver-failure floors + T5 decision-input) from §3.1; bash strict mode; `--plan-a` / `--plan-b` CLI.
- **docs/p8tune/clean_prec_none_baseline.md**: anchors the mode-C-tune reference set with 9 §sections (codepath-equivalence + submit-template-provenance + raw-18-cell + cross-N invariance + ROI ratio + solver-failure + keliya-smoke-anchor + decision-input + mode-C-tune-reference).
- **keliya cleaned-PREC_NONE smoke anchor** (cn14 SHUD `37be0fe` build `SHUD_ENABLE_PROFILE=1`): `rivqdown.dat` SHA12 = `1bfe6a30856e`; 15-key cvode_stats snapshot; PR-C G3 4-way CI gate contract: `unset` / `""` / `"0"` / `"5"` invocations MUST produce bit-identical output.
- **Decision-input verdict**: hard-evidence trigger `ncfl > 0 per cell` ALREADY SATISFIED — heihe `ncfl=85` + heihe_x4 `ncfl=3620` per §3.1 → maxl-sweep-verdict capability enters full 60-cell sweep mode (no probe-only fallback needed).
- **Bundled PR-0 deferred fixes**:
  - `docs/p8pre/capstone.md` §5.1 table ratio cell `4.527` → `4.526` (3 rows) + cite to `n8_profile_verdict.md` §3.4 derivation (per PR-0 review-correctness deferred note).
  - `openspec/changes/p8tune-spgmr-maxl/proposal.md` L21 "3-way" → "4-way" drift fix (gitignored, local-only refinement per PR-0 fixture-review informational note).

## Why

PR-A anchors the cleaned-PREC_NONE baseline so the maxl sweep (PR-D) has a stable reference set for G4 (solver-work) / G6 (no-solver-regression) / G7 (hydrology-A4) gate comparisons. PR-0 already corrected the future-candidate gate citations from `ncfn < 6/47` (Step 2 PREC_LEFT+identity floors, retained only as negative-control anchor) to `≤ 7 (heihe) ∧ ≤ 51 (heihe_x4)` (production PREC_NONE floors per §3.1). PR-A formalizes those §3.1 numbers into a citable baseline doc + tool + server smoke artifact.

Plan A path saves ~5.5h server compute by reusing Step 1 PR-B aggregator output — provable because SHUD `7a1dc8f` and `37be0fe` CVODE codepath is bit-identical (cleanup tail reverts only PR-D's PREC_LEFT change + identity preconditioner deletion). Plan B (18-cell server re-run) remains the spec-defined fallback if codepath diverges; current codepath check shows revert-of-PR-D only → Plan A confirmed.

OpenSpec change: `p8tune-spgmr-maxl` (capability `clean-prec-none-baseline`).

## Scope

3 files (873 insertions, 3 deletions):
- `tools/p8tune/aggregate_clean_baseline.sh` (NEW, 434 lines, executable)
- `docs/p8tune/clean_prec_none_baseline.md` (NEW, 429 lines, 9 sections)
- `docs/p8pre/capstone.md` (§5.1 3-cell ratio fix, 13 insertions + 3 deletions)

No `.c`/`.cpp`/`.h`/`Makefile` changes. No SHUD submodule pointer change (still `37be0fe`). No public API surface change.

Local-only (gitignored, not in PR diff):
- `openspec/changes/p8tune-spgmr-maxl/proposal.md` L21 4-way fix
- `.review-evidence/p8tune-pr-a/codepath_equivalence_diff.txt`
- `.review-evidence/p8tune-pr-a/keliya_smoke_artifact.txt`
- `.review-evidence/p8tune-pr-a/keliya_smoke_job.out`

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] `bash tools/p8tune/aggregate_clean_baseline.sh --plan-a` → exit 0, 5 tables emitted
- [x] `bash tools/p8tune/aggregate_clean_baseline.sh --plan-b` (stub) → exit 1 on missing scratch dir (expected guard)
- [x] Plan A codepath equivalence: `git diff 7a1dc8f..37be0fe -- src/Equations/cvode_config.cpp src/Equations/*precond* src/Equations/*spgmr*` inside SHUD = revert-of-PR-D only
- [x] keliya smoke server-side: Slurm job `9620` on cn14, ExitCode `0:0`, wall `00:01:47` (88s including build + run)
- [x] cross-N invariance N=1/4/8: 30/30 cells bit-identical per §3.1 (B1a S4 OMP-neutrality regression detector)
- [x] PR-0 deferred fix `4.527 → 4.526`: capstone.md §5.1 L161-163 = `4.526`
- [x] PR-0 deferred fix proposal.md L21 → 4-way: applied (gitignored)
- [x] Scope check: 3 doc/tool files; no source/Makefile; no SHUD pointer change

## Agent Review

(Populated by Phase 8 after the cross-review + verifier loop completes on the final HEAD.)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #364.
