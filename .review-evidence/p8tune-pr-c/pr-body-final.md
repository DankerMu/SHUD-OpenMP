## Summary

- **SHUD source change**: Adds `SHUD_SPGMR_MAXL` env-var hook in `SHUD/src/Equations/cvode_config.cpp` (helper `get_spgmr_maxl_from_env()` + modified `SUNLinSol_SPGMR` call-site at L324 + provenance log).
- **Single-file source surface**: `src/Equations/cvode_config.cpp` only (+67/-2); no Makefile / SUNDIALS-vendored / header / sibling source change. SHUD submodule commit `6ce17d6` on `openmp-baseline` (NEVER master per project rule).
- **4 safety constraints**:
  - unset / "" / "0" → return 0 (silent default; bit-identical to SHUD 37be0fe production)
  - "5" / "10" / "15" / "20" / "30" → parse + emit `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` log + return
  - any other value (e.g. "7", "50", "foo", "-1", "+5", "05", "5x", "10.0") → stderr error + `myexit(ERRCVODE)` BEFORE any SPGMR allocation
- **G3 4-way bit-identical CI gate VERIFIED on cn14** (Slurm 9626, SHUD 6ce17d6 build SHUD_ENABLE_PROFILE=1):
  - All 4 invocations rivqdown.dat SHA12 = `1bfe6a30856e` matches PR-A anchor exactly
  - All 15 canonical cvode_stats keys bit-identical to PR-A anchor
  - Cross-run cmp: run1 == run2 == run3 == run4 byte-identical
  - Stdout provenance log discipline: 0/0/0/1 lines per IM D15 L235-238
- **PREC_NONE preserved**: `grep -nE 'PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity'` returns 0 matches
- **Mac build**: `make shud` + `make shud_omp` both exit 0
- **Implementer parser unit tests**: 19/19 PASS including strict-whitelist edge cases

## Why

PR-A established cleaned-PREC_NONE baseline + keliya smoke anchor. PR-B verdict gated full 60-cell sweep. PR-C provides the runtime knob (env var `SHUD_SPGMR_MAXL`) that PR-D needs to iterate over `maxl ∈ {5, 10, 15, 20, 30}` per cell, while guaranteeing default-unset bit-identical equivalence to SHUD 37be0fe (so 18-cell baseline reuse from PR-A remains valid).

The 4-way default-equivalence (`unset` / `""` / `"0"` / `"5"` all bit-identical) honors SUNDIALS docs (`maxl ≤ 0` → default 5) and IM D15 invariant.

OpenSpec change: `p8tune-spgmr-maxl` (capability `spgmr-maxl-env-hook`).

## Scope

Outer repo PR (1 file):
- `SHUD` submodule pointer: `37be0fe` → `6ce17d6` (1 line)

SHUD submodule commit `6ce17d6` (1 file):
- `SHUD/src/Equations/cvode_config.cpp` (+67 / -2 lines): `<errno.h>` include + helper L248-308 + call-site L324

No Makefile / header / SUNDIALS-vendored / sibling source change. No B1a/B1b baseline modification.

Local-only (gitignored, not in PR diff):
- `openspec/changes/p8tune-spgmr-maxl/design.md` D15 Invariant Matrix authored
- `.review-evidence/p8tune-pr-c/{g3_verdict.md, g3_4way_evidence.tar, round1-*.md, comment-*.md, pr-body-final.md}`

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] Mac `make shud` + `make shud_omp` → both exit 0
- [x] Mac grep `PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity` → 0 matches
- [x] Single-file source surface within SHUD → exactly `src/Equations/cvode_config.cpp`
- [x] Implementer parser unit tests 19/19 PASS (strict whitelist + leading-zero rejector + char-by-char `[0-9]` pre-strtol check)
- [x] cn14 server build with `SHUD_ENABLE_PROFILE=1` → exit 0
- [x] cn14 Slurm job 9626 (CPU partition, cn14, /scratch/.p8tune-runs/pr-c-g3-gate/) — Slurm 三铁律 compliance
- [x] **G3 4-way bit-identical gate**: all 4 invocations SHA12 = `1bfe6a30856e` matches PR-A anchor
- [x] G3 15-key cvode_stats bit-identical to PR-A anchor (all 15 keys match)
- [x] G3 cross-run cmp byte-equivalence: run1 == run2 == run3 == run4
- [x] G3 stdout provenance log discipline: 0/0/0/1 lines per IM D15 L235-238
- [x] Phase 0.5 fixture review (IM D15 validation): APPROVE 7/7
- [x] Phase 4 round 1: 4 reviewers parallel @ `768c905` (review-correctness + review-spec-compliance + review-integration + review-security-perf) → all CLEAN 0 findings + 10 non-blocking notes
- [x] Phase 4.5 verifier gate → empty verdict (0 candidates)
- [x] Phase 7 final review (Gap Sweep) @ `768c905` → CLEAN, APPROVE merge gate
- [x] CI 5/5 PASS @ `768c905`
- [x] SHUD master branch isolation verified: `git branch -r --contains 6ce17d6` = openmp-baseline ONLY

## Agent Review

- **Reviewer agents used**: `reviewer` (Phase 0.5 fixture review — IM D15 validation); `reviewer` × 4 (Round 1 Phase 4 parallel: review-correctness + review-spec-compliance + review-integration + review-security-perf @ `768c905`); `reviewer` (Phase 7 Gap Sweep independent-final @ `768c905`)
- **Reviewed outer HEAD SHA**: `768c905f8f078e7ece27bc4d8e4efb4ab0a1b825`
- **SHUD pin under review**: `6ce17d6`
- **Review evidence**: consolidated bundle comment posted below
- **OpenSpec change**: `p8tune-spgmr-maxl`; fixture level: expanded (PR-C high-intensity); selected risk packs: Config/project setup + Schema/columns/units (CVODE 15-key contract) + Legacy compat (default-unset bit-identical) + Error handling (myexit fail-fast) + Bitwise reproducibility (G3 4-way) + Server/local data partition (cn14 build pin) + Invariant/state (PREC_NONE preservation)
- **Phase 4.5 verifier verdict**: empty (Round 1 4 reviewers all reported 0 actionable findings; nothing to verify); persisted at `.review-evidence/p8tune-pr-c/round1-phase45-verifier.md`
- **Key findings addressed**: 0 actionable findings. 10 non-blocking notes from Round 1 — all informational/disclosure (1 deferred cosmetic L259 comment ref in SHUD source; rest are positive observations on defense-in-depth + honest sbatch ExitCode disclosure)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #366.
