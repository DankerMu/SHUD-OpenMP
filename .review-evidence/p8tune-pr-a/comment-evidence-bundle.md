# Cross-Review Evidence Bundle — PR #370

**Reviewed head SHA**: `c6db15f9510c68cb8854f931f821a4faa654a655` (Round 1 ran at `3494c6ac`; orchestrator applied 1-line doc typo fix at `c6db15f` per round 1 non-blocking notes; Phase 7 re-audited on `c6db15f`)
**OpenSpec change**: `p8tune-spgmr-maxl` (capability `clean-prec-none-baseline`)
**Fixture level**: expanded / compact-doc-tool (PR-A scope)
**Repair intensity**: low (Plan A path)

## Phase 4 Round 1 — Risk-adaptive cross-review (3 parallel reviewers)

### Reviewer agent: `review-correctness`
- **Reviewed head SHA**: `3494c6ac1435c1616cbddc8c5058f5c715f0d14e`
- **Summary**: CLEAN — 0 critical, 0 warning; 10/10 correctness checks pass
- **Findings**: None.
- **Non-blocking notes** (2):
  - N1: `docs/p8tune/clean_prec_none_baseline.md:303` path typo `./build/shud` → `./shud` (binary actually at `./shud` per SHUD Makefile output). **Addressed at `c6db15f`.**
  - N2: Slurm sacct triplet (JobID/ExitCode/Elapsed) in doc not in `.review-evidence/keliya_smoke_job.out`. Triplet is in A2 implementer report; informational only.
- **Hard-evidence verified**:
  - SHUD diff `7a1dc8f..37be0fe` byte-matches embedded version (revert-of-PR-D only confirmed)
  - Aggregator both modes: `--plan-a` exit 0 + 5 tables (127 lines); `--plan-b` exit 1 with explanatory gate
  - Capstone §5.1 ratio derivation: `30509/6741 = 4.5259 → 4.526` confirmed authoritative
  - 15-key smoke snapshot byte-identical across 3 sources (artifact + job log + doc)

### Reviewer agent: `review-spec-compliance`
- **Reviewed head SHA**: `3494c6ac1435c1616cbddc8c5058f5c715f0d14e`
- **Summary**: CLEAN — all 8 review dimensions pass
- **Per-task DONE/MISSING**:
  - §2.1 Plan A §3.1 extraction: DONE
  - §2.2 Plan A ssh template check: DONE (missing → derivation provenance documented)
  - §2.3 Plan A SHUD diff: DONE (revert-of-PR-D only confirmed)
  - §2.4 Plan A aggregator script: DONE
  - §2.5 keliya smoke on cn14/cn15: DONE (Slurm job 9620 ExitCode 0:0)
  - §2.6 Plan B fallback submit template: SKIPPED (Plan A passed)
  - §2.7 Plan B fallback 18-cell run: SKIPPED (Plan A passed)
  - §2.8 baseline doc with 5 tables: DONE
  - §2.9 cross-N invariance verification: DONE (30/30 cells PASS per §3.1)
  - §2.10 cite SHUD pin equivalence as mode-C-tune reference: DONE
- **Findings**: None.
- **Non-blocking notes**: 3 informational (T1 rep triplication spec-compliant; rivqdown SHA12 prose annotation Plan A reuse stance; PR-0 deferred fix bundle disclosed)

### Reviewer agent: `review-integration`
- **Reviewed head SHA**: `3494c6ac1435c1616cbddc8c5058f5c715f0d14e`
- **Summary**: APPROVE — all 10 cross-PR integration contracts hold
- **Findings**: None.
- **Non-blocking notes** (2):
  - N1: §keliya-smoke-anchor needs "serial-only smoke; OMP build via `make shud_omp` validated separately by PR-C G1" clarification. **Addressed at `c6db15f`.**
  - N2: T1 rep triplication semantics — PR-E aggregator joins must dedupe on (case, N) before any rep-level statistics. Deferred to PR-E informational.

## Phase 4.5 — Independent finding-verification gate

| Candidate # | Source reviewer | Verdict |
|---|---|---|
| (none) | — | — (no candidates to verify) |

All 3 round 1 reviewers reported zero actionable findings per `finding-contract.md`. No verifier subagent spawned. Verdict table persisted at `.review-evidence/p8tune-pr-a/round1-phase45-verifier.md`.

## Post-round-1 CI-only-equivalent doc fix

Orchestrator applied 1-line patch at `c6db15f` (was `3494c6ac`):
- `docs/p8tune/clean_prec_none_baseline.md:303` — `./build/shud keliya` → `./shud keliya` + clarification "serial-only smoke; OMP build via `make shud_omp` validated separately by PR-C G1"

Per phase-flow Phase 8 ci-only classification: log/evidence plumbing analogue that does not alter runtime/product behavior, public contracts, production config, security boundaries, or test semantics. Does NOT increment cross-review round counter. Phase 7 Gap Sweep ran on the new SHA `c6db15f` as the canonical final-head check.

## Phase 7 — Independent final review (Gap Sweep)

- **Reviewer agent**: independent-final (clean-context)
- **Reviewed head SHA**: `c6db15f9510c68cb8854f931f821a4faa654a655`
- **Verdict**: CLEAN — Reject-When precision applied; no real defect surfaced beyond round 1 + c6db15f fix
- **Independent re-verification**:
  - `git diff 3494c6a..c6db15f` confirms surgical 1-line patch (zero collateral)
  - Baseline doc preserved at 429 lines with all 9 sections at original byte offsets
  - Aggregator `--plan-a` re-run exit 0, all 5 tables emit
  - `openspec validate p8tune-spgmr-maxl --strict` PASS
  - SHUD submodule pointer `37be0fe` unchanged
  - rivqdown SHA12 = `1bfe6a30856e` + 15-key snapshot byte-identical
  - Capstone §5.1 still reads `30509 | 4.526` with HTML provenance comment intact
- **Per-check assessment**: 10/10 pass
- **Pre-merge readiness**: APPROVE

## CI status

| Check | Result |
|---|---|
| setup | pass (2s) |
| build-and-compare (1, keliya) | pass (58s) |
| asan-ubsan (keliya) | pass (37s) |
| asan-ubsan (qhh) | pass (5s) |
| tools-tests (manifest schema + forcing_dir union tests) | pass (10s) |

All 5 required CI checks pass at frozen HEAD `c6db15f`.

## Round-counter summary

| Phase | Round | Verdict |
|---|---|---|
| 4 cross-review | round 1 (3 parallel reviewers @ 3494c6a) | CLEAN (0 findings) |
| 4.5 verifier | round 1 | — (no candidates) |
| Post-round-1 CI-only doc fix | — | applied (c6db15f) |
| 6/6.2/6.5 | — | SKIPPED (CI-only-equiv classification) |
| 7 final review | — | CLEAN (Gap Sweep @ c6db15f, Reject-When applied) |
| CI | — | 5/5 PASS @ c6db15f |

**Comprehensive rounds**: 1 (single round 1 CLEAN; CI-only doc fix does not count as new round).
**Gate net catch**: 0 (no defects caught by review/verify loop beyond Phase 2 local verification + CI; 2 non-blocking notes addressed by CI-only doc fix).
**Residual deferred**: 1 (PR-E rep dedup informational note from integration N2).
