# Cross-Review Evidence Bundle — PR #372

**Reviewed outer HEAD SHA**: `768c905f8f078e7ece27bc4d8e4efb4ab0a1b825` (single SHA throughout — no post-round-1 fixes)
**SHUD pin under review**: `6ce17d6` (origin/openmp-baseline)
**OpenSpec change**: `p8tune-spgmr-maxl` (capability `spgmr-maxl-env-hook`)
**Fixture level**: expanded (whole change) / PR-C high-intensity (single SHUD source + 4-way bit-identical CI gate)
**Repair intensity**: high (D0 risk-triage: required Invariant Matrix D15 authoring; full 4-reviewer panel)

## Phase 0.5 Fixture Review (Invariant Matrix authoring + validation)

D15 Invariant Matrix authored at `openspec/changes/p8tune-spgmr-maxl/design.md` L218-258 (governing invariant + source-of-truth identity = PR-A SHA12 `1bfe6a30856e` + 8 surface categories + 11 regression rows + boundary-surface checklist).

Fixture reviewer (Phase 0.5) verdict: **APPROVE** — all 7 checks PASS (IM completeness / spec↔tasks↔IM consistency / risk pack coverage / boundary-surface checklist / expanded fixture justified / non-goals discipline / openspec validate strict).

## Phase 4 Round 1 — Risk-adaptive cross-review (4 parallel reviewers)

### Reviewer agent: `review-correctness`
- **Reviewed**: outer `768c905` + SHUD `6ce17d6`
- **Summary**: 10/10 PASS, APPROVE
- **Findings**: None.
- **Non-blocking notes** (2): N1 stale "L259" comment ref deferred; N2 defense-in-depth redundancy informational

### Reviewer agent: `review-spec-compliance`
- **Summary**: 8/8 PASS, APPROVE
- **Per-task DONE/MISSING**: §3.1-§3.5 DONE; §3.6 PARTIAL (server-side build implicit via Slurm 9626; Mac build per project §1.1.1 server-only quantitative convention); §3.7-§3.8 DONE
- **Findings**: None.
- **Non-blocking notes** (3): §3.6 Mac build cleanup possible; sbatch ExitCode honest disclosure; impl exceeds spec L31 minimum

### Reviewer agent: `review-integration`
- **Summary**: 10/10 PASS, APPROVE
- **Findings**: None.
- **Non-blocking notes** (2): sbatch parser bug already disclosed in g3_verdict.md; outer branch state informational
- **Cross-PR contract verified**: PR-A baseline reuse (env-unset bit-identical preserves 18-cell numbers); PR-D consumer contract (provenance log format exact match); PR-E aggregator contract (15-key schema preserved); SHUD master isolation; B1a/B1b untouched; OMP path neutrality

### Reviewer agent: `review-security-perf`
- **Summary**: 11/11 PASS, APPROVE
- **Findings**: None.
- **Non-blocking notes** (3): char-whitelist defense-in-depth + ERANGE catch + fflush discipline
- **Bitwise reproducibility verified**: all 4 invocations SHA12=`1bfe6a30856e` match anchor + 15 keys bit-identical + cross-run cmp byte-identical

## Phase 4.5 — Independent finding-verification gate

| Candidate # | Source reviewer | Verdict |
|---|---|---|
| (none) | — | — (no candidates to verify) |

All 4 round 1 reviewers reported zero actionable findings per `finding-contract.md`. No verifier subagent spawned. Verdict table persisted at `.review-evidence/p8tune-pr-c/round1-phase45-verifier.md`.

## Phase 7 — Independent final review (Gap Sweep)

- **Reviewer agent**: independent-final (clean-context)
- **Reviewed**: outer `768c905` + SHUD `6ce17d6`
- **Verdict**: CLEAN — Reject-When precision applied; no real defect surfaced
- **Independent re-verification** (NOT relying on prior reviewer reports):
  - Outer diff scope: 1 file (SHUD pointer)
  - SHUD diff scope: 1 file (cvode_config.cpp +67/-2)
  - Helper logic per-branch trace verified (NULL/""/"0"/"5"/"10"-"30"/invalid all match spec)
  - Call-site PREC_NONE preserved
  - G3 evidence re-extracted: 4× SHA12 = `1bfe6a30856e` + 15 keys = anchor; cross-run cmp identical × 3; log discipline 0/0/0/1
  - PREC_LEFT regression grep = 0 matches (repo-wide)
  - SHUD master isolation: `git branch -r --contains 6ce17d6` = openmp-baseline ONLY
  - openspec validate strict PASS
  - Invariant Matrix D15 11 regression rows all traceable
  - Oracle integrity preserved
- **Per-check**: 10/10 pass
- **Pre-merge readiness**: APPROVE

## G3 4-way bit-identical CI gate (cn14 server verification — THE integration gate)

**Slurm job**: 9626 (CPU partition, cn14, /scratch/.p8tune-runs/pr-c-g3-gate/)
**SHUD pin under test**: 6ce17d6 (origin/openmp-baseline)
**PR-A anchor pin**: 37be0fe (cleaned-PREC_NONE production baseline)
**Anchor rivqdown.dat SHA12**: `1bfe6a30856e`

### 4-way rivqdown.dat SHA12 result

| Run | Invocation | SHA12 | Anchor match |
|---|---|---|---|
| 1 | `unset SHUD_SPGMR_MAXL && ./shud keliya` | `1bfe6a30856e` | PASS |
| 2 | `SHUD_SPGMR_MAXL= ./shud keliya` (empty) | `1bfe6a30856e` | PASS |
| 3 | `SHUD_SPGMR_MAXL=0 ./shud keliya` | `1bfe6a30856e` | PASS |
| 4 | `SHUD_SPGMR_MAXL=5 ./shud keliya` | `1bfe6a30856e` | PASS |

Cross-run cmp: run1 == run2 == run3 == run4 byte-identical.

### 4-way 15-key cvode_stats result

All 4 runs produce IDENTICAL cvode_stats.txt matching PR-A anchor:
- nfe=112248, nfeLS=116421, nni=112247, nli=116421, nsetups=0, netf=5, nst=110917, npe=0, nps=0, ncfn=205, ncfl=42, lenrw=23294, leniw=53, lenrwLS=21474, leniwLS=42

### Stdout provenance log discipline (IM D15 L235-238)

| Run | Env state | `[CVODE] SPGMR maxl=` log count | IM L235-238 expected | Match |
|---|---|---:|---:|---|
| 1 | unset | 0 | 0 (silent default) | PASS |
| 2 | "" empty | 0 | 0 (silent default) | PASS |
| 3 | "0" | 0 | 0 (silent default) | PASS |
| 4 | "5" | 1 | 1 (provenance emitted; artifact unchanged) | PASS |

Slurm ExitCode 1:0 was caused by sbatch script's grep parser bug (used `^${KEY}[[:space:]]` but cvode_stats format is `key=value`). The parser bug caused false-fail flags but did NOT affect run execution, artifact capture, SHA12 computation, or cross-run cmp equivalence. Manual re-verification (this bundle) definitively proves G3 PASS.

## CI status

| Check | Result |
|---|---|
| setup | pass (4s) |
| build-and-compare (1, keliya) | pass (1m4s) |
| asan-ubsan (keliya) | pass (36s) |
| asan-ubsan (qhh) | pass (5s) |
| tools-tests (manifest schema + forcing_dir union tests) | pass (9s) |

All 5 required CI checks pass at frozen HEAD `768c905`.

## Round-counter summary

| Phase | Round | Verdict |
|---|---|---|
| 0.5 fixture review | — | APPROVE (IM D15 + boundary-surface checklist authored + validated) |
| 4 cross-review | round 1 (4 parallel reviewers @ 768c905) | CLEAN (0 findings, 10 non-blocking notes) |
| 4.5 verifier | round 1 | — (no candidates) |
| 5/6/6.2/6.5 | — | SKIPPED (no findings; cosmetic L259 comment deferred) |
| 7 final review | — | CLEAN (Gap Sweep @ 768c905, Reject-When applied) |
| CI | — | 5/5 PASS @ 768c905 |
| G3 gate | server cn14 Slurm 9626 | PASS (4-way bit-identical to PR-A anchor) |

**Comprehensive rounds**: 1 (clean single-round flow; no post-review fixes; no SHA drift).
**Gate net catch**: 0 (no defects caught by review/verify loop beyond Phase 2 local + G3 server gate).
**Residual deferred**: 1 (cosmetic stale L259 comment in SHUD source helper; non-binding annotation).
