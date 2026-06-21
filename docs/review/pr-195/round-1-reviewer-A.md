# Round 1 Reviewer A — PR #195 (S5c-C nFCall vs nfe correctness/bitwise/spec)

Reviewed outer SHA: `6281f50` (code-state at review time)
Reviewed SHUD SHA: `d7f5a8b` (code-state at review time)

## Verdict
**clean (0 blocking) + 3 suggestions**

## Blocking Findings
None.

## Suggestions (non-blocking)

### S1 — Line-number drift in f.cpp comment block
`SHUD/src/Model/f.cpp:57`. Comment cites `Model_Data.hpp L60-63 alt counters` but `nFCall5` lives at L64. Fix: `L60-63` → `L60-64`. **FIXED post-review** at SHUD `d82d36e`.

### S2 — Stale `Model/f.cpp:62` reference
`SHUD/src/Model/shud.cpp:398` and `SHUD/B1b_CHANGELOG.md:175`. After S5c-C comment insertion, `MD->nFCall++;` moved from L56 (pre-PR) to L61 (post-PR). Both sites still write L62. **FIXED post-review** at SHUD `d82d36e`.

### S3 — diff column header polish
`SHUD/B1b_CHANGELOG.md:182`. Header `diff` could be `diff (nFCall-nfe)` for clarity. **DEFERRED** — pure doc polish; arithmetic correctness verified on all 6 rows.

## Coverage Confirmed (9 verification items)
1. f.cpp comment accuracy + `++` line non-mutation
2. shud.cpp coupled path nfcall.txt emit placement + format
3. shud.cpp uncouple path nfcall.txt emit (single emit, no 4× overwrite)
4. Bitwise neutrality (file write downstream of .dat output; no RHS path change)
5. B1b_CHANGELOG.md S5c-C section completeness (6 cases × 4 cols + Slurm + 5 gates)
6. Spec 3 scenarios for "nFCall vs nfe 严格分离" all confirmed
7. nFCall1..5 (f_surf/f_unsat/f_gw/f_river/f_lake) UNCHANGED
8. No `cv_mem->` / SUNDIALS internal access
9. MAXQUE / SHUD_ENABLE_PROFILE / Model_Data.hpp / Macros.hpp / cvode_config.cpp UNCHANGED

## Praise (informational)
- Symmetric coupled-vs-uncouple emission catches the uncouple-path edge case
- Error-handling pattern reused verbatim from cvode_stats.txt (consistency)
- CI workflow correctly treats nfcall.txt missing as warning, not fail
- 6-row diff column arithmetic checks out exactly (no transcription errors)
