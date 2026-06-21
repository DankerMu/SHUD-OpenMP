# tools/cvode_stats_diff/

Strict 15-key bitwise diff + nFCall-exclusion unit test for SHUD's CVODE final
stats snapshot. Owned by openspec change `b1b-baseline-completion` (capability
`s5c-solver-diagnostics`); see `openspec/changes/b1b-baseline-completion/` for
the requirement narratives + design rationale (D10 nFCall vs nfe semantics).

## Files

| File | Purpose |
|---|---|
| `cvode_stats_diff.sh` | POSIX shell strict 15-key diff. Usage: `cvode_stats_diff.sh <new.txt> <golden.txt>`. Exit 0 iff every canonical key matches byte-for-byte; any MISSING / UNKNOWN / DUPLICATE / value mismatch = exit 1. Inline `CANONICAL_KEYS=` is the runtime authority. |
| `canonical_15_keys.yaml` | Single source-of-truth schema for the 15-key set + the `excluded_keys` block (nFCall family) + the `diagnostic_keys` block (`-DSHUD_ENABLE_DIAGNOSTICS` extras). Cross-checked in CI against the shell script via the Python unit test. |
| `test_15key_excludes_nfcall.py` | Python 3 unit test (pyyaml). Asserts (a) yaml has exactly 15 canonical keys; (b) all 6 `nFCall*` variants are excluded; (c) no overlap between canonical and diagnostic blocks; (d) shell script `CANONICAL_KEYS` matches yaml in-sync. Run via `uv run python tools/cvode_stats_diff/test_15key_excludes_nfcall.py`. |
| `test_cvode_stats_diff.sh` | POSIX shell 6-scenario acceptance test for `cvode_stats_diff.sh` (identical / mismatch / missing / unknown / duplicate / unknown-in-golden). Runs in CI before the SUNDIALS build to fail fast on helper-script regressions. |
| `fixtures/sample_cvode_stats.txt` | Minimal 15-key fixture used as a static reference + smoke input. Realistic small values; do NOT add diagnostic keys here (the fixture is meant to model an OFF-build snapshot). |

## Canonical 15-key set

`nfe`, `nfeLS`, `nni`, `nli`, `nsetups`, `netf`, `nst`, `npe`, `nps`, `ncfn`,
`ncfl`, `lenrw`, `leniw`, `lenrwLS`, `leniwLS`. These are the actual SUNDIALS
CVODE 6.0.0 stats emitted by `SHUD/src/Equations/cvode_config.cpp::PrintFinalStats()`.

## nFCall channel (S5c-C, #175)

`nFCall` is SHUD's own RHS-kernel entry counter (`Model_Data::nFCall`, declared
at `SHUD/src/ModelData/Model_Data.hpp:58`; incremented in
`SHUD/src/Model/f.cpp:62`). It is emitted to a SEPARATE file
`<output>/nfcall.txt` (one `nFCall=<N>` line) and is NOT part of the 15-key
gate. Rationale: CVODE's `nfe` counter (number of RHS calls observed by the
solver) may diverge from SHUD's `nFCall` due to finite-difference Jacobian
re-callbacks; treating them as a single key would force a permanent value
mismatch on every run. The `test_15key_excludes_nfcall.py` unit test enforces
this contract — adding any `nFCall*` to `canonical_15_keys` will fail CI.

## Spec / design references

- Spec: `openspec/changes/b1b-baseline-completion/specs/s5c-solver-diagnostics/spec.md`
  (Requirement "nFCall 与 CVODE 15-key nfe 严格分离")
- Design: `openspec/changes/b1b-baseline-completion/design.md` (D10)
- Historical context: F19 PR #54 round 2 (the original `nFCall` drop from the
  inline CANONICAL_KEYS set).
