# Round 1 Reviewer B — PR #195 (S5c-C CI/tooling/yaml/unit test)

Reviewed outer SHA: `6281f50` (code-state at review time)
Reviewed SHUD SHA: `d7f5a8b` (code-state at review time)

## Verdict
**clean (0 blocking) + 2 suggestions**

## Blocking Findings
None.

## Negative-test validation (empirical)
- `uv run --with pyyaml python tools/cvode_stats_diff/test_15key_excludes_nfcall.py` exits 0 against PR contents.
- Inject `nFCall` into yaml `canonical_15_keys` → exit 1 (`canonical_15_keys size = 16, expected 15`).
- Inject `nFCall` into shell `CANONICAL_KEYS` → exit 1 (yaml↔sh sync drift caught).
- `cvode_stats_diff.sh fixtures/sample_cvode_stats.txt fixtures/sample_cvode_stats.txt` exits 0.

## Suggestions (non-blocking)

### S1 — `Read nFCall` step missing `data_available == 'true'` gate
`.github/workflows/serial-baseline.yml:792`. Other case-output-consuming steps in the matrix job ALL gate on `&& steps.data_probe.outputs.data_available == 'true'`. Without it, the step emits a misleading `nfcall.txt missing` warning on undeployed-data matrix entries. **FIXED post-review** at outer `5d0ab03`.

### S2 — awk parse-failure noise
`.github/workflows/serial-baseline.yml:799-801`. Empty-string default renders `MISSING` in notice title; could be a `::warning` if empty. **DEFERRED** — pure polish; informational-only step never fails CI.

## Coverage Confirmed (10 verification items)
1. canonical_15_keys.yaml: 15 keys match cvode_stats_diff.sh L54 char-for-char; 6 excluded; 19 diagnostic (matches cvode_config.cpp L121-164)
2. test_15key_excludes_nfcall.py: size + excluded + diagnostic + yaml↔sh sync assertions all correct
3. README.md: documents all 4 artefacts + spec + D10 + F19 history
4. fixtures/sample_cvode_stats.txt: 15 canonical keys, no nFCall, parses with cvode_stats_diff.sh
5. CI workflow steps: correctly gated + correct placement + correct indentation
6. `${{ matrix.case }}` substitutions correct
7. `python3 -m pip install` exception documented at workflow L436-443
8. cvode_stats_diff.sh CANONICAL_KEYS line UNCHANGED by diff
9. No accidental edits (exactly 5 expected files + SHUD pointer)
10. excluded_keys covers nFCall + nFCall1..5 (matches Model_Data.hpp L58 + L60-64)

## Praise (informational)
- Sync-test bidirectionality (yaml↔sh; belt-and-braces nFCall absence check)
- Comment density (every consumer named, spec strings cited verbatim)
- CI exception rationale (8-line block explains uv-vs-pip-install choice)
- Negative-test self-validation (script catches every realistic regression vector)
