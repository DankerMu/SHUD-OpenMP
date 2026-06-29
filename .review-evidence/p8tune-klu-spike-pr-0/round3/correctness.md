# PR-0 Round 3 — Correctness Review (Post Round 2 Fix Commit 7fc325b)

## Review: G1+G4 regression-fix verification on top of round-2 head 50d2a4b

### Summary

G1 and G4 both correctly addressed. The G1 fix is surgical, semantically correct per KLU 7.12.2, observable in the ordering-matrix smoke log (2 of 3 cells take the skip path with diagnostic), and proven by post-fix data: COLAMD numeric_lnz+unz=47598 vs the natural-ordering 2.1M entries on the same problem — i.e., the skipped preflight is correctly avoiding both the spurious-OOM and the spurious-PASS arm of UB. G4 removed the legacy log line without breaking adjacent advisory hint or downstream consumers (sweep across `tools/p8tune.*`, `openspec/`, and `docs/` shows zero LIVE callers of the `[klu] pre-flight:` log idiom). No new correctness issues surfaced.

### Findings

#### 🟢 Praise: G1 fix is textbook-correct
`tools/p8tune.D/klu_analyze_factor.cpp:265-292`

The gate `ordering_id == 0 && Symbolic->lnz > 0 && Symbolic->unz > 0` is doubly defensive: (a) `ordering_id==0` directly encodes the KLU 7.12.2 contract that lnz/unz are only populated for AMD; (b) the `lnz > 0 && unz > 0` belt+braces guards against any AMD-internal failure mode where klu_analyze succeeds at the `if (!Symbolic) return 1` gate but leaves lnz/unz unset. The skip-path diagnostic `PREFLIGHT_AFTER_ANALYZE skipped (ordering=%s lnz=%lld unz=%lld; AMD-only)` uses `long long` cast (signed) which correctly preserves the -1 sentinel for observability — exactly what PR-A needs to confirm the skip-path was taken in production. The 8-line comment block (L264-272) captures the root cause (`static_cast<size_t>(-2.0)` is UB) and the consequence (spurious OOM / spurious PASS across 8/16 non-AMD cells) so future readers cannot accidentally re-introduce.

#### 🟢 Praise: Symbolic memory management is leak-free on both branches
`tools/p8tune.D/klu_analyze_factor.cpp:284, 308, 312, 319-320, 341-342`

Audit of all `return` sites after `Symbolic = klu_analyze(...)`:
- L284: AMD preflight OOM exit → frees Symbolic before return 0 ✓
- L308: klu_factor returns NULL with KLU_OUT_OF_MEMORY → frees Symbolic before return 0 ✓
- L312: klu_factor returns NULL with other status → frees Symbolic before return 1 ✓
- L319-320: post-factor RSS exceeds CN_RAM → frees BOTH Numeric and Symbolic before return 0 ✓
- L341-342: happy path → frees both ✓
- Non-AMD skip branch (L287-292) does NOT return — falls through to klu_factor at L297, then through the standard cleanup paths. No leak.

The NEW ELSE branch added by G1 is correctly fall-through (not an early return), so Symbolic stays owned by the existing cleanup at L308/L312/L319-320/L341-342. No regression.

#### 🟢 Praise: Empirical proof that the UB was real
`.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log:23, 24, 38, 39`

Post-G1 binary (`klu_analyze_factor` SHA `58b0f9e7`) confirms `symbolic_lnz_est=-1 symbolic_unz_est=-1` for both COLAMD and natural orderings — exactly the EMPTY sentinel KLU 7.12.2 documents. The pre-G1 (round-2 50d2a4b) computation `static_cast<size_t>(Symbolic->lnz + Symbolic->unz)` would have been `static_cast<size_t>(-2.0) = 18446744073709551614` (UB but on most platforms wraps to ~16 EB), then `× 24 × 1.5 / 1` → overflows to a still-massive size_t > rss_budget on essentially any host, triggering spurious `KLU_OOM_DETECTED reason=preflight_after_analyze` on 8/16 PR-A cells. (On other compiler/optimization paths, the conversion of -2.0 to size_t is UB and the compiler is licensed to assume the cast cannot produce a value > size_t max — i.e., a spurious PASS path is also possible.) The G1 fix kills both arms.

#### 🟢 Praise: G4 removal is clean
`tools/p8tune.D/klu_analyze_factor.cpp:237`

Cross-verified:
1. `grep -rn 'pre-flight:'` across `tools/p8tune.D/` returns 0 hits (post-G4 binary will not emit the legacy line).
2. The same grep across all of `tools/`, `openspec/`, and `docs/` shows zero LIVE consumers of the legacy `[klu] pre-flight:` log idiom — no aggregator, no awk script, no test fixture parses it. (3 remaining "pre-flight" hits are CONCEPT-level references in `cn_node_ram.h:10` comment, `tasks.md:18` task description, and `spec.md:134` REQ wording — these refer to "the pre-flight check" as an abstract concept, still accurate, NOT the log prefix string.)
3. Adjacent logic intact: L235-237 retains `PREFLIGHT_HINT pattern_est_bytes=...` (the advisory hint), L239 is the blank-line separator before `// --- 4. klu_analyze (symbolic) ---` — both untouched.
4. The smoke fixture `mac_smoke_keliya_klu_factor.log:5` still contains the legacy line, but is a non-load-bearing one-time round-1 capture (not consumed by anything; pure historical evidence). The fresh `mac_smoke_keliya_klu_ordering_matrix.log` from the round-3 binary correctly shows zero `pre-flight:` hits across 3 ordering runs.

#### 🟢 Praise: Round-3 keliya tool-correctness gate re-run clean
`.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_dense_fd_cross_check.{log,tsv}`

Dense FD cross-check re-run shows max_rel_err ≤ 2.21e-07 across all 5 nonzero blocks (surf×surf, unsat×{surf,unsat,gw}, gw×{unsat,gw,river}, river×river), well within the 1.0e-06 threshold. Overall verdict PASS. Confirms G1/G4 source changes did not perturb the FD-color path (no rebuild artifact corruption, no link-order change that surfaces numerical noise). Wall budget: 2c gate stays sub-second per spec §3.

#### 🟢 Praise: README schema synced (G2 fix is correct)
`tools/p8tune.D/README.md:166`

OOM reason enumeration now reads `<preflight_after_analyze | klu_factor_OOM | post_factor_rss_exceeds_cn_ram>` — exactly the 3 strings emitted by `klu_analyze_factor.cpp` at L282, L306, L317. PR-B aggregator (when implemented) can grep this README schema and the regex will match all emitted lines. No round-3 e02-style drift.

### Regression Sweep (Targeted Checks)

Per the round-3 brief, I explicitly verified:

1. **Did the new ELSE branch leak Symbolic if a later path returns 1?** NO. The ELSE branch falls through to klu_factor at L297. The only `return 1` after that point is L312 (klu_factor failed with non-OOM status), which correctly frees Symbolic at L312. Audited above.

2. **Does the removed log line have any test fixture or downstream consumer?** NO. Grep across `tools/p8tune.*`, `openspec/changes/p8tune-klu-spike/`, `docs/` returns ZERO live callers of the `[klu] pre-flight:` log idiom. The 3 hits found are conceptual references to "pre-flight check" as architecture, not the literal log string. The single fixture `mac_smoke_keliya_klu_factor.log:5` carries the legacy string in historical evidence only — no aggregator parses it.

3. **Has spike_run.sh been re-run with the keliya tool-correctness gate after G1/G4?** YES — `mac_smoke_keliya_dense_fd_cross_check.{log,tsv}` were re-generated at 2026-06-28T07:03Z (TSV mtime), max_rel_err stays ≤ 2.21e-07 (within ≤ 1e-6 threshold). And `mac_smoke_keliya_klu_ordering_matrix.log` was built from binary SHA `58b0f9e7...` (timestamped 11:03:20Z) and shows all 3 orderings completing cleanly:
   - amd:    preflight FIRES (`symbolic_lnz+unz=33088 est_bytes=1191168 rss_budget=129869709311`) → klu_factor success
   - colamd: preflight SKIPS with diagnostic (`lnz=-1 unz=-1; AMD-only`) → klu_factor success
   - natural: preflight SKIPS with diagnostic → klu_factor success (numeric_lnz=1.06M, fill_ratio=205.9, takes 345ms — the natural-ordering pathology that justifies the AMD/COLAMD comparison)
   Zero `KLU_OOM_DETECTED` lines (correctly — keliya at 1785×1785 with 173 GiB budget cannot reach OOM).

### Verdict

APPROVE — G1 (the critical regression fix) and G4 (the log-hygiene cleanup) both land correctly, with comprehensive diagnostic observability for PR-A, leak-free state machine, and verified-clean tool-correctness gate. No new findings; the round-2 critical e01 + minor e04 are both confirmed RESOLVED.

