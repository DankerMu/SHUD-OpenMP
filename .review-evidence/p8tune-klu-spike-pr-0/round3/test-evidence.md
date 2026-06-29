Reviewer agent: review-test-evidence
Review round: round 3 (post round-2 fix G1-G4)
Reviewed head SHA: 7fc325b6986bc7be3256bc839cdba4b5eea91e5c (outer) / 41d9a17 (SHUD)
Round 2 head: 50d2a4b
Round 2 verdict: APPROVE + 1 non-blocking note (e05 dense binary determinism — DEFERRED per spec scope).

Summary: Round 2 critical e01 (UB on `Symbolic->lnz/unz` cast for non-AMD orderings) is genuinely ADDRESSED. New ordering matrix log captures all 3 orderings (amd / colamd / natural) reaching klu_factor with the gated preflight: AMD takes the decisive branch (`PREFLIGHT_AFTER_ANALYZE symbolic_lnz+unz=33088 est_bytes=1191168 rss_budget=129869709311`), COLAMD + natural take the new diagnostic skip path (`PREFLIGHT_AFTER_ANALYZE skipped (... AMD-only); relying on post-factor RSS check`). e02 README enum updated. e03 + e04 source changes confirmed in diff. e05 properly deferred. One stale-evidence finding below.

Findings:

- Severity: minor
  Failure class: Stale smoke-evidence file (G4 removal not reflected)
  Contract or invariant: Brief task #2 — "`mac_smoke_keliya_klu_factor.log` should reflect G4 (no more `pre-flight:` line) + still PASS".
  Scenario or repro: `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_factor.log` mtime is `06:44:18` (round 2 / pre-G4), whereas `klu_analyze_factor.cpp` was edited at `07:02` (G4 source removal) and the binary rebuilt at `07:06`. Line 5 of the file still contains the legacy diagnostic `[klu] pre-flight: A nnz=10255 est_bytes=770560 cn_ram=185528156160` that G4 deleted at source L238-239 (confirmed by `git diff 50d2a4b..7fc325b -- tools/p8tune.D/klu_analyze_factor.cpp` showing the removal). The committed evidence therefore documents PRE-G4 behavior, not the actual post-G4 binary that ships in PR-0.
  Required test or evidence: Re-run `spike_run.sh keliya amd 1` against the post-G4 binary; overwrite `mac_smoke_keliya_klu_factor.log`. Verify the resulting file matches the AMD section of `mac_smoke_keliya_klu_ordering_matrix.log` (which IS post-G4 — generated at 07:03:20 against binary SHA `58b0f9e7...`, where line 5 correctly has no `pre-flight:` text — see ordering matrix L5-19).
  Sibling surfaces: PR-A aggregator (`tools/p8tune.D/aggregate_klu_spike.sh`, not yet authored) will parse per-cell logs from the PR-A sweep — those will be post-G4 by construction since PR-A binaries are rebuilt on cn-node. No PR-A risk; risk is local to PR-0 acceptance §1.15(e) evidence integrity.
  Blocks merge: no
  Impact: Acceptance §1.15(e) ("klu_analyze_factor PASS without OOM") is still proven by both files (peak_rss=3145728B in both) — the substantive PASS verdict is unaffected. The cost is reviewer trust + reproducibility audit clarity: the file purports to be the canonical keliya AMD+BTF1 PR-0 smoke artifact, but actually captures a transient pre-G4 binary state. PR-B aggregator robustness is also slightly weakened — if it ever runs against this stale file, it would parse a now-removed string. (Mitigated because the ordering matrix log is the authoritative post-G4 reference.)
  Requested fix: Either (a) regenerate `mac_smoke_keliya_klu_factor.log` from the post-G4 binary (preferred — restores 1:1 evidence-to-binary correspondence), OR (b) drop `mac_smoke_keliya_klu_factor.log` entirely and point §1.15(e) to the `amd btf=1` section of `mac_smoke_keliya_klu_ordering_matrix.log` (it already covers the same combo + adds the 2 missing orderings).

Non-blocking notes:

- Round 2 e01 (CRITICAL) — ADDRESSED via G1. Source guard at `tools/p8tune.D/klu_analyze_factor.cpp:273` reads `if (ordering_id == 0 && Symbolic->lnz > 0 && Symbolic->unz > 0)`. Inline comment L264-272 cites both upstream root cause (KLU 7.12.2 `klu.h:36-39` AMD-only field contract) and round-2 provenance (`PR-0 reviewer round 2 finding e01 / G1 fix`). New diagnostic format `PREFLIGHT_AFTER_ANALYZE skipped (ordering=%s lnz=%lld unz=%lld; AMD-only); relying on post-factor RSS check` at L288-291. The static_cast<size_t>(-2.0) UB path is eliminated.

- Ordering matrix smoke (new file) — PASSES brief gate. `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log` (2719 bytes, mtime 07:03:20):
  - L1-3: header captures build-on timestamp `2026-06-28T11:03:20Z` + `klu_analyze_factor SHA: 58b0f9e7...` (an intermediate binary hash from the run window; current binary at 07:06 has different SHA `d352d4...` due to clean rebuild post-run — this is benign).
  - L5-18 (ordering=amd btf=1): `PREFLIGHT_AFTER_ANALYZE symbolic_lnz+unz=33088 est_bytes=1191168 rss_budget=129869709311` (decisive estimate path fires), `numeric_lnz=16544 numeric_unz=16544 fill_ratio=3.2265 peak_rss=3145728B` → klu_factor SUCCEEDS, no OOM.
  - L20-33 (ordering=colamd btf=1): `PREFLIGHT_AFTER_ANALYZE skipped (ordering=colamd lnz=-1 unz=-1; AMD-only); relying on post-factor RSS check` (G1 diagnostic fires, confirming EMPTY sentinel that previously triggered UB), `numeric_lnz=23799 numeric_unz=23799 fill_ratio=4.6414 peak_rss=3555328B` → klu_factor SUCCEEDS.
  - L35-48 (ordering=natural btf=0): same skip diagnostic with `ordering=natural`, `numeric_lnz=1055998 numeric_unz=1055998 fill_ratio=205.95 peak_rss=35700736B` → klu_factor SUCCEEDS (slow as expected for natural ordering on a 2D mesh, but no OOM at keliya scale). All 3 reach klu_factor + succeed.

- Dense FD cross-check TSV regeneration — VERIFIED. `mac_smoke_keliya_dense_fd_cross_check.tsv` mtime is `07:03:32`, AFTER the G1 source edit at 07:02 — implementer's claim of regeneration post-G1 holds. TSV still reports the 3 distinct nonzero rel_err values from round 2 (max 2.128e-07 on unsat×gw, ≤ 1e-6 PASS), and the `overall PASS` line. G1-G4 source changes do not touch `fd_color_jacobian` or `dense_fd_cross_check.py`, so the regression-stability assertion is consistent with the unchanged numerics. (Note: the implementer's quoted max_rel_err=2.139e-07 appears to be a typo — actual maximum across all blocks is 2.128e-07 at unsat×gw row 9 + 2.139e-08 at unsat×unsat row 8. The TSV is correct; the commit message just transposed digits.)

- G2 README enum update — VERIFIED in diff. `git diff 50d2a4b..7fc325b -- tools/p8tune.D/README.md` shows exactly the line-167 change: `preflight_estimate` → `preflight_after_analyze` in the OOM-as-data-point reason enumeration. Aligns with the actual emitted strings at `klu_analyze_factor.cpp:282,306,317`. Documentation now matches code.

- G3 (e03 spec errata for .sp.{mesh,riv,rivseg,att}) — NOT in diff because openspec tree is gitignored per project convention (`.gitignore:13` matches `openspec/changes/`). Per implementer commit message, the change lives in work-tree only. Confirmed `tools/p8tune.D/verify_adjacency_keliya.py` source-of-truth Python implementation still reads the correct file extensions; behavior was already correct in round 1.

- G4 (legacy pre-flight line removed) — source change VERIFIED in diff (`klu_analyze_factor.cpp` L238-239 hunk removes both the printf and its arg list). New ordering matrix log L5-18 confirms the line is absent from current binary output. The stale `mac_smoke_keliya_klu_factor.log` is the only evidence file that still shows the legacy text (finding above).

- Acceptance sub-gates §1.15 (a)-(f) post-G1-G4 evidence mapping:
  - (a) 5-block CSC dump → `mac_smoke_keliya_adjacency.log` (round 2; G1-G4 do not touch dump_adjacency). UNCHANGED.
  - (b) per-block nnz exact-match → `mac_smoke_keliya_verify_nnz.log` + `.tsv` (round 2; G1-G4 do not touch verify_adjacency_keliya.py). UNCHANGED.
  - (c) χ ≤ 30 → `mac_smoke_keliya_chi.log` + `mac_smoke_keliya_chi_assertion.log` (round 2; G1-G4 do not touch fd_color_jacobian). UNCHANGED.
  - (d) brute-force dense FD cross-check ≤ 1e-6 → `mac_smoke_keliya_dense_fd_baseline.log` + `mac_smoke_keliya_dense_fd_cross_check.tsv` (regenerated 07:03:32 post-G1; PASS, max 2.128e-07).
  - (e) klu_analyze_factor PASS without OOM → covered BOTH by `mac_smoke_keliya_klu_factor.log` (stale-pre-G4 caveat per finding above) AND by the AMD section of `mac_smoke_keliya_klu_ordering_matrix.log` (post-G4 authoritative). Net: PASS verdict holds.
  - (f) determinism 2× sha256 → `mac_smoke_keliya_determinism.txt` (round 2; G1-G4 do not touch fd_color_jacobian). UNCHANGED.

- Format drift risk for PR-B aggregator — LOW. The new `PREFLIGHT_AFTER_ANALYZE skipped` diagnostic format is well-formed key=value, uses the same prefix `[klu] PREFLIGHT_AFTER_ANALYZE` as the AMD branch, and the README §output-format §klu_analyze_factor stdout block already documents `[klu]`-prefix logs as parser-target lines. PR-B aggregator (task 3.1, not yet authored) will need to skip non-numeric lnz values when computing fill estimate from preflight — but the post-factor `numeric_lnz/unz` fields in `cell_summary` are unaffected and remain the canonical fill-axis data source per spec §Scenario "Fill axis threshold". No new evidence file format introduced.

- e05 dense binary determinism — properly DEFERRED. Spec REQ-2 Scenario "FD probe determinism" at `spec.md:49-53` is scoped to the colored binary only. G1-G4 do not affect this. PR-A may add as hardening (per round 2 reviewer note).

- Falsification audit: scanned the 3 changed files for any short-circuit / hardcoded-PASS paths analogous to round 1 c01. `klu_analyze_factor.cpp` L273-292 guard is genuine: the `else` branch unconditionally prints the diagnostic (no silent skip), and the post-factor RSS check at L302 + L316-322 still gates true OOM emission regardless of which preflight branch ran. README enum tweak is documentation-only. No falsification pattern.

- Bonus: a sibling reviewer (integration scope) already published `.review-evidence/p8tune-klu-spike-pr-0/round3/integration.md` confirming the 3-file change envelope, clean rebuild, and submodule pointer stability (`41d9a17` unchanged). My findings are scoped strictly to test-evidence and do not duplicate that audit.

Verdict: APPROVE (1 minor stale-evidence finding documented above; non-blocking — substantive PASS verdict for §1.15(e) is preserved by the new ordering matrix log).
