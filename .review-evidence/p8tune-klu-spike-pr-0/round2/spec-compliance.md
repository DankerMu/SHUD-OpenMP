Reviewer agent: review-spec-compliance
Review round: round 2
Reviewed head SHA: 50d2a4bddacbfa3ef5b3e1c25d760555103c5556

Summary: All 3 round-1 spec-compliance findings (c01, c05, c08) ADDRESSED. F6/F7/F8 spec amendments are correctly worded and `openspec validate p8tune-klu-spike --strict --no-interactive` returns `Change 'p8tune-klu-spike' is valid`. Two new minor findings from the fix delta: (a) README §output-format L166 OOM-reason enumeration `preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram` contradicts the C++ source which emits `preflight_after_analyze` (F4 renamed the preflight reason but left README stale), violating REQ-8 L255 "binary / text format SHALL conform to a schema documented in tools/p8tune.D/README.md §output-format"; (b) F1's `--brute-force-dense` CLI flag is genuinely new but is enumerated in the print_usage block and documented in the spec amendment's enabling clause (REQ-7 envelope ⊇ `dense_fd_cross_check.py`), so under REQ-8 L256 "new flags MAY be added without OpenSpec amendment" this is acceptable additive expansion. No spec drift from F4/F5/F9/F10/F11.

Round 1 finding verification:

- c01 (critical, spec REQ-3 dense FD cross-check SHALL clause not implemented): ADDRESSED
  - Evidence file `tools/p8tune.D/fd_color_jacobian.cpp:182-184` print_usage now documents `--brute-force-dense` flag; main loop at `:205,214,237,256-258,351-377` executes a true per-column independent FD probe when the flag is set, emitting `<prefix>_numeric_J_dense.bin` per `:425-427`.
  - Companion `tools/p8tune.D/dense_fd_cross_check.py:106-156` now loads BOTH the colored and dense binaries, computes per-pattern-entry `rel_err = |dut - ref| / max(|ref|, |dut|, 1e-12)`, and aggregates max_rel_err per 5-block pair. Verified against measured evidence at `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_dense_fd_cross_check.tsv:7-9` (unsat×{surf,unsat,gw} blocks show 2.041e-07, 2.139e-08, 2.128e-07 respectively — all ≤ 1e-6 threshold), overall=PASS at L33.
  - Spike driver `tools/p8tune.D/spike_run.sh:108-113` gates the dense + cross-check invocation on `CASE=keliya AND ORDERING=amd AND BTF=1`, matching spec REQ-7 L219 "dense FD cross-check on keliya numeric J" acceptance clause.

- c05 (major, χ ≤ 30 keliya bound not programmatically asserted): ADDRESSED
  - `tools/p8tune.D/fd_color_jacobian.cpp:244-255` implements case-aware gate: `chi_threshold = (case_name == "keliya") ? 30 : 50; if (chi > chi_threshold) { ... return 2; }`. Asserted ALWAYS (not just under `--report-chi-only`), so a regression in ColPack ordering or CSC pattern is caught in the full FD-probe pipeline.
  - Comment at `:244-249` explicitly cites "PR-0 reviewer round 1 finding c05 / F5 fix" and spec REQ-2 Scenario "Column coloring via Welsh-Powell".
  - Evidence at `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_chi_assertion.log:13-21` confirms exit_code=0 with chi=16 (well under the 30 bound).

- c08 (minor, spec REQ-7 L215 envelope omits .gitignore + dense_fd_cross_check.py): ADDRESSED
  - In-tree spec at `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md:215` now enumerates both `SHUD/.gitignore (additive amendment to suppress libshud.a + _libshud_obj/ build artifacts from polluting fresh clones — the documented carve-out)` and `tools/p8tune.D/dense_fd_cross_check.py (real brute-force dense FD vs colored FD cross-check per REQ-3 keliya tool-correctness gate)`.
  - All 12 tracked files under `tools/p8tune.D/` (incl. `.gitignore`) and the SHUD pointer bump (`bc919f5 → 41d9a17`, the chore commit "gitignore libshud.a + _libshud_obj/ build artifacts") fall within the amended envelope. No file boundary violation remains.

F7/F8 quality verification (spec amendments also addressed by this fix delta):

- F7 (REQ-1 L18 carve-out caveat): correctly amended. New text reads "AND PR-0 SHALL NOT bump the SHUD submodule pointer EXCEPT for the additive `libshud.a` carve-out commit(s) on `openmp-baseline` per REQ-7 (pin advances 6ce17d6 → openmp-baseline HEAD; carve-out also includes a `.gitignore` amendment to suppress `libshud.a` + `_libshud_obj/` build artifacts from polluting fresh clones)". Wording is precise — the EXCEPT clause is tightly scoped (pin advances 6ce17d6 → openmp-baseline HEAD and only the carve-out commits), the `.gitignore` carve-out is explicitly enumerated, and the REQ-7 cross-reference is correct. Matches the SHUD pointer bump diff exactly (`bc919f5 → 41d9a17` = `chore(makefile): gitignore libshud.a + _libshud_obj/ build artifacts` adding `/libshud.a` + `/_libshud_obj/` to SHUD's `.gitignore`).

- F8 (REQ-3 file extensions): correctly amended. Spec at `:57` reads `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att` (per shud.cpp file-reading log evidence). Same names at `:77` for the keliya tool-correctness gate. Matches `tools/p8tune.D/verify_adjacency_keliya.py:81-84` actual file opens (`keliya.sp.mesh`, `keliya.sp.riv`, `keliya.sp.rivseg`, `keliya.sp.att`). Stale `.sa / .riv / .lake` text from prior spec version is gone.

New findings (round 2):

- Severity: minor
  Failure class: doc_contradicts_source
  Contract or invariant: spec REQ-8 L255 "each binary / text format SHALL conform to a schema documented in `tools/p8tune.D/README.md` §output-format" — the README is the canonical schema source.
  Scenario or repro: `tools/p8tune.D/README.md:166` (in §output-format under "OOM-as-data-point") enumerates the OOM reason set as `preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram`. But `tools/p8tune.D/klu_analyze_factor.cpp:274,293,304` emits `reason=preflight_after_analyze`, `reason=klu_factor_OOM`, `reason=post_factor_rss_exceeds_cn_ram`. F4 renamed the first reason (per the F4 implementer narrative "Move RSS preflight after klu_analyze") but the README was not updated. An aggregator (PR-B) author following the README to parse OOM lines would write a grep / regex for `preflight_estimate` and silently miss every `preflight_after_analyze` cell, mis-classifying them as parse-failures instead of valid `rss_overflow` data points per spec REQ-5 Scenario "OOM-as-data-point".
  Required test or evidence: Either (a) update `README.md:166` to read `reason=<preflight_after_analyze | klu_factor_OOM | post_factor_rss_exceeds_cn_ram>` to match the actual source emit, OR (b) rename the cpp emit at `klu_analyze_factor.cpp:274` to `preflight_estimate` (less preferred because preflight_after_analyze is the more descriptive name given F4 moved the check to after klu_analyze).
  Sibling surfaces: aggregator parser in PR-B (tasks.md §3.1 "scan cell results, parse klu_factor symbolic-flops + numeric-wall + fill ratio + RSS") will reference this README; mismatch propagates.
  Blocks merge: no
  Impact: PR-0 internally consistent (the actual cpp output is correct); only the README schema is stale. Caught at PR-B aggregator authoring time at latest, but the schema-is-source-of-truth invariant per REQ-8 L255 is technically violated NOW.
  Requested fix: Update `tools/p8tune.D/README.md:166` to enumerate `preflight_after_analyze` instead of `preflight_estimate`.

- Severity: suggestion (non-blocking)
  Failure class: stale_task_reference
  Contract or invariant: openspec change `tasks.md` task-to-spec cross-reference convention (per tasks.md frontmatter "Tasks below reference spec.md Requirements using REQ-N named anchors").
  Scenario or repro: `openspec/changes/p8tune-klu-spike/tasks.md:22` task 1.14 says "read `SHUD/Basins/keliya/input/*.sa / *.riv / *.lake` files" — but the F8-corrected spec at `spec.md:77` says `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att`. The actual implementation at `verify_adjacency_keliya.py:81-84` reads `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att` (matches spec). So tasks.md is the stale doc; this is pre-existing in tasks.md (not introduced by the fix delta) but was missed by F8 which only corrected the spec.md side.
  Required test or evidence: Update `tasks.md:22` to match the F8-corrected spec wording: `*.sp.mesh / *.sp.riv / *.sp.rivseg / *.sp.att`.
  Sibling surfaces: none — the spec is authoritative and validates clean; tasks.md is reader-facing only.
  Blocks merge: no
  Impact: Future reviewer reading task 1.14 to verify PR-0 acceptance would look for the wrong file extensions; trivial to resolve.
  Requested fix: One-line tasks.md edit.

Non-blocking notes:

- F1 (`--brute-force-dense` CLI flag) is a genuinely new CLI flag, but is acceptable under REQ-8 L256 "CLI arguments for the three tool binaries (`dump_adjacency`, `fd_color_jacobian`, `klu_analyze_factor`) SHALL be additive-only after PR-0 freeze: new flags MAY be added without OpenSpec amendment". PR-0 has not yet merged → no "freeze" line crossed; even after merge, this flag is additive. No drift.
- F4's `reason=preflight_after_analyze` diagnostic prefix is NOT enumerated in spec REQ-5 L141 (the spec wording mandates the diagnostic line *format* `KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N>` but doesn't enumerate reason codes). The `reason=...` suffix is additive vocabulary; spec doesn't forbid it. README is the schema source per REQ-8 L255 and that's where the inconsistency lives (covered as new finding above).
- F5 χ thresholds 30/50 are explicitly documented in spec REQ-2 Scenario "Column coloring via Welsh-Powell" at `spec.md:37-38`. No spec drift from F5's case-aware gate implementation.
- F9 (dump_adjacency.cpp comment hazard fix at L63-71) is text-only — no contract change, no spec impact.
- F10 (spike_run.sh case-name whitelist at L51-57) tightens the script-level validation but the spec REQ-4 Scenario "4-case definition" enumeration `{keliya, heihe, heihe_x4, heihe_x16}` is the source of truth; no drift.
- F11 (README determinism repro recipe at L209-217) is documentation-only addition — no spec amendment needed; reads as a §troubleshooting subsection consistent with REQ-2 Scenario "FD probe determinism".
- Determinism evidence `mac_smoke_keliya_determinism.txt` (06:03) was NOT regenerated in the fix re-run, but the current `keliya_numeric_J.bin` (06:43, post-F1/F3/F5) computes to the same hash `e122cde9...` as the prior determinism snapshot — so the determinism contract still holds despite the evidence file being older than the commit. Acceptable.
- Build-time CN_NODE_RAM_BYTES consistency check (spec REQ-5 L135) remains out-of-scope for PR-0 (aggregator lands in PR-B); confirmed unchanged from round 1.
- openspec strict validation: `openspec validate p8tune-klu-spike --strict --no-interactive` returns `Change 'p8tune-klu-spike' is valid`. F6/F7/F8 amendments do not break validation.
