Reviewer agent: review-spec-compliance
Review round: round 3
Reviewed head SHA: 7fc325b (fix commit on top of 50d2a4b round-2 head)
Fix delta inspected: `git diff 50d2a4b..7fc325b` — 3 files (README.md, klu_analyze_factor.cpp, mac_smoke_keliya_klu_ordering_matrix.log).

Summary: Both round-2 spec-compliance findings (e02 README OOM-reason drift, e03 stale `.sa/.riv/.lake` in tasks.md/design.md) ADDRESSED. The G1 implementation change (skip preflight when ordering ≠ AMD) does NOT introduce new spec drift — the new `[klu] PREFLIGHT_AFTER_ANALYZE skipped (...)` line is internal diagnostic stdout and is NOT enumerated by any SHALL clause in REQ-5 or REQ-8. Gap sweep across REQ-1..REQ-8 surfaces no new spec violations introduced by the G1/G2/G3 fix delta. Verdict: APPROVE (no new findings).

---

Round 2 finding verification:

- e02 (minor, doc_contradicts_source, README OOM-reason enumeration drift): ADDRESSED
  - Evidence: `git diff 50d2a4b..7fc325b -- tools/p8tune.D/README.md` shows L166 changed from `preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram` to `preflight_after_analyze | klu_factor_OOM | post_factor_rss_exceeds_cn_ram`.
  - Direct read of `tools/p8tune.D/README.md:166` confirms: `KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=<preflight_after_analyze | klu_factor_OOM | post_factor_rss_exceeds_cn_ram>`.
  - Source-side reasons emitted by `klu_analyze_factor.cpp` at L282 (`reason=preflight_after_analyze`), L306 (`reason=klu_factor_OOM`), L317 (`reason=post_factor_rss_exceeds_cn_ram`) — all 3 now match the README enumeration exactly. REQ-8 L255 "binary / text format SHALL conform to a schema documented in README.md §output-format" invariant restored.

- e03 (suggestion, stale_task_reference, tasks.md L22 + design.md L71 stale `.sa/.riv/.lake`): ADDRESSED
  - openspec/changes/ is gitignored per project convention (`.gitignore:13`); fix lives in working tree only. Verified by direct grep on filesystem.
  - `openspec/changes/p8tune-klu-spike/tasks.md:22` now reads `read SHUD/Basins/keliya/input/*.sp.{mesh,riv,rivseg,att} files` (matches spec.md:77 + implementation `verify_adjacency_keliya.py:81-84` actual file opens).
  - `openspec/changes/p8tune-klu-spike/design.md:71` now reads `External file reader (mesh .sp.mesh / river .sp.riv / river-segment .sp.rivseg / attribute .sp.att files)` (matches spec.md:57).
  - grep sweep for residual `\.sa|bare \.riv|\.lake` across all 3 openspec fixture files returns only the legitimate `.sp.{mesh,riv,rivseg,att}` substring matches inside the corrected text. No stale references remain.
  - openspec strict validation (round 2 confirmed PASS) unaffected — extension token is inside prose, not a structural element.

---

G1 spec-drift analysis (new code path: `klu_analyze_factor.cpp:287-292` "skipped (AMD-only)" diagnostic line):

The G1 fix adds a non-AMD branch that emits one new stdout line:
```
[klu] PREFLIGHT_AFTER_ANALYZE skipped (ordering=<O> lnz=<L> unz=<U>; AMD-only); relying on post-factor RSS check
```

Spec analysis against SHALL clauses that could plausibly enumerate diagnostic output:

1. REQ-5 Scenario "OOM-as-data-point" (spec.md:138-143): Mandates ONLY the `KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N>` line on OOM conditions. Says nothing about non-OOM diagnostics, preflight traces, or skip-conditions. The new "skipped" line fires on a non-OOM path → out of scope of this scenario. No drift.

2. REQ-5 Scenario "Per-case axis machine-readable" (spec.md:155-172): Defines aggregator output schema (KV blocks in `aggregate_verdict.txt`). This is PR-B scope, downstream of `klu_analyze_factor.cpp` stdout. The new line does not enter aggregator KVs. No drift.

3. REQ-5 Scenario "Numeric factor determinism" (spec.md:181-186): Requires symbolic-flops / nnz(L+U) / numeric-wall determinism. The "skipped" line is informational text, not a determinism-tracked field. Verified empirically via `mac_smoke_keliya_klu_ordering_matrix.log:9` (amd cell: real preflight numbers) vs L24/L39 (colamd/natural cells: skipped line with stable `lnz=-1 unz=-1` sentinel values). Deterministic across runs. No drift.

4. REQ-8 Scenario "Tool output format stability" (spec.md:252-257):
   - L255: "each binary / text format SHALL conform to a schema documented in `tools/p8tune.D/README.md` §output-format". The README §output-format L150-167 enumerates only the `cell_summary` KV block + the `KLU_OOM_DETECTED` line. It does NOT close-enumerate other diagnostic prefixes (existing `[klu] PREFLIGHT_HINT ...` L236 and `[klu] test matrix M = ...` L221 are likewise undocumented in the README schema, and were accepted in round 1/2). The schema is implicitly open-set for non-data-bearing `[klu] ...` trace lines.
   - L256: "new flags MAY be added without OpenSpec amendment; existing flags MUST NOT be removed or renamed without an OpenSpec change". The G1 fix changes no CLI flag, only stdout output. Not applicable.
   - No drift.

5. REQ-5 Scenario "Fill axis threshold" (spec.md:122-128): Aggregator fill ratio = `nnz(L+U) / nnz(A)` is sourced from `Numeric->lnz + Numeric->unz` (the post-factor numeric struct), NOT from the Symbolic preflight estimate. Cross-checked at `klu_analyze_factor.cpp:326` (`fill_ratio = (Numeric->lnz + Numeric->unz) / std::max(Anz, 1.0)`). For non-AMD cells the cell_summary still emits a VALID `fill_ratio` (e.g. `mac_smoke_keliya_klu_ordering_matrix.log:31` colamd fill_ratio=4.6414, L46 natural fill_ratio=205.9479). The fill axis is unaffected by the preflight skip. No drift.

Conclusion: G1 introduces no new SHALL violation. The skip-emission pattern is consistent with the existing `fd_color_jacobian.cpp:234,316` "skipped" diagnostic vocabulary. No SHALL clause needs amendment.

---

Gap sweep (REQ-1..REQ-8 clauses NOT previously checked):

Spec contains 114 SHALL clauses. Previously verified by round 1/2: REQ-1 (PR boundary), REQ-2 (χ thresholds), REQ-3 (keliya tool-correctness gate + dense FD cross-check + file extensions), REQ-5 (OOM-as-data-point reason enumeration, fill/RSS/wall axes), REQ-7 (PR-0 envelope incl. .gitignore + dense_fd_cross_check.py), REQ-8 (CLI additive flags).

New surface checked in round 3:

- REQ-2 Scenario "FD probe determinism" (spec.md:49-53): Mac smoke recorded `fd_color_jacobian keliya` twice with sha256sum equal in round-1 evidence `mac_smoke_keliya_determinism.txt`. Round 3 fix touches only `klu_analyze_factor.cpp` (not `fd_color_jacobian.cpp`), so determinism contract is preserved. NO drift.
- REQ-3 Scenario "5-block adjacency CSC output" (spec.md:67-72): Adjacency dump deterministic + per-block nnz header — fix delta does not touch `dump_adjacency.cpp`. NO drift.
- REQ-4 Scenarios "4-case definition" / "4-combo definition" / "Slurm 三铁律" / "Pre-submission environment gate" (spec.md:86-116): Out of PR-0 scope (PR-A sweep). No fix-delta touchpoint. NO drift.
- REQ-5 Scenario "Wall axis threshold" (spec.md:145-153): Pinned baseline at `tools/p8tune.D/spgmr_baseline_walls.h` — fix delta does not touch this header. NO drift.
- REQ-6 ADR 4-branch decision tree (spec.md:188-206): PR-B scope. No fix-delta touchpoint. NO drift.
- REQ-7 Scenarios "PR-A/B/C boundary" + "Time budget" (spec.md:221-246): All non-PR-0 scope, file envelopes for downstream PRs. NO drift introduced.

No SHALL clause uncovered as previously-unchecked-and-now-violated. The fix delta's scope (1 cpp file + 1 README line + 1 evidence log + 2 openspec fixture lines) is too narrow to plausibly drift outside REQ-5/REQ-8/REQ-3, all of which are explicitly checked above.

---

openspec strict validation: not re-run in round 3 (round 2 confirmed PASS at `Change 'p8tune-klu-spike' is valid`; fix delta does not modify spec.md → validation result unchanged).

---

New findings (round 3): NONE.

Verdict: APPROVE — both round 2 findings ADDRESSED; G1 introduces no new spec drift; no SHALL clauses uncovered by gap sweep. Ready to merge from spec-compliance perspective.
