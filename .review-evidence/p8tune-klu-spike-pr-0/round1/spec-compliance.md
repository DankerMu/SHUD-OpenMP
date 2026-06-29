Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Summary: PR-0 file boundary + zero-source-patch + adjacency tool-correctness all CLEAN; two material spec deviations — (a) dense_fd_cross_check.py does NOT execute the required brute-force dense FD comparison (hardcoded max_rel_err=0), (b) χ ≤ 30 keliya sanity bound is reported but NOT programmatically asserted as a gate.

Invariant Matrix Coverage:
- REQ-1 zero-source-patch: covered — `git diff 6ce17d6..bc919f5 -- src/ include/` returns empty; SHUD/Makefile diff is purely additive (single libshud.a block at L619-660 + paired clean-rule extension at L673-675; existing `shud`/`shud_omp`/`shud_asan`/`smoke_*`/`test_adjacency_fallback` targets at L506/523/245/551/587/610 untouched).
- REQ-1 carve-out 2-hunk interpretation: covered — both hunks belong to the same logical libshud.a unit (target body + paired clean-rule). Spec wording "additive `libshud.a` archive target" is satisfied; no existing target modified.
- REQ-2 Welsh-Powell coloring: covered — `fd_color_jacobian.cpp:157` calls `PartialDistanceTwoColoring("SMALLEST_LAST", "COLUMN_PARTIAL_DISTANCE_TWO")` via `BipartiteGraphPartialColoringInterface`; this is the ColPack idiom equivalent to spec's "JacobianGraphColoring + DISTANCE_TWO". χ=16 reported, ≤ 30 keliya bound met (data-wise).
- REQ-2 FD probe via rhs_core: covered — `fd_color_jacobian.cpp:259-261` mirrors `shud.cpp:118-121` (`new Model_Data + loadinput + initialize`); `:315/:341` call `MD->rhs_core(Y, DY, t0, ExecPolicy::Serial)`; `:319-322` ε formula = `sqrt(DBL_EPSILON) * (|Y[k]| + 1)` (CPR standard).
- REQ-2 FD probe determinism: covered — `mac_smoke_keliya_determinism.txt` shows two runs sha256 = `e122cde9...` (PASS).
- REQ-3 in-process Init + AoS walk: covered — `dump_adjacency.cpp:408-410` calls lowercase `loadinput()/initialize()`; walks `MD->Ele[i].nabr/lakenabr` (:175, :223), `MD->Riv[i].down/toLake/frLake` (:199, :236, :240), `MD->RivSeg[k]` (:186), `MD->Ele[i].iLake` (:209).
- REQ-3 rivNode NOT dereferenced: covered — grep finds only the doc comment at `dump_adjacency.cpp:21` warning against it; no dereference site.
- REQ-3 5-block CSC schema: covered — header per README §output-format matches (magic ADJC + version + 4 dims + total_nnz + 5×5 per_block_nnz + col_ptr + row_idx); row blocks indexed via `block_of_row()` at `dump_adjacency.cpp:103` per `NY = 3·NumEle + NumRiv + NumLake`.
- REQ-3 keliya tool-correctness exact-match: covered — `verify_adjacency_keliya.py` is independent (reads `.sa/.riv/.lake/.rivseg/.att`, no C++ link); `mac_smoke_keliya_verify_nnz.tsv` shows ALL 25 cells PASS, total_nnz=10255 matches.
- REQ-3 dense FD cross-check ≤ 1e-6: MISSING — see Critical finding below.
- REQ-5 Fill / RSS / Wall headers: covered — `cn_node_ram.h:32` constexpr, `spgmr_baseline_walls.h:37` constexpr; RSS pre-flight at `klu_analyze_factor.cpp:228-239`; wall baseline pinned at 0.226579 = 1489.76/6575 per epic #362 PR-D #373 (TSV `maxl_sweep_summary.tsv` in evidence).
- REQ-5 OOM-as-data-point: covered — `klu_analyze_factor.cpp:236/272/283` emits `KLU_OOM_DETECTED case=… ordering=… btf=… peak_rss_bytes=…` to stdout + exit 0; 3 reason codes (preflight_estimate / klu_factor_OOM / post_factor_rss_exceeds_cn_ram). Aggregator-friendly.
- REQ-7 PR-0 file boundary: covered — 26 files match allowed list; forbidden files (spike_array.sbatch / run_cell.sh / aggregate_klu_spike.sh / render_verdict.sh / docs/adr/0005 / docs/p8tune/klu_spike_verdict.md / SHUD/src/) all absent (verified via ls + grep).
- REQ-7 minor amendment claim: out-of-scope — brief mentions ".gitignore + dense_fd_cross_check.py minor amendments documented in PR description" but `git diff` on `.gitignore` at outer repo is empty; only NEW `tools/p8tune.D/.gitignore` (a fresh file) and `dense_fd_cross_check.py` (NEW file) — both inside `tools/p8tune.D/`, both implicitly allowed under "tools/p8tune.D/*". Not a boundary violation.
- REQ-8 README §output-format: covered — README §output-format documents all 3 schemas (CSC, NDNJ J binary, klu_factor cell_summary KV) + format-version footer "v1 (2026-06-28)" + CLI usage in each .cpp print_usage + spec REQ-8 stability paragraph.
- Master plan §P8-tune.D status flip: covered — L2447 now reads `[OPEN, executing PR-0 #380] per 2026-06-28`, matches spec REQ-7 PR-0 list bullet.
- Design D2 §M=I-γ·J: out-of-scope (implementation-detail) — spike-internal test matrix construction (`klu_analyze_factor.cpp:189-220`, γ=1.0) is reasonable; the test matrix M's pattern is what KLU factors against, and the documented rationale (fill_ratio/numeric_wall scale with pattern, not diagonal magnitude) is sound. No PR-A spec amendment required if PR-A reuses the same wrapper.

Findings:

- Severity: critical
  Failure class: spec_requirement_not_implemented
  Contract or invariant: spec REQ-3 Scenario "keliya tool-correctness gate via independent ground-truth reference" — "AND in addition, the FD-probed numeric J for keliya SHALL be cross-checked against a brute-force dense FD on keliya (NumY ≈ 1.5K is small enough for an O(NumY) column-by-column dense baseline) — relative error ≤ 1e-6 per nonzero entry … AND PR-0 SHALL NOT merge without keliya tool-correctness PASS (both nnz exact-match and dense FD cross-check)"
  Scenario or repro: Read `tools/p8tune.D/dense_fd_cross_check.py:139-144` — `max_rel_err = 0.0` is HARDCODED. The script's docstring at `:18-34` explicitly admits "We don't actually need to re-run a full dense FD"; instead it performs finite/bounded checks on the colored-FD output alone. `mac_smoke_keliya_dense_fd_cross_check.tsv` shows `max_rel_err = 0.0e+00` for all 25 blocks — this is the hardcoded constant, not an actual measurement against an independent dense baseline.
  Required test or evidence: Either (a) implement a true brute-force dense FD path that re-probes each column k with v_c = e_k (CHI=NumY, NumY=1785 columns × 2 rhs_core calls = ~3570 calls, tractable on Mac in ≤1 min), compute J_dense[i,k] = (DY_plus[i] - DY[i]) / ε, and emit per-nonzero rel_err = |J_color[i,k] - J_dense[i,k]| / max(|J_dense[i,k]|, 1e-12) with max ≤ 1e-6 OR (b) file an OpenSpec amendment downgrading this clause to the finite-bounded self-consistency check actually performed (and update the dense_fd_cross_check.py docstring / TSV verdict semantics).
  Sibling surfaces: none — keliya-specific by spec, no parallel pattern elsewhere in repo.
  Blocks merge: yes
  Impact: REQ-3's last bullet ("SHALL NOT merge without … dense FD cross-check") is unmet. Without an independent dense baseline, a future regression in the colored-FD code path (e.g. wrong ε in Y_perturbed reset, or accidental column-aggregation across colors) would silently pass the current sanity gate.
  Requested fix: Add a dense FD probe (column-by-column) inside `dense_fd_cross_check.py` (or a sibling tool) that loads the CSC, invokes `fd_color_jacobian` with a hypothetical `--single-color k` mode OR re-implements rhs_core probing via a thin C wrapper, and emits per-entry rel_err. Update the TSV's `max_rel_err` column to reflect actual measured values. Re-run keliya smoke and commit fresh evidence.

- Severity: major
  Failure class: spec_assertion_not_enforced
  Contract or invariant: spec REQ-2 Scenario "Column coloring via Welsh-Powell" — "for the keliya tool-correctness-gate case (NumY=1.5K, simpler mesh), χ SHALL be bounded ≤ 30 (tighter sanity bound …). **This bound is asserted by the Mac smoke gate (task 1.13)**."
  Scenario or repro: `fd_color_jacobian.cpp:228-237` only PRINTS `chromatic_number=16`; no `if (chi > 30) { abort with exit 1; }` gate exists. `spike_run.sh:90-91` invokes fd_color_jacobian but never parses χ or compares to threshold 30. The `mac_smoke_keliya_chi.log` evidence shows χ=16 (passes the bound by data), but the assertion is a SOFT spec gate — a future change to ColPack or to the adjacency walk that bumps χ>30 would silently pass the smoke without alarm.
  Required test or evidence: Either (a) add an explicit gate in `fd_color_jacobian.cpp` after `:228` — e.g. when invoked with `--report-chi-only` AND `--case keliya`, assert `chi <= 30` and `exit 1` on violation OR (b) add a shell-level grep gate in `spike_run.sh` (or a new mac_smoke driver script) extracting `chromatic_number=N` from fd_color stdout and exiting non-zero if N > 30 (case-conditional on keliya).
  Sibling surfaces: similar enforcement needed for production cases χ ≤ 50 bound (spec REQ-2 same Scenario, third bullet) — but that lives in PR-A, not PR-0.
  Blocks merge: no (data passes; only the future-regression guard is missing)
  Impact: REQ-2's "asserted by the Mac smoke gate" clause is unmet at the code-gate level. Verdict is data-correct now but not regression-armored.
  Requested fix: Add the explicit gate in `fd_color_jacobian.cpp` `--report-chi-only` exit path (before the existing `return 0;` at L237) — case-aware: keliya bound 30, heihe/heihe_x4/heihe_x16 bound 50 per spec.

- Severity: minor
  Failure class: scope_compliance_under_documented
  Contract or invariant: spec REQ-7 Scenario "PR-0 tool PR boundary" allowed-list does NOT explicitly enumerate `tools/p8tune.D/.gitignore` or `tools/p8tune.D/dense_fd_cross_check.py`, though both fall under "tools/p8tune.D/*.cpp + tools/p8tune.D/Makefile + … + tools/p8tune.D/verify_adjacency_keliya.py + tools/p8tune.D/README.md" implicit envelope.
  Scenario or repro: `tools/p8tune.D/.gitignore` (21 lines, NEW) and `tools/p8tune.D/dense_fd_cross_check.py` (185 lines, NEW) are committed but not enumerated in the spec's allowed list. The PR description should explicitly note these as PR-0 in-scope additions (since both are tooling support, not forbidden-list items).
  Required test or evidence: Either amend the PR description to call out both files OR amend the OpenSpec change file boundary list (one-line addendum) to enumerate them.
  Sibling surfaces: none.
  Blocks merge: no
  Impact: Future Phase-4 reviewer-pack scoping may flag these as out-of-list; trivial to resolve via PR description note.
  Requested fix: Add to PR description body: "Also adds tools/p8tune.D/.gitignore (build-artifact filter) and tools/p8tune.D/dense_fd_cross_check.py (keliya cross-check helper); both implicit under spec REQ-7 'tools/p8tune.D/*' envelope."

Non-blocking notes:
- `klu_analyze_factor.cpp:189-220` adds `M = I - γ·J` (γ=1.0) test matrix construction not literally in spec REQ-5 wording. Implementer-reported deviation #5; mitigation rationale (`klu_factor` needs non-singular matrix; pattern stays identical) is sound and documented in source comments. No spec amendment needed.
- README §output-format format-version footer "v1 (2026-06-28)" present (REQ-8 last clause satisfied).
- 4-case × 4-ordering matrix (REQ-4) is out-of-scope for PR-0 (lives in PR-A); spike_run.sh correctly accepts {keliya|heihe|heihe_x4|heihe_x16} × {natural|amd|colamd} × {0|1} = 24 combos but enforces this only via CLI validation, not 16-cell array generation. Correct for PR-0 scope.
- `cn_node_ram.h` provenance complete (cn14 job 9755 2026-06-28T09:57:21Z, 181179840 kB → 185528156160 bytes, evidence at `cn_ram_probe.log`).
