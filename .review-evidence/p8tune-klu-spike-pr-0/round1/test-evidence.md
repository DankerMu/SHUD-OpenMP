Reviewer agent: review-test-evidence
Review round: round 1
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Summary: All 16 Mac-smoke acceptance gates report PASS in the evidence files, the boundary diff matches REQ-7 and the verify_adjacency_keliya Python ref is genuinely independent — but the "dense FD cross-check" never actually performs the brute-force dense FD comparison the spec REQ-3 scenario mandates, so sub-gate (d) is unverified.

Findings:

- Severity: critical
  Failure class: Untested / falsified evidence
  Contract or invariant: spec REQ-3 Scenario "keliya tool-correctness gate via independent ground-truth reference" — "the FD-probed numeric J for keliya SHALL be cross-checked against a brute-force dense FD on keliya (NumY ≈ 1.5K is small enough for an O(NumY) column-by-column dense baseline) — relative error ≤ 1e-6 per nonzero entry"
  Scenario or repro: Inspect `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/dense_fd_cross_check.py:100-128, 140-150`: `cross_check()` only reads `keliya_numeric_J.bin` and computes max_abs / n_finite_viol; no second J is ever produced via dense FD probing. `write_tsv()` then hardcodes `max_rel_err = 0.0` (L144) and prints "PASS … (distance-2 invariant)" (L178). Evidence `mac_smoke_keliya_dense_fd_cross_check.tsv` shows every block reports `max_rel_err = 0.0e+00` because the value is a literal constant, not a measurement. Repro: `grep 'max_rel_err = 0.0' tools/p8tune.D/dense_fd_cross_check.py` returns the hardcoded line; nothing in the script invokes a 1785-column dense probe nor compares two J arrays. Furthermore the docstring (L31-34, L102-106) admits "We don't actually need to re-run a full dense FD" — directly contradicting the SHALL clause.
  Required test or evidence: Author a true dense FD reference path that calls `MD->rhs_core(Y, DY_base, …)` once and then `MD->rhs_core(Y + ε·e_k, DY_pert_k, …)` for each k ∈ [0, NumY) with ε identical to the colored-FD formula (`sqrt(DBL_EPSILON)·(|Y[k]|+1)`). Then compute `J_dense[i,k] = (DY_pert_k[i] - DY_base[i]) / ε` for the CSC pattern entries and emit per-block `max_rel_err = max |J_color - J_dense| / max(|J_color|,|J_dense|,eps)`. NumY=1785 ⇒ ~1786 RHS evals, seconds on keliya 90-day. Re-run Mac smoke; replace the TSV; ensure each block's measured max_rel_err ≤ 1e-6.
  Sibling surfaces: spec REQ-2 Scenario "FD probe via existing SHUD rhs_core" (same ε formula assumption); README §troubleshooting "Non-deterministic numeric J binary" (currently presumes determinism, no numerical-correctness gate); PR-A acceptance — without a real cross-check on PR-0, PR-A's per-cell numeric J binaries inherit an unverified correctness baseline.
  Blocks merge: yes
  Impact: Sub-gate (d) is the only correctness check on the actual FD numeric values; the colored-FD output could differ from a true dense FD (e.g., wrong ε, wrong column-mask construction, color/row index miscoupling) and the smoke would still report PASS. PR-A then ships 16 cells of numeric J binaries that no one has ever validated against a ground truth — an undetected bug here propagates into the GO/NO-GO verdict at PR-B.
  Requested fix: Implement actual dense FD probing in `dense_fd_cross_check.py` (or a new `dense_fd_baseline` C++ tool that links `libshud.a` the same way `fd_color_jacobian.cpp` does — needed because the Python script has no SHUD bindings); compute real per-entry rel err; re-emit the TSV with non-zero max_rel_err values; commit the regenerated evidence under the same path.

- Severity: major
  Failure class: Untested code path
  Contract or invariant: spec REQ-5 Scenario "OOM-as-data-point" — "the tool SHALL exit with status 0 (NOT non-zero) AND SHALL emit a diagnostic line `KLU_OOM_DETECTED ...`"
  Scenario or repro: PR-0 Mac smoke only exercises `peak_rss_bytes=3145728` (3.1 MB) vs `CN_NODE_RAM_BYTES=185528156160` (173 GiB) — six orders of magnitude below threshold (see `mac_smoke_keliya_klu_factor.log:12`). None of the three `KLU_OOM_DETECTED` branches in `tools/p8tune.D/klu_analyze_factor.cpp` (L236 preflight_estimate, L272 klu_factor_OOM, L283 post_factor_rss_exceeds_cn_ram) execute. PR-A will be the first time these paths run, and PR-A's matrix is exactly where OOM is hypothesized to occur (heihe_x16). If the diagnostic prints to stderr instead of stdout, or exits non-zero, or the format string drifts from the aggregator's regex, PR-A loses real signal.
  Required test or evidence: Add a smoke that forces the OOM path on Mac — e.g. a hidden CLI flag `--force-oom={preflight|factor|post}` or a build-time `-DCN_NODE_RAM_BYTES_OVERRIDE=1024`, plus a smoke run that captures stdout and greps for the exact `KLU_OOM_DETECTED case=…` prefix. Attach the captured output as evidence (e.g., `mac_smoke_keliya_klu_oom_simulated.log`).
  Sibling surfaces: PR-B aggregator regex (tasks 3.1, 3.3); REQ-5 "rss_overflow data point" classification; PR-A acceptance criterion "any OOM cell reported with KLU_OOM_DETECTED diagnostic".
  Blocks merge: no (PR-0 spec scope is keliya smoke), but flag for PR-A to gate on
  Impact: First-use bugs in the OOM diagnostic format would silently lose the headline data point of the entire spike — heihe_x16 OOM is exactly the verdict-deciding cell.
  Requested fix: Add an `--oom-self-test` mode + a Mac smoke that exercises it and captures the diagnostic line, before PR-A submits its array.

- Severity: minor
  Failure class: Spec phrasing vs implementation
  Contract or invariant: spec REQ-3 Scenario keliya gate — "reads keliya `.sa/.riv/.lake` files" (matches tasks 1.14 wording); implementation reads `.sp.mesh / .sp.riv / .sp.rivseg / .sp.att`
  Scenario or repro: `verify_adjacency_keliya.py:80-89` reads `keliya.sp.mesh`, `keliya.sp.riv`, `keliya.sp.rivseg`, `keliya.sp.att`. The spec/task wording references `.sa/.riv/.lake` — neither `.sa` nor `.lake` files exist for keliya (`ls SHUD/Basins/keliya/input/keliya/` lacks them). Behaviorally the Python ref is correct (independent codepath, exact-match nnz vs C++ dump), but the spec text disagrees with the actual file extensions used.
  Required test or evidence: None — evidence shows it works.
  Sibling surfaces: tasks.md 1.14; spec REQ-3 first Scenario clause.
  Blocks merge: no
  Impact: Future maintainers may chase the wrong file names when adding lake/aux-coupling validation for heihe_x16 (which DOES have lakes).
  Requested fix: One-line errata in spec / tasks: replace `.sa/.riv/.lake` with `.sp.mesh/.sp.riv/.sp.rivseg/.sp.att` to match actual SHUD AutoSHUD output convention. Or amend the docstring of `verify_adjacency_keliya.py` to note the spec-vs-impl drift.

- Severity: minor
  Failure class: Determinism repro recipe undocumented
  Contract or invariant: README §troubleshooting "Non-deterministic numeric J binary" exists but does NOT document the exact 2× `fd_color_jacobian` + `sha256sum` recipe that produced the PASS line in `mac_smoke_keliya_determinism.txt`
  Scenario or repro: `mac_smoke_keliya_determinism.txt` shows two sha256 hashes (identical) + `DETERMINISM: PASS`. The README lists symptoms but not the verification sequence. A future contributor needs the exact commands to re-trigger the gate after a code change.
  Required test or evidence: Append to README §troubleshooting a "Repro" subsection: `cd SHUD/Basins/keliya && fd_color_jacobian --case keliya && sha256sum keliya_numeric_J.bin > /tmp/h1 && rm keliya_numeric_J.bin && fd_color_jacobian --case keliya && sha256sum keliya_numeric_J.bin > /tmp/h2 && diff /tmp/h1 /tmp/h2`.
  Sibling surfaces: spec REQ-2 Scenario "FD probe determinism"; PR-A re-runs.
  Blocks merge: no
  Impact: Determinism regressions discovered at PR-A would require reverse-engineering the gate.
  Requested fix: 3-line repro recipe in README.

Non-blocking notes:
- PR boundary diff (14 source + 11 evidence files + 2 minor amendments: top-level Makefile, `tools/p8tune.D/.gitignore`) matches REQ-7 "PR-0 tool PR boundary". No forbidden file (`spike_array.sbatch`, `run_cell.sh`, aggregator, render, ADR-0005, docs/p8tune, `SHUD/src/`) appears. SHUD pointer bumps from `6ce17d6` → `bc919f5` — review-blocker only if the bumped SHUD HEAD contains anything beyond the `libshud.a` carve-out target (verifier should check).
- Master plan flip line at L2447 correct: `[ACTIVE TRIGGER]` → `[OPEN, executing PR-0 #380]` matches tasks 1.16 and the PR/issue cross-reference.
- SPGMR baseline derivation reproducible from `maxl_sweep_summary.tsv` rows 32-34: sorted [1484.22, 1489.76, 1498.68] → median 1489.76 / nst 6575 = 0.226578... → header value `0.226579` (consistent to 6 sig figs).
- cn-RAM constant matches probe: `MemTotal: 181179840 kB × 1024 = 185528156160 bytes` (cn14, Slurm 9755, 2026-06-28T09:57:21Z) → `cn_node_ram.h:32` `CN_NODE_RAM_BYTES = 185528156160ULL`.
- Per-5-block dump sum: 1868+484+484+534 + 484+484+484 + 484+484+1868+534 + 534+534+995 = 10255 ✓ matches `total_nnz=10255`; verify_nnz TSV exact-matches all 9 nonzero + 16 zero blocks (zero off-by-one).
- χ=16 ≤ 30 PASS (REQ-2 keliya tighter bound).
- Mac smoke logs 116-117 show 2 colors (cols=198 + cols=104), not all 16 — minor logging gap, not a finding.
