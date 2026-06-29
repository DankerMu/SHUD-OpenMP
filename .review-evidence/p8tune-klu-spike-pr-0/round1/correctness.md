```text
Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Summary: Spike tool is functionally correct end-to-end (keliya evidence corroborates) and the C++ memory/type/ordering invariants hold; one major finding: dense_fd_cross_check.py is NOT an independent dense-FD cross-check (it inspects only the colored-FD output and hardcodes max_rel_err=0), which weakens REQ-3's "tool-correctness gate via independent ground-truth reference" for the FD-numeric axis.

Findings:
- Severity: major
  Failure class: spec gate not implemented (false-pass cross-check)
  Contract or invariant: spec REQ-3 Scenario "keliya tool-correctness gate via independent ground-truth reference" — "the FD-probed numeric J for keliya SHALL be cross-checked against a brute-force dense FD on keliya (NumY ≈ 1.5K is small enough for an O(NumY) column-by-column dense baseline) — relative error ≤ 1e-6 per nonzero entry"
  Scenario or repro: `tools/p8tune.D/dense_fd_cross_check.py` lines 18-34 + 140-147 explicitly admit "We don't actually need to re-run a full dense FD" and hardcode `max_rel_err = 0.0`. The script ONLY reads the colored-FD `_numeric_J.bin` and per-block tallies max-abs / NaN/Inf / zero counts. No independent FD probe (would require either driving `fd_color_jacobian` with χ=NumY=1785 single-column probes, or running an independent Python brute-force FD against rhs_core). Evidence TSV `mac_smoke_keliya_dense_fd_cross_check.tsv` reports `max_rel_err 0.0e+00` for every block — by construction, not by measurement. This means a buggy colored-FD output that produces self-consistent but wrong numeric J values would PASS the gate.
  Required test or evidence: Either (a) implement an independent dense-FD reference (sub-second for NumY=1785: shell out to `fd_color_jacobian` with a forced-NumY-color path, or write a Python-level rhs_core re-implementation, or add a `--single-column-probe k` CLI to fd_color_jacobian and invoke it for every k), then compute max_rel_err = max |dut - ref| / max(|ref|, atol) with atol=1e-12; OR (b) explicitly weaken the spec REQ-3 line to acknowledge the cross-check is structural-only.
  Sibling surfaces: heihe / heihe_x4 / heihe_x16 cells in PR-A — same gate semantic will be transitively weak there too.
  Blocks merge: no (PR-0 is a tool-spike; PR-B aggregator depends on this gate having teeth; should fix before PR-A starts producing verdict data)
  Impact: REQ-3 gate is structurally hollow for FD numeric correctness; only the adjacency-pattern gate (verify_adjacency_keliya.py) is genuinely independent.
  Requested fix: Add a real dense-FD pathway. Simplest path: add `--brute-force-dense` mode to `fd_color_jacobian.cpp` that iterates k=0..NumY-1 and probes one column at a time (≈ 1785 rhs_core calls, ~30s on Mac), emits a parallel `_numeric_J_dense.bin`. Have dense_fd_cross_check.py load BOTH binaries and compute real elementwise `max |dut - ref| / max(|ref|, 1e-12)`.

- Severity: minor
  Failure class: stale-comment hazard / fragile global-symbol dependency
  Contract or invariant: REQ-3 implementation note "in-process libshud.a Init walk"; comment at dump_adjacency.cpp:72-75 claims "shud.o IS pulled in by Model_Data initialize() (via the g_numa_first_touch_enabled extern)"
  Scenario or repro: The spike depends on `globalY` being non-null when `MD->LoadIC()` (fd_color_jacobian.cpp:266) calls `Sub2Global` (MD_initialize.cpp:131) which writes `globalY[0..NumY)`. Confirmed allocation chain: `MD->initialize()` → `malloc_Y()` → `globalY = new double[NumY]` (Model_Data.cpp:87). So `globalY` IS valid. Separately, `shud.cpp:124` also has `globalY = new double[NY]` inside `SHUD()` — UNUSED by the spike but the symbol `g_numa_first_touch_enabled` (defined at shud.cpp:50) is referenced by MD_rhs_core.cpp:69 as `extern int`. With libshud.a (archive linkage), the linker pulls in shud.o ONLY if some symbol it defines is referenced. The spike's chain: rhs_core → references `g_numa_first_touch_enabled` extern → pulls in shud.o → also defines `globalY` (and `timeNow`). If a future refactor moves `g_numa_first_touch_enabled` out of shud.cpp, the spike will silently fail to link `globalY`/`timeNow` and the comment block at dump_adjacency.cpp:66-75 becomes a trap that says "remove the duplicates" — leading a future maintainer to actively break the build.
  Required test or evidence: Either (a) move `globalY` and `timeNow` extern definitions into a dedicated `shud_globals.cpp` TU that the spike explicitly links (no chance of dead-stripping), or (b) replace the comment at dump_adjacency.cpp:66-75 with a static_assert / link-time check that asserts these globals are reachable.
  Sibling surfaces: fd_color_jacobian.cpp:286-293 uses `extern double *globalY` + nullptr-check; klu_analyze_factor.cpp doesn't depend on these globals (reads its own binary).
  Blocks merge: no
  Impact: Build fragility against future SHUD refactors; comment block is confusingly written as "if you see X, remove Y" which is the opposite of safe advice.
  Requested fix: Replace dump_adjacency.cpp:66-75 comment with a one-line note: "globalY + timeNow are defined in shud.cpp and transitively pulled into libshud.a via MD_rhs_core.cpp's `extern int g_numa_first_touch_enabled` reference. Do NOT add duplicate definitions here." Optionally add a `shud_globals.cpp` carve-out.

- Severity: minor
  Failure class: misleading log evidence — possibly spec-relevant
  Contract or invariant: REQ-2 implicit assumption that FD numeric J entries are meaningful at IC; fd_color_jacobian.cpp line 394 `n_zero=N` printf
  Scenario or repro: Evidence `mac_smoke_keliya_fd_color.log` reports `total_nnz=10255 n_zero=7757` — 75.6% of FD-probed J entries are exactly zero at t0. The spike's design.md notes IC has "~40% zero diagonals" but the actual measured rate is much higher across all entries. dense_fd_cross_check.tsv confirms `surf×unsat = 0/484 nonzero`, `gw×surf = 0/484 nonzero` — entire 5-block sub-matrices are zero. For the KLU verdict axes (fill_ratio, numeric_factor_wall), this is benign — KLU works on the SPARSITY PATTERN (CSC col_ptr/row_idx) which the test matrix `M = I - γ·J` always carries; the diagonal injection `Ax[p] += 1.0` ensures M has no exact-zero structural entries on the diagonal. But it does mean the spike's FD J is structurally OVER-COMPLETE for this case — it has 7757 "wasted" pattern entries that KLU will still factor against. PR-A's KLU walls on Mac are therefore an UPPER bound on what a "tight" J pattern would produce; this caveat should be visible in the aggregator output and the spec.
  Required test or evidence: Either (a) document that the FD-probe is the OUTER bound (super-set) of the true Jacobian sparsity — and downstream KLU verdicts should be interpreted as conservative; OR (b) post-process the J binary to PRUNE structural zeros and re-run KLU on the pruned pattern as a second data point.
  Sibling surfaces: aggregate_klu_spike.sh in PR-B will inherit this caveat per cell.
  Blocks merge: no
  Impact: Verdict thresholds (fill_ratio, wall) computed against a 5× over-padded sparsity pattern; may produce false NO-GO verdicts at heihe_x4/heihe_x16 if pattern padding pushes fill_ratio above the 1.5× SPGMR break-even.
  Requested fix: Add to klu_analyze_factor.cpp summary printout: `n_zero_J_entries=<N>  effective_nnz=<total_nnz - n_zero>  fill_ratio_effective=<Numeric->lnz+unz / effective_nnz>`. Aggregator should also surface both ratios.

Non-blocking notes:
- spike_run.sh + probe_cn_ram.sbatch have set -euo pipefail (note: probe_cn_ram.sbatch uses `set -uo` not `set -euo` — line 35 — which means a failed `cat /proc/meminfo` would not abort. Acceptable for a probe-only job, but inconsistent with spike_run.sh).
- KLU ordering codes: 0=AMD, 1=COLAMD, 2=natural-via-klu_analyze_given — verified against /opt/homebrew/include/suitesparse/klu.h:52. Spike mapping is correct.
- ε formula `sqrt(DBL_EPSILON) * (|Y[k]| + 1)` (fd_color_jacobian.cpp:319-322) is per-column k (not per-row i) and correctly degrades to `sqrt(DBL_EPSILON)` when Y[k]=0. CPR form is correct.
- omp_set_num_threads(1) is the first statement of main() in fd_color_jacobian.cpp:196, called BEFORE ColPack construction at line 228. Determinism guarantee holds.
- Link order in Makefile L91-96 (klu before amd/btf/colamd/suitesparseconfig) is correct for ld --no-undefined LIFO resolution.
- CSC determinism: dump_adjacency.cpp:134-139 std::sort + std::unique per column ensures bytewise reproducibility. PASS.
- Memory: all `new` paired with `delete` (Model_Data + FileIn/FileOut in dump_adjacency.cpp:440-442 and fd_color_jacobian.cpp:397-399). No leaks observed on the happy path. Failure paths (return 1 after MD allocation) DO leak — minor.
- Type safety: peak_rss_bytes uses size_t (klu_analyze_factor.cpp:96), CN_NODE_RAM_BYTES is constexpr size_t (cn_node_ram.h:32), 185528156160 fits in uint64. PASS.
- KLU γ=1 deviation #5: M = -γJ + I·diag correctly constructed; `n_diag_modified=1785/1785` confirmed in evidence log.
- OOM-as-data-point: klu_analyze_factor.cpp lines 235-238 + 270-276 + 282-287 print `KLU_OOM_DETECTED` and return 0. Format matches REQ-5 Scenario (case/ordering/btf/peak_rss_bytes). PASS.
- verify_adjacency_keliya.py is genuinely independent (reads `.sp.mesh/.sp.riv/.sp.rivseg/.sp.att` directly, no C++ deps); evidence TSV shows all 25 blocks PASS bitwise. PASS.
- Top-level Makefile (L33-39): `libshud.a` rule recurses into SHUD; `shud_spike` depends on libshud.a target. Sequential ordering guaranteed even under `make -j8`. PASS.
- SHUD/Makefile `SHUD_SRC_NOMAIN` confirmed at L548 (pre-existing wildcard, reused by libshud.a additive target). PASS.
- 75.6% n_zero FD entries (n_zero=7757/10255) is suspicious and would benefit from a brief design.md / README addendum explaining cold-start expected behavior.
```
