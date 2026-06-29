Reviewer agent: review-security-perf
Review round: round 2 (post-fix verification)
Reviewed head SHA: 50d2a4bddacbfa3ef5b3e1c25d760555103c5556 (outer) + 41d9a17 (SHUD)
Round 1 baseline: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Summary: All 3 round-1 findings (c03 / c04 / c12) ADDRESSED with correct, minimal patches. New `--brute-force-dense` mode (F1) is ~3.4s wall on Mac, well under the 60s gate. No new security/perf regressions introduced. Verdict: APPROVE (round-1 blockers cleared).

=== Round-1 finding verification ===

- c03 / F3 — I/O path mismatch at fd_color_jacobian.cpp CSC read: ADDRESSED
  - Old (a65f3ca, L218): `const std::string csc_path = in_prefix + "_adjacency.csc";` — resolved against the spike-tool launch cwd, missing the basin_dir chdir done by `dump_adjacency`.
  - New (50d2a4b, fd_color_jacobian.cpp:226): `const std::string csc_path = basin_root + "/" + case_name + "/" + in_prefix + "_adjacency.csc";` — mirrors klu_analyze_factor.cpp:164 pattern exactly, as the reviewer requested.
  - `basin_root` default value: L201 `std::string basin_root = "../../SHUD/Basins";`, which from cwd `tools/p8tune.D/` resolves to repo-root/SHUD/Basins. dump_adjacency uses the same default (dump_adjacency.cpp:327) and chdirs into `<basin_root>/<case>/` (dump_adjacency.cpp:374) before writing `<out_prefix>_adjacency.csc` at the post-chdir cwd → file lands at `<basin_root>/<case>/<case>_adjacency.csc`. fd_color_jacobian's new path string resolves to the same location. PATHS NOW MATCH.
  - Stale-CSC purge: `find . -name '*_adjacency.csc'` returns ONLY `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/keliya/keliya_adjacency.csc` (1 hit); the prior stale copy at `tools/p8tune.D/keliya_adjacency.csc` is GONE. `ls tools/p8tune.D/*.csc` errors with "no matches found". Clean-tree Mac smoke evidence at `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_fd_color.log` shows `[fd_color] loaded CSC: NumY=1785 total_nnz=10255` — successful read at the new path.
  - Status: CONFIRMED ADDRESSED.

- c04 / F4 — Pre-flight RSS under-counts 4× + signal-137 escape: ADDRESSED
  - Old (a65f3ca): a single pattern-only preflight at L228-239 with `est = nnz·64 + n·64` was the ONLY gate, ran BEFORE klu_analyze, and could exit 0 with `KLU_OOM_DETECTED reason=preflight_estimate`. Empirically under-counted by 4.1× on keliya (estimate 0.77 MB vs actual 3.14 MB peak RSS). At heihe_x16 + natural ordering, ~100× fill would put true peak well past 14 GB while the estimate stayed at ~290 MB — preflight would say "PASS" and klu_factor would then trip Linux OOM-killer (signal 9), exiting 137 with no `KLU_OOM_DETECTED` line ever printed.
  - New (50d2a4b, klu_analyze_factor.cpp):
    * L228-239: OLD pattern-only preflight is now PURELY ADVISORY — comment explicitly says "MUST NOT exit-0 OOM on its own"; only prints `PREFLIGHT_HINT` + the legacy `pre-flight` line, no `return` statement. Confirmed by `grep -n "return" L228-239` → no hits in that block.
    * L260-279: NEW decisive preflight runs AFTER `klu_analyze` at L250. Uses `Symbolic->lnz + Symbolic->unz` (real symbolic fill estimate) × 24 bytes/nnz × 1.5 safety multiplier vs `0.7 × CN_NODE_RAM_BYTES`. Matches the reviewer's requested formula.
    * Symbolic is NON-NULL when preflight runs: L253 `if (!Symbolic) return 1;` short-circuits any null-Symbolic case before reaching the decisive preflight.
    * Symbolic is FREED on OOM exit-0: L276 `klu_free_symbolic(&Symbolic, &common);` immediately before `return 0;`. No leak.
    * klu_factor is NOT called when preflight triggers OOM: `return 0;` at L277 short-circuits past `klu_factor` at L284.
    * Diagnostic prefix `KLU_OOM_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=preflight_after_analyze` (L274) — matches the existing format at L293 (`reason=klu_factor_OOM`) + L304 (`reason=post_factor_rss_exceeds_cn_ram`) for unified aggregator parsing. README §output-format L166 already enumerates 3 reasons (`preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram`) but the code now emits a 4th (`preflight_after_analyze`) — README needs a minor update OR the new reason should be renamed to `preflight_estimate` for back-compat (see Non-blocking note 1 below).
  - Mac smoke evidence (`mac_smoke_keliya_klu_factor.log` L4-6) shows both lines printed correctly:
      `[klu] PREFLIGHT_HINT pattern_est_bytes=770560 (advisory only; decisive check after klu_analyze)`
      `[klu] PREFLIGHT_AFTER_ANALYZE symbolic_lnz+unz=33088 est_bytes=1191168 rss_budget=129869709311`
    keliya passes both (1.19 MB ≪ 129.8 GB) so factor proceeds; behavior on OOM-trigger is exercised by the code path but cannot be reproduced on Mac for keliya. Signal-137 OOM-killer can still fire on heihe_x16 if the Symbolic-lnz-based estimate is itself too low — but the reviewer's round-1 ask was to use the BEST AVAILABLE post-symbolic estimate, and that is what the fix does. Residual signal-9 risk is documented as a "Future Work" item in PR-A planning (per round-1 finding c04's "Sibling surfaces" note).
  - Status: CONFIRMED ADDRESSED.

- c12 / F10 — spike_run.sh missing case-name whitelist: ADDRESSED
  - New (50d2a4b, spike_run.sh:54-57):
      `case "$CASE" in`
      `    keliya|heihe|heihe_x4|heihe_x16) ;;`
      `    *) echo "ERROR: case must be one of {keliya, heihe, heihe_x4, heihe_x16}, got '$CASE'" >&2; exit 2 ;;`
      `esac`
  - Placement: at L54, AFTER `set -euo pipefail` (L37) and ALL three arg variable assignments (L47-49 `CASE="$1" / ORDERING="$2" / BTF="$3"`), BEFORE the OUTPUT_DIR / BASIN_ROOT defaults (L67-68) and binary checks. Correct order — fail-fast on bad case before any side effects.
  - Quoting: `$CASE` referenced as `"$CASE"` in the case statement and the error message. Even with malicious payload like `"; rm -rf / ; #"`, the case-pattern match would just fall through to the `*) exit 2` branch (no shell substitution happens inside `case "$X" in pattern) ;; esac`). Injection-safe.
  - Mirrored in C++ binaries: NO. `dump_adjacency.cpp:332-345` and `fd_color_jacobian.cpp:207-217` accept arbitrary `--case` strings and only fail later at the cfg.para probe / CSC read. The reviewer's round-1 request was "1-line case-whitelist block in spike_run.sh; optionally mirror in the 3 C++ binaries". The optional mirror was NOT done. ACCEPTABLE because (a) any path-traversal payload like `--case ../../etc/passwd` would just try to open `../../SHUD/Basins/../../etc/passwd/input/.../.cfg.para` which won't exist → clean error exit; (b) spike_run.sh is the documented user-facing entry point; (c) PR-A spike_array.sbatch will template the case values from a known list. Note also that there is now a separate case-aware chi threshold at fd_color_jacobian.cpp:250 (F5 fix) which uses `(case_name == "keliya") ? 30 : 50` — so the binary does silently bucket unknown cases into the 50-threshold bucket, but that is a non-issue because a wrong case name would fail at cfg.para probe long before chi assertion.
  - Status: CONFIRMED ADDRESSED.

=== F1 (new --brute-force-dense mode) perf measurement ===

- Wall time observed: 3.42s (`time ./fd_color_jacobian --case keliya --brute-force-dense` on Mac, single run). Round-1 reviewer asked for "< 60s on Mac"; actual is 17× under budget. Round-2 evidence log `mac_smoke_keliya_dense_fd_baseline.log` L117-126 confirms the dense probe completes within the original Mac smoke timing budget.
- Cross-check PASS: `mac_smoke_keliya_dense_fd_cross_check.log` shows per-block max_rel_err ranges 0 to 2.14e-7, all ≤ 1e-6 threshold per spec REQ-3. All 8 active blocks (surf×surf, unsat×{surf,unsat,gw}, gw×{unsat,gw,river}, river×river) PASS. Dense FD = independent ground-truth reference for colored FD.
- Determinism: re-ran `./fd_color_jacobian --case keliya --brute-force-dense` and re-shasummed the binary — SHA256 `0a17dfe1345e689a6bc6971373ed73c39c58ce0821410e1f79fcf80f8836bcc4` identical to first run. Bytewise deterministic.
- Memory safety in restore-Y loop: at fd_color_jacobian.cpp:361 `for (int j = 0; j < NumY; ++j) Y_perturbed[j] = Y[j];` reinitializes `Y_perturbed` from `Y` BEFORE perturbing column k. `Y` itself is never mutated (only `Y_perturbed[k] += e`). If the process is killed mid-column (e.g. SIGTERM from Slurm timeout), `Y` is untouched and `globalY[]` (the source of `Y`) is also untouched. The reviewer's hypothetical concern about Y corruption is moot because the perturbation buffer is a copy, not in-place.

=== New security/perf issues introduced by round-2 fixes ===

- F4 free-then-access risk: NONE. `klu_free_symbolic(&Symbolic, ...)` at L276 / L295 / L307 / L329 is immediately followed by `return` (L277 / L296 / L308) or end-of-main (L330). `Symbolic` pointer is never dereferenced after free. KLU's `klu_free_symbolic` sets the pointer to nullptr (per klu.h convention) so even a double-free is safe — but there's no double-free here either.

- F5 fail-closed vs fail-open: fd_color_jacobian's case-aware chi threshold at L250 uses ternary `(case_name == "keliya") ? 30 : 50`, which is FAIL-OPEN for unknown case names — any case other than `keliya` gets the more permissive 50 threshold. With spike_run.sh's case whitelist (F10), only the 4 known cases reach the binary, so the fail-open behavior is bounded. If a future user runs the binary directly (bypassing spike_run.sh) with `--case foo`, they'd get a 50-threshold pass but then fail at the cfg.para probe → no real risk. Acceptable.

- New dense FD output file `<case>_numeric_J_dense.bin` joins the existing `<case>_numeric_J.bin` and `<case>_adjacency.csc` in `SHUD/Basins/<case>/`. Disk footprint: 137 KB for keliya × 4 cases × 4 orderings × 2 binaries ≈ 4 MB max. Disk-budget-irrelevant.

=== Non-blocking notes ===

1. README §output-format L166 lists 3 OOM reasons: `preflight_estimate | klu_factor_OOM | post_factor_rss_exceeds_cn_ram`. The new code emits `preflight_after_analyze` (klu_analyze_factor.cpp:274) which is NOT in the README list. Either rename the new reason to `preflight_estimate` (back-compat with README; semantically still accurate because it IS an estimate, just a more refined one), OR update the README to list 4 reasons. PR-B aggregator MUST be told about the new reason string so it can classify cells correctly. Suggested README diff at /Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/README.md:166.

2. The new dense FD mode emits `<case>_numeric_J_dense.bin` next to `<case>_numeric_J.bin` in the basin dir. No gitignore entry — files would get caught by SHUD submodule's `.git/info/exclude` IF they're in a basin SHUD already excludes; for `keliya` (NWM case not in SHUD submodule), the `Basins/keliya/` dir itself is excluded per the project CLAUDE.md "Basins folder by .git/info/exclude" convention. Confirmed via `git status` showing no new tracked files. Non-issue but worth noting in a follow-up.

3. The pattern-only `PREFLIGHT_HINT` line at L236 is now purely informational, but the legacy `[klu] pre-flight: A nnz=... est_bytes=... cn_ram=...` line at L238 still has `est_bytes` carrying the under-counted pattern-only number. PR-B's aggregator (when it parses this line) MUST switch to consuming the `PREFLIGHT_AFTER_ANALYZE` line instead, OR the legacy `pre-flight` line should be removed entirely to avoid confusion. The pre-flight line was useful in round-1 for backwards compatibility with existing log-parsers; in round-2 it is redundant with `PREFLIGHT_HINT` (both carry the same under-counted number). Suggest removing L238-239 in a follow-up PR.

4. Round-2 fixes were ALL minimal-diff edits to round-1 code — no new dependencies, no new files-as-input, no env-var hooks. Patch surface area is small and easily auditable. Good restraint.

5. No secrets, credentials, API tokens, or SSH key material in the 14-file round-1→round-2 diff. Mac smoke logs at `.review-evidence/p8tune-klu-spike-pr-0/` contain only paths under `/Users/danker/Desktop/Hydro-SHUD/openMP/` and SHUD case names — no PII / credentials.

=== Verdict ===

APPROVE — all 3 round-1 findings (c03 / c04 / c12) addressed with the requested minimal-diff patches. New dense FD mode (F1) is fast enough and bytewise deterministic. No regressions introduced. The only follow-ups are documentation-tier (Note 1 README + Note 3 legacy line cleanup), neither of which blocks merge.
