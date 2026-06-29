Reviewer agent: review-invariant-state
Review round: round 2 (post-fix)
Reviewed head SHA: 50d2a4bddacbfa3ef5b3e1c25d760555103c5556
Round 1 head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
SHUD pin: bc919f5 → 41d9a17 (forward-only on `openmp-baseline`)

Summary: All round-1 invariant-state findings resolved cleanly. F2 fix (SHUD .gitignore) lands as a single-hunk additive `.gitignore` patch on `openmp-baseline` with zero src/ touch; outer pointer bumped. Governing invariant (zero SHUD source patch) still holds. F1 dense-FD addition stays within the 3-binary contract (CLI flag on existing fd_color_jacobian, not a 4th binary). F4 preflight reorder cleanly extends OOM-as-data-point from 3 → 3 reasons (replaced pre-analyze exit-0 with post-analyze exit-0; klu_factor_OOM + post_factor_rss_exceeds_cn_ram retained). F5 chi gate added with case-aware threshold + clean exit code 2 propagating through `pipefail`. No new invariant gaps surfaced by the fix delta.

Verification of round 1 findings:

- c02 (SHUD .gitignore missing libshud.a / _libshud_obj/): ADDRESSED
  - Evidence: `cd SHUD && git diff bc919f5..41d9a17` returns ONLY 4 added lines in `.gitignore` (`# P8-tune.D ...` header + `/libshud.a` + `/_libshud_obj/`). Zero src/ tests/ Makefile touch.
  - Reachability: `git branch -r --contains 41d9a17` returns `origin/openmp-baseline` — pushed, not just local.
  - Ancestry: `git log openmp-baseline -3 --oneline` shows forward-only: `41d9a17` → `bc919f5` → `6ce17d6`. No history rewrite.
  - Outer bump: `git diff a65f3ca..50d2a4b -- SHUD` shows clean `bc919f5..41d9a17` pointer move.
- c15 (omp_set_num_threads(1) call asymmetry): NOT RE-RAISED per task instructions (REFUTED in round 1 phase 4.5 — klu_analyze_factor + dump_adjacency have no OMP regions).

Invariant Matrix Re-coverage (fix head):

- Governing invariant (zero SHUD source patch): COVERED — `git diff bc919f5..41d9a17 -- src/ tests/ Makefile` returns 0 lines; only `.gitignore` (root-level, NOT src/) modified. Carve-out remains strictly within Makefile (round 1) + .gitignore (round 2 addition). No header/source/test modification across either pin bump.
- Source-of-truth (SHUD pin lifecycle): COVERED — pin lifecycle: `3aec657 (B0) → 6ce17d6 (P1f SPGMR_MAXL) → bc919f5 (PR-0 round 1 Makefile carve-out) → 41d9a17 (PR-0 round 2 .gitignore patch)`. Forward-only on `openmp-baseline`. Remote reachability verified.
- Producers (3 binaries): COVERED — `tools/p8tune.D/Makefile:110-116` defines exactly 3 build targets (`dump_adjacency`, `fd_color_jacobian`, `klu_analyze_factor`). F1's `--brute-force-dense` is a CLI flag on `fd_color_jacobian` (cpp:182 usage + cpp:214 parser), NOT a 4th binary. REQ-2/3/5 3-binary contract preserved.
- Validators (2 Python scripts): COVERED — `dense_fd_cross_check.py` rewritten end-to-end (232 LOC vs round 1 stub). Imports stay stdlib-only: `argparse / math / struct / sys / pathlib` (`grep -E "^import|^from " | wc -l` = 7). Zero new C++ deps; binary reader uses pure `struct.unpack` against documented NDNJ magic + version=1 layout (same independence as round 1's `verify_adjacency_keliya.py`).
- Public entrypoints (3 binaries CLI + spike_run.sh + probe_cn_ram.sbatch): COVERED — `spike_run.sh` (137 LOC) chains 3 stages cleanly; F10 case whitelist added (L54-65) blocking `case ∉ {keliya, heihe, heihe_x4, heihe_x16}` with `exit 2`. F1 stage 2b/2c added GATED on `CASE==keliya && ORDERING==amd && BTF==1` (L107-112) — only the tool-correctness cell triggers dense FD probe + Python cross-check. Stage 2b reuses the same `$FDCJ` binary with `--brute-force-dense` flag; stage 3 (KLU) runs unchanged after.
- Evidence/audit: COVERED — 14 evidence files total (11 round-1 + 3 new): `cn_ram_probe.log`, `mac_smoke_keliya_{adjacency, fd_color, klu_factor, verify_nnz, dense_fd_cross_check, determinism, chi}.{log, tsv, txt}`, `maxl_sweep_summary.tsv` (round 1) + `mac_smoke_keliya_J_dense_sha256.txt`, `mac_smoke_keliya_chi_assertion.log`, `mac_smoke_keliya_dense_fd_baseline.log` (round 2 new). All present at `.review-evidence/p8tune-klu-spike-pr-0/`.
- Regression rows (keliya correctness gate + sha256 determinism + OOM-as-data-point): COVERED
  - keliya correctness: F1 real measurement (TSV `overall=PASS`, 25/25 blocks `verdict=PASS`, `n_colors_dut=16` vs `n_colors_ref_dense=1785` ≡ NumY confirming dense probe ran).
  - sha256 determinism: PASS (round-1 evidence preserved; F1 dense binary has its own sha256 anchor `0a17dfe1...8836bcc4` for cross-run repro).
  - OOM-as-data-point exit-0 path: 3 reasons retained, NOT 4 (F4 REPLACED pre-analyze exit-0 with post-analyze exit-0). Current reasons: `preflight_after_analyze` (klu_analyze_factor.cpp:274), `klu_factor_OOM` (cpp:293), `post_factor_rss_exceeds_cn_ram` (cpp:304). The pre-analyze `pattern_est_bytes` log at L236-239 is now ADVISORY ONLY (no exit-0 path attached) — see explicit comment "MUST NOT exit-0 OOM on its own" at L232.

Implementer Deviation Tracking:

- M = I - γ·J convention (round 1 Deviation #5): PRESERVED.
  - `grep "GAMMA\|M = I - \|test matrix" tools/p8tune.D/*.cpp` shows these strings appear ONLY in `klu_analyze_factor.cpp` (build step 2 at L189-226). Both `fd_color_jacobian.cpp` modes (colored at L378-418 + brute-force-dense at L351-377) probe RAW J — they write `J_values[p] = (DY_plus[i] - DY_base[i]) / e` with no γ scaling. M = I - γ·J is then materialized by `klu_analyze_factor` reading raw J from the binary and building `Ax[p] = -GAMMA * j.values[p]` + diagonal +=1 at L203-220.
  - This is correct for both modes: the spec's REQ-3 cross-check is "FD-probed J vs dense-FD-probed J" — both sides operate on raw J so they're directly comparable per-entry. The M = I - γ·J transformation is only relevant for the KLU numeric factor wall measurement (REQ-5).
  - F4 preflight reorder operates on the M-derived `Symbolic->lnz + unz` (the post-analyze decisive check at L266-279) — consistent with the convention since `klu_analyze` is given `Ap/Ai` from the pattern (which is shared between J and M — γ·J + I doesn't introduce new fill).

New State-Machine Implications from Fixes:

- F1 `--brute-force-dense` determinism: COVERED.
  - `omp_set_num_threads(1)` is at `fd_color_jacobian.cpp:198` — BEFORE the mode-branching parser at L205+ — so it applies to BOTH colored and brute-force-dense paths. New mode inherits the same single-thread determinism.
  - Evidence: J_dense_sha256.txt anchors the binary at `0a17dfe1...`; any re-run with the same SHUD pin + ColPack version should reproduce. (Note: this is a Mac-side anchor; CN may diverge per `numeric J vs ground-truth tolerance` but pattern + sha256 reproduce per-platform.)
- F4 preflight reorder state graph: CONSISTENT.
  - Round 1 state graph: `klu_analyze → pre_analyze_preflight (exit-0 OR continue) → klu_factor → (OOM exit-0 OR continue) → post_factor_rss_check (exit-0 OR success)`.
  - Round 2 state graph: `klu_analyze → preflight_after_analyze (exit-0 OR continue) → klu_factor → (OOM exit-0 OR continue) → post_factor_rss_check (exit-0 OR success)`.
  - Delta: the OOM-exit gate moved from BEFORE klu_analyze to AFTER klu_analyze. The pre-analyze log is now advisory only (no exit-0). Total exit paths still 4: `success` + 3× `exit-0 OOM-as-data-point` + (implicit `exit-1` on klu_analyze failure unrelated to RAM at L254-256). State graph is consistent and arguably IMPROVED — uses real `Symbolic->lnz+unz` data instead of pattern-only lower bound.
- F5 chi gate state transition: CONSISTENT.
  - Added EARLY exit at `fd_color_jacobian.cpp:251-255`: `chi > chi_threshold → exit 2`.
  - `spike_run.sh:36` has `set -euo pipefail` + every stage piped through `tee -a "$LOG"` — `pipefail` ensures any non-zero exit code from the binary propagates AFTER tee logging completes, aborting the wrapper cleanly with exit 2. OOM-as-data-point still exits 0 (cleanly distinguishable from chi-overflow exit 2).
  - Exit code matrix is now: `0 = success OR OOM-as-data-point`, `1 = klu_analyze failed / non-OOM klu_factor failed`, `2 = chi gate failed (case ∉ whitelist OR chi > threshold)`. Cleanly orthogonal.

Gap Sweep (new):

- README `make shud_spike` artifact untracking: now masked at the source-of-truth (SHUD/.gitignore) — c02 closed at the root, not via README workaround. No residual concern.
- `dense_fd_cross_check.py` CASE_DIMS registry hardcodes keliya only — if a future epic tries to dense-FD heihe/heihe_x4, the script returns 1 with a clear "dims for case '<x>' not registered" error (L209-211). Bounded by the keliya-only spike_run.sh gate at L107 — no risk of silent stale-output. Acceptable for PR-0 scope.
- `--brute-force-dense` semantics: 1 rhs_core per column × NumY columns. For keliya NumY=1785 → ~30s on Mac (per F1 commentary at fd_color_jacobian.cpp:356-358). Intractable for heihe (NumY~25K → ~hours) — but spike_run.sh gates correctly. NO state-machine risk.

Findings:

No new blocking or non-blocking findings from invariant-state lens. Round 1 c02 closed; c15 stays REFUTED (not re-raised per task contract).

Non-blocking notes:

- Pre-analyze pattern_est_bytes log at klu_analyze_factor.cpp:236-239 is retained as advisory hint with explicit "MUST NOT exit-0" comment. Defensible — preserves the original telemetry value (downstream cell-summary aggregation may want both pre- and post-analyze estimates) without re-introducing the c04 escape path. Future PR-A cell-summary parser should explicitly use `preflight_after_analyze` not `pre-flight` if it wants the decisive number.
- F5 chi threshold dispatch `(case_name == "keliya") ? 30 : 50` is implicit-default-allowed (any non-keliya case gets 50). This is permissive: if a typo'd case name slipped past spike_run.sh whitelist, it would still get the 50 threshold instead of erroring. spike_run.sh whitelist (F10) covers this — but the binary itself remains permissive. Tolerable since the wrapper is the authoritative entrypoint per spec.
- SHUD-side `.gitignore` ordering: round 2 added entries at the END of `.gitignore` (lines 47-49), preserving all prior entries and the round-1 `tests/s1d_*_smoke` block intact. No interaction with the openmp-baseline branch's `tests/` exclusion patterns.
