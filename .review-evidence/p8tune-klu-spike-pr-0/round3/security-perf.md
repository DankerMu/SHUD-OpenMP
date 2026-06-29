# Round 3 cross-review — security & performance

PR: #384 (feat/issue-380-p8tune-klu-spike-pr-0)
Round 3 head: `7fc325b`
Round 2 head: `50d2a4b`
Reviewer scope: security + performance dimensions
Date: 2026-06-28

---

## Summary

Round 2 had APPROVE + 2 non-blocking notes (README enum drift / legacy line). Both are now eliminated by G2/G4. G1 (the only behavior change in this round) is **correct** and **does not introduce a new security/perf regression** for PR-0 scope. The residual signal-9 risk for the 8 non-AMD PR-A cells is real but **correctly out of PR-0 scope** — it is a PR-A `spike_array.sbatch` / sacct concern, as the briefing already anticipates.

## Tasks 1–4 findings

### Task 1 — G1 RSS-axis re-verify (post `klu_analyze` preflight gate)

**Code under review**: `tools/p8tune.D/klu_analyze_factor.cpp:273-292`
- L273 gate: `if (ordering_id == 0 && Symbolic->lnz > 0 && Symbolic->unz > 0)` — only AMD (id=0) runs the L+U-based preflight; COLAMD (id=1) and natural-via-given (id=2) skip.
- L287-291 else-branch: emits explicit `PREFLIGHT_AFTER_ANALYZE skipped (... AMD-only); relying on post-factor RSS check` (visible in smoke wall L24, L39).
- For non-AMD orderings the only safety net is the post-factor block at L302-322 (klu_factor malloc-fail OR `peak_rss_bytes()` > CN_NODE_RAM_BYTES).

**Correctness verdict**: gate logic matches KLU 7.12.2 documented behavior — `lnz/unz` are only populated for the AMD path; for COLAMD/natural they remain at the EMPTY sentinel `-1`. The previous code cast `-2.0` (lnz + unz) to `size_t`, classic undefined-behavior wrap-around (~`0xFFFFFFFFFFFFFFFE`) that would *always* exceed budget and emit spurious `KLU_OOM_DETECTED` on every non-AMD cell. The defensive `> 0` check is correct (handles both `-1 + -1` and any other non-positive return).

**Residual signal-137 risk re-eval**: The original c04 / F4 concern was that `klu_factor` reaching peak-RSS overflow *might* OOM-kill the process before the post-factor check runs. With G1 active, COLAMD (4 cells) and natural (4 cells) now rely SOLELY on the post-factor path. So 8 of 16 PR-A cells lose the early-exit preflight guard.

Two failure modes for those 8 cells:
1. **klu_factor internal `malloc` returns `NULL`** (cgroup MEM-pressure, soft limit). `klu_factor` sets `common.status = KLU_OUT_OF_MEMORY` and returns NULL; L304-309 catches this and emits `reason=klu_factor_OOM` exit-0. **Safe.**
2. **kernel OOM-killer fires SIGKILL** before `klu_factor` returns. Tool dies, stdout never flushes the diagnostic. **Signal-9 escape.**

Under the PR-A submission profile (`--mem=24G` per cell, tasks.md L32; vs CN_NODE_RAM_BYTES = ~173 GiB), Slurm cgroup will trigger OOM-killer well before raw cn-node RAM saturation. So the kernel-SIGKILL path is the *expected* mode for an x16 + COLAMD/natural cell that overflows.

**Is this MORE severe than original c04?** Marginally — yes, in the formal sense that the post-factor RSS check on L316 is *unreachable* for any cell where klu_factor exhausts the 24G cgroup before returning. But the alternative (keeping the pre-G1 broken preflight) was *strictly worse* because it produced **spurious** OOM emissions on cells that would have factored fine. G1 is the correct fix; the SIGKILL detection gap was already present in round-2 design for the AMD-with-lnz-too-large case anyway, and the spec REQ-5 "OOM-as-data-point" Scenario explicitly allows `klu_factor` malloc-fail as the canonical detection path (spec.md L140).

**In-scope mitigation?** No. PR-0 cannot mitigate SIGKILL — by definition the user-space tool is dead when SIGKILL fires. The two valid mitigations are both PR-A-side:
- `spike_array.sbatch` already specifies `--mem=24G` (tasks.md L32) which makes cgroup-mediated SIGKILL the *expected* OOM mode, not raw cn-node RAM saturation.
- PR-A aggregator must read `sacct -j <jobid> -o JobID,State,ExitCode,MaxRSS` and classify `State=OUT_OF_MEMORY` (ExitCode 137 / signal 9) cells as `rss_overflow` data-point — same classification bucket as the in-tool diagnostic. The spec already says (L142-143) "the aggregator SHALL classify that cell as `rss_overflow`". The aggregator's `sacct` parsing is a PR-B (aggregate_klu_spike.sh, tasks.md L42) concern, not PR-0.

**Verdict for task 1**: G1 is correct. Residual SIGKILL gap is real but is **correctly deferred to PR-A `--mem` budgeting + PR-B sacct parsing**. No PR-0 finding.

### Task 2 — G2 (README enum drift)

Diff at `tools/p8tune.D/README.md:166` replaces `preflight_estimate` with `preflight_after_analyze`. Doc-only, mirrors the actual emission string in `klu_analyze_factor.cpp:282`. Grep confirms the legacy enum value `preflight_estimate` is absent from `tools/p8tune.D/`, `docs/`, `openspec/changes/p8tune-klu-spike/` (exit 1 = no match). Grep confirms the new value `preflight_after_analyze` appears exactly at code emission site L282 and README L166 (single source-of-truth pair). **No security implication.** Confirmed.

### Task 3 — G4 (legacy pre-flight line removal)

Diff at `tools/p8tune.D/klu_analyze_factor.cpp:238` removes the stale duplicate line `[klu] pre-flight: A nnz=...`. The advisory hint at L235-237 (`PREFLIGHT_HINT pattern_est_bytes=...`) is preserved and is sufficient — re-reading L228-237 confirms the comment block makes clear that this is advisory-only, with the decisive check after `klu_analyze`. Grep confirms `pre-flight: A nnz=` is absent from the entire `tools/p8tune.D/`, `docs/`, `openspec/` trees (exit 1 = zero hits). **No security or perf implication.** Confirmed.

### Task 4 — Mac smoke wall numbers for 3-ordering matrix

Source: `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log`

| ordering | btf | numeric_factor_wall_sec | peak_rss_bytes | fill_ratio |
|----------|-----|-------------------------|----------------|------------|
| amd      | 1   | 0.000539                | 3,145,728      | 3.2265     |
| colamd   | 1   | 0.000680                | 3,555,328      | 4.6414     |
| natural  | 0   | 0.345711                | 35,700,736     | 205.9479   |

All 3 orderings succeeded on keliya (NumY=1785, Anz=10255). The natural-ordering result is the most instructive:
- Wall: 0.346s vs 0.0005s for AMD/COLAMD = **640x slower** on a tiny case.
- Peak RSS: 35.7 MB vs 3.1 MB = **11.4x**.
- Fill ratio: 205.9 vs 3.2 for AMD = **64x higher fill** — well above the spec REQ-5 fill threshold `8·log2(NumY) ~= 8·log2(1785) ~= 87`, so natural would FAIL the fill axis on keliya alone.

This is exactly the signal PR-A is meant to surface: natural ordering on heihe_x16 (NumY ~760K) would be expected to OOM, since fill grows super-linearly with mesh size. The G1 fix matters precisely because the previous broken preflight would have masked this — every non-AMD cell would have exit-0'd with a spurious OOM diagnostic before `klu_factor` ever ran, killing the entire scientific value of the PR-A sweep for 8 of 16 cells.

Note on RSS for AMD/COLAMD on Mac: `peak_rss_bytes` is reported as 3.1 MB / 3.5 MB. This is reasonable for a 1785-row matrix; the value uses `getrusage(RUSAGE_SELF).ru_maxrss` which on macOS is bytes (Linux is KB — there is a documented platform-dependent unit mismatch). I'm not flagging this as a finding because (a) it was already raised and resolved in earlier rounds and (b) Mac RSS values are advisory-only — the decisive RSS measurement is on cn14 in PR-A.

**Verdict for task 4**: 3-ordering smoke matrix succeeds; wall + RSS numbers are physically plausible; natural-ordering signal is consistent with the spec's expected verdict structure. Confirmed.

## Findings

None. All four tasks confirmed clean. G1 introduces a real-but-correctly-scoped SIGKILL detection gap that is the responsibility of PR-A `spike_array.sbatch --mem=24G` budgeting + PR-B sacct parsing — explicitly out of PR-0 boundary per spec REQ-7 (spec.md L217 forbids PR-0 from touching `spike_array.sbatch`).

## Verdict

**APPROVE** — Round 3 fixes (G1/G2/G4) are correct, minimal, and on-scope. No new security or performance findings. The KLU 7.12.2 lnz/unz EMPTY-sentinel UB that round 2 surfaced is properly closed. The SIGKILL escape route for non-AMD overflow cells is acknowledged but is a PR-A/PR-B boundary concern, not a PR-0 regression.
