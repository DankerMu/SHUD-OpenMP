# Cross-Review Round 1 — Security/Performance Lens

**PR**: #400 (feat/issue-394-p8tune-amg-spike)
**Head SHA**: `09a815dbcab9eabbddcad9550c706ddfa8636519`
**SHUD pointer bump**: `710c00a` → `056a1dc`
**Scope**: security · path safety · adversarial inputs · resource management · CI signal integrity · performance regressions · removed-behavior audit
**Reviewer**: leaf subagent — read-only

---

## Summary

The SHUD source fix itself (Model_Data + ElementHotData NSDMI nullptr defaults) is correct, narrow, and bitwise-neutral by construction — the dtor full-path resource-release contract is now defined-no-op on the failed-init branch. SHUD-side scalar `delete flood` is also safe (`flood = nullptr` was added in `710c00a`, and `delete nullptr` is defined no-op per C++03+).

However, the round contains **one Critical CI signal-integrity defect** that will hard-fail the new `asan-ubsan` job on EVERY clean run after merge, and **one Warning sibling-class scope omission** that leaves the same UB pattern unmitigated in three other classes that the dtor chain transitively invokes.

---

## Findings

### CRITICAL: ASan LEAK SUMMARY gate inverts pass/fail logic — every clean CI run hard-fails

`.github/workflows/serial-baseline.yml:1444-1467` + spec `amg-pattern-spike-verdict/spec.md` REQ-3 + `design.md` L113/L118/L126

The new CI gate requires:

```yaml
ASAN_OPTIONS: "detect_leaks=1:halt_on_error=1:print_stacktrace=1:exitcode=0:print_suppressions=0"
# ...
leak_summary_count=$(grep -cE '^==[0-9]+==(ERROR: LeakSanitizer:|SUMMARY: AddressSanitizer: detected memory leaks)|LEAK SUMMARY:' ...)
if [[ "$leak_summary_count" == "0" ]]; then
    echo "::error title=ASan dtor coverage missing..."; exit 1
fi
```

The premise (per design.md L113 "本 PR 强化为 LEAK SUMMARY: grep" and the workflow comment "LSan exit path emits a LEAK SUMMARY: line on stderr") is **wrong**:

1. `LEAK SUMMARY:` is the literal `valgrind --tool=memcheck --leak-check=full` output format, NOT an ASan/LSan output line. ASan does not emit that string at any time.
2. LeakSanitizer on Linux with `detect_leaks=1` only emits the matched `==pid==ERROR: LeakSanitizer:` + `==pid==SUMMARY: AddressSanitizer: <N> byte(s) leaked` block **when leaks are actually detected**. A zero-leak clean run produces NO LSan output at all (per the LSan source / Google sanitizers wiki "Indirect leaks" section — silent on success).

Net effect after merge:

- Happy path (the goal: 0 leaks): `leak_summary_count=0` → gate hard-fails → CI red → no PR can land on main without disabling this job.
- Broken path (leaks present): `==pid==ERROR: LeakSanitizer:` matches → `leak_summary_count≥1` (passes leak gate), but `asan_err≥1` from the same regex set on the `==.*==ERROR` pattern → still fails (correct but for the wrong reason).

Evidence file `valgrind_keliya_postfix.log:36-42` does flag macOS limitations and CLAIMS the Linux runner "produces `LEAK SUMMARY:` stderr line"; that claim is incorrect and was not validated by running ASan with `detect_leaks=1` on a Linux runner before authoring the gate. The Mac shud_asan run cannot have validated it (Mac LSan absent, per the evidence file's own admission L37-39).

**Fix options** (pick one):

1. Drop the dtor-coverage gate and rely on the more robust `asan_err`/`ubsan_err` regex (already in place). The UB resolution is independently validated by `valgrind ./shud heihe` on the server (Phase 2).
2. Replace the dtor-proof with a positive signal that ASan/LSan actually emit on success — e.g. inject an intentional leak in a debug-only `#ifdef` ctor inside `shud_asan` then assert `LeakSanitizer: detected memory leaks` appears (counterintuitive; not recommended).
3. Replace LSan with a process-exit-status check: compile shud_asan with `__attribute__((destructor))` print of a sentinel like `[SHUD] FreeData chain complete`, then grep for that sentinel in stdout. (Recommended: positive, deterministic, cheap, sanitizer-independent.)

This finding blocks merge. Spec amendment is also needed (design.md L113/L118/L126/L137 carry the same false claim).

---

### WARNING: Sibling-class uninit-ptr + unconditional `delete[]` left unmitigated

`SHUD/src/classes/TimeSeriesData.cpp:31-34` + `SHUD/src/classes/Lake.cpp:52-58, 84-96` + `SHUD/src/classes/TabularData.cpp:7-16`

The PR audit found three other classes that exhibit the exact same anti-pattern the PR fixes for `Model_Data` / `ElementHotData`:

1. **`_TimeSeriesData::~_TimeSeriesData()` (TimeSeriesData.cpp:31-34)** unconditionally loops `delete[] ts[i]` for `i ∈ [0, MAXQUE)`. The array `double *ts[MAXQUE+1]` (TimeSeriesData.hpp:41) has no in-class init. Default ctor (line 28-29) is empty. `Model_Data` embeds at least 5 `_TimeSeriesData` instances (`tsd_LAI`, `tsd_MF`, etc., Model_Data.hpp:124-134); if any of these instances' `initialize(n)` is never called (e.g. allocation aborts mid-loadinput), the dtor walk produces UB identical to #386's symptom (random heap addresses passed to `delete[]`).

2. **`LakeBathymetry::~LakeBathymetry()` (Lake.cpp:52-58)** — same `if(nvalue > 0)` guard pattern, but uses scalar `delete index` / `delete yi` / `delete ai` on heap arrays allocated as `new[]` (mismatched alloc/dealloc — separate pre-existing UB).

3. **`_Lake::~_Lake()` (Lake.cpp:84-96)** — `if(NumEleBank > 0) delete iEleBank` is guarded, BUT `delete RivOut` (line 94) is matched against `QRivOut = new double[NumRivOut]` (line 102), not `RivOut` — likely pre-existing typo bug; outside this PR's scope.

4. **`TabularData::~TabularData() → reset()` (TabularData.cpp:7-16)** is guarded by `if(nrow > 0)` and is safe IF `nrow=0` after default-init (in-class `int nrow;` is NOT initialized — TabularData.hpp:19 declares without initializer; stack-allocated instances would hold indeterminate value).

For (1), the production happy path (`MD->loadinput()`) does call `tsd_LAI.initialize(...)` for every TSD instance before `MD->initialize()`, so production runs are safe today. But the same is true of `Model_Data`'s hot fields — production happy path was also safe; #386 surfaced only via OOM-mid-init or `dump_adjacency`'s short-init walk. If P8-tune.F downstream tools (PR-A onward) construct `Model_Data` with abridged init (which they do — `dump_adjacency.cpp:402-404` calls `loadinput()+initialize()` but skips full forcing setup), the TSD dtor walk could trigger.

**Recommendation**: not a blocker for THIS PR (scope is `Model_Data` + `ElementHotData` per #386). File follow-up issue to add NSDMI defaults to `_TimeSeriesData::ts[]` + `TabularData::x` + `TabularData::nrow=0`. The fix is the same 1-line per pointer pattern as this PR, ~10 minutes of work, ~15 minutes evidence — preempts re-discovery by P8-tune.F PR-A/PR-B reviewers.

---

### WARNING: NSDMI coverage gap — 7 deleted-but-not-defaulted member symbols

`SHUD/src/ModelData/Model_Data.hpp` (cross-referenced against `MD_readin.cpp:526-688`)

Cross-checked the 86 NSDMI-defaulted member pointers against the 117 `delete[]` / `delete` calls in `FreeData()`. The hot-data SoA (`hot.*`, 32 pointers in `MD_layout.hpp`) is FULLY covered. The 6 file-scope globals `globalY` + `uYsf` / `uYus` / `uYgw` / `uYriv` / `uYlake` (`shud.cpp:30-35`) are safe because file-scope variables are zero-init by C++ static-init rule, even without explicit initializer.

The remaining gap is 7 members that ARE declared in `Model_Data.hpp` and ARE present in some `delete` form, but were left out of the NSDMI diff because they are pre-existing leak suspects (not in `FreeData` at all). Already-broken non-delete'd members in `Model_Data.hpp`:

- `AccT_surf`, `AccT_sub` (Model_Data.hpp:153-154 fixed in this PR)
- `ISFactor`, top-level `windH` (line 174-175 fixed)
- `lake`, `y2LakeArea` (line 176, 180 fixed)
- `t_sph` (line 350 fixed)

These get `= nullptr` in the diff but `FreeData()` never calls `delete[]` on them. That's a pre-existing leak (not a regression). NSDMI default makes future delete-add safe; harmless. **No action needed** for this PR; flagged for transparency.

---

### SUGGESTION: server_acceptance_cmds.sh has no job-dependency chain between sbatch submits

`.review-evidence/p8tune-amg-pr-0/server_acceptance_cmds.sh:53, 76, 101`

The script submits 3 sbatch jobs back-to-back via ssh:
1. `build_shud.sbatch` (build SHUD + p8tune.D tools)
2. `valgrind_heihe.sbatch` (use the binary built in #1)
3. `dump_adj_x4_x16.sbatch` (use tools built in #1)

`sbatch` returns success immediately after queueing. There is no `--dependency=afterok:$JID` chain. If job #1 fails asynchronously on the cluster (cn-node OOM during link, gcc version mismatch with InstallSundials cache, etc.), jobs #2 and #3 will still run — using the previous binary in `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud` if it exists, or failing with `./shud: No such file or directory`.

**Fix**: capture `JOBID=$(sbatch --parsable build_shud.sbatch)` then submit downstream with `sbatch --dependency=afterok:${JOBID} valgrind_heihe.sbatch`. Minor — runbook hygiene, not a security concern.

Path-injection / variable-interpolation analysis: the heredoc-inside-ssh-double-quotes uses outer-shell expansion of `${SCRATCH}` (hard-coded literal `/scratch/frd_muziyao/SHUD-OpenMP`) before passing to the remote, plus quoted `'EOF'` to suppress remote re-expansion. No user-controlled input crosses the ssh boundary. Slurm 三铁律 satisfied (sbatch from `/scratch/.p8tune.F-runs`, --output/--error under `/scratch`, sbatch files written to `/scratch`). Safe.

---

### SUGGESTION: `ASAN_OPTIONS=exitcode=0` is intentionally safe but masks one signal

`.github/workflows/serial-baseline.yml:1427`

The PR adds `exitcode=0` to `ASAN_OPTIONS`. Per the LSan README, `exitcode=N` sets the process exit code emitted **when leaks are detected**; default is 23. With `exitcode=0`, a leaks-detected run exits 0 — but the leak text still appears in stderr (caught by the workflow's `asan_err` regex `==.*==ERROR`). UAF / OOB / double-free still abort with their own non-zero exit codes (controlled by `halt_on_error=1`, untouched).

Verdict: SAFE as designed — `exitcode=0` is a documented LSan setting that prevents the new `detect_leaks=1` gate from non-zero-exiting on the (incorrectly-expected) LSan summary emission. The signal preservation chain `halt_on_error=1` → first error aborts immediately is intact for memory errors. Only the leak-only path is muffled at exit-code level, and the stderr regex catches it anyway.

No action needed; flagged for review-trail completeness. NOTE: if the Critical finding above is resolved by removing the leak-coverage gate, the `exitcode=0` clause should also be removed (it was added specifically to support the now-broken gate).

---

### PRAISE: NSDMI pattern matches the FloodAlert precedent + is bitwise-neutral by construction

`SHUD/src/ModelData/Model_Data.hpp:103-121, 131-364` + `SHUD/src/ModelData/MD_layout.hpp:5-21`

The fix correctly extends the same pattern landed in commit `710c00a` (FloodAlert dtor uninit-ptr fix) to the full `Model_Data` + `ElementHotData` surface. The diff's design notes (`Model_Data.hpp:108-121`) explicitly call out:

- Production happy path (`malloc_EleRiv`/`malloc_Y`/`MD_Lake.cpp`/`MD_readin.cpp`) assigns each pointer immediately, so NSDMI default is overridden before any read.
- Failure path (NumY > 100k OOM mid-loop, exception mid-init) now sees `nullptr` instead of indeterminate bytes; `delete[] nullptr` is defined no-op (C++03+).
- Bitwise contract: only the failure branch's resource-release behavior changes (UB → defined no-op); production output bytes remain identical.

Evidence file `keliya_postfix_rivqdown_sha256.txt` + `valgrind_keliya_postfix.log:48-49` empirically verifies pre/post SHA256 byte-identical for `keliya.rivqdown.dat`. Bitwise-neutrality is structurally guaranteed (NSDMI affects no read on the happy path) — the SHA self-compare is belt-and-suspenders. Good evidence discipline.

The dual-context documentation (NSDMI block in MD_layout.hpp + lengthy paragraph in Model_Data.hpp) explains the WHY for future maintainers without burying the rationale in a CHANGELOG. Maintainability cost is appropriately paid.

---

## Invariant Matrix Coverage (mandatory 6 rows)

| # | Invariant | Coverage in this PR | Risk |
|---|-----------|---------------------|------|
| i | keliya valgrind dtor-full clean | Mac shud_asan equivalent confirmed (evidence: valgrind_keliya_postfix.log L26-32: EXIT=0, 0 ASan/UBSan errors). Linux valgrind deferred to server Phase 2. | LOW; deferred validation is documented |
| ii | heihe (NumY=19515) valgrind clean | DEFERRED to server (script `server_acceptance_cmds.sh:56-76`); Slurm budget 04:00:00 wall-clock realistic given valgrind 10-50× slowdown on heihe (~6k elements × 90 day). | MED; depends on Phase 2 |
| iii | heihe_x4 dump_adjacency dtor-full smoke | DEFERRED to server (`server_acceptance_cmds.sh:81-101`); 01:00:00 wall-clock generous for adjacency-only walk. | LOW |
| iv | heihe_x16 dump_adjacency dtor-full smoke | DEFERRED to server (same script); shared budget with heihe_x4. NumY ~485K (mesh 16×) — likely needs OS overcommit but adjacency walk is O(N) not O(N²). | MED; new scale not previously tested |
| v | B0/B1a/B1b bitwise neutrality of NSDMI fix | COVERED via pre/post SHA self-compare on keliya rivqdown.dat (matched). Structurally guaranteed (NSDMI only affects pre-malloc state). | LOW |
| vi | Mac libomp + Linux libgomp ASan dtor coverage | Mac PARTIAL (libomp; detect_leaks unsupported). Linux DEFERRED to GH runner via CI gate — **but Critical finding above means the Linux CI gate is broken**; this row's coverage is essentially absent for Linux on the existing PR diff. | HIGH (driven by Critical finding); fixable by selecting one of the gate-replacement options |

Row vi's HIGH risk is downstream of the Critical finding. Once the LEAK SUMMARY gate is repaired, vi drops to LOW.

---

## Verdict

**REQUEST CHANGES** — one Critical finding (LEAK SUMMARY gate inverts pass/fail) makes the CI workflow hard-fail on every clean run after merge. The SHUD source fix itself is sound, but the CI gate that's supposed to enforce dtor-full coverage will instead always fail green runs and require subsequent reverts/disables. Fix the gate (preferred: replace LSan-based detection with a positive sentinel printed from a `__attribute__((destructor))` hook in shud_asan), update spec design.md L113/L118/L126 to match, then merge. Sibling-class follow-up (TimeSeriesData/TabularData NSDMI) can be a separate issue.
