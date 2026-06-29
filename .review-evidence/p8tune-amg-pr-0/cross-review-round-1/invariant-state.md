# Review: PR #400 — Invariant / State-Machine / Compatibility (round 1)

PR head: `09a815dbcab9eabbddcad9550c706ddfa8636519`
SHUD submodule HEAD: `056a1dce4b75e79779242b4796e178bebe89680b` (openmp-baseline)
Branch: `feat/issue-394-p8tune-amg-spike`
Scope (this reviewer): Invariant Matrix coverage, dtor state machine completeness, sibling-surface audit, backward-compat, ASan/CI gate semantics.
Working tree audited under: `/Users/danker/Desktop/Hydro-SHUD/openMP/`

## Summary

The NSDMI fix for `Model_Data` + `ElementHotData` is correct in its narrow scope and the source-of-truth diff (`SHUD` 710c00a → 056a1dc) exactly matches the evidence file `shud_source_fix.diff` (byte-identical, 367 lines, two files). Bitwise neutrality holds on keliya (rivqdown SHA256 identity). However, the PR claims **closure** of the dtor-uninit invariant across the whole `Model_Data` → `FreeData` → `~SubClass` chain, and that claim is **incomplete**: there are at least two unfixed sibling surfaces that exhibit the same "uninit raw ptr + unconditional `delete[]` in dtor" pattern the PR is supposed to eradicate. Additionally, the new CI gate's `LEAK SUMMARY:` grep is logically inverted — on a leak-clean run the gate fails. Verdict-class: **REQUEST CHANGES**.

## Findings

### Critical: CI dtor-coverage gate inverts on leak-clean runs (will hard-fail green Linux jobs)
`.github/workflows/serial-baseline.yml:1427` + `:1449` + `:1463-1467`

The new env is `ASAN_OPTIONS: "detect_leaks=1:halt_on_error=1:print_stacktrace=1:exitcode=0:print_suppressions=0"`. The gate then asserts:

```
leak_summary_count=$(grep -cE '^==[0-9]+==(ERROR: LeakSanitizer:|SUMMARY: AddressSanitizer: detected memory leaks)|LEAK SUMMARY:' sanitizer_run_${{ matrix.case }}.stderr.log || true)
...
if [[ "$leak_summary_count" == "0" ]]; then
  echo "::error title=ASan dtor coverage missing (...)"
  exit 1
fi
```

LSan only emits `ERROR: LeakSanitizer:` / `SUMMARY: AddressSanitizer: detected memory leaks` / `LEAK SUMMARY:` **when leaks are actually detected**. The `__lsan_do_leak_check` atexit hook (libsanitizer/lsan/lsan_common.cpp) silently returns when the leak set is empty — no stderr output. The author's claim in `valgrind_keliya_postfix.log` ("On Linux GH runner the flag flip from 0→1 produces `LEAK SUMMARY:` stderr line") is empirically untested on this codebase under LSan and is contrary to documented LSan behavior. The PR's prior evidence shows Mac stderr is *empty* under the same option set (Mac LSan port absent, but the asymmetry undermines the assumed cross-toolchain behavior).

Concrete consequence:
- If `shud_asan keliya` is leak-clean (the expected post-fix state since `delete MD` at `SHUD/src/Model/shud.cpp:359/591` triggers `~Model_Data → FreeData`, and `fin/fout` are deleted at `:606-607`), then `leak_summary_count == 0` and the job **fails with "dtor coverage missing"** — exactly the opposite of intent.
- If the run leaks, both gates fire (`asan_err` first), so the gate appears to work, but only because of pre-existing leaks (e.g. SUNDIALS internal allocations, the `Model_Data::lake` array that is `new`'d but never `delete[]`'d — see `MD_Lake.cpp:36` vs absence in `FreeData()`).

Why this matters for the invariant claim: the gate is supposed to be the canonical regression sentinel for future `_exit/abort` reintroductions. As written, it actually rewards leaky teardown and punishes clean teardown. Future contributors who reduce a leak will silently break CI.

Fix options:
- Use ASan `verbosity=1` and grep for the LSan "Checking for leaks" line (printed unconditionally when `detect_leaks=1` runs), e.g. `__lsan::DoLeakCheck` does emit `Checking for memory leaks` only at verbosity≥1.
- Inject a deliberate small leak in a test-only path so the SUMMARY always appears (anti-pattern — pollutes prod).
- Replace LSan-based dtor proof with a strace-based or LD_PRELOAD-counter check on `_exit/abort` system calls.
- Or simply use a Model_Data dtor-trace marker (printf at the end of `~Model_Data`) and grep for it in the stdout log; the marker presence is a direct, unambiguous dtor-completion signal.

Status: needs verification by an actual `make shud_asan && ASAN_OPTIONS=... ./shud_asan keliya` on Linux GH runner. **Until verified, the gate cannot be trusted to mean what it claims**.

### Critical: invariant closure leaves at least two sibling-surface ptr+dtor patterns unfixed
`SHUD/src/classes/Lake.cpp:84-95` (`_Lake::~_Lake`), `SHUD/src/classes/Lake.cpp:53-58` (`LakeBathymetry::~LakeBathymetry`), `SHUD/src/classes/TimeSeriesData.cpp:31-37` (`_TimeSeriesData::~_TimeSeriesData`)

Per the brief's "Reusable unsafe pattern → audit full sibling surface" requirement, the governing invariant is *all* SHUD classes with raw-ptr members + unconditional `delete[]` in dtor must be NSDMI-safe. The PR audits `Model_Data` + `ElementHotData` + `FloodAlert` (already in 710c00a) but misses:

1. **`_TimeSeriesData::~_TimeSeriesData()`** (`SHUD/src/classes/TimeSeriesData.cpp:31`):
   ```
   for (int i = 0; i < MAXQUE; i++) {
       delete[] ts[i];
   }
   ```
   `ts[MAXQUE+1]` declared at `TimeSeriesData.hpp:41` as `double *ts[MAXQUE + 1];` with **no default-init**. `ts[i]` is assigned only in `_TimeSeriesData::initialize()` (`TimeSeriesData.cpp:56`). If a `_TimeSeriesData` instance is destroyed without ever calling `initialize()`, every `delete[] ts[i]` operates on indeterminate stack/heap bytes — the same UAF/heap-corruption signature as #386.
   
   This matters because `Model_Data` holds **eleven** value-typed `_TimeSeriesData` members: `tsd_LAI, tsd_MF, tsd_eleSS, tsd_eyBC, tsd_eqBC, tsd_ryBC, tsd_rqBC, tsd_lyBC, tsd_lqBC` (lines 118-127 of Model_Data.hpp, plus `tsd_weather` heap-ptr now defaulted). On every `~Model_Data`, the eleven sub-TSD dtors run **unconditionally**. Any load path that aborts before all eleven TSDs are `initialize()`d (e.g. lake-disabled cases never call `read_tsLake*`) triggers the bug at scale — but the brief's invariant explicitly demands closure at NumY=485K where this becomes statistically likely.
   
   Fix: NSDMI `double *ts[MAXQUE + 1] = {};` (zero-initializes all slots) + guard `if (ts[i]) delete[] ts[i];` in dtor.

2. **`_Lake::~_Lake()`** (`SHUD/src/classes/Lake.cpp:84-95`):
   ```
   if(NumEleBank > 0){
       delete iEleBank;     // wrong: array allocated with new[], scalar delete = UB
       delete QEleSurf;     // ditto
       delete QEleGW;       // ditto
   }
   if(NumRivIn > 0){ delete QRivIn; }
   if(NumRivOut > 0){ delete RivOut; }
   ```
   Two layered bugs: (a) integer-counter guard does not catch the partial-init window (NumEleBank assigned at `readLake()` BEFORE `new int[NumEleBank]` — abort/OOM between leaves pointer indeterminate while counter > 0); (b) scalar `delete` on `new[]`-allocated arrays is undefined behavior on the happy path. This is unreachable under keliya/heihe (no lakes) but is on path for qhh and any heihe_x16-with-lake variant. The brief explicitly names heihe_x16 (485K NumY) as an in-scope row.
   
   The same scalar-delete-of-array UB also lives in `LakeBathymetry::~LakeBathymetry()` (`Lake.cpp:53-58`).

3. Side note (pre-existing, not introduced by PR-0): `Model_Data::lake` is `new _Lake[NumLake]` (`MD_Lake.cpp:36`) but **never** `delete[]`-d in `FreeData()`. This is an outright leak — and it is what would let the broken `~_Lake` lie dormant. Once anyone fixes the leak, the cascade triggers the sibling bug.

The PR's commit message states "Closes part of #386 (full closure pending server-side heihe_x4/x16 dump_adjacency dtor-full validation in orchestrator Phase 2)" — but the orchestrator Phase 2 plan in `server_acceptance_cmds.sh` only verifies the two binaries don't crash; it does not audit `_TimeSeriesData` / `_Lake` / `LakeBathymetry` and will not catch these regressions when a lake case is exercised. The brief's "Reusable unsafe pattern" requirement is **not** satisfied.

### Critical: tools/p8tune.D/fd_color_jacobian.cpp success-path leaks MD/fout/fin, defeating dtor-coverage intent
`tools/p8tune.D/fd_color_jacobian.cpp:472-473`

The whole point of swapping `_exit(0)` → `return 0` (per spec amg-pattern-spike-verdict REQ-3 Scenario "`_exit(0)` workaround removal") is to let the C++ destructor chain run. The success-path return at line 473 is reached after `Model_Data *MD = new Model_Data(fin, fout)` at line 286 and `FileIn *fin / FileOut *fout` `new` earlier. Error paths at lines 308, 318, 432 correctly `delete MD; delete fout; delete fin;` — **but the success path does not**. So on the success exit, the heap-allocated `MD`, `fout`, `fin` leak; their dtors **never run**; the post-fix `~Model_Data → FreeData → delete[] *` chain is exercised **zero times** on the happy path of this binary.

This means: (a) the spec REQ-3 dtor-coverage claim for `fd_color_jacobian` is structurally false; (b) any future LSan run on this binary will report MD+fout+fin as leaks; (c) if the CI gate is fixed (per critical #1 above) it would actually catch this here, but the gate runs `shud_asan`, not `fd_color_jacobian`.

`dump_adjacency.cpp:442` has the same issue: `MD` allocated at the corresponding init block is leaked on success (please cross-check at line ~430 in your editor — I confirmed `delete MD` is absent in the success path tail and `return ok ? 0 : 1` is reached with `MD` still live).

Fix: before the success `return 0`, add `delete MD; delete fout; delete fin;` mirroring the error-path teardown. Or convert to `std::unique_ptr<Model_Data>` for both binaries and let RAII handle teardown across all return paths.

### Warning: Invariant Matrix Row vi ("Mac libomp + Linux libgomp cross-toolchain") is partial, brief's lens-2 test fails
`.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log` (Acceptance section)

The evidence file itself states:
```
- Row vi (Mac libomp + Linux libgomp): PARTIAL (Mac covered; Linux deferred)
```
The brief's critical lens 2 explicitly asks "Does the cross-toolchain row (vi) cover both Mac libomp AND Linux libgomp at heihe-scale or only keliya-scale?". Answer: Mac is keliya-scale only (NumY=1785), Linux is fully deferred. The matrix row is **not** closed by this PR. The brief allows "server-deferred rows (ii)(iii)(iv): is there a written, executable plan + responsible party". `server_acceptance_cmds.sh` is well-structured and executable, AND there is a responsible party (orchestrator Phase 2). So Row vi has the same status as ii/iii/iv — deferred with a runbook. That satisfies the brief's "more than just TODO" bar.

However, the brief's lens 1 ("Is the evidence ACTUALLY testing the failure mode (UAF at NumY > 100k), or is it just baseline (no failure mode to test)?") is the real concern: Mac keliya at NumY=1785 has **no failure mode to test** — the UAF only manifested at heihe_x4 (NumEle=40046, NumY~124K). So the COVERED row v (bitwise neutral on keliya) and COVERED row i (keliya valgrind/ASan clean) both run at a scale far below the failure threshold. The actual fix-validation evidence lives entirely in the deferred server rows. The PR's effective on-platform validation is "fix did not break the small case". That's necessary but **not sufficient** evidence of the invariant closure claim in the head commit message.

The PR should be honest about this in its commit message: change "Closes part of #386" to something like "Resolves the diagnosed root cause for #386 at small scale; large-scale validation deferred to orchestrator Phase 2".

### Warning: state-machine transition gap — partial-init failure path between `NumX = ...` and `arr = new ...`
`SHUD/src/ModelData/MD_readin.cpp` (general pattern) + `SHUD/src/classes/Lake.cpp:99-129` (readLake)

The fix protects against the trivial "ctor ran but malloc_EleRiv never ran" failure mode. It does **not** protect against the more realistic mid-`malloc_EleRiv()` OOM path. Many SHUD init functions follow the pattern `NumX = parse_from_file(); arr = new T[NumX];`. If allocation fails between the count assignment and the `new`, the integer-counter > 0 guard pattern (used by `_Lake`, `LakeBathymetry`, and even the new `if(NumLake > 0) { delete[] yLakeStg; ... }` block at `MD_readin.cpp:600-608`) will pass the guard, then deref an uninitialized pointer.

The PR's NSDMI fix correctly handles the Model_Data / ElementHotData heap arrays via `= nullptr` (deletes become no-op even if alloc failed mid-way). But the integer-counter pattern in `_Lake` / `LakeBathymetry` / `if(NumLake > 0)` blocks is **not** equivalent. The transition `partially-set-NumX → alloc-failed → arr-still-indeterminate → dtor-fires` is unsafe in those code paths. The brief asks for "verify all dtor transitions in the chain are safe under partial-init failure" — they are not, for the lake substructure.

### Warning: backward-compat ABI claim relies on no struct padding change, not actually verified
`shud_source_fix.diff` (both files)

The PR comments assert "Bitwise contract: NSDMI default-init does NOT change the writes that subsequently land in these pointers, so SHUD output bytes remain neutral vs B0/B1a/B1b baselines." This is plausibly true because (a) no member added/removed, (b) initializer is constant-expression `nullptr`, (c) struct layout is unchanged. But the PR provides **no `sizeof(Model_Data)` before/after check** and **no offset-of dump**. On rare compilers (esp. cross-toolchain — the very platform axis the brief stresses), aggregate-initialization rules and `[[no_unique_address]]` interactions can change padding when NSDMI is added to a previously trivially-initialized struct. Empirical bitwise neutral on keliya rivqdown is *output* identity, not *struct layout* identity — they coincide here by happy accident (assignments overwrite the defaults), but the compatibility argument is asserted, not measured.

Mitigation: add a tiny script that prints `sizeof(Model_Data)` + `offsetof(Model_Data, hot)` + `offsetof(Model_Data, NumLake)` (representative pre/post-NSDMI fields) and snapshots them in the evidence dir.

### Suggestion: comments in `dump_adjacency.cpp:434-439` cross-reference the wrong file
`tools/p8tune.D/dump_adjacency.cpp:434-439`

The comment says "see fd_color_jacobian.cpp main exit comment for the full rationale." That is fine for context but the dump_adjacency call site is the simpler binary and reviewers should see the rationale here. Inline a 2-line summary at this site (or extract the rationale into a shared header comment) so the dependency-on-sibling-file is not a stumble for future readers.

### Suggestion: invariant matrix's "evidence/audit" row claims 5 files but the actual contents are operational, not analytical
`.review-evidence/p8tune-amg-pr-0/`

The 5 evidence files are runbook + log + SHA + diff. There is no written audit of the sibling-surface check (the brief's "audit full sibling surface" output). Adding a `sibling_surface_audit.md` that enumerates every SHUD class with raw-ptr+dtor (and the verdict per class: safe / fixed / needs-fix) would close the brief's audit requirement explicitly.

### Praise: SHUD diff vs evidence diff is byte-identical
`SHUD/` (056a1dc) ↔ `.review-evidence/p8tune-amg-pr-0/shud_source_fix.diff`

`diff <(cd SHUD && git diff 710c00a 056a1dc) shud_source_fix.diff` returns no output. The evidence file is an exact transcript of the actual change — not summarized, not re-rendered. This is the best evidence-anchoring pattern: future investigators can rebuild the diff from the snapshot without trusting the author. The `shud_fix_sha.txt` correctly records the parent SHA `710c00a4` and fix SHA `056a1dce4b75e79779242b4796e178bebe89680b`, both verified against the actual submodule HEAD.

### Praise: `_exit → return` swap rationale is correct and well-justified
`tools/p8tune.D/{dump_adjacency,fd_color_jacobian}.cpp` comments

Once the leak in Findings #3 is fixed, this swap is the right move. The comments correctly identify the dtor-coverage motive and link to the spec scenario. The flag-include update (`<unistd.h>  // chdir, optind` instead of `// _exit`) is a nice attention-to-detail bonus.

## Invariant Matrix Coverage Scoreboard

Applying the brief's three critical lenses to each of the 6 rows:

| Row | Description | Lens 1 (tests failure mode?) | Lens 2 (cross-toolchain?) | Lens 3 (deferred + plan + owner?) | Status |
|-----|-------------|------------------------------|---------------------------|------------------------------------|--------|
| i   | keliya valgrind (Mac) | No — NumY=1785 < failure threshold | Mac libomp only | n/a (covered) | **Weak** |
| ii  | heihe (NumY=19515) | Borderline — closer but still < UAF threshold | Linux libgomp deferred | Yes, runbook in server_acceptance_cmds.sh, owner = orchestrator Phase 2 | **Deferred-with-plan** |
| iii | heihe_x4 (NumY=124K) | YES — original failure case | Linux libgomp | Yes, runbook + owner | **Deferred-with-plan** |
| iv  | heihe_x16 (NumY=485K) | YES — exceeds threshold | Linux libgomp | Yes, runbook + owner | **Deferred-with-plan** |
| v   | B0/B1a/B1b bitwise neutral | n/a (compat check, not failure check) | Mac keliya only | n/a (covered) | **Weak** (keliya only) |
| vi  | Mac libomp + Linux libgomp | Partial | PARTIAL | Plan exists for Linux | **Deferred-with-plan** |

Net: only row i is "covered now" by the brief's strict reading, and even that doesn't exercise the failure mode. The PR is structurally "fix the diagnosed root cause, run small-case smoke, defer real validation to server orchestration". That is **acceptable as a PR-0 in a multi-PR epic**, but the head commit's "resolve #386" wording overclaims. Closer to honest: "diagnose + fix + small-case smoke; full closure pending server validation Phase 2".

## State-Machine Analysis: dtor lifecycle under failure

Brief's stated transition:
- Pre-fix: NEW (uninit) → assigned-by-malloc_EleRiv → ... but if malloc_EleRiv aborted mid-way, leftover ptrs in NEW state → dtor `delete[]` on uninit → CORRUPTION
- Post-fix: NEW → NSDMI-default-init (nullptr) → assigned-if-malloc-succeeded → ... if abort, ptr stays nullptr → dtor `delete[] nullptr` = no-op → SAFE

This is correct **for the ~50 Model_Data heap ptrs + ~30 ElementHotData ptrs covered by the diff**. It is **incorrect** for:
- `_TimeSeriesData::ts[]` (see Critical #2.1) — still NEW (uninit) → dtor → CORRUPTION
- `_Lake::iEleBank / QEleSurf / QEleGW / QRivIn / RivOut` and `LakeBathymetry::index / yi / ai` — guarded by integer counter that can be partial-set; still corruption window exists
- `Model_Data::lake` (heap _Lake array) is never deleted, so the per-element `~_Lake` cascade is unreachable for now — but only by virtue of an unrelated bug; any future cleanup would expose the issue

State machine is partially correct, not closed across the chain.

## Identity collapse: OK
`Model_Data` is one-per-process in `./shud`. No sibling/aggregate identity risk. The PR does not change this.

## Backward compat for unchanged consumers

- **ABI**: unverified at struct-layout level (see Warning #5). Plausibly safe by C++ rules; not measured.
- **CVODE wireup**: unchanged. `MD->rhs_core(Y, DY, t, policy)` signature untouched. OK.
- **Baselines B0/B1a/B1b**: keliya rivqdown bitwise neutral confirmed. Other cases deferred. The brief asks "Verify for heihe (Mac runnable) too if possible" — that check is NOT in the evidence dir. Heihe is the Mac runnable larger case (NumY=19515) and should be feasible locally before relying on server Phase 2.

## Verdict

**REQUEST CHANGES** — three Critical findings:

1. CI dtor-coverage gate is logically inverted: clean leak-free runs (the post-fix expected state) will hard-fail with "dtor coverage missing".
2. Sibling-surface invariant closure is incomplete: `_TimeSeriesData::ts[]` (11 value members in Model_Data exercise this dtor on every shud exit), `_Lake::~_Lake`, and `LakeBathymetry::~LakeBathymetry` exhibit the exact same unsafe pattern PR-0 is supposed to eradicate. The "Closes part of #386" wording overclaims.
3. `tools/p8tune.D/fd_color_jacobian.cpp` and `dump_adjacency.cpp` success paths leak `MD/fout/fin`, so the `_exit → return` swap does NOT achieve the dtor-coverage intent for those binaries.

Plus two Warnings worth addressing before merge:
4. Invariant Matrix's actually-validated rows do not exercise the documented failure mode (NumY > 100k) — the PR's effective on-platform proof is "small case still works".
5. State machine is incomplete under partial-init failure for the integer-counter-guarded surfaces (lake substructure).

The diff itself is well-crafted and the evidence-anchoring (diff ↔ source byte-equality) is exemplary. The fix is correct in its narrow scope and is a useful prereq for downstream PR-A. The blocker is the over-broad invariant-closure claim and the inverted CI gate. Addressing findings 1 + 2 + 3 (with 4 acknowledged in commit message and 5 noted as known follow-up) would change the verdict to **APPROVE**.

