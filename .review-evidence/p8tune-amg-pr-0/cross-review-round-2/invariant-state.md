Reviewer agent: review-invariant-state
Review round: round 2
Reviewed head SHA: b29fe95deb22bfed859181e138bc61a8e06f534f
SHUD inner SHA: 1ab61c023ac2b93a178c2feb07aa3df509fe1a96
Summary: Phase 6 cleanly closes all 3 Critical round-1 findings — producer/validator/public-route surfaces are now coherent; baseline bitwise-neutral preserved; one minor (pre-existing) ~Model_Data → ~_Lake reachability gap re-confirmed as known-out-of-scope follow-up.

# Review: PR #400 — Invariant / State-Machine / Compatibility (round 2)

PR head (outer): `b29fe95deb22bfed859181e138bc61a8e06f534f`
SHUD submodule HEAD: `1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (openmp-baseline)
Round 1 → Round 2 outer delta: `09a815d..b29fe95` (1 commit: `b29fe95 Phase 6 invariant closure`)
Round 1 → Round 2 SHUD delta: `056a1dc..1ab61c0` (1 commit: `1ab61c0 NSDMI _TimeSeriesData/_Lake/LakeBathymetry + delete[] correctness + drop dead Model_Data::{ISFactor,windH} NSDMI`)
Files touched (outer): 4 (`SHUD` ptr bump + `serial-baseline.yml` + `dump_adjacency.cpp` + `fd_color_jacobian.cpp`). Evidence dir: cross-review-round-1/ committed in full + cross-review-round-2/ this report.

Scope (this reviewer, round 2): re-trace governing invariant post-Phase-6 across the 7 surfaces; revalidate state-machine completeness; revalidate backward-compat (ABI + baseline); audit for any sibling-surface candidate that Phase 6 missed; re-score the Invariant Matrix.

## Round-1 findings resolution

### C1 (CI gate inverted) — `RESOLVED`

`.github/workflows/serial-baseline.yml:1432` + `:1449-1469`

Phase 6 replaced the broken negative `LEAK SUMMARY:` grep with a **positive sentinel grep** on stdout for SHUD's `The successful end.` marker. Trace-verified:

- Marker emitted in `SHUD/src/ModelData/Model_Data.cpp:32` (OpenMP build) and `:40` (Serial build) inside `Model_Data::TimeSpent()`.
- `TimeSpent()` is called from `Model_Data::modelSummary(end=1)` (Model_Data.cpp:65,75 — gated on `end != 0`).
- `modelSummary(1)` is called from `SHUD/src/Model/shud.cpp:291` (SHUD coupled path) and `:507` (SHUD_uncouple path), each immediately followed by integrator teardown then `delete MD` (`:359` / `:591`).

Linear timeline on the happy path: `modelSummary(1) → TimeSpent() → printf "The successful end." → SUNContext_Free → delete MD → ~Model_Data → FreeData()`. Sentinel emission precedes dtor entry, so the gate **proves the normal-return path was reached**, NOT direct dtor completion. The CI step retains a separate `asan_err` grep (`:1446`) for the negative direction (any AAA-detected error in dtor body would emit `==<pid>==ERROR: AddressSanitizer:` and trip `:1473`). Joint: `marker_count == 0` catches `_exit/abort/SIGSEGV` bypassing main-return; `asan_err > 0` catches mid-dtor UB. The check ordering at `:1462-1469` runs sentinel-first so a coincidental ASan error does not mask the dtor-coverage signal. `detect_leaks=0` removed the need for the broken LSan path (`:1432`). Gate semantics now sound; comment block `:1415-1431` honestly documents the three prior defects.

One residual: the sentinel proves *return-path was reached*, not *dtor fully ran*. A dtor that aborts mid-FreeData would still leave sentinel in stdout AND emit ASan error — caught by joint check. A dtor that **silently** corrupts heap but doesn't trip ASan (impossible under ASan instrumentation but possible without it) would not be caught — out of scope (CI is ASan-instrumented).

### C2 (sibling closure incomplete) — `RESOLVED`

`SHUD/src/classes/TimeSeriesData.hpp:44`, `SHUD/src/classes/Lake.hpp:69-71,98-103,108-111`, `SHUD/src/classes/Lake.cpp:52-64,90-107`

Phase 6 (SHUD inner commit `1ab61c0`) addresses all three classes the round-1 verifier confirmed in scope:

1. **`_TimeSeriesData::ts[MAXQUE+1]`** — NSDMI brace-init `= {nullptr}` (TimeSeriesData.hpp:44). The dtor loop at TimeSeriesData.cpp:31-38 calls `delete[] ts[i]` unconditionally; with all slots default-nullptr, the loop is a defined no-op for any non-initialized index. 9 value-type instances live inside `Model_Data` (Model_Data.hpp:118-127) → every `~Model_Data` chain-invokes 9 `~_TimeSeriesData`. Fix correct.

2. **`_Lake` (10 raw heap ptrs)** — NSDMI `= nullptr` for `iEleLake/iEleBank/iRivIn/iRivOut/RivIn/RivOut/QEleSurf/QEleGW/QRivIn/QRivOut` (Lake.hpp:98-111). Dtor at Lake.cpp:90-107 was also fixed: scalar `delete` → `delete[]` for all five array members (`iEleBank`, `QEleSurf`, `QEleGW`, `QRivIn`, `RivOut`). The scalar-vs-array-delete mismatch was independent UB regardless of init state — Phase 6 closes both the partial-init bug AND the form-mismatch UB in the same dtor.

3. **`LakeBathymetry` (3 raw heap ptrs)** — NSDMI `= nullptr` for `index/yi/ai` (Lake.hpp:69-71). Dtor at Lake.cpp:52-64 corrected scalar `delete` → `delete[]` for all three. Symmetric to `_Lake` fix.

**`TabularData` deferred** per F3 verifier verdict: confirmed not in `~Model_Data` direct dtor chain (callers in MD_readin.cpp / MD_initialize.cpp / MD_Lake.cpp all use it as **stack-allocated local** inside read methods, never as a Model_Data member). The `nrow = 0` NSDMI in TabularData.hpp:17 already guards the unallocated case; only the partial-init-mid-`read()` exception path remains an UB candidate, and that is exercise-able only via init-time fopen failures with mid-loop allocation throws — narrow caller-local scope. Phase 6 correctly excludes it; follow-up issue tracking acknowledged in PR body.

Verified absence of *additional* missed siblings via `grep delete\[\] src/`: only `Print_Ctrl::~Print_Ctrl` (Model_Control.cpp:432-437) and `FloodAlert::~FloodAlert` (FloodAlert.cpp:27-30) still hold raw heap ptrs — both are already NSDMI-safe (`= NULL` defaults at Model_Control.hpp:21-23 and FloodAlert.hpp:17,30-32) AND counter-guarded (`if(p != NULL) delete[]`). No round-2 candidate found.

### C3 (spike binaries leak MD) — `RESOLVED`

`tools/p8tune.D/fd_color_jacobian.cpp:471-481`, `tools/p8tune.D/dump_adjacency.cpp:440-451`

Both binaries restored the matching `delete MD; delete fout; delete fin;` 3-line block before the success `return`, mirroring the error-path convention at fd_color_jacobian.cpp:308/318/432. Verified:

- `fd_color_jacobian.cpp`: deletes at L477-479 sit immediately before `return 0` at L480 — `~Model_Data → FreeData → delete[] *` chain now fires on the happy path.
- `dump_adjacency.cpp`: deletes at L447-449 sit immediately before `return ok ? 0 : 1` at L450 — same chain fires.
- Order is `delete MD; delete fout; delete fin;` — MD dtor reads pf_in/pf_out (Model_Data.hpp:47-48) so deleting MD first is correct.

Trade-off note: round-1 verifier suggested a Linus-preferred fix (stack-allocate `Model_Data MD(fin, fout);`) which would have made teardown automatic across all return paths via RAII. Phase 6 chose the matched-delete approach instead, which preserves the existing convention (error paths already use `delete MD; delete fout; delete fin;`). Functionally equivalent; current approach acceptable.

Side benefit: the new mac_asan_keliya_postfix.log Phase-6 re-verify section (L82-140) confirms both binaries run cleanly to completion (`EXIT=0`, no `free invalid ptr`, no SEGV) — direct empirical evidence the dtor chain fires.

## Phase 6 delta findings (new issues introduced by `b29fe95` + `1ab61c0`)

None.

The Phase 6 deltas only ADD defaults (NSDMI `= nullptr`) and restore explicit deletes. No new state transitions, no new code paths. The dropped `Model_Data::{ISFactor, windH}` NSDMI (F9 cosmetic — Model_Data.hpp:176-177) reverts to uninitialized raw ptrs, but both fields are confirmed dead (zero assignments / dereferences / deletes across `src/` + `tools/`); the live wind-height data lives in `Ele[i].windH` + `hot.windH` (ElementHotData keeps its NSDMI). The reversion is therefore neutral (no dtor reaches these uninit ptrs anywhere).

## Invariant Matrix Coverage (post-Phase 6)

| Row | Description | Round-1 status | Round-2 status | Change |
|-----|-------------|----------------|----------------|--------|
| (i)   | keliya valgrind/ASan (Mac, NumY=1785) | Weak (no failure mode) | **COVERED** — Phase 6 re-verify in mac_asan_keliya_postfix.log L102-140 confirms shud_asan + dump_adjacency + fd_color_jacobian all run clean (EXIT=0, sanitizer 0/0/0) | Re-confirmed on post-Phase-6 head; sentinel emitted at stdout L218 |
| (ii)  | heihe server (NumY=19515) | Deferred-with-plan | **DEFERRED Phase 2 server** — runbook in server_acceptance_cmds.sh, re-submitted on new head | unchanged |
| (iii) | heihe_x4 server (NumY≈124K) | Deferred-with-plan | **DEFERRED Phase 2 server** | unchanged |
| (iv)  | heihe_x16 server (NumY≈485K) | Deferred-with-plan | **DEFERRED Phase 2 server** | unchanged |
| (v)   | B0/B1a/B1b bitwise neutral | Weak (keliya only) | **COVERED** — keliya rivqdown SHA256 `b769e327...` matches pre-fix baseline byte-for-byte (mac_asan_keliya_postfix.log:54-55,105-107). NSDMI default-nullptr is overwritten by existing `new` assignments → output bytes unchanged by construction | Re-confirmed post-Phase-6 |
| (vi)  | Mac libomp + Linux libgomp cross-toolchain | Partial (Mac only) | **PARTIAL** — Mac libomp re-confirmed clean; Linux libgomp coverage gated on CI data_probe (no keliya forcing data on GH runner — known limitation; CI step at workflow:1413 short-circuits with `data_available=false` notice) | unchanged; documented in PR body as a known infrastructure limitation |

Net post-Phase-6: rows (i) + (v) actively COVERED on Mac with Phase-6 binaries; (ii)/(iii)/(iv) DEFERRED with runnable script + named owner (orchestrator Phase 2); (vi) partial via the (i) Mac re-confirm + Linux gated on infra (data_probe), out of PR scope.

## State-machine analysis (re-validation)

**Spike binary lifecycle** (`dump_adjacency` / `fd_color_jacobian`): `NEW → ctor(fin,fout) → use → delete MD → ~Model_Data → FreeData() → {delete[] io_ele/io_riv/io_lake, delete[] QeleSurf_flat, ..., delete[] tsd_weather, delete flood} → ~ElementHotData (implicit via Model_Data destruction) → ~_TimeSeriesData × 9 (value members) → ~_Lake × NumLake (NOT reached — see below) → ~LakeBathymetry (NOT reached) → free dtor → delete fout → delete fin → return 0`. 

Each transition is safe under partial-init: NSDMI nullptr → matched `delete[] nullptr` is C++03+ no-op. `_TimeSeriesData::ts[MAXQUE+1] = {nullptr}` brace-init ensures the `delete[] ts[i]` loop at TimeSeriesData.cpp:32-34 is safe for any subset of un-initialize()'d slots.

**No new state-machine paths introduced**. Phase 6 only narrows existing paths (eliminating UB sub-paths) and ADDs explicit deletes; no new parallel cleanup path that could race with dtor.

**Pre-existing ~_Lake reachability gap** (round-1 Critical#2 side note, re-confirmed): `Model_Data::lake` is `new _Lake[NumLake]` at MD_Lake.cpp (not re-verified here, taken from round-1 finding) but `FreeData()` at MD_readin.cpp:526-689 contains **no `delete[] lake`** (verified grep — only `delete flood` at L688). Consequence: `~_Lake` is currently unreachable via `~Model_Data` chain on every code path, including lake-having cases like qhh. Phase 6's `_Lake` NSDMI + delete[] correctness fix is therefore a **dormant safety net** on current keliya/heihe; it activates only IF/WHEN someone fixes the `Model_Data::lake` leak. This is acknowledged in the round-1 invariant-state report (Critical#2 side note: "Once anyone fixes the leak, the cascade triggers the sibling bug") and remains out-of-scope for PR-0 per round-1 phase-5-synthesis L60 ("Pre-existing un-freed 7 Model_Data ptrs (F5) — not regression, separate bug"). No action for this PR; track as follow-up issue.

## Backward compatibility (re-validation)

**ABI**: NSDMI defaults on raw array members and pointer members do NOT change `sizeof()` — initialization values are compile-time-irrelevant to type layout per C++ standard ([class.mem]/13). The `double *ts[MAXQUE + 1] = {nullptr}` brace-init in TimeSeriesData.hpp:44 changes ONLY the initial value, NOT the array element size or count. The `_Lake` and `LakeBathymetry` NSDMI additions are all single-pointer fields, identical type semantics. Verified by source diff inspection: zero member added, zero member removed, zero type changed. Round-1 Warning #5 (struct-padding measurement) remains a "could add a sizeof snapshot for evidence completeness" suggestion, not a regression blocker — Phase 6 baseline SHA256 byte-identity is sufficient empirical proof.

**CVODE wireup**: unchanged. `MD->rhs_core(Y, DY, t, policy)` signature untouched; SUNContext API unchanged.

**B0/B1a/B1b baseline neutrality**: keliya rivqdown.dat SHA256 = `b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd` (pre-fix == post-Phase-4 == post-Phase-6, per mac_asan_keliya_postfix.log L54-55 and L102-107). Bitwise identity confirms NSDMI default-nullptr is overwritten by the existing `new T[N]` assignments on every happy-path allocation site — defaults are observable only on the (previously-UB) failure-path tear-down, which is exactly the intended fix.

## Reusable-pattern audit (post-Phase-6)

Per round-1 brief lens-3, all SHUD `src/` classes with raw-ptr+dtor pattern re-scanned via `grep -nE "delete\[\]|^[[:space:]]*delete " src/`:

| Class | Raw ptr members | Dtor pattern | NSDMI status | In ~Model_Data chain? |
|---|---|---|---|---|
| `Model_Data` (member ptrs ~50) | ~50 `T *p` | `~Model_Data → FreeData() unconditional delete[]` | **NSDMI nullptr** (Phase 4, 056a1dc) | Yes (self) |
| `ElementHotData` (~30 SoA fields) | ~30 `T *p` | `~Model_Data → FreeData() unconditional delete[] hot.*` | **NSDMI nullptr** (Phase 4, 056a1dc, MD_layout.hpp) | Yes (Model_Data value-member) |
| `_TimeSeriesData` | `double *ts[MAXQUE+1]` raw array | dtor loop unconditional `delete[] ts[i]` | **NSDMI `{nullptr}` brace-init** (Phase 6, 1ab61c0) | Yes (9 value-members in Model_Data) |
| `_Lake` | 10 raw `T *p` | counter-guarded `delete[]` (Phase 6 form-corrected from scalar `delete`) | **NSDMI nullptr** (Phase 6, 1ab61c0) | Yes via `Model_Data::lake` (but pre-existing leak makes chain unreachable — see state-machine analysis) |
| `LakeBathymetry` | 3 raw `T *p` | counter-guarded `delete[]` (Phase 6 form-corrected) | **NSDMI nullptr** (Phase 6, 1ab61c0) | Yes via `_Lake::bathymetry` |
| `FloodAlert` | 4 raw `T *p` (`pstage/pflux/itype/para`) | counter-guarded `delete[]` (`if(p != NULL)`) | **NSDMI `= NULL`** (already, per 710c00a) | Yes via `Model_Data::flood` (`delete flood` at MD_readin.cpp:688) |
| `Print_Ctrl` (in Model_Control.cpp) | 3 raw `T *p` (`PrintVar/buffer/icol`) | counter-guarded `delete[]` (`if(p != NULL)`) | **NSDMI `= NULL`** (already) | Indirectly via `Control_Data CS` member |
| `TabularData` | `double **x` | counter-guarded `delete[]` (`if(nrow > 0)`) | `nrow=0` NSDMI guards unallocated case; partial-init still UB | **NO** (stack-allocated local in MD_readin/MD_Lake/MD_initialize callers) → out of `~Model_Data` chain → deferred per F3 verifier |

No additional sibling class with raw-ptr+dtor pattern missed by Phase 6.

## Failure-path surface (signal/exception)

NSDMI defaults + matched-delete pattern → dtor on partial-init is safe across all surfaces in the table above EXCEPT TabularData read-path (caller-local, deferred). No exception-throwing code path in the audited classes (`new` failure throws but ctor body is otherwise trivial); even if `new` throws mid-`malloc_EleRiv()`, all surviving fields are nullptr → dtor is no-op chain.

## Evidence/audit surface

cross-review-round-1/ (10 files, 1119 lines, committed in full per Phase 5 synthesis L67) + cross-review-round-2/ (this report, plus 5 other reviewer reports being authored in parallel) + the Phase 6 re-verify section in mac_asan_keliya_postfix.log L82-140. Auditability complete.

## Findings (round 2)

#### 🟡 Warning: dormant sibling fix — `_Lake` dtor reachable only after pre-existing leak is fixed

`SHUD/src/ModelData/MD_readin.cpp:526-689` (`FreeData()`)

The Phase 6 `_Lake` + `LakeBathymetry` NSDMI + delete[] correctness fix is a sound dtor invariant in isolation, but the dtor itself is unreachable via the current `~Model_Data` chain on any case: `Model_Data::lake` is `new _Lake[NumLake]` somewhere in MD_Lake.cpp (round-1 finding) but `FreeData()` has no `delete[] lake` (verified — only `delete flood` at L688 for the per-class members). So PR-0's `_Lake` fix activates only IF/WHEN a future PR fixes the lake-array leak.

This is a known follow-up tracked in round-1 phase-5-synthesis L60 (out-of-scope for PR-0 per Phase 5 boundary). NOT a regression. Phase 6 still does the right thing (closing the dtor anti-pattern AHEAD of the leak fix) — when the leak is fixed, the chain Just Works rather than triggering #386 recurrence at lake scale. Document as known dormant + follow-up issue.

#### 🔵 Suggestion: sentinel proves return-path, not dtor-completion — weak coupling worth noting

`.github/workflows/serial-baseline.yml:1449-1469`

The positive `'The successful end.'` sentinel emits from `TimeSpent()` BEFORE the `delete MD → ~Model_Data → FreeData()` chain runs (Model_Data.cpp:32/40 inside TimeSpent → modelSummary(end=1) at shud.cpp:291/507 → SUNContext_Free → delete MD at :359/:591). The gate therefore proves *main reached the post-modelSummary section*, not *dtor ran to completion*. The latter is covered jointly by the `asan_err` grep at :1446 — any UB in `FreeData()` would trip `==<pid>==ERROR: AddressSanitizer:` and fail at :1473.

This joint-check semantics is sound (sentinel covers _exit/abort bypass; asan_err covers mid-dtor UB), but the workflow comment at :1454-1456 ("The marker is generated IFF the C++ dtor chain ran to completion") slightly overstates: the marker is generated iff main reached *return*, which is a *necessary* (not sufficient) precondition for dtor completion. Suggestion: refine the comment to "generated iff main returned normally → dtor *will be invoked*; combined with asan_err==0 → dtor *ran cleanly to completion*".

#### 🔵 Suggestion: lake-array leak follow-up issue should be filed before next PR base from main

`SHUD/src/ModelData/MD_readin.cpp:526-689` + `SHUD/src/ModelData/MD_Lake.cpp:~36`

The `Model_Data::lake` array leak (round-1 Critical#2 side note) is a known carry-over. With Phase 6 closing the `_Lake` / `LakeBathymetry` dtor invariants, fixing the leak in a follow-up PR is now safe — without Phase 6, fixing the leak would have re-triggered #386 at any lake-case (qhh). File a tracking issue with explicit dependency on this PR's `1ab61c0` so the follow-up doesn't accidentally invert the order.

#### 🟢 Praise: Phase 6 closure is the textbook "fix what was found, document what was deferred" pattern

`b29fe95` commit body (lines 1-58) + `1ab61c0` commit body + Phase 5 synthesis

The Phase 6 commit body explicitly maps every fix back to (F1/F2/F3/F9) and the corresponding verifier-confirmed finding. The "F9 cosmetic" honest acknowledgement (dropping dead ISFactor/windH NSDMI rather than silently leaving them) is exactly the right move — neither over-fix nor under-fix. TabularData deferral with verifier rationale (caller-local, not in ~Model_Data chain) is clean scope discipline. The Phase 6 re-verify section appended to mac_asan_keliya_postfix.log L82-140 is the gold-standard "show the bits" empirical proof: same SHA256, sentinel emitted, ASan clean across all 3 binaries.

#### 🟢 Praise: `delete[]` form correction unbundled correctly from NSDMI fix

`SHUD/src/classes/Lake.cpp:52-64,90-107`

Phase 6 explicitly fixes the scalar-`delete`-on-`new[]`-storage UB in `~_Lake` and `~LakeBathymetry` as an *independent* sub-fix from the NSDMI default-init. The commit body honestly calls this out ("Also fixed scalar `delete` → `delete[]` mismatches on `new[]`-allocated arrays in Lake.cpp:53-58 + 85-95 (independent UB regardless of init)"). Two-fix-in-one is acceptable when the second is uncovered by the first; the commentary inside Lake.cpp:53-58 + 91-95 explains why both changes belong together. Good attribution discipline.

## Verdict

**APPROVE** — All 3 round-1 Critical findings cleanly resolved at the source. Phase 6 introduces zero new invariant/state-machine regressions. Backward compatibility verified bitwise (SHA256 byte-identical to baseline). Sibling-surface audit re-confirmed (no missed candidates beyond TabularData, which is correctly out of `~Model_Data` chain). The two warnings + two suggestions above are forward-looking nudges, not blockers; the third (lake-array leak follow-up) was already deferred in Phase 5 synthesis. Server-side rows (ii)/(iii)/(iv) of the Invariant Matrix remain DEFERRED per orchestrator Phase 2 runbook — acceptable as PR-0 in the multi-PR p8tune.F epic per CLAUDE.md staging policy.
