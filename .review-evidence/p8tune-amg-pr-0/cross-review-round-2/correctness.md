# Round 2 Correctness Review — PR #400 (P8-tune.F PR-0) Phase 6 delta

- **Reviewer agent**: review-correctness (leaf, read-only)
- **Review round**: round 2
- **Reviewed head SHA**: `b29fe95deb22bfed859181e138bc61a8e06f534f`
- **SHUD inner SHA**: `1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (parent `056a1dc` → `1ab61c0` on `openmp-baseline`)
- **Round 1 head**: `09a815dbcab9eabbddcad9550c706ddfa8636519`
- **Round 1 owner findings**: F1 (spike binary success-path MD leak — Critical), F2 (CI LEAK SUMMARY gate logically unreachable — Critical). Both CONFIRMED by Phase 4.5 verifier.

---

## Summary

**Both round-1 Critical findings (F1, F2) are RESOLVED in the Phase 6 delta.** The sibling-class invariant closure (F3) and dead-field cleanup (F9) on the SHUD inner pointer bump are correctness-sound and bitwise-neutral (post-fix Mac `rivqdown.dat` SHA256 = `b769e327…c028bd` = pre-fix baseline, evidence: `mac_asan_keliya_postfix.log:106`). No new correctness regressions surfaced on the Phase 6 delta.

One non-blocking observation (Minor): the F2 "positive sentinel" gate proves "main loop + `modelSummary(1)` ran", which is a strict subset of "dtor chain ran to completion" — the sentinel is printed at `shud.cpp:291` / `shud.cpp:507` BEFORE the final `delete MD;` at `shud.cpp:359` / `shud.cpp:591`. The gate is still correct (the ASan `halt_on_error=1` + `asan_err != 0` second-stage check catches any dtor-time UB), but the implementer's claim that "marker is generated IFF the C++ dtor chain ran to completion" is slightly stronger than what the code actually proves. See Observation O1 below.

---

## Round-1 findings resolution

### F1 (spike binary MD leak → dtor not exercised on success path): RESOLVED

**Evidence — `tools/p8tune.D/dump_adjacency.cpp:440-451`** and **`tools/p8tune.D/fd_color_jacobian.cpp:471-481`**:

```cpp
// dump_adjacency.cpp end of main:
std::fflush(stdout);
std::fflush(stderr);
delete MD;
delete fout;
delete fin;
return ok ? 0 : 1;
```

Verifications performed:

1. **Placement order**: `fflush(stdout) → fflush(stderr) → delete MD → delete fout → delete fin → return`. Correct ordering — streams flushed first (so any dtor-time abort doesn't lose buffered output); MD deleted before fout/fin (matches construction-order-reversed: `fin` then `fout` constructed at L398-399 of dump_adjacency, then `MD = new Model_Data(fin, fout)` at L402; reversed dtor order is `MD, fout, fin`); return last. Symmetric with the error paths at `fd_color_jacobian.cpp:308`, `:318`, `:432` which use the same `delete MD; delete fout; delete fin; return 1;` triple. **PASS**.

2. **OOM-mid-`new` edge case**: If `new Model_Data(fin, fout)` itself throws (OOM), `MD` is never assigned and control jumps out of `main` via unhandled exception — `delete MD` is NEVER reached. Safe because (a) C++ guarantees `MD` is not assigned if `new` throws, so the variable holds whatever the previous statement left (uninitialized for `Model_Data *MD = new ...;` on first use, but execution never reaches `delete MD`), and (b) an unhandled exception in `main` calls `std::terminate` → `_exit`, which bypasses dtors but also bypasses the gate. **No regression vs round-1 state**.

3. **BUILD-step earlier failure**: Both spike binaries pre-flight check `fopen(cfg_path)` (dump_adjacency:366) and `chdir` (dump_adjacency:375) BEFORE allocating `fin/fout/MD`; early `return 1` from those checks runs ZERO heap deletes — that's correct because nothing was allocated yet. The new deletes only execute on the path where `new MD` succeeded. **PASS**.

4. **Dtor chain coverage**: Per Phase 6 re-verify log (`mac_asan_keliya_postfix.log:123-130`), `./dump_adjacency --case keliya` exits 0 and `~Model_Data → FreeData()` chain "fires on success path (no free invalid ptr)". The new code is functionally exercised end-to-end on Mac at keliya scale. Server-scale validation (heihe / heihe_x4 / heihe_x16) remains deferred to orchestrator Phase 2 per `server_acceptance_cmds.sh`. **PASS for Mac scale; server scale = deferred but unchanged from round-1 scope**.

**Verdict on F1**: closure is correct, complete, and matches the round-1 "Option 1 fix" recommendation verbatim (`delete MD; delete fout; delete fin; return 0;`). The PR description's "destructor chain runs to completion" claim is now factually true for spike binaries on the success path.

---

### F2 (CI LEAK SUMMARY gate logically unreachable on leak+dtor-full path): RESOLVED

**Evidence — `.github/workflows/serial-baseline.yml:1412-1478`**:

The delta swaps the three-way-broken negative gate (`detect_leaks=1` + grep `LEAK SUMMARY:`) for a positive sentinel gate (`grep -cE 'The successful end\.' sanitizer_run_…stdout.log`).

Verifications performed:

1. **Sentinel string is the canonical SHUD normal-exit marker**: `grep -rn 'The successful end' SHUD/src/` returns 2 hits, both at `SHUD/src/ModelData/Model_Data.cpp:32` (OMP-enabled path) and `:40` (serial path), inside `Model_Data::TimeSpent()`. `TimeSpent()` is called from `modelSummary(int end=1)` at `Model_Data.cpp:65` / `:75`. `modelSummary(1)` is invoked at `shud.cpp:291` (`SHUD()` coupled path) and `shud.cpp:507` (`SHUD_uncouple()` path). Both invocations are on the normal flow AFTER `MainLoop` / `MD->summary()` / `MD->CS.ExportResults()` and BEFORE the final `delete MD;`. If `_exit(0)` / `abort()` / SIGSEGV bypasses this code, the marker is absent — gate fires. **PASS — sentinel is the right canonical marker**.

2. **stdout/stderr split is correct**: workflow lines 1437-1439 redirect stdout to `sanitizer_run_${{ matrix.case }}.stdout.log` (FD 1) and stderr to `…stderr.log` (FD 2) separately. SHUD's `screeninfo()` (which emits the sentinel) is defined in `SHUD/src/print.cpp` and writes to stdout via `fprintf(stdout, …)` (verified via `grep -n screeninfo SHUD/src/print.*`). Marker therefore lands in stdout.log — the file the new gate greps. **PASS**.

3. **Gate ordering**: positive sentinel checked FIRST (workflow L1465-1469), THEN ASan/UBSan errors/warnings (L1473-1477). Comment at L1462-1464 explicitly states the rationale: "check positive sentinel FIRST so the dtor-coverage regression signal isn't masked by an ASan error that happens to fire on the same broken run." Order is correct: a regression that bypasses normal exit AND triggers ASan would fire BOTH gates, but the sentinel gate would attribute the failure to dtor-bypass first (which is the more specific signal). **PASS**.

4. **Case-sensitive regex match**: `grep -cE 'The successful end\.'` — the `.` is correctly escaped via `\.`; the regex matches the literal string `The successful end.` (case-sensitive, with the period as a literal). Verified the actual emitted string at `Model_Data.cpp:32`/`:40` is exactly `"\n\nThe successful end. \n\n"` — the trailing space after `.` is included in the emit, the regex only requires the prefix up through the period, so match is positive on both emission sites. **PASS**.

5. **YAML lint**: `uv run --with pyyaml python -c "yaml.safe_load(open('.github/workflows/serial-baseline.yml'))"` parses cleanly; 4 jobs visible (`setup, tools-tests, build-and-compare, asan-ubsan`); `asan-ubsan` step list intact. **PASS**.

6. **Round-1 root cause closed**: the prior `asan_err` regex `==.*==ERROR` no longer competes with a `LEAK SUMMARY` check because the leak-summary gate is gone. `detect_leaks=0` means LSan never emits `==<pid>==ERROR: LeakSanitizer: …` to stderr, so `asan_err` only catches genuine AddressSanitizer ERRORs. **PASS**.

**Verdict on F2**: closure is correct. Implementation matches the round-1 "Option C" recommendation (sentinel grep instead of leak emission as dtor proof), but uses an existing-in-SHUD print rather than a new `atexit` hook, which is simpler and avoids any new SHUD source-side surface.

---

### Phase 6 Sibling closure findings (round-2 newly-introduced — F3, F9)

These were closure-only commits, not new findings. Both checked for correctness as part of round-2 scope.

#### F3 (SHUD inner: sibling-class `~Model_Data` chain NSDMI + `delete` → `delete[]`) — CORRECT

**Files**: `SHUD/src/classes/{TimeSeriesData.hpp, Lake.hpp, Lake.cpp}`. Diff via `git show 1ab61c0`.

Verifications performed:

1. **`_TimeSeriesData::ts[MAXQUE+1]` brace-init NSDMI** (TimeSeriesData.hpp:44 = `double *ts[MAXQUE + 1] = {nullptr};`):
   - C++11 brace-init on a raw pointer array zero-initializes ALL elements, NOT just the first. Per [dcl.init.list] / [dcl.init.aggr], an aggregate-initializer with one element zero-fills the rest. `{nullptr}` here defaults every `ts[i]` to nullptr. **Valid C++11+**. **PASS**.
   - Dtor at `TimeSeriesData.cpp:31-34` loops `for (int i = 0; i < MAXQUE; i++) { delete[] ts[i]; }`. **Loop range is `MAXQUE`, NOT `MAXQUE+1`** — the last slot `ts[MAXQUE]` (which is the wrap-around slot for the pRing buffer) is never freed even after this fix. This is **pre-existing condition** (predates 1ab61c0; see `git show 90a518b:src/classes/TimeSeriesData.cpp`); the PR did not touch the loop range and is not responsible. NOT a regression introduced by F3.
   - `ts[i]` is `double*` allocated via `new double[n]` at `TimeSeriesData.cpp:56` (`ts[i] = new double[n];`). Dtor's `delete[] ts[i]` is the correct array form. **PASS — pre-fix was already correct on form; the NSDMI guards partial-init where some `ts[i]` slots are nullptr while others are valid**.

2. **`_Lake` NSDMI** (Lake.hpp:98-111):
   - 10 raw heap ptrs get `= nullptr`: `iEleLake, iEleBank, iRivIn, iRivOut, RivIn, RivOut, QEleSurf, QEleGW, QRivIn, QRivOut`. All are declared `int *` or `double *`; NSDMI syntax is standard. **PASS**.

3. **`LakeBathymetry` NSDMI** (Lake.hpp:69-71): `index/yi/ai` get `= nullptr`. **PASS**.

4. **`_Lake::~_Lake()` `delete` → `delete[]`** (Lake.cpp:91-105):
   - 5 dtor lines changed: `iEleBank`, `QEleSurf`, `QEleGW`, `QRivIn`, `RivOut`. Cross-referenced against alloc sites:
     - `iEleBank = new int[NumEleBank]` (Lake.cpp:124) — `delete[]` correct (array).
     - `QEleSurf = new double[NumEleBank]` (Lake.cpp:110) — `delete[]` correct.
     - `QEleGW = new double[NumEleBank]` (Lake.cpp:109) — `delete[]` correct.
     - `QRivIn = new double[NumRivIn]` (Lake.cpp:112) — `delete[]` correct.
     - `RivOut = new int[NumRivOut]` (Lake.cpp:138) — `delete[]` correct.
   - All 5 form-corrections match `new[]` allocation site. Pre-fix scalar `delete` was independent UB (separate from uninit-ptr UB) per `[expr.delete]/2`. **PASS**.
   - **Pre-existing leak (not introduced/regressed)**: `iEleLake`, `iRivIn`, `iRivOut`, `RivIn`, `QRivOut` are `new[]`-allocated (in `MD_Lake.cpp:92,94,95` and `Lake.cpp:128,113` respectively) but NEVER freed in `~_Lake()`. The PR-0 commit message explicitly limits F3 scope to form-correction of EXISTING dtor calls + NSDMI; missing frees are out of round-2 scope and were also out of round-1 scope. Same condition was present at the pre-PR baseline `90a518b` (2022). **NOT a regression**.

5. **`LakeBathymetry::~LakeBathymetry()` `delete` → `delete[]`** (Lake.cpp:50-58):
   - 3 dtor lines changed: `index`, `yi`, `ai`. Cross-referenced against alloc at `Lake.cpp:147,148,149` (in `InitValue()`):
     - `index = new int[nvalue]` — `delete[]` correct.
     - `yi = new double[nvalue]` — `delete[]` correct.
     - `ai = new double[nvalue]` — `delete[]` correct.
   - **PASS**.

6. **Normal-init-path behavior change**: NSDMI defaults are overwritten by `new` assignments in `readLake()` / `Initialize()` / `InitValue()` BEFORE any read. There is no read-before-assign on the production path. Bitwise-neutrality confirmed via SHA256 self-compare (`mac_asan_keliya_postfix.log:106`). **PASS**.

#### F9 (drop dead `Model_Data::{ISFactor, windH}` top-level NSDMI) — CORRECT

**File**: `SHUD/src/ModelData/Model_Data.hpp:170-177`.

Verifications performed:

1. **Top-level `Model_Data::ISFactor` is dead**: `grep -rn '\bISFactor\b' SHUD/src/` returns ZERO live uses (only the 1 declaration + 2 doc-comment references in the F9 change itself). **PASS — truly dead**.

2. **Top-level `Model_Data::windH` is dead**: `grep -rn '\bwindH\b\|->windH\|\.windH' SHUD/src/` distinguishes:
   - `Ele[i].windH` — assigned at `MD_initialize.cpp:224` (`Ele[i].windH = HeightWindMeasure;`) and copied to `hot.windH[i]` at `Model_Data.cpp:383`. Live, kept.
   - `hot.windH[i]` — read at `MD_ET.cpp:93` (`WindProfile(2.0, t_wind[i], hot.windH[i], …)`); allocated at `Model_Data.cpp:222` (`hot.windH = new double[NumEle];`); freed at `MD_readin.cpp:662` (`delete[] hot.windH;`); zero-initialized at `Model_Data.cpp:276`. Live, NSDMI on the `MD_layout.hpp` side is kept.
   - `Model_Data::windH` (top-level, removed in F9) — ZERO assignment, dereference, or `delete` references tree-wide. **PASS — truly dead, and not confused with the live `hot.windH` or `Ele[i].windH`**.

3. **`*ISFactor` and `*windH` declarations still exist** (Model_Data.hpp:176-177); F9 only removed the `= nullptr` NSDMI initializer. The fields are now uninitialized again — but since they have ZERO callers, there is no read-before-write risk. The reviewer note in round-1 recommended either (a) delete the fields entirely or (b) drop the NSDMI and follow up. F9 takes path (b). The fields are still declared, just no longer pretending to be safe. **Cosmetic, NOT a correctness regression**.

---

## Phase 6 delta findings (NEW)

- **None.**

The Phase 6 delta cleanly closes F1 + F2 + F3 + F9. No new correctness defects surfaced.

---

## Observations (non-blocking)

### O1 (Minor): F2 sentinel proves "modelSummary(1) ran", which is a strict subset of "dtor chain ran"

**File**: `.github/workflows/serial-baseline.yml:1449-1469` workflow comment + `SHUD/src/ModelData/Model_Data.cpp:22-43` + `SHUD/src/Model/shud.cpp:291,359,507,591`.

**Failure class**: documentation accuracy — claim is slightly stronger than reality, but the gate semantics are still correct.

**Scenario**: workflow comment at line 1454-1456 reads "The marker is generated IFF the C++ dtor chain ran to completion (the converse direction is what we want: positive emission proves full dtor coverage)". The actual call chain is:

1. `main()` → `SHUD()` → `SHUD(fin, fout)` (coupled) or `SHUD_uncouple(fin, fout)`.
2. Coupled path at `shud.cpp:291`: `MD->modelSummary(1)` → `TimeSpent()` → `screeninfo("\n\nThe successful end. \n\n")` — **sentinel printed here**.
3. AFTER sentinel, lines `shud.cpp:292-359` execute: `N_VDestroy(udata/du)`, `fopen/PrintFinalStats/fclose`, `fopen/fprintf nfcall/fclose`, `CVodeFree`, `SUNContext_Free`, `delete MD` (← actual dtor here).
4. Same shape on uncouple path: sentinel at `shud.cpp:507`, then 86 lines later `delete MD;` at `:591`.

**Why it matters**: if the dtor chain itself crashes (e.g., `~Model_Data → FreeData → delete[]` of a corrupt pointer), the sentinel will already have been emitted to stdout. `marker_count > 0` (gate passes the first check). BUT — and this is why the gate is still correct — `halt_on_error=1` + ASan will catch the dtor-time UB and emit `==<pid>==ERROR: AddressSanitizer:` to stderr → `asan_err > 0` → gate fires at the SECOND check (L1473) with the "non-clean" message.

The gate's net behavior is therefore correct (any dtor-time UB still fails the job), but the attribution would be slightly misleading: the error message would say "ASan/UBSan non-clean" rather than "dtor-bypass regression", even though the underlying cause IS a dtor regression. This is purely cosmetic — a developer investigating the failure would still find the ASan stack trace pointing into `~Model_Data` / `FreeData`.

**Why it doesn't matter for round-2 verdict**: the round-1 F2 spec text was "regression protection for re-introduction of `_exit/abort` workaround" — the new sentinel gate does catch exactly that (`_exit/abort` BEFORE `modelSummary(1)` would suppress the sentinel; `_exit/abort` AFTER `modelSummary(1)` but BEFORE `delete MD` would emit the sentinel but the dtor-bypass would not be caught by this gate alone — however, in the spike-binary context where this regression would land, the regression would also be visible via the spike binary's own ASan run, which would NOT see the marker because spike binaries don't print "The successful end."). The gate is good enough.

**Fix**: optional. Either (a) accept current wording as a slight over-claim, or (b) revise comment L1454-1456 to read "The marker is generated IFF normal main() return path reached `modelSummary(1)`; any `_exit/abort/crash` before that point suppresses the marker. Dtor-time UB after the marker is caught by the second-stage ASan/UBSan error check below."

**Blocks merge**: NO. Documentation drift, gate semantics correct.

---

## Removed-behavior audit lens

- **F1 removed behavior**: `_exit(0)` short-circuit at end of `dump_adjacency.cpp` / `fd_color_jacobian.cpp` main was already removed in round 1. Round 2 RESTORED the `delete MD; delete fout; delete fin;` triple on the success path. Net behavior: dtor chain now runs to completion on success path (vs round 1 where MD/fout/fin were leaked raw ptrs, vs pre-round-1 where `_exit(0)` skipped both stdio cleanup and dtors). Mac smoke (keliya, NumEle=484) confirms no SEGV / no `free(): invalid pointer`. **Server-scale verification (heihe / heihe_x4 / heihe_x16) deferred to orchestrator Phase 2 — unchanged from round-1 scope**.

- **F2 removed behavior**: `LEAK SUMMARY:` negative gate replaced with positive `The successful end.` sentinel. Verified no other CI consumer relies on the absence of LEAK SUMMARY gate: `grep -rn 'LEAK SUMMARY' .github/` returns zero hits outside the (now-removed) gate location; no Make target, no script, no other workflow references it. **PASS — clean removal**.

- **F9 removed behavior**: NSDMI `= nullptr` on `Model_Data::ISFactor` and `Model_Data::windH` (top-level dead fields) removed. Round-1 reviewer note (this reviewer) recommended either removing the fields entirely OR dropping the NSDMI with a follow-up note. F9 takes the second option. Fields remain declared but uninitialized. Verified ZERO live callers via `grep -rn '\bISFactor\b\|->ISFactor\|->windH\|\.windH' SHUD/src/` (filtered to exclude `hot.windH` and `Ele[i].windH` which ARE the live fields). **PASS — safe to drop; no read-before-write risk because no reads exist at all**.

---

## Invariant Matrix Coverage (Round 2 update)

| # | Row | Round 1 | Round 2 | Evidence |
|---|-----|---------|---------|----------|
| i | Mac keliya valgrind (or ASan equivalent) zero-error | covered | covered | `mac_asan_keliya_postfix.log:109-121` post-Phase-6 re-run: 0 ASan errors, 0 UBSan errors, sentinel emitted. |
| ii | Server heihe (NumY=19515) valgrind 0 definite/indirect | deferred | deferred | Unchanged; orchestrator Phase 2. |
| iii | Server heihe_x4 (NumEle=40046) dump_adjacency dtor-full smoke | deferred + finding | deferred (finding resolved) | F1 closure means the spike binary success path now DOES exercise `~Model_Data → FreeData()` chain; Mac smoke at keliya scale confirms (log L123-130). Server scale validation remains deferred. |
| iv | Server heihe_x16 (NumY=485K) dump_adjacency dtor-full | deferred + finding | deferred (finding resolved) | Same as row iii. |
| v | B0/B1a/B1b bitwise neutral on keliya | covered | covered (extended) | Pre-fix SHA `b769e327…c028bd` = post-fix SHA (mac_asan_keliya_postfix.log:54-55, :106). F3 sibling-class fix is also bitwise-neutral (same SHA confirmed at L106-107 post-1ab61c0). |
| vi | Mac libomp + Linux libgomp dtor parity | partial | partial (gate now correct) | Mac covered (sentinel emitted under shud_asan); Linux GH runner CI will now correctly gate the same sentinel via the new positive-check (was broken in round 1 per F2; now resolved). Linux end-to-end run requires forcing data deployment which is out-of-scope per spec (data_available=false skip path is documented). |

Net: 3/6 covered, 1/6 partial (Linux deferred for orthogonal data-deployment reason), 2/6 deferred to server. Of the deferred rows, NONE are at risk anymore — F1 closure means the spike binary actually does exercise the dtor when it eventually runs on server.

---

## Verdict

**APPROVE-WITH-MINOR** — both round-1 Criticals (F1 + F2) are CLOSED. Sibling-class invariant closure (F3) and dead-field cleanup (F9) on SHUD inner pointer bump are correctness-sound. Bitwise neutrality on Mac keliya is confirmed. No new correctness regressions.

Single non-blocking Observation O1: the F2 sentinel proves "main loop + `modelSummary(1)` ran", which is a strict subset of "dtor chain ran to completion" — the gate's actual semantics are correct (dtor-time UB caught by second-stage ASan/UBSan error check), but the workflow comment over-claims slightly. Cosmetic; recommend a 2-line comment refresh in a follow-up but does not block merge.

Server-scale validation (heihe / heihe_x4 / heihe_x16) remains deferred to orchestrator Phase 2 per `server_acceptance_cmds.sh`. Round-2 scope is closure of round-1 findings on the Phase 6 delta + new findings on the delta itself; nothing in this round changes the server-deferral scope.

---

## Files reviewed (absolute paths)

- `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/dump_adjacency.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/fd_color_jacobian.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.github/workflows/serial-baseline.yml`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TimeSeriesData.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TimeSeriesData.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/Lake.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/Lake.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_readin.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_Lake.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_layout.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/Model/shud.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/main.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/mac_asan_keliya_postfix.log`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/cross-review-round-1/correctness.md` (this reviewer's round-1)
