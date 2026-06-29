# Correctness Review — PR #400 (P8-tune.F PR-0)

- **Head SHA**: `09a815dbcab9eabbddcad9550c706ddfa8636519`
- **SHUD submodule**: `710c00a → 056a1dc` (openmp-baseline)
- **Scope**: NSDMI nullptr defaults for Model_Data + ElementHotData heap-ptr members; `_exit(0)` → `return` in p8tune.D spike tools; CI `detect_leaks=1` + LEAK SUMMARY grep gate.
- **Round**: 1
- **Reviewer**: Correctness brief (leaf reviewer)

---

## Summary

The NSDMI patch on `Model_Data.hpp` / `MD_layout.hpp` is correctly written, syntactically valid, ABI-neutral (sizeof/layout unchanged — NSDMI affects default value, not member declaration order), and consistent with the prior `io_ele/io_riv/io_lake` pattern from S5d.2-5a. C++ `delete[] nullptr` is a defined no-op (`[expr.delete]/2`), so the unconditional `FreeData()` chain is safe under partial init / mid-malloc-loop OOM as claimed. Bitwise SHA256 self-compare on keliya `rivqdown.dat` is byte-identical (evidence file: `keliya_postfix_rivqdown_sha256.txt`).

**However**, two correctness defects in the *gate/workaround-removal* half of the PR undermine the stated invariant ("dtor full coverage" + "regression protection"):

1. **CI LEAK SUMMARY gate is logically unreachable on the success path.** When SHUD has any leak (which it does — pre-existing un-freed `AccT_surf`/`AccT_sub`/`lake`/`t_sph`/`fu_Surf`/etc.), the line `==<pid>==ERROR: LeakSanitizer: detected memory leaks` triggers the `asan_err` regex (`==.*==ERROR`) at workflow line 1458 and the job hard-fails with exit 1 BEFORE the `leak_summary_count` gate at line 1463 is even evaluated. The gate cannot green-light any combination of (leaks, no-leaks) × (dtor-full, dtor-skipped). It is currently SUCCESS only because forcing data isn't deployed to the GH runner so the `Run … under ASan + UBSan` step is conditionally skipped.

2. **Spike tools' `return 0` does NOT invoke the SHUD destructor chain.** Both `dump_adjacency.cpp` and `fd_color_jacobian.cpp` heap-allocate `MD = new Model_Data(...)`, `fin = new FileIn`, `fout = new FileOut` and never `delete` them on the success path. `return 0` from `main` destroys only automatic-storage objects; heap objects held by raw pointers are leaked, so `~Model_Data() → FreeData()` is never called on the happy path. The PR description claim "destructor chain runs to completion (delivers leak-summary signal to ASan + valgrind)" is provably false for these binaries. The NSDMI fix's *runtime* validation must come from `./shud` (which does have `delete MD` at `shud.cpp:359/591`), not from the spike binaries.

The bitwise-neutrality + spec REQ-3-pattern-(i) "root cause fix" portion of the PR is sound. The validation-loop portion is broken and should be fixed before merge.

---

## Findings

### 🔴 Critical: CI sanitizer gate's `asan_err` regex masks the LEAK SUMMARY presence check
`.github/workflows/serial-baseline.yml:1441,1449,1458,1463`

**Failure class**: control-flow / unreachable-positive-branch

**Contract**: per spec amg-pattern-spike-verdict REQ-3 + workflow line 1454-1457 comment "Hard-fail on any ASan/UBSan error/warning OR on absent LEAK SUMMARY emission (= dtor bypass regression)". Intent: distinct gates so a leak-bearing dtor-complete run passes (LEAK SUMMARY visible, no errors), and a dtor-bypass run fails (no LEAK SUMMARY).

**Scenario**: when LeakSanitizer detects any leak (which it will, given SHUD's many pre-existing un-freed pointers — see siblings below), it emits this canonical line to stderr:

```
==12345==ERROR: LeakSanitizer: detected memory leaks
```

**Evidence**: This line matches both regexes:
- `asan_err` regex `'AddressSanitizer.*ERROR|==.*==ERROR'` → MATCH via `==.*==ERROR` alternation. Verified locally: `grep -cE` returns `1`.
- `leak_summary_count` regex `'^==[0-9]+==(ERROR: LeakSanitizer:|SUMMARY: AddressSanitizer: detected memory leaks)|LEAK SUMMARY:'` → MATCH via first alternation. Verified: returns `1`.

Workflow control flow:
```
1458:  if [[ "$asan_err" != "0" || ... ]]; then
1459:    echo "::error title=ASan/UBSan ... non-clean::ASan=$asan_err ..."
1461:    exit 1
1462:  fi
1463:  if [[ "$leak_summary_count" == "0" ]]; then  # UNREACHABLE on the leak+dtor-full path
```

Result: when SHUD has any leak, `asan_err >= 1`, the first `if` fires, exit 1. The intended "leaks visible = dtor ran = job PASS" branch is unreachable. The CI is currently green only because `data_available=false` causes the run step itself to skip; the gate has never been exercised end-to-end. Verified via `gh run view 28355785721 --log`: keliya/qhh both report `::notice ... sanitizer run SKIPPED`.

**Why it matters**: The PR commits to "regression protection for any future re-introduction of `_exit/abort` workaround" (workflow comment line 1457). Once forcing data is deployed to the runner (or once someone runs this gate against heihe on the server), this gate will hard-fail every PR — not because of any regression, but because real leaks exist. Then someone will silence it incorrectly. Net effect: the protection is illusory, and worse, it'll get neutered the first time it fires.

**Siblings**: all 4 cases in the `asan-ubsan` matrix (keliya, qhh) face the identical fault — single shared regex.

**Blocks merge**: YES. This is the central "regression protection gate" the PR claims; if it's broken the PR's safety story collapses.

**Fix**: Two equivalent options:
- (A) Tighten `asan_err` regex to exclude the LSan ERROR line, e.g. `'AddressSanitizer.*ERROR|==[0-9]+==ERROR: (AddressSanitizer|StackSanitizer)'` (NOT match LeakSanitizer ERROR). Then leak-bearing runs trigger only the leak_summary gate (which would now correctly mean "dtor ran"); zero-leak runs would still need a different proof-of-dtor (see option C).
- (B) Re-order: check `leak_summary_count` first; if present, treat the LSan ERROR line as expected (don't count it in asan_err); else fail with "dtor bypass" message.
- (C) Better: stop relying on leak emission as dtor proof. Instead, add a sentinel `std::atexit([]{ std::fprintf(stderr, "[dtor-full] reached SHUD shutdown\n"); });` (or equivalent global static dtor) and grep for that token. This is robust to whatever LSan reports.

---

### 🔴 Critical: `return 0` in spike tools does NOT invoke `~Model_Data()` — dtor coverage claim is false
`tools/p8tune.D/fd_color_jacobian.cpp:282-286,308,318,432,473` and `tools/p8tune.D/dump_adjacency.cpp:398-402,442`

**Failure class**: removed-behavior audit — the `_exit(0)` removal was intended to enable `~Model_Data() → FreeData()`, but the new code never invokes it on the success path.

**Contract**: PR description and design.md D5 + new code comment claim the change "Restore[s] normal `return` semantics so destructor chain runs to completion (delivers leak-summary signal to ASan + valgrind)". Spec amg-pattern-spike-verdict REQ-3 Scenario "_exit(0) workaround removal" is the requirement being implemented.

**Scenario**: `fd_color_jacobian.cpp` main:
```cpp
282:    FileIn *fin = new FileIn;
283:    FileOut *fout = new FileOut;
286:    Model_Data *MD = new Model_Data(fin, fout);
...
308:        delete MD; delete fout; delete fin;   // error path
318:        delete MD; delete fout; delete fin;   // error path
432:        delete MD; delete fout; delete fin;   // error path
...
473:    return 0;   // SUCCESS path — no delete of MD/fin/fout
```

`dump_adjacency.cpp` is identical: heap-allocates `MD/fin/fout` (lines 398-402), never deletes on success (line 442 just returns).

**Evidence**: C++ standard `[basic.stc.dynamic]` + `[basic.start.term]` — `return` from `main` destroys automatic-storage and global-static objects but NOT objects allocated with `new` and held only by raw pointers. `~Model_Data()` (which calls `FreeData()`, which exercises the very `delete[]` chain whose UB was the subject of #386) is therefore never invoked on a successful spike-tool run.

**Why it matters**:
1. The PR's regression-protection mechanism for these spike binaries — "future re-introduction of `_exit` would silently skip dtor; LEAK SUMMARY presence catches that" — does not work because dtor never runs even WITH the fix in place.
2. The spike tools cannot validate the #386 fix end-to-end. The new code path is byte-for-byte equivalent (in terms of dtor execution) to the old `_exit(0)` path; the only difference is whether C runtime atexit handlers run.
3. Asymmetry with error paths (which DO call `delete MD`) means error-path runs and success-path runs exercise different cleanup; a future bug in `~Model_Data` would manifest only in the error path.

**Siblings**:
- `dump_adjacency.cpp:398-402,442` — identical pattern.
- `tools/p8tune.D/` has no other spike binaries needing the same audit per `tools/p8tune.D/Makefile`; covered.

**Blocks merge**: YES. The spike-tool half of the workaround-removal is a no-op at runtime (no dtor invocation either way). Either delete the change as cosmetic, OR add `delete MD; delete fout; delete fin;` (or convert to stack-allocated locals / `std::unique_ptr`) so the dtor actually runs and the stated regression-protection actually exists.

**Fix**:
```cpp
// Option 1: explicit delete (matches existing error paths)
std::fflush(stdout); std::fflush(stderr);
delete MD; delete fout; delete fin;
return 0;
```
or
```cpp
// Option 2: stack-allocated; dtors run via normal scope exit
FileIn fin;
FileOut fout;
Model_Data MD(&fin, &fout);
MD.loadinput();
...
return 0;   // ~Model_Data runs here
```

(Option 2 is preferred — Linus would write it as raw stack values; no need for new/delete dance.)

---

### 🟡 Warning: stale docstring claims `delete tsd_weather` (singular), but actual code is `delete[] tsd_weather`
`SHUD/src/ModelData/Model_Data.hpp:104-110` (NSDMI patch's new docstring) vs `SHUD/src/ModelData/MD_readin.cpp:685`

**Failure class**: documentation drift — could mislead the next maintainer.

**Contract**: docstring describes the dtor behavior the NSDMI defaults protect against. Accuracy matters because a future PR could read this and assume `delete` (singular) is in use → write `tsd_weather = new _TimeSeriesData;` (singular) → leak/UB.

**Scenario**: The new docstring at `Model_Data.hpp:104-110` reads: "FreeData()'s unconditional `delete[]` / `delete` chain ..." (plural mentioning both array and scalar form). The actual `FreeData()` at `MD_readin.cpp:685` uses `delete[] tsd_weather;` (array form), matching the allocation at `MD_readin.cpp:377` `tsd_weather = new _TimeSeriesData[NumForc];` (array form). The `delete` (singular) phrasing only matches `delete flood` at line 688 (single FloodAlert). The docstring is technically OK as a general description but conflates two different fields.

**Evidence**: `grep -n 'delete' src/ModelData/MD_readin.cpp` shows 1 singular `delete flood;` and ~80 `delete[]` calls.

**Why it matters**: low — just docstring clarity. No active bug.

**Siblings**: same docstring in `MD_layout.hpp:60-71` is correct (mentions `delete[]` only).

**Blocks merge**: NO.

**Fix**: clarify the Model_Data.hpp docstring to either say "`delete[]` chain" (matches all but flood) or explicitly enumerate the singular case: "(array `delete[]` for most fields; scalar `delete` for `flood`)".

---

### 🟡 Warning: NSDMI patch covers fields that are never assigned AND never freed — dead-field clutter
`SHUD/src/ModelData/Model_Data.hpp:170-171` (`ISFactor`, `windH`)

**Failure class**: scope creep — patch touches fields that have no `delete[]` in FreeData and no `new` assignment anywhere in SHUD source.

**Scenario**: Cross-referenced PR's NSDMI list vs FreeData()'s `delete[]` list. The fields below get `= nullptr` NSDMI but are not deleted in FreeData() (so the "defended no-op" is moot) AND are never assigned (so they remain `nullptr` forever — pure dead code):
- `Model_Data::ISFactor` (Model_Data.hpp:170) — declared, never written, never deleted.
- `Model_Data::windH` (Model_Data.hpp:171) — declared, never written, never deleted. NB: this is the *Model_Data* member, NOT `hot.windH`, which IS used and IS deleted.

These also exist BUT are leaked (alloc'd in malloc_EleRiv/MD_Lake but missing from FreeData) — same patch hit them; no harm but no help either:
- `AccT_surf`, `AccT_sub`, `lake`, `y2LakeArea`, `t_sph`, `fu_Surf`/`fu_Sub` — wait, `fu_Surf`/`fu_Sub` ARE in FreeData (lines 597-598). Recheck: `AccT_surf/AccT_sub/lake/y2LakeArea/t_sph` are NOT in FreeData. Confirmed leaks.

**Why it matters**: docstring claims "Production happy-path allocates these in malloc_EleRiv()/malloc_Y()/MD_Lake.cpp/MD_readin.cpp" — but the dead fields (`ISFactor`, `windH` Model_Data member) are NOT allocated anywhere. Misleading; either delete the fields entirely or remove from the patch with a one-line comment "// declared but unused — TODO drop in S6c".

**Siblings**: dead fields in `MD_layout.hpp` would need separate review, but spot-check shows all of those have matching `hot.X = new ...` in `Model_Data.cpp:200-230` so they are real.

**Blocks merge**: NO. Cosmetic. But worth a follow-up issue.

**Fix**: file a P8-tune-followup issue to delete `Model_Data::ISFactor` and `Model_Data::windH` (NOT `hot.windH`) — they are dead. Separately, fix the FreeData() leaks for `AccT_surf/AccT_sub/lake/y2LakeArea/t_sph` — these are pre-existing, but they now matter because the ASan gate (once it actually runs) will fail on them. Tracked as a separate issue; not this PR's scope.

---

### 🟢 Praise: NSDMI patch is exemplary — symmetric, ABI-neutral, bitwise-confirmed
`SHUD/src/ModelData/Model_Data.hpp:117-220`, `SHUD/src/ModelData/MD_layout.hpp:62-115`

The fix follows the canonical C++ idiom for this class of bug (uninit class-member ptr) — single-line NSDMI at declaration site, no constructor changes, no semantics change beyond default value. It mirrors the prior `io_ele/io_riv/io_lake` and `FloodAlert::itype/fid` precedents in the same SHUD source tree, so reviewers can confirm the pattern at a glance. The bitwise SHA256 self-compare on `keliya.rivqdown.dat` (`b769e327...c028bd` pre-fix == post-fix) is the right experimental verification.

The detailed docstring at `Model_Data.hpp:103-121` correctly identifies the failure modes (OOM mid-malloc_EleRiv, exception mid-init, partial init in spike tooling) and explicitly states the bitwise contract. This is how every fix-for-a-symptom-without-changing-output PR should be documented.

---

## Invariant Matrix Coverage

Per PR-0 Invariant Matrix in design.md / proposal.md, six rows mandatory:

| # | Row | Status | Evidence |
|---|-----|--------|----------|
| i | Mac keliya valgrind (or ASan equivalent) zero-error | covered | `valgrind_keliya_postfix.log` lines 26-33: EXIT=0, 0 ASan errors. NSDMI mechanism — `delete[] nullptr` is defined no-op per `[expr.delete]/2`. |
| ii | Server heihe (NumY=19515) valgrind 0 definite/indirect | out-of-scope-for-this-review | DEFERRED per `valgrind_keliya_postfix.log:69` to orchestrator Phase 2; `server_acceptance_cmds.sh` is the runbook. Cannot verify until server jobs land. |
| iii | Server heihe_x4 (NumEle=40046) dump_adjacency dtor-full smoke | **MISSING evidence + finding** | DEFERRED per evidence file line 70. **Critical**: even when run, the spike tool's `return 0` does NOT invoke ~Model_Data (see Critical-2), so "dtor-full" is a misnomer for this binary. |
| iv | Server heihe_x16 (NumY=485K) dump_adjacency dtor-full | same as iii | DEFERRED + same finding. |
| v | B0/B1a/B1b bitwise neutral on keliya | covered | `keliya_postfix_rivqdown_sha256.txt` matches the pre-fix value cited in `valgrind_keliya_postfix.log:48`. SHA256 byte-identical. |
| vi | Mac libomp + Linux libgomp dtor parity | partial | Mac covered via ASan-substitute-for-valgrind (`valgrind_keliya_postfix.log:73`). Linux GH-runner CI is SKIPPED (data_available=false) — not covered end-to-end, AND once it runs the LEAK SUMMARY gate is broken (see Critical-1). |

Net: 2/6 covered, 1/6 partial, 3/6 deferred to server validation; of the deferred-to-server rows, 2 (iii, iv) are also at risk because the spike-tool dtor never runs.

---

## Verdict

**REQUEST CHANGES** — The Model_Data NSDMI fix itself is sound and bitwise-neutral as claimed. But:

1. The CI LEAK SUMMARY gate is logically unreachable on the leak+dtor-full success path (Critical-1) — `asan_err` regex catches LSan's `ERROR` line and fires before the leak_summary gate. Once forcing data lands, this will hard-fail; until then, CI is green only because the run step is skipped. The "regression protection" claim is false.
2. The spike tools' `_exit(0) → return 0` cosmetic does not invoke `~Model_Data()` on the success path because MD/fin/fout are leaked raw pointers (Critical-2). The "dtor chain runs to completion" claim is false; the change is effectively no-op for dtor coverage purposes.

Fix at least the two Criticals before merge. The two Warnings (docstring drift; dead fields) are non-blocking but worth a follow-up issue.

The core SHUD source-level fix (NSDMI on 80+ pointers) is the right fix for #386 and should be retained as-is.

---

## Files reviewed (absolute paths)

- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_layout.hpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_readin.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/Model/shud.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/dump_adjacency.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/fd_color_jacobian.cpp`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.github/workflows/serial-baseline.yml`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/shud_source_fix.diff`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/keliya_postfix_rivqdown_sha256.txt`
- `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/server_acceptance_cmds.sh`
