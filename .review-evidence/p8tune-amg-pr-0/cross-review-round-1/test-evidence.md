## Review: PR #400 — P8-tune.F PR-0 (Test & Evidence Coverage scope)

Head SHA: `09a815dbcab9eabbddcad9550c706ddfa8636519`
Branch: `feat/issue-394-p8tune-amg-spike`
Scope: Test & Evidence Coverage only (other dimensions out of scope for this reviewer).

### Summary

Source-fix coverage is comprehensive (every `delete[]`/`delete` target in `Model_Data::FreeData()` now has an NSDMI nullptr default) and Mac local memory-safety evidence is concrete and reproducible. However, two evidence claims in the PR body materially overstate what was actually verified: (a) the "CI exercises the new LEAK SUMMARY: grep gate" claim does not hold — the only CI run on this head SHA short-circuited at `data_probe (data_available=false)` and never executed the ASan runtime step that contains the gate; (b) the bitwise-neutral SHA256 self-compare relies on a single hash value that appears as both pre-fix and post-fix in the same evidence file with no independent capture of the pre-fix run. The actual failure-mode workload (NumY > 100k) is not exercised anywhere on Mac and is correctly deferred to server Phase 2, but the spec's REQ-3 invariant matrix rows (ii)-(iv) therefore remain entirely uncovered as of this head SHA.

### Findings

#### 🔴 Critical: PR body's CI verification claim is not supported by the actual CI run

`/Users/danker/Desktop/Hydro-SHUD/openMP/.github/workflows/serial-baseline.yml:1378-1391` and PR body "Verification" section.

The PR body claims:

> `serial-baseline.yml asan-ubsan (keliya)` job exercises the new `LEAK SUMMARY:` grep gate. Job fails on (i) any ASan/UBSan error/warning, or (ii) `LEAK SUMMARY:` absent (= dtor bypass regression).

The actual CI run on head SHA `09a815d` (run id `28355785721`, job `asan-ubsan (keliya)` id `83998346479`) shows:

```
##[notice]CI runner has no keliya forcing/input data; build verification PASSED, sanitizer run SKIPPED.
```

The `Probe ${{ matrix.case }} data availability` step set `data_available=false` because `SHUD/Basins/keliya/forcing/X*.csv` and `SHUD/Basins/keliya/input/keliya/keliya.cfg.para` are not present on the GH runner. Every subsequent step — including `Run keliya under ASan + UBSan` and the new `LEAK SUMMARY` grep gate — is gated on `steps.data_probe.outputs.data_available == 'true'` and therefore did NOT execute. The job "passes" only because none of the runtime assertions ran, not because the gate validated dtor coverage.

Why it matters: the "CI provides Linux libgomp + LeakSanitizer regression coverage on the dtor-full path" claim in the Verification table and in invariant-matrix row (vi) is currently aspirational, not actual. Any future regression that re-introduces `_exit(0)` in `tools/p8tune.D/*.cpp` or that breaks the dtor chain would NOT be caught by CI in its current state — there's no Linux runtime path executed under ASan with this workflow as configured.

Recommended fix (pick one):
- Update the PR body Verification section to say "the gate is wired in workflow YAML but CI keliya runtime is currently skipped pending forcing/input deployment; canonical Linux validation is the orchestrator Phase 2 server runs," AND keep the test plan `[ ]` on the CI item, AND move invariant-matrix row (vi) Linux side from PARTIAL to DEFERRED until either (i) keliya forcing is deployed to the GH runner or (ii) a smaller bundled fixture is added.
- OR add a tiny self-contained fixture under `SHUD/Basins/keliya/` so `data_probe` flips to `true` and the gate actually runs (this is the canonical fix per spec REQ-3).

The test plan's CI checkbox is correctly unchecked (`[ ] CI ...`), so the box state itself is honest; the misleading content is in the Verification table and invariant-matrix row (vi) where the same gap is described as "exercises the new ... gate" / "PARTIAL — Mac covered; Linux deferred to CI + Phase 2," implying CI is doing work it is not.

#### 🟡 Warning: Pre/post bitwise-neutral SHA256 evidence has no independent pre-fix witness

`/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log:46-49` + `/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/keliya_postfix_rivqdown_sha256.txt:1`.

Bitwise-neutral claim:

```
pre-fix SHA  = b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd
post-fix SHA = b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd
```

I verified the post-fix hash matches the on-disk file (`shasum -a 256 SHUD/Basins/keliya/output/keliya.out/keliya.rivqdown.dat` → `b769e3270e...`). I cannot independently verify the pre-fix value: the repo contains no separately-captured pre-fix artifact (e.g. `keliya_prefix_rivqdown.dat`, a separate `keliya_prefix_rivqdown_sha256.txt`, a CI artifact upload, or a commit-message-recorded hash from before the NSDMI was applied). The pre-fix figure appears only inside the post-fix run's commentary, identical to the post-fix value, so the "MATCH" verdict is currently a single-source claim.

This is a weaker form of the row-(v) invariant-matrix gate than the spec implies. The reasoning is sound a priori (NSDMI nullptr defaults are overwritten by the existing `new` assignments in `malloc_EleRiv()` / `malloc_Y()` / `MD_Lake.cpp` / `MD_readin.cpp`, so happy-path writes are unchanged), but per the test-evidence rubric the gate exists to catch reasoning errors. The B0-archived `benchmarks/keliya/B0_output/keliya.rivqdown.dat` SHA is `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc`, which is a different value entirely — that's expected (B0 archive is a different run length than the 90-day deployed window) but it means the B0 archive can't be used to cross-check this claim either.

Recommended fix: record the pre-fix hash in a separate file with a clearly distinct provenance comment, OR add a commit message body / PR comment with the timestamp + hash of the pre-fix run + the exact command used, so the pre-fix value has a witness independent of the post-fix evidence file.

#### 🟡 Warning: Real failure-mode workload (NumY > 100k) is not exercised on Mac and the workaround Mac evidence cannot prove the failure is fixed

`/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log:60-64`.

The PR body and issue #386 both state the bug only triggers at NumY > 100k (heihe_x4 ≈ 124K, heihe_x16 ≈ 485K). The Mac ASan post-fix run is on keliya (NumY=1500 — about 80× smaller than the failure threshold). The post-fix evidence file is therefore a happy-path baseline test (proves the fix didn't break the small case) rather than a failure-mode acceptance test (proves the fix fixes the bug). The only failure-mode tests in the entire PR are server-side `dump_adjacency heihe_x4 / heihe_x16`, which are DEFERRED to orchestrator Phase 2 and have no result artifacts yet.

This is structurally correct (Mac can't run valgrind and has no heihe_x4-scale data), but it means rows (ii)-(iv) of the invariant matrix have zero coverage as of this head SHA. The PR text marks them as `⏳ DEFERRED` honestly, but no SHUD-side synthetic OOM / partial-init unit test was added either. Consequently the only assurance the failure mode is actually fixed is reasoning-by-source-inspection (NSDMI defaults make `delete[] nullptr` defined a no-op, per C++03 §5.3.5/2), not a positive run.

Recommended fix: either (a) add a SHUD-side gtest/CMake unit test under `SHUD/tests/` that constructs a `Model_Data` with the default ctor (no malloc_EleRiv call), invokes `~Model_Data()`, and asserts ASan-clean exit — this would be runnable on Mac without large data; OR (b) make explicit in the PR body that "merge depends on orchestrator Phase 2 server results posting BEFORE merge, NOT after" so reviewers don't approve based on Mac-only positive evidence for what is fundamentally a large-NumY bug.

#### 🔵 Suggestion: Filename `valgrind_keliya_postfix.log` misleads at a glance

`/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log` (filename).

The file is named `valgrind_*.log` but contents are an `ASan + UBSan` substitute run (Mac has no valgrind for Darwin AArch64). The body of the file explains this clearly in the "Platform constraint note" section, and per CLAUDE.md "all alternative tooling documented" the substitution is acceptable. But a reviewer scanning a file index would reasonably expect valgrind output. The filename leaks expectation into the wrong tool.

Recommended fix: rename to `mac_asan_keliya_postfix.log` (or `keliya_postfix_sanitizer.log`) and update references in the PR body and `server_acceptance_cmds.sh` accordingly. Low priority — does not affect merge readiness — but improves index hygiene for future audits.

#### 🔵 Suggestion: `server_acceptance_cmds.sh` is a runbook, not an executable test result

`/Users/danker/Desktop/Hydro-SHUD/openMP/.review-evidence/p8tune-amg-pr-0/server_acceptance_cmds.sh:1-111`.

The file is well-structured (Slurm三铁律 compliant: sbatch from `/scratch`, `--output/--error` in `/scratch`, scripts in `/scratch`) and clearly labels itself as a runbook (line 7: "DO NOT run from inside this implementer subagent"). However, this is the entirety of the gating evidence for invariant-matrix rows (ii), (iii), (iv) and for half of row (vi). For a reader who only looks at the evidence directory, it may not be obvious that no actual server result has been collected — there's no companion `server_acceptance_results.md`, no expected-output sentinel sample, no placeholder file with "RESULTS PENDING" sentinel.

Recommended fix: when reviewing this PR for merge, the orchestrator/verifier should require either (a) the actual server result `*.out` / `*.err` rsync'd back into `.review-evidence/p8tune-amg-pr-0/` with `BUILD_OK`, `VG_DONE` with `ERROR SUMMARY: 0`, and `BOTH_OK` sentinels visible, OR (b) explicit text in the merge commit / PR body saying "merging WITH known DEFERRED server evidence; downstream PR-A/B/C/D+E will catch any heihe_x4/x16 regression at their own server validation step." Current state is ambiguous.

#### 🟢 Praise: NSDMI coverage is exhaustive and matches the `FreeData()` delete set 1-to-1

`/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.hpp:99-...` + `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_layout.hpp:57-95` + `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_readin.cpp:533-688`.

I cross-checked every `delete[]` / `delete` in `Model_Data::FreeData()` (MD_readin.cpp:528-688) against the pointer declarations in `Model_Data.hpp` and `MD_layout.hpp`. Every pointer that gets `delete[]`'d (io_ele / io_riv / io_lake — already done in S5d.2-5a; QeleSurf_flat ... t_mf ... RivSeg / Riv / Riv_Type / rivNode; hot.nabr_flat ... hot.ImpAF; Ele / Node / Soil / Geol / LandC; tsd_weather) has a matching `= nullptr` NSDMI. The only pointer-typed Model_Data members WITHOUT NSDMI are `pf_in` / `pf_out` (line 47-48), and those are correctly excluded — they are never `delete[]`'d in `FreeData()` and are unconditionally set by the `Model_Data(FileIn*, FileOut*)` ctor. The single non-array `delete flood` at MD_readin.cpp:688 also has its corresponding `FloodAlert *flood = nullptr` at Model_Data.hpp:158. No false positives, no gaps.

#### 🟢 Praise: `_exit(0)` removal does not collide with the only other `_exit` test in SHUD

`/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/tests/s1d_strictomp_assert_smoke.cpp:75`.

`SHUD/tests/s1d_strictomp_assert_smoke.cpp` uses `_exit(0)` in a deliberate child-process pattern (`fork()` → child runs `MD->rhs_core(...)` under `ExecPolicy::StrictOMP`, parent uses `waitpid` to assert `WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT`). The `_exit(0)` there is the "if abort didn't fire, deliberately exit clean so parent sees not-SIGABRT" branch, completely unrelated to the `tools/p8tune.D/*.cpp` `_exit(ok?0:1)` workaround. The "Removed-behavior audit lens" search confirms no test depends on the removed `_exit` semantics in `tools/p8tune.D/dump_adjacency.cpp` or `tools/p8tune.D/fd_color_jacobian.cpp`.

### Verdict

REQUEST CHANGES — the source-fix is correct and well-scoped, but two evidence-claim issues must be resolved before merge: (1) the CI verification claim does not match the actual CI run on this head SHA (gate is wired in YAML but never exercised); (2) the bitwise-neutral pre-fix SHA has no independent witness. Both can be resolved by edits to the PR body and `.review-evidence/p8tune-amg-pr-0/` without changing source code. Server-side acceptance evidence (rows ii-iv) remains correctly DEFERRED and must land before final merge per the orchestrator Phase 2 contract.
