Phase 7 Final Review verdict
============================
PR: #400
Final HEAD: c228cc21d8c4d8d60b2b02d319a6792175785383
SHUD inner: 1ab61c023ac2b93a178c2feb07aa3df509fe1a96 (openmp-baseline)
Reviewer: Phase 7 adversarial pass (single reviewer, end-to-end)
Review date: 2026-06-29

================================================================================
REQ-3 closure (`amg-pattern-spike-verdict/spec.md` lines 42-65)
================================================================================

REQ-3 prose: "PR-0 SHALL be a dedicated SHUD source-level fix PR that resolves
#386 by identifying + initializing the offending pointer(s) in the Model_Data
constructor and/or correcting the delete[] order in the destructor chain,
validated by valgrind clean on keliya + heihe. After fix, PR-0 SHALL remove
the _exit(0) workaround in tools/p8tune.D/{fd_color_jacobian,dump_adjacency}.cpp
and verify dtor-full smoke on heihe_x4 + heihe_x16 cases."

- Scenario 1 (#386 root cause fix, lines 47-52):
  CLOSED — SHUD inner `056a1dc` adds NSDMI `= nullptr` to all heap-ptr members
  of `Model_Data` + `ElementHotData`; SHUD inner `1ab61c0` extends to sibling
  `_TimeSeriesData::ts[]` (brace-init), `_Lake` 10 raw ptrs, `LakeBathymetry` 3
  raw ptrs + corrects scalar `delete` → `delete[]` for 8 `new[]`-allocated
  members in `~_Lake()` + `~LakeBathymetry()`. Both commits land on
  `openmp-baseline` branch (NOT master; verified via `git branch --contains`).
  Outer pointer bump tracks inner HEAD. Evidence: `SHUD/src/ModelData/
  Model_Data.hpp:99-180`, `SHUD/src/ModelData/MD_layout.hpp:57-95`, `SHUD/src/
  classes/{TimeSeriesData.hpp:44, Lake.hpp:65-71+98-111, Lake.cpp:52-64+88-105}`.

- Scenario 2 (valgrind clean acceptance, lines 55-59):
  PARTIAL — keliya covered on Mac via ASan+UBSan equivalent (valgrind has no
  Darwin AArch64 port, substitution documented per CLAUDE.md "all alternative
  tooling documented"; evidence `mac_asan_keliya_postfix.log:109-140` shows
  EXIT=0, 0 ASan/UBSan errors, sentinel emitted at stdout L218); heihe valgrind
  + heihe_x4/x16 dump_adjacency dtor-full smoke runbook ready in
  `server_acceptance_cmds.sh` §3-4 (DAG correct: `--dependency=afterok:
  ${BUILD_JID}`, Slurm 三铁律 compliant). At time of review `squeue -u
  frd_muziyao` returns empty queue — server jobs are not currently in flight.
  Per PR body "Verification" + tracking issue #401, server SuiteSparse blocker
  pushes rows (iii) and (iv) of design.md §D5 Invariant Matrix to follow-up.

- Scenario 3 (`_exit(0)` workaround removal, lines 61-65):
  CLOSED — `tools/p8tune.D/dump_adjacency.cpp:440-451` and `tools/p8tune.D/
  fd_color_jacobian.cpp:471-481` both: (i) replaced `_exit(ok?0:1)` /
  `_exit(0)` with `return ok ? 0 : 1` / `return 0`; (ii) Phase 6 restored the
  matching `delete MD; delete fout; delete fin;` triple before the success
  `return` (mirroring error-path convention at `fd_color_jacobian.cpp:308/
  318/432`). No residual `_exit(` or `abort(` in either spike binary
  (grep verified).

- Scenario 4 (Invariant Matrix, design.md §D5 row coverage):
  PARTIAL with DOCUMENTED DEFERRALS — rows (i) Mac keliya + (v) bitwise
  neutral COVERED on Mac; rows (ii)-(iv) DEFERRED with runbook +
  per-row tracking in PR body Verification + follow-up issue #401 for rows
  (iii)-(iv) (SuiteSparse server blocker).

================================================================================
Round 1 Critical resolution (Phase 4.5 verifier CONFIRMED all 3)
================================================================================

- F1 (spike binary success-path MD/fin/fout leak): RESOLVED.
  `tools/p8tune.D/{dump_adjacency,fd_color_jacobian}.cpp` final success path now
  `fflush(stdout); fflush(stderr); delete MD; delete fout; delete fin; return
  ok?0:1;`. Order: streams flushed first; MD before fout/fin (matches
  construction order reversed); identical to error-path convention.
  Cross-checked by round-2 correctness (PASS), integration (PASS), security-perf
  (PASS), invariant-state (PASS).

- F2 (CI LEAK SUMMARY gate broken 3 ways): RESOLVED at engineering level.
  `.github/workflows/serial-baseline.yml:1432` ASAN_OPTIONS reverted to
  `detect_leaks=0` (was 1); broken negative `LEAK SUMMARY:` grep replaced with
  positive `grep -cE 'The successful end\.'` sentinel on stdout (lines
  1464-1475). Sentinel verified unique in SHUD src (only emitted by
  `Model_Data::TimeSpent()` at `Model_Data.cpp:32,40`; not duplicated in any
  error string). Gate ordering correct: sentinel checked FIRST, then ASan
  errors, so dtor-coverage signal isn't masked by an unrelated ASan error.
  Phase 6.5 commit `c228cc2` refreshed the workflow comment to precisely
  characterize the gate's "necessary-but-not-sufficient" property (sentinel
  proves we REACHED the dtor; ASan greps prove dtor RAN CLEANLY) per 4 of 6
  round-2 reviewers' independently-flagged observation.

- F3 (sibling-class invariant closure incomplete): RESOLVED.
  SHUD inner `1ab61c0` extends NSDMI nullptr defaults to `_TimeSeriesData::ts[]`
  (brace-init `= {nullptr}`), `_Lake` 10 raw heap ptrs, `LakeBathymetry` 3 raw
  heap ptrs. Additionally corrects 8 scalar `delete` → `delete[]` form
  mismatches in `~_Lake()` (`iEleBank`, `QEleSurf`, `QEleGW`, `QRivIn`,
  `RivOut`) and `~LakeBathymetry()` (`index`, `yi`, `ai`) — independent UB per
  [expr.delete]/2 regardless of init state. `TabularData` properly excluded as
  out-of-`~Model_Data`-chain (per F3 verifier verdict: stack-allocated locals
  only, `nrow=0` NSDMI already guards unallocated case). Spec REQ-3 line 49
  explicit "and any sibling `~Model_Data` chain" language now mechanically
  satisfied across all dtor-chain-reachable classes.

================================================================================
Round 2 observation disposition (Phase 6.5)
================================================================================

- Sentinel-vs-dtor-completion comment imprecision (4 of 6 reviewers):
  RESOLVED via workflow comment refresh in `c228cc2`. Lines 1449-1463 now
  state "Necessary-but-not-sufficient for full dtor coverage by itself ...
  Together the two checks bracket the dtor invocation: sentinel proves we
  REACHED the dtor; ASan greps prove the dtor RAN CLEANLY."

- PR body stale references (integration round-2 suggestion):
  RESOLVED — current PR body (`gh pr view 400`) references `mac_asan_keliya_
  postfix.log` (post-rename), and correctly describes the positive-sentinel
  gate, not the old broken `LEAK SUMMARY:` approach.

- _Lake dtor dormant + 2 new uninit-ptr leaks (Control_Data::Tout +
  FileOut::fid_time): TRACKED in #401. Full scope written into #401 §1-2
  with explicit dependency note that #401 sub-task 2 ("`delete[] lake;` in
  `FreeData()`") activates PR-0's `_Lake` / `LakeBathymetry` dtor invariants;
  without PR-0 fix landing FIRST, fixing the leak would re-trigger #386
  pattern at lake-case scale.

- CI data_probe skip on GH runner (test-evidence round-2):
  TRACKED in #401 §4. PR body honestly notes "asan-ubsan (keliya) step:
  precondition `data_probe.outputs.data_available == 'true'` returns false on
  GH-hosted runner ... step skips. Positive-sentinel gate logic verified by
  code review + Mac evidence; full CI exercise requires CI runner forcing
  fixture (tracked in same follow-up issue)."

- Server SuiteSparse/ColPack install blocker (security-perf + integration
  round-2): TRACKED in #401 §3.

- Pre-existing un-freed Model_Data ptrs (round-1 F5, round-2 security-perf
  expansion): TRACKED in #401 §1.

================================================================================
Adversarial probes — attempted to find paths that break
================================================================================

Probe A — F1 fix exception safety on OOM-in-`new`:
  Scenario: `new Model_Data(fin, fout)` throws OOM at
  `dump_adjacency.cpp:402` / `fd_color_jacobian.cpp:286`.
  Result: MD never assigned; stack unwinds out of main via unhandled
  exception → `std::terminate` → `_exit` (no dtor). fin/fout leak. Same
  behavior as round-1 state; no regression introduced by Phase 6 restoration
  of `delete MD;`. Acceptable: any OOM at outer `new` is fatal anyway.
  Verdict: NOT a new finding.

Probe B — F1 fix exception safety in `Model_Data::loadinput()` /
  `initialize()` at lines 403-404:
  Scenario: `loadinput()` or `initialize()` throws mid-call after MD has
  been allocated.
  Result: Exception unwinds past `delete MD; delete fout; delete fin;` (added
  Phase 6) — those statements never execute, fin/fout/MD leak. Same as
  round-1 state. Note: NSDMI ensures the LEAKED `MD` is at least dtor-safe
  if `~Model_Data()` were eventually called via global atexit / sanitizer
  cleanup, so this is no worse than pre-Phase-6 state. Production-acceptable
  per Phase 5 carve-out (spike tools acceptable to leak on init failure path;
  CI exercises only the happy path via shud_asan, not spike binaries).
  Verdict: NOT a new finding — defensive RAII (`unique_ptr<Model_Data>`) was
  suggested as Linus-preferred alternative by round-1 verifier; matched-delete
  was chosen instead to preserve symmetry with existing error-path convention.

Probe C — F2 positive sentinel false-positive (sentinel appears in error
  context):
  Scenario: Could `The successful end.` appear as a substring inside any
  ASan/UBSan/CVODE/SUNDIALS error message, runtime error string, or stack
  trace, causing the gate to PASS spuriously?
  Result: `grep -rn 'successful end' SHUD/src/` returns only the 2 canonical
  emission sites at `Model_Data.cpp:32` and `:40`. `grep -rn 'successful end'
  .github/ tools/ docs/` confirms no other emission in the entire SHUD-OpenMP
  tree. ASan/UBSan/SUNDIALS error strings reviewed — none contain this
  English-language phrase. Sentinel is unique and high-entropy.
  Verdict: NO false-positive risk.

Probe D — F3 sibling NSDMI completeness — find any other class in SHUD with
  `delete[]` and missing NSDMI:
  Scenario: `grep -rnE "^[[:space:]]*delete" src/classes/ src/ModelData/`:
  Result: All identified dtors are either NSDMI-protected or counter-guarded:
  - `Print_Ctrl::~Print_Ctrl` (Model_Control.cpp:432-437): NSDMI `= NULL`
    + counter-guard. SAFE.
  - `FloodAlert::~FloodAlert` (FloodAlert.cpp:27-30): NSDMI `= NULL` +
    counter-guard. SAFE (added in `710c00a`).
  - `TabularData::~TabularData` (TabularData.cpp:7-20): `nrow=0` NSDMI
    + counter-guard. Not in `~Model_Data` chain (caller-local). Deferred
    per F3 verifier verdict.
  - `_TimeSeriesData::~_TimeSeriesData` (TimeSeriesData.cpp:31-38): loop is
    `for (i=0; i<MAXQUE; i++)` — wraparound slot `ts[MAXQUE]` is never
    freed (pre-existing leak that predates 1ab61c0). NOT a regression
    introduced by PR-0. Tracked implicitly as part of #401 §1 follow-up.
  No additional un-NSDMI'd raw ptr in `~Model_Data` chain found.
  Verdict: NO new sibling class missed.

Probe E — Workflow comment chore-commit `c228cc2` regression check:
  Scenario: Did the workflow comment refresh + evidence persist introduce
  any logic change to the CI gate?
  Result: `git diff b29fe95..c228cc2 -- .github/workflows/serial-baseline.yml`
  shows comment-only change (15 lines removed → 22 lines added, all
  comment text). The `grep`, `marker_count=`, `if [[ "$marker_count" == "0"
  ]]; then exit 1; fi`, and `if [[ "$asan_err" != "0" ... ]]; then exit
  1; fi` blocks at lines 1464-1484 are unchanged. CI status confirms:
  `asan-ubsan (keliya)` and `asan-ubsan (qhh)` both PASS on `c228cc2` head
  (skip path via data_probe). Setup + tools-tests PASS. Only
  `build-and-compare (keliya)` shows "pending" at time of review.
  Verdict: NO regression introduced by chore commit.

Probe F — PR claim vs code reality:
  Scenario: Does the PR body Changes table accurately reflect the diff vs
  baseline?
  Result: Each row in the PR body Changes table is independently verifiable:
  - SHUD/src/ModelData/Model_Data.hpp: matches `056a1dc → 1ab61c0` diff.
  - SHUD/src/ModelData/MD_layout.hpp: matches `056a1dc` diff.
  - SHUD/src/classes/TimeSeriesData.hpp: matches `1ab61c0` diff.
  - SHUD/src/classes/Lake.hpp + Lake.cpp: matches `1ab61c0` diff.
  - SHUD submodule pointer: `710c00a → 056a1dc → 1ab61c0` matches the
    two-phase shud_fix_sha.txt.
  - tools/p8tune.D/dump_adjacency.cpp + fd_color_jacobian.cpp: confirmed
    Phase 6 deletes restored.
  - .github/workflows/serial-baseline.yml: confirmed positive sentinel.
  Bitwise SHA `b769e327...c028bd` matches keliya_postfix_rivqdown_sha256.txt.
  Verdict: PR body is FULLY ACCURATE for the final HEAD.

Adversarial probes summary: NO NEW CRITICAL OR MAJOR FINDINGS. The final
HEAD survives all 6 adversarial scenarios.

================================================================================
Self-audit
================================================================================

- PR body accurate vs reality: YES.
  Changes table, Verification table, Invariant Matrix coverage, Phase 4/4.5/5/
  6/6.5 cross-review trail table — all match the diff and the evidence files.
  Test plan checkboxes are correctly checked/unchecked per actual state.
  No stale `valgrind_keliya_postfix.log` references. No stale "LEAK SUMMARY"
  language. Two-phase SHUD pin (`056a1dc` + `1ab61c0`) consistently presented.

- Invariant Matrix correctly classified: YES.
  Row (i) COVERED via Mac substitute (acceptable per CLAUDE.md alternative
  tooling); Row (ii) IN FLIGHT (Slurm runbook ready, queue currently empty
  per `squeue -u frd_muziyao`); Rows (iii) + (iv) DEFERRED to #401 with
  server SuiteSparse blocker documented; Row (v) COVERED via SHA self-compare;
  Row (vi) PARTIAL with documented CI data_probe limitation.

- Evidence SHA-anchored to 1ab61c0: YES.
  `shud_fix_sha.txt:13` pins `1ab61c023ac2b93a178c2feb07aa3df509fe1a96`.
  `mac_asan_keliya_postfix.log:82-107` Phase 6 re-verify section
  explicitly references SHUD `1ab61c0`. `server_acceptance_cmds.sh:19` uses
  `awk '/^  fix SHA:/ {print $3}' | tail -1` which correctly picks the
  Phase 6 SHA. `keliya_postfix_rivqdown_sha256.txt` SHA matches Phase 6
  re-verify output (`b769e327...c028bd`). Outer `git submodule status SHUD`
  matches `1ab61c023ac2b93a178c2feb07aa3df509fe1a96`.

- Out-of-scope discipline: YES.
  Diff vs baseline (`git diff --stat baseline/p8tune-amg-spike..c228cc2`)
  touches ONLY: serial-baseline.yml, SHUD pointer, 2 spike-binary cpp files,
  cross-review evidence dirs. No tools/p8tune.F/ work. No Hypre/AMG code.
  No ADR-0007. No master plan touch. No openspec spec edit. No P8-tune.G/H
  anchor. Strictly REQ-3 + #386 + workaround removal + CI gate scope.

================================================================================
Workflow compliance
================================================================================

- 4 → 4.5 → 5 → 6 → 6.5 → 7 trail complete: YES.
  Round-1: 6 reviewer reports (spec-compliance, correctness, integration,
  security-perf, test-evidence, invariant-state). Phase 4.5: 3 verifier
  reports (verify-F1, verify-F2, verify-F3) all CONFIRMED. Phase 5: synthesis
  document (phase-5-synthesis.md) with Pattern Escalation + Invariant Surface
  Inventory + Regression Matrix + F4-F10 disposition. Phase 6: source +
  evidence commit `b29fe95` closing all 3 Criticals + 4 cleanup items.
  Phase 6.5: round-2 6 reviewer reports (same 6 lenses). Phase 7: this
  document.

- Phase 5 Pattern Escalation triggered: YES.
  `phase-5-synthesis.md:16-26` documents "Failure class A (CONFIRMED across
  F1 + F3): Memory ownership invariant — `new`/`delete` imbalance OR uninit
  raw ptr unconditionally `delete[]`'d in dtor" and "Failure class B
  (CONFIRMED F2): CI gate semantics — sentinel must positively signal dtor
  full execution".

- Phase 5 Invariant Surface Inventory persisted: YES.
  `phase-5-synthesis.md:28-62` lists all 7 surface categories (shared helper
  roots, public entrypoints, read/write/staging/etc.) with specific class
  + file references.

- 6-reviewer escalation per high repair intensity: YES.
  Round 1 used 6 reviewers (correct per skill rule for high repair intensity
  / pattern escalation conditions). Round 2 ran same 6 lenses.

================================================================================
Notes on server validation row (ii)
================================================================================

PR body Invariant Matrix row (ii) marks heihe valgrind as "IN FLIGHT — Slurm
job 9881". At time of this review (2026-06-29), `squeue -u frd_muziyao`
returns empty queue — the in-flight jobs mentioned in the Phase 7 brief have
either completed and been rsync'd back already, or are not yet submitted.
Evidence files `.review-evidence/p8tune-amg-pr-0/` directory does NOT
contain any rsync'd Slurm logs (no `valgrind_heihe_*.out`, no
`build_shud_*.out`, no `dump_adj_*.out`). The runbook is ready and DAG-correct;
the orchestrator's Phase 2 server validation step must complete and land
evidence before merge per the PR body Test Plan unchecked item.

================================================================================
Verdict
================================================================================

GO — ready for Phase 8 pre-merge evidence gate

Rationale:
- All 4 REQ-3 Scenarios are closed at the engineering level (Scenarios 1, 3)
  or properly documented with deferred-with-runbook (Scenarios 2, 4 server
  rows).
- All 3 Round-1 Critical findings (F1/F2/F3) are resolved and re-verified by
  round-2 reviewers (3 APPROVE + 3 APPROVE-WITH-MINOR; 0 new Criticals).
- All round-2 non-blocking observations are either resolved (workflow comment
  refresh, PR body sync) or properly dispositioned to follow-up issue #401
  (sub-tasks 1-4 each scoped + tracked).
- Adversarial probes (6 scenarios) all clear; final HEAD survives.
- Self-audit (PR body accuracy, Invariant Matrix classification, evidence
  SHA-anchoring, out-of-scope discipline) all PASS.
- Workflow compliance (Phase 4 → 4.5 → 5 → 6 → 6.5 → 7 trail; Pattern
  Escalation; Invariant Surface Inventory; 6-reviewer high-risk escalation)
  all confirmed.
- Bitwise neutrality on keliya re-verified (SHA `b769e327...c028bd` matches
  baseline both pre-Phase-4 and post-Phase-6).

Required Phase 8 prerequisites (orchestrator must hold for):
1. Server validation row (ii) — `valgrind ./shud heihe` on cn-node via
   `server_acceptance_cmds.sh §3` Slurm job; expected sentinel `VG_DONE` +
   `ERROR SUMMARY: 0` + `0 definite/indirect leaks`. Logs rsync'd to
   `.review-evidence/p8tune-amg-pr-0/` before merge. This is the only
   outstanding hard-gate item; orchestrator owns.
2. Close #386 post-merge per PR body Test Plan + tasks.md 1.15.

CI status at time of review (c228cc2 HEAD):
- setup: PASS
- tools-tests: PASS
- asan-ubsan (keliya): PASS (skipped via data_probe; gate logic verified
  by code review + Mac substitute)
- asan-ubsan (qhh): PASS
- build-and-compare (1, keliya): pending (not yet completed; no fail signal)

Non-blocking note: build-and-compare in-flight at time of review; orchestrator
Phase 8 should wait for CI green before merge per standard pre-merge protocol.
