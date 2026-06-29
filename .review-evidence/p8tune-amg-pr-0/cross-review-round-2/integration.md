Reviewer agent: review-integration
Review round: round 2
Reviewed head SHA: b29fe95deb22bfed859181e138bc61a8e06f534f
SHUD inner SHA: 1ab61c023ac2b93a178c2feb07aa3df509fe1a96
Summary: Round-1 Critical findings C1 (spike-binary happy-path leak) and C2 (CI dtor-coverage gate broken three ways) are both resolved with surgical, well-commented diffs. SHUD producer/consumer binding stays the gold standard (outer `1ab61c0` == inner HEAD == `origin/openmp-baseline` == evidence pin). Sibling NSDMI / `delete[]` fix in inner `1ab61c0` is correct in shape and code, with one caveat that does NOT block: the `_Lake *lake` heap object is leaked in `Model_Data::FreeData()` (pre-existing, acknowledged in commit body + round-1 invariant-state), so the sibling-class fixes are defensive-only on production paths today — they pay off the day the pre-existing leak is patched, or the day a spike binary directly drives a `_Lake` instance through dtor. Two minor doc-debt items remain in the PR body (stale evidence filename + stale CI-gate description). APPROVE-WITH-MINOR.

Round-1 findings resolution:
- C1 (F1 spike binary MD leak): resolved
  - `tools/p8tune.D/dump_adjacency.cpp:442-447` and `tools/p8tune.D/fd_color_jacobian.cpp:471-476` now end with the 3-line cleanup block (`delete MD; delete fout; delete fin;`) before the success-path `return`, symmetric to the error-path convention at fd_color_jacobian.cpp:308/318/432. Comment explicitly cites the round-1 F1 closure rationale. Evidence: `git diff 09a815d..b29fe95 -- tools/p8tune.D/` shows additive 8-line and 7-line blocks; nothing else in the file moved.
  - PR body claim "C++ destructor chain runs to completion" is now factually correct for the spike binaries' success path (it was the round-1 critique that this was false).
- C2 (F2 CI gate broken): resolved
  - `.github/workflows/serial-baseline.yml:1412-1478` replaces the negative LSan-summary grep with a POSITIVE `'The successful end.'` stdout sentinel check, gates it BEFORE the asan_err short-circuit (so an unrelated ASan error can't mask a dtor regression), and removes the `exitcode=0` ASan option (no longer needed). The new env line is `ASAN_OPTIONS: "detect_leaks=0:halt_on_error=1:print_stacktrace=1:print_suppressions=0"`.
  - Sentinel emission site verified in SHUD: `screeninfo("\n\nThe successful end. \n\n");` printed by `Model_Data::TimeSpent()` (SHUD/src/ModelData/Model_Data.cpp:32+40), called from `Model_Data::modelSummary(end=1)` (SHUD/src/Model/shud.cpp:291 / :507) right before `delete MD;` at shud.cpp:359/591. Sentinel does NOT itself prove dtor coverage (it prints in main loop completion, before dtor runs), but the combined gate (sentinel-present AND ASan/UBSan-clean) is structurally correct: any dtor SEGV/double-free/UAF still trips ASan with `halt_on_error=1`. Worded slightly more strongly than strictly true in the workflow comment ("marker is generated IFF the C++ dtor chain ran to completion" — that's the converse, not the actual direction), but the gate is correct end-to-end.
  - `mac_asan_keliya_postfix.log:117` confirms Mac sentinel emission ("Stdout L218: 'The successful end.'"); on the broken side, replacing `return 0` with `_exit(0)` in the spike binaries would (post-fix) not affect the sentinel itself (sentinel comes from shud_asan, not the spike), so the gate's regression-protection target is correctly scoped to shud_asan main.

Phase 6 delta findings:

- Severity: warning
  Failure class: producer/consumer-binding (sibling-class fix is correct, but production reach is gated on a pre-existing leak)
  Contract or invariant: design.md §D5 governing invariant "Model_Data 构造 → 使用 → 析构链全程零 invalid pointer" + spec REQ-3 line 49 explicit "sibling `~Model_Data` chain" inclusion.
  Scenario or repro: SHUD inner `1ab61c0` correctly fixes the uninit-ptr + scalar→array `delete` UB in `_Lake` (`_Lake::~_Lake()` at Lake.cpp:88-108) and `LakeBathymetry` (`~LakeBathymetry()` at Lake.cpp:53-63). However, the consumer chain `~Model_Data() → ... → delete[] lake → ~_Lake → delete[] iEleBank/...` is broken at the second hop: `_Lake *lake = nullptr` is `new _Lake[NumLake]`-allocated in `MD_Lake.cpp:36` but `Model_Data::FreeData()` (MD_readin.cpp:526-660) NEVER calls `delete[] lake`. `grep -rn 'delete\[\] lake\|delete\s*lake[ ;]' src/` returns zero hits. So in production:
    - Path "_Lake → LakeBathymetry NSDMI / delete[] correctness fix" is unreachable from `~Model_Data` because `lake` is leaked, so `~_Lake` and `~LakeBathymetry` are never called by the model CLI.
    - Path "_TimeSeriesData fix" IS reachable (9 value-type members of Model_Data; their dtors run inline on `~Model_Data`).
  This is acknowledged in commit body §"Failure classes out of scope" ("7 pre-existing un-freed Model_Data ptrs (AccT_*/lake/y2LakeArea/t_sph, pre-existing leak — not regression)") and was flagged in round-1 invariant-state finding §3 ("Side note (pre-existing, not introduced by PR-0)") + correctness §1.
  Required test or evidence: No test required for PR-0 scope; this is acknowledged carve-out. The pre-existing-leak follow-up issue (referenced in round-1 phase-5-synthesis.md as "follow-up issue (spawn)") should be filed against `Model_Data::FreeData()` to add `if (NumLake > 0) delete[] lake;` so the sibling-class fixes start paying off in production. Recommend including this in the PR description's "Known follow-ups" section, or as an explicit deferred-work issue link, since otherwise the F3 sibling fix looks more impactful than it is.
  Sibling surfaces: design.md §D5 governing invariant (currently violated by pre-existing leak — separate from PR-0 regression scope); spec REQ-3 line 49 sibling-chain wording (covered structurally; reach gated on FreeData fix).
  Blocks merge: no
  Impact: low for PR-0 (defensive correctness landed; spike binary dump_adjacency/fd_color_jacobian flows are NOT affected by lake leak because keliya has NumLake=0 per the evidence log L68; heihe / heihe_x4 / heihe_x16 also have no lake elements per the NWM case manifests). The fix becomes load-bearing when (a) the FreeData lake-leak follow-up lands, or (b) a future spike binary directly drives a `_Lake` instance through dtor (e.g. lake-specific scaffolding).
  Requested fix: Add 1 line to the PR description Test Plan or Known-Follow-Ups: "[ ] File follow-up issue for pre-existing 7-ptr leak in Model_Data::FreeData (AccT_*/lake/y2LakeArea/t_sph) so the sibling-class NSDMI/delete[] fix in 1ab61c0 becomes reachable in production".

- Severity: suggestion
  Failure class: stale-doc-debt (PR body text not synced with Phase 6 reality)
  Contract or invariant: PR description should reflect the actual code state on the head SHA being reviewed.
  Scenario or repro: PR-400 body still has two stale references introduced before Phase 6:
    (a) Verification table cell "Mac ASan + UBSan keliya 0 errors" → Evidence column reads `.review-evidence/p8tune-amg-pr-0/valgrind_keliya_postfix.log` — the file was renamed to `mac_asan_keliya_postfix.log` in Phase 6 (F7 closure per round-1 integration finding #3 + round-2 spec-compliance noted the rename). The OLD filename no longer exists on disk; only round-1 review-evidence directory retains historical references.
    (b) Changes table row ".github/workflows/serial-baseline.yml" still describes the OLD broken approach: "ASan detect_leaks 0 → 1 + exitcode=0; new grep gate fails the job if `LEAK SUMMARY:` stderr line absent". Phase 6 changed this to a POSITIVE `'The successful end.'` stdout sentinel with `detect_leaks=0`. The current PR-body description would mislead a reviewer reading PR-only (without diff) about the gate's semantics.
  Required test or evidence: Two-line PR body edit:
    (a) Change `valgrind_keliya_postfix.log` → `mac_asan_keliya_postfix.log` in the Verification table.
    (b) Replace the workflow row with: "ASan dtor-coverage gate rewritten to positive sentinel grep (`'The successful end.'` stdout); `detect_leaks=0`, no `exitcode=0`. Sentinel-absent → hard-fail dtor bypass. Phase 6 F2 closure."
  Sibling surfaces: PR description Test Plan checkbox for "CI `asan-ubsan (keliya)`: PASS with `LEAK SUMMARY:` present" — should be updated to "with sentinel present" for the same reason.
  Blocks merge: no
  Impact: cosmetic; the diff itself is authoritative and trumps PR body text. But a future reader using the PR body as the spec-of-record would be misled.
  Requested fix: Edit PR body to sync with Phase 6 (2-line + 1 checkbox-line update).

- Severity: praise
  Failure class: SHUD submodule producer/consumer binding integrity (re-verified post-Phase-6)
  Contract or invariant: CLAUDE.md SHUD submodule 工作流 — inner commit on `openmp-baseline` only, outer pointer bump tracks inner HEAD, evidence file pins agree byte-for-byte.
  Scenario or repro: Verified four-way identity chain:
    - outer `git diff 09a815d..b29fe95 -- SHUD` → `056a1dc → 1ab61c0` (exactly the F3 sibling-class fix SHA from `shud_fix_sha.txt:13`)
    - inner `cd SHUD && git rev-parse HEAD` → `1ab61c023ac2b93a178c2feb07aa3df509fe1a96`
    - inner `git branch -a --contains 1ab61c0` → `openmp-baseline` + `remotes/origin/openmp-baseline` (NOT master — workflow respected)
    - upstream `git ls-remote https://github.com/SHUD-System/SHUD.git openmp-baseline` → `1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (push completed before outer push, per the runbook ordering)
    - evidence `shud_fix_sha.txt` updated to two-phase format documenting BOTH `056a1dc` (Phase 4 Model_Data + ElementHotData) AND `1ab61c0` (Phase 6 siblings); no stale single-SHA pin
    - `mac_asan_keliya_postfix.log:107` pins post-fix SHA256 `b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd` == round-1 baseline (bitwise neutrality preserved across NSDMI + delete[] correctness fix)
  Sibling surfaces: design.md §D5 Surfaces / Failure paths row; spec REQ-3 root-cause-fix scope (now includes the explicit "sibling chain" wording).
  Blocks merge: no
  Impact: the SHUD submodule story across two phases (Phase 4 Model_Data, Phase 6 siblings) stays clean. Rollback is `git revert` on outer pointer + `git revert` on inner openmp-baseline (no master pollution to clean up; no orphaned tags).
  Requested fix: none.

- Severity: praise
  Failure class: removed-behavior audit (negative-space verification)
  Contract or invariant: removed code paths should be the WRONG ones; no upstream consumer should have relied on the removed behavior.
  Scenario or repro:
    - Scalar `delete` → `delete[]` change at Lake.cpp:53-58 + 85-95: the removed scalar form was UB on `new[]`-allocated storage (Lake.cpp:99/101/111/117/123 + Lake.cpp:134-136 all use `new[]`); no upstream caller could have relied on scalar-`delete` semantics because that path was UB and happened to work only because the element types are trivially-destructible primitives. The `delete[]` switch is correctness-only with no observable behavior change on conforming implementations. Bitwise neutrality SHA self-compare (mac_asan_keliya_postfix.log:107) confirms.
    - `exitcode=0` ASan option removed from CI: it was needed ONLY to neuter the broken LSan-summary path (which would have non-zero-exited on the LEAK SUMMARY emission). With the positive sentinel gate, regular ASan exit semantics are correct — `halt_on_error=1` still hard-fails on first UAF/OOB/double-free, and a clean run exits 0 naturally. No other CI consumer relies on this option; verified `grep -rn 'exitcode' .github/` returns only the deleted line (now absent).
    - F9 dead-field NSDMI revert: `ISFactor` and `windH` (top-level Model_Data, NOT `hot.windH` which is the live ElementHotData field used at MD_ET.cpp:93 + Model_Data.cpp:222/276/383). `grep -rn '\bISFactor\b' src/ tools/` confirms only the declaration line + the Phase 6 comment; no readers, no writers, no `delete`. NSDMI was technically correct but redundant; removal does not regress anything.
  Sibling surfaces: design.md §D5 invariant matrix (no row depends on the removed behavior); spec REQ-3 dtor coverage scenario (sentinel-based gate is the new contract).
  Blocks merge: no
  Impact: clean negative-space audit; no hidden consumer was depending on any of the three removed behaviors.
  Requested fix: none.

Invariant Matrix Coverage (post-Phase-6):
- row (i) `valgrind ./shud keliya` (NumY=1500): COVERED on Mac via shud_asan equivalent (EXIT=0, 0 ASan/UBSan errors, sentinel emitted). Linux valgrind canonical run deferred to orchestrator Phase 2 (runbook in server_acceptance_cmds.sh §3).
- row (ii) `valgrind ./shud heihe` (NumY=19515): DEFERRED — Slurm sbatch via cn-node, runbook ready (server_acceptance_cmds.sh §3, `--dependency=afterok:${BUILD_JID}` chained). Phase 6 commit body explicitly notes "Old jobs 9866/9867/9868 will be cancelled + re-submitted on this head" because F1+F3 changed runtime behavior.
- row (iii) `dump_adjacency heihe_x4` (NumEle=40046): DEFERRED — same runbook §4. **Now genuinely tests dtor-full path on success** (F1 fix restored the deletes). Round-1 weakness here ("spike binaries do not delete MD on the happy path") is resolved.
- row (iv) `dump_adjacency heihe_x16` (NumY~485K): DEFERRED — same as row (iii); same F1 resolution applies.
- row (v) bitwise neutral vs B0/B1a/B1b: COVERED + RE-VERIFIED. Post-Phase-6 SHA256 `b769e327…c028bd` matches both the round-1 pre-fix baseline AND the round-1 post-fix value, confirming the sibling-class fix is also bitwise-neutral. Mac evidence: mac_asan_keliya_postfix.log:106.
- row (vi) Mac libomp + Linux libgomp cross-toolchain: PARTIAL on Mac (covered post-Phase-6); Linux libgomp covered AT BUILD TIME (CI build job compiles SHUD against libgomp on every PR head) but NOT covered AT RUNTIME because the GH runner data_probe still skips the actual sanitizer run (no NWM forcing data on the runner). This is a documented known limitation (commit body §"Failure classes out of scope" item 3 + workflow comment). Net Linux runtime evidence still depends on orchestrator Phase 2 server jobs.

Cross-cutting boundaries verified:
- SHUD pointer + inner commit + remote push: 3-way agreement on `1ab61c0` (outer diff + inner HEAD + remote ls-remote). ✓
- PR body claim "C++ destructor chain runs to completion": now actually true post-F1 fix for both spike binaries' success path. PR body title text still aligns. Inner SHUD commit message accurately describes the sibling-chain fix. ✓ (modulo the stale-doc-debt suggestion above)
- Evidence files self-consistency:
  - `shud_fix_sha.txt:13` pins `1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (Phase 6 fix SHA) ✓
  - `shud_fix_sha.txt:7` keeps Phase 4 pin `056a1dce4b75e79779242b4796e178bebe89680b` as parent (correct two-phase format) ✓
  - `mac_asan_keliya_postfix.log` renamed from `valgrind_keliya_postfix.log` (git mv with R052 similarity score, content preserved); in-file header note explains why (F7 closure) ✓
  - `server_acceptance_cmds.sh:19` reads SHUD_FIX_SHA from shud_fix_sha.txt with `tail -1` → picks up the Phase 6 `1ab61c0` line correctly ✓
  - `keliya_postfix_rivqdown_sha256.txt` SHA `b769e327…c028bd` matches the post-Phase-6 SHA in mac_asan_keliya_postfix.log:107 ✓
  - No stale `056a1dc` reference in any evidence file outside the legitimate two-phase parent-pin context ✓

Verdict: APPROVE-WITH-MINOR — both round-1 Criticals are surgically resolved; SHUD submodule producer/consumer chain is gold-standard across both phases. The remaining items are: (1) one warning flagging that the F3 sibling-class fix's production reach is gated on a pre-existing FreeData lake leak (acknowledged in commit body, but should be tracked as an explicit follow-up issue), and (2) one cosmetic suggestion to sync 2 stale references in the PR body description. Neither blocks merge; both should be addressed before the merge button is pressed.
