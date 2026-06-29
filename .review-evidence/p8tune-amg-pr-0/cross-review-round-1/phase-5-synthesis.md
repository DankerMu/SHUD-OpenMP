# PR-0 Phase 5 Fix Synthesis + Pattern Escalation

PR: #400  
Head SHA: `09a815dbcab9eabbddcad9550c706ddfa8636519`  
Review round: round 1 (6-reviewer high-risk escalation)  
Synthesis date: 2026-06-29

## Verified findings (Phase 4.5 verdicts)

| ID | Severity | Failure class | Verdict | Sites |
|---|---|---|---|---|
| F1 | Critical | Memory ownership — dtor not invoked on spike main success path | CONFIRMED | `tools/p8tune.D/dump_adjacency.cpp:398-402,440-442` + `tools/p8tune.D/fd_color_jacobian.cpp:282-286,471-473` |
| F2 | Critical | CI validation gate semantics inverted / unreachable / skipped | CONFIRMED (3 sub-defects) | `.github/workflows/serial-baseline.yml:1413,1441,1449,1458,1463,1466` |
| F3 | Critical | Sibling-class invariant closure incomplete (same uninit-ptr + delete anti-pattern) | CONFIRMED | `SHUD/src/classes/TimeSeriesData.{cpp:28-38,hpp:41}` + `SHUD/src/classes/Lake.cpp:{37,52-58,82-95}+.hpp:{65-67,88-101}` |

## Pattern escalation: **yes**

**Failure class A (CONFIRMED across F1 + F3)**: *Memory ownership invariant — `new`/`delete` imbalance OR uninit raw ptr unconditionally `delete[]`'d in dtor*

**Invariant (one sentence)**: Every heap pointer that may be `delete[]`/`delete`'d (either by class dtor or by explicit caller) MUST be either (a) initialized to nullptr in its ctor (NSDMI preferred) or (b) matched 1:1 with a `delete` in the same scope as its `new`. `delete[]` is required for arrays, `delete` (scalar) for non-arrays — mismatches are independent UB.

**Trigger**: 2 of 6 round-1 reviewers (correctness, integration) cited F1's exact pattern; 2 (security-perf, invariant-state) cited F3's sibling-class pattern. F1 verifier confirmed via line refs + commit `a7eb922` ownership convention; F3 verifier confirmed via spec REQ-3 line 49 explicit "sibling `~Model_Data` chain" inclusion. Same failure class, different scope (tool main vs SHUD class) — class-level closure required, not per-line.

**Failure class B (CONFIRMED F2)**: *CI gate semantics — sentinel must positively signal dtor full execution, not negatively absent leak*

**Invariant**: The dtor-coverage CI gate MUST emit (or grep for) a stdout/stderr marker that is **produced iff** the C++ destructor chain ran to completion. Leak-absence (no `LEAK SUMMARY:` line) is the WRONG signal because it conflates "dtor full + no leak" with "dtor skipped + no time to emit summary" with "step didn't run at all".

## Invariant Surface Inventory (per Phase 5 high-risk hard-gate)

**Shared helper roots**:
- `SHUD/src/ModelData/` — `Model_Data` + `MD_layout` (already fixed in 056a1dc) 
- `SHUD/src/classes/` — `_TimeSeriesData`, `_Lake`, `LakeBathymetry` (NEW — Phase 6 scope per F3 verifier)

**Public entrypoints**:
- `./shud <case>` CLI main (stack-allocates Model_Data per implementer ASan log; dtor runs by accident — OK)
- `tools/p8tune.D/dump_adjacency` main (Phase 6 fix: restore success-path `delete MD; delete fout; delete fin;` or stack-allocate)
- `tools/p8tune.D/fd_color_jacobian` main (same)

**Read surfaces**: none changed

**Write/delete/overwrite surfaces**:
- SHUD output `.dat` files — bitwise neutrality verified pre-fix (keliya); Phase 6 fix MUST preserve

**Staging/publish/rollback surfaces**:
- SHUD submodule `openmp-baseline` branch — Phase 6 SHUD inner fix needs new commit on top of `056a1dc`, push openmp-baseline

**Producer/consumer evidence boundaries**:
- `.review-evidence/p8tune-amg-pr-0/` — Phase 6: update `server_acceptance_cmds.sh` (F6 mkdir+dep bugs) + rename `valgrind_keliya_postfix.log` → `mac_asan_keliya_postfix.log` (F7)

**Stale-state/idempotency boundaries**:
- CI ASan job — Phase 6: replace `LEAK SUMMARY:` gate with positive sentinel; reorder exit-on-asan_err vs leak_summary check; consider data_probe skip behavior (mark skip as FAIL, OR install forcing fixture on runner, OR document skip is non-evidence)

**Unchanged downstream consumers** (verify NOT regressed):
- B0/B1a/B1b baseline archives (`tools/compare_snapshot`)
- P8-tune.D archived spec + aggregator
- CVODE callers of `MD->rhs_core`

**Surfaces intentionally out of scope**:
- `TabularData` (F3 verifier confirmed anti-pattern exists, but `nrow=0` NSDMI guards unallocated case + class is caller-local not in `~Model_Data` chain) — defer to follow-up issue
- Pre-existing un-freed 7 `Model_Data` ptrs (F5 — not regression, separate bug)
- Cross-toolchain server validation (already deferred to orchestrator Phase 2)

## Regression Matrix (high-risk required: 1+ adversarial per affected surface category)

| Surface | Input | Expected |
|---|---|---|
| `tools/p8tune.D/dump_adjacency` success path | keliya NumY=1500 | exit 0, `delete MD; delete fout; delete fin;` runs, ~Model_Data + ~ElementHotData + ~_TimeSeriesData + ~_Lake fire in order, ASan reports 0 leaks |
| `tools/p8tune.D/fd_color_jacobian` success path | keliya NumY=1500 | same as above + J binary written + dtor chain |
| `tools/p8tune.D/dump_adjacency` server | heihe_x4 NumEle=40046 | dtor-full exit 0, no heap corruption (regression that motivated #386) |
| `tools/p8tune.D/dump_adjacency` server | heihe_x16 NumY≈485K | same |
| `_TimeSeriesData::~_TimeSeriesData()` | partial-init via exception mid-ctor | unallocated `ts[i]` slots are nullptr → `delete[] nullptr` no-op (safe) |
| `_Lake::~_Lake()` | partial-init via OOM in readLake() between alloc calls | unallocated ptrs are nullptr → safe; allocated ones get `delete[]` (NOT `delete` — fix Lake.cpp:86-88,91,94 array-vs-scalar mismatch) |
| `LakeBathymetry::~LakeBathymetry()` | same pattern | safe; fix Lake.cpp:52-58 `delete` → `delete[]` |
| CI ASan keliya job post-fix | leak-clean dtor-full run | grep gate PASSES (positive sentinel emitted), CI green |
| CI ASan keliya job post-fix | injected `_exit(0)` regression | grep gate FAILS (positive sentinel absent), CI red |
| B0/B1a/B1b bitwise neutrality | keliya `.dat` SHA256 | byte-identical pre/post Phase 6 fix (NSDMI defaults can't change output; spike binary delete-additions also can't change SHUD CLI output since they're tool-main only) |
| Unchanged P8-tune.D archived tools | `tools/p8tune.D/aggregate_klu_spike.sh` | runs same as before (no interface change) |

## F4-F10 disposition (orchestrator decision, no verifier needed)

| ID | Source | Disposition |
|---|---|---|
| F4 | test-evidence | Non-blocking note. Pre-fix SHA witness artefact cannot be retroactively captured; cite #386 evidence (cn-node jobid 9793/9792) in PR comment. |
| F5 | security-perf | Non-blocking note. 7 ptrs (`AccT_surf/AccT_sub/lake/y2LakeArea/t_sph`) never `delete[]`'d in `FreeData()` — pre-existing leak, not regression. Defer to follow-up issue (spawn). |
| F6 | security-perf | **Fix in Phase 6**: update committed `server_acceptance_cmds.sh` (add `mkdir -p` before `cd` + `--dependency=afterok:$JID` between jobs). |
| F7 | spec-compliance + test-evidence | **Fix in Phase 6**: rename `valgrind_keliya_postfix.log` → `mac_asan_keliya_postfix.log` (+ in-file header line "valgrind substitute — Mac ASan+UBSan"). |
| F8 | spec-compliance | Non-blocking note. Linux valgrind keliya not booked in `server_acceptance_cmds.sh`; rely on server heihe + CI keliya ASan instead. Document in PR comment. |
| F9 | correctness | **Fix in Phase 6**: remove NSDMI from dead `Model_Data::{ISFactor, windH}` (top-level windH; the `hot.windH` in ElementHotData is real and stays) — cosmetic cleanup. |
| F10 | spec-compliance | Non-blocking note. Cite #386 stack trace evidence as the canonical pre-fix repro (jobid 9762/9778). |

## Phase 6 implementer brief (cross-cutting closure — one task, two failure classes)

See `phase-6-implementer-brief.md` (created next).

## Server validation impact

Server Slurm jobs 9866/9867/9868 (build/valgrind heihe/dump_adj x4-x16) were queued at outer SHA `09a815d`. After Phase 6 fix push (new outer + new SHUD inner SHA), server needs:
- `git pull --recurse-submodules` on feat branch
- Re-submit build + valgrind + dump_adj (or wait for current jobs to complete and confirm they pass against the *old* head, then re-verify on *new* head)

Orchestrator decision (post-Phase 6 push): cancel old jobs if not started, re-submit on new head. If old jobs already ran and passed, they're still meaningful (NSDMI source fix unchanged) but the F1 fix (spike binary deletes) won't be exercised — re-submit required for full coverage.

---

**Status**: Pattern escalation engaged. Invariant Surface Inventory persisted (this file). Next: Phase 6 implementer with cross-cutting closure brief. Then Phase 6.5 re-run cross-review (same 6 reviewers per skill rule).
