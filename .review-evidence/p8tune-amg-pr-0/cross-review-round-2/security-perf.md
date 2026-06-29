# Cross-Review Round 2 — Security/Performance Lens

Reviewer agent: review-security-perf
Review round: round 2
Reviewed head SHA: `b29fe95deb22bfed859181e138bc61a8e06f534f`
SHUD inner SHA:    `1ab61c023ac2b93a178c2feb07aa3df509fe1a96` (parent `056a1dc` on `openmp-baseline`)

Summary: round-1 F2 / F3 / F6 / F8 all closed at the *engineering* level; one residual semantic gap in F2 (positive sentinel prints BEFORE the dtor chain runs, so a silent dtor-skip regression that doesn't crash would still pass the gate). The CI is no longer hard-fail-on-clean — that critical defect is resolved. Spec docs (`design.md` / `tasks.md`) still reference the old broken gate but that perimeter belongs to spec-compliance lane.

---

## Round-1 findings resolution

- **F2 (CI LEAK SUMMARY broken)**: **resolved as a hard-CI-fail-on-clean defect; partial as a dtor-coverage gate**.
  - The new env `ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:print_stacktrace=1:print_suppressions=0` (`.github/workflows/serial-baseline.yml:1432`) plus positive sentinel grep on `'The successful end\.'` against `sanitizer_run_*.stdout.log` (L1457, L1465-1469) is well-formed: clean runs PASS, `_exit(0)` regressions in `shud.cpp` main body FAIL, sentinel-check runs FIRST so it isn't masked by `asan_err` fall-through. Per-Mac evidence (`mac_asan_keliya_postfix.log:117`) confirms the sentinel is emitted on stdout L218 of a clean keliya run — the gate will match.
  - **Residual gap**: `'The successful end.'` is printed by `Model_Data::TimeSpent()` (`SHUD/src/ModelData/Model_Data.cpp:32,40`), called from `modelSummary(1)` at `SHUD/src/Model/shud.cpp:291` (coupled path) and L507 (uncouple path). `delete MD` (which triggers `~Model_Data() → FreeData()` — the chain whose UB was #386's root cause) is invoked at `shud.cpp:359` / L591, ~68 lines / multiple `CVodeFree` + `SUNContext_Free` + profile dumps LATER. So the sentinel marks "main reached normal return path", NOT "dtor chain ran to completion". Three scenarios:
    1. Dtor crashes with UAF/OOB/double-free → `halt_on_error=1` triggers `==pid==ERROR: AddressSanitizer:` → `asan_err` grep at L1446 catches it → exit 1. **OK**.
    2. Dtor entire chain regressed silently (e.g. someone adds `_exit(0)` AFTER `modelSummary(1)` print but BEFORE `delete MD` at L359, or `~Model_Data()` is no-op'd) → sentinel printed, no ASan error, no UBSan error → gate PASSES. **Gap**.
    3. `_exit(0)` in main body BEFORE `modelSummary(1)` → sentinel absent → gate FAILS. **OK** (this is the explicit regression mode `a7eb922` introduced).
  - The gap (scenario 2) is narrow: any future regression that touches code between L291 and L359 of `shud.cpp` would have to silently skip the dtor chain WITHOUT triggering ASan. Probability is low but the gate's stated contract ("sentinel emission proves dtor chain ran") is overpromised. The fix is to move the sentinel emission later — e.g. `__attribute__((destructor))` global hook printing a unique token AFTER `~Model_Data()` completes, or relocate the existing `TimeSpent()` "successful end" print into a post-`delete MD` shud.cpp main-tail line. Both are local SHUD-side changes, NOT this PR's scope. Acceptable to ship as-is given Phase 5's "scope = #386 invariant closure, not gate redesign".
  - Verifier action: the comment block `.github/workflows/serial-baseline.yml:1454-1456` saying "marker is generated IFF the C++ dtor chain ran to completion" overstates by stating both directions. The forward direction ("dtor full → sentinel present") is `False` in the literal IFF sense (sentinel is BEFORE dtor); the reverse direction ("sentinel absent → dtor skipped on main body") is what actually holds.
  - Removed-behavior audit lens: `LEAK SUMMARY:` regex removal — grep across `.github/`, `tools/`, `docs/`, `openspec/` shows NO other consumer of `LEAK SUMMARY:` outside the workflow + the now-stale spec docs (`openspec/changes/p8tune-amg-spike/design.md:113,118,126,137` + `tasks.md:13` — spec lane). No CI step, no downstream tool, no rsync transformation breaks. `exitcode=0` removal is also safe — the only ASan-options consumer is this single env block.

- **F3 sibling-class warning**: **resolved (NSDMI added)** with one deferred follow-up per Phase 5 disposition.
  - NSDMI nullptr added to: `_TimeSeriesData::ts[MAXQUE+1]` brace-init at `TimeSeriesData.hpp:43` (`= {nullptr}`); `LakeBathymetry::{index,yi,ai}` at `Lake.hpp:65-67`; `_Lake::{iEleLake,iEleBank,iRivIn,iRivOut,RivIn,RivOut,QEleSurf,QEleGW,QRivIn,QRivOut}` at `Lake.hpp:96-101 + 108-111`. Per-ctor cost: ~13 nullptr stores — O(1) per `Model_Data` instantiation, negligible.
  - `delete[]` correctness fix at `Lake.cpp:60-62` (LakeBathymetry) + `Lake.cpp:97-99,102,105` (_Lake) — six scalar `delete` corrected to `delete[]` to match `new[]` allocations. For POD pointee types (`int*`/`double*`) the libc free path is identical, so no perf delta; the standards-correctness change eliminates a class of warnings + future-proofs against non-trivially-destructible array element types. No security regression.
  - **Deferred follow-up (acknowledged in Phase 5 disposition)**: `TabularData::{x, nrow}` — `nrow=0` NSDMI default IS in `TabularData.hpp:17` so `if(nrow > 0)` guard makes the unallocated case safe (`delete[] x` never runs). `double **x` itself has no NSDMI but the counter guard suffices. Not embedded in `Model_Data` (used as caller-local helper). Acceptable to ship.

- **F5 7 pre-existing un-freed ptrs**: **deferred per Phase 5 synthesis** (`docs/cross-review-round-1/phase-5-synthesis.md:60,84`). No new follow-up issue commit visible in `git log` since Phase 6 push. Suggest the orchestrator capture this as an open issue before the epic closes, OR document the deferral in the PR description so it doesn't get lost when the next reviewer audits `FreeData()`. Severity unchanged from round 1: not a regression, not in PR-0 scope.

- **F6 server script**: **resolved**.
  - `mkdir -p ${SCRATCH}/.p8tune.F-runs` now precedes `cd ${SCRATCH}/.p8tune.F-runs` (`server_acceptance_cmds.sh:39`) — fresh-deployment failure fixed.
  - Build → valgrind / dump_adj DAG wired via `--dependency=afterok:${BUILD_JID}` (L66-69, L94-97, L124-127) — valgrind + dump_adj will not run against stale binary.
  - `make libshud.a -j4` added at L56 — matches the new SHUD build target needed by p8tune.D tools.
  - **Path safety**: `SCRATCH` is a hard-coded literal; no user input crosses ssh boundary; heredocs use `'EOF'` (single-quoted delimiter) to suppress remote re-expansion of the embedded `set -euo pipefail` etc. The only outer-expanded variable inside the ssh quoted body is `${SCRATCH}` itself, which is literal. **SAFE**.
  - **Slurm 三铁律 compliance**: `#SBATCH --output=${SCRATCH}/.p8tune.F-runs/*.out` (L49-50, L85-86, L113-114) — all in `/scratch` shared FS ✓. `sbatch --parsable` submitted from `cd ${SCRATCH}/.p8tune.F-runs && sbatch` (L67, L95, L125) — submitted from `/scratch` ✓. sbatch scripts written via heredoc into `${SCRATCH}/.p8tune.F-runs/` (L41-61, L75-93, L103-123) — scripts in `/scratch` ✓. **PASS** all three rules.
  - **Minor parsing fragility (not a defect, observation only)**: `SHUD_FIX_SHA="$(awk '/^  fix SHA:/ {print $3}' ... | tail -1)"` picks the last `fix SHA:` line in `shud_fix_sha.txt`. Today that's the Phase 6 SHA (intended). If a future Phase 7+ entry is appended ABOVE (some changelogs do "newest on top"), `tail -1` would pick the wrong SHA. The current `shud_fix_sha.txt` has Phase 6 below Phase 4 — convention is newest-at-bottom; safe as long as the convention holds.

- **F8 exitcode=0 cleanup**: **resolved**. Removed from `ASAN_OPTIONS` per F2 closure (L1432). `halt_on_error=1` preserved so UAF/OOB/double-free still hard-fail. `detect_leaks=0` matches the Mac platform reality (LSan unsupported on Darwin AArch64 per evidence file L43-44) and pivots dtor-coverage proof to the positive sentinel approach. CI runtime budget: positive sentinel grep is O(N) on stdout file size — negligible (stdout is ~200 lines for keliya 90-day).

---

## Phase 6 delta findings (Sec/Perf-specific)

### Suggestion: positive sentinel emits BEFORE dtor chain — gate contract overstates coverage

`.github/workflows/serial-baseline.yml:1454-1456` + `SHUD/src/Model/shud.cpp:291,359,507,591` + `SHUD/src/ModelData/Model_Data.cpp:32,40`

Detailed scenario analysis above (F2 residual gap). Not a blocker for THIS PR — the gate is strictly better than the broken round-1 version, and the specific regression class it MISSES (silent dtor skip without crash, code added between sentinel print and `delete MD`) is narrow + unlikely. Recommend a follow-up Issue to either (a) move the sentinel print to a `__attribute__((destructor))` global hook so it provably runs AFTER all heap dtors, or (b) add a second positive marker emitted from `~Model_Data()` itself (`screeninfo("[dtor] Model_Data freed\n");` last line) and grep for that token instead/additionally.

The current implementation is acceptable for ship if the PR description / spec docs are updated to say "marker proves main() reached normal return path; ASan halt_on_error catches dtor crashes; complete dtor-skip would also miss output writes and trigger downstream snapshot diff" — i.e. defense in depth, not single-source proof.

### Suggestion: spec docs (`design.md`, `tasks.md`) still describe the broken `LEAK SUMMARY:` gate

`openspec/changes/p8tune-amg-spike/design.md:113,118,126,137` + `openspec/changes/p8tune-amg-spike/tasks.md:13`

Lines unchanged in Phase 6 (`git diff 09a815d..b29fe95 -- openspec/` is empty). Workflow code now diverges from spec text. Out of scope for Sec/Perf lane (this is spec-compliance lane's territory) but flagged here since the gate semantics changed in code without corresponding spec update.

### Praise: F3 sibling-class closure is bitwise-neutral by construction + delete[] form correction is C++-standards-compliant

`SHUD/src/classes/Lake.hpp:65-67,96-111` + `Lake.cpp:50-105` + `TimeSeriesData.hpp:43`

NSDMI defaults are overwritten by existing `new[]` assignments in `MD_Lake.cpp:36-43,92-95` and `readLake()` calls — production happy path is byte-identical. Mac evidence (`mac_asan_keliya_postfix.log:104-107`) confirms `keliya.rivqdown.dat` SHA256 matches pre-Phase-6 baseline `056a1dc`. The `delete` → `delete[]` array-form correction is a true UB fix per `[expr.delete]/2`: scalar `delete` on `new[]` storage is independent UB even when the element type is trivially-destructible (most C++ ABIs happen to "work" for POD because malloc/free pair up, but it's not guaranteed and ASan / UBSan / hardened allocators can flag it). Doing it doubly-safe via counter-guard + NSDMI nullptr is conservative defense-in-depth.

### Praise: server script DAG + Slurm 三铁律 compliance

`.review-evidence/p8tune-amg-pr-0/server_acceptance_cmds.sh:39,66-69,94-97,124-127`

Build → (valgrind || dump_adj) DAG via `--parsable` + `--dependency=afterok:${BUILD_JID}` is the textbook Slurm pattern. The downstream jobs run in parallel once build is OK (both depend on BUILD_JID, not on each other), which respects the 04:00:00 valgrind budget without making dump_adj wait. The 3-job summary echo (L130-133) gives the orchestrator the JIDs needed for `sacct` polling. Good runbook discipline.

### Note (no action): sibling-helper audit completeness

`SHUD/src/classes/Model_Control.hpp:130` (`double *Tout;`) + `SHUD/src/classes/IO.hpp:140` (`FILE *fid_time;`)

Two additional raw uninit-ptr declarations in `Control_Data` and `FileOut`. Both are PRE-EXISTING and NOT in the `~Model_Data()` dtor chain that #386 exercises:
- `Control_Data::Tout` — `delete Tout;` is commented out at `Model_Control.cpp:78` → no dtor call → no UB risk. Pre-existing memory leak (not regression).
- `FileOut::fid_time` — opened via `fopen` in `IO.cpp:187`, used for writes, never `fclose`'d. `FileOut` has no destructor definition (default implicit doesn't close `FILE*`). Pre-existing resource leak (not regression).

Neither is a CURRENT UB site. Defer to the same "pre-existing leaks follow-up issue" already raised by F5 in round 1.

---

## Invariant Matrix Coverage (Sec/Perf-relevant rows)

| # | Invariant | Round-2 status | Notes |
|---|-----------|----------------|-------|
| i | keliya valgrind dtor-full clean (Mac ASan substitute) | CONFIRMED post-Phase-6 | `mac_asan_keliya_postfix.log:113-121` EXIT=0, 0 errors, sentinel emitted L218 |
| ii | heihe valgrind clean (Linux server) | DEFERRED | `server_acceptance_cmds.sh` runbook ready; DAG-correct |
| iii-iv | heihe_x4 / heihe_x16 dump_adj smoke | DEFERRED | same script, build dependency wired |
| v | bitwise neutrality | CONFIRMED | keliya rivqdown SHA256 match (`mac_asan_keliya_postfix.log:106-107`) |
| vi | Mac libomp + Linux libgomp ASan dtor coverage | UPGRADED HIGH→LOW-MEDIUM | round-1 HIGH driven by broken gate; Phase 6 fix lands but residual sentinel-placement gap leaves slight overpromise. Functional pass/fail behavior is correct on clean + on `_exit`-before-print regression |

---

## Verdict

**APPROVE-WITH-MINOR** — All four round-1 findings (F2, F3, F6, F8) closed at the engineering level. The CI is no longer hard-fail-on-clean (critical defect resolved). Two minor follow-ups suggested (not blocking):

1. Update spec docs (`openspec/changes/p8tune-amg-spike/design.md:113,118,126,137` + `tasks.md:13`) to describe positive-sentinel gate semantics instead of stale `LEAK SUMMARY:` story — spec-compliance lane.
2. File a follow-up Issue to either move the sentinel emission to a post-dtor hook (`__attribute__((destructor))`) or add a second marker emitted from `~Model_Data()` to close the F2 residual gap (silent dtor-skip without crash). Owned by P8-tune.F PR-A or a small standalone PR.

Pre-existing leaks (F5 round 1 + the 2 additional `Control_Data::Tout` / `FileOut::fid_time` raw-ptr suspects noted above) remain deferred per Phase 5 disposition — recommend the orchestrator open a single tracking Issue before the epic closes.
