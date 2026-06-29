Reviewer agent: review-integration
Review round: round 2 (post-fix)
Reviewed head SHA: 50d2a4bddacbfa3ef5b3e1c25d760555103c5556
SHUD pointer (round 2): 41d9a172610c1c628fcf4b1b0a4f7c19f4afc854 (on openmp-baseline; bumped from bc919f5)

Summary: Round 1 integration findings c02 and c11 are both fully ADDRESSED. F2 (SHUD-side gitignore amendment) is real, pushed to remote `openmp-baseline`, additive only, and the outer submodule pointer correctly tracks it. F7 (spec REQ-1 carve-out caveat) is present in the working-tree spec.md and openspec strict validation PASSes. No new build-system regressions or envelope drift introduced by the fix. Spec amendments live in `openspec/changes/` which the project gitignores — a documented project convention. Round 2 verdict: APPROVE from integration perspective.

Verification of round 1 findings:

- c02 (major): SHUD-tracked .gitignore missing libshud.a + _libshud_obj/ — fresh-clone pollution.
  Round 2 status: **ADDRESSED**
  Evidence:
    - SHUD commit `41d9a17` ("chore(makefile): gitignore libshud.a + _libshud_obj/ build artifacts") adds exactly the two required patterns to the tracked `.gitignore`:
        ```
        # P8-tune.D KLU spike additive carve-out — libshud.a archive + obj dir
        /libshud.a
        /_libshud_obj/
        ```
      Verified via `cd SHUD && git show 41d9a17 -- .gitignore` (4 lines added; no other changes).
    - **Remote push confirmed**: `git ls-remote https://github.com/SHUD-System/SHUD.git openmp-baseline` returns `41d9a172610c1c628fcf4b1b0a4f7c19f4afc854 refs/heads/openmp-baseline` — the commit is on the upstream branch, not just local.
    - Outer pointer bump confirmed: `git show 50d2a4b -- SHUD` shows `-Subproject commit bc919f5… / +Subproject commit 41d9a17…`.
    - Cumulative SHUD diff from pre-spike pin: `git diff 6ce17d6..41d9a17 --stat` reports only `.gitignore | 4 ++++` + `Makefile | 45 ++++++…` — strictly additive carve-out, no `src/` or `include/` files touched (matches REQ-1 Scenario "Tool authoring with no SHUD source patch" L15).
    - No other SHUD-tracked files mutated; no extra lines smuggled into .gitignore.

- c11 (minor CONFIRMED): REQ-1 vs REQ-7 contradiction over submodule pointer bump.
  Round 2 status: **ADDRESSED**
  Evidence:
    - Working-tree `openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md` L18 now reads (verbatim):
        ```
        - **AND** PR-0 SHALL NOT bump the SHUD submodule pointer EXCEPT for the additive `libshud.a` carve-out commit(s) on `openmp-baseline` per REQ-7 (pin advances 6ce17d6 → openmp-baseline HEAD; carve-out also includes a `.gitignore` amendment to suppress `libshud.a` + `_libshud_obj/` build artifacts from polluting fresh clones)
        ```
      The carve-out caveat resolves the prior pin-freeze vs Makefile carve-out contradiction.
    - `openspec validate p8tune-klu-spike --strict --no-interactive` → `Change 'p8tune-klu-spike' is valid` (PASS).
    - Caveat is materially consistent with REQ-7 L215 enumerated `SHUD/Makefile` + `SHUD/.gitignore` carve-outs.
    - Note: `openspec/changes/` is project-gitignored (root `.gitignore` L13), so the amendment lives only in the working tree, not in any commit. This is a documented project convention (see git ls-files: no spec file is tracked) — not a defect of the fix.

F2 deep-dive (additional integration checks beyond round 1):
  - SHUD remote HEAD = `41d9a17` (confirmed via `git ls-remote`); submodule URL unchanged (`SHUD-System/SHUD`, no fork/redirect).
  - SHUD pre-existing .gitignore content (45 lines through `tests/s1d_configd_nvec_smoke.dSYM/`) preserved verbatim; new section appended at EOF.
  - Outer pointer goes `bc919f5 → 41d9a17` cleanly; no submodule URL drift in `.gitmodules` (verified via diff scan).
  - `git status` post-fix shows `modified: SHUD (untracked content)` — this is the expected presence of un-committed `libshud.a` build artifact under the gitignored carve-out, NOT a tracked-file drift. Build hygiene now works as intended.

New build issues from F1 (`--brute-force-dense` CLI flag on fd_color_jacobian):
  - **Makefile rebuild**: top-level `Makefile` and `tools/p8tune.D/Makefile` are NOT modified in 50d2a4b (verified via `git diff a65f3ca..50d2a4b -- Makefile tools/p8tune.D/Makefile` returns empty). The new flag adds a code branch inside `fd_color_jacobian.cpp` only; no new TU, no new linked lib, no rule change needed. `make shud_spike` recipe (`g++ fd_color_jacobian.cpp $(SHUD_LIBSHUD) -lklu -lColPack …`) continues to apply unchanged.
  - **Evidence file placement**: three new evidence files committed under correct `.review-evidence/p8tune-klu-spike-pr-0/` directory (verified via `git ls-tree --name-only 50d2a4b .review-evidence/p8tune-klu-spike-pr-0/`):
      - `mac_smoke_keliya_J_dense_sha256.txt` (A, sha256 of the dense J binary)
      - `mac_smoke_keliya_chi_assertion.log` (A, chi=16 ≤ 30 gate)
      - `mac_smoke_keliya_dense_fd_baseline.log` (A, brute-force dense run log)
    These match the REQ-7 envelope (`.review-evidence/p8tune-klu-spike-pr-0/`) — none gitignored, all properly tracked.

New build issues from F4 (RSS preflight reorder):
  - `klu_analyze_factor.cpp` includes unchanged (verified head-of-file scan — same `klu.h` + `chrono` + `cn_node_ram.h` set as round 1).
  - No new extern declarations; uses `Symbolic->lnz + Symbolic->unz` which are pre-existing struct fields on `klu_symbolic` from SuiteSparse `klu.h`.
  - Added `klu_free_symbolic(&Symbolic, &common)` before the OOM-path `return 0` at the new check site — correct lifetime management (no leak).
  - Smoke log `mac_smoke_keliya_klu_factor.log` shows the new `PREFLIGHT_AFTER_ANALYZE` and `PREFLIGHT_HINT` lines firing in the expected order (after symbolic factor, before numeric factor).

Spec amendments (F6/F7/F8) validation:
  - All three amendments land in the working-tree spec.md (verified via grep).
  - `openspec validate p8tune-klu-spike --strict --no-interactive` → PASS.
  - Spec file is NOT tracked in git (per project `.gitignore` rule `openspec/changes/`) — this is the project convention. The amendments are visible to reviewers via the working tree and PR description, not via the git diff.

Top-level Makefile + tools/p8tune.D/Makefile in fix commit:
  - Neither file is touched in 50d2a4b (verified via name-status). The `shud_spike: libshud.a` chain established in round 1 (`a65f3ca`) is intact:
      - `Makefile:35` libshud.a recipe → `$(MAKE) -C SHUD libshud.a`
      - `Makefile:40` shud_spike: libshud.a → `$(MAKE) -C tools/p8tune.D shud_spike`
  - No regression risk from the fix on build system.

File envelope (REQ-7 L215) gap sweep across both PR commits (`a65f3ca` + `50d2a4b`):

  | File path | Allowed by REQ-7? | Notes |
  |---|---|---|
  | `.review-evidence/p8tune-klu-spike-pr-0/*` | YES | Round-2 added 3 new files all under this allowed prefix |
  | `Makefile` (top-level) | YES (REQ-7 "top-level Makefile (new — wires `make shud_spike`)") | |
  | `SHUD` (submodule pointer) | YES (REQ-1 L18 carve-out caveat post-F7) | Bumped to 41d9a17 |
  | `SHUD_openMP_master_plan.md` | YES (REQ-7 "1-line status flip + PR-0 link") | |
  | `tools/p8tune.D/.gitignore` | **Not explicitly enumerated** | Pre-existing from a65f3ca; not modified by fix. Soft envelope drift carried over from round 1 (out of round 2 scope) |
  | `tools/p8tune.D/Makefile` | YES | Not touched in fix |
  | `tools/p8tune.D/README.md` | YES (REQ-7 enumerates explicitly) | F11 appended determinism recipe |
  | `tools/p8tune.D/cn_node_ram.h` | YES (REQ-7 enumerates) | |
  | `tools/p8tune.D/dense_fd_cross_check.py` | YES (REQ-7 post-F6 amendment enumerates explicitly) | |
  | `tools/p8tune.D/dump_adjacency.cpp` | YES (REQ-7 `*.cpp`) | F9 comment fix |
  | `tools/p8tune.D/fd_color_jacobian.cpp` | YES (REQ-7 `*.cpp`) | F1/F3/F5 code |
  | `tools/p8tune.D/klu_analyze_factor.cpp` | YES (REQ-7 `*.cpp`) | F4 RSS reorder |
  | `tools/p8tune.D/probe_cn_ram.sbatch` | YES (REQ-7 enumerates) | |
  | `tools/p8tune.D/spgmr_baseline_walls.h` | YES (REQ-7 enumerates) | |
  | `tools/p8tune.D/spike_run.sh` | YES (REQ-7 enumerates) | F10 case whitelist |
  | `tools/p8tune.D/verify_adjacency_keliya.py` | YES (REQ-7 enumerates) | |

  No forbidden files appear in the diff. Fix commit touches strictly a subset of round 1's file set + new evidence files; no scope creep.

Non-blocking notes:
- Spec file is not in git but openspec validate passes — fine under project convention; reviewers should treat the working-tree spec as the source of truth for PR-0 review.
- `tools/p8tune.D/.gitignore` envelope soft-drift inherited from round 1 — recommend a 1-line REQ-7 amendment in a future spec hygiene PR to enumerate it explicitly, but does not block this PR.
- SHUD submodule shows `modified: SHUD (untracked content)` post-build — this is the gitignored libshud.a sitting in the working tree, not a tracked-file drift. Build hygiene is now correct.
- CI/CD compatibility: the fix introduces no new dependency (still `g++` / `SuiteSparse` / `ColPack` / `uv` / SUNDIALS), no new env-var requirement, no new sbatch script. PR-A handoff path unaffected.

Round 2 verdict: APPROVE (integration). All round 1 findings resolved cleanly; no new integration risks introduced.
