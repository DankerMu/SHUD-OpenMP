Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0

Summary: PR-0 integration largely sound — top-level Makefile recursion, SHUD/Makefile additive carve-out, tool-local .gitignore all check out — but the SHUD-side commit `bc919f5` neglected to update the SHUD-tracked `.gitignore` for the new `libshud.a` + `_libshud_obj/` artifacts, causing post-build SHUD submodule pollution on any other clone (server, fresh CI, peer dev). One spec-internal contradiction (REQ-1 vs REQ-7) is also worth flagging for verifier adjudication.

Findings:

- Severity: major
  Failure class: Build artifact pollutes submodule working tree on every fresh clone
  Contract or invariant: SHUD submodule HEAD `bc919f5` adds `libshud.a` archive target + `_libshud_obj/` object dir but does NOT update SHUD-tracked `.gitignore`. Per `git ls-tree HEAD .gitignore` (SHUD), the tracked `.gitignore` is unchanged (still blob `6925adfdcbced7b2624406ea7e2aa7d7d63d07de`, last touched in commit `3fe5fdef…`). The current clone shows `libshud.a` + `_libshud_obj/` under "Ignored files" ONLY because `.git/modules/SHUD/info/exclude` has lines `/libshud.a` + `/_libshud_obj/` added LOCALLY (per `git check-ignore -v libshud.a _libshud_obj/`). `info/exclude` is per-clone state and is NOT pushed to `origin/openmp-baseline`.
  Scenario or repro:
    1. Fresh clone on server: `git clone … && cd SHUD-OpenMP && git submodule update --init`
    2. `cd SHUD && make libshud.a` (or `cd .. && make shud_spike`)
    3. `git status` inside `SHUD/` → `libshud.a` + `_libshud_obj/` appear as Untracked
    4. `cd .. && git status` → shows `M SHUD` (submodule dirty)
    5. Any subsequent `git add SHUD` or `git submodule update --remote` workflow becomes contaminated
  Required test or evidence: Apply this fix in the SHUD submodule (forwarding a second `chore` commit to `openmp-baseline` + bumping outer pointer):
    ```
    cd SHUD
    cat >> .gitignore <<'IG'

    # P8-tune.D KLU spike additive carve-out — libshud.a archive + obj dir
    /libshud.a
    /_libshud_obj/
    IG
    git commit -am "chore(makefile): gitignore libshud.a + _libshud_obj/ build artifacts"
    git push origin openmp-baseline
    cd .. && git add SHUD && git commit -m "chore: bump SHUD pointer for libshud.a gitignore"
    ```
    Then re-verify on a clean clone that `git status` is clean after `make libshud.a`.
  Sibling surfaces: outer `.gitignore` ALSO does not contain `SHUD/libshud.a` or `SHUD/_libshud_obj/` (verified — `git check-ignore -v SHUD/libshud.a` → "Pathspec ... is in submodule 'SHUD'", outer ignore inert for submodule-internal paths). The fix MUST be on the SHUD side, not the outer side.
  Blocks merge: yes
  Impact: Every server build will leave the SHUD submodule visibly dirty in `git status`, breaking the basic invariant that build artifacts don't show up in version control. Mirrors a Phase 7 hygiene failure pattern seen in `.review-evidence/p8pre-pr-a-prep/integration.md` (gitignore drift). Long-tail effect: PR-A sweep authors on the server will see `M SHUD` and may inadvertently commit/bump the submodule, breaking the documented PR-0 pin.
  Requested fix: Add the two lines above (`/libshud.a` + `/_libshud_obj/`) to SHUD's TRACKED `.gitignore` via a second commit on `openmp-baseline` and a follow-up pointer bump. README §troubleshooting could also document "if `git status` shows `M SHUD` after `make libshud.a`, your SHUD checkout predates pointer XX — fetch latest openmp-baseline".

- Severity: minor
  Failure class: Internal spec contradiction between REQ-1 and REQ-7 re: submodule pointer bump
  Contract or invariant: Spec REQ-1 Scenario "Tool authoring with no SHUD source patch" L18 states verbatim: "**AND** PR-0 SHALL NOT bump the SHUD submodule pointer (pin stays at `6ce17d6`, the current live SHUD HEAD per ADR-0004 §References)". Spec REQ-7 Scenario "PR-0 tool PR boundary" L215 permits `SHUD/Makefile (additive `libshud.a` archive target only — the documented carve-out)`. Adding a target to `SHUD/Makefile` REQUIRES a SHUD-side commit, which REQUIRES an outer pointer bump. PR bumps `6ce17d6 → bc919f5` to land the carve-out.
  Scenario or repro: `git diff baseline/p8tune-klu-spike..HEAD -- SHUD` shows `-Subproject commit 6ce17d6… / +Subproject commit bc919f5…`.
  Required test or evidence: Verifier should adjudicate which clause governs. Two paths:
    (a) Treat REQ-7's enumerated carve-out as overriding REQ-1's pin freeze (the implementer's clear interpretation per tasks 1.3/1.11) and amend REQ-1 in a follow-up tasks update to read "SHALL NOT bump the SHUD submodule pointer EXCEPT for the additive `libshud.a` carve-out on `openmp-baseline`".
    (b) Restructure the carve-out so the spike tool consumes SHUD source files via path includes + per-tool .o compilation (no `libshud.a` archive, no SHUD/Makefile edit, no submodule bump). Significantly more work + may violate REQ-1 "SHALL NOT modify any file under `SHUD/src/`" if header re-includes pull in compile-time defines.
  Sibling surfaces: tasks.md 1.3 + 1.11 + 1.17 all assume path (a). spec.md REQ-1 L18 is the only assertion of path (b).
  Blocks merge: no (interpretive — pre-existing spec drafting issue, not a code defect; orchestrator should request a spec amendment in PR-B or a hotfix to the change manifest)
  Impact: Future readers / contributors will see the pointer bump and may wrongly conclude PR-0 violated REQ-1. Document drift.
  Requested fix: Add a 1-line note to spec REQ-1 L18: "(EXCEPT for the additive carve-out commit on `openmp-baseline` per REQ-7)". Or relocate the pin-freeze constraint to a separate "non-carve-out" clause.

- Severity: minor
  Failure class: `spike_run.sh` cannot honor README claim of "exit 0 on KLU_OOM_DETECTED"
  Contract or invariant: `tools/p8tune.D/spike_run.sh` L29-31 header documents: "Exit code: 0 if all 3 stages PASS or KLU_OOM_DETECTED (per REQ-5 OOM-as-data-point); 1 if dump_adjacency / fd_color_jacobian / klu_analyze_factor failed with non-zero exit + no OOM detected."
  Scenario or repro: The script uses `set -euo pipefail` (L37) and runs each stage via `"$KLU" … 2>&1 | tee -a "$LOG"`. With pipefail, a non-zero exit from `klu_analyze_factor` (whether genuine error OR OOM-as-data-point) will abort the script before the `done` marker. There is no exit-code capture or post-stage parse of stdout for `KLU_OOM_DETECTED`, so the script cannot distinguish OOM (which README §output-format L165-167 says should be exit 0) from a real failure.
  Required test or evidence: A 16-cell PR-A sweep that includes a deliberate-OOM cell (e.g., heihe_x16 + COLAMD + BTF on a deliberately-undersized node) — verify the wrapper exits 0 with KLU_OOM_DETECTED in the log and the aggregator can count it as a data point, not as a missing cell.
  Sibling surfaces: aggregator implementation (PR-B) will need to recognize OOM cells; if spike_run.sh exits non-zero on OOM, the orchestrator will mark the slot as failed.
  Blocks merge: no (PR-0 is tool-authoring; the wrapper's PR-A semantics can be fixed in PR-A's tasks.md 2.x range)
  Impact: PR-A sweep may report false-failure cells, inflating retry count.
  Requested fix: Replace each stage's pipeline with `if ! "$STAGE" … 2>&1 | tee -a "$LOG"; then if grep -q '^KLU_OOM_DETECTED' "$LOG"; then echo "...OOM at stage X — continuing as data point"; else exit 1; fi; fi`. Alternatively flip `set -e` off and check `${PIPESTATUS[0]}` after each stage.

Non-blocking notes:
- Tool-local `tools/p8tune.D/.gitignore` correctly hides all observed build artifacts (verified via `git check-ignore -v` against `dump_adjacency` / `dump_adjacency.dSYM/` / `output/` / `keliya_adjacency.csc` / `.o`); `git status --short tools/p8tune.D/` is clean.
- `SHUD/Makefile` `libshud.a` recipe correctly uses bare `SHUD_BUILD_CFLAGS + INCLUDES` (no `SHUD_DUMP_DEFINE` / `SHUD_OMP_RHS_DEFINE` / `SHUD_NVEC_OMP_DEFINE` / `SHUD_PROFILE_DEFINE`) — Config-A-clean archive per comment claim L640-642.
- `SHUD_SRC_NOMAIN` (L548) is pre-existing (defined for `smoke_strictomp` + `test_adjacency_fallback`); reuse here is appropriate.
- All 5 legacy SHUD targets intact: `shud:` (L506), `shud_omp:` (L523), `shud_asan:` (L245), `smoke_strictomp:` (L551), `smoke_configd:` (L587), `test_adjacency_fallback:` (L610). `clean:` (L666) extended with `rm -f libshud.a` + `rm -rf _libshud_obj/` correctly.
- Top-level Makefile recursion ordering correct: `shud_spike: libshud.a` forces SHUD/Makefile recursion before tools recursion regardless of `-j`.
- KLU static link order correct: `-lklu -lamd -lbtf -lcolamd -lsuitesparseconfig` resolves dependencies in the right order for static archives.
- Both Python scripts use `#!/usr/bin/env -S uv run python` — uv-only rule honored.
- `probe_cn_ram.sbatch` correctly satisfies Slurm 三铁律: `--output`/`--error` both under `/scratch/.../p8tune-klu-spike-pr-0/`, header docs instruct `sbatch` from `/scratch`, no `/tmp` references.
- Per-tool Makefile correctly parameterizes `SUITESPARSE_INC` / `COLPACK_LIB` / etc via `?=`, so server can override at `make` CLI.
- README §build-environment-survey + §link-line + §output-format + §troubleshooting all present and complete.
