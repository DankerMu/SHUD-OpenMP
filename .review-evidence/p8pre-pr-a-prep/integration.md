Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d
Summary: Integration surfaces all clean — output path schema, per-cell artifact set, build-flag gating, CI, gitignore, render dry-run all align with downstream PR-B/C/F contracts; one non-blocking gitignore note + one suggestion on JID placeholder grep robustness.

Findings:
- None.

Non-blocking notes:

1. (Checklist #4 — gitignore coverage) `.gitignore` L19-31 enumerates `.s*-runs/` + `.p1c-runs/` + `.p1d-runs/` + `.p1e-runs/` explicitly but has NO entry for `.p8pre-runs/`. The wrapper comment at `tools/p8pre/render_n8_profile.sh:27` claims it is "auto-gitignored by the outer repo's .gitignore '.s*-runs/' pattern style" — this is incorrect: the existing pattern is `.s*-runs/` which would match `.s6c12a-runs/` but NOT `.p8pre-runs/` (different first char). Verified empirically: no `.p8pre-runs/` in `.gitignore`. On the server, `.p8pre-runs/` lives under `/scratch/.../SHUD-OpenMP/` which IS a git working tree, so untracked clutter can leak into `git status`. RECOMMEND: add `.p8pre-runs/` (or `.p8pre-pr-*-runs/`) entry to outer `.gitignore` either in this PR (1-line change) or in the PR-A run scope (#341) before first submission. Non-blocking because the path is on server `/scratch` and the dry-run on Mac doesn't create the dir; no immediate leak.

2. (Checklist #6 — JID placeholder collision risk) Placeholder format `__PREV_JID_<case>_N<n>_rep<r-1>__` (e.g. `__PREV_JID_heihe_x4_N8_rep2__`) is grep-unique against `__CASE__` / `__N__` / `__REP__` / `__NODE__` markers (different prefix `__PREV_JID_`), but the runner in #341 must substitute placeholders in the `sbatch ... --dependency=afterany:<placeholder>` LINE (stdout), NOT in the rendered `.sbatch` files (which contain `__CASE__` / `__N__` / `__REP__` / `__NODE__` already expanded by awk). Recommend the #341 runner doc explicitly call this out: substitution target is the captured stdout sbatch invocation line, not the file body. Verified the placeholder appears 9× in dry-run stdout (one per dependent rep × 2 cases × 3 N × 2 dependent reps = 12; observed 12 via `grep -c afterany`).

3. (Checklist #12) Confirmed `bash tools/p8pre/render_n8_profile.sh 2>/dev/null | grep -c "_N2_"` returns `0`; total sbatch lines = 18; PASS.

4. (Checklist #1, #2, #3, #5, #7, #8, #9, #10, #11) All PASS:
   - #1 Output dir: `tools/p8pre/render_n8_profile.sh:53` sets `OUT_DIR=/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/rendered`; per-cell artifact dir `submit_n8_profile_template.sbatch:48-49,59` writes to `/scratch/.../.p8pre-runs/__CASE___N__N___rep__REP__/`. Matches PR-B (tasks §3.1) + rsync mirror (§2.4) path convention `<case>_N<n>_rep<r>`.
   - #2 Per-cell artifacts: template L113-119 copies `profile.yaml` + `cvode_stats.txt` + `rivqdown.dat` into `$CELL_DIR`; `slurm.out` + `slurm.err` written via `#SBATCH --output/--error` L48-49. All 5 artifacts required by tasks §2.3 covered.
   - #3 Build flag gating: template L17,86 documents pre-flight build `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` and ONLY `./shud __CASE__` per cell (L93) — no per-cell rebuild. The Mode C build with profile flag is correctly delegated to PR-A run preflight (tasks §2.0 in spirit / handoff doc §5).
   - #5 `SCRIPT_DIR` discovery (`render_n8_profile.sh:48`) uses canonical `BASH_SOURCE[0]` + `cd && pwd` idiom — robust to symlink and absolute path invocation.
   - #7 `.p8pre-runs/` namespace distinct from `.p1e-i-runs/`; no risk of cross-aggregator confusion.
   - #8 No `pr-i-runs` (deprecated) references in new files. Only `docs/p8pre/step1_prep.md` mentions it as a historical correction note (pre-existing, not in this PR).
   - #9 CI `.github/workflows/serial-baseline.yml` has empty `on:` and `jobs:` (workflow disabled) per inspection — no path glob risk.
   - #10 Spec compliance (3-nm gate including `GOMP_parallel`) documented at `docs/p8pre/pr_a_prep_evidence.md:38-47` Table — covered by correctness reviewer; cross-checked here for traceability.
   - #11 PR-C capstone doc `docs/p8pre/n8_profile_baseline.md` (tasks §4.1) is correctly distinguished from this prep doc `docs/p8pre/pr_a_prep_evidence.md` — the latter explicitly references the former at L132-147 as "downstream", not conflated.

