---
title: "p8pre-spike Step 1 PR-A prep — build / case / template / dry-run evidence"
date: 2026-06-26
version: 0.1 (Issue #339/#340 implementer landing — PR-A run scope is #341)
status: "PR-A prep evidence chain. 4 sections: cn14 Mode C build PASS + 3 nm gates, server case basin SHALL met, template + wrapper provenance, 18-cell render dry-run sanity. PR-A actual sbatch submission is out of scope here — handed off to #341."
related_docs:
  - "openspec/changes/p8pre-spike/proposal.md (epic rationale)"
  - "openspec/changes/p8pre-spike/design.md §D4 + §D5 + §D7"
  - "openspec/changes/p8pre-spike/tasks.md §1 (PR-A prep) + §2 (PR-A run, downstream)"
  - "openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md (build target + cell-count Scenario)"
  - "docs/p8pre/step1_prep.md (P1e PR-I baseline anchor)"
  - "docs/case_deployment_map.md §2.2 (heihe / heihe_x4 server deployment)"
  - "tools/p1e_2x2_sbatch_template.sbatch (prototype this template mirrors)"
---

# p8pre-spike Step 1 PR-A prep — evidence

## §1 cn14 Mode C build + nm gate verification

**Where**: server `frd_muziyao@210.77.77.22:32099`, compute node `cn14` (CPU partition,
idle at verification time), `srun --partition=CPU --nodelist=cn14 --cpus-per-task=4
--time=00:15:00`. Raw log: `.review-evidence/p8pre-pr-a-prep/cn14_build_evidence.log`.

**Build command** (matches spec n8-mode-c-profile-recheck Scenario "build target is
Mode C with profile instrumentation" L14-20):

```
cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD
make clean
make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1
```

**SHUD pin verification** (tasks.md §1.1): `git rev-parse --short HEAD` inside `SHUD/`
on cn14 returns `7a1dc8f` — matches expected P1e ship pin.

**Build exit code**: 0 (`./shud is compiled successfully!`).

**nm SHALL gates** (3-check set per Epic #339 brief + tasks.md §1.2):

| Gate | Symbol | SHALL | Observed | Verdict |
|---|---|---:|---:|---|
| 1a | `N_VNew_Serial` (Serial NVector linked) | ≥ 1 | 1 | PASS |
| 1b | `N_VNew_OpenMP` (OpenMP NVector NOT linked) | = 0 | 0 | PASS |
| 1c | `GOMP_parallel` (libgomp backing StrictOMP RHS pragma) | ≥ 1 | 1 | PASS |

Without gate 1c, the StrictOMP RHS pragma could silently compile out under a missing
`-fopenmp` link path; the linkage check proves Mode C is real and not a no-op.

## §2 Server case basin verification (heihe + heihe_x4)

**Where**: cn14 via the same `srun` channel. Raw log:
`.review-evidence/p8pre-pr-a-prep/cn14_case_evidence.log`. Source-of-truth for expected
sizes is `docs/case_deployment_map.md §2.2` (active 2026-06-26).

| Item | Path | SHALL | Observed | Verdict |
|---|---|---|---:|---|
| heihe forcing.trimmed (basin-local) | `.../SHUD/Basins/heihe/forcing.trimmed/` | 29M | 29M | PASS |
| heihe tsd.forc 2nd line | `.../heihe/input/heihe/heihe.tsd.forc` | points at `.../heihe/forcing.trimmed` | `/scratch/.../SHUD/Basins/heihe/forcing.trimmed` | PASS |
| heihe_x4 forcing/ subdir | `.../SHUD/Basins/heihe_x4/forcing/` | 286M (per map §2.2) | 286M | PASS |
| heihe_x4 basin total | `.../SHUD/Basins/heihe_x4/` | ≥ 200M (brief gate) | 2.3G | PASS |
| heihe_x4 tsd.forc | `.../heihe_x4/input/heihe_x4/heihe_x4.tsd.forc` | exists, 2nd line points at basin-local `forcing/` | second line = `/scratch/.../SHUD/Basins/heihe_x4/forcing` | PASS |
| heihe_x4 project name == basin folder | (per case_deployment_map §1) | no alias | confirmed (`input/heihe_x4/heihe_x4.cfg.para` discovered via `find ... -name '*.tsd.forc'`) | PASS |

Per CLAUDE.md "服务器" + case_deployment_map.md "heihe_x4 常驻... 禁止重生" rule, no
regeneration was attempted — the existing AutoSHUD pipeline 2026-06-17 artifact is intact.

## §3 Template + wrapper provenance

| File | Purpose | Status | Outer commit |
|---|---|---|---|
| `tools/p8pre/submit_n8_profile_template.sbatch` | Single-cell Slurm template, markers `__CASE__` / `__N__` / `__REP__` / `__NODE__`. Pinned Mode C build target, per-cell artifact dir, determinism env (`OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `OMP_NUM_THREADS=__N__` + `SHUD_RHS_THREADS=__N__`), provenance log header. | written (not yet staged) | `<staging>` |
| `tools/p8pre/render_n8_profile.sh` | Wrapper expanding 18 cells via awk substitution. Writes rendered `.sbatch` files into `/scratch/.../.p8pre-runs/rendered/` when invoked on server (where `/scratch` exists); on Mac dry-run prints only. Singleton chain per `(case, N)` via `--dependency=afterany:__PREV_JID_*__` placeholders for runner to resolve. POSIX bash + awk only. | written (not yet staged) | `<staging>` |
| `docs/p8pre/pr_a_prep_evidence.md` | This doc. | written (not yet staged) | `<staging>` |

**Working tree context**: outer branch `feat/issue-340-p8pre-pr-a-prep` (HEAD `c4f000d`
from `baseline/p8pre`), SHUD submodule pin `7a1dc8f`. No SHUD pointer bump in this
prep PR (per Step 1 invariance — pin only bumps in Step 2 PR-D).

**Style mirror**: both new files mirror `tools/p1e_2x2_sbatch_template.sbatch`
(p1e PR-B prototype) — leading comment block explains substitution markers, Slurm
三铁律 compliance, output path conventions. `__BUILD__` marker dropped (always
Mode C per Epic #338 §"What Changes"). `__NODE__` marker added (cn14 for heihe,
cn15 for heihe_x4) replacing the p1e prototype's hardcoded `cn05`.

## §4 18-cell render dry-run sanity

Run: `bash tools/p8pre/render_n8_profile.sh > .review-evidence/p8pre-pr-a-prep/render_18cell_sanity.txt`
on Mac (no `/scratch` → wrapper skips file write, prints sbatch lines + summary).
Full evidence: `.review-evidence/p8pre-pr-a-prep/render_18cell_sanity.txt`.

### §4.1 Coverage matrix

| case | N | rep | rendered filename | node pin | dependency |
|---|---:|---:|---|---|---|
| heihe | 1 | 1 | `submit_heihe_N1_rep1.sbatch` | cn14 | (none, first rep) |
| heihe | 1 | 2 | `submit_heihe_N1_rep2.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N1_rep1__` |
| heihe | 1 | 3 | `submit_heihe_N1_rep3.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N1_rep2__` |
| heihe | 4 | 1 | `submit_heihe_N4_rep1.sbatch` | cn14 | (none) |
| heihe | 4 | 2 | `submit_heihe_N4_rep2.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N4_rep1__` |
| heihe | 4 | 3 | `submit_heihe_N4_rep3.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N4_rep2__` |
| heihe | 8 | 1 | `submit_heihe_N8_rep1.sbatch` | cn14 | (none) |
| heihe | 8 | 2 | `submit_heihe_N8_rep2.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N8_rep1__` |
| heihe | 8 | 3 | `submit_heihe_N8_rep3.sbatch` | cn14 | afterany:`__PREV_JID_heihe_N8_rep2__` |
| heihe_x4 | 1 | 1 | `submit_heihe_x4_N1_rep1.sbatch` | cn15 | (none) |
| heihe_x4 | 1 | 2 | `submit_heihe_x4_N1_rep2.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N1_rep1__` |
| heihe_x4 | 1 | 3 | `submit_heihe_x4_N1_rep3.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N1_rep2__` |
| heihe_x4 | 4 | 1 | `submit_heihe_x4_N4_rep1.sbatch` | cn15 | (none) |
| heihe_x4 | 4 | 2 | `submit_heihe_x4_N4_rep2.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N4_rep1__` |
| heihe_x4 | 4 | 3 | `submit_heihe_x4_N4_rep3.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N4_rep2__` |
| heihe_x4 | 8 | 1 | `submit_heihe_x4_N8_rep1.sbatch` | cn15 | (none) |
| heihe_x4 | 8 | 2 | `submit_heihe_x4_N8_rep2.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N8_rep1__` |
| heihe_x4 | 8 | 3 | `submit_heihe_x4_N8_rep3.sbatch` | cn15 | afterany:`__PREV_JID_heihe_x4_N8_rep2__` |

### §4.2 SHALL summary

| Check | SHALL | Observed | Verdict |
|---|---:|---:|---|
| Total sbatch lines | 18 | 18 | PASS |
| N=2 cells | 0 | 0 | PASS (per design D4 — N=2 monotonic, P1e PR-I already archived) |
| heihe cells (NOT heihe_x4) | 9 | 9 | PASS |
| heihe_x4 cells | 9 | 9 | PASS |
| Unique (case, N) groups | 6 | 6 | PASS |
| First-rep cells (no `--dependency`) | 6 | 6 | PASS |
| Dependent-rep cells (`--dependency=afterany:...`) | 12 | 12 | PASS |
| Node pin: heihe → cn14 only | 9 | 9 | PASS |
| Node pin: heihe_x4 → cn15 only | 9 | 9 | PASS |

Node-pin spot-check via awk substitution:
- `heihe N=8 rep=1` → `#SBATCH --nodelist=cn14`
- `heihe_x4 N=8 rep=1` → `#SBATCH --nodelist=cn15`

## §5 Next-step handoff

- **#341 PR-A run scope**: actual sbatch submission. Runner SHALL `cd
  /scratch/frd_muziyao/SHUD-OpenMP` then `bash tools/p8pre/render_n8_profile.sh`
  (this WILL `mkdir -p .p8pre-runs/rendered/` and write 18 rendered files since
  `/scratch` exists on server), then pipe stdout through a JID-capture loop that
  substitutes `__PREV_JID_<case>_N<n>_rep<r-1>__` with the actual job-id returned
  by each `sbatch` invocation. Single-stream per (case, N): 6 first-rep
  `sbatch` calls fire immediately, 12 subsequent calls wait on `afterany` of
  the prior rep's JID.
- **Pre-flight on #341**: one-shot build outside sbatch on cn14 with the SAME
  build command verified in §1, so each cell does NOT pay a rebuild cost
  (template just `./shud __CASE__` — see template L78).
- **PR-B / PR-C downstream** (tasks.md §3-§4): aggregator + verdict + capstone
  consume 18-cell artifacts mirrored to `/tmp/p8pre_n8_profile/` per
  case_deployment_map convention.

---

Generated: 2026-06-26 by implementer subagent (Issue #339/#340).
