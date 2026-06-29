# Cross-review round 3: invariant + state-machine verification

**PR**: #384 / feat/issue-380-p8tune-klu-spike-pr-0
**Round 2 head**: 50d2a4b
**Round 3 head**: 7fc325b
**Reviewer scope**: 5 invariants (zero SHUD patch / state-machine consistency / advisory hygiene / 3-binary contract / fixture handling)

---

## Task 1: Zero SHUD src patch invariant — CONFIRMED

`git diff 50d2a4b..7fc325b -- SHUD` → empty.
SHUD submodule HEAD = `41d9a172610c1c628fcf4b1b0a4f7c19f4afc854` (unchanged from round 2). No upstream pointer bump, no SHUD-local commits between rounds.

## Task 2: State-machine consistency after G1 — CONFIRMED

Three OOM emission sites enumerated in `tools/p8tune.D/klu_analyze_factor.cpp`:

| Site | Line | reason= | Path-entry condition |
|------|------|---------|----------------------|
| RSS preflight (post-analyze) | L282 | `preflight_after_analyze` | `ordering_id==0 && Symbolic->lnz>0 && Symbolic->unz>0 && est > 0.7·CN_RAM` |
| KLU numeric factor failure | L306 | `klu_factor_OOM` | `Numeric==NULL && common.status==KLU_OUT_OF_MEMORY` |
| Post-factor RSS overrun | L317 | `post_factor_rss_exceeds_cn_ram` | `rss_post > CN_NODE_RAM_BYTES` |

State graph (round 3, AMD path):
`klu_analyze → preflight_after_analyze (exit-0 OR continue) → klu_factor → klu_factor_OOM (exit-0) OR continue → post_factor_rss check → exit-0 OR PASS`

State graph (round 3, COLAMD / natural path):
`klu_analyze → skip_log (printf "PREFLIGHT_AFTER_ANALYZE skipped ... AMD-only") → klu_factor → klu_factor_OOM (exit-0) OR continue → post_factor_rss check → exit-0 OR PASS`

Total reasons = 3, matches G2-updated README L166 enum `<preflight_after_analyze | klu_factor_OOM | post_factor_rss_exceeds_cn_ram>`. No new reasons; no dead reasons. State machine consistent.

Round 2 finding e01 root cause (lnz/unz = -1 sentinel cast to size_t = UB) addressed: round-3 evidence log `mac_smoke_keliya_klu_ordering_matrix.log` shows AMD lnz=16544, COLAMD/natural lnz=-1, and the skip-log path now correctly fires for non-AMD without dereferencing the sentinel into the arithmetic estimator.

## Task 3: G4 advisory logging hygiene — CONFIRMED

`grep -n "pre-flight" tools/p8tune.D/*` → no matches.
Legacy line `[klu] pre-flight: A nnz=... est_bytes=... cn_ram=...` (was L238-239 round 2) is **removed**.

PREFLIGHT_HINT line preserved at L236-237:
```
std::printf("[klu] PREFLIGHT_HINT pattern_est_bytes=%zu (advisory only; decisive check after klu_analyze)\n",
            est_pre_factor_hint_bytes);
```
Hint label and explanatory parenthetical kept verbatim. Single advisory remains; legacy duplicate gone.

## Task 4: 3-binary contract — CONFIRMED

`git diff 50d2a4b..7fc325b --name-only`:
- `tools/p8tune.D/README.md` (G2 reason rename)
- `tools/p8tune.D/klu_analyze_factor.cpp` (G1 + G4 inline fix)
- `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_ordering_matrix.log` (new G1 verification log)

`git diff 50d2a4b..7fc325b -- '*Makefile*'` → empty. No Makefile change.
`ls tools/p8tune.D/` → 3 binaries unchanged (`dump_adjacency`, `fd_color_jacobian`, `klu_analyze_factor`). No new binary surface.

## Task 5: Spec amendment work-tree-only handling — CONFIRMED non-blocking

`git check-ignore -v openspec/changes/p8tune-klu-spike/{tasks.md,design.md}` → both ignored by `.gitignore:13` rule `openspec/changes/`.
`git ls-files openspec/changes/p8tune-klu-spike/` → empty (no tracked files).
Working tree clean.

Convention: `openspec/changes/` is project-gitignored as work-only scratch (round 1 F6/F7/F8 followed same pattern and was approved by parent agent). G3 edits do not surface to downstream PR readers via git; fixture is consumed only by the local OpenSpec change author. Not a breaking invariant.

Single caveat (advisory only, not a finding): if downstream reviewers need the amended tasks.md / design.md for archival, they must be copied into the PR description body or into `.review-evidence/` (which IS tracked). The PR description / scratchpad currently lives outside this diff; no action required from this round unless project convention shifts.

---

## Findings: NONE NEW

Round 2 finding e01 (preflight UB on KLU_EMPTY sentinel) is **closed** by G1 — empirical evidence in the new keliya ordering matrix log shows the AMD-only guard correctly fires for `ordering=colamd` and `ordering=natural` with `lnz=-1 unz=-1`. The skip-log path replaces the spurious OOM/PASS verdict path.

Round 2 G2/G4 README + advisory hygiene cleanups verified.
Round 1 c02 stays closed (no regression).

---

## Verdict

**APPROVE** — All 5 invariants hold. State machine is consistent (3 reasons, no dead branches, AMD-only guard correct). Zero SHUD src patch. 3-binary contract intact. Openspec work-tree-only handling matches established project convention. No new findings.

