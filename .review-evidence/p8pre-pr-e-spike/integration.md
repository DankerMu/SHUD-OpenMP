# PR #358 — Integration review (round 1)

Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 2eb5d0f
Branch: feat/issue-346-p8pre-pr-e-server-spike
Scope: 3 tool scripts + 1 doc; no SHUD source change (PR-E data-only spike).

## Summary

PR-E is data-only and conforms to the contracts that PR-F #347 (aggregator/verdict) and PR-G #348 (ADR-0003 NO-GO) downstream-consume. All 10 integration checks pass except one minor evidence-archive gitignore gap.

## Integration checks

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | PR-F readiness — 18 cells × 3 file types in `/tmp/p8pre_identity_spike/` | PASS (Y) | `find … -name profile_B0.yaml \| wc -l = 18`; same for `cvode_stats.txt` and `*.rivqdown.dat` |
| 1 | PR-F can read jid_table / server_nm.log / server_build_provenance.log | PASS (Y) | All three files present in mirror; jid_table.txt has 18 rows JID 9531-9548 |
| 1 | identity_spike_run.md §4 table reproducible from cell_stats.txt + jid_table.txt | PASS (Y) | cell_stats.txt schema (10 cols) joins jid_table.txt (4 cols) on (case, N, rep); §4 table matches |
| 2 | ADR-0003 citable data for §7 ncfn observation | PASS (Y) | identity_spike_run.md §7 documents heihe ncfn=6, heihe_x4 ncfn=47, both deterministic — direct NO-GO citation |
| 2 | ADR-0003 citable data for §6 cross-N invariance + PREC_NONE→PREC_LEFT nst/nfe shift | PASS (Y) | §6 documents nst/nfe shift table (Step1 vs Step2) — supports D8 fall-back rationale |
| 3 | PR-E does NOT touch cvode_config.cpp / MD_precond_identity.{h,cpp} | PASS (Y) | `grep -rn 'MD_precond_identity\|cvode_config.cpp' tools/p8pre/ docs/p8pre/identity_spike_run.md` returns no matches; PR-D-impl files untouched, leaving D8 revert path clean for PR-G |
| 4 | Server-dir subdirectory layout disjoint from PR-A `.p8pre-runs/heihe_*` | PASS (Y) | PR-E writes under `.p8pre-runs/identity_spike/<case>_N<n>_rep<r>/`; PR-A wrote under `.p8pre-runs/<case>_N<n>_rep<r>/`. Sub-namespace separation is clean |
| 4 | `rendered/` subdirs disjoint (`.p8pre-runs/rendered/` vs `.p8pre-runs/identity_spike/rendered/`) | PASS (Y) | render_identity_spike.sh:53 hardcodes `.p8pre-runs/identity_spike/rendered`; render_n8_profile.sh hardcodes `.p8pre-runs/rendered` |
| 5 | Mac dry-run gracefully skips /scratch write | PASS (Y) | render_identity_spike.sh L77 `if [ -d "$(dirname "$OUT_DIR")" ]` guards mkdir + WRITE_RENDERED; Mac has no `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/` so guard yields WRITE_RENDERED=0 cleanly |
| 6 | PR-E cell-data structured same way as PR-A | PASS (Y) | cell_stats.txt schema header `case N rep nst nfe ncfn nps npe wall_total t_precond_setup` is PR-A 8-column + 2 extras; aggregator reads per-cell YAML+stats directly (not cell_stats.txt), so identical per-cell file layout is the binding contract — confirmed |
| 7 | `.p8pre-runs/` + `.p8pre-pr-*-runs/` in outer .gitignore | PASS (Y) | .gitignore L33-34 confirms patterns |
| 7 | `.review-evidence/` in outer .gitignore | **FAIL (N)** | `git check-ignore` returns empty; .gitignore grep finds no `review-evidence` entry. Currently untracked (status `?? .review-evidence/`) but not protected from accidental `git add -A` — see Finding 1 below |
| 7 | No untracked files leak into PR diff | PASS (Y) | `git diff --name-only main..HEAD` shows only intended files |
| 8 | SHUD pointer unchanged at 5276167 | PASS (Y) | `git submodule status SHUD` = `5276167eea67184d801905f54dc805d2cd61db2d` (matches PR-D #357 HEAD) |
| 9 | CI compat — no workflow references identity_spike scripts or docs/p8pre/ | PASS (Y) | `grep -r 'run_identity_spike\|docs/p8pre' .github/workflows/` returns nothing |
| 10 | review-loop-log.jsonl tail is at #357 finalize state | PASS (Y) | `tail -1 docs/review-loop-log.jsonl` → `pr=357` (Phase 8 #358 entry not yet appended, expected) |

## Findings

### Finding 1 — Suggestion: `.review-evidence/` not gitignored

`.review-evidence/p8pre-pr-e-spike/cell_stats.txt` and downstream rounds' reports (this file included) are untracked but the directory is not in `.gitignore`. A future contributor running `git add -A` will commit per-round review evidence. This is a project-wide pattern (P1e and earlier rounds use the same dir) and should be added to `.gitignore` consistently. Suggested addition near `.p8pre-runs/` block:

```
# Review-evidence scratch (per-round reviewer outputs; not source of record)
.review-evidence/
```

Out of scope for PR-E (this is data-capture only), but worth flagging now so PR-F #347 (which will also write `.review-evidence/p8pre-pr-f-*/`) can include the fix.

## Non-blocking notes

- The `delta_pct` table in identity_spike_run.md §5 uses median(rep1,rep2,rep3) where rep1 of heihe N=1 (137.08s) and rep3 (121.62s) span a ~13% range. Median (137.27s) is a defensible choice but PR-F should explicitly cite the median sort algorithm in the verdict doc to avoid downstream interpretation drift. (Informational; PR-F owns gate-4 verdict.)
- heihe_x4 N=1 rep3 (1273.80s) is 14% faster than rep1 (1491.05s) — likely cold-cache rep1 vs warm-cache rep3. The afterany singleton chain in render_identity_spike.sh:124-130 means reps run serially, so warm-cache convergence is expected. Not an integration concern.

## Verdict

APPROVE — PR-E is a clean data-only capture: downstream consumers (PR-F aggregator at #347, PR-G ADR-0003 at #348) have all required inputs (18 cells × 3 file types + jid_table + nm log + provenance + cell_stats), namespace isolation from PR-A artifacts is enforced, SHUD pointer is forward-only unchanged, and PR-D-impl revert path is preserved. One out-of-scope suggestion (`.review-evidence/` gitignore) does not block merge.
