# Correctness Review — PR #350 round 1

**Reviewer agent**: review-correctness
**Reviewed head SHA**: 5827e237c86150dd4a16e8f68eeffc211f66c46f
**Scope**: docs/p1e/p1e_perf_baseline.md (4 line edits) + docs/p8pre/step1_prep.md (new, 74 lines)

## Summary

Diff is faithful to its narrow scope: 15-key list rewritten verbatim from `tools/cvode_stats_diff/canonical_15_keys.yaml`, three `.pr-i-runs/` → `.p1e-i-runs/` path replacements applied at L83/156/157/183, no collateral edits to neighboring `.pr-d-runs/` / `.pr-j-runs/` / `.pr-c-runs/` strings, and all inline-quoted numbers in `step1_prep.md` match the cited source-of-truth doc rows. No CONFIRMED correctness bug found.

## Evidence audit (issue 339 AC checklist)

| Check | Result | Evidence |
|---|---|---|
| 15-key list at p1e §3.5 L83 == yaml L13-27 character-by-character + same order | PASS | L83 prints `nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS` — identical to yaml |
| 4 occurrences of `.pr-i-runs/` → `.p1e-i-runs/` (per pre-edit grep L83/156/157/183) | PASS — diff hunks confirm exactly 4 changes; post-edit grep returns 4 hits at the same line numbers | `git diff` + `grep -n` |
| Sibling paths `.pr-d-runs/` (L184), `.pr-j-runs/` (L185), `.pr-c-runs/` (L186) untouched | PASS | post-edit grep |
| `grep -c "nlcf" docs/p1e/p1e_perf_baseline.md` = 0 | PASS | grep returns 0 |
| canonical_15_keys.yaml untouched | PASS | `git diff` exits 0 with empty output |
| step1_prep.md §2 wall median: heihe (504/511/488/473), heihe_x4 (1340/1051/850/775) match p1e §3.1 L43-44 | PASS | L41-44 in p1e source identical |
| step1_prep.md §3 nst ladder: heihe 6698×5 / heihe_x4 6575×5 + max\|Δ\|=0 match p1e §3.4 L74-75 | PASS | source rows identical |
| step1_prep.md §4 nfe baselines (heihe=6943, heihe_x4=6741) match `p1e_pr_i_strict_omp_verification.md` §3.1 | PASS | cell roster L107-128 confirms both values |
| AC oracle integrity — no test/spec weakened | PASS — no tests in diff; tasks.md §0 unchanged; issue body AC mirror tasks.md §0.2 line-for-line | manual diff |
| `grep -c "p1e-i-runs" docs/p1e/p1e_pr_i_strict_omp_verification.md ≥ 8` | PASS (= 8) | grep |

## Findings

### Non-blocking observation 1 — issue §0.2 third bullet is satisfied but not visible in this diff

`openspec/changes/p8pre-spike/tasks.md` §0.2 third bullet and issue #339 In-Scope item 4 both mandate `fix docs/p2a/p2a_profile_baseline.md §9 v0.5 row: remove any nlcf typo`. The PR diff touches zero p2a files. Verified `grep -c "nlcf" docs/p2a/p2a_profile_baseline.md` = 0 on the head SHA — AC is in fact met (likely cleaned in prior commit `386da4d` GPT Pro fact-check). The AC "grep = 0" is the only acceptance condition for that bullet and it holds, so this is not a CONFIRMED regression. Flag for verifier: confirm the p2a `nlcf` was historically present and was removed pre-PR, so the PR description should mention "p2a already clean at fork point" to avoid future reviewers re-opening the task. No code/contract risk.

### No correctness/security/contract findings

Reviewer's recall-biased scan surfaced no nameable failure scenario. The diff is a strict-replacement edit set: every replaced token has a 1:1 canonical authority (yaml line / source-of-truth doc row), and the new file injects only data already in the verification doc. There is no logic, no parser, no test oracle, and no production code path touched. The replacement preserves grammatical/semantic intent of the original §3.5 sentence (the prefix "SHUD canonical 15-key (per ...yaml)" tightens provenance vs. the unsourced original list).

## Verdict

APPROVE — diff matches issue 339 In-Scope + AC items 1, 2, 3, 5 (p1e doc edits) and the spirit of item 4 (p2a `nlcf`=0 already holds). All inline-quoted numbers verified against canonical sources. No tests weakened. canonical_15_keys.yaml untouched.

---

```
Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 5827e237c86150dd4a16e8f68eeffc211f66c46f
Summary: Diff faithfully replaces 15-key list (canonical_15_keys.yaml verbatim) + 4 path strings (.pr-i-runs/→.p1e-i-runs/ at L83/156/157/183); new step1_prep.md inline-quoted wall/nst/nfe baselines exactly match source-of-truth rows; no collateral damage, no test/spec weakening.
Findings:
- None.
Non-blocking notes:
- Issue 339 In-Scope item 4 (p2a §9 v0.5 nlcf typo removal) is not visible in this diff but is already satisfied at head SHA (grep -c nlcf docs/p2a/p2a_profile_baseline.md = 0; likely cleaned by prior commit 386da4d). PR description could note "p2a already clean at fork point" for traceability. No correctness risk.
```
