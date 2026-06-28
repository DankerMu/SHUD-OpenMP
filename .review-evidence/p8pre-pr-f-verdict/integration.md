# review-integration — PR #359 (p8pre PR-F verdict NO-GO) — round 1

Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 75a757c
Summary: PR-F integration READY for PR-G #348 consumption — all 12 checklist items pass; ZERO blocking findings; one forward-carry note on `.gitignore` patch + one forward-carry note on `tools/compare_snapshot/` raw-double hardening.

## Checklist verification

1. **PR-G ADR-0003 quotability** — verdict doc §4 hard gate table (L113-118) cleanly enumerates 4 gates with PASS/FAIL + evidence column; §5 soft gate table (L141-144) enumerates 2 gates with FAIL/PASS + max_ulp ≈ 9×10¹⁵ values + structural-divergence rationale (L146 "5,155 of 214,252 positions"); §7 NO-GO (L179) is explicit and unambiguous.

2. **Design D8 revert tail** — §7 L184-191 lists all 7 PR-G actions with explicit file paths: `docs/adr/0003-precond-spike-decision.md` (write), `cvode_config.cpp:259` (revert SUN_PREC_LEFT → SUN_PREC_NONE), `MD_precond_identity.{h,cpp}` (delete + Makefile unlink), `t_precond_setup` Timer bucket (delete or carve), `baseline/p8pre` (close), master plan §P8-precond.0 (cross-ref update), optional `docs/p8pre/p8pre_summary.md`.

3. **Master plan §P8-precond.0 quotable summary** — §7 L190 explicitly directs cross-ref + ROI ceiling + NO-GO outcome + re-estimate effort; §6 L168-169 quantifies `ncfn=6 / ncfn=47` floor + `nfeLS/nfe = 1.811` ratio for ROI ceiling docs.

4. **PR-D jok-mirror forward carry** — verdict doc §2.1 L58 cites `cvDiurnal_kry.c` L716 (Precond) + L760 (PSolve); §10 ref [3] re-cites; §7 L194 explicitly forwards spec L26 wording correction to PR-G + #349.

5. **PR-E forward carries** —
   - Slurm Elapsed + ExitCode columns: §3.6 Table 1 (L84) has both columns for all 18 cells.
   - `.gitignore`: `git diff baseline/p8pre...75a757c -- .gitignore` = empty; outer `.gitignore` L34 has `.p8pre-runs/` but **NOT** `.review-evidence/`. PR-F did not patch; must forward to PR-G. NB: `.review-evidence/` is currently untracked (caught by `git status`), so the omission is non-blocking but PR-G must add it before any commits accidentally include the directory.
   - median sort algorithm: aggregator L169 + L171-174 `median3()` documents middle-of-3-sorted via `sort -n | sed -n '2p'`; verdict doc §3.4 L76 mirrors policy.

6. **compare_snapshot fallback logic** — aggregator L20 (comment) + L237 (`np.spacing` numpy implementation) + verdict doc §3.5 L78 (spec L102-106 invocation) + §8.5 L210 (format-gap limitation). Fallback path correctly handles compare_snapshot format mismatch. Forward note for PR-G or future epic: `tools/compare_snapshot/compare_snapshot` should be hardened to handle raw-double format (no magic header) as a P9+ tools-tech-debt epic.

7. **Outer .gitignore compliance** — `.p8pre-runs/` is in `.gitignore` L34 (verified); `/tmp/p8pre_identity_spike/aggregate_verdict.txt` exists at `/tmp` and is NOT in `git ls-files` (verified non-committed); `.review-evidence/` is untracked but `.gitignore`-omitted (PR-F forward-carry; see item 5).

8. **SHUD upstream state** — `git submodule status SHUD` = `5276167...` (unchanged from PR-D #357 / PR-E #358 baseline); pointer clean.

9. **CI compat** — Aggregator is bash + `uv run --quiet --with numpy python` (no system pip / no PYTHONPATH leakage); no CI workflow runs it; verdict doc is a `docs/p8pre/*.md` doc with no CI gate. Compat-safe.

10. **Diff scope sanity** — Exactly 2 expected files (`docs/p8pre/identity_spike_verdict.md` +250, `tools/p8pre/aggregate_identity_spike.sh` +569); single commit `75a757c`; no other commits in PR-F branch since `baseline/p8pre`.

11. **openspec validate p8pre-spike --strict --no-interactive** = `Change 'p8pre-spike' is valid` (exit 0 verified).

12. **review-loop-log.jsonl untouched** — `git diff baseline/p8pre...75a757c -- review-loop-log.jsonl` = empty (verified).

## Findings

- **None.** Zero blocking integration findings. PR-F is ready for merge from the integration-readiness perspective.

## Non-blocking notes

- **N1 (forward-carry to PR-G)**: `.gitignore` does not include `.review-evidence/`. Currently `.review-evidence/` is untracked locally (PR-F did not commit anything there), but PR-G or later phases will likely commit evidence artifacts. PR-G should add `.review-evidence/` to `.gitignore` alongside `.p8pre-runs/` (L34) when adjacent edits are made, OR PR-G should commit the evidence intentionally as part of its ADR-0003 audit trail.

- **N2 (forward-carry to post-P8 tools epic)**: `tools/compare_snapshot/compare_snapshot` only consumes SHUD-magic-headered binary snapshot format. The aggregator at L20 + L237 documents the workaround (inline numpy `np.spacing`), and the verdict doc §8.5 L210 notes this as a documented limitation. A future tools-tech-debt epic should harden `compare_snapshot` to accept raw-double `rivqdown.dat`-style outputs to remove the inline-numpy dependency from acceptance-testing scripts.

- **N3 (positive observation)**: Aggregator median policy (L169-174) cleanly documents middle-of-3-sorted algorithm and is mirrored in verdict doc §3.4 L76. PR-E forward-carry item satisfied without ambiguity.

- **N4 (positive observation)**: Verdict doc §7 L184-191 PR-G action list is exceptionally well-structured — 7 numbered actions with explicit file paths + line numbers. Downstream consumption by PR-G #348 will be friction-free.

- **N5 (positive observation)**: Verdict doc §6.2 L160-169 ROI quantification table (nst/nfe/nfeLS/nps/npe/ncfn/ncfl breakdown) is quotable verbatim into master plan §P8-precond.0 + ADR-0003; provides clean operational ROI ceiling for any future P8-precond candidate.

## Verdict

**APPROVE** — PR-F integration-ready; all 12 checklist items satisfied; downstream consumption by PR-G #348 (ADR-0003 + master plan §P8-precond.0 + design D8 revert tail + p8pre_summary.md) will be friction-free. The 2 non-blocking forward-carry notes are standard project hygiene (gitignore + tools hardening) and do not block merge.
