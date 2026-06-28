# Reviewer agent: review-spec-compliance
- Review round: round 1
- Reviewed head SHA: 2eb5d0f
- Summary: PR-E captures all 18-cell PR-F-input data (gate-1 nm, ncfn, nps/npe, wall, t_precond_setup, rivqdown) per spec L74-130 with correct PR-F-deferral framing; openspec strict PASS; SHUD pointer correctly unchanged from PR-D (5276167) and forward-only descendant of 7a1dc8f. One non-blocking phrasing nit in doc §6.

## Spec-compliance checklist verification

| # | Spec line | Status | Evidence |
|---|---|---|---|
| 1 | tasks.md §7.0–§7.4 (PR-E SHALL scope) | PASS | server_build_step4.log L65 `make exit=0`; L71 3-symbol nm OK; submit_step5.log shows 18 sbatch + jid_table; rsync_step1.log + cell_stats.txt confirm 18 cells |
| 2 | spec L60-67 gate-1 build + 3-symbol nm | PASS | /tmp/p8pre_identity_spike/server_nm.log: `T PSetupIdentity`, `T PSolveIdentity`, `U CVodeSetPreconditioner`; server_build_step4.log L82 ELF shud rebuilt; doc §3 archives the 3 lines verbatim |
| 3 | spec L74-79 gate-2 ncfn data captured | PASS (verdict deferred to PR-F) | cell_stats.txt all 18 rows: heihe `ncfn=6`, heihe_x4 `ncfn=47` (non-zero but deterministic); doc §7 L139-160 neutrally captures + explicitly defers strict adjudication to PR-F (L159-160) |
| 4 | spec L80-86 gate-3 nps>0 AND npe>0 | PASS (data captured for PR-F) | cell_stats.txt all 18 rows: heihe nps=18163 / npe=77; heihe_x4 nps=37695 / npe=158; doc §4 Tab.1 |
| 5 | spec L92-100 gate-4 wall data captured | PASS (verdict deferred to PR-F) | cell_stats.txt wall_total column 18 rows; doc §5 Tab cites Step 1 PR-A baseline + computes |delta_pct| as observational only; explicitly says "PR-F #347 will compute the formal verdict" (L106) |
| 6 | spec L102-106 soft gate 5 rivqdown.dat present | PASS | 18 cell dirs each contain `<case>.rivqdown.dat`; sample sizes heihe N1_rep1 = 1.71 MB, heihe_x4 N8_rep3 = 3.10 MB; SHA12 + max_ulp computation correctly deferred to PR-F |
| 7 | spec L108-113 soft gate 6 t_precond_setup emit | PASS | `grep -c "t_precond_setup" *.yaml` = 1 for all 18 yaml files (catch-all auto-emit per §6.0a hit); cell_stats.txt t_precond_setup column all 18 values > 0 (8.255e-6 .. 1.4535e-5 s) |
| 8 | spec L116-130 doc not encroaching on PR-F verdict | PASS (with nit) | identity_spike_run.md does NOT have a "verdict" or "recommendation" section; YAML L9 says `verdict_adjudication: PR-F SHUD-OpenMP#347 (not in scope here)`; uses "neutral data + provenance" framing (L18); the strings "PASS" appearing at L50/115/117 are explicitly framed either as "evidence CAPTURED" (L50, PR-F input) or "PASS within identity run" (L115/117, scoped strictly to intra-spike cross-N invariance, not vs PR-A baseline which is PR-F's gate-4) — borderline; see Suggestion #1 |
| 9 | spec L148-154 Step 2 forward-only descendant + SHUD ptr invariance vs PR-D | PASS | git diff main..HEAD: SHUD bump 7a1dc8f → 5276167 (PR-D inheritance); HEAD~1 SHUD already 5276167 (PR-E does NOT re-bump); upstream `refs/heads/openmp-baseline-p8pre` HEAD = 5276167; `git merge-base 5276167 7a1dc8f` returns `7a1dc8f6...` exactly (linear descendant ✓) |
| 10 | openspec validate p8pre-spike --strict --no-interactive | PASS | exit 0; stdout `Change 'p8pre-spike' is valid` |
| 11 | Slurm 三铁律 | PASS | run_identity_spike.sh L65 `cd "$REPO"` (= /scratch/.../SHUD-OpenMP); template L48-49 #SBATCH --output/--error pinned /scratch; template L57 REPO=/scratch + L58 SHUD_DIR=$REPO/SHUD + L60 CELL_DIR under /scratch (no /tmp refs) |

## Findings

None.

## Non-blocking notes

- Suggestion 1 (`docs/p8pre/identity_spike_run.md:50,115,117`): the strings "PASS" appear three times. L50 is acceptable ("evidence CAPTURED" disambiguates from PR-F's "gate PASS verdict"). L115/L117 say "PASS within identity run" for cross-N nst/nfe/ncfn/nps/npe invariance. This is a legitimate intra-spike observation (not a PR-F gate, since PR-F gate-4 compares to PR-A baseline, not intra-spike), but a reader skimming may misread it as PR-F-style verdict. Consider tightening to "observed invariant" / "consistent" to fully eliminate ambiguity. Not blocking — the YAML header `verdict_adjudication: PR-F SHUD-OpenMP#347 (not in scope here)` (L9) and §3/§5/§7 explicit deferrals make the scope unambiguous on second read.
- Observation: ncfn = 6 (heihe) / 47 (heihe_x4) is non-zero. Per strict spec L74-79 reading, PR-F gate-2 = FAIL. PR-E correctly captures + frames neutrally (doc §7 L153-158 explains the analytical prior). Not PR-E's concern; flagging only as a heads-up for PR-F #347 reviewer.
