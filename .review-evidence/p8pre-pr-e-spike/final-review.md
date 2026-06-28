Reviewer agent: phase-7-final-review
Review round: final
Reviewed head SHA: 291a0a4

Summary: Gap sweep clean — Phase 6 outer_pin fix verified at L6+L30, all 5 CI checks SUCCESS, 18×3 file bundle + provenance complete, no Critical/Warning gaps found beyond Phase 4.

Gap Sweep findings (NOT already in Phase 4):
- None.

Phase 6 fix verification:
- L6 outer_pin: correct (`2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6`)
- L30 Outer pin: correct (`2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6`)
- Stale `f800bb` references in doc: 0 (also 0 across `tools/p8pre/` + `docs/p8pre/`)

Completion self-audit (11 ACs from PR body @ 291a0a4):

| AC | Verdict |
|---|---|
| 3 new tool scripts + Mac dry-run 18-cell render | PASS (3 files in `tools/p8pre/`; doc §1-2 confirms render) |
| Server build `make shud OMP_RHS=1 PROFILE=1` exit 0 | PASS (server_build_provenance.log present) |
| Server 3-symbol nm gate-1 evidence captured | PASS (server_nm.log; doc §3 cites 3 symbols) |
| Mode C nm gate maintained | PASS (cited in doc §3 + build provenance) |
| 18-cell Slurm submit (JID 9531..9548) | PASS (doc §4 + jid_table.txt) |
| 18/18 jobs COMPLETED ExitCode 0:0 | PASS (doc §4 declarative; cell_stats.txt has 18 data rows, 0 MISSING_FILE) |
| rsync server → Mac `/tmp/p8pre_identity_spike/` 18 cells | PASS (18 cell dirs + bundle files present) |
| Per-cell cvode_stats: nps>0+npe>0 全 18 | PASS (heihe nps=18163/npe=77, heihe_x4 nps=37695/npe=158 across 18 rows) |
| 18-cell t_precond_setup emit到 profile_B0.yaml extras | PASS (col 10 populated 18/18, no zeros/blanks) |
| Cross-N CVODE invariance within identity run | PASS (doc §6: heihe identical; heihe_x4 identical N=1/4/8 × 3 rep) |
| identity_spike_run.md NEUTRAL data capture | PASS (doc §5/§7 explicitly defer verdict to PR-F #347) |

Oracle integrity: PASS (`git diff baseline/p8pre...291a0a4` = 4 files: 3 tool scripts + 1 doc; 0 changes under `openspec/`, 0 to `.gitmodules`, 0 to tests/CI/SHUD submodule pointer — still `5276167`; SHUD upstream `openmp-baseline-p8pre` HEAD = `5276167` unchanged; C8 compliance: no master pollution)

CI status: PASS (5/5 SUCCESS: setup, tools-tests, build-and-compare keliya, asan-ubsan keliya, asan-ubsan qhh; doc+bash-only diff so no CI surface affected)

Forward action notes (not merge-blocking):
- PR-F #347 aggregator output SHOULD include Elapsed + ExitCode columns (Phase 4 review-documentation Suggestion carried forward).
- `.review-evidence/p8pre-pr-e-spike/` is currently NOT in `.gitignore` (Phase 4 review-integration Suggestion); evidence files are staged in working tree but not committed — recommend gitignore add before next PR cycle to prevent accidental inclusion.
- PR-G #348 ADR-0003 cite-readiness CONFIRMED: doc §7 (ncfn=6/47 deterministic data) + §6 (cross-N invariance + PREC_NONE→PREC_LEFT shift table heihe nst 6698→6599, nfe 6943→6696; heihe_x4 nst 6575→6569, nfe 6741→6775) provide direct citation anchors for adjudication and design D8 fall-back rationale.
- PR body AC table heading "PASS @ head `2eb5d0f`" lags by one fix-up commit (HEAD is 291a0a4); cosmetic — Phase 8 may amend in body refresh if desired, but `2eb5d0f` is the documentary commit and AC content remains valid since 291a0a4 only corrected the outer_pin metadata.

Final-review verdict: Clean
