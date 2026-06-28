## Phase 4 Cross-Review Evidence Bundle (round 1, expanded fixture)

Reviewer agents: `review-correctness`, `review-spec-compliance`, `review-integration`, `review-documentation`
Review round: round 1
Reviewed head SHA: `2eb5d0f`
Local evidence: `.review-evidence/p8pre-pr-e-spike/{correctness,spec-compliance,integration,documentation}.md`

### review-correctness — APPROVE

Summary: PR-E spike tooling + evidence doc reconciles cleanly against Step 1 PR-A reference, fixture spec, and 18-cell raw data; identity 3-symbol nm gate-1 captured; orchestrator zone respected.

Findings: **None.**

Key verification:
- 3 script forks diff cleanly vs Step 1 PR-A; `run_identity_spike.sh` L107-131 adds Phase B.2 identity 3-symbol nm gate-1 (PSetupIdentity/PSolveIdentity/CVodeSetPreconditioner each ≥1, FAIL exit 2)
- Mac dry-run on `render_identity_spike.sh` outputs 18 sbatch lines + 1 summary; job-name + dependency chain + paths all correct
- `server_nm.log` 3 symbols verified (T/T/U), `server_build_provenance.log` gcc 13.3.0 + libsundials_cvode.so.6 + libgomp.so.1 complete
- §4 18-row table cross-validates field-by-field against `cell_stats.txt` + `jid_table.txt` (JID 9531..9548)
- Cross-N invariance §6 PASS: heihe (6599/6696/6/18163/77) × heihe_x4 (6569/6775/47/37695/158) identical across 9 cells per case
- ncfn §7 data correct (heihe=6, heihe_x4=47), neutral framing cites spec L74-79
- Soft gate 6 §8 ratio independently computed; 6+ orders of magnitude below 5% budget
- §5 wall median + delta 6 rows recomputed identically; max |delta| heihe=2.64% < 10%, heihe_x4=1.09% < 5%

Non-blocking note: §4 "All 18 jobs ExitCode 0" is prose claim, not table column; PR-F aggregator can add ExitCode column.

### review-spec-compliance — APPROVE

Summary: PR-E satisfies all 11 spec-compliance checklist items: gate-1 server nm 3-symbol evidence captured, all 18 cells emit nps/npe/ncfn/wall/t_precond_setup + rivqdown.dat, doc neutrally defers gate adjudication to PR-F #347, SHUD pointer correctly unchanged from PR-D at 5276167 (forward-only descendant), openspec strict PASS, Slurm 三铁律 paths all under /scratch.

Findings: **None.**

Non-blocking notes:
- Suggestion: §6 L115/L117 phrase "PASS within identity run" on cross-N invariance lines is legitimate intra-spike observation (not PR-F gate-4 which compares to PR-A baseline). YAML L9 `verdict_adjudication: PR-F` and §3/§5/§7 explicit deferrals disambiguate.
- Observation: ncfn = 6 (heihe) / 47 (heihe_x4) deterministic across N + reps. Per strict spec L74-79, PR-F gate-2 = FAIL → ADR-0003 NO-GO + design D8 fall-back PREC_NONE. PR-E correctly captures + analytically frames; heads-up for PR-F #347 reviewer.

### review-integration — APPROVE

Summary: PR-E is data-only, all 10 integration contracts pass; downstream PR-F #347 + PR-G #348 have all required inputs.

Findings:
- Suggestion (out-of-scope, non-blocking): `.review-evidence/` not in outer `.gitignore` — currently untracked but unprotected from accidental `git add -A`. Suggest adding near `.p8pre-runs/` block in PR-F #347 since PR-F will write more rounds. Not blocking PR-E.

Non-blocking notes:
- identity_spike_run.md §5 median computation method correct; PR-F should explicitly cite median sort algorithm in verdict doc
- heihe_x4 N=1 rep3 wall (1273.80s) 14% faster than rep1 (1491.05s) consistent with cold→warm cache convergence in afterany-chained singleton reps; not an integration concern

### review-documentation — REQUEST CHANGES → RESOLVED Phase 6

Summary: identity_spike_run.md is a well-structured neutral data-capture run log; style matches Step 1 PR-A reference and stays in PR-E scope.

Findings:
- **Critical**: outer_pin in YAML frontmatter L6 + §2 L30-31 referenced `f800bb2` (Part 1 tool-scripts commit) instead of doc-introducing commit. Downstream PR-F/PR-G ADR-0003 cite outer_pin as provenance anchor — stale SHA produces logically impossible self-reference.
- Suggestion (non-blocking): §4 raw table omits Slurm Elapsed + ExitCode columns vs Step 1 PR-A doc convention. PR-F aggregator output can add this.

Non-blocking notes:
- Style match confirmed (no Abstract / H1-H3 / Methodology sections; lightweight execution log convention)
- Neutral data capture confirmed: gates 2/3/4/5/6 framed as "data captured, PR-F adjudicates"; gate 1 correctly called CAPTURED per spec L66-72
- Spec cross-refs verified accurate (L74-79 gate 2, L92-100 gate 4, L102-106 soft gate 5, L108-113 soft gate 6); SHUD pin 5276167 cited at L5+L29+L189
- PREC_NONE→PREC_LEFT shift framing (§6 L119-136) explains CVLS path divergence + hands SHA12 verdict to PR-F max_ulp fallback — exactly what PR-F needs

### Phase 4.5 Verifier — SKIPPED on Critical (self-evident fact)

Rationale: Critical claim "outer_pin SHA does not match HEAD" verifiable by `git rev-parse HEAD` + `grep outer_pin docs/p8pre/identity_spike_run.md` directly. Verifier subagent adjudication on plain-fact reading adds 0 information value.

### Phase 6 orchestrator-direct fix @ `291a0a4`

- L6: `outer_pin: f800bb2...` → `2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6` (doc-introducing commit per Step 1 PR-A convention)
- L30: same update
- 0 stale `f800bb2` references remain in `tools/p8pre/` or `docs/p8pre/` after fix

Suggestion (§4 Elapsed + ExitCode columns) carried as non-blocking forward note for PR-F #347 aggregator output.

### Phase 6.5 SKIPPED

Rationale: Phase 7 Gap Sweep will independently examine `docs/p8pre/identity_spike_run.md` L1-35 for the same lines + verify 0 stale SHA refs in the file tree.

### Round 1 verdict

**Clean after Phase 6.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE at `291a0a4`. Proceed to Phase 7.

### Forward action notes (carry to downstream PRs)

- PR-F #347: `aggregate_identity_spike.sh` output should include per-cell Slurm Elapsed + ExitCode columns for ADR-0003 audit traceability
- PR-F #347: add `.review-evidence/` to outer `.gitignore` near `.p8pre-runs/` block
- PR-F verdict doc: cite explicit median sort algorithm
