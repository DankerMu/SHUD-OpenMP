## Summary

p8pre-spike Step 2 PR-E run capstone. Server-side 18-cell identity-precond spike at SHUD pin `5276167` (PR-D wire-up from #357). Captures gate-1 evidence (server build + 3-symbol nm) + 18-cell raw stats (nst/nfe/ncfn/nps/npe/wall/t_precond_setup) + cross-N CVODE invariance + neutral data observation. Closes #346, refs #338.

### Deliverables

| 文件 | 用途 |
|---|---|
| `tools/p8pre/submit_identity_spike_template.sbatch` (NEW) | Fork of Step 1 PR-A template with `.p8pre-runs/identity_spike/` subdir + `p8pre_identity_` job-name prefix |
| `tools/p8pre/render_identity_spike.sh` (NEW) | 18-cell render wrapper matching identity spike subdir + prefix |
| `tools/p8pre/run_identity_spike.sh` (NEW) | Server-side runner with Phase B.2 identity 3-symbol nm gate-1 evidence capture |
| `docs/p8pre/identity_spike_run.md` (NEW, 191 行) | Lightweight execution log: §1 Purpose + §2 Build provenance + §3 Gate-1 nm evidence + §4 18-cell raw table + §5 wall-vs-baseline observation + §6 Cross-N invariance + PREC_NONE→PREC_LEFT shift + §7 ncfn observation + §8 Soft gate 6 ratio + §9 References |

### Acceptance Criteria (PASS @ head `291a0a4`)

| AC | 实测 |
|---|---|
| 3 new tool scripts created + Mac dry-run 18-cell render | ✓ |
| Server build `make shud OMP_RHS=1 PROFILE=1` exit 0 | ✓ gcc 13.3.0, no errors (1 unrelated -Wformat-truncation warning) |
| Server 3-symbol nm gate-1 evidence captured | ✓ PSetupIdentity / PSolveIdentity / CVodeSetPreconditioner 全 ≥1 |
| Mode C nm gate maintained (N_VNew_Serial≥1 / OMP=0 / GOMP≥1) | ✓ |
| 18-cell Slurm submit success | ✓ JID 9531..9548 contiguous, singleton afterany chain per (case, N) |
| 18/18 jobs COMPLETED ExitCode 0:0 | ✓ 84 min critical path |
| rsync server → Mac `/tmp/p8pre_identity_spike/` 18 cells | ✓ 47.78 MB transferred |
| Per-cell cvode_stats: nps>0 + npe>0 全 18 | ✓ heihe nps=18163/npe=77, heihe_x4 nps=37695/npe=158 |
| 18-cell t_precond_setup emit到 profile_B0.yaml extras: | ✓ |
| Cross-N CVODE invariance within identity run | ✓ heihe (nst=6599, nfe=6696) identical N=1/4/8; heihe_x4 (nst=6569, nfe=6775) identical N=1/4/8 |
| identity_spike_run.md NEUTRAL data capture | ✓ Phase 6 outer_pin SHA fix |

### Key data observation (gate adjudication by PR-F #347)

**Gate 2 ncfn observation (data only — PR-F adjudicates per spec L74-79)**:
- heihe: ncfn=6 per cell (deterministic 9/9)
- heihe_x4: ncfn=47 per cell (deterministic 9/9)

Per spec `p8precond-zero-identity-spike` L74-79: gate 2 criterion = `ncfn=0` strict. Identity preconditioner offers no help to stiff Jacobian linear solver (P^-1 = I → SPGMR effectively unpreconditioned), propagating to CVODE nonlinear Newton retries. **PR-F #347 will adjudicate gate 2 verdict from this data per spec → expected NO-GO + design D8 PREC_NONE fall-back.**

**Preliminary wall ratios (PR-F adjudicates gate 4 per spec L92-100)**:
- Delta range: −2.64% to +1.09% vs Step 1 PR-A baseline
- All 6 (case, N) groups within ε(heihe)=0.10 and ε(heihe_x4)=0.05 thresholds

**Soft gate 6 setup overhead (PR-F adjudicates per spec L108-113)**:
- `t_precond_setup / t_wall_total` ratio: 6+ orders of magnitude below 5% threshold across all 18 cells

**PREC_NONE → PREC_LEFT CVODE shift**:
- heihe baseline (nst=6698 / nfe=6943) → identity (nst=6599 / nfe=6696) (Δ -99 / -247)
- heihe_x4 baseline (nst=6575 / nfe=6741) → identity (nst=6569 / nfe=6775) (Δ -6 / +34)
- Expected: PREC_LEFT switch changes CVLS state machine even though P^-1=I numerically; soft gate 5 SHA12 strict equality expected to FAIL → max_ulp ≤ 1024 fallback per spec L106 (PR-F adjudicates)

## Agent Review

- Reviewer agents used: `review-correctness`, `review-spec-compliance`, `review-integration`, `review-documentation` (Phase 4 round 1, expanded fixture, 4 parallel reviewers) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED on Critical (self-evident factual SHA mismatch — `grep outer_pin` vs `git rev-parse HEAD`)
- Reviewed head SHA: `2eb5d0f` (Phase 4) → `291a0a4` (Phase 7 post Phase 6 fix)
- Review evidence: see this PR's comments — Phase 4 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Server/local partition + File IO + Concurrency + Spec compliance + Documentation
- Key findings addressed: 1 CONFIRMED Critical (outer_pin SHA L6/L30) → Phase 6 orchestrator-direct fixed. 0 merge-blocking PLAUSIBLE. 3 forward action notes carried to PR-F #347 (Elapsed+ExitCode columns / `.review-evidence/` gitignore / median sort cite).

## Test plan

- [x] 3 new tool scripts created + Mac dry-run validated 18-cell stdout + bash -n syntax PASS
- [x] Server git sync + SHUD pointer 5276167 + build OMP_RHS=1+PROFILE=1 exit 0
- [x] Server nm gate-1 evidence 3-symbol PASS
- [x] 18-cell Slurm submit (JID 9531..9548) + 18/18 COMPLETED ExitCode 0:0
- [x] rsync 47.78MB to /tmp/p8pre_identity_spike/ verified 18 cells × 3 file types
- [x] cell_stats.txt 18×9 populated, nps>0/npe>0/t_precond_setup present 全 18
- [x] identity_spike_run.md 191 行 lightweight execution log written
- [x] Phase 6 outer_pin SHA fix applied (Critical resolved)
- [x] openspec validate p8pre-spike --strict --no-interactive exit 0
- [x] Phase 4 round 1 expanded cross-review (3 APPROVE + 1 REQUEST CHANGES → Phase 6 RESOLVED)
- [x] Phase 7 independent final review (11/11 AC PASS, Phase 6 fix verified, clean)
- [x] CI: 5/5 PASS @ 291a0a4
- [ ] Auto-merge after pre-merge evidence hard-gate
