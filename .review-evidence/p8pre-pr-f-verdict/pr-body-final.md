## Summary

p8pre-spike Step 2 PR-F verdict adjudication slice. Aggregator parses 18 cells from PR-E artifact + writes academic-paper-style verdict doc. **Verdict = NO-GO** (gate 2 ncfn FAIL deterministic + gate 5 max_ulp ≈ 9×10¹⁵ structural divergence) → PR-G #348 ADR-0003 NO-GO + design D8 PREC_NONE fall-back. Closes #347, refs #338.

### Deliverables

| 文件 | 用途 |
|---|---|
| `tools/p8pre/aggregate_identity_spike.sh` (NEW, 569 lines) | POSIX bash + awk + sha256sum + uv-python numpy aggregator. Parses 18 cells × profile_B0.yaml + cvode_stats.txt + .rivqdown.dat; computes per (case, N) wall_median + nps_median + npe_median + ncfn_max + per-cell SHA12 + max_ulp. Evaluates 4 hard-gate + 2 soft-gate. Emits structured `aggregate_verdict.txt` |
| `docs/p8pre/identity_spike_verdict.md` (NEW, 250 lines) | Academic-paper-style 11 sections (YAML + Abstract + §1-§10): §3 raw data 18×15 + §4 hard gate table + §5 soft gate table + §6 ROI implication + §7 ADR-0003 NO-GO recommendation + §8 Limitations + §9 Future Work + §10 References |

### 4 Hard gate verdict

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| 1 | Build PASS | server nm 3-symbol (PSetupIdentity / PSolveIdentity / CVodeSetPreconditioner) | **PASS** | server_nm.log 1 hit each |
| 2 | Zero conv failure | ncfn=0 全 18 cells | **FAIL** | heihe ncfn=6 (9/9), heihe_x4 ncfn=47 (9/9) deterministic |
| 3 | nps+npe accumulation | nps>0 AND npe>0 per cell | **PASS** | min_nps=18163, min_npe=77 全 18 |
| 4 | Wall non-regression | \|wall_identity - baseline\|/baseline ≤ ε(case) per (case, N) | **PASS** | max delta heihe=2.64% (<10%), heihe_x4=1.09% (<5%) |

### 2 Soft gate verdict

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| 5 | Cross-N tolerance | SHA12 strict OR max_ulp ≤ 1024 | **FAIL** | strict 18/18 mismatch (no cell matches baseline a2023ccd2de4 / b5e4b0a2cf83); fallback 18/18 max_ulp ≈ 9×10¹⁵ vastly exceeds 1024; structural divergence at 5,155 of 214,252 rivqdown positions |
| 6 | Setup overhead | t_precond_setup / wall ≤ 0.05 per cell | **PASS** | max ratio = 1.01×10⁻⁷, 6 orders below 5×10⁻² threshold |

### Spike verdict: **NO-GO**

Per spec L74-79 + design D5: ANY hard gate FAIL → Step 2 spike verdict = NO-GO. Gate 2 (`ncfn=0` strict) + Gate 5 (`max_ulp ≤ 1024` fallback) both FAIL.

### ADR-0003 recommendation (PR-G #348 scope)

**NO-GO** + design D8 PREC_NONE fall-back. PR-G #348 SHALL:
1. Write `docs/adr/0003-precond-spike-decision.md` documenting NO-GO + rationale
2. Revert `cvode_config.cpp:259` PREC_LEFT → PREC_NONE
3. Delete `MD_precond_identity.{h,cpp}`
4. Update master plan §P8-precond.0 with NO-GO outcome
5. Close `baseline/p8pre` branch (no further work this line)
6. Cite jok-mirror SUNDIALS canonical pattern (cvDiurnal_kry.c L716/L760) for future spec L26 wording correction (PR-D forward carry)

### Acceptance Criteria (PASS @ head `75a757c`)

| AC | 实测 |
|---|---|
| aggregator 解析 18 cell + server_nm.log | ✓ 569 lines POSIX bash |
| 4 hard gate 全部 evaluated (PASS/FAIL each) | ✓ 1 PASS / 1 FAIL / 1 PASS / 1 PASS |
| 2 soft gate 全部 evaluated (PASS/FAIL/DEFER each) | ✓ 1 FAIL / 1 PASS |
| identity_spike_verdict.md 完成 (7+ section academic) | ✓ 250 lines 11 sections (Abstract + §1-§10) |
| §3 raw data 含 per-cell t_precond_setup 列 + Slurm Elapsed + ExitCode (PR-E forward carry) | ✓ 18×15 |
| §4 hard gate verdict table 含 4 项明确决策 | ✓ |
| §7 ADR-0003 推荐 = NO-GO | ✓ |
| openspec validate strict | ✓ exit 0 |
| SHUD pin 5276167 unchanged | ✓ no SHUD source change |

## Agent Review

- Reviewer agents used: `review-spec-compliance`, `review-correctness`, `review-documentation`, `review-integration` (Phase 4 round 1, expanded fixture, 4 parallel reviewers) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED (0 PLAUSIBLE candidates — 4/4 APPROVE 全 0 findings)
- Reviewed head SHA: `75a757c`
- Review evidence: see this PR's comments — Phase 4 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Numerical stability + File IO + Spec compliance + Documentation
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking. 5 cosmetic Suggestions carried as forward notes to PR-G #348

## Test plan

- [x] aggregator 569 lines + bash -n syntax PASS + exit 0
- [x] 4 hard gate + 2 soft gate evaluated; structured `aggregate_verdict.txt` emitted
- [x] identity_spike_verdict.md 250 lines academic-paper-style 11 sections + §3 18×15 raw data table
- [x] openspec validate p8pre-spike --strict --no-interactive exit 0
- [x] Phase 4 round 1 expanded cross-review (4/4 APPROVE 0 findings)
- [x] Phase 7 independent final review: clean (9/9 AC + CI 5/5 PASS + PR-G readiness PASS + aggregator reproducibility PASS)
- [ ] Auto-merge after pre-merge evidence hard-gate
