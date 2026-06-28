## Summary

p8pre-spike Step 2 P8-precond-0 spike PR-D impl slice. SHUD submodule pointer bump to `5276167` (forward-only descendant of `7a1dc8f`) — adds `MD_precond_identity.{h,cpp}` + `cvode_config.cpp` `PREC_LEFT` wire + `CVodeSetPreconditioner` + `CVodeSetLSetupFrequency(50)`. Mac local sanity + keliya smoke PASS. Closes #345, refs #338.

### Deliverables

**SHUD upstream (5276167 on openmp-baseline-p8pre, pushed)**:
- NEW `SHUD/src/Equations/MD_precond_identity.h` (40 lines)
- NEW `SHUD/src/Equations/MD_precond_identity.cpp` (61 lines, RAII Timer + jok-mirror + N_VScale)
- EDIT `SHUD/src/Equations/cvode_config.cpp` (+24/-3)

**Outer repo**:
- SHUD pointer bump 7a1dc8f → 5276167

### Acceptance Criteria (全 PASS @ head `1e45e16`)

| AC | 实测 |
|---|---|
| 名字碰撞预检 0 hit | ✓ `grep -rnE "PSetup\|PSolve" SHUD/src/` empty |
| timer.cpp emit 预检 catch-all 自动 emit | ✓ `t_precond_setup` 不在 `kKnownRawOrCanonical[]` L188-191, fallback NOT fired |
| 3 文件 SHUD 改动 (新建 2 + 编辑 1) | ✓ |
| Mac build `make shud SHUD_ENABLE_OPENMP_RHS=1` exit 0 | ✓ |
| Mac build `+ SHUD_ENABLE_PROFILE=1` exit 0 | ✓ |
| `nm ./shud` 3-symbol verify | ✓ PSetup/PSolve count=2, CVodeSetPreconditioner count=1 |
| Mac `./shud keliya` exit 0 + profile_B0.yaml emit | ✓ |
| `extras: t_precond_setup` present (soft gate 6 evidence) | ✓ value=0.000037944 |
| `cvode_stats nps > 0 AND npe > 0` (gate 3) | ✓ nps=209227, npe=1232 |
| SHUD HEAD 5276167 forward-only descendant of 7a1dc8f | ✓ merge-base = 7a1dc8f exactly |
| 外层 pointer bump commit message 含 "feat(p8pre)" prefix | ✓ |
| openspec validate strict | ✓ exit 0 |

### Mac smoke result (keliya N=1)

```
shud exit: 0
profile_B0.yaml extras:
  t_CVODE_raw:    113.488523000
  t_wall_total:   117.364403000
  t_precond_setup: 0.000037944
cvode_stats:
  nst=101042
  nfe=102095
  nps=209227   ← gate 3 PASS
  npe=1232     ← gate 3 PASS
  nni=102094
  netf=7
```

### Implementer Deviation (jok-mirror pattern)

Brief originally specified `*jcurPtr = SUNFALSE` unconditional in `PSetupIdentity`. v1 attempt with that pattern yielded **npe=0**, violating gate 3 (`npe > 0` required per spec Scenario "gate 3 nps and npe accumulation").

Root cause: SUNDIALS 6.0.0 CVLS internal accounting only increments `cvls_mem->npe` inside the `jok == SUNFALSE` rebuild branch. With unconditional SUNFALSE, the rebuild branch never triggers updated bookkeeping → npe stays at 0.

**Fix**: jok-mirror pattern matching SUNDIALS canonical `cvDiurnal_kry.c` L716/L760 example:
```cpp
*jcurPtr = jok ? SUNFALSE : SUNTRUE;
```

Numerical equivalence preserved: `PSolveIdentity` always does `N_VScale(1.0, r, z)` (P=I) regardless of `*jcurPtr` value — the pattern only affects CVLS state-machine bookkeeping (preconditioner cache scheduling). Result: npe=1232 PASS while preserving P=I contract.

Documented in `.review-evidence/p8pre-pr-d-impl/smoke_analysis.txt` for Phase 4 reviewer verification. Phase 4 spec-compliance reviewer independently verified jok-mirror IS canonical via `SHUD/InstallSundials/example/cvode/serial/cvDiurnal_kry.c` L716/L760 + recommends future openspec L26 wording patch.

### Forward-only descendant guarantee

- new SHUD HEAD `5276167`
- `merge-base 5276167 7a1dc8f = 7a1dc8f` strict
- → linear descendant per spec p8precond-zero-identity-spike Scenario L148-154

## Agent Review

- Reviewer agents used: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf` (Phase 4 round 1, expanded fixture, 4 parallel reviewers) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: SKIPPED (0 PLAUSIBLE candidates — 4/4 APPROVE 全 0 findings)
- Reviewed head SHA: `1e45e16`
- Review evidence: see this PR's comments — Phase 4 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `expanded`; selected risk packs: Public API/CLI + File IO + Concurrency (Timer + SUNDIALS) + Numerical stability + Spec compliance + Server/local partition + Documentation
- Key findings addressed: 0 CONFIRMED, 0 merge-blocking. 3 forward notes carried (jok-mirror openspec patch / PREC_LEFT cosmetic upgrade / B1b bitwise neutrality soft gate 5 handling)

## Test plan

- [x] 2 pre-flight checks (name collision + timer.cpp emit)
- [x] 3 SHUD file changes on openmp-baseline-p8pre
- [x] Mac build x 2 build matrices (OMP_RHS=1, OMP_RHS=1+PROFILE=1)
- [x] 3-symbol nm verify
- [x] Mac keliya smoke: exit 0 + profile yaml + extras + nps/npe
- [x] SHUD commit + push + forward-only descendant verify
- [x] Outer pointer bump
- [x] openspec validate strict exit 0
- [x] Phase 4 round 1 expanded cross-review (4/4 APPROVE)
- [x] Phase 7 independent final review: clean (12/12 AC, CI 5/5)
- [ ] Auto-merge after pre-merge evidence hard-gate
