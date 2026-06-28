## Summary

p8pre-spike Step 1 capstone PR-C slice. Academic-paper-style `docs/p8pre/n8_profile_baseline.md` anchors the gate-4 `wall_step1_baseline_median(case, N)` for Step 2 PR-F #347 hard-gate; case_deployment_map + master plan cross-refs land. Phase 6 working-tree-only fix corrects openspec text drift §3 → §5.1 (matching academic-style Results structure). Closes #343, refs #338.

### Deliverables (tracked diff)

| 文件 | 用途 |
|---|---|
| `docs/p8pre/n8_profile_baseline.md` (NEW, 367 行) | Academic-paper-style capstone: YAML metadata + Abstract + §1 Introduction (H1/H2/H3 hypotheses) + §2 Related Work + §3 Methodology + §4 Experimental Setup + §5 Results (§5.1 gate-4 raw data + §5.2 invariance + §5.3 baseline anchor + §5.4 ROI ratios + §5.5 branch verdict) + §6 Discussion + §7 Limitations + §8 Conclusion + §9 Future Work + §10 References |
| `docs/case_deployment_map.md` (MODIFIED +27) | §5.1 新 18-row table archives N=8 Mode C profile yaml paths |
| `SHUD_openMP_master_plan.md` (MODIFIED +12) | §P8-precond.0 prep cross-ref block at L2356 |

### Phase 6 working-tree-only fix (openspec/changes/ per .gitignore:13)

5 处 §3 → §5.1 substitution 跨 4 gitignored openspec/changes/p8pre-spike/ 文件（`tasks.md` L41+L77, `specs/n8-mode-c-profile-recheck/spec.md` L17+L113, `specs/p8precond-zero-identity-spike/spec.md` L90, `design.md` L26+L107）. Mirrors PR #353 cand-02 precedent. Fix archives to persistent `openspec/specs/` at #349.

### Acceptance Criteria (全 PASS @ head `43bd6f2`)

| AC | 实测 |
|---|---|
| `docs/p8pre/n8_profile_baseline.md` 完成 academic-paper-style 10 § + Abstract | ✓ 367 行 |
| §5.1 raw data table 含 6 cell wall_median (gate-4 baseline source) | ✓ heihe + heihe_x4 × N=1/4/8 |
| 4 absolute baseline anchors quoted (heihe nst=6698/nfe=6943, heihe_x4 nst=6575/nfe=6741) | ✓ §5.3 |
| ROI verdict (r_min=1.819, r_max=4.526, branch a) quoted from PR-B | ✓ §5.4 + §5.5 |
| ADR-0003 NO-GO NOT drafted (branch a path) | ✓ correctly omitted |
| `docs/case_deployment_map.md` §5.1 補 18 yaml paths | ✓ |
| master plan §P8-precond.0 prep cross-ref | ✓ L2356 |
| openspec validate strict | ✓ exit 0 (post Phase 6 fix) |
| SHUD pin 7a1dc8f unchanged | ✓ |
| All 13/14 cited file paths resolve (1 forward ref to PR-F #347 aggregate_identity_spike.sh — expected) | ✓ |
| Phase 6 openspec/changes/ fix 7 anchor citations | ✓ all `§5.1`, 0 stale `§3` to baseline doc |

### Wall median table (gate-4 baseline anchor)

```
case        N    n_reps    wall_median(s)    nst_median    nfe_median
heihe       1    3         140.797           6698          6943
heihe       4    3          95.734           6698          6943
heihe       8    3          89.732           6698          6943
heihe_x4    1    3        1412.895           6575          6741
heihe_x4    4    3         849.704           6575          6741
heihe_x4    8    3         743.552           6575          6741
```

Source: `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/profile_B0.yaml` `extras.t_wall_total`, sorted-middle median per (case, N). PR-F #347 gate-4 reads this column verbatim as Step 2 baseline.

### Step 1 GATE → PROCEED (再确认)

- `r_min = 1.819 ≥ 1.5` → **Step 1 GATE PASS**
- Next: Step 2 P8-precond-0 identity spike (#344 / #345 / #346 / #347 / #348 / ADR-0003) → #349 openspec archive + Epic close

## Agent Review

- Reviewer agents used: `review-correctness`, `review-integration` (Phase 4 round 1) + `review-integration` (Phase 6.5 round 2 post-fix) + `phase-7-final-review` (Phase 7 Gap Sweep)
- Phase 4.5 verifier: 1 candidate (cand-01 §3 anchor drift) → CONFIRMED + merge-blocking
- Phase 6 orchestrator-direct fix: 5 substitutions in working-tree-only openspec/changes/ (per `.gitignore:13` transient policy)
- Phase 6.5 round 2: cand-01 RESOLVED, 0 new findings
- Reviewed head SHA: `43bd6f2` (tracked unchanged; openspec/changes fix is working-tree-only)
- Review evidence: see this PR's comments — Phase 4 + 4.5 + 6.5 bundle / Phase 7 final review
- OpenSpec change: `p8pre-spike`; fixture level: `compact`; selected risk packs: Documentation + Legacy-compatibility + Spec compliance (gate-4 anchor for PR-F #347 downstream consumer)
- Key findings addressed: 1 CONFIRMED → 1 RESOLVED in Phase 6 + 1 disclosure (openspec/changes/ fix gitignored, working-tree-only, archives at #349)

## Test plan

- [x] `openspec validate p8pre-spike --strict --no-interactive` exit 0 (round 1 + post Phase 6)
- [x] 4 absolute baselines + 10 invariance Δ=0 + 6 wall_median + 6 ROI ratios verified
- [x] All 13/14 cited paths resolve (1 forward ref PR-F #347 expected)
- [x] Branch verdict reaffirmed (no ADR-0003 NO-GO drafted)
- [x] Phase 4 round 1 compact cross-review (correctness APPROVE / integration 1 Warning candidate)
- [x] Phase 4.5 verifier on cand-01 (CONFIRMED + merge-blocking)
- [x] Phase 6 fix: 5 §3 → §5.1 across 4 openspec/changes/ files
- [x] Phase 6.5 round 2 integration re-review (RESOLVED, 0 new findings)
- [x] Phase 7 independent final review: clean, APPROVE merge
- [x] CI: 5/5 PASS (asan-ubsan keliya/qhh, build-and-compare keliya, setup, tools-tests)
- [ ] Auto-merge after pre-merge evidence hard-gate
