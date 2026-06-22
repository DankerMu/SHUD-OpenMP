# B1a vs B1b Diff Report — S6b.3.1 (issue #159, f_update_omp uYgw asymmetry)

| Field | Value |
|---|---|
| Fix ID | S6b.3.1 |
| Source | issue `DankerMu/SHUD-OpenMP#159` (S2.6 follow-up) |
| Original target | `SHUD/src/ModelData/MD_f_omp.cpp:124` (deleted file) |
| Code change in this PR | NONE — non-bug, auto-resolved |
| Zero-impact | YES |
| Diff vs B1a-tag | NONE (no code modification, no benchmark output delta) |

## Classification

**zero-impact fix** per spec.md L55 + master plan §S6b L1495 — this
S6b.3 candidate is dispositioned as "NOT A BUG" because the original
target file (`MD_f_omp.cpp`) was already physically removed by S2
capstone PR-8 #152 (SHUD commit `22777e5`), and the surviving serial
RHS update sites (`f_update` in `MD_update.cpp`, `rhs_update` in
`MD_rhs_core.cpp`) already use the direct-alias form
`uYgw[i] = Y[iGW];` that #159 advocated for the dormant OMP twin.

There is therefore no code patch, no SHUD source diff, and no benchmark
output delta to report in this S6b.3.1 sub-PR. The "diff report" here
exists per spec.md L53-55 Requirement "每个 fix SHALL ... 产出独立
`docs/diff_reports/B1a_vs_B1b_diff_<fix_id>.md` diff report".

## Influence range

- Impacted benchmark cases: **0**
- Impacted `.dat` outputs: **0**
- Impacted physical fields: **0**
- Bitwise outcome vs B1a-tag: **identical** (no code change)

## Evidence

### A. `MD_f_omp.cpp` is no longer in the SHUD source tree

```text
$ find /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD -name "MD_f_omp*"
(empty)

$ ls /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_f*.cpp
/.../MD_f.cpp
/.../MD_f_uncouple.cpp
```

### B. The deletion commit on `openmp-baseline`

```text
$ cd SHUD && git log openmp-baseline --diff-filter=D --name-only --oneline | grep -B1 "MD_f_omp"
22777e5 S2 capstone: delete MD_f_omp.cpp + retire LEGACY_RHS + SHUD_LEGACY_OMP_RHS (PR-8 #152)
src/ModelData/MD_f_omp.cpp
```

### C. The surviving serial sites already use direct-alias

```cpp
// SHUD/src/ModelData/MD_update.cpp:63-86 — Model_Data::f_update
if(Ele[i].iBC == 0){ // NO BC
//    uYgw[i] = max(0.0, Y[iGW]);
    uYgw[i] = Y[iGW];   // <-- the form #159 asked OMP variant to adopt
    Ele[i].QBC = 0.;
}

// SHUD/src/Model/MD_rhs_core.cpp:55-86 — Model_Data::rhs_update
if(Ele[i].iBC == 0){ // NO BC
//    uYgw[i] = max(0.0, Y[iGW]);
    uYgw[i] = Y[iGW];   // <-- same form
    Ele[i].QBC = 0.;
}
```

The commented-out `max(0.0, Y[iGW])` lines are preserved as historical
artefact so future readers can trace why direct-alias was adopted; the
active code is the form that #159 advocated.

## D9 fast-path contribution

Per design.md D9 trigger #3 ("S6b.3 全部候选评估结论为 zero-impact"),
this S6b.3.1 disposition contributes a "zero-impact" verdict (1 of 1
S6b.3 candidates resolved to non-bug). Combined with PR-12 #184
(S6b.1 zero-impact) and #186 (S6b.2 status — pending PI review #185),
the D9 fast-path eligibility is **gated only by S6b.2's outcome**.

## Cross-references

- spec.md L37: clause (d) "若评估结论为'非 bug' 或'延后到 P1+ 处理'，
  仍 SHALL 写入 changelog 解释"
- spec.md L39-41: Scenario "#159 评估结论 zero-impact"
- spec.md L53-55: Requirement "每个 fix 独立 commit + 独立 diff report"
- spec.md L59: Scenario S6b.1 diff report 完整 (parallel precedent for
  this report's structure)
- design.md D8 / D9: "每个 fix 独立 commit" + "B1a/B1b 快速路径"
- master plan §S6b L1495: "若 fix 不影响任何 benchmark 输出 ... 标记
  为'zero-impact fix'"
- `docs/s6b3_candidates.md` (S6b.3 audit roster): single-row entry
  for this candidate
- `SHUD/B1b_CHANGELOG.md` S6b.3.1 section (this PR): mirrored
  disposition row in the single-source-of-truth changelog
