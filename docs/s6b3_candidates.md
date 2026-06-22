# S6b.3 Candidate Audit — S2 Follow-up Bug Roster

| Field | Value |
|---|---|
| Audit date | 2026-06-21 |
| Auditor | Phase-1 implementer agent (issue #187) |
| Scope | All "S2 阶段记录但推迟到 B1b" bug候选 — completes spec.md §S6b.3 Requirement L35–L49 + design.md R5 |
| Outer branch at audit | `feat/issue-187-b1b-s6b-3` (base `baseline/B1b@e044cbe`) |
| SHUD HEAD at audit | `bd7714a` on `openmp-baseline` |
| Candidate count | **1** (issue #159 only) |
| Empty-fallback line required | NO (count > 0 — spec L47–L49 Scenario applies to 0-candidate case) |

## Audit method

Per spec.md L43–L46 Scenario "S6b.3 候选 issue 数量与 S2 record 表对齐" the
roster equals the union of (a) commits on `baseline/B1b` not on
`baseline/B1a` whose subject mentions `S6b-followup` or `S2 follow-up`,
(b) GitHub issues labelled `S6b-followup` in `DankerMu/SHUD-OpenMP`, and
(c) `residual_deferred` entries in `docs/review-loop-log.jsonl` from
PR-1..PR-12 explicitly tied to S2 / S6b semantics.

### 1. Commit-message grep (outer repo)

```text
$ git log --grep="S6b-followup" --oneline baseline/B1b ^baseline/B1a
(empty)

$ git log --grep="S2.*follow-up" --oneline baseline/B1b ^baseline/B1a
(empty)
```

→ 0 commits tagged with these tokens on the `B1b` chain. The label
convention exists in spec.md but no historical commit carries it.

### 2. GitHub issue label scan

```text
$ gh issue list --label "S6b-followup" --repo DankerMu/SHUD-OpenMP --state all --json number,title,state
[]

$ gh label list --repo DankerMu/SHUD-OpenMP | grep -i "follow\|S6b\|S2"
s2-strict   S2 strict OpenMP 阶段 (P1-P7)   #5319e7
```

→ The `S6b-followup` label is referenced by spec.md but was never created
in the GitHub repo. The only S2-flavoured label is `s2-strict`, which
covers P1-P7 strict-mode tracking and is orthogonal to the
"deferred-bug" semantics audited here.

### 3. review-loop-log `residual_deferred` mining (manual)

Hand-scan of every `residual_deferred > 0` note in
`docs/review-loop-log.jsonl` (PR-1..PR-12). Only matches naming S2 / S6b
semantics are listed:

| PR | issue | residual_deferred note excerpt | Maps to a candidate? |
|---|---|---|---|
| PR-2 | #145 | "Pack 4 cross-review CLEAN x3 (1 WARN uYgw asymmetry PLAUSIBLE per verifier, deferred to **#159 follow-up**)" + "residual_deferred = D13 backend gap + **uYgw P1+ alignment**" | YES — this IS issue #159 |
| PR-12 | #156 | "**#159 (S2.6 P1+ pre-req) retained out of B1a scope**" | YES — same issue #159 |
| PR-1 | #144 | "cand-02/03/06 wontfix defer PR-12" + "server heihe/heihe_x4 deferred to PR-12 capstone" | NO — orchestrator-fixture candidates + server-run sequencing; PR-12 capstone closed both; no S2 bug-fix semantic |
| PR-3 | #146 | "S2.7 already satisfied by S1a rhs_update PURE CARRY-OVER" + "S2.8 implementation deferred to PR-9 #153 paired with S3a fun_Seg_* dead-+= rem" | NO — implementation-sequencing within S2 chain, closed by PR-9; not a B1b-deferred bug |
| PR-10 | #182 | "S5d.4 review-fix M1/M2/M3/m4 follow-up + residual deferred list" | NO — S5d.4 review residuals were dispositioned within PR-10 + S5d capstone PR-11 |
| PR-12 (S6b.1) | #184 | none — clean; `residual_deferred=0` is implied by absence | NO |

→ Only issue **#159** appears as a true S2-class bug deferred from B1a
to B1b structural-fix window. All other `residual_deferred` mentions
are orchestrator-scope, implementation-sequencing, or server-pending
items that closed within the same Stage chain.

### 4. spec.md / design.md cross-reference

Both `openspec/changes/b1b-baseline-completion/specs/s6b-bugfix-application/spec.md`
L37 + `design.md` L227 explicitly name "#159 (S2.6 follow-up)" as the
only known candidate, and acknowledge the rest of the roster is
discovered via audit (this document). Both documents pre-budget the
0-3 candidate range.

→ Audit converges on exactly **1** candidate: issue **#159**.

## Candidates

### Candidate S6b.3.1 — issue #159 (S2.6 follow-up: `f_update_omp` `uYgw` `iBC == 0` asymmetry)

| Field | Value |
|---|---|
| Source PR/issue | `DankerMu/SHUD-OpenMP#159`, opened from PR-2 #145 Pack 1 Finding 2 / Phase 4.5 verifier verdict PLAUSIBLE |
| Original target file | `SHUD/src/ModelData/MD_f_omp.cpp` (L124, `iBC == 0` branch, `uYgw[i] = max(0.0, Y[iGW]);`) |
| Symptom described | Dormant TU `MD_f_omp.cpp::f_update_omp` had a 3-aligned-1-asymmetric form: `uYsf` / `uYus` / `uYriv` matched serial `f_update` direct-alias (`Y[iSF/iUS/iRIV]`) but `uYgw` retained the older `max(0.0, Y[iGW])` clamp. spec carve-out at PR-2 spec.md L109 + design.md L290 marked the asymmetry as "dormant-path historical quirk", deferred to P1+ OMP re-activation. |
| Default-build impact | ZERO. `MD_f_omp.cpp` was filtered out of the default build (Makefile L366-370 of the PR-2 era). |
| Evaluation conclusion | **NOT A BUG (auto-resolved by S2 capstone PR-8 #152)** |

#### Evidence chain

1. **PR-8 #152 (S2 capstone) physically deleted `MD_f_omp.cpp`**:

   ```text
   $ cd SHUD && git log openmp-baseline --diff-filter=D --name-only --oneline | grep -B1 -A0 "MD_f_omp" | head
   22777e5 S2 capstone: delete MD_f_omp.cpp + retire LEGACY_RHS + SHUD_LEGACY_OMP_RHS (PR-8 #152)
   src/ModelData/MD_f_omp.cpp
   ```

   The target TU of the asymmetry no longer exists in the source tree
   on `openmp-baseline`. There is therefore no code site to align.

2. **Verification: `MD_f_omp.cpp` absent from current SHUD source tree**:

   ```text
   $ find /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD -name "MD_f_omp*"
   (empty)
   $ ls /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/MD_f*.cpp
   /.../SHUD/src/ModelData/MD_f.cpp
   /.../SHUD/src/ModelData/MD_f_uncouple.cpp
   ```

3. **Verification: the two surviving `uYgw[i] = ...` sites in
   `iBC == 0` branches already use serial direct-alias** (i.e. the very
   form #159 asked the OMP variant to adopt):

   - `SHUD/src/ModelData/MD_update.cpp:63-86` `Model_Data::f_update`:

     ```cpp
     if(Ele[i].iBC == 0){ // NO BC
     //   uYgw[i] = max(0.0, Y[iGW]);
         uYgw[i] = Y[iGW];
         Ele[i].QBC = 0.;
     }
     ```

   - `SHUD/src/Model/MD_rhs_core.cpp:55-86` `Model_Data::rhs_update`:

     ```cpp
     if(Ele[i].iBC == 0){ // NO BC
     //   uYgw[i] = max(0.0, Y[iGW]);
         uYgw[i] = Y[iGW];
         Ele[i].QBC = 0.;
     }
     ```

   The commented-out `max(0.0, Y[iGW])` is preserved as a historical
   comment so future readers can trace the lineage; the active code
   is the direct-alias form that #159 advocated.

#### Out-of-scope live `max(0.0, Y[i])` site (forward defense)

PR #203 review MN1 (`af5eb35f22058f59e`) noted one LIVE site at
`SHUD/src/ModelData/MD_update.cpp:22` inside `Model_Data::f_updatei`
case 3 (`iBC == 0` branch) still uses `uYgw[i] = max(0.0, Y[i]);`.
This callback is `f_gw` registered to `CVode(mem3, ...)` at
`SHUD/src/Model/shud.cpp:336,389`, reachable ONLY when SHUD is built
with `-DSHUD_uncouple` and run with CLI `-g` (uncoupled GW-only mode).

This site is **OUT OF #159's scope** — #159 names the dormant `_omp`
variant `MD_f_omp.cpp::f_update_omp`, not the uncoupled-mode
`f_updatei`. The asymmetry pattern survives here because uncoupled
mode is not on the B1b coupled-RHS code path (B1b 90-day baseline + 7
benchmark cases all run coupled, never `-g`). Future work touching
uncoupled mode should evaluate alignment with `f_update` /
`rhs_update`; recording here so the next audit catches it.

#### Disposition per spec.md L37 clause (d)

Spec L37 mandates "若评估结论为'非 bug' 或'延后到 P1+ 处理'，仍 SHALL
写入 changelog 解释". This audit records #159 as **NOT A BUG (auto-resolved
by PR-8 #152)**: the asymmetry was eliminated structurally (target file
deleted), and the surviving serial sites already use the desired
direct-alias form. There is no code change required for S6b.3.1. The
disposition is written to:

- `SHUD/B1b_CHANGELOG.md` S6b.3.1 section (this PR)
- GitHub issue #159 comment with link to the CHANGELOG SHA (this PR)
- `docs/diff_reports/B1a_vs_B1b_diff_s6b_3_1.md` "zero-impact /
  no-code-change" report (this PR)

Issue #159 will be closed on PR merge per PR-12 closing convention
("Closes #N" + manual `gh issue close` if needed when PR base !=
`main`).

## Roster summary

| Seq | Source | Title | Conclusion | Action |
|---|---|---|---|---|
| S6b.3.1 | issue #159 | f_update_omp uYgw iBC==0 asymmetry | NOT A BUG — auto-resolved by PR-8 #152 (`MD_f_omp.cpp` deleted; serial path already direct-alias) | CHANGELOG explanation + diff-report stub + #159 comment + close on PR merge |

Total candidates: **1**. Total code-change candidates: **0**. Total
B1a→B1b numerical-output deltas attributable to S6b.3: **0**.

## Closing note

Per design.md D9 fast-path trigger #3 ("S6b.3 全部候选评估结论为
zero-impact (含 0 候选 / 非 bug / 延后到 B1b 之外)"), this audit
contributes a "non-bug auto-resolved" verdict toward the D9 trigger
chain. Combined with S6b.1 zero-impact (PR-12 #184/#202 verified) and
S6b.2 status (#186 — to be determined by PI review #185), the D9
fast-path eligibility hinges only on S6b.2's outcome.
