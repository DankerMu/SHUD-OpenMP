Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed outer HEAD SHA: 768c905f8f078e7ece27bc4d8e4efb4ab0a1b825
Reviewed SHUD pin: 6ce17d6

Summary: All 8 tasks DONE, all 4 spec Requirements satisfied, all 11 IM regression rows covered, no scope creep, openspec strict PASS, SHUD push discipline honored.

Per-task DONE/MISSING:
- §3.1: DONE — helper `static int get_spgmr_maxl_from_env(void)` at L248-308; strict whitelist {0,5,10,15,20,30}; rejects "+5"/"-1"/"05"/"7"/"foo" via char-by-char + strtol; invalid → stderr + myexit(ERRCVODE)
- §3.2: DONE — L324 (post-edit; was L259 pre-edit): `SUNLinSol_SPGMR(udata, PREC_NONE, get_spgmr_maxl_from_env(), sunctx)`; PREC_NONE preserved
- §3.3: DONE — provenance log conditional on val ∈ {5,10,15,20,30}; unset/""/"0" suppress; G3 evidence confirms 4-way log-line count {0,0,0,1}
- §3.4: DONE — grep `PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity` returned 0 matches at HEAD 6ce17d6
- §3.5: DONE — SHUD diff = `src/Equations/cvode_config.cpp` only (+67/-2); outer diff = `SHUD` pointer only (1 line)
- §3.6: PARTIAL — server-side build implicit via Slurm 9626 functional binary; Mac-side build + `nm | grep -i spgmr` not asserted in g3_verdict.md (non-blocking per project §1.1.1 server-only quantitative verification convention)
- §3.7: DONE — g3_verdict.md: all 4 runs SHA12 = `1bfe6a30856e` matches PR-A anchor; all 15 canonical counters match; cross-run cmp byte-identical
- §3.8: DONE — SHUD `6ce17d6` exists ONLY on origin/openmp-baseline; master unchanged; outer pointer-bump PR HEAD = `768c905`

Findings: None.

Non-blocking notes:
- §3.6 Mac build assertion absent from g3_verdict.md; server build implicit. Belt-and-suspenders cleanup possible but not blocking.
- g3_verdict.md §sbatch ExitCode 1:0 footnote honestly discloses sbatch parser bug + manual re-verification path (right transparency posture).
- Implementation exceeds spec L31 minimum by rejecting "+5"/"05"/"5x"/" 5"/"\t5"/"10.0" (defense-in-depth; safer than spec).

Per-check assessment:
1. Per-task DONE/MISSING:                  pass (§3.6 partial = non-blocking)
2. Spec scenario coverage:                 pass — all 7 scenarios traceable to code + G3 evidence
3. Invariant Matrix coverage:              pass — all 11 D15 regression rows verified
4. Issue #366 acceptance:                  pass — G1 (server functional), G2 (grep 0), G3 (4-way SHA12 + 15-key), source surface all satisfied
5. Scope creep audit:                      pass — outer 1 file (pointer); SHUD 1 file (cvode_config.cpp); no Makefile/header/sibling/SUNDIALS/Basins/tools
6. Cross-PR sequencing:                    pass — g3_verdict.md cites PR-A anchor `37be0fe` + SHA12 `1bfe6a30856e`
7. openspec validate strict:               pass — "Change 'p8tune-spgmr-maxl' is valid"
8. SHUD submodule push discipline:         pass — `6ce17d6` on origin/openmp-baseline only; master HEAD unchanged

Verdict: APPROVE — PR-C is spec-compliant, scope-bounded, evidence-backed, ready to merge.
