Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 3494c6ac1435c1616cbddc8c5058f5c715f0d14e

Summary: PR-A delivers a coherent baseline anchor with strong codepath-equivalence proof, real keliya smoke data, and a working aggregator (exit=0 on Plan A, exit=1 fallback on Plan B). All 10 cross-PR integration contracts hold; two non-blocking presentation gaps worth flagging for PR-C/PR-D consumers.

Findings:
- None.

Non-blocking notes:
- N1 (PR-C reproducibility detail): `docs/p8tune/clean_prec_none_baseline.md` §keliya-smoke-anchor L300 lists `Build flags: make shud SHUD_ENABLE_PROFILE=1 -j 8`, but task §3.6 requires PR-C to also build `make shud_omp` and §3.7 4-way gate runs against `./shud` (serial) only. Cite is consistent (the smoke anchor is serial-only by design), but PR-C implementer reading L294-303 should be told explicitly "serial-only smoke; OMP build separately validated via G1 only." A one-line note would prevent PR-C reviewer ambiguity. Path/line: `docs/p8tune/clean_prec_none_baseline.md:300-303`.
- N2 (T1 "rep" semantics presentational): aggregator emits 3 identical rows per (case, N) in T1 with the same median value (lines 219-241 of `tools/p8tune/aggregate_clean_baseline.sh`), with each row labeled `rep=1/2/3` but actually holding the median. The footnote "median == min == max" (clean_prec_none_baseline.md L163-167) covers it textually, but downstream PR-E aggregator should not naively join PR-A rep rows with PR-D rep rows as if they were independent observations — they're median triplicates. Cross-ref to §3.2 Δ=0 strict invariance is correct; just worth flagging that PR-E joins must dedupe on (case, N) before any rep-level statistics. Path: `tools/p8tune/aggregate_clean_baseline.sh:219-241`.

Per-check assessment:
1. PR-C G3 4-way contract:                  pass
2. PR-D 60-cell decision input:             pass
3. PR-D G6 no-solver-regression baseline:   pass
4. PR-E mode-C-tune reference:              pass
5. PR-A → PR-0 cross-doc consistency:       pass
6. Aggregator CLI contract:                 pass
7. Plan B template provenance:              pass
8. Smoke job reproducibility (toolchain):   pass
9. tasks §0 cross-PR dependency:            pass
10. No premature ADR-0004 commitment:       pass

## Per-check evidence summary

1. **PR-C G3 4-way contract** — §keliya-smoke-anchor (L288-336) provides all required fields: rivqdown.dat SHA12 `1bfe6a30856e`, all 15 canonical-key values (lines 314-331), build env table (server/node/SHUD pin/compiler/linker/build flags/Slurm job ID, L294-301), and exact run command `unset SHUD_SPGMR_MAXL && ./build/shud keliya` (L303). Contract at L333 explicitly enumerates the 4-way gate predicate matching spec L51-61.

2. **PR-D decision input** — §decision-input-table (L386-398) provides hard-evidence trigger status for both cases (`ncfl=85>0` heihe; `ncfl=3620>0` heihe_x4), saturation indicators (4.527 heihe_x4 at threshold; 1.820 heihe below), explicit verdict `"hard-evidence satisfied → full 60-cell sweep"` (L396-397). Unambiguous input for PR-B gate.

3. **PR-D G6 baseline** — §solver-failure-table (L264-281) provides BOTH N=1 and N=8 rows for each case. Floor values verified: heihe ncfn=7 ncfl=85 netf=0 (L271-272); heihe_x4 ncfn=51 ncfl=3620 netf=0 (L273-274). Matches spec L88-91 exactly. Aggregator script (`tools/p8tune/aggregate_clean_baseline.sh:73-79, 162-198`) enforces these via runtime floor verification with exit=1 on mismatch.

4. **PR-E mode-C-tune reference** — §mode-C-tune-reference (L400-414) cites §codepath-equivalence as upstream evidence and cross-refs `openspec/glossary.md` term. Confirmed glossary L294-296 holds the mode-C-tune definition with the per-(case, maxl) anchor structure required by PR-E aggregator T1-T8.

5. **PR-A → PR-0 cross-doc consistency** — Verified: corrected PR-0 ncfn citations at `docs/adr/0003-precond-spike-decision.md:70`, `docs/p8pre/identity_spike_verdict.md:169`, `docs/p8pre/capstone.md:243` all read "`ncfn_candidate ≤ 7 (heihe) ∧ ncfn_candidate ≤ 51 (heihe_x4)`" referencing n8_profile_verdict.md §3.1. Same source ratifies PR-A §solver-failure-table values. No divergence.

6. **Aggregator CLI contract** — `--plan-a` default emits 5 tables from §3.1 (verified: exit=0, 127 lines, 0 FLOOR FAIL). `--plan-b` gates on missing scratch dir with exit=1 + helpful NOTE redirected to stderr (verified via `bash -x`). Tool name `aggregate_clean_baseline.sh` is distinct from PR-E's planned `aggregate_maxl_sweep.sh` — no flag collision since they are separate binaries.

7. **Plan B template provenance** — §submit-template-provenance (L104-132) documents ssh probe showing template missing, then provides a 5-row derivation table (OpenMP wrapper / case deployment / Slurm ancestor / SHUD pin / build flags) with concrete path to existing rendered ancestor at `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/sbatch/Cheihe_x4_N1_rep1.sbatch`. Provenance is reusable if Plan B is later triggered.

8. **Smoke job reproducibility** — §keliya-smoke-anchor build-env table (L296-301) documents the toolchain dependency explicitly: cn14, gcc 13.3.0-6ubuntu2~24.04, libsundials_cvode.so.6 absolute path, Slurm job ID 9620. PR-C implementer has the exact toolchain pin to match. Implicit but adequate: cn14 must be the build/run node for G3 to hold.

9. **tasks §0 cross-PR dependency** — PR sequencing (tasks L5-13 + design D11 L188-196): PR-A produces both the decision-input table consumed by PR-B AND the keliya smoke anchor consumed by PR-C. PR-A output fulfills both downstream deps.

10. **No premature ADR-0004 commitment** — §decision-input-table contains only INPUT data (hard-evidence trigger status + saturation ratio) and the sweep-mode verdict ("full 60-cell sweep"), which is a PR-B-scoped gating decision per spec L7-22, NOT an ADR-0004 outcome branch. No mention of GO/NO-GO/default-bump/Optional-knob/Diagnostic anywhere in PR-A doc — appropriate scoping.

Verdict: APPROVE — all 10 integration contracts satisfied; two non-blocking presentation notes (N1/N2) that PR-C/PR-E implementers should be aware of but do not block PR-A merge.
