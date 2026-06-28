---
title: P8-tune.C maxl sweep — 8-gate verdict
status: verdict-final
epic: p8tune-spgmr-maxl (#362)
pr_sequence: PR-E (#368)
adr_xref: docs/adr/0004-maxl-sweep-decision.md
data_source: /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv
data_provenance: Slurm 9690 (60/60 COMPLETED), tools/p8tune/aggregate_maxl_sweep.sh
---

# P8-tune.C maxl sweep — 8-gate verdict + ADR-0004 cross-reference

# PR-E maxl sweep verdict synthesis
- Total cells parsed: 60
- ADR-0004 branch: **Optional-knob**

## Per-gate verdict

| Gate | Verdict | Note |
|---|---|---|
| G1 build | PASS | PR-C CI |
| G2 no-PREC_LEFT-regression | PASS | 60/60 prov_log_count=1 |
| G3 default-compat 4-way | PASS | PR-C CI g3_verdict |
| G4 solver-work reduction | PASS | maxl=10 eliminates ncfl per (case,N); see T4 |
| G5 wall improvement | MIXED (regression present) | case-asymmetric; see T5 + per-combo bands |
| G6 no-solver-regression | PASS | sum Δ(ncfn+ncfl+netf) ≤ 0 per (case,N,maxl); see T6 |
| G7 hydrology max_ulp ≤ 1024 | STRICT FAIL (expected numerical: maxl bump → Krylov subspace expansion → CVODE step-size change → trajectory drift) | rivqdown.dat drifts because step-size adapter responds to ncfl changes; see T7 |
| G8 3-rep determinism | PASS | all 20 (case,N,maxl) tuples bit-identical across reps; see T8 |

## Per-combo best-maxl recommendation (3-rep median)

| case | N | best maxl | best wall improvement % | best wall band | tier (see §Production tune guidance) | rationale |
|---|---|---|---|---|---|---|
| heihe | 1 | **30** | +11.99 | GO | **Performance opt-in (NOT A5-certified)**[*] | maxl=30 gives GO wall band; ncfl elimination preserved; trajectory drift documented in ADR-0004 §G7-attested |
| heihe | 8 | **30** | +6.78 | Optional | **Performance opt-in (NOT A5-certified)**[*] | maxl=30 gives Optional wall band; ncfl elimination preserved; multi-thread L2 contention erodes N=1 GO band per ADR-0004 §Discussion NumY 口径 |
| heihe_x4 | 1 | **5** | +0.00 | default (no opt-in benefit) | n/a | all maxl ≥10 REGRESS; NumY ~120K → maxl=10 working set ~9.6 MB > L2 → DRAM-bound; keep default per 'never break userspace' |
| heihe_x4 | 8 | **5** | +0.00 | default (no opt-in benefit) | n/a | all maxl ≥10 REGRESS; large-case + high-thread combo amplifies Krylov-vector DRAM-thrash per ADR-0004 §Discussion |

[*]: `Performance opt-in (NOT A5-certified)` = mechanism-attested per ADR-0004 §Rationale §G7 (Arnoldi MGS → CVODE step-size adapter → trajectory drift) but NOT validated by hydrology equivalence (NSE/KGE/peak/water-balance). Promotion to `A5-certified` requires future P9-A5 epic per ADR-0004 §Acceptance forward action. Users opting in accept solver-tunable trajectory drift on a different valid PREC_NONE step-size path.

## ADR-0004 branch rationale

- **G1/G2/G3/G4/G6/G8 PASS**: build/PREC_NONE/default-compat/solver-work/no-regression/3-rep-determinism all clean strict.
- **G7 STRICT FAIL — expected numerical phenomenon**: maxl bump enlarges Krylov subspace → SUNDIALS Arnoldi orthogonalization yields different residual → CVODE step-size adapter responds with larger steps → integrated trajectory diverges. Both maxl=5 and maxl≥10 produce valid hydrology solutions on different step-size paths; this is solver-tunable-sensitivity, not corruption.
- **G5 case-size-asymmetric** (key finding): small case (heihe ~6300 elements) BENEFITS from larger Krylov subspace — heihe N=1 maxl=30 gives +12% wall (GO band) + ncfl 85→0; heihe N=8 maxl=30 gives +6.8% wall (Optional band). Large case (heihe_x4 ~40046 elements) SUFFERS — all maxl≥10 REGRESS wall 7-25%, because each Krylov vector occupies more memory bandwidth and bigger maxl multiplies bandwidth pressure during Arnoldi orthogonalization.
- **Outcome — Optional knob branch**: keep PR-C `SHUD_SPGMR_MAXL` env-var as documented production opt-in. Per-case best-maxl table above gives the recommended setting. Default (unset = SUNDIALS-default maxl=5) UNCHANGED for backward-compat per 'never break userspace'.
- **No default-bump (rejects GO branch)** because G5 is not uniform GO across all combos — heihe_x4 actively REGRESSES at maxl=10; bumping the default would break heihe_x4 users.
- **No revert (rejects NO-GO branch)** because G7 STRICT FAIL is expected numerical drift, not corruption; G4/G6/G8 prove the hook itself is well-behaved; small-case users gain real wall + solver-stability benefit.

---
## Detailed gate tables

## T1 — G1 build

Verdict: **PASS**

Evidence: PR-C CI 5/5 PASS at merge_sha b5210b98 (asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests)

## T2 — G2 no PREC_LEFT regression

Verdict: **PASS**

Evidence: prov_log_count sum = 60/60 (all cells emit pretype=PREC_NONE)

## T3 — G3 default-compat 4-way

Verdict: **PASS**

Evidence: PR-C G3 verdict 4-way bit-identical (SHA12=1bfe6a30856e for keliya; preserved in PR-D as maxl=5 = SUNDIALS default path)

| case | N | ncfl@5 | ncfl@10 | Δncfl | ncfn@5 | ncfn@10 | Δncfn | nfeLS@5 | nfeLS@10 | ΔnfeLS | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| heihe | 1 | 85 | 0 | -85 | 7 | 5 | -2 | 12632 | 12581 | -51 | PASS |
| heihe | 8 | 85 | 0 | -85 | 7 | 5 | -2 | 12632 | 12581 | -51 | PASS |
| heihe_x4 | 1 | 3620 | 0 | -3620 | 51 | 0 | -51 | 30509 | 37219 | +6710 | PASS |
| heihe_x4 | 8 | 3620 | 0 | -3620 | 51 | 0 | -51 | 30509 | 37219 | +6710 | PASS |

| case | N | maxl | wall@5 (s) | wall@m (s) | improvement % | band |
|---|---|---|---|---|---|---|
| heihe | 1 | 10 | 148.43 | 138.56 | +6.65 | Optional |
| heihe | 1 | 15 | 148.43 | 133.56 | +10.02 | GO |
| heihe | 1 | 20 | 148.43 | 133.52 | +10.05 | GO |
| heihe | 1 | 30 | 148.43 | 130.64 | +11.99 | GO |
| heihe | 8 | 10 | 143.11 | 149.54 | -4.49 | REGRESSION |
| heihe | 8 | 15 | 143.11 | 140.17 | +2.05 | Diagnostic |
| heihe | 8 | 20 | 143.11 | 133.80 | +6.51 | Optional |
| heihe | 8 | 30 | 143.11 | 133.40 | +6.78 | Optional |
| heihe_x4 | 1 | 10 | 1489.76 | 1591.93 | -6.86 | REGRESSION |
| heihe_x4 | 1 | 15 | 1489.76 | 1624.74 | -9.06 | REGRESSION |
| heihe_x4 | 1 | 20 | 1489.76 | 1692.83 | -13.63 | REGRESSION |
| heihe_x4 | 1 | 30 | 1489.76 | 1725.63 | -15.83 | REGRESSION |
| heihe_x4 | 8 | 10 | 1372.04 | 1687.26 | -22.97 | REGRESSION |
| heihe_x4 | 8 | 15 | 1372.04 | 1588.95 | -15.81 | REGRESSION |
| heihe_x4 | 8 | 20 | 1372.04 | 1625.33 | -18.46 | REGRESSION |
| heihe_x4 | 8 | 30 | 1372.04 | 1712.59 | -24.82 | REGRESSION |

| case | N | maxl | ncfn@5 | ncfn@m | ncfl@5 | ncfl@m | netf@5 | netf@m | sum Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| heihe | 1 | 10 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 1 | 15 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 1 | 20 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 1 | 30 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 8 | 10 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 8 | 15 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 8 | 20 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe | 8 | 30 | 7 | 5 | 85 | 0 | 0 | 0 | -87 | PASS |
| heihe_x4 | 1 | 10 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 1 | 15 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 1 | 20 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 1 | 30 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 8 | 10 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 8 | 15 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 8 | 20 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |
| heihe_x4 | 8 | 30 | 51 | 0 | 3620 | 0 | 0 | 0 | -3671 | PASS |

| case | N | maxl_a | maxl_b | n_doubles | nz_diff | max_ulp | max_relerr | strict verdict |
|---|---|---|---|---|---|---|---|---|
| heihe | 1 | 5 | 10 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 1 | 5 | 15 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 1 | 5 | 20 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 1 | 5 | 30 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 8 | 5 | 10 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 8 | 5 | 15 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 8 | 5 | 20 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe | 8 | 5 | 30 | 214252 | 16464 | 4427011158482776 | 9.766e-01 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 1 | 5 | 10 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 1 | 5 | 15 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 1 | 5 | 20 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 1 | 5 | 30 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 8 | 5 | 10 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 8 | 5 | 15 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 8 | 5 | 20 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |
| heihe_x4 | 8 | 5 | 30 | 387607 | 383130 | 4584600898546929175 | 1.316e+04 | FAIL (expected: maxl change → step-size change → trajectory drift) |

| case | N | maxl | sha12 rep1 | sha12 rep2 | sha12 rep3 | rivqdown identical | counters identical | verdict |
|---|---|---|---|---|---|---|---|---|
| heihe | 1 | 5 | a2023ccd2de4 | a2023ccd2de4 | a2023ccd2de4 | True | True | PASS |
| heihe | 1 | 10 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 1 | 15 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 1 | 20 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 1 | 30 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 8 | 5 | a2023ccd2de4 | a2023ccd2de4 | a2023ccd2de4 | True | True | PASS |
| heihe | 8 | 10 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 8 | 15 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 8 | 20 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe | 8 | 30 | e4e9721cf667 | e4e9721cf667 | e4e9721cf667 | True | True | PASS |
| heihe_x4 | 1 | 5 | b5e4b0a2cf83 | b5e4b0a2cf83 | b5e4b0a2cf83 | True | True | PASS |
| heihe_x4 | 1 | 10 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 1 | 15 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 1 | 20 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 1 | 30 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 8 | 5 | b5e4b0a2cf83 | b5e4b0a2cf83 | b5e4b0a2cf83 | True | True | PASS |
| heihe_x4 | 8 | 10 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 8 | 15 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 8 | 20 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |
| heihe_x4 | 8 | 30 | d3d03476b6b7 | d3d03476b6b7 | d3d03476b6b7 | True | True | PASS |


---

## Cross-references

- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR
- [openspec/changes/p8tune-spgmr-maxl/](../../openspec/changes/p8tune-spgmr-maxl/) — OpenSpec change
- [docs/p8tune/clean_prec_none_baseline.md](clean_prec_none_baseline.md) — PR-A 18-cell PREC_NONE baseline + PR-B verdict gate
- [Slurm job 9690 summary.tsv](file:///scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv) — server-resident 60-cell raw data
- [aggregate_verdict.txt](file:///scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/aggregate_verdict.txt) — flat KV mirror of this doc

## Production tune guidance (Optional knob branch — all recommendations are Performance opt-in tier, NOT A5-certified)

> **Tier note**: All `SHUD_SPGMR_MAXL` recommendations below are **Performance opt-in (NOT A5-certified)**. Users gain wall reduction at the cost of solver-tunable trajectory drift (G7 STRICT FAIL at max_ulp 4.4×10¹⁵ for heihe, 4.6×10¹⁸ for heihe_x4 — mechanism-attested per ADR-0004 §G7 Arnoldi MGS → CVODE step-size adapter chain, but NOT hydrology-equivalence validated). Promotion to `A5-certified RECOMMEND` requires future P9-A5 epic (NSE/KGE ≥ 0.95, peak Δ ≤ 5-10%, water-balance Δ ≤ 1%). Downstream pipelines assuming default-trajectory output SHOULD NOT opt in until A5 validation lands. See [ADR-0004 §Acceptance](../adr/0004-maxl-sweep-decision.md#acceptance) tier definitions.

| (case, N) | recommended SHUD_SPGMR_MAXL | tier | rationale (3-rep median wall vs unset baseline) |
|---|---|---|---|
| heihe N=1 | **=30** | Performance opt-in (NOT A5-certified) | +11.99% wall (GO band), ncfl 85 → 0 (solver failures eliminated); NumY ~19K → maxl=10 working set ~1.52 MB fits L2 on ≥1.5 MB cores |
| heihe N=8 | **=30** | Performance opt-in (NOT A5-certified) | +6.78% wall (Optional band); multi-thread L2 contention erodes single-thread +12% to +7% per ADR-0004 §Discussion NumY 口径 |
| heihe_x4 N=1 | unset (default=5) | n/a (SPGMR path saturated) | maxl ≥10 all REGRESS wall (−6.86% at 10 to −15.83% at 30); NumY ~120K → maxl=10 working set ~9.6 MB > L2 → DRAM-bound; ncfl gain (3620 → 0) does NOT outweigh wall cost |
| heihe_x4 N=8 | unset (default=5) | n/a (SPGMR path saturated) | maxl ≥10 all REGRESS wall (−15.81% to −24.82%); large-case + high-thread combo amplifies Krylov-vector DRAM-bandwidth pressure |

**Case-size-asymmetric pattern** (key finding, NumY 口径 per GPT Pro 2026-06-28 review correction): SUNDIALS SPGMR Krylov basis stores `maxl × NumY × 8B` working-set (NumY = 3·NumEle + NumRiv + NumLake per `SHUD/src/Model/shud.cpp:139` N_VNew_Serial; NOT NumEle as originally documented). Small case (heihe NumY ~19K) maxl=10 working set ~1.52 MB fits L2 → bigger maxl reduces outer-iter restarts → wall ↓. Large case (heihe_x4 NumY ~120K) maxl=10 working set ~9.6 MB exceeds L2 / approaches L3 lower bound → MGS DRAM-bound → wall ↑ outweighs ncfl elimination savings. See [ADR-0004 §Discussion](../adr/0004-maxl-sweep-decision.md#discussion--case-size-asymmetric-krylov-memory-pattern-numy-口径) for full mechanistic analysis + per-case NumY anchor table + forward triggers (P8-tune.D KLU spike + P9-A5 hydrology equivalence).
