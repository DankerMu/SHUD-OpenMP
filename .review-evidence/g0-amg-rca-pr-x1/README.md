# g0-amg-rca-pr-x1 — PR-X1 AMG tolerance + EpsLin sanity spike

Slurm job 10248 (8-cell array, heihe_x4 90-day, N=1 each).

## Verdict

```
MARKER:PR_X1_VERDICT_BEGIN
best_cell=0
best_ncfn=98286
best_nst_ratio=38.3267
best_wall_ratio=17.8149
amg_reopens=false
num_present=8
num_amg_ok=5
spgmr_baseline_ncfn=49
spgmr_baseline_nst=6572
spgmr_baseline_wall_sec=1566.5518531799316
MARKER:PR_X1_VERDICT_END
```

## Matrix outcome

| Cell | AMG_TOL | EpsLin   | State     | nst     | ncfn    | ncfl | wall_total_s | Node    |
|------|---------|----------|-----------|---------|---------|------|--------------|---------|
| 0    | 1e-7    | default  | COMPLETED | 251,883 | 98,286  | 0    | 27,908       | cn23 (7-way contended) |
| 1    | 1e-9    | default  | COMPLETED | 253,558 | 99,095  | 0    | 28,117       | cn23 (7-way contended) |
| 2    | 1e-11   | default  | TIMEOUT   | NA      | NA      | NA   | >28,800      | cn23 (7-way contended) |
| 3    | 1e-13   | default  | TIMEOUT   | NA      | NA      | NA   | >28,800      | cn23 (7-way contended) |
| 4    | 1e-7    | 0.005    | COMPLETED | 257,610 | 104,795 | 0    | 28,142       | cn23 (7-way contended) |
| 5    | 1e-9    | 0.005    | COMPLETED | 256,525 | 103,970 | 0    | 28,553       | cn23 (7-way contended) |
| 6    | 1e-11   | 0.005    | TIMEOUT   | NA      | NA      | NA   | >28,800      | cn23 (7-way contended) |
| 7    | 1e-13   | 0.005    | COMPLETED | 256,138 | 104,390 | 0    | 22,871       | cn08 (alone) |

G0 baseline reference (PR-A, heihe_x4 alone): nst=254,756, ncfn=100,138, wall=23,660s.
SPGMR baseline reference (PR-0, heihe_x4): nst=6,572, ncfn=49, wall=1,566.55s.

## Hypothesis test

H0 (refuted): "ncfn=100k explosion in G0 AMG is caused by loose Hypre solve tol + CVODE EpsLin
mismatch → Hypre 'early-outs' → bad Newton correction → Newton fails → step retries."

Across 4 orders-of-magnitude in `SHUD_AMG_TOL` (1e-7 → 1e-13) and 10× change in `SHUD_CVODE_EPSLIN`
(0.05 default → 0.005), `ncfn` stays in a tight 98,286-104,795 window. The Hypre/CVODE tolerance
pair has effectively zero leverage on outer Newton control failure rate.

`ncfl = 0` across the board (Hypre inner solve never fails).

Conclusion: the ncfn explosion is intrinsic to CVODE outer Newton + step controller when AMG
replaces SPGMR — NOT a tolerance mismatch. AMG path cannot be saved by inner-solve tuning.
Triggers PR-X2 P8 retrospective ADR-0008 closure.

## Limitations

- 3 cells (2, 3, 6) hit 8h Slurm timelimit on contended cn23 (7 cells sharing one node + sparse
  AMG is memory-bandwidth-bound). Their data would have been at least as bad as cell-0 (1e-7,
  default), since they used tighter AMG_TOL which strictly increases per-solve Hypre iters
  without reducing ncfn (cell-7 evidence). Loss does not change the verdict.
- All cells AMG-OK on completion (no integration crash); the failure is wall + ncfn, not
  correctness.
