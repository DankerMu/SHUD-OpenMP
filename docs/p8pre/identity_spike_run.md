---
title: p8pre Step 2 PR-E — identity-precond 18-cell spike execution log
epic: SHUD-OpenMP#338
issue: SHUD-OpenMP#346
SHUD_pin: 5276167
outer_pin: f800bb21d92daaf81d1cbe18cdfddc0a9649eb9e
date: 2026-06-27
spike_phase: Step 2 PR-E (data capture)
verdict_adjudication: PR-F SHUD-OpenMP#347 (not in scope here)
---

# p8pre Step 2 PR-E — identity-precond 18-cell spike execution log

## §1 Purpose

Capture 18-cell identity-precond integration data for downstream PR-F #347
(4 hard-gate + 2 soft-gate verdict) and PR-G #348 ADR-0003 adjudication.
NOT a verdict doc — neutral data + provenance only. Verdict adjudication
per spec L74-113 hard/soft gate criteria is owned by PR-F #347.

## §2 Build provenance

- Server: 210.77.77.22, Slurm CPU partition, cn14 (heihe) + cn15 (heihe_x4)
- Compiler: `gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0`
- Linked libs (`ldd shud_omp`):
  - `libsundials_cvode.so.6 => /scratch/frd_muziyao/SHUD-OpenMP/SHUD/InstallSundials/lib/libsundials_cvode.so.6`
  - `libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1`
- Build flags: `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`
- SHUD pin: `5276167` (PR-D impl, identity precond stub + cvode_config PREC_LEFT)
- Outer pin: `f800bb21d92daaf81d1cbe18cdfddc0a9649eb9e`
  (feat/issue-346-p8pre-pr-e-server-spike)

Provenance log archived at `/tmp/p8pre_identity_spike/server_build_provenance.log`.

## §3 Gate-1 evidence (server nm 3-symbol)

3-symbol nm verify on server-built binary at
`/tmp/p8pre_identity_spike/server_nm.log`:

```
                 U CVodeSetPreconditioner
0000000000021830 T PSetupIdentity
0000000000021880 T PSolveIdentity
```

`PSetupIdentity` and `PSolveIdentity` are defined (`T` symbols) and
`CVodeSetPreconditioner` is correctly resolved against the linked
`libsundials_cvode.so.6` (`U` symbol — undefined-in-binary, linker-pulled).

PR-F adjudication: **gate 1 (server build PASS + 3-symbol linked)
— evidence CAPTURED**.

## §4 18-cell run table

Slurm submit batch (18 jobs at JID 9531..9548) executed 2026-06-27, partition
CPU, cn14 (heihe N1..N8) + cn15 (heihe_x4 N1..N8). All 18 jobs ExitCode 0.

Raw per-cell data source: `.review-evidence/p8pre-pr-e-spike/cell_stats.txt`
(9-column tab table) cross-referenced with
`/tmp/p8pre_identity_spike/jid_table.txt` (JID-cell mapping).

| JID | case | N | rep | nst | nfe | ncfn | nps | npe | t_wall_total (s) | t_precond_setup (s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9531 | heihe | 1 | 1 | 6599 | 6696 | 6 | 18163 | 77 | 137.079454928 | 0.000008872 |
| 9532 | heihe | 1 | 2 | 6599 | 6696 | 6 | 18163 | 77 | 137.273595557 | 0.000008923 |
| 9533 | heihe | 1 | 3 | 6599 | 6696 | 6 | 18163 | 77 | 121.619109402 | 0.000008255 |
| 9534 | heihe | 4 | 1 | 6599 | 6696 | 6 | 18163 | 77 | 94.048899532 | 0.000008824 |
| 9535 | heihe | 4 | 2 | 6599 | 6696 | 6 | 18163 | 77 | 93.846238023 | 0.000008767 |
| 9536 | heihe | 4 | 3 | 6599 | 6696 | 6 | 18163 | 77 | 93.499076194 | 0.000009155 |
| 9537 | heihe | 8 | 1 | 6599 | 6696 | 6 | 18163 | 77 | 88.216195572 | 0.000008812 |
| 9538 | heihe | 8 | 2 | 6599 | 6696 | 6 | 18163 | 77 | 87.621432953 | 0.000008766 |
| 9539 | heihe | 8 | 3 | 6599 | 6696 | 6 | 18163 | 77 | 88.037626219 | 0.000008885 |
| 9540 | heihe_x4 | 1 | 1 | 6569 | 6775 | 47 | 37695 | 158 | 1491.049945147 | 0.000014313 |
| 9541 | heihe_x4 | 1 | 2 | 6569 | 6775 | 47 | 37695 | 158 | 1428.286663562 | 0.000013806 |
| 9542 | heihe_x4 | 1 | 3 | 6569 | 6775 | 47 | 37695 | 158 | 1273.799670059 | 0.000013011 |
| 9543 | heihe_x4 | 4 | 1 | 6569 | 6775 | 47 | 37695 | 158 | 858.281215947 | 0.000014087 |
| 9544 | heihe_x4 | 4 | 2 | 6569 | 6775 | 47 | 37695 | 158 | 858.296898825 | 0.000012654 |
| 9545 | heihe_x4 | 4 | 3 | 6569 | 6775 | 47 | 37695 | 158 | 847.850103372 | 0.000013947 |
| 9546 | heihe_x4 | 8 | 1 | 6569 | 6775 | 47 | 37695 | 158 | 748.337365513 | 0.000013830 |
| 9547 | heihe_x4 | 8 | 2 | 6569 | 6775 | 47 | 37695 | 158 | 747.720080245 | 0.000014535 |
| 9548 | heihe_x4 | 8 | 3 | 6569 | 6775 | 47 | 37695 | 158 | 749.529491602 | 0.000014242 |

## §5 Preliminary wall-vs-baseline observation (informational)

PR-F #347 owns gate-4 verdict per spec L92-100; this section is observational
only.

Step 1 PR-A baseline `wall_step1_baseline_median(case, N)` cited from
`docs/p8pre/n8_profile_baseline.md` §5.1 Table 1 (median of 3 reps per cell).
Identity-spike median computed from §4 raw data (middle of 3 sorted
`t_wall_total` per cell).

| case | N | baseline_wall_median (s) | identity_wall_median (s) | delta (s) | delta_pct |
|---|---:|---:|---:|---:|---:|
| heihe | 1 | 140.797 | 137.079 | −3.718 | −2.64% |
| heihe | 4 | 95.734 | 93.846 | −1.888 | −1.97% |
| heihe | 8 | 89.732 | 88.038 | −1.694 | −1.89% |
| heihe_x4 | 1 | 1412.895 | 1428.287 | +15.392 | +1.09% |
| heihe_x4 | 4 | 849.704 | 858.281 | +8.577 | +1.01% |
| heihe_x4 | 8 | 743.552 | 748.337 | +4.785 | +0.64% |

Per spec L92-100 gate 4: `|wall_identity_median - wall_step1_baseline_median| /
wall_step1_baseline_median ≤ ε(case)` where `ε(heihe) = 0.10` and
`ε(heihe_x4) = 0.05`. All observed |delta_pct| are well below the respective
thresholds (max heihe = 2.64% vs 10% budget; max heihe_x4 = 1.09% vs 5%
budget). PR-F #347 will compute the formal verdict.

## §6 Cross-N CVODE invariance observation

For each case, `nst + nfe + ncfn + nps + npe` SHALL be identical across
N=1/4/8 (RHS work invariance within Mode C strict OMP scope per Step 1 PR-A
baseline observation).

- heihe: nst=6599, nfe=6696, ncfn=6, nps=18163, npe=77 — identical across
  N=1/4/8 × 3 reps (9 cells) — **PASS within identity run**
- heihe_x4: nst=6569, nfe=6775, ncfn=47, nps=37695, npe=158 — identical
  across N=1/4/8 × 3 reps (9 cells) — **PASS within identity run**

NB (soft gate 5 SHA12 strict baseline reasoning): identity-precond
integration `nst/nfe` DIFFERS from Step 1 PR-A PREC_NONE baseline:

| case | quantity | Step 1 PR-A (PREC_NONE) | Step 2 PR-E (PREC_LEFT + identity) |
|---|---|---:|---:|
| heihe | nst | 6698 | 6599 |
| heihe | nfe | 6943 | 6696 |
| heihe_x4 | nst | 6575 | 6569 |
| heihe_x4 | nfe | 6741 | 6775 |

This shift is **EXPECTED**: switching `PREC_NONE → PREC_LEFT` changes the
CVLS internal state machine (preconditioned vs unpreconditioned SPGMR path,
different solve flow, different Krylov subspace convergence pattern) even
though identity `P^-1 = I` is numerically a no-op on the residual vector.
This observation is documented for PR-F soft gate 5 SHA12-baseline
reasoning — bitwise drift vs Step 1 PR-A `final_t.nc` SHA12 anchor is
expected → max_ulp ≤ 1024 numeric-tolerance fallback path likely; PR-F
adjudicates per spec L102-107.

## §7 ncfn observation (data for PR-F gate 2)

Spec gate 2 (p8precond-zero-identity-spike L74-79) criterion = `ncfn = 0`
strict per cell. Observed data:

| case | ncfn per cell (identical across all 9 cells per case) |
|---|---:|
| heihe | 6 |
| heihe_x4 | 47 |

`ncfn` (CVODE nonlinear convergence failure count, accessed via
`CVodeGetNumNonlinSolvConvFails`) is deterministic across N=1/4/8 and
across 3 reps within each case — non-zero but non-random.

The observed deterministic non-zero ncfn is consistent with identity
preconditioning offering no help to the linear solver (`P^-1 = I` → SPGMR
effectively unpreconditioned), which propagates to CVODE nonlinear Newton
retries on stiff Jacobian dynamics. This matches the analytical prior in
design D5 (identity precond is a wiring smoke-test, not a numerical
acceleration).

PR-F #347 adjudication: **gate 2 verdict (per spec L74-79 strict
criterion = `ncfn = 0`) from this data**.

## §8 Soft gate 6 evidence (data for PR-F)

Per spec L108-113 soft gate 6 operational definition: `t_precond_setup_ns /
wall_ns ≤ 0.05` per cell. Observed extremes across the 18 cells:

| case | ratio_min | ratio_max | budget |
|---|---:|---:|---:|
| heihe | 6.472e-08 | 1.009e-07 | ≤ 5.0e-02 |
| heihe_x4 | 9.599e-09 | 1.944e-08 | ≤ 5.0e-02 |

Order of magnitude: all 18 cells are 6+ orders of magnitude below the 5%
budget (preliminary; PR-F formal verdict). The identity precond stub's
per-call cost is dominated by `PSetupIdentity` Timer RAII overhead alone
(~8-14 μs total cumulative per cell observed in §4 raw `t_precond_setup`
column), consistent with the ~0.18 μs/call observed in PR-D keliya Mac
smoke (npe=77 heihe / 158 heihe_x4 → ~110-90 ns/call amortized).

## §9 References

- Epic: SHUD-OpenMP#338
- Issue: SHUD-OpenMP#346
- PR-D (binary + precond wire): SHUD-OpenMP#357 (merged 4e0bf39)
- Step 1 PR-A baseline doc: `docs/p8pre/n8_profile_baseline.md` (PR-C #355)
- Spec: `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md` L74-113
- Design: `openspec/changes/p8pre-spike/design.md` D5 (4 hard gate + 2 soft gate)
- Forward consumer: PR-F #347 (aggregator + verdict + identity_spike_verdict.md)
- Forward consumer: PR-G #348 (ADR-0003 + p8pre_summary.md)
- SHUD upstream branch: `openmp-baseline-p8pre` HEAD `5276167`
- Raw evidence archive: `/tmp/p8pre_identity_spike/` (18 cell dirs + jid_table.txt + server_nm.log + server_build_provenance.log + cell_stats.txt; 47.78 MB)
- Aggregator input: `.review-evidence/p8pre-pr-e-spike/cell_stats.txt`
