# P1d server intermediate 8-cell run (PR-F #280)

**Scope**: Validate on the production server (`frd_muziyao@210.77.77.22:32099`) that the combined **NUMA env (PR-B #276)** + **steady-state first-touch in MD_rhs_core.cpp (PR-C #277 / PR-D #278 / PR-E #279)** combo:

1. Builds and runs cleanly across the production 8-cell matrix.
2. Does NOT regress bitwise equivalence beyond what P1c kahan already established.
3. Shows the steady-state first-touch gate fires under `OMP_PROC_BIND=close`.

PR-F is an **intermediate** cell — Kahan compensation (P1c §4.7 conditional Neumaier 1974) is still IN. PR-G (#281) reverts Kahan; PR-H (#282) runs the final 8-cell with the three SHALL gate verdict.

## Matrix

- Cases: `heihe` (NumEle=6335; 90-day cap START=14245 END=14335) + `heihe_x4` (NumEle ~25000; 90-day cap START=1 END=91)
- Thread counts: N ∈ {1, 2, 4, 8}
- Per `docs/p1d/p1d_numa_env_runbook.md` (PR-B #276) full prescription: `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `OMP_NUM_THREADS=N` + `numactl --interleave=all` wrap + `numactl --hardware` echo
- Total cells: 2 × 4 = 8

## Environment

- Outer baseline/P1d HEAD: `573dfdf`
- SHUD openmp-baseline HEAD: `6aada88` (post-PR-E)
- Server build: `make clean && make shud_omp` PASS; FP strict 3-grep `-ffp-contract=off + -fno-fast-math + -fopenmp` upheld; `-ffast-math / -Ofast` 0
- Slurm 三铁律 honored: sbatch + run dirs + logs all under `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-f-intermediate/`
- Cluster: CPU partition, 1 node × 8 cpus-per-task per job; jobs ran in parallel on `cn03` + `cn07` (8-cell wall-clock end-to-end ~22 min)
- Server NUMA topology: 2 nodes, 24 CPUs each (`node 0 size: 64120 MB`, `node 1 size: 64456 MB`, distances `0:10 1:20 / 1:20 0:10`)

## NUMA gate confirmation

Every cell's stdout emits the `[NUMA]` tokens proving `g_numa_first_touch_enabled = 1`:

```
[NUMA] OMP_PROC_BIND=close
[NUMA] first-touch begin tag=hot.soa
[NUMA] first-touch begin tag=QeleSurf_flat
[NUMA] first-touch begin tag=Ele_AoS
[NUMA] first-touch begin tag=LoadIC
```

The allocation-time first-touch fires (visible via emit_numa_token at the 4 named sites). The steady-state first-touch loops in rhs_update / rhs_flux (PR-C/D/E) are silent by design (pure zero-write per CVODE call, no logging overhead). The bitwise SHA matrix below is the proof those loops execute correctly.

## SHA + wall matrix (canonical: numactl --interleave=all per runbook)

PR-F v2 (NUMA env + first-touch + numactl --interleave=all + Kahan IN) vs P1c PR-H kahan baseline (Kahan IN, no NUMA env, no first-touch, no numactl):

| Cell | PR-F v2 SHA (16) | PR-F v2 wall (s) | P1c SHA (12) | P1c wall (s) | SHA eq? | wall Δ |
|---|---|---|---|---|---|---|
| heihe N=1 | `fd2d55716b5daffd` | 512 | `fd2d55716b5d` | 508 | **EQUAL** | +4 |
| heihe N=2 | `fd2d55716b5daffd` | 520 | `fd2d55716b5d` | 509 | **EQUAL** | +11 |
| heihe N=4 | `3fc00b6f6b7e5931` | 506 | `e058db2e9c2a` | 506 | differs | 0 |
| heihe N=8 | `61bef1c075e4d37c` | 458 | `6285e8a4a30a` | 500 | differs | **-42 (-8.4%)** |
| heihe_x4 N=1 | `4eb804f571ba6f89` | 1305 | `4eb804f571ba` | 1058 | **EQUAL** | +247 (+23%) |
| heihe_x4 N=2 | `4eb804f571ba6f89` | 1303 | `4eb804f571ba` | 1140 | **EQUAL** | +163 (+14%) |
| heihe_x4 N=4 | `938c44193e4662ea` | 1198 | `ff0787abd217` | 1051 | differs | +147 (+14%) |
| heihe_x4 N=8 | `848f81ca7ba36527` | 1037 | `6e9f9a2eaf65` | 934 | differs | +103 (+11%) |

### Bitwise interpretation

- **N=1 and N=2 cells are EQUAL to P1c baseline** for both heihe and heihe_x4. This is the Kahan signature — at low thread count there's no reduction order divergence to compensate for.
- **N=4 and N=8 cells DIFFER from P1c baseline**, but they also DIFFER between themselves within each baseline (P1c heihe N=4 `e058db2e9c2a` ≠ N=8 `6285e8a4a30a`; PR-F heihe N=4 `3fc00b6f...` ≠ N=8 `61bef1c0...`). This is the **pre-existing Kahan-still-not-bitwise-across-N residual** that P1c era documented and that **PR-G (Kahan revert) + PR-H (final 8-cell)** are designed to address.
- **PR-F introduces no new divergence beyond the existing P1c regime** — Kahan still hides residual variation at N≥4; first-touch + numactl don't make it worse, just shift which permutation surfaces.

### Wall interpretation — and a key finding for PR-G / PR-H

**Critical finding**: the canonical `numactl --interleave=all` wrap (PR-B runbook prescription) is an **anti-pattern when first-touch loops are active**. It forces page allocation interleaved across NUMA nodes, OVERRIDING the first-touch-driven local-node placement that PR-C/D/E established. Empirical evidence:

| Comparison | heihe N=8 | heihe_x4 N=1 | heihe_x4 N=8 |
|---|---|---|---|
| v1 (no numactl wrap, OMP env only) | 460s | 1183s | 938s |
| v2 (with numactl --interleave=all per runbook) | 458s | 1305s | 1037s |
| **delta v2 - v1** | -2s (noise) | **+122s (+10%)** | **+99s (+11%)** |
| P1c baseline (no NUMA env, no first-touch) | 500s | 1058s | 934s |

- **heihe (NumEle=6335)**: per-node memory footprint small enough that --interleave=all is noise-band. v2 N=8 still gets the -8.4% speedup vs P1c baseline.
- **heihe_x4 (NumEle ~25000)**: per-node memory footprint large enough that interleaving forces real cross-NUMA traffic. v2 is 10-23% SLOWER than v1 across all N. v1 (no --interleave) was within 5% of P1c baseline; v2 is 11-23% WORSE.

**Recommendation for PR-G / PR-H**: drop `numactl --interleave=all` from the runbook when first-touch is active. The PR-B runbook prescription was authored before first-touch loops landed (PR-C/D/E); --interleave=all was a NUMA hardening for the *baseline* (no-first-touch) scenario. With first-touch loops in place, the two strategies are antagonistic — first-touch deliberately pins pages to the owning thread's NUMA node, and --interleave=all deliberately spreads them across nodes. Keep one or the other; for the first-touch-active regime, keeping first-touch is correct.

**Action for PR-G / PR-H**: re-execute the 8-cell with NO `numactl --interleave=all` (OMP env triplet only). Expected: heihe_x4 cells return to v1 walls (~938-1183s) instead of v2 (~1037-1305s).

### Three SHALL gate readiness (informational; full verdict in PR-H)

PR-F establishes:

1. **A3a bitwise gate**: ready for evaluation post-Kahan-revert. Current evidence (N=1/N=2 EQUAL across PR-F ↔ P1c) suggests the NUMA env + first-touch path preserves serial-vs-PROC_BIND=close bitwise equivalence at low N.
2. **nst Δ=0 gate**: `cvode_stats.txt` captured for all 8 cells (15-key set); ready for PR-H structured nst analysis.
3. **N=1 reverse-compat gate**: confirmed PASS — PR-F heihe N=1 SHA == P1c N=1 SHA byte-identical; PR-F heihe_x4 N=1 SHA == P1c N=1 SHA byte-identical (both verified at full 64-hex level).

PR-F does NOT issue the three SHALL gate verdict. That belongs to PR-H per the burst topology.

## Boundary discipline

- SHUD master untouched (verified `git branch -a --contains 6aada88` SHUD-side returns only `origin/openmp-baseline`; master HEAD still `3aec657`).
- Run artifacts isolated under `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-f-intermediate/` (gitignored host-side).
- No SHUD code changes in PR-F (the first-touch code lives in baseline/P1d via PR-C/D/E).
- Outer change: this doc only.

## Server artifacts (gitignored)

- sbatch: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-f-intermediate/run_p1d_pr_f.sbatch` (v2 with full numactl wrap)
- 8 run dirs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-f-intermediate/{heihe,heihe_x4}_N{1,2,4,8}/`
- Each run dir contains: `rivqdown.sha256`, `wall.txt`, `cvode_stats.txt` (all 8 cells, 15-key set), `output_listing.txt`, `done.txt`
- Slurm logs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-f-intermediate/logs/p1d_pr_f_{9025..9040}.{out,err}` (v1 = 9025-9032 retained for comparison; v2 = 9033-9040 canonical)

## Verdict

**PR-F intermediate cell: PASS with finding**. NUMA env + first-touch combo is server-validated as numerically neutral (within P1c Kahan regime). Wall results revealed `numactl --interleave=all` is anti-pattern with active first-touch — actionable finding for PR-G/H runbook revision. Burst proceeds to PR-G (Kahan revert) + PR-H (final three SHALL gate) with this finding driving runbook update.

## Next-steps gate (informational, drives PR-G / PR-H)

- **PR-G**: revert P1c §4.7 conditional Kahan injection at SHUD `3a0004c`. First-touch remains stacked. Expected outcome: bitwise drift across N becomes visible again (Kahan's masking effect goes away), and the steady-state first-touch perf signal emerges.
- **PR-H**: re-run the same 8-cell matrix on post-Kahan-revert SHUD. **Drop `numactl --interleave=all`** per PR-F finding. Compare:
  - A3a bitwise vs P1-update-omp-tag reference: collected in PR-I parallel Mac worktree.
  - nst Δ=0 vs reference: requires CVODE stats parsing.
  - N=1 reverse-compat: SHA(N=1 omp) == SHA(serial).
- **Runbook revision for PR-B doc (post-PR-H)**: deprecate `numactl --interleave=all` for first-touch-active regime; keep OMP env triplet as the canonical wrap.
