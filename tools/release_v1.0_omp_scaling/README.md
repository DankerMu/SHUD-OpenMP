# Release v1.0 OMP scaling validation harness

**Purpose:** Author the serial-vs-parallel scaling evidence table for
[`RELEASE.md`](../../RELEASE.md) v1.0 by measuring `heihe_x4` (40,046
elements) wall time and CVODE statistics at N in {1, 2, 4, 8, 16} OMP
threads and gating each non-baseline N through the A5 hydrology-
acceptance pipeline (`tools/a5/`) against the N=1 reference.

**Scope lock:** heihe_x4 only, 90-day truncation, SPGMR production
baseline (no research env hooks). Reference tree = N=1 cell output.

## Files

```
tools/release_v1.0_omp_scaling/
+-- scaling_array.sbatch     # 5-cell Slurm array driver (server only)
+-- run_scaling_cell.sh      # single-cell runner (dispatched by sbatch;
|                            #   also Mac-smokeable with SHUD_BIN override)
+-- run_a5_scaling.sh        # A5 fanout: per-N vs N=1 reference
+-- aggregate_scaling.sh     # emits MARKER:RELEASE_V1_0_SCALING_VERDICT
+-- README.md                # this file
```

## Full pipeline (server)

Slurm 三铁律 compliance is critical -- see `CLAUDE.md`. All artifacts
live under `/scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/<RUN_ID>/`.

```bash
# 1) Submit the 5-cell array (node-exclusive, --cpus-per-task=16).
RUN_ID="release-v1.0-scaling-$(date -u +%Y%m%d-%H%M%S)"
mkdir -p /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}
cd      /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}
sbatch --export=ALL,RUN_ID=${RUN_ID} \
    /scratch/frd_muziyao/SHUD-OpenMP/tools/release_v1.0_omp_scaling/scaling_array.sbatch

# 2) After all 5 array tasks COMPLETED, run A5 fanout (login node is fine
#    -- uv-managed Python + reads N=1 reference against N=2/4/8/16).
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/release_v1.0_omp_scaling/run_a5_scaling.sh \
    /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}

# 3) Emit the aggregate MARKER block (feeds RELEASE.md scaling table).
bash /scratch/frd_muziyao/SHUD-OpenMP/tools/release_v1.0_omp_scaling/aggregate_scaling.sh \
    /scratch/frd_muziyao/SHUD-OpenMP/.release-v1.0-scaling-runs/${RUN_ID}
```

## Expected outcome

Per ADR-0010, P1e StrictOMP RHS is the production baseline. The single-
point P1e capstone reported ~1.7x wall speedup on heihe_x4 without a
full N sweep. This harness is expected to yield, on a node-exclusive
run:

- N=1 wall: ~1600 s (matches the ADR-0007/ADR-0008 SPGMR heihe_x4
  baseline of 1566.5 s)
- N=8 speedup: 3-5x (target `PRODUCTION_APPROVED`, i.e. >= 2.5x)
- N=16 speedup: 4-8x (subject to memory-bandwidth ceiling)
- All N: A5 verdict PASS (trajectory equivalence preserved -- P1e RHS
  is deterministic under strict tolerance, so thread count MUST NOT
  change the discharge trajectory beyond A5's numerical tolerance)

Sub-1.5x speedup at N=8 with A5=PASS still produces `CONDITIONAL`
verdict (safe but limited parallel benefit). Any N with A5=FAIL flips
the verdict to `BLOCKED` (thread count broke trajectory equivalence --
must not ship).

## MARKER block schema

`aggregate_scaling.sh` emits (illustrative fields; `<value>` placeholders):

```
MARKER:RELEASE_V1_0_SCALING_VERDICT_BEGIN
case=heihe_x4
cfg_reltol=1e-4
cfg_period_days=90
n1_wall_total_sec=<value>
n1_wall_setup_sec=<value>
n1_wall_solve_sec=<value>
n1_nst=<value>
n1_ncfn=<value>
n1_verdict_class=SPGMR_OK|SHUD_NONZERO_EXIT|SCALING_WALL_OVERFLOW|MALFORMED
n2_wall_total_sec=<value>
n2_wall_setup_sec=<value>
n2_wall_solve_sec=<value>
n2_nst=<value>
n2_ncfn=<value>
n2_verdict_class=<...>
n2_speedup_total=<n1/n2>
n2_speedup_solve=<n1_solve/n2_solve>
n2_parallel_efficiency=<speedup_total / 2>
n2_a5_verdict=PASS|FAIL|UNKNOWN
n2_a5_weighted_score=<0.0-1.0 or NA>
n4_wall_total_sec=<...>
... (n4, n8, n16 identical shape)
overall_scaling_verdict=PRODUCTION_APPROVED|CONDITIONAL|BLOCKED
overall_reasoning=<one-line explanation>
MARKER:RELEASE_V1_0_SCALING_VERDICT_END
```

Decision matrix (implemented in `aggregate_scaling.sh` bottom):

| Any A5=FAIL | Any A5=UNKNOWN | N=8 speedup | Verdict                 |
|-------------|----------------|-------------|-------------------------|
| YES         | -              | -           | BLOCKED                 |
| NO          | YES            | -           | BLOCKED                 |
| NO          | NO             | >= 2.5      | PRODUCTION_APPROVED     |
| NO          | NO             | 1.5-2.5     | CONDITIONAL             |
| NO          | NO             | < 1.5       | CONDITIONAL (weak-gain) |

## Production environment enforcement

Both `scaling_array.sbatch` and `run_scaling_cell.sh` explicitly:
- `unset SHUD_AMG_TOL SHUD_CVODE_EPSLIN SHUD_CVODE_RELTOL` (all research hooks off)
- `export SHUD_LINSOL=spgmr` (production selector)
- `export OMP_PROC_BIND=close`
- `export OMP_PLACES=cores`

This guarantees the scaling numbers reflect what a production user gets
(no research configuration leakage between array tasks).

## Local Mac smoke (keliya, tooling wiring only)

`heihe_x4` is server-only (40,046 elements + 32 GiB memory envelope
too heavy for a Mac); the tooling was smoke-tested against `keliya`
(484 elements, ~30 s runtime) locally to verify the runner script
wires correctly. Numbers are not comparable to server heihe_x4 -- see
the release commit body for the smoke evidence.

To reproduce a Mac smoke:

```bash
cd /path/to/openMP/SHUD
# Override the hardcoded heihe_x4 case via a quick shim, since
# run_scaling_cell.sh is scoped to heihe_x4 production only:
OMP_NUM_THREADS=1 SHUD_LINSOL=spgmr ./shud_omp keliya  # sequential
OMP_NUM_THREADS=2 SHUD_LINSOL=spgmr ./shud_omp keliya  # 2-thread
```

The runner itself is scoped to `heihe_x4` because production scaling
evidence for `RELEASE.md` must be single-basin comparable.

## References

- [`RELEASE.md`](../../RELEASE.md) -- v1.0 production manifest (this
  harness authors the scaling replacement for the single-point 1.7x
  claim)
- [`docs/adr/0010-cpu-acceleration-status-consolidation.md`](../../docs/adr/0010-cpu-acceleration-status-consolidation.md)
  -- CPU acceleration status decision archive
- [`docs/p1e/p1e_academic_summary.md`](../../docs/p1e/p1e_academic_summary.md)
  -- P1e RHS academic summary (single-point 1.7x source)
- [`tools/a5/README.md`](../a5/README.md) -- A5 pipeline contract
- [`tools/p9.spot/p9_spot_array.sbatch`](../p9.spot/p9_spot_array.sbatch)
  -- template (2-cell sbatch array)
- [`tools/p8tune.G0-rca/amg_rca_array.sbatch`](../p8tune.G0-rca/amg_rca_array.sbatch)
  -- template (8-cell array with --exclusive)
- [`CLAUDE.md`](../../CLAUDE.md) -- project-wide constraints, Slurm 三铁律
