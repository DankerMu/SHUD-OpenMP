# GCC G-E1 spot leg (PR-N1 P1 — cross-toolchain validation)

**Verdict: GCC-SPOT G-E1 PASS.** On the server GCC-13 toolchain, Config C and
Config E, each at N∈{1,8}, on heihe (project `heihe`, NumEle 6335, 90-day
window 14245→14335), produce **bitwise-identical** model outputs:

```
E@N1 == E@N8 == C@N1 == C@N8   (11 files each, sorted-manifest SHA ff3e6b1d…)
```

This validates the hybrid-override mechanism on the GCC production toolchain
(CI `serial-baseline.yml` + the PR-N2 server matrix run GCC), not only Apple
clang — the P1 concern that the clang `optnone` FMA-restoration side effect
"has no reason to transfer to gcc" is resolved: on gcc it does not need to,
because the plain override already bitwise-matches the gcc-built library.

## Why it passes on gcc (instruction-level, `gcc_fma_disasm.txt`)

On x86_64/gcc-13 the vendored `libsundials_nvecserial.so` reductions and our
Config E override symbols are **both scalar non-FMA** (`mulsd`+`addsd`, `fmadd=0`
on both sides; matching op counts — dotprod 2, wsqrsum 3). They fold in the same
sequential scalar order, so they agree regardless of `SHUD_NVEC_NOOPT`. A
5000-dataset numerical sweep on the server confirms 0/5000 divergence for the
plain override, the `optnone`/`optimize("O0")` variant, and the
`optimize("O0","no-tree-vectorize")` variant alike (see the parent
`optnone_fold_order.evidence.txt` §4–§5).

Contrast with Apple clang / ARM (parent evidence §2–§3): there the library uses
scalar **`fmadd`** while SHUD's `-ffp-contract=off` strips FMA and `-O2`
vectorizes the override → the plain override diverges (dot 4785/5000) and
`SHUD_NVEC_NOOPT` (clang `optnone`) is REQUIRED to restore the scalar-FMA fold.

## Reproduction

- Binaries: `.p12-nvec-runs/gcc_spot_bin/shud_omp_{C,E}` built on the server at
  SHUD `p12-nvec` with `HYPRE_INCDIR=/scratch/frd_muziyao/local/hypre-3.1.0/include`,
  `OPENBLAS_LIBDIR=/scratch/frd_muziyao/local/openblas/lib`, MPI libdir on
  `LIBRARY_PATH` (see `gcc_spot_bin/env.sh`).
- Run: `.p12-nvec-runs/gcc_spot.sbatch` — Slurm, partition CPU, `--exclusive`,
  from `/scratch`, `--output`/`--error` on `/scratch` (三铁律). Each cell:
  `SHUD_LINSOL=spgmr`, `SHUD_RHS_THREADS` **unset**, `OMP_NUM_THREADS`=N,
  `OMP_PROC_BIND=close`, `OMP_PLACES=cores`. Manifest = sha256 of every
  model-output file except `diag_*.csv`, `nvec_prof.csv`, `profile_B0.yaml`,
  `*.time.csv`, `*.time`.

## Files

| file | what |
|------|------|
| `gcc_spot_run.log` | full Slurm run: 4 cells (C/E × N1/N8) + compare + verdict |
| `sha/{C,E}_n{1,8}.sha` | per-cell model-output SHA manifests (11 files each, all identical) |
| `gcc_fma_disasm.txt` | gcc disassembly: override symbols + library both scalar non-FMA |

Note: the spot-leg binaries were built at SHUD `3e0ab23` (the `optimize("O0")`
gcc attribute); the shipped commit strengthens the gcc attribute to
`optimize("O0","no-tree-vectorize")`, which was verified to compile on gcc-13
and to leave the override symbols scalar non-FMA (identical fold) — i.e. the
spot-leg bitwise result is unchanged by that comment/attribute refinement, since
the attribute is correctness-inert on gcc.
