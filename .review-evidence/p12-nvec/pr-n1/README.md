# P12-nvec PR-N1 — Config E (hybrid NVector) evidence

Config E = Config C StrictOMP RHS + OpenMP NVector **element-wise** ops +
SHUD-owned **serial reduction overrides** (generic ops table). Build leg:

```
make -C SHUD shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1
```

Epic thesis: parallel NVector element-wise ops with **bitwise identity to
Config C at every thread count** (the Config D reduction-order drift is removed
by running every reduction as a serial generic-API loop).

All legs are LOCAL Mac (Apple Silicon), keliya (NY=1785, 90-day truncation),
`SHUD-System/SHUD @ p12-nvec`. Source: `src/Model/MD_nvec_hybrid.{hpp,cpp}`,
audit `SHUD/docs/p12-nvec/nvec_reduction_audit.md`.

---

## G-E1 HARD gate — PASS

`ge1_bitwise.sh` (verdict in `ge1_run.log`). Config E on keliya at
N∈{1,2,4,8}: model-output SHA manifest per leg, all identical to **each other**
AND to the **Config C baseline** (`../pr-n0/sha/leg-a-baseline.sha`, 13 files).
Manifest excludes the instrumentation artifacts (`diag_*.csv`,
`nvec_prof.csv`, `profile_B0.yaml`, `*.time.csv`) — identical exclusion set to
PR-N0. No tolerance fallback.

| leg | NUM_OPENMP (cfg) | OMP_NUM_THREADS | SHUD_RHS_THREADS | sorted-manifest SHA (first 24) | vs leg-n1 | vs Config C baseline |
|-----|------------------|-----------------|------------------|--------------------------------|-----------|----------------------|
| n1  | 1                | 1               | unset            | `14db6b3c53068314f30c8a0e` | —         | IDENTICAL            |
| n2  | 2                | 2               | unset            | `14db6b3c53068314f30c8a0e` | IDENTICAL | IDENTICAL            |
| n4  | 4                | 4               | unset            | `14db6b3c53068314f30c8a0e` | IDENTICAL | IDENTICAL            |
| n8  | 8                | 8               | unset            | `14db6b3c53068314f30c8a0e` | IDENTICAL | IDENTICAL            |

All four legs and the baseline share manifest SHA `14db6b3c…` (13 files each,
including `keliya.rivqdown.dat` — the primary solver output). **G-E1: PASS**
(Mac Apple clang / ARM).

**Cross-toolchain validation (GCC).** Because the reduction-fold match is
compiler-dependent (see the root-cause section), PR-N1 also carries a server
**GCC-13 G-E1 spot leg** (`gcc_spot_leg/`): Config C and Config E on heihe
(NumEle 6335, 90-day) each at N∈{1,8} → **E@N1 == E@N8 == C@N1 == C@N8**
bitwise (sorted-manifest SHA `ff3e6b1d…`). **GCC-SPOT G-E1: PASS.** This covers
the GCC production toolchain (CI `serial-baseline.yml` + the PR-N2 server
matrix), not only Apple clang.

Equivalence chain: Config C (Serial NVector) is bitwise thread-count-invariant
(P1e StrictOMP RHS + no cross-thread NVector accumulation); the PR-N0
`leg-a-baseline.sha` is that Config C/A reference. Config E reproduces it
because (a) element-wise ops are index-fixed (identical FP order for any thread
count and vs Serial) and (b) every reduction is a serial generic-API loop whose
codegen is pinned to reproduce the library's scalar fold (next section).

### Isolation requirement (evidence-integrity note)

The gate MUST run with **nothing else writing to `keliya.out`**. An earlier
non-isolated invocation overlapped a concurrent keliya run and produced a
spurious leg-n8 manifest mismatch (contaminated `elevprcp/elevnetprcp/rivqdown`
capture). Re-run in isolation, all four legs are identical — the committed
`ge1_run.log` + `sha/leg-n*.sha` are from the clean isolated run.

---

## Root cause + fix: scalar-FMA codegen match (platform-specific to clang/ARM)

**Correction:** an earlier version of this note attributed the divergence to
"`-O2` reassociating the accumulation." That is mechanically wrong — IEEE
FP-add reassociation is illegal at plain `-O2` without `-ffast-math`, and
neither clang nor gcc does it (a `-O2 -ffp-contract=off` loop matches an
`-O0`-same-flags loop 0/N). The true mechanism is FMA-contraction state plus
auto-vectorization, and it is **platform-specific**.

Config C's reference reductions are the **SUNDIALS library** serial functions
(`N_VDotProd_Serial`, `N_VWSqrSumLocal_Serial`, …). SHUD's `-ffp-contract=off`
(B0 IEEE-754 lockdown) forces two codegen choices on our override that the
vendored library (built with **no `-ffp-contract` flag** → default contraction)
does not make:
1. **FMA stripped** — the library fuses `sum += x[i]*y[i]` into a single-rounding
   `fma` on FMA targets; `-ffp-contract=off` splits it into two roundings.
2. **Vectorization** — at `-O2` the non-FMA reduction auto-vectorizes into a
   pairwise/tree fold order ≠ the library's sequential scalar loop.

Measured (`optnone_fold_order.evidence.txt`, override bodies vs library, 5000
datasets, EXACT shipped flags `-O2 -ffp-contract=off -fno-fast-math`):

| variant | Mac (Apple clang / ARM) | server (x86_64 / gcc-13) |
|---------|--------------------------|--------------------------|
| plain override (no attr) | **diverges** — dot 4785/5000, wsqrsum 1819/5000 | **matches** — 0/5000 |
| `SHUD_NVEC_NOOPT` (shipped) | **matches** — 0/5000 | matches — 0/5000 |
| `-ffp-contract=on` only | diverges — 4769/5000 (still vectorized) | — |
| default-contract `-fno-vectorize` | matches — 0/5000 | — |

FMA disassembly (`fmadd`/`fmla` in the reduction loop): Mac clang — library
**scalar `fmadd`**, plain override **non-FMA + vectorized** (mismatch), optnone
override **scalar, FMA restored** (match). Server gcc — library **scalar
non-FMA** (`mulsd`+`addsd`), plain override **scalar non-FMA** (already
matches). So the divergence exists only where the library uses FMA and SHUD's
flag strips it — Apple clang / ARM; on x86_64/gcc both sides are scalar non-FMA
and agree without any attribute.

**Fix (spec-faithful):** each FP-folding override carries `SHUD_NVEC_NOOPT` —
`__attribute__((optnone))` on clang (disables vectorization AND discards the
function-level `-ffp-contract=off`, restoring default contraction → the library's
scalar-FMA fold), `__attribute__((optimize("O0","no-tree-vectorize")))` on gcc
(scalar + default contraction; not required for correctness on x86 but pins the
intent and guards a future gcc that might vectorize). Bodies remain **plain
serial loops over the generic API** (`N_VGetArrayPointer` / `N_VGetLength`) — NO
`N_V*_Serial` call, NO content-struct macro (both spec-prohibited). The attribute
only constrains codegen, not the source logic.

**Honest fragility:** the bitwise-to-C guarantee rests on a **per-compiler
codegen coincidence** — on clang, the `optnone`-restores-contraction side
effect; on gcc, the fact that `-O2 -ffp-contract=off` already yields the
library's scalar non-FMA fold. It does **NOT** rest on the library's `-O` level
(the earlier note's claim). If a future toolchain changes the library's
contraction default or clang's optnone behaviour, the G-E1 SHA gate (Mac clang +
the PR-N2 server gcc matrix + CI GCC) is the backstop. The copy-into-Serial +
library-call alternative is **spec-prohibited** here (no `N_V*_Serial` in
overrides), so this codegen pin is the spec-compliant path. Cross-toolchain
proof: `gcc_spot_leg/` (server gcc-13 G-E1 heihe verdict).

---

## Composition + thread topology (task 2.5)

### (a) PROF × HYBRID leg — PASS

`prof-hybrid-keliya.*` — Config E build + `SHUD_NVEC_PROF=1`, keliya, N=8. The
profiler wraps **after** the hybrid overrides install (overrides first, shims
outermost), so each reduction shim delegates to the override.
`prof-hybrid-keliya.nvec_prof.csv` header reports **`backend=hybrid`**
(NY=1785, nthreads=8). Assert lines (stdout log):

```
[NVEC_HYBRID] clone-propagation smoke assert: base=10 N_VClone=10 N_VCloneEmpty=10 -> PASS
[NVEC_PROF]   clone-propagation smoke assert: base=31 N_VClone=31 N_VCloneEmpty=31 -> PASS
[NVEC_PROF]   PROF x HYBRID composition assert: reduction_slots_wrapped=20 live_is_shim=20 delegate_is_override=20 -> PASS
[NVEC_PROF]   dump ... backend=hybrid NY=1785 nthreads=8  (wrapped_entries=31 invoked(calls>0)=12)
```

The composition assert (`nvec_prof_reduction_delegates_match`, extended per the
nvec-op-profile "composition with hybrid overrides" scenario) verifies, for all
**20** wrapped reduction slots, that the live ops-table entry holds the
profiler shim (≠ stock OpenMP address) AND its captured delegate equals a
SHUD override address (`nvec_hybrid_addr_is_override`).

**Reductions CVODE actually invoked** (`calls > 0`): `nvdotprod` (814,191
calls — SPGMR Gram-Schmidt) and `nvwrmsnorm` (509,659 calls — WRMS error
norm). Both are overridden. The other 18 wrapped reduction slots have
`calls = 0` because CVODE + SPGMR on this problem never invokes them (no masked
norms, constraints, min-quotient; the aliased `*local` kernels are unreached
because CVODE calls the standard slots directly for a single-process NVector).
Reading the spec scenario "every reduction-op row `calls > 0`" as "every
reduction CVODE **invoked** is measured (no unwrapped/clobbered reduction)":
both invoked reductions have `calls > 0` and both delegate to the override —
satisfied. Full CSV committed so a reviewer sees exactly which slots CVODE
drove; element-wise ops (`nvlinearsum` 1.55M, `nvscale` 1.42M calls) stay
stock-OpenMP-parallel, which is the point of Config E.

### (b) OpenMP nesting default-off — CONFIRMED

`omp_nesting_check.log` (probe linked against the same libomp the Config E
binary uses, `SHUD_RHS_THREADS` unset, no nesting env):

```
omp_get_max_active_levels()  = 1
nested-region active threads = 1  (1 => nesting default-off, no 2nd active level)
VERDICT: nesting-default-off = CONFIRMED
```

**Thread-topology paragraph.** `N` is the single cfg-numthreads knob
(`MD->CS.num_threads`, cfg `NUM_OPENMP`) driving BOTH the StrictOMP RHS threads
and the OpenMP-NVector threads via `N_VNew_OpenMP(NY, MD->CS.num_threads, …)`.
`SHUD_RHS_THREADS` is **unset in every leg** (recorded above and in each
`leg-n*.stdout.log` as `P1e startup: SHUD_RHS_THREADS=(unset) ->
omp_set_num_threads(N)`); `OMP_NUM_THREADS` is set coherently to the same N.
The StrictOMP RHS `#pragma omp parallel` regions (in `MD_rhs_core.cpp`) and the
OpenMP-NVector element-wise ops are invoked **serially from CVODE at distinct
points** — a `CVode()` step runs the RHS eval, then the linear-solver NVector
ops, never one inside the other — so they **never nest**. With nesting
default-off, even if they did overlap textually no second active parallel level
would spawn. (At N=1 the NVector backend floors at 2 threads —
`max(CS.num_threads, pf_in->numthreads)` in `MD_readin.cpp` — but Config E is
bitwise-deterministic regardless of thread count, so the floor does not affect
the gate; leg-n1 still matches the Config C baseline.)

### (c) ASan/UBSan leg (bonus — not required by spec/tasks/brief/CI)

`asan-hybrid-keliya.*` — `shud_asan` built Config E
(`SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1`, `-fsanitize=address,undefined`)
+ `SHUD_NVEC_PROF=1`, keliya, N=4, `ASAN_OPTIONS=…:halt_on_error=1`. Result:
exit 0, sentinel "The successful end." present, **0 ASan errors, 0 UBSan
errors, 0 sanitizer warnings**, all hybrid + profiler asserts PASS. The
ops-table pointer manipulation (override install + profiler wrap + clone
propagation) is memory-clean. (The CI `serial-baseline.yml` ASan job builds
`make shud_asan` in the DEFAULT config — Config C / serial — so it exercises
the hybrid TU only as its no-op `#else` fallback, which links cleanly; Config E
under ASan is this local leg.)

---

## Build wiring (task 2.3)

- `SHUD_NVEC_HYBRID` defaults to 0; default `make shud` / `make shud_omp`
  behavior byte-identical (hybrid TU compiles as no-op `#else` fallback).
- `SHUD_NVEC_HYBRID=1` **without** `SHUD_USE_OPENMP_NVECTOR=1` → parse-time
  `$(error)` loud abort (`Makefile:264`). Invalid values → `$(error)`
  (`Makefile:268`).
- Config C `make shud_omp` (`openMP NVector: OFF (Serial backend)`), Config D
  `make shud_omp SHUD_USE_OPENMP_NVECTOR=1` (`openMP NVector: ON`, build-only),
  Config E (adds `[NVEC_HYBRID] … overrides installed` startup line) all
  compile with identifiable markers.
- **Zero diff** in RHS files `src/Model/f.cpp` and `src/Model/MD_rhs_core.cpp`.

---

## Files

| file | what |
|------|------|
| `ge1_bitwise.sh` / `ge1_run.log` | Mac clang G-E1 runner + isolated PASS log |
| `sha/leg-n{1,2,4,8}.sha` | four-leg model-output manifests (all == baseline) |
| `leg-n{1,2,4,8}.std{out,err}.log` | per-leg run logs (topology recorded) |
| `gcc_spot_leg/` | server GCC-13 G-E1 spot leg (heihe C/E × N∈{1,8}) — PASS + FMA disasm |
| `optnone_fold_order.evidence.txt` | corrected root-cause + fix, per-variant divergence + FMA disasm, BOTH toolchains |
| `prof-hybrid-keliya.*` | PROF×HYBRID composition leg (csv + asserts) |
| `omp_nesting_check.log` | nesting default-off probe |
| `asan-hybrid-keliya.*` | Config E ASan/UBSan clean leg (bonus) |

Audit table + full mechanism rationale: `SHUD/docs/p12-nvec/nvec_reduction_audit.md`,
`SHUD/src/Model/MD_nvec_hybrid.{hpp,cpp}`.
