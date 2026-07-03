# v1.1.1 default-flip — review evidence

`make shud_omp` default flipped **Config C → Config E** (user decision post-#441;
precedent = v1.0.1 Config-C flip #430). Config E is bitwise-identical to Config C
at every thread count (G-E1 + PR-N2, single rivqdown SHA across 19 runs) and
1.41×@N16 faster — a pure Pareto default upgrade with zero golden/CI breakage.
Config E2 (order-shifted, re-baselined) stays opt-in.

SHUD build under test: `openmp-baseline` @ this PR's SHUD commit (Makefile flag-
default flip only; recipe body byte-unchanged from v1.1). Platform: Apple
clang / ARM (Mac), SUNDIALS/CVODE 6.0.0.

## Makefile design (mirrors the v1.0 SHUD_ENABLE_OPENMP_RHS flip)

Goal-scoped defaults, CLI `?=`-override still wins:

- `SHUD_USE_OPENMP_NVECTOR ?= 1` under the `shud_omp` goal (else `?= 0`).
- `SHUD_NVEC_HYBRID ?= 1` nested under `shud_omp` ∧ NVECTOR=1 (else `?= 0`).

Net semantics:

| Command | Config | NVECTOR | HYBRID | DETRED |
|---|---|:-:|:-:|:-:|
| `make shud_omp` | **E** (default) | 1 | 1 | 0 |
| `make shud_omp SHUD_USE_OPENMP_NVECTOR=0` | **C** (single-flag opt-out) | 0 | 0 | 0 |
| `make shud_omp SHUD_NVEC_HYBRID=0` | *(refused)* — Config D guard | 1 | 0 | — |
| `make shud_omp SHUD_NVEC_HYBRID=0 SHUD_ALLOW_CONFIG_D=1` | **D** (build-only smoke) | 1 | 0 | 0 |
| `make shud_omp SHUD_NVEC_DETRED=1` | **E2** (short form; HYBRID defaults 1) | 1 | 1 | 1 |
| `make shud` (+ `shud_asan`, `smoke_configd`) | A/C | 0 | 0 | 0 |

Config D foot-gun guard: post-flip a single negative flag
`make shud_omp SHUD_NVEC_HYBRID=0` (NVECTOR still defaulting 1) would silently
yield Config D — the REFUTED nondeterministic config (reduction-order drift,
10–25% cross-thread; design D2 / ADR-0011). The parse-time guard fails loud
unless `SHUD_ALLOW_CONFIG_D=1` is also passed. Scoped to the `shud_omp` goal so
`smoke_configd` (a different goal) is unaffected.

## Known pattern property — goal-scoped defaults are per-`make`-run global

`$(filter shud_omp,$(MAKECMDGOALS))` keys the defaults on the goal list of the
whole `make` invocation, not per target. Co-invoking serial and OMP in one
command (`make shud shud_omp`) therefore applies the shud_omp defaults
(NVECTOR/HYBRID=1 — and RHS=1, the identical pre-existing property of the
v1.0.1 flip at Makefile:149-153) to BOTH goals: the serial `shud` recipe would
compile with the OMP defines and link `nvecopenmp`. Single-goal `make shud` is
unaffected (leg 6). Build serial and OMP binaries in separate `make`
invocations. Inherited from the v1.0.1 pattern, not introduced by this PR
(review finding Minor 4, PR #452).

## Files

- `flag_matrix.log` — build + marker/link + runtime-marker inspection across the
  7-cell matrix. v2, regenerated on this PR's actual SHUD commit after review
  (PR #452 Minors 1+2): the v1 log carried a pre-pointer-bump HEAD stamp and
  counted the expected E2-format-string presence in the default-E binary as a
  failure; v2 stamps the build-tree HEAD and asserts the authoritative runtime
  stdout marker per hybrid leg.
- `sha/defaultE_n{1,8}.manifest.sha` — keliya output SHA256 manifests, default-E
  build at N∈{1,8}.
- `sha/baseline_C_n8.manifest.sha` — Config C baseline lineage
  (copied from `.review-evidence/p12-nvec/pr-n3/keliya_det/C_n8.manifest.sha`;
  in that evidence set C_n8 == E_n8 == E2_n8 byte-for-byte at production B=4096
  — that identity IS the C==E==E2 bitwise proof for keliya, NY=1986 < 4096 ⇒
  E2's fixed tree degenerates to one block ≡ plain serial fold).
- `link-check/config{E_default,C_optout,D_allowed,E2_short,serial_shud}.txt` —
  `otool -L` (nvecopenmp/libomp linkage) + compiled-in NVEC config strings +
  SHUD_RHS_THREADS StrictOMP marker count, per config.

## Verdicts

**Flag matrix** (`flag_matrix.log`, v2): all 8 legs PASS (`pass=8 fail=0`).
strings(1) limitation, handled explicitly: BOTH `NVEC config` format-string
literals are compiled into any hybrid binary (runtime if/else on
`nvec_hybrid_detred_active()`; the E2 literal carries `%d` placeholders), so
string presence cannot discriminate Config E vs E2 — the harness records the
E2 literal in the default-E binary as EXPECTED and asserts the authoritative
runtime stdout marker instead. Default build prints
`NVEC config: Config E (serial reduction overrides; DETRED=off)`; both E2 legs
print `NVEC config: Config E2 (fixed-tree deterministic reductions; B=4096, …)`
and no Config E line.

**keliya bitwise gate**:
- `defaultE_n1.manifest.sha` == `baseline_C_n8.manifest.sha` — BITWISE MATCH
- `defaultE_n8.manifest.sha` == `baseline_C_n8.manifest.sha` — BITWISE MATCH
- `defaultE_n1` == `defaultE_n8` — thread-count-invariant

The flipped default reproduces the Config C lineage byte-for-byte at every
thread count. Serial `make shud` keliya `keliya.rivqdown.dat` SHA also equals the
C baseline (serial path unchanged).

**Link discriminator** (`link-check/`):
- Config E (default): links `libsundials_nvecopenmp` + `libomp`; NVEC config
  strings present.
- Config C (opt-out): NO `nvecopenmp`; keeps `libomp` (StrictOMP RHS); NO NVEC
  config strings.
- Config D (allowed): links `nvecopenmp`; NO NVEC config strings (no hybrid
  overrides).
- Serial `shud`: NO `nvecopenmp`, NO `libomp`.

CI note: CI default build is now Config E; the keliya bitwise gate stays green
because E==C bitwise — that is the entire point of the flip.
