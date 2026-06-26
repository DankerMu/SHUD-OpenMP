# P1e Phase 1 Verdict — PASS

- **Date**: 2026-06-25
- **Decision**: Phase 1 PASS (per spec `p1e-strict-omp-rhs` design D1 + D12.1 happy path criteria, restricted to mode A SHALL scope)
- **Mac evidence**: [`docs/p1e/p1e_pr_c_2x2_mac.md`](p1e_pr_c_2x2_mac.md) (PR-C #326, parent commit `b7dd275` → merge `dcd8b48` series; SHUD pin `9a422e5`)
- **Server evidence**: [`docs/p1e/p1e_pr_d_2x2_server.md`](p1e_pr_d_2x2_server.md) (PR-D #327, parent commit `dffe965` → merge `e5a08d4` series; SHUD pin `9a422e5`)
- **Total cell coverage**: 96 cells (48 Mac + 48 server) = 2 build × 4 case × 4 N × 3 reps
- **SHALL gate scope**: mode A (NVECTOR_SERIAL + serial RHS) across 4 cases × 4 N × 3 reps = 48 mode A cells
- **Unblocks**: PR-F (#314) `ExecPolicy::StrictOMP` impl — mode A SHALL baseline established as the reference hash family for all subsequent strict-mode work

---

## 1. Verdict header rationale

The P1e epic (`p1e-strict-omp-rhs`) splits its evidence collection into two
phases:

- **Phase 1** (this verdict): characterise the existing **mode A** (serial
  NVECTOR + serial RHS) and **mode B** (NVECTOR_OPENMP + serial RHS)
  builds on the 4-case scale ladder (keliya 484, qhh 4773, heihe 6335,
  heihe_x4 ~25k). Mode A establishes the **SHALL reference hash family**
  for every downstream strict-OMP gate; mode B characterises the NVECTOR
  parallelism upper bound.
- **Phase 2** (deferred to PR-I, #317): characterise mode C (StrictOMP RHS,
  serial NVECTOR) and mode D (StrictOMP RHS + NVECTOR_OPENMP) once PR-G
  (#315) wires the build-system + dispatch surface.

PR-E (this document) closes Phase 1 by aggregating the 96 cells across
Mac (PR-C) and server (PR-D), checking the SHALL criteria from spec
design D1, and declaring the routing framework for Phase 2 per design
D12. It is a docs-only PR — **no source change, no SHUD pin bump** — so
the PR-F implementer can proceed against a frozen, audited Phase 1
baseline.

---

## 2. SHALL gate summary — 96 cells

Per spec design D1 / D12.1 the two SHALL gates for Phase 1 are:

- **AC1**: mode A produces bitwise-identical `<case>.rivqdown.sha256`
  across all 3 reps of every (case, N) cell.
- **AC2**: mode A produces bitwise-identical
  `<case>.rivqdown.sha256` across all 4 N values for each case at rep=1
  (cross-N stability under `OMP_NUM_THREADS`).

Mode A is the SHALL reference because (a) it does not engage the
NVECTOR_OPENMP non-associative reduction surface and (b) it must remain
the rivqdown-hash anchor for PR-F/G/I to compare strict-mode runs
against.

### 2.1 AC1 — mode A 3-rep bitwise per (case, N)   **SHALL gate**

Combined 16 groups (4 cases × 4 N), each row representing 3 reps.
Per-cell `rivqdown_sha12` (first 12 hex of SHA256) shown for rep1;
rep2/rep3 verified identical in the source docs (PR-C §3.1, PR-D §3.1).

| case      | NumEle | N  | platform | rep1 SHA12     | reps unique | verdict |
|-----------|-------:|---:|----------|----------------|------------:|:-------:|
| keliya    |    484 |  1 | Mac      | `b769e3270e1c` |           1 |  PASS   |
| keliya    |    484 |  2 | Mac      | `b769e3270e1c` |           1 |  PASS   |
| keliya    |    484 |  4 | Mac      | `b769e3270e1c` |           1 |  PASS   |
| keliya    |    484 |  8 | Mac      | `b769e3270e1c` |           1 |  PASS   |
| qhh       |   4773 |  1 | Mac      | `ccc7dd09d018` |           1 |  PASS   |
| qhh       |   4773 |  2 | Mac      | `ccc7dd09d018` |           1 |  PASS   |
| qhh       |   4773 |  4 | Mac      | `ccc7dd09d018` |           1 |  PASS   |
| qhh       |   4773 |  8 | Mac      | `ccc7dd09d018` |           1 |  PASS   |
| heihe     |   6335 |  1 | server   | `a2023ccd2de4` |           1 |  PASS   |
| heihe     |   6335 |  2 | server   | `a2023ccd2de4` |           1 |  PASS   |
| heihe     |   6335 |  4 | server   | `a2023ccd2de4` |           1 |  PASS   |
| heihe     |   6335 |  8 | server   | `a2023ccd2de4` |           1 |  PASS   |
| heihe_x4  | ~25000 |  1 | server   | `b5e4b0a2cf83` |           1 |  PASS   |
| heihe_x4  | ~25000 |  2 | server   | `b5e4b0a2cf83` |           1 |  PASS   |
| heihe_x4  | ~25000 |  4 | server   | `b5e4b0a2cf83` |           1 |  PASS   |
| heihe_x4  | ~25000 |  8 | server   | `b5e4b0a2cf83` |           1 |  PASS   |

**AC1 result: PASS — 16 / 16 groups bitwise-identical across 3 reps.**

### 2.2 AC2 — mode A cross-N stability per case (rep=1)   **SHALL gate**

| case      | NumEle | platform | N=1            | N=2            | N=4            | N=8            | unique | verdict |
|-----------|-------:|----------|----------------|----------------|----------------|----------------|-------:|:-------:|
| keliya    |    484 | Mac      | `b769e3270e1c` | `b769e3270e1c` | `b769e3270e1c` | `b769e3270e1c` |      1 |  PASS   |
| qhh       |   4773 | Mac      | `ccc7dd09d018` | `ccc7dd09d018` | `ccc7dd09d018` | `ccc7dd09d018` |      1 |  PASS   |
| heihe     |   6335 | server   | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` |      1 |  PASS   |
| heihe_x4  | ~25000 | server   | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` |      1 |  PASS   |

**AC2 result: PASS — 4 / 4 cases produce SHA invariant under
`OMP_NUM_THREADS`.**

Combined AC1 + AC2 closure: **mode A meets all Phase 1 SHALL
criteria**. Mode A `<case>.rivqdown.sha256` is now the SHALL reference
family for downstream strict-mode work (PR-F → PR-J).

### 2.3 Informational gates (AC3 / AC4 / AC6)

| gate | description                                  | Mac (PR-C §3.3-3.6)                   | server (PR-D §3.3-3.6)            | verdict |
|------|----------------------------------------------|---------------------------------------|-----------------------------------|:-------:|
| AC3  | per-(build,case) CVODE stat determinism      | 4/4 groups identical × 12 reps        | 4/4 groups identical × 12 reps    | PASS    |
| AC4  | mode A vs mode B SHA (NVEC non-assoc)        | 8/8 groups diff (expected)            | 8/8 groups diff (expected)        | PASS-as-designed |
| AC6  | cv_y_hash line-count consistency             | 47/48 normal, 1 IO transient (WARN)   | 48/48 normal                      | PASS-with-WARN |

AC3 confirms that, within a fixed (build, case) pair, the solver
produces the same step / function / netf counts across 12 (N, rep)
cells. AC4 shows mode A and mode B always disagree at the SHA level
(NVECTOR_OPENMP reductions are non-associative by design); this is
**informational, not a SHALL gate**. AC6's one Mac anomaly
(`A_qhh_N2_rep2` returned 3021 of 12960 cv_y_hash lines) is an auxiliary
IO transient — the rivqdown SHA and CVODE stats for that cell are
bit-identical to the other two reps, so AC1 still PASSes. The server
batch produced zero AC6 anomalies after the PR-C ARG_MAX hot-fix. The
IO-hardening item is tracked for PR-G but priority is downgraded.

---

## 3. Mode A determinism analysis (the SHALL reference)

Mode A is the only build mode that **must** retain bitwise determinism
across all N values for all 96 cells; this section captures why the
Phase 1 PASS holds across two OMP runtimes.

### 3.1 Why mode A is thread-invariant by construction

Mode A is built with `NVECTOR_SERIAL` + the serial RHS (the current
SHUD `RhsLoopExecutor::Serial` path). The `ExecPolicy::StrictOMP` case
label at `MD_rhs_core.cpp:802-811` (per PR-A audit against SHUD pin
9a422e5; PR-F implementer to re-verify before stubbing) is
`#ifdef SHUD_ENABLE_OPENMP_RHS`-guarded and excluded from mode A/B
binaries; dispatch reaches only `ExecPolicy::Serial`. PR-F will
replace the stub once the flag is wired. With no
`#pragma omp parallel for reduction(+:…)` in the integration hot path,
the floating-point sum order is fixed; bit pattern of every state
update is identical across runs and across `OMP_NUM_THREADS`.

The empirical PASS on AC2 confirms there is **no hidden parallel
section** leaking non-determinism into mode A — a prerequisite for
using mode A as the SHALL reference in PR-F/I.

### 3.2 Mode A wall-time symmetry across N (Mac libomp vs server libgomp)

Mean wall time (seconds, 3 reps per cell):

| case      | platform | N=1     | N=2     | N=4     | N=8     | N1/N8 ratio |
|-----------|----------|--------:|--------:|--------:|--------:|------------:|
| keliya    | Mac (libomp)    |  30.7 |  30.3 |  30.3 |  35.0 |       0.88× |
| qhh       | Mac (libomp)    |  97.7 | 101.0 | 101.7 | 109.3 |       0.89× |
| heihe     | server (libgomp)| 526.0 | 506.0 | 507.0 | 508.0 |       1.04× |
| heihe_x4  | server (libgomp)|1343.7 |1341.0 |1338.3 |1339.0 |       1.00× |

Observations:

- **Server (libgomp)** is essentially N-invariant: wall time tracks
  within ±1 % across N=1…8. This is the textbook signature of a
  serial-RHS integrator with no parallel section consuming the extra
  threads.
- **Mac (libomp)** shows a mild slowdown going N=4 → N=8 (`0.88×` and
  `0.89×`) — this is the libomp scheduler waking idle worker threads on
  Apple Silicon for no parallel work; threads still pay context-switch
  cost without participating in the integration. Not a determinism
  violation (AC2 still PASS); a thread-pool warm-up artifact in libomp's
  default scheduler.

The combined Mac (libomp) + server (libgomp) PASS on AC2 establishes
that mode A is bit-deterministic on **both OMP runtimes the project
ships against** (macOS dev / Linux production).

### 3.3 Implication for PR-F onwards

PR-F and PR-I will compare strict-mode (C, D) rivqdown SHAs against the
mode A reference family. The 48 mode A SHAs from Phase 1 (12 mode A
cells × 4 cases = 48 SHA samples reducing to 4 unique SHAs per case) is
the durable hash anchor: any subsequent strict-mode change that does
not match these 4 SHAs at N=1..8 × rep=1..3 fails the SHALL gate.

---

## 4. Mode B NVEC overhead / speedup pattern (4-case full ladder)

Mode B is **NOT a SHALL gate** for PR-E. It is the operating-envelope
characterisation for whether shipping NVECTOR_OPENMP as an opt-in is
useful prior to the StrictOMP RHS work.

### 4.1 Combined 4-case synthesis

| Case      | NumEle | Platform | B/A @ N=1 | Best B speedup (B N=1 → B N=k) | Best N | Speedup at SHALL-gate N=8 | Notes |
|-----------|-------:|----------|----------:|-------------------------------:|-------:|--------------------------:|-------|
| keliya    |    484 | Mac      | **6.64×** overhead | 1.07× (N=8)              |   N=8  | 1.07× | NVEC dead at this scale |
| qhh       |   4773 | Mac      | 1.18× overhead     | 1.00× (essentially flat) |    —   | ≈ 1.00× | near-neutral |
| heihe     |   6335 | server   | 1.02× overhead     | 1.06× (N=2; saturates)   |   N=2  | ≈ 1.06× | neutral |
| heihe_x4  | ~25000 | server   | 1.03× overhead     | **1.15× (N=2; saturates)** | **N=2** | **1.149×** | first OMP win in P1e |

### 4.2 Key insight — NVEC overhead size-dependence cliff

The 4-point ladder reveals a sharp **cliff between 484 and 4773 cells**
in NVECTOR_OPENMP usefulness:

- At 484 cells (keliya) the per-step team-spawn + barrier + reduction
  merge cost dwarfs the per-element vector work; mode B is **6.6×
  slower** than mode A even at N=1. Adding threads (N=8) recovers only
  1.07×; mode B is **dead at small-basin scale**.
- At 4773 cells (qhh) mode B drops to **1.18× overhead at N=1** and
  produces no measurable speedup across N=1…8. Near-neutral, not
  useful.
- At 6335 cells (heihe, server) mode B is **statistically neutral**
  (`B/A = 1.02×`) at N=1 with a weak 1.06× intra-N speedup at N=2 that
  saturates.
- At ~25 000 cells (heihe_x4) mode B finally produces a **first OMP
  win**: **1.15× speedup from N=1 → N=2**, then immediate saturation
  (N=4 and N=8 are within run-to-run noise of N=2). The 1.15× ceiling
  is already reached by N=2 and held at N=8 (the SHALL-gate measurement
  point per design.md D7 L266-267); heihe_x4 N=8 = 1.149× is the prior
  that PR-I's mode D must materially exceed to satisfy the D7 SHALL.

The N=2 saturation has a clear physical reading: NVECTOR_OPENMP wraps
**only the vector-op fraction** of the integrator's per-step cost; in
heihe_x4 that fraction is ≈ 15 % of total wall. Once two threads
amortise the parallel-region overhead, the remaining 85 % serial RHS
clamps the speedup. **The 1.15× ceiling is far below the per-case
speedup SHALL in design D7 (heihe ≥1.3×, heihe_x4 ≥1.5× at N=8)**;
therefore RHS parallelism (modes C/D via PR-F/G) is **essential, not
optional**, for any meaningful speedup.

### 4.3 Cross-build CVODE stat drift (informational)

| build | case      | nst   | nfe   | nni   | netf | drift vs A |
|:-----:|-----------|------:|------:|------:|-----:|------------|
| A     | keliya    |111130 |112463 |112462 |    7 | reference  |
| B     | keliya    |100244 |102496 |102495 |    8 | −9.8 % nst |
| A     | qhh       | 13000 | 13270 | 13269 |    0 | reference  |
| B     | qhh       | 13000 | 13279 | 13278 |    0 | +0.07 % nfe|
| A     | heihe     |  6698 |  6943 |  6942 |    0 | reference  |
| B     | heihe     |  6713 |  6992 |  6991 |    0 | +0.22 % nst|
| A     | heihe_x4  |  6575 |  6741 |  6740 |    0 | reference  |
| B     | heihe_x4  |  6572 |  6719 |  6718 |    0 | −0.05 % nst|

NVECTOR_OPENMP reduction-order non-associativity shifts CVODE's local
truncation-error estimate by floating-point ε per RHS call, which
propagates into the adaptive step controller. The drift is not
monotonic in problem size (heihe is +0.22 %, heihe_x4 is −0.05 %, sign
flips). All drift values are within the 1-2 % envelope master plan §6
anticipates for any reduction-order-sensitive change. Mode B is
**self-consistent** (same nst across all 12 (N, rep) cells per build)
— that is the only invariant mode B claims.

---

## 5. Decision routing per spec design D12

Spec `p1e-strict-omp-rhs/design.md` D12 defines four decision branches
for the F-route (StrictOMP RHS) outcome. PR-E declares the **framework
placeholder** for each branch; the **actual amend** of the Phase 2
verdict (filling in which branch fired) is **owned by PR-I (#317)** per
tasks §2.7 last bullet.

### 5.1 D12 branches and PR-E placeholder status

| branch | spec criterion | Phase 1 data availability | PR-E status |
|--------|----------------|---------------------------|-------------|
| **D12.1 (happy path)** | mode C cross-N bitwise PASS + mode C nst Δ = 0 vs mode A + per-case speedup SHALL (heihe ≥1.3×, heihe_x4 ≥1.5× at N=8, per design D7) | mode C not yet built (requires PR-G `-fopenmp` wiring + PR-F StrictOMP impl) | **NOT verifiable yet** — placeholder for PR-I amend; phase 1 mode A SHALL ref established |
| **D12.2 (Path 2 fallback)** | mode C cross-N FAIL | contingency, not triggered | placeholder — only relevant if PR-I observes cross-N divergence |
| **D12.3 (Path 3 fallback)** | mode C cross-N PASS but BOTH cases < own threshold (heihe<1.3× AND heihe_x4<1.5× at N=8, per design D7 / tasks §4.6 AND-gate) | contingency, would trigger PR-N block-Jacobi precond | placeholder — only relevant if PR-I observes weak speedup on both cases |
| **D12.4 (Path 4 deferred)** | none of the above pass | contingency, would trigger ADR-0003 KLU pattern spike | placeholder — only relevant if PR-I observes total failure |

PR-E's role: lock in the framework so PR-I's amend has a stable
structure to fill against. The mode B data already collected provides
the **expected upper-bound prior** for PR-I — if mode C delivers less
than mode B's 1.149× heihe_x4 N=8 speedup the work has gone backwards;
if mode C scales past N=2 the basin is the RHS not the solver (master
plan §6 D2-D3 anticipated outcome).

### 5.2 Default assumption per spec design D12

Per spec design.md L489 the **default working assumption is that PR-N
block-Jacobi precond (D12.3) will be triggered**. PR-E does not contest
this assumption — Phase 2 mode C/D data will either reinforce it (if
BOTH cases < own threshold: heihe<1.3× AND heihe_x4<1.5× at N=8 per
design D7 / tasks §4.6 AND-gate) or override it (if happy path holds).
PR-I owns the override decision.

### 5.3 Phase 2 amend ownership

PR-E **does not** declare D12.1 / D12.2 / D12.3 / D12.4 for Phase 2.
This is **explicit deferral to PR-I (#317)** per tasks §2.7. PR-I
collects mode C/D wall-time + cross-N bitwise data on the server 24-cell
matrix and amends this verdict with the actual D12 outcome.

---

## 6. Phase 2 verdict (PR-I amend per tasks §4.6.3)

**Amend status**: This section was a PR-E placeholder; PR-I (#317)
amends it with the actual mode C 24-cell server data + D12 routing
decision per tasks §4.6.3 [PR-I 必做]. Source data:
[`docs/p1e/p1e_pr_i_strict_omp_verification.md`](p1e_pr_i_strict_omp_verification.md) §2-§3.
Aggregator: `tools/p1e_aggregate_pr_i_shall.sh`. Mode D 96-cell amend
(per tasks §2.5.1 + §2.6.1) deferred to PR-C-phase2 / PR-D-phase2 once
PR-G post-merge work runs; PR-I scope is mode C 24-cell only.

### 6.1 Phase 2 mode C 24-cell data (PR-I source)

**Identity**: server `210.77.77.22:32099`, Slurm cn14 (heihe stream) +
cn15 (heihe_x4 stream), gcc 13.3.0 + libgomp, SHUD pin
`3341368d2d0854924d2286925c8575df52cc97a0`, mode C binary sha256
`1bfdc4c99b038301b6ba1fb48e2d935f449476e52cc4e42fe5301c0e7d637616`.
Run window 2026-06-25 16:41Z → 20:52Z (~4h10m parallel streams).

**AC-S1 — mode C cross-N bitwise per case** (one unique SHA across 4 N × 3 reps = 12 cells):

| case      | NumEle | unique_SHAs | rep1 SHA12     | rep2 SHA12     | rep3 SHA12     | verdict |
|-----------|-------:|------------:|----------------|----------------|----------------|:-------:|
| heihe     |   6335 |           1 | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` |  PASS   |
| heihe_x4  | ~25000 |           1 | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` |  PASS   |

**AC-S2 — mode C SHA == mode A reference SHA per case** (cross-mode
bitwise equality vs PR-D #312 LOCKED mode A reference per §1.1 of
`p1e_pr_i_strict_omp_verification.md`):

| case      | mode C SHA (N=1,rep=1) | mode A reference SHA (PR-D) | match | verdict |
|-----------|------------------------|-----------------------------|:-----:|:-------:|
| heihe     | `a2023ccd2de43543`     | `a2023ccd2de43543`          | same  |  PASS   |
| heihe_x4  | `b5e4b0a2cf83b2a4`     | `b5e4b0a2cf83b2a4`          | same  |  PASS   |

**AC-S3 — D7 per-case speedup AND-gate** (median of 3 reps wall, per
design D7 protocol; AND-gate per tasks §4.6 — D12.3 fires only if BOTH
< threshold):

| case      | N=1 wall median (s) | N=8 wall median (s) | speedup | threshold | per-case verdict |
|-----------|--------------------:|--------------------:|--------:|----------:|:----------------:|
| heihe     |                 504 |                 473 |  1.066× |     1.3×  |     **FAIL**     |
| heihe_x4  |                1340 |                 775 |  1.729× |     1.5×  |       PASS       |

AC-S3 D7 AND-gate result: **PARTIAL** — exactly one case meets
threshold. AND-gate semantics: BOTH FAIL triggers D12.3 block-Jacobi
fallback; here heihe_x4 PASS prevents D12.3 trigger and routes to tasks
§4.6.2 partial-closure decision point.

**nst ladder check** (per `openspec/specs/p1d-numa-governance/spec.md`
nst ladder Requirement — heihe Δ=0 strict, heihe_x4 |Δ|≤2):

| case      | ref nst (mode A) | N=1 | N=2 | N=4 | N=8 | max Δ to ref | ladder |
|-----------|-----------------:|----:|----:|----:|----:|-------------:|:------:|
| heihe     |             6698 |6698 |6698 |6698 |6698 |            0 |  PASS  |
| heihe_x4  |             6575 |6575 |6575 |6575 |6575 |            0 |  PASS  |

### 6.2 D12 routing decision (per tasks §4.6 + §10.4 + design.md L325-328)

Spec design D12 defines 4 branches based on SHALL + speedup outcomes.
Evaluation of PR-I results against each:

- [ ] **D12.1 (happy path)** — mode C cross-N PASS + nst Δ=0 + per-case
      speedup SHALL PASS (BOTH cases meet own threshold). Eval: AC-S1
      PASS + AC-S2 PASS + nst Δ=0 confirmed, but heihe 1.066× < 1.3×
      threshold → **NOT triggered** (per-case speedup SHALL not met for
      heihe).
- [ ] **D12.2 (Path 2 fallback)** — mode C cross-N FAIL → NVECTOR_REPRO_OMP
      custom backend. Eval: AC-S1 PASS (cross-N bitwise on both cases) →
      **NOT triggered**.
- [ ] **D12.3 (Path 3 fallback)** — cross-N PASS but **BOTH cases** <
      own threshold (heihe<1.3× AND heihe_x4<1.5× per tasks §4.6
      AND-gate) → trigger PR-N P1e.8 block-Jacobi precond. Eval:
      heihe FAIL (1.066×) but heihe_x4 PASS (1.729×) → AND-gate **NOT
      satisfied** → **NOT triggered**.
- [ ] **D12.4 (Path 4 deferred)** — none of D12.1/.2/.3 apply + need
      deeper solver refactor → ADR-0003 KLU spike. Eval: D12.3 not
      triggered (heihe_x4 already at 1.729×), no total-failure
      condition → **NOT triggered**.

- [x] **§4.6.2 partial-closure → SHIP** (active path) — none of the
      strict D12.1/.2/.3/.4 branches triggered because the D12.3
      AND-gate semantics require BOTH cases below threshold, which did
      not happen (heihe_x4 1.729× ≥ 1.5×). Per tasks §4.6.2:

      > 4.6.2 单 case 不达 threshold（另一 case 已达）：进 partial
      > closure 决策点（用户决策 ship vs fallback；倾向 ship 当
      > heihe_x4 达 1.5× 时）

      User confirmed SHIP per spec default (heihe_x4 production target
      hits 1.729× ≥ 1.5× threshold; heihe small-case 1.066× shortfall
      documented as design-expected OMP overhead floor — see §6.3).

### 6.3 SHIP rationale + heihe small-case carve-out

**Why SHIP**: The strict-omp RHS strategy (`ExecPolicy::StrictOMP` per
PR-F + `-fopenmp` wiring per PR-G + steady-state first-touch removal
per PR-H) delivers bitwise-correct (AC-S1 + AC-S2 PASS) and materially
faster (heihe_x4 1.729× ≥ 1.5×) results on the **production-target
mesh density** (~25k cells, real NWM basin refinement). The heihe
6335-cell shortfall is not a determinism failure; it is the expected
small-mesh OMP overhead floor that the design D7 asymmetric thresholds
(1.3× small / 1.5× large) explicitly acknowledge.

**heihe small-case carve-out** (per design D7 + tasks §4.6.2):
- heihe 6335-cell N=8 speedup 1.066× is below the 1.3× threshold but
  above unity (no regression). Cross-N bitwise (AC-S1) + cross-mode
  bitwise (AC-S2) + nst Δ=0 all PASS → correctness invariants intact.
- Mode B Phase 1 evidence (§4.2 above) already showed NVEC overhead
  size-dependence cliff: small basins (484, 4773 cells) saw NVECTOR_OPENMP
  net-negative; heihe (6335) was statistically neutral. Mode C continues
  this size-dependence pattern — RHS parallel overhead at 6335 cells
  divided across N=8 threads on cn14 NUMA (~3k cells per NUMA node) does
  not amortize the fork-join cost.
- heihe_x4 (~25k cells) at 1.729× **confirms the strict-omp RHS strategy
  scales correctly on production-scale meshes**. The size-dependence
  cliff is a documented limitation, not a defect.

**Cross-ref**: Full PR-I data + verdict detail in
[`docs/p1e/p1e_pr_i_strict_omp_verification.md`](p1e_pr_i_strict_omp_verification.md):
- §1 identity (commit / SHUD pin / binary sha256 / job IDs)
- §2 24-cell roster (per-cell wall + nst + nfe + SHA12)
- §3.1-3.3 AC-S1 / AC-S2 / AC-S3 verdicts
- §4 wall-time tables (mean + median + per-rep)
- §5 nst ladder
- §6 reproducibility
- §8 D12 routing + SHIP recommendation

**Mode D 96-cell deferred**: tasks §2.5.1 + §2.6.1 mode C/D 96-cell
Phase 2 amend (PR-C-phase2 Mac + PR-D-phase2 server) is gated by
PR-G post-merge work. PR-I scope is mode C 24-cell server only; mode D
amend will append to this §6 in a follow-up PR per tasks §3.6.1.

### 6.4 Phase 2 forward-looking inputs (retained from PR-E)

PR-E's forward-looking priors below were partially confirmed by PR-I
data:

- **Mode B intra-N saturation at N=2** (PR-D §3.5, §4.2): mode C
  heihe_x4 scaling slope N=1→2 1.291×, N=2→4 1.190×, N=4→8 1.125×
  shows continued (but diminishing) returns past N=2 — **strict-omp
  RHS does open up scaling past the NVEC N=2 plateau** at production
  scale. Confirms the master plan §6 D2-D3 prediction that RHS
  parallelism is the right lever (not NVECTOR_OPENMP).
- **Mode B at heihe_x4 = 1.149× at SHALL-gate N=8**: mode C heihe_x4 =
  1.729× materially exceeds this **and** the D7 1.5× SHALL — strict-omp
  RHS delivers the heihe_x4 speedup that NVECTOR_OPENMP alone could not.
- **NVEC overhead size-dependence cliff**: mode C heihe (6335) 1.066×
  confirms the size-dependence pattern carries into strict-omp RHS;
  small-case overhead floor is a strategy-independent property, not a
  mode C defect. heihe_x4 (~25k) crosses the break-even cleanly.

---

## 7. PR-F readiness

Phase 1 SHALL gates satisfied per §2 + §3; PR-F unblocked — may begin
`ExecPolicy::StrictOMP` impl per spec design D2 (single
`#pragma omp parallel` enclosing `rhs_update` / `rhs_flux` /
`rhs_apply` 3-phase fan-out), replacing the `std::abort()` stub at
`SHUD/src/Model/MD_rhs_core.cpp:802-811` (per PR-A audit against SHUD
pin 9a422e5; PR-F implementer to re-verify before stubbing).

---

## 8. Forward action items

The PR sequence for P1e past this verdict, in dependency order:

- **PR-F (#314)**: implement `ExecPolicy::StrictOMP` per spec design D2
  (single `#pragma omp parallel` + 3-phase `rhs_update` / `rhs_flux` /
  `rhs_apply` fan-out). Replaces `std::abort()` stub at
  `SHUD/src/Model/MD_rhs_core.cpp:802-811` (per PR-A audit against SHUD
  pin 9a422e5; PR-F implementer to re-verify before stubbing).
  Unblocked by PR-E.
- **PR-G (#315)**: Makefile `-fopenmp` wiring + per-thread split for
  mode C / mode D builds. Required to compile the StrictOMP path.
- **PR-H (#316)**: steady-state first-touch removal per spec design D7
  (NUMA-friendly allocation pattern). Required for predictable
  cross-N walltime on the server.
- **PR-I (#317)**: server 24-cell 3 SHALL gate (mode C/D on 2 cases × 4
  N × 3 reps) **and** amend Phase 2 verdict (§6 placeholders → actual
  D12.1/.2/.3/.4 outcome). Owner of this document's §6 amend.
- **PR-J (#318)**: Mac SHALL closure — 4 cases × N=1 reverse-compat
  verification per spec design D7. If PR-N triggered (per tasks §4.6
  D12.3), PR-J/K/L/M pause until PR-N closes.
- **PR-K (#319)**: capstone docs — unified
  `docs/p1e/p1e_2x2_experiment.md` weaving Mac + server narrative,
  unified aggregator refactor (deferred from PR-C/D per PR-D §8), and
  `docs/p1e/INDEX.md`.

---

## 9. Files produced by PR-E

| Path                              | Purpose                                                   |
|-----------------------------------|-----------------------------------------------------------|
| `docs/p1e/p1e_2x2_verdict.md`     | This document (Phase 1 verdict + D12 placeholder framework) |

No source change, no SHUD pin bump, no aggregator change. PR-E is a
pure synthesis PR: the durable evidence remains the SHA tables in
PR-C / PR-D; this document is the audit-trail layer that says "Phase 1
PASS, proceed to PR-F, defer Phase 2 amend to PR-I".

---

## 10. Reproducibility footprint

PR-E does not run any code. To verify the verdict, re-aggregate the
PR-C / PR-D evidence:

```bash
# Mac (PR-C)
tools/p1e_aggregate_mac.sh all       # prints PR-C tables
tools/p1e_aggregate_mac.sh ac1       # exit 0 = PASS
tools/p1e_aggregate_mac.sh ac2

# Server (PR-D); requires rsync of artifact tree first per PR-D §6
tools/p1e_aggregate_server.sh all    # prints PR-D tables
tools/p1e_aggregate_server.sh ac1
tools/p1e_aggregate_server.sh ac2
```

All four `ac1` / `ac2` invocations must exit 0 for §2 PASS to hold.
The per-cell SHA tables in PR-C §2 and PR-D §2 are the durable evidence
for the verdict; this document only synthesises them.
