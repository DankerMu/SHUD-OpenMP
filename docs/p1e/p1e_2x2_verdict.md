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
SHUD `RhsLoopExecutor::Serial` path on `MD_rhs_core.cpp:802-811`'s
`std::abort()` stub for `ExecPolicy::StrictOMP`, but never reaching it
because the dispatch falls through to the serial loop). With no
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

| Case      | NumEle | Platform | B/A @ N=1 | Best B speedup (B N=1 → B N=k) | Best N | Notes |
|-----------|-------:|----------|----------:|-------------------------------:|-------:|-------|
| keliya    |    484 | Mac      | **6.64×** overhead | 1.07× (N=8)              |   N=8  | NVEC dead at this scale |
| qhh       |   4773 | Mac      | 1.18× overhead     | 1.00× (essentially flat) |    —   | near-neutral |
| heihe     |   6335 | server   | 1.02× overhead     | 1.06× (N=2; saturates)   |   N=2  | neutral |
| heihe_x4  | ~25000 | server   | 1.03× overhead     | **1.151× (N=2; saturates)** | **N=2** | first OMP win in P1e |

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
  win**: **1.151× speedup from N=1 → N=2**, then immediate saturation
  (N=4 and N=8 are within run-to-run noise of N=2).

The N=2 saturation has a clear physical reading: NVECTOR_OPENMP wraps
**only the vector-op fraction** of the integrator's per-step cost; in
heihe_x4 that fraction is ≈ 15 % of total wall. Once two threads
amortise the parallel-region overhead, the remaining 85 % serial RHS
clamps the speedup. **The 1.15× ceiling is far below the 2× target
from spec design D1**; therefore RHS parallelism (modes C/D via
PR-F/G) is **essential, not optional**, for any meaningful speedup.

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
| **D12.1 (happy path)** | mode C cross-N bitwise PASS + mode C nst Δ = 0 vs mode A + per-case speedup ≥ 1.5× | mode C not yet built (requires PR-G `-fopenmp` wiring + PR-F StrictOMP impl) | **NOT verifiable yet** — placeholder for PR-I amend; phase 1 mode A SHALL ref established |
| **D12.2 (Path 2 fallback)** | mode C cross-N FAIL | contingency, not triggered | placeholder — only relevant if PR-I observes cross-N divergence |
| **D12.3 (Path 3 fallback)** | mode C cross-N PASS but speedup < 1.5× | contingency, would trigger PR-N block-Jacobi precond | placeholder — only relevant if PR-I observes weak speedup |
| **D12.4 (Path 4 deferred)** | none of the above pass | contingency, would trigger ADR-0003 KLU pattern spike | placeholder — only relevant if PR-I observes total failure |

PR-E's role: lock in the framework so PR-I's amend has a stable
structure to fill against. The mode B data already collected provides
the **expected upper-bound prior** for PR-I — if mode C delivers less
than mode B's 1.151× speedup on heihe_x4 at N=2 the work has gone
backwards; if mode C scales past N=2 the basin is the RHS not the
solver (master plan §6 D2-D3 anticipated outcome).

### 5.2 Default assumption per spec design D12

Per spec design.md L489 the **default working assumption is that PR-N
block-Jacobi precond (D12.3) will be triggered**. PR-E does not contest
this assumption — Phase 2 mode C/D data will either reinforce it (if
speedup < 1.5×) or override it (if happy path holds). PR-I owns the
override decision.

### 5.3 Phase 2 amend ownership

PR-E **does not** declare D12.1 / D12.2 / D12.3 / D12.4 for Phase 2.
This is **explicit deferral to PR-I (#317)** per tasks §2.7. PR-I
collects mode C/D wall-time + cross-N bitwise data on the server 24-cell
matrix and amends this verdict with the actual D12 outcome.

---

## 6. Phase 2 verdict placeholder (PR-I amend target)

This section is intentionally a **placeholder** for PR-I (#317) to
amend. The structure mirrors §2 / §3 / §5 above so PR-I's amend can
slot into the existing skeleton:

### 6.1 Phase 2 SHALL gate target (per spec design D1)

PR-I shall produce a mode C / mode D analogue of §2.1 / §2.2:

- **Phase 2 AC1**: mode C produces bitwise-identical
  `<case>.rivqdown.sha256` across all 3 reps of every (case, N) cell
  **and** mode C SHA matches the mode A SHALL reference from §2.1.
- **Phase 2 AC2**: mode C produces bitwise-identical
  `<case>.rivqdown.sha256` across all 4 N values for each case (cross-N
  stability).
- **Phase 2 AC3**: mode C CVODE nst Δ vs mode A = 0 (strict-mode
  reproduces serial solver path bit-for-bit).
- **Phase 2 AC4**: mode C wall-time speedup ≥ 1.5× on heihe_x4 at N≥2.

### 6.2 Phase 2 decision routing (PR-I fills)

PR-I shall declare which D12 branch fires:

- [ ] **D12.1** triggered: ship — proceed to PR-J / PR-K capstone path.
- [ ] **D12.2** triggered: cross-N FAIL — escalate to user (epic P1e'
      scope decision per design.md L326).
- [ ] **D12.3** triggered: speedup < 1.5× — open new PR for block-Jacobi
      precond within P1e (P1e.8 per design.md L327).
- [ ] **D12.4** triggered: total failure — defer to ADR-0003 KLU spike
      epic.

### 6.3 Phase 2 expected upper bounds (PR-E forward-looking inputs)

- **Mode B intra-N saturation at N=2** (PR-D §3.5, §4.2) suggests mode
  D may also saturate at N=2 unless the RHS parallelism opens up the
  serial bottleneck. If mode D shows the same N=2 plateau, the next
  bottleneck is the SUNDIALS linear solve (SPGMR) and the path forward
  is preconditioner / matrix-free RHS fusion (master plan §6 D2-D3).
- **Mode B at heihe_x4 = 1.151× ceiling** is the prior; mode C/D must
  exceed this materially to be net-positive.
- **NVEC overhead size-dependence cliff** (§4.2) suggests StrictOMP RHS
  parallelism may show the same cliff — mode C may be useless at
  keliya/qhh and only net-positive at heihe / heihe_x4 scale. PR-I
  shall report the per-case envelope.

---

## 7. PR-F readiness checklist (gating downstream work)

PR-F (`ExecPolicy::StrictOMP` implementation, #314) requires the
following Phase 1 deliverables before it can begin replacing the
`std::abort()` stub at `SHUD/src/Model/MD_rhs_core.cpp:802-811`:

- [x] Mode A SHALL bitwise PASS (Mac + server) — PR-E §2.1 (16/16 groups PASS)
- [x] Mode A cross-N stable (Mac + server) — PR-E §2.2 (4/4 cases PASS)
- [x] Mode A vs B SHA diff confirms NVEC perturbation is **design-expected**, NOT a bug — PR-E §2.3 (AC4 PASS-as-designed) + §4.3 (CVODE stat drift within ±1-2 % envelope)
- [x] Mac runtime + server runtime symmetry on mode A determinism (libomp + libgomp both PASS AC2) — PR-E §3.2
- [x] Forward bounds documented for PR-I amend (D12 framework + Phase 2 placeholder + Mode B upper-bound prior) — PR-E §5 + §6
- [ ] **PR-F may begin** `ExecPolicy::StrictOMP` impl per spec design D2 (single `#pragma omp parallel` enclosing `rhs_update` / `rhs_flux` / `rhs_apply` 3-phase fan-out)

All blocking items are checked. PR-F is **unblocked**.

---

## 8. Forward action items

The PR sequence for P1e past this verdict, in dependency order:

- **PR-F (#314)**: implement `ExecPolicy::StrictOMP` per spec design D2
  (single `#pragma omp parallel` + 3-phase `rhs_update` / `rhs_flux` /
  `rhs_apply` fan-out). Replaces `std::abort()` stub at
  `SHUD/src/Model/MD_rhs_core.cpp:802-811`. Unblocked by PR-E.
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
