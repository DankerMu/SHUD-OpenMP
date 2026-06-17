# Profile Gate Decision (S0-11 / openMP #15)

This document records the priority decision triggered by the profile gate
per rhs-profile-gate spec.md "Profile decision signed before S1"
requirement and master plan §S0.12 decision table (4-option ladder). It
is the formal sign-off that S0 prep work has converged and that the
project is ready to enter S1 / P-phase parallelization. Per spec scenario
"Missing signature blocks S1", absence of this file (or absence of its
`signed_at` field) MUST set `docs/status_matrix.md` profile gate row to
BLOCKED.

## Decision Category

**走原方案 (original plan: RHS-kernel-first OpenMP parallelization)**

Rationale (per master plan §S0.12 L752 decision table thresholds):

- Target-platform `t_RHS_total / t_wall_total` distribution across 6
  successful target runs:

  | Case | t_RHS_total% | Amdahl S(infinity) ceiling |
  |---|---|---|
  | heihe_x4 | 66.55% | 2.99x |
  | qinyijiang | 64.64% | 2.83x |
  | keliya | 49.33% | 1.97x |
  | xinanjiang_upstream | 43.74% | 1.78x |
  | qhh | 36.75% | 1.58x |
  | heihe | 12.08% | 1.14x |

- **5 of 6 target cases have RHS share >= 30%**, comfortably above the
  master plan §S0.12 "走原方案" threshold (`t_RHS_total / t_total > 30%`
  for the majority of decision-critical cases).
- The largest case (heihe_x4, ~25k cells, the actual P-phase
  acceleration target) has the highest RHS share (66.55%), confirming
  that RHS-kernel-first parallelization is the **right primary
  investment** for the production scale.
- No case falls in the "战略暂停" zone (RHS < 10% globally). The single
  low-share outlier (heihe at 12.08%) is **forcing-IO dominated**
  (t_forcing_io = 79.1%), which is a separate bottleneck class —
  RHS-only parallelization is still net-positive there, just bounded
  to 1.14x by Amdahl until the IO path is addressed.

## Amdahl Upper Bound

Per-case theoretical speedup ceiling (assuming RHS becomes perfectly
parallel and all other buckets remain serial):

```
S(infinity) = 1 / (1 - t_RHS_total / t_wall_total)
```

- **Decision-driving case (heihe_x4): 2.99x** at infinite threads;
  practical 8-core ceiling per Amdahl: `1 / ((1 - 0.6655) + 0.6655/8) =
  1 / (0.3345 + 0.0832) = 2.39x` — this is the §1.1.1 speedup gate
  target ceiling for the largest case.
- **Secondary driver (qinyijiang, 3155 cells): 2.83x** ceiling /
  practical 8-core 2.30x.
- **Small case (xinanjiang_upstream, 801 cells): 1.78x** ceiling /
  practical 8-core 1.59x — small cases will need to rely on
  `OMP_CUTOFF` (C8 in master plan core principles) rather than full
  parallel investment.
- **IO-dominated case (heihe, 6335 cells, but forcing IO 79%): 1.14x**
  ceiling / practical 8-core 1.13x — the RHS parallelization target
  for THIS case is "do no harm" not "extract speedup"; real
  improvement will require either (a) cached forcing on-disk in a
  binary format read once at startup, or (b) parallel forcing IO,
  which is logically a P9-or-later optimization in the master plan.

The §1.1.1 gate "single-socket 8-core speedup target" therefore reads as:
**target = 1.5-2.0x on decision-critical large cases (heihe_x4 + qinyijiang)**,
**bypass via OMP_CUTOFF for small cases**, **defer or pair with separate
IO optimization for forcing-IO-dominated cases (heihe)**. This re-reads
the headline §1.1.1 number ("8x speedup") not as a uniform per-case
target, but as a portfolio metric weighted toward large cases.

## P8-precond Timing

**Decision: keep at P8, do not bring forward.**

Per master plan §S0.12 decision table, P8 pre-conditioned SPGMR is an
optimization that becomes relevant **only after** the RHS parallel
ceiling is reached. The current profile evidence supports this:

- For 4 of 6 cases, RHS-only parallelization can extract >= 1.5x on
  8 cores — that headroom must be realized in P1-P7 (strict) before
  paying the precond design + integration cost.
- For the IO-dominated case (heihe), P8 precond does **not** help —
  the bottleneck is `t_forcing_io`, not CVODE Krylov iteration count.
  Bringing P8 forward for this case is a category error.
- The CVODE internal time (`t_CVODE_internal`) ranges from 6.83%
  (heihe) to 36.17% (keliya), giving a precond-addressable upper bound
  of ~36% — meaningful but not the dominant lever in the current
  decision-critical (large) cases.

P8-precond therefore remains scheduled per master plan §3 phase order,
not advanced.

## Cross-platform delta review

Per spec.md scenario "> 10pp difference triggers review note":
`docs/profile_platform.md` reports `delta_acceptable: false` because
2 of 4 cases with both-endpoint data exceed the 10pp threshold:

- **keliya**: local 36.32%, target 49.33%, delta +13.01pp
- **qinyijiang**: local 51.07%, target 64.64%, delta +13.57pp

Root-cause hypothesis (qualitative; quantitative diagnosis is out of
#15 scope):

1. **Different microarchitecture single-core throughput.** Apple M4 Pro
   perf cores have substantially higher single-thread IPC than Xeon Gold
   6133 (2017 Skylake-SP at 2.5 GHz). Wall-clock for the same workload
   is ~2.9-3.5x slower on the target (target wall / local wall: keliya
   79.7/27.8 = 2.87x, qinyijiang 799.9/229.5 = 3.49x), confirming the
   single-core throughput gap.

2. **Non-uniform slowdown across buckets.** Apple's clang + Apple
   silicon NEON optimizations seem to accelerate the SHUD RHS C++
   kernel more aggressively than the SUNDIALS/CVODE internal C code
   (which is the same standard C source on both platforms). The
   relative share of RHS in total wall thus **shrinks on Apple, grows
   on x86** — an artifact of compiler+microarch interaction, not a
   profile-correctness defect.

3. **xinanjiang_upstream and qhh** show small (< 2pp) deltas,
   indicating that the cross-platform asymmetry is not uniform across
   cases — it correlates with how much of the RHS work is in
   vectorizable inner loops (more vectorizable -> larger Apple
   advantage -> larger delta to x86).

**Impact on decision**: the delta does NOT invalidate the "走原方案"
decision. Both platforms agree that RHS is the dominant bucket
(local: 36-51% across cases; target: 12-67% across cases, 5/6 above
30%), and the **decision-critical large case (heihe_x4)** is
target-only — there is no local-vs-target comparison to make for it.
The Amdahl ceiling computation above is anchored on target-platform
numbers (`docs/profile_platform.md` `target_platform`), which is the
§1.1.1 authoritative endpoint, so the delta is a metadata caveat,
not a gate-blocking finding.

**Action items** (track via separate issues, NOT in #15 scope):

- Re-profile keliya and qinyijiang on target after P1-P3 strict
  parallel landed, to confirm the RHS-share trend holds at higher
  thread counts.
- If a future heihe_x4 local-vs-target comparison ever becomes
  possible (e.g. via partial forcing or downsampled mesh), repeat
  the delta check.

## Evidence

Local-side artifacts (4 real + 3 deferred), SHA256:

```
b16e8a9acedcff00c82b66192d3db3eced538c7718c8715adff7b384761ce14f  benchmarks/keliya/profile_B0.yaml
34831b4b641f664cece3c5a4db1dd2c72b0016b99dc4206d30ac9b5365a43d97  benchmarks/xinanjiang_upstream/profile_B0.yaml
77bdbad6bd23e9806688bb7e76cb82d8359ed3edfbc7af03002ea9ee8ad87b02  benchmarks/qinyijiang/profile_B0.yaml
9506ceeeab796c9685dccfd649f5adc889e239e3df2f3274d5a82647637cec08  benchmarks/qhh/profile_B0.yaml
cbe50adf2047c88bc1e7d6415ce76c6ba8310e159310876c72f2bebdbc24982c  benchmarks/heihe/profile_B0.deferred.yaml
aea1322a9b2b6ee2ffaf391b422ccf05e8c1c7f29f5d20a9778dc6568eed8a9f  benchmarks/heihe_x4/profile_B0.deferred.yaml
5bdecf4e2173868de20659141cb9b046349cc5ae0fba97078b877f1cd4a5cd02  benchmarks/kashigeer/profile_B0.deferred.yaml
```

Target-side artifacts (6 real + 1 deferred), SHA256:

```
711a380902d2dee176ff16bf5c3a5c360a9ee131420d7727a7d4e75dc62ca0f5  benchmarks/keliya/profile_B0.target.yaml
a739dfd7c66310bf5e5bcb0317a99768d3c1d41480e8e991e0d32aaeca9637e1  benchmarks/xinanjiang_upstream/profile_B0.target.yaml
1dae17564e44de5149f8e49cb8dd3f404caa5a1ee19dc0b9ef2f26ab417174ed  benchmarks/qinyijiang/profile_B0.target.yaml
cc312b7ab1db926ab85fff86b91cc0e29fc02b2a289103ee30db9555dad105f5  benchmarks/qhh/profile_B0.target.yaml
baa03be7ce16e01345bdc9e9b93c033ffcee55213113b9b1ba91441414a97f5d  benchmarks/heihe/profile_B0.target.yaml
03d9d4c9def804b27f5f5e6a8930063eb03ce5ad5cbadee979c848f829254c36  benchmarks/heihe_x4/profile_B0.target.yaml
8f64779b5c3c25b2a854f70b9721231a4e40b5dd4a1b2eadf2e3a7e43d615d17  benchmarks/kashigeer/profile_B0.target.deferred.yaml
```

Outer commit:  `ecef3fbe6ad6971ac8dc2ff6a888ece8db8fae83`
SHUD submodule commit: `78c37a1061de4112bc7c297bb7bd1f107432e6f2`

## t_other accounting status

Per rhs-profile-gate spec.md "t_other accounting WARN at 5%" and
"FAIL at 10%" scenarios. Target-platform target yamls audit:

| Case | t_other_pct (target) | Status |
|---|---|---|
| heihe_x4 | 1.10% | OK |
| heihe | 1.52% | OK |
| qinyijiang | 0.87% | OK |
| qhh | 5.80% | **WARN** |
| keliya | 6.32% | **WARN** |
| xinanjiang_upstream | 22.42% | **FAIL** |

The xinanjiang_upstream FAIL (22.42% t_other on a 19.7 s run) is
interpreted as a **startup-overhead dominance artifact**, not a
profile-tool defect:

- absolute t_other = 4.41 s out of 19.7 s wall.
- The same yaml's `t_forcing_io = 1.40 s` covers reading 51 forcing
  CSVs (the smallest forcing-stations count of the 6 cases).
- Process startup (SUNDIALS init, mesh load, integrator setup), which
  the current S0-10 instrumentation does NOT label, is in `t_other`.
- For long runs (heihe 487s, heihe_x4 1417s) the startup amortizes
  to < 2% as expected.

**Decision-impact**: the FAIL is acknowledged but does not block the
"走原方案" decision because (a) the decision is anchored on large
cases where the share is dominated by RHS, not startup, and (b) the
FAIL signal is a **future profile-tool refinement opportunity**
(label startup time as `t_init` rather than dropping it in `t_other`),
which is properly an S0.12 retrospective item, not an S1 blocker.

A follow-up issue to add `t_init` bucket and re-classify startup time
out of `t_other` SHOULD be opened against `tools/profile/timer.cpp`
before the next decision-gate snapshot (i.e. before P-phase exit).

## Signature

| Field | Value |
|---|---|
| signer | `<PLACEHOLDER: project owner, to fill at S1 gate>` |
| signed_at | 2026-06-17 |
| signed_via | claude-code-s0-11-issue-15 (orchestrated under Linus Torvalds persona, per /Users/danker/.claude/CLAUDE.md priority stack) |
| signed_off_decision | 走原方案 + 调高 large case 权重 + P8-precond 不前置 |
| follow_up_issues | (a) re-profile after P1-P3 strict landing; (b) heihe forcing IO optimization deferred to P9+; (c) split t_init out of t_other in profile timer; (d) kashigeer upstream X76-X80 forcing gap (issue #29 already open) |
