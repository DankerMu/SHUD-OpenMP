# P12-nvec Spike Brief — deterministic hybrid NVector parallelization

```yaml
epic: P12-nvec
status: EPIC CLOSED (capstone 2026-07-02) — TIER1_ADOPT + TIER2_GO + G-E4 PASS (Config E2 certified)
date: 2026-07-02
branches:
  outer: research/p12-nvec   (from main d4ec052, post-P11-osc capstone)
  shud:  p12-nvec            (from openmp-baseline 75afb2b; PR-N2 makes ZERO SHUD change, pointer @ d8f736c)
decision_gates: TIER1_ADOPT | TIER1_CLOSE ; TIER2_GO | TIER2_NO-GO (nested, Tier 2 gated on Tier 1)
verdict:
  g_e1: PASS (PR-N1 keliya bitwise; PR-N2 heihe_x4 C≡E identical counters + rivqdown SHA at N∈{1,8,16})
  g_e2: TIER1_ADOPT (heihe_x4 speedup(E/C) ≥ 1.10× at N=8 and N=16; bar 1.10× pinned)
  g_e3: TIER2_GO ((i) TIER1_ADOPT ∧ (ii) reduction share 35.68%/30.46% ≥ 10% ∧ (iii) Amdahl proj ≥ 1.15×)
  g_e4: PASS (PR-N3: keliya prod-B/forced-B256 + heihe_x4 cross-N bitwise → A4 documented → A5 tightened nse=1.0000/kge=0.9999; E2/E wall 1.377×@N8 / 1.369×@N16, projection delta −0.42%; re-baseline accepted)
  authority: docs/p12-nvec/tier1_verdict.md + docs/adr/0011-p12-nvec-tier1-verdict-and-tier2-gate.md + docs/p12-nvec/pr_n3_ge4_verdict.md + docs/p12-nvec/pr_n3_rebaseline_decision.md
```

## One-sentence goal

Lift the Config C Amdahl ceiling (sp@16 = 1.946×, serial fraction ≈ 48% on
heihe_x4) by parallelizing CVODE-internal NVector **element-wise** operations
with OpenMP while keeping every **reduction** serial — preserving the release
guarantee (bitwise-identical results across thread counts AND vs the serial
baseline) — and, ONLY if profile + ROI evidence demands it, a gated Tier 2
that replaces serial reductions with fixed-tree deterministic reductions at
the cost of a one-time controlled re-baseline.

## Motivating evidence

| fact | value | source |
|---|---|---|
| Config C scaling ceiling | sp@8=1.804×, sp@16=1.946× (heihe_x4, Slurm 10764) | RELEASE.md §Scaling profile (cpu-accel-v1.0.2) |
| Amdahl serial fraction @x4 | ≈ 0.48 (from sp@16=1.946) | same matrix |
| Config D (full OpenMP NVector) | REFUTED — trajectory drift 10-25% at day 90 | P1e config matrix; root cause = `reduction(+:sum)` order varies with thread count |
| Krylov work grows with mesh | nfeLS/nfe = 1.819 (heihe, N≈19k) → 4.526 (heihe_x4, NY≈120k) | docs/p8pre/n8_profile_verdict.md §3.1 |
| NumY thresholds | heihe_x4 NY ≈ 120k > 100k SUNDIALS OpenMP-NVector viability threshold; keliya NY ≈ 1.5k (below — bitwise-gate case only, not a perf case) | master plan §P8-NVector 规模门槛 |
| Step count is physics | P11-osc REAL_DYNAMICS — nst not reducible; ONLY per-step cost remains attackable | docs/p11-osc/diagnosis_verdict.md |
| All solver-substitution lines closed | KLU/AMG/GPU per ADR-0008; outer-policy per ADR-0009; P10 deferred per ADR-0010 | ADRs |

Legacy anchors (this epic executes what they anticipated):
- master plan **§P8-NVector** (原 P8a): decision matrix "NumY > 100,000 → 启用（仍需 A/B）";
  task P8-NVector.2 (vector-op share profiling, <10% skip rule); task P8-NVector.5
  already flagged the cross-thread reduction ULP hazard we now design around.
- master plan **legacy §P9 deterministic reduction / compensated summation**: the
  Tier-2 strategy menu (fixed pairwise / Kahan-Neumaier / deterministic tree) +
  the acceptance frame ("偏离 B1b 不是错误——不能和并行 bug 混在一起") + the
  "new numerical reference 单独立项" clause = the re-baseline discipline Tier 2 uses.

## Core determinism argument (single source; design.md D2 summarizes — this section is authoritative)

NVector operations split into two determinism classes:

1. **Element-wise ops** (`nvlinearsum`, `nvprod`, `nvdiv`, `nvscale`, `nvabs`,
   `nvinv`, `nvaddconst`, `nvconst`, `nvcompare`, …): each output element
   `z[i]` is computed from inputs at the same index by a fixed per-element
   expression. OpenMP `parallel for` over `i` changes WHICH thread computes an
   element, never the floating-point operation order WITHIN an element →
   bitwise-identical for any thread count, and bitwise-identical to the serial
   backend (same per-element special-case branches as nvector_serial).
   Same determinism class as the StrictOMP RHS owner-local-write rule.
2. **Reduction ops** (`nvdotprod`, `nvwrmsnorm`, `nvwrmsnormmask`, `nvmaxnorm`,
   `nvmin`, `nvwl2norm`, `nvl1norm`, `nvminquotient`, `nvconstrmask`,
   `nvinvtest`, fused `nvdotprodmulti` / `nvwrmsnormvectorarray` /
   `nvwrmsnormmaskvectorarray`, + any local/single-buffer variants): accumulate
   across elements. OpenMP `reduction(...)` partitions the accumulation by
   thread count → rounding-order drift → the exact Config D failure.

**Tier 1 (hybrid NVector)** parallelizes class 1 and pins class 2 to serial
loops → expected FULL bitwise equality with Config C at every thread count
(A3a-strength gate). **Tier 2** replaces class-2 serial loops with fixed-tree
deterministic reductions (block size FIXED, independent of thread count;
combine order FIXED) → cross-thread bitwise holds, but the summation order
differs from today's serial left-to-right once → one-time re-baseline + A5
certification of the new reference.

## Mechanism (Tier 1)

- Create `N_VNew_OpenMP(NY, nthreads, sunctx)` as today (Config D machinery),
  then overwrite the reduction entries of `v->ops` with SHUD-owned serial
  implementations written against the **generic** API (`N_VGetArrayPointer` +
  plain loop). NEVER call `*_Serial` backend functions on an OpenMP-content
  vector (content-struct layout aliasing is undefined-behavior territory).
- CVODE allocates every internal temporary via `N_VClone(udata)`; clone copies
  the ops table → the override propagates to ALL internal vectors automatically.
  (Verify `N_VCloneEmpty` path too; audit that no CVODE 6.0.0 path re-fetches
  stock OpenMP ops.)
- The exact override list is pinned by auditing `nvector_openmp.c` for
  `reduction(` pragmas (implementation task deliverable, committed as an audit
  table in the PR).
- Build config: new compile-time flag `SHUD_NVEC_HYBRID` (default 0/off; only
  meaningful with `SHUD_USE_OPENMP_NVECTOR=1`) → **Config E** = Config C RHS +
  OpenMP NVector + serial-reduction override. Default builds unchanged; release
  line (cpu-accel-v1.0.x) untouched.

| Config | RHS | NVector | determinism |
|---|---|---|---|
| C (release) | StrictOMP | Serial | bitwise cross-N == serial |
| D (research, refuted) | StrictOMP | OpenMP (stock) | drift 10-25% |
| **E (this epic)** | StrictOMP | OpenMP element-wise + serial reductions | expected bitwise cross-N == Config C |
| E2 (Tier 2, conditional; `SHUD_NVEC_DETRED`) | StrictOMP | OpenMP element-wise + fixed-tree reductions | bitwise cross-N; one-time shift vs C → re-baseline + A5 |

## Diagnosis first (PR-N0): op-share profile

Existing timer buckets stop at `t_CVODE_raw` (whole CVode call) — they cannot
see inside. PR-N0 adds an env-gated (`SHUD_NVEC_PROF=1`, strict `=1` predicate
per P11-osc discipline) ops-wrapper profiler: wrap every NVector op pointer
with a counting/accumulating shim (monotonic clock; ~ns overhead per call vs
~µs per op at NY≈120k; normative gate = measured wall overhead ≤ 2% on the
heihe_x4 Config C N=16 leg, 3-run median gate-on vs gate-off). Composition
order pinned: hybrid overrides (Config E) install first, profiler wraps last
(outermost). Deliverable: per-op-class share table — element-wise vs
reduction vs non-NVector remainder of `t_CVODE_raw` — with pinned
measurement cells (all profile legs on `SHUD_ENABLE_PROFILE=1` builds; N =
cfg numthreads, `SHUD_RHS_THREADS` unset; every row states its N): heihe_x4
90-day server Slurm Config C @ N∈{1,16} + keliya sanity Mac Config C @ N=1 +
heihe_x16 short window (pinned 10 model-days) Config C @ N=16. G-E3 consumes
the N=16 shares; one Config E @ N=16 leg follows PR-N1 (filed with PR-N2) as
the t_red cross-check. Default path untouched (gate off = stock behavior;
bitwise gate evidence per P11-osc precedent).

## Kill / ROI gates (quantified; authoritative prose — specs pin identical thresholds and cells)

- **G-E1 (Tier-1 bitwise)**: Config E at N∈{1,2,4,8} on keliya → all model
  outputs SHA-identical to each other AND to Config C baseline. HARD gate: any
  mismatch → investigate; if the mismatch is in element-wise ops → implementation
  bug (fix); if traced to an unlisted reduction → extend override list. No
  tolerance fallback inside Tier 1 (that fallback IS Tier 2).
- **G-E2 (Tier-1 ROI)**: heihe_x4 server matrix N∈{1,8,16}, Config E vs Config C:
  TIER1_ADOPT if wall improvement ≥ 1.10× at N=8 or N=16 with A3a bitwise PASS
  (expected: identical trajectories → identical nst/nfe/ncfn/ncfl/netf;
  wall-only comparison); else TIER1_CLOSE (document, keep code默认关闭 or
  revert per verdict).
- **G-E3 (Tier-2 GO)**: ALL of — (i) Tier-1 verdict = TIER1_ADOPT; (ii) PR-N0
  profile shows reduction-op share ≥ 10% of t_CVODE_raw at heihe_x4 or x16,
  each read from its pinned Config C N=16 profile leg (legacy P8-NVector.2
  rule; the share is N-dependent, so the gate leg is pinned); (iii) Amdahl
  projection ≥ 1.15× additional wall over adopted Config E at N=16 — pinned
  inputs and formula: t_red = PR-N0 heihe_x4 Config C N=16 absolute reduction
  total_ns (serial reduction time is N-invariant; transfers to Config E via
  the bitwise-identical trajectory; cross-checked by the PR-N2 Config E N=16
  profile leg), wall_E16 = PR-N2 measured Config E median wall at N=16,
  projection = wall_E16 / (wall_E16 − t_red·(1 − 1/16)). Any one fails →
  TIER2_NO-GO (closure note; fixed-tree code NOT written).
- **G-E4 (Tier-2 acceptance, only on GO)**: cross-thread bitwise — keliya
  N∈{1,2,4,8} identical at production B PLUS a forced-small-B leg (≥ 4 blocks,
  ≥ 2 combine levels; production B=4096 ≥ keliya NY≈1.5k would leave the tree
  untested) AND heihe_x4 cross-N N∈{8,16} identical (cvode_stats counters +
  rivqdown SHA, from the E2-vs-E server matrix) — + A4 max_ulp vs Config E
  reference documented + full A5 hydrology acceptance via tools/a5 run WITH
  the TIGHTENED override config `tools/a5/config/a5_thresholds.p12_tier2.yaml`
  (PR-N3 deliverable; tightened vs tool defaults 0.90/0.85/±4/±3% per P11-osc
  physics-adjacent precedent: nse/kge min 0.99, peak_timing_offset
  max_abs_steps 1 [≤ 1 output interval], runoff_volume_ratio 0.99–1.01
  [≤ 1%], water_balance_residual max 0.05 tool cap; tools/a5 has no cross-run
  "non-degrading" semantic → candidate-vs-reference residuals recorded
  side-by-side, candidate ≤ reference required) + explicit new-golden
  re-baseline decision recorded (mode-C-tune precedent: A3a does not apply
  across the reduction-order change; applies within). Fail → revert to
  Config E (`SHUD_NVEC_DETRED` stays 0).

## PR mapping (stacked on research/p12-nvec; manual issue close per CLAUDE.md)

- **PR-N0** (SHUD + outer evidence): `SHUD_NVEC_PROF=1` ops-wrapper profiler +
  bitwise-neutrality evidence + op-share tables (x4 C@N∈{1,16} / x16 10-day
  C@N=16 / keliya C@N=1) + share verdict.
- **PR-N1** (SHUD + pointer bump): hybrid NVector override (`SHUD_NVEC_HYBRID`,
  Config E) + reduction-audit table + G-E1 keliya bitwise evidence legs.
- **PR-N2** (outer): server Slurm scaling matrix Config E vs C (incl. the
  Config E N=16 gate-on profile leg) + G-E2 verdict + ADR-0011 + master plan
  §P12-nvec anchor + glossary §P12-nvec 集合 + Tier-2 gate decision (G-E3)
  recorded.
- **PR-N3** (CONDITIONAL on G-E3 GO; SHUD + outer): fixed-tree deterministic
  reductions behind compile-time flag `SHUD_NVEC_DETRED` (default 0, requires
  SHUD_NVEC_HYBRID=1 → Config E2; block size fixed; Neumaier compensation
  optional per legacy §P9 menu) + tightened A5 override config + G-E4
  acceptance + re-baseline decision doc.
- **Capstone**: research/p12-nvec → main merge + openspec archive.

## Constraints (inherited, non-negotiable)

- SHUD commits → `openmp-baseline` lineage only via branch `p12-nvec`; no master push; no fork.
- Server runs: Slurm 三铁律 (sbatch from /scratch; --output on /scratch; scripts on /scratch); never login node.
- 90-day case truncation; keliya gates on Mac; heihe_x4/x16 on server only.
- Python via uv only. Default builds bitwise-identical (CI build-and-compare green).
- Release line cpu-accel-v1.0.x immutable; Config E ships only through a future tag after adoption.

## Out of scope

- GPU (closed, ADR-0008); domain decomposition (§P10, own planning turn);
  fused-op ENABLEMENT (CVODE fused ops stay at stock defaults — only override
  what stock uses); the decoupled 5-solver loop vectors (shud.cpp:495+, stay
  Serial); RHS changes of any kind (f.cpp untouched); MPI/ManyVector.
