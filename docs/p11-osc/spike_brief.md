# P11-osc Spike Brief — minute-scale CVODE stepping: oscillation vs real dynamics

```yaml
epic: P11-osc
status: CLOSED — verdict REAL_DYNAMICS (2026-07-02)
verdict: REAL_DYNAMICS per docs/p11-osc/diagnosis_verdict.md (epic #433; PR #437/#438/#439; control sanity PASS)
date: 2026-07-01 (opened) / 2026-07-02 (closed)
branches:
  outer: research/p11-osc   (from main d127edc)
  shud:  p11-osc            (from openmp-baseline db4ccdb, v1.0.2 tip)
decision_gate: OSC_CONFIRMED | REAL_DYNAMICS | INCONCLUSIVE
```

## One-sentence goal

Localize why `qinyijiang` / `keliya` CVODE steps at minute-scale (19× more
steps than `heihe` for the same 90-day horizon) — numerical flux oscillation
vs real fast dynamics — via read-only dt / state-flip diagnostics; commit to
exactly one A5-gated limiter candidate ONLY if oscillation is confirmed.

## Motivating evidence (B0 archives, 90-day windows, serial Config A)

| case | NumEle | nst | mean dt | netf | ncfn/nst | wall (est. serial) |
|---|---:|---:|---:|---:|---:|---:|
| heihe_x4 | 40,046 | 6,568 | 19.7 min | 0 | 0.70% | ~22 min (measured) |
| heihe | 6,335 | 6,571 | 19.7 min | 0 | 0.03% | ~3.5 min |
| xinanjiang_upstream | 801 | 6,566 | 19.7 min | 2 | 0.02% | <1 min |
| qhh (+lake) | 4,773 | 13,000 | 10.0 min | 0 | 0.02% | ~2 min |
| **keliya** | 484 | **101,188** | **77 s** | 5 | 0.20% | ~30 s |
| **qinyijiang** | 3,155 | **127,536** | **61 s** | 0 | 0.06% | **~30 min (est.)** |

(kashigeer B0 archive lacks `cvode_stats.txt` — blocked deferred-upstream:
the X76 forcing band is missing on both endpoints and `./shud ksge` aborts
before CVODE is constructed, per `benchmarks/kashigeer/B0_output/DEFERRED.txt`;
refresh only after the upstream forcing fix.)

Three structural facts:

1. **nst is forcing/physics-driven, not mesh-driven**: heihe (6,335 ele) and
   heihe_x4 (40,046 ele) take the *same* ~6.57k steps. Parallel RHS (P1e)
   attacks per-step cost; it cannot touch step count.
2. **qinyijiang burns heihe_x4-class wall at 1/13 the mesh size** — its wall
   is step-count-driven. If nst dropped to heihe-like ~6.5k, wall shrinks ~19×.
   This is the only ≥10× CPU opportunity left after ADR-0008/0009/0010 closures.
3. **CVODE is not struggling on qinyijiang**: netf=0, ncfn=0.06%. The error
   controller is *contentedly* walking 1-minute steps. This is compatible with
   both hypotheses below — aggregate stats cannot separate them.

## Hypotheses

- **H-osc (numerical)**: inter-cell flux oscillation (sign-flipping lateral
  surface/subsurface flux, wet/dry threshold chatter, river–element exchange
  reversal) keeps local error estimates high, pinning dt at minutes. Damping
  the oscillation is a legitimate, physics-preserving acceleration.
- **H-dyn (physical)**: the basin's hydrologic response genuinely operates at
  minute timescales in the benchmark window (storm bursts, thin soils, steep
  channels). dt is honest; damping would falsify the physics. Line closes.

## Diagnosis method (read-only, default bitwise-neutral)

### D1. Per-call dt trace (SHUD instrumentation, env `SHUD_DIAG_DT=1`)

Driver structure (SHUD `src/Model/shud.cpp:245`): CVode is called in
`CV_NORMAL` once per `CS.SolverStep` forcing/ET interval inside
`while (t < tnext)`. After each return, read cumulative counters and emit one
CSV row per interval:

```
t_next, delta_nst, delta_nfe, delta_ncfn, delta_netf, h_last, h_cur
```

- `CVodeGetNumSteps` / `CVodeGetLastStep` / `CVodeGetCurrentStep` deltas are
  **read-only** — no solver state mutation, no trajectory risk, no
  SUNDIALS_BUILD_WITH_MONITORING rebuild, no CV_ONE_STEP mode switch.
- Header carries `project_name` (the `./shud <project>` arg; qinyijiang runs
  as `nanlin`) + `solverstep_min` (raw CS.SolverStep, minutes). Canonical
  interval-mean-dt formula (identical in design.md D1 + dt-step-trace spec):
  `mean_dt_seconds = 60 * solverstep_min / delta_nst` — the minutes→seconds
  conversion is owned by the analyzer; `delta_nst = 0` rows are skipped
  (0 contribution to burst numerator and nst denominator). Localizes
  collapse in model time.
- Env unset or ≠ exact string `1` → zero file I/O, zero extra calls (whole
  block behind a strict `=1` compare); default binary path untouched.
  Gate proof: keliya B0 bitwise SHA pre/post.
- Precedent: `SHUD_DUMP_CV_Y` env-gated dump (P1e PR-B, shud.cpp:262) — gate
  *placement* only; its presence-only predicate is deliberately NOT reused.

### D2. State-delta flip counters (SHUD instrumentation, env `SHUD_DIAG_OSC=1`)

Per-edge flux instrumentation at accepted-step granularity is expensive and
trap-prone (f() is invoked at *trial* states — raw f-call history is
ill-defined). Cheap accepted-boundary proxy instead: after each CVode return,
diff per-entity states read from the CVODE state vector
(`N_VGetArrayPointer(udata)` slices `[sf n1 | us n1 | gw n1 | riv n2]` per
`functions.hpp:83-90`; sf/us/gw/riv are family labels — NOT the
`uYsf`/`uYus`/`uYgw` RHS scratch globals, which hold the last internal f()
trial evaluation at t_internal ≠ tout, the P1e PR-B0 hazard class) against
the previous interval and count sign alternations of the interval delta
(rise→fall→rise = 1 flip; deltas ≤ the 1e-6 m epsilon dead-floor hold the
previous sign). O(NY) compare + one state copy per interval.

Output: per-entity cumulative flip counts (typed `ele`/`riv` rows, separate
index spaces) + per-day aggregate totals keyed by
`day_index = floor(t_next / 1440)` (CSV at run end; header carries
project_name / solverstep_min / epsilon_m; exact schemas pinned in the
osc-flip-counters spec). Oscillating cell pairs see-saw water → their
storage deltas alternate sign at high frequency; real monotone
drainage/recharge does not.

### D3. Analyzer (outer `tools/osc_diag/`, Python uv project)

Consumes D1+D2 CSVs (plus an explicit `--case` arg mapping project_name →
benchmark case, since qinyijiang traces say `nanlin`), emits:

- dt histogram (`dt_histogram.csv`) + **burst share**: fraction of total nst
  spent in intervals with mean-dt below thresholds {60 s, 10 s} — the
  speedup ceiling estimator (wall ∝ nst; eliminating bursts caps gain at
  1/(1−burst_share)).
- top-K flip elements + spatial concentration (what % of ELEMENT flips —
  sf+us+gw summed per element — live in the top 1% of elements; river rows
  reported separately, outside the concentration population).
- temporal cross-check: per-day Spearman ρ between daily flip totals and the
  daily count of sub-60 s mean-dt intervals.
- `MARKER:OSC_DIAG_VERDICT` block: verdict + all gate inputs, machine-readable;
  fail-closed (malformed input → non-zero exit, no MARKER block).
- DEFERRED (not in the initial analyzer): adjacency clustering of top-K flip
  elements (are they neighbor pairs?) — D1/D2 CSVs carry no mesh topology;
  candidate for the single INCONCLUSIVE refinement iteration.

### Case matrix (all 90-day, local Mac, serial build)

Serial only: instrumentation sits outside the RHS and P1e A3a bitwise
equivalence pins serial == Config C trajectories, so a Config C leg adds no
diagnostic information.

| case | role | est. wall/run |
|---|---|---:|
| qinyijiang | primary sufferer (127k steps, ~30 min wall) | ~30 min |
| keliya | fast-iteration sufferer (101k steps) | ~30 s |
| xinanjiang_upstream | healthy control (nst=6,566, mean dt 19.7 min, netf=2) | <1 min |

Healthy-control slot = xinanjiang_upstream, NOT heihe: heihe is
`endpoint: server-only` (`benchmarks/heihe/manifest.yaml`) with no Mac
deployment (`docs/case_deployment_map.md` §2.1); xinanjiang_upstream is an
equally healthy profile and locally runnable. No Slurm required; iteration
loop is entirely local. 90-day iron rule holds.

## Kill-gate (quantified, evaluated by D3 verdict)

Machine decision function — TOTAL over all input combinations; identical
text in design.md §Verdict gate and the osc-diag-analyzer spec (single
source; all thresholds pinned here, nothing decided in-tool). Gate inputs
(qinyijiang run):

1. burst_share_60s = fraction of total nst in intervals at mean-dt < 60 s;
   pass = ≥ 0.50 (ceiling ≥ 2×, i.e. beats the closed parallel line before
   any limiter work is justified);
2. top-1% element concentration = share of total element flips (sf+us+gw
   summed per element; rivers reported separately) carried by the top 1% of
   elements; pass = ≥ 0.50 (localized culprits, not basin-wide behavior);
3. flip↔dt temporal correlation = per-day Spearman rank correlation ρ
   between daily flip totals and the daily count of sub-60 s mean-dt
   intervals over the 90-day window; pass = ρ ≥ 0.5 (mechanistic link).

**OSC_CONFIRMED** = (1) AND (2) AND (3).
**REAL_DYNAMICS** = NOT (1) OR NOT (2) — burst share < 50%, or flips
spatially ~uniform (top-1% concentration < 50%).
→ write closure note, close epic, no limiter work.
**INCONCLUSIVE** = (1) AND (2) AND NOT (3) — bursty and localized but no
temporal link → at most ONE follow-up diagnostic refinement (e.g. per-edge
flux instrumentation on top-K elements, or adjacency clustering of top-K
flip elements), then forced re-verdict. No limiter work on an inconclusive
verdict.

Forcing-tracking (do flip spikes track forcing peaks instead of dt minima?)
is NOT a machine-verdict input — no forcing series enters the analyzer; it
is human-assessed corroborating evidence recorded in
`docs/p11-osc/diagnosis_verdict.md`.

Epic aggregation: epic verdict = the qinyijiang MARKER verdict, valid only
if the xinanjiang_upstream control sanity passes (burst_share_60s < 5%);
keliya is corroborating evidence only.

## Optimization phase (GATED — design-only until OSC_CONFIRMED)

If confirmed: exactly **one** limiter candidate chosen from the diagnosed
culprit class (surface lateral deadband / wet-dry smoothing / river-exchange
relaxation / groundwater transmissivity smoothing), implemented behind
`SHUD_FLUX_LIMITER=<name>` env gate, default-off.

Hard design constraints (from CVODE semantics):
- **No f()-call-history state.** CVode re-evaluates f at trial and rejected
  states; "previous call direction" is undefined. Admissible forms: pure
  smooth reformulation (e.g. smoothmax, C1 transitions) or hysteresis keyed
  exclusively off *accepted-step* state updated in a post-CVode-return hook.
- **Mass conservation**: one edge = one flux; limiter applies to the shared
  edge flux, never per-cell asymmetrically.
- **A5 gate tightened for physics-adjacent change**: NSE ≥ 0.99 AND
  KGE ≥ 0.99 AND peak timing offset ≤ 1 output interval AND runoff volume
  ratio within 1% AND water-balance residual non-degrading — stricter than
  the default 0.95 scaling gate, because trajectory change is intended here.
- **ROI gate**: qinyijiang nst reduction ≥ 30% AND wall ≥ 1.5×, else candidate
  is rejected (mirror of P1e AC-S3 discipline).

## Non-goals

- No global linear solver substitution (P8 CLOSED-FINAL per ADR-0008).
- No reltol-family retuning (P9 CLOSED per ADR-0009).
- No GPU, no domain decomposition (P10 remains design-gated per ADR-0010 —
  this epic does NOT reuse the P10 name or touch its gate).
- No production default change of any kind in this epic; everything is
  env-gated and default-off.
- heihe/heihe_x4 need no treatment — healthy profiles per the B0 evidence
  table (the local run-matrix control slot is xinanjiang_upstream, since
  heihe is server-only).

## Branch / merge model (user-approved 2026-07-01; topology pinned at review)

- Outer work → `research/p11-osc`; PR-D1/D2/D3 are STACKED PRs based on
  `research/p11-osc` (non-default base → GitHub close-keywords inert →
  manual `gh issue close` per CLAUDE.md branch model); a post-verdict
  capstone PR merges `research/p11-osc` → `main` — nothing from this epic
  reaches `main` pre-verdict.
- SHUD instrumentation commits → SHUD branch `p11-osc` (NOT openmp-baseline);
  merge `p11-osc` → `openmp-baseline` only after the spike verdict lands.
  Master remains untouched; no forks; `.gitmodules` unchanged.

## References

- B0 archives: `benchmarks/<case>/B0_output/cvode_stats.txt` (evidence table)
- ADR-0008 (P8 closure), ADR-0009 (P9 closure), ADR-0010 (program status; P10
  gate definition — namespace this epic deliberately avoids)
- `docs/p1e/p1e_academic_summary.md` (nst mesh-invariance, Config matrix)
- SHUD driver: `SHUD/src/Model/shud.cpp:219-248`; env-gate precedent L257-262
- A5 pipeline: `tools/a5/` (PR #422/#425)
