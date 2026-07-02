# ADR-0011: P12-nvec Tier-1 Hybrid NVector Adoption + Tier-2 Deterministic-Reduction Gate

## Status: Accepted (2026-07-02)

- **Date**: 2026-07-02
- **Deciders**: DankerMu + Claude orchestrator (per PR-N2 #444 heihe_x4 90-day scaling matrix evidence + G-E2/G-E3 pinned-rule verdicts)
- **Owner**: SHUD-OpenMP 改造工程 / P12-nvec deterministic hybrid NVector epic
- **Tags**: p12-nvec / tier1 / tier2 / nvector / openmp / hybrid / deterministic-reduction / bitwise / amdahl / roi / adoption / gate
- **Supersedes**: none
- **Superseded by**: none
- **Related**: ADR-0010 (CPU acceleration status consolidation; §Deferred P10 + §NOT-doing frame the forward space this epic acts in) + ADR-0009 (P9 outer-policy closure; the §P9 1.5× ROI bar precedent the G-E2 1.10× bar is contrasted against) + ADR-0008 (P8 solver-substitution closure; Config D drift root cause) + ADR-0002 (P1e Path 1 StrictOMP RHS baseline that Config C/E build on) + master plan §P12-nvec (this ADR) + §P11-osc (adjacent closure entry; "confirmed path creates no implementation" discipline mirrored in the TIER2_NO-GO first-class-outcome design) + `docs/p12-nvec/tier1_verdict.md` (measured verdict doc) + `docs/p12-nvec/spike_brief.md` + `.review-evidence/p12-nvec/{pr-n0,pr-n1,pr-n2}/`

---

## Context

Config C (P1e StrictOMP RHS + Serial NVector, release cpu-accel-v1.0.2) is the
production CPU baseline. Its parallel speedup saturates at sp@16 = 1.946× on
heihe_x4 (Amdahl serial fraction ≈ 0.48). PR-N0 profiling (`SHUD_NVEC_PROF`
ops-wrapper, `SHUD_ENABLE_PROFILE=1` builds) located the serial remainder
INSIDE the `CVode()` call: at Config C N=16 on heihe_x4, of `t_CVODE_raw` =
400.430 s, NVector element-wise ops are **49.926%**, reductions **35.677%**,
non-NVector remainder 14.397%. At N=1 the same absolute NVector work is a
smaller share (StrictOMP RHS dominates single-threaded), so the NVector share
GROWS with N — exactly the Amdahl ceiling.

The P12-nvec epic (`openspec/changes/p12-nvec`, epic #441) attacks this in two
tiers, each strictly gated:

- **Tier 1 (Config E, PR-N1)**: build on the existing
  `SHUD_USE_OPENMP_NVECTOR=1` creation path (`N_VNew_OpenMP`), then overwrite
  ONLY the reduction ops of the vector's `ops` table with SHUD-owned serial
  loops (generic API, `src/Model/MD_nvec_hybrid.{hpp,cpp}`, ~150 LOC). Element-
  wise ops keep the stock OpenMP implementation. Determinism argument: element-
  wise ops compute each output element from same-index inputs by a fixed per-
  element expression, so OpenMP `parallel for` over the index changes which
  thread computes an element but never the per-element FP order → bitwise-
  identical across thread counts AND vs Config C; reductions stay serial left-
  to-right → no thread-count-dependent rounding drift (the exact Config D
  failure mode, refuted at 10-25% trajectory drift per ADR-0008). Compile-time
  flag `SHUD_NVEC_HYBRID` (default 0; Makefile errors on HYBRID=1 without the
  NVector flag). PR-N1 G-E1 HARD gate: keliya Config E N∈{1,2,4,8} all model
  outputs SHA-identical to each other AND to Config C → **PASS** (plus a GCC
  cross-toolchain spot leg on heihe, and PROF×HYBRID composition evidence).

- **Tier 2 (Config E2, PR-N3, CONDITIONAL)**: replace the Tier-1 serial
  reduction loops with fixed-tree deterministic reductions (compile-time flag
  `SHUD_NVEC_DETRED`, block size B independent of thread count, fixed binary-
  tree combine) — cross-thread bitwise by construction, but the summation order
  shifts once vs Config C serial → a one-time controlled re-baseline + A5
  certification of the new reference.

PR-N2 (#444) ran the server Slurm scaling matrix that decides (a) whether
Tier-1 delivers enough wall ROI to ADOPT (G-E2) and (b) whether Tier-2 is worth
building (G-E3). This ADR records both verdicts and their consequences.

---

## Decision

**Adopt Config E as the Tier-1 hybrid NVector build leg (TIER1_ADOPT), and open
Tier-2 (TIER2_GO).**

1. **G-E2 = TIER1_ADOPT.** Pinned rule (verbatim, bar NOT moved):
   *TIER1_ADOPT iff speedup(E/C) ≥ 1.10 at N=8 or N=16 AND bitwise PASS, else
   TIER1_CLOSE.* The heihe_x4 90-day matrix returned bitwise PASS at every N
   (Config C ≡ Config E: identical cvode_stats counters + equal `rivqdown` SHA
   at N∈{1,8,16}) and speedup(E/C) = **1.4124×** at N=16 / **1.3094×** at
   N=8 — both clear the 1.10× bar decisively → **TIER1_ADOPT**. Config E ships
   as an opt-in build leg (`make shud_omp SHUD_USE_OPENMP_NVECTOR=1
   SHUD_NVEC_HYBRID=1`); release-tag promotion is a post-epic follow-up (release
   line cpu-accel-v1.0.x is NOT modified; default `make shud`/`make shud_omp`
   builds stay byte-identical, CI build-and-compare green).

2. **G-E3 = TIER2_GO.** GO iff ALL three inputs pass, each measurement cell
   pinned: (i) Tier-1 = TIER1_ADOPT → **PASS**; (ii) PR-N0 Config C N=16
   reduction share ≥ 10% on heihe_x4 (35.677%) OR heihe_x16 (30.461%), verbatim
   either/OR aggregation → **PASS**; (iii) Amdahl projection
   `wall_E16 / (wall_E16 − t_red·(1 − 1/16))` ≥ 1.15× with `t_red` =
   142.860 s (PR-N0 heihe_x4 Config C N=16 absolute reduction total_ns, cross-
   checked by this PR's Config E N=16 gate-on profile leg) and `wall_E16` =
   **491.618 s** (this matrix's Config E median at N=16) → projection =
   **1.3744×** ≥ 1.15 → **PASS**. All three pass → **TIER2_GO**: PR-N3
   (fixed-tree deterministic reductions) is unblocked (issue #445 blocked-by
   marker lifted with a gate-values comment). This PR is the sole owner of the
   GO/NO-GO consequence execution; the capstone only verifies.

The forward direction preserves ADR-0010: P10 CPU domain decomposition stays
DESIGN-ONLY (deferred), GPU is NOT pursued, and P1e StrictOMP RHS +
`SHUD_SPGMR_MAXL` small-case opt-in remains the production baseline — Config E
adds an opt-in acceleration leg for large NY (≥ 100k) cases without disturbing
any of that.

---

## Outcome Table

heihe_x4 90-day, {Config C, Config E} × N∈{1,8,16}, 3-run median wall
(runner-measured model wall; `SHUD_RHS_THREADS` unset; N = single cfg-NUM_OPENMP
knob = OMP_NUM_THREADS). All cells: nst=6575 nfe=6741 ncfn=51 ncfl=3620 netf=0
(identical across C/E and across N — full bitwise determinism).

<!-- OUTCOME_TABLE_START -->
| Config | N=1 wall (s) | N=8 wall (s) | N=16 wall (s) |
|---|---|---|---|
| C | 1288.953 | 724.325 | 694.343 |
| E | 890.502 | 553.194 | 491.618 |
| speedup(E/C) | 1.4474× | 1.3094× | 1.4124× |
<!-- OUTCOME_TABLE_END -->

Bitwise cross-check (equal-N C@N vs E@N): PASS at N∈{1,8,16} — identical
counters + equal `rivqdown` SHA `b5e4b0a2…` (single value across ALL 18 wall
runs + the profile leg). Cross-N determinism (bonus): identical too.

---

## Evidence anchors

- **Matrix + bitwise + G-E2/G-E3**: `.review-evidence/p12-nvec/pr-n2/README.md`
  (18 wall runs + 1 profile leg; job IDs 11336-11342 on cn10/11/12/14/15/16/22;
  build commit SHUD d8f736c; `analyze_matrix.py` recomputes medians + verdicts).
- **G-E3(ii) reduction shares + G-E3(iii) t_red**:
  `.review-evidence/p12-nvec/pr-n0/README.md` §5c (heihe_x4 35.677% / t_red
  142.860 s) + §5d (heihe_x16 30.461%).
- **G-E1 Tier-1 bitwise + Config E mechanism**:
  `.review-evidence/p12-nvec/pr-n1/README.md` (keliya 4-leg SHA + audit table +
  clone-propagation assert + GCC spot leg).
- **Config E N=16 gate-on profile leg (t_red cross-check)**:
  `.review-evidence/p12-nvec/pr-n2/profile_leg/` (Config E reduction total_ns vs
  PR-N0 t_red under the bitwise-identical trajectory).
- **Verdict doc (measured ratios + formula)**: `docs/p12-nvec/tier1_verdict.md`.

---

## Consequences

**Positive:**
- Large-NY CPU acceleration beyond the Config C Amdahl ceiling, at ZERO
  re-qualification cost: Config E is trajectory-identical to Config C, so every
  existing acceptance artifact (A3a bitwise, A5 hydrology) transfers unchanged.
- ~150 LOC surface, all behind a default-off compile-time flag; release line and
  default builds untouched (auditable, revertible = build without the flag).
- G-E3 GO means the epic proceeds to the higher-payoff Tier-2 (parallel
  reductions) with the re-baseline discipline pre-agreed (design D4 + legacy §P9
  "new numerical reference 单独立项").

**Negative / risks:**
- Config E is opt-in only; production adoption at scale still needs a release-tag
  promotion (deferred follow-up) and per-deployment build-leg selection.
- Tier-2 (PR-N3) carries a one-time re-baseline + tightened-A5 certification cost
  (`a5_thresholds.p12_tier2.yaml`); if G-E4 fails, revert to Config E
  (`SHUD_NVEC_DETRED` stays 0) per the amendment policy below.
- keliya small-NY (NY≈1.5k) Config E is not a perf win (OpenMP fork/join per op
  dominates); Config E is targeted at NY ≥ 100k (design R2). Not a regression
  risk since the flag is off by default and opt-in.

---

## Acceptance criteria

- [x] heihe_x4 90-day matrix: 6 cells (C/E × N∈{1,8,16}) × 3-run median wall +
  nst/nfe/ncfn/ncfl/netf per cell.
- [x] Equal-N bitwise cross-check PASS (identical counters + equal rivqdown SHA)
  at N∈{1,8,16}; cross-N determinism recorded as bonus.
- [x] Config E N=16 gate-on profile leg (`SHUD_NVEC_PROF=1` +
  `SHUD_ENABLE_PROFILE=1`) filed as the G-E3(iii) t_red cross-check.
- [x] G-E2 verdict computed by the pinned rule (bar 1.10× NOT moved); single-word
  verdict = TIER1_ADOPT; keliya wall informational-only.
- [x] G-E3 decision: all three inputs numeric with pinned cells + verbatim
  formula + either/OR aggregation → single verdict = TIER2_GO.
- [x] Consequence executed: issue #445 blocked-by marker lifted with gate values.
- [x] `SHUD_RHS_THREADS` unset in every cell + recorded; 90-day truncation kept;
  三铁律 (sbatch/output/scripts on /scratch, exclusive nodes) honored.
- [x] SHUD submodule pointer unchanged (pure outer PR; Config E code final from
  PR-N1).

---

## Forward action

1. **PR-N3 (Tier-2, now UNBLOCKED)**: implement fixed-tree deterministic
   reductions (`SHUD_NVEC_DETRED`, Config E2) + tightened A5 override config +
   G-E4 acceptance (cross-thread bitwise → A4 ulp report → full A5) + re-baseline
   decision doc. Gated order: determinism first, accuracy second.
2. **Config E release-tag promotion** (post-epic follow-up, NOT this change):
   promote Config E from opt-in build leg to a tagged release once field
   validation on the target large-NY deployments accrues.
3. **Capstone**: merge `research/p12-nvec` → `main` + openspec archive + SHUD
   `p12-nvec` → `openmp-baseline` merge-back per this adoption verdict.

---

## Amendment policy

This ADR records the PR-N2 Tier-1 adoption + Tier-2 GO decision. If PR-N3's G-E4
acceptance FAILS (cross-thread bitwise or full A5), Tier-2 is reverted to Config
E (`SHUD_NVEC_DETRED` stays 0) and this ADR is amended with the G-E4 failure
evidence + revert note (append-only; the §Status + §Decision Tier-1 adoption
stays byte-identical, since Tier-1 stands on its own G-E2 evidence). Any change
to the pinned bars (G-E2 1.10×, G-E3 1.15×, G-E3(ii) 10%) requires a new ADR, not
an amendment here.

```bash
# Audit that the §Decision Tier-1 clause is unchanged since acceptance:
git show HEAD:docs/adr/0011-p12-nvec-tier1-verdict-and-tier2-gate.md \
  | sed -n '/^## Decision/,/^## Outcome Table/p'
```

---

## References

### Internal ADRs (本仓库)
- `docs/adr/0010-cpu-acceleration-status-and-p10-decision.md` — CPU acceleration
  status; forward space + NOT-doing list this epic acts within.
- `docs/adr/0009-p9-cvode-outer-policy-closure.md` — P9 1.5× ROI bar precedent
  (contrasted by the G-E2 1.10× trajectory-identical bar).
- `docs/adr/0008-cpu-acceleration-closure-and-forward-plan.md` — Config D
  refutation (thread-count reduction drift) that Tier-1's serial-reduction
  override designs around.
- `docs/adr/0002-solver-path-selection.md` — P1e StrictOMP RHS Path 1 baseline.

### Epic docs (本仓库)
- `docs/p12-nvec/tier1_verdict.md` — measured G-E2/G-E3 verdict doc.
- `docs/p12-nvec/spike_brief.md` — epic goal + Kill/ROI gate prose.
- `openspec/changes/p12-nvec/design.md` — §D2 (Config E mechanism) + §D3 (G-E2 +
  1.10× bar) + §D4 (G-E3 三输入 + formula).
- `openspec/changes/p12-nvec/specs/tier1-scaling-verdict/spec.md` — pinned
  thresholds + scenarios.
- `openspec/glossary.md` §P12-nvec 集合 — term definitions.

### Master plan sections
- `SHUD_openMP_master_plan.md` §P12-nvec — single anchor line (this ADR).
- §P8-NVector (原 P8a) + legacy §P9 deterministic reduction — the anchors this
  epic executes.

### PR sequence
- PR-N0 #447 — `SHUD_NVEC_PROF` profiler + share tables (G-E3 inputs).
- PR-N1 #448 — Config E hybrid NVector + G-E1 bitwise PASS.
- PR-N2 #444 — this: server matrix + G-E2/G-E3 verdicts + this ADR + docs sync.
- PR-N3 #445 — Tier-2 fixed-tree reductions (UNBLOCKED by G-E3 GO).

### Evidence directories (`.review-evidence/`)
- `.review-evidence/p12-nvec/pr-n0/` — profiler + shares + t_red.
- `.review-evidence/p12-nvec/pr-n1/` — Config E + G-E1.
- `.review-evidence/p12-nvec/pr-n2/` — server scaling matrix + profile leg.

### External dependencies
- SUNDIALS/CVODE 6.0.0 — OpenMP NVector backend (`nvector_openmp.c`) + generic
  ops ABI (`sundials_nvector.h`) that Config E's overrides target.
