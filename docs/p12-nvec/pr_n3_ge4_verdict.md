# P12-nvec PR-N3 — Config E2 fixed-tree deterministic reductions: G-E4 verdict

```yaml
epic: P12-nvec
doc: pr_n3_ge4_verdict
date: 2026-07-02
pr: PR-N3 (#445)
gate: G-E4 (pinned order: cross-thread bitwise -> A4 ulp report -> full A5 tightened)
shud_commit: f78e031            # p12-nvec (G-E4 evidence at ce4bcef; Note-1 abort fix + neutrality re-verify)
config: Config E2 (SHUD_NVEC_DETRED=1; block B=4096; Neumaier=0/plain)
verdict: G-E4 PASS -> Config E2 CERTIFIED (re-baseline recorded; see pr_n3_rebaseline_decision.md)
```

## Scope

Records the G-E4 acceptance chain for Config E2 (fixed-tree deterministic
reductions, `SHUD_NVEC_DETRED`), executed in the spec-pinned order
(determinism BEFORE accuracy — legacy §P9 discipline: the summation-order
shift must never be conflated with a parallel bug). All thresholds are pinned
in `openspec/changes/p12-nvec/specs/tier2-det-reduction/spec.md` and
`design.md §D4`; nothing decided ad hoc here. Raw evidence:
`.review-evidence/p12-nvec/pr-n3/`.

## E2 design summary (what was built)

- **Block size B = 4096** (production, `SHUD_NVEC_DETRED_B` default), a fixed
  compile-time constant INDEPENDENT of thread count. Rationale: design §D4
  names 4096 as the reference; it degenerates keliya (NY=1986 -> 1 block) to a
  plain fold — which is exactly why the forced-small-B=256 leg exists — while
  giving heihe_x4 (NY≈120k) ≈30 blocks / ~5 combine levels of real tree.
- **In-block fold**: serial in index order, carrying the existing
  `SHUD_NVEC_NOOPT` codegen pin (clang `optnone` / gcc `O0,no-tree-vectorize`)
  so the vectorizer cannot interleave the per-block accumulation (spec letter:
  "serial in index order"). Reuses PR-N1's proven mechanism.
- **Combine order**: fixed bottom-up binary tree (`det_tree_combine`), a pure
  function of the block COUNT (hence of NY and B). Partials land in a per-block
  array indexed by block id; the `omp for` thread→block mapping is dynamic but
  writes disjoint slots, so it cannot affect any value or the combine order.
- **Which slots**: only the SUMMATION reductions (dotprod / wrmsnorm[mask] /
  wl2norm / l1norm + aliased `*local` + wsqrsum[mask]local + dotprodmultilocal)
  get the tree; non-summation reductions (min / maxnorm / invtest / constrmask
  / minquotient) keep the Tier-1 serial bodies — no combine order to fix, and
  already cross-thread deterministic.
- **Neumaier**: implemented plain first (design §D4). DECISION: **NOT enabled**
  (`SHUD_NVEC_DETRED_NEUMAIER` stays 0). Basis is the measured A4 evidence
  (§G-E4(2) below) — the compensated path is compiled-in and one flag away.

## G-E4(1) cross-thread bitwise — PASS (BEFORE A4/A5)

### keliya gate (Mac clang, production-B four legs + forced-small-B multi-level)

`.review-evidence/p12-nvec/pr-n3/keliya_det/SUMMARY.txt`. keliya
NY = 3·NumEle + NumRiv + NumLake = 3·484 + 534 + 0 = **1986**.

| leg | N | blocks (B) | combine levels | manifest sha256[:32] | note |
|---|---|---|---|---|---|
| C_n8 (Config C ref) | 8 | — (serial) | — | `801c2f798aae163f154c464bc2b8730c` | serial NVector |
| E_n8 (Config E, DETRED off) | 8 | 1 (=C) | 0 | `801c2f798aae163f154c464bc2b8730c` | **== C: "E2 off leaves Config E intact"** |
| E2 prod N=1 | 1 | 1 (4096) | 0 | `801c2f798aae163f154c464bc2b8730c` | degenerate single block |
| E2 prod N=2 | 2 | 1 (4096) | 0 | `801c2f798aae163f154c464bc2b8730c` | |
| E2 prod N=4 | 4 | 1 (4096) | 0 | `801c2f798aae163f154c464bc2b8730c` | |
| E2 prod N=8 | 8 | 1 (4096) | 0 | `801c2f798aae163f154c464bc2b8730c` | |
| **E2 b256 N=1** | 1 | 8 (256) | 3 | `8d742592dde01cda66ee51976f870ff5` | multi-level tree |
| **E2 b256 N=8** | 8 | 8 (256) | 3 | `8d742592dde01cda66ee51976f870ff5` | **== b256 N=1 (cross-N)** |

- Production-B four legs (N∈{1,2,4,8}): **all SHA-identical** (`801c2f79…`). ✓
- Forced-small-B=256 legs (N∈{1,8}, 8 blocks / 3 combine levels): **SHA-identical
  within equal B** (`8d742592…`). ✓ The tree combine IS exercised and is
  cross-thread-deterministic.
- Compared within equal B: b256 (`8d742592…`) ≠ production (`801c2f79…`) — EXPECTED
  (B changes the summation tree; N must not). The production-B legs degenerate to
  a single block == the plain fold == Config C, which is why the b256 leg is the
  one that actually stresses the multi-level combine.
- **"E2 off leaves Config E intact"**: E_n8 == C_n8 == `801c2f79…`. Config E built
  at the SAME SHUD commit (ce4bcef) without `SHUD_NVEC_DETRED` reproduces Config C
  bitwise → revert to Config E is a build-flag flip, not a git revert. (Config E
  == Config C is the PR-N1 G-E1 result; PR-N1's canonical Config C keliya baseline
  manifest is `14db6b3c…` in a different sort/scope — the equality that matters
  here is the fresh ce4bcef C_n8 == E_n8 == E2(prod), all `801c2f79…`.)

### heihe_x4 production-scale cross-N (server gcc-13, NY≈120k ≈30 blocks)

From the E2 wall-matrix cells (§4.3 runs), all 3 reps each:

| cell | N | nst | nfe | ncfn | ncfl | netf | rivqdown sha256[:24] |
|---|---|---|---|---|---|---|---|
| E2_n8  | 8  | 6574 | 6728 | 50 | 3688 | 0 | `f70209c477cb80db09381cec` |
| E2_n16 | 16 | 6574 | 6728 | 50 | 3688 | 0 | `f70209c477cb80db09381cec` |

**Identical cvode_stats counters AND equal rivqdown SHA across N=8/16** (and across
all 3 reps within each cell). ✓ The production multi-level tree + dynamic
thread→block mapping is deterministic at scale, at zero extra cost from the
wall-measurement runs.

**G-E4(1) PASS** on both the keliya gate and the heihe_x4 cross-N check.

## G-E4(2) A4 max_ulp report — DOCUMENTED (no threshold)

`.review-evidence/p12-nvec/pr-n3/a4_ulp/A4_REPORT.txt` (tool:
`.../tools/a4_ulp.py`, uv-run, reuses the tools/a5 SHUD reader).

- **keliya production-B**: E2 vs E `rivqdown` max_ulp=**0**, max_abs_diff=**0**
  (single-block degenerate == plain fold; the tree is not exercised here).
- **heihe_x4 production-B** (E2 vs E, per output):
  - `rivqdown` : max_abs_diff = **4.203e+04** (identical 957/383130)
  - `eleveta`  : max_abs_diff = **1.036e-05**
  - `eleygw`   : max_abs_diff = **6.405e-04**
  - `elevprcp` : max_abs_diff = **0** (forcing input, solver-independent)
  - (raw `max_ulp` magnitudes are dominated by ref==0 vs candidate-tiny elements,
    where ULP distance spans most of uint64 range — not a faithful smallness
    measure with zeros present; `max_abs_diff` is the physical companion.)

**Interpretation (load-bearing for Neumaier):** the E2-vs-E difference is NOT an
in-block rounding effect. The fixed-tree combine changes the WRMS error-norm
summation ORDER, which perturbs CVODE's adaptive step decisions (nst 6575→6574,
ncfl 3620→3688; both deterministic across threads per G-E4(1)) → a
DIFFERENT-but-valid integration path. This is the anticipated one-time
summation-order shift (design §D4).

**Neumaier decision: NOT enabled**, basis = this A4 evidence. Neumaier
compensates summation accuracy WITHIN a fixed order; it cannot make the tree
order and the serial-fold order agree, so it would not shrink this delta. And
the tightened A5 already PASSES at nse=1.0/kge=0.9999 with the plain tree — no
accuracy deficit to compensate. Compensated path stays compile-time-available
(`SHUD_NVEC_DETRED_NEUMAIER=1`).

## G-E4(3) FULL A5 — PASS (tightened p12_tier2.yaml vs Config C reference)

`tools/a5` on heihe_x4, candidate = Config E2, reference = Config C, config =
`tools/a5/config/a5_thresholds.p12_tier2.yaml` (PR-N3 deliverable).
`.review-evidence/p12-nvec/pr-n3/a5report/E2_vs_C/`.

| metric | value | tightened threshold | pass |
|---|---|---|---|
| nse | **1.0000** | ≥ 0.99 | ✓ |
| kge | **0.9999** | ≥ 0.99 | ✓ |
| peak_magnitude_ratio | 1.0002 | [0.90, 1.10] | ✓ |
| peak_timing_offset | **0 steps** | ≤ 1 | ✓ |
| runoff_volume_ratio | **0.9999** | [0.99, 1.01] | ✓ |
| monthly_bias_mae | 0.0001 | ≤ 0.05 | ✓ |
| water_balance_residual | NaN | ≤ 0.05 | informational (see below) |
| **weighted_score** | **1.0000** | ≥ 0.85 | **PASS** |

Despite the raw-discharge max_abs_diff at flood peaks (A4), the INTEGRATED
hydrology is near-perfect: nse=1.0000 means the E2 discharge timeseries is
indistinguishable from Config C at the acceptance cadence.

**Control (sanity):** E-vs-C A5 with the same tightened config → PASS, nse=1.0000,
kge=1.0000 (Config E == Config C bitwise → trivially perfect). This confirms the
tightened config is correctly calibrated (a bitwise-identical config scores
perfect; the summation-shifted E2 also clears the 0.99 bars).

### water_balance_residual — candidate-vs-reference (spec requirement)

tools/a5 computes WB per-run with no cross-run "non-degrading" semantic, so the
spec requires the residuals recorded side-by-side with candidate ≤ reference.

| run | water_balance_residual | status |
|---|---|---|
| reference (Config C) | NaN | `unavailable_no_mesh_metadata` (PR-Y2 informational downgrade) |
| candidate (Config E2) | NaN | `unavailable_no_mesh_metadata` (PR-Y2 informational downgrade) |

Both are NaN under the identical mesh-metadata-unavailable state (heihe_x4 output
tree lacks the area/porosity metadata A5's area-weighted closure needs — the same
condition recorded in the P9 heihe_x4 A5 spot-check, PR-Z1). The candidate ≤
reference requirement is therefore **unenforceable on heihe_x4**: both sides are
NaN per the PR-Y2 informational downgrade (`unavailable_no_mesh_metadata`,
ADR-0010:95 precedent), so neither a numeric comparison nor a degradation check is
defined. The **six streamflow metrics are the operational A5 gate** here
(nse/kge/peak_magnitude/peak_timing/runoff_volume/monthly_bias — all PASS at the
tightened bars), and the WB residual contributes nothing to the weighted score.
No degradation introduced by E2.

## Post-review delta (SHUD ce4bcef → f78e031, review Note-1 + Note-2)

The G-E4 evidence above was generated at SHUD `ce4bcef`. Two non-blocking review
items were closed at `f78e031`, provably neutral on the shipped success path:

- **Note-1 (robustness):** the four fixed-tree reductions now allocate the
  block-partial array via `det_alloc_partials()`, which ABORTS LOUDLY
  (`std::abort` after an stderr message) on malloc failure. Binding design
  decision: NO silent serial fallback — a fallback would compute a different
  summation order and silently break the determinism contract. `nb` is bounded
  (heihe_x4 nb≈31 → ~248 bytes), so a failure is effectively OOM. This adds a
  branch on the malloc-FAILURE path only; the success path is byte-unchanged.
- **Note-2 (coverage):** ASan+UBSan CLEAN (0 findings) on the E2 B=256
  multi-block malloc path (keliya N=8; `.../asan_e2_b256/ASAN_VERDICT.txt`).
- **Behavior-neutrality re-verified at f78e031:** keliya E2 prod-B N=8 →
  `801c2f79…` (== ce4bcef) and E2 B=256 N=8 → `8d742592…` (== ce4bcef)
  (`.../keliya_det/sanity_E2*.manifest.sha`); CI `build-and-compare (keliya)`
  re-verifies the default-build bitwise gate. → the ce4bcef G-E4 evidence
  remains valid for the shipped `f78e031` code.

## Verdict

**G-E4 PASS** in the pinned order: cross-thread bitwise (keliya
production-B + forced-small-B + heihe_x4 cross-N) → A4 ulp report (documented,
Neumaier=plain justified) → full A5 tightened (nse=1.0/kge=0.9999, PASS).
Config E2 is CERTIFIED; the epic records an explicit re-baseline decision
(`pr_n3_rebaseline_decision.md`). Neither G-E4(1) nor G-E4(3) failed, so the
revert-to-Config-E / ADR-0011-amendment branch does NOT apply.

## References

- Spec: `openspec/changes/p12-nvec/specs/tier2-det-reduction/spec.md`
- Design: `design.md §D4`
- PR-N2 gate (TIER2_GO): `docs/p12-nvec/tier1_verdict.md` (G-E3 = 1.3744× projection)
- Evidence: `.review-evidence/p12-nvec/pr-n3/` (keliya_det / server_matrix / a4_ulp / a5report)
- Re-baseline: `docs/p12-nvec/pr_n3_rebaseline_decision.md`
- SHUD commit: `f78e031` (p12-nvec; G-E4 evidence generated at `ce4bcef`, neutrality re-verified — see evidence README addendum)
