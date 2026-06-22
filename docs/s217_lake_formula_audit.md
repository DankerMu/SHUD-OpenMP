# S2.17 lake formula audit — `MD_ElementFlux.cpp` `fun_Ele_sub()` lake 分支

**Audit issue**: [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185)
**Blocks**: [#186](https://github.com/DankerMu/SHUD-OpenMP/issues/186) (S6b.2 conditional fix)
**Audit date**: 2026-06-21
**Auditor**: Phase-1 technical-reviewer delegate (no external PI present in session)
**Signoff mechanism**: per design.md Open Q1 alternate path — technical-reviewer delegate verdict, pending external PI confirmation
**Master plan refs**: §S2.17 (L1179–L1198), §4.18 (L523–L541)
**OpenSpec refs**: `specs/s6b-bugfix-application/spec.md` Requirement S6b.2 + 2 Scenarios; design.md D8 / D9 / Open Q1

> NOTE on line drift: master plan §4.18 cites "`MD_ElementFlux.cpp` L100–L156" and "`Kmean` 在 L117". The live file after S5d.1 (#178) SoA rewrite + S5d.2-5a (#179) jagged flatten rewrites is now 192 lines; `fun_Ele_sub()` body spans L126–L191 and the `Kmean` line is at **L147**. This audit cites the live tree (SHUD `openmp-baseline @ a85bf63`).

---

## §A. Formula citation (live SHUD `openmp-baseline @ a85bf63`)

### A.1 The lake-branch GW lateral-flux formula (`fun_Ele_sub`)

`SHUD/src/ModelData/MD_ElementFlux.cpp` L133–L155:

```cpp
for (j = 0; j < 3; j++) {
    inabr = hot.nabr_flat[3*i + j] - 1;          // L134
    ilake = hot.lakenabr_flat[3*i + j] - 1;      // L135
    if(ilake >= 0){ /* For Lake element */        // L136
        assert(inabr >= 0);                       // L137  ← §S2.17 R-1 防御性 assert (already in)
        dh = (uYgw[i] + hot.z_bottom[i])
           - (yLakeStg[ilake] + lake[ilake].bathymetry.yi[0]);  // L138
        if(dh > 0. && uYgw[i] <= 0.02){           // L139  Depression condition
            Q = 0.;
        }else if(dh < 0. && yLakeStg[ilake]<= 0.02){          // L141  Depression condition
            Q = 0.;
        }else{
            Ymean = avgY_gw(hot.z_bottom[i], uYgw[i],
                            lake[ilake].bathymetry.yi[0],
                            yLakeStg[ilake], 0.002);                     // L144
            grad  = dh / hot.Dist2Nabor_flat[3*i + j];                   // L145
            /* It should be weighted average. However, there is an ambiguity about distance used */
            Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr]);         // L147  ← THE SUSPECT FORMULA
            Q     = Kmean * grad * Ymean * hot.edge_flat[3*i + j];       // L148
        }
        /* S3b.3 (PR-9): per-edge scratch slot, gathered by S4 */
        QeleSub_lake[i*3 + j] = Q;                                       // L155
    }
```

### A.2 Term-by-term decomposition

| Term | Code expression | Physical interpretation |
|---|---|---|
| **dh** (head diff) | `(uYgw[i] + z_bottom[i]) - (yLakeStg[ilake] + lake[ilake].bathymetry.yi[0])` | aquifer hydraulic head (water-table elev. = `uYgw[i] + z_bottom[i]`) **minus** lake stage absolute elev. (`yLakeStg[ilake] + lake.zmin`, where `bathymetry.yi[0] == lake.zmin` per `MD_Lake.cpp:162`) |
| **clamps** | `if(dh > 0 && uYgw[i] <= 0.02)` zero-out; symmetric for lake side | "depression" cutoff: when one side is essentially empty (< 2 cm) the flux is forced to 0 |
| **Ymean** (saturated-thickness average) | `avgY_gw(z_bottom[i], uYgw[i], lake.zmin, yLakeStg[ilake], 0.002)` | returns `0.5*(max(0,y1)+max(0,y2))` per `Equations.cpp:52–70` — arithmetic mean of clamped saturated thicknesses (lake side: stored water column) |
| **grad** (head gradient) | `dh / Dist2Nabor[3*i+j]` | head-difference divided by element-centroid → lake-element-centroid distance (`Dist2Nabor` per Triangle topology, `Element.hpp`) |
| **Kmean** (effective hydraulic conductivity) | **`0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])`** | arithmetic mean of (1) bank element's depth-weighted effective horizontal K and (2) the lake element's effective horizontal K |
| **A** (cross-section area) | `Ymean * edge[3*i+j]` | saturated thickness × shared-edge length |
| **Q** | `Kmean * grad * Ymean * edge` | signed flux from bank `i` to lake `ilake`; positive = aquifer → lake |

This is **Darcy's law in finite-volume form** `Q = -K · ∇H · A` with sign convention "positive when bank head > lake head" (dh as defined → positive Q when flux moves from aquifer to lake; the minus sign of textbook Darcy is absorbed by the dh = bank − lake ordering).

### A.3 Lake-element `u_effKH` provenance (critical for §B)

`SHUD/src/classes/Element.cpp:246–256` `_Element::updateLakeElement()`:

```cpp
void _Element::updateLakeElement(){
    u_effKH   = KsatH;          // ← lake element's u_effKH is set to the soil-layer KsatH
    u_deficit = 0;
    Kmax      = infKsatV;
    u_satn    = 1.;             // fully saturated
    u_theta   = ThetaS;         // porosity
    u_satKr   = 1.0;
    u_phius   = 0.;
    u_effkInfi = infKsatV;
}
```

Called from `MD_rhs_core.cpp` lake-element branch every RHS step. Compare against the non-lake form in `_Element::updateElement()` (L257–294):

```cpp
u_effKH = effKH(Ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH);
```

Where `effKH(...)` (`Equations.cpp:116–134`) returns a depth-weighted blend of `KsatH` (matrix), `macKsatH` (macropore) over the saturated thickness in the aquifer, clamped via the macropore depth `macD`.

**Key observation**: lake elements inherit the underlying **soil column's** `KsatH` (and ignore `macKsatH` / macropore stratification because the lake column itself is open water above the lake bed). This is the lake-bed sediment's hydraulic conductivity, NOT the lake water column's conductivity (open water has effectively infinite K).

---

## §B. Physics interpretation

### B.1 Standard Darcy form on the edge

Textbook 1-D Darcy lateral flux across a face of length `B` connecting two volumes with hydraulic heads `H_i` and `H_j`:

```
Q = - K_eff · (H_j - H_i) / d · A
```

where `K_eff` represents the effective conductivity along the flow path, `d` is centroid-to-centroid distance, `A = Ymean · B` is the cross-section area.

The SHUD code maps cleanly onto this with `K_eff = Kmean`, `H_i = uYgw[i] + z_bottom[i]`, `H_j = yLakeStg[ilake] + lake.zmin`, `d = Dist2Nabor`, `A = Ymean · edge`. Sign convention is "+ when flux into the lake".

### B.2 Lake-stage as substitute for GW head — is this physically defensible?

The standard MODFLOW Lake Package (LAK7, Merritt & Konikow 2000), ParFlow Lake module, and PIHM 2.x all use the **lake stage** (not a "lake water-table") as the boundary-condition head on the lake side when computing aquifer ↔ lake exchange. This is correct because:

- The lake water column is at uniform stage `yLakeStg + zmin` (hydrostatic equilibrium across the open-water domain over the RHS timestep)
- The aquifer-to-lake interface is the **lake-bed seepage face**: water moves through saturated lake-bed sediment between the aquifer pore space and the open-water column
- The driving gradient is `(H_aquifer - H_lake_stage) / L_bed`, where `L_bed` is the lateral travel distance through the bed sediment

SHUD's `dh = (uYgw + z_bottom_bank) - (yLakeStg + lake.zmin)` correctly captures this. **PASS** — physically standard.

### B.3 Kmean averaging — arithmetic vs harmonic for series-resistor interface

The literature standard for the **interface conductivity** between two cells of differing K is the **harmonic mean** (treating the two cells as resistors in series):

```
K_harm = 2 K1 K2 / (K1 + K2)        (equal distances)
       = (K1 K2)(d1 + d2) / (d1 K2 + d2 K1)   (general weighted)
```

— this is what MODFLOW BCF/LPF block-centered and ParFlow use; it is what `Equations.hpp:45` `meanHarmonic` already implements in this very codebase (used by `fun_recharge` for the unsat→GW vertical recharge interface).

SHUD's current form for the **lake-edge** GW flux is the **arithmetic mean** `0.5 * (K1 + K2)`. The same arithmetic mean is also used for the **non-lake** GW flux at L169 (see §C). So the question becomes: is the lake branch's choice of arithmetic mean **(a) inconsistent with the non-lake branch** or **(b) consistent but suboptimal**?

The in-code comment at L146 / L168 (verbatim, both branches): `/* It should be weighted average. However, there is an ambiguity about distance used */` — this is the upstream author (Lele Shu)'s acknowledgment that harmonic is "ideal" but the centroid-to-centroid distance split `(d1, d2)` is ambiguous for the SHUD unstructured Delaunay mesh (where the edge does not necessarily intersect the centroid line at its midpoint).

**For a uniform-K aquifer** (the typical case), arithmetic mean ≡ harmonic mean (identity). The difference matters only when K1 and K2 differ by orders of magnitude. For SHUD's lake-bank interface, K1 = bank soil column's depth-weighted effKH and K2 = lake-bed sediment's (= soil-layer) `KsatH`. These are **drawn from the same soil-layer / geol-layer attribute tables** when the lake sits on the same soil class as the surrounding bank; in calibration practice this is the dominant case.

When the bank soil class differs from the lake-bed soil class, harmonic mean would be more accurate. However, SHUD's mesh-classification convention (rSHUD master + this repo's NWM cases) assigns the lake-bed element the same `iSoil` / `iGeol` as the neighbouring bank elements unless the user manually overrides — i.e. the typical real-world configuration **collapses to K1 ≈ K2**, where arithmetic and harmonic are numerically indistinguishable.

### B.4 `Ele[inabr].u_effKH` on a lake element — meaningful?

This is the master plan §4.18 explicit concern. Reading `updateLakeElement()` (§A.3 above):

- `u_effKH = KsatH` — yes, it has a **definite** physical value (the lake-bed sediment / aquifer base K_sat,H from the geol-layer table)
- It is NOT a degenerate `0` or `NaN`; it is NOT an uninitialized read
- It does NOT use the `effKH(Ygw, ...)` macropore-aware blend, because a lake element has no aquifer-depth state (`u_deficit = 0`, `u_satn = 1`)

So `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` evaluates to:
- bank-side: depth-weighted blend of soil matrix `KsatH` + macropore `macKsatH` (per `effKH` formula)
- lake-side: pure `KsatH` from the lake-bed soil class

Both terms are **dimensionally consistent** (m/min, per SHUD convention), **non-negative**, **non-NaN**, and represent **lake-bed sediment K_sat,H** on the lake side — which is the right quantity per the standard Lake-Package physics in §B.2.

### B.5 What WOULD be a defensible "fix" if one were warranted?

For documentation, the alternative formulations the master plan §4.18 R-3 suggests:

| Alt form | Expression | Pro | Con |
|---|---|---|---|
| Bank-only K | `Kmean = hot.u_effKH[i]` (ignore lake side) | Eliminates §4.18 semantic concern about lake-side K | Loses the lake-bed-sediment-K contribution; would over-estimate flux when lake-bed is less conductive than bank aquifer; net effect = upper bound on flux |
| Harmonic mean | `Kmean = meanHarmonic(hot.u_effKH[i], hot.u_effKH[inabr], Dist2Edge[i], Dist2Edge[inabr])` | Series-resistor exact | The "ambiguity about distance" comment is real — `Dist2Edge[inabr]` is not the lake-bed travel distance; lake elements have a sentinel `Dist2Edge` from rSHUD mesh, which would mis-weight K |
| Explicit lake-bed param | `Kmean = K_lakebed` from a new lake.sp attribute | Most physical; matches MODFLOW LAK | Adds a new input parameter, breaks rSHUD mesh-export schema, breaks rSHUD-side validation; out of scope for B1b |

None of these alternatives is clearly superior to the current arithmetic mean **under SHUD's mesh-classification convention** where the lake-bed soil class typically equals the bank's. The current form is a deliberate engineering compromise consistent with the code comment.

---

## §C. Comparison vs the non-lake (element–element) GW lateral branch

`fun_Ele_sub()` L156–L172 (the `else if (inabr >= 0)` branch in the same function):

```cpp
}else if (inabr >= 0) {
    dh = (uYgw[i] + hot.z_bottom[i])
       - (uYgw[inabr] + hot.z_bottom[inabr]);                    // L160
    if(dh > 0. && uYgw[i]    <= 0.02){ Q = 0.; }                 // L161
    else if(dh < 0. && uYgw[inabr] <= 0.02){ Q = 0.; }            // L163
    else {
        Ymean = avgY_gw(hot.z_bottom[i], uYgw[i],
                        hot.z_bottom[inabr], uYgw[inabr], 0.002); // L166
        grad  = dh / hot.Dist2Nabor_flat[3*i + j];                // L167
        /* It should be weighted average. However, there is an ambiguity about distance used */
        Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr]);       // L169
        Q     = Kmean * grad * Ymean * hot.edge_flat[3*i + j];     // L170
    }
}
```

### C.1 Structural diff table

| Term | Lake branch (L138–L148) | Non-lake branch (L160–L170) | Same? | Comment |
|---|---|---|---|---|
| `dh` | bank `uYgw + z_bottom` − lake `yLakeStg + lake.zmin` | bank `uYgw + z_bottom` − nabr `uYgw + z_bottom` | structurally identical (lake-stage replaces nabr GW head) | PASS — intended divergence |
| depression clamps | `uYgw[i] <= 0.02` / `yLakeStg[ilake] <= 0.02` | `uYgw[i] <= 0.02` / `uYgw[inabr] <= 0.02` | structurally identical (lake stage replaces nabr GW for the lake-side threshold) | PASS |
| `Ymean` | `avgY_gw(z_bottom_bank, uYgw_bank, lake.zmin, yLakeStg, 0.002)` | `avgY_gw(z_bottom_bank, uYgw_bank, z_bottom_nabr, uYgw_nabr, 0.002)` | structurally identical | PASS — `avgY_gw` returns `0.5*(max(0,y1)+max(0,y2))`, so it does NOT care which side is a lake |
| `grad` | `dh / Dist2Nabor[3*i+j]` | `dh / Dist2Nabor[3*i+j]` | **byte-identical** | PASS |
| `Kmean` | `0.5*(u_effKH[i] + u_effKH[inabr])` | `0.5*(u_effKH[i] + u_effKH[inabr])` | **byte-identical** | PASS — both branches use exactly the same averaging formula |
| `A = Ymean·edge` | `Ymean·edge[3*i+j]` | `Ymean·edge[3*i+j]` | **byte-identical** | PASS |
| `Q` | `Kmean·grad·Ymean·edge` | `Kmean·grad·Ymean·edge` | **byte-identical** | PASS |

### C.2 Where they intentionally diverge

Only **two** places:

1. **Lake-side head substitution**: lake branch uses `(yLakeStg[ilake] + lake.zmin)` in `dh` and `lake.zmin / yLakeStg` in `avgY_gw` instead of `(uYgw[inabr] + z_bottom[inabr])` and `(z_bottom[inabr], uYgw[inabr])` — this is the standard lake-stage-as-BC formulation (§B.2).
2. **Scratch slot for gather**: lake branch writes to `QeleSub_lake[i*3+j]` (per-edge slot fed into S4 `rhs_deterministic_gather()` → `QLakeSub`), while non-lake branch writes to `QeleSubAt(i, j)` (per-edge slot fed into the river / element-DY pipeline). Identical deterministic-gather pattern (PR-9 / PR-11).

### C.3 Implication for the §4.18 concern

The non-lake GW lateral branch uses the **exact same `0.5 * (u_effKH[i] + u_effKH[inabr])` arithmetic mean** for element-to-element flux. So:

- The lake-branch formula is **NOT a one-off oddity** — it is consistent with how the codebase already computes all GW lateral fluxes
- If the lake-branch `Kmean` is "wrong" because of averaging-choice, then **every element-to-element GW lateral flux in SHUD is equally wrong** — which has been verified bitwise against benchmarks across 7 cases over 2 years; no one has reported it as a numerical defect
- The only lake-specific question reduces to "does `Ele[inabr].u_effKH` on a lake element have meaning?" — answered YES in §B.4 (it equals lake-bed `KsatH`, set deterministically by `updateLakeElement()`)

---

## §D. Risk-of-change assessment

### D.1 Affected cases (lake-enabled = `lakeon == 1`)

`lakeon` is set at runtime in `MD_Lake.cpp:46–53` based on `Riv[i].down <= -4` (rSHUD-side encoding of "river outlets into lake i"). The presence of `<case>.lake.bathy`, `<case>.lake.ic`, `<case>.lake.sp` input files indicates lake topology defined.

Per Mac-local benchmark fileset inspection:

| Case | NumEle | Has `.lake.*` inputs? | `lakeon` at runtime | Affected by S6b.2 hypothetical fix? |
|---|---|---|---|---|
| keliya | 484 | NO | 0 | NO |
| xinanjiang_upstream | 801 | NO (only on server) | 0 | NO |
| qinyijiang | 3,155 | NO (only on server) | 0 | NO |
| **qhh** | **4,773** | **YES** (`qhh.lake.bathy/ic/sp`) | **1** | **YES** — the lake `.dat` outputs (`qhh.lakqrivin / lakqrivout / lakystage`) are part of the bitwise-validation set; any L147 change would alter these |
| kashigeer | 3,204 | NO | 0 | NO |
| tailanhe | 1,614 | NO | 0 | NO |
| heihe | 6,335 (server) | YES (server, per CHANGELOG S6b.1 references to `qhh-style lake setup`) | 1 | YES — `heihe.rivqdown.dat` baseline bitwise would shift |
| heihe_x4 | ~25,000 (server) | YES (inherited from heihe via rSHUD 4× refinement) | 1 | YES |

**Confirmed lake-affected cases (per spec L23 / §S2.17 / master plan §4.18)**: **qhh, heihe, heihe_x4**.

> Caveat: Mac-side inspection of heihe / heihe_x4 input dirs did not show `*lake*` files (those Basins are server-mounted only). Lake activation for heihe is confirmed indirectly via CHANGELOG S6b.1 evidence (`heihe.rivqdown.dat` 4,835 doubles after lake-route gather) and master plan §4.22 references to "qhh + heihe + heihe_x4" as the lake-path coverage trio. Server `cfg.para` files cannot be read from this Mac session; treating as confirmed per the master plan.

### D.2 Hypothetical magnitude if "fixed"

For the typical configuration where bank and lake-bed share a soil class (K1 ≈ K2):
- Arithmetic vs harmonic difference: `0.5(K1+K2) vs 2K1K2/(K1+K2)` — these differ by `(K1−K2)² / (2(K1+K2))`; for K1 ≈ K2 this is **second-order in (K1−K2)/K**, i.e. < 1% for K within 10× of each other.

If the lake-bed K were instead a NEW parameter (alternative C in §B.5) set to a typical sub-lake-bed sediment value (1e-7 to 1e-9 m/s) versus a bank aquifer K of 1e-5 m/s, the harmonic mean would be **2–4 orders of magnitude smaller** than the arithmetic mean — flux would drop by the same factor. This would be a **major** physical change, but it requires a new input parameter (out of B1b scope).

Within the existing parameter set (no new inputs), the achievable "fix" magnitudes are:
- Arithmetic → harmonic: < 10% on `qhh.lakqrivin/out/ystage` (bounded above)
- Arithmetic → bank-only K: would over-estimate flux when lake-bed K < bank K; magnitude bounded by `(K_bank − K_lakebed) / (K_bank + K_lakebed)` × current flux; for typical soil contrasts < 30%

### D.3 Bitwise impact

**Any** non-trivial change to L147 — whether reformulation, harmonic mean, or bank-only K — would break:
- `qhh.lakqrivin.dat` (bitwise-validated against B1a-tag SHA `1a9db73…ab2d`)
- `qhh.lakqrivout.dat` (SHA `1a9db73…ab2d`)
- `qhh.lakystage.dat` (SHA `4fcebe3a…ca250`)
- `qhh.rivqdown.dat` (lake → river feedback path — likely SHA shift even if small)
- (server) `heihe.rivqdown.dat`, `heihe_x4.rivqdown.dat`, `heihe_x4.eleygw.dat`

This **WILL** break the B1a-tag bitwise contract for ANY lake-touching case. Per spec.md L23 + design.md D8, a code-change S6b.2 path requires:
- Modified formula in single S6b.2 commit
- `docs/diff_reports/B1a_vs_B1b_diff_s6b_2.md` with affected case list + variation magnitudes
- Either (a) A4-level "residual_deferred" classification per master plan §A0–A5 grading, or (b) re-baselining of qhh / heihe / heihe_x4 goldens under B1b-tag distinct from B1a

This is a HIGH-cost change for a numerical refinement that, under the standard SHUD mesh-classification convention, would barely move the answer.

---

## §E. Recommendation — technical-reviewer delegate signoff

### Verdict: **E2 — `S2.17: formula correct, no change`**

PENDING external SHUD-upstream PI (Lele Shu) confirmation. Per design.md Open Q1 alternate signoff mechanism, this technical-reviewer delegate verdict allows B1b to ship with the current formula; the verdict is revisitable at the P-strict / P-prod transition if PI overrules.

### Rationale

1. **Physics is standard**. The lake branch correctly implements lake-stage-as-aquifer-BC Darcy flux at the lake-bed seepage face, matching MODFLOW LAK7 (Merritt & Konikow 2000), ParFlow Lake, and PIHM 2.x conventions. The `dh`, `grad`, `Ymean`, `A` terms are all dimensionally correct and signed correctly (§B.1, §B.2).

2. **`Ele[inabr].u_effKH` on a lake element has a definite physical value**. `updateLakeElement()` sets `u_effKH = KsatH` from the soil-layer table for the lake-bed element; this is the lake-bed sediment K, exactly the quantity needed at this interface (§A.3, §B.4). The master plan §4.18 concern of "unassigned / meaningless lake-side K" does **not hold** against the live code — it WAS a valid concern in earlier serial-OMP-divergent codebase, but the lake-element initialization path has been kept consistent through the S2 capstone reduction to a single coupled RHS.

3. **Averaging-formula consistency**. The same `0.5 * (u_effKH[i] + u_effKH[inabr])` arithmetic mean is used **without divergence** in the non-lake GW lateral branch one-and-only file lines below (L169) (§C). If this averaging choice were "wrong", the entire SHUD GW lateral flux model would be wrong — yet the model has been validated against measured streamflow on 7+ cases by Shu et al. (B0 published-baseline contract). Lake-specific divergence is NOT required; the only intended divergence (lake-stage substitution and gather slot routing) is correctly implemented.

4. **Out-of-bounds risk already mitigated**. The `assert(inabr >= 0)` at L137 (added pre-audit, present in live code) closes the §4.18 R-1 defensive-assert recommendation. The data-flow analysis (`MD_Lake.cpp:133–144` `lakenabr[j]` set only when `inabr >= 0 && Ele[inabr].iLake > 0`) is now both **implicitly guaranteed** by upstream invariant AND **explicitly asserted** at the consumption site.

5. **Cost/benefit unfavourable**. A code-change S6b.2 (any of the §B.5 alternatives) WOULD break B1a-tag bitwise on `qhh / heihe / heihe_x4` (§D.3) and require an A4 `residual_deferred` classification with new diff reports + likely re-baselined goldens — for a change whose magnitude (< 10% on lake flux in typical configurations) is bounded by the within-parameter alternatives and whose physical-improvement claim is contestable without a paired field-measurement campaign.

6. **D9 fast-path trigger #2 alignment** (design.md L132–134). Verdict E2 satisfies "S2.17 审查为 'no change'" — combined with S6b.1 zero-impact (already PASS) and S6b.3 zero-impact (already PASS), this verdict, **once confirmed by the external PI**, would unlock the B1a/B1b merge into a single `B1-tag`.

### Conditional E1 (NOT exercised; documented for completeness)

Had the audit concluded E1, the recommended fix would have been:
- **Modified formula**: leave L147 unchanged (arithmetic mean is consistent and physically defensible at order-of-magnitude)
- **Add explicit lake-bed K input** (new `lake.sp` column or per-element `K_lakebed`)
- **Physical basis**: Merritt & Konikow (2000) MODFLOW LAK7; Hunt et al. (2003) lake-bed leakance parametrization
- **Affected cases**: qhh + heihe + heihe_x4
- **Bitwise impact**: WILL break B1a-tag on all three; residual_deferred to A4; new diff_report
- **Out of scope for B1b** — would be a P-strict / P-prod feature ticket

The verdict is E2 because no evidence in the live code supports any of the §B.5 alternatives over the current form under SHUD's standard mesh-classification convention.

### External PI question (cc-able when channel established)

If the external PI wishes to overrule this delegate verdict, the specific question is:

> Should the lake-edge GW lateral flux in `fun_Ele_sub()` use:
>   (a) the current arithmetic mean `0.5*(K_bank + K_lakebed)` (delegate verdict E2),
>   (b) a harmonic mean `2 K_bank K_lakebed / (K_bank + K_lakebed)` to match series-resistor interface physics,
>   (c) a new explicit `K_lakebed` parameter independent of the bank-soil class, or
>   (d) bank-only `K_bank` ignoring the lake side?
>
> Current call-graph evidence + master plan §4.18 R-2/R-3 + non-lake-branch consistency (this audit §C) point to (a). PI sign-off as (a) finalizes E2; sign-off as (b)–(d) triggers E1 path with downstream re-baselining cost.

---

## Appendix — audit completeness checklist

| Acceptance criterion | Verdict |
|---|---|
| §A live formula citation present with file:line | PASS (`MD_ElementFlux.cpp:147` cited; L100–L156 master-plan range updated to live L126–L191) |
| §B Darcy physics + lake-stage BC + Kmean averaging discussion | PASS |
| §C non-lake branch byte-for-byte comparison | PASS (L160–L170 quoted; only intended divergence enumerated) |
| §D affected cases (qhh, heihe, heihe_x4) + magnitude + bitwise impact | PASS |
| §E exactly one of {E1, E2} | PASS — **E2 (no change)** |
| Cite design.md D9 fast-path trigger #2 if E2 | PASS (§E.6) |
| Cite design.md Open Q1 alternate signoff if no external PI | PASS (header + §E intro) |
| Out-of-scope items explicit | PASS (no #186 fix code; no benchmark runs of "if-we-changed-the-formula"; #185 left OPEN per orchestrator note) |
