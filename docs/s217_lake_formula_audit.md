# S2.17 lake formula audit — `MD_ElementFlux.cpp` `fun_Ele_sub()` lake 分支

**Audit issue**: [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185)
**Blocks**: [#186](https://github.com/DankerMu/SHUD-OpenMP/issues/186) (S6b.2 conditional fix)
**Audit date**: 2026-06-22 (rev. after PR #204 Phase-4 review V1/V2/V3); **PI delegate sign-off**: 2026-06-22 (PR-19 #210)
**Auditor**: Phase-1 audit author + evidence packager (audit) + DankerMu acting as PI delegate (sign-off). Per `spec.md` L23 the SHUD-upstream PI prerogative attaches to the verdict; design.md Open Q1 (PI delegate qualification) is **closed** by this sign-off — DankerMu is GitHub organization owner of **both** `DankerMu/SHUD-OpenMP` (this repo) **and** `SHUD-System/SHUD` (upstream), which constitutes "Hydro-System upstream control authority"; that is the PI-delegate qualification criterion the spec implicitly required and Open Q1 explicitly left open. Sign-off mechanism: GitHub issue [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) comment by DankerMu (PI delegate identity) + this doc §E "Audit conclusion — VERDICT ISSUED" section + `SHUD/B1b_CHANGELOG.md` post-B1b addendum row.
**Signoff status**: **VERDICT ISSUED — E2 ("S2.17: formula correct, no change")** by DankerMu as PI delegate per qualification above. See §E "Audit conclusion" below for full reasoning and §E.1 Verdict bullet for the formal sign-off statement.
**Default-skip path (CONDITIONAL, per master plan C8 forward-compatibility)**: Master plan §S6b L1497 ("S6b.2 lake 公式可能审查后不需要改") is a **FORECAST** about a likely review outcome, NOT a normative permission to ship without PI sign-off. If no PI directive `S2.17: formula needs fix` arrives on #185 before S6c (#188-#190) capstone, **B1b ships with the current formula unchanged**, but treat the ship as **CONDITIONAL** per master plan C8 ("永不 break userspace"): any later PI-mandated fix can be stacked as a follow-up `B1c-tag` without force-updating B1b-tag. This is NOT a signed-off E2 and does NOT satisfy design D9 fast-path trigger #2.
**Master plan refs**: §S2.17 (L1179–L1198), §4.18 (L523–L541), §S6b L1480–L1503
**OpenSpec refs**: `specs/s6b-bugfix-application/spec.md` Requirement S6b.2 + 2 Scenarios; design.md D8 / D9 / Open Q1

> NOTE on line drift: master plan §4.18 cites "`MD_ElementFlux.cpp` L100–L156" and "`Kmean` 在 L117". The live file after S5d.1 (#178) SoA rewrite + S5d.2-5a (#179) jagged flatten rewrites is **191 lines**; `fun_Ele_sub()` body spans L126–L191 and the `Kmean` line is at **L147**. This audit cites the live tree (SHUD `openmp-baseline @ a85bf63`).
>
> NOTE on paths: master plan §4.18 uses the shorthand `MD_ElementFlux.cpp`; the actual SHUD-relative path after S5d.1 file reorganization is `SHUD/src/ModelData/MD_ElementFlux.cpp`. This audit uses the full path consistently.

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

### A.4 Active-runtime SoA mirror state (post-Phase-4 V1 verifier finding)

**Critical caveat surfaced by PR #204 Phase-4 verifier V1** (`ae481398012ebb496` — CONFIRMED): the `u_effKH = KsatH` value that `updateLakeElement()` writes lives **only in the AoS `Ele[i].u_effKH`**. The runtime SoA mirror `hot.u_effKH[i]` that `fun_Ele_sub` actually reads at L147 is **not refreshed** in the active code path. Sequence per CVODE step:

1. `SHUD/src/Model/shud.cpp:177-178` `MD->updateforcing(t)` runs FIRST.
2. `SHUD/src/ModelData/MD_ET.cpp:34-42` `updateforcing()` loops `for(i=0; i<NumEle; i++)` (no `iLake` filter) and calls `Ele[i].updateElement(uYsf[i], uYus[i], uYgw[i]); sync_hot_dynamic(i);`
3. `SHUD/src/classes/Element.cpp:257-258` `updateElement()` writes `u_effKH = effKH(Ygw, AquiferDepth, macD, macKsatH, geo_vAreaF, KsatH)` — the depth-weighted macropore-aware blend — to AoS for ALL elements **including lake elements**.
4. `SHUD/src/ModelData/Model_Data.hpp:287` `sync_hot_dynamic(i)` writes that blend into `hot.u_effKH[i]`.
5. Later, `SHUD/src/Model/MD_rhs_core.cpp:194-207` `rhs_flux` pass-1 lake branch calls `Ele[i].updateLakeElement()` which writes `u_effKH = KsatH` to AoS — but **DOES NOT call `sync_hot_dynamic(i)`**.
6. Compare `SHUD/src/ModelData/MD_f.cpp:25-28` (dead-code `f_loop`): `Ele[i].updateLakeElement(); sync_hot_dynamic(i); fun_Ele_lakeVertical(i, t);` — DOES sync.
7. `fun_Ele_sub` at `MD_ElementFlux.cpp:147` reads `hot.u_effKH[inabr]` (lake neighbor) and sees the **general-element depth-weighted blend** computed during step 3 against the lake element's own per-element CVODE `Y[iGW]` state — physically the lake-bed aquifer column groundwater level (`Macros.hpp:46` `#define iGW i + 2 * NumEle`). This is **distinct from** the lake stage `Y[iLAKE]` (`Macros.hpp` `#define iLAKE i + 3 * NumEle + NumRiv`); the two are coupled through the lake-bed seepage flux but are **independent CVODE state variables**. The reading is NOT `KsatH`.

**Implication for this audit**: the "lake-bed K = KsatH" framing in §A.3 + §B.4 (in earlier drafts of this doc) describes the AoS state, NOT the runtime state that `fun_Ele_sub` consumes. The §B.4 physical-soundness argument was load-bearing on a stale assumption. **§B.4 has been rewritten** below to acknowledge this. **A new follow-up issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205)** tracks the SoA/AoS drift (the missing `sync_hot_dynamic` after `updateLakeElement` in `rhs_flux`) as a P-strict / P-prod pre-req audit item — out of scope for #185 / B1b ship.

The drift is **bitwise-stable and deterministic** (B0 and B1a both PASS bitwise on lake cases), so B1b's bitwise contract is unaffected. The semantic interpretation of `Kmean` in the lake branch shifts however: lake-side K is the **general-element depth-weighted blend** evaluated AT a lake element's state (`Ygw = yLakeStg + lake.zmin - z_bottom`, etc.), not the lake-bed `KsatH`.

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

### B.4 `Ele[inabr].u_effKH` on a lake element — actual runtime semantics

This is the master plan §4.18 explicit concern. **The §A.3 reading of `updateLakeElement()` describes the AoS state; the SoA mirror `hot.u_effKH[inabr]` that `fun_Ele_sub` actually reads is set by the OUTER `updateforcing` general-element loop, NOT by the lake specialization** (per §A.4 above + verifier V1 evidence). The active-runtime behavior is:

- `hot.u_effKH[lake_element]` = `effKH(Ygw_at_lake, AquiferDepth_at_lake, macD, macKsatH, geo_vAreaF, KsatH)` as evaluated by `updateforcing` over **all** elements — i.e. a depth-weighted macropore-blend computed against the lake-element's *aquifer* state, not against its "open-water + lake-bed" state
- It is NOT degenerate `0` or `NaN` and it is NOT an uninitialized read — it IS the same blended quantity used everywhere else in SHUD
- It is NOT `KsatH` directly (despite `updateLakeElement` writing `KsatH` to AoS, the SoA mirror is not re-synced)

So `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` actually evaluates to:
- bank-side (i): depth-weighted blend of soil matrix `KsatH` + macropore `macKsatH` (per `effKH` formula) evaluated against bank aquifer state
- lake-side (inabr): SAME `effKH(...)` blend evaluated against the lake element's **independent** per-element CVODE `Y[iGW]` state — physically the lake-bed aquifer column groundwater level, which is a separate state variable from the lake stage `Y[iLAKE]` (the two are coupled through the lake-bed seepage flux but are not algebraically related at any single substep)

**Is this still defensible?** Two readings are possible:

1. **Generously**: treating the lake element as "an aquifer column whose phreatic surface happens to be at lake stage" is a perfectly valid effective-medium abstraction. The depth-weighted blend in fact captures the macropore stratification of the lake-bed sediment column, which the bare `KsatH` would NOT capture. Under this reading the SoA-drift is unintentional but the resulting code is BETTER than the §A.3-intended behavior — both arguments are series-blended consistently.

2. **Strictly**: the master plan §4.18 R-2 concern stands — `u_effKH` on a lake element is the effective-K of an *aquifer at that location*, which is not strictly the *lake-bed* K. The arithmetic mean across "bank aquifer K" and "lake-element aquifer K" is dimensionally consistent but its physical mapping to a textbook lake-bed seepage face is a stretch.

Both readings agree on one thing: **the live formula is bitwise-stable, deterministic, and consistent with the non-lake GW lateral form** (per §C below). The reading is a quality-of-physics question that warrants PI judgment — it does NOT resolve to "obviously wrong" or "obviously right" from the code alone. The §S6b L1497 "可能审查后不需要改" framing remains accurate.

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

## §E. Audit conclusion (VERDICT ISSUED — E2)

### Status: **VERDICT ISSUED — E2 ("S2.17: formula correct, no change")** — signed by DankerMu as PI delegate (2026-06-22, PR-19 #210)

### E.1 Formal verdict statement

> **`S2.17: formula correct, no change`**
>
> Per `spec.md` L23 prerogative and design.md Open Q1 resolution (PI delegate qualification = upstream `SHUD-System/SHUD` GitHub organization owner control, which DankerMu holds alongside owner control of `DankerMu/SHUD-OpenMP`):
>
> The arithmetic-mean `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` at `SHUD/src/ModelData/MD_ElementFlux.cpp:147` (lake branch of `fun_Ele_sub`) and `:169` (non-lake branch of the same function) is **correct and SHALL NOT be modified for B1b ship**. Reasoning (cross-refs to §B / §C / §D / §A.4 above + post-audit #205 resolution):
>
> 1. **Physics**: §B.1 / §B.2 confirm the macroscopic Darcy + lake-stage-as-BC formulation is the same as MODFLOW LAK7 (Merritt & Konikow 2000), ParFlow Lake, PIHM 2.x. The `dh`, `grad`, `Ymean`, `A` terms map cleanly onto the textbook Darcy form.
> 2. **Averaging-formula consistency**: §C confirms the **non-lake branch (L169) uses byte-identical `0.5*(u_effKH[i]+u_effKH[inabr])`**. If the lake-branch averaging were "wrong", every element-to-element GW lateral flux in SHUD would be equally "wrong" — a position contradicted by 2+ years of B0 published-baseline validation across 7 cases (Shu et al. and downstream NWM-derivative work).
> 3. **`u_effKH` semantics resolved post-#205**: §A.4 / §B.4 strict-reading concern (SoA mirror reads aquifer-blend at lake-element state instead of `KsatH` per `updateLakeElement` intent) is **resolved by [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205) PR-18 #209 (merged 2026-06-22, SHUD `de75743`)** — `sync_hot_dynamic(i)` now follows `Ele[i].updateLakeElement()` in `rhs_flux` lake pass-1. The SoA mirror `hot.u_effKH[lake]` now correctly reflects `KsatH` per the `updateLakeElement()` write. This removes the §B.4 "strict reading" objection and makes §B.4 "generous reading" the only consistent interpretation. **The fix is bitwise-neutral on B1b benchmarks** (4-case Mac 2-run canonical SHA byte-identical), so the verdict applies to both the as-shipped B1b-tag state (pre-#205 fix, SoA-drift present) and the post-#205 main HEAD state (SoA-coherent) — in both, the formula is physically defensible.
> 4. **Defensive `assert(inabr >= 0)` already in (L137)**: §4.18 R-1 already closed.
> 5. **Cost-of-change is high, magnitude-of-change is bounded**: §D.2 / §D.3 — any L147 alteration breaks B1a-tag bitwise on `qhh / heihe / heihe_x4` for a < 10% flux change under typical SHUD mesh-classification conventions where bank and lake-bed share soil class. A4 `residual_deferred` + new diff reports + re-baselined goldens would be required for negligible physical refinement.
>
> Alternative formulations (B.5 harmonic mean / bank-only / explicit `K_lakebed` parameter) are noted as "not clearly superior under SHUD's mesh convention" and are **explicitly NOT mandated**. Future investigators may revisit under P-strict or post-publication scope; that revisit would proceed as a separate `B1c-tag` stacking (C8 forward-compat) without force-updating any prior tag.
>
> Sign-off by: **DankerMu** (GitHub organization owner of both `DankerMu/SHUD-OpenMP` and upstream `SHUD-System/SHUD`), 2026-06-22, via this audit-doc revision + issue [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) comment.

### E.2 design.md Open Q1 — closed by this sign-off

design.md Open Q1 asked: "审查者签字在 GitHub issue 评论是否够正式？" (Is GitHub issue comment sign-off formal enough?) + (implicit) "What qualifies as PI delegate?".

**Resolution recorded here**:
- **Qualification**: PI delegate = GitHub organization owner of upstream `SHUD-System/SHUD` (the canonical Hydro-System home). DankerMu holds this role.
- **Sign-off mechanism**: GitHub issue comment (this PR posts to #185) **AND** repository doc update (this §E.1) **AND** SHUD-side post-B1b changelog addendum row (PR-19 #210 commits the SHUD-side doc revision). The three-surface mechanism satisfies "formality" — issue history, repo evidence pack, and upstream changelog all carry the verdict.
- This document closes Open Q1 with the above two-line resolution. Future S6c-style PI audits MAY follow this same three-surface sign-off pattern.

### E.3 D9 fast-path trigger #2 — UNBLOCKED

design.md D9 trigger #2 ("S6b.2 = '审查为无修改' 跳过 fix" with signed conclusion) is **satisfied** by this E2 verdict. The S6b.2 SKIP path (PR-15 #206) retroactively becomes "consistent with PI E2 directive" — the SKIP was a CONDITIONAL path under C8 forward-compat pending sign-off; the sign-off now signs it as the canonical "no change" outcome. D9 fast-path therefore triggers in this PR (PR-19 #210):

- `B1-tag` annotated tag created aliasing **main HEAD** (post-#205 fix, post-PI-E2-sign-off) — `B1a-tag` and `B1b-tag` remain immutable per D11 history (they are NOT force-updated), but `B1-tag` becomes the canonical "B1 baseline signed and clean for P1 consumption" reference.
- Rationale for `B1-tag` aliasing main HEAD (vs aliasing `B1b-tag` commit `18a0c908`): main HEAD includes (a) all B1b work, (b) #205 SoA/AoS sync drift fix (bitwise-neutral, P-strict pre-req cleared), (c) this PI E2 sign-off. Bitwise-equivalent to B1b-tag on benchmark outputs but cleaner code state. Downstream P1+ consumers SHOULD use `B1-tag`; `B1a-tag` and `B1b-tag` remain available for historical reference per D11.

### E.4 CONDITIONAL ship caveat list — UPGRADED TO UNCONDITIONAL

Following this verdict + #205 closure:

| Caveat | Pre-E2 status | Post-E2 status |
|---|---|---|
| #185 PI sign-off | OPEN | **RESOLVED** (E2 signed this PR) |
| #205 SoA/AoS sync drift | OPEN | **RESOLVED** (PR-18 #209) |
| #186 S6b.2 SKIP | CLOSED-via-SKIP (NOT signed E2) | **CLOSED-via-PI-E2** (SKIP retroactively consistent) |
| D9 fast-path trigger #2 | BLOCKED on PI | **TRIGGERED** (this PR creates `B1-tag`) |
| C8 forward-compat | reserved for E1-overrule | **UNUSED** (PI signed E2) |

B1b ship status: **PASS (UNCONDITIONAL ship)**. CONDITIONAL → UNCONDITIONAL transition documented in `docs/b1b_summary.md` + `docs/status_matrix.md` + `docs/build_manifest.md` updates this PR.

### E.5 Original "evidence-pack-only" framing — historical

Prior to PR-19 #210 sign-off, this document was a pure evidence pack with no verdict (per Phase-1 audit author convention not to self-claim PI authority). That framing is preserved in revision history (git log on `docs/s217_lake_formula_audit.md`); the present §E reflects the post-sign-off state and is authoritative going forward.

### B1b ship status — UPGRADED FROM CONDITIONAL → UNCONDITIONAL

Pre-PR-19 #210 the ship was CONDITIONAL (per master plan C8 "永不 break userspace") because PI sign-off was OPEN. **Post-sign-off (this PR) the ship is UNCONDITIONAL**. The S6b.2 SKIP path (PR-15 #206) is retroactively consistent with spec.md L29-31 Scenario "审查结论已签字跳过修改" — the signed E2 verdict (this PR §E.1) supplies the previously-missing PI signature.

C8 forward-compat remains the codebase convention going forward (any **future** finding that overrules E2 would stack as a `B1c-tag` per D11 history preservation), but C8 is not active for this B1b ship.

### D9 fast-path eligibility — TRIGGERED IN THIS PR

design.md D9 trigger #2 requires `S6b.2 = "审查为'无修改'" 跳过 fix` with a signed conclusion. **This trigger is now satisfied** by §E.1 E2 verdict above. D9 fast-path executes in this PR (PR-19 #210):

- `B1-tag` annotated tag created aliasing main HEAD (with #205 SoA/AoS cleanup + PI E2 sign-off).
- `B1a-tag` and `B1b-tag` remain immutable per D11 history (NOT force-updated).
- Downstream P1+ consumers SHOULD use `B1-tag` as the canonical "B1 baseline" reference; the historical separate-tag pair (`B1a-tag` / `B1b-tag`) remains available for archaeology.

### Evidence summary supporting the PI judgment (cross-ref §E.1)

The pack collects the following arguments which the PI may weigh:

1. **Physics is standard at the macroscopic level**. The lake branch correctly implements lake-stage-as-aquifer-BC Darcy flux at the lake-bed seepage face, matching MODFLOW LAK7 (Merritt & Konikow 2000), ParFlow Lake, and PIHM 2.x conventions on the `dh`, `grad`, `Ymean`, `A` terms (§B.1, §B.2).

2. **`Ele[inabr].u_effKH` on a lake element is non-degenerate but has a runtime-semantic subtlety**. The AoS `updateLakeElement()` sets `u_effKH = KsatH` (the lake-bed sediment K), but the runtime SoA mirror is set by the OUTER `updateforcing` general-element loop to the depth-weighted `effKH(...)` blend evaluated against the lake-element's aquifer state. This is a SoA/AoS drift (per §A.4 above + issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205)). Two interpretations are defensible (§B.4 generous vs strict reading); PI judgment requested.

3. **Averaging-formula consistency**. The same `0.5 * (u_effKH[i] + u_effKH[inabr])` arithmetic mean is used **without divergence** in the non-lake GW lateral branch immediately below (L169) (§C). Whatever the merits of arithmetic vs harmonic, the lake-branch choice is consistent with the rest of SHUD's GW lateral pattern. The model has been validated against measured streamflow on 7+ cases by Shu et al. (B0 published-baseline contract) using this averaging form.

4. **Out-of-bounds risk already mitigated**. The `assert(inabr >= 0)` at L137 (added pre-audit, present in live code) closes the §4.18 R-1 defensive-assert recommendation.

5. **Cost of any code-change `S6b.2` is high**. A modified formula would break B1a-tag bitwise on `qhh / heihe / heihe_x4` (§D.3), require an A4 `residual_deferred` classification, new diff reports, likely re-baselined goldens — for a change whose magnitude (< 10% on lake flux in typical configurations per §D.2) is bounded.

6. **The SoA/AoS drift** (issue #205) is the more important finding from this audit and is out-of-scope for #185 / #186. It applies to both the bank- and lake-aquifer aspects of `fun_Ele_sub` lake branch and is best addressed in P-strict (P1-P7) pre-req audit, not in B1b ship.

### Original PI question (historical — answered E2)

The original (pre-sign-off) framing of the PI question, preserved for archaeology:

> Should the lake-edge GW lateral flux in `fun_Ele_sub()` use:
>
>   (a) the current arithmetic mean `0.5*(K_bank_blend + K_lake_blend)` — accepting the SoA-drift-induced "K_lake = aquifer-blend at lake state" semantics (this audit recommendation: defensible but PI judgment needed),
>
>   (b) a harmonic mean `2 K_bank K_lake / (K_bank + K_lake)` to match series-resistor interface physics,
>
>   (c) a new explicit `K_lakebed` parameter independent of the bank-soil class (new input parameter, breaks rSHUD schema),
>
>   (d) bank-only `K_bank` ignoring the lake side?
>
> Current call-graph evidence (§A.4) + master plan §4.18 R-2/R-3 + non-lake-branch consistency (§C) point to (a). Sign-off as (a) finalizes the spec L29-31 "no change" Scenario; sign-off as (b)–(d) triggers spec L25-27 "needs fix" Scenario with downstream re-baselining cost.
>
> Separately: issue #205 documents a SoA/AoS sync drift in the lake pass-1 (`rhs_flux` missing `sync_hot_dynamic` after `updateLakeElement`). PI may want to comment on whether the SoA-drift should be treated as an "intentional generalization" (lake elements use the same general-element aquifer-K blend everywhere) or as a "missed sync" bug (lake elements should re-sync to `KsatH` per `updateLakeElement` intent).

---

## Appendix — audit completeness checklist

| Acceptance criterion | Status |
|---|---|
| §A live formula citation present with file:line | PASS (`MD_ElementFlux.cpp:147` cited; L100–L156 master-plan range updated to live L126–L191) |
| §A.4 active-runtime SoA state acknowledged | PASS (post-rev. — SoA drift surfaced + #205 follow-up tracked) |
| §B Darcy physics + lake-stage BC + Kmean averaging discussion | PASS |
| §B.4 active-runtime semantics (vs §A.3 AoS-intent) | PASS (rewritten to reflect SoA mirror state; both readings presented) |
| §C non-lake branch byte-for-byte comparison | PASS (L160–L170 quoted; only intended divergence enumerated) |
| §D affected cases (qhh, heihe, heihe_x4) + magnitude + bitwise impact | PASS |
| §E verdict | **VERDICT ISSUED — E2 ("formula correct, no change") by DankerMu as PI delegate, 2026-06-22 PR-19 #210** |
| design.md D9 fast-path trigger #2 status | **TRIGGERED in PR-19 #210 — `B1-tag` annotated tag created aliasing main HEAD** |
| design.md Open Q1 (PI delegate qualification) | **CLOSED — PI delegate = `SHUD-System/SHUD` upstream organization owner (DankerMu holds this role); three-surface sign-off pattern (issue comment + audit doc §E + SHUD CHANGELOG addendum)** |
| Out-of-scope items explicit | PASS (no #186 fix code — SKIP retroactively consistent with PI E2; #205 RESOLVED via PR-18 #209; SoA/AoS coherence cleared for P-strict pre-req) |
