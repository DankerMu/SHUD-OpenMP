# P1.0 pre-audit: updateElement / updateRiver / lake.update + f_updatei case 1-5

Reviewer-only audit for `openspec/changes/p1-update-omp/` PR-C (issue #215).
No SHUD source change. Output binds design.md D9 path selection and gates
PR-D / PR-E / PR-F implementation.

## 1. Audit target & scope

- 5 functions + 1 dispatcher under audit (read set / write set / shared object
  writes / RNG-time-IO / verdict):
  1. `_Element::updateElement(double, double, double)` — `SHUD/src/classes/Element.cpp:257`
  2. `_River::updateRiver(double)` — `SHUD/src/classes/River.cpp:49`
  3. `_Lake::update()` — `SHUD/src/classes/Lake.cpp:104`
  4. `Model_Data::f_updatei(Y, DY, t, flag)` case 1-5 — `SHUD/src/ModelData/MD_update.cpp:6-62`
  5. `Model_Data::f_update(Y, DY, t)` three owner loops — `SHUD/src/ModelData/MD_update.cpp:63-154`
- Audit type: reviewer-only, no source change.
- Linked spec: `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md`
  Requirement **"P1.0 pre-audit"**.
- Decision binding: `openspec/changes/p1-update-omp/design.md` decision D9
  (a / b.1 / b.2 path selection).
- Audit anchors:
  - outer commit: `008913be8bb2b9be3720dbbfa01e309a9a34ee22` (main HEAD, 2026-06-22)
  - SHUD submodule: `017c629e0359845821e51bb0b172ad02452a2541`

### 1.1 Helper functions confirmed pure / re-entrant

All callees inside the 5 functions are stateless (input → output, no
hidden state, no IO, no RNG, no global mutation):

| Helper | Defined at | Property |
| --- | --- | --- |
| `effKH(Ygw, aqDepth, MacD, Kmac, AF, Kmx)` | `Equations.cpp:116` | pure; `myexit` only on terminal error (each thread sees its own Ele's geometry) |
| `satKfun(elemSatn, n)` | `Equations.cpp:136` | pure |
| `sat2psi(elemSatn, alpha, n)` | `Equations.hpp:31` (inline) | pure |
| `fixMaxValue(x, defVal)` | `functions.hpp:183` (inline) | pure |
| `fun_TopWidth / fun_CrossArea / fun_CrossPerem / fun_EqWidth` | `River.hpp:115-127` (inline) | pure |
| `LakeBathymetry::toparea(y)` | `Lake.cpp:59-78` | reads only `this->yi[], ai[], nvalue` (this = lake-local instance member); no mutation; safe |
| `_TimeSeriesData::getX(t, col)` | `TimeSeriesData.cpp:122-125` | returns `ts[iNow][col]`; pure read; thread-safe per S5a #176 contract ("thread-safe read-only after movePointer") |

`movePointer` (the only mutator on `_TimeSeriesData`) is **not** called
inside any of the 5 audit targets. It runs once per outer step outside
RHS / update parallel regions. Confirmed by grep of MD_update.cpp.

### 1.2 Index macros are owner-local

From `SHUD/src/Model/Macros.hpp:44-48`:

```
#define iSF     i
#define iUS     i + NumEle
#define iGW     i + 2 * NumEle
#define iRIV    i + 3 * NumEle
#define iLAKE   i + 3 * NumEle + NumRiv
```

Each macro expands to a unique offset keyed on the per-iteration `i`.
Distinct loop iterations therefore touch disjoint `Y[]` slots — no false
sharing of array slots between threads.

## 2. `_Element::updateElement(double Ysurf, double Yunsat, double Ygw)` audit table

| Aspect | Finding |
| --- | --- |
| Signature | `void _Element::updateElement(double Ysurf, double Yunsat, double Ygw)` |
| Caller args | `uYsf[i], uYus[i], uYgw[i]` from `f_update` element loop (`MD_update.cpp:74,75,78,82`); `Y[]`-derived values from `f_updatei` case 1/2/3 |
| Read set (member) | `AquiferDepth`, `macD`, `macKsatH`, `geo_vAreaF`, `KsatH`, `infKsatV`, `hAreaF`, `macKsatV`, `ThetaS`, `ThetaR`, `Alpha`, `Beta` (all read-only after `copyGeol/copySoil/copyLandc` at init) |
| Read set (param) | `Ysurf` (unused inside body — note: parameter is bound but only `Yunsat`, `Ygw` consumed), `Yunsat`, `Ygw` |
| Write set (member) | `u_effKH`, `u_deficit`, `Kmax`, `u_satn`, `u_theta`, `u_satKr`, `u_phius`, `u_effkInfi` — **all `this`-instance members** |
| Global read | none |
| Global write | none |
| RNG / time / IO | none (no `rand`, no `time`, no `fprintf` except `#ifdef DEBUG` block which is commented out) |
| External calls | `effKH`, `satKfun`, `sat2psi`, `max` — all pure (§1.1) |
| Shared object write | **none** — every write goes through `this->u_*` private/public scalar members; each thread holds a distinct `&Ele[i]`, so `this` is owner-local |
| Thread-safety verdict | **safe** |

Note: `Ysurf` is a dead parameter (declared but unread inside body).
Documented as observation, not a defect.

## 3. `_River::updateRiver(double newY)` audit table

| Aspect | Finding |
| --- | --- |
| Signature | `void _River::updateRiver(double newY)` |
| Caller arg | `uYriv[i]` from `f_update` river loop (`MD_update.cpp:111`); `f_updatei` case 4 (`MD_update.cpp:41`) |
| Read set (member) | `BottomWidth`, `bankslope`, `Length` (all read-only after `applyParameter/initialRiver` at init) |
| Read set (param) | `newY` |
| Write set (member) | `u_Ystage`, `u_topWidth`, `u_CSarea`, `u_CSperem`, `u_eqWidth`, `u_TopArea` — **all `this`-instance members** |
| Global read | none |
| Global write | none |
| RNG / time / IO | none |
| External calls | `fun_TopWidth`, `fun_CrossArea`, `fun_CrossPerem`, `fun_EqWidth`, `fixMaxValue` — all pure inline (§1.1) |
| Cross-river dependency | **none** — body never indexes `Riv[neighbor]`, never reads `down`, `RivOut`, or any topology field. State update is strictly local to `Riv[i]`. (Contrast: `updateFrDownstream` does read `DownRiv[idown]`, but that runs at init only, not on update.) |
| Shared object write | **none** |
| Thread-safety verdict | **safe** |

## 4. `_Lake::update()` audit table

| Aspect | Finding |
| --- | --- |
| Signature | `void _Lake::update()` (no params; reads `this->yStage`, `this->zmin`) |
| Caller setup | `f_update` lake loop sets `yLakeStg[i] = Y[iLAKE]; lake[i].yStage = yLakeStg[i];` (`MD_update.cpp:137-138`) BEFORE calling `lake[i].update()` |
| Read set (member) | `yStage`, `zmin`, `bathymetry` (LakeBathymetry instance owned by `this`) |
| Write set (member) | `u_toparea` — owner-local |
| Global read | none |
| Global write | none |
| RNG / time / IO | none |
| External calls | `bathymetry.toparea(y)` — operates on `this->bathymetry`, a per-lake instance; pure read of its own `yi/ai/nvalue` arrays (§1.1) |
| Shared object write | **none** |
| Caller-side audit | L137 `yLakeStg[i] = Y[iLAKE]` — owner-local array write (iLAKE = i + 3·NumEle + NumRiv, disjoint per `i`). L138 `lake[i].yStage = yLakeStg[i]` — owner-local member write on distinct `&lake[i]`. Both safe. |
| Thread-safety verdict | **safe** |

## 5. `f_update` three owner loops (`MD_update.cpp:63-154`)

| Loop | Range | Reads | Writes (owner-local only) | External calls | Safety |
| --- | --- | --- | --- | --- | --- |
| **element** L64-L105 | `i = 0 .. NumEle-1` | `Y[iSF], Y[iUS], Y[iGW]`, `Ele[i].iBC`, `tsd_eyBC/eqBC` (via `getX`, pure read), `t` | `QeleSubAt(i,j)`, `QeleSurfAt(i,j)` for j=0..2 (flat[3*i+j], owner-local); `QeleSubTot[i]`, `QeleSurfTot[i]`; `uYsf[i]`, `uYus[i]`, `uYgw[i]`; `Ele[i].QBC`, `Ele[i].yBC`; `qEleExfil[i]`, `qEleInfil[i]` | `tsd_eyBC.getX`, `tsd_eqBC.getX` (pure, §1.1); `max` (pure) | **safe** — all writes keyed on `i`; all `Ele[i]` are distinct objects per thread |
| **river-update** L107-L125 | `i = 0 .. NumRiv-1` | `Y[iRIV]`, `Riv[i].BC`, `tsd_rqBC/ryBC` (via `getX`, pure read), `t` | `uYriv[i]`; `Riv[i].updateRiver(uYriv[i])` — fully owner-local per §3; `Riv[i].qBC`, `Riv[i].yBC` | `Riv[i].updateRiver` (§3); `tsd_rqBC.getX`, `tsd_ryBC.getX` (pure); `CheckNANi` (DEBUG only) | **safe** |
| **river-clear** L127-L131 | `i = 0 .. NumRiv-1` | none (write-only zero) | `QrivSurf[i], QrivSub[i], QrivUp[i]` — owner-local | none | **safe** |
| **element-clear** L132-L135 | `i = 0 .. NumEle-1` | none | `Qe2r_Surf[i], Qe2r_Sub[i]` — owner-local | none | **safe** |
| **lake** L136-L147 | `i = 0 .. NumLake-1` | `Y[iLAKE]` | `yLakeStg[i]`; `lake[i].yStage`; `lake[i].update()` (§4 — owner-local); `y2LakeArea[i]`; `QLakeSub[i], QLakeSurf[i], qLakeEvap[i], qLakePrcp[i], QLakeRivIn[i], QLakeRivOut[i]` — all owner-local | `lake[i].update` (§4) | **safe** |
| **DY-clear** L148-L150 | `i = 0 .. NumY-1` | none | `DY[i]` — owner-local | none | **safe** (out of P1 scope but listed for completeness) |

No cross-iteration dependency (no `Ele[i+k]` / `Riv[neighbor]` / `lake[j]`
reads with `j != i`). No reduction. No I/O inside the loops (only the
post-loop `shud_rhs_dump_point` outside the parallel candidate region).
All BC reads go through `_TimeSeriesData::getX`, which is documented
S5a-thread-safe (§1.1) given `movePointer` is sequential and called
elsewhere.

## 6. `f_updatei` case 1-5 audit + cross-mapping vs `f_update`

`f_updatei(Y, DY, t, flag)` is an alternative entry point used for
partial state refreshes. It is **out of PR-D/E/F parallelization scope**
per spec NG7 — this section is provided for P2a reviewer reference only.

| Case | Range | Semantics | Reads | Writes (owner-local) | vs `f_update` equivalent | Safety |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | `i = 0 .. NumEle-1` (L8-12) | `uYsf` refresh w/ floor=0 | `Y[i]` | `uYsf[i]` | subset of f_update element loop write set; identical floor-at-zero semantics (L74 has no floor — minor divergence, BUT this is the by-design `flag` separation) | **safe** |
| **2** | `i = 0 .. NumEle-1` (L13-17) | `uYus` refresh w/ floor=0 | `Y[i]` | `uYus[i]` | subset of f_update element loop (L75 has no floor) | **safe** |
| **3** | `i = 0 .. NumEle-1` (L18-32) | `uYgw` + Element BC update | `Y[i]`, `Ele[i].iBC`, `tsd_eyBC/eqBC.getX`, `t` | `uYgw[i]`, `Ele[i].QBC`, `Ele[i].yBC` | matches f_update element-loop BC sub-block (L76-86); same `getX` pure-read; same per-`Ele[i]` member writes | **safe** |
| **4** | `i = 0 .. NumRiv-1` (L33-53) | `uYriv` + `Riv[i].updateRiver` + Riv BC | `Y[i]`, `Riv[i].BC`, `tsd_rqBC/ryBC.getX`, `t` | `uYriv[i]`, `QrivUp[i]=0`, `QrivDown[i]=0`, `Riv[i].updateRiver(uYriv[i])` writes (§3), `Riv[i].qBC`, `Riv[i].yBC` | matches f_update river-update loop (L107-125) write set; `updateRiver` thread-safety condition identical (proven §3) | **safe** |
| **5** | `i = 0 .. NumLake-1` (L54-58) | `uYlake` refresh w/ floor=0 | `Y[i]` | `uYlake[i]` | subset of f_update lake loop (L137 has no floor; f_updatei applies floor) | **safe** |

**Conclusion on f_updatei**: read/write sets for every case are a subset
of (or identical to) the corresponding `f_update` owner loop. Thread-
safety inherits from §2-5. The `updateRiver` call in case 4 carries the
same safety condition as the call in `f_update` river loop — both rely
on `_River::updateRiver` being owner-local-only, which §3 establishes.

The functional divergence between `f_updatei` cases and `f_update`
(presence vs absence of `(Y[i] >= 0.) ? Y[i] : 0.` floor) is **a
semantics-level design choice, not a thread-safety concern**. Audit
makes no recommendation on parallelizing `f_updatei` itself — that
decision is deferred to P2a per spec NG7.

## 7. Conclusion + sign-off

### 7.1 Verdict

**(a) safe** — all 5 functions + `f_updatei` case 1-5 + `f_update` three
owner loops perform strictly owner-local member / array writes, with:

- **No shared object writes**: every write target is either (i) an array
  slot indexed by the loop variable `i` (uYsf/uYus/uYgw/uYriv/uYlake,
  QeleSub*/QeleSurf*/Qe2r_*, QrivSurf/Sub/Up/Down, QLake*, qLakeEvap/Prcp,
  yLakeStg, y2LakeArea, qEleExfil/Infil) or (ii) a member of the
  per-iteration object `Ele[i]` / `Riv[i]` / `lake[i]` (QBC, yBC, qBC,
  yStage, u_*).
- **No global mutation**: no writes to `tsd_*` (only `getX` pure reads),
  no writes to model-level scalars (`NumEle`, `NumRiv`, `NumLake`, etc.
  are read-only), no static-local mutation observed.
- **No RNG / time / IO** inside the loop bodies (DEBUG `CheckNANi` /
  `CheckNANij` are read-only sanity checks; `shud_rhs_dump_point` is
  outside the loop bodies under `#ifdef SHUD_DUMP_RHS`).
- **No cross-iteration data dependency**: no `Ele[i+k]`, `Riv[neighbor]`,
  or `lake[j != i]` reads inside the bodies. Neighbor topology (`down`,
  `nabr[]`, `lakenabr[]`, `iEleBank[]`) is only read by `*Flux` routines,
  not by the update functions audited here.
- **All external callees pure / re-entrant**: §1.1 enumerates the 12
  helpers used; all are stateless or read-only on per-instance owned
  data with no inter-thread sharing.

### 7.2 design D9 path

**Path (a) selected**: in-scope, no PR-Cfix needed. PR-D / PR-E / PR-F
may proceed directly to add `#pragma omp parallel for` over the three
`f_update` owner loops (element / river-update / lake) without any
source-code restructuring of the audited functions.

`f_updatei` case 1-5 are **not** in PR-D/E/F parallelization scope per
spec NG7; this audit serves only as forward reference for P2a reviewer.

### 7.3 Five-function thread-safety summary (one-line each)

| # | Function | Verdict | Rationale |
| --- | --- | --- | --- |
| 1 | `_Element::updateElement` | **safe** | writes only `this->u_*` on owner-local `Ele[i]`; pure helper calls |
| 2 | `_River::updateRiver` | **safe** | writes only `this->u_*` on owner-local `Riv[i]`; no `Riv[neighbor]` access |
| 3 | `_Lake::update` | **safe** | writes only `this->u_toparea` on owner-local `lake[i]`; `bathymetry.toparea` pure read |
| 4 | `f_updatei` case 1-5 | **safe** | read/write sets are subset of f_update equivalents; same getX-only BC reads |
| 5 | `f_update` three loops | **safe** | every write keyed on `i`; no cross-iteration reads; BC reads via pure `tsd_*.getX` |

### 7.4 Sign-off metadata

- signed_at: 2026-06-22
- signer: DankerMu (project owner)
- signed_against_commit: outer `008913be8bb2b9be3720dbbfa01e309a9a34ee22` + SHUD `017c629e0359845821e51bb0b172ad02452a2541`
- task ref: `openspec/changes/p1-update-omp/tasks.md` task 3.1-3.5 + 3.5b
- spec ref: `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md` Requirement "P1.0 pre-audit"
- design ref: `openspec/changes/p1-update-omp/design.md` decision D9 path (a)
- PR: #215 (PR-C)
