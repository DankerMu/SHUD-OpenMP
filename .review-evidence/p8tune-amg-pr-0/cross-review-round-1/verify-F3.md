Verifier verdict for: F3 — Sibling-class invariant closure incomplete
Reviewed head SHA: 09a815dbcab9eabbddcad9550c706ddfa8636519
Verdict: CONFIRMED

Anti-pattern existence per class:

  - _TimeSeriesData: CONFIRMED
    - ctor (TimeSeriesData.cpp:28-29): `_TimeSeriesData::_TimeSeriesData(){}` — empty body, NO member init
    - members (TimeSeriesData.hpp:41): `double *ts[MAXQUE + 1];` — raw ptr array, NOT default-initialized (no `= {nullptr}` NSDMI)
    - dtor (TimeSeriesData.cpp:31-38): unconditional `delete[] ts[i]` loop with NO null-check, NO init guard
    - Model_Data reachability: 9 value-type TSD members at Model_Data.hpp:118-127
      (`tsd_LAI, tsd_MF, tsd_eleSS, tsd_eyBC, tsd_eqBC, tsd_ryBC, tsd_rqBC, tsd_lyBC, tsd_lqBC`)
      — ~Model_Data() implicitly invokes ~_TimeSeriesData() on each, EVERY shud exit.
      Plus `_TimeSeriesData *tsd_weather = nullptr` at line 117 (NSDMI'd PR-0; but the
      pointee, if `new`-allocated and partially init, still suffers same UB via its dtor)

  - _Lake: CONFIRMED
    - ctor (Lake.cpp:82-83): `_Lake::_Lake(){}` — empty body
    - members (Lake.hpp:88-93,98-101): `int *iEleLake, *iEleBank, *iRivIn, *iRivOut, *RivIn, *RivOut;
      double *QEleSurf, *QEleGW, *QRivIn, *QRivOut;` — ALL raw uninit ptrs (no NSDMI)
    - dtor (Lake.cpp:84-96): guards by `NumEleBank>0` / `NumRivIn>0` / `NumRivOut>0`
      counters (Lake.hpp:85-87 init these counters to `NA_VALUE = -9999`, NOT zero),
      so the guard `if(NumEleBank > 0)` evaluates FALSE on uninit→safe-by-coincidence,
      BUT once `readLake()` runs and sets `NumEleBank` to a real value (line 110), the
      ptr is written by `new int[NumEleBank]` (line 111); if a downstream `new` in the
      same fn throws (lines 117, 123 — `new int[NumRivIn]`, `new int[NumRivOut]`), the
      partial state is `iEleBank` allocated + `RivIn` uninit + `NumRivIn>0` already set
      by fscanf line 116 → dtor's `delete RivIn` on uninit garbage = UB
    - ALSO: scalar `delete` on `new[]`-allocated arrays at lines 86-88, 91, 94 is UB
      per C++ spec regardless of init state (Lake.cpp:86 `delete iEleBank` vs Lake.cpp:111
      `new int[NumEleBank]`)
    - Model_Data reachability: `_Lake *lake = nullptr;` at Model_Data.hpp:172 — NSDMI-safe
      against unallocated case, but allocated-and-partial-init reaches dtor via
      `delete[] lake` chain in FreeData()

  - LakeBathymetry: CONFIRMED
    - ctor (Lake.cpp:37): `LakeBathymetry::LakeBathymetry(){}` — empty
    - members (Lake.hpp:64-67): `int nvalue = NA_VALUE; int *index; double *yi; double *ai;`
      — `nvalue` initialized to `NA_VALUE` (-9999, NOT zero), ptrs uninit
    - dtor (Lake.cpp:52-58): `if(nvalue > 0) { delete index; delete yi; delete ai; }`
      — guard works for unallocated case (nvalue=-9999), BUT:
      (a) `InitValue(n)` at lines 132-141 sets `nvalue=n` BEFORE allocating, and if
      `new int[nvalue]` (line 134) succeeds but `new double[nvalue]` (line 135 or 136)
      throws, partial state: nvalue=n>0, index allocated, yi/ai uninit → dtor's
      `delete yi/ai` on garbage = UB
      (b) scalar `delete` on `new[]`-allocated arrays is UB regardless
    - Model_Data reachability: `_Lake::bathymetry` is value-type member (Lake.hpp:94),
      reached via `~_Lake` → `~LakeBathymetry`. NB: Currently `LakeBathymetry::read` is
      fully commented (Lake.cpp:39-51) and `_Lake::readBathymetry` is no-op (Lake.cpp:128-130),
      so the realistic OOM-mid-init scenario via `InitValue` requires a non-test caller
      path; the anti-pattern (scalar delete on new[]) is independent of caller.

  - TabularData: CONFIRMED (anti-pattern exists; weaker dtor-chain reachability from Model_Data)
    - ctor (TabularData.cpp:3-5): empty
    - members (TabularData.hpp:17-20): `int nrow = 0; int ncol = 0; double **x;` — `nrow`
      DEFAULT-INITIALIZED to 0, so dtor's `reset()` guard `if(nrow > 0)` (TabularData.cpp:11)
      DOES protect the unallocated case — this is the partial closure already present
    - dtor → reset() (TabularData.cpp:7-20): `delete[] x[i]` then `delete[] x` — uses correct
      `delete[]` (not scalar delete like _Lake/LakeBathymetry)
    - HOWEVER: `read()` sets `nrow = sscanf-value` at line 32 BEFORE `x = new double*[nrow]`
      at line 37; if any `new double[ncol]` at line 39 in the inner loop throws on row k,
      partial state is nrow>0, x[0..k-1] allocated, x[k..nrow-1] uninit → reset()
      `delete[] x[i]` for i>=k = UB on garbage
    - Model_Data reachability: TabularData is NOT a direct Model_Data member; only used
      as local in `LakeBathymetry::read` (currently commented out, Lake.cpp:40) and
      via callers (MD_readin.cpp etc) as stack/local objects. The "fires on every shud exit"
      claim does NOT hold via Model_Data dtor chain specifically — TabularData lifetimes
      are caller-scoped, and stack unwinding during read/init is the trigger path

Scope verdict (per spec REQ-3 + design.md §D5):

  - in PR-0 scope: yes — design.md §D5 PR-0 Invariant Matrix (line 107):
    "`Model_Data` 构造 → 使用 → 析构链全程零 invalid pointer / 零 double-free / 零 use-after-free"
    AND line 109 "Source-of-truth identity/contract: ... sibling `SubClass` 析构链中每个
    pointer 字段必须满足 (i) 构造函数 `nullptr` init, 或 (ii) dtor 顺序保 init 之后再 `delete[]`.
    涵盖范围由 PR-0 任务 1.1 `grep delete\[\]` 列出, 不依赖人工记忆."
    AND line 113 "Producers: `SHUD/src/ModelData/MD_*.{c,cpp,h}` 构造函数 + `SHUD/src/Equations/*` init helper"
    — note: scope text explicitly names "sibling `SubClass` 析构链", which directly covers
    _TimeSeriesData (value-type sub-member of Model_Data, ~Model_Data() invokes 9 copies)
    and _Lake/LakeBathymetry (via `_Lake *lake` pointee dtor). TabularData is the
    weakest fit: it is not a Model_Data SubClass; reached only via init-time local
    objects in MD_readin.cpp callers, hence within "全程" interpretation IF read-time
    exception path is in scope (likely yes per "构造 → 使用 → 析构链全程").
    BUT line 109 also constrains coverage to "PR-0 任务 1.1 `grep delete\[\]` 列出" —
    PR-0's commit message (SHUD 056a1dc) shows the grep was scoped to ModelData/ only
    (NOT SHUD/src/classes/), so the spec's own scope mechanism (grep result) excluded
    these siblings by mechanical filter. This is a spec-vs-implementation gap: governing
    invariant (line 107) says "全程", scope mechanism (line 109) says "by grep result",
    and PR-0 grep was narrower than the governing invariant requires.

Overall verdict: CONFIRMED (anti-pattern present in all 4 sibling classes; spec §D5
governing invariant text explicitly covers "sibling SubClass 析构链" which includes
_TimeSeriesData + _Lake + LakeBathymetry as constructible failure scenarios via
~Model_Data; TabularData reachability is via init-time exception not ~Model_Data direct
dtor chain, so it's weakly in scope under "构造 → 使用 → 析构链全程")

Recommended action: mixed
  - In-PR-0: extend NSDMI nullptr defaults + dtor null-guards to _TimeSeriesData::ts[],
    _Lake (all 6 int* + 4 double* members), LakeBathymetry (index/yi/ai). Also fix
    scalar `delete` vs `delete[]` mismatch in _Lake + LakeBathymetry dtors (independent
    bug, UB regardless of init state). These three are reachable via ~Model_Data dtor
    chain and are squarely inside §D5 "sibling SubClass 析构链" text.
  - Defer-to-follow-up: TabularData partial-init-during-read exception path. TabularData
    is NOT a direct Model_Data member; reached only as caller-scoped locals during
    init. Documenting as known-narrow-scope-gap acceptable IF PR-0 description explicitly
    notes "TabularData::read exception-during-allocation is out-of-scope for PR-0; tracked
    as follow-up issue".
  - Alternative: explicit Invariant Matrix downgrade per candidate's option (ii) — change
    §D5 line 107 governing invariant from "析构链全程" to "Model_Data ptr-members only"
    and acknowledge sibling-class anti-pattern as known non-goal. This honestly closes
    the PR-vs-spec gap WITHOUT extending fix scope, but it weakens the invariant the
    PR claims to establish.

Evidence:
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TimeSeriesData.cpp:28-29 (empty ctor), :31-38 (unguarded delete[] loop)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TimeSeriesData.hpp:41 (`double *ts[MAXQUE+1]` uninit raw ptr array)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/Lake.cpp:82-83 (empty ctor), :84-96 (counter-guarded but scalar delete on new[]), :37 (LakeBathymetry empty ctor), :52-58 (LakeBathymetry dtor scalar delete on new[]), :132-141 (InitValue allocates after nvalue=n)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/Lake.hpp:64-67 (LakeBathymetry: nvalue=NA_VALUE, ptrs uninit), :85-87 (_Lake counters init to NA_VALUE), :88-101 (10 raw ptrs uninit)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TabularData.cpp:7-9 + :10-20 (dtor → reset; nrow>0 guard does cover unallocated, NOT partial-init), :32-39 (nrow set BEFORE new allocations)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/classes/TabularData.hpp:17-19 (`nrow=0` NSDMI does protect unallocated case; double **x uninit)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/src/ModelData/Model_Data.hpp:117-127 (1 ptr + 9 value TSD members), :172 (_Lake *lake NSDMI-defaulted to nullptr)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/openspec/changes/p8tune-amg-spike/design.md:107 (governing invariant "全程"), :109 ("sibling SubClass 析构链中每个 pointer 字段"), :118-119 (regression rows do NOT include sibling-class smoke), :130-138 (boundary checklist scopes to ModelData ctor/dtor + FreeData, NOT siblings)
  - /Users/danker/Desktop/Hydro-SHUD/openMP/openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md:49 (REQ-3 Scenario "grep `delete\[\]` and constructor patterns in `SHUD/src/ModelData/` (and any sibling `~Model_Data` chain)") — the parenthetical explicitly extends to "sibling `~Model_Data` chain", which is precisely the _TimeSeriesData + _Lake + LakeBathymetry siblings

Note: Spec text in §D5 line 107 + REQ-3 line 49 explicitly covers "sibling ~Model_Data chain"; the gap is implementation-vs-spec, not spec scope ambiguity. Severity rating critical vs major depends on whether you weight the spec's letter (sibling siblings ARE in scope per text) over its mechanical scope (grep was narrower).
