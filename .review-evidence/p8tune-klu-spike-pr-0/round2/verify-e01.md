Verifier verdict for: e01
Reviewed head SHA: 50d2a4bddacbfa3ef5b3e1c25d760555103c5556
Verdict: CONFIRMED
Evidence:
  - KLU header `/opt/homebrew/Cellar/suite-sparse/7.12.2/include/suitesparse/klu.h` L36-39 documents
    explicitly: `/* only computed if the AMD ordering is chosen: */` followed by
    `double symmetry; double est_flops; double lnz, unz; /* estimated nz in L and U, including diagonals */`.
    The `lnz` and `unz` fields therefore have well-defined semantics ONLY when AMD ordering is used.
  - `tools/p8tune.D/klu_analyze_factor.cpp` L150-157 builds an `ordering_id` from {amd=0, colamd=1, natural=2}.
    L244-251 dispatches: `ordering_id==2` -> `klu_analyze_given(...)` (identity P/Q), otherwise
    `klu_analyze(...)` with `common.ordering = ordering_id` (0=AMD, 1=COLAMD).
  - L266-279 (the new post-analyze preflight introduced in F4) is UNCONDITIONALLY executed for ALL three
    orderings: it computes
      `est_after_analyze_bytes = static_cast<size_t>(Symbolic->lnz + Symbolic->unz) * 24ULL * 3ULL / 2ULL;`
    with NO `ordering_id == 0` guard. The companion log line at L271-272 prints the raw
    `Symbolic->lnz + Symbolic->unz` as `%.0f`.
  - For ordering ∈ {colamd, natural}, per the upstream contract in klu.h L36-39 these double fields are
    not computed by `klu_analyze` (COLAMD path) or by `klu_analyze_given` (natural path). The reviewer's
    citation to upstream KLU source matches the header contract: such fields are left at the EMPTY (-1)
    sentinel for non-AMD paths. `static_cast<size_t>(-2.0)` is implementation-defined / UB per
    C++ [conv.fpint]; on Apple Silicon clang it typically wraps to a value near SIZE_MAX, which would
    immediately exceed the `rss_budget = 0.7 * CN_NODE_RAM_BYTES` test at L269-273 and emit a spurious
    `KLU_OOM_DETECTED ... reason=preflight_after_analyze` line (L274-275), causing the cell to exit 0
    BEFORE `klu_factor` is ever invoked — directly violating REQ-5 Scenario "OOM-as-data-point"
    (spec L138-143), which requires OOM to be classified only when numeric factor truly exhausts RAM.
  - PR-0 mac smoke evidence at `.review-evidence/p8tune-klu-spike-pr-0/mac_smoke_keliya_klu_factor.log`
    covers only `case=keliya ordering=amd btf=1` (L8); the colamd / natural orderings (6 of 8 cells in
    the PR-A 4-case × 3-ordering × 2-btf matrix that route through this preflight without AMD-populated
    lnz/unz) are NOT smoke-tested, so the UB path is unexercised in the merging branch.
Note: The bug reopens REQ-5 for 6/8 PR-A cells (3 ordering values × 2 btf values × any case ÷ skip AMD); the c04 round-1 fix only repaired AMD.
