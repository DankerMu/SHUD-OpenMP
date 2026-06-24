# P1d PR-G — SHUD Kahan (P1c §4.7 Neumaier) revert

**Scope**: Surgical removal of the P1c §4.7 conditional Kahan / Neumaier-1974 compensation injected in PR-K2 (SHUD `3a0004c`) from the 3 reduction helpers in `SHUD/src/Model/MD_rhs_core.cpp`, while preserving the PR-C/D/E NUMA steady-state first-touch loops added in `6aada88`. The revert is the first half of the PR-F → PR-G → PR-H strategy: PR-G removes the masking layer so PR-H can measure whether the first-touch + B0-ordering combo alone clears the three SHALL gate, or whether Kahan must be reinstated.

## What was reverted (4 changes)

All changes localised to `SHUD/src/Model/MD_rhs_core.cpp`. SHUD diff scope = this single file (7 insertions, 47 deletions vs PR-E HEAD `6aada88`).

1. **`#include <cmath>`** (was line 38–40, post-PR-E shift) + its 3-line "P1c §4.7 conditional Kahan injection: std::fabs used by Neumaier compensation branch in fixed_*_sum helpers below. Explicit include avoids transitive dependency." comment block — removed. `<cmath>` was only needed for `std::fabs` in the Neumaier branches.
2. **`fixed_pairwise_sum_range`** (now ends at L278 post-revert): the 4-line Neumaier-compensated pair-join `const double t = lo + hi; const double c = (std::fabs(lo) >= std::fabs(hi)) ? (lo - t) + hi : (hi - t) + lo; return t + c;` reverted to the original single line `return lo + hi;`. The 7-line P1c §4.7 doc-comment block above the function ("P1c §4.7 Kahan injection (Neumaier 1974, conditional): every recursive pair-join...") and the 3-line in-body "P1c §4.7 Neumaier-compensated pair-join. Branch on magnitude..." comment were both deleted.
3. **`fixed_leftfold_sum_indexed`** (now ends at L298 post-revert): the 5-line Neumaier-compensated leftfold loop `double acc = 0.0; double c = 0.0; for (int i : idx) { const double x = src[i]; const double t = acc + x; c += (std::fabs(acc) >= std::fabs(x)) ? (acc - t) + x : (x - t) + acc; acc = t; } return acc + c;` reverted to the original 3-line `double acc = 0.0; for (int i : idx) acc += src[i]; return acc;`. The 5-line P1c §4.7 doc-comment block above the function was deleted.
4. **`fixed_leftfold_sum_pair_indexed`** (now ends at L313 post-revert): identical Neumaier reversion pattern on the pair-list overload. Reduced to the original 3-line `double acc = 0.0; for (const auto& ej : pairs) acc += src[ej.first * stride + ej.second]; return acc;`. The 5-line P1c §4.7 doc-comment block above the function was deleted.

Post-revert grep gates in `SHUD/src/Model/MD_rhs_core.cpp`:

- `grep -n "Kahan\|Neumaier"` = **0 hits** ✓
- `grep -n "P1c §4.7"` = **0 hits** ✓
- `grep -n "cmath"` = **0 hits** ✓ (no orphan include)
- `grep -n "fabs"` = 2 hits (both `fabs(DY[...])` in pre-existing legacy DEBUG-block COMMENTED-OUT printf code at L722 / L738 — not Kahan related; pre-existed since well before P1c)

## What was preserved (invariant matrix)

- **PR-C element first-touch** (`rhs_update` L62 extern decl + L82-98 gated block + L62-84 design comment) — byte-identical.
- **PR-D river first-touch** (`rhs_flux` L324-355 gated block + design comment) — byte-identical (line numbers shifted -1 vs PR-D HEAD due to 1-line diff in helper end-of-function brace counts; the block itself is unchanged).
- **PR-E lake first-touch** (`rhs_update` L169-207 gated block + design comment + 6-field subset table) — byte-identical.
- All 4 `shud_rhs_dump_point("f_update" / "f_loop_before_passvalue" / "f_loop" / "f_applyDY", ...)` tag literals — unchanged.
- All other `Model_Data::` methods (`rhs_apply` etc.) and the call sites that consume the 3 reduction helpers — unchanged.
- Active `#pragma omp` count in `MD_rhs_core.cpp` = **3** (L84 element / L195 lake / L349 river first-touch) — unchanged vs PR-E baseline.
- SHUD build files (`Makefile.am`, `configure.ac`, `Makefile`) and FP flag policy — unchanged.

## Why (strategy)

PR-K2 (Kahan injection, June 2026) was triggered after server PR-H showed `heihe |Δ_nst|=225`, `heihe_x4 |Δ_nst|=3`, A3a FAIL both cases. Neumaier compensation in the 3 reduction helpers masked the |Δ_nst| failure modes for the §4.7 SHALL gate. After PR-C/D/E added NUMA steady-state first-touch loops (a different layer of intervention targeting memory locality rather than FP order), the question became: is Kahan still necessary, or do the first-touch loops (+ existing B0 traversal ordering) alone provide enough determinism for the three SHALL gate?

PR-F established the server intermediate measurement WITH Kahan still in. PR-G removes Kahan so PR-H can produce the final 8-cell verdict. Two outcomes:

- **PR-H PASS without Kahan** → codebase simplifies (no Kahan dependency to maintain); the first-touch + ordering combo is sufficient.
- **PR-H FAIL without Kahan** → revert PR-G (restore Kahan), document the floor that first-touch alone provides, and ship the Kahan-in regime.

## SHA matrix (keliya 90-day, N=1, 3 configs)

Comparison anchors:

- **PR-E baseline** (`6aada88`, has BOTH Kahan + PR-C/D/E first-touch) — what we are diffing from. Documented in `docs/p1d/p1d_first_touch_design.md` L400-402.
- **Pre-PR-K2 baseline** (`de9545d`, has NEITHER Kahan nor first-touch — Kahan was introduced in `3a0004c`, first-touch in `a2085de` / `7023ee9` / `6aada88`) — the "anchor" reference: if `post-PR-G == de9545d` then Kahan and first-touch are BOTH no-ops in this case/config combo (first-touch was independently proven no-op by PR-C/D/E OQ1; Kahan is what we are testing here).

3 × 3 cross matrix:

| Config | PR-E baseline (`6aada88`, has Kahan) | Pre-PR-K2 (`de9545d`, no Kahan) | post-PR-G (no Kahan) | Verdict |
| --- | --- | --- | --- | --- |
| `./shud keliya` (serial, OMP env unset) | `afceb9222aa4d8f0bc083a30db1100f0567040567d8cea936edba94dbe24c757` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | `post-PR-G == de9545d` ✓ ; `post-PR-G != PR-E baseline` ✓ (Kahan WAS a math change even at N=1 — branch elimination in helper changes FP sequence; revert correctly restored pre-K2 result) |
| `shud_omp` @ `NUM_OPENMP=1`, OMP_PROC_BIND unset (first-touch gate skipped) | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` | `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` | `post-PR-G == de9545d` ✓ ; `post-PR-G != PR-E baseline` ✓ |
| `shud_omp` @ `NUM_OPENMP=1`, `OMP_PROC_BIND=close OMP_PLACES=cores` (first-touch gate active) | `f7e9aad66488ee7e872c4d0b1de462f73944e5aeb98eb9427ddcf6a9805c183e` | `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` | `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` | `post-PR-G == de9545d` ✓ ; `post-PR-G != PR-E baseline` ✓ |

**Interpretation**

- `post-PR-G == de9545d` across all 3 configs proves the revert is mathematically clean: removing Kahan returns the helper outputs to the pre-K2 algebraic sum identity, and the PR-C/D/E first-touch loops contribute zero observable change (independently established in `docs/p1d/p1d_first_touch_design.md` OQ1 bitwise sections).
- `post-PR-G != PR-E baseline` across all 3 configs is the **intended** signal of PR-G: Kahan WAS providing a numerical effect (even in the serial / N=1 path the helper's pair-join `(lo - t) + hi` adds a low-order term that perturbs CVODE downstream and ripples through `nst` / `nfe` adaptive stepping). The keliya 90-day case shows pre-K2 `nst=101188` vs PR-E baseline `nst=97217` vs post-PR-G `nst=101188` — confirming `nst` returns to the pre-K2 trajectory.
- The deterministic-vs-serial gate behaviour (omp@N=1 unset vs close producing identical SHA) is preserved: the first-touch loop fires under `PROC_BIND=close` but writes pure zeros to fields that are re-written immediately by the original serial code path, so the field set seen by CVODE is bitwise-identical regardless of gate state.

## Pragma + flag invariants (post-revert)

- `#pragma omp` active count in `MD_rhs_core.cpp` = **3** (unchanged vs PR-E)
- Negative grep gate in `SHUD/src/`:
  - `schedule\((dynamic|guided)\)` = **0 hits** ✓
  - `#pragma omp atomic` = **0 hits** ✓
  - `SHUD_USE_DETERMINISTIC_REDUCTION|SHUD_DET_REDUCT|SHUD_PAIRWISE` = **0 hits** ✓
- FP strict 3-grep gate in `SHUD/Makefile`: `-ffp-contract=off` ≥ 1, `-fno-fast-math` ≥ 1, `-fopenmp` ≥ 1; `-ffast-math` / `-Ofast` only present in `DISALLOWED_FLAGS` guard list (not in active `CXX_BASE_FLAGS`).
- `cd SHUD && make clean && make shud_omp` PASS; `make shud` PASS.

## Next step

PR-H runs the final 8-cell server matrix (`heihe` + `heihe_x4` × N ∈ {1, 2, 4, 8}) on PR-G HEAD and produces the three SHALL gate verdict (|Δ_nst| / |Δ_nfe| / max_abs_err per §4.7). The outcome determines whether the codebase ships in the simpler Kahan-out regime or restores Kahan via a future re-injection PR.

---

**Outer fixture**: `feat/issue-281-p1d-pr-g` (parent `cb90767` = baseline/P1d post-PR-F)
**SHUD fixture**: `openmp-baseline` working tree (revert pending submodule bump after orchestrator commit)
**Local validation cell**: Mac (Apple Silicon), keliya 90-day, N=1 only (per CLAUDE.md §1.1.1: quantitative gates live on server; Mac is dev-only reference)
