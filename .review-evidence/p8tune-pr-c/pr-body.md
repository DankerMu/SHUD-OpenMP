## Summary

- **SHUD source change**: Adds `SHUD_SPGMR_MAXL` env-var hook in `SHUD/src/Equations/cvode_config.cpp` (helper `get_spgmr_maxl_from_env()` + modified `SUNLinSol_SPGMR` call-site + provenance log).
- **Single-file source surface**: `src/Equations/cvode_config.cpp` only; no Makefile / no SUNDIALS-vendored / no header / no sibling source change. SHUD submodule commit `6ce17d6` on `openmp-baseline` branch (NEVER master per project rule).
- **4 safety constraints** per spec scenarios L11-35:
  - unset / "" / "0" → return 0 (silent default; bit-identical to current SHUD 37be0fe production behavior)
  - "5" / "10" / "15" / "20" / "30" → parse + emit `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` log + return parsed value
  - any other value (e.g. "7", "50", "foo", "-1", "+5", "05") → stderr error + `myexit(ERRCVODE)` BEFORE any SPGMR allocation (fail-fast)
- **G3 4-way bit-identical CI gate VERIFIED on cn14** (Slurm job 9626, SHUD 6ce17d6 build SHUD_ENABLE_PROFILE=1):
  - All 4 invocations (`unset` / `""` / `"0"` / `"5"`) produce `rivqdown.dat` SHA12 = `1bfe6a30856e` matching PR-A `§keliya-smoke-anchor` exactly
  - All 15 canonical cvode_stats keys bit-identical to PR-A anchor (nfe=112248, nfeLS=116421, nni=112247, nli=116421, nsetups=0, netf=5, nst=110917, npe=0, nps=0, ncfn=205, ncfl=42, lenrw=23294, leniw=53, lenrwLS=21474, leniwLS=42)
  - Cross-run `cmp` byte-equivalence: run1 == run2 == run3 == run4
  - Stdout provenance log discipline: 0/0/0/1 lines per IM D15 L235-238 (only "5" emits the log)
- **PREC_NONE preserved**: `grep -nE 'PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity' src/Equations/cvode_config.cpp` returns 0 matches; ADR-0003 PREC_NONE production decision unchanged.
- **Mac build**: `make shud` + `make shud_omp` both exit 0.
- **Implementer-side parser unit tests**: 19/19 PASS including strict-whitelist edge cases (`+5`, `05`, ` 5`, `5 `, `5x`, `10.0`, `-1`, `foo`, `7`, `50`, overflow).

## Why

PR-A established the cleaned-PREC_NONE baseline + keliya smoke anchor. PR-B verdict gated full 60-cell sweep mode. PR-C now provides the runtime knob (env var `SHUD_SPGMR_MAXL`) that PR-D needs to iterate over `maxl ∈ {5, 10, 15, 20, 30}` per cell, while guaranteeing default-unset bit-identical equivalence to SHUD 37be0fe (so 18-cell baseline reuse from PR-A remains valid).

The 4-way default-equivalence (`unset` / `""` / `"0"` / `"5"` all bit-identical) honors SUNDIALS docs (`maxl ≤ 0` → default 5) and the IM D15 invariant: default-unset MUST produce bit-identical CVODE output to production baseline; any opt-in MUST honor SUNDIALS semantics + preserve PREC_NONE; any invalid value MUST fail-fast via `myexit(ERRCVODE)`.

OpenSpec change: `p8tune-spgmr-maxl` (capability `spgmr-maxl-env-hook`).

## Scope

Outer repo PR (2 files):
- `SHUD` submodule pointer: `37be0fe` → `6ce17d6` (1 line)

SHUD submodule commit `6ce17d6` (1 file):
- `SHUD/src/Equations/cvode_config.cpp` (+69 / -4 lines)
  - `<errno.h>` include
  - `static int get_spgmr_maxl_from_env(void)` helper (strict whitelist parser + fail-fast abort + provenance log)
  - call-site at L324: `SUNLinSol_SPGMR(udata, PREC_NONE, get_spgmr_maxl_from_env(), sunctx)`

No Makefile / header / SUNDIALS-vendored / sibling source change. No CVODE tolerance / step controller / preconditioner change. No B1a/B1b baseline modification.

Local-only (gitignored, not in PR diff):
- `openspec/changes/p8tune-spgmr-maxl/` (D15 Invariant Matrix authored)
- `.review-evidence/p8tune-pr-c/{g3_verdict.md, g3_4way_evidence.tar}`

## Test plan

- [x] `openspec validate p8tune-spgmr-maxl --strict` → PASS
- [x] Mac `make shud` + `make shud_omp` → both exit 0
- [x] Mac `grep -nE 'PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity' SHUD/src/Equations/cvode_config.cpp` → 0 matches (no regression)
- [x] Mac `cd SHUD && git diff --name-only HEAD~1..HEAD` → `src/Equations/cvode_config.cpp` (single-file surface)
- [x] Implementer-side parser unit tests (19/19 PASS) — strict whitelist + leading-zero rejector + char-by-char `[0-9]` pre-strtol check
- [x] cn14 server build with `SHUD_ENABLE_PROFILE=1` → exit 0
- [x] cn14 Slurm job 9626 (CPU partition, cn14, /scratch/.p8tune-runs/pr-c-g3-gate/) — Slurm 三铁律 compliance
- [x] G3 4-way bit-identical gate: all 4 invocations rivqdown.dat SHA12 = `1bfe6a30856e` matches PR-A anchor
- [x] G3 4-way 15-key cvode_stats bit-identical to PR-A anchor (all 15 keys match)
- [x] G3 cross-run cmp byte-equivalence: run1 == run2 == run3 == run4
- [x] G3 stdout provenance log discipline: 0/0/0/1 lines per IM D15 L235-238

## Agent Review

(Populated by Phase 8.)

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Closes #366.
