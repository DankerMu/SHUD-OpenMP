# review-security-perf — PR #357 round 1

Head SHA: 1e45e16 / SHUD 5276167
Files reviewed: SHUD/src/Equations/MD_precond_identity.{h,cpp}, SHUD/src/Equations/cvode_config.cpp

## Security checklist results

1. **Hardcoded credentials**: None. Only literal `50` (LSetupFrequency) and `1.0` (N_VScale factor). Comments contain only spec/PR refs.
2. **Memory safety**:
   - `PSetupIdentity`: reads `jok` (input by value), writes `*jcurPtr` (single byte). No buffer access, no allocation, no uninit. `jcurPtr` is a SUNDIALS contract output — CVLS guarantees non-null per cvode_ls.h:57.
   - `PSolveIdentity`: delegates to `N_VScale_Serial`. Verified at `SHUD/cvode-6.0.0/src/nvector/serial/nvector_serial.c:524-535` — handles `z == x` aliasing explicitly via `VScaleBy_Serial` branch.
3. **N_Vector aliasing**: Safe. SUNDIALS contract permits `r == z`; serial impl branches on it.
4. **Timer RAII + exception safety**: `tools/profile/timer.h:43-62` — destructor calls `add_elapsed`; copy/move deleted. Even if `*jcurPtr = …` threw (impossible — POD write), Timer would still flush. Correct RAII.
5. **Thread safety**: Timer hot path is `std::atomic<int64_t>` add (timer.cpp:65). CVLS preconditioner callbacks are invoked serially by CVODE; bucket accumulation across parallel SHUD threads would still be safe.

## Performance checklist results

1. **PSetupIdentity overhead**: measured `t_precond_setup = 37.944 μs` over 1232 npe calls = ~30 ns/call. Ratio vs wall (117.36s) = **3.23e-7** — 5 orders of magnitude under the 5% soft-gate-6 threshold. Confirms spike headroom for real preconditioners.
2. **PSolveIdentity per-call cost**: `N_VScale(1.0, r, z)` is O(N), one memcpy-equivalent loop. Called 209,227× during 90-day keliya run; cost absorbed in `t_CVODE_internal`. Identity is the minimum non-trivial baseline.
3. **PREC_LEFT vs PREC_NONE wall delta**: out of scope for keliya smoke; gate-4 anchor at heihe/heihe_x4 in PR-E/PR-F.
4. **LSetupFrequency(50)**: `grep -rn` shows exactly ONE `CVodeSetLSetupFrequency` call (cvode_config.cpp:283). No double-config.
5. **No new O(N²)**: both callbacks are O(1) and O(N). No loops introduced beyond N_VScale.

## Findings

- None.

## Non-blocking notes

- Makefile wiring: `SRC_DIR/Equations/*.cpp` glob (Makefile:383) auto-picks `MD_precond_identity.cpp`; no Makefile edit needed.
- `t_precond_setup` correctly NOT in `kKnownRawOrCanonical[]` (timer.cpp:187-190), so it auto-surfaces under `extras:` per timer.cpp:204 catch-all — matches spec p8precond-zero-identity-spike L108-113.
- Canonical Precond convention (`jok ? SUNFALSE : SUNTRUE`) matches SUNDIALS `cvDiurnal_kry.c` exemplar; CVodeGetNumPrecEvals counter behaves as expected (npe=1232 confirms increment fires).

## Output contract

Reviewer agent: review-security-perf
Review round: round 1
Reviewed head SHA: 1e45e16
Summary: Identity preconditioner stub is memory-safe, aliasing-safe, RAII-correct, and overhead-trivial (3.23e-7 of wall vs 5% gate); spike feasibility validated.
Findings:
- None.
Non-blocking notes:
- Makefile auto-glob covers new file; no Makefile edit needed.
- `t_precond_setup` auto-surfaces under `extras:` via timer.cpp catch-all (correct by design).
- Canonical jcurPtr convention matches SUNDIALS cvDiurnal_kry exemplar; npe=1232 confirms counter wired.
