# B0 Build Manifest

> Authoritative source of truth for reproducing the B0 baseline binary.
> Any change to flags, SUNDIALS version, or compiler requires an OpenSpec change
> against `openspec/changes/<change>/specs/build-environment-lockdown/spec.md`.

## 1. Linux config (target deployment platform — go/no-go authority)
- Compiler: GCC 12, invoked as `g++-12` or `CXX=g++-12 make shud`
- Base flags: `-O2 -g -ffp-contract=off -fno-fast-math -std=c++14` (from `CXX_BASE_FLAGS`)
- OpenMP compile flag: `-fopenmp` (Linux UNAME_S branch in Makefile)
- OpenMP link flag: `-lgomp`
- SUNDIALS: 6.0.0 installed under `SHUD/InstallSundials/` via `./configure`
- SUNDIALS link: `-lsundials_cvode -lsundials_nvecserial` (serial) + `-lsundials_nvecopenmp` (omp)
- Verification command: `make shud && make shud_omp` after `./configure`

## 2. macOS config (development, Apple Silicon)
- Compiler: Apple Clang (resolved via PATH `g++` → wrapper; on this host: Apple clang 17.0.0)
- libomp: `brew install libomp`; prefix auto-detected via `$(brew --prefix libomp)` (`/opt/homebrew/opt/libomp` on Apple Silicon)
- Base flags: same as Linux
- OpenMP compile flag: `-Xpreprocessor -fopenmp`
- OpenMP link flag: `-L$(brew --prefix libomp)/lib -lomp`
- SUNDIALS: same install path
- Cross-platform note: macOS numbers are development-only; the §1.1.1 quantitative go/no-go is decided on Linux only (see master plan §1.1.1).

## 3. OpenMP runtime env (both platforms)
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`
- `OMP_NUM_THREADS` set per benchmark `manifest.yaml`
- `NumEle < OMP_CUTOFF` triggers serial fallback (per master plan §C8)

## 4. SHUD submodule pin
- Upstream: `https://github.com/SHUD-System/SHUD.git`
- Working branch on upstream: `openmp-baseline` (long-lived, derived from `3aec657`; not master)
- Initial B0 commit: `3aec657` (master plan §S0.10)
- Current submodule HEAD: `78c37a1061de4112bc7c297bb7bd1f107432e6f2` (S0-10 / #14 timer instrumentation @ PROFILE=0/DUMP=0; updated at each SHUD-touching PR-merge and at B0-tag time)
- Verify locally: `git -C SHUD rev-parse HEAD`

## B0-tag (S0-13 / #17)

The `B0-tag` lightweight git tag pins the exact `(outer, SHUD submodule)`
commit pair that the A0 acceptance gate certified. B1a regression checks
diff against `git show B0-tag:benchmarks/<case>/B0_output/` byte-for-byte.

- **Outer repo tag**: `B0-tag` on `baseline/current` at the commit that
  merged S0-13 PR. Verify: `git rev-parse B0-tag`.
- **SHUD submodule pin at B0-tag**: `78c37a1061de4112bc7c297bb7bd1f107432e6f2`
  (the submodule pointer captured by the outer tag commit). Verify:
  `git show B0-tag --stat -- SHUD`.
- **Date**: 2026-06-17
- **Authority for "what's in B0"**: `docs/status_matrix.md` B0 row
  (5 PASS local + heihe PASS @ server + heihe_x4 PASS @ server +
  kashigeer N/A deferred-upstream); A0 Acceptance Checklist 9/9 PASS
  per S0-13 spec amendment.
- **Manifest digests** (SHA256 of each `benchmarks/<case>/manifest.yaml`
  at B0-tag): see `benchmarks/INDEX.md` row evidence + `git show B0-tag`
  for the recursive tree of B0 archive files.
- **Branch protection**: `baseline/current` has the
  `.github/workflows/serial-baseline.yml` `build-and-compare` required
  check enabled at B0-tag time. The `skip-baseline-ci` label-bypass
  privilege is removed at B0-tag merge so subsequent S1+ PRs cannot
  silently land without baseline verification.

## Disallowed flags (Makefile guard)
- `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`
- Binary correctness is contractually guaranteed in all forms of injection (CLI, env, MAKEFLAGS, etc.). The user-facing loud-error UX is provided for CLI form only; env-form injection on the two project-local lock variables is silent (binary safe, but no `$(error …)` emitted — see Layer 3 caveat below).
- Three layers protect the lock:
  1. **Layer 1 — `filter` over 8 carriers (loud error):** scans for any disallowed flag inside the 6 standard carriers (`CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, `MAKEOVERRIDES`, `MAKEFLAGS`) + the 2 project-local lock variables (`SHUD_BUILD_CFLAGS`, `CXX_BASE_FLAGS`). Word-level `filter` catches `make shud CXXFLAGS=-Ofast`, `make shud CFLAGS=-Ofast`, etc.
  2. **Layer 2 — anchored `=value` scan on `$(MAKEOVERRIDES)` (loud error):** iterates over the `VAR=value` tokens of `$(MAKEOVERRIDES)` and uses `filter %=<flag>` to match exact `VAR=<disallowed-flag>` CLI assignments. Catches `make shud SHUD_BUILD_CFLAGS=-Ofast` / `CXX_BASE_FLAGS=-Ofast`, which the `override :=` directive on the lock variables would otherwise silently drop. The anchored form (vs the earlier literal `findstring`) avoids false-positives on legitimate values that contain a disallowed flag as a substring (e.g. `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`). `DISALLOWED_FLAGS` itself is `override`-protected so `make DISALLOWED_FLAGS=` cannot disarm the scan list.
  3. **Layer 3 — `override … :=` on `CXX_BASE_FLAGS` + `SHUD_BUILD_CFLAGS` (silent guarantee):** per GNU make manual, the `override` directive on `:=` causes make-CLI assignments to be silently ignored. Recipes invoke `$(SHUD_BUILD_CFLAGS)` (alias of `CXX_BASE_FLAGS`) directly, so the locked flag set is what the compiler actually sees. **Caveat — env-form injection is silent:** `SHUD_BUILD_CFLAGS=-Ofast make shud` is silently ignored (binary safe — recipe still uses locked flags — but no error emitted, because Layer 1/2 only see CLI-form assignments via `MAKEOVERRIDES`). This is intentional: binary correctness is what we contractually guarantee; loud-error UX is provided for the most common injection vector (CLI form). A future deeper fix (Layer 3 env-var origin probe via `$(origin VAR)` checks) is tracked as a follow-up.
- Recipes invoke `$(SHUD_BUILD_CFLAGS)` (alias of `CXX_BASE_FLAGS`) directly, so a user-supplied `make CFLAGS=…` cannot clobber the locked flag set — it is captured by Layer 1 and fails the build before any compile.
- Compiler default is pinned via `$(origin CXX)` check (legacy `CXX ?= g++` was a no-op vs GNU make's built-in `c++`); env / CLI `CXX=...` is still honored.

## SUNDIALS version + install-completeness guard
- `check_sundials` Makefile target enforces (anchored regex; no substring matches):
  - `^#define SUNDIALS_VERSION_MAJOR 6$` in `SHUD/InstallSundials/include/sundials/sundials_config.h`
  - `^#define SUNDIALS_VERSION_MINOR 0$` (B0 requires 6.0.x; 6.1+ rejected; PATCH unenforced)
  - `libsundials_cvode.*` present under `$(SUNDIALS_DIR)/lib/`
  - `libsundials_nvecserial.*` present under `$(SUNDIALS_DIR)/lib/`
- `check_sundials_omp` additionally requires `libsundials_nvecopenmp.*`; `shud_omp` depends on it (`shud` depends on `check_sundials` only).
- `./configure` mirrors the same MAJOR + MINOR + lib check both in the idempotent short-circuit and in the post-install report.
- `./configure` always re-extracts `cvode-6.0.0/` (deletes and untar) to avoid stale / partial trees from interrupted prior runs poisoning cmake.

## macOS libomp guard
- `make shud_omp` checks `LIBOMP_PREFIX` (from `brew --prefix libomp`); if empty, errors with the install instruction. Only gated on `shud_omp` so `make shud` (serial) still works without libomp.
- Linux: empty `LIB_OMP` is wrapped in `$(if …)` so no bare `-L` token appears on the link line.

## SUNDIALS_DIR override discipline
- For B0 reproducibility, `SUNDIALS_DIR` MUST point at `SHUD/InstallSundials/` (the bundled install).
- Override is technically possible (`make shud SUNDIALS_DIR=/external/path`), but the resulting binary is NOT B0-comparable and MUST NOT be used for the benchmark archive (#8), CI (#13), or A0 acceptance (#17). The `check_sundials` target only verifies the targeted SUNDIALS' MAJOR + MINOR version and presence of the required libs — not its compile flags or build provenance. A system-installed SUNDIALS built with `-Ofast` will pass the guard but break A3a bitwise.
- If a host has external SUNDIALS 6.0.x satisfying the locked flag set, it MAY be used; record the override and rationale in the PR description.

## Installed SUNDIALS (current host)
- Version: `6.0.0`
- Install size: `26M`
- Path: `SHUD/InstallSundials/`

## CHANGELOG (S0-13 amendment)

- S0-13 / #17: kashigeer reclassified from `local-and-server` to
  `deferred-upstream` in `benchmarks/INDEX.md`; `status-matrix` +
  `rhs-profile-gate` specs amended so deferred-upstream cells are
  N/A-not-blocking; `B0-tag` section added above; `docs/profile_decision.md`
  signed by DankerMu against outer `a860eae5` + SHUD `78c37a1` via
  delegated grant 2026-06-17.

## Prior CHANGELOG
- `fea5922` (PR #16 / issue #3): initial B0 build-environment lockdown — locked flag set, SUNDIALS major-version guard, idempotent `./configure`, macOS libomp discovery via `brew --prefix`.
- PR #18 round 2 (`c9368fd` in SHUD): invariant-closure round 2 — sealed `CFLAGS` / `CPPFLAGS` / `LDFLAGS` bypass paths in the disallowed-flag scan; pinned `CXX` via `$(origin CXX)` so `c++` no longer wins by default; tightened SUNDIALS guard with anchored regex on MAJOR + new MINOR check + `libsundials_cvode.*` / `libsundials_nvecserial.*` / (for omp) `libsundials_nvecopenmp.*` stat; added `check_sundials_omp` for the OpenMP target; added macOS libomp `$(error)` on `shud_omp`; cleaned bare-`-L` token via `$(if …)`; configure always re-extracts `cvode-6.0.0/`; documented `SUNDIALS_DIR` override discipline.
- PR #18 round 3 (`a9327b1` in SHUD): invariant-closure round 3 — sealed `SHUD_BUILD_CFLAGS` / `CXX_BASE_FLAGS` bypass paths surfaced by round-2 verifier. Two-tier protection: (1) `override … :=` on both lock variables (silently ignores make-CLI override per GNU make semantics); (2) two-layer disallowed-flag guard — Layer 1 `filter` extended to include the 2 lock variables (defense-in-depth), Layer 2 new `findstring` scan on `$(MAKEOVERRIDES)` catches CLI assignments to the lock variables and escalates `override`'s silent drop into a loud `$(error)`. Manifest §"Disallowed flags" updated to enumerate all 8 carriers.
- PR #18 round 5 — W-R4 Warning-closure: `override`-protected `DISALLOWED_FLAGS` so `make DISALLOWED_FLAGS=` cannot disarm the scan; replaced Layer 2 literal `findstring` trio with anchored `=value` iteration over `$(DISALLOWED_FLAGS)` (`filter %=<flag>` on `$(MAKEOVERRIDES)` tokens) to avoid false-positive on paths containing `-Ofast` as substring (e.g. `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`); corrected manifest §"Disallowed flags" to honestly state env-form lock-var injection is silent (binary safe via `override :=`, no `$(error)` emitted) and reframed the 3-layer protection model accordingly.
