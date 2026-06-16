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
- Current submodule HEAD: `a9327b146c85b1f3a95433ab77a58d28831846a7` (PR #18 round 3: SHUD_BUILD_CFLAGS / CXX_BASE_FLAGS bypass sealed via `override :=` + 2-layer scan; updated at each SHUD-touching PR-merge and at B0-tag time)
- Verify locally: `git -C SHUD rev-parse HEAD`

## Disallowed flags (Makefile guard)
- `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`
- Build fails at `make` parse time if any disallowed flag appears in ANY of the 8 user-controllable carriers: `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, `LDFLAGS`, `MAKEOVERRIDES`, `MAKEFLAGS`, `SHUD_BUILD_CFLAGS`, `CXX_BASE_FLAGS`.
- Two layers protect the lock:
  1. **`override … :=` on `CXX_BASE_FLAGS` + `SHUD_BUILD_CFLAGS`** — per GNU make manual, the `override` directive on `:=` causes make-CLI assignments to be silently ignored. So even if the disallowed-flag scan is reverted, the recipe still uses the locked flag set.
  2. **Two-layer disallowed-flag guard:**
     - Layer 1 (`filter`, word-level): scans the 6 standard carriers + the 2 project-local lock variables. Catches `make shud CXXFLAGS=-Ofast` etc.
     - Layer 2 (`findstring` on `$(MAKEOVERRIDES)`): catches CLI assignments to the lock variables (`make shud SHUD_BUILD_CFLAGS=-Ofast` / `CXX_BASE_FLAGS=-Ofast`). Without this layer, the `override` directive would silently drop the injection, leaving the user mistakenly thinking their flag was accepted. Layer 2 escalates to a loud `$(error …)` with `MAKEOVERRIDES` contents echoed.
- Recipes invoke `$(SHUD_BUILD_CFLAGS)` (alias of `CXX_BASE_FLAGS`) directly, so a user-supplied `make CFLAGS=…` cannot clobber the locked flag set — it is captured by the scan and fails the build before any compile.
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

## CHANGELOG
- `fea5922` (PR #16 / issue #3): initial B0 build-environment lockdown — locked flag set, SUNDIALS major-version guard, idempotent `./configure`, macOS libomp discovery via `brew --prefix`.
- PR #18 round 2 (`c9368fd` in SHUD): invariant-closure round 2 — sealed `CFLAGS` / `CPPFLAGS` / `LDFLAGS` bypass paths in the disallowed-flag scan; pinned `CXX` via `$(origin CXX)` so `c++` no longer wins by default; tightened SUNDIALS guard with anchored regex on MAJOR + new MINOR check + `libsundials_cvode.*` / `libsundials_nvecserial.*` / (for omp) `libsundials_nvecopenmp.*` stat; added `check_sundials_omp` for the OpenMP target; added macOS libomp `$(error)` on `shud_omp`; cleaned bare-`-L` token via `$(if …)`; configure always re-extracts `cvode-6.0.0/`; documented `SUNDIALS_DIR` override discipline.
- PR #18 round 3 (`a9327b1` in SHUD): invariant-closure round 3 — sealed `SHUD_BUILD_CFLAGS` / `CXX_BASE_FLAGS` bypass paths surfaced by round-2 verifier. Two-tier protection: (1) `override … :=` on both lock variables (silently ignores make-CLI override per GNU make semantics); (2) two-layer disallowed-flag guard — Layer 1 `filter` extended to include the 2 lock variables (defense-in-depth), Layer 2 new `findstring` scan on `$(MAKEOVERRIDES)` catches CLI assignments to the lock variables and escalates `override`'s silent drop into a loud `$(error)`. Manifest §"Disallowed flags" updated to enumerate all 8 carriers.
