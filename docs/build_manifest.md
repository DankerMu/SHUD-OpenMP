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
- Current submodule HEAD: `fea5922752bc443fdad441541db50a44d1903bfa` (build env lockdown commit; updated at each SHUD-touching PR-merge and at B0-tag time)
- Verify locally: `git -C SHUD rev-parse HEAD`

## Disallowed flags (Makefile guard)
- `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`
- Build fails at `make` parse time if any appear in `CXXFLAGS` or `MAKEOVERRIDES`.

## SUNDIALS major-version guard
- `check_sundials` Makefile target greps `define SUNDIALS_VERSION_MAJOR 6` in `SHUD/InstallSundials/include/sundials/sundials_config.h`
- `shud` / `shud_omp` targets depend on `check_sundials`; any other major fails the build.

## Installed SUNDIALS (current host)
- Version: `6.0.0`
- Install size: `26M`
- Path: `SHUD/InstallSundials/`
