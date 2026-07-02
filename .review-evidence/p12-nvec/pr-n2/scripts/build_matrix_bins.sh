#!/bin/bash
# P12-nvec PR-N2 — build the 3 matrix binaries at SHUD d8f736c (server login node, quick).
#   shud_omp_C       = Config C  : make shud_omp                                   (StrictOMP RHS + Serial NVector)
#   shud_omp_E       = Config E  : make shud_omp SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1
#   shud_omp_E_prof  = Config E + SHUD_ENABLE_PROFILE=1  (the ONE gate-on profiler leg, N=16)
# HYPRE/OPENBLAS/MPI link paths per Makefile L460-461 + PR-N1 gcc_spot recipe.
set -euo pipefail
REPO=/scratch/frd_muziyao/SHUD-OpenMP
SHUD=$REPO/SHUD
BINDIR=$REPO/.p12-nvec-runs/matrix/bin
mkdir -p "$BINDIR"

export HYPRE_INCDIR=/scratch/frd_muziyao/local/hypre-3.1.0/include
export HYPRE_LIBDIR=/scratch/frd_muziyao/local/hypre-3.1.0/lib
export MPI_INCDIR=/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu/openmpi/include
export OPENBLAS_LIBDIR=/scratch/frd_muziyao/local/openblas/lib
# link-time library search (Hypre needs openblas + mpi; sundials in InstallSundials)
export LIBRARY_PATH=/scratch/frd_muziyao/local/openblas/lib:/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu/openmpi/lib:/scratch/frd_muziyao/local/usr/lib/x86_64-linux-gnu:/scratch/frd_muziyao/local/hypre-3.1.0/lib:$SHUD/InstallSundials/lib

cd "$SHUD"
echo "=== build context ==="
echo "host=$(hostname)  gcc=$(g++ --version | head -1)"
echo "SHUD HEAD=$(git rev-parse HEAD)  (expect d8f736c...)"
[ "$(git rev-parse HEAD)" = "d8f736c53d5cc9c8a22a86f98ae53858309c0a17" ] && echo "SHUD @ d8f736c CONFIRMED" || { echo "FATAL: SHUD not at d8f736c"; exit 3; }

build_one() {
  local name="$1"; shift
  echo ""
  echo "=== building $name : make shud_omp $* ==="
  make clean >/dev/null 2>&1 || true
  make shud_omp "$@" \
      HYPRE_INCDIR="$HYPRE_INCDIR" HYPRE_LIBDIR="$HYPRE_LIBDIR" \
      MPI_INCDIR="$MPI_INCDIR" OPENBLAS_LIBDIR="$OPENBLAS_LIBDIR" \
      2>&1 | tail -4
  cp -f "$SHUD/shud_omp" "$BINDIR/$name"
  echo "  -> $BINDIR/$name  ($(stat -c%s "$BINDIR/$name") bytes)"
}

build_one shud_omp_C
build_one shud_omp_E      SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1
build_one shud_omp_E_prof SHUD_USE_OPENMP_NVECTOR=1 SHUD_NVEC_HYBRID=1 SHUD_ENABLE_PROFILE=1

echo ""
echo "=== built binaries ==="
ls -la "$BINDIR"
echo "=== sanity: symbol markers ==="
for b in shud_omp_C shud_omp_E shud_omp_E_prof; do
  printf "%-18s hybrid_install=%s nvec_prof=%s\n" "$b" \
    "$(nm -C "$BINDIR/$b" 2>/dev/null | grep -c nvec_hybrid_install || true)" \
    "$(nm -C "$BINDIR/$b" 2>/dev/null | grep -c nvec_prof_install || true)"
done
echo "BUILD_DONE"
