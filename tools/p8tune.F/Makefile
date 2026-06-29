# tools/p8tune.F/Makefile
#
# P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-A). Builds the
# AMG spike binary:
#   - boomeramg_setup_solve    (new — wraps Hypre IJMatrix + BoomerAMG
#                                Setup/Solve around numeric J binary
#                                produced by P8-tune.D fd_color_jacobian)
#
# Plus 2 symlinks to existing P8-tune.D binaries (REQ-2 shell-out reuse,
# p8tune.D source frozen):
#   - dump_adjacency           → ../p8tune.D/dump_adjacency
#   - fd_color_jacobian        → ../p8tune.D/fd_color_jacobian
#
# Link dependencies for boomeramg_setup_solve:
#   - $(SHUD_LIBSHUD)          (transitive — Model_Data + rhs_core wiring
#                                inherited from p8tune.D pattern even though
#                                this binary doesn't init Model_Data itself;
#                                spec REQ-1 mandates libshud.a archive link)
#   - Hypre (libHYPRE)         BoomerAMG/IJMatrix/IJVector + MPI runtime
#   - open-mpi (libmpi)        Hypre transitively links MPI; we add explicit
#                                -lmpi for MPI_Init/MPI_Finalize calls
#   - openblas (libopenblas)   Hypre dylib depends on it for dense ops
#   - SuiteSparse + ColPack + SUNDIALS — pulled in by libshud.a's transitive
#                                refs (kept identical to p8tune.D link line
#                                so future SHUD-link symbol additions don't
#                                break this build)
#   - libomp (Mac) / libgomp (Linux) — OpenMP runtime
#
# Hypre version note: the spec design doc references Hypre 2.30.0;
# brew currently ships 3.1.0 (HYPRE_RELEASE_NUMBER 30100, HYPRE_BIGINT=1).
# The BoomerAMG API surface used here (Create/Setup/Solve/SetInterpType/
# SetCoarsenType/GetNumIterations/GetCumNnzAP/GetFinalRelativeResidualNorm)
# is stable across 2.30 → 3.x; 3.x adds GPU APIs without breaking the CPU
# BoomerAMG contract. See README.md §1 for the version reconciliation note.

CXX ?= g++
# -O2 deliberate: P8-tune.F PR-0 #394 fixed the SHUD Model_Data dtor
# uninit-ptr UB (NSDMI nullptr defaults), eliminating the heap-corruption
# UB that forced P8-tune.D Makefile to fall back to -O1. -O2 is now safe
# for this tool (which doesn't even init Model_Data — it only reads the
# numeric J binary). If a -O2 build issue surfaces post-merge, fall back
# to -O1 with a documented note.
CXXFLAGS ?= -O2 -g -ffp-contract=off -fno-fast-math -std=c++14 -Wall

# --- platform-conditional libomp / libgomp ---
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  LIBOMP_PREFIX ?= /opt/homebrew/opt/libomp
  OMP_CFLAGS    := -Xpreprocessor -fopenmp -I$(LIBOMP_PREFIX)/include
  OMP_LFLAGS    := -L$(LIBOMP_PREFIX)/lib -lomp
else
  OMP_CFLAGS    :=
  OMP_LFLAGS    := -lgomp
endif

# --- Hypre paths (brew default on Mac; override on server via HYPRE_PREFIX) ---
HYPRE_PREFIX ?= /opt/homebrew/opt/hypre
HYPRE_INC    := $(HYPRE_PREFIX)/include
HYPRE_LIB    := $(HYPRE_PREFIX)/lib

# --- MPI paths (Hypre brew transitively links open-mpi) ---
MPI_PREFIX ?= /opt/homebrew/opt/open-mpi
MPI_INC    := $(MPI_PREFIX)/include
MPI_LIB    := $(MPI_PREFIX)/lib

# --- openblas (Hypre dylib runtime dep) ---
OPENBLAS_PREFIX ?= /opt/homebrew/opt/openblas
OPENBLAS_LIB    := $(OPENBLAS_PREFIX)/lib

# --- SuiteSparse paths (transitive via libshud.a refs) ---
SUITESPARSE_INC ?= /opt/homebrew/include/suitesparse
SUITESPARSE_LIB ?= /opt/homebrew/lib

# --- ColPack paths (transitive via libshud.a refs) ---
COLPACK_INC ?= $(HOME)/.local/include/ColPack_headers
COLPACK_LIB ?= $(HOME)/.local/lib/shared_library
COLPACK_LIB_FALLBACK ?= $(HOME)/.local/lib/archive
# ColPack version probe: ColPack ships a cmake ConfigVersion file with
# the canonical PACKAGE_VERSION; no header macro exists. Parse the .cmake
# file at build time; fall back to "unknown" if not found. This eliminates
# the cell_summary `colpack_version=unknown` sentinel for any environment
# that has a standard ColPack install (PR-A Mac, PR-B server post-#401).
COLPACK_CMAKE_VERSION ?= $(HOME)/.local/lib/ColPack_libs/ColPackConfigVersion.cmake
# grep the `set(PACKAGE_VERSION "X.Y.Z")` line and extract the quoted
# value. Two-stage pipe (grep then cut) keeps the cmake `(` away from
# Make's $(shell) parser which would interpret an unescaped open-paren.
# Comment lines mentioning PACKAGE_VERSION in prose are filtered out
# by grep's leading-^set anchor.
COLPACK_VERSION := $(shell grep -E '^set.PACKAGE_VERSION' $(COLPACK_CMAKE_VERSION) 2>/dev/null | cut -d'"' -f2)
ifeq ($(COLPACK_VERSION),)
  COLPACK_VERSION := unknown
endif

# --- SHUD paths ---
SHUD_DIR ?= ../../SHUD
SHUD_SRC ?= $(SHUD_DIR)/src
SHUD_LIB ?= $(SHUD_DIR)
SHUD_LIBSHUD := $(SHUD_LIB)/libshud.a

# SHUD submodule pin (compile-time embed). Eliminates the
# cell_summary `shud_pin=runtime` sentinel — the PR-C aggregator strict
# parser will see the actual 40-char SHA. `git rev-parse HEAD` runs on the
# SHUD submodule HEAD at build time; falls back to "unknown" if not in a
# git checkout (e.g. tarball install).
SHUD_PIN_SHA := $(shell git -C $(SHUD_DIR) rev-parse HEAD 2>/dev/null)
ifeq ($(SHUD_PIN_SHA),)
  SHUD_PIN_SHA := unknown
endif

# Version-pin defines injected into the cpp build.
VERSION_DEFINES := \
  -DCOLPACK_VERSION_STR=\"$(COLPACK_VERSION)\" \
  -DSHUD_PIN_SHA=\"$(SHUD_PIN_SHA)\"

# --- SUNDIALS (resolved from SHUD's InstallSundials) ---
SUNDIALS_DIR ?= $(SHUD_DIR)/InstallSundials
SUNDIALS_INC := $(SUNDIALS_DIR)/include
SUNDIALS_LIB := $(SUNDIALS_DIR)/lib

INCLUDES := \
  -I$(HYPRE_INC) \
  -I$(MPI_INC) \
  -I$(SHUD_SRC)/Model \
  -I$(SHUD_SRC)/ModelData \
  -I$(SHUD_SRC)/classes \
  -I$(SHUD_SRC)/Equations \
  -I$(SUNDIALS_INC) \
  -I$(SUITESPARSE_INC) \
  -I$(COLPACK_INC) \
  -I.

LDFLAGS := \
  -L$(HYPRE_LIB) \
  -L$(MPI_LIB) \
  -L$(OPENBLAS_LIB) \
  -L$(SUITESPARSE_LIB) \
  -L$(COLPACK_LIB) \
  -L$(COLPACK_LIB_FALLBACK) \
  -L$(SUNDIALS_LIB)

# RPATH so dylibs resolve at runtime (mirror of p8tune.D pattern, extended
# with Hypre + MPI + openblas).
RPATH := -Wl,-rpath,$(HYPRE_LIB) \
         -Wl,-rpath,$(MPI_LIB) \
         -Wl,-rpath,$(OPENBLAS_LIB) \
         -Wl,-rpath,$(SUNDIALS_LIB) \
         -Wl,-rpath,$(SUITESPARSE_LIB) \
         -Wl,-rpath,$(COLPACK_LIB)

# Link line. Order matters for static libs: libshud.a is an archive; pull
# its symbols before SuiteSparse/SUNDIALS deps so unresolved Model_Data
# refs trigger transitive lookups. Hypre + MPI + openblas come after for
# their own resolution.
LIBS := \
  -lm \
  -lHYPRE \
  -lmpi \
  -lopenblas \
  -lsundials_cvode \
  -lsundials_nvecserial \
  -lklu -lamd -lbtf -lcolamd -lsuitesparseconfig \
  -lColPack

BINARIES := boomeramg_setup_solve
SYMLINKS := dump_adjacency fd_color_jacobian

# `dump_adjacency` / `fd_color_jacobian` are symlinks into ../p8tune.D/.
# Declare them .PHONY so Make re-creates the symlink whenever upstream
# changes (rather than treating the symlink-file existing as up-to-date
# and leaving a dangling link when the upstream binary is rebuilt or
# deleted by `make -C ../p8tune.D clean`). The recipes themselves are
# idempotent (`ln -sf`). Cross-review-round-1 finding C3.
.PHONY: shud_spike all clean dump_adjacency fd_color_jacobian

shud_spike all: $(BINARIES) $(SYMLINKS)

# boomeramg_setup_solve links against libshud.a (built by SHUD/Makefile
# `libshud.a` target which P8-tune.D PR-0 added). The top-level Makefile
# (../../Makefile) recurses into SHUD first; invoking `make` from this
# dir alone REQUIRES libshud.a to already exist.
boomeramg_setup_solve: boomeramg_setup_solve.cpp $(SHUD_LIBSHUD)
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(VERSION_DEFINES) $(INCLUDES) $< $(SHUD_LIBSHUD) $(LDFLAGS) $(LIBS) $(OMP_LFLAGS) $(RPATH) -o $@

# Symlink reuse of P8-tune.D binaries per REQ-2 shell-out convention:
# p8tune.D source is frozen, so we cannot refactor those tools to a library
# API; we shell out via subprocess. Symlinks under tools/p8tune.F/ keep
# the invocation path stable from this directory's perspective.
# Targets are .PHONY (see above) so the symlink is re-created on every
# `make` — guards against a deleted upstream binary leaving a dangling
# symlink that Make would otherwise consider up-to-date.
dump_adjacency:
	@if [ ! -x ../p8tune.D/dump_adjacency ]; then \
	  echo "tools/p8tune.F: ERROR: ../p8tune.D/dump_adjacency not built; run 'make -C ../p8tune.D all' first" >&2; \
	  exit 1; \
	fi
	ln -sf ../p8tune.D/dump_adjacency $@

fd_color_jacobian:
	@if [ ! -x ../p8tune.D/fd_color_jacobian ]; then \
	  echo "tools/p8tune.F: ERROR: ../p8tune.D/fd_color_jacobian not built; run 'make -C ../p8tune.D all' first" >&2; \
	  exit 1; \
	fi
	ln -sf ../p8tune.D/fd_color_jacobian $@

clean:
	@rm -f $(BINARIES) $(SYMLINKS)
	@echo "tools/p8tune.F: cleaned $(BINARIES) $(SYMLINKS)"
