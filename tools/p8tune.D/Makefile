# tools/p8tune.D/Makefile
#
# P8-tune.D KLU pattern-only spike (PR-0). Builds the 3 spike binaries
# (dump_adjacency / fd_color_jacobian / klu_analyze_factor) by linking
# against:
#   - $(SHUD_LIB)/libshud.a (SHUD framework objects MINUS main.cpp;
#     built via the additive `make libshud.a` carve-out target landed
#     in SHUD/Makefile at L639 per openspec change p8tune-klu-spike
#     REQ-1 Scenario "Tool authoring with no SHUD source patch")
#   - SuiteSparse: KLU + AMD + BTF + COLAMD + suitesparseconfig
#   - ColPack: column-coloring + Curtis-Powell-Reid seed gen
#   - SUNDIALS: nvector_serial + sundials_core (transitively pulled in
#     by Model_Data.hpp's nvector_serial include)
#   - libomp (Mac) / libgomp (Linux): ColPack's parallel coloring routines
#
# Explicit link line — NOT pkg-config: SuiteSparse Ubuntu apt and Homebrew
# do NOT ship .pc files; ColPack CMake install ships its own cmake config
# (ColPackConfig.cmake) but no .pc file. Documented in README.md
# §link-line.
#
# Install path defaults:
#   - SuiteSparse: /opt/homebrew (Mac brew) — server overrides via
#     SUITESPARSE_INC / SUITESPARSE_LIB on the `make` command line
#   - ColPack: $(HOME)/.local (built from source per README §build-environment-survey)
#   - SHUD libshud.a: ../../SHUD (relative to tools/p8tune.D)
#   - SUNDIALS: pulled from SHUD/InstallSundials/ via SHUD's existing dir

CXX ?= g++
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

# --- SuiteSparse paths ---
SUITESPARSE_INC ?= /opt/homebrew/include/suitesparse
SUITESPARSE_LIB ?= /opt/homebrew/lib

# --- ColPack paths (built from source per README) ---
COLPACK_INC ?= $(HOME)/.local/include/ColPack_headers
# ColPack's cmake install puts libColPack in lib/shared_library/ on Mac
# and lib/ on Linux (depending on CMake's GNUInstallDirs detection).
# Default to shared_library/ which matches the Mac install; on Linux
# override COLPACK_LIB on the make CLI if needed.
COLPACK_LIB ?= $(HOME)/.local/lib/shared_library
# Also link against ColPack's static archive fallback (lib/archive/).
COLPACK_LIB_FALLBACK ?= $(HOME)/.local/lib/archive

# --- SHUD paths ---
SHUD_DIR ?= ../../SHUD
SHUD_SRC ?= $(SHUD_DIR)/src
SHUD_LIB ?= $(SHUD_DIR)
SHUD_LIBSHUD := $(SHUD_LIB)/libshud.a

# --- SUNDIALS (resolved from SHUD's InstallSundials) ---
SUNDIALS_DIR ?= $(SHUD_DIR)/InstallSundials
SUNDIALS_INC := $(SUNDIALS_DIR)/include
SUNDIALS_LIB := $(SUNDIALS_DIR)/lib

INCLUDES := \
  -I$(SHUD_SRC)/Model \
  -I$(SHUD_SRC)/ModelData \
  -I$(SHUD_SRC)/classes \
  -I$(SHUD_SRC)/Equations \
  -I$(SUNDIALS_INC) \
  -I$(SUITESPARSE_INC) \
  -I$(COLPACK_INC) \
  -I.

LDFLAGS := \
  -L$(SUITESPARSE_LIB) \
  -L$(COLPACK_LIB) \
  -L$(COLPACK_LIB_FALLBACK) \
  -L$(SUNDIALS_LIB)

# RPATH so dylibs resolve at runtime (mirror of SHUD's RPATH pattern).
RPATH := -Wl,-rpath,$(SUNDIALS_LIB) \
         -Wl,-rpath,$(SUITESPARSE_LIB) \
         -Wl,-rpath,$(COLPACK_LIB)

# Link line. Order matters for static libs (libshud.a is archive; pull
# its symbols before SuiteSparse/SUNDIALS deps so unresolved Model_Data
# refs trigger transitive lookups).
LIBS := \
  -lm \
  -lsundials_cvode \
  -lsundials_nvecserial \
  -lklu -lamd -lbtf -lcolamd -lsuitesparseconfig \
  -lColPack

BINARIES := dump_adjacency fd_color_jacobian klu_analyze_factor

.PHONY: shud_spike all clean

shud_spike all: $(BINARIES)

# Each binary links against libshud.a (built by SHUD/Makefile libshud.a target).
# We do NOT have a build-rule dependency on $(SHUD_LIBSHUD) here — the
# top-level Makefile (../../Makefile) recurses into SHUD first to build
# libshud.a, then into here. Invoking `make` from this dir alone REQUIRES
# libshud.a to already exist.

dump_adjacency: dump_adjacency.cpp $(SHUD_LIBSHUD)
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(INCLUDES) $< $(SHUD_LIBSHUD) $(LDFLAGS) $(LIBS) $(OMP_LFLAGS) $(RPATH) -o $@

fd_color_jacobian: fd_color_jacobian.cpp $(SHUD_LIBSHUD)
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(INCLUDES) $< $(SHUD_LIBSHUD) $(LDFLAGS) $(LIBS) $(OMP_LFLAGS) $(RPATH) -o $@

klu_analyze_factor: klu_analyze_factor.cpp $(SHUD_LIBSHUD)
	$(CXX) $(CXXFLAGS) $(OMP_CFLAGS) $(INCLUDES) $< $(SHUD_LIBSHUD) $(LDFLAGS) $(LIBS) $(OMP_LFLAGS) $(RPATH) -o $@

clean:
	@rm -f $(BINARIES)
	@echo "tools/p8tune.D: cleaned $(BINARIES)"
