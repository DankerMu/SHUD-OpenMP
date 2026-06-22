# tools/check_sizeof/Makefile — S5d 汇总验收 (#183) T8.1
#
# Builds the 1-TU sizeof emission tool against the SHUD worktree headers.
# Invoked by ../check_sizeof.sh; standalone:
#   make -C tools/check_sizeof
#   ./tools/check_sizeof/check_sizeof
#
# No SHUD lib link — only headers are needed; the tool is a self-contained
# program that prints sizeof(ElementHotData) and sizeof(_Element).

SHUD_ROOT     ?= $(CURDIR)/../../SHUD
SHUD_SRC      ?= $(SHUD_ROOT)/src

# SUNDIALS install path (matches SHUD/Makefile InstallSundials/cvode-6.0.0).
# The tool only needs sundials headers transitively (Model_Data.hpp etc.
# include them); no link against libsundials.
SUNDIALS_DIR  ?= $(SHUD_ROOT)/InstallSundials

CXX           ?= c++
CXXSTD        ?= -std=c++17
CXXFLAGS      ?= -O0 -Wall -Wno-unused-variable -Wno-deprecated-declarations
INCLUDES      := \
    -I$(SUNDIALS_DIR)/include \
    -I$(SHUD_SRC)/Model \
    -I$(SHUD_SRC)/ModelData \
    -I$(SHUD_SRC)/classes \
    -I$(SHUD_SRC)/Equations

TARGET        := check_sizeof
SRC           := check_sizeof.cpp

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXSTD) $(CXXFLAGS) $(INCLUDES) -o $@ $<

clean:
	rm -f $(TARGET) $(TARGET).o
