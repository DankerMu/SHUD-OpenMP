# compare_snapshot — standalone CLI build (S0-7, openmp issue #9).
#
# Plain C++17, no SUNDIALS / SHUD link. Works on a stock Ubuntu 22.04
# runner (CI baseline) and on macOS via Apple Clang's g++ wrapper.

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra
INCLUDES  := -I../rhs_snapshot

TARGET    := compare_snapshot
SOURCES   := main.cpp

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(SOURCES) ../rhs_snapshot/format.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(SOURCES)

clean:
	rm -f $(TARGET)
