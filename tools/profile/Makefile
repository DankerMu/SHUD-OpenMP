# tools/profile — standalone profile timer library (S0-8a, openMP #10).
#
# Builds libshud_profile.a from timer.cpp. The SHUD submodule's Makefile
# does NOT link against this archive in the #10 scope; SHUD instead picks
# up timer.cpp directly via `-I$(CURDIR)/../tools/profile` and appends
# the source to its compile line when SHUD_ENABLE_PROFILE=1 (matches the
# MD_rhs_dump replication precedent — SHUD submodule + outer repo share
# headers but never link across the boundary at build time).
#
# This Makefile exists so the tools/profile TU can be sanity-built in CI
# (S0-9) standalone without the SHUD checkout.
#
# Plain C++17, no SUNDIALS / SHUD dependency.

CXX       ?= g++
CXXFLAGS  ?= -std=c++17 -O2 -Wall -Wextra -DSHUD_ENABLE_PROFILE=1
AR        ?= ar
ARFLAGS   := rcs

TARGET    := libshud_profile.a
SOURCES   := timer.cpp
OBJECTS   := $(SOURCES:.cpp=.o)
HEADERS   := timer.h

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(AR) $(ARFLAGS) $@ $^

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) $(OBJECTS)
