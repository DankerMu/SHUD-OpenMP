# Top-level Makefile — SHUD-OpenMP outer repo.
#
# Per openspec change `p8tune-klu-spike` task 1.11 + spec §REQ-7
# Scenario "PR-0 tool PR boundary": this is a NEW top-level Makefile
# (PR-0 task 1.0 verified no top-level Makefile existed). Its only
# concern is wiring `make shud_spike` to recurse into:
#
#   1. SHUD/Makefile libshud.a   (additive carve-out — see SHUD/Makefile
#      ~L639 P8-tune.D KLU spike section)
#   2. tools/p8tune.D/Makefile shud_spike
#      (which builds dump_adjacency + fd_color_jacobian + klu_analyze_factor)
#
# This Makefile does NOT interfere with the normal SHUD build flow —
# users running `make shud` / `make shud_omp` SHALL continue to do so
# from the SHUD/ subdirectory. The spike tool's `make shud_spike` is
# a separate code path requiring SuiteSparse + ColPack external deps
# (see tools/p8tune.D/README.md for install).

.PHONY: shud_spike libshud.a clean_spike help

help:
	@echo "Top-level Makefile (SHUD-OpenMP outer repo) targets:"
	@echo "  make shud_spike    - build P8-tune.D KLU spike tool (per openspec change p8tune-klu-spike)"
	@echo "  make libshud.a     - build SHUD framework objects as static archive (carve-out for spike tool)"
	@echo "  make clean_spike   - remove spike binaries + libshud.a artifacts"
	@echo ""
	@echo "Normal SHUD build (serial / OpenMP) lives in SHUD/Makefile:"
	@echo "  cd SHUD && make shud      - build serial shud executable"
	@echo "  cd SHUD && make shud_omp  - build OpenMP shud_omp executable"
	@echo "  cd SHUD && make help      - SHUD Makefile help"

# Build libshud.a archive via SHUD/Makefile's additive carve-out target.
libshud.a:
	$(MAKE) -C SHUD libshud.a

# Build the 3 spike binaries. Depends on libshud.a being built first
# (the tools/p8tune.D/Makefile expects ../../SHUD/libshud.a to exist).
shud_spike: libshud.a
	$(MAKE) -C tools/p8tune.D shud_spike

# Cleanup. Does NOT touch the regular SHUD `clean` target — only spike artifacts.
clean_spike:
	$(MAKE) -C tools/p8tune.D clean
	@rm -f SHUD/libshud.a
	@rm -rf SHUD/_libshud_obj
	@echo "cleaned spike binaries + libshud.a + _libshud_obj/"
