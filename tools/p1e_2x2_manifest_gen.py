#!/usr/bin/env python3
"""tools/p1e_2x2_manifest_gen.py

P1e PR-B (issue #310) — generate docs/p1e/p1e_2x2_manifest.yaml.

192 cells = 4 build (A/B/C/D) x 4 case x 4 N (1,2,4,8) x 3 reps. Cases are
platform-split per design D1 + CLAUDE.md double-env section:

  Mac local: keliya, qhh
  Server:    heihe, heihe_x4

Phase split (per design D1):
  Phase 1 (PR-B/C/D pre-implementation): mode A + B = 96 cells
  Phase 2 (PR-G post-implementation):    mode C + D = 96 cells

The yaml is consumed by PR-C (Mac driver) and PR-D (server driver) as the
authoritative cell list; expected_sha / expected_nst / expected_wall are
left empty for PR-C/D (Phase 1) and PR-G post-merge (Phase 2) to fill in.

Run with uv per CLAUDE.md Python iron rule:
  uv run tools/p1e_2x2_manifest_gen.py [--out docs/p1e/p1e_2x2_manifest.yaml]

The script depends on PyYAML; uv resolves it on the fly via the inline
script metadata header below.
"""

# /// script
# requires-python = ">=3.9"
# dependencies = ["PyYAML>=6.0"]
# ///

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "error: PyYAML not available. Re-run via `uv run "
        "tools/p1e_2x2_manifest_gen.py` so the inline script-deps header "
        "pulls it in.",
        file=sys.stderr,
    )
    sys.exit(2)


BUILDS = ("A", "B", "C", "D")
NS = (1, 2, 4, 8)
REPS = (1, 2, 3)

# Per design D1 + CLAUDE.md double-env section.
PLATFORM_CASES = {
    "mac": ("keliya", "qhh"),
    "server": ("heihe", "heihe_x4"),
}


def build_phase(build: str) -> int:
    """Phase 1: mode A/B (pre-implementation). Phase 2: mode C/D (post)."""
    return 1 if build in ("A", "B") else 2


def case_platform(case: str) -> str:
    for platform, cases in PLATFORM_CASES.items():
        if case in cases:
            return platform
    raise ValueError(f"unknown case '{case}'")


def gen_cells() -> list[dict]:
    cells: list[dict] = []
    # Iterate in a stable, human-readable order: build -> case -> N -> rep
    all_cases = PLATFORM_CASES["mac"] + PLATFORM_CASES["server"]
    for build in BUILDS:
        for case in all_cases:
            for n in NS:
                for rep in REPS:
                    cell_id = f"{build}_{case}_N{n}_rep{rep}"
                    cells.append(
                        {
                            "id": cell_id,
                            "phase": build_phase(build),
                            "platform": case_platform(case),
                            "case": case,
                            "build": build,
                            "N": n,
                            "rep": rep,
                            # Filled in by:
                            #   Phase 1 (mode A/B) — PR-C (Mac) + PR-D (server)
                            #   Phase 2 (mode C/D) — PR-G post-merge run
                            "expected_sha": "",
                            "expected_nst": 0,
                            "expected_wall": 0.0,
                        }
                    )
    return cells


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate docs/p1e/p1e_2x2_manifest.yaml (192 cells)."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/p1e/p1e_2x2_manifest.yaml"),
        help="output yaml path (default: docs/p1e/p1e_2x2_manifest.yaml)",
    )
    args = parser.parse_args()

    cells = gen_cells()
    n_phase_1 = sum(1 for c in cells if c["phase"] == 1)
    n_phase_2 = sum(1 for c in cells if c["phase"] == 2)
    assert len(cells) == 192, f"expected 192 cells, got {len(cells)}"
    assert n_phase_1 == 96, f"expected 96 Phase 1 cells, got {n_phase_1}"
    assert n_phase_2 == 96, f"expected 96 Phase 2 cells, got {n_phase_2}"

    doc = {
        "schema_version": 1,
        "epic": "P1e",
        "pr": "PR-B (#310)",
        "design_ref": "openspec/changes/p1e-strict-omp-rhs/design.md D1 + D6",
        "tasks_ref": "openspec/changes/p1e-strict-omp-rhs/tasks.md §2.2",
        "cell_count": len(cells),
        "phase_1_count": n_phase_1,
        "phase_2_count": n_phase_2,
        "build_modes": {
            "A": "make shud                          -> Serial NVECTOR + Serial RHS (canonical baseline)",
            "B": "make shud_omp                      -> OpenMP NVECTOR + Serial RHS (= current shud_omp)",
            "C": "make shud SHUD_ENABLE_OPENMP_RHS=1 -> Serial NVECTOR + StrictOMP RHS (P1e production candidate)",
            "D": "make shud_omp SHUD_ENABLE_OPENMP_RHS=1 -> OpenMP NVECTOR + StrictOMP RHS (research boundary)",
        },
        "thread_counts": list(NS),
        "reps_per_n": len(REPS),
        "platform_cases": {k: list(v) for k, v in PLATFORM_CASES.items()},
        "cells": cells,
    }

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        # sort_keys=False keeps the natural dict ordering (id-first, etc.)
        # default_flow_style=False forces block style for nested mappings,
        # which makes the cells list readable as one cell per line group.
        yaml.dump(doc, f, sort_keys=False, default_flow_style=False, width=120)

    print(
        f"wrote {out_path}: cells={len(cells)} "
        f"(phase_1={n_phase_1}, phase_2={n_phase_2})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
