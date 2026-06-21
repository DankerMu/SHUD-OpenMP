#!/usr/bin/env python3
"""check_no_bare_flat_index.py — S5d.2-5a (#179) RHS hot-path accessor gate.

Asserts every hot-path TU that touches QeleSurf_flat / QeleSub_flat
contains ZERO bare `QeleSurf_flat[...]` or `QeleSub_flat[...]` indexing
expressions in active code lines. Every access MUST go through the
inline accessors `QeleSurfAt(i, j)` / `QeleSubAt(i, j)` declared in
SHUD/src/ModelData/Model_Data.hpp.

HOT_PATH_FILES enumerates the union of "every TU that calls
QeleSurfAt / QeleSubAt" — established by
  grep -rn 'QeleSurfAt\\|QeleSubAt' SHUD/src/ModelData/
during PR #197 (review A-S2 / B-B5). MD_ET.cpp was dropped from the
list — it contains no Q-array access. MD_rhs_core.cpp does not exist
in this tree (the corresponding hot-path code lives in MD_f.cpp).

Rationale (design D3 + R2 mitigation): the inline accessor centralizes
the row-major index expression `3*i + j` so a future SIMD / NUMA tuning
lands in one place; it also gives a single DEBUG bounds-check insertion
point and prevents 3*j+i index-flip bugs at call sites.

Comment lines (`//` line-tail comments + `/* ... */` block comments) are
stripped before matching so doc references like `// QeleSurf_flat[3*i+j]`
do not trip the gate.

Exit 0 = PASS; non-zero = FAIL with failing-line list.

Run:
  uv run python tools/check_manifest/check_no_bare_flat_index.py

CI exception: invoked via `python3 tools/check_manifest/check_no_bare_flat_index.py`
at the same stage as check_hot_fields.py — no pyyaml dependency, so no pip
install step required. Documented exception to CLAUDE.md "uv only" because
the script has zero third-party deps.
"""
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
# Files that ACCESS QeleSurfAt / QeleSubAt (PR #197 audit, review A-S2):
#   MD_ElementFlux.cpp  — 3 sites (zero-init + surface + sub writes)
#   MD_f.cpp            — 4 sites (f_update zero-loop + f_applyDYi sum)
#   MD_f_uncouple.cpp   — 2 sites (uncoupled-GW total sum)
#   MD_update.cpp       — 2 sites (zero-init in f_update fast path)
# MD_ET.cpp dropped: no Q-array references at all (audit returns 0 hits).
# MD_rhs_core.cpp not present in this tree (legacy fork retired by S2 capstone).
HOT_PATH_FILES = [
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_ElementFlux.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_f.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_f_uncouple.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_update.cpp",
]

# Pattern: `QeleSurf_flat[<anything>]` or `QeleSub_flat[<anything>]`
# (active code only; comments are stripped first). The `\b` left
# boundary keeps tokens like `MyQeleSurf_flat[...]` from matching
# (none exist today, but defense in depth).
PATTERN = re.compile(r"\b(QeleSurf_flat|QeleSub_flat)\s*\[")
BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    violators = []
    for f in HOT_PATH_FILES:
        if not f.exists():
            fail(f"missing hot-path file {f}")
        text = f.read_text()
        # Strip block comments first so `/* ... QeleSurf_flat[3*i+j] ... */`
        # is removed wholesale.
        stripped = BLOCK_COMMENT.sub("", text)
        for lineno, line in enumerate(stripped.splitlines(), 1):
            # Strip line-tail comments so `// QeleSurf_flat[...]` doesn't trip.
            code = line.split("//", 1)[0]
            for m in PATTERN.finditer(code):
                violators.append(
                    f"{f.relative_to(REPO_ROOT)}:{lineno}: bare "
                    f"{m.group(1)}[...] — use QeleSurfAt(i,j) / QeleSubAt(i,j) "
                    f"accessor instead"
                )

    if violators:
        fail(
            "Hot-path bare flat-index grep gate failed; "
            "every hot-path read/write of QeleSurf_flat / QeleSub_flat "
            "MUST go through the QeleSurfAt(i,j) / QeleSubAt(i,j) "
            "inline accessors. Violations:\n  " + "\n  ".join(violators)
        )

    print(
        f"PASS: {len(HOT_PATH_FILES)} hot-path files have 0 bare "
        f"QeleSurf_flat[...] / QeleSub_flat[...] indexing"
    )


if __name__ == "__main__":
    main()
