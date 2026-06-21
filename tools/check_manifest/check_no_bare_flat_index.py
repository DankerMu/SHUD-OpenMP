#!/usr/bin/env python3
"""check_no_bare_flat_index.py — S5d.2-5a (#179) RHS hot-path accessor gate.

Asserts the three RHS hot-path TUs
  SHUD/src/ModelData/MD_ElementFlux.cpp
  SHUD/src/ModelData/MD_f.cpp
  SHUD/src/ModelData/MD_ET.cpp
contain ZERO bare `QeleSurf_flat[...]` or `QeleSub_flat[...]` indexing
expressions in active code lines. Every access MUST go through the
inline accessors `QeleSurfAt(i, j)` / `QeleSubAt(i, j)` declared in
SHUD/src/ModelData/Model_Data.hpp.

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
RHS_FILES = [
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_ElementFlux.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_f.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_ET.cpp",
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
    for f in RHS_FILES:
        if not f.exists():
            fail(f"missing RHS file {f}")
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
            "RHS hot path bare flat-index grep gate failed; "
            "every hot-path read/write of QeleSurf_flat / QeleSub_flat "
            "MUST go through the QeleSurfAt(i,j) / QeleSubAt(i,j) "
            "inline accessors. Violations:\n  " + "\n  ".join(violators)
        )

    print(
        f"PASS: {len(RHS_FILES)} RHS files have 0 bare "
        f"QeleSurf_flat[...] / QeleSub_flat[...] indexing"
    )


if __name__ == "__main__":
    main()
