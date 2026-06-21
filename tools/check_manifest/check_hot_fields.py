#!/usr/bin/env python3
"""check_hot_fields.py — S5d.1 (#178) RHS hot path grep gate.

Asserts:
  1. Every hot field from docs/s5d_hot_fields.yaml is DECLARED in
     SHUD/src/ModelData/MD_layout.hpp.
  2. RHS hot path files (MD_ElementFlux.cpp / MD_f.cpp / MD_ET.cpp)
     contain ZERO instances of `Ele[<expr>].<hot-field>` for any field
     declared in the yaml. Method invocations (Ele[i].updateElement /
     updateLakeElement / Flux_Infiltration / Flux_Recharge) are NOT
     hot fields per the design D2 double-track contract: they keep AoS
     dispatch + are followed by sync_hot_dynamic(i) refresh.

Exit 0 = PASS; non-zero = FAIL with failing-line list.

Run:
  uv run --with pyyaml python tools/check_manifest/check_hot_fields.py

CI exception: when uv is not available at this stage of CI (the runner
sets up uv later for the manifest schema validation), invoke via
  python3 -m pip install --quiet pyyaml && \
    python3 tools/check_manifest/check_hot_fields.py
This is the documented exception to the CLAUDE.md "uv only" rule because
pyyaml is the sole dependency and a single pip install is sufficient.
"""
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FAIL: pyyaml missing — uv run --with pyyaml python ...", file=sys.stderr)
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
YAML_PATH = REPO_ROOT / "docs" / "s5d_hot_fields.yaml"
HEADER_PATH = REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_layout.hpp"
RHS_FILES = [
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_ElementFlux.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_f.cpp",
    REPO_ROOT / "SHUD" / "src" / "ModelData" / "MD_ET.cpp",
]


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not YAML_PATH.exists():
        fail(f"missing {YAML_PATH}")
    with YAML_PATH.open() as f:
        data = yaml.safe_load(f)
    hot_fields = data.get("hot_fields", [])
    if not hot_fields:
        fail("hot_fields empty in yaml")

    field_names = [entry["name"] for entry in hot_fields]
    if len(set(field_names)) != len(field_names):
        fail("duplicate field name in hot_fields list")

    # Check 1: MD_layout.hpp contains every yaml field name as a declaration.
    if not HEADER_PATH.exists():
        fail(f"missing {HEADER_PATH}")
    header_text = HEADER_PATH.read_text()
    for entry in hot_fields:
        name = entry["name"]
        size = entry["size_per_ele"]
        token = f"{name}_flat" if size == 3 else name
        if not re.search(r"\b" + re.escape(token) + r"\b", header_text):
            fail(
                f"MD_layout.hpp missing field declaration for `{token}` "
                f"(yaml name `{name}`, size_per_ele={size})"
            )

    # Check 2: RHS hot path files have 0 hits of `Ele[<expr>].<field>`
    # for any yaml-declared hot field. Comment-only mentions are excluded
    # by stripping `//` and `/* ... */` content; only code lines count.
    pattern_parts = "|".join(re.escape(entry["name"]) for entry in hot_fields)
    pattern = re.compile(r"Ele\[[^\]]+\]\.(?:" + pattern_parts + r")\b")
    block_comment = re.compile(r"/\*.*?\*/", re.DOTALL)
    violators = []
    for f in RHS_FILES:
        if not f.exists():
            fail(f"missing RHS file {f}")
        text = f.read_text()
        # Strip block comments so `/* Ele[i].nabr */` doesn't trip the gate.
        stripped = block_comment.sub("", text)
        for lineno, line in enumerate(stripped.splitlines(), 1):
            # Strip line-comment tail so `// Ele[i].nabr` doesn't trip.
            code = line.split("//", 1)[0]
            for m in pattern.finditer(code):
                violators.append(
                    f"{f.relative_to(REPO_ROOT)}:{lineno}: {m.group(0)}"
                )
    if violators:
        fail(
            "RHS hot path Ele[..].<hot-field> grep gate failed; "
            "must use hot.<field>[<index>] instead. Violations:\n  "
            + "\n  ".join(violators)
        )

    print(f"PASS: {len(hot_fields)} hot fields declared in MD_layout.hpp")
    print(f"PASS: RHS 3 files have 0 Ele[..].<hot-field> hits")


if __name__ == "__main__":
    main()
