#!/usr/bin/env python3
"""test_15key_excludes_nfcall.py — assert canonical_15_keys.yaml excludes nFCall.

Owned by openspec change b1b-baseline-completion (s5c-solver-diagnostics).
Spec requirement: "15-key snapshot 不包含 nFCall" + "nFCall 独立 channel 上报"
+ design D10 (F19 round-2 nFCall vs nfe semantics).

Run:
  uv run python tools/cvode_stats_diff/test_15key_excludes_nfcall.py

Exit 0 = PASS; non-zero = FAIL with diagnostic.
"""
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("FAIL: pyyaml not installed; run via `uv run python`", file=sys.stderr)
    sys.exit(2)

YAML_PATH = Path(__file__).parent / "canonical_15_keys.yaml"
DIFF_SH = Path(__file__).parent / "cvode_stats_diff.sh"

def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)

def main() -> None:
    if not YAML_PATH.exists():
        fail(f"missing canonical_15_keys.yaml at {YAML_PATH}")
    if not DIFF_SH.exists():
        fail(f"missing cvode_stats_diff.sh at {DIFF_SH}")
    with YAML_PATH.open() as f:
        data = yaml.safe_load(f)
    keys = data.get("canonical_15_keys", [])
    excluded = data.get("excluded_keys", [])
    diagnostic = data.get("diagnostic_keys", [])

    if len(keys) != 15:
        fail(f"canonical_15_keys size = {len(keys)}, expected 15")

    nfcall_variants = ["nFCall", "nFCall1", "nFCall2", "nFCall3", "nFCall4", "nFCall5"]
    for v in nfcall_variants:
        if v in keys:
            fail(f"{v} present in canonical_15_keys (spec: must be excluded)")
        if v not in excluded:
            fail(f"{v} missing from excluded_keys (spec: must be enumerated)")

    for d in diagnostic:
        if d in keys:
            fail(f"diagnostic key {d!r} present in canonical_15_keys (spec violation)")

    sh_text = DIFF_SH.read_text()
    sh_line = next(
        (ln for ln in sh_text.splitlines() if ln.lstrip().startswith("CANONICAL_KEYS=")),
        None,
    )
    if sh_line is None:
        fail("cvode_stats_diff.sh: CANONICAL_KEYS line not found")
    sh_keys = sh_line.split('"', 2)[1].split() if '"' in sh_line else []
    if sorted(sh_keys) != sorted(keys):
        fail(
            "cvode_stats_diff.sh CANONICAL_KEYS out of sync with yaml: "
            f"sh={sorted(sh_keys)} yaml={sorted(keys)}"
        )
    for v in nfcall_variants:
        if v in sh_keys:
            fail(f"{v} present in cvode_stats_diff.sh CANONICAL_KEYS")

    print(f"PASS: canonical_15_keys.yaml + cvode_stats_diff.sh both exclude {nfcall_variants}")
    print(f"PASS: canonical_15_keys size = 15; no diagnostic key overlap")
    print(f"PASS: cvode_stats_diff.sh CANONICAL_KEYS == yaml canonical_15_keys (in sync)")

if __name__ == "__main__":
    main()
