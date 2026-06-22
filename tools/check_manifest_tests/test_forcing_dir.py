#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///
#
# tools/check_manifest_tests/test_forcing_dir.py
#
# Self-contained tests for _check_forcing_dir() in tools/check_manifest.py.
# Covers the four fixtures called out in m7-forcing-trim spec Requirement
# "tools/check_manifest.py 兼容升级":
#
#   1. legacy str (backward compatibility)
#   2. new dict {original_path, trimmed_path: str}
#   3. kashigeer-style dict {original_path, trimmed_path: null}
#   4. malformed dict {trimmed_path: ...} (no original_path) — must reject
#
# Run with `uv run tools/check_manifest_tests/test_forcing_dir.py`. Exit 0
# on all pass; exit 1 on any failure with details.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_check_manifest():
    spec_path = Path(__file__).resolve().parents[1] / "check_manifest.py"
    spec = importlib.util.spec_from_file_location("check_manifest", spec_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load check_manifest from {spec_path}")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass(__module__) can resolve the parent.
    sys.modules["check_manifest"] = mod
    spec.loader.exec_module(mod)
    return mod


def fresh_result(cm):
    return cm.ValidationResult(case="fixture", manifest_path=Path("/dev/null"))


def expect_pass(cm, payload, label):
    r = fresh_result(cm)
    cm._check_forcing_dir(r, payload)
    if r.errors:
        print(f"[FAIL] {label}: unexpected errors: {r.errors}")
        return False
    print(f"[PASS] {label}")
    return True


def expect_fail(cm, payload, label, error_substr):
    r = fresh_result(cm)
    cm._check_forcing_dir(r, payload)
    if not r.errors:
        print(f"[FAIL] {label}: expected error, got none")
        return False
    if not any(error_substr in e for e in r.errors):
        print(
            f"[FAIL] {label}: error '{error_substr}' not found; got {r.errors}"
        )
        return False
    print(f"[PASS] {label} (errors: {r.errors})")
    return True


def main() -> int:
    cm = load_check_manifest()

    all_ok = True

    # Fixture 1: legacy scalar string. Backward compatible — must pass.
    all_ok &= expect_pass(
        cm,
        {"forcing_dir": "SHUD/Basins/keliya/forcing/"},
        "fixture 1: legacy str",
    )

    # Fixture 2: new dict with both fields.
    all_ok &= expect_pass(
        cm,
        {
            "forcing_dir": {
                "original_path": "SHUD/Basins/keliya/forcing/",
                "trimmed_path": "SHUD/Basins/keliya/forcing.trimmed/",
            }
        },
        "fixture 2: new dict with trimmed_path",
    )

    # Fixture 3: kashigeer-style dict with trimmed_path null.
    all_ok &= expect_pass(
        cm,
        {
            "forcing_dir": {
                "original_path": "SHUD/Basins/kashigeer/forcing/",
                "trimmed_path": None,
            }
        },
        "fixture 3: kashigeer trimmed_path null",
    )

    # Fixture 4: malformed dict — missing original_path — must raise.
    all_ok &= expect_fail(
        cm,
        {
            "forcing_dir": {
                "trimmed_path": "SHUD/Basins/keliya/forcing.trimmed/",
            }
        },
        "fixture 4: malformed dict (no original_path)",
        "original_path",
    )

    # Bonus: missing forcing_dir entirely.
    all_ok &= expect_fail(
        cm,
        {},
        "bonus: missing forcing_dir",
        "missing required field",
    )

    # Bonus: trimmed_path with wrong type (e.g. int).
    all_ok &= expect_fail(
        cm,
        {
            "forcing_dir": {
                "original_path": "SHUD/Basins/keliya/forcing/",
                "trimmed_path": 42,
            }
        },
        "bonus: trimmed_path wrong type",
        "expected str or null",
    )

    # Bonus: original_path with wrong type.
    all_ok &= expect_fail(
        cm,
        {
            "forcing_dir": {
                "original_path": 42,
                "trimmed_path": "SHUD/Basins/keliya/forcing.trimmed/",
            }
        },
        "bonus: original_path wrong type",
        "expected str",
    )

    print("")
    print("===== summary =====")
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
