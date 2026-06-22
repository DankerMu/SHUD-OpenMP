#!/usr/bin/env python3
"""check_omp_env.py — S5d.4 (#182) manifest omp_env schema gate.

Asserts every `benchmarks/<case>/manifest.yaml` carries:

    omp_env:
      OMP_PROC_BIND: close
      OMP_PLACES: cores

Both keys MUST be present and non-empty under the top-level `omp_env`
mapping. Per spec s5d-data-layout-soa-numa Requirement
"线程绑定 run script 与 manifest 字段必填" Scenario "7 case manifest 全有
omp_env 字段": every registered benchmark manifest needs the two keys;
absence of either is a HARD fail.

Default values per design D5 + master plan §S5d.4.3:
    OMP_PROC_BIND: close   (consumer-thread affinity for first-touch)
    OMP_PLACES:    cores   (one thread per core, no SMT siblings)

Exit codes:
    0 — every benchmarks/<case>/manifest.yaml passes the schema gate
    1 — at least one manifest is missing/empty/wrong value
    2 — pyyaml unavailable

Run (local dev):
    uv run --with pyyaml python tools/check_manifest/check_omp_env.py

CI invocation: see .github/workflows/serial-baseline.yml step
"S5d.4 manifest omp_env schema gate (#182)"; same bare-python3 +
pyyaml exception applies as check_hot_fields.py (uv setup happens
later in the workflow, so this early-stage gate pip-installs pyyaml
on-the-fly when uv is unavailable).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print(
        "FAIL: pyyaml missing — install via "
        "`python3 -m pip install --quiet pyyaml` or "
        "`uv run --with pyyaml python ...`",
        file=sys.stderr,
    )
    sys.exit(2)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"

# Per design D5 + master plan §S5d.4.3 defaults. Allowed values are
# constrained to the documented set so a future drift to e.g.
# `OMP_PROC_BIND: spread` ships through CI review, not silently.
REQUIRED_KEYS = ("OMP_PROC_BIND", "OMP_PLACES")
DEFAULT_VALUES = {
    "OMP_PROC_BIND": "close",
    "OMP_PLACES": "cores",
}


def gather_manifests() -> list[Path]:
    """Return the list of benchmarks/<case>/manifest.yaml paths.

    Iterates over benchmarks/<case>/manifest.yaml in sorted order so
    failure messages are stable (deterministic CI output across runs).
    """
    if not BENCHMARKS_DIR.is_dir():
        return []
    paths: list[Path] = []
    for case_dir in sorted(BENCHMARKS_DIR.iterdir()):
        if not case_dir.is_dir():
            continue
        manifest = case_dir / "manifest.yaml"
        if manifest.is_file():
            paths.append(manifest)
    return paths


def validate_manifest(path: Path, violations: list[str]) -> None:
    """Append all violations for `path` to the shared list (no exit)."""
    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        violations.append(f"{path.relative_to(REPO_ROOT)}: YAML parse error: {exc}")
        return
    if not isinstance(data, dict):
        violations.append(
            f"{path.relative_to(REPO_ROOT)}: manifest root must be a mapping"
        )
        return

    omp_env = data.get("omp_env")
    if omp_env is None:
        violations.append(
            f"{path.relative_to(REPO_ROOT)}: missing top-level `omp_env` block "
            f"(spec s5d-data-layout-soa-numa L101-103)"
        )
        return
    if not isinstance(omp_env, dict):
        violations.append(
            f"{path.relative_to(REPO_ROOT)}: `omp_env` must be a mapping, "
            f"got {type(omp_env).__name__}"
        )
        return

    for key in REQUIRED_KEYS:
        if key not in omp_env:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: omp_env.{key} key missing "
                f"(expected `{DEFAULT_VALUES[key]}`)"
            )
            continue
        val = omp_env[key]
        if val is None or (isinstance(val, str) and val.strip() == ""):
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: omp_env.{key} empty "
                f"(expected `{DEFAULT_VALUES[key]}`)"
            )
            continue
        if str(val) != DEFAULT_VALUES[key]:
            violations.append(
                f"{path.relative_to(REPO_ROOT)}: omp_env.{key}=`{val}` "
                f"differs from canonical default `{DEFAULT_VALUES[key]}`"
            )


def main() -> int:
    manifests = gather_manifests()
    if not manifests:
        print(
            f"FAIL: no manifests found under {BENCHMARKS_DIR}; "
            "benchmark registry empty?",
            file=sys.stderr,
        )
        return 1

    violations: list[str] = []
    for manifest in manifests:
        validate_manifest(manifest, violations)

    if violations:
        print(
            "FAIL: omp_env schema gate violations:\n  " + "\n  ".join(violations),
            file=sys.stderr,
        )
        return 1

    print(
        f"PASS: {len(manifests)} manifests carry "
        f"omp_env.{{OMP_PROC_BIND={DEFAULT_VALUES['OMP_PROC_BIND']},"
        f" OMP_PLACES={DEFAULT_VALUES['OMP_PLACES']}}}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
