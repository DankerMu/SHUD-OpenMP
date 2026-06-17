#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml>=6.0"]
# ///
#
# tools/check_manifest.py
#
# Schema validator for benchmarks/<case>/manifest.yaml registered under
# openspec/changes/s0-baseline-lock/specs/benchmark-registry/spec.md.
# Master plan §S0.2 defines the field template; project extensions add
# `endpoint`, `derived_from`, `refine_factor`.
#
# Modes:
#   tools/check_manifest.py <case>      validate single benchmarks/<case>/manifest.yaml
#                                       exit 0 on PASS, 1 on FAIL
#   tools/check_manifest.py --all       iterate benchmarks/*/manifest.yaml, print one
#                                       PASS/FAIL line per file plus summary; exit
#                                       0 if every present manifest PASSes, 1 otherwise
#   tools/check_manifest.py -h|--help   usage
#
# Behavior intentionally similar to tools/bootstrap_check.sh (terse PASS/FAIL
# lines, no abort on per-case failure in --all mode, helpful hints on FAIL).
#
# Run with `uv` per CLAUDE.md (no naked python / python3 / pip):
#   uv run tools/check_manifest.py --all
# uv auto-installs PyYAML into an ephemeral env from the PEP 723 header above.

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover — only triggers without uv
    sys.stderr.write(
        "ERROR: PyYAML not importable. Run via `uv run tools/check_manifest.py`\n"
        "       (the PEP 723 header pulls pyyaml automatically), or in a venv\n"
        "       install pyyaml manually.\n"
        f"       underlying error: {exc}\n"
    )
    sys.exit(2)


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Master plan §S0.2 required scalar fields → expected Python type(s).
# `None` is allowed wherever the field is a placeholder pre-S0.8 (e.g.
# `expected_walltime_sec`). Type widening (int|float, str|None) is captured
# as a tuple of accepted types.
REQUIRED_SCALAR_FIELDS: dict[str, tuple[type, ...]] = {
    "project_name": (str,),
    "NumEle": (int,),
    "NumRiv": (int,),
    "NumLake": (int,),
    "NumY": (int,),
    "input_dir": (str,),
    "forcing_dir": (str,),
    "forcing_duration_days": (int,),
    "has_cryosphere": (bool,),
    "has_lake": (bool,),
    "has_BC_SS": (bool,),
    "dry_wet_transition": (bool,),
    "run_command": (str,),
    "threads": (int,),
    # expected_walltime_sec is allowed to be None (placeholder until S0.8).
    "expected_walltime_sec": (int, float, type(None)),
}

# Nested fields handled by dedicated validators below.
REQUIRED_NESTED_FIELDS = ("snapshot_probe", "output_compare")

# Project-extension fields (master plan §S0.12 case×endpoint table).
EXTENSION_FIELDS: dict[str, tuple[type, ...]] = {
    "endpoint": (str,),
    "derived_from": (str, type(None)),
    "refine_factor": (int, type(None)),
}

VALID_ENDPOINTS = {"local-and-server", "server-only", "local-only", "deferred-upstream"}

SNAPSHOT_REQUIRED_KEYS = ("t_values", "Y_source", "arrays_to_dump")
OUTPUT_COMPARE_REQUIRED_KEYS = (
    "full_run_regression",
    "output_files",
    "cvode_stats_file",
    "water_balance_file",
)


@dataclass
class ValidationResult:
    case: str
    manifest_path: Path
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent  # tools/check_manifest.py → repo root
BENCHMARKS_ROOT = REPO_ROOT / "benchmarks"


# ---------------------------------------------------------------------------
# Type-check helpers
# ---------------------------------------------------------------------------

def _type_names(types: tuple[type, ...]) -> str:
    """Pretty-print a tuple of expected types."""
    return " | ".join(t.__name__ if t is not type(None) else "null" for t in types)


def _check_scalar(
    result: ValidationResult,
    payload: dict[str, Any],
    field_name: str,
    accepted_types: tuple[type, ...],
    path_prefix: str = "",
) -> None:
    """Validate scalar field presence + type; record errors into `result`."""
    if field_name not in payload:
        result.errors.append(
            f"{path_prefix}{field_name}: missing required field"
        )
        return
    value = payload[field_name]
    # YAML booleans are bool (subclass of int), so reject bool when only int
    # was requested.
    if int in accepted_types and bool not in accepted_types and isinstance(value, bool):
        result.errors.append(
            f"{path_prefix}{field_name}: expected {_type_names(accepted_types)}, "
            f"got bool ({value!r})"
        )
        return
    if not isinstance(value, accepted_types):
        result.errors.append(
            f"{path_prefix}{field_name}: expected {_type_names(accepted_types)}, "
            f"got {type(value).__name__} ({value!r})"
        )


def _check_snapshot_probe(
    result: ValidationResult,
    probe: Any,
    forcing_duration_days: Any,
) -> None:
    """Validate snapshot_probe sub-structure per master plan §S0.2."""
    if not isinstance(probe, dict):
        result.errors.append(
            f"snapshot_probe: expected mapping, got {type(probe).__name__}"
        )
        return
    for key in SNAPSHOT_REQUIRED_KEYS:
        if key not in probe:
            result.errors.append(
                f"snapshot_probe.{key}: missing required field"
            )
    t_values = probe.get("t_values")
    if t_values is not None:
        if not isinstance(t_values, list):
            result.errors.append(
                f"snapshot_probe.t_values: expected list, "
                f"got {type(t_values).__name__}"
            )
        else:
            if len(t_values) < 3:
                result.errors.append(
                    f"snapshot_probe.t_values: need >=3 elements, "
                    f"got {len(t_values)}"
                )
            for idx, tv in enumerate(t_values):
                if isinstance(tv, bool) or not isinstance(tv, (int, float)):
                    result.errors.append(
                        f"snapshot_probe.t_values[{idx}]: expected number, "
                        f"got {type(tv).__name__} ({tv!r})"
                    )
            # Short-duration cap rule (spec scenario "Short-duration case caps
            # at end-of-run"). Only check when both pieces are well-formed
            # numbers; if forcing_duration_days is missing/invalid the scalar
            # check above already records that.
            if (
                isinstance(forcing_duration_days, int)
                and not isinstance(forcing_duration_days, bool)
                and forcing_duration_days < 100
                and all(
                    isinstance(tv, (int, float)) and not isinstance(tv, bool)
                    for tv in t_values
                )
            ):
                cap = forcing_duration_days * 86400
                max_tv = max(t_values)
                if max_tv > cap:
                    result.errors.append(
                        f"snapshot_probe.t_values: max value {max_tv} exceeds "
                        f"forcing_duration_days*86400={cap} (duration "
                        f"{forcing_duration_days} d < 100 d)"
                    )

    y_source = probe.get("Y_source")
    if y_source is not None and not isinstance(y_source, str):
        result.errors.append(
            f"snapshot_probe.Y_source: expected str, "
            f"got {type(y_source).__name__}"
        )

    arrays = probe.get("arrays_to_dump")
    if arrays is not None:
        if not isinstance(arrays, list) or not arrays:
            result.errors.append(
                f"snapshot_probe.arrays_to_dump: expected non-empty list, "
                f"got {type(arrays).__name__}"
            )
        else:
            for idx, name in enumerate(arrays):
                if not isinstance(name, str):
                    result.errors.append(
                        f"snapshot_probe.arrays_to_dump[{idx}]: "
                        f"expected str, got {type(name).__name__}"
                    )


def _check_output_compare(
    result: ValidationResult,
    payload: dict[str, Any],
) -> None:
    """Validate output_compare sub-structure including lake-file invariant."""
    oc = payload.get("output_compare")
    if not isinstance(oc, dict):
        result.errors.append(
            f"output_compare: expected mapping, got {type(oc).__name__}"
        )
        return

    for key in OUTPUT_COMPARE_REQUIRED_KEYS:
        if key not in oc:
            result.errors.append(
                f"output_compare.{key}: missing required field"
            )

    if "full_run_regression" in oc and not isinstance(
        oc["full_run_regression"], bool
    ):
        result.errors.append(
            f"output_compare.full_run_regression: expected bool, "
            f"got {type(oc['full_run_regression']).__name__}"
        )

    for key in ("cvode_stats_file", "water_balance_file"):
        if key in oc and not isinstance(oc[key], str):
            result.errors.append(
                f"output_compare.{key}: expected str, "
                f"got {type(oc[key]).__name__}"
            )

    output_files = oc.get("output_files")
    if output_files is not None:
        if not isinstance(output_files, list) or not output_files:
            result.errors.append(
                f"output_compare.output_files: expected non-empty list, "
                f"got {type(output_files).__name__}"
            )
        else:
            for idx, fname in enumerate(output_files):
                if not isinstance(fname, str):
                    result.errors.append(
                        f"output_compare.output_files[{idx}]: expected str, "
                        f"got {type(fname).__name__}"
                    )

    # Lake-file consistency: has_lake=true requires at least one "lak" file
    # in output_files; has_lake=false rejects any "lak" file.
    has_lake = payload.get("has_lake")
    if isinstance(output_files, list) and isinstance(has_lake, bool):
        lake_files = [
            f for f in output_files
            if isinstance(f, str) and ".lak" in Path(f).name.lower()
        ]
        if has_lake and not lake_files:
            result.errors.append(
                "output_compare.output_files: has_lake=true but no "
                "lake (.lak*) output file listed"
            )
        if not has_lake and lake_files:
            result.errors.append(
                "output_compare.output_files: has_lake=false but lake "
                f"output file(s) present: {lake_files}"
            )


def _check_lake_consistency(result: ValidationResult, payload: dict[str, Any]) -> None:
    """NumLake and has_lake must agree."""
    num_lake = payload.get("NumLake")
    has_lake = payload.get("has_lake")
    if isinstance(num_lake, int) and isinstance(has_lake, bool):
        if has_lake and num_lake < 1:
            result.errors.append(
                f"has_lake=true but NumLake={num_lake}; expected NumLake>=1"
            )
        if not has_lake and num_lake != 0:
            result.errors.append(
                f"has_lake=false but NumLake={num_lake}; expected NumLake==0"
            )


def _check_numy_formula(result: ValidationResult, payload: dict[str, Any]) -> None:
    """NumY MUST equal 3*NumEle + NumRiv + NumLake (Model_Data.cpp:73)."""
    num_ele = payload.get("NumEle")
    num_riv = payload.get("NumRiv")
    num_lake = payload.get("NumLake")
    num_y = payload.get("NumY")
    if (
        isinstance(num_ele, int) and not isinstance(num_ele, bool)
        and isinstance(num_riv, int) and not isinstance(num_riv, bool)
        and isinstance(num_lake, int) and not isinstance(num_lake, bool)
        and isinstance(num_y, int) and not isinstance(num_y, bool)
    ):
        expected = 3 * num_ele + num_riv + num_lake
        if num_y != expected:
            result.errors.append(
                f"NumY={num_y} but 3*NumEle+NumRiv+NumLake={expected} "
                f"(3*{num_ele}+{num_riv}+{num_lake})"
            )


def _check_extensions(result: ValidationResult, payload: dict[str, Any]) -> None:
    """Validate endpoint / derived_from / refine_factor."""
    for fname, accepted in EXTENSION_FIELDS.items():
        _check_scalar(result, payload, fname, accepted)

    endpoint = payload.get("endpoint")
    if isinstance(endpoint, str) and endpoint not in VALID_ENDPOINTS:
        result.errors.append(
            f"endpoint: expected one of {sorted(VALID_ENDPOINTS)}, got {endpoint!r}"
        )

    # derived_from must be a known case ID when set (sanity check: not empty).
    derived = payload.get("derived_from")
    if isinstance(derived, str) and not derived.strip():
        result.errors.append("derived_from: empty string; use null or a case ID")

    refine = payload.get("refine_factor")
    if isinstance(refine, int) and not isinstance(refine, bool) and refine <= 0:
        result.errors.append(
            f"refine_factor: expected positive int or null, got {refine}"
        )


# ---------------------------------------------------------------------------
# Top-level validator
# ---------------------------------------------------------------------------

def validate_manifest(manifest_path: Path, case: str) -> ValidationResult:
    """Load and validate a single manifest file; return ValidationResult."""
    result = ValidationResult(case=case, manifest_path=manifest_path)

    if not manifest_path.is_file():
        result.errors.append(f"manifest file not found: {manifest_path}")
        return result

    try:
        with manifest_path.open("r", encoding="utf-8") as fp:
            payload = yaml.safe_load(fp)
    except yaml.YAMLError as exc:
        result.errors.append(f"YAML parse error: {exc}")
        return result

    if not isinstance(payload, dict):
        result.errors.append(
            f"top-level: expected mapping, got {type(payload).__name__}"
        )
        return result

    for fname, accepted in REQUIRED_SCALAR_FIELDS.items():
        _check_scalar(result, payload, fname, accepted)

    for nested in REQUIRED_NESTED_FIELDS:
        if nested not in payload:
            result.errors.append(f"{nested}: missing required field")

    # Nested validators (best-effort; gracefully handle missing).
    _check_snapshot_probe(
        result, payload.get("snapshot_probe"), payload.get("forcing_duration_days")
    )
    _check_output_compare(result, payload)

    # Cross-field invariants.
    _check_lake_consistency(result, payload)
    _check_numy_formula(result, payload)

    # Project extensions.
    _check_extensions(result, payload)

    # Soft sanity: project_name should match input_dir tail when both are
    # well-formed strings. This catches typos like input_dir referring to
    # another case.
    project_name = payload.get("project_name")
    input_dir = payload.get("input_dir")
    if isinstance(project_name, str) and isinstance(input_dir, str):
        # input_dir convention: ".../<project_name>/" (trailing slash optional).
        tail = Path(input_dir.rstrip("/")).name
        if tail and tail != project_name:
            result.errors.append(
                f"input_dir tail '{tail}' does not match project_name "
                f"'{project_name}'"
            )

    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_result(result: ValidationResult, prefix: str = "") -> None:
    rel = result.manifest_path.relative_to(REPO_ROOT) if result.manifest_path.is_absolute() else result.manifest_path
    if result.ok:
        print(f"{prefix}[PASS] {result.case:<22} {rel}")
    else:
        print(f"{prefix}[FAIL] {result.case:<22} {rel}")
        for err in result.errors:
            print(f"{prefix}       - {err}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_single(case: str) -> int:
    """Validate a single case; return exit code."""
    manifest_path = BENCHMARKS_ROOT / case / "manifest.yaml"
    result = validate_manifest(manifest_path, case)
    _print_result(result)
    return 0 if result.ok else 1


def run_all() -> int:
    """Validate every benchmarks/*/manifest.yaml; return exit code."""
    if not BENCHMARKS_ROOT.is_dir():
        print(f"ERROR: benchmarks dir not found: {BENCHMARKS_ROOT}", file=sys.stderr)
        return 1

    case_dirs = sorted(
        d for d in BENCHMARKS_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if not case_dirs:
        print(f"WARN: no benchmark cases under {BENCHMARKS_ROOT}")
        return 0

    pass_count = 0
    fail_count = 0
    skip_count = 0
    skipped: list[str] = []
    failed: list[str] = []

    for case_dir in case_dirs:
        case = case_dir.name
        manifest_path = case_dir / "manifest.yaml"
        if not manifest_path.is_file():
            # heihe_x4 etc. are intentionally absent before S0-5; surface as
            # SKIP without failing the batch (spec defers heihe_x4 to issue #7).
            rel = manifest_path.relative_to(REPO_ROOT)
            print(f"[SKIP] {case:<22} {rel} (no manifest.yaml)")
            skip_count += 1
            skipped.append(case)
            continue
        result = validate_manifest(manifest_path, case)
        _print_result(result)
        if result.ok:
            pass_count += 1
        else:
            fail_count += 1
            failed.append(case)

    print("")
    print("===== summary =====")
    print(f"PASS: {pass_count}")
    print(f"FAIL: {fail_count}")
    print(f"SKIP: {skip_count}")
    if failed:
        print(f"failed: {' '.join(failed)}")
    if skipped:
        print(f"skipped: {' '.join(skipped)}")

    return 1 if fail_count > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_manifest.py",
        description=(
            "Validate benchmarks/<case>/manifest.yaml against the schema in "
            "openspec/changes/s0-baseline-lock/specs/benchmark-registry/spec.md."
        ),
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "case",
        nargs="?",
        help="case ID (validates benchmarks/<case>/manifest.yaml)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="validate every benchmarks/*/manifest.yaml present",
    )
    args = parser.parse_args(argv)

    if args.all:
        return run_all()
    return run_single(args.case)


if __name__ == "__main__":
    sys.exit(main())
