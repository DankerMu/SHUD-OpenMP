"""End-to-end CLI test.

Builds a pair of synthetic SHUD output directories (with rivqdown +
elevprcp + eleveta + eleygw .dat fixtures), invokes the `a5` CLI as a
library function, and verifies the emitted MARKER block + JSON report.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from a5.cli import main

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _write_dat(
    path: Path, nc: int, n_steps: int, values: np.ndarray, start_yyyymmdd: int = 20260101
) -> None:
    """Write a minimal SHUD .dat file with `values.shape == (n_steps, nc)`."""
    assert values.shape == (n_steps, nc)
    with open(path, "wb") as f:
        header = b"a5-cli-test".ljust(1024, b"\x00")
        f.write(header)
        f.write(np.array([start_yyyymmdd], dtype="<f8").tobytes())
        f.write(np.array([nc], dtype="<f8").tobytes())
        f.write(np.arange(1, nc + 1, dtype="<f8").tobytes())
        for k in range(n_steps):
            row = np.empty(nc + 1, dtype="<f8")
            row[0] = 1440.0 * (k + 1)  # daily cadence
            row[1:] = values[k]
            f.write(row.tobytes())


def _write_case_dir(base: Path, prj: str, discharge: np.ndarray, precip: np.ndarray,
                    et: np.ndarray, gw: np.ndarray) -> Path:
    """Write a full SHUD output directory named `<prj>.out` under `base`."""
    d = base / f"{prj}.out"
    d.mkdir(parents=True, exist_ok=True)
    _write_dat(d / f"{prj}.rivqdown.dat", nc=discharge.shape[1], n_steps=discharge.shape[0], values=discharge)
    _write_dat(d / f"{prj}.elevprcp.dat", nc=precip.shape[1], n_steps=precip.shape[0], values=precip)
    _write_dat(d / f"{prj}.eleveta.dat", nc=et.shape[1], n_steps=et.shape[0], values=et)
    _write_dat(d / f"{prj}.eleygw.dat", nc=gw.shape[1], n_steps=gw.shape[0], values=gw)
    return d


@pytest.fixture()
def paired_case_dirs(tmp_path: Path):
    """Build reference and (near-perfect) candidate output dirs.

    Candidate is reference * 1.005 (0.5% inflation) — this comfortably
    passes the default thresholds.
    """
    rng = np.random.default_rng(seed=7)
    n_steps = 60  # 60 daily rows -> spans two calendar months
    n_riv = 5
    n_ele = 8

    ref_q = rng.uniform(low=100.0, high=1000.0, size=(n_steps, n_riv))
    # Give the outlet (last col) a clear peak so peak-metrics are meaningful
    ref_q[30, -1] = 5000.0
    ref_p = rng.uniform(low=0.5, high=5.0, size=(n_steps, n_ele))
    ref_e = rng.uniform(low=0.2, high=1.5, size=(n_steps, n_ele))
    ref_gw = np.cumsum(rng.normal(loc=0.0, scale=0.01, size=(n_steps, n_ele)), axis=0)

    scale = 1.005
    cand_q = ref_q * scale
    cand_p = ref_p * scale
    cand_e = ref_e * scale
    cand_gw = ref_gw * scale

    ref_dir = _write_case_dir(tmp_path / "ref", "syn", ref_q, ref_p, ref_e, ref_gw)
    cand_dir = _write_case_dir(tmp_path / "cand", "syn", cand_q, cand_p, cand_e, cand_gw)
    return ref_dir, cand_dir


def test_cli_end_to_end_pass(paired_case_dirs, capsys, tmp_path: Path) -> None:
    ref_dir, cand_dir = paired_case_dirs
    out_dir = tmp_path / "a5-report"
    cfg = Path(__file__).parent.parent / "config" / "a5_thresholds.default.yaml"

    # Loosen thresholds for the synthetic-data end-to-end test. Rationale:
    #   * NSE, KGE, peak, runoff-volume are kept tight — they catch the
    #     interesting regressions (shape drift, timing).
    #   * Water balance residual is coarse without per-cell area weights
    #     (see cli._compute_metrics comment); random synthetic P/ET/GW
    #     produces huge residuals. Real SHUD outputs sit near 0.05 because
    #     the model conserves mass — the metric's absolute scale is not
    #     what this end-to-end test is exercising.
    loose = tmp_path / "loose.yaml"
    loose.write_text(
        (cfg.read_text())
        .replace("min: 0.97", "min: 0.95")
        .replace("max: 1.03", "max: 1.05")
        .replace("max: 0.05\n  weight: 0.75", "max: 1000.0\n  weight: 0.75")
    )

    rc = main(
        [
            "--reference",
            str(ref_dir),
            "--candidate",
            str(cand_dir),
            "--config",
            str(loose),
            "--case-name",
            "syn_test",
            "--out",
            str(out_dir),
        ]
    )
    stdout = capsys.readouterr().out
    assert rc == 0, f"expected PASS exit 0, got {rc}. stdout=\n{stdout}"

    assert "MARKER:A5_VERDICT_BEGIN" in stdout
    assert "MARKER:A5_VERDICT_END" in stdout
    assert "case=syn_test" in stdout
    assert "verdict=PASS" in stdout

    # Reports emitted
    js = out_dir / "a5_metrics.json"
    md = out_dir / "a5_verdict.md"
    assert js.exists() and md.exists()
    payload = json.loads(js.read_text())
    assert payload["schema_version"] == "A5-v1.0.0"
    assert payload["overall"]["verdict"] == "PASS"
    assert payload["case"] == "syn_test"


def test_cli_end_to_end_fail_when_candidate_diverges(paired_case_dirs, capsys, tmp_path: Path) -> None:
    ref_dir, cand_dir = paired_case_dirs
    # Overwrite candidate rivqdown with a broken signal (50% low) -> NSE fails
    cand_q = np.zeros((60, 5), dtype=np.float64) + 100.0  # near-constant
    _write_dat(cand_dir / "syn.rivqdown.dat", nc=5, n_steps=60, values=cand_q)

    cfg = Path(__file__).parent.parent / "config" / "a5_thresholds.default.yaml"
    out_dir = tmp_path / "a5-report-fail"
    rc = main(
        [
            "--reference",
            str(ref_dir),
            "--candidate",
            str(cand_dir),
            "--config",
            str(cfg),
            "--case-name",
            "syn_fail",
            "--out",
            str(out_dir),
        ]
    )
    stdout = capsys.readouterr().out
    assert rc == 1, f"expected FAIL exit 1, got {rc}. stdout=\n{stdout}"
    assert "verdict=FAIL" in stdout


def test_cli_help_smoke() -> None:
    """`a5 --help` prints usage and exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "a5.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "hydrology-acceptance" in result.stdout.lower()
    assert "--reference" in result.stdout


# ---------- PR-Y2: water_balance fallback wiring at CLI layer ---------------


def test_cli_water_balance_falls_back_to_nan_without_mesh_metadata_pr_y2(
    paired_case_dirs, capsys, tmp_path: Path
) -> None:
    """CLI end-to-end: without mesh metadata (.sp / .att / .mesh) in the
    reference tree, water_balance_residual must be reported as NaN and the
    overall verdict PASSes (informational-downgrade path per PR-Y2).

    Regression pin: pre-PR-Y2 this run produced a spurious ~1e13
    water_balance_residual and overall FAIL (a5_weighted_score=0.8636).
    """
    ref_dir, cand_dir = paired_case_dirs
    out_dir = tmp_path / "a5-report-y2"
    cfg = Path(__file__).parent.parent / "config" / "a5_thresholds.default.yaml"

    rc = main(
        [
            "--reference",
            str(ref_dir),
            "--candidate",
            str(cand_dir),
            "--config",
            str(cfg),
            "--case-name",
            "syn_y2",
            "--out",
            str(out_dir),
        ]
    )
    stdout = capsys.readouterr().out
    assert rc == 0, f"expected PASS exit 0 under PR-Y2 fallback; got {rc}"
    # MARKER emitted with water_balance_residual=NaN
    assert "water_balance_residual=NaN" in stdout
    assert "verdict=PASS" in stdout

    # Report JSON: water_balance_residual value = "NaN" (JSON sentinel),
    # pass=True (informational), and status="unavailable_no_mesh_metadata".
    payload = json.loads((out_dir / "a5_metrics.json").read_text())
    wb = payload["metrics"]["water_balance_residual"]
    assert wb["value"] == "NaN"
    assert wb["pass"] is True
    assert wb["status"] == "unavailable_no_mesh_metadata"
    # Overall verdict PASS + weighted_score reflects other metrics only.
    assert payload["overall"]["verdict"] == "PASS"
    assert payload["overall"]["weighted_score"] > 0.85


def test_cli_water_balance_never_emits_1e13_scale_nonsense_pr_y2(
    paired_case_dirs, capsys, tmp_path: Path
) -> None:
    """Regression pin: no CLI invocation should ever emit a
    water_balance_residual anywhere near the pre-Y2 1e13 blowup.

    This assertion holds unconditionally under PR-Y2 because the CLI
    either (a) emits NaN via the fallback, or (b) emits a finite bounded
    residual via the Tier-2 area-weighted computation. In neither branch
    is the raw metric fed dimensionally inconsistent inputs.
    """
    ref_dir, cand_dir = paired_case_dirs
    out_dir = tmp_path / "a5-report-y2-nonsense"
    cfg = Path(__file__).parent.parent / "config" / "a5_thresholds.default.yaml"

    rc = main(
        [
            "--reference",
            str(ref_dir),
            "--candidate",
            str(cand_dir),
            "--config",
            str(cfg),
            "--case-name",
            "syn_no_nonsense",
            "--out",
            str(out_dir),
        ]
    )
    assert rc in (0, 1)  # do not care about verdict; only that we ran
    payload = json.loads((out_dir / "a5_metrics.json").read_text())
    wb_val = payload["metrics"]["water_balance_residual"]["value"]
    if wb_val == "NaN":
        return  # NaN fallback → definitionally not 1e13
    # If a finite value ever appears (Tier-2 path), it must be bounded.
    assert isinstance(wb_val, (int, float))
    assert abs(float(wb_val)) < 1000.0, (
        f"water_balance_residual={wb_val} exceeded PR-Y2 sanity ceiling; "
        f"the pre-Y2 bug produced 1e13-scale values — regression."
    )
