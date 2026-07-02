"""Parsers for the three P11-osc diagnostic CSVs (pinned schemas).

Schemas are pinned to the real emitter `SHUD/src/Model/MD_osc_diag.hpp`
(PR-D1) and the dt-step-trace / osc-flip-counters specs.

diag_dt_trace.csv
    # project_name=<name> solverstep_min=<minutes>
    t_next,delta_nst,delta_nfe,delta_ncfn,delta_netf,h_last,h_cur

diag_osc_flips.csv
    # project_name=<name> solverstep_min=<minutes> epsilon_m=<m>
    entity_type,entity_id,flips_sf,flips_us,flips_gw,flips_stage
    (entity_type in {ele, riv}; ele rows carry sf/us/gw with flips_stage=0;
     riv rows carry flips_stage with element families=0; entity_id 1-based
     within its own index space)

diag_osc_flips_daily.csv
    # project_name=<name> solverstep_min=<minutes> epsilon_m=<m>
    day_index,flips_sf,flips_us,flips_gw,flips_stage,flips_total
    (day_index = floor(t_next / 1440); one row per model day, per-day
     AGGREGATE totals)

FAIL CLOSED: every structural violation (missing file, missing/short header,
wrong column set, non-numeric field, empty body, row skips the delta_nst=0
rule inconsistently) raises MalformedTraceError. The CLI turns any such
error into a non-zero exit with NO MARKER block.

`delta_nst = 0` rows are NOT dropped by the parser: they are returned intact
so the consumer (metrics) can apply the spec's skip rule (0 contribution to
burst numerator AND nst denominator). Keeping the drop in one place avoids a
double-skip bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class MalformedTraceError(ValueError):
    """Raised when any diagnostic CSV is missing, truncated, or unparsable.

    The CLI maps this to a non-zero exit and emits no MARKER block (spec
    Requirement: fail closed on malformed input).
    """


# ---------------------------------------------------------------------------
# Header parsing (shared by all three CSVs)
# ---------------------------------------------------------------------------

def _parse_kv_header(line: str, path: Path) -> dict[str, str]:
    """Parse a `# key=value key=value ...` header line into a dict.

    The emitter writes `# project_name=%s solverstep_min=%.10g[ epsilon_m=%.10g]`
    (space-separated key=value tokens after a leading `#`).
    """
    stripped = line.strip()
    if not stripped.startswith("#"):
        raise MalformedTraceError(
            f"{path}: expected a '# key=value ...' header on line 1; got {line!r}"
        )
    body = stripped[1:].strip()
    kv: dict[str, str] = {}
    for tok in body.split():
        if "=" not in tok:
            raise MalformedTraceError(
                f"{path}: malformed header token {tok!r} (expected key=value)"
            )
        key, _, val = tok.partition("=")
        if not key:
            raise MalformedTraceError(
                f"{path}: malformed header token {tok!r} (empty key)"
            )
        kv[key] = val
    return kv


def _require_header_float(kv: dict[str, str], key: str, path: Path) -> float:
    if key not in kv:
        raise MalformedTraceError(f"{path}: header missing required key {key!r}")
    try:
        return float(kv[key])
    except (TypeError, ValueError) as exc:
        raise MalformedTraceError(
            f"{path}: header {key}={kv[key]!r} is not a float"
        ) from exc


def _require_header_str(kv: dict[str, str], key: str, path: Path) -> str:
    if key not in kv:
        raise MalformedTraceError(f"{path}: header missing required key {key!r}")
    return kv[key]


def _read_nonempty_lines(path: Path) -> list[str]:
    """Read a text CSV, dropping only fully blank trailing/interior lines.

    A completely empty file, or one with only a header, is malformed
    (fail closed).
    """
    if not path.exists():
        raise MalformedTraceError(f"{path}: file does not exist")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MalformedTraceError(f"{path}: cannot read: {exc}") from exc
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    if not lines:
        raise MalformedTraceError(f"{path}: file is empty")
    return lines


# ---------------------------------------------------------------------------
# diag_dt_trace.csv
# ---------------------------------------------------------------------------

_DT_COLUMNS = (
    "t_next",
    "delta_nst",
    "delta_nfe",
    "delta_ncfn",
    "delta_netf",
    "h_last",
    "h_cur",
)


@dataclass(frozen=True)
class DtTrace:
    """Parsed diag_dt_trace.csv.

    Attributes:
        project_name:   header project_name (the ./shud <arg>; nanlin, etc.).
        solverstep_min: header solverstep_min (minutes; NEVER from cfg.para).
        t_next:         (N,) model-time minutes at each interval boundary.
        delta_nst:      (N,) CVODE step-count delta per interval (>= 0).
        delta_nfe/ncfn/netf: (N,) companion counter deltas (>= 0).
        h_last/h_cur:   (N,) last / current internal step size.
    """

    project_name: str
    solverstep_min: float
    t_next: np.ndarray
    delta_nst: np.ndarray
    delta_nfe: np.ndarray
    delta_ncfn: np.ndarray
    delta_netf: np.ndarray
    h_last: np.ndarray
    h_cur: np.ndarray

    @property
    def n_rows(self) -> int:
        return int(self.t_next.size)


def parse_dt_trace(path: str | Path) -> DtTrace:
    """Parse diag_dt_trace.csv with the pinned schema. Fail closed."""
    path = Path(path)
    lines = _read_nonempty_lines(path)
    if len(lines) < 3:
        raise MalformedTraceError(
            f"{path}: truncated — need header + column row + >=1 data row, "
            f"got {len(lines)} non-empty line(s)"
        )

    kv = _parse_kv_header(lines[0], path)
    project_name = _require_header_str(kv, "project_name", path)
    solverstep_min = _require_header_float(kv, "solverstep_min", path)
    if solverstep_min <= 0.0:
        raise MalformedTraceError(
            f"{path}: solverstep_min={solverstep_min} must be > 0"
        )

    col_line = lines[1].strip()
    cols = tuple(c.strip() for c in col_line.split(","))
    if cols != _DT_COLUMNS:
        raise MalformedTraceError(
            f"{path}: column header {cols} != expected {_DT_COLUMNS}"
        )

    rows: list[tuple[float, int, int, int, int, float, float]] = []
    for lineno, raw in enumerate(lines[2:], start=3):
        fields = raw.split(",")
        if len(fields) != len(_DT_COLUMNS):
            raise MalformedTraceError(
                f"{path}:{lineno}: expected {len(_DT_COLUMNS)} fields, "
                f"got {len(fields)}: {raw!r}"
            )
        try:
            t_next = float(fields[0])
            d_nst = int(fields[1])
            d_nfe = int(fields[2])
            d_ncfn = int(fields[3])
            d_netf = int(fields[4])
            h_last = float(fields[5])
            h_cur = float(fields[6])
        except ValueError as exc:
            raise MalformedTraceError(
                f"{path}:{lineno}: non-numeric field in {raw!r}: {exc}"
            ) from exc
        if d_nst < 0 or d_nfe < 0 or d_ncfn < 0 or d_netf < 0:
            raise MalformedTraceError(
                f"{path}:{lineno}: negative counter delta (cumulative counters "
                f"are monotone non-decreasing): {raw!r}"
            )
        rows.append((t_next, d_nst, d_nfe, d_ncfn, d_netf, h_last, h_cur))

    arr = np.array(rows, dtype=np.float64)
    return DtTrace(
        project_name=project_name,
        solverstep_min=solverstep_min,
        t_next=arr[:, 0].copy(),
        delta_nst=arr[:, 1].astype(np.int64),
        delta_nfe=arr[:, 2].astype(np.int64),
        delta_ncfn=arr[:, 3].astype(np.int64),
        delta_netf=arr[:, 4].astype(np.int64),
        h_last=arr[:, 5].copy(),
        h_cur=arr[:, 6].copy(),
    )


# ---------------------------------------------------------------------------
# diag_osc_flips.csv (per-entity)
# ---------------------------------------------------------------------------

_FLIPS_COLUMNS = (
    "entity_type",
    "entity_id",
    "flips_sf",
    "flips_us",
    "flips_gw",
    "flips_stage",
)


@dataclass(frozen=True)
class FlipCounts:
    """Parsed diag_osc_flips.csv (per-entity cumulative flip counts).

    Attributes:
        project_name / solverstep_min / epsilon_m: header fields.
        ele_flips_total: (n_ele,) per-element flips = flips_sf+us+gw summed.
                         Population for the top-1% concentration metric.
        riv_flips_stage: (n_riv,) per-river stage-family flips. Reported
                         separately, OUTSIDE the concentration denominator.
    """

    project_name: str
    solverstep_min: float
    epsilon_m: float
    ele_flips_total: np.ndarray
    riv_flips_stage: np.ndarray

    @property
    def n_ele(self) -> int:
        return int(self.ele_flips_total.size)

    @property
    def n_riv(self) -> int:
        return int(self.riv_flips_stage.size)


def parse_osc_flips(path: str | Path) -> FlipCounts:
    """Parse diag_osc_flips.csv with the pinned schema. Fail closed.

    Per-element flips are summed across the sf/us/gw families (the
    concentration population is elements only). River `riv` rows carry
    flips_stage only and are returned separately.
    """
    path = Path(path)
    lines = _read_nonempty_lines(path)
    if len(lines) < 3:
        raise MalformedTraceError(
            f"{path}: truncated — need header + column row + >=1 data row, "
            f"got {len(lines)} non-empty line(s)"
        )

    kv = _parse_kv_header(lines[0], path)
    project_name = _require_header_str(kv, "project_name", path)
    solverstep_min = _require_header_float(kv, "solverstep_min", path)
    epsilon_m = _require_header_float(kv, "epsilon_m", path)

    cols = tuple(c.strip() for c in lines[1].split(","))
    if cols != _FLIPS_COLUMNS:
        raise MalformedTraceError(
            f"{path}: column header {cols} != expected {_FLIPS_COLUMNS}"
        )

    ele: list[int] = []
    riv: list[int] = []
    for lineno, raw in enumerate(lines[2:], start=3):
        fields = raw.split(",")
        if len(fields) != len(_FLIPS_COLUMNS):
            raise MalformedTraceError(
                f"{path}:{lineno}: expected {len(_FLIPS_COLUMNS)} fields, "
                f"got {len(fields)}: {raw!r}"
            )
        etype = fields[0].strip()
        try:
            f_sf = int(fields[2])
            f_us = int(fields[3])
            f_gw = int(fields[4])
            f_stage = int(fields[5])
        except ValueError as exc:
            raise MalformedTraceError(
                f"{path}:{lineno}: non-numeric flip count in {raw!r}: {exc}"
            ) from exc
        if min(f_sf, f_us, f_gw, f_stage) < 0:
            raise MalformedTraceError(
                f"{path}:{lineno}: negative flip count in {raw!r}"
            )
        if etype == "ele":
            # ele rows carry element families; flips_stage MUST be 0 per schema.
            if f_stage != 0:
                raise MalformedTraceError(
                    f"{path}:{lineno}: ele row has flips_stage={f_stage} != 0"
                )
            ele.append(f_sf + f_us + f_gw)
        elif etype == "riv":
            # riv rows carry stage only; element families MUST be 0 per schema.
            if f_sf != 0 or f_us != 0 or f_gw != 0:
                raise MalformedTraceError(
                    f"{path}:{lineno}: riv row has non-zero element families "
                    f"in {raw!r}"
                )
            riv.append(f_stage)
        else:
            raise MalformedTraceError(
                f"{path}:{lineno}: entity_type {etype!r} not in {{ele, riv}}"
            )

    if not ele:
        raise MalformedTraceError(
            f"{path}: no `ele` rows — element flip population is empty"
        )

    return FlipCounts(
        project_name=project_name,
        solverstep_min=solverstep_min,
        epsilon_m=epsilon_m,
        ele_flips_total=np.array(ele, dtype=np.int64),
        riv_flips_stage=np.array(riv, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# diag_osc_flips_daily.csv (per-day aggregate)
# ---------------------------------------------------------------------------

_DAILY_COLUMNS = (
    "day_index",
    "flips_sf",
    "flips_us",
    "flips_gw",
    "flips_stage",
    "flips_total",
)


@dataclass(frozen=True)
class DailyFlips:
    """Parsed diag_osc_flips_daily.csv (one row per model day).

    Attributes:
        project_name / solverstep_min / epsilon_m: header fields.
        day_index:    (D,) day_index = floor(t_next / 1440), ascending.
        flips_total:  (D,) per-day aggregate flips (sf+us+gw+stage).
    """

    project_name: str
    solverstep_min: float
    epsilon_m: float
    day_index: np.ndarray
    flips_total: np.ndarray

    @property
    def n_days(self) -> int:
        return int(self.day_index.size)


def parse_osc_flips_daily(path: str | Path) -> DailyFlips:
    """Parse diag_osc_flips_daily.csv with the pinned schema. Fail closed.

    Validates the emitter invariant flips_total == flips_sf+us+gw+stage on
    every row.
    """
    path = Path(path)
    lines = _read_nonempty_lines(path)
    if len(lines) < 3:
        raise MalformedTraceError(
            f"{path}: truncated — need header + column row + >=1 data row, "
            f"got {len(lines)} non-empty line(s)"
        )

    kv = _parse_kv_header(lines[0], path)
    project_name = _require_header_str(kv, "project_name", path)
    solverstep_min = _require_header_float(kv, "solverstep_min", path)
    epsilon_m = _require_header_float(kv, "epsilon_m", path)

    cols = tuple(c.strip() for c in lines[1].split(","))
    if cols != _DAILY_COLUMNS:
        raise MalformedTraceError(
            f"{path}: column header {cols} != expected {_DAILY_COLUMNS}"
        )

    days: list[int] = []
    totals: list[int] = []
    for lineno, raw in enumerate(lines[2:], start=3):
        fields = raw.split(",")
        if len(fields) != len(_DAILY_COLUMNS):
            raise MalformedTraceError(
                f"{path}:{lineno}: expected {len(_DAILY_COLUMNS)} fields, "
                f"got {len(fields)}: {raw!r}"
            )
        try:
            day = int(fields[0])
            f_sf = int(fields[1])
            f_us = int(fields[2])
            f_gw = int(fields[3])
            f_stage = int(fields[4])
            f_total = int(fields[5])
        except ValueError as exc:
            raise MalformedTraceError(
                f"{path}:{lineno}: non-numeric field in {raw!r}: {exc}"
            ) from exc
        if f_total != f_sf + f_us + f_gw + f_stage:
            raise MalformedTraceError(
                f"{path}:{lineno}: flips_total={f_total} != "
                f"sf+us+gw+stage={f_sf + f_us + f_gw + f_stage} in {raw!r}"
            )
        days.append(day)
        totals.append(f_total)

    day_arr = np.array(days, dtype=np.int64)
    if day_arr.size != np.unique(day_arr).size:
        raise MalformedTraceError(f"{path}: duplicate day_index rows present")

    return DailyFlips(
        project_name=project_name,
        solverstep_min=solverstep_min,
        epsilon_m=epsilon_m,
        day_index=day_arr,
        flips_total=np.array(totals, dtype=np.int64),
    )
