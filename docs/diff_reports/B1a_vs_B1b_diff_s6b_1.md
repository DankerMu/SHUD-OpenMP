# B1a vs B1b diff report — S6b.1 (AccTemperature divide-zero guard)

**Fix ID**: S6b.1
**Issue**: [#184](https://github.com/DankerMu/SHUD-OpenMP/issues/184)
**SHUD commit**: see `SHUD/B1b_CHANGELOG.md` row "S6b.1"
**Outer PR**: against `baseline/B1b`
**Status**: **zero-impact fix**

This is the first diff report under `docs/diff_reports/`; the directory
is created here as the precedent for all subsequent S6b.* fixes per
design.md D8 ("S6b 每个 fix 独立 commit + 独立 diff report").

## Summary

`_AccTemp::getACC()` in `SHUD/src/classes/AccTemperature.hpp` previously
returned `ACC / que.size()` unconditionally. When `que` is empty (no
push call has yet enqueued a value), the division `/ 0` produced NaN
that, on the `CS.cryosphere == true` branch in `MD_ET.cpp` (L155-156),
propagated into `fu_Surf` / `fu_Sub` and from there into surface
infiltration and subsurface recharge. The fix wraps the return in
`que.empty() ? 0.0 : ACC / que.size()` (master plan §4.12 / §S2.15).

The fix is **zero-impact** against all current B1a-tag benchmark
goldens because the `_AccTemp` class-default `Time_start = -9999.`
(AccTemperature.hpp L17) guarantees the first `push(x, tnow)` call
trips the `(tnow - Time_start) >= 1440.` guard for any non-negative
model time and enqueues a value before `getACC()` is invoked from
`MD_ET.cpp:155-156`. The empty-queue path is therefore only reachable
on `_AccTemp` instances where `getACC()` is called BEFORE any `push`,
which the current SHUD call graph does not do. The patch is defensive:
it removes the NaN attractor from the implementation while preserving
the observed numerical sequence on all benchmark cases.

## Affected cases

| Case | CRYOSPHERE | Reach `MD_ET.cpp:155-156` ? | Bitwise vs B1a-tag | NaN in output | Note |
|---|---|---|---|---|---|
| keliya (Mac) | 1 | YES (first 1440 min) | PASS (90d, see below) | 0 (9/9 binaries via `.s6b-1-runs/scan_nan.sh`) | cryosphere=1 + 90d truncation; AccTemperature path exercised; Mac 1-day NaN-elim auxiliary witness |
| xinanjiang_upstream (Mac) | 1 | YES (first 1440 min) | PASS (90d) | not separately scanned | bitwise PASS implies numerical sequence identical to B1a-tag |
| qinyijiang (Mac) | 1 | YES (first 1440 min) | PASS (90d) | not separately scanned | same |
| qhh (Mac) | 0 | NO (cryosphere branch off) | PASS (90d, 4/4 files) | not applicable | AccTemperature code path entirely bypassed |
| heihe (server) | 1 | YES (first 1440 min) | not separately re-run (deferred to S6c capstone per design.md D9 trigger #1) | 0 (7/7 binaries via `.s6b-1-runs/scan_nan.sh`) | primary witness per issue #184 "Runs On"; Slurm 三铁律 job 8627 cn07 ExitCode 0:0 elapsed 06:04 |

## Unaffected cases

`kashigeer` is N/A per master plan §4.22 (Mac-side mesh tooling out of
scope; goldens only archived for the 5 lake/topology cases).
`heihe_x4` lives on server (CLAUDE.md "双端实验环境"); the bitwise
re-run is deferred to S6c capstone validation. `heihe` itself is
covered in the table above as the server-primary NaN-elimination
witness (Slurm 三铁律 job 8627). Both are `CRYOSPHERE=1` per
`cfg.para` audit; the same first-push guarantee applies, so
zero-impact is expected to hold.

## Variance description (numerical)

- **Pre-fix expression** at `AccTemperature.hpp:61` (single-statement
  body): `return ACC / que.size();`
- **Post-fix expression** at `AccTemperature.hpp:67`:
  `return que.empty() ? 0.0 : ACC / que.size();`
- For all reachable call sites in the current SHUD call graph,
  `que.size() >= 1` holds at the time `getACC()` is invoked, so the
  conditional always selects the `ACC / que.size()` branch. The
  bitwise SHA256 on 4 cases × 8 goldens (keliya rivqdown,
  xinanjiang rivqdown + eleygw, qinyijiang nanlin rivqdown, qhh
  rivqdown + lakqrivin + lakqrivout + lakystage) reproduces B1a-tag
  byte-for-byte.
- The only behavioural change is on a hypothetical reachable path
  where `getACC()` is called against an `_AccTemp` instance whose
  `que` is empty. That call would have returned `NaN` previously
  and now returns `0.0`.

## Physical interpretation (why 0.0 is the right fallback)

`_AccTemp::getACC()` returns the running mean of the most recent
`MaxLen` daily-averaged temperatures stored in `que`. When `que` is
empty, no fully-elapsed 24-hr window has been accumulated yet —
there is no temperature history to average. The `fu_Surf` / `fu_Sub`
consumers in `MD_ET.cpp:157-158` compute
`1. - FrozenFraction(ta, AccT_*_max, AccT_*_min)`; substituting
`ta = 0.0` gives a `FrozenFraction` that depends on the model's
configured `AccT_*_max` / `AccT_*_min` window, and crucially yields a
**finite** infiltration / recharge multiplier rather than NaN. Master
plan §4.12 documents the choice: "加 guard `que.empty() ? 0.0 :
ACC/que.size()`" — 0.0 is the spec-blessed default for "no
accumulated history".

The alternative defaults — `NaN`, `-9999`, or
`std::numeric_limits<double>::quiet_NaN()` — all force a defective
RHS evaluation. Returning 0 means "no frozen-fraction damping yet",
which on the cryosphere branch behaves like an unfrozen surface and
is the most physically conservative startup assumption.

## Zero-impact marker

**Confirmed zero impact on all 4 Mac benchmark cases via 90-day
NUM_OPENMP=1 bitwise SHA256 vs B1a-tag (8/8 PASS).** See
`SHUD/B1b_CHANGELOG.md` row "S6b.1" for the full SHA table.

Cryosphere first-1440-min NaN elimination evidence (reproducible via
`.s6b-1-runs/scan_nan.sh` / `scan_nan.py`):

- **Server heihe (primary, `CRYOSPHERE=1`, 1-day truncation)**: 0 NaN
  / 0 Inf across 7 produced output binaries (`DY.dat`,
  `Debug_Table_Element.csv`, `Debug_Table_River.csv`, `heihe.SHUD`,
  `heihe.flood.csv`, `heihe.rivqdown.dat`, `heihe.time.csv`). 0 NaN
  / 0 Inf hits in Slurm stdout + stderr grep. Slurm 三铁律 job
  `8627`, partition `CPU`, node `cn07`, state `COMPLETED`,
  ExitCode `0:0`, elapsed `00:06:04`. Logs:
  `.s6b-1-runs/server/heihe_1day_8627.{out,err,_scan.log}`,
  `.s6b-1-runs/server/run_heihe_s6b1.sbatch`.
- **Mac keliya (auxiliary, `CRYOSPHERE=1`, 1-day truncation)**: 0 NaN
  / 0 Inf across 9 produced output binaries (`keliya.rivqdown.dat`,
  `keliya.elevnetprcp.dat`, `keliya.elevprcp.dat`, `DY.dat`,
  `Debug_Table_Element.csv`, `Debug_Table_River.csv`,
  `keliya.flood.csv`, `keliya.time.csv`, `keliya.SHUD`). 0 NaN / 0
  Inf hits in stdout + stderr grep
  (`.s6b-1-runs/keliya_1day_stdout.log`,
  `.s6b-1-runs/keliya_1day_stderr.log`,
  `.s6b-1-runs/keliya_1day_scan.log`).
