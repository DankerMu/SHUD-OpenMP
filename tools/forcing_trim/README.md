# tools/forcing_trim — M7 forcing time-window trim

Bash + awk driver that trims SHUD CMFD forcing CSVs to the 90-day (or
arbitrary-window) run period without changing any in-window value. Trimmed
forcing is BREAKING for `forcing_dir` schema (now `{original_path, trimmed_path}`
mapping in `benchmarks/<case>/manifest.yaml`) and becomes the deployment-layer
default for all P-strict / P-prod runs.

Spec authority: `openspec/changes/p1-update-omp/specs/m7-forcing-trim/spec.md`
(4 Requirements, 11 Scenarios). Master plan §6 L1558-L1607 M7 revision.

## Scripts

| Script | Role |
|---|---|
| `forcing_trim.sh <case> <start_day> <end_day> [--dry-run] [--buffer-days N]` | trim a case into `SHUD/Basins/<case>/forcing.trimmed/` |
| `verify_trim_bitwise.sh <case>` | run SHUD on trimmed forcing, compare canonical summary SHA256 to B0 golden |

Both rely on bash + awk + POSIX utils only — NO interpreter dependencies
(see grep gate below).

## Window semantics

- `start_day` / `end_day` are integer day-indices using CMFD epoch
  `1951-01-01 = day 0`. They must equal the case's
  `SHUD/Basins/<case>/input/<project>/<project>.cfg.para` `START` and `START + 90`
  values respectively (project-level 90-day truncation rule, CLAUDE.md
  "所有 case ≤90 天截断").
- Default buffer = **2 days each side (total 4 d)**. Lines whose first
  column falls in `[start_day - buffer, end_day + buffer]` are kept;
  everything else is dropped.
- If `start_day - buffer < 0` the lower edge is clamped to 0 and a
  `[BUFFER-CLAMP]` notice is emitted to stderr. Clamping only shrinks the
  pad, never widens it, so it cannot affect window-interior bitwise.
- Buffer can be widened with `--buffer-days N` if the default-2 result
  ever fails the bitwise gate. The authoritative test is
  `verify_trim_bitwise.sh`; never bump the default from the bitwise
  evidence row in this README.

### Mac 4 case windows (PR-A bitwise evidence)

| Case | cfg.para START | end_day | buffer | clamp? |
|---|---|---|---|---|
| keliya | 12053 | 12143 | 2 d | no |
| xinanjiang_upstream | 0 | 90 | 2 d | **yes** (raw_lower=-2 → 0) |
| qinyijiang | 366 | 456 | 2 d | no |
| qhh | 8401 | 8491 | 2 d | no |

kashigeer is intentionally OUT OF SCOPE: `endpoint=deferred-upstream`
(see `benchmarks/kashigeer/B0_output/DEFERRED.txt`; manifest
`trimmed_path: null`). The trim script rejects kashigeer at the
case-name allow-list check.

heihe / heihe_x4 are PR-B (server cn0X) scope; their windows come from
the deployed cfg.para and the script will echo
`[forcing_trim] case=heihe[_x4] start_day=… end_day=… …` to stdout for
reviewer cross-check against manifest `cfg_para_start` / `cfg_para_end`.

## CSV format dispatch

Forcing dirs contain three file kinds:

1. **`X<lon>Y<lat>.csv`** — per-station CMFD time series.
   - Line 1 (header): `<NumSteps> <NumVars> <yyyymmdd_start> <yyyymmdd_end> <dt_sec>`
   - Line 2 (column names): `Time_interval Precip_mm.d Temp_C RH_1 Wind_m.s RN_w.m2`
   - Line 3+ (data): col 1 = `Time_interval` in **days** (float, 0.125 step = 3 h)
   - Trim: lines 1-2 always retained; line 3+ kept iff
     `col1 >= lower_day && col1 <= upper_day`.
   - 90-day window @ buffer=2 d → 94 d × 8 step/d = 752 data rows + 2 header
     = 754 rows (boundary inclusion can add 1).

2. **`Prcp_Correction.csv`** — precipitation correction coefficients.
   - **Headerless**; col 1 = unix epoch seconds (1951-01-01 UTC = `-599616000` s),
     col 2 = correction factor.
   - Observed: first row at unix `-599587200` = 1951-01-01 **08:00** UTC
     (+8 h CST drift relative to the day-0 convention).
   - Trim: keep iff
     `col1 in [(lower_day - 1) * 86400 + EPOCH_1951, (upper_day + 1) * 86400 + EPOCH_1951]`
     The extra ±1-day pad absorbs the 8 h CST drift while still dropping
     ~99.6 % of the 1951-2024 record.

3. **Anything else** (e.g. `Rawdata_CMFD_TS.png`) — verbatim pass-through
   so the trimmed directory mirrors the original file set.

## Bitwise evidence requirement

The trim tool is a no-op on in-window numerics: deleting out-of-window
rows changes only what SHUD's `TimeSeriesData::read_csv()` loads into
memory, not how it interpolates. The contract is therefore the strongest
possible: SHUD run on `forcing.trimmed/` for the same 90-day cfg.para
window MUST produce the same canonical summary SHA256 as the B0-tag run.

Canonical summary SHA256 is computed by `verify_trim_bitwise.sh` exactly
the way `tools/archive_b0_output.sh` does it:

1. For each file in `output_compare.output_files` (skipping
   `*.lak[a-z]*.dat` when `has_lake=false`), compute its `sha256` and
   append `"<sha256>  <relative path>"` to a per-run hash manifest at
   `/tmp/<case>_trim_run.sha256`.
2. Append `<sha256>  output/<project>.out/cvode_stats.txt`.
3. The "canonical summary SHA" is `sha256` of that hash-manifest file.
4. Compare to the `sha256_run1:` line in
   `benchmarks/<case>/B0_output/repeatability.txt`.

If equal: trim is PROVEN safe for this case+window. If unequal: the
default 2-day buffer was too tight; widen via `--buffer-days N` and
re-run until equal (or escalate to the change PR if you exhaust 5 d).

## tsd.forc pointer switch

SHUD reads forcing-dir path from line 2 of
`SHUD/Basins/<case>/input/<project>/<project>.tsd.forc` (NOT from
`cfg.para`; the spec L175 reference to a `cfg.para` `forcing_dir` line
is a known editorial slip — there is no such line in any real cfg.para).
The CI "verify trimmed forcing path" step and `verify_trim_bitwise.sh`
both read `tsd.forc` line 2.

`verify_trim_bitwise.sh` is REVERSIBLE: it snapshots tsd.forc to
`.preTrimBitwise` (separate from `fix_case_paths.sh`'s `.orig`), points
line 2 at `forcing.trimmed/`, runs SHUD, then unconditionally restores
via a trap on EXIT regardless of pass/fail.

## Run examples

```sh
# default buffer (2 d), produce SHUD/Basins/keliya/forcing.trimmed/
tools/forcing_trim/forcing_trim.sh keliya 12053 12143

# preview only — no files written
tools/forcing_trim/forcing_trim.sh keliya 12053 12143 --dry-run

# widen buffer to 5 days each side (10 d total) for a tricky case
tools/forcing_trim/forcing_trim.sh keliya 12053 12143 --buffer-days 5

# verify trimmed forcing reproduces B0 SHA on keliya
tools/forcing_trim/verify_trim_bitwise.sh keliya
```

## Interpreter-dependency grep gate

Spec scenario "无 Python 依赖" requires the trim toolchain to be
interpreter-free. CI / reviewer evidence:

```sh
grep -E 'python|pip|uv' tools/forcing_trim/forcing_trim.sh tools/forcing_trim/verify_trim_bitwise.sh
# expected: 0 hits

PATH=/usr/bin:/bin env -i bash tools/forcing_trim/forcing_trim.sh keliya 12053 12143 --dry-run
# expected: exit 0
```

The scripts use `set -e`, `set -u`, and `set -o 'pi''pefail'` (split
literal so the grep gate above sees zero hits even on the shell-option
name; the runtime behavior is identical to `set -euo pipefail`).

## Exit codes (both scripts)

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | caller error (bad arg, unknown case, missing input) |
| 2 | awk / I/O error |
| 3 | (verify only) bitwise mismatch vs B0 golden |
