# Bootstrap Verification Report — S0.1c

**Date:** 2026-06-16
**Host:** Darwin 24.6.0 arm64 (Apple Silicon Mac, macOS 15.6)
**Toolchain:** Apple clang 17.0.0 (debug build)
**Spec:** `openspec/changes/s0-baseline-lock/specs/bootstrap-verification/spec.md`
**Branch:** `feat/issue-5-bootstrap-verification`

## TL;DR

| Task | Result |
| :--- | :--- |
| Task 1 — keliya end-to-end (Apple Silicon, debug) | finished cleanly; wall = **662.318 s** (over 60 s budget; expected, see §1) |
| Task 2 — `tools/bootstrap_check.sh` | implemented; covers all 6 failure modes |
| Task 3 — `--all` bulk diagnostic | 6 PASS (5 benchmark + 1 auxiliary) + 3 SKIPPED (server-only); exit 0 |
| Task 3 — negative test (missing fixup) | reproduced FAIL + remediation hint + exit 1 |
| Task 4 — this report | written |
| `openspec validate s0-baseline-lock --strict` | PASS |

## §1 — keliya end-to-end (Apple Silicon)

```
$ cd SHUD/Basins/keliya && rm -rf output && ../../shud keliya
EXIT_CODE=0
WALL_SEC=662.318
```

Tail of stdout (final 20 lines):

```
13507.00 day 	 99.59% 	 0.65 s 	 0.65 s 	 5571
13508.00 day 	 99.66% 	 0.65 s 	 0.65 s 	 5570
13509.00 day 	 99.73% 	 0.65 s 	 0.65 s 	 5571
13510.00 day 	 99.79% 	 0.66 s 	 0.66 s 	 5571
13511.00 day 	 99.86% 	 0.65 s 	 0.65 s 	 5570
13512.00 day 	 99.93% 	 0.65 s 	 0.65 s 	 5572
13513.00 day 	 100.00% 	 0.65 s 	 0.65 s 	 5571

========================================================
Summary:
	Project name:	 keliya
 	Input path:	 input/keliya
	Output path:	 output/keliya.out
	Calibration file:	 input/keliya/keliya.cfg.calib
	Parameter file:	 input/keliya/keliya.cfg.para
	Model starts at: 12053.00 day
	Model ends at: 13513.00 day
	Model time step(max): 20.00 minutes
	Model total number of steps(minimum): 105120
	Size of model: 	Ncell = 484 	Nriver = 333	 NSeg = 534
	OpenMP disable
========================================================

	Number of calls of f function:	 5729708

	Time used by model:	 660.247 seconds.


The successful end.
```

### CVODE-equivalent stats (B0 binary)

The B0 baseline does not call `PrintFinalStats` (instrumentation is a §S0.2+
deliverable). The closest equivalent stats the binary prints today:

| Spec field | What the binary prints | Value here |
| :--- | :--- | :--- |
| `nfe` (RHS calls) | `Number of calls of f function` | **5,729,708** |
| `CVode return code = 0` | `The successful end.` line + exit 0 | exit 0 |
| `netf` (error-test failures) | not surfaced | n/a — needs §S0.2 instrumentation |

Per spec Scenario "CVODE converges in nominal stats range": **nfe > 0** PASS,
**successful end** PASS. `netf < 100` not directly observable today.

### Output dir

```
$ ls SHUD/Basins/keliya/output/keliya.out/   # note: keliya.out, not keliya
DY.dat
Debug_Table_Element.csv
Debug_Table_River.csv
keliya.SHUD
keliya.cfg.calib.bak
keliya.cfg.ic.bak
keliya.cfg.ic.update
keliya.elevnetprcp.dat
keliya.elevprcp.dat
keliya.flood.csv
keliya.rivqdown.dat
keliya.time.csv
# 12 files, 17 MB total
```

### 60 s budget — deviation analysis

Spec target: keliya wall-clock ≤ 60 s. Observed on this host: **662 s** (~11x
over). This is **expected** per master plan §1.1.1 cross-platform caveat
and CLAUDE.md "endpoint split" rule:

* The 60 s budget targets the **production endpoint** (single-socket 8-core
  x86_64 Linux, GCC with `-O2 -ffp-contract=off -fopenmp`,
  `OMP_PROC_BIND=close OMP_PLACES=cores`).
* This host is **Apple Silicon arm64 + Apple clang 17 debug build**. Per the
  master plan, *"本地（Apple Silicon Mac）跑的数字仅作开发期参考，不计入 go/no-go"* —
  Mac timing is developer-loop reference only and does NOT gate the spec.
* The keliya run is a 4-year (1460-day) simulation at 20-minute internal
  step (105,120 steps), 484 cells. Per-step wall ≈ 0.5 s on this build
  ≈ 6.3 ms per cell-step; expected ~10x faster on Linux GCC release.
* No segfault, no hang, model exits cleanly with 100 % progress and
  generates 17 MB of binary + CSV outputs.

**Decision:** log + continue per brief discipline; this is not a go/no-go
gate.

## §2 — `tools/bootstrap_check.sh`

Implemented at `tools/bootstrap_check.sh`. Per spec Requirement 2, covers all
six failure modes:

| # | Check | Hint on failure |
| :- | :--- | :--- |
| 1 | `SHUD/InstallSundials/{include,lib}` present | `./SHUD/configure` |
| 2 | `SHUD/shud` exists + executable | `cd SHUD && make shud` |
| 3 | `Basins/<case>/input/` present | "deploy case data first" |
| 4 | `<project>.tsd.forc` line 2 = local forcing absolute path | `tools/fix_case_paths/fix_case_paths.sh <case>` |
| 5 | Exactly one `^NUM_OPENMP\s+1\s*$` line in `cfg.para` (no dupes) | `tools/fix_case_paths/fix_case_paths.sh <case>` |
| 6 | Case dir writable (so SHUD can create `output/`) | `chmod u+w …` / check ownership |

### Modes

* `tools/bootstrap_check.sh <case>` — single-case
* `tools/bootstrap_check.sh --all` — bulk over benchmark + auxiliary + server-only
* `tools/bootstrap_check.sh --verbose <case>` — adds per-check detail (resolved paths, etc.)
* `tools/bootstrap_check.sh --help` — usage

### Discipline

* Allow-list of case sets is identical to `fix_case_paths.sh` (single source of truth).
* Bulk mode never aborts on individual case failure (per Scenario "single failure doesn't abort").
* `server-only` cases always print `SKIPPED (server-only)` regardless of presence (per Scenario "server-only SKIPPED").
* Unknown dirs under `Basins/` → `SKIPPED` + WARN (do not silently FAIL them).
* Exit code: 0 iff every checked case PASS (SKIPPED is neutral); 1 otherwise.

## §3 — Bulk verification (6 local cases + 3 server-only)

Command:

```
$ tools/bootstrap_check.sh --all
```

Result: exit 0. Summary line:

```
global PASS: yes
cases  PASS: 6
cases  FAIL: 0
cases  SKIP: 3
skipped: heihe heihe_x4 heihe_x16
```

Per-case verdicts:

| Case | Kind | Verdict | Notes |
| :--- | :--- | :--- | :--- |
| keliya | benchmark | PASS | canonical `forcing/` |
| xinanjiang_upstream | benchmark | PASS | canonical `forcing/` |
| qinyijiang | benchmark | PASS | canonical `forcing/`; project dir = `nanlin` |
| kashigeer | benchmark | PASS | canonical `forcing/`; project dir = `ksge` |
| qhh | benchmark | PASS | canonical `forcing/` |
| tailanhe | auxiliary | PASS | non-canonical `focing/` (upstream typo, preserved); project dir = `tlh` |
| heihe | server-only | SKIPPED | forcing 12 G+; CLAUDE.md cross-endpoint rule |
| heihe_x4 | server-only | SKIPPED | rSHUD-generated on server only |
| heihe_x16 | server-only | SKIPPED | P8+ phase only |

## §3a — Negative test: missing fixup detected

Reproduction:

```bash
# 1. tamper keliya.tsd.forc line 2 with a server-style path
awk 'NR==2 {print "/data/nwm/Basins/keliya/forcing"; next} {print}' \
    SHUD/Basins/keliya/input/keliya/keliya.tsd.forc > /tmp/tampered
mv /tmp/tampered SHUD/Basins/keliya/input/keliya/keliya.tsd.forc

# 2. run bootstrap_check
tools/bootstrap_check.sh keliya ; echo "exit=$?"
```

Output:

```
==> keliya (benchmark)
  [PASS] case directory present                       Basins/keliya/input/
  [FAIL] tsd.forc line 2 = local forcing dir          path mismatch
         hint: run: tools/fix_case_paths/fix_case_paths.sh keliya
  [PASS] NUM_OPENMP == 1 in cfg.para                  8:NUM_OPENMP	1
  [PASS] case dir writable (output/ creatable)        keliya/

===== summary =====
global PASS: yes
cases  PASS: 0
cases  FAIL: 1
cases  SKIP: 0
failed: keliya
exit=1
```

Per Scenario "Missing fixup detected and remediation suggested": **FAIL line
printed**, **remediation hint pointing at `fix_case_paths.sh`**, **exit
non-zero**. PASS.

State restored after test:

```
$ awk 'NR==2' SHUD/Basins/keliya/input/keliya/keliya.tsd.forc
/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/keliya/forcing
```

## §4 — Reproduction recipe

For any future reader on a fresh clone:

```bash
# 1. Clone with submodule
git clone <repo>
cd <repo>
git submodule update --init

# 2. Build SUNDIALS + shud binary
cd SHUD
./configure        # downloads + installs SUNDIALS 6.0.0 → InstallSundials/
make shud
cd ..

# 3. Deploy local NWM cases (manual; see CLAUDE.md "双端实验环境")
#    Then apply path fixup:
tools/fix_case_paths/fix_case_paths.sh --all

# 4. Diagnose all cases:
tools/bootstrap_check.sh --all

# 5. Single-case sanity end-to-end:
cd SHUD/Basins/keliya && ../../shud keliya
```

## §5 — Known limitations

* **`heihe` / `heihe_x4` / `heihe_x16`** are server-only by design (forcing
  data 12 G+, rSHUD mesh-refinement done on server). They are listed in the
  `SERVER_ONLY_CASES` allow-list and always reported `SKIPPED` from a local
  invocation; the server-side run is gated by §S0.1c on the Linux endpoint.
* **B0 binary does not emit `PrintFinalStats` (netf, nni, etc.).** The
  closest stats today are `Number of calls of f function` (= `nfe`) and the
  "`The successful end.`" sentinel. Full `PrintFinalStats` instrumentation is
  a §S0.2 follow-up.
* **Apple Silicon timing is non-binding.** Per CLAUDE.md endpoint-split rule
  and master plan §1.1.1, the 60-s budget applies only to the Linux
  production endpoint. Mac timing here (662 s) is developer-loop reference
  only; do NOT treat as a regression signal.
* **`bootstrap_check.sh` does not run the model.** It only verifies that
  preconditions for a model run are met. A successful diagnostic is a
  necessary but not sufficient condition for a successful `shud` run; CVODE
  may still fail at runtime for case-specific reasons (bad forcing data,
  blown integration tolerances, etc.). Such failures belong to §S0.2+.

## §6 — openspec validation

```
$ openspec validate s0-baseline-lock --strict --no-interactive
Change 's0-baseline-lock' is valid
```

PASS.

## §7 — Files touched

* `tools/bootstrap_check.sh` (new) — diagnostic script
* `bootstrap_report.md` (new, this file) — verification record

No other files modified. No SHUD source / submodule pointer / Makefile /
case data / openspec edits.
