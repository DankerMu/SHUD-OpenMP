# P1 full-run output regression bitwise validation — Mac 4 case

## Scope (tasks.md L52-L55 / spec L143-L161)

Per `openspec/changes/p1-update-omp/tasks.md` L54:

> 5.3 Mac 本地 4 case 用 P1 候选 commit 跑 `tools/archive_b0_output.sh <case> 3`
>     完成 3-run 自洽 + 完整 run canonical summary SHA ≡ B1b/B1-tag baseline
>     + CVODE 15-key stats identical

Precision level: **A1 (refactor equivalence)** per master plan §2.2 —
serial `shud` binary built off the P1 candidate commit must reproduce the
B1b canonical summary SHA bitwise. This is the same canonical-SHA gate
applied at B0/B1a/B1b archival (`docs/status_matrix.md` L13/L64/L93/L132/L155).

- 4 Mac case × `tools/archive_b0_output.sh <case> 3`
- Two gates per case:
  - **G1 self-determinism**: 3-run canonical summary SHA identical
  - **G2 vs B1b**: canonical summary SHA ≡
    `benchmarks/<case>/B0_output/repeatability.txt sha256_run1`
- SHUD pin: `07c677f` (3-pragma stack — PR-D element + PR-E river + PR-F lake)
- Binary: `SHUD/shud` (serial; built `make shud`).  CI
  `serial-baseline / build-and-compare` exercises this same binary +
  same gate per `.github/workflows/serial-baseline.yml` L699/L917.

## Tag chain & golden source

- B0 ≡ B1a ≡ B1b ≡ B1 per `docs/status_matrix.md` L107-L113 +
  L155 (4 Mac case canonical SHA chain).
- Golden source: `benchmarks/<case>/B0_output/repeatability.txt` field
  `sha256_run1` (set at B0-tag archive, frozen through B1a/B1b).
- Canonical SHA algorithm (= `tools/archive_b0_output.sh` lines 109-117 +
  318-362): per-file SHA256 over the manifest `output_files` set that
  was actually produced by the run + `cvode_stats.txt`, written one
  `<hash>  <path>` per line to `/tmp/<case>_run<N>.sha256`; the canonical
  summary is SHA256 of that hash-file's bytes.

## Archive subset honest-disclosure

The manifest `output_compare.output_files` field lists 6 / 6 / 6 / 9 dat
files per case (= §S0.13 spec-layer forward placeholder).  `cfg.para
DT_*=0` disables most channels by deployment default, so the **physically
produced** archive subset captured by the canonical SHA is:

| case | files actually hashed under canonical SHA |
|---|---|
| keliya | `keliya.rivqdown.dat` + `cvode_stats.txt`                            (2) |
| xinanjiang_upstream | `xinanjiang.rivqdown.dat` + `xinanjiang.eleygw.dat` + `cvode_stats.txt` (3) |
| qinyijiang | `nanlin.rivqdown.dat` + `cvode_stats.txt`                            (2) |
| qhh | `qhh.rivqdown.dat` + `qhh.lakystage.dat` + `qhh.lakqrivin.dat` + `qhh.lakqrivout.dat` + `cvode_stats.txt` (5) |

The remaining manifest entries (`eleysurf`/`eleyunsat`/`eleysnow`/`eleygw`
where DT_*=0, `rivystage`, `flood.csv`) are not produced by SHUD with
current `cfg.para` settings — `archive_b0_output.sh` (line 340-348) +
B0-tag `missing_manifest_files` field record this pre-existing NWM
deployment gap (see S0-4 / issue #11).  The single canonical summary SHA
covers what is actually produced; that is the historical B0/B1a/B1b
gate and what `sha256_run1` records.

## 4-case × 2-gate matrix

| case | new canonical SHA | golden sha256_run1 | G1 self-det | G2 vs B1b |
|---|---|---|---|---|
| keliya | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | PASS | PASS |
| xinanjiang_upstream | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | PASS | PASS |
| qinyijiang | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | PASS | PASS |
| qhh | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | PASS | PASS |

**Aggregate: G1 self-determinism 4/4 PASS · G2 vs B1b 4/4 PASS = 8/8 PASS.**

## CVODE 15-key stats identical (issue #220 acceptance L9)

`tools/cvode_stats_diff/cvode_stats_diff.sh <new> <golden>` against
each case's new `cvode_stats.txt` (from run 3 — the last run of the
`archive_b0_output.sh <case> 3` invocation) vs
`benchmarks/<case>/B0_output/cvode_stats.txt`:

| case | cvode_stats_diff exit | verdict |
|---|---:|---|
| keliya | 0 | PASS |
| xinanjiang_upstream | 0 | PASS |
| qinyijiang | 0 | PASS |
| qhh | 0 | PASS |

All 15 canonical CVODE keys (`nfe / nfeLS / nni / nli / nsetups / netf
/ nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS`,
per `tools/cvode_stats_diff/canonical_15_keys.yaml` + design D10) byte-
identical between new run and B1b golden across all 4 case.

## Per-case wall times (90d × 3 run, NUM_OPENMP=1, serial shud)

| case | run1 (s) | run2 (s) | run3 (s) |
|---|---:|---:|---:|
| keliya | 28 | 28 | 27 |
| xinanjiang_upstream | 4 | 5 | 4 |
| qinyijiang | 242 | 239 | 238 |
| qhh | 83 | 86 | 91 |

(host: Darwin 24.6.0 arm64, Apple Silicon Mac local; dev-only numbers,
not §1.1.1 acceptance.)

## Cross-link — three independent evidence streams for A0/A1 verification

1. **PR-H canonical RHS snapshot bitwise** (mid-state, `docs/p1_rhs_snapshot_bitwise.md`)
   — 12/12 PASS for Mac 4 case at NUM_OPENMP=1 vs B1b RHS dumps
   (`shud_omp` with `SHUD_DUMP_RHS=1`).
2. **PR-I full-run canonical summary SHA bitwise** (this document)
   — 4/4 PASS for Mac 4 case at NUM_OPENMP=1 vs B1b archived
   `sha256_run1` (`shud` serial via `tools/archive_b0_output.sh`).
3. **CI `serial-baseline / build-and-compare(keliya)`** has gone GREEN
   on every PR-D / PR-E / PR-F / PR-H merge (≥ 7 GREEN runs) using the
   same `make shud` + per-`*.dat` SHA256 gate
   (`.github/workflows/serial-baseline.yml` L905-L955).

Combined, A0/A1 baseline preservation under the 3-pragma stack
(PR-D element loop + PR-E river loop + PR-F lake loop) is corroborated by
three orthogonal gates: mid-RHS snapshot bitwise, end-of-run canonical
summary bitwise, and per-file CI byte-compare.

## OMP-runtime sub-note (not a gate)

For completeness — `shud_omp @ OMP_NUM_THREADS=1` (the OpenMP runtime
binary with `-fopenmp` linked + the 3-pragma stack active) does NOT
match the serial `sha256_run1` golden bitwise across the 4 cases
(observed canonical SHAs: keliya `9365d812…`, xj_up `b49722dc…`,
qinyijiang `98a91323…`, qhh `8ef63a56…`; all 3-run self-deterministic).
This is expected and not a regression:

- B0/B1a/B1b golden was archived from the serial `shud` binary
  (`binary: …/SHUD/shud` recorded in each `repeatability.txt`).
- `shud_omp` links the OMP runtime and activates `#pragma omp parallel
  for` regions even at one thread — the reduction tree, loop scheduler,
  and FMA selection differ from the no-`-fopenmp` serial path.
- Master plan §2.2 A0 / A1 gates compare **serial vs serial**; OMP-binary
  full-run bitwise is the A3a gate (P7 strict), and even there §2.2
  requires "**same-thread** bitwise" (NUM_OPENMP=N vs NUM_OPENMP=N),
  not NUM_OPENMP=1 (OMP) vs serial.
- The OMP-runtime `shud_omp @ N=1` is exercised against B1b at the
  RHS snapshot (mid-state) level by PR-H (12/12 PASS) — that is the
  P1 / A2 precision gate per master plan §2.2.

This sub-note documents the observed numerical equality boundary
between the two binaries; it does not enter the PR-I verdict.

## Reproduce

```bash
cd /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD && make shud
cd /Users/danker/Desktop/Hydro-SHUD/openMP
for c in keliya xinanjiang_upstream qinyijiang qhh; do
  tools/fix_case_paths/fix_case_paths.sh "$c"
  tools/archive_b0_output.sh "$c" 3
  # archive script writes B0_output/repeatability.txt with sha256_run1..3;
  # PASS iff sha256_run1 == golden's sha256_run1 (pre-script snapshot).
done
```

Driver script + per-case archive logs preserved under
`.s2-103/pr-i/` (gitignored scratch tree).

## Signing

- signed_at: 2026-06-22
- signer: DankerMu
- signed_against_outer_commit: `3fb14ee` (main HEAD at PR-I start;
  PR-H capstone `docs(p1): review-loop-log — #219 PR-H capstone`)
- signed_against_SHUD_commit: `07c677f` (P1 3-pragma stack;
  PR-F `[#218 PR-F] MD_update.cpp lake loop #pragma omp parallel for`)
- gate: A1 refactor equivalence (serial `shud` vs B1b) — 4/4 case
  canonical summary SHA bitwise PASS, 4/4 case 3-run self-determinism
  PASS = **8/8 PASS**.
