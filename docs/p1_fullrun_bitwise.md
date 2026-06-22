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

---

## Server section (heihe + heihe_x4) — PR-J #221

### Scope (spec L136-L142; tasks 5.5-5.6)

Per `openspec/changes/p1-update-omp/tasks.md` L57-L58 (server full-run
bitwise vs B1b @ NUM_OPENMP=1):

> 5.5 server cn0X 跑 `tools/archive_b0_output.sh heihe 3` + canonical
>     summary SHA ≡ B1b golden
> 5.6 server cn0X 跑 `tools/archive_b0_output.sh heihe_x4 3` + canonical
>     summary SHA ≡ B1b golden

Precision level: **A1 (refactor equivalence)** per master plan §2.2 —
serial `shud` on server GCC strict-FP toolchain must reproduce the
B1b `sha256_run1` bitwise on **2 server case** (heihe NumEle=6335 +
heihe_x4 ~25k via rSHUD v2.5 4× mesh refine). Same algorithm + same
script as Mac PR-I (`tools/archive_b0_output.sh`), exercising the
3-pragma stack at NUM_OPENMP=1 where pragmas are no-op (serial build
has `-fopenmp` NOT linked — see triple-grep gate below).

This section completes the §1.1.1 server-side acceptance leg of P1:
Mac PR-I covered the 4 dev-only case for refactor equivalence; PR-J
covers the 2 production-scale case for **server-platform A1
verification**.

### Slurm three-iron-rule compliance

Per `CLAUDE.md` server policy:

1. ✓ `sbatch` submitted from `/scratch/frd_muziyao/SHUD-OpenMP/.s221-runs/`
   (not `/users/$USER` — policy block)
2. ✓ `#SBATCH --output / --error` paths under `/scratch` shared FS
   (not compute node `/tmp` — would be lost on job end, manifest as
   ExitCode 127)
3. ✓ Script + binary references (`tools/archive_b0_output.sh`,
   `SHUD/shud`) live in `/scratch`, not `/tmp`

Scratch tree: `/scratch/frd_muziyao/SHUD-OpenMP/.s221-runs/` (dot-
prefixed → auto-gitignored). Local mirror of stdout + new/golden
`repeatability.txt` under `.s2-103/pr-j/`.

### 2-case × 2-gate matrix

| case | NumEle | new canonical sha256_run1 | golden sha256_run1 | G1 self-det | G2 vs B1b | jobid | Elapsed | node |
|---|---:|---|---|---|---|---|---|---|
| heihe | 6335 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | PASS | PASS | 8794 | 00:27:18 | cn03 |
| heihe_x4 | ~25000 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | PASS | PASS | 8795 | 01:08:45 | cn03 |

**Aggregate: G1 self-determinism 2/2 PASS · G2 vs B1b 2/2 PASS = 4/4 PASS.**

Both cases archived 3 identical SHAs across runs (G1) and matched the
B1b-tag-era golden `sha256_run1` bitwise (G2) — same hash field as
recorded at original archive in `benchmarks/<case>/B0_output/repeatability.txt`.

### Per-case 3-run walls (90d × NUM_OPENMP=1, serial shud on cn03)

| case | run1 (s) | run2 (s) | run3 (s) | mean (s) |
|---|---:|---:|---:|---:|
| heihe | 545 | 534 | 537 | 539 |
| heihe_x4 | 1367 | 1366 | 1370 | 1368 |

cn03 vs original B1b archive host (Linux 6.8.0-90 / 6.8.0-49) — same
kernel family, walls within ~5% of golden-era 522/505/528s (heihe)
and 1216/1211/1214s (heihe_x4). Wall delta is unrelated to numerical
output (bitwise PASS already covers numerical equivalence).

### Server compile — strict FP triple-grep gate

`make clean && make shud` on server (`/scratch/frd_muziyao/SHUD-OpenMP/SHUD`),
log captured at `.s221-runs/make_shud_server.log`:

| flag | grep -c | required | verdict |
|---|---:|---:|---|
| `-fopenmp` | 0 | 0 (serial build, no OMP runtime) | PASS |
| `-ffp-contract=off` | 2 | ≥ 1 (disable FMA contraction) | PASS |
| `-fno-fast-math` | 2 | ≥ 1 (disable relaxed FP) | PASS |

Compile cmd (sample): `g++ -O2 -g -ffp-contract=off -fno-fast-math
-std=c++14 …` — identical FP-determinism flag set as Mac toolchain;
GCC vs clang produce different binary SHA (`3e9e5629…` server vs
`00ea9d80…` original B1b archive) but **output is bitwise identical**
because both compilers honor the same strict-FP envelope at NUM_OPENMP=1.

### Cross-link — three independent server evidence streams

1. **PR-B #213 trim bitwise vs B0-tag** (single-`rivqdown.dat` SHA;
   different hash layer than canonical summary):
   - heihe: `55abad28…`
   - heihe_x4: `f90601ef…`
2. **PR-J #221 full-run canonical summary SHA bitwise vs B1b** (this section):
   - heihe: `675c927c…b95ba8a67`
   - heihe_x4: `3fbcbd5c…fea646c06`
3. **CI `serial-baseline / build-and-compare(keliya)`** consistently GREEN
   on PR-D / PR-E / PR-F / PR-H / PR-I merges.

PR-B exercises per-file numerical fidelity at single output channel;
PR-J exercises the full archive's canonical summary (hash of hash-file)
covering all produced channels + `cvode_stats.txt`. Different hash
algorithms, same A1 baseline preservation conclusion. trim cfg.para 90d
window did not perturb numerical output → bitwise stable across PR-A/B
forcing trim plumbing + PR-D/E/F 3-pragma stack.

### Reproduce

```bash
# On server, from /scratch (Slurm three-iron-rule):
ssh -p 32099 frd_muziyao@210.77.77.22
cd /scratch/frd_muziyao/SHUD-OpenMP
git fetch origin && git checkout main && git pull --recurse-submodules origin main
# Verify pin: main HEAD ≡ fac3da0 + SHUD ≡ 07c677f
cd .s221-runs
sbatch run_heihe.sbatch       # jobid 8794
sbatch run_heihe_x4.sbatch    # jobid 8795
# After completion: each sbatch backups B0_output/repeatability.txt,
# runs archive_b0_output.sh <case> 3, validates new sha256_run1 == golden
# sha256_run1 + 3-run identical, then restores golden (keeps git tracked
# file clean). New-run repeatability saved to .s221-runs/<case>_new_repeatability.txt.
```

sbatch scripts + Slurm logs + new/golden `repeatability.txt` preserved
under `.s221-runs/` on server and mirrored to `.s2-103/pr-j/` locally
(both gitignored scratch trees).

### Signing — server section

- signed_at: 2026-06-22
- signer: DankerMu
- signed_against_outer_commit: `fac3da0` (main HEAD at PR-J start;
  PR-I capstone `docs(p1): review-loop-log — #220 PR-I capstone (Mac 4-case full-run 8/8 PASS)`)
- signed_against_SHUD_commit: `07c677f` (P1 3-pragma stack;
  PR-F `[#218 PR-F] MD_update.cpp lake loop #pragma omp parallel for`)
- server binary: `SHUD/shud` (serial, no `-fopenmp`),
  `sha256 = 3e9e56295528b0399aff928d1b44d708da87b37777ea81e0de216a3d12a975f3`
  on cn03 (Linux 6.8.0-90-generic x86_64, GCC strict FP)
- gate: A1 refactor equivalence (server serial `shud` vs B1b) — 2/2 case
  canonical summary SHA bitwise PASS, 2/2 case 3-run self-determinism
  PASS = **4/4 PASS**.
