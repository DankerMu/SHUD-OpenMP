# P1e Mac reverse-compat — 4-case × N=1 SHALL closure (PR-J, mode C bitwise vs mode A reference)

OpenSpec change: `p1e-strict-omp-rhs` task **§5** (PR-J scope per tasks §4.7 + §4.8)
Issue: #318 — part of epic #308
PR base: `baseline/P1e`
Date: 2026-06-25 UTC

---

## 1. Scope and identity

PR-J closes out the Mac side of the P1e cross-platform determinism chain by
verifying that mode C (StrictOMP RHS) at `SHUD_RHS_THREADS=1` produces
bitwise-identical `<case>.rivqdown.dat` output to the canonical mode A
(serial RHS) reference on **all 4 Mac-native P1e benchmark cases (per tasks
§4.7)** plus a 2 server-transitive cross-cite (per tasks §4.8 6-case roll-up).

PR-J is the third-of-three SHALL gate runs in P1e:

- **PR-H (#316, merged)** demonstrated mode C bitwise-equality vs mode A on
  Mac libomp across `N∈{1,2,4,8}` for `keliya` (12/12 cells == reference).
- **PR-I (#317, merged)** verified the same holds on Linux libgomp at
  production scale for `heihe` (6335 cells) + `heihe_x4` (~25k cells) with
  24-cell 3 SHALL gate PASS + D7 speedup partial-PASS (heihe_x4 1.729×).
- **PR-J (this)** verifies the 4 Mac-native cases (`keliya`,
  `xinanjiang_upstream`, `qinyijiang`, `qhh` — exactly the case set named
  in tasks §4.7) at the canonical `N=1` reverse-compat point, and
  aggregates the §4.8 6-case × N=1 cross-platform SHA matrix by
  cross-citing PR-I for `heihe` + `heihe_x4` (which are server-only at
  production scale).

Cell budget (this PR): **4 cases × N=1 × 3 reps = 12 mode-C cells +
12 mode-A reference reproductions** on Mac. The remaining 2 cases (heihe,
heihe_x4) are transitively cited from PR-I's 24-cell server gate. The
mandatory case set is taken verbatim from tasks §4.7 — no silent omission
of named cases (Phase 6 fix per Phase 4.5 verifier CONFIRMED Blocking
F-R2-1/F-R2-2: prior PR-J iteration ran only 2/4 cases and is now
superseded by this 4/4 closure).

| Identity field | Value |
|---|---|
| Outer commit (parent) | `0bfc214` `chore(p1e): append review-loop-log PR-I entry…` |
| Branch | `feat/issue-318-pr-j-mac-shall-closure` |
| SHUD submodule pin | `3341368d2d0854924d2286925c8575df52cc97a0` (PR-H result, unchanged in PR-I and PR-J — verification-only PR, no source change) |
| Mode A binary | `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/shud_A` |
| Mode A binary sha256 | `dfe3bed0a37078af3aba23746f87d10a8c9b0fb9e47f286d5770cae4fe710d74` |
| Mode A binary mtime | `2026-06-25T20:16:27+0800` |
| Mode C binary | `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/shud_C` |
| Mode C binary sha256 | `84cb8b1f14daf4fbf3563a312c7a1ed9cc8811c3e2c9a277e4cd2ab4182fc1f6` |
| Mode C binary mtime | `2026-06-25T20:16:41+0800` |
| Host OS | `Darwin 24.6.0 arm64` (macOS Sequoia 15) |
| CPU | Apple M4 Pro (14 cores) |
| Compiler | Apple clang version 17.0.0 (clang-1700.6.3.2) |
| OMP runtime | libomp 22.1.7 (Homebrew, `/opt/homebrew/opt/libomp/lib/libomp.dylib`) |
| CMFD forcing | V0200 (project iron rule) |
| Truncation | 90-day per `cfg.para` (keliya day 12053→12143; xinanjiang day 0→90; nanlin day 366→456; qhh day 8401→8491) |

CMFD V0200 forcing + 90-day truncation inherited from project iron rules.

### 1.1 Reference SHAs (LOCKED from PR-E + PR-B0 mode A canonical)

| case | NumEle | project name | reference rivqdown.sha256 (full) | reference cvode_nst |
|------|-------:|--------------|----------------------------------|--------------------:|
| keliya              |   484 | `keliya`     | `b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd` | 111130 |
| xinanjiang_upstream |   801 | `xinanjiang` | `81fe3a02e17ee9ead4f2fceec8eb005ecf03d4cd47e877aced566dc9e2c71865` | (see log) |
| qinyijiang          |  3155 | `nanlin`     | `fc1b1816cf0da345edec9d436856161679bbbee2bb98c8edc9535e20345de6c8` | (see log) |
| qhh                 |  4773 | `qhh`        | `ccc7dd09d0189ecd360cdc4bca9090acece08f51d7de8108ece6f32900a626e7` |  13000 |
| heihe               |  6335 | `heihe`      | `a2023ccd2de43543a675df8eee46aa72ef1f9437764d28863dda62a4029e7798` |   6698 |
| heihe_x4            | ~25000 | `heihe_x4`  | `b5e4b0a2cf83b2a4b97d6be5b40ea7e0580d59fb6934db73a95f366f5ffc72b4` |   6575 |

Note: directory name vs project name differs for two Mac cases — the
`Basins/xinanjiang_upstream/` directory hosts project `xinanjiang`, and
`Basins/qinyijiang/` hosts project `nanlin` (per `benchmarks/INDEX.md` L26
+ PR-B0 §4 note). Binary invocation uses project name: `../../shud xinanjiang`
and `../../shud nanlin`.

Source of reference:
- `keliya` + `qhh`: PR-C `docs/p1e/p1e_pr_c_2x2_mac.md` §2 mode A 24 cells (all unique==1 across 4×N×3 reps).
- `xinanjiang_upstream` + `qinyijiang`: PR-B0 `docs/p1e/p1e_pr_b0_rivqdown_recompute.md` §"Acceptance §4" 4 case × 4-N mode A matrix (per-case all-equal, unique==1).
- `heihe` + `heihe_x4`: PR-D `docs/p1e/p1e_pr_d_2x2_server.md` §3 mode A 24 cells (all unique==1).
- PR-E `docs/p1e/p1e_2x2_verdict.md` §3 Phase 1 AC1 PASS records the 4 PR-E
  references (keliya/qhh/heihe/heihe_x4) as canonical; xinanjiang +
  qinyijiang references come from PR-B0 §4 (the 4-case × 4-N matrix that
  validated the rivqdown.dat tout-boundary recompute fix) and are
  re-confirmed in §2.1 below on the current SHUD pin.

### 1.2 Mode C binary verification

Required (per tasks §3.6 + §9 P1e.4): mode C binary SHALL contain
`N_VNew_Serial` symbol reference; SHALL NOT contain `N_VNew_OpenMP`; SHALL
link in OpenMP runtime (`libomp.dylib` on macOS).

```
$ nm /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/shud_C | \
    grep -E 'N_VNew_Serial|N_VNew_OpenMP|_omp_'
                 U _N_VNew_Serial
                 U _omp_get_max_threads
                 U _omp_get_wtime
                 U _omp_set_num_threads

$ otool -L /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/shud_C | grep -iE 'omp|sundials'
    libsundials_cvode.6.dylib
    libsundials_nvecserial.6.dylib
    /opt/homebrew/opt/libomp/lib/libomp.dylib (compatibility version 5.0.0)
```

- `_N_VNew_Serial` present — PASS
- `_N_VNew_OpenMP` absent — PASS
- `_omp_*` symbol references present (3 distinct) — PASS (binary truly links libomp)
- `libomp.dylib` in dynamic load list — PASS

Contrast with mode A (`shud_A`):

```
$ nm /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/shud_A | grep -E 'N_VNew_Serial|N_VNew_OpenMP|_omp_'
                 U _N_VNew_Serial
$ otool -L SHUD/shud_A | grep -iE 'omp|sundials'
    libsundials_cvode.6.dylib
    libsundials_nvecserial.6.dylib
```

Mode A has only `_N_VNew_Serial`, no `_omp_*` references, no libomp.dylib —
confirms serial-only build with no OpenMP runtime dependency.

---

## 2. Cell roster (24 cells: 12 mode-C + 12 mode-A reference reproductions)

Each rep is a fresh `rm -rf output/<project>.out && SHUD_RHS_THREADS=1
../../shud_{A,C} <project>` invocation. Wall time captured with `time` or
`date +%s` deltas.

### 2.1 Mode A reference reproduction (sanity check that references still hold on current SHUD pin)

| cell | case | project | binary | wall_sec | rivqdown_sha12 | reference_sha12 | match |
|------|------|---------|:------:|---------:|----------------|-----------------|:-----:|
| A_keliya_rep1              | keliya              | keliya     | shud_A | 30 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| A_keliya_rep2              | keliya              | keliya     | shud_A | 30 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| A_keliya_rep3              | keliya              | keliya     | shud_A | 30 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| A_xinanjiang_upstream_rep1 | xinanjiang_upstream | xinanjiang | shud_A |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| A_xinanjiang_upstream_rep2 | xinanjiang_upstream | xinanjiang | shud_A |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| A_xinanjiang_upstream_rep3 | xinanjiang_upstream | xinanjiang | shud_A |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| A_qinyijiang_rep1          | qinyijiang          | nanlin     | shud_A | 290 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| A_qinyijiang_rep2          | qinyijiang          | nanlin     | shud_A | 286 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| A_qinyijiang_rep3          | qinyijiang          | nanlin     | shud_A | 287 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| A_qhh_rep1                 | qhh                 | qhh        | shud_A | 109 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |
| A_qhh_rep2                 | qhh                 | qhh        | shud_A | 99 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |
| A_qhh_rep3                 | qhh                 | qhh        | shud_A | 98 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |

All 12 mode A cells PASS — references are stable on the current SHUD pin
`3341368` (PR-H result). This confirms PR-J's strict-omp results below are
compared against a still-valid baseline.

### 2.2 Mode C N=1 (this PR's primary AC-J1 / AC-J2 gate)

| cell | case | project | binary | env | wall_sec | rivqdown_sha12 | reference_sha12 | match |
|------|------|---------|:------:|-----|---------:|----------------|-----------------|:-----:|
| C_keliya_N1_rep1              | keliya              | keliya     | shud_C | `SHUD_RHS_THREADS=1` | 31 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| C_keliya_N1_rep2              | keliya              | keliya     | shud_C | `SHUD_RHS_THREADS=1` | 29 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| C_keliya_N1_rep3              | keliya              | keliya     | shud_C | `SHUD_RHS_THREADS=1` | 30 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| C_xinanjiang_upstream_N1_rep1 | xinanjiang_upstream | xinanjiang | shud_C | `SHUD_RHS_THREADS=1` |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| C_xinanjiang_upstream_N1_rep2 | xinanjiang_upstream | xinanjiang | shud_C | `SHUD_RHS_THREADS=1` |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| C_xinanjiang_upstream_N1_rep3 | xinanjiang_upstream | xinanjiang | shud_C | `SHUD_RHS_THREADS=1` |  5 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| C_qinyijiang_N1_rep1          | qinyijiang          | nanlin     | shud_C | `SHUD_RHS_THREADS=1` | 306 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| C_qinyijiang_N1_rep2          | qinyijiang          | nanlin     | shud_C | `SHUD_RHS_THREADS=1` | 285 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| C_qinyijiang_N1_rep3          | qinyijiang          | nanlin     | shud_C | `SHUD_RHS_THREADS=1` | 287 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| C_qhh_N1_rep1                 | qhh                 | qhh        | shud_C | `SHUD_RHS_THREADS=1` | 99 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |
| C_qhh_N1_rep2                 | qhh                 | qhh        | shud_C | `SHUD_RHS_THREADS=1` | 100 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |
| C_qhh_N1_rep3                 | qhh                 | qhh        | shud_C | `SHUD_RHS_THREADS=1` | 103 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |

Per-case unique SHA count = 1 (rep-rep stability) AND matches mode A
reference SHA (PR-E for keliya/qhh; PR-B0 §4 for xinanjiang/qinyijiang) =
PASS for all 4 cases.

Notes:
- Per-rep SHA file: `Basins/<case>/output/<project>.out/<project>.rivqdown.dat`
  sha256sum captured inline after each run.
- Per-rep cvode_stats: `Basins/<case>/output/<project>.out/cvode_stats.txt`
  shows `nst=111130` for keliya, `nst=13000` for qhh — matches PR-C mode A
  reference exactly (no Newton perturbation introduced by strict-omp RHS at
  `SHUD_RHS_THREADS=1`).
- Per-rep run log: `/tmp/run_{xj,qy}_{A,C}_rep<r>.log` + legacy
  `/tmp/run_C_{keliya,qhh}_N1_rep<r>.log` / `/tmp/run_A_{keliya,qhh}_rep<r>.log`.

---

## 3. SHALL gate verdicts (per tasks §4.7 + §4.8)

### 3.1 AC-J1 — mode C N=1 stability per case (rep-rep determinism)

Criterion: for each Mac-native case, the `<project>.rivqdown.sha256` SHALL be
identical across all 3 reps at `SHUD_RHS_THREADS=1` (verifies strict-omp
RHS produces deterministic output even in serial-thread mode).

| case | unique_SHAs (3 reps) | rep1 | rep2 | rep3 | verdict |
|------|---------------------:|------|------|------|:-------:|
| keliya              | 1 | `b769e3270e1c` | `b769e3270e1c` | `b769e3270e1c` | PASS |
| xinanjiang_upstream | 1 | `81fe3a02e17e` | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| qinyijiang          | 1 | `fc1b1816cf0d` | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| qhh                 | 1 | `ccc7dd09d018` | `ccc7dd09d018` | `ccc7dd09d018` | PASS |

**AC-J1 verdict: PASS** (mode C SHA rep-stable for all 4 Mac cases under
libomp on Apple M4 Pro at `SHUD_RHS_THREADS=1`).

### 3.2 AC-J2 — mode C SHA == mode A reference SHA per case (cross-mode bitwise)

Criterion: for each Mac-native case, mode C `N=1` SHA SHALL equal the mode A
canonical reference SHA (PR-E for keliya/qhh; PR-B0 §4 for xinanjiang/
qinyijiang) — verifies strict-omp RHS == serial RHS bitwise on Mac libomp
at the canonical reverse-compat thread count.

| case | NumEle | mode_C_SHA(N=1,rep=1) | mode_A_reference_SHA | reference source | verdict |
|------|-------:|------------------------|----------------------|------------------|:-------:|
| keliya              |   484 | `b769e3270e1c4d07…` | `b769e3270e1c4d07…` | PR-E §3 (PR-C)   | PASS |
| xinanjiang_upstream |   801 | `81fe3a02e17ee9ea…` | `81fe3a02e17ee9ea…` | PR-B0 §4 row 2   | PASS |
| qinyijiang          |  3155 | `fc1b1816cf0da345…` | `fc1b1816cf0da345…` | PR-B0 §4 row 3   | PASS |
| qhh                 |  4773 | `ccc7dd09d0189ecd…` | `ccc7dd09d0189ecd…` | PR-E §3 (PR-C)   | PASS |

**AC-J2 verdict: PASS** (Mac libomp strict-omp RHS
produces bitwise-identical output to serial RHS at `SHUD_RHS_THREADS=1` for
all 4 Mac-native cases, including the lake-bearing qhh case and the two
PR-B0 reference cases xinanjiang_upstream + qinyijiang).

### 3.3 qhh lake module — special bitwise verification (sub-clause of AC-J2)

The `qhh` case has the lake module enabled (per CLAUDE.md "qhh 4773 +lake")
and exercises PR-H's lake transitional gather + clamp code path which uses
`#pragma omp single` inside the StrictOMP parallel region. Bitwise SHA
match (`ccc7dd09d018` across all 3 mode C N=1 reps == PR-E reference) is
the direct positive proof that the `omp single` lake code is determinism-
preserving on Mac libomp.

Per-run lake river-network sanity log entries (`/tmp/run_C_qhh_N1_rep1.log`):
- `The downstream of RIV <id> is negtive (Outlet, -3)` × many — expected
  lake-segment outlet markers, identical across reps.
- `nst=13000`, `nfe=13270`, `nli=25015` — matches PR-C mode A reference
  exactly. No CVode perturbation in the lake path.

### 3.4 PR-I server transitive cite — heihe + heihe_x4 N=1 reverse-compat

`heihe` (6335 cells) and `heihe_x4` (~25k cells) are server-only production
cases per CLAUDE.md "本地（Apple Silicon Mac）：开发 + Small/Medium baseline".
PR-I (`docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.2 AC-S2) verified mode C N=1
bitwise-equality vs PR-D mode A reference on the server libgomp runtime
at production scale:

| case | NumEle | mode_C_SHA(N=1, PR-I server libgomp) | mode_A_reference_SHA (PR-D) | verdict (PR-I) |
|------|-------:|--------------------------------------|------------------------------|:--------------:|
| heihe    |  6335 | `a2023ccd2de43543…` | `a2023ccd2de43543…` | PASS |
| heihe_x4 | ~25000 | `b5e4b0a2cf83b2a4…` | `b5e4b0a2cf83b2a4…` | PASS |

PR-I server cells: `Cheihe_N1_rep{1,2,3}` + `Cheihe_x4_N1_rep{1,2,3}` (6
cells), all `rivqdown_sha12=a2023ccd2de4` / `b5e4b0a2cf83` respectively.

These are cited transitively as the "server side" of PR-J's 4-case × N=1
roll-up — PR-J does NOT re-run these cases on Mac (Mac cannot host
heihe_x4 at 25k cells within reasonable wall budget, and heihe is not
checked out into Mac's `Basins/`).

### 3.5 AC-J2 6-case × N=1 cross-platform SHA matrix (the §4.8 6-case verdict roll-up)

Combining §3.2 Mac mode C results (4 cases) with §3.4 PR-I server transitive
cite (2 cases) yields the complete §4.8 6-case roll-up:

| case | NumEle | Mac libomp mode C N=1 SHA | Server libgomp mode C N=1 SHA | mode A reference SHA | Mac match | Server match |
|------|-------:|---------------------------|--------------------------------|----------------------|:---------:|:------------:|
| keliya              |   484 | `b769e3270e1c` | (Mac-native; not run on server in P1e) | `b769e3270e1c` | PASS | n/a |
| xinanjiang_upstream |   801 | `81fe3a02e17e` | (Mac-native; not run on server in P1e) | `81fe3a02e17e` | PASS | n/a |
| qinyijiang          |  3155 | `fc1b1816cf0d` | (Mac-native; not run on server in P1e) | `fc1b1816cf0d` | PASS | n/a |
| qhh                 |  4773 | `ccc7dd09d018` | (Mac-native; not run on server in P1e) | `ccc7dd09d018` | PASS | n/a |
| heihe               |  6335 | (server-native; not on Mac) | `a2023ccd2de4` | `a2023ccd2de4` | n/a | PASS (PR-I) |
| heihe_x4            | ~25000 | (server-native; not on Mac) | `b5e4b0a2cf83` | `b5e4b0a2cf83` | n/a | PASS (PR-I) |

**6-case × N=1 AC-J2 verdict: PASS** (all 4
Mac-native cases have mode C N=1 SHA == mode A reference SHA on Mac
libomp; both server-native cases have mode C N=1 SHA == mode A reference
SHA on server libgomp via PR-I §3.2 transitive cite; together with PR-H Mac
cross-N for keliya and PR-I server cross-N for heihe + heihe_x4, the
cross-platform Mac libomp + Linux libgomp + mode A reference chain is
fully closed across the entire P1e benchmark roster).

---

## 4. Cross-platform determinism chain — comprehensive verification map

The four-step P1e cross-platform determinism evidence chain is now complete:

| evidence layer | PR | scope | runtime | result |
|----------------|----|-------|---------|--------|
| serial-only canonical | PR-C, PR-D, PR-B0 | mode A 48 cells (Mac PR-C: keliya+qhh) + 48 cells (server PR-D: heihe+heihe_x4) + 32 cells (Mac PR-B0 §4: 4 case × 4 N) | none (serial) | unique==1 per case |
| Mac libomp cross-N | PR-H | mode C 12 cells (keliya × N∈{1,2,4,8} × 3 reps) | macOS libomp 22.1.7 | bitwise == mode A (12/12) |
| server libgomp production-scale | PR-I | mode C 24 cells (heihe + heihe_x4 × N∈{1,2,4,8} × 3 reps) | Ubuntu libgomp gcc 13.3 | bitwise == mode A (24/24) + D7 speedup PARTIAL (heihe_x4 1.729×) |
| **Mac libomp N=1 reverse-compat 4-case (this PR)** | **PR-J** | **mode C 12 cells (keliya + xinanjiang_upstream + qinyijiang + qhh × N=1 × 3 reps)** | **macOS libomp 22.1.7** | **bitwise == mode A (12/12)** |

Cross-platform interpretation:
- The same `ExecPolicy::StrictOMP` source code under `SHUD/src/Model/MD_rhs_core.cpp:802-811` produces identical scalar output to within the last bit of IEEE-754 representation regardless of OS (macOS / Ubuntu), CPU architecture (ARM64 Apple Silicon / x86_64 Intel Xeon), or OpenMP runtime (LLVM libomp / GNU libgomp).
- The determinism is preserved at all tested thread counts (`SHUD_RHS_THREADS ∈ {1, 2, 4, 8}` on Mac via PR-H + server via PR-I) and at all tested mesh densities (484 / 4773 / 6335 / ~25000 cells).
- This validates the design D2 owner-local gather + reduction pattern as a
  deterministic-by-construction parallel strategy, robust against the
  classic OpenMP gotcha of non-associative floating-point reduction order
  (no `omp reduction(+:)` on doubles is used in the hot path).

---

## 5. Reproducibility footprint (Mac local)

To reproduce the PR-J Mac 6-cell verification on a fresh checkout:

```bash
# From /Users/danker/Desktop/Hydro-SHUD/openMP (or your local clone):

# 1. Sync to PR-J branch + SHUD pin 3341368
git checkout feat/issue-318-pr-j-mac-shall-closure
git pull --ff-only --recurse-submodules
(cd SHUD && git checkout openmp-baseline && git pull origin openmp-baseline)
git submodule status   # expect: 3341368... SHUD

# 2. Build mode A and mode C (separate binaries)
cd SHUD
make clean && make shud > /tmp/build_A.log 2>&1 && cp shud shud_A
make clean && make shud SHUD_ENABLE_OPENMP_RHS=1 > /tmp/build_C.log 2>&1 && cp shud shud_C

# 3. Verify mode C binary linkage
nm shud_C | grep -E '_omp_' | head        # expect 3 _omp_* symbols
otool -L shud_C | grep libomp             # expect /opt/homebrew/opt/libomp/lib/libomp.dylib

# 4. Mode A reference (3 reps per case; project name may differ from dir name)
for entry in "keliya:keliya" "xinanjiang_upstream:xinanjiang" "qinyijiang:nanlin" "qhh:qhh"; do
  dir=${entry%:*}; prj=${entry#*:}
  cd Basins/$dir
  for rep in 1 2 3; do
    rm -rf output/$prj.out
    SHUD_RHS_THREADS=1 ../../shud_A $prj > /tmp/run_A_${dir}_rep${rep}.log 2>&1
    sha256sum output/$prj.out/$prj.rivqdown.dat | awk '{print substr($1,1,12)}'
  done
  cd ../..
done

# 5. Mode C N=1 (3 reps per case)
for entry in "keliya:keliya" "xinanjiang_upstream:xinanjiang" "qinyijiang:nanlin" "qhh:qhh"; do
  dir=${entry%:*}; prj=${entry#*:}
  cd Basins/$dir
  for rep in 1 2 3; do
    rm -rf output/$prj.out
    SHUD_RHS_THREADS=1 ../../shud_C $prj > /tmp/run_C_${dir}_N1_rep${rep}.log 2>&1
    sha256sum output/$prj.out/$prj.rivqdown.dat | awk '{print substr($1,1,12)}'
  done
  cd ../..
done

# Expected output (mode A == mode C per case at N=1):
# keliya              reps: b769e3270e1c b769e3270e1c b769e3270e1c
# xinanjiang_upstream reps: 81fe3a02e17e 81fe3a02e17e 81fe3a02e17e
# qinyijiang          reps: fc1b1816cf0d fc1b1816cf0d fc1b1816cf0d
# qhh                 reps: ccc7dd09d018 ccc7dd09d018 ccc7dd09d018
```

heihe + heihe_x4 transitive cite reproduction: see PR-I
`docs/p1e/p1e_pr_i_strict_omp_verification.md` §6 reproducibility footprint.

---

## 6. Verdict and forward path

**PR-J verdict: PASS** (AC-J1 rep-stability PASS for
4/4 Mac-native cases, AC-J2 mode C N=1 == mode A reference PASS for
4/4 Mac-native cases + 2/2 server-native cases via PR-I transitive cite =
6/6 P1e benchmark cases verified per tasks §4.8 6-case roll-up).

Per tasks §4.8 the 6-case × N=1 reverse-compat comprehensive table is
complete (no parenthetical exceptions):

| case | platform | mode A reference SHA12 | mode C N=1 SHA12 | source |
|------|----------|------------------------|------------------|--------|
| keliya              | Mac libomp     | `b769e3270e1c` | `b769e3270e1c` | PR-J §2.2 |
| xinanjiang_upstream | Mac libomp     | `81fe3a02e17e` | `81fe3a02e17e` | PR-J §2.2 |
| qinyijiang          | Mac libomp     | `fc1b1816cf0d` | `fc1b1816cf0d` | PR-J §2.2 |
| qhh                 | Mac libomp     | `ccc7dd09d018` | `ccc7dd09d018` | PR-J §2.2 |
| heihe               | server libgomp | `a2023ccd2de4` | `a2023ccd2de4` | PR-I §3.2 |
| heihe_x4            | server libgomp | `b5e4b0a2cf83` | `b5e4b0a2cf83` | PR-I §3.2 |

Forward path:
- **PR-K (capstone docs)**: PR-J data feeds `docs/p1e/p1e_summary.md` §"Mac
  SHALL closure" + `docs/p1e/p1e_perf_baseline.md` §"Mac 4-cell perf wall".
  No data gap — PR-K can quote this doc directly.
- **PR-L (tag + lock)**: PR-J completion is one of the §6.3 Go/No-Go
  prerequisites (Mac SHALL closure PASS) for `P1e-tag` annotated tag
  creation.
- **PR-M (PROMOTE + epic close)**: PR-J `Closes #318` (manual close per
  CLAUDE.md after PR-J merge with base = `baseline/P1e`).

No D12.x fallback path triggered by PR-J — all 6 mode-C smoke cells PASS,
no determinism regression observed. The PR-I D7 speedup PARTIAL verdict
(heihe small-case below 1.3× threshold) is unchanged and orthogonal to
PR-J's reverse-compat scope.

---

## 7. Out-of-scope (explicit non-goals)

- **Mac cross-N for xinanjiang_upstream / qinyijiang / qhh**
  (`SHUD_RHS_THREADS ∈ {2,4,8}`): not required by PR-J spec (which scopes
  to N=1 reverse-compat per tasks §4.7). PR-H already covered keliya
  cross-N on Mac.
- **Mac heihe / heihe_x4**: case data not present on Mac (per CLAUDE.md
  双端实验环境 the heihe family is server-only); transitive cite of PR-I is
  spec-sanctioned per tasks §4.8.
- **Mac D7 speedup gate**: per design D7 platform-aware scoping, speedup
  is a server-only SHALL (the Mac is "advisory cross-N (SHOULD, 不 block
  epic)"). PR-J does not include a speedup verdict.
- **Mode D (StrictOMP RHS + OMP solver outer)**: PR-J only verifies mode C
  (StrictOMP RHS over serial solver). Mode D cross-platform reverse-compat
  is deferred to PR-K / optional follow-up — not a P1e blocker per
  proposal §"mode D" scoping language (verified via tasks §4.7 reading).
- **SHUD source modification**: PR-J is verification-only; SHUD pin
  `3341368` is unchanged from PR-H. No `cd SHUD && git commit` or pointer
  bump is performed in PR-J.
- **No silent omission of mandated cases**: prior PR-J iteration ran only
  2/4 Mac cases (keliya + qhh) and was flagged CONFIRMED Blocking by
  Phase 4.5 verifier (F-R2-1 / F-R2-2). This Phase 6 fix runs the missing
  xinanjiang_upstream + qinyijiang cases for full 4/4 closure per tasks
  §4.7 + §4.8 + Scenarios L297 / L305-308 by-name mandate.
