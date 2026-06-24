# P1d PR-I: Mac P1-update-omp-tag reference collection (#283)

**Scope**: Collect Mac (Apple Silicon) reference SHA + nst + wall matrix at the pre-P1c era anchor `P1-update-omp-tag`, so PR-J can compare post-P1d Mac output against this baseline (A3a bitwise on Mac cases). PR-I is **reference collection only** — no comparison, no judgement.

## Anchor

| Item | Value |
|---|---|
| Tag | `P1-update-omp-tag` (server-local annotated tag — outer `003f58dc` / SHUD `07c677fe`) |
| Outer commit | `003f58dc079116ef2161d2f96006228ef0e013d0` |
| Outer subject | `docs(p1): review-loop-log — #223 PR-K2 capstone (server scaling §1.1.1 WARNING)` |
| SHUD commit | `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47` |
| SHUD subject | `[#218 PR-F] MD_update.cpp lake loop #pragma omp parallel for` |
| Tag annotation | `P1 update-omp epic capstone — MD_update.cpp 3-pragma OpenMP stack` |

The tag itself is not on `origin` yet (server-local), but both anchors are reachable: outer commit via `origin/main` history, SHUD commit via `origin/openmp-baseline`. Worktree creation used `git checkout 003f58dc` directly.

## Worktree

- Path: `/Users/danker/Desktop/Hydro-SHUD/openMP/.p1d-pr-i-worktree/P1-update-omp-anchor`
- Outer HEAD verified: `003f58d` (detached)
- SHUD HEAD verified: `07c677f`
- `git submodule update --init --recursive` PASS (clean clone, source files present)
- Status: **LEFT INTACT** for PR-J reuse (per task spec T5)

## Build evidence (Mac local, Apple Silicon)

- Host: Apple M4 Pro, Darwin 24.6.0 arm64
- Compiler: Apple clang 17.0.0 (clang-1700.6.3.2)
- SUNDIALS: reused via symlink → `../../SHUD/InstallSundials` (main worktree's 6.0.0 install)
- `make clean && make shud_omp 2>&1 | tail -10` → `./shud_omp is compiled successfully!`
- `make shud 2>&1 | tail -10` → `./shud is compiled successfully!`

### FP gate (anchor-era 2-flag form)

```
$ grep -c -- "-fno-fast-math" Makefile      → 1
$ grep -c -- "-ffp-contract=off" Makefile   → 1
$ grep -c -- "-fno-associative-math" Makefile → 0
```

The anchor-era `Makefile` uses the 2-flag strict gate (`-ffp-contract=off -fno-fast-math` on line 24 `CXX_BASE_FLAGS`). `-fno-associative-math` was added later (post-anchor). On clang, `-fno-fast-math` already implies non-associative FP, so determinism at this anchor is intact. We record it as-is — the modern 3-grep gate is a post-anchor invariant, not an anchor-era requirement.

`-ffast-math` / `-Ofast` count = 0 (unsafe FP not present).

### Binaries

| Binary | Size (bytes) | mtime |
|---|---|---|
| `shud` | 252696 | 2026-06-24 09:14:45 |
| `shud_omp` | 253912 | 2026-06-24 09:14:33 |

Both `Mach-O 64-bit executable arm64`.

## Reference matrix (6 cells)

Mac case set per CLAUDE.md (small + medium with lake): **keliya** (NumEle=484) + **qhh** (NumEle=4773, lake). All cases at 90-day cap (`END = START + 90`, day-index mode per project rule).

For each (case, mode), `output/<case>.out/<case>.rivqdown.dat` SHA256 + `cvode_stats.txt` nst + `/usr/bin/time -p` real wall.

Modes:
- **serial** = `./shud <case>` (no OMP env, cfg `NUM_OPENMP=1`)
- **omp@N=1** = `OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=1 ./shud_omp <case>` (cfg `NUM_OPENMP=1`)
- **omp@N=8** = `OMP_PROC_BIND=close OMP_PLACES=cores OMP_NUM_THREADS=8 ./shud_omp <case>` (cfg `NUM_OPENMP=8`)

### keliya (NumEle=484, START=12053 END=12143, 90-day)

| Mode | SHA256 of `keliya.rivqdown.dat` | nst | wall (s) |
|---|---|---|---|
| serial | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | 101188 | 27.23 |
| omp@N=1 | `b23e15b94c0f67becbf73a45ea08e84f62680614e85e9a9ac15eac6033a51a1a` | 111272 | 65.24 |
| omp@N=8 | `c7a052803a5bae1ac1b8f8043197381d1b91ef085742b48aaa82f976e0ab38a3` | 97217 | 183.77 |

### qhh (NumEle=4773, lake, START=8401 END=8491, 90-day)

| Mode | SHA256 of `qhh.rivqdown.dat` | nst | wall (s) |
|---|---|---|---|
| serial | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | 13000 | 83.82 |
| omp@N=1 | `38ef0414d9ffa9310828183ce769db97c8c6c1e8714d71330bae78c0a2c0641c` | 13000 | 85.11 |
| omp@N=8 | `e184a6c5e0807fb9451f24f60cc2fa7637ef8e1802e32fdd0325e3b5af508f4d` | 13031 | 127.55 |

## Observations (informational, not gating)

- **keliya** (tiny 484 cells): N=8 wall (183.77s) > serial (27.23s) by 6.7× — small case is dominated by OMP overhead. CLAUDE.md §1.1.1 quantification rule already restricts performance verdicts to server. Mac numbers are dev reference only.
- **qhh** (4773 cells + lake): N=8 wall (127.55s) > serial (83.82s) by 1.5× — still oversubscribed for this case size on Apple Silicon. Reference value lies in the SHA, not the wall.
- **nst variance**: keliya serial=101188, N=1=111272, N=8=97217 — CVODE step count varies across mode because integrator is reactive to FP noise + OMP scheduling. This is **expected at the anchor era** (pre-P1c kahan, pre-P1d first-touch). PR-J will compare post-P1d Mac SHA + nst against these references to detect whether the P1d stack accidentally regressed Mac-side determinism.
- qhh serial vs qhh omp@N=1 SHA differ (`d9a4...` vs `38ef...`) but nst identical (13000). Same anchor-era expectation.

## Run dir + capture files

- `/Users/danker/Desktop/Hydro-SHUD/openMP/.p1d-pr-i-runs/round-1/`
  - `keliya_serial.sha` · `keliya_serial.nst` · `keliya_serial.wall`
  - `keliya_omp_N1.sha` · `keliya_omp_N1.nst` · `keliya_omp_N1.wall`
  - `keliya_omp_N8.sha` · `keliya_omp_N8.nst` · `keliya_omp_N8.wall`
  - `qhh_serial.sha` · `qhh_serial.nst` · `qhh_serial.wall`
  - `qhh_omp_N1.sha` · `qhh_omp_N1.nst` · `qhh_omp_N1.wall`
  - `qhh_omp_N8.sha` · `qhh_omp_N8.nst` · `qhh_omp_N8.wall`

Per CLAUDE.md `.p1d-runs/` family is gitignored as a dot-prefixed scratch dir; only the SHA values in this doc are version-controlled.

## Isolation notes (provenance integrity)

- Worktree has its own copy of `Basins/keliya/input/` and `Basins/qhh/input/` (cfg.para edits stay local).
- `forcing/` + `forcing.trimmed/` symlinked back to the main worktree's `SHUD/Basins/<case>/` (read-only inputs, ~228M keliya / ~1.4G qhh — shared safely).
- Fresh `Basins/<case>/output/` per worktree — no contamination of the main worktree's output directories.
- Original cfg.para preserved as `.bak` (via `sed -i.bak`) inside the worktree's input copy.

## Out of scope (per PR-I task spec)

- PR-J comparison logic (post-P1d Mac vs this reference) — belongs to PR-J
- Server case set (heihe / heihe_x4) — covered by PR-H server
- Multi-thread bitwise judgement — PR-I just collects; PR-J judges

## Worktree path for PR-J reuse

```
/Users/danker/Desktop/Hydro-SHUD/openMP/.p1d-pr-i-worktree/P1-update-omp-anchor
```

PR-J should:
1. Reuse the same worktree (already built, already has isolated Basins/<case>/input/).
2. Switch the worktree to post-P1d HEAD (or build post-P1d in a sibling worktree).
3. Re-run the 6 cells with identical commands.
4. Diff post-P1d SHA vs this PR-I reference table per cell.
5. Verdict per A3a (bitwise) only on Mac cases — leave performance to server (CLAUDE.md §1.1.1).
