# P1e PR-I — Server 24-cell Mode-C SHALL Gate + D7 Speedup AND-gate Verdict

OpenSpec change: `p1e-strict-omp-rhs` task **§4** (PR-I scope per design D7)
Issue: #317 — part of epic #308
PR base: `baseline/P1e`
Date: 2026-06-25 → 2026-06-26 UTC

---

## 1. Scope and identity

PR-I executes the **3 SHALL hard-gate verification** for P1e strict-omp mode
on the server (`210.77.77.22:32099`, Slurm CPU partition) at real production
scale: `heihe` (6335 cells) + `heihe_x4` (~25k cells). All 24 cells run the
mode C build (`make shud SHUD_ENABLE_OPENMP_RHS=1` → `N_VNew_Serial` + true
`ExecPolicy::StrictOMP` RHS parallel region under libgomp).

PR-I is the second-of-three SHALL gate runs in P1e:
- **PR-H (#316, merged)** demonstrated mode C bitwise-equality vs mode A on
  Mac libomp (12/12 cells == reference SHA). PR-I verifies the same holds on
  Linux libgomp + real NUMA + 25k-cell scale.
- **PR-J (next)** = Mac 4-case N=1 reverse-compat. PR-I is the server
  precondition for PR-J.

Cell budget: **2 cases × 4 OMP_NUM_THREADS × 3 reps = 24 cells**, all mode C.
AC-S2 mode-A vs mode-C equality is verified against the **PR-D #312 LOCKED
reference SHAs** rather than running additional mode A cells (mode A was
already exhaustively certified in PR-D Phase 1 + AC1/AC2 PASS for both
cases × all 4 N × 3 reps).

| Identity field | Value |
|---|---|
| Outer commit (parent of PR-I) | `ccd7f00` `chore(p1e): append review-loop-log PR-H entry…` |
| Branch | `feat/issue-317-pr-i-server-shall-gate` |
| SHUD submodule pin | `3341368d2d0854924d2286925c8575df52cc97a0` (PR-H result, unchanged in PR-I) |
| Driver | `tools/p1e_2x2_runner.sh` (PR-B contract; reused unchanged) |
| Aggregator | `tools/p1e_aggregate_pr_i_shall.sh` (new in this PR) |
| Mode A binary | `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud_A` |
| Mode A binary sha256 | `68f0358b38315d7d77bee0005088afa9461d72e95d20a8aff8af73b9780214c9` |
| Mode A binary mtime | `2026-06-26T00:36:14+0800` |
| Mode C binary | `/scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud_C` |
| Mode C binary sha256 | `1bfdc4c99b038301b6ba1fb48e2d935f449476e52cc4e42fe5301c0e7d637616` |
| Mode C binary mtime | `2026-06-26T00:36:31+0800` |
| Host OS | Ubuntu 24.04.2 LTS, kernel `Linux 6.8.0-57-generic x86_64` |
| Compute nodes | cn14 (heihe stream) + cn15 (heihe_x4 stream), Slurm CPU partition |
| CPU | Intel Xeon (cn14/cn15 family; sockets=2 cores=40, see §6 reproducibility) |
| Compiler | gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| OMP runtime | libgomp.so.1 from gcc 13.3 (`/lib/x86_64-linux-gnu/libgomp.so.1`) |
| Heihe stream job IDs | 9331–9342 (chained `--dependency=afterany:<prev>` on cn14) |
| Heihe_x4 stream job IDs | 9343–9354 (chained `--dependency=afterany:<prev>` on cn15) |
| Run window | 2026-06-25T16:41:05Z → 2026-06-25T20:52:00Z (~4h10m wall, parallel streams; cn14 heihe ~1h40m, cn15 heihe_x4 ~4h10m) |
| Scratch artifact root (server) | `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/` |
| Local mirror | `/tmp/p1e_pr_i_server_runs/` |

Cell naming: `C<case>_N<N>_rep<r>` (e.g. `Cheihe_N1_rep1`,
`Cheihe_x4_N8_rep3`) — concatenated build+case letter, matching PR-D server
convention. Per-cell artifact dir on server:
`/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/C<case>_N<N>_rep<r>/`.

CMFD V0200 forcing + 90-day truncation inherited from project iron rules.

### 1.1 Reference SHAs (LOCKED from PR-D #312 mode A)

PR-D Phase 1 mode A ran 24 cells per case × AC1/AC2 PASS → these SHAs are
the canonical reference for AC-S2:

| case | reference rivqdown.sha256 (full) | reference cvode_nst |
|------|----------------------------------|--------------------:|
| heihe    | `a2023ccd2de43543a675df8eee46aa72ef1f9437764d28863dda62a4029e7798` | 6698 |
| heihe_x4 | `b5e4b0a2cf83b2a4b97d6be5b40ea7e0580d59fb6934db73a95f366f5ffc72b4` | 6575 |

Source of reference: `docs/p1e/p1e_pr_d_2x2_server.md` §3.1 AC1 PASS rows + §3.3
AC3 per-(build,case) baseline table. PR-D archived per-cell SHA file at
`/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/2x2/Aheihe_N1_rep1/heihe.rivqdown.sha256`
on server (verified during PR-I prep).

### 1.2 Mode C binary verification (per tasks §9 P1e.4 PR-I checklist)

Required (per tasks): mode C binary SHALL contain `N_VNew_Serial` symbol
reference; SHALL NOT contain `N_VNew_OpenMP`; SHALL link in OpenMP runtime
(`GOMP_parallel` on Linux):

```
$ nm /scratch/frd_muziyao/SHUD-OpenMP/SHUD/shud_C | grep -E 'N_VNew_Serial|N_VNew_OpenMP|GOMP_parallel' | sort | uniq
                 U GOMP_parallel@GOMP_4.0
                 U N_VNew_Serial
```

- `N_VNew_Serial` present  PASS
- `N_VNew_OpenMP` absent  PASS
- `GOMP_parallel@GOMP_4.0` present  PASS (binary truly links libgomp)

Contrast with mode A (`shud_A`):
```
$ nm SHUD/shud_A | grep -E 'N_VNew_Serial|N_VNew_OpenMP|GOMP_parallel'
                 U N_VNew_Serial
```

Mode A has only `N_VNew_Serial`, no GOMP — confirms serial-only build.

---

## 2. Cell roster (24 cells)

| cell | case | N | rep | wall_sec | cvode_nst | cvode_nfe | cvode_netf | rivqdown_sha12 | cv_y_h0_sha12 |
|------|------|--:|--:|---------:|----------:|----------:|-----------:|----------------|---------------|
| Cheihe_N1_rep1 | heihe | 1 | 1 | 513 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N1_rep2 | heihe | 1 | 2 | 504 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N1_rep3 | heihe | 1 | 3 | 504 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N2_rep1 | heihe | 2 | 1 | 491 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N2_rep2 | heihe | 2 | 2 | 490 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N2_rep3 | heihe | 2 | 3 | 490 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N4_rep1 | heihe | 4 | 1 | 480 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N4_rep2 | heihe | 4 | 2 | 480 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N4_rep3 | heihe | 4 | 3 | 481 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N8_rep1 | heihe | 8 | 1 | 473 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N8_rep2 | heihe | 8 | 2 | 473 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_N8_rep3 | heihe | 8 | 3 | 472 | 6698 | 6943 | 0 | `a2023ccd2de4` | `1941120e3437` |
| Cheihe_x4_N1_rep1 | heihe_x4 | 1 | 1 | 1338 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N1_rep2 | heihe_x4 | 1 | 2 | 1344 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N1_rep3 | heihe_x4 | 1 | 3 | 1340 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N2_rep1 | heihe_x4 | 2 | 1 | 1038 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N2_rep2 | heihe_x4 | 2 | 2 | 1037 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N2_rep3 | heihe_x4 | 2 | 3 | 1040 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N4_rep1 | heihe_x4 | 4 | 1 | 873 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N4_rep2 | heihe_x4 | 4 | 2 | 872 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N4_rep3 | heihe_x4 | 4 | 3 | 868 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N8_rep1 | heihe_x4 | 8 | 1 | 776 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N8_rep2 | heihe_x4 | 8 | 2 | 775 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |
| Cheihe_x4_N8_rep3 | heihe_x4 | 8 | 3 | 775 | 6575 | 6741 | 0 | `b5e4b0a2cf83` | `5be104e28df2` |

Notes:
- Per-cell sha files: `<runs>/C<case>_N<N>_rep<r>/<case>.rivqdown.sha256`
- Per-cell wall: `<runs>/C<case>_N<N>_rep<r>/wall.sec`
- Per-cell cvode_stats: `<runs>/C<case>_N<N>_rep<r>/cvode_stats.txt`
- Per-cell CV_Y state-hash sequence: `<runs>/C<case>_N<N>_rep<r>/cv_y_hash.txt`

---

## 3. SHALL gate verdicts

### 3.1 AC-S1 — mode C cross-N bitwise per case (SHALL gate)

Criterion: for each case, the `<case>.rivqdown.sha256` SHALL be identical
across all 12 cells (4 OMP_NUM_THREADS values × 3 reps). One unique SHA per
case = PASS.

| case | unique_SHAs | rep1_N1 | rep2_N1 | rep3_N1 | rep1_N2 | rep1_N4 | rep1_N8 | verdict |
|------|-----------:|---------|---------|---------|---------|---------|---------|:-------:|
| heihe | 1 | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` | `a2023ccd2de4` | PASS |
| heihe_x4 | 1 | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` | `b5e4b0a2cf83` | PASS |

**AC-S1 verdict: PASS** (mode C SHA bitwise-identical across all N and reps for both cases — strict-omp RHS produces deterministic output under libgomp + cn14/cn15 NUMA at production scale)

### 3.2 AC-S2 — mode C SHA == PR-D mode A reference SHA per case (SHALL gate)

Criterion: for each case, mode C SHA SHALL equal the PR-D #312 LOCKED mode A
reference SHA (per §1.1). This verifies that adding `ExecPolicy::StrictOMP`
RHS parallelism produces bitwise-identical output to the serial RHS path on
server libgomp — same finding as PR-H Mac demonstration.

| case | mode_C_SHA(N=1,rep=1) | reference_SHA(PR-D mode A) | match | verdict |
|------|------------------------|----------------------------|:-----:|:-------:|
| heihe | `a2023ccd2de43543` | `a2023ccd2de43543` | same | PASS |
| heihe_x4 | `b5e4b0a2cf83b2a4` | `b5e4b0a2cf83b2a4` | same | PASS |

**AC-S2 verdict: PASS** (mode C matches PR-D mode A canonical → strict-omp RHS == serial RHS bitwise on server libgomp; cross-platform bitwise neutrality holds for Mac libomp [PR-H] **and** Linux libgomp [PR-I] at production scale)

### 3.3 AC-S3 — D7 speedup AND-gate (SHALL gate)

Criterion (per design D7 + tasks §4.3 + §4.6):
- `wall(heihe, N=1) / wall(heihe, N=8) ≥ 1.3×` AND
- `wall(heihe_x4, N=1) / wall(heihe_x4, N=8) ≥ 1.5×`
- AND-gate (NOT OR-gate per tasks §4.6): D12.3 fallback triggers only if
  BOTH cases below own threshold; if exactly one fails, tasks §4.6.2
  partial-closure decision applies.
- Wall protocol: median of 3 reps per cell (per design D7 wall-measurement).

| case | N=1 wall_median (s) | N=8 wall_median (s) | speedup | threshold | per-case verdict |
|------|---------------------:|---------------------:|--------:|----------:|:----------------:|
| heihe    | 504  | 473 | 1.066x | 1.3x | **FAIL** |
| heihe_x4 | 1340 | 775 | 1.729x | 1.5x | **PASS** |

**AC-S3 D7 AND-gate verdict: PARTIAL** — exactly one case meets threshold. Per tasks §4.6.2: partial-closure user decision point. Since `heihe_x4` (the production-target large-mesh case) achieves **1.729× ≥ 1.5×** on the 25k-cell mesh, the spec defaults to **prefer ship** rather than block on the small-case (`heihe` 6335 cells) speedup shortfall.

Interpretation of `heihe` 1.066× shortfall:
- **Not a determinism failure** (AC-S1 + AC-S2 PASS confirm strict-omp RHS is bitwise-correct under libgomp).
- **Most likely cause**: heihe at 6335 cells is below the OMP_CUTOFF break-even on libgomp + cn14 NUMA (Xeon Gold class). RHS work per element does not amortize OpenMP fork-join overhead at this mesh density when each NUMA node sees ~3k cells with N=8. PR-H Mac libomp showed similar pattern at smaller scale (Mac M-series shared L3, fewer NUMA artifacts).
- **D7 design AND-gate intentionally allows asymmetric thresholds** (1.3× small, 1.5× large) precisely to acknowledge this small-case overhead floor. heihe_x4 1.729× confirms the strict-omp RHS strategy scales correctly on the actual production target.

---

## 4. Speedup and wall-time tables (informational)

Mean of 3 reps (integer seconds) per cell:

| case | N=1 mean | N=2 mean | N=4 mean | N=8 mean | N1/N8 mean |
|------|---------:|---------:|---------:|---------:|-----------:|
| heihe | 507.0 | 490.3 | 480.3 | 472.7 | 1.073x |
| heihe_x4 | 1340.7 | 1038.3 | 871.0 | 775.3 | 1.729x |

Median of 3 reps (integer seconds) per cell (used by AC-S3 SHALL gate):

| case | N=1 median | N=2 median | N=4 median | N=8 median | N1/N8 median |
|------|-----------:|-----------:|-----------:|-----------:|-------------:|
| heihe | 504 | 490 | 480 | 473 | 1.066x |
| heihe_x4 | 1340 | 1038 | 872 | 775 | 1.729x |

Per-cell raw wall (sec):

| case | N | rep1 | rep2 | rep3 |
|------|--:|-----:|-----:|-----:|
| heihe | 1 | 513 | 504 | 504 |
| heihe | 2 | 491 | 490 | 490 |
| heihe | 4 | 480 | 480 | 481 |
| heihe | 8 | 473 | 473 | 472 |
| heihe_x4 | 1 | 1338 | 1344 | 1340 |
| heihe_x4 | 2 | 1038 | 1037 | 1040 |
| heihe_x4 | 4 | 873 | 872 | 868 |
| heihe_x4 | 8 | 776 | 775 | 775 |

Per-rep variance is tight (<2% IQR for all cells), so wall measurements are
trustworthy — the heihe small-case shortfall is a true overhead floor, not
measurement noise.

Scaling slope on heihe_x4: ratios are N=1→2 1.291×, N=2→4 1.190×, N=4→8 1.125×.
Diminishing returns are consistent with serial Newton solve + linear-solver preconditioner being a fixed overhead per CVode step, while only the RHS reaction-network evaluation is parallelized via `ExecPolicy::StrictOMP`.

---

## 5. nst stability (informational + p1d-numa-governance nst ladder)

Per `openspec/specs/p1d-numa-governance/spec.md` nst ladder Requirement:
- heihe: `nst` cross-N Δ=0 strict
- heihe_x4: `|Δ_nst| ≤ 2` (mesh-refinement SPGMR convergence ladder)

Reference values from PR-D #312 mode A baseline (per §1.1): heihe nst=6698,
heihe_x4 nst=6575.

| case | ref_nst (mode A) | N=1 rep1 | N=2 rep1 | N=4 rep1 | N=8 rep1 | max_delta_to_ref | ladder_OK |
|------|-----------------:|---------:|---------:|---------:|---------:|-----------------:|:---------:|
| heihe | 6698 | 6698 | 6698 | 6698 | 6698 | 0 | PASS |
| heihe_x4 | 6575 | 6575 | 6575 | 6575 | 6575 | 0 | PASS |

`max_delta_to_ref = 0` for both cases — strict-omp RHS does not perturb the CVode Newton iteration count, confirming that the linear-solver preconditioner sees bitwise-identical RHS input regardless of OpenMP thread count.

---

## 6. Reproducibility footprint

To re-submit the PR-I 24-cell experiment on the server:

```bash
# On the server, from /scratch/frd_muziyao/SHUD-OpenMP:

# 1. Sync to baseline/P1e + SHUD pin 3341368
git checkout baseline/P1e
git pull --ff-only --recurse-submodules
(cd SHUD && git checkout openmp-baseline && git pull origin openmp-baseline)
git submodule status   # expect: 3341368... SHUD

# 2. Build mode A + mode C (one-time; sentinel keeps shud binary mapped to mode C)
cd SHUD
make clean && make shud > /tmp/build_A.log 2>&1 && cp shud shud_A
make clean && make shud SHUD_ENABLE_OPENMP_RHS=1 > /tmp/build_C.log 2>&1
cp shud shud_C
cp shud_C shud && echo "C" > .last_build_mode_shud   # prime runner sentinel
cd ..

# 3. Generate 24 sbatch (per /tmp/p1e_i_sbatch_gen_v2.sh pattern; see also
#    /scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/sbatch/)
mkdir -p .p1e-i-runs/{sbatch,logs}
# (see runner script — substitutes case/N/rep into template; one sbatch per cell)

# 4. Chain-submit using afterany dependency (CRITICAL: don't use singleton,
#    don't omit dependency — concurrent cells on the same node race-write
#    Basins/<case>/output/<case>.out/*.bin)
cd .p1e-i-runs
PREV=""
for s in sbatch/Cheihe_N{1,2,4,8}_rep{1,2,3}.sbatch; do
  if [[ -z "$PREV" ]]; then JOBID=$(sbatch "$s" | awk '{print $NF}'); fi
  if [[ -n "$PREV" ]]; then JOBID=$(sbatch --dependency=afterany:$PREV "$s" | awk '{print $NF}'); fi
  PREV=$JOBID
done
# Repeat for Cheihe_x4_* on a different node.
```

Aggregate locally after rsyncing the small-file artifacts:

```bash
mkdir -p /tmp/p1e_pr_i_server_runs
rsync -avh -e "ssh -p 32099" \
  --include='*/' \
  --include='cvode_stats.txt' --include='cell.meta' \
  --include='cv_y_hash.txt'   --include='wall.sec' \
  --include='*.rivqdown.sha256' --exclude='*' \
  frd_muziyao@210.77.77.22:/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/ \
  /tmp/p1e_pr_i_server_runs/

tools/p1e_aggregate_pr_i_shall.sh all
```

The aggregator is **idempotent**: it derives everything from per-cell
artifact files and re-runs cleanly. Exit code = 0 unless an AC-S* SHALL
gate FAILS.

Expected wall budgets per cell on Xeon (cn14/cn15 family):
- heihe (6335) mode C: ~5-10 min/cell (N=1 slowest, N=8 fastest if speedup works)
- heihe_x4 (~25k) mode C: ~15-25 min/cell
- Total per-cell × 12 per stream = ~120 min (heihe) + ~300 min (heihe_x4)
- Parallel streams (one per node) → wall budget ≈ 5h

---

## 7. Rerun history

| Date (UTC) | Action | Trigger | Result |
|------------|--------|---------|--------|
| 2026-06-25T16:37:05Z | mode A + mode C built on cn14 login (gcc 13.3, SHUD pin 3341368) | PR-I Phase 2 | binaries sha256 recorded in §1 identity table |
| 2026-06-25T16:41:05Z | First sbatch submitted (job 9331 = `Cheihe_N1_rep1`, `--dependency=afterany` chain start cn14) | PR-I Phase 3 | 12 heihe cells chained 9331→9342 |
| 2026-06-25T16:42:11Z | Second stream submitted (job 9343 = `Cheihe_x4_N1_rep1`, chain start cn15) | PR-I Phase 3 | 12 heihe_x4 cells chained 9343→9354 |
| 2026-06-25T20:52:00Z | Last cell completed (job 9354 = `Cheihe_x4_N8_rep3`, all 24 ExitCode 0) | sacct verify | total wall ~4h10m for slower stream |
| 2026-06-25T20:53:00Z | rsync server → `/tmp/p1e_pr_i_server_runs/` (25 MB; per-cell artifacts only) | PR-I Phase 4 | 24 cell dirs + sbatch + logs mirrored |
| 2026-06-25T20:55:00Z | `tools/p1e_aggregate_pr_i_shall.sh all` ran cleanly | PR-I Phase 4 | exit 0; AC-S1+S2 PASS, AC-S3 PARTIAL |

---

## 8. Decision implications (D12 routing)

### 8.1 SHALL gate summary

| Gate | Result | Notes |
|------|:------:|-------|
| AC-S1 (mode C cross-N bitwise per case) | **PASS** | both cases 1 unique SHA across 12 cells; strict-omp RHS deterministic under libgomp |
| AC-S2 (mode C SHA == PR-D mode A reference) | **PASS** | both cases match PR-D #312 canonical; strict-omp RHS == serial RHS bitwise on Linux libgomp |
| AC-S3 (D7 speedup AND-gate) | **PARTIAL** | heihe 1.066× < 1.3× FAIL; heihe_x4 1.729× ≥ 1.5× PASS |

### 8.2 D12 routing decision (per tasks §4.6 + §10.4)

The four-way D12 decision tree from tasks §2.7 / §10.4:

- **D12.1 (happy path)**: AC-S1+S2 PASS **AND** both cases meet per-case speedup threshold → ship + proceed to PR-J (Mac SHALL closure). → NOT triggered (heihe fails 1.3×).
- **D12.2 (cross-N FAIL)**: AC-S1 FAILS → NVECTOR_REPRO_OMP custom backend → new epic. → NOT triggered (AC-S1 PASSes).
- **D12.3 (cross-N PASS but both speedups < threshold)**: triggers PR-N P1e.8 block-Jacobi precond. → NOT triggered (heihe_x4 already meets 1.5×).
- **D12.4 (ADR-0003 KLU pattern spike)**: deferred to next epic. → NOT triggered.

**Active branch**: tasks §4.6.2 partial-closure decision point. The spec text:

> 4.6.2 单 case 不达 threshold（另一 case 已达）：进 partial closure 决策点（用户决策 ship vs fallback；倾向 ship 当 heihe_x4 达 1.5× 时）

heihe_x4 achieves 1.729×, exceeding 1.5×. Per spec default, **PR-I recommends ship**.

### 8.3 Recommended next steps

1. **PR-I PR body** flags this as the D12 routing data point and **requests user decision** on §4.6.2 ship-vs-fallback per spec.
2. If user confirms ship: PR-J (Mac SHALL closure) inherits the PARTIAL verdict; PR-K capstone closes P1e epic with explicit "heihe small-case overhead floor" as a known limitation (documented in p1e_summary.md per tasks §6 capstone scope).
3. If user opts fallback: PR-N P1e.8 block-Jacobi precond is built to amortize Newton solve serial overhead and re-test heihe specifically (cost: ~1-2 weeks added P1e scope; speedup improvement uncertain because heihe Newton step is already short).
4. **Reverse-compat warning**: PR-I confirms strict-omp RHS does not break serial path determinism (mode C SHA == mode A SHA); this guarantees PR-J `OMP_NUM_THREADS=1` reverse-compat will trivially pass.

### 8.4 amend `docs/p1e/p1e_2x2_verdict.md` Phase 2 (per tasks §4.6.3)

This PR-I doc establishes the data; the actual amend of `p1e_2x2_verdict.md`
Phase 2 chapter happens in the same PR (per tasks §4.6.3) **after** user confirms
the D12 routing decision in §8.3 above. PR-I body explicitly asks for that
confirmation before the merge.

---

## 9. Files produced by PR-I

| Path | Purpose |
|---|---|
| `docs/p1e/p1e_pr_i_server_shall.md` | This document |
| `tools/p1e_aggregate_pr_i_shall.sh` | Idempotent server 24-cell aggregator + AC-S{1,2,3} gates |

Per-cell artifacts live under `/scratch/frd_muziyao/SHUD-OpenMP/.p1e-i-runs/`
on the server and are mirrored to `/tmp/p1e_pr_i_server_runs/` locally for
aggregation; not committed to the repo (project rule: ephemeral artifact
dirs stay in dot-prefixed `.p1e-i-runs/` server-side and `/tmp/...` locally).
