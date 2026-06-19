# P7 acceptance results — A3a + A3b artifact schema

This directory holds the server-side P7 exit gate artifacts consumed by
the `p7-acceptance-gate` CI workflow (`.github/workflows/a3-gate.yml`).
Artifacts are produced by:

- `tools/server_validation/run_p7_wallclock.sbatch` (wall-clock acceptance)
- #100 task 8.9 (A3a full-run bitwise vs B1a-tag — 4 cases x 1 thread-count)
- #100 task 8.10 (A3b cross-thread ULP sweep — 2 cases x 4 thread-counts)

The CI gate is **opt-in**: it activates only on PRs that touch this
directory OR carry one of the labels `p7` / `acceptance-gates`. Pre-#100
the directory contains only this README + (eventually) two manually-
populated JSON files. The gate SKIPS gracefully when either JSON is
missing so PRs in the P1-P6 range pass through unaffected.

## Owner

openspec change `s2-strict-omp-full`:
- this README + JSON schema: task 9.3 (#101)
- A3a.json content: task 8.9 (#100)
- A3b.json content: task 8.10 (#100)
- consuming workflow: `.github/workflows/a3-gate.yml` (#101)

Spec authority: `openspec/changes/s2-strict-omp-full/specs/strict-omp-acceptance-gates/spec.md`
Requirements 3 (A3a) + 4 (A3b) + 6 (snapshot 48/48).

## Files (populated by #100 task 8.9 / 8.10)

### A3a.json

A3a = same-thread full-run bitwise vs B1a-tag for the 4-case core
portfolio (master plan §2.2 A3a + spec Requirement 3).

**Producer**: #100 task 8.9 (P7 exit operator runs the 4 cases under
`OMP_NUM_THREADS=4`, computes SHA256 vs B1a-tag goldens, runs
`tools/check_goldens/check_goldens.sh` (48/48 PASS) + `tools/cvode_stats_diff/cvode_stats_diff.sh`
(exit 0) on each case, writes the JSON).

**Schema**:

```json
{
  "version": "1",
  "tag_base": "B1a-tag",
  "tag_base_commit_sha": "64569b3fa1826122262242e7cf14686384269cc9",
  "tag_base_shud_pin": "58327c5a114052ffe8f25b6d3e2aec6b404963f2",
  "head_commit_sha": "<P7 PR head commit sha>",
  "head_shud_pin": "<P7 PR SHUD pin>",
  "omp_num_threads": 4,
  "cases": [
    {
      "name": "keliya",
      "dat_sha256_match": true,
      "cvode_stats_diff_exit": 0,
      "evidence_log": "tools/server_validation/p7_acceptance_results/A3a_keliya_dat_sha.txt"
    },
    {
      "name": "xinanjiang_upstream",
      "dat_sha256_match": true,
      "cvode_stats_diff_exit": 0,
      "evidence_log": "tools/server_validation/p7_acceptance_results/A3a_xinanjiang_upstream_dat_sha.txt"
    },
    {
      "name": "qinyijiang",
      "dat_sha256_match": true,
      "cvode_stats_diff_exit": 0,
      "evidence_log": "tools/server_validation/p7_acceptance_results/A3a_qinyijiang_dat_sha.txt"
    },
    {
      "name": "qhh",
      "dat_sha256_match": true,
      "cvode_stats_diff_exit": 0,
      "evidence_log": "tools/server_validation/p7_acceptance_results/A3a_qhh_dat_sha.txt"
    }
  ],
  "snapshot_48_48_pass": true,
  "snapshot_check_exit": 0,
  "result": "PASS"
}
```

**Gate logic** (`.github/workflows/a3-gate.yml`):

- `result == "PASS"` AND
- every case has `dat_sha256_match == true` AND `cvode_stats_diff_exit == 0` AND
- `snapshot_48_48_pass == true` AND `snapshot_check_exit == 0` AND
- `tag_base_commit_sha == "64569b3fa1826122262242e7cf14686384269cc9"` (frozen B1a-tag commit anchor)

Any field missing -> gate FAIL with a per-field actionable diagnostic.

### A3b.json

A3b = cross-thread ULP upper bound (master plan §2.2 A3b + spec Requirement 4).

**Producer**: #100 task 8.10 (P7 exit operator runs qinyijiang + heihe_x4
each at `OMP_NUM_THREADS` in {1, 2, 4, 8}, computes pairwise max_ulp(DY)
+ max_abs_diff(state) + CVODE step diff per spec).

**Schema**:

```json
{
  "version": "1",
  "head_commit_sha": "<P7 PR head commit sha>",
  "head_shud_pin": "<P7 PR SHUD pin>",
  "cases": [
    {
      "name": "qinyijiang",
      "num_ele": 3155,
      "thread_counts": [1, 2, 4, 8],
      "pairwise": [
        {"a": 1, "b": 2, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 1, "b": 4, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 1, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 2, "b": 4, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 2, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 4, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0}
      ],
      "water_balance_cross_thread_diff": 0.0
    },
    {
      "name": "heihe_x4",
      "num_ele": 25000,
      "thread_counts": [1, 2, 4, 8],
      "pairwise": [
        {"a": 1, "b": 2, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 1, "b": 4, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 1, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 2, "b": 4, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 2, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0},
        {"a": 4, "b": 8, "max_ulp_dy": 0, "max_abs_diff_state": 0.0, "nst_a": 0, "nst_b": 0, "nst_diff_pct": 0.0}
      ],
      "water_balance_cross_thread_diff": 0.0
    }
  ],
  "result": "PASS"
}
```

**Gate logic** (`.github/workflows/a3-gate.yml`):

- `result == "PASS"` AND
- every pairwise entry satisfies: `max_ulp_dy <= 4` AND `max_abs_diff_state < 1e-12` AND `nst_diff_pct < 0.1` AND
- every case has `water_balance_cross_thread_diff < 1e-10`

A3b case-selection contract (spec §D13 + Requirement 4):

- **Required**: qinyijiang + heihe_x4 (both NumEle > OMP_CUTOFF=1024)
- **Excluded**: keliya (NumEle=484) + xinanjiang_upstream (NumEle=801) — cutoff fallback to serial
- **Bonus / nice-to-have**: qhh (lake topology) + heihe (IO-heavy) + kashigeer

If qhh / heihe data is also present in the JSON, the gate verifies the
same thresholds but does NOT block on their absence.

## Pre-#100 state (current)

Both JSON files are absent. The `a3-gate.yml` workflow skips with a
WARNING-level annotation noting the missing artifacts but does NOT fail.
Once #100 task 8.9 + 8.10 land, the P7 PR pushes the two JSON files into
this directory + reruns the gate, which then enforces the thresholds
above.
