/* SPDX-License-Identifier: MIT
 *
 * tools/p8tune.G0/spgmr_baseline_walls_g0.h — case-specific SPGMR per-step
 * walls for G0-5 wall-signal gate (P8-tune.G0).
 *
 * Owned exclusively by PR-0 (issue #409). Per IA-4 alignment:
 *   - PR-B (#411) MUST NOT mutate this header.
 *   - On drift detection by PR-B, PR-B emits
 *     `tools/p8tune.G0/spgmr_baseline_drift_detected.md` and a follow-up
 *     PR-0-style hotfix re-pins the values.
 *
 * Per IA-8 wall-budget contingency (issue #409 §Server Wall-Budget
 * Estimate + Contingency):
 *   - `heihe_x4` Slurm budget: --time=12:00:00  (expected wall ~3-5h)
 *   - `heihe_x16` Slurm budget: --time=24:00:00 (expected wall ~12-18h)
 *   - On overrun for either case, the constant is pinned to the
 *     `WALL_SIGNAL_UNKNOWN` sentinel (NaN) and PR-B G0-5 evaluation
 *     excludes that case from the OR-aggregate per spec REQ
 *     "Wall-signal gate evaluation uses case-specific SPGMR baselines"
 *     scenario "Missing case-specific baseline yields wall_signal_unknown".
 *
 * Initial PR-0 ship status: both constants are pinned to
 * `WALL_SIGNAL_UNKNOWN` because the heihe_x4 + heihe_x16 Slurm baseline
 * jobs are submitted asynchronously after PR-0 lands and may complete
 * before or after PR-0 merges into baseline/p8tune-amg-g0-spike. The
 * follow-up PR-0-style hotfix replaces the sentinels with measured
 * float values once the Slurm jobs return. PR-A onward MAY block on the
 * non-sentinel values being pinned (G0-5 requires at least one
 * non-sentinel for PASS); the contingency permits G0-5 PASS via the
 * OTHER case if one is WALL_SIGNAL_UNKNOWN.
 *
 * Historical anchor (P8-tune.D PR-D #373, ADR-0004 PRD 60-cell baseline)
 * is preserved as `SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S
 * = 0.226579` in tools/p8tune.D/spgmr_baseline_walls.h. That anchor is
 * NOT used by G0-5 evaluation; this G0-scoped header supersedes it for
 * the heihe_x4 / heihe_x16 wall-signal gate.
 *
 * Setup-inclusion convention: setup_included (per PR-0 task 1.2 spike
 * outcome documented in tools/p8tune.G0/PRE0_SPIKE_NOTES.md §1.2). The
 * AMG-side wall comparison MUST apply the same convention:
 *   amg_wall_per_step_sec = (amg_total_setup_wall_sec
 *                           + amg_total_solve_wall_sec) / n_cvode_steps
 */

#ifndef P8TUNE_G0_SPGMR_BASELINE_WALLS_G0_H
#define P8TUNE_G0_SPGMR_BASELINE_WALLS_G0_H

#include <math.h>  /* NAN */

/*
 * Sentinel for IA-8 wall-overrun contingency. Resolved to NaN so
 * PR-B aggregator's `isnan(SPGMR_PER_STEP_HEIHE_X4_S)` (or _x16) check
 * triggers the wall_signal_unknown emission per spec.
 */
#define WALL_SIGNAL_UNKNOWN  (NAN)

/*
 * heihe_x4 case-specific SPGMR per-step wall (seconds).
 *
 * Hot-patched 2026-06-30 from Slurm job 10012 baseline run on cn23:
 *   .p8tune.G0-runs/heihe_x4-spgmr-baseline/{wall-total.txt,cvode_stats.txt}
 *   WALL_TOTAL_SEC = 1566.5518531799316
 *   nst             = 6572
 *   per_step        = 1566.5518531799316 / 6572 = 0.238369 s
 *
 * Convention: setup_included (per PR-0 spike 1.2).
 * Provenance: heihe_x4 NumEle=40046, OMP_NUM_THREADS=1, BDF/Newton/SPGMR,
 *   90-day SHORT, exit 0, Slurm wall=00:26:08.
 */
#define SPGMR_PER_STEP_HEIHE_X4_S   (0.238369)

/*
 * heihe_x16 case-specific SPGMR per-step wall (seconds).
 *
 * Hot-patched 2026-06-30 from Slurm job 10013 baseline run on cn23:
 *   .p8tune.G0-runs/heihe_x16-spgmr-baseline/{wall-total.txt,cvode_stats.txt}
 *   WALL_TOTAL_SEC = 6244.509971141815
 *   nst             = 6556
 *   per_step        = 6244.509971141815 / 6556 = 0.952489 s
 *
 * Convention: setup_included.
 * Provenance: heihe_x16 NumEle≈252K (6.3× heihe_x4), OMP_NUM_THREADS=1,
 *   BDF/Newton/SPGMR, 90-day SHORT, exit 0, Slurm wall=01:44:08.
 */
#define SPGMR_PER_STEP_HEIHE_X16_S  (0.952489)

#endif /* P8TUNE_G0_SPGMR_BASELINE_WALLS_G0_H */
