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
 * Initial PR-0 ship value: WALL_SIGNAL_UNKNOWN.
 * Hotfix replacement target: float value from
 *   .p8tune.G0-runs/heihe-x4-spgmr-baseline/cell-00.{out,log,time}
 * on /scratch/frd_muziyao/SHUD-OpenMP/ cn-node Slurm job (NumEle=40046,
 * --time=12:00:00, OMP_NUM_THREADS=1, BDF/Newton/SPGMR, 90-day SHORT).
 * Convention: setup_included.
 */
#define SPGMR_PER_STEP_HEIHE_X4_S   (WALL_SIGNAL_UNKNOWN)

/*
 * heihe_x16 case-specific SPGMR per-step wall (seconds).
 *
 * Initial PR-0 ship value: WALL_SIGNAL_UNKNOWN.
 * Hotfix replacement target: float value from
 *   .p8tune.G0-runs/heihe-x16-spgmr-baseline/cell-00.{out,log,time}
 * on /scratch/frd_muziyao/SHUD-OpenMP/ cn-node Slurm job (NumEle=~160k,
 * 6.3× heihe_x4 NumEle, --time=24:00:00, OMP_NUM_THREADS=1, BDF/Newton/
 * SPGMR, 90-day SHORT). Convention: setup_included.
 *
 * If the Slurm job overruns --time=24:00:00 wall budget, this constant
 * stays at WALL_SIGNAL_UNKNOWN per IA-8 contingency.
 */
#define SPGMR_PER_STEP_HEIHE_X16_S  (WALL_SIGNAL_UNKNOWN)

#endif /* P8TUNE_G0_SPGMR_BASELINE_WALLS_G0_H */
