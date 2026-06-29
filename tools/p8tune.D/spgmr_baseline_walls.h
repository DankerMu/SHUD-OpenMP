// tools/p8tune.D/spgmr_baseline_walls.h
//
// P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
// `p8tune-klu-spike` spec §REQ-5 Scenario "Wall axis threshold":
//
//   `SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline` SHALL reference
//   epic #362 (`p8tune-spgmr-maxl`) PR-D #373 60-cell sweep baseline — NOT
//   this epic's PR-A, which is the new 16-cell KLU sweep. Numerical anchor:
//   heihe_x4 N=1 maxl=5 median wall=1489.76s / nst=6575 ≈ 0.227 s/step.
//
//   AND the aggregator SHALL read the median wall from
//   /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv
//   (3-rep median for the row matching case=heihe_x4, N=1, maxl=5)
//   AND to avoid re-parsing epic #362 artifacts at PR-B time, the numeric
//   baseline value SHALL also be pinned into `tools/p8tune.D/spgmr_baseline_walls.h`
//   (a header analogous to `cn_node_ram.h`) so aggregator can fall back to
//   the pinned constant if `_summary.tsv` is unreachable.
//
// Derivation (verified at PR-0 from
// .review-evidence/p8tune-klu-spike-pr-0/maxl_sweep_summary.tsv):
//
//   case=heihe_x4 N=1 maxl=5
//   3 rep wall_s values (sorted):  1484.22  1489.76  1498.68
//   3-rep median (2nd):            1489.76
//   nst (steps):                   6575
//   per-step wall:                 1489.76 / 6575  =  0.226579 s/step
//
// Anchor in spec docs / master plan:                  0.227 s/step (rounded).
// Pinned constant below is the full-precision value for aggregator math.

#pragma once

// 3-rep median wall (heihe_x4 N=1 maxl=5) divided by nst — the per-step
// SPGMR baseline used by the wall axis threshold:
//   (numeric_factor_wall / refactor_freq + N_solve · solve_wall)
//     < 0.7 × SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S
static constexpr double SPGMR_PER_STEP_WALL_FROM_ADR0004_PRD_60CELL_BASELINE_S = 0.226579;

// Source-of-truth artifact path (server). PR-B aggregator reads this TSV
// and falls back to the pinned constant if the file is unreachable.
#define SPGMR_BASELINE_SOURCE_TSV \
    "/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv"

// Provenance string for aggregator KV emission.
#define SPGMR_BASELINE_PROVENANCE \
    "epic#362 PR-D #373 60-cell sweep; heihe_x4 N=1 maxl=5 3-rep median=1489.76s / nst=6575"
