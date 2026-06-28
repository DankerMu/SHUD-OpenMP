// tools/p8tune.D/cn_node_ram.h
//
// P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
// `p8tune-klu-spike` spec §REQ-5 Scenario "RSS axis threshold":
//
//   `CN_NODE_RAM_BYTES` SHALL be measured at PR-0 via `cat /proc/meminfo`
//   on cn14 (or equivalent representative cn-node) and embedded BOTH in
//   `tools/p8tune.D/cn_node_ram.h` (as a C++ `static constexpr size_t
//   CN_NODE_RAM_BYTES = <measured>;` used by `klu_analyze_factor.cpp`
//   for pre-flight check) AND in the aggregator threshold KV (used by
//   `aggregate_klu_spike.sh` for verdict).
//
// Measurement: `tools/p8tune.D/probe_cn_ram.sbatch` was submitted to
// Slurm on 2026-06-28 (job id 9755) and ran on cn14:
//
//   hostname:        cn14
//   date_utc:        2026-06-28T09:57:21Z
//   slurm_job_id:    9755
//   MemTotal:        181179840 kB  =  185528156160 bytes  (≈ 173.0 GiB)
//
// Evidence: `.review-evidence/p8tune-klu-spike-pr-0/cn_ram_probe.log`.
//
// The PR-A 16-cell sweep also lands on the cn[05-06,09,14-19,23-24]
// pool per master plan §P8-tune.D L107; we assume the cluster is
// homogeneous (all cn-nodes have the same DDR config). PR-A may extend
// the probe to confirm — for PR-0 we treat cn14 as representative.

#pragma once

#include <cstddef>

static constexpr std::size_t CN_NODE_RAM_BYTES = 185528156160ULL;

static_assert(CN_NODE_RAM_BYTES >= 32ULL * 1024 * 1024 * 1024,
              "cn-node RAM < 32 GB; revisit heihe_x16 RSS axis threshold");
