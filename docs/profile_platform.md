# Profile Platform Declaration (S0-11 / openMP #15)

This document declares the two endpoints on which SHUD-OpenMP B0 profile data
is captured, per rhs-profile-gate spec.md "Platform declaration document"
requirement (master plan §S0.12). Both endpoints share the same source
(SHUD submodule `78c37a1`, outer `ecef3fb`, `SHUD_ENABLE_PROFILE=1` flag),
the same flag set (`-O2 -g -ffp-contract=off -fno-fast-math -std=c++14`,
strict IEEE-754), and the same single-thread (B0) execution policy. Any
difference between the two endpoints' profile bucket distributions is
therefore attributable to the platform itself (CPU microarchitecture,
memory subsystem, compiler), not to source or flag drift.

## local_platform

Local Apple-Silicon Mac development host (per CLAUDE.md "双端实验环境"):

| Field | Value |
|---|---|
| os | Darwin 24.6.0 arm64 (xnu-11417.140.69~1) |
| cpu | Apple M4 Pro; 14 physical cores (14 logical, 1 thread/core) |
| numa | unified memory architecture (no NUMA partitioning) |
| compiler | Apple clang 17.0.0 (clang-1700.6.3.2) |
| flags | -O2 -g -ffp-contract=off -fno-fast-math -std=c++14 |
| binary_path | SHUD/shud (serial B0; SHUD_ENABLE_PROFILE=1) |
| role | development + S0 prep; **NOT** authoritative for §1.1.1 speedup gate |

## target_platform

Server endpoint Slurm compute node (per CLAUDE.md "双端实验环境" and
"要在服务器上工作时"). Authoritative for §1.1.1 quantitative gates.

| Field | Value |
|---|---|
| os | Linux 6.8.0-90-generic x86_64 |
| cpu | Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz; 2 sockets x 20 cores (40 logical, 1 thread/core) |
| numa | 2 nodes (node0 cpus 0-19, node1 cpus 20-39); ~96 GB per node |
| compiler | gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| flags | -O2 -g -ffp-contract=off -fno-fast-math -std=c++14 (serial B0; no -fopenmp) |
| binary_sha256 | 6808894b52ed79669ea563451377f99096c37992d890fad09219015da0773c98 |
| slurm_node | cn08 (all 6 profile runs landed on cn08 via Slurm CPU partition) |
| role | §1.1.1 speedup gate evidence + final P-prod sign-off |

Note on master plan §5 S0.12 "目标平台 = 单插槽 8 物理核" expectation:
the actual server node (`cn08`) is dual-socket 20-core. For the **B0
profile** (single-thread, no OpenMP), the socket / NUMA layout does not
affect bucket distribution — the work runs entirely on a single core
pinned by Slurm. P1-P7 strict-parallel runs will need to pin to a
single socket (`numactl -N 0`) to match the "single-socket 8 physical
core" verification target; that constraint is layered on top of #15
and is out of scope here.

## decision_consistency

The following table compares the `t_RHS_total / t_wall_total` ratio
(the metric that drives the Amdahl ceiling and the parallel-priority
decision) between the two endpoints, for the 4 cases that have BOTH
local and target real profile_B0 yamls. The 3 remaining cases:

- `heihe` — local DEFERRED (forcing too large for local dev host),
  target REAL → no local-vs-target diff computable.
- `heihe_x4` — same as heihe (server-only by case x endpoint matrix),
  target REAL → no local-vs-target diff computable.
- `kashigeer` — both endpoints DEFERRED via upstream forcing gap
  (issue #29) → no profile data exists on either side.

For cases with both sides:

| Case | local t_RHS%/t_wall% | target t_RHS%/t_wall% | delta (pp) | wall_local (s) | wall_target (s) |
|---|---|---|---|---|---|
| keliya | 36.32% | 49.33% | **+13.01** | 27.8 | 79.7 |
| xinanjiang_upstream | 44.72% | 43.74% | -0.98 | 4.1 | 19.7 |
| qinyijiang | 51.07% | 64.64% | **+13.57** | 229.5 | 799.9 |
| qhh | 38.00% | 36.75% | -1.25 | 75.7 | 215.8 |

| Metric | Value |
|---|---|
| max_abs_delta_pp | 13.57 (qinyijiang) |
| cases_over_10pp_threshold | 2 / 4 (keliya, qinyijiang) |
| **delta_acceptable** | **false** |

Per spec.md scenario "> 10pp difference triggers review note", because
`max_abs_delta_pp > 10`, this document MUST be paired with a
"Cross-platform delta review" section in `docs/profile_decision.md`
explaining the deviation and its impact on the gate decision (see
"Cross-platform delta review" section in `docs/profile_decision.md`).

## SHA256 digests

Local-side profile artifacts (4 real + 3 deferred):

```
b16e8a9acedcff00c82b66192d3db3eced538c7718c8715adff7b384761ce14f  benchmarks/keliya/profile_B0.yaml
34831b4b641f664cece3c5a4db1dd2c72b0016b99dc4206d30ac9b5365a43d97  benchmarks/xinanjiang_upstream/profile_B0.yaml
77bdbad6bd23e9806688bb7e76cb82d8359ed3edfbc7af03002ea9ee8ad87b02  benchmarks/qinyijiang/profile_B0.yaml
9506ceeeab796c9685dccfd649f5adc889e239e3df2f3274d5a82647637cec08  benchmarks/qhh/profile_B0.yaml
cbe50adf2047c88bc1e7d6415ce76c6ba8310e159310876c72f2bebdbc24982c  benchmarks/heihe/profile_B0.deferred.yaml
aea1322a9b2b6ee2ffaf391b422ccf05e8c1c7f29f5d20a9778dc6568eed8a9f  benchmarks/heihe_x4/profile_B0.deferred.yaml
5bdecf4e2173868de20659141cb9b046349cc5ae0fba97078b877f1cd4a5cd02  benchmarks/kashigeer/profile_B0.deferred.yaml
```

Target-side profile artifacts (6 real + 1 deferred):

```
711a380902d2dee176ff16bf5c3a5c360a9ee131420d7727a7d4e75dc62ca0f5  benchmarks/keliya/profile_B0.target.yaml
a739dfd7c66310bf5e5bcb0317a99768d3c1d41480e8e991e0d32aaeca9637e1  benchmarks/xinanjiang_upstream/profile_B0.target.yaml
1dae17564e44de5149f8e49cb8dd3f404caa5a1ee19dc0b9ef2f26ab417174ed  benchmarks/qinyijiang/profile_B0.target.yaml
cc312b7ab1db926ab85fff86b91cc0e29fc02b2a289103ee30db9555dad105f5  benchmarks/qhh/profile_B0.target.yaml
baa03be7ce16e01345bdc9e9b93c033ffcee55213113b9b1ba91441414a97f5d  benchmarks/heihe/profile_B0.target.yaml
03d9d4c9def804b27f5f5e6a8930063eb03ce5ad5cbadee979c848f829254c36  benchmarks/heihe_x4/profile_B0.target.yaml
8f64779b5c3c25b2a854f70b9721231a4e40b5dd4a1b2eadf2e3a7e43d615d17  benchmarks/kashigeer/profile_B0.target.deferred.yaml
```
