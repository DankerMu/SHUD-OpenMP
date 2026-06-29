---
title: P8-tune.F BoomerAMG pattern-only spike — 5-axis per-case verdict
status: verdict-final-PR-C
epic: p8tune-amg-spike (#393)
pr_sequence: PR-C
adr_xref: docs/adr/0007-amg-spike-decision.md
data_source: .review-evidence/p8tune-amg-pr-b/cells/
data_provenance: Slurm 9896 16-cell array sweep (PR-B #396 / PR #403)
aggregator: tools/p8tune.F/aggregate_amg_spike.sh
renderer: tools/p8tune.F/render_verdict.sh
---

# P8-tune.F BoomerAMG pattern-only spike — verdict

> **Generated** by `tools/p8tune.F/render_verdict.sh` from PR-B 16-cell sweep evidence.
> **Spec** anchor: [openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md) §REQ-5 + §REQ-6.
> **ADR**: [docs/adr/0007-amg-spike-decision.md](../adr/0007-amg-spike-decision.md).

## Top-line verdict

**verdict_branch = `NO-GO-both`**

> heihe_x4 fails ['axis4_cycle'] (max margin 1.333×)

**Axis 4 amended verdict (FYI for ADR §Discussion)**: `GO`

> all 4 cases all 5 axes PASS for best combo

Per PR-A H3 disclosure, `cycle_complexity = 2 × operator_complexity` is a hard-coded estimate in current `boomeramg_setup_solve.cpp` implementation (NOT measurement from HYPRE telemetry). Axis 4 mechanically tracks Axis 5 across all 16 observed cells. The aggregator emits the **strict** verdict as the canonical anchor; the amended verdict above is provided to inform ADR-0007 §Discussion treatment of Axis 4 as a non-discriminating diagnostic.

## Threshold rationale

Per spec REQ-5 Scenario "Axis threshold constants pinned in shared header" + "5-axis threshold evaluation per case":

| Axis | Metric | Threshold | Derivation |
|------|--------|-----------|------------|
| 1 | `setup_wall_sec` | `< 0.237908` s | 1.5 × 0.7 × SPGMR_PER_STEP_SEC = 1.5 × WALL_BUDGET_APPLY_SEC (setup amortization allowance) |
| 2 | `apply_wall_sec` | `< 0.158605` s | 0.7 × SPGMR_PER_STEP_SEC (0.226579 s, P8-tune.D pinned baseline heihe_x4 N=1 maxl=5 3-rep median) |
| 3 | `peak_rss_bytes` | `< 129869709312` bytes (≈ 130 GiB) | 0.7 × CN_NODE_RAM_BYTES (185528156160 bytes ≈ 173 GiB cn-node RAM) |
| 4 | `cycle_complexity` | `< 1.5` (ratio) | V-cycle internal op count / NumY (canonical AMG hierarchy quality bound) |
| 5 | `operator_complexity` | `< 2.0` (ratio) | sum coarse grid sizes / fine grid size (canonical AMG memory bound) |

## Per-case T-tables

Best combo per case = `min(setup_wall_sec + apply_wall_sec)`; tiebreaker = `min(operator_complexity)` per spec REQ-5.

### keliya

**Best combo**: NN=02, interp_type=6, coarsen_type=21, NumY=1785, nnz_A=10255

| Axis | Value | Threshold | Margin (V/T) | Status |
|------|-------|-----------|--------------|--------|
| 1 Setup wall (s) | 0.000561 | 0.23790795 | 0.002358 | PASS |
| 2 Apply wall (s) | 0.001031 | 0.1586053 | 0.006500 | PASS |
| 3 Peak RSS (bytes) | 19906560 | 129869709311.99998 | 0.000153 | PASS |
| 4 Cycle complexity | 2.0059 | 1.5 | 1.337267 | FAIL |
| 5 Operator complexity | 1.0029 | 2.0 | 0.501450 | PASS |

**Overall (strict)**: `FAIL`  |  failing axes: `axis4_cycle`  |  max failing margin: `1.337267`

### heihe

**Best combo**: NN=07, interp_type=8, coarsen_type=8, NumY=21357, nnz_A=120485

| Axis | Value | Threshold | Margin (V/T) | Status |
|------|-------|-----------|--------------|--------|
| 1 Setup wall (s) | 0.001452 | 0.23790795 | 0.006103 | PASS |
| 2 Apply wall (s) | 0.001667 | 0.1586053 | 0.010510 | PASS |
| 3 Peak RSS (bytes) | 36040704 | 129869709311.99998 | 0.000278 | PASS |
| 4 Cycle complexity | 2.0 | 1.5 | 1.333333 | FAIL |
| 5 Operator complexity | 1.0 | 2.0 | 0.500000 | PASS |

**Overall (strict)**: `FAIL`  |  failing axes: `axis4_cycle`  |  max failing margin: `1.333333`

### heihe_x4

**Best combo**: NN=08, interp_type=6, coarsen_type=8, NumY=124395, nnz_A=653387

| Axis | Value | Threshold | Margin (V/T) | Status |
|------|-------|-----------|--------------|--------|
| 1 Setup wall (s) | 0.009476 | 0.23790795 | 0.039831 | PASS |
| 2 Apply wall (s) | 0.018179 | 0.1586053 | 0.114618 | PASS |
| 3 Peak RSS (bytes) | 116314112 | 129869709311.99998 | 0.000896 | PASS |
| 4 Cycle complexity | 2.0 | 1.5 | 1.333333 | FAIL |
| 5 Operator complexity | 1.0 | 2.0 | 0.500000 | PASS |

**Overall (strict)**: `FAIL`  |  failing axes: `axis4_cycle`  |  max failing margin: `1.333333`

### heihe_x16

**Best combo**: NN=12, interp_type=6, coarsen_type=8, NumY=485250, nnz_A=2481548

| Axis | Value | Threshold | Margin (V/T) | Status |
|------|-------|-----------|--------------|--------|
| 1 Setup wall (s) | 0.037785 | 0.23790795 | 0.158822 | PASS |
| 2 Apply wall (s) | 0.078349 | 0.1586053 | 0.493987 | PASS |
| 3 Peak RSS (bytes) | 399769600 | 129869709311.99998 | 0.003078 | PASS |
| 4 Cycle complexity | 2.0 | 1.5 | 1.333333 | FAIL |
| 5 Operator complexity | 1.0 | 2.0 | 0.500000 | PASS |

**Overall (strict)**: `FAIL`  |  failing axes: `axis4_cycle`  |  max failing margin: `1.333333`

## Raw 16-cell aggregate TSV

Full data at `.review-evidence/p8tune-amg-pr-c/aggregate.tsv`:

```tsv
nn	case	interp_type	coarsen_type	NumY	nnz_A	setup_wall_sec	apply_wall_sec	peak_rss_bytes	cycle_complexity	operator_complexity	residual_reduction_v1	verdict_class	hypre_version	colpack_version	shud_pin
00	keliya	6	8	1785	10255	0.001094	0.001257	19988480	2.0043	1.0021	50.8449	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
01	keliya	14	10	1785	10255	0.00057	0.00103	20316160	2.0043	1.0021	73.8267	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
02	keliya	6	21	1785	10255	0.000561	0.001031	19906560	2.0059	1.0029	57.215	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
03	keliya	8	8	1785	10255	0.000582	0.00102	20254720	2.0043	1.0021	50.8449	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
04	heihe	6	8	21357	120485	0.001558	0.001666	35717120	2	1	158532.1781	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
05	heihe	14	10	21357	120485	0.001634	0.001692	36044800	2	1	158532.1781	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
06	heihe	6	21	21357	120485	0.002623	0.002238	36020224	2.0213	1.0106	158648.6364	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
07	heihe	8	8	21357	120485	0.001452	0.001667	36040704	2	1	158532.1781	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
08	heihe_x4	6	8	124395	653387	0.009476	0.018179	116314112	2	1	11411.2725	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
09	heihe_x4	14	10	124395	653387	0.010401	0.018096	116326400	2	1	11411.2725	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
10	heihe_x4	6	21	124395	653387	0.013382	0.017854	116301824	2.0003	1.0001	11419.1688	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
11	heihe_x4	8	8	124395	653387	0.009986	0.017854	116305920	2	1	11411.2725	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
12	heihe_x16	6	8	485250	2481548	0.037785	0.078349	399769600	2	1	22538.0005	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
13	heihe_x16	14	10	485250	2481548	0.041478	0.07934	399769600	2	1	22538.0005	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
14	heihe_x16	6	21	485250	2481548	0.048697	0.079398	399769600	2.0001	1	22553.5964	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
15	heihe_x16	8	8	485250	2481548	0.039249	0.077976	399769600	2	1	22538.0005	PASS	3.1.0	unknown	1ab61c023ac2b93a178c2feb07aa3df509fe1a96
```

## Machine-readable verdict block

Canonical KV block consumed by ADR-0007 §Decision auto-fill per spec REQ-6:

```
# AGGREGATE_VERDICT_BEGIN
# p8tune-amg-spike aggregate verdict (PR-C)
# Generated by tools/p8tune.F/aggregate_amg_spike.sh
# Spec: openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md
#       REQ-4 marker-vs-class binary; REQ-5 5-axis + 4-branch; REQ-6 ADR-0007
#
# Pinned baseline constants (P8-tune.D, REUSED per spec REQ-5):
#   SPGMR_PER_STEP_SEC         = 0.226579
#   CN_NODE_RAM_BYTES          = 185528156160
# Derived thresholds:
#   WALL_BUDGET_SETUP_SEC      = 0.237908
#   WALL_BUDGET_APPLY_SEC      = 0.158605
#   WALL_BUDGET_RSS_BYTES      = 129869709312
#   CYCLE_COMPLEXITY_THRESHOLD = 1.5
#   OPERATOR_COMPLEXITY_THRESHOLD = 2.0
#
verdict_branch=NO-GO-both
verdict_branch_reason="heihe_x4 fails ['axis4_cycle'] (max margin 1.333×)"

# Axis 4 amendment disclosure (per PR-A H3 + issue #397):
# cycle_complexity = 2 × operator_complexity is a hard-coded estimate
# in current boomeramg_setup_solve.cpp implementation, NOT measurement
# from HYPRE telemetry. Axis 4 mechanically tracks Axis 5 in all 16
# observed cells (cycle ≈ 2.0 uniformly). Recommend treating Axis 4
# as a non-discriminating diagnostic in ADR-0007 §Discussion. The
# strict verdict above stands as the auto-typed canonical anchor;
# the amended branch below is FYI for ADR §Discussion.
verdict_branch_axis4_amended=GO
verdict_branch_axis4_amended_reason="all 4 cases all 5 axes PASS for best combo"

# CASE_VERDICT_BEGIN:keliya
keliya_best_combo_nn=02
keliya_best_combo_interp_type=6
keliya_best_combo_coarsen_type=21
keliya_best_combo_NumY=1785
keliya_best_combo_nnz_A=10255
keliya_axis1_setup_status=PASS
keliya_axis1_setup_value=0.000561
keliya_axis1_setup_threshold=0.23790795
keliya_axis1_setup_margin=0.002358
keliya_axis2_apply_status=PASS
keliya_axis2_apply_value=0.001031
keliya_axis2_apply_threshold=0.1586053
keliya_axis2_apply_margin=0.006500
keliya_axis3_memory_status=PASS
keliya_axis3_memory_value=19906560
keliya_axis3_memory_threshold=129869709311.99998
keliya_axis3_memory_margin=0.000153
keliya_axis4_cycle_status=FAIL
keliya_axis4_cycle_value=2.0059
keliya_axis4_cycle_threshold=1.5
keliya_axis4_cycle_margin=1.337267
keliya_axis5_operator_status=PASS
keliya_axis5_operator_value=1.0029
keliya_axis5_operator_threshold=2.0
keliya_axis5_operator_margin=0.501450
keliya_all_pass=false
keliya_failing_axes=axis4_cycle
keliya_max_failing_margin=1.337267
keliya_overall=FAIL
# CASE_VERDICT_END:keliya

# CASE_VERDICT_BEGIN:heihe
heihe_best_combo_nn=07
heihe_best_combo_interp_type=8
heihe_best_combo_coarsen_type=8
heihe_best_combo_NumY=21357
heihe_best_combo_nnz_A=120485
heihe_axis1_setup_status=PASS
heihe_axis1_setup_value=0.001452
heihe_axis1_setup_threshold=0.23790795
heihe_axis1_setup_margin=0.006103
heihe_axis2_apply_status=PASS
heihe_axis2_apply_value=0.001667
heihe_axis2_apply_threshold=0.1586053
heihe_axis2_apply_margin=0.010510
heihe_axis3_memory_status=PASS
heihe_axis3_memory_value=36040704
heihe_axis3_memory_threshold=129869709311.99998
heihe_axis3_memory_margin=0.000278
heihe_axis4_cycle_status=FAIL
heihe_axis4_cycle_value=2.0
heihe_axis4_cycle_threshold=1.5
heihe_axis4_cycle_margin=1.333333
heihe_axis5_operator_status=PASS
heihe_axis5_operator_value=1.0
heihe_axis5_operator_threshold=2.0
heihe_axis5_operator_margin=0.500000
heihe_all_pass=false
heihe_failing_axes=axis4_cycle
heihe_max_failing_margin=1.333333
heihe_overall=FAIL
# CASE_VERDICT_END:heihe

# CASE_VERDICT_BEGIN:heihe_x4
heihe_x4_best_combo_nn=08
heihe_x4_best_combo_interp_type=6
heihe_x4_best_combo_coarsen_type=8
heihe_x4_best_combo_NumY=124395
heihe_x4_best_combo_nnz_A=653387
heihe_x4_axis1_setup_status=PASS
heihe_x4_axis1_setup_value=0.009476
heihe_x4_axis1_setup_threshold=0.23790795
heihe_x4_axis1_setup_margin=0.039831
heihe_x4_axis2_apply_status=PASS
heihe_x4_axis2_apply_value=0.018179
heihe_x4_axis2_apply_threshold=0.1586053
heihe_x4_axis2_apply_margin=0.114618
heihe_x4_axis3_memory_status=PASS
heihe_x4_axis3_memory_value=116314112
heihe_x4_axis3_memory_threshold=129869709311.99998
heihe_x4_axis3_memory_margin=0.000896
heihe_x4_axis4_cycle_status=FAIL
heihe_x4_axis4_cycle_value=2.0
heihe_x4_axis4_cycle_threshold=1.5
heihe_x4_axis4_cycle_margin=1.333333
heihe_x4_axis5_operator_status=PASS
heihe_x4_axis5_operator_value=1.0
heihe_x4_axis5_operator_threshold=2.0
heihe_x4_axis5_operator_margin=0.500000
heihe_x4_all_pass=false
heihe_x4_failing_axes=axis4_cycle
heihe_x4_max_failing_margin=1.333333
heihe_x4_overall=FAIL
# CASE_VERDICT_END:heihe_x4

# CASE_VERDICT_BEGIN:heihe_x16
heihe_x16_best_combo_nn=12
heihe_x16_best_combo_interp_type=6
heihe_x16_best_combo_coarsen_type=8
heihe_x16_best_combo_NumY=485250
heihe_x16_best_combo_nnz_A=2481548
heihe_x16_axis1_setup_status=PASS
heihe_x16_axis1_setup_value=0.037785
heihe_x16_axis1_setup_threshold=0.23790795
heihe_x16_axis1_setup_margin=0.158822
heihe_x16_axis2_apply_status=PASS
heihe_x16_axis2_apply_value=0.078349
heihe_x16_axis2_apply_threshold=0.1586053
heihe_x16_axis2_apply_margin=0.493987
heihe_x16_axis3_memory_status=PASS
heihe_x16_axis3_memory_value=399769600
heihe_x16_axis3_memory_threshold=129869709311.99998
heihe_x16_axis3_memory_margin=0.003078
heihe_x16_axis4_cycle_status=FAIL
heihe_x16_axis4_cycle_value=2.0
heihe_x16_axis4_cycle_threshold=1.5
heihe_x16_axis4_cycle_margin=1.333333
heihe_x16_axis5_operator_status=PASS
heihe_x16_axis5_operator_value=1.0
heihe_x16_axis5_operator_threshold=2.0
heihe_x16_axis5_operator_margin=0.500000
heihe_x16_all_pass=false
heihe_x16_failing_axes=axis4_cycle
heihe_x16_max_failing_margin=1.333333
heihe_x16_overall=FAIL
# CASE_VERDICT_END:heihe_x16

# Provenance:
#   spgmr_baseline_walls.h = /Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/spgmr_baseline_walls.h
#   cn_node_ram.h          = /Users/danker/Desktop/Hydro-SHUD/openMP/tools/p8tune.D/cn_node_ram.h
#   run_dir                = .review-evidence/p8tune-amg-pr-b/cells
#   cells_parsed           = 16/16
#   hypre_version          = 3.1.0
#   colpack_version        = unknown
#   shud_pin               = 1ab61c023ac2b93a178c2feb07aa3df509fe1a96
# AGGREGATE_VERDICT_END
```

## Provenance

- **Hypre version**: `3.1.0`
- **ColPack version**: `unknown` (per PR-B M2 closure sentinel — ColPack reused via shell-out from P8-tune.D)
- **SHUD pin**: `1ab61c023ac2b93a178c2feb07aa3df509fe1a96`
- **SPGMR baseline**: `0.226579` s/step (P8-tune.D pinned in `tools/p8tune.D/spgmr_baseline_walls.h`)
- **CN node RAM**: `185528156160` bytes (P8-tune.D pinned in `tools/p8tune.D/cn_node_ram.h`)

## Cross-references

- [docs/adr/0007-amg-spike-decision.md](../adr/0007-amg-spike-decision.md) — ADR-0007 (4-branch decision; this verdict is the §Decision anchor)
- [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md) — ADR-0005 (KLU NO-GO at heihe_x4; triggered this AMG retreat)
- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR-0004 (SPGMR baseline anchor)
- [openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md) — capability spec
- [.review-evidence/p8tune-amg-pr-b/](../../.review-evidence/p8tune-amg-pr-b/) — PR-B 16-cell evidence directory
- [.review-evidence/p8tune-amg-pr-c/aggregate.tsv](../../.review-evidence/p8tune-amg-pr-c/aggregate.tsv) — this verdict's raw aggregate data
- [.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt](../../.review-evidence/p8tune-amg-pr-c/aggregate_verdict.txt) — machine-readable verdict
- [tools/p8tune.D/spgmr_baseline_walls.h](../../tools/p8tune.D/spgmr_baseline_walls.h) — pinned SPGMR baseline constant
- [tools/p8tune.D/cn_node_ram.h](../../tools/p8tune.D/cn_node_ram.h) — pinned cn-node RAM constant
