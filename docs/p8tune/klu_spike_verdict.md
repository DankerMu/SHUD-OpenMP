---
title: P8-tune.D KLU pattern-only spike — 3-axis per-case verdict
status: verdict-final
epic: p8tune-klu-spike (#379)
pr_sequence: PR-B
adr_xref: docs/adr/0005-klu-spike-decision.md
data_source: .review-evidence/p8tune-klu-spike-pr-a/cells/
data_provenance: Slurm 9762 (NN=00-07) + 9794+9812+9828+9829 re-runs (NN=08-15)
aggregator: tools/p8tune.D/aggregate_klu_spike.sh
---

# P8-tune.D KLU pattern-only spike — verdict

> **Generated** by `tools/p8tune.D/render_verdict.sh` from PR-A 16-cell sweep evidence.
> **Spec** anchor: [openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md) §REQ-5 / §REQ-6.
> **ADR**: [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md).

## Top-line verdict

| case      | fill axis | RSS axis | wall axis | overall | recommended action  |
|-----------|-----------|----------|-----------|---------|---------------------|
| keliya    | PASS      | PASS     | PASS      | GO      | klu-env-var-opt-in  |
| heihe     | PASS      | PASS     | PASS      | GO      | klu-env-var-opt-in  |
| heihe_x4  | PASS      | PASS     | FAIL      | Optional | use-future-amg      |
| heihe_x16 | PASS      | PASS     | FAIL      | NO-GO   | use-future-amg      |

**Case-aware branch fires**: `true` (per spec REQ-5 Scenario "Case-aware/Optional branch auto-population" — small cases GO, large cases NO-GO/Optional on wall axis).

**Decisive cell**: heihe_x4 → recommended next epic = `p8-tune.E-klu-impl` (priority = `medium`).

## Threshold rationale

Per spec REQ-5 Scenarios "Fill axis threshold" / "RSS axis threshold" / "Wall axis threshold":

### Fill axis: `fill_ratio < 8 · log₂(NumY)`

Rationale: 2D mesh PDE nested-dissection theoretical optimum ≈ log₂(NumY); the 8× factor allows real-world AMD/COLAMD deviation while still bounding LU pivoting cost. Per-case thresholds:

| case      | NumY (measured) | 8·log₂(NumY) threshold |
|-----------|-----------------|------------------------|
| keliya    | 1785            | 86.4                  |
| heihe     | 21357           | 115.1                  |
| heihe_x4  | 124395          | 135.4                  |
| heihe_x16 | 485250          | 151.1                  |

### RSS axis: `peak_RSS < 0.7 × CN_NODE_RAM_BYTES`

`CN_NODE_RAM_BYTES` = `185528156160` (measured at PR-0 via `cat /proc/meminfo` on cn14; ≈ 172.8 GiB total node RAM).

Threshold: 0.7 × 172.8 GiB ≈ 121.0 GiB. Rationale: allows multi-cell parallel execution on a cn-node without OOM (per spec REQ-5 Scenario "RSS axis threshold").

### Wall axis: `(numeric_factor_wall / refactor_freq) + (N_solve · solve_wall) < 0.7 × SPGMR_per_step_wall`

`SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s` = `0.226579` (heihe_x4 N=1 maxl=5 3-rep median = 1489.76s / nst=6575 ≈ 0.227 s/step; pinned in `tools/p8tune.D/spgmr_baseline_walls.h`).

Budget: 0.7 × 0.2266 = 0.1586 s/step.

KLU per-step estimate uses `refactor_freq=10` (conservative — refactor every 10 CVODE steps) and `N_solve=5` (typical per-step Newton iterations) with `solve_wall ≈ 0.1 × numeric_factor_wall` (triangular L/U solves are ~10× cheaper than factorization for KLU). All knobs are env-var-tunable in the aggregator (`WALL_REFACTOR_FREQ`, `WALL_N_SOLVE`, `WALL_SOLVE_FACTOR`, `WALL_SPGMR_BUDGET_FRACTION`).

## Per-case T-tables (all 4 ordering combos per case)

### keliya

| NN | ordering | btf | fill_ratio | numeric_wall_s | peak_rss_mb | verdict_class | note            |
|----|----------|-----|------------|----------------|-------------|---------------|-----------------|
| 00 | natural  | 1   |   205.9479 |       0.913238 |       28.12 | PASS          |  |
| 01 | amd      | 0   |     3.2265 |       0.001418 |        4.38 | PASS          | **best-combo** |
| 02 | amd      | 1   |     3.2265 |       0.001457 |        4.38 | PASS          |  |
| 03 | colamd   | 1   |     4.6414 |       0.002102 |        4.38 | PASS          |  |

**3-axis synthesis**:

- **Fill axis**: `PASS` — best-combo amd+btf0 fill_ratio=3.23 < threshold=8·log₂(1785)=86.4
- **RSS axis**: `PASS` — best-combo peak_rss=0.004 GiB < 0.7×CN_NODE_RAM=121.0 GiB
- **Wall axis**: `PASS` — per-step estimate=(0.0014/10) + (5·0.10·0.0014) = 0.0009 s < budget=0.7·0.2266=0.1586 s
- **Overall verdict**: `GO`
- **NO-GO axis** (when applicable): `clean_GO`
- **Recommended action**: `klu-env-var-opt-in`

### heihe

| NN | ordering | btf | fill_ratio | numeric_wall_s | peak_rss_mb | verdict_class | note            |
|----|----------|-----|------------|----------------|-------------|---------------|-----------------|
| 04 | natural  | 1   |  2304.2871 |    1630.143313 |     3185.25 | PASS          |  |
| 05 | amd      | 0   |     5.3874 |       0.038368 |       15.16 | PASS          |  |
| 06 | amd      | 1   |     5.3874 |       0.038266 |       15.28 | PASS          | **best-combo** |
| 07 | colamd   | 1   |     8.9559 |       0.077088 |       20.12 | PASS          |  |

**3-axis synthesis**:

- **Fill axis**: `PASS` — best-combo amd+btf1 fill_ratio=5.39 < threshold=8·log₂(21357)=115.1
- **RSS axis**: `PASS` — best-combo peak_rss=0.015 GiB < 0.7×CN_NODE_RAM=121.0 GiB
- **Wall axis**: `PASS` — per-step estimate=(0.0383/10) + (5·0.10·0.0383) = 0.0230 s < budget=0.7·0.2266=0.1586 s
- **Overall verdict**: `GO`
- **NO-GO axis** (when applicable): `clean_GO`
- **Recommended action**: `klu-env-var-opt-in`

### heihe_x4

| NN | ordering | btf | fill_ratio | numeric_wall_s | peak_rss_mb | verdict_class | note            |
|----|----------|-----|------------|----------------|-------------|---------------|-----------------|
| 08 | natural  | 1   |          — |              — |    15460.22 | fill_overflow | fill_overflow data point |
| 09 | amd      | 0   |     8.3477 |       0.500049 |       88.81 | PASS          |  |
| 10 | amd      | 1   |     8.3477 |       0.494465 |       89.15 | PASS          | **best-combo** |
| 11 | colamd   | 1   |    15.0578 |       1.241740 |      139.45 | PASS          |  |

**3-axis synthesis**:

- **Fill axis**: `PASS` — best-combo amd+btf1 fill_ratio=8.35 < threshold=8·log₂(124395)=135.4; natural-ordering pathology surfaced as fill_overflow data point (decisive, not blocking)
- **RSS axis**: `PASS` — best-combo peak_rss=0.087 GiB < 0.7×CN_NODE_RAM=121.0 GiB
- **Wall axis**: `FAIL` — per-step estimate=(0.4945/10) + (5·0.10·0.4945) = 0.2967 s >= budget=0.7·0.2266=0.1586 s
- **Overall verdict**: `Optional`
- **NO-GO axis** (when applicable): `wall_overflow`
- **Recommended action**: `use-future-amg`

### heihe_x16

| NN | ordering | btf | fill_ratio | numeric_wall_s | peak_rss_mb | verdict_class | note            |
|----|----------|-----|------------|----------------|-------------|---------------|-----------------|
| 12 | natural  | 1   |          — |              — |    16452.81 | fill_overflow | fill_overflow data point |
| 13 | amd      | 0   |    11.0780 |       4.742845 |      405.65 | PASS          |  |
| 14 | amd      | 1   |    11.0780 |       4.739916 |      407.68 | PASS          | **best-combo** |
| 15 | colamd   | 1   |    20.6337 |      11.708146 |      678.99 | PASS          |  |

**3-axis synthesis**:

- **Fill axis**: `PASS` — best-combo amd+btf1 fill_ratio=11.08 < threshold=8·log₂(485250)=151.1; natural-ordering pathology surfaced as fill_overflow data point (decisive, not blocking)
- **RSS axis**: `PASS` — best-combo peak_rss=0.398 GiB < 0.7×CN_NODE_RAM=121.0 GiB
- **Wall axis**: `FAIL` — per-step estimate=(4.7399/10) + (5·0.10·4.7399) = 2.8439 s >= budget=0.7·0.2266=0.1586 s
- **Overall verdict**: `NO-GO`
- **NO-GO axis** (when applicable): `wall_overflow`
- **Recommended action**: `use-future-amg`

## Amended REQ-5 scenarios (PR-A landed)

### Tool-bound data point (KLU 32-bit-int index overflow)

Per spec REQ-5 Scenario "Tool-bound data point (KLU 32-bit-int index overflow)" (amended in PR-A):

- **Trigger**: `klu_factor` returns `common.status == KLU_TOO_LARGE` (status code `-4` in SuiteSparse KLU; "integer overflow has occurred" per `klu.h`).
- **Tool behavior**: emit `KLU_INDEX_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> peak_rss_bytes=<N> reason=klu_factor_status_KLU_TOO_LARGE_int32_index_overflow` and exit 0 (NOT non-zero).
- **Aggregator classification**: `fill_overflow` data point — the 32-bit signed index space of `klu_factor` cannot hold `nnz(L+U)` for this ordering, i.e. equivalent to fill-axis hard-failing as a tool-bound limit. NOT a Slurm task failure.
- **Observed in PR-A**: NN=08 (heihe_x4 natural+BTF) and NN=12 (heihe_x16 natural+BTF). Both natural-ordering cells hit the int32 index cap; AMD/COLAMD orderings stay well within. The 64-bit-index `klu_l_*` API would be an implementation choice for P8-tune.E and is out of scope for this pattern-only spike.

### Wall-budget data point (Slurm TIMEOUT)

Per spec REQ-5 Scenario "Wall-budget data point (Slurm TIMEOUT)" (amended in PR-A):

- **Trigger**: cell exceeds Slurm `--time` wall budget; `spike_array.sbatch` SIGTERM trap emits `KLU_WALL_OVERFLOW_DETECTED case=<C> ordering=<O> btf=<B> elapsed_sec=<N> wall_budget_sec=<W>` to the cell log on a best-effort basis.
- **Tool behavior**: exit 0 with the marker (or sacct fallback when the trap window is insufficient).
- **Aggregator classification**: `wall_overflow` data point — PASS as Slurm task on marker, FAIL on wall axis. NOT a Slurm task failure.
- **Observed in PR-A**: NN=08 had an initial TIMEOUT on the 9794 sbatch (which had `--time=00:30:00` override; the natural-ordering factor extrapolates >6500s). After widening to 4h the re-run (job 9828) completed in 2h01m, but with `KLU_TOO_LARGE` (re-classified as fill_overflow). No clean wall_overflow marker fires in the final 16-cell evidence set; the wall axis verdict is derived per the formula above from the AMD best-combo `numeric_factor_wall_sec`.

## Machine-readable verdict block (canonical KV)

```
# p8tune-klu-spike aggregate verdict (PR-B)
# Generated by tools/p8tune.D/aggregate_klu_spike.sh
# Spec: openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5 + REQ-6
#
# Wall-axis interpretation parameters (env-tunable):
#   WALL_REFACTOR_FREQ          = 10.0
#   WALL_N_SOLVE                = 5.0
#   WALL_SOLVE_FACTOR           = 0.1
#   WALL_SPGMR_BUDGET_FRACTION  = 0.7
#   WALL_BUDGET_S (derived)     = 0.158605
#
keliya_KLU_fill_axis                    = PASS
keliya_KLU_rss_axis                     = PASS
keliya_KLU_wall_axis                    = PASS
keliya_KLU_overall_verdict              = GO
keliya_KLU_NO_GO_axis                   = clean_GO
keliya_KLU_NO_GO_diagnostic             = "all 3 axes PASS for best-combo amd+btf0"
keliya_KLU_fill_axis_diagnostic         = "best-combo amd+btf0 fill_ratio=3.23 < threshold=8·log₂(1785)=86.4"
keliya_KLU_rss_axis_diagnostic          = "best-combo peak_rss=0.004 GiB < 0.7×CN_NODE_RAM=121.0 GiB"
keliya_KLU_wall_axis_diagnostic         = "per-step estimate=(0.0014/10) + (5·0.10·0.0014) = 0.0009 s < budget=0.7·0.2266=0.1586 s"
keliya_recommended_action               = klu-env-var-opt-in

heihe_KLU_fill_axis                     = PASS
heihe_KLU_rss_axis                      = PASS
heihe_KLU_wall_axis                     = PASS
heihe_KLU_overall_verdict               = GO
heihe_KLU_NO_GO_axis                    = clean_GO
heihe_KLU_NO_GO_diagnostic              = "all 3 axes PASS for best-combo amd+btf1"
heihe_KLU_fill_axis_diagnostic          = "best-combo amd+btf1 fill_ratio=5.39 < threshold=8·log₂(21357)=115.1"
heihe_KLU_rss_axis_diagnostic           = "best-combo peak_rss=0.015 GiB < 0.7×CN_NODE_RAM=121.0 GiB"
heihe_KLU_wall_axis_diagnostic          = "per-step estimate=(0.0383/10) + (5·0.10·0.0383) = 0.0230 s < budget=0.7·0.2266=0.1586 s"
heihe_recommended_action                = klu-env-var-opt-in

heihe_x4_KLU_fill_axis                  = PASS
heihe_x4_KLU_rss_axis                   = PASS
heihe_x4_KLU_wall_axis                  = FAIL
heihe_x4_KLU_overall_verdict            = Optional
heihe_x4_KLU_NO_GO_axis                 = wall_overflow
heihe_x4_KLU_NO_GO_diagnostic           = "per-step estimate=(0.4945/10) + (5·0.10·0.4945) = 0.2967 s >= budget=0.7·0.2266=0.1586 s"
heihe_x4_KLU_fill_axis_diagnostic       = "best-combo amd+btf1 fill_ratio=8.35 < threshold=8·log₂(124395)=135.4; natural-ordering pathology surfaced as fill_overflow data point (decisive, not blocking)"
heihe_x4_KLU_rss_axis_diagnostic        = "best-combo peak_rss=0.087 GiB < 0.7×CN_NODE_RAM=121.0 GiB"
heihe_x4_KLU_wall_axis_diagnostic       = "per-step estimate=(0.4945/10) + (5·0.10·0.4945) = 0.2967 s >= budget=0.7·0.2266=0.1586 s"
heihe_x4_recommended_action             = use-future-amg

heihe_x16_KLU_fill_axis                 = PASS
heihe_x16_KLU_rss_axis                  = PASS
heihe_x16_KLU_wall_axis                 = FAIL
heihe_x16_KLU_overall_verdict           = NO-GO
heihe_x16_KLU_NO_GO_axis                = wall_overflow
heihe_x16_KLU_NO_GO_diagnostic          = "per-step estimate=(4.7399/10) + (5·0.10·4.7399) = 2.8439 s >= budget=0.7·0.2266=0.1586 s"
heihe_x16_KLU_fill_axis_diagnostic      = "best-combo amd+btf1 fill_ratio=11.08 < threshold=8·log₂(485250)=151.1; natural-ordering pathology surfaced as fill_overflow data point (decisive, not blocking)"
heihe_x16_KLU_rss_axis_diagnostic       = "best-combo peak_rss=0.398 GiB < 0.7×CN_NODE_RAM=121.0 GiB"
heihe_x16_KLU_wall_axis_diagnostic      = "per-step estimate=(4.7399/10) + (5·0.10·4.7399) = 2.8439 s >= budget=0.7·0.2266=0.1586 s"
heihe_x16_recommended_action            = use-future-amg

heihe_x4_recommended_next_epic                          = p8-tune.E-klu-impl
heihe_x4_recommended_next_epic_priority                 = medium

case_aware_branch_fires                                 = true

CN_NODE_RAM_BYTES                                       = 185528156160
SPGMR_per_step_wall_from_ADR0004_PRD_60cell_baseline_s  = 0.226579
```

## Raw data appendix

Full 16-cell aggregate TSV (`.review-evidence/p8tune-klu-spike-pr-b/aggregate.tsv`):

```tsv
nn	case	ordering	btf	NumY	Anz	fill_ratio	numeric_lnz_plus_unz	nnz_A	symbolic_wall_s	numeric_wall_s	peak_rss_mb	chromatic_number	verdict_class
00	keliya	natural	1	1785	10255	205.9479	2111996	10255	0.000231	0.913238	28.12	16	PASS
01	keliya	amd	0	1785	10255	3.2265	33088	10255	0.001070	0.001418	4.38	16	PASS
02	keliya	amd	1	1785	10255	3.2265	33088	10255	0.001196	0.001457	4.38	16	PASS
03	keliya	colamd	1	1785	10255	4.6414	47598	10255	0.001614	0.002102	4.38	16	PASS
04	heihe	natural	1	21357	120485	2304.2871	277632026	120485	0.002825	1630.143313	3185.25	18	PASS
05	heihe	amd	0	21357	120485	5.3874	649100	120485	0.012847	0.038368	15.16	18	PASS
06	heihe	amd	1	21357	120485	5.3874	649100	120485	0.014555	0.038266	15.28	18	PASS
07	heihe	colamd	1	21357	120485	8.9559	1079052	120485	0.017271	0.077088	20.12	18	PASS
08	heihe_x4	natural	1	124395							15460.22	16	fill_overflow
09	heihe_x4	amd	0	124395	653387	8.3477	5454266	653387	0.080795	0.500049	88.81	16	PASS
10	heihe_x4	amd	1	124395	653387	8.3477	5454266	653387	0.089403	0.494465	89.15	16	PASS
11	heihe_x4	colamd	1	124395	653387	15.0578	9838584	653387	0.092550	1.241740	139.45	16	PASS
12	heihe_x16	natural	1	485250							16452.81	20	fill_overflow
13	heihe_x16	amd	0	485250	2481548	11.0780	27490538	2481548	0.454577	4.742845	405.65	20	PASS
14	heihe_x16	amd	1	485250	2481548	11.0780	27490538	2481548	0.506251	4.739916	407.68	20	PASS
15	heihe_x16	colamd	1	485250	2481548	20.6337	51203610	2481548	0.427978	11.708146	678.99	20	PASS
```

## Cross-references

- [docs/adr/0005-klu-spike-decision.md](../adr/0005-klu-spike-decision.md) — ADR-0005 (4-branch decision tree from this verdict)
- [docs/adr/0004-maxl-sweep-decision.md](../adr/0004-maxl-sweep-decision.md) — ADR-0004 (SPGMR maxl Optional-knob; baseline wall anchor)
- [openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md](../../openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md) — capability spec (gitignored under `openspec/changes/`)
- [.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md](../../.review-evidence/p8tune-klu-spike-pr-a/SWEEP_RESULTS.md) — PR-A 16-cell evidence narrative
- [.review-evidence/p8tune-klu-spike-pr-a/SPEC_AMENDMENTS.md](../../.review-evidence/p8tune-klu-spike-pr-a/SPEC_AMENDMENTS.md) — REQ-5 + REQ-7 amendments landed in PR-A
- [tools/p8tune.D/cn_node_ram.h](../../tools/p8tune.D/cn_node_ram.h) — pinned `CN_NODE_RAM_BYTES` constant (RSS axis denominator)
- [tools/p8tune.D/spgmr_baseline_walls.h](../../tools/p8tune.D/spgmr_baseline_walls.h) — pinned SPGMR per-step wall constant (wall axis numerator anchor)
