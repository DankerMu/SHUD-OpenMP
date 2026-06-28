# PR-C G3 4-way bit-identical CI gate — VERDICT

**SHUD pin under test**: `6ce17d6` (openmp-baseline; pushed to origin/SHUD-System/SHUD/openmp-baseline)
**PR-A anchor pin**: `37be0fe` (cleaned-PREC_NONE baseline keliya smoke from `docs/p8tune/clean_prec_none_baseline.md` §keliya-smoke-anchor)
**Anchor rivqdown.dat SHA12**: `1bfe6a30856e`
**Server / node**: `frd_muziyao@210.77.77.22:32099` / `cn14`
**Slurm job**: `9626` (CPU partition, cn14, wall 00:05:55, ExitCode 1:0 — exit due to sbatch grep parser bug, NOT G3 fail; manually re-verified below)

## Verdict: G3 PASS

All 4 invocations on the patched SHUD `6ce17d6` binary produce bit-identical artifacts to PR-A cleaned-PREC_NONE anchor (cn14 reproducibility).

### 4-way rivqdown.dat SHA12 (Invariant Matrix D15 evidence/audit + spec scenario "keliya smoke 4-way equivalence")

| Run | Invocation | rivqdown SHA12 | Anchor match |
|---|---|---|---|
| 1 | `unset SHUD_SPGMR_MAXL && ./shud keliya` | `1bfe6a30856e` | PASS |
| 2 | `SHUD_SPGMR_MAXL= ./shud keliya` (empty) | `1bfe6a30856e` | PASS |
| 3 | `SHUD_SPGMR_MAXL=0 ./shud keliya` | `1bfe6a30856e` | PASS |
| 4 | `SHUD_SPGMR_MAXL=5 ./shud keliya` | `1bfe6a30856e` | PASS |

Cross-run `cmp` byte-equivalence: run1 vs run2 IDENTICAL; run1 vs run3 IDENTICAL; run1 vs run4 IDENTICAL.

### 4-way 15-key cvode_stats.txt (per canonical_15_keys.yaml)

All 4 runs produce identical cvode_stats.txt values matching anchor:

| Key | Value | Anchor | Match |
|---|---:|---:|---|
| nfe | 112248 | 112248 | PASS |
| nfeLS | 116421 | 116421 | PASS |
| nni | 112247 | 112247 | PASS |
| nli | 116421 | 116421 | PASS |
| nsetups | 0 | 0 | PASS |
| netf | 5 | 5 | PASS |
| nst | 110917 | 110917 | PASS |
| npe | 0 | 0 | PASS |
| nps | 0 | 0 | PASS |
| ncfn | 205 | 205 | PASS |
| ncfl | 42 | 42 | PASS |
| lenrw | 23294 | 23294 | PASS |
| leniw | 53 | 53 | PASS |
| lenrwLS | 21474 | 21474 | PASS |
| leniwLS | 42 | 42 | PASS |

### Stdout provenance log discipline (Invariant Matrix D15 L235-238)

| Run | Env state | `[CVODE] SPGMR maxl=` log lines | IM row | Expected | Match |
|---|---|---:|---|---:|---|
| 1 | unset | 0 | L235 (silent default) | 0 | PASS |
| 2 | "" empty | 0 | L236 (silent default = unset) | 0 | PASS |
| 3 | "0" | 0 | L237 (silent default; bit-identical to unset) | 0 | PASS |
| 4 | "5" | 1 | L238 (provenance emitted for valid value; artifact unchanged) | 1 | PASS |

### Notes on sbatch job ExitCode 1:0

The Slurm job exited 1 due to a bug in the sbatch script's cvode_stats parser (used grep `^${KEY}[[:space:]]` but actual format is `key=value`). The parser bug caused false-fail flags on 15-key check but DID NOT affect:
- Run execution (all 4 ./shud invocations exited 0)
- Artifact capture (all rivqdown.dat + cvode_stats.txt captured correctly)
- SHA12 computation (all 4 = anchor)
- Cross-run cmp byte-equivalence (all 4 identical)

Manual re-verification (this file) confirms all 15 keys + SHA12 PASS for all 4 runs. The sbatch script was not re-run after parser fix because the captured artifacts already definitively prove G3 PASS.

### Evidence archive

- `g3_4way_evidence.tar` (80KB) — full sbatch script + 4× stdout + 4× stderr + 4× cvode_stats.txt + g3_9626.{out,err}
- Server source: `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/pr-c-g3-gate/`
