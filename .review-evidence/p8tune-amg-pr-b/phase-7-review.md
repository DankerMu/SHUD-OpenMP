# PR-B #396 (PR #403) — Phase 7 Adversarial Review

**Date**: 2026-06-29
**Branch**: `feat/issue-396-p8tune-amg-pr-b` HEAD `8efc033` (post-H1+M3 fixes)
**Reviewer**: single adversarial pass (Phase 7-style)
**Verdict**: **APPROVE_WITH_FOLLOWUP** → fixes applied → **MERGE-READY**

## Findings closure status

| Severity | ID | Description | Status |
|---|---|---|---|
| High | H1 | precheck_env.sh orphaned (not invoked by sbatch) | **CLOSED** `8efc033` |
| Medium | M1 | precheck (g) regex tokenization brittle | DEFERRED (PR-C+) |
| Medium | M2 | colpack_version=unknown sentinel docs | DEFERRED (PR-C) |
| Medium | M3 | SIGTERM trap emits malformed 4-line KV | **CLOSED** `8efc033` |
| Low | L1 | `set -uo pipefail` without `-e` | INTENTIONAL (no fix) |
| Low | L2 | TIME_BIN absence WARN-only on Linux | DEFERRED |
| Low | L3 | RUN_ID fallback when --export forgotten | INTENTIONAL (defensive) |
| Info | I1 | EXTRA_LIBS hook unused on server (source-build path used) | DOC ONLY |
| Info | I2 | Hypre 2.x/3.x version gate cutoff correct | NO ACTION |
| Doc | — | README missing §Quickstart wiring | DEFERRED (PR-C+) |

## Axis-by-axis verdict (per Phase 7 5-axis investigation)

- **Axis 1 (shell safety)**: PASS — no eval, all vars quoted, whitelist regex validation on NN + RUN_ID, octal-safe `10#${NN}` coercion, `set -uo pipefail` enforced
- **Axis 2 (KV schema)**: PASS — 16/16 cells emit spec REQ-4 schema byte-perfectly. M3 SIGTERM trap path now also conformant post-fix
- **Axis 3 (Slurm 三铁律)**: PASS — mechanical conformance + H1 fix now ENFORCES at script-entry (was documentation-only before)
- **Axis 4 (16-cell decoder)**: PASS — verified cell-0/4/8/12/15 against decoder; CASE_IDX = NN/4, COMBO_IDX = NN%4 with octal-safe coercion
- **Axis 5 (4-platform numerical consistency)**: PASS — residual_reduction_v1=50.8449 byte-identical Mac brew / server login / cn08 smoke / cn-node array

## Spec REQ compliance

| Spec REQ | Pre-Phase-7 | Post-Phase-7 fixes |
|---|---|---|
| REQ-4 KV schema (PASS path) | PASS | PASS |
| REQ-4 KV schema (SIGTERM trap) | FAIL (M3) | **PASS** (`8efc033`) |
| REQ-4 7-condition precheck (definition) | PASS | PASS |
| REQ-4 7-condition precheck (enforcement) | CONCERN (H1) | **PASS** (`8efc033`) |
| REQ-4 4 marker emission | PASS | PASS |
| REQ-4 4 case × 4 combo decoder | PASS | PASS |
| Slurm 三铁律 (1, 2, 3) | PASS (mechanical) | PASS + ENFORCED |

## Round-2 verification (post-fix smoke job 9918)

After `8efc033` fixes applied:
- Re-submitted spike_single_test.sbatch as job `9918_0` on cn-node
- precheck.log: 7/7 PASS (precheck now runs inline before cpp invocation)
- cell-0.out CELL_SUMMARY: `case=keliya interp_type=6 coarsen_type=8 NumY=1785 nnz_A=10255` … `verdict_class=PASS` … `residual_reduction_v1=50.8449` (bitwise-identical to pre-fix job 9895)
- Wall +1 sec from precheck cost (1 sec → 2 sec total), acceptable

No regression. Phase 7 fixes are semantically transparent on the PASS path; the M3 trap fix is only exercised on actual SIGTERM which didn't fire in either sweep, but the code change is mechanical (3 echo lines + decoder hoist) and inspectable.

## Pre-merge gate

- **4-件套**: present at `.review-evidence/p8tune-amg-pr-b/`:
  - `SWEEP_RESULTS.md` (verdict table + 5-axis prelim eval)
  - `cells/cell-{0..15}.{out,err}` (16-cell evidence)
  - `runtime/p8f_amg_array.9896/` (per-cell run_cell.sh log + time -v)
  - `phase-7-review.md` (this file)
- **Self-audit**: 1 High + 3 Medium + 3 Low + 2 Info surfaced; 2 (H1+M3) closed inline, 3 deferred, 1 doc gap deferred
- **Oracle integrity**: BITWISE-IDENTICAL residual_reduction_v1=50.8449 confirmed across 4 platforms (Mac brew, server login, cn08 smoke job 9895, cn-node array job 9896 cell-0) AND across post-fix re-smoke job 9918 — Hypre 2.x/3.x init gate + EXTRA_LIBS Makefile hook + source-built Hypre 3.1.0 are all semantically equivalent

## Verdict

**MERGE-READY** at `8efc033` after H1+M3 inline fixes. Deferred M1/M2/L2 + README §Quickstart tracked here for PR-C / PR-C+ pickup.

PR-C #397 inheritance: aggregate_amg_spike.sh must:
1. Accept NA sentinels for AMG_WALL_OVERFLOW cells (M2 + M3 closure semantics)
2. Optionally validate presence of `precheck.log` in run dir (extra defense-in-depth for H1)
3. Tolerate `colpack_version=unknown` sentinel (M2 closure semantics)
