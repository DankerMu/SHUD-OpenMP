## 工作情况说明（Merge 前）

- **关联 Issue**：#366（p8tune-spgmr-maxl epic #362 的 PR-C of 6 — 高强度 source-change PR）
- **PR**：#372
- **冻结提交**：outer `768c905f8f078e7ece27bc4d8e4efb4ab0a1b825` + SHUD `6ce17d6d479d1c1e8b8c63d2c1367ed9dca9ee52`
- **PR base**：`baseline/p8tune`（继承 PR-B #371 merged at `46a5ee32`）
- **Phase 0.5 强制门**：Invariant Matrix D15 已 author + Phase 0.5 fixture review APPROVE（7/7 PASS）

### 背景与目标

PR-A 建 cleaned-PREC_NONE 18-cell baseline + cn14 keliya smoke anchor (rivqdown.dat SHA12 `1bfe6a30856e` + 15-key snapshot)。PR-B verdict gate 决定 PR-D 走 full 60-cell sweep。**PR-C provide PR-D 需要的 runtime maxl knob**（env var `SHUD_SPGMR_MAXL`），同时**保证 default-unset bit-identical 等价于 SHUD 37be0fe**（这样 PR-A 18-cell baseline reuse 仍 valid，且 PR-A doc 不需要 update）。

**4-way default-equivalence**（`unset` / `""` / `"0"` / `"5"` 全 bit-identical）兑现 SUNDIALS docs (`maxl ≤ 0` → default 5) + IM D15 governing invariant: default-unset MUST produce bit-identical CVODE output to production baseline; any opt-in MUST honor SUNDIALS semantics + preserve PREC_NONE; any invalid value MUST fail-fast via `myexit(ERRCVODE)`。

### 本次具体改动

| 类型 | 文件 | 改动 |
|---|---|---|
| SHUD source | `SHUD/src/Equations/cvode_config.cpp` | +67/-2 行（单文件 source surface per spec L67-72 + IM L244）：(a) `<errno.h>` include 显式添加；(b) `static int get_spgmr_maxl_from_env(void)` helper L248-308（strict whitelist {0,5,10,15,20,30} + 3-belt 校验：per-char `[0-9]` pre-scan + no-leading-zero guard + strtol endptr/errno cross-check；invalid → fprintf stderr + myexit(ERRCVODE) BEFORE SPGMR allocation；val ∈ {5,10,15,20,30} → fprintf stdout `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` + fflush + return val；NULL/""/"0" → silent return 0）；(c) call-site L324: `LS = SUNLinSol_SPGMR(udata, PREC_NONE, get_spgmr_maxl_from_env(), sunctx)` — PREC_NONE 保留 |
| Outer SHUD pointer | `SHUD` | `37be0fe → 6ce17d6`（单行 submodule pointer bump） |
| 本地 fix（gitignored） | `openspec/changes/p8tune-spgmr-maxl/design.md` D15 | Invariant Matrix authored: governing invariant + source-of-truth identity (PR-A SHA12 `1bfe6a30856e` + 15-key per `canonical_15_keys.yaml`) + 8 surface categories + 11 regression rows + boundary-surface checklist |

**Diff stats**：outer 1 file (1 line); SHUD 1 file (+67/-2)。
**Scope guarantee**：0 Makefile/header/SUNDIALS-vendored/sibling source；0 CVODE 容忍度/步长/precond 变更；0 B1a/B1b baseline modification；SHUD master 分支未触（`git branch -r --contains 6ce17d6` = `openmp-baseline` only）。

### 测试与验证

| 验证 | 结果 |
|---|---|
| openspec validate strict | PASS (`Change 'p8tune-spgmr-maxl' is valid`) |
| Mac `make shud` + `make shud_omp` | 两 exit 0 |
| Mac grep `PREC_LEFT|CVodeSetPreconditioner|CVodeSetLSetupFrequency|MD_precond_identity` | 0 matches (no regression per spec L41-45) |
| Mac single-file surface check | SHUD diff = `src/Equations/cvode_config.cpp` only；outer diff = `SHUD` pointer only |
| Implementer-side parser unit tests | 19/19 PASS（包括 strict-whitelist edge cases `+5` / `05` / ` 5` / `5 ` / `5x` / `10.0` / `-1` / `foo` / `7` / `50` / overflow `10000000000000000000`）|
| Server cn14 build with `SHUD_ENABLE_PROFILE=1` | exit 0 |
| Server Slurm job 9626 (cn14, CPU partition, /scratch/.p8tune-runs/pr-c-g3-gate/, Slurm 三铁律 compliant) | functional binary execution |
| **G3 4-way bit-identical CI gate** | **PASS** — all 4 runs SHA12 = `1bfe6a30856e` matches PR-A anchor exactly |
| G3 15-key cvode_stats | all 4 runs bit-identical to PR-A anchor (nfe=112248, nfeLS=116421, ..., leniwLS=42) |
| G3 cross-run cmp byte-equivalence | run1 == run2 == run3 == run4 byte-identical |
| G3 stdout provenance log discipline | 0/0/0/1 lines per IM D15 L235-238（unset/""/"0"=silent; "5"=1 log line）|
| CI | 5/5 PASS @ 768c905 (setup 4s / build-and-compare keliya 1m4s / asan-ubsan keliya 36s / asan-ubsan qhh 5s / tools-tests 9s) |

### Review 与修复闭环

| Phase | Reviewer | 轮次 | 结论 |
|---|---|---|---|
| 0.5 fixture review (IM 强制门) | 1 reviewer focused on D15 | — | APPROVE 7/7 |
| 4 cross-review (high-intensity 4-reviewer 全 panel) | review-correctness + review-spec-compliance + review-integration + review-security-perf（4 并行）| round 1 @ 768c905 | 四 CLEAN，0 actionable findings + 10 non-blocking informational/disclosure notes |
| 4.5 verifier gate | (空：0 candidates) | round 1 | — |
| 5/6/6.2/6.5 | — | — | SKIPPED（no findings to fix；cosmetic L259 comment 显式 defer 避免又一轮 SHUD push cycle for non-binding annotation）|
| 7 final review (Gap Sweep) | independent-final @ 768c905 | — | CLEAN, Reject-When 精度门应用, APPROVE merge |

**Phase 6 修复**：0（clean single-round flow，无 SHA drift；1 cosmetic deferred annotation）。

### 兼容性、风险与已知限制

- **PR-C 是 PR-D 直接 dependency hub**：PR-D #367 60-cell sweep 用 `SHUD_SPGMR_MAXL=<k>` per cell 调用本 hook；provenance log format `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` 完全匹配 PR-D tasks §4.6 grep target
- **PR-A baseline reuse 兼容性 PROVEN**：G3 evidence run-1 (unset) = SHA12 `1bfe6a30856e` 与 PR-A anchor bit-identical；PR-A 18-cell 表 + decision-input-table 不需要 update
- **PR-E aggregator schema preserved**：PR-C 不修改 cvode_stats.txt emission；15-key canonical_15_keys.yaml schema unchanged
- **SHUD master branch isolation**：`6ce17d6` 仅在 origin/openmp-baseline（NEVER master per project rule）；forward-only linear-history from 37be0fe
- **B1a-tag / B1b 隔离**：no rebase / no force-push / no historical rewrite
- **CVODE/SUNDIALS pin 稳定**：SUNDIALS 6.0.0 pin unchanged；唯一新 include 是 `<errno.h>` (stdlib)
- **OMP path neutrality**：env read 在 SetCVODE serial init context（called from `shud.cpp:177` before time-loop OMP region）；stateless function；N=1/N=8 identical behavior
- **已知 limitations**（non-blocking informational, 不阻塞 merge）：
  - cosmetic stale "L259" comment ref in helper（实际 call-site 已 shift 到 L324 due to helper insertion）— 留作 future cleanup PR; 不修以避免又一轮 SHUD push
  - sbatch script 内 grep parser bug（用 `^${KEY}[[:space:]]` 但 cvode_stats 是 `key=value` 格式）— g3_verdict.md L55-63 honest disclosure；不影响 artifact bits（已 manual re-verify PASS）；future cleanup PR 可以修 sbatch parser 或重跑 gate cleanly for archival hygiene

### 维护者关注点

- PR-C merge 后 PR-D #367（60-cell server sweep）可启动；需要约 6h server compute (5 maxl × 2 case × 2 N × 3 rep = 60 cells × ~6 min/cell on cn14)
- PR-E #368 等 PR-D 完整产出后聚合 8-table verdict + 写 ADR-0004（4-branch outcome adjudication）
- 无额外人工关注点

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
