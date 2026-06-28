## 工作情况说明（Merge 前）

- **关联 Issue**：#364（p8tune-spgmr-maxl epic #362 的 PR-A of 6）
- **PR**：#370
- **冻结提交**：`c6db15f9510c68cb8854f931f821a4faa654a655`
- **PR base**：`baseline/p8tune`（继承 PR-0 #369 merged at `2582352523`）

### 背景与目标

PR-0 corrected 了 future-candidate gate citations 到 `ncfn_candidate ≤ 7/51`。PR-A 把这些 7/51 production floors formalize 为 citable baseline doc + tool + server smoke artifact，作为 maxl 60-cell sweep（PR-D）的 G4/G6/G7 gate 参考集 + spgmr-maxl-env-hook（PR-C）的 G3 4-way CI gate bit-identical anchor。

**Plan A path 节省 ~5.5h server compute**：SHUD `7a1dc8f..37be0fe` 的 CVODE codepath 是 bit-identical（cleanup tail 仅 revert PR-D 的 PREC_LEFT 改动 + identity preconditioner 删除），所以 Step 1 PR-B aggregator output (`n8_profile_verdict.md` §3.1) 18-cell 表可直接复用。Plan B（18-cell server re-run）作为 spec-defined fallback，仅在 codepath diverge 时触发；本次 codepath check 显示 revert-of-PR-D only → Plan A 确认。

### 本次具体改动

| 类型 | 文件 | 改动 |
|---|---|---|
| 新 tool | `tools/p8tune/aggregate_clean_baseline.sh` | 434 行 bash strict mode + `--plan-a` 默认（从 §3.1 提取 15 keys 生成 5 张表）+ `--plan-b` stub（gate on missing scratch dir，exit 1）+ floor verification gate（heihe ncfn=7/ncfl=85/netf=0; heihe_x4 ncfn=51/ncfl=3620/netf=0 严格匹配） |
| 新 doc | `docs/p8tune/clean_prec_none_baseline.md` | 429 行，9 §sections：codepath-equivalence（SHUD diff embedded）+ submit-template-provenance（Plan B 模板 missing 已 ssh 确认；derivation path 已 document）+ raw-18-cell-table（15 canonical keys per canonical_15_keys.yaml）+ cross-N-invariance-table（30/30 cells PASS B1a S4 OMP-neutrality；N=4 retained as regression detector even though sweep matrix omits）+ roi-ratio-table（authoritative heihe_x4 nfeLS=30509 cite）+ solver-failure-table（cleaned-PREC_NONE floors enumerated；标注 "NOT 6/47 from Step 2 PR-F"）+ keliya-smoke-anchor（cn14 SHUD 37be0fe build；rivqdown SHA12 1bfe6a30856e + 15-key snapshot；PR-C G3 4-way CI gate contract）+ decision-input-table（hard-evidence ncfl>0 satisfied → full 60-cell sweep verdict input）+ mode-C-tune-reference（SHUD pin equivalence proof cite per glossary D9） |
| 修 fix | `docs/p8pre/capstone.md` §5.1 | 3-row ratio cells `4.527 → 4.526`（heihe_x4 N=8 nfeLS/nfe）+ 5-line HTML 注释 cite `n8_profile_verdict.md` §3.4 derivation。PR-0 review-correctness deferred note bundled in |
| 本地 fix（gitignored） | `openspec/changes/p8tune-spgmr-maxl/proposal.md` L21 | `3-way equivalence → 4-way equivalence`，align design D0/D8 G3 + spec spgmr-maxl-env-hook L51-61 + tasks §3.7 (unset/empty/"0"/"5") |

**Diff stats**：3 tracked files (873 insertions + 3 deletions) + 1 local-only proposal.md fix。
**Scope guarantee**：0 .c/.cpp/.h/.sh source；0 Makefile；0 SHUD submodule pointer change（仍 `37be0fe`）。

### 测试与验证

| 验证 | 结果 |
|---|---|
| openspec validate strict | PASS (`Change 'p8tune-spgmr-maxl' is valid`) |
| `bash aggregate_clean_baseline.sh --plan-a` | exit 0，5 tables emit，floor gate PASS |
| `bash aggregate_clean_baseline.sh --plan-b` (stub) | exit 1 on missing scratch dir (correct guard) |
| Plan A codepath equivalence | `git diff 7a1dc8f..37be0fe -- cvode_config.cpp *precond*` = revert-of-PR-D only confirmed |
| keliya smoke server-side | Slurm job 9620 on cn14, ExitCode 0:0, wall 88s（build 60s + run ~30s）|
| keliya smoke artifact | rivqdown SHA12=`1bfe6a30856e` + 15-key cvode_stats snapshot 完整 |
| Cross-N invariance N=1/4/8 | 30/30 cells bit-identical per §3.1 |
| PR-0 deferred fixes | capstone §5.1 4.527→4.526 applied; proposal.md L21 4-way applied (local) |
| Scope check | 3 doc/tool files only; no source/Makefile/SHUD pointer change |
| CI | 5/5 PASS (setup / build-and-compare keliya / asan-ubsan keliya / asan-ubsan qhh / tools-tests) |

### Review 与修复闭环

| Phase | Reviewer | 轮次 | 结论 |
|---|---|---|---|
| 4 cross-review | review-correctness + review-spec-compliance + review-integration（3 并行）| round 1 @ 3494c6ac | 三 CLEAN，0 actionable findings + 7 non-blocking notes |
| 4.5 verifier gate | (空：0 candidates) | round 1 | — |
| Post-round-1 fix | orchestrator-direct doc patch | — | 1-line fix @ c6db15f: addr correctness N1 (`./build/shud→./shud`) + integration N1 (OMP build 注解) |
| 6/6.2/6.5 | — | — | SKIPPED（CI-only-equivalent doc fix per phase-flow 8）|
| 7 final review (Gap Sweep) | independent-final @ c6db15f | — | CLEAN, Reject-When 精度门应用, APPROVE merge |

**Phase 6 修复**：1 orchestrator-direct CI-only-equivalent doc 修复（1-char path correction + 1-line OMP scope 注解）。

### 兼容性、风险与已知限制

- **PR-A 是 epic dependency hub**：PR-B/C/D/E 都依赖此 baseline anchor
  - PR-C G3 4-way CI gate：`unset/""/"0"/"5"` 4 种 invocation MUST bit-identical to `rivqdown SHA12 = 1bfe6a30856e` + 15-key snapshot（cn14 toolchain pin）
  - PR-D 60-cell sweep mode = "full sweep"（hard-evidence ncfl>0 satisfied）；G6 no-solver-regression 用 §solver-failure-table 作 anchor
  - PR-E aggregator T1-T8 用 §mode-C-tune-reference 作 baseline reference
- **SHUD submodule 未触**：pin 保持 `37be0fe`
- **Plan B template missing**：spec-defined fallback path 已 document derivation provenance；如未来 codepath diverge 需 Plan B，参考 `docs/p8tune/clean_prec_none_baseline.md §submit-template-provenance` 派生
- **已知 limitation（informational, 不阻塞 merge）**：
  - integration N2 deferred 给 PR-E：T1 rep triplication（3 identical median rows per (case, N)）；PR-E aggregator join 需 dedupe on (case, N) before rep-level statistics
  - aggregator script 无 SPDX-License-Identifier header（项目无此前例；implementer 已 follow `tools/p1e_aggregate_*.sh` 实际惯例）

### 维护者关注点

- PR-A merge 后立即触发 PR-B（verdict gate doc，~1-2h Mac）+ PR-C（SHUD source env-var hook，~4-6h with 4-way CI gate）。这两个 PR 可并行
- PR-D 等 PR-C merge 后才能 sweep
- PR-E 等 PR-D 完整产出后聚合 + 写 ADR-0004
- 无额外人工关注点

---

🤖 Generated with [Claude Code](https://claude.com/claude-code)
