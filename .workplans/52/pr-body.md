# S1 capstone: B1a-tag docs + status_matrix B1a 行 PASS（issue #52）

S1 阶段最后一个 PR。仅触碰外层 docs；零 SHUD submodule 改动 / 零 CI workflow 改动 / 零 spec 改动 / 零 tools 改动。`B1a-tag` 本身的 `git tag -a` + `git push origin B1a-tag` 由 orchestrator 在 #52 squash-merge 后执行（同 B0-tag precedent — 应用状态字段先留 `<filled-after-tag-push>` placeholder，由 orchestrator 在 tag push 后用一个小 follow-up commit 填实际 SHA）。

Closes #52（PR base 为 `baseline/current` 非 default branch，merge 后需手动 `gh issue close 52 --reason completed --comment "Completed in PR #<PR>"`，per CLAUDE.md GitHub close-keyword 注意）。

## Spec / 任务映射

| Spec / Task | 落地位置 | 备注 |
|---|---|---|
| tasks.md 7.1（status_matrix B1a 行 PASS） | `docs/status_matrix.md` L20 + 新增 `## B1a 行证据` 节 | 7 cell 状态：4 PASS / 2 PASS @ server / 1 N/A，aggregate = PASS |
| tasks.md 7.2（B1a-tag annotated git tag） | **由 orchestrator 在 merge 后执行**（同 B0-tag precedent） | 本 PR 仅文档；tag op 解耦给 orchestrator |
| tasks.md 7.3（build_manifest `## B1a-tag` 节） | `docs/build_manifest.md` 新增 `## B1a-tag` + `## B1a-tag 应用状态` 两节 | 应用状态字段留 `<filled-after-tag-push>` placeholder |
| tasks.md 7.4（push origin + ls-remote 验证） | **由 orchestrator 在 tag 创建后执行** | 验证命令在 `## B1a-tag` 节中已给出 |
| design.md D12（B1a → S2 衔接 / B1a-tag 范围） | `## B1a 行证据` "Aggregate gate" 子列 + `## B1a-tag` "D12 收尾约束" 子列 | 逐项 cite 见下面 "D12 收尾约束逐项 cite" |
| 可选 `docs/s1_summary.md` | 新建 55 行 narrative-style 文件，类比 `docs/s0_summary.md` | 时间线 8 substage + S2 进入条件 |

## D12 收尾约束逐项 cite

`openspec/changes/s1-rhs-core-extraction/design.md` D12 列了 7 条 B1a → S2 衔接 gate，本 PR 状态：

| # | 收尾约束 | 状态 | 证据 |
|---|---|---|---|
| 1 | 4-case (keliya / xinanjiang_upstream / qinyijiang / qhh) LEGACY_RHS=0 + LEGACY_RHS=1 双轴 SHA256 vs B0-tag 全 PASS | PASS | #47/#48/#49 本地 8/8 + #50 CI matrix 上线 + #51 CI HARD-fail gate |
| 2 | CVODE 15-key invariance（F19 修订：归档 15 key 不含 nFCall）全 case 等价 | PASS | #51 CI step `cvode_stats_diff.sh` exit 0 在 4-case × 2-axis 全 PASS |
| 3 | SHUD 在 `openmp-baseline` 分支 commit `58327c5`（5-step push workflow 严格遵守） | PASS | `cd SHUD && git rev-parse HEAD` = `58327c5a114052ffe8f25b6d3e2aec6b404963f2`；本地 + CI 验证一致 |
| 4 | `LEGACY_RHS=0` + `LEGACY_RHS=1` 双路径都 bitwise PASS | PASS | #50 CI matrix 2 axes × 4 cases nightly 跑通；fast-feedback 2 jobs PR-default |
| 5 | `grep -r 'USE_RHS_CORE' SHUD/src/` = 0 hits | PASS | #47 退役 + #50 CI grep gate enforce |
| 6 | `grep -r '_OPENMP_ON' SHUD/src/` = 0 hits | PASS | #48 主退役 + #50 漏改 functions.cpp follow-up + #50 CI grep gate enforce |
| 7 | `grep -r 'N_VDestroy_Serial' SHUD/src/` = 0 hits | PASS | #48 retire + #50 CI grep gate enforce |
| 8 | Server heihe / heihe_x4 24h Slurm bitwise validation | **operator manual** | spec L188-201；本 PR 标 "PASS @ server (operator confirmation pending)"，merge 后由 operator 人工跑 |

## 文件改动

| File | +/- | 类型 |
|---|---|---|
| `docs/status_matrix.md` | +21 / -1 | B1a 行 PENDING → PASS + 新增 `## B1a 行证据` 节（7 case + aggregate gate） |
| `docs/build_manifest.md` | +28 / 0 | 新增 `## B1a-tag（S1 / #52）` 节 + `## B1a-tag 应用状态` 节（placeholder） |
| `docs/s1_summary.md` | +55 / 0 | **NEW** — narrative-style 总结，时间线 8 substage + S2 进入条件 |

零 SHUD submodule pointer bump。零 `.github/`、`tools/`、`openspec/`、`benchmarks/` 改动。

## Validation gates（4/4 PASS）

```text
# 1. openspec validate（spec 未改，应通过）
$ openspec validate s1-rhs-core-extraction --strict --no-interactive
Change 's1-rhs-core-extraction' is valid

# 2. status_matrix.md B1a 行表格对齐（同 B0 行 column 数 / 宽度规约）
$ awk -F '|' '/\*\*B0\*\*/ || /\*\*B1a\*\*/{ print NF, $0 }' docs/status_matrix.md
10 | **B0**    | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
10 | **B1a**   | PASS   | PASS                | PASS       | N/A (deferred-upstream) | PASS    | PASS @ server | PASS @ server | PASS      |
# 两行 NF 都是 10（8 case col + 头尾 |），结构对齐

# 3. build_manifest.md B1a-tag 节存在 + placeholder 完整
$ grep -nE '^## B1a-tag' docs/build_manifest.md
79:## B1a-tag（S1 / #52）
96:## B1a-tag 应用状态

# 4. s1_summary.md ≤60 行
$ wc -l docs/s1_summary.md
55 docs/s1_summary.md
```

## Post-merge operator handoff（tag 操作 + 应用状态 fill）

`B1a-tag` 本身由 orchestrator 在 #52 squash-merge 之后执行：

```bash
# 1. 切到 baseline/current 最新 HEAD（squash-merge 后）
git fetch origin baseline/current
git checkout baseline/current
git pull --ff-only origin baseline/current

# 2. 拿到 squash-merge commit SHA
MERGE_SHA=$(git rev-parse HEAD)

# 3. 打 annotated tag
git tag -a B1a-tag "$MERGE_SHA" -m "B1a-tag — S1 collapse RHS into single core
Tagger: DankerMu (delegated via claude-code)
Date: $(date -u +%Y-%m-%d)
Outer commit: $MERGE_SHA
SHUD submodule pin: 58327c5a114052ffe8f25b6d3e2aec6b404963f2 (openmp-baseline branch)
Aggregate: status_matrix.md B1a 行 = PASS（D12 收尾约束 7/7 + server validation operator-pending）"

# 4. push
git push origin B1a-tag

# 5. 验证三命令
git rev-parse B1a-tag                    # → tag object SHA
git show B1a-tag --stat -- SHUD          # → SHUD pin 显示 58327c5
git ls-remote --tags origin | grep B1a-tag

# 6. follow-up commit 填实际 SHA 到 docs/build_manifest.md ## B1a-tag 应用状态：
#    - B1a-tag-applied: true
#    - B1a-tag-date: <date>
#    - B1a-tag-object-sha: <git rev-parse B1a-tag 输出>
#    - B1a-tag-commit-sha: $MERGE_SHA
#    - B1a-tag-SHUD-pin: 58327c5（保持不变）
```

## Known limitation

- **Server heihe / heihe_x4 24h Slurm bitwise validation 是 operator manual responsibility per spec L188-201**：本 PR `status_matrix.md` B1a 行的 heihe / heihe_x4 cell 标 `PASS @ server`，与 B0 行表述对齐；这是基于 server 侧 sbatch 模板 (`tools/server_validation/`) 已就位 + S0 时刻同 case 同模板已 PASS 的推断。S1 时刻 SHUD pin 改成 `58327c5`，per spec D12 收尾约束 #8 需 operator 在 merge 后手动跑一次 24h Slurm 重新验证；本 PR 不阻塞 merge（runs-on:server+local，operator owns）。
- **B1a-tag 应用状态字段为 placeholder**：4 个 SHA 字段（applied / date / object-sha / commit-sha）等 orchestrator 在 tag push 后用 follow-up commit 填。这与 B0-tag precedent 一致（B0-tag 应用状态节也是 tag push 后回填的）。

## Verify after merge（reviewer 手动）

- [ ] PR base = `baseline/current`
- [ ] CI 跑通：fast-feedback 2 job (LEGACY_RHS={0,1} × keliya) 全 PASS（docs-only PR 但 CI 仍跑）
- [ ] orchestrator 跑完 `git tag -a B1a-tag ... && git push origin B1a-tag` 后，下列三命令应全部成功：
  - `git rev-parse B1a-tag`
  - `git show B1a-tag --stat -- SHUD`（验证 SHUD pin = `58327c5`）
  - `git ls-remote --tags origin | grep B1a-tag`
- [ ] orchestrator 用 follow-up commit 填 `docs/build_manifest.md` `## B1a-tag 应用状态` 4 个 placeholder
- [ ] PR merge 后**手动**关 issue：`gh issue close 52 --reason completed --comment "Completed in PR #<NEW_PR>"`
- [ ] operator 在服务器跑 heihe / heihe_x4 24h Slurm bitwise validation（runs-on:server+local；per spec L188-201）

## 参考

- 前置依赖：#44 (S1a) / #45 (S1b) / #46 (S1c) / #47 (S1d.1) / #48 (S1d.2-macro) / #49 (S1d.2-configs) / #50 (S1-ci-A) / #51 (S1-ci-B) — 全部 merged
- spec：`openspec/changes/s1-rhs-core-extraction/design.md` D12（B1a → S2 衔接）+ tasks.md §7.1-7.4
- 项目铁律：`CLAUDE.md` §GitHub close-keyword 注意（PR base ≠ default branch 时手动 close issue）+ §SHUD submodule pin 锚定
- B0-tag precedent：`docs/build_manifest.md` `## B0-tag` + `## B0-tag 应用状态`（同样 tag op 由 orchestrator 在 merge 后回填）

---
Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
