## Conventions

本 spec 沿用 `s2-semantic-merge` 的 Conventions（Case Scope / Lake-related 输出文件清单 / B0-tag 引用），不重复列出。

PR-12 内部顺序硬约束（design.md D11，违反则 rollback）：
1. PR-12 内顺带 `openspec archive b1a-finalization` → 4 specs PROMOTED 到 `openspec/specs/<capability>/spec.md`（committed，diff 包含 4 个新增 spec 文件）+ change 文件夹 RELOCATED 到 `openspec/changes/archive/2026-06-20-b1a-finalization/`（local-only per `.gitignore` convention，匹配前两次 archive precedent `2026-06-19-s1-rhs-core-extraction/` + `2026-06-19-s2-pre-spec-housekeeping/`）
2. 6 case 全量 bitwise + grep gates 验证（CI PR-12 阶段）
3. PR-12 merge 入 `baseline/B1a`
4. `git push origin baseline/B1a:main`（fast-forward main HEAD）
5. `git tag -f B1a-tag <main-HEAD-commit> && git push origin B1a-tag --force`（tag 指向 main HEAD 上存在的 commit）
6. `gh api -X PUT repos/DankerMu/SHUD-OpenMP/branches/baseline/B1a/protection` 设 `lock_branch=true`
7. `docs/review-loop-log.jsonl` append capstone 记录
8. `CLAUDE.md` 双端 sync（本地 + rsync 服务器）

## ADDED Requirements

### Requirement: B1a 定版 6 case 全量 bitwise vs B0-tag

工程 SHALL 在 B1a capstone PR-12 中跑完整 6 case bitwise vs B0-tag 验证。每 case 跑 90-day truncation，`rivqdown.dat` + 所有 lake-related .dat + 12 张 snapshot bitwise + CVODE 15-key 全 PASS（master plan §3 B1a 验收 + §5 A1）。

#### Scenario: 6 case 全量 bitwise PASS

- **WHEN** B1a capstone PR-12 merge gate 检查
- **THEN** 4-case Mac (`keliya` / `xinanjiang_upstream` / `qinyijiang` / `qhh`) `rivqdown.dat` SHA256 SHALL 全部 byte-equal `benchmarks/<case>/B0_output/rivqdown.dat`
- **THEN** 服务器 Slurm 跑 `heihe` + `heihe_x4` `rivqdown.dat` + lake-related .dat (按 Conventions 清单) SHA256 SHALL 全部 byte-equal B0_output 归档
- **THEN** 12 张 snapshot (4-case × t=1d/10d/100d) bitwise vs golden binary SHALL exit 0 via `tools/compare_snapshot/compare_snapshot`
- **THEN** 6 case CVODE 15-key (`nfe / nfeLS / nni / nli / nsetups / netf / nst / npe / nps / ncfn / ncfl / lenrw / leniw / lenrwLS / leniwLS`) SHALL byte-equal B0-tag 归档（不含 `nFCall`）
- **THEN** `kashigeer` 仍为 `N/A (deferred-upstream)`（上游 X76 forcing 缺失，CI matrix 排除）

### Requirement: B1a 定版 grep gates 全 0 hits + CI workflow 最终状态

工程 SHALL 在 B1a capstone PR-12 中验证 SHUD 源码 grep gate 全部 0 hits：`MD_f_omp.cpp` 整文件不存在；`PassValue\b` 0 hits；既有 `_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 0 hits 持续生效；新增 `SHUD_LEGACY_OMP_RHS` 0 hits（master plan §5 S2 capstone + S3c.3 + S1 既有 gate）。

同时工程 SHALL 在 PR-12 验证 `.github/workflows/serial-baseline.yml` 达到 B1a 定版最终状态：matrix 已收敛为 `LEGACY_RHS=0` 单轴 + 6 case；包含全部 7 个 grep gate step（`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` / `SHUD_LEGACY_OMP_RHS` / `LEGACY_RHS` / `MD_f_omp.cpp 0 hits` / `PassValue\b 0 hits`）；包含 `docs/topology_manifest.yaml` schema 校验 step + adjacency fallback unit test step；fetch-tags / B0-tag smoke / snapshot 90d HARD-fail / CVODE 15-key / nm 符号 / SHA256 compare 全部保留。

#### Scenario: 所有 grep gate 0 hits

- **WHEN** B1a capstone PR-12 merge gate 检查
- **THEN** `ls SHUD/src/ModelData/MD_f_omp.cpp` SHALL exit non-zero
- **THEN** `grep -rn 'PassValue\b' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'SHUD_LEGACY_OMP_RHS\|LEGACY_RHS' SHUD/src/ SHUD/Makefile` SHALL = 0 hits
- **THEN** `grep -rn '_OPENMP_ON' SHUD/src/` SHALL = 0 hits（S1 已 enforce 持续生效）
- **THEN** `grep -rn 'USE_RHS_CORE' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'N_VDestroy_Serial' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'f_update_omp\|f_loop_omp\|f_applyDY_omp' SHUD/src/` SHALL = 0 hits（tree-wide）

#### Scenario: CI workflow 最终状态符合 B1a 定版

- **WHEN** B1a capstone PR-12 merge
- **THEN** `.github/workflows/serial-baseline.yml` `jobs.<j>.strategy.matrix` SHALL 不包含 `legacy_rhs:` 维度（只跑单轴 build）
- **THEN** CI workflow SHALL 包含 7 个 grep gate step（按 Requirement 描述列出的 7 个）
- **THEN** CI workflow SHALL 包含 `docs/topology_manifest.yaml` schema 校验 step + adjacency fallback unit test step
- **THEN** CI workflow case matrix SHALL = 6 case（kashigeer 排除）+ snapshot 90d HARD-fail + CVODE 15-key + SHA256 compare 持续生效

### Requirement: B1a-tag force-update 到 B1a final commit（push main → tag force-update 顺序）

工程 SHALL 在 B1a capstone PR-12 merge 后按以下严格顺序执行（顺序硬约束，design.md D11 + Conventions PR-12 顺序）：

1. PR-12 merge 入 `baseline/B1a`
2. 先 `git push origin baseline/B1a:main`（fast-forward main HEAD）
3. 然后 `git tag -f B1a-tag <main-HEAD-commit> && git push origin B1a-tag --force`（tag 指向 main HEAD 上存在的 commit）

理由：tag 永远指向 main HEAD 上存在的 commit，外部下游 fetch tag 不会拿到只在 baseline/B1a 上的 commit。force-update 操作 SHALL 在 capstone PR body 与 `docs/b1a_summary.md` 时间线明确记录；旧 commit `64569b3` 仍保留在 git history 作 S1d-end 历史 reference 可访问（master plan §3 B1a 定义 + 本 change design.md D2）。

#### Scenario: B1a-tag 指向 B1a final commit（push main 在 tag 之前）

- **WHEN** B1a capstone PR-12 merge 后按上述顺序执行
- **THEN** 第 2 步执行后 `gh api repos/DankerMu/SHUD-OpenMP/branches/main --jq '.commit.sha'` SHALL 等于 PR-12 squash-merge commit SHA
- **THEN** 第 3 步执行后 `git rev-parse B1a-tag^{}` SHALL 等于 B1a capstone PR-12 squash-merge commit SHA
- **THEN** `git rev-parse B1a-tag^{}` SHALL **不**再等于 `64569b3`
- **THEN** `git log 64569b3` SHALL exit 0（旧 commit 仍在 history，作 S1d-end snapshot 可访问）
- **THEN** B1a capstone PR body 与 `docs/b1a_summary.md` 时间线 SHALL 明确写明 "旧 B1a-tag commit `64569b3` 已被 force-update 到 `<new-commit>`"
- **THEN** `git tag -l --contains <new-commit>` SHALL 包含 `B1a-tag`，验证 tag 已落到新 commit

### Requirement: `status_matrix.md` B1a 行 IN-PROGRESS → PASS

工程 SHALL 在 B1a capstone PR-12 中修改 `docs/status_matrix.md` L20 B1a 行 7 case + aggregate：`IN-PROGRESS` → `PASS`（与 B0 行同 layout）；同步删除 L54 B1a 行证据段的 "2026-06-20 修订" disclaimer + 每 case 的 "S2/S3/S4 验证待补" 字样；evidence 列内容更新为 S0+S1+S2+S3+S4 完整覆盖。

#### Scenario: status_matrix B1a 行升级 PASS

- **WHEN** B1a capstone PR-12 merge
- **THEN** `docs/status_matrix.md` L20 B1a 行 7 case + aggregate SHALL 与 B0 行同 layout：`PASS` / `PASS` / `PASS` / `N/A (deferred-upstream)` / `PASS` / `PASS @ server` / `PASS @ server` / `PASS`
- **THEN** `docs/status_matrix.md` B1a 行证据 段 SHALL 不再包含 "2026-06-20 修订" 或 "S2/S3/S4 验证待补" 字样
- **THEN** evidence 段每 case 的 "证据" 列 SHALL 引用 S2/S3/S4 关键 commit / PR + 完整 6 case bitwise PASS 证据

### Requirement: `b1a_summary.md` 标题升级 "完成" + 时间线 + B1a-tag 处理

工程 SHALL 在 B1a capstone PR-12 中修改 `docs/b1a_summary.md`：(a) 标题 "B1a Baseline 进度（IN PROGRESS）" → "B1a Baseline 完成"；(b) 开头声明从 "当前进度：S0 + S1 完成；S2/S3/S4 全未做" 更新为 "S0–S4 全部完成于 <date>"；(c) "B1a-tag 的处理" 段更新为 "已 force-update 旧 `B1a-tag` 从 `64569b3` 到 `<new-commit>`"；(d) 时间线段追加 S2.1–S2.17 + S2 capstone + S3a/S3b/S3c + S4 + B1a capstone 各 PR 引用。

#### Scenario: b1a_summary 升级完成

- **WHEN** B1a capstone PR-12 merge
- **THEN** `docs/b1a_summary.md` 标题第一行 SHALL = `# B1a Baseline 完成`
- **THEN** 开头声明 SHALL 包含 "S0–S4 全部完成" 字样，不再有 "IN-PROGRESS" / "S2/S3/S4 全未做" / "过早签证"
- **THEN** "B1a-tag 处理" 段 SHALL 写明 "force-update 完成：旧 commit `64569b3` → 新 commit `<final-commit>`"
- **THEN** 时间线 SHALL 包含 S2.1 / S2.2 / ... / S2.17 / S2 capstone / S3a / S3b / S3c / S4 / B1a capstone 全部 PR 编号

### Requirement: PR-12 内顺带 archive + 收尾 4 动作（lock + push main + log + CLAUDE.md sync）

工程 SHALL 在 B1a capstone PR-12 内 + merge 后完成 5 个动作（archive + 收尾 4 动作）。Archive 在 PR-12 内顺带做（diff 包含 4 个 spec 文件移动），不在 PR-12 merge 后单独 PR（理由 design.md D11：lock 后 baseline/B1a 无法再写）；其它 4 个动作（lock branch + push main + log append + CLAUDE.md sync）按 Conventions PR-12 顺序执行。

#### Scenario: 收尾 5 动作全部完成（顺序符合 Conventions）

- **WHEN** B1a capstone PR-12 内 + merge 后执行收尾
- **THEN** **PR-12 内**：4 个 spec 文件 SHALL 已 PROMOTE 到 `openspec/specs/<capability>/spec.md`（committed 新增，4 个 capability：`s2-semantic-merge` / `s3-deterministic-gather` / `s4-adjacency-topology` / `b1a-capstone`）
- **THEN** **PR-12 内**：`openspec/changes/b1a-finalization/` 目录 SHALL 已 RELOCATE 到 `openspec/changes/archive/2026-06-20-b1a-finalization/`（local-only per `.gitignore` convention，不进 git diff，匹配前两次 archive precedent）
- **THEN** **merge 后第 4 步**：`git push origin baseline/B1a:main` 完成；`gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/B1a --jq '.commit.sha'` SHALL == `gh api repos/DankerMu/SHUD-OpenMP/branches/main --jq '.commit.sha'`（fast-forward 后 baseline/B1a == main HEAD）
- **THEN** **merge 后第 5 步**：tag force-update 完成（详见 B1a-tag Requirement Scenario）
- **THEN** **merge 后第 6 步**：`gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/B1a/protection --jq '.lock_branch.enabled'` SHALL = `true`；`allow_force_pushes=false` + `allow_deletions=false`
- **THEN** **merge 后第 7 步**：`docs/review-loop-log.jsonl` 最后一行 SHALL 包含 `"change":"b1a-finalization"` + `"verdict":"clean"` + `"capstone_commit":"<final-commit>"` + `"b1a_tag_force_updated_from":"64569b3"`
- **THEN** **merge 后第 8 步**：`CLAUDE.md` 双端 sync 完成（`diff CLAUDE.md` Mac vs 服务器 SHALL 一致）；CLAUDE.md "B1a 进度" 行 SHALL 由 "B1a IN-PROGRESS（S0+S1 PASS / S2-S4 PENDING）" 改为 "B1a 完成（S0-S4 PASS）"，并删除 "B1a-tag 是过早签证" disclaimer
- **THEN** `ls openspec/specs/s2-semantic-merge/spec.md openspec/specs/s3-deterministic-gather/spec.md openspec/specs/s4-adjacency-topology/spec.md openspec/specs/b1a-capstone/spec.md` SHALL 全部 exit 0
- **THEN** `openspec list` SHALL 不再列出 `b1a-finalization` change（已 archived）
