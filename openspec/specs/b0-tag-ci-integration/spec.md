# b0-tag-ci-integration Specification

## Purpose

S1 CI 闸落地。把 B0-tag bitwise neutrality 验证组装为每 PR 跑的 active gate：4 local case (keliya / xinanjiang_upstream / qinyijiang / qhh) × `*.dat` SHA256 + 24 snapshot golden + CVODE stats 15-key 全键 diff + 4 类宏移除 grep gate + invariant-sweep + 3-Config Makefile build + skip-baseline-ci label 反验，全 data-independent；fast-feedback (keliya) ≤ 2 分钟、full-bitwise 走 `full-bitwise` label / nightly cron。

## Scope

**Scope Note (S0 ci-serial-baseline 关系澄清)**: 本 capability 在 S0 `ci-serial-baseline` capability 的基础上扩展：(a) `build-and-compare` job case scope 从 keliya 单一扩到 keliya / xinanjiang_upstream / qinyijiang / qhh 4 case；(b) `skip-baseline-ci` label 退役（S0 阶段为通融，S1 后必删）；(c) `actions/checkout` 必须 `fetch-tags: true`；(d) `LEGACY_RHS=0 / 1` 两轴 build matrix。S0 `ci-serial-baseline` 的 Requirement 仍在底层 active；本 capability 不删任何 S0 Requirement，只 ADDED 严格更紧的 gate。proposal.md 中将 `ci-serial-baseline` 移至 'Modified Capabilities' subsection 反映此关系（提示信息：proposal.md 由 Implementer A 维护，本 spec 仅记录 scope 关系不直接 cross-edit）。
## Requirements
### Requirement: CI 4 local case bitwise check 覆盖范围

`.github/workflows/serial-baseline.yml` 的 `build-and-compare` job SHALL 把 B0-tag bitwise check 从 keliya 单 case 扩展到 4 个 local-and-server case：`keliya` / `xinanjiang_upstream` / `qinyijiang` / `qhh`。每个 case 跑完后 MUST 对 `benchmarks/<case>/B0_output/` 下每个 `*.dat` / snapshot 输出与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 做 SHA256 对比。`kashigeer` SHALL 排除（`endpoint = deferred-upstream`，A0 阶段已按 S0-13 spec 修订项排除）；`heihe` / `heihe_x4` SHALL 排除（`endpoint = server-only`，GitHub-hosted runner 无 forcing 数据与磁盘配额，仍由服务器 Slurm 手动归档承担）。

#### Scenario: CI 在 4 个 local case 上对 B0-tag 做 SHA256 对比

- **WHEN** workflow 在 `full-bitwise` 模式或 nightly schedule 触发
- **THEN** CI SHALL 依次对 `keliya` / `xinanjiang_upstream` / `qinyijiang` / `qhh` 4 个 case build + run + 对比
- **AND** 每个 case 当前 run 产出的 `B0_output/<file>` SHA256 MUST 与 `git show B0-tag:benchmarks/<case>/B0_output/<file>` SHA256 完全相等
- **AND** `kashigeer` / `heihe` / `heihe_x4` 不在 matrix 中，相关 case key MUST 不出现在 workflow job 输出里

#### Scenario: server-only case 在 CI 显式标注跳过原因

- **WHEN** 任何 PR 或 push 触发 workflow
- **THEN** workflow log MUST 含一条 `::notice` 说明 heihe / heihe_x4 因 endpoint=server-only 不在 CI 跑，并指向 `benchmarks/INDEX.md`
- **AND** 该 notice MUST NOT 标记 workflow 为 failure；server-only case 验收走服务器手动 Slurm 归档（不入 CI 范围）

#### Scenario: kashigeer 不被纳入 CI matrix

- **WHEN** workflow 触发任意模式（PR / push / nightly / full-bitwise label）
- **THEN** CI MUST NOT 试图 build / run / compare kashigeer
- **AND** workflow log MUST 含一条 `::notice` 说明 kashigeer endpoint=deferred-upstream，per S0-13 spec amendment 跳过

#### Scenario: 4 case 之外的额外 case 不会被偷偷加入

- **WHEN** 维护者新增 `benchmarks/<other-case>/manifest.yaml` 但未走 spec 修订
- **THEN** CI MUST NOT 自动把新 case 加入 bitwise matrix
- **AND** 新增 case 进入 CI MUST 走本 capability 的 spec 修订 + matrix 显式列名

### Requirement: PR fast-feedback 路径只跑 keliya

PR 触发 workflow 的默认（无 `full-bitwise` label）路径 SHALL 仅对 keliya build + run + B0-tag bitwise 对比，end-to-end wall-clock MUST ≤ 2 分钟（含 SUNDIALS warm cache）。其余 3 个 local case（xinanjiang_upstream / qinyijiang / qhh）只在 nightly schedule 或 PR 标 `full-bitwise` label 时跑，以避免 PR 反馈被全 4 case 串行跑膨胀到 5+ 分钟。Fast-feedback PASS MUST 是 PR 进入 review 的必要条件；full-bitwise PASS 是 merge 到 `baseline/current` 的必要条件之一。

#### Scenario: PR 默认触发只跑 keliya

- **WHEN** PR 打到 `baseline/current` 且 PR 未带 `full-bitwise` label
- **THEN** workflow `build-and-compare` job MUST 仅对 keliya 跑 build + run + B0-tag SHA256 对比
- **AND** end-to-end wall-clock SHALL ≤ 2 分钟（SUNDIALS warm cache 下）
- **AND** workflow log MUST 含一条 `::notice` 说明 fast-feedback 模式只跑 keliya，full-bitwise 需要带 label 或等 nightly

#### Scenario: PR 标 full-bitwise label 后跑全 4 case

- **WHEN** PR 加上 `full-bitwise` label 并 push 新 commit（或 re-run workflow）
- **THEN** workflow MUST 对 keliya / xinanjiang_upstream / qinyijiang / qhh 4 个 case 全部 build + run + 对比
- **AND** 任一 case bitwise 失败 MUST 让 job 整体 fail（PASS/FAIL 在 job summary 中按 case 聚合列出）
- **AND** wall-clock 不再受 2 分钟限制，但 job timeout SHALL ≥ 15 分钟以容纳 4 case 串行

### Requirement: B0-tag 引用方式与对比命令

CI MUST 使用 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 作为 golden 数据源。`B0-tag` 是 annotated tag，tag object SHA = `95ddc375`，指向 commit `884cfb13`（外层 `baseline/current` 上的 S0 wrap-up commit，SHUD pin = `78c37a1`）。CI MUST NOT 直接 checkout `benchmarks/<case>/B0_output/` 工作区文件作 golden（避免工作区被 PR 误改污染对比），MUST 显式走 `git show B0-tag:...` 拿历史快照。对比方式 MUST 是 SHA256 完全相等；任何 SHA256 不等即 FAIL。

#### Scenario: CI 通过 git show 拉取 B0-tag 数据

- **WHEN** workflow 进入 compare 步骤
- **THEN** 每个 `<case>/<file>` 对比 MUST 先执行 `git show B0-tag:benchmarks/<case>/B0_output/<file> | sha256sum` 取 golden SHA256
- **AND** 再对当前 run 产出 `benchmarks/<case>/B0_output/<file>`（或对应 case run 目录下生成的等价文件）`sha256sum` 取 new SHA256
- **AND** 两者 byte-for-byte 相等才算 PASS，否则 FAIL

#### Scenario: B0-tag 不可达时 CI 显式 fail 而非 silent skip

- **WHEN** workflow 在 checkout 后 `git rev-parse B0-tag^{commit}` 失败（tag 被删 / 仓库 shallow 漏 tag）
- **THEN** workflow MUST 以 `::error` 退出（非 silent skip），错误信息含 `B0-tag missing or unreachable; cannot perform bitwise check`
- **AND** 此种 fail 算 infrastructure failure，但不允许通过加 `skip-baseline-ci` 旁路（label 已删，见后述）

#### Scenario: checkout 必须 fetch tags 与完整历史避免 shallow 漏 tag

- **WHEN** workflow 调用 `actions/checkout@v4` step
- **THEN** step config MUST 满足以下二者之一：① `fetch-tags: true` 显式打开；② `fetch-depth: 0` 拉全历史
- **AND** workflow 在 build / run 之前 MUST 先执行 `git rev-parse B0-tag^{commit}` 显式验证 tag 存在；若失败 MUST 立即 `::error` 退出（错误正文含 `B0-tag not reachable: ensure actions/checkout fetch-tags: true or fetch-depth: 0`）
- **AND** MUST NOT 走默认 shallow checkout（漏 tag 时直接进入 build/run 浪费 CI 资源后才 fail）

#### Scenario: 工作区 B0_output 被 PR 修改不影响对比

- **WHEN** PR 内含对 `benchmarks/<case>/B0_output/<file>` 工作区文件的修改
- **THEN** CI compare 步骤仍以 `git show B0-tag:...` 为 golden（不读工作区文件作 golden）
- **AND** 工作区修改本身不会让 SHA256 对比"假性 PASS"
- **AND** workflow log MUST 含一条 `::warning` 提示 B0_output 被 PR 改动（业务上 B0_output 应 immutable）

### Requirement: 90-day truncation 在 CI 自动应用

CI setup 阶段 SHALL 在每个 case build / run 前自动把 `SHUD/Basins/<case>/input/<case>/<case>.cfg.para` 的 `END` 字段重写为 `START + 90`（day-index 制；90 天 = +90）。该改写 MUST 由 `tools/fix_case_paths/fix_case_paths.sh <case>` 在 workflow setup 步骤调用，MUST NOT 由 spec 层 manifest 改动承担——`benchmarks/<case>/manifest.yaml` 的 `forcing_duration_days` 字段保留 case 的 full forcing 长度（spec 层 vs 部署层分离，per CLAUDE.md 项目纪律）。90 天截断在 CI 内是 deployment-layer transform，不入 SHUD submodule，不入 PR diff。

#### Scenario: fix_case_paths 在 run 前改写 cfg.para

- **WHEN** workflow 跑某 case 前
- **THEN** workflow MUST 调用 `tools/fix_case_paths/fix_case_paths.sh <case>`
- **AND** 该脚本 MUST 把 `SHUD/Basins/<case>/input/<case>/<case>.cfg.para` 的 `END` 行改为 `START + 90`
- **AND** 同步把 `tsd.forc` 的 forcing 路径改为 CI runner 绝对路径、`NUM_OPENMP` 置 1（serial mode）

#### Scenario: manifest forcing_duration_days 不被改写

- **WHEN** workflow 改写 cfg.para
- **THEN** `benchmarks/<case>/manifest.yaml` 的 `forcing_duration_days` 字段 MUST 保留原 spec 层数值（不被改为 90）
- **AND** CI MUST NOT 试图把 deployment-layer truncation 推回 spec 层

#### Scenario: 90 天截断窗口外的 snapshot t_value HARD fail（不允许 silent skip / partial-skip）

- **WHEN** workflow 跑 dump-on binary，`manifest.yaml` 声明 N 个 snapshot probes（统一为 `t_values: [86400, 2592000, 7776000]`，即 1d / 30d / 90d，均 ≤ 90d 截断），CI MUST 在每个 case 上行使全部 N 个 t_value 的对比
- **THEN** 若 `N > 0` 且存在任一 t_value > END（END = START + 90d），workflow MUST 以 `::error title=snapshot t_value out of 90d window` 退出非零（HARD fail），错误正文含 `<case> manifest snapshot t_value <T> exceeds 90d truncation END; either re-archive within window or amend manifest`
- **AND** workflow MUST NOT 通过 `::notice` / `partial-skip` / silent pass 任意一种方式让 PR check 仍然显示绿
- **AND** 全部 N 个 t_value 的 SHA256 对比 MUST 在 CI 内逐一执行；任一 mismatch 同样 FAIL
- **AND** 该 HARD fail 强制 manifest spec 与 90d 截断窗口在 review 时即对齐，无逃逸路径

### Requirement: SHA256 mismatch 的错误处理

任一 (case, file) 对的 SHA256 mismatch SHALL 让 `build-and-compare` job 退出非零状态码（非 silent 续跑）。错误消息 MUST 是结构化、可机器解析的，至少含以下字段：发散文件的完整路径（`benchmarks/<case>/B0_output/<file>`）、`B0-tag` golden SHA256（前缀至少 12 字符）、当前 run SHA256（前缀至少 12 字符）、建议的 local 复现命令（`tools/compare_snapshot/compare_snapshot <golden.bin> <new.bin>` 用于 snapshot；`diff <(xxd golden.dat) <(xxd new.dat) | head` 用于 dat）。Mismatch artifact MUST 上传到 `baseline-failure-diff` artifact（沿用 S0-9 既有命名）。

#### Scenario: 单 case 单文件 mismatch 时错误消息含可复现命令

- **WHEN** qinyijiang 的 `snapshot_t2592000.bin` SHA256 与 `B0-tag` 不等
- **THEN** workflow MUST 以 `::error title=B0-tag bitwise FAIL` 退出非零
- **AND** 错误正文 MUST 含字面字符串 `FAIL: qinyijiang/B0_output/snapshot_t2592000.bin SHA256 mismatch`
- **AND** 含 `Expected <golden-sha-12+>` 与 `Got <new-sha-12+>` 两行
- **AND** 含建议复现命令 `tools/compare_snapshot/compare_snapshot benchmarks/qinyijiang/B0_output/snapshot_t2592000.bin <new-snapshot-path>`

#### Scenario: 多 case 多文件 mismatch 时聚合报告

- **WHEN** full-bitwise 模式下 keliya 的 `keliya.rivqdown.dat` 与 qhh 的 `qhh.rivqdown.dat` 同时 mismatch
- **THEN** workflow MUST 把两条 `::error` 都打印（不在第一条 mismatch 处早退）
- **AND** job summary MUST 含一个聚合表，按 (case, file, expected-sha, actual-sha) 行列出全部 mismatch
- **AND** job exit code MUST 非零

#### Scenario: PASS 时不输出 false-positive 错误

- **WHEN** 4 个 case 所有文件 SHA256 都与 `B0-tag` 一致
- **THEN** workflow MUST 以成功状态退出
- **AND** workflow log MUST 含一条 `::notice title=B0-tag bitwise PASS` 列出 case 与对比文件数
- **AND** MUST NOT 产生 `baseline-failure-diff` artifact（artifact step 受 `if: failure()` 守护）

### Requirement: status-matrix proposer 保留 + 不自动 push

S0-12 引入的 status-matrix proposer 步骤 SHALL 在 S1 CI 中保留：fast-feedback 与 full-bitwise PASS 时，对 PR 留 comment 提议 `docs/status_matrix.md` 的对应 cell diff（B1a 行对应 case 标 PASS 的 suggested change）。CI MUST NOT 直接 push 到 `baseline/current`（per `status-matrix` spec L19 避免并发写竞争）。Proposer comment 必须含稳定 HTML 注释标记 `<!-- s0-12-matrix-proposer -->` 以保证幂等（同 PR 多次 push 不重复评论）。

#### Scenario: full-bitwise PASS 后 proposer 留 comment

- **WHEN** PR 标 `full-bitwise` label 且 4 case 全部 PASS
- **THEN** workflow MUST 在 PR 上留一条 comment 含 4 case 行的 status_matrix.md suggested diff
- **AND** comment 起始 MUST 含 `<!-- s0-12-matrix-proposer -->` 注释锚点
- **AND** workflow MUST NOT push 任何 commit 到 `baseline/current` 或工作区分支

#### Scenario: 同 PR 多次 PASS 不重复 comment

- **WHEN** 同一 PR 第二次 PASS（重 push commit 后）
- **THEN** workflow MUST 用 `gh pr view` 检测已有 `<!-- s0-12-matrix-proposer -->` comment
- **AND** 若存在则 `::notice` 跳过、MUST NOT 再次留同样 comment
- **AND** 若 PR 内 commit hash 已变，proposer 可选更新 commit ref（但保留同一锚点）

#### Scenario: PR 跨 case mismatch 时 proposer 不留 PASS comment

- **WHEN** full-bitwise 模式下任一 case 任一文件 mismatch
- **THEN** proposer 步骤受 `if: success()` 守护 MUST NOT 执行
- **AND** workflow MUST NOT 留任何"建议把 B1a 行标 PASS"的 comment

### Requirement: skip-baseline-ci label 已删 + 必需 check

`skip-baseline-ci` label 在 S0-13 (PR #35) 已从仓库删除，S1 CI workflow MUST NOT 允许任何 label / 任何方式旁路 `build-and-compare` job 的 bitwise check。`baseline/current` 分支保护规则 MUST 把 `serial-baseline / build-and-compare` 设为 required status check：fast-feedback 失败的 PR MUST NOT 能 merge；full-bitwise 失败的 PR（带 label 时）同样 MUST NOT 能 merge。

#### Scenario: 仓库内无 skip-baseline-ci label

- **WHEN** 维护者执行 `gh label list --repo DankerMu/SHUD-OpenMP --json name`
- **THEN** 返回的 label 列表 MUST NOT 含 `skip-baseline-ci`
- **AND** workflow YAML 内 `contains(github.event.pull_request.labels.*.name, 'skip-baseline-ci')` 即使存在也 trivially false（dead-code 性质留作历史 transition 记录，符合 S0 `ci-serial-baseline` spec `Skip label respected during S0 development` Requirement 描述）

#### Scenario: skip-baseline-ci label 在仓库层显式删除验证

- **WHEN** 在 CI 验证 step 中执行 `gh label list --repo DankerMu/SHUD-OpenMP | grep -c skip-baseline-ci`
- **THEN** 命令 SHALL 输出 `0`（label 不存在），exit code 不重要但 stdout 必须是 `0`
- **AND** 若维护者误重建 `skip-baseline-ci` label，CI MUST 在该 step 以 `::error` fail，错误消息含 `skip-baseline-ci label MUST NOT exist; deleted in PR #35 (S0-13)`

#### Scenario: PR 在 build-and-compare 失败时被 branch protection 拦截

- **WHEN** PR 打到 `baseline/current` 且 `build-and-compare` 退出非零
- **THEN** GitHub PR UI MUST 显示 `serial-baseline / build-and-compare` 为 required check failing
- **AND** PR merge 按钮 MUST 被禁用（admin override 仅在显式 spec-amended bypass 下允许，本 spec 不允许）
- **AND** 重试 / 修复后再次 PASS 方可 merge

#### Scenario: 任意 label / commit message keyword 都不能旁路 check

- **WHEN** PR 携带任意 label（含历史遗留命名）或 commit message 含 `[skip ci]` / `[ci skip]` 等关键字
- **THEN** workflow MUST 仍然执行 `build-and-compare` job 全流程
- **AND** branch protection MUST 仍然把该 job 设为 required，不允许 bypass

### Requirement: workflow trigger 事件覆盖

`serial-baseline.yml` workflow `on:` 块 SHALL 覆盖以下触发事件：① `push` 到任一分支（含 `baseline/current` 与 feature branch）；② `pull_request` 打到 `main` 或 `baseline/current`；③ `schedule` cron 每日 03:00 UTC 跑一次 nightly full-bitwise（4 case 全跑）。Nightly 跑 MUST 在 `baseline/current` 最新 commit 上执行，PASS / FAIL 通过 GitHub Actions notification + repo issue 通知。Nightly 模式 implicitly 等价于带 `full-bitwise` label 的 PR 跑（4 case 全跑、不受 fast-feedback 2 分钟限制）。

#### Scenario: push 到 baseline/current 触发 fast-feedback

- **WHEN** maintainer push commit 到 `baseline/current`
- **THEN** workflow MUST 触发 `build-and-compare` job
- **AND** 默认走 fast-feedback 模式（仅 keliya）
- **AND** 若 commit message 显式含 `[full-bitwise]` 关键字（可选 future 扩展）才升级到 4 case，否则 keliya only

#### Scenario: PR 打到 baseline/current 触发 fast-feedback

- **WHEN** PR opened / reopened / synchronize 事件 + base = `baseline/current`
- **THEN** workflow MUST 触发，默认 fast-feedback 模式
- **AND** PR 带 `full-bitwise` label 时升级到 4 case 模式

#### Scenario: PR label 增删事件触发重跑

- **WHEN** `pull_request` 的 types 设置 SHALL 包括 `opened, reopened, synchronize, labeled, unlabeled`
- **THEN** 向 PR 加 `full-bitwise` label（`labeled` 事件）MUST 立刻重跑 workflow 升级到 4 case 模式
- **AND** 从 PR 移除 `full-bitwise` label（`unlabeled` 事件）MUST 立刻重跑 workflow 降级回 fast-feedback 模式
- **AND** workflow YAML 内 `on.pull_request.types` MUST 显式列出五种 type；缺失任一会导致 label 切换不重跑（仅 push 新 commit 才生效），违背"标签即开关"的使用语义

#### Scenario: nightly cron 03:00 UTC 跑全 4 case

- **WHEN** UTC 时间 03:00 到达
- **THEN** workflow MUST 在 `baseline/current` 最新 commit 上自动触发
- **AND** 走 4 case full-bitwise 模式
- **AND** 任一 mismatch MUST 产生 GitHub Actions failure notification，并由 maintainer 决定是否开 issue 跟踪

#### Scenario: docs-only PR 仍然跑 CI 不做 path filter

- **WHEN** PR 仅修改 `docs/` 下文件
- **THEN** workflow MUST 仍触发 `build-and-compare` job
- **AND** workflow MUST 走 fast-feedback 模式跑完（build cache 命中下 wall-clock 仍 ≤ 2 分钟）
- **AND** MUST NOT 用 `paths:` / `paths-ignore:` filter 跳过——CI 保守起见对所有 PR 都执行
- **AND**（Trade-off note）docs-only PR 也走 full `build-and-compare` 是保守选择（避免后续 docs PR 误启 paths-ignore exception 滑到代码改动）；S1 close 后视实际 PR 量决定是否引入 `paths-ignore` + label-based re-trigger（follow-up issue 跟踪）

### Requirement: snapshot t_values 在 CI 全 case 统一为 [86400, 2592000, 7776000]

所有进入 CI bitwise matrix 的 case（keliya / xinanjiang_upstream / qinyijiang / qhh）`benchmarks/<case>/manifest.yaml` 的 `t_values` 字段 SHALL 统一为 `[86400, 2592000, 7776000]`（1d / 30d / 90d，秒为单位），全部 ≤ 90 天截断窗口。全部 24 张 golden（4 case × 3 t_values × 2 probe-point：after-PassValue 落 `snapshot_t<sec>.bin` + before-PassValue 落 `snapshot_t<sec>_before_passvalue.bin`，per Implementer A tasks 0.1a/0.1b split + flux spec L163 24-golden alignment）SHALL 在 S1 pre-S1a task 0.1a/0.1b（re-archival 任务）中按统一 t_values 重新归档于 `benchmarks/<case>/B0_output/`。CI MUST 逐 case 逐 t_value 逐 probe-point 对全部 6 张 snapshot（3 t_value × 2 probe-point）做 `tools/compare_snapshot/compare_snapshot` 二进制对比，exit code 0 才算 PASS。

#### Scenario: 全 4 case manifest.yaml 的 t_values 字段统一

- **WHEN** 在 CI setup step 执行 `for c in keliya xinanjiang_upstream qinyijiang qhh; do yq -r '.snapshots.t_values' benchmarks/$c/manifest.yaml; done`
- **THEN** 4 case 全部 SHALL 输出 `[86400, 2592000, 7776000]`（或语义等价的 YAML 列表表示）
- **AND** 任一 case 的 t_values 与统一值不符 MUST 让 CI `::error` 退出

#### Scenario: 24 张 snapshot golden 重新归档（12 张 after-PassValue + 12 张 before-PassValue，pre-S1a task 0.1a/0.1b 前置）

- **WHEN** CI 执行 `for c in keliya xinanjiang_upstream qinyijiang qhh; do for t in 86400 2592000 7776000; do for suffix in '' '_before_passvalue'; do git show B0-tag:benchmarks/$c/B0_output/snapshot_t${t}${suffix}.bin > /dev/null; done; done; done`
- **THEN** 24 张文件 SHALL 在 B0-tag 内全部可访问（12 张 after-PassValue `snapshot_t<sec>.bin` 由 task 0.1a 归档 + 12 张 before-PassValue `snapshot_t<sec>_before_passvalue.bin` 由 task 0.1b 归档，S1b flux 阶段 CI 必须比对两者以验证 ApplyDY 前后 RHS 中间态）
- **AND** 若任一缺失 MUST 让 CI `::error` 退出，错误正文含 `snapshot golden missing in B0-tag; re-archival task 0.1a/0.1b not yet landed`

#### Scenario: CI 对每 case 全 6 张 snapshot（3 t_value × 2 probe-point）跑 compare_snapshot

- **WHEN** workflow 在 full-bitwise 模式下跑某 case
- **THEN** workflow MUST 对该 case 全部 3 个 t_value 调用 `tools/compare_snapshot/compare_snapshot <golden.bin> <new.bin>`；每个 t_value 实际产出 2 张 golden（after-PassValue `snapshot_t<v>.bin` + before-PassValue `snapshot_t<v>_before_passvalue.bin`），S1b 阶段 CI 必须比对两者；每次 exit code = 0 才算该 snapshot PASS
- **AND** 单 case 共 6 次 compare_snapshot 调用（3 t_value × 2 probe-point），全部 exit 0 才算 snapshot gate PASS
- **AND** 任一 snapshot 对比 fail MUST 让该 case 整体 FAIL（与 `*.dat` SHA256 fail 同等待遇）

### Requirement: CVODE stats 15-key archive bitwise check 作 CI 强 gate

CI MUST 在每个 case bitwise compare step 中显式对 `benchmarks/<case>/B0_output/cvode_stats.txt` 进行 canonical 15-key 全键 diff，调用共享工具 `tools/cvode_stats_diff/cvode_stats_diff.sh`（task 0.2 创建）。Canonical 15-key set 定义详 `openspec/glossary.md` §CVODE canonical 15-key set（F19 修订：归档不含 `nFCall`，由独立 capability 跟踪）。任一键缺失 / 数值不等 MUST 让 case FAIL；该 gate 与 `*.dat` SHA256 / snapshot 对比并列，三 gate 任一 fail 则 case FAIL。

#### Scenario: CI cvode_stats_diff 15-key 全键 PASS

- **WHEN** workflow 跑完某 case，进入 compare step
- **THEN** workflow MUST 执行 `tools/cvode_stats_diff/cvode_stats_diff.sh <run.txt> <(git show B0-tag:benchmarks/<case>/B0_output/cvode_stats.txt)`，exit code = 0 才算 CVODE stats PASS
- **AND** 工具 stdout MUST 列出 canonical 15-key 全键 matched 的 summary 行（详 `openspec/glossary.md` §CVODE canonical 15-key set）
- **AND** 该 PASS 与 `*.dat` SHA256 PASS / snapshot PASS 共同构成 case 总 PASS 条件

#### Scenario: CVODE stats 任一键 mismatch 时 CI 错误消息格式

- **WHEN** 当前 run 的 `nfeLS` 值与 B0-tag 归档值不等
- **THEN** `cvode_stats_diff.sh` SHALL exit 非零，stdout / stderr 内 MUST 含 `MISMATCH key=nfeLS expected=<gold> got=<new>`
- **AND** workflow 捕获该错误并 `::error title=CVODE stats mismatch` 退出非零
- **AND** 错误正文 MUST 列出 canonical 15-key 全键中全部 mismatched 键的 (key, expected, got) 三元组（不在首个 mismatch 处早退）

#### Scenario: B0_output/cvode_stats.txt 缺失时 CI 显式 fail

- **WHEN** 当前 run 的 archive 步骤未生成 `cvode_stats.txt`（dump 路径失败 / binary stats helper 未编入）
- **THEN** `cvode_stats_diff.sh` SHALL exit 非零，错误正文含 `<path> not found; run did not emit cvode_stats.txt`
- **AND** workflow 以 `::error title=CVODE stats archive missing` 退出非零
- **AND** MUST NOT silently 跳过 CVODE 对比（缺失即 FAIL，与 mismatch 同等待遇）

### Requirement: CI build matrix 跑 LEGACY_RHS=0 与 =1 两轴

CI workflow SHALL 在 build matrix 中显式包含两个 compile 轴 `LEGACY_RHS=0`（默认 = B1a 路径）与 `LEGACY_RHS=1`（legacy 路径）。PR fast-feedback 模式下两轴 MUST 都对 keliya 跑 bitwise（合计 wall-clock ≤ 4 分钟）；nightly cron 模式下两轴 MUST 都对 4 个 local case 跑 bitwise（合计 wall-clock 不受 4 分钟限制，但 job timeout ≥ 25 分钟）。两轴中任一 fail 则 PR 整体 fail，merge 被 branch protection 拦截。

#### Scenario: PR fast-feedback 两轴跑 keliya bitwise

- **WHEN** PR opened/reopened/synchronize 触发 fast-feedback
- **THEN** workflow MUST 在 build matrix 中产出 `{ rhs_path: LEGACY_RHS=0 }` 与 `{ rhs_path: LEGACY_RHS=1 }` 两个 job
- **AND** 两 job 分别用对应 make 标志编译并对 keliya 跑 90 天截断
- **AND** 两 job 分别独立做 `*.dat` SHA256 + snapshot + CVODE 15-key 对比 vs B0-tag
- **AND** 两 job 合计 end-to-end wall-clock SHALL ≤ 4 分钟（SUNDIALS warm cache 下）

#### Scenario: nightly cron 两轴 × 4 case bitwise

- **WHEN** UTC 03:00 nightly 触发
- **THEN** workflow MUST 在 build matrix 中产出 2 (rhs_path) × 4 (case) = 8 个 job
- **AND** 每个 job 独立跑 build + run + 三 gate 对比（SHA256 / snapshot / CVODE 15-key）
- **AND** job timeout SHALL ≥ 25 分钟，覆盖 4 case 串行 + 两轴 parallel 的整体 wall-clock

#### Scenario: 任一轴 fail 则 PR / nightly 整体 fail

- **WHEN** `LEGACY_RHS=0` 轴 PASS 但 `LEGACY_RHS=1` 轴 mismatch
- **THEN** workflow 整体 status MUST 为 failure
- **AND** branch protection MUST 把两个矩阵 job (`LEGACY_RHS=0`, `LEGACY_RHS=1`) 同时设为 required status check
- **AND** PR merge 按钮 MUST 被禁用直到两轴皆 PASS

### Requirement: 宏移除 grep gate 作 CI 强检查

CI workflow SHALL 在每次 build-compare run 中执行 macro removal grep gate step，按当前 SHUD submodule pin 所对应的 S1 substage 验证宏移除已就位：

- **Post-S1d.1（USE_RHS_CORE 退役）**：`grep -r 'USE_RHS_CORE' SHUD/src/` MUST 返回 ZERO 匹配
- **Post-S1d.2（_OPENMP_ON / NV_DATA_OMP / NV_DATA_S / N_VDestroy_Serial 退役）**：
  - `grep -r '_OPENMP_ON' SHUD/src/` MUST 返回 ZERO
  - `grep -r 'NV_DATA_OMP\|NV_DATA_S' SHUD/src/Model/f.cpp` MUST 返回 ZERO
  - `grep -r 'N_VDestroy_Serial' SHUD/src/` MUST 返回 ZERO

每个 grep 一行 CI step（或 shell 子命令），任一非零匹配 MUST 让 step `::error` 退出。该 step MUST 与 build / bitwise compare 并列作为 required check。

#### Scenario: Post-S1d.1 USE_RHS_CORE grep gate

- **WHEN** PR 已 merge S1d.1 之后（submodule pin advances 到含 `USE_RHS_CORE` 退役的 commit），workflow 触发
- **THEN** workflow MUST 执行 `grep -r 'USE_RHS_CORE' SHUD/src/`
- **AND** 命令 SHALL 返回 ZERO 匹配（exit code 1，stdout 空），CI step PASS
- **AND** 若意外存在残留命中，CI MUST 以 `::error title=USE_RHS_CORE not fully retired` 退出非零

#### Scenario: Post-S1d.2 _OPENMP_ON grep gate

- **WHEN** PR 已 merge S1d.2 之后，workflow 触发
- **THEN** workflow MUST 执行 `grep -r '_OPENMP_ON' SHUD/src/`
- **AND** 命令 SHALL 返回 ZERO 匹配，CI step PASS
- **AND** 若残留命中，CI MUST 以 `::error title=_OPENMP_ON not fully retired` 退出非零

#### Scenario: Post-S1d.2 NV_DATA_OMP / NV_DATA_S grep gate

- **WHEN** PR 已 merge S1d.2 之后，workflow 触发
- **THEN** workflow MUST 执行 `grep -rE 'NV_DATA_OMP|NV_DATA_S' SHUD/src/Model/f.cpp`
- **AND** 命令 SHALL 返回 ZERO 匹配（f.cpp 6 处 N_Vector 解包已统一为 `N_VGetArrayPointer`）
- **AND** 残留命中 MUST 让 CI `::error title=NV_DATA_* not fully retired in f.cpp` 退出非零

#### Scenario: Post-S1d.2 N_VDestroy_Serial grep gate

- **WHEN** PR 已 merge S1d.2 之后，workflow 触发
- **THEN** workflow MUST 执行 `grep -r 'N_VDestroy_Serial' SHUD/src/`
- **AND** 命令 SHALL 返回 ZERO 匹配（已统一为 generic `N_VDestroy`）
- **AND** 残留命中 MUST 让 CI `::error title=N_VDestroy_Serial not fully retired` 退出非零

#### Scenario: S1d.1 / S1d.2 未 merge 时 grep gate 不强制

- **WHEN** PR 尚未 merge 至 S1d.1（submodule pin 仍指向 S1c 完成态）
- **THEN** `USE_RHS_CORE` grep 仍会返回非零（脚手架尚在），但 CI MUST 根据 substage tag / branch context 知道当前阶段尚未触达 grep gate 准入条件，SHALL `::notice` 提示 substage 未达而非 `::error`
- **AND** 若 PR 描述显式声明 "lands S1d.1"，则 grep gate 升级为 `::error`

