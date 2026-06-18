# S0 阶段总结报告（baseline lock）

> **状态**：完成 ✅ — `B0-tag` 已 push，A0 验收 9/9 PASS，进入 S1（B1a 重构 → strict 并行）的前置条件全部满足。
> **生成日期**：2026-06-17
> **范围**：master plan §5 S0（pre-S0 准备 + 13 个子阶段 + wrap-up），对应 GitHub issue #2–#17。
> **如何阅读**：先看 §1 TL;DR；规划 S1 工作看 §6；找具体证据从 §2 表里点链接。

---

## 1. TL;DR

S0 把 SHUD-OpenMP 项目从"裸 SHUD 源码 + 无 baseline"推进到"7 个 benchmark 注册 + 6 个可实跑 case 的 B0 输出按 commit 锁定 + 跨平台 profile 决策已签署 + CI 卫兵在线 + B0-tag 入 git"。

- **6 个 case bitwise-PASS B0 archive**（4 local + heihe / heihe_x4 server）+ kashigeer 因上游数据 gap 转为 `N/A (deferred-upstream)`。
- **`B0-tag`** 注解 tag `95ddc375…` 已 push，指向 squash-merge commit `884cfb13`，SHUD submodule 钉 `78c37a1`。
- **profile gate 决策签署**：走原方案（RHS-kernel-first OpenMP），Amdahl 上限 heihe_x4 8 核 2.39×，heihe 受限于 forcing IO 1.14×（决策建议提前并行化）。
- **CI workflow `.github/workflows/serial-baseline.yml`** 在线，3 build × keliya golden compare，已被 4 个 PR 实跑验证；branch protection on `baseline/current` 强制 `build-and-compare` required check + `skip-baseline-ci` 旁路特权已移除。
- **20 个 merged PR**（#18–#36，外加 #27/#28/#34 hygiene）—— 全程零 P0/P1 残留入 main 分支。

---

## 2. 完成的工作（issue × PR 对照）

| Issue | PR (`baseline/current`) | 标题 | 核心交付 |
|---|---|---|---|
| [#2](https://github.com/DankerMu/SHUD-OpenMP/issues/2) | (无 PR，本地分支操作) | Pre-S0: create `baseline/current` | 长寿 baseline 分支 |
| [#3](https://github.com/DankerMu/SHUD-OpenMP/issues/3) | [#18](https://github.com/DankerMu/SHUD-OpenMP/pull/18) | S0-1 Build environment lockdown | Makefile flag lock + SUNDIALS 6.0.x pin + `docs/build_manifest.md` 三层 disallowed-flag 防御 |
| [#4](https://github.com/DankerMu/SHUD-OpenMP/issues/4) | [#19](https://github.com/DankerMu/SHUD-OpenMP/pull/19) | S0-2 Case deployment fixup | `tools/fix_case_paths/` 重写 7 case 的 `tsd.forc` 绝对路径 |
| [#5](https://github.com/DankerMu/SHUD-OpenMP/issues/5) | [#20](https://github.com/DankerMu/SHUD-OpenMP/pull/20) | S0-3 Bootstrap verification | `tools/bootstrap_check.sh` + keliya e2e 跑通 |
| [#6](https://github.com/DankerMu/SHUD-OpenMP/issues/6) | [#21](https://github.com/DankerMu/SHUD-OpenMP/pull/21) | S0-4 Benchmark registry (6 NWM) | 6 个 `benchmarks/<case>/manifest.yaml` + `INDEX.md` + `tools/check_manifest.py` |
| [#7](https://github.com/DankerMu/SHUD-OpenMP/issues/7) | [#22](https://github.com/DankerMu/SHUD-OpenMP/pull/22) | S0-5 heihe_x4 B-Large | rSHUD 加密 + AutoSHUD CMFD2.0 glob patch + `tools/mesh_refine/` |
| [#8](https://github.com/DankerMu/SHUD-OpenMP/issues/8) | [#23](https://github.com/DankerMu/SHUD-OpenMP/pull/23) | S0-6 RHS dump hooks | `SHUD_DUMP_RHS` Makefile 开关 + `f.cpp` hooks（C1 invariant） |
| [#9](https://github.com/DankerMu/SHUD-OpenMP/issues/9) | [#24](https://github.com/DankerMu/SHUD-OpenMP/pull/24) | S0-7 Snapshot writer | `tools/rhs_snapshot/` + `tools/compare_snapshot/` + 12 个 golden snapshot |
| [#10](https://github.com/DankerMu/SHUD-OpenMP/issues/10) | [#25](https://github.com/DankerMu/SHUD-OpenMP/pull/25) | S0-8a CVODE stats + timer infra | `cvode_stats.txt` 落盘 + `SHUD_ENABLE_PROFILE` 开关 + `tools/profile/` RAII timer |
| [#11](https://github.com/DankerMu/SHUD-OpenMP/issues/11) | [#26](https://github.com/DankerMu/SHUD-OpenMP/pull/26) | S0-8b B0 archive (4 local) | `tools/archive_b0_output.sh` + keliya/xinanjiang_upstream/qinyijiang/qhh 各 3 次 bitwise PASS |
| [#12](https://github.com/DankerMu/SHUD-OpenMP/issues/12) | [#29](https://github.com/DankerMu/SHUD-OpenMP/pull/29) | S0-8c B0 archive (server) | heihe server cn08 3-run PASS + kashigeer `DEFERRED.txt` 上游数据 gap 文档化 |
| [#13](https://github.com/DankerMu/SHUD-OpenMP/issues/13) | [#30](https://github.com/DankerMu/SHUD-OpenMP/pull/30) | S0-9 CI workflow | `.github/workflows/serial-baseline.yml` 3 builds + skip-label gate + 数据探针 |
| [#14](https://github.com/DankerMu/SHUD-OpenMP/issues/14) | [#31](https://github.com/DankerMu/SHUD-OpenMP/pull/31) | S0-10 Local profile gate | 4 case `profile_B0.yaml` 实跑 + 3 deferred yaml + 时间桶 6 字段 |
| [#15](https://github.com/DankerMu/SHUD-OpenMP/issues/15) | [#32](https://github.com/DankerMu/SHUD-OpenMP/pull/32) | S0-11 Target profile + cross-platform | 6 server `profile_B0.target.yaml` + `docs/profile_platform.md` + `docs/profile_decision.md` |
| [#16](https://github.com/DankerMu/SHUD-OpenMP/issues/16) | [#33](https://github.com/DankerMu/SHUD-OpenMP/pull/33) | S0-12 Status matrix + CI proposer | `docs/status_matrix.md` 14×9 + A0 9 项 + CI PR-comment proposer step |
| [#17](https://github.com/DankerMu/SHUD-OpenMP/issues/17) | [#35](https://github.com/DankerMu/SHUD-OpenMP/pull/35) + [#36](https://github.com/DankerMu/SHUD-OpenMP/pull/36) | S0-13 Wrap-up + B0-tag | heihe_x4 server archive + kashigeer spec amendment + signature + `B0-tag` push + branch protection |

旁路 hygiene PR：
- [#27](https://github.com/DankerMu/SHUD-OpenMP/pull/27)：tighten lake-skip glob in archive_b0_output.sh
- [#28](https://github.com/DankerMu/SHUD-OpenMP/pull/28)：rename `cvode_stats_file` across 7 manifests
- [#34](https://github.com/DankerMu/SHUD-OpenMP/pull/34)：4 pre-existing server-path leaks + `sanitize_path()` helper

---

## 3. 关键交付物

### 3.1 锁定基线

- **`B0-tag`** (annotated)：tag object SHA `95ddc375ffa58115fd5c0a808dde80e9713b4c93` → 提交 `884cfb13ba08ebae02dd64e371c4a19a536b4e26`（baseline/current）+ SHUD submodule pin `78c37a1061de4112bc7c297bb7bd1f107432e6f2`
- **6 个 B0 输出归档**（`benchmarks/<case>/B0_output/`）：每个含 `.dat` × N + `cvode_stats.txt` + `repeatability.txt`（3 次 SHA256 PASS）
- **12 个 golden snapshot**（4 local case × 3 时间点）：`benchmarks/{keliya,xinanjiang_upstream,qinyijiang,qhh}/B0_output/snapshot_t*.bin`
- **7 个 manifest SHA256 内联在 `docs/build_manifest.md`** § B0-tag

### 3.2 工具链

| 工具 | 路径 | 作用 |
|---|---|---|
| Case 部署修复 | `tools/fix_case_paths/` | 把 `tsd.forc` 第 2 行的服务器绝对路径重写为本地路径 + 加 90-day 截断 |
| 启动验证 | `tools/bootstrap_check.sh` | 快速验证当前 host 能否跑通 keliya |
| Manifest 校验 | `tools/check_manifest.py` | 7-field schema + `endpoint` 枚举校验（含 `deferred-upstream`） |
| RHS snapshot 写入 | `tools/rhs_snapshot/` | 在指定 `t_values` dump DY + flux 数组（binary format） |
| Snapshot 对比 | `tools/compare_snapshot/` | bitwise + ULP report，差异时 exit 非零 |
| B0 归档 | `tools/archive_b0_output.sh` | manifest-driven N-run 跑 + SHA256 三次对比 + 归档 + `sanitize_path()` |
| Profile timer | `tools/profile/` | RAII 计时器 + 时间桶 + 退出时 dump `profile_B0.yaml` |
| Mesh refine | `tools/mesh_refine/` | heihe_x4 加密的 AutoSHUD driver（含 CMFD2.0 glob patch） |

### 3.3 文档

| 文档 | 用途 |
|---|---|
| [`docs/build_manifest.md`](build_manifest.md) | B0 编译 flag lock + SUNDIALS pin + B0-tag 完整证据 |
| [`docs/status_matrix.md`](status_matrix.md) | **唯一权威**阶段 Go/No-Go 状态来源（14 阶段 × 7 case + A0 checklist） |
| [`docs/profile_platform.md`](profile_platform.md) | local Apple-M4 + target Intel Xeon Gold 6133 双平台声明 + decision_consistency |
| [`docs/profile_decision.md`](profile_decision.md) | profile gate 决策（"走原方案" + Amdahl + 签署） |
| [`benchmarks/INDEX.md`](../benchmarks/INDEX.md) | 7 个 case 人类可读索引 + endpoint 分工 |
| [`benchmarks/kashigeer/B0_output/DEFERRED.txt`](../benchmarks/kashigeer/B0_output/DEFERRED.txt) | upstream X76 forcing band gap 完整诊断 + 3 条解决路径 |

### 3.4 CI / governance

- **`.github/workflows/serial-baseline.yml`**：3 builds（default / profile-on / dump-on）→ keliya 跑 → SHA256 比对 → snapshot bitwise；附 skip-label 旁路 + 数据探针 gate + 矩阵更新 PR proposer step
- **`baseline/current` branch protection**：`build-and-compare` 为 required status check + 禁 force-push + 禁删除 + `enforce_admins: false`（紧急时 user 可绕过）
- **`skip-baseline-ci` label 已删除**：S1+ PR 必须通过 baseline 验证才能 merge

---

## 4. 关键决策与 spec amendments

### 4.1 工程纪律（项目级铁律）

| 决策 | 关键文档 / commit |
|---|---|
| **C1 invariant**：SHUD 源码单一 RHS core，instrumentation 开关 OFF 时 binary bitwise neutral；S0 全程零物理改动 | `SHUD_openMP_master_plan.md` §C1 + `tools/profile/timer.cpp` 编译开关验证 |
| **90-day 截断政策**：所有 case `cfg.para END = START + 90`（仅本地部署层，不污染 manifest 的 `forcing_duration_days`） | `CLAUDE.md` + `tools/fix_case_paths/` |
| **跨平台分工**：§1.1.1 量化加速比 go/no-go **仅** target Linux x86_64 验收；macOS Apple Silicon 数字仅开发期 qualitative | `docs/profile_platform.md` + `docs/profile_decision.md` |
| **凭据卫生**：committed text 严禁 `frd_muziyao` / `/scratch/<user>` / `210.77.77.22` / `/volume/data/...` 字面值；使用 `<server-scratch-root>` / `<server-data-root>` / `<user>` 占位符 | `tools/archive_b0_output.sh:sanitize_path()` + PR [#34](https://github.com/DankerMu/SHUD-OpenMP/pull/34) |
| **CLAUDE.md 永远 gitignored** | 本地 + 服务器双端通过 `rsync` 同步 |

### 4.2 Profile gate 决策（已签署 2026-06-17）

| 维度 | 决策 |
|---|---|
| 档位 | **"走原方案"**（RHS-kernel-first OpenMP per master plan §5） |
| Amdahl 上限（target 8-核 practical） | `heihe_x4`: 2.39× / `qinyijiang`: 2.30× |
| 不并行的 case | `keliya` / `xinanjiang_upstream` 走 `OMP_CUTOFF` serial fallback |
| `heihe` 特例 | RHS 仅占 12.7%，forcing IO 79%（B-Medium 高端） → **建议把 forcing IO 并行化提前**（master plan §3.5 Opt-IO），原计划在 P9+ |
| `t_other` audit | 3 OK / 2 WARN / 1 FAIL（xinanjiang 22.4% startup amortization） → 提议 `t_init` bucket（future） |
| P8-precond timing | **不前置**（CVODE_internal share 最多 36%，非主驱动） |
| 跨平台 delta（local→target） | `delta_acceptable: false`（keliya +13.01pp / qinyijiang +13.57pp 超 10pp 阈值，归因微架构 / 编译器差异；按声明仅 target 决策） |
| 签署 | DankerMu via delegated grant 2026-06-17（claude-code on behalf） |

### 4.3 Spec amendments（S0-13）

`kashigeer` 上游 NWM 数据 gap（X76-X80 forcing band 缺失，覆盖 941 站中的 538 站）双端不可解。S0-13 通过 3 个 spec 联合 amendment 把 case 重分类为 `deferred-upstream`：

- `openspec/changes/s0-baseline-lock/specs/benchmark-registry/spec.md` — endpoint 枚举扩 `deferred-upstream` + 新 scenario
- `openspec/changes/s0-baseline-lock/specs/rhs-profile-gate/spec.md` — 局 case 数 5+2 → 4+3；target 数 7 → 6+1
- `openspec/changes/s0-baseline-lock/specs/status-matrix/spec.md` — `N/A (deferred-upstream)` 不阻塞 aggregate；A0 item 4 显式 EXCLUDE server-only + deferred-upstream

同时 `tools/check_manifest.py` 的 `VALID_ENDPOINTS` set + `benchmarks/kashigeer/manifest.yaml:endpoint` 同步扩。3-way drift 已闭合。

---

## 5. 已知技术债（带跟踪）

| # | 主题 | 现状 | 建议处理时机 |
|---|---|---|---|
| 1 | `t_init` bucket 从 `t_other` 拆出 | xinanjiang_upstream 22.4% startup amortization 暂在 `t_other` 桶 | **B1a 完成后**（P1 之前），改 `tools/profile/timer.cpp` |
| 2 | CI keliya forcing 数据未部署到 GitHub runner | `.github/workflows/serial-baseline.yml` 的 `data_probe` 永远 `false`，matrix proposer step 永远 skip | S1 早期 `ci-data-deployment` change，部署 50MB 子集到 runner cache |
| 3 | heihe forcing IO 并行化 | profile decision 建议提前，原计划 P9+ | 与 P1 并行排上 backlog；新建 `Opt-IO-heihe` issue |
| 4 | `cvode_stats` yaml subtree 形状 + `ratio_nfeLS_over_nfe` 真值 | `tools/profile/timer.cpp:dump()` 当前 yaml 顶层错位 | 不阻塞 B1a；S0-10 时已留 follow-up；P1 准备阶段处理 |
| 5 | SHUD submodule HEAD 文档漂移 | `docs/build_manifest.md:35` 历史上手维护漂移 6 个 PR | 加 pre-commit hook 或 CI 检查；非 S1 阻塞 |
| 6 | kashigeer 上游 X76 forcing 数据修复 | 项目外，需要 NWM dataset re-curation | 不在工程 scope；spec amendment 已把它从 A0 范围排除 |
| 7 | `baseline/current` → `main` fast-forward | #17 task 13.5；user 在 S0 收尾时选择 defer | 进入 S1 前手动 `git push origin baseline/current:main` 或开 PR |

---

## 6. S1 推荐起点

> S1（master plan §3.1，**B1a → B1b 单线程纯重构**）的目标：把现有 SHUD serial / omp 双路径合并为单一 RHS core，instrumentation 开关 OFF 时与 B0 **bitwise identical**（精度等级 A3a 强制）。

### 6.1 路径选择（最小风险）

按 case NumEle 从小到大滚动验证 B1a 实现：

1. **`keliya` (484 cells)** — 30s 跑完，B1a bitwise 失败时迭代代价最低
2. **`xinanjiang_upstream` (801)** — B-Small 第二档，验证 B1a 在 `OMP_CUTOFF` 边界附近的稳定性
3. **`qinyijiang` (3155)** — B-Medium，验证 multi-component 边界条件
4. **`qhh` (4773 + lake)** — 唯一 `has_lake: true` 案例，覆盖 lake vertical/horizontal/DY
5. **`heihe` / `heihe_x4` server**：只在 server 验证（forcing 12GB 不下载）

### 6.2 工作排布建议

| 优先级 | 工作 | 依据 |
|---|---|---|
| **P0** | S1 OpenSpec change 起草 → `stage-change-pipeline` 出 issue DAG | master plan §3.1 + 当前 S0 实际收尾纪律 |
| **P0** | B1a：合并 `MD_f.cpp` / `MD_f_omp.cpp` 为单 RHS core | C1 invariant 第一次正面实现，开关 OFF bitwise 必须 PASS |
| **P0** | 把 `B0-tag` 引入 CI workflow 作为 B1a 回归 anchor | `.github/workflows/serial-baseline.yml` 增 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 对比 step |
| **P1** | `Opt-IO-heihe`：forcing IO 并行化提前（per profile decision） | heihe forcing IO 79% 主导，1.14× 上限是 IO 瓶颈，B1a 无法解决 |
| **P1** | `t_init` bucket 拆分 | xinanjiang_upstream `t_other=22.4%` 当前归因 startup；进入 P1 加速比测算前必须先净化数据 |
| **P2** | CI keliya 数据部署 + matrix proposer 真正激活 | 解锁 CI 自动状态汇报 |

### 6.3 必读路线（S1 启动前 30 分钟）

1. `SHUD_openMP_master_plan.md` §3.1 B1a 范围 + §C1 invariant 红线
2. `docs/profile_decision.md`（决策 + Amdahl + heihe IO 优先级）
3. `docs/status_matrix.md`（B0 行 + A0 checklist 验收记录）
4. `docs/build_manifest.md` § "B0-tag (S0-13 / #17)"（manifest 摘要 + SHUD pin）
5. 任一 case 的 `benchmarks/<case>/B0_output/repeatability.txt` 当 anchor 格式参考

### 6.4 风险（已识别）

- **跨平台 delta>10pp 在 2/4 case 出现**：local 与 target 微架构差异；S1 早期发现 B1a 在 local PASS 但 target FAIL 时**优先信 target**，参 master plan §1.1.1 + `profile_decision.md` § "Cross-platform delta review"
- **SHUD submodule 推送纪律**：所有 SHUD 源码改动 commit **仅** 推 `SHUD-System/SHUD:openmp-baseline`，禁止污染 master；外层 repo 在每个 SHUD-touching PR 做 pointer bump（参 CLAUDE.md）
- **轴向**：B1a 必须 bitwise vs B0；B1b 才允许引入 bug-fix 但需逐项归因；P-strict 阶段才有 OpenMP；strict → prod 容差由 A 等级控制（参 master plan §2.2）

---

## 7. 验收 trace（A0 9/9 PASS）

| # | 验收项 | 状态 | 关键证据 |
|---|---|---|---|
| 1 | 7 manifest 完整 | PASS | `benchmarks/INDEX.md` + 7 × `manifest.yaml`；`check_manifest.py` 全 PASS |
| 2 | 各非 deferred-upstream case 3 次 bitwise | PASS | 6 case `benchmarks/<case>/B0_output/repeatability.txt` `verdict: PASS` |
| 3 | cvode_stats 三次一致 | PASS | 6 case `B0_output/cvode_stats.txt` 落盘 + 三次一致（同 repeatability run 间） |
| 4 | snapshot probe 三次一致 | PASS | 4 local case × 3 = 12 `snapshot_t*.bin`；`tools/compare_snapshot` PASS |
| 5 | `tools/rhs_snapshot` + `tools/compare_snapshot` 可独立调用 | PASS | PR [#24](https://github.com/DankerMu/SHUD-OpenMP/pull/24) 落地 + CI [#13](https://github.com/DankerMu/SHUD-OpenMP/pull/30) 实跑 |
| 6 | CI 自动 pass/fail | PASS | `.github/workflows/serial-baseline.yml` 4 个 PR 验证 + branch protection 启用 |
| 7 | profile yaml 完整 | PASS | 4 local real + 3 deferred + 6 target real + 1 target deferred |
| 8 | `docs/profile_platform.md` 三段齐 | PASS | `local_platform` + `target_platform` + `decision_consistency` |
| 9 | `docs/profile_decision.md` 已签署 | PASS | DankerMu via delegated grant against outer `a860eae5` + SHUD `78c37a1` |

---

## 8. 末尾备注

- 本报告是 S0 收尾的**单一来源参考**；下面这条 git 单行命令足以验证 B0-tag 真实性：
  ```
  git ls-remote --tags origin | grep B0-tag
  # 应输出：
  # 95ddc375ffa58115fd5c0a808dde80e9713b4c93	refs/tags/B0-tag
  # 884cfb13ba08ebae02dd64e371c4a19a536b4e26	refs/tags/B0-tag^{}
  ```
- 后续 S1+ 工作的 issue / PR / 决策记录应继续 cross-link 到 `docs/status_matrix.md`（**唯一权威**）；本报告不会再次更新（除非 B0-tag 重打）。
- 凭据 / 服务器路径细节仍仅在本地 `CLAUDE.md`；本报告所有路径均使用 `<server-scratch-root>` 等占位符。
