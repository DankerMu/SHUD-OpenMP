# SHUD-OpenMP 文档索引

本文件是 `docs/` 目录的入口索引，按 **epic 阶段 + 文档类型** 组织。

- 详细 epic capstone 见各 `*_summary.md`
- 过程证据 / detail report 见 epic 子目录（`p1/`、`p1c/`、`p1d/`、`p1e/`、`b1b/` 等）
- 单 source-of-truth（status matrix、build manifest 等）保留在顶层，便于跨 epic 共享
- 历史归档（已淘汰阶段产物）汇入 `archive/<era>/`

权威路线见仓库根 `SHUD_openMP_master_plan.md` 与 `CLAUDE.md`。

---

## 顶层 entry-point 文件（capstone + 全局基础设施）

顶层保留的文件可分为 4 组：

### 1. Epic capstone summary（epic 完成态入口）

每一个 P-stage epic 结束时生成 `<epic>_summary.md` 作为该 epic 的 capstone entry-point。

| 文件 | Epic | 状态 |
|---|---|---|
| [`p1_summary.md`](p1_summary.md) | P1 update-omp baseline | 完成 2026-06-22 |
| [`p1c_summary.md`](p1c/p1c_summary.md) | P1c deterministic-reduction | 完成 2026-06-23 |
| [`p1e_summary.md`](p1e_summary.md) | P1e StrictOMP RHS, F path | 完成 2026-06-25（current） |

注：
- **P1c summary** 在 `docs/p1c/` 子目录（早期阶段约定，未上提顶层）
- **P1d 无顶层 summary**；详见 [`docs/p1d/p1d_report.md`](p1d/p1d_report.md)（go/no-go verdict 入此）
- **P1b 未独立成 summary**（被 P1c 吸收）

### 2. B-chain summary（基线锚点）

B 系列基线是 OpenMP 改造的精度参照锚点（vs B0 / vs B1b bitwise 比对）。

| 文件 | 基线 | 说明 |
|---|---|---|
| [`b0_summary.md`](b0_summary.md) | B0 baseline | 原始串行基线 |
| [`b1a_summary.md`](b1a_summary.md) | B1a baseline | 第一次 OpenMP 入栈基线（`baseline/B1a` 已 lock） |
| [`b1b_summary.md`](b1b_summary.md) | B1b baseline | 当前活动 B-chain head（S5* SoA + S6b bug fix 完成） |
| [`b0_vs_b1b_water_balance_report.md`](b0_vs_b1b_water_balance_report.md) | B0 vs B1b | 跨基线 water balance 比较（验证 B1b 物理一致性） |

历史 frozen baselines：`baseline/B1a` 已 lock_branch=true。当前活动开发线 = `baseline/B1b`。

### 3. 全局基础设施 / 单 source-of-truth

跨 epic 共享、引用频次高（>20 次）的基础设施 doc。

| 文件 | 用途 | 引用频次 |
|---|---|---|
| [`status_matrix.md`](status_matrix.md) | 阶段 × benchmark case 状态矩阵 | ~75 |
| [`build_manifest.md`](build_manifest.md) | build provenance（编译选项 / 版本指纹 / SHUD hash） | ~49 |
| [`profile_decision.md`](profile_decision.md) | profile-based optimization 决策记录 | ~27 |
| [`profile_platform.md`](profile_platform.md) | profile 测试平台描述（Mac M4 Pro vs Linux cn03） | (见 build_manifest 索引) |
| [`topology_manifest.yaml`](topology_manifest.yaml) | 网格 topology manifest（hot fields + first-touch 分区） | ~23 |

这些文件 **强烈反对下沉到 epic 子目录** — 跨阶段引用密度过高，下沉会破坏所有相对路径。

### 4. Cross-run accountability log（append-only JSON Lines）

跨 epic 累计、永久 append-only 的审计日志。

| 文件 | 内容 |
|---|---|
| [`review-loop-log.jsonl`](review-loop-log.jsonl) | 每个 issue/PR 的 reviewer 数 × round 数 × gate_net_catch × verdicts × residual_deferred |
| [`stage-pipeline-log.jsonl`](stage-pipeline-log.jsonl) | stage-change-pipeline skill 执行 log（OpenSpec change → SHALL gate verify → tag） |

格式：JSON Lines（每行一个 JSON 对象），按时间顺序 append。**不要** 用 git rebase squash 改 history；append-only 是这两个 log 的硬约束。

---

## Epic 子目录（detail docs）

每个 epic 阶段的过程产物（audit / report / experiment / 设计稿）按 epic 名归到子目录。

| 子目录 | Epic / 用途 | 内容摘要 | 文件数 |
|---|---|---|---|
| [`p1/`](p1/) | P1 update-omp baseline | audit_update_funcs + perf_baseline + RHS_snapshot + fullrun_bitwise | 4 |
| [`p1c/`](p1c/) | P1c deterministic-reduction | reduction site enumeration + Kahan injection design + reverse-compat verify + capstone summary | 12 |
| [`p1d/`](p1d/) | P1d NUMA + first-touch + Kahan revert | NUMA env runbook + first-touch 设计 + go/no-go verdict + capstone report | 13 |
| [`p1e/`](p1e/) | P1e F path StrictOMP RHS | 2×2 experiment + StrictOMP 设计 + perf 表 + SHALL gate verification + tag/lock + academic summary | 19 |
| [`b1b/`](b1b/) | B1b 时代 audit 产物 | S2 semantic diff + S2.17 lake formula audit + S5d hot fields YAML + S6b3 candidates | 4 |
| [`adr/`](adr/) | Architecture Decision Records | 长期架构决策账本 | 2 (ADR-0001 SoA hot fields / ADR-0002 solver-path) |
| [`audit/`](audit/) | 跨 epic audit 产物 | 单次专项 audit（如 qinyijiang regression 调查） | 1 |
| [`diff_reports/`](diff_reports/) | B1a vs B1b s6b 阶段 diff | snapshot 比对 报告（s6b_1 / s6b_2 / s6b_3_1） | 3 |
| [`review/`](review/) | PR review 详细记录 | 早期 PR review notes（pr-191..pr-289） | 9 |
| [`archive/`](archive/) | 归档（已淘汰 / 早期阶段产物） | s2-era evidence dotfile 等 | 含子目录 `s2-era/` (6 files) |

---

## 命名约定

| 模式 | 位置 | 例 |
|---|---|---|
| Epic capstone summary | 顶层 `docs/<epic>_summary.md` | `p1e_summary.md` |
| Epic detail docs | 子目录 `docs/<epic>/<epic>_*.md` | `docs/p1e/p1e_2x2_experiment.md` |
| 跨 epic / 全局基础设施 | 顶层 `docs/<name>.md` | `status_matrix.md` |
| 历史归档 | `docs/archive/<era>/<name>.md` | `docs/archive/s2-era/s2-pr1-evidence.md` |
| append-only log | 顶层 `docs/<name>.jsonl` | `review-loop-log.jsonl` |
| YAML manifest | 顶层 / epic 内 `*.yaml` | `topology_manifest.yaml`, `docs/b1b/s5d_hot_fields.yaml` |

**约定补充**：
- Epic 子目录用 **lowercase 短名**（`p1`, `p1c`, `b1b`），不带 `docs_` 前缀
- 大小写归一：B-chain 标识统一 lowercase（`b0_vs_b1b_*.md`，非 `B0_vs_B1b_*.md`）
- 隐藏 dotfile（`.s2-prN-evidence.md`）归档时去前导 dot，使其在 `ls docs/archive/s2-era/` 中可见

---

## 找文档的快速路径

| 我想找... | 去哪... |
|---|---|
| 某 epic 的总入口 | 顶层 `<epic>_summary.md`（P1d 除外 → `docs/p1d/p1d_report.md`） |
| 某 case × 阶段当前状态 | `status_matrix.md` |
| 某次 build 的 SHUD hash / 编译选项 | `build_manifest.md` |
| 某次 OpenSpec change 的 review 累计 | `review-loop-log.jsonl` |
| B1a vs B1b 数值精度差异 | `docs/diff_reports/B1a_vs_B1b_*.md` |
| 长期架构决策（为什么用 SoA / 为什么走 SPGMR） | `docs/adr/*.md` |
| P1e 2×2 实验设计与结果 | `docs/p1e/p1e_2x2_experiment.md`, `docs/p1e/p1e_perf_2x2.md` |
| S5d hot fields 分类 | `docs/b1b/s5d_hot_fields.yaml` |
| 已归档 S2 阶段证据 | `docs/archive/s2-era/s2-prN-evidence.md` |

---

## 历史变更

- **2026-06-25**：docs/ 顶层梳理 B 折中方案
  - 建 `docs/p1/` + `docs/b1b/` + `docs/archive/s2-era/` 3 个子目录
  - 移 14 文件（4 P1 detail + 4 B1b audit + 6 隐藏 s2 evidence dotfile）
  - 大小写归一 `B0_vs_B1b_water_balance_report.md` → `b0_vs_b1b_water_balance_report.md`
  - 引用更新 138 处 file-replacements
  - 顶层文件计数 27 → 13
  - 新增本 README.md INDEX
