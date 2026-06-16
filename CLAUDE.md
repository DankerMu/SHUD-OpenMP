# SHUD-OpenMP — Agent 指南

## 项目速览

- **是什么**：SHUD 全耦合水文模型的 OpenMP 并行加速工程。唯一权威路线是 `SHUD_openMP_master_plan.md`（v1.2）——所有改造遵循其阶段划分与门控，不绕过。
- **技术栈**：C++ / SUNDIALS-CVODE 6.0.0 / OpenMP。
- **SHUD 源码是 git submodule**：`SHUD/`（来自 `SHUD-System/SHUD`，pinned to `3aec657`）。改 SHUD 源码即改 submodule，注意区分上层 openMP repo 的提交与 submodule 内提交。
- **关键命令**（在 `SHUD/` 内执行）：
  - 装依赖：`./configure`（下载并安装 SUNDIALS/CVODE 6.0.0 到 `InstallSundials`）
  - 编译串行：`make clean && make shud`
  - 编译 OpenMP：`make shud_omp`
  - 其它：`make all` · `make check` · `make help` · `make clean`；非 gcc 工具链先改 `Makefile`
  - 精度验收：靠 benchmark 算例对基线做 **bitwise / 容差对比**。<!-- TODO: benchmark 体系是 master plan S0 的产物，建好 benchmarks/ + RHS snapshot 工具后补具体对比命令 -->
- **目录**：`SHUD/`（submodule 源码）、`archive/`（被 master plan 取代的旧方案，只读参考）、`figures/`（方案插图）、`rAnalysis/`、`Script/`。

## 领域规则与禁区（改造铁律，细节见 master plan）

- **三阶段递进，逐阶段门控，不通过不进入下一阶段**：预并行 S0–S6 → strict 并行 P1–P7 → production 并行 P8–P9。
- **基线纪律**：B0（原样）→ B1a（纯重构，必须与 B0 **bitwise identical**）→ B1b（bug-fix 后，差异逐项归因）→ P-strict（与 B1b bitwise）→ P-prod（容差内可解释、可复现）。
- **精度等级 A0–A5**：strict 阶段 A3a（同线程数完整 run 与 B1b **bitwise identical**）强制；对外发表水文结果只需 A3a + A4，A3c 仅加分。
- **核心原则 C1–C8**：唯一 RHS core（serial/omp 不再各自演化，OpenMP 只是 execution policy）· compute 与 gather 分离 · strict 阶段不改物理 · CVODE 内部并行晚于 RHS 并行 · 阶段门控 · profile-driven 优先级（占比 <10% 不投并行预算）· fork-join 最小化 · 小流域 `NumEle < OMP_CUTOFF` 走 serial。
- **跨平台验收分工**：本地（Apple Silicon Mac）跑的数字仅开发期参考；加速比 go/no-go **只在目标部署平台**验收（单插槽 8 物理核 x86_64 Linux，`-O2 -ffp-contract=off -fopenmp`，`OMP_PROC_BIND=close OMP_PLACES=cores`）。

## 已装能力（投影在 `.claude/`，平台：Claude Code）

两个 pack，对上本工程：

- **agentic-issue-delivery** — 把 master plan 的 S/P 阶段任务落成带门控的交付：
  - `gh-create-issue` / `clarify` / `brainstorming`：阶段任务转 issue、澄清需求
  - `stage-change-pipeline` + `subagent-workflow`：带审核闸的变更流水线，契合"每步门控"
  - `grill-me` / `grill-with-docs`：方案 / 需求压测（本工程方案严谨，适配）
  - `implementation-planning` / `review` / `risk-adaptive-cross-review`：规划与多视角评审
- **codebase-stewardship** — 对上"双路径合一 + 去冗余 + 定方向"：
  - `improve-codebase-architecture`：C1 唯一 RHS core 重构（serial/omp 两套 RHS 合一为同一 kernel 的 execution policy）
  - `entropy-review` / `repo-entropy-audit`：serial/omp 双路径、被 `PassValue()` 覆盖的死 `+=` 正是 entropy
  - `future-aware-architecture`：`_OPENMP_ON` 解耦为三正交宏、热字段 SoA layout 等方向决策
  - `control-plane-auditor`：编译开关 / 宏 / 运行时配置的控制面审计

## 项目本地适配（living 文件，建议建）

- `openspec/glossary.md` — 本工程术语密集，**强烈建议建**单一术语表：B0/B1a/B1b、A0–A5、RHS core、SPGMR/Krylov/nfeLS、owner-local gather、first-touch/NUMA、OMP_CUTOFF…；由 `grill-with-docs` / `improve-codebase-architecture` 维护。
- `docs/adr/NNNN-slug.md` — 长期架构决策账本：宏三正交化、KLU vs preconditioned SPGMR、热字段 SoA 抽取…
- 以上是 `SHUD_openMP_master_plan.md` 的**补充**，不替代；master plan 仍是阶段路线的唯一权威。

## 反熵约定

根指令保持精简。能力操作细节在各 `.claude/skills/*/SKILL.md`；改造细节在 master plan 与各阶段产物；不在本文件展开。子树需细化时就近新增 scoped 指令文件。

## Observable Completion

完工附一行 `Execution Summary: agents=…; skills=…; tools=…; verification=…; limits=…`；保持事实，不展开隐藏推理。strict 阶段的 `verification` 应说清对哪条基线、达到哪个精度等级（如 `A3a bitwise vs B1b @ 4 threads`）。

## Claude Code Notes

- 知识域类 skill（如 `grill-me`、`future-aware-architecture`）自动触发率低，优先显式 `/skill-name` 调用。
- 投影副本（`.claude/skills`、`.claude/agents`）由 my-agents 安装生成，勿手改；改 canonical 包后重装。
- 涉及编译 / 跑模型的重活，遵循全局工作流：Claude 做 intake / 规划 / 验证，执行交 `codeagent`。
