# P1d.1 — NUMA env 标准化 runbook

P1d epic PR-B (#276) 交付物。本文件是 **P1d NUMA / OMP 环境约定的唯一 source-of-record**：server sbatch 调用约定、三行 NUMA env、Mac OMP env（informational）、`numactl` primary/fallback 决策、以及 PR-B 阶段的实测拓扑 + 3-cell plumbing 验证证据。

参考：spec `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` Requirement "NUMA env 标准化（P1d.1）"；design `…/design.md` D1（phase ordering NUMA env → first-touch）/ R2（unexpected NUMA topology）/ OQ2（interleave=all vs cpubind）。

## §0 scope / non-scope

**本文件 owns（ENV only）**：
- server sbatch 三行 NUMA env + 调用约定
- Mac OMP env（弱绑定 informational only）
- `numactl` primary（`--interleave=all`）+ fallback（`--cpubind=0 --membind=0`）决策
- PR-B 实测 cn0X 拓扑 + 3-cell env-plumbing 验证

**本文件 NOT owns（显式分离，勿在此重复）**：
- **first-touch loop 源码设计**（element/river/lake zero-write loop 字段集、`schedule(static)` + `default(none)` pattern）→ 归 `docs/p1d/p1d_first_touch_design.md`（PR-K / PR-C 作者，per spec OQ1 设计 D2）。本 runbook 只管 *env*，不复制 first-touch 循环设计。
- A3a bitwise / nst 跨 N 的 determinism closure → PR-F（intermediate 8-cell）+ PR-H（final，post first-touch + Kahan revert）。PR-B **不**做 determinism 验收。

## §1 server sbatch 调用约定

### §1.1 模板位置与用法

sbatch 模板（gitignored 服务器侧基础设施，由 orchestrator 创建，**不入 repo**）：

```
/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/run_p1d_case.sbatch
```

用法：

```bash
# 从 /scratch 下提交（Slurm 三铁律 #1），<mode> 默认 interleave
sbatch run_p1d_case.sbatch <case> <N> [interleave|fallback]
#   <case>  : heihe / heihe_x4 / keliya / ...
#   <N>     : OMP_NUM_THREADS（1 / 2 / 4 / 8）
#   <mode>  : interleave（默认，primary）| fallback（cpubind=0+membind=0）
```

Slurm 三铁律对齐（per `CLAUDE.md` §"Slurm 三铁律"）：① 从 `/scratch` 提交；② `#SBATCH --output/--error` 路径在 `/scratch` 共享盘（compute node `/tmp` node-local，作业结束即丢，sacct 会显示 ExitCode 127）；③ 脚本引用的 patch / hash / run.sh 全放 `/scratch`。提交后落在 `cn05-06,09,14-19,23-24` 等 CPU 分区 idle 节点，**禁 login node 跑 SHUD**。

### §1.2 三行 NUMA env（spec grep gate = 3）

模板在 `numactl --interleave=all shud_omp <case>` 这一步前置标准化三行（单-task sbatch step，binary 直跑、`numactl` 直接 wrap，**不**经 `srun` 包裹 —— 与 P1c era 模板一致；per spec Scenario "server sbatch template 含 NUMA env"）：

```bash
export OMP_PROC_BIND=close      # 线程贴近放置，触发 NUMA first-touch 有意义
export OMP_PLACES=cores         # 绑定到物理 core
numactl --interleave=all <binary>   # primary：跨 NUMA node round-robin 内存分配
```

模板同时把 `env | grep ^OMP_` + `numactl --hardware` echo 进 job log（per spec Scenario "AND sbatch log 应含 numactl --hardware 输出"），使每跑的实测拓扑可追溯。

### §1.3 实测 cn03 NUMA 拓扑（cn0X CPU 分区代表节点）

PR-B 在 cn03（CPU 分区，cpus-per-task=8）经 `numactl --hardware` 实测（per design R2 实测落地）：

| 项 | 值 |
|---|---|
| NUMA nodes | 2（0–1），dual-socket |
| node 0 cpus | 0–19，size 95338 MB |
| node 1 cpus | 20–39，size 96714 MB |
| node distances | 0→0=10, 0→1=21, 1→0=21, 1→1=10 |

cross-node 访问 ≈ 2.1× local（21 vs 10）→ first-touch + `OMP_PROC_BIND=close` 在此 dual-socket 拓扑上 **确实重要**（非单 socket UMA 的 no-op）。这印证 design D1 phase ordering：先把 NUMA env 立起来，再上 source-level first-touch。

## §2 Mac OMP env（informational only）

Apple Silicon Mac 本地开发为 env 一致性同步设置（per spec Scenario "Mac OMP env 同步"）：

```bash
export OMP_PROC_BIND=close
export OMP_PLACES=cores
```

显式声明 **这是 informational / 弱绑定**：
- Mac libomp 对 `OMP_PROC_BIND` / `OMP_PLACES` 仅弱支持，不保证硬绑定。
- Apple Silicon 是单 socket UMA，**NUMA first-touch N/A** —— `tools/numa_check.sh` 在无 `numactl` 的 host（Apple Silicon / macOS）走 single-socket UMA 路径，直接 emit `socket_count: 1` + `numa_first_touch: N/A (single-socket UMA)`，exit 0。
- 无 `numactl` wrap（macOS 无 libnuma，Homebrew 无 numactl）。

Mac 数字仅 dev-reference（per `CLAUDE.md` §1.1.1：量化目标只在服务器验收）。Mac N=1 在 P1d 内的角色是 reverse-compat（PR-I/PR-J），non-Mac SHALL gate（A3a + nst 跨 N）严格限定 server。

## §3 numactl primary / fallback 决策（task 2.4）

| 模式 | 命令 | 角色 |
|---|---|---|
| primary | `numactl --interleave=all <binary>` | 默认（`sbatch … interleave`），跨 NUMA node round-robin，覆盖面更广 |
| fallback | `numactl --cpubind=0 --membind=0 <binary>` | 单 NUMA node 强制 binding（`sbatch … fallback`），仅当 interleave 与 build/run 冲突时启用 |

**决策（recorded）**：**interleave=all 保持 primary**。

依据（per spec Scenario "env 不可用时的 fallback（build 冲突触发）" + design OQ2）：
- PR-B 在 cn03 实测：`numactl --interleave=all` 与 SHUD `make shud_omp`（SHUD pin `3a0004c`）**无 build / run 冲突**（无 wrapper 报错 / 链接失败）。
- fallback path 已 **验证可用**：keliya N=4 fallback 跑通，wall=80s，与 `shud_omp` 无冲突 → fallback 随时可切，但当前无触发条件。
- 因此 spec Scenario 的 fallback 触发条件（interleave 与 build 冲突）**未命中**，interleave=all 维持 default。

二级 fallback 触发口（per spec 第二个 fallback Scenario / design OQ2 决策时机）：若 **PR-F** 8-cell intermediate 在 `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `numactl --interleave=all` 下三 SHALL gate 任一 FAIL，则 PR-G 阶段 fallback 至 `--cpubind=0 --membind=0` 重跑，决策记入 `docs/p1d/p1d_summary.md` §"NUMA env fallback rationale"。本 PR-B 不触此口（PR-B 只验 env plumbing，不验 determinism）。

## §4 验证证据（PR-B env-plumbing 3-cell）

keliya，SHUD pin `3a0004c`，90 天截断（START 12053 / END 12143，per 项目铁律 §"所有 case ≤90 天截断"）：

| Cell | NUMA mode | wall | nfe | rivqdown.dat sha256 | Slurm jobid |
|---|---|---|---|---|---|
| keliya N=1 | interleave | 87s | 110509 | `2ef2a42079d089789509c0738aa85e5a8a8e9f0ecdd081628c1ad0474341c373` | 8967 |
| keliya N=4 | interleave | 79s | 103703 | `126e8e8daf81799cd84792a058008977ba096b007049065904b2964ac104c411` | 8968 |
| keliya N=4 | fallback   | 80s | 110429 | `b319598b7aed45f87cbd66ff20fa5a817a2206750e19d8fe7dc450df0691a18e` | 8969 |

提交合规：从 `/scratch` `sbatch`；`--output/--error` 在 `/scratch`；compute 落 cn03/cn11（非 login node）。

> **NOTE（重要）**：上表跨 N 的 SHA **差异是预期的** —— 这是 **pre-first-touch + Kahan-injected（SHUD `3a0004c`）** 状态。PR-B 只验证 **NUMA-env plumbing 机制**（env 生效 + 拓扑捕获 + fallback 可用），**不**验证 cross-thread determinism。A3a / nst 跨 N 的 determinism closure 是 **PR-F**（intermediate）+ **PR-H**（final，post first-touch + Kahan revert），**不在 PR-B scope**。

## §5 tooling 引用（既有，勿重复其逻辑）

| 工具 | 一句话 |
|---|---|
| `tools/numa_check.sh` | NUMA 拓扑探针：捕获 `numactl --hardware` 到 `<run_dir>/numa_topo.log`，emit `socket_count` / `numa_first_touch` 摘要；multi-socket 且 `OMP_PROC_BIND` 未设时 exit 3（first-touch 结构性失效告警）。Apple Silicon 走 single-socket UMA 路径 exit 0。 |
| `tools/run_omp.sh` | OMP 线程绑定 wrapper：未设时 export `OMP_PROC_BIND=close` / `OMP_PLACES=cores` / `OMP_NUM_THREADS=1`（caller env 优先，不剥夺 SLURM override），把最终 OMP_* 状态打到 stderr，再 `exec` 转发到 SHUD 二进制。 |

server sbatch 模板的 env 行与上述两工具的 default 对齐（`OMP_PROC_BIND=close` + `OMP_PLACES=cores`），sbatch 在其之上追加 `numactl --interleave=all` 这一 server-only 内存策略层。

## §6 引用来源（source of truth）

| 文档 | 内容 |
|---|---|
| `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` | Requirement "NUMA env 标准化（P1d.1）" + 4 Scenarios（含两条 fallback） |
| `openspec/changes/p1d-numa-governance/design.md` | D1 phase ordering / R2 NUMA 拓扑 / OQ2 interleave vs cpubind |
| `docs/p1d/p1d_first_touch_design.md` | first-touch loop 源码设计（PR-K/PR-C；本 runbook 不重复） |
| `docs/p1d/p1d_summary.md` | P1d capstone + "NUMA env fallback rationale" / "NUMA 拓扑实测" 节（PR-K） |
| `tools/numa_check.sh` / `tools/run_omp.sh` | NUMA 探针 + OMP 绑定 wrapper（既有 S5d.4 工件） |
| `docs/p1c_summary.md` | P1c carve-out → P1d hand-off（§5.2 P9/P1d 治理范围列项） |
| `CLAUDE.md` | Slurm 三铁律 + 双端实验环境 §服务器 + §1.1.1 量化目标只在服务器验收 |
