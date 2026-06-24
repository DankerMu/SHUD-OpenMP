# P1d — epic executive report

executive 层 report，面向项目所有者 + 跨 epic 复盘。详细数据 / 实测表 / 错误纠正逐项分析见 `docs/p1d/p1d_summary.md` + `docs/p1d/p1d_perf_baseline.md` + `docs/p1d/p1d_numa_root_cause.md`。

## §1 Epic ID + 时间线

| Field | Value |
|---|---|
| Epic | #274 |
| 起 | 2026-06-22 (P1c PARTIAL CLOSURE 后 P1d intake) |
| 止 | 2026-06-24 (PR-K capstone + master plan v1.5 / M10 merged) |
| 跨度 | 3 天（2 天 burst + 1 天 capstone 修订） |
| PR 总数 | 13 + PR-C0 insertion = **14 PR** |
| 平均 wall（含 server Slurm + Mac local + GPT 复查 + 修订） | ~12-15 hours engineer time |

## §2 Status

**PARTIAL CLOSURE via E′ containment path**。

不是简单 E (ship serial-only + walk away from OMP)，是 E′：保留全部 P1d 工程结果 + 4-mode spec 把 strict 承诺限定到正确 mode + 把 P1e (F 路) 作为下个 epic 把 "真正应并行的 RHS 还没并行" 补上。

详 `docs/p1d/p1d_summary.md` §5 8 项动作。

## §3 What was attempted

P1c PARTIAL CLOSURE 后 carve-out 是 "upstream parallel writer first-touch / NUMA-affinity 治理"。P1d 按这条 hypothesis：

1. **NUMA env standardization**：server sbatch `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `numactl --interleave=all` (PR-B `docs/p1d/p1d_numa_env_runbook.md`)
2. **steady-state first-touch loops** in `SHUD/src/Model/MD_rhs_core.cpp` element + river + lake 3 owner blocks (PR-C/D/E)
3. **Kahan revert**：撤 P1c §4.7 conditional Kahan/Neumaier injection 看 first-touch 单干能不能关 A3a + nst Δ=0 (PR-G)
4. **server final 8-cell 验收**：heihe + heihe_x4 × N∈{1,2,4,8} 3 SHALL gate (PR-H)

期望：3 SHALL gate 全 PASS → P1c carve-out closure → 进 P2a。

## §4 What was delivered

| 类别 | 交付物 |
|---|---|
| Codebase fact-check | 5/5 codebase 事实核查 (per `docs/p1d/p1d_summary.md` §6) 写入 master plan §6 P1d.2 + PR-H post-verdict 修订 |
| Corrected root cause | NVECTOR_OPENMP `reduction(+:sum) schedule(static)` 是 cross-N 散度根因（详 `docs/p1d/p1d_numa_root_cause.md` §4） |
| 4-mode spec rewrite | `serial` / `strict-omp` / `det-omp` / `fast-omp` 四模式拆分（PR-M PROMOTE scope） |
| Mac reference data | PR-I 独立 worktree 切 `P1-update-omp-tag` 跑 keliya + qhh × {serial, omp@N=1, omp@N=8} 6-cell anchor (`docs/p1d/p1d_pr_i_p1_update_omp_reference.md`) |
| Kahan revert (PR-G) | 4 surgical revert in `MD_rhs_core.cpp`；Mac 9-SHA matrix 证明 revert 干净 + N=1 byte-identical to pre-K2 + P1-update-omp-tag canonical |
| Master plan v1.5 / M10 | P1d epic capstone 修订 §6 + §7 + §8 全部章节，含 P1e 新章节 |
| Capstone docs ≥11 | `docs/p1d/` 11 份必备 doc + PR-K self-evidence doc + 2 doc updates |

## §5 What was NOT delivered

| 项 | 状态 | 原因 |
|---|---|---|
| §4.4 A3a 跨 N strict bitwise (server only) | FAIL | NVECTOR_OPENMP reduction 顺序不固定；P1d hypothesis 错误（first-touch 治不了 N_Vector reduction） |
| §4.5 nst Δ=0 (heihe) + \|Δ\|≤2 (heihe_x4) | FAIL | 同上 (CVODE WRMS norm 走 N_Vector reduction → adaptive step size 不固定) |
| 加速比 ≥1.5× | NOT MET (heihe 1.13× / heihe_x4 1.27×) | 真正应并行的 RHS 还没并行（StrictOMP 是 abort 桩），1.27× 距 Amdahl 上界 2.39× 还有 1.12× 差距 |
| production-quality OMP path | NOT MET | E′ 把 `shud_omp` 标 `fast-omp experimental, non-production`；production 默认 NUM_OPENMP=1 (serial) |

**关键**：以上失败**不是 P1d 工程实施错误**，是 P1d hypothesis 错误。NUMA env + first-touch + Kahan revert 全 stack 实施都正确（PR-G Mac 9-SHA 证明），但 hypothesis 选错了 fix layer——cross-N 散度根因在 SUNDIALS NVECTOR_OPENMP backend 内部，不在 SHUD owner-compute layer。

## §6 Why E′ over E

**Option E**: ship serial-only + walk away from OMP + 关 P1d。简单 E 的问题：

- 把 P1d 工程结果丢掉（PR-C/D/E first-touch loops + PR-G Kahan revert 的工作量浪费）
- 没解释 fact-check finding（Serial RHS + NVECTOR_OPENMP 这个奇怪组合 + StrictOMP abort 桩）
- 没指明 P1e 怎么做（"strict OMP path 怎么实现" 没答案）
- spec 不诚实（保留 P1c era 的 SHALL gate 文本，但实际跑 fast-omp mode）

**Option E′**: containment closure，含 4-mode spec rewrite + 错误叙事更正 + first-touch deprecation note + 指向 P1e。E′ 优势：

- 保留全部 P1d 工程结果（first-touch loops 标 deprecated 但代码留着；Kahan revert 保留作 P1e baseline）
- 4-mode spec 把 strict 承诺限定到 `strict-omp` mode（待 P1e 实现），不弱化承诺 + 不假装当前 `fast-omp` 是 strict
- P1e (F 路) 作为下个 epic 把 "真正应并行的 RHS 还没并行" 补上，启动前置已写明（2×2 build matrix 因果实验）
- spec 诚实：`fast-omp` 标 experimental + non-production；`serial` mode 是 production default

E′ 是 "Linus 路线": 不假装 P1d 成功，但也不放弃 P1d 的工程工作；把问题归类到正确层，把下一步明确指出。

## §7 Why F over B

**Option B**: SPGMR → KLU refactor。fact-check 后 B 的问题：

- KLU 不能单独 fix。CVODE WRMS norm 还是过 N_Vector reduction → 换 KLU 不换 NVector 仍跨 N 漂
- determinism 与 solver 选择**正交**——KLU vs SPGMR 选哪个不影响 cross-N 散度根因
- KLU full Jacobian factorize 要量化 fill ratio + memory peak + factor wall，是 P2/P3 决策
- KLU 是 production solver 优化，不是 strict closure

**Option F**: Serial N_Vector + StrictOMP RHS。F 优势：

- 用现有 SUNDIALS 不改 vendor（Serial N_Vector 是 SUNDIALS stock backend）
- 可立即用 deterministic gather（spec 已设计好 `rhs_deterministic_gather()` 基础设施）
- hydrology RHS 真并行 → 真去吃 RHS 66.55% wall → 理想 2.39× 上界可达
- closure 不要求 vendor change

P1e 首选 F 路。若 mode C (Serial NVec + StrictOMP RHS) 跨 N 仍漂，再 fallback 到 ADR-0001 的二选项 NVECTOR_REPRO_OMP；KLU 仍推到 P2/P3。

## §8 Outputs

| 类别 | Item | 状态 |
|---|---|---|
| Git tags | `P1d-tag` annotated (D11 6-tag chain: B0 → B1a → B1b → B1 → P1-update-omp → P1c → **P1d**) | PR-L PENDING |
| Baseline branches | `baseline/P1d` (HEAD `a19fb5e`, lock_branch=true) | PR-L PENDING |
| Master plan revision | v1.5 / M10 (merged 2026-06-24) | DONE |
| OpenSpec specs | `p1d-numa-governance` (PROMOTE per PR-M) + `p1d-capstone` (PROMOTE per PR-M) | PR-M PENDING |
| Docs | 11 必备 + PR-K self-evidence + 2 update = **14 doc 总计 in `docs/p1d/`** | 本 PR |
| ADR | `docs/adr/0001-solver-path.md` 4-way solver comparison | Phase 2(e) 并行 agent owns (not this PR) |
| P1e openspec | `openspec/changes/p1e-strict-omp-rhs/` proposal + design + tasks + specs | Phase 2(e) 并行 agent owns (not this PR) |
| Status matrix | P1d 行 PARTIAL CLOSURE + P1e 行 PENDING | 本 PR |
| Build manifest | SHUD pin trail P1d 段 + FP gate 3-flag 形 | 本 PR |

## §9 Lessons learned

| # | Lesson | 来源 |
|---|---|---|
| 1 | **measure rivqdown before declaring nst Δ as the gate** | P1c era 全部用 nst Δ 作 SHALL gate；P1d PR-H 实测 rivqdown.dat 后才发现 10-25% mean_rel 这个工程 fail 量级；Kahan 注入只压 nst 没修 rivqdown。P1e 验收 SHALL 加 rivqdown.dat mean_rel ≤ULP gate |
| 2 | **verify execution path actually executes the parallel code before crediting parallelism** | P1d 14-PR burst 实施全 stack first-touch + Kahan revert + NUMA env，但没人 grep `ExecPolicy::Serial` vs `StrictOMP` 实际跑哪条；事实核查后才发现 hydrology RHS 完全单线程。P1e 启动前 2×2 build matrix 实验**必跑**，验证假设 |
| 3 | **reduction-order issues affect WRMS norm, not just owner-gather** | P1c 全力 fix `MD_rhs_core.cpp` 8 sites 内部 reduction（Kahan 注入）；P1d 改 NUMA env + first-touch；都没碰到 N_Vector layer reduction。WRMS norm + Krylov inner product 走 SUNDIALS NVECTOR_OPENMP，是 cross-N 散度的另一独立 source |
| 4 | **fact-check claims about library internals against actual source** | PR-H 初版诊断 "drift origin 在 SPGMR multi-threaded preconditioner" 是从 GPT 推理来的，没 grep `CVodeSetPreconditioner`；fact-check #5 一行 grep 就推翻 |
| 5 | **stage hypothesis 与 stage 工程实施分开 verify** | P1d 工程实施全部正确（PR-G Mac 9-SHA 证 revert 干净 + first-touch loops 字段集 OQ1 设计严密 + NUMA env runbook 完整）；但 hypothesis 错了——选错了 fix layer。lesson：epic intake 阶段不要只验工程方案 feasibility，要验 hypothesis 是否触及真正根因 |
| 6 | **GPT Pro 双重独立复查 + codebase 事实核查 catch this** | 单 agent 容易自我循环（基于初版诊断推导细节）；两轮独立 GPT Pro 复查 + 5/5 codebase 事实核查在 1 天内推翻 4 项初版错误诊断。后续 strict 阶段 capstone 默认走双重复查 |

## §10 Acknowledgments

- **GPT Pro 双重复查**：2 轮独立复查（Pro1 + Pro2）catch 4 项初版错误诊断 + 提出 NVECTOR_REPRO_OMP / block-Jacobi precond fallback 路径 + 出 `rivqdown.dat` 输出缓存 audit warning
- **spec 纪律**：M10 history immutable preserved——P1c spec 字面 "carve-out 推 P9 行" 不动；P1d closure 通过本 doc + `docs/p1d/p1d_summary.md` §10 显式记录 "P9 字面 = P1d 语义同义" 映射
- **D11 6-tag chain 完整性**：B0-tag → B1a-tag → B1b-tag → B1-tag → P1-update-omp-tag → P1c-tag → **P1d-tag** (PR-L) 全部 annotated + immutable + 不允许 force-update
- **CLAUDE.md project iron rules**：90-day 截断 / `uv` Python / Slurm 三铁律 / SHUD submodule master 不动 全部 PR-A → PR-K 严格遵守

## §11 Forward plan

### §11.1 Immediate next (PR-L + PR-M, 同 P1d epic 内)

- **PR-L**: 创建 `P1d-tag` annotated + `baseline/P1d` lock (D11 6-tag chain). 详 `docs/p1d/p1d_tag_and_lock.md`（PR-L 作者生成）
- **PR-M**: PROMOTE 2 specs (`p1d-numa-governance` + `p1d-capstone`) + archive to `openspec/changes/archive/2026-06-24-p1d-numa-governance/` + glossary 4 新术语 + jsonl 双追加 (stage-pipeline-log + post-stage-cleanup) + Epic #274 close + propose `p1e-strict-omp-rhs` openspec change

### §11.2 Next epic (P1e, new epic)

- **P1e intake**: openspec `p1e-strict-omp-rhs` change propose（Phase 2(e) 并行 agent owns）
- **P1e.1 启动前置实验**: 2×2 build matrix × N∈{1,2,4,8} × 3 reps 因果实验（4 build = A/B/C/D per `docs/p1d/p1d_summary.md` §9.3）
  - 判据若 mode C cross-N bitwise + nst Δ=0 + 加速 ≥1.5× → F 路成立 → P1e.2-7 实施
  - 判据若 mode C 失败 → 进 ADR-0001 评估二选项 NVECTOR_REPRO_OMP
- **P1e.2-7**: StrictOMP RHS 实施 + deterministic gather 复用 + first-touch 重新设计 + `rivqdown.dat` 输出 audit + 3 SHALL gate strict-omp mode 验收
- **P1e.8 capstone**: `P1e-tag` (D11 7-tag chain) + `baseline/P1e` lock + ADR-0001 close out

### §11.3 Long-term (P2a+)

- **P2a 启动前置**: 原 "P1c-tag lock + P1c.3 A3a 全通过" 不再充分。新前置 = **P1e (F 路) 完成 strict-omp mode 实现 + 3 SHALL gate 在 strict-omp mode 内通过 + 加速比 ≥1.5× + `P1e-tag` 已 push + `baseline/P1e` lock + ADR-0001 close out**
- **ADR-0001 (solver-path)**: Phase 2(e) 并行 agent 创建 4 路 solver 对比（Serial NVec + StrictOMP RHS / Deterministic NVECTOR_REPRO_OMP / SPGMR + block-Jacobi precond / KLU sparse direct），不阻塞 F 路

## §12 References

| 文档 | 用途 |
|---|---|
| 本文件 (`docs/p1d/p1d_report.md`) | epic executive report |
| `docs/p1d/p1d_summary.md` | epic capstone narrative (§1-§12 详细) |
| `docs/p1d/p1d_perf_baseline.md` | wall + Amdahl + Mac 6-cell + CPU-hour cost ROI |
| `docs/p1d/p1d_numa_root_cause.md` | 技术 autopsy (§4 N_Vector reduction + §5 first-touch 失效 + §6 Kahan orthogonal axes + §7 4 错误诊断纠正) |
| `docs/p1d/p1d_first_touch_design.md` | PR-C/D/E first-touch 字段集 OQ1 设计（M10 后 steady-state 标 DEPRECATED） |
| `docs/p1d/p1d_numa_env_runbook.md` | PR-B NUMA env standardization runbook |
| `docs/p1d/p1d_kahan_revert.md` | PR-G SHUD Kahan revert + Mac 9-SHA 证 revert 干净 |
| `docs/p1d/p1d_pr_f_intermediate_run.md` | PR-F Kahan IN 8-cell + `--interleave=all` anti-pattern finding |
| `docs/p1d/p1d_pr_h_final_run.md` | PR-H final 3 SHALL gate verdict + E′ post-verdict 修订 |
| `docs/p1d/p1d_pr_i_p1_update_omp_reference.md` | PR-I Mac `P1-update-omp-tag` worktree 6-cell anchor |
| `docs/p1d/p1d_pr_k_capstone_run.md` | PR-K self-evidence (本 PR) |
| `docs/p1d/p1d_tag_and_lock.md` | PR-L `P1d-tag` + branch lock (PR-L 作) |
| `SHUD_openMP_master_plan.md` v1.5 / M10 | §6 P1d + §6 P1e + §7.2 RISK-NEW1/2 + §7.3 stage rows + §8.1 4-mode block |
| `openspec/changes/p1d-numa-governance/` | P1d openspec change (PR-M PROMOTE) |
| `openspec/changes/p1e-strict-omp-rhs/` | P1e openspec change (Phase 2(e) 并行 agent owns) |
| `docs/adr/0001-solver-path.md` | 4 路 solver 对比 ADR (Phase 2(e) 并行 agent owns) |
| `docs/status_matrix.md` | P1d 行 PARTIAL CLOSURE + P1e 行 PENDING（本 PR 更新） |
| `docs/build_manifest.md` | SHUD pin trail P1d 段 + FP gate 3-flag 形（本 PR 更新） |
