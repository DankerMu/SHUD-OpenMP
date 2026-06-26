# P1e PR-N — block-Jacobi preconditioner (PLACEHOLDER, NOT TRIGGERED)

> **Status (PR-K 2026-06-25)**: **not triggered** (D12.3 fallback path not exercised, per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §8.2 routing decision)
>
> This file is a **conditional placeholder** per spec `p1e-capstone` Requirement "docs/p1e/ ≥14 doc" Scenario + tasks §7.10.2 "若 D12.3 PR-N 未触发：SHALL 创建 `docs/p1e/p1e_pr_n_block_jacobi.md` placeholder note"。当 P1e epic 内 D12.3 路由实际未触发时，本 doc 仅做存在性占位以满足 capstone spec doc-count Requirement，不含 PR-N 实施数据。

## §1 PR-N 触发条件

per `openspec/changes/p1e-strict-omp-rhs/tasks.md` §8 P1e.8 fallback：

> **触发条件**：PR-I 3 SHALL gate PASS + **两 case 均** < 各自 threshold (per task 4.6)

即 D12.3 AND-gate: **BOTH** heihe < 1.3× AND heihe_x4 < 1.5×。

## §2 PR-I 实测结果 (per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.3)

| case | sp@8 实测 | threshold | per-case verdict |
|---|---:|---:|:---:|
| heihe | 1.066× | ≥1.3× | FAIL |
| heihe_x4 | 1.729× | ≥1.5× | PASS |

**AND-gate eval**: heihe FAIL + heihe_x4 PASS → AND-gate **不满足** → D12.3 PR-N **不触发**。

## §3 实际触发路径

per `docs/p1e/p1e_2x2_verdict.md` §6.2 + tasks §4.6.2：

> 4.6.2 单 case 不达 threshold（另一 case 已达）：进 partial closure 决策点（用户决策 ship vs fallback；倾向 ship 当 heihe_x4 达 1.5× 时）

实际触发：**§4.6.2 partial-closure → SHIP** (user 决策)。详 `docs/p1e/p1e_2x2_verdict.md` §6.3 + §6.4 SHIP rationale + small-case carve-out。

## §4 PR-N 假设实施 scope (reference only, 不是本 epic deliverable)

若 D12.3 在 future epic 重新评估时被触发（例如 P2a 阶段引入新 case 后 BOTH FAIL）：

per `openspec/changes/p1e-strict-omp-rhs/tasks.md` §8.1：

- **element 块**：3×3 (surface / unsat / GW) 小块独立 setup + solve
- **river / lake 块**：1×1 标量
- 通过 `CVodeSetPreconditioner(cvode_mem, p_setup, p_solve)` 注册
- 验证 `grep -nE 'CVodeSetPreconditioner.*p_setup.*p_solve' SHUD/src/Equations/cvode_config.cpp` ≥1 命中

per §8.2-§8.4：

- 验证 `nli` 下降 ≥ 30% + `nfeLS` 下降 ≥ 30%（vs PR-I baseline cvode_stats.txt）
- 8-cell wall verification — heihe + heihe_x4 N=8 加速比闭合至各自 threshold
- 数据归档至本 doc (本 placeholder 升级为实际 PR-N 数据 doc)

PR-N 闭合后才允许恢复 PR-J/K/L/M (per tasks §4.6.1)。本 epic D12.3 未触发，PR-J/K/L/M 直接顺序推进。

## §5 forward considerations (forthcoming)

ADR-0002 Path 3 (SPGMR + block-Jacobi physics-based preconditioner) 在本 P1e epic 内属 "P2 optimization (paired with Path 1)" 标记，不是单独 fallback path。future epic 若评估 Path 3 实施时（与 Path 1 配合提速）：

- 触发条件不一定要 D12.3 AND-gate；P2a 阶段可独立评估 ROI
- ADR-0002 Decision Matrix 该行 status 可由 "P2 optimization (paired with Path 1)" 升至 "implemented (P2X epic close)"
- 本 doc 占位可被升级或被 P2X 阶段新建的实际 PR-N 实施 doc 替代

## §6 References

- spec `p1e-strict-omp-rhs` L368-L383 (P1e.8 fallback Requirement)
- tasks `p1e-strict-omp-rhs` §8 P1e.8 (PR-N-conditional)
- tasks `p1e-strict-omp-rhs` §7.10.2 (本 placeholder 创建 SHALL)
- ADR-0002 §"Candidate Paths" Path 3 (SPGMR + block-Jacobi physics-based preconditioner)
- `docs/p1e/p1e_pr_i_strict_omp_verification.md` §8.2 D12 routing decision (heihe_x4 PASS → D12.3 not triggered)
- `docs/p1e/p1e_2x2_verdict.md` §6.2 D12 routing decision (active path = §4.6.2 SHIP)
