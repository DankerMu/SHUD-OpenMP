## Purpose

规约 trimmed forcing 上重测 profile_B0（heihe + heihe_x4）+ Opt-IO 硬性前置判断更新（基于 trimmed `t_forcing_io / t_total` 实测占比 vs 50% 阈值）+ status_matrix Opt-IO 行同步 + profile_B0 yaml 命名约定（trimmed/non-trimmed 并存）。本 capability 是 P1 epic 的 Phase B（PR-G #214），决定 §1.1.1 加速比验收中 heihe 是否仍为 Opt-IO 硬性前置 case。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/p1-update-omp/specs/<capability>/spec.md PROMOTE 而来（#226 P1 capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。
- Opt-IO 决策二选一 sign-off 模式：(a) 退回可选 / (b) 仍硬性前置；与 S0.12 sign-off 字段对齐 (`signed_at` + `signer`)。
- profile_B0 yaml 文件 trimmed/non-trimmed 严禁覆盖 — 命名后缀强制 `.trimmed.yaml` 以便历史比对。
- Slurm 三铁律强制执行（详 CLAUDE.md §"Slurm 三铁律"）。

## Requirements

### Requirement: trimmed forcing 上重测 profile_B0 (heihe + heihe_x4)

server cn0X (Slurm 三铁律 enforced) SHALL 用 trimmed forcing 在 90 天 cfg 截断下重测 heihe + heihe_x4 的 `profile_B0` 测量，产出 `benchmarks/heihe/profile_B0.target.trimmed.yaml` + `benchmarks/heihe_x4/profile_B0.target.trimmed.yaml`，包含 S5c 7-bucket timer（`t_forcing_io / t_RHS_total / t_CVODE_internal / t_ET / t_init / t_other / t_total`）+ 3-run identical 校验。

#### Scenario: heihe trimmed profile 测量

- **WHEN** server cn0X 跑 heihe trimmed forcing 90 天 NUM_OPENMP=1，3-run
- **THEN** `benchmarks/heihe/profile_B0.target.trimmed.yaml` 产出，含 7-bucket 数据 + 3-run identical SHA256

#### Scenario: heihe_x4 trimmed profile 测量

- **WHEN** server cn0X 跑 heihe_x4 trimmed forcing 90 天 NUM_OPENMP=1，3-run
- **THEN** `benchmarks/heihe_x4/profile_B0.target.trimmed.yaml` 产出，含 7-bucket 数据 + 3-run identical SHA256

---

### Requirement: Opt-IO 硬性前置判断更新

`docs/profile_decision.md` SHALL 在 trimmed profile 实测后更新 "Opt-IO 硬性前置" 章节，写明 trim 后 heihe 的 `t_forcing_io / t_total` 实测值 + 是否仍触发 §5 L1533 的 `> 50%` 硬性前置阈值。判断结论 SHALL 二选一：

- **(a) 退回可选**：trim 后 heihe `t_forcing_io / t_total < 50%` → Opt-IO 回到 §5 原 "B1b 锁定后任意时间执行" 可选定位；本 change 不阻塞 P1；下游 P-strict 全部完成后再评估
- **(b) 仍硬性前置**：trim 后 heihe `t_forcing_io / t_total >= 50%` → Opt-IO 仍是 §1.1.1 验收的 heihe 硬性前置；需单独立 change（不在本 p1-update-omp 范围）；heihe case 在 Opt-IO 完成前不进入 §1.1.1 加速比统计

判断结论 SHALL 在 `docs/profile_decision.md` 添加 sign-off 字段（`signed_at` + `signer`），与既有 S0.12 sign-off 模式一致。

#### Scenario: 退回可选路径 sign-off

- **WHEN** heihe trimmed `t_forcing_io / t_total < 50%`
- **THEN** `docs/profile_decision.md` "Opt-IO" 章节写入 "(a) 退回可选"，含 trimmed 实测占比 + signed_at 字段 + signer 字段

#### Scenario: 仍硬性前置路径 sign-off

- **WHEN** heihe trimmed `t_forcing_io / t_total >= 50%`
- **THEN** `docs/profile_decision.md` "Opt-IO" 章节写入 "(b) 仍硬性前置"，含 trimmed 实测占比 + signed_at + signer + 后续 change name reservation（如 "see future change: opt-io-forcing-cache"）

---

### Requirement: status_matrix Opt-IO 行同步

`docs/status_matrix.md` Opt-IO 行 SHALL 根据 profile_decision.md 决策结论同步：

- **(a) 退回可选** → Opt-IO 行状态变更：从 "PENDING (B1b 锁定后任意时间执行；heihe 硬性前置)" 改为 "PENDING (可选；M7 trim 后 heihe 退回可选 per PR-G profile retest)"
- **(b) 仍硬性前置** → Opt-IO 行状态保持 "PENDING (heihe 硬性前置)"，但 heihe 行加 caveat "Opt-IO 完成前不进入 §1.1.1 加速比统计"

#### Scenario: 退回可选时 status_matrix 更新

- **WHEN** profile_decision.md 决策 = (a) 退回可选
- **THEN** status_matrix Opt-IO 行更新；引用 PR-G + profile retest yaml SHA

#### Scenario: 仍硬性前置时 status_matrix 不变

- **WHEN** profile_decision.md 决策 = (b) 仍硬性前置
- **THEN** status_matrix Opt-IO 行保持，heihe 行加 footnote

---

### Requirement: profile_B0 yaml 文件命名约定

trimmed profile 产物 SHALL 命名 `profile_B0.target.trimmed.yaml`（与既有 `profile_B0.target.yaml` 并存，不覆盖既有非 trimmed 测量供历史比对）。`docs/profile_decision.md` 引用时 SHALL 明确写 "trimmed" 或 "non-trimmed" 上下文，不混用。

#### Scenario: trimmed yaml 与 non-trimmed yaml 并存

- **WHEN** `ls benchmarks/heihe/profile_B0.*.yaml`
- **THEN** 至少含 `profile_B0.target.yaml`（非 trimmed）+ `profile_B0.target.trimmed.yaml`（trimmed）两文件
