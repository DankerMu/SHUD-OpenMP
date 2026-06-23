# P1 update-omp 基线总结

## 背景与定义

P1 基线 (P1 baseline) 指 master plan §3 所定义的 "B1b 之上、`MD_update.cpp` 三处所有者循环 (owner loop) 完成 OpenMP 并行化后" 的首个候选并行基线 (first parallel candidate baseline)。其工作阶段涵盖 master plan §5 中的 P1.A–P1.D 四个子阶段，并以六个 phase × 十四个 PR + 一次 tag 锁定收束。本文档为 P1 epic 的封顶 (capstone) 总结。

P1 工作于 2026-06-22 完成，并以 annotated git tag `P1-update-omp-tag` 锁定；基线分支 `baseline/P1` 同步开启 D11 不可变性保护。后续 P2 及其子阶段将从 `main` 重新分支，不再回写 P1。

## 完成状态一览

| 项目 | 内容 |
|---|---|
| Epic 状态 | P1 epic COMPLETE (14/14 PR + 1 tag + 1 baseline 分支锁定) |
| Annotated tag | `P1-update-omp-tag` = commit `003f58d` / SHUD pin `07c677f`，D11 immutable |
| 基线分支 | `baseline/P1` 已锁定 (`lock_branch=true` + `enforce_admins=true`) |
| 强制验收门 (forced acceptance gate) | `NUM_OPENMP=1` vs B1b bitwise 24 / 24 PASS (3 个 anchor) |
| §1.1.1 verdict | **WARNING** (不阻塞 P1 锁定)：heihe sp@8 = 1.08×，heihe_x4 sp@8 = 1.14×；N ≥ 4 strict-A3a 4 / 6 FAIL，转入 P7 final-fusion 遗留债务 |
| 后续阶段 | P2 从 `main` 新建分支；`baseline/P1` 仅作历史比对参照物 |

---

## 1. 完成定义

P1 epic 由 6 个 phase 共 14 个 PR 与一次 tag 锁定组成。各 phase 的范围与对应 PR 列于下表。

| Phase | 范围 | PR |
|---|---|---|
| A — forcing | M7 trim 工具 + 7 个案例 manifest + bitwise vs B0 | PR-A #212, PR-B #213 |
| B — profile | trimmed forcing 重测 + Opt-IO 决策 (a) 可选化 | PR-G #214 |
| C.audit | 5 个 update 函数 + `f_updatei` case 1–5 预审计 (a) safe | PR-C #215 |
| C.implement | `MD_update.cpp` 三处 owner loop 添加 pragma | PR-D #216 element / PR-E #217 river / PR-F #218 lake |
| C.verify | RHS snapshot + 全流程 (full-run) + 标度 (scaling) | PR-H #219 / PR-I #220 / PR-J #221 / PR-K1 #222 / PR-K2 #223 |
| D — lock | tag + docs + PROMOTE | PR-L #224 / PR-M #225 / PR-N #226 |

完整时间线参见下文 §4。

## 2. 旧版错误复盘

不适用。P1 是 B1b 之后首个 parallel candidate baseline，无历史回滚或旧版错误。

附注：B1a 曾因 S1d-end refactor 一次性 force-update tag 至 `f7f992c`；该事件以 D11 immutability 约束在 B1b 之后所有 baseline 上强制生效，本 P1 亦受其约束。

## 3. `P1-update-omp-tag` 的处理

锁定字段见下表。

| Field | Value |
|---|---|
| annotated tag object SHA | `ff21c75c8e968d5e47ca53b015425360be9ac879` |
| deref commit SHA | `003f58dc079116ef2161d2f96006228ef0e013d0` |
| SHUD pin | `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47` (openmp-baseline 分支) |
| Anchor 时刻 | PR-K2 #223 capstone log append 后 main HEAD |
| `baseline/P1` 分支 protection | `lock_branch=true` + `enforce_admins=true` + no force-push + no delete |

D11 强制：tag 一次锁死，禁止 force-update。后续追溯式变更 (例如 P2 子阶段堆叠) 采用前向兼容的 `P1c-tag` stacking 或 `P2-*` stacking 路径，参 master plan C8。

`P1-update-omp-tag` 凝固于 PR-K2 #223 evidence 落地与 log append 之后；其后 PR-M 的 docs PROMOTE 与 PR-N 的 spec PROMOTE 在 `main` 侧继续推进，但不进入 tag 内部。

## 4. 时间线 (14 PR)

| PR | Issue | Phase / Stage | Scope |
|---|---|---|---|
| A | #212 | A / I-A.1 | M7 forcing trim 工具 + Mac 端 4 案例 trim + 7 案例 manifest schema |
| B | #213 | A / I-A.2 | 服务器 heihe + heihe_x4 trim + Slurm 验证 |
| G | #214 | B / I-B.1 | trimmed forcing 重测 + Opt-IO (a) 退回可选 |
| C | #215 | C.audit / I-C.1 | 5 函数 + `f_updatei` case 1–5 预审计 (零源码改动) |
| D | #216 | C.impl / I-C.2 | element loop pragma (SHUD 017c629 → 6a9e684) |
| E | #217 | C.impl / I-C.3 | river loop pragma (SHUD 6a9e684 → 08898a3) |
| F | #218 | C.impl / I-C.4 | lake loop pragma (SHUD 08898a3 → 07c677f) |
| H | #219 | C.verify / I-C.5 | Mac 4 案例 RHS snapshot 12 / 12 PASS |
| I | #220 | C.verify / I-C.6 | Mac 4 案例 full-run 8 / 8 + CVODE 4 / 4 PASS |
| J | #221 | C.verify / I-C.7 | 服务器 2 案例 full-run 4 / 4 PASS |
| K1 | #222 | C.verify / I-C.8a | Mac 16-cell scaling 16 / 16 A3a (NG1 dev-only) |
| K2 | #223 | C.verify / I-C.8b | 服务器 8-cell scaling，§1.1.1 WARNING |
| L | #224 | D / I-D.1 | `P1-update-omp-tag` + `baseline/P1` 锁定 (tag-only，无 PR) |
| M | #225 | D / I-D.2 | `p1_summary` + `status_matrix` + `build_manifest` |
| N | #226 | D / I-D.3 | spec PROMOTE + archive + glossary + Epic #211 关闭 |

## 5. Capstone 验证结果

### 5.1 P1 强制验收门 — `NUM_OPENMP=1` vs B1b bitwise (24 / 24 PASS)

| Anchor | Source | Scope | Result |
|---|---|---|---|
| 1 | PR-H #219 | Mac 4 案例 RHS snapshot vs B1b canonical | **12 / 12 PASS** |
| 2 | PR-I #220 | Mac 4 案例 full-run canonical SHA + CVODE 15-key | **8 / 8 PASS** |
| 3 | PR-J #221 | 服务器 2 案例 full-run canonical SHA | **4 / 4 PASS** |

详细证据见 `p1_rhs_snapshot_bitwise.md`、`p1_fullrun_bitwise.md`。

### 5.2 §1.1.1 wall-time 与 speedup (PR-K2 服务器端)

| Case | NumEle | sp@2 | sp@4 | sp@8 | P7 strict T | P9 final T | verdict |
|---|---:|---:|---:|---:|---|---|---|
| heihe | 6335 | 0.96× | 1.05× | **1.08×** | "不独立验收" (master plan §1.1.1 + §5) | 4.5× | within Amdahl-bound 1.13× ✓ |
| heihe_x4 | 40046 | 1.01× | 1.08× | **1.14×** | 2.2× | 6.0× | < P7 strict；P1 起点报告 |

heihe Medium 案例由于 trim 后 IO 占比仅 1.90%，仍触发 master plan §5 中的 "不独立验收" carve-out 条款。heihe_x4 Large 案例 1.14× 是 P1 首个 OMP 候选基线的起点 (尚未引入 S5d SoA、owner-local gather 并行、`OMP_CUTOFF`、N_Vector 并行)。

### 5.3 A3a / A3b strict (PR-K2 相对 N = 1 同二进制基线)

| Case | N = 2 A3a | N = 4 A3a / A3b | N = 8 A3a / A3b |
|---|---|---|---|
| heihe | **PASS bitwise** | FAIL / FAIL | FAIL / FAIL |
| heihe_x4 | **PASS bitwise** | FAIL / FAIL | FAIL / FAIL |

N = 2 bitwise PASS 构成 PR-D / E / F owner-local 设计的强证据。N ≥ 4 处的 dual-FAIL 通过 CVODE nst 轨迹分叉 (bifurcation) 体现：heihe nst 在 N = 1 / 2 / 4 / 8 下分别为 6773 / 6773 / **6585** / **6684**，归类为 P7 final-fusion 遗留债务 (见 §7)。Mac 16-cell A3a 全部 PASS (NG1 dev-only，不计入 §1.1.1 — 4 案例 × 4 N，全部经 snapshot probe 实现 bitwise)。

详细数据见 `p1_perf_baseline.md`。

### 5.4 §1.1.1 verdict

**WARNING — 不阻塞 P1 epic**。依据如下：

- design D5 NG3；
- master plan §1.1.2 (P1 / P7 退出门分离)；
- spec `p1-state-update-parallel` L164–L181 (A3b-fallback + WARNING allowed at P1 standalone)；
- spec L205–L209 (PR-N PROMOTE 升级覆盖 N ≥ 4 fallback bucket 处的 dual-FAIL)。

## 6. 移交至 P2a / P7

P2 及其子阶段均自 `main` 分支起步 (master plan §3)。

| Stage | Scope |
|---|---|
| P2a | `f_updatei` case 1–5 OMP (spec NG7 保留) |
| P2b | `MD_f.cpp` 中 rhs_flux 计算与 gather 并行 |
| P3+ | owner-local gather 并行 + `OMP_CUTOFF` + N_Vector 并行 |
| P7 | final-fusion deterministic reduction (修复 §7 所列债务) |
| P9 | production deterministic N_Vector + §1.1.1 P9 final 全部达成 |

D9 fast-path 触发条件：若 P2a 与 P2b 全部判定为 (a) safe 且 bitwise vs B1-tag PASS，则采纳前向兼容的 P1c-tag stacking 方案，将其堆叠于 `P1-update-omp-tag` 之上，不再新建 P-strict-tag。判定时机为 P2 capstone PR。

## 7. 遗留事项 (PR-N 与 P2+ 继承)

| # | Debt | Source | Disposition |
|---|---|---|---|
| 1 | spec L184 对 dual-FAIL 处理表述歧义 | F-K2-1 reviewer K2 finding | **FIXED**：PR-N spec PROMOTE L205–L209 |
| 2 | p1-capstone spec 7-section schema 与更深结构不符 | F-M-2 reviewer M finding | **FIXED**：PR-N spec PROMOTE L58 / L69 改为 "≥ 7 topics" |
| 3 | 轨迹分叉根因表述仍停留在 hypothesis 级 | F-K2-2 reviewer K2 finding | P2+ 阶段精确化：对 tree-reduction-depth、FMA、scheduler-locale 做 bisect 定位，产出 `docs/p1_a3a_root_cause.md` |
| 4 | P7 final-fusion deterministic-reduction | §5.3 根因 | P7 capstone 引入 fixed-shape pairwise canonical reduction，或在 P9 阶段引入 deterministic N_Vector；目标使 A3a strict 在 N ∈ {2, 4, 8} 处全部 PASS |
| 5 | `_before_passvalue` 中段管道漂移，涉及 3 案例 (xj_up / qinyijiang / qhh) | PR-H diagnostic addendum | 转 P2a 或 PR-N follow-up issue；不阻塞 P1 (canonical 12-cell 与下游全部 PASS) |

## 8. B-chain 不可变链 (D11 enforced)

```
B0-tag    (884cfb13 / SHUD 78c37a1, 2026-06-17)
B1a-tag   (f7f992c  / SHUD 0b3998d, 2026-06-21, 由 S1d-end 一次性 force-update)
B1b-tag   (18a0c908 / SHUD 71b3a1ae, 2026-06-22 PR-16 #207)
B1-tag    (ed054b4  / SHUD 017c629, 2026-06-22 PR-19 #210 D9 fast-path #2)
P1-update-omp-tag (003f58d / SHUD 07c677f, 2026-06-22 PR-L #224)
  ↑ baseline/P1 D11 locked
```

(上表所列为 commit SHA；annotated tag object SHA 见 §3。)

## 9. `P1-update-omp-tag` 验证

```bash
git ls-remote --tags origin | grep P1-update-omp-tag
# refs/tags/P1-update-omp-tag       ff21c75c…  ← annotated tag object
# refs/tags/P1-update-omp-tag^{}    003f58d…   ← deref commit

git show P1-update-omp-tag --no-patch --format=fuller
# Tagger: DankerMu <mumzy@mail.ustc.edu.cn>
# 7-bullet P1 fix list / SHUD pin 07c677f / 6-case canonical SHA / scaling 概要

gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection \
  --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'
# {"allow_deletions":false, "allow_force_pushes":false, "enforce_admins":true, "lock_branch":true}
```

## 详细证据索引

- `docs/p1_audit_update_funcs.md` — PR-C 预审计
- `docs/p1_rhs_snapshot_bitwise.md` — PR-H snapshot 12 / 12 + diagnostic addendum
- `docs/p1_fullrun_bitwise.md` — PR-I Mac + PR-J 服务器 canonical SHA
- `docs/p1_perf_baseline.md` — PR-K1 Mac scaling + PR-K2 服务器 scaling
- `docs/build_manifest.md` §"P1-update-omp-tag" — build provenance
- `docs/status_matrix.md` 第 P1 行 — 单行汇总
- `openspec/specs/p1-state-update-parallel/spec.md` + `openspec/specs/p1-capstone/spec.md` — PROMOTE 之后的 canonical spec
- `openspec/glossary.md` §"P1 first-parallel-candidate baseline 集合" — 新增 7 个术语
