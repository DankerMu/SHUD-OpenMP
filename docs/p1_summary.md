# P1 update-omp Baseline 完成

> P1 = master plan §3 定义的 "B1b + MD_update.cpp 三 owner loop 并行后的 first parallel candidate baseline"。本文档是 P1 epic capstone summary。

## TL;DR

- **状态**: P1 epic COMPLETE (14/14 PR + 1 tag + 1 baseline branch lock)
- **Tag**: `P1-update-omp-tag` = `003f58d` / SHUD pin `07c677f` / D11 immutable
- **Baseline**: `baseline/P1` D11 locked (lock_branch + enforce_admins)
- **强制门**: NUM_OPENMP=1 vs B1b bitwise **24/24 PASS** (3 anchor)
- **§1.1.1 verdict**: **WARNING** (不阻塞 P1 lock) — heihe sp@8=1.08×, heihe_x4 sp@8=1.14×; N≥4 strict-A3a 4/6 FAIL → P7 final-fusion debt
- **下游**: P2 from `main` 新分支；`baseline/P1` 仅作历史比对

---

## 1. 完成定义

P1 = 6 phase × 14 PR + 1 tag-lock:

| Phase | Scope | PR |
|---|---|---|
| A — forcing | M7 trim 工具 + 7 case manifest + bitwise vs B0 | PR-A #212, PR-B #213 |
| B — profile | trimmed forcing retest + Opt-IO 决策 (a) optional | PR-G #214 |
| C.audit | 5 update 函数 + f_updatei case 1-5 pre-audit (a) safe | PR-C #215 |
| C.implement | MD_update.cpp 三 owner loop pragma | PR-D #216 element / PR-E #217 river / PR-F #218 lake |
| C.verify | RHS snapshot + full-run + scaling | PR-H #219 / PR-I #220 / PR-J #221 / PR-K1 #222 / PR-K2 #223 |
| D — lock | tag + docs + PROMOTE | PR-L #224 / PR-M #225 / PR-N #226 |

完整时间线见下 §4。

## 2. 旧版错误复盘

**不适用** — P1 是 B1b 后首个 parallel candidate baseline，无回滚 / 旧版错误。

(B1a 曾因 S1d-end refactor force-update tag 一次至 `f7f992c`；此条以 D11 immutable 在 B1b 后所有 baseline 上 enforced，含本 P1。)

## 3. `P1-update-omp-tag` 的处理

| Field | Value |
|---|---|
| annotated tag object SHA | `ff21c75c8e968d5e47ca53b015425360be9ac879` |
| deref commit SHA | `003f58dc079116ef2161d2f96006228ef0e013d0` |
| SHUD pin | `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47` (openmp-baseline 分支) |
| Anchor 时刻 | PR-K2 #223 capstone log append 后 main HEAD |
| `baseline/P1` 分支 protection | `lock_branch=true` + `enforce_admins=true` + no force-push + no delete |

**D11 强制**: 一次锁死，禁止 force-update。后续 retroactive (e.g. P2 sub-stage stacking) 走 forward-compat **P1c-tag stacking / P2-* stacking** 路径 (master plan C8)。

`P1-update-omp-tag` 凝固在 PR-K2 #223 evidence 落地 + log append 之后；本 PR-M docs PROMOTE + PR-N spec PROMOTE 在 main 侧继续推进，**不进 tag 内部**。

## 4. 时间线 (14 PR)

| PR | Issue | Phase / Stage | Scope |
|---|---|---|---|
| A | #212 | A / I-A.1 | M7 forcing trim 工具 + Mac 4 case trim + 7-case manifest schema |
| B | #213 | A / I-A.2 | server heihe + heihe_x4 trim + Slurm verify |
| G | #214 | B / I-B.1 | trimmed forcing retest + Opt-IO (a) 退回可选 |
| C | #215 | C.audit / I-C.1 | 5 函数 + f_updatei case 1-5 pre-audit (zero src) |
| D | #216 | C.impl / I-C.2 | element loop pragma (SHUD 017c629→6a9e684) |
| E | #217 | C.impl / I-C.3 | river loop pragma (SHUD 6a9e684→08898a3) |
| F | #218 | C.impl / I-C.4 | lake loop pragma (SHUD 08898a3→07c677f) |
| H | #219 | C.verify / I-C.5 | Mac 4-case RHS snapshot 12/12 PASS |
| I | #220 | C.verify / I-C.6 | Mac 4-case full-run 8/8 + CVODE 4/4 PASS |
| J | #221 | C.verify / I-C.7 | server 2-case full-run 4/4 PASS |
| K1 | #222 | C.verify / I-C.8a | Mac 16-cell scaling 16/16 A3a (NG1 dev-only) |
| K2 | #223 | C.verify / I-C.8b | server 8-cell scaling, §1.1.1 WARNING |
| L | #224 | D / I-D.1 | P1-update-omp-tag + baseline/P1 lock (tag-only, no PR) |
| M | #225 | D / I-D.2 | p1_summary + status_matrix + build_manifest |
| N | #226 | D / I-D.3 | spec PROMOTE + archive + glossary + Epic #211 close |

## 5. Capstone 验证结果

### 5.1 P1 强制门 — NUM_OPENMP=1 vs B1b bitwise (24/24 PASS)

| Anchor | Source | Scope | Result |
|---|---|---|---|
| 1 | PR-H #219 | Mac 4-case RHS snapshot vs B1b canonical | **12 / 12 PASS** |
| 2 | PR-I #220 | Mac 4-case full-run canonical SHA + CVODE 15-key | **8 / 8 PASS** |
| 3 | PR-J #221 | server 2-case full-run canonical SHA | **4 / 4 PASS** |

详 `p1_rhs_snapshot_bitwise.md`、`p1_fullrun_bitwise.md`。

### 5.2 §1.1.1 wall + speedup (PR-K2 server)

| Case | NumEle | sp@2 | sp@4 | sp@8 | P7 strict T | P9 final T | verdict |
|---|---:|---:|---:|---:|---|---|---|
| heihe | 6335 | 0.96× | 1.05× | **1.08×** | "不独立验收" (master plan §1.1.1 + §5) | 4.5× | within Amdahl-bound 1.13× ✓ |
| heihe_x4 | 40046 | 1.01× | 1.08× | **1.14×** | 2.2× | 6.0× | < P7 strict; P1 起点报告 |

heihe Medium 因 trim 后 IO 占比 1.90% 仍触发 master plan §5 "不独立验收" carve-out。heihe_x4 Large 1.14× 是 P1 first OMP candidate 起点（no S5d SoA / no owner-local gather parallel / no OMP_CUTOFF / no N_Vector parallel）。

### 5.3 A3a / A3b strict (PR-K2 vs N=1 same-binary baseline)

| Case | N=2 A3a | N=4 A3a/A3b | N=8 A3a/A3b |
|---|---|---|---|
| heihe | **PASS bitwise** | FAIL / FAIL | FAIL / FAIL |
| heihe_x4 | **PASS bitwise** | FAIL / FAIL | FAIL / FAIL |

N=2 bitwise PASS = PR-D/E/F owner-local design 强证据。N≥4 dual-FAIL via CVODE nst bifurcation：heihe nst N=1/2/4/8 = 6773 / 6773 / **6585** / **6684**；P7 final-fusion debt（见 §7）。Mac 16-cell A3a 全 PASS（NG1 dev-only, 不计入 §1.1.1 — 4 case × 4 N × all bitwise via snapshot probe）。

详 `p1_perf_baseline.md`。

### 5.4 §1.1.1 verdict

**WARNING — P1 epic 不阻塞** per:
- design D5 NG3
- master plan §1.1.2 (P1/P7 退出门分离)
- spec `p1-state-update-parallel` L164-L181 (A3b-fallback + WARNING allowed at P1 standalone)
- spec L205-L209 (PR-N PROMOTE 升级覆盖 dual-FAIL at N≥4 fallback bucket)

## 6. Hand-off → P2a / P7

P2 sub-stages from `main`（master plan §3）:

| Stage | Scope |
|---|---|
| P2a | `f_updatei` case 1-5 OMP（spec NG7 reserved） |
| P2b | `MD_f.cpp` rhs_flux compute + gather 并行 |
| P3+ | owner-local gather parallel + OMP_CUTOFF + N_Vector parallel |
| P7 | final-fusion deterministic reduction（见 §7 debt 修复） |
| P9 | production deterministic N_Vector + §1.1.1 P9 final 全数达成 |

**D9 fast-path 触发条件**: 若 P2a/P2b 全 (a) safe + bitwise vs B1-tag PASS，走 forward-compat P1c-tag stacking 在 `P1-update-omp-tag` 之上 stack，不新建 P-strict-tag。判定时机在 P2 capstone PR。

## 7. Forward debts (PR-N + P2+ inherit)

| # | Debt | Source | Disposition |
|---|---|---|---|
| 1 | spec L184 ambiguous dual-FAIL handling | F-K2-1 reviewer K2 finding | **FIXED** in PR-N spec PROMOTE L205-L209 |
| 2 | p1-capstone spec 7-section schema vs deeper structure | F-M-2 reviewer M finding | **FIXED** in PR-N spec PROMOTE L58/L69 "≥7 topics" |
| 3 | trajectory-bifurcation root-cause framing 当前 hypothesis-level | F-K2-2 reviewer K2 finding | P2+ 精确化：bisect tree-reduction-depth vs FMA vs scheduler-locale；产 `docs/p1_a3a_root_cause.md` |
| 4 | P7 final-fusion deterministic-reduction | §5.3 根因 | P7 capstone 引入 fixed-shape pairwise canonical reduction 或 P9 deterministic N_Vector；目标 A3a strict N∈{2,4,8} 全 PASS |
| 5 | `_before_passvalue` mid-pipeline drift 3 case (xj_up/qinyijiang/qhh) | PR-H diagnostic addendum | P2a / PR-N follow-up issue; 不阻塞 P1（canonical 12-cell + 下游全 PASS） |

## 8. B-chain immutable (D11 enforced)

```
B0-tag    (884cfb13 / SHUD 78c37a1, 2026-06-17)
B1a-tag   (f7f992c  / SHUD 0b3998d, 2026-06-21, force-updated once from S1d-end)
B1b-tag   (18a0c908 / SHUD 71b3a1ae, 2026-06-22 PR-16 #207)
B1-tag    (ed054b4  / SHUD 017c629, 2026-06-22 PR-19 #210 D9 fast-path #2)
P1-update-omp-tag (003f58d / SHUD 07c677f, 2026-06-22 PR-L #224)
  ↑ baseline/P1 D11 locked
```

(commit SHA shown; annotated tag object SHA 见 §3。)

## 9. 验证 P1-update-omp-tag

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

## 详细 evidence 索引

- `docs/p1_audit_update_funcs.md` — PR-C audit
- `docs/p1_rhs_snapshot_bitwise.md` — PR-H snapshot 12/12 + diagnostic addendum
- `docs/p1_fullrun_bitwise.md` — PR-I Mac + PR-J server canonical SHA
- `docs/p1_perf_baseline.md` — PR-K1 Mac scaling + PR-K2 server scaling
- `docs/build_manifest.md` §"P1-update-omp-tag" — build provenance
- `docs/status_matrix.md` 第 P1 行 — single-line summary
- `openspec/specs/p1-state-update-parallel/spec.md` + `openspec/specs/p1-capstone/spec.md` — post-PROMOTE canonical spec
- `openspec/glossary.md` §"P1 first-parallel-candidate baseline 集合" — 7 new terms
