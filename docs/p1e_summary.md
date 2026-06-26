# P1e strict-omp 基线总结

## 背景与定义

P1e 基线 (P1e baseline) 指 master plan §3 / §6 P1e.1-7 所定义的 "P1d 之上、`ExecPolicy::StrictOMP` 完成 RHS 并行化后" 的 production-candidate 并行基线。其工作范围由 ADR-0002 Path 1 (Serial NVec + StrictOMP RHS) 锁定，涵盖 6 phase × 14 PR + 一次 tag 锁定 + 一次 OpenSpec PROMOTE 收束。本文件是 P1e epic (#308) 的封顶 (capstone) 总结。

P1e 工作于 2026-06-24 → 2026-06-25 两天内完成，并以 annotated git tag `P1e-tag` 锁定；基线分支 `baseline/P1e` 由 PR-L 同步开启 D11 不可变性保护。后续 P2a 及其子阶段将从 `main` 重新分支，不再回写 P1e。

## 完成状态一览

| 项目 | 内容 |
|---|---|
| Epic 状态 | P1e epic CLOSED (14/14 PR + 1 tag + 1 baseline 分支锁定 + 2 OpenSpec PROMOTE) |
| Annotated tag | `P1e-tag` = annotated object `25023eff` / deref commit `11687b75` / SHUD pin `3341368d`，D11 immutable |
| 基线分支 | `baseline/P1e` 已锁定 (`lock_branch=true` + `enforce_admins=true` + no force-push + no delete) |
| 强制验收门 (3 SHALL gate) | AC-S1 mode C 跨 N bitwise PASS + AC-S2 mode C SHA == mode A reference SHA PASS + AC-S3 D7 速度比 AND-gate PARTIAL |
| §1.1.1 verdict | **SHIP via §4.6.2 partial-closure** (heihe sp@8 1.066× < 1.3× FAIL, heihe_x4 sp@8 1.729× ≥ 1.5× PASS; AND-gate BOTH FAIL 才触发 D12.3 → 不满足；user 决策 SHIP) |
| 后续阶段 | P2a 从 `main` 新建分支；`baseline/P1e` 仅作历史比对参照物 (vs P1d / vs P-strict) |

---

## 1. 完成定义

P1e epic 由 6 个 phase 共 14 个 PR 与一次 tag 锁定组成。各 phase 的范围与对应 PR 列于下表。

| Phase | 范围 | PR |
|---|---|---|
| A — intake + audit | rivqdown.dat cache 审计 + 8 doc 初稿 + `recompute_for_output` helper | PR-A #309, PR-B0 #311 |
| B — 2×2 driver | 2×2 build matrix runner + CV_Y hash 工具 + manifest yaml | PR-B #310 |
| C — Phase 1 因果实验 | Mac + server mode A/B 192 cell + verdict aggregation + D12 routing placeholder | PR-C #312 / PR-D #313 / PR-E #314 |
| D — StrictOMP 实施 | `ExecPolicy::StrictOMP` impl + `-fopenmp` auto-wire + `SHUD_RHS_THREADS` env + first-touch removal + omp single→omp for | PR-F #315 / PR-G #315 / PR-H #316 |
| E — Phase 2 SHALL closure | server 24-cell SHALL 三门 + Mac 4-case N=1 reverse-compat | PR-I #317 / PR-J #318+#333 |
| F — capstone + tag + PROMOTE | 17-doc capstone + `P1e-tag` annotated + `baseline/P1e` lock + 2 spec PROMOTE + 4 glossary terms + Epic close | PR-K #319+#334 / PR-L #320+#335 / PR-M #321+#336 |

完整时间线参见下文 §4。

## 2. 旧版错误复盘

不适用。P1e 是 P1d epic E′ containment closure 之后 ADR-0002 Path 1 (Serial NVec + StrictOMP RHS) 首次实施的新 epic，无 P1e 历史回滚或旧版错误。

附注：P1d epic 完成时遗留 forward debt "真正应并行的 RHS 还没并行" (master plan v1.5 / M10 §6 P1d.7)。本 epic 通过 P1e.6 §"P1d carve-out closure" 关闭该 debt — 详 `docs/p1e/p1e_summary.md` §6。

## 3. `P1e-tag` 的处理

锁定字段见下表。

| Field | Value |
|---|---|
| annotated tag object SHA | `25023eff32d1fa317b045cbc786f379fac9e522c` |
| deref commit SHA | `11687b756dd53bb634df391bcbeb64b3cef5c750` |
| SHUD pin | `3341368d2d0854924d2286925c8575df52cc97a0` (openmp-baseline 分支) |
| Anchor 时刻 | PR-K #334 capstone post-merge review-loop-log append 后 `baseline/P1e` HEAD |
| `baseline/P1e` 分支 protection | `lock_branch=true` + `enforce_admins=true` + no force-push + no delete |

D11 强制：tag 一次锁死，禁止 force-update。tag 创建于 PR-L #335；annotated message 引用 `docs/p1e/p1e_summary.md` + `docs/p1e/p1e_perf_baseline.md`，body 含 14-PR cross-ref 与 3 SHALL gate verdict 与 4-mode build matrix 与 §4.6.2 partial-closure SHIP narrative。tag message 内 PR-L / PR-M 占位 `<TBD>` 不通过 retagging 修正，PR# 映射 (PR-L #335 / PR-M #336) 记录在 `docs/p1e/p1e_summary.md` §10 R2 F-R2-1 forward note。

`P1e-tag` 凝固于 PR-K capstone log append 后；其后 PR-M 的 docs PROMOTE 与 spec PROMOTE 在 `baseline/P1e` 侧继续推进，但不进入 tag 内部 (per P1d 模式避免 amend 循环)。

## 4. 时间线 (14 PR)

| PR | Issue | Phase / Stage | Scope |
|---|---|---|---|
| A | #309 | A | rivqdown.dat cache 审计 + 8 doc 初稿 |
| B | #310 | B | 2×2 build matrix runner + CV_Y hash 工具 + manifest yaml |
| B0 | #311 | A | rivqdown.dat tout-boundary recompute via `Model_Data::recompute_for_output(N_Vector, double)` helper |
| C | #312 | C / Phase 1 | Mac 2×2 mode A/B 48 cell raw evidence + 3 SHALL gate verdict |
| D | #313 | C / Phase 1 | server 2×2 mode A/B 48 cell raw evidence + 3 SHALL gate verdict |
| E | #314 | C / Phase 1 | Phase 1 verdict aggregation + D12 routing placeholder |
| F | #315 | D / impl | SHUD `ExecPolicy::StrictOMP` impl 单 `#pragma omp parallel` region + omp single scaffolding |
| G | #315 | D / impl | SHUD Makefile `-fopenmp` 自动 wire + `SHUD_RHS_THREADS` env split + shud.cpp 两段 guard |
| H | #316 | D / impl | 删 `MD_rhs_core.cpp` L62-95 / L169-203 / L324-354 三处 steady-state first-touch loops + omp single → omp for schedule(static) |
| I | #317 | E / SHALL | server SHALL 24 cell (heihe + heihe_x4 × 4 N × 3 reps) + 3 SHALL gate verdict + D12 routing |
| J | #318/#333 | E / SHALL | Mac 4-case × N=1 reverse-compat 12 cell + spec-canonical 4-case 闭合 (#333 Phase 6 fix 补 xinanjiang + qinyijiang) |
| K | #319/#334 | F / capstone | docs/p1e/ 17 doc capstone + spec L192 amend + ADR-0002 close-out + review-loop-log PR-K 追加 |
| L | #320/#335 | F / lock | `P1e-tag` annotated 创建 + `baseline/P1e` lock_branch=true + D11 7-tag chain verify |
| M | #321/#336 | F / PROMOTE | OpenSpec 2 spec PROMOTE + glossary 4 新术语 + jsonl 双追加 + Epic #308 close + status_matrix SHIP-LOCKED |

## 5. Capstone 验证结果

### 5.1 P1e 3 SHALL gate — AC-S1 + AC-S2 mode C bitwise (24 cell PASS)

| Gate | Source | Scope | Result |
|---|---|---|---|
| AC-S1 cross-N | PR-I #317 | heihe + heihe_x4 × N∈{1,2,4,8} × 3 reps, 2 case 各 unique SHA = 1 | **24/24 PASS** |
| AC-S2 mode C == mode A | PR-I #317 | heihe `a2023ccd2de4` == PR-D ref; heihe_x4 `b5e4b0a2cf83` == PR-D ref | **2/2 PASS** |
| 6-case cross-platform SHA | PR-J #333 | 4 Mac case (libomp) + 2 server case (libgomp transitive cite from PR-I) | **6/6 PASS** |

详细证据见 `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3 + `docs/p1e/p1e_mac_reverse_compat.md` §3。

### 5.2 AC-S3 D7 速度比 AND-gate (PR-I 服务器端 median of 3 reps)

| Case | NumEle | N=1 (s) | N=8 (s) | sp@8 | threshold | per-case verdict |
|---|---:|---:|---:|---:|---:|:---:|
| heihe | 6335 | 504 | 473 | **1.066×** | ≥1.3× | FAIL |
| heihe_x4 | 40046 | 1340 | 775 | **1.729×** | ≥1.5× | PASS |

AND-gate semantics (per tasks §4.6 + design D12)：`BOTH FAIL` 才触发 D12.3 block-Jacobi fallback；single case FAIL 走 §4.6.2 partial-closure 决策点。heihe small-case 1.066× 不达 1.3× threshold 不是 implementation bug，是 **OpenMP runtime 固定开销 + cache locality 反转 + NUMA migration 在 6335 cells 规模下的物理 limit**（per `docs/p1e/p1e_perf_baseline.md` §6 v0.2 修正，原 "fork-join event 5e8" 量级写错，实际单 parallel region per RHS，fork-join 仅 ~2e4 量级）；heihe_x4 production-target mesh (NumEle=40046) 1.729× ≥ 1.5× 是真正的 ROI 收益 site。

nst Δ=0 strict ladder：heihe + heihe_x4 各 4 N nst = case-fixed (Δ=0) → `p1d-numa-governance` nst ladder 闭合 (P1d mode B era 跨 N drift → P1e mode C 闭合)。

### 5.3 Mac N=1 reverse-compat (PR-J 4-case × N=1 × 3 reps)

| Case | NumEle | mode C N=1 SHA12 | mode A ref SHA12 | match |
|---|---:|---|---|:---:|
| keliya | 484 | `b769e3270e1c` | `b769e3270e1c` | PASS |
| xinanjiang_upstream | 801 | `81fe3a02e17e` | `81fe3a02e17e` | PASS |
| qinyijiang | 3155 | `fc1b1816cf0d` | `fc1b1816cf0d` | PASS |
| qhh | 4773 | `ccc7dd09d018` | `ccc7dd09d018` | PASS |

4/4 PASS。同一 `ExecPolicy::StrictOMP` 源码在 macOS libomp 22.1.7 (Apple M4 Pro arm64) + Linux libgomp gcc 13.3 (Intel Xeon x86_64) 双 runtime + 双 ISA 下产出位级完全相同 — 验证 design D2 owner-local gather + reduction 是 deterministic-by-construction 并行策略 (hot path 内无 `omp reduction(+:)` double)。

详细数据见 `docs/p1e/p1e_perf_baseline.md`。

### 5.4 §1.1.1 verdict — SHIP via §4.6.2 partial-closure

**SHIP via §4.6.2 partial-closure** — 不阻塞 P1e epic。依据如下：

- AC-S1 + AC-S2 全 PASS (bitwise + cross-platform determinism 闭合)；
- AC-S3 AND-gate (BOTH FAIL 才触发 D12.3) 不满足，single case FAIL 走 §4.6.2 partial-closure；
- heihe_x4 production-target mesh 1.729× ≥ 1.5× 满足 ROI threshold；
- heihe 1.066× 小 case 不达 1.3× per design D7 asymmetric thresholds + OMP overhead floor 设计预期 carve-out；
- D12 4 branch eval = D12.1 / .2 / .3 / .4 全 NOT triggered (placeholder `docs/p1e/p1e_pr_n_block_jacobi.md` 写出留 non-trigger 占位)；
- user 决策 SHIP per `docs/p1e/p1e_2x2_verdict.md` §6.3 + §6.4 SHIP rationale；
- ADR-0002 Status `Accepted (2026-06-24)` → `Implemented (P1e epic close, 2026-06-25)`。

## 6. 移交至 P2a / P9 / 未来 epic

P2a 及其子阶段均自 `main` 分支起步 (master plan §3)。

| Stage | Scope |
|---|---|
| P2a | `OMP_SCHEDULE` per-case tuning + cache-line padding for owner-local SoA + NUMA cross-socket migration cost quant |
| 未来 epic | mode D (OpenMP NVec + StrictOMP RHS) 96-cell research verification (本 epic deferred per tasks §2.5.1) |
| 未来 epic | Mac advisory cross-N (PR-J SHOULD layer，本 epic 只做 N=1 SHALL closure) |
| ADR-0003 forthcoming | KLU spike (D12.4 NOT triggered，留 forthcoming epic) |
| P9 | production deterministic N_Vector + §1.1.1 P9 final 全部达成 (P1e SHIP 仅闭合 strict-omp mode 至 production candidate) |

P2a entry condition (per `docs/p1e/p1e_report.md` §10 + ADR-0002 §"Consequences" Positive 第 4 项)：`P1e-tag` push DONE + `baseline/P1e` lock DONE + 3 SHALL gate (AC-S1 + AC-S2 PASS + AC-S3 partial-closure SHIP) DONE + heihe_x4 sp@8 ≥ 1.5× DONE + ADR-0002 Status `Implemented` DONE + OpenSpec 2 spec PROMOTE DONE + glossary 4 新术语入册 DONE + jsonl epic close-out entry 存在 DONE。`docs/status_matrix.md` P2a row 由 PR-K 更新为 prerequisites = P1e。

## 7. 遗留事项 (P2a / 未来 epic 继承)

| # | Debt | Source | Disposition |
|---|---|---|---|
| 1 | heihe small-case sp@8 1.066× < 1.3× threshold | PR-I AC-S3 per-case FAIL | **ACCEPTED carve-out**：per design D7 asymmetric thresholds + OMP overhead floor 设计预期；`SHUD_RHS_THREADS=1` for heihe production default |
| 2 | mode D (OpenMP NVec + StrictOMP RHS) 96-cell 未验 | tasks §2.5.1 deferred | 未来 epic：research 边界，不是 production gate；placeholder 已在 `docs/p1e/p1e_2x2_experiment.md` §5 写出 |
| 3 | Mac advisory cross-N (PR-J SHOULD layer) 未做 | PR-J `p1e_mac_reverse_compat.md` §6 forward | 未来 epic：本 epic 仅 N=1 SHALL closure，cross-N Mac advisory 留 future epic |
| 4 | KLU pattern spike (ADR-0003 forthcoming) | D12.4 NOT triggered | ADR-0003 forthcoming epic：D12.3 既未触发，无递进条件至 D12.4 |
| 5 | P1e-tag annotated message 内 PR-L / PR-M 占位 `<TBD>` | PR-L #335 review-loop-log F-R2-1 deferred | **DOCUMENTED**：Tag object immutable per D11；PR# 映射 (PR-L #335 / PR-M #336) 记 `docs/p1e/p1e_summary.md` §10 + 本文 §3 |
| 6 | rivqdown.dat recompute helper perf cost +7.7% | PR-B0 Phase 6 fix per outer #323 Phase 4.5 verifier | **ACCEPTED**：output 语义正确性优先于 wall cost；P2a 可考虑做 batched recompute |

## 8. B-chain 不可变链 (D11 enforced)

```
B0-tag             (884cfb13 / SHUD 78c37a1, 2026-06-17)
B1a-tag            (f3a7ff1e / SHUD 0b3998d, 2026-06-21, 由 S1d-end 一次性 force-update)
B1b-tag            (96e224da / SHUD 71b3a1ae, 2026-06-22 PR-16 #207)
B1-tag             (0c0621c9 / SHUD 017c629, 2026-06-22 PR-19 #210 D9 fast-path #2)
P1-update-omp-tag  (ff21c75c / SHUD 07c677f, 2026-06-22 PR-L #224)
P1c-tag            (1da5eb97 / SHUD 3a0004c, 2026-06-23)
P1d-tag            (a82bf336 / SHUD 210ac19, 2026-06-24)
P1e-tag            (25023eff / SHUD 3341368, 2026-06-25 PR-L #335)
  ↑ baseline/P1e D11 locked
```

(上表所列为 annotated tag object SHA prefix；deref commit SHA / SHUD pin 详 §3 与各阶段 capstone summary。D11 immutability re-verify：前 7 tag SHA 在 PR-L 执行后经 `git rev-parse` 实地比对全 PASS — 6 historical tag objects 未受 PR-L 操作影响。)

## 9. `P1e-tag` 验证

```bash
git ls-remote --tags origin | grep P1e-tag
# refs/tags/P1e-tag       25023eff32d1fa317b045cbc786f379fac9e522c  ← annotated tag object
# refs/tags/P1e-tag^{}    11687b756dd53bb634df391bcbeb64b3cef5c750  ← deref commit

git show P1e-tag --no-patch --format=fuller
# Tagger: DankerMu <mumzy@mail.ustc.edu.cn>
# P1e epic capstone — SHIP via §4.6.2 partial-closure
# ADR-0002 Path 1 Implemented / 3 SHALL gate verdict / SHUD pin trail
# 4-mode build matrix / 14-PR cross-ref / D11 7-tag chain re-verify

gh api repos/DankerMu/SHUD-OpenMP/branches/baseline%2FP1e/protection \
  --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'
# {"allow_deletions":false,"allow_force_pushes":false,"enforce_admins":true,"lock_branch":true}
```

## 详细证据索引

- `docs/p1e/p1e_summary.md` — capstone source of truth (PR-K) + R2 F-R2-1 forward note (PR-L → PR-M PR# 映射)
- `docs/p1e/p1e_report.md` — executive 层 report (面向 owner + 跨 epic 复盘 + D11 7-tag chain final state)
- `docs/p1e/p1e_2x2_experiment.md` — 2×2 build matrix Phase 1/2 实验设计 + 180 cell breakdown
- `docs/p1e/p1e_2x2_verdict.md` — Phase 1 + Phase 2 verdict + §6 D12 routing + §4.6.2 partial-closure SHIP rationale
- `docs/p1e/p1e_pr_c_2x2_mac.md` / `p1e_pr_d_2x2_server.md` — Mac + server Phase 1 mode A/B raw evidence
- `docs/p1e/p1e_pr_b0_rivqdown_recompute.md` — rivqdown.dat tout-boundary recompute helper + 4 case × 4-N 验收
- `docs/p1e/p1e_strict_omp_design.md` — design D2 (single parallel region) + D4 (first-touch removal)
- `docs/p1e/p1e_thread_split.md` — `SHUD_RHS_THREADS` vs `OMP_NUM_THREADS` env split rationale
- `docs/p1e/p1e_first_touch_removal.md` — PR-H 3 处 steady-state first-touch loop 删除分析
- `docs/p1e/p1e_pr_i_strict_omp_verification.md` — server 24-cell 3 SHALL gate verification raw data
- `docs/p1e/p1e_mac_reverse_compat.md` — Mac 4-case × N=1 SHALL closure + 6-case cross-platform SHA matrix
- `docs/p1e/p1e_perf_baseline.md` — Mac 16-cell + server PR-I 24-cell wall + 速度比 + small-case 三因素分析
- `docs/p1e/p1e_tag_and_lock.md` — `P1e-tag` annotated procedure + `baseline/P1e` lock 实施记录 + D11 7-tag verify
- `docs/p1e/p1e_pr_n_block_jacobi.md` — D12.3 NOT triggered 占位 note
- `docs/p1e/p1e_rivqdown_cache_audit.md` — PR-A rivqdown.dat cache 审计
- `docs/p1e/p1e_toolchain_investigation.md` — tasks §11.A toolchain investigation NOT triggered 占位
- `docs/adr/0002-solver-path.md` — ADR-0002 Path 1 SELECTED + Implementation closure (Status: Implemented 2026-06-25)
- `docs/build_manifest.md` "P1e-tag" 节 — build provenance
- `docs/status_matrix.md` 第 P1e 行 — 单行 SHIP-LOCKED 汇总
- `openspec/specs/p1e-strict-omp-rhs/spec.md` (11 reqs) + `openspec/specs/p1e-capstone/spec.md` (10 reqs) — PROMOTE 之后的 canonical spec
- `openspec/glossary.md` §"P1e strict-omp F-path baseline 集合" — 新增 4 个术语 (`P1e-tag` / `baseline/P1e` / `strict-omp mode` / `2×2 build matrix`) + P1d carve-out term status 更新
- `SHUD_openMP_master_plan.md` §6 P1e.1-7 (v1.5 / M10) — stage 路线
