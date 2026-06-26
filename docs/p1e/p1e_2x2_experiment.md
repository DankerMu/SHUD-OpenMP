# P1e — 2×2 build matrix 综合实验

P1e.2 因果实验 capstone 综合表。合并 PR-C Mac 48 cell + PR-D server 48 cell (Phase 1) + PR-I mode C 24 cell (Phase 2 server) + PR-J Mac mode C 4 cell (Phase 2 N=1) 数据，统一表达 4 build × 4 N × 4 case 矩阵在 Phase 1 + Phase 2 的全数据 + D12 routing 实际触发分支。本 doc 是 PR-K capstone 输出，作 PR-L `P1e-tag` annotated message + PR-M PROMOTE 的引用。

源 doc：

- `docs/p1e/p1e_pr_c_2x2_mac.md` (Phase 1 Mac 48 cell mode A+B)
- `docs/p1e/p1e_pr_d_2x2_server.md` (Phase 1 server 48 cell mode A+B)
- `docs/p1e/p1e_2x2_verdict.md` (Phase 1 verdict + Phase 2 PR-I amend §6)
- `docs/p1e/p1e_pr_i_strict_omp_verification.md` (Phase 2 server mode C 24 cell)
- `docs/p1e/p1e_mac_reverse_compat.md` (Phase 2 Mac mode C 4 cell × N=1)

## §1 实验设计回顾

per ADR-0002 §"Validation Plan" + tasks §2 + §4：

| Phase | 范围 | scope | PR ownership |
|---|---|---|---|
| Phase 1 | mode A + mode B | 4 case × 4 N × 3 reps × 2 mode = 96 cell per platform → 192 cell total | PR-C (Mac) + PR-D (server) + PR-E (verdict) |
| Phase 2 | mode C (+ mode D deferred) | 2 case × 4 N × 3 reps × 1 mode = 24 cell server + 4 case × N=1 × 3 reps × 1 mode = 12 cell Mac | PR-I (server) + PR-J (Mac N=1) |

Mode D Phase 2 (96 cell) 显式 **deferred**: per tasks §2.5.1 + §2.6.1 mode D 是 research 边界，不是本 epic verdict input；scope 留 future epic 或 ADR-0002 spike。

## §2 4-mode build matrix 综合表

(同 `docs/p1e/p1e_perf_baseline.md` §2 表，此处复述以便本 doc 独立可读)：

| Mode | Target | NVector | RHS | 实测 phase |
|---|---|---|---|---|
| A | `shud` | `N_VNew_Serial` | `ExecPolicy::Serial` | Phase 1 (PR-C/D) |
| B | `shud_omp` | `N_VNew_OpenMP` | `ExecPolicy::Serial` | Phase 1 (PR-C/D) |
| C | `shud SHUD_ENABLE_OPENMP_RHS=1` | `N_VNew_Serial` | `ExecPolicy::StrictOMP` | Phase 2 (PR-I + PR-J) |
| D | `shud_omp SHUD_ENABLE_OPENMP_RHS=1` | `N_VNew_OpenMP` | `ExecPolicy::StrictOMP` | **deferred** |

## §3 Phase 1 综合 (mode A + mode B, 96 cell per platform)

### 3.1 mode A 3-rep bitwise per (case, N) — AC1 SHALL gate

per `docs/p1e/p1e_pr_c_2x2_mac.md` §3.1 + `docs/p1e/p1e_pr_d_2x2_server.md` §3.1：

| platform | case | 4 N × 3 reps = 12 cell unique SHA | verdict |
|---|---|---|:---:|
| Mac | keliya | 1 | PASS |
| Mac | xinanjiang_upstream | 1 | PASS |
| Mac | qinyijiang | 1 | PASS |
| Mac | qhh | 1 | PASS |
| server | heihe | 1 | PASS |
| server | heihe_x4 | 1 | PASS |
| server | (PR-D 2-case subset; Mac PR-C 4-case full) | — | — |

**结论**：Phase 1 mode A 全 24-cell × 3-rep bitwise 跨 N + 跨 reps 全等 → solver toolchain (GCC 13.3.0 + libgomp on server / Apple Clang + libomp on Mac) 自身 deterministic → tasks §11.A toolchain investigation **不触发**。

### 3.2 mode A cross-N stability per case (rep=1) — AC2 SHALL gate

mode A 自身 ExecPolicy::Serial（不并行），跨 N 仅是 NVector backend `omp_set_num_threads(MD->CS.num_threads)` 设值，N_Vector serial backend 不实际使用线程数 → 跨 N 行为 invariant → AC2 全 PASS。

### 3.3 mode B 跨 N drift 复现 — informational

per `docs/p1e/p1e_pr_c_2x2_mac.md` §3.4 + `docs/p1e/p1e_pr_d_2x2_server.md` §3.4 + §4：

| platform | case | mode B unique SHAs across 4 N | mode B vs mode A SHA12 N=1 |
|---|---|---:|---|
| Mac | keliya | 3-4 (跨 N drift) | 同 mode A (single-thread N=1 边界) |
| Mac | xinanjiang_upstream | 3-4 | 同 |
| Mac | qinyijiang | 4 | 同 |
| Mac | qhh | 4 | 同 |
| server | heihe | 3-4 (PR-H 10-25% rel) | 同 |
| server | heihe_x4 | 4 | 同 |

**结论**：mode B 跨 N drift 复现 → 验证 ADR-0002 Fact #1+#2: `f.cpp::54` 始终 ExecPolicy::Serial RHS + NVector backend `N_VDotProd_OpenMP` reduction tree 跨 N 不固定 → drift 来自 NVector reduction，不是 RHS race。

### 3.4 mode B 加速比 — informational

per `docs/p1e/p1e_pr_d_2x2_server.md` §3.5：

| case | mode B sp@8 (median 3 reps) |
|---|---:|
| heihe | 1.13× (Amdahl IO bound) |
| heihe_x4 | 1.27× (NVector fork-join + Serial RHS 上限) |

**结论**：mode B 跨 N drift + 不达 ROI threshold → P1d era `shud_omp` (= mode B) 不能直接 ship strict mode → 需 mode C 验证（=本 epic Phase 2 SHALL gate）。

## §4 Phase 2 综合 (mode C, 24 cell server + 4 cell Mac × N=1)

### 4.1 mode C cross-N bitwise per case — AC-S1 SHALL gate

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.1：

| platform | case | 4 N × 3 reps = 12 cell unique SHA | rep1 SHA12 | verdict |
|---|---|---:|---|:---:|
| server | heihe | 1 | `a2023ccd2de4` | PASS |
| server | heihe_x4 | 1 | `b5e4b0a2cf83` | PASS |

### 4.2 mode C SHA == mode A reference SHA — AC-S2 SHALL gate

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.2 + `docs/p1e/p1e_mac_reverse_compat.md` §3.5：

| platform | case | mode C SHA16 (N=1, rep=1) | mode A reference SHA16 | match | verdict |
|---|---|---|---|:---:|:---:|
| server | heihe | `a2023ccd2de43543` | `a2023ccd2de43543` (PR-D #312) | same | PASS |
| server | heihe_x4 | `b5e4b0a2cf83b2a4` | `b5e4b0a2cf83b2a4` (PR-D #312) | same | PASS |
| Mac | keliya | per PR-J §3.5 | per PR-C #312 | same | PASS |
| Mac | xinanjiang_upstream | per PR-J §3.5 | per PR-C #312 | same | PASS |
| Mac | qinyijiang | per PR-J §3.5 | per PR-C #312 | same | PASS |
| Mac | qhh | per PR-J §3.5 | per PR-C #312 | same | PASS |

**6-case roll-up**：6/6 case mode C SHA == mode A reference SHA → AC-S2 **全 PASS**（per `docs/p1e/p1e_mac_reverse_compat.md` §3.5 AC-J2 6-case roll-up）。

### 4.3 mode C D7 per-case speedup — AC-S3 SHALL gate

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §3.3 + §4 + `docs/p1e/p1e_perf_baseline.md` §3.2：

| case | N=1 wall (s) | N=8 wall (s) | sp@8 | threshold | per-case |
|---|---:|---:|---:|---:|:---:|
| heihe | 504 | 473 | 1.066× | ≥1.3× | FAIL |
| heihe_x4 | 1340 | 775 | 1.729× | ≥1.5× | PASS |

AND-gate: BOTH FAIL 才触发 D12.3，本实验 heihe FAIL + heihe_x4 PASS → AND-gate **不满足** → §4.6.2 partial-closure 决策点 (SHIP)。

### 4.4 mode C nst Δ stability — informational

per `docs/p1e/p1e_pr_i_strict_omp_verification.md` §5：

| case | ref nst | 4 N × nst max |Δ| | ladder |
|---|---:|---:|:---:|
| heihe | 6698 | 0 | PASS (Δ=0 strict) |
| heihe_x4 | 6575 | 0 | PASS (Δ=0 strict) |

## §5 Phase 1 + Phase 2 总细胞数核算

| Phase | platform | matrix shape | cell count |
|---|---|---|---:|
| Phase 1 | Mac | 4 case × 4 N × 3 reps × 2 mode | 96 |
| Phase 1 | server | 2 case × 4 N × 3 reps × 2 mode (server scope = heihe + heihe_x4, per PR-D §2) | 48 |
| Phase 2 | server | 2 case × 4 N × 3 reps × 1 mode (mode C) | 24 |
| Phase 2 | Mac | 4 case × N=1 × 3 reps × 1 mode (mode C) | 12 |
| Phase 2 deferred | mode D × {Mac + server} | 4 case × 4 N × 3 reps × 1 mode × 2 platform | 96 (deferred) |
| **本 epic 实跑** | | | **180** |
| **本 epic + mode D deferred** | | | **276 (capstone spec 上界)** |

**注**：`openspec/changes/p1e-strict-omp-rhs/specs/p1e-capstone/spec.md` Requirement "docs/p1e/ ≥14 doc" Scenario 提到 "192 cell" 是 Phase 1 全 mode A/B × Mac+server 跨平台理想上界（与本 epic 实测 Phase 1 144 cell 区别：server scope=2 case vs Mac scope=4 case）。本 epic 实跑 180 cell = Phase 1 144 + Phase 2 mode C 36；含 mode D Phase 2 96-cell deferred 后 capstone 上界 = 180 actual + 96 deferred = **276** (Mac 132 + server 144 若 mode D 全跑); coverage = 180 / 276 = **65.2%**。

## §6 D12 routing 实际触发分支

per `docs/p1e/p1e_2x2_verdict.md` §6.2 + tasks §4.6 + design.md L325-328：

| Branch | 条件 | 实测 eval | 结果 |
|---|---|---|:---:|
| D12.1 (happy path) | mode C cross-N PASS + nst Δ=0 + per-case speedup SHALL PASS (BOTH) | AC-S1+S2+nst PASS, but heihe sp@8 1.066× < 1.3× → BOTH 条件未满 | **NOT triggered** |
| D12.2 (Path 2 fallback) | mode C cross-N FAIL → NVECTOR_REPRO_OMP 自研后端 | AC-S1 PASS (cross-N bitwise on both) → 条件未满 | **NOT triggered** |
| D12.3 (Path 3 fallback) | cross-N PASS + **BOTH** cases < own threshold → PR-N block-Jacobi precond | heihe FAIL + heihe_x4 PASS → AND-gate **不满足** | **NOT triggered** |
| D12.4 (Path 4 deferred) | none of D12.1/.2/.3 + deeper solver refactor needed → ADR-0003 KLU spike | D12.3 未触发, no total-failure | **NOT triggered** |
| §4.6.2 partial-closure | 单 case 不达 threshold + 另一 case 达 → user 决策 ship vs fallback | heihe FAIL + heihe_x4 PASS (1.729× ≥ 1.5×) | **triggered (active path: SHIP)** |

**实际触发**：§4.6.2 partial-closure → SHIP。其他 4 branch 全 NOT triggered。

placeholder doc `docs/p1e/p1e_pr_n_block_jacobi.md` 已 PR-K 写出 (note: "not triggered (D12.3 fallback path not exercised)")，作 spec p1e-capstone Requirement "docs/p1e/ ≥14 doc" Scenario 合规占位（per tasks §7.10.2）。

## §7 Phase 2 Mode D 96-cell 数据 (deferred)

Mode D (`OpenMP NVec + StrictOMP RHS`) Phase 2 96-cell 显式 **deferred to future epic / ADR-0002 spike**：

- per tasks §2.5.1 + §2.6.1：mode D 是 research 边界，不是本 epic verdict input
- per ADR-0002 §"Decision Matrix"：Path 1 SELECTED → mode C/D 关系是 production vs research，本 epic 仅 prod
- **缺失数据替代**：mode D 跨 N drift 模式预期与 mode B ≈（NVECTOR_OPENMP reduction 主因）→ 由 Phase 1 mode B drift 数据外推（per `docs/p1e/p1e_pr_d_2x2_server.md` §4 F2）

future epic 触发 mode D 实跑的条件（非阻塞 P1e）：

- ADR-0003 (forthcoming) NVECTOR_REPRO_OMP 评估时
- Path 2 重新评估时（若 mode C 在 P2 阶段需 NVECTOR 后端配合）

## §8 verdict + 路径决策

**verdict**: AC-S1 + AC-S2 全 PASS, AC-S3 PARTIAL, D12 4 branch 全 NOT triggered, §4.6.2 partial-closure → **SHIP**.

**SHIP rationale** (per `docs/p1e/p1e_2x2_verdict.md` §6.3)：

1. strict-omp RHS 在 production-target mesh density (heihe_x4 ~25k cells) 达 1.729× ≥ 1.5× threshold
2. 6/6 case mode C SHA == mode A reference SHA (bitwise cross-mode 跨平台)
3. heihe small-case 1.066× 不达是 OMP overhead floor，设计预期，非 implementation bug → per §6 carve-out 接受
4. nst Δ=0 跨 N strict closure (mode B era 跨 N Δ 不闭合 → mode C 闭合 = strict-omp 实质成果)

**forward to PR-L**：`P1e-tag` annotated procedure (引用本 doc + p1e_summary.md + p1e_perf_baseline.md) + `baseline/P1e` lock (lock_branch=true)。

**forward to PR-M**：2 spec PROMOTE + glossary 4 new terms + jsonl 双追加 + epic close-out。
