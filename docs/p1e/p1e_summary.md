# P1e — capstone summary

P1e epic (`P1e.1 ~ P1e.7` per master plan §6 v1.5 M10 + tasks.md §1-§7) 总结。P1e 阶段 13 sub-PR + capstone consolidation 全部 merged 到 `baseline/P1e` 分支。本文件作 PR-L `P1e-tag` annotated message + PR-M PROMOTE archive 的 source of truth，且通过 §"P1d carve-out closure" 章关闭 master plan §6 P1d.7 forward debt。

## §1 Status

| Field | Value |
|---|---|
| Epic | #283 |
| Status | **SHIP via §4.6.2 partial-closure** |
| Date | 2026-06-25 |
| SHUD pin trail | P1d `210ac191...` → P1e `3341368d2d0854924d2286925c8575df52cc97a0` (PR-F StrictOMP impl + PR-G -fopenmp wiring + SHUD_RHS_THREADS env + PR-H first-touch removal + omp single→omp for, `openmp-baseline` pushed; net +ExecPolicy enum / +1 single omp parallel region / -3 steady-state first-touch loops vs P1d) |
| Outer HEAD | `<P1e-tag deref TBD by PR-M post-merge>` |
| Master plan revision | v1.5 / M10（P1e 章在 §6 P1e.1-7 内冻结） |
| ADR-0002 | **Implemented (P1e epic close, 2026-06-25)** — Path 1 (Serial NVec + StrictOMP RHS) SELECTED + executed + 验收 SHIP；详 `docs/adr/0002-solver-path.md` §"Implementation closure" |
| 4-mode 状态 | `strict-omp` mode = production candidate（PR-I SHALL gate AC-S1 + AC-S2 全 PASS，AC-S3 per-case AND-gate PARTIAL → §4.6.2 partial-closure SHIP） |

P1e 不是简单 "ship strict-omp + 全 case 达 ≥1.5×" 的完美 closure：heihe (6335 cells, small-case) AC-S3 D7 speedup 1.066× < 1.3× threshold；heihe_x4 (~25k cells, production-target mesh) AC-S3 D7 speedup 1.729× ≥ 1.5× threshold。AND-gate semantics (per tasks §4.6 + design D12) 要求**两 case 同时 FAIL** 才触发 D12.3 block-Jacobi fallback；单 case FAIL 走 §4.6.2 partial-closure 决策点。用户决策 SHIP（per `docs/p1e/p1e_2x2_verdict.md` §6.3 + §6.4 SHIP rationale + small-case carve-out）。

详 §3 + §5 + §6 P1d carve-out closure。

## §2 Epic scope

**14 PR total** (11 base sub-PR `A,B,C,D,E,F,G,H,I,J` + PR-B0 audit-required + PR-K capstone + PR-L tag + PR-M PROMOTE)，于 2026-06-24 → 2026-06-25 两天内完成：

| PR | # | Scope | Status |
|---|---|---|---|
| PR-A | #309 | rivqdown.dat cache audit + 8 doc 初稿 | MERGED |
| PR-B | #310 | 2×2 build matrix runner + CV_Y hash tool + manifest yaml | MERGED |
| PR-B0 | #311 | rivqdown.dat tout-boundary recompute via recompute_for_output helper | MERGED |
| PR-C | #312 | Mac 2×2 mode A/B Phase 1: 48 cell raw evidence + 3 SHALL gate verdict | MERGED |
| PR-D | #313 | server 2×2 mode A/B Phase 1: 48 cell raw evidence + 3 SHALL gate verdict | MERGED |
| PR-E | #314 | Phase 1 verdict aggregation + D12 routing placeholder | MERGED |
| PR-F | #315 | SHUD `ExecPolicy::StrictOMP` impl per design D2 (single omp parallel region + omp single scaffolding) | MERGED |
| PR-G | #315 | SHUD Makefile -fopenmp auto-wire + SHUD_RHS_THREADS env split + shud.cpp two-guard form per tasks §3.5.2 | MERGED |
| PR-H | #316 | SHUD MD_rhs_core.cpp: remove 3 steady-state first-touch loops + convert omp single → omp for schedule(static) per design D4/D2 | MERGED |
| PR-I | #317 | server SHALL closure: heihe + heihe_x4 × N∈{1,2,4,8} × 3 reps = 24 cell + 3 SHALL gate verdict + D12 routing | MERGED |
| PR-J | #318/#333 | Mac N=1 reverse-compat closure: 4 case × N=1 raw evidence + spec-canonical doc name | MERGED |
| PR-K | #319 (本 PR) | capstone docs consolidation: 8 new docs to bring docs/p1e/ to **≥14** (canonical per capstone spec L7; actual count = 17) + spec L192 amend + ADR-0002 close-out | (本 PR) |
| PR-L | TBD | `P1e-tag` annotated procedure + `baseline/P1e` lock | PENDING |
| PR-M | TBD | OpenSpec PROMOTE 2 spec + glossary 4 new terms + jsonl entries + epic close | PENDING |

## §3 What was attempted

P1d epic 关闭后 carve-out 是 "真正应并行的 RHS 还没并行" — 当前 `shud_omp` 实测仅 Serial 水文 RHS + OpenMP N_Vector backend，N_Vector 内 reduction 不固定造成跨 N 不 bitwise。P1e (F 路) per ADR-0002 Path 1：

1. **2×2 build matrix 因果实验** (PR-A/B/B0/C/D/E)：4 build × 4 N × 3 reps × 4 case = **192 cell theoretical upper bound** (Phase 1 mode A+B 全跑)；**actually executed 180 cell** (Phase 1 144 cell mode A/B + Phase 2 mode C 36 cell); mode D 96 cell deferred。详 `docs/p1e/p1e_2x2_experiment.md` §5 breakdown。Mode A (Serial NVec + Serial RHS) 作 canonical reference；Mode B (OpenMP NVec + Serial RHS = 当前 `shud_omp`) 作 control 复现 PR-H 10-25% 散度；Mode C (Serial NVec + StrictOMP RHS) 是 P1e production 候选；Mode D (OpenMP NVec + StrictOMP RHS) 是 research 边界。
2. **rivqdown.dat tout-boundary recompute** (PR-B0)：审计发现 PCtrl-aliased output cache 是 internal cache (CV_NORMAL 模式下内部步可超过 tout)，新增 `Model_Data::recompute_for_output(N_Vector, double)` helper 在 MainLoop 内 `summary` 与 `ExportResults` 之间调 `rhs_update + rhs_flux + rhs_apply` 全链。
3. **`ExecPolicy::StrictOMP` 实施** (PR-F)：`ExecPolicy::Serial` 之外新增第 2 case，把 `rhs_update / rhs_flux / rhs_apply / rhs_deterministic_gather` 4 个 RHS 方法的 outer call 放到单个 `#pragma omp parallel` region 内，所有线程一起 enter 每个方法。
4. **`-fopenmp` 自动 wire + `SHUD_RHS_THREADS` env split** (PR-G)：Makefile 内 `SHUD_ENABLE_OPENMP_RHS=1` 自动加 `-fopenmp` + 不要求 user manually pass；shud.cpp startup 单点 read `SHUD_RHS_THREADS` env (`omp_set_num_threads(getenv("SHUD_RHS_THREADS")?:omp_get_max_threads())`) 形成 RHS 并行度 canonical knob，与 `OMP_NUM_THREADS` (NVECTOR 用) 解耦。
5. **steady-state first-touch removal + omp single → omp for** (PR-H)：删 `MD_rhs_core.cpp` L62-95 / L169-203 / L324-354 三处 steady-state first-touch loops（StrictOMP outer parallel region 已存在时这些 inner `#pragma omp parallel for` 会嵌套，违反 single-region rule）；把 PR-F 的 `omp single` 改造为 `omp for schedule(static)` work-sharing；保留 allocation-time first-touch (Model_Data.cpp::malloc_EleRiv) + load-time first-touch (MD_initialize.cpp::LoadIC)。
6. **server SHALL gate 验收** (PR-I)：server Slurm cn14 + cn15 跑 24 cell (heihe + heihe_x4 × 4 N × 3 reps)。SHA `a2023ccd2de4` (heihe) / `b5e4b0a2cf83` (heihe_x4) 跨 4 N × 3 reps 全等 → AC-S1 PASS；mode C SHA == PR-D mode A reference SHA → AC-S2 PASS；nst Δ=0 跨 4 N → nst ladder PASS；heihe 1.066× / heihe_x4 1.729× → AC-S3 PARTIAL (heihe < 1.3×, heihe_x4 ≥ 1.5×)。
7. **Mac reverse-compat 验收** (PR-J)：Mac local 跑 4 case × N=1 mode C，4 case SHA 全等于各自 PR-E mode A reference SHA → N=1 reverse-compat PASS。

## §4 What was NOT attempted

- **D12.2 (NVECTOR_REPRO_OMP 自研后端)**: 不触发 — AC-S1 跨 N bitwise PASS, mode C path 自身 deterministic, 无需后端 fallback。
- **D12.3 (block-Jacobi precond → PR-N)**: 不触发 — AND-gate 要求**两 case 同时** < 各自 threshold；本 epic 实测 heihe FAIL 但 heihe_x4 PASS → AND-gate **不满足**。Placeholder `p1e_pr_n_block_jacobi.md` 已 PR-K 写出，留 PR-N "未触发" 占位。
- **D12.4 (KLU spike, ADR-0003 forthcoming)**: 不触发 — D12.3 既未触发，无递进条件至 D12.4；KLU 议题留 ADR-0003 forthcoming epic（不在 P1e scope）。
- **mode D 96-cell**：本 epic 仅跑 mode A/B Phase 1 (96 cell) + mode C 24-cell。Mode D (per tasks §2.5.1 + §2.6.1) 显式 deferred — 与 ADR-0002 + master plan §6 P1e 一致，mode D 是 research 边界，不是 production gate。
- **Mac advisory cross-N (PR-J SHOULD layer)**：仅做 N=1 SHALL closure；cross-N Mac advisory 留 future epic（PR-J `p1e_mac_reverse_compat.md` §6 forward 已记）。

## §5 What changed in baseline

| Field | P1d-tag (`a82bf336`) | P1e-tag（本 epic） | 改动来源 |
|---|---|---|---|
| SHUD pin | `210ac191...` (Kahan revert + first-touch loops, 全 mode 仍 Serial RHS) | `3341368d2d0854924d2286925c8575df52cc97a0` (ExecPolicy::StrictOMP + -fopenmp wire + SHUD_RHS_THREADS env + 3 first-touch loops 删 + omp single→omp for) | PR-F #315 + PR-G #315 + PR-H #316 |
| Build flag matrix | `shud` (serial) + `shud_omp` (= mode B 当前 prod) | `shud` (mode A = serial) + `shud` w/ `SHUD_ENABLE_OPENMP_RHS=1` (= mode C = P1e production 候选) + `shud_omp` (= mode B 仍可用) + `shud_omp` w/ `SHUD_ENABLE_OPENMP_RHS=1` (= mode D 仅 research) | PR-G Makefile -fopenmp 自动 wire |
| Default production thread count | `OMP_NUM_THREADS=1` (per P1d E′ closure: shud_omp 默认 serial) | `SHUD_RHS_THREADS` per case roll-out（heihe_x4 推荐 ≥4, heihe 推荐 1 per §6.3 carve-out） | PR-G + PR-I PARTIAL closure |
| rivqdown.dat 输出语义 | internal cache (CV_NORMAL 模式下内部步) | tout-boundary recompute (rhs_update + rhs_flux + rhs_apply 全链) | PR-B0 helper |
| `p1d-numa-governance` Requirement nst ladder | 受限于 mode B (Serial RHS + OpenMP NVector)，N≥4 nst Δ 不闭合 | mode C 4 N 全 Δ=0 (PR-I §3 nst ladder 表) | mode C deterministic-by-construction |

## §6 P1d carve-out closure

P1d epic 关闭时 forward debt = "真正应并行的 RHS 还没并行" (master plan v1.5 / M10 §6 P1d.7)。本 epic P1e 通过 ADR-0002 Path 1 (Serial NVec + StrictOMP RHS) 完整闭环：

1. **factual delivery**：mode C build 真编出（PR-G 通过 binary symbol verification: `nm ./shud | grep _omp_set_num_threads` 命中 + `nm ./shud | grep N_VNew_Serial` 命中 + `nm ./shud | grep N_VNew_OpenMP` 不命中）。Runtime verify: `SHUD_RHS_THREADS=4 ./shud keliya` 输出 `P1e startup: SHUD_RHS_THREADS=4 -> omp_set_num_threads(4)`。
2. **bitwise reproducibility 闭合**：PR-I AC-S1 mode C 跨 N × 3 reps SHA 全等（heihe `a2023ccd2de4` / heihe_x4 `b5e4b0a2cf83`），AC-S2 mode C SHA == mode A reference SHA → 4-mode 表 strict-omp mode "production candidate" 状态从 "P1e 验收前" 升至 "P1e 验收后 SHIP"。
3. **speedup 部分闭合**：heihe_x4 1.729× ≥ 1.5× threshold 满足 production-target mesh ROI；heihe 1.066× 不达 1.3×，per §6.3 small-case carve-out 设计预期。
4. **production default 转移**：P1d era `NUM_OPENMP=1` (Serial fallback) → P1e era `SHUD_RHS_THREADS` per case roll-out (heihe_x4 推荐 ≥4)。P1d era 的 `shud_omp` (mode B) 不删除（保留作 NVector backend 研究），但不再是 production default。
5. **first-touch 遗产清理**：P1d.2.1/.2/.3 三处 steady-state first-touch loops 在 PR-H 删除（rationale: StrictOMP outer parallel region 与之嵌套违反 single-region rule，且 PR-H 实测删除后 mode A 仍 bitwise 通过 = 这些 loops 在 Serial path 上从来没被需要）。allocation-time + load-time first-touch 保留（PR-H 内 verify: `Model_Data::malloc_EleRiv` L302-317 flat3 zero-write 完整 + `MD_initialize.cpp::LoadIC` 完整）。
6. **glossary P1d carve-out term status 更新**：PR-M task 7.14 will append `**Status (P1e epic close, 2026-06-25)**: CLOSED via P1e epic 2026-06-25，参见 docs/p1e/p1e_summary.md §"P1d carve-out closure"` 到既存 term 末尾（per `openspec/changes/p1e-strict-omp-rhs/specs/p1e-capstone/spec.md` Requirement "glossary 4 新术语" Scenario "glossary 命名空间不冲突 + 既存 P1d carve-out term status 更新"）。

## §7 Verdict per 3 SHALL gate + ADR-0002 Decision Matrix

| 验收项 | 目标 | 实测 | 验收 |
|---|---|---|---|
| AC-S1 mode C 跨 N × 3 reps bitwise | 2 case 各 unique SHA = 1 | heihe + heihe_x4 各 unique SHA = 1 | PASS |
| AC-S2 mode C SHA == mode A reference SHA | per-case 同 SHA | heihe `a2023ccd2de43543` == PR-D ref `a2023ccd2de43543`; heihe_x4 `b5e4b0a2cf83b2a4` == PR-D ref `b5e4b0a2cf83b2a4` | PASS |
| AC-S3 D7 speedup per-case threshold | heihe ≥ 1.3× + heihe_x4 ≥ 1.5× (AND-gate 触发 D12.3 需 BOTH FAIL) | heihe 1.066× (FAIL); heihe_x4 1.729× (PASS) | **PARTIAL** → 进 §4.6.2 partial-closure 决策点 |
| nst Δ=0 跨 N (informational) | 2 case Δ=0 | heihe + heihe_x4 各 4 N nst = case-fixed | PASS |
| Mac N=1 reverse-compat (PR-J) | 4 case N=1 mode C SHA == 各自 mode A ref SHA | 4 case 全 PASS | PASS |

ADR-0002 Decision Matrix Path 1 SELECTED 行：

| Path | Reproducibility | Speedup at N=8 | 实测 |
|---|---|---|---|
| 1 — Serial NVec + StrictOMP RHS | strong | 1.5-2.4× (target) | strong (4N + 3rep bitwise + Mac+server cross-platform); heihe_x4 1.729× 达上界中段, heihe 1.066× 小 case 不达 → 4-mode `strict-omp` 列 "SHIP via §4.6.2 partial-closure" |

## §8 Verification reproducibility footprint

### server (PR-I)

```bash
# build C (Serial NVec + StrictOMP RHS)
cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD
make shud SHUD_ENABLE_OPENMP_RHS=1  # -fopenmp 自动 wire per PR-G
nm ./shud | grep N_VNew_Serial      # SHALL ≥1 hit
nm ./shud | grep N_VNew_OpenMP      # SHALL 0 hit
nm ./shud | grep GOMP_parallel      # SHALL ≥1 hit (Linux libgomp)

# run heihe N=8 (one of 24 cells)
sbatch --array=0-23 .p1e-i-runs/run_pr_i_24cell.sbatch
# 见 docs/p1e/p1e_pr_i_strict_omp_verification.md §6
```

### Mac (PR-J)

```bash
# build C (Serial NVec + StrictOMP RHS)
cd /Users/danker/Desktop/Hydro-SHUD/openMP/SHUD
make shud SHUD_ENABLE_OPENMP_RHS=1  # -Xpreprocessor -fopenmp 自动 wire per PR-G

# run 4 case × N=1 mode C
for case in keliya xinanjiang_upstream qinyijiang qhh; do
  SHUD_RHS_THREADS=1 ./shud $case
done
# 见 docs/p1e/p1e_mac_reverse_compat.md §5
```

## §9 Forward handoff to PR-L / PR-M / P2a

### PR-L (`P1e-tag` annotated procedure + `baseline/P1e` lock)

- tag annotated message body 引用 `docs/p1e/p1e_summary.md` (本文件) + `docs/p1e/p1e_perf_baseline.md`
- baseline/P1e lock via `gh api ... lock_branch=true`
- D11 6 tag chain 升至 **7 tag chain**（B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag / P1e-tag）

### PR-M (PROMOTE + Epic close + post-merge 行政)

- 2 spec PROMOTE: `p1e-strict-omp-rhs` + `p1e-capstone` 至 `openspec/specs/`
- archive `openspec/changes/p1e-strict-omp-rhs` 至 `openspec/changes/archive/2026-06-XX-p1e-strict-omp-rhs/` (local gitignored)
- glossary 4 new terms: `P1e-tag` / `baseline/P1e` / `strict-omp mode` / `2×2 build matrix` + 既存 `P1d carve-out` term status 追加
- jsonl 双追加 (pipeline summary + Epic close-out) 含 `"workflow":"subagent-workflow"`
- post-merge: epic close + baseline lock_branch=true + main fast-forward + D11 final 7-tag verify
- task 5.4 amend: post-PR-L-merge 在 §10 "验证 P1e-tag" 章填实 SHA

### P2a entry condition

P2a 前置改链：原 P1c-tag + P1c.3 充分性不足 → 新前置 = `P1e-tag` lock + `baseline/P1e` lock + 3 SHALL gate (AC-S1 + AC-S2 + AC-S3 partial) + ADR-0002 Implemented status + heihe_x4 ≥ 1.5× 加速 + glossary 4 new terms 入册 + jsonl epic close-out entry 存在。`docs/status_matrix.md` P2a row 由 PR-K 更新为 prerequisites = P1e。

## §10 验证 P1e-tag

_capstone-time SHA: filled by PR-M (#336) per task 5.4 + 7.22_

PR-M post-PR-L-merge amend (per task 5.4 + 7.22 + spec p1e-capstone Scenario "tag 验证章 amend by PR-M")：

- **tag-object SHA**: `git rev-parse P1e-tag` → `25023eff32d1fa317b045cbc786f379fac9e522c`
- **deref commit SHA**: `git rev-parse P1e-tag^{}` → `11687b756dd53bb634df391bcbeb64b3cef5c750`
- **deref commit 内容覆盖**：pre-PROMOTE HEAD（PR-K post-merge log append `chore(p1e): append review-loop-log PR-K entry (post-merge accountability for #334)`，不含 PR-M PROMOTE diff）
- **annotated tag message 引用文档**：
  - `docs/p1e/p1e_summary.md` (本文件)
  - `docs/p1e/p1e_perf_baseline.md`
- **annotated tag message verification**：`git tag -l --format='%(contents)' P1e-tag | grep p1e_summary.md`
- **D11 7-tag chain final state**：B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag / **P1e-tag (新增)** — 前 6 tag SHA 不变 (immutable per master plan §6 D11)

### R2 F-R2-1 forward note (PR-L → PR-M PR# 映射)

P1e-tag annotated tag message body 含 literal `<TBD>` placeholders for PR-L + PR-M PR numbers（per PR-L #335 review-loop-log F-R2-1 deferred）。Tag object immutable per D11 chain discipline；PR numbers 不通过 retagging 修正，而是在本 doc 记录映射：

- PR-L → **#335** (URL: https://github.com/DankerMu/SHUD-OpenMP/pull/335)
- PR-M → **#336** (URL: https://github.com/DankerMu/SHUD-OpenMP/pull/336)
