## 工作情况说明（Merge 前）

- 关联 Issue：#345
- PR：#357
- 冻结提交（外层）：`1e45e16`
- SHUD upstream：`openmp-baseline-p8pre` HEAD = `5276167`
- 上游 Epic：#338 (p8pre-spike Step 2 PR-D impl)
- 前序 PR：#356（PR-D pre-flight，API verify + SHUD branch fork）

### 背景与目标

p8pre-spike **Step 2 P8-precond-0 spike PR-D impl slice**（depends on #344 / #356 SHUD `openmp-baseline-p8pre` fork from `7a1dc8f`，generates the binary that PR-E #346 / PR-F #347 consume）。本 PR 完成 7 件事：

1. **2 pre-flight check 落地** — name collision `grep -rnE "PSetup|PSolve" SHUD/src/` 0 hit / timer.cpp emit pre-check 验证 `t_precond_setup` 不在 `kKnownRawOrCanonical[]` L188-191 → catch-all 自动 emit → timer.cpp 0 edit
2. **SHUD source 新建 2 文件** — `MD_precond_identity.h` (40 行) + `MD_precond_identity.cpp` (61 行) on `openmp-baseline-p8pre` 分支
3. **SHUD source 编辑 1 文件** — `cvode_config.cpp` (+24/-3): `#include "MD_precond_identity.h"` + `SUNLinSol_SPGMR(udata, 0, 0, sunctx)` → `SUNLinSol_SPGMR(udata, PREC_LEFT, 0, sunctx)` + `CVodeSetPreconditioner(cvode_mem, PSetupIdentity, PSolveIdentity)` + `CVodeSetLSetupFrequency(cvode_mem, 50)` post `CVodeSetLinearSolver` check_flag
4. **Mac local build sanity** — `make shud SHUD_ENABLE_OPENMP_RHS=1` 双构建矩阵 exit 0 (含 PROFILE=1)
5. **3-symbol nm verify** — PSetupIdentity / PSolveIdentity / CVodeSetPreconditioner 全链接成功
6. **Mac keliya N=1 smoke** — `./shud keliya` exit 0 + profile_B0.yaml `extras: t_precond_setup: 0.000037944` (soft gate 6 evidence path PASS) + cvode_stats nps=209227 / npe=1232 (gate 3 PASS)
7. **SHUD commit + push openmp-baseline-p8pre + 外层 pointer bump** — forward-only descendant of `7a1dc8f` 验证（merge-base 严格相等）

### 本次具体改动

| 文件 / 上游状态 | 改动概要 |
|---|---|
| `SHUD/src/Equations/MD_precond_identity.h` (新, 40 行) | `extern "C"` CVLsPrecSetupFn + CVLsPrecSolveFn 声明 |
| `SHUD/src/Equations/MD_precond_identity.cpp` (新, 61 行) | PSetupIdentity (RAII Timer + jok-mirror) + PSolveIdentity (N_VScale 1.0 r→z) |
| `SHUD/src/Equations/cvode_config.cpp` (+24/-3) | include + PREC_LEFT switch + CVodeSetPreconditioner + CVodeSetLSetupFrequency(50) 注册 |
| **外层 SHUD pointer bump** | `7a1dc8f` → `5276167` (forward-only descendant 验证) |
| `tools/profile/timer.cpp` | 未改（catch-all 自动 emit `t_precond_setup` 到 `extras:` 验证 pre-check 期望成立） |

无 `.gitmodules` 改动，无 SHUD master pollution（commit 仅在 `openmp-baseline-p8pre` 长寿分支上），无 `openspec/changes/` 改动，无 CI rule 改动，无 master plan / case_deployment_map 改动。

### 测试与验证

**Mac local build**:
- `make shud SHUD_ENABLE_OPENMP_RHS=1` exit 0
- `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` exit 0
- `make -n shud SHUD_ENABLE_OPENMP_RHS=1 | grep MD_precond_identity` ≥ 1 hit（glob auto-pickup）

**3-symbol nm verify** (PROFILE=1 binary):
- `_PSetupIdentity` T (defined)
- `_PSolveIdentity` T (defined)
- `_CVodeSetPreconditioner` U (imported from SUNDIALS lib)
- counts: PSetup/PSolve = 2 ✓, CVodeSetPreconditioner = 1 ✓

**Mac keliya N=1 smoke** (`./shud keliya`):
- exit 0
- `profile_B0.yaml` extras: `t_precond_setup: 0.000037944` (soft gate 6 evidence path ✓)
- `cvode_stats`: nps=209227 ✓, npe=1232 ✓ (gate 3 PASS)
- nst=101042, nfe=102095, nni=102094, netf=7

**Forward-only descendant verify**:
- new SHUD HEAD = `5276167`
- `merge-base 5276167 7a1dc8f = 7a1dc8f` exactly（strict forward-only per spec L148-154）

**openspec**: `openspec validate p8pre-spike --strict --no-interactive` exit 0
**CI**: 5/5 PASS（asan-ubsan keliya/qhh + build-and-compare keliya + setup + tools-tests）

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (expanded, 4 parallel reviewers, **全 APPROVE 0 findings**):
  - `review-spec-compliance`: 10/10 PASS（spec L26 vs gate-3 内部 inconsistency 已由 implementer 用 SUNDIALS canonical jok-mirror 解决；recommends 未来 openspec patch 给 PR-F #347 / #349）
  - `review-correctness`: 8/8 PASS（signatures 与 cvode_ls.h:57-63 完全匹配 + jok-mirror 与 cvDiurnal_kry.c L716/L760 canonical 一致 + cvode_config.cpp edits 精确 + 3-symbol nm + smoke 全验证）
  - `review-integration`: 12/12 PASS（.gitmodules 未触 + SHUD master 未触 + CVLS API order canonical + Mac vs server toolchain risk LOW + /tmp namespace disjoint）
  - `review-security-perf`: PASS（POD-write + N_VScale_Serial 显式 z==x aliasing 安全 per nvector_serial.c:531 + Timer RAII thread-safe + overhead 3.23e-7 vs 5% threshold 留 6 orders headroom）
- **Phase 4.5 verifier**: SKIPPED (0 PLAUSIBLE candidates)
- **Phase 5/6**: SKIPPED (cross-review clean)
- **Phase 7 final review** (Gap Sweep): **clean**，0 new findings，12/12 AC PASS，oracle integrity PASS，CI 5/5 PASS

### Implementer Deviation 解释

**Brief**: PSetupIdentity 中 `*jcurPtr = SUNFALSE` 无条件。
**Implementer 实际**: `*jcurPtr = jok ? SUNFALSE : SUNTRUE`（jok-mirror per SUNDIALS canonical `cvDiurnal_kry.c` L716/L760）。

**Root cause**: SUNDIALS 6.0.0 CVLS 内部 `cvls_mem->npe` 只在 `jok == SUNFALSE` rebuild 分支内 incrementing。Unconditional SUNFALSE 给 npe=0 → 违反 gate 3 (`npe > 0`)。

**Numerical equivalence**: `PSolveIdentity` always `N_VScale(1.0, r, z)` → P=I 不变；jok-mirror 只影响 CVLS state machine bookkeeping (precond cache scheduling)，不改变 preconditioner 数学等价性。

**Result**: npe=1232 ✓ + 保持 P=I contract。Phase 4 spec-compliance reviewer 独立 verified against SUNDIALS canonical example，**APPROVE**。

### 兼容性、风险与已知限制

- 无 API 兼容性破坏（pure addition: 2 新文件 + 1 文件 +24/-3 edits）
- SHUD upstream `openmp-baseline` master 分支未触（C8 不污染）
- `openmp-baseline-p8pre` 分支上已有 1 commit（5276167）— PR-E #346 server 将 `git fetch` + `git checkout 5276167` 直接编译
- **forward note**: `PREC_LEFT` 是 SUNDIALS 6.0.0 中 `SUN_PREC_LEFT` 的 deprecated alias（sundials_iterative.h:55-58）— 6.0.0 baseline 可用，但留 ADR-0003 / 未来 SUNDIALS bump cosmetic upgrade
- **forward note (jok-mirror canonical)**: spec L26 `*jcurPtr = SUNFALSE` literal 与 gate-3 (`npe > 0`) 内部矛盾；implementer 用 jok-mirror 解决，等待 PR-F #347 verdict doc cite + 未来 openspec patch 修订 L26 wording
- **forward note (B1b bitwise neutrality)**: 切到 PREC_LEFT 必然破 B1b PREC_NONE bitwise identity（CVLS state machine 引入额外 ops）；soft gate 5 fallback (max_ulp ≤ 1024) 在 PR-F #347 处理

### 维护者关注点

- 无额外关注点。下一步 **#346 PR-E**：
  - server `git fetch SHUD` + checkout 5276167
  - server build `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` + 3-symbol nm verify
  - server 18-cell identity spike (heihe / heihe_x4 × N=1/4/8 × 3 reps, singleton afterany chain per (case, N))
  - ~6h compute wait（30 min wakeup poll）
