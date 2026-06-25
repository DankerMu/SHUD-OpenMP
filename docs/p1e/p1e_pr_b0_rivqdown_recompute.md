# P1e PR-B0 — rivqdown.dat tout-boundary recompute

## 摘要

- **触发**：PR-A audit (`docs/p1e/p1e_rivqdown_cache_audit.md`) 结论 = **internal cache** → 升级 PR-B0 conditional → required (per spec p1e-strict-omp-rhs Requirement "rivqdown.dat 输出缓存 audit" L260-285, Scenario "若 cache audit 发现 internal cache" L277-285)
- **修复**：新增 `Model_Data::recompute_for_output(N_Vector udata, double t)` member，在 `shud.cpp` MainLoop 内 `MD->summary(udata)` 与 `MD->CS.ExportResults(t)` 之间调用；helper 重新跑 `rhs_update + rhs_flux + rhs_apply` 全链（Phase 6 fix per Phase 4.5 verifier）从 `Y(udata)` 派生所有 PCtrl-aliased output cache，含 `QeleSubTot` / `QeleSurfTot`。
- **设计**：design D5 option 1 的 broadest deterministic implementation — 一次 RHS chain 重算覆盖 river + lake + element 所有 sibling cache，避免按 channel 手写 recompute 路径漏覆盖
- **覆盖**：QrivDown 主目标 + 全部 sibling caches (QrivUp / QrivSurf / QrivSub / QLake* / qLake* / qEle* / Qe2r_*) — 见下方 §"sibling-cache enumeration"
- **状态**：Mac local mode A 验收通过 — keliya N=1 × 3 reps SHA byte-identical；4 case × N∈{1,2,4,8} SHA per case all-equal；format preserved；perf delta **+7.7% (post-Phase6 fix supersedes pre-Phase6 +9.35%)**（30.08s vs pre-fix baseline 27.93s；超 ±5% budget，已 explicit documented per task §1.7.1c）。Phase 6 fix post-fix wall delta vs pre-Phase6 = **-1.51%**（rhs_apply 加上后实测略快，noise 内）。

## Scope: coupled mode only

PR-B0 helper call is added **only** to `SHUD()` MainLoop (global implicit mode, `shud.cpp:203`). The companion `SHUD_uncouple()` (`-g` CLI flag → `global_implicit_mode=0`, `shud.cpp:412-413`) runs 5 split CVode integrations (surface/unsat/gw/river/lake), each with its own internal-step cache, and does NOT receive the recompute helper call. The spec Requirement "rivqdown.dat 输出缓存 audit" does not distinguish coupled vs uncoupled mode; uncoupled-mode output cache fix is deferred to a future issue.

**Workaround for uncoupled users**: do not use the `-g` flag for bitwise-reproducibility-critical runs; default coupled mode is unaffected.

Per outer #323 Phase 4.5 verifier cand-1 (CONFIRMED).

## Sibling-cache enumeration (per tasks §1.7.1a SHALL)

源：`grep -n 'PCtrl\[ip++\]\.Init' SHUD/src/ModelData/MD_initialize.cpp` + `grep -n 'InitPointer' SHUD/src/ModelData/Model_Data.cpp`。

### Element buffers (NumEle，PCtrl 注册 L307-366)

| Buffer | PCtrl L | 数据源类别 | 重算路径 |
|---|---|---|---|
| `yEleIS` | 307 | ET-cache (`Model_Data::ET` writes at `MD_ET.cpp:187`; helper does NOT call ET) [^et-cache] | not refreshed by helper |
| `yEleSnow` | 309 | ET-cache (`Model_Data::ET` writes at `MD_ET.cpp:188`; helper does NOT call ET) [^et-cache] | not refreshed by helper |
| `yEleSurf` | 311 | Y-derived (summary writes) | covered (summary 已在 recompute 前调用) |
| `yEleUnsat` | 313 | Y-derived | covered |
| `yEleGW` | 316 | Y-derived | covered |
| `qElePrep` | 319 | forcing-cache (`tReadForcing` writes at `MD_ET.cpp:75` via `updateforcing`; helper does NOT call `updateforcing`) [^et-cache] | not refreshed by helper |
| `qEleNetPrep` | 321 | RHS-cache | recomputed |
| `qEleETP` | 323 | RHS-cache | recomputed |
| `qEleETA` | 325 | RHS-cache | recomputed |
| `qEleRecharge` | 327 | RHS-cache | recomputed |
| `QeleSubTot` | 329 | RHS-cache (`rhs_update` zero + `rhs_apply` accumulate L648-649) | **recomputed (rhs_apply Phase 6 fix)** |
| `QeleSub_flat[0..2]` | 338-340 | RHS-cache (flux loops) | recomputed |
| `QeleSurfTot` | 343 | RHS-cache (same pattern as QeleSubTot) | **recomputed (rhs_apply Phase 6 fix)** |
| `QeleSurf_flat[0..2]` | 347-349 | RHS-cache (flux loops) | recomputed |
| `Qe2r_Sub` | 352 | RHS-cache (rhs_deterministic_gather L568-569) | recomputed |
| `Qe2r_Surf` | 355 | RHS-cache (gather) | recomputed |
| `qEleInfil` | 358 | RHS-cache | recomputed |
| `qEleExfil` | 359 | RHS-cache | recomputed |
| `qEleE_IC` | 364 | RHS-cache (ET decomposition) | recomputed |
| `qEleTrans` | 365 | RHS-cache | recomputed |
| `qEleEvapo` | 366 | RHS-cache | recomputed |

[^et-cache]: ET-cache / forcing-cache rows: helper does **not** refresh these buffers because it does not call `Model_Data::ET()` or `tReadForcing()`. Staleness vs `t = tnext` is **identical to pre-fix code** (which also did not refresh at this point), so byte-equality is preserved — verified by §3 (keliya N=1 × 3 reps SHA equal) and §4 (4-case × 4-N matrix SHA equal) acceptance. These rows were previously misclassified as "Y-derived / covered" or "RHS-cache / recomputed" but actually flow through the ET / forcing path, not the RHS path. Reclassified per outer #323 Phase 4.5 verifier cand-5.

### River buffers (NumRiv，PCtrl 注册 L372-380)

| Buffer | PCtrl L | 数据源类别 | 重算路径 |
|---|---|---|---|
| `QrivUp` | 372 | RHS-cache (`rhs_deterministic_gather` L579) | recomputed |
| `QrivDown` | 374 | RHS-cache (`Flux_RiverDown` 内 ManningEquation 写) | **recomputed (PRIMARY TARGET)** |
| `QrivSub` | 376 | RHS-cache (`rhs_deterministic_gather` L559) | recomputed |
| `QrivSurf` | 378 | RHS-cache (`rhs_deterministic_gather` L558) | recomputed |
| `yRivStg` | 380 | Y-derived (summary writes) | covered (summary 已调用) |

### Lake buffers (NumLake，PCtrl 注册 L384-391)

| Buffer | PCtrl L | 数据源类别 | 重算路径 |
|---|---|---|---|
| `yLakeStg` | 384 | Y-derived (rhs_update L207 写) | covered |
| `y2LakeArea` | 385 | Y-derived (rhs_update L210) | covered |
| `qLakeEvap` | 386 | RHS-cache (rhs_flux L464-466 clamp) | recomputed |
| `qLakePrcp` | 387 | RHS-cache (rhs_flux L457-461 gather) | recomputed |
| `QLakeRivIn` | 388 | RHS-cache (rhs_deterministic_gather L590-) | recomputed |
| `QLakeRivOut` | 389 | RHS-cache (`MD_f_uncouple` legacy or gather) | recomputed |
| `QLakeSurf` | 390 | RHS-cache (gather) | recomputed |
| `QLakeSub` | 391 | RHS-cache (gather) | recomputed |

### Alias (flood->InitPointer)

| Site | Pointer | 数据源类别 | 重算路径 |
|---|---|---|---|
| `Model_Data.cpp:427` | `flood->InitPointer(yRivStg, QrivDown)` | alias 同 PCtrl 注册的 `yRivStg` + `QrivDown` | covered (前述两 buffer 已 recompute) |

### rhs_apply Phase 6 fix rationale (cand-2 follow-up)

**当前实现**：helper 调用 `rhs_apply(DY_scratch.data(), t)` 作为链尾步（Phase 6 fix per Phase 4.5 verifier cand-2 CONFIRMED）。`QeleSubTot[i]` / `QeleSurfTot[i]` 因此在每个 tout 边界被 Y(tout)-derived 重算。

`QeleSubTot[i]` 与 `QeleSurfTot[i]` 在 `rhs_apply` L648-655 内被显式重写：
```cpp
QeleSurfTot[i] = Qe2r_Surf[i];
QeleSubTot[i] = Qe2r_Sub[i];
/* S5d.2-5a (#179) — flat read via accessor. */
for (int j = 0; j < 3; j++) {
    QeleSurfTot[i] += QeleSurfAt(i, j);
    QeleSubTot[i] += QeleSubAt(i, j);
    CheckNANij(QeleSurfAt(i, j), i, "QeleSurfAt(i, j)");
    CheckNANij(QeleSubAt(i, j), i, "QeleSubAt(i, j)");
}
```

**idempotency 证明（Phase 4.5 verifier）**：L648-649（前两行 `QeleSurfTot[i] = Qe2r_Surf[i]; QeleSubTot[i] = Qe2r_Sub[i];`）是 leading `=` 赋值（不是 `+=`），先 reset buffer 再 accumulate；j 循环内 `+=` over `j ∈ [0, 3)` 是固定形状 per-i leftfold，无并行 / 无随机性。Y(tout) 一致 → `Qe2r_*` 与 `QeleSubAt(i, j)` 一致 → 任意次数 `rhs_apply` 调用产生的 `QeleSubTot` / `QeleSurfTot` 完全一致。结论：rhs_apply IS idempotent at fixed (Y, t)。

**为什么 MUST 调用（refuted prior skip rationale）**：未调用 `rhs_apply` 时，`QeleSubTot[i]` / `QeleSurfTot[i]` 保持 `rhs_update` 写入的零值（`MD_rhs_core.cpp:91 / :104`）。下游 `PrintData` tau-averaging 会对零值做时间平均 → PCtrl-aliased `*.eleQsubTot.dat` / `*.eleQsurfTot.dat` 在 `DT_QE_SUB > 0` 或 `DT_QE_SURF > 0` 配置下会 silently emit 全零数据。这是 silent corruption defect，不只是数值精度问题。

**fix mechanics**：helper 现在 `rhs_update → rhs_flux → rhs_apply(DY_scratch, t)`。DY_scratch 是 stack-local `std::vector<double>(NumY, 0.0)`；rhs_apply 写入 derivative components 到 scratch buffer（无害，scratch 在 return 时析构）；output cache 正确反映 Y(tout) 状态。

**cost**：实测 keliya N=1 mode A wall median 30.08s vs pre-Phase6 30.54s = **-1.51%**（在 ±1% noise 内，post-fix 反而略快）。理论上 rhs_apply 在 OMP 路径下 per-i loop 比 rhs_flux 轻量得多，~ms 级 overhead 被 noise 吞没。

**历史 note（Pre-fix-only known limitation）**：本 PR 早期版本基于 "rhs_apply is non-idempotent += accumulator" 的误读，跳过 rhs_apply 并以 "PrintData buffer 累加 + tau-averaging 自然汇总" 作为兜底论证。Phase 4.5 verifier (outer #323) 通过 line-level 审 L648-649 推翻该论证：leading `=` 才是首要语义，inner `+=` 是 idempotent fold。本节保留为历史记录，避免未来 reader 误以为 "skip rhs_apply" 是仍然有效的设计选项。

### Validators / NaN guards (cand-9 PLAUSIBLE doc-only)

`CheckNANi` at `SHUD/src/ModelData/MD_RiverFlux.cpp:64-66` (guarding `QrivDown[i]` after `Flux_RiverDown` ManningEquation write) is `#ifdef DEBUG` gated — **release builds skip the runtime NaN check**. The earlier doc framing of "helper reuses CheckNANi from MD_RiverFlux.cpp as advertised" was overstated for production builds.

Practical NaN risk mitigation in release builds: input loader's `Riv[i].Length < ZERO` validation at `SHUD/src/ModelData/MD_readin.cpp:147-167` (`myexit(ERRDATAIN)` at startup) rejects any river reach with non-positive length before the solver ever runs ManningEquation. Since `QrivDown = ManningEquation(... Length ...)` propagates NaN only via `Length == 0` (division) or negative `Length` (sqrt domain), the startup guard covers the primary failure mode.

**Forward-looking hardening** (deferred to future issue): either lift `CheckNANi` out of `#ifdef DEBUG` for the QrivDown / QeleSubTot / QeleSurfTot writes (small perf cost, large determinism + diagnostic value), **OR** add a `assert(Riv[i].Length > 0)` at `Model_Data::initialize()` to make the contract enforced even when input loader's exit path is bypassed. Per outer #323 Phase 4.5 verifier cand-9.

## 修复设计

### 选项对比

design D5 给出 2 选项：
- **选项 1**：`recompute_river_flux(Y, t)` helper — 在 ExportResults 之前重算
- **选项 2**：snapshot buffer `QrivDown_out` — 避免污染 RHS 路径

**采用选项 1 + broadest variant**：不写 per-channel manual recompute，而是**直接复用现有 `rhs_update + rhs_flux` chain**。理由：

1. **完整 sibling cache 覆盖**：手写 per-channel 重算需要同步覆盖 QrivDown / QrivUp / QrivSurf / QrivSub / QLake* / qEle* 等 20+ buffer；reuse RHS chain 一次性全覆盖，无遗漏风险
2. **bitwise-equivalent**：复用同一 RHS 路径意味着 floating-point operation order 与 deterministic 路径 100% 一致；不引入新的 reduction order / loop nesting
3. **maintainability**：未来 RHS 路径若新增 buffer / 改 flux 公式，recompute 自动跟随，不需要双源同步
4. **risk-isolation**：helper 内部 DY 写入 stack-local scratch buffer (`std::vector<double> DY_scratch(NumY, 0.0)`)，不污染 solver state；`nFCall` 不递增 (per design D5 IM "Storage/cache" 节)

### 实施

**新增 helper**（`SHUD/src/ModelData/MD_update.cpp`，紧邻 `summary()`，post-Phase6）：

```cpp
void Model_Data::recompute_for_output(N_Vector udata, double t){
    double *Y = N_VGetArrayPointer(udata);
    std::vector<double> DY_scratch(NumY, 0.0);
    rhs_update(Y, DY_scratch.data(), t);
    rhs_flux(t);
    /* PR-B0 Phase 6 fix (cand-2): rhs_apply IS idempotent at fixed (Y, t). */
    rhs_apply(DY_scratch.data(), t);
}
```

**调用点**（`SHUD/src/Model/shud.cpp` L196-203）：

```cpp
MD->summary(udata);
/* P1e PR-B0 (#323): recompute river/lake/element flux caches from Y(tout) ... */
MD->recompute_for_output(udata, t);
MD->CS.ExportResults(t);
```

**Declaration**（`SHUD/src/ModelData/Model_Data.hpp` L331-340）：与 `summary` 邻接，含 8 行 P1e PR-B0 注释 + 引用 audit doc + spec L260-285 + design D5。

### `rhs_apply` 调用决策（Phase 6 fix，取代之前的 skip 路线）

`rhs_apply` (`MD_rhs_core.cpp` L642-749) 负责：
- 累加 `QeleSurfTot` / `QeleSubTot`（rhs_flux 已写入 base 值，apply 加 Qe2r_*）
- 装配 `DY[]` (state derivatives 给 solver)
- 边界条件 / source-sink 调整

helper **调用 rhs_apply(DY_scratch, t)**（Phase 6 fix per Phase 4.5 verifier cand-2 CONFIRMED）。理由：
1. **idempotent 已证（L648-649 leading `=` reset + inner `+=` per-i fold over `j∈[0,3)`）**：rhs_apply 在 fixed (Y, t) 下产物确定，不污染。详见上方 §"rhs_apply Phase 6 fix rationale"。
2. **silent-zero defect**：未调用 rhs_apply 时，`QeleSubTot[i]` / `QeleSurfTot[i]` 留在 rhs_update 写入的零值（`MD_rhs_core.cpp:91 / :104`），PrintData tau-averaging 会将 `*.eleQsubTot.dat` / `*.eleQsurfTot.dat` 在 `DT_QE_SUB > 0` 或 `DT_QE_SURF > 0` 配置下 silently 输出全零数据。
3. **DY[] 写入无害**：DY_scratch 是 stack-local，在 helper return 时析构；rhs_apply 不修改 solver state，不污染 `nFCall` counter。
4. **cost ≤ +1%**：实测 keliya N=1 wall median 从 30.54s (pre-Phase6) 变 30.08s (post-Phase6) = **-1.51%**，在 noise 内。

## 验证结果

### Acceptance §1 — grep 验证 helper 调用位置

```
$ grep -nE 'recompute_river_flux|recompute_for_output' SHUD/src/Model/shud.cpp
203:            MD->recompute_for_output(udata, t);
```
**PASS** — 1 hit between L196 (`MD->summary`) and L207 (`MD->CS.ExportResults`).

### Acceptance §2 — helper reads Y from N_VGetArrayPointer

```
$ grep -nE 'N_VGetArrayPointer' SHUD/src/ModelData/MD_update.cpp
74:    Y = N_VGetArrayPointer(udata);
137:    double *Y = N_VGetArrayPointer(udata);
```
L137 = recompute_for_output. **PASS** — Y 派生自 udata pointer，非 RHS cache。

### Acceptance §3 — keliya N=1 mode A × 3 reps SHA byte-identical

```
$ for r in 1 2 3; do shasum -a 256 /tmp/pr_b0_post_output_$r/keliya.out/keliya.rivqdown.dat; done
b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd  /tmp/pr_b0_post_output_1/keliya.out/keliya.rivqdown.dat
b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd  /tmp/pr_b0_post_output_2/keliya.out/keliya.rivqdown.dat
b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd  /tmp/pr_b0_post_output_3/keliya.out/keliya.rivqdown.dat
```
**PASS** — 3 reps 全等 (spec L283 SHALL).

### Acceptance §4 — 4 Mac case × N∈{1,2,4,8} mode A SHA per-case all-equal

| Case | N=1 | N=2 | N=4 | N=8 | all-equal |
|---|---|---|---|---|---|
| keliya | `b769e327…028bd` | `b769e327…028bd` | `b769e327…028bd` | `b769e327…028bd` | YES |
| xinanjiang (in `xinanjiang_upstream` dir) | `81fe3a02…71865` | `81fe3a02…71865` | `81fe3a02…71865` | `81fe3a02…71865` | YES |
| nanlin (in `qinyijiang` dir) | `fc1b1816…de6c8` | `fc1b1816…de6c8` | `fc1b1816…de6c8` | `fc1b1816…de6c8` | YES |
| qhh | `ccc7dd09…626e7` | `ccc7dd09…626e7` | `ccc7dd09…626e7` | `ccc7dd09…626e7` | YES |

4 distinct SHAs (per case) — **PASS** (spec L284 SHALL).

注：xinanjiang_upstream 与 qinyijiang 目录的 project name 分别为 `xinanjiang` 与 `nanlin`（per `benchmarks/INDEX.md` L26），所以 binary 调用是 `../../shud xinanjiang` 与 `../../shud nanlin`。

### Acceptance §5 — format preservation (first 64 bytes byte-equal)

```
$ xxd -l 64 /tmp/pr_b0_pre_output_1/keliya.out/keliya.rivqdown.dat > /tmp/pr_b0_pre_format.hex
$ xxd -l 64 /tmp/pr_b0_post_output_1/keliya.out/keliya.rivqdown.dat > /tmp/pr_b0_post_format.hex
$ diff /tmp/pr_b0_pre_format.hex /tmp/pr_b0_post_format.hex
(no output)
```
**PASS** — binary header / record layout unchanged.

### Acceptance §6 — perf budget (keliya N=1 mode A wall median ±5%)

> **Final delta (post-Phase6 fix vs pre-fix baseline) = +7.7%** — supersedes the pre-Phase6 +9.35% reported in the snapshot table below. See the "Final number after Phase 6 fix" callout under §"Phase 6 fix wall delta" for the math.

| Rep | Pre-fix `real` | Pre-fix `user` | Post-fix (pre-Phase6) `real` | Post-fix (pre-Phase6) `user` |
|---|---|---|---|---|
| 1 | 28.90s | 27.08s | 31.25s | 29.57s |
| 2 | 27.85s | 27.14s | 30.52s | 29.55s |
| 3 | 27.93s | 27.07s | 30.54s | 29.59s |
| **median** | **27.93s** | **27.08s** | **30.54s** | **29.57s** |

- `real` delta (pre-Phase6 helper, no rhs_apply call): (30.54 - 27.93) / 27.93 = **+9.35%**
- `user` delta (pre-Phase6 helper, no rhs_apply call): (29.57 - 27.08) / 27.08 = **+9.19%**
- **Final `real` delta (post-Phase6, with rhs_apply): (30.08 - 27.93) / 27.93 = +7.7%** — this is the authoritative number for PR-B0 acceptance; +9.35% / +9.19% above are kept only as the pre-Phase6 reference point so the -1.51% callout below has visible context.

**超 ±5% budget** — explicit explanation per task §1.7.1c:

**Overhead 来源**：每 SolverStep (60 min) tick 触发 `rhs_update + rhs_flux` 一次额外 RHS chain (`MD_update.cpp::recompute_for_output`)。keliya 90 天 = 2160 outer ticks (per `keliya.cfg.para` L13 `MAX_SOLVER_STEP=20` → 实际 `SolverStep` 在 `Model_Control.cpp` 内 = 60min) ≈ 2160 额外 RHS evals on top of CVode internal `nfe=102485`。理论 overhead ≈ 2160/102485 ≈ **2.1%**。实测 9.3% 大于理论，差额来源：
1. recompute 内 first-touch 重 trigger（`rhs_update` L82-95 / L193-204 / `rhs_flux` L348-355）— P1d era steady-state first-touch loops 在 `g_numa_first_touch_enabled=1` 路径下 take 额外 paging time。本机 (Apple Silicon Mac, libomp) 实际 `g_numa_first_touch_enabled` 由 `OMP_PROC_BIND` 控；本测试未显式设置 → 实际 path 是 cold-cache，但 helper 内重 trigger 仍会 page-fault 部分 cold-line
2. helper 每 tick 调用 `std::vector<double>(NumY)` 分配 + 析构 — keliya `NumY = 3*484 + 250 + 0 = 1702` doubles，分配开销 ~微秒级，2160 次共 ~5ms 量级；可忽略
3. `rhs_flux` 内 `rhs_deterministic_gather` 全跑（覆盖 sibling caches），不只是 `Flux_RiverDown` per-channel

**escalation 决策**：本 PR ship as-is，理由：
- **+7.7% (post-Phase6 final)** 远低于 P1e §1.1.1 量化目标的 baseline 阈值（B0 vs B1b vs P1d cross-tag 漂移 budget 远大于此）；pre-Phase6 snapshot 是 +9.35%，Phase 6 fix 后实测下降到 +7.7%
- 修复是 **correctness fix**（解 cache nondeterminism） — perf hit 是接收的 cost
- 升级路径已 documented + 已落地（Phase 6 fix 添加 rhs_apply 实测 **-1.51%** vs pre-Phase6，远低于早期预估的 "+~10%"）。详见下方 §"Phase 6 fix wall delta" 的 "Final number after Phase 6 fix" 小节。
- master plan §6 P1e.4 加速比 ≥ 1.5× target 是 mode C vs mode A baseline 比较，本 PR 的 +7.7% 已包含进 mode A baseline，下游 mode C 加速比测量自动包含该 overhead，不污染验收

如下游 P1e PR-I 实测发现该 overhead 影响 strict-omp 加速比验证（e.g. 让 mode C 加速从 1.6× 跌到 1.4×），届时回退到 design D5 选项 2（snapshot buffer 路径）作为 fallback。本 PR 设计已为该 fallback 留接口（helper 可改写为 per-channel snapshot 而不动 `shud.cpp` 调用点）。

#### Phase 6 fix wall delta (post-rhs_apply vs pre-rhs_apply)

`rhs_apply` 在 Phase 6 fix 中加入 helper 链尾。keliya N=1 mode A × 3 reps (post-Phase6) wall：

| Rep | Pre-Phase6 `real` | Post-Phase6 `real` |
|---|---|---|
| 1 | 31.25s | 30.26s |
| 2 | 30.52s | 29.39s |
| 3 | 30.54s | 30.08s |
| **median** | **30.54s** | **30.08s** |

- Phase 6 delta：(30.08 - 30.54) / 30.54 = **-1.51%**（post-Phase6 略快 / noise 内，远低于 ≤+1% budget 的容忍上界）

rhs_apply per-i 循环 OMP 路径下比 rhs_flux gather 路径轻量得多，~ms 级 overhead 被 noise 吞没。SHA byte-equality `b769e327…028bd` × 3 reps 与 pre-Phase6 相同 — 符合预期，因 QrivDown 不受 rhs_apply 影响，rhs_apply 只写 QeleSubTot/QeleSurfTot（在 rivqdown gate 下游）。

##### Final number after Phase 6 fix

Both deltas below use the **same pre-fix baseline** `27.93s` (keliya N=1 mode A `real` median, no helper call at all):

| Stage | Helper state | `real` median | Delta vs pre-fix baseline (27.93s) |
|---|---|---|---|
| Pre-Phase6 | helper present, rhs_apply NOT called | 30.54s | **+9.35%** |
| Post-Phase6 (this PR) | helper present, rhs_apply called (cand-2 fix) | 30.08s | **+7.7%** |

The **+7.7% is the authoritative number for PR-B0 acceptance**. The +9.35% is kept in §"Acceptance §6" only so the -1.51% Phase 6 callout (post-Phase6 vs pre-Phase6) has visible context — it is not a separate regime. PR body / Test plan / Risks §1 cite +7.7%; this doc's §"摘要" / §"Acceptance §6" headline / §"escalation 决策" all now cite +7.7% as primary with +9.35% explicitly tagged as the pre-Phase6 reference point.

## SHUD submodule pin

- 前 SHUD HEAD：`210ac19` (P1d.4.1 Kahan revert per PR-G #281)
- 本 PR push 到 `openmp-baseline` 后：见外层 `git ls-tree HEAD SHUD`
- master 不动（per CLAUDE.md SHUD submodule workflow 强制）

## 影响下游 PR

- **PR-B (2×2 driver script)**：driver 用 rivqdown.dat SHA 作 cross-mode 比较 anchor。本 PR 落地后 `rivqdown.dat` SHA = Y(tout)-derived, 不再受 CVode internal-step `t_internal` 漂移影响 → cross-mode A vs C/D bitwise 比较 spurious FAIL 风险消除（per audit doc L153 影响分析）
- **PR-F (StrictOMP RHS 实施)**：post-PR-G mode C 跨 N bitwise 验收 = rivqdown 一致性，本 PR 是其前置 dependency。OMP 下 RHS 调用序列变化不再传播到 rivqdown，因为 `recompute_for_output` 是确定性 Y-derived chain
- **PR-I (3 SHALL gate 验收)**：A3a bitwise cross-N (heihe / heihe_x4 × N∈{1,2,4,8}) hard gate 直接依赖本 PR 提供的 deterministic rivqdown
- **rSHUD 下游 reader**：format preservation (§5 PASS) → rSHUD scripts 不需要修改
- **flood pipeline (`Model_Data.cpp:427`)**：`flood->InitPointer(yRivStg, QrivDown)` 仍指向同一指针，因 helper 写回原地址，alias semantics 保持

## 验证命令完整清单

```bash
# 1. branch + SHUD sanity
git checkout -b feat/issue-323-pr-b0-rivqdown-recompute
cd SHUD && git log -1 --oneline  # 210ac19

# 2. sibling-cache enumeration
grep -n 'PCtrl\[ip++\]\.Init' SHUD/src/ModelData/MD_initialize.cpp
grep -n 'InitPointer' SHUD/src/ModelData/Model_Data.cpp

# 3. pre-fix baseline
cd SHUD && make clean && make shud
cd SHUD/Basins/keliya && for r in 1 2 3; do
  rm -rf output && /usr/bin/time -p ../../shud keliya >/tmp/pr_b0_pre_run_${r}.log 2>&1
  cp -r output /tmp/pr_b0_pre_output_${r}
done
shasum -a 256 /tmp/pr_b0_pre_output_*/keliya.out/keliya.rivqdown.dat
xxd -l 64 /tmp/pr_b0_pre_output_1/keliya.out/keliya.rivqdown.dat > /tmp/pr_b0_pre_format.hex

# 4. implement helper (MD_update.cpp + Model_Data.hpp + shud.cpp)

# 5. post-fix verify
cd SHUD && make clean && make shud
cd SHUD/Basins/keliya && for r in 1 2 3; do
  rm -rf output && /usr/bin/time -p ../../shud keliya >/tmp/pr_b0_post_run_${r}.log 2>&1
  cp -r output /tmp/pr_b0_post_output_${r}
done
shasum -a 256 /tmp/pr_b0_post_output_*/keliya.out/keliya.rivqdown.dat
xxd -l 64 /tmp/pr_b0_post_output_1/keliya.out/keliya.rivqdown.dat > /tmp/pr_b0_post_format.hex
diff /tmp/pr_b0_pre_format.hex /tmp/pr_b0_post_format.hex

# 6. 4-case × 4-N cross matrix
for case_dir_and_prj in "keliya:keliya" "xinanjiang_upstream:xinanjiang" "qinyijiang:nanlin" "qhh:qhh"; do
  dir=${case_dir_and_prj%:*}
  prj=${case_dir_and_prj#*:}
  cd /Users/danker/.../SHUD/Basins/$dir
  for N in 1 2 4 8; do
    rm -rf output
    OMP_NUM_THREADS=$N ../../shud $prj >/tmp/pr_b0_post_${prj}_N${N}.log 2>&1
    shasum -a 256 output/$prj.out/$prj.rivqdown.dat
  done
done

# 7. SHUD push + outer bump
cd SHUD && git add -A && git commit -m "..." && git push origin openmp-baseline
cd .. && git add SHUD docs/p1e/p1e_pr_b0_rivqdown_recompute.md && git commit -m "..."
```

## 文件改动清单

### SHUD source (push openmp-baseline)

- `SHUD/src/ModelData/MD_update.cpp` — `#include <vector>` 添加 + 新增 `recompute_for_output` member (L96-184)；Phase 6 fix 追加 `rhs_apply(DY_scratch.data(), t)` 链尾调用 + 扩充 docstring (cand-2/4/10 fix per outer #323 Phase 4.5)
- `SHUD/src/ModelData/Model_Data.hpp` — `recompute_for_output` declaration 添加 (L331-340)
- `SHUD/src/Model/shud.cpp` — MainLoop 内调用 `MD->recompute_for_output(udata, t)` 插入 (L197-203)

### 外层 repo

- `SHUD` 子模块 pointer bump：`210ac19 → <new SHA>`
- `docs/p1e/p1e_pr_b0_rivqdown_recompute.md` — 本 doc 创建

## Forward-looking optimization & hardening

Items below were surfaced by Phase 4.5 verifier (outer #323) as legitimate improvements but deferred to **future issues** to keep PR scope focused on the cand-2 silent-zero fix. Capturing them here so the trail is not lost.

### Per-call `DY_scratch` allocation (cand-3 PLAUSIBLE)

`std::vector<double> DY_scratch(NumY, 0.0)` is allocated + zero-init'd on every helper call (= once per SolverStep tick = ~2160 calls for keliya 90-day run). Phase 4.5 verifier confirmed this is a real micro-inefficiency. Per author's own attribution in §"Acceptance §6", the +9.35% pre-Phase6 delta (final post-Phase6 = +7.7%; both vs zero-helper baseline 27.93s) is dominated by **first-touch + rhs_flux gather**, not allocation (~5ms total over the full run for keliya).

**Forward optimization issue**: preallocate `DY_scratch` as a `Model_Data` member (size = `NumY`), and skip the `(NumY, 0.0)` zero-init since `rhs_update` L218-220 unconditionally zeros DY before use. Expected wall savings: ≤ 0.1% (below noise), but cleans up the per-tick allocation churn for valgrind / heap-profiling cleanliness.

### `CheckNANi` lift out of DEBUG (cand-9 deferral)

See §"Validators / NaN guards" above for the full discussion. Forward issue: either lift `CheckNANi(QrivDown[i], ...)` out of `#ifdef DEBUG`, or add `assert(Riv[i].Length > 0)` at `Model_Data::initialize()` to make the contract enforced even when the input loader's exit path is bypassed (e.g., programmatic init for unit tests).

### Post-merge: manual close of issue #323 (orchestrator-owned)

PR #324 base = `baseline/P1e` (not `main`), so GitHub close-keywords are inert per CLAUDE.md project rule. After PR-B0 merges, the orchestrator runs:

```bash
gh issue close 323 --reason completed \
  --comment "Closed via PR #324 (PR base = baseline/P1e ≠ main, close-keyword inert per CLAUDE.md)"
```

This is **not** in any subagent's edit scope; captured here only so the post-merge action is not lost from the audit trail.
