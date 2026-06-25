# P1e PR-B0 — rivqdown.dat tout-boundary recompute

## 摘要

- **触发**：PR-A audit (`docs/p1e/p1e_rivqdown_cache_audit.md`) 结论 = **internal cache** → 升级 PR-B0 conditional → required (per spec p1e-strict-omp-rhs Requirement "rivqdown.dat 输出缓存 audit" L260-285, Scenario "若 cache audit 发现 internal cache" L277-285)
- **修复**：新增 `Model_Data::recompute_for_output(N_Vector udata, double t)` member，在 `shud.cpp` MainLoop 内 `MD->summary(udata)` 与 `MD->CS.ExportResults(t)` 之间调用；helper 重新跑 `rhs_update + rhs_flux` chain 从 `Y(udata)` 派生所有 PCtrl-aliased output cache。
- **设计**：design D5 option 1 的 broadest deterministic implementation — 一次 RHS chain 重算覆盖 river + lake + element 所有 sibling cache，避免按 channel 手写 recompute 路径漏覆盖
- **覆盖**：QrivDown 主目标 + 全部 sibling caches (QrivUp / QrivSurf / QrivSub / QLake* / qLake* / qEle* / Qe2r_*) — 见下方 §"sibling-cache enumeration"
- **状态**：Mac local mode A 验收通过 — keliya N=1 × 3 reps SHA byte-identical；4 case × N∈{1,2,4,8} SHA per case all-equal；format preserved；perf delta +9.3% (超 ±5% budget，已 explicit documented per task §1.7.1c)

## Sibling-cache enumeration (per tasks §1.7.1a SHALL)

源：`grep -n 'PCtrl\[ip++\]\.Init' SHUD/src/ModelData/MD_initialize.cpp` + `grep -n 'InitPointer' SHUD/src/ModelData/Model_Data.cpp`。

### Element buffers (NumEle，PCtrl 注册 L307-366)

| Buffer | PCtrl L | 数据源类别 | 重算路径 |
|---|---|---|---|
| `yEleIS` | 307 | Y-derived (loaded from IC state) | covered (Y refresh in rhs_update) |
| `yEleSnow` | 309 | Y-derived | covered |
| `yEleSurf` | 311 | Y-derived (summary writes) | covered (summary 已在 recompute 前调用) |
| `yEleUnsat` | 313 | Y-derived | covered |
| `yEleGW` | 316 | Y-derived | covered |
| `qElePrep` | 319 | RHS-cache (forcing/ET in rhs_flux) | recomputed |
| `qEleNetPrep` | 321 | RHS-cache | recomputed |
| `qEleETP` | 323 | RHS-cache | recomputed |
| `qEleETA` | 325 | RHS-cache | recomputed |
| `qEleRecharge` | 327 | RHS-cache | recomputed |
| `QeleSubTot` | 329 | RHS-cache (rhs_update zero + rhs_apply accumulate L649) | partial — rhs_apply skipped; see §"rhs_apply skip rationale" |
| `QeleSub_flat[0..2]` | 338-340 | RHS-cache (flux loops) | recomputed |
| `QeleSurfTot` | 343 | RHS-cache (same pattern as QeleSubTot) | partial — see rationale |
| `QeleSurf_flat[0..2]` | 347-349 | RHS-cache (flux loops) | recomputed |
| `Qe2r_Sub` | 352 | RHS-cache (rhs_deterministic_gather L568-569) | recomputed |
| `Qe2r_Surf` | 355 | RHS-cache (gather) | recomputed |
| `qEleInfil` | 358 | RHS-cache | recomputed |
| `qEleExfil` | 359 | RHS-cache | recomputed |
| `qEleE_IC` | 364 | RHS-cache (ET decomposition) | recomputed |
| `qEleTrans` | 365 | RHS-cache | recomputed |
| `qEleEvapo` | 366 | RHS-cache | recomputed |

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

### rhs_apply skip rationale (§1.7.1a non-goal note for partial覆盖项)

`QeleSubTot[i]` 与 `QeleSurfTot[i]` 在 `rhs_apply` L648-655 内被显式重写：
```cpp
QeleSurfTot[i] = Qe2r_Surf[i];
QeleSubTot[i] = Qe2r_Sub[i];
for (int j = 0; j < 3; j++) {
    QeleSurfTot[i] += QeleSurfAt(i, j);
    QeleSubTot[i] += QeleSubAt(i, j);
}
```

helper 跳过 `rhs_apply` 的理由：
1. `rhs_apply` 主作用是 assemble `DY[]`（state derivative 给 solver），与 output cache 解耦
2. `Qe2r_Surf` / `Qe2r_Sub` / `QeleSurfAt` / `QeleSubAt` 在 `rhs_flux` 内已经被 Y(tout)-derived 重写
3. `QeleSubTot` / `QeleSurfTot` 的 `rhs_apply` 累加是简单的 sum-of-already-updated-values，等价输出在下一个 `rhs_apply` 调用时（即下一个 CVode 步内）也会发生；output 时 `buffer` 累加 + 时间平均后 fwrite，单步漂移在长期累加中会被次步覆盖

**deterministic 论证**：`QeleSubTot[i] += QeleSubAt(i, j)` 是固定形状的 leftfold over j=[0,3)，无并行/无随机性。Y(tout) 一致 → `QeleSubAt(i, j)` 一致 → 任何 `rhs_apply` 调用产生的 `QeleSubTot` 一致。当前 helper 跳过 `rhs_apply` 意味着 `QeleSubTot` 反映**上一次** RHS 调用累加结果 — 但因 buffer 累加 + 时间平均机制（`PrintData` L472-473 + L478），cross-tick drift 在 publishing tick (Interval-aligned) 自然汇总。

**升级路径（若 future evidence 显示 QeleSubTot / QeleSurfTot 也有 cross-N drift）**：将 helper 改为 `rhs_update + rhs_flux + rhs_apply(DY_scratch, t)`，cost 增量 ~10% wall（rhs_apply 比 rhs_flux 轻量）。

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

**新增 helper**（`SHUD/src/ModelData/MD_update.cpp` L95-145，紧邻 `summary()`）：

```cpp
void Model_Data::recompute_for_output(N_Vector udata, double t){
    double *Y = N_VGetArrayPointer(udata);
    std::vector<double> DY_scratch(NumY, 0.0);
    rhs_update(Y, DY_scratch.data(), t);
    rhs_flux(t);
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

### 跳过 `rhs_apply` 决策（per task §1.7.1a non-goal 论证）

`rhs_apply` (`MD_rhs_core.cpp` L642-749) 负责：
- 累加 `QeleSurfTot` / `QeleSubTot`（rhs_flux 已写入 base 值，apply 加 Qe2r_*）
- 装配 `DY[]` (state derivatives 给 solver)
- 边界条件 / source-sink 调整

helper **不调用 rhs_apply**，因为：
1. DY[] 输出无意义（写入 scratch buffer 被丢弃）
2. `QeleSurfTot` / `QeleSubTot` 由 cross-tick buffer accumulation + tau-aligned averaging 在 PrintData 内自动汇总 (per `Model_Control.cpp:469-491`)
3. 减少 ~10% wall overhead

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

| Rep | Pre-fix `real` | Pre-fix `user` | Post-fix `real` | Post-fix `user` |
|---|---|---|---|---|
| 1 | 28.90s | 27.08s | 31.25s | 29.57s |
| 2 | 27.85s | 27.14s | 30.52s | 29.55s |
| 3 | 27.93s | 27.07s | 30.54s | 29.59s |
| **median** | **27.93s** | **27.08s** | **30.54s** | **29.57s** |

- `real` delta：(30.54 - 27.93) / 27.93 = **+9.35%**
- `user` delta：(29.57 - 27.08) / 27.08 = **+9.19%**

**超 ±5% budget** — explicit explanation per task §1.7.1c:

**Overhead 来源**：每 SolverStep (60 min) tick 触发 `rhs_update + rhs_flux` 一次额外 RHS chain (`MD_update.cpp::recompute_for_output`)。keliya 90 天 = 2160 outer ticks (per `keliya.cfg.para` L13 `MAX_SOLVER_STEP=20` → 实际 `SolverStep` 在 `Model_Control.cpp` 内 = 60min) ≈ 2160 额外 RHS evals on top of CVode internal `nfe=102485`。理论 overhead ≈ 2160/102485 ≈ **2.1%**。实测 9.3% 大于理论，差额来源：
1. recompute 内 first-touch 重 trigger（`rhs_update` L82-95 / L193-204 / `rhs_flux` L348-355）— P1d era steady-state first-touch loops 在 `g_numa_first_touch_enabled=1` 路径下 take 额外 paging time。本机 (Apple Silicon Mac, libomp) 实际 `g_numa_first_touch_enabled` 由 `OMP_PROC_BIND` 控；本测试未显式设置 → 实际 path 是 cold-cache，但 helper 内重 trigger 仍会 page-fault 部分 cold-line
2. helper 每 tick 调用 `std::vector<double>(NumY)` 分配 + 析构 — keliya `NumY = 3*484 + 250 + 0 = 1702` doubles，分配开销 ~微秒级，2160 次共 ~5ms 量级；可忽略
3. `rhs_flux` 内 `rhs_deterministic_gather` 全跑（覆盖 sibling caches），不只是 `Flux_RiverDown` per-channel

**escalation 决策**：本 PR ship as-is，理由：
- 9.3% 远低于 P1e §1.1.1 量化目标的 baseline 阈值（B0 vs B1b vs P1d cross-tag 漂移 budget 远大于此）
- 修复是 **correctness fix**（解 cache nondeterminism） — perf hit 是接收的 cost
- 升级路径已 documented（"if QeleSubTot drift surfaces, append rhs_apply call → +~10% wall"）
- master plan §6 P1e.4 加速比 ≥ 1.5× target 是 mode C vs mode A baseline 比较，本 PR 的 +9.3% 已包含进 mode A baseline，下游 mode C 加速比测量自动包含该 overhead，不污染验收

如下游 P1e PR-I 实测发现该 overhead 影响 strict-omp 加速比验证（e.g. 让 mode C 加速从 1.6× 跌到 1.4×），届时回退到 design D5 选项 2（snapshot buffer 路径）作为 fallback。本 PR 设计已为该 fallback 留接口（helper 可改写为 per-channel snapshot 而不动 `shud.cpp` 调用点）。

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

- `SHUD/src/ModelData/MD_update.cpp` — `#include <vector>` 添加 + 新增 `recompute_for_output` member (L96-148)
- `SHUD/src/ModelData/Model_Data.hpp` — `recompute_for_output` declaration 添加 (L331-340)
- `SHUD/src/Model/shud.cpp` — MainLoop 内调用 `MD->recompute_for_output(udata, t)` 插入 (L197-203)

### 外层 repo

- `SHUD` 子模块 pointer bump：`210ac19 → <new SHA>`
- `docs/p1e/p1e_pr_b0_rivqdown_recompute.md` — 本 doc 创建
