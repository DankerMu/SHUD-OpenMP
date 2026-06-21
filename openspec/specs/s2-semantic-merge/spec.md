## Conventions

### Case Scope

本 spec 所有 Requirement 中的 "7 case" / "6 case" / "4-case + heihe + heihe_x4" 等表述统一定义如下：

- **总 case 集合**（7 个）：`keliya` / `xinanjiang_upstream` / `qinyijiang` / `kashigeer` / `qhh` / `heihe` / `heihe_x4`
- **bitwise 验收 case 集合**（6 个）：`kashigeer` 永远 `N/A (deferred-upstream)`（上游 X76 forcing 缺失，CI matrix 排除），所有 bitwise 验收 case-set 实际是 6 个：4-case Mac local (`keliya` / `xinanjiang_upstream` / `qinyijiang` / `qhh`) + 服务器 Slurm (`heihe` / `heihe_x4`)
- **lake-related case** 子集（3 个）：`qhh` / `heihe` / `heihe_x4`（含 lake topology 实体）
- 当某 Requirement 说 "7 case ... SHALL bitwise = B0-tag"，实际验收的是 6 case bitwise + kashigeer N/A
- 所有 case 跑 90-day truncation（`END = START + 90`，CLAUDE.md 项目级铁律）

### Lake-related 输出文件清单

每个 Requirement 涉及 "lake-related .dat" 指以下文件（具体 case 输出目录下）：

- `LakeStg.dat` — lake 存量时间序列
- `LakeQin.dat` — lake 总入流
- `LakeQout.dat` — lake 总出流
- `LakeEvap.dat` — lake 蒸发
- `LakePrcp.dat` — lake 降水
- `LakeSurf.dat` — lake 表面流（来自 element 入流）
- `LakeSub.dat` — lake 地下流（来自 element 入流）
- `LakeRivIn.dat` — lake 来自 river 的入流

只有 lake-related case（`qhh` / `heihe` / `heihe_x4`）会产生这些文件。SHA256 bitwise 验证时 8 个文件 + `rivqdown.dat` 全部 byte-equal `benchmarks/<case>/B0_output/<file>`。

### B0-tag 引用

本 spec 的 "B0-tag" 指 `B0-tag` annotated tag 指向的 commit（`git rev-parse B0-tag^{}` = `884cfb13...`）。验收用 `benchmarks/<case>/B0_output/<file>` 归档与当前 run 输出 SHA256 byte-equal。

## ADDED Requirements

### Requirement: B1a 期间 S1 grep gate 持续 0 hits

工程 SHALL 在 S2/S3/S4 全部子项 PR 中持续保持 S1 已 enforce 的 4 个 grep gate 0 hits（master plan §5 S1d.5）。任一 PR 违反 SHALL 阻塞 merge。

#### Scenario: 4 grep gate 持续 0 hits

- **WHEN** 任意 S2.x / S3a.x / S3b.x / S3c.x / S4.x / capstone PR merge gate 检查
- **THEN** `grep -rn '_OPENMP_ON' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'USE_RHS_CORE' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'N_VDestroy_Serial' SHUD/src/` SHALL = 0 hits
- **THEN** `grep -rn 'SHUD_USE_OPENMP_NVECTOR' SHUD/src/` SHALL 仅有 `Macros.hpp` 1 处 define（S1d.3 引入的合法位置），其它源码文件 0 hits

### Requirement: S2.1 lake vertical 语义合并

工程 SHALL 把 serial `updateLakeElement()` + `fun_Ele_lakeVertical()`（`SHUD/src/ModelData/MD_f.cpp:11-16` + `SHUD/src/ModelData/MD_ElementFlux.cpp:2-17`）的 lake vertical 语义显式纳入 `rhs_core()`，OMP 路径缺失的 lake vertical 调用 SHALL 不被回填进 core——直接采 serial 语义（master plan §5 S2.1 原则）。

#### Scenario: lake vertical 语义合并后 bitwise vs B0

- **WHEN** S2.1 改动在 `baseline/B1a` PR 中 merge
- **AND** 6 case bitwise 集（4-case Mac local + 2 server）跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` SHA256 SHALL 与 `benchmarks/<case>/B0_output/rivqdown.dat` byte-equal
- **THEN** lake-related case（`qhh` / `heihe` / `heihe_x4`）所有 lake-related .dat（按 Conventions 清单）SHALL byte-equal B0_output 归档
- **THEN** CVODE 15-key（不含 `nFCall`）SHALL byte-equal B0-tag 归档

### Requirement: S2.2 lake horizontal 语义合并

工程 SHALL 把 serial `fun_Ele_lakeHorizon()`（`SHUD/src/ModelData/MD_f.cpp:28-29` + `SHUD/src/ModelData/MD_ElementFlux.cpp:18-23`）的 lake horizontal 语义显式纳入 `rhs_core()`，OMP 路径缺失的 lake horizontal 调用 SHALL 不被回填进 core（master plan §5 S2.2 原则）。

#### Scenario: lake horizontal 语义合并后 bitwise vs B0

- **WHEN** S2.2 改动在 `baseline/B1a` PR 中 merge
- **AND** 6 case bitwise 集跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + lake-related case 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** CVODE 15-key SHALL byte-equal B0-tag 归档

### Requirement: S2.3 ET flux 非 lake element 调用

工程 SHALL 在 `rhs_core()` 中显式对非 lake element 调用 `f_etFlux()`（`SHUD/src/ModelData/MD_ET.cpp:167-228`），OMP 路径缺失的 ET flux 调用 SHALL 补回 core 内（master plan §5 S2.3 原则）。

#### Scenario: ET flux 语义补充后 bitwise vs B0 + lake element 不被调用

- **WHEN** S2.3 改动 merge
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** 包含 lake 的 case (`qhh` / `heihe` / `heihe_x4`) 中 lake element 上 `f_etFlux()` SHALL 不被调用（通过 code review 在 `rhs_core()` 内 ET 段确认 `if (Ele[i].iLake > 0) continue;` 或等价 guard 存在；serial 行为保持）

### Requirement: S2.4 river DY 公式采 serial 计算路径

工程 SHALL 把 `rhs_core()` 中 river DY 计算路径统一为 serial 公式（`SHUD/src/ModelData/MD_f.cpp:119-141`）——使用 `length` + `area clamp` + `fun_dAtodY()`；OMP `SHUD/src/ModelData/MD_f_omp.cpp:54-65` 的直接除 `u_TopArea` 公式 SHALL 弃用（master plan §5 S2.4 原则）。

#### Scenario: river DY 采 serial 公式后 bitwise vs B0

- **WHEN** S2.4 改动 merge
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** 对所有 river segment `i`，`DY[i]` SHALL 由 serial 公式 `dArea = ...; clamp(area, ...); fun_dAtodY(dArea, ...)` 计算，不出现 `DY[i] = QrivIn[i] / u_TopArea[i]` 模式

### Requirement: S2.5 lake DY 完整纳入 core

工程 SHALL 把 serial 完整 lake DY 计算（`SHUD/src/ModelData/MD_f.cpp:142-153`）显式纳入 `rhs_core()`，OMP 缺失的 lake DY 计算路径 SHALL 不进入 core（master plan §5 S2.5 原则）。

#### Scenario: lake DY 完整后 bitwise vs B0（lake 与 non-lake case 都验证）

- **WHEN** S2.5 改动 merge
- **AND** 6 case bitwise 集（含非 lake case `keliya` / `xinanjiang_upstream` / `qinyijiang` + lake case `qhh` / `heihe` / `heihe_x4`）跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag（**非 lake case 也要 PASS** —— 即使 S2.5 只动 lake DY，bitwise gate 也要确认非 lake case 没有被 side-effect 污染）
- **THEN** 3 lake case 的所有 lake-related .dat SHALL byte-equal B0_output 归档

### Requirement: S2.6 负状态 clamp 采 serial 不 clamp 语义

工程 SHALL 把 `rhs_core()` 中负状态访问语义统一为 serial 路径（`SHUD/src/ModelData/MD_update.cpp:70-74`）—— `uYsf = Y[iSF]` 不做 `max(0, Y)` clamp；OMP `SHUD/src/ModelData/MD_f_omp.cpp:116-117` 的 `(Y[iSF] >= 0) ? Y[iSF] : 0` clamp SHALL 弃用（master plan §5 S2.6 原则）。

#### Scenario: 负状态 clamp 采 serial 后 bitwise vs B0

- **WHEN** S2.6 改动 merge
- **THEN** 4-case Mac local `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `MD_f_omp.cpp` `f_update_omp` body 内 `uYsf` / `uYus` / `uYriv` 赋值 SHALL 是 `Y[iSF/iUS/iRIV]` 直接 alias，不带 `(>= 0) ? ... : 0` 三元表达式（dormant path 语义对齐 serial `MD_update.cpp:73-77` no-clamp 形式）
- **THEN** PR-2 SHALL **不**改动 `rhs_core()`（serial path `MD_update.cpp:73-77` 已经是 no-clamp 直接 alias 形式；本 PR 仅对齐 dormant OMP path 与 serial 语义）；`uYgw` 在 `f_update_omp:120` 的 `max(0.0, Y[iGW])` 形式不在 S2.6 scope 内（与 iBC 分支共构，属 dormant path 历史 quirk）

### Requirement: S2.7 lake 初始化完整 reset

工程 SHALL 在 `rhs_core()` 中保留 serial 路径（`SHUD/src/ModelData/MD_update.cpp:132-143`）的 lake 完整 flux 清零，OMP 缺失的 lake 初始化路径 SHALL 不进入 core（master plan §5 S2.7 原则）。

**关键约束（PR 顺序）**：S2.7 PR-3 SHALL **早于** S2.1/S2.2/S2.5/S2.11 (lake 合并 PR-4) merge 入 `baseline/B1a`；否则 lake 合并后跨 RHS 调用 stale lake 数组 → bitwise fail。

#### Scenario: lake 完整初始化后 bitwise vs B0

- **WHEN** S2.7 改动 merge
- **AND** lake-related case (`qhh` / `heihe` / `heihe_x4`) 跑 90-day truncation
- **THEN** 3 lake case 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** 每次 `rhs_core()` 入口 lake flux 数组 5 个 (`QLakeSurf` / `QLakeSub` / `QLakeRivIn` / `qLakeEvap` / `qLakePrcp`，注 `q` lowercase 与 SHUD 源码符号一致) SHALL 全部清零（不残留上一步值）
- **NOTE** `QLakeRivOut[i] = 0.;` 在 `MD_rhs_core.cpp:114` 同段也 zero，不在 S2.7 spec scope（lake DY 公式 needs 它，但 list 仅按 master plan §5 S2.7 原则定 5 项）

### Requirement: S2.8 Qe2r/QrivSurf/Sub 清零统一到 update 阶段（DEFER to PR-9）

工程 SHALL 把 `rhs_core()` 中 `Qe2r_Surf` / `Qe2r_Sub` / `QrivSurf` / `QrivSub` 清零操作从 `PassValue()`（`SHUD/src/ModelData/MD_f.cpp:184-225`）移到 `f_update()` 路径（`SHUD/src/ModelData/MD_update.cpp:127-134`）—— 与 serial 路径语义一致（master plan §5 S2.8 原则）。

**重大 ordering 修正（PR-3 #146 implementer 发现）**：S2.8 假设 PassValue 内 4 数组 zero 是 redundant 是 **wrong**。`MD_RiverFlux.cpp:107-108, 121-122` 的 `fun_Seg_surface()` / `fun_Seg_sub()` 在 rhs_flux segment pass (`MD_rhs_core.cpp:185-188` NumSegmt loop) 中 `+=` 到 4 数组：

```cpp
// MD_RiverFlux.cpp:107-108 (fun_Seg_surface body)
QrivSurf[iRiv]    +=  QsegSurf[i];
Qe2r_Surf[iEle]   += -QsegSurf[i];
// MD_RiverFlux.cpp:121-122 (fun_Seg_sub body)
QrivSub[iRiv]  += QsegSub[i];
Qe2r_Sub[iEle] += -QsegSub[i];
```

所以 rhs flow 是：rhs_update entry zero → rhs_flux segment pass `+=` (4 数组 = sum1) → PassValue re-zero (清掉 sum1) → PassValue NumSegmt loop `+=` (4 数组 = sum2 == sum1)。若直接删 PassValue zero，PassValue NumSegmt 在 sum1 之上再 `+=` sum2 → **double-count** → 4-case 4/4 bitwise FAIL vs B0（implementer 实测验证）。

**S2.8 defer 到 PR-9 (#153)**：与 S3a 一起做。S3a "删 PassValue 已覆盖的死 +="（即 `MD_RiverFlux.cpp:107-108, 121-122` 这 4 个 segment-pass `+=`） 删除后，PassValue zero 不再 essential（rhs_update entry zero 是唯一 zero 源），S2.8 deletion 才能做且不破坏 bitwise。PR-3 不实施 S2.8 代码改动；本 spec 标记 DEFER + 详细记录。

#### Scenario: S2.8 DEFER 记录 (PR-3 record-only)

- **WHEN** PR-3 #146 merge
- **THEN** `docs/.s2-pr3-evidence.md` SHALL 记录 S2.8 defer 决定 + `MD_RiverFlux.cpp:107-108,121-122` 4 个 segment-pass `+=` 是 hidden writer + 推理链
- **THEN** PR-3 SHALL **不**改动 `MD_f.cpp` `PassValue()` body（保留 4 个数组 zero）
- **THEN** 4-case Mac local + qhh `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag（PR-3 record-only，0 code change → 0 runtime impact）

#### Scenario: S2.8 实际实施（推迟到 PR-9 #153）

- **WHEN** PR-9 #153 merge（含 S3a 删除 `MD_RiverFlux.cpp:107-108,121-122` 4 个死 `+=`）
- **AND** PR-9 同 PR 内删除 `PassValue()` body L187-188 + L191-194 共 6 行（4 数组 zero + NumEle 空 loop）
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `awk '/^void Model_Data::PassValue\(/,/^}$/' SHUD/src/ModelData/MD_f.cpp | grep -cE '^[[:space:]]*(QrivSurf|QrivSub|Qe2r_Surf|Qe2r_Sub)\[[a-zA-Z_]+\][[:space:]]*=[[:space:]]*0\.?;'` SHALL = 0
- **THEN** `awk '/^void Model_Data::PassValue\(/,/^}$/' SHUD/src/ModelData/MD_f.cpp | grep -cE 'QrivSurf\[[a-zA-Z_]+\][[:space:]]*\+='` SHALL = 0（S3a 删除 segment-pass 死 += 后 PassValue NumSegmt loop 是唯一 += 源；若 fun_Seg_* 仍有 += 视为 S3a 未完成）

### Requirement: S2.9 `f_applyDY_omp` data race 修复 + 多线程 repeat-determinism

工程 SHALL 修复 `f_applyDY_omp()` 中 `area` / `isf` / `ius` / `igw` 4 个局部变量声明位置（`SHUD/src/ModelData/MD_f_omp.cpp:10-16`）——从 parallel region 外 `default(shared)` 改为 for 循环内部局部声明或显式 `private`（master plan §5 S2.9 原则 + §4.6）。**注意**：data race 只在多线程 OpenMP 执行时显式 manifest；单线程 baseline bitwise 不能反向证明 race 已修；本 Scenario 同时验证两路。

**B1a Dormant Path Limitation**：`MD_f_omp.cpp` 在 B1a 范围内是 dormant TU——默认 Makefile (`SHUD/Makefile:366-370`) 通过 `filter-out` 排除该文件；即使 `SHUD_LEGACY_OMP_RHS=1` build 编译进去，`f_update_omp / f_loop_omp / f_applyDY_omp` 3 个符号也 NOT reachable at runtime from `f()`（`SHUD/Makefile:197-200`、`SHUD/src/Model/Macros.hpp:32-35`）。故 S2.9 race fix **本质是 dormant structural fix**：代码对齐 race-free 形态供 P1+ 重启 OMP RHS path 使用，**B1a 范围内 race 修复无法被 runtime exercise**。Config C 的 `make shud_omp` 默认 `LEGACY_RHS=0`，跑的是 `rhs_core()` serial path + OpenMP NVector backend；Config C repeat-determinism **仅证** rhs_core serial + NVector backend 多线程 deterministic，**不证** `f_applyDY_omp` race 修复有效。任何"Config C 验证 race 已修"叙述视为 false claim 需 refute。

#### Scenario: data race 修复后单线程 bitwise + 多线程 repeat-determinism

- **WHEN** S2.9 改动 merge
- **AND** **Config A**（`make shud` 默认 LEGACY_RHS=0，serial 二进制，无 `_OPENMP`）：4-case Mac local + qhh 跑 90-day truncation
- **THEN** Config A：4-case + qhh `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag（default build 不编译 `MD_f_omp.cpp`，PR-2 改动 0 runtime 影响）
- **AND** **Config B**（`make shud_omp` 默认 LEGACY_RHS=0 + `OMP_NUM_THREADS=1`）：4-case Mac local 跑 90-day truncation
- **THEN** Config B：4-case `rivqdown.dat` SHA256 **NOT REQUIRED** to bitwise = B0-tag — `make shud_omp` 隐式启用 `SHUD_USE_OPENMP_NVECTOR=1` (S1d.3) 切换到 OpenMP NVector backend，浮点 reduction order 与 Serial NVector backend (B0 baseline) 不一致是 **S1d.3 引入的 pre-existing baseline gap**，不在 PR-2 dormant-path structural fix 修复范围。本 PR 验证手段：pre-PR-2 `openmp-baseline` HEAD stash test 同样 4/4 FAIL 同 SHA256，confirm PR-2 0 contribute。Config B "single-thread OpenMP NVector backend SHALL bitwise = Serial NVector backend" 是 NVector backend 浮点 reduction-order 投资决策，**defer 到 P 阶段**（如需 strict bitwise = B0 across backends，要修改 OpenMP NVector backend reduction primitive 或采用 Kahan/pairwise summation；属 P1+ 后续 backend hardening 范畴）。
- **THEN** Config B：建议替换为 `make shud_omp` 4-case 跑 2 次 **repeat-determinism** byte-equal（与 Config C 同形式但 T=1），验证 OpenMP NVector backend 单线程 deterministic（最低 sanity；orchestrator 实施时如 already covered by Config C 也可记 N/A 转 Config C）
- **AND** **Config C**（`make shud_omp` 默认 LEGACY_RHS=0 + `OMP_NUM_THREADS=4` + `OMP_PROC_BIND=close OMP_PLACES=cores`）：4-case Mac local 跑 90-day truncation **2 次（同样输入）**
- **THEN** Config C：两次跑出的 `rivqdown.dat` SHA256 SHALL byte-equal **（验证 OpenMP NVector backend 4 线程 + `rhs_core()` serial path 下 repeat-determinism；不强制 = B0；多线程 ≠ B0 是 expected）；本测仅证 NVector backend 多线程 deterministic，不 exercise `f_applyDY_omp`，故 不证 S2.9 race fix 有效（详 dormant path Limitation 段）**
- **THEN** `awk '/^void Model_Data::f_applyDY_omp\(/,/^}$/' SHUD/src/ModelData/MD_f_omp.cpp` 函数 body 内：4 vars `area / isf / ius / igw` 声明 SHALL 在 for body 内 declare-at-use 或在 `#pragma omp parallel ... private(...)` clause 内列出；`awk '/^void Model_Data::f_applyDY_omp\(/,/#pragma omp parallel/' SHUD/src/ModelData/MD_f_omp.cpp | grep -cE '^[[:space:]]*(double|int)[[:space:]]+(area|isf|ius|igw)\b'` SHALL = 0（parallel region 前段无 4 vars 声明） **（awk pattern 必须含 `Model_Data::` C++ qualified prefix + 锚定 `/^}$/` close brace；同 S2.10/S2.14 PR-1 cand-04 lesson）**
- **THEN** `make clean && make shud SHUD_LEGACY_OMP_RHS=1` SHALL 编译通过（dormant TU 重整后 LEGACY=1 build 仍 link 兼容；3 个 `_omp` 符号仍由 header 声明可见）
- **THEN** `SHUD/src/ModelData/Model_Data.hpp:261-263` 3 个 `_omp` 函数声明 SHALL 保留（PR-8 capstone 删除前必须可编译 `LEGACY_RHS=1` build）
- **THEN** snapshot bitwise（12 张，4-case × t=1d/10d/100d）via `tools/compare_snapshot/compare_snapshot` SHALL exit 0（继承 PR-1 PATH A 决定，不重判定）

### Requirement: S2.10 `updateforcing()` 孤立 `#pragma omp for` 移除

工程 SHALL 从 `SHUD/src/ModelData/MD_ET.cpp:12-14` 移除 `updateforcing()` 内的孤立 `#pragma omp for`（无外层 parallel region 包裹）。**SHALL 不**用 `#pragma omp parallel` 包裹此 loop——`movePointer()` 必须保持串行调用语义（master plan §5 S2.10 原则 + S5a 串行契约）。

#### Scenario: 孤立 omp for 移除后 bitwise vs B0 + 串行契约保持（函数体 scope）

- **WHEN** S2.10 改动 merge
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `awk '/^void Model_Data::updateforcing/,/^}/' SHUD/src/ModelData/MD_ET.cpp | grep -cE '^[[:space:]]*#pragma omp'` SHALL = 0 **（注意：(1) awk pattern 必须含 `Model_Data::` C++ qualified prefix，否则 vacuous match；(2) anchored regex `^[[:space:]]*#pragma omp` 是 directive count，排除 narrative comment 内字面值 — fix cand-04 ack PR-1 verifier round 1；(3) 是 `updateforcing()` 函数体内 grep，不是整个 `MD_ET.cpp` 文件 grep；其它函数 `ET()` 的 pragma 在 S2.14 处理）**
- **THEN** `tsd_weather[i].movePointer(t)` 调用 (NumForc loop) SHALL **不**被任何 `#pragma omp parallel` 包裹（code review）

### Requirement: S2.11 lake element DY=0 显式置零

工程 SHALL 在 `rhs_core()` 中显式对 lake element 置零 `DY[i]` / `DY[ius]` / `DY[igw]`（`SHUD/src/ModelData/MD_f.cpp:108-112`），OMP 缺失的此置零操作 SHALL 补进 core（master plan §5 S2.11 原则）。

#### Scenario: lake element DY=0 后 bitwise vs B0（lake 与 non-lake 都验证）

- **WHEN** S2.11 改动 merge
- **AND** 6 case bitwise 集（含非 lake case + lake case）跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag（**非 lake case 也要 PASS** —— 即使 S2.11 只动 lake element，bitwise gate 也要确认非 lake case 没有被 side-effect 污染）
- **THEN** 3 lake case 的 `rivqdown.dat` + 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** 对每个 lake element index `i_lake`，`DY[i_lake]` / `DY[i_lake+ius_offset]` / `DY[i_lake+igw_offset]` SHALL 在 `rhs_core()` 出口为 0.0

### Requirement: S2.12 uncoupled 路径 clamp 差异记录

工程 SHALL 在 `docs/s2_semantic_diff_report.md` 记录 `f_updatei()`（uncoupled 路径，`SHUD/src/ModelData/MD_update.cpp:3-59`）与 `f_update()`（coupled 路径，L60-147）之间 `max(0, Y)` clamp 差异；当前方案以 coupled 路径为并行主线，uncoupled 暂不纳入并行改造（master plan §5 S2.12 原则）。

#### Scenario: uncoupled clamp 差异 record-only

- **WHEN** S2.12 record 在 PR-7a 中 merge
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 包含一节 "S2.12 uncoupled 路径 clamp 差异" 按 schema 字段写明（详见 `docs/s2_semantic_diff_report.md` schema Requirement）
- **THEN** SHUD 源码 `MD_update.cpp` `f_updatei()` 函数 body SHALL 未被本 change 改动

### Requirement: S2.13 全局变量裸指针记录

工程 SHALL 在 `docs/s2_semantic_diff_report.md` 记录 `uYsf` / `uYus` / `uYgw` / `uYriv` / `uYlake` / `timeNow` 6 个全局变量裸指针的当前使用模式（读/写时机），不做迁移；迁移延迟到 P8+ / LibSHUD / Reentrant RHS 阶段（master plan §5 S2.13 原则）。

#### Scenario: 全局变量裸指针 record-only

- **WHEN** S2.13 record 在 PR-7a 中 merge
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 包含一节 "S2.13 全局变量裸指针" 按 schema 字段写明每个变量的 `读 by:` / `写 by:` / `赋值时机:` + 处置决策 "推迟到 P8+ Reentrant RHS"
- **THEN** SHUD 源码 `shud.cpp` / `Macros.hpp` 全局变量定义 SHALL 未被本 change 改动

### Requirement: S2.14 `ET()` 孤立 `#pragma omp for` 移除 + 16 个标量移入循环体

工程 SHALL 从 `SHUD/src/ModelData/MD_ET.cpp:106-165` 移除 `ET()` 内孤立 `#pragma omp for`；同时把 16 个 element-local 标量（`T`, `LAI`, `MF`, `prcp`, `snFrac`, `snAcc`, `snMelt`, `snStg`, `icAcc`, `icEvap`, `icStg`, `icMax`, `vgFrac`, `ta_surf`, `ta_sub`, `i`）从循环外移入 for 循环 body 内部；`DT_min` 为循环不变量保持共享。**SHALL 不**用 `#pragma omp parallel` 包裹 `ET()`——并行化属 P2a 阶段（master plan §5 S2.14 原则 + §4.11）。

#### Scenario: ET() omp for 移除 + 标量整理后 bitwise vs B0（函数体 scope）

- **WHEN** S2.14 改动 merge
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `awk '/^void Model_Data::ET\(/,/^}/' SHUD/src/ModelData/MD_ET.cpp | grep -cE '^[[:space:]]*#pragma omp'` SHALL = 0 **（同 S2.10 注：awk pattern 必须含 `Model_Data::` C++ qualified prefix；anchored regex 排除 narrative comment 字面值 — fix cand-04 ack PR-1 verifier round 1；是 `ET()` 函数体内 grep，不是整个文件）**
- **THEN** 16 个 element-local 标量 SHALL 全部在 `for (int i = 0; i < NumEle; ++i) { ... }` body 内部 declare（grep `awk '/^void Model_Data::ET\(/,/^}/'` 提取 `ET()` 函数体后，标量声明位置 SHALL 在 `for (` 之后）

### Requirement: S2.15 `AccTemperature.getACC()` 除零 record + B1b 修复占位

工程 SHALL 在 `docs/s2_semantic_diff_report.md` 记录 `AccTemperature.hpp:60-62` 的 `que.size()==0` 时除零 → NaN 风险；本 change（B1a 阶段）**不**修复（修会改输出违反 B1a==B0）；修复推迟到 B1b S6a 阶段（master plan §5 S2.15 原则）。

#### Scenario: AccTemperature 除零 record-only

- **WHEN** S2.15 record 在 PR-7a 中 merge
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 包含一节 "S2.15 AccTemperature 除零" 按 schema 字段写明 file:line + 风险描述 + 处置决策 "B1b/S6a 修复（加 guard `que.empty() ? 0.0 : ACC/que.size()`）"
- **THEN** SHUD 源码 `AccTemperature.hpp` `getACC()` 函数 body SHALL 未被本 change 改动

### Requirement: S2.16 OpenMP N_Vector 当前使用已由 S1d.3–S1d.5 解决

工程 SHALL 在 `docs/s2_semantic_diff_report.md` 确认 `shud.cpp:58-59` 的 `N_VNew_OpenMP` 使用已由 S1d.3 (`SHUD_USE_OPENMP_NVECTOR` 宏) + S1d.4 (`N_VGetArrayPointer` 统一) + S1d.5 (`N_VDestroy` generic) 三个步骤解决；S2 阶段**无需额外处理**（master plan §5 S2.16 原则 + §4.21 + §4.13）。

#### Scenario: N_Vector 当前使用确认零 action

- **WHEN** S2.16 record 在 PR-7a 中 merge
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 包含一节 "S2.16 OpenMP N_Vector 当前使用" 按 schema 字段写明 "已由 S1d.3-S1d.5 解决，零 action"
- **THEN** S1 grep gate `N_VDestroy_Serial 0 hits` SHALL 持续生效（B1a 期间 grep gate Requirement 已覆盖）

### Requirement: S2.17 `fun_Ele_sub()` lake 分支 assert + 公式变更推迟

工程 SHALL 在 `SHUD/src/ModelData/MD_ElementFlux.cpp` `fun_Ele_sub()` 的 lake 分支入口（L107 附近）加 `assert(inabr >= 0)` 防御性检查；SHALL 在 `docs/s2_semantic_diff_report.md` 记录 lake element `u_effKH` 赋值来源和物理意义分析；本 change（B1a 阶段）**不**改公式（master plan §5 S2.17 原则 + §4.18）。

#### Scenario: S2.17 assert 加入 + DEBUG 6 case 不 trigger + record diff

- **WHEN** S2.17 改动在 PR-7b merge
- **AND** 6 case bitwise 集（4-case Mac local + heihe + heihe_x4 server）DEBUG build (`make shud DEBUG=1`) 跑 90-day truncation
- **THEN** `assert(inabr >= 0)` SHALL **不**在任何 case trigger（验证 `lakenabr[]` 数据流确保 `inabr >= 0`）；若 trigger SHALL 升级为 BLOCKER 阻塞 PR-7b merge，回到 brainstorming 重新决定是否需要在 B1a 内修公式（违反 B1a==B0 → 必须独立 B1b 阶段）
- **THEN** RELEASE build 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 包含一节 "S2.17 fun_Ele_sub lake 分支" 按 schema 字段写明 file:line + 处置决策 "S2 加 assert，公式变更推迟到 B1b/S6a"

### Requirement: `docs/s2_semantic_diff_report.md` schema

工程 SHALL 新建 `docs/s2_semantic_diff_report.md` markdown 文件，包含 S2.12 / S2.13 / S2.15 / S2.16 / S2.17 共 5 项 record-only 子项的处置决策。每项 SHALL 按以下 7 字段 schema 写入。

#### Scenario: schema 字段完整 + 5 项齐全

- **WHEN** PR-7a (record-only 4 项) + PR-7b (S2.17 补完) merge
- **THEN** `docs/s2_semantic_diff_report.md` 顶层 SHALL 包含 5 个 markdown section: `## S2.12 ...` / `## S2.13 ...` / `## S2.15 ...` / `## S2.16 ...` / `## S2.17 ...`
- **THEN** 每个 section SHALL 包含以下 7 字段（按顺序）:
  1. `**Master plan reference:**` — verbatim 引用 master plan §5 S2.x 段落开头一句话
  2. `**File:line(s):**` — 涉及代码的精确 file:line（如 `SHUD/src/ModelData/MD_update.cpp:3-59`）
  3. `**Risk / observation:**` — 当前代码的风险或观察（如 "uncoupled 路径 clamp 与 coupled 路径不一致"）
  4. `**Decision (record-only / defer to B1b / etc.):**` — 处置决策（5 项均为 record-only / defer 类）
  5. `**Reason for deferral:**` — 推迟到 B1b/S6a/P8+ 的理由（与 master plan §5 原则一致）
  6. `**Verification:**` — 如何确认 record-only 真未改动代码（`git diff B0-tag <PR-7-commit> -- <file>` SHALL 无 diff）
  7. `**Linked future issue (if any):**` — 若已开 B1b 跟踪 issue，列出 issue 编号；否则 "无（B1a 完成后再开 B1b Epic）"
- **THEN** `git diff B0-tag <PR-7b 最终 commit> -- SHUD/src/` SHALL 不包含 5 项 record-only 涉及的 file:line 范围内的实质改动（S2.17 assert 加入除外）

### Requirement: S2 capstone — 删除 `MD_f_omp.cpp` + 退役 `SHUD_LEGACY_OMP_RHS`

工程 SHALL 在 S2.1–S2.17 全部完成后，作为 S2 capstone PR-8 删除 `SHUD/src/ModelData/MD_f_omp.cpp` 整文件，删除 `SHUD/src/ModelData/Model_Data.hpp` 中 `f_update_omp` / `f_loop_omp` / `f_applyDY_omp` 三个声明，删除**全 SHUD 源码树**中所有对 `_omp` 三函数的调用（不仅 `Model/f.cpp`），退役 `SHUD_LEGACY_OMP_RHS` 宏定义和所有 `#ifdef`/`#ifndef` 使用，同步 `SHUD/Makefile` 删除 `LEGACY_RHS=1` build target，同步 `.github/workflows/serial-baseline.yml` matrix 删除 `LEGACY_RHS=1` 轴并新增 `MD_f_omp.cpp 0 hits` grep gate（master plan §5 S2 capstone）。

#### Scenario: S2 capstone 后 _omp 路径 0 hits + bitwise vs B0（tree-wide scope）

- **WHEN** S2 capstone PR-8 merge
- **AND** PR-1 / PR-2 / PR-3 / PR-4 / PR-5 / PR-6 / PR-7a / PR-7b 都已 merge 入 `baseline/B1a`（前置）
- **THEN** `ls SHUD/src/ModelData/MD_f_omp.cpp` SHALL exit non-zero（文件不存在）
- **THEN** `grep -rn 'f_update_omp\|f_loop_omp\|f_applyDY_omp' SHUD/src/` SHALL = 0 hits **（注意：tree-wide grep，包括 `SHUD/src/Model/f.cpp` 历史注释、`SHUD/src/ModelData/Model_Data.hpp` 头文件声明、所有头文件、所有 cpp 文件）**
- **THEN** `grep -rn 'SHUD_LEGACY_OMP_RHS' SHUD/src/ SHUD/Makefile` SHALL = 0 hits
- **THEN** `grep -n 'LEGACY_RHS' SHUD/Makefile` SHALL = 0 hits
- **THEN** `.github/workflows/serial-baseline.yml` matrix SHALL 只跑 `LEGACY_RHS=0` 单轴（不再有 `LEGACY_RHS=1` 配置）+ 新增 `MD_f_omp.cpp 0 hits` grep gate step
- **THEN** 6 case bitwise 集 `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `docs/s2_semantic_diff_report.md` SHALL 完整（5 项 record-only + S2 capstone 时间线 section 追加）
