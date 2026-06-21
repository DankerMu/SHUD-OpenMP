## Purpose

记录 B1a S3 阶段 deterministic gather 重构契约（PassValue → rhs_deterministic_gather + 死代码删除）。

## Conventions

本 spec 沿用 `s2-semantic-merge` 的 Conventions（Case Scope / Lake-related 输出文件清单 / B0-tag 引用），不重复列出。

### S3b → S3c 过渡期 lake gather 处置

PR-9（S3a/S3b）把 lake 共享写拆为 per-edge / per-element contribution slot 后，到 PR-11（S3c.3 PassValue 重写）merge 前的 window 内，lake 汇总（`QLakeSurf` / `QLakeSub` / `QLakeRivIn` / `qLakeEvap` / `qLakePrcp`）SHALL 由 **`PassValue()` 内部扩出的临时 gather 段**完成：按 per-edge / per-element slot 求和回写 lake 数组，顺序仍 = B0 serial loop order。PR-11 S3c.3 删 `PassValue()` 时同时退役此临时 gather。

理由：S3a/S3b 拆 (compute) 与 S3c 重写 (gather) 在两个 PR，中间 phase 不能让 lake 数组失去 reset / 累加，否则 bitwise 立即 fail。S3c 依赖 S4 adjacency list（见 design.md D5），所以 PR 顺序是 PR-9 (S3a/S3b + temp gather) → PR-10 (S4) → PR-11 (S3c)。

## Requirements

### Requirement: S3a.1 删除 `QrivSurf[iRiv] += QsegSurf[i]` 死代码

工程 SHALL 删除 `SHUD/src/ModelData/MD_RiverFlux.cpp:107` 的 `QrivSurf[iRiv] += QsegSurf[i]`；`PassValue()` 已从 `QsegSurf` 重新累加（master plan §5 S3a.1 + §4.7）。

#### Scenario: S3a.1 死代码删除后 bitwise vs B0

- **WHEN** S3a.1 改动 merge（PR-9 内一个独立 commit）
- **AND** 6 case bitwise 集跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `grep -n 'QrivSurf\[.*\] += QsegSurf' SHUD/src/ModelData/MD_RiverFlux.cpp` SHALL = 0 hits

### Requirement: S3a.2 删除 `Qe2r_Surf[iEle] += -QsegSurf[i]` 死代码

工程 SHALL 删除 `SHUD/src/ModelData/MD_RiverFlux.cpp:108` 的 `Qe2r_Surf[iEle] += -QsegSurf[i]`（master plan §5 S3a.2）。

#### Scenario: S3a.2 死代码删除后 bitwise vs B0

- **WHEN** S3a.2 改动 merge（PR-9 内独立 commit）
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `grep -n 'Qe2r_Surf\[.*\] +=.*QsegSurf' SHUD/src/ModelData/MD_RiverFlux.cpp` SHALL = 0 hits

### Requirement: S3a.3 删除 `QrivSub[iRiv] += QsegSub[i]` 死代码

工程 SHALL 删除 `SHUD/src/ModelData/MD_RiverFlux.cpp:121` 的 `QrivSub[iRiv] += QsegSub[i]`（master plan §5 S3a.3）。

#### Scenario: S3a.3 死代码删除后 bitwise vs B0

- **WHEN** S3a.3 改动 merge（PR-9 内独立 commit）
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `grep -n 'QrivSub\[.*\] += QsegSub' SHUD/src/ModelData/MD_RiverFlux.cpp` SHALL = 0 hits

### Requirement: S3a.4 删除 `Qe2r_Sub[iEle] += -QsegSub[i]` 死代码

工程 SHALL 删除 `SHUD/src/ModelData/MD_RiverFlux.cpp:122` 的 `Qe2r_Sub[iEle] += -QsegSub[i]`（master plan §5 S3a.4）。

#### Scenario: S3a.4 死代码删除后 bitwise vs B0 + Seg flux 纯函数

- **WHEN** S3a.4 改动 merge（PR-9 内独立 commit）
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `grep -n 'Qe2r_Sub\[.*\] +=.*QsegSub' SHUD/src/ModelData/MD_RiverFlux.cpp` SHALL = 0 hits
- **THEN** 经过 S3a.1–S3a.4 后 `fun_Seg_surface()` / `fun_Seg_sub()` 变成纯函数：只写 `QsegSurf[i]` / `QsegSub[i]`，不写任何 accumulator（code review 验证）

### Requirement: S3b.1 `QLakeRivIn[toLake] += QrivDown[i]` 拆为 per-river slot + 临时 PassValue gather

工程 SHALL 把 `SHUD/src/ModelData/MD_RiverFlux.cpp:24` 的 `Flux_RiverDown()` 中 `QLakeRivIn[toLake] += QrivDown[i]` 共享写改为：`Flux_RiverDown()` 只写 `QrivDown[i]`；lake 汇总 SHALL 暂时在 `PassValue()` 内部扩出的临时 gather 段完成（按 `for i in 0..NumRiv: if Riv[i].toLake > 0: QLakeRivIn[Riv[i].toLake-1] += QrivDown[i]`），到 PR-11 S3c.3 退役（master plan §5 S3b.1 + Conventions 过渡期处置）。

#### Scenario: S3b.1 拆分后 bitwise vs B0（lake case）

- **WHEN** S3b.1 改动 merge（PR-9 内独立 commit）
- **AND** lake-related case (`qhh` / `heihe` / `heihe_x4`) 跑 90-day truncation
- **THEN** 3 lake case `rivqdown.dat` + `LakeRivIn.dat` + 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** `awk '/Flux_RiverDown/,/^}/' SHUD/src/ModelData/MD_RiverFlux.cpp | grep -c 'QLakeRivIn\[.*\] +='` SHALL = 0 hits（函数体 scope）
- **THEN** `PassValue()` 函数体内 SHALL 临时包含 `QLakeRivIn[*] += QrivDown[*]` 累加段（PR-11 退役前）

### Requirement: S3b.2 `QLakeSurf[ilake] += Q` 拆为 per-edge slot + 临时 PassValue gather

工程 SHALL 把 `SHUD/src/ModelData/MD_ElementFlux.cpp:52` 的 `fun_Ele_surface()` lake 分支中 `QLakeSurf[ilake] += Q` 共享写改为：`fun_Ele_surface()` 写 per-edge contribution slot；lake 汇总 SHALL 暂时在 `PassValue()` 内部扩出的临时 gather 段完成，到 PR-11 S3c.3 退役（master plan §5 S3b.2）。

#### Scenario: S3b.2 拆分后 bitwise vs B0（lake case）

- **WHEN** S3b.2 改动 merge（PR-9 内独立 commit）
- **AND** lake-related case 跑 90-day truncation
- **THEN** 3 lake case `LakeSurf.dat` + 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** `awk '/fun_Ele_surface/,/^}/' SHUD/src/ModelData/MD_ElementFlux.cpp | grep -c 'QLakeSurf\[.*\] +='` SHALL = 0 hits（函数体 scope）
- **THEN** `PassValue()` 函数体内 SHALL 临时包含 `QLakeSurf[*]` lake gather 累加段

### Requirement: S3b.3 `QLakeSub[ilake] += Q` 拆为 per-edge slot + 临时 PassValue gather

工程 SHALL 把 `SHUD/src/ModelData/MD_ElementFlux.cpp:121` 的 `fun_Ele_sub()` lake 分支中 `QLakeSub[ilake] += Q` 共享写改为：`fun_Ele_sub()` 写 per-edge contribution slot；lake 汇总 SHALL 暂时在 `PassValue()` 内部扩出的临时 gather 段完成，到 PR-11 S3c.3 退役（master plan §5 S3b.3）。

#### Scenario: S3b.3 拆分后 bitwise vs B0（lake case）

- **WHEN** S3b.3 改动 merge（PR-9 内独立 commit）
- **AND** lake-related case 跑 90-day truncation
- **THEN** 3 lake case `LakeSub.dat` + 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** `awk '/fun_Ele_sub/,/^}/' SHUD/src/ModelData/MD_ElementFlux.cpp | grep -c 'QLakeSub\[.*\] +='` SHALL = 0 hits（函数体 scope）
- **THEN** `PassValue()` 函数体内 SHALL 临时包含 `QLakeSub[*]` lake gather 累加段

### Requirement: S3b.4 `qLakeEvap` / `qLakePrcp` 拆为 per-element slot + 临时 PassValue gather

工程 SHALL 把 `SHUD/src/ModelData/MD_f.cpp:15-16` 的 `qLakeEvap[..] += ...` 和 `qLakePrcp[..] += ...` 共享写改为：`f_loop()` 路径写 per-element contribution slot；lake 汇总 SHALL 暂时在 `PassValue()` 内部扩出的临时 gather 段完成，到 PR-11 S3c.3 退役（master plan §5 S3b.4 + Conventions 过渡期处置）。

**注意**：master plan 与 b1a_summary 表述为 `MD_f.cpp:15-16` 在 `f_loop()` 路径 — `MD_f.cpp` 是 ModelData 目录下，不是 Model/ 目录。

#### Scenario: S3b.4 拆分后 bitwise vs B0（lake case）

- **WHEN** S3b.4 改动 merge（PR-9 内独立 commit）
- **AND** lake-related case 跑 90-day truncation
- **THEN** 3 lake case `LakeEvap.dat` + `LakePrcp.dat` + 所有 lake-related .dat SHALL byte-equal B0_output 归档
- **THEN** `awk '/f_loop/,/^}/' SHUD/src/ModelData/MD_f.cpp | grep -cE 'qLakeEvap\[.*\] \+=|qLakePrcp\[.*\] \+='` SHALL = 0 hits（函数体 scope）
- **THEN** `PassValue()` 函数体内 SHALL 临时包含 `QLakeEvap[*]` / `QLakePrcp[*]` lake gather 累加段

### Requirement: S3c.1 `QrivSurf` / `QrivSub` segment→river 累加用 adjacency list

工程 SHALL 用 S4 输出的 `seg_by_riv[ir]` adjacency list 重写 `SHUD/src/ModelData/MD_f.cpp:167-174` 的 segment → river surface/sub flux 累加。新 gather 形式：`for ir in 0..NumRiv-1: for k in seg_by_riv[ir]: QrivSurf[ir] += QsegSurf[k]; QrivSub[ir] += QsegSub[k]`。**SHALL** 保证 gather 顺序与 B0 serial loop 的原始数组索引顺序完全一致（master plan §5 S3c.1 + L1260-1264）。

#### Scenario: S3c.1 用 adjacency list 后 bitwise vs B0

- **WHEN** S3c.1 改动 merge（PR-11 内独立 commit，前置：PR-10 S4 已 merge）
- **AND** 6 case bitwise 集跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `QrivSurf` / `QrivSub` 累加路径 SHALL 使用 `seg_by_riv[ir]`；不再有 `for i in 0..NumSegmt: QrivSurf[RivSeg[i].iRiv-1] += ...` 形式（grep `RivSeg\[.*\]\.iRiv` 不出现在 `MD_rhs_core.cpp` `rhs_deterministic_gather()` 函数体内）

### Requirement: S3c.2 `QrivUp[iDownStrm] += -QrivDown[i]` 用 adjacency list

工程 SHALL 用 S4 输出的 `upstream_by_down[ir]` adjacency list 重写 `SHUD/src/ModelData/MD_f.cpp:177` 的 downstream river upstream 汇总。新 gather 形式：`for ir in 0..NumRiv-1: for k in upstream_by_down[ir]: QrivUp[ir] += -QrivDown[k]`。SHALL 保证顺序 = B0 serial loop order（master plan §5 S3c.2）。

#### Scenario: S3c.2 用 adjacency list 后 bitwise vs B0

- **WHEN** S3c.2 改动 merge（PR-11 内独立 commit）
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** `QrivUp` 累加路径 SHALL 使用 `upstream_by_down[ir]`；不再有 `for i in 0..NumRiv: QrivUp[Riv[i].down-1] += ...` 形式

### Requirement: S3c.3 `PassValue()` 整体替换为 `rhs_deterministic_gather()` (`MD_rhs_core.cpp`)

工程 SHALL 删除 `SHUD/src/ModelData/MD_f.cpp:156-196` 的 `PassValue()` 函数定义。SHALL 新建 `rhs_deterministic_gather()` 函数 **在 `SHUD/src/Model/MD_rhs_core.cpp`**（**不**新建 `MD_gather.cpp`；理由：避免文件数膨胀，rhs core 自身已是入口，gather 放此处语义自然 — design.md D12）。新函数合并 S3c.1 + S3c.2 + S3b.1-S3b.4 的所有 lake gather 逻辑，使用 S4 7 个 adjacency list。`SHUD/src/Model/MD_rhs_core.cpp:216` 调 `PassValue()` SHALL 改为调 `rhs_deterministic_gather()`。SHALL 同步删除 `SHUD/src/ModelData/MD_f_omp.cpp:99` 的 `PassValue()` 调用（实际上 PR-8 S2 capstone 已删整个 `MD_f_omp.cpp`，本 PR 顺带验证 tree-wide 0 hits）。SHALL 不引入新 `#pragma omp parallel` 区段（master plan §5 S3c.3 + §C7 fork-join 最小化）。

#### Scenario: S3c.3 PassValue 完整重写后 bitwise vs B0 + tree-wide 0 hits

- **WHEN** S3c.3 改动 merge（PR-11 capstone 内）
- **AND** 6 case bitwise 集跑 90-day truncation
- **THEN** 6 case `rivqdown.dat` + CVODE 15-key SHALL bitwise = B0-tag
- **THEN** lake-related case 所有 lake-related .dat SHALL byte-equal B0_output 归档（验证 lake gather 重写后与 B0 一致）
- **THEN** `grep -rn 'PassValue\b' SHUD/src/` SHALL = 0 hits（tree-wide grep — PassValue 函数定义 + 所有调用都已删/替换）
- **THEN** `rhs_deterministic_gather()` 函数定义 SHALL 在 `SHUD/src/Model/MD_rhs_core.cpp` 内
- **THEN** `rhs_deterministic_gather()` 函数体内 SHALL 使用 7 个 adjacency list（`seg_by_riv` / `seg_by_ele` / `upstream_by_down` / `riv_in_by_lake` / `ele_by_lake` / `lake_bank_edge_by_lake` / `edge_by_ele`）
- **THEN** `rhs_deterministic_gather()` 函数体内 SHALL 无 `#pragma omp parallel` directive
- **THEN** 水量平衡误差（lake / river / element 总流入 - 流出 - 蓄变化）SHALL 与 B0-tag 相同 case 跑出的误差 byte-equal
