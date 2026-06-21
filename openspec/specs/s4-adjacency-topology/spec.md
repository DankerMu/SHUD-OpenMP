## Conventions

本 spec 沿用 `s2-semantic-merge` 的 Conventions（Case Scope / Lake-related 输出文件清单 / B0-tag 引用），不重复列出。

S4 PR-10 内部 SHALL 按 S4.1 → S4.2 → S4.3 → S4.4 → S4.5 → S4.6 → S4.7 顺序分 7 个独立 commit，每个 commit 跑 4-case Mac bitwise + 与 PR-9 临时扩的 `PassValue()` 内部 gather 顺序 bitwise identical 验证（design.md D4 / D5）。

## ADDED Requirements

### Requirement: B1a 期间 S1 grep gate 持续 0 hits（S4 PR-10 范围）

工程 SHALL 在 PR-10 S4 改动中持续保持 S1 已 enforce 的 grep gate 0 hits（与 s2-semantic-merge spec 中同名 Requirement 等价；列出仅为 PR-10 CI gate 独立可验证）。

#### Scenario: 4 grep gate 在 PR-10 持续 0 hits

- **WHEN** PR-10 S4 merge gate 检查
- **THEN** `_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 在 `SHUD/src/` SHALL = 0 hits
- **THEN** `SHUD_USE_OPENMP_NVECTOR` SHALL 仅在 `Macros.hpp` 1 处 define

### Requirement: S4 新建 `MD_adjacency.hpp` + `MD_adjacency.cpp`

工程 SHALL 在 PR-10 新建 `SHUD/src/ModelData/MD_adjacency.hpp` 与 `SHUD/src/ModelData/MD_adjacency.cpp`，包含 7 个 adjacency list 数据结构 + 构建函数（详见后续 S4.1–S4.7 Requirement）+ 3 条 `id == index+1` assert + fallback path（id != index+1 时用 array index）+ fallback path unit test（详见后续 Requirement）（design.md D12）。

#### Scenario: 新文件落地 + 头文件被 rhs core 引用

- **WHEN** PR-10 S4 merge
- **THEN** `ls SHUD/src/ModelData/MD_adjacency.hpp SHUD/src/ModelData/MD_adjacency.cpp` SHALL 全部 exit 0
- **THEN** `SHUD/src/Model/MD_rhs_core.cpp` 文件顶部 SHALL 包含 `#include "MD_adjacency.hpp"`
- **THEN** `SHUD/Makefile` SHALL 把 `MD_adjacency.cpp` 加入 SOURCE 列表（与 `MD_f.cpp` 等同级出现）
- **THEN** `SHUD/src/ModelData/MD_adjacency.hpp` 文件 SHALL 声明 7 个 adjacency list 的 extern + 一个 build 入口（如 `void build_adjacency_lists(struct mesh* mesh_ptr, struct riv_t* riv_ptr, ...)`）

### Requirement: S4.1 `seg_by_riv[ir]` adjacency list 构建

工程 SHALL 构建 `seg_by_riv[ir]` adjacency list：对每个 river index `ir`，记录所有 segment index `iseg` 使得 `RivSeg[iseg].iRiv - 1 == ir`，**按 B0 serial loop 的原始数组索引升序**（不是 id 升序，除非 `RivSeg[i].id == i + 1` assert 通过）。用途：river 汇总 segment surface/sub flux（master plan §5 S4.1）。

#### Scenario: S4.1 list 构建 + B0 顺序一致

- **WHEN** S4.1 改动 merge（PR-10 内独立 commit）
- **AND** 6 case bitwise 集跑 90-day truncation
- **THEN** 对每个 river `ir`，`seg_by_riv[ir]` 内部 segment index 顺序 SHALL 等价于 B0 `for (i = 0; i < NumSegmt; ++i) if (RivSeg[i].iRiv - 1 == ir) yield i` 的遍历顺序
- **THEN** PR-9 临时扩的 `PassValue()` 内部 `QrivSurf` / `QrivSub` 累加顺序 SHALL 与新 `seg_by_riv` 顺序 bitwise identical（验证通过：用 S4.1 list 重做 gather，rivqdown.dat SHA256 与 PR-9 merge 后归档 byte-equal）
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `seg_by_riv` 条目，记录 `sort_rule: B0 iseg array index ascending` + `b0_source: SHUD/src/ModelData/MD_f.cpp:170-171`

### Requirement: S4.2 `seg_by_ele[ie]` adjacency list 构建

工程 SHALL 构建 `seg_by_ele[ie]` adjacency list：对每个 element index `ie`，记录所有 segment index `iseg` 使得 `RivSeg[iseg].iEle - 1 == ie`，按 B0 `iseg` 数组索引升序。用途：element 汇总 segment 交换 flux（master plan §5 S4.2）。

#### Scenario: S4.2 list 构建 + B0 顺序一致

- **WHEN** S4.2 改动 merge（PR-10 内独立 commit）
- **THEN** 对每个 element `ie`，`seg_by_ele[ie]` 内部 segment index 顺序 SHALL 等价于 B0 `for (i = 0; i < NumSegmt; ++i) if (RivSeg[i].iEle - 1 == ie) yield i` 顺序
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `seg_by_ele` 条目 + `b0_source: SHUD/src/ModelData/MD_f.cpp:172-173`

### Requirement: S4.3 `upstream_by_down[ir]` adjacency list 构建

工程 SHALL 构建 `upstream_by_down[ir]` adjacency list：对每个 downstream river index `ir`，记录所有 upstream river index `iup` 使得 `Riv[iup].down - 1 == ir`，按 B0 `iriv` 数组索引升序。用途：downstream river 汇总 upstream downflow（master plan §5 S4.3）。

#### Scenario: S4.3 list 构建 + B0 顺序一致

- **WHEN** S4.3 改动 merge（PR-10 内独立 commit）
- **THEN** 对每个 downstream river `ir`，`upstream_by_down[ir]` 内部 upstream index 顺序 SHALL 等价于 B0 `for (i = 0; i < NumRiv; ++i) if (Riv[i].down - 1 == ir) yield i` 顺序
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `upstream_by_down` 条目 + `b0_source: SHUD/src/ModelData/MD_f.cpp:177`

### Requirement: S4.4 `riv_in_by_lake[ilake]` adjacency list 构建

工程 SHALL 构建 `riv_in_by_lake[ilake]` adjacency list：对每个 lake index `ilake`，记录所有汇入 river index `iriv` 使得 `Riv[iriv].toLake - 1 == ilake`，按 B0 `iriv` 数组索引升序。用途：lake 汇总 river 入流（master plan §5 S4.4）。

#### Scenario: S4.4 list 构建 + lake case 顺序一致

- **WHEN** S4.4 改动 merge（PR-10 内独立 commit）
- **AND** lake-related case (`qhh` / `heihe` / `heihe_x4`) 跑 90-day truncation
- **THEN** 对每个 lake `ilake`，`riv_in_by_lake[ilake]` 内部 river index 顺序 SHALL 等价于 B0 `Flux_RiverDown()` 中 `for (i = 0; i < NumRiv; ++i) if (Riv[i].toLake - 1 == ilake) yield i` 顺序
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `riv_in_by_lake` 条目 + `b0_source: Flux_RiverDown() (SHUD/src/ModelData/MD_RiverFlux.cpp:24)`

### Requirement: S4.5 `ele_by_lake[ilake]` adjacency list 构建

工程 SHALL 构建 `ele_by_lake[ilake]` adjacency list：对每个 lake index `ilake`，记录所有 lake element index `iele` 使得 `Ele[iele].iLake - 1 == ilake`，按 B0 `iele` 数组索引升序。用途：lake 汇总 element evap/precip（master plan §5 S4.5）。

#### Scenario: S4.5 list 构建 + lake case 顺序一致

- **WHEN** S4.5 改动 merge（PR-10 内独立 commit）
- **AND** lake-related case 跑 90-day truncation
- **THEN** 对每个 lake `ilake`，`ele_by_lake[ilake]` 内部 element index 顺序 SHALL 等价于 B0 `for (i = 0; i < NumEle; ++i) if (Ele[i].iLake - 1 == ilake) yield i` 顺序
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `ele_by_lake` 条目 + `b0_source: SHUD/src/ModelData/MD_f.cpp:15-16`

### Requirement: S4.6 `lake_bank_edge_by_lake[ilake]` adjacency list 构建

工程 SHALL 构建 `lake_bank_edge_by_lake[ilake]` adjacency list：对每个 lake index `ilake`，记录所有 (element index `iele`, edge index `j` ∈ {0,1,2}) 二元组使得 `Ele[iele].lakenabr[j] - 1 == ilake`，按 B0 `iele` 数组索引升序，每个 element 内按 `j = 0, 1, 2` 顺序。用途：lake 汇总岸边 element flux（master plan §5 S4.6）。

#### Scenario: S4.6 list 构建 + 二维顺序一致

- **WHEN** S4.6 改动 merge（PR-10 内独立 commit）
- **AND** lake-related case 跑 90-day truncation
- **THEN** 对每个 lake `ilake`，`lake_bank_edge_by_lake[ilake]` 内部 `(iele, j)` 二元组顺序 SHALL 等价于 B0 `for (i = 0; i < NumEle; ++i) for (j = 0; j < 3; ++j) if (Ele[i].lakenabr[j] - 1 == ilake) yield (i, j)` 顺序
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `lake_bank_edge_by_lake` 条目 + `b0_source: fun_Ele_surface/sub() element loop × edge loop`

### Requirement: S4.7 `edge_by_ele[ie]` adjacency list 构建

工程 SHALL 构建 `edge_by_ele[ie]` adjacency list：对每个 element index `ie`，记录三个边 `j ∈ {0, 1, 2}` 的 (邻居 element index `inabr = Ele[ie].nabr[j] - 1`, edge index `j`) 二元组，按固定 `j = 0, 1, 2` 顺序。用途：element 汇总三邻边 flux（master plan §5 S4.7）。

#### Scenario: S4.7 list 构建 + edge 顺序一致

- **WHEN** S4.7 改动 merge（PR-10 内独立 commit）
- **THEN** 对每个 element `ie`，`edge_by_ele[ie]` 内部 `(inabr, j)` 二元组顺序 SHALL 严格按 `j = 0, 1, 2`
- **THEN** `docs/topology_manifest.yaml` SHALL 包含 `edge_by_ele` 条目 + `b0_source: 原始 3-neighbor 循环 (fun_Ele_*() 内)`

### Requirement: `id == index + 1` assert 加入 + 失败回退到 array index 排序 + fallback unit test

工程 SHALL 在 PR-10 引入的 adjacency list 构建代码内显式加入三条 assert：`assert(RivSeg[i].id == i + 1)` ∀ `i ∈ [0, NumSegmt)`、`assert(Riv[i].id == i + 1)` ∀ `i ∈ [0, NumRiv)`、`assert(Ele[i].id == i + 1)` ∀ `i ∈ [0, NumEle)`。若任一 assert 失败，adjacency list 构建 SHALL 使用原始数组索引顺序（而非 id sort）—— 即 fallback path。`docs/topology_manifest.yaml` 的 `asserts` 段 SHALL 记录每个 case 的 pass/fail 结果（master plan §5 L1304-1313）。

工程 SHALL 同 PR 新增 fallback path 单测覆盖：用 synthetic mock entity 数组（人为构造 `id != index+1` 数据，如 `RivSeg[0].id = 7, RivSeg[1].id = 3, RivSeg[2].id = 5`），验证 adjacency list 构建 fallback 到 array index 顺序后 output 与 array-index ordering 等价（design.md D12）。

#### Scenario: 三条 assert 在所有 6 case 都 pass + fallback 单测 PASS

- **WHEN** PR-10 S4 merge
- **AND** 6 case bitwise 集 Mac + server DEBUG build (`make shud DEBUG=1`) 跑 90-day truncation
- **THEN** 三条 assert SHALL 全部 pass（即对所有 case，三个实体的 id 与数组 index+1 严格等价）
- **THEN** `docs/topology_manifest.yaml` `asserts` 段 SHALL 包含三个 entity (RivSeg / Riv / Ele) × 6 case 共 18 个条目，全部 `pass`；外加 `kashigeer` × 3 entity = 3 个 `N/A (deferred-upstream)` 条目（总计 21 条目，与 7 case × 3 entity 对齐）
- **THEN** 若任一 case 任一 entity assert 失败 → adjacency list 构建 SHALL 走 array index 顺序 fallback path，manifest 记录 `fail` 并附 `fallback_note`
- **AND** fallback 单测（独立 unit test 文件，如 `SHUD/tests/test_adjacency_fallback.cpp` 或 `tools/test_adjacency/`）覆盖
- **THEN** 单测用 synthetic mock entity 数组 `id != index+1` 触发 fallback path
- **THEN** 单测 SHALL 验证 fallback 后 adjacency list output 与 array-index ordering 计算的 expected output 完全相等
- **THEN** 单测在 CI workflow `serial-baseline.yml` 内有独立 step 执行 `make test_adjacency_fallback && ./test_adjacency_fallback`，失败阻塞 PR-10 merge

### Requirement: `docs/topology_manifest.yaml` schema + 单文件 + CI 校验

工程 SHALL 在 PR-10 新建 `docs/topology_manifest.yaml` 单文件，按以下 schema 写入 7 个 adjacency list 元数据 + 21 个 assert 结果（master plan §5 L1318 + design.md D9）。

#### Scenario: manifest 文件存在 + schema 合规 + CI 校验通过

- **WHEN** PR-10 S4 merge
- **THEN** `ls docs/topology_manifest.yaml` SHALL exit 0
- **THEN** YAML root SHALL 包含 `adjacency_lists` (list of 7) + `asserts` (list of 21 entries, 3 entity × 7 case，其中 kashigeer × 3 为 `N/A (deferred-upstream)`)
- **THEN** 每个 `adjacency_lists` 条目 SHALL 包含 `name` / `purpose` / `sort_rule` / `b0_source` / `iter_pattern` 五字段
- **THEN** 每个 `asserts` 条目 SHALL 包含 `entity` / `case` / `rule` / `result` (pass / fail / N/A) + 若 fail 则 `fallback_note`
- **THEN** CI workflow `.github/workflows/serial-baseline.yml` SHALL 新增一个 schema 校验 step（`uv run python -c '...yaml.safe_load'` 或 `yq` 验证），失败则 PR 阻塞 merge
