## Purpose

规约 S5b RHS scratch arrays 所有权归并 + lake reset 顺序前置 + RHS print 重组（audit + 序结构）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/b1b-baseline-completion/specs/<capability>/spec.md PROMOTE 而来（#190 S6c-12c capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。

## Requirements

### Requirement: scratch 数组必须每元素只有唯一 owner 写入

`Model_Data` 上的 RHS scratch 数组 SHALL 满足 owner-only-write 约束：每个数组 index 在 RHS 一次 evaluation 内只能被一个固定 owner（element/river/lake）写入，禁止跨 owner 共享 `+=`。涉及的最小数组集合 = `qEleSurf` / `qEleSub` / `qEleInfil` / `qEleExfil` / `QeleSurf` / `QeleSub` / `Qseg*` / `Qriv*` / `QLake*`（声明位置：`SHUD/src/Model_Data.hpp` L121–L184）。审计结果 SHALL 记入 `docs/topology_manifest.yaml` 关联节，每个数组列 owner 类型 + 写入代码行号。

#### Scenario: 9 类 scratch 数组完成 owner 表
- **WHEN** 审计 RHS 路径上对上述 9 类数组的所有写入
- **THEN** 每类数组写入点的 owner（element index `i` / river index `r` / lake index `l`）唯一可识别；交叉写入（A owner 写 B owner index）数量为 0；审计表写入 `docs/topology_manifest.yaml` `s5b_scratch_ownership` 节

#### Scenario: PR-11 deterministic gather 不被引入新共享写
- **WHEN** S5b 完成后 grep `SHUD/src/MD_*.cpp` 寻找 `PassValue\b` 与 `+=` 模式
- **THEN** PR-11 已替换的 `rhs_deterministic_gather()` 仍是唯一 gather 入口；新增共享 `+=` 数量为 0

---

### Requirement: Element/River/Lake update 方法只改自身字段

`Ele[i].updateElement()` / `Riv[i].updateRiver()` / `lake[i].update()` SHALL 满足 self-only-write 约束：只写 `Ele[i]` / `Riv[i]` / `lake[i]` 的字段，不写其他 element/river/lake 实例字段、不写全局变量、不调用任何会触发跨 owner 写的 helper。审计 SHALL 覆盖每个 update 方法的完整 call graph 一层（不递归全树）。

#### Scenario: Element update 内部 grep 干净
- **WHEN** 在 `SHUD/src/Element.cpp` `updateElement()` 方法体内 grep `Ele\[` / `Riv\[` / `lake\[` / `MD->`
- **THEN** `Ele[` 出现仅限 `Ele[id-1]` / `Ele[i]` 形式且 i 已经是该方法的 self index；`Riv[` / `lake[` / `MD->` 出现 0 次

#### Scenario: River update 内部 grep 干净
- **WHEN** 在 `SHUD/src/River.cpp` `updateRiver()` 方法体内做同样 grep
- **THEN** `Riv[` 仅自指；其他 owner 数组 0 次

#### Scenario: Lake update 内部 grep 干净
- **WHEN** 在 `SHUD/src/Lake.cpp` `update()` 方法体内做同样 grep
- **THEN** `lake[` 仅自指；其他 owner 数组 0 次

---

### Requirement: RHS 内 diagnostic 写入必须改 RHS 后串行输出

RHS 路径中 debug print / NaN check / 任何 std::cout/stderr/printf 类副作用 SHALL 改写为：(a) 写入 per-element diagnostic buffer（buffer 自身满足 owner-only-write）；(b) RHS 一次 evaluation 完成、回到串行段后统一 flush 到 stderr 或 log 文件。原始 in-RHS print 语句 SHALL 替换或移除，grep `RHS|rhs_core|rhs_kernel` 区段内不得留 `std::cout` / `printf` / `cerr`。

#### Scenario: RHS 路径 print 类调用清零
- **WHEN** 在 `SHUD/src/MD_f.cpp` / `MD_ElementFlux.cpp` / `MD_ET.cpp` 的 RHS 路径 grep `printf|std::cout|std::cerr|fprintf`
- **THEN** 命中 0 行；所有 diagnostic 写入改 per-element buffer + 串行段 flush

#### Scenario: diagnostic buffer 在 RHS 一次评估内 owner-only 写
- **WHEN** 审计 diagnostic buffer 的所有写入
- **THEN** 每个 buffer slot 只被 owner 元素的处理代码写一次

---

### Requirement: lake reset 顺序与 owner-only-write 一致

qhh case lake 算例中 `QLake*` / `lakystage` / `lakqrivin` / `lakqrivout` 等 lake 输出的 reset / clear 操作 SHALL 满足 lake-owner 唯一写入约束：(a) 每个 lake-owned scratch slot 的 reset 在该 lake 的 owner 元素任何 element 写入**之前**完成（S2.8 lake reset 前置原则，PR-3 #146 已落地）；(b) 审计 `Lake.cpp` / `MD_update.cpp` / `MD_f.cpp` lake 分支 reset 顺序，记录到 `docs/topology_manifest.yaml` `s5b_lake_reset_order` 节；(c) qhh 3 lake outputs (`lakystage` / `lakqrivin` / `lakqrivout`) 的 reset 在 element loop 之前。

#### Scenario: lake reset 在 element 写入之前
- **WHEN** trace qhh 90 天 RHS 第一次 evaluation 上 lake reset 路径与 element loop 顺序
- **THEN** reset 调用 line number < element loop 第一次写 `QLake*` 的 line number

#### Scenario: lake reset 写 docs/topology_manifest.yaml
- **WHEN** 检查 `docs/topology_manifest.yaml` 中 `s5b_lake_reset_order` 节
- **THEN** 含 lake reset 调用点 (file:line) 与 element loop 第一次 lake-relevant 写入点 (file:line) 的对照

#### Scenario: qhh 3 lake outputs bitwise vs B1a-tag
- **WHEN** S5b 完成 commit 上跑 qhh 90 天截断 NUM_OPENMP=1
- **THEN** lakystage / lakqrivin / lakqrivout 3 个输出 SHA256 与 B1a-tag golden 完全一致

---

### Requirement: S5b 完成后完整 run 与 B1a bitwise identical

S5b 收尾时，4 case Mac endpoint + 2 case 服务器 Slurm endpoint SHA256 SHALL 与 B1a-tag golden 完全一致；kashigeer 维持 deferred-upstream N/A。

#### Scenario: 6 case bitwise vs B1a-tag 全通过
- **WHEN** S5b 完成 commit 上跑 6 case bitwise (NUM_OPENMP=1, 90 天截断)
- **THEN** 全部 PASS（含 qhh 3 lake outputs）
