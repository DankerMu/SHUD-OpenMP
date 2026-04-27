# 06 风险登记与 go/no-go 检查表

本节回答：**完整路线里有哪些新风险？如何判断能不能进入下一阶段？**

## 1. 总体风险分级

| 风险等级 | 含义 | 处理原则 |
|---|---|---|
| R0 | 不影响结果，只影响结构或性能 | 可继续，但需回归 |
| R1 | 可能产生 bit 差异，但可定位 | 暂停进入 strict 并行，先锁定 B1 |
| R2 | 可能改变物理语义 | 必须单独立项，不混入加速路线 |
| R3 | 可能导致非确定性或 data race | 必须阻断，不能进入下一阶段 |
| R4 | 可能导致守恒破坏或 solver 不稳定 | 必须回滚或重新设计 |

## 2. 主要风险登记

### 风险 1：继续维护两套 RHS 路径

- **来源**：`f.cpp` 中 serial 与 `_omp` 分别调用不同 RHS 函数链。
- **后果**：并行结果差异无法归因。
- **等级**：R3。
- **控制措施**：建立唯一 RHS core；OpenMP 只作为 execution policy。
- **go/no-go**：未完成唯一 RHS core，不进入 P1。

### 风险 2：lake/ET/river DY 语义未对齐

- **来源**：当前 serial 与 `_omp` 路径在 lake、ET、river DY 处理上存在明显差异。
- **后果**：并行结果改变物理过程，不是单纯并行误差。
- **等级**：R2/R3。
- **控制措施**：B1 继承 serial 语义；所有差异单独记录。
- **go/no-go**：未完成 semantic diff report，不进入 P1。

### 风险 3：shared floating accumulation

- **来源**：`PassValue()` 中 segment→river、segment→element、upstream→downstream 的 `+=`。
- **后果**：atomic/reduction 虽可避免 data race，但不能保证 bitwise identical。
- **等级**：R3。
- **控制措施**：compute flux 与 deterministic gather 拆分。
- **go/no-go**：存在 shared floating `+=`，不进入 P4/P5。

### 风险 4：OpenMP reduction 顺序不确定

- **来源**：OpenMP 规范指出 reduction values 的组合位置和顺序 unspecified。
- **后果**：与 B1 不 bitwise identical，多次运行或不同线程数可能有差异。
- **等级**：strict 阶段 R3；production 阶段 R1。
- **控制措施**：strict 阶段禁止普通 floating reduction；production 阶段使用 deterministic reduction 并设容差。
- **go/no-go**：P1–P7 不允许普通 floating reduction。

### 风险 5：forcing cache 改变时间采样语义

- **来源**：当前 `getX()` 是当前 queue 行取值；若缓存时顺手插值，会改变模型。
- **后果**：输出差异被误判为并行差异。
- **等级**：R2。
- **控制措施**：forcing cache 先保持 B0 语义；插值独立进入精度路线。
- **go/no-go**：cache `getX()` 未通过 bitwise 比较，不锁定 B1。

### 风险 6：CVODE OpenMP N_Vector 提前引入

- **来源**：SUNDIALS 支持 OpenMP N_Vector，但其 norm/dot product reduction 可能改变 solver 路径。
- **后果**：即使 RHS 正确，完整 run 不再 bitwise identical。
- **等级**：R1/R2。
- **控制措施**：先完成 P7 full RHS OpenMP + serial CVODE；再进入 P8。
- **go/no-go**：P7 未通过，不进入 P8。

### 风险 7：编译器优化改变浮点行为

- **来源**：fast-math、FMA contraction、不同优化级别、不同 BLAS/SUNDIALS 版本。
- **后果**：难以判断差异来自代码还是编译环境。
- **等级**：R2/R3。
- **控制措施**：strict 阶段固定工具链；禁止 fast-math；记录 compile manifest。
- **go/no-go**：没有 compile manifest，不进行 bitwise 回归判定。

### 风险 8：未初始化数组或旧值残留

- **来源**：serial/omp update/init 覆盖不一致，尤其 lake flux、river flux、Qe2r 等。
- **后果**：结果非确定性，或线程数相关。
- **等级**：R4。
- **控制措施**：统一 reset stage；debug 模式 fill NaN/sentinel；RHS snapshot 检测。
- **go/no-go**：存在未初始化读写，不进入 P1。

### 风险 9：诊断/日志输出破坏并行确定性

- **来源**：RHS 内 debug print、NaN check 输出、文件 flush。
- **后果**：并行运行输出顺序不确定，甚至影响性能。
- **等级**：R1/R3。
- **控制措施**：RHS 内只写 diagnostic buffer；RHS 后串行输出。
- **go/no-go**：strict 阶段禁止 RHS 内多线程文件输出。

### 风险 10：生产模式被误当成 strict 模式

- **来源**：CVODE vector 并行、Krylov solver、deterministic tree reduction 等都可能改变 bitwise 结果。
- **后果**：研发团队对“正确并行”的标准混乱。
- **等级**：R2。
- **控制措施**：明确 StrictOMP 和 ProductionOMP 两种模式。
- **go/no-go**：P8/P9 报告必须标注为 production tolerance，不得与 strict bitwise 混用。

## 3. 阶段 go/no-go 检查表

### 进入 S1 前

- [ ] B0 benchmark set 已确定；
- [ ] B0 单线程重复运行 bitwise identical；
- [ ] CVODE stats 可导出；
- [ ] RHS snapshot harness 初版可用；
- [ ] 编译环境已固定。

### 进入 S3 前

- [ ] 唯一 RHS core 初版完成；
- [ ] `policy=Serial` 与 B0 bitwise identical；
- [ ] lake/ET/river DY/update/init 差异已有报告；
- [ ] 没有把物理修正混入 RHS core 重构。

### 进入 S8/B1 锁定前

- [ ] compute flux 与 gather 已拆分；
- [ ] topology manifest 已生成；
- [ ] forcing cache 与 B0/B1 `getX()` bitwise identical；
- [ ] 所有 owner mapping 已记录；
- [ ] B1 单线程完整 run 可复现；
- [ ] B0→B1 差异如有，已写 changelog。

### 进入 P1 前

- [ ] B1 已锁定；
- [ ] strict OpenMP 编译选项确定；
- [ ] 禁用 fast-math；
- [ ] `schedule(static)` 规则确定；
- [ ] RHS 子阶段的 owner 写入规则已审查；
- [ ] 不存在共享浮点 `+=`。

### 进入 P4/P5 前

- [ ] element vertical 并行已通过 bitwise；
- [ ] element horizontal flux 的写入 owner 已明确；
- [ ] segment flux 函数只写 `Qseg*`；
- [ ] gather list 排序与 B1 一致；
- [ ] 禁止 atomic floating add。

### 进入 P7 前

- [ ] P1–P6 每阶段 RHS snapshot 均 bitwise identical；
- [ ] 完整 run 与 B1 bitwise identical；
- [ ] CVODE stats identical；
- [ ] 多线程重复运行 deterministic。

### 进入 P8 前

- [ ] P7 full RHS OpenMP + serial CVODE 通过；
- [ ] production tolerance 已定义；
- [ ] CVODE stats 输出完整；
- [ ] 已明确要试的 solver：OpenMP N_Vector / KLU / GMRES / FGMRES / preconditioner；
- [ ] 已准备 small/medium/large benchmark。

### 进入 P9 前

- [ ] P8 solver 路径已稳定；
- [ ] production 结果 deterministic；
- [ ] 与 B1 的误差在容差内；
- [ ] 水量平衡和 hydrograph 指标不恶化；
- [ ] 新 summation 策略独立记录，不再声称 bitwise B1。

## 4. 回滚策略

每个阶段都应保留上一阶段 tag：

```text
B0-tag
S1-rhs-core-tag
S3-gather-split-tag
B1-tag
P1-update-omp-tag
P2-vertical-omp-tag
P5-gather-omp-tag
P7-full-rhs-omp-tag
P8-cvode-prod-tag
```

如果出现差异：

1. 先停在 RHS snapshot，不跑完整 run；
2. 找 first mismatch array；
3. 找 first mismatch index；
4. 判断是公式差异、求和顺序差异、未初始化、还是共享写；
5. 回滚到上一 tag，单独修复。

## 5. 最重要的管理建议

把开发任务分成三类，不要混合：

```text
A 类：reference preserving refactor
B 类：parallel execution policy
C 类：numerical/physical improvement
```

A 类和 B 类在 strict 阶段必须 bitwise identical。C 类必须单独进入 production 或物理精度路线，不得夹带进并行正确性验证。
