# SHUD 并行前对齐与完整并行改造路线包

本包用于替代前一版只讲“对齐”的文档。新版把三部分合成一条完整路线：

1. **并行前单线程 parallel-ready 预优化**：把 SHUD 先整理成可验证、可拆分、可复现的单线程参考实现。
2. **strict 并行改造阶段**：在不改变单线程参考结果的前提下，逐步打开 OpenMP 并行。
3. **production 并行阶段**：在可解释容差内引入 CVODE 向量层、线性求解器和确定性 reduction，以换取更高速度。

核心结论：

> 不应该直接在当前 SHUD 的 `_omp` 路径上继续堆并行。正确路线是先形成唯一 RHS core 和 B1 parallel-ready serial reference，再让后续所有并行阶段只改变执行策略，不夹带物理路径修复、共享累加拆分或数据结构重排。

## 包内文件

| 文件 | 用途 |
|---|---|
| `docs/00_index.md` | 总览、阶段关系、B0/B1/并行基线定义 |
| `docs/01_source_observations.md` | 基于当前 SHUD 源码的关键问题观察 |
| `docs/02_pre_parallel_alignment.md` | 并行前必须完成的对齐工作 |
| `docs/03_single_thread_parallel_ready.md` | 单线程 parallel-ready 预优化路线 |
| `docs/04_parallel_phases.md` | 后续每个并行阶段要改造哪些计算 |
| `docs/05_accuracy_acceptance.md` | 每阶段精度、一致性和回归验收标准 |
| `docs/06_risk_register.md` | 风险登记、go/no-go 检查表 |
| `assets/roadmap_mermaid.md` | 可直接放入 Markdown 的 Mermaid 路线图 |
| `SHUD_parallel_full_plan.md` | 以上内容的合并版，方便一次性阅读 |

## 使用建议

先读 `SHUD_parallel_full_plan.md` 获得完整逻辑，再根据开发阶段查阅对应分文件。实际研发时建议按以下顺序执行：

```text
B0 historical serial base
  → 单线程对齐与预优化
  → B1 parallel-ready serial reference
  → owner-local strict OpenMP
  → deterministic production OpenMP / CVODE / linear solver
```

## 文档依据

本包依据当前公开 SHUD 源码、OpenMP reduction 规范和 SUNDIALS/CVODE 文档整理。重点参考：

- SHUD `f.cpp`: https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/Model/f.cpp
- SHUD `MD_f.cpp`: https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f.cpp
- SHUD `MD_f_omp.cpp`: https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_f_omp.cpp
- SHUD `MD_update.cpp`: https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/ModelData/MD_update.cpp
- SHUD `TimeSeriesData.cpp`: https://raw.githubusercontent.com/SHUD-System/SHUD/master/src/classes/TimeSeriesData.cpp
- OpenMP reduction 规范: https://www.openmp.org/spec-html/5.0/openmpsu107.html
- SUNDIALS/CVODE introduction: https://sundials.readthedocs.io/en/latest/cvode/Introduction_link.html
- SUNDIALS NVECTOR API: https://sundials.readthedocs.io/en/latest/nvectors/NVector_API_link.html
