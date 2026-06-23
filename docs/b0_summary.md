# B0 基线总结

## 背景与定义

B0 基线指 SHUD 源码在未做任何 OpenMP 改造前的单线程编译产物，对应 master plan §3 定义的"参照基线 (reference baseline)"。其工作阶段为 §5 中的 S0 ("锁定 B0 历史基线")。本项目中 **stage 指工作阶段** (S0/S1/...)，**baseline 指工作产物** (B0/B1a/...)，二者一一对应；S0 阶段的产物即 B0。

B0 工作于 2026-06-17 完成，并以 annotated git tag `B0-tag` 锁定。后续 B1a 重构 (涵盖 S1–S4 阶段) 即以此为起点。

## 阶段工作内容

本阶段未涉及任何 SHUD 源码修改，工作集中于编译环境、基准案例 (benchmark cases) 注册、输出归档与 CI 守门基础设施。具体内容包括：

1. **编译环境固化**：GCC 12、`-O2 -ffp-contract=off`，SUNDIALS 6.0.0；该工具链作为 B 阶段全部 baseline 的强制约束。
2. **基准案例注册**：7 个案例覆盖网格规模从 484 (keliya) 至约 25,000 cells (heihe_x4)；每个案例配置一份 `manifest.yaml` 描述输入路径、运行参数与 RHS 探针配置。
3. **RHS 探针 (probe) 支持**：在 SHUD 中添加编译开关，开启时可 dump RHS 中间量及计时桶 (timing bucket)；关闭时与原始源码 bitwise 一致。
4. **B0 输出归档**：6 个可实际运行的案例完成 3 次重复运行 (3-run repeatability)，SHA256 完全一致后归档至 `benchmarks/<case>/B0_output/`。
5. **平台 profile**：本地 Mac (Apple Silicon) 与服务器 (Xeon) 各完成一份 profile，并签署决策文档：原方案为 RHS-kernel-first OpenMP；针对 heihe 案例 (其 IO 占比达 79%) 单独提出 forcing IO 并行化前置策略。
6. **CI 守门**：建立 `serial-baseline.yml` workflow，每次 PR 自动执行三种 build 并对 keliya 案例做 bitwise 比对。
7. **基线锁定**：上述状态全部锁入 `B0-tag` annotated tag。

A0 阶段验收 9 项全部通过。

## 仓库交付物

由于 SHUD 源码层未做任何修改，本阶段交付物为基础设施、文档与归档：

| 类别 | 路径 / 标识 | 说明 |
|---|---|---|
| Tag | `B0-tag` → commit `884cfb13` / SHUD pin `78c37a1` | B1a 重构后所有 bitwise 对比的参照物 |
| 基准案例 | `benchmarks/` | 7 个 `manifest.yaml` + 6 份 B0 归档 (4 local + heihe / heihe_x4 server) + 12 张 snapshot 黄金参考 + kashigeer `DEFERRED.txt` |
| 工具 | `tools/` | profile 计时器、snapshot 读写比对、3-run 归档器、案例部署修复、manifest 校验 |
| 文档 | `docs/status_matrix.md` / `docs/profile_decision.md` / `docs/build_manifest.md` | 阶段 go/no-go 状态来源；profile 决策依据；编译环境合约 |
| CI | `.github/workflows/serial-baseline.yml` | 每 PR 自动 3-build + keliya bitwise vs B0 |

后续工作不应再实现上述工具，应直接复用。

## 遗留事项

本阶段结束时尚有两项遗留议题，均不阻塞 B1a 起步：

1. **heihe forcing IO 占比 79%**：RHS 并行化的 Amdahl 加速上限仅约 1.14×。`profile_decision.md` 建议将 forcing IO 并行化由原计划 P9+ 阶段提前与 P1 阶段并行进行 (master plan §1.1.1)。
2. **xinanjiang_upstream `t_other` 桶占比 22%**：当前 profile 时间桶划分粒度不足，FortranIO 初始化、mesh 加载、config 解析等几秒级固定开销未单独归类，在短时长运行 (19.7 s wall) 的小型案例中被 `t_other` 桶整体吸收。建议在 timer 中增加 `t_init` 桶以分离该开销。B1a 阶段不受影响；P1 阶段开始加速比计算前需先补全，避免分母污染。

## 后续阶段

B1a baseline 定义见 master plan §3：S0–S4 完成后的重构等价 (refactor-equivalent) 单线程结果，须与 B0 bitwise identical。当前进度详 [`docs/b1a_summary.md`](b1a_summary.md)。

## B0-tag 验证

```bash
git ls-remote --tags origin | grep B0-tag
# 95ddc375...  refs/tags/B0-tag
# 884cfb13...  refs/tags/B0-tag^{}
```
