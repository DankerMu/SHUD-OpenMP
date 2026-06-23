# Profile 平台声明（S0-11 / openMP #15）

## 背景与定义

本文档声明 SHUD-OpenMP B0 剖析 (profile) 数据采集所依据的两个端点，对应 rhs-profile-gate spec.md "Platform declaration document" 条目（master plan §S0.12）。两端共享同一份源码（SHUD submodule `78c37a1`、outer `ecef3fb`、`SHUD_ENABLE_PROFILE=1`）、同一套编译参数（`-O2 -g -ffp-contract=off -fno-fast-math -std=c++14`，严格 IEEE-754）、以及同一个单线程 (B0) 执行策略。两端 profile 时间桶 (timing bucket) 分布的任何差异，均归因于平台本身（CPU 微架构、内存子系统、编译器），与源码或编译参数漂移无关。

本文档使用术语：参照基线 (reference baseline) 指 B0 单线程产物；目标平台 (target platform) 指 §1.1.1 加速比 gate 的权威评估端点；本地平台 (local platform) 指开发期参考端点。

## local_platform

本地平台为 Apple Silicon Mac 开发机（详见 CLAUDE.md "双端实验环境"），其配置如下：

| 字段 | 值 |
|---|---|
| os | Darwin 24.6.0 arm64 (xnu-11417.140.69~1) |
| cpu | Apple M4 Pro；14 物理核（14 逻辑核，1 thread/core） |
| numa | 统一内存架构（无 NUMA 分区） |
| compiler | Apple clang 17.0.0 (clang-1700.6.3.2) |
| flags | -O2 -g -ffp-contract=off -fno-fast-math -std=c++14 |
| binary_path | SHUD/shud（serial B0；SHUD_ENABLE_PROFILE=1） |
| 定位 | 开发与 S0 预热阶段；不作为 §1.1.1 加速比 gate 的权威证据 |

## target_platform

目标平台为服务器端 Slurm 计算节点（详见 CLAUDE.md "双端实验环境" 及 "要在服务器上工作时" 章节），为 §1.1.1 量化 gate 的权威端点。其配置如下：

| 字段 | 值 |
|---|---|
| os | Linux 6.8.0-90-generic x86_64 |
| cpu | Intel(R) Xeon(R) Gold 6133 CPU @ 2.50GHz；2 socket × 20 core（40 逻辑核，1 thread/core） |
| numa | 2 节点（node0 cpus 0-19、node1 cpus 20-39）；每节点约 96 GB |
| compiler | gcc 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1) |
| flags | -O2 -g -ffp-contract=off -fno-fast-math -std=c++14（serial B0；无 -fopenmp） |
| binary_sha256 | 6808894b52ed79669ea563451377f99096c37992d890fad09219015da0773c98 |
| slurm_node | cn08（6 次 profile 跑均落在 cn08，经由 Slurm CPU 分区调度） |
| 定位 | §1.1.1 加速比 gate 的证据来源及最终 P-prod 签字端点 |

关于 master plan §5 S0.12 中 "目标平台 = 单插槽 8 物理核" 的预期需做如下说明：实际服务器节点 cn08 为双插槽 20 核架构。对 B0 profile（单线程、无 OpenMP）而言，插槽与 NUMA 布局并不影响 bucket 分布——任务整体运行在 Slurm 绑定的单核上。P1–P7 strict-parallel 运行则需通过 `numactl -N 0` 绑定至单插槽，以匹配 "单插槽 8 物理核" 的验证目标；该约束叠加于 #15 之上，不属于本 issue 范围。

## decision_consistency

下表比较两端的 `t_RHS_total / t_wall_total` 比值（即驱动 Amdahl 上界与并行优先级决策的核心指标），覆盖同时具有 local 与 target 真实 `profile_B0.yaml` 的 4 个 case。其余 3 个 case 的状态为：

1. `heihe` — local 端 DEFERRED（forcing 数据在本地开发机上过大），target 端 REAL，因此无 local-vs-target 差异可计算。
2. `heihe_x4` — 同 heihe（case × endpoint 矩阵规定为 server-only），target 端 REAL，无 local-vs-target 差异可计算。
3. `kashigeer` — 两端均 DEFERRED，因上游 forcing 数据缺口（issue #29），两端均无 profile 数据。

两端均有数据的 case 对比结果如下：

| Case | local t_RHS%/t_wall% | target t_RHS%/t_wall% | delta (pp) | wall_local (s) | wall_target (s) |
|---|---|---|---|---|---|
| keliya | 36.32% | 49.33% | **+13.01** | 27.8 | 79.7 |
| xinanjiang_upstream | 44.72% | 43.74% | -0.98 | 4.1 | 19.7 |
| qinyijiang | 51.07% | 64.64% | **+13.57** | 229.5 | 799.9 |
| qhh | 38.00% | 36.75% | -1.25 | 75.7 | 215.8 |

差异汇总指标：

| 指标 | 值 |
|---|---|
| max_abs_delta_pp | 13.57（qinyijiang） |
| cases_over_10pp_threshold | 2 / 4（keliya、qinyijiang） |
| **delta_acceptable** | **false** |

按 spec.md 场景 "> 10pp 差异触发 review note" 的规定，由于 `max_abs_delta_pp > 10`，本文档必须与 `docs/profile_decision.md` 中 "跨平台差异审查" 章节配对，对偏差成因及其对 gate 决策的影响给出说明（详见 `docs/profile_decision.md`）。

## SHA256 摘要

Local 端 profile 产物（4 真实 + 3 deferred）的 SHA256 摘要如下：

```
b16e8a9acedcff00c82b66192d3db3eced538c7718c8715adff7b384761ce14f  benchmarks/keliya/profile_B0.yaml
34831b4b641f664cece3c5a4db1dd2c72b0016b99dc4206d30ac9b5365a43d97  benchmarks/xinanjiang_upstream/profile_B0.yaml
77bdbad6bd23e9806688bb7e76cb82d8359ed3edfbc7af03002ea9ee8ad87b02  benchmarks/qinyijiang/profile_B0.yaml
9506ceeeab796c9685dccfd649f5adc889e239e3df2f3274d5a82647637cec08  benchmarks/qhh/profile_B0.yaml
cbe50adf2047c88bc1e7d6415ce76c6ba8310e159310876c72f2bebdbc24982c  benchmarks/heihe/profile_B0.deferred.yaml
aea1322a9b2b6ee2ffaf391b422ccf05e8c1c7f29f5d20a9778dc6568eed8a9f  benchmarks/heihe_x4/profile_B0.deferred.yaml
5bdecf4e2173868de20659141cb9b046349cc5ae0fba97078b877f1cd4a5cd02  benchmarks/kashigeer/profile_B0.deferred.yaml
```

Target 端 profile 产物（6 真实 + 1 deferred）的 SHA256 摘要如下：

```
711a380902d2dee176ff16bf5c3a5c360a9ee131420d7727a7d4e75dc62ca0f5  benchmarks/keliya/profile_B0.target.yaml
a739dfd7c66310bf5e5bcb0317a99768d3c1d41480e8e991e0d32aaeca9637e1  benchmarks/xinanjiang_upstream/profile_B0.target.yaml
1dae17564e44de5149f8e49cb8dd3f404caa5a1ee19dc0b9ef2f26ab417174ed  benchmarks/qinyijiang/profile_B0.target.yaml
cc312b7ab1db926ab85fff86b91cc0e29fc02b2a289103ee30db9555dad105f5  benchmarks/qhh/profile_B0.target.yaml
baa03be7ce16e01345bdc9e9b93c033ffcee55213113b9b1ba91441414a97f5d  benchmarks/heihe/profile_B0.target.yaml
03d9d4c9def804b27f5f5e6a8930063eb03ce5ad5cbadee979c848f829254c36  benchmarks/heihe_x4/profile_B0.target.yaml
8f64779b5c3c25b2a854f70b9721231a4e40b5dd4a1b2eadf2e3a7e43d615d17  benchmarks/kashigeer/profile_B0.target.deferred.yaml
```
