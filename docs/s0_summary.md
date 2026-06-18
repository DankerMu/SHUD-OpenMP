# S0 总结

> S0（baseline lock）完成于 2026-06-17。`B0-tag` 已打。可以进入 S1。

## 这一阶段做了什么

把项目从"裸 SHUD 源码"推进到"一份可被任何后续重构当作参照物的基线"：

- 锁住编译环境（GCC 12、`-O2 -ffp-contract=off`、SUNDIALS 6.0.0）。
- 注册 7 个 benchmark 案例（最小 484 cells 的 keliya 到最大 ~25k 的 heihe_x4），每个写一份 `manifest.yaml`。
- 让 SHUD 二进制能在编译开关 ON 时 dump RHS 中间量和时间桶，OFF 时与原始 SHUD bitwise 一致。
- 把 6 个可实跑案例的 B0 输出归档进 `benchmarks/<case>/B0_output/`，每个都 3 次跑过、SHA256 完全一致。
- 在两个平台（本地 Mac + 服务器 Xeon）各跑一份 profile，签了一份决策文档：原方案 RHS-kernel-first OpenMP，heihe 因 IO 占 79% 单独拉出来提前并行化。
- 建了一个 CI workflow 守门，开了 branch protection。
- 把所有这些状态钉到一个 git tag：`B0-tag`。

A0 验收 9 项全过。

## 现在仓库里有什么

**代码层面什么都没改**——S0 是基础设施 + 文档 + 历史归档。具体落下的东西：

- `B0-tag`：annotated git tag，指向 commit `884cfb13`、SHUD submodule pin `78c37a1`。这是 B1a 重构后所有 bitwise 对比的参照物。
- `benchmarks/`：7 个 manifest + 6 份 B0 archive（4 local + heihe / heihe_x4 server）+ 12 张 snapshot golden + kashigeer 一份 `DEFERRED.txt` 解释为何跑不动。
- `tools/`：profile 计时器、snapshot 写入 / 对比、3-run 归档器、case 部署修复、manifest 校验。后续工作直接复用，不要再写一遍。
- `docs/`：`status_matrix.md` 是阶段 go/no-go 唯一真实状态来源；`profile_decision.md` 是决策依据；`build_manifest.md` 是编译环境合约。本文件是一次性总结，**不会再更新**。
- `.github/workflows/serial-baseline.yml`：每个 PR 自动跑 3 种 build 然后 keliya bitwise 对比 B0。`baseline/current` 现在要求这个 check 通过才能 merge。

## 已知没解决的问题

- **kashigeer** 跑不了：上游 NWM 数据缺 X76-X80 forcing band，双端都缺。spec 已经把它标成 `deferred-upstream`，A0 验收排除它。要么重新拉数据，要么换 case 集合，二选一不在工程范围内。
- **heihe forcing IO 占 79%**：RHS 并行化 Amdahl 上限只有 1.14×。决策文档建议把 forcing IO 并行化从 P9+ 提前到与 P1 并行做。
- **xinanjiang_upstream `t_other` 22%**：是 startup 摊销，不是 profile 工具坏。等 timer 加一个 `t_init` 桶就分得清。
- **CI 跑不到真正的对比**：GitHub runner 上没部署 keliya 的 forcing 数据，CI 只到 build 步就停。早期 S1 应该解决这个。
- 还有几个文档漂移 / yaml 子树形状的小事，不影响 B1a 起步。

## 下一步：S1

S1 的目标是 B1a——把 SHUD 现在的 serial / omp 两套 RHS 实现合并成同一份代码，instrumentation 关掉时和 B0 **bitwise identical**。这不是改物理、不是改算法，纯重构。

推荐做法：

1. 从 keliya 开始（484 cells，30 秒跑完）。bitwise 不过的时候迭代成本最低。
2. 重构通过后扩到 xinanjiang_upstream → qinyijiang → qhh。heihe / heihe_x4 只在服务器上验。
3. 把 `B0-tag` 引进 CI workflow：每个 PR 跑 `git show B0-tag:benchmarks/<case>/B0_output/<file>` 和当前输出做 SHA256 对比。
4. heihe forcing IO 提前的事可以和 B1a 并行排另一个 issue 上手。

开 S1 工作前花 20 分钟看：master plan §3.1（B1a 范围）+ §C1（红线）+ `docs/profile_decision.md`（为什么走原方案）+ `docs/status_matrix.md`（当前状态）。

## 验证 B0-tag

```
git ls-remote --tags origin | grep B0-tag
# 95ddc375...  refs/tags/B0-tag
# 884cfb13...  refs/tags/B0-tag^{}
```
