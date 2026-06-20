# B0 Baseline 总结

> B0 = master plan §3 定义的"SHUD 原样编译的单线程结果"。本文件对应的 work stage 是 master plan §5 S0（"锁定 B0 历史基线"）。**stage = 工作阶段（S0/S1/...）**；**baseline = 工作产物（B0/B1a/...）**；S0 一对一产出 B0。
>
> 完成于 2026-06-17。`B0-tag` 已打。B1a 工作（S1 / S2 / S3 / S4）从这里起步。

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
- `docs/`：`status_matrix.md` 是阶段 go/no-go 唯一真实状态来源；`profile_decision.md` 是决策依据；`build_manifest.md` 是编译环境合约。
- `.github/workflows/serial-baseline.yml`：每个 PR 自动跑 3 种 build 然后 keliya bitwise 对比 B0。

## B0 之后的两件遗留事项

只有两件，都不阻塞 B1a 起步：

- **heihe forcing IO 占 79%**：RHS 并行化 Amdahl 上限只有 1.14×。决策文档建议把 forcing IO 并行化从 P9+ 提前到与 P1 并行排上来。
- **xinanjiang_upstream `t_other` 22%**：profile 时间桶切得不够细，FortranIO init / mesh load / config 解析这些几秒 fixed-cost 被短跑（19.7s wall）的小 case 稀释不掉，全掉进 t_other。给 timer 加一个 `t_init` 桶就分出来了。B1a 阶段不影响，P1 开始算加速比之前补一下，避免分母被污染。

## 下一步：B1a

B1a baseline 定义见 master plan §3：**"S0–S4 完成后"** 的重构等价单线程结果，必须与 B0 bitwise identical。当前进度见 [`docs/b1a_summary.md`](b1a_summary.md)。

## 验证 B0-tag

```
git ls-remote --tags origin | grep B0-tag
# 95ddc375...  refs/tags/B0-tag
# 884cfb13...  refs/tags/B0-tag^{}
```
