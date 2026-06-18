# S1 总结

> S1（B1a refactor — 唯一 RHS core）完成于 2026-06-18。`B1a-tag` 已打。可以进入 S2。

## 这一阶段做了什么

把 SHUD 的 RHS 路径从 "serial / omp 两套并行演化" 收敛到 "一份 kernel + ExecPolicy 派生"，并把 `_OPENMP_ON` 这个三义宏拆解成三正交宏，整个过程对 B0 binary 保持 100% bitwise neutrality：

- 抽 RHS core 到 `Model_Data::rhs_core(Y, DY, t, ExecPolicy)`，serial / omp 不再各自维护一份 update / flux / apply。
- 三正交宏拆分：`SHUD_ENABLE_OPENMP_RHS`（启用 OMP RHS path）、`SHUD_USE_OPENMP_NVECTOR`（CVODE 用 OpenMP NVector backend）、`SHUD_LEGACY_OMP_RHS`（保留 `_omp` 旧符号 link-in，便于回退验证）。
- 全树退役：`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 三个标识在 `SHUD/src/` 下 grep 结果归零。
- N_Vector 接口统一：`N_VGetArrayPointer` 取代 `NV_DATA_S / NV_DATA_OMP` 二选一；generic `N_VDestroy` 取代类型特化的 `N_VDestroy_Serial`，避免 §4.19 type-tag 不匹配 UB。
- CI 4-case × LEGACY_RHS 双轴 matrix 在 PR fast-feedback (2 jobs) 与 nightly cron (8 jobs) 都常开；snapshot 90d 窗口 HARD-fail gate、CVODE 15-key 全键 diff、4 个 grep gate 全部固化在 workflow。
- 把所有这些状态钉到一个新 git tag：`B1a-tag`。

## 时间线

8 个 substage 全部按 D12 收尾约束完成：

- S1a (#44) — RHS update / f_update 抽取 + 周期性 IC backup probe。
- S1b (#45) — RHS flux（PassValue 边界）+ before-PassValue 12 张 snapshot golden。
- S1c (#46) — RHS apply / river DY + 4-case 8/8 + 24/24 snapshot；negative test 验 gate 起作用。
- S1d.1 (#47) — `LEGACY_RHS` 原子开关 + `rhs_core(ExecPolicy)` 落地。
- S1d.2 (#48 + #49) — 三正交宏 + `_OPENMP_ON` 退役 + N_Vector 统一 + 4 Config 矩阵固化为 Makefile target。
- S1-ci-A (#50) — CI 4-case × LEGACY_RHS 双轴 matrix + 4 grep gates。
- S1-ci-B (#51) — fetch-tags + B0-tag smoke + snapshot 90d HARD-fail + CVODE 15-key diff + label types 扩展。
- B1a-tag (#52) — 本 PR：docs + tag，把 S1 全套 commit pin 到一个 reference point。

## 现在仓库里有什么

代码层面（vs S0）：SHUD `78c37a1` → `58327c5`；外层 commit `884cfb13` → `4ca0b04`-or-later。具体落下的东西：

- `B1a-tag`：annotated git tag，指向 #52 PR squash-merge commit、SHUD submodule pin `58327c5`。S2 起作 strict 阶段零参考点。
- `SHUD/src/`：RHS core 在 `Model_Data::rhs_core`；`_omp` 路径函数仅作 LEGACY 回退保留，不再独立演化；OpenMP RHS 真实 execution policy 留给 S2 (`StrictOMP` 当前是 `std::abort` stub per R10)。
- CI：`.github/workflows/serial-baseline.yml` 8-gate（fetch-tags / B0-tag smoke / skip-baseline-ci label 反验 / snapshot 90d / 4 grep / CVODE 15-key / nm 符号 / SHA256 compare）。
- docs：`status_matrix.md` B1a 行全 PASS；`build_manifest.md` 增 `## B1a-tag` 章节。本文件是一次性总结，**不会再更新**。

## 下一步：S2

S2 起 strict 阶段进入条件已就位：

1. B1a-tag binary 是 strict 阶段的 zero-参考点（A3a bitwise 基准从 B0 切到 B1a）。
2. CI 4-case × LEGACY_RHS 双轴 matrix 是 strict 阶段进入条件已就位（每 PR 自动 gate）。
3. heihe forcing IO 提前并行化（S0 决策遗留）可与 S2 P1 并行排另一个 issue。
4. R10 `StrictOMP` stub → 真实 RHS OpenMP execution policy 实现是 S2 主线。

开 S2 工作前花 20 分钟看：master plan §3.2（P1 范围）+ §C1（红线，"OpenMP 只是 execution policy"）+ `docs/profile_decision.md`（为什么走原方案）+ 本文件。

## 验证 B1a-tag

```
git ls-remote --tags origin | grep B1a-tag
# <object>  refs/tags/B1a-tag
# <commit>  refs/tags/B1a-tag^{}
```
