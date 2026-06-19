# S1 总结

> S1（B1a refactor — 唯一 RHS core）完成于 2026-06-18。`B1a-tag` 已打，post-tag P3 follow-up sweep 全部清空（zero open issue）。可以进入 S2。

## 这一阶段做了什么

把 SHUD 的 RHS 路径从"serial / omp 两套并行演化"收敛到"一份 kernel + ExecPolicy 派生"，把 `_OPENMP_ON` 这个三义宏拆成三正交宏，整个过程对 B0 binary 保持 100% bitwise neutrality；然后把 B1a 这件事本身从代码层延伸到 CI 闸 / 文档 / 工具层：

- 抽 RHS core 到 `Model_Data::rhs_core(Y, DY, t, ExecPolicy)`，serial / omp 不再各自维护一份 update / flux / apply。
- 三正交宏：`SHUD_ENABLE_OPENMP_RHS`（启用 OMP RHS path）、`SHUD_USE_OPENMP_NVECTOR`（CVODE 用 OpenMP NVector backend）、`SHUD_LEGACY_OMP_RHS`（保留 `_omp` 旧符号 link-in，便于回退验证）。
- 全树退役：`_OPENMP_ON` / `USE_RHS_CORE` / `N_VDestroy_Serial` 三个标识在 `SHUD/src/` 下 grep 结果归零；`N_VGetArrayPointer` 取代 `NV_DATA_S / NV_DATA_OMP` 二选一；generic `N_VDestroy` 取代特化版避免 §4.19 type-tag 不匹配 UB。
- CI 4-case × LEGACY_RHS 双轴 matrix 在 PR fast-feedback 与 nightly cron 都常开；snapshot 90d HARD-fail gate、CVODE 15-key 全键 diff、4 个 grep gate 全部固化在 workflow。
- post-tag P3 follow-up sweep（PR #71 + #72）：archive_b0_output mktemp + check_invariant_sweep meta-tool + CI invariant-sweep job + 24 张 after-PassValue golden 从 zero-payload 升级到 f_applyDY diagnostic + CI dynamic cfg.para START 解析；3 张 fixture 文档纠错（gitignored、in-place 修订）。
- 把所有这些状态钉到一个新 git tag：`B1a-tag`（指向 PR #70 squash-merge commit，SHUD pin `58327c5`）。

## 时间线

11 个 PR 串行落地（每一步都按 D12 收尾约束 + 5-step SHUD submodule push workflow）：

- S1a (#44 → PR #58) — RHS update / f_update 抽取 + 周期性 IC backup probe。
- S1b (#45 → PR #62) — RHS flux（PassValue 边界）+ before-PassValue 12 张 snapshot golden。
- S1c (#46 → PR #64) — RHS apply / river DY + 4-case 8/8 + 24/24 snapshot；negative test 验 gate 起作用。
- S1d.1 (#47 → PR #65) — `LEGACY_RHS` 原子开关 + `rhs_core(ExecPolicy)` 落地（`StrictOMP` 是 `std::abort` stub per R10，留给 S2）。
- S1d.2 (#48 → PR #66) — 三正交宏 + `_OPENMP_ON` 退役 + N_Vector 统一。
- S1d.2-configs (#49 → PR #67) — 4 Config 矩阵固化为 Makefile target（`smoke_strictomp` / `smoke_configd`）+ N_VGetVectorID probe。
- S1-ci-A (#50 → PR #68) — CI 4-case × LEGACY_RHS 双轴 matrix + 4 grep gates。
- S1-ci-B (#51 → PR #69) — fetch-tags + B0-tag smoke + snapshot 90d HARD-fail + CVODE 15-key diff + label types 扩展。
- B1a-tag (#52 → PR #70) — docs + tag：把 S1 全套 commit pin 到一个 reference point。
- P3 follow-up Group B (#56 + #57 → PR #71) — archive_b0_output mktemp + check_invariant_sweep tool + CI dynamic START parse。
- P3 follow-up Group C (#55 → PR #72) — 24 张 after-PassValue golden 从 `f_update` (zero) 重生到 `f_applyDY` (diagnostic)，零 SHUD 改动。

11 PR 一共 0 个 P0/P1 越过 verifier 进 merged main；每个 PR 都 1-2 round 内 review clean。

## 现在仓库里有什么

代码层面（vs S0）：SHUD `78c37a1` → `58327c5`；外层 commit `884cfb13` → `144989d`。具体落下的东西：

- `B1a-tag`：annotated git tag `4fafb8e570a020833395c7f57fe84eaabc7c7319`，指向 commit `64569b3fa1826122262242e7cf14686384269cc9`（PR #70 squash-merge），SHUD pin `58327c5a114052ffe8f25b6d3e2aec6b404963f2`。S2 起作 strict 阶段零参考点。
- `SHUD/src/`：RHS core 在 `Model_Data::rhs_core`；`_omp` 路径函数仅作 LEGACY 回退保留，不再独立演化；OpenMP RHS 真实 execution policy 留给 S2。
- `benchmarks/`：4 case × 24 张 snapshot golden（12 before-PassValue Qe2r_Surf + 12 after-PassValue 升级后的整合 DY） + 4 张 repeatability_snapshots_after_passvalue.txt（PR #72 新加）+ 既有 B0 archive 不变。
- `tools/`：S1 期间新增 `cvode_stats_diff/`（15-key invariance）、`server_validation/`（heihe Slurm）、`snapshot_repeatability/`（参数化 `--site={before,after}`）、`check_goldens/`（48/48 sweep）、`check_invariant_sweep.sh` + `invariants.yaml` + `test_check_invariant_sweep.sh`（P3 meta-tool 防 pattern P1 漂移）。
- CI：`.github/workflows/serial-baseline.yml` 8-gate（fetch-tags / B0-tag smoke / skip-baseline-ci label 反验 / snapshot 90d HARD-fail / 4 grep / CVODE 15-key / nm 符号 / SHA256 compare）+ 新增 `invariant-sweep` job 在 build-and-compare 前 fail-fast。
- docs：`status_matrix.md` B1a 行全 PASS；`build_manifest.md` 增 `## B1a-tag` + `#55 hook site 修正` 节，24 张 after-PassValue SHA 已更新。本文件是一次性总结，**不会再更新**。

## 进入 S2 前要处理的事

只有两件，都不阻塞 P1 起步：

- **heihe forcing IO 占 79%**（S0 决策遗留）：RHS 并行化 Amdahl 上限 1.14×，决策文档建议从 P9+ 提前到 P1 并行排另一个 issue。S1 期间未动，仍然 valid。
- **openspec/changes/s1-rhs-core-extraction tasks.md 0/84 checkoff**：S1 全程没维护 tasks.md 的 `[x]`；fixture 已被消费完毕但没 `openspec archive`。这是 housekeeping，可在 S2 任何时候补，不影响 strict 阶段进入。

> CI 范围维持 keliya × LEGACY_RHS 双轴是**有意的设计**——xinanjiang_upstream / qinyijiang / qhh 走本地 + heihe / heihe_x4 走服务器手动验证；4-case × 90-day 在 S1 期间已 16+ 次跑过 bitwise PASS（S1c #46 4-case + #48/#49 + #55 重生 + #71 invariant gate 验证）。把 forcing data 拉进 CI 既慢又贵且与本地验证冗余，不在 S2 议程内。

## 下一步：S2

S2 起 strict 阶段进入条件已就位：

1. **R10 `StrictOMP` 是 `std::abort` stub → S2 P1 第一刀**：S1d.1 落 ExecPolicy 枚举时只实现 `Serial`，`StrictOMP` / `ProductionOMP` 当前触发 abort。把真实 OpenMP execution policy 接上 `rhs_core` kernel（compute / gather 分离 + owner-local gather + fork-join 最小化）是 P1 主线工作。
2. B1a-tag binary 是 strict 阶段 zero-参考点（A3a bitwise 基准从 B0 切到 B1a）。
3. CI keliya × LEGACY_RHS 双轴 matrix 是每 PR 自动 gate；新增 invariant-sweep job 防 pattern P1 (fix-1-site-miss-N-siblings) 类回归；非 keliya case 在本地 + 服务器人工验证（4-case × 90-day 在 S1 已 16+ 次验证过 bitwise）。
4. 24 张 after-PassValue + 24 张 before-PassValue + B0 archive 三组共 60+ 个对比点全部 diagnostic（非 zero-payload），任何 S2 RHS execution policy 改动都能 byte-diff 出来。
5. heihe forcing IO 提前并行化（S0 决策遗留）可与 S2 P1 并行排另一个 issue。

开 S2 工作前花 20 分钟看：master plan §3.2（P1 范围）+ §C1（红线，"OpenMP 只是 execution policy"）+ `docs/profile_decision.md`（为什么走原方案）+ 本文件。

## 验证 B1a-tag

```
git rev-parse B1a-tag
# 4fafb8e570a020833395c7f57fe84eaabc7c7319
git rev-parse B1a-tag^{commit}
# 64569b3fa1826122262242e7cf14686384269cc9
git ls-remote --tags origin | grep B1a-tag
# 4fafb8e5...  refs/tags/B1a-tag
# 64569b3f...  refs/tags/B1a-tag^{}
git show B1a-tag --stat -- SHUD | grep '^Subproject'
# Subproject commit 58327c5a114052ffe8f25b6d3e2aec6b404963f2
```
