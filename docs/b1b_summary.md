# B1b 基线总结

## 背景与定义

B1b 基线指 master plan §3 所定义的"B1a + S5* 结构改造 + S6b 缺陷修复后的可并行就绪单线程参照 (parallel-ready serial reference)"。本项目中 **stage 指工作阶段**（S5a/S5b/S5c/S5d/S6b/S6c），**baseline 指工作产物**（B0/B1a/B1b），二者不存在一一对应关系；B1b 并非任何单一 stage 的产物，而是 S5a + S5b + S5c + S5d + S6b 全部完成后才能签证的检查点。S6c 为 lock 与 capstone 阶段。

截至 PROMOTE 时点，S5–S6b–S6c 已全部完成（PR-1 #191 至 PR-16 #207 已合入 `baseline/B1b`；PR-17 = 本 PROMOTE PR 落 main）。B1b epic 编号为 #172。

## 旧版错误复盘（不适用）

B1b 不存在旧版 `s5_summary.md` / `s6_summary.md` 的历史错误（本文件首版即采用 7 节结构，并严格遵循 `docs/b1a_summary.md` 中"B1a 概念坐标"错位修正经验）。S5/S6b 整段亦未出现"将单一 stage 等同于 B1b"的早期 status-matrix 误签问题——`docs/status_matrix.md` 中 B1b 行自 epic 起即标 PENDING，直至 #190 PROMOTE 才转为 PASS（CONDITIONAL ship）。

## 完成定义

B1b 完成定义与实际交付对照如下：

| master plan B1b 契约（§3 + §S5 + §S6b） | 实际交付 |
|---|---|
| S5a — forcing thread-safety audit + TimeSeriesData 注释（audit-only） | #176 PR #192 |
| S5b — scratch arrays + lake reset 顺序 + RHS print 重组（audit-only + 序结构） | #177 PR #193 |
| S5c-A — CVODE 7 stats + SHUD_ENABLE_DIAGNOSTICS gate | #173 PR #191 |
| S5c-B — RHS 7-bucket timer + forcing I/O + heihe_x4 780 s 验证 | #174 PR #194 |
| S5c-C — nFCall 与 nfe 分离 + 15-key CI gate | #175 PR #195 |
| S5d.1 — ElementHotData SoA + manifest + grep gate | #178 PR #196 |
| S5d.2-5a — jagged QeleSurf/QeleSub flatten + ASan/UBSan CI | #179 PR #197 |
| S5d.2-5b — selective small array SoA + Riv/RivSeg audit | #180 PR #198 |
| S5d.3 — parallel first-touch + [NUMA] log token | #181 PR #199 |
| S5d.4 — tools/run_omp.sh + manifest omp_env + numa_check | #182 PR #200 |
| S5d 汇总 — sizeof + cache miss + NUMA 加速比 + ADR + glossary | #183 PR #201 |
| S6b.1 — AccTemperature guard + cryosphere NaN | #184 PR #202 |
| S6b.2 — lake formula SKIP path（master plan §S6b L1497 FORECAST + C8 forward-compat） | #186 PR #206 |
| S6b.3 — S2 follow-up bug audit | #187 PR #203 |
| S2.17 lake formula PI 审查 — evidence pack（PI sign-off OPEN） | #185 PR #204 |
| S6c-12a — B1b 3-run + B0 vs B1b 水量平衡 + Go/No-Go 7 项 evidence | #188 PR #207 |
| S6c-12b — B1b-tag annotated 创建 + push + baseline/B1b 分支 lock | #189（local + tag push + branch protection） |
| S6c-12c — b1b_summary + status_matrix + archive + PROMOTE + jsonl 双追加 | #190 PR #2XX（本 PR，base=main） |

## `B1b-tag` 的处理

`B1b-tag` 的发布约束如下：

1. `B1b-tag` 指向 commit `18a0c9085f494d1cf228c7be4adf27d9132d05dd` / SHUD pin `71b3a1ae4ef82e165134a18469c7d0a79284b67f`（openmp-baseline 分支）。
2. annotated tag object SHA = `96e224daad8cb9c93f855851724f8d45468391c2`。
3. **D11 强制约束：一次锁死禁止 force-update**（与 B1a-tag 的 force-update 历史不同；B1b 一次到位，不允许后续 retroactive update）。
4. `baseline/B1b` 分支保护已锁定：`lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false`。
5. `B1b-tag` 所指 commit (`18a0c908`) 即 #188 S6c-12a 落地 + #188 post-merge log append 之后的 HEAD；docs PROMOTE（本 #190 PR）在 main 侧推进，不进入 `B1b-tag` 内部——B1b-tag 凝固于 evidence + log append 状态。

## B1b 完成时间线

B1b 由 PR-1 #191 至 PR-16 #207 加 tag-only #189 与本 PROMOTE PR (#190) 构成：

- PR-1 #191 [S5c-A #173]：CVODE 7 stats + SHUD_ENABLE_DIAGNOSTICS gate。
- PR-2 #192 [S5a #176]：forcing thread-safety audit + TimeSeriesData 注释（audit-only）。
- PR-3 #193 [S5b #177]：scratch arrays + lake reset 顺序 + RHS print 重组。
- PR-4 #194 [S5c-B #174]：RHS 7-bucket timer + forcing I/O + heihe_x4 780 s。
- PR-5 #195 [S5c-C #175]：nFCall 与 nfe 分离 + 15-key CI gate。
- PR-6 #196 [S5d.1 #178]：ElementHotData SoA + manifest + grep gate。
- PR-7 #197 [S5d.2-5a #179]：jagged QeleSurf/QeleSub flatten + ASan/UBSan CI。
- PR-8 #198 [S5d.2-5b #180]：selective small array SoA + Riv/RivSeg audit。
- PR-9 #199 [S5d.3 #181]：parallel first-touch + [NUMA] log token。
- PR-10 #200 [S5d.4 #182]：tools/run_omp.sh + manifest omp_env + numa_check。
- PR-11 #201 [S5d 汇总 #183]：sizeof + cache miss + NUMA 加速比 + ADR + glossary。
- PR-12 #202 [S6b.1 #184]：AccTemperature guard + cryosphere NaN。
- PR-13 #203 [S6b.3 #187]：S2 follow-up bug audit。
- PR-14 #204 [S2.17 #185]：lake formula PI 审查 evidence pack（无 PI sign-off）。
- PR-15 #206 [S6b.2 #186]：SKIP path 实现。
- PR-16 #207 [S6c-12a #188]：B1b 3-run + B0 vs B1b 水量平衡 + Go/No-Go 7 项。
- (#189) S6c-12b：B1b-tag 创建 + push + 分支 lock（无 PR；local 操作）。
- PR-17 [S6c-12c #190 本 PR]：b1b_summary + status_matrix + archive + PROMOTE + jsonl 双追加。

## S5 + S6b 后续 hand-off

### S5a — forcing thread-safety audit（#176, master plan §S5a）

仅做 audit；TimeSeriesData / forcing pipeline 的状态机注释与多线程入场点说明已落地。未触动 SHUD/src 数值代码。B1a → B1b 在此 axis 上 bitwise neutral。

### S5b — scratch arrays + lake reset 顺序（#177, master plan §S5b）

完成 scratch arrays 所有权归并、lake reset 顺序前置（参考 S2.7 lake reset / S2.8 PassValue 临时扩 lake gather 模式）以及 RHS print 重组。B1a → B1b 在此 axis 上 bitwise neutral：服务器 cn03 验证 heihe（JobId 8678，wall 479 s）与 heihe_x4（JobId 8679，wall 1192 s）的 `.rivqdown.dat` SHA256 与 PR-12 B1a-tag golden 逐字节一致（已在 issue #177 补录 server validation）。

### S5c — solver diagnostics（#173 + #174 + #175, master plan §S5c）

1. S5c-A：CVODE 7 个 integrator stats hook + 编译期 gate `SHUD_ENABLE_DIAGNOSTICS`。
2. S5c-B：RHS 7-bucket timer + forcing I/O segregation + heihe_x4 780 s 性能基线。
3. S5c-C：`nFCall`（user RHS）与 `nfe`（CVODE integrator）分离 + `tools/cvode_stats_diff/` 15-key CI gate（excludes nfcall）。

### S5d — data layout SoA + NUMA（#178 至 #183, master plan §S5d）

1. S5d.1：`ElementHotData` SoA + 平 3 行主序 idiom + `sync_hot_dynamic(i)` AoS→SoA mirror refresh + manifest + grep gate。
2. S5d.2-5a：jagged `QeleSurf` / `QeleSub` flatten + ASan/UBSan CI 全开。
3. S5d.2-5b：selective small array SoA + Riv/RivSeg audit（保留 AoS 但增加 audit footprint）。
4. S5d.3：parallel first-touch + `[NUMA] PARALLEL_FIRST_TOUCH` log token。
5. S5d.4：`tools/run_omp.sh` env discipline + manifest `omp_env` 节 + `tools/numa_check.sh`。
6. S5d 汇总：sizeof scan + cache miss profile + NUMA 加速比 + ADR-0001（SoA hot fields）+ `openspec/glossary.md` 术语条目。

### S6b — bug fix application（#184 + #186 + #187, master plan §S6b）

1. S6b.1：AccTemperature guard（积温保护）+ cryosphere NaN 边界 fence。
2. S6b.2：lake formula **SKIP path**（master plan §S6b L1497 FORECAST + C8 forward-compat；未触动 SHUD/src；详 `SHUD/B1b_CHANGELOG.md` S6b.2 行"source byte-identical to B1a"）。
3. S6b.3：S2 follow-up bug audit（#159 verdict: NOT-A-BUG，无 fix action）。
4. S2.17 (#185)：lake formula PI 审查 — evidence pack 已投出，PI sign-off OPEN；spec L23 + design Q1 将 E1/E2 verdict 保留给 PI；本次 B1b ship 不视为已签 E2。

## S6c-12a B1b capstone 验证结果（2026-06-22）

**Mac 4-case 3-run SHA-identical，canonical summary SHA ≡ B0 `repeatability.txt sha256_run1`**：

| Case | summary SHA256（3 runs 全一致） | wall (s) | ≡ B0-tag |
|---|---|---|---|
| keliya | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | 26/26/27 | YES |
| xinanjiang_upstream | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | 4/5/4 | YES |
| qinyijiang | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | 245/244/239 | YES |
| qhh (lake, 5 outputs) | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | 89/89/88 | YES |

Per-file SHA 与 `benchmarks/<case>/B0_output/<file>` 交叉核对：**13 个文件 byte-identical**，逐文件枚举如下。

| Case | file | B1b run-3 SHA256 | B0 archive SHA256 | match |
|---|---|---|---|---|
| keliya | `keliya.rivqdown.dat` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | `89686fb8…99a8fc` | YES |
| keliya | `cvode_stats.txt` | `fdf8662c022620b7f04a5f2d994440065ac559f57c9245ae347bff7c8a190e57` | `fdf8662c…a190e57` | YES |
| xinanjiang_upstream | `xinanjiang.eleygw.dat` | `f6e86f013f4f92d1c99429eafb27ec38cc7fc417e6d7d9aeef1725f8fa0a46a1` | `f6e86f01…0a46a1` | YES |
| xinanjiang_upstream | `xinanjiang.rivqdown.dat` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | `3794e7d3…4023020` | YES |
| xinanjiang_upstream | `cvode_stats.txt` | `77196a7da79b94176306eaa806580d810ad9b26bcc7b3ec43e4ae8c86496a097` | `77196a7d…496a097` | YES |
| qinyijiang | `nanlin.rivqdown.dat` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | `48036c5e…6dfbc57` | YES |
| qinyijiang | `cvode_stats.txt` | `58f36d72bbb7141c09491b4df4fb9de69c6d7cfa786fa062fc60ea4fb57ab164` | `58f36d72…57ab164` | YES |
| qhh (lake) | `qhh.rivqdown.dat` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | `d9a42798…fa7b62ce` | YES |
| qhh (lake) | `qhh.lakystage.dat` | `4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250` | `4fcebe3a…2e0ca250` | YES |
| qhh (lake) | `qhh.lakqrivin.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db738…d575ab2d` | YES |
| qhh (lake) | `qhh.lakqrivout.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db738…d575ab2d` | YES |
| qhh (lake) | `cvode_stats.txt` | `91df2bcf9b4aa48cbafa50dfde15983a0f7b797083f82e3416454494a8a957f9` | `91df2bcf…8a957f9` | YES |

合计：2 (keliya) + 3 (xinanjiang) + 2 (qinyijiang) + 5 (qhh) + 1 (heihe cvode in T2) = **13 个 byte-identical 文件**（cvode_stats 算法每 case 各一份）。详见 `docs/B0_vs_B1b_water_balance_report.md` 中进一步 6/6 closure-error PASS。

**服务器 2-case 3-run SHA-identical（cn03 节点，遵守 Slurm 三项强制约束）**：

| Case | Slurm JobId（3 runs sequential afterany） | wall (s) per run | summary SHA256 | rivqdown ≡ B0/B1a golden |
|---|---|---|---|---|
| heihe | 8662 / 8663 / 8664 (cn03) | 480 / 479 / 480 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | YES (`55abad28…`) |
| heihe_x4 | 8665 / 8666 / 8667 (cn03) | 1196 / 1192 / 1191 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | YES (`f90601ef…`) |

**水量平衡（`docs/B0_vs_B1b_water_balance_report.md`）**：bitwise identity 蕴含 closure-error delta = 0（逐 bit 比较）于全部 6 个案例（4 Mac + 2 Server），远低于 spec 0.1% 相对容差门限。

**Go/No-Go 7 项 checklist（spec b1b-capstone）**：

| # | 检查项 | verdict |
|---|---|---|
| 1 | B1b-tag 创建 + 锁定 | PASS（#189 完成于本 PROMOTE PR 之前；详下文"验证 B1b-tag"）|
| 2 | 6-case 3-run repeatability | PASS（T1 + T2）|
| 3 | 6-case bitwise vs B0 | PASS（4 Mac canonical SHA ≡ B0 + 2 Server SHA ≡ B0/B1a golden）|
| 4 | 水量平衡不恶化 | PASS（closure_error delta = 0，逐 bit）|
| 5 | S6b 子任务全闭环 | PASS（#184 / #186 / #187 全部 closed）|
| 6 | SHUD `B1b_CHANGELOG.md` 完整 | PASS（12 节 S5a–S6b.3 全部在 openmp-baseline）|
| 7 | CI 全绿（serial-baseline 4 job set） | PASS（PR-16 #207 CI 4/4 SUCCESS）|

PROMOTE 时点 7/7 PASS。

## B1b ship status — CONDITIONAL → UNCONDITIONAL (PR-19 #210 PI E2 sign-off + #205 cleared)

B1b 整体已由 CONDITIONAL ship 升级为 **UNCONDITIONAL ship**（PR-19 #210 合入后生效）。caveat 清单与处置（详 `SHUD/B1b_CHANGELOG.md` S6b.2/S6b.4/S6b.5 + `docs/s217_lake_formula_audit.md` §E）：

1. **#185（S2.17 lake formula PI 审查）— RESOLVED (E2 signed)**：DankerMu 作为 `SHUD-System/SHUD` upstream organization owner 在 PR-19 #210 中签署 E2 "formula correct, no change"。design.md Open Q1 同步关闭（PI delegate qualification = upstream-org ownership；three-surface sign-off pattern = issue comment + audit doc §E + SHUD CHANGELOG addendum row）。详 `docs/s217_lake_formula_audit.md` §E.1 verdict statement 与 §E.2 Open Q1 resolution。
2. **#186（S6b.2）— CLOSED-via-PI-E2 (retroactively consistent)**：PR-15 #206 的 SKIP path 在 #185 E2 sign-off 后，由"FORECAST per C8 forward-compat"升级为"consistent with signed PI E2 directive"，等价于 spec.md L29-31 Scenario "审查结论已签字跳过修改"的事后满足。
3. **#205（`rhs_flux` lake pass-1 中 SoA/AoS sync drift）— RESOLVED (post-B1b cleanup before P1)**：SHUD commit `de75743`（fix）+ `9a376f7`（CHANGELOG 行）落于 `openmp-baseline`；外层 PR-18 #209（pointer bump `71b3a1a` → `9a376f7` + docs sync）已合入 `main`；4-case Mac（keliya / xinanjiang_upstream / qinyijiang / qhh）2-run canonical SHA 与 B1b-tag baseline 逐字节一致（bitwise-neutral on B1b benchmarks）；P-strict 前置条件清除。此 fix 同时增强了 #185 E2 verdict —— audit doc §A.4/§B.4 中 strict-reading 之忧由 SoA-sync 修复而消解（lake-side `u_effKH` 当前 SoA/AoS-coherent，正确 reflect 由 `updateLakeElement` 写入的 `KsatH` 意图）。依据 D11，此项不追溯纳入 B1b（B1b-tag annotated message 创建于 #189 immutable 时刻；本节"B1b ship status"为 post-tag doc surface 权威）。
4. **D9 fast-path trigger #2**（S2.17 审查为 'no change' 跳过 fix）— **TRIGGERED（本 PR PR-19 #210）**。在 #205 RESOLVED + #185 E2 sign-off 后双门均通：`B1-tag` annotated tag 已创建并 alias 至 `main` HEAD（含 #205 cleanup + PI E2 sign-off）；`B1a-tag`（`f7f992c…`）与 `B1b-tag`（`18a0c908…`）保持 immutable per D11，不作 force-update；下游 P1+ 工作 SHOULD 以 `B1-tag` 作为 canonical "B1 baseline" 引用。
5. **C8 forward-compat（master plan "永不 break userspace"）— UNUSED FOR THIS SHIP**：PI 签 E2 不签 E1，B1c-tag stacking 未触发。C8 仍为 codebase convention；未来若 P-strict 阶段任何 finding overrule E2 verdict，可走 B1c-tag stacking 继续 honour D11。

> **immutable-tag 与 docs 之间的已知 minor 漂移（历史保留）**：`B1b-tag` 的 annotated message（#189 push 完成后已 immutable per D11）在创建时刻所列 4 项 caveat（`#185` / `#205` / `#186` / `C8`）当时全部 OPEN/CONDITIONAL；本节"B1b ship status"（post-tag doc surface）反映的是 PR-19 #210 sign-off 之后的 RESOLVED/UNCONDITIONAL 状态。三处 doc surface（本文 + `docs/status_matrix.md` + `docs/build_manifest.md`）作为权威；tag message 为 commit-pinned snapshot；`B1-tag` annotated tag（本 PR 新增）作为"post-cleanup signed-off" canonical pointer，下游 P1+ 优先使用 `B1-tag`。

## 验证 `B1b-tag`

```
git ls-remote --tags origin | grep B1b-tag
# refs/tags/B1b-tag        96e224daad8cb9c93f855851724f8d45468391c2  ← annotated tag object SHA
# refs/tags/B1b-tag^{}     18a0c9085f494d1cf228c7be4adf27d9132d05dd  ← dereferenced commit SHA（SHUD pin 71b3a1ae）

git show B1b-tag --no-patch --format=fuller
# Tagger:     DankerMu <mumzy@mail.ustc.edu.cn>
# 17-bullet S5*+S6b 列表 / SHUD pin 71b3a1ae / zero-impact: yes / CONDITIONAL ship caveats

gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/B1b/protection --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'
# {"allow_deletions":false, "allow_force_pushes":false, "enforce_admins":true, "lock_branch":true}
```

B1b-tag 一次锁死，禁止 force-update（D11）；任何后续 retroactive 更新（如 PI 后续 sign-off on #185）走 forward-compat B1c-tag stacking 路径（C8）。
