# B1b Baseline 完成

> B1b = master plan §3 定义的 "B1a + S5* 结构改造 + S6b bug fix 后的 parallel-ready serial reference"。**stage = 工作阶段（S5a/S5b/S5c/S5d/S6b/S6c）**；**baseline = 工作产物（B0/B1a/B1b）**；B1b 不是单一 stage 的产物，而是 **S5a + S5b + S5c + S5d + S6b 全部完成后**才能签字的检查点。S6c 是 lock + capstone 阶段。
>
> S5–S6b–S6c 全部完成（PR-1 #191 through PR-16 #207 已 merged 进 `baseline/B1b`；PR-17 = 本 PROMOTE PR 落 main）。B1b epic = #172。

## 旧版错误复盘（不适用）

B1b 无旧版 `s5_summary.md` / `s6_summary.md` 历史错误（本文件首版即 7-section 结构 + 严格遵守 docs/b1a_summary.md "B1a 概念坐标" 错位修正经验）。S5/S6b 整段无 "把单一 stage 等同于 B1b" 的早期 status-matrix 误签问题——`docs/status_matrix.md` B1b 行从 epic 开始即标 PENDING，直到 #190 PROMOTE 才转 PASS（CONDITIONAL ship）。

## 完成定义

| master plan B1b 契约（§3 + §S5 + §S6b） | 实际做的 |
|---|---|
| S5a — forcing thread-safety audit + TimeSeriesData 注释（audit-only） | #176 PR #192 |
| S5b — scratch arrays + lake reset 顺序 + RHS print 重组（audit-only + 序结构） | #177 PR #193 |
| S5c-A — CVODE 7 stats + SHUD_ENABLE_DIAGNOSTICS gate | #173 PR #191 |
| S5c-B — RHS 7-bucket timer + forcing I/O + heihe_x4 780s validation | #174 PR #194 |
| S5c-C — nFCall vs nfe 分离 + 15-key CI gate | #175 PR #195 |
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
| S6c-12a — B1b 3-run + B0 vs B1b water balance + Go/No-Go 7 项 evidence | #188 PR #207 |
| S6c-12b — B1b-tag annotated 创建 + push + baseline/B1b 分支 lock | #189（local + tag push + branch protection） |
| S6c-12c — b1b_summary + status_matrix + archive + PROMOTE + jsonl 双追加 | #190 PR #2XX（本 PR，base=main） |

## `B1b-tag` 的处理

- `B1b-tag` = `18a0c9085f494d1cf228c7be4adf27d9132d05dd` / SHUD pin `71b3a1ae4ef82e165134a18469c7d0a79284b67f`（openmp-baseline 分支）。
- annotated tag object SHA = `96e224daad8cb9c93f855851724f8d45468391c2`。
- **D11 强制：一次锁死禁止 force-update**（与 B1a-tag force-update 历史**不同**：B1b 一次到位，不允许后续 retroactive update）。
- `baseline/B1b` 分支 protection 已 lock：`lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false`。
- `B1b-tag` 指向的 commit (`18a0c908`) 即 #188 S6c-12a 落地 + #188 post-merge log append 之后的 HEAD；docs PROMOTE（本 #190 PR）在 main 侧推进，**不进 `B1b-tag` 内部**——B1b-tag 凝固在 evidence + log append 状态。

## B1b 完成时间线

PR-1 #191 through PR-16 #207 + tag-only #189 + 本 PROMOTE PR (#190)：

- PR-1 #191 [S5c-A #173] — CVODE 7 stats + SHUD_ENABLE_DIAGNOSTICS gate
- PR-2 #192 [S5a #176] — forcing thread-safety audit + TimeSeriesData 注释（audit-only）
- PR-3 #193 [S5b #177] — scratch arrays + lake reset 顺序 + RHS print 重组
- PR-4 #194 [S5c-B #174] — RHS 7-bucket timer + forcing I/O + heihe_x4 780s
- PR-5 #195 [S5c-C #175] — nFCall vs nfe 分离 + 15-key CI gate
- PR-6 #196 [S5d.1 #178] — ElementHotData SoA + manifest + grep gate
- PR-7 #197 [S5d.2-5a #179] — jagged QeleSurf/QeleSub flatten + ASan/UBSan CI
- PR-8 #198 [S5d.2-5b #180] — selective small array SoA + Riv/RivSeg audit
- PR-9 #199 [S5d.3 #181] — parallel first-touch + [NUMA] log token
- PR-10 #200 [S5d.4 #182] — tools/run_omp.sh + manifest omp_env + numa_check
- PR-11 #201 [S5d 汇总 #183] — sizeof + cache miss + NUMA 加速比 + ADR + glossary
- PR-12 #202 [S6b.1 #184] — AccTemperature guard + cryosphere NaN
- PR-13 #203 [S6b.3 #187] — S2 follow-up bug audit
- PR-14 #204 [S2.17 #185] — lake formula PI 审查 evidence pack（无 PI sign-off）
- PR-15 #206 [S6b.2 #186] — SKIP path implementation
- PR-16 #207 [S6c-12a #188] — B1b 3-run + B0 vs B1b water balance + Go/No-Go 7 项
- (#189) S6c-12b — B1b-tag 创建 + push + branch lock（无 PR；local 操作）
- PR-17 [S6c-12c #190 本 PR] — b1b_summary + status_matrix + archive + PROMOTE + jsonl 双追加

## S5 + S6b 后续 hand-off

### S5a — forcing thread-safety audit（#176, master plan §S5a）

audit-only；TimeSeriesData / forcing pipeline 状态机注释 + 多线程入场点说明落地。**未触 SHUD/src 数值代码**。B1a → B1b 此 axis bitwise neutral。

### S5b — scratch arrays + lake reset 顺序（#177, master plan §S5b）

scratch arrays 所有权归并 + lake reset 顺序前置（参考 S2.7 lake reset / S2.8 PassValue 临时扩 lake gather pattern）+ RHS print 重组。**B1a → B1b 此 axis bitwise neutral**：服务器 cn03 验证 heihe (8678 wall 479s) + heihe_x4 (8679 wall 1192s) `.rivqdown.dat` SHA256 与 PR-12 B1a-tag golden byte-identical（在 issue #177 补录 server validation）。

### S5c — solver diagnostics（#173 + #174 + #175, master plan §S5c）

- S5c-A: CVODE 7 个 integrator stats hook + `SHUD_ENABLE_DIAGNOSTICS` 编译期 gate。
- S5c-B: RHS 7-bucket timer + forcing I/O segregation + heihe_x4 780s 性能 baseline。
- S5c-C: `nFCall` (user RHS) vs `nfe` (CVODE integrator) 分离 + `tools/cvode_stats_diff/` 15-key CI gate（excludes nfcall）。

### S5d — data layout SoA + NUMA（#178 .. #183, master plan §S5d）

- S5d.1: `ElementHotData` SoA + 平 3 行主序 idiom + `sync_hot_dynamic(i)` AoS→SoA mirror refresh + manifest + grep gate。
- S5d.2-5a: jagged `QeleSurf` / `QeleSub` flatten + ASan/UBSan CI 全开。
- S5d.2-5b: selective small array SoA + Riv/RivSeg audit（保留 AoS 但加 audit footprint）。
- S5d.3: parallel first-touch + `[NUMA] PARALLEL_FIRST_TOUCH` log token。
- S5d.4: `tools/run_omp.sh` env discipline + manifest `omp_env` 节 + `tools/numa_check.sh`。
- S5d 汇总: sizeof scan + cache miss profile + NUMA 加速比 + ADR-0001 (SoA hot fields) + `openspec/glossary.md` term entries。

### S6b — bug fix application（#184 + #186 + #187, master plan §S6b）

- S6b.1: AccTemperature guard（积温保护）+ cryosphere NaN 边界 fence。
- S6b.2: lake formula **SKIP path**（master plan §S6b L1497 FORECAST + C8 forward-compat；**未触 SHUD/src**；详 `SHUD/B1b_CHANGELOG.md` S6b.2 row "source byte-identical to B1a"）。
- S6b.3: S2 follow-up bug audit（#159 verdict: NOT-A-BUG，无 fix action）。
- S2.17 (#185): lake formula PI 审查 — evidence pack 投出，**PI sign-off OPEN**；spec L23 + design Q1 reserve E1/E2 verdict for PI；本 B1b ship 不能视为 signed E2。

## S6c-12a B1b capstone 验证结果（2026-06-22）

**Mac 4-case 3-run SHA-identical + canonical summary SHA ≡ B0 `repeatability.txt sha256_run1`**：

| Case | summary SHA256（all 3 runs identical） | wall (s) | ≡ B0-tag |
|---|---|---|---|
| keliya | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | 26/26/27 | YES |
| xinanjiang_upstream | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | 4/5/4 | YES |
| qinyijiang | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | 245/244/239 | YES |
| qhh (lake, 5 outputs) | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | 89/89/88 | YES |

Per-file SHA cross-check vs `benchmarks/<case>/B0_output/<file>`: **13 files byte-identical**，逐文件枚举如下。

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

合计：2 (keliya) + 3 (xinanjiang) + 2 (qinyijiang) + 5 (qhh) + 1 (heihe cvode in T2) = **13 byte-identical files**（cvode_stats 算法每 case 独立一份）。详 `docs/B0_vs_B1b_water_balance_report.md` 进一步 6/6 closure-error PASS。

**服务器 2-case 3-run SHA-identical（cn03，Slurm 三铁律遵守）**：

| Case | Slurm JobId (3 runs sequential afterany) | wall (s) per run | summary SHA256 | rivqdown ≡ B0/B1a golden |
|---|---|---|---|---|
| heihe | 8662 / 8663 / 8664 (cn03) | 480 / 479 / 480 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | YES (`55abad28…`) |
| heihe_x4 | 8665 / 8666 / 8667 (cn03) | 1196 / 1192 / 1191 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | YES (`f90601ef…`) |

**水量平衡（`docs/B0_vs_B1b_water_balance_report.md`）**: bitwise identity 蕴含 closure-error delta = 0 bit-by-bit on all 6 cases（4 Mac + 2 Server），远低于 spec 0.1% 相对容差门限。

**Go/No-Go 7 项 checklist（spec b1b-capstone）**:

| # | item | verdict |
|---|---|---|
| 1 | B1b-tag 创建 + 锁定 | PASS（#189 完成于本 PROMOTE PR 之前；详下文 "验证 B1b-tag"）|
| 2 | 6-case 3-run repeatability | PASS（T1 + T2）|
| 3 | 6-case bitwise vs B0 | PASS（4 Mac canonical SHA ≡ B0 + 2 Server SHA ≡ B0/B1a golden）|
| 4 | 水量平衡不恶化 | PASS（closure_error delta = 0 bit-by-bit）|
| 5 | S6b sub-tasks 全闭环 | PASS（#184 / #186 / #187 全 closed）|
| 6 | SHUD `B1b_CHANGELOG.md` 完整 | PASS（12 sections S5a–S6b.3 全在 openmp-baseline）|
| 7 | CI 全绿（serial-baseline 4 job set） | PASS（PR-16 #207 CI 4/4 SUCCESS）|

7/7 PASS @ PROMOTE 时点。

## B1b CONDITIONAL ship status

本 B1b 整体仍为 CONDITIONAL ship，下游消费 B1b-tag 的 P1+ 工作 SHALL 显式承认以下 caveats（详 `SHUD/B1b_CHANGELOG.md` S6b.2 + `docs/s217_lake_formula_audit.md` §E）：

- **#185 (S2.17 lake formula PI 审查) — OPEN**: PR #204 evidence pack 投出后无 SHUD-upstream PI Lele Shu sign-off。spec.md L23 + design.md Open Q1 reserve E1/E2 verdict for PI；本 B1b 的 3-run + water-balance evidence **insufficient** to claim a signed E2.
- **#186 (S6b.2) — CLOSED via PR #206 SKIP**: master plan §S6b L1497 FORECAST + C8 forward-compatibility，**NOT a signed E2**。
- **#205 (SoA/AoS sync drift in `rhs_flux` lake pass-1) — OPEN**: P-strict pre-req audit；3-run bitwise 自洽证 drift deterministic、bitwise-stable，本 B1b ship 不阻，但 underlying SoA-sync gap 留给 P-strict 解决。
- **D9 fast-path trigger #2** (S2.17 审查为 'no change' 跳过 fix) — **BLOCKED on PI sign-off**。S6c proceed with separate `B1a-tag` and `B1b-tag` per design D11（D9 fast-path 未满足 → 不 merge `B1-tag`）。
- **C8 forward-compat（master plan "永不 break userspace"）**: any later PI `S2.17: formula needs fix` directive on #185 stacks as a follow-up `B1c-tag` without force-updating B1b-tag（D11 lock honoured）。

> **immutable-tag-vs-docs 已知 minor 漂移**：`B1b-tag` 注解消息（#189 push 完已 immutable per D11）列出 4 项 caveat（`#185` / `#205` / `#186` / `C8`），**D9 fast-path BLOCKED 仅记录在本文 + `docs/status_matrix.md` + `docs/build_manifest.md` 三处**。这是 D11 一次锁死契约的直接后果：annotated tag message 创建于 #189 完成时刻；D9 fast-path 的明确语义化只有在 #190 PROMOTE 时（本 PR）才在三个 doc surface 上对齐补齐。下游下游用户应以三 doc surface 为权威，tag message 作为 commit-pinned snapshot。

## 验证 B1b-tag

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
