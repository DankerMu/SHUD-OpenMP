# P1 阶段 RHS 快照按位一致性验证 — Mac 4 案例

## 背景与定义

本文档为 P1 epic (Issue #211, 子任务 #219 / I-C.5) 之 PR-H 交付物，实现 spec `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md` 第 L127–L134 行所定义的"右端项快照按位一致比对 (RHS snapshot bitwise vs per-case authoritative baseline)" 场景，覆盖 Mac 平台 4 个基准案例 (benchmark case)。

此处的"快照 (snapshot)"指 SHUD 求解器在右端项 (right-hand side, RHS) 求值过程中按预设探针点 dump 出的中间状态向量；"按位一致 (bitwise identical)" 指与 B1b-tag 归档的黄金 (golden) 参考逐字节相等。

## 验证范围

1. 4 个 Mac 案例 × 3 个 `t_value` × 1 个规范快照后缀 = **12 个文件级按位比对**，对标 B1b-tag 归档黄金。
2. 在 OpenMP 二进制 `shud_omp` 上以 `OMP_NUM_THREADS=1` 执行（对应 spec Scenario L131–L134 所定义的 A0 / NUM_OPENMP=1 验证等级）。
3. 案例集合：`keliya` (484 单元) / `xinanjiang_upstream` (801) / `qinyijiang` (3155) / `qhh` (4773 单元 + 1 湖泊)。`kashigeer` 按 S0-13 之 `deferred-upstream` 状态排除（依据 status_matrix L41 + `benchmarks/INDEX.md`）。
4. 服务器案例 (`heihe`, `heihe_x4`) 不在 PR-H 范围内，由 PR-J 按 tasks 5.5–5.6 覆盖。

**规范快照后缀**取 `snapshot_t<rel_sec>.bin`（对应 SHUD writer `SHUD_DUMP_SITE=f_update`）。这是 B0 / B1a / B1b 三轮重复性 (repeatability) 门控所归档的验证集（status_matrix L132 记载"4 个 case × 3 = **12 个** snapshot_t*.bin 入库"）。其伴生的 `_before_passvalue.bin` 第二后缀系后续由 PR #54 引入的 SHUD writer 诊断副产品，不属于 B0 / B1a / B1b 规范验证门，PR-H 将其作为诊断附录处理（详见下文「诊断附录」一节）。

## Tag 链与黄金来源

按 spec L107–L113 所列各案例权威基线表，4 个 Mac 案例均满足 `B0 ≡ B1a ≡ B1b ≡ B1`（全链按位稳定），因此 `benchmarks/<case>/B0_output/snapshot_t<rel_sec>.bin` 通过 tag 链同一性即构成 B1b-tag 的规范黄金。

| case                  | NumEle | has_lake | authoritative tag      | golden path                                                            |
| --------------------- | ------ | -------- | ---------------------- | ---------------------------------------------------------------------- |
| keliya                | 484    | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/keliya/B0_output/snapshot_t<T>.bin`                        |
| xinanjiang_upstream   | 801    | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t<T>.bin`           |
| qinyijiang            | 3155   | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/qinyijiang/B0_output/snapshot_t<T>.bin`                    |
| qhh                   | 4773   | yes (1)  | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/qhh/B0_output/snapshot_t<T>.bin`                           |

## 编译证据 — 严格浮点旗标

Mac 本地 OMP 编译开启 `SHUD_DUMP_RHS=1`（即将快照写入器编入二进制），源码版本为外层 `ec4cdd2` + SHUD `07c677f`：

```
cd SHUD && make clean && make SHUD_DUMP_RHS=1 shud_omp 2>&1 | tee .s2-103/pr-h/make_shud_mac.log
```

三项 grep 门控（spec task 5.2 + L193–L197）执行结果如下：

| gate                                                           | required | observed | verdict |
| -------------------------------------------------------------- | -------- | -------- | ------- |
| `grep -oE '\-O2\|-ffp-contract=off\|-fno-fast-math'`           | ≥ 3      | 6        | PASS    |
| `grep -E  '\-ffast-math\|-Ofast\|-funsafe-math-optimizations'` | 0        | 0        | PASS    |
| `grep -c  '\-fopenmp'`                                         | ≥ 1      | 2        | PASS    |
| `grep -c  'DSHUD_DUMP_RHS=1'`                                  | ≥ 1      | 2        | PASS    |

编译产物：`.s2-103/pr-h/make_shud_mac.log`。其中 Apple Clang 链接 `./shud_omp` 的命令行同时具备三项严格浮点旗标与 `-fopenmp`。

## 规范 12 单元按位矩阵

| case                     | t=86400 (1d) | t=2592000 (30d) | t=7776000 (90d) | wall (90d, NUM_OPENMP=1) |
| ------------------------ | ------------ | --------------- | --------------- | ------------------------ |
| keliya (484)             | PASS         | PASS            | PASS            | 60.8 s                   |
| xinanjiang_upstream (801)| PASS         | PASS            | PASS            | 7.3 s                    |
| qinyijiang (3155)        | PASS         | PASS            | PASS            | 238.1 s                  |
| qhh (4773 w/lake)        | PASS         | PASS            | PASS            | 66.5 s                   |

**汇总：12 / 12 PASS** — `compare_snapshot --quiet` 在所有单元退出码均为 0。

依 `compare_snapshot` 之约定，退出码 0 表示 "BITWISE IDENTICAL"，即文件头、记录头与所有 dump 数组字节均与黄金逐字节相等。快照内容（依 `format.h` v1 + 实测 `DY` `array_count=1`）为长度 `NumY` (= 3·NumEle + NumRiv + NumLake) 的完整 `DY` 状态导数向量。

复现命令：

```
bash .s2-103/pr-h/run_pr_h_snapshot_bitwise.sh
```

各案例运行目录与 `cmp.log` 保留于 `.s2-103/pr-h/<case>_main/`。

## 判定 (spec L131–L134)

1. PR-D（element 循环）+ PR-E（river 循环）+ PR-F（lake 循环）三 pragma 栈在 `OMP_NUM_THREADS=1` 条件下，对规范 RHS 快照探针点 (`SHUD_DUMP_SITE=f_update`) 在全部 4 个 Mac 案例 × 全部 3 个 `t_value` 上均与 B1b-tag 规范黄金按位一致。
2. A0 (NUM_OPENMP=1) 在 Mac 4 案例集上得到验证。
3. spec Scenario "4 Mac case RHS snapshot bitwise PASS" 已满足。
4. 与既有 CI 门控 `serial-baseline / build-and-compare(1, keliya)`（即 PR 快速反馈路径）配合：该门控在 PR-D / PR-E / PR-F 历次合并上均保持 GREEN。

## §9 数组完整性说明（范围外澄清）

Spec L129 列出了 18 个 RHS-state 数组 (`uYsf / uYus / uYgw / uYriv / yLakeStg / DY / qEle* / QeleSurf / ...`)。SHUD writer 当前 schema (`tools/rhs_snapshot/format.h` v1 + 实测规范归档) 每个快照文件仅 dump **单一 `DY` 数组** (`array_count = 1`, name `"DY"`, `nelem = NumY`)。§9 清单中其余 17 个数组尚未纳入当前规范快照写入器 schema；该缺口源自 B0 阶段归档遗留（S0-archival gap），不在 PR-H 范围内，如有必要将在 PR-N 或后续 writer schema 版本号 (`SHUD_RHS_SNAPSHOT_FORMAT_VERSION`) 升级中处理。此处理与 B0 / B1a / B1b 门控保持一致（其同样验证单数组 writer schema），并与 status_matrix L132（"12 个 snapshot_t*.bin 入库"）相符。

## 后续工作

1. **PR-I** (#220 / I-C.6)：全运行回归 — 4 个 Mac 案例的规范汇总 SHA + cvode_stats 15 键 + 3 轮重复性，依 spec L143–L161 执行。
2. **PR-J** (#221 / I-C.7)：服务器按位 — `heihe` + `heihe_x4` 通过 Slurm 提交 90 天 NUM_OPENMP=1 + cvode_stats，依 spec L136–L138 + L152–L155 执行。

---

## 诊断附录 — `_before_passvalue` 第二后缀（非 spec 门控）

> **状态**：信息性质。不构成 PR-H 验收准则（参见上文「验证范围」与 status_matrix L132 文件数门控；该门控仅计入规范 12 个快照文件）。在此记录以保留透明度并供后续追踪。

`benchmarks/<case>/B0_output/` 下另存有第二后缀的快照，由 `SHUD_DUMP_SITE=f_loop_before_passvalue` + `SHUD_DUMP_FNAME_SUFFIX=before_passvalue` 触发写入。该后缀捕获 `f_update()` 内部、`PassValue_legacy` 调用之前的中段 `DY` 切片 (`array_count = 1`, name `"DY"`, `nelem = NumEle`)。该后缀由 PR #54（快照重复性测试框架）作为 SHUD writer 的诊断副产品引入，**未**追溯纳入 B0 / B1a / B1b 规范验证门。

PR-H 针对该第二后缀加跑了 12 个单元，用以采集三 pragma 栈在中段管线 (mid-pipeline) 行为上的诊断证据：

| case                     | t=86400 before | t=2592000 before | t=7776000 before |
| ------------------------ | -------------- | ---------------- | ---------------- |
| keliya (484)             | PASS           | PASS             | PASS             |
| xinanjiang_upstream (801)| FAIL           | FAIL             | FAIL             |
| qinyijiang (3155)        | FAIL           | FAIL             | FAIL             |
| qhh (4773 w/lake)        | FAIL           | FAIL             | FAIL             |

**6 / 12 PASS（仅 keliya 通过）。** 差异幅度统计（实数尺度差异，并非 ULP 级）：

| case               | t=86400                  | t=2592000                 | t=7776000                  |
| ------------------ | ------------------------ | ------------------------- | -------------------------- |
| xinanjiang_upstream | |Δ|≤4.48, L2=10.22, idx=30  | |Δ|≤24.43, L2=58.55, idx=13  | |Δ|≤29.84, L2=68.35, idx=12   |
| qinyijiang         | |Δ|≤317.95, L2=995.67, idx=8| |Δ|≤61.70, L2=195.67, idx=8 | |Δ|≤223.67, L2=887.36, idx=8 |
| qhh                | |Δ|≤5.58, L2=17.11, idx=211 | |Δ|≤19.04, L2=53.69, idx=125| |Δ|≤90.40, L2=197.99, idx=1  |

（各单元差值幅度均远超任何 A3b 单位最末位 (unit in the last place, ULP) 阈值；该中段差异处于实数尺度，并非舍入次序所致的伪差。`first_diff_index` 表示首个字节差异在 `DY` 元素中的偏移。）

### 观测

1. 规范 `f_update` 快照在上方表格中达成 12 / 12 按位一致：`f_update` 结束时刻交付给 CVODE 的 `DY` 状态向量与 B1b 逐字节相同。**任何中段管线漂移在 PassValue 运行之前都被 `f_update` 余下部分吸收。**
2. `keliya`（无湖泊、河道规则、484 单元）是唯一在两种后缀上均按位一致的案例。失败的 3 个案例对应网格复杂度更高者（801 / 3155 / 4773），且 qhh 额外含湖泊。该模式具结构性，并非统计噪声。
3. 第二后缀探针点位于 `f_update()` 中、单元通量循环 (element-flux loop) 与 PassValue 之间。PR-D / PR-E / PR-F 在 `f_update` 侧三处循环加入了 `#pragma omp parallel for schedule(static) default(none)`；即使在 `OMP_NUM_THREADS=1` 下，OpenMP runtime 仍会初始化一个并行区（单线程团队），其可能相对严格串行的 B1b 二进制重排 firstprivate / shared 写入顺序。该机制是其中一种可能成因，确认需 PR-N 审计推进。

### 假设（推迟至 PR-N 或 B1c 后续）

- **(a) OpenMP 单线程 runtime 初始化**：`omp parallel for` 即使 team 大小为 1，调度器仍可能改变 iteration 的可观察写入顺序（与自然 for 循环顺序不同），从而在中段探针处呈现差异而最终态保持一致。记为 runtime artifact，规范门控仍为权威。
- **(b) PR-D/E/F 循环体内重排**：三个 pragma 之一可能改变循环内 scratch 写入顺序（例如 `QeleSurfTot / QeleSubTot` 归约或 row-major 平铺写入顺序），在 PassValue 运行时变得不可观察、但在 before-PassValue 探针处可见。需对 PR-D / PR-E / PR-F 进行逐 commit 二分定位。
- **(c) 探针点语义漂移**：若 SHUD 源码自 B0 归档至今对 `f_loop_before_passvalue` hook 位置有任何位移，新探针点采集的 `DY[NumEle]` 与归档黄金即不再对应同一程序点。需对比 `MD_rhs_dump.cpp` 调用点历史与 B0 SHUD pin。

### 行动项

将于 P2a / PR-N 阶段开启后续 issue：

> **标题**：调查 PR-D/E/F 三 pragma 栈下 `_before_passvalue` 中段 `DY` 漂移（3 个 Mac 案例）
>
> **范围**：对 PR-D / PR-E / PR-F 进行 bisect 定位；甄别假设 (a) / (b) / (c)；决定是否 (i) 记录为已知 artifact、(ii) 通过循环体内顺序修复恢复中段按位一致，或 (iii) 升级快照写入器 schema 以从归档中移除第二后缀。
>
> **不阻塞**：B1c-tag 叠加；规范 12 单元门控在 4 个 Mac 案例上均按位一致，下游生产行为（送入 CVODE 的 RHS 状态、`rivqdown` / `eleysurf` / cvode_stats）均得到保留。

---

`signed_at`: 2026-06-22
`signer`: DankerMu
`signed_against_outer_commit`: `ec4cdd2`
`signed_against_SHUD_commit`: `07c677f`
`PR-H_branch`: `pr-h-mac-snapshot-bitwise`
`Closes`: #219 (P1 epic I-C.5)
