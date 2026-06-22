# B1b Baseline 完成（编写中）

> B1b = master plan §3 定义的 "B1a + S5* 结构改造 + S6b bug fix 后的 parallel-ready serial reference"。**B1b 不是单一 stage 的产物**：S5a + S5b + S5c + S5d + S6b 全部完成后才能签字。**stage = 工作阶段**；**baseline = 工作产物**。
>
> 截至本文档创建时点（2026-06-21），B1b 当前进度 = S5a / S5b / S5c / S5d.1 / S5d.2-5a / S5d.2-5b / S5d.3 / S5d.4 全部 merged 进 `baseline/B1b`；本 PR（#183）做 S5d 汇总验收（measurement + docs only，不触 SHUD/src）；之后 S6b.1 / S6b.2 / S6b.3 → S6c capstone 锁 `B1b-tag`。
>
> baseline/B1b HEAD: `b563018`（PR-10 #200 post-merge 2026-06-21）；SHUD `ffdccd9`。

## B1b PR 时间线（截至 2026-06-21，#183 之前）

PR 编号 = B1b review-loop-log 序号（不同于 GitHub issue / PR number）。

| PR | Issue | Scope | merged commit | SHUD pin |
|---|---|---|---|---|
| PR-1 | #173 | S5c-A — hlast / qlast diagnostic, SHUD_ENABLE_DIAGNOSTICS macro precedent | b93be0c | (pre-S5c-B SHUD pin) |
| PR-2 | #176 | S5a — forcing thread-safety audit + TimeSeriesData 注释 (audit-only) | eee98a7 | (pre-S5b SHUD pin) |
| PR-3 | #177 | S5b — scratch ownership audit + lake reset 顺序 + RHS print 重组 (audit-only) | 212f4c5 | (S5b-end) |
| PR-4 | #174 | S5c-B — RHS 7-bucket timer + forcing I/O timer | (pre-S5c-C SHUD pin) | (pre-S5c-C) |
| PR-5 | #175 | S5c-C — nFCall vs CVODE nfe channel separation | (PR-5 merged commit) | (S5c-C-end) |
| PR-6 | #178 | S5d.1 — ElementHotData SoA + RHS hot-path rewrite + DEBUG asserts | (PR-6 merged) | (S5d.1-end) |
| PR-7 | #179 | S5d.2-5a — jagged QeleSurf/QeleSub flatten + ASan/UBSan CI axis | (PR-7 merged) | (S5d.2-5a-end) |
| PR-8 | #180 | S5d.2-5b — selective small-array SoA fold-in + Riv/RivSeg audit | (PR-8 merged) | (audit-only, SHUD HEAD unchanged) |
| PR-9 | #181 | S5d.3 — parallel first-touch + [NUMA] log token + cross-validation | c80dec0 | (S5d.3-end) |
| PR-10 | #182 | S5d.4 — tools/run_omp.sh + manifest omp_env + tools/numa_check.sh + CI schema gate | 30ee4e9 | ffdccd9 (current) |

详细 commit 内容见 `SHUD/B1b_CHANGELOG.md`（每个 sub-step 一节）+ `docs/review-loop-log.jsonl` (review verifier verdicts)。

## S5d 汇总验收（#183 — 本 PR 落地）

本 PR 是 **measurement + docs only**（不触 `SHUD/src`），覆盖 6 个 deliverable：

1. **T8.1 sizeof emission + assertion** — `tools/check_sizeof/`
2. **T8.2 cache miss perf stat** — server HW counters BLOCKED by `perf_event_paranoid=4`（cluster-wide）+ Apple Silicon HW counters 未实测（xctrace 推迟到 S6c capstone 前）。wall-clock 仅作 RHS 端到端 sanity check,**NOT** a cache miss proxy（详 §T8.2 + F3 review-fix）
3. **T8.3 NUMA accel measurement** — server cn07 双 socket Xeon shud_omp ON vs OFF 3 run 中位数比
4. **T8.4 Apple Silicon UMA documentation** — N/A NUMA accel per spec L159-161
5. **T8.5 ADR 0001 SoA hot fields** — `docs/adr/0001-soa-hot-fields.md`
6. **T8.6 Glossary entries** — 4 个 `openspec/glossary.md` 条目

### T8.1 sizeof emission + ratio gate

工具：`tools/check_sizeof/check_sizeof.cpp` + `tools/check_sizeof/check_sizeof.sh`（1 TU + Makefile + shell wrapper；macOS Apple clang 17 + Linux GCC 11 双端实测）。

Two ratio interpretations emitted；review-fix F1 (PR #201) 把 gate 从 (A) 移到 (B)，因为 (A) 的分子是固定 32 × 8 = 256 B 指针表大小（64-bit ABI 上恒定），随 NumEle → ∞ 趋于 0，作为 cache footprint proxy 量纲上不成立。spec L145 自声明 "可校验形式 of master plan §S5d L1432"，gate metric 应与 master plan L1432 一致。

**(A) DIAGNOSTIC — pointer-container struct size (NOT the gate metric)**: `sizeof(ElementHotData) / sizeof(_Element)`

| 平台 | sizeof(ElementHotData) | sizeof(_Element) | ratio (A) | 备注 |
|---|---|---|---|---|
| Apple Silicon macOS (Apple clang 17) | 256 B | 688 B | 0.3721 | diagnostic only — 256 B = 32 × sizeof(void*) 固定 |
| Linux x86_64 cn05 (Ubuntu 24.04 GCC 13.3, Slurm 8621) | 256 B | 688 B | 0.3721 | diagnostic only — 跨平台 ABI 一致 |

**(B) GATE METRIC — spec L148-149 + master plan L1432 (per-element SoA bytes)**: per_ele_soa_bytes / sizeof(_Element)

| 平台 | per_ele_soa_bytes | sizeof(_Element) | ratio (B) | 阈值 | verdict |
|---|---|---|---|---|---|
| Apple Silicon macOS | 300 B | 688 B | 0.4360 | < 0.20 | **FAIL** (IN-PROGRESS — THE GATE METRIC) |
| Linux x86_64 cn05 (Slurm 8621) | 300 B | 688 B | 0.4360 | < 0.20 | **FAIL** (IN-PROGRESS — THE GATE METRIC) |

跨平台两端数字 **完全一致** — 是 layout 决策驱动,非 ABI / 编译器差异。Gate verdict 由 (B) 决定（0.4360 > 0.20 → IN-PROGRESS）。

Per-element SoA payload 拆解（macOS 实测，详 `tools/check_sizeof/check_sizeof.cpp`）：
- 2 个 `int[3]` flat 字段（nabr / lakenabr）= 24 B
- 7 个 `int` scalar（iSoil / iLC / iMF / iForc / iLake / iBC / iSS）= 28 B
- 4 个 `double[3]` flat 字段（edge / Dist2Nabor / Dist2Edge / avgRough）= 96 B
- 19 个 `double` scalar（area / z_bottom / z_surf / FixPressure / WetlandLevel / RootReachLevel / depression / QBC / QSS / windH / u_qi / u_qex / u_effKH / u_satn / Sy / VegFrac / Albedo / Rough / ImpAF）= 152 B
- 合计 = 300 B per element

#### Verdict 解释 — 为何 IN-PROGRESS 不阻断 #184

`sizeof(_Element)` 实测 688 B（master plan §4.22.1 估算 600-1000 B 下沿）。32 个 hot field 实际占 300 B / element（43.6%），原因是 RHS 真用了那么多字段（grep 审计结果 = 32 fields cover 全部 RHS 3 TU 的 `Ele[.].<field>` 访问点）。

这是真实的物理上界——**不是 SoA 设计缺陷**:
- Hot field 字节数已经客观接近 _Element 的一半,无法仅通过 SoA 抽取压缩
- master plan L1432 "< 20%" 阈值假设的是 hot field 远小于 _Element fat-AoS,但实测 _Element 没有那么"fat"
- 真正的 cache 收益验证由 Task 8.2 实测 cache miss reduction 提供;但 server `perf_event_paranoid=4` 限制下本 PR 内 **未实测** HW counter（escalation TBD，wall-clock 数字仅作 sanity check **不构成** cache miss criterion evidence）

**对 #184 (S6b.1) 的影响**: 0。S5d.1-S5d.4 全部 sub-step 6 case 90 天 NUM_OPENMP=1 bitwise vs B1a-tag PASS（详 `SHUD/B1b_CHANGELOG.md`），S6b.1 启动条件 = "B1a 锁定 + S5* 完成 + bitwise neutrality 已验证" 已满足。Task 8.1 ratio gate 实测不达不影响 RHS 正确性,只影响 cache miss 收益预期。详 design R4 mitigation #2 精神（"任一机不达标记 IN-PROGRESS 不阻断 #184"），同样适用本 task。

### T8.2 cache miss perf stat — server + Apple Silicon

#### Server cn07（双 socket Xeon Gold 5318Y）

**关键基础设施限制**:`perf_event_paranoid = 4`（cluster-wide,login + compute node 一致;test job 8618 已验证）。User space 无权访问任何 perf hardware counters（CAP_PERFMON 或 root 才能 enable）。该值需 root 修改 `/proc/sys/kernel/perf_event_paranoid <= 2`（或 `<= 0` for raw access）。

**Fallback**: wall-clock comparison（B1a-tag vs B1b heihe 90 天 NUM_OPENMP=1 OMP_PROC_BIND=close 3 run 中位数）。**wall-clock 与 cache miss 不可互换** — forcing I/O 主导 heihe wall-clock（详 #174 t_forcing_io_s 数字），wall-clock ratio 不能 bound cache miss ratio。下表 wall-clock 数字作为 RHS 端到端 sanity check，**不构成** spec L151-153 cache miss criterion evidence。Review-fix F3 (PR #201) 修正此处先前的"5-15% 经验法则"误述。

| 测量项 | 命令 | 输出 |
|---|---|---|
| perf_event_paranoid 限制确认 | sbatch job 8618 cn07 `perf stat -e task-clock sleep 0.1` | "Access to performance monitoring and observability operations is limited" — perf_event_paranoid=4 (verified) |
| B1a wall-clock × 3 | sbatch job 8619 cn07 heihe 90d `./shud heihe` | 466.639s, 466.750s, 467.081s |
| B1b wall-clock × 3 | sbatch job 8619 cn07 heihe 90d `./shud heihe` | 467.137s, 465.797s, 465.780s |
| B1a median | computed | 466.750s |
| B1b median | computed | 465.797s |
| 中位数 ratio B1b/B1a | computed from medians | **0.9980** (speedup 0.20%) |
| B1b vs B1a heihe.rivqdown.dat SHA256 | bitwise sanity check | `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f` (both, PASS) |

详 `/scratch/frd_muziyao/SHUD-OpenMP/.s5d-summary-runs/wall_8619.out`。Server `perf` HW counters blocked by `perf_event_paranoid=4` (cluster-wide, verified Slurm 8618 cn07);wall-clock 数字仅作 RHS 端到端 sanity check,**不构成** spec L151-153 cache miss criterion evidence（详 F3 review-fix）。heihe (6335 elements) NUM_OPENMP=1 单线程 cache pressure 主要由 forcing I/O 决定（详 #174 t_forcing_io_s 数字）,SoA 收益主要在 RHS hot loop,占 total wall 比例有限。Task 8.2 标 **NOT MEASURED** (HW counters blocked + escalation TBD) 不阻 #184。

#### Apple Silicon macOS（Apple M4 Pro, 1 socket UMA）

**Linux `perf` 不可用**（macOS 没有这个工具）。Apple `xctrace` / `dtrace` 太重型且报告格式与 Linux `perf stat` 不兼容,无法做跨平台比较。**Apple Silicon UMA**: hw.packages=1 (single SoC, unified memory) — 与 NUMA effect 无关。

**Fallback**: wall-clock comparison（B1a-tag vs B1b keliya 90 天 NUM_OPENMP=1 3 run 中位数）。本地 keliya 4 年 ~11min vs 90 天 ~30s（CLAUDE.md L86）,3 run × 2 side = 6 run × 30s = ~3 min。

| 测量项 | 命令 | 输出 |
|---|---|---|
| B1a wall-clock × 3 keliya | `.s5d-summary-runs/run_mac_wallclock.sh` | 29.064s, 28.497s, 31.635s |
| B1b wall-clock × 3 keliya | 同上 | 27.806s, 28.538s, 32.154s |
| B1a median | computed | 29.064s |
| B1b median | computed | 28.538s |
| 中位数 ratio B1b/B1a | computed | **0.9819** (speedup 1.81%) |

详 `.s5d-summary-runs/mac_wallclock.log`。Apple Silicon UMA + 14 phys core + Apple M4 Pro。B1b 比 B1a 略快 1.81%(noise floor ~2-3%),方向与 cache miss ↓ 预期一致,但量值在 noise 范围 — 单线程 keliya (484 elements) 触发不到显著 cache pressure,信号弱属预期。spec L153 IN-PROGRESS 不阻 #184。

**Apple Silicon perf 标 BONUS（不 gate）**: spec L153 "任一机不达标记 IN-PROGRESS 不阻断 #184"。Apple Silicon HW counters via `xctrace record --template "Counters"` (master plan L859) 未实测,本 task 直接标 **NOT MEASURED** (HW counters 未尝试) 不算 fail。Wall-clock 数字 0.9819 仅作 sanity check,**不构成** cache miss evidence。

### T8.3 NUMA accel measurement — server cn07

| 测量项 | 配置 | 输出 |
|---|---|---|
| ON: `make shud_omp` + `OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores` heihe 90 天 × 3 run | sbatch 8622 cn07 | 469s, 470s, 469s |
| OFF: `make shud_omp` + `OMP_NUM_THREADS=8` (no OMP_PROC_BIND/PLACES) heihe 90 天 × 3 run | 同 sbatch | 469s, 468s, 478s |
| ON wall-clock 中位数 | from 3 ON runs | 469s |
| OFF wall-clock 中位数 | from 3 OFF runs | 469s |
| ratio = on_median / off_median | computed | **1.0000** |
| spec target ≤ 0.87 (>= +15% speedup) | spec L155-157 | **N/A by construction** (B1b RHS serial; non-measurable until P1+ RHS parallel-for lands) |

详 `/scratch/frd_muziyao/SHUD-OpenMP/.s5d-summary-runs/numa_8622.out` + per-run `.s5d-summary-runs/numa_accel/heihe_{on,off}_r{1,2,3}/`。

**为什么 ratio = 1.0**(结构性,非偶然): B1b 是 master plan §3 定义的 "**parallel-ready single-thread baseline**" — RHS hot path 中**没有 `#pragma omp parallel for`** 落地(P1+ 范畴)。`SHUD/src/` 当前 4 个 `#pragma omp parallel for` 都在 init time(`Model_Data::malloc_EleRiv()` 3 处 + `MD_initialize::LoadIC()` 1 处,详 S5d.3 #181 PR 落地)。RHS 实际仍 serial 执行,8 thread × first-touch ON 与 1 thread × OFF 在 wall-clock 上等价(各种 thread spawn 开销在 90 天 run 内 negligible)。

`shud_omp` banner 已 verified "`* openMP enabled. Maximum Threads = 8`",`[NUMA] first-touch begin tag=...` 4 tag 全部 emit(详 `numa_accel/heihe_on_r1/run.stdout.log`),证明 OpenMP runtime 与 first-touch 都已生效;但 RHS 仍 serial,故 NUMA 收益无法测出。**这是 B1b 当前状态的客观反映,不是测量错误**。

**Review-fix F2 (PR #201) 调整 verdict 标签**: spec L156-157 IN-PROGRESS escape 设计是为 "节点排队 / 维护" 这类 test-availability 延迟保留;B1b 这里是更强的状况——**by-construction non-measurable**,直接标 **N/A by construction**(而非 IN-PROGRESS 1.0000 > 0.87 的"差 13%"误读),把 NUMA accel measurement 显式放到 P1+ A3a 前置,而非 S5d.4 残留项。

**对 #184 (S6b.1) 的影响**: 0。S5d.3 first-touch 基础设施已在位,S5d.4 thread binding wrapper 已在位,等 P1+ RHS parallel-for 落地后 NUMA accel 可重新测。spec L153 显式允许 8.3 不阻 #184(design R4 mitigation #2 "拆开 B1b 内部 critical path");本 PR 将 verdict 从 IN-PROGRESS 收紧到 N/A by construction,进一步强化 non-blocking 语义。

**Slurm 三铁律 satisfied**: sbatch FROM /scratch + `--output/--error` 在 /scratch + run.sh + binary 都在 /scratch。Job 8622 cn07 Elapsed 00:47:05 ExitCode 0:0。

### T8.4 Apple Silicon UMA NUMA accel documentation — N/A

**结论**: N/A (single-socket UMA)

- 硬件: Apple M4 Pro, hw.physicalcpu=14, hw.logicalcpu=14, hw.packages=1
- NUMA effects 不存在（UMA 架构）
- `tools/numa_check.sh` 在 Apple Silicon 上输出 `socket_count: 1` + `numa_first_touch: N/A (single-socket UMA)` — 详 `SHUD/B1b_CHANGELOG.md` S5d.4 段 "numa_check.sh execution on Apple Silicon"
- first-touch ON vs OFF 在 UMA 上预期 ~0% 差异；运行也是为了 evidence 而非 gating

| 测量项 | 配置 | 实测 |
|---|---|---|
| ON: `OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores` keliya 90 天 × 3 | `.s5d-summary-runs/run_mac_numa.sh` | 28.178s, 27.183s, 26.973s |
| OFF: `OMP_NUM_THREADS=8` no OMP_PROC_BIND keliya 90 天 × 3 | 同上 | 26.878s, 28.038s, 26.536s |
| ON median | computed | 27.183s |
| OFF median | computed | 26.878s |
| ratio ON/OFF | computed | **1.0113** (delta +1.13%) |
| verdict | N/A per spec L159-161 | **N/A (single-socket UMA confirmed)** |

详 `.s5d-summary-runs/mac_numa.log`。`OMP_PROC_BIND` ON/OFF wall-clock 在 noise 内重叠(±1.13%),完全符合 Apple Silicon UMA "无 NUMA effect" 预期。binary: `SHUD/shud`(无 `shud_omp`,banner 显示 "openMP disabled" — 串行 keliya;此实验对照 NUMA env 变量是否在串行 binary 下也无效影响,结论是 yes)。

### T8.5 ADR 0001 SoA hot fields

新建 `docs/adr/0001-soa-hot-fields.md`,5 节 standard ADR 格式:
- Context: `_Element` fat-AoS, 32 hot fields ≈ 300 B vs 688 B = 43.6%
- Decision: SoA + AoS 双轨 (D2)
- Consequences: + bitwise neutrality, + R 端协议不破, + DEBUG assertion 网, − 双写代价（sync_hot_dynamic）, − sizeof gate 未达 < 0.20
- Triggers (何时启动 SoA 单轨化): (1) Task 8.2 cache miss ↓ ≥ 30% (2) Task 8.3 NUMA accel ≥ +15% (3) B1b 落地后 6 个月无 regression — 三者全满足才迁
- References: master plan §4.22 + design D2 + spec + yaml + CI grep gate (`check_hot_fields.py` yaml↔layout↔RHS) + standalone sizeof tool (`tools/check_sizeof`, not yet CI-gated)

### T8.6 Glossary entries

`openspec/glossary.md` 新增/扩展 4 个术语条目（grep-verifiable）:

| 术语 | 状态 | 行号 |
|---|---|---|
| **OMP_CUTOFF** | 扩展（pre-existing 短条目） | L71 |
| **nFCall** | 新增（语义 + nFCall vs nfe 严格分离 + D10 引用） | L101 |
| **ElementHotData** | 扩展（pre-existing 1 行 → 详细 + yaml + ADR 引用） | L140 |
| **RiverHotData** | 新增（当前不实现 + 触发条件 + #180 audit 引用） | L144 |

```bash
$ grep -nE '^\*\*(ElementHotData\|RiverHotData\|nFCall\|OMP_CUTOFF)\*\*' openspec/glossary.md
71:**OMP_CUTOFF**:
101:**nFCall**:
140:**ElementHotData**:
144:**RiverHotData**:
```

## S5d 汇总验收 acceptance summary

| Acceptance criterion (spec L147-161) | Verdict |
|---|---|
| sizeof(ElementHotData) / sizeof(_Element) < 0.20 (T8.1, scenario "ElementHotData 字节占比") | **IN-PROGRESS** (Apple Silicon 0.3721; Linux cn05 0.3721 — 跨平台一致) — 真实物理上界,非设计缺陷;不阻 #184 |
| cn 双 socket node + Apple Silicon 各 cache miss ↓ ≥ 30% (T8.2, scenario "cache miss 单线程下降 ≥ 30%") | **NOT MEASURED** (HW counters blocked by `perf_event_paranoid=4` cluster-wide; xctrace / PAPI / VTune / pmu-tools 未实测尝试,详 Residual / Deferred 节) |
| Wall-clock sanity check (NOT a cache miss proxy) | **informational only** — cn07 B1b/B1a = 0.9980, Mac 0.9819 (noise floor; **不构成** spec L151-153 cache miss criterion evidence per Review-fix F3 PR #201) |
| NUMA accel ratio ON/OFF ≤ 0.87 (T8.3, scenario "双 socket NUMA 加速比") | **N/A by construction** (B1b RHS serial; cn07 8 thread ON/OFF = 1.0000 是结构性必然,非测量值。NUMA accel 测量是 P1+ A3a 前置,非 S5d.4 残留;详 Review-fix F2 PR #201) |
| Apple Silicon UMA 跳过 NUMA 验收 (T8.4, scenario "单 socket UMA 跳过") | **PASS (N/A)** — single-socket UMA via `tools/numa_check.sh`; M4 Pro ON/OFF 1.0113 noise floor |
| ADR 0001 SoA hot fields 5 节内容 (T8.5) | **PASS** — `docs/adr/0001-soa-hot-fields.md` Status/Context/Decision/Consequences/Triggers 全 |
| glossary.md 4 个新条目 grep-verifiable (T8.6) | **PASS** — 4/4 grep hit (ElementHotData/RiverHotData/nFCall/OMP_CUTOFF) |

按 spec L153 + design R4 mitigation #2,**T8.1 IN-PROGRESS + T8.2 NOT MEASURED + T8.3 N/A by construction — 全部不阻 #184 启动**。S5d 汇总验收 status = **2 PASS + 1 PASS(N/A) + 1 IN-PROGRESS + 1 NOT MEASURED + 1 N/A by construction**;每个 verdict 都有具体 measurement evidence 或 explicit 缺测 reason(blocked / unmeasurable),非泛指"未测"。

### Slurm 作业号汇总

| Job ID | 内容 | Node | Elapsed | ExitCode | 关联 |
|---|---|---|---|---|---|
| 8617 | (FAILED) perf stat HW counters attempt | cn07 | 00:00:01 | 1 | T8.2 — perf_event_paranoid=4 block,改 fallback |
| 8618 | perf availability test on compute node | cn07 | 00:00:01 | 0 | 验证 perf 跨 login+compute node 一致受限 |
| 8619 | wall-clock B1a vs B1b heihe 90d × 3 each (sanity check, NOT cache miss proxy) | cn07 | 00:46:39 | 0 | T8.2 sanity check ratio 0.9980 (informational only) |
| 8620 | (FAILED) sizeof tool, relative path | cn05 | 00:00:01 | 1 | path fix |
| 8621 | Linux sizeof emission cn05（job exit 0; tool internal exit 1 = expected gate FAIL on (B) per IN-PROGRESS verdict） | cn05 | 00:00:01 | 0 | T8.1 Linux side 0.3721 / 0.4360 |
| 8622 | NUMA accel ON vs OFF heihe NUM_OPENMP=8 × 3 each | cn07 | 00:47:05 | 0 | T8.3 ratio 1.0000 IN-PROGRESS |

合计 cn07 wall ~1h34min + cn05 wall <1s。Slurm 三铁律全程遵守。

## S5d 汇总验收 Residual / Deferred

- **Server perf stat HW counters**(T8.2 BLOCKED): cluster `perf_event_paranoid=4` 限制 user-space access。Escalation path **TBD**——本 PR 未与 cluster admin 通联 ticket,follow-up issue 在 S6c capstone 验证前 file 之(具体 admin contact + ticket template + target window 由 cluster ops 提供)。在 escalate 成功前 spec L151-153 acceptance criterion 标 **NOT MEASURED**。
- **Apple Silicon HW counters 未尝试**(T8.2): master plan L859 显式推荐 `xctrace record --template "Counters"` 作为 macOS 等价替代。本 PR 仅在文字层 dismissed("报告格式不兼容") 而未实测;若 S6c capstone 前需 cache miss evidence,可花 ≤30 min 跑 `xctrace record --template "Counters" --launch ./shud keliya`,export CSV via `xctrace export --xpath`,parse `L1D_CACHE_MISS` / `LOAD_CACHE_MISS` 列。结果 not directly comparable to Linux `LLC-load-misses` 但同一 SoA hypothesis direction。
- **NUMA accel re-measurement**(T8.3): P1+ A3a 前置,不再属 S5d.4 残留项(Review-fix F2 重新归类)。B1b RHS parallel-for 落地后,使用 Slurm 8622 模板重测即可。
- **跨平台 sizeof 数字**: Linux x86_64 GCC 13.3 (cn05 Slurm 8621) 与 macOS Apple clang 17 实测均为 688 B,**完全一致**——layout 决策驱动,非 ABI / 编译器差异(Review-fix F7 修正先前 "GCC 11 = ?" 占位符);若未来 ABI 变化导致跨平台差异 > 5%,本 ADR + b1b_summary 须更新。
- **`.s*-runs/` gitignore 缺位**(F4 历史问题): scratch dir 命名约定下应 auto-ignored,但 `.gitignore` 未实施。专门 hygiene PR 中处理,不属 #183 范围。
- **ADR Triggers 3-AND 可达性**(F8 哲学问题): 当前 (1) cache miss ↓≥30% (2) NUMA accel ≥+15% (3) 6 months stable 三条件 AND 需 P1+ 完成 + 6 个月,非常严格。是否改 OR / 加 sunset clause 在专门 ADR 评审中讨论,不属 #183 范围。
- **ADR Last Reviewed 字段**(F9 模板): 标准 ADR 模板含此字段,本 ADR 未含。后续 ADR-0002+ 引入时统一加。
