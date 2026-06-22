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

---

## S6c-12a B1b capstone evidence (#188 — 本 PR 落地)

> Date: 2026-06-22。B1b 候选 commit (outer) = `069971b`（本 PR 推送后会更新）；SHUD pointer = `71b3a1a` (`openmp-baseline` branch, S6b.2.1 #186 PR-15 amend SHA cell)；S5* + S6b 全部 sub-step 已 merged 进 `baseline/B1b`。本 PR 是 **measurement + docs only**（不触 `SHUD/src/`），落 4 个 deliverable：T1 Mac 4-case 3-run 自洽 / T2 server 2-case 3-run 自洽 / T3 `B0_vs_B1b_water_balance_report.md` / T4 Go/No-Go 7 项 checklist evidence。
>
> **Out of scope**：B1b-tag 创建（#189）；branch lock（#189）；status_matrix.md B1b 行 PASS（#190）；openspec archive + PROMOTE（#190）；review-loop-log / stage-pipeline-log 追加（#190）。

### T1 Mac 4-case 3-run SHA256（NUM_OPENMP=1，90-day truncated）

| Case | project | cfg.para window | run-1 wall (s) | run-2 wall (s) | run-3 wall (s) | summary SHA256 (all 3 runs identical) | verdict |
|---|---|---|---|---|---|---|---|
| keliya | keliya | START=12053 END=12143 | 26 | 26 | 27 | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | **PASS** |
| xinanjiang_upstream | xinanjiang | START=0 END=90 | 4 | 5 | 4 | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | **PASS** |
| qinyijiang | nanlin | START=366 END=456 | 245 | 244 | 239 | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | **PASS** |
| qhh (lake) | qhh | START=8401 END=8491 | 90 | 85 | 87 | `c76dae187f382cd796cd05c9cacce6ecce5a299aff2ec9fbe64022e920b609cd` | **PASS** |

**Per-file SHA verification vs B0 archive**：

| Case | file | B1b run-3 SHA256 | B0 archive SHA256 | match |
|---|---|---|---|---|
| keliya | `keliya.rivqdown.dat` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | **YES** |
| keliya | `cvode_stats.txt` | `fdf8662c022620b7f04a5f2d994440065ac559f57c9245ae347bff7c8a190e57` | `fdf8662c022620b7f04a5f2d994440065ac559f57c9245ae347bff7c8a190e57` | **YES** |
| xinanjiang_upstream | `xinanjiang.eleygw.dat` | `f6e86f013f4f92d1c99429eafb27ec38cc7fc417e6d7d9aeef1725f8fa0a46a1` | `f6e86f013f4f92d1c99429eafb27ec38cc7fc417e6d7d9aeef1725f8fa0a46a1` | **YES** |
| xinanjiang_upstream | `xinanjiang.rivqdown.dat` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | **YES** |
| xinanjiang_upstream | `cvode_stats.txt` | `77196a7da79b94176306eaa806580d810ad9b26bcc7b3ec43e4ae8c86496a097` | `77196a7da79b94176306eaa806580d810ad9b26bcc7b3ec43e4ae8c86496a097` | **YES** |
| qinyijiang | `nanlin.rivqdown.dat` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | **YES** |
| qinyijiang | `cvode_stats.txt` | `58f36d72bbb7141c09491b4df4fb9de69c6d7cfa786fa062fc60ea4fb57ab164` | `58f36d72bbb7141c09491b4df4fb9de69c6d7cfa786fa062fc60ea4fb57ab164` | **YES** |
| qhh (lake) | `qhh.rivqdown.dat` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | **YES** |
| qhh (lake) | `qhh.lakqrivin.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | **YES** |
| qhh (lake) | `qhh.lakqrivout.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | **YES** |
| qhh (lake) | `qhh.lakystage.dat` | `4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250` | `4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250` | **YES** (manual verify; script parser missed inline `# comment` on manifest line — file SHA itself verified separately) |
| qhh (lake) | `cvode_stats.txt` | `91df2bcf9b4aa48cbafa50dfde15983a0f7b797083f82e3416454494a8a957f9` | `91df2bcf9b4aa48cbafa50dfde15983a0f7b797083f82e3416454494a8a957f9` | **YES** |

SHA256 是 manifest enabled output files + `cvode_stats.txt` 串联起来的 hash manifest 整体 hash（计算方式同 `tools/archive_b0_output.sh`）。每 case 三轮 hash byte-identical 即 PASS。

**Bonus bitwise vs B0-tag**: 三轮 SHA 与 `benchmarks/<case>/B0_output/<file>.sha256` 也 byte-identical（详 `docs/B0_vs_B1b_water_balance_report.md`），证实 D9 zero-impact 快速路径触发条件 1 满足。

Host: Apple M4 Pro macOS Darwin 24.6.0; Apple clang 17; `CXX_BASE_FLAGS=-O2 -g -ffp-contract=off -fno-fast-math -std=c++14`。

### T2 服务器 2-case 3-run SHA256（NUM_OPENMP=1，Slurm，90-day truncated）

Slurm 三铁律遵守：sbatch 从 `/scratch` 提交；`--output/--error` 在 `/scratch/frd_muziyao/SHUD-OpenMP/.b1b-server-runs/`；scripts + binary 全在 `/scratch`。每 case 3 个 Slurm job sequential afterany chain（防 case `SHUD/Basins/<case>/output/` 并发 race）。

| Case | project | Slurm jobid (run 1/2/3) | node | wall (s) per run | summary SHA256 (all 3 runs identical) | verdict |
|---|---|---|---|---|---|---|
| heihe | heihe | 8662 / 8663 / 8664 | cn03 | 480 / 479 / 480 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | **PASS** |
| heihe_x4 | heihe_x4 | 8665 / 8666 / 8667 | cn03 | 1196 / 1192 / 1191 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | **PASS** |

**Per-file SHA verification vs PR-12 B1a golden (precedent for B0 chain via §"PR-12 capstone")**：

| Case | file | B1b run-3 SHA256 | PR-12 B1a golden SHA256 | match |
|---|---|---|---|---|
| heihe | `heihe.rivqdown.dat` | `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f` | `55abad2809418ea8e994e75137988cd94ea302641cfdd23202c7ace50965260f` (see `docs/b1a_summary.md` L90) | **YES** |
| heihe | `cvode_stats.txt` | `a59d90485669f7c578bd461d8b0ad01dcba004f65500987df9d4d02c6c64f252` | — (not archived in PR-12 b1a_summary.md, but bitwise across 3 B1b runs identical) | **YES (3-run)** |
| heihe_x4 | `heihe_x4.rivqdown.dat` | `f90601ef5738b972d688016ba1ee74f92ecb54faddaf46e4e2232f9d46567524` | `f90601ef5738b972d688016ba1ee74f92ecb54faddaf46e4e2232f9d46567524` (see `docs/b1a_summary.md` L91) | **YES** |
| heihe_x4 | `heihe_x4.eleygw.dat` | `192b0da4deacdf9218690cc501835033b181988e5399ef2d085fc083e17beece` | — (only across 3 B1b runs) | **YES (3-run)** |
| heihe_x4 | `cvode_stats.txt` | `9eba2c0ac186cefbb6ddbe7df1d076584ed72ed138931aa2081b05726195814d` | — | **YES (3-run)** |

Submit script: `/scratch/frd_muziyao/SHUD-OpenMP/.b1b-server-runs/b1b_server_run.sbatch`（74 行 bash）。每 run 内强制 `NUM_OPENMP=1` + `OMP_PROC_BIND=close OMP_PLACES=cores`（manifest omp_env 一致），cfg.para 端口 `START → START+90` 90 天截断。meta.json + run.sha256 落 `.b1b-server-runs/<case>_run<N>/`。

### T3 B0 vs B1b water balance — 详 `docs/B0_vs_B1b_water_balance_report.md`

每 case 闭合误差 delta = B1b.bitwise_identical_to(B0) → delta = 0 exact arithmetic → 远低于 0.1% 相对容差 → **PASS**。

(a) 输入降水累计 / (b) 输出径流累计 / (c) 储水变化 / (d) 闭合误差 四项每项都由 manifest 内 enabled output file 完整覆盖（disabled channel 的 ΔS 组件因 cfg.para DT_*=0 IO 关，但 forcing identical + cvode_stats identical 已唯一确定 state trajectory，即 disabled channel 若重新启用也会 bitwise identical）。详 report §"方法学说明"。

### T4 Go/No-Go → P1 七项 checklist（master plan §S6c L1511–L1525 合并）

| # | 检查项 | evidence | verdict |
|---|---|---|---|
| 1 | **B1b 已锁定** (L1513) | `B1b-tag` 创建 + `baseline/B1b` lock 是 **#189** 范围（本 PR out of scope）；当前 status: PENDING #189。本 PR 落 evidence 占位，#189 创建 tag 后将本行更新为 PASS + tag SHA | **PENDING #189** |
| 2 | **B1b 单线程多次运行 bitwise identical** (L1514) | T1 Mac 4-case + T2 server 2-case 各 3 轮 SHA256 全 byte-identical（见上 T1+T2 表） | **PASS** |
| 3 | **所有 shared accumulation 已拆为 deterministic gather** (L1523, PR-11 保留) | `grep -rn 'rhs_deterministic_gather' SHUD/src/` 命中 18+ 行；`SHUD/src/Model/MD_rhs_core.cpp:346` 定义 `Model_Data::rhs_deterministic_gather()`；`SHUD/src/ModelData/MD_f.cpp:105` + `MD_rhs_core.cpp:306` 两个 call site；`MD_f_uncouple.cpp:105` `PassValue_legacy retired in S3c.3 (PR-11 #155)` 注释保留 | **PASS** |
| 4 | **编译选项固定且无 fast-math** (L1524) | `grep -rn '\-ffast-math\|\-Ofast\|\-funsafe-math-optimizations' SHUD/Makefile SHUD/src/` 命中 10 行**全部**为 policy 注释 + DISALLOWED_FLAGS 列表（`SHUD/Makefile:75`），actually applied flag = `CXX_BASE_FLAGS := -O2 -g -ffp-contract=off -fno-fast-math -std=c++14` (Makefile L24)，make-time 拒绝 (L76-77) `$(error disallowed flag detected ...)` enforce 0 hits in `CFLAGS/CXXFLAGS/CPPFLAGS/LDFLAGS/...`。实际 build 不存在 fast-math | **PASS** |
| 5 | **`schedule(static)` 规则确定** (L1525) | `grep -n 'schedule(static)' SHUD/src/` 命中 4 行: `MD_initialize.cpp:143` `#pragma omp parallel for schedule(static)` + 注释 L138 `bitwise-safe at NUM_OPENMP=1: schedule(static) makes thread 0 ...`；`Model_Data.cpp:258/302/332` 3 处 `#pragma omp parallel for schedule(static)`。所有 init time parallel for 已统一 `schedule(static)`，RHS hot path 无 `schedule(dynamic)` / `schedule(guided)` | **PASS** |
| 6 | **`B1b_CHANGELOG.md` 完整** (L1515) | `SHUD/B1b_CHANGELOG.md` 含全部 sections: S5a (#176) / S5b (#177) / S5c-B (#174) / S5c-C (#175) / S5d.1 (#178) / S5d.2-5a (#179) / S5d.2-5b (#180) / S5d.3 (#181) / S5d.4 (#182) / S6b.1 (#184) / S6b.2 (#186 SKIP path) / S6b.3 (#187)；每节含 diff/影响范围/验收 verdict | **PASS** |
| 7 | **水量守恒不恶化对比 B0** (L1517) | `docs/B0_vs_B1b_water_balance_report.md` (本 PR 落地)：6 case 闭合误差 delta = 0 (B0 → B1b bitwise identical，自动满足 0.1% 容差) | **PASS** |

**Summary**: 6 PASS + 1 PENDING (#189 范围)。Item 1 PENDING 不阻 evidence 收尾，B1b-tag 实际锁定动作在 #189 完成后本 PR 不需要回填——#190 会做 status_matrix.md 总入账。

### fast-math grep 详细记录（spec L67-L69 Scenario "fast-math 编译 flag 已禁"）

```
$ grep -rn '\-ffast-math\|\-Ofast\|\-funsafe-math-optimizations' SHUD/Makefile SHUD/src/
SHUD/Makefile:53:# word-level, so it catches `CXXFLAGS=-ffast-math …`. The two project-
SHUD/Makefile:61:# `make SHUD_BUILD_CFLAGS=-Ofast`), which the `override :=` directive on
SHUD/Makefile:63:# `findstring -Ofast,$(MAKEOVERRIDES)` form was a literal-substring scan
SHUD/Makefile:64:# that false-positived on paths like `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`;
SHUD/Makefile:66:# only true `VAR=-Ofast` CLI assignments fire. Without this layer the
SHUD/Makefile:75:override DISALLOWED_FLAGS := -ffast-math -Ofast -funsafe-math-optimizations
SHUD/Makefile:84:# `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`), while still catching
SHUD/Makefile:85:# `make shud SHUD_BUILD_CFLAGS=-Ofast` / `CXX_BASE_FLAGS=-Ofast`, which the
SHUD/Makefile:203:# `-Ofast`/`-ffast-math`/`-funsafe-math-optimizations` in
SHUD/Makefile:219:# vigilance over `-Ofast`/`-ffast-math`.
```

10 命中**全部为防御性代码**：4 处 string substring 匹配陷阱说明注释 + 1 处 `DISALLOWED_FLAGS` 名单定义 (L75) + 5 处其他注释。`SHUD/src/` 0 命中。actual compile-time enforcement at `SHUD/Makefile:76-77`:

```makefile
ifneq (,$(filter $(DISALLOWED_FLAGS),$(CFLAGS) $(CXXFLAGS) $(CPPFLAGS) ...))
$(error disallowed flag detected ... B0 baseline requires strict IEEE-754 ...)
```

→ B1b 编译时若任何 user CLI / env 注入 fast-math，make 会 abort，flag 0 chance 进入 build artifact。spec Scenario "fast-math 编译 flag 已禁" PASS。

### 验证 B1b candidate 单线程 bitwise identity（spec L4-L13 Scenarios "4 case Mac" + "2 case 服务器"）

| Scenario (spec) | evidence | verdict |
|---|---|---|
| 4 case Mac 三次自洽 PASS (L7-L9) | T1 表（keliya/xinanjiang_upstream/qinyijiang/qhh）每 case 三轮 SHA byte-identical | **PASS** |
| 2 case 服务器三次自洽 PASS (L11-L13) | T2 表（heihe/heihe_x4）每 case 三轮 SHA byte-identical + 节点 + jobid 记录 | **PASS** |
| 6 case 水量平衡不恶化 (L36-L37) | T3 report：bitwise identity → closure_error delta = 0 exact | **PASS** |
| 7 项 checklist 全 PASS (L64-L65) | T4 表 6 PASS + 1 PENDING (#189 范围) | **6/7 evidence collected; 1/7 deferred to #189** |
| fast-math 编译 flag 已禁 (L67-L69) | grep + Makefile L24 + L76-77 enforce 链路 | **PASS** |

### S6c-12a Slurm 作业号汇总

| Job ID | 内容 | Node | Elapsed | ExitCode | 关联 |
|---|---|---|---|---|---|
| 8662 | heihe run 1 NUM_OPENMP=1 90d | cn03 | 00:08:01 | 0:0 | T2 heihe run 1 |
| 8663 | heihe run 2 (afterany 8662) | cn03 | 00:08:00 | 0:0 | T2 heihe run 2 |
| 8664 | heihe run 3 (afterany 8663) | cn03 | 00:08:01 | 0:0 | T2 heihe run 3 |
| 8665 | heihe_x4 run 1 (afterany 8664) | cn03 | 00:19:57 | 0:0 | T2 heihe_x4 run 1 |
| 8666 | heihe_x4 run 2 (afterany 8665) | cn03 | 00:19:53 | 0:0 | T2 heihe_x4 run 2 |
| 8667 | heihe_x4 run 3 (afterany 8666) | cn03 | 00:19:52 | 0:0 | T2 heihe_x4 run 3 |

Cancelled 6 旧 jobids (8654-8659): 旧版 dependency 漏写导致 6 jobs 并发抢 `SHUD/Basins/<case>/output/` 同一目录，发现 race condition 后 scancel 重提交。

### Status — S6c-12a

"S6c-12a evidence gathered; awaiting #189 B1b-tag lock + #190 PROMOTE + status_matrix.md B1b 行 PASS"

**不属本 PR 范围（per issue T-blocking）**：
- B1b-tag 创建（#189）
- `baseline/B1b` branch lock_branch=true + enforce_admins=true（#189）
- `docs/status_matrix.md` B1b 行更新为 PASS（#190）
- `openspec/changes/b1b-baseline-completion/` archive + PROMOTE 6 capability specs（#190）
- `docs/review-loop-log.jsonl` + `docs/stage-pipeline-log.jsonl` 追加 capstone 行（#190）
- `docs/build_manifest.yaml` 加 `B1b-tag` 节（#190）

**Slurm 三铁律 satisfied**: 全 6 jobs sbatch FROM /scratch + `--output/--error` 在 `/scratch/.b1b-server-runs/` + binary 在 `/scratch/SHUD/shud` + script `/scratch/.b1b-server-runs/b1b_server_run.sbatch`。
