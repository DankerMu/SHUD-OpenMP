## Purpose

规约 P1.0 pre-audit (`_Element::updateElement` / `_River::updateRiver` / `_Lake::update` + `f_updatei` case 1-5) + MD_update.cpp `f_update()` 三 owner loop (element L64-L105 / river L107-L125 / lake L136-L147) 并行化 + 三 owner loop 统一 pragma clause 全局 CI gate + per-case authoritative baseline tag 表 + RHS snapshot bitwise vs baseline + 完整 run bitwise vs baseline + CVODE 15-key stats identical (NUM_OPENMP=1) + NUM_OPENMP 1/2/4/8 scaling + A3a/A3b 报告 + strict FP flags compliance evidence。本 capability 是 P1 epic 的 Phase C（PR-C #215 audit + PR-D #216 element + PR-E #217 river + PR-F #218 lake + PR-H #219 snapshot + PR-I #220 Mac full-run + PR-J #221 server full-run + PR-K1 #222 Mac scaling + PR-K2 #223 server scaling），为 first parallel candidate baseline。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/p1-update-omp/specs/<capability>/spec.md PROMOTE 而来（#226 P1 capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。
- 三 owner loop 统一 pragma form：`#pragma omp parallel for schedule(static) default(none) shared(<显式列出>) private(i)`；**禁止** `schedule(dynamic|guided)` / `#pragma omp atomic` / floating `+=` / `reduction(+:sum)`（§8.1 strict 禁止项）。
- per-case authoritative baseline = B1b-tag canonical（tag chain 一致 case 同时 ≡ B0/B1a/B1-tag golden 作为加分一致性校验）。
- P1 forced gate = NUM_OPENMP=1 bitwise + CVODE 15-key identical；跨线程 bitwise（A3c）**不强制**（详 design D5）；A3a 失败 SHALL fall back to A3b（ULP ≤ 4 + max_abs_diff < 1e-12）。
- 跨线程 A3a + A3b 双 FAIL（如 server N≥4 实测）属 P7 final-fusion deterministic-reduction debt（master plan §6 P7），**不阻** P1 lock per design D5 NG3；spec 验收 Scenario SHALL 覆盖该 dual-FAIL 状态、CVODE nst drift logging、并 cross-ref P7 work scope。
- Mac dev-only scaling 数字 SHALL 标 NG1（不计入 §1.1.1 go/no-go）；服务器 cn0X 是 §1.1.1 加速比验收起点。
- strict FP flags 三 grep gate：(a) `-O2 / -ffp-contract=off / -fno-fast-math` 各 ≥ 1 hit；(b) `-ffast-math / -Ofast / -funsafe-math-optimizations` 0 hit；(c) `-fopenmp` ≥ 1 hit。

## Requirements

### Requirement: P1.0 pre-audit — updateElement / updateRiver / lake.update 共享写审查

audit-only PR SHALL 产 `docs/p1/p1_audit_update_funcs.md`，逐函数列 read/write set：

- `_Element::updateElement(double Ysurf, double Yunsat, double Ygw)` (`SHUD/src/classes/Element.cpp:257`) — callee 形参 Ysurf/Yunsat/Ygw 对应 caller 实参 uYsf[i]/uYus[i]/uYgw[i]；element-local read/write 是否触及任何全局 / 共享对象
- `_River::updateRiver(double newY)` (`SHUD/src/classes/River.cpp:49`) — callee 形参 newY 对应 caller 实参 uYriv[i]；river-local read/write 是否触及任何全局 / 共享对象
- `_Lake::update()` (`SHUD/src/classes/Lake.cpp:104`) — **无入参**；同时审 caller-side 赋值 `lake[i].yStage = yLakeStg[i]`（位于 `SHUD/src/ModelData/MD_update.cpp:138-139`）是否 owner-local；callee 内 read/write 是否触及任何全局 / 共享对象
- `Model_Data::f_updatei()` case 1-5 五个 owner loop（`SHUD/src/ModelData/MD_update.cpp:6-62`）— 与 `f_update()` 三 loop 的 read/write 集异同；本 change 不并行化（NG7），但审查供 P2a 参考

audit 结论 SHALL 二选一：

- **(a) safe**：三函数均 owner-local，无共享对象写 → P1.1/P1.2/P1.3 实施可直接进行
- **(b) unsafe**：发现某函数有共享对象写 → 按 design **D9** 分类 (b.1) bitwise-stable 单独立 fix 增 sub-issue 在本 change 内做；(b.2) non-bitwise-stable 推迟 B1c-tag stacking

audit 输出 SHALL 引用具体行号 + variable 名 + read/write 类型（如 `Ele[i].u_satKr = ...` 是 owner-local write）。

#### Scenario: safe 路径 sign-off

- **WHEN** audit 三函数均 owner-local
- **THEN** `docs/p1/p1_audit_update_funcs.md` 结论 = (a) safe；P1.1/P1.2/P1.3 PR 可启动

#### Scenario: unsafe 路径 sign-off（D9 分类）

- **WHEN** audit 发现共享写
- **THEN** `docs/p1/p1_audit_update_funcs.md` 结论 = (b) unsafe，且 SHALL 按 design D9 明确分类 (b.1) bitwise-stable fix（in-scope 新增第 15 sub-issue + PR-Cfix）或 (b.2) non-bitwise-stable fix（推迟 B1c-tag stacking）；本 change 据分类增加 fix sub-issue 或推迟相关 owner loop PR

#### Scenario: f_updatei audit 覆盖

- **WHEN** 审 `f_updatei()` case 1-5
- **THEN** audit table 含五个 case 的 read/write 集 + 与 `f_update()` 三 loop 的对应关系（case 4 含 `Riv[i].updateRiver()` 调用，与 f_update river loop 是同函数的另一个调用点）；不并行化结论显式记录

---

### Requirement: MD_update.cpp element loop 并行化 (L64-L105)

`SHUD/src/ModelData/MD_update.cpp` `f_update()` element loop **L64-L105**（单 outer `for (i=0; i<NumEle; i++)`，body 含：L67-L73 QeleSubAt/QeleSurfAt/QeleSubTot/QeleSurfTot 清零；L74-L75 uYsf/uYus 赋值；L76-L86 Ele BC 更新；L87-L88 qEleExfil/qEleInfil 清零）SHALL 加 `#pragma omp parallel for schedule(static) default(none) shared(<显式列出>) private(i)` 单 pragma 覆盖整个 outer for。禁止 `schedule(dynamic)` / `schedule(guided)` / `atomic` floating `+=` / `reduction(+:sum)`（§8.1 strict 禁止项）。reset loops L127-L135（NumRiv+NumEle 闲置 owner-local）与 DY zero loop L148-L150（NumY-bounded，不属于任何 owner loop）**不**在本 change 范围。

#### Scenario: element loop 加 pragma（行窗口断言）

- **WHEN** `awk 'NR>=63 && NR<=105 && /#pragma omp parallel for/' SHUD/src/ModelData/MD_update.cpp | wc -l`
- **THEN** ≥ 1（element loop 起点 L64 前一行 L63 函数头 +/- 1，pragma 必须位于 L63-L64 行间窗口）；同时 `schedule(static)` 显式声明

#### Scenario: element loop 含 default(none) + private(i)

- **WHEN** `awk 'NR>=63 && NR<=65' SHUD/src/ModelData/MD_update.cpp | grep -E 'default\(none\).*private\(i\)|private\(i\).*default\(none\)'`
- **THEN** ≥ 1 hit（两 clause 都出现在 element loop pragma 行）

#### Scenario: 无 dynamic / guided / atomic

- **WHEN** `grep -E 'schedule\((dynamic|guided)\)|#pragma omp atomic' SHUD/src/ModelData/MD_update.cpp`
- **THEN** 0 hit

---

### Requirement: MD_update.cpp river loop 并行化 (L107-L125)

`SHUD/src/ModelData/MD_update.cpp` `f_update()` river loop **L107-L125**（单 outer `for (i=0; i<NumRiv; i++)`，body 含：L108 uYriv 赋值 + L111 `Riv[i].updateRiver(uYriv[i])` 调用 + L112-L121 Riv BC 更新 + L122-L124 DEBUG check）SHALL 加 `#pragma omp parallel for schedule(static) default(none) shared(<显式列出>) private(i)` 单 pragma 覆盖整个 outer for。前置 P1.0 audit 必须确认 `_River::updateRiver()` owner-local（详 P1.0 pre-audit Requirement）。

#### Scenario: river loop 加 pragma（行窗口断言）

- **WHEN** `sed -n '106,126p' SHUD/src/ModelData/MD_update.cpp | grep -c '#pragma omp parallel for'`
- **THEN** ≥ 1（river loop 起点 L107 前一行 L106 +/- 1）；`schedule(static)` 显式声明

#### Scenario: river loop 含 default(none) + private(i)

- **WHEN** `sed -n '106,108p' SHUD/src/ModelData/MD_update.cpp | grep -E 'default\(none\).*private\(i\)|private\(i\).*default\(none\)'`
- **THEN** ≥ 1 hit

---

### Requirement: MD_update.cpp lake loop 并行化 (L136-L147)

`SHUD/src/ModelData/MD_update.cpp` `f_update()` lake loop **L136-L147**（单 outer `for (i=0; i<NumLake; i++)`，body 含：L137 yLakeStg 赋值 + L138 `lake[i].yStage = yLakeStg[i]` + L139 `lake[i].update()` 无参调用 + L140 y2LakeArea + L141-L146 QLakeSub/Surf/qLakeEvap/qLakePrcp/QLakeRivIn/Out 清零）SHALL 加 `#pragma omp parallel for schedule(static) default(none) shared(<显式列出>) private(i)` 单 pragma 覆盖整个 outer for。前置 P1.0 audit 必须确认 `_Lake::update()` (无入参，详 P1.0 audit) owner-local。lake loop 中**不**汇总跨 element/river/lake 的 flux（§6 P1 禁止项）。

#### Scenario: lake loop 加 pragma（行窗口断言）

- **WHEN** `sed -n '135,148p' SHUD/src/ModelData/MD_update.cpp | grep -c '#pragma omp parallel for'`
- **THEN** ≥ 1（lake loop 起点 L136 前一行 L135 +/- 1）

#### Scenario: lake loop 含 default(none) + private(i)

- **WHEN** `sed -n '135,137p' SHUD/src/ModelData/MD_update.cpp | grep -E 'default\(none\).*private\(i\)|private\(i\).*default\(none\)'`
- **THEN** ≥ 1 hit

#### Scenario: lake loop 内无跨 owner 汇总

- **WHEN** 检查 lake loop body 内代码
- **THEN** 无 `qLakeEvap[lake] += ...` / `QLakeRivIn[lake] += ...` 跨 owner 写

---

### Requirement: 三 owner loop 统一 pragma clause 全局 CI gate

P1 候选 commit 上 SHALL 通过统一 CI gate 校验三 owner loop pragma clause 一致性。

#### Scenario: 全局 default(none) + private(i) hit count = 3

- **WHEN** `grep -E '#pragma omp parallel for[^\n]*default\(none\)[^\n]*private\(' SHUD/src/ModelData/MD_update.cpp | wc -l`
- **THEN** ≥ 3（element / river / lake 三 pragma 均含 `default(none)` 与 `private(`）

---

### Requirement: per-case authoritative baseline tag 表

P1 候选 commit bitwise 验证 SHALL 引用下列 per-case authoritative baseline tag 表（避免 B0/B1a/B1b/B1-tag 多 tag 指代歧义）。当多 tag 历史一致（B0=B1a=B1b=B1）时 P1 SHALL 同时 ≡ 三者；当 case 历史存在 canonical SHA rebase（如 qhh B1b 阶段 SHA rebase），SHALL 以表中 "P1 authoritative" 列为 binding：

| Case | P1 authoritative baseline | tag chain 一致性 |
|---|---|---|
| keliya | B1b-tag canonical | B0 = B1a = B1b = B1（bitwise-stable 全链一致）|
| xinanjiang_upstream | B1b-tag canonical | B0 = B1a = B1b = B1 |
| qinyijiang | B1b-tag canonical | B0 = B1a = B1b = B1 |
| qhh | B1b-tag canonical | B1b rebased canonical SHA（PR #188 验证）；P1 同 B1b/B1-tag；B0/B1a 历史值仅参考 |
| heihe | B1b-tag canonical | B0 = B1a = B1b = B1 |
| heihe_x4 | B1b-tag canonical | B0 = B1a = B1b = B1 |

P1 SHALL 同时验 (a) ≡ "P1 authoritative" 列；(b) 在 tag chain 一致 case 上，B0/B1a-tag golden 也同字节，作为加分一致性校验。

#### Scenario: per-case baseline 表对齐

- **WHEN** 检查 `docs/p1/p1_fullrun_bitwise.md` / `docs/p1/p1_rhs_snapshot_bitwise.md`
- **THEN** 每 case 验收节明确写出 baseline tag（默认 B1b-tag canonical）+ tag chain 一致性脚注

---

### Requirement: RHS snapshot bitwise vs per-case authoritative baseline

P1 候选 commit 上 SHALL 跑 RHS snapshot probe（任一 NumEle，单 RHS 调用）+ §9 master plan 列的所有 snapshot 数组（`uYsf / uYus / uYgw / uYriv / yLakeStg / DY / qEle* / QeleSurf / QeleSub / QsegSurf / QsegSub / QrivSurf / QrivSub / QrivUp / QrivDown / QLake* / qLakeEvap / qLakePrcp`）的 bitwise 比较 vs **per-case authoritative baseline 表** 列出的 tag，4 Mac case + 2 server case 全部 PASS。

#### Scenario: 4 Mac case RHS snapshot bitwise PASS

- **WHEN** P1 候选 commit 上 4 Mac case (keliya / xinanjiang_upstream / qinyijiang / qhh) 跑 RHS snapshot probe（NUM_OPENMP=1）
- **THEN** 所有 §9 列出的数组与 per-case authoritative baseline（默认 B1b-tag canonical）同字节

#### Scenario: 2 server case RHS snapshot bitwise PASS

- **WHEN** server cn0X 上 heihe + heihe_x4 跑 RHS snapshot probe（NUM_OPENMP=1）
- **THEN** 所有数组与 per-case authoritative baseline（默认 B1b-tag canonical）同字节

---

### Requirement: 完整 run bitwise vs per-case authoritative baseline + CVODE 15-key stats identical (NUM_OPENMP=1)

P1 候选 commit 上 NUM_OPENMP=1 SHALL 跑完整 90 天截断 run，4 Mac case canonical summary SHA + 2 server case rivqdown SHA SHALL 与 **per-case authoritative baseline 表** 列出的 tag（默认 B1b-tag canonical）同字节；tag chain 一致 case 同步与 B0/B1a/B1-tag golden 对齐（作为加分一致性校验）；CVODE 15-key stats（`nst / nfe / nfeLS / netf / nni / nli / nps / npe / nge / nsetups / nliSetup / nfQe / netfQe / nniS / nliS`）SHALL identical。3-run 自洽（同 binary 同 NUM_OPENMP=1 三次连跑 SHA 三次一致）。

#### Scenario: 4 Mac case full-run bitwise + 3-run identical

- **WHEN** `tools/archive_b0_output.sh <case> 3` 在 P1 候选 commit 上跑 4 Mac case
- **THEN** 每 case 三轮 SHA256 三次一致；canonical summary SHA ≡ per-case authoritative baseline（默认 B1b-tag canonical）；tag chain 一致 case (keliya/xinanjiang_upstream/qinyijiang) 同步 ≡ B0 `repeatability.txt sha256_run1`；qhh 仅 ≡ B1b-tag canonical（B0/B1a 历史 SHA rebase 前不强制）

#### Scenario: 2 server case full-run bitwise + 3-run identical

- **WHEN** server cn0X 上 heihe + heihe_x4 三次 Slurm 90 天截断 NUM_OPENMP=1
- **THEN** 每 case 三轮 SHA256 三次一致；rivqdown ≡ per-case authoritative baseline（默认 B1b-tag canonical）+ B0/B1a-tag golden (heihe `55abad28…` / heihe_x4 `f90601ef…`)（tag chain 一致 case）

#### Scenario: CVODE 15-key stats identical

- **WHEN** P1 候选 commit 上跑 6 case，对比 cvode_stats.txt vs per-case authoritative baseline（默认 B1b-tag canonical）
- **THEN** 所有 15 key 完全一致

---

### Requirement: NUM_OPENMP 1/2/4/8 scaling 测试 + A3a/A3b 报告

P1 候选 commit 上 SHALL 跑 NUM_OPENMP=1/2/4/8 scaling 测试，产 `docs/p1/p1_perf_baseline.md`，含每 case × 每 NUM_OPENMP 的：
- wall-clock (s)
- speedup vs NUM_OPENMP=1
- canonical summary SHA + 与 NUM_OPENMP=1 baseline 的 bitwise/ULP 比较
- A3a verdict (bitwise) 或 A3b verdict (ULP ≤ 4 + max_abs_diff < 1e-12)

scaling 测试 SHALL 覆盖：Mac 4 case 开发期参考（不计入 §1.1.1 go/no-go），server 2 case heihe + heihe_x4（§1.1.1 加速比验收起点）。NUM_OPENMP>1 跨线程 bitwise 不强制（详 design D5）。

#### Scenario: Mac scaling 报告

- **WHEN** Mac 本地 4 case × 4 N（1/2/4/8）跑完
- **THEN** `docs/p1/p1_perf_baseline.md` 含 16 行表（case × N），每行 wall + speedup + A3a/A3b verdict

#### Scenario: server scaling 报告

- **WHEN** server cn0X heihe + heihe_x4 × 4 N 跑完
- **THEN** `docs/p1/p1_perf_baseline.md` 含 8 行 server 表 + Slurm jobid 记录

#### Scenario: A3b 兜底允许通过（A3a FAIL but A3b PASS 单路径）

- **WHEN** 某 NUM_OPENMP>1 配置 RHS snapshot 不通过 A3a (bitwise) 但通过 A3b (ULP≤4 + max_abs_diff < 1e-12)
- **THEN** P1 lock 不阻塞；`docs/p1/p1_perf_baseline.md` 标 A3b verdict + 注 "P7 final fusion 阶段调试"

#### Scenario: A3a + A3b dual-FAIL 状态（per F-K2-1 PROMOTE upgrade #226）

- **WHEN** 某 NUM_OPENMP>1 配置（如 server N≥4 实测）RHS snapshot **同时** 不通过 A3a (bitwise) 与 A3b (max_abs_diff 远超 1e-12 threshold + max_ulp 远超 4 threshold + n_diff > 90%) — 实测案例：PR-K2 #223 server N=4/N=8 vs N=1 same-binary baseline heihe + heihe_x4 共 4-cell dual-FAIL（max_abs_diff 4-20e5 / max_ulp ~9e18 / n_diff 98.4-98.7%）
- **THEN** P1 lock **不阻塞** per design D5 NG3 + master plan §6 P7 final-fusion deterministic-reduction debt；`docs/p1/p1_perf_baseline.md` SHALL：(a) 标 dual-FAIL verdict（不掩饰为单 A3b fallback）；(b) 记录 CVODE `nst` drift evidence（如 heihe N=1/2/4/8 = 6773/6773/6585/6684 = 跨 N 步数 bifurcation 实证）；(c) cross-ref P7 final-fusion deterministic-reduction work scope（master plan §6 P7：fork-join 最小化 + chunk-fixed `schedule(static)` 消除 N-dependent reduction tree depth transition）；(d) 显式声明根因不在 PR-D/E/F 三 pragma 的 owner-local 设计（同一 binary 在 N=2 vs N=1 是 A3a bitwise PASS 的，dichotomy 在 N≥4 reduction-tree depth 增长时出现），后续 root-cause framing 走 F-K2-2 post-promote

---

### Requirement: strict FP flags compliance 验证 + 可执行 evidence

P1 候选 commit 的 SHUD 编译 SHALL 用 §8.1.1 compiler matrix 规定的 strict FP flags（Mac Apple Clang ≥ 13: `-O2 -ffp-contract=off -fno-fast-math -fopenmp`；server GCC ≥ 10: `-O2 -ffp-contract=off -fno-fast-math -fopenmp`）。编译 log SHALL 包含完整 flags 串供 grep 验证。Mac 端 SHALL 通过 PR-H 描述附 `make_shud_mac.log`；server 端 SHALL 通过 PR-J 描述附 `make_shud_server.log`。每个 PR 描述 SHALL 含三 grep gate 输出：(a) 必有子串 hit ≥ 1；(b) 禁止子串 0 hit；(c) `-fopenmp` 显式 ≥ 1。

#### Scenario: Mac 编译 flags 三 grep gate（PR-H 描述）

- **WHEN** 在 Mac 本地执行 `make shud 2>&1 | tee make_shud_mac.log` 并附 PR-H 描述
- **THEN** (a) `grep -E '\-O2|\-ffp-contract=off|\-fno-fast-math' make_shud_mac.log | wc -l` ≥ 3；(b) `grep -E '\-ffast-math|\-Ofast|\-funsafe-math-optimizations' make_shud_mac.log | wc -l` = 0；(c) `grep -c '\-fopenmp' make_shud_mac.log` ≥ 1

#### Scenario: server 编译 flags 三 grep gate（PR-J 描述）

- **WHEN** 在 server cn0X 执行 `make shud 2>&1 | tee make_shud_server.log` 并附 PR-J 描述
- **THEN** (a) `grep -E '\-O2|\-ffp-contract=off|\-fno-fast-math' make_shud_server.log | wc -l` ≥ 3；(b) `grep -E '\-ffast-math|\-Ofast|\-funsafe-math-optimizations' make_shud_server.log | wc -l` = 0；(c) `grep -c '\-fopenmp' make_shud_server.log` ≥ 1

#### Scenario: CI workflow strict FP flags gate (optional)

- **WHEN** `.github/workflows/serial-baseline.yml` 含 step "verify strict FP flags"
- **THEN** step 内执行 `grep -E '\-O2|\-ffp-contract=off|\-fno-fast-math' build.log` ≥ 3 hit + `grep -E '\-ffast-math|\-Ofast|\-funsafe-math-optimizations' build.log` 0 hit；未配置该 step 时由 PR-H / PR-J 描述补 evidence
