## Mode taxonomy (M10 修订, E' closure)

P1d epic capstone 引入 4-mode spec 把 strict 承诺限定到正确 mode：

| Mode | N_Vector | RHS | Bitwise gate | Build | Production status |
|---|---|---|---|---|---|
| `serial` | Serial | Serial | N=1 vs `P1-update-omp-tag` canonical strict | `make shud` | **production default** |
| `strict-omp` | Serial | StrictOMP | cross-N bitwise + nst Δ=0 + N=1 reverse-compat strict | `make shud SHUD_ENABLE_OPENMP_RHS=1` (待 P1e 实现) | **production candidate** |
| `det-omp` | NVECTOR_REPRO_OMP (custom) | StrictOMP | cross-N bitwise + nst Δ=0 (same) | fallback if `strict-omp` 加速不够 | P2 后续优化 |
| `fast-omp` | NVECTOR_OPENMP (stock) | StrictOMP / Serial | MAY 不可复现，明确 non-production | `make shud_omp` (current default) | **research artifact only** |

当前 (P1d E′ closure 时) 只有 `serial` mode 通过 strict 验收；`strict-omp` 待 P1e (F 路) 实现；`fast-omp` 明确 non-production。

## ADDED Requirements

### Requirement: NUMA env 标准化（P1d.1）

P1d 实施 SHALL 在 server sbatch template + Mac OMP env 同步标准化 `OMP_PROC_BIND=close` + `OMP_PLACES=cores`；server 侧追加 `numactl --interleave=all`。所有 P1d.2/.3 跑均在此 env 下执行。

#### Scenario: server sbatch template 含 NUMA env

- **WHEN** 检查 P1d 后续 sbatch 模板（`/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/*.sbatch`）
- **THEN** 文件 SHALL 含三行: `export OMP_PROC_BIND=close` + `export OMP_PLACES=cores` + `numactl --interleave=all` 在 `srun shud_omp <case>` 前置
- **AND** sbatch log 应含 `numactl --hardware` 输出（实测 cn0X NUMA 拓扑入 P1d.1 文档）

#### Scenario: Mac OMP env 同步

- **WHEN** Mac local run P1d binary
- **THEN** run script 或 environment SHALL 含 `export OMP_PROC_BIND=close` + `export OMP_PLACES=cores`（Mac libomp 弱绑定 informational only, 但保证 env 一致）

#### Scenario: env 不可用时的 fallback（build 冲突触发）

- **WHEN** 实测 cn0X `numactl --interleave=all` 与 SHUD `make shud_omp` 冲突（build 层 wrapper 报错 / 链接失败）
- **THEN** SHALL fallback 至 `numactl --cpubind=0 --membind=0` 单 NUMA node 强制 binding + 记录 fallback 决策入 `docs/p1d/p1d_summary.md` §"NUMA 拓扑实测" 节

#### Scenario: env 不可用时的 fallback（PR-F intermediate FAIL 触发，per OQ2）

- **WHEN** PR-F 8-cell intermediate verification 在 `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `numactl --interleave=all` 配置下三 SHALL gate 任一 FAIL（per OQ2 决策时机）
- **THEN** PR-G 阶段 SHALL fallback 至 `numactl --cpubind=0 --membind=0` 单 NUMA node 强制 binding 重跑 + 记录决策入 `docs/p1d/p1d_summary.md` §"NUMA env fallback rationale" 节
- **AND** 若 fallback 重跑仍 FAIL → 触发 R6 unbounded escape hatch，进 §10 FAIL 处理流程（不在本 change scope 解决）

### Requirement: Legacy dead-code deletion P1d.2.0 PR-C0

P1d 实施 SHALL 在 PR-C0（element first-touch 之前）删除三个 dead legacy 函数定义 + 三个声明：`SHUD/src/ModelData/MD_update.cpp::f_update`、`SHUD/src/ModelData/MD_f.cpp::f_loop`、`SHUD/src/ModelData/MD_f.cpp::f_applyDY` 函数体；`SHUD/src/ModelData/Model_Data.hpp` L334/L342/L348 三个 declaration。保留 `f_updatei` / `f_loopET` / `f_loop1..5` / `f_applyDY_surf` / `f_applyDY_unsat` / `f_applyDY_gw` / `f_applyDY_river` / `f_applyDYi`（uncouple path 由 `f.cpp::f_surf/f_unsat/f_gw/f_river/f_lake` 在用）+ 保留 SHUD_DUMP_RHS 标签字符串 `"f_update"` / `"f_loop"` / `"f_applyDY"` 字面（dump golden contract 不动）。

> **Mid-stream revision (2026-06-24)**: original P1d.2 design targeted `MD_update.cpp::f_update / f_loop / f_applyDY`. Phase 4 cross-review (PR #290 round 1) discovered those 3 functions are DEAD CODE — zero call sites under `grep -rEn "f_update[ \t]*\(" SHUD/src --include='*.cpp' --include='*.hpp'`. CVODE entry `SHUD/src/Model/f.cpp:54` dispatches exclusively to `MD->rhs_core(...)`, which calls `rhs_update / rhs_flux / rhs_apply` (`SHUD/src/Model/MD_rhs_core.cpp`). S5c PR-8 refactor created the `rhs_*` triplet as "PURE CARRY-OVER" copies (per `MD_rhs_core.cpp:1-25` header) and retired the `f_*` originals as legacy reference. Inserting first-touch into `f_update` is a no-op for the production hot path. Per user Option C decision, P1d.2.0 (NEW prerequisite) deletes the 3 dead functions before P1d.2.1-3 retarget to the live `rhs_*` owners. Spec scenarios below reflect this corrected plan; proposal.md "What Changes" and design.md D2 carry a deferred rewrite note (sync at PR-K capstone).

#### Scenario: legacy 3 函数 deletion verification

- **WHEN** PR-C0 实施后 `grep -rEn '\bModel_Data::(f_update|f_loop|f_applyDY)\s*\(' SHUD/src --include='*.cpp' --include='*.hpp'`
- **THEN** SHALL 0 hit（3 个 definitions 全删）
- **AND** `grep -n 'void f_update\|void f_loop(double t)\|void f_applyDY(double' SHUD/src/ModelData/Model_Data.hpp` SHALL 0 hit（3 declarations 全删；不包含 `f_updatei` / `f_loopET` / `f_loop1` 等带后缀的 live 声明）

#### Scenario: live uncouple-path functions 不受影响

- **WHEN** PR-C0 实施后
- **THEN** `grep -n 'void f_updatei\|void f_applyDYi\|void f_loopET\|void f_loop[1-5](' SHUD/src/ModelData/Model_Data.hpp` SHALL ≥4 hit（uncouple path API surface 保留）
- **AND** `SHUD/src/Model/f.cpp` 内 `f_surf / f_unsat / f_gw / f_river / f_lake` 5 个 uncouple entry 的调用链（`f_updatei / f_loopX / f_applyDYi`）SHALL 全部仍可 compile + link

#### Scenario: SHUD_DUMP_RHS 标签字符串保留

- **WHEN** PR-C0 实施后 `grep -n 'shud_rhs_dump_point("f_update"\|shud_rhs_dump_point("f_loop"\|shud_rhs_dump_point("f_applyDY"' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** SHALL ≥3 hit（标签字符串保留于 rhs_update / rhs_flux / rhs_apply 内 emit 点；dump golden 兼容性维持）
- **AND** `SHUD/src/ModelData/MD_rhs_dump.h` / `MD_rhs_dump.cpp` 注释 + 默认值 `"f_update"` 字面保留

#### Scenario: PR-C0 build + bitwise gate

- **WHEN** PR-C0 删 3 函数 + 3 声明后 `cd SHUD && make clean && make shud_omp && make shud`
- **THEN** build PASS（无 unresolved symbol）
- **AND** keliya N=1 `output/keliya.out/keliya.rivqdown.dat` SHA256 SHALL 与 deletion 前（SHUD@`3a0004c`）byte-identical（删 dead code 不影响 numerical path）

### Requirement: MD_rhs_core.cpp rhs_* live path parallel first-touch（P1d.2）

P1d 实施 SHALL 在 `SHUD/src/Model/MD_rhs_core.cpp` 的 live update 路径（`rhs_update` / `rhs_flux` 内 river block / `rhs_flux` 内 lake block）前置 parallel first-touch loop，使 hot 数组的 page-fault 在 RHS 评估的并行 region 内由对应 owner 触发。

背景：`SHUD/src/ModelData/Model_Data.cpp::malloc_EleRiv` L251-L346 内已存在 allocation-time parallel first-touch（由 `g_numa_first_touch_enabled` 守门，shud.cpp L49/L74/L88 由 `OMP_PROC_BIND` 触发），但仅 malloc 一次性触发，不能修复长时 CVODE 推进期的 page migration / 跨 NUMA cache-line ping-pong。P1d.2 在 `rhs_update` / `rhs_flux` 内前置 steady-state first-touch loop 是每步 RHS 评估期的 warm-up，与 allocation-time first-touch **互补**（不替换）。所有 first-touch loop SHALL 由 `g_numa_first_touch_enabled` 守门（与 allocation-time first-touch 一致策略）。

> Note: 原 spec 文本曾将 `MD_update.cpp::f_update` 误识为 live owner，PR-C0 deletion 后该误识自然消除。本 Requirement 全部 Scenario 以 `MD_rhs_core.cpp` 为权威。

#### Scenario: element first-touch loop（P1d.2.1, PR-C, live path）

- **WHEN** `grep -nE '#pragma omp parallel for' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** 在 `rhs_update` 函数（L58-L149 区间）element owner for-i 循环前 SHALL 命中一处 first-touch loop pattern, `schedule(static)` + `default(none)` + 仅 zero-write 元素 hot 字段
- **AND** 该 loop SHALL 由 `if (g_numa_first_touch_enabled) { ... }` 守门（与 `Model_Data.cpp::malloc_EleRiv` L302-L317 allocation-time first-touch gate 同模式）
- **AND** loop body 仅写 element-owned 字段 = 0.0；不引入 reduction / 累加 / non-zero 写
- **AND** `make shud_omp` build PASS + keliya N=1 `output/keliya.out/keliya.rivqdown.dat` SHA256 与 PR-C0 deletion 后 baseline byte-identical（first-touch zero-write 不改变 serial path）

#### Scenario: element first-touch 字段集归档（P1d.2.1 前置，PR-C）

- **WHEN** PR-C 实施 element first-touch loop 前
- **THEN** `docs/p1d/p1d_first_touch_design.md` §"字段集 grep 输出" 章 SHALL 含 cross-pragma 字段宇宙 grep（per OQ1 设计 D2，PR-C 单 doc 覆盖 element/river/lake 三 live block owner-local write 全集，避免 PR-D/PR-E 重复建 doc）：
  - 主 grep（element + river + 公共 stems）：`grep -nE 'QeleSurf|QeleSub|QrivSurf|QrivSub|QrivUp|qLake|yLakeStg|y2LakeArea|Qe2r_' SHUD/src/Model/MD_rhs_core.cpp`
  - 补充 grep（lake 大写 stems，case-sensitive — `qLake` 小写不命中 `QLake*`）：`grep -nE 'QLakeSub|QLakeSurf|QLakeRivIn|QLakeRivOut|qLakeEvap|qLakePrcp' SHUD/src/Model/MD_rhs_core.cpp`
- **AND** doc 含两 grep 实际命中行清单 union（去重后入 §"字段集 grep 输出" 章表格）；行号引用 `MD_rhs_core.cpp` live 函数（rhs_update / rhs_flux / rhs_apply）
- **AND** PR-C 实施时 element first-touch zero-write 字段集严格匹配 element-owned 子集（rhs_update body 写入字段：`QeleSurfAt` / `QeleSubAt` / `QeleSurfTot` / `QeleSubTot` / `Qe2r_Surf` / `Qe2r_Sub`；river / lake 字段不在 PR-C scope，仅 OQ1 doc 归档以服务 PR-D / PR-E）
- **AND** doc lake 子集 SHALL 非空（含 `QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` 至少 4 stems 命中行 in `MD_rhs_core.cpp`），作 PR-E 实施前置 verification gate（OQ1 决策落地）

#### Scenario: river first-touch loop（P1d.2.2, PR-D, live path）

- **WHEN** `rhs_flux` 函数内 river owner block（`MD_rhs_core.cpp::rhs_flux` 内对 river 字段的 reset / accumulate 段；实施时 implementer 在 PR-D 内 grep 精确行号入 task doc）
- **THEN** 前置 SHALL 含 river first-touch loop, `schedule(static)` + `default(none)` + `if (g_numa_first_touch_enabled)` 守门 + 仅 zero-write river-owned 字段（`QrivSurf` / `QrivSub` / `QrivUp` 等，per OQ1 doc RIVER 子集）
- **AND** 与 element first-touch 完全独立（不共享 hot 字段）
- **AND** PR-D 合并后 keliya N=1 byte-identical 至 PR-C 后 baseline

#### Scenario: lake first-touch loop（P1d.2.3, PR-E, live path）

- **WHEN** `rhs_update` 函数内 lake owner block（`MD_rhs_core.cpp::rhs_update` 内 `for (i = 0; i < NumLake; i++)` 循环, post-PR-D tree ~L172-183，对 lake 字段的 reset / accumulate 段；实施时 implementer 在 PR-E 内 grep 精确行号入 task doc）
- **AND** spec drift note: 原 PROMOTE 文本曾写 "rhs_flux 函数内 lake owner block"，PR-C / PR-D round-1 review 实测确认 lake-owned per-i zero writes 全部位于 `rhs_update` lake block（`QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` / `qLakeEvap` / `qLakePrcp` 6 fields 在 L177-182 zero-write），`rhs_flux` lake 段仅 accumulate（不 reset），故 PR-E 实施 site 与 PR-C element block 同 site（`rhs_update`）；PR-E 实施时同步 spec 修正
- **THEN** 前置 SHALL 含 lake first-touch loop, `schedule(static)` + `default(none)` + `if (g_numa_first_touch_enabled)` 守门 + 仅 zero-write lake-owned 字段（`QLakeSub` / `QLakeSurf` / `QLakeRivIn` / `QLakeRivOut` / `qLakeEvap` / `qLakePrcp` 6 fields, per OQ1 doc LAKE 子集；`yLakeStg` / `y2LakeArea` 排除 — 二者是 persistent state writer, 在 lake block 内 `= Y[iLAKE]` / `= lake[i].u_toparea` 非零写, 不是 pure-zero 候选）
- **AND** Mac CI 通过 + qhh asan/ubsan PASS

#### Scenario: first-touch 不引入 reduction order 改变（PR-C / PR-D / PR-E 增量验证）

- **WHEN** PR-C / PR-D / PR-E 各自合并后跑 keliya N=1 单线程
- **THEN** snapshot output SHALL 与 上一个 PR merge 后的 baseline byte-identical（first-touch zero-write 不影响 serial numerical path；`g_numa_first_touch_enabled` gate 在 OMP_PROC_BIND unset 时保证 loop skip + 保持 deterministic-vs-serial behavior）
- **AND** PR-E 合并后额外跑 qhh N=1 vs PR-D 后 baseline byte-identical（验证 lake-bearing case 不受新增 lake first-touch 影响）

### Requirement: Kahan revert（P1d.3.1, PR-G）

P1d 实施 SHALL revert PR-I #265 Kahan 注入（SHUD pin `3a0004c` → 等价 `de9545d` + P1d.2 first-touch commits stacked），通过 `git apply -R docs/p1c/p1c_kahan_patch.diff` 或等效手动 revert 实现，保留 4 helpers 结构但回到 naive `+=` 累加。

#### Scenario: SHUD source revert verified

- **WHEN** `grep -nE 'fabs|Neumaier|c \+=' SHUD/src/Model/MD_rhs_core.cpp` post-PR-G
- **THEN** SHALL 不命中（Neumaier compensation patch 已 revert）
- **AND** 4 helper functions（`fixed_pairwise_sum_indexed` / `fixed_pairwise_sum_range` / `fixed_leftfold_sum_indexed` / `fixed_leftfold_sum_pair_indexed`）保留结构 + 回到 naive `acc += src[i]`

#### Scenario: Kahan revert 后 SHUD pin 一致性

- **WHEN** 外层 `git ls-tree HEAD SHUD` post-PR-G
- **THEN** SHUD pin SHALL 是 P1d.2 first-touch 3 commits + Kahan revert commit 的最新 SHA（非 `de9545d` 字面，而是其等价 numerical 行为 + first-touch 叠加）
- **AND** SHUD `openmp-baseline` 分支 push 包含上述 commit 序列

#### Scenario: Kahan revert 后 N=1 SHA 候选恢复

- **WHEN** P1d.3.2 PR-H 跑 heihe N=1 NUM_OPENMP=1 单线程
- **THEN** 输出 `output/heihe.out/heihe.rivqdown.dat` SHA256 SHALL 等同 `P1-update-omp-tag` canonical SHA（`7f22bd6faa438d50...`）

### Requirement: 三 SHALL gate 严格 hard gate（P1d.3.2 server + P1d.4.2 Mac）

P1d 实施 SHALL 在 PR-H 阶段跑 server 8-cell（heihe + heihe_x4 × N ∈ {1,2,4,8}），并在 PR-J 阶段跑 Mac 4-cell（4 case × N=1），合计验证三 SHALL gate 全 PASS。任一项 FAIL → P1d 不 closure。Mac 仅 N=1 验证（reverse-compat），non-Mac SHALL gate（A3a + nst 跨 N）严格限定 server（per design D7 Mac informational only）。

> **Mode binding (M10 修订, E' closure)**: 本 Requirement 三 SHALL gate 的 mode 绑定 per 上方 4-mode 表。
> - L130 A3a bitwise SHALL gate: applies to `strict-omp` + `det-omp` modes (待 P1e 实现 strict-omp); for current `fast-omp` mode (i.e. 现有 `make shud_omp` build) 该 gate 是 INFORMATIONAL ONLY (PR-H 实测 FAIL 是 NVECTOR_OPENMP reduction tree 顺序根因，per master plan v1.5 / M10 §6 P1d)。
> - L139 nst Δ + ladder SHALL gate: same — applies to `strict-omp` + `det-omp` modes; for `fast-omp` mode 是 INFORMATIONAL ONLY (CVODE WRMS norm 同样过 N_Vector OMP reduction)。
> - L145 N=1 reverse-compat SHALL gate: applies to `serial` + `strict-omp` modes; **该 gate IS met by `serial` mode** per PR-G clean Kahan revert + PR-I Mac 6-cell anchor + Mac 9-SHA matrix byte-identical 证据。

#### Scenario: §4.4 A3a bitwise 跨 N (server only)

- **WHEN** P1d.3.2 PR-H server 8-cell 实测
- **THEN** heihe + heihe_x4 每 case 的 N ∈ {1,2,4,8} 4 cell `output/<case>.out/<case>.rivqdown.dat` SHA256 SHALL 全等
- **AND** 双 case = 2 distinct SHA（heihe SHA, heihe_x4 SHA）
- **AND** (Mode binding) gate evaluation applies to `strict-omp` + `det-omp` modes; `fast-omp` mode 实测 FAIL is INFORMATIONAL per E' closure narrative

#### Scenario: §4.5 nst 跨 N (server only, heihe Δ=0 强制 + heihe_x4 |Δ| ≤ 2)

- **WHEN** P1d.3.2 PR-H 8-cell `output/<case>.out/cvode_stats.txt` 读取
- **THEN** heihe `nst` 跨 N ∈ {1,2,4,8} SHALL 全等（Δ=0 严格 hard gate）
- **AND** heihe_x4 `nst` 跨 N SHALL `|Δ_nst| ≤ 2`（per D4 ladder，反映 mesh 加密后 SPGMR 收敛差异，仍属 strict 非 softening）
- **AND** (Mode binding) gate evaluation applies to `strict-omp` + `det-omp` modes; `fast-omp` mode 实测 FAIL is INFORMATIONAL per E' closure narrative

#### Scenario: NUM_OPENMP=1 reverse-compat（6-case 矩阵）

- **WHEN** P1d.3.2 PR-H + P1d.4.2 PR-J 完成
- **THEN** 6 case（heihe + heihe_x4 server + 4 Mac case: keliya / xinanjiang_upstream / qinyijiang / qhh）× NUM_OPENMP=1 `output/<case>.out/<case>.rivqdown.dat` SHA256 SHALL byte-identical 至 `P1-update-omp-tag` canonical SHA
- **AND** (Mode binding) gate evaluation applies to `serial` + `strict-omp` modes; the `serial` mode result IS met per PR-G + PR-I + Mac 9-SHA matrix evidence (N=1 byte-identical confirmed)

#### Scenario: 任一 SHALL FAIL → P1d 不 closure

- **WHEN** PR-H 三 SHALL gate 任一项 FAIL
- **THEN** PR-H verdict SHALL 为 FAIL + epic issue 标 blocked + 不进入 PR-I/J/K/L/M
- **AND** 由用户决策 P1d 延展或 master plan 修订（不在本 change scope）

> **M10 修订 (E' closure, 2026-06-24)**: 用户决策走 E' containment closure path (per master plan v1.5 / M10 §6 P1d.4) — 保留 PR-H FAIL verdict + 完成 PR-I (Mac reference) + PR-K (capstone docs) + PR-L (P1d-tag annotated) + PR-M (PROMOTE + Epic close)；P1d 不走原 closure 路径，走 PARTIAL CLOSURE via E' path。原 Scenario "任一 SHALL FAIL → P1d 不 closure" 在 E' closure 框架下被 ↓ 下方 "P1d E' containment closure verdict" Scenario 取代。

#### Scenario: P1d E' containment closure verdict

- **WHEN** P1d epic PR-H 8-cell server final run completes
- **AND** 3 SHALL gate verdict measured (A3a bitwise FAIL / nst Δ FAIL / N=1 reverse-compat PARTIAL)
- **THEN** P1d closure SHALL be classified as **PARTIAL CLOSURE via E' containment path**
- **AND** 4-mode spec SHALL be adopted with strict gate binding per mode (above)
- **AND** P1e (F path) SHALL be opened as next epic to implement `strict-omp` mode (Serial N_Vector + StrictOMP RHS) per ADR-0002 Path 1 SELECTED
- **AND** PR-C/D/E steady-state first-touch loops SHALL be tagged DEPRECATED (no owner-compute consumer in current Serial RHS path)
- **AND** PR-G Kahan revert SHALL be preserved (`serial` mode N=1 vs `P1-update-omp-tag` canonical SHA byte-identical confirms clean revert)

### Requirement: Mac SHALL Scenario closure（P1d.4, spec p1c L154-157）

P1d 实施 SHALL 闭合 spec `p1c-deterministic-reduction` L154-157 SHALL Scenario "Mac N=1 反向兼容: 4 Mac case 与 P1-update-omp-tag Mac canonical SHA bitwise"，即采集 P1 era Mac N=1 rivqdown.dat reference + P1d-binary Mac 比对验证 byte-identical。

#### Scenario: P1-update-omp-tag Mac binary 重 build + 跑

- **WHEN** P1d.4.1 PR-I 在 Mac 上独立 worktree 切 `P1-update-omp-tag`（避免污染 main worktree，per tasks §5 前置） + `git submodule update --init --recursive` + `cd SHUD && ./configure && make shud_omp`
- **THEN** build PASS + 4 Mac case × N=1 `output/<case>.out/<case>.rivqdown.dat` 跑成功
- **AND** SHA256 archived 入 `docs/p1d/p1d_mac_reference.md`（新建）

#### Scenario: P1d-binary Mac 比对（per-case）

- **WHEN** P1d.4.2 PR-J 在 Mac 上（PR-G merge 后 SHUD pin 已固定 + `git submodule update`）build P1d-binary（SHUD pin = P1d.3.1 revert 后 SHA + P1d.2 first-touch stacked）+ 跑 4 Mac case × N=1
- **THEN** 每 case 独立验收 P1d-binary `output/<case>.out/<case>.rivqdown.dat` SHA256 SHALL byte-identical 至 PR-I 采集的 P1-update-omp-tag Mac reference SHA
- **AND** qhh case 显式 per `openspec/specs/p1c-deterministic-reduction/spec.md` L157 caveat（PR #188 rebase tag-chain caveat）允许 deferred → partial closure 默认接受，记入 `docs/p1d/p1d_mac_reference.md` §"P1d-binary per-case 比对" 节（与 PR-I reference SHA 同 doc，不单独 split）

#### Scenario: P1-update-omp-tag Mac build 失败时 graceful degradation（per-case）

- **WHEN** P1d.4.1 PR-I `make shud_omp` FAIL（SUNDIALS 链接问题或 Apple clang 兼容性）— per-case 而非整 PR
- **THEN** 该单 case Spec L154-157 SHALL 标 partial closure + `docs/p1d/p1d_mac_reference.md` 记录失败原因 + ADR 决策；**其它 case 仍归档 reference SHA 入 doc，不因单 case fail 撤回整 PR-I**
- **AND** 同 case P1d-binary Mac N=1 自一致性内部验证 PASS（跨 P1d-binary build 不同 invocation 字节等同）作 partial verification

### Requirement: baseline/P1d 分支 + P1d-tag（P1d.5）

P1d epic capstone SHALL 创建 `baseline/P1d` 分支（从 `baseline/P1c` HEAD `c58e04f` 分出）+ `P1d-tag` annotated tag + branch lock。

#### Scenario: baseline/P1d 分支创建

- **WHEN** P1d.0 epic intake 阶段
- **THEN** `gh api repos/DankerMu/SHUD-OpenMP/git/refs --method POST` 创建 `refs/heads/baseline/P1d` 指向 `baseline/P1c` HEAD（`c58e04f`）
- **AND** PR-A..PR-M base = `baseline/P1d`

#### Scenario: P1d-tag annotated 创建

- **WHEN** PR-L 合并后立即 (post-merge action)
- **THEN** `git tag -a P1d-tag <baseline/P1d HEAD> -m '<message>' && git push origin P1d-tag`
- **AND** annotated message SHALL 含: P1c PARTIAL CLOSURE → P1d closure narrative + NUMA env + first-touch + Kahan revert 三 phase stack + 三 SHALL gate verdict 实测 SHA + D11 5 → 6 tag chain immutability baseline + SHUD pin trail

#### Scenario: baseline/P1d branch lock

- **WHEN** PR-M post-merge action
- **THEN** `gh api repos/.../branches/baseline/P1d/protection --method PUT --field lock_branch=true --field enforce_admins=true --field allow_force_pushes=false --field allow_deletions=false`
- **AND** verify `gh api repos/.../branches/baseline/P1d --jq '.protection.lock_branch.enabled'` = true

#### Scenario: D11 6 tag chain 不可变验证

- **WHEN** P1d epic capstone 完成时
- **THEN** 6 tag SHA `git rev-parse <tag>` SHALL 全部不变 (B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag 5 historical + P1d-tag 新增)
- **AND** baseline/P1c lock 状态不动

### Requirement: P1c carve-out + Mac SHALL Scenario 闭环验证

P1d 实施 SHALL 通过 P1d.3 三 SHALL gate + P1d.4 Mac closure，间接关闭 spec `p1c-deterministic-reduction` L100-L103 carve-out Scenario + L154-157 Mac SHALL Scenario。本 change 不直接修改 `openspec/specs/p1c-deterministic-reduction/spec.md`（PROMOTE 历史保留）。P1c spec L103 字面写 "carve-out 推 P9 行"（PROMOTE 当时未引入 P1d 子阶段），由 master plan §6 P1c.7 (2026-06-23) 修订重命名为 "carve-out 推 P1d 行"；P1d 实施视 P9 字面为 P1d 语义同义，此映射记录在 `docs/p1d/p1d_summary.md` §"P1c carve-out closure" 章明示（per D11 PROMOTE 历史 immutability，保留 P1c spec 字面 P9，不 retroactive 改 PROMOTE 内容）。

#### Scenario: P1c carve-out Scenario 闭环 documented

- **WHEN** P1d epic capstone PR-K 写 `docs/p1d/p1d_summary.md`
- **THEN** SHALL 含 §"P1c carve-out closure" 章 + 明确 P1d 三 SHALL gate verdict 表 + 引用 spec p1c-deterministic-reduction L100-L103 / L154-157 路径
- **AND** glossary 已存在术语 "P1d carve-out (writer noise governance)"（per glossary L216 entry，命名本身即 P1c epic forward debt）由 PR-M PROMOTE 阶段更新 status: 由 "PARTIAL CLOSURE / pending P1d epic" → "P1d carve-out CLOSED via P1d epic 2026-06-XX"（术语 key 与 status 述语统一为 P1d；narrative 注明源于 P1c epic forward debt）

### Requirement: 三 negative grep gate 保留（P1d-wide）

P1d 实施 SHALL 保留 P1c era 已建立的 3 negative grep gate（per `openspec/specs/p1c-deterministic-reduction/spec.md` L76/L81/L86），即 P1d epic 期间任何 SHUD 改造 commit 不可引入：
- 新 macro pattern: `SHUD_USE_DETERMINISTIC_REDUCTION` / `SHUD_DET_REDUCT` / `SHUD_PAIRWISE`
- `schedule(dynamic|guided)` in `MD_rhs_core.cpp`
- `#pragma omp atomic` in `SHUD/src/`

#### Scenario: 三 negative grep gate post-P1d

- **WHEN** P1d epic capstone 完成时
- **THEN** 三 grep 命令均返回 0 hit (per P1c capstone established)
- **AND** PR-K capstone 文档显式 re-verify
