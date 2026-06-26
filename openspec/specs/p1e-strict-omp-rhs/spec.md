# p1e-strict-omp-rhs Specification

## Purpose
TBD - created by archiving change p1e-strict-omp-rhs. Update Purpose after archive.
## Requirements
### Requirement: 2×2 build matrix 因果实验（P1e.2 分两 phase 实施）

P1e 实施 SHALL 完成 2×2 build matrix 因果实验：4 build × N∈{1,2,4,8} × 3 repeats × 4 case = 192 cell 完整跑批 + verdict + decision branch routing。**两 phase 实施**（per design D1，解决 "mode C 依赖 PR-F 才能 run" 与 "P1e 实施前必跑 2×2" 的循环）：

- **Phase 1 (PR-B/C/D，pre-implementation)**：mode A + mode B 96 cell 必跑；通过 "A 跨 N bitwise + B 跨 N 不 bitwise" 判据才允许进 PR-F StrictOMP 实施
- **Phase 2 (PR-G post-merge，post-implementation)**：mode C + mode D 96 cell；验证 mode C 跨 N bitwise + 加速比 + decision branch routing 最终结论

#### Scenario: 2×2 实验 cell 完整跑批

- **WHEN** P1e.2 PR-C (Mac) + PR-D (server) 完成
- **THEN** 4 build × 4 N × 3 reps × 4 case = 192 cell 全部 cell 数据归档：
  - 每 cell `cvode_stats.txt` 15-key set 完整 (`nst/nfe/nli/nni/netf/ncfn` 等)
  - 每 cell `<case>.rivqdown.dat` SHA256
  - 每 cell wall (s)
  - 每 cell CV_Y state vector hash (tools/p1e_cv_y_hash/ 工具)
- **AND** Mac 96 cell + server 96 cell 数据分别归档至 `docs/p1e/p1e_pr_c_2x2_mac.md` + `docs/p1e/p1e_pr_d_2x2_server.md`
- **AND** 综合表归档至 `docs/p1e/p1e_2x2_verdict.md`（含 4 mode × 4 N × 3 reps × 4 case 全数据 + decision branch routing 结论）

#### Scenario: 2×2 mode A 同 build 同 N × 3 reps bitwise 守门

- **WHEN** 读 `docs/p1e/p1e_2x2_verdict.md`
- **THEN** mode A (Serial NVec + Serial RHS) 每 case 每 N 的 3 reps `rivqdown.dat` SHA256 SHALL byte-identical
- **AND** mode A 跨 N (N=1,2,4,8) 同 case `rivqdown.dat` SHA256 SHALL byte-identical（solver 自身在 fixed build 下 deterministic 的必要前提）
- **AND** 若 mode A 不满足此 Scenario，P1e SHALL 暂停 + 调查 toolchain (per design OQ1 + R1b)

#### Scenario: mode A toolchain 调查产出物

- **WHEN** mode A Phase 1 守门 FAIL 触发暂停
- **THEN** SHALL 在 `docs/p1e/p1e_toolchain_investigation.md` 记录：
  - (a) 失败 cell 详情 (case / N / rep / SHA delta)
  - (b) 编译器 + libomp/libgomp + glibc + SUNDIALS pin 完整 version 报表
  - (c) bisect 历史 (vs P1d-tag mode A reference SHA 何时开始漂)
  - (d) 调查结论 + 恢复条件 (toolchain pin 或调整 → mode A 跨 reps 重恢复)
- **AND** 满足 (d) 后 PR-C/D 重跑 Phase 1，否则 epic 永久 blocked + 用户决策点

#### Scenario: 2×2 mode B 跨 N 不等 + mode A 跨 N 相等 → NVECTOR_OPENMP reduction 是主因

- **WHEN** PR-E verdict 综合
- **THEN** mode A (Serial NVec) 跨 N bitwise 同时 mode B (OpenMP NVec) 跨 N 不 bitwise → 确认 NVECTOR_OPENMP reduction 是 cross-N 散度的主因（不是 RHS race）
- **AND** verdict 写入 `docs/p1e/p1e_2x2_verdict.md` 含 实测 SHA + cross-N delta + 因果论证

#### Scenario: 2×2 mode C 跨 N bitwise + nst Δ=0 + 加速 per-case threshold PASS → F 路成立

- **WHEN** PR-E Phase 1 verdict 起草 + PR-I Phase 2 verdict amend（PR-I 是 mode C 24-cell + 加速比 per-case threshold 数据生产者，自然 owner Phase 2 verdict amend，详 tasks §4.6.3）
- **THEN** mode C (Serial NVec + StrictOMP RHS) 满足：
  - heihe + heihe_x4 跨 N∈{1,2,4,8} `rivqdown.dat` SHA256 全等
  - heihe `nst` 跨 N Δ=0；heihe_x4 `|Δ_nst| ≤ 2`
  - heihe N=8 加速比 ≥ **1.3×** (Amdahl 上界 ~1.7×)
  - heihe_x4 N=8 加速比 ≥ **1.5×** (Amdahl 上界 ~2.39×)
- **AND** decision branch routing → D12.1 happy path → P1e ship → 解锁 P2a
- **AND** Phase 2 verdict 填实 ownership = PR-I（tasks §4.6.3）；`docs/p1e/p1e_2x2_verdict.md` Phase 2 section 含 D12.1/.2/.3/.4 实际触发分支判定 + 决策证据 + cross-ref `docs/p1e/p1e_pr_i_strict_omp_verification.md`

#### Scenario: 2×2 mode C 跨 N FAIL → D12.2 fallback Path 2

- **WHEN** mode C 跨 N 仍分叉
- **THEN** verdict 写入 `docs/p1e/p1e_2x2_verdict.md` 含 RHS race / 共享 state / phase deps 三类可能根因排查清单
- **AND** decision branch routing → D12.2 fallback ADR-0002 Path 2 (NVECTOR_REPRO_OMP custom backend) → 用户决策点（scope 扩展或开新 epic）
- **AND** P1e.3 实施 SHALL 暂停（不进 PR-F）

#### Scenario: 2×2 mode C 跨 N PASS 但**两 case 均** < 各自 threshold → D12.3 fallback Path 3

- **WHEN** mode C 跨 N bitwise + nst Δ 满足，但 heihe N=8 < 1.3× **AND** heihe_x4 N=8 < 1.5×
- **THEN** decision branch routing → D12.3 fallback ADR-0002 Path 3 (SPGMR + block-Jacobi precond)
- **AND** P1e.3 实施可进 PR-F (StrictOMP RHS 仍要做) + 同时触发 P1e.8 block-Jacobi precond 新 PR-N

#### Scenario: 单 case 不达 threshold（另一 case 已达）→ partial closure 决策点

- **WHEN** mode C 跨 N bitwise + nst Δ 满足，且仅一 case < threshold（典型：heihe < 1.3×，heihe_x4 ≥ 1.5×）
- **THEN** 进 partial closure 决策点（用户决策 ship vs fallback；倾向 ship 当 heihe_x4 ≥ 1.5× 时，因 heihe_x4 是更具代表性的 mesh 规模 case）

### Requirement: `ExecPolicy::StrictOMP` 实施（P1e.3.1, PR-F）

P1e 实施 SHALL 在 `SHUD/src/Model/MD_rhs_core.cpp:802-811` 当前 `std::abort()` 桩位置实施真正 `ExecPolicy::StrictOMP` 路径：单 `#pragma omp parallel` 外层 + phase-based `#pragma omp for nowait/barrier` 内层 + `default(none) shared(...) private(...)` 严格 + 隐式 barrier 同步。

#### Scenario: `std::abort()` 桩替换

- **WHEN** PR-F 实施后 `grep -nE 'case ExecPolicy::StrictOMP|case ExecPolicy::ProductionOMP' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** `StrictOMP` case body SHALL **不** 含 `std::abort()` 调用
- **AND** `ProductionOMP` case body 可保留 `std::abort()`（P1e 不实施 ProductionOMP，留作 P8+ 范围）
- **AND** `default:` case (catches enumerator additions + ABI drift, MD_rhs_core.cpp L809-811) 保留 `std::abort()`
- **AND** `grep -cE 'std::abort\(\)' SHUD/src/Model/MD_rhs_core.cpp` 结果 SHALL ≤ 2（ProductionOMP + default case 保留；StrictOMP 已替换）

#### Scenario: 单 `#pragma omp parallel` 外层

- **WHEN** PR-F 实施后 `grep -cE '#pragma omp parallel' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** `StrictOMP` case body 内 `#pragma omp parallel` SHALL 出现且仅出现 1 次（不嵌套，不重复）
- **AND** 此 parallel region 内可有多个 `#pragma omp for` directive（phase-based for）

#### Scenario: `default(none)` 严格变量可见性

- **WHEN** PR-F 实施后 `grep -nE '#pragma omp parallel' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** 命中行 SHALL 含 `default(none)` clause
- **AND** 同行（或续行）显式列出 `shared(...)` + `private(...)` clauses，所有变量分类完整（编译期 catch 遗漏）

#### Scenario: phase-based for + implicit barrier

- **WHEN** `StrictOMP` case body 内 `#pragma omp for` directive 命中
- **THEN** 同 phase 内 for 可带 `nowait` clause（性能优化）；phase 切换处最后一个 for SHALL **不** 带 `nowait`（OpenMP 隐式 barrier 同步 phase）
- **AND** `schedule(static)` SHALL 出现在每个 `#pragma omp for` directive 上（per master plan §8.1 strict 禁止 `dynamic|guided`）

#### Scenario: 复用 `rhs_deterministic_gather()` 4 helpers

- **WHEN** PR-F 实施后 `grep -nE 'fixed_pairwise_sum_range|fixed_pairwise_sum_indexed|fixed_leftfold_sum_indexed|fixed_leftfold_sum_pair_indexed' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** SHALL ≥4 命中（P1c era 4 helpers 全部保留 + 在 StrictOMP path 内复用）
- **AND** 不引入新 reduction helper（避免与 P1c era infrastructure 重复）

#### Scenario: build PASS + keliya 跨 N bitwise 守门（PR-F: N=1 only / PR-G: N=1,2,4,8）

- **WHEN** PR-F 实施后 `cd SHUD && make clean && make shud SHUD_ENABLE_OPENMP_RHS=1`
- **THEN** build PASS（mode C build）+ keliya × N=1 mode C smoke (build PASS + run not abort)
- **AND** PR-F 阶段 N>1 跨 N bitwise **暂不**验证 — 因 Makefile 仍未附加 `-fopenmp`，mode C `#pragma omp parallel` 会编译成 serial code (_OPENMP undefined)
- **WHEN** PR-G 实施后 `cd SHUD && make clean && make shud SHUD_ENABLE_OPENMP_RHS=1`（PR-G 已附加 `-fopenmp` + 链 libomp）
- **THEN** keliya × N∈{1,2,4,8} mode C `rivqdown.dat` SHA256 全等（small case 守门）

### Requirement: mode C build 含 -fopenmp + 链 OpenMP runtime（P1e.3.2, PR-G）

P1e 实施 SHALL 在 PR-G 阶段扩展 `SHUD/Makefile` 让 `SHUD_ENABLE_OPENMP_RHS=1` 隐式同时附加 `CXX_OPENMP_CFLAGS` (`-fopenmp`) + `CXX_OPENMP_LFLAGS` (`-lomp`/`-lgomp`)。否则 mode C/D `#pragma omp parallel` 会被编译成 serial code，违 design D6。

#### Scenario: Makefile 自动附加 -fopenmp

- **WHEN** PR-G 实施后 `make -n shud SHUD_ENABLE_OPENMP_RHS=1`
- **THEN** recipe SHALL 含 `-fopenmp`（或平台等价：Darwin `-Xpreprocessor -fopenmp`）+ libomp/libgomp 链接 flag

#### Scenario: mode C binary 真链入 OpenMP runtime

- **WHEN** PR-G 后 `make shud SHUD_ENABLE_OPENMP_RHS=1` build 出 binary
- **THEN** Linux: `nm ./shud | grep GOMP_parallel` SHALL ≥1 命中；Darwin: `nm ./shud | grep _omp_get_max_threads` 或 `otool -L ./shud | grep libomp` SHALL ≥1 命中
- **AND** `nm ./shud | grep N_VNew_Serial` SHALL 含 symbol 引用；`nm ./shud | grep N_VNew_OpenMP` SHALL 不命中

#### Scenario: Macros.hpp omp.h 守门扩展

- **WHEN** PR-G 实施后 `grep -nE 'omp\.h|defined.*SHUD_ENABLE_OPENMP_RHS' SHUD/src/Model/Macros.hpp`
- **THEN** `<omp.h>` include 守门 SHALL 含 `defined(SHUD_ENABLE_OPENMP_RHS)`（union with `_OPENMP` + `SHUD_USE_OPENMP_NVECTOR`），避免 mode C 下 omp_set_num_threads declaration not found
- **AND** **path 正确性**：`SHUD/src/Model/Macros.hpp` 真实存在（`find SHUD -name Macros.hpp` 返回此唯一路径）；`SHUD/src/include/Macros.hpp` **不存在**，禁止 spec/design/tasks 任何引用使用错误路径（grep 若返回 0 hits 则属 vacuous PASS）
- **AND** **pre-extension sanity**：扩展前 `grep -c 'defined(_OPENMP) || defined(SHUD_USE_OPENMP_NVECTOR)' SHUD/src/Model/Macros.hpp` SHALL ≥1 hit（确认 anchor 真实存在，避免 PASS 是假阳）

#### Scenario: omp_set_num_threads 在 build C 下可达（P0 TE-1b 防漏）

- **WHEN** PR-G 实施后 build C (`make shud SHUD_ENABLE_OPENMP_RHS=1`，`SHUD_USE_OPENMP_NVECTOR` undefined)
- **THEN** `SHUD/src/Model/shud.cpp` 内 `omp_set_num_threads` 调用点 SHALL **被编译进 binary**（不能因守门只有 `#ifdef SHUD_USE_OPENMP_NVECTOR` 而被预处理 elide）
- **AND** verify by source grep: `grep -B2 -A2 'omp_set_num_threads' SHUD/src/Model/shud.cpp` 显示守门 SHALL 满足以下两种等价形式之一（**union form** 或 **拆为两段独立守门 form**，per tasks §3.5.2 显式允许 "或拆为两段"）：
  - **union form**：单个 call site 由 `#if defined(SHUD_USE_OPENMP_NVECTOR) || defined(SHUD_ENABLE_OPENMP_RHS)` 守门，一处包住 `omp_set_num_threads`
  - **拆为两段 form**：两个独立 call site，一个由 `#ifdef SHUD_USE_OPENMP_NVECTOR` 守门（保留历史 mode B/D NVector parity，参数为 `MD->CS.num_threads`），另一个由 `#if defined(SHUD_ENABLE_OPENMP_RHS)` 守门（StrictOMP RHS startup single-point set，参数为 `getenv("SHUD_RHS_THREADS")` ?: `omp_get_max_threads()`）
  - 二选一即满足本 Scenario（不能是单 `#ifdef SHUD_USE_OPENMP_NVECTOR` 一处而无 `SHUD_ENABLE_OPENMP_RHS` 守门）
  - PR-G 实际选择 **拆为两段 form**（reference: `SHUD/src/Model/shud.cpp` L132 NVector branch + L157 SHUD_ENABLE_OPENMP_RHS branch；rationale: mode D 下两支独立 set 顺序可读 + comment block 分别 anchored 到各自语义，per `docs/p1e/p1e_thread_split.md`）
- **AND** verify by binary symbol: build C 后 Linux `nm ./shud | grep omp_set_num_threads` 或 Darwin `nm ./shud | grep _omp_set_num_threads` SHALL ≥1 hit（call site 真编译入 binary）
- **AND** verify by runtime: `SHUD_RHS_THREADS=4 ./shud keliya` 输出含 `Number of Threads = 4` 或等价 diagnostic（per "mode C runtime thread count diagnostic" Scenario，cross-ref）— 若守门未扩 union，此 diagnostic 将退化为 `omp_get_max_threads()` 默认值（典型 = 物理核数，**不**等于 4）
- **AND** verify cross-file consistency: `SHUD/src/Model/MD_rhs_core.cpp` 内 `omp_set_num_threads` SHALL == 0 hit（per "RHS 入口不 re-set thread count" Scenario，确保唯一调用点在 shud.cpp startup）

### Requirement: thread count split（P1e.3.2, PR-G）

P1e 实施 SHALL 拆分 runtime env-var pair：`SHUD_RHS_THREADS` 控 RHS 并行度，`OMP_NUM_THREADS` 控 NVECTOR（P1e build 下 NVECTOR backend = `N_VNew_Serial`，故不读 `OMP_NUM_THREADS`）。`omp_set_num_threads` 调用 SHALL 在 startup 阶段单点 set（不在 RHS hot path 内反复调用）。

#### Scenario: `SHUD_RHS_THREADS` env 单点 startup set

- **WHEN** PR-G 实施后 `grep -nE 'SHUD_RHS_THREADS|getenv.*RHS_THREADS' SHUD/src/Model/shud.cpp SHUD/src/Equations/cvode_config.cpp`
- **THEN** SHALL ≥1 命中（startup 阶段 read `SHUD_RHS_THREADS` env）
- **AND** 实现：`omp_set_num_threads(getenv("SHUD_RHS_THREADS") ? atoi(getenv("SHUD_RHS_THREADS")) : omp_get_max_threads())`（在 `main()` cvode_config setup 之前调用）

#### Scenario: RHS 入口不 re-set thread count（避免 hot path 副作用）

- **WHEN** PR-G 实施后 `grep -cE 'omp_set_num_threads' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** SHALL == 0（`MD_rhs_core.cpp` 内 `ExecPolicy::StrictOMP` case body 不含 `omp_set_num_threads` 调用，避免 hot path getenv + 全局 runtime state mutate）
- **AND** RHS parallel region 用 `#pragma omp parallel num_threads(N)` clause 显式覆盖 thread count（不依赖外部 set）

#### Scenario: build flag `SHUD_ENABLE_OPENMP_RHS=1` 触发 `N_VNew_Serial`

- **WHEN** PR-G 实施后 `make shud SHUD_ENABLE_OPENMP_RHS=1` build
- **THEN** binary `./shud` (per Makefile L313 BUILDDIR=. + L321 TARGET_EXEC=$(BUILDDIR)/shud) 在 link 阶段 SHALL 链接 `sundials_nvecserial`（**不** 链 `sundials_nvecopenmp`）
- **AND** `nm ./shud | grep N_VNew` SHALL 含 `N_VNew_Serial` 符号引用
- **AND** `nm ./shud | grep N_VNew_OpenMP` SHALL **不** 命中（或仅在 dead code path 内）

#### Scenario: `omp_set_num_threads` 调用从 `SHUD_USE_OPENMP_NVECTOR` 条件内移出（positive grep）

- **WHEN** PR-G 实施后 `grep -nE 'omp_set_num_threads' SHUD/src/Model/shud.cpp`
- **THEN** SHALL ≥1 命中，**且**位于 `#ifdef SHUD_USE_OPENMP_NVECTOR` 守门**之外**（grep -B5 -A5 验证：调用点附近 5 行内不被 `#ifdef SHUD_USE_OPENMP_NVECTOR` ... `#endif` 包裹；或换 union 守门 `#if defined(SHUD_USE_OPENMP_NVECTOR) || defined(SHUD_ENABLE_OPENMP_RHS)`）
- **AND** P1e build (mode C: `SHUD_USE_OPENMP_NVECTOR=0` + `SHUD_ENABLE_OPENMP_RHS=1`) 下 `omp_set_num_threads` 仍可调用

#### Scenario: mode C runtime thread count diagnostic

- **WHEN** PR-G build C binary 跑 `SHUD_RHS_THREADS=4 ./shud keliya`
- **THEN** 输出 SHALL 含 `Number of Threads = 4` 或等价 diagnostic（确认 startup `omp_set_num_threads` 生效 + binary 真链入 OpenMP runtime）

### Requirement: PR-C/D/E P1d era steady-state first-touch removal（P1e.3.5, PR-H）

P1e 实施 SHALL 删除 `SHUD/src/Model/MD_rhs_core.cpp` 内 P1d era PR-C/D/E 添加的 3 处 steady-state first-touch loops（element block L62-95 / lake block L169-193 / river block L324-358，含 `if (g_numa_first_touch_enabled)` gate + `P1d.2.[123] (#27X) — steady-state ... first-touch warm-up` tag comments）。**保留** `SHUD/src/ModelData/Model_Data.cpp::malloc_EleRiv` 内 allocation-time first-touch (L257/L301/L331)；**保留** `SHUD/src/ModelData/MD_initialize.cpp::LoadIC` 内 load-time first-touch (L141-142)；**保留** P1c era 4 helpers；**保留** PR-G Kahan revert。

#### Scenario: steady-state first-touch loops 删除验证（anchor 在 MD_rhs_core.cpp 实际 P1d-era tokens）

- **WHEN** PR-H 实施后 `grep -cE 'P1d\.2\.[123].*steady-state.*first-touch' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** SHALL == 0（P1d era 3 处 tag comments 全删）
- **AND** `grep -c 'if (g_numa_first_touch_enabled)' SHUD/src/Model/MD_rhs_core.cpp` SHALL == 0（3 处 gate 全删）
- **AND** 删除前 pre-removal sanity：两 grep 应分别 ≥3 命中（avoid vacuous PASS — 确认 anchor 在删除前真实存在）
- **AND** `grep -nE '\[NUMA\] first-touch' SHUD/src/Model/MD_rhs_core.cpp` SHALL 0 hit

#### Scenario: allocation-time first-touch 保留（Model_Data.cpp）

- **WHEN** PR-H 实施后 `grep -nE 'first-touch begin tag=|g_numa_first_touch_enabled' SHUD/src/ModelData/Model_Data.cpp`
- **THEN** SHALL ≥1 hit（allocation-time first-touch 模式保留，tags hot.soa/QeleSurf_flat/Ele_AoS at L257/L301/L331）
- **AND** `Model_Data::malloc_EleRiv` 内 first-touch 逻辑不动

#### Scenario: load-time first-touch 保留（MD_initialize.cpp::LoadIC）

- **WHEN** PR-H 实施后 `grep -nE 'g_numa_first_touch_enabled' SHUD/src/ModelData/MD_initialize.cpp`
- **THEN** SHALL ≥1 hit（load-time LoadIC first-touch 保留，L141-142 tag LoadIC — load-time 一次性触发，mirrors allocation-time semantics，不在 RHS hot path）

#### Scenario: P1c era 4 helpers + PR-G Kahan revert 保留

- **WHEN** PR-H 实施后 `grep -nE 'fixed_pairwise_sum_range|fixed_pairwise_sum_indexed|fixed_leftfold_sum_indexed|fixed_leftfold_sum_pair_indexed' SHUD/src/Model/MD_rhs_core.cpp`
- **THEN** SHALL ≥4 hit（4 helpers 结构保留）
- **AND** `grep -nE 'fabs|Neumaier|c \+=' SHUD/src/Model/MD_rhs_core.cpp` SHALL 0 hit（Kahan/Neumaier compensation 已 revert，与 P1d post-PR-G 状态一致）

### Requirement: rivqdown.dat 输出缓存 audit（P1e.0, PR-A + 必要时 PR-B0）

P1e 实施 SHALL 在 PR-A epic intake 阶段 audit `SHUD/src/Model/shud.cpp` 内 `rivqdown.dat` 输出代码段，确认数据源是否为 internal cache (solver 内部最后一次 RHS 留下的 `FluxRiv` 缓存) 还是按 `tout` 边界 recompute。若发现 internal cache，新增 **PR-B0**（在 PR-B 2×2 driver script 之前）单独做 rivqdown 修复，不混入 PR-B 2×2 scope。

#### Scenario: cache audit 判据 + 报告归档

- **WHEN** PR-A audit 完成
- **THEN** `docs/p1e/p1e_rivqdown_cache_audit.md` SHALL 含：
  - rivqdown.dat 输出代码段路径 (file:line)
  - **判据**：在 output 代码段附近 grep `FluxRiv\[i\]\.q|FluxRiv\[i\]\.[A-Za-z]+`
    - 若直接出现在 output stream（e.g. `out << FluxRiv[i].q`）→ **internal cache**（unsafe）
    - 若先 `recompute_flux(Y, tout, &q_local)` 再 `out << q_local` → **tout recompute**（OK）
  - 数据源类别 (internal cache / `tout` recompute / 其它)
  - 若 internal cache → 修复计划入 PR-B0（新 PR）
  - 若 `tout` recompute → 标 "OK，不需修复"
- **AND** audit 命令清单 (grep / sed / 代码 trace) 入 doc

#### Scenario: 若 cache audit 发现 internal cache

- **WHEN** audit 结论 = internal cache
- **THEN** PR-B0 SHALL 重写 rivqdown.dat 输出为按 `tout` 边界 recompute (per design D5)
- **AND** 重写后 mode A 同 build 同 N × 3 reps `rivqdown.dat` SHA256 byte-identical (验证修复不破坏 deterministic)
- **AND** 修复后 keliya 4 case × N∈{1,2,4,8} mode A rivqdown.dat SHA256 全等 (确保 fix 不引入新非确定性)

### Requirement: 3 SHALL gate strict in strict-omp mode（P1e.4, PR-I + PR-J）

P1e 实施 SHALL 在 PR-I 阶段跑 server 8-cell（heihe + heihe_x4 × N∈{1,2,4,8}），并在 PR-J 阶段跑 Mac 4-cell（4 case × N=1），合计验证 3 SHALL gate 全 PASS in strict-omp mode (mode C build)。任一项 FAIL → P1e 不 closure（与 P1d D4 严格 hard gate 一致风格）。

#### Scenario: §4.4 A3a bitwise 跨 N (server SHALL hard gate)

- **WHEN** PR-I server 24-cell 实测 in mode C build (`make shud SHUD_ENABLE_OPENMP_RHS=1`, 2 case × 4 N × 3 reps)
- **THEN** heihe + heihe_x4 每 case 的 N∈{1,2,4,8} × 3 reps `output/<case>.out/<case>.rivqdown.dat` SHA256 SHALL 全等
- **AND** 双 case = 2 distinct SHA（heihe SHA, heihe_x4 SHA）

#### Scenario: §4.4b A3a Mac advisory cross-N (SHOULD, 不 block epic)

- **WHEN** PR-J Mac 4-case 实测 in mode C build
- **THEN** 4 Mac case (keliya / xinanjiang_upstream / qinyijiang / qhh) × N∈{1,2,4,8} `rivqdown.dat` SHA256 advisory bitwise — 任一 Mac case 跨 N FAIL → 入 `docs/p1e/p1e_mac_reverse_compat.md` advisory note（libomp 弱保证）；不 block epic verdict (per design D7 platform-aware scoping)

#### Scenario: §4.5 nst 跨 N (server only, heihe Δ=0 强制 + heihe_x4 |Δ| ≤ 2)

- **WHEN** PR-I 24-cell `output/<case>.out/cvode_stats.txt` 读取
- **THEN** heihe `nst` 跨 N∈{1,2,4,8} SHALL 全等（Δ=0 严格 hard gate）
- **AND** heihe_x4 `nst` 跨 N SHALL `|Δ_nst| ≤ 2` (per `openspec/specs/p1d-numa-governance/spec.md` nst ladder Requirement 复用，反映 mesh 加密后 SPGMR 收敛差异；非 softening)

#### Scenario: NUM_OPENMP=1 reverse-compat（6-case 矩阵）

- **WHEN** PR-I (server portion) + PR-J (Mac portion) 完成
- **THEN** 6 case（heihe + heihe_x4 server + 4 Mac case: keliya / xinanjiang_upstream / qinyijiang / qhh）× NUM_OPENMP=1 `output/<case>.out/<case>.rivqdown.dat` SHA256 SHALL byte-identical 至 `P1-update-omp-tag` canonical SHA
- **AND** server heihe N=1 SHA SHALL == `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`（PR-H 实测确认的 canonical，per `docs/p1d/p1d_pr_h_final_run.md` § "PR-H SHA matrix"）

#### Scenario: 任一 SHALL FAIL → P1e 不 closure

- **WHEN** PR-I 三 SHALL gate 任一项 FAIL
- **THEN** PR-I verdict SHALL 为 FAIL + epic issue 标 blocked + 不进入 PR-J/K/L/M (除非 §10 FAIL 流程触发 D12 fallback routing)
- **AND** 由用户决策 P1e 延展或 master plan 修订（不在本 change scope）

### Requirement: 加速比 acceptance per-case threshold at N=8

P1e 实施 SHALL 在 PR-I server 24-cell 验证 heihe + heihe_x4 × N=8 加速比 per-case threshold in strict-omp mode (mode C build)。两 case Amdahl 上界不同 → 不同 threshold（heihe RHS 占比较低 ~1.7× 上界 / heihe_x4 RHS 占 ~76% → 2.39× 上界）。

#### Scenario: 加速比 wall 测量协议

- **WHEN** PR-I server 24-cell 实测 in mode C build
- **THEN** 每 case 每 N SHALL 跑 **3 reps**（与 2×2 matrix 协议一致 per design D1）
- **AND** wall = **median(3 reps)** （不是 single rep / mean）— 消除 OS noise
- **AND** 3 reps SHA256 byte-identical 守门（确保 noise 来自 OS schedule 而非 binary 不一致）
- **AND** server 8-cell verification cell 数 = 2 case × 4 N × 3 reps = 24 cell

#### Scenario: heihe N=8 加速比验收 (threshold 1.3×)

- **WHEN** PR-I server 24-cell 实测 in mode C build
- **THEN** heihe N=8 wall median (s) SHALL ≤ heihe N=1 wall median (s) / 1.3
- **AND** 加速比 (= N=1 wall median / N=8 wall median) SHALL ≥ **1.3×** (Amdahl 上界 ~1.7×)

#### Scenario: heihe_x4 N=8 加速比验收 (threshold 1.5×)

- **WHEN** PR-I server 24-cell 实测 in mode C build
- **THEN** heihe_x4 N=8 wall median (s) SHALL ≤ heihe_x4 N=1 wall median (s) / 1.5
- **AND** 加速比 SHALL ≥ **1.5×** (Amdahl 上界 ~2.39×，per early profile RHS 66.55%)
- **AND** profile 重测 (per OQ3, PR-B 2×2 scaffolding 内顺带跑) 若发现 RHS 占比与 P1d era 数字差异 ≥10%，threshold 可基于实证调整（须更新本 spec + design D7）

#### Scenario: 加速比 fallback 触发 — 两 case 均 < threshold → D12.3 fallback Path 3

- **WHEN** PR-I heihe N=8 < 1.3× **AND** heihe_x4 N=8 < 1.5×
- **THEN** decision branch routing → D12.3 fallback ADR-0002 Path 3 → 触发 P1e.8 block-Jacobi precond 新 PR-N
- **AND** SHALL gate 跨 N bitwise + nst 仍 PASS 时，PR-N 闭合至两 case 均 ≥ 各自 threshold 后才进 PR-J/K/L/M
- **AND** 仅一 case 不达 threshold（另一 case 已达，典型：heihe < 1.3× 而 heihe_x4 ≥ 1.5×）→ partial closure 决策点（用户决策 ship vs fallback；倾向 ship 当 heihe_x4 ≥ 1.5× 时）

### Requirement: P1e.8 block-Jacobi precond fallback (conditional, PR-N)

P1e.8 PR-N **conditional 触发**条件：PR-I SHALL gate PASS + 两 case 加速比均 < 各自 threshold (per task 4.6 + D12.3)。PR-N 实施 ADR-0002 Path 3 (SPGMR + block-Jacobi physics precond)。

#### Scenario: precond 注册

- **WHEN** PR-N 实施后 `grep -nE 'CVodeSetPreconditioner.*p_setup.*p_solve' SHUD/src/Equations/cvode_config.cpp`
- **THEN** SHALL ≥1 命中（CVODE 注册 block-Jacobi precond setup/solve callback）

#### Scenario: iteration 下降验证

- **WHEN** PR-N 实施后跑 PR-I 同 8 cell
- **THEN** `cvode_stats.txt` nli SHALL 较 PR-I baseline 下降 ≥30%
- **AND** nfeLS SHALL 较 PR-I baseline 下降 ≥30%

#### Scenario: wall closure 验证

- **WHEN** PR-N 实施 + iteration 下降验证 PASS
- **THEN** heihe + heihe_x4 N=8 加速比 SHALL 闭合至各自 threshold (heihe ≥1.3× + heihe_x4 ≥1.5×)
- **AND** 闭合后才允许恢复 PR-J/K/L/M
- **AND** PR-N 数据归档 `docs/p1e/p1e_pr_n_block_jacobi.md`

### Requirement: baseline/P1e 分支 + P1e-tag (D11 7-tag chain)（P1e.5）

P1e epic capstone SHALL 创建 `baseline/P1e` 分支（从 `baseline/P1d` HEAD 分出）+ `P1e-tag` annotated tag + branch lock。

#### Scenario: baseline/P1e 分支创建

- **WHEN** P1e.0 epic intake 阶段
- **THEN** `gh api repos/DankerMu/SHUD-OpenMP/git/refs --method POST` 创建 `refs/heads/baseline/P1e` 指向 `baseline/P1d` HEAD (post-P1d PR-M 合并 SHA)
- **AND** PR-A..PR-M base = `baseline/P1e`

#### Scenario: P1e-tag annotated 创建

- **WHEN** PR-L 合并后立即 (post-merge action)
- **THEN** `git tag -a P1e-tag <baseline/P1e HEAD> -m '<message>' && git push origin P1e-tag`
- **AND** annotated message SHALL 含: P1d E′ containment closure → P1e F 路 strict-omp closure narrative + 2×2 build matrix 因果实验结论 (A/B/C/D mode 实测 + Phase 1/2 verdicts) + F 路实施细节 (abort 桩替换 + 单 parallel region + phase-based for + 复用 4 helpers) + 3 SHALL gate strict-omp mode verdict 实测 SHA + 加速比实测 (heihe + heihe_x4 × N=8 vs N=1 per-case threshold) + D11 6 → 7 tag chain immutability baseline + SHUD pin trail
- **AND** PR-M (post-PR-L-merge) SHALL amend `docs/p1e/p1e_summary.md` §"验证 P1e-tag" 章填实 tag-object SHA + deref commit SHA — P1e-tag deref commit 是 pre-PROMOTE HEAD（仿 P1d 模式，避免 PR-K 写后 PR-L 再 amend 循环）；§13 placeholder 在 PR-K 创建，PR-M 填实 SHA

#### Scenario: baseline/P1e branch lock

- **WHEN** PR-M post-merge action
- **THEN** `gh api repos/.../branches/baseline/P1e/protection --method PUT --field lock_branch=true --field enforce_admins=true --field allow_force_pushes=false --field allow_deletions=false`
- **AND** verify `gh api repos/.../branches/baseline/P1e --jq '.protection.lock_branch.enabled'` = true

#### Scenario: D11 historical 6-tag SHA re-verify (PR-L 内置守门)

- **WHEN** PR-L post-merge 创建 P1e-tag 之前
- **THEN** 6 historical tag SHA `git rev-parse <tag>` SHALL 全部不变 (B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag) — 与 P1e epic 启动时刻一致
- **AND** baseline/P1d lock 状态不动；baseline/P1c lock 状态不动
- **AND** D11 final 7-tag immutability verification (含 P1e-tag) 在 `p1e-capstone` spec Requirement "D11 immutability final verification (7 tag chain)" 内 PR-M post-merge 执行（避免与本 implementation spec 重复）

### Requirement: 三 negative grep gate 保留（P1e-wide）

P1e 实施 SHALL 保留 P1c / P1d era 已建立的 3 negative grep gate（per `openspec/specs/p1c-deterministic-reduction/spec.md` L76/L81/L86 + `openspec/specs/p1d-numa-governance/spec.md` 同 Requirement），即 P1e epic 期间任何 SHUD 改造 commit 不可引入：

- 新 macro pattern: `SHUD_USE_DETERMINISTIC_REDUCTION` / `SHUD_DET_REDUCT` / `SHUD_PAIRWISE`
- `schedule(dynamic|guided)` in `MD_rhs_core.cpp`
- `#pragma omp atomic` in `SHUD/src/`

#### Scenario: 三 negative grep gate post-P1e

- **WHEN** P1e epic capstone 完成时
- **THEN** 三 grep 命令均返回 0 hit (per P1c/P1d capstone established)
- **AND** PR-K capstone 文档显式 re-verify

---

