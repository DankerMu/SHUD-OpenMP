## Purpose

规约 S5d 数据布局（ElementHotData SoA + 平 3 行主序 + jagged flatten + parallel first-touch + NUMA env discipline）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/b1b-baseline-completion/specs/<capability>/spec.md PROMOTE 而来（#190 S6c-12c capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。

## Requirements

### Requirement: ElementHotData SoA 容器抽取 RHS hot 字段

系统 SHALL 新建 `SHUD/src/MD_layout.hpp` 提供 `ElementHotData` SoA 容器，承载 §4.22.1 列出的 RHS hot path 字段。最小字段集 = `int nabr_flat[NumEle*3]` / `double edge_flat[NumEle*3]` / `double area[NumEle]` / `double u_effKH[NumEle]`（其余按 §4.22.1 表完整覆盖，约 40 字段）。字段集 SHALL 从 `docs/b1b/s5d_hot_fields.yaml` 机器可读 schema（每字段：name + type + size_per_ele + AoS_source_field）派生，避免人肉对照漂移。`_Element` AoS 容器 SHALL 保留不动，用于 init / IO / calibration 非 hot path。RHS hot path 文件 `MD_ElementFlux.cpp` / `MD_f.cpp` / `MD_ET.cpp` SHALL 改读 SoA 容器。

#### Scenario: docs/b1b/s5d_hot_fields.yaml 是 ElementHotData 字段集 source-of-truth
- **WHEN** S5d.1 实施前检查 `docs/b1b/s5d_hot_fields.yaml` 与 master plan §4.22.1 RHS hot 字段表
- **THEN** yaml 字段集覆盖 §4.22.1 全部 hot 字段；ElementHotData 声明严格按 yaml 派生

#### Scenario: ElementHotData 字段覆盖 §4.22.1 表全部 RHS hot 字段
- **WHEN** 比对 `MD_layout.hpp` 中 ElementHotData 字段与 `docs/b1b/s5d_hot_fields.yaml` schema
- **THEN** SoA 覆盖率 = 100%（每个 yaml 条目在 SoA 中有对应数组）

#### Scenario: RHS hot path 改读 SoA
- **WHEN** S5d.1 完成 commit 上 grep `MD_ElementFlux.cpp` / `MD_f.cpp` / `MD_ET.cpp` 中 `Ele\[i\]\.nabr` / `Ele\[i\]\.edge` / `Ele\[i\]\.area` 等 hot 字段访问
- **THEN** 命中 0 行；hot path 全部走 `hot.nabr_flat[3*i+j]` / `hot.area[i]` 形式

#### Scenario: 非 hot path 不改动
- **WHEN** grep init / IO / calibration 路径对 `_Element` 字段的访问
- **THEN** 路径不变；`_Element` AoS 容器在非 hot path 仍可用

#### Scenario: DEBUG 一致性 assertion 通过
- **WHEN** DEBUG 编译并跑 keliya 90 天截断 NUM_OPENMP=1
- **THEN** RHS 入口抽样 assert `hot.area[i] == Ele[i].area` 全通过；release 编译时 assertion 被宏剥离

---

### Requirement: jagged 数组扁平化为一维 flat

系统 SHALL 把 `Model_Data.hpp` L121–L184 上的 `QeleSurf` / `QeleSub` 二维 jagged 数组改为一维 `double *QeleSurf_flat` / `double *QeleSub_flat`（大小 `NumEle*3`），并提供 inline 访问器 `QeleSurfAt(i,j)` / `QeleSubAt(i,j)`。`malloc_EleRiv()` 中嵌套 malloc/free SHALL 删除，改单次 contiguous 分配。`Ele[].iupdGW[3]` / `Ele[].iupdSF[3]` 频繁访问的 SHALL 并入 ElementHotData；`Riv[i]` / `RivSeg[i]` 内部数组按访问频率审计后并入 RiverHotData SoA 或保留。

#### Scenario: QeleSurf/QeleSub 改一维
- **WHEN** S5d.2 完成 commit 上 grep `QeleSurf` / `QeleSub` 类型声明
- **THEN** 仅存在一维 `double *_flat` 形式；二维 `double **` 形式 0 次出现

#### Scenario: 嵌套 malloc 全删
- **WHEN** grep `malloc_EleRiv` 中 `new double*[` / `new double[` 嵌套模式
- **THEN** 嵌套模式 0 次；单次 contiguous 分配 1 次

#### Scenario: inline 访问器全 hot 路径生效
- **WHEN** grep RHS 路径中 `QeleSurfAt` / `QeleSubAt` 使用
- **THEN** 所有 hot path 访问点使用访问器，无裸 `_flat[...]` 索引（保证可读 + 重命名安全）

#### Scenario: Riv/RivSeg 内部小数组审计落地
- **WHEN** S5d.2 完成 commit 上 grep `River.hpp` / `River.cpp` / `Model_Data.hpp` `RivSeg` 区段对 `double[N]` / `double*` 成员的声明
- **THEN** 每个被 hot path (`MD_f.cpp` river 分支 + `MD_ElementFlux.cpp` river 分支) 频繁访问 (≥ 每 RHS evaluation 一次) 的成员，要么并入 `RiverHotData` SoA（新建于 `MD_layout.hpp`），要么在 `docs/topology_manifest.yaml` `s5d2_riv_audit` 节标注"保留 AoS"+ 理由

---

### Requirement: S5d.2 必须通过 ASan + UBSan job

S5d.2 jagged → flat 改造涉及指针寻址重构（design R2 风险类）。S5d.2 完成 commit SHALL 通过 ASan + UBSan 双 sanitizer build 跑 6 case 90 天截断 NUM_OPENMP=1，不得有任一 sanitizer error 或 warning；任一 sanitizer 报告越界写 / use-after-free / undefined behavior SHALL 阻断 S5d.3 启动。

#### Scenario: ASan / UBSan 6 case 全 clean
- **WHEN** S5d.2 完成 commit 用 `-fsanitize=address,undefined -fno-omit-frame-pointer` 编译跑 6 case 90 天截断 NUM_OPENMP=1（kashigeer N/A）
- **THEN** ASan 报告 0 errors 0 leaks；UBSan 报告 0 errors 0 warnings；run log 落 `<run_dir>/sanitizer_report.txt`

#### Scenario: CI workflow 加 sanitizer 临时 axis
- **WHEN** 检查 `.github/workflows/serial-baseline.yml` S5d.2 落地 PR 上
- **THEN** matrix 含 `sanitizer=asan-ubsan` 临时 axis，跑 keliya + qhh 2 case 90 天截断；S5d 全部 sub-step 完成 / S6b 启动后该 axis 可移除

---

### Requirement: parallel first-touch 初始化必须发生在线程绑定之后

系统 SHALL 在 `malloc_EleRiv()` 中对每个 SoA 数组完成分配后立刻执行 `#pragma omp parallel for schedule(static) for (i=0; i<NumEle; ++i) for (j=0; j<3; ++j) arr[3*i+j] = 0.0;` 形式的并行初始化。`_Element*` 大对象 placement-new 后 SHALL 同样 parallel touch 一次。`LoadIC()` 完成后 SHALL 追加一次额外 parallel touch 把 IC 内存归属转移到将来处理的线程。所有 parallel touch SHALL 在 `OMP_PROC_BIND` 已设置后执行，否则 NUMA 归属错乱。

`shud.cpp` 启动期 SHALL emit 确定性 log token：(a) `[NUMA] OMP_PROC_BIND=<val>` 在 `getenv` 检查后立即输出（含缺失场景 `[NUMA] OMP_PROC_BIND=unset`）；(b) `[NUMA] first-touch begin tag=<arr_name>` 在每个 first-touch 调用点之前。两类 token 用于 grep 顺序断言。

#### Scenario: 三处 first-touch 全部命中
- **WHEN** grep `malloc_EleRiv` / `LoadIC` 区段中 `#pragma omp parallel for` 出现
- **THEN** 至少 3 处：SoA 数组分配后 + `_Element` placement-new 后 + LoadIC 收尾后

#### Scenario: log token 顺序断言 OMP_PROC_BIND 在 first-touch 之前
- **WHEN** 跑 keliya 90 天截断（任意配置）输出 stderr/log 到 `<run_dir>/run.log`
- **THEN** `grep -n '\[NUMA\] OMP_PROC_BIND=' <run_dir>/run.log` 返回的最小行号 < `grep -n '\[NUMA\] first-touch begin' <run_dir>/run.log` 返回的最小行号

#### Scenario: OMP_PROC_BIND 缺失时不做 first-touch 优化
- **WHEN** 不设 `OMP_PROC_BIND` 跑 keliya
- **THEN** log 含 `[NUMA] OMP_PROC_BIND=unset` + warning + 跳过 first-touch 阶段（按 design R3 mitigation #2）

#### Scenario: 单线程 first-touch 与 B1a bitwise 一致
- **WHEN** S5d.3 完成 commit 上跑 6 case 90 天截断 NUM_OPENMP=1（kashigeer N/A）
- **THEN** SHA256 全 PASS vs B1a-tag（parallel touch 写入的是 init 值，没改运算）

---

### Requirement: 线程绑定 run script 与 manifest 字段必填

系统 SHALL 新建 `tools/run_omp.sh` 包装 SHUD 二进制调用：(a) export `OMP_PROC_BIND=close` `OMP_PLACES=cores` `OMP_NUM_THREADS=<N>`；(b) 然后调 `./shud <project_path>`；(c) 启动时打印线程绑定状态到 stderr。`shud.cpp` 初始化段 SHALL 检查 `getenv("OMP_PROC_BIND")`，缺失时输出 warning（不强制覆盖）。`benchmarks/<case>/manifest.yaml` 每个 case SHALL 加必填字段 `omp_env: { OMP_PROC_BIND: close, OMP_PLACES: cores }`。

#### Scenario: run_omp.sh 提供完整 OMP 环境
- **WHEN** 检查 `tools/run_omp.sh` 内容
- **THEN** 至少包含 3 个 `export OMP_*` + 1 个 `./shud` 调用 + 1 个 stderr 状态打印

#### Scenario: shud.cpp warning 路径生效
- **WHEN** 不设 `OMP_PROC_BIND` 跑 SHUD（任意 case）
- **THEN** stderr 输出 warning 行 `OMP_PROC_BIND not set, NUMA first-touch may be ineffective`

#### Scenario: 7 case manifest 全有 omp_env 字段
- **WHEN** 检查 `benchmarks/keliya/manifest.yaml` 等 7 case manifest
- **THEN** 每个 manifest 有 `omp_env.OMP_PROC_BIND` 与 `omp_env.OMP_PLACES` 字段

---

### Requirement: NUMA 探测工具与 run log 落盘

系统 SHALL 新建 `tools/numa_check.sh` 在每次 P1+ benchmark run 启动期调用 `numactl --hardware`，把输出存入 `<run_dir>/numa_topo.log`，并提取 socket 数 / node 数到 run summary。多 socket 机器若未启用 `OMP_PROC_BIND` 或未做 first-touch，summary SHALL 标 `numa_first_touch: WARNING`。

#### Scenario: numa_check.sh 输出包含硬件拓扑
- **WHEN** 在任一 CPU 分区双 socket Xeon idle node (cn05-06,09,14-19,23-24 中任意) 上调 `tools/numa_check.sh`
- **THEN** `numa_topo.log` 包含 `available: 2 nodes` 类似行；run summary `socket_count: 2`

#### Scenario: 本地 Mac 单 socket UMA 跳过 NUMA 验收
- **WHEN** 在 Apple Silicon Mac 上跑 `tools/numa_check.sh`
- **THEN** summary `socket_count: 1`，NUMA 加速比验收一律标 N/A (single-socket-UMA)

---

### Requirement: S5d 四 sub-step 每步独立 bitwise == B1a

S5d.1 / S5d.2 / S5d.3 / S5d.4 SHALL 每完成一步立即跑 6 case bitwise vs B1a-tag (NUM_OPENMP=1, 90 天截断)；任一 sub-step 失败 SHALL 阻断后续 sub-step 启动。

#### Scenario: S5d.1 独立 bitwise 通过
- **WHEN** S5d.1 完成 commit 上跑 6 case 90 天截断 NUM_OPENMP=1 bitwise (kashigeer N/A)
- **THEN** 全 PASS；进入 S5d.2 启动条件满足

#### Scenario: S5d.2 独立 bitwise 通过 + ASan/UBSan clean
- **WHEN** S5d.2 完成 commit 上跑 6 case 90 天截断 NUM_OPENMP=1 bitwise (kashigeer N/A)，同时 ASan + UBSan 6 case 同样跑通
- **THEN** bitwise 全 PASS + sanitizer 0 errors；进入 S5d.3 启动条件满足

#### Scenario: S5d.3 独立 bitwise 通过
- **WHEN** S5d.3 完成 commit 上跑 6 case 90 天截断 NUM_OPENMP=1 bitwise (kashigeer N/A)
- **THEN** 全 PASS；进入 S5d.4 启动条件满足

#### Scenario: S5d.4 独立 bitwise 通过
- **WHEN** S5d.4 完成 commit 上跑 6 case 90 天截断 NUM_OPENMP=1 bitwise (kashigeer N/A)
- **THEN** 全 PASS；S5d 进入汇总验收

---

### Requirement: S5d 汇总验收 cache miss + NUMA 加速比

S5d 汇总验收 SHALL 同时满足：(a) `sizeof(ElementHotData) / sizeof(_Element) < 0.20`（per-element 字节占比；master plan §S5d L1432 "显著小于" 的可校验形式）；(b) 单线程 perf stat 测 L1/L2 cache miss 率较 B1a 下降 ≥ 30%（双 socket idle node + 单 socket Apple Silicon 各测一组）；(c) 双 socket idle node 上 8 线程加速比 first-touch ON vs OFF ≥ +15%（NUMA-bound 案例；**该验证项可标 IN-PROGRESS 不阻断 S6b 启动**——design R4 mitigation #2）；(d) 单 socket UMA 机器 NUMA 加速比验收标 N/A 不计 fail。

#### Scenario: ElementHotData 字节占比
- **WHEN** 编译 S5d 完成版并输出 `sizeof(ElementHotData)` + `sizeof(_Element)`
- **THEN** `sizeof(ElementHotData) / sizeof(_Element) < 0.20`

#### Scenario: cache miss 单线程下降 ≥ 30%
- **WHEN** 在任一双 socket Xeon idle node (cn05-06,09,14-19,23-24) 与 Apple Silicon 上分别 `perf stat -e LLC-load-misses,LLC-loads,L1-dcache-load-misses,L1-dcache-loads` 跑 heihe 90 天截断 NUM_OPENMP=1
- **THEN** 两机各自相对 B1a 下降 ≥ 30%；任一机不达标记 IN-PROGRESS 不阻断 S6b 启动

#### Scenario: 双 socket NUMA 加速比 ≥ +15%（可 IN-PROGRESS）
- **WHEN** 在任一双 socket Xeon idle node 上跑 heihe 90 天截断 NUM_OPENMP=8 first-touch ON 与 OFF 各 3 run；sbatch 从 `/scratch` 提交，`--output/--error` 在 `/scratch`，遵守 Slurm 三铁律
- **THEN** ON wall-clock 中位数 / OFF wall-clock 中位数 ≤ 0.87；未达标可标 IN-PROGRESS 不阻断 S6b 启动（节点排队 / 维护是可接受的延迟原因）

#### Scenario: 单 socket UMA 跳过 NUMA 验收
- **WHEN** 在 Apple Silicon 上跑同 first-touch ON/OFF 对照
- **THEN** 验收记 N/A (single-socket-UMA)；不计入 fail
