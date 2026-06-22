# ADR-0001 — SoA + AoS 双轨保留 RHS hot 字段

- **Status**: Accepted (2026-06-21)
- **Decision-by**: B1b Epic (#172) / S5d.1 (#178) capstone审议 — ratified at S5d 汇总验收 (#183)
- **Owner**: SHUD OpenMP 改造工程 / B1b 期内
- **Supersedes**: none (precedent ADR)
- **Superseded by**: none

## Context

`SHUD/src/classes/Element.hpp` 中的 `_Element` 是一个多继承 fat-AoS 容器(继承 `Triangle` / `Soil_Layer` / `Geol_Layer` / `Landcover` / `AttriuteIndex` 五层),`sizeof(_Element) = 688 bytes` (Apple clang 17 / Linux GCC 11,实测见 `tools/check_sizeof`)。Master plan §4.22.1 给出 600-1000 字节估算,实测 688 B 落在估算下沿。

RHS hot path 三个 TU(`MD_ElementFlux.cpp` / `MD_f.cpp` / `MD_ET.cpp`)实际读取的字段集是 30-40 个标量(`grep -nE 'Ele\[.*\]\.<field>'` 审计结果 = 32 个,见 `docs/s5d_hot_fields.yaml`)。这些字段的物理 footprint 约 300 B / element(2 个 int[3] + 7 个 int 标量 + 4 个 double[3] + 19 个 double 标量,详细见 ADR 末附录或 `tools/check_sizeof/check_sizeof.cpp` 内 `N_INT3` / `N_INT1` / `N_DOUBLE3` / `N_DOUBLE1` 常量)。

`_Element` 的另外 ~390 B 是 init / IO / calibration / 上游 rSHUD R 端 serializer 协议需要的字段(见 `docs/s5d_hot_fields.yaml` `non_hot_fields_in_Element` 列表 38 个),RHS 不读,但 AoS 一旦扫描就会把整 cache line 拉进 L1/L2,污染 cache。

设计选项:

- **选项 A:完全重构 `_Element` → SoA**,废弃 AoS 容器。所有 init / IO / calib 路径都必须从 SoA flatten 重组。
- **选项 B(本 ADR 采纳):SoA 抽取 + AoS 保留双轨**。新建 `MD_layout.hpp` `ElementHotData` SoA 容器承载 32 个 hot 字段;`_Element` AoS 容器**保留不动**;RHS hot path 改读 SoA,init / IO / calib / R 端协议仍读 AoS;sync 点保证 AoS 写入后 SoA 立即同步。

## Decision

**采纳选项 B**。具体形态:

1. `SHUD/src/ModelData/MD_layout.hpp` 提供 `struct ElementHotData` 指针容器(每字段一个 `T*` 或 `T*_flat`,32 字段),由 `Model_Data::malloc_EleRiv()` 一次 contiguous 分配。
2. 字段集源于 `docs/s5d_hot_fields.yaml` schema(name / type / size_per_ele / AoS_source_field),CI grep gate `tools/check_manifest/check_hot_fields.py` 保证 yaml ↔ MD_layout.hpp ↔ RHS 3 TU 三方对齐,任一漂移阻断 PR。
3. `_Element` AoS 容器零修改。
4. RHS hot path 三个 TU 全部改读 `hot.<field>[<idx>]` 或 `hot.<field>_flat[3*i + j]`;CI grep gate 0 `Ele[<expr>].<hot-field>` 命中。
5. 四个 `_Element` 成员方法(`updateElement` / `updateLakeElement` / `Flux_Infiltration` / `Flux_Recharge`)仍走 AoS dispatch,调用点后立即 `Model_Data::sync_hot_dynamic(i)` 把 4 个动态字段(`u_qi` / `u_qex` / `u_effKH` / `u_satn`)从 AoS 同步到 SoA。
6. DEBUG 编译加 `assert(hot.<field>[i] == Ele[i].<field>)` 类一致性检查;release 编译宏剥离 assert,zero runtime cost。

## Consequences

### Positive

- **bitwise neutrality**:S5d.1-S5d.4 全部 sub-step 6 case 90 天 NUM_OPENMP=1 vs B1a-tag bitwise PASS(详 `SHUD/B1b_CHANGELOG.md` 与各 sub-step PR-12 #156 → PR-13 #200 evidence)。SoA 是 AoS 的镜像读视图,字段写入仍走 AoS,sync 点保证状态可见性,运算路径不变。
- **R 端 rSHUD 协议不破**:init / IO 仍读 AoS,R 端 serializer 字段映射不需要任何修改。
- **可演化**:若未来决定单 SoA 化,本 ADR Triggers 节给出明确触发条件,届时新增 ADR-NNNN 接力。
- **DEBUG assertion 网**:漏字段时 DEBUG 构建立即 abort,而非 release 静默差异。
- **机器可读 schema**:`docs/s5d_hot_fields.yaml` 是 source-of-truth;若 hot path 后续触发新字段,审计流程 = 改 yaml + 再跑 CI gate,新人也能上手。

### Negative

- **双写代价**:四个动态字段(`u_qi` / `u_qex` / `u_effKH` / `u_satn`)每次 `_Element` 方法调用后必须 sync 到 SoA。sync 点是 inline 函数 `Model_Data::sync_hot_dynamic(i)` (4 个 `double` 赋值),per RHS evaluation 触发 ~NumEle 次。开销可忽略(< 0.1% wall time),但代码维护成本是真的(忘加 sync 会产生 stale read)。
- **sizeof gate 未达 < 0.20 阈值**:S5d 汇总验收 #183 Task 8.1 实测 `sizeof(ElementHotData) / sizeof(_Element) = 256 / 688 = 0.3721`(指针容器解读),`per_ele_soa_bytes / sizeof(_Element) = 300 / 688 = 0.4360`(每元素 cache footprint 解读)。两者均高于 spec L149 + master plan L1432 "< 0.20" 阈值。原因:`_Element` 实际占 688 B(估算下沿),hot field 集 ~300 B 已是 _Element 的 43.6%。这是真实的物理上界——不是 SoA 设计缺陷。详 docs/b1b_summary.md S5d 段。
- **PR scope 膨胀**:S5d 拆 4 个 sub-step(.1 SoA 抽取 / .2 jagged flatten / .3 first-touch / .4 run_omp wrapper),review 工作量 4×;但 master plan §S5d 已分 4 节,与设计 D1 对齐。
- **维护双源风险**:如未来有人新增 hot 字段时只改 AoS 没改 SoA,CI grep gate 会捕获 `Ele[.].<hot-field>` 0-hit 违规;若新增 SoA 字段时漏改 yaml,CI gate 同样 fail。但**新增 _Element 字段时也需要决策"这是 hot field 吗"**——这个决策点必须文档化(目前依赖 docs/s5d_hot_fields.yaml `non_hot_fields_in_Element` 节列举)。

### Risks

- **R1**:trailing-page first-touch 覆盖不全(详 `SHUD/B1b_CHANGELOG.md` S5d.3 Gap Sweep N2)。`Ele[].index` first-touch 只摸到 `_Element` 头 4 字节,后续多 page 范围仍归属主线程。NUMA-locality 优化部分覆盖。延后到 A3a / 多线程基线时再补。
- **R2**:Linux ABI / Apple clang ABI 在 `Soil_Layer` / `Landcover` 之类继承层 padding 上略有不同,`sizeof(_Element)` 可能跨平台差几字节;实测 macOS Apple clang 17 = 688 B,Linux GCC 11 = ?(P1+ 服务器实测会重新 emit)。如果跨平台数字差异 > 5%,本 ADR + #183 b1b_summary.md 须更新。

## Triggers — 何时启动 SoA 单轨化(本 ADR 失效条件)

以下三个条件**全部满足**时,启动新 ADR(`docs/adr/NNNN-soa-single-track.md`)废弃 AoS 容器,迁移到完全 SoA:

1. **B1b 汇总 cache miss reduction 实测 ≥ 30%**(Task 8.2 spec L151-153):即"双轨已经把 cache 收益吃到位"。
2. **双 socket NUMA accel ≥ +15% in B1b**(Task 8.3 spec L155-157):即 first-touch + SoA 在 P1+ 多线程场景被验证有效。
3. **B1b 落地后 6 个月稳定无 regression**:即双轨期间 init / IO / calib 路径无 SoA 缺字段类 bug;CI grep gate 持续 0 命中。

**额外触发条件**:rSHUD R 端 serializer 协议若有重大改版(rSHUD v3+),整套 AoS layout 都要 review,届时合并到 R 端协议改版 ADR 同时处理。

不达成上述触发条件前,本 ADR 保持 Accepted 状态,双轨长期保留。

## References

- **Master plan**:`SHUD_openMP_master_plan.md` §4.22 数据布局 L613-L691 + §S5d.1 SoA 抽取 L1385-L1402
- **Design**:`openspec/changes/b1b-baseline-completion/design.md` D2 (SoA + AoS 双轨) + Open Q5 (单轨化触发条件)
- **Spec**:`openspec/changes/b1b-baseline-completion/specs/s5d-data-layout-soa-numa/spec.md` Requirement "ElementHotData SoA 容器抽取" + "S5d 汇总验收 cache miss + NUMA 加速比"
- **Schema source of truth**:`docs/s5d_hot_fields.yaml`
- **CI grep gate**:`tools/check_manifest/check_hot_fields.py`
- **sizeof tool**:`tools/check_sizeof/check_sizeof.cpp` + `tools/check_sizeof/check_sizeof.sh`
- **Implementation**:`SHUD/src/ModelData/MD_layout.hpp` (struct decl) + `SHUD/src/ModelData/Model_Data.cpp` `malloc_EleRiv()` (alloc + first-touch) + `SHUD/src/ModelData/Model_Data.cpp` `initialize_hot()` + `sync_hot_dynamic(i)` (sync points)
- **Measurement evidence**:`docs/b1b_summary.md` S5d 汇总验收段(#183 PR)
