# P1 update-omp Baseline 完成

> P1 = master plan §3 定义的 "B1b + MD_update.cpp 三 owner loop 并行后的 first parallel candidate baseline"。**stage = 工作阶段（P1 epic 13 PR + 1 capstone tag-lock）**；**baseline = 工作产物（B0/B1a/B1b/B1/P1）**；P1 不是单一 sub-stage 的产物，而是 **M7 forcing trim + Opt-IO 决策 + P1.0 pre-audit + element/river/lake 三 pragma 落地 + RHS snapshot + full-run + scaling 6 个 phase** 全部完成后才能签字的检查点。Phase D 是 lock + capstone 阶段。
>
> P1 epic = #211（D9 fast-path may trigger if P2 sub-stages stack as "P1-update-omp-tag extension"；P7 capstone 创建 separate `P-strict-tag`/`baseline/P-strict` 是 reserved Open Question #5）。13 PR (#212/#213/#214/#215/#216/#217/#218/#219/#220/#221/#222/#223/#224) + tag-lock PR-L #224 已 merge 进 `main`；PR-M (#225 本 PR) = P1 docs capstone（status_matrix + build_manifest + 本 summary）。

## 完成定义

| master plan P1 契约（§3 + §S5d → P1 衔接 + p1-state-update-parallel spec） | 实际做的 |
|---|---|
| Phase A — M7 forcing trim（Mac 4 case + server heihe/heihe_x4 trim + 7-case manifest schema 升级） | #212 PR-A + #213 PR-B |
| Phase B — Profile retest + Opt-IO 决策（heihe trimmed `t_forcing_io/t_total < 5%`） | #214 PR-G（trimmed heihe 1.90% / heihe_x4 0.19%；Opt-IO 改判 (a) 退回可选） |
| Phase C.audit — `_Element::updateElement` / `_River::updateRiver` / `_Lake::update` + f_updatei case 1-5 + `f_update` 三 loop 5 函数 pre-audit | #215 PR-C（design D9 path (a) safe verdict 全 PASS，零源码改动） |
| Phase C.implement — MD_update.cpp 三 owner loop `#pragma omp parallel for default(none)` | #216 PR-D element loop + #217 PR-E river loop + #218 PR-F lake loop（SHUD pin trail 017c629 → 6a9e684 → 08898a3 → 07c677f） |
| Phase C.verify.snapshot — Mac 4-case RHS snapshot bitwise vs B1b/B1-tag NUM_OPENMP=1 | #219 PR-H（12/12 PASS） |
| Phase C.verify.fullrun.mac — Mac 4-case canonical summary SHA + CVODE 15-key vs B1b/B1-tag NUM_OPENMP=1 | #220 PR-I（8/8 PASS = 4 G1 self-det + 4 G2 vs B1b） |
| Phase C.verify.fullrun.server — server heihe + heihe_x4 canonical summary SHA vs B1b/B1-tag NUM_OPENMP=1 | #221 PR-J（4/4 PASS = 2 G1 + 2 G2，jobid 8794/8795 cn03） |
| Phase C.verify.scaling.mac — Mac 4-case × NUM_OPENMP {1,2,4,8} A3a/A3b 与 §1.1.1 T 列对比（NG1 dev-only） | #222 PR-K1（16/16 A3a PASS bitwise；不计入 §1.1.1 go/no-go per NG1） |
| Phase C.verify.scaling.server — server 2-case × NUM_OPENMP {1,2,4,8} A3a/A3b + §1.1.1 验收 | #223 PR-K2（jobid 8796/8797 cn03；§1.1.1 WARNING 不阻塞 per design D5 NG3） |
| Phase D.tag — `P1-update-omp-tag` annotated + `baseline/P1` 分支 lock_branch=true | #224 PR-L（tag `ff21c75c` deref `003f58d`） |
| Phase D.docs — `docs/p1_summary.md` + status_matrix P1 行 + build_manifest P1 entry | #225 PR-M（本 PR） |
| Phase D.PROMOTE — 4 capability spec PROMOTE → `openspec/specs/*` + archive change + glossary + jsonl 双追加 | #226 PR-N（next） |

## `P1-update-omp-tag` 的处理

- `P1-update-omp-tag` = `ff21c75c8e968d5e47ca53b015425360be9ac879`（annotated）deref `003f58dc079116ef2161d2f96006228ef0e013d0` / SHUD pin `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（`openmp-baseline` 分支 HEAD on `SHUD-System/SHUD`）。
- annotated tag object SHA = `ff21c75c…`；deref commit SHA = `003f58d…`（= PR-K2 #223 capstone log append commit 时刻）。
- **D11 强制：一次锁死禁止 force-update**（与 B1a-tag force-update 历史**不同**，与 B1b-tag 一致）。任何后续 retroactive 更新（如 P2 sub-stage stacking）走 forward-compat **P1c-tag stacking / P2-* stacking** 路径（master plan C8）。
- `baseline/P1` 分支 protection 已 lock：`lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false`。
- `P1-update-omp-tag` 指向的 commit (`003f58d`) 即 PR-K2 #223 squash-merge + #223 post-merge log append 之后的 HEAD；docs PROMOTE（本 #225 PR + 后续 #226 PR-N）在 main 侧推进，**不进 `P1-update-omp-tag` 内部**——P1 tag 凝固在 evidence + log append 状态。

## P1 完成时间线

13 PR + 1 capstone tag-lock + 1 docs capstone PR：

- PR-A #212 [Phase A / I-A.1] — M7 forcing trim 工具 + Mac 4 case trim + bitwise vs B0-tag + 7-case manifest forcing_dir schema 升级
- PR-B #213 [Phase A / I-A.2] — server heihe + heihe_x4 trim + Slurm verify + bitwise vs B0-tag golden
- PR-G #214 [Phase B / I-B.1] — profile retest M7 trim + Opt-IO 决策 (a) 退回可选（heihe 1.90% / heihe_x4 0.19% << 5% 严格门）
- PR-C #215 [Phase C.audit / I-C.1] — P1.0 pre-audit 5 函数 + f_updatei case 1-5 全 (a) safe（reviewer-only PR，零 SHUD 源码改动；design D9 path (a) 决策）
- PR-D #216 [Phase C.implement / I-C.2] — MD_update.cpp element loop L64-L105 `#pragma omp parallel for schedule(static) default(none)`（SHUD `017c629` → `6a9e684`）
- PR-E #217 [Phase C.implement / I-C.3] — MD_update.cpp river loop L107-L125 同模式 pragma（SHUD `6a9e684` → `08898a3`）
- PR-F #218 [Phase C.implement / I-C.4] — MD_update.cpp lake loop L136-L147 同模式 pragma（SHUD `08898a3` → `07c677f`，3-pragma stack 完整）
- PR-H #219 [Phase C.verify.snapshot / I-C.5] — Mac 4-case RHS snapshot bitwise vs B1b/B1-tag canonical golden（12/12 PASS）
- PR-I #220 [Phase C.verify.fullrun.mac / I-C.6] — Mac 4-case canonical summary SHA + CVODE 15-key vs B1b/B1-tag（8/8 PASS）
- PR-J #221 [Phase C.verify.fullrun.server / I-C.7] — server heihe + heihe_x4 canonical summary SHA vs B1b/B1-tag（4/4 PASS，Slurm jobid 8794/8795 cn03）
- PR-K1 #222 [Phase C.verify.scaling.mac / I-C.8a] — Mac 4-case × N {1,2,4,8} = 16-cell A3a PASS bitwise（NG1 dev-only，不计入 §1.1.1）
- PR-K2 #223 [Phase C.verify.scaling.server / I-C.8b] — server 2-case × N {1,2,4,8} = 8-cell wall + 6-cell A3a/A3b（§1.1.1 WARNING per design D5 NG3）
- PR-L #224 [Phase D.tag / I-D.1] — `P1-update-omp-tag` annotated 创建 + `baseline/P1` 分支 lock
- PR-M #225 [Phase D.docs / I-D.2，本 PR] — `docs/p1_summary.md` + status_matrix P1 行 + build_manifest P1 entry
- PR-N #226 [Phase D.PROMOTE / I-D.3，next] — 4 capability spec PROMOTE + archive + glossary + jsonl 双追加 + close Epic #211

## SHUD pin trail（PR-D / PR-E / PR-F 3-pragma stack 落地序列）

| Stage | Outer PR | SHUD pin from → to | Change scope |
|---|---|---|---|
| Phase C.audit pre-source | (PR-C #215) | `017c629` (unchanged) | reviewer-only audit, 零源码改动 |
| PR-D #216 element pragma | element loop | `017c629` → `6a9e684` | `MD_update.cpp` L64-L105 `#pragma omp parallel for schedule(static) default(none) shared(…) private(i)` |
| PR-E #217 river pragma | river loop | `6a9e684` → `08898a3` | `MD_update.cpp` L107-L125 同模式 pragma |
| PR-F #218 lake pragma | lake loop | `08898a3` → `07c677f` | `MD_update.cpp` L136-L147 同模式 pragma（3-pragma stack 完整）|

P1-update-omp-tag 时刻的 SHUD pin = `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（`openmp-baseline` branch HEAD，PR-F merge 后 stable through PR-H/I/J/K1/K2 验证阶段 + tag-lock）。

## 3-pragma stack 详情

`SHUD/src/ModelData/MD_update.cpp` 的 `Model_Data::f_update(Y, DY, t)` 三 owner loop：

| Loop | 行号 | 写集（owner-local，per loop iter `i`） | NumIter |
|---|---|---|---|
| element loop | L64-L105 | `QeleSubAt(i,j)` / `QeleSurfAt(i,j)` j=0..2 (flat[3i+j]) + `QeleSubTot[i]` / `QeleSurfTot[i]` + `uYsf[i]` / `uYus[i]` / `uYgw[i]` + `Ele[i].QBC` / `Ele[i].yBC` + `qEleExfil[i]` / `qEleInfil[i]` | NumEle |
| river loop | L107-L125 | `uYriv[i]` + `Riv[i].updateRiver(uYriv[i])` 内部 6 个 `this->u_*` + `Riv[i].qBC` / `Riv[i].yBC` | NumRiv |
| lake loop | L136-L147 | `yLakeStg[i]` + `lake[i].yStage` + `lake[i].update()` 内部 `this->u_toparea` + `y2LakeArea[i]` + 6 个 lake flux 清零（QLakeSub/Surf/qLakeEvap/qLakePrcp/QLakeRivIn/Out）| NumLake |

3 个 loop 均：(a) 写目标按 `i` 索引（disjoint slots，无 false sharing per Macros.hpp `iSF`/`iUS`/`iGW`/`iRIV`/`iLAKE` `i + k*NumEle`-style offset）；(b) 无 cross-iter 依赖（无 `Ele[i+k]` / `Riv[neighbor]` / `lake[j!=i]` 读）；(c) 无 reduction / 无 I/O / 无全局可变状态（BC 通过 `tsd_*.getX()` 纯读，S5a 已 audit thread-safe per `_TimeSeriesData::getX` 设计）；详 PR-C audit `docs/p1_audit_update_funcs.md` §2-§5 + design D9 path (a)。

**未触 loop（本 change 不并行 per spec NG）**：
- L127-L131 river-clear loop（NumRiv 闲置 reset，pragma 收益 << 启动开销）
- L132-L135 element-clear loop（NumEle 闲置 reset）
- L148-L150 DY-zero loop（NumY-bounded，跨 element/river/lake 拼 NumY 不对齐 owner-local 索引模式）
- L6-L62 `f_updatei` case 1-5 五 owner loop（NG7：本 change 仅并行 `f_update`；`f_updatei` 留 P2a reviewer 参考）

## Validation evidence（3 anchor + 16-cell scaling + WARNING）

### Anchor 1：Mac 4-case RHS snapshot bitwise（PR-H #219）

`shud_omp @ NUM_OPENMP=1` + `SHUD_DUMP_RHS=1` + `SHUD_DUMP_SITE=f_update`，4 case × 3 t-anchor (1d/30d/90d) = 12-cell vs `benchmarks/<case>/B0_output/snapshot_t<rel_sec>.bin` (= B0 ≡ B1a ≡ B1b ≡ B1-tag canonical golden by tag-chain identity)：

| Case | t=86400 (1d) | t=2592000 (30d) | t=7776000 (90d) | wall (NUM_OPENMP=1, 90d) |
|---|---|---|---|---|
| keliya (484) | PASS | PASS | PASS | 60.8 s |
| xinanjiang_upstream (801) | PASS | PASS | PASS | 7.3 s |
| qinyijiang (3155) | PASS | PASS | PASS | 238.1 s |
| qhh (4773 w/ lake) | PASS | PASS | PASS | 66.5 s |

**12 / 12 PASS** = `compare_snapshot --quiet` exit 0 across all cells（file header + record header + 全 `DY` array 长度 NumY bytes 全部 byte-equal）。详 [`docs/p1_rhs_snapshot_bitwise.md`](p1_rhs_snapshot_bitwise.md)。

### Anchor 2：Mac 4-case full-run canonical summary SHA + CVODE 15-key（PR-I #220）

`shud` serial (无 `-fopenmp` link，3-pragma 在 serial build 内 no-op) + `tools/archive_b0_output.sh <case> 3`，4 case × 2 gate (G1 self-determinism / G2 vs B1b golden)：

| Case | new canonical SHA | golden sha256_run1 | G1 | G2 |
|---|---|---|---|---|
| keliya | `a27e3fb51eb72e1955ff2f429889d009f20803a6e1135bfde866fe4706549e3d` | `a27e3fb5…` | PASS | PASS |
| xinanjiang_upstream | `fe6dd4edc94c9581f382d1c732c28c7cc56dda857793b70ed8b989fea1fef394` | `fe6dd4ed…` | PASS | PASS |
| qinyijiang | `383e4099d6f71acfa31b8006fab946cf05c255c6dedae7de24273f90b322b174` | `383e4099…` | PASS | PASS |
| qhh | `3a86e24c1b6a3a0cf71300c1e32cd9013e69e9effd1c543c285ac714d2cf2c9e` | `3a86e24c…` | PASS | PASS |

**8 / 8 PASS**（4 G1 + 4 G2）+ CVODE 15-key 4/4 exit 0（`tools/cvode_stats_diff/cvode_stats_diff.sh`，全 15 个 canonical CVODE keys `nfe/nfeLS/nni/nli/nsetups/netf/nst/npe/nps/ncfn/ncfl/lenrw/leniw/lenrwLS/leniwLS` byte-identical vs B1b golden）。详 [`docs/p1_fullrun_bitwise.md`](p1_fullrun_bitwise.md) §3-§6。

### Anchor 3：server 2-case full-run canonical summary SHA（PR-J #221）

cn03 Slurm（三铁律：`/scratch` 下 `sbatch` + `/scratch` output 路径 + 全引用脚本 in `/scratch`）：

| Case | NumEle | new canonical sha256_run1 | golden sha256_run1 | G1 | G2 | jobid | Elapsed | node |
|---|---|---|---|---|---|---|---|---|
| heihe | 6335 | `675c927c9f7195166a0ea10cfa246173978ca40c608860e8f0a9065b95ba8a67` | `675c927c…` | PASS | PASS | 8794 | 00:27:18 | cn03 |
| heihe_x4 | ~25000 | `3fbcbd5c0c572c8877013e3eb519f68add2281f60ea329834c8473efea646c06` | `3fbcbd5c…` | PASS | PASS | 8795 | 01:08:45 | cn03 |

**4 / 4 PASS**（2 G1 + 2 G2）。Server binary `SHUD/shud` (serial, 无 `-fopenmp` link) sha256 = `3e9e56295528b0399aff928d1b44d708da87b37777ea81e0de216a3d12a975f3` on cn03（GCC 13.3.0-6ubuntu2~24.04.1，strict FP 3-grep gate PASS：`-O2/-ffp-contract=off/-fno-fast-math ≥ 1 hit each` + `-ffast-math/-Ofast/-funsafe-math-optimizations` 0 hit）。详 [`docs/p1_fullrun_bitwise.md`](p1_fullrun_bitwise.md) §"Server section"。

### Mac 16-cell scaling（PR-K1 #222，NG1 dev-only）

Mac 4 case × NUM_OPENMP ∈ {1,2,4,8} = 16-cell wall + speedup + A3a 与 `benchmarks/<case>/B0_output/snapshot_t7776000.bin` (= B1b-tag canonical) 比对：

**Aggregate: 16 / 16 A3a PASS · 0 / 16 A3b fallback · 0 FAIL**（`max_ulp = 0` across 全 16 cells；3-pragma stack 在 owner-local 写入模式下完全 OMP-permutation-invariant，design D5 "ideal" lane）。

Mac speedup 表现 sub-linear / anti-scale（M4 Pro 4P+10E 混 core 类型 + libomp on darwin 不可靠 `OMP_PROC_BIND`，per design D5 "small case anti-scale" 解释 + NG1 不计入 §1.1.1）。详 [`docs/p1_perf_baseline.md`](p1_perf_baseline.md) §1。

### Server 8-cell wall + 6-cell A3a/A3b（PR-K2 #223，§1.1.1 acceptance）

cn03 Slurm `--cpus-per-task=8` `OMP_PROC_BIND=close OMP_PLACES=cores`，2 case × N {1,2,4,8} = 8-cell wall + 6-cell A3a/A3b（N=2/4/8 vs N=1 same-binary baseline）：

| Case | sp@2 | sp@4 | sp@8 | N=2 A3a | N=4 A3a/A3b | N=8 A3a/A3b |
|---|---:|---:|---:|---|---|---|
| heihe (6335 / NumY 21357) | 0.96× | 1.05× | **1.08×** | PASS bitwise | FAIL / FAIL | FAIL / FAIL |
| heihe_x4 (40046 / NumY 124395) | 1.01× | 1.08× | **1.14×** | PASS bitwise | FAIL / FAIL | FAIL / FAIL |

**N=2 cells**: 2 / 2 A3a PASS bitwise（强 PR-D/E/F design 证据：低线程数 + owner-local pragma OMP-permutation-invariant 维持）；**N=4 / N=8 cells**: 4 / 4 A3a + A3b double-FAIL with trajectory bifurcation（heihe nst per N: 6773 / 6773 / 6585 / 6684，N≥4 OMP scheduling reorders 隐含 reduction-tree-depth transition → CVODE re-selects step size against different RHS sample → trajectory bifurcates；详 PR-K2 §2.6 + 下文 §"§1.1.1 verdict"）。

详 [`docs/p1_perf_baseline.md`](p1_perf_baseline.md) §2 + server binary `SHUD/shud_omp` sha256 = `b637537c53ff446b9885f949c19f20e50eba53296ef417ea5a5924fa803b2865` (GCC 13.3.0)。

## P1 forced gate verification（NUM_OPENMP=1 vs B1b/B1-tag bitwise）

Per spec L131-L134 + master plan §2.2 A0 / A1：3 独立 anchor 在 NUM_OPENMP=1 下全 PASS：

| Anchor | source | scope | result |
|---|---|---|---|
| 1 | PR-H #219 (`docs/p1_rhs_snapshot_bitwise.md`) | Mac 4-case `shud_omp` RHS snapshot mid-state vs B1b canonical | 12 / 12 PASS bitwise |
| 2 | PR-I #220 (`docs/p1_fullrun_bitwise.md` §1-§6) | Mac 4-case `shud` serial full-run canonical SHA + CVODE 15-key vs B1b | 8 / 8 PASS + 4/4 CVODE |
| 3 | PR-J #221 (`docs/p1_fullrun_bitwise.md` §"Server section") | server 2-case `shud` serial full-run canonical SHA vs B1b | 4 / 4 PASS |

Cross-binary equivalence（`shud_omp @ N=1` ≠ `shud` serial canonical SHA but RHS snapshot ≡ B1b）= 已 PR-I §"OMP-runtime sub-note" 显式 disclosed：B0/B1a/B1b golden 由 serial binary 归档；`shud_omp` link `-fopenmp` runtime 即使 N=1 也激活 `omp parallel for` 团队（size=1）+ 不同 reduction tree / scheduler / FMA selection；master plan §2.2 A0/A1 gate 比 serial-vs-serial（PR-I/PR-J），OMP-binary full-run bitwise 是 A3a gate（P7 strict）且要求 same-thread (NUM_OPENMP=N vs NUM_OPENMP=N)，不是 N=1(OMP) vs serial。`shud_omp @ N=1` 对 B1b 的中段比对走 RHS snapshot 层（PR-H 12/12 PASS = A2 precision gate per master plan §2.2）。

## §1.1.1 verdict — WARNING（P1 epic 不阻塞 per design D5 NG3 + master plan §6 P7 final-fusion 债）

**WARNING 双根因**：

### 根因 1：wall / speedup 低于 P7 strict M

| Case | scale | P7 strict 8-core M / T | P9 final 8-core M / T | 实测 N=8 sp | gap 解读 |
|---|---|---|---|---:|---|
| heihe | Medium (IO-mitigated) | "不独立验收" (master plan §1.1.1 + §5 Opt-IO 因 trim-after-M7 退回可选) | 3.0× / 4.5× | 1.08× | small-case fork-join 开销 + Amdahl-bound serial fraction（B1b 起点 P5/P3 owner-local gather 仍 serial）|
| heihe_x4 | Large | 1.8× / 2.2× | 4.5× / 6.0× | 1.14× | first OMP candidate（no S5d SoA / no owner-local gather parallel / no OMP_CUTOFF / no N_Vector parallel）+ P7 final-fusion 待引入 |

**heihe Amdahl-bound 1.13×**（理论上限 = 1 / (s + (1-s)/N)，s ≈ 0.88 fork-join + sequential overhead）+ "不独立验收" carve-out per master plan §1.1.1 → heihe 不阻塞。**heihe_x4 1.14×**（P7 strict M=1.8× 退出门 / T=2.2× 严格 timing）是 P1 起点报告，预期 P2+/P7 闭合。

### 根因 2：A3a / A3b strict 在 N ≥ 4 dual-FAIL

CVODE nst bifurcation：heihe nst per N: 6773 (N=1) / 6773 (N=2) / **6585 (N=4)** / **6684 (N=8)**；heihe_x4 nst per N: 6571 (N=1) / 6571 (N=2) / **6570 (N=4)** / **6572 (N=8)**。

Root cause hypothesis（PR-K2 §2.6 framing）：B1b S2 P3-P5 owner-local gather（`rhs_deterministic_gather()`，PR-11 #155 时刻退役 PassValue → 改 tree-reduction-based gather over NumEle）在 N > 2 transition 触发 reduction-tree-depth 跃迁（log2(N) 树深从 depth-1 (N=2) → depth-2 (N=4) → depth-3 (N=8)），每层结合律不满足 (a+b)+c ≠ a+(b+c) 在 FP 下 → tree reorder 产生 ULP-level RHS sample → CVODE step-size adaptation 重选 → trajectory 在 90-day terminus 漂到 `max_abs ≈ 5e5`（chaotic ODE long-integration regime）。

P7 final-fusion deterministic-reduction debt（master plan §6 P7 scope）= 修复路径：(i) tree-reduction 改 fixed-shape (e.g., pairwise canonical order over fixed adjacency list)；或 (ii) P9 deterministic N_Vector（OMP-thread-invariant reduction）。

**P1 不阻塞**：design D5 NG3 + master plan §1.1.2 + spec p1-state-update-parallel L164-L181 allow A3b-fallback / WARNING at **P1 standalone**；强 anchor = NUM_OPENMP=1 vs B1b 3-anchor 全 PASS（上节）+ N=2 bitwise PASS（强 PR-D/E/F design 证据）。

## Forward debts（PR-N + P2 + P7 inherit）

| Debt | Source | Disposition |
|---|---|---|
| F-K2-1 | spec `p1-state-update-parallel` L184 Scenario "A3a/A3b vs N=1 same-binary baseline" 当前 ambiguous dual-FAIL handling | PR-N task 6.8b 升级 spec wording 明确 dual-FAIL at N≥4 落 WARNING bucket per D5 NG3 |
| F-K2-2 | PR-K2 §2.6 "trajectory bifurcation via CVODE nst drift" framing 当前 hypothesis-level | P2+ root-cause framing 精确化：bisect tree-reduction-depth transition vs FMA selection vs scheduler-locale；产物 `docs/p1_a3a_root_cause.md` |
| P7 final-fusion deterministic-reduction | master plan §6 P7 scope；PR-K2 §2.6 + 本 summary §"§1.1.1 verdict" 根因 2 | P7 capstone 引入 fixed-shape pairwise canonical reduction OR P9 deterministic N_Vector；目标 A3a/A3b strict at N ∈ {2,4,8} 全 PASS |
| `_before_passvalue` 2nd-suffix mid-pipeline drift 3 case (xj_up/qinyijiang/qhh) | PR-H #219 diagnostic addendum (`docs/p1_rhs_snapshot_bitwise.md` § Diagnostic addendum) | P2a / PR-N follow-up issue (#TBD) bisect 假设 (a)/(b)/(c)；不阻塞 P1（canonical 12-cell gate PASS + 下游 rivqdown/cvode_stats 全 PASS）|

## B-chain immutable (D11 protection enforced)

```
B0-tag (884cfb13 / SHUD 78c37a1, 2026-06-17)
  └─ B1a-tag (f7f992c / SHUD 0b3998d, 2026-06-21 capstone, force-updated once from S1d-end 64569b3)
      └─ B1b-tag (18a0c908 / SHUD 71b3a1ae, 2026-06-22 PR-16 #207 capstone, D11 immutable)
          └─ B1-tag (ed054b4 / SHUD 017c629, 2026-06-22 PR-19 #210 D9 fast-path trigger #2,
              aliases main HEAD with #205 cleanup + PI E2 sign-off)
              └─ P1-update-omp-tag (003f58d / SHUD 07c677f, 2026-06-22 PR-L #224 capstone)
                  ↑ baseline/P1 lock_branch=true / enforce_admins=true
```

P1-update-omp-tag 后下游 P2 阶段优先从 `main` 分新分支；`baseline/P1` 仅作 frozen baseline 历史比对参照，不再 base 新 PR；P-strict / P-prod baseline branch 在对应阶段开启时建立（reserved Open Question #5）。

## Next phase

**P2 sub-stages from baseline/P1**（master plan §3 路线）：
- P2a — `f_updatei` case 1-5 OMP（spec NG7 reserved；本 change 留给 P2a reviewer）
- P2b — `MD_f.cpp` rhs_flux compute + gather 并行（B1b S3c `rhs_deterministic_gather()` baseline → owner-local-fanout pattern）
- P3+ — owner-local gather parallel + OMP_CUTOFF + N_Vector parallel
- P7 final-fusion — deterministic reduction (上节 forward debts 根因 2)
- P9 — production deterministic N_Vector + §1.1.1 P9 final 6.0× T 全数达成

**D9 fast-path 可能触发 P1-stack-extension**：如果 P2a/P2b sub-stage 全部 (a) safe + bitwise vs B1-tag PASS，可走 forward-compat P1c-tag stacking 在 `P1-update-omp-tag` 之上 stack，不需 P-strict-tag 新建；判定时机在 P2 capstone PR。详 design D9。

## 验证 P1-update-omp-tag

```
git ls-remote --tags origin | grep P1-update-omp-tag
# refs/tags/P1-update-omp-tag         ff21c75c8e968d5e47ca53b015425360be9ac879  ← annotated tag object SHA
# refs/tags/P1-update-omp-tag^{}      003f58dc079116ef2161d2f96006228ef0e013d0  ← dereferenced commit SHA（SHUD pin 07c677f）

git show P1-update-omp-tag --no-patch --format=fuller
# Tagger:     DankerMu <mumzy@mail.ustc.edu.cn>
# 7-bullet P1 fix list / SHUD pin 07c677f / 6-case canonical SHA / M7 trim + scaling 概要
# D11 immutable + Aliases master plan §3 P1 baseline + 13/14 PR merged

gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'
# {"allow_deletions":false, "allow_force_pushes":false, "enforce_admins":true, "lock_branch":true}
```

P1-update-omp-tag 一次锁死，禁止 force-update（D11）；任何后续 retroactive 更新走 forward-compat P1c-tag / P2-* stacking 路径（C8）。
