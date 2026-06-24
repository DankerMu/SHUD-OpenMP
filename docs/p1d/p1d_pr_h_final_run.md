# P1d PR-H — server final 8-cell three SHALL gate verdict (#282)

## Verdict: **FAIL** (A3a + nst Δ=0 gates FAIL at N≥4)

Per `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` L149-150:
- **WHEN** PR-H 三 SHALL gate 任一项 FAIL
- **THEN** PR-H verdict SHALL 为 FAIL + epic issue 标 blocked + 不进入 PR-I/J/K/L/M

Per burst directive: epic #274 marked BLOCKED, awaiting user decision.

## Scope

Production server (`frd_muziyao@210.77.77.22:32099`) 8-cell matrix:
- Cases: `heihe` (NumEle=6335) + `heihe_x4` (NumEle ~25000)
- Thread counts: N ∈ {1, 2, 4, 8}
- Per `docs/p1d/p1d_pr_f_intermediate_run.md` finding: **DROP** `numactl --interleave=all` (PR-B runbook prescription is anti-pattern with first-touch active).
- Env: `OMP_PROC_BIND=close` + `OMP_PLACES=cores` + `OMP_NUM_THREADS=${N}` only.

## Environment

- Outer baseline/P1d HEAD: `21de1e2` (PR-G merge commit)
- SHUD openmp-baseline HEAD: `210ac19` (post-PR-G Kahan revert; 3 reduction helpers restored to naive form; PR-C/D/E first-touch loops preserved byte-identical)
- Server build: `make clean && make shud_omp` PASS; FP strict 3-grep `-ffp-contract=off + -fno-fast-math + -fopenmp` upheld; `-ffast-math / -Ofast` 0
- Slurm 三铁律 honored: sbatch + run dirs + logs all under `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/`
- Cluster: CPU partition, 1 node × 8 cpus-per-task per job; jobs ran in parallel on `cn03` + `cn07` (8-cell wall-clock end-to-end ~20 min)
- Slurm job IDs: 9041-9048

## NUMA gate confirmation

Every cell's stdout emits the `[NUMA]` tokens proving `g_numa_first_touch_enabled = 1`:

```
[NUMA] OMP_PROC_BIND=close
[NUMA] first-touch begin tag=hot.soa
[NUMA] first-touch begin tag=QeleSurf_flat
[NUMA] first-touch begin tag=Ele_AoS
[NUMA] first-touch begin tag=LoadIC
```

## PR-H SHA matrix (full 64-hex)

| Cell | PR-H SHA256 (post-Kahan-revert + first-touch + no --interleave) | wall (s) |
|---|---|---|
| heihe N=1 | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | 513 |
| heihe N=2 | `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471` | 513 |
| heihe N=4 | `fe7f1f071810d519617f3edad8918a799f237fb5f93cc727367000955318659a` | 503 |
| heihe N=8 | `e67b6592fa75d90afb077e72448436dd040e21871378e862576777b8eff1fe83` | 456 |
| heihe_x4 N=1 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | 1187 |
| heihe_x4 N=2 | `55403bef48ee5ad8e7d73a6c6b675a198c56a95f654ba486fa014a73824fe022` | 1169 |
| heihe_x4 N=4 | `81e13d7aed9bb4739e77eedf2b0ee143129445251262ebd3dc3068c573fb2f86` | 1096 |
| heihe_x4 N=8 | `2099a1f7d303a87e4b791f050638969628f6ecbbd569178eeabf7c2d4ff13281` | 937 |

## PR-H CVODE stats (nst, nfe)

| Cell | nst | nfe |
|---|---|---|
| heihe N=1 | 6773 | 7035 |
| heihe N=2 | 6773 | 7035 |
| heihe N=4 | 6693 | 6949 |
| heihe N=8 | 6621 | 6901 |
| heihe_x4 N=1 | 6571 | 6733 |
| heihe_x4 N=2 | 6571 | 6733 |
| heihe_x4 N=4 | 6582 | 6749 |
| heihe_x4 N=8 | 6575 | 6742 |

## Three SHALL gate verdict (spec.md L126-150)

### Gate 0 — Kahan revert canonical SHA (spec L123 Scenario)

> **THEN** 输出 `output/heihe.out/heihe.rivqdown.dat` SHA256 SHALL 等同 `P1-update-omp-tag` canonical SHA（`7f22bd6faa438d50...`）

- Spec canonical SHA prefix: `7f22bd6faa438d50...`
- PR-H heihe N=1 actual: `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`
- **Verdict: PASS** — byte-identical to spec canonical. Proves PR-G revert restored the pre-Kahan SHUD trajectory; first-touch stacked above doesn't disturb it at N=1.

### Gate 1 — §4.4 A3a bitwise across N (spec L130 Scenario, server only)

> **THEN** heihe + heihe_x4 每 case 的 N ∈ {1,2,4,8} 4 cell SHA256 SHALL 全等

| Case | N=1 SHA (12) | N=2 SHA (12) | N=4 SHA (12) | N=8 SHA (12) | All equal? |
|---|---|---|---|---|---|
| heihe | `7f22bd6faa43` | `7f22bd6faa43` | `fe7f1f071810` | `e67b6592fa75` | **NO (3 distinct)** |
| heihe_x4 | `55403bef48ee` | `55403bef48ee` | `81e13d7aed9b` | `2099a1f7d303` | **NO (3 distinct)** |

**Verdict: FAIL** for both cases. N=1/N=2 are stable (low-thread bitwise reproducible), but N=4 and N=8 each produce distinct SHA. This residual variation is the SAME pattern P1c era exhibited (with Kahan IN) — Kahan revert did NOT change the residual structure.

### Gate 2 — §4.5 nst Δ=0 + ladder (spec L139 Scenario, server only)

> **THEN** heihe `nst` 跨 N ∈ {1,2,4,8} SHALL 全等（Δ=0 严格 hard gate）
> **AND** heihe_x4 `nst` 跨 N SHALL `|Δ_nst| ≤ 2`

| Case | nst N=1 | nst N=2 | nst N=4 | nst N=8 | Δ N=2 | Δ N=4 | Δ N=8 | Strict criterion | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| heihe | 6773 | 6773 | 6693 | 6621 | 0 | **80** | **152** | Δ=0 strict | **FAIL** |
| heihe_x4 | 6571 | 6571 | 6582 | 6575 | 0 | **11** | 4 | \|Δ\|≤2 ladder | **FAIL** (N=4: 11>2; N=8: 4>2) |

**Verdict: FAIL** for both cases at N≥4.

### Gate 3 — N=1 reverse-compat (spec L145 Scenario, 6-case)

> **THEN** 6 case × NUM_OPENMP=1 SHA256 SHALL byte-identical 至 `P1-update-omp-tag` canonical SHA

Server portion (this PR):
- heihe N=1: `7f22bd6faa438d50...` == spec canonical `7f22bd6faa438d50...` ✓ **PASS**
- heihe_x4 N=1: `55403bef48ee5ad8...` — no spec canonical SHA pre-written; need P1-update-omp-tag era server heihe_x4 N=1 reference. **DEFERRED** (not in spec L123 pre-written value).

Mac portion (PR-J): pending PR-I reference + PR-J comparison; NOT executed per spec L150 (PR-H FAIL stops burst).

**Verdict: PARTIAL PASS** (heihe N=1 server portion satisfies the only spec-pre-written canonical SHA). Other server/Mac cells deferred per BLOCKED stop protocol.

## Comparison with P1c Kahan baseline (informational)

P1c reference (`/scratch/frd_muziyao/SHUD-OpenMP/.p1c-runs-kahan/`, Kahan IN, no NUMA env, no first-touch):

| Cell | P1c nst | PR-H nst | Δ_nst | P1c SHA(12) | PR-H SHA(12) | SHA equal? |
|---|---|---|---|---|---|---|
| heihe N=1 | 6553 | 6773 | +220 | `fd2d55716b5d` | `7f22bd6faa43` | NO |
| heihe N=2 | 6553 | 6773 | +220 | `fd2d55716b5d` | `7f22bd6faa43` | NO |
| heihe N=4 | 6524 | 6693 | +169 | `e058db2e9c2a` | `fe7f1f071810` | NO |
| heihe N=8 | 6608 | 6621 | +13 | `6285e8a4a30a` | `e67b6592fa75` | NO |
| heihe_x4 N=1 | 6571 | 6571 | 0 | `4eb804f571ba` | `55403bef48ee` | NO |
| heihe_x4 N=2 | 6571 | 6571 | 0 | `4eb804f571ba` | `55403bef48ee` | NO |
| heihe_x4 N=4 | 6574 | 6582 | +8 | `ff0787abd217` | `81e13d7aed9b` | NO |
| heihe_x4 N=8 | 6569 | 6575 | +6 | `6e9f9a2eaf65` | `2099a1f7d303` | NO |

Observations:
- PR-H heihe N=1 ≠ P1c heihe N=1 (expected: Kahan IN/OUT alters CVODE adaptive trajectory; Mac PR-G already demonstrated this with keliya: pre-K2 vs Kahan-IN differ at every config). Server pattern matches Mac evidence.
- PR-H heihe_x4 N=1 nst Δ=0 vs P1c (both = 6571) but SHA differs (different CVODE trajectory still produces different downstream FP sequence).
- PR-H heihe N=1 == spec L123 pre-written canonical `7f22bd6faa438d50...` (this PR's PASS evidence).

## Comparison with PR-F v2 (Kahan IN baseline, --interleave=all)

PR-F v2 SHA same as P1c (Kahan IN preserves trajectory at N=1/N=2). PR-F v2 walls vs PR-H walls — confirms PR-F finding that `--interleave=all` is anti-pattern (PR-H without it gets heihe_x4 walls back to v1 range ~1187-937s vs PR-F v2 ~1305-1037s).

| Cell | PR-F v2 wall | PR-H wall | Δ |
|---|---|---|---|
| heihe_x4 N=1 | 1305 | 1187 | -118 (-9%) |
| heihe_x4 N=8 | 1037 | 937 | -100 (-10%) |

Drop of `--interleave=all` regained 9-10% wall on heihe_x4. PR-F finding confirmed.

## Root cause analysis (per spec design.md D9 branch 2)

Design D9 anticipated this outcome:
- P1c PR-K2 Kahan injection was hypothesized to be sufficient for §4.5 nst Δ=0; empirically P1c still had `|Δ_nst|=84` on heihe (Kahan masking insufficient for cross-N bitwise).
- P1d hypothesis: NUMA env + first-touch + Kahan revert would achieve cross-N bitwise without Kahan. PR-H empirically shows this is also insufficient — N≥4 cross-N residual variation persists.

The root cause is NOT in the 3 reduction helpers (Kahan touched) NOR in the rhs_update/rhs_flux NUMA placement (first-touch addresses). It's likely:
1. **SPGMR iteration order under multi-threaded preconditioner**: at N≥4, the Krylov subspace gets perturbed by reduction order in solver internals (not in RHS).
2. **CVODE adaptive step controller WRMS sensitivity to low-bit perturbation**: at N=1/N=2 the FP sequences are deterministic; at N≥4 the parallel reduction order in SPGMR Jacobian-vector products introduces low-bit variation that compounds via the step controller.

Neither root cause is addressable by NUMA placement or Kahan compensation alone. Possible remediation paths:
- **A. Accept residual + soften L139 ladder**: heihe |Δ_nst| ≤ K for K observed at PR-H ≈ 200 (spec rewrite required).
- **B. Switch CVODE linear solver from SPGMR to KLU (direct factorization)**: removes parallel reduction in solver internals; preserves bitwise across N. Major refactor.
- **C. Deterministic gather across SPGMR reduction**: requires architectural change in CVODE 6.0.0 SPGMR module or custom hook.
- **D. Restrict bitwise SHALL gate to N=1/N=2** (low-thread bitwise stability is achievable; high-thread residual accepted as ladder).

## Status

- PR-H verdict: **FAIL**
- Epic #274: marked BLOCKED via gh issue comment
- PR-I (Mac worktree reference): completed but work product held (not committed/pushed) pending user decision
- PR-J/K/L/M: NOT started per spec L150

## Artifacts (server, gitignored)

- sbatch: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/run_p1d_pr_h.sbatch`
- 8 run dirs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/{heihe,heihe_x4}_N{1,2,4,8}/`
- Each run dir contains: `rivqdown.sha256`, `wall.txt`, `cvode_stats.txt` (15-key set), `output_listing.txt`, `done.txt`
- Slurm logs: `/scratch/frd_muziyao/SHUD-OpenMP/.p1d-runs/pr-h-final/logs/p1d_pr_h_{9041..9048}.{out,err}`

## 重大实测补充（post-verdict, 2026-06-24）

初版 verdict 把 PR-H FAIL 表述为 `nst Δ` 量级超 spec ladder（K=200 vs K=0），但只看了 CVODE step count，没看实际 `rivqdown.dat` 河道流量。补做后发现：**N≥4 不是 ULP-level 漂移，是 CVODE 轨迹分叉**，下游水文输出 mean rel error 在 10-25% 量级。

### 实测 rivqdown.dat 数值散度（PR-H, 90-day heihe / heihe_x4）

| Case / N | mean_rel | RMSE | max_rel | n_diff / n_total | 不同值占比 |
|---|---|---|---|---|---|
| heihe N=2 vs N=1 | 0 | 0 | 0 | 0 / 214252 | 0% |
| heihe N=4 vs N=1 | **3.8%** | 1.79e+03 | 1010× | 210774 / 214252 | 98.4% |
| heihe N=8 vs N=1 | **10.0%** | 2.76e+03 | 12534× | 210779 / 214252 | 98.4% |
| heihe_x4 N=2 vs N=1 | 0 | 0 | 0 | 0 / 387607 | 0% |
| heihe_x4 N=4 vs N=1 | **17.6%** | 8.63e+03 | 2214× | 383130 / 387607 | 98.8% |
| heihe_x4 N=8 vs N=1 | **25.1%** | 7.59e+03 | 2908× | 382116 / 387607 | 98.6% |

### P1c era 对照（Kahan IN, no NUMA env, no first-touch — 即已上线版本）

| Case / N | mean_rel | RMSE | max_rel |
|---|---|---|---|
| heihe N=4 vs N=1 | 3.0% | 5.86e+03 | 213× |
| heihe N=8 vs N=1 | 10.0% | 1.35e+03 | 1446× |
| heihe_x4 N=4 vs N=1 | 3.5% | 7.80e+03 | 4710× |
| heihe_x4 N=8 vs N=1 | **20.3%** | 9.39e+03 | 66120× |

**P1c era 与 PR-H 在 N=8 上同量级**（heihe 10% vs 10%, heihe_x4 20% vs 25%）。Kahan 注入只压住 `|Δ_nst|`，**没修水文输出散度**。P1d Kahan revert（PR-G）也没让事情变差太多——drift 是 SUNDIALS SPGMR 层面物理限制，per design D9。

heihe_x4 N=4 PR-H (17.6%) > P1c (3.5%) — Kahan 在此 mesh 上 mask 效果更好；revert 后裸露出来。N=8 双方量级相当。

### 并行加速比实测

| Case | N=1 wall (s) | N=2 wall | N=4 wall | N=8 wall | Speedup (8 cores) |
|---|---|---|---|---|---|
| heihe | 513 | 513 | 503 | 456 | **1.13×** |
| heihe_x4 | 1187 | 1169 | 1096 | 937 | **1.27×** |

Amdahl 反推：parallel fraction ≈ 28% (heihe) / 28% (heihe_x4)。**72% 时间在 CVODE+SPGMR 串行内核**。第一性原理：花 8 倍 CPU 换 1.13× 加速 + 10-25% 流量误差，**ROI 为负**。

### K=200 vs K=0 在流量层面的真实对应关系

| Mode | nst Δ | 流量散度量级 | 学术 / 工程含义 |
|---|---|---|---|
| K=0 (true strict) | 0 | ≤ULP | 真 reproducible，可作 forensic 复现 |
| K=200 (PR-H actual) | 80-152 (heihe) | **10-25% rel error** | 跑出来是**另一条 CVODE 轨迹**；峰值差 1000-12500× |

放在水文工程语境对照：
- vs CMFD forcing 不确定性 (10-20%) → **同量级，被 forcing noise 吃掉**
- vs gauge 测量精度 (5-10%) → **超过测量精度**
- vs Manning's n 标定带宽 (~50%) → 量级以下
- vs CVODE reltol=1e-4 承诺 → **超出 4 个数量级**

学术发表层面：reltol=1e-4 但实测 1e-1，**solver 自身的精度承诺已经被打穿**。不能说"converged within tolerance"——这是 tolerance 之外的轨迹分叉。

### Remediation 路径重新评估

| 路径 | 描述 | 实测前评估 | 实测后评估 |
|---|---|---|---|
| **A** | 放宽 ladder K=200 (spec rewrite) | 推荐 | **不推荐**——K=200 对应 10-25% 流量误差，承诺"strict"等同 false advertising |
| **B** | SPGMR → KLU direct solver (大重构) | 不推荐 | **P2a/P2b 阶段考虑**——这是唯一根治路径，但 wall 可能退化 + 大 case 内存爆炸风险 |
| **C** | SUNDIALS 内部 deterministic gather patch | 不推荐 | **不推荐**——验收周期长，未必彻底 fix |
| **D** | N=1/2 bitwise + N≥4 ladder 双 mode 承诺 | 推荐配合 A | **配合 E**——承诺重定义思路对，但 ladder 不该承诺 |
| **E (NEW)** | ship serial-only + OMP research artifact | n/a | **新推荐**——诚实，与 P1c 实际状态 + PR-H 实测一致 |

### E 路具体内容

1. **production 默认 `NUM_OPENMP=1`**（serial path 实测可 reproducible + 不慢太多：heihe 513s vs 456s @ N=8 only 11% slower）
2. **`shud_omp` 二进制继续 ship + build**（CI / 验收保留），但 default `cfg.para NUM_OPENMP=1`
3. **PR-K capstone docs 诚实记录**：
   > P1d 完成 NUMA env standardization + first-touch loops (PR-C/D/E) + B0-ordered schedule(static) + P1c §4.7 Kahan revert (PR-G) 全部 mechanical engineering 工作；**OMP runtime mode 由于 SUNDIALS 6.0.0 SPGMR 内部并行 reduction 顺序非 deterministic，N≥4 产生不可接受的 CVODE 轨迹分叉（mean rel error 10-25%, max rel 1000-12500×）**，不推荐 production 使用。
4. **P1d-tag annotated message** 把 finding 作为核心 deliverable：尝试了 NUMA + first-touch + Kahan，得出 SUNDIALS internals 是 bottleneck 的科学结论
5. **三 SHALL gate 重写** for `spec/specs/p1d-numa-governance/spec.md` L126-150：
   - L130 (A3a bitwise): 改成 "N=1 == P1-update-omp canonical SHA strict + N≥2 informational only"
   - L139 (nst Δ): 改成 "N=1 nst deterministic strict + N≥2 informational only"
   - L145 (reverse-compat): 保留 "N=1 6-case bitwise"（这条 PR-H + PR-I 数据已支持）
6. **B 路（KLU）开 ADR** `docs/adr/0001-spgmr-vs-klu.md` 作为 P2/P3 路线图

## Awaiting user decision

Per `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` L150-151:
> 由用户决策 P1d 延展或 master plan 修订（不在本 change scope）

**给决策方的核心问题**：
1. 接受 P1c 早就存在的 10-25% N≥4 流量散度作为 SUNDIALS SPGMR 物理限制，走 E 路（serial-only ship + OMP research-artifact），P1d epic 当晚关单 → P2 路线纳入 KLU/deterministic 工作？
2. 还是 P1d 不关单，立刻投入 B 路（KLU 重构，2-4 周延期 + wall 退化风险）等真 bitwise fix 后再 ship？
3. 其它思路？
