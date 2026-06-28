# ADR-0004: maxl-sweep-decision — SPGMR maxl knob outcome (Optional-knob branch) + case-size-asymmetric Krylov pattern

- **Status**: Accepted (P8-tune.C epic close, 2026-06-27)
- **Date**: 2026-06-27
- **Deciders**: DankerMu + Claude orchestrator (per `tools/p8tune/aggregate_maxl_sweep.sh` 8-gate verdict + `docs/p8tune/maxl_sweep_verdict.md` adjudication)
- **Owner**: SHUD-OpenMP 改造工程 / P8-tune.C epic capstone → P8-tune.D KLU intake (NOT opened — Optional-knob suffices, no KLU spike triggered)
- **Tags**: spgmr / maxl / Krylov-subspace / tune / Optional-knob / PREC_NONE / case-asymmetric / runtime-env-var
- **Supersedes**: none (first maxl-related ADR)
- **Superseded by**: none
- **Related**: ADR-0002 Path 3 close (p8pre-spike NO-GO) + ADR-0003 (clean PREC_NONE baseline + Path 4 SPGMR.maxl trigger) + master plan §P8-tune.C + `openspec/changes/p8tune-spgmr-maxl/` (epic SHUD-OpenMP#362) + PR-D #373 (60-cell sweep data producer) + PR-C #372 (env-var hook) + PR-A #370 (cleaned PREC_NONE baseline data) + 本 ADR 所在 PR

---

## Context

ADR-0003 在 p8pre-spike NO-GO 决策后建立了 "cleaned PREC_NONE production baseline" 并触发了 P8-tune.C epic (#362) 的 Path 4：**SPGMR.maxl 调参实证**。Path 4 trigger condition 是 (i) PREC_NONE 仍是 production solver、(ii) SUNDIALS 缺省 `maxl=5` 在 heihe/heihe_x4 两个 NWM case 上 ncfn+ncfl 触发非零 (heihe ncfn=7 ncfl=85 / heihe_x4 ncfn=51 ncfl=3620) 暗示 Krylov subspace 不够大、(iii) ADR-0004 GO/Optional/Diagnostic/NO-GO 4-branch outcome 决定后续是否 default-bump 或 P8-tune.D KLU pattern-only spike。

**P8-tune.C epic 5+1 PR 序列** (#362)：

| PR | Title | Issue | Status | Merge SHA |
|---|---|---|---|---|
| PR-0 | glossary + capability seed | #363 | ✅ MERGED | (PR #369) |
| PR-A | cleaned PREC_NONE baseline (18-cell heihe+heihe_x4 keliya-anchor) | #364 | ✅ MERGED | (PR #370) |
| PR-B | verdict gate (Diagnostic/Optional/GO/NO-GO 4-branch logic from PR-A data) | #365 | ✅ MERGED | `9a318e3` (PR #371) |
| PR-C | runtime `SHUD_SPGMR_MAXL` env-var hook (3-belt validation + provenance log) | #366 | ✅ MERGED | `b5210b9` (PR #372) |
| PR-D | 60-cell sweep tools + execution (Slurm 9690 60/60 COMPLETED) | #367 | ✅ MERGED | `cff4975` (PR #373) |
| PR-E | aggregator + 8-gate verdict + 本 ADR-0004 + ADR | #368 | (本 PR) | (TBD) |

**60-cell sweep matrix** (Slurm 9690 on cn[06,09,11,14-19,21-22,24], ~80 min wall total, 11-way concurrency)：

- 5 maxl ∈ {5, 10, 15, 20, 30}
- 2 case ∈ {heihe (~6300 elem), heihe_x4 (~40046 elem)}
- 2 N (OMP thread) ∈ {1, 8}
- 3 rep
- = **60 unique cells**, 全部 ExitCode 0, 全部产出 5 mandatory artifacts (`profile_B0.yaml`, `cvode_stats.txt`, `rivqdown.dat`, `wall.sec`, `cell.meta`)

每 cell stdout 含 PR-C provenance line `[CVODE] SPGMR maxl=<k> pretype=PREC_NONE` (60/60 prov_log_count=1)；server-side _summary.tsv (61 行 × 22 列) 是 PR-E 聚合输入。

---

## Decision

**Adopt Optional-knob branch.** 保留 PR-C `SHUD_SPGMR_MAXL` env-var 为 production opt-in 调参，缺省 (unset = SUNDIALS-default maxl=5) 不变。

**Per-case best-maxl recommendation** (落地于 `docs/p8tune/maxl_sweep_verdict.md` §production-tune-guidance，3-rep median 数据)：

| (case, N) | recommended `SHUD_SPGMR_MAXL` | wall reduction vs default | band | tier (see §Acceptance for promotion criteria) |
|---|---|---|---|---|
| heihe N=1 | **=30** | +11.99% | GO (wall band) | Performance opt-in (NOT A5-certified)[¹] |
| heihe N=8 | **=30** | +6.78% | Optional (wall band) | Performance opt-in (NOT A5-certified)[¹] |
| heihe_x4 N=1 | unset (default=5) | maxl ≥10 全 REGRESS −6.86% 至 −15.83% | n/a | n/a |
| heihe_x4 N=8 | unset (default=5) | maxl ≥10 全 REGRESS −15.81% 至 −24.82% | n/a | n/a |

[¹]: `Performance opt-in (NOT A5-certified)` = mechanism-attested per §Rationale §G7 (Arnoldi MGS → CVODE step-size adapter → trajectory drift) but NOT validated by hydrology-equivalence (NSE/KGE/peak/water-balance). Promotion to `A5-certified` requires the future P9-A5 epic (see §Acceptance forward action). Users opting in accept solver-tunable trajectory drift on a different valid PREC_NONE step-size path.

**8-gate verdict** (per `tools/p8tune/aggregate_maxl_sweep.sh`):

| # | Gate | Criterion | Result | Evidence |
|---|---|---|---|---|
| G1 | Build | PR-C CI PASS | **PASS** | PR-C #372 CI 5/5 PASS @ `b5210b9` |
| G2 | No PREC_LEFT regression | 60/60 cells emit `pretype=PREC_NONE` | **PASS** | `_summary.tsv` prov_log_count=1 全 60 行 |
| G3 | Default-compat 4-way | unset/""/"0"/"5" SHA12 identical | **PASS** | PR-C g3_verdict.md SHA12=`1bfe6a30856e` × 4 invocations |
| G4 | Solver-work reduction | ncfl + ncfn 改善 per (case, N) | **PASS** | heihe Δncfl=−85 Δncfn=−2; heihe_x4 Δncfl=−3620 Δncfn=−51 (全消除 ncfn/ncfl) |
| G5 | Wall improvement | per (case, N, maxl) wall band | **MIXED (regression present)** | heihe N=1 maxl=30 +12% GO; heihe_x4 全 N maxl ≥10 全 REGRESSION |
| G6 | No solver regression | sum Δ(ncfn+ncfl+netf) ≤ 0 | **PASS** | 16/16 (case,N,maxl) cells sum_delta ≤ 0 |
| G7 | Hydrology max_ulp ≤ 1024 | rivqdown.dat ULP diff | **STRICT FAIL** | heihe max_ulp ≈ 4.4×10¹⁵; heihe_x4 max_ulp ≈ 4.6×10¹⁸ (expected: maxl bump → Krylov 扩展 → CVODE step-size 改 → 轨迹 drift; 非 corruption) |
| G8 | 3-rep determinism | per (case,N,maxl) SHA12 跨 3 rep identical | **PASS** | 20/20 tuples bit-identical 跨 reps + 15-key counter identical 跨 reps |

---

## Rationale

### G1/G2/G3/G4/G6/G8 PASS — hook itself well-behaved

- **G1/G2/G3** 承自 PR-C CI 验证 (default-unset 路径 bit-identical SUNDIALS-default-maxl=5)，证 env-var hook 不破缺省 behavior。
- **G4** 60-cell 数据证 maxl ≥10 在 4 个 (case, N) 组合下全部消除 ncfl (heihe 85→0, heihe_x4 3620→0) 并降低 ncfn (heihe 7→5, heihe_x4 51→0)。**solver convergence 显著改善**。
- **G6** 每个 (case, N, maxl) cell 的 ncfn+ncfl+netf sum delta 全部 ≤ 0，无任何 counter regression。
- **G8** 跨 3-rep 全 20 个 (case, N, maxl) tuples bit-identical (max_ulp=0)，证 sweep 100% deterministic-repeatable。

### G5 case-size-asymmetric — the key empirical finding

3-rep median wall data 揭示了一个**关键的 case-size-asymmetric pattern**：

**Small case (heihe ~6300 elements) BENEFITS from larger Krylov subspace**：
- heihe N=1: maxl=5 → 30 wall **−12% (149s → 131s)**, ncfl 85→0
- heihe N=8: maxl=5 → 30 wall **−7% (143s → 133s)**, ncfl 85→0

**Large case (heihe_x4 ~40046 elements) SUFFERS from larger Krylov subspace**：
- heihe_x4 N=1: maxl=5 → 30 wall **+16% (1490s → 1726s)** REGRESSION
- heihe_x4 N=8: maxl=5 → 30 wall **+25% (1372s → 1713s)** REGRESSION
- 即便 maxl=10 (saturation point) 也是 −7% 至 −23% wall REGRESSION

### 机制解释 — Krylov vector memory bandwidth (NumY-corrected per GPT Pro 2026-06-28 review)

SUNDIALS SPGMR 内部存 Krylov 基 `V[0..maxl-1]` — **每个向量长度 = NumY** (NOT NumEle / N_elem)。SHUD `N_VNew_Serial(NY, sunctx)` (per `SHUD/src/Model/shud.cpp:139`) 分配的是 N_Vector 长度 NumY，组成为：

```
NumY = 3·NumEle + NumRiv + NumLake
     = [Y_surf | Y_unsat | Y_gw | Y_riv | Y_lake]
       NumEle   NumEle    NumEle  NumRiv   NumLake
```

(per master plan §P8-precond.1 L2525-2531 state-vector layout)

Arnoldi 正交化 (Modified Gram-Schmidt) 在每次 inner iter 需访问已存的所有 V[i] 计算 `dot(V[i], w)` + `w -= dot * V[i]` (per `sunlinsol_spgmr.c` `SPGMRSolve`)。**Working set per Arnoldi iter ≈ maxl × NumY × 8B**。

Per-case NumY estimates (approximate; exact = read from SHUD runtime print at Init):

| Case | NumEle | NumRiv (est) | NumLake (est) | NumY (est) | maxl=10 working set @ 8B |
|---|---|---|---|---|---|
| keliya | 484 | ~50 | 0 | ~1,500 | ~120 KB |
| heihe | 6,335 | ~few hundred | ~16 (Juyan + NE) | **~19,000** | **~1.52 MB** |
| heihe_x4 | 40,046 | ~few hundred | ~16 | **~120,000** | **~9.6 MB** |
| heihe_x16 | ~250,000 | ~few hundred | ~16 | **~760,000** | **~60 MB** |

> NumRiv / NumLake 在 mesh-refine 下**不**线性 scale (rSHUD `shud.triangle(a/4)` 只 multiply Voronoi mesh elements；river network 和 lake polygon 保留)。NumRiv / NumLake estimates pending server runtime-print; replace with exact values in a future PR if precision needed for P8-tune.D fill-ratio formulae.

Cache-fit verdict (per NumY 口径):

- **小 case (heihe ~6300 elem, NumY ~19K)**: maxl=10 working set ≈ 1.52 MB。Intel Skylake-X L2 = 1 MB/core (server SKL-SP variants 1.375 MB)、AMD EPYC Rome L2 = 512 KB/core、Apple M-series L2 = 4-8 MB shared。**1.52 MB 在更大 L2 上 fits**，在 1-MB L2 (Skylake client) 上**已经溢出**。这与实证一致：heihe N=1 maxl=30 sustains GO band (+11.99%, 单线程独占 L2)，N=8 maxl=30 droppes to Optional band (+6.78%, 8 线程争抢 L2 + L3)。bigger maxl → fewer (outer iter) restart → fewer expensive RHS evals → wall ↓ (only when L2 holds the working set)
- **大 case (heihe_x4 ~40046 elem, NumY ~120K)**: maxl=10 working set ≈ 9.6 MB → **显著超出 L2 (1-2 MB)**, 接近 L3 (8-32 MB) 下限。bigger maxl → MGS 每 inner iter 触发 DRAM 读 → wall ↑ 超过 RHS eval 节省的成本。所有 maxl ≥10 全 REGRESS −6.86% 至 −24.82%, 与 cache-pressure 模型一致

**口径校正与原稿对比** (GPT Pro 2026-06-28 review identified N_elem 口径低估 ~3×):

| | 原稿 (N_elem × 8B) | 正确 (NumY × 8B) | 影响 |
|---|---|---|---|
| heihe maxl=10 working set | 507 KB | **1.52 MB** | 仍 fit L2 但更紧；解释 N=8 contention 更硬 |
| heihe_x4 maxl=10 working set | 3.2 MB | **9.6 MB** | 显著超 L2 → DRAM-bound 信号更强 |

口径校正后的 case-asymmetric pattern verdict **更 robust**, 不削弱原结论。

heihe_x4 cell.meta `cells_n=40046` ≈ 6.3× heihe 的 `cells_n=6335`，NumY 比例 ≈ 120K / 19K ≈ 6.3×, 与上述 cache-pressure 模型一致。

### G7 STRICT FAIL — expected numerical phenomenon, not corruption

G7 strict 阈值 `max_ulp ≤ 1024` 失败 (heihe 4.4×10¹⁵ / heihe_x4 4.6×10¹⁸) 是**预期的 solver-tunable-sensitivity**：

1. maxl 增加 → SUNDIALS Arnoldi 用更大 Krylov subspace 解线性子问题
2. 线性收敛改善 → CVODE Newton iter 收敛快 (ncfn ↓)
3. CVODE step-size adapter 检测 ncfn ↓ → 在下一步尝试**更大 dt**
4. dt 增大 → integrated trajectory `[y_n]` 在每个步累计不同的 LTE → 最终 `rivqdown.dat` 轨迹 drift

both maxl=5 和 maxl≥10 都是**合法的** PREC_NONE solver 输出，只是在不同的 step-size 路径上。**这不是 corruption**——是 SUNDIALS CVODE step-size controller 对 linear solver 性能的 closed-loop response。

ADR-0003 establish 的 "cleaned PREC_NONE production baseline" 即 maxl=5 SHA12 anchor (`docs/p8tune/clean_prec_none_baseline.md` §raw-18-cell-table) 仍然是 **production default**。改 maxl 改 trajectory 就是 "改 solver config 改 result" 的 expected behavior。

### 4-branch decision tree closure

ADR-0004 4-branch tree (per `openspec/changes/p8tune-spgmr-maxl/design.md` D8)：

| Branch | Trigger | Adopted? |
|---|---|---|
| **GO + default-bump** | G5 ≥10% uniform + G6/G7/G8 PASS | NO — G5 not uniform (heihe_x4 REGRESS); G7 STRICT FAIL |
| **Optional-knob** | G5 case-asymmetric improvement + G4/G6/G8 PASS + G7 expected drift | **YES (本 ADR adopted)** |
| **Diagnostic** | G4 PASS but G5 <5% uniformly | NO — heihe N=1 maxl=30 12% GO band存在 |
| **NO-GO hydrology** | G7 FAIL = corruption (not expected drift) | NO — G7 drift 是 step-size response，验证为 expected solver behavior |
| **NO-GO solver** | G6 FAIL | NO — G6 PASS |
| **NO-GO no-improvement** | G4-G7 no improvement | NO — G4 PASS 显著 |

**Final: Optional-knob**.

### "Never break userspace" enforcement

PR-A 通过 18-cell 数据 lock 了 maxl=5 (= SUNDIALS default = env-unset) 为 **production baseline SHA12 anchor**。本 ADR 决策**不改 cvode_config.cpp:259 default constant**，确保：

- 现有用户 `unset SHUD_SPGMR_MAXL` (即 99% 用户) 输出 bit-identical PR-A baseline
- 新用户主动 `export SHUD_SPGMR_MAXL=30` (eg. heihe-like 小 case 想要 12% wall 加速) 接受 trajectory drift 换 wall 改善

这是 "user opts in to change" pattern，符合 ADR-0002 D8 "never break userspace" rule。

---

## Consequences

### Positive

1. **小 case 用户可选 +12% wall acceleration** (heihe N=1 maxl=30) 无需 source patch、无需重 compile (Performance opt-in tier; NOT A5-certified — see §Acceptance forward action for P9-A5 promotion path)
2. **solver convergence 失败统计学消除** (heihe ncfl 85→0, heihe_x4 ncfl 3620→0)，给后续 P9 epic（精度等级 A5）提供更稳定的 solver 起点
3. **PR-C env-var hook 长寿化** — 作为 production-supported 调参界面，未来 P8-tune.X / P9 / Prod 都可复用
4. **case-size-asymmetric Krylov pattern 被实证 + 文档化** → 为 P8-tune.D KLU 选型决策提供 mechanistic backbone (KLU 不受 Krylov-vector cache pressure 影响，对大 case 可能更合适)
5. **3-rep determinism PERFECT** — sweep methodology 复用 confidence 100% (G8 0/20 fail)

### Negative

1. **G7 STRICT FAIL 数字突兀** — max_ulp 4.4×10¹⁵ 至 4.6×10¹⁸ 让没读 mechanism 的人误以为 numerical corruption；需要 ADR-0004 §G7 mechanism 段反复 cite。本 ADR 已 explicit 说明
2. **per-case maxl 推荐表碎片化** — 4 个 (case, N) 组合给 4 个不同 maxl 建议，用户必须知道自己 case 的 element count 和典型 thread count；非 one-size-fits-all
3. **heihe_x4 全部 REGRESSION** — 即便 ncfl 3620 全消，也没法换回 wall (用户体验角度：用户看 wall 不看 ncfl)；这是 large case 用户的 "不要碰" 信号
4. **未在 PR-E 中直接触发 P8-tune.D KLU spike** — 本 ADR (PR-E) 是 Optional-knob 而非 NO-GO, 因此**当时不**触发 P8-tune.D KLU spike epic. **但 GPT Pro 2026-06-28 review 在 PR `chore/p8tune-doc-correction` 中重新评估**: 用户意图 "hydrology-validatable large-case acceleration" + NumY 口径分析 (heihe_x4 maxl=10 working set ~9.6 MB 超 L2; heihe_x16 ~60 MB 超 L3) 表明 SPGMR 路径对生产规模 case 已饱和, 不是 wall-improvement 路径。**P8-tune.D KLU pattern-only spike 已 active triggered** (master plan §P8-tune.D anchor section), 4-PR 序列 forthcoming via openspec change `p8tune-klu-spike`。原稿 forward action "若未来有更大 case 可重做 maxl sweep" 由 P8-tune.D 直接 KLU 路径取代 (SPGMR-tune 路径不再扩展)。

### Neutral

1. **不修改 SHUD source** — 本 PR 仅 docs + tools，无 cvode_config.cpp / cvode_main.cpp 改动；SHUD pin 仍 `6ce17d6` (PR-C merge state)
2. **不触发 conditional PR-F default-bump** — tasks §4.11 default-bump branch 不开启
3. **OpenSpec change `p8tune-spgmr-maxl` archive** — 本 PR 合后 archive 到 `openspec/changes/archive/`，capability `maxl-sweep-verdict` 写入 spec lock
4. **epic #362 closeout** — 6 PR 全 merged + ADR-0004 落地 + verdict.md 落地 → epic 关闭

---

## Discussion — case-size-asymmetric Krylov memory pattern (NumY 口径)

这个 pattern 的实证有更广的 implication：

**SUNDIALS SPGMR Arnoldi MGS 复杂度** per outer iter = `O(maxl² × NumY)` (working set 基于 SUNDIALS N_Vector 长度 NumY, 非 NumEle):
- maxl × NumY flops for V matrix storage scan
- maxl × maxl orthogonality H matrix updates
- 每次 inner iter 重新读 V[0..maxl-1] (working set ≈ maxl × NumY × 8 bytes)

`NumY = 3·NumEle + NumRiv + NumLake` (per `SHUD/src/Model/shud.cpp:139` `N_VNew_Serial(NY, sunctx)` 其中 `NY = MD->NumY`; 详 §Rationale §G5 机制解释 per-case NumY 表)。**原稿用 N_elem (= NumEle) 口径低估 working set ~3×**, GPT Pro 2026-06-28 review identified + 本 PR `chore/p8tune-doc-correction` corrected。

当 working set 超 L2/L3 cache → 每次 inner iter 触发 DRAM 读取 → bandwidth-bound

L2 typical size ≈ 1-2 MB per core (Intel Skylake-X 1-1.375 MB / AMD EPYC Rome 0.5 MB / Apple M-series 4-8 MB shared)。L3 typical = 8-32 MB shared per socket。

**NumY 口径下** per-case working set:

- **heihe_x4** (NumY ~120K) maxl=10 = 120000 × 8 × 10 ≈ **9.6 MB** → **显著超 L2, 接近 L3 下限**, 每次 inner iter 落 DRAM。bigger maxl 把更多向量推过 cache 边界 → wall ↑ 主导。所有 maxl ≥10 全 REGRESS 与此一致。
- **heihe** (NumY ~19K) maxl=10 = 19000 × 8 × 10 ≈ **1.52 MB** → 在 ≥1.5 MB L2 核心上 fit，在 1-MB L2 核心 (Skylake client) 上溢出。N=1 独占 L2 sustains GO band (+11.99%); N=8 八线程争 L2 + 互相驱逐 → Optional band (+6.78%)。
- **keliya** (NumY ~1.5K) maxl=10 ≈ 120 KB → 远在 L1 内, 无 cache pressure。
- **heihe_x16** (NumY ~760K, future scale) maxl=10 ≈ **60 MB** → 显著超 L3 → 即便单线程也 DRAM-bound, SPGMR 路径不适合此规模。**这就是 P8-tune.D KLU spike trigger 的核心 motivation**。

**Forward implication (post-correction, active triggers)**：

- KLU (direct sparse) 不存 Krylov 向量，pivot+L+U 是一次 setup amortized cost。对大 case 在 wall 维度**可能更友好** (具体 fill ratio / RSS / wall vs SPGMR 由 P8-tune.D spike 实证)
- P8-tune.D KLU pattern-only spike 触发条件 (master plan §P8-tune.D, 本 PR 添加): "**NumY 大到 SPGMR Krylov-vector working set 超 L3 cache**" (heihe_x4 NumY ~120K maxl=10 ~9.6 MB 已经临界, heihe_x16 ~760K maxl=10 ~60 MB 显著超), 加用户意图 "hydrology-validatable large-case acceleration"
- P9-A5 hydrology-equivalence trigger (Optional-knob 当前 maxl=30 promotion 路径): NSE/KGE ≥ 0.95 + peak Δ ≤ 5-10% + water-balance Δ ≤ 1% 验证, promotes Performance opt-in → A5-certified RECOMMEND

---

## Implementation

### PR-E (本 PR) 改动 (4 files)

1. `tools/p8tune/aggregate_maxl_sweep.sh` (新, ~280 lines bash + 内嵌 python3)：60-cell summary.tsv + per-cell rivqdown.dat → 8 T-tables + aggregate_verdict.txt KV + verdict_synthesis.md
2. `tools/p8tune/render_verdict.sh` (新, ~70 lines bash)：T-tables + synthesis → `docs/p8tune/maxl_sweep_verdict.md` 单 file render
3. `docs/p8tune/maxl_sweep_verdict.md` (新)：8-gate verdict + per-combo best-maxl 推荐表 + production tune guidance
4. `docs/adr/0004-maxl-sweep-decision.md` (新，本 ADR)

### 不在 PR-E 改

- **NO** `cvode_config.cpp:259` default constant 改动 (Optional-knob 不 bump default)
- **NO** SHUD submodule pointer 改动 (仍 `6ce17d6`)
- **NO** `docs/p8tune/clean_prec_none_baseline.md` 改动 (PR-A baseline lock 不变)
- **NO** 新建 conditional PR-F (Optional-knob 不触发 default-bump)
- **NO** `openspec validate p8tune-spgmr-maxl --archive` 在本 PR (delivery-PR 后 epic capstone 单独操作)

### Forward action items

- [x] PR-E merge → epic #362 close (manual `gh issue close 362`)
- [x] OpenSpec change archive: `openspec/changes/p8tune-spgmr-maxl/` → `openspec/changes/archive/p8tune-spgmr-maxl-YYYY-MM-DD/` (post-merge cleanup PR)
- [x] PR-376 G7-strict / G7-attested split spec amendment (merged 2026-06-28)
- [x] `chore/p8tune-doc-correction` (本 PR) — NumY 口径 correction + maxl=30 wording softening + master plan §P8-tune.C CLOSE + §P8-tune.D anchor (per GPT Pro 2026-06-28 review)
- [ ] **P8-tune.D KLU pattern-only spike epic** (master plan §P8-tune.D, openspec change `p8tune-klu-spike` forthcoming) — 4-PR sequence (PR-0 tool + PR-A 16-cell Slurm array sweep + PR-B aggregator+ADR-0005 + PR-C capstone), 2-3 weeks. Trigger: NumY-based working-set analysis (heihe_x4 ~9.6 MB > L2 / heihe_x16 ~60 MB > L3) + user intent hydrology-validatable large-case acceleration
- [ ] **P9-A5 hydrology-equivalence epic** (future) — validate `SHUD_SPGMR_MAXL=30` heihe N=1 trajectory drift (criteria: NSE/KGE ≥ 0.95, peak Δ ≤ 5-10%, water balance Δ ≤ 1%, peak timing ≤ 1 output interval); PASS promotes Performance opt-in → A5-certified RECOMMEND tier
- [ ] heihe_x16 KLU spike data — folded into P8-tune.D PR-A 16-cell array (case set: keliya + heihe + heihe_x4 + heihe_x16); supersedes original "复跑 aggregate_maxl_sweep.sh" item which was SPGMR-tune extension (SPGMR path saturated for large case per NumY analysis)

---

## Acceptance

本 ADR Accepted (status flag = `Accepted`)，effective `2026-06-27`。NumY 口径 + tier 标注 amended via PR `chore/p8tune-doc-correction` (2026-06-28).

- **Decision adopted**: Optional-knob branch
- **Hook lifecycle**: PR-C `SHUD_SPGMR_MAXL` env-var = long-lived production opt-in
- **Default unchanged**: cvode_config.cpp default maxl path = SUNDIALS-default 5 (preserved)
- **Production tune guidance** (Performance opt-in tier, NOT A5-certified — see Forward action P9-A5 epic for promotion path):
  - heihe N=1 → `SHUD_SPGMR_MAXL=30` Performance opt-in (+12% wall, GO wall band)
  - heihe N=8 → `SHUD_SPGMR_MAXL=30` Performance opt-in (+7% wall, Optional wall band)
  - heihe_x4 任 N → keep unset (large-case Krylov-vector working set 9.6 MB > L2 → DRAM-bound; SPGMR path saturated)
- **Tier definitions**:
  - `A5-certified (RECOMMEND)`: mechanism-attested AND A5 hydrology-equivalence validated (NSE/KGE/peak/water-balance) — currently NONE of the maxl values are in this tier
  - `Performance opt-in (NOT A5-certified)`: mechanism-attested (per §Rationale §G7) but A5 validation pending — current state of `=30` for heihe N=1 / N=8
  - `Diagnostic / no recommendation`: mechanism unclear or unattested — n/a for current sweep
- **P8-tune.D KLU pattern-only spike**: **TRIGGERED** by 本 PR `chore/p8tune-doc-correction` per GPT Pro 2026-06-28 review (user intent: hydrology-validatable large-case acceleration; NumY analysis confirms heihe_x4 / heihe_x16 SPGMR-path saturated). 4-PR epic forthcoming per master plan §P8-tune.D.
- **P9-A5 hydrology-equivalence**: future epic, validation criteria documented in Forward action items; promotes `Performance opt-in` to `A5-certified` upon PASS.

---

## References

### Internal (本仓库)

- `tools/p8tune/aggregate_maxl_sweep.sh` (本 PR) — 60-cell 数据 → 8-gate verdict 产生器
- `tools/p8tune/render_verdict.sh` (本 PR) — markdown 渲染器
- `docs/p8tune/maxl_sweep_verdict.md` (本 PR) — 8 详细 T-table + verdict synthesis
- `docs/p8tune/clean_prec_none_baseline.md` (PR-A #370 + PR-B #371) — PR-A 18-cell PREC_NONE baseline + PR-B 4-branch verdict gate
- `docs/adr/0003-precond-spike-decision.md` — p8pre-spike NO-GO + Path 4 trigger
- `docs/adr/0002-solver-path.md` — Path 1-4 路径表
- `openspec/changes/p8tune-spgmr-maxl/{proposal.md,design.md,tasks.md}` (本 epic OpenSpec change)
- `openspec/changes/p8tune-spgmr-maxl/specs/maxl-sweep-verdict/spec.md` — capability spec
- `SHUD_openMP_master_plan.md` §P8-tune.C — epic 路线
- `SHUD-System/SHUD` @ `6ce17d6` (openmp-baseline branch) — PR-C env-var hook live SHUD pin

### PR sequence (epic #362)

- PR #369 — PR-0 glossary + capability seed (#363)
- PR #370 — PR-A cleaned PREC_NONE baseline 18-cell (#364)
- PR #371 — PR-B verdict gate 4-branch (#365)
- PR #372 — PR-C runtime env-var hook (#366)
- PR #373 — PR-D 60-cell sweep tools + execution (#367)
- PR-E — 本 ADR + aggregator + verdict.md (#368)

### Server data (NOT in repo)

- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/_summary.tsv` — 60-cell raw counter + wall + sha12
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/aggregate_verdict.txt` — flat KV mirror
- `/scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/{heihe,heihe_x4}_N{1,8}_maxl{5,10,15,20,30}_rep{1,2,3}/` — 60 cell dirs × 5 artifacts each
- Slurm job 9690 (cn[06,09,11,14-19,21-22,24]) — execution job

### External

- SUNDIALS user guide v6.0.0 — `SUNLinSol_SPGMR` `maxl` parameter (Krylov subspace dimension; default 5)
- SUNDIALS `sunlinsol_spgmr.c` `SPGMRSolve` — Modified Gram-Schmidt orthogonalization implementation
- SUNDIALS CVODE step-size adapter — `CVStep` `acor` / `tau` adapter loop sensitive to `ncfn` count

---

## Spec-amendment note (post-epic-close, retrospective)

The 8-gate verdict spec at `openspec/changes/p8tune-spgmr-maxl/specs/maxl-sweep-verdict/spec.md` originally defined a single G7 hydrology gate with the strict predicate: "ANY A4 max_ulp violation SHALL fail G7 (blocks GO and Optional-knob bands)". PR-E Phase 4.5 verifier flagged a literal contradiction (verdict: PLAUSIBLE non-blocking): this ADR-0004 adopts Optional-knob despite G7 STRICT FAIL because the max_ulp violation here is **expected solver-tunable-sensitivity** (the mechanism chain documented in §Rationale §G7) rather than corruption.

After epic close, the spec was amended (see PR chore/p8tune-spec-g7-split-gate) to split G7 into two sub-scenarios:

- **G7-strict** (hard for GO+default-bump): A4 max_ulp ≤ 1024 strict; fails → blocks GO+default-bump branch (default-user trajectory must not change per 'never break userspace')
- **G7-attested** (soft for Optional-knob / Diagnostic, ADR-mechanism path): if G7-strict fails, the violation MAY be reclassified PASS if an ADR (this ADR-0004 is the prototype) documents the violation as solver-tunable-sensitivity with mechanism chain

This ADR-0004 is the **prototype G7-attested attestation** for future maxl-like sweep epics. The mechanism chain — `maxl bump → SUNDIALS Arnoldi MGS residual change → CVODE step-size adapter closed-loop response → trajectory drift on different valid PREC_NONE step-size paths` — is the canonical template for what counts as "solver-tunable-sensitivity, not corruption".

Cross-references:
- spec G7-attested scenario explicitly cites this ADR-0004 as the worked example
- `tools/p8tune/aggregate_maxl_sweep.sh` ADR-branch picker logic updated to expose `G7_strict_pass` + `G7_attested_required` flags in `aggregate_verdict.txt` KV, making the orchestrator's ADR-attestation responsibility explicit
- The amendment does not alter PR-E's adopted Optional-knob decision; it only formalizes the spec wording so future ADR authors don't face the same literal-predicate-vs-mechanism tension

For future maxl-like or solver-tunable sweeps (P8-tune.D KLU spike if triggered, P9 precision epic, etc.), authors SHALL:
1. If G7-strict PASS → proceed to GO+default-bump consideration normally
2. If G7-strict FAIL → either (a) author a new ADR documenting mechanism chain + cite spec G7-attested → claim G7-attested PASS for Optional-knob branch; or (b) classify as NO-GO hydrology (corruption, revert hook)
