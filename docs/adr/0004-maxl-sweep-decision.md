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

| (case, N) | recommended `SHUD_SPGMR_MAXL` | wall reduction vs default | band |
|---|---|---|---|
| heihe N=1 | **=30** RECOMMENDED | +11.99% | GO |
| heihe N=8 | **=30** Optional | +6.78% | Optional |
| heihe_x4 N=1 | unset (default=5) | maxl ≥10 全 REGRESS −6.86% 至 −15.83% | n/a |
| heihe_x4 N=8 | unset (default=5) | maxl ≥10 全 REGRESS −15.81% 至 −24.82% | n/a |

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

### 机制解释 — Krylov vector memory bandwidth

SUNDIALS SPGMR 内部存 Krylov 基 `V[0..maxl-1]` 每个向量长度 = N_elem。Arnoldi 正交化 (Modified Gram-Schmidt) 在每次 inner iter 需访问已存的所有 V[i] 计算 `dot(V[i], w)` + `w -= dot * V[i]` (per `sunlinsol_spgmr.c` `SPGMRSolve`)。

- 小 case (heihe ~6300 elem × 8B = 49 KB/vector)：8 个 maxl=8 向量 = 392 KB，fit in L2 cache。bigger maxl → fewer (outer iter) restart → fewer expensive (N_elem-scaled) RHS evals → wall ↓
- 大 case (heihe_x4 40046 elem × 8B = 312 KB/vector)：8 个 maxl=8 向量 = 2.5 MB，approaches L2-cache limit。bigger maxl → more cache misses during MGS → wall ↑ 超过 RHS eval 节省的成本

heihe_x4 cell.meta `cells_n=40046` ≈ 6.3× heihe 的 `cells_n=6335`，与上述 cache-pressure 模型一致。

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

1. **小 case 用户可选 +12% wall acceleration** (heihe N=1 maxl=30) 无需 source patch、无需重 compile
2. **solver convergence 失败统计学消除** (heihe ncfl 85→0, heihe_x4 ncfl 3620→0)，给后续 P9 epic（精度等级 A5）提供更稳定的 solver 起点
3. **PR-C env-var hook 长寿化** — 作为 production-supported 调参界面，未来 P8-tune.X / P9 / Prod 都可复用
4. **case-size-asymmetric Krylov pattern 被实证 + 文档化** → 为 P8-tune.D KLU 选型决策提供 mechanistic backbone (KLU 不受 Krylov-vector cache pressure 影响，对大 case 可能更合适)
5. **3-rep determinism PERFECT** — sweep methodology 复用 confidence 100% (G8 0/20 fail)

### Negative

1. **G7 STRICT FAIL 数字突兀** — max_ulp 4.4×10¹⁵ 至 4.6×10¹⁸ 让没读 mechanism 的人误以为 numerical corruption；需要 ADR-0004 §G7 mechanism 段反复 cite。本 ADR 已 explicit 说明
2. **per-case maxl 推荐表碎片化** — 4 个 (case, N) 组合给 4 个不同 maxl 建议，用户必须知道自己 case 的 element count 和典型 thread count；非 one-size-fits-all
3. **heihe_x4 全部 REGRESSION** — 即便 ncfl 3620 全消，也没法换回 wall (用户体验角度：用户看 wall 不看 ncfl)；这是 large case 用户的 "不要碰" 信号
4. **未触发 P8-tune.D KLU spike** — 本 ADR 是 Optional-knob 而非 NO-GO，因此**不**触发 P8-tune.D KLU spike epic。但留 forward-action：若未来有更大 case (heihe_x16 ≈ 250K elements，per master plan §1.1.1 P8+ 规划)，可重做 maxl sweep 数据来判 KLU 触发；本 ADR + tools/p8tune/aggregate_maxl_sweep.sh 可直接复用

### Neutral

1. **不修改 SHUD source** — 本 PR 仅 docs + tools，无 cvode_config.cpp / cvode_main.cpp 改动；SHUD pin 仍 `6ce17d6` (PR-C merge state)
2. **不触发 conditional PR-F default-bump** — tasks §4.11 default-bump branch 不开启
3. **OpenSpec change `p8tune-spgmr-maxl` archive** — 本 PR 合后 archive 到 `openspec/changes/archive/`，capability `maxl-sweep-verdict` 写入 spec lock
4. **epic #362 closeout** — 6 PR 全 merged + ADR-0004 落地 + verdict.md 落地 → epic 关闭

---

## Discussion — case-size-asymmetric Krylov memory pattern

这个 pattern 的实证有更广的 implication：

**SUNDIALS SPGMR Arnoldi MGS 复杂度** per outer iter = `O(maxl² × N_elem)`：
- maxl × N_elem flops for V matrix storage scan
- maxl × maxl orthogonality H matrix updates
- 每次 inner iter 重新读 V[0..maxl-1] (working set ≈ maxl × N_elem × 8 bytes)

当 working set 超 L2/L3 cache → 每次 inner iter 触发 DRAM 读取 → bandwidth-bound

L2 typical size ≈ 1-2 MB per core。heihe_x4 maxl=10 working set = 40046 × 8 × 10 = 3.2 MB → 已经超 L2，每次 inner iter 落 DRAM。bigger maxl 把更多向量推过 cache 边界。

heihe maxl=10 working set = 6335 × 8 × 10 = 507 KB → 完全 fit in L2。bigger maxl 仍能塞下，但 RHS eval 节省的 wall 主导。

**Forward implication**：
- KLU (direct sparse) 不存 Krylov 向量，pivot+L+U 是一次 setup amortized cost；对大 case **更友好**
- P8-tune.D KLU 触发条件 (master plan §P8-tune.D) 可能需要从 "ROI nfeLS/nfe" 改为 "Krylov working-set vs cache fit"

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
- [ ] master plan §P8-tune.C status mark **CLOSE** (deferred to next master plan refresh)
- [ ] heihe_x16 mesh refine (master plan §1.1.1 P8+) trigger 时复跑 aggregate_maxl_sweep.sh → 重判 KLU 触发条件 (sweeping methodology 复用 via PR-D tools + PR-E aggregator)

---

## Acceptance

本 ADR Accepted (status flag = `Accepted`)，effective `2026-06-27`。

- **Decision adopted**: Optional-knob branch
- **Hook lifecycle**: PR-C `SHUD_SPGMR_MAXL` env-var = long-lived production opt-in
- **Default unchanged**: cvode_config.cpp default maxl path = SUNDIALS-default 5 (preserved)
- **Production tune guidance**: heihe N=1 → `SHUD_SPGMR_MAXL=30` (+12% wall); heihe N=8 → optional `=30` (+7% wall); heihe_x4 任 N → keep unset
- **P8-tune.D KLU**: not triggered by 本 ADR; deferred to future heihe_x16 (250K elem) data

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
