# P1d PR-L — P1d-tag annotated + baseline/P1d lock prep (D11 immutable, E′ closure path)

PR-L = tag-only docs PR per master plan v1.5 / M10 §6 P1d.5 + design D6 (P1d-tag annotated message) + design D11 (5 → 6 tag chain immutability)。本 PR 提交 docs 记录 P1d-tag 创建过程 + branch lock 程序 + D11 immutability 验证 baseline。实际 `git tag` + `gh api ... protection` 操作在 PR-L 合并后由 orchestrator 立即执行（post-merge SHA 即为 tag deref；PR-L commit 时 P1d-tag 尚未创建）。

## §1 Status

| Field | Value |
|---|---|
| PR-L Status (commit time) | **PENDING** (P1d-tag 待 PR-L 合并后 orchestrator 创建) |
| Tag name | `P1d-tag` |
| Tag type | annotated (per design D6 + D11) |
| Closure path | **E′ containment closure** (per master plan v1.5 / M10 §6 P1d.4) |
| Tag deref SHA | `<post-PR-L merge SHA>` (待 PR-L 合并后由 PR-M 填实) |
| SHUD pin (tag deref 下) | `210ac191...` (post-PR-G Kahan revert; `openmp-baseline` pushed; PR-L docs-only 不动 SHUD pointer) |
| 外层 baseline/P1d HEAD (tag 时刻) | `<post-PR-L merge SHA>` (= PR-K `3516c23` + PR-K post-K2 `158dd51` + PR-L commit) |
| baseline/P1d lock | deferred to **post-PR-M**（PR-M 还要 PROMOTE 2 spec + archive，提前 lock 会阻 PR-M merge） |
| Tag annotated message | per §3 下述（D6 5 required fields 全填实，9 项 per build_manifest §P1d-tag annotated message） |

## §2 P1c PARTIAL → P1d E′ closure narrative（design D6 field #1）

P1c epic 走 PARTIAL CLOSURE + P9 carve-out（per `openspec/specs/p1c-deterministic-reduction/spec.md` L100-L103）：8-site canonical-reduction Requirement CLOSED（4 helper functions 覆盖 10 line anchors → 8 logical sites），但 §4.4 A3a cross-N + §4.5 nst Δ=0 cross-N **carve-out 推 P9**（drift 不在 8 sites 内部，原 hypothesis "upstream parallel writer first-touch / NUMA-affinity governance"）。

P1d epic 接 P1c carve-out 作 hypothesis 验证：

- **P1d.1** NUMA env standardization（`OMP_PROC_BIND=close` + `OMP_PLACES=cores`，PR-B #276）
- **P1d.2** first-touch 治理（PR-C/C0/D/E，在 `MD_rhs_core.cpp` 的 `rhs_update` element + `rhs_flux` river + `rhs_update` lake 三 owner block 前置 steady-state first-touch loop）
- **P1d.3** Kahan revert（PR-G #281，去 P1c §4.7 Neumaier 1974 compensation，回到 naive `+=`，看 first-touch + B0 ordering 单独是否够稳）
- **P1d.3.2 PR-H 实测 verdict = FAIL**（详 §3 表）

PR-H FAIL 后两轮 GPT Pro 复查 + **5/5 codebase 事实核查**（per `docs/p1d/p1d_summary.md` §6）揭示**初版 4 个根因诊断全错**（详 §7 of summary）：

- SPGMR **没注册任何 preconditioner**（`SHUD/src/Equations/cvode_config.cpp:259`）
- `shud_omp` 当前实际跑的是 **Serial 水文 RHS + OpenMP N_Vector backend**（`f.cpp:54` 调 `ExecPolicy::Serial`，`MD_rhs_core.cpp:802-811` 三 case `std::abort()` 桩）
- PR-C/D/E steady-state first-touch loops 是为**完全没发生的** parallel RHS owner-compute 做的页面预放置——consumer 是单线程，根本无视 NUMA locality → **infrastructure 保留但 M10 标 DEPRECATED**
- 真正的 cross-N divergence 根因 = `NVECTOR_OPENMP` 的 `N_VDotProd_OpenMP` + `N_VWSqrSumLocal_OpenMP` 用 `reduction(+:sum) schedule(static)`，**跨 N reduction tree 顺序不固定**

故 P1d 不走原 plan capstone，走 **E′ containment closure path**（per master plan v1.5 / M10 §6 P1d.4 + `docs/p1d/p1d_summary.md` §5）：保留全部 P1d 工程结果 + 4-mode spec rewrite 把 strict 承诺限定到正确 mode + 把 P1e (F 路) 作为下个 epic 把 "真正应并行的 RHS 还没并行" 这件事补上。

## §3 Tag annotated message 草拟（PR-L orchestrator post-merge 使用）

orchestrator 在 PR-L 合并后用以下命令创建 tag（见 §4 完整命令）；message 体即下述代码块全文（不含外层 ``` 围栏）：

```
P1d epic capstone — E' containment closure

Status: PARTIAL CLOSURE via E' path (master plan v1.5 / M10 §6 P1d.4)

Why E' over original goal:
- PR-H 8-cell empirical revealed cross-N divergence root cause is
  NVECTOR_OPENMP N_VDotProd_OpenMP reduction(+:sum) schedule(static),
  not the owner-gather helpers nor any SPGMR preconditioner.
- Codebase fact-check: shud_omp actually runs Serial RHS + OpenMP
  N_Vector backend; ExecPolicy::StrictOMP path is std::abort() stub
  in MD_rhs_core.cpp:802-811.
- PR-C/D/E steady-state first-touch loops are setup for owner-compute
  that doesn't happen (consumer is single-thread) → infrastructure
  preserved but tagged DEPRECATED.

Delivered:
1. NUMA env standardization (PR-B): OMP_PROC_BIND=close + OMP_PLACES=cores
2. PR-C/D/E first-touch loops (DEPRECATED per E' closure, retained as
   historical artifact + P1e (F path) redesign reference)
3. PR-G Kahan revert (SHUD 3a0004c → 210ac19): clean revert proven
   via Mac 9-SHA matrix (post-PR-G == pre-K2 byte-identical)
4. Master plan v1.5 / M10 sync revision: +180 / -7 lines additive
5. 4-mode spec rewrite (serial / strict-omp / det-omp / fast-omp)
6. PR-K capstone docs (4 new docs + 2 updates + deprecation note)
7. ADR-0002 (solver-path) 4-way comparison: Path 1 SELECTED for P1e

3 SHALL gate verdict (PR-H 8-cell server, 90-day):
- L123 Kahan revert canonical heihe N=1 SHA byte-identical: PASS
- L130 A3a bitwise cross-N: FAIL (3 distinct SHA / case)
- L139 nst Δ + ladder: FAIL (heihe Δ=80@N=4 / 152@N=8;
  heihe_x4 Δ=11@N=4 / 4@N=8)
- L145 N=1 reverse-compat: PARTIAL (server heihe partial)

Forward handoff (P1e = F path next epic):
- Serial N_Vector + StrictOMP RHS (replace abort stubs)
- 2x2 build matrix causal experiment mandatory before code change
- Reuse rhs_deterministic_gather infrastructure
- Delete steady-state first-touch; preserve allocation-time
- ADR-0002 Path 1 SELECTED per docs/adr/0002-solver-path.md

SHUD pin trail:
- P1c era: 3a0004c (Kahan IN, 2026-06-22)
- P1d era: 210ac19 (PR-G Kahan revert + PR-C/D/E first-touch
  stacked, 2026-06-23)
- P1d-tag pins outer baseline/P1d HEAD = (filled at tag-create
  time post-merge) with SHUD submodule pinned at 210ac19

D11 6-tag chain: B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag /
P1c-tag / P1d-tag (this tag, NEW). Historical 5 SHA immutable.

References:
- master plan v1.5 / M10 §6 P1d + §6 P1e (M10 sync revision)
- docs/p1d/ {summary, perf_baseline, numa_root_cause,
  first_touch_design (DEPRECATED note), pr_h_final_run,
  pr_i_p1_update_omp_reference, kahan_revert, numa_env_runbook,
  pr_f_intermediate_run, report, tag_and_lock} (11 docs)
- docs/adr/0002-solver-path.md (4-way solver evaluation)
- openspec/changes/p1d-numa-governance/ (PROMOTE pending PR-M)
- openspec/changes/p1e-strict-omp-rhs/ (local drafts, P1e launch)
- PR chain: #292..#302 (PR-C0/C/D/E/F/G/H/I, M10, K, ADR-0002)
- Epic: #274 (P1d, closes via PR-M after PROMOTE + archive)
```

> 该 message 同时满足 design D6 的 5 必填字段（P1c → P1d narrative / NUMA + first-touch + Kahan 三 phase stack / 3 SHALL gate verdict 含实测 SHA + 数字 / D11 5→6 chain immutability baseline / SHUD pin trail）+ build_manifest §"P1d-tag annotated message" L479-L491 的 9 项扩展（containment closure / 5/5 fact-check / 4-mode rewrite / first-touch deprecation note / Kahan revert 保留 / 指向 P1e + ADR-0002 / SHUD pin 变更 / 11 PR cross-ref / D11 historical immutability re-verify）。

### §3.1 PR-H 实测数据（message 内引用，data source）

- **L123 Kahan revert canonical**：`heihe N=1` SHA = `7f22bd6faa438d509399ecc8c0f587accb4f72ea31de76fd3c5f58099e8d4471`（== spec L123 canonical 前缀 `7f22bd6faa438d50...`，byte-identical 全 64-hex）→ PASS
- **L130 A3a bitwise cross-N**：heihe N=1≡N=2 ≠ N=4 ≠ N=8（3 distinct SHA / case）；heihe_x4 同样 3 distinct → FAIL
- **L139 nst Δ + ladder**：heihe Δ=0 strict ⇒ N=4 Δ=80 / N=8 Δ=152；heihe_x4 \|Δ\|≤2 ladder ⇒ N=4 Δ=11 / N=8 Δ=4 → FAIL
- **L145 N=1 reverse-compat**：spec 仅预写 heihe N=1 canonical `7f22bd6f...`，server heihe PASS；heihe_x4 N=1 SHA `55403bef48ee5ad8...` 无预写 reference → PARTIAL

详 `docs/p1d/p1d_pr_h_final_run.md` Verdict + SHA matrix + nst stats section（本 doc 不重复 8-cell wall + speedup 表）。

### §3.2 SHUD pin trail（message 内引用，data source）

| 阶段 | SHUD commit | 说明 |
|---|---|---|
| P1c-tag era | `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` | Kahan IN (P1c §4.7 conditional Neumaier 注入 4 reduction helpers) |
| P1d PR-C/D/E mid-stream | `de9545d` → `a2085de` → `7023ee9` → `6aada88` | first-touch element / river / lake 三 loop 叠 |
| P1d-tag (post-PR-G) | **`210ac191...`** (最终 P1d-tag pin) | PR-G Kahan revert（4 surgical revert in `MD_rhs_core.cpp`：去 `#include <cmath>` + 3 helpers Neumaier branch，净 +7/-47 vs `6aada88`）；first-touch 保留 byte-identical；Mac 9-SHA matrix 证明 revert clean（per `docs/p1d/p1d_kahan_revert.md` §"SHA matrix"） |

P1d-tag deref commit 时刻 `git ls-tree P1d-tag SHUD` 应输出 `160000 commit 210ac191...`。

## §4 P1d-tag 创建命令（PR-L orchestrator post-merge 立即执行）

```bash
# Step 1 — sync local + capture post-PR-L SHA
cd /Users/danker/Desktop/Hydro-SHUD/openMP
git fetch origin baseline/P1d
git checkout baseline/P1d
git pull --ff-only origin baseline/P1d
POST_PR_L_SHA=$(git rev-parse HEAD)
echo "Tag will be at: ${POST_PR_L_SHA}"

# Step 2 — write message to /tmp/p1d-tag-msg.txt（§3 message 完整文本）
# orchestrator 把 §3 代码块全文（不含外层 ``` 围栏）写入 /tmp/p1d-tag-msg.txt

# Step 3 — create P1d-tag annotated
git tag -a P1d-tag ${POST_PR_L_SHA} -F /tmp/p1d-tag-msg.txt

# Step 4 — push tag
git push origin P1d-tag

# Step 5 — verify tag-object SHA + deref commit SHA
git rev-parse P1d-tag           # → tag-object SHA (annotated 新生成)
git rev-parse P1d-tag^{}        # → POST_PR_L_SHA (= deref commit)
git show P1d-tag^{}:SHUD | head -1 || git ls-tree P1d-tag SHUD
# expected: 160000 commit 210ac191... (SHUD submodule pin)
```

## §5 D11 5 → 6 tag chain immutability re-verification baseline

D11 (design D11 "5 historical tag SHA 永不变 + P1d-tag 仅追加 = 6 chain") 验证：post-PR-L 创建 P1d-tag 后跑下述命令，5 historical SHA 必须与本节 `pre-PR-L` 列字面一致。

```bash
git rev-parse B1-tag B1a-tag B1b-tag P1-update-omp-tag P1c-tag P1d-tag
# Expected: 6 SHA outputs; first 5 immutable from P1c epic; P1d-tag new
```

| Tag | SHA (PR-L commit time, 2026-06-24 实测) | 状态 |
|---|---|---|
| `B1-tag` | `0c0621c986e54e371c5a176850d1eb981150010e` | ✓ historical immutable |
| `B1a-tag` | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | ✓ historical immutable |
| `B1b-tag` | `96e224daad8cb9c93f855851724f8d45468391c2` | ✓ historical immutable |
| `P1-update-omp-tag` | `ff21c75c8e968d5e47ca53b015425360be9ac879` | ✓ historical immutable |
| `P1c-tag` | `1da5eb9734680fc61e68f6091964c38fc5f67c6f` | ✓ historical immutable |
| `P1d-tag` | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` | ✓ NEW (PR-L post-merge, 6th chain) |

> 注：本节 5 historical SHA 取自 PR-L commit 时刻 `git rev-parse <tag>` 实测输出（顺序 `B1 / B1a / B1b / P1-update-omp / P1c`）。任一 SHA 改变 → D11 immutability 违反，orchestrator abort + git tag --verify + 回滚 force-tag-update（per D11 NG3 + NG7 rule）。post-tag 验证由 PR-M `docs/p1d/p1d_summary.md` §13 表 §13.2 填入 PR-M 验证时刻 SHA，对比 PR-L 列即可。

### §5.1 SHA correction note (PR-M fix)

**PR-L 阶段记录错位 → PR-M 校正**：PR-L agent 录入 §5 表时把 6 行 SHA 的 SHA 值错位放置（off-by-one：B1-tag 槽位上写了 P1c-tag SHA，B1a-tag 槽位上写了 B1-tag SHA，等等）。PR-M post-merge 时刻跑 `git rev-parse B1-tag B1a-tag B1b-tag P1-update-omp-tag P1c-tag P1d-tag` 实测输出为：

```
0c0621c986e54e371c5a176850d1eb981150010e  # B1-tag
f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d  # B1a-tag
96e224daad8cb9c93f855851724f8d45468391c2  # B1b-tag
ff21c75c8e968d5e47ca53b015425360be9ac879  # P1-update-omp-tag
1da5eb9734680fc61e68f6091964c38fc5f67c6f  # P1c-tag
a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0  # P1d-tag (NEW)
```

PR-M 已用实测数据替换 §5 表（上方修订版本）。**D11 historical immutability 未违反** — 5 historical SHA 与 `docs/p1c_tag_and_lock.md` 同期记录一致；仅 PR-L 录入位置错位，SHA 值本身正确。`git tag --verify` 全 6 tag 通过。

## §6 baseline/P1d branch lock 程序（deferred to post-PR-M per issue #287）

baseline/P1d branch 不在 PR-L 阶段 lock。Rationale: PR-M PROMOTE 2 specs (`p1d-numa-governance` + `p1d-capstone`) + archive change 仍需 merge 到 `baseline/P1d`，若 `lock_branch=true` 提前生效则 PR-M 无法 merge（per `docs/p1c_tag_and_lock.md` §4 同一 deferred 逻辑）。

Lock 命令将在 **PR-M 合并后** 立即执行：

```bash
# Step 1 — confirm PR-M merged + baseline/P1d HEAD 是 PR-M 合并 SHA
gh pr view <PR-M-number> --repo DankerMu/SHUD-OpenMP --json mergedAt --jq '.mergedAt'

# Step 2 — apply branch protection (lock_branch=true, enforce_admins=true,
# allow_force_pushes=false, allow_deletions=false)
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1d/protection \
  --method PUT \
  --field lock_branch=true \
  --field enforce_admins=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_pull_request_reviews=null \
  --field required_status_checks=null \
  --field restrictions=null

# Step 3 — verify lock active
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1d --jq '.protection'
# expect: lock_branch.enabled=true + enforce_admins.enabled=true +
# allow_force_pushes.enabled=false + allow_deletions.enabled=false
```

## §7 main fast-forward 程序（deferred to post-PR-M per issue #287）

post-PR-M lock 完成后，把 `baseline/P1d` HEAD fast-forward 到 `main`（与 P1c 同 pattern；main 是 GitHub default，PR base 走 main 时 `Closes #N` 关 issue 才正常）：

```bash
# 前置：local main 已 up-to-date
git fetch origin main baseline/P1d
git rev-parse origin/main origin/baseline/P1d

# fast-forward push (orchestrator 执行)
git push origin baseline/P1d:main

# verify
git rev-parse origin/main
git rev-parse origin/baseline/P1d
# expected: 两 SHA 相等
```

## §8 Hand-off → PR-M

PR-L 合并 + post-merge orchestrator 完成 §4 tag 创建后：

- ✓ P1d-tag annotated 存在，不可变（per D11 + NG7）
- ✓ baseline/P1d HEAD **未 lock**（允许 PR-M merge）
- ✓ D11 historical 5 tags SHA 不变（详 §5 表）
- → **PR-M (issue #287)** PROMOTE 2 specs (`p1d-numa-governance` + `p1d-capstone`) + archive change to `openspec/changes/archive/2026-06-24-p1d-numa-governance/` + glossary 4 新术语 + jsonl 双追加（`stage-pipeline-log.jsonl` + post-stage-cleanup if any）+ Epic #274 close + 填实 `docs/p1d/p1d_summary.md` §13 placeholder（5 表的 TBD (PR-M) 单元格）+ propose `p1e-strict-omp-rhs` openspec change
- → **Final task** (per `docs/p1c_tag_and_lock.md` §7 同 pattern): post-PR-M lock baseline/P1d + D11 verify post-lock + main fast-forward (per §6 + §7)

## §9 References

| 文档 | 内容 |
|---|---|
| 本文件 (`docs/p1d/p1d_tag_and_lock.md`) | P1d-tag annotated procedure + branch lock + D11 6-chain immutability baseline |
| `docs/p1d/p1d_summary.md` | P1d capstone source of truth（§13 PR-M 填实 P1d-tag 验证表） |
| `docs/p1d/p1d_pr_h_final_run.md` | PR-H final 8-cell verdict + SHA matrix + nst stats（§3.1 data source） |
| `docs/p1d/p1d_kahan_revert.md` | PR-G SHUD Kahan revert + Mac 9-SHA matrix（§3.2 SHUD pin trail data source） |
| `docs/p1c_tag_and_lock.md` | P1c-tag procedure template（本 doc 仿照其结构 + §6/§7 deferred lock 同 pattern） |
| `docs/build_manifest.md` §"P1d-tag preparation" L431-L491 | SHUD pin trail（§3.2 data source）+ annotated message 9 required items（§3 完整 message 草拟） |
| `openspec/changes/p1d-numa-governance/specs/p1d-numa-governance/spec.md` L175-L200 | "baseline/P1d 分支 + P1d-tag" Requirement + 3 Scenarios（branch creation / annotated 创建 / branch lock）+ D11 5→6 immutability Scenario |
| `openspec/changes/p1d-numa-governance/design.md` D6 + D11 | annotated message 5 required fields + 6-tag chain immutability + NG3/NG7 rule |
| `SHUD_openMP_master_plan.md` v1.5 / M10 §6 P1d.5 + §6 P1d.6 | baseline lock + tag (D11 6-tag chain) + Go/No-Go → P1e |
