## 工作情况说明（Merge 前）

- 关联 Issue：#346
- PR：#358
- 冻结提交（外层）：`291a0a4`（post Phase 6 outer_pin SHA 修正）
- 上游 Epic：#338 (p8pre-spike Step 2 PR-E)
- 前序 PR：#357（PR-D impl，5276167 identity precond binary）

### 背景与目标

p8pre-spike **Step 2 PR-E server-side identity-precond 18-cell spike**（依赖 #345 PR-D impl 落 identity precond stub 到 SHUD pin 5276167）。本 PR 完成 4 件事：

1. **3 个 fork tool scripts** — `tools/p8pre/{submit_identity_spike_template.sbatch, render_identity_spike.sh, run_identity_spike.sh}`，Step 1 PR-A 母本的 identity 变体（`.p8pre-runs/identity_spike/` subdir + `p8pre_identity_` 前缀 + Phase B.2 identity 3-symbol nm gate-1）
2. **服务器 build + gate-1 evidence** — gcc 13.3.0 + libgomp + libsundials 6 build exit 0；3-symbol nm verify（PSetupIdentity / PSolveIdentity / CVodeSetPreconditioner 全 ≥1）；Mode C nm gate 维持（Serial≥1 / OMP=0 / GOMP≥1）；gate-1 evidence 写入 `/scratch/.../identity_spike/server_nm.log`
3. **18-cell Slurm spike + rsync** — JID `9531..9548` singleton afterany chain per (case, N)，84 min 关键路径，18/18 COMPLETED ExitCode `0:0`；rsync 47.78 MB 到 Mac `/tmp/p8pre_identity_spike/`
4. **写 lightweight execution log** — `docs/p8pre/identity_spike_run.md` 191 行 NEUTRAL data capture（gate-1 PASS + 18-cell raw table + 6 (case, N) wall observation + cross-N invariance + ncfn observation + soft gate 6 ratio + references），verdict adjudication 留 PR-F #347

### 本次具体改动

| 文件 | 改动概要 |
|---|---|
| `tools/p8pre/submit_identity_spike_template.sbatch` (新) | Step 1 PR-A 母本 fork，`.p8pre-runs/identity_spike/` subdir |
| `tools/p8pre/render_identity_spike.sh` (新) | 18-cell render wrapper + `p8pre_identity_` job-name 前缀 |
| `tools/p8pre/run_identity_spike.sh` (新) | server runner + Phase B.2 identity 3-symbol nm gate-1 (~25 行 新增) |
| `docs/p8pre/identity_spike_run.md` (新, 191 行) | NEUTRAL data capture run log (9 §, YAML frontmatter, 18-cell raw table, observation tables) |

无 SHUD 源码改动（PR-D 已在 #357 完成），无 `.gitmodules` 改动，无 SHUD master pollution，无 `openspec/changes/` 改动，无 CI rule 改动，无 master plan / case_deployment_map 改动。

### 关键 evidence 与数据观察

**Gate-1 evidence**（server nm 3-symbol，PASS）：
```
                 U CVodeSetPreconditioner   ← imported from libsundials_cvode.so.6
0000000000021830 T PSetupIdentity            ← defined
0000000000021880 T PSolveIdentity            ← defined
```

**18-cell ExitCode**：全 `0:0`（COMPLETED）— gate 2 partial evidence ✓

**ncfn observation**（deterministic 跨 9 cells per case）：
- heihe: `ncfn=6`（all 9 cells, N=1/4/8 × 3 reps 都一样）
- heihe_x4: `ncfn=47`（all 9 cells, deterministic）

per spec L74-79 严格 gate 2 = `ncfn=0` → **PR-F #347 将 adjudicate gate 2 verdict** — 数据指向 NO-GO + design D8 fall-back PREC_NONE。Identity precond stub 不帮助 stiff Jacobian 的 SPGMR convergence（P^-1=I → 无实际预处理效果），propagate 到 CVODE Newton 出现 retries。这正是 P8-precond-0 spike 设计要验证的 ROI 假设。

**Cross-N CVODE invariance within identity run**（PASS）：
- heihe: nst=6599, nfe=6696 全 9 cells 一致 ✓
- heihe_x4: nst=6569, nfe=6775 全 9 cells 一致 ✓
- 切到 PREC_LEFT 后 CVODE shift（heihe nst Δ −99 / nfe Δ −247；heihe_x4 nst Δ −6 / nfe Δ +34）属预期（CVLS state machine 变化）

**Preliminary wall observation**（informational only，gate 4 verdict 留 PR-F）：
- delta_pct 范围 −2.64% 到 +1.09% vs Step 1 PR-A baseline
- 全 6 (case, N) groups 在 ε(heihe)=0.10 / ε(heihe_x4)=0.05 阈值内

**Soft gate 6 ratio**（data only）：
- `t_precond_setup / t_wall_total` 范围 6+ orders of magnitude 低于 5% 阈值
- 全 18 cells 远低于 budget

### Review 与修复闭环

- **Phase 0.5 fixture review**: SKIPPED (p8pre-spike change 已在 #339 PR #350 通过)
- **Phase 4 round 1 cross-review** (expanded, 4 parallel reviewers @ `2eb5d0f`):
  - `review-correctness`: APPROVE 0 findings（10 non-blocking notes，3-script fork 与 PR-A 母本 diff 干净 + Mac dry-run 18 sbatch lines + nm/build provenance/cell stats/cross-N invariance/ncfn 数据/soft gate 6 ratio/wall 6 行全独立复算一致）
  - `review-spec-compliance`: APPROVE 0 findings（11/11 checklist items PASS + 2 non-blocking notes，神经性 framing + heads-up about ncfn=6/47 deterministic 指向 PR-F NO-GO）
  - `review-integration`: APPROVE 1 Suggestion non-blocking（`.review-evidence/` gitignore，留 PR-F #347 加）+ 2 non-blocking notes（median sort algorithm cite + heihe_x4 N=1 cold→warm cache 14% wall variance）
  - `review-documentation`: **REQUEST CHANGES** 1 Critical（outer_pin SHA mismatch L6 + L30 stale `f800bb2`）+ 1 Suggestion non-blocking（§4 missing Elapsed+ExitCode columns）
- **Phase 4.5 verifier on Critical**: SKIPPED — self-evident factual error（`grep outer_pin` vs `git rev-parse HEAD`），verifier 在 plain-fact 上无附加 information value
- **Phase 6 orchestrator-direct fix @ `291a0a4`**:
  - L6 `outer_pin: f800bb2...` → `2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6`（doc-introducing commit per Step 1 PR-A convention）
  - L30 same update
  - 0 stale `f800bb` references after fix
  - §4 Elapsed+ExitCode Suggestion 留 non-blocking carry note 到 PR-F #347
- **Phase 6.5**: SKIPPED（Phase 7 Gap Sweep 会独立 examine the same lines）
- **Phase 7 final review** (Gap Sweep) @ `291a0a4`: **clean** — 11/11 AC PASS + Phase 6 fix verified（L6/L30 correct + 0 stale `f800bb` refs）+ oracle integrity PASS + CI 5/5 PASS + mergeStateStatus=CLEAN

### 兼容性、风险与已知限制

- 无 API 兼容性破坏（pure addition: 3 个 tool scripts + 1 个 doc）
- SHUD upstream `openmp-baseline` master 分支未触（C8 不污染）
- SHUD pointer 5276167 unchanged（PR-D #357 已 bump，PR-E 不改 SHUD 源码）
- **forward note (jok-mirror canonical)**: 残留 PR-D #357 forward note — spec L26 wording 修订留 #347/#349
- **forward note (ncfn 数据指向 NO-GO)**: per spec L74-79 严格 gate 2 = `ncfn=0`，PR-F #347 verdict adjudication 几乎确定走 NO-GO → ADR-0003 PREC_NONE fall-back（design D8）。这是 spike 设计要验证的 ROI 假设结果
- **forward note (3 non-blocking PR-F carries)**: §4 Elapsed+ExitCode columns / `.review-evidence/` gitignore / median sort algorithm cite

### 维护者关注点

- 无额外关注点。下一步 **#347 PR-F**：
  - 实现 `tools/p8pre/aggregate_identity_spike.sh` parse 18 cells
  - 计算 4 hard-gate verdict（gate 1 build PASS ✓ + gate 2 ncfn=0 → 大概率 FAIL + gate 3 nps/npe>0 PASS + gate 4 wall ε 范围内 PASS）+ 2 soft-gate（gate 5 SHA12 strict 几乎肯定 FAIL → max_ulp ≤ 1024 fallback + gate 6 ratio ≤ 0.05 PASS）
  - 写 academic `docs/p8pre/identity_spike_verdict.md`（gate 2 FAIL → Step 2 spike verdict = NO-GO + ADR-0003 draft input）
  - cite PR-D jok-mirror SUNDIALS canonical
