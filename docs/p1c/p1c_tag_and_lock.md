# P1c PR-L — P1c-tag annotated + baseline/P1c lock prep (D11 immutable)

PR-L = tag-only PR per master plan §6 P1c.5 stage hand-off。本 PR 提交 docs 记录 P1c-tag 创建过程 + branch lock 程序 + D11 immutability 验证 baseline。实际 `git tag` + `gh api ... protection` 操作在 PR-L 合并后立即执行 (post-merge SHA 即为 tag deref)。

## §1 P1c-tag 元数据

| Field | Value (pre-merge plan) |
|---|---|
| Tag name | `P1c-tag` |
| Tag type | annotated (per design D11) |
| Tag deref SHA | `<post-PR-L merge SHA>` (待 PR-L 合并后填) |
| SHUD pin (tag deref 下) | `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` (Kahan-injected, openmp-baseline pushed; 不变 post-PR-L, 因 PR-L 是 docs-only 不改 SHUD pointer) |
| 外层 baseline/P1c HEAD (tag 时刻) | `<post-PR-L merge SHA>` |
| Tag annotated message | per §2 下述 |

## §2 Tag annotated message (草拟)

```
P1c deterministic-reduction + capstone (2026-06-23)

8-site canonical-reduction Requirement CLOSED:
- 4 helper functions (fixed_pairwise_sum_indexed / fixed_pairwise_sum_range /
  fixed_leftfold_sum_indexed / fixed_leftfold_sum_pair_indexed) cover 10
  line anchors → 8 logical sites (PR-B/C/D/E #258/#259/#260/#261)
- Helper-wrap bitwise-equivalent at NUM_OPENMP=1 (server PR-J §2 实证):
  P1 era N=1 SHA `7f22bd6f...` ≡ P1c pre-Kahan N=1 SHA `7f22bd6f...` (heihe)
- 3 negative grep gates PASS (新宏 0 / schedule(dynamic|guided) 0 /
  #pragma omp atomic 0)

Kahan/Neumaier conditional path APPLIED (per spec L107-L128):
- Kahan held-in-reserve patch (docs/p1c/p1c_kahan_patch.diff PR-G #263) applied
  at PR-I (#265) on §4.7 trigger fire (PR-H #264 8-cell A3a + nst FAIL)
- 4 helpers gain Neumaier 1974 (Kahan-Babuška variant) compensation
- SHUD pin: de9545d → 3a0004c (openmp-baseline pushed; master untouched)
- nst delta improvement: heihe |Δ| 225 → 84 (~63%); heihe_x4 |Δ| 3 → 5
  (within noise)

PARTIAL CLOSURE + P9 carve-out (per master plan §3 fallback option 2):
- §4.4 A3a cross-N + §4.5 nst Δ=0 cross-N: CARVE-OUT 推 P9 stage (upstream
  parallel writer first-touch / NUMA-affinity governance)
- design D9 decision branch 2 CONFIRMED: drift origin OUTSIDE 8 sites

Reverse-compat documented (PR-J #266):
- NUM_OPENMP=1 binary-level reverse-compat NOT preserved at Kahan-injected
  baseline/P1c HEAD (Neumaier changes acc order even at serial)
- D11 tag-immutability preserved (P1-update-omp-tag annotated tag SHA
  unchanged)

D11 immutability verification baseline:
- P1-update-omp-tag SHA: ff21c75c8e968d5e47ca53b015425360be9ac879 (unchanged)
- B1-tag SHA: 0c0621c986e54e371c5a176850d1eb981150010e (unchanged)
- B1a-tag SHA: f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d (unchanged)
- B1b-tag SHA: 96e224daad8cb9c93f855851724f8d45468391c2 (unchanged)

Sources of truth:
- docs/p1c/p1c_summary.md §1-§10 (capstone)
- docs/p1c/p1c_a3a_root_cause.md §"design D9 decision branch 判定" (终判)
- docs/p1c/p1c_perf_baseline.md §2-§6 (perf + D7 reframing)
- docs/p1c/p1c_pr_h_server_first_run.md (8-cell pre-Kahan)
- docs/p1c/p1c_pr_i_kahan_injection.md (8-cell Kahan + carve-out 决策)
- docs/p1c/p1c_pr_j_reverse_compat.md (3-layer matrix)
- docs/p1c/p1c_kahan_patch.diff (held-in-reserve patch, applied at PR-I)
```

## §3 Tag 创建命令 (post-PR-L 合并立即执行)

```bash
# Step 1 — sync local + capture post-PR-L SHA
cd /Users/danker/Desktop/Hydro-SHUD/openMP
git fetch origin baseline/P1c
git checkout baseline/P1c && git pull --ff-only
POST_PR_L_SHA=$(git rev-parse HEAD)
echo "Tag will be at: ${POST_PR_L_SHA}"

# Step 2 — create P1c-tag annotated
git tag -a P1c-tag ${POST_PR_L_SHA} -F <(cat <<'EOF'
P1c deterministic-reduction + capstone (2026-06-23)

[§2 message 完整内容]
EOF
)

# Step 3 — push tag
git push origin P1c-tag

# Step 4 — verify (D11 immutability check)
git rev-parse P1c-tag           # → tag SHA (新生成)
git rev-parse P1c-tag^{commit}  # → POST_PR_L_SHA (= deref)
git ls-tree P1c-tag SHUD        # → 160000 commit 3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4

# Step 5 — D11 historical immutability re-verify (4 tags 不变)
for tag in P1-update-omp-tag B1-tag B1a-tag B1b-tag; do
  printf "%s -> " $tag
  git rev-parse $tag
done
# Expected output (compare against §1 PR-L pre-merge plan + §2 annotated message):
#   P1-update-omp-tag -> ff21c75c8e968d5e47ca53b015425360be9ac879
#   B1-tag -> 0c0621c986e54e371c5a176850d1eb981150010e
#   B1a-tag -> f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d
#   B1b-tag -> 96e224daad8cb9c93f855851724f8d45468391c2
```

## §4 baseline/P1c branch lock 程序 (deferred to post-PR-M)

baseline/P1c branch 不在 PR-L 阶段 lock。Rationale: PR-M (#256) PROMOTE 2 specs + archive change 仍需 merge 到 baseline/P1c, 若 lock_branch=true 提前生效则 PR-M 无法 merge。

Lock 命令将在 PR-M 合并后立即执行 (作 Final task per task list #15):

```bash
# Step 1 — confirm PR-M merged + baseline/P1c HEAD 是 PR-M 合并 SHA
gh pr view 268-OR-NEXT --repo DankerMu/SHUD-OpenMP --json mergedAt --jq '.mergedAt'

# Step 2 — apply branch protection (lock_branch=true, enforce_admins=true,
# allow_force_pushes=false, allow_deletions=false)
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c/protection \
  --method PUT \
  --field lock_branch=true \
  --field enforce_admins=true \
  --field allow_force_pushes=false \
  --field allow_deletions=false \
  --field required_pull_request_reviews=null \
  --field required_status_checks=null \
  --field restrictions=null

# Step 3 — verify lock active
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c --jq '.protection'
# expect: lock_branch.enabled=true + enforce_admins.enabled=true +
# allow_force_pushes.enabled=false + allow_deletions.enabled=false
```

## §5 D11 immutability 验证 baseline

D11 (design D11 "tag SHA 永不变 + branch lock 不允许 force-push") 在 P1c 完成时对以下 tags 仍生效:

| Tag | SHA (P1c 完成 pre-PR-L 时刻验证, 应与 P1 era / B1 / B1a / B1b 完成时 SHA 完全一致) | 状态 |
|---|---|---|
| P1-update-omp-tag | `ff21c75c8e968d5e47ca53b015425360be9ac879` | ✓ UNCHANGED |
| B1-tag | `0c0621c986e54e371c5a176850d1eb981150010e` | ✓ UNCHANGED |
| B1a-tag | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | ✓ UNCHANGED |
| B1b-tag | `96e224daad8cb9c93f855851724f8d45468391c2` | ✓ UNCHANGED |
| P1c-tag | `<TBD post-PR-L>` | NEW |

PR-L 合并后立即 verify:
```bash
for tag in P1-update-omp-tag B1-tag B1a-tag B1b-tag; do
  echo "$tag -> $(git rev-parse $tag)"
done
```

任一 SHA 改变 → D11 immutability 违反, abort + git tag --verify + 回滚 force-tag-update (per D11 NG3 rule).

## §6 baseline/P1c branch lock 矩阵 (待 PR-M post)

post-PR-M post-lock 验证:

```bash
gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c --jq '.protection.lock_branch.enabled, .protection.enforce_admins.enabled, .protection.allow_force_pushes.enabled, .protection.allow_deletions.enabled'
```

期望 4 行: `true / true / false / false`.

## §7 Hand-off → PR-M

PR-L 合并 + 立即 tag 创建后:
- ✓ P1c-tag annotated 存在, 不可变 (per D11)
- ✓ baseline/P1c HEAD 未 lock (允许 PR-M merge)
- ✓ D11 historical 4 tags SHA 不变
- → PR-M (#256) PROMOTE 2 specs (p1c-deterministic-reduction + p1c-capstone) + archive change to `openspec/changes/archive/2026-06-23-p1c-deterministic-reduction/` + glossary 4 新术语 + jsonl 双追加 (stage-pipeline-log.jsonl + post-stage-cleanup if any) + Epic #243 close
- → Final task (#15): lock baseline/P1c + D11 verify post-lock
