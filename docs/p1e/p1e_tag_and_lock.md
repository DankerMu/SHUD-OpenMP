# P1e-tag annotated procedure + `baseline/P1e` lock (PR-L EXECUTED)

> **Status (PR-L 2026-06-25)**: PR-L 已执行 `git tag -a P1e-tag` + `git push origin P1e-tag` + `baseline/P1e` branch protection lock 流程，本 doc 实填 SHA + tag annotated message。前置 PR-K capstone 已 merge（HEAD `11687b7`）。
>
> per `openspec/changes/p1e-strict-omp-rhs/tasks.md` §5.1–§5.3: `[PR-L]` 三步全 DONE。

## §1 PR-K 占位说明

本 doc 在 PR-K capstone 内做存在性占位（per spec p1e-capstone L23 `p1e_tag_and_lock.md: P1e-tag annotated procedure + baseline/P1e lock` 必备项）。PR-L 内的实际 deliverables：

1. `git tag -a P1e-tag -m '<annotated message body>'` (本 doc §3 annotated message body)
2. `git push origin P1e-tag`
3. `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1e/protection --method PUT --field lock_branch=true ...` (lock baseline/P1e)
4. amend 本 doc + `docs/p1e/p1e_summary.md` §10 填实 tag-object SHA + deref commit SHA (per tasks §5.4 + §7.22)
5. D11 final 7-tag chain verify (per `docs/p1e/p1e_report.md` §9)

## §2 P1e-tag deref commit 确定

per tasks §5.3：

- PR-K 合并 + PR-L 启动时，`baseline/P1e` HEAD = `11687b756dd53bb634df391bcbeb64b3cef5c750`（PR-K capstone post-merge review-loop-log entry）
- `P1e-tag` deref = `11687b756dd53bb634df391bcbeb64b3cef5c750` (pre-PR-M PROMOTE, 仿 P1d 模式避免 amend 循环)
- PR-M PROMOTE 后 baseline/P1e HEAD 会前进，但 `P1e-tag` 不更新（D11 immutability）

## §3 P1e-tag annotated message body (PR-L 创建时 -m 内容)

per `docs/build_manifest.md` "P1e-tag annotated message（PR-L 草拟，PR-K 仅引用）" 节 + `docs/p1e/p1e_summary.md` §9 PR-L + `docs/p1e/p1e_report.md` §10 forward handoff to P2a：

```
P1e epic capstone — SHIP via §4.6.2 partial-closure

ADR-0002 Path 1 (Serial NVec + StrictOMP RHS) Implemented (P1e epic close, 2026-06-25).

3 SHALL gate verdict:
  AC-S1 mode C cross-N bitwise: PASS (heihe a2023ccd2de4 + heihe_x4 b5e4b0a2cf83, unique SHA = 1)
  AC-S2 mode C SHA == mode A reference SHA: PASS (6-case roll-up Mac + server)
  AC-S3 D7 per-case speedup: PARTIAL (heihe sp@8 1.066× < 1.3× FAIL, heihe_x4 sp@8 1.729× ≥ 1.5× PASS)
  AND-gate (BOTH FAIL 触发 D12.3) 不满足 → §4.6.2 partial-closure → user 决策 SHIP

D12 4 branch eval: D12.1/.2/.3/.4 全 NOT triggered (per docs/p1e/p1e_2x2_verdict.md §6.2)
Active path: §4.6.2 partial-closure → SHIP (heihe small-case OMP overhead floor carve-out per docs/p1e/p1e_perf_baseline.md §6)

SHUD pin trail:
  P1d-tag deref: 210ac191... (Kahan revert + first-touch loops, 全 mode Serial RHS)
  P1e-tag deref: 3341368d2d0854924d2286925c8575df52cc97a0 (ExecPolicy::StrictOMP + -fopenmp wire + SHUD_RHS_THREADS env + 3 first-touch removal + omp single→omp for)

Build matrix (P1d 2 build → P1e 4 mode):
  A: shud (Serial NVec + Serial RHS) — canonical reference
  B: shud_omp (OpenMP NVec + Serial RHS) — 历史 prod
  C: shud SHUD_ENABLE_OPENMP_RHS=1 (Serial NVec + StrictOMP RHS) — P1e production SHIP
  D: shud_omp SHUD_ENABLE_OPENMP_RHS=1 (OpenMP NVec + StrictOMP RHS) — research deferred

Production default: SHUD_RHS_THREADS per case (heihe `=1` carve-out / heihe_x4 `=4` 推荐).

14-PR cross-ref:
  PR-A #309  audit
  PR-B #310  2×2 runner
  PR-B0 #311 rivqdown recompute helper
  PR-C #312  Mac Phase 1 mode A/B
  PR-D #313  server Phase 1 mode A/B
  PR-E #314  Phase 1 verdict
  PR-F #315  ExecPolicy::StrictOMP impl
  PR-G #315  -fopenmp wire + SHUD_RHS_THREADS env
  PR-H #316  first-touch removal + omp single→omp for
  PR-I #317  server SHALL closure (24 cell)
  PR-J #318+#333  Mac N=1 reverse-compat
  PR-K #319  capstone docs/p1e/ ≥14 + spec amend + ADR-0002 close-out
  PR-L #<TBD>  P1e-tag + baseline/P1e lock (本 PR)
  PR-M #<TBD>  PROMOTE 2 spec + glossary 4 new terms + epic close

ADR-0002 Status: Implemented (P1e epic close, 2026-06-25)
4-mode strict-omp 列: SHIP via §4.6.2 partial-closure (P1e 验收后)

D11 historical immutability re-verify:
  B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag SHAs 全部不变
  P1e-tag 新增完成 7-tag chain

References:
  docs/p1e/p1e_summary.md
  docs/p1e/p1e_perf_baseline.md
```

PR-L 创建 tag 时按上述模板 -m，amend 实际 SHA + 完成创建步骤。

## §4 baseline/P1e lock 步骤 (PR-L 已实施 2026-06-25)

**注**：GitHub branch-protection API 对 nullable object field 要求 literal JSON `null`，不接受 `gh -f / -F` string-encoded。改用 `--input -` stdin JSON：

```bash
cat <<'JSON' | gh api -X PUT /repos/DankerMu/SHUD-OpenMP/branches/baseline%2FP1e/protection --input -
{
  "required_status_checks": null,
  "enforce_admins": true,
  "required_pull_request_reviews": null,
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "lock_branch": true
}
JSON
```

验证 lock（实地输出 2026-06-25）：

```bash
$ gh api repos/DankerMu/SHUD-OpenMP/branches/baseline%2FP1e/protection \
    --jq '{lock_branch: .lock_branch.enabled, enforce_admins: .enforce_admins.enabled, allow_force_pushes: .allow_force_pushes.enabled, allow_deletions: .allow_deletions.enabled}'
{"allow_deletions":false,"allow_force_pushes":false,"enforce_admins":true,"lock_branch":true}
```

`baseline/P1e` 自此与 `baseline/B1a`（CLAUDE.md "已 lock_branch=true"）同 lock 等级。

## §5 D11 final 7-tag chain verify (PR-L 完成后)

per `docs/p1e/p1e_report.md` §9：

| tag | 期望 SHA | 验证 |
|---|---|---|
| B1-tag | `0c0621c986e54e371c5a176850d1eb981150010e` | `git rev-parse B1-tag` 与 P1d era 一致 PASS |
| B1a-tag | `f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d` | 同上 PASS（CLAUDE.md `f7f992c` 是 short prefix of pre-tag commit；annotated tag object SHA 不同） |
| B1b-tag | `96e224daad8cb9c93f855851724f8d45468391c2` | 同上 PASS |
| P1-update-omp-tag | `ff21c75c8e968d5e47ca53b015425360be9ac879` | 同上 PASS |
| P1c-tag | `1da5eb9734680fc61e68f6091964c38fc5f67c6f` | 同上 PASS |
| P1d-tag | `a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0` | 同上 PASS |
| **P1e-tag** | annotated object **`25023eff32d1fa317b045cbc786f379fac9e522c`** / deref commit `11687b756dd53bb634df391bcbeb64b3cef5c750` | **PR-L 已 `git tag -a` + `git push origin P1e-tag` DONE 2026-06-25** |

> PR-L 执行后 D11 immutability 6 tag SHA 经 `git rev-parse` 实地比对，全 PASS（pre-existing tag objects 未受 PR-L 操作影响）。P1e-tag 7-tag chain 完成。
>
> §5.4 PR-M amend `docs/p1e/p1e_summary.md` §"验证 P1e-tag" 章节直接引用以上两个 SHA（不再需要重新 `git rev-parse`）。

```bash
# PR-L 完成后 verify 全 7 tag (含 P1e-tag 新增):
for t in B1-tag B1a-tag B1b-tag P1-update-omp-tag P1c-tag P1d-tag P1e-tag; do
  echo "$t: $(git rev-parse $t)"
done
```

实地输出（2026-06-25 PR-L 完成后）：

```
B1-tag: 0c0621c986e54e371c5a176850d1eb981150010e
B1a-tag: f3a7ff1efe20c94de2fda73a17d74fb3a0016c1d
B1b-tag: 96e224daad8cb9c93f855851724f8d45468391c2
P1-update-omp-tag: ff21c75c8e968d5e47ca53b015425360be9ac879
P1c-tag: 1da5eb9734680fc61e68f6091964c38fc5f67c6f
P1d-tag: a82bf3361b5e4dcbc1f07ca22e99a917b00b78f0
P1e-tag: 25023eff32d1fa317b045cbc786f379fac9e522c
```

7-tag chain 完整、D11 immutability 6 historical tag 全 PASS、P1e-tag 新增 DONE。

## §6 References

- spec `p1e-capstone` L23 (`p1e_tag_and_lock.md` 必备项)
- tasks `p1e-strict-omp-rhs` §5.1 (PR-L 草拟 procedure) + §5.3 (deref commit 确定) + §5.4 (post-merge amend SHA)
- tasks `p1e-strict-omp-rhs` §6.4 (PR-M 前置: P1e-tag push + baseline/P1e lock)
- `docs/p1e/p1e_summary.md` §9 forward handoff (PR-L)
- `docs/p1e/p1e_report.md` §9 D11 7-tag chain final state
- `docs/build_manifest.md` "P1e-tag preparation" + "P1e-tag annotated message" 节
- master plan v1.5 §6 P1e.5
