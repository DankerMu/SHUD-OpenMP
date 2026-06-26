## ADDED Requirements

### Requirement: PR-K2 #223 服务器 8-cell 复跑 success gate

P1c 实施完成后 SHALL 在服务器 cn0X (Slurm) 重新运行 PR-K2 #223 实验配置：heihe + heihe_x4 各 4 N ({1, 2, 4, 8}) = 8 cell，每 cell 跑 90 天 cfg.para 截断，`OMP_PROC_BIND=close OMP_PLACES=cores`。

复跑 SHALL 满足：

- **A3a 全 PASS**：`rivqdown.dat` SHA256 在 N ∈ {1, 2, 4, 8} 下 byte-identical (heihe 4 N 一组 + heihe_x4 4 N 一组)
- **CVODE `nst` 全 N 相等**：heihe nst 跨 4 N 字面相等 (Δ=0 强制)；heihe_x4 nst 跨 4 N 字面相等 (Δ=0 强制；若残留 \|Δ\|≤2 触发 SPGMR-noise ladder, per p1c-deterministic-reduction §"heihe_x4 SPGMR-noise ladder")
- **CVODE 15-key 全 N 相等**：每 case 4 N 之间 15 keys 字面相等
- **若首跑 FAIL** → 触发 §4.7 conditional Kahan injection + PR-K2 二跑；二跑同样需满足上述三条件

#### Scenario: 8-cell A3a bitwise

- **WHEN** P1c 实施完成 + 服务器 cn0X 跑 heihe + heihe_x4 各 4 N
- **AND** `sha256sum .s223-runs/heihe_N{1,2,4,8}/output/heihe.out/heihe.rivqdown.dat` 与同样路径下 heihe_x4
- **THEN** heihe 4 SHA 完全相等；heihe_x4 4 SHA 完全相等

#### Scenario: nst + 15-key 跨 N 相等

- **WHEN** 读两 case 各 4 N 的 `cvode_stats.txt`
- **THEN** 每 case 跨 4 N 的 15 keys 逐项字面相等 (heihe + heihe_x4 共 8 个 stats 文件，分两组各 4 个比对)

---

### Requirement: P1c-tag annotated + immutable + push origin

P1c capstone PR squash-merge 到 `main` 后，SHALL 创建 annotated tag `P1c-tag` aliasing 该 merge commit，tag message 含：

- commit SHA + SHUD pin (openmp-baseline branch HEAD at P1c capstone)
- "P1c — deterministic-reduction tree-shape 固化" 简述
- 与 P1-update-omp-tag 的 forward-compat stacking 关系声明
- 8 站点 (= 10 line anchors) 改造摘要 (fixed-shape pairwise; Kahan if introduced)
- PR-K2 #223 复跑结果摘要 (heihe + heihe_x4 各 4 N canonical SHA 一致 + nst 全等 + Kahan 引入与否)
- master plan §6 P1c.6 移交事项引用

tag 创建后 SHALL 立即 push origin。**P1c-tag 自身亦 D11 immutable**：创建后**禁止** force-update / 重新指向其他 commit；annotated tag object SHA + dereferenced commit SHA 在 capstone 后任何时间均与 capstone-time 记录值字面相等。

#### Scenario: P1c-tag annotated + 含必要字段

- **WHEN** `git show P1c-tag --no-patch --format=fuller`
- **THEN** tag 是 annotated (非 lightweight)，含上述所有字段

#### Scenario: origin 远端 tag 同步

- **WHEN** `git ls-remote --tags origin | grep P1c-tag`
- **THEN** 返回 annotated tag object SHA + dereferenced commit SHA 两行

#### Scenario: P1c-tag 自身 immutable (D11)

- **WHEN** capstone 后任何时间执行 `git show P1c-tag --format=%H` 与 `git rev-list -n1 P1c-tag`
- **THEN** 两者输出 SHALL 字面等于 `docs/p1c/p1c_summary.md` §"验证 P1c-tag" 节记录的 capstone-time tag-object SHA 与 deref commit SHA (D11 immutable enforced; 禁止 force-update / re-target)

#### Scenario: P1-update-omp-tag 不被修改 (D11)

- **WHEN** P1c capstone PR merge 后 `git show P1-update-omp-tag --no-patch --format=fuller`
- **AND** 与本 change 实施前的 `P1-update-omp-tag` 输出做 diff
- **THEN** 两次输出字面相等 (D11 immutable enforced)；annotated tag object SHA = `ff21c75c…`、deref commit SHA = `003f58d…` 未变

---

### Requirement: baseline/P1c 分支创建 + D11 lock

`baseline/P1c` 分支 SHALL 从 `main` 的 P1c capstone merge commit 分出 (与 `P1c-tag` aliasing 同一 commit)，并启用 protection rule `lock_branch=true + enforce_admins=true + allow_force_pushes=false + allow_deletions=false` (与 baseline/B1b / baseline/P1 一致，D11 enforced)。

#### Scenario: baseline/P1c 分支存在 + 锁定

- **WHEN** `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1c/protection --jq '{lock_branch:.lock_branch.enabled, enforce_admins:.enforce_admins.enabled, allow_force_pushes:.allow_force_pushes.enabled, allow_deletions:.allow_deletions.enabled}'`
- **THEN** `{lock_branch:true, enforce_admins:true, allow_force_pushes:false, allow_deletions:false}`

---

### Requirement: docs/p1c/p1c_* 三文档产出 (≥7 topic per p1-capstone F-M-2 PROMOTE upgrade)

P1c capstone SHALL 产 3 个 docs (位置 / 模板 / 必要内容如下表)：

| 文档 | 模板锚 | 必要内容 |
|---|---|---|
| `docs/p1c/p1c_summary.md` | `docs/p1_summary.md` ≥7 topic 结构 (per F-M-2 upgrade) | **必备 ≥7 topic**：(1) 完成定义 / (2) 旧版错误复盘 [P1c 无独立旧版时写一行 "不适用 — P1c 为 P1 后置新阶段"] / (3) P1c-tag 处理 / (4) 时间线 / (5) hand-off → P2a / (6) capstone 验证结果 / (7) 验证 P1c-tag；实际写作可深至 9–12 节 (deeper sub-section inline 在所属 topic 内) |
| `docs/p1c/p1c_perf_baseline.md` | `docs/p1/p1_perf_baseline.md` §2 服务器 scaling | PR-K2 复跑 8-cell A3a + nst 跨 N 表 + wall 数据 + 与 P1 实测的对比 + Kahan 引入与否说明 + Mac 辅助预筛节 (若执行) |
| `docs/p1c/p1c_a3a_root_cause.md` | 新文档 (吸收 F-K2-2 reviewer finding) | 4 项必备字段 (per "p1c_a3a_root_cause 吸收 F-K2-2 + 量化数据" Scenario)：(a) pre-fix DY divergence first-occurrence step + element ID + bit-level diff；(b) post-fix same probe showing zero divergence；(c) per-site ULP delta table (8 站点 × N {2,4,8})；(d) 显式 confirm/refute tree-reduction-depth N>2 hypothesis with data |

#### Scenario: 三 docs 存在且填好

- **WHEN** `ls docs/p1c/p1c_summary.md docs/p1c/p1c_perf_baseline.md docs/p1c/p1c_a3a_root_cause.md`
- **THEN** 三文件均存在；逐文件检查含上述"必要内容"列字段

#### Scenario: p1c_summary.md 7 topic 必备 + 顺序对齐

- **WHEN** 检查 `docs/p1c/p1c_summary.md` topic 标题
- **THEN** 上述 7 必备 topic 全部出现；topic 顺序与 p1_summary.md / b1b_summary.md 一致；deeper sub-section 数量不计 (允许 9–12 节)

#### Scenario: p1c_a3a_root_cause 吸收 F-K2-2 + 量化数据

- **WHEN** 读 `docs/p1c/p1c_a3a_root_cause.md`
- **THEN** SHALL 显式引用 PR-K2 #223 reviewer F-K2-2 finding (P1 epic F-K2-2 残留 debt)；SHALL 含 4 项必备字段：(a) pre-fix DY divergence first-occurrence (step / element ID / bit-level diff)；(b) post-fix same probe showing zero divergence；(c) per-site ULP delta table (8 站点 × N {2,4,8})；(d) 显式 confirm/refute tree-reduction-depth N>2 hypothesis with data backing

---

### Requirement: status_matrix + build_manifest + glossary 同步

P1c capstone SHALL 同步更新：

1. `docs/status_matrix.md`：加 P1c 行 PASS verdict (引 P1c-tag + p1c_summary.md cross-link)
2. `docs/build_manifest.md`：append "P1c-tag 应用状态" 节 (与 P1-update-omp-tag 节同 pattern；含 SHUD pin + binary sha256 from `make_shud_mac.log` + `make_shud_server.log` + GCC version + strict FP flags 3-grep + Slurm jobids)
3. `openspec/glossary.md`：加 P1c 集合术语条目 (P1c-tag, baseline/P1c, fixed-shape pairwise canonical reduction, Kahan compensated summation, deterministic-reduction tree-shape, ...)

#### Scenario: status_matrix P1c 行

- **WHEN** `grep -n "P1c" docs/status_matrix.md`
- **THEN** P1c 行存在且 verdict = PASS；schema 与 B0/B1a/B1b/B1/P1 行一致

#### Scenario: build_manifest P1c 节

- **WHEN** `grep -nE "P1c-tag|P1c-tag 应用状态" docs/build_manifest.md`
- **THEN** 存在 "P1c-tag 应用状态" 节；含 SHUD pin + binary sha256 + GCC + strict FP flags + Slurm jobids

#### Scenario: glossary P1c 集合

- **WHEN** `grep -nE "P1c-tag|baseline/P1c|fixed-shape pairwise|Kahan" openspec/glossary.md`
- **THEN** 至少 4 个新术语条目 (P1c-tag / baseline/P1c / fixed-shape pairwise canonical reduction / Kahan compensated summation)；命令在 Mac BSD grep + Linux GNU grep 均返回一致结果（`-E` 强制 ERE 不依赖 implementation-defined `\|` 语义）

---

### Requirement: PROMOTE + archive + jsonl 双追加 + Epic 关闭

P1c capstone PR SHALL：

1. PROMOTE `openspec/changes/p1c-deterministic-reduction/specs/p1c-deterministic-reduction/spec.md` → `openspec/specs/p1c-deterministic-reduction/spec.md`
2. PROMOTE `openspec/changes/p1c-deterministic-reduction/specs/p1c-capstone/spec.md` → `openspec/specs/p1c-capstone/spec.md`
3. archive `openspec/changes/p1c-deterministic-reduction/` → `openspec/changes/archive/<YYYY-MM-DD>-p1c-deterministic-reduction/` (gitignored archive 模式，同 P1 epic)
4. `docs/review-loop-log.jsonl` + `docs/stage-pipeline-log.jsonl` 各追加 P1c epic close 行
5. Epic 关闭 (`Closes #<epic-num>`)；所有 sub-issue 关闭

#### Scenario: PROMOTE 完整

- **WHEN** `ls openspec/specs/p1c-deterministic-reduction/spec.md openspec/specs/p1c-capstone/spec.md`
- **THEN** 两文件均存在；diff vs archive 仅头部转换 (`## ADDED Requirements` → `## Purpose / Conventions / Requirements`) + 受控 wording fix

#### Scenario: jsonl 双追加

- **WHEN** `tail -1 docs/review-loop-log.jsonl docs/stage-pipeline-log.jsonl`
- **THEN** 两文件末行均为 P1c epic close 行 (含 epic / tag / baseline_branch / verdict 等字段)

#### Scenario: Epic 关闭

- **WHEN** P1c capstone PR merge
- **THEN** Epic issue state = CLOSED；所有 sub-issue state = CLOSED

---

### Requirement: master plan §6 P1c.2 文件名修正移交 (M10) + 文档全量无 stale MD_f.cpp 引用

P1c capstone SHALL 在 `docs/p1c/p1c_summary.md` §"Hand-off" 节或独立 sub-task 中显式记录：master plan §6 P1c.2 表中文件名 `SHUD/src/Model/MD_f.cpp` 与实际仓库 `MD_rhs_core.cpp` 不一致，**延后**至 M10 修订段同步更新；本 change 不动 master plan (per design D1 / Q3 决策)。

同时，除 M10 hand-off 段与 "dead-code mirror" 显式声明外，docs/p1c/p1c_* 与 openspec/specs/p1c-* 范围内**禁止**残留任何 stale MD_f.cpp 引用（防止 P1c 文档误导读者认为实际改的是 MD_f.cpp）。

#### Scenario: M10 移交事项显式记录

- **WHEN** 读 `docs/p1c/p1c_summary.md` 或 `docs/p1c/p1c_a3a_root_cause.md`
- **THEN** SHALL 含一段说明 master plan §6 P1c.2 typo + M10 修订段同步范围 + 本 change 不动 master plan 的理由 (design D1)

#### Scenario: 文档全量无 stale MD_f.cpp 引用

- **WHEN** `grep -rn 'MD_f.cpp' docs/p1c/p1c_* openspec/specs/p1c-* 2>/dev/null | grep -v 'Hand-off\|M10\|legacy dead-code mirror\|dead-code mirror'`
- **THEN** 返回 0 hits (所有 MD_f.cpp 引用仅出现在 M10 hand-off 段或 dead-code mirror 显式声明段)
