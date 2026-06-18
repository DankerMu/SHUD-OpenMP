# S1-ci-A: build matrix + grep gates + nightly cron (issue #50)

把 `serial-baseline.yml` 从单轴 keliya-only 扩成 **LEGACY_RHS × case** 双维 matrix，加上 4 个 S1d.1/S1d.2 宏移除 grep gates、case-参数化、nightly cron。只动 `.github/workflows/serial-baseline.yml`，SHUD submodule 不动。

## Spec 映射

| Spec 条目（`b0-tag-ci-integration/spec.md`） | 落地位置 | 备注 |
|----|----|----|
| L276-301 Requirement "CI build matrix 跑 LEGACY_RHS=0 与 =1 两轴" | `setup` job + `build-and-compare.strategy.matrix` | 2 轴：`LEGACY_RHS=0`（默认 = B1a）+ `LEGACY_RHS=1`（legacy 回退）|
| L280-286 Scenario "PR fast-feedback 两轴跑 keliya bitwise" | setup job 的 fast-feedback 分支 | 2 axes × 1 case = 2 jobs，wall-clock 目标 ≤4 min |
| L288-293 Scenario "nightly cron 两轴 × 4 case bitwise" | setup job 的 full-bitwise 分支 + `on.schedule.cron: '0 3 * * *'` | 2 axes × 4 cases = 8 jobs，timeout 25 min |
| L295-300 Scenario "任一轴 fail 则 PR / nightly 整体 fail" | `strategy.fail-fast: false` + 各 matrix job 独立 exit code | branch protection 在外侧把两轴都设 required check |
| L302-340 Requirement "宏移除 grep gate 作 CI 强检查" | 新 step "Macro removal grep gates (S1d.1 + S1d.2)" | 4 个 grep，全部 0 命中才 PASS |
| L314-319 Scenario "Post-S1d.1 USE_RHS_CORE grep gate" | grep #1 | 现 SHUD pin ce6ee3a post-S1d.2，0 命中 |
| L321-326 Scenario "Post-S1d.2 _OPENMP_ON grep gate" | grep #2 | 0 命中 |
| L328-333 Scenario "Post-S1d.2 NV_DATA_OMP / NV_DATA_S grep gate" | grep #3（仅 `f.cpp`）| 0 命中 |
| L335-340 Scenario "Post-S1d.2 N_VDestroy_Serial grep gate" | grep #4 | 0 命中 |
| L85-100 Requirement "90-day truncation 在 CI 自动应用" | "Fix case paths" + 新 step "Enforce 90-day truncation" | `fix_case_paths.sh` 本身不动 START/END（spec 描述与现实有 gap），inline awk 显式重写 END=START+90 |
| L110-128 Requirement "SHA256 mismatch 的错误处理" | "Compare ... output vs B0 golden (SHA256, axis ...)" | `::error` 含 case 名、文件路径、expected/got SHA256、复现命令 |
| L20-23（spec scope note）+ #49 precedent "LEGACY_RHS=1 必须保留 `f_*_omp` 符号" | "Verify legacy _omp symbols present (LEGACY_RHS=1 only)" | `nm` 数 3 个符号，不等 3 即 FAIL |

## Validation evidence

### 1. YAML 解析 + 结构（pyyaml via uv）

```
$ uv run --with pyyaml python -c "import yaml; data=yaml.safe_load(open('.github/workflows/serial-baseline.yml')); ..."
YAML parse OK
jobs: ['setup', 'build-and-compare']
triggers: ['push', 'pull_request', 'schedule']
push branches: ['**']
pull_request branches: ['main', 'baseline/current']
schedule: [{'cron': '0 3 * * *'}]
build-and-compare matrix strategy: {'fail-fast': False, 'matrix': '${{ fromJson(needs.setup.outputs.matrix) }}'}
build-and-compare needs: setup
setup outputs: {'matrix': '${{ steps.matrix.outputs.matrix }}', 'mode': '${{ steps.matrix.outputs.mode }}'}
build-and-compare timeout-minutes: 25
build-and-compare steps: 28
```

注：本地无 yamllint；用 pyyaml safe_load 等价 schema 检查。`on:` key 在 YAML 1.1 是 boolean true 别名（PyYAML 行为）；GitHub Actions 解析器单独识别 `on:`，CI 端无问题。

### 2. 本地 grep gate dry-run（4 条命令 vs 当前 SHUD/src/）

| Grep | 命令 | exit | matches |
|----|----|----|----|
| 1 | `grep -rn 'USE_RHS_CORE' SHUD/src/` | 1 | 0 |
| 2 | `grep -rn '_OPENMP_ON' SHUD/src/` | 1 | 0 |
| 3 | `grep -nE 'NV_DATA_OMP\|NV_DATA_S' SHUD/src/Model/f.cpp` | 1 | 0 |
| 4 | `grep -rn 'N_VDestroy_Serial' SHUD/src/` | 1 | 0 |

全部 0 命中（grep exit 1 = no match，符合预期），证明 S1d.2 退役已就位，CI 上线后 grep gate step 会通过。

### 3. openspec validate

```
$ openspec validate s1-rhs-core-extraction --strict --no-interactive
Change 's1-rhs-core-extraction' is valid
```

### 4. matrix JSON 实际展开

**PR-default（fast-feedback）**：

```json
{"rhs_path":["LEGACY_RHS=0","LEGACY_RHS=1"],"case":["keliya"]}
```

→ 2 个 matrix entry：`(LEGACY_RHS=0, keliya)`、`(LEGACY_RHS=1, keliya)`

**nightly cron + PR `full-bitwise` label**：

```json
{"rhs_path":["LEGACY_RHS=0","LEGACY_RHS=1"],"case":["keliya","xinanjiang_upstream","qinyijiang","qhh"]}
```

→ 8 个 matrix entry：2 axes × 4 cases。

### 5. 关键步骤变化 (diff 概览)

- `+ setup` job（emit matrix JSON）
- `+ on.schedule.cron: '0 3 * * *'`
- `+ Macro removal grep gates (S1d.1 + S1d.2)`（数据无关，4 个 grep）
- `+ Verify legacy _omp symbols present (LEGACY_RHS=1 only)`（`nm` 数 3 符号）
- `+ Enforce 90-day truncation (END = START + 90)`（inline awk）
- `+ Compute case-relative snapshot abs-min triple`（按当前 case START 算 abs-min；不再硬编 keliya 数值）
- 全部 keliya 硬编路径 → `${{ matrix.case }}`
- 全部 build step → `${{ matrix.rhs_path }}` 附加到 `make` 参数
- `timeout-minutes: 10 → 25`
- 失败 artifact 名 → `baseline-failure-diff-${{ matrix.case }}-${{ matrix.rhs_path }}`（避免 matrix 冲突）
- status-matrix proposer 加 `matrix.rhs_path == 'LEGACY_RHS=0'` gate（legacy axis 不写 B1a 行）+ HTML marker 加 case scope（`<!-- s0-12-matrix-proposer:<case> -->`）

行数：`+450 / -120` 左右（含注释）。

## 已知限制

- **forcing 数据仍未部署到 GitHub runner**：4 个 case 的 `data_probe` 全 `data_available=false`，bitwise compare / snapshot / status-matrix proposer 全部进入 skip 分支。这是 spec L277 "fast-feedback wall-clock ≤4 min" 的有效路径——build + grep gates 仍跑，gate 部分（4 个 grep + golden manifest + tool selftests）真实生效；run+compare 部分是 scaffold，等数据部署 PR 落地后自动激活。
- **out-of-scope（已注释延后到 #51）**：`fetch-tags: true` / `fetch-depth: 0`；CVODE 15-key stats diff in matrix；snapshot t_values 90-day HARD-fail step；`skip-baseline-ci` label 删除 + PR label types 扩展（`labeled/unlabeled`）。spec L189-211 / L251-274 / L102-108 / L160-188 都已在 spec 落，由 #51 实施。
- **`fix_case_paths.sh` 与 spec L92-93 存在 gap**：spec 写 "该脚本 MUST 把 cfg.para 的 END 行改为 START + 90"，但脚本现状只重写 `tsd.forc` line 2 + 设 `NUM_OPENMP=1`，不动 START/END。本 PR 用 inline awk 兜底（"Enforce 90-day truncation" step），符合 CLAUDE.md §90-day-truncation 的 deployment-layer 定位；脚本本身的契约修订留给后续单独 PR（修改 tool 是 spec change，超出 #50 范围）。

## Test plan

- [ ] PR 触发 → `setup` job emit fast-feedback matrix (2 entries) → 2 个 matrix job 全部跑 build + grep gate + tool selftest；data_probe = false 时 run+compare 跳过 with notice
- [ ] PR 加 `full-bitwise` label → matrix 升到 8 entries（2 × 4 cases）
- [ ] nightly cron → schedule 触发 → 8-job 全跑
- [ ] forcing 数据未部署：build / grep / golden manifest / nm symbol assert 全 PASS；run+compare 进 skip 分支输出 `::notice`，job 整体 PASS
- [ ] forcing 数据部署后（#51 或单独 data PR 落地）：run+compare 全跑、SHA256 / snapshot / profile neutrality 全 gate 生效；status-matrix proposer 在 LEGACY_RHS=0 axis PASS 时留 PR comment
