# B0 构建清单

> 复现 B0 baseline 二进制的权威来源。
> 任何对 flag、SUNDIALS 版本或编译器的改动，都必须配一个 OpenSpec change，落在
> `openspec/changes/<change>/specs/build-environment-lockdown/spec.md`。

## 1. Linux 配置（目标部署平台 — go/no-go 权威端）
- 编译器：GCC 12，通过 `g++-12` 或 `CXX=g++-12 make shud` 调用
- 基础 flag：`-O2 -g -ffp-contract=off -fno-fast-math -std=c++14`（取自 `CXX_BASE_FLAGS`）
- OpenMP 编译 flag：`-fopenmp`（Makefile 里 Linux UNAME_S 分支）
- OpenMP 链接 flag：`-lgomp`
- SUNDIALS：6.0.0，通过 `./configure` 装到 `SHUD/InstallSundials/`
- SUNDIALS 链接：`-lsundials_cvode -lsundials_nvecserial`（serial）+ `-lsundials_nvecopenmp`（omp）
- 验证命令：`./configure` 之后 `make shud && make shud_omp`

## 2. macOS 配置（开发用，Apple Silicon）
- 编译器：Apple Clang（通过 PATH 中的 `g++` 包装解析；本机：Apple clang 17.0.0）
- libomp：`brew install libomp`；前缀通过 `$(brew --prefix libomp)` 自动探测（Apple Silicon 上为 `/opt/homebrew/opt/libomp`）
- 基础 flag：同 Linux
- OpenMP 编译 flag：`-Xpreprocessor -fopenmp`
- OpenMP 链接 flag：`-L$(brew --prefix libomp)/lib -lomp`
- SUNDIALS：同上安装路径
- 跨平台说明：macOS 数字仅供开发期参考；§1.1.1 的量化 go/no-go 只在 Linux 决（见 master plan §1.1.1）。

## 3. OpenMP 运行时 env（两个平台都用）
- `OMP_PROC_BIND=close`
- `OMP_PLACES=cores`
- `OMP_NUM_THREADS` 按各 benchmark 的 `manifest.yaml` 设置
- `NumEle < OMP_CUTOFF` 触发 serial fallback（master plan §C8）

## 4. SHUD submodule pin
- Upstream：`https://github.com/SHUD-System/SHUD.git`
- 上游工作分支：`openmp-baseline`（长寿命，从 `3aec657` 派生；不是 master）
- 初始 B0 commit：`3aec657`（master plan §S0.10）
- 当前 submodule HEAD：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`（S0-10 / #14 timer 仪器化，PROFILE=0 / DUMP=0；每个改 SHUD 的 PR merge 时以及打 B0-tag 时更新）
- 本地验证：`git -C SHUD rev-parse HEAD`

## B0-tag（S0-13 / #17）

`B0-tag` 轻量 git tag pin 住了 A0 验收门认证的那一对 `(outer, SHUD submodule)` commit。B1a 回归比对就是逐字节 diff `git show B0-tag:benchmarks/<case>/B0_output/`。

> **节有效性说明**：下面的命令（`git rev-parse B0-tag`、`git show B0-tag --stat -- SHUD`）只有在项目所有者 `git tag -a B0-tag <merge-commit-sha>` + `git push origin B0-tag` 之后才能成功；这一动作发生在 S0-13 PR merge 之后。在那之前 `B0-tag` 不存在。
> 实时状态见下面的 `## B0-tag 应用状态`。

- **外层 repo tag**：`B0-tag` 在 `baseline/current` 分支上、S0-13 PR（#35）的 squash-merge commit 上打。打完验证：`git rev-parse B0-tag`。
- **B0-tag 时刻的 SHUD submodule pin**：`78c37a1061de4112bc7c297bb7bd1f107432e6f2`（外层 tag commit 抓住的 submodule pointer）。打完验证：`git show B0-tag --stat -- SHUD`。
- **日期**：2026-06-17
- **Tagger**：DankerMu（项目所有者；GitHub `@DankerMu`）；tag push 本身由 claude-code 代 DankerMu 执行，授权基于 `docs/profile_decision.md` 签字同一份 2026-06-17 的 delegated grant。该授权允许编排式代理打 tag；tag message 记录 `Tagger: DankerMu (delegated via claude-code)`。
- **"B0 里有什么" 的权威**：`docs/status_matrix.md` 的 B0 行（本地 4 个 PASS + heihe PASS @ 服务器 + heihe_x4 PASS @ 服务器 + kashigeer N/A deferred-upstream = 6 PASS + 1 N/A）；A0 验收 checklist 9/9 PASS，按 S0-13 spec 修订。

- **Manifest 摘要**（B0-tag merge commit 时每份 `benchmarks/<case>/manifest.yaml` 的 SHA256）：

  | Case                | `manifest.yaml` 的 SHA256                                            |
  |---------------------|----------------------------------------------------------------------|
  | `keliya`             | `db9e19eceb2a99027cf06be37b72b61d2f049b282650aa514cd8a1008678e8aa` |
  | `xinanjiang_upstream`| `e1ff2c61e112aa32bb816bacada8f70561cc62283ae58530ad302680b6e75aef` |
  | `qinyijiang`         | `75679b91f9bfbb53741b90f1a748ccee1b95a1342c1a4d7042533d1626a4b496` |
  | `kashigeer`          | `5a856f370eb3c3aa302a94b44bbc297c512ab2b6e94e0ed12e2c22cd5fe9e942` |
  | `qhh`                | `65dc067a6930d4485f7f88111fecee5e9a7d050239c95f7b1deb6caad14e5033` |
  | `heihe`              | `c98569188a60ed74b134910e478eadca21711d32aaed7c3e710426604b9b386b` |
  | `heihe_x4`           | `18f71e3dbf2355a121140119cb2649824eed785c97ccbd1cfbc19e9fa4afafb7` |

  算法：`shasum -a 256 benchmarks/<case>/manifest.yaml`（Linux 上用 `sha256sum`）。B1a 回归检查时不匹配是硬 FAIL：B1a 时刻的 registry 状态必须等于 B0-tag pin 住的状态。

- **分支保护**：B0-tag 打完时，`baseline/current` 启用了 `.github/workflows/serial-baseline.yml` 的 `build-and-compare` required check。`skip-baseline-ci` label-bypass 权限在 B0-tag merge 时移除，避免 S1+ 的 PR 静默绕过 baseline 验证。

## B0-tag 应用状态

| 字段 | 值 |
|---|---|
| `B0-tag-applied` | `true` |
| `B0-tag-date` | `2026-06-17` |
| `B0-tag-object-sha` | `95ddc375ffa58115fd5c0a808dde80e9713b4c93`（annotated） |
| `B0-tag-commit-sha` | `884cfb13ba08ebae02dd64e371c4a19a536b4e26`（PR #35 squash-merge 到 `baseline/current`） |
| `SHUD-submodule-pin` | `78c37a1061de4112bc7c297bb7bd1f107432e6f2` |

上一节的验证命令（`git rev-parse B0-tag`、`git show B0-tag --stat -- SHUD`、`git ls-remote --tags origin | grep B0-tag`）在 tag push（2026-06-17）后都成功。上一节的 "Section validity" 提示已是历史信息——保留以备审计。

## B1a-tag（S1 / #52）

`B1a-tag` annotated git tag pin 住了 S1 全套 substage（S1a-S1d.2 + S1-ci-A/B）完成后认证的那一对 `(outer, SHUD submodule)` commit。S2 起回归比对 baseline 切到 `B1a-tag`（B0-tag 仍可比对，但 B1a-tag 是 strict 阶段更精确的零参考点）。

> **节有效性说明**：下面的命令（`git rev-parse B1a-tag`、`git show B1a-tag --stat -- SHUD`）只有在项目所有者 `git tag -a B1a-tag <merge-commit-sha>` + `git push origin B1a-tag` 之后才能成功；这一动作发生在 S1-#52 PR merge 之后。在那之前 `B1a-tag` 不存在。
> 实时状态见下面的 `## B1a-tag 应用状态`。

- **外层 repo tag**：`B1a-tag` 在 `baseline/current` 分支上、#52 PR 的 squash-merge commit 上打。打完验证：`git rev-parse B1a-tag`。
- **B1a-tag 时刻的 SHUD submodule pin**：`58327c5`（外层 tag commit 抓住的 submodule pointer；post-#50 functions.cpp `_OPENMP_ON` fix）。打完验证：`git show B1a-tag --stat -- SHUD`。
- **D12 收尾约束**：
  - 4-case + 2-case bitwise PASS vs B0-tag（kashigeer N/A deferred-upstream）
  - CVODE 15-key byte-equal（F19 修订：归档 15 key 不含 nFCall）
  - SHUD `openmp-baseline` 对应 commit `58327c5`
  - `LEGACY_RHS=0` + `LEGACY_RHS=1` 双路径都 bitwise PASS
  - `grep -r 'USE_RHS_CORE' SHUD/src/` / `grep -r '_OPENMP_ON' SHUD/src/` / `grep -r 'N_VDestroy_Serial' SHUD/src/` 三 grep 全 0 hits

## B1a-tag 应用状态

| 字段 | 值 |
|---|---|
| `B1a-tag-applied` | `<filled-after-tag-push>` |
| `B1a-tag-date` | `<filled-after-tag-push>` |
| `B1a-tag-object-sha` | `<filled-after-tag-push>`（annotated） |
| `B1a-tag-commit-sha` | `<filled-after-tag-push>`（PR #52 squash-merge 到 `baseline/current`） |
| `B1a-tag-SHUD-pin` | `58327c5` |

`B1a-tag` push 后，下列命令应全部成功：
- `git rev-parse B1a-tag` → tag object SHA
- `git show B1a-tag --stat -- SHUD` → SHUD pin 显示 `58327c5`
- `git ls-remote --tags origin | grep B1a-tag` → 远端 tag 存在

## B1b-tag（S5 + S6b + S6c-12 / Epic #172）

`B1b-tag` annotated git tag pin 住了 S5a/S5b/S5c-A/S5c-B/S5c-C/S5d.1/S5d.2-5a/S5d.2-5b/S5d.3/S5d.4/S5d-summary + S6b.1/S6b.2/S6b.3 + S2.17 audit + S6c-12a B1b 3-run capstone 全部完成的 `(outer, SHUD submodule)` commit。B1c+ 起回归比对 baseline 切到 `B1b-tag`（B0-tag / B1a-tag 仍可比对，但 B1b-tag 是 P-strict 阶段更精确的零参考点）。

> **节有效性说明**：下面的命令（`git rev-parse B1b-tag`、`git show B1b-tag --stat -- SHUD`）只有在项目所有者 `git tag -a B1b-tag <merge-commit-sha>` + `git push origin B1b-tag` 之后才能成功；这一动作发生在 #189 中（2026-06-22）。在那之前 `B1b-tag` 不存在。
> 实时状态见下面的 `## B1b-tag 应用状态`。

> **D11 强制**：B1b-tag 一次锁死，**禁止 force-update**（与 B1a-tag force-update 历史不同）。任何后续 retroactive 更新（如 #185 PI sign-off）走 forward-compat **B1c-tag stacking** 路径（master plan C8）。

- **外层 repo tag**：`B1b-tag` 在 `baseline/B1b` 分支上、PR-16 #207 squash-merge + #188 post-merge log append (commit `18a0c908`) 上打。
- **B1b-tag 时刻的 SHUD submodule pin**：`71b3a1ae4ef82e165134a18469c7d0a79284b67f`（外层 tag commit 抓住的 submodule pointer，`SHUD/B1b_CHANGELOG.md` 12 sections evidence trail 完整）。
- **D11 / D12 收尾约束**：
  - 4-case Mac canonical summary SHA ≡ `benchmarks/<case>/B0_output/repeatability.txt sha256_run1`（keliya / xinanjiang_upstream / qinyijiang / qhh）
  - 2-case Server (cn03) Slurm 8662-8667 rivqdown ≡ B0/B1a-tag golden（heihe `55abad28…` / heihe_x4 `f90601ef…`）
  - `B1b_CHANGELOG.md` 12 sections（S5a / S5b / S5c-B / S5c-C / S5d.1 / S5d.2-5a / S5d.2-5b / S5d.3 / S5d.4 / S6b.1 / S6b.2 / S6b.3）+ openmp-baseline push 5-step workflow 严格遵守
  - `baseline/B1b` 分支 protection `lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false` enforced
  - ship 状态 = **UNCONDITIONAL** (PR-19 #210 PI E2 sign-off + #205 cleared)，原 CONDITIONAL caveat 5 项处置（详 `docs/b1b_summary.md` §"B1b ship status" + `docs/b1b/s217_lake_formula_audit.md` §E）：
    - `#185` (S2.17 PI 审查) — **RESOLVED (E2 signed)** via PR-19 #210（DankerMu 作为 `SHUD-System/SHUD` upstream-org owner 签 E2 "formula correct, no change"；design.md Open Q1 同时关闭：PI delegate = upstream-org owner / three-surface sign-off pattern = issue comment + audit doc §E + SHUD CHANGELOG）
    - `#186` (S6b.2) — **CLOSED-via-PI-E2**：原 SKIP path 在 #185 E2 sign-off 后从 "FORECAST per C8" 升级为 "consistent with signed PI E2 directive"
    - `#205` (rhs_flux SoA/AoS sync drift) — **RESOLVED (post-B1b cleanup before P1)** via SHUD `de75743` (fix) + `9a376f7` (CHANGELOG) on `openmp-baseline` + 外层 PR-18 #209 (pointer bump `71b3a1a` → `9a376f7` + docs sync)；4-case Mac 2-run canonical SHA bitwise vs B1b-tag baseline；NOT retroactively part of B1b per D11；P-strict pre-req gap closed；同时 strengthens #185 E2 verdict（audit §A.4/§B.4 strict-reading concern 消解）
    - D9 fast-path trigger #2 — **TRIGGERED in PR-19 #210**：`B1-tag` annotated tag 创建 aliasing main HEAD（含 #205 cleanup + PI E2 sign-off）；`B1a-tag` (`f7f992c…`) + `B1b-tag` (`18a0c908…`) 保留 immutable per D11 history 不 force-update；下游 P1+ SHOULD use `B1-tag`
    - C8 forward-compat — **UNUSED for this ship**（PI 签 E2 不签 E1，B1c-tag stacking 不触发）；仍是 codebase convention 留给未来可能的 P-strict overrule

## B1b-tag 应用状态

| 字段 | 值 |
|---|---|
| `B1b-tag-applied` | `true` |
| `B1b-tag-date` | `2026-06-22` |
| `B1b-tag-object-sha` | `96e224daad8cb9c93f855851724f8d45468391c2`（annotated） |
| `B1b-tag-commit-sha` | `18a0c9085f494d1cf228c7be4adf27d9132d05dd`（PR-16 #207 squash-merge + #188 post-merge log append on `baseline/B1b`） |
| `B1b-tag-SHUD-pin` | `71b3a1ae4ef82e165134a18469c7d0a79284b67f` |
| `B1b-conditional-ship` | `yes`（详 `docs/b1b_summary.md` §"B1b CONDITIONAL ship status"） |

`B1b-tag` push 后命令全部成功（已 2026-06-22 验证）：
- `git rev-parse B1b-tag` → `96e224da…`
- `git rev-parse B1b-tag^{}` → `18a0c908…`
- `git show B1b-tag --stat -- SHUD` → SHUD pin 显示 `71b3a1ae`
- `git ls-remote --tags origin | grep B1b-tag` → 远端 tag 存在
- `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/B1b/protection` → `lock_branch=true`

## S1-pre snapshot golden re-archival

为支持 S1（B1a refactor-equivalent serial reference），在 pre-S1a 重新归档 12 张 after-PassValue snapshot golden（4 case × 3 t_values）。

**理由**：原 S0-7 12 张 golden 由 4-year full run 产生，timestamps（绝对分钟，如 keliya `t=17,357,760` ≈ START+1d、qhh `t=12,241,440` ≈ START+100d、qinyijiang `t=671,040` ≈ START+100d、xinanjiang_upstream `t=144,000` = START+100d）覆盖到 case-relative 100 day mark；S1 CI 与本地 bitwise gate 全部在 ≤90 天截断窗口运行（CLAUDE.md "所有 case 一律 ≤90 天截断"），第三张 100d snapshot 不可达，必须按新 t_values 重新归档。

**新统一 t_values（case-relative seconds）**：`[86400, 2592000, 7776000]` = 1 d / 30 d / 90 d，全部 ≤ 90 天窗口。

**Hook 单位换算契约**：`MD_rhs_dump.cpp` 的 `SHUD_DUMP_T_VALUES` 接收**绝对分钟数**（SHUD t-unit = minutes，见 `MD_rhs_dump.h` line 59 + format header）。Runner 必须把 manifest 的 case-relative seconds 转成绝对分钟：`abs_min = START_day × 1440 + (case_rel_sec / 60)`。Archive 文件命名采用 case-relative seconds（`snapshot_t<sec>.bin`）以与 manifest `snapshot_probe.t_values` 对齐；hook 默认输出文件名是绝对分钟（`snapshot_t<abs_min>.bin`），归档前需 rename。

**重新归档命令**（B0-tag binary，90-day cfg.para 截断）：

```bash
git checkout B0-tag
cd SHUD && git checkout 78c37a1 && cd ..
git submodule update --recursive
cd SHUD && make clean && make shud SHUD_DUMP_RHS=1 && cd ..
# 部署层 cfg.para 截断（gitignored via SHUD/.git/info/exclude → /Basins/）：
# 每个 case 设 END = START + 90 day
# 然后对每个 case 跑 SHUD 同时把 t_values 转绝对分钟传 hook，再 rename 归档：
for case_proj in "keliya keliya" "xinanjiang_upstream xinanjiang" "qinyijiang nanlin" "qhh qhh"; do
  set -- $case_proj
  case=$1; proj=$2
  start_day=$(awk '$1=="START" {print $2; exit}' SHUD/Basins/$case/input/$proj/$proj.cfg.para)
  start_min=$((start_day * 1440))
  # offsets: 1d=1440 min, 30d=43200 min, 90d=129600 min
  abs_t="$((start_min + 1440)),$((start_min + 43200)),$((start_min + 129600))"
  cd SHUD/Basins/$case
  rm -rf output/$proj.out
  SHUD_DUMP_T_VALUES="$abs_t" \
  SHUD_DUMP_T_TOL=60 \
  SHUD_DUMP_OUTPUT_DIR="$(pwd)/output/$proj.out" \
  SHUD_DUMP_CASE_ID="$proj" \
  SHUD_DUMP_SITE="f_update" \
  ../../shud $proj
  cd ../../..
  # rename to case-relative seconds + cp to archive (per-case mapping in tools/rhs_snapshot/README)
done
```

**12 张新 golden SHA256**（B0-tag = `884cfb13` outer + SHUD pin `78c37a1`，2026-06-18 archived；本地 macOS Apple Silicon、Apple Clang 17.0.0；server re-archival 在 S1a #44 验证一致）：

| Case                  | t_value (s) | File                                                       | SHA256                                                             |
|-----------------------|-------------|------------------------------------------------------------|--------------------------------------------------------------------|
| `keliya`              | `86400`     | `benchmarks/keliya/B0_output/snapshot_t86400.bin`          | `53b078c56edd9f907e791cb9850e75f939d193a31025ecc9d0e1f080e9340c06` |
| `keliya`              | `2592000`   | `benchmarks/keliya/B0_output/snapshot_t2592000.bin`        | `e8361a54416135a2840b037de06c85fb3ec02617625876fcca5639d1ea5be9d1` |
| `keliya`              | `7776000`   | `benchmarks/keliya/B0_output/snapshot_t7776000.bin`        | `7e55dd5727602b234880650b3f8933902e601d319414be3e2d49770947666f94` |
| `xinanjiang_upstream` | `86400`     | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t86400.bin`   | `68b37777f57d48dd86ffa6d3d2fa0262ad8f5bc87d5d1bb0009c1d42af93cee4` |
| `xinanjiang_upstream` | `2592000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t2592000.bin` | `9a008094e8929b9e71a06f83cf43528a887a6628059f0e3b13d4c9008298627d` |
| `xinanjiang_upstream` | `7776000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t7776000.bin` | `5557a4c3b77eb1ec090054ff31b4a3d536536709a1446e66bd6f96a69ae8c977` |
| `qinyijiang`          | `86400`     | `benchmarks/qinyijiang/B0_output/snapshot_t86400.bin`      | `b252aaf6e1a05e9ada4898ec41b7836dffa6b4ce8613c566664b12fd4de20b56` |
| `qinyijiang`          | `2592000`   | `benchmarks/qinyijiang/B0_output/snapshot_t2592000.bin`    | `ef4ac2c84163c5f63215367b3d97f981092ccaf7783b13924d027ebc00af61a4` |
| `qinyijiang`          | `7776000`   | `benchmarks/qinyijiang/B0_output/snapshot_t7776000.bin`    | `3348dc00ccea1f83be28c5afcb0c8c9ebdd335e14bd29affe3237987e93ca671` |
| `qhh`                 | `86400`     | `benchmarks/qhh/B0_output/snapshot_t86400.bin`             | `588307a13ad8c8711bfa4db08c896b1e679cb4848777fcc0f42b79e2e2ce1e9e` |
| `qhh`                 | `2592000`   | `benchmarks/qhh/B0_output/snapshot_t2592000.bin`           | `0aefb90c8586948b6eae285a3bdf13313fbef6269ce1bf61f1dbcf9fb3d58ca0` |
| `qhh`                 | `7776000`   | `benchmarks/qhh/B0_output/snapshot_t7776000.bin`           | `218dbc817992798f466eab3e7102c7a493826a4cab9ced36961aa04e2243aa6a` |

算法：`shasum -a 256 benchmarks/<case>/B0_output/snapshot_t<v>.bin`（Linux 上用 `sha256sum`）。binary 内 `RecordHeader.t_value` 字段保留**绝对分钟**（不可改、跟 hook spec 一致；见 `format.h`）；只文件名是 case-relative seconds（对齐 manifest）。

**旧 S0-7 golden 处理**：保留作 historical reference（不删除）。命名是绝对分钟（如 `snapshot_t17357760.bin`、`snapshot_t17370720.bin`、`snapshot_t17500320.bin`），与新归档（case-relative seconds 命名）**文件名不冲突**，共存于同一 archive 目录。manifest.yaml 的 `snapshot_probe.t_values` 已 point 到新 t_values；下游 S1 bitwise gate 一律使用**新 golden**。

> Sanity check note：全 4 case 的新 1d golden（`snapshot_t86400.bin`）与旧 S0-7 4-year-run goldens 在**同绝对仿真时间**（abs_min = START_day × 1440 + 1440）byte-equal（verified with `tools/compare_snapshot`）。这证明：新归档命令实际驱动的 B0 binary + B0 cfg.para forcing 早期段 → byte-equal output。新旧 goldens 在 spec 角色上不同——新 goldens（case-relative seconds 命名）是 S1 B1a CI 的唯一权威基准；旧 goldens 仅作 historical reference 共存于 archive 目录。

### Before-PassValue 12 张 golden

为支持 S1b 阶段的 PassValue 边界 snapshot 比对（design.md D11 + D13；tasks.md task 0.1b），在 pre-S1a 第二轮归档 12 张 before-PassValue snapshot golden（4 case × 3 t_values，与上面 12 张 after-PassValue golden 同 t-集合 `[86400, 2592000, 7776000]`）。这组 golden 由 `MD_f.cpp:67` PassValue() 调用**之前**插入的 `shud_rhs_dump_point("f_loop_before_passvalue", t, Qe2r_Surf, NumEle)` 钩子产生，与现有 12 张 `snapshot_t<v>.bin`（来自 `MD_update.cpp:151` "f_update" 钩子，即 f_update 末端 DY=0 数组）**语义对照**：

| Snapshot 后缀 | 来源 site tag | 钩子位点 | 实际 dump payload | 语义 |
|---|---|---|---|---|
| 无（`snapshot_t<v>.bin`） | `"f_update"` | `SHUD/src/ModelData/MD_update.cpp:151` | `DY[0..NumY-1]`（已被 L147-149 reset to 0） | f_update 末端 RHS clear 后 baseline |
| `_before_passvalue.bin` | `"f_loop_before_passvalue"` | `SHUD/src/ModelData/MD_f.cpp:67`（PassValue 之前） | `Qe2r_Surf[0..NumEle-1]`（per-element-to-river surface flux written by `PassValue()`，长度 NumEle） | f_loop 内 lake→ET→element→segment→river DY 计算完毕、PassValue() 即将清零并重算前的元素→河道地表通量状态 |

**PR #54 round-1 fix F4 — payload re-pick**：本节最初的 12 张 golden（PR #43 落地）使用 `QeleSurfTot` 作为 payload，但验证组 (verifier) 发现 `QeleSurfTot` 在 `f_update` 末端被 reset to 0，且不在 PassValue() 的 write set 内 → snapshot 全 0（除了 header），无法用于 S1b 的 before-vs-after PassValue diff 验证。F4 把 payload 改为 `Qe2r_Surf`，它是 PassValue() write set 的成员（PassValue 第 189-200 行先清零再从 QsegSurf 累加），所以本探针抓到的是**前一次** PassValue 的 Qe2r_Surf 残留——B1a 重构若改动 PassValue 调度顺序或语义即可在此 byte-diff。验证：3/4 case (xinanjiang_upstream / qinyijiang / qhh) 的 Qe2r_Surf 在 ≥1 个 t_value 上有 nonzero values（说明探针抓到了真实 flux 状态）；keliya 因为是干旱内陆 case，90 day 窗口内 overland-to-river 通量全 0（与 hydrology 一致，非 bug）。F4 fix 的 12 张新 SHAs 见下表，与旧 QeleSurfTot 版相比：keliya 3 张 SHA unchanged（两数组都全 0），其它 9 张全部 DIFF。

**SHUD_DUMP_FNAME_SUFFIX 写入器扩展**：两个 site 在同一 run 内 dump 同一 t_value 会因为旧写入器只输出 `snapshot_t<v>.bin` 而 collision overwrite，#43 在 `SHUD/src/ModelData/MD_rhs_dump.cpp::init_config()` 引入新环境变量 `SHUD_DUMP_FNAME_SUFFIX`：

- 默认空字符串 → filename 保持 `snapshot_t<v>.bin`（与 PR #53 的 12 张文件名 + SHA256 完全 back-compat）
- 非空 → filename 改为 `snapshot_t<v>_<suffix>.bin`（#43 用 `before_passvalue`）
- 路径分隔符 `/` 或 `\\` 被拒绝（path-traversal guard，setting `c.disabled = true` + stderr 提示）
- 全部扩展在 `#ifdef SHUD_DUMP_RHS` 内 → DUMP=0 编译产物字节零变更（compile-switch neutrality 硬契约，本次 4 case 全部 PASS）

写入器副本 `tools/rhs_snapshot/writer.cpp` 同步增加 `compose_snapshot_filename()` + `compose_snapshot_filename_from_env()`（per MD_rhs_dump.cpp:10-19 SCHEMA DUPLICATION NOTE），以便未来外层 profile 工具走 writer 时行为一致。

**重新归档命令**（B0-tag binary + #43 patch，90-day cfg.para 截断，本地 macOS Apple Silicon Apple Clang 17.0.0）：

```bash
cd SHUD && make clean && make SHUD_DUMP_RHS=1 SHUD_ENABLE_PROFILE=0 shud && cd ..

for c in keliya:keliya:12053 xinanjiang_upstream:xinanjiang:0 \
         qinyijiang:nanlin:366 qhh:qhh:8401; do
  case=$(echo "$c" | cut -d: -f1)
  proj=$(echo "$c" | cut -d: -f2)
  start_day=$(echo "$c" | cut -d: -f3)
  start_min=$((start_day * 1440))
  abs_t="$((start_min + 1440)),$((start_min + 43200)),$((start_min + 129600))"
  staging="/tmp/snapshot_staging_$case"
  rm -rf "$staging" && mkdir -p "$staging"
  cd SHUD/Basins/$case
  rm -rf output/$proj.out
  SHUD_DUMP_T_VALUES="$abs_t" SHUD_DUMP_T_TOL=60 \
  SHUD_DUMP_OUTPUT_DIR="$staging" SHUD_DUMP_CASE_ID="$proj" \
  SHUD_DUMP_SITE="f_loop_before_passvalue" \
  SHUD_DUMP_FNAME_SUFFIX="before_passvalue" \
    ../../shud $proj
  cd ../../..
  # 文件名 abs-min → case-relative-sec rename + 复制到 benchmarks/
  for rel_sec in 86400 2592000 7776000; do
    case "$rel_sec" in
      86400) abs_min=$((start_min + 1440)) ;;
      2592000) abs_min=$((start_min + 43200)) ;;
      7776000) abs_min=$((start_min + 129600)) ;;
    esac
    cp "$staging/snapshot_t${abs_min}_before_passvalue.bin" \
       "benchmarks/$case/B0_output/snapshot_t${rel_sec}_before_passvalue.bin"
  done
done
```

**12 张 before-PassValue golden SHA256**（2026-06-18 archived；PR #54 round-1 F4 fix 用 Qe2r_Surf 取代 QeleSurfTot 后**重新归档**；本地 macOS Apple Silicon、Apple Clang 17.0.0；server re-archival 在 S1a 流程内手动一致性验证）：

| Case                  | t_value (s) | File                                                                          | SHA256                                                             |
|-----------------------|-------------|-------------------------------------------------------------------------------|--------------------------------------------------------------------|
| `keliya`              | `86400`     | `benchmarks/keliya/B0_output/snapshot_t86400_before_passvalue.bin`            | `672a6213379d8c767944f40c3d254ab36ee757e891fe1cb64f5e327143567580` |
| `keliya`              | `2592000`   | `benchmarks/keliya/B0_output/snapshot_t2592000_before_passvalue.bin`          | `69c2130f7414f3bdeff0d66a35f481a79f0bf62177144b624c8c05b60705f5a7` |
| `keliya`              | `7776000`   | `benchmarks/keliya/B0_output/snapshot_t7776000_before_passvalue.bin`          | `a0960a5cabbcea03a94bc853ddcbfa99489ef74f03c0ecd74c848a7556f9593d` |
| `xinanjiang_upstream` | `86400`     | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t86400_before_passvalue.bin`   | `9fa63a8386770b1643049a43351a2bd235790e8db5c3857145eb2bf8b86bc528` |
| `xinanjiang_upstream` | `2592000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t2592000_before_passvalue.bin` | `6523dae92221a8a0ee6416c6ecc7271e0c2085b12c8157fd69edeb41fc8d6cdb` |
| `xinanjiang_upstream` | `7776000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t7776000_before_passvalue.bin` | `cdfb07f60baa1bf91db9fa7672037aa3ceacb50aa999815d7029b7eea2ced1fd` |
| `qinyijiang`          | `86400`     | `benchmarks/qinyijiang/B0_output/snapshot_t86400_before_passvalue.bin`        | `4baffc0dacebf61970d92d781e85eb65e6db8aef87ce38c1667e8b8190f84cf1` |
| `qinyijiang`          | `2592000`   | `benchmarks/qinyijiang/B0_output/snapshot_t2592000_before_passvalue.bin`      | `4e7798ee53fb772450d3e58574888798c8dc8b732567ec93c94e9f6c607d3b4d` |
| `qinyijiang`          | `7776000`   | `benchmarks/qinyijiang/B0_output/snapshot_t7776000_before_passvalue.bin`      | `4ef0a05b89cf9681e9a3e8686e7c469939473f56d5a42321f0105d6070b93c04` |
| `qhh`                 | `86400`     | `benchmarks/qhh/B0_output/snapshot_t86400_before_passvalue.bin`               | `2ed538cb9fab354e04f5dadfa166463a324399988d5a19c1934966854669807c` |
| `qhh`                 | `2592000`   | `benchmarks/qhh/B0_output/snapshot_t2592000_before_passvalue.bin`             | `bc2d5adc0a9ad96d11185464308974db580ee5f50d2e799e08eed928773bc1d8` |
| `qhh`                 | `7776000`   | `benchmarks/qhh/B0_output/snapshot_t7776000_before_passvalue.bin`             | `bddeaa2e3d32c0842fead9d65e380212550cb918fce8a1c44ea6d30a8087473b` |

算法：`shasum -a 256 benchmarks/<case>/B0_output/snapshot_t<v>_before_passvalue.bin`（Linux 上用 `sha256sum`）。文件 size = `40 (FileHeader) + 12 (RecordHeader) + 4 (name_len) + 2 ("DY") + 8 (nelem) + 8 * NumEle (payload)`：keliya `3938 B` (NumEle=484)、xinanjiang_upstream `6474 B` (NumEle=801)、qinyijiang `25306 B` (NumEle=3155)、qhh `38250 B` (NumEle=4773)，与各 case `Model_Data::Qe2r_Surf` 长度一致（== NumEle，per `Model_Data.hpp:134` + `PassValue()` 第 189-192 行 zero-reset 循环）；binary header 与 `tools/rhs_snapshot/format.h v1` 完全相同（version 1, magic "SHRH"）。

**PR #53 12 张 after-PassValue golden 仍 unchanged**：本节扩展不修改 PR #53 已落入的 12 张 `snapshot_t<v>.bin`，文件名 + SHA256 全部稳定（实测同 commit 内对比，全 12 张 byte-equal vs L123-134 列表）。`SHUD_DUMP_FNAME_SUFFIX` 默认空字符串保证 back-compat。

### 3-run repeatability evidence

Each case ships `benchmarks/<case>/B0_output/repeatability_snapshots.txt` documenting 3 independent runs × 3 t-values = 9 SHA256 rows. All 9 SHAs per file are identical across the 3 runs → deterministic.

Producer: `tools/snapshot_repeatability/run.sh <case>` — runs the SHUD build with SHUD_DUMP_RHS=1, dumps before-PassValue snapshots, computes SHA256, writes the file.

Drift detector: `tools/check_goldens/check_goldens.sh` cross-checks the 12 SHA256 rows in repeatability_snapshots.txt against the 12 shipped `snapshot_t<v>_before_passvalue.bin` files; fails on any mismatch.

**SHUD_DUMP_FNAME_SUFFIX 启用方式**（任意需要二者并存的 dump run 通用模式）：

```bash
# After-PassValue (legacy filename，与 PR #53 golden 对齐):
SHUD_DUMP_SITE=f_update                                  ./shud <proj>
# Before-PassValue (新 #43 钩子 + 新文件名):
SHUD_DUMP_SITE=f_loop_before_passvalue \
SHUD_DUMP_FNAME_SUFFIX=before_passvalue                  ./shud <proj>
```

两组命令对同一 run 跑出**不同**文件（无 collision overwrite），同 `SHUD_DUMP_OUTPUT_DIR` 可并存。

**Cross-reference**：design.md D13 给出 site / writer / filename 三层 reconciliation 的完整契约；本节是 D13 + D11 落地的部署侧记录。

### 1.5.a 部署层 cfg.para 模板（S1a #44 周期性 IC backup probe 用）

S1a 1.5.a 验证：在 90-day 窗口内通过 `Update_IC_STEP = 43200`（30 day, minutes）触发周期性 IC backup（`MD_update.cpp:234` 的 `PrintInit` 调用）；B1a 重构后 `rhs_core(ExecPolicy::Serial)` 路径下 backup 时刻的 RHS 评估状态 vs B0 binary 同时刻 byte-equal。**SHUD 本身没有 CVODE re-init 路径**（grep 验证 `CVodeReInit` 0 hits）；warm-restart 本质是"中途 IC 持久化 + 下次冷启动 from .ic"，不是 mid-run state 注入。本 issue (#42) 仅交付**模板/文档片段**，实际跑由 S1a #44 执行。

```text
# 部署层 cfg.para override 示例
# 文件：SHUD/Basins/keliya/input/keliya/keliya.cfg.para
#
# 在 90 天窗口内触发周期性 IC backup：
#   START          12053    # day-index, 1951-01-01 epoch
#   END            12143    # = START + 90, 项目级铁律截断
#   INIT_MODE      3        # INIT_MODE = 3 (default): load IC from .ic file at startup;
#                           # 与 Update_IC_STEP 配合使下次 cold restart 从最近 backup 起步
#   Update_IC_STEP 43200    # 43200 min = 30 day; window markers at +30, +60, +90 day
#                           # (SHUD 解析 keyword 大小写不敏感；单位 = 分钟，
#                           #  默认 1440 见 Model_Control.hpp:111)
#
# 此 override 由 SHUD/.git/info/exclude 的 /Basins/ pattern 屏蔽，
# 不污染 SHUD submodule、不入 outer repo PR、按 case 部署时直接 in-place 改。
```

**注意事项**（给 #44）：
- `Update_IC_STEP` 单位是**分钟**（非 day-index）；30 day 必须写 `43200`，不能写 `30`。SHUD 在 `MD_update.cpp:234` 用 `t_long % CS.UpdateICStep` 判定 backup boundary（`t_long` 也是分钟单位），写 30 会让每 30 min 触发一次。
- backup 边界点必须落在 ≤ END（即 `START_min + Update_IC_STEP` 与 `START_min + 2*Update_IC_STEP` 都 ≤ `START_min + 90*1440`）。43200 min step 在 90 day 窗口里给 2 个中途触发点（+30 day、+60 day）+ 1 个边界（+90 day），足够覆盖周期性 backup 路径。
- B1a bitwise gate by #44 task 1.5.a：在 backup boundary 时刻（`t mod Update_IC_STEP == 0`，例如 +30d / +60d）dump RHS snapshot，与 B0 binary 跑同输入到同时刻的 snapshot byte-equal。注意 SHUD 不做 mid-run CVODE state 注入，因此 backup 本身只是文件持久化、不影响后续积分；本 probe 只验证 backup boundary 那一拍的 RHS 评估状态 B1a vs B0 byte-equal。

**CI hookup 已完成（PR #54）**：`.github/workflows/serial-baseline.yml` 的 `SHUD_DUMP_T_VALUES` 已使用新 abs-min `[17357760, 17399520, 17485920]`（keliya，`START_day=12053 × 1440 + {1440, 43200, 129600} min`）；missing snapshot 走 `::error` hard-fail，旧 silent-skip via `::notice` 路径已删除。其它 case 同公式：xinanjiang_upstream (START=0) `[1440, 43200, 129600]`；qinyijiang (START=366) `[528480, 570240, 656640]`；qhh (START=8401) `[12098880, 12140640, 12227040]`——其余 3 case 的 nightly extension 由 capability `b0-tag-ci-integration` task 6.1 / 6.2 落地。当前 keliya 单 case PR fast-feedback 已 gate B1a 新 24 张 golden。

## 禁用 flag（Makefile 守卫）
- `-ffast-math`、`-Ofast`、`-funsafe-math-optimizations`
- 二进制正确性在任何形式的注入（CLI、env、MAKEFLAGS 等）下都契约保证。用户面"显性报错"UX 只对 CLI 形式提供；针对两个项目内 lock 变量的 env 形式注入是静默（二进制安全，但不会 emit `$(error …)`——见下面 Layer 3 警示）。
- 三层保护锁定：
  1. **Layer 1 — 在 8 个载体上 `filter`（显性报错）**：扫描任何禁用 flag 是否出现在 6 个标准载体（`CFLAGS`、`CXXFLAGS`、`CPPFLAGS`、`LDFLAGS`、`MAKEOVERRIDES`、`MAKEFLAGS`）+ 2 个项目内 lock 变量（`SHUD_BUILD_CFLAGS`、`CXX_BASE_FLAGS`）里。词级 `filter` 抓得到 `make shud CXXFLAGS=-Ofast`、`make shud CFLAGS=-Ofast` 等。
  2. **Layer 2 — `$(MAKEOVERRIDES)` 上锚定的 `=value` 扫描（显性报错）**：遍历 `$(MAKEOVERRIDES)` 的 `VAR=value` token，用 `filter %=<flag>` 精确匹配 `VAR=<禁用 flag>` 的 CLI 赋值。抓得到 `make shud SHUD_BUILD_CFLAGS=-Ofast` / `CXX_BASE_FLAGS=-Ofast`——否则 lock 变量上的 `override :=` 会静默丢弃这些注入。锚定形式（相比早期字面 `findstring`）避免对合法值（含子串）的误报，例如 `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`。`DISALLOWED_FLAGS` 自身受 `override` 保护，避免 `make DISALLOWED_FLAGS=` 解除扫描列表。
  3. **Layer 3 — 对 `CXX_BASE_FLAGS` + `SHUD_BUILD_CFLAGS` 用 `override … :=`（静默保证）**：按 GNU make 手册，`:=` 上的 `override` 让 make-CLI 赋值被静默忽略。Recipe 直接展开 `$(SHUD_BUILD_CFLAGS)`（`CXX_BASE_FLAGS` 的别名），所以编译器看到的就是锁定后的 flag 集。**警示——env 形式注入是静默**：`SHUD_BUILD_CFLAGS=-Ofast make shud` 被静默忽略（二进制安全，因为 recipe 仍用锁定 flag），但不报错——因为 Layer 1/2 只通过 `MAKEOVERRIDES` 看到 CLI 形式赋值。这是有意的：契约只保证二进制正确性；显性报错 UX 服务于最常见的注入向量（CLI 形式）。后续更深的修复（Layer 3 的 env 来源探测，用 `$(origin VAR)`）作为 follow-up 跟踪。
- Recipe 直接展开 `$(SHUD_BUILD_CFLAGS)`（`CXX_BASE_FLAGS` 的别名），所以用户提供的 `make CFLAGS=…` 没法盖掉锁定 flag——Layer 1 会抓到，编译之前就 fail。
- 编译器默认通过 `$(origin CXX)` check pin 住（旧的 `CXX ?= g++` 在 GNU make 的内置 `c++` 面前是个 no-op）；env / CLI `CXX=...` 仍受尊重。

## SUNDIALS 版本 + 安装完整性守卫
- `check_sundials` Makefile target 强制（锚定 regex，不做子串匹配）：
  - `SHUD/InstallSundials/include/sundials/sundials_config.h` 中 `^#define SUNDIALS_VERSION_MAJOR 6$`
  - `^#define SUNDIALS_VERSION_MINOR 0$`（B0 要求 6.0.x；6.1+ 拒；PATCH 不强制）
  - `$(SUNDIALS_DIR)/lib/` 下存在 `libsundials_cvode.*`
  - `$(SUNDIALS_DIR)/lib/` 下存在 `libsundials_nvecserial.*`
- `check_sundials_omp` 额外要求 `libsundials_nvecopenmp.*`；`shud_omp` 依赖它（`shud` 只依赖 `check_sundials`）。
- `./configure` 在幂等短路和安装后的报告里都跑同样的 MAJOR + MINOR + lib 检查。
- `./configure` 总是重新解压 `cvode-6.0.0/`（删了重 untar），避免上次中断留下的脏 / 残缺目录污染 cmake。

## macOS libomp 守卫
- `make shud_omp` 检查 `LIBOMP_PREFIX`（来自 `brew --prefix libomp`）；空则报错给装机说明。只 gate 在 `shud_omp` 上，`make shud`（serial）即使没有 libomp 也能跑。
- Linux：`LIB_OMP` 为空被 `$(if …)` 包住，避免链接行上出现裸 `-L`。

## SUNDIALS_DIR 覆盖纪律
- 为了 B0 可复现，`SUNDIALS_DIR` 必须指向 `SHUD/InstallSundials/`（捆绑的安装）。
- 覆盖在技术上是可以的（`make shud SUNDIALS_DIR=/external/path`），但结果二进制不 B0-comparable，**不得**用于 benchmark 归档（#8）、CI（#13）或 A0 验收（#17）。`check_sundials` 只验证目标 SUNDIALS 的 MAJOR + MINOR 和必需 lib 的存在——不验证它的编译 flag 或构建 provenance。一个系统装的、用 `-Ofast` 构建的 SUNDIALS 会通过守卫，但会破 A3a bitwise。
- 如果主机上有满足锁定 flag 集的外部 SUNDIALS 6.0.x，**可以**用；在 PR 描述里记录覆盖事实和理由。

## 当前主机的 SUNDIALS 安装
- 版本：`6.0.0`
- 安装大小：`26M`
- 路径：`SHUD/InstallSundials/`

## P1-update-omp-tag（P1 epic / Issue #211 capstone）

`P1-update-omp-tag` annotated git tag pin 住了 P1 epic 全套 phase（Phase A m7-forcing-trim + Phase B profile-retest-m7 / Opt-IO 决策 + Phase C.audit p1-state-update-parallel pre-audit + Phase C.implement PR-D element + PR-E river + PR-F lake 3-pragma stack + Phase C.verify PR-H snapshot + PR-I/J fullrun + PR-K1/K2 scaling + Phase D.tag PR-L）完成后认证的那一对 `(outer, SHUD submodule)` commit。P2+ 起回归比对 baseline 切到 `P1-update-omp-tag`（B0-tag / B1a-tag / B1b-tag / B1-tag 仍可比对，但 P1-update-omp-tag 是 P2 阶段的首个 parallel-candidate 参考点）。

> **节有效性说明**：下面的命令（`git rev-parse P1-update-omp-tag`、`git show P1-update-omp-tag --stat -- SHUD`）只有在项目所有者 `git tag -a P1-update-omp-tag <merge-commit-sha>` + `git push origin P1-update-omp-tag` 之后才能成功；这一动作发生在 PR-L #224 中（2026-06-22）。在那之前 `P1-update-omp-tag` 不存在。
> 实时状态见下面的 `## P1-update-omp-tag 应用状态`。

> **D11 强制**：P1-update-omp-tag 一次锁死，**禁止 force-update**（与 B1b-tag 一致；与 B1a-tag force-update 历史不同）。任何后续 retroactive 更新（如 P2 sub-stage stacking）走 forward-compat **P1c-tag stacking / P2-* stacking** 路径（master plan C8）；design D9 fast-path 可能触发 "P1-update-omp-tag extension" 走 P1c 不新建 P-strict baseline。

- **外层 repo tag**：`P1-update-omp-tag` 在 `main` 分支上、PR-K2 #223 squash-merge + #223 post-merge log append (commit `003f58d`) 上打（PR-L #224 capstone 完成）。
- **P1-update-omp-tag 时刻的 SHUD submodule pin**：`07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（外层 tag commit 抓住的 submodule pointer，`openmp-baseline` 分支 HEAD；3-pragma stack 完整 = element loop L64-L105 + river loop L107-L125 + lake loop L136-L147 三 `#pragma omp parallel for schedule(static) default(none)` 已落地）。
- **D11 / D12 收尾约束**：
  - Mac 4-case RHS snapshot bitwise vs B1b/B1-tag canonical golden = 12/12 PASS (`shud_omp @ NUM_OPENMP=1` + `SHUD_DUMP_RHS=1` + `SHUD_DUMP_SITE=f_update` per `docs/p1/p1_rhs_snapshot_bitwise.md`)
  - Mac 4-case full-run canonical summary SHA ≡ `benchmarks/<case>/B0_output/repeatability.txt sha256_run1` + CVODE 15-key byte-identical = 8/8 + 4/4 PASS (`shud` serial via `tools/archive_b0_output.sh` per `docs/p1/p1_fullrun_bitwise.md` §1-§6)
  - Server 2-case (heihe / heihe_x4) cn03 Slurm `shud` serial full-run canonical summary SHA ≡ B1b/B1-tag golden = 4/4 PASS (jobid 8794/8795 per `docs/p1/p1_fullrun_bitwise.md` §"Server section")
  - PR-D / PR-E / PR-F 5-step push workflow 严格遵守（SHUD pin trail `017c629 → 6a9e684 → 08898a3 → 07c677f`）
  - `baseline/P1` 分支 protection `lock_branch=true` + `enforce_admins=true` + `allow_force_pushes=false` + `allow_deletions=false` enforced
  - §1.1.1 verdict = **WARNING（P1 epic 不阻塞）** per design D5 NG3 + master plan §6 P7 final-fusion debt（详 `docs/p1_summary.md` §"§1.1.1 verdict"）：heihe sp@8=1.08× (Amdahl-bound ~1.13× ✓ + "不独立验收" carve-out)；heihe_x4 sp@8=1.14× (P7 strict M=1.8× 退出门，P1 起点报告)；A3a/A3b strict at N≥4 4/6 cells dual-FAIL with CVODE nst bifurcation (heihe nst per N: 6773/6773/6585/6684)；root cause hypothesis = B1b S3c owner-local gather tree-reduction-depth N>2 transition；P7 final-fusion deterministic-reduction debt logged

## P1-update-omp-tag 应用状态

| 字段 | 值 |
|---|---|
| `P1-update-omp-tag-applied` | `true` |
| `P1-update-omp-tag-date` | `2026-06-22` |
| `P1-update-omp-tag-object-sha` | `ff21c75c8e968d5e47ca53b015425360be9ac879`（annotated） |
| `P1-update-omp-tag-commit-sha` | `003f58dc079116ef2161d2f96006228ef0e013d0`（PR-K2 #223 squash-merge + #223 post-merge log append on `main` 之后；PR-L #224 capstone 创建 tag） |
| `P1-update-omp-tag-SHUD-pin` | `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47`（`openmp-baseline` branch HEAD） |
| `P1-warning-not-blocked` | `yes`（§1.1.1 verdict = WARNING per design D5 NG3；详 `docs/p1_summary.md` §"§1.1.1 verdict"；P7 final-fusion debt logged） |

`P1-update-omp-tag` push 后命令全部成功（已 2026-06-22 验证）：
- `git rev-parse P1-update-omp-tag` → `ff21c75c…`
- `git rev-parse P1-update-omp-tag^{}` → `003f58d…`
- `git show P1-update-omp-tag --stat -- SHUD` → SHUD pin 显示 `07c677f`
- `git ls-remote --tags origin | grep P1-update-omp-tag` → 远端 tag 存在
- `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection` → `lock_branch=true`

### P1 build provenance（PR-K2 #223 capstone 时刻 + PR-L #224 tag lock）

| 项 | 值 | 来源 |
|---|---|---|
| Stage | P1 (update-omp) | master plan §3 P1 行 |
| SHUD pin | `07c677fe3b449f706a2b1f9663ae3cdd60aa7b47` | `openmp-baseline` branch HEAD post-PR-F |
| Outer commit (P1-update-omp-tag^{}) | `003f58dc079116ef2161d2f96006228ef0e013d0` | PR-K2 #223 post-merge log append on `main` |
| Mac binary (`SHUD/shud_omp`) sha256 | **NOT PINNED** (Apple Clang versioning drift; reproducible via `make shud_omp` post-checkout `P1-update-omp-tag^{}` on matching Apple Clang 17.0.0 toolchain; ephemeral binary kept under `.s2-103/pr-k1/` scratch only) | `docs/p1/p1_perf_baseline.md` §1.2 / `.s2-103/pr-k1/` |
| Mac binary (`SHUD/shud` serial) sha256 | **NOT PINNED** (Apple Clang versioning drift; reproducible via `make shud` post-checkout `P1-update-omp-tag^{}` on matching Apple Clang 17.0.0 toolchain) | `docs/p1/p1_fullrun_bitwise.md` §"Build evidence" |
| Server binary (`SHUD/shud_omp`) sha256 | `b637537c53ff446b9885f949c19f20e50eba53296ef417ea5a5924fa803b2865` (built in-sbatch on cn03 per PR-K2) | `docs/p1/p1_perf_baseline.md` §2.1 / §2.3 |
| Server binary (`SHUD/shud` serial) sha256 | `3e9e56295528b0399aff928d1b44d708da87b37777ea81e0de216a3d12a975f3` (cn03 PR-J #221) | `docs/p1/p1_fullrun_bitwise.md` §"Server section" L307 |
| Mac compiler | Apple Clang 17.0.0 (clang-1700.6.3.2) arm64-apple-darwin24.6.0 | `docs/p1/p1_perf_baseline.md` §1.2 |
| Linux compiler (server cn03) | GCC `13.3.0-6ubuntu2~24.04.1` | `docs/p1/p1_perf_baseline.md` §2.3 + `docs/p1/p1_fullrun_bitwise.md` §"Server section" L250 |
| Strict FP flags (Mac + server) | `-O2 -g -ffp-contract=off -fno-fast-math -std=c++14`（serial `make shud`）; 同上 + `-fopenmp` / `-Xpreprocessor -fopenmp`（OMP `make shud_omp`，per platform）— 3-grep gate `-O2 / -ffp-contract=off / -fno-fast-math ≥ 1 hit each` + `-ffast-math / -Ofast / -funsafe-math-optimizations` 0 hit 全 PASS | `docs/p1/p1_rhs_snapshot_bitwise.md` §"Build evidence" + `docs/p1/p1_fullrun_bitwise.md` §"Server compile" + 本文档 §1 / §2 |
| OMP runtime env | `OMP_PROC_BIND=close OMP_PLACES=cores`（server PR-K2 `--cpus-per-task=8`）; Mac libomp（NG1 dev-only，不强制 binding per design D5）| `docs/p1/p1_perf_baseline.md` §2.3 |
| Slurm jobids (server cn03) | PR-J: `8794` heihe (Elapsed 00:27:18) + `8795` heihe_x4 (Elapsed 01:08:45)；PR-K2: `8796` heihe (Elapsed 00:09:10) + `8797` heihe_x4 (Elapsed 01:05:41) | `docs/p1/p1_fullrun_bitwise.md` §"Server section" L220-L221 + `docs/p1/p1_perf_baseline.md` §2.1 |
| Tag | `P1-update-omp-tag` = `ff21c75c…` (annotated) deref `003f58d…` | `git ls-remote --tags origin` |
| Baseline branch | `baseline/P1` (D11 locked: `lock_branch=true / enforce_admins=true / allow_force_pushes=false / allow_deletions=false`) | `gh api repos/DankerMu/SHUD-OpenMP/branches/baseline/P1/protection` |
| Cases (per status_matrix L23 P1 行) | keliya / xinanjiang_upstream / qinyijiang / qhh PASS (Mac local) + heihe / heihe_x4 PASS @ server cn03 + kashigeer N/A deferred-upstream | `docs/status_matrix.md` §"P1 行证据" |

## P1d-tag preparation（P1d epic capstone via E′ containment path / Issue #274）

P1d epic 走 **E′ containment closure** 后 `P1d-tag` annotated git tag 将 pin 住 P1c → P1d 阶段 13-PR + PR-C0 insertion = 14 PR 完成时刻的 `(outer, SHUD submodule)` commit。`P1d-tag` 由 PR-L 创建 + push（PR-K capstone 仅记录 SHUD pin trail，tag 创建动作不在 PR-K scope）。本节是 PR-K capstone 阶段的 SHUD pin trail 引用 + 后续 PR-L tag 创建前置 doc。

> **D11 强制**：P1d-tag 一次锁死，**禁止 force-update**（与 B1b-tag / P1-update-omp-tag / P1c-tag 一致）。任何后续 retroactive 更新走 forward-compat **P1e-tag stacking** 路径（master plan C8）。

### SHUD pin trail（P1c → P1d）

| 阶段 | SHUD pin | 说明 |
|---|---|---|
| P1c capstone (PR-K2 #223 / `P1c-tag`) | `3a0004c4c2a9a1d8eb586aba45186f8a2ff79df4` | Kahan-injected（P1c §4.7 conditional Neumaier 1974 注入 4 reduction helpers） |
| P1d PR-C/D/E (steady-state first-touch) | `de9545d` → `a2085de` → `7023ee9` → `6aada88` | 在 `MD_rhs_core.cpp::rhs_update` element + `rhs_flux` river + `rhs_update` lake 三 owner block 前置插 first-touch loop（M10 修订后这 3 loop 标 DEPRECATED — see `docs/p1d/p1d_first_touch_design.md` + `docs/p1d/p1d_numa_root_cause.md` §5） |
| P1d PR-G (Kahan revert) | **`210ac191...`** (final P1d SHUD pin) | 4 surgical revert in `MD_rhs_core.cpp`（去掉 `#include <cmath>` + 3 helpers 内 Neumaier branch），net +7/-47 vs `6aada88`；Mac 9-SHA matrix 证明 revert 干净 (per `docs/p1d/p1d_kahan_revert.md` §"SHA matrix"); post-PR-G == pre-K2 `de9545d` byte-identical at N=1 (server heihe N=1 SHA == `7f22bd6faa438d50...` = `P1-update-omp-tag` canonical) |

### `baseline/P1d` outer HEAD 推进

| 阶段 | Outer baseline/P1d HEAD | PR / merge note |
|---|---|---|
| PR-A (intake) | (post-intake commits) | epic propose + .gitignore + tasks |
| PR-B (NUMA env runbook) | `<merge SHA>` | `docs/p1d/p1d_numa_env_runbook.md` + 3-cell PR-B env-plumbing verification |
| PR-C0 (legacy dead-code deletion) | `<merge SHA>` | delete `f_update / f_loop / f_applyDY` 3 函数 + 3 declarations |
| PR-C / PR-D / PR-E (first-touch element / river / lake) | `<merge SHA>` × 3 | `MD_rhs_core.cpp` 3 #pragma 区 stack |
| PR-F (intermediate Kahan IN baseline) | `573dfdf` | server 8-cell + `--interleave=all` anti-pattern finding |
| PR-G (Kahan revert) | `21de1e2` | 4 surgical revert + Mac 9-SHA matrix |
| PR-H (final 8-cell + E′ post-verdict 修订) | `6f2522b` | 3 SHALL gate verdict FAIL + post-verdict 修订 |
| PR-I (Mac `P1-update-omp-tag` 6-cell anchor) | (PR-I merge SHA) | 独立 worktree 切 `P1-update-omp-tag` 跑 keliya + qhh × {serial, omp@N=1, omp@N=8} |
| PR-K (capstone docs, 本 PR) | `<post-PR-K merge SHA>` | 4 new + 2 update + 1 self-evidence doc (本 doc) |
| PR-L (next, tag-only) | `<post-PR-L merge SHA>` = `P1d-tag` deref | `P1d-tag` annotated + `baseline/P1d` lock |
| PR-M (last, PROMOTE) | `<post-PR-M merge SHA>` | PROMOTE 2 specs + archive + glossary + jsonl + Epic close + propose `p1e-strict-omp-rhs` |

PR-K capstone 时刻 outer HEAD = `a19fb5e`（master plan v1.5 / M10 merged 2026-06-24）；PR-L tag deref 将是 PR-K merge 后的 SHA + PR-L docs append。

### FP gate（与 P1c 一致，无变化）

P1d epic 全期 SHUD `Makefile` FP gate = **3-flag 形** `-O2 -ffp-contract=off -fno-fast-math -fopenmp`（per master plan §8.1.1）：

| Flag | shud | shud_omp |
|---|---|---|
| `-O2` | ≥1 | ≥1 |
| `-ffp-contract=off` | ≥1 | ≥1 |
| `-fno-fast-math` | ≥1 | ≥1 |
| `-fopenmp` (linux) / `-Xpreprocessor -fopenmp` (mac) | — | ≥1 |
| `-ffast-math` / `-Ofast` / `-funsafe-math-optimizations` | 0 | 0 |

PR-G post-revert + PR-K verify：3-grep ≥1 + 0-grep =0 全通过（无 FP gate 变化 vs P1c）。

### P1d-tag annotated message（PR-L 草拟，PR-K 仅引用）

PR-L 阶段创建 `P1d-tag` 时 annotated message 必须含（per master plan §6 P1d.5）：

1. Containment closure narrative（E′ path, 不是简单 E）
2. 5/5 codebase 事实核查（fact #1-#5 verbatim）
3. 4-mode spec rewrite (`serial` / `strict-omp` / `det-omp` / `fast-omp`)
4. PR-C/D/E steady-state first-touch deprecation note
5. PR-G Kahan revert 保留 + N=1 byte-identical to pre-K2 canonical
6. 指向 P1e (F 路) + 2×2 build matrix 因果实验 + ADR-0002 (solver-path)
7. SHUD pin 变更（P1c `3a0004c` → P1d `210ac19`）
8. PR-C/C0/D/E/F/G/H/I/K/L/M 11 PR cross-ref（PR-A/B 在 epic intake，PR-J cancel after PR-H FAIL）
9. D11 historical immutability re-verify (P1-update-omp-tag / B1-tag / B1a-tag / B1b-tag / P1c-tag SHAs 不变)

详 `docs/p1d/p1d_tag_and_lock.md`（PR-L 作者创建）+ `docs/p1d/p1d_report.md` §11 forward plan + master plan v1.5 §6 P1d.5。

## P1e-tag preparation（P1e epic capstone via §4.6.2 partial-closure / Issue #283）

P1e epic 走 **§4.6.2 partial-closure → SHIP** 后 `P1e-tag` annotated git tag 将 pin 住 P1d → P1e 阶段 13-PR + PR-J Phase 6 修订 = 14 PR 完成时刻的 `(outer, SHUD submodule)` commit。`P1e-tag` 由 PR-L 创建 + push（PR-K capstone 仅记录 SHUD pin trail + build flag matrix 文档化，tag 创建动作不在 PR-K scope）。本节是 PR-K capstone 阶段的 SHUD pin trail 引用 + 后续 PR-L tag 创建前置 doc。

> **D11 强制**：P1e-tag 一次锁死，**禁止 force-update**（与 B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag 一致）。任何后续 retroactive 更新走 forward-compat **P2-* stacking** 路径（master plan C8）。

### SHUD pin trail（P1d → P1e）

| 阶段 | SHUD pin | 说明 |
|---|---|---|
| P1d capstone (PR-L #301 / `P1d-tag`) | `210ac191...` | Kahan revert + first-touch loops stacked (E′ containment closure), `openmp-baseline` pushed, 全 mode 仍 Serial RHS |
| P1e PR-B0 (#311) | `7013de0...` → `f883914...` → `cc554cd...` → `9a422e5...` | rivqdown.dat tout-boundary recompute via recompute_for_output helper (Phase 6 add rhs_apply fix QeleSubTot/QeleSurfTot silent-zero + cv_y dump filename pad %015.6f → %020.6f for outer #310 F-R1-3) |
| P1e PR-F (#315) | `fb3cfe4...` → `226e3ab...` | shud.cpp SHUD_DUMP_CV_Y env-gated udata dump + ExecPolicy::StrictOMP impl per P1e-F design D2 |
| P1e PR-G (#315) | `85c8215...` | -fopenmp wiring auto + SHUD_RHS_THREADS env split + 拆为两段守门 form per tasks §3.5.2 (NVector branch `#ifdef SHUD_USE_OPENMP_NVECTOR` L132 + StrictOMP branch `#if defined(SHUD_ENABLE_OPENMP_RHS)` L157, 2 个独立 omp_set_num_threads call site) |
| P1e PR-H (#316, final) | **`3341368d2d0854924d2286925c8575df52cc97a0`** (final P1e SHUD pin) | remove 3 steady-state first-touch loops at MD_rhs_core.cpp L62-95 (element) / L169-203 (lake) / L324-354 (river) per design D4; convert omp single → omp for schedule(static) per design D2 + TSan-confirmed nowait 规则 (末 loop SHALL NOT nowait 守 phase 边界); mode A path bitwise preserved per PR-I §1.2 reproduction |
| PR-K (本 PR) | (no SHUD change) | capstone docs/p1e/ ≥12 + spec L192 amend + ADR-0002 close-out |
| PR-L (next, tag-only) | `<post-PR-L merge SHA>` = `P1e-tag` deref | `P1e-tag` annotated + `baseline/P1e` lock |
| PR-M (final, PROMOTE) | `<post-PR-M merge SHA>` | 2 spec PROMOTE + glossary 4 new terms + jsonl 双追加 + epic close |

### Build flag matrix 文档化（P1e era 新增）

P1e PR-G 后 build matrix 由 P1d era 的 2 build (`shud` + `shud_omp`) 升至 **4 mode**（per `docs/p1e/p1e_perf_baseline.md` §2 + `docs/p1e/p1e_thread_split.md` §3-§4）：

| Mode | 命令 | `SHUD_USE_OPENMP_NVECTOR` | `SHUD_ENABLE_OPENMP_RHS` | NVector backend | RHS path | 用途 |
|---|---|---|---|---|---|---|
| A | `make shud` | undef | undef | `N_VNew_Serial` | `ExecPolicy::Serial` | canonical reference baseline (mode A) |
| B | `make shud_omp` | `=1` | undef | `N_VNew_OpenMP` | `ExecPolicy::Serial` | 历史 prod (P1c/d era), drift control |
| C | `make shud SHUD_ENABLE_OPENMP_RHS=1` | undef | `=1` | `N_VNew_Serial` | `ExecPolicy::StrictOMP` | **P1e production 候选 (SHIP via §4.6.2)** |
| D | `make shud_omp SHUD_ENABLE_OPENMP_RHS=1` | `=1` | `=1` | `N_VNew_OpenMP` | `ExecPolicy::StrictOMP` | research 边界 (Phase 2 96-cell deferred) |

`-fopenmp` 自动 wire (PR-G Makefile)：`SHUD_ENABLE_OPENMP_RHS=1` 触发：

- Linux: 自动加 `-fopenmp` (compile + link)
- Darwin: 自动加 `-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include` (compile) + `-L$(brew --prefix libomp)/lib -lomp` (link)

binary symbol verification (per `docs/p1e/p1e_thread_split.md` §6 + `openspec/changes/p1e-strict-omp-rhs/specs/p1e-strict-omp-rhs/spec.md` "build C binary symbol" Scenario)：

| 验证 | command | 期望 (mode C) |
|---|---|---|
| NVector=Serial | `nm ./shud \| grep N_VNew_Serial` | ≥1 hit |
| NVector≠OpenMP | `nm ./shud \| grep N_VNew_OpenMP` | 0 hit |
| Linux OpenMP runtime 真链 | `nm ./shud \| grep GOMP_parallel` | ≥1 hit |
| Darwin OpenMP runtime 真链 | `nm ./shud \| grep _omp_set_num_threads` | ≥1 hit (Apple Clang 前导 `_`) |
| 守门形式 (PR-G 实施 = 拆为两段) | `grep -B2 -A2 'omp_set_num_threads' SHUD/src/Model/shud.cpp` | L132 `#ifdef SHUD_USE_OPENMP_NVECTOR` + L157 `#if defined(SHUD_ENABLE_OPENMP_RHS)` (二选一允许 union form, per spec L192 amend by PR-K) |

### P1e-tag annotated message（PR-L 草拟，PR-K 仅引用）

PR-L 阶段创建 `P1e-tag` 时 annotated message 必须含（per master plan §6 P1e.5 + `docs/p1e/p1e_summary.md` §9.PR-L）：

1. SHIP via §4.6.2 partial-closure narrative（不是简单 D12.1 happy path）
2. 3 SHALL gate verdict（AC-S1 + AC-S2 PASS + AC-S3 PARTIAL with carve-out rationale）
3. D12 4 branch eval (D12.1/.2/.3/.4 全 NOT triggered) + §4.6.2 active path 引用
4. 4-mode build matrix 完整 (mode A/B/C/D + binary symbol verify)
5. `SHUD_RHS_THREADS` per-case 运营建议 (heihe `=1` carve-out / heihe_x4 `=4` 推荐)
6. 指向 `docs/p1e/p1e_summary.md` + `docs/p1e/p1e_perf_baseline.md` (per tasks §7.A 说明)
7. SHUD pin 变更（P1d `210ac19` → P1e `3341368`）
8. PR-A/B/B0/C/D/E/F/G/H/I/J/K/L/M 14 PR cross-ref
9. D11 historical immutability re-verify (B1-tag / B1a-tag / B1b-tag / P1-update-omp-tag / P1c-tag / P1d-tag SHAs 不变, P1e-tag 新增)
10. ADR-0002 Status update reference (`Accepted (2026-06-24)` → `Implemented (P1e epic close, 2026-06-25)`)

详 `docs/p1e/p1e_summary.md` §9 forward handoff (PR-L) + `docs/p1e/p1e_report.md` §9 D11 7-tag chain final state + master plan v1.5 §6 P1e.5。

## CHANGELOG（S0-13 修订）

- S0-13 / #17：kashigeer 在 `benchmarks/INDEX.md` 里从 `local-and-server` 重分类为 `deferred-upstream`；`status-matrix` + `rhs-profile-gate` spec 修订，让 deferred-upstream 单元格成为 N/A 不阻塞；上面新增 `B0-tag` 一节；`docs/profile_decision.md` 由 DankerMu 通过 2026-06-17 的 delegated grant 签字，针对外层 `a860eae5` + SHUD `78c37a1`。

## 早期 CHANGELOG
- `fea5922`（PR #16 / issue #3）：初版 B0 构建环境锁定——锁定 flag 集、SUNDIALS 主版本守卫、幂等 `./configure`、macOS 通过 `brew --prefix` 探测 libomp。
- PR #18 round 2（SHUD 内 `c9368fd`）：invariant-closure 第 2 轮——封死禁用 flag 扫描里的 `CFLAGS` / `CPPFLAGS` / `LDFLAGS` 绕路；通过 `$(origin CXX)` pin 住 `CXX`，让 `c++` 不再默认胜出；用锚定 regex 收紧 SUNDIALS 守卫，新增 MINOR check + 对 `libsundials_cvode.*` / `libsundials_nvecserial.*` /（对 omp）`libsundials_nvecopenmp.*` 的 stat；新增 `check_sundials_omp` 给 OpenMP target；macOS 上 `shud_omp` 加 libomp `$(error)`；用 `$(if …)` 清掉裸 `-L` token；configure 总是重解压 `cvode-6.0.0/`；写明 `SUNDIALS_DIR` 覆盖纪律。
- PR #18 round 3（SHUD 内 `a9327b1`）：invariant-closure 第 3 轮——封死 round-2 verifier 暴露的 `SHUD_BUILD_CFLAGS` / `CXX_BASE_FLAGS` 绕路。两层保护：（1）两个 lock 变量都 `override … :=`（按 GNU make 语义静默忽略 make-CLI override）；（2）两层禁用 flag 守卫——Layer 1 `filter` 扩展到包含这 2 个 lock 变量（depth-in-defense），Layer 2 新增 `$(MAKEOVERRIDES)` 上的 `findstring` 扫描，抓到对 lock 变量的 CLI 赋值，把 `override` 的静默丢弃升级为显性 `$(error)`。Manifest §"禁用 flag" 更新列出所有 8 个载体。
- PR #18 round 5 —— W-R4 Warning 收敛：给 `DISALLOWED_FLAGS` 加 `override` 保护，避免 `make DISALLOWED_FLAGS=` 解除扫描；把 Layer 2 字面 `findstring` 三件套换成锚定 `=value` 迭代（`filter %=<flag>` 跑在 `$(MAKEOVERRIDES)` token 上），避免对路径含 `-Ofast` 子串的误报（例如 `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`）；订正 manifest §"禁用 flag" 诚实说明 env 形式 lock 变量注入是静默（通过 `override :=` 二进制安全，无 `$(error)` emit），并相应重述三层保护模型。
