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

> Sanity check note：keliya / qhh 的新 1d golden 与旧 4-year run 同 START 偏移处 bitwise 相同（同 START + 同 B0 binary + 同 forcing 早期段 → deterministic identical state），xinanjiang_upstream / qinyijiang 不同（推测旧归档跑了不同 forcing 段或 SPINUP 区间）。两者差异不影响新 golden 的 spec 角色——新 golden 是 B1a CI 的唯一权威基准。

### 1.5.a 部署层 cfg.para 模板（S1a #44 warm-restart probe 用）

S1a 1.5.a 验证 `INIT_MODE=3` warm-restart 路径下 B1a 重构后 SHUD CVODE re-init 行为不变。本 issue (#42) 仅交付**模板/文档片段**，实际跑由 S1a #44 执行。

```text
# 部署层 cfg.para override 示例
# 文件：SHUD/Basins/keliya/input/keliya/keliya.cfg.para
#
# 在 90 天窗口内中途触发 CVODE re-init：
#   START         12053    # day-index, 1951-01-01 epoch
#   END           12143    # = START + 90, 项目级铁律截断
#   INIT_MODE     3        # warm-restart hot-start mode (rewrite IC at restart)
#   UpdateICStep  30       # 30 day; window markers at [START, +30, +60, +90]
#
# 此 override 由 SHUD/.git/info/exclude 的 /Basins/ pattern 屏蔽，
# 不污染 SHUD submodule、不入 outer repo PR、按 case 部署时直接 in-place 改。
```

**注意事项**（给 #44）：
- `UpdateICStep` 字段值必须使 mid-window 触发点落在 ≤ END（即 `START + UpdateICStep` 与 `START + 2*UpdateICStep` 都 ≤ `START + 90`）。30d step 在 90d 窗口里给 2 个中途触发点（+30、+60）+ 1 个边界（+90），足够覆盖 warm-restart 路径。
- B1a bitwise neutrality（vs B0）要求 warm-restart 后 CVODE state 与冷启动到同时间点 bitwise 相同（同输入、同 RHS core），由 #44 通过 snapshot `t=2592000`（30d）跨 warm-restart cycle 验证。

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

## CHANGELOG（S0-13 修订）

- S0-13 / #17：kashigeer 在 `benchmarks/INDEX.md` 里从 `local-and-server` 重分类为 `deferred-upstream`；`status-matrix` + `rhs-profile-gate` spec 修订，让 deferred-upstream 单元格成为 N/A 不阻塞；上面新增 `B0-tag` 一节；`docs/profile_decision.md` 由 DankerMu 通过 2026-06-17 的 delegated grant 签字，针对外层 `a860eae5` + SHUD `78c37a1`。

## 早期 CHANGELOG
- `fea5922`（PR #16 / issue #3）：初版 B0 构建环境锁定——锁定 flag 集、SUNDIALS 主版本守卫、幂等 `./configure`、macOS 通过 `brew --prefix` 探测 libomp。
- PR #18 round 2（SHUD 内 `c9368fd`）：invariant-closure 第 2 轮——封死禁用 flag 扫描里的 `CFLAGS` / `CPPFLAGS` / `LDFLAGS` 绕路；通过 `$(origin CXX)` pin 住 `CXX`，让 `c++` 不再默认胜出；用锚定 regex 收紧 SUNDIALS 守卫，新增 MINOR check + 对 `libsundials_cvode.*` / `libsundials_nvecserial.*` /（对 omp）`libsundials_nvecopenmp.*` 的 stat；新增 `check_sundials_omp` 给 OpenMP target；macOS 上 `shud_omp` 加 libomp `$(error)`；用 `$(if …)` 清掉裸 `-L` token；configure 总是重解压 `cvode-6.0.0/`；写明 `SUNDIALS_DIR` 覆盖纪律。
- PR #18 round 3（SHUD 内 `a9327b1`）：invariant-closure 第 3 轮——封死 round-2 verifier 暴露的 `SHUD_BUILD_CFLAGS` / `CXX_BASE_FLAGS` 绕路。两层保护：（1）两个 lock 变量都 `override … :=`（按 GNU make 语义静默忽略 make-CLI override）；（2）两层禁用 flag 守卫——Layer 1 `filter` 扩展到包含这 2 个 lock 变量（depth-in-defense），Layer 2 新增 `$(MAKEOVERRIDES)` 上的 `findstring` 扫描，抓到对 lock 变量的 CLI 赋值，把 `override` 的静默丢弃升级为显性 `$(error)`。Manifest §"禁用 flag" 更新列出所有 8 个载体。
- PR #18 round 5 —— W-R4 Warning 收敛：给 `DISALLOWED_FLAGS` 加 `override` 保护，避免 `make DISALLOWED_FLAGS=` 解除扫描；把 Layer 2 字面 `findstring` 三件套换成锚定 `=value` 迭代（`filter %=<flag>` 跑在 `$(MAKEOVERRIDES)` token 上），避免对路径含 `-Ofast` 子串的误报（例如 `SUNDIALS_DIR=/opt/sundials-Ofast-tuned`）；订正 manifest §"禁用 flag" 诚实说明 env 形式 lock 变量注入是静默（通过 `override :=` 二进制安全，无 `$(error)` emit），并相应重述三层保护模型。
