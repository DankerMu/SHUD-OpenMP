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
| `B1a-tag-applied` | `true` |
| `B1a-tag-date` | `2026-06-18` |
| `B1a-tag-object-sha` | `4fafb8e570a020833395c7f57fe84eaabc7c7319`（annotated） |
| `B1a-tag-commit-sha` | `64569b3fa1826122262242e7cf14686384269cc9`（PR #70 squash-merge 到 `baseline/current`） |
| `B1a-tag-SHUD-pin` | `58327c5a114052ffe8f25b6d3e2aec6b404963f2` |

`B1a-tag` push 后，下列命令应全部成功：
- `git rev-parse B1a-tag` → tag object SHA
- `git show B1a-tag --stat -- SHUD` → SHUD pin 显示 `58327c5`
- `git ls-remote --tags origin | grep B1a-tag` → 远端 tag 存在

## S1-pre snapshot golden re-archival

为支持 S1（B1a refactor-equivalent serial reference），在 pre-S1a 重新归档 12 张 after-PassValue snapshot golden（4 case × 3 t_values）。**注意：本节 12 张 SHA 表已由 PR #71 / issue #55 第三次重生（详见本节末"#55 hook site 修正"段）；表格仅记录最新 SHA 值，旧 SHA 不保留**——历史 SHA 可经 `git log -- docs/build_manifest.md` 追溯。

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
  SHUD_DUMP_SITE="f_applyDY" \
  ../../shud $proj
  cd ../../..
  # rename to case-relative seconds + cp to archive (per-case mapping in tools/rhs_snapshot/README)
done
```

> Note：`SHUD_DUMP_SITE="f_applyDY"` 是 PR #71 / #55 第三次重归档采用的 site；初版（PR #53）写 `"f_update"`，但该 hook 落在 `MD_update.cpp:151`，紧接 L147-149 的 `DY[i]=0` reset 循环——dump 出来的 24 张 golden payload 全零，仅作 header-pin tampering detector，无 B1a 回归诊断价值。`f_applyDY` hook 落在 `MD_f.cpp:180` + `MD_rhs_core.cpp:350`（rhs_apply 末端），dump 整合后的 DY 状态。文件名 schema 不变（`snapshot_t<v>.bin`），只 SHA 因 payload 升级而变。

**12 张新 golden SHA256**（B0-tag = `884cfb13` outer + SHUD pin `78c37a1`，PR #71 / #55 re-archival on 2026-06-18；本地 macOS Apple Silicon、Apple Clang 17.0.0；server re-archival 在 S1a #44 验证一致）：

| Case                  | t_value (s) | File                                                       | SHA256                                                             |
|-----------------------|-------------|------------------------------------------------------------|--------------------------------------------------------------------|
| `keliya`              | `86400`     | `benchmarks/keliya/B0_output/snapshot_t86400.bin`          | `47ff82e4f996547e701c5b0a9fc6901110358719d301439246284ca51b6a8013` |
| `keliya`              | `2592000`   | `benchmarks/keliya/B0_output/snapshot_t2592000.bin`        | `b9cfc537eab7a7a654a94698c73259db7e188177193f59750c5277aa23c378ed` |
| `keliya`              | `7776000`   | `benchmarks/keliya/B0_output/snapshot_t7776000.bin`        | `a942dd32905ef8699775aba1df409a978877cc0099db4ebf1957a6649ecd9ba6` |
| `xinanjiang_upstream` | `86400`     | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t86400.bin`   | `3989ecbde70a4030b40efb1eab409fbe103c9ead75a015eecebe2027685faf55` |
| `xinanjiang_upstream` | `2592000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t2592000.bin` | `3b637b2d116e06fc0cab9edaf5d32b30703c79044b94e3a9896be8db401f69ab` |
| `xinanjiang_upstream` | `7776000`   | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t7776000.bin` | `ae5eac268c7e098c4ed869186d0dd499d71a84feac26cd71101982f7c584f9f8` |
| `qinyijiang`          | `86400`     | `benchmarks/qinyijiang/B0_output/snapshot_t86400.bin`      | `883bb0f50cf49f29a030cecd56e006ae27333355eab4b9617d0e062f341239cf` |
| `qinyijiang`          | `2592000`   | `benchmarks/qinyijiang/B0_output/snapshot_t2592000.bin`    | `6f183481afd6f3a3938953042a8f0e892a0de02d92a36a87049b659524c74e62` |
| `qinyijiang`          | `7776000`   | `benchmarks/qinyijiang/B0_output/snapshot_t7776000.bin`    | `798b5e0a119d285c7bef1ecd48d305db1211333cd8a55543e48b22a5a277eae0` |
| `qhh`                 | `86400`     | `benchmarks/qhh/B0_output/snapshot_t86400.bin`             | `6c75107d16ca5449953f101701120f131e56f42478583b6ff8a890179aae7413` |
| `qhh`                 | `2592000`   | `benchmarks/qhh/B0_output/snapshot_t2592000.bin`           | `186c6cba659d1be87734f6a186f4292ba95e439d22d5b500bf87ce0e1c9113b5` |
| `qhh`                 | `7776000`   | `benchmarks/qhh/B0_output/snapshot_t7776000.bin`           | `5a606aabf5808914eba1f84b442565addeb4fe74ae53ab2f5073f84c7498d604` |

算法：`shasum -a 256 benchmarks/<case>/B0_output/snapshot_t<v>.bin`（Linux 上用 `sha256sum`）。binary 内 `RecordHeader.t_value` 字段保留**绝对分钟**（不可改、跟 hook spec 一致；见 `format.h`）；只文件名是 case-relative seconds（对齐 manifest）。

**旧 S0-7 golden 处理**：保留作 historical reference（不删除）。命名是绝对分钟（如 `snapshot_t17357760.bin`、`snapshot_t17370720.bin`、`snapshot_t17500320.bin`），与新归档（case-relative seconds 命名）**文件名不冲突**，共存于同一 archive 目录。manifest.yaml 的 `snapshot_probe.t_values` 已 point 到新 t_values；下游 S1 bitwise gate 一律使用**新 golden**。

#### #55 hook site 修正（PR #71）

**问题**：PR #53 第一版 12 张 after-PassValue golden 由 `SHUD_DUMP_SITE="f_update"` 产生，该 hook 位于 `SHUD/src/ModelData/MD_update.cpp:151`，紧接 L147-149 的 `for(i=0;i<NumY;i++) DY[i]=0;` reset 循环——所有 12 张 golden 的 payload 全字节零（仅 header (66 字节；qhh 因 lake section 加 8 字节 = 74 字节) 非零），只能当作 header-pin tampering detector，**无 B1a 回归诊断价值**（详见 issue #55；与 PR #54 round-1 F4 fix 同源缺陷：probe payload 落在错的位置）。

**修正**：PR #71 采用 Option 3 — 复用 SHUD 已有的 `f_applyDY` hook（位于 `SHUD/src/ModelData/MD_f.cpp:180` 和 `SHUD/src/Model/MD_rhs_core.cpp:350`，rhs_apply 末端，dump 整合后的 `DY[0..3*NumEle+NumRiv+NumLake-1]`）。零 SHUD 源码改动、零 submodule pointer bump，仅外层 repo 改 `SHUD_DUMP_SITE` env var + 重生成 12 张 golden binary + 24 SHA 表。文件名 schema `snapshot_t<v>.bin` 不变（与 CI compare 路径 + manifest 完全对齐）；payload 从 zero-sentinel 升级为 post-integration diagnostic。

**Payload 非零验证**：4/4 case 在所有 3 个 t_value 上 payload 都包含大量非零字节（keliya ~6000+ non-zero / 14282 B；xinanjiang_upstream/qinyijiang/qhh ~75%+ non-zero）。**keliya 前 32 字节全零**是 case hydrology 而非 bug——keliya 是干旱内陆 case，第一批 element 的 DY (groundwater + unsaturated + surface) 在 90 day 窗口内的早期被 flux balance 推到 0；与 build_manifest L182 附近 F4 fix 关于 keliya 干旱 case 的描述同源。

**对旧 S0-7 golden 的关系**：旧的 4-year-run goldens（绝对分钟命名）payload 同样来自 `f_update` site，也是零 payload；仅作 historical reference 共存。下游 S1 / P3 bitwise gate 一律使用 PR #71 / #55 重生的新 12 张。

### Before-PassValue 12 张 golden

为支持 S1b 阶段的 PassValue 边界 snapshot 比对（design.md D11 + D13；tasks.md task 0.1b），在 pre-S1a 第二轮归档 12 张 before-PassValue snapshot golden（4 case × 3 t_values，与上面 12 张 after-PassValue golden 同 t-集合 `[86400, 2592000, 7776000]`）。这组 golden 由 `MD_f.cpp:67` PassValue() 调用**之前**插入的 `shud_rhs_dump_point("f_loop_before_passvalue", t, Qe2r_Surf, NumEle)` 钩子产生，与上面 12 张 `snapshot_t<v>.bin`（PR #71 / #55 之后由 `MD_f.cpp:180` / `MD_rhs_core.cpp:350` 的 `"f_applyDY"` 钩子产生，rhs_apply 末端 DY 整合后状态）**语义对照**：

| Snapshot 后缀 | 来源 site tag | 钩子位点 | 实际 dump payload | 语义 |
|---|---|---|---|---|
| 无（`snapshot_t<v>.bin`） | `"f_applyDY"`（PR #71 / #55 修正；前身是 `"f_update"` zero-payload，详见上面 "#55 hook site 修正" 段） | `SHUD/src/ModelData/MD_f.cpp:180` + `SHUD/src/Model/MD_rhs_core.cpp:350` | `DY[0..3*NumEle+NumRiv+NumLake-1]`（rhs_apply 整合后的 DY 整合状态） | f_applyDY 末端、PassValue 已经完成，DY 含 lake+ET+element+segment+river 全部累加结果 |
| `_before_passvalue.bin` | `"f_loop_before_passvalue"` | `SHUD/src/ModelData/MD_f.cpp:67`（PassValue 之前） | `Qe2r_Surf[0..NumEle-1]`（per-element-to-river surface flux written by `PassValue()`，长度 NumEle） | f_loop 内 lake→ET→element→segment→river DY 计算完毕、PassValue() 即将清零并重算前的元素→河道地表通量状态 |

**PR #54 round-1 fix F4 — payload re-pick**：本节最初的 12 张 golden（PR #43 落地）使用 `QeleSurfTot` 作为 payload，但验证组 (verifier) 发现 `QeleSurfTot` 在 `f_update` 末端被 reset to 0，且不在 PassValue() 的 write set 内 → snapshot 全 0（除了 header），无法用于 S1b 的 before-vs-after PassValue diff 验证。F4 把 payload 改为 `Qe2r_Surf`，它是 PassValue() write set 的成员（PassValue 第 189-200 行先清零再从 QsegSurf 累加），所以本探针抓到的是**前一次** PassValue 的 Qe2r_Surf 残留——B1a 重构若改动 PassValue 调度顺序或语义即可在此 byte-diff。验证：3/4 case (xinanjiang_upstream / qinyijiang / qhh) 的 Qe2r_Surf 在 ≥1 个 t_value 上有 nonzero values（说明探针抓到了真实 flux 状态）；keliya 因为是干旱内陆 case，90 day 窗口内 overland-to-river 通量全 0（与 hydrology 一致，非 bug）。F4 fix 的 12 张新 SHAs 见下表，与旧 QeleSurfTot 版相比：keliya 3 张 SHA unchanged（两数组都全 0），其它 9 张全部 DIFF。

**SHUD_DUMP_FNAME_SUFFIX 写入器扩展**：两个 site 在同一 run 内 dump 同一 t_value 会因为旧写入器只输出 `snapshot_t<v>.bin` 而 collision overwrite，#43 在 `SHUD/src/ModelData/MD_rhs_dump.cpp::init_config()` 引入新环境变量 `SHUD_DUMP_FNAME_SUFFIX`：

- 默认空字符串 → filename 保持 `snapshot_t<v>.bin`（与 PR #53 / PR #71 的 12 张 after-PassValue 文件名 schema back-compat；SHA256 在 PR #71 / #55 hook site 修正后变化，详见上面 "#55 hook site 修正" 段）
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

**PR #53 12 张 after-PassValue golden 已被 PR #71 / #55 重生**：本节 before-PassValue 扩展不修改文件名 schema，但 PR #71 / #55 fix 把 after-PassValue 12 张 binary 重生（hook site `f_update` → `f_applyDY`），文件名 `snapshot_t<v>.bin` 不变、SHA256 全部更新——见上面 after-PassValue SHA256 表 + "#55 hook site 修正" 段。`SHUD_DUMP_FNAME_SUFFIX` 默认空字符串仍保证 filename schema back-compat。

### 3-run repeatability evidence

Each case ships **two** per-side repeatability files under `benchmarks/<case>/B0_output/`:

- `repeatability_snapshots.txt`                    — before-PassValue side (PR #54 round-3 F27).
- `repeatability_snapshots_after_passvalue.txt`    — after-PassValue side (PR #71 / #55 follow-up).

Each file documents 3 independent runs × 3 t-values = 9 SHA256 rows. All 9 SHAs per file are identical across the 3 runs → deterministic. 4 cases × 2 sides × 3 unique-bin SHAs = 24 unique-bin checks total.

Producer: `tools/snapshot_repeatability/run.sh <case> [--site=before|after]` — runs the SHUD build with SHUD_DUMP_RHS=1, dumps snapshots at the requested site (`f_loop_before_passvalue` or `f_applyDY`), computes SHA256, writes the appropriate per-side file. Default `--site=before` preserves PR #54 behavior.

Drift detector: `tools/check_goldens/check_goldens.sh` cross-checks 24 build_manifest.md SHA rows (12 after + 12 before) + 12 unique-bin rows from before-PassValue repeatability files + 12 unique-bin rows from after-PassValue repeatability files = **48 entries total**; fails on any mismatch.

**SHUD_DUMP_FNAME_SUFFIX + SHUD_DUMP_SITE 启用方式**（任意需要二者并存的 dump run 通用模式）：

```bash
# After-PassValue (PR #71 / #55: rhs_apply 末端 DY 整合状态; filename 无 suffix):
SHUD_DUMP_SITE=f_applyDY                                 ./shud <proj>
# Before-PassValue (PR #43 钩子 + filename suffix):
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

**CI hookup scaffold 已就位（PR #54 + #57 dynamic-START parse），但 data-conditional silent skip**：`.github/workflows/serial-baseline.yml` 通过 cfg.para 动态 awk 解析 START（PR #57 / commit `57505ad` 起替代原硬编 abs-min triplet），公式 `abs_min = START_day × 1440 + {1440, 43200, 129600}`（相当于 case-rel +1d / +30d / +90d）。snapshot HARD-fail gate + bitwise compare vs B0-tag golden 步骤逻辑完整，但因 forcing data 不在 CI runner（per workflow 顶部 "SCOPE" + docs/s1_summary.md），`data_probe.outputs.data_available` 永远为 `false` → 所有 30+ run+compare step silent skip with notice。**CI 实际 gate 不含 bitwise**——仅 build + 4 macro grep + invariant-sweep + B0-tag smoke + skip-label deletion verify。bitwise / CVODE 15-key / snapshot 90d gate 全部依赖本地 + 服务器人工验证（4-case × 90-day 在 S1 已 16+ 次跑过 PASS，详见 docs/s1_summary.md "下一步：S2" §3）。post-PR #74-CI-followup 简化（删 nightly cron + full-bitwise label）后 matrix 固定 keliya × LEGACY_RHS={0,1} = 2 jobs，scaffold 步骤保留作 future forcing-data deployment 的就绪态。

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
