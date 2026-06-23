## Purpose

规约 M7 forcing trim 工具实现、6 case (kashigeer N/A) trim 完成 + bitwise vs B0-tag 验证、heihe(_x4) forcing init wall 与占比目标、case manifest `forcing_dir` schema BREAKING 升级（scalar → mapping `{original_path, trimmed_path}`）、`tools/check_manifest.py` 校验同步升级、CI matrix `forcing_trimmed=1` 轴与 forcing_present 条件 SKIP。本 capability 是 P1 epic 的 Phase A（PR-A #212 + PR-B #213）。

## Conventions

- 章节顺序锚定 Purpose / Conventions / Requirements。
- Requirement 标题严格匹配 B1a-precedent 模板（### Requirement: …），Scenario 用 #### Scenario: 标识。
- 本 spec 由 openspec/changes/p1-update-omp/specs/<capability>/spec.md PROMOTE 而来（#226 P1 capstone 2026-06-22），原始 change spec 的 "## ADDED Requirements" 头部已替换为 system-spec 等价的 Purpose+Conventions+Requirements 三段结构。
- 工具实现严守"无 Python 依赖"铁律：纯 bash + awk + standard unix utils；CI runner 不引入额外 deps。
- forcing_dir schema 升级为 BREAKING change：legacy scalar-string 在 PR-A merge 之后不再容忍；kashigeer `trimmed_path: null` 是 deferred-upstream 唯一豁免位。
- 所有 case 90 天截断验收（per CLAUDE.md project-level 铁律）；forcing 数据 V0200 only（V0106 已淘汰）。

## Requirements

### Requirement: M7 forcing trim 工具实现

`tools/forcing_trim/forcing_trim.sh` SHALL 是一个 bash + awk 实现的 CLI 工具，入参 `<case-name> <start_day> <end_day>`，遍历 `SHUD/Basins/<case>/forcing/*.csv`，skip 时间戳在 `[start_day - buffer, end_day + buffer]` 窗口外的行（buffer 默认 2 天，可由 `--buffer-days N` 旗标覆盖；详 buffer 规范 Requirement），输出到 `SHUD/Basins/<case>/forcing.trimmed/<station>.csv`（与输入 station 命名一致，仅行数缩短）。工具 SHALL 不依赖 Python / R / NumPy / pandas，纯 bash + awk + standard unix utils。工具 SHALL 提供 `--dry-run` 模式仅打印计划，不写文件。工具 SHALL 提供 `tools/forcing_trim/README.md` 文档。

#### Scenario: 工具入参三参数 PASS

- **WHEN** 执行 `tools/forcing_trim/forcing_trim.sh keliya 12053 12143`（start_day=12053, end_day=12143，对应 keliya 90 天截断；默认 buffer=2 天即左右各 2 天，总 4 天）
- **THEN** 输出目录 `SHUD/Basins/keliya/forcing.trimmed/` 创建，包含与 `SHUD/Basins/keliya/forcing/` 同数量的 `<station>.csv` 文件，每个文件行数 ≈ 752 (94 天 × 8 个 3h 步/天 = 90 天 + 左右各 2 天 buffer × 8 步) + header；exit code = 0

#### Scenario: dry-run 模式

- **WHEN** 执行 `tools/forcing_trim/forcing_trim.sh keliya 12053 12143 --dry-run`
- **THEN** stdout 打印计划（每文件计划保留行数 / 删除行数），不创建 `forcing.trimmed/` 目录；exit code = 0

#### Scenario: 无 Python 依赖

- **WHEN** 在屏蔽 `python python3 pip uv` 的最小环境执行 `PATH=/usr/bin:/bin env -i bash tools/forcing_trim/forcing_trim.sh keliya 12053 12143 --dry-run`
- **THEN** 执行成功（exit 0）；`which python python3 pip uv` 全部 not found 不阻断；`grep -E 'python|pip|uv' tools/forcing_trim/forcing_trim.sh tools/forcing_trim/verify_trim_bitwise.sh` 0 hit（工具源 grep 静态证据）

---

### Requirement: buffer 默认 2 天 + bitwise 触发可扩展 + clamp

工具默认 buffer SHALL = 2 天（即左右各 2 天，总 4 天）；若 bitwise gate（详 "trimmed forcing bitwise vs B0-tag 同字节" Requirement）失败，工具 SHALL 支持 `--buffer-days N` 旗标（N 整数，单位天）覆盖默认；最终 buffer 选择 SHALL 记录在 `tools/forcing_trim/README.md` 含 bitwise 证据链。当 `start_day - buffer < 0`（CMFD epoch 1951-01-01 = day 0），工具 SHALL **clamp lower window edge 到 0** 并 emit `[BUFFER-CLAMP]` 警告到 stderr（不阻断 trim）；clamp 后的 buffer 不超过 raw buffer，bitwise gate SHALL 仍可通过。

#### Scenario: 默认 buffer 边界保留与丢弃

- **WHEN** 默认 buffer=2 天执行 `forcing_trim.sh keliya 12053 12143`
- **THEN** trimmed CSV 含时间戳 `start_day - 2 = 12051` 与 `end_day + 2 = 12145` 的行（窗内边界保留）；不含 `start_day - 3 = 12050` 与 `end_day + 3 = 12146` 的行（窗外严格丢弃）

#### Scenario: `--buffer-days` 旗标覆盖

- **WHEN** 执行 `forcing_trim.sh keliya 12053 12143 --buffer-days 5`
- **THEN** trimmed CSV 含 `start_day - 5 = 12048` 与 `end_day + 5 = 12148` 行；不含 `12047` 与 `12149` 行

#### Scenario: 负 lower bound clamp（xinanjiang_upstream）

- **WHEN** 执行 `forcing_trim.sh xinanjiang_upstream 0 90`（默认 buffer=2，lower bound = -2 < 0）
- **THEN** 工具 emit `[BUFFER-CLAMP]` 警告到 stderr；trimmed CSV 首条时间戳 = day 0（非 day -2）；bitwise gate vs B0-tag 仍 PASS（clamped buffer ≤ raw buffer）

---

### Requirement: 6 case trim 完成 (kashigeer N/A 单独列)

6 个 NWM benchmark case SHALL 必做 trim 完成；kashigeer 单独列 N/A (deferred-upstream)：

| Case | start_day | end_day | trim 端 |
|---|---|---|---|
| keliya | 12053 | 12143 | Mac 本地 |
| xinanjiang_upstream | 0 | 90 | Mac 本地 |
| qinyijiang | 366 | 456 | Mac 本地 |
| qhh | 8401 | 8491 | Mac 本地 |
| heihe | (per cfg.para) | + 90 | server cn0X |
| heihe_x4 | (per cfg.para) | + 90 | server cn0X |
| kashigeer | N/A (deferred-upstream) | N/A | 不做 |

start_day / end_day 严格按 case `cfg.para` 中 `START` / `END` 字段读取（day-index 制，1951-01-01 epoch）。

#### Scenario: 6 case trim 完成

- **WHEN** 6 case (keliya / xinanjiang_upstream / qinyijiang / qhh / heihe / heihe_x4) 全部跑完 `forcing_trim.sh`
- **THEN** 每 case `SHUD/Basins/<case>/forcing.trimmed/` 目录存在，包含 station 数 = 原 forcing 目录 station 数

#### Scenario: kashigeer 仍 deferred-upstream

- **WHEN** 检查 kashigeer trim 状态
- **THEN** 状态 = N/A (deferred-upstream)，与 status_matrix kashigeer 列 N/A 一致

#### Scenario: server case cfg.para 整数 day-index 校验

- **WHEN** server cn0X 跑 `forcing_trim.sh heihe <start_day> <end_day>` 或 `forcing_trim.sh heihe_x4 <start_day> <end_day>`
- **THEN** 工具 stdout 显式打印 resolved `[start_day, end_day]` 双整数；这两整数 SHALL = `cfg.para` 中 `START` 整数字段 + `START + 90`；CI / reviewer SHALL 对照 `benchmarks/<case>/manifest.yaml` 的 `cfg_para_start` / `cfg_para_end` 字段 exact match；不匹配则 PR-B 不可 merge

---

### Requirement: trimmed forcing bitwise vs B0-tag 同字节

每个 trim 完成的 case SHALL 用 trimmed forcing 在 90 天 cfg 截断下跑一次完整 SHUD，输出的 canonical summary SHA256 SHALL 与 B0-tag 同 case 同字节（即 trim 仅删窗外行，窗内数值未动；B0-tag 是用全量 forcing 跑出的同 90 天截断结果）。验证脚本 `tools/forcing_trim/verify_trim_bitwise.sh <case>` 实现该校验。

#### Scenario: 4 Mac case bitwise vs B0-tag PASS

- **WHEN** 在 Mac 本地，4 case (keliya / xinanjiang_upstream / qinyijiang / qhh) 用 trimmed forcing 跑 90 天截断
- **THEN** 每 case canonical summary SHA256 ≡ `benchmarks/<case>/B0_output/repeatability.txt sha256_run1`

#### Scenario: 2 server case bitwise vs B0-tag PASS

- **WHEN** 在 server cn0X (Slurm 三铁律 enforced)，heihe + heihe_x4 用 trimmed forcing 跑 90 天截断
- **THEN** 每 case `rivqdown.dat` SHA256 ≡ B0-tag golden (heihe `55abad28…` / heihe_x4 `f90601ef…`)

---

### Requirement: heihe / heihe_x4 trimmed forcing init wall < 10s

heihe + heihe_x4 case 用 trimmed forcing 90 天截断跑时，SHUD 启动期 forcing init 耗时（S5c 7-bucket timer 的 `t_forcing_io` bucket，含全部站 forcing CSV load）SHALL < 10s（heihe_x4 vs trim 前 780s，>75× 提速）。验证在 server cn0X (Slurm job log)。

#### Scenario: heihe forcing init 实测 <10s

- **WHEN** 在 cn0X Slurm 跑 heihe trimmed forcing 90 天 NUM_OPENMP=1 baseline run
- **THEN** stderr / stdout log 中 `[timer] t_forcing_io = X.Xs`，X.X < 10.0

#### Scenario: heihe_x4 forcing init 实测 <10s

- **WHEN** 在 cn0X Slurm 跑 heihe_x4 trimmed forcing 90 天 NUM_OPENMP=1 baseline run
- **THEN** stderr / stdout log 中 `[timer] t_forcing_io = X.Xs`，X.X < 10.0

---

### Requirement: heihe / heihe_x4 `t_forcing_io / t_total < 5%`

heihe + heihe_x4 trimmed forcing 90 天截断跑的 S5c 7-bucket timer 中 `t_forcing_io / t_total` SHALL **同时** < 5%（即 forcing init 不再是 wall 的主导部分，关闭 5%-50% 中间带未决空隙；详 `profile-retest-m7` spec L21-37 Opt-IO 决策 50% 阈值）。

#### Scenario: heihe t_forcing_io 占比 <5%

- **WHEN** heihe trimmed 90 天 NUM_OPENMP=1 run 完成
- **THEN** 7-bucket timer `t_forcing_io / t_total` < 0.05

#### Scenario: heihe_x4 t_forcing_io 占比 <5%

- **WHEN** heihe_x4 trimmed 90 天 NUM_OPENMP=1 run 完成
- **THEN** 7-bucket timer `t_forcing_io / t_total` < 0.05

---

### Requirement: case manifest `forcing_dir` schema BREAKING 升级

7 个 case `benchmarks/<case>/manifest.yaml` 的 `forcing_dir` 字段 SHALL 从单 scalar 字符串升级为 mapping `{original_path, trimmed_path}` 双子字段（**Schema BREAKING**：所有现有读 `forcing_dir` 为 string 的 CI / tool 入口同步升级；不保留 legacy string 兼容）。kashigeer 因 deferred-upstream，`trimmed_path` 字段 SHALL 设 `null` + 注释 "deferred-upstream"。所有 P-strict / P-prod 部署期 SHALL 改读 `forcing_dir.trimmed_path`。

#### Scenario: 6 case manifest schema 升级

- **WHEN** 检查 6 case manifest `forcing_dir` 字段
- **THEN** schema = `{original_path: "<path>", trimmed_path: "<path>"}`，两路径都可解析

#### Scenario: kashigeer trimmed_path null

- **WHEN** 检查 kashigeer manifest `forcing_dir.trimmed_path`
- **THEN** 值 = `null` + 注释 "deferred-upstream"

#### Scenario: schema 升级清单同步

- **WHEN** PR-A merge 后 grep `forcing_dir` 在仓库
- **THEN** 所有 hit 处（`.github/workflows/serial-baseline.yml` step "verify trimmed forcing path"、`tools/forcing_trim/`、`tools/check_manifest.py`、`tools/fix_case_paths/`）SHALL 改读 mapping `forcing_dir.original_path` 或 `forcing_dir.trimmed_path`，不保留 legacy scalar-string 读取路径

---

### Requirement: `tools/check_manifest.py` 兼容升级

`tools/check_manifest.py` SHALL 同步升级 `forcing_dir` 校验：从 `(str,)` 改为 union `(str, dict)`，并新增 dict-schema 校验 sub-routine：(a) 含 `original_path` 字段 (str 必填)；(b) `trimmed_path` 字段是 str 或 None（kashigeer N/A 允许）；(c) malformed dict（缺 `original_path`）SHALL raise validation error。配套 test fixtures SHALL 含 legacy str / new dict trimmed / kashigeer null / malformed 四种用例。

#### Scenario: legacy str 仍兼容（过渡期）

- **WHEN** 在 PR-A merge 前的 legacy manifest 上跑 `tools/check_manifest.py`
- **THEN** `forcing_dir: "<path>"` (str) PASS（向后兼容；过渡期容忍）

#### Scenario: new dict trimmed PASS

- **WHEN** 在 PR-A merge 后的新 manifest 上跑 `tools/check_manifest.py`
- **THEN** `forcing_dir: {original_path: "<p1>", trimmed_path: "<p2>"}` PASS（dict 模式）

#### Scenario: kashigeer trimmed null PASS

- **WHEN** 跑 `tools/check_manifest.py` 在 kashigeer manifest
- **THEN** `forcing_dir: {original_path: "<p>", trimmed_path: null}` PASS（null 允许）

#### Scenario: malformed dict 拒绝

- **WHEN** 跑 `tools/check_manifest.py` 在含 `forcing_dir: {trimmed_path: "<p>"}`（缺 `original_path`）的 manifest
- **THEN** validation error raise，exit code 非 0

---

### Requirement: CI matrix `forcing_trimmed=1` 轴新增 + forcing_present 条件 SKIP

`.github/workflows/serial-baseline.yml` matrix SHALL 新增 `forcing_trimmed` 轴，值集 = `[1]`（强制 trimmed），不保留 `0` 值（避免双倍 CI 时间）。新增 step "Verify trimmed forcing path" SHALL 在 run 前 **仅在 `forcing_present=true` 时执行**（与现有 L739-L752 + L1276-L1288 SKIP 路径对齐）：从 `SHUD/Basins/<case>/input/<project>/<project>.tsd.forc` **第 2 行**读取 forcing 目录绝对路径，与 `benchmarks/<case>/manifest.yaml :: forcing_dir.trimmed_path`（**awk extractor**, zero extra runner deps）做**绝对路径 exact match**；不匹配则 `::error file=benchmarks/<case>/manifest.yaml::tsd.forc line 2 (forcing pointer) != manifest forcing_dir.trimmed_path` 阻断；forcing 缺失则 `::notice` SKIP（不阻断），与现有 forcing-absent 路径行为一致。

#### Scenario: matrix 新增 forcing_trimmed=1 列

- **WHEN** 检查 `serial-baseline.yml` matrix 块
- **THEN** matrix 含 `forcing_trimmed: [1]` 轴

#### Scenario: cfg pointer verify step 存在（forcing_present=true 路径）

- **WHEN** 检查 workflow steps 且 `forcing_present=true`
- **THEN** 含 step "Verify trimmed forcing path"；从 `SHUD/Basins/<case>/input/<project>/<project>.tsd.forc` 第 2 行读 forcing 目录绝对路径，与 `benchmarks/<case>/manifest.yaml :: forcing_dir.trimmed_path`（awk extractor）做 exact match；不匹配则 `::error file=benchmarks/<case>/manifest.yaml::tsd.forc line 2 (forcing pointer) != manifest forcing_dir.trimmed_path` 阻断

#### Scenario: forcing 缺失 → verify step SKIP（不 ::error）

- **WHEN** workflow runner 不部署该 case forcing 数据（`forcing_present=false`）
- **THEN** "verify trimmed forcing path" step 输出 `::notice title=<case> data not deployed::verify-trimmed-forcing-path SKIPPED (forcing absent)`，不 `::error`，不阻断 build verification + grep gate 通过路径
