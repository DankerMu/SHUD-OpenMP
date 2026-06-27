---
title: "Case deployment canonical map (双端 source-of-truth)"
date: 2026-06-26
version: 1.0 (initial — derived from 2026-06-26 双端实测 P2a profile session)
status: "active — 所有 profile/baseline/SHALL gate 工作前必读"
related_docs:
  - "CLAUDE.md §双端实验环境 L40-L46 (条款式约束)"
  - "docs/p2a/p2a_profile_baseline.md (v0.4 profile data 引用本表 canonical)"
  - "tools/forcing_trim/README.md (M7 forcing trim 工具)"
---

# Case deployment canonical map

本表是 Mac 本地 + server 双端 case basin 部署状态的 **唯一权威**。CLAUDE.md L40-L46 是条款式约束，**真实部署状态以本表为准**。

## §1 命名 convention (易踩坑)

**basin folder name ≠ SHUD project name** — 项目有 implicit 映射，CLAUDE.md L41 仅列了 basin folder：

| Basin folder (`SHUD/Basins/<>`) | SHUD project name (`./shud <name>` 入参) | cfg.para 位置 |
|---|---|---|
| keliya | `keliya` | `input/keliya/keliya.cfg.para` |
| qhh | `qhh` | `input/qhh/qhh.cfg.para` |
| heihe | `heihe` | `input/heihe/heihe.cfg.para` |
| heihe_x4 | `heihe_x4` | `input/heihe_x4/heihe_x4.cfg.para` |
| **kashigeer** | **`ksge`** | `input/ksge/ksge.cfg.para` |
| **qinyijiang** | **`nanlin`** | `input/nanlin/nanlin.cfg.para` |
| **tailanhe** | **`tlh`** | `input/tlh/tlh.cfg.para` |
| **xinanjiang_upstream** | **`xinanjiang`** | `input/xinanjiang/xinanjiang.cfg.para` |

**Pitfall**: 用 `find SHUD/Basins/<basin>/input/<basin>/` 误报 MISSING，应该 `find SHUD/Basins/<basin>/input/ -name '*.cfg.para'` 自适应发现。

## §2 双端 case basin 实测部署状态 (2026-06-26 21:00 UTC+8)

### §2.1 Mac (`/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/`)

| Basin | Project | NumEle | cfg START/END | window | forcing/ | forcing.trimmed/ | 可跑? |
|---|---|---:|---|---|---|---|---|
| keliya | keliya | 484 | 12053 / 12143 | **90 d ✓** | local 228M / 33 csv | 1.1M / 33 csv | ✅ |
| qhh | qhh | 4773 (+lake) | 8401 / 8491 | **90 d ✓** | local 1.4G / 386 csv | 11M / 386 csv | ✅ |
| qinyijiang | nanlin | 3155 | 366 / 456 | **90 d ✓** | local 278M / 94 csv | 2.5M / 94 csv | ✅ |
| xinanjiang_upstream | xinanjiang | 801 | 0 / 90 | **90 d ✓** | local 58M / 51 csv | 1.5M / 51 csv | ✅ |
| kashigeer | ksge | 3204 | 20088 / 22279 | **2191 d** ⚠️ | local 2.7G / 402 csv | — | ⚠️ 需 90d truncate |
| tailanhe | tlh | 1614 | 0 / 27028 | **74 yr** ⚠️ | **no forcing/** | — | ❌ broken (tsd.forc path typo `focing`, 路径不存在) |
| heihe_x4 | heihe_x4 | 40046 | 1 / 1095 | **3 yr** ⚠️ | **no forcing/** | — | ❌ server artifact rsync 漏 forcing |

**Mac 4 case profile canonical 集** (P2a v0.4 用): keliya / xinanjiang_upstream / qinyijiang / qhh

### §2.2 Server (`/scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/`)

| Basin | Project | NumEle | cfg START/END | window | forcing/ | forcing.trimmed/ | 可跑? |
|---|---|---:|---|---|---|---|---|
| **heihe** | heihe | 6335 | 14245 / 14335 | **90 d ✓** | SYMLINK → /volume/data/nwm/Basins/heihe/forcing | **29 M / 1710 csv** ✓ | ✅ canonical baseline |
| **heihe_x4** | heihe_x4 | 40046 | 1 / 91 | **90 d ✓** | local 286M / 1694 csv | 26 M / 1694 csv | ✅ canonical baseline |
| keliya | keliya | 484 | 12053 / 12143 | **90 d ✓** | local 140M / 33 csv | — | ⚠️ 缺 trimmed |
| qhh | qhh | 4773 | 8401 / 8491 | **90 d ✓** | SYMLINK → /volume/data/nwm/Basins/qhh/forcing | — | ⚠️ 缺 trimmed |
| qinyijiang | nanlin | 3155 | 366 / 456 | **90 d ✓** | SYMLINK → /volume/data/nwm/Basins/qinyijiang/forcing | — | ⚠️ 缺 trimmed |
| xinanjiang_upstream | xinanjiang | 801 | 0 / 90 | **90 d ✓** | SYMLINK → /volume/data/nwm/Basins/xinanjiang_upstream/forcing | — | ⚠️ 缺 trimmed |

**Server canonical baseline 集**: heihe (forcing.trimmed canonical) + heihe_x4 (forcing/ canonical, 3yr 已是 90d 邻近 subset)

## §3 SHUD pin (双端一致)

| 端 | SHUD pin | branch | 状态 |
|---|---|---|---|
| Mac (outer submodule pointer) | `7a1dc8f` | openmp-baseline | ✓ |
| Server (working copy) | `7a1dc8f` | openmp-baseline | ✓ |

`7a1dc8f` = P2a profile fix (nested-Timer removal in MD_ET.cpp + Model_Control.cpp)。建在 P1e `3341368` 之上。

## §4 forcing 部署模式 (易混淆)

### §4.1 三种部署模式

1. **`forcing/` local** (basin-local NVMe csv): heihe_x4 server, Mac 全部, server keliya
2. **`forcing/` SYMLINK → /volume/...** (NFS read): server heihe / qhh / qinyijiang / xinanjiang_upstream
3. **`forcing.trimmed/` local** (forcing_trim 90d 子集): server heihe / heihe_x4, Mac keliya / qhh / qinyijiang / xinanjiang_upstream

### §4.2 SHUD 实际读 forcing 的路径

由 `<project>.tsd.forc` 第 2 行 (absolute path) 决定，**不是** basin/forcing/ 直接命中。

- Server heihe: 第 2 行 → `/scratch/.../SHUD/Basins/heihe/forcing.trimmed` (canonical fair-compare, P2a v0.4 用)
- Server heihe_x4: 第 2 行 → `/scratch/.../SHUD/Basins/heihe_x4/forcing` (basin-local 286M, 3yr 自带, 已 90d 邻近)
- Mac (全部): 第 2 行 → `/Users/.../SHUD/Basins/<case>/forcing` (basin-local，**未切换 trimmed**)

### §4.3 dataset 时段长度差异 (P2a outlier root cause)

| Basin | forcing dataset 时段 | dataset 大小 | 单 csv 大小 |
|---|---|---:|---:|
| NWM canonical (`/volume/data/nwm/Basins/<case>/forcing/`) | CMFD V0200 全 74yr (1951-2024) | 7-12 GB | 4-7 MB |
| heihe_x4 basin-local | AutoSHUD 派生 3yr 子集 | 286 MB | 170 KB |
| heihe forcing.trimmed (M7 trim) | 90-day + 2d buffer = 94d | 29 MB | 17 KB |

**SHUD updateforcing 每 inner step 在 csv 中线性扫描当前时间 row → IO wall ∝ csv 大小**。同 case (heihe v0.2 vs v0.4) 缩 413× dataset → forcing wall 缩 24× → 总 wall 缩 75%。

## §5 已跑 N=1 profile (canonical SHIP set)

| Case | Platform | SHUD pin | wall (s) | yaml path | 状态 |
|---|---|---|---:|---|---|
| keliya | Mac | `7cc46d8` (pre-fix) | 30.23 | `/tmp/p2a_profile_mac/keliya_N1.yaml` | v0.1 含 ×2 double-count |
| xinanjiang_upstream | Mac | `7cc46d8` | 4.73 | `/tmp/.../xinanjiang_upstream_N1.yaml` | v0.1 |
| qinyijiang | Mac | `7cc46d8` | 285.61 | `/tmp/.../qinyijiang_N1.yaml` | v0.1 |
| qhh | Mac | `7cc46d8` | 97.30 | `/tmp/.../qhh_N1.yaml` | v0.1 |
| qhh (re-run) | Mac | `7a1dc8f` (fixed) | 97.19 | `/tmp/.../qhh_N1.yaml` (overwrite) | v0.2 fair |
| heihe (v0.2) | Server cn08 | `7a1dc8f` | 523.05 | `/tmp/p2a_profile_server/heihe_N1.yaml` | **REJECTED** dataset size artifact |
| heihe (v0.4) | Server cn08 | `7a1dc8f` | 134.87 | `/tmp/.../heihe_v4_N1.yaml` | ✓ canonical fair-compare |
| heihe_x4 | Server cn08 | `7a1dc8f` | 1373.23 | `/tmp/.../heihe_x4_N1.yaml` | ✓ canonical |

### §5.1 p8pre-spike Step 1 PR-A — N=8 Mode C profile baseline (18-cell SHIP set, 2026-06-27)

p8pre-spike Step 1 PR-A 18-cell 矩阵 (2 case × 3 N × 3 rep), Mode C profile build (`SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`), SHUD pin `7a1dc8f`, server cn14 (heihe) + cn15 (heihe_x4). 详 [`docs/p8pre/n8_profile_baseline.md`](p8pre/n8_profile_baseline.md) §5.1 (gate-4 anchor for PR-F #347)。Wall = profile_B0.yaml `extras.t_wall_total`。状态：✓ canonical (branch a PROCEED Step 2)。

| Case | N | rep | SHUD pin | wall (s) | yaml path |
|---|---:|---:|---|---:|---|
| heihe | 1 | 1 | `7a1dc8f` | 141.790 | `/tmp/p8pre_n8_profile/heihe_N1_rep1/profile_B0.yaml` |
| heihe | 1 | 2 | `7a1dc8f` | 140.797 | `/tmp/p8pre_n8_profile/heihe_N1_rep2/profile_B0.yaml` |
| heihe | 1 | 3 | `7a1dc8f` | 125.216 | `/tmp/p8pre_n8_profile/heihe_N1_rep3/profile_B0.yaml` |
| heihe | 4 | 1 | `7a1dc8f` | 96.509 | `/tmp/p8pre_n8_profile/heihe_N4_rep1/profile_B0.yaml` |
| heihe | 4 | 2 | `7a1dc8f` | 95.734 | `/tmp/p8pre_n8_profile/heihe_N4_rep2/profile_B0.yaml` |
| heihe | 4 | 3 | `7a1dc8f` | 95.068 | `/tmp/p8pre_n8_profile/heihe_N4_rep3/profile_B0.yaml` |
| heihe | 8 | 1 | `7a1dc8f` | 89.732 | `/tmp/p8pre_n8_profile/heihe_N8_rep1/profile_B0.yaml` |
| heihe | 8 | 2 | `7a1dc8f` | 89.744 | `/tmp/p8pre_n8_profile/heihe_N8_rep2/profile_B0.yaml` |
| heihe | 8 | 3 | `7a1dc8f` | 89.324 | `/tmp/p8pre_n8_profile/heihe_N8_rep3/profile_B0.yaml` |
| heihe_x4 | 1 | 1 | `7a1dc8f` | 1474.953 | `/tmp/p8pre_n8_profile/heihe_x4_N1_rep1/profile_B0.yaml` |
| heihe_x4 | 1 | 2 | `7a1dc8f` | 1412.895 | `/tmp/p8pre_n8_profile/heihe_x4_N1_rep2/profile_B0.yaml` |
| heihe_x4 | 1 | 3 | `7a1dc8f` | 1268.620 | `/tmp/p8pre_n8_profile/heihe_x4_N1_rep3/profile_B0.yaml` |
| heihe_x4 | 4 | 1 | `7a1dc8f` | 849.704 | `/tmp/p8pre_n8_profile/heihe_x4_N4_rep1/profile_B0.yaml` |
| heihe_x4 | 4 | 2 | `7a1dc8f` | 849.756 | `/tmp/p8pre_n8_profile/heihe_x4_N4_rep2/profile_B0.yaml` |
| heihe_x4 | 4 | 3 | `7a1dc8f` | 840.095 | `/tmp/p8pre_n8_profile/heihe_x4_N4_rep3/profile_B0.yaml` |
| heihe_x4 | 8 | 1 | `7a1dc8f` | 743.556 | `/tmp/p8pre_n8_profile/heihe_x4_N8_rep1/profile_B0.yaml` |
| heihe_x4 | 8 | 2 | `7a1dc8f` | 743.552 | `/tmp/p8pre_n8_profile/heihe_x4_N8_rep2/profile_B0.yaml` |
| heihe_x4 | 8 | 3 | `7a1dc8f` | 742.919 | `/tmp/p8pre_n8_profile/heihe_x4_N8_rep3/profile_B0.yaml` |

Server source mirror: `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/<case>_N<n>_rep<r>/`。`wall_step1_baseline_median(case, N)` (gate-4 anchor) = [`docs/p8pre/n8_profile_baseline.md`](p8pre/n8_profile_baseline.md) §5.1 Table 1。

### §5.2 p8pre-spike Step 2 PR-E — identity-precond 18-cell spike (NO-GO data archive, 2026-06-27)

p8pre-spike Step 2 PR-E 18-cell 矩阵 (2 case × 3 N × 3 rep), Mode C + PREC_LEFT identity stub build (`make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`), SHUD pin `5276167` (PR-D #357 forward-only descendant 含 `MD_precond_identity.{h,cpp}` + `cvode_config.cpp:259` PREC_NONE→PREC_LEFT + `CVodeSetPreconditioner` + `CVodeSetLSetupFrequency(50)`), server cn14 (heihe) + cn15 (heihe_x4)。详 [`docs/p8pre/identity_spike_run.md`](p8pre/identity_spike_run.md) §4 + [`docs/p8pre/identity_spike_verdict.md`](p8pre/identity_spike_verdict.md) §3。Wall = `profile_B0.yaml extras.t_wall_total`; t_precond_setup = `profile_B0.yaml extras.t_precond_setup` (new PR-D §6.2 Timer bucket)。

**状态**: NO-GO 数据 archive (per [`docs/adr/0003-precond-spike-decision.md`](adr/0003-precond-spike-decision.md), gate 2 ncfn FAIL + gate 5 max_ulp ≈9×10¹⁵ FAIL)。`baseline/p8pre` 分支 + identity code revert 推 #349 archive 或 separate cleanup PR。

| Case | N | rep | SHUD pin | wall (s) | yaml path |
|---|---:|---:|---|---:|---|
| heihe | 1 | 1 | `5276167` | 137.079 | `/tmp/p8pre_identity_spike/heihe_N1_rep1/profile_B0.yaml` |
| heihe | 1 | 2 | `5276167` | 137.274 | `/tmp/p8pre_identity_spike/heihe_N1_rep2/profile_B0.yaml` |
| heihe | 1 | 3 | `5276167` | 121.619 | `/tmp/p8pre_identity_spike/heihe_N1_rep3/profile_B0.yaml` |
| heihe | 4 | 1 | `5276167` | 94.049 | `/tmp/p8pre_identity_spike/heihe_N4_rep1/profile_B0.yaml` |
| heihe | 4 | 2 | `5276167` | 93.846 | `/tmp/p8pre_identity_spike/heihe_N4_rep2/profile_B0.yaml` |
| heihe | 4 | 3 | `5276167` | 93.499 | `/tmp/p8pre_identity_spike/heihe_N4_rep3/profile_B0.yaml` |
| heihe | 8 | 1 | `5276167` | 88.216 | `/tmp/p8pre_identity_spike/heihe_N8_rep1/profile_B0.yaml` |
| heihe | 8 | 2 | `5276167` | 87.621 | `/tmp/p8pre_identity_spike/heihe_N8_rep2/profile_B0.yaml` |
| heihe | 8 | 3 | `5276167` | 88.038 | `/tmp/p8pre_identity_spike/heihe_N8_rep3/profile_B0.yaml` |
| heihe_x4 | 1 | 1 | `5276167` | 1491.050 | `/tmp/p8pre_identity_spike/heihe_x4_N1_rep1/profile_B0.yaml` |
| heihe_x4 | 1 | 2 | `5276167` | 1428.287 | `/tmp/p8pre_identity_spike/heihe_x4_N1_rep2/profile_B0.yaml` |
| heihe_x4 | 1 | 3 | `5276167` | 1273.800 | `/tmp/p8pre_identity_spike/heihe_x4_N1_rep3/profile_B0.yaml` |
| heihe_x4 | 4 | 1 | `5276167` | 858.281 | `/tmp/p8pre_identity_spike/heihe_x4_N4_rep1/profile_B0.yaml` |
| heihe_x4 | 4 | 2 | `5276167` | 858.297 | `/tmp/p8pre_identity_spike/heihe_x4_N4_rep2/profile_B0.yaml` |
| heihe_x4 | 4 | 3 | `5276167` | 847.850 | `/tmp/p8pre_identity_spike/heihe_x4_N4_rep3/profile_B0.yaml` |
| heihe_x4 | 8 | 1 | `5276167` | 748.337 | `/tmp/p8pre_identity_spike/heihe_x4_N8_rep1/profile_B0.yaml` |
| heihe_x4 | 8 | 2 | `5276167` | 747.720 | `/tmp/p8pre_identity_spike/heihe_x4_N8_rep2/profile_B0.yaml` |
| heihe_x4 | 8 | 3 | `5276167` | 749.529 | `/tmp/p8pre_identity_spike/heihe_x4_N8_rep3/profile_B0.yaml` |

Server source mirror: `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/identity_spike/<cell>/`。Slurm JID 9531-9548 全 ExitCode 0; aggregator verdict: gate 2 ncfn deterministic FAIL (heihe ncfn=6, heihe_x4 ncfn=47 跨 18/18); gate 5 max_ulp ≈9×10¹⁵ ≫ A4 1024 阈值 (5,155/214,252 positions structurally diverge)。

## §6 已知缺陷 / 待补 (carve-out)

1. **Mac heihe_x4 broken** (no forcing/, cfg.para 3yr 不 truncate) — 不能 Mac 跑。fix: 从 server rsync `SHUD/Basins/heihe_x4/forcing/` (286 MB) + cfg.para truncate
2. **Mac tailanhe broken** (tsd.forc typo `focing` + no forcing/ + 74yr 全量) — 备用 case 不入 benchmark，低优先级
3. **Mac kashigeer 6yr cfg.para** — 需 90-day truncate
4. **Server keliya / qhh / qinyijiang / xinanjiang_upstream 缺 forcing.trimmed/** — 若做 fair-compare server profile 需补 `tools/forcing_trim/`
5. **Mac 3 case (keliya / xinanjiang / qinyijiang) profile 仍 v0.1 double-count** — 若 P2b/P5 epic 需要精确占比可单独 re-run with `7a1dc8f`

## §7 部署 SOP (forcing_trim 推荐默认)

未来部署 server case baseline 默认走 `tools/forcing_trim/forcing_trim.sh`:

```bash
# 通用模板
SSH="ssh -p 32099 frd_muziyao@210.77.77.22"
$SSH "
cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD/Basins/<case>
# 确保 forcing/ symlink 或 local 存在
# 跑 forcing_trim (输出 forcing.trimmed/)
start=\$(awk '/^[[:space:]]*START/ {print \$2}' input/<project>/<project>.cfg.para)
bash /scratch/.../tools/forcing_trim/forcing_trim.sh <case> \$start \$((start + 90))
# 切 tsd.forc 第 2 行到 forcing.trimmed/
TRIMMED=/scratch/.../SHUD/Basins/<case>/forcing.trimmed
sed -i \"2c\$TRIMMED\" input/<project>/<project>.tsd.forc
"
```

**bitwise-equivalent** per M7 spec (`tools/forcing_trim/verify_trim_bitwise.sh <case>`) — trimmed 与 full forcing 在 90d window 内 SHA256 一致。

## §8 next step (本 doc 后续)

- 加补 case 时更新本表
- forcing.trimmed/ 补齐到 server 其它 4 case 后更新 §2.2
- Mac 缺陷 (§6.1-3) 修复后更新 §2.1
- profile 跑新 case 后更新 §5 SHIP set

---
Generated: 2026-06-26 by orchestrator (P2a profile fair-compare 双端实测后梳理)
