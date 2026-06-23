# P1c — summary (seed; PR-K capstone fills full ≥7-topic structure)

> **Status**: seed file (PR-F #249)。PR-K capstone 将填入 (1)–(7) 全套主题。本文件先 seed §"capstone 验证结果" 中两个可在 PR-E 后端立即填充的子节 (§2.6 三 negative grep gates + §2.7 10-anchor coverage)。

## §"capstone 验证结果" 占位 (PR-F, post-PR-E pin de9545d)

### §2.6 三 negative grep gates (per spec Scenarios L76 / L81 / L86)

| Gate | Command | Expected | Actual | Verdict |
|---|---|---|---|---|
| (a) 新宏 | `grep -rE 'SHUD_USE_DETERMINISTIC_REDUCTION\|SHUD_DET_REDUCT\|SHUD_PAIRWISE' SHUD/` | 0 hits | 0 hits | **PASS** |
| (b) schedule | `grep -nE 'schedule\(' SHUD/src/Model/MD_rhs_core.cpp` | 0 hits OR static-only | 0 hits | **PASS** |
| (c) atomic | `grep -rn '#pragma omp atomic' SHUD/src/` | 0 hits | 0 hits | **PASS** |

三 gate 全 PASS — PR-A..PR-E 未引入 spec 禁止的宏 / `dynamic`-`guided` schedule / `#pragma omp atomic`，per spec §"全 RHS reduction 站点 grep 清单完整覆盖" + §"MD_rhs_core.cpp 8 reduction 站点 fixed-shape pairwise 改造" §禁止子句。

### §2.7 10 line anchors → 8 logical sites coverage (per spec L63-66 Scenario)

PR-A..PR-E 已将 10 个 write-target line anchors 全部包裹在 `fixed_pairwise_sum_indexed` / `fixed_leftfold_sum_indexed` / `fixed_leftfold_sum_pair_indexed` helper 调用中 (`SHUD/src/Model/MD_rhs_core.cpp`)。post-PR-E 行号见下表：

| # | 写目标变量 | PR | helper 调用 line (post-PR-E, SHUD@de9545d) | docs/p1c_reduction_sites.md row | Helper |
|---|---|---|---|---|---|
| 1 | `qLakeEvap` | PR-B | L346 | row 1 (L278 → 346) | `fixed_pairwise_sum_indexed` |
| 2 | `qLakePrcp` | PR-B | L348 | row 2 (L279 → 348) | `fixed_pairwise_sum_indexed` |
| 3 | `QrivSurf` | PR-C | L447 | row 3 (L374 → 447) | `fixed_leftfold_sum_indexed` |
| 4 | `QrivSub` | PR-C | L448 | row 4 (L375 → 448) | `fixed_leftfold_sum_indexed` |
| 5 | `Qe2r_Surf` | PR-C | L457 | row 5 (L382 → 457) | `-fixed_leftfold_sum_indexed` |
| 6 | `Qe2r_Sub` | PR-C | L458 | row 6 (L383 → 458) | `-fixed_leftfold_sum_indexed` |
| 7 | `QrivUp` | PR-D | L468 | row 7 (L392 → 468) | `-fixed_leftfold_sum_indexed` |
| 8a | `QLakeRivIn` | PR-E | L480 | row 8 (L406 → 480) | `fixed_leftfold_sum_indexed` |
| 8b | `QLakeSurf` | PR-E | L491 | row 8 (L420 → 491) | `fixed_leftfold_sum_pair_indexed` |
| 8c | `QLakeSub` | PR-E | L501 | row 8 (L433 → 501) | `fixed_leftfold_sum_pair_indexed` |

10 write targets → 8 logical sites (合并 8a/8b/8c 为单一 logical site "lake gathers 组"，per spec Conventions §"站点计数定义")。Coverage complete。

#### 验证 grep 复现

```bash
# 1) helper call sites in MD_rhs_core.cpp (匹配 10 anchors)
grep -nE 'fixed_(pairwise|leftfold)_sum' SHUD/src/Model/MD_rhs_core.cpp \
  | grep -vE '^[0-9]+:static inline' \
  | head -20

# 2) write-target name × line cross-check
grep -nE '(qLakeEvap|qLakePrcp|QrivSurf|QrivSub|Qe2r_Surf|Qe2r_Sub|QrivUp|QLakeRivIn|QLakeSurf|QLakeSub)' \
  SHUD/src/Model/MD_rhs_core.cpp | head -30
```

Cross-walk with [`docs/p1c_reduction_sites.md`](p1c_reduction_sites.md) §"类 1 — 已覆盖" + §"Line-number 等价表" — 写目标变量列逐项对齐 (行号已漂移到 helper-call 形式，但函数名 / 变量名一致)。

#### §"## §Capstone deeper structure (placeholder for PR-K)" — 后续 PR-K 填写

PR-K 将填入 capstone summary 完整 ≥7 主题结构 (per spec p1c-capstone 章节):

- (1) 完成定义
- (2) 旧版错误复盘 [P1c 为新阶段, 无独立旧版, N/A]
- (3) P1c-tag 处理
- (4) 时间线
- (5) hand-off → P2a
- (6) capstone 验证结果 (this section 已 seed §2.6 / §2.7)
- (7) P1c-tag 验证命令
