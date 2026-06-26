# P1e — toolchain investigation (PLACEHOLDER, NOT TRIGGERED)

> **Status (PR-K 2026-06-25)**: **not triggered** (mode A Phase 1 同 build 同 N × 3 reps + 跨 N 全 bitwise PASS, per `docs/p1e/p1e_pr_c_2x2_mac.md` §3.1 + `docs/p1e/p1e_pr_d_2x2_server.md` §3.1)
>
> This file is a **conditional placeholder** per spec `p1e-capstone` Requirement "docs/p1e/ ≥14 doc 必备" Scenario + tasks §11.A "toolchain investigation (mode A Phase 1 FAIL 触发)"。当 P1e epic 内 mode A Phase 1 实际未 FAIL 时，本 doc 仅做存在性占位以满足 capstone spec doc-count Requirement，不含 toolchain investigation 实施数据。

## §1 触发条件

per `openspec/changes/p1e-strict-omp-rhs/tasks.md` §11.A：

> 若 task 2.5 或 2.6 mode A 同 build 同 N × 3 reps 或跨 N 不 bitwise → SHALL 在 `docs/p1e/p1e_toolchain_investigation.md` 记录：
> - (a) 失败 cell 详情 (case / N / rep / SHA delta)
> - (b) 编译器 + libomp/libgomp + glibc + SUNDIALS pin 完整 version 报表
> - (c) bisect 历史 (vs P1d-tag mode A reference SHA 何时开始漂)
> - (d) 调查结论 + 恢复条件 (toolchain pin 或调整 → mode A 跨 reps 重恢复)

per `openspec/changes/p1e-strict-omp-rhs/tasks.md` §11 OQ1：

> **OQ1** (mode A 同 build 同 N reps bitwise): PR-C 实施前守门，若 NO → P1e 暂停查 toolchain → 走 task 11.A 调查流程

即只在 mode A Phase 1 reps 或跨 N FAIL 时触发本文档。

## §2 Phase 1 实测结果 (per `docs/p1e/p1e_pr_c_2x2_mac.md` §3.1 + `docs/p1e/p1e_pr_d_2x2_server.md` §3.1)

| platform | case | mode A 12-cell (4 N × 3 reps) unique SHA count | verdict |
|---|---|---:|:---:|
| Mac | keliya | 1 | PASS |
| Mac | xinanjiang_upstream | 1 | PASS |
| Mac | qinyijiang | 1 | PASS |
| Mac | qhh | 1 | PASS |
| server | heihe | 1 | PASS |
| server | heihe_x4 | 1 | PASS |

**全 6 case × 12 cell = 72 cell mode A bitwise PASS** → OQ1 mode A 守门通过 → toolchain investigation **不触发** → 本 placeholder 留作 conditional doc 占位。

## §3 toolchain 当前 baseline (reference only)

per `docs/p1e/p1e_perf_baseline.md` §1：

| 端 | toolchain |
|---|---|
| Mac | Apple Clang 17.0.0 + libomp 17.x + Apple Darwin 24.6.0 |
| server | GCC 13.3.0 + libgomp + Linux Ubuntu 24.04 + glibc (Ubuntu default) |
| SUNDIALS | 6.0.0 (pinned, P1d → P1e era unchanged) |

mode A 全 PASS 表明上述 toolchain 在 SHUD `f.cpp::54` `MD->rhs_core(..., ExecPolicy::Serial)` Serial 路径上 deterministic-by-construction，无任何 bitwise drift。

## §4 forward considerations (forthcoming)

若 future epic (P2/P3) 发现 mode A 跨 N 或跨 reps 出现 bitwise drift（例如升级 SUNDIALS 或 GCC 后）：

- 升级前先做 mode A Phase 1 守门 (per OQ1 模式)
- 若 FAIL 触发本 doc 升级为实际 toolchain investigation 报告，按 §1 (a-d) 4 项 enumerate
- bisect target: vs P1e-tag mode A reference SHA (P1d era reference 仍可作 cross-epic 比对锚点)

## §5 References

- spec `p1e-strict-omp-rhs` (OQ1 section)
- tasks `p1e-strict-omp-rhs` §11 OQ1 + §11.A toolchain investigation
- spec `p1e-capstone` Requirement "docs/p1e/ ≥14 doc 必备" Scenario (本 placeholder 创建 SHALL)
- `docs/p1e/p1e_pr_c_2x2_mac.md` §3.1 (Mac mode A AC1 PASS)
- `docs/p1e/p1e_pr_d_2x2_server.md` §3.1 (server mode A AC1 PASS)
