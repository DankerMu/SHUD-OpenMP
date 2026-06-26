# P1e — first-touch removal (PR-H)

P1e PR-H (#316) 在 `SHUD/src/Model/MD_rhs_core.cpp` 删除的 3 处 steady-state first-touch loops 记录 + before/after diff + allocation-time + load-time first-touch 保留 verify。本 doc 是 spec `p1e-strict-omp-rhs` Requirement "first-touch 治理" 的 capstone 引用。

## §1 删除范围

per PR-H commit `3341368d2d0854924d2286925c8575df52cc97a0` body：

> Per design D4: removed 3 inner steady-state first-touch loops at MD_rhs_core.cpp L62-95 (element), L169-203 (lake), L324-354 (river) (formerly P1d.2.{1,2,3}, gated by g_numa_first_touch_enabled). Under the StrictOMP outer parallel region introduced in PR-F, those inner `#pragma omp parallel for` directives would create nested parallel regions; the steady-state warm-up is also redundant once the outer team distributes the same work via owner-local omp-for shares.

| 删除 site | 原 line range | 原 owner block | P1d.2 子任务 |
|---|---|---|---|
| 1 | `SHUD/src/Model/MD_rhs_core.cpp` L62-95 | element bucket (`QeleSurfAt / QeleSubAt / QeleSurfTot / QeleSubTot / Qe2r_Surf / Qe2r_Sub`) | P1d.2.1 (PR-C #277) |
| 2 | `SHUD/src/Model/MD_rhs_core.cpp` L169-203 | lake bucket | P1d.2.2 (PR-D #278) |
| 3 | `SHUD/src/Model/MD_rhs_core.cpp` L324-354 | river bucket | P1d.2.3 (PR-E #279) |

## §2 删除 site 1 — element bucket (P1d.2.1)

### 2.1 Before (P1d era, SHUD pin `210ac191...`)

per `git diff 226e3ab..3341368 -- src/Model/MD_rhs_core.cpp` extract:

```cpp
void Model_Data::rhs_update(double *Y, double *DY, double t){
    /* P1d.2.1 (#277) — steady-state element first-touch warm-up.
     * Mirrors allocation-time flat3 zero-write in Model_Data.cpp::
     * malloc_EleRiv L302-317; complements (does NOT replace) it by
     * re-touching the element-owned hot flux/scratch arrays on each
     * CVODE RHS evaluation so each owner thread page-faults its own
     * NUMA-local pages every step. Pure zero-write, no reduction /
     * accumulation / non-zero write (design D2 + R1). Every field
     * below is re-zeroed or fully overwritten LATER in this same
     * rhs_update body before any read:
     *   QeleSubAt(i,j)  -> re-zeroed L70   (`QeleSubAt(i,j)  = 0.`)
     *   QeleSurfAt(i,j) -> re-zeroed L71   (`QeleSurfAt(i,j) = 0.`)
     *   QeleSubTot[i]   -> re-zeroed L72   (`QeleSubTot[i]   = 0.`)
     *   QeleSurfTot[i]  -> re-zeroed L73   (`QeleSurfTot[i]  = 0.`)
     *   Qe2r_Surf[i]    -> re-zeroed L134  (`Qe2r_Surf[i]    = 0.`)
     *   Qe2r_Sub[i]     -> re-zeroed L135  (`Qe2r_Sub[i]     = 0.`)
     * Field set = element-owned subset only (no river / lake fields)
     * per OQ1 doc docs/p1d/p1d_first_touch_design.md. Gated by
     * g_numa_first_touch_enabled to match allocation-time gate
     * (preserves deterministic-vs-serial behavior when OMP_PROC_BIND
     * is unset; see shud.cpp L41-L77). */
    if (g_numa_first_touch_enabled) {
        int i;
#pragma omp parallel for schedule(static) default(none) shared(QeleSurfTot, QeleSubTot, Qe2r_Surf, Qe2r_Sub) private(i)
        for (i = 0; i < NumEle; i++) {
            for (int j = 0; j < 3; j++) {
                QeleSurfAt(i, j) = 0.0;
                QeleSubAt(i, j)  = 0.0;
            }
            QeleSurfTot[i] = 0.0;
            QeleSubTot[i]  = 0.0;
            Qe2r_Surf[i]   = 0.0;
            Qe2r_Sub[i]    = 0.0;
        }
    }

    for (int i = 0; i < NumEle; i++) {
```

### 2.2 After (P1e era, SHUD pin `3341368`)

```cpp
void Model_Data::rhs_update(double *Y, double *DY, double t){
    /* P1e PR-H (#316, design D2) — `#pragma omp for schedule(static)`
     * on each top-level loop below. When invoked from
     * `ExecPolicy::StrictOMP` the directives work-share the iteration
     * space across the outer team; when invoked from `ExecPolicy::Serial`
     * (no enclosing parallel region) each `omp for` becomes an
     * orphaned construct and OpenMP semantics execute it in the
     * encountering thread (= serial), so mode A behaviour is
     * preserved bit-for-bit. The five upstream loops use `nowait`
     * because their owner-local writes target disjoint slots; only
     * the final `for (i < NumY) DY[i] = 0.` omits `nowait` so its
     * implicit barrier closes Phase 1 before rhs_flux reads DY. */
    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NumEle; i++) {
```

### 2.3 删除 rationale

per PR-H commit body + design D4：

1. **nested parallel region 违反 single-region rule**：PR-F StrictOMP 已开 outer `#pragma omp parallel` region；inner `#pragma omp parallel for` 嵌套 → 违反 spec p1e-strict-omp-rhs Requirement "单 parallel region"
2. **steady-state warm-up redundant**：outer team 在 PR-H 改造后的 `#pragma omp for schedule(static)` 第一次 enter 时已 distribute owner-local work；each thread 第一次访问自己 owner slot 时即 fault-in NUMA-local pages，无需 inner pre-touch
3. **allocation-time first-touch 保留**：`Model_Data.cpp::malloc_EleRiv` L302-317 flat3 zero-write 仍在 single-thread init 阶段做一次 fault-in（per §5 verify）；mode A path 进 RHS hot path 时 page-fault 已完成

## §3 删除 site 2 — lake bucket (P1d.2.2)

### 3.1 Before (P1d era L169-203)

(P1d era PR-D #278 添加的 lake first-touch loop, 与 element block 相同模式：`if (g_numa_first_touch_enabled)` guard + `#pragma omp parallel for` + lake-owned 字段 zero-write)

### 3.2 After (PR-H 删除)

lake-owned 字段 (LK_yLake / LK_DY / LK_qLake* 等) 改由 P1e StrictOMP outer region 内 lake bucket 的 `#pragma omp for schedule(static)` work-sharing 完成 fault-in (per §2.2 element bucket 同模式)。

## §4 删除 site 3 — river bucket (P1d.2.3)

### 4.1 Before (P1d era L324-354)

(P1d era PR-E #279 添加的 river first-touch loop, 同 element/lake 模式：river-owned 字段 zero-write `if (g_numa_first_touch_enabled)`)

### 4.2 After (PR-H 删除)

river-owned 字段 (uYriv / QrivSurf / QrivSub / QrivUp / QrivDown 等) 改由 P1e StrictOMP outer region 内 river bucket 的 `#pragma omp for schedule(static)` work-sharing 完成 fault-in。

## §5 allocation-time + load-time first-touch 保留 verify

per PR-H commit body：

> The allocation-time first-touch in Model_Data.cpp::malloc_EleRiv + load-time first-touch in MD_initialize.cpp::LoadIC remain (they fire once outside the RHS hot path)

verify command + 实测结果：

### 5.1 allocation-time first-touch (Model_Data.cpp::malloc_EleRiv L302-317)

```bash
$ grep -nA2 "flat3 zero-write\|first-touch\|page-fault" SHUD/src/ModelData/Model_Data.cpp | head -30
```

期望 ≥1 hit 含 P1d era 的 flat3 zero-write loop（PR-C #277 添加，PR-H 不删）。

### 5.2 load-time first-touch (MD_initialize.cpp::LoadIC)

```bash
$ grep -nA2 "first-touch\|LoadIC" SHUD/src/ModelData/MD_initialize.cpp | head -20
```

期望 ≥1 hit 含 P1d era 的 load-time first-touch（PR-C/D/E 添加，PR-H 不删）。

### 5.3 extern flag declaration 保留

per PR-H commit 修订：`extern int g_numa_first_touch_enabled;` declaration 在 `MD_rhs_core.cpp` 内保留（即使本文件 3 处 first-touch loops 删除），rationale per PR-H commit body：

> The extern declaration is kept here purely so legacy non-StrictOMP modes that may still want to consult the flag can do so via this translation unit if needed.

`shud.cpp` L41-L77 内 `g_numa_first_touch_enabled` 的定义 + `emit_numa_token()` SHUD() entry L70-91 设值逻辑全部保留（P1d era infrastructure 不动）。

## §6 mode A path bitwise preserved verify

PR-H 删除 3 处 first-touch loop 后 mode A (ExecPolicy::Serial) path 是否仍 bitwise 对比 P1d era 关键 verify。per PR-H commit body claim：

> mode A behaviour is preserved bit-for-bit

verify path：mode A reference SHA 是 PR-D #312 (P1d era + PR-A-E reorg, SHUD pin `210ac191`) 锁定 → P1e PR-I 跑 mode A reproduction 仍命中同 SHA：

| case | mode A reference SHA12 (PR-D #312) | P1e era mode A reproduction SHA12 (PR-I §1.2) | match |
|---|---|---|:---:|
| heihe | `a2023ccd2de4` | `a2023ccd2de4` | same |
| heihe_x4 | `b5e4b0a2cf83` | `b5e4b0a2cf83` | same |

PR-H 删 first-touch 后 mode A 仍 bitwise reproduce → 这些 first-touch loops 在 Serial path 上**从来没改变过 SHA**（因为它们的字段在同 RHS evaluation 内被 re-zero 或 fully overwrite，per §2.1 注释列表）→ 它们只为 (假设的) parallel writer first-touch 而存在；实际 P1c/d era `f.cpp` 始终 ExecPolicy::Serial 调 RHS → 这些 loop 一直是 dead-write-then-overwrite cycle。

## §7 与 P1c/d era 路径决策的关系

P1c/d era 假设：RHS 跨 N drift 是 NUMA writer first-touch 问题 → 加 steady-state first-touch warm-up → fix。

P1d capstone fact-check 5/5 推翻：drift 真正来自 NVECTOR_OPENMP `reduction(+:sum) schedule(static)` 跨 N reduction tree 不固定 (per ADR-0002 Fact #1+#2+#3)。RHS 始终是 Serial path，writer first-touch hypothesis 根本不适用。

P1e PR-H 通过 (1) 删 3 处 first-touch loop + (2) 实证 mode A 仍 bitwise = 把 P1d era 这 3 处 loop "其实未必有用" 这一点 documented 化 + (3) 为 StrictOMP single-region rule 让路 = 把死循环让位给真正在用的 worksharing directive。

## §8 验证 grep gate (per spec p1e-strict-omp-rhs)

| 验证项 | command | 期望 |
|---|---|---|
| 3 处 first-touch loop 删除 | `grep -c 'g_numa_first_touch_enabled' SHUD/src/Model/MD_rhs_core.cpp` | extern declaration 保留 → 1 (header extern only, 不含执行 site) |
| inner #pragma omp parallel 不再嵌套 | `grep -c '#pragma omp parallel\b' SHUD/src/Model/MD_rhs_core.cpp` (不含 `parallel for`) | 1 (单 StrictOMP region, per `docs/p1e/p1e_strict_omp_design.md` §2) |
| inner #pragma omp parallel for 不再出现 | `grep -c '#pragma omp parallel for' SHUD/src/Model/MD_rhs_core.cpp` | 0 |
| allocation-time first-touch 保留 | `grep -c 'first-touch\|page-fault\|flat3' SHUD/src/ModelData/Model_Data.cpp` | ≥1 |
| load-time first-touch 保留 | `grep -c 'first-touch\|LoadIC' SHUD/src/ModelData/MD_initialize.cpp` | ≥1 |
| extern declaration 保留 | `grep -c 'extern.*g_numa_first_touch_enabled' SHUD/src/Model/MD_rhs_core.cpp` | 1 |

## §9 forward considerations

- **dead code 进一步 cleanup**：P1c/d era 的 `g_numa_first_touch_enabled` flag + `emit_numa_token()` 在 P1e era 仅 `shud.cpp` 内 set + `MD_rhs_core.cpp` 内 extern read but never used。future epic 可考虑彻底删除（包括 `shud.cpp` set 逻辑）以减少 cognitive load。
- **allocation-time first-touch policy 优化**：`Model_Data.cpp::malloc_EleRiv` 内 flat3 zero-write 当前 single-thread；P2a 可评估是否改为 `#pragma omp parallel for` (init phase 不计 hot path) 以提升 NUMA-local distribution。
