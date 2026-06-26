# P1e — `ExecPolicy::StrictOMP` 实现细节

P1e PR-F (#315) + PR-H (#316) 在 `SHUD/src/Model/MD_rhs_core.cpp` 实施的 `ExecPolicy::StrictOMP` 设计细节 + rationale。本 doc 是 spec `p1e-strict-omp-rhs` Requirement "ExecPolicy::StrictOMP 实施" + design.md D2 / D4 的 capstone 引用。

## §1 ExecPolicy enum 设计

`SHUD/src/Model/MD_rhs_core.cpp` (P1e era) 内 `ExecPolicy` enum:

```cpp
enum class ExecPolicy {
    Serial,        // canonical — 单线程,无 #pragma omp 进入,mode A/B 使用
    StrictOMP,     // PR-F 新增 — 单 #pragma omp parallel region,所有 RHS 方法 omp for share,mode C/D 使用 (mode C only in P1e; mode D deferred per tasks §2.5.1)
    // 历史 abort 桩 (P1d era ProductionOMP/...) 已删除
};
```

call chain：
- `f.cpp::54` `MD->rhs_core(Y, DY, t, ExecPolicy::Serial)` — `f()` 永久 Serial entry（per ADR-0002 Fact #1）
- mode C 入口：(P1e PR-F TBD entrypoint, per design D2) — 当 `SHUD_ENABLE_OPENMP_RHS=1` 编入 binary，`f()` 内 ExecPolicy 选择由 build flag 控

per ADR-0002 + design D2，mode C 触发 ExecPolicy::StrictOMP 的具体决策是 build-time（不是 runtime branching），避免运行时分支预测污染 hot path。

## §2 单 parallel region rationale

per design D2 + tasks §3.4：

```cpp
// pseudo-code, illustrating single parallel region invariant
void Model_Data::rhs_core(double *Y, double *DY, double t, ExecPolicy policy) {
    switch (policy) {
    case ExecPolicy::Serial:
        rhs_update(Y, DY, t);
        rhs_flux(Y, DY, t);
        rhs_apply(Y, DY, t);
        rhs_deterministic_gather(Y, DY, t);
        break;
    case ExecPolicy::StrictOMP:
        #pragma omp parallel default(none) shared(...) firstprivate(t)
        {
            // 4 method invocations inside single parallel region.
            // Each method internally has `#pragma omp for schedule(static)`
            // on top-level loops — those becomes worksharing under this team.
            rhs_update(Y, DY, t);
            rhs_flux(Y, DY, t);
            rhs_apply(Y, DY, t);
            rhs_deterministic_gather(Y, DY, t);
        }
        break;
    }
}
```

**单 parallel region invariant 三个核心约束**（per design D2 + spec p1e-strict-omp-rhs Requirement "单 parallel region"）：

1. **#pragma omp parallel 在 `rhs_core` 内单点出现**：4 method 调用全包在同一个 region 内。验证：`grep -c '#pragma omp parallel' SHUD/src/Model/MD_rhs_core.cpp` SHALL == 1。
2. **4 method 内部不再嵌套 `#pragma omp parallel`**：method 内只允许 `#pragma omp for schedule(static)` (orphaned construct，在外部 region 内作 worksharing；在 Serial path 下成 orphaned + serial fallback)。验证：`grep -c '#pragma omp parallel' SHUD/src/Model/MD_rhs_core.cpp` SHALL == 1 (上述 method 内部不增加 count)。
3. **4 method 顺序固定**：rhs_update → rhs_flux → rhs_apply → rhs_deterministic_gather，每 method 内有 phase barrier (omp for 末 implicit barrier 除最后一个 nowait 或 nowait + explicit barrier 替代)。

## §3 Phase-based for 设计 (PR-H)

per design D4 + tasks §3.6：

PR-F 初版 `#pragma omp single` 调 4 method（per design D2 scaffolding；2026-06-25 commit 226e3ab）；PR-H 把它改造为 `#pragma omp for schedule(static)` work-sharing。

4 method 内每个 top-level loop 的 directive：

```cpp
// rhs_update example (per SHUD commit 3341368)
void Model_Data::rhs_update(double *Y, double *DY, double t) {
    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NumEle; i++) {
        // ... element bucket
    }

    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NumRiv; i++) {
        // ... river bucket (uYriv set + lake module)
    }

    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NumRiv; i++) {
        QrivSurf[i] = 0.; QrivSub[i] = 0.; QrivUp[i] = 0.;
    }

    #pragma omp for schedule(static) nowait
    for (int i = 0; i < NumEle; i++) {
        Qe2r_Surf[i] = 0.; Qe2r_Sub[i] = 0.;
    }

    // 最后一 loop 不能 nowait — rhs_flux 后续会读 DY[iSF]/[iUS]/[iGW]
    #pragma omp for schedule(static)
    for (int i = 0; i < NumY; i++) DY[i] = 0.;
    // implicit barrier here closes Phase 1 (rhs_update) before Phase 2 (rhs_flux) reads DY
}
```

**3 个 nowait 规则**（per design D4 + PR-H commit 3341368 + TSan race fix annotation）：

1. **同 bucket 内的 loops** (e.g. element bucket 多 loop) 可 nowait — owner-local writes 落在 disjoint slot，下一 loop 不读上一 loop 的结果。
2. **同 bucket 跨 buffer 的 loops** (e.g. NumEle 在 zero-init buffers 后接 init Qe2r) 可 nowait — 同上 disjoint。
3. **最后 loop SHALL NOT nowait** — phase 边界由 implicit barrier 守门。具体来说：rhs_update 末 `DY[i] = 0.` 不 nowait → rhs_flux 才能安全读 DY。同理 rhs_flux 末非 nowait → rhs_apply 安全；rhs_apply 末非 nowait → rhs_deterministic_gather 安全；rhs_deterministic_gather 末非 nowait → 外部 single region 关闭。

**TSan-confirmed race scope**（per PR-H commit body）：

> ET bucket cannot carry nowait because the lateral bucket reads `hot.u_effKH[inabr]` (neighbour SoA slot written by sync_hot_dynamic(i) in ET). Same reasoning forces no-nowait on lateral / segment / river buckets feeding into rhs_deterministic_gather() — every cross-thread read of a freshly written field requires the trailing barrier.

## §4 complexity analysis

### 4.1 Serial path (ExecPolicy::Serial, mode A/B)

- 4 method per RHS evaluation
- 每 method M 个 top-level loop, 每 loop O(NumEle) 或 O(NumRiv) 或 O(NumY)
- 每 RHS evaluation total cost = O(NumEle + NumRiv + NumY) × constant
- CVODE 6698 internal steps × ~3 RHS evals per step → O(20000 × NumEle) per 90-day run

### 4.2 StrictOMP path (ExecPolicy::StrictOMP, mode C/D)

- 单 `#pragma omp parallel` region per RHS evaluation, region 开销 ~µs 量级 (libgomp benchmark)
- 4 method 内 ~10 `#pragma omp for` directive, 每 directive worksharing overhead ~ns 量级
- 每 RHS evaluation overhead = 1 × parallel region create/destroy + (10 × omp for entry + 10 × implicit barrier) ≈ 数 µs
- 累计 overhead = 6698 × 3 × ~µs = ~20 ms (vs heihe wall 504 s, 0.004%) — negligible at large case
- per-thread compute = O((NumEle + NumRiv + NumY) / N) per RHS evaluation
- 理想 N=8 speedup = N / (1 + N×overhead_frac / compute_frac)

### 4.3 实测 vs 理论 (heihe_x4)

- 实测 sp@8 = 1.729× (per `docs/p1e/p1e_perf_baseline.md` §3.2)
- Amdahl-反推 serial fraction f ≈ 63.4%
- serial fraction 来源：(1) `f.cpp` 单线程入口 + (2) CVODE 内部 SUNLinSol_SPGMR 单线程矩阵向量乘 + (3) PR-B0 recompute_for_output helper 单线程（每 output step 调一次）+ (4) `summary` / `ExportResults` 单线程
- 4 项 serial fraction 累加 ≈ 60-70%, 与实测 63% 吻合

## §5 与 P1d era first-touch 的关系（PR-H 删除）

P1d.2.1/.2/.3 在 `SHUD/src/Model/MD_rhs_core.cpp` L62-95 / L169-203 / L324-354 添加的 3 处 inner `#pragma omp parallel for` steady-state first-touch loops，PR-H 删除原因（per design D4 + commit 3341368 commit body）：

1. **nested parallel region 违反 single-region rule**：StrictOMP 已开 outer parallel region；inner `#pragma omp parallel` 嵌套 → 违反 §2 第 1 约束。
2. **steady-state warm-up redundant**：outer team 在第一个 omp for 已 distribute 同 work；each owner thread 自然在第一次 RHS evaluation 内 fault-in NUMA-local pages。无需 inner pre-touch。
3. **保留 allocation-time + load-time first-touch**：`Model_Data.cpp::malloc_EleRiv` flat3 zero-write (allocation) + `MD_initialize.cpp::LoadIC` (load) 这两处仍在 single-thread init 阶段 fault-in 一次；StrictOMP path 进 RHS hot path 时 page-fault 已完成。

## §6 alternative considered + 否决

per design D2 / D4 评估表：

| Alternative | 设计 | 否决理由 |
|---|---|---|
| Nested parallel region | inner omp parallel per method | 违反 single-region rule；nested OMP overhead 高；GCC libgomp 默认禁用 nested |
| Task-based | omp task per loop | task 调度开销 > worksharing 开销 for fixed-size loops；deterministic 困难 |
| Dynamic schedule | omp for schedule(dynamic) | bitwise determinism 受 schedule 影响（owner-local fold 顺序不固定）→ 否决 |
| Guided schedule | omp for schedule(guided) | 同 dynamic 否决 |
| **selected: static schedule + single region** | omp parallel + omp for schedule(static) | bitwise deterministic + lowest overhead + 直观 worksharing |

## §7 验证 grep gate (per spec p1e-strict-omp-rhs)

| 验证项 | command | 期望 |
|---|---|---|
| 单 parallel region | `grep -c '#pragma omp parallel\b' SHUD/src/Model/MD_rhs_core.cpp` (不含 `parallel for`) | 1 |
| omp for 多处 worksharing | `grep -c '#pragma omp for' SHUD/src/Model/MD_rhs_core.cpp` | ≥10 |
| schedule(static) 一致 | `grep -c 'schedule(static)' SHUD/src/Model/MD_rhs_core.cpp` | ≥10 |
| dynamic/guided 禁用 | `grep -cE 'schedule\((dynamic\|guided)\)' SHUD/src/Model/MD_rhs_core.cpp` | 0 |
| 末 loop 非 nowait | `grep -A1 'rhs_deterministic_gather' SHUD/src/Model/MD_rhs_core.cpp \| tail -1` | non-nowait |
| TSan annotation 存在 | `grep -c 'TSan' SHUD/src/Model/MD_rhs_core.cpp` | ≥1 |

## §8 forward considerations

- **P2a `OMP_SCHEDULE` tuning**：本 epic hardcode schedule(static)；P2a 评估 dynamic/guided 的 bitwise impact + 速度 trade-off
- **cache-line padding for owner-local SoA fields**：per `docs/adr/0001-soa-hot-fields.md` 已 documented 但 deferred；P2a 评估
- **NUMA cross-socket migration cost**：dual-socket server 实测时若 fork-join 频次过高 → P2a 评估 task affinity binding
