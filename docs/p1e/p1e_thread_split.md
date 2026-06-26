# P1e — `SHUD_RHS_THREADS` vs `OMP_NUM_THREADS` runbook

P1e PR-G (#315) 实施的 thread-count env split + Makefile `SHUD_ENABLE_OPENMP_RHS` build flag 使用说明。本 doc 作 spec p1e-strict-omp-rhs Requirement "thread count split" 的 capstone runbook + user-facing 运营指南。

## §1 两个 env-var 的职责

P1e era runtime env-var pair：

| Env-var | 控制对象 | scope | P1e 模式下含义 |
|---|---|---|---|
| `SHUD_RHS_THREADS` | StrictOMP RHS path 并行度（`#pragma omp parallel` team size） | `SHUD/src/Model/shud.cpp` startup 单点 set | mode C/D **唯一 canonical knob** for RHS 线程数 |
| `OMP_NUM_THREADS` | NVECTOR backend 线程数（`N_VNew_OpenMP` 内 `omp_get_max_threads()` 默认值） | OpenMP runtime 全局 default | mode B/D NVECTOR 后端使用；mode C 因 NVECTOR=Serial 不使用 |

**canonical pair**：mode C 仅看 `SHUD_RHS_THREADS`；mode D 同时受 `SHUD_RHS_THREADS` (RHS) + `OMP_NUM_THREADS` (NVector) 双控（mode D 是 research 边界，本 epic 不 ship）。

## §2 shud.cpp startup single-point set

per `SHUD/src/Model/shud.cpp` L142-L170 (P1e era, SHUD pin `3341368`)：

```cpp
#if defined(SHUD_ENABLE_OPENMP_RHS)
    {
        const char* shud_rhs_threads_env = getenv("SHUD_RHS_THREADS");
        int rhs_threads = shud_rhs_threads_env ? atoi(shud_rhs_threads_env) : 0;
        if (rhs_threads <= 0) {
            rhs_threads = omp_get_max_threads();
        }
        omp_set_num_threads(rhs_threads);
        fprintf(stdout, "P1e startup: SHUD_RHS_THREADS=%s -> omp_set_num_threads(%d); omp_get_max_threads=%d\n",
                shud_rhs_threads_env ? shud_rhs_threads_env : "(unset)",
                rhs_threads,
                omp_get_max_threads());
    }
#endif
```

priority chain：

1. `SHUD_RHS_THREADS=N` (N>0) → `omp_set_num_threads(N)`
2. `SHUD_RHS_THREADS=` (空) 或 unset → fallback to `omp_get_max_threads()`（itself driven by `OMP_NUM_THREADS` if set, else OpenMP runtime default = #logical cores）
3. `SHUD_RHS_THREADS=0` 或负 → 同 unset (treated as fallback)

## §3 build flag `SHUD_ENABLE_OPENMP_RHS` 使用

PR-G Makefile -fopenmp 自动 wire：

```makefile
# SHUD/Makefile (P1e era)
SHUD_ENABLE_OPENMP_RHS ?= 0  # default off

ifeq ($(SHUD_ENABLE_OPENMP_RHS),1)
    CXX_BASE_FLAGS += -DSHUD_ENABLE_OPENMP_RHS=1
    ifeq ($(UNAME_S),Linux)
        CXX_BASE_FLAGS += -fopenmp
        LDFLAGS += -fopenmp
    endif
    ifeq ($(UNAME_S),Darwin)
        CXX_BASE_FLAGS += -Xpreprocessor -fopenmp -I$(shell brew --prefix libomp)/include
        LDFLAGS += -L$(shell brew --prefix libomp)/lib -lomp
    endif
endif
```

user 不需手动 pass `-fopenmp`：`make shud SHUD_ENABLE_OPENMP_RHS=1` 即触发 Makefile 自动 wire (Linux + Darwin 两端)。

## §4 build × env 矩阵 + runtime 行为

| build flag | binary | NVector backend | RHS path | `SHUD_RHS_THREADS` 生效 | `OMP_NUM_THREADS` 生效 |
|---|---|---|---|:---:|:---:|
| 无 (default mode A) | `shud` | Serial | Serial | NO (`#ifdef SHUD_ENABLE_OPENMP_RHS` 未触发) | NO (NVector=Serial) |
| `SHUD_USE_OPENMP_NVECTOR=1` (= mode B = `shud_omp`) | `shud_omp` | OpenMP | Serial | NO | YES (NVector 内 reduction) |
| `SHUD_ENABLE_OPENMP_RHS=1` (= mode C) | `shud` | Serial | StrictOMP | **YES** (canonical) | NO (NVector=Serial) |
| `SHUD_USE_OPENMP_NVECTOR=1 SHUD_ENABLE_OPENMP_RHS=1` (= mode D) | `shud_omp` | OpenMP | StrictOMP | YES (RHS path) | YES (NVector reduction) |

mode C 是本 epic SHIP target；mode A 是 canonical reference；mode B 是 P1c/d era 历史 prod；mode D 是 research 边界 (Phase 2 96-cell deferred)。

## §5 runtime diagnostic

mode C startup 输出 (per shud.cpp L165 `fprintf` block)：

```
$ SHUD_RHS_THREADS=4 ./shud keliya
... (其他 init log) ...
P1e startup: SHUD_RHS_THREADS=4 -> omp_set_num_threads(4); omp_get_max_threads=4
... (其余 simulation log) ...
```

验证用 (per `openspec/.../spec.md` "mode C runtime thread count diagnostic" Scenario)：

| env | 期望 startup line |
|---|---|
| `SHUD_RHS_THREADS=4 ./shud keliya` | `omp_set_num_threads(4)` + `omp_get_max_threads=4` |
| `SHUD_RHS_THREADS=8 ./shud keliya` | `omp_set_num_threads(8)` + `omp_get_max_threads=8` |
| `SHUD_RHS_THREADS=1 ./shud keliya` | `omp_set_num_threads(1)` + `omp_get_max_threads=1` |
| `./shud keliya` (unset) | `SHUD_RHS_THREADS=(unset)` + `omp_set_num_threads(<default>)` + `omp_get_max_threads=<default>` |

## §6 binary symbol verification

per `openspec/.../spec.md` "build C binary symbol" Scenario：

```bash
# mode C build
cd SHUD
git checkout 3341368d2d0854924d2286925c8575df52cc97a0
make clean && make shud SHUD_ENABLE_OPENMP_RHS=1

# verify mode C 真链 OpenMP runtime
nm ./shud | grep N_VNew_Serial             # SHALL ≥1 hit (NVector=Serial 真链)
nm ./shud | grep N_VNew_OpenMP             # SHALL 0 hit  (NVector!=OpenMP 真不链)

# Linux
nm ./shud | grep GOMP_parallel             # SHALL ≥1 hit (libgomp 真链入)
ldd ./shud | grep -E 'libgomp\|libomp'     # SHALL ≥1 hit

# Darwin
nm ./shud | grep _omp_set_num_threads      # SHALL ≥1 hit (Apple Clang 下符号前导 _)
otool -L ./shud | grep -E 'libomp\|libgomp' # SHALL ≥1 hit (Apple Silicon: libomp)
```

shud.cpp 守门形式 (PR-G 实施)：**拆为两段 form**（per spec L192 拆为两段允许 + tasks §3.5.2 "或拆为两段"）：

| call site | 守门 | thread count source |
|---|---|---|
| L132 | `#ifdef SHUD_USE_OPENMP_NVECTOR` | `MD->CS.num_threads` (legacy NVector parity) |
| L157 | `#if defined(SHUD_ENABLE_OPENMP_RHS)` | `getenv("SHUD_RHS_THREADS")` ?: `omp_get_max_threads()` |

union form (`#if defined(SHUD_USE_OPENMP_NVECTOR) \|\| defined(SHUD_ENABLE_OPENMP_RHS)`) 也合规 (per spec amend by PR-K, 二选一)，但 PR-G 选择拆为两段以分别 anchored 到各自语义。

## §7 user-facing 运营建议

### 7.1 heihe small-case (NumEle = 6335)

per `docs/p1e/p1e_perf_baseline.md` §6 small-case carve-out：

| `SHUD_RHS_THREADS` | wall (s) | speedup vs N=1 |
|---|---:|---:|
| 1 | 504 | 1.00× |
| 4 | 488 | 1.033× |
| 8 | 473 | 1.066× |

**建议**：heihe small-case 默认 `SHUD_RHS_THREADS=1`（与 P1d era 默认一致）。8 线程仅 6.6% 收益 + fork-join overhead 不划算。若 user 需要快速 turnaround 接受 ~6% 收益，可设 `SHUD_RHS_THREADS=8`。

### 7.2 heihe_x4 production-target mesh (NumEle=40046)

per `docs/p1e/p1e_perf_baseline.md` §3.2：

| `SHUD_RHS_THREADS` | wall (s) | speedup vs N=1 |
|---|---:|---:|
| 1 | 1340 | 1.000× |
| 2 | 1051 | 1.275× |
| 4 | 850 | 1.576× |
| 8 | 775 | 1.729× |

**建议**：heihe_x4 默认 `SHUD_RHS_THREADS=4` (达 ≥1.5× threshold) 或 `=8` (1.729×，接近 Amdahl 上界)。production 推荐 `=4` 以平衡 ROI + 节点利用率。

### 7.3 mode B legacy 兼容

P1c/d era `shud_omp` (mode B) 不删除：仍可用 `OMP_NUM_THREADS=N ./shud_omp <case>` 跑历史脚本。但 mode B 跨 N 不 bitwise (per `docs/p1e/p1e_pr_d_2x2_server.md` §3.4)，不适用于 strict reproducibility 场景。

### 7.4 Slurm sbatch 推荐

per CLAUDE.md "Slurm 三铁律"：

```bash
#!/bin/bash
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/heihe_x4_N8.out
#SBATCH --error=/scratch/frd_muziyao/SHUD-OpenMP/.p1e-runs/heihe_x4_N8.err

# mode C w/ 8 RHS threads (heihe_x4 推荐)
export SHUD_RHS_THREADS=8
export OMP_PROC_BIND=close
export OMP_PLACES=cores

cd /scratch/frd_muziyao/SHUD-OpenMP/SHUD
./shud heihe_x4
```

注：`OMP_NUM_THREADS` 不设 (mode C NVector=Serial 不需要)。`OMP_PROC_BIND` + `OMP_PLACES` 保留 from P1d era 习惯（mode C path 内 RHS thread 仍受 these env 影响）。

## §8 与 P1d era 的差异

| 维度 | P1d era | P1e era |
|---|---|---|
| RHS 并行度 env | (无 — RHS 始终 Serial per ADR-0002 Fact #1) | `SHUD_RHS_THREADS` (新增, mode C/D) |
| NVector 并行度 env | `OMP_NUM_THREADS` (mode B `shud_omp` 使用) | 同 (mode B/D 仍使用，mode C 不使用) |
| production default | `OMP_NUM_THREADS=1` (P1d E′ closure: serial fallback) | `SHUD_RHS_THREADS` per case roll-out (heihe_x4 `=4` 推荐) |
| build target | `shud` (serial) + `shud_omp` (mode B) | `shud` + `shud_omp` (二者均 + `SHUD_ENABLE_OPENMP_RHS=1` opt-in 成 mode C/D) |
| build flag wire | manual `-fopenmp` (user 自己 pass) | automatic (Makefile `SHUD_ENABLE_OPENMP_RHS=1` 自动 wire `-fopenmp` Linux / `-Xpreprocessor -fopenmp -lomp` Darwin) |
