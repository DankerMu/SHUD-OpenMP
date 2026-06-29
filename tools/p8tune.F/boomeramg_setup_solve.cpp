/* tools/p8tune.F/boomeramg_setup_solve.cpp
 *
 * P8-tune.F BoomerAMG/Hypre pattern-only spike (PR-A). Per openspec change
 * `p8tune-amg-spike` spec §REQ-1 (zero-source-patch + zero-CVODE-wireup),
 * §REQ-2 (FD-color Jacobian reuse via shell-out to p8tune.D), §REQ-4
 * (cell_summary KV schema + 5 verdict_class values).
 *
 * Flow (mirror P8-tune.D klu_analyze_factor.cpp pattern):
 *
 *   1. CLI arg parse: --case <name> --interp-type <N> --coarsen-type <N>
 *      [--basin-root <path>] [--in <prefix>] [--n-solve <N>] [--rhs-mode <m>]
 *
 *   2. Pre-flight: shell out to ../p8tune.D/dump_adjacency + fd_color_jacobian
 *      via subprocess (REQ-2 Scenario "FD-color Jacobian reuse from P8-tune.D").
 *      ABORT on non-zero exit. (The 4-combo Mac smoke loop in README.md §5
 *      runs these once outside the loop, so this binary's pre-flight is a
 *      defensive idempotent check; if the .bin already exists with non-zero
 *      size we skip the shell-out.)
 *
 *   3. Read numeric J binary at $basin_root/$case/$in_prefix_numeric_J.bin
 *      (JBin format: magic NDNJ + version=1 + NumY + total_nnz + n_colors +
 *      col_ptr[NumY+1] + row_idx[total_nnz] + values[total_nnz]).
 *
 *   4. Build CVODE-BDF-equivalent test matrix M = I - γ·J (γ=1.0 unit-step
 *      BDF surrogate, mirror klu_analyze_factor:188-230). Convert CSC →
 *      CSR (Hypre IJMatrix takes row-major).
 *
 *   5. HYPRE_Initialize / MPI_Init (MPI_COMM_SELF — single-process spike).
 *      HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, NumY-1, 0, NumY-1, ...) +
 *      SetObjectType(HYPRE_PARCSR) + Initialize() + per-row SetValues +
 *      Assemble() + GetObject(&ParCSR).
 *
 *   6. HYPRE_IJVectorCreate b, x — b = ones (or random per --rhs-mode), x = 0.
 *
 *   7. HYPRE_BoomerAMGCreate(&solver)
 *      SetInterpType(interp_type_arg)
 *      SetCoarsenType(coarsen_type_arg)
 *      SetMaxIter(100), SetTol(1e-8), SetPrintLevel(1)
 *      SetCumNnzAP(1.0)   ← enables operator-complexity tracking; required
 *                            for cell_summary KV `operator_complexity` field
 *                            since Hypre 3.1.0 omits HYPRE_BoomerAMGGetCycle
 *                            Complexity / GetOperatorComplexity getters.
 *
 *   8. Timed setup: HYPRE_BoomerAMGSetup(solver, A, b_par, x_par) →
 *      setup_wall_sec.
 *
 *   9. residual_reduction_v1: run ONE V-cycle (MaxIter=1, Tol=0) to capture
 *      the true 1-step residual reduction ratio per spec REQ-4 Scenario
 *      "AMG_SOLVE_DIVERGE marker on convergence failure". Then restore
 *      MaxIter=100/Tol=1e-8 for the main timed solve loop. (Without this
 *      separation, GetFinalRelativeResidualNorm returns the post-convergence
 *      ratio ~1e-8, making 1/r meaningless as a v1 metric — round-1 M1.)
 *
 *  10. Loop n_solve times: reset x = 0 via SetValues+Assemble ONLY (NO
 *      Initialize call, per Hypre 3.1.0 IJ docs — Initialize is for
 *      first-use only; calling on an assembled vector is UB — round-1 C1),
 *      timed apply HYPRE_BoomerAMGSolve → average apply_wall_sec.
 *
 *  11. HYPRE_BoomerAMGGetCumNnzAP → operator_complexity = cum_nnz_AP / nnz_A
 *      (Hypre canonical definition: sum of nnz across all A-level operators
 *      divided by fine-grid nnz).
 *      cycle_complexity = 2.0 × operator_complexity is a V-cycle estimate
 *      (1 pre-smoothing + 1 post-smoothing pass per level, weight=1) — NOT
 *      a measurement. Hypre 3.1.0 public API has no direct GetCycleComplexity;
 *      see README.md §6 and PR-C ADR-0007 §Discussion for the axis-independence
 *      caveat (Axis 4 mechanically tracks Axis 5 — round-1 H3).
 *
 *  12. peak_rss_bytes from mach (Mac) / getrusage (Linux).
 *
 *  13. Emit REQ-4 cell_summary KV block to stdout (fixed order for
 *      aggregator parser stability):
 *        CELL_SUMMARY_BEGIN
 *        case=<C> interp_type=<I> coarsen_type=<S> NumY=<N> nnz_A=<NNZ>
 *        setup_wall_sec=<S> apply_wall_sec=<A> peak_rss_bytes=<R>
 *        cycle_complexity=<CC> operator_complexity=<OC> residual_reduction_v1=<R1>
 *        verdict_class=<PASS|AMG_OOM|AMG_SETUP_DIVERGE|AMG_SOLVE_DIVERGE|AMG_WALL_OVERFLOW>
 *        hypre_version=<HV> colpack_version=<CV> shud_pin=<SHA>
 *        CELL_SUMMARY_END
 *      hypre_version probed at runtime via HYPRE_Version() (round-1 M5);
 *      colpack_version + shud_pin embedded at compile time via Makefile
 *      preprocessor defines (round-1 M6).
 *
 *  14. Marker emission paths (REQ-4 Scenarios 2-5):
 *      - HYPRE_BoomerAMGSetup nonzero return OR num_levels==0 OR
 *        setup_wall_sec > 2.0×WALL_BUDGET_SETUP_SEC
 *                                             → MARKER:AMG_SETUP_DIVERGE_DETECTED
 *                                              + verdict_class=AMG_SETUP_DIVERGE
 *        (round-1 H1: nlevels==0 + wall-budget triggers added per spec REQ-4)
 *      - HYPRE_BoomerAMGSolve nonzero return OR final_res > 1.0
 *        OR residual_reduction_v1 < 2.0
 *                                             → MARKER:AMG_SOLVE_DIVERGE_DETECTED
 *                                              + verdict_class=AMG_SOLVE_DIVERGE
 *        (round-1 H2: residual_reduction_v1<2.0 trigger added per spec REQ-4)
 *      - std::bad_alloc OR peak_rss > 8 GiB pin (Mac PR-A scale; PR-B server
 *        replaces with CN_NODE_RAM_BYTES probe)
 *                                             → MARKER:AMG_OOM_DETECTED
 *                                              + verdict_class=AMG_OOM
 *      - SIGTERM trap (PR-B Slurm budget overflow)
 *                                             → MARKER:AMG_WALL_OVERFLOW_DETECTED
 *                                              + verdict_class=AMG_WALL_OVERFLOW
 *      All marker paths still exit 0 (valid data point per REQ-4); only
 *      truly catastrophic build/CLI failures exit 1.
 *
 * Usage:
 *   ./boomeramg_setup_solve --case <name> --interp-type <N> --coarsen-type <N>
 *                           [--basin-root <path>=../../SHUD/Basins]
 *                           [--in <prefix>=case_name]
 *                           [--n-solve <N>=5]
 *                           [--rhs-mode {ones|random}=ones]
 *
 * Refs:
 *   openspec/changes/p8tune-amg-spike/specs/amg-pattern-spike-verdict/spec.md
 *     REQ-1 (zero-source-patch + zero-CVODE-wireup)
 *     REQ-2 (FD-color Jacobian reuse via shell-out)
 *     REQ-4 (cell_summary KV schema + 5 verdict_class values + marker
 *            emission paths)
 *   HYPRE 3.1.0 public headers: /opt/homebrew/Cellar/hypre/3.1.0/include/
 */

#include <algorithm>
#include <cerrno>       // errno for fork/execvp diagnostics (C2)
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>      // open() for fork/execvp dev/null redirect (C2)
#include <new>          // std::bad_alloc
#include <random>
#include <string>
#include <sys/stat.h>
#include <sys/wait.h>   // waitpid / WIFEXITED / WEXITSTATUS (C2)
#include <unistd.h>     // fork / execvp / dup2 / _exit (C2)
#include <vector>

#include <mpi.h>

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_krylov.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
// Internal Hypre header: needed for hypre_ParAMGDataNumLevels accessor
// macro (Hypre 3.1.0 omits the HYPRE_BoomerAMGGetNumLevels public getter
// that the spec REQ-4 AMG_SETUP_DIVERGE Scenario references; the macro
// reads the same `num_levels` field directly from the hypre_ParAMGData
// struct). Round-1 H1.
#include "_hypre_parcsr_ls.h"

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <sys/resource.h>
#endif

namespace {

constexpr uint32_t JBIN_MAGIC = 0x4A4E444Eu;  // 'N''D''N''J' little-endian
constexpr double   GAMMA       = 1.0;           // unit-step BDF surrogate
constexpr int      DEFAULT_N_SOLVE = 5;
// Mac PR-A peak-RSS pin (PR-B replaces with CN_NODE_RAM_BYTES * 0.95):
constexpr size_t   MAC_OOM_PIN_BYTES = 8ULL * 1024 * 1024 * 1024;  // 8 GiB

// Setup wall budget for AMG_SETUP_DIVERGE trigger (spec REQ-4 Scenario
// "AMG_SETUP_DIVERGE marker on hierarchy build failure"). Derived per
// spec REQ-5: WALL_BUDGET_SETUP_SEC = 1.5 × 0.7 × SPGMR_PER_STEP_SEC
// where SPGMR_PER_STEP_SEC = 0.226579 (P8-tune.D heihe_x4 N=1 maxl=5 3-rep
// median; pinned in tools/p8tune.D/spgmr_baseline_walls.h). PR-B/PR-C will
// replace this hardcode by including the shared header. Round-1 H1.
constexpr double WALL_BUDGET_SETUP_SEC = 1.5 * 0.7 * 0.226579;  // ≈ 0.237908 s

// ColPack version + SHUD pin defines: injected by Makefile via
// -DCOLPACK_VERSION_STR=... + -DSHUD_PIN_SHA=... at compile time
// (round-1 M6). Fall back to "unknown" sentinel if undefined (e.g.,
// hand-invoked g++ outside the project Makefile).
#ifndef COLPACK_VERSION_STR
#define COLPACK_VERSION_STR "unknown"
#endif
#ifndef SHUD_PIN_SHA
#define SHUD_PIN_SHA "unknown"
#endif

struct JBin {
    uint64_t NumY = 0, total_nnz = 0, n_colors = 0;
    std::vector<int64_t> col_ptr;
    std::vector<int32_t> row_idx;
    std::vector<double> values;
};

// Read JBin binary with full short-read detection. Every std::fread return
// is checked against the expected element count; on mismatch, emit a
// precise error (path + which field truncated) and close+return false.
// Prevents truncated .bin files (NFS hiccup, disk-full mid-write, SIGKILL
// mid-fd_color_jacobian) from silently feeding stale heap garbage into
// HYPRE_BoomerAMGSetup. Round-1 C5 (R5 + R1 + R2 + R3 convergent finding).
bool read_jbin(const std::string &path, JBin &j) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[amg] ERROR: cannot open %s\n", path.c_str());
        return false;
    }
    uint32_t magic = 0, version = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1) {
        std::fprintf(stderr, "[amg] ERROR: short read on magic in %s\n", path.c_str());
        std::fclose(fp); return false;
    }
    if (magic != JBIN_MAGIC) {
        std::fprintf(stderr, "[amg] ERROR: bad magic 0x%08x in %s (expected 0x%08x NDNJ)\n",
                     magic, path.c_str(), JBIN_MAGIC);
        std::fclose(fp); return false;
    }
    if (std::fread(&version, sizeof(version), 1, fp) != 1) {
        std::fprintf(stderr, "[amg] ERROR: short read on version in %s\n", path.c_str());
        std::fclose(fp); return false;
    }
    if (version != 1u) {
        std::fprintf(stderr, "[amg] ERROR: unsupported JBin version %u (expected 1)\n", version);
        std::fclose(fp); return false;
    }
    if (std::fread(&j.NumY, sizeof(uint64_t), 1, fp) != 1) {
        std::fprintf(stderr, "[amg] ERROR: short read on NumY in %s\n", path.c_str());
        std::fclose(fp); return false;
    }
    if (std::fread(&j.total_nnz, sizeof(uint64_t), 1, fp) != 1) {
        std::fprintf(stderr, "[amg] ERROR: short read on total_nnz in %s\n", path.c_str());
        std::fclose(fp); return false;
    }
    if (std::fread(&j.n_colors, sizeof(uint64_t), 1, fp) != 1) {
        std::fprintf(stderr, "[amg] ERROR: short read on n_colors in %s\n", path.c_str());
        std::fclose(fp); return false;
    }
    j.col_ptr.resize(j.NumY + 1);
    if (std::fread(j.col_ptr.data(), sizeof(int64_t), j.NumY + 1, fp) != (j.NumY + 1)) {
        std::fprintf(stderr, "[amg] ERROR: short read on col_ptr (expected %llu entries) in %s\n",
                     (unsigned long long)(j.NumY + 1), path.c_str());
        std::fclose(fp); return false;
    }
    j.row_idx.resize(j.total_nnz);
    if (j.total_nnz > 0 &&
        std::fread(j.row_idx.data(), sizeof(int32_t), j.total_nnz, fp) != j.total_nnz) {
        std::fprintf(stderr, "[amg] ERROR: short read on row_idx (expected %llu entries) in %s\n",
                     (unsigned long long)j.total_nnz, path.c_str());
        std::fclose(fp); return false;
    }
    j.values.resize(j.total_nnz);
    if (j.total_nnz > 0 &&
        std::fread(j.values.data(), sizeof(double), j.total_nnz, fp) != j.total_nnz) {
        std::fprintf(stderr, "[amg] ERROR: short read on values (expected %llu entries) in %s\n",
                     (unsigned long long)j.total_nnz, path.c_str());
        std::fclose(fp); return false;
    }
    std::fclose(fp);
    return true;
}

size_t peak_rss_bytes() {
#ifdef __APPLE__
    mach_task_basic_info_data_t info;
    mach_msg_type_number_t cnt = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  (task_info_t)&info, &cnt) == KERN_SUCCESS) {
        return static_cast<size_t>(info.resident_size_max);
    }
    return 0;
#else
    struct rusage r;
    if (getrusage(RUSAGE_SELF, &r) == 0) {
        // ru_maxrss is in KB on Linux, in bytes on BSD/Mac
        return static_cast<size_t>(r.ru_maxrss) * 1024;
    }
    return 0;
#endif
}

bool file_exists_nonempty(const std::string &path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return false;
    return st.st_size > 0;
}

// Shell out to ./<binary> --case <case>  (REQ-2 Scenario "FD-color
// Jacobian reuse from P8-tune.D"). Returns 0 on success, non-zero on
// subprocess failure. The subprocess chdirs to basin_dir internally
// (see dump_adjacency.cpp:375 / fd_color_jacobian.cpp:268), so we don't
// need to set CWD here — we just provide the relative path to the symlink.
//
// Invocation convention: boomeramg_setup_solve is launched from
// tools/p8tune.F/ (per README §5), where the Makefile created symlinks
// `./dump_adjacency` + `./fd_color_jacobian` → `../p8tune.D/<binary>`.
// The default --basin-root=../../SHUD/Basins resolves correctly from that
// CWD. The subprocess inherits CWD; the p8tune.D binaries chdir internally.
//
// Implementation: fork() + execvp() rather than std::system() so that
// case_name and basin_root flow as argv tokens — the shell is bypassed
// entirely, eliminating the shell-injection class (round-1 C2). Child
// silences stdout/stderr via dup2 to /dev/null. Parent waits + reports
// exit code (WIFEXITED + WEXITSTATUS for signal-vs-exit distinction —
// round-1 L1 bonus). Returns 0 on clean exit-0; non-zero (exit code
// or 128+signo) otherwise.
int shell_out_preflight(const std::string &binary_name,
                        const std::string &case_name,
                        const std::string &basin_root) {
    const std::string binary_path = "./" + binary_name;
    pid_t pid = fork();
    if (pid < 0) {
        std::fprintf(stderr, "[amg] ERROR: fork() failed for %s: %s\n",
                     binary_name.c_str(), std::strerror(errno));
        return -1;
    }
    if (pid == 0) {
        // child: silence stdout/stderr by redirecting to /dev/null, then
        // execvp the spike subprocess. argv tokens bypass any shell parsing.
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            dup2(devnull, STDOUT_FILENO);
            dup2(devnull, STDERR_FILENO);
            close(devnull);
        }
        // execvp signature requires non-const char *; cast away const for
        // the argv array — POSIX-defined safe pattern (the strings are not
        // mutated by execvp's lookup).
        char *argv[] = {
            const_cast<char *>(binary_path.c_str()),
            const_cast<char *>("--case"),
            const_cast<char *>(case_name.c_str()),
            const_cast<char *>("--basin-root"),
            const_cast<char *>(basin_root.c_str()),
            nullptr
        };
        execvp(argv[0], argv);
        // execvp only returns on failure.
        _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        std::fprintf(stderr, "[amg] ERROR: waitpid() failed for %s: %s\n",
                     binary_name.c_str(), std::strerror(errno));
        return -1;
    }
    if (WIFEXITED(status)) {
        const int rc = WEXITSTATUS(status);
        if (rc != 0) {
            std::fprintf(stderr, "[amg] ERROR: %s exited with status %d\n",
                         binary_name.c_str(), rc);
        }
        return rc;
    }
    if (WIFSIGNALED(status)) {
        const int signo = WTERMSIG(status);
        std::fprintf(stderr, "[amg] ERROR: %s killed by signal %d\n",
                     binary_name.c_str(), signo);
        return 128 + signo;
    }
    std::fprintf(stderr, "[amg] ERROR: %s exited abnormally (status=0x%x)\n",
                 binary_name.c_str(), status);
    return -1;
}

// Bracketed marker emission for stdout. Per REQ-4 marker-vs-class
// naming: stdout marker uses `_DETECTED` suffix, KV value omits it.
void emit_marker(const char *marker_class) {
    std::printf("MARKER:%s_DETECTED\n", marker_class);
    std::fflush(stdout);
}

// Probe Hypre version at runtime via HYPRE_Version(). Returns a
// transient C-string (Hypre allocates with hypre_TAlloc on first call,
// reuses on subsequent calls — no caller free). On unexpected NULL the
// fallback "unknown" sentinel is emitted. Round-1 M5 — eliminates the
// hardcoded HYPRE_VERSION_STR "3.1.0" which silently lied if brew bumped.
const char *hypre_version_runtime() {
    char *ver = nullptr;
    if (HYPRE_Version(&ver) != 0 || ver == nullptr) {
        return "unknown";
    }
    // Hypre's HYPRE_Version writes something like "HYPRE Release Version 3.1.0".
    // Aggregator cares about the dotted version triple — strip leading
    // boilerplate so the cell_summary KV stays compact + machine-parseable.
    // Search for the first digit; if found, use the suffix from there.
    for (const char *p = ver; *p; ++p) {
        if (*p >= '0' && *p <= '9') return p;
    }
    return ver;
}

// Emit cell_summary KV block (REQ-4 Scenario "cell_summary KV block schema").
// Fixed line order for aggregator parser stability.
void emit_cell_summary(const std::string &case_name,
                       int interp_type, int coarsen_type,
                       uint64_t NumY, uint64_t nnz_A,
                       double setup_wall_sec, double apply_wall_sec,
                       size_t peak_rss_bytes_val,
                       double cycle_complexity, double operator_complexity,
                       double residual_reduction_v1,
                       const char *verdict_class) {
    std::printf("CELL_SUMMARY_BEGIN\n");
    std::printf("case=%s interp_type=%d coarsen_type=%d NumY=%llu nnz_A=%llu\n",
                case_name.c_str(), interp_type, coarsen_type,
                (unsigned long long)NumY, (unsigned long long)nnz_A);
    std::printf("setup_wall_sec=%.6f apply_wall_sec=%.6f peak_rss_bytes=%zu\n",
                setup_wall_sec, apply_wall_sec, peak_rss_bytes_val);
    std::printf("cycle_complexity=%.4f operator_complexity=%.4f residual_reduction_v1=%.4f\n",
                cycle_complexity, operator_complexity, residual_reduction_v1);
    std::printf("verdict_class=%s\n", verdict_class);
    // hypre_version: runtime probe via HYPRE_Version() (round-1 M5).
    // colpack_version / shud_pin: compile-time embed via Makefile
    // -DCOLPACK_VERSION_STR + -DSHUD_PIN_SHA (round-1 M6). Both default
    // to "unknown" if the Makefile probe failed or the cpp is built
    // outside the project Makefile.
    std::printf("hypre_version=%s colpack_version=%s shud_pin=%s\n",
                hypre_version_runtime(), COLPACK_VERSION_STR, SHUD_PIN_SHA);
    std::printf("CELL_SUMMARY_END\n");
    std::fflush(stdout);
}

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --case <name> --interp-type <N> --coarsen-type <N>\n"
                 "       [--basin-root <path>] [--in <prefix>] [--n-solve <N>] [--rhs-mode {ones|random}]\n"
                 "  --case          case name (e.g. keliya)\n"
                 "  --interp-type   Hypre BoomerAMG interpolation type (e.g. 6, 8, 14)\n"
                 "  --coarsen-type  Hypre BoomerAMG coarsening type (e.g. 8, 10, 21)\n"
                 "  --basin-root    parent of Basins/<case>/ (default: ../../SHUD/Basins)\n"
                 "  --in            JBin prefix; expects <prefix>_numeric_J.bin (default: <case>)\n"
                 "  --n-solve       number of repeated solves for apply-wall median (default: 5)\n"
                 "  --rhs-mode      RHS vector mode: ones | random (default: ones)\n",
                 prog);
}

// SIGTERM trap — Slurm wall-overflow → emit AMG_WALL_OVERFLOW marker + KV
// + exit 0 (per REQ-4 Scenario "SIGTERM trap"). PR-A keliya scale this
// path is dormant; PR-B exercises it under the 8h Slurm budget.
//
// Async-signal-safety (round-1 C4): the handler MUST NOT touch printf,
// fflush, malloc, std::string, mach_task_self, or any of the std::* I/O
// helpers — none are listed in POSIX.1-2017 §2.4.3 as async-signal-safe.
// Instead, the handler only flips a volatile sig_atomic_t flag. The main
// thread polls the flag at safe points (top of solve loop iterations,
// just after Setup, and after the V-cycle-1 probe) and emits the marker
// + cell_summary there, where stdio mutexes are guaranteed to be quiescent.
struct CellState {
    std::string case_name;
    int interp_type = -1;
    int coarsen_type = -1;
    uint64_t NumY = 0;
    uint64_t nnz_A = 0;
    double setup_wall_sec = 0.0;
    double apply_wall_sec = 0.0;
    bool setup_done = false;
};
CellState g_cell;

volatile sig_atomic_t g_sigterm_pending = 0;

extern "C" void sigterm_handler(int /*signo*/) {
    // Only async-signal-safe operation: a volatile sig_atomic_t store.
    g_sigterm_pending = 1;
}

// Poll g_sigterm_pending at safe-point boundaries. If set, emit the
// AMG_WALL_OVERFLOW marker + cell_summary block (now safe — main thread
// context, no signal-handler-stack constraints) and exit 0 per REQ-4.
// emit_marker / emit_cell_summary are defined above in this namespace.
void check_sigterm_at_safe_point() {
    if (g_sigterm_pending == 0) return;
    emit_marker("AMG_WALL_OVERFLOW");
    emit_cell_summary(g_cell.case_name, g_cell.interp_type, g_cell.coarsen_type,
                      g_cell.NumY, g_cell.nnz_A,
                      g_cell.setup_wall_sec, g_cell.apply_wall_sec,
                      peak_rss_bytes(), 0.0, 0.0, 0.0,
                      "AMG_WALL_OVERFLOW");
    std::exit(0);  // normal exit OK from main thread context; destructors
                   // + atexit run; SIGTERM handler did NOT touch them.
}

}  // namespace

int main(int argc, char **argv) {
    // Install SIGTERM trap for Slurm wall-overflow (PR-B); dormant on Mac.
    std::signal(SIGTERM, sigterm_handler);

    // --- CLI parse ---
    std::string case_name;
    int interp_type = -1;
    int coarsen_type = -1;
    std::string basin_root = "../../SHUD/Basins";
    std::string in_prefix;
    int n_solve = DEFAULT_N_SOLVE;
    std::string rhs_mode = "ones";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--case" && i + 1 < argc) { case_name = argv[++i]; }
        else if (a == "--interp-type" && i + 1 < argc) { interp_type = std::atoi(argv[++i]); }
        else if (a == "--coarsen-type" && i + 1 < argc) { coarsen_type = std::atoi(argv[++i]); }
        else if (a == "--basin-root" && i + 1 < argc) { basin_root = argv[++i]; }
        else if (a == "--in" && i + 1 < argc) { in_prefix = argv[++i]; }
        else if (a == "--n-solve" && i + 1 < argc) { n_solve = std::atoi(argv[++i]); }
        else if (a == "--rhs-mode" && i + 1 < argc) { rhs_mode = argv[++i]; }
        // Accept --interp-type=N and --coarsen-type=N forms for shell-loop ergonomics
        else if (a.rfind("--interp-type=", 0) == 0) { interp_type = std::atoi(a.c_str() + 14); }
        else if (a.rfind("--coarsen-type=", 0) == 0) { coarsen_type = std::atoi(a.c_str() + 15); }
        else if (a.rfind("--n-solve=", 0) == 0) { n_solve = std::atoi(a.c_str() + 10); }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "[amg] ERROR: unknown arg '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (case_name.empty() || interp_type < 0 || coarsen_type < 0) {
        print_usage(argv[0]); return 1;
    }
    if (in_prefix.empty()) in_prefix = case_name;
    if (n_solve < 1) n_solve = 1;
    if (rhs_mode != "ones" && rhs_mode != "random") {
        std::fprintf(stderr, "[amg] ERROR: --rhs-mode must be 'ones' or 'random' (got '%s')\n", rhs_mode.c_str());
        return 1;
    }
    g_cell.case_name = case_name;
    g_cell.interp_type = interp_type;
    g_cell.coarsen_type = coarsen_type;

    std::printf("[amg] config: case=%s interp_type=%d coarsen_type=%d n_solve=%d rhs_mode=%s\n",
                case_name.c_str(), interp_type, coarsen_type, n_solve, rhs_mode.c_str());

    // --- Pre-flight: ensure numeric J binary exists ---
    // README.md §5 convention runs dump_adjacency + fd_color_jacobian once
    // outside the 4-combo loop. If the .bin is already present (non-zero
    // size), skip the shell-out; otherwise invoke the symlinked p8tune.D
    // binaries via subprocess (REQ-2 Scenario "FD-color Jacobian reuse").
    const std::string jbin_path = basin_root + "/" + case_name + "/" + in_prefix + "_numeric_J.bin";
    const std::string csc_path  = basin_root + "/" + case_name + "/" + in_prefix + "_adjacency.csc";
    if (!file_exists_nonempty(csc_path)) {
        std::printf("[amg] pre-flight: %s missing or empty; invoking dump_adjacency...\n", csc_path.c_str());
        int rc = shell_out_preflight("dump_adjacency", case_name, basin_root);
        if (rc != 0) {
            std::fprintf(stderr, "[amg] ERROR: dump_adjacency exited with rc=%d\n", rc);
            return 1;
        }
    }
    if (!file_exists_nonempty(jbin_path)) {
        std::printf("[amg] pre-flight: %s missing or empty; invoking fd_color_jacobian...\n", jbin_path.c_str());
        int rc = shell_out_preflight("fd_color_jacobian", case_name, basin_root);
        if (rc != 0) {
            std::fprintf(stderr, "[amg] ERROR: fd_color_jacobian exited with rc=%d\n", rc);
            return 1;
        }
    }

    // --- Read numeric J binary ---
    JBin j;
    if (!read_jbin(jbin_path, j)) return 1;
    std::printf("[amg] loaded numeric J: NumY=%llu total_nnz=%llu n_colors=%llu\n",
                (unsigned long long)j.NumY, (unsigned long long)j.total_nnz,
                (unsigned long long)j.n_colors);

    const HYPRE_BigInt n = static_cast<HYPRE_BigInt>(j.NumY);
    g_cell.NumY = j.NumY;
    g_cell.nnz_A = j.total_nnz;

    // --- Build M = I - γ·J in CSC, then transpose to CSR for Hypre IJMatrix ---
    // (Mirror klu_analyze_factor:188-230. γ=1.0 unit-step BDF surrogate;
    // exact γ does not matter for pattern/AMG-spike axes since hierarchy
    // and convergence scale with diagonal magnitude not absolute value.)
    std::vector<double> Ax(j.total_nnz);
    for (uint64_t p = 0; p < j.total_nnz; ++p) Ax[p] = -GAMMA * j.values[p];

    int n_diag_modified = 0;
    for (HYPRE_BigInt col = 0; col < n; ++col) {
        const int64_t s = j.col_ptr[col];
        const int64_t e = j.col_ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            if (j.row_idx[p] == col) {
                Ax[p] += 1.0;
                ++n_diag_modified;
                break;
            }
        }
    }
    std::printf("[amg] test matrix M = I - %.6f * J  (diagonal set on %d / %lld cols)\n",
                GAMMA, n_diag_modified, (long long)n);

    // CSC → CSR conversion. Each (col, row, val) tuple becomes a CSR entry
    // (row, col, val). Group by row.
    std::vector<std::vector<int64_t>> row_cols(n);     // CSR col indices per row
    std::vector<std::vector<double>>  row_vals(n);     // CSR values per row
    for (HYPRE_BigInt col = 0; col < n; ++col) {
        const int64_t s = j.col_ptr[col];
        const int64_t e = j.col_ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            const int32_t row = j.row_idx[p];
            row_cols[row].push_back(col);
            row_vals[row].push_back(Ax[p]);
        }
    }
    // Sort each row's columns ascending for deterministic Hypre IJMatrix
    // ingestion (and aligned (col,val) co-sort).
    for (HYPRE_BigInt r = 0; r < n; ++r) {
        const size_t k = row_cols[r].size();
        std::vector<std::pair<int64_t, double>> pairs(k);
        for (size_t i = 0; i < k; ++i) pairs[i] = {row_cols[r][i], row_vals[r][i]};
        std::sort(pairs.begin(), pairs.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t i = 0; i < k; ++i) {
            row_cols[r][i] = pairs[i].first;
            row_vals[r][i] = pairs[i].second;
        }
    }

    // --- MPI + HYPRE init ---
    // Hypre 3.x brew is MPI-built (open-mpi). MPI_COMM_SELF for single-process
    // spike. HYPRE_Initialize is mandatory (utilities header: "Required" tag).
    int mpi_argc = 1;
    char  mpi_arg0[] = "boomeramg_setup_solve";
    char *mpi_argv[2] = {mpi_arg0, nullptr};
    char **mpi_argv_ptr = mpi_argv;
    if (MPI_Init(&mpi_argc, &mpi_argv_ptr) != MPI_SUCCESS) {
        std::fprintf(stderr, "[amg] ERROR: MPI_Init failed\n");
        return 1;
    }
    HYPRE_Initialize();

    // --- Build Hypre IJMatrix ---
    HYPRE_IJMatrix A_ij = nullptr;
    HYPRE_ParCSRMatrix A_par = nullptr;
    HYPRE_IJVector b_ij = nullptr, x_ij = nullptr;
    HYPRE_ParVector b_par = nullptr, x_par = nullptr;
    HYPRE_Solver solver = nullptr;

    try {
        if (HYPRE_IJMatrixCreate(MPI_COMM_SELF, 0, n - 1, 0, n - 1, &A_ij) != 0) {
            std::fprintf(stderr, "[amg] ERROR: HYPRE_IJMatrixCreate failed\n");
            HYPRE_Finalize(); MPI_Finalize();
            return 1;
        }
        HYPRE_IJMatrixSetObjectType(A_ij, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A_ij);

        // Per-row insert. For single-process MPI_COMM_SELF + HYPRE_BIGINT=1,
        // rows/cols are HYPRE_BigInt = long long.
        for (HYPRE_BigInt r = 0; r < n; ++r) {
            HYPRE_Int    ncols   = static_cast<HYPRE_Int>(row_cols[r].size());
            if (ncols == 0) continue;
            std::vector<HYPRE_BigInt> cols_big(ncols);
            for (HYPRE_Int k = 0; k < ncols; ++k) cols_big[k] = row_cols[r][k];
            HYPRE_BigInt row_idx = r;
            HYPRE_IJMatrixSetValues(A_ij, /*nrows=*/1, &ncols,
                                    &row_idx, cols_big.data(),
                                    row_vals[r].data());
        }
        HYPRE_IJMatrixAssemble(A_ij);
        HYPRE_IJMatrixGetObject(A_ij, (void **)&A_par);

        // --- Build b (RHS) and x (zero initial guess) ---
        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n - 1, &b_ij);
        HYPRE_IJVectorSetObjectType(b_ij, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b_ij);

        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n - 1, &x_ij);
        HYPRE_IJVectorSetObjectType(x_ij, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x_ij);

        std::vector<HYPRE_BigInt> indices(n);
        for (HYPRE_BigInt i = 0; i < n; ++i) indices[i] = i;

        std::vector<double> b_vals(n);
        if (rhs_mode == "ones") {
            for (HYPRE_BigInt i = 0; i < n; ++i) b_vals[i] = 1.0;
        } else {
            // rhs_mode == "random"; fixed seed for determinism across runs
            std::mt19937_64 rng(0x5EEDULL);
            std::uniform_real_distribution<double> dist(-1.0, 1.0);
            for (HYPRE_BigInt i = 0; i < n; ++i) b_vals[i] = dist(rng);
        }
        std::vector<double> x_vals(n, 0.0);

        HYPRE_IJVectorSetValues(b_ij, n, indices.data(), b_vals.data());
        HYPRE_IJVectorSetValues(x_ij, n, indices.data(), x_vals.data());
        HYPRE_IJVectorAssemble(b_ij);
        HYPRE_IJVectorAssemble(x_ij);
        HYPRE_IJVectorGetObject(b_ij, (void **)&b_par);
        HYPRE_IJVectorGetObject(x_ij, (void **)&x_par);

        // |b| (L2 norm) for residual_reduction_v1 numerator.
        double b_norm_l2 = 0.0;
        for (HYPRE_BigInt i = 0; i < n; ++i) b_norm_l2 += b_vals[i] * b_vals[i];
        b_norm_l2 = std::sqrt(b_norm_l2);

        // --- Build BoomerAMG solver ---
        HYPRE_BoomerAMGCreate(&solver);
        HYPRE_BoomerAMGSetInterpType(solver, interp_type);
        HYPRE_BoomerAMGSetCoarsenType(solver, coarsen_type);
        HYPRE_BoomerAMGSetMaxIter(solver, 100);
        HYPRE_BoomerAMGSetTol(solver, 1e-8);
        HYPRE_BoomerAMGSetPrintLevel(solver, 1);
        // Enable cumulative-nnz tracking → operator_complexity proxy
        // (Hypre 3.1.0 has no direct GetOperatorComplexity getter; see
        //  README.md §6 Hypre-version-reconciliation note).
        HYPRE_BoomerAMGSetCumNnzAP(solver, 1.0);

        // --- Timed setup ---
        auto t0_setup = std::chrono::steady_clock::now();
        HYPRE_Int setup_rc = HYPRE_BoomerAMGSetup(solver, A_par, b_par, x_par);
        auto t1_setup = std::chrono::steady_clock::now();
        g_cell.setup_wall_sec = std::chrono::duration<double>(t1_setup - t0_setup).count();
        g_cell.setup_done = true;

        // SIGTERM safe point: just after Setup completes. If a Slurm wall
        // overflow arrived during the (long) Setup phase, emit marker now
        // rather than continue into Solve. Round-1 C4.
        check_sigterm_at_safe_point();

        // Probe AMG hierarchy size via internal accessor — Hypre 3.1.0
        // public API omits HYPRE_BoomerAMGGetNumLevels; the
        // hypre_ParAMGDataNumLevels macro on the solver opaque pointer
        // (cast to hypre_ParAMGData*) reads the same `num_levels` field.
        // Required for AMG_SETUP_DIVERGE trigger per spec REQ-4. Round-1 H1.
        const HYPRE_Int num_levels =
            hypre_ParAMGDataNumLevels((hypre_ParAMGData *)solver);

        // Check post-setup RSS for Mac OOM pin (PR-A 8 GiB; PR-B overrides).
        const size_t rss_after_setup = peak_rss_bytes();
        if (rss_after_setup > MAC_OOM_PIN_BYTES) {
            emit_marker("AMG_OOM");
            emit_cell_summary(case_name, interp_type, coarsen_type, j.NumY, j.total_nnz,
                              g_cell.setup_wall_sec, 0.0, rss_after_setup,
                              0.0, 0.0, 0.0, "AMG_OOM");
            HYPRE_BoomerAMGDestroy(solver);
            HYPRE_IJVectorDestroy(b_ij);
            HYPRE_IJVectorDestroy(x_ij);
            HYPRE_IJMatrixDestroy(A_ij);
            HYPRE_Finalize(); MPI_Finalize();
            return 0;
        }

        // AMG_SETUP_DIVERGE per spec REQ-4 Scenario: OR'd triggers
        //   (a) nonzero setup return
        //   (b) num_levels == 0  (hierarchy collapse)
        //   (c) setup_wall_sec > 2.0 × WALL_BUDGET_SETUP_SEC (budget overflow
        //                                                     but not yet
        //                                                     Slurm SIGTERM)
        // Round-1 H1.
        const bool setup_diverge = (setup_rc != 0)
                                || (num_levels == 0)
                                || (g_cell.setup_wall_sec > 2.0 * WALL_BUDGET_SETUP_SEC);
        if (setup_diverge) {
            std::fprintf(stderr,
                         "[amg] AMG_SETUP_DIVERGE: setup_rc=%d num_levels=%d setup_wall_sec=%.6f budget=%.6f\n",
                         (int)setup_rc, (int)num_levels,
                         g_cell.setup_wall_sec, 2.0 * WALL_BUDGET_SETUP_SEC);
            emit_marker("AMG_SETUP_DIVERGE");
            emit_cell_summary(case_name, interp_type, coarsen_type, j.NumY, j.total_nnz,
                              g_cell.setup_wall_sec, 0.0, rss_after_setup,
                              0.0, 0.0, 0.0, "AMG_SETUP_DIVERGE");
            HYPRE_BoomerAMGDestroy(solver);
            HYPRE_IJVectorDestroy(b_ij);
            HYPRE_IJVectorDestroy(x_ij);
            HYPRE_IJMatrixDestroy(A_ij);
            HYPRE_Finalize(); MPI_Finalize();
            return 0;
        }

        // --- V-cycle-1 probe for residual_reduction_v1 (round-1 M1) ---
        // The main timed solve loop uses MaxIter=100 + Tol=1e-8 to measure
        // apply_wall_sec at convergence. Calling
        // GetFinalRelativeResidualNorm AFTER that returns ~1e-8 (post-
        // convergence ratio), so 1.0/r is ~1e8 — meaningless as the
        // 1-step reduction ratio the spec REQ-4 AMG_SOLVE_DIVERGE Scenario
        // calls for (< 2.0 threshold). Run ONE V-cycle here in isolation
        // (MaxIter=1, Tol=1e-300 so the convergence check never fires and
        // x_par actually receives the V-cycle update) and compute the true
        // residual norm via Hypre vector ops — GetFinalRelativeResidualNorm
        // is unreliable for this case (returns 0 when Hypre's internal
        // residual computation is skipped, which happens at MaxIter=1).
        HYPRE_BoomerAMGSetMaxIter(solver, 1);
        HYPRE_BoomerAMGSetTol(solver, 1e-300);
        // Reset x_par = 0 via SetValues + Assemble only (no Initialize
        // re-call on an assembled vector — round-1 C1).
        HYPRE_IJVectorSetValues(x_ij, n, indices.data(), x_vals.data());
        HYPRE_IJVectorAssemble(x_ij);
        HYPRE_IJVectorGetObject(x_ij, (void **)&x_par);
        HYPRE_Int v1_rc = HYPRE_BoomerAMGSolve(solver, A_par, b_par, x_par);
        // v1_rc==HYPRE_ERROR_CONV(256) is EXPECTED here — we deliberately
        // set Tol=1e-300 so the convergence check never passes; the V-cycle
        // still ran + x_par updated. Treat the probe as successful so long
        // as the rc is 0 or pure HYPRE_ERROR_CONV (no other bits set).
        const bool v1_ok = (v1_rc == 0) || (v1_rc == HYPRE_ERROR_CONV);

        // Manually compute residual_reduction_v1 = ||b|| / ||b - A*x_v1||.
        // We need a scratch ParVector for r = b - A*x. Allocate via IJ so it
        // owns its own storage; destroy after probing.
        HYPRE_IJVector r_ij = nullptr;
        HYPRE_ParVector r_par = nullptr;
        HYPRE_IJVectorCreate(MPI_COMM_SELF, 0, n - 1, &r_ij);
        HYPRE_IJVectorSetObjectType(r_ij, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(r_ij);
        // Initialize r := 0 via SetValues + Assemble.
        std::vector<double> r_zeros(n, 0.0);
        HYPRE_IJVectorSetValues(r_ij, n, indices.data(), r_zeros.data());
        HYPRE_IJVectorAssemble(r_ij);
        HYPRE_IJVectorGetObject(r_ij, (void **)&r_par);

        // r = b
        HYPRE_ParVectorCopy(b_par, r_par);
        // r = -A*x + 1*r = b - A*x
        HYPRE_ParCSRMatrixMatvec(-1.0, A_par, x_par, 1.0, r_par);

        // ||b||_2  and  ||r||_2
        HYPRE_Real b_dot = 0.0, r_dot = 0.0;
        HYPRE_ParVectorInnerProd(b_par, b_par, &b_dot);
        HYPRE_ParVectorInnerProd(r_par, r_par, &r_dot);
        const double b_norm = std::sqrt((double)b_dot);
        const double r_norm = std::sqrt((double)r_dot);

        // V-cycle-1 reduction ratio: well-conditioned matrices give 5-50;
        // very stiff preconditioners on tiny matrices may yield much larger
        // values (1 V-cycle nukes residual to machine zero). Both are valid
        // PASS cases — only values < 2.0 trigger AMG_SOLVE_DIVERGE.
        double residual_reduction_v1 = 0.0;
        if (v1_ok && r_norm > 0.0 && b_norm > 0.0) {
            residual_reduction_v1 = b_norm / r_norm;
        } else if (v1_ok && r_norm == 0.0) {
            // Residual exact zero — converged in 1 V-cycle (machine precision).
            // Cap at 1e12 so the KV stays a printable double, well above the
            // < 2.0 SOLVE_DIVERGE threshold.
            residual_reduction_v1 = 1e12;
        }
        // (else: v1 probe had a non-CONV Hypre error — leave
        //  residual_reduction_v1 = 0.0; the SOLVE_DIVERGE trigger below
        //  will catch it via the < 2.0 check.)

        HYPRE_IJVectorDestroy(r_ij);

        // Restore convergence settings for the main timed loop.
        HYPRE_BoomerAMGSetMaxIter(solver, 100);
        HYPRE_BoomerAMGSetTol(solver, 1e-8);

        // Clear Hypre's accumulated error flag. The probe deliberately ran
        // with Tol=1e-300 + MaxIter=1, so Hypre raised HYPRE_ERROR_CONV
        // (256) into the global flag. Without clearing, every subsequent
        // Hypre call (Solve, GetCumNnzAP, HYPRE_Version inside our cell-
        // summary emit) inherits the flag and returns non-zero, which the
        // main timed loop would mis-interpret as a SOLVE_DIVERGE.
        HYPRE_ClearAllErrors();

        // SIGTERM safe point: after the V-cycle-1 probe. Round-1 C4.
        check_sigterm_at_safe_point();

        // --- Timed solve loop (average over n_solve) ---
        // Per Hypre 3.1.0 IJ docs, HYPRE_IJVectorInitialize is for
        // first-use only; re-calling on an assembled vector inside the
        // loop is UB and was making apply_wall_sec from iter 2+ unreliable
        // — round-1 C1. Reset x via SetValues + Assemble only.
        double apply_wall_sum = 0.0;
        HYPRE_Real final_res_rel = 0.0;
        HYPRE_Int solve_rc_last = 0;
        for (int iter = 0; iter < n_solve; ++iter) {
            // SIGTERM safe point: between solve iterations. Round-1 C4.
            check_sigterm_at_safe_point();

            // Reset x = 0 each iteration via SetValues+Assemble ONLY.
            HYPRE_IJVectorSetValues(x_ij, n, indices.data(), x_vals.data());
            HYPRE_IJVectorAssemble(x_ij);
            HYPRE_IJVectorGetObject(x_ij, (void **)&x_par);

            auto t0_apply = std::chrono::steady_clock::now();
            HYPRE_Int rc = HYPRE_BoomerAMGSolve(solver, A_par, b_par, x_par);
            auto t1_apply = std::chrono::steady_clock::now();
            apply_wall_sum += std::chrono::duration<double>(t1_apply - t0_apply).count();
            solve_rc_last = rc;
            if (rc != 0) break;
        }
        g_cell.apply_wall_sec = apply_wall_sum / std::max(1, n_solve);

        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(solver, &final_res_rel);

        // AMG_SOLVE_DIVERGE per spec REQ-4 Scenario: OR'd triggers
        //   (a) nonzero solve return
        //   (b) final_res_rel > 1.0  (iteration diverged from initial guess)
        //   (c) residual_reduction_v1 < 2.0  (V-cycle stagnating below
        //       canonical convergence threshold per design R3)
        // Round-1 H2.
        const size_t rss_final = peak_rss_bytes();
        const bool solve_diverge = (solve_rc_last != 0)
                                 || (final_res_rel > 1.0)
                                 || (residual_reduction_v1 < 2.0);

        // Operator complexity from CumNnzAP / nnz_A (Hypre canonical).
        HYPRE_Real cum_nnz_AP = 0.0;
        HYPRE_BoomerAMGGetCumNnzAP(solver, &cum_nnz_AP);
        double operator_complexity = (j.total_nnz > 0)
            ? (cum_nnz_AP / static_cast<double>(j.total_nnz))
            : 0.0;
        // Cycle complexity: Hypre 3.1.0 omits the direct getter
        // (HYPRE_BoomerAMGGetCycleComplexity is absent from public headers).
        // Standard V-cycle estimate: 2 × operator_complexity (1 pre-smoothing
        // + 1 post-smoothing pass per level, weight=1). KNOWN LIMITATION
        // (round-1 H3): for W-cycles or aggressive coarsening this
        // systematically under-counts; PR-C aggregator's Axis 4 (cycle
        // complexity < 1.5) mechanically tracks Axis 5 (operator complexity
        // < 2.0) here at 2× linkage. ADR-0007 §Discussion must disclose
        // this as a definitional caveat — Axis 4 and Axis 5 are NOT
        // independent diagnostics under the Hypre-3.1.0-API constraint.
        // See README.md §6 and openspec/changes/p8tune-amg-spike/design.md D2.
        double cycle_complexity = 2.0 * operator_complexity;

        const char *verdict = solve_diverge ? "AMG_SOLVE_DIVERGE" : "PASS";
        if (solve_diverge) {
            std::fprintf(stderr,
                         "[amg] AMG_SOLVE_DIVERGE: solve_rc=%d final_res_rel=%.6e residual_reduction_v1=%.6f\n",
                         (int)solve_rc_last, (double)final_res_rel,
                         residual_reduction_v1);
            emit_marker("AMG_SOLVE_DIVERGE");
        }

        emit_cell_summary(case_name, interp_type, coarsen_type, j.NumY, j.total_nnz,
                          g_cell.setup_wall_sec, g_cell.apply_wall_sec, rss_final,
                          cycle_complexity, operator_complexity,
                          residual_reduction_v1, verdict);

        // --- Cleanup (always before HYPRE_Finalize / MPI_Finalize) ---
        HYPRE_BoomerAMGDestroy(solver);
        HYPRE_IJVectorDestroy(b_ij);
        HYPRE_IJVectorDestroy(x_ij);
        HYPRE_IJMatrixDestroy(A_ij);
    }
    catch (const std::bad_alloc &) {
        // Heap exhaustion mid-construction. Emit AMG_OOM marker + KV.
        emit_marker("AMG_OOM");
        emit_cell_summary(case_name, interp_type, coarsen_type, j.NumY, j.total_nnz,
                          g_cell.setup_wall_sec, g_cell.apply_wall_sec,
                          peak_rss_bytes(), 0.0, 0.0, 0.0, "AMG_OOM");
        // Best-effort cleanup; may itself throw, but we're exiting.
        if (solver)  HYPRE_BoomerAMGDestroy(solver);
        if (b_ij)    HYPRE_IJVectorDestroy(b_ij);
        if (x_ij)    HYPRE_IJVectorDestroy(x_ij);
        if (A_ij)    HYPRE_IJMatrixDestroy(A_ij);
        HYPRE_Finalize(); MPI_Finalize();
        return 0;
    }

    HYPRE_Finalize();
    MPI_Finalize();
    return 0;
}
