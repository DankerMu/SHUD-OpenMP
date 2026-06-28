/* tools/p8tune.D/klu_analyze_factor.cpp
 *
 * P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
 * `p8tune-klu-spike` spec §REQ-5 "Three-axis hard verdict SHALL apply
 * (fill + RSS + wall) for per-case GO decision".
 *
 * Reads `<case>_numeric_J.bin` produced by fd_color_jacobian, drives
 * SuiteSparse KLU through:
 *
 *   1. klu_defaults(&common); set ordering + BTF per CLI.
 *   2. klu_analyze(n, Ap, Ai, &common)              — symbolic factor
 *      → reports Symbolic->lnz, unz, nzoff, nblocks, maxblock,
 *        symmetry, est_flops.
 *   3. klu_factor(Ap, Ai, Ax, Symbolic, &common)    — numeric factor
 *      → reports Numeric->lnz, unz, max_lnz_block, wall_sec.
 *
 * Emits per-cell summary to stdout (aggregator-parseable).
 *
 * Pre-flight: compare estimated peak RSS vs CN_NODE_RAM_BYTES from
 * cn_node_ram.h. If exceeded OR if klu_factor returns NULL with
 * common.status == KLU_OUT_OF_MEMORY, emit `KLU_OOM_DETECTED` per
 * REQ-5 Scenario "OOM-as-data-point" and exit 0 (NOT non-zero).
 *
 * Usage:
 *   ./klu_analyze_factor --case <name> --ordering {natural|amd|colamd} --btf {0|1}
 *                        [--basin-root <path>] [--in <prefix>]
 *
 * Refs:
 *   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-5
 *   SuiteSparse KLU klu.h (e.g. /opt/homebrew/include/suitesparse/klu.h)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "klu.h"

#ifdef __APPLE__
#include <mach/mach.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#include "cn_node_ram.h"

namespace {

constexpr uint32_t JBIN_MAGIC = 0x4A4E444Eu;  // 'N''D''N''J'

struct JBin {
    uint64_t NumY = 0, total_nnz = 0, n_colors = 0;
    std::vector<int64_t> col_ptr;
    std::vector<int32_t> row_idx;
    std::vector<double> values;
};

bool read_jbin(const std::string &path, JBin &j) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[klu] ERROR: cannot open %s\n", path.c_str());
        return false;
    }
    uint32_t magic = 0, version = 0;
    std::fread(&magic, sizeof(magic), 1, fp);
    if (magic != JBIN_MAGIC) {
        std::fprintf(stderr, "[klu] ERROR: bad magic 0x%08x in %s (expected 0x%08x NDNJ)\n",
                     magic, path.c_str(), JBIN_MAGIC);
        std::fclose(fp); return false;
    }
    std::fread(&version, sizeof(version), 1, fp);
    if (version != 1u) {
        std::fprintf(stderr, "[klu] ERROR: unsupported version %u\n", version);
        std::fclose(fp); return false;
    }
    std::fread(&j.NumY, sizeof(uint64_t), 1, fp);
    std::fread(&j.total_nnz, sizeof(uint64_t), 1, fp);
    std::fread(&j.n_colors, sizeof(uint64_t), 1, fp);
    j.col_ptr.resize(j.NumY + 1);
    std::fread(j.col_ptr.data(), sizeof(int64_t), j.NumY + 1, fp);
    j.row_idx.resize(j.total_nnz);
    std::fread(j.row_idx.data(), sizeof(int32_t), j.total_nnz, fp);
    j.values.resize(j.total_nnz);
    std::fread(j.values.data(), sizeof(double), j.total_nnz, fp);
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

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --case <name> --ordering {natural|amd|colamd} --btf {0|1} [--basin-root <path>] [--in <prefix>]\n"
                 "  --case        case name (e.g. keliya)\n"
                 "  --ordering    KLU ordering: natural | amd | colamd\n"
                 "  --btf         BTF preordering: 0 (off) | 1 (on)\n"
                 "  --basin-root  parent of Basins/<case>/  (default: ../../SHUD/Basins)\n"
                 "  --in          input prefix; expects <prefix>_numeric_J.bin  (default: <case>)\n",
                 prog);
}

}  // namespace

int main(int argc, char **argv) {
    std::string case_name;
    std::string ordering_name;
    int btf_flag = -1;
    std::string basin_root = "../../SHUD/Basins";
    std::string in_prefix;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--case" && i + 1 < argc) { case_name = argv[++i]; }
        else if (a == "--ordering" && i + 1 < argc) { ordering_name = argv[++i]; }
        else if (a == "--btf" && i + 1 < argc) { btf_flag = std::atoi(argv[++i]); }
        else if (a == "--basin-root" && i + 1 < argc) { basin_root = argv[++i]; }
        else if (a == "--in" && i + 1 < argc) { in_prefix = argv[++i]; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "[klu] ERROR: unknown arg '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (case_name.empty() || ordering_name.empty() || (btf_flag != 0 && btf_flag != 1)) {
        print_usage(argv[0]); return 1;
    }
    if (in_prefix.empty()) in_prefix = case_name;

    int ordering_id = -1;
    if (ordering_name == "amd") ordering_id = 0;
    else if (ordering_name == "colamd") ordering_id = 1;
    else if (ordering_name == "natural") ordering_id = 2;  // user P=I, Q=I via klu_analyze_given
    else {
        std::fprintf(stderr, "[klu] ERROR: unknown ordering '%s' (use natural|amd|colamd)\n", ordering_name.c_str());
        return 1;
    }

    // --- 1. Read numeric J binary ---
    // klu_analyze_factor runs from the spike-tool launch dir; the J binary
    // produced by fd_color_jacobian lives in basin_dir/<prefix>_numeric_J.bin
    // (fd_color_jacobian chdir'd into basin_dir before writing). Honor the
    // same convention here so the per-cell driver doesn't need to copy files.
    std::string jbin_path = basin_root + "/" + case_name + "/" + in_prefix + "_numeric_J.bin";
    JBin j;
    if (!read_jbin(jbin_path, j)) return 1;
    std::printf("[klu] loaded numeric J: NumY=%llu total_nnz=%llu n_colors=%llu\n",
                (unsigned long long)j.NumY, (unsigned long long)j.total_nnz,
                (unsigned long long)j.n_colors);

    // --- 2. klu_defaults + ordering/btf wiring ---
    klu_common common;
    klu_defaults(&common);
    common.btf = btf_flag;
    if (ordering_id != 2) {
        common.ordering = ordering_id;  // 0=AMD, 1=COLAMD
    }

    // KLU expects CSC int arrays (Ap, Ai). Our col_ptr is int64_t; KLU has
    // both 32-bit (klu_*) and 64-bit (klu_l_*) APIs. The default klu_*
    // expects 32-bit int Ap/Ai. Convert.
    std::vector<int> Ap(j.NumY + 1);
    std::vector<int> Ai(j.total_nnz);
    for (int64_t k = 0; k <= (int64_t)j.NumY; ++k) Ap[k] = static_cast<int>(j.col_ptr[k]);
    for (int64_t k = 0; k < (int64_t)j.total_nnz; ++k) Ai[k] = j.row_idx[k];

    const int n = static_cast<int>(j.NumY);

    // Build CVODE-BDF-equivalent test matrix M = I - γ·J, mirroring what
    // SUNLinSol_KLU would factor inside CVODE's Newton iteration. γ = 1/h
    // dominates the diagonal so M is non-singular even where J has zero
    // diagonals (the spike's FD-probed J has ~40% zero diagonals at the IC
    // because many state derivatives are flux-coupled with no self-feedback
    // at t=0, e.g. dry-bed surface cells). γ = 1.0 is a unit-step BDF
    // surrogate; the EXACT γ doesn't matter for fill_ratio / numeric_wall
    // axes — pattern of nonzeros + RSS scale linearly with the test matrix
    // size, not with the diagonal magnitude.
    //
    // This is required by REQ-5 — the wall axis must measure a SUCCESSFUL
    // numeric factor wall; klu_factor returning SINGULAR on the raw J
    // would mean we have no wall data point to compare against SPGMR.
    constexpr double GAMMA = 1.0;
    std::vector<double> Ax(j.total_nnz);
    for (int64_t p = 0; p < (int64_t)j.total_nnz; ++p) {
        Ax[p] = -GAMMA * j.values[p];
    }
    // Add 1 to diagonal entries (assumes diagonal exists at every column;
    // dump_adjacency guarantees this via the diagonal-always pass).
    int n_diag_modified = 0;
    for (int col = 0; col < n; ++col) {
        const int s = Ap[col];
        const int e = Ap[col + 1];
        for (int p = s; p < e; ++p) {
            if (Ai[p] == col) {
                Ax[p] += 1.0;
                ++n_diag_modified;
                break;
            }
        }
    }
    std::printf("[klu] test matrix M = I - %.6f * J  (diagonal set on %d / %d cols)\n",
                GAMMA, n_diag_modified, n);
    if (n_diag_modified < n) {
        std::fprintf(stderr, "[klu] WARN: %d cols missing diagonal entry in CSC; KLU may still succeed\n",
                     n - n_diag_modified);
    }

    // --- 3. Pre-flight RSS estimate ---
    // Rough lower bound: nnz(A) × 16B (one int row_idx + one double value) plus
    // SuiteSparse working set (estimated 4× nnz(A) when AMD permutation arrays added).
    // CN_NODE_RAM_BYTES comes from cn_node_ram.h.
    const size_t est_pre_factor_bytes = static_cast<size_t>(j.total_nnz) * 64ULL + n * 64ULL;
    std::printf("[klu] pre-flight: A nnz=%llu est_bytes=%zu cn_ram=%zu\n",
                (unsigned long long)j.total_nnz, est_pre_factor_bytes, (size_t)CN_NODE_RAM_BYTES);
    if (est_pre_factor_bytes > CN_NODE_RAM_BYTES) {
        std::printf("KLU_OOM_DETECTED case=%s ordering=%s btf=%d peak_rss_bytes=%zu reason=preflight_estimate\n",
                    case_name.c_str(), ordering_name.c_str(), btf_flag, est_pre_factor_bytes);
        return 0;
    }

    // --- 4. klu_analyze (symbolic) ---
    auto t0_sym = std::chrono::steady_clock::now();
    klu_symbolic *Symbolic = nullptr;
    if (ordering_id == 2) {
        // natural: user-supplied identity permutations
        std::vector<int> P(n), Q(n);
        for (int k = 0; k < n; ++k) { P[k] = k; Q[k] = k; }
        Symbolic = klu_analyze_given(n, Ap.data(), Ai.data(), P.data(), Q.data(), &common);
    } else {
        Symbolic = klu_analyze(n, Ap.data(), Ai.data(), &common);
    }
    auto t1_sym = std::chrono::steady_clock::now();
    if (!Symbolic) {
        std::fprintf(stderr, "[klu] ERROR: klu_analyze failed (common.status=%d)\n", common.status);
        return 1;
    }
    const double sym_wall_s =
        std::chrono::duration<double>(t1_sym - t0_sym).count();

    // --- 5. klu_factor (numeric) ---
    // Uses the BDF-equivalent test matrix M = I - γ·J (Ax) constructed above.
    auto t0_num = std::chrono::steady_clock::now();
    klu_numeric *Numeric = klu_factor(Ap.data(), Ai.data(), Ax.data(), Symbolic, &common);
    auto t1_num = std::chrono::steady_clock::now();
    const double num_wall_s =
        std::chrono::duration<double>(t1_num - t0_num).count();

    const size_t rss_post = peak_rss_bytes();

    if (!Numeric) {
        if (common.status == KLU_OUT_OF_MEMORY) {
            std::printf("KLU_OOM_DETECTED case=%s ordering=%s btf=%d peak_rss_bytes=%zu reason=klu_factor_OOM\n",
                        case_name.c_str(), ordering_name.c_str(), btf_flag, rss_post);
            klu_free_symbolic(&Symbolic, &common);
            return 0;
        }
        std::fprintf(stderr, "[klu] ERROR: klu_factor failed (common.status=%d)\n", common.status);
        klu_free_symbolic(&Symbolic, &common);
        return 1;
    }

    if (rss_post > CN_NODE_RAM_BYTES) {
        std::printf("KLU_OOM_DETECTED case=%s ordering=%s btf=%d peak_rss_bytes=%zu reason=post_factor_rss_exceeds_cn_ram\n",
                    case_name.c_str(), ordering_name.c_str(), btf_flag, rss_post);
        klu_free_numeric(&Numeric, &common);
        klu_free_symbolic(&Symbolic, &common);
        return 0;
    }

    // --- 6. Emit summary ---
    const double Anz = static_cast<double>(j.total_nnz);
    const double fill_ratio = (Numeric->lnz + Numeric->unz) / std::max(Anz, 1.0);

    std::printf("[klu] cell_summary\n");
    std::printf("  case=%s ordering=%s btf=%d\n", case_name.c_str(), ordering_name.c_str(), btf_flag);
    std::printf("  NumY=%d Anz=%.0f\n", n, Anz);
    std::printf("  symbolic_lnz_est=%.0f symbolic_unz_est=%.0f est_flops=%.3e\n",
                Symbolic->lnz, Symbolic->unz, Symbolic->est_flops);
    std::printf("  nblocks=%d maxblock=%d nzoff=%d symmetry=%.6f\n",
                Symbolic->nblocks, Symbolic->maxblock, Symbolic->nzoff, Symbolic->symmetry);
    std::printf("  numeric_lnz=%d numeric_unz=%d max_lnz_block=%d max_unz_block=%d\n",
                Numeric->lnz, Numeric->unz, Numeric->max_lnz_block, Numeric->max_unz_block);
    std::printf("  fill_ratio=%.4f (numeric_lnz+unz)/Anz\n", fill_ratio);
    std::printf("  symbolic_wall_sec=%.6f numeric_factor_wall_sec=%.6f\n", sym_wall_s, num_wall_s);
    std::printf("  peak_rss_bytes=%zu (CN_NODE_RAM_BYTES=%zu)\n", rss_post, (size_t)CN_NODE_RAM_BYTES);

    klu_free_numeric(&Numeric, &common);
    klu_free_symbolic(&Symbolic, &common);
    return 0;
}
