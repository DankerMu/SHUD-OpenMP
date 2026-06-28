/* tools/p8tune.D/fd_color_jacobian.cpp
 *
 * P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
 * `p8tune-klu-spike` spec §REQ-2 "Spike tool SHALL acquire Jacobian
 * sparsity via FD colored Jacobian using existing rhs_core dispatcher":
 *
 *   1. Read adjacency CSC (output of dump_adjacency).
 *   2. Build CSR-style row index array indexed by COLUMN (for ColPack:
 *      bipartite Jacobian graph where columns are the right vertices).
 *   3. ColPack BipartiteGraphPartialColoringInterface::PartialDistanceTwoColoring
 *      with "SMALLEST_LAST" ordering + "COLUMN_PARTIAL_DISTANCE_TWO" =
 *      Welsh-Powell-equivalent column coloring. Report chromatic number χ.
 *   4. Init Model_Data per shud.cpp:118-121 to obtain a meaningful Y baseline.
 *   5. For each color c: build seed vector v_c[i] = (color[i]==c ? 1 : 0).
 *      Probe DY_plus  = rhs_core(Y + ε·v_c, t0, Serial) and DY_base = rhs_core(Y, t0, Serial).
 *      Compute J[i,k] = (DY_plus[i] - DY_base[i]) / ε  for rows i where
 *      adjacency CSC has (i,k) and color[k]==c.
 *      ε = sqrt(DBL_EPSILON) * (|Y[k]| + 1).
 *   6. Emit numeric J binary `<case>_numeric_J.bin`:
 *        uint32_t  magic = 0x4A4E444E ('N' 'D' 'N' 'J')  little-endian
 *        uint32_t  version = 1
 *        uint64_t  NumY, total_nnz, n_colors
 *        int64_t   col_ptr[NumY + 1]
 *        int32_t   row_idx[total_nnz]
 *        double    values[total_nnz]    (numeric J entries in CSC order)
 *
 *   Determinism: ColPack's PartialDistanceTwoColoring("SMALLEST_LAST", ...)
 *   has deterministic tie-breaking; ε formula is purely arithmetic; row
 *   ordering is inherited from sorted CSC. Re-running yields bytewise
 *   identical output.
 *
 * Usage:
 *   ./fd_color_jacobian --case <name> [--basin-root <path>] [--in <prefix>]
 *                       [--out <prefix>] [--report-chi-only]
 *
 * --report-chi-only mode: load CSC, run coloring, emit χ to stdout only;
 * skips Model_Data init + FD probe + binary write. Used by Mac smoke
 * (task 1.15 (c)) to gate "χ ≤ 30 for keliya" before the heavy FD work.
 *
 * Refs:
 *   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-2
 *   ColPack Examples/SampleDrivers/Basic/Generate_seed_matrix_for_Jacobian.cpp
 */

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <omp.h>

#include "ColPackHeaders.h"
#include "CommandIn.hpp"
#include "IO.hpp"
#include "Model_Data.hpp"

// `SRC_MEM_ADOLC` is a macro (defined to 1) in ColPack/Definitions.h —
// NOT a ColPack:: enum. We reference it bare.
using ColPack::BipartiteGraphPartialColoringInterface;

namespace {

constexpr uint32_t CSC_MAGIC = 0x434A4441u;   // 'A''D''J''C'
constexpr uint32_t JBIN_MAGIC = 0x4A4E444Eu;  // 'N''D''N''J'

struct CSC {
    uint64_t NumEle = 0, NumRiv = 0, NumLake = 0, NumY = 0;
    uint64_t total_nnz = 0;
    uint64_t per_block_nnz[25];
    std::vector<int64_t> col_ptr;
    std::vector<int32_t> row_idx;
};

bool read_csc(const std::string &path, CSC &out) {
    FILE *fp = std::fopen(path.c_str(), "rb");
    if (!fp) {
        std::fprintf(stderr, "[fd_color] ERROR: cannot open %s\n", path.c_str());
        return false;
    }
    uint32_t magic = 0, version = 0;
    if (std::fread(&magic, sizeof(magic), 1, fp) != 1 || magic != CSC_MAGIC) {
        std::fprintf(stderr, "[fd_color] ERROR: bad magic 0x%08x in %s\n", magic, path.c_str());
        std::fclose(fp); return false;
    }
    if (std::fread(&version, sizeof(version), 1, fp) != 1 || version != 1u) {
        std::fprintf(stderr, "[fd_color] ERROR: unsupported version %u\n", version);
        std::fclose(fp); return false;
    }
    std::fread(&out.NumEle, sizeof(uint64_t), 1, fp);
    std::fread(&out.NumRiv, sizeof(uint64_t), 1, fp);
    std::fread(&out.NumLake, sizeof(uint64_t), 1, fp);
    std::fread(&out.NumY, sizeof(uint64_t), 1, fp);
    std::fread(&out.total_nnz, sizeof(uint64_t), 1, fp);
    std::fread(out.per_block_nnz, sizeof(uint64_t), 25, fp);
    uint64_t csc_col_count = 0;
    std::fread(&csc_col_count, sizeof(uint64_t), 1, fp);
    if (csc_col_count != out.NumY) {
        std::fprintf(stderr, "[fd_color] ERROR: csc_col_count=%llu mismatches NumY=%llu\n",
                     (unsigned long long)csc_col_count, (unsigned long long)out.NumY);
        std::fclose(fp); return false;
    }
    out.col_ptr.resize(out.NumY + 1);
    std::fread(out.col_ptr.data(), sizeof(int64_t), out.NumY + 1, fp);
    out.row_idx.resize(out.total_nnz);
    if (out.total_nnz > 0) {
        std::fread(out.row_idx.data(), sizeof(int32_t), out.total_nnz, fp);
    }
    std::fclose(fp);
    return true;
}

// Convert CSC (column-major) → row-major adjacency suitable for ColPack
// BipartiteGraphPartialColoringInterface (SRC_MEM_ADOLC layout: per row,
// uip2[i][0] = nnz count, uip2[i][1..] = column indices).
unsigned int **csc_to_uip2(const CSC &csc, int *row_count_out, int *col_count_out) {
    const int64_t NumY = static_cast<int64_t>(csc.NumY);
    std::vector<std::vector<int>> rows(NumY);
    for (int64_t col = 0; col < NumY; ++col) {
        const int64_t s = csc.col_ptr[col];
        const int64_t e = csc.col_ptr[col + 1];
        for (int64_t p = s; p < e; ++p) {
            int32_t r = csc.row_idx[p];
            rows[r].push_back(static_cast<int>(col));
        }
    }
    unsigned int **uip2 = new unsigned int *[NumY];
    for (int64_t i = 0; i < NumY; ++i) {
        const int cnt = static_cast<int>(rows[i].size());
        uip2[i] = new unsigned int[cnt + 1];
        uip2[i][0] = static_cast<unsigned int>(cnt);
        for (int k = 0; k < cnt; ++k) {
            uip2[i][k + 1] = static_cast<unsigned int>(rows[i][k]);
        }
    }
    *row_count_out = static_cast<int>(NumY);
    *col_count_out = static_cast<int>(NumY);
    return uip2;
}

void free_uip2(unsigned int **uip2, int row_count) {
    for (int i = 0; i < row_count; ++i) delete[] uip2[i];
    delete[] uip2;
}

// Compute per-column color via ColPack. Returns chromatic number χ.
int run_column_coloring(const CSC &csc, std::vector<int> &color) {
    int row_count = 0, col_count = 0;
    unsigned int **uip2 = csc_to_uip2(csc, &row_count, &col_count);

    BipartiteGraphPartialColoringInterface *g =
        new BipartiteGraphPartialColoringInterface(SRC_MEM_ADOLC, uip2, row_count, col_count);
    g->PartialDistanceTwoColoring("SMALLEST_LAST", "COLUMN_PARTIAL_DISTANCE_TWO");

    std::vector<int> right_colors;
    g->GetRightVertexColors(right_colors);

    color.assign(csc.NumY, -1);
    for (int64_t k = 0; k < (int64_t)right_colors.size() && k < (int64_t)csc.NumY; ++k) {
        color[k] = right_colors[k];
    }

    int chi = g->GetRightVertexColorCount();

    delete g;
    free_uip2(uip2, row_count);
    return chi;
}

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --case <name> [--basin-root <path>] [--in <prefix>] [--out <prefix>] [--report-chi-only]\n"
                 "  --case             case name (e.g. keliya)\n"
                 "  --basin-root       parent of Basins/<case>/  (default: ../../SHUD/Basins)\n"
                 "  --in               CSC input prefix; expects <prefix>_adjacency.csc  (default: <case>)\n"
                 "  --out              numeric J output prefix; emits <prefix>_numeric_J.bin (default: <case>)\n"
                 "  --report-chi-only  load CSC + run coloring + emit χ to stdout; skip FD probe + J binary write\n",
                 prog);
}

}  // namespace

int main(int argc, char **argv) {
    // Pin OpenMP to 1 thread. Required for FD probe determinism:
    //   1. ColPack's parallel coloring (SMPGCColoring) has non-deterministic
    //      tie-breaking at OMP_NUM_THREADS > 1 — produces different χ across runs.
    //   2. SHUD's rhs_core() under ExecPolicy::Serial doesn't spawn threads,
    //      but it shares the OpenMP runtime; pinning to 1 here forces a
    //      reproducible single-threaded environment for the entire spike tool.
    // Per spec REQ-2 Scenario "FD probe determinism" + REQ-5 Scenario
    // "Numeric factor determinism".
    omp_set_num_threads(1);

    std::string case_name;
    std::string basin_root = "../../SHUD/Basins";
    std::string in_prefix;
    std::string out_prefix;
    bool chi_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--case" && i + 1 < argc) { case_name = argv[++i]; }
        else if (a == "--basin-root" && i + 1 < argc) { basin_root = argv[++i]; }
        else if (a == "--in" && i + 1 < argc) { in_prefix = argv[++i]; }
        else if (a == "--out" && i + 1 < argc) { out_prefix = argv[++i]; }
        else if (a == "--report-chi-only") { chi_only = true; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); return 0; }
        else { std::fprintf(stderr, "[fd_color] ERROR: unknown arg '%s'\n", a.c_str()); print_usage(argv[0]); return 1; }
    }
    if (case_name.empty()) { print_usage(argv[0]); return 1; }
    if (in_prefix.empty()) in_prefix = case_name;
    if (out_prefix.empty()) out_prefix = case_name;

    const std::string csc_path = in_prefix + "_adjacency.csc";

    // --- 1. Read CSC adjacency ---
    CSC csc;
    if (!read_csc(csc_path, csc)) return 1;
    std::printf("[fd_color] loaded CSC: NumY=%llu total_nnz=%llu\n",
                (unsigned long long)csc.NumY, (unsigned long long)csc.total_nnz);

    // --- 2. ColPack column coloring ---
    std::vector<int> color;
    const int chi = run_column_coloring(csc, color);
    std::printf("[fd_color] chromatic_number=%d  (NumY=%llu, density=%.5f)\n",
                chi, (unsigned long long)csc.NumY,
                static_cast<double>(csc.total_nnz) /
                    (static_cast<double>(csc.NumY) * static_cast<double>(csc.NumY)));

    if (chi_only) {
        std::printf("[fd_color] --report-chi-only mode: exiting after chromatic-number report.\n");
        return 0;
    }

    // --- 3. Model_Data init (mirror SHUD/src/Model/shud.cpp:118-121) ---
    std::string basin_dir = basin_root + "/" + case_name;
    if (chdir(basin_dir.c_str()) != 0) {
        std::fprintf(stderr, "[fd_color] ERROR: cannot chdir to '%s'\n", basin_dir.c_str());
        return 1;
    }
    CommandIn CLI;
    {
        std::vector<char> a0(argv[0], argv[0] + std::strlen(argv[0]) + 1);
        std::vector<char> a1(case_name.begin(), case_name.end());
        a1.push_back('\0');
        char *fake_argv[3] = {a0.data(), a1.data(), nullptr};
        extern int optind;
        optind = 1;
        CLI.parse(2, fake_argv);
    }
    FileIn *fin = new FileIn;
    FileOut *fout = new FileOut;
    CLI.setFileIO(fin, fout);

    Model_Data *MD = new Model_Data(fin, fout);
    MD->loadinput();
    MD->initialize();
    // Mirror SHUD/src/Model/shud.cpp:172-174 — LoadIC() populates the
    // per-element y* arrays AND globalY[] (via LoadIC's Sub2Global call at
    // MD_initialize.cpp:131). SetIC2Y(udata) then mirrors to the N_Vector
    // — we skip that since we use globalY[] directly as our Y baseline.
    MD->LoadIC();

    // Forcing + ET arrays must be populated before rhs_core can produce
    // meaningful DY (otherwise qElePrep / qEleETP / qElePotE etc. are
    // zero and surf/unsat/gw flux derivatives are trivially zero).
    // Mirror shud.cpp:213/221 main-loop prelude with t = StartTime.
    const double t0 = MD->CS.StartTime;
    MD->updateforcing(t0);
    MD->ET(t0, t0 + MD->CS.SolverStep);
    std::printf("[fd_color] Model_Data init: NumY=%d (CSC NumY=%llu); t0=%.6f StartTime=%.6f SolverStep=%.6f\n",
                MD->NumY, (unsigned long long)csc.NumY,
                t0, MD->CS.StartTime, MD->CS.SolverStep);
    if ((int64_t)MD->NumY != (int64_t)csc.NumY) {
        std::fprintf(stderr, "[fd_color] ERROR: NumY mismatch between CSC (%llu) and Model_Data (%d)\n",
                     (unsigned long long)csc.NumY, MD->NumY);
        delete MD; delete fout; delete fin;
        return 1;
    }
    const int NumY = MD->NumY;

    // LoadIC's Sub2Global already populated globalY in shud.cpp:35's slot.
    // Read it as our baseline Y.
    extern double *globalY;
    if (globalY == nullptr) {
        std::fprintf(stderr, "[fd_color] ERROR: globalY is nullptr after LoadIC; Sub2Global unexpectedly skipped\n");
        delete MD; delete fout; delete fin;
        return 1;
    }

    // --- 5. FD colored Jacobian probe. For each color c:
    //          v_c[i] = (color[i] == c ? 1 : 0)
    //          DY_base  = rhs_core(Y, t0, Serial)
    //          DY_plus  = rhs_core(Y + ε·v_c, t0, Serial)
    //          For each k with color[k]==c, for each row i with (i,k) in CSC:
    //              J[i,k] = (DY_plus[i] - DY_base[i]) / ε_k
    //          where ε_k = sqrt(DBL_EPSILON) * (|Y[k]| + 1).
    //        Storage: numeric J in CSC order, parallel to csc.row_idx.

    std::vector<double> Y(NumY);
    for (int k = 0; k < NumY; ++k) Y[k] = globalY[k];

    std::vector<double> DY_base(NumY, 0.0);
    std::vector<double> DY_plus(NumY, 0.0);
    std::vector<double> Y_perturbed(NumY, 0.0);

    // Mirror f.cpp:25 — set the global timeNow before each rhs_core call
    // (some flux paths read timeNow for time-varying lookups).
    extern double timeNow;
    timeNow = t0;
    MD->rhs_core(Y.data(), DY_base.data(), t0, ExecPolicy::Serial);

    std::vector<double> J_values(csc.total_nnz, 0.0);

    auto eps_of = [&](int k) -> double {
        const double sq = std::sqrt(DBL_EPSILON);
        return sq * (std::abs(Y[k]) + 1.0);
    };

    const int n_colors = chi;
    std::vector<int> cols_in_color;
    cols_in_color.reserve(NumY / std::max(1, n_colors) + 4);
    for (int c = 0; c < n_colors; ++c) {
        cols_in_color.clear();
        for (int k = 0; k < NumY; ++k) {
            if (color[k] == c) cols_in_color.push_back(k);
        }
        if (cols_in_color.empty()) continue;

        for (int k = 0; k < NumY; ++k) {
            Y_perturbed[k] = Y[k];
        }
        for (int k : cols_in_color) {
            Y_perturbed[k] += eps_of(k);
        }
        timeNow = t0;
        MD->rhs_core(Y_perturbed.data(), DY_plus.data(), t0, ExecPolicy::Serial);

        // Recover J entries. Each row i intersects AT MOST ONE column k in
        // the group (definition of distance-2 column coloring); so
        // J[i,k] = (DY_plus[i] - DY_base[i]) / ε_k is well-defined for the
        // group probe.
        for (int k : cols_in_color) {
            const double e = eps_of(k);
            const int64_t s = csc.col_ptr[k];
            const int64_t end = csc.col_ptr[k + 1];
            for (int64_t p = s; p < end; ++p) {
                const int32_t i = csc.row_idx[p];
                J_values[p] = (DY_plus[i] - DY_base[i]) / e;
            }
        }

        if ((c & 7) == 0) {
            std::fprintf(stdout, "[fd_color] color %d/%d (cols=%zu) probed\n",
                         c + 1, n_colors, cols_in_color.size());
            std::fflush(stdout);
        }
    }

    // --- 6. Emit numeric J binary ---
    const std::string out_path = out_prefix + "_numeric_J.bin";
    FILE *fp = std::fopen(out_path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "[fd_color] ERROR: cannot open %s for write\n", out_path.c_str());
        delete MD; delete fout; delete fin;
        return 1;
    }
    const uint32_t magic = JBIN_MAGIC;
    const uint32_t version = 1u;
    std::fwrite(&magic, sizeof(magic), 1, fp);
    std::fwrite(&version, sizeof(version), 1, fp);
    const uint64_t NumY_out = csc.NumY;
    const uint64_t total_nnz_out = csc.total_nnz;
    const uint64_t n_colors_out = static_cast<uint64_t>(n_colors);
    std::fwrite(&NumY_out, sizeof(uint64_t), 1, fp);
    std::fwrite(&total_nnz_out, sizeof(uint64_t), 1, fp);
    std::fwrite(&n_colors_out, sizeof(uint64_t), 1, fp);
    std::fwrite(csc.col_ptr.data(), sizeof(int64_t), csc.NumY + 1, fp);
    std::fwrite(csc.row_idx.data(), sizeof(int32_t), csc.total_nnz, fp);
    std::fwrite(J_values.data(), sizeof(double), csc.total_nnz, fp);
    std::fclose(fp);

    double abs_min = 1e300, abs_max = 0.0;
    int n_zero = 0;
    for (double v : J_values) {
        const double a = std::abs(v);
        if (a == 0.0) ++n_zero;
        else { if (a < abs_min) abs_min = a; if (a > abs_max) abs_max = a; }
    }
    std::printf("[fd_color] J binary written: %s  total_nnz=%llu n_zero=%d  |v|_min=%.3e |v|_max=%.3e\n",
                out_path.c_str(), (unsigned long long)csc.total_nnz, n_zero, abs_min, abs_max);

    delete MD;
    delete fout;
    delete fin;
    return 0;
}
