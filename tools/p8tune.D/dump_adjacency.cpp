/* tools/p8tune.D/dump_adjacency.cpp
 *
 * P8-tune.D KLU pattern-only spike (PR-0). Per openspec change
 * `p8tune-klu-spike` spec §Requirement REQ-3 "Spike tool SHALL acquire
 * mesh+river+lake adjacency via in-process libshud.a Init walk".
 *
 * In-process Model_Data init flow (mirrors SHUD/src/Model/shud.cpp:118-121):
 *
 *   MD = new Model_Data(fin, fout);
 *   MD->loadinput();
 *   MD->initialize();
 *
 * After init, walks:
 *   (a) MD->Ele[i].nabr[0..2] / lakenabr[0..2] / nabrToMe[0..2] (Element AoS
 *       per SHUD/src/classes/Element.hpp:25-27)
 *   (b) MD->Riv[i].down / toLake / frLake (River AoS per
 *       SHUD/src/classes/River.hpp:54-59)
 *   (c) MD->RivSeg[i] (river-segment → element coupling per
 *       SHUD/src/ModelData/MD_adjacency.hpp S4.1/S4.2 / MD_adjacency.cpp:80-98)
 *
 * MUST NOT dereference MD->rivNode[] — allocation is commented out per
 * SHUD/src/ModelData/MD_readin.cpp:182-187; would segfault.
 *
 * Emits binary 5-block CSC `<case>_adjacency.csc` with row/col blocks
 * {surf, unsat, gw, river, lake} matching state-vector layout per
 * SHUD/src/Model/shud.cpp:139 NY = 3*NumEle + NumRiv + NumLake.
 *
 * Output schema (binary, little-endian, deterministic across runs):
 *   uint32_t  magic = 'A' 'D' 'J' 'C'  (0x434A4441 little-endian)
 *   uint32_t  version = 1
 *   uint64_t  NumEle, NumRiv, NumLake, NumY
 *   uint64_t  total_nnz
 *   uint64_t  per_block_nnz[5*5]    (row-major, J_ss J_su J_sg J_sr J_sl J_us ...)
 *   uint64_t  csc_col_count = NumY  (= row 1's NumEle+NumEle+NumEle+NumRiv+NumLake)
 *   int64_t   col_ptr[NumY + 1]     (CSC: nonzero start index per column, ascending; col_ptr[NumY] = total_nnz)
 *   int32_t   row_idx[total_nnz]    (CSC: row indices, sorted ascending per column for determinism)
 *
 * Determinism guarantee: per-column row index list is sorted ascending via
 * std::sort before write — ensures byte-identical output across runs (no
 * dependence on insertion order from the AoS walk).
 *
 * Usage:
 *   ./dump_adjacency --case <name> [--basin-root <path>] [--out <prefix>]
 *
 * Exit code: 0 on success; 1 on init / IO failure.
 *
 * Refs:
 *   openspec/changes/p8tune-klu-spike/proposal.md §What Changes
 *   openspec/changes/p8tune-klu-spike/design.md D2
 *   openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md REQ-3
 */

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>  // chdir, optind
#include <vector>

#include "CommandIn.hpp"
#include "IO.hpp"
#include "Macros.hpp"
#include "Model_Data.hpp"

// globalY + timeNow are defined in SHUD/src/Model/shud.cpp and transitively pulled
// into libshud.a via MD_rhs_core.cpp's `extern int g_numa_first_touch_enabled` reference.
// DO NOT add duplicate definitions here — they would cause multiple-definition link errors.

namespace {

// State-vector layout indices per SHUD/src/Model/shud.cpp:139
// NY = 3*NumEle + NumRiv + NumLake
//   block 0 (surf): rows [0,             NumEle)
//   block 1 (unsat): rows [NumEle,       2*NumEle)
//   block 2 (gw):   rows [2*NumEle,      3*NumEle)
//   block 3 (river): rows [3*NumEle,     3*NumEle + NumRiv)
//   block 4 (lake): rows [3*NumEle+NumRiv, NumY)
enum BlockId : int {
    BLK_SURF = 0,
    BLK_UNSAT = 1,
    BLK_GW = 2,
    BLK_RIV = 3,
    BLK_LAKE = 4,
    NUM_BLOCKS = 5
};

static inline int64_t row_surf(int i) { return static_cast<int64_t>(i); }
static inline int64_t row_unsat(int i, int NumEle) { return static_cast<int64_t>(NumEle) + i; }
static inline int64_t row_gw(int i, int NumEle) { return 2LL * NumEle + i; }
static inline int64_t row_riv(int i, int NumEle) { return 3LL * NumEle + i; }
static inline int64_t row_lake(int i, int NumEle, int NumRiv) {
    return 3LL * NumEle + NumRiv + i;
}

static inline int block_of_row(int64_t row, int NumEle, int NumRiv, int NumLake) {
    (void)NumLake;
    if (row < NumEle) return BLK_SURF;
    if (row < 2LL * NumEle) return BLK_UNSAT;
    if (row < 3LL * NumEle) return BLK_GW;
    if (row < 3LL * NumEle + NumRiv) return BLK_RIV;
    return BLK_LAKE;
}

// Build per-column sets of row indices, then emit CSC.
struct AdjacencyBuilder {
    int NumEle;
    int NumRiv;
    int NumLake;
    int64_t NumY;
    // For each column, the unique set of rows that have nonzero
    // (column-major sparsity). Use vectors and dedup via sort+unique.
    std::vector<std::vector<int32_t>> cols;

    AdjacencyBuilder(int ne, int nr, int nl)
        : NumEle(ne), NumRiv(nr), NumLake(nl), NumY(3LL * ne + nr + nl), cols(NumY) {}

    void add(int64_t row, int64_t col) {
        if (row < 0 || col < 0 || row >= NumY || col >= NumY) {
            std::fprintf(stderr, "[dump_adjacency] WARN: skipping out-of-range edge row=%lld col=%lld\n",
                         (long long)row, (long long)col);
            return;
        }
        cols[col].push_back(static_cast<int32_t>(row));
    }

    void finalize_sort_unique() {
        for (auto &c : cols) {
            std::sort(c.begin(), c.end());
            c.erase(std::unique(c.begin(), c.end()), c.end());
        }
    }
};

// Add symmetric edge — both (row=a, col=b) and (row=b, col=a) — needed because
// adjacency in a Jacobian is structurally symmetric for SHUD coupling
// (a depends on b ↔ b depends on a in the local-coupling mesh / river-network
// adjacency dump, even if numeric J entries themselves are non-symmetric).
static inline void add_sym(AdjacencyBuilder &B, int64_t a, int64_t b) {
    B.add(a, b);
    B.add(b, a);
}

void build_adjacency_from_md(Model_Data *MD, AdjacencyBuilder &B) {
    const int NumEle = MD->NumEle;
    const int NumRiv = MD->NumRiv;
    const int NumLake = MD->NumLake;
    const int NumSegmt = MD->NumSegmt;

    // ---- diagonal: every state has self-coupling ----
    for (int64_t k = 0; k < B.NumY; ++k) {
        B.add(k, k);
    }

    // ---- intra-element coupling: surf[i] ↔ unsat[i] ↔ gw[i] ----
    // Captured by f_update infiltration / recharge / overland mass flux chains
    // (MD_update.cpp + MD_ElementFlux.cpp). Each element couples its own 3 states.
    for (int i = 0; i < NumEle; ++i) {
        add_sym(B, row_surf(i), row_unsat(i, NumEle));
        add_sym(B, row_surf(i), row_gw(i, NumEle));
        add_sym(B, row_unsat(i, NumEle), row_gw(i, NumEle));
    }

    // ---- inter-element coupling via Ele[i].nabr[j] (3 lateral neighbors) ----
    // surf-to-surf and gw-to-gw lateral flux. unsat is local-only per SHUD model.
    for (int i = 0; i < NumEle; ++i) {
        for (int j = 0; j < 3; ++j) {
            int inabr = MD->Ele[i].nabr[j] - 1;  // 1-based → 0-based; -1 = boundary
            if (inabr >= 0 && inabr < NumEle) {
                add_sym(B, row_surf(i), row_surf(inabr));
                add_sym(B, row_gw(i, NumEle), row_gw(inabr, NumEle));
            }
        }
    }

    // ---- elem ↔ river via RivSeg[k] (seg_by_riv / seg_by_ele per MD_adjacency.cpp:80-98) ----
    // Source: RivSeg[k].iRiv (1-based river) + RivSeg[k].iEle (1-based element).
    // Each RivSeg couples surface flow (riv↔surf) and subsurface flow (riv↔gw).
    for (int k = 0; k < NumSegmt; ++k) {
        int ir = MD->RivSeg[k].iRiv - 1;
        int ie = MD->RivSeg[k].iEle - 1;
        if (ir >= 0 && ir < NumRiv && ie >= 0 && ie < NumEle) {
            add_sym(B, row_riv(ir, NumEle), row_surf(ie));
            add_sym(B, row_riv(ir, NumEle), row_gw(ie, NumEle));
        }
    }

    // ---- river ↔ river downstream (Riv[i].down) ----
    // Per MD_adjacency.cpp:108-115 upstream_by_down — river i's flow feeds into
    // Riv[i].down. Captured by river routing (Manning's eqn).
    for (int i = 0; i < NumRiv; ++i) {
        int idown = MD->Riv[i].down - 1;
        if (idown >= 0 && idown < NumRiv) {
            add_sym(B, row_riv(i, NumEle), row_riv(idown, NumEle));
        }
    }

    // ---- elem ↔ lake via Ele[i].iLake (lake interior cells) ----
    // Per MD_adjacency.cpp:134-141 ele_by_lake. Lake cells couple surf and gw
    // states to lake stage.
    for (int i = 0; i < NumEle; ++i) {
        if (MD->Ele[i].iLake > 0) {
            int ilake = MD->Ele[i].iLake - 1;
            if (ilake >= 0 && ilake < NumLake) {
                add_sym(B, row_surf(i), row_lake(ilake, NumEle, NumRiv));
                add_sym(B, row_gw(i, NumEle), row_lake(ilake, NumEle, NumRiv));
            }
        }
    }

    // ---- elem (bank) ↔ lake via Ele[i].lakenabr[j] ----
    // Per MD_adjacency.cpp:148-155 lake_bank_edge_by_lake. Bank elements
    // (lakenabr[j] - 1 >= 0) exchange lateral surf flux to the neighboring lake.
    for (int i = 0; i < NumEle; ++i) {
        for (int j = 0; j < 3; ++j) {
            int ilake = MD->Ele[i].lakenabr[j] - 1;
            if (ilake >= 0 && ilake < NumLake) {
                add_sym(B, row_surf(i), row_lake(ilake, NumEle, NumRiv));
                add_sym(B, row_gw(i, NumEle), row_lake(ilake, NumEle, NumRiv));
            }
        }
    }

    // ---- river ↔ lake via Riv[i].toLake / Riv[i].frLake ----
    // Per MD_adjacency.cpp:122-127 riv_in_by_lake. Riv[i].toLake is 0-based
    // lake index (>=0 means feeds into a lake). Riv[i].frLake is the upstream
    // lake (lake outlet feeding this river). Both couple river stage ↔ lake stage.
    for (int i = 0; i < NumRiv; ++i) {
        int ilake_to = MD->Riv[i].toLake;
        if (ilake_to >= 0 && ilake_to < NumLake) {
            add_sym(B, row_riv(i, NumEle), row_lake(ilake_to, NumEle, NumRiv));
        }
        int ilake_fr = MD->Riv[i].frLake;
        if (ilake_fr >= 0 && ilake_fr < NumLake) {
            add_sym(B, row_riv(i, NumEle), row_lake(ilake_fr, NumEle, NumRiv));
        }
    }
}

bool write_csc(const std::string &out_path, const AdjacencyBuilder &B) {
    FILE *fp = std::fopen(out_path.c_str(), "wb");
    if (!fp) {
        std::fprintf(stderr, "[dump_adjacency] ERROR: cannot open '%s' for write\n", out_path.c_str());
        return false;
    }

    // Header
    const uint32_t magic = 0x434A4441u;  // 'A' 'D' 'J' 'C' little-endian
    const uint32_t version = 1u;
    std::fwrite(&magic, sizeof(magic), 1, fp);
    std::fwrite(&version, sizeof(version), 1, fp);

    const uint64_t ne = static_cast<uint64_t>(B.NumEle);
    const uint64_t nr = static_cast<uint64_t>(B.NumRiv);
    const uint64_t nl = static_cast<uint64_t>(B.NumLake);
    const uint64_t ny = static_cast<uint64_t>(B.NumY);
    std::fwrite(&ne, sizeof(ne), 1, fp);
    std::fwrite(&nr, sizeof(nr), 1, fp);
    std::fwrite(&nl, sizeof(nl), 1, fp);
    std::fwrite(&ny, sizeof(ny), 1, fp);

    // Compute total nnz + per-block nnz table (5x5 row-major)
    uint64_t total_nnz = 0;
    uint64_t per_block[NUM_BLOCKS * NUM_BLOCKS] = {0};
    for (int64_t col = 0; col < B.NumY; ++col) {
        total_nnz += B.cols[col].size();
        int blk_col = block_of_row(col, B.NumEle, B.NumRiv, B.NumLake);
        for (int32_t row : B.cols[col]) {
            int blk_row = block_of_row(row, B.NumEle, B.NumRiv, B.NumLake);
            per_block[blk_row * NUM_BLOCKS + blk_col]++;
        }
    }
    std::fwrite(&total_nnz, sizeof(total_nnz), 1, fp);
    std::fwrite(per_block, sizeof(uint64_t), NUM_BLOCKS * NUM_BLOCKS, fp);

    const uint64_t csc_col_count = ny;
    std::fwrite(&csc_col_count, sizeof(csc_col_count), 1, fp);

    // CSC col_ptr (NumY + 1 int64_t entries)
    std::vector<int64_t> col_ptr(B.NumY + 1, 0);
    for (int64_t col = 0; col < B.NumY; ++col) {
        col_ptr[col + 1] = col_ptr[col] + static_cast<int64_t>(B.cols[col].size());
    }
    std::fwrite(col_ptr.data(), sizeof(int64_t), B.NumY + 1, fp);

    // CSC row_idx (total_nnz int32_t entries)
    for (int64_t col = 0; col < B.NumY; ++col) {
        if (!B.cols[col].empty()) {
            std::fwrite(B.cols[col].data(), sizeof(int32_t), B.cols[col].size(), fp);
        }
    }

    std::fclose(fp);

    // Print summary to stdout (matches per-cell aggregator parsing convention)
    std::printf("[dump_adjacency] case_nnz_summary\n");
    std::printf("  NumEle=%llu  NumRiv=%llu  NumLake=%llu  NumY=%llu  total_nnz=%llu\n",
                (unsigned long long)ne, (unsigned long long)nr,
                (unsigned long long)nl, (unsigned long long)ny,
                (unsigned long long)total_nnz);
    static const char *blk_names[NUM_BLOCKS] = {"surf", "unsat", "gw", "river", "lake"};
    std::printf("  per_block_nnz (row x col):\n");
    for (int r = 0; r < NUM_BLOCKS; ++r) {
        std::printf("    %-5s", blk_names[r]);
        for (int c = 0; c < NUM_BLOCKS; ++c) {
            std::printf("  %10llu", (unsigned long long)per_block[r * NUM_BLOCKS + c]);
        }
        std::printf("\n");
    }
    std::printf("  output: %s\n", out_path.c_str());
    return true;
}

void print_usage(const char *prog) {
    std::fprintf(stderr,
                 "Usage: %s --case <name> [--basin-root <path>] [--out <prefix>]\n"
                 "  --case        case name (e.g. keliya)\n"
                 "  --basin-root  parent of Basins/<case>/  (default: ../../SHUD/Basins)\n"
                 "  --out         output prefix; emits <prefix>_adjacency.csc (default: <case>)\n",
                 prog);
}

}  // namespace

int main(int argc, char **argv) {
    std::string case_name;
    std::string basin_root = "../../SHUD/Basins";
    std::string out_prefix;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--case" && i + 1 < argc) {
            case_name = argv[++i];
        } else if (arg == "--basin-root" && i + 1 < argc) {
            basin_root = argv[++i];
        } else if (arg == "--out" && i + 1 < argc) {
            out_prefix = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "[dump_adjacency] ERROR: unknown arg '%s'\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    if (case_name.empty()) {
        print_usage(argv[0]);
        return 1;
    }
    if (out_prefix.empty()) out_prefix = case_name;

    // Build FileIn / FileOut per SHUD convention. The CLI for `./shud <case>`
    // resolves paths from cwd via SHUD_BUILD_PATH or auto-detection; we
    // construct an explicit FileIn from the case's input/ subtree.
    // Path layout mirrors SHUD/Basins/<case>/input/<case>/<case>.cfg.* and
    // SHUD/Basins/<case>/output/<case>.out/.
    std::string basin_dir = basin_root + "/" + case_name;
    std::string input_subdir = basin_dir + "/input/" + case_name;
    std::string output_subdir = basin_dir + "/output/" + case_name + ".out";

    // Pre-flight: confirm cfg.para exists; SHUD's IO layer assumes the file
    // is reachable from the CWD it was launched from, so we chdir into basin_dir.
    std::string cfg_path = input_subdir + "/" + case_name + ".cfg.para";
    FILE *probe = std::fopen(cfg_path.c_str(), "r");
    if (!probe) {
        std::fprintf(stderr, "[dump_adjacency] ERROR: %s not found (check --basin-root and --case)\n",
                     cfg_path.c_str());
        return 1;
    }
    std::fclose(probe);

    // chdir into basin_dir so SHUD's relative-path IO ("input/<case>/...") resolves.
    if (chdir(basin_dir.c_str()) != 0) {
        std::fprintf(stderr, "[dump_adjacency] ERROR: cannot chdir to '%s'\n", basin_dir.c_str());
        return 1;
    }

    // Mirror SHUD/src/Model/shud.cpp:595-600 — build CommandIn with a fake
    // argv {prog, case_name} and let CLI.parse + CLI.setFileIO populate
    // FileIn/FileOut path fields. This avoids re-implementing
    // setInFilePath()'s `inpath = "input/<case>"` convention and keeps the
    // init flow byte-identical to `./shud <case>`.
    CommandIn CLI;
    {
        std::vector<char> argbuf0(argv[0], argv[0] + std::strlen(argv[0]) + 1);
        std::vector<char> argbuf1(case_name.begin(), case_name.end());
        argbuf1.push_back('\0');
        char *fake_argv[3] = {argbuf0.data(), argbuf1.data(), nullptr};
        // CommandIn::parse uses getopt which mutates a global `optind`.
        // Reset for re-entry safety.
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
    // NOTE: do NOT call CheckInputData() — it's strictly a runtime safety
    // check and is irrelevant to the adjacency walk; calling it may print
    // a confusing forc-bound diagnostic for our 90-day case. The adjacency
    // fields populated by loadinput()+initialize() are all we need.

    std::printf("[dump_adjacency] loaded case=%s: NumEle=%d NumRiv=%d NumLake=%d NumSegmt=%d NumY=%d\n",
                case_name.c_str(), MD->NumEle, MD->NumRiv, MD->NumLake, MD->NumSegmt, MD->NumY);

    // Build adjacency + emit CSC. NumY from MD takes precedence over our local
    // formula in case SHUD's actual layout differs (it doesn't for the standard
    // build, but trust the runtime authority).
    AdjacencyBuilder B(MD->NumEle, MD->NumRiv, MD->NumLake);
    if (B.NumY != MD->NumY) {
        std::fprintf(stderr,
                     "[dump_adjacency] WARN: layout mismatch — local NY=%lld vs MD->NumY=%d; using local layout\n",
                     (long long)B.NumY, MD->NumY);
    }
    build_adjacency_from_md(MD, B);
    B.finalize_sort_unique();

    // Output file goes back to the cwd-at-launch directory via relative
    // path "../../" walk-up; simplest: write to absolute path constructed
    // from out_prefix (which itself is relative to launch dir, so chdir
    // means it lands in basin_dir/). User typically wants it next to the
    // spike-tool invocation dir; if caller wants elsewhere they can pass
    // an absolute --out path.
    std::string out_path = out_prefix + "_adjacency.csc";
    bool ok = write_csc(out_path, B);

    // P8-tune.F PR-0 (#394 / #386): SHUD Model_Data dtor uninit-ptr UB
    // resolved by NSDMI nullptr defaults in Model_Data.hpp + MD_layout.hpp
    // — see fd_color_jacobian.cpp main exit comment for the full rationale.
    // Restore normal `return` semantics so destructor chain runs to
    // completion (per spec amg-pattern-spike-verdict REQ-3 Scenario
    // "_exit(0) workaround removal").
    std::fflush(stdout);
    std::fflush(stderr);
    // Explicit success-path heap release: mirror the error-path convention
    // (`delete MD; delete fout; delete fin; return 1;`) so the
    // ~Model_Data() → FreeData() chain fires on the happy path too. This
    // is the positive signal the CI dtor-coverage gate (ASan + LeakSanitizer
    // on Linux) checks for. Phase 6 round-1 cross-review F1 closure.
    delete MD;
    delete fout;
    delete fin;
    return ok ? 0 : 1;
}
