/* compare_snapshot — bitwise + ULP diff for SHUD RHS snapshot files.
 *
 * S0-7 (openmp issue #9). Standalone C++17 — no SUNDIALS / SHUD link.
 *
 * Exit codes (spec):
 *   0  BITWISE IDENTICAL
 *   1  bitwise differs (per-array ULP report on stdout)
 *   2  format mismatch (magic / version / truncated)
 *   3  golden file missing (file b does not exist)
 *
 * Usage:
 *   compare_snapshot <a.bin> <b.bin> [flags]
 *     --quiet
 *     --max-arrays-shown=N
 *     --exit-on-first-diff
 *     --help
 */
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <ios>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "format.h"

namespace {

struct Array {
    std::string         name;
    uint64_t            nelem;
    std::vector<double> data;
};

struct Snapshot {
    ShudSnapshotFileHeader   fh;
    ShudSnapshotRecordHeader rh;
    std::vector<Array>       arrays;
};

enum LoadStatus {
    LOAD_OK            = 0,
    LOAD_NOT_FOUND     = 3,
    LOAD_TRUNCATED     = 21,  /* internal codes; printer maps to spec exit codes */
    LOAD_MAGIC_BAD     = 22,
    LOAD_VERSION_BAD   = 23,
    LOAD_IO_ERROR      = 24,
};

bool path_exists(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0;
}

/* Returns LOAD_* status. On non-LOAD_OK callers must consult `out` only
 * for partial data already filled (most callers ignore on failure). */
LoadStatus load_snapshot(const std::string& path, Snapshot& out) {
    if (!path_exists(path)) {
        return LOAD_NOT_FOUND;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return LOAD_IO_ERROR;
    }
    f.seekg(0, std::ios::end);
    const std::streampos file_size = f.tellg();
    f.seekg(0, std::ios::beg);

    if (file_size < static_cast<std::streampos>(
                        sizeof(ShudSnapshotFileHeader) +
                        sizeof(ShudSnapshotRecordHeader))) {
        return LOAD_TRUNCATED;
    }

    if (!f.read(reinterpret_cast<char*>(&out.fh),
                sizeof(out.fh))) {
        return LOAD_TRUNCATED;
    }
    if (std::memcmp(out.fh.magic, SHUD_RHS_SNAPSHOT_MAGIC, 4) != 0) {
        return LOAD_MAGIC_BAD;
    }
    if (out.fh.version !=
        static_cast<uint32_t>(SHUD_RHS_SNAPSHOT_FORMAT_VERSION)) {
        return LOAD_VERSION_BAD;
    }
    if (!f.read(reinterpret_cast<char*>(&out.rh),
                sizeof(out.rh))) {
        return LOAD_TRUNCATED;
    }

    out.arrays.reserve(out.rh.array_count);
    for (uint32_t i = 0; i < out.rh.array_count; ++i) {
        Array a;
        uint32_t name_len = 0;
        if (!f.read(reinterpret_cast<char*>(&name_len), sizeof(name_len))) {
            return LOAD_TRUNCATED;
        }
        a.name.assign(name_len, '\0');
        if (name_len > 0 && !f.read(a.name.data(), name_len)) {
            return LOAD_TRUNCATED;
        }
        uint64_t nelem = 0;
        if (!f.read(reinterpret_cast<char*>(&nelem), sizeof(nelem))) {
            return LOAD_TRUNCATED;
        }
        a.nelem = nelem;
        a.data.resize(nelem);
        if (nelem > 0 &&
            !f.read(reinterpret_cast<char*>(a.data.data()),
                    static_cast<std::streamsize>(nelem * sizeof(double)))) {
            return LOAD_TRUNCATED;
        }
        out.arrays.push_back(std::move(a));
    }
    return LOAD_OK;
}

/* Standard 1-step ULP distance between adjacent doubles: bit-cast to
 * int64 with sign-magnitude correction, then absolute difference. */
uint64_t ulp_diff(double x, double y) {
    if (x == y) return 0;
    if (std::isnan(x) || std::isnan(y)) return UINT64_MAX;
    int64_t ix, iy;
    std::memcpy(&ix, &x, sizeof(ix));
    std::memcpy(&iy, &y, sizeof(iy));
    /* Map negatives so that adjacency works across zero. */
    if (ix < 0) ix = INT64_MIN - ix;
    if (iy < 0) iy = INT64_MIN - iy;
    int64_t d = ix - iy;
    return (d < 0) ? static_cast<uint64_t>(-d) : static_cast<uint64_t>(d);
}

void print_help() {
    std::printf(
        "compare_snapshot — diff two SHUD RHS snapshot binaries.\n"
        "\n"
        "Usage:\n"
        "  compare_snapshot <a.bin> <b.bin> [flags]\n"
        "\n"
        "Flags:\n"
        "  --quiet                  suppress per-array detail on identical files\n"
        "  --max-arrays-shown=N     cap the number of differing-array reports\n"
        "  --exit-on-first-diff     stop scanning at the first differing array\n"
        "  --help                   print this message and exit 0\n"
        "\n"
        "Exit codes:\n"
        "  0  bitwise identical\n"
        "  1  bitwise differs (ULP report on stdout)\n"
        "  2  format mismatch (magic/version/truncated)\n"
        "  3  golden snapshot not found (path b does not exist)\n");
}

}  /* namespace */

int main(int argc, char** argv) {
    bool        quiet               = false;
    std::size_t max_arrays_shown    = SIZE_MAX;
    bool        exit_on_first_diff  = false;
    std::vector<std::string> positional;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--help" || a == "-h") {
            print_help();
            return 0;
        } else if (a == "--quiet") {
            quiet = true;
        } else if (a == "--exit-on-first-diff") {
            exit_on_first_diff = true;
        } else if (a.rfind("--max-arrays-shown=", 0) == 0) {
            const std::string v = a.substr(std::string("--max-arrays-shown=").size());
            char *end = nullptr;
            unsigned long n = std::strtoul(v.c_str(), &end, 10);
            if (end == v.c_str() || *end != '\0') {
                std::fprintf(stderr, "invalid --max-arrays-shown value: %s\n", v.c_str());
                return 2;
            }
            max_arrays_shown = static_cast<std::size_t>(n);
        } else if (!a.empty() && a[0] == '-') {
            std::fprintf(stderr, "unknown flag: %s\n", a.c_str());
            return 2;
        } else {
            positional.push_back(a);
        }
    }

    if (positional.size() != 2) {
        std::fprintf(stderr, "usage: compare_snapshot <a.bin> <b.bin> [flags]\n");
        return 2;
    }

    const std::string& path_a = positional[0];
    const std::string& path_b = positional[1];

    Snapshot A, B;
    LoadStatus sa = load_snapshot(path_a, A);
    LoadStatus sb = load_snapshot(path_b, B);

    /* Golden missing — spec exit 3 keyed off file `b`. If file `a` also
     * missing, surface that as well but b's missing-ness dominates. */
    if (sb == LOAD_NOT_FOUND) {
        std::fprintf(stderr, "golden snapshot not found\n");
        return 3;
    }
    if (sa == LOAD_NOT_FOUND) {
        std::fprintf(stderr, "golden snapshot not found\n");
        return 3;
    }

    if (sa == LOAD_TRUNCATED || sb == LOAD_TRUNCATED) {
        std::fprintf(stderr, "snapshot file truncated\n");
        return 2;
    }
    if (sa == LOAD_MAGIC_BAD || sb == LOAD_MAGIC_BAD ||
        sa == LOAD_VERSION_BAD || sb == LOAD_VERSION_BAD) {
        std::fprintf(stderr, "format mismatch\n");
        return 2;
    }
    if (sa != LOAD_OK || sb != LOAD_OK) {
        std::fprintf(stderr, "snapshot load error\n");
        return 2;
    }

    /* Structural compare */
    if (A.rh.array_count != B.rh.array_count) {
        std::fprintf(stderr, "format mismatch\n");
        return 2;
    }

    bool        any_diff      = false;
    std::size_t arrays_shown  = 0;

    for (std::size_t i = 0; i < A.arrays.size(); ++i) {
        const Array& a = A.arrays[i];
        const Array& b = B.arrays[i];
        if (a.name != b.name || a.nelem != b.nelem) {
            std::fprintf(stderr, "format mismatch\n");
            return 2;
        }
        uint64_t max_ulp        = 0;
        double   max_abs_diff   = 0.0;
        double   l2_sq          = 0.0;
        uint64_t first_diff_idx = static_cast<uint64_t>(-1);
        for (uint64_t k = 0; k < a.nelem; ++k) {
            const double x = a.data[k];
            const double y = b.data[k];
            if (x == y) continue;
            const uint64_t u = ulp_diff(x, y);
            if (u > max_ulp) max_ulp = u;
            const double d = std::fabs(x - y);
            if (d > max_abs_diff) max_abs_diff = d;
            l2_sq += (x - y) * (x - y);
            if (first_diff_idx == static_cast<uint64_t>(-1)) {
                first_diff_idx = k;
            }
        }
        if (first_diff_idx != static_cast<uint64_t>(-1)) {
            any_diff = true;
            if (arrays_shown < max_arrays_shown) {
                std::printf(
                    "%s: max_ulp=%llu max_abs_diff=%.17g first_diff_index=%llu l2_norm_diff=%.17g\n",
                    a.name.c_str(),
                    static_cast<unsigned long long>(max_ulp),
                    max_abs_diff,
                    static_cast<unsigned long long>(first_diff_idx),
                    std::sqrt(l2_sq));
                ++arrays_shown;
            }
            if (exit_on_first_diff) {
                return 1;
            }
        }
    }

    if (!any_diff) {
        std::printf("BITWISE IDENTICAL\n");
        (void) quiet;  /* identical line is always one line; quiet only
                         suppresses per-array detail when also identical
                         — there is none in that case. */
        return 0;
    }
    return 1;
}
