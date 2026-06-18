/* writer.cpp — standalone snapshot writer (S0-7, openmp issue #9).
 *
 * Pure libc / C++17. See writer.h header doc for layout and constraints.
 *
 * #43 (S1-pre-B) adds `compose_snapshot_filename` + `compose_snapshot_filename_from_env`
 * to mirror SHUD_DUMP_FNAME_SUFFIX disambiguation; kept in lock-step
 * with `SHUD/src/ModelData/MD_rhs_dump.cpp::init_config()` and the
 * snprintf in `shud_rhs_dump_point()` (per SCHEMA DUPLICATION NOTE
 * at MD_rhs_dump.cpp:10-19).
 */
#include "writer.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace shud_snap {

int write_snapshot(const std::string&             path,
                   const std::string&             case_id,
                   double                         t_value,
                   const std::vector<ArraySpec>&  arrays) {
    FILE *fp = std::fopen(path.c_str(), "wb");
    if (!fp) {
        return -1;
    }

    /* File header */
    ShudSnapshotFileHeader fh;
    std::memcpy(fh.magic, SHUD_RHS_SNAPSHOT_MAGIC, 4);
    fh.version = static_cast<uint32_t>(SHUD_RHS_SNAPSHOT_FORMAT_VERSION);
    std::memset(fh.case_id, 0, sizeof(fh.case_id));
    {
        const std::size_t n = case_id.size() < sizeof(fh.case_id)
                                ? case_id.size() : sizeof(fh.case_id);
        std::memcpy(fh.case_id, case_id.data(), n);
    }
    std::fwrite(&fh, sizeof(fh), 1, fp);

    /* Record header */
    ShudSnapshotRecordHeader rh;
    rh.t_value     = t_value;
    rh.array_count = static_cast<uint32_t>(arrays.size());
    std::fwrite(&rh, sizeof(rh), 1, fp);

    /* Arrays */
    for (const auto& a : arrays) {
        const uint32_t name_len = static_cast<uint32_t>(a.name.size());
        std::fwrite(&name_len, sizeof(name_len), 1, fp);
        if (name_len > 0) {
            std::fwrite(a.name.data(), 1, name_len, fp);
        }
        const uint64_t nelem = a.nelem;
        std::fwrite(&nelem, sizeof(nelem), 1, fp);
        if (nelem > 0 && a.data != nullptr) {
            std::fwrite(a.data, sizeof(double), nelem, fp);
        }
    }

    std::fclose(fp);
    return 0;
}

std::string compose_snapshot_filename(double             t_value,
                                      const std::string& suffix) {
    /* Path-traversal guard: mirror MD_rhs_dump.cpp init_config(). */
    if (!suffix.empty()) {
        if (suffix.find('/')  != std::string::npos ||
            suffix.find('\\') != std::string::npos) {
            return std::string();
        }
        /* F5 fix: 64-char length cap (same as MD_rhs_dump.cpp::init_config).
         * Returns empty string so callers route through the same skip-write
         * path as the path-traversal rejection. */
        if (suffix.size() > 64) {
            return std::string();
        }
    }
    char buf[128];
    if (suffix.empty()) {
        std::snprintf(buf, sizeof(buf), "snapshot_t%.0f.bin", t_value);
    } else {
        std::snprintf(buf, sizeof(buf), "snapshot_t%.0f_%s.bin",
                      t_value, suffix.c_str());
    }
    return std::string(buf);
}

std::string compose_snapshot_filename_from_env(double t_value) {
    const char *sfx = std::getenv("SHUD_DUMP_FNAME_SUFFIX");
    const std::string suffix = (sfx && sfx[0]) ? sfx : "";
    return compose_snapshot_filename(t_value, suffix);
}

}  /* namespace shud_snap */
