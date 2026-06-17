/* writer.cpp — standalone snapshot writer (S0-7, openmp issue #9).
 *
 * Pure libc / C++17. See writer.h header doc for layout and constraints.
 */
#include "writer.h"

#include <cstdint>
#include <cstdio>
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

}  /* namespace shud_snap */
