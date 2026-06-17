/* writer.h — standalone snapshot writer (S0-7, openmp issue #9).
 *
 * No SUNDIALS / SHUD dependency. C++17 + libc only. Used by:
 *   - the SHUD instrumentation in `SHUD/src/ModelData/MD_rhs_dump.cpp`
 *     (REPLICATED there because the submodule cannot link against
 *     outer-repo objects; the replica MUST stay byte-equivalent;
 *     S0-9 CI will diff schema headers to catch drift)
 *   - any future host-side dump tooling outside SHUD
 *
 * Byte order: host is assumed little-endian (target: x86_64 / Apple
 * Silicon). A future BE port MUST add explicit byte-swap calls; the
 * v1 writer just `fwrite`s raw bytes.
 */
#ifndef SHUD_RHS_SNAPSHOT_WRITER_H
#define SHUD_RHS_SNAPSHOT_WRITER_H

#include <cstdint>
#include <string>
#include <vector>

#include "format.h"

namespace shud_snap {

struct ArraySpec {
    std::string   name;
    const double *data;
    uint64_t      nelem;
};

/* Write a single-record snapshot file at `path`.
 *
 * On success: file contains FileHeader + RecordHeader + arrays per
 * format.h. Returns 0.
 *
 * On fopen failure: returns -1, no file produced. The caller is
 * responsible for emitting a diagnostic.
 *
 * `case_id` longer than 31 chars is silently truncated (header field
 * is 32 bytes including the implicit NUL pad).
 */
int write_snapshot(const std::string&             path,
                   const std::string&             case_id,
                   double                         t_value,
                   const std::vector<ArraySpec>&  arrays);

}  /* namespace shud_snap */

#endif /* SHUD_RHS_SNAPSHOT_WRITER_H */
