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
 *
 * Filename suffix (#43 / S1-pre-B): host-side tooling that composes
 * snapshot paths SHOULD route through `compose_snapshot_filename()`
 * below so the same env-var convention used by SHUD's replicated
 * writer (SHUD_DUMP_FNAME_SUFFIX) is honored. See `compose_snapshot_filename`
 * doc + `MD_rhs_dump.cpp` SCHEMA DUPLICATION NOTE for the
 * disambiguation contract that lets `f_update` and `f_loop_before_passvalue`
 * goldens coexist in the same output dir.
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

/* Compose a snapshot filename `snapshot_t<t_value>[_<suffix>].bin`.
 *
 * Mirrors the disambiguation logic in `SHUD/src/ModelData/MD_rhs_dump.cpp`
 * (#43 / S1-pre-B): when `suffix` is empty, returns the legacy form
 * `snapshot_t<t_value>.bin` (bitwise back-compat with PR #53 goldens);
 * when non-empty, returns `snapshot_t<t_value>_<suffix>.bin`.
 *
 * Suffix rejection (returns empty string; callers MUST check + skip write):
 *   1. Path-traversal guard: `/` or `\\` in suffix.
 *   2. Length cap (F5): suffix longer than 64 chars.
 * Order matters: path-traversal is checked first so the caller can tell
 * the failure mode by re-checking the suffix.
 *
 * `t_value` is formatted with `%.0f` to mirror the in-tree writer; two
 * targets within 1.0 of each other will collide under this precision
 * — the SHUD-side init guard rejects such target sets at config load.
 *
 * `compose_snapshot_filename_from_env` reads `SHUD_DUMP_FNAME_SUFFIX`
 * from the process env (empty if unset) and delegates to the
 * explicit-suffix overload; intended for host-side tooling that wants
 * the same env contract as SHUD's replicated writer.
 */
std::string compose_snapshot_filename(double             t_value,
                                      const std::string& suffix);
std::string compose_snapshot_filename_from_env(double t_value);

}  /* namespace shud_snap */

#endif /* SHUD_RHS_SNAPSHOT_WRITER_H */
