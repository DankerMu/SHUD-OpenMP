/* SHUD RHS snapshot binary format — authoritative version header.
 *
 * Owned by the outer SHUD-OpenMP repo (`tools/rhs_snapshot/`); read by
 * the snapshot writer / compare tools (S0-7, openmp issue #9).
 *
 * The SHUD submodule keeps a sister copy at
 *   SHUD/src/ModelData/MD_rhs_dump.h
 * with the SAME `SHUD_RHS_SNAPSHOT_FORMAT_VERSION` value so the hook
 * call site (in MD_f.cpp / MD_update.cpp) and the writer impl agree on
 * schema. A drift between the two is a build-script CI check (S0-7).
 *
 * Bump VERSION when the binary payload schema changes; never reuse a
 * previous version number.
 *
 * S0-6 (#8) scope: hook insertion + format version pin only. Writer
 * impl (binary layout, magic, byte order, payload arrays) lands in S0-7.
 */
#ifndef SHUD_RHS_SNAPSHOT_FORMAT_H
#define SHUD_RHS_SNAPSHOT_FORMAT_H

#define SHUD_RHS_SNAPSHOT_FORMAT_VERSION 1
#define SHUD_RHS_SNAPSHOT_MAGIC          "SHUDRHS"  /* 7 chars; writer pads to 8 with NUL */

#endif /* SHUD_RHS_SNAPSHOT_FORMAT_H */
