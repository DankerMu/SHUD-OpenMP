/* SHUD RHS snapshot binary format — authoritative version + layout header.
 *
 * Owned by the outer SHUD-OpenMP repo (`tools/rhs_snapshot/`); read by
 * the snapshot writer / compare tools (S0-7, openmp issue #9).
 *
 * The SHUD submodule keeps a sister copy at
 *   SHUD/src/ModelData/MD_rhs_dump.h
 * with the SAME `SHUD_RHS_SNAPSHOT_FORMAT_VERSION` value so the hook
 * call site (in MD_f.cpp / MD_update.cpp) and the writer impl agree on
 * schema. A drift between the two is a build-script CI check (S0-9).
 *
 * Bump VERSION when the binary payload schema changes; never reuse a
 * previous version number.
 *
 * Endianness: file is little-endian (target machines are x86_64 Linux
 * and Apple Silicon, both LE). A future big-endian port MUST add a
 * byte-swap path at read time; the writer assumes host == LE.
 *
 * Layout (file = FileHeader + RecordHeader + Array[array_count]):
 *
 *   FileHeader    (40 bytes, packed)
 *     char     magic[4]      = "SHRH"  (no NUL terminator)
 *     uint32   version       = SHUD_RHS_SNAPSHOT_FORMAT_VERSION
 *     char     case_id[32]   (NUL-padded; truncated on overrun)
 *
 *   RecordHeader (12 bytes, packed) — exactly one per file in v1
 *     double   t_value       (model time at probe; SHUD t-unit = minutes)
 *     uint32   array_count
 *
 *   Array[i]     (variable length)
 *     uint32   name_len
 *     char     name[name_len]            (no NUL terminator)
 *     uint64   nelem
 *     double   data[nelem]               (raw, little-endian)
 *
 * S0-7 (#9) ships v1 with array_count = 1 (single "DY" array per
 * snapshot). Multi-array probes are deferred to a later spec rev.
 */
#ifndef SHUD_RHS_SNAPSHOT_FORMAT_H
#define SHUD_RHS_SNAPSHOT_FORMAT_H

#include <cstdint>

#define SHUD_RHS_SNAPSHOT_FORMAT_VERSION 1
#define SHUD_RHS_SNAPSHOT_MAGIC          "SHRH"   /* 4 chars exactly, no NUL */

#pragma pack(push, 1)

struct ShudSnapshotFileHeader {
    char     magic[4];     /* "SHRH" */
    uint32_t version;      /* SHUD_RHS_SNAPSHOT_FORMAT_VERSION */
    char     case_id[32];  /* NUL-padded */
};

struct ShudSnapshotRecordHeader {
    double   t_value;
    uint32_t array_count;
};

#pragma pack(pop)

static_assert(sizeof(ShudSnapshotFileHeader)   == 40,
              "ShudSnapshotFileHeader must be exactly 40 bytes; struct layout drifted");
static_assert(sizeof(ShudSnapshotRecordHeader) == 12,
              "ShudSnapshotRecordHeader must be exactly 12 bytes; struct layout drifted");

#endif /* SHUD_RHS_SNAPSHOT_FORMAT_H */
