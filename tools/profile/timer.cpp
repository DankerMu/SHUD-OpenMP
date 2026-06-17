/* timer.cpp — profile bucket accumulator + YAML dump (S0-8a, openMP #10).
 *
 * Only compiled into the SHUD binary when SHUD_ENABLE_PROFILE is defined
 * at translation-unit scope (Makefile passes -DSHUD_ENABLE_PROFILE=1).
 * The whole impl is wrapped in #ifdef so a stray build of this TU
 * without the flag is a harmless no-op TU.
 *
 * S0-10 will add the real instrumentation hook points inside SHUD source;
 * #10 ships skeleton output (all buckets present, values default to 0.0
 * when nothing has accumulated).
 */
#include "timer.h"

#ifdef SHUD_ENABLE_PROFILE

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace shud_profile {

namespace {

/* Bucket = monotonically-accumulated nanoseconds, lock-free atomic add.
 * We store `unique_ptr<atomic>` (rather than the atomic by value) so
 * the unordered_map can rehash without UB on atomic copy/move. */
using BucketMap =
    std::unordered_map<std::string,
                       std::unique_ptr<std::atomic<int64_t>>>;

struct Registry {
    BucketMap  buckets;
    std::mutex mu;
};

Registry &registry() {
    static Registry r;
    return r;
}

}  /* namespace */

void add_elapsed(const char *bucket, std::chrono::nanoseconds dt) {
    if (bucket == nullptr) return;
    const int64_t ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(dt).count();
    Registry &r = registry();
    std::atomic<int64_t> *slot = nullptr;
    {
        std::lock_guard<std::mutex> lk(r.mu);
        auto it = r.buckets.find(bucket);
        if (it == r.buckets.end()) {
            auto p = std::make_unique<std::atomic<int64_t>>(0);
            slot   = p.get();
            r.buckets.emplace(std::string(bucket), std::move(p));
        } else {
            slot = it->second.get();
        }
    }
    /* Hot path: lock-free atomic add. */
    slot->fetch_add(ns, std::memory_order_relaxed);
}

namespace {

/* Buckets emitted by `dump`, master plan §S0.12. Order preserved so
 * the YAML output is deterministic; missing-from-registry buckets emit
 * 0.0 as the skeleton placeholder. */
const char *const kCanonicalBuckets[] = {
    "t_RHS_kernel",
    "t_RHS_total",
    "t_CVODE_internal",
    "t_forcing_io",
    "t_ET",
    "t_output",
    "t_other",
};
constexpr size_t kCanonicalBucketCount =
    sizeof(kCanonicalBuckets) / sizeof(kCanonicalBuckets[0]);

double load_seconds(Registry &r, const char *name) {
    /* Caller holds r.mu. */
    auto it = r.buckets.find(name);
    if (it == r.buckets.end()) return 0.0;
    const int64_t ns = it->second->load(std::memory_order_relaxed);
    return static_cast<double>(ns) * 1e-9;
}

}  /* namespace */

void dump(const char *yaml_path) {
    if (yaml_path == nullptr) return;
    FILE *fp = std::fopen(yaml_path, "w");
    if (fp == nullptr) {
        /* Non-fatal: profile dump is observability, not correctness. */
        std::fprintf(stderr,
                     "[shud_profile] fopen('%s') failed; profile not "
                     "written.\n",
                     yaml_path);
        return;
    }
    Registry &r = registry();
    std::lock_guard<std::mutex> lk(r.mu);

    std::fprintf(fp, "# SHUD profile (S0-8a skeleton, openMP #10)\n");
    std::fprintf(fp, "# Units: seconds. Buckets not yet instrumented "
                     "report 0.0.\n");
    std::fprintf(fp, "buckets:\n");
    for (size_t i = 0; i < kCanonicalBucketCount; ++i) {
        const char *name = kCanonicalBuckets[i];
        const double sec = load_seconds(r, name);
        std::fprintf(fp, "  %s: %.9f\n", name, sec);
    }
    /* Surface any extra buckets the source code starts emitting before
     * the canonical list is updated. Skipped if absent. */
    bool extras_header_emitted = false;
    for (const auto &kv : r.buckets) {
        bool is_canonical = false;
        for (size_t i = 0; i < kCanonicalBucketCount; ++i) {
            if (kv.first == kCanonicalBuckets[i]) {
                is_canonical = true;
                break;
            }
        }
        if (is_canonical) continue;
        if (!extras_header_emitted) {
            std::fprintf(fp, "extras:\n");
            extras_header_emitted = true;
        }
        const int64_t ns =
            kv.second->load(std::memory_order_relaxed);
        std::fprintf(fp, "  %s: %.9f\n", kv.first.c_str(),
                     static_cast<double>(ns) * 1e-9);
    }
    /* CVODE stats placeholder — #10 ships the empty subtree; populated
     * in a later stage when CVODE-stat ingestion lands. */
    std::fprintf(fp, "cvode_stats:\n");
    std::fclose(fp);
}

}  /* namespace shud_profile */

#endif  /* SHUD_ENABLE_PROFILE */
