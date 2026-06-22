// tools/check_sizeof/check_sizeof.cpp - S5d 汇总验收 (#183) Task 8.1
//
// Two ratio interpretations are emitted; the GATE (exit code 0/1) is on
// (B), the per-element cache-meaningful metric specified by master plan
// §S5d L1432 (spec L145 self-declares as "可校验形式 of master plan
// §S5d L1432"). (A) remains emitted for diagnostic transparency only.
//
//   (A) DIAGNOSTIC — pointer-container struct size (NOT the gate metric):
//         sizeof(ElementHotData) / sizeof(_Element)
//       ElementHotData is a POINTER container (N pointer fields, each
//       sizeof(void*)). The numerator is a fixed 32 × 8 = 256 B on any
//       64-bit ABI, independent of NumEle. As NumEle → ∞ this ratio
//       goes to 0 — dimensionally meaningless as a cache-footprint
//       proxy. Kept emitted for review continuity; review-fix F1
//       (PR #201) moved the gate off this metric.
//
//   (B) GATE METRIC — master plan §S5d L1432 + spec L148-149:
//         sum_over_hot_fields(sizeof(type) * size_per_ele) / sizeof(_Element)
//       i.e. per-element SoA byte footprint vs per-element AoS footprint.
//       This is the cache-meaningful number that drives §4.22.5 cache
//       miss reduction estimate. Tools/perf_stat ratio in Task 8.2
//       measures the actual cache effect downstream.
//
// Field-type / size_per_ele schema is fixed by docs/s5d_hot_fields.yaml
// (32 hot fields, audited by tools/check_manifest/check_hot_fields.py).
// The hardcoded SUM here is derived from that yaml and is asserted via
// static_assert(N_FIELDS == 32). Yaml drift -> CI gate first; both
// numbers become stale on the SAME PR, so review catches it.
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "Macros.hpp"
#include "ModelConfigure.hpp"      /* Soil_Layer / Geol_Layer / Landcover */
#include "Node.hpp"                /* _Node — used by Element ctor */
#include "Equations.hpp"           /* is_sm_et helpers */
#include "Element.hpp"             /* Triangle + _Element */
#include "MD_layout.hpp"           /* ElementHotData */

/* Per-element SoA byte footprint, derived field-by-field from
 * docs/s5d_hot_fields.yaml schema + MD_layout.hpp. 32 fields total.
 *
 * The classification mirrors MD_layout.hpp declarations:
 *   - 2 int[3] flats              : nabr_flat, lakenabr_flat
 *   - 7 int scalars               : iSoil, iLC, iMF, iForc, iLake, iBC, iSS
 *   - 4 double[3] flats           : edge_flat, Dist2Nabor_flat,
 *                                   Dist2Edge_flat, avgRough_flat
 *   - 19 double scalars           : area, z_bottom, z_surf, FixPressure,
 *                                   WetlandLevel, RootReachLevel,
 *                                   depression, QBC, QSS, windH,
 *                                   u_qi, u_qex, u_effKH, u_satn,
 *                                   Sy, VegFrac, Albedo, Rough, ImpAF
 *
 * Sanity: 2+7+4+19 = 32 fields, matches yaml + MD_layout.hpp.
 */
static constexpr int    N_INT3      = 2;
static constexpr int    N_INT1      = 7;
static constexpr int    N_DOUBLE3   = 4;
static constexpr int    N_DOUBLE1   = 19;
static constexpr int    N_FIELDS    = N_INT3 + N_INT1 + N_DOUBLE3 + N_DOUBLE1;

static_assert(N_FIELDS == 32,
              "hot field count drift — re-derive from docs/s5d_hot_fields.yaml");

int main(void) {
    /* (B) Per-element SoA byte footprint. */
    const std::size_t per_ele_int3    = N_INT3    * sizeof(int)    * 3;
    const std::size_t per_ele_int1    = N_INT1    * sizeof(int);
    const std::size_t per_ele_dbl3    = N_DOUBLE3 * sizeof(double) * 3;
    const std::size_t per_ele_dbl1    = N_DOUBLE1 * sizeof(double);
    const std::size_t per_ele_total   = per_ele_int3 + per_ele_int1
                                      + per_ele_dbl3 + per_ele_dbl1;

    const std::size_t soa_struct      = sizeof(ElementHotData);  /* pointer container */
    const std::size_t aos_per_ele     = sizeof(_Element);        /* one AoS element */
    const double ratio_literal        = static_cast<double>(soa_struct)
                                      / static_cast<double>(aos_per_ele);  /* (A) diagnostic */
    const double ratio_real           = static_cast<double>(per_ele_total)
                                      / static_cast<double>(aos_per_ele);  /* (B) GATE */
    const double threshold            = 0.20;  /* spec L149 + master plan L1432 */

    std::printf("=== S5d 汇总验收 sizeof emission (Task 8.1) ===\n");
    std::printf("hot_field_count           = %d (yaml + MD_layout.hpp)\n", N_FIELDS);
    std::printf("per_ele_int3   (2 fields) = %zu bytes\n", per_ele_int3);
    std::printf("per_ele_int1   (7 fields) = %zu bytes\n", per_ele_int1);
    std::printf("per_ele_dbl3   (4 fields) = %zu bytes\n", per_ele_dbl3);
    std::printf("per_ele_dbl1   (19 fields)= %zu bytes\n", per_ele_dbl1);
    std::printf("per_ele_soa_bytes         = %zu bytes  (sum, SoA payload per element)\n",
                per_ele_total);
    std::printf("sizeof(ElementHotData)    = %zu bytes  (struct = %d pointers × %zu B)\n",
                soa_struct, N_FIELDS, sizeof(void*));
    std::printf("sizeof(_Element)          = %zu bytes  (fat-AoS, master plan §4.22.1)\n",
                aos_per_ele);
    std::printf("\n");
    std::printf("Ratio (A) DIAGNOSTIC — pointer-container struct size (NOT the gate metric):\n");
    std::printf("  sizeof(ElementHotData) / sizeof(_Element) = %zu / %zu = %.4f\n",
                soa_struct, aos_per_ele, ratio_literal);
    std::printf("  (numerator = %d × sizeof(void*) = %zu B, fixed on any 64-bit ABI;\n",
                N_FIELDS, soa_struct);
    std::printf("   independent of NumEle — dimensionally meaningless for cache footprint)\n");
    std::printf("Ratio (B) GATE METRIC — master plan §S5d L1432 + spec L148-149:\n");
    std::printf("  per_ele_soa_bytes / sizeof(_Element) = %zu / %zu = %.4f\n",
                per_ele_total, aos_per_ele, ratio_real);
    std::printf("  threshold = %.4f  -> %s (per-element SoA cache footprint)\n",
                threshold, ratio_real < threshold ? "PASS" : "FAIL");
    std::printf("\n");

    /* Gate on (B) — the per-element cache-meaningful metric from
     * master plan §S5d L1432. Review-fix F1 (PR #201) moved the gate
     * here from (A) which was dimensionally meaningless. (A) stays
     * emitted for diagnostic transparency.
     */
    if (ratio_real < threshold) {
        std::printf("VERDICT: PASS gate on (B) (%.4f < %.4f)\n",
                    ratio_real, threshold);
        std::printf("NOTE: (A) diagnostic ratio %.4f shown for transparency only.\n",
                    ratio_literal);
        std::printf("      see docs/b1b_summary.md S5d section for full discussion.\n");
        return 0;
    }
    std::printf("VERDICT: FAIL gate on (B) (%.4f >= %.4f)\n",
                ratio_real, threshold);
    std::printf("NOTE: (A) diagnostic ratio %.4f shown for transparency only.\n",
                ratio_literal);
    return 1;
}
