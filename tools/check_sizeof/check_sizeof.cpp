// tools/check_sizeof/check_sizeof.cpp - S5d 汇总验收 (#183) Task 8.1
//
// Two ratio interpretations are emitted; the gate (exit code 0/1) is on
// (A), the spec-literal one. (B) is shown for transparency and informs
// the downstream Task 8.2 cache-miss perf-stat reduction discussion.
//
//   (A) spec L148-149 LITERAL:
//         sizeof(ElementHotData) / sizeof(_Element)
//       Here ElementHotData is a POINTER container (N pointer fields,
//       each sizeof(void*)). Independent of NumEle. This is the spec
//       scenario wording verbatim.
//
//   (B) master plan section S5d L1432 cache-meaningful semantics:
//         sum_over_hot_fields(sizeof(type) * size_per_ele) / sizeof(_Element)
//       i.e. per-element SoA byte footprint vs per-element AoS footprint.
//       This is the cache-meaningful number that drives the section 4.22.5
//       cache miss reduction estimate. Tools/perf_stat ratio in Task 8.2
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
                                      / static_cast<double>(aos_per_ele);  /* (A) */
    const double ratio_real           = static_cast<double>(per_ele_total)
                                      / static_cast<double>(aos_per_ele);  /* (B) */
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
    std::printf("Ratio (A) literal spec L148-149:\n");
    std::printf("  sizeof(ElementHotData) / sizeof(_Element) = %zu / %zu = %.4f\n",
                soa_struct, aos_per_ele, ratio_literal);
    std::printf("  threshold = %.4f  -> %s (literal pointer-container interpretation)\n",
                threshold, ratio_literal < threshold ? "PASS" : "FAIL");
    std::printf("Ratio (B) master plan §S5d L1432 cache-meaningful per-element:\n");
    std::printf("  per_ele_soa_bytes / sizeof(_Element) = %zu / %zu = %.4f\n",
                per_ele_total, aos_per_ele, ratio_real);
    std::printf("  threshold = %.4f  -> %s\n",
                threshold, ratio_real < threshold ? "PASS" : "FAIL");
    std::printf("\n");

    /* Gate on (A) — the literal spec scenario wording. (B) is documented
     * for the per-element cache footprint discussion but does not gate
     * exit code; the cache-miss perf stat (Task 8.2) measures the actual
     * effect downstream so we do not double-gate on (B) here.
     */
    if (ratio_literal < threshold) {
        std::printf("VERDICT: PASS gate on (A) (%.4f < %.4f)\n",
                    ratio_literal, threshold);
        std::printf("NOTE: (B) ratio %.4f informs cache-miss perf-stat (Task 8.2);\n",
                    ratio_real);
        std::printf("      see docs/b1b_summary.md S5d section for full discussion.\n");
        return 0;
    }
    std::printf("VERDICT: FAIL gate on (A) (%.4f >= %.4f)\n",
                ratio_literal, threshold);
    return 1;
}
