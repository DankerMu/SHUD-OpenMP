# B1a vs B1b diff report — S6b.2 (lake formula audit — SKIP-default path)

**Fix ID**: S6b.2.1 (SKIP-implementation row; companion to the S6b.2 audit row already in `B1b_CHANGELOG.md`)
**Disposition**: **SKIP** — no code change to `SHUD/src/ModelData/MD_ElementFlux.cpp`
**Audit issue**: [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) (S2.17 lake formula PI review) — **OPEN, no PI sign-off received**
**Implementation issue**: [#186](https://github.com/DankerMu/SHUD-OpenMP/issues/186) (S6b.2 conditional fix-or-skip)
**Follow-up issue**: [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205) (SoA/AoS sync drift surfaced during the audit — out of scope for B1b; queued for P-strict pre-req)
**Evidence pack**: [`docs/b1b/s217_lake_formula_audit.md`](../s217_lake_formula_audit.md) (merged at PR #204)
**Status**: **zero-impact (no code change) — CONDITIONAL ship per master plan §S6b L1497 + C8 forward-compatibility**

This report fulfils the spec.md L61-63 Scenario "S6b.2 跳过时 diff report 仍存在" literal contract:

> WHEN S6b.2 跳过（S2.17 审查为"无修改"）
> THEN `docs/diff_reports/B1a_vs_B1b_diff_s6b_2.md` 仍生成，内容 = "S2.17 reviewed as correct; no code change; no diff"

## Disposition statement

**S2.17 was reviewed as an evidence-pack only; no PI sign-off received before this PR; B1b ships with the current lake-branch formula UNCHANGED per master plan §S6b L1497 + C8 forward-compatibility.**

Master plan §S6b L1497 reads (verbatim):

> 快速路径：…S6b.1（AccTemperature 除零）只在 cryosphere 启用且前 1440 分钟触发，S6b.2（lake 公式）**可能审查后不需要改**。

The evidence pack [`docs/b1b/s217_lake_formula_audit.md`](../s217_lake_formula_audit.md) §E (merged at PR #204) explicitly defers the E1/E2 verdict to the SHUD-upstream PI per `specs/s6b-bugfix-application/spec.md` L23. The Phase-1 audit author is **not** a PI delegate; design.md Open Q1 (delegate qualification governance) remains open. Absent a PI directive `S2.17: formula needs fix` arriving on #185 before the S6c (#188-#190) capstone, the de-facto path is **SKIP** — no patch is applied to `SHUD/src/ModelData/MD_ElementFlux.cpp:147` and the bitwise contract against `B1a-tag` on `qhh / heihe / heihe_x4` is preserved by construction.

This ship is **CONDITIONAL** per master plan C8 ("永不 break userspace"): if the PI later directs E1 (formula needs fix), the C8 forward-compatibility convention applies — a follow-up `B1c-tag` would stack the fix on top of B1b without force-updating `B1b-tag` (D11 lock honoured). This is **NOT** a signed-off E2 and does **NOT** satisfy design.md D9 fast-path trigger #2.

## Affected cases

Per evidence pack §D.1 the lake-branch formula at `MD_ElementFlux.cpp:147` is reachable only on cases with `lakeon == 1` (rSHUD-side `Riv[i].down <= -4` encoding + `<case>.lake.bathy / .lake.ic / .lake.sp` inputs present):

| Case | `lakeon` | Has `.lake.*` inputs? | Affected by this SKIP disposition? |
|---|---|---|---|
| keliya | 0 | NO | NO (lake branch dead code) |
| xinanjiang_upstream | 0 | NO | NO |
| qinyijiang | 0 | NO | NO |
| qhh (Mac) | 1 | YES | NO — code unchanged, output identical to B1a-tag |
| kashigeer | 0 | NO | NO (N/A per master plan) |
| tailanhe | 0 | NO | NO |
| heihe (server) | 1 | YES | NO — code unchanged, output identical to B1a-tag |
| heihe_x4 (server) | 1 | YES (inherited from heihe via rSHUD 4×) | NO — code unchanged, output identical to B1a-tag |

## Variance description

- **Diff vs `B1a-tag` on lake-affected `.dat` outputs**: **NONE** (no code change → byte-for-byte identical).
- **Affected `.dat` files**: 0 — the SKIP path leaves `MD_ElementFlux.cpp:147` `Kmean = 0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` byte-for-byte unchanged.
- **Bitwise outcome on `qhh.lakqrivin / .lakqrivout / .lakystage / .rivqdown` (Mac)** + **`heihe.rivqdown` / `heihe_x4.rivqdown` / `heihe_x4.eleygw` (server)**: identical SHA256 to `B1a-tag` worktree goldens (no re-run required since the source code path is unmodified).
- **Hypothetical magnitude IF the formula were "fixed"** (preserved for reference per evidence pack §D.2): for `K_bank ≈ K_lakebed` (typical SHUD mesh-classification convention), the arithmetic-vs-harmonic spread is < 10%; bank-only K would over-estimate flux by `(K_bank − K_lakebed) / (K_bank + K_lakebed)` × current flux, bounded < 30% for typical soil contrasts. None of these alternatives apply at SKIP path.

## Physical interpretation

Per evidence pack §B (full text in `docs/b1b/s217_lake_formula_audit.md`):

1. **Physics is standard at the macroscopic level** — the lake branch correctly implements lake-stage-as-aquifer-BC Darcy flux at the lake-bed seepage face, matching MODFLOW LAK7 (Merritt & Konikow 2000), ParFlow Lake, and PIHM 2.x conventions on `dh`, `grad`, `Ymean`, `A` terms.
2. **Averaging-formula consistency** — the same `0.5 * (hot.u_effKH[i] + hot.u_effKH[inabr])` arithmetic mean is used **byte-for-byte** in the non-lake GW lateral branch at L169 (evidence pack §C). The lake-branch formula is NOT a one-off oddity.
3. **Out-of-bounds risk already mitigated** — `assert(inabr >= 0)` at L137 closes the master plan §S2.17 R-1 recommendation; this is already in the live tree (pre-audit) and no additional code change is needed.
4. **`Ele[inabr].u_effKH` on a lake element — runtime-semantic subtlety** acknowledged in evidence pack §A.4 + §B.4: AoS `_Element::updateLakeElement()` writes `u_effKH = KsatH` but the runtime SoA mirror `hot.u_effKH[lake]` that `fun_Ele_sub` actually reads is set by the OUTER `updateforcing` general-element loop to the depth-weighted `effKH(...)` blend evaluated against the lake-element's aquifer state. This is the SoA/AoS sync drift tracked at issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205) — **out of scope for #186 / B1b ship**, queued for P-strict pre-req audit. The drift is bitwise-stable and deterministic, so the B1b bitwise contract is unaffected.
5. **Cost of any code-change S6b.2 would be high** — would break `B1a-tag` bitwise on `qhh / heihe / heihe_x4`; would require A4 `residual_deferred` classification or re-baselined goldens.

## Forward-compatibility note (master plan C8)

If the SHUD-upstream PI subsequently directs E1 on #185 (`S2.17: formula needs fix`), the fix lands as a **follow-up `B1c-tag`** stacked on top of `B1b-tag`. The `B1b-tag` lock (design.md D11) is **NOT** force-updated. Any later patched lake-edge formula:

- creates a new `S6b.2.2` row in `B1b_CHANGELOG.md` (or `B1c_CHANGELOG.md` per phasing policy at that time);
- generates a successor diff report (e.g. `B1b_vs_B1c_diff_s6b_2_2.md`) describing the bitwise delta on `qhh / heihe / heihe_x4`;
- carries the PI directive URL on #185 as the trigger reference.

This convention preserves the C8 contract ("永不 break userspace"): any later PI-mandated change can be stacked without rewriting B1b history.

## D9 fast-path status — BLOCKED

design.md D9 trigger #2 requires `S6b.2 = "审查为'无修改'" 跳过 fix` with a **signed** PI conclusion. The SKIP-default path implemented here is **unsigned** (PI sign-off pending on #185), so this trigger is **NOT** satisfied. D9 fast-path (B1a / B1b merge into `B1-tag`) remains **GATED** until PI sign-off arrives. S6c (#188-#190) proceeds with **separate** `B1a-tag` and `B1b-tag` per D9-not-satisfied, both subsequently subject to D11 force-update prohibition.

## Cross-references

- `openspec/changes/b1b-baseline-completion/specs/s6b-bugfix-application/spec.md` L29-31 (Scenario "审查结论已签字跳过修改" — partial match: text body satisfied, signature path replaced by C8 conditional ship) + L61-63 (Scenario "S6b.2 跳过时 diff report 仍存在" — fully satisfied by this report)
- `openspec/changes/b1b-baseline-completion/design.md` D8 (S6b 每个 fix 独立 commit + 独立 diff report — satisfied) + D9 (fast-path trigger #2 BLOCKED) + D11 (B1b-tag 不 force-update — honoured by the C8 follow-up plan above)
- `SHUD_openMP_master_plan.md` §S6b L1497 ("S6b.2 lake 公式可能审查后不需要改" — interpreted as a forecast, not a normative skip permission) + §C8 ("永不 break userspace") + §S2.17 (L1179-L1198) + §4.18 (L523-L541)
- Evidence pack: [`docs/b1b/s217_lake_formula_audit.md`](../s217_lake_formula_audit.md) (merged at PR #204; sections §A live formula citation, §B physics interpretation, §C non-lake-branch comparison, §D affected-cases table, §E verdict-pending evidence summary)
- Issue [#185](https://github.com/DankerMu/SHUD-OpenMP/issues/185) (S2.17 PI review) — remains **OPEN**; this PR does NOT close it
- Issue [#186](https://github.com/DankerMu/SHUD-OpenMP/issues/186) (S6b.2 fix-or-skip) — SKIP path implemented here
- Issue [#205](https://github.com/DankerMu/SHUD-OpenMP/issues/205) (SoA/AoS sync drift) — separate scope; tracked for P-strict pre-req
- `SHUD/B1b_CHANGELOG.md` S6b.2 section — S6b.2 audit row (pre-existing, from PR #204) + S6b.2.1 SKIP-implementation row (this PR)
- Companion precedent: [`docs/diff_reports/B1a_vs_B1b_diff_s6b_3_1.md`](B1a_vs_B1b_diff_s6b_3_1.md) (S6b.3.1 NOT-A-BUG zero-impact disposition; same single-row-CHANGELOG + diff-report-stub pattern)
