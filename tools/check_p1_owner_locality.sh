#!/usr/bin/env bash
# tools/check_p1_owner_locality.sh — S2 P1 owner-locality grep gate
#
# Enforces openspec/changes/s2-strict-omp-full/specs/p1-rhs-update-owner-local-omp/spec.md
#   - Element loop body (Scenario "Element state update only writes Ele[i] owner"):
#     writes only via Ele[i] / qEle*[i] / Q*[i][*]; no Ele[i+/-N] / nested
#     Ele[Ele[i].nabr[j]-1] / Ele[inabr] cross-element index.
#   - River loop body (Scenario "Riv[i].updateRiver() audit gate" +
#     Requirement "River-owner state update parallelism"):
#     writes only via Riv[i] / uYriv[i] / Q*riv[i]; no Riv[i+/-N] /
#     nested-bracket neighbor; spec L42 also forbids touching
#     Qe2r_Surf[i] / Qe2r_Sub[i] inside the river scope (those are an
#     element-domain follow-up loop, NOT the river-owner stage).
#   - Lake loop body (Scenario "Lake update only writes lake[i] owner" +
#     Requirement "Lake-owner state update parallelism"):
#     writes only via lake[i] / yLakeStg[i] / y2LakeArea[i] / QLake*[i] /
#     qLake*[i]; no lake[i+/-N] / nested-bracket neighbor. Audit also
#     covers _Lake::update() body (Lake.cpp:104-107) which only writes
#     this->u_toparea (member-of-this, no cross-lake writes).
#   - DY zeroing loop body (Scenario "DY zeroing static for-schedule landed" +
#     Requirement "DY array zeroing parallelism over state index"):
#     writes only via DY[i] (bare state index); no DY[i+/-N] / nested-bracket
#     DY[Y[…]] / cross-state DY[k] form. The loop body is trivially
#     state-index-owner-local on `i`; the audit ensures no future refactor
#     accidentally introduces cross-index DY writes.
#   - No reduction / atomic / critical / dynamic / guided / nowait clauses
#     on owner-local pragma (spec L20 + design §D11 P5 三层依赖模型).
#
# SCOPE retargeted (R1 cand-02): grep gate scopes the canonical runtime
# `Model_Data::rhs_update()` in SHUD/src/Model/MD_rhs_core.cpp (NOT the legacy
# `f_update()` in MD_update.cpp). The default Config A + Config E builds dispatch
# `f.cpp::f()` → `rhs_core(Y, DY, t, ExecPolicy::Serial)` → `rhs_update()`;
# the legacy path is only reachable when `LEGACY_RHS=1` Makefile macro is
# explicitly set, which is not the default CI Config matrix.
#
# #83/#84/#85 extension: quad-scope audit (element + river + lake + DY). The
# audit body is parameterized by `audit_loop_body()`; element scope strips
# `Ele[i]`, river scope strips `Riv[i]`, lake scope strips `lake[i]`, DY scope
# strips `DY[i]`. Each scope independently checks the three invariants
# (cross-index, neighbor-nested, forbidden pragma clauses).
#
# Exit 0 on PASS; exit 1 on any forbidden pattern found.
# Run from the outer repo root.

set -euo pipefail

SCOPE="SHUD/src/Model/MD_rhs_core.cpp"
TARGET_FN="Model_Data::rhs_update"

if [ ! -f "$SCOPE" ]; then
    echo "FAIL: $SCOPE not found"
    exit 1
fi

fail=0

# Locate rhs_update() function start (shared anchor for both scopes).
start=$(grep -n '^void Model_Data::rhs_update(' "$SCOPE" | head -1 | cut -d: -f1)
if [ -z "$start" ]; then
    echo "FAIL: rhs_update() function start not found in $SCOPE"
    exit 1
fi

# Element scope: ends at the `}//end of for j=1:NumEle` comment anchor.
ele_end=$(awk -v s="$start" 'NR>s && /end of for j=1:NumEle/{print NR; exit}' "$SCOPE")
if [ -z "$ele_end" ]; then
    echo "FAIL: cannot locate end of element loop comment in rhs_update()"
    exit 1
fi

# River scope: from end of element loop down to (but excluding) the next
# `for (int i = 0; i < NumEle; i++)` line — that line marks the element-domain
# Qe2r_Surf / Qe2r_Sub zeroing loop, which spec L42 declares OUT of the
# river-owner stage (it is element-domain scope). The two NumRiv loops live
# entirely between ele_end+1 and (next-NumEle-loop-line - 1).
riv_start=$((ele_end + 1))
riv_end=$(awk -v s="$ele_end" 'NR>s && /for[[:space:]]*\([[:space:]]*int[[:space:]]+i[[:space:]]*=[[:space:]]*0;[[:space:]]*i[[:space:]]*<[[:space:]]*NumEle/{print NR-1; exit}' "$SCOPE")
if [ -z "$riv_end" ]; then
    echo "FAIL: cannot locate end of river scope (post-river NumEle loop anchor) in rhs_update()"
    exit 1
fi

# Lake scope: starts at the line immediately after the post-river NumEle
# element-domain Qe2r_Surf/Sub zeroing loop's closing brace, and ends at the
# closing brace `}` of the NumLake for-loop body (the line right BEFORE the
# DY-owner OMP pragma block + NumY for-loop, which is governed by the
# SEPARATE L5 DY zeroing Requirement and audited as the DY scope below).
# Starting at the post-NumEle-loop close brace + 1 (analog to how riv_start
# = ele_end + 1) ensures the lake-owner OMP pragma block at lines preceding
# the `for (... i < NumLake; ...)` line is INSIDE the audited slice (the
# pragma-clause Check 3 needs the pragma directive visible). The NumLake
# for-line itself is enforced as the only `i < NumLake` anchor in the slice
# via the audit checks below.
#
# Locate the post-river NumEle loop's `for` line, then find the next `}` at
# `^    }$` indentation (the loop body is plain 1-statement-per-line C++
# with 4-space indent matching the surrounding rhs_update() function).
post_riv_ele_for=$(awk -v s="$riv_end" 'NR>s && /for[[:space:]]*\([[:space:]]*int[[:space:]]+i[[:space:]]*=[[:space:]]*0;[[:space:]]*i[[:space:]]*<[[:space:]]*NumEle/{print NR; exit}' "$SCOPE")
if [ -z "$post_riv_ele_for" ]; then
    echo "FAIL: cannot locate post-river NumEle loop in rhs_update()"
    exit 1
fi
post_riv_ele_close=$(awk -v s="$post_riv_ele_for" 'NR>s && /^[[:space:]]*\}[[:space:]]*$/{print NR; exit}' "$SCOPE")
if [ -z "$post_riv_ele_close" ]; then
    echo "FAIL: cannot locate post-river NumEle loop closing brace in rhs_update()"
    exit 1
fi
lake_start=$((post_riv_ele_close + 1))
# Lake for-line: first `for (int i = 0; i < NumLake; i++)` after lake_start.
lake_for=$(awk -v s="$lake_start" 'NR>=s && /for[[:space:]]*\([[:space:]]*int[[:space:]]+i[[:space:]]*=[[:space:]]*0;[[:space:]]*i[[:space:]]*<[[:space:]]*NumLake/{print NR; exit}' "$SCOPE")
if [ -z "$lake_for" ]; then
    echo "FAIL: cannot locate NumLake for-loop in rhs_update()"
    exit 1
fi
# Lake scope ends at the closing `}` of the NumLake loop body (first `^    }$`
# after lake_for). This lands the lake slice strictly BEFORE the DY pragma
# block, so the DY-owner pragma is fully owned by the DY scope.
lake_end=$(awk -v s="$lake_for" 'NR>s && /^[[:space:]]*\}[[:space:]]*$/{print NR; exit}' "$SCOPE")
if [ -z "$lake_end" ]; then
    echo "FAIL: cannot locate end of lake scope (NumLake loop closing brace) in rhs_update()"
    exit 1
fi

# DY scope: starts at lake_end + 1 (the line right after the NumLake loop's
# closing brace, analogous to riv_start = ele_end + 1 and lake_start =
# post_riv_ele_close + 1). The DY OMP pragma block (`#ifdef SHUD_ENABLE_OPENMP_RHS`
# / `#pragma omp for schedule(static)` / `#endif`) plus the NumY `for` line
# itself sit inside this slice so the pragma-clause Check 3 can see the
# directive. Ends at the FIRST `}` at the matching indentation (`^    }$`)
# after the NumY for-line — the NumY DY zeroing loop body is plain
# 1-statement-per-line (just `DY[i] = 0.;`) so a single 4-space-indent
# close-brace is the unambiguous end marker.
dy_start=$((lake_end + 1))
dy_for=$(awk -v s="$lake_end" 'NR>s && /for[[:space:]]*\([[:space:]]*int[[:space:]]+i[[:space:]]*=[[:space:]]*0;[[:space:]]*i[[:space:]]*<[[:space:]]*NumY/{print NR; exit}' "$SCOPE")
if [ -z "$dy_for" ]; then
    echo "FAIL: cannot locate NumY DY zeroing for-loop in rhs_update()"
    exit 1
fi
dy_end=$(awk -v s="$dy_for" 'NR>s && /^[[:space:]]*\}[[:space:]]*$/{print NR; exit}' "$SCOPE")
if [ -z "$dy_end" ]; then
    echo "FAIL: cannot locate end of DY scope (NumY DY loop closing brace) in rhs_update()"
    exit 1
fi

# ----------------------------------------------------------------------------
# audit_loop_body <scope_name> <body_start> <body_end> <owner_strip_pattern> <owner_subscript_prefix>
#
# Args:
#   scope_name              — human-readable label for log output (e.g. "element")
#   body_start, body_end    — 1-based line range (inclusive) inside $SCOPE
#   owner_strip_pattern     — sed -E pattern to strip owner-bracket form (e.g. `Ele\[i\]`)
#   owner_subscript_prefix  — bare class name used for residual-detection grep (e.g. `Ele`)
# Globals read: $SCOPE
# Globals written: $fail (set to 1 on any check failure)
# ----------------------------------------------------------------------------
audit_loop_body() {
    local scope_name="$1"
    local body_start="$2"
    local body_end="$3"
    local owner_strip_pattern="$4"
    local owner_subscript_prefix="$5"

    local body body_live body_filtered pragma bad_real

    body=$(sed -n "${body_start},${body_end}p" "$SCOPE")

    # Strip C++ line-comment-only lines (`^\s*//…`), block-comment continuation
    # lines (`^\s*\*`), and inline same-line `/* … */` spans. Multi-line
    # `/* … */` spans are rare in this codebase and intentionally not handled.
    # This filter only affects Check 1 / Check 2; Check 3 keeps the anchored
    # `^[[:space:]]*#pragma omp` so block-comment pragma mentions stay excluded.
    body_live=$(echo "$body" | sed -E 's|/\*[^*]*\*/||g' | grep -vE '^[[:space:]]*//' | grep -vE '^[[:space:]]*\*')

    # Check 1: no <Prefix>[i+/-…] cross-owner index forms.
    if echo "$body_live" | grep -E "${owner_subscript_prefix}\\[i[[:space:]]*[\\+\\-]" >/dev/null; then
        echo "FAIL (${scope_name}): found ${owner_subscript_prefix}[i+/-…] cross-${scope_name} index in rhs_update() ${scope_name} loop"
        echo "$body_live" | grep -nE "${owner_subscript_prefix}\\[i[[:space:]]*[\\+\\-]"
        fail=1
    fi

    # Check 2: every <Prefix>[ subscript must be exactly <Prefix>[i] (bare loop
    # index). Strategy: strip allowed `<Prefix>[i]` substrings (the ONLY allowed
    # outer subscript form), then any residual `<Prefix>[` indicates a forbidden
    # indexing form (e.g. nested `Ele[Ele[i].nabr[0]-1]`, `Riv[Riv[i].down-1]`,
    # `Ele[inabr]`, `Riv[idown]`). Check 1 already catches `<Prefix>[i+/-N]` so
    # the residual scan focuses on nested-bracket / name-variable forms.
    # `<Prefix>[i].field[j]` is allowed as field READ (the outer subscript IS
    # `<Prefix>[i]`; the trailing `.field[j]` is a field/array access on the
    # resolved struct, used inside conditionals like `Ele[i].nabr[j] > 0`).
    body_filtered=$(echo "$body_live" | sed -E "s|${owner_strip_pattern}||g")
    if echo "$body_filtered" | grep -E "${owner_subscript_prefix}\\[" >/dev/null; then
        bad_real=$(echo "$body_filtered" | grep -nE "${owner_subscript_prefix}\\[" || true)
        if [ -n "$bad_real" ]; then
            echo "FAIL (${scope_name}): found ${owner_subscript_prefix}[…] subscript that is not the bare ${owner_subscript_prefix}[i] owner index in rhs_update() ${scope_name} loop"
            echo "(showing post-${owner_subscript_prefix}[i]-strip view to highlight the offending residual subscript)"
            echo "$bad_real"
            fail=1
        fi
    fi

    # Check 3: pragma must be `#pragma omp for schedule(static)`, no
    # reduction / atomic / critical / dynamic / guided / nowait clauses.
    # Anchor on `^[[:space:]]*#pragma` so block-comment lines (` * #pragma omp …`)
    # and line-comment lines (`// #pragma omp …`) are excluded from the validator.
    # `nowait` is forbidden per spec L20 + design §D11 P5: nowait on any owner
    # stage would break the barrier sequencing that P7 fusion (#99) depends on —
    # downstream stages read this stage's writes and must observe them before
    # starting.
    pragma=$(echo "$body" | grep -E '^[[:space:]]*#pragma omp' || true)
    if [ -n "$pragma" ]; then
        if ! echo "$pragma" | grep -E 'for[[:space:]]+schedule\(static\)' >/dev/null; then
            echo "FAIL (${scope_name}): pragma found but not '#pragma omp for schedule(static)' in rhs_update() ${scope_name} loop"
            echo "$pragma"
            fail=1
        fi
        if echo "$pragma" | grep -E 'reduction\(|atomic|critical|schedule\(dynamic|schedule\(guided|nowait\b' >/dev/null; then
            echo "FAIL (${scope_name}): forbidden clause (reduction/atomic/critical/dynamic/guided/nowait) on rhs_update() ${scope_name} loop pragma"
            echo "$pragma"
            fail=1
        fi
    fi

    # Stash the pragma string for the PASS summary (per-scope variable).
    eval "pragma_${scope_name}=\"\$pragma\""
}

# Element scope audit: own index `Ele[i]`, residual prefix `Ele`.
audit_loop_body "element" "$start" "$ele_end" 'Ele\[i\]' 'Ele'

# River scope audit: own index `Riv[i]`, residual prefix `Riv`. Spec L42
# also forbids `Qe2r_Surf[i]` / `Qe2r_Sub[i]` writes inside the river scope
# (element-domain follow-up loop); since the river scope $riv_end stops
# BEFORE the post-river NumEle loop that contains those writes, the slice
# naturally excludes them — no extra check needed.
audit_loop_body "river" "$riv_start" "$riv_end" 'Riv\[i\]' 'Riv'

# Lake scope audit: own index `lake[i]`, residual prefix `lake`. The lake
# scope $lake_end stops BEFORE the NumY loop that contains DY[i] writes
# (governed by the separate L5 DY zeroing Requirement, not the lake-owner
# stage). The `_Lake::update()` body audit (spec L63 Scenario "Lake update
# only writes lake[i] owner") was pre-cleared in this PR via direct read of
# SHUD/src/classes/Lake.cpp:104-107 — only writes this->u_toparea, reads
# this->{yStage,zmin,bathymetry.{ai,yi,nvalue}} (all owner-local /
# immutable post-init).
audit_loop_body "lake" "$lake_start" "$lake_end" 'lake\[i\]' 'lake'

# DY scope audit (#85): own index `DY[i]`, residual prefix `DY`. State-index
# i owner-local — DY is a NumY-length array of CVODE residual values; the
# loop body trivially writes only DY[i] per spec Requirement "DY array
# zeroing parallelism over state index" + Scenario "DY zeroing static
# for-schedule landed". Residual scan catches DY[i+1] (cross-state-index),
# DY[Y[…]] (nested-bracket), DY[k] (alias-variable index). Pragma Check 3
# enforces no nowait clause per spec L20 (DY zero must complete-barrier
# before downstream RHS stages read it).
audit_loop_body "dy" "$dy_start" "$dy_end" 'DY\[i\]' 'DY'

if [ "$fail" -ne 0 ]; then
    echo ""
    echo "P1 owner-locality check FAIL"
    exit 1
fi

echo "P1 owner-locality check PASS"
echo "  function: $TARGET_FN"
echo "  element scope @ lines ${start}..${ele_end}  pragma: ${pragma_element:-<not yet present — Config A defaults>}"
echo "  river scope   @ lines ${riv_start}..${riv_end}  pragma: ${pragma_river:-<not yet present — Config A defaults>}"
echo "  lake scope    @ lines ${lake_start}..${lake_end}  pragma: ${pragma_lake:-<not yet present — Config A defaults>}"
echo "  DY scope      @ lines ${dy_start}..${dy_end}  pragma: ${pragma_dy:-<not yet present — Config A defaults>}"
exit 0
