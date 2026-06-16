#!/usr/bin/env bash
# tools/bootstrap_check.sh
#
# Bootstrap diagnostic for SHUD-OpenMP: answers "why can't keliya (or any
# other local NWM case) run?" before the user spends 60s+ debugging by hand.
#
# Contract: openspec/changes/s0-baseline-lock/specs/bootstrap-verification/
#           spec.md (3 Requirements, 7 Scenarios). Covers the six most common
#           failure modes recorded in master plan §5 S0.1c:
#
#   1. SUNDIALS not installed              → hint:  ./SHUD/configure
#   2. SHUD/shud binary not built          → hint:  cd SHUD && make shud
#   3. Case directory not deployed         → hint:  deploy case data first
#   4. fix_case_paths not applied          → hint:  tools/fix_case_paths/fix_case_paths.sh <case>
#   5. NUM_OPENMP != 1 in cfg.para         → hint:  tools/fix_case_paths/fix_case_paths.sh <case>
#   6. case dir / output path not writable → hint:  chmod or check ownership
#
# Modes:
#   bootstrap_check.sh <case>     single-case mode (returns 0 if all PASS, 1 if any FAIL)
#   bootstrap_check.sh --all      bulk mode: benchmark + auxiliary (returns 0 if all PASS, 1 if any FAIL)
#   bootstrap_check.sh -h|--help  usage
#
# Bulk failures NEVER abort the batch (Scenario "Single failure doesn't abort
# bulk"); server-only cases ALWAYS print SKIPPED rather than failing
# (Scenario "server-only SKIPPED").
#
# Verbose: pass --verbose for the full per-check detail (path resolved, what
# was matched, etc.). Default is one terse line per check.

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
SHUD_ROOT="$REPO_ROOT/SHUD"
BASINS_ROOT="$SHUD_ROOT/Basins"
SUNDIALS_ROOT="$SHUD_ROOT/InstallSundials"
SHUD_BIN="$SHUD_ROOT/shud"

# Allow-listed case sets — mirror fix_case_paths.sh on purpose so that the
# two tools agree on what is local / server-only.
BENCHMARK_CASES=(keliya xinanjiang_upstream qinyijiang kashigeer qhh)
AUXILIARY_CASES=(tailanhe)
SERVER_ONLY_CASES=(heihe heihe_x4 heihe_x16)

VERBOSE=0

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

log()    { printf '%s\n' "$*"; }
verb()   { if [[ $VERBOSE -eq 1 ]]; then printf '    %s\n' "$*"; fi; }
warn()   { printf 'WARN: %s\n' "$*" >&2; }
err()    { printf 'ERROR: %s\n' "$*" >&2; }

pass_line() { printf '  [PASS] %-44s %s\n' "$1" "${2:-}"; }
fail_line() { printf '  [FAIL] %-44s %s\n' "$1" "${2:-}"; }
skip_line() { printf '  [SKIP] %-44s %s\n' "$1" "${2:-}"; }
hint_line() { printf '         hint: %s\n' "$1"; }

# -----------------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------------

in_array() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        [[ "$item" == "$needle" ]] && return 0
    done
    return 1
}

classify_case() {
    local case_name="$1"
    if in_array "$case_name" "${BENCHMARK_CASES[@]}"; then
        printf 'benchmark'
    elif in_array "$case_name" "${AUXILIARY_CASES[@]}"; then
        printf 'auxiliary'
    elif in_array "$case_name" "${SERVER_ONLY_CASES[@]}"; then
        printf 'server-only'
    else
        printf 'unknown'
    fi
}

# -----------------------------------------------------------------------------
# Path resolution for a case (same contract as fix_case_paths.sh)
# Sets: CASE_DIR, INPUT_DIR, PROJECT_NAME, TSD_FORC, CFG_PARA,
#       FORCING_DIR, FORCING_DIR_KIND
# Returns 0 on success; non-zero if structure is malformed.
# -----------------------------------------------------------------------------
resolve_case_paths() {
    local case_name="$1"

    CASE_DIR="$BASINS_ROOT/$case_name"
    INPUT_DIR=""
    PROJECT_NAME=""
    TSD_FORC=""
    CFG_PARA=""
    FORCING_DIR=""
    FORCING_DIR_KIND="missing"

    [[ -d "$CASE_DIR" ]]       || return 1
    [[ -d "$CASE_DIR/input" ]] || return 2

    # Locate unique .tsd.forc under input/*/
    local matches=()
    local f
    while IFS= read -r -d '' f; do
        matches+=("$f")
    done < <(find "$CASE_DIR/input" -mindepth 2 -maxdepth 2 -name '*.tsd.forc' -print0 2>/dev/null)

    if [[ ${#matches[@]} -eq 0 ]]; then
        return 3
    fi
    if [[ ${#matches[@]} -gt 1 ]]; then
        return 4
    fi

    TSD_FORC="${matches[0]}"
    INPUT_DIR="$(dirname "$TSD_FORC")"
    PROJECT_NAME="$(basename "$INPUT_DIR")"
    CFG_PARA="$INPUT_DIR/${PROJECT_NAME}.cfg.para"

    # Forcing dir: canonical or 'focing/' upstream typo.
    if [[ -d "$CASE_DIR/forcing" ]]; then
        FORCING_DIR="$CASE_DIR/forcing"
        FORCING_DIR_KIND="canonical"
    elif [[ -d "$CASE_DIR/focing" ]]; then
        FORCING_DIR="$CASE_DIR/focing"
        FORCING_DIR_KIND="non-canonical-focing"
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Global checks (no case argument)
# -----------------------------------------------------------------------------

# check_sundials → 0 PASS / 1 FAIL
check_sundials() {
    local header_ok=0 lib_ok=0
    if [[ -f "$SUNDIALS_ROOT/include/sundials/sundials_config.h" ]]; then
        header_ok=1
    fi
    # Match both .a and .so / .dylib (platform-dependent).
    if compgen -G "$SUNDIALS_ROOT/lib/libsundials_cvode.*" >/dev/null 2>&1; then
        lib_ok=1
    fi

    if [[ $header_ok -eq 1 && $lib_ok -eq 1 ]]; then
        pass_line "sundials install present" "header + cvode lib found"
        verb "header: $SUNDIALS_ROOT/include/sundials/sundials_config.h"
        # Pick the first matching cvode lib filename for verbose output.
        local _libs=("$SUNDIALS_ROOT"/lib/libsundials_cvode.*)
        verb "lib   : ${_libs[0]}"
        return 0
    fi
    fail_line "sundials install present" "header_ok=$header_ok lib_ok=$lib_ok"
    hint_line "run: ./SHUD/configure"
    return 1
}

# check_shud_binary → 0 PASS / 1 FAIL
check_shud_binary() {
    if [[ -x "$SHUD_BIN" ]]; then
        pass_line "SHUD/shud binary built" "$(basename "$SHUD_BIN")"
        verb "binary: $SHUD_BIN"
        return 0
    fi
    fail_line "SHUD/shud binary built" "missing or not executable"
    hint_line "run: cd SHUD && make shud"
    return 1
}

# -----------------------------------------------------------------------------
# Per-case checks
# -----------------------------------------------------------------------------

# check_case_dir <case>  → 0 PASS / 1 FAIL
check_case_dir() {
    local case_name="$1"
    if [[ -d "$BASINS_ROOT/$case_name/input" ]]; then
        pass_line "case directory present" "Basins/$case_name/input/"
        return 0
    fi
    fail_line "case directory present" "Basins/$case_name/input/ missing"
    hint_line "deploy case data first (see CLAUDE.md cross-endpoint table)"
    return 1
}

# check_tsd_forc_local <case>  → 0 PASS / 1 FAIL
# Expects line 2 of <project>.tsd.forc to be the absolute path of the case's
# forcing dir as resolved on THIS host. If it's still a server path, we FAIL
# and point at fix_case_paths.sh.
check_tsd_forc_local() {
    local case_name="$1"
    if [[ -z "${TSD_FORC:-}" || ! -f "$TSD_FORC" ]]; then
        fail_line "tsd.forc local path" "tsd.forc not located"
        hint_line "case structure malformed; check Basins/$case_name/input/<project>/"
        return 1
    fi

    local expected="" actual=""
    if [[ "$FORCING_DIR_KIND" == "missing" ]]; then
        fail_line "tsd.forc local path" "forcing dir missing (no forcing/ or focing/)"
        hint_line "deploy forcing data into Basins/$case_name/forcing/"
        return 1
    fi
    expected="$(cd "$FORCING_DIR" && pwd -P)"
    actual="$(awk 'NR==2 {print; exit}' "$TSD_FORC")"

    if [[ "$actual" == "$expected" ]]; then
        pass_line "tsd.forc line 2 = local forcing dir" "$(basename "$FORCING_DIR")/"
        verb "expected: $expected"
        verb "actual  : $actual"
        return 0
    fi

    fail_line "tsd.forc line 2 = local forcing dir" "path mismatch"
    verb "expected: $expected"
    verb "actual  : $actual"
    hint_line "run: tools/fix_case_paths/fix_case_paths.sh $case_name"
    return 1
}

# check_num_openmp <case>  → 0 PASS / 1 FAIL
# Requires exactly one `^NUM_OPENMP\s+1$` line (TAB or spaces both fine).
# Multiple NUM_OPENMP lines → FAIL with dedupe hint (also handled by fixup
# tool).
check_num_openmp() {
    local case_name="$1"
    if [[ -z "${CFG_PARA:-}" || ! -f "$CFG_PARA" ]]; then
        fail_line "NUM_OPENMP == 1 in cfg.para" "cfg.para not located"
        hint_line "case structure malformed; check Basins/$case_name/input/<project>/"
        return 1
    fi

    local matches
    matches="$(grep -nE '^NUM_OPENMP[[:space:]]+1[[:space:]]*$' "$CFG_PARA" || true)"
    local match_count
    match_count="$(printf '%s' "$matches" | grep -c . || true)"

    if [[ "$match_count" -eq 1 ]]; then
        # Also verify no second NUM_OPENMP definition lurks.
        local all_count
        all_count="$(grep -cE '^NUM_OPENMP[[:space:]]' "$CFG_PARA" || true)"
        if [[ "$all_count" -gt 1 ]]; then
            fail_line "NUM_OPENMP == 1 in cfg.para" "duplicate NUM_OPENMP lines ($all_count)"
            hint_line "run: tools/fix_case_paths/fix_case_paths.sh $case_name (dedupes)"
            return 1
        fi
        pass_line "NUM_OPENMP == 1 in cfg.para" "${matches}"
        return 0
    fi

    fail_line "NUM_OPENMP == 1 in cfg.para" "expected exactly 1 match, got $match_count"
    hint_line "run: tools/fix_case_paths/fix_case_paths.sh $case_name"
    return 1
}

# check_output_writable <case>  → 0 PASS / 1 FAIL
# SHUD writes to <case>/output/<project>/ at runtime; the case dir must be
# writable. We do not pre-create output/; SHUD does that itself.
check_output_writable() {
    local case_name="$1"
    if [[ -z "${CASE_DIR:-}" ]]; then
        fail_line "case dir writable" "case dir not resolved"
        return 1
    fi
    if [[ -w "$CASE_DIR" ]]; then
        pass_line "case dir writable (output/ creatable)" "$(basename "$CASE_DIR")/"
        verb "case dir: $CASE_DIR"
        return 0
    fi
    fail_line "case dir writable (output/ creatable)" "$CASE_DIR not writable"
    hint_line "chmod u+w $CASE_DIR  (or check ownership)"
    return 1
}

# -----------------------------------------------------------------------------
# Per-case driver
# -----------------------------------------------------------------------------

# run_case_checks <case>  → prints case header + check lines, sets CASE_RESULT
# CASE_RESULT ∈ { PASS | FAIL | SKIPPED }
run_case_checks() {
    local case_name="$1"
    local kind
    kind="$(classify_case "$case_name")"

    log ""
    log "==> $case_name ($kind)"

    # Server-only → always SKIPPED, regardless of whether dir exists.
    if [[ "$kind" == "server-only" ]]; then
        skip_line "server-only case" "do not deploy on this endpoint"
        CASE_RESULT="SKIPPED"
        return 0
    fi

    # Unknown: in bulk (--all) mode tolerate as SKIPPED (leftover dir under
    # Basins/ should not break the batch). In single-case mode treat as
    # FAIL because the most common cause is a user typo of the case name.
    if [[ "$kind" == "unknown" ]]; then
        if [[ "${BULK_MODE:-0}" == "1" ]]; then
            skip_line "unknown case" "not in BENCHMARK/AUXILIARY/SERVER_ONLY lists"
            warn "case '$case_name' not in any allow-list; skipping diagnostic"
            CASE_RESULT="SKIPPED"
        else
            fail_line "unknown case" "not in BENCHMARK/AUXILIARY/SERVER_ONLY lists"
            hint_line "did you mistype? known cases: $(printf '%s ' "${BENCHMARK_CASES[@]}" "${AUXILIARY_CASES[@]}")"
            CASE_RESULT="FAIL"
        fi
        return 0
    fi

    # Case-dir check must come first; later checks depend on structure.
    if ! check_case_dir "$case_name"; then
        CASE_RESULT="FAIL"
        return 0   # do not propagate non-zero up; FAIL is recorded in CASE_RESULT
    fi

    # Resolve paths; if structure is malformed report FAIL and bail on
    # remaining per-case checks.
    if ! resolve_case_paths "$case_name"; then
        local rc=$?
        case "$rc" in
            3) fail_line "tsd.forc locatable" "no .tsd.forc found under input/*/" ;;
            4) fail_line "tsd.forc locatable" "multiple .tsd.forc matched (ambiguous)" ;;
            *) fail_line "tsd.forc locatable" "resolve_case_paths rc=$rc" ;;
        esac
        hint_line "verify Basins/$case_name/input/<project>/<project>.tsd.forc"
        CASE_RESULT="FAIL"
        return 0
    fi
    verb "project: $PROJECT_NAME"
    verb "tsd.forc: ${TSD_FORC#"$REPO_ROOT"/}"
    verb "cfg.para: ${CFG_PARA#"$REPO_ROOT"/}"
    verb "forcing : ${FORCING_DIR#"$REPO_ROOT"/} ($FORCING_DIR_KIND)"

    local rc=0
    check_tsd_forc_local "$case_name" || rc=1
    check_num_openmp     "$case_name" || rc=1
    check_output_writable "$case_name" || rc=1

    if [[ $rc -eq 0 ]]; then
        CASE_RESULT="PASS"
    else
        CASE_RESULT="FAIL"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Bulk driver
# -----------------------------------------------------------------------------

# Build the bulk list: every BENCHMARK + AUXILIARY case (always included);
# plus every SERVER_ONLY case (so the bulk run can SKIPPED-report them); plus
# any unknown dir present under Basins/ (so user gets a warning).
build_bulk_list() {
    BULK_CASES=()

    local c
    for c in "${BENCHMARK_CASES[@]}"; do
        BULK_CASES+=("$c")
    done
    for c in "${AUXILIARY_CASES[@]}"; do
        BULK_CASES+=("$c")
    done
    for c in "${SERVER_ONLY_CASES[@]}"; do
        BULK_CASES+=("$c")
    done

    # Add unknown dirs present locally (so they're surfaced, not silent).
    local d name
    if [[ -d "$BASINS_ROOT" ]]; then
        for d in "$BASINS_ROOT"/*/; do
            [[ -d "$d" ]] || continue
            name="$(basename "$d")"
            case "$name" in
                .*) continue ;;
            esac
            if ! in_array "$name" "${BULK_CASES[@]}"; then
                BULK_CASES+=("$name")
            fi
        done
    fi
}

# -----------------------------------------------------------------------------
# Top-level dispatch
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage:
  bootstrap_check.sh <case>          diagnose a single local case
  bootstrap_check.sh --all           diagnose all local benchmark + auxiliary
                                     cases; server-only cases reported SKIPPED
  bootstrap_check.sh --verbose ...   add per-check detail

Allow-listed cases:
  benchmark  : $(printf '%s ' "${BENCHMARK_CASES[@]}")
  auxiliary  : $(printf '%s ' "${AUXILIARY_CASES[@]}")
  server-only: $(printf '%s ' "${SERVER_ONLY_CASES[@]}")

Exit code:
  0  every checked case PASS (SKIPPED counts as neutral)
  1  one or more checked cases FAIL, or arg/usage error
EOF
}

main() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 1
    fi

    local case_arg=""
    local all=0
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --all)         all=1; shift ;;
            --verbose|-v)  VERBOSE=1; shift ;;
            -h|--help)     usage; exit 0 ;;
            --)            shift; break ;;
            -*)            err "unknown option: $1"; usage >&2; exit 1 ;;
            *)             case_arg="$1"; shift ;;
        esac
    done

    if [[ $all -eq 1 && -n "$case_arg" ]]; then
        err "--all and a positional case argument are mutually exclusive"
        usage >&2
        exit 1
    fi
    if [[ $all -eq 0 && -z "$case_arg" ]]; then
        err "give a case name or pass --all"
        usage >&2
        exit 1
    fi

    if [[ ! -d "$BASINS_ROOT" ]]; then
        err "Basins root not found: $BASINS_ROOT"
        exit 1
    fi

    log "SHUD-OpenMP bootstrap diagnostic"
    log "repo: $REPO_ROOT"
    log "host: $(uname -s) $(uname -m)"
    log ""
    log "[global checks]"

    local global_fail=0
    check_sundials   || global_fail=1
    check_shud_binary || global_fail=1

    local total_pass=0 total_fail=0 total_skip=0
    local CASE_RESULT
    local failed_names=() skipped_names=()

    if [[ $all -eq 1 ]]; then
        BULK_MODE=1
        build_bulk_list
        local c
        for c in "${BULK_CASES[@]}"; do
            CASE_RESULT="FAIL"
            # Bulk discipline: even if a single case throws inside its
            # function, never abort the batch. run_case_checks returns 0
            # by design and writes its verdict into CASE_RESULT, but
            # belt-and-suspenders: catch any unexpected non-zero.
            run_case_checks "$c" || true
            case "$CASE_RESULT" in
                PASS)    total_pass=$((total_pass+1)) ;;
                FAIL)    total_fail=$((total_fail+1)); failed_names+=("$c") ;;
                SKIPPED) total_skip=$((total_skip+1)); skipped_names+=("$c") ;;
            esac
        done
    else
        CASE_RESULT="FAIL"
        run_case_checks "$case_arg" || true
        case "$CASE_RESULT" in
            PASS)    total_pass=1 ;;
            FAIL)    total_fail=1; failed_names+=("$case_arg") ;;
            SKIPPED) total_skip=1; skipped_names+=("$case_arg") ;;
        esac
    fi

    log ""
    log "===== summary ====="
    log "global PASS: $([[ $global_fail -eq 0 ]] && echo yes || echo no)"
    log "cases  PASS: $total_pass"
    log "cases  FAIL: $total_fail"
    log "cases  SKIP: $total_skip"
    if [[ ${#failed_names[@]} -gt 0 ]]; then
        log "failed: $(printf '%s ' "${failed_names[@]}")"
    fi
    if [[ ${#skipped_names[@]} -gt 0 ]]; then
        log "skipped: $(printf '%s ' "${skipped_names[@]}")"
    fi

    if [[ $global_fail -ne 0 || $total_fail -ne 0 ]]; then
        exit 1
    fi
    exit 0
}

main "$@"
