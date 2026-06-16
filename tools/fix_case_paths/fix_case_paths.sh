#!/usr/bin/env bash
# tools/fix_case_paths/fix_case_paths.sh
#
# Rewrites server-deployment paths in NWM cases under SHUD/Basins/<case>/ to
# the developer's local absolute paths, and forces NUM_OPENMP=1 for the
# serial baseline (S0.1b). See openspec/changes/s0-baseline-lock/specs/
# case-deployment-fixup/spec.md for the contract.
#
# Modes (--dry-run is an ORTHOGONAL flag that may combine with any mode):
#   fix_case_paths.sh <case>                      apply (mutates)
#   fix_case_paths.sh --dry-run <case>            preview apply (no mutation)
#   fix_case_paths.sh --restore <case>            restore from .orig (mutates)
#   fix_case_paths.sh --dry-run --restore <case>  preview restore (no mutation)
#   fix_case_paths.sh --restore --dry-run <case>  same; flag order is free
#   fix_case_paths.sh --all                       bulk apply across benchmark+auxiliary
#   fix_case_paths.sh --all --dry-run             preview bulk apply
#   fix_case_paths.sh --all --restore             bulk restore
#   fix_case_paths.sh --all --dry-run --restore   preview bulk restore
#
# Backups: on first apply, original files are snapshotted to <name>.orig;
# subsequent applies re-restore from .orig before rewriting (idempotent
# without overwriting the upstream snapshot).
#
# Output: human-readable per-case summary plus a machine-parseable
# PASS (benchmark)/PASS (auxiliary)/FAIL/RESTORED prefix.

set -euo pipefail
IFS=$'\n\t'

# -----------------------------------------------------------------------------
# Paths & constants
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd -P)"
BASINS_ROOT="$REPO_ROOT/SHUD/Basins"

# Allow-listed case sets per master plan §1 and CLAUDE.md domain rules.
# Anything not in these lists under Basins/ → SKIP with WARN (defensive
# against unknown / leftover dirs).
BENCHMARK_CASES=(keliya xinanjiang_upstream qinyijiang kashigeer qhh)
AUXILIARY_CASES=(tailanhe)
# Server-only cases must never be touched from a local-Mac invocation
# (forcing data sizes 12 G+; CLAUDE.md "cross-platform endpoints" rule).
SERVER_ONLY_CASES=(heihe heihe_x4 heihe_x16)

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }
# log_warn / log_error aliases matching the brief's naming.
log_warn()  { warn "$@"; }
log_error() { err "$@"; }

# -----------------------------------------------------------------------------
# Classification helpers
# -----------------------------------------------------------------------------

# in_array <needle> <array...>
in_array() {
    local needle="$1"; shift
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

classify_case() {
    local case_name="$1"
    if in_array "$case_name" "${BENCHMARK_CASES[@]}"; then
        printf 'benchmark'
    elif in_array "$case_name" "${AUXILIARY_CASES[@]}"; then
        printf 'auxiliary'
    else
        printf 'unknown'
    fi
}

# -----------------------------------------------------------------------------
# SIGINT / EXIT trap with in-flight rollback (Fix A3)
# -----------------------------------------------------------------------------
#
# Atomicity contract:
#   - INT/TERM handler performs rollback IMMEDIATELY and exits 130. We do
#     not defer to EXIT, because bash queues signals delivered during
#     external commands (e.g., sleep, awk, mv): if we deferred, the script
#     would resume the next statement, finish mutating, and clear the
#     in-flight markers before EXIT ran — losing the rollback window.
#   - In apply_case: before any mutation set IN_PROGRESS_CASE=<name> and
#     IN_PROGRESS_TSD_FORC / IN_PROGRESS_CFG_PARA. On clean end clear them.
#   - EXIT trap: only a safety net for the (rare) case where INT/TERM
#     handler couldn't run to completion (e.g., second signal during
#     rollback). For natural exit it is a no-op.

INTERRUPTED=0
IN_PROGRESS_CASE=""
IN_PROGRESS_TSD_FORC=""
IN_PROGRESS_CFG_PARA=""

_rollback_in_flight() {
    if [[ -z "$IN_PROGRESS_CASE" ]]; then
        return 0
    fi
    if [[ -n "$IN_PROGRESS_TSD_FORC" && -f "${IN_PROGRESS_TSD_FORC}.orig" ]]; then
        cp -p "${IN_PROGRESS_TSD_FORC}.orig" "$IN_PROGRESS_TSD_FORC" \
            || printf 'WARN: rollback of %s failed\n' "$IN_PROGRESS_TSD_FORC" >&2
    fi
    if [[ -n "$IN_PROGRESS_CFG_PARA" && -f "${IN_PROGRESS_CFG_PARA}.orig" ]]; then
        cp -p "${IN_PROGRESS_CFG_PARA}.orig" "$IN_PROGRESS_CFG_PARA" \
            || printf 'WARN: rollback of %s failed\n' "$IN_PROGRESS_CFG_PARA" >&2
    fi
}

_on_interrupt() {
    INTERRUPTED=1
    if [[ -z "$IN_PROGRESS_CASE" ]]; then
        printf 'interrupted; no in-flight case to roll back\n' >&2
        # Clear EXIT trap so it doesn't double-fire.
        trap - EXIT
        exit 130
    fi
    printf 'interrupted, rolling back %s\n' "$IN_PROGRESS_CASE" >&2
    _rollback_in_flight
    printf 'interrupted, rolled back %s\n' "$IN_PROGRESS_CASE" >&2
    # Clear EXIT trap so it doesn't double-fire on our explicit exit.
    trap - EXIT
    exit 130
}

_on_exit_rollback() {
    # Safety net: if INT/TERM handler couldn't complete rollback (e.g.,
    # second signal during rollback), still try once on script exit.
    if [[ $INTERRUPTED -eq 1 && -n "$IN_PROGRESS_CASE" ]]; then
        printf 'EXIT trap: rolling back %s (interrupt handler incomplete)\n' \
            "$IN_PROGRESS_CASE" >&2
        _rollback_in_flight
    fi
}

# -----------------------------------------------------------------------------
# Case-layout discovery
# -----------------------------------------------------------------------------
#
# Cases use an inconsistent layout:
#
#     SHUD/Basins/<case>/input/<project>/<project>.tsd.forc
#     SHUD/Basins/<case>/input/<project>/<project>.cfg.para
#     SHUD/Basins/<case>/forcing/     (or focing/ for tailanhe upstream typo)
#
# where <project> may differ from <case> (e.g., qinyijiang→nanlin,
# tailanhe→tlh, xinanjiang_upstream→xinanjiang). We discover <project>
# by listing input/ and locating the unique .tsd.forc.

# Resolve case paths.
# Sets globals: CASE_DIR, INPUT_DIR, PROJECT_NAME, TSD_FORC, CFG_PARA,
#               FORCING_DIR, FORCING_DIR_KIND
# FORCING_DIR_KIND ∈ {canonical, non-canonical-focing, missing}
# Returns 0 on success; prints reason and returns non-zero on failure.
resolve_case_paths() {
    local case_name="$1"
    CASE_DIR="$BASINS_ROOT/$case_name"
    if [[ ! -d "$CASE_DIR" ]]; then
        err "$case_name: case directory not found ($CASE_DIR)"
        return 1
    fi
    if [[ ! -d "$CASE_DIR/input" ]]; then
        err "$case_name: input/ directory missing"
        return 1
    fi

    # Locate the unique tsd.forc under input/*/
    local matches=()
    local f
    while IFS= read -r -d '' f; do
        matches+=("$f")
    done < <(find "$CASE_DIR/input" -mindepth 2 -maxdepth 2 -name '*.tsd.forc' -print0 2>/dev/null)

    if [[ "${#matches[@]}" -eq 0 ]]; then
        err "$case_name: no *.tsd.forc found under input/"
        return 1
    fi
    if [[ "${#matches[@]}" -gt 1 ]]; then
        err "$case_name: multiple *.tsd.forc files found (${matches[*]})"
        return 1
    fi

    TSD_FORC="${matches[0]}"
    INPUT_DIR="$(dirname "$TSD_FORC")"
    PROJECT_NAME="$(basename "$INPUT_DIR")"
    CFG_PARA="$INPUT_DIR/$PROJECT_NAME.cfg.para"

    # Fix A4: resolve forcing-dir with focing/ typo handling.
    if [[ -d "$CASE_DIR/forcing" ]]; then
        FORCING_DIR="$CASE_DIR/forcing"
        FORCING_DIR_KIND="canonical"
    elif [[ -d "$CASE_DIR/focing" ]]; then
        FORCING_DIR="$CASE_DIR/focing"
        FORCING_DIR_KIND="non-canonical-focing"
        log_warn "$case_name: using non-canonical forcing dir 'focing/' (upstream typo); model runtime should consume from this path"
    else
        FORCING_DIR=""
        FORCING_DIR_KIND="missing"
    fi
    return 0
}

# -----------------------------------------------------------------------------
# Edit primitives
# -----------------------------------------------------------------------------

# has_trailing_newline <file>
# Returns 0 iff the final byte of <file> is a newline.
has_trailing_newline() {
    local f="$1"
    [[ -s "$f" ]] || return 0  # empty file: trivially "ends with EOF"
    local last
    last="$(tail -c1 "$f" | od -An -c | tr -d ' \n')"
    [[ "$last" == '\n' ]]
}

# strip_trailing_newline <file>
# Truncates the final byte of <file> if it is a newline.
strip_trailing_newline() {
    local f="$1"
    [[ -s "$f" ]] || return 0
    if has_trailing_newline "$f"; then
        local sz
        sz="$(wc -c <"$f" | tr -d ' ')"
        if [[ "$sz" -gt 0 ]]; then
            local tmp
            tmp="$(mktemp "${f}.XXXXXX")" || return 1
            head -c "$((sz - 1))" "$f" >"$tmp" || { rm -f "$tmp"; return 1; }
            mv "$tmp" "$f" || { rm -f "$tmp"; return 1; }
        fi
    fi
}

# rewrite_tsd_forc <tsd_forc> <new_line_2> <dry_run_flag>
# dry_run_flag is "true" or "false".
# Fix A2: every FS op is return-checked; failures propagate non-zero.
rewrite_tsd_forc() {
    local tsd_forc="$1"
    local new_line2="$2"
    local dry_run="$3"

    if [[ ! -f "$tsd_forc" ]]; then
        err "tsd.forc missing: $tsd_forc"
        return 1
    fi

    local current_line2
    current_line2="$(awk 'NR==2 {print; exit}' "$tsd_forc")" || return 1

    log "  tsd.forc:  ${tsd_forc#$REPO_ROOT/}"
    if [[ "$dry_run" == "true" ]]; then
        log "    WOULD: rewrite line 2"
        log "    before line 2: $current_line2"
        log "    after  line 2: $new_line2"
        return 0
    fi

    log "    before line 2: $current_line2"
    log "    after  line 2: $new_line2"

    local had_trailing_nl=0
    has_trailing_newline "$tsd_forc" && had_trailing_nl=1

    local tmp
    tmp="$(mktemp "${tsd_forc}.XXXXXX")" || { err "mktemp failed for $tsd_forc"; return 1; }
    # awk's print always emits ORS=\n, so the temp file ends in a newline.
    if ! awk -v line2="$new_line2" 'NR==2 {print line2; next} {print}' "$tsd_forc" >"$tmp"; then
        rm -f "$tmp"
        err "awk rewrite failed for $tsd_forc"
        return 1
    fi
    if ! mv "$tmp" "$tsd_forc"; then
        rm -f "$tmp"
        err "mv failed for $tsd_forc"
        return 1
    fi
    if [[ $had_trailing_nl -eq 0 ]]; then
        if ! strip_trailing_newline "$tsd_forc"; then
            err "strip_trailing_newline failed for $tsd_forc"
            return 1
        fi
    fi
    return 0
}

# rewrite_cfg_para_num_openmp <cfg_para> <dry_run_flag>
# Fix B1: detect duplicates, dedupe (keep first), WARN with line numbers.
# Fix A2: hard-fail on every FS op error.
rewrite_cfg_para_num_openmp() {
    local cfg_para="$1"
    local dry_run="$2"

    if [[ ! -f "$cfg_para" ]]; then
        err "cfg.para missing: $cfg_para"
        return 1
    fi

    local existing
    existing="$(grep -nE '^NUM_OPENMP[[:space:]]' "$cfg_para" || true)"

    # Count matching lines for duplicate detection.
    local match_count=0
    if [[ -n "$existing" ]]; then
        match_count="$(printf '%s\n' "$existing" | wc -l | tr -d ' ')"
    fi

    log "  cfg.para:  ${cfg_para#$REPO_ROOT/}"
    if [[ "$match_count" -ge 2 ]]; then
        local line_nums
        line_nums="$(printf '%s\n' "$existing" | cut -d: -f1 | tr '\n' ',' | sed 's/,$//')"
        log_warn "    NUM_OPENMP appears on multiple lines ($line_nums); will dedupe (keep first, drop the rest)"
    fi

    if [[ -n "$existing" ]]; then
        log "    NUM_OPENMP existing: $existing"
        log "    NUM_OPENMP target  : NUM_OPENMP<TAB>1"
    else
        log_warn "    NUM_OPENMP line missing; will append at EOF"
        log "    NUM_OPENMP target  : NUM_OPENMP<TAB>1 (appended)"
    fi

    if [[ "$dry_run" == "true" ]]; then
        log "    WOULD: rewrite NUM_OPENMP -> 1"
        return 0
    fi

    local had_trailing_nl=0
    has_trailing_newline "$cfg_para" && had_trailing_nl=1

    local tmp
    tmp="$(mktemp "${cfg_para}.XXXXXX")" || { err "mktemp failed for $cfg_para"; return 1; }

    if [[ -n "$existing" ]]; then
        # Replace the FIRST NUM_OPENMP line with `NUM_OPENMP\t1`, DROP any
        # subsequent NUM_OPENMP lines (dedupe), preserve everything else.
        if ! awk -v t="$(printf '\t')" '
            $1=="NUM_OPENMP" && !seen { printf "NUM_OPENMP%s1\n", t; seen=1; next }
            $1=="NUM_OPENMP" && seen  { next }
            { print }
        ' "$cfg_para" >"$tmp"; then
            rm -f "$tmp"
            err "awk rewrite failed for $cfg_para"
            return 1
        fi
        if ! mv "$tmp" "$cfg_para"; then
            rm -f "$tmp"
            err "mv failed for $cfg_para"
            return 1
        fi
        if [[ $had_trailing_nl -eq 0 ]]; then
            if ! strip_trailing_newline "$cfg_para"; then
                err "strip_trailing_newline failed for $cfg_para"
                return 1
            fi
        fi
    else
        # Append a NUM_OPENMP line. Preserve original byte sequence verbatim,
        # ensure a separator newline, then append the canonical line.
        if ! cat "$cfg_para" >"$tmp"; then
            rm -f "$tmp"
            err "cat failed for $cfg_para"
            return 1
        fi
        if [[ $had_trailing_nl -eq 0 ]]; then
            if ! printf '\n' >>"$tmp"; then
                rm -f "$tmp"
                err "printf newline failed for $tmp"
                return 1
            fi
        fi
        if ! printf 'NUM_OPENMP\t1\n' >>"$tmp"; then
            rm -f "$tmp"
            err "printf NUM_OPENMP failed for $tmp"
            return 1
        fi
        if ! mv "$tmp" "$cfg_para"; then
            rm -f "$tmp"
            err "mv failed for $cfg_para"
            return 1
        fi
    fi
    return 0
}

# snapshot_orig <file>
# Creates <file>.orig if missing. Never overwrites an existing .orig.
# Fix A2: hard-fail on cp error.
snapshot_orig() {
    local target="$1"
    local orig="${target}.orig"
    if [[ -f "$orig" ]]; then
        log "    .orig exists: ${orig#$REPO_ROOT/} (preserved)"
    else
        if ! cp -p "$target" "$orig"; then
            err "snapshot_orig: cp failed for $target → $orig"
            return 1
        fi
        log "    .orig created: ${orig#$REPO_ROOT/}"
    fi
    return 0
}

# restore_from_orig <file> <dry_run_flag>
# Restores <file> from <file>.orig and removes the backup.
# Fix A1: respect dry_run (no mutation, no .orig deletion).
# Fix A2: hard-fail on cp/rm error.
restore_from_orig() {
    local target="$1"
    local dry_run="$2"
    local orig="${target}.orig"
    if [[ ! -f "$orig" ]]; then
        warn "no .orig backup for $target"
        return 1
    fi
    if [[ "$dry_run" == "true" ]]; then
        log "    WOULD: restore ${target#$REPO_ROOT/} from .orig and delete .orig"
        return 0
    fi
    if ! cp -p "$orig" "$target"; then
        err "restore_from_orig: cp failed for $orig → $target"
        return 1
    fi
    if ! rm "$orig"; then
        err "restore_from_orig: rm failed for $orig"
        return 1
    fi
    log "    restored: ${target#$REPO_ROOT/} (and removed .orig)"
    return 0
}

# -----------------------------------------------------------------------------
# High-level operations per case
# -----------------------------------------------------------------------------

# apply_case <case_name> <dry_run_flag>
# dry_run_flag is "true" or "false".
# Returns 0 on success, non-zero on failure.
apply_case() {
    local case_name="$1"
    local dry_run="$2"

    if ! resolve_case_paths "$case_name"; then
        return 1
    fi

    local kind
    kind="$(classify_case "$case_name")"

    log ""
    log "==> $case_name ($kind)"
    log "    case dir : ${CASE_DIR#$REPO_ROOT/}"
    log "    project  : $PROJECT_NAME"

    # Fix A4: FORCING_DIR_KIND drives behavior; "missing" → FAIL.
    case "$FORCING_DIR_KIND" in
        canonical)        ;;
        non-canonical-focing)
            log "    forcing  : non-canonical 'focing/' (upstream typo; used as-is)"
            ;;
        missing)
            log_error "$case_name: neither forcing/ nor focing/ found under $CASE_DIR"
            return 1
            ;;
    esac

    local new_line2
    # readlink -f is GNU-only on macOS, so pwd -P is portable enough.
    new_line2="$(cd "$FORCING_DIR" && pwd -P)" || return 1

    # Fix A3: track in-flight case so EXIT trap can roll back on SIGINT.
    if [[ "$dry_run" != "true" ]]; then
        IN_PROGRESS_CASE="$case_name"
        IN_PROGRESS_TSD_FORC="$TSD_FORC"
        IN_PROGRESS_CFG_PARA="$CFG_PARA"
    fi

    # Snapshot before edits (only on a true mutating run).
    if [[ "$dry_run" != "true" ]]; then
        if [[ -f "$TSD_FORC" ]]; then
            if ! snapshot_orig "$TSD_FORC"; then
                IN_PROGRESS_CASE=""
                IN_PROGRESS_TSD_FORC=""
                IN_PROGRESS_CFG_PARA=""
                return 1
            fi
        else
            err "$case_name: tsd.forc missing — cannot snapshot"
            IN_PROGRESS_CASE=""
            IN_PROGRESS_TSD_FORC=""
            IN_PROGRESS_CFG_PARA=""
            return 1
        fi
        if [[ -f "$CFG_PARA" ]]; then
            if ! snapshot_orig "$CFG_PARA"; then
                IN_PROGRESS_CASE=""
                IN_PROGRESS_TSD_FORC=""
                IN_PROGRESS_CFG_PARA=""
                return 1
            fi
        else
            err "$case_name: cfg.para missing — cannot snapshot"
            IN_PROGRESS_CASE=""
            IN_PROGRESS_TSD_FORC=""
            IN_PROGRESS_CFG_PARA=""
            return 1
        fi
        # Idempotency: restore from .orig before re-applying so the result
        # is deterministic regardless of the working file's current state.
        if ! cp -p "${TSD_FORC}.orig" "$TSD_FORC"; then
            err "$case_name: cp failed restoring tsd.forc from .orig (pre-apply)"
            IN_PROGRESS_CASE=""
            IN_PROGRESS_TSD_FORC=""
            IN_PROGRESS_CFG_PARA=""
            return 1
        fi
        if ! cp -p "${CFG_PARA}.orig" "$CFG_PARA"; then
            err "$case_name: cp failed restoring cfg.para from .orig (pre-apply)"
            IN_PROGRESS_CASE=""
            IN_PROGRESS_TSD_FORC=""
            IN_PROGRESS_CFG_PARA=""
            return 1
        fi
    fi

    if ! rewrite_tsd_forc "$TSD_FORC" "$new_line2" "$dry_run"; then
        IN_PROGRESS_CASE=""
        IN_PROGRESS_TSD_FORC=""
        IN_PROGRESS_CFG_PARA=""
        return 1
    fi
    if ! rewrite_cfg_para_num_openmp "$CFG_PARA" "$dry_run"; then
        IN_PROGRESS_CASE=""
        IN_PROGRESS_TSD_FORC=""
        IN_PROGRESS_CFG_PARA=""
        return 1
    fi

    # Clean end → clear in-flight markers so EXIT trap won't roll back.
    IN_PROGRESS_CASE=""
    IN_PROGRESS_TSD_FORC=""
    IN_PROGRESS_CFG_PARA=""
    return 0
}

# restore_case <case_name> <dry_run_flag>
restore_case() {
    local case_name="$1"
    local dry_run="$2"

    if ! resolve_case_paths "$case_name"; then
        return 1
    fi

    log ""
    if [[ "$dry_run" == "true" ]]; then
        log "==> $case_name (restore, dry-run)"
    else
        log "==> $case_name (restore)"
    fi
    log "    case dir : ${CASE_DIR#$REPO_ROOT/}"

    # Fix A3: track in-flight for restore too (so SIGINT rollback works).
    if [[ "$dry_run" != "true" ]]; then
        IN_PROGRESS_CASE="$case_name"
        IN_PROGRESS_TSD_FORC="$TSD_FORC"
        IN_PROGRESS_CFG_PARA="$CFG_PARA"
    fi

    local rc=0
    if [[ -f "${TSD_FORC}.orig" ]]; then
        restore_from_orig "$TSD_FORC" "$dry_run" || rc=1
    else
        warn "    no .orig for $(basename "$TSD_FORC"); skipping"
    fi
    if [[ -f "${CFG_PARA}.orig" ]]; then
        restore_from_orig "$CFG_PARA" "$dry_run" || rc=1
    else
        warn "    no .orig for $(basename "$CFG_PARA"); skipping"
    fi

    IN_PROGRESS_CASE=""
    IN_PROGRESS_TSD_FORC=""
    IN_PROGRESS_CFG_PARA=""
    return $rc
}

# -----------------------------------------------------------------------------
# Enumeration for --all (Fix B2: allow-list only)
# -----------------------------------------------------------------------------

list_local_cases() {
    # Print BENCHMARK + AUXILIARY cases that exist locally; SKIP+WARN
    # for unknown dirs; never include server-only cases.
    local present=()
    local d name
    for d in "$BASINS_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        case "$name" in
            .*) continue ;;
        esac
        present+=("$name")
    done

    # Emit in BENCHMARK-then-AUXILIARY order; warn on unknown.
    local emitted=()
    local c
    for c in "${BENCHMARK_CASES[@]}"; do
        if in_array "$c" "${present[@]}"; then
            printf '%s\n' "$c"
            emitted+=("$c")
        fi
    done
    for c in "${AUXILIARY_CASES[@]}"; do
        if in_array "$c" "${present[@]}"; then
            printf '%s\n' "$c"
            emitted+=("$c")
        fi
    done

    # Warn for any present dir not in benchmark/auxiliary/server-only sets.
    for c in "${present[@]}"; do
        if in_array "$c" "${BENCHMARK_CASES[@]}"; then continue; fi
        if in_array "$c" "${AUXILIARY_CASES[@]}"; then continue; fi
        if in_array "$c" "${SERVER_ONLY_CASES[@]}"; then
            warn "skipping server-only case under Basins/: $c (do not deploy locally)"
            continue
        fi
        warn "skipping unknown dir under Basins/: $c (not in BENCHMARK_CASES, AUXILIARY_CASES, or SERVER_ONLY_CASES)"
    done
}

# -----------------------------------------------------------------------------
# Top-level dispatch
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage:
  fix_case_paths.sh <case>                       apply fixup
  fix_case_paths.sh --dry-run <case>             preview without writing
  fix_case_paths.sh --restore <case>             revert from .orig backups
  fix_case_paths.sh --dry-run --restore <case>   preview restore (no mutation)
  fix_case_paths.sh --all                        bulk apply
  fix_case_paths.sh --all --dry-run              bulk preview apply
  fix_case_paths.sh --all --restore              bulk restore
  fix_case_paths.sh --all --dry-run --restore    bulk preview restore

Notes:
  --dry-run and --restore are orthogonal: any combination is valid.
  Flag order is unrestricted.
EOF
}

main() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 2
    fi

    # Fix A1: dry_run and mode are ORTHOGONAL.
    local MODE="apply"        # apply | restore
    local DRY_RUN="false"     # true | false
    local case_arg=""
    local all=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)  DRY_RUN="true"; shift ;;
            --restore)  MODE="restore"; shift ;;
            --all)      all=1; shift ;;
            --help|-h)  usage; exit 0 ;;
            --)         shift; break ;;
            -*)         err "unknown option: $1"; usage >&2; exit 2 ;;
            *)          case_arg="$1"; shift ;;
        esac
    done

    if [[ ! -d "$BASINS_ROOT" ]]; then
        err "Basins root not found: $BASINS_ROOT"
        exit 1
    fi

    # Fix A3: register traps now that option parsing succeeded.
    trap '_on_interrupt' INT TERM
    trap '_on_exit_rollback' EXIT

    if [[ $all -eq 1 ]]; then
        if [[ -n "$case_arg" ]]; then
            err "--all and a case argument are mutually exclusive"
            exit 2
        fi
        local failures=0
        local cases
        cases="$(list_local_cases)"
        if [[ -z "$cases" ]]; then
            err "no local cases found under $BASINS_ROOT"
            exit 1
        fi

        local mode_label="$MODE"
        [[ "$DRY_RUN" == "true" ]] && mode_label="${mode_label} (dry-run)"
        log "Bulk $mode_label over local cases under SHUD/Basins/:"
        local name
        while IFS= read -r name; do
            local kind
            kind="$(classify_case "$name")"
            case "$MODE" in
                apply)
                    if apply_case "$name" "$DRY_RUN"; then
                        if [[ "$DRY_RUN" == "true" ]]; then
                            log "PASS ($kind, dry-run): $name"
                        else
                            log "PASS ($kind): $name"
                        fi
                    else
                        log "FAIL ($kind): $name"
                        failures=$((failures + 1))
                    fi
                    ;;
                restore)
                    if restore_case "$name" "$DRY_RUN"; then
                        if [[ "$DRY_RUN" == "true" ]]; then
                            log "RESTORED ($kind, dry-run): $name"
                        else
                            log "RESTORED ($kind): $name"
                        fi
                    else
                        log "FAIL ($kind): $name"
                        failures=$((failures + 1))
                    fi
                    ;;
            esac
            # Bulk: if interrupted mid-loop, EXIT trap will rollback the
            # in-flight case; do not start the next one.
            if [[ $INTERRUPTED -eq 1 ]]; then
                break
            fi
        done <<<"$cases"

        log ""
        log "Bulk $mode_label complete. failures=$failures"
        if [[ $failures -gt 0 ]]; then
            exit 1
        fi
        exit 0
    fi

    if [[ -z "$case_arg" ]]; then
        err "missing case name"
        usage >&2
        exit 2
    fi

    # Fix B3: server-only case refusal on direct invocation.
    if in_array "$case_arg" "${SERVER_ONLY_CASES[@]}"; then
        err "case '$case_arg' is server-only; bulk --all skips it; direct invocation is refused (data is 12 G+, do not download locally)"
        exit 2
    fi

    local kind
    kind="$(classify_case "$case_arg")"
    if [[ "$kind" == "unknown" ]]; then
        warn "case '$case_arg' is not in BENCHMARK_CASES or AUXILIARY_CASES; processing anyway since invoked explicitly"
    fi

    case "$MODE" in
        apply)
            if apply_case "$case_arg" "$DRY_RUN"; then
                log ""
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "PASS ($kind, dry-run): $case_arg"
                else
                    log "PASS ($kind): $case_arg"
                fi
                exit 0
            else
                log ""
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "FAIL ($kind, dry-run): $case_arg"
                else
                    log "FAIL ($kind): $case_arg"
                fi
                exit 1
            fi
            ;;
        restore)
            if restore_case "$case_arg" "$DRY_RUN"; then
                log ""
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "RESTORED ($kind, dry-run): $case_arg"
                else
                    log "RESTORED ($kind): $case_arg"
                fi
                exit 0
            else
                log ""
                if [[ "$DRY_RUN" == "true" ]]; then
                    log "FAIL ($kind, restore, dry-run): $case_arg"
                else
                    log "FAIL ($kind, restore): $case_arg"
                fi
                exit 1
            fi
            ;;
    esac
}

main "$@"
