#!/usr/bin/env bash
# tools/fix_case_paths/fix_case_paths.sh
#
# Rewrites server-deployment paths in NWM cases under SHUD/Basins/<case>/ to
# the developer's local absolute paths, and forces NUM_OPENMP=1 for the
# serial baseline (S0.1b). See openspec/changes/s0-baseline-lock/specs/
# case-deployment-fixup/spec.md for the contract.
#
# Modes:
#   fix_case_paths.sh <case>             apply fixup to a single case
#   fix_case_paths.sh --dry-run <case>   print planned edits, change nothing
#   fix_case_paths.sh --restore <case>   restore from .orig backups
#   fix_case_paths.sh --all              bulk apply across SHUD/Basins/*
#   fix_case_paths.sh --all --restore    bulk restore across SHUD/Basins/*
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

# Benchmark cases per master plan §1 / brief. Anything else under Basins/
# (except heihe and hidden dirs) is treated as auxiliary.
BENCHMARK_CASES=(keliya xinanjiang_upstream qinyijiang kashigeer qhh)

# Cases that are server-only (NWM forcing too large to deploy locally).
SERVER_ONLY_CASES=(heihe)

# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

log()  { printf '%s\n' "$*"; }
warn() { printf 'WARN: %s\n' "$*" >&2; }
err()  { printf 'ERROR: %s\n' "$*" >&2; }

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
    else
        printf 'auxiliary'
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
#     SHUD/Basins/<case>/forcing/
#
# where <project> may differ from <case> (e.g., qinyijiang→nanlin,
# tailanhe→tlh, xinanjiang_upstream→xinanjiang). We discover <project>
# by listing input/ and locating the unique .tsd.forc.

# Resolve case paths.
# Sets globals: CASE_DIR, INPUT_DIR, PROJECT_NAME, TSD_FORC, CFG_PARA, FORCING_DIR
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
    FORCING_DIR="$CASE_DIR/forcing"
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
    # tail -c1 strips trailing newline in command-substitution; test the
    # raw byte via od instead.
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
        # Portable truncate: compute size-1 and re-write.
        local sz
        sz="$(wc -c <"$f" | tr -d ' ')"
        if [[ "$sz" -gt 0 ]]; then
            local tmp
            tmp="$(mktemp "${f}.XXXXXX")"
            head -c "$((sz - 1))" "$f" >"$tmp"
            mv "$tmp" "$f"
        fi
    fi
}

# rewrite_tsd_forc <tsd_forc> <new_line_2> [--dry-run]
# Rewrites line 2 in-place; preserves all other bytes including the
# trailing-newline state of the original.
rewrite_tsd_forc() {
    local tsd_forc="$1"
    local new_line2="$2"
    local dry_run="${3:-}"

    if [[ ! -f "$tsd_forc" ]]; then
        err "tsd.forc missing: $tsd_forc"
        return 1
    fi

    local current_line2
    current_line2="$(awk 'NR==2 {print; exit}' "$tsd_forc")"

    log "  tsd.forc:  ${tsd_forc#$REPO_ROOT/}"
    log "    before line 2: $current_line2"
    log "    after  line 2: $new_line2"

    if [[ "$dry_run" == "--dry-run" ]]; then
        return 0
    fi

    local had_trailing_nl=0
    has_trailing_newline "$tsd_forc" && had_trailing_nl=1

    local tmp
    tmp="$(mktemp "${tsd_forc}.XXXXXX")"
    # awk's print always emits ORS=\n, so the temp file ends in a newline.
    awk -v line2="$new_line2" 'NR==2 {print line2; next} {print}' "$tsd_forc" >"$tmp"
    mv "$tmp" "$tsd_forc"
    if [[ $had_trailing_nl -eq 0 ]]; then
        strip_trailing_newline "$tsd_forc"
    fi
    return 0
}

# rewrite_cfg_para_num_openmp <cfg_para> [--dry-run]
# Ensures exactly one `NUM_OPENMP\t1` line. Appends if missing.
# All other lines byte-preserved.
rewrite_cfg_para_num_openmp() {
    local cfg_para="$1"
    local dry_run="${2:-}"

    if [[ ! -f "$cfg_para" ]]; then
        err "cfg.para missing: $cfg_para"
        return 1
    fi

    local existing
    existing="$(grep -nE '^NUM_OPENMP[[:space:]]' "$cfg_para" || true)"

    log "  cfg.para:  ${cfg_para#$REPO_ROOT/}"
    if [[ -n "$existing" ]]; then
        log "    NUM_OPENMP existing: $existing"
        log "    NUM_OPENMP target  : NUM_OPENMP<TAB>1"
    else
        warn "    NUM_OPENMP line missing; will append at EOF"
        log "    NUM_OPENMP target  : NUM_OPENMP<TAB>1 (appended)"
    fi

    if [[ "$dry_run" == "--dry-run" ]]; then
        return 0
    fi

    local had_trailing_nl=0
    has_trailing_newline "$cfg_para" && had_trailing_nl=1

    local tmp
    tmp="$(mktemp "${cfg_para}.XXXXXX")"
    if [[ -n "$existing" ]]; then
        # Replace every line whose first whitespace-delimited field is
        # exactly NUM_OPENMP with `NUM_OPENMP\t1`. Tab is set via -v t=...
        # because POSIX awk strings don't always parse \t literals.
        awk -v t="$(printf '\t')" '
            $1=="NUM_OPENMP" { printf "NUM_OPENMP%s1\n", t; next }
            { print }
        ' "$cfg_para" >"$tmp"
        mv "$tmp" "$cfg_para"
        # Preserve original trailing-newline state.
        if [[ $had_trailing_nl -eq 0 ]]; then
            strip_trailing_newline "$cfg_para"
        fi
    else
        # Append a NUM_OPENMP line. Preserve original byte sequence
        # verbatim, then ensure a separator newline, then append the
        # canonical line. The result always ends in \n because we just
        # added a line; that's intentional — the appended line itself
        # needs proper line termination so downstream parsers see it.
        cat "$cfg_para" >"$tmp"
        if [[ $had_trailing_nl -eq 0 ]]; then
            printf '\n' >>"$tmp"
        fi
        printf 'NUM_OPENMP\t1\n' >>"$tmp"
        mv "$tmp" "$cfg_para"
    fi
    return 0
}

# snapshot_orig <file>
# Creates <file>.orig if missing. Never overwrites an existing .orig.
snapshot_orig() {
    local target="$1"
    local orig="${target}.orig"
    if [[ -f "$orig" ]]; then
        log "    .orig exists: ${orig#$REPO_ROOT/} (preserved)"
    else
        cp -p "$target" "$orig"
        log "    .orig created: ${orig#$REPO_ROOT/}"
    fi
}

# restore_from_orig <file>
# Restores <file> from <file>.orig and removes the backup.
restore_from_orig() {
    local target="$1"
    local orig="${target}.orig"
    if [[ ! -f "$orig" ]]; then
        warn "no .orig backup for $target"
        return 1
    fi
    cp -p "$orig" "$target"
    rm "$orig"
    log "    restored: ${target#$REPO_ROOT/} (and removed .orig)"
    return 0
}

# -----------------------------------------------------------------------------
# High-level operations per case
# -----------------------------------------------------------------------------

# apply_case <case_name> [--dry-run]
# Returns 0 on success, non-zero on failure.
apply_case() {
    local case_name="$1"
    local dry_run="${2:-}"

    if ! resolve_case_paths "$case_name"; then
        return 1
    fi

    local kind
    kind="$(classify_case "$case_name")"

    log ""
    log "==> $case_name ($kind)"
    log "    case dir : ${CASE_DIR#$REPO_ROOT/}"
    log "    project  : $PROJECT_NAME"

    if [[ ! -d "$FORCING_DIR" ]]; then
        # Spec doesn't strictly require forcing/ to exist for the rewrite
        # (it's just a string in tsd.forc); still warn so the operator
        # knows downstream model runs will fail until data is staged.
        warn "    forcing dir missing: ${FORCING_DIR#$REPO_ROOT/} (rewrite will still record this path)"
    fi

    local new_line2
    # Use the conventional path (don't follow symlinks unnecessarily;
    # readlink -f is GNU-only on macOS, so pwd -P is portable enough).
    if [[ -d "$FORCING_DIR" ]]; then
        new_line2="$(cd "$FORCING_DIR" && pwd -P)"
    else
        new_line2="$FORCING_DIR"
    fi

    # Snapshot before edits (only on a true mutating run).
    if [[ "$dry_run" != "--dry-run" ]]; then
        if [[ -f "$TSD_FORC" ]]; then
            snapshot_orig "$TSD_FORC"
        else
            err "$case_name: tsd.forc missing — cannot snapshot"
            return 1
        fi
        if [[ -f "$CFG_PARA" ]]; then
            snapshot_orig "$CFG_PARA"
        else
            err "$case_name: cfg.para missing — cannot snapshot"
            return 1
        fi
        # Idempotency: restore from .orig before re-applying so the result
        # is deterministic regardless of the working file's current state.
        cp -p "${TSD_FORC}.orig" "$TSD_FORC"
        cp -p "${CFG_PARA}.orig" "$CFG_PARA"
    fi

    if ! rewrite_tsd_forc "$TSD_FORC" "$new_line2" "$dry_run"; then
        return 1
    fi
    if ! rewrite_cfg_para_num_openmp "$CFG_PARA" "$dry_run"; then
        return 1
    fi

    return 0
}

# restore_case <case_name>
restore_case() {
    local case_name="$1"

    if ! resolve_case_paths "$case_name"; then
        return 1
    fi

    log ""
    log "==> $case_name (restore)"
    log "    case dir : ${CASE_DIR#$REPO_ROOT/}"

    local rc=0
    if [[ -f "${TSD_FORC}.orig" ]]; then
        restore_from_orig "$TSD_FORC" || rc=1
    else
        warn "    no .orig for $(basename "$TSD_FORC"); skipping"
    fi
    if [[ -f "${CFG_PARA}.orig" ]]; then
        restore_from_orig "$CFG_PARA" || rc=1
    else
        warn "    no .orig for $(basename "$CFG_PARA"); skipping"
    fi

    return $rc
}

# -----------------------------------------------------------------------------
# Enumeration for --all
# -----------------------------------------------------------------------------

list_local_cases() {
    # Print one case name per line, skipping hidden dirs and the
    # server-only set. Sort for deterministic batch ordering.
    local d name
    for d in "$BASINS_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        name="$(basename "$d")"
        case "$name" in
            .*) continue ;;
        esac
        if in_array "$name" "${SERVER_ONLY_CASES[@]}"; then
            continue
        fi
        printf '%s\n' "$name"
    done | sort
}

# -----------------------------------------------------------------------------
# Top-level dispatch
# -----------------------------------------------------------------------------

usage() {
    cat <<EOF
Usage:
  fix_case_paths.sh <case>                apply fixup
  fix_case_paths.sh --dry-run <case>      preview without writing
  fix_case_paths.sh --restore <case>      revert from .orig backups
  fix_case_paths.sh --all                 bulk apply to all local cases
  fix_case_paths.sh --all --restore       bulk restore all local cases
EOF
}

main() {
    if [[ $# -eq 0 ]]; then
        usage >&2
        exit 2
    fi

    local mode="apply"
    local case_arg=""
    local all=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dry-run)  mode="dry-run"; shift ;;
            --restore)  mode="restore"; shift ;;
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

        log "Bulk $mode over local cases under SHUD/Basins/:"
        local name
        while IFS= read -r name; do
            local kind
            kind="$(classify_case "$name")"
            case "$mode" in
                apply|dry-run)
                    local dry=""
                    [[ "$mode" == "dry-run" ]] && dry="--dry-run"
                    if apply_case "$name" "$dry"; then
                        log "PASS ($kind): $name"
                    else
                        log "FAIL ($kind): $name"
                        failures=$((failures + 1))
                    fi
                    ;;
                restore)
                    if restore_case "$name"; then
                        log "RESTORED ($kind): $name"
                    else
                        log "FAIL ($kind): $name"
                        failures=$((failures + 1))
                    fi
                    ;;
            esac
        done <<<"$cases"

        log ""
        log "Bulk $mode complete. failures=$failures"
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

    local kind
    kind="$(classify_case "$case_arg")"

    case "$mode" in
        apply)
            if apply_case "$case_arg"; then
                log ""
                log "PASS ($kind): $case_arg"
                exit 0
            else
                log ""
                log "FAIL ($kind): $case_arg"
                exit 1
            fi
            ;;
        dry-run)
            if apply_case "$case_arg" "--dry-run"; then
                log ""
                log "PASS ($kind, dry-run): $case_arg"
                exit 0
            else
                log ""
                log "FAIL ($kind, dry-run): $case_arg"
                exit 1
            fi
            ;;
        restore)
            if restore_case "$case_arg"; then
                log ""
                log "RESTORED ($kind): $case_arg"
                exit 0
            else
                log ""
                log "FAIL ($kind, restore): $case_arg"
                exit 1
            fi
            ;;
    esac
}

main "$@"
