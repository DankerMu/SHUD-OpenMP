#!/usr/bin/env bash
# tools/test_check_invariant_sweep.sh — selftest for check_invariant_sweep.sh
#
# Purpose (PR #71 round-1 F1 fix follow-up): the old check_invariant_sweep.sh
# `local globs=( $paths_csv )` form silently pre-expanded `tools/**/*.sh` to
# whatever happened to be on disk at parse time, AND on bash <4 (macOS default,
# no globstar) `**` collapsed to `*`. The result: top-level `tools/*.sh` and
# any post-snapshot file additions were never scanned, and the entire
# `invariant-sweep` CI job was a silent no-op. Verifier confirmed via a
# `bad_var=<tmp-pid-token>` injection (see TMP_MARKER below) that returned
# `[OK] absent in 9 file(s)` exit 0 when it MUST have been [MISMATCH] exit 1.
#
# This selftest injects controlled bad-token sentinels into multiple TRACKED
# files and asserts the live sweep flips to [MISMATCH] + non-zero exit each
# time. If any injection fails to surface, the sweep is back to the
# pre-fix no-op behavior and the test exits non-zero. Restores the originals
# unconditionally via EXIT trap so a Ctrl-C / SIGTERM mid-run still leaves
# the worktree clean.
#
# Usage:
#   bash tools/test_check_invariant_sweep.sh
#
# Exit codes:
#   0  all injected scenarios were detected (last line: SELFTEST: N/N PASS)
#   1  at least one scenario was missed (last line: SELFTEST: K/N FAIL)
#   2  setup / teardown error (tool / spec missing, restore failed, etc.)

set -eu
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEEP="$REPO/tools/check_invariant_sweep.sh"
SPEC="$REPO/tools/invariants.yaml"

[ -x "$SWEEP" ] || [ -r "$SWEEP" ] || {
    echo "selftest setup error: missing $SWEEP" >&2
    exit 2
}
[ -r "$SPEC" ] || {
    echo "selftest setup error: missing $SPEC" >&2
    exit 2
}

# -----------------------------------------------------------------------------
# Restore registry: each injection appends `path|sha256-before` so the EXIT
# trap can verify the post-restore content matches the pre-injection bytes.
# -----------------------------------------------------------------------------
REGISTRY="$(mktemp "/tmp/inv_sweep_selftest_registry_XXXXXX")"
# Buffer of cleanup paths (injected files we copied to .bak siblings).
BACKUPS=()

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

# shellcheck disable=SC2317  # invoked via `trap ... EXIT`, not unreachable
restore_all() {
    # Restore in reverse insertion order so multi-injection on the same file
    # unwinds cleanly. The first .bak ever made for a given path is the
    # canonical pre-test state.
    local idx
    for (( idx=${#BACKUPS[@]}-1 ; idx>=0 ; idx-- )); do
        local entry="${BACKUPS[idx]}"
        local path="${entry%%|*}"
        local bak="${entry##*|}"
        if [ -f "$bak" ]; then
            cp -f "$bak" "$path"
            rm -f "$bak"
        fi
    done
    # Verify checksum-restored state matches initial registry.
    local fail=0
    while IFS='|' read -r path expected; do
        [ -z "$path" ] && continue
        actual="$(sha256_of "$path")"
        if [ "$actual" != "$expected" ]; then
            echo "selftest teardown WARN: $path sha256 drifted ($expected -> $actual)" >&2
            fail=1
        fi
    done < "$REGISTRY"
    rm -f "$REGISTRY"
    if [ "$fail" -ne 0 ]; then
        return 2
    fi
}
trap 'restore_all || exit 2' EXIT

inject() {
    # inject <path> <literal-marker>
    # Appends a sentinel line to <path> and registers a .bak for restore.
    local path="$1" marker="$2"
    if [ ! -f "$path" ]; then
        echo "selftest setup error: cannot inject into missing $path" >&2
        return 2
    fi
    local bak="${path}.selftest.bak"
    # Only snapshot+register on first touch of this path; multi-inject same
    # file is allowed but the canonical restore source is the FIRST .bak.
    if [ ! -f "$bak" ]; then
        cp "$path" "$bak"
        local pre_sha
        pre_sha="$(sha256_of "$path")"
        printf '%s|%s\n' "$path" "$pre_sha" >> "$REGISTRY"
        BACKUPS+=("${path}|${bak}")
    fi
    # Append using printf to avoid `echo` flag-eating + ensure trailing newline.
    printf '\n%s\n' "$marker" >> "$path"
}

restore_one() {
    # restore_one <path>  — undo the most recent injection for <path>.
    local path="$1"
    local bak="${path}.selftest.bak"
    if [ -f "$bak" ]; then
        cp -f "$bak" "$path"
        rm -f "$bak"
    fi
    # Drop the matching BACKUPS entry so EXIT trap does not double-restore.
    local new_backups=() entry
    for entry in "${BACKUPS[@]}"; do
        case "$entry" in
            "${path}|"*) continue;;
            *) new_backups+=("$entry");;
        esac
    done
    BACKUPS=("${new_backups[@]+"${new_backups[@]}"}")
}

# -----------------------------------------------------------------------------
# Inject sites — chosen to cover the F1 regression vectors:
#   * Top-level `tools/<name>.sh`            : would be MISSED if `**/*` -> `*`
#   * Nested `tools/<sub>/<name>.sh`         : the only paths the buggy
#                                              snapshot expansion ever scanned
#   * Whole-repo `**/*` invariant (PII)      : verifies the F3 PII hex-decode
#                                              path also catches injected IPs
#   * `.github/workflows/*.yml` literal      : verifies hardcoded_start_day
#                                              invariant's literal-path glob
#                                              flow (CSV split correctness)
# At least 3 invariants × ≥2 sites each per brief.
# -----------------------------------------------------------------------------

# Site candidates (each must exist tracked).
TOP_LEVEL_SH="$REPO/tools/archive_b0_output.sh"
NESTED_SH="$REPO/tools/cvode_stats_diff/cvode_stats_diff.sh"
WORKFLOW_YML="$REPO/.github/workflows/serial-baseline.yml"

for f in "$TOP_LEVEL_SH" "$NESTED_SH" "$WORKFLOW_YML"; do
    if [ ! -f "$f" ]; then
        echo "selftest setup error: missing fixture file $f" >&2
        exit 2
    fi
done

# Sentinel markers — constructed at runtime via concatenation so the literal
# tokens do NOT appear in this script's source. Two reasons:
#   1. This file is itself under tools/**/*.sh, so a literal sentinel here
#      would self-trigger its invariant on the first sweep before any
#      injection runs (the selftest would always fail at phase 0 baseline).
#   2. The PII / START sentinels would similarly self-trigger their respective
#      invariants once the F1 fix correctly expands `**/*` repo-wide.
# strip_comments() in check_invariant_sweep.sh only filters lines whose FIRST
# non-space char is `#` — it does NOT strip trailing # comments. So end-of-
# line annotations here also have to avoid the literal token text. We use
# placeholder words ("tmp-pid-token" / "pii-ip" / "start-day") in trailing
# comments so the lines remain self-documenting without re-introducing the
# very pattern we are testing for.
#
# shellcheck disable=SC2034  # MARKERS consumed via shell-only expansion below
TMP_MARKER='/tmp/'"$"'$'                              # token for tmp-pid invariant
PII_MARKER='210.77''.77.22'                           # token for pii-ip invariant
START_MARKER='1''2053'                                # token for start-day invariant

# -----------------------------------------------------------------------------
# Test runner: each invocation runs check_invariant_sweep.sh against one
# invariant id and asserts exit + output classification.
# -----------------------------------------------------------------------------

PASS=0
FAIL=0

assert_mismatch() {
    # assert_mismatch <description> <invariant-id>
    local desc="$1" id="$2"
    local out rc
    out="$(bash "$SWEEP" "$id" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -eq 1 ] && printf '%s' "$out" | grep -q "^\[MISMATCH\]"; then
        echo "  PASS: $desc"
        PASS=$((PASS+1))
    else
        echo "  FAIL: $desc (rc=$rc, expected 1; output below)"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL+1))
    fi
}

assert_ok() {
    # assert_ok <description> <invariant-id>
    local desc="$1" id="$2"
    local out rc
    out="$(bash "$SWEEP" "$id" 2>&1)" && rc=0 || rc=$?
    if [ "$rc" -eq 0 ] && printf '%s' "$out" | grep -q "^\[OK\]"; then
        echo "  PASS: $desc"
        PASS=$((PASS+1))
    else
        echo "  FAIL: $desc (rc=$rc, expected 0; output below)"
        printf '%s\n' "$out" | sed 's/^/    /'
        FAIL=$((FAIL+1))
    fi
}

# -----------------------------------------------------------------------------
# Phase 0 — baseline: live gate currently PASSes (no injections yet).
# -----------------------------------------------------------------------------
echo "[phase 0] baseline (no injections) — expect [OK] on all 3 invariants"
assert_ok "baseline tmp_dollar_pid_absence" tmp_dollar_pid_absence
assert_ok "baseline pii_server_endpoint_absence" pii_server_endpoint_absence
assert_ok "baseline hardcoded_start_day_absence" hardcoded_start_day_absence

# -----------------------------------------------------------------------------
# Phase 1 — tmp_dollar_pid_absence × 3 injection sites
# (top-level tools/*.sh + nested tools/<sub>/*.sh)
# Both sites MUST trigger [MISMATCH]; pre-fix, only the nested site fired.
# -----------------------------------------------------------------------------
echo "[phase 1] tmp_dollar_pid_absence injections"

# NOTE: injected lines MUST NOT start with `#` — check_invariant_sweep.sh's
# strip_comments() filters `^[[:space:]]*#` so a `# probe: $TOKEN` injection
# would be silently swallowed and the gate would report [OK] regardless of
# the actual content. Use non-comment shell / yaml syntax instead.

inject "$TOP_LEVEL_SH" "echo 'selftest probe: $TMP_MARKER'"
assert_mismatch "inject TOP_LEVEL_SH -> tmp_dollar_pid_absence" tmp_dollar_pid_absence
restore_one "$TOP_LEVEL_SH"

inject "$NESTED_SH" "echo 'selftest probe (nested): $TMP_MARKER'"
assert_mismatch "inject NESTED_SH -> tmp_dollar_pid_absence" tmp_dollar_pid_absence
restore_one "$NESTED_SH"

# Inject both at once, assert single MISMATCH still catches both.
inject "$TOP_LEVEL_SH" "echo 'selftest probe pair: $TMP_MARKER'"
inject "$NESTED_SH" "echo 'selftest probe pair: $TMP_MARKER'"
assert_mismatch "inject BOTH (top + nested) -> tmp_dollar_pid_absence" tmp_dollar_pid_absence
restore_one "$TOP_LEVEL_SH"
restore_one "$NESTED_SH"

# -----------------------------------------------------------------------------
# Phase 2 — pii_server_endpoint_absence (hex_token + **/* sweep)
# Inject into both a tools/*.sh and a workflow yml; both MUST trigger.
# -----------------------------------------------------------------------------
echo "[phase 2] pii_server_endpoint_absence injections (hex_token path)"

inject "$TOP_LEVEL_SH" "echo 'selftest probe: ssh user@${PII_MARKER}'"
assert_mismatch "inject TOP_LEVEL_SH -> pii_server_endpoint_absence" pii_server_endpoint_absence
restore_one "$TOP_LEVEL_SH"

# yaml inject must NOT be `# ...` (stripped). Use a benign scalar value line.
inject "$WORKFLOW_YML" "selftest_probe_host: ${PII_MARKER}"
assert_mismatch "inject WORKFLOW_YML -> pii_server_endpoint_absence" pii_server_endpoint_absence
restore_one "$WORKFLOW_YML"

# -----------------------------------------------------------------------------
# Phase 3 — hardcoded_start_day_absence (CSV-listed literal paths)
# The spec lists two literal paths under expected_absent_in. Inject into the
# workflow yml first; then verify the CSV-split correctly separates list
# entries (regression vector: comma-quoting drift).
# -----------------------------------------------------------------------------
echo "[phase 3] hardcoded_start_day_absence injections (literal-path CSV)"

inject "$WORKFLOW_YML" "selftest_probe_start: ${START_MARKER}"
assert_mismatch "inject WORKFLOW_YML -> hardcoded_start_day_absence" hardcoded_start_day_absence
restore_one "$WORKFLOW_YML"

SNAP_RUN_SH="$REPO/tools/snapshot_repeatability/run.sh"
if [ -f "$SNAP_RUN_SH" ]; then
    inject "$SNAP_RUN_SH" "echo 'selftest probe: START=${START_MARKER}'"
    assert_mismatch "inject SNAP_RUN_SH -> hardcoded_start_day_absence" hardcoded_start_day_absence
    restore_one "$SNAP_RUN_SH"
else
    echo "  SKIP: $SNAP_RUN_SH not present; phase 3 second-site check omitted (non-blocking)"
fi

# -----------------------------------------------------------------------------
# Phase 4 — final baseline assert: after every restore, all 3 invariants OK.
# -----------------------------------------------------------------------------
echo "[phase 4] final baseline re-check"
assert_ok "final tmp_dollar_pid_absence" tmp_dollar_pid_absence
assert_ok "final pii_server_endpoint_absence" pii_server_endpoint_absence
assert_ok "final hardcoded_start_day_absence" hardcoded_start_day_absence

TOTAL=$((PASS + FAIL))
if [ "$FAIL" -eq 0 ]; then
    printf '\nSELFTEST: %d/%d PASS\n' "$PASS" "$TOTAL"
    exit 0
fi
printf '\nSELFTEST: %d/%d FAIL\n' "$FAIL" "$TOTAL" >&2
exit 1
