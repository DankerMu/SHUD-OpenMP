Reviewer agent: review-security-perf
Review round: round 1
Reviewed head SHA: 49a2d519f3d63a0b6e1cdd6e9f39e7bba138ffe8
Summary: Clean — `set -euo pipefail` enforced, no eval, no injection on awk -v / arith, 1.6 s wall on Mac (well under 5 s budget).

Findings:
- None.

Non-blocking notes:
- **Security – malicious YAML/stats content cannot inject.** Verified empirically: a `cvode_stats.txt` line `nfe=6943; touch /tmp/X` parses as a tainted string (awk -F=, `$2` captures everything after first =). Subsequent `$((v8 - v1))` (L428) and `$((nfeLS - nli))` (L624) reject it as math syntax error, `set -e` exits 1 — fail-closed. `awk -v a="$val"` (L416-417, 445-446, 457-460) does not eval inputs. If untrusted mirrors ever become a vector, add `printf -v _ '%d' "$v"` guard before arith.
- **Security – path traversal contained.** All ~14 uses of `$P8PRE_MIRROR_DIR` (L93-95, 103, 133, 175-176, 206-207, 511) are double-quoted; only ops on concat paths are `[ -d ]`, `[ -r ]`, `grep -qE`, `awk -F=`. No `eval` / `find -exec`.
- **Security – regex anchoring tight.** L137 `grep -qE "^${bad}="` blocks substring false-positives on REJECT keys. L514-515 `awk '$4 ~ /^[0-9]+$/'` strictly filters JID column to digits before stdout/doc interpolation.
- **Perf – fork-heavy but under budget.** ~432 cell-file awk parses + ~272 `read_median` calls (3 subshells each) ≈ 1.2 k subprocess spawns; 1.6 s wall on Mac (0.42 s user — fork/exec dominates).
- **Perf – low-hanging opt (FYI).** Phase D (L202-236) re-opens each cvode_stats.txt 15× and each profile_B0.yaml 9×. A single awk pass per file emitting all keys would cut to 36 spawns. `column_idx_of_metric` (L336-347) re-reads header on every call; memoize once to a bash assoc array. Not blocking at current scale, would matter if matrix grows.
- **Perf – stdout/doc volume bounded.** 6 table rows + ~15 lines stdout; verdict doc ~270 lines deterministic. No runaway.
- **Maint – `case` as loop var** is a bash reserved word (`case ... esac`). Works because parser uses position, but renaming to `_case` removes cognitive overhead. Only `case "$branch" in` at L811 is in inner scope so no collision. Minor nit.
- **Correctness – median-of-3** at L305-307 is the textbook three-compare-swap; produces middle element of any (a,b,c). No off-by-one.
