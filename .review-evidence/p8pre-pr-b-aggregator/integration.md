Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: 49a2d519f3d63a0b6e1cdd6e9f39e7bba138ffe8
Summary: PR-B integration contract holds — branch letter is grep-parseable, verdict doc §3 is self-contained for PR-C/PR-F handoff, input schema matches PR-A run output, no CI workflow regression.

Findings:
- None.

Non-blocking notes:

- Branch-letter forward compat (checks 1, 4): aggregator emits `branch: <letter>` to stdout (`tools/p8pre/aggregate_n8_profile.sh:548-550`) on its own line. Verdict doc has 2 grep-stable anchors: `status: "branch: a ..."` in YAML frontmatter (L5) and the §3.5 bold header `**branch: a ...**` (L135). ADR-0003 draft generator (PR-G #348) can use `grep -Eo '^status:.*branch: [a-d]'` against frontmatter (single hit) or `grep -Eom1 '^\*\*branch: [a-d]'` against §3.5 (single hit). Token format unambiguous: `[a-d]` is a literal post-colon char, not "branch" prose. Note: `branch` substring also appears in §1 enumeration L39-42 as bulleted descriptions ("branch (a)" with parentheses) — that pattern is distinct from `branch: <letter>` so no false positive. BLOCKED states emit `branch: BLOCKED (...)` (L542) — downstream consumers must treat BLOCKED as non-letter sentinel.

- Input contract matches PR-A (check 2): pre-flight (`tools/p8pre/aggregate_n8_profile.sh:100-122`) iterates `${P8PRE_MIRROR_DIR}${case}_N${n}_rep${rep}/{profile_B0.yaml,cvode_stats.txt}` — exact match to PR-A's `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/` mirror layout documented in `docs/p8pre/n8_profile_run.md:120`. Filenames `profile_B0.yaml` and `cvode_stats.txt` are exact. `jid_table.txt` lookup (L511-517) gracefully degrades to `n/a` if absent (non-fatal).

- Output contract matches PR-C consumer (check 3, 5): verdict doc §3.1-§3.4 (L70-131) are pure data tables, no prose-only conclusions. §3.5 §3.5 (L133-141) has explicit `r_min = 1.819 ; r_max = 4.526` on L140 — PR-F can cite both directly. §3 sections are numbered, self-contained, programmatically consumable. PR-C can quote §3 tables verbatim.

- `P8PRE_MIRROR_DIR` env var convention (check 6): default `/tmp/p8pre_n8_profile/` documented in header (L5-6); trailing slash normalized (L44 `${VAR%/}/`); missing dir → clean FATAL exit 1 (L94-97); per-cell missing → clean FATAL exit 1 (L106-113). No symlink-specific handling but bash `-d`/`-r` follow symlinks transparently.

- Sibling script compliance (check 7): `set -euo pipefail` (L36) matches `run_n8_profile.sh:37`. Exit codes 0/1/2/3 documented in header (L28-33) and emitted via `block_exit` precedence (L486-504). Header style matches sibling tools.

- CI workflow compatibility (check 8): `.github/workflows/serial-baseline.yml` does NOT glob `tools/p8pre/*.sh` (grep returned no matches). No CI runs this aggregator. The `/tmp/p8pre_n8_profile/` dependency is server-side only; safe for CI.

- `cvode_stats_diff/` separation (check 9): aggregator inlines `CANONICAL_KEYS` (L58) with explicit comment (L54-58) noting drift must be caught by upstream test. No functional duplication: cvode_stats_diff diffs 2 cvode_stats.txt files; this aggregator computes medians + ratios + branch verdict across 18 cells. Different problem space.

- Gitignore + scope (check 10, 11): no `.gitignore` change in this PR (already covered by #341 PR #353 via `.p8pre-runs/` + `.p8pre-pr-*-runs/`). `.review-evidence/` is gitignored repo-wide. `openspec/changes/p8pre-spike/` untouched (orchestrator-only zone preserved). PR-B scope is 2 new files (aggregator + verdict doc) — clean.

- Branch-letter ambiguity edge case: §3.5 has 2 grep-parseable lines (L5 frontmatter + L135 §3.5 header). PR-G ADR-0003 generator should anchor on the frontmatter `^status:` line (deterministic position) rather than the `**branch:` markdown bold form. Frontmatter is the canonical parse point.
