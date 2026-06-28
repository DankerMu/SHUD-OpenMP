## Phase 4 Cross-Review Evidence Bundle (round 1)

Reviewer agents: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf`
Review round: round 1
Reviewed head SHA: `49a2d51`
Local evidence: `.review-evidence/p8pre-pr-b-aggregator/{spec-compliance,correctness,integration,security-perf}.md`

### review-spec-compliance — APPROVE

Summary: All 8 spec-compliance items PASS — aggregator faithfully implements every scenario in `n8-mode-c-profile-recheck/spec.md`; live run output (exit 0, branch: a) proves end-to-end gate behavior on the 18-cell mirror.

Findings: **None.**

Non-blocking notes:
- Canonical 15-key set is inline-duplicated from `tools/cvode_stats_diff/canonical_15_keys.yaml` rather than parsed; L55-57 acknowledges this with flat-dependency rationale, defers to yaml unit test as drift gate.
- REJECT_KEYS covers the 5 spec-mandated typo keys exactly (`nlcf` / `nfevals` / `hcur` / `qcur` / `hin`); diagnostic keys `nFCall*` / `hlast` / `qlast` / `t_rhs_*` not enforced (spec L38 doesn't mandate them — script correctly scoped).

### review-correctness — APPROVE

Summary: Aggregator numerics, REJECT logic, branch tree, and verdict doc all correct; no findings.

Findings: **None.**

Non-blocking notes:
- Median impl (`aggregate_n8_profile.sh:301-307`): 3-element sort-by-3-swaps correctly puts median in `b`; verified manually.
- Cross-N invariance (L370): integer string `!=` comparison, correct for Δ=0 strict on integer counters.
- Ratio scale (L416): `awk printf "%.3f"`. 12632/6943 → "1.819" + 30509/6741 → "4.526" match stdout/verdict doc exactly.
- Branch tree (L462-480): r_min=1.819 ≥ 1.5 fires `ge_proceed=1` → branch="a". Tree mutually exclusive + exhaustive (a → b → d → c default); branch d (heterogeneity) correctly precedes branch c.
- REJECT typo regex (L137): `^${bad}=` anchored on start-of-line. `nlcf=` matches; `ncfl=` does NOT match — no false positive on the valid `ncfl` canonical key.
- `nfeLS = nli` identity (verdict §3.4): SUNDIALS CVLS SPGMR + FD-Jvp identity; data confirms `nfeLS − nli = 0` for all 6 groups.
- §5 expected 10-25% speedup is forward-looking for future non-trivial preconditioner; PR-D #345 identity-precond spike will only verify API gating (`ncfn=0, nps>0, npe>0`).
- No SHUD pin change; no openspec/changes/p8pre-spike/ edits; no canonical_15_keys.yaml edits — diff cleanly scoped to 2 new files (+1130).

### review-integration — APPROVE

Summary: PR-B integration contract holds — branch letter is grep-parseable, verdict doc §3 self-contained for PR-C/PR-F handoff, input schema matches PR-A run output, no CI workflow regression.

Findings: **None.**

Non-blocking notes:
- Branch-letter forward compat: stdout emits `branch: <letter>` on its own line (L548-550). Verdict doc has 2 grep-stable anchors — YAML frontmatter `status: "branch: a ..."` (L5) + §3.5 bold header `**branch: a ...**` (L135).
- BLOCKED states emit `branch: BLOCKED (...)` (L542) — downstream consumers treat BLOCKED as non-letter sentinel.
- Input contract pre-flight iterates `${P8PRE_MIRROR_DIR}${case}_N${n}_rep${rep}/{profile_B0.yaml,cvode_stats.txt}` — exact match to docs/p8pre/n8_profile_run.md mirror layout.
- Output contract: §3.1-§3.4 pure data tables; §3.5 (L133-141) explicit `r_min = 1.819 ; r_max = 4.526` on L140 — PR-F can cite both directly.
- `P8PRE_MIRROR_DIR` documented (L5-6); trailing slash normalized (L44); missing dir → clean FATAL exit 1 (L94-97).
- Sibling convention: `set -euo pipefail` (L36) matches `run_n8_profile.sh:37`. Exit codes 0/1/2/3 documented (L28-33) + emitted via `block_exit` precedence (L486-504).
- CI compat: `.github/workflows/serial-baseline.yml` does NOT glob `tools/p8pre/*.sh`. No CI exposure to `/tmp` dependency.
- `cvode_stats_diff/` separation: aggregator inlines `CANONICAL_KEYS` (L58) with drift-gate comment (L54-58). Different problem space — no duplication.
- `.gitignore` covered by #341 PR #353 already.
- ADR-0003 generator hint: PR-G should anchor on `^status:` frontmatter (deterministic position) rather than markdown bold form.

### review-security-perf — APPROVE

Summary: Clean — `set -euo pipefail` enforced, no eval, no injection on awk -v / arith, 1.6s wall on Mac (well under 5s budget).

Findings: **None.**

Non-blocking notes:
- Security: malicious YAML/stats content cannot inject. `nfe=6943; touch /tmp/X` parses as tainted string but `$((v8 - v1))` + `$((nfeLS - nli))` reject as math syntax error; `set -e` exits 1 (fail-closed).
- Security: path traversal contained. All ~14 uses of `$P8PRE_MIRROR_DIR` double-quoted; only ops are `[ -d ]`, `[ -r ]`, `grep -qE`, `awk -F=`. No `eval` / `find -exec`.
- Regex anchoring tight: L137 `grep -qE "^${bad}="` blocks substring false-positives. L514-515 `awk '$4 ~ /^[0-9]+$/'` strictly filters JID column.
- Perf: ~432 cell-file awk parses + ~272 `read_median` calls (3 subshells each) ≈ 1.2k subprocess spawns; 1.6s wall, 0.42s user.
- Perf microopt (FYI): Phase D (L202-236) re-opens each cvode_stats.txt 15× + profile_B0.yaml 9×; single-pass awk would cut to 36 spawns. `column_idx_of_metric` (L336-347) re-reads header every call; memoize to bash assoc array.

### Phase 4.5 Verifier — SKIPPED

Rationale: 0 candidates with concrete failure scenarios — 4/4 reviewers APPROVE, all findings = None. Phase 4.5 precision-bias on expanded fixture still requires verifier on PLAUSIBLE candidates, but none exist.

### Round 1 verdict

**Clean.** 0 CONFIRMED + 0 merge-blocking PLAUSIBLE. Proceed to Phase 7.
