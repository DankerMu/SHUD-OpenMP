## Phase 4 Cross-Review Evidence Bundle (rounds 1 + 2)

Reviewer agents: `review-spec-compliance`, `review-correctness`, `review-integration`, `review-security-perf`
Review rounds: round 1 (initial) + round 2 (post-fix integration)
Reviewed head SHA: `83a8864` (post Phase 6 fix; round 1 was `d8602d0`)
Local evidence: `.review-evidence/p8pre-pr-a-run/{spec-compliance,correctness,integration,security-perf,integration-round-2}.md`

### Round 1 (head `d8602d0`)

#### review-spec-compliance — APPROVE

Summary: All 6 spec scenarios PASS against rsync mirror; one brief-vs-spec inconsistency (Scenario 5 nFCall) is REFUTED because spec text contains no such requirement.

Findings: **None.**

Verification: heihe `nst=6698/nfe=6943`, heihe_x4 `nst=6575/nfe=6741` bitwise across N=1/4/8 — matches step1_prep.md §3 + §4 absolute baseline anchor.

#### review-correctness — APPROVE

Summary: All 8 checklist items pass — runner logic, template cwd/binary path, bucket-sum gate semantics, cross-N CVODE bitwise invariance, rsync mirror integrity, doc accuracy, SHUD pin all verified clean.

Findings: **None.**

Non-blocking notes:
1. BSUM% gate is algebraically a clamp-detection canary (`tools/profile/timer.cpp:149-156` defines `t_other` as residual), not an independent double-count check. Implementer's `n8_profile_run.md` §4 acknowledges this.
2. Runner exit-4 at L200-204 exits before writing ERR row for missing-PREV_JID; header comment promises "partial jid_table.txt left as-is" — technically honored (prior rows preserved) but no marker for failing cell. Minor.

#### review-integration — REQUEST CHANGES (2 Warnings → Phase 4.5 → both CONFIRMED → Phase 6 fix → round 2 RESOLVED)

Summary round 1: Runner ↔ wrapper JID protocol matches, bucket-sum policy faithful, run wall data clean — but 2 issues block clean handoff to PR-B (#342).

Findings:
- **Warning cand-01**: `.gitignore` carry-forward MISSING — `.gitignore` has `.s*-runs/` glob but not `.p8pre-runs/`; `git check-ignore .p8pre-runs/x` exits 1. Sibling p1d/p1e are explicitly listed (.gitignore:27-28); same fix needed for p8pre.
- **Warning cand-02**: spec/tasks vs implementation filename drift — `openspec/changes/p8pre-spike/{specs/n8-mode-c-profile-recheck/spec.md:17,53, tasks.md §2.3/§3.1}` say `profile.yaml`, but SHUD emits `profile_B0.yaml` (shud.cpp:352,579) and PR-A sbatch/rsync/doc all use `profile_B0.yaml`. PR-B aggregator authored against tasks will read 0 files.

Non-blocking notes:
- rsync 5min AFTER last job completion (iron rule respected).
- SHUD pin 7a1dc8f recorded; PR-D forward-only descent provable.
- Bucket-sum excludes `t_RHS_kernel`, matches `timer.cpp:152-156` exactly.

#### review-security-perf — APPROVE

Summary: Security-clean and performance-prudent; all checklist items pass with one defensive-code note on unquoted sbatch arg expansion and one observation re P1e wall delta (already explained by step1_prep.md §0).

Findings: **None.**

Key non-blocking notes:
1. `run_n8_profile.sh:215` unquoted `$sbatch_args` expansion is safe today (fixed-alphabet trusted input + JID regex-validated at L228 before stash); migrate to bash array if future PRs accept less-trusted args.
2. Slurm 三铁律 verified end-to-end via `submit_log.txt:1` cwd + template L48-49 #SBATCH paths.
3. Wall delta vs P1e PR-I (heihe N=1 147s vs 504s, 3.4× faster) is EXPECTED, not regression. `docs/p8pre/step1_prep.md` §0: P1e PR-I baseline built at SHUD `3341368d` WITHOUT `SHUD_ENABLE_PROFILE=1` — Timer instrumentation overhead bias. PR-A run IS the new gate-4 baseline. heihe_x4 aligns tightly (+7.6%/-0.3%).
4. Singleton chain (design D4) honored; 70min total wall matches 3×~24min heihe_x4 N=1 dominant chain.

### Phase 4.5 Verifier (round 1 candidates)

| candidate | verdict | rationale |
|---|---|---|
| cand-01 `.gitignore` carry-forward | **CONFIRMED** | `.gitignore:16` `.s*-runs/` anchors literally on `s`; `git check-ignore .p8pre-runs/x` exits 1; sibling p1d/p1e enumerated; fix in-scope per #340 carry-forward |
| cand-02 spec/tasks filename drift | **CONFIRMED** | shud.cpp:352,579 emits `profile_B0.yaml`; openspec/changes/p8pre-spike/ cites `profile.yaml` at 6+ sites; PR-B implementer at tasks §3.1 would glob `profile.yaml` → 0 matches → silent verdict corruption |

### Phase 6 fix (commit `83a8864`)

- **cand-01**: `.gitignore` L33-35 inserted (1 comment + 2 patterns) after p1e block. `git check-ignore .p8pre-runs/x` + `.p8pre-pr-a-runs/y` both exit 0.
- **cand-02**: 9 substitutions across 4 files: `tasks.md` (§2.3/§3.1/§6.6a/§8.7 = 5), `specs/n8-mode-c-profile-recheck/spec.md` (L17/L53 = 4), `design.md` (D6 = 1), `specs/p8precond-zero-identity-spike/spec.md` (L110/L113 = 2). NOTE: `openspec/changes/` is project-gitignored per `.gitignore:13` "OpenSpec transient changes" policy — fix lives in local working tree, archives to persistent `openspec/specs/` at #349. All subsequent p8pre-spike implementers fire from this working tree → fix effective for PR-B/C/D/E/F/G.

### Round 2 integration re-review (head `83a8864`)

#### review-integration — APPROVE (post-fix)

Summary: Both round-1 Warnings RESOLVED; fixes minimal, additive, no new integration risks.

Findings: **None.**

Resolution status:
- cand-01: **RESOLVED** (`.gitignore` +4 lines; 4 check-ignore tests all exit 0; p1e sibling block intact)
- cand-02: **RESOLVED** (grep `profile\.yaml openspec/changes/p8pre-spike/` = 0; grep `profile_B0\.yaml` = 11 hits across 4 files; spec heading structure intact)

Non-blocking notes:
- `openspec validate p8pre-spike --strict` PASS.
- `openspec/specs/` (epic-archived dir list) contains no stale `profile.yaml` references for p8pre scope.

### Cumulative verdict

**Clean.** Round 2 closes the loop on round-1 Warnings. 0 outstanding findings. Proceed to Phase 7.
