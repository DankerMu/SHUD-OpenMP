Reviewer agent: review-integration
Review round: round 1
Reviewed head SHA: d8602d0fb4e609c106eaa5f79e973830182ac150
Summary: Runner ↔ wrapper JID protocol matches, bucket-sum policy faithful, run wall data clean — but `.gitignore` carry-forward from §1.4 is MISSING and spec/tasks still cite `profile.yaml` while doc + sbatch + rsync all use `profile_B0.yaml`, leaving a documentation drift that PR-B aggregator will trip on.

Findings:

- 🟡 Warning — `.gitignore` carry-forward MISSING (checklist #2)
  `.gitignore:1-44`
  Contains `.s*-runs/` glob but NOT `.p8pre-runs/`. `git check-ignore .p8pre-runs/x` → exit 1 (no match). The `.s*-runs/` glob does not shell-match `.p8pre-runs/` (`s*` requires literal leading `s`). On server `git status` will show `.p8pre-runs/` as untracked clutter. Sibling lines 27-28 explicitly enumerate `.p1d-runs/` + `.p1e-runs/` because the same gap existed before. Fix: add `.p8pre-runs/` + `.p8pre-pr-*-runs/` + `.p8pre-pr-*-worktree/` after L34. Tasks.md §1.4 PR-A prep carry-forward (per #340) appears never honored.

- 🟡 Warning — spec/tasks vs implementation filename drift (checklist #1)
  `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md:53` + `tasks.md:25,30`
  Spec L53 says aggregator reads `profile.yaml`; tasks §2.3 + §3.1 also say `profile.yaml`. But SHUD `shud.cpp:352,579` hardcodes `profile_B0.yaml`; sbatch template L119 + verification gates + rsync log + tasks §6.6a/§8.7 all produce/expect `profile_B0.yaml`. The 18-cell run physically created `profile_B0.yaml` everywhere. PR-B authored against tasks.md §3.1 will read 0 files unless caught. Doc `n8_profile_run.md` §4 is on-message but spec is the laggard. Recommend doc-only correction PR before PR-B starts: update spec L53 + tasks §2.3,3.1 to `profile_B0.yaml`.

- 🟢 Praise — JID placeholder protocol verified end-to-end (checklist #3)
  `tools/p8pre/render_n8_profile.sh:128-133` ↔ `run_n8_profile.sh:198-205,241`
  Render emits `__PREV_JID_<case>_N<n>_rep<r-1>__`; runner regex `__PREV_JID_([A-Za-z0-9_]+)__` captures full key incl. `heihe_x4` underscore and looks up `PREV_JID[<case>_N<n>_rep<r>]` populated at L241. Submit log confirms heihe_x4 N1 rep2 → JID 9519 dependency. Belt-and-braces `singleton` --job-name guard at L123 even survives JID-capture bugs.

- 🔵 Suggestion — verification.txt "worst_bsum_abs_err=0% at cell=" empty cell field (checklist #5)
  `.review-evidence/p8pre-pr-a-run/verification.txt:34`
  Cosmetic: when all cells = 0%, "worst" cell name is empty. No semantic impact.

Non-blocking notes:
- rsync timing OK (checklist #8): monitor "ALL 18 DONE 22:45:45", rsync_log mtime `Jun 26 22:50:07` — 5 min AFTER last job.
- SHUD pin `7a1dc8f` recorded in submit_log L4 + doc §2 (checklist #9): PR-D forward-only descent provable.
- Bucket-sum policy (checklist #10): doc §4 excludes `t_RHS_kernel`, matches timer.cpp:152-156 algebra exactly.
- Template change scoped to `tools/p8pre/submit_n8_profile_template.sbatch` only; `tools/p1e_2x2_sbatch_template.sbatch` untouched (checklist #4).
- `tools/cvode_stats_diff/canonical_15_keys.yaml` untouched (checklist #7).
- `.github/workflows/serial-baseline.yml` does not glob `tools/p8pre/**` or `docs/p8pre/**` (checklist #6).
- Gate names ART/CANON15/REJECT/EXTRAS/BSUM%/RC0 in doc §4 match verification.txt header L11.
- `n8_profile_run.md` §8 internal links resolve.
