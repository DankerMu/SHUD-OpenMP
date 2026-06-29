Verifier verdict for: c02
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Verdict: CONFIRMED
Evidence:
  1. SHUD submodule HEAD confirmed = bc919f5b8c6e1af9e7bb8f11e0452ac69dd3e18e
     (`cd SHUD && git rev-parse HEAD` -> bc919f5...).
  2. SHUD tracked .gitignore does NOT contain libshud.a or _libshud_obj
     patterns. Full dump of `git show HEAD:.gitignore` shows entries for
     Build/, shud, shud_omp, shud_asan, tests/s1d_*_smoke, InstallSundials/,
     cvode-6.0.0/ — but no `libshud.a` and no `_libshud_obj/`. `grep -nE
     "(libshud|_libshud|\.a$|obj)"` returned "(no match in HEAD .gitignore)".
  3. Local-only mask present at
     /Users/danker/Desktop/Hydro-SHUD/openMP/.git/modules/SHUD/info/exclude
     containing:
        /libshud.a
        /_libshud_obj/
     plus a self-documenting comment: "P8-tune.D KLU spike (openspec change
     p8tune-klu-spike) — build artifacts of the additive `make libshud.a`
     target. Per .git/info/exclude (NOT committed .gitignore): this exclude
     is local-only and does not pollute the SHUD upstream tree."
     This file lives under .git/modules/ and is per-clone — never pushed.
  4. Outer-repo .gitignore (openMP/.gitignore) also does NOT cover these
     patterns (`grep -nE '(libshud|SHUD/_libshud)'` -> no match). Outer
     can't help anyway because the artifacts land INSIDE the SHUD
     submodule working tree, where outer ignore rules don't apply to
     submodule-internal status.
  5. Scenario constructibility: any fresh clone (e.g. PR-A sweep on
     cn-nodes) lacks .git/modules/SHUD/info/exclude. After `make
     shud_spike`, `cd SHUD && git status` will list `?? libshud.a` and
     `?? _libshud_obj/` (untracked), and outer `git status` will report
     `M SHUD` (submodule has untracked changes) — the exact failure
     described in the candidate.
Note: The committed local-exclude comment itself acknowledges the gap (calls out "NOT committed .gitignore"), which doubles as on-disk confirmation that the omission is known to be load-bearing.
