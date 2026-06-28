# Verify cand-02: spec/tasks filename drift

## Verdict: CONFIRMED

## Evidence

**Drift confirmed at frozen HEAD d8602d0**:

1. `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md:17` and `:53` reference `profile.yaml` (bare) in the WHEN clause of the aggregator scenario.
2. `openspec/changes/p8pre-spike/tasks.md:25` (§2.3) and `:30` (§3.1) also reference `profile.yaml`. §3.1 explicitly states: "parse 18 profile.yaml + 18 cvode_stats.txt from `/tmp/p8pre_n8_profile/<case>_N<n>_rep<r>/`".
3. **Actual emit filename = `profile_B0.yaml`**: `SHUD/src/Model/shud.cpp:352` and `:579` both: `snprintf(prof_path, sizeof(prof_path), "%s/profile_B0.yaml", ...)`.
4. **Already aligned in PR-A artifacts**: `tools/p8pre/submit_n8_profile_template.sbatch:112` comment + `:119` verification loop both use `profile_B0.yaml`. `tools/profile/timer.h:14` documentation example also uses `profile_B0.yaml`.
5. **/tmp mirror confirms reality**: `/tmp/p8pre_n8_profile/heihe_N1_rep1/` contains `profile_B0.yaml` (not `profile.yaml`).

**Scenario realism**: PR-B implementer reading tasks.md §3.1 verbatim would code `glob("/tmp/.../profile.yaml")` → 0 matches → empty median → silent corruption of `wall_step1_baseline_median`. Risk is realistic because §3.1 is the canonical task contract for PR-B and the path is verbatim.

**Scope ambiguity**: The fix touches spec/tasks (orchestrator-owned per workflow), not PR-A implementation. PR-A authored the template+sbatch artifacts that already use the correct `profile_B0.yaml`, so PR-A discovered the drift. Fix could land here (single-purpose breach: spec edit mixed with run impl) or in a sibling spec-fix PR before PR-B starts. Drift is real and material; verdict is CONFIRMED on the factual claim. Scope decision belongs to orchestrator.

## Sibling surfaces
- spec.md L17 (`SHUD_ENABLE_PROFILE=1 ... emits profile.yaml`) — same drift, narrative description.
- spec.md L53 (aggregator path) — primary drift, machine-readable contract.
- tasks.md L25 §2.3 + L30 §3.1 + L62 §6.6a + L63 §6.7 — `profile.yaml` recurrences in PR-B/PR-D scope.
