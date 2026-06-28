Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: d8602d0fb4e609c106eaa5f79e973830182ac150
Summary: All 8 checklist items pass — runner logic, template cwd/binary path, bucket-sum gate semantics, cross-N CVODE bitwise invariance, rsync mirror integrity, doc accuracy, and SHUD pin all verified clean.
Findings:
- None.
Non-blocking notes:
- BSUM% gate is algebraically a clamp-detection canary, not an independent double-count check: `t_other` is defined in `tools/profile/timer.cpp:152-156` as the wall residual (`wall - CVODE_raw - forcing - ET - output`, clamped ≥0) and `t_CVODE_internal` at L149-151 as `CVODE_raw - RHS_total` (clamped ≥0). When neither clamp fires, the six excluded-from-RHS_kernel buckets sum to `t_wall_total` by construction → BSUM ≡ 0. Implementer's `docs/p8pre/n8_profile_run.md` §4 acknowledges this honestly ("零钳触发 = 测量良性"). The gate still catches negative-residual clamp events (which would be the actual bug surface), so the test is well-typed; the framing in the doc is sufficient and does not need to change. Flagging here only so PR-B verdict text does not regress into claiming "BSUM=0 proves no double-count".
- Runner exit-4 path at L200-204 exits BEFORE recording an ERR row in jid_table.txt when a PREV_JID lookup misses (`exit 4` fires before the `printf ... ERR_` lines used in the sbatch-rc fallback paths). The script header comment at L32-34 says "partial jid_table.txt left as-is for triage" — that promise is honored (already-written rows remain) but no marker line is added for the failing cell. Minor doc/code drift, not a correctness defect.
