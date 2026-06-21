# Round 1 Reviewer — PR #194 (S5c-B RHS 7-bucket timer + forcing I/O)

Reviewed outer SHA: `b726979` · SHUD SHA: `56ab14e`

## Verdict
**clean** (0 blocking findings; 2 informational Suggestions)

## Suggestions (non-blocking)
1. **SHUD_DUMP_RHS probe outside bucket scope** (MD_rhs_core.cpp:290-292): the `shud_rhs_dump_point("f_loop_before_passvalue",...)` call sits between river-bucket close (L273) and gather-bucket open (L300). Under combined `-DSHUD_ENABLE_DIAGNOSTICS -DSHUD_DUMP_RHS` builds the probe is uncounted. Default build invisible. Mention in M7 ADR if percentages drift on SHUD_DUMP_RHS-enabled runs.
2. **`%.3f` percentage rendering jitter**: produces 99.999%/100.001% on server (vs exact 100.000% local). Well within spec [99.5%, 100.5%]. Informational only.

## Coverage Confirmed
- MD_diagnostics.hpp empty-under-OFF: entire `namespace shud_diag` body gated by single outer `#ifdef SHUD_ENABLE_DIAGNOSTICS` at L27 closed at L71. `<chrono>` include is INSIDE the macro guard.
- 7-bucket mapping accurate to spec verbatim names (update/ET/lateral/segment/river/gather/applyDY); each name appears as scope comment at the corresponding code region.
- No RHS arithmetic mutation: 3 removal lines exactly match restorations within wrap braces (`rhs_deterministic_gather()` / `rhs_update(Y,DY,t)` / `rhs_apply(DY,t)`). Loop bodies inside new `{...}` byte-identical to pre-PR.
- read_csv() wrap function-body level: `ScopeTimer` at L65 BEFORE the `if (!eof)` early-return → measurement includes early-return path.
- cvode_stats.txt key=value format consistent with existing 15-key + #173 hlast/qlast.
- pct sum formula correct (`100.0 * (double)g_rhs_timer_ns[X] / total_d`); divide-by-zero guard emits 0.0 if t_total=0.
- PR boundary respected: 5 files only (B1b_CHANGELOG + new hpp + MD_rhs_core.cpp + TimeSeriesData.cpp + cvode_config.cpp); no f.cpp/Model_Data/Makefile/outer-repo edit.
- t_forcing_io_s IN-PROGRESS root-cause correct: 90d × 8 rec/day = 720 ≪ MAXQUE=10000 → single-pass per CSV.
- Server Slurm 8567 evidence: changelog records job/cn08/wall/exitcode/3 bitwise PASS + pct sums.

## Praise (informational)
- Header design: textbook empty-when-OFF — zero new symbols, zero static init, zero include side effects under OFF.
- Bucket-mapping comments at each ScopeTimer scope provide good signal-to-noise for future #175/S6c/M7 work.
- CHANGELOG IN-PROGRESS note dimensionally correct, 3 resolution options listed, scope deferral stated, diagnostic-only invariant restated.
