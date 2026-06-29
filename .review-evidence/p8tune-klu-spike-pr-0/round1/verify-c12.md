Verifier verdict for: c12
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Verdict: CONFIRMED
Evidence: spike_run.sh L47-58 — `CASE="$1"` is assigned but no whitelist follows; only `ORDERING` (L51-54: `natural|amd|colamd`) and `BTF` (L55-58: `0|1`) are validated. L87 then passes the unvalidated `$CASE` straight to `"$DUMP" --case "$CASE" --basin-root "$BASIN_ROOT"`, so a typo (e.g. `heihe_x32`) proceeds past arg-parse and fails at dump_adjacency's basin/cfg.para probe rather than upfront. The usage block L41 already lists the 4 valid cases `{keliya, heihe, heihe_x4, heihe_x16}`, matching REQ-4.
Note: Minor UX gap (asymmetric validation), no security implication — argv is double-quoted; fix is a 3-line `case` block mirroring L51-54.
