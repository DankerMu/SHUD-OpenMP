Round 1 verdict summary
Reviewed head SHA: a65f3ca175405e128ec15b7fe7f07c8932903bf0
Fixture level: high / broad-expanded → recall-biased (CONFIRMED + PLAUSIBLE block unless out-of-scope)
Reviewers run: spec-compliance, correctness, integration, security-perf, test-evidence, invariant-state (6-reviewer high-risk pack)
Candidates collected (post-dedup): 15
Verifier subagents: 15 (1-per-candidate, parallel)

| ID | Failure class | Cited by | Severity | Verdict | Action |
|----|---------------|----------|----------|---------|--------|
| c01 | Spec gate not implemented (dense FD cross-check stub) | spec-compliance + correctness + test-evidence | critical | **CONFIRMED** | FIX F1 — real dense FD probe + recompute TSV |
| c02 | SHUD .gitignore missing libshud.a / _libshud_obj/ | integration + invariant-state | major | **CONFIRMED** | FIX F2 — SHUD-side commit + outer pointer bump |
| c03 | fd_color_jacobian CSC path mismatch (Mac smoke passed only due to stale CSC) | security-perf | major | **CONFIRMED** | FIX F3 — mirror klu_analyze_factor:164 path pattern |
| c04 | RSS preflight estimator under-counts ~4× + signal-137 escape | security-perf | major | **CONFIRMED** | FIX F4 — move preflight after klu_analyze; recompute from Symbolic->lnz+unz |
| c05 | χ≤30 keliya bound not programmatically asserted | spec-compliance | major | PLAUSIBLE | FIX F5 (high fixture blocks PLAUSIBLE) — add gate in --report-chi-only path |
| c06 | OOM-as-data-point untested on Mac smoke | test-evidence | major | PLAUSIBLE | DEFER — verifier confirms spec scopes OOM-path test to PR-A §2.9 |
| c07 | spike_run.sh OOM exit-code confusion | integration | minor | **REFUTED** | DROP — KLU exits 0 on OOM per spec L141 + impl |
| c08 | spec REQ-7 envelope omits .gitignore + dense_fd_cross_check.py | spec-compliance | minor | **CONFIRMED** | FIX F6 — spec amendment |
| c09 | Stale comment hazard re globalY transitive symbol | correctness | minor | **CONFIRMED** | FIX F9 — rewrite comment |
| c10 | 75.6% n_zero FD entries diagnostic gap | correctness | minor | PLAUSIBLE | DEFER — verifier confirms spec defines nnz(A)=pattern-nnz; PR-B aggregator scope |
| c11 | REQ-1 vs REQ-7 contradiction (pin freeze vs Makefile carve-out) | integration | minor | **CONFIRMED** | FIX F7 — REQ-1 L18 carve-out caveat |
| c12 | spike_run.sh missing case-name whitelist | security-perf | minor | **CONFIRMED** | FIX F10 — 1-line case whitelist |
| c13 | spec phrasing `.sa/.riv/.lake` vs actual `.sp.{mesh,riv,rivseg,att}` | test-evidence | minor | **CONFIRMED** | FIX F8 — spec REQ-3 errata |
| c14 | README determinism repro recipe undocumented | test-evidence | minor | **CONFIRMED** | FIX F11 — 3-line README §troubleshooting recipe |
| c15 | omp_set_num_threads(1) asymmetry across 3 binaries | invariant-state | minor | **REFUTED** | DROP — dump_adjacency + klu_analyze_factor have no OMP regions |

Pattern escalation check: 3+ same failure class? Three spec-amendment findings (c08/c11/c13) are spec-drafting drift, NOT implementation-pattern; no pattern freeze triggered. No invariant audit required for Phase 6.2.

Phase 5 fix groups (by failure class):
- Group A (CRITICAL spec gate): F1
- Group B (MAJOR pipeline integrity): F2, F3, F4
- Group C (MAJOR/PLAUSIBLE soft gate): F5
- Group D (MINOR spec amendments): F6, F7, F8
- Group E (MINOR hygiene): F9, F10, F11

Deferrals recorded for PR description:
- c06 → PR-A §2.9 16-cell sweep gate (per verifier "spec doesn't require PR-0 to test OOM path")
- c10 → PR-B aggregator may add effective_nnz axis (per verifier "REQ-5 satisfied since spec defines nnz(A)=pattern-nnz")

REFUTED rationale:
- c07: spec REQ-5 L141 + klu_analyze_factor L238/275/287 all return 0 on OOM → spike_run.sh pipefail sees exit 0 → no abort → wrapper exits 0 (advertised contract honored)
- c15: grep tools/p8tune.D/*.cpp confirms only fd_color_jacobian.cpp:55 includes <omp.h>; klu/dump_adjacency have no OMP regions to pin
