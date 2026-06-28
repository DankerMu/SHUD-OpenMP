Reviewer agent: review-documentation
Review round: round 1
Reviewed head SHA: 2eb5d0f
Summary: identity_spike_run.md is a well-structured neutral data-capture run log; style matches Step 1 PR-A reference and stays in PR-E scope; one Critical outer_pin SHA mismatch + one Suggestion on §4 column count.

Findings:

- Critical — outer_pin in YAML frontmatter does not match feat branch HEAD.
  `docs/p8pre/identity_spike_run.md:6` declares `outer_pin: f800bb21d92daaf81d1cbe18cdfddc0a9649eb9e`, but `git rev-parse HEAD` on feat/issue-346-p8pre-pr-e-server-spike returns `2eb5d0fb68edf07482d3c7a45ff954b4c1c933c6`. Build provenance §2 L30-31 repeats the same incorrect SHA. Either the doc was committed earlier and a subsequent commit advanced HEAD without bumping the frontmatter, or f800bb21 is a stale pre-rebase SHA. Update both occurrences to 2eb5d0f (or the final pre-merge SHA) before merge so downstream PR-F #347 and PR-G #348 can pin to the correct outer commit. Needs verification: confirm whether f800bb21 exists as an ancestor commit on the branch.

- Suggestion — §4 raw table column count mismatch with checklist contract.
  `docs/p8pre/identity_spike_run.md:62-63` table headers expose 11 data columns (JID, case, N, rep, nst, nfe, ncfn, nps, npe, t_wall_total, t_precond_setup), but the review checklist item 4 expects 13 columns including `Slurm Elapsed` and `ExitCode`. Both fields exist (sacct State / ExitCode 0:0 mentioned in §4 prose L56) but are collapsed to a single prose sentence. For PR-F aggregator traceability and ADR-0003 audit, restoring `Slurm Elapsed (mm:ss)` and `ExitCode` columns matches the Step 1 PR-A reference convention (`docs/p8pre/n8_profile_run.md:46-65` table) and provides per-cell provenance without forcing readers to join against jid_table.txt.

Non-blocking notes:

- Style match — non-academic execution-log convention is correctly applied: no Abstract / no H1-H3 hypothesis / no Methodology / Results / Discussion / Limitations / Conclusion / Future Work; YAML frontmatter present (8 fields); 9 § sections (Purpose / Build provenance / Gate-1 evidence / 18-cell run table / Preliminary wall-vs-baseline / Cross-N invariance / ncfn observation / Soft gate 6 / References) align with the Step 1 PR-A reference structure. CLAUDE.md user-pref consistency satisfied (run log, not stage summary).
- Neutral data capture — gates 2/3/4/5/6 are correctly framed as "data captured, PR-F adjudicates" (§5 L86 "PR-F #347 owns gate-4 verdict", §7 L159 "PR-F #347 adjudication", §8 L173-174 "preliminary; PR-F formal verdict"). Gate 1 (build PASS + 3-symbol linked) is correctly called CAPTURED at §3 L50-51 — this is PR-E scope per spec L66-72. No "Verdict" or "Recommendation" section present.
- Spec cross-refs verified accurate: gate 2 ncfn=0 at spec L74-79 (matches §7 L142); gate 4 case-aware ε at spec L92-100 (matches §5 L102-103); soft gate 5 SHA12 → max_ulp ≤ 1024 at spec L102-106 (matches §6 L136); soft gate 6 ratio ≤ 0.05 at spec L108-113 (matches §8 L163-166). SHUD pin 5276167 cited at L5 + L29 + L189.
- PREC_NONE→PREC_LEFT shift framing at §6 L119-136 is technically sound: explains CVLS state-machine path divergence and explicitly hands SHA12 strict-equality verdict to PR-F soft gate 5 with max_ulp fallback context — exactly what PR-F needs.
- Sample value cross-check: cell_stats.txt row 2 (heihe N=1 rep=1) → nst=6599 nfe=6696 ncfn=6 nps=18163 npe=77 wall=137.079454928 t_precond_setup=0.000008872 matches §4 L64 row JID 9531 exactly. Row 12 (heihe_x4 N=1 rep=2) → wall=1428.286663562 matches §4 L74 row JID 9541. No transcription errors found in sampled cells.
- Markdown integrity — all tables render (header + alignment + data rows); code block at §3 has no language hint but content is `nm` output, language hint not required; section numbering §1-§9 sequential, no gaps; no emoji present.
- §9 References has one path that is server-local (`/tmp/p8pre_identity_spike/`) cited as "raw evidence archive" at L190 — this is a known-acceptable handoff pattern matching Step 1 PR-A reference (`docs/p8pre/n8_profile_run.md:117-122` cites `/tmp/p8pre_n8_profile/`); reviewer-machine reachability not required since the canonical aggregator input is also cited at L191 `.review-evidence/p8pre-pr-e-spike/cell_stats.txt`.

Verdict: REQUEST CHANGES — outer_pin SHA mismatch is a blocking provenance error; §4 column omission is a non-blocking improvement.
