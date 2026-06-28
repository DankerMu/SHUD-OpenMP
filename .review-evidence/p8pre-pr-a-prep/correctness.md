Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 20a7ec1e03a7d65b52c638cdabb4af3c3b37aa0d
Summary: Template + wrapper + evidence doc are correct against design D4/D5/D7 and Slurm 三铁律; all 14 checklist items pass with one minor evidence-doc precision gap. No SHUD changes. No nested AI delegation occurred during review.

Findings:

- 🔵 Suggestion: evidence-doc SHALL row imprecisely worded as fixed value
  `docs/p8pre/pr_a_prep_evidence.md:60`
  Row "heihe_x4 forcing/ subdir" SHALL column says "286M (per map §2.2)" Observed 286M PASS. The actual brief gate per tasks.md §1.3 is `≥ 200M`; 286M is the current documented value, not an exact SHALL. As written, a future basin re-pack landing at 290M would falsely fail. Suggest rewording to "≥ 200M (286M documented in map §2.2)". Non-blocking — does not affect run readiness or PR-A handoff.

Non-blocking notes:

- Item 1 (markers): rendered awk substitution leaves zero residual `__MARKER__` tokens; only `__BUILD__` mention survives at `submit_n8_profile_template.sbatch:16` inside a comment narrating the diff vs the p1e prototype. Inactive.
- Item 2 (三铁律): `--output`/`--error` pinned `/scratch/.../.p8pre-runs/...` (L48-49), `cd "$SHUD_DIR"` at L62, `./shud __CASE__` at L93 from absolute `$SHUD_DIR=/scratch/.../SHUD`.
- Item 3 (Mode C flags): build intentionally one-shot outside template per L84-87 comment + `docs/p8pre/pr_a_prep_evidence.md:27-31` confirms `make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1` exit 0.
- Item 4 (determinism): L67-71 `OMP_PROC_BIND=close`, `OMP_PLACES=cores`, `OMP_NUM_THREADS=__N__`, `SHUD_RHS_THREADS=__N__` — exact P1e PR-I SOP.
- Item 5 (N=2 excluded): `render_n8_profile.sh:57` `NS=(1 4 8)`; full-file grep + render output show 0 N=2 references.
- Item 6 (node pin): L62-63 `NODE_OF[heihe]=cn14`, `NODE_OF[heihe_x4]=cn15`. Spot-check confirms heihe N=8 → cn14, heihe_x4 N=8 → cn15.
- Item 7 (18 cells): `render_18cell_sanity.txt:5-22` 18 sbatch lines; guard at `render_n8_profile.sh:142-145` enforces exit-code-2 on mismatch.
- Item 8 (singleton chain): L128-133 placeholder `__PREV_JID_${case}_N${n}_rep${rep}__`; rep1 standalone, rep2 deps rep1, rep3 deps rep2. Verified in render output (6 first-rep cells no `--dependency`, 12 dependent).
- Item 9 (nm gates): `cn14_build_evidence.log:45-52` Serial=1, OpenMP=0, libgomp GOMP_parallel=1. All three SHALL gates PASS.
- Item 10 (SHUD pin): `cn14_build_evidence.log:3` `7a1dc8f` exact, matches tasks §1.1.
- Item 11 (basin sizes): heihe forcing.trimmed 29M basin-local (NOT NFS); heihe_x4 basin 2.3G ≥ 200M; forcing/ 286M; tsd.forc 2nd line basin-local. "禁止重生" honored.
- Item 12 (evidence accuracy): SHUD pin/build/nm/sizes/cell-count all cross-match raw logs. Only gap is the Finding above.
- Item 13 (POSIX): no python/node/jq/yq/perl hits in either tool file.
- Item 14 (no SHUD change): diff stat shows 3 files, none under SHUD/; SHUD pointer unchanged.
- Handoff contract: `render_n8_profile.sh:77-82` distinguishes server-write vs Mac-dry-run; placeholder substitution by issue #341 runner documented at L88-91 + evidence doc L134-141.
