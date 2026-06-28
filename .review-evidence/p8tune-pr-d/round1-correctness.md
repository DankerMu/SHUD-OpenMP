Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 59cb239faeb014099f3befd0a5fa788a41ac5409

Summary: PR-D correctness sound across all 10 dimensions — per-cell isolation prevents race, env-var contract matches PR-C, decoder produces complete 60-cell coverage, summary.tsv anchors cleanly to PR-A floors. APPROVE.

Findings: None.

Non-blocking notes:
- N1: sbatch L9 hardcodes idle node list comment (cosmetic; matches --array=1-60%11 cap)
- N2: cell.meta hardcodes shud_enable_profile=1 / shud_enable_openmp_rhs=1 from precondition (binary build provenance via SHUD_pin commit)
- N3: cfg.para 90-day truncation is deployment-layer (source Basins/<case>/input/) not per-cell
- N4: heihe_x4 N=8 wall +2-5% vs N=1 walls within {10..30} maxl band; PR-E G6 territory

Per-check assessment:
1. Per-cell arg validation:                   pass
2. Per-cell working dir isolation:            pass
3. Env-var setup correctness:                 pass
4. Wall capture correctness:                  pass
5. 5-artifact copy correctness:               pass
6. Provenance log verification:               pass
7. Exit code semantics:                       pass
8. Sbatch task_id decoder:                    pass
9. Slurm 三铁律:                              pass
10. Summary.tsv data sanity:                  pass

Verdict: APPROVE — all 10/10 PASS; 4 non-blocking informational notes.
