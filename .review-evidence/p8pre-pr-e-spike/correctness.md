Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 2eb5d0f
Summary: PR-E spike tooling + evidence doc reconciles cleanly against Step 1 PR-A reference, fixture spec, and 18-cell raw data; identity 3-symbol nm gate-1 captured; orchestrator zone respected; APPROVE.

Findings:
- None.

Non-blocking notes:

1. Outer diff scope (checklist #9, #10) — `git diff baseline/p8pre...2eb5d0f --name-only` returns exactly the 4 expected files (3 tool scripts + 1 doc). No SHUD pointer bump (PR-D #357 already pinned `5276167`; `git submodule status SHUD` confirms `5276167eea67...`). No `openspec/changes/` modification — orchestrator zone respected. Forward-only descendant rule (spec L148-154) preserved.

2. 3 tool-script forks (checklist #1) — diff against Step 1 PR-A counterparts shows ONLY the expected substitutions:
   - `run_identity_spike.sh`: RUNS_DIR → `.p8pre-runs/identity_spike`, RENDER_WRAPPER → `render_identity_spike.sh`, log-tag `[run_n8_profile]→[run_identity_spike]` everywhere, **+ NEW Phase B.2 identity 3-symbol nm gate-1 block at L107-131** (verify_identity_symbols function checks PSetupIdentity ≥1 / PSolveIdentity ≥1 / CVodeSetPreconditioner ≥1, writes `server_nm.log`, exits 2 on FAIL).
   - `render_identity_spike.sh`: TEMPLATE + OUT_DIR + singleton_name prefix `p8pre_identity_` updated; log tag updated.
   - `submit_identity_spike_template.sbatch`: all `.p8pre-runs/` → `.p8pre-runs/identity_spike/`, job-name `p8pre_identity___CASE___N__N___rep__REP__`, log tag `[p8pre_identity_spike]`. SHUD binary invocation + OMP env + artifact harvest (profile_B0.yaml + cvode_stats.txt + .rivqdown.dat) unchanged.
   - `bash -n` syntax PASS on both `.sh` files.

3. Mac dry-run (checklist #2) — `bash render_identity_spike.sh` emits **18 sbatch lines + 1 trailing render summary = 19 stdout total**. Job names confirmed `p8pre_identity_<case>_N<n>_singleton` per spec; dependency chains rep1→rep2→rep3 use `__PREV_JID_<case>_N<n>_rep<r-1>__` placeholders correctly; paths all point under `.p8pre-runs/identity_spike/rendered/`.

4. Server build + nm evidence (checklist #3) — `server_nm.log` shows `T PSetupIdentity` + `T PSolveIdentity` + `U CVodeSetPreconditioner` (3 symbols, all required counts ≥1 satisfied per gate-1 spec L70-72). `server_build_provenance.log` documents gcc 13.3.0 (Ubuntu 24.04) + libsundials_cvode.so.6 + libgomp.so.1. Doc §2-3 match these files verbatim.

5. 18-cell data (checklist #4) — Each of 18 doc §4 rows reconciles exactly with `cell_stats.txt` (case/N/rep/nst/nfe/ncfn/nps/npe/wall_total/t_precond_setup) and JID column matches `jid_table.txt` (JID 9531..9548 monotonic, contiguous). ExitCode = 0:0 attested by doc §4 narrative ("All 18 jobs ExitCode 0").

6. Cross-N invariance (checklist #5) — Verified from raw data: heihe (nst=6599, nfe=6696, ncfn=6, nps=18163, npe=77) identical across N=1/4/8 × 3 reps; heihe_x4 (nst=6569, nfe=6775, ncfn=47, nps=37695, npe=158) identical across all 9 cells. Doc §6 claim correct.

7. ncfn observation (checklist #6) — heihe ncfn=6 / heihe_x4 ncfn=47 confirmed deterministic across 9 cells per case. Doc §7 cites spec L74-79 strict criterion (`ncfn = 0`), frames as PR-F gate-2 adjudication scope, and **does NOT pre-empt verdict** (uses "PR-F #347 adjudication", no direct FAIL call). Wording correctly neutral.

8. Soft gate 6 (checklist #7) — Per-cell `t_precond_setup / t_wall_total` ratios independently verified: heihe range 6.47e-08 .. 1.01e-07 (matches doc §8 6.472e-08 .. 1.009e-07); heihe_x4 range 9.60e-09 .. 1.94e-08 (matches doc 9.599e-09 .. 1.944e-08). All 18 cells ~6+ orders of magnitude below 5e-2 budget — claim accurate.

9. Wall comparison §5 (checklist #8) — Recomputed identity median per (case,N) from §4 raw + recomputed delta vs `n8_profile_baseline.md` Table 1 anchors (140.797 / 95.734 / 89.732 / 1412.895 / 849.704 / 743.552). All 6 rows reconcile exactly (heihe N=1 -2.64%, N=4 -1.97%, N=8 -1.89%; heihe_x4 N=1 +1.09%, N=4 +1.01%, N=8 +0.64%). Max |delta_pct| heihe = 2.64% << ε=10%; max heihe_x4 = 1.09% << ε=5% — well within budget. Doc §5 correctly frames as "observational only" + defers formal verdict to PR-F.

10. Suggested doc polish (zero-impact, non-blocking): §4 narrative claim "All 18 jobs ExitCode 0" is asserted in prose but not backed by a column in the table or a side-by-side sacct grep snippet in evidence. Captured `cell_stats.txt` carries CVODE counters but no ExitCode column; future PR-F aggregator should make this explicit.

Verdict: APPROVE — all 10 checklist items pass; tool forks are minimal-diff and correctly scoped; identity gate-1 evidence is captured per spec; 18-cell raw data reconciles bit-exactly with doc tables; verdict-doc boundary (data-only, PR-F owns adjudication) respected.
