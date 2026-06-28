Reviewer agent: review-spec-compliance
Review round: round 1
Reviewed head SHA: d8602d0fb4e609c106eaa5f79e973830182ac150
Summary: All 6 spec scenarios PASS against rsync mirror; one brief-vs-spec inconsistency (Scenario 5 nFCall) is REFUTED because spec text contains no such requirement.

Findings:
- None.

Non-blocking notes:

- Scenario 1 (18-cell COMPLETED w/ exit 0): PASS. `verification.txt:33-35` reports `all_pass=1 block_count=0` over 18 cells. JID range 9510-9527 matches `jid_table.txt:1-18` (2 cases × 3 N × 3 reps). RC0 column OK for all 18 (`verification.txt:13-30`). Cross-checked `slurm.out` provenance lines on `heihe_N1_rep1` show `host=cn14`, `SHUD pin=7a1dc8f`, OMP env exported per template.

- Scenario 2 (5 artifacts per cell): PASS, with brief typo note. Spot-checked `heihe_N1_rep1 / heihe_N4_rep2 / heihe_x4_N1_rep1 / heihe_x4_N8_rep3`: each contains exactly `cvode_stats.txt`, `<case>.rivqdown.dat`, `profile_B0.yaml`, `slurm.err`, `slurm.out`. Actual schema matches spec §line 35 `<scratch>/.../cvode_stats.txt` and doc §4 (`docs/p8pre/n8_profile_run.md:72-73`). **Brief typo**: brief listed `<case>.cvode_stats.txt`, but spec + actual files use unprefixed `cvode_stats.txt`. Implementation correctly follows spec.

- Scenario 3 (bucket breakdown + bsum invariant): PASS. Sampled 5 cells (`heihe_N1_rep1 / heihe_N8_rep1 / heihe_x4_N1_rep1 / heihe_x4_N4_rep2 / heihe_x4_N8_rep3`): all 7 buckets present under `buckets:`, `extras:` contains `t_CVODE_raw + t_wall_total`. Bsum (excluding `t_RHS_kernel`) - `t_wall_total` = exactly 0.0000% across all sampled cells, matching `verification.txt:13-30` worst error report. The 0% identity follows from `timer.cpp:117-127` derivation algebra and the spec line 56 ±2% allowance is satisfied by a wide margin.

- Scenario 4 (cross-N invariance + absolute anchors): PASS. Verified all 5 counters (`nst/nfe/nfeLS/nni/nli`) bitwise identical across N=1/4/8 per case via grep on rsync mirror. Anchor values match exactly: **heihe nst=6698, nfe=6943** and **heihe_x4 nst=6575, nfe=6741** (`p1e_perf_baseline.md §3.4` + `p1e_pr_i_strict_omp_verification.md §3.1` per brief). Commit message claim and `n8_profile_run.md:107-112` Table §5 anchor values reproduce on actual artifacts.

- Scenario 5 (nFCall semantics): REFUTED. Brief asserts spec requires a `nFCall` counter shipped via separate `nfcall.txt`, but the full text of `openspec/changes/p8pre-spike/specs/n8-mode-c-profile-recheck/spec.md` (read in full) contains **zero** references to `nFCall` or `nfcall`. Search across `proposal.md / design.md / tasks.md / spec.md` returns no matches. The canonical 15-key set defined at spec line 31 (`nfe/nfeLS/nni/nli/nsetups/netf/nst/npe/nps/ncfn/ncfl/lenrw/leniw/lenrwLS/leniwLS`) does not include `nFCall`. Per spec line 38, additional keys like `hcur/qcur/hin` are explicitly OUT-of-scope unless `SHUD_ENABLE_DIAGNOSTICS=1`. No `nfcall.txt` file exists anywhere in the rsync mirror. Implementation correctly follows the actual spec; the brief's Scenario 5 was likely copy-pasted from a draft revision.

- Scenario 6 (Slurm 三铁律): PASS. `submit_n8_profile_template.sbatch:48-49` pins `#SBATCH --output/--error` to `/scratch/frd_muziyao/SHUD-OpenMP/.p8pre-runs/.../slurm.{out,err}`. `run_n8_profile.sh:61` enforces `cd $REPO` (= `/scratch/...`) before `sbatch`. Rsync was performed POST-completion per `n8_profile_run.md:124` ("rsync POST 所有 Slurm jobs COMPLETED"); cross-checked rsync_log timestamp 22:50 vs last sacct completion ~02:43:35Z (2026-06-27).
