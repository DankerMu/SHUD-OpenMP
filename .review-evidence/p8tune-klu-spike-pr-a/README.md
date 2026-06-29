# .review-evidence/p8tune-klu-spike-pr-a/

PR-A review-evidence directory for openspec change [`p8tune-klu-spike`](../../openspec/changes/p8tune-klu-spike/).

## Scope

This directory holds the **server-side 16-cell Slurm array sweep artifacts** produced by
[`tools/p8tune.D/spike_array.sbatch`](../../tools/p8tune.D/spike_array.sbatch) +
[`tools/p8tune.D/run_cell.sh`](../../tools/p8tune.D/run_cell.sh).

PR-A authors **only** the dispatcher + per-cell wrapper. The directory is **populated**
post-merge by the orchestrator after the server run completes, via:

```bash
# (server side) once Slurm array job finishes
rsync -avh -e "ssh -p 32099" \
    frd_muziyao@210.77.77.22:/scratch/frd_muziyao/SHUD-OpenMP/.p8tune.D-runs/<run-id>/cell-*.{out,err,log,J.bin,time} \
    .review-evidence/p8tune-klu-spike-pr-a/cell-results/
```

Per spec REQ-7 PR-A boundary L218, the expected per-cell flat layout is:

```
cell-results/
  cell-00.out  cell-00.err  cell-00.log  cell-00.J.bin  cell-00.time   # keliya     natural btf=1
  cell-01.out  cell-01.err  cell-01.log  cell-01.J.bin  cell-01.time   # keliya     amd     btf=0
  cell-02.out  cell-02.err  cell-02.log  cell-02.J.bin  cell-02.time   # keliya     amd     btf=1
  cell-03.out  cell-03.err  cell-03.log  cell-03.J.bin  cell-03.time   # keliya     colamd  btf=1
  cell-04..07                                                          # heihe      × 4 orderings (natural+1, amd-0, amd+1, colamd+1)
  cell-08..11                                                          # heihe_x4   × 4 orderings (same)
  cell-12..15                                                          # heihe_x16  × 4 orderings (same)
```

## Acceptance gate (PR-A merge prerequisite per tasks §2.9)

- All 16 cells either PASS or REPORTED-OOM (`KLU_OOM_DETECTED` diagnostic line in
  `cell-NN.log` per spec REQ-5 Scenario "OOM-as-data-point")
- No cell exits Slurm with non-zero status that lacks a `KLU_OOM_DETECTED` line
  (OOM-as-data-point is `exit 0`; real failures are `exit 1` and need root-cause
  debugging, not re-queue)
- Per-cell numeric J (`cell-NN.J.bin`) archived for PR-B aggregator consumption
- Per-cell `/usr/bin/time -v` output (`cell-NN.time`) archived for the RSS axis
  verdict computation in PR-B

## Out of PR-A scope

- Aggregator (`aggregate_klu_spike.sh`) and `aggregate_verdict.txt` belong to PR-B
- ADR-0005 authoring belongs to PR-B
- Master plan capstone edits + canonical spec archive belong to PR-C

## References

- Spec: [`openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md`](../../openspec/changes/p8tune-klu-spike/specs/klu-pattern-spike-verdict/spec.md) REQ-4, REQ-5, REQ-7
- Tasks: [`openspec/changes/p8tune-klu-spike/tasks.md`](../../openspec/changes/p8tune-klu-spike/tasks.md) §2.1 — §2.10
- Issue: https://github.com/DankerMu/SHUD-OpenMP/issues/381
- PR-0 evidence (Mac smoke + cn-RAM probe): [`../p8tune-klu-spike-pr-0/`](../p8tune-klu-spike-pr-0/)
