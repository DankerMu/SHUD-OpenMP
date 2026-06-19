# P7 A3c (cross-thread full-run bitwise) status

**Status**: SCAFFOLD pre-#100 server execution. Final result will be filled
in by `#100 task 8.10` byproduct + cross-referenced from the P7 PR.

**Owner**: openspec change `s2-strict-omp-full` (task 9.6 #101 scaffold).
**Decision authority**: master plan `SHUD_openMP_master_plan.md` §2.2 L155
  + spec `openspec/changes/s2-strict-omp-full/specs/strict-omp-acceptance-gates/spec.md`
  Scenario "A3c is optional, not blocking".

## Context

A3c is the most stringent precision tier under master plan §2.2 A0-A5:

| Tier | Scope | Threshold | P7 gate |
|---|---|---|---|
| A0   | 3-run repeatability same thread count | bitwise | enforced (all stages) |
| A2   | RHS snapshot vs baseline | bitwise | enforced per sub-stage |
| A3a  | full-run same thread count vs baseline | bitwise | **enforced P7 exit** |
| A3b  | full-run cross-thread count | max_ulp(DY) <= 4, max_abs_diff < 1e-12 | **enforced P7 exit** |
| A3c  | full-run cross-thread count | **bitwise** | **OPTIONAL / not blocking** |

Master plan §2.2 L155 explicitly downgrades A3c to "加分项, 不阻塞 P7 退出":

> A3c (跨线程数完整 run bitwise) 是**加分项**，不阻塞 P7 退出。

Spec `strict-omp-acceptance-gates` Requirement 4 Scenario "A3c is optional,
not blocking" enforces the same boundary:

> WHEN P7 退出 sign-off 时 A3c (跨线程数完整 run bitwise) 未达到
> THEN P7 退出 SHALL 不被阻塞，但 SHALL 在 `docs/P7_A3c_status.md` 解释为什么没达到 (plan §6 P7.5 L1921)
> THEN A3c 若达到 SHALL 记入 P7 交付物，作为加分项

This doc is the deliverable for the "未达到 -> 解释" branch.

## Why A3c is typically unreachable

OpenMP execution policy (`StrictOMP`) introduces non-deterministic reduction
ordering for any accumulation that is not strictly per-owner-local. SHUD's
P1 + P5 + P6 owner-local refactors eliminate gather races for the element /
river / lake / DY scopes, **but the CVODE integrator step (run-level) still
produces summation-order-dependent state when the per-step DY vector has
been computed under different thread counts**:

- DY[i] @ thread-count=4 is bitwise identical to DY[i] @ thread-count=4
  across runs (A3a / A0 enforced).
- DY[i] @ thread-count=4 vs DY[i] @ thread-count=8 differs by O(ULP) due
  to summation grouping inside OpenMP reduction clauses (verified by A3b
  upper bound, max_ulp <= 4).
- ULP differences accumulate over the full CVODE run (~ O(nst) steps), so
  the final `.dat` outputs are NOT bitwise identical across thread counts
  even though the per-step ULP gap is tiny. This is consistent with the
  expected outcome described in spec Requirement 4 + master plan §2.2 L155.

Therefore the **expected** A3c result post-P7 is:

- A3c thread-1 vs thread-2: typically **MISS** (ULP > 0 accumulated)
- A3c thread-1 vs thread-4: typically **MISS**
- A3c thread-1 vs thread-8: typically **MISS**
- A3c thread-2 vs thread-4: typically **MISS**
- A3c thread-2 vs thread-8: typically **MISS**
- A3c thread-4 vs thread-8: typically **MISS**

If any pair MEETS A3c (true bitwise) post-P7, it is recorded as a bonus
in the P7 PR + here (move row to "Achieved" section below).

## Measurement plan

Filled in by #100 task 8.10 (A3b cross-thread sweep). For each case in
{qinyijiang, heihe_x4} (the two A3b-mandatory cases per design §D13),
write the SHA256 of the final `.dat` outputs at each thread-count
{1, 2, 4, 8}; A3c PASS iff all 4 SHA256s are equal.

| Case | Thread-1 SHA | Thread-2 SHA | Thread-4 SHA | Thread-8 SHA | A3c verdict |
|---|---|---|---|---|---|
| qinyijiang | TBD | TBD | TBD | TBD | TBD |
| heihe_x4 | TBD | TBD | TBD | TBD | TBD |

## Conclusion (filled post-#100 execution)

> _Placeholder: filled by P7 PR follow-up._
>
> Conditional template:
>   - If any case PASSES A3c (all 4 SHA equal), append to "Achieved A3c
>     cases" + cite as bonus in P7 PR body.
>   - If all cases MISS A3c (typical expected outcome), record the per-
>     pair ULP delta + nst delta from #100 task 8.10 here. P7 exit
>     proceeds unimpacted; A3a + A3b gates remain enforced.

## Achieved A3c cases (bonus)

| Case | Evidence | PR |
|---|---|---|
| _(none yet — fill if any case meets cross-thread bitwise post-P7)_ | — | — |

## References

- master plan §2.2 A0-A5 hierarchy (L120-L160 region) + L155 A3c downgrade
- master plan §6 P7.5 L1921 (A3c未达 -> 写 docs/P7_A3c_status.md)
- spec `strict-omp-acceptance-gates` Requirement 4 Scenario "A3c is optional"
- #100 task 8.10 (A3b cross-thread sweep, A3c by-product)
- #101 task 9.6 (this doc, scaffold)
