---
title: "p8pre-spike Step 2 PR-D pre-flight — SUNDIALS API verification + SHUD fork evidence"
date: 2026-06-27
version: 0.1
status: "SUNDIALS 6.0.0 preconditioner API alive (4 grep PASS); SHUD openmp-baseline-p8pre forked from 7a1dc8f (P1e ship pin) with forward-only descendant guarantee"
related_docs:
  - "openspec/changes/p8pre-spike/tasks.md §5 (PR-D pre-flight)"
  - "openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md Scenario 'Step 2 forward-only descendant extension' (L148-154)"
  - "openspec/changes/p8pre-spike/design.md D6 (API choice)"
  - "CLAUDE.md SHUD submodule workflow (C8 forward-compat)"
  - ".review-evidence/p8pre-pr-d-preflight/api_grep.txt (4 grep raw)"
  - ".review-evidence/p8pre-pr-d-preflight/fork_evidence.txt (fork raw)"
---

# p8pre-spike Step 2 PR-D pre-flight — evidence

## §1 目的

PR-D pre-flight slice: verify SUNDIALS 6.0.0 preconditioner API exists in the
locally-installed `SHUD/InstallSundials/include/cvode/{cvode.h, cvode_ls.h}`
headers **before** PR-D impl (#345) writes `MD_precond_identity.{h,cpp}` or
wires `cvode_config.cpp` to `PREC_LEFT`. Also: fork the SHUD submodule working
branch `openmp-baseline-p8pre` from the mandatory `7a1dc8f` P1e ship pin so
that the PR-D impl pointer bump (Step 2) is provably a linear descendant of
the Step 1 P1e ship state. Out-of-scope for this slice: any impl, any outer
submodule pointer bump, any PR open.

## §2 API verification (4 grep PASS)

All 4 SHALL symbols expected by design.md D6 are present in the installed
SUNDIALS 6.0.0 headers at SHUD pin `7a1dc8f`:

| # | Symbol                       | Header     | Line | Verdict |
|---|------------------------------|------------|------|---------|
| 1 | `CVodeSetLSetupFrequency`    | `cvode.h`  | 132  | PASS    |
| 2 | `CVodeSetJacEvalFrequency`   | `cvode_ls.h` | 91 | PASS    |
| 3 | `CVLsPrecSetupFn` (typedef)  | `cvode_ls.h` | 57 (typedef), 99 (CVodeSetPreconditioner arg) | PASS |
| 4 | `CVLsPrecSolveFn` (typedef)  | `cvode_ls.h` | 61 (typedef), 100 (CVodeSetPreconditioner arg) | PASS |

Raw: `.review-evidence/p8pre-pr-d-preflight/api_grep.txt`.

Notes:

- `CVLsPrecSetupFn` and `CVLsPrecSolveFn` each appear twice: once as the
  function-pointer typedef (`typedef int (*CVLsPrecSetupFn)(...)` at L57 and
  `typedef int (*CVLsPrecSolveFn)(...)` at L61), and once as the argument
  type in `CVodeSetPreconditioner(void *cvode_mem, CVLsPrecSetupFn pset,
  CVLsPrecSolveFn psolve)` at L98-100. Both citation sites are valid; impl
  side will use the typedefs to declare `MD_precond_identity_setup` and
  `MD_precond_identity_solve` signatures and pass them to
  `CVodeSetPreconditioner`.
- `CVodeSetLSetupFrequency(void *cvode_mem, long int msbp)` is the
  preconditioner-setup-frequency knob the design relies on (msbp = max steps
  between preconditioner setups; identity preconditioner allows msbp very
  high but PR-D will use a conservative default).

## §3 REJECT: `CVodeSetMaxConvFails` is NOT a substitute

`CVodeSetMaxConvFails` exists in `cvode.h` at L133 (adjacent to
`CVodeSetLSetupFrequency` at L132 — `grep` confirms both as separate
`SUNDIALS_EXPORT int` declarations). They are NOT interchangeable:

| Function                     | Signature                                       | Subsystem | Purpose |
|------------------------------|-------------------------------------------------|-----------|---------|
| `CVodeSetLSetupFrequency`    | `(void *cvode_mem, long int msbp)`              | CVODE linear-solver setup loop | msbp = max steps between LSetup (and hence preconditioner-setup) calls |
| `CVodeSetMaxConvFails`       | `(void *cvode_mem, int maxncf)`                 | CVODE nonlinear-iteration convergence | maxncf = max nonlinear-solver convergence failures per step before step-size reduction |

`msbp` (long int, step count) controls **how often** the linear-solver setup
runs (which triggers the preconditioner setup callback). `maxncf` (int,
failure count) controls **how many** nonlinear conv failures CVODE tolerates
before backing off the step size. PR-D design D6 picks
`CVodeSetLSetupFrequency` precisely because we want msbp control. Reviewers
who suggest swapping to `CVodeSetMaxConvFails` are confusing two distinct
SUNDIALS subsystems — the function exists but solves a different problem.

## §4 SHUD branch fork evidence

Per spec p8precond-zero-identity-spike Scenario "Step 2 forward-only
descendant extension" L148-154 + CLAUDE.md SHUD submodule workflow C8.

Pre-fork SHUD HEAD (working tree, before branch creation):

```
$ cd SHUD && git rev-parse HEAD
7a1dc8f6ea9e5496f516255406ee3563d397959b
```

Matches expected P1e ship pin `7a1dc8f` (commit message:
`fix(profile): remove inner-function Timers that nested with shud.cpp call-sites`).

Origin branch state before fork (empty — branch does not yet exist):

```
$ git ls-remote --heads origin openmp-baseline-p8pre
(no output)
```

Branch creation + push:

```
$ git checkout -b openmp-baseline-p8pre 7a1dc8f
Switched to a new branch 'openmp-baseline-p8pre'
$ git push -u origin openmp-baseline-p8pre
 * [new branch]      openmp-baseline-p8pre -> openmp-baseline-p8pre
branch 'openmp-baseline-p8pre' set up to track 'origin/openmp-baseline-p8pre'.
```

Post-push origin state confirms branch exists at expected SHA:

```
$ git ls-remote --heads origin openmp-baseline-p8pre
7a1dc8f6ea9e5496f516255406ee3563d397959b	refs/heads/openmp-baseline-p8pre
```

Raw: `.review-evidence/p8pre-pr-d-preflight/fork_evidence.txt`.

## §5 Forward-only descendant guarantee

Spec L151 requires that the new SHUD SHA (post-PR-D-impl bump) be a LINEAR
DESCENDANT of `7a1dc8f` such that
`git -C SHUD merge-base <new SHA> 7a1dc8f` returns exactly `7a1dc8f`. At the
pre-flight checkpoint (no commits on `openmp-baseline-p8pre` beyond the fork
point), the forward-only criterion is provable in its strictest form
(branch HEAD == fork base):

```
rev-parse openmp-baseline-p8pre: 7a1dc8f6ea9e5496f516255406ee3563d397959b
rev-parse 7a1dc8f:               7a1dc8f6ea9e5496f516255406ee3563d397959b
merge-base openmp-baseline-p8pre 7a1dc8f: 7a1dc8f6ea9e5496f516255406ee3563d397959b

PASS: rev-parse openmp-baseline-p8pre == rev-parse 7a1dc8f
PASS: merge-base == rev-parse 7a1dc8f
```

Both PASS. After PR-D impl (#345) lands its identity-preconditioner commit
on `openmp-baseline-p8pre`, the second check
(`merge-base == rev-parse 7a1dc8f`) MUST continue to hold while the first
check (`rev-parse openmp-baseline-p8pre == rev-parse 7a1dc8f`) will
naturally fail — that is the expected linear-descendant signature per spec
L151 + L153 ("a single linear child of 7a1dc8f").

Why `7a1dc8f` exactly: it is the P1e ship pin (D11-locked
`baseline/P1e` HEAD; see `docs/p8pre/step1_prep.md` for the P1e PR-I
baseline anchor that this epic measures against). Forking from any other
SHA would either (a) re-introduce divergence from P1e ship state, breaking
the "P1e SHIP preservation" SHALL at spec L140, or (b) include unrelated
upstream churn whose semantic effect on RHS/preconditioner numerics is not
covered by P1e AC-S1/S2/S3 evidence.

`.gitmodules` is untouched in this slice (spec L154 — the `branch =
openmp-baseline` field stays as-is; working-branch shift to
`openmp-baseline-p8pre` is reflected by the outer pointer SHA bump in
PR-D impl, not by editing `.gitmodules`).

## §6 引用

- `openspec/changes/p8pre-spike/tasks.md` §5 (PR-D pre-flight task list)
- `openspec/changes/p8pre-spike/specs/p8precond-zero-identity-spike/spec.md`
  Scenario "Step 2 forward-only descendant extension" L148-154
- `openspec/changes/p8pre-spike/design.md` D6 (preconditioner API choice
  rationale)
- `CLAUDE.md` — SHUD submodule workflow (rule C8 forward-compat: all OMP
  work commits land on long-lived non-master branches, in this epic the
  `openmp-baseline-p8pre` branch forked from `openmp-baseline` at `7a1dc8f`)
- `docs/p8pre/step1_prep.md` (P1e PR-I baseline anchor — the soft-gate-5
  strict-bitwise expected value PR-D impl will compare against)
- Raw evidence:
  - `.review-evidence/p8pre-pr-d-preflight/api_grep.txt`
  - `.review-evidence/p8pre-pr-d-preflight/fork_evidence.txt`
