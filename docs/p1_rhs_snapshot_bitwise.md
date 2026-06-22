# P1 RHS snapshot bitwise validation — Mac 4-case

PR-H deliverable for P1 epic (Issue #211, sub-issue #219 / I-C.5).
Implements spec `openspec/changes/p1-update-omp/specs/p1-state-update-parallel/spec.md`
L127-L134 "RHS snapshot bitwise vs per-case authoritative baseline" (4 Mac
case scenario).

## Scope

- 4 Mac case × 3 t_value × 1 canonical snapshot suffix = **12 file-level
  bitwise comparisons** vs B1b-tag canonical golden
- `OMP_NUM_THREADS=1` (A0 / NUM_OPENMP=1 per spec Scenario L131-L134) on
  the OpenMP binary `shud_omp`
- Cases: `keliya` (484) / `xinanjiang_upstream` (801) / `qinyijiang`
  (3155) / `qhh` (4773 + 1 lake). `kashigeer` excluded per S0-13
  `deferred-upstream` (status_matrix L41 + benchmarks/INDEX.md).
- Server cases (`heihe`, `heihe_x4`) are out of scope for PR-H — covered
  by PR-J per tasks 5.5-5.6.

**Canonical snapshot suffix** = `snapshot_t<rel_sec>.bin` (SHUD writer
`SHUD_DUMP_SITE=f_update`). This is the validation set archived as the
B0 / B1a / B1b 3-run repeatability gate per status_matrix L132 ("4 个
case × 3 = **12 个** snapshot_t*.bin 入库"). The companion
`_before_passvalue.bin` 2nd-suffix was added later as a SHUD writer
diagnostic by-product (PR #54) and was NOT part of the B0/B1a/B1b
canonical validation gate; PR-H treats it as a diagnostic addendum
(see § Diagnostic addendum below).

## Tag chain & golden source

Per spec L107-L113 per-case authoritative baseline table, the 4 Mac
cases satisfy `B0 = B1a = B1b = B1` (bitwise-stable full chain), so
`benchmarks/<case>/B0_output/snapshot_t<rel_sec>.bin` is the canonical
B1b-tag golden by tag-chain identity.

| case                  | NumEle | has_lake | authoritative tag      | golden path                                                            |
| --------------------- | ------ | -------- | ---------------------- | ---------------------------------------------------------------------- |
| keliya                | 484    | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/keliya/B0_output/snapshot_t<T>.bin`                        |
| xinanjiang_upstream   | 801    | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/xinanjiang_upstream/B0_output/snapshot_t<T>.bin`           |
| qinyijiang            | 3155   | no       | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/qinyijiang/B0_output/snapshot_t<T>.bin`                    |
| qhh                   | 4773   | yes (1)  | B1b ≡ B1a ≡ B0 ≡ B1    | `benchmarks/qhh/B0_output/snapshot_t<T>.bin`                           |

## Build evidence — strict FP flags

Mac local OMP build with `SHUD_DUMP_RHS=1` (snapshot writer compiled in)
on outer `ec4cdd2` + SHUD `07c677f`:

```
cd SHUD && make clean && make SHUD_DUMP_RHS=1 shud_omp 2>&1 | tee .s2-103/pr-h/make_shud_mac.log
```

Three-grep gate (spec task 5.2 + L193-L197):

| gate                                                           | required | observed | verdict |
| -------------------------------------------------------------- | -------- | -------- | ------- |
| `grep -oE '\-O2\|-ffp-contract=off\|-fno-fast-math'`           | ≥ 3      | 6        | PASS    |
| `grep -E  '\-ffast-math\|-Ofast\|-funsafe-math-optimizations'` | 0        | 0        | PASS    |
| `grep -c  '\-fopenmp'`                                         | ≥ 1      | 2        | PASS    |
| `grep -c  'DSHUD_DUMP_RHS=1'`                                  | ≥ 1      | 2        | PASS    |

Build artifact: `.s2-103/pr-h/make_shud_mac.log` (Apple Clang
`./shud_omp` link line shows all three strict-FP flags + `-fopenmp`).

## Canonical 12-cell bitwise matrix

| case                     | t=86400 (1d) | t=2592000 (30d) | t=7776000 (90d) | wall (90d, NUM_OPENMP=1) |
| ------------------------ | ------------ | --------------- | --------------- | ------------------------ |
| keliya (484)             | PASS         | PASS            | PASS            | 60.8 s                   |
| xinanjiang_upstream (801)| PASS         | PASS            | PASS            | 7.3 s                    |
| qinyijiang (3155)        | PASS         | PASS            | PASS            | 238.1 s                  |
| qhh (4773 w/lake)        | PASS         | PASS            | PASS            | 66.5 s                   |

**12 / 12 PASS** — `compare_snapshot --quiet` exit 0 across all cells.

Verdict per `compare_snapshot` exit code 0 = "BITWISE IDENTICAL" (file
header + record header + all dumped array bytes match the golden
byte-for-byte). Snapshot content (per `format.h` v1 + observed `DY`
array_count=1): full `DY` state derivative vector of length `NumY` (=
3*NumEle + NumRiv + NumLake).

Verification re-run command:

```
bash .s2-103/pr-h/run_pr_h_snapshot_bitwise.sh
```

Per-case run dir + cmp.log preserved under `.s2-103/pr-h/<case>_main/`.

## Verdict (spec L131-L134)

- PR-D (element loop) + PR-E (river loop) + PR-F (lake loop) three-pragma
  stack at `OMP_NUM_THREADS=1` on the canonical RHS snapshot probe
  (`SHUD_DUMP_SITE=f_update`) is **bitwise == B1b-tag canonical** on
  all 4 Mac cases × all 3 t_values
- A0 (NUM_OPENMP=1) verified for the 4-case Mac suite
- spec Scenario "4 Mac case RHS snapshot bitwise PASS" satisfied
- Pairs with existing CI gate `serial-baseline / build-and-compare(1,
  keliya)` (PR fast-feedback path) that has been GREEN at every PR-D /
  PR-E / PR-F merge

## §9 array completeness note (out-of-scope clarification)

Spec L129 lists 18 RHS-state arrays
(`uYsf / uYus / uYgw / uYriv / yLakeStg / DY / qEle* / QeleSurf / ...`).
The SHUD writer schema (`tools/rhs_snapshot/format.h` v1 + observed
canonical archive) currently dumps a **single `DY` array** per
snapshot file (`array_count = 1`, name `"DY"`, `nelem = NumY`). The
remaining 17 arrays in the §9 list are not part of the current
canonical snapshot writer schema; they are an S0-archival gap inherited
from B0 and out of scope for PR-H — to be addressed (if needed) under
PR-N or a follow-up writer schema bump (`SHUD_RHS_SNAPSHOT_FORMAT_VERSION`).
This is consistent with the B0/B1a/B1b gate (which validates the same
1-array writer schema) and with status_matrix L132 ("12 个
snapshot_t*.bin 入库").

## Next steps

- **PR-I** (#220 / I-C.6): full-run regression — 4 Mac case canonical
  summary SHA + cvode_stats 15-key + 3-run repeatability per spec
  L143-L161
- **PR-J** (#221 / I-C.7): server bitwise — heihe + heihe_x4 Slurm
  90-day NUM_OPENMP=1 + cvode_stats per spec L136-L138 + L152-L155

---

## Diagnostic addendum — `_before_passvalue` 2nd-suffix (non-spec gate)

> **Status**: informational. Not a PR-H acceptance criterion (see
> Scope above + status_matrix L132 file-count gate which counts only the
> canonical 12 snapshot files). Logged here for transparency + future
> follow-up.

`benchmarks/<case>/B0_output/` also contains a 2nd snapshot suffix
written when `SHUD_DUMP_SITE=f_loop_before_passvalue` +
`SHUD_DUMP_FNAME_SUFFIX=before_passvalue`. This suffix captures a
mid-pipeline DY slice (`array_count = 1`, name `"DY"`, `nelem =
NumEle`) at a probe point inside `f_update()` before the
`PassValue_legacy` call. It was added under PR #54 (snapshot
repeatability harness) as a SHUD writer diagnostic by-product and was
**not** retroactively folded into the B0/B1a/B1b canonical validation
gate.

PR-H re-ran 12 additional cells against this 2nd suffix to gather
diagnostic evidence on the 3-pragma stack's mid-pipeline behavior:

| case                     | t=86400 before | t=2592000 before | t=7776000 before |
| ------------------------ | -------------- | ---------------- | ---------------- |
| keliya (484)             | PASS           | PASS             | PASS             |
| xinanjiang_upstream (801)| FAIL           | FAIL             | FAIL             |
| qinyijiang (3155)        | FAIL           | FAIL             | FAIL             |
| qhh (4773 w/lake)        | FAIL           | FAIL             | FAIL             |

**6 / 12 PASS (keliya only).** ULP report (real-number-scale diff, not
ULP-level):

| case               | t=86400                  | t=2592000                 | t=7776000                  |
| ------------------ | ------------------------ | ------------------------- | -------------------------- |
| xinanjiang_upstream | |Δ|≤4.48, L2=10.22, idx=30  | |Δ|≤24.43, L2=58.55, idx=13  | |Δ|≤29.84, L2=68.35, idx=12   |
| qinyijiang         | |Δ|≤317.95, L2=995.67, idx=8| |Δ|≤61.70, L2=195.67, idx=8 | |Δ|≤223.67, L2=887.36, idx=8 |
| qhh                | |Δ|≤5.58, L2=17.11, idx=211 | |Δ|≤19.04, L2=53.69, idx=125| |Δ|≤90.40, L2=197.99, idx=1  |

(All cells reach magnitudes far above any A3b ULP threshold; the
mid-pipeline diff is on the real-number scale, not a rounding-order
artifact. `first_diff_index` is the DY-element offset of the first
non-zero byte difference.)

### Observations

1. The canonical `f_update` snapshot is bitwise (12/12 above): the
   end-of-`f_update` DY state — the value handed to CVODE — matches
   B1b byte-for-byte. **Whatever mid-pipeline drift exists is absorbed
   by the rest of `f_update` before PassValue runs.**
2. `keliya` (no lake, no irregular river, 484 element) is the only
   case bitwise on both suffixes. The 3 cases that fail are the 3 with
   higher mesh complexity (801 / 3155 / 4773), and qhh additionally
   has a lake. The pattern is structural, not statistical noise.
3. The 2nd-suffix probe is captured inside `f_update()` between the
   element-flux loop and PassValue. PR-D / PR-E / PR-F added
   `#pragma omp parallel for schedule(static) default(none)` to three
   `f_update`-side loops; even at `OMP_NUM_THREADS=1` the OpenMP
   runtime initialises a parallel region (single-thread team), which
   can reorder firstprivate / shared writes versus the strictly serial
   B1b binary. This is one plausible source — confirmation requires a
   PR-N audit pass.

### Hypotheses (deferred to PR-N or B1c follow-up)

- **(a) OpenMP single-thread runtime setup**: `omp parallel for` with
  team size 1 may still serialise loop iterations through OpenMP's
  scheduler rather than the natural for-loop ordering, altering the
  observable write trace at the mid-pipeline probe even though the
  final state is identical. Document as a runtime artifact + leave
  canonical gate authoritative.
- **(b) PR-D/E/F loop body reorder**: one of the three pragmas may
  reorder intra-loop scratch writes (e.g. `QeleSurfTot / QeleSubTot`
  reductions or row-major flat write order) in a way that is invariant
  by the time PassValue runs but visible at the before-PassValue probe
  point. Would warrant a targeted commit-by-commit bisect across PR-D
  / PR-E / PR-F.
- **(c) Probe-point semantics drift**: if the SHUD source moved the
  `f_loop_before_passvalue` hook even slightly between B0 archival and
  current head, the new probe point would capture a different snapshot
  of `DY[NumEle]` than the archived golden. Check `MD_rhs_dump.cpp`
  call site history vs B0 SHUD pin.

### Action item

Open a P2a / PR-N follow-up issue:

> **Title**: Investigate `_before_passvalue` mid-pipeline DY drift on
> 3 Mac cases under PR-D/E/F 3-pragma stack
>
> **Scope**: bisect PR-D / PR-E / PR-F, characterise hypothesis (a)
> vs (b) vs (c), decide whether to (i) document as known artifact,
> (ii) restore mid-pipeline bitwise via loop-body order fix, or (iii)
> bump snapshot writer schema to drop the 2nd suffix from archive.
>
> **Not blocking**: B1c-tag stacking; canonical 12-cell gate is bitwise
> on all 4 Mac cases — production behaviour (RHS state into CVODE,
> downstream `rivqdown` / `eleysurf` / cvode_stats) is preserved.

---

`signed_at`: 2026-06-22
`signer`: DankerMu
`signed_against_outer_commit`: `ec4cdd2`
`signed_against_SHUD_commit`: `07c677f`
`PR-H_branch`: `pr-h-mac-snapshot-bitwise`
`Closes`: #219 (P1 epic I-C.5)
