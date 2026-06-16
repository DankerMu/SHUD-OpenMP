# Case deployment fixup report — Issue #4 (S0.1b)

- **Round 1 timestamp**: 2026-06-16T19:23:27Z
- **Round 2 timestamp**: 2026-06-16T19:44:06Z (this update; includes A1–A4 / B1–B3 fixes)
- **Host OS**: Darwin 24.6.0 arm64 (Apple Silicon Mac, local development)
- **Branch**: `feat/issue-4-case-deployment-fixup`
- **Tool**: `tools/fix_case_paths/fix_case_paths.sh`
- **Spec**: `openspec/changes/s0-baseline-lock/specs/case-deployment-fixup/spec.md`
- **OpenSpec strict validation**: PASS (`openspec validate s0-baseline-lock --strict --no-interactive` → "Change 's0-baseline-lock' is valid")

The fixup tool rewrites the server-deployment path on line 2 of each
`<project>.tsd.forc` to the local absolute path of that case's
`forcing/` directory, and forces `NUM_OPENMP\t1` in each
`<project>.cfg.para` for the S0.1b serial baseline. The original files
are snapshotted to `*.orig` on first run and never overwritten on
subsequent runs.

## Layout discovery

Each case directory uses an inconsistent project subdirectory naming:

| Case (= `SHUD/Basins/<case>/`) | Project subdir (`input/<project>/`) | Kind |
| --- | --- | --- |
| `keliya` | `keliya` | benchmark |
| `kashigeer` | `ksge` | benchmark |
| `qhh` | `qhh` | benchmark |
| `qinyijiang` | `nanlin` | benchmark |
| `tailanhe` | `tlh` | auxiliary |
| `xinanjiang_upstream` | `xinanjiang` | benchmark |

The tool discovers `<project>` by globbing `SHUD/Basins/<case>/input/*/*.tsd.forc`
(unique match required).

## Per-case results

All 6 local cases passed `--all` apply (`failures=0`, exit 0).

### keliya (benchmark) — PASS

- File: `SHUD/Basins/keliya/input/keliya/keliya.tsd.forc`
  - `before line 2`: `/scratch/st_liyunhan/liyunhan/keliya_niya_small/deploy/forcing`
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/keliya/forcing`
- File: `SHUD/Basins/keliya/input/keliya/keliya.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8, tab-separated)
- Other lines: byte-preserved (only line 2 of tsd.forc + line 8 of cfg.para changed)

### kashigeer (benchmark) — PASS

- File: `SHUD/Basins/kashigeer/input/ksge/ksge.tsd.forc`
  - `before line 2`: `/scratch/st_liwz/ksge/forcing_cleaned_rn1360`
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/kashigeer/forcing`
- File: `SHUD/Basins/kashigeer/input/ksge/ksge.cfg.para`
  - `NUM_OPENMP`: already `1` upstream → `1` (no-op, line 8); cfg.para file is bit-identical to `.orig`
- Other lines: byte-preserved

### qhh (benchmark) — PASS

- File: `SHUD/Basins/qhh/input/qhh/qhh.tsd.forc`
  - `before line 2`: `/scratch/st_zhanghx/qhh/qhh_zenodo/Modeling/forcing`
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/qhh/forcing`
- File: `SHUD/Basins/qhh/input/qhh/qhh.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8)
- Other lines: byte-preserved

### qinyijiang (benchmark) — PASS

- File: `SHUD/Basins/qinyijiang/input/nanlin/nanlin.tsd.forc`
  - `before line 2`: `/scratch/wangjj/nanlin/deploy/forcing`
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/qinyijiang/forcing`
- File: `SHUD/Basins/qinyijiang/input/nanlin/nanlin.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8)
- Other lines: byte-preserved

### tailanhe (auxiliary) — PASS

- File: `SHUD/Basins/tailanhe/input/tlh/tlh.tsd.forc`
  - `before line 2`: `/scratch/wenxin/tlh_0/deployforcing4`
  - `after  line 2` (round 2, Fix A4): `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/tailanhe/focing`
- File: `SHUD/Basins/tailanhe/input/tlh/tlh.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8)
- Other lines: byte-preserved
- **Round 2 update (Fix A4)**: tailanhe ships with `SHUD/Basins/tailanhe/focing/`
  (typo, no `r`) and no `forcing/` directory. The round-1 tool wrote the
  canonical `forcing/` path and WARNed; this caused downstream `./shud
  tailanhe` to fail at runtime because line 2 didn't resolve. The round-2
  tool detects the upstream typo and writes the actually-existing path
  (`focing/`) with a loud WARN; if upstream is later corrected to
  `forcing/`, the tool will auto-pick that path. See §"Known runtime
  caveats" below.

### xinanjiang_upstream (benchmark) — PASS

- File: `SHUD/Basins/xinanjiang_upstream/input/xinanjiang/xinanjiang.tsd.forc`
  - `before line 2`: `/scratch/wangjj/xinanjiang/deploy/forcing`
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/xinanjiang_upstream/forcing`
- File: `SHUD/Basins/xinanjiang_upstream/input/xinanjiang/xinanjiang.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8)
- Other lines: byte-preserved

## Idempotency verification

Re-running `./tools/fix_case_paths/fix_case_paths.sh --all` over the
already-fixed-up state produced exit code 0 and **bit-identical**
working files (SHA256 unchanged). The `.orig` snapshots were also
preserved (never overwritten).

Method: captured `shasum -a 256` of all 12 working files + 12 `.orig`
files, ran `--all` again, re-captured, diffed — `diff` was empty for
both sets. The tool prints `.orig exists: ... (preserved)` on each
re-run to make this audit-visible.

## Restore reversibility verification

`./tools/fix_case_paths/fix_case_paths.sh --all --restore` returned exit 0
and printed `RESTORED (benchmark)` / `RESTORED (auxiliary)` per case.

After restore, the working files matched the upstream pre-apply SHA256
**bit-exact**. Verified by diffing `shasum -a 256` of all 12 files
post-restore against the pre-apply snapshot captured before the first
`--all`:

| File | Pre-apply SHA256 | Post-restore SHA256 |
| --- | --- | --- |
| `SHUD/Basins/keliya/input/keliya/keliya.tsd.forc`               | `72ef4012…d1e21ea` | `72ef4012…d1e21ea` |
| `SHUD/Basins/keliya/input/keliya/keliya.cfg.para`               | `45d5d4f3…ba9dca` | `45d5d4f3…ba9dca` |
| `SHUD/Basins/kashigeer/input/ksge/ksge.tsd.forc`                | `6c2f3f9b…1dd5f5` | `6c2f3f9b…1dd5f5` |
| `SHUD/Basins/kashigeer/input/ksge/ksge.cfg.para`                | `5ac1f237…f6bb3e` | `5ac1f237…f6bb3e` |
| `SHUD/Basins/qhh/input/qhh/qhh.tsd.forc`                        | `0cd156ff…f499f26` | `0cd156ff…f499f26` |
| `SHUD/Basins/qhh/input/qhh/qhh.cfg.para`                        | `a9b237db…87fdbe` | `a9b237db…87fdbe` |
| `SHUD/Basins/qinyijiang/input/nanlin/nanlin.tsd.forc`           | `ec67e33a…0653c54` | `ec67e33a…0653c54` |
| `SHUD/Basins/qinyijiang/input/nanlin/nanlin.cfg.para`           | `2accca95…ae4843d` | `2accca95…ae4843d` |
| `SHUD/Basins/tailanhe/input/tlh/tlh.tsd.forc`                   | `e7b557d4…171fdff` | `e7b557d4…171fdff` |
| `SHUD/Basins/tailanhe/input/tlh/tlh.cfg.para`                   | `a130dd61…39dd198` | `a130dd61…39dd198` |
| `SHUD/Basins/xinanjiang_upstream/input/xinanjiang/xinanjiang.tsd.forc` | `aa266d83…978606aed` | `aa266d83…978606aed` |
| `SHUD/Basins/xinanjiang_upstream/input/xinanjiang/xinanjiang.cfg.para` | `42441f4e…b92658e` | `42441f4e…b92658e` |

(Full SHA256 captured in tool execution logs; truncated here for
readability — every pair compared equal.)

`.orig` files were removed on restore. After this verification,
`--all` was re-applied to leave data in the fixed-up state for
downstream issues to consume.

## Failure-mode verification (spec §"Single case failure does not abort batch")

Reproduced by hiding `SHUD/Basins/keliya/input/keliya/keliya.tsd.forc`
and running `--all`:

```
PASS (benchmark): kashigeer
FAIL (benchmark): keliya
PASS (benchmark): qhh
PASS (benchmark): qinyijiang
PASS (auxiliary): tailanhe
PASS (benchmark): xinanjiang_upstream

Bulk apply complete. failures=1
```

Script exit code: `1` (non-zero). Already-processed cases retained
their fixup + `.orig` (consistent state, per spec §"Mid-batch failure
leaves consistent state").

## Final state

After this fixup, all 6 local NWM/auxiliary cases are configured for
local S0.1b runs:

- `<project>.tsd.forc` line 2 → local absolute path to `<case>/forcing/`
- `<project>.cfg.para` `NUM_OPENMP` line → `1`
- `.orig` backups present (bit-identical to upstream NWM deployment)

`heihe` is intentionally not touched (server-only, 12 G forcing).

## PR boundary

Committed under this PR:

- `tools/fix_case_paths/fix_case_paths.sh`
- `fixup_report.md` (this file)

NOT committed (gitignored via `SHUD/.git/info/exclude` per project
nwm-data discipline):

- Any file under `SHUD/Basins/**` — including the edited `.tsd.forc`
  / `.cfg.para` and their `.orig` snapshots. They exist only in the
  local working tree.

`git status` after fixup shows only `tools/` and `fixup_report.md` as
new — no Basins data leaked.

## Round 2 — fixes A1–A4 / B1–B3 / C1

Round 1 verifier confirmed 4 Critical issues and 3 Warning issues
(documented in the round-1 review). Round 2 closes them by failure class.
Per-fix design + verification result below.

### Fix A1 — Flag-parser orthogonality (closes C1: destructive flag-combo)

**Root cause (round 1)**: `--dry-run --restore <case>` evaluated as
`mode=restore` (last-write-wins on a single `mode` variable), executing
destructive restore while user expected preview. Empirically reproducible.

**Round 2 fix**: `DRY_RUN ∈ {true,false}` and `MODE ∈ {apply,restore}`
are now orthogonal in the option-parser loop. All 4 mode×dry-run combos
work; flag order is unrestricted. Every mutating op in the restore path
(`cp`, `rm`) is gated behind `if [[ "$dry_run" != "true" ]]; then ...`
with a `WOULD: …` log prefix when dry-run.

**Verification** (all PASS):

- `--dry-run --restore keliya` → `.orig` preserved + working file SHA
  unchanged from pre-cmd state.
- `--restore --dry-run keliya` (reversed flag order) → same.
- `--dry-run keliya` (no `--restore`) → no mutation.
- `--restore keliya` (no `--dry-run`) → mutation as expected, working
  file SHA matched former `.orig` SHA bit-exact.
- `--all --dry-run --restore` → 12 `.orig` files preserved, 0 mutations
  across all 6 cases (SHA snapshot before == after).

### Fix A2 — Explicit FS-op return-check (closes C2: false PASS on FS failure)

**Root cause (round 1)**: `mktemp`/`cp`/`mv` calls inside
`rewrite_tsd_forc`, `rewrite_cfg_para_num_openmp`, `snapshot_orig`,
`apply_case` were unchecked. With `if ! apply_case "$name"; then`, bash
suppresses `set -e` inside the function, so an unchecked `mktemp`
failure could fall through to `return 0` → PASS reported on broken case.

**Round 2 fix**: every FS mutation now chains
`|| { cleanup_partial; return <nonzero>; }`. Specifically `mktemp`,
`awk > tmp`, `mv tmp target`, `cp -p`, `rm`, `cat`, `printf` — all
return-checked. Partial temp files are unlinked on failure paths.

**Verification**: synthetic fixture `/tmp/fix_a2_fixture/SHUD/Basins/fakecase/`
with `chmod -w` on `input/fakecase/`. Result:

```
cp: …/fakecase.tsd.forc.orig: Permission denied
ERROR: snapshot_orig: cp failed for …
FAIL (unknown): fakecase
--- exit code: 1 ---
```

PASS criterion met: non-zero exit, FAIL printed, no silent PASS.

### Fix A3 — SIGINT/EXIT trap with in-flight rollback (closes C3: atomicity)

**Root cause (round 1)**: no `trap` registered; SIGINT during `apply_case`
between rewrite_tsd_forc and rewrite_cfg_para left the case in a
half-state (line 2 of tsd.forc rewritten, cfg.para untouched).

**Round 2 fix**: `_on_interrupt` (registered on `INT`/`TERM`) performs
immediate rollback from `.orig` and exits 130. We do *not* defer to the
EXIT trap, because bash queues signals delivered during external commands
(awk/mv/sleep): a deferred design would let the script resume, finish
mutating, and clear the in-flight markers before EXIT ran. `_on_exit_rollback`
remains as a safety net for double-signal scenarios. Per `apply_case`
and `restore_case`: before any mutation, set
`IN_PROGRESS_CASE` / `IN_PROGRESS_TSD_FORC` / `IN_PROGRESS_CFG_PARA`;
clear them on clean end. Bulk loop checks `INTERRUPTED` after each case
and breaks rather than starting the next.

**Verification** (synthetic fixture, sleep injected between the two
rewrites; SIGTERM sent at t=1.5s while sleep is active):

```
exit=130
stderr:
  interrupted, rolling back fakecase
  interrupted, rolled back fakecase
```

Pre-apply tsd.forc/cfg.para SHA256 → post-interrupt SHA256: **bit-equal**.
Rollback succeeded.

Natural exit (no SIGINT, normal completion) → no `rolling back` /
`EXIT trap` stderr; working file in applied state, not pristine. PASS.

### Fix A4 — tailanhe focing/ typo handling (closes C4: silent PASS → broken downstream)

**Root cause (round 1)**: tool wrote `forcing/` (canonical spelling per
spec) but tailanhe upstream has only `focing/`. Downstream
`./shud tailanhe` would fail at runtime because tsd.forc line 2 didn't
resolve to an existing dir.

**Round 2 fix**: `resolve_case_paths` now sets `FORCING_DIR_KIND ∈
{canonical, non-canonical-focing, missing}`. Priority order: `forcing/` →
`focing/` → FAIL. When `focing/` is used, emits a loud WARN
("model runtime should consume from this path"); if both are absent,
returns non-zero with an explicit error message.

**Verification**:

- tailanhe → tsd.forc line 2 now = `…/SHUD/Basins/tailanhe/focing` (the
  actually-existing dir). WARN emitted: `tailanhe: using non-canonical
  forcing dir 'focing/' (upstream typo); model runtime should consume
  from this path`. Exit 0, PASS (auxiliary).
- All other 5 cases → still resolve to `forcing/`; no behavior change.
- Synthetic `test_case/` with no `forcing/` nor `focing/` → explicit
  `ERROR: test_case: neither forcing/ nor focing/ found under …` + FAIL
  + exit 1.

### Fix B1 — NUM_OPENMP duplicate detection (closes W-Corr-4)

**Root cause (round 1)**: spec requires "唯一一行 `NUM_OPENMP\t1`", but
awk substituted ALL matching lines rather than asserting uniqueness. No
case in the current 6 has duplicates, but the tool was not defensive.

**Round 2 fix**: count matches first; if ≥2, log WARN with line numbers
("`NUM_OPENMP appears on multiple lines (L1,L2,…); will dedupe (keep
first, drop the rest)`"). The dedupe awk pattern uses a sentinel
`seen=1` flag: first match → rewrite to `NUM_OPENMP\t1`; subsequent
matches → drop.

**Verification**: synthetic `dupcase/dup.cfg.para` with three
`NUM_OPENMP` lines (8/4/2 values on lines 2/4/6). Result:

```
WARN: NUM_OPENMP appears on multiple lines (2,4,6); will dedupe (keep first, drop the rest)
```

Post-apply file:

```
OTHER_PARAM	5
NUM_OPENMP	1
MORE	1
END
```

Exactly 1 `NUM_OPENMP` line; value = 1; tab-separated (`od -c` confirmed
`NUM_OPENMP\t1\n`). PASS.

### Fix B2 — Case allow-list for --all (closes W-Int-3)

**Root cause (round 1)**: `list_local_cases` iterated `Basins/*/` and
skipped only `SERVER_ONLY_CASES` + hidden dirs. Any leftover dir under
`Basins/` would be silently processed.

**Round 2 fix**: `list_local_cases` iterates
`BENCHMARK_CASES ∪ AUXILIARY_CASES`, intersected with actually-present
directories. Unknown dirs → SKIP with explicit WARN
("`skipping unknown dir under Basins/: <name>`"); server-only dirs →
SKIP with WARN ("`skipping server-only case under Basins/: <name>`").
Emission order is benchmark-then-auxiliary for determinism.

**Verification**: synthetic fixture with `keliya` + `kashigeer` +
`mystery_case` + `heihe`. `--all` output:

```
WARN: skipping server-only case under Basins/: heihe (do not deploy locally)
WARN: skipping unknown dir under Basins/: mystery_case (not in BENCHMARK_CASES, AUXILIARY_CASES, or SERVER_ONLY_CASES)
PASS (benchmark): keliya
PASS (benchmark): kashigeer
```

`mystery_case` and `heihe` correctly skipped. PASS.

### Fix B3 — Server-only case direct-invoke refusal (closes W-Int-2)

**Round 2 fix**: after option parsing, before `apply_case` / `restore_case`,
check if `case_arg ∈ SERVER_ONLY_CASES`. If so, exit 2 with explicit
message: "`case '<name>' is server-only; bulk --all skips it; direct
invocation is refused (data is 12 G+, do not download locally)`".

**Verification**: direct invoke of `heihe` / `heihe_x4` / `heihe_x16`
all exit 2 with the expected message; no FS mutation. PASS.

### Fix C1 — fixup_report.md updates (Suggestion)

Added §"Known runtime caveats", §"Reproduction recipe", §"Round 2
verification matrix" below.

## Round 2 verification matrix

| Fix | Spec area | Verification | Result |
| --- | --- | --- | --- |
| A1 | Flag orthogonality (5 sub-tests) | `--dry-run --restore`, reversed, dry-run apply, real restore, `--all --dry-run --restore` | PASS |
| A2 | FS-op return-check | read-only parent dir → non-zero exit + FAIL printed | PASS |
| A3 | SIGINT trap rollback | SIGTERM during injected sleep, pre/post SHA bit-equal | PASS |
| A3 | Natural exit safety | no rollback msgs on clean completion, applied state preserved | PASS |
| A4 | tailanhe focing/ | line 2 = `.../tailanhe/focing`, WARN emitted | PASS |
| A4 | Other 5 cases | line 2 = `.../<case>/forcing`, no WARN | PASS |
| A4 | Missing forcing/focing | explicit FAIL with error message | PASS |
| B1 | Duplicate NUM_OPENMP | WARN with line nums, dedupe to 1 line `NUM_OPENMP\t1` | PASS |
| B2 | Allow-list --all | unknown + server-only dirs SKIP with WARN | PASS |
| B3 | Server-only refusal | direct invoke of heihe / heihe_x4 / heihe_x16 → exit 2 | PASS |
| — | 6-case regression | `--all` PASS×6, failures=0 | PASS |
| — | Idempotency | re-run `--all` SHA bit-equal | PASS |
| — | Restore reversibility | `--all --restore` SHA bit-equal to former .orig | PASS |
| — | OpenSpec strict | `openspec validate s0-baseline-lock --strict --no-interactive` | PASS |

## Known runtime caveats

- **tailanhe `focing/` vs `forcing/`**: upstream NWM data for `tailanhe`
  ships a directory named `focing/` (missing `r`). The fixup tool detects
  this typo and writes line 2 of `tlh.tsd.forc` pointing to the
  actually-existing `focing/` so that downstream `./shud tailanhe` will
  resolve the path successfully. A loud WARN is emitted on every apply.
  If a future upstream NWM zip corrects the dir to `forcing/`, the tool
  will auto-pick the canonical name without further changes.
- **Idempotent re-apply restores from `.orig`**: each `--all` / single-
  case apply call internally restores the working file from `.orig`
  before re-applying the edits. This guarantees deterministic result
  regardless of the file's current state, but it means a manual edit to
  a working tsd.forc / cfg.para between runs will be silently reverted.
  Use `--dry-run` to preview without overwriting.

## Reproduction recipe

```bash
# Starting from a fresh NWM checkout (6 cases under SHUD/Basins/<case>/
# with upstream content; no .orig files anywhere).

cd /Users/danker/Desktop/Hydro-SHUD/openMP

# 1. Snapshot baseline SHA256 (sanity, optional).
shasum -a 256 SHUD/Basins/*/input/*/*.tsd.forc \
              SHUD/Basins/*/input/*/*.cfg.para > /tmp/pre.sha

# 2. First apply.
./tools/fix_case_paths/fix_case_paths.sh --all
# Expected: PASS×6 (5 benchmark + 1 auxiliary), failures=0, 12 .orig files.

# 3. Idempotent re-apply — must produce SHA-equal result.
shasum -a 256 SHUD/Basins/*/input/*/*.tsd.forc \
              SHUD/Basins/*/input/*/*.cfg.para > /tmp/post1.sha
./tools/fix_case_paths/fix_case_paths.sh --all
shasum -a 256 SHUD/Basins/*/input/*/*.tsd.forc \
              SHUD/Basins/*/input/*/*.cfg.para > /tmp/post2.sha
diff /tmp/post1.sha /tmp/post2.sha   # must be empty (idempotent)

# 4. Restore — working files must SHA-equal pre-apply state.
./tools/fix_case_paths/fix_case_paths.sh --all --restore
# Expected: RESTORED×6, failures=0, 0 .orig files remaining.
shasum -a 256 SHUD/Basins/*/input/*/*.tsd.forc \
              SHUD/Basins/*/input/*/*.cfg.para > /tmp/post_restore.sha
diff /tmp/pre.sha /tmp/post_restore.sha   # must be empty (bit-exact)

# 5. Re-apply for downstream issues.
./tools/fix_case_paths/fix_case_paths.sh --all
```

Server-side recipe is identical but excludes `heihe*` (server-only;
12 G+ forcing). Direct invocation of any `SERVER_ONLY_CASES` member is
refused with exit 2.
