# Case deployment fixup report — Issue #4 (S0.1b)

- **Timestamp**: 2026-06-16T19:23:27Z
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
  - `after  line 2`: `/Users/danker/Desktop/Hydro-SHUD/openMP/SHUD/Basins/tailanhe/forcing`
- File: `SHUD/Basins/tailanhe/input/tlh/tlh.cfg.para`
  - `NUM_OPENMP`: `8` → `1` (line 8)
- Other lines: byte-preserved
- **Note (forcing dir typo)**: tailanhe ships with `SHUD/Basins/tailanhe/focing/`
  (typo, no `r`) and no `forcing/` directory. The fixup tool writes the
  conventional `SHUD/Basins/tailanhe/forcing/` path per spec and emits a
  WARN. Downstream consumers either need to fix the directory name in the
  upstream NWM zip or extend the tool to follow the typo — out of scope
  for this fixup (the tool's job is path translation, not data layout).

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
