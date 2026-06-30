# G0-1 SPGMR baseline anchor (Mac, keliya 90-day SHORT)

Frozen pre-G0 SPGMR baseline produced under P8-tune.G0 PR-0 Phase 6.

## Provenance

- Platform: macOS (Apple Silicon)
- Cell: keliya (NumEle=484, 90-day truncated cfg.para)
- Build: `make HYPRE_INCDIR=/opt/homebrew/include HYPRE_LIBDIR=/opt/homebrew/lib shud`
- Compiler: Apple clang (Xcode 16+)
- SUNDIALS: 6.0.0 (vendored at SHUD/InstallSundials)
- Hypre: 3.1.0 (brew); linked but not exercised (default SHUD_LINSOL=spgmr path)
- Run command: `env -u SHUD_LINSOL ./shud keliya` (Basins/keliya/ as cwd)

## manifest.sha256 contents

```
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  DY.dat
c9137e84a82d9c4f954a7f8fd1235b96c93f0849ce8df26c0235a561274c437d  keliya.elevnetprcp.dat
0c51b7aaf4ab9c9f1e27b08beea0ee6a8c363e3c194986a2957b718a41cdc5e7  keliya.elevprcp.dat
b769e3270e1c4d075e7913bf0d0a229530200ae4b11663bdfa4a0cc3c9c028bd  keliya.rivqdown.dat
5a18ce3dd303d99fa8e2fc48dca99621622a8c708c571532a34ebc39598182ef  cvode_stats.txt
```

## DY.dat 0-byte transparency

`DY.dat` is the canonical empty-file SHA256 (`e3b0c4...`). This is
expected behavior for keliya 90-day SHORT runs with the default build
flags (`SHUD_DUMP_RHS=0`). SHUD emits `DY.dat` as a 0-byte file when
no RHS snapshot data is recorded. The file is included in the manifest
for completeness (presence-check + size-check) rather than nonzero
content verification.

If a future review requires a nonzero `DY.dat`, rebuild with
`make SHUD_DUMP_RHS=1 shud` and re-run; the resulting `DY.dat` will
contain RHS snapshot bytes and the manifest will need to be regenerated.

## cvode_stats.txt as G0-6 anchor

The `cvode_stats.txt` file captures the SUNDIALS 6.0 solver statistics
required by G0-6 solver-stats-documented gate:
- `nfe` / `nfeLS` / `nni` / `nli` / `nsetups` / `netf`

Plus 9 additional fields (`nst`, `npe`, `nps`, `ncfn`, `ncfl`, `lenrw`,
`leniw`, `lenrwLS`, `leniwLS`). Diagnostics fields (`hlast`, `qlast`,
RHS 7-bucket timers, forcing I/O timer) are gated behind
`SHUD_ENABLE_DIAGNOSTICS` and absent in this baseline.

## Default-compat invariant (G0-1)

The G0-1 default-compat gate compares any future SPGMR run's
`Output/*.dat` SHA256 set against this manifest. Bit-identity is the
PASS condition; any divergence (including different cvode_stats.txt
field set, e.g., diagnostics-on build) FAILs G0-1.

The Hypre/AMG path (`SHUD_LINSOL=amg`) is opt-in and does NOT need to
produce bit-identical bytes — it is exercised separately under G0-4
(integrated-completes) and G0-5 (wall-signal) gates.
