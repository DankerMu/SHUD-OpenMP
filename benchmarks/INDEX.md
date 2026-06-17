# SHUD-OpenMP Benchmark Registry — INDEX

Human-readable index of the `benchmarks/<case>/manifest.yaml` set tracked by
this repo. Authoritative schema lives in
`openspec/changes/s0-baseline-lock/specs/benchmark-registry/spec.md`;
machine validation in `tools/check_manifest.py`. Master plan §S0.2 fixes the
field set, §S0.12 fixes the case×endpoint split that the `endpoint` column
encodes here.

Cases are bucketed by NumEle into **B-Small** (< 1 000), **B-Medium**
(1 000 – 10 000), and **B-Large** (10 000 – 100 000); §1.1.1 quantified
speedup goals are bucketed the same way. The `endpoint` column makes the
§S0.12 cross-platform split schema-explicit: `local-and-server` runs on
both, `server-only` runs on the production Linux node only (forcing too
heavy to localize, or mesh derived on server). `tailanhe` is a local
auxiliary case (not in this registry).

| Case ID               | NumEle  | endpoint           | Use / notes                                                                 | Source                                          |
|-----------------------|--------:|--------------------|-----------------------------------------------------------------------------|-------------------------------------------------|
| `keliya`              |    484  | local-and-server   | B-Small, CI primary; OMP_CUTOFF serial-fallback, A0 repeatability anchor    | NWM dataset (Tarim, arid)                       |
| `xinanjiang_upstream` |    801  | local-and-server   | B-Small backup; sits on OMP_CUTOFF=1024 boundary                            | NWM dataset (Yangtze sub-basin)                 |
| `qinyijiang`          |  3 155  | local-and-server   | B-Medium low; sparse river network typical load                             | NWM dataset (project name: `nanlin`)            |
| `kashigeer`           |  3 204  | local-and-server   | B-Medium; river/PassValue topology stress test (NumRiv/NumEle = 0.77)       | NWM dataset (project name: `ksge`)              |
| `qhh` (NWM version)   |  4 773  | local-and-server   | B-Medium; **sole `has_lake: true` case** — covers lake vertical/horizontal/DY | NWM dataset (Qinghai Lake; not the SHUD demo)   |
| `heihe`               |  6 335  | server-only        | B-Medium high; forcing 12 GB+, cryosphere path coverage                     | NWM dataset (Hexi inland river)                 |
| `heihe_x4`            | 40 046  | server-only        | B-Large; AutoSHUD-refined from `heihe` (NumCells=25340, 4× baseline 6335)   | AutoSHUD v2.5.0 (patched) + rSHUD v2.5.0 master |

> `heihe_x16` (XLarge, ~100 000) is deferred until P8 (master plan §5 S0.5
> note) and is **not** registered here.
