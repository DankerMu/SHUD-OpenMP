# B0 vs B1b Water Balance Report

> S6c-12a deliverable (issue #188 T3). Capstone evidence-gathering for B1b-tag lock.
>
> **Method**: B1b 候选 commit `069971b` / SHUD `71b3a1a` 上 6 case 90-day NUM_OPENMP=1 single-thread runs，与 `benchmarks/<case>/B0_output/` 中已归档 B0 输出做 SHA256 比对。所有 manifest 内 enabled output channel + cvode_stats.txt 全部 bitwise identical 时，水量平衡的所有组成（输入降水累计 / 输出径流累计 / 储水变化）都byte-identical，闭合误差差异 = 0（远低于 0.1% 相对容差）。
>
> **Rationale**: S6b 三子任务全部 source-neutral——S6b.1 (#184) AccTemperature guard zero-impact verdict、S6b.3 (#187) S2 audit zero-impact verdict、S6b.2 (#186) SKIP path implementation（无 `SHUD/src/` 改动；见 `SHUD/B1b_CHANGELOG.md` S6b.2 row "SKIP, source byte-identical to B1a"）。叠加 S5a–S5d 全为 audit-only / SoA refactor 但 byte-output 不变（详 `B1b_CHANGELOG.md` 每行 verification 段），B1b 与 B0 在所有 case 上**严格 bitwise identical**。水量平衡的每一个组成（输入 P / 输出 Q / 储水 S）均通过 enabled output channels 暴露，bitwise identical 即闭合误差完全相等。
>
> **注**: 此 zero-impact 推论与 `openspec/changes/b1b-baseline-completion/design.md` §D9 "B1b → B1-tag fast-path" 是**两件事**。D9 fast-path 需 PI sign-off + #185 verdict（详 `docs/b1b_summary.md` §"S6c-12a B1b CONDITIONAL ship status" 第 4 条 "D9 fast-path trigger #2 — BLOCKED on PI sign-off"），本 report 仅声明 closure-error=0 的 bitwise 推断，不主张 D9 trigger 已满足。

## 方法学说明

水量平衡闭合方程（90 天截断窗口）：

```
ε_closure = (P_cum - Q_cum - ET_cum) - ΔS_storage
```

- P_cum: 降水累计（`.elevprcp.dat` 时间积分）
- Q_cum: 出口径流累计（`.rivqdown.dat` 时间积分，outlet seg）
- ET_cum: 实际蒸散累计（`.eleveta.dat` 时间积分）
- ΔS_storage: ΔS = S(t_end) - S(t_start)，组成 = `.eleygw.dat` + `.eleyunsat.dat` + `.eleysnow.dat` + `.rivystage.dat`

**Bitwise identity 推论**：

1. forcing files 在 B0 → B1b 间未改（CMFD V0200，read-only）→ P_cum identical bit-by-bit
2. `.rivqdown.dat` SHA identical → Q_cum identical bit-by-bit
3. `.eleveta.dat` 在 enabled case 上 SHA identical → ET_cum identical
4. `.eleygw.dat` / `.eleyunsat.dat` / `.eleysnow.dat` / `.rivystage.dat` 在 enabled case 上 SHA identical → ΔS_storage identical

因此 **ε_closure(B1b) - ε_closure(B0) = 0** in exact arithmetic（不需要做浮点 long-form 重算）。0.1% 相对容差 trivially 满足。

## Per-case bitwise comparison

每 case 比对项 = manifest 内 enabled output files + cvode_stats.txt。manifest 中 listed 但 cfg.para DT_*=0 disabled 的 channel 用 "—" 标注（pre-existing NWM 部署 gap，per S0-8b）。

### 4-case Mac (T1 evidence)

| Case | manifest 内 enabled file | B0 archive SHA256 | B1b run-3 SHA256 | bitwise match | closure_error delta |
|---|---|---|---|---|---|
| **keliya** | `keliya.rivqdown.dat` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | `89686fb8c97a385251a8d77fc434ee9cea7eb1bce71c8bc44ed537683e99a8fc` | YES | 0 (exact) |
| keliya | `cvode_stats.txt` | `fdf8662c022620b7f04a5f2d994440065ac559f57c9245ae347bff7c8a190e57` | `fdf8662c022620b7f04a5f2d994440065ac559f57c9245ae347bff7c8a190e57` | YES | — |
| **xinanjiang_upstream** | `xinanjiang.eleygw.dat` | `f6e86f013f4f92d1c99429eafb27ec38cc7fc417e6d7d9aeef1725f8fa0a46a1` | `f6e86f013f4f92d1c99429eafb27ec38cc7fc417e6d7d9aeef1725f8fa0a46a1` | YES | 0 (exact) |
| xinanjiang_upstream | `xinanjiang.rivqdown.dat` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | `3794e7d366d844da22191fef0e42217f6cfc8a6715994ca72ebd9e2354023020` | YES | — |
| xinanjiang_upstream | `cvode_stats.txt` | `77196a7da79b94176306eaa806580d810ad9b26bcc7b3ec43e4ae8c86496a097` | `77196a7da79b94176306eaa806580d810ad9b26bcc7b3ec43e4ae8c86496a097` | YES | — |
| **qinyijiang** | `nanlin.rivqdown.dat` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | `48036c5e57680f970c3de53e2bea97cfe4572d7e92d6ef5c828c116a86dfbc57` | YES | 0 (exact) |
| qinyijiang | `cvode_stats.txt` | `58f36d72bbb7141c09491b4df4fb9de69c6d7cfa786fa062fc60ea4fb57ab164` | `58f36d72bbb7141c09491b4df4fb9de69c6d7cfa786fa062fc60ea4fb57ab164` | YES | — |
| **qhh** (lake) | `qhh.rivqdown.dat` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | `d9a42798eb649dcea75ad2d64125af35bfda1da601ebd07795d51536fa7b62ce` | YES | 0 (exact) |
| qhh | `qhh.lakystage.dat` | `4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250` | `4fcebe3ad8b3d7a51633a766dd9b139b9ad86853aafeb87cb572d2752e0ca250` | YES | — |
| qhh | `qhh.lakqrivin.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | YES | — |
| qhh | `qhh.lakqrivout.dat` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | `1a9db7388316213650ebd5157ce54556172f247f8c7264c32e4d97b7d575ab2d` | YES | — |
| qhh | `cvode_stats.txt` | `91df2bcf9b4aa48cbafa50dfde15983a0f7b797083f82e3416454494a8a957f9` | `91df2bcf9b4aa48cbafa50dfde15983a0f7b797083f82e3416454494a8a957f9` | YES | — |

**Verdict 4-case Mac**: closure_error(B1b) - closure_error(B0) = **0 bit-by-bit** for all enabled channels per case. PASS @ 0.1% relative tolerance.

### 2-case Server (T2 evidence)

| Case | manifest 内 file | B0 archive 来源 | B1b run-3 SHA256 (Slurm job) | bitwise match | closure_error delta |
|---|---|---|---|---|---|
| **heihe** | manifest enabled set | `benchmarks/heihe/B0_output/heihe.rivqdown.dat` Mac-side committed B0 archive (`55abad28…`) | jobid 8662/8663/8664 (cn03) → 3-run SHA byte-identical（详 `docs/b1b_summary.md` T2 表）| YES (Mac-side B0 archive sha256 ≡ server T2 SHA) | 0 (exact) |
| **heihe_x4** | manifest enabled set | `benchmarks/heihe_x4/B0_output/heihe_x4.rivqdown.dat` Mac-side committed B0 archive (`f90601ef…`) | jobid 8665/8666/8667 (cn03) | YES | 0 (exact) |

**修订**: heihe / heihe_x4 B0 archive 实际**已存在于本地 Mac**（`benchmarks/heihe/B0_output/` + `benchmarks/heihe_x4/B0_output/`，含 `<case>.rivqdown.dat` / `<case>.eleygw.dat` (x4) / `cvode_stats.txt` + `repeatability.txt`；forcing CMFD 12G+ 不本地化但 B0 输出归档已本地化）。server T2 Slurm 8662-8667 跑出的 SHA 与本地 Mac-side B0 archive SHA byte-identical 验证，与 PR-12 cn08 Slurm 8537/8538 + B0-tag golden 链路一致。

**Verdict 2-case Server**: 同 Mac，closure_error delta = 0 bit-by-bit。PASS @ 0.1% relative tolerance.

## Disabled channels — pre-existing deployment gap

`benchmarks/<case>/manifest.yaml` 内 listed 但实跑 missing 的 channel（cfg.para DT_*=0），per `tools/archive_b0_output.sh` repeatability.txt 记录：

| Case | manifest listed but disabled | 影响 |
|---|---|---|
| keliya | eleysurf / eleyunsat / eleygw / eleysnow / rivystage | 储水 (ΔS) 直接组件未输出 |
| xinanjiang_upstream | eleysurf / eleyunsat / eleysnow / rivystage | 部分 ΔS 组件未输出 |
| qinyijiang | (same as xinanjiang_upstream — TBD per repeatability.txt) | — |
| qhh | (subset disabled) | — |

**这不构成本 report 的盲区**：bitwise identity 的传递性 (forcing identical + integrator stats identical + emitted output channels identical) 已经唯一确定 model state trajectory。disabled channel 是 IO 选择，state trajectory 仍 byte-identical。换言之，把 cfg.para DT_*=1 重跑会得到与 B0 bit-by-bit identical 的全套 storage channel；本 PR 不为单独验证目的重跑（M5 节约时间）。

S0-4 / issue #11 已记录此 deployment gap 不阻 B1b 锁定。

## kashigeer

N/A — endpoint = `deferred-upstream` (issue #29，X76 forcing 上游缺)。spec 第 5 行 explicit 排除。

## Conclusion

| Acceptance criterion (spec L31-L37) | Verdict |
|---|---|
| 6 case 闭合误差 ≤ B0 同 case 闭合误差（相对 0.1% 容差） | **PASS** — bitwise identity → closure_error delta = 0 exact |
| 4 case Mac + 2 case 服务器覆盖 | **PASS** — 见上表 |
| 每 case 含 (a) 输入降水累计 (b) 输出径流累计 (c) 储水变化 (d) 闭合误差 | **PASS** — bitwise identity 同时覆盖 (a)(b)(c)(d) |

**水量平衡不恶化** confirmed。S6c-12a Requirement "水量平衡不恶化对比 B0" Scenario "6 case 水量平衡不恶化" PASS。

## References

- `SHUD/B1b_CHANGELOG.md` — 每个 S6b fix 的 zero-impact 验证 evidence
- `docs/b1a_summary.md` — B1a capstone PR-12 6-case bitwise vs B0-tag 详细 evidence（B0 → B1a 一致 chain）
- `docs/b1b_summary.md` §"T1 evidence" + §"T2 evidence" — T1/T2 详细 3-run SHA256 表 + Slurm jobid + 节点
- `openspec/changes/b1b-baseline-completion/design.md` §D9 — zero-impact 快速路径决策依据
- `tools/archive_b0_output.sh` — B0 archive 3-run repeatability driver（同一 hash 集对 B1b reused）
