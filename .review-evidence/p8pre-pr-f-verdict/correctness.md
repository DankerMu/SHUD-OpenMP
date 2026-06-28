Reviewer agent: review-correctness
Review round: round 1
Reviewed head SHA: 75a757c
Summary: Aggregator logic + verdict adjudication are correct and reproducible against the 18-cell mirror; doc §3 + YAML frontmatter match gate outputs. No Critical or Warning findings.

Findings:
- None.

Diff/scope checks:
- `git diff baseline/p8pre...75a757c -- SHUD .gitmodules openspec/`: empty.
- `git diff baseline/p8pre...75a757c --name-only`: exactly 2 files (doc + aggregator).
- `openspec validate p8pre-spike --strict --no-interactive`: exit 0.
- `bash -n tools/p8pre/aggregate_identity_spike.sh`: SYNTAX_OK.

Aggregator logic audit (`tools/p8pre/aggregate_identity_spike.sh`):
- Median3 (L171-174): `sort -n | sed -n '2p'` middle-of-3. Cross-checked heihe_N1 (137.079,137.273,121.619 → 137.079) and heihe_x4_N1 (1491.050,1428.287,1273.800 → 1428.287) — match stdout.
- Gate 2 (L307-326): universal-quantifier "any ncfn>0 → FAIL" per spec L77. Observed 18/18 violations; FAIL emitted.
- Gate 4 delta (L375): `|a-b|/b` matches spec L94-95. heihe_N1 2.640% and heihe_x4_N1 1.089% match.
- Gate 5 ULP (L202-250): `np.spacing(max(|a|,|b|))` with EQ + finite + size-mismatch guards. 18/18 cells mode=OK, max_ulp ≈ 9×10¹⁵ ≫ 1024; FAIL emitted.
- SHA12 (L266,L271): `sha256sum | cut -c1-12` correct.

YAML frontmatter cross-check (`identity_spike_verdict.md:11-20`): verdict=NO-GO ✓, gate_2=FAIL ✓, gate_5=FAIL ✓, adr_recommendation=NO-GO ✓, SHUD_pin=5276167 ✓ — match aggregate_verdict.txt + run log.

§3 Table 1 (18 rows × 15 columns): row 2 wall_total 137.079 matches cell_stats L2 (137.079454928). Row 11 wall_total 1491.050 matches L11 (1491.049945147). All 18 present; counters deterministic (heihe ncfn=6, heihe_x4 ncfn=47).

§4 Gate-4 baselines (L122-129): 6 (case,N) tuples match `BASELINE_WALL_MEDIAN[]` aggregator L78-83 + `docs/p8pre/n8_profile_baseline.md` §5.1.

Output: `/tmp/p8pre_identity_spike/aggregate_verdict.txt` (135 lines structured KV), byte-identical mirror under `.review-evidence/p8pre-pr-f-verdict/`. PR-G #348 consumable.

Non-blocking notes:

(1) Suggestion — max_ulp precision (`identity_spike_verdict.md:143,152`): doc says "≈ 9×10¹⁵ across all cells"; aggregator emits 3 rounded values (8.99e15 / 9.00e15 / 9.01e15; raw 8,985,309,273,003,056 → 9,007,166,730,727,710). Range "[8.99 — 9.01] × 10¹⁵" would tighten §5 reproducibility. Does not affect FAIL.

(2) Needs verification — "5,155 / 214,252" structural divergence claim (`:146`): aggregator computes max_ulp scalar only, not the zero-position differential. Forward-carried from manual numpy investigation not reproduced by script. For verdict the max_ulp ≫ 1024 alone is decisive; §8.4 records alternative bitwise 154,665/214,252 — adequate carve-out disclosure. Future round could instrument `n_diff_zero_positions` emission.

(3) Praise — Gate 1 symbol-grep (L295-298): `grep -c` per symbol with composite ≥1 each test correctly implements spec L70-72. server_nm.log has exactly 3 matching lines (U CVodeSetPreconditioner, T PSetupIdentity, T PSolveIdentity); PASS emitted.

Verdict: APPROVE — spec L74-130 gate boundaries faithfully implemented; cross-checks pass; diff scope clean; openspec validate strict exit 0. Suggestion + needs-verification are cosmetic and do not block merge.
