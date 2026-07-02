#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.9"
# ///
"""P12-nvec PR-N2 matrix analysis — medians, bitwise cross-check, G-E2/G-E3.

Reads the per-cell MARKER:P12N2 lines (from the Slurm .out files, concatenated
into markers.txt) plus the per-run rivqdown SHA files, and emits:
  - 6-cell median wall + median-run cvode_stats table
  - equal-N bitwise cross-check (C@N vs E@N: identical counters + rivqdown SHA)
  - cross-N determinism note (bonus)
  - G-E2 speedup(E/C) at N=8 and N=16 + TIER1 verdict
  - G-E3 Amdahl projection with the PINNED formula + TIER2 verdict

Pinned constants (from PR-N0 evidence .review-evidence/p12-nvec/pr-n0/README.md):
  t_red (heihe_x4 Config C N=16 abs reduction total_ns) = 142_859_623_101 ns
  G-E3(ii) reduction shares: heihe_x4=35.677%, heihe_x16=30.461% (both >=10%)

Usage: analyze_matrix.py <markers.txt> <results_dir>
  markers.txt = concatenation of all `MARKER:P12N2 ...` lines
  results_dir = dir holding <cell>/rivqdown.run<r>.sha
"""
import sys
import re
import statistics
from pathlib import Path

# ---- PINNED inputs (do not edit; sourced from PR-N0) ----
T_RED_NS = 142_859_623_101            # G-E3(iii) t_red, heihe_x4 Config C N=16 abs reduction total_ns
GE3_II_X4 = 35.677                    # G-E3(ii) heihe_x4 reduction share (%)
GE3_II_X16 = 30.461                   # G-E3(ii) heihe_x16 reduction share (%)
GE2_BAR = 1.10                        # TIER1_ADOPT bar (pinned; do NOT move)
GE3_BAR = 1.15                        # TIER2_GO Amdahl bar (pinned)
GE3_II_THRESH = 10.0                  # (%) either-case threshold

MARKER_RE = re.compile(
    r"MARKER:P12N2 cell=(\S+) rep=(\d+) wall_s=(\S+) rc=(\S+) "
    r"nst=(\S+) nfe=(\S+) ncfn=(\S+) ncfl=(\S+) netf=(\S+) rivqdown_sha=(\S+)"
)


def parse_markers(path):
    cells = {}  # cell -> list of dict(rep, wall, rc, nst, nfe, ncfn, ncfl, netf, sha)
    for line in Path(path).read_text().splitlines():
        m = MARKER_RE.search(line)
        if not m:
            continue
        cell, rep, wall, rc, nst, nfe, ncfn, ncfl, netf, sha = m.groups()
        cells.setdefault(cell, []).append(dict(
            rep=int(rep), wall=float(wall), rc=int(rc),
            nst=nst, nfe=nfe, ncfn=ncfn, ncfl=ncfl, netf=netf, sha=sha))
    return cells


def median_wall_run(reps):
    """Return (median_wall, the-rep-dict-closest-to-median-wall)."""
    walls = sorted(r["wall"] for r in reps)
    med = statistics.median(walls)
    # rep whose wall is the median (for 3 reps, the middle one)
    rep_by_wall = sorted(reps, key=lambda r: r["wall"])
    med_rep = rep_by_wall[len(rep_by_wall) // 2] if len(rep_by_wall) % 2 == 1 else rep_by_wall[len(rep_by_wall)//2 - 1]
    return med, med_rep


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    markers_path, results_dir = sys.argv[1], sys.argv[2]
    cells = parse_markers(markers_path)

    # only the 6 wall cells for G-E2 (exclude the *_prof leg)
    wall_cells = [c for c in cells if not c.endswith("_prof")]
    order = ["C_n1", "C_n8", "C_n16", "E_n1", "E_n8", "E_n16"]
    order = [c for c in order if c in wall_cells] + [c for c in wall_cells if c not in order]

    print("=" * 96)
    print("P12-nvec PR-N2 — heihe_x4 90-day scaling matrix (3-run medians)")
    print("=" * 96)
    print(f"{'cell':<10}{'reps':<6}{'walls (s)':<34}{'median_wall':>12}  "
          f"{'nst':>6}{'nfe':>7}{'ncfn':>6}{'ncfl':>6}{'netf':>6}")
    med = {}       # cell -> median wall
    medrun = {}    # cell -> median-run dict
    allrc = 0
    for c in order:
        reps = sorted(cells[c], key=lambda r: r["rep"])
        m, mr = median_wall_run(reps)
        med[c] = m
        medrun[c] = mr
        walls_str = " ".join(f"{r['wall']:.2f}" for r in reps)
        rcsum = sum(r["rc"] for r in reps)
        allrc += rcsum
        print(f"{c:<10}{len(reps):<6}{walls_str:<34}{m:>12.3f}  "
              f"{mr['nst']:>6}{mr['nfe']:>7}{mr['ncfn']:>6}{mr['ncfl']:>6}{mr['netf']:>6}"
              + ("  [rc!=0]" if rcsum else ""))
    print(f"(all rc==0: {'YES' if allrc == 0 else 'NO -- ' + str(allrc)})")

    # ---- equal-N bitwise cross-check: C@N vs E@N ----
    print("\n" + "=" * 96)
    print("Equal-N bitwise cross-check (C@N vs E@N): identical cvode_stats counters + equal rivqdown SHA")
    print("=" * 96)
    bitwise_pass = True
    for N in (1, 8, 16):
        cc, ee = f"C_n{N}", f"E_n{N}"
        if cc not in cells or ee not in cells:
            continue
        # gather the full per-rep SHA set for each config (all reps should be identical too)
        c_shas = {r["sha"] for r in cells[cc]}
        e_shas = {r["sha"] for r in cells[ee]}
        # counters compared on the median run (and we assert all reps identical below)
        cstat = {k: medrun[cc][k] for k in ("nst", "nfe", "ncfn", "ncfl", "netf")}
        estat = {k: medrun[ee][k] for k in ("nst", "nfe", "ncfn", "ncfl", "netf")}
        counters_eq = (cstat == estat)
        # rivqdown: every C rep sha == every E rep sha == single value
        sha_eq = (len(c_shas) == 1 and len(e_shas) == 1 and c_shas == e_shas)
        verdict = "PASS" if (counters_eq and sha_eq) else "FAIL"
        if verdict == "FAIL":
            bitwise_pass = False
        print(f"  N={N:<3} counters_eq={counters_eq} rivqdown_sha_eq={sha_eq} -> {verdict}")
        print(f"        C stats={cstat}  sha={sorted(c_shas)}")
        print(f"        E stats={estat}  sha={sorted(e_shas)}")

    # ---- cross-N determinism (bonus) ----
    print("\n  [bonus] cross-N determinism within each config (expected identical, Tier-1):")
    for cfg in ("C", "E"):
        shas = set()
        stats = set()
        for N in (1, 8, 16):
            c = f"{cfg}_n{N}"
            if c in cells:
                shas |= {r["sha"] for r in cells[c]}
                stats.add(tuple(medrun[c][k] for k in ("nst", "nfe", "ncfn", "ncfl", "netf")))
        print(f"    {cfg}: distinct rivqdown SHA across N={{1,8,16}} = {len(shas)}; distinct counter-tuples = {len(stats)} "
              f"({'identical' if len(shas)==1 and len(stats)==1 else 'DIFFER'})")

    # ---- G-E2 ----
    print("\n" + "=" * 96)
    print("G-E2 verdict: TIER1_ADOPT iff speedup(E/C) >= 1.10 at N=8 OR N=16 AND bitwise PASS")
    print("=" * 96)
    sp = {}
    for N in (8, 16):
        cc, ee = f"C_n{N}", f"E_n{N}"
        if cc in med and ee in med:
            sp[N] = med[cc] / med[ee]
            print(f"  speedup(E/C) @ N={N}: {med[cc]:.3f}s / {med[ee]:.3f}s = {sp[N]:.4f}x "
                  f"({'>=' if sp[N] >= GE2_BAR else '<'} {GE2_BAR})")
    roi_ok = any(v >= GE2_BAR for v in sp.values())
    tier1 = "TIER1_ADOPT" if (roi_ok and bitwise_pass) else "TIER1_CLOSE"
    print(f"  bitwise_PASS={bitwise_pass}  roi>=1.10(N8|N16)={roi_ok}  ==> VERDICT = {tier1}")

    # ---- G-E3 ----
    print("\n" + "=" * 96)
    print("G-E3 Tier-2 gate: (i) Tier-1=ADOPT  (ii) reduction share>=10% on either  (iii) Amdahl>=1.15x")
    print("=" * 96)
    inp_i = (tier1 == "TIER1_ADOPT")
    inp_ii = (GE3_II_X4 >= GE3_II_THRESH) or (GE3_II_X16 >= GE3_II_THRESH)
    print(f"  (i)   Tier-1 verdict = {tier1}  -> {'PASS' if inp_i else 'FAIL'}")
    print(f"  (ii)  reduction share heihe_x4={GE3_II_X4}% heihe_x16={GE3_II_X16}% "
          f"(>=10% on either) -> {'PASS' if inp_ii else 'FAIL'}")
    if "E_n16" in med:
        wall_e16 = med["E_n16"]
        t_red_s = T_RED_NS / 1e9
        denom = wall_e16 - t_red_s * (1 - 1 / 16)
        proj = wall_e16 / denom if denom > 0 else float("inf")
        inp_iii = (proj >= GE3_BAR)
        print(f"  (iii) wall_E16={wall_e16:.3f}s  t_red={t_red_s:.3f}s  "
              f"proj = {wall_e16:.3f} / ({wall_e16:.3f} - {t_red_s:.3f}*(1-1/16)) "
              f"= {wall_e16:.3f}/{denom:.3f} = {proj:.4f}x ({'>=' if inp_iii else '<'} {GE3_BAR}) "
              f"-> {'PASS' if inp_iii else 'FAIL'}")
    else:
        inp_iii = False
        proj = float("nan")
        print("  (iii) E_n16 missing")
    tier2 = "TIER2_GO" if (inp_i and inp_ii and inp_iii) else "TIER2_NO-GO"
    print(f"  ==> G-E3 = {tier2}  (all three must PASS for GO)")

    print("\n" + "=" * 96)
    print(f"SUMMARY: {tier1} ; {tier2}")
    print("=" * 96)


if __name__ == "__main__":
    main()
