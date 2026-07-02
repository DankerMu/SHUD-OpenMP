#!/bin/zsh
# P11-osc PR-D1 trace/flip self-check (task 1.3).
# Usage: zsh selfcheck.sh <output_dir> <baseline_sha_manifest>
# Re-runnable against any keliya SHUD_DIAG_DT=1/SHUD_DIAG_OSC=1 run output.
set -e
OUT="${1:?output dir}"
TRACE="$OUT/diag_dt_trace.csv"
FL="$OUT/diag_osc_flips.csv"
DL="$OUT/diag_osc_flips_daily.csv"
STATS="$OUT/cvode_stats.txt"
fail=0

echo "== CHECK1: Sum(delta_nst) == cvode_stats nst =="
SUM_NST=$(awk -F, 'NR>2 && $0 !~ /^#/ {s+=$2} END{print s}' "$TRACE")
CV_NST=$(grep '^nst=' "$STATS" | cut -d= -f2)
echo "  Sum(delta_nst)=$SUM_NST  cvode_stats nst=$CV_NST"
[ "$SUM_NST" = "$CV_NST" ] && echo "  PASS" || { echo "  FAIL"; fail=1; }

echo "== CHECK2: trace row_count == CVode driver-loop invocations =="
ROWS=$(awk -F, 'NR>2 && $0 !~ /^#/' "$TRACE" | wc -l | tr -d ' ')
# keliya cfg.para: START=12053 END=12143 SolverStep(MAX_SOLVER_STEP)=20 min
EXP=$(awk 'BEGIN{print int((12143-12053)*1440/20)}')
echo "  rows=$ROWS  expected=(END-START)*1440/SolverStep=$EXP"
[ "$ROWS" = "$EXP" ] && echo "  PASS" || { echo "  FAIL"; fail=1; }

echo "== CHECK3: every delta_* >= 0 =="
NEG=$(awk -F, 'NR>2 && $0 !~ /^#/ && ($2<0||$3<0||$4<0||$5<0) {c++} END{print c+0}' "$TRACE")
echo "  rows with a negative delta = $NEG"
[ "$NEG" = "0" ] && echo "  PASS" || { echo "  FAIL"; fail=1; }

echo "== CHECK4: flip CSV schema (row counts, typed rows, 1-based ids) =="
NELE=$(grep -c '^ele,' "$FL"); NRIV=$(grep -c '^riv,' "$FL")
BADE=$(awk -F, '/^ele,/ && $6+0!=0 {c++} END{print c+0}' "$FL")
BADR=$(awk -F, '/^riv,/ && ($3+0!=0||$4+0!=0||$5+0!=0) {c++} END{print c+0}' "$FL")
EMAX=$(awk -F, '/^ele,/{if($2>m)m=$2}END{print m}' "$FL")
RMAX=$(awk -F, '/^riv,/{if($2>m)m=$2}END{print m}' "$FL")
echo "  ele rows=$NELE (id max=$EMAX)  riv rows=$NRIV (id max=$RMAX)"
echo "  ele-with-nonzero-stage=$BADE  riv-with-nonzero-ele-family=$BADR"
[ "$BADE" = "0" ] && [ "$BADR" = "0" ] && echo "  PASS" || { echo "  FAIL"; fail=1; }

echo "== CHECK5: daily flips_total == sf+us+gw+stage; daily total == per-entity total =="
BADT=$(awk -F, 'NR>2 && $0 !~ /^#/ && $6+0 != $2+$3+$4+$5 {c++} END{print c+0}' "$DL")
DT=$(awk -F, 'NR>2 && $0 !~ /^#/ {s+=$6} END{print s}' "$DL")
PE=$(awk -F, 'NR>2 && $0 !~ /^#/ {s+=$3+$4+$5+$6} END{print s}' "$FL")
echo "  daily rows with inconsistent total=$BADT  daily_total=$DT  per_entity_total=$PE"
[ "$BADT" = "0" ] && [ "$DT" = "$PE" ] && echo "  PASS" || { echo "  FAIL"; fail=1; }

echo "== CHECK6: headers carry required metadata =="
echo "  dt header:   $(head -1 "$TRACE")"
echo "  flip header: $(head -1 "$FL")"

if [ -n "$2" ] && [ -f "$2" ]; then
  echo "== CHECK7: model-output SHA (enabled run) == baseline =="
  cd "$OUT"
  TMP=$(mktemp)
  find . -type f ! -name 'diag_*.csv' ! -name '*.time.csv' | sort | while read -r f; do shasum -a 256 "$f"; done > "$TMP"
  if diff -q "$2" "$TMP" >/dev/null; then echo "  PASS (bitwise-neutral even when enabled)"; else echo "  FAIL"; diff "$2" "$TMP" | head; fail=1; fi
  rm -f "$TMP"
fi

echo ""
[ "$fail" = "0" ] && echo "SELFCHECK: ALL PASS" || echo "SELFCHECK: FAILURES PRESENT"
exit $fail
