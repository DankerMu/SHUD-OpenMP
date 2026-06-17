#!/bin/bash
# Drive AutoSHUD pipeline (Step1/Step2/Step3) for heihe_x4 mesh + input generation.
# All.R itself only does `source('StepN...')` and won't propagate argv; we invoke each
# step Rscript individually with the cfg path so commandArgs(TRUE)[1] resolves correctly.
# Pre-reqs: build_dem_mosaic.sh has already produced the DEM mosaic.
set -euo pipefail

REPO=${REPO:-/scratch/frd_muziyao/SHUD-OpenMP}
CFG=$REPO/tools/mesh_refine/heihe_x4.autoshud.txt
AUTOSHUD=${AUTOSHUD:-/scratch/frd_muziyao/NWM/AutoSHUD}
CASE_DIR=$REPO/SHUD/Basins/heihe_x4
LOG=$CASE_DIR/autoshud_run.log

if [ ! -f "$CFG" ]; then
  echo "FATAL: missing AutoSHUD cfg: $CFG" >&2
  exit 1
fi
if [ ! -f "$CASE_DIR/dem/heihe_dem_mosaic.tif" ]; then
  echo "FATAL: missing DEM mosaic. Run build_dem_mosaic.sh first." >&2
  exit 1
fi
if [ ! -f "$AUTOSHUD/Step1_RawDataProcessng.R" ]; then
  echo "FATAL: AutoSHUD not at $AUTOSHUD" >&2
  exit 1
fi

mkdir -p "$CASE_DIR"

cd "$AUTOSHUD"
echo "=== heihe_x4 AutoSHUD pipeline ===" | tee "$LOG"
date | tee -a "$LOG"
echo "cfg=$CFG" | tee -a "$LOG"

for step in Step1_RawDataProcessng Step2_DataSubset Step3_BuidModel; do
  echo "--- $step ---" | tee -a "$LOG"
  date +%H:%M:%S | tee -a "$LOG"
  if ! Rscript "$step.R" "$CFG" 2>&1 | tee -a "$LOG"; then
    echo "FATAL: $step failed. See $LOG" >&2
    exit 1
  fi
done

echo "=== Done ===" | tee -a "$LOG"
date | tee -a "$LOG"

echo
echo "=== Output check ==="
ls -l "$CASE_DIR/input/heihe_x4/" 2>/dev/null || echo "WARN: input/heihe_x4/ not populated"
echo
echo "=== sp.mesh header (verify NumEle) ==="
head -2 "$CASE_DIR/input/heihe_x4/heihe_x4.sp.mesh" 2>/dev/null || echo "WARN: sp.mesh missing"
