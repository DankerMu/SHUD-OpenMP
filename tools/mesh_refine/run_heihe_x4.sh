#!/bin/bash
# Drive AutoSHUD pipeline (Step1/Step2/Step3) for heihe_x4 mesh + input generation.
# All.R itself only does `source('StepN...')` and won't propagate argv; we invoke each
# step Rscript individually with the cfg path so commandArgs(TRUE)[1] resolves correctly.
# Pre-reqs: build_dem_mosaic.sh has already produced the DEM mosaic.
set -euo pipefail

# Set REPO and AUTOSHUD env vars to your server paths; defaults are
# placeholder strings — credential leaks cannot ride in via committed
# defaults. The .autoshud.txt template ships with the same placeholders;
# this script sed-substitutes them with $REPO at runtime before invoking
# AutoSHUD's Rscript steps.
REPO=${REPO:-<server-scratch-root>/SHUD-OpenMP}
CFG_TEMPLATE=$REPO/tools/mesh_refine/heihe_x4.autoshud.txt
AUTOSHUD=${AUTOSHUD:-<server-scratch-root>/NWM/AutoSHUD}
NWM_DATA=${NWM_DATA:-<server-data-root>/nwm/Basins}
LDAS_DATA=${LDAS_DATA:-<server-data-root>/ForcingData/CMFD2.0/Data_forcing_03hr_010deg}
SOIL_DATA=${SOIL_DATA:-<server-data-root>/SpatialData/World/Soil/HWSD/HWSD_RASTER}
LANDUSE_DATA=${LANDUSE_DATA:-<server-data-root>/SpatialData/World/Landuse/USGS_LCI/LCType.tif}
CASE_DIR=$REPO/SHUD/Basins/heihe_x4
CFG_RUNTIME=$CASE_DIR/heihe_x4.autoshud.runtime.txt
LOG=$CASE_DIR/autoshud_run.log

if [ ! -f "$CFG_TEMPLATE" ]; then
  echo "FATAL: missing AutoSHUD cfg template: $CFG_TEMPLATE" >&2
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

# Substitute placeholders in the cfg template with resolved env paths.
# AutoSHUD reads the cfg verbatim and does not expand env vars itself,
# so this script does the expansion once into a runtime file.
sed -e "s|<server-scratch-root>/SHUD-OpenMP|${REPO}|g" \
    -e "s|<server-data-root>/nwm/Basins|${NWM_DATA}|g" \
    -e "s|<server-data-root>/ForcingData/CMFD2.0/Data_forcing_03hr_010deg|${LDAS_DATA}|g" \
    -e "s|<server-data-root>/SpatialData/World/Soil/HWSD/HWSD_RASTER|${SOIL_DATA}|g" \
    -e "s|<server-data-root>/SpatialData/World/Landuse/USGS_LCI/LCType.tif|${LANDUSE_DATA}|g" \
    "$CFG_TEMPLATE" > "$CFG_RUNTIME"

cd "$AUTOSHUD"
echo "=== heihe_x4 AutoSHUD pipeline ===" | tee "$LOG"
date | tee -a "$LOG"
echo "cfg_template=$CFG_TEMPLATE" | tee -a "$LOG"
echo "cfg_runtime=$CFG_RUNTIME" | tee -a "$LOG"

for step in Step1_RawDataProcessng Step2_DataSubset Step3_BuidModel; do
  echo "--- $step ---" | tee -a "$LOG"
  date +%H:%M:%S | tee -a "$LOG"
  if ! Rscript "$step.R" "$CFG_RUNTIME" 2>&1 | tee -a "$LOG"; then
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
