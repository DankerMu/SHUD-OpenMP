#!/bin/bash
# Build heihe DEM mosaic from 36 ASTER GDEM v3 tiles (N37-N42 × E097-E102).
# heihe basin bbox (verified via sf::st_bbox of domain.shp): 97.08-102.02E, 37.72-42.72N.
# Server-only: ASTER tiles live under <server-data-root>/SpatialData/World/DEM/Aster_GDEM.
# Set REPO and ASTER env vars to your server paths; defaults are placeholder
# strings that fail loudly if unset so credential leaks cannot ride in via
# committed defaults.
set -euo pipefail

REPO=${REPO:-<server-scratch-root>/SHUD-OpenMP}
OUT_DIR=$REPO/SHUD/Basins/heihe_x4/dem
ASTER=${ASTER:-<server-data-root>/SpatialData/World/DEM/Aster_GDEM}
OUT_TIF=$OUT_DIR/heihe_dem_mosaic.tif
VRT=$OUT_DIR/heihe_dem.vrt

mkdir -p "$OUT_DIR"

# Collect 36 tiles (N37-N42 × E097-E102), fail on any missing.
TILES=()
for lat in 37 38 39 40 41 42; do
  for lon in 097 098 099 100 101 102; do
    tile=$ASTER/ASTGTMV003_N${lat}E${lon}_dem.tif
    if [ ! -f "$tile" ]; then
      echo "FATAL: missing DEM tile: $tile" >&2
      exit 1
    fi
    TILES+=("$tile")
  done
done
echo "Found ${#TILES[@]} ASTER GDEM tiles for heihe region."

# Build VRT (virtual mosaic), then translate to compressed GeoTIFF.
echo "Building VRT mosaic → $VRT"
gdalbuildvrt "$VRT" "${TILES[@]}" >/dev/null

echo "Translating VRT → compressed GeoTIFF: $OUT_TIF"
gdal_translate -of GTiff \
  -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=IF_NEEDED \
  "$VRT" "$OUT_TIF"

echo
echo "=== Mosaic info ==="
gdalinfo "$OUT_TIF" | head -15
echo
ls -lh "$OUT_TIF"
