#!/usr/bin/env bash
# tools/p8tune/run_maxl_cell.sh
#
# p8tune-spgmr-maxl PR-D — Per-cell driver for the 60-cell maxl sweep.
#
# Drives ONE sweep cell on the server compute node:
#   - sets SHUD_SPGMR_MAXL (PR-C env-var hook) + SHUD_RHS_THREADS / OMP_NUM_THREADS
#     (the canonical Mode C thread-count knob per shud.cpp:152-167; overrides
#     cfg.para NUM_OPENMP) + OMP_PROC_BIND/OMP_PLACES (determinism)
#   - runs `./shud <case>` from Basins/<case>/ (per PR-A keliya pattern)
#   - captures wall via /usr/bin/time -f %e
#   - copies 5 per-cell artifacts to /scratch/.../p8tune-runs/maxl_sweep/<cell>/
#   - verifies provenance log line per PR-C contract
#
# Precondition: SHUD binary at /scratch/.../SHUD/shud must be a Mode C build
#   (`make shud SHUD_ENABLE_OPENMP_RHS=1 SHUD_ENABLE_PROFILE=1`) — orchestrator
#   handles build before sbatch submission.
#
# Usage:
#   bash run_maxl_cell.sh <case> <N> <maxl> <rep>
#
# Args:
#   case  ∈ {heihe, heihe_x4}        (CLAUDE.md project-level cases)
#   N     ∈ {1, 8}                   (OMP threads per spec sweep matrix)
#   maxl  ∈ {5, 10, 15, 20, 30}      (per spec L43 sweep range)
#   rep   ∈ {1, 2, 3}                (3 reps per cell for G8 repeatability)
#
# Output cell dir: /scratch/frd_muziyao/SHUD-OpenMP/.p8tune-runs/maxl_sweep/<case>_N<N>_maxl<maxl>_rep<rep>/
#
# Artifacts (per spec L45 + tasks §4.6):
#   profile_B0.yaml  (from Basins/<case>/output/<case>.out/profile_B0.yaml)
#   cvode_stats.txt  (from Basins/<case>/output/<case>.out/cvode_stats.txt; 15 canonical keys)
#   rivqdown.dat     (from Basins/<case>/output/<case>.out/<case>.rivqdown.dat)
#   wall.sec         (from /usr/bin/time -f %e capture)
#   cell.meta        (case / N / maxl / rep / SHUD_pin / hostname / start_utc / end_utc / exit_code)
#   stdout.log       (tee'd from shud stdout)
#   stderr.log       (tee'd from shud stderr)
#
# Exit code:
#   0  shud exited 0 AND all 5 mandatory artifacts copied AND provenance line OK
#   1  validation OR run OR copy OR provenance FAILED
#
# Slurm三铁律 compliance: all output paths under /scratch/; no /tmp.
#
# References:
#   - spec   : openspec/changes/p8tune-spgmr-maxl/specs/maxl-sweep-verdict/spec.md L37-53
#   - tasks  : openspec/changes/p8tune-spgmr-maxl/tasks.md §4.3, §4.6
#   - PR-C   : SHUD pin 6ce17d6 (env-var hook live; stdout log per tasks §3.3)
#   - PR-A   : docs/p8tune/clean_prec_none_baseline.md §keliya-smoke-anchor (build pattern)
#   - keys   : tools/cvode_stats_diff/canonical_15_keys.yaml

set -euo pipefail

# ----- Argument validation ----------------------------------------------------
if [[ $# -ne 4 ]]; then
    echo "ERROR: usage: $0 <case> <N> <maxl> <rep>" >&2
    exit 1
fi

CASE="$1"
N="$2"
MAXL="$3"
REP="$4"

# Case whitelist (CLAUDE.md project-level cases; only heihe + heihe_x4 enter sweep)
case "${CASE}" in
    heihe|heihe_x4) ;;
    *) echo "ERROR: case must be one of {heihe, heihe_x4} (got: ${CASE})" >&2; exit 1 ;;
esac

# N whitelist (spec L43 sweep matrix; N=4 omitted per design D9)
case "${N}" in
    1|8) ;;
    *) echo "ERROR: N must be one of {1, 8} (got: ${N})" >&2; exit 1 ;;
esac

# maxl whitelist (spec L43; SHUD_SPGMR_MAXL safety constraints in glossary)
case "${MAXL}" in
    5|10|15|20|30) ;;
    *) echo "ERROR: maxl must be one of {5, 10, 15, 20, 30} (got: ${MAXL})" >&2; exit 1 ;;
esac

# rep whitelist (3 reps per cell)
case "${REP}" in
    1|2|3) ;;
    *) echo "ERROR: rep must be one of {1, 2, 3} (got: ${REP})" >&2; exit 1 ;;
esac

# ----- Server-side paths (Slurm三铁律: under /scratch/) -----------------------
SHUD_ROOT="/scratch/frd_muziyao/SHUD-OpenMP"
SHUD_DIR="${SHUD_ROOT}/SHUD"
SOURCE_PROJ_DIR="${SHUD_DIR}/Basins/${CASE}"
SWEEP_ROOT="${SHUD_ROOT}/.p8tune-runs/maxl_sweep"
CELL_DIR="${SWEEP_ROOT}/${CASE}_N${N}_maxl${MAXL}_rep${REP}"

# Per-cell isolated working dir to avoid Basins/<case>/output/ race conditions
# under parallel array execution (60 cells × 11 concurrent). Each cell stages a
# symlink forest of input + its own writable output dir.
WORK_DIR="${CELL_DIR}/work"
PROJ_DIR="${WORK_DIR}"
CASE_OUT="${WORK_DIR}/output/${CASE}.out"

mkdir -p "${CELL_DIR}" "${WORK_DIR}/input/${CASE}" "${WORK_DIR}/output"

# Symlink input files (read-only shared; absolute paths so cd safe)
for f in "${SOURCE_PROJ_DIR}/input/${CASE}"/*; do
    [[ -e "${f}" ]] || continue
    ln -sfn "${f}" "${WORK_DIR}/input/${CASE}/$(basename "${f}")"
done

# forcing/ is referenced via absolute path in cfg.para; no symlink needed
# but ln it just in case any case uses relative reference
if [[ -d "${SOURCE_PROJ_DIR}/forcing" ]]; then
    ln -sfn "${SOURCE_PROJ_DIR}/forcing" "${WORK_DIR}/forcing"
fi

# ----- Provenance metadata (header) -------------------------------------------
SHUD_PIN=$(git -C "${SHUD_DIR}" rev-parse --short HEAD 2>/dev/null || echo "unknown")
HOSTNAME=$(hostname)
START_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ----- Environment (sweep contract per spec L44 + PR-C hook) ------------------
# Thread-count knob precedence (per SHUD shud.cpp:152-167, Mode C build):
#   SHUD_RHS_THREADS is the canonical knob and overrides MD->CS.num_threads
#   from cfg.para NUM_OPENMP. PR-A p8pre 18-cell pattern exported BOTH
#   OMP_NUM_THREADS and SHUD_RHS_THREADS — we mirror that here so the same
#   N parameterization works whether cfg.para's NUM_OPENMP is 1 or 8.
# SHUD_ENABLE_PROFILE / SHUD_ENABLE_OPENMP_RHS are BUILD-TIME flags compiled
# into the binary (Mode C) — exporting them at runtime is harmless but is
# noted here for provenance only.
export SHUD_SPGMR_MAXL="${MAXL}"
export SHUD_RHS_THREADS="${N}"
export OMP_NUM_THREADS="${N}"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

# ----- Copy shud binary into per-cell work dir (per PR-A keliya pattern) ------
if [[ ! -x "${SHUD_DIR}/shud" ]]; then
    echo "ERROR: shud binary not found at ${SHUD_DIR}/shud" >&2
    exit 1
fi
cp -f "${SHUD_DIR}/shud" "${WORK_DIR}/shud"

# (No need to rm CASE_OUT: per-cell work dir is fresh per invocation)

# ----- Run shud with /usr/bin/time wall capture -------------------------------
cd "${WORK_DIR}"

set +e
/usr/bin/time -f "%e" -o "${CELL_DIR}/wall.sec" \
    ./shud "${CASE}" \
    > "${CELL_DIR}/stdout.log" \
    2> "${CELL_DIR}/stderr.log"
EXIT_CODE=$?
set -e

END_UTC=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# ----- Artifact copy (per spec L45) ------------------------------------------
COPY_FAIL=0

if [[ -f "${CASE_OUT}/profile_B0.yaml" ]]; then
    cp "${CASE_OUT}/profile_B0.yaml" "${CELL_DIR}/profile_B0.yaml"
else
    echo "ERROR: profile_B0.yaml missing at ${CASE_OUT}/" >&2
    COPY_FAIL=1
fi

if [[ -f "${CASE_OUT}/cvode_stats.txt" ]]; then
    cp "${CASE_OUT}/cvode_stats.txt" "${CELL_DIR}/cvode_stats.txt"
else
    echo "ERROR: cvode_stats.txt missing at ${CASE_OUT}/" >&2
    COPY_FAIL=1
fi

if [[ -f "${CASE_OUT}/${CASE}.rivqdown.dat" ]]; then
    cp "${CASE_OUT}/${CASE}.rivqdown.dat" "${CELL_DIR}/rivqdown.dat"
else
    echo "ERROR: ${CASE}.rivqdown.dat missing at ${CASE_OUT}/" >&2
    COPY_FAIL=1
fi

if [[ ! -f "${CELL_DIR}/wall.sec" ]]; then
    echo "ERROR: wall.sec capture failed (no file)" >&2
    COPY_FAIL=1
fi

# ----- Provenance verification (PR-C stdout log line per tasks §3.3) ---------
PROV_FAIL=0
EXPECTED_LOG="[CVODE] SPGMR maxl=${MAXL} pretype=PREC_NONE"
if ! grep -qF "${EXPECTED_LOG}" "${CELL_DIR}/stdout.log"; then
    echo "ERROR: provenance line missing from stdout: '${EXPECTED_LOG}'" >&2
    PROV_FAIL=1
fi

# ----- cell.meta (provenance KV) ---------------------------------------------
cat > "${CELL_DIR}/cell.meta" <<EOF
case=${CASE}
N=${N}
maxl=${MAXL}
rep=${REP}
SHUD_pin=${SHUD_PIN}
hostname=${HOSTNAME}
start_utc=${START_UTC}
end_utc=${END_UTC}
exit_code=${EXIT_CODE}
omp_num_threads=${N}
omp_proc_bind=spread
shud_enable_profile=1
shud_enable_openmp_rhs=1
shud_spgmr_maxl=${MAXL}
artifact_dir=${CELL_DIR}
EOF

# ----- Final verdict ----------------------------------------------------------
if [[ "${EXIT_CODE}" -eq 0 && "${COPY_FAIL}" -eq 0 && "${PROV_FAIL}" -eq 0 ]]; then
    echo "CELL OK: ${CASE} N=${N} maxl=${MAXL} rep=${REP} -> ${CELL_DIR}"
    exit 0
else
    echo "CELL FAIL: ${CASE} N=${N} maxl=${MAXL} rep=${REP} (exit=${EXIT_CODE} copy_fail=${COPY_FAIL} prov_fail=${PROV_FAIL})" >&2
    exit 1
fi
