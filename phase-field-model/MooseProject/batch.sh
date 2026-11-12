#!/bin/bash
# ==========================================================
# Author: Hao Tang (Nov 2025)
# Purpose: Batch run multiple MOOSE input files efficiently
#
# Each case runs on 8 MPI ranks with the template's GMRES + ASM(LU) solver (fastest per case in
# the benchmark). Each running case is pinned to its own block of physical cores (8 consecutive
# cores share an L3 on Threadripper); SMT siblings are left idle.
# Usage: [MOOSE_EXEC=...] [NPROC_PER_JOB=8] [MAX_PARALLEL=N] ./batch.sh
# ==========================================================

# ===== Config =====
INPUT_DIR="${INPUT_DIR:-generated_inputs}"
LOG_DIR="${LOG_DIR:-logs}"
MOOSE_EXEC="${MOOSE_EXEC:-/home/lithium/HaoResearch/li_electrodeposition/li_electrodeposition-opt}"
CONDA_ENV="${CONDA_ENV:-moose}"
NPROC_PER_JOB="${NPROC_PER_JOB:-8}"
RESERVED_CORES="${RESERVED_CORES:-2}"   # physical cores left for the system
# =========================

# one logical CPU per physical core (first SMT sibling of each core)
CORES=($(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | awk -F, '!seen[$2","$3]++ {print $1}'))
NCORES=$(( ${#CORES[@]} - RESERVED_CORES ))
MAX_SLOTS=$(( NCORES / NPROC_PER_JOB ))
MAX_PARALLEL="${MAX_PARALLEL:-$MAX_SLOTS}"
(( MAX_PARALLEL > MAX_SLOTS )) && MAX_PARALLEL=$MAX_SLOTS
if (( MAX_PARALLEL < 1 )); then
    echo "NPROC_PER_JOB=${NPROC_PER_JOB} exceeds the ${NCORES} available physical cores"; exit 1
fi

[ -x "$MOOSE_EXEC" ] || { echo "MOOSE executable not found: $MOOSE_EXEC"; exit 1; }
if (( NPROC_PER_JOB > 1 )); then
    eval "$(conda shell.bash hook)" && conda activate "$CONDA_ENV" || { echo "conda env $CONDA_ENV not found"; exit 1; }
fi

# BLAS/OpenMP threads would oversubscribe the pinned cores
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p "$LOG_DIR"

FILES=(${INPUT_DIR}/case_*.i)
[ -e "${FILES[0]}" ] || { echo "No input files in ${INPUT_DIR}/"; exit 1; }
TOTAL=${#FILES[@]}

echo "---------------------------------------------"
echo " 🧩 Starting batch run of ${TOTAL} MOOSE cases"
echo "    Using ${NPROC_PER_JOB} cores per job"
echo "    Up to ${MAX_PARALLEL} parallel jobs on ${NCORES} physical cores"
echo "---------------------------------------------"

# slot s owns physical cores CORES[s*NPROC_PER_JOB .. (s+1)*NPROC_PER_JOB-1]
declare -A PID_SLOT
FREE_SLOTS=($(seq 0 $((MAX_PARALLEL - 1))))

launch() {
    local file=$1 slot=$2
    local base=$(basename "$file" .i)
    local logfile="${LOG_DIR}/${base}.out"
    local cpus=$(IFS=,; echo "${CORES[*]:$((slot * NPROC_PER_JOB)):$NPROC_PER_JOB}")

    echo "🚀 Launching ${base} on CPU ${cpus} ..."
    if (( NPROC_PER_JOB == 1 )); then
        taskset -c "$cpus" "$MOOSE_EXEC" -i "$file" > "$logfile" 2>&1 &
    else
        mpiexec -n "$NPROC_PER_JOB" -bind-to "user:${cpus}" \
            "$MOOSE_EXEC" -i "$file" > "$logfile" 2>&1 &
    fi
    PID_SLOT[$!]=$slot
}

for file in "${FILES[@]}"; do
    if (( ${#FREE_SLOTS[@]} == 0 )); then
        wait -n -p done_pid
        FREE_SLOTS+=(${PID_SLOT[$done_pid]})
        unset "PID_SLOT[$done_pid]"
    fi
    launch "$file" "${FREE_SLOTS[0]}"
    FREE_SLOTS=("${FREE_SLOTS[@]:1}")
done

wait

echo "✅ All MOOSE simulations finished."
echo "Logs saved in: ${LOG_DIR}/"
