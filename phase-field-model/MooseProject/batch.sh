#!/bin/bash
# ==========================================================
# Author: Hao Tang (Nov 2025)
# Purpose: Batch run multiple MOOSE input files efficiently
# ==========================================================

# ===== Config =====
INPUT_DIR="generated_inputs"
LOG_DIR="logs"
MOOSE_EXEC="/home/xtanghao/MooseProject/newt/newt-opt"
NPROC_PER_JOB=4
MAX_PARALLEL=4

# =========================

mkdir -p "$LOG_DIR"

FILES=(${INPUT_DIR}/case_*.i)
TOTAL=${#FILES[@]}

echo "---------------------------------------------"
echo " 🧩 Starting batch run of ${TOTAL} MOOSE cases"
echo "    Using ${NPROC_PER_JOB} cores per job"
echo "    Up to ${MAX_PARALLEL} parallel jobs"
echo "---------------------------------------------"

running_jobs=0

for file in "${FILES[@]}"; do
    base=$(basename "$file" .i)
    logfile="${LOG_DIR}/${base}.out"

    echo "🚀 Launching ${base} ..."
    mpiexec -n ${NPROC_PER_JOB} ${MOOSE_EXEC} -i "$file" > "$logfile" 2>&1 &

    ((running_jobs++))

    if (( running_jobs >= MAX_PARALLEL )); then
        wait -n
        ((running_jobs--))
    fi
done

wait

echo "✅ All MOOSE simulations finished."
echo "Logs saved in: ${LOG_DIR}/"
