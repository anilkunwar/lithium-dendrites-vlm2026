#!/bin/bash
# ==========================================================
# Author: Hao Tang (Nov 2025)
# Purpose: Batch run multiple MOOSE input files efficiently
#
# Each case runs on 8 MPI ranks with the template's GMRES + ASM(LU) solver (fastest per case in
# the benchmark). Each running case is pinned to its own block of physical cores (8 consecutive
# cores share an L3 on Threadripper); SMT siblings are left idle.
# Every finished case appends one row to $SUMMARY_CSV: wall time, exit code, how the run ended
# (tip reached, Li balance error, retreat, end time, solver failure, ...), converged / failed solves,
# last model time and dt, last tip_x and li_balance_error, and the case parameters from its .json.
# Usage: [MOOSE_EXEC=...] [NPROC_PER_JOB=8] [MAX_PARALLEL=N] [SUMMARY_CSV=...] ./batch.sh
# ==========================================================

# ===== Config =====
INPUT_DIR="${INPUT_DIR:-generated_inputs}"
LOG_DIR="${LOG_DIR:-logs}"
RESULTS_DIR="${RESULTS_DIR:-results}"      # file_base in the template: results/<case>/<case>
SUMMARY_CSV="${SUMMARY_CSV:-run_summary.csv}"
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

# one summary row per finished case (parsed from its log, its CSV output and its .json)
record() {
    local base=$1 rc=$2 start=$3 cpus=$4
    python3 - "$base" "$rc" "$start" "$(date +%s)" "$cpus" "$LOG_DIR/$base.out" \
        "$RESULTS_DIR/$base/$base.csv" "$INPUT_DIR/$base.json" "$SUMMARY_CSV" "$NPROC_PER_JOB" <<'PY'
import csv, json, os, re, sys
base, rc, start, end, cpus, log, out_csv, par_json, summary, nproc = sys.argv[1:]
text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read()) if os.path.exists(log) else ""
steps = re.findall(r"^Time Step (\d+), time = ([0-9.eE+-]+), dt = ([0-9.eE+-]+)", text, re.M)
if "Li balance error exceeded" in text:
    status = "li_balance_error"
elif "tip is within" in text:
    status = "tip_reached"
elif "has retreated" in text:
    status = "retreated"
elif int(rc) == 124:
    status = "timeout"
elif int(rc) == 0 and "Finished Executing" in text:
    status = "end_time"
elif re.search(r"[Tt]imestep cut below|dtmin|Solve failed", text):
    status = "solver_failed"
else:
    status = "error"
row = {"case": base, "status": status, "exit_code": rc,
       "wall_s": int(end) - int(start), "nproc": nproc, "cpus": cpus,
       "converged_solves": text.count("Solve Converged"),
       "failed_solves": text.count("Did NOT Converge"),
       "last_step": steps[-1][0] if steps else "", "last_time_ms": steps[-1][1] if steps else "",
       "last_dt_ms": steps[-1][2] if steps else ""}
if os.path.exists(out_csv):
    rows = list(csv.DictReader(open(out_csv)))
    if rows:
        row["tip_x"] = rows[-1].get("tip_x", "")
        row["li_balance_error"] = rows[-1].get("li_balance_error", "")
if os.path.exists(par_json):
    row.update({f"param_{k}": v for k, v in json.load(open(par_json)).items()})
new = not os.path.exists(summary)
with open(summary, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
print(f"   {base}: {status}, exit {rc}, {row['wall_s']} s, tip_x {row.get('tip_x', '')}, "
      f"li_balance_error {row.get('li_balance_error', '')}")
PY
}

# slot s owns physical cores CORES[s*NPROC_PER_JOB .. (s+1)*NPROC_PER_JOB-1]
declare -A PID_SLOT PID_CASE PID_START PID_CPUS
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
    PID_CASE[$!]=$base
    PID_START[$!]=$(date +%s)
    PID_CPUS[$!]=$cpus
}

# wait for one case, record it and free its slot
reap() {
    local done_pid rc
    wait -n -p done_pid
    rc=$?
    record "${PID_CASE[$done_pid]}" "$rc" "${PID_START[$done_pid]}" "${PID_CPUS[$done_pid]}"
    FREE_SLOTS+=(${PID_SLOT[$done_pid]})
    unset "PID_SLOT[$done_pid]" "PID_CASE[$done_pid]" "PID_START[$done_pid]" "PID_CPUS[$done_pid]"
}

for file in "${FILES[@]}"; do
    (( ${#FREE_SLOTS[@]} == 0 )) && reap
    launch "$file" "${FREE_SLOTS[0]}"
    FREE_SLOTS=("${FREE_SLOTS[@]:1}")
done

while (( ${#PID_SLOT[@]} > 0 )); do
    reap
done

echo "✅ All MOOSE simulations finished."
echo "Logs saved in: ${LOG_DIR}/, run summary in: ${SUMMARY_CSV}"
