#!/bin/bash
# =============================================================================
# Submit houses SEQUENTIALLY — one house at a time, months in parallel batches.
#
# Avoids QOSMaxSubmitJobPerUserLimit by:
#   1. Processing one house at a time (wait for completion before next house)
#   2. Splitting months into mini-batches (e.g., 12 at a time) instead of
#      one big array job — so at most BATCH_SIZE jobs in the queue at once.
#
# Flow per house:
#   Batch 1: months 0-11 (parallel) → WAIT
#   Batch 2: months 12-23 (parallel) → WAIT
#   ...
#   M2 + reports + cleanup → WAIT
#   → next house
#
# After ALL houses: aggregate reports → identification ALL → comparison
#
# Usage:
#     bash scripts/sbatch_sequential_houses.sh [experiment_name]
#
# Examples:
#     bash scripts/sbatch_sequential_houses.sh                    # default: exp015_hole_repair
#     bash scripts/sbatch_sequential_houses.sh exp015_hole_repair
# =============================================================================

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"

EXPERIMENT_NAME="${1:-exp015_hole_repair}"
BATCH_SIZE=12        # Max months to submit at once (= max jobs in queue)
POLL_INTERVAL=30     # Seconds between squeue polls

# ---- TS set ONCE ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"
REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"
ALL_EXPERIMENTS_DIR="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$EXPERIMENT_OUTPUT"
mkdir -p "$REPORTS_DIR"

# Initialize timing CSV
TIMING_FILE="${EXPERIMENT_OUTPUT}/house_timing.csv"
echo "house_id,n_months,start_time,end_time,elapsed_seconds,elapsed_human,status" > "$TIMING_FILE"

# Pre-create experiment metadata
cd "${PROJECT_ROOT}/experiment_pipeline"
python -c "
import sys; sys.path.insert(0, 'src')
from core.config import get_experiment, save_experiment_metadata
exp = get_experiment('${EXPERIMENT_NAME}')
save_experiment_metadata(exp, '${EXPERIMENT_OUTPUT}')
print('Experiment metadata saved to ${EXPERIMENT_OUTPUT}')
"
cd "${PROJECT_ROOT}"

# Helper: count unique months
count_unique_months() {
    local house_dir="$1"
    local house_id="$2"
    local -A seen_months
    local count=0
    for f in "$house_dir"/${house_id}_[0-9][0-9]_[0-9][0-9][0-9][0-9].pkl \
             "$house_dir"/${house_id}_[0-9][0-9]_[0-9][0-9][0-9][0-9].csv; do
        [ -f "$f" ] || continue
        local month_str
        month_str=$(basename "$f" | sed "s/^${house_id}_//" | sed 's/\..*//')
        if [[ -z "${seen_months[$month_str]}" ]]; then
            seen_months["$month_str"]=1
            count=$((count + 1))
        fi
    done
    echo "$count"
}

# Helper: wait for a SLURM job to finish (poll squeue)
wait_for_job() {
    local job_id="$1"
    while true; do
        local status
        status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        if [ -z "$status" ]; then
            break
        fi
        sleep "$POLL_INTERVAL"
    done
}

echo "============================================================"
echo "SEQUENTIAL HOUSE SUBMISSION (months in batches of ${BATCH_SIZE})"
echo "============================================================"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Timestamp:   $TIMESTAMP"
echo "Output dir:  $EXPERIMENT_OUTPUT"
echo "Batch size:  $BATCH_SIZE months per submission"
echo "Poll interval: ${POLL_INTERVAL}s"
echo ""

# Collect all house IDs
HOUSE_IDS=()
for house_dir in "$DATA_DIR"/*/; do
    house_id=$(basename "$house_dir")
    [[ "$house_id" =~ ^[0-9]+$ ]] || continue
    N_MONTHS=$(count_unique_months "$house_dir" "$house_id")
    [ "$N_MONTHS" -eq 0 ] && continue
    HOUSE_IDS+=("$house_id")
done

TOTAL_HOUSES=${#HOUSE_IDS[@]}
echo "Found ${TOTAL_HOUSES} houses to process"
echo ""

HOUSE_COUNT=0
FAILED_HOUSES=()

for house_id in "${HOUSE_IDS[@]}"; do
    house_dir="${DATA_DIR}/${house_id}"
    N_MONTHS=$(count_unique_months "$house_dir" "$house_id")
    HOUSE_COUNT=$((HOUSE_COUNT + 1))

    echo "------------------------------------------------------------"
    echo "[${HOUSE_COUNT}/${TOTAL_HOUSES}] House ${house_id} (${N_MONTHS} months) — $(date '+%H:%M:%S')"
    echo "------------------------------------------------------------"

    # =================================================================
    # Step 1: Submit months in mini-batches
    # =================================================================
    BATCH_START=0
    BATCH_NUM=0
    HOUSE_FAILED=false

    while [ "$BATCH_START" -lt "$N_MONTHS" ]; do
        BATCH_END=$((BATCH_START + BATCH_SIZE - 1))
        if [ "$BATCH_END" -ge "$N_MONTHS" ]; then
            BATCH_END=$((N_MONTHS - 1))
        fi
        BATCH_NUM=$((BATCH_NUM + 1))
        BATCH_COUNT=$((BATCH_END - BATCH_START + 1))

        ARRAY_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_seq_months_${house_id}_b${BATCH_NUM}_XXXXXX.sh")

        cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=mon_${house_id}
#SBATCH --output=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_%A_%a.out
#SBATCH --error=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --array=${BATCH_START}-${BATCH_END}

echo "========================================"
echo "House ${house_id} — month \$SLURM_ARRAY_TASK_ID / $((N_MONTHS - 1))"
echo "Batch ${BATCH_NUM}: indices ${BATCH_START}-${BATCH_END}"
echo "Array job: \$SLURM_ARRAY_JOB_ID[\$SLURM_ARRAY_TASK_ID]"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

python -u scripts/run_single_month.py \
    --house_id ${house_id} \
    --month_index \$SLURM_ARRAY_TASK_ID \
    --experiment_name ${EXPERIMENT_NAME} \
    --output_path ${EXPERIMENT_OUTPUT} \
    --input_path ${DATA_DIR}

echo "Month \$SLURM_ARRAY_TASK_ID done: \$(date)"
EOF

        ARRAY_JOB_ID=$(sbatch "$ARRAY_SCRIPT" 2>&1 | awk '{print $4}')
        rm -f "$ARRAY_SCRIPT"

        if [[ ! "$ARRAY_JOB_ID" =~ ^[0-9]+$ ]]; then
            echo "  ERROR: Failed to submit batch ${BATCH_NUM} for house ${house_id}: ${ARRAY_JOB_ID}"
            HOUSE_FAILED=true
            break
        fi

        echo "  Batch ${BATCH_NUM}: months ${BATCH_START}-${BATCH_END} (${BATCH_COUNT} tasks) -> job ${ARRAY_JOB_ID}"

        # Wait for this batch to finish before submitting next
        wait_for_job "$ARRAY_JOB_ID"

        BATCH_START=$((BATCH_END + 1))
    done

    if [ "$HOUSE_FAILED" = true ]; then
        FAILED_HOUSES+=("$house_id")
        echo "  SKIPPING house ${house_id} due to submission error"
        continue
    fi

    echo "  All ${N_MONTHS} months done — $(date '+%H:%M:%S')"

    # =================================================================
    # Step 2: Submit M2 + reports + cleanup (no dependency needed — months done)
    # =================================================================
    POST_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_seq_post_${house_id}_XXXXXX.sh")

    cat > "$POST_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=post_${house_id}
#SBATCH --output=${LOG_DIR}/post_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/post_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0

echo "========================================"
echo "House ${house_id} — M2 + REPORTS"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

START_EPOCH=\$(date +%s)
START_TIME=\$(date '+%Y-%m-%d %H:%M:%S')

python -u scripts/run_identification.py \
    --experiment_dir ${EXPERIMENT_OUTPUT} \
    --house_id ${house_id}

EXIT_CODE=\$?

END_EPOCH=\$(date +%s)
END_TIME=\$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=\$((END_EPOCH - START_EPOCH))
HOURS=\$((ELAPSED / 3600))
MINS=\$(( (ELAPSED % 3600) / 60 ))
SECS=\$((ELAPSED % 60))
ELAPSED_HUMAN="\${HOURS}h \${MINS}m \${SECS}s"

if [ \$EXIT_CODE -eq 0 ]; then
    STATUS="OK"
else
    STATUS="FAIL(exit=\${EXIT_CODE})"
fi

echo "${house_id},${N_MONTHS},\$START_TIME,\$END_TIME,\$ELAPSED,\$ELAPSED_HUMAN,\$STATUS" >> ${TIMING_FILE}

echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"

# --- Reports ---
if [ \$EXIT_CODE -eq 0 ]; then
    echo "Generating reports for house ${house_id}..."

    cd "${PROJECT_ROOT}/disaggregation_analysis"
    python scripts/run_dynamic_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish segregation \
        2>&1
    echo "  Segregation report: exit \$?"

    cd "${PROJECT_ROOT}/identification_analysis"
    python scripts/run_identification_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish identification \
        2>&1
    echo "  Identification report: exit \$?"

    # --- Cleanup ---
    echo "Cleaning up pkl files for house ${house_id}..."
    cd "${PROJECT_ROOT}/experiment_pipeline"
    python -c "
import sys; sys.path.insert(0, 'src')
from identification.cleanup import cleanup_after_reports
from pathlib import Path
r = cleanup_after_reports(Path('${EXPERIMENT_OUTPUT}'), '${house_id}')
print(f'  Cleanup: {r[\"dirs_deleted\"]} directories removed')
"
fi

echo "End: \$(date)"
EOF

    POST_JOB_ID=$(sbatch "$POST_SCRIPT" 2>&1 | awk '{print $4}')
    rm -f "$POST_SCRIPT"

    if [[ ! "$POST_JOB_ID" =~ ^[0-9]+$ ]]; then
        echo "  ERROR: Failed to submit post job for house ${house_id}: ${POST_JOB_ID}"
        FAILED_HOUSES+=("$house_id")
        continue
    fi

    echo "  Post job: ${POST_JOB_ID}"
    wait_for_job "$POST_JOB_ID"
    echo "  House ${house_id} complete — $(date '+%H:%M:%S')"
done

echo ""
echo "============================================================"
echo "All ${HOUSE_COUNT} houses processed"
if [ ${#FAILED_HOUSES[@]} -gt 0 ]; then
    echo "FAILED houses: ${FAILED_HOUSES[*]}"
fi
echo "============================================================"
echo ""

# =============================================================
# Aggregates — after ALL houses (submit with dependency chain)
# =============================================================
echo "Submitting aggregate jobs..."

# Aggregate segregation
AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_XXXXXX.sh")
cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_seg
#SBATCH --output=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0

echo "AGGREGATE: Segregation — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/disaggregation_analysis"
python scripts/run_dynamic_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish segregation \
    --aggregate-only \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
AGG_SEG_JOB=$(sbatch "$AGG_SEG_SCRIPT" 2>&1 | awk '{print $4}')
rm -f "$AGG_SEG_SCRIPT"
echo "Aggregate segregation:       ${AGG_SEG_JOB}"

# Identification ALL (depends on agg_seg)
IDENT_ALL_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_all_XXXXXX.sh")
cat > "$IDENT_ALL_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ident_all
#SBATCH --output=${LOG_DIR}/ident_all_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/ident_all_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${AGG_SEG_JOB}

echo "IDENTIFICATION ALL HOUSES — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish identification \
    --aggregate-only \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
IDENT_ALL_JOB=$(sbatch "$IDENT_ALL_SCRIPT" 2>&1 | awk '{print $4}')
rm -f "$IDENT_ALL_SCRIPT"
echo "Identification ALL:          ${IDENT_ALL_JOB}"

# Cross-experiment comparison (depends on ident_all)
COMPARE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_compare_XXXXXX.sh")
cat > "$COMPARE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=${LOG_DIR}/compare_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/compare_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${IDENT_ALL_JOB}

echo "CROSS-EXPERIMENT COMPARISON — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"
python scripts/compare_experiments.py \
    --scan \
    --scan-dir ${ALL_EXPERIMENTS_DIR} \
    --output-dir ${REPORTS_DIR}/comparison \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
COMPARE_JOB=$(sbatch "$COMPARE_SCRIPT" 2>&1 | awk '{print $4}')
rm -f "$COMPARE_SCRIPT"
echo "Cross-experiment comparison:  ${COMPARE_JOB}"

echo ""
echo "============================================================"
echo "Output:   ${EXPERIMENT_OUTPUT}"
echo "Reports:  ${REPORTS_DIR}/"
echo "Timing:   ${TIMING_FILE}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "============================================================"
