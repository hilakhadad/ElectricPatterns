#!/bin/bash
# =============================================================================
# Submit pipeline jobs for normalization experiments (exp016, exp017, exp019).
#
# Differences from sbatch_run_houses.sh:
#   - Loops over 3 normalization experiments automatically
#   - Skips pre-analysis (not normalization-dependent, already done in previous runs)
#   - Skips per-house report generation inside pipeline jobs
#   - Only generates AGGREGATE reports (agg_seg + ident_all) after all houses finish
#
# Each experiment gets its own TS-based output directory.
#
# Usage:
#     bash scripts/sbatch_run_norm_experiments.sh
# =============================================================================

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"

EXPERIMENTS=("exp016_ma_detrend" "exp017_phase_balance" "exp019_combined_norm")

# ---- Fast houses list ----
COMPLETED_HOUSES_FILE="${PROJECT_ROOT}/experiment_pipeline/completed_houses.txt"

mkdir -p "$LOG_DIR"

# Load completed houses into associative array
declare -A COMPLETED
if [ -f "$COMPLETED_HOUSES_FILE" ]; then
    while IFS= read -r hid; do
        [[ -z "$hid" || "$hid" =~ ^[[:space:]]*# ]] && continue
        hid=$(echo "$hid" | xargs)
        COMPLETED["$hid"]=1
    done < "$COMPLETED_HOUSES_FILE"
    echo "Loaded ${#COMPLETED[@]} fast houses from $COMPLETED_HOUSES_FILE"
else
    echo "ERROR: completed_houses.txt not found at:"
    echo "  $COMPLETED_HOUSES_FILE"
    exit 1
fi

# Helper: count unique months for a house directory
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

# =============================================================================
# Loop over experiments
# =============================================================================
for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    EXPERIMENT_OUTPUT="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"
    REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"
    TIMING_FILE="${EXPERIMENT_OUTPUT}/house_timing.csv"

    mkdir -p "$EXPERIMENT_OUTPUT"
    mkdir -p "$REPORTS_DIR"
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

    echo ""
    echo "============================================================"
    echo "Submitting: ${EXPERIMENT_NAME}"
    echo "============================================================"
    echo "Output:  ${EXPERIMENT_OUTPUT}"
    echo "Reports: ${REPORTS_DIR}"
    echo ""

    HOUSE_COUNT=0
    NORMAL_COUNT=0
    MONTHLY_COUNT=0
    ALL_FINAL_JOBS=()

    for house_dir in "$DATA_DIR"/*/; do
        house_id=$(basename "$house_dir")

        # Skip non-numeric directory names
        if ! [[ "$house_id" =~ ^[0-9]+$ ]]; then
            continue
        fi

        N_MONTHS=$(count_unique_months "$house_dir" "$house_id")
        if [ "$N_MONTHS" -eq 0 ]; then
            echo "  House ${house_id}: no monthly files found, skipping"
            continue
        fi

        if [[ -n "${COMPLETED[$house_id]}" ]]; then
            # =============================================================
            # FAST HOUSE — single sequential job (pipeline only, no reports)
            # =============================================================
            JOB_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_house_${house_id}_XXXXXX.sh")

            cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=pipe_${house_id}
#SBATCH --output=${LOG_DIR}/house_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/house_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
##SBATCH --time=08:00:00

echo "========================================"
echo "House ${house_id} (${N_MONTHS} months) — SEQUENTIAL"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

START_EPOCH=\$(date +%s)
START_TIME=\$(date '+%Y-%m-%d %H:%M:%S')

python -u scripts/test_single_house.py --house_id ${house_id} --experiment_name ${EXPERIMENT_NAME} --output_path ${EXPERIMENT_OUTPUT} --skip_visualization

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

echo "========================================"
echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"
echo "========================================"
echo "End: \$(date)"
EOF

            JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
            rm -f "$JOB_SCRIPT"

            ALL_FINAL_JOBS+=($JOB_ID)
            echo "  House ${house_id} (${N_MONTHS} months) -> Pipe ${JOB_ID} [SEQUENTIAL]"
            NORMAL_COUNT=$((NORMAL_COUNT + 1))

        else
            # =============================================================
            # SLOW HOUSE — month-level parallel + M2 (no per-house reports)
            # =============================================================

            # Step 1: SBATCH array — one task per month
            ARRAY_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_months_${house_id}_XXXXXX.sh")

            cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=mon_${house_id}
#SBATCH --output=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_%A_%a.out
#SBATCH --error=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --array=0-$((N_MONTHS - 1))

echo "========================================"
echo "House ${house_id} — MONTHLY mode"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Month index: \$SLURM_ARRAY_TASK_ID / $((N_MONTHS - 1))"
echo "Array job: \$SLURM_ARRAY_JOB_ID[\$SLURM_ARRAY_TASK_ID]"
echo "Node: \$SLURM_JOB_NODELIST"
echo "TS: ${TIMESTAMP}"
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

            ARRAY_JOB_ID=$(sbatch "$ARRAY_SCRIPT" | awk '{print $4}')
            rm -f "$ARRAY_SCRIPT"

            # Step 2: M2 (identification only, no reports) — runs AFTER all months
            POST_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_post_${house_id}_XXXXXX.sh")

            cat > "$POST_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=post_${house_id}
#SBATCH --output=${LOG_DIR}/post_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/post_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
##SBATCH --time=02:00:00
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}

echo "========================================"
echo "House ${house_id} — M2 (post-monthly)"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Depends on array job: ${ARRAY_JOB_ID}"
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
    STATUS="OK-MONTHLY"
else
    STATUS="FAIL-MONTHLY(exit=\${EXIT_CODE})"
fi

echo "${house_id},${N_MONTHS},\$START_TIME,\$END_TIME,\$ELAPSED,\$ELAPSED_HUMAN,\$STATUS" >> ${TIMING_FILE}

echo "========================================"
echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"
echo "========================================"
echo "End: \$(date)"
EOF

            POST_JOB_ID=$(sbatch "$POST_SCRIPT" | awk '{print $4}')
            rm -f "$POST_SCRIPT"

            ALL_FINAL_JOBS+=($POST_JOB_ID)
            echo "  House ${house_id} (${N_MONTHS} months) -> Array ${ARRAY_JOB_ID} [${N_MONTHS} tasks], Post ${POST_JOB_ID} [MONTHLY]"
            MONTHLY_COUNT=$((MONTHLY_COUNT + 1))
        fi

        HOUSE_COUNT=$((HOUSE_COUNT + 1))
    done

    echo ""
    echo "Submitted ${HOUSE_COUNT} houses: ${NORMAL_COUNT} sequential, ${MONTHLY_COUNT} monthly"

    # =============================================================
    # AGGREGATE REPORTS (only if we submitted jobs)
    # =============================================================
    if [ ${#ALL_FINAL_JOBS[@]} -gt 0 ]; then
        FINAL_DEPS=$(IFS=:; echo "${ALL_FINAL_JOBS[*]}")

        # --- Aggregate segregation report ---
        AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_XXXXXX.sh")
        cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_seg_${EXPERIMENT_NAME}
#SBATCH --output=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${FINAL_DEPS}

echo "========================================"
echo "AGGREGATE: Segregation (M1)"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

echo "Generating segregation_report.html..."
cd "${PROJECT_ROOT}/disaggregation_analysis"
python scripts/run_dynamic_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish segregation \
    2>&1 | tail -10
echo "Exit: \$? — End: \$(date)"
EOF
        AGG_SEG_JOB=$(sbatch "$AGG_SEG_SCRIPT" | awk '{print $4}')
        rm -f "$AGG_SEG_SCRIPT"
        echo "Aggregate segregation:       ${AGG_SEG_JOB} (afterany all pipelines)"

        # --- Identification report for ALL houses ---
        IDENT_ALL_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_all_XXXXXX.sh")
        cat > "$IDENT_ALL_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ident_${EXPERIMENT_NAME}
#SBATCH --output=${LOG_DIR}/ident_all_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/ident_all_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${AGG_SEG_JOB}

echo "========================================"
echo "IDENTIFICATION REPORT — ALL HOUSES"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

echo "Generating identification reports (per-house + cross-house + aggregate)..."
cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish identification \
    2>&1 | tail -10
echo "Exit: \$? — End: \$(date)"
EOF
        IDENT_ALL_JOB=$(sbatch "$IDENT_ALL_SCRIPT" | awk '{print $4}')
        rm -f "$IDENT_ALL_SCRIPT"
        echo "Identification (ALL houses): ${IDENT_ALL_JOB} (afterany agg_seg — runs LAST)"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "${EXPERIMENT_NAME} submitted."
    echo "Output:  ${EXPERIMENT_OUTPUT}"
    echo "Reports: ${REPORTS_DIR}"
    echo "Timing:  ${TIMING_FILE}"
    echo "------------------------------------------------------------"

    # Small delay between experiments to ensure different timestamps
    sleep 2
done

echo ""
echo "============================================================"
echo "ALL NORMALIZATION EXPERIMENTS SUBMITTED"
echo "============================================================"
echo "Experiments: ${EXPERIMENTS[*]}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "============================================================"
