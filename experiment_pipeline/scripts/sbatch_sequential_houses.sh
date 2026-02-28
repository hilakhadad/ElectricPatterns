#!/bin/bash
# =============================================================================
# Run all houses with sliding-window job submission.
#
# Manages up to CONCURRENT_HOUSES (default 4) houses simultaneously,
# filling job slots up to the QOS limit. As jobs finish, new ones fill in.
#
# Per house: submit months → when all months done → submit post (M2+reports)
# Multiple houses overlap: house B's months run while house A's post runs.
#
# Avoids QOSMaxSubmitJobPerUserLimit by checking squeue before each submission.
#
# Usage:
#     bash scripts/sbatch_sequential_houses.sh [experiment] [max_jobs] [concurrent_houses]
#
# Examples:
#     bash scripts/sbatch_sequential_houses.sh                              # defaults
#     bash scripts/sbatch_sequential_houses.sh exp015_hole_repair 100 4     # explicit
# =============================================================================

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"

EXPERIMENT_NAME="${1:-exp015_hole_repair}"
MAX_JOBS_OVERRIDE="${2:-}"
CONCURRENT_HOUSES="${3:-4}"
POLL_INTERVAL=20
SAFETY_MARGIN=5

# ---- TS set ONCE ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"
REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"
ALL_EXPERIMENTS_DIR="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments"

mkdir -p "$LOG_DIR" "$EXPERIMENT_OUTPUT" "$REPORTS_DIR"

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

# ---- Auto-detect QOS limit ----
detect_max_submit() {
    local qos
    qos=$(sacctmgr -n -P show assoc where user="$USER" format=qos 2>/dev/null | head -1)
    if [ -n "$qos" ]; then
        local limit
        limit=$(sacctmgr -n -P show qos "$qos" format=MaxSubmitJobsPerUser 2>/dev/null | head -1)
        if [[ "$limit" =~ ^[0-9]+$ ]] && [ "$limit" -gt 0 ]; then
            echo "$limit"
            return
        fi
    fi
    echo "0"
}

if [ -n "$MAX_JOBS_OVERRIDE" ]; then
    MAX_JOBS="$MAX_JOBS_OVERRIDE"
    echo "Job limit: ${MAX_JOBS} (user override)"
else
    MAX_JOBS=$(detect_max_submit)
    if [ "$MAX_JOBS" -eq 0 ]; then
        MAX_JOBS=40
        echo "Job limit: ${MAX_JOBS} (fallback — could not auto-detect)"
    else
        echo "Job limit: ${MAX_JOBS} (auto-detected from QOS)"
    fi
fi

EFFECTIVE_LIMIT=$((MAX_JOBS - SAFETY_MARGIN))
if [ "$EFFECTIVE_LIMIT" -lt 5 ]; then
    EFFECTIVE_LIMIT=5
fi

# ---- Helpers ----
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

count_my_jobs() {
    squeue -u "$USER" -h 2>/dev/null | wc -l
}

# Count running month jobs for a specific house
count_house_month_jobs() {
    local house_id="$1"
    squeue -u "$USER" -h -o "%j" 2>/dev/null | grep -c "^mon_${house_id}_" || echo 0
}

wait_for_job() {
    local job_id="$1"
    while true; do
        local status
        status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        if [ -z "$status" ]; then break; fi
        sleep "$POLL_INTERVAL"
    done
}

# Submit a single month job; prints job ID
submit_month() {
    local house_id="$1"
    local month_idx="$2"
    local n_months="$3"

    local SCRIPT
    SCRIPT=$(mktemp "${LOG_DIR}/sbatch_mon_${house_id}_${month_idx}_XXXXXX.sh")

    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=mon_${house_id}_${month_idx}
#SBATCH --output=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_${month_idx}.out
#SBATCH --error=${LOG_DIR}/month_${house_id}_${TIMESTAMP}_${month_idx}.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:0

echo "House ${house_id} — month ${month_idx} / $((n_months - 1))"
echo "Start: \$(date)"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"

python -u scripts/run_single_month.py \
    --house_id ${house_id} \
    --month_index ${month_idx} \
    --experiment_name ${EXPERIMENT_NAME} \
    --output_path ${EXPERIMENT_OUTPUT} \
    --input_path ${DATA_DIR}

echo "Month ${month_idx} done: \$(date)"
EOF

    local job_id
    job_id=$(sbatch "$SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
    rm -f "$SCRIPT"
    echo "$job_id"
}

# Submit post job (M2 + reports + cleanup); prints job ID
submit_post() {
    local house_id="$1"
    local n_months="$2"

    local SCRIPT
    SCRIPT=$(mktemp "${LOG_DIR}/sbatch_post_${house_id}_XXXXXX.sh")

    cat > "$SCRIPT" << EOF
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

if [ \$EXIT_CODE -eq 0 ]; then STATUS="OK"; else STATUS="FAIL(exit=\${EXIT_CODE})"; fi

echo "${house_id},${n_months},\$START_TIME,\$END_TIME,\$ELAPSED,\$ELAPSED_HUMAN,\$STATUS" >> ${TIMING_FILE}
echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"

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

    local job_id
    job_id=$(sbatch "$SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
    rm -f "$SCRIPT"
    echo "$job_id"
}

# ===========================================================================
echo "============================================================"
echo "SLIDING WINDOW — ${CONCURRENT_HOUSES} houses, fill to limit"
echo "============================================================"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Timestamp:        $TIMESTAMP"
echo "Output dir:       $EXPERIMENT_OUTPUT"
echo "QOS job limit:    $MAX_JOBS"
echo "Effective limit:  $EFFECTIVE_LIMIT (after ${SAFETY_MARGIN} reserved)"
echo "Concurrent houses: $CONCURRENT_HOUSES"
echo "Poll interval:    ${POLL_INTERVAL}s"
echo ""

# Collect all house IDs
ALL_HOUSE_IDS=()
for house_dir in "$DATA_DIR"/*/; do
    house_id=$(basename "$house_dir")
    [[ "$house_id" =~ ^[0-9]+$ ]] || continue
    N_MONTHS=$(count_unique_months "$house_dir" "$house_id")
    [ "$N_MONTHS" -eq 0 ] && continue
    ALL_HOUSE_IDS+=("$house_id")
done

TOTAL_HOUSES=${#ALL_HOUSE_IDS[@]}
echo "Found ${TOTAL_HOUSES} houses to process"
echo ""

# ---- Per-house state (associative arrays) ----
declare -A H_TOTAL       # total months
declare -A H_NEXT        # next month index to submit
declare -A H_POST        # post job ID ("" = not submitted, "done" = finished)

# Queue of houses not yet started
QUEUE_IDX=0              # index into ALL_HOUSE_IDS
ACTIVE_HOUSES=()         # house IDs currently being processed
COMPLETED_COUNT=0
FAILED_HOUSES=()

# ---- Main loop ----
while [ "$COMPLETED_COUNT" -lt "$TOTAL_HOUSES" ]; do

    # --- 1. Fill active house pool ---
    while [ ${#ACTIVE_HOUSES[@]} -lt "$CONCURRENT_HOUSES" ] && [ "$QUEUE_IDX" -lt "$TOTAL_HOUSES" ]; do
        hid="${ALL_HOUSE_IDS[$QUEUE_IDX]}"
        house_dir="${DATA_DIR}/${hid}"
        QUEUE_IDX=$((QUEUE_IDX + 1))

        H_TOTAL[$hid]=$(count_unique_months "$house_dir" "$hid")
        H_NEXT[$hid]=0
        H_POST[$hid]=""

        ACTIVE_HOUSES+=("$hid")
        echo "[$(date '+%H:%M:%S')] Started house ${hid} (${H_TOTAL[$hid]} months) — [${COMPLETED_COUNT}/${TOTAL_HOUSES} done, ${#ACTIVE_HOUSES[@]} active]"
    done

    # --- 2. Submit months — fill available slots across active houses ---
    CURRENT_JOBS=$(count_my_jobs)
    AVAILABLE=$((EFFECTIVE_LIMIT - CURRENT_JOBS))

    if [ "$AVAILABLE" -gt 0 ]; then
        SUBMITTED_TOTAL=0

        for hid in "${ACTIVE_HOUSES[@]}"; do
            [ "$AVAILABLE" -le 0 ] && break
            # Skip houses that finished submitting all months
            [ "${H_NEXT[$hid]}" -ge "${H_TOTAL[$hid]}" ] && continue

            SUBMITTED_HOUSE=0
            while [ "$AVAILABLE" -gt 0 ] && [ "${H_NEXT[$hid]}" -lt "${H_TOTAL[$hid]}" ]; do
                JOB_ID=$(submit_month "$hid" "${H_NEXT[$hid]}" "${H_TOTAL[$hid]}")

                if [[ ! "$JOB_ID" =~ ^[0-9]+$ ]]; then
                    echo "  ERROR submitting month ${H_NEXT[$hid]} for house ${hid}: ${JOB_ID}"
                    # Mark as all submitted to skip further attempts
                    H_NEXT[$hid]=${H_TOTAL[$hid]}
                    FAILED_HOUSES+=("$hid")
                    break
                fi

                H_NEXT[$hid]=$((${H_NEXT[$hid]} + 1))
                AVAILABLE=$((AVAILABLE - 1))
                SUBMITTED_HOUSE=$((SUBMITTED_HOUSE + 1))
                SUBMITTED_TOTAL=$((SUBMITTED_TOTAL + 1))
            done

            if [ "$SUBMITTED_HOUSE" -gt 0 ]; then
                echo "  House ${hid}: submitted ${SUBMITTED_HOUSE} months (${H_NEXT[$hid]}/${H_TOTAL[$hid]})"
            fi
        done
    fi

    # --- 3. Check for houses with all months done → submit post job ---
    for hid in "${ACTIVE_HOUSES[@]}"; do
        # Skip if months not all submitted yet
        [ "${H_NEXT[$hid]}" -lt "${H_TOTAL[$hid]}" ] && continue
        # Skip if post already submitted
        [ -n "${H_POST[$hid]}" ] && continue

        # Check if any month jobs still running for this house
        RUNNING=$(count_house_month_jobs "$hid")
        if [ "$RUNNING" -eq 0 ]; then
            POST_ID=$(submit_post "$hid" "${H_TOTAL[$hid]}")
            if [[ "$POST_ID" =~ ^[0-9]+$ ]]; then
                H_POST[$hid]="$POST_ID"
                echo "[$(date '+%H:%M:%S')] House ${hid}: all months done, post job ${POST_ID}"
            else
                echo "  ERROR submitting post for house ${hid}: ${POST_ID}"
                H_POST[$hid]="done"
                FAILED_HOUSES+=("$hid")
            fi
        fi
    done

    # --- 4. Check for completed houses (post job finished) ---
    NEW_ACTIVE=()
    for hid in "${ACTIVE_HOUSES[@]}"; do
        if [ "${H_POST[$hid]}" = "done" ]; then
            # Already marked done (error case)
            COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
            echo "[$(date '+%H:%M:%S')] House ${hid} complete [${COMPLETED_COUNT}/${TOTAL_HOUSES}]"
            continue
        fi

        if [[ "${H_POST[$hid]}" =~ ^[0-9]+$ ]]; then
            status=$(squeue -j "${H_POST[$hid]}" -h -o "%T" 2>/dev/null)
            if [ -z "$status" ]; then
                COMPLETED_COUNT=$((COMPLETED_COUNT + 1))
                echo "[$(date '+%H:%M:%S')] House ${hid} complete [${COMPLETED_COUNT}/${TOTAL_HOUSES}]"
                continue
            fi
        fi

        NEW_ACTIVE+=("$hid")
    done
    ACTIVE_HOUSES=("${NEW_ACTIVE[@]}")

    # --- 5. Sleep before next poll ---
    if [ "$COMPLETED_COUNT" -lt "$TOTAL_HOUSES" ]; then
        sleep "$POLL_INTERVAL"
    fi
done

echo ""
echo "============================================================"
echo "All ${TOTAL_HOUSES} houses processed"
if [ ${#FAILED_HOUSES[@]} -gt 0 ]; then
    echo "FAILED houses: ${FAILED_HOUSES[*]}"
fi
echo "============================================================"
echo ""

# =============================================================
# Aggregates — after ALL houses
# =============================================================
echo "Submitting aggregate jobs..."

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
AGG_SEG_JOB=$(sbatch "$AGG_SEG_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
rm -f "$AGG_SEG_SCRIPT"
echo "Aggregate segregation:       ${AGG_SEG_JOB}"

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
IDENT_ALL_JOB=$(sbatch "$IDENT_ALL_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
rm -f "$IDENT_ALL_SCRIPT"
echo "Identification ALL:          ${IDENT_ALL_JOB}"

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
COMPARE_JOB=$(sbatch "$COMPARE_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
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
