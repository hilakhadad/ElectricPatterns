#!/bin/bash
# =============================================================================
# Submit per-HOUSE SBATCH jobs — with automatic per-house reports and aggregate.
#
# Two modes per house:
#   SEQUENTIAL: Houses listed in COMPLETED_HOUSES_FILE run as a single job
#               (all months + iterations + identification in one process).
#   MONTHLY:    Houses NOT in the file run month-level parallel jobs
#               (SBATCH array for disaggregation + dependent identification job).
#
# Each house job starts with its own pre-analysis report, then runs the
# pipeline, then generates per-house segregation + identification reports.
# No house waits for other houses' pre-analysis to finish.
#
# A final aggregate job (depends on all houses) verifies that all per-house
# reports exist, then generates the combined reports for all three types.
#
# Output structure:
#   {experiment}/reports/
#     ├── house_report.html           — House pre-analysis aggregate
#     ├── house_reports/              — House pre-analysis per-house
#     ├── segregation_report.html     — M1 disaggregation aggregate
#     ├── segregation_reports/        — M1 per-house
#     ├── identification_report.html  — M2 identification aggregate
#     └── identification_reports/     — M2 per-house
#
# Place completed_houses.txt next to this script (same directory).
# Generate from a previous run's timing CSV:
#     awk -F',' 'NR>1 && $7=="OK" {print $1}' house_timing.csv > scripts/completed_houses.txt
#
# Or manually list fast house IDs, one per line.
#
# Usage:
#     bash scripts/sbatch_run_houses.sh
# =============================================================================

# ---- Configuration ----
PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
DATA_DIR="${PROJECT_ROOT}/INPUT/HouseholdData"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"

EXPERIMENT_NAME="exp015_hole_repair"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments/${EXPERIMENT_NAME}"

# ---- Unified reports directory ----
# All reports (house, segregation, identification) go into one directory.
REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"

# ---- Fast houses list ----
COMPLETED_HOUSES_FILE="${PROJECT_ROOT}/experiment_pipeline/completed_houses.txt"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$EXPERIMENT_OUTPUT"
mkdir -p "$REPORTS_DIR"

# Initialize timing CSV
TIMING_FILE="${EXPERIMENT_OUTPUT}/house_timing.csv"
echo "house_id,n_months,start_time,end_time,elapsed_seconds,elapsed_human,status" > "$TIMING_FILE"

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
    echo ""
    echo "  This file is REQUIRED. Without it, no jobs will be submitted."
    echo "  Create it from a previous run's timing CSV:"
    echo "    awk -F',' 'NR>1 && \$7==\"OK\" {print \$1}' house_timing.csv > completed_houses.txt"
    echo ""
    echo "  Or manually list fast house IDs, one per line."
    exit 1
fi

echo "============================================================"
echo "Submitting pipeline jobs (with automatic report generation)"
echo "============================================================"
echo "Experiment:  $EXPERIMENT_NAME"
echo "Data dir:    $DATA_DIR"
echo "Output dir:  $EXPERIMENT_OUTPUT"
echo "Reports dir: $REPORTS_DIR"
echo "Timing log:  $TIMING_FILE"
echo ""

HOUSE_COUNT=0
NORMAL_COUNT=0
MONTHLY_COUNT=0

# Track all final job IDs for the aggregate report dependency
ALL_FINAL_JOBS=()

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

for house_dir in "$DATA_DIR"/*/; do
    house_id=$(basename "$house_dir")

    # Skip non-numeric directory names
    if ! [[ "$house_id" =~ ^[0-9]+$ ]]; then
        continue
    fi

    # Count unique months (deduplicated across pkl/csv)
    N_MONTHS=$(count_unique_months "$house_dir" "$house_id")

    if [ "$N_MONTHS" -eq 0 ]; then
        echo "  House ${house_id}: no monthly files found, skipping"
        continue
    fi

    if [[ -n "${COMPLETED[$house_id]}" ]]; then
        # =============================================================
        # FAST HOUSE — single sequential job:
        #   1. Pre-analysis report (per-house)
        #   2. Pipeline (disaggregation + identification)
        #   3. Per-house segregation + identification reports
        # =============================================================
        JOB_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_house_${house_id}_XXXXXX.sh")

        cat > "$JOB_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=pipe_${house_id}
#SBATCH --output=${LOG_DIR}/house_${house_id}_%j.out
#SBATCH --error=${LOG_DIR}/house_${house_id}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
##SBATCH --time=08:00:00

echo "========================================"
echo "House ${house_id} (${N_MONTHS} months) — SEQUENTIAL"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

# --- Step 0: Per-house pre-analysis report ---
echo "Generating pre-analysis report for house ${house_id}..."
cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --houses ${house_id} \
    --input-dir ${DATA_DIR} \
    --output-dir ${REPORTS_DIR} \
    --publish house \
    2>&1 | tail -3
echo "  Pre-analysis report: exit \$?"
echo ""

# --- Step 1: Pipeline ---
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

flock -x "${TIMING_FILE}.lock" -c "echo '${house_id},${N_MONTHS},'\"\$START_TIME\"','\"\$END_TIME\"','\"\$ELAPSED\"','\"\$ELAPSED_HUMAN\"','\"\$STATUS\"'' >> ${TIMING_FILE}"

echo "========================================"
echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"
echo "========================================"

# --- Step 2: Generate per-house reports (only on success) ---
if [ \$EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Generating reports for house ${house_id}..."
    REPORT_START=\$(date +%s)

    cd "${PROJECT_ROOT}/disaggregation_analysis"
    python scripts/run_dynamic_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish segregation \
        2>&1 | tail -5
    echo "  Segregation report: exit \$?"

    cd "${PROJECT_ROOT}/identification_analysis"
    python scripts/run_identification_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish identification \
        2>&1 | tail -5
    echo "  Identification report: exit \$?"

    REPORT_END=\$(date +%s)
    REPORT_ELAPSED=\$((REPORT_END - REPORT_START))
    echo "  Reports generated in \${REPORT_ELAPSED}s"
fi

echo "End: \$(date)"
EOF

        JOB_ID=$(sbatch "$JOB_SCRIPT" | awk '{print $4}')
        rm -f "$JOB_SCRIPT"

        ALL_FINAL_JOBS+=($JOB_ID)
        echo "  House ${house_id} (${N_MONTHS} months) -> Job ${JOB_ID} [SEQUENTIAL]"
        NORMAL_COUNT=$((NORMAL_COUNT + 1))

    else
        # =============================================================
        # SLOW HOUSE — pre-analysis + month-level parallel + ident + reports
        # =============================================================

        # Step 0: Per-house pre-analysis (no dependencies, runs immediately)
        PRE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_pre_${house_id}_XXXXXX.sh")

        cat > "$PRE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=pre_${house_id}
#SBATCH --output=${LOG_DIR}/pre_${house_id}_%j.out
#SBATCH --error=${LOG_DIR}/pre_${house_id}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
##SBATCH --time=00:30:00

echo "========================================"
echo "House ${house_id} — PRE-ANALYSIS"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --houses ${house_id} \
    --input-dir ${DATA_DIR} \
    --output-dir ${REPORTS_DIR} \
    --publish house

echo "Pre-analysis done for house ${house_id}: \$(date)"
EOF

        PRE_JOB_ID=$(sbatch "$PRE_SCRIPT" | awk '{print $4}')
        rm -f "$PRE_SCRIPT"

        # Step 1: SBATCH array — one task per month (depends on pre-analysis)
        ARRAY_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_months_${house_id}_XXXXXX.sh")

        cat > "$ARRAY_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=mon_${house_id}
#SBATCH --output=${LOG_DIR}/month_${house_id}_%A_%a.out
#SBATCH --error=${LOG_DIR}/month_${house_id}_%A_%a.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --array=0-$((N_MONTHS - 1))
#SBATCH --dependency=afterok:${PRE_JOB_ID}

echo "========================================"
echo "House ${house_id} — MONTHLY mode"
echo "Month index: \$SLURM_ARRAY_TASK_ID / $((N_MONTHS - 1))"
echo "Array job: \$SLURM_ARRAY_JOB_ID[\$SLURM_ARRAY_TASK_ID]"
echo "Node: \$SLURM_JOB_NODELIST"
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

        # Step 2: Identification + reports — runs AFTER all months complete
        IDENT_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_${house_id}_XXXXXX.sh")

        cat > "$IDENT_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ident_${house_id}
#SBATCH --output=${LOG_DIR}/ident_${house_id}_%j.out
#SBATCH --error=${LOG_DIR}/ident_${house_id}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
##SBATCH --time=02:00:00
#SBATCH --dependency=afterok:${ARRAY_JOB_ID}

echo "========================================"
echo "House ${house_id} — IDENTIFICATION + REPORTS (post-monthly)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Depends on array job: ${ARRAY_JOB_ID}"
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

flock -x "${TIMING_FILE}.lock" -c "echo '${house_id},${N_MONTHS},'\"\$START_TIME\"','\"\$END_TIME\"','\"\$ELAPSED\"','\"\$ELAPSED_HUMAN\"','\"\$STATUS\"'' >> ${TIMING_FILE}"

echo "========================================"
echo "House ${house_id}: \$STATUS (\$ELAPSED_HUMAN)"
echo "========================================"

# --- Generate per-house reports (only on success) ---
if [ \$EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Generating reports for house ${house_id}..."
    REPORT_START=\$(date +%s)

    cd "${PROJECT_ROOT}/disaggregation_analysis"
    python scripts/run_dynamic_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish segregation \
        2>&1 | tail -5
    echo "  Segregation report: exit \$?"

    cd "${PROJECT_ROOT}/identification_analysis"
    python scripts/run_identification_report.py \
        --experiment ${EXPERIMENT_OUTPUT} \
        --houses ${house_id} \
        --output-dir ${REPORTS_DIR} \
        --publish identification \
        2>&1 | tail -5
    echo "  Identification report: exit \$?"

    REPORT_END=\$(date +%s)
    REPORT_ELAPSED=\$((REPORT_END - REPORT_START))
    echo "  Reports generated in \${REPORT_ELAPSED}s"
fi

echo "End: \$(date)"
EOF

        IDENT_JOB_ID=$(sbatch "$IDENT_SCRIPT" | awk '{print $4}')
        rm -f "$IDENT_SCRIPT"

        ALL_FINAL_JOBS+=($IDENT_JOB_ID)
        echo "  House ${house_id} (${N_MONTHS} months) -> Pre ${PRE_JOB_ID}, Array ${ARRAY_JOB_ID} [${N_MONTHS} tasks], Ident+Reports ${IDENT_JOB_ID} [MONTHLY]"
        MONTHLY_COUNT=$((MONTHLY_COUNT + 1))
    fi

    HOUSE_COUNT=$((HOUSE_COUNT + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${HOUSE_COUNT} houses total:"
echo "  Sequential (fast): ${NORMAL_COUNT}"
echo "  Monthly (slow):    ${MONTHLY_COUNT}"
echo ""

# =============================================================
# FINAL: 3 separate aggregate jobs — each checks its own directory
# and generates its aggregate independently.
# =============================================================
if [ ${#ALL_FINAL_JOBS[@]} -gt 0 ]; then
    DEPS=$(IFS=:; echo "${ALL_FINAL_JOBS[*]}")

    # --- Aggregate 1: House pre-analysis ---
    AGG_HOUSE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_house_XXXXXX.sh")
    cat > "$AGG_HOUSE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_house
#SBATCH --output=${LOG_DIR}/agg_house_%j.out
#SBATCH --error=${LOG_DIR}/agg_house_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${DEPS}

echo "========================================"
echo "AGGREGATE: House Pre-Analysis"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${HOUSE_COUNT}
ACTUAL=\$(ls ${REPORTS_DIR}/house_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) house reports. Skipping aggregate."
    echo "Run manually: cd ${PROJECT_ROOT}/house_analysis && python scripts/run_analysis.py --input-dir ${DATA_DIR} --output-dir ${REPORTS_DIR} --publish house"
    exit 1
fi

echo "All present. Generating house_report.html..."
cd "${PROJECT_ROOT}/house_analysis"
python scripts/run_analysis.py \
    --input-dir ${DATA_DIR} \
    --output-dir ${REPORTS_DIR} \
    --publish house \
    2>&1 | tail -10
echo "Exit: \$? — End: \$(date)"
EOF
    AGG_HOUSE_JOB=$(sbatch "$AGG_HOUSE_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_HOUSE_SCRIPT"
    echo "Aggregate house pre-analysis:  ${AGG_HOUSE_JOB}"

    # --- Aggregate 2: Segregation (M1) ---
    AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_XXXXXX.sh")
    cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_segregation
#SBATCH --output=${LOG_DIR}/agg_segregation_%j.out
#SBATCH --error=${LOG_DIR}/agg_segregation_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${DEPS}

echo "========================================"
echo "AGGREGATE: Segregation (M1)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${HOUSE_COUNT}
ACTUAL=\$(ls ${REPORTS_DIR}/segregation_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) segregation reports. Skipping aggregate."
    echo "Run manually: cd ${PROJECT_ROOT}/disaggregation_analysis && python scripts/run_dynamic_report.py --experiment ${EXPERIMENT_OUTPUT} --output-dir ${REPORTS_DIR} --publish segregation"
    exit 1
fi

echo "All present. Generating segregation_report.html..."
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
    echo "Aggregate segregation (M1):    ${AGG_SEG_JOB}"

    # --- Aggregate 3: Identification (M2) ---
    AGG_IDENT_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_ident_XXXXXX.sh")
    cat > "$AGG_IDENT_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_identification
#SBATCH --output=${LOG_DIR}/agg_identification_%j.out
#SBATCH --error=${LOG_DIR}/agg_identification_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${DEPS}

echo "========================================"
echo "AGGREGATE: Identification (M2)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${HOUSE_COUNT}
ACTUAL=\$(ls ${REPORTS_DIR}/identification_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) identification reports. Skipping aggregate."
    echo "Run manually: cd ${PROJECT_ROOT}/identification_analysis && python scripts/run_identification_report.py --experiment ${EXPERIMENT_OUTPUT} --output-dir ${REPORTS_DIR} --publish identification"
    exit 1
fi

echo "All present. Generating identification_report.html..."
cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish identification \
    2>&1 | tail -10
echo "Exit: \$? — End: \$(date)"
EOF
    AGG_IDENT_JOB=$(sbatch "$AGG_IDENT_SCRIPT" | awk '{print $4}')
    rm -f "$AGG_IDENT_SCRIPT"
    echo "Aggregate identification (M2): ${AGG_IDENT_JOB}"
fi

echo ""
echo "Output:      ${EXPERIMENT_OUTPUT}"
echo "Reports:     ${REPORTS_DIR}/"
echo "Timing:      ${TIMING_FILE}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Timing:   cat ${TIMING_FILE}  (after jobs complete)"
echo "============================================================"
