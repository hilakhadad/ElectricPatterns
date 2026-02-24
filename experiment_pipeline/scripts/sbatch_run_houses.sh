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
# Execution order:
#   Phase 1 — Per-house (all houses in parallel):
#     Every house:      pre-analysis report (separate fast job)
#     Sequential house: pipeline (M1+M2) → segregation + identification reports
#     Monthly house:    month array (M1) → M2 + segregation + identification reports
#
#   Phase 2 — Aggregates:
#     agg_pre:   aggregate pre-analysis    (afterany ALL pre-analysis jobs — runs EARLY)
#     agg_seg:   aggregate segregation     (afterany ALL pipeline jobs + agg_pre)
#     ident_all: identification ALL houses (afterany agg_seg — runs LAST)
#
# Dependency chain:
#   pre-analysis jobs (fast) ──→ agg_pre (runs while pipelines still running)
#                                   │
#   pipeline jobs (slow) ───────────┼──→ agg_seg ──→ ident_all
#
# TS (timestamp) is set ONCE when this script starts. All output paths use
# the same TS so that re-running creates a new directory without overwriting.
#
# Output structure:
#   {experiment}_{TS}/
#     ├── run_0/ ... run_3/           — Pipeline output per iteration
#     ├── device_sessions/            — M2 output (JSON per house)
#     ├── house_timing.csv            — Timing log
#     └── reports/
#         ├── house_report.html           — Aggregate pre-analysis
#         ├── house_reports/              — Per-house pre-analysis
#         ├── segregation_report.html     — Aggregate segregation (M1)
#         ├── segregation_reports/        — Per-house segregation
#         ├── identification_report.html  — Aggregate identification (M2)
#         └── identification_reports/     — Per-house identification
#
# Place completed_houses.txt next to this script (same directory).
# Generate from a previous run's timing CSV:
#     awk -F',' 'NR>1 && $7=="OK" {print $1}' house_timing.csv > completed_houses.txt
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

# ---- TS set ONCE — used for ALL output paths ----
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_OUTPUT="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments/${EXPERIMENT_NAME}_${TIMESTAMP}"

# ---- Unified reports directory ----
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

# Pre-create experiment metadata (once, from login node — avoids race condition
# when multiple SBATCH jobs call save_experiment_metadata simultaneously)
cd "${PROJECT_ROOT}/experiment_pipeline"
python -c "
import sys; sys.path.insert(0, 'src')
from core.config import get_experiment, save_experiment_metadata
exp = get_experiment('${EXPERIMENT_NAME}')
save_experiment_metadata(exp, '${EXPERIMENT_OUTPUT}')
print('Experiment metadata saved to ${EXPERIMENT_OUTPUT}')
"
cd "${PROJECT_ROOT}"

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
echo "Timestamp:   $TIMESTAMP"
echo "Data dir:    $DATA_DIR"
echo "Output dir:  $EXPERIMENT_OUTPUT"
echo "Reports dir: $REPORTS_DIR"
echo "Timing log:  $TIMING_FILE"
echo ""

HOUSE_COUNT=0
NORMAL_COUNT=0
MONTHLY_COUNT=0

# Track job IDs for Phase 2 dependencies
ALL_PRE_JOBS=()     # Pre-analysis jobs (fast) — for agg_pre
ALL_FINAL_JOBS=()   # Pipeline/final jobs (slow) — for agg_seg

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

    # =================================================================
    # STEP 0: Pre-analysis report — ALWAYS a separate fast job
    # (same for both sequential and monthly houses)
    # =================================================================
    PRE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_pre_${house_id}_XXXXXX.sh")

    cat > "$PRE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=pre_${house_id}
#SBATCH --output=${LOG_DIR}/pre_${house_id}_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/pre_${house_id}_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --gres=gpu:0
##SBATCH --time=00:30:00

echo "========================================"
echo "House ${house_id} — PRE-ANALYSIS"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
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
    ALL_PRE_JOBS+=($PRE_JOB_ID)

    if [[ -n "${COMPLETED[$house_id]}" ]]; then
        # =============================================================
        # FAST HOUSE — single sequential job (depends on pre-analysis):
        #   1. Pipeline (disaggregation + identification)
        #   2. Per-house segregation + identification reports
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
#SBATCH --dependency=afterok:${PRE_JOB_ID}

echo "========================================"
echo "House ${house_id} (${N_MONTHS} months) — SEQUENTIAL"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "Node: \$SLURM_JOB_NODELIST"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

# --- Step 1: Pipeline (M1 + M2) ---
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

# --- Step 2: Per-house reports (only on success) ---
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
        echo "  House ${house_id} (${N_MONTHS} months) -> Pre ${PRE_JOB_ID}, Pipe ${JOB_ID} [SEQUENTIAL]"
        NORMAL_COUNT=$((NORMAL_COUNT + 1))

    else
        # =============================================================
        # SLOW HOUSE — month-level parallel + M2 + reports
        # Pre-analysis already submitted above.
        # =============================================================

        # Step 1: SBATCH array — one task per month (depends on pre-analysis)
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
#SBATCH --dependency=afterok:${PRE_JOB_ID}

echo "========================================"
echo "House ${house_id} — MONTHLY mode"
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

        # Step 2: M2 + reports — runs AFTER all months complete
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
echo "House ${house_id} — M2 + REPORTS (post-monthly)"
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

        POST_JOB_ID=$(sbatch "$POST_SCRIPT" | awk '{print $4}')
        rm -f "$POST_SCRIPT"

        ALL_FINAL_JOBS+=($POST_JOB_ID)
        echo "  House ${house_id} (${N_MONTHS} months) -> Pre ${PRE_JOB_ID}, Array ${ARRAY_JOB_ID} [${N_MONTHS} tasks], Post ${POST_JOB_ID} [MONTHLY]"
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
# PHASE 2: Aggregate reports
#
# Dependency chain:
#   ALL_PRE_JOBS (fast) ──→ agg_pre (runs early, while pipelines still running)
#                              │
#   ALL_FINAL_JOBS (slow) ─────┼──→ agg_seg ──→ ident_all
# =============================================================
if [ ${#ALL_FINAL_JOBS[@]} -gt 0 ]; then
    PRE_DEPS=$(IFS=:; echo "${ALL_PRE_JOBS[*]}")
    FINAL_DEPS=$(IFS=:; echo "${ALL_FINAL_JOBS[*]}")

    # --- Aggregate 1: House pre-analysis (runs EARLY — only waits for pre-analysis jobs) ---
    AGG_HOUSE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_house_XXXXXX.sh")
    cat > "$AGG_HOUSE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_house
#SBATCH --output=${LOG_DIR}/agg_house_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_house_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${PRE_DEPS}

echo "========================================"
echo "AGGREGATE: House Pre-Analysis"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${HOUSE_COUNT}
ACTUAL=\$(ls ${REPORTS_DIR}/house_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) house reports. Generating aggregate with available reports."
fi

echo "Generating house_report.html..."
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
    echo "Aggregate house pre-analysis:  ${AGG_HOUSE_JOB} (afterany pre-analysis jobs — runs EARLY)"

    # --- Aggregate 2: Segregation (waits for ALL pipelines + agg_pre) ---
    AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_XXXXXX.sh")
    cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_seg
#SBATCH --output=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/agg_seg_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00
#SBATCH --dependency=afterany:${FINAL_DEPS}:${AGG_HOUSE_JOB}

echo "========================================"
echo "AGGREGATE: Segregation (M1)"
echo "SLURM_JOBID: \$SLURM_JOBID"
echo "TS: ${TIMESTAMP}"
echo "Start: \$(date)"
echo "========================================"

module load anaconda
source activate nilm_new

EXPECTED=${HOUSE_COUNT}
ACTUAL=\$(ls ${REPORTS_DIR}/segregation_reports/house_*.html 2>/dev/null | wc -l)
echo "Per-house reports: \${ACTUAL} / \${EXPECTED}"

if [ "\$ACTUAL" -lt "\$EXPECTED" ]; then
    echo "WARNING: Missing \$((EXPECTED - ACTUAL)) segregation reports. Generating aggregate with available reports."
fi

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
    echo "Aggregate segregation (M1):    ${AGG_SEG_JOB} (afterany all pipelines + agg_pre)"

    # =============================================================
    # PHASE 3: Identification report for ALL houses (runs LAST)
    # Generates: per-house identification reports + cross-house
    # pattern matching + aggregate identification report.
    # =============================================================
    IDENT_ALL_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_all_XXXXXX.sh")
    cat > "$IDENT_ALL_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ident_all
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
    echo "Identification (ALL houses):   ${IDENT_ALL_JOB} (afterany agg_seg — runs LAST)"
fi

echo ""
echo "============================================================"
echo "Output:      ${EXPERIMENT_OUTPUT}"
echo "Reports:     ${REPORTS_DIR}/"
echo "Timing:      ${TIMING_FILE}"
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Timing:   cat ${TIMING_FILE}  (after jobs complete)"
echo "============================================================"
