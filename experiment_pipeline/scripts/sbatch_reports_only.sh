#!/bin/bash
# =============================================================================
# Generate ALL reports for existing experiment directories.
#
# Runs: pre-analysis aggregate → segregation reports → identification reports
#       → cross-experiment comparison
#
# Each experiment gets its own aggregate jobs. Comparison runs once at the end.
# Uses sliding-window submission to respect QOS limits.
#
# Usage:
#     bash scripts/sbatch_reports_only.sh <experiment_dirs...>
#
# Examples:
#     bash scripts/sbatch_reports_only.sh /path/to/exp015_20260228_061111
#     bash scripts/sbatch_reports_only.sh /path/to/exp016_* /path/to/exp017_*
#     bash scripts/sbatch_reports_only.sh $(ls -d /path/to/experiments/exp01{5,6,7,8}*)
# =============================================================================

PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"
POLL_INTERVAL=20

mkdir -p "$LOG_DIR"

if [ $# -eq 0 ]; then
    echo "Usage: bash scripts/sbatch_reports_only.sh <experiment_dir> [experiment_dir2] ..."
    echo ""
    echo "Example:"
    echo "  bash scripts/sbatch_reports_only.sh /home/hilakese/ElectricPatterns_new/experiment_pipeline/OUTPUT/experiments/exp015_hole_repair_20260228_061111"
    exit 1
fi

EXPERIMENT_DIRS=("$@")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

wait_for_job() {
    local job_id="$1"
    while true; do
        local status
        status=$(squeue -j "$job_id" -h -o "%T" 2>/dev/null)
        if [ -z "$status" ]; then break; fi
        sleep "$POLL_INTERVAL"
    done
}

echo "============================================================"
echo "REPORTS ONLY — ${#EXPERIMENT_DIRS[@]} experiment(s)"
echo "============================================================"
for d in "${EXPERIMENT_DIRS[@]}"; do
    echo "  $(basename "$d")"
done
echo ""

ALL_AGG_JOBS=()

for EXPERIMENT_OUTPUT in "${EXPERIMENT_DIRS[@]}"; do
    # Validate
    if [ ! -d "$EXPERIMENT_OUTPUT" ]; then
        echo "WARNING: Directory not found: $EXPERIMENT_OUTPUT — skipping"
        continue
    fi

    EXP_NAME=$(basename "$EXPERIMENT_OUTPUT")
    REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"
    mkdir -p "$REPORTS_DIR"

    echo "------------------------------------------------------------"
    echo "[${EXP_NAME}]"
    echo "------------------------------------------------------------"

    # ---- 1. Segregation report (all houses + aggregate) ----
    SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_rpt_seg_XXXXXX.sh")
    cat > "$SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=rpt_seg_${EXP_NAME:0:20}
#SBATCH --output=${LOG_DIR}/rpt_seg_${TIMESTAMP}_${EXP_NAME:0:30}_%j.out
#SBATCH --error=${LOG_DIR}/rpt_seg_${TIMESTAMP}_${EXP_NAME:0:30}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0

echo "SEGREGATION REPORT: ${EXP_NAME} — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/disaggregation_analysis"
python scripts/run_dynamic_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish segregation \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    SEG_JOB=$(sbatch "$SEG_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
    rm -f "$SEG_SCRIPT"
    echo "  Segregation report:    ${SEG_JOB}"

    # ---- 2. Identification report (all houses + aggregate) — after segregation ----
    IDENT_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_rpt_ident_XXXXXX.sh")
    cat > "$IDENT_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=rpt_id_${EXP_NAME:0:20}
#SBATCH --output=${LOG_DIR}/rpt_ident_${TIMESTAMP}_${EXP_NAME:0:30}_%j.out
#SBATCH --error=${LOG_DIR}/rpt_ident_${TIMESTAMP}_${EXP_NAME:0:30}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${SEG_JOB}

echo "IDENTIFICATION REPORT: ${EXP_NAME} — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/identification_analysis"
python scripts/run_identification_report.py \
    --experiment ${EXPERIMENT_OUTPUT} \
    --output-dir ${REPORTS_DIR} \
    --publish identification \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    IDENT_JOB=$(sbatch "$IDENT_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
    rm -f "$IDENT_SCRIPT"
    echo "  Identification report: ${IDENT_JOB}"

    ALL_AGG_JOBS+=("$IDENT_JOB")
done

# ---- 3. Cross-experiment comparison (after all reports) ----
if [ ${#EXPERIMENT_DIRS[@]} -gt 1 ]; then
    # Build dependency on all identification jobs
    DEP_STR=$(IFS=:; echo "${ALL_AGG_JOBS[*]}")

    # Build --experiments list
    EXP_PATHS=""
    for d in "${EXPERIMENT_DIRS[@]}"; do
        EXP_PATHS="${EXP_PATHS} ${d}"
    done

    # Output dir: inside the first experiment's reports
    COMPARE_OUTPUT="${EXPERIMENT_DIRS[0]}/reports/comparison"

    COMPARE_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_compare_XXXXXX.sh")
    cat > "$COMPARE_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=compare
#SBATCH --output=${LOG_DIR}/compare_${TIMESTAMP}_%j.out
#SBATCH --error=${LOG_DIR}/compare_${TIMESTAMP}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --gres=gpu:0
#SBATCH --dependency=afterany:${DEP_STR}

echo "CROSS-EXPERIMENT COMPARISON — \$(date)"
module load anaconda
source activate nilm_new

cd "${PROJECT_ROOT}/experiment_pipeline"
python scripts/compare_experiments.py \
    --experiments ${EXP_PATHS} \
    --output-dir ${COMPARE_OUTPUT} \
    2>&1
echo "Exit: \$? — End: \$(date)"
EOF
    COMPARE_JOB=$(sbatch "$COMPARE_SCRIPT" 2>&1 | grep "Submitted batch job" | awk '{print $4}')
    rm -f "$COMPARE_SCRIPT"
    echo ""
    echo "Cross-experiment comparison: ${COMPARE_JOB}"
    echo "  Output: ${COMPARE_OUTPUT}"
fi

echo ""
echo "============================================================"
echo "Monitor: squeue -u \$USER"
echo "============================================================"
