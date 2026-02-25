#!/bin/bash
# =============================================================================
# Submit ONLY aggregate reports for normalization experiments.
# No pipeline, no pre-analysis — just agg_seg + ident_all.
#
# Auto-discovers the latest output directory for each experiment.
#
# Usage:
#     bash scripts/sbatch_run_norm_reports_only.sh
# =============================================================================

PROJECT_ROOT="/home/hilakese/ElectricPatterns_new"
LOG_DIR="${PROJECT_ROOT}/experiment_pipeline/logs"
EXPERIMENTS_BASE="${PROJECT_ROOT}/experiment_pipeline/OUTPUT/experiments"

EXPERIMENTS=("exp016_ma_detrend" "exp017_phase_balance" "exp019_combined_norm")

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "Submitting AGGREGATE REPORTS ONLY"
echo "============================================================"
echo ""

for EXPERIMENT_NAME in "${EXPERIMENTS[@]}"; do
    # Find the latest output directory for this experiment
    EXPERIMENT_OUTPUT=$(ls -dt "${EXPERIMENTS_BASE}/${EXPERIMENT_NAME}_"* 2>/dev/null | head -1)

    if [ -z "$EXPERIMENT_OUTPUT" ]; then
        echo "WARNING: No output directory found for ${EXPERIMENT_NAME}, skipping"
        echo ""
        continue
    fi

    REPORTS_DIR="${EXPERIMENT_OUTPUT}/reports"
    mkdir -p "$REPORTS_DIR"

    echo "------------------------------------------------------------"
    echo "Experiment: ${EXPERIMENT_NAME}"
    echo "Output:     ${EXPERIMENT_OUTPUT}"
    echo "Reports:    ${REPORTS_DIR}"
    echo "------------------------------------------------------------"

    # --- Aggregate segregation report ---
    AGG_SEG_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_agg_seg_XXXXXX.sh")
    cat > "$AGG_SEG_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=agg_seg_${EXPERIMENT_NAME}
#SBATCH --output=${LOG_DIR}/agg_seg_${EXPERIMENT_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/agg_seg_${EXPERIMENT_NAME}_%j.err
#SBATCH --partition=main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:0
##SBATCH --time=04:00:00

echo "========================================"
echo "AGGREGATE: Segregation (M1)"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Output dir: ${EXPERIMENT_OUTPUT}"
echo "SLURM_JOBID: \$SLURM_JOBID"
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
    echo "  agg_seg:   ${AGG_SEG_JOB}"

    # --- Identification report for ALL houses ---
    IDENT_ALL_SCRIPT=$(mktemp "${LOG_DIR}/sbatch_ident_all_XXXXXX.sh")
    cat > "$IDENT_ALL_SCRIPT" << EOF
#!/bin/bash
#SBATCH --job-name=ident_${EXPERIMENT_NAME}
#SBATCH --output=${LOG_DIR}/ident_all_${EXPERIMENT_NAME}_%j.out
#SBATCH --error=${LOG_DIR}/ident_all_${EXPERIMENT_NAME}_%j.err
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
echo "Output dir: ${EXPERIMENT_OUTPUT}"
echo "SLURM_JOBID: \$SLURM_JOBID"
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
    echo "  ident_all: ${IDENT_ALL_JOB} (afterany agg_seg)"
    echo ""
done

echo "============================================================"
echo "ALL REPORT JOBS SUBMITTED"
echo "============================================================"
echo "Monitor: squeue -u \$USER"
echo "============================================================"
